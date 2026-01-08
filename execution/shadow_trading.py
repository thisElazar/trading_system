#!/usr/bin/env python3
"""
Shadow Trading System
=====================
Paper trade new strategies alongside live trading to build confidence.

Features:
- Run strategies in shadow mode without real capital
- Track simulated fills and P&L
- Compare shadow vs live performance
- Build confidence scores over time
- Auto-graduate strategies when proven
- Catch strategy degradation before it affects real money

Usage:
    shadow = ShadowTrader()

    # Add strategy to shadow mode
    shadow.add_strategy("new_momentum_v2", initial_capital=10000)

    # Process signals (paper trade)
    shadow.process_signal(signal)

    # Check if ready to graduate
    if shadow.is_ready_to_graduate("new_momentum_v2"):
        live_trader.add_strategy("new_momentum_v2")

    # Compare shadow vs live
    comparison = shadow.compare_to_live("momentum")
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import uuid

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASES, DIRS

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

class ShadowStatus(Enum):
    """Status of a shadow strategy."""
    ACTIVE = "active"
    PAUSED = "paused"
    GRADUATED = "graduated"
    RETIRED = "retired"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class ShadowPosition:
    """A simulated position."""
    symbol: str
    side: PositionSide
    shares: int
    entry_price: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class ShadowTrade:
    """Record of a shadow trade."""
    id: Optional[int] = None
    trade_id: str = ""
    strategy: str = ""
    symbol: str = ""
    side: str = ""  # buy, sell
    shares: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_minutes: int = 0
    simulated_slippage: float = 0.0
    status: str = "open"  # open, closed


@dataclass
class ShadowStrategy:
    """A strategy running in shadow mode."""
    name: str
    status: ShadowStatus = ShadowStatus.ACTIVE
    start_time: str = ""
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    positions: Dict[str, ShadowPosition] = field(default_factory=dict)

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_capital: float = 10000.0

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()
        self.peak_capital = self.initial_capital

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def return_pct(self) -> float:
        return ((self.current_capital / self.initial_capital) - 1) * 100

    @property
    def days_active(self) -> int:
        start = datetime.fromisoformat(self.start_time)
        return (datetime.now() - start).days


@dataclass
class ComparisonReport:
    """Comparison between shadow and live performance."""
    strategy: str
    shadow_metrics: Dict[str, float]
    live_metrics: Dict[str, float]
    correlation: float
    tracking_error: float
    shadow_better: bool
    analysis: str


# ============================================================================
# MAIN SHADOW TRADER
# ============================================================================

class ShadowTrader:
    """
    Manage shadow trading for strategy validation.

    Shadow trading allows you to paper trade strategies in parallel
    with live trading, building confidence before deploying real capital.
    """

    # Simulated slippage (basis points)
    SIMULATED_SLIPPAGE_BPS = 5.0

    def __init__(self, db_path: Path = None):
        """
        Initialize shadow trader.

        Args:
            db_path: Path to database (default from config)
        """
        self.db_path = db_path or DATABASES.get('trades', DIRS['db'] / 'trades.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()

        # In-memory strategy cache
        self.strategies: Dict[str, ShadowStrategy] = {}
        self._load_strategies()

        logger.info(f"ShadowTrader initialized with {len(self.strategies)} strategies")

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Create shadow trading tables."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Shadow strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_strategies (
                name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                current_capital REAL NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                peak_capital REAL NOT NULL,
                config TEXT DEFAULT '{}'
            )
        """)

        # Shadow trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                shares INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL DEFAULT 0,
                entry_time TEXT NOT NULL,
                exit_time TEXT DEFAULT '',
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                hold_minutes INTEGER DEFAULT 0,
                simulated_slippage REAL DEFAULT 0,
                status TEXT DEFAULT 'open'
            )
        """)

        # Shadow positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                shares INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                current_price REAL DEFAULT 0,
                max_profit REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                UNIQUE(strategy, symbol)
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_trades_strategy
            ON shadow_trades(strategy)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_trades_status
            ON shadow_trades(status)
        """)

        conn.commit()
        conn.close()

    def _load_strategies(self):
        """Load strategies from database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM shadow_strategies")
            for row in cursor.fetchall():
                strategy = ShadowStrategy(
                    name=row['name'],
                    status=ShadowStatus(row['status']),
                    start_time=row['start_time'],
                    initial_capital=row['initial_capital'],
                    current_capital=row['current_capital'],
                    total_trades=row['total_trades'],
                    winning_trades=row['winning_trades'],
                    total_pnl=row['total_pnl'],
                    max_drawdown=row['max_drawdown'],
                    peak_capital=row['peak_capital'],
                )

                # Load positions
                cursor.execute("""
                    SELECT * FROM shadow_positions WHERE strategy = ?
                """, (strategy.name,))

                for pos_row in cursor.fetchall():
                    strategy.positions[pos_row['symbol']] = ShadowPosition(
                        symbol=pos_row['symbol'],
                        side=PositionSide(pos_row['side']),
                        shares=pos_row['shares'],
                        entry_price=pos_row['entry_price'],
                        entry_time=pos_row['entry_time'],
                        current_price=pos_row['current_price'],
                        max_profit=pos_row['max_profit'],
                        max_drawdown=pos_row['max_drawdown'],
                    )

                self.strategies[strategy.name] = strategy

        finally:
            conn.close()

    def _save_strategy(self, strategy: ShadowStrategy):
        """Save strategy to database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO shadow_strategies (
                    name, status, start_time, initial_capital, current_capital,
                    total_trades, winning_trades, total_pnl, max_drawdown, peak_capital
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.name, strategy.status.value, strategy.start_time,
                strategy.initial_capital, strategy.current_capital,
                strategy.total_trades, strategy.winning_trades,
                strategy.total_pnl, strategy.max_drawdown, strategy.peak_capital
            ))

            # Save positions
            for symbol, pos in strategy.positions.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO shadow_positions (
                        strategy, symbol, side, shares, entry_price,
                        entry_time, current_price, max_profit, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy.name, pos.symbol, pos.side.value, pos.shares,
                    pos.entry_price, pos.entry_time, pos.current_price,
                    pos.max_profit, pos.max_drawdown
                ))

            conn.commit()

        finally:
            conn.close()

    # ========================================================================
    # STRATEGY MANAGEMENT
    # ========================================================================

    def add_strategy(
        self,
        name: str,
        initial_capital: float = 10000.0,
    ) -> ShadowStrategy:
        """
        Add a new strategy to shadow trading.

        Args:
            name: Strategy name
            initial_capital: Starting capital for shadow trading

        Returns:
            The created ShadowStrategy
        """
        if name in self.strategies:
            logger.warning(f"Strategy {name} already exists in shadow trading")
            return self.strategies[name]

        strategy = ShadowStrategy(
            name=name,
            initial_capital=initial_capital,
            current_capital=initial_capital,
        )

        self.strategies[name] = strategy
        self._save_strategy(strategy)

        logger.info(f"Added strategy {name} to shadow trading with ${initial_capital:,.0f}")
        return strategy

    def remove_strategy(self, name: str):
        """Remove a strategy from shadow trading."""
        if name in self.strategies:
            del self.strategies[name]

            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM shadow_strategies WHERE name = ?", (name,))
                cursor.execute("DELETE FROM shadow_positions WHERE strategy = ?", (name,))
                conn.commit()
            finally:
                conn.close()

            logger.info(f"Removed strategy {name} from shadow trading")

    def pause_strategy(self, name: str):
        """Pause shadow trading for a strategy."""
        if name in self.strategies:
            self.strategies[name].status = ShadowStatus.PAUSED
            self._save_strategy(self.strategies[name])
            logger.info(f"Paused shadow trading for {name}")

    def resume_strategy(self, name: str):
        """Resume shadow trading for a strategy."""
        if name in self.strategies:
            self.strategies[name].status = ShadowStatus.ACTIVE
            self._save_strategy(self.strategies[name])
            logger.info(f"Resumed shadow trading for {name}")

    def get_active_strategies(self) -> List[str]:
        """
        Get list of active shadow trading strategies.

        Returns:
            List of strategy names that are actively being shadow traded
        """
        return [
            name for name, strategy in self.strategies.items()
            if strategy.status == ShadowStatus.ACTIVE
        ]

    def get_strategy(self, name: str) -> Optional[ShadowStrategy]:
        """
        Get a specific shadow strategy by name.

        Args:
            name: Strategy name

        Returns:
            ShadowStrategy if found, None otherwise
        """
        return self.strategies.get(name)

    def register_variation(
        self,
        base_strategy: str,
        variation_name: str,
        variation_params: Dict[str, Any] = None,
        initial_capital: float = 10000.0
    ) -> ShadowStrategy:
        """
        Register a variation of an existing strategy for A/B testing.

        Use this to test parameter changes, new indicators, or other
        modifications without affecting the live strategy.

        Args:
            base_strategy: Name of the strategy being varied
            variation_name: Unique name for this variation (e.g., "momentum_v2_fast_exit")
            variation_params: Dict of parameter differences for tracking
            initial_capital: Starting capital for shadow trading

        Returns:
            The created ShadowStrategy for the variation

        Example:
            shadow.register_variation(
                base_strategy="momentum",
                variation_name="momentum_tight_stops",
                variation_params={"stop_pct": 0.02, "take_profit_pct": 0.04}
            )
        """
        if variation_name in self.strategies:
            logger.warning(f"Variation {variation_name} already exists")
            return self.strategies[variation_name]

        # Create the variation
        strategy = self.add_strategy(
            name=variation_name,
            initial_capital=initial_capital,
            min_trades=20,  # Lower bar for variations (just need comparison data)
            min_win_rate=0.50,
            min_profit_factor=1.2,
            min_days=7
        )

        # Store variation metadata
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            config = {
                'is_variation': True,
                'base_strategy': base_strategy,
                'variation_params': variation_params or {},
                'created_at': datetime.now().isoformat()
            }
            cursor.execute("""
                UPDATE shadow_strategies SET config = ? WHERE name = ?
            """, (json.dumps(config), variation_name))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Registered variation '{variation_name}' of '{base_strategy}'")
        return strategy

    def get_variations(self, base_strategy: str) -> List[str]:
        """
        Get all variations of a base strategy.

        Args:
            base_strategy: Name of the base strategy

        Returns:
            List of variation names
        """
        variations = []
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name, config FROM shadow_strategies")
            for row in cursor.fetchall():
                config = json.loads(row['config']) if row['config'] else {}
                if config.get('is_variation') and config.get('base_strategy') == base_strategy:
                    variations.append(row['name'])
        finally:
            conn.close()
        return variations

    def compare_variations(self, base_strategy: str, days: int = 30) -> Dict[str, Dict]:
        """
        Compare performance of a strategy and all its variations.

        Args:
            base_strategy: Name of the base strategy
            days: Days of data to compare

        Returns:
            Dict mapping strategy/variation name to metrics
        """
        results = {}

        # Get base strategy metrics
        if base_strategy in self.strategies:
            results[base_strategy] = self.get_strategy_metrics(base_strategy, days)
            results[base_strategy]['is_base'] = True

        # Get variation metrics
        for variation in self.get_variations(base_strategy):
            if variation in self.strategies:
                results[variation] = self.get_strategy_metrics(variation, days)
                results[variation]['is_base'] = False

        # Rank by profit factor
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].get('profit_factor', 0),
            reverse=True
        )

        # Add rank
        for i, (name, metrics) in enumerate(ranked):
            results[name]['rank'] = i + 1

        return results

    # ========================================================================
    # TRADING OPERATIONS
    # ========================================================================

    def process_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        price: float,
        shares: int = None,
        position_pct: float = 0.1,
    ) -> Optional[str]:
        """
        Process a trading signal in shadow mode.

        Args:
            strategy: Strategy name
            symbol: Stock symbol
            signal_type: 'buy', 'sell', or 'exit'
            price: Current price
            shares: Number of shares (or calculate from position_pct)
            position_pct: Position size as % of capital

        Returns:
            Trade ID if a trade was executed
        """
        if strategy not in self.strategies:
            logger.warning(f"Strategy {strategy} not in shadow trading")
            return None

        strat = self.strategies[strategy]
        if strat.status != ShadowStatus.ACTIVE:
            logger.debug(f"Strategy {strategy} is not active")
            return None

        # Calculate shares if not provided
        if shares is None:
            position_value = strat.current_capital * position_pct
            shares = int(position_value / price)

        if shares <= 0:
            logger.debug(f"Insufficient capital for trade")
            return None

        # Apply simulated slippage
        slippage = price * (self.SIMULATED_SLIPPAGE_BPS / 10000)
        if signal_type == 'buy':
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        trade_id = f"SHADOW_{strategy}_{symbol}_{uuid.uuid4().hex[:8]}"

        if signal_type == 'buy':
            return self._open_position(strat, symbol, PositionSide.LONG, shares, fill_price, trade_id, slippage)
        elif signal_type == 'sell' and symbol in strat.positions:
            return self._close_position(strat, symbol, fill_price, trade_id, slippage)
        elif signal_type == 'exit' and symbol in strat.positions:
            return self._close_position(strat, symbol, fill_price, trade_id, slippage)

        return None

    def _open_position(
        self,
        strategy: ShadowStrategy,
        symbol: str,
        side: PositionSide,
        shares: int,
        price: float,
        trade_id: str,
        slippage: float,
    ) -> str:
        """Open a new shadow position."""
        # Close existing position if any
        if symbol in strategy.positions:
            self._close_position(strategy, symbol, price, f"{trade_id}_close", slippage)

        # Create position
        position = ShadowPosition(
            symbol=symbol,
            side=side,
            shares=shares,
            entry_price=price,
            entry_time=datetime.now().isoformat(),
            current_price=price,
        )

        strategy.positions[symbol] = position

        # Record trade
        trade = ShadowTrade(
            trade_id=trade_id,
            strategy=strategy.name,
            symbol=symbol,
            side='buy' if side == PositionSide.LONG else 'short',
            shares=shares,
            entry_price=price,
            entry_time=datetime.now().isoformat(),
            simulated_slippage=slippage,
            status='open',
        )

        self._save_trade(trade)
        self._save_strategy(strategy)

        logger.info(f"Shadow opened: {symbol} {shares} @ ${price:.2f} ({strategy.name})")
        return trade_id

    def _close_position(
        self,
        strategy: ShadowStrategy,
        symbol: str,
        price: float,
        trade_id: str,
        slippage: float,
    ) -> str:
        """Close an existing shadow position."""
        if symbol not in strategy.positions:
            return None

        position = strategy.positions[symbol]

        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * position.shares
        else:
            pnl = (position.entry_price - price) * position.shares

        pnl_pct = ((price / position.entry_price) - 1) * 100
        if position.side == PositionSide.SHORT:
            pnl_pct = -pnl_pct

        # Calculate hold time
        entry_time = datetime.fromisoformat(position.entry_time)
        hold_minutes = int((datetime.now() - entry_time).total_seconds() / 60)

        # Update strategy stats
        strategy.total_trades += 1
        strategy.total_pnl += pnl
        strategy.current_capital += pnl

        if pnl > 0:
            strategy.winning_trades += 1

        # Track drawdown
        if strategy.current_capital > strategy.peak_capital:
            strategy.peak_capital = strategy.current_capital
        else:
            current_dd = (strategy.peak_capital - strategy.current_capital) / strategy.peak_capital
            if current_dd > strategy.max_drawdown:
                strategy.max_drawdown = current_dd

        # Record trade
        trade = ShadowTrade(
            trade_id=trade_id,
            strategy=strategy.name,
            symbol=symbol,
            side='sell',
            shares=position.shares,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_minutes=hold_minutes,
            simulated_slippage=slippage,
            status='closed',
        )

        self._save_trade(trade)

        # Remove position
        del strategy.positions[symbol]
        self._save_strategy(strategy)

        logger.info(
            f"Shadow closed: {symbol} @ ${price:.2f} "
            f"P&L: ${pnl:.2f} ({pnl_pct:.1f}%) ({strategy.name})"
        )
        return trade_id

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all shadow positions.

        Args:
            prices: Dict mapping symbol to current price
        """
        for strategy in self.strategies.values():
            for symbol, position in strategy.positions.items():
                if symbol in prices:
                    price = prices[symbol]
                    position.current_price = price

                    # Calculate unrealized P&L
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = (price - position.entry_price) * position.shares
                    else:
                        position.unrealized_pnl = (position.entry_price - price) * position.shares

                    # Track max profit/drawdown
                    if position.unrealized_pnl > position.max_profit:
                        position.max_profit = position.unrealized_pnl
                    if position.unrealized_pnl < -position.max_drawdown:
                        position.max_drawdown = abs(position.unrealized_pnl)

    def _save_trade(self, trade: ShadowTrade):
        """Save trade to database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO shadow_trades (
                    trade_id, strategy, symbol, side, shares,
                    entry_price, exit_price, entry_time, exit_time,
                    pnl, pnl_pct, hold_minutes, simulated_slippage, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.strategy, trade.symbol, trade.side,
                trade.shares, trade.entry_price, trade.exit_price,
                trade.entry_time, trade.exit_time, trade.pnl, trade.pnl_pct,
                trade.hold_minutes, trade.simulated_slippage, trade.status
            ))
            conn.commit()

        finally:
            conn.close()

    # ========================================================================
    # METRICS HELPERS
    # ========================================================================

    def _calculate_profit_factor(self, name: str) -> float:
        """Calculate profit factor for a strategy."""
        conn = self._get_conn()

        try:
            df = pd.read_sql_query("""
                SELECT pnl FROM shadow_trades
                WHERE strategy = ? AND status = 'closed'
            """, conn, params=[name])

            if df.empty:
                return 0.0

            gains = df[df['pnl'] > 0]['pnl'].sum()
            losses = abs(df[df['pnl'] < 0]['pnl'].sum())

            if losses == 0:
                return float('inf') if gains > 0 else 0.0

            return gains / losses

        finally:
            conn.close()

    # ========================================================================
    # COMPARISON AND ANALYSIS
    # ========================================================================

    def compare_to_live(
        self,
        shadow_strategy: str,
        live_strategy: str = None,
        days: int = 30,
    ) -> ComparisonReport:
        """
        Compare shadow strategy performance to live trading.

        Args:
            shadow_strategy: Name of shadow strategy
            live_strategy: Name of live strategy (default: same name)
            days: Days to compare

        Returns:
            ComparisonReport with detailed comparison
        """
        if live_strategy is None:
            live_strategy = shadow_strategy

        shadow_metrics = self.get_strategy_metrics(shadow_strategy, days)

        # Get live metrics from execution tracker
        live_metrics = self._get_live_metrics(live_strategy, days)

        # Calculate correlation if we have daily returns
        shadow_returns = self._get_daily_returns(shadow_strategy, days)
        live_returns = self._get_live_daily_returns(live_strategy, days)

        if len(shadow_returns) > 5 and len(live_returns) > 5:
            # Align dates
            common_dates = set(shadow_returns.index) & set(live_returns.index)
            if len(common_dates) > 5:
                aligned_shadow = shadow_returns.loc[list(common_dates)]
                aligned_live = live_returns.loc[list(common_dates)]
                correlation = aligned_shadow.corr(aligned_live)
                tracking_error = (aligned_shadow - aligned_live).std()
            else:
                correlation = 0.0
                tracking_error = 0.0
        else:
            correlation = 0.0
            tracking_error = 0.0

        # Determine if shadow is better
        shadow_better = (
            shadow_metrics.get('return_pct', 0) > live_metrics.get('return_pct', 0) and
            shadow_metrics.get('win_rate', 0) >= live_metrics.get('win_rate', 0)
        )

        # Analysis
        if shadow_better:
            analysis = "Shadow strategy outperforming live. Consider parameter update."
        elif shadow_metrics.get('return_pct', 0) < live_metrics.get('return_pct', 0) * 0.5:
            analysis = "Shadow significantly underperforming. Review strategy logic."
        else:
            analysis = "Shadow and live performance comparable."

        return ComparisonReport(
            strategy=shadow_strategy,
            shadow_metrics=shadow_metrics,
            live_metrics=live_metrics,
            correlation=correlation,
            tracking_error=tracking_error,
            shadow_better=shadow_better,
            analysis=analysis
        )

    def get_strategy_metrics(self, name: str, days: int = 30) -> Dict[str, float]:
        """Get performance metrics for a shadow strategy."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT * FROM shadow_trades
                WHERE strategy = ? AND status = 'closed' AND exit_time >= ?
            """, conn, params=[name, cutoff])

            if df.empty:
                return {'error': 'No trades in period'}

            return {
                'total_trades': len(df),
                'win_rate': (df['pnl'] > 0).mean(),
                'avg_pnl': df['pnl'].mean(),
                'total_pnl': df['pnl'].sum(),
                'return_pct': df['pnl_pct'].mean(),
                'avg_hold_minutes': df['hold_minutes'].mean(),
                'best_trade': df['pnl'].max(),
                'worst_trade': df['pnl'].min(),
                'profit_factor': self._calculate_profit_factor(name),
            }

        finally:
            conn.close()

    def calculate_sharpe_ratio(self, name: str, days: int = 30) -> float:
        """
        Calculate annualized Sharpe ratio from daily P&L.

        Args:
            name: Strategy name
            days: Number of days to analyze

        Returns:
            Annualized Sharpe ratio (0.0 if insufficient data)
        """
        import numpy as np

        daily_returns = self._get_daily_returns(name, days)

        if len(daily_returns) < 5:
            # Need at least 5 days for meaningful Sharpe
            return 0.0

        mean_return = daily_returns.mean()
        std_return = daily_returns.std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # Annualized Sharpe: (mean / std) * sqrt(252)
        sharpe = (mean_return / std_return) * np.sqrt(252)

        return float(sharpe)

    def get_strategy_max_drawdown(self, name: str, days: int = 30) -> float:
        """
        Calculate maximum drawdown for a strategy.

        Args:
            name: Strategy name
            days: Number of days to analyze

        Returns:
            Max drawdown as negative percentage (e.g., -15.5 for 15.5% drawdown)
        """
        import numpy as np

        daily_returns = self._get_daily_returns(name, days)

        if len(daily_returns) < 2:
            return 0.0

        # Calculate cumulative returns
        cumulative = daily_returns.cumsum()

        # Calculate running maximum
        running_max = cumulative.cummax()

        # Calculate drawdown at each point
        drawdown = cumulative - running_max

        # Return minimum (most negative) drawdown
        max_dd = drawdown.min()

        return float(max_dd) if not np.isnan(max_dd) else 0.0

    def _get_daily_returns(self, name: str, days: int) -> pd.Series:
        """Get daily returns for a shadow strategy."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT DATE(exit_time) as date, SUM(pnl) as daily_pnl
                FROM shadow_trades
                WHERE strategy = ? AND status = 'closed' AND exit_time >= ?
                GROUP BY DATE(exit_time)
            """, conn, params=[name, cutoff])

            if df.empty:
                return pd.Series(dtype=float)

            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')['daily_pnl']

        finally:
            conn.close()

    def _get_live_metrics(self, name: str, days: int) -> Dict[str, float]:
        """Get live trading metrics (placeholder - integrate with live tracker)."""
        # This would integrate with your live execution tracker
        # For now, return placeholder
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'return_pct': 0.0,
            'note': 'Live metrics integration pending'
        }

    def _get_live_daily_returns(self, name: str, days: int) -> pd.Series:
        """Get live daily returns (placeholder)."""
        return pd.Series(dtype=float)

    # ========================================================================
    # REPORTING
    # ========================================================================

    def print_status(self):
        """Print status of all shadow strategies."""
        print("\n" + "=" * 70)
        print("SHADOW TRADING STATUS")
        print("=" * 70)

        if not self.strategies:
            print("\nNo strategies in shadow trading.")
            print("=" * 70)
            return

        for name, strategy in sorted(self.strategies.items()):
            print(f"\n{name.upper()} [{strategy.status.value}]")
            print("-" * 50)
            print(f"  Capital: ${strategy.current_capital:,.2f} (started: ${strategy.initial_capital:,.2f})")
            print(f"  Return: {strategy.return_pct:.1f}%")
            print(f"  Trades: {strategy.total_trades} ({strategy.win_rate:.1%} win rate)")
            print(f"  Max Drawdown: {strategy.max_drawdown:.1%}")
            print(f"  Days Active: {strategy.days_active}")

            if strategy.positions:
                print(f"\n  Open Positions:")
                for symbol, pos in strategy.positions.items():
                    print(f"    {symbol}: {pos.shares} @ ${pos.entry_price:.2f} "
                          f"(unrealized: ${pos.unrealized_pnl:.2f})")

        print("\n" + "=" * 70)

    def get_all_positions(self) -> Dict[str, List[Dict]]:
        """Get all open positions across all strategies."""
        result = {}
        for name, strategy in self.strategies.items():
            if strategy.positions:
                result[name] = [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side.value,
                        'shares': pos.shares,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                    }
                    for pos in strategy.positions.values()
                ]
        return result


# ============================================================================
# FACTORY
# ============================================================================

_shadow_trader: Optional[ShadowTrader] = None

def get_shadow_trader() -> ShadowTrader:
    """Get or create global shadow trader."""
    global _shadow_trader
    if _shadow_trader is None:
        _shadow_trader = ShadowTrader()
    return _shadow_trader


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SHADOW TRADING SYSTEM DEMO")
    print("=" * 60)

    shadow = ShadowTrader()

    # Add a test strategy
    shadow.add_strategy(
        "test_momentum_v2",
        initial_capital=10000,
        min_trades=10,  # Lower for demo
        min_days=1,  # Lower for demo
    )

    # Simulate some trades
    print("\nSimulating trades...")

    trades = [
        ("AAPL", "buy", 175.00),
        ("AAPL", "sell", 178.50),
        ("MSFT", "buy", 350.00),
        ("MSFT", "sell", 345.00),
        ("GOOGL", "buy", 140.00),
        ("GOOGL", "sell", 145.00),
        ("NVDA", "buy", 500.00),
        ("NVDA", "sell", 510.00),
        ("AMZN", "buy", 180.00),
        ("AMZN", "sell", 177.00),
    ]

    for symbol, action, price in trades:
        shadow.process_signal(
            strategy="test_momentum_v2",
            symbol=symbol,
            signal_type=action,
            price=price,
            position_pct=0.1
        )

    # Print status
    shadow.print_status()
