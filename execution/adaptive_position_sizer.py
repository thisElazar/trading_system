"""
Adaptive Position Sizer
=======================
Dynamic position sizing that adjusts based on:
- Recent strategy performance (Sharpe ratio)
- VIX regime (market volatility)
- Current drawdown level
- Win rate momentum (improving vs declining)
- Kelly criterion integration

This is a drop-in enhancement to the existing position_sizer.py that provides
smarter, context-aware position sizing instead of fixed 2% risk per trade.

Usage:
    from execution.adaptive_position_sizer import AdaptivePositionSizer

    sizer = AdaptivePositionSizer()
    shares, dollar_amount = sizer.calculate_position_size(
        strategy='mean_reversion',
        symbol='AAPL',
        price=175.00,
        stop_loss=170.00,
        context={'atr': 3.50}
    )
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOTAL_CAPITAL, RISK_PER_TRADE, MAX_POSITION_SIZE,
    MAX_POSITIONS, CASH_BUFFER_PCT, MAX_POSITION_PCT,
    MAX_DRAWDOWN_PCT, DATABASES, VIX_REGIMES,
    get_vix_regime
)

# Import ADAPTIVE_SIZING from config if available, otherwise use local defaults
try:
    from config import ADAPTIVE_SIZING as CONFIG_ADAPTIVE_SIZING
    _use_config_adaptive_sizing = True
except ImportError:
    _use_config_adaptive_sizing = False
    CONFIG_ADAPTIVE_SIZING = None

from execution.position_sizer import PositionSizer, PositionSize

logger = logging.getLogger(__name__)


# =============================================================================
# ADAPTIVE SIZING CONFIGURATION (defaults, can be overridden by config.py)
# =============================================================================

_DEFAULT_ADAPTIVE_SIZING = {
    # Strategy performance scaling
    "sharpe_scale_min": 0.5,        # Minimum scalar when Sharpe < 0
    "sharpe_scale_max": 1.5,        # Maximum scalar when Sharpe > 1
    "sharpe_neutral": 0.5,          # Sharpe ratio for 1.0x scaling
    "sharpe_lookback_days": 60,     # Days to calculate rolling Sharpe

    # VIX regime scaling
    "vix_low_scalar": 1.2,          # Scale up in low VIX (< 15)
    "vix_normal_scalar": 1.0,       # Normal sizing (15-25)
    "vix_high_scalar": 0.7,         # Scale down in high VIX (25-35)
    "vix_extreme_scalar": 0.4,      # Aggressive scale down (> 35)

    # Drawdown scaling (progressive)
    "drawdown_thresholds": [0.05, 0.10, 0.15, 0.20],  # 5%, 10%, 15%, 20%
    "drawdown_scalars": [1.0, 0.8, 0.6, 0.4, 0.25],   # Corresponding scalars

    # Win rate momentum
    "win_rate_lookback_short": 10,  # Recent trades
    "win_rate_lookback_long": 30,   # Baseline trades
    "win_rate_improvement_bonus": 0.2,  # +20% if improving
    "win_rate_decline_penalty": 0.15,   # -15% if declining

    # Kelly criterion
    "kelly_fraction": 0.25,         # Use 25% of full Kelly (conservative)
    "kelly_min_trades": 20,         # Minimum trades before Kelly kicks in

    # Combined scalar limits
    "combined_scalar_min": 0.25,    # Never go below 25% of base size
    "combined_scalar_max": 1.5,     # Never exceed 150% of base size

    # Logging
    "log_decisions": True,          # Log sizing decisions to DB
}

# Use config values if available, otherwise use defaults
ADAPTIVE_SIZING = CONFIG_ADAPTIVE_SIZING if _use_config_adaptive_sizing else _DEFAULT_ADAPTIVE_SIZING


@dataclass
class SizingContext:
    """Context information for position sizing decisions."""
    vix_level: float = 20.0
    vix_regime: str = "normal"
    current_drawdown: float = 0.0
    strategy_sharpe: float = 0.0
    strategy_win_rate: float = 0.5
    recent_win_rate: float = 0.5
    total_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    kelly_fraction: float = 0.0

    # Individual scalars
    strategy_scalar: float = 1.0
    regime_scalar: float = 1.0
    drawdown_scalar: float = 1.0
    momentum_scalar: float = 1.0
    kelly_scalar: float = 1.0

    # Final combined scalar
    combined_scalar: float = 1.0


@dataclass
class AdaptivePositionResult:
    """Result of adaptive position sizing calculation."""
    symbol: str
    shares: int
    dollar_value: float

    # Base sizing info
    base_risk_amount: float
    base_shares: int
    base_dollar_value: float

    # Adaptive adjustments
    context: SizingContext = field(default_factory=SizingContext)
    final_scalar: float = 1.0

    # Risk metrics
    risk_amount: float = 0.0
    stop_distance: float = 0.0
    position_pct: float = 0.0

    # Constraints
    was_capped: bool = False
    cap_reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'dollar_value': self.dollar_value,
            'base_shares': self.base_shares,
            'base_dollar_value': self.base_dollar_value,
            'final_scalar': self.final_scalar,
            'strategy_scalar': self.context.strategy_scalar,
            'regime_scalar': self.context.regime_scalar,
            'drawdown_scalar': self.context.drawdown_scalar,
            'momentum_scalar': self.context.momentum_scalar,
            'kelly_scalar': self.context.kelly_scalar,
            'vix_level': self.context.vix_level,
            'current_drawdown': self.context.current_drawdown,
            'strategy_sharpe': self.context.strategy_sharpe,
            'was_capped': self.was_capped,
            'cap_reason': self.cap_reason,
        }


class AdaptivePositionSizer:
    """
    Adaptive position sizing that adjusts based on market conditions
    and strategy performance.

    Replaces fixed 2% risk per trade with dynamic sizing that:
    - Scales down when strategy is underperforming (Sharpe < 0)
    - Scales down in high VIX environments
    - Progressively reduces size as drawdown increases
    - Rewards improving win rates
    - Integrates Kelly criterion for optimal sizing
    """

    def __init__(
        self,
        total_capital: float = TOTAL_CAPITAL,
        risk_per_trade: float = RISK_PER_TRADE,
        max_position_size: float = MAX_POSITION_SIZE,
        max_positions: int = MAX_POSITIONS,
        cash_buffer_pct: float = CASH_BUFFER_PCT,
        max_position_pct: float = MAX_POSITION_PCT,
        config: Dict = None
    ):
        """
        Initialize adaptive position sizer.

        Args:
            total_capital: Total portfolio capital
            risk_per_trade: Base risk per trade (default 2%)
            max_position_size: Maximum dollar value per position
            max_positions: Maximum concurrent positions
            cash_buffer_pct: Percentage to keep in cash
            max_position_pct: Maximum position as % of portfolio
            config: Override adaptive sizing configuration
        """
        self.total_capital = total_capital
        self.base_risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.cash_buffer_pct = cash_buffer_pct
        self.max_position_pct = max_position_pct

        # Merge config with defaults
        self.config = {**ADAPTIVE_SIZING, **(config or {})}

        # Base position sizer for fallback
        self._base_sizer = PositionSizer(
            total_capital=total_capital,
            risk_per_trade=risk_per_trade,
            max_position_size=max_position_size,
            max_positions=max_positions,
            cash_buffer_pct=cash_buffer_pct,
            max_position_pct=max_position_pct
        )

        # Cache for strategy stats
        self._strategy_stats_cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

        # Current state
        self._current_vix: Optional[float] = None
        self._current_drawdown: float = 0.0
        self._positions: Dict[str, float] = {}
        self._cash: float = total_capital

        logger.info(
            f"AdaptivePositionSizer initialized: capital=${total_capital:,.0f}, "
            f"base_risk={risk_per_trade:.1%}"
        )

    # =========================================================================
    # CORE POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        strategy: str,
        symbol: str,
        price: float,
        stop_loss: float,
        context: Dict = None
    ) -> Tuple[int, float]:
        """
        Calculate adaptive position size.

        Args:
            strategy: Strategy name (e.g., 'mean_reversion')
            symbol: Stock symbol
            price: Current price
            stop_loss: Stop loss price
            context: Additional context (atr, volatility, etc.)

        Returns:
            Tuple of (shares, dollar_amount)
        """
        context = context or {}

        # Build sizing context
        sizing_ctx = self._build_sizing_context(strategy, context)

        # Calculate base position size
        stop_distance = abs(price - stop_loss)
        if stop_distance <= 0:
            stop_distance = price * 0.05  # Default 5% stop

        base_risk_amount = self.total_capital * self.base_risk_per_trade
        base_shares = int(base_risk_amount / stop_distance) if stop_distance > 0 else 0
        base_dollar_value = base_shares * price

        # Apply adaptive scaling
        combined_scalar = sizing_ctx.combined_scalar
        adjusted_shares = int(base_shares * combined_scalar)
        adjusted_dollar_value = adjusted_shares * price

        # Apply constraints
        was_capped = False
        cap_reason = ""

        # Cap at max position size
        if adjusted_dollar_value > self.max_position_size:
            adjusted_dollar_value = self.max_position_size
            adjusted_shares = int(adjusted_dollar_value / price)
            was_capped = True
            cap_reason = "max_position_size"

        # Cap at max position percentage
        max_pct_value = self.total_capital * self.max_position_pct
        if adjusted_dollar_value > max_pct_value:
            adjusted_dollar_value = max_pct_value
            adjusted_shares = int(adjusted_dollar_value / price)
            was_capped = True
            cap_reason = f"max_position_pct ({self.max_position_pct:.0%})"

        # Cap at available capital
        available = self._get_available_capital()
        if adjusted_dollar_value > available:
            adjusted_dollar_value = available * 0.95
            adjusted_shares = int(adjusted_dollar_value / price)
            was_capped = True
            cap_reason = "available_capital"

        # Ensure positive values
        adjusted_shares = max(0, adjusted_shares)
        adjusted_dollar_value = adjusted_shares * price

        # Create result
        result = AdaptivePositionResult(
            symbol=symbol,
            shares=adjusted_shares,
            dollar_value=adjusted_dollar_value,
            base_risk_amount=base_risk_amount,
            base_shares=base_shares,
            base_dollar_value=base_dollar_value,
            context=sizing_ctx,
            final_scalar=combined_scalar,
            risk_amount=adjusted_shares * stop_distance,
            stop_distance=stop_distance,
            position_pct=(adjusted_dollar_value / self.total_capital * 100) if self.total_capital > 0 else 0,
            was_capped=was_capped,
            cap_reason=cap_reason
        )

        # Log decision
        if self.config["log_decisions"]:
            self._log_sizing_decision(strategy, result)

        logger.debug(
            f"Adaptive size for {symbol}: {adjusted_shares} shares "
            f"(${adjusted_dollar_value:,.0f}) | scalar={combined_scalar:.2f} | "
            f"strategy={sizing_ctx.strategy_scalar:.2f}, "
            f"regime={sizing_ctx.regime_scalar:.2f}, "
            f"drawdown={sizing_ctx.drawdown_scalar:.2f}"
        )

        return adjusted_shares, adjusted_dollar_value

    # =========================================================================
    # SCALAR CALCULATIONS
    # =========================================================================

    def get_strategy_scalar(self, strategy_name: str) -> float:
        """
        Calculate position size scalar based on strategy performance.

        - Sharpe < 0: Scale down to sharpe_scale_min (0.5)
        - Sharpe = sharpe_neutral (0.5): Scale at 1.0
        - Sharpe > 1: Scale up to sharpe_scale_max (1.5)

        Args:
            strategy_name: Name of the strategy

        Returns:
            Float scalar between sharpe_scale_min and sharpe_scale_max
        """
        stats = self._get_strategy_stats(strategy_name)

        if not stats or stats.get('total_trades', 0) < 10:
            return 1.0  # Not enough data, use base sizing

        sharpe = stats.get('sharpe_ratio', 0.0) or 0.0

        min_scalar = self.config["sharpe_scale_min"]
        max_scalar = self.config["sharpe_scale_max"]
        neutral_sharpe = self.config["sharpe_neutral"]

        if sharpe <= 0:
            # Linear interpolation from min_scalar to 1.0 for Sharpe -1 to 0
            scalar = min_scalar + (1.0 - min_scalar) * max(0, sharpe + 1)
            return max(min_scalar, scalar)
        elif sharpe <= neutral_sharpe:
            # Between 0 and neutral: 1.0
            return 1.0
        else:
            # Above neutral: scale up linearly to max
            # At Sharpe 1.0, we want to reach max_scalar
            excess = sharpe - neutral_sharpe
            scalar = 1.0 + (max_scalar - 1.0) * min(excess / (1.0 - neutral_sharpe), 1.0)
            return min(max_scalar, scalar)

    def get_regime_scalar(self, vix_level: float = None) -> float:
        """
        Calculate position size scalar based on VIX regime.

        - Low VIX (< 15): Scale up to 1.2
        - Normal (15-25): 1.0
        - High (25-35): Scale down to 0.7
        - Extreme (> 35): Scale down to 0.4

        Args:
            vix_level: Current VIX level (uses cached if None)

        Returns:
            Float scalar based on VIX regime
        """
        if vix_level is None:
            vix_level = self._get_current_vix()

        regime = get_vix_regime(vix_level)

        scalars = {
            "low": self.config["vix_low_scalar"],
            "normal": self.config["vix_normal_scalar"],
            "high": self.config["vix_high_scalar"],
            "extreme": self.config["vix_extreme_scalar"]
        }

        return scalars.get(regime, 1.0)

    def get_drawdown_scalar(self, current_drawdown: float = None) -> float:
        """
        Calculate position size scalar based on current drawdown.

        Progressive scaling:
        - 0-5%: 1.0
        - 5-10%: 0.8
        - 10-15%: 0.6
        - 15-20%: 0.4
        - >20%: 0.25

        Args:
            current_drawdown: Current drawdown as decimal (e.g., 0.10 for 10%)

        Returns:
            Float scalar based on drawdown level
        """
        if current_drawdown is None:
            current_drawdown = self._current_drawdown

        thresholds = self.config["drawdown_thresholds"]
        scalars = self.config["drawdown_scalars"]

        # Find appropriate bucket
        for i, threshold in enumerate(thresholds):
            if current_drawdown < threshold:
                return scalars[i]

        # Beyond all thresholds
        return scalars[-1]

    def get_win_rate_momentum_scalar(self, strategy_name: str) -> float:
        """
        Calculate scalar based on win rate momentum.

        Compares recent win rate (last 10 trades) to baseline (last 30 trades).
        - Improving: +20% bonus
        - Stable: 1.0
        - Declining: -15% penalty

        Args:
            strategy_name: Name of the strategy

        Returns:
            Float scalar (0.85 to 1.2)
        """
        stats = self._get_strategy_stats(strategy_name)

        if not stats:
            return 1.0

        recent_wr = stats.get('recent_win_rate', 0.5)
        baseline_wr = stats.get('win_rate', 0.5)

        if baseline_wr == 0:
            return 1.0

        improvement = recent_wr - baseline_wr

        if improvement > 0.05:  # > 5% improvement
            return 1.0 + self.config["win_rate_improvement_bonus"]
        elif improvement < -0.05:  # > 5% decline
            return 1.0 - self.config["win_rate_decline_penalty"]
        else:
            return 1.0

    def get_kelly_scalar(self, strategy_name: str) -> float:
        """
        Calculate Kelly criterion position scalar.

        Kelly formula: f* = (p * b - q) / b
        Where: p = win rate, q = 1-p, b = avg_win/avg_loss

        We use fractional Kelly (25%) to reduce variance.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Float scalar (can be > 1.0 for high-edge strategies)
        """
        stats = self._get_strategy_stats(strategy_name)

        if not stats:
            return 1.0

        total_trades = stats.get('total_trades', 0)
        if total_trades < self.config["kelly_min_trades"]:
            return 1.0  # Not enough data

        win_rate = stats.get('win_rate', 0.5)
        avg_win = abs(stats.get('avg_win', 0.01))
        avg_loss = abs(stats.get('avg_loss', 0.01))

        if avg_loss == 0:
            avg_loss = 0.01  # Prevent division by zero

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  # Win/loss ratio

        if b <= 0:
            return 1.0

        # Full Kelly
        kelly = (p * b - q) / b

        # Apply fraction and cap
        kelly_scaled = kelly * self.config["kelly_fraction"]

        # Kelly gives allocation fraction; convert to scalar relative to base
        # If Kelly suggests 10% and base is 2%, scalar would be 5.0
        # But we cap this at a reasonable level
        if kelly_scaled <= 0:
            return 0.5  # Edge is negative, reduce sizing

        # Normalize around the base risk (2%)
        kelly_scalar = kelly_scaled / self.base_risk_per_trade

        # Cap between 0.5 and 2.0
        return max(0.5, min(2.0, kelly_scalar))

    def get_combined_scalar(
        self,
        strategy_name: str = None,
        vix_level: float = None,
        current_drawdown: float = None
    ) -> float:
        """
        Calculate combined position size scalar.

        Product of all individual scalars, capped between 0.25 and 1.5.

        Args:
            strategy_name: Name of the strategy
            vix_level: Current VIX level
            current_drawdown: Current drawdown as decimal

        Returns:
            Float combined scalar between 0.25 and 1.5
        """
        # Get individual scalars
        strategy_scalar = self.get_strategy_scalar(strategy_name) if strategy_name else 1.0
        regime_scalar = self.get_regime_scalar(vix_level)
        drawdown_scalar = self.get_drawdown_scalar(current_drawdown)
        momentum_scalar = self.get_win_rate_momentum_scalar(strategy_name) if strategy_name else 1.0
        kelly_scalar = self.get_kelly_scalar(strategy_name) if strategy_name else 1.0

        # Combine (multiplicative)
        combined = strategy_scalar * regime_scalar * drawdown_scalar * momentum_scalar

        # Kelly is averaged in rather than multiplied (to reduce variance)
        combined = (combined + kelly_scalar) / 2

        # Apply caps
        min_scalar = self.config["combined_scalar_min"]
        max_scalar = self.config["combined_scalar_max"]

        return max(min_scalar, min(max_scalar, combined))

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def _build_sizing_context(self, strategy_name: str, context: Dict) -> SizingContext:
        """Build complete sizing context for a trade."""
        # Get VIX
        vix_level = context.get('vix_level') or self._get_current_vix()
        vix_regime = get_vix_regime(vix_level)

        # Get drawdown
        current_drawdown = context.get('current_drawdown', self._current_drawdown)

        # Get strategy stats
        stats = self._get_strategy_stats(strategy_name)

        # Calculate individual scalars
        strategy_scalar = self.get_strategy_scalar(strategy_name)
        regime_scalar = self.get_regime_scalar(vix_level)
        drawdown_scalar = self.get_drawdown_scalar(current_drawdown)
        momentum_scalar = self.get_win_rate_momentum_scalar(strategy_name)
        kelly_scalar = self.get_kelly_scalar(strategy_name)

        # Combined scalar
        combined = strategy_scalar * regime_scalar * drawdown_scalar * momentum_scalar
        combined = (combined + kelly_scalar) / 2
        combined = max(
            self.config["combined_scalar_min"],
            min(self.config["combined_scalar_max"], combined)
        )

        return SizingContext(
            vix_level=vix_level,
            vix_regime=vix_regime,
            current_drawdown=current_drawdown,
            strategy_sharpe=stats.get('sharpe_ratio', 0.0) if stats else 0.0,
            strategy_win_rate=stats.get('win_rate', 0.5) if stats else 0.5,
            recent_win_rate=stats.get('recent_win_rate', 0.5) if stats else 0.5,
            total_trades=stats.get('total_trades', 0) if stats else 0,
            avg_win=stats.get('avg_win', 0.0) if stats else 0.0,
            avg_loss=stats.get('avg_loss', 0.0) if stats else 0.0,
            kelly_fraction=self.config["kelly_fraction"],
            strategy_scalar=strategy_scalar,
            regime_scalar=regime_scalar,
            drawdown_scalar=drawdown_scalar,
            momentum_scalar=momentum_scalar,
            kelly_scalar=kelly_scalar,
            combined_scalar=combined
        )

    # =========================================================================
    # DATABASE INTEGRATION
    # =========================================================================

    def _get_strategy_stats(self, strategy_name: str) -> Optional[Dict]:
        """
        Get strategy statistics from performance.db.

        Returns cached data if available and fresh.
        """
        # Check cache
        now = datetime.now()
        if (self._cache_timestamp is not None and
            (now - self._cache_timestamp) < self._cache_ttl and
            strategy_name in self._strategy_stats_cache):
            return self._strategy_stats_cache.get(strategy_name)

        # Fetch from database
        try:
            db_path = DATABASES.get('performance')
            if not db_path or not db_path.exists():
                return None

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get strategy stats
            cursor.execute("""
                SELECT * FROM strategy_stats WHERE strategy = ?
            """, (strategy_name,))
            row = cursor.fetchone()

            if row:
                stats = dict(row)

                # Get recent daily stats for rolling metrics
                cursor.execute("""
                    SELECT
                        SUM(wins) as recent_wins,
                        SUM(losses) as recent_losses,
                        AVG(sharpe_20d) as recent_sharpe
                    FROM strategy_daily
                    WHERE strategy = ?
                    AND date >= date('now', '-10 days')
                """, (strategy_name,))
                recent = cursor.fetchone()

                if recent:
                    recent_wins = recent['recent_wins'] or 0
                    recent_losses = recent['recent_losses'] or 0
                    recent_total = recent_wins + recent_losses
                    if recent_total > 0:
                        stats['recent_win_rate'] = recent_wins / recent_total
                    else:
                        stats['recent_win_rate'] = stats.get('win_rate', 0.5)

                conn.close()

                # Update cache
                self._strategy_stats_cache[strategy_name] = stats
                self._cache_timestamp = now

                return stats

            conn.close()
            return None

        except Exception as e:
            logger.warning(f"Error fetching strategy stats: {e}")
            return None

    def _get_current_vix(self) -> float:
        """Get current VIX level."""
        if self._current_vix is not None:
            return self._current_vix

        try:
            from data.fetchers.vix import get_current_vix
            self._current_vix = get_current_vix()
        except Exception as e:
            logger.warning(f"Could not get VIX: {e}")
            self._current_vix = 20.0  # Default

        return self._current_vix

    def _get_available_capital(self) -> float:
        """Get available capital for new positions."""
        buffer = self.total_capital * self.cash_buffer_pct
        return max(self._cash - buffer, 0)

    def _log_sizing_decision(self, strategy: str, result: AdaptivePositionResult):
        """Log position sizing decision to database."""
        try:
            db_path = DATABASES.get('performance')
            if not db_path:
                return

            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sizing_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    shares INTEGER,
                    dollar_value REAL,
                    base_shares INTEGER,
                    base_dollar_value REAL,
                    final_scalar REAL,
                    strategy_scalar REAL,
                    regime_scalar REAL,
                    drawdown_scalar REAL,
                    momentum_scalar REAL,
                    kelly_scalar REAL,
                    vix_level REAL,
                    current_drawdown REAL,
                    strategy_sharpe REAL,
                    was_capped INTEGER,
                    cap_reason TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                INSERT INTO sizing_decisions (
                    timestamp, strategy, symbol, shares, dollar_value,
                    base_shares, base_dollar_value, final_scalar,
                    strategy_scalar, regime_scalar, drawdown_scalar,
                    momentum_scalar, kelly_scalar, vix_level,
                    current_drawdown, strategy_sharpe, was_capped,
                    cap_reason, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                strategy,
                result.symbol,
                result.shares,
                result.dollar_value,
                result.base_shares,
                result.base_dollar_value,
                result.final_scalar,
                result.context.strategy_scalar,
                result.context.regime_scalar,
                result.context.drawdown_scalar,
                result.context.momentum_scalar,
                result.context.kelly_scalar,
                result.context.vix_level,
                result.context.current_drawdown,
                result.context.strategy_sharpe,
                1 if result.was_capped else 0,
                result.cap_reason,
                json.dumps(result.to_dict())
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Error logging sizing decision: {e}")

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def update_capital(self, total_capital: float, cash: float = None):
        """Update capital levels."""
        self.total_capital = total_capital
        if cash is not None:
            self._cash = cash
        self._base_sizer.update_capital(total_capital, cash)

    def update_positions(self, positions: Dict[str, float]):
        """Update current positions."""
        self._positions = positions
        positions_value = sum(positions.values())
        self._cash = self.total_capital - positions_value
        self._base_sizer.update_positions(positions)

    def update_drawdown(self, drawdown_pct: float):
        """Update current drawdown level."""
        self._current_drawdown = max(0.0, drawdown_pct)
        self._base_sizer.update_drawdown(drawdown_pct)

    def update_vix(self, vix_level: float):
        """Update current VIX level."""
        self._current_vix = vix_level

    def invalidate_cache(self):
        """Invalidate cached strategy stats."""
        self._strategy_stats_cache.clear()
        self._cache_timestamp = None

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def get_sizing_analysis(self, strategy_name: str) -> Dict:
        """
        Get detailed analysis of current sizing parameters for a strategy.

        Returns:
            Dict with all scalar values and explanations
        """
        vix_level = self._get_current_vix()
        stats = self._get_strategy_stats(strategy_name)

        analysis = {
            'strategy': strategy_name,
            'timestamp': datetime.now().isoformat(),

            # Current state
            'current_state': {
                'vix_level': vix_level,
                'vix_regime': get_vix_regime(vix_level),
                'current_drawdown': self._current_drawdown,
                'total_capital': self.total_capital,
                'base_risk_per_trade': self.base_risk_per_trade,
            },

            # Strategy stats
            'strategy_stats': stats or {},

            # Individual scalars
            'scalars': {
                'strategy': {
                    'value': self.get_strategy_scalar(strategy_name),
                    'sharpe': stats.get('sharpe_ratio') if stats else None,
                    'explanation': self._explain_strategy_scalar(strategy_name)
                },
                'regime': {
                    'value': self.get_regime_scalar(vix_level),
                    'vix': vix_level,
                    'explanation': self._explain_regime_scalar(vix_level)
                },
                'drawdown': {
                    'value': self.get_drawdown_scalar(),
                    'drawdown_pct': self._current_drawdown,
                    'explanation': self._explain_drawdown_scalar()
                },
                'momentum': {
                    'value': self.get_win_rate_momentum_scalar(strategy_name),
                    'explanation': self._explain_momentum_scalar(strategy_name)
                },
                'kelly': {
                    'value': self.get_kelly_scalar(strategy_name),
                    'explanation': self._explain_kelly_scalar(strategy_name)
                }
            },

            # Combined
            'combined_scalar': self.get_combined_scalar(strategy_name, vix_level),

            # Effective sizing
            'effective_risk_per_trade': (
                self.base_risk_per_trade *
                self.get_combined_scalar(strategy_name, vix_level)
            )
        }

        return analysis

    def _explain_strategy_scalar(self, strategy_name: str) -> str:
        """Explain strategy scalar value."""
        stats = self._get_strategy_stats(strategy_name)
        if not stats:
            return "No strategy data available, using base sizing (1.0)"

        sharpe = stats.get('sharpe_ratio', 0.0)
        scalar = self.get_strategy_scalar(strategy_name)

        if sharpe < 0:
            return f"Sharpe ratio ({sharpe:.2f}) is negative, scaling down to {scalar:.2f}"
        elif sharpe > 1.0:
            return f"Sharpe ratio ({sharpe:.2f}) is excellent, scaling up to {scalar:.2f}"
        else:
            return f"Sharpe ratio ({sharpe:.2f}) is moderate, scalar is {scalar:.2f}"

    def _explain_regime_scalar(self, vix_level: float) -> str:
        """Explain regime scalar value."""
        regime = get_vix_regime(vix_level)
        scalar = self.get_regime_scalar(vix_level)

        explanations = {
            "low": f"VIX ({vix_level:.1f}) is low - favorable conditions, scaling up to {scalar:.2f}",
            "normal": f"VIX ({vix_level:.1f}) is normal - using base sizing ({scalar:.2f})",
            "high": f"VIX ({vix_level:.1f}) is elevated - scaling down to {scalar:.2f}",
            "extreme": f"VIX ({vix_level:.1f}) is extreme - defensive mode, scaling down to {scalar:.2f}"
        }

        return explanations.get(regime, f"VIX is {vix_level:.1f}, scalar is {scalar:.2f}")

    def _explain_drawdown_scalar(self) -> str:
        """Explain drawdown scalar value."""
        dd = self._current_drawdown
        scalar = self.get_drawdown_scalar()

        if dd < 0.05:
            return f"Drawdown ({dd:.1%}) is minimal - using full sizing ({scalar:.2f})"
        elif dd < 0.15:
            return f"Drawdown ({dd:.1%}) is moderate - reducing size to {scalar:.2f}"
        else:
            return f"Drawdown ({dd:.1%}) is significant - defensive sizing at {scalar:.2f}"

    def _explain_momentum_scalar(self, strategy_name: str) -> str:
        """Explain momentum scalar value."""
        stats = self._get_strategy_stats(strategy_name)
        if not stats:
            return "No trade history, using base sizing (1.0)"

        recent = stats.get('recent_win_rate', 0.5)
        baseline = stats.get('win_rate', 0.5)
        scalar = self.get_win_rate_momentum_scalar(strategy_name)

        if scalar > 1.0:
            return f"Win rate improving ({recent:.0%} recent vs {baseline:.0%} baseline) - bonus applied ({scalar:.2f})"
        elif scalar < 1.0:
            return f"Win rate declining ({recent:.0%} recent vs {baseline:.0%} baseline) - penalty applied ({scalar:.2f})"
        else:
            return f"Win rate stable ({baseline:.0%}) - no adjustment ({scalar:.2f})"

    def _explain_kelly_scalar(self, strategy_name: str) -> str:
        """Explain Kelly scalar value."""
        stats = self._get_strategy_stats(strategy_name)
        scalar = self.get_kelly_scalar(strategy_name)

        if not stats or stats.get('total_trades', 0) < self.config["kelly_min_trades"]:
            return f"Insufficient trades for Kelly criterion - using base sizing ({scalar:.2f})"

        win_rate = stats.get('win_rate', 0.5)
        avg_win = stats.get('avg_win', 0.0)
        avg_loss = stats.get('avg_loss', 0.0)

        if scalar > 1.0:
            return f"Kelly suggests increased size (win rate {win_rate:.0%}, avg win/loss ratio {abs(avg_win/avg_loss) if avg_loss else 0:.2f}) - scalar {scalar:.2f}"
        else:
            return f"Kelly suggests reduced size based on edge metrics - scalar {scalar:.2f}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_adaptive_sizer(capital: float = TOTAL_CAPITAL) -> AdaptivePositionSizer:
    """Create an adaptive position sizer with default configuration."""
    return AdaptivePositionSizer(total_capital=capital)


def calculate_adaptive_position(
    strategy: str,
    symbol: str,
    price: float,
    stop_loss: float,
    capital: float = TOTAL_CAPITAL,
    context: Dict = None
) -> Tuple[int, float]:
    """
    Quick adaptive position calculation.

    Args:
        strategy: Strategy name
        symbol: Stock symbol
        price: Current price
        stop_loss: Stop loss price
        capital: Total capital
        context: Additional context

    Returns:
        Tuple of (shares, dollar_amount)
    """
    sizer = AdaptivePositionSizer(total_capital=capital)
    return sizer.calculate_position_size(strategy, symbol, price, stop_loss, context)


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Adaptive Position Sizer Test")
    print("=" * 60)

    sizer = AdaptivePositionSizer(
        total_capital=100000,
        risk_per_trade=0.02,
        max_position_size=15000
    )

    print(f"\nCapital: ${sizer.total_capital:,.0f}")
    print(f"Base risk per trade: {sizer.base_risk_per_trade:.1%}")

    # Test different scenarios
    print("\n" + "-" * 40)
    print("Scenario 1: Normal conditions")
    sizer.update_vix(18.0)
    sizer.update_drawdown(0.03)

    shares, amount = sizer.calculate_position_size(
        strategy='mean_reversion',
        symbol='AAPL',
        price=175.00,
        stop_loss=170.00
    )
    print(f"  AAPL: {shares} shares (${amount:,.0f})")

    print("\n" + "-" * 40)
    print("Scenario 2: High VIX")
    sizer.update_vix(35.0)

    shares, amount = sizer.calculate_position_size(
        strategy='mean_reversion',
        symbol='AAPL',
        price=175.00,
        stop_loss=170.00
    )
    print(f"  AAPL: {shares} shares (${amount:,.0f})")

    print("\n" + "-" * 40)
    print("Scenario 3: In drawdown")
    sizer.update_vix(20.0)
    sizer.update_drawdown(0.12)

    shares, amount = sizer.calculate_position_size(
        strategy='mean_reversion',
        symbol='AAPL',
        price=175.00,
        stop_loss=170.00
    )
    print(f"  AAPL: {shares} shares (${amount:,.0f})")

    print("\n" + "-" * 40)
    print("Scalar Analysis")

    # Reset to normal
    sizer.update_vix(20.0)
    sizer.update_drawdown(0.05)

    analysis = sizer.get_sizing_analysis('mean_reversion')

    print(f"\n  Strategy: {analysis['strategy']}")
    print(f"  VIX: {analysis['current_state']['vix_level']:.1f} ({analysis['current_state']['vix_regime']})")
    print(f"  Drawdown: {analysis['current_state']['current_drawdown']:.1%}")

    print("\n  Scalars:")
    for name, data in analysis['scalars'].items():
        print(f"    {name}: {data['value']:.2f} - {data['explanation']}")

    print(f"\n  Combined scalar: {analysis['combined_scalar']:.2f}")
    print(f"  Effective risk per trade: {analysis['effective_risk_per_trade']:.2%}")

    print("\n" + "=" * 60)
    print("Test complete")
