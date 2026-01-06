"""
Optimized Backtesting Engine
============================
Dramatically faster backtesting through:
1. Pre-indexed DataFrames (convert once, not per-loop iteration)
2. Lazy data loading (only prepare full data when strategy needs it)
3. Vectorized price lookups (numpy arrays, not DataFrame access)
4. Strategy type hints (skip heavy ops for infrequent strategies)

Performance target: 2-5 seconds per backtest (vs 30 seconds original)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import uuid

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VALIDATION, TRANSACTION_COSTS_BPS, get_transaction_cost
from strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    run_id: str
    strategy: str
    start_date: str
    end_date: str
    
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    
    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Validation
    meets_threshold: bool = False
    vs_research_pct: float = 0.0
    
    # Details
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'run_id': self.run_id,
            'strategy': self.strategy,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'meets_threshold': self.meets_threshold,
            'vs_research_pct': self.vs_research_pct,
        }


@dataclass 
class SimulatedTrade:
    """A simulated trade during backtesting."""
    symbol: str
    side: str
    entry_date: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    target_price: float
    
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    def close(self, exit_date: datetime, exit_price: float, reason: str):
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        
        if self.side == 'BUY':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100
        
        self.pnl -= (self.commission + self.slippage)


class PreIndexedData:
    """
    Pre-indexed market data for fast lookups.
    
    Converts DataFrames once at init, then uses numpy arrays
    for O(1) price lookups instead of DataFrame slicing.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Pre-index all data for fast access.
        
        Args:
            data: Dict of symbol -> DataFrame with OHLCV
        """
        self.symbols: List[str] = []
        self.dates: np.ndarray = None  # Unified date array
        self.date_to_idx: Dict[pd.Timestamp, int] = {}
        
        # Per-symbol data as numpy arrays
        self.close_prices: Dict[str, np.ndarray] = {}
        self.atr_values: Dict[str, np.ndarray] = {}
        self.symbol_dates: Dict[str, np.ndarray] = {}  # Each symbol's date array
        self.symbol_date_idx: Dict[str, Dict[pd.Timestamp, int]] = {}  # Date -> row index per symbol
        
        # Original DataFrames (for strategies that need full data)
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        self._build_index(data)
    
    def _build_index(self, data: Dict[str, pd.DataFrame]):
        """Build all indexes from data."""
        # First pass: collect all dates and normalize DataFrames
        all_dates: Set[pd.Timestamp] = set()
        
        for symbol, df in data.items():
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                else:
                    logger.warning(f"Symbol {symbol} has no datetime - skipping")
                    continue
            
            # Remove timezone
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            
            if len(df) < 20:
                continue
            
            # Store normalized DataFrame
            self.dataframes[symbol] = df
            self.symbols.append(symbol)
            
            # Collect dates
            all_dates.update(df.index.tolist())
            
            # Store symbol-specific date index
            self.symbol_dates[symbol] = df.index.values
            self.symbol_date_idx[symbol] = {d: i for i, d in enumerate(df.index)}
            
            # Extract numpy arrays for fast access
            self.close_prices[symbol] = df['close'].values
            self.atr_values[symbol] = df['atr'].values if 'atr' in df.columns else np.full(len(df), df['close'].mean() * 0.02)
        
        # Create unified date array
        self.dates = np.array(sorted(all_dates))
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        
        logger.debug(f"Pre-indexed {len(self.symbols)} symbols, {len(self.dates)} dates")
    
    def get_price(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """Get close price for symbol on date (O(1) lookup)."""
        if symbol not in self.symbol_date_idx:
            return None
        
        idx_map = self.symbol_date_idx[symbol]
        
        # Find the most recent date <= requested date
        if date in idx_map:
            return float(self.close_prices[symbol][idx_map[date]])
        
        # Binary search for most recent available date
        symbol_dates = self.symbol_dates[symbol]
        mask = symbol_dates <= date
        if not mask.any():
            return None
        
        idx = np.where(mask)[0][-1]
        return float(self.close_prices[symbol][idx])
    
    def get_prices_on_date(self, date: pd.Timestamp) -> Dict[str, float]:
        """Get all available prices for a date."""
        prices = {}
        for symbol in self.symbols:
            price = self.get_price(symbol, date)
            if price is not None:
                prices[symbol] = price
        return prices
    
    def get_atr(self, symbol: str, date: pd.Timestamp) -> float:
        """Get ATR for symbol on date."""
        if symbol not in self.symbol_date_idx:
            return 0.02  # Default 2%
        
        idx_map = self.symbol_date_idx[symbol]
        
        if date in idx_map:
            return float(self.atr_values[symbol][idx_map[date]])
        
        # Use last available
        symbol_dates = self.symbol_dates[symbol]
        mask = symbol_dates <= date
        if not mask.any():
            return 0.02
        
        idx = np.where(mask)[0][-1]
        return float(self.atr_values[symbol][idx])
    
    def get_data_slice(self, symbol: str, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Get DataFrame slice up to end_date (for strategies that need full history).
        
        This is the expensive operation - only call when necessary!
        """
        if symbol not in self.dataframes:
            return None
        
        df = self.dataframes[symbol]
        
        # Fast slice using pre-computed index
        if end_date in self.symbol_date_idx[symbol]:
            end_idx = self.symbol_date_idx[symbol][end_date] + 1
            return df.iloc[:end_idx]
        
        # Fallback to mask (slower)
        return df[df.index <= end_date]
    
    def get_data_slices(self, symbols: List[str], end_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Get DataFrame slices for multiple symbols."""
        return {s: self.get_data_slice(s, end_date) for s in symbols if self.get_data_slice(s, end_date) is not None}


class OptimizedBacktester:
    """
    Optimized backtesting engine.
    
    Key optimizations:
    1. Pre-indexed data for O(1) price lookups
    2. Lazy DataFrame slicing (only when strategy needs full data)
    3. Strategy frequency hints (skip heavy ops for infrequent strategies)
    4. Vectorized calculations where possible
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_per_trade: float = 0,
                 slippage_model: str = 'volatility'):
        
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_model = slippage_model
        
        # State during backtest
        self._cash = initial_capital
        self._positions: Dict[str, SimulatedTrade] = {}
        self._closed_trades: List[SimulatedTrade] = []
        self._equity_curve: List[Tuple[datetime, float]] = []
    
    def _reset(self):
        """Reset state for new backtest."""
        self._cash = self.initial_capital
        self._positions = {}
        self._closed_trades = []
        self._equity_curve = []
    
    def _reset_strategy(self, strategy: BaseStrategy):
        """Reset strategy state for fresh backtest."""
        if hasattr(strategy, 'last_rebalance_month'):
            strategy.last_rebalance_month = None
        if hasattr(strategy, 'last_regime'):
            strategy.last_regime = None
        if hasattr(strategy, 'current_positions'):
            strategy.current_positions = {}
    
    def _estimate_slippage(self, price: float, atr: float, side: str) -> float:
        """Estimate slippage based on volatility."""
        if self.slippage_model == 'none':
            return 0.0
        
        if self.slippage_model == 'fixed':
            return price * 0.001
        
        # Volatility-based
        atr_pct = atr / price if price > 0 else 0.02
        slippage_pct = min(atr_pct * 0.10, 0.005)
        return price * slippage_pct
    
    def _calculate_position_size(self, price: float, signal: Signal, available_capital: float) -> float:
        """Calculate position size based on signal and risk management."""
        if signal.position_size_pct:
            target_value = available_capital * signal.position_size_pct
        else:
            target_value = available_capital * 0.10
        
        target_value = min(target_value, self._cash * 0.95)
        return target_value / price if price > 0 else 0
    
    def _get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            trade.quantity * prices.get(symbol, trade.entry_price)
            for symbol, trade in self._positions.items()
        )
        return self._cash + positions_value
    
    def _is_action_day(self, strategy: BaseStrategy, current_date: pd.Timestamp, day_index: int) -> bool:
        """
        Check if strategy needs to act on this day.
        
        For monthly strategies, skip expensive data prep on non-action days.
        """
        # Check for strategy frequency hints
        frequency = getattr(strategy, 'rebalance_frequency', 'daily')
        
        if frequency == 'monthly':
            # Only act on first trading day of month
            if day_index == 0:
                return True
            # Check if month changed
            prev_date = self._last_processed_date
            if prev_date is None:
                return True
            return current_date.month != prev_date.month
        
        return True  # Daily strategies always act
    
    def run(self,
            strategy: BaseStrategy,
            data: Dict[str, pd.DataFrame],
            start_date: datetime = None,
            end_date: datetime = None,
            vix_data: pd.DataFrame = None) -> BacktestResult:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Strategy instance to test
            data: Dict mapping symbol to DataFrame with OHLCV + indicators
            start_date: Backtest start date
            end_date: Backtest end date
            vix_data: Optional VIX data for regime detection
            
        Returns:
            BacktestResult with performance metrics
        """
        self._reset()
        self._reset_strategy(strategy)
        self._last_processed_date = None
        
        run_id = str(uuid.uuid4())[:8]
        
        # Pre-index all data (do expensive work ONCE)
        indexed = PreIndexedData(data)
        
        if len(indexed.symbols) == 0:
            logger.error("No valid data after indexing")
            return BacktestResult(run_id=run_id, strategy=strategy.name, start_date="", end_date="")
        
        # Determine date range
        dates = indexed.dates
        
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            if start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            dates = dates[dates >= start_date]
        
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            if end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            dates = dates[dates <= end_date]
        
        if len(dates) == 0:
            logger.error("No dates in range")
            return BacktestResult(run_id=run_id, strategy=strategy.name, start_date="", end_date="")
        
        # Pre-process VIX data
        vix_regimes = {}
        if vix_data is not None and 'regime' in vix_data.columns:
            # Ensure proper index
            if not isinstance(vix_data.index, pd.DatetimeIndex):
                if 'timestamp' in vix_data.columns:
                    vix_data = vix_data.set_index('timestamp')
            if vix_data.index.tz is not None:
                vix_data.index = vix_data.index.tz_localize(None)
            
            # Pre-compute regime for each date
            for d in dates:
                mask = vix_data.index <= d
                if mask.any():
                    vix_regimes[d] = vix_data.loc[mask, 'regime'].iloc[-1]
                else:
                    vix_regimes[d] = 'normal'
        
        actual_start = dates[0]
        actual_end = dates[-1]
        
        logger.info(f"Backtesting {strategy.name} from {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')} ({len(dates)} days)")
        
        # Track which symbols we need full data for (only when generating signals)
        symbols_needing_data: Set[str] = set(indexed.symbols)
        
        # Main backtest loop
        for i, current_date in enumerate(dates):
            # Get current prices (FAST - numpy lookup)
            current_prices = indexed.get_prices_on_date(current_date)
            
            # Get VIX regime
            vix_regime = vix_regimes.get(current_date, 'normal')
            
            # Check existing positions for exits (uses current_prices only - FAST)
            positions_to_close = []
            for symbol, trade in self._positions.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                
                # Check stop loss
                if trade.side == 'BUY' and current_price <= trade.stop_loss:
                    positions_to_close.append((symbol, current_price, 'stop_loss'))
                elif trade.side == 'SELL' and current_price >= trade.stop_loss:
                    positions_to_close.append((symbol, current_price, 'stop_loss'))
                # Check target
                elif trade.side == 'BUY' and current_price >= trade.target_price:
                    positions_to_close.append((symbol, current_price, 'target'))
                elif trade.side == 'SELL' and current_price <= trade.target_price:
                    positions_to_close.append((symbol, current_price, 'target'))
            
            # Close positions
            for symbol, exit_price, reason in positions_to_close:
                trade = self._positions.pop(symbol)
                atr = indexed.get_atr(symbol, current_date)
                slippage = self._estimate_slippage(exit_price, atr, 'SELL' if trade.side == 'BUY' else 'BUY')
                
                if trade.side == 'BUY':
                    exit_price -= slippage
                else:
                    exit_price += slippage
                
                trade.slippage += slippage * trade.quantity
                trade.close(current_date.to_pydatetime(), exit_price, reason)
                
                self._cash += trade.quantity * exit_price
                self._closed_trades.append(trade)
            
            # Check if this is an action day for signal generation
            is_action_day = self._is_action_day(strategy, current_date, i)
            
            signals = []
            if is_action_day:
                # EXPENSIVE: Only get full data slices when needed
                current_data = indexed.get_data_slices(list(symbols_needing_data), current_date)
                
                # Add backtest date for strategies that need it
                for df in current_data.values():
                    df.attrs['backtest_date'] = current_date
                    break  # Only need once
                
                current_positions = list(self._positions.keys())
                signals = strategy.generate_signals(current_data, current_positions, vix_regime)
            
            # Process signals
            for signal in signals:
                if signal.signal_type == SignalType.CLOSE:
                    if signal.symbol in self._positions:
                        trade = self._positions.pop(signal.symbol)
                        exit_price = current_prices.get(signal.symbol, trade.entry_price)
                        trade.close(current_date.to_pydatetime(), exit_price, 'signal')
                        self._cash += trade.quantity * exit_price
                        self._closed_trades.append(trade)
                
                elif signal.signal_type == SignalType.BUY:
                    if signal.symbol in self._positions:
                        continue
                    
                    if signal.symbol not in current_prices:
                        continue
                    
                    price = current_prices[signal.symbol]
                    quantity = self._calculate_position_size(price, signal, self._cash)
                    
                    if quantity <= 0:
                        continue
                    
                    atr = indexed.get_atr(signal.symbol, current_date)
                    slippage = self._estimate_slippage(price, atr, 'BUY')
                    entry_price = price + slippage
                    
                    cost = quantity * entry_price + self.commission_per_trade
                    if cost > self._cash:
                        quantity = (self._cash - self.commission_per_trade) / entry_price
                    
                    if quantity <= 0:
                        continue
                    
                    trade = SimulatedTrade(
                        symbol=signal.symbol,
                        side='BUY',
                        entry_date=current_date.to_pydatetime(),
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=signal.stop_loss or entry_price * 0.95,
                        target_price=signal.target_price or entry_price * 1.10,
                        commission=self.commission_per_trade,
                        slippage=slippage * quantity
                    )
                    
                    self._positions[signal.symbol] = trade
                    self._cash -= quantity * entry_price + self.commission_per_trade
            
            # Record equity
            portfolio_value = self._get_portfolio_value(current_prices)
            self._equity_curve.append((current_date, portfolio_value))
            
            self._last_processed_date = current_date
        
        # Close remaining positions
        for symbol, trade in list(self._positions.items()):
            exit_price = current_prices.get(symbol, trade.entry_price)
            trade.close(actual_end.to_pydatetime(), exit_price, 'backtest_end')
            self._closed_trades.append(trade)
        
        return self._calculate_metrics(strategy.name, run_id, actual_start, actual_end)
    
    def _calculate_metrics(self, strategy_name: str, run_id: str,
                           start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        
        result = BacktestResult(
            run_id=run_id,
            strategy=strategy_name,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not self._equity_curve:
            return result
        
        # Extract equity series
        equity_values = [e[1] for e in self._equity_curve]
        result.equity_curve = equity_values
        
        # Returns
        result.total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Annualize
        days = len(self._equity_curve)
        years = days / 252
        if years > 0:
            result.annual_return = ((1 + result.total_return / 100) ** (1 / years) - 1) * 100
        
        # Daily returns for risk metrics
        equity_series = pd.Series(equity_values)
        daily_returns = equity_series.pct_change().dropna()
        
        if len(daily_returns) > 0:
            result.volatility = daily_returns.std() * np.sqrt(252) * 100
            
            mean_return = daily_returns.mean() * 252
            std_return = daily_returns.std() * np.sqrt(252)
            result.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            result.sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax
            result.max_drawdown_pct = drawdown.min() * 100
            result.max_drawdown = (cummax - equity_series).max()
        
        # Trade statistics
        result.total_trades = len(self._closed_trades)
        result.winning_trades = sum(1 for t in self._closed_trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in self._closed_trades if t.pnl <= 0)
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades * 100
            
            pnls = [t.pnl for t in self._closed_trades]
            result.avg_trade_pnl = np.mean(pnls)
            
            winners = [t.pnl for t in self._closed_trades if t.pnl > 0]
            losers = [t.pnl for t in self._closed_trades if t.pnl <= 0]
            
            result.avg_winner = np.mean(winners) if winners else 0
            result.avg_loser = np.mean(losers) if losers else 0
            
            gross_profit = sum(winners) if winners else 0
            gross_loss = abs(sum(losers)) if losers else 0
            # Clamp profit_factor to avoid infinity propagation through analysis
            result.profit_factor = min(10.0, gross_profit / gross_loss) if gross_loss > 0 else 10.0
        
        # Costs
        result.total_commission = sum(t.commission for t in self._closed_trades)
        result.total_slippage = sum(t.slippage for t in self._closed_trades)
        
        # Validation
        if strategy_name in VALIDATION:
            min_sharpe = VALIDATION[strategy_name].get('min_sharpe', 0)
            research_sharpe = VALIDATION[strategy_name].get('research_sharpe', 0)
            
            result.meets_threshold = result.sharpe_ratio >= min_sharpe
            result.vs_research_pct = (result.sharpe_ratio / research_sharpe * 100) if research_sharpe > 0 else 0
        
        # Trade details
        result.trades = [
            {
                'symbol': t.symbol,
                'side': t.side,
                'entry_date': str(t.entry_date),
                'entry_price': t.entry_price,
                'exit_date': str(t.exit_date),
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason
            }
            for t in self._closed_trades
        ]
        
        return result


# Alias for drop-in replacement
Backtester = OptimizedBacktester


def run_backtest(strategy: BaseStrategy, data: Dict[str, pd.DataFrame], **kwargs) -> BacktestResult:
    """Convenience function to run a single backtest."""
    backtester = OptimizedBacktester(**kwargs)
    return backtester.run(strategy, data)


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("OPTIMIZED BACKTESTER BENCHMARK")
    print("=" * 60)
    
    # Import and create test data
    from data.cached_data_manager import CachedDataManager
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from config import DIRS
    
    # Load real data
    print("\nLoading data...")
    data_mgr = CachedDataManager()
    data_mgr.load_all()
    
    # Get top 100 symbols
    metadata = data_mgr.get_all_metadata()
    top_symbols = sorted(metadata.items(), key=lambda x: x[1].get('dollar_volume', 0), reverse=True)[:100]
    data = {s: data_mgr.get_bars(s) for s, _ in top_symbols}
    
    print(f"Testing with {len(data)} symbols")
    
    # Load VIX
    vix_path = DIRS.get('vix') / 'vix.parquet'
    vix_data = None
    if vix_path.exists():
        vix_data = pd.read_parquet(vix_path)
        if 'timestamp' in vix_data.columns:
            vix_data = vix_data.set_index('timestamp')
        if vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)
        vix_data['regime'] = 'normal'
        vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
        vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
    
    strategy = VolManagedMomentumStrategy()
    
    # Benchmark
    print("\nRunning optimized backtest...")
    start_time = time.time()
    
    backtester = OptimizedBacktester(initial_capital=100000)
    result = backtester.run(strategy, data, vix_data=vix_data)
    
    elapsed = time.time() - start_time
    
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"Time: {elapsed:.2f}s")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Annual Return: {result.annual_return:.1f}%")
    print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")
    print(f"Win Rate: {result.win_rate:.1f}%")
