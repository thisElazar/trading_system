"""
Backtesting Engine
==================
Walk-forward backtesting with proper train/test splits.
Prevents overfitting by never peeking at future data.

Key features:
- Walk-forward validation (rolling windows)
- Transaction cost modeling
- Slippage estimation
- Regime-aware performance attribution

Transaction Cost Model
======================
This backtester models realistic transaction costs:

1. Bid-Ask Spread: 7 bps base (conservative for liquid stocks)
   - Small caps may have 20-50 bps spreads
   - This is the MINIMUM cost you pay to cross the spread

2. Volatility Slippage: 20% of ATR
   - Captures market movement during execution
   - More volatile stocks = more slippage
   - Reflects the reality that you rarely get the exact price you see

3. Maximum Slippage: 1.5% cap
   - Prevents unrealistic costs in extreme scenarios
   - Small cap illiquid names can hit this cap
   - Some small caps can have 2%+ effective spreads

4. Commission: $0 (Alpaca)
   - Note: Payment for order flow may still affect execution quality
   - "Free" trades aren't truly free - costs are hidden in execution

Total Round-Trip Cost Estimate:
- Large caps (>$10B): 15-30 bps
- Mid caps ($2-10B): 30-60 bps
- Small caps (<$2B): 60-150 bps

These estimates are CONSERVATIVE by design. Actual costs may be lower for
patient limit orders, but can be significantly higher for urgent market orders
or during volatile periods. The GA optimizer should use 'conservative' costs
to avoid overfitting to unrealistically low friction assumptions.

Cost Model Options:
- 'conservative': Higher costs, safer for optimization (spread: 10 bps, ATR: 25%, max: 2%)
- 'moderate': Balanced assumptions (spread: 7 bps, ATR: 20%, max: 1.5%)
- 'aggressive': Lower costs, use only for comparison (spread: 5 bps, ATR: 10%, max: 0.5%)

Reference: Almgren & Chriss (2000), Kissell & Glantz (2003)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import uuid
import bisect

import pandas as pd
import numpy as np

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VALIDATION, TRANSACTION_COSTS_BPS, get_transaction_cost
from strategies.base import BaseStrategy, Signal, SignalType
from utils.timezone import normalize_dataframe, normalize_timestamp, normalize_index

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
    avg_slippage_pct: float = 0.0  # Average slippage as % of trade value
    total_slippage_cost: float = 0.0  # Alias for total_slippage for clarity

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
    side: str  # 'BUY' or 'SELL'
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
        
        # Subtract costs
        self.pnl -= (self.commission + self.slippage)


class Backtester:
    """
    Walk-forward backtesting engine.

    Key principles:
    1. Never peek at future data
    2. Include realistic transaction costs
    3. Model slippage based on volatility
    4. Track regime-specific performance

    Cost Model Options:
    - 'conservative': Higher costs, safer for GA optimization to avoid overfitting
    - 'moderate': Balanced assumptions for typical backtesting
    - 'aggressive': Lower costs, use only for comparison (not recommended)
    """

    # Cost model parameters: spread_bps, atr_pct, max_slippage_pct
    COST_MODELS = {
        'conservative': {'spread_bps': 10, 'atr_pct': 0.25, 'max_pct': 0.02},
        'moderate': {'spread_bps': 7, 'atr_pct': 0.20, 'max_pct': 0.015},
        'aggressive': {'spread_bps': 5, 'atr_pct': 0.10, 'max_pct': 0.005},
    }

    def __init__(self,
                 initial_capital: float = 100000,
                 commission_per_trade: float = 0,  # Alpaca is commission-free
                 cost_model: str = 'moderate',  # 'conservative', 'moderate', 'aggressive'
                 slippage_model: str = 'volatility'):  # 'fixed', 'volatility', 'none' (legacy)
        """
        Initialize backtester with transaction cost model.

        Args:
            initial_capital: Starting capital in dollars
            commission_per_trade: Commission per trade (default 0 for Alpaca)
            cost_model: Cost model to use - 'conservative', 'moderate', or 'aggressive'
                       GA optimization should use 'conservative' to avoid overfitting
            slippage_model: Legacy parameter, kept for compatibility
                           'none' will disable all slippage regardless of cost_model
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_model = slippage_model

        # Set cost parameters based on model
        if cost_model not in self.COST_MODELS:
            logger.warning(f"Unknown cost_model '{cost_model}', using 'moderate'")
            cost_model = 'moderate'
        self.cost_model = cost_model
        self.cost_params = self.COST_MODELS[cost_model]

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
    
    def calculate_slippage(self, price: float, atr: float, volume: float = None) -> float:
        """
        Calculate realistic slippage including bid-ask spread and volatility.

        This model combines:
        1. Base bid-ask spread (minimum cost to cross the spread)
        2. Volatility-based slippage (market movement during execution)
        3. Optional volume-based market impact (for larger orders)

        Research basis:
        - Small caps: 10-50 bps spread typical
        - Large caps: 2-10 bps spread typical
        - Market impact: ~0.1% per $1M traded (Almgren 2005)

        Args:
            price: Current price per share
            atr: Average True Range (14-day typical)
            volume: Trade volume in dollars (optional, for market impact)

        Returns:
            Estimated slippage in dollars per share (always positive)
        """
        # Legacy support: 'none' disables all slippage
        if self.slippage_model == 'none':
            return 0.0

        # Legacy support: 'fixed' uses simple 0.1% slippage
        if self.slippage_model == 'fixed':
            return price * 0.001

        # Get cost parameters from selected model
        spread_bps = self.cost_params['spread_bps']
        atr_pct = self.cost_params['atr_pct']
        max_pct = self.cost_params['max_pct']

        # 1. Base bid-ask spread (minimum transaction cost)
        # This is what you pay just to cross the spread
        base_spread = price * (spread_bps / 10000)

        # 2. Volatility-based slippage
        # Captures market movement during order execution
        # More volatile stocks = more slippage
        if price > 0 and atr > 0:
            vol_slippage = atr * atr_pct
        else:
            # Fallback: assume 2% daily volatility, use 20% of that
            vol_slippage = price * 0.02 * atr_pct

        # Total slippage = spread + volatility
        total_slippage = base_spread + vol_slippage

        # Cap at maximum percentage to avoid unrealistic costs
        # Small caps can legitimately have 1-2% slippage, so cap is meaningful
        max_slippage = price * max_pct
        total_slippage = min(total_slippage, max_slippage)

        return total_slippage

    def _estimate_slippage(self, price: float, atr: float, side: str) -> float:
        """
        Legacy wrapper for calculate_slippage.

        Deprecated: Use calculate_slippage() instead.

        Args:
            price: Current price
            atr: Average True Range
            side: 'BUY' or 'SELL' (not used in new model)

        Returns:
            Estimated slippage in dollars per share
        """
        return self.calculate_slippage(price, atr)
    
    def _calculate_position_size(self, 
                                  price: float, 
                                  signal: Signal,
                                  available_capital: float) -> float:
        """Calculate position size based on signal and risk management."""
        if signal.position_size_pct:
            target_value = available_capital * signal.position_size_pct
        else:
            # Default to 10% of available capital
            target_value = available_capital * 0.10
        
        # Cap at available cash
        target_value = min(target_value, self._cash * 0.95)
        
        return target_value / price if price > 0 else 0
    
    def _get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            trade.quantity * prices.get(symbol, trade.entry_price)
            for symbol, trade in self._positions.items()
        )
        return self._cash + positions_value
    
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
        
        # Validate data has datetime indices
        for symbol, df in list(data.items()):
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
                else:
                    logger.warning(f"Symbol {symbol} has no datetime index or timestamp column")
                    continue
            
            # Remove timezone awareness for consistency
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
                data[symbol] = df
        
        run_id = str(uuid.uuid4())[:8]
        
        # Determine date range
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)
        
        if not all_dates:
            logger.error("No data provided")
            return BacktestResult(run_id=run_id, strategy=strategy.name,
                                  start_date="", end_date="")
        
        if start_date is None:
            start_date = all_dates[0]
        if end_date is None:
            end_date = all_dates[-1]
        
        # Filter dates
        dates = [d for d in all_dates if start_date <= d <= end_date]
        
        logger.debug(f"Backtesting {strategy.name} from {start_date} to {end_date} ({len(dates)} days)")
        
        # Pre-compute sorted date lists for binary search (O(log n) vs O(n))
        symbol_sorted_dates = {}
        symbol_date_idx = {}
        for symbol, df in data.items():
            sorted_dates = sorted(df.index.tolist())
            symbol_sorted_dates[symbol] = sorted_dates
            symbol_date_idx[symbol] = {d: i for i, d in enumerate(df.index)}
        
        # Main backtest loop
        for i, current_date in enumerate(dates):
            # Get data up to current date using binary search (O(log n) per symbol)
            current_data = {}
            current_prices = {}
            
            for symbol, df in data.items():
                sorted_dates = symbol_sorted_dates[symbol]
                idx_map = symbol_date_idx[symbol]
                # Binary search: find rightmost date <= current_date
                pos = bisect.bisect_right(sorted_dates, current_date)
                if pos > 0:
                    last_date = sorted_dates[pos - 1]
                    end_idx = idx_map[last_date] + 1
                    current_data[symbol] = df.iloc[:end_idx]
                    current_prices[symbol] = df['close'].iloc[end_idx - 1]
            
            # Determine VIX regime
            vix_regime = 'normal'
            if vix_data is not None and 'regime' in vix_data.columns:
                # Ensure both sides are comparable (handle timezone issues)
                try:
                    current_date_normalized = pd.Timestamp(current_date)
                    current_date_normalized = normalize_timestamp(current_date_normalized)
                    
                    # Check if current date is within VIX data range
                    vix_max_date = vix_data.index.max()
                    if current_date_normalized > vix_max_date:
                        # Beyond VIX data - use 'normal' as default
                        vix_regime = 'normal'
                    else:
                        vix_mask = vix_data.index <= current_date_normalized
                        if vix_mask.sum() > 0:
                            vix_regime = vix_data.loc[vix_mask, 'regime'].iloc[-1]
                except Exception as e:
                    logger.debug(f"VIX regime lookup failed: {e}")
                    vix_regime = 'normal'
            
            # Check existing positions for exits
            positions_to_close = []
            for symbol, trade in self._positions.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                df = current_data.get(symbol)
                
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
            
            # Close positions (stop-loss and target exits)
            for symbol, exit_price, reason in positions_to_close:
                trade = self._positions.pop(symbol)

                # Apply slippage on exit - always costs money to exit
                atr = current_data[symbol]['atr'].iloc[-1] if 'atr' in current_data[symbol].columns else exit_price * 0.02
                slippage = self.calculate_slippage(exit_price, atr)

                if trade.side == 'BUY':
                    exit_price -= slippage  # Receive less when selling
                else:
                    exit_price += slippage  # Pay more when covering short

                trade.slippage += slippage * trade.quantity
                trade.close(current_date, exit_price, reason)

                self._cash += trade.quantity * exit_price
                self._closed_trades.append(trade)
            
            # Generate new signals
            current_positions = list(self._positions.keys())
            signals = strategy.generate_signals(current_data, current_positions, vix_regime)
            
            # Process signals
            for signal in signals:
                if signal.signal_type == SignalType.CLOSE:
                    if signal.symbol in self._positions:
                        trade = self._positions.pop(signal.symbol)
                        exit_price = current_prices.get(signal.symbol, trade.entry_price)

                        # Apply slippage on signal-based exit
                        df = current_data.get(signal.symbol)
                        atr = df['atr'].iloc[-1] if df is not None and 'atr' in df.columns else exit_price * 0.02
                        slippage = self.calculate_slippage(exit_price, atr)

                        if trade.side == 'BUY':
                            exit_price -= slippage  # Receive less when selling
                        else:
                            exit_price += slippage  # Pay more when covering short

                        trade.slippage += slippage * trade.quantity
                        trade.close(current_date, exit_price, 'signal')
                        self._cash += trade.quantity * exit_price
                        self._closed_trades.append(trade)
                
                elif signal.signal_type == SignalType.BUY:
                    if signal.symbol in self._positions:
                        continue  # Already have position
                    
                    if signal.symbol not in current_prices:
                        continue
                    
                    price = current_prices[signal.symbol]
                    quantity = self._calculate_position_size(price, signal, self._cash)
                    
                    if quantity <= 0:
                        continue
                    
                    # Apply slippage on entry - always costs money to enter
                    df = current_data.get(signal.symbol)
                    atr = df['atr'].iloc[-1] if df is not None and 'atr' in df.columns else price * 0.02
                    slippage = self.calculate_slippage(price, atr)
                    entry_price = price + slippage  # Pay more when buying
                    
                    # Check if we have enough cash
                    cost = quantity * entry_price + self.commission_per_trade
                    if cost > self._cash:
                        quantity = (self._cash - self.commission_per_trade) / entry_price
                    
                    if quantity <= 0:
                        continue
                    
                    trade = SimulatedTrade(
                        symbol=signal.symbol,
                        side='BUY',
                        entry_date=current_date,
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
        
        # Close any remaining positions at end (with slippage)
        for symbol, trade in list(self._positions.items()):
            exit_price = current_prices.get(symbol, trade.entry_price)

            # Apply slippage on end-of-backtest exit
            df = current_data.get(symbol) if symbol in current_data else None
            atr = df['atr'].iloc[-1] if df is not None and 'atr' in df.columns else exit_price * 0.02
            slippage = self.calculate_slippage(exit_price, atr)

            if trade.side == 'BUY':
                exit_price -= slippage  # Receive less when selling
            else:
                exit_price += slippage  # Pay more when covering short

            trade.slippage += slippage * trade.quantity
            trade.close(end_date, exit_price, 'backtest_end')
            self._closed_trades.append(trade)
        
        # Calculate metrics
        return self._calculate_metrics(strategy.name, run_id, start_date, end_date)
    
    def _calculate_metrics(self, 
                           strategy_name: str,
                           run_id: str,
                           start_date: datetime,
                           end_date: datetime) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        
        result = BacktestResult(
            run_id=run_id,
            strategy=strategy_name,
            start_date=start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
            end_date=end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
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
            # Volatility (annualized)
            result.volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio (assuming 0% risk-free rate)
            mean_return = daily_returns.mean() * 252
            std_return = daily_returns.std() * np.sqrt(252)
            result.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            result.sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            
            # Max drawdown
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
        result.total_slippage_cost = result.total_slippage  # Alias for clarity

        # Calculate average slippage as percentage of trade value
        if result.total_trades > 0:
            total_trade_value = sum(
                t.entry_price * t.quantity for t in self._closed_trades
            )
            if total_trade_value > 0:
                result.avg_slippage_pct = (result.total_slippage / total_trade_value) * 100

        # Validation against research benchmarks
        if strategy_name in VALIDATION:
            min_sharpe = VALIDATION[strategy_name].get('min_sharpe', 0)
            research_sharpe = VALIDATION[strategy_name].get('research_sharpe', 0)
            
            result.meets_threshold = result.sharpe_ratio >= min_sharpe
            result.vs_research_pct = (result.sharpe_ratio / research_sharpe * 100) if research_sharpe > 0 else 0
        
        # Store trade details
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
    
    def run_walk_forward(self,
                         strategy: BaseStrategy,
                         data: Dict[str, pd.DataFrame],
                         train_days: int = 252,
                         test_days: int = 63,
                         step_days: int = 21) -> List[BacktestResult]:
        """
        Run walk-forward analysis with rolling windows.
        
        Args:
            strategy: Strategy to test
            data: Historical data
            train_days: Training window size
            test_days: Testing window size  
            step_days: Days to step forward each iteration
            
        Returns:
            List of BacktestResult for each test window
        """
        results = []

        # Get date range - only include datetime-like indices
        all_dates = set()
        for df in data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())
            elif hasattr(df.index[0], 'year'):  # Check if datetime-like
                all_dates.update(df.index.tolist())

        if not all_dates:
            logger.warning("No valid datetime indices found in data")
            return results

        all_dates = sorted(all_dates)

        if len(all_dates) < train_days + test_days:
            logger.warning("Insufficient data for walk-forward analysis")
            return results
        
        # Walk forward
        start_idx = train_days
        
        while start_idx + test_days <= len(all_dates):
            train_start = all_dates[start_idx - train_days]
            train_end = all_dates[start_idx - 1]
            test_start = all_dates[start_idx]
            test_end = all_dates[min(start_idx + test_days - 1, len(all_dates) - 1)]
            
            logger.info(f"Walk-forward: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
            
            # Run backtest on test period only
            # (In a full implementation, you'd optimize on train and test on test)
            result = self.run(strategy, data, test_start, test_end)
            results.append(result)
            
            start_idx += step_days
        
        return results


def run_backtest(strategy: BaseStrategy,
                 data: Dict[str, pd.DataFrame],
                 **kwargs) -> BacktestResult:
    """Convenience function to run a single backtest."""
    backtester = Backtester(**kwargs)
    return backtester.run(strategy, data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Backtesting Engine Test")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    
    def create_test_data(n_days: int = 504) -> Dict[str, pd.DataFrame]:
        """Create synthetic test data for multiple symbols."""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        data = {}
        
        for symbol in symbols:
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
            
            # Random walk with drift
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = 100 * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, n_days)
            }, index=dates)
            
            # Add ATR
            df['atr'] = (df['high'] - df['low']).rolling(14).mean()
            
            data[symbol] = df
        
        return data
    
    # Create test data
    test_data = create_test_data()
    
    # Create a simple test strategy
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    
    strategy = VolManagedMomentumStrategy()
    
    # Run backtest with different cost models to compare
    print("\nRunning backtests with different cost models...")
    print("=" * 60)

    for cost_model in ['aggressive', 'moderate', 'conservative']:
        backtester = Backtester(initial_capital=100000, cost_model=cost_model)
        result = backtester.run(strategy, test_data)

        print(f"\n{'-' * 40}")
        print(f"BACKTEST RESULTS - Cost Model: {cost_model.upper()}")
        print(f"{'-' * 40}")
        print(f"Strategy: {result.strategy}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"\nReturns:")
        print(f"  Total Return: {result.total_return:.2f}%")
        print(f"  Annual Return: {result.annual_return:.2f}%")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"  Volatility: {result.volatility:.2f}%")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Avg Trade P&L: ${result.avg_trade_pnl:.2f}")
        print(f"\nTransaction Costs:")
        print(f"  Total Slippage: ${result.total_slippage:.2f}")
        print(f"  Avg Slippage: {result.avg_slippage_pct:.3f}%")
        print(f"\nValidation:")
        print(f"  Meets Threshold: {result.meets_threshold}")
        print(f"  vs Research: {result.vs_research_pct:.1f}%")

    print("\n" + "=" * 60)
    print("NOTE: GA optimization should use 'conservative' cost model")
    print("to avoid overfitting to unrealistically low transaction costs.")
    print("=" * 60)
