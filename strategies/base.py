"""
Base Strategy
=============
Abstract base class for all trading strategies.
Defines the interface and common functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
from pathlib import Path

import pandas as pd

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRATEGIES, VALIDATION, get_transaction_cost

# Import canonical types from core
from core.types import Signal, Side, SignalStatus

# Backward compatibility alias - use Side instead in new code
SignalType = Side

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time_hours: float = 0.0
    
    def meets_threshold(self, strategy_name: str) -> bool:
        """Check if metrics meet validation thresholds."""
        if strategy_name not in VALIDATION:
            return True  # No threshold defined
        
        min_sharpe = VALIDATION[strategy_name].get('min_sharpe', 0)
        return self.sharpe_ratio >= min_sharpe
    
    def vs_research_pct(self, strategy_name: str) -> Optional[float]:
        """Calculate percentage of research benchmark achieved."""
        if strategy_name not in VALIDATION:
            return None
        
        research_sharpe = VALIDATION[strategy_name].get('research_sharpe')
        if not research_sharpe:
            return None
        
        return (self.sharpe_ratio / research_sharpe) * 100


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    To implement a strategy:
    1. Inherit from BaseStrategy
    2. Implement generate_signals()
    3. Optionally override should_close_position()
    """
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy identifier (must match key in config.STRATEGIES)
        """
        self.name = name
        self.config = STRATEGIES.get(name, {})
        self.metrics = StrategyMetrics()
        self._is_enabled = self.config.get('enabled', True)
        
        # Validation thresholds
        self.validation_config = VALIDATION.get(name, {})
        self.min_sharpe = self.validation_config.get('min_sharpe', 0)
        self.research_sharpe = self.validation_config.get('research_sharpe')
        
        logger.debug(f"Initialized strategy: {name} (enabled: {self._is_enabled})")
    
    @property
    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._is_enabled
    
    @is_enabled.setter
    def is_enabled(self, value: bool):
        """Enable or disable strategy."""
        self._is_enabled = value
        logger.info(f"Strategy {self.name} {'enabled' if value else 'disabled'}")
    
    @property
    def allocation_pct(self) -> float:
        """Get strategy's capital allocation percentage."""
        return self.config.get('allocation_pct', 0.1)
    
    @property
    def max_positions(self) -> int:
        """Get maximum concurrent positions for this strategy."""
        return self.config.get('max_positions', 5)
    
    @property
    def rebalance_frequency(self) -> str:
        """Get rebalance frequency."""
        return self.config.get('rebalance_frequency', 'daily')
    
    @abstractmethod
    def generate_signals(self, 
                         data: Dict[str, pd.DataFrame],
                         current_positions: List[str] = None,
                         vix_regime: str = None) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Dict mapping symbol to DataFrame with OHLCV + indicators
            current_positions: List of symbols currently held by this strategy
            vix_regime: Current VIX regime ('low', 'normal', 'high', 'extreme')
            
        Returns:
            List of Signal objects
        """
        pass
    
    def should_close_position(self,
                              symbol: str,
                              current_price: float,
                              entry_price: float,
                              stop_loss: float,
                              target_price: float,
                              peak_price: float,
                              entry_time: datetime,
                              data: pd.DataFrame = None) -> Optional[Signal]:
        """
        Check if an open position should be closed.
        
        Default implementation checks stop loss and target.
        Override for strategy-specific exit logic.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            entry_price: Entry price of position
            stop_loss: Stop loss price
            target_price: Target price
            peak_price: Highest price since entry
            entry_time: When position was opened
            data: Optional DataFrame with current market data
            
        Returns:
            Signal to close, or None to hold
        """
        # Check stop loss
        if current_price <= stop_loss:
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                reason="Stop loss hit"
            )
        
        # Check target
        if current_price >= target_price:
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                reason="Target reached"
            )
        
        # Check trailing stop (15% from peak)
        trailing_stop = peak_price * 0.85
        if current_price <= trailing_stop:
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                reason=f"Trailing stop hit (peak: ${peak_price:.2f})"
            )
        
        return None
    
    def filter_by_liquidity(self, 
                            df: pd.DataFrame, 
                            min_avg_volume: int = 500000,
                            min_price: float = 10,
                            max_price: float = 500) -> bool:
        """
        Check if a stock meets liquidity requirements.
        
        Args:
            df: DataFrame with OHLCV data
            min_avg_volume: Minimum 20-day average volume
            min_price: Minimum stock price
            max_price: Maximum stock price
            
        Returns:
            True if stock is liquid enough
        """
        if df.empty or len(df) < 20:
            return False
        
        avg_volume = df['volume'].tail(20).mean()
        current_price = df['close'].iloc[-1]
        
        return (avg_volume >= min_avg_volume and 
                min_price <= current_price <= max_price)
    
    def adjust_for_transaction_costs(self, 
                                     signal: Signal,
                                     market_cap: float) -> Signal:
        """
        Adjust signal strength for expected transaction costs.
        
        Research shows costs can reduce Sharpe by 3-5x in illiquid markets.
        
        Args:
            signal: Original signal
            market_cap: Estimated market cap for the symbol
            
        Returns:
            Signal with adjusted strength
        """
        cost_pct = get_transaction_cost(market_cap) * 100  # Convert to percentage
        
        # Reduce strength proportionally to expected cost
        # A 2% cost on a 0.8 strength signal becomes 0.8 * (1 - 0.02/0.05) = 0.48
        # This penalizes high-cost trades
        cost_penalty = min(cost_pct / 5, 1.0)  # Normalize to 5% max cost
        adjusted_strength = signal.strength * (1 - cost_penalty * 0.4)
        
        signal.strength = max(adjusted_strength, 0)
        signal.metadata['cost_penalty'] = cost_penalty
        signal.metadata['original_strength'] = signal.strength
        
        return signal
    
    def adjust_for_regime(self, signal: Signal, vix_regime: str) -> Signal:
        """
        Adjust signal based on VIX regime.
        
        Research shows:
        - Mean reversion strengthens 40-60% in high volatility
        - Momentum weakens in high volatility (crash risk)
        
        Args:
            signal: Original signal
            vix_regime: Current VIX regime
            
        Returns:
            Signal with regime-adjusted strength
        """
        # Default implementation - override in subclasses
        return signal
    
    def update_metrics(self, trade_result: dict):
        """
        Update strategy performance metrics after a trade closes.
        
        Args:
            trade_result: Dict with trade outcome (pnl, pnl_pct, etc.)
        """
        self.metrics.total_trades += 1
        
        pnl = trade_result.get('pnl', 0)
        if pnl > 0:
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
        
        self.metrics.total_pnl += pnl
        self.metrics.win_rate = (
            self.metrics.winning_trades / self.metrics.total_trades 
            if self.metrics.total_trades > 0 else 0
        )
        
        # Check if we should auto-disable
        if self.metrics.total_trades >= 20:
            if self.metrics.win_rate < 0.35:
                logger.warning(
                    f"Strategy {self.name} auto-disabled: "
                    f"win rate {self.metrics.win_rate:.1%} below 35% threshold"
                )
                self.is_enabled = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.is_enabled})"


class LongOnlyStrategy(BaseStrategy):
    """Base class for long-only strategies."""
    
    def should_close_position(self, symbol: str, current_price: float,
                              entry_price: float, stop_loss: float,
                              target_price: float, peak_price: float,
                              entry_time: datetime,
                              data: pd.DataFrame = None) -> Optional[Signal]:
        """Long-only exit logic."""
        # Use parent logic
        return super().should_close_position(
            symbol, current_price, entry_price, stop_loss,
            target_price, peak_price, entry_time, data
        )


class LongShortStrategy(BaseStrategy):
    """Base class for long-short strategies."""
    
    def should_close_position(self, symbol: str, current_price: float,
                              entry_price: float, stop_loss: float,
                              target_price: float, peak_price: float,
                              entry_time: datetime,
                              data: pd.DataFrame = None,
                              side: str = "LONG") -> Optional[Signal]:
        """Long-short exit logic."""
        if side == "LONG":
            # Stop loss: price drops below stop
            if current_price <= stop_loss:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=1.0,
                    price=current_price,
                    reason="Stop loss hit (LONG)"
                )
            # Target: price rises to target
            if current_price >= target_price:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=1.0,
                    price=current_price,
                    reason="Target reached (LONG)"
                )
        else:  # SHORT
            # Stop loss: price rises above stop
            if current_price >= stop_loss:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=1.0,
                    price=current_price,
                    reason="Stop loss hit (SHORT)"
                )
            # Target: price drops to target
            if current_price <= target_price:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=1.0,
                    price=current_price,
                    reason="Target reached (SHORT)"
                )
        
        return None


if __name__ == "__main__":
    # Test Signal creation
    print("Testing Signal dataclass...")
    
    signal = Signal(
        timestamp=datetime.now(),
        symbol="AAPL",
        strategy="test_strategy",
        signal_type=SignalType.BUY,
        strength=0.75,
        price=150.00,
        stop_loss=145.00,
        target_price=165.00,
        reason="Test signal"
    )
    
    print(f"Signal: {signal}")
    print(f"As dict: {signal.to_dict()}")
    
    # Test metrics
    print("\nTesting StrategyMetrics...")
    metrics = StrategyMetrics(
        total_trades=100,
        winning_trades=55,
        win_rate=0.55,
        sharpe_ratio=1.5
    )
    
    print(f"Metrics: {metrics}")
    print(f"Meets threshold for vol_managed_momentum: {metrics.meets_threshold('vol_managed_momentum')}")
    print(f"vs Research %: {metrics.vs_research_pct('vol_managed_momentum'):.1f}%")
