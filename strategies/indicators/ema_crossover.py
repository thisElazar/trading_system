"""
EMA Crossover Strategy
======================
Simple moving average crossover for single-symbol backtesting.

Signal Logic:
- BUY: Fast EMA crosses above Slow EMA (golden cross)
- SELL (close): Fast EMA crosses below Slow EMA (death cross)

Default Parameters:
- Fast EMA: 9 periods
- Slow EMA: 30 periods

This is a visualization/exploration strategy, not production.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy

    Generates buy signals when fast EMA crosses above slow EMA,
    and close signals when fast crosses below slow.

    Parameters can be tuned via constructor for exploration.
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 30,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10):
        super().__init__("ema_crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Position tracking (internal state for multi-day backtests)
        self._in_position = {}  # symbol -> bool

    def generate_signals(self, data: Dict[str, pd.DataFrame],
                        current_positions: List[str] = None,
                        vix_regime: str = None) -> List[Signal]:
        """
        Generate signals based on EMA crossover.

        Args:
            data: Dict mapping symbol to DataFrame with OHLCV data
            current_positions: List of symbols currently held
            vix_regime: Current VIX regime (unused by this strategy)
        """
        signals = []
        current_positions = current_positions or []

        for symbol, df in data.items():
            if df is None or len(df) < self.slow_period + 5:
                continue

            try:
                signal = self._check_symbol(symbol, df, current_positions)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")

        return signals

    def _check_symbol(self, symbol: str, df: pd.DataFrame,
                     current_positions: List[str]) -> Optional[Signal]:
        """Check a single symbol for crossover signals."""
        # Calculate EMAs
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # Check for crossover (need at least 2 rows to compare)
        if len(df) < 2:
            return None

        # Current and previous values
        curr_fast = df['ema_fast'].iloc[-1]
        curr_slow = df['ema_slow'].iloc[-1]
        prev_fast = df['ema_fast'].iloc[-2]
        prev_slow = df['ema_slow'].iloc[-2]

        current_price = df['close'].iloc[-1]
        current_date = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()

        # Check if we have a position (either from backtest state or internal tracking)
        in_position = symbol in current_positions or self._in_position.get(symbol, False)

        # Golden cross: fast crosses above slow -> BUY
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            if not in_position:
                self._in_position[symbol] = True
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=0.6,
                    price=current_price,
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    target_price=current_price * (1 + self.take_profit_pct),
                    reason='EMA golden cross',
                    metadata={
                        'fast_ema': curr_fast,
                        'slow_ema': curr_slow,
                    }
                )

        # Death cross: fast crosses below slow -> CLOSE
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            if in_position:
                self._in_position[symbol] = False
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=0.6,
                    price=current_price,
                    reason='EMA death cross',
                    metadata={
                        'fast_ema': curr_fast,
                        'slow_ema': curr_slow,
                    }
                )

        return None

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters for tuning."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
        }

    def set_params(self, **kwargs):
        """Update parameters."""
        if 'fast_period' in kwargs:
            self.fast_period = int(kwargs['fast_period'])
        if 'slow_period' in kwargs:
            self.slow_period = int(kwargs['slow_period'])
        if 'stop_loss_pct' in kwargs:
            self.stop_loss_pct = float(kwargs['stop_loss_pct'])
        if 'take_profit_pct' in kwargs:
            self.take_profit_pct = float(kwargs['take_profit_pct'])
