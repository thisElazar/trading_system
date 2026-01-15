"""
MACD Signal Strategy
====================
Trend-following based on MACD line and signal line crossovers.

Signal Logic:
- BUY: MACD line crosses above signal line (bullish crossover)
- CLOSE: MACD line crosses below signal line (bearish crossover)

Default Parameters:
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Signal EMA: 9 periods

This is a visualization/exploration strategy, not production.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MACDSignalStrategy(BaseStrategy):
    """
    MACD Signal Strategy

    Generates buy signals when MACD crosses above signal line,
    and close signals when MACD crosses below signal line.

    Parameters can be tuned via constructor for exploration.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10):
        super().__init__("macd_signal")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Position tracking (internal state for multi-day backtests)
        self._in_position = {}  # symbol -> bool

    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator components."""
        ema_fast = prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signals(self, data: Dict[str, pd.DataFrame],
                        current_positions: List[str] = None,
                        vix_regime: str = None) -> List[Signal]:
        """
        Generate signals based on MACD crossovers.

        Args:
            data: Dict mapping symbol to DataFrame with OHLCV data
            current_positions: List of symbols currently held
            vix_regime: Current VIX regime (unused by this strategy)
        """
        signals = []
        current_positions = current_positions or []

        for symbol, df in data.items():
            if df is None or len(df) < self.slow_period + self.signal_period + 5:
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
        """Check a single symbol for MACD crossover signals."""
        df = df.copy()
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])

        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # Need valid values
        if df['macd'].isna().iloc[-1] or df['macd_signal'].isna().iloc[-1]:
            return None

        # Current and previous values
        curr_macd = df['macd'].iloc[-1]
        curr_signal = df['macd_signal'].iloc[-1]
        prev_macd = df['macd'].iloc[-2]
        prev_signal = df['macd_signal'].iloc[-2]
        curr_hist = df['macd_hist'].iloc[-1]

        current_price = df['close'].iloc[-1]
        current_date = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()

        # Check if we have a position (either from backtest state or internal tracking)
        in_position = symbol in current_positions or self._in_position.get(symbol, False)

        # Bullish crossover: MACD crosses above signal line -> BUY
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            if not in_position:
                self._in_position[symbol] = True
                # Strength based on histogram magnitude
                strength = min(0.8, 0.5 + abs(curr_hist) / current_price * 10)
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    target_price=current_price * (1 + self.take_profit_pct),
                    reason='MACD bullish crossover',
                    metadata={
                        'macd': curr_macd,
                        'signal': curr_signal,
                        'histogram': curr_hist,
                    }
                )

        # Bearish crossover: MACD crosses below signal line -> CLOSE
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            if in_position:
                self._in_position[symbol] = False
                strength = min(0.8, 0.5 + abs(curr_hist) / current_price * 10)
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=strength,
                    price=current_price,
                    reason='MACD bearish crossover',
                    metadata={
                        'macd': curr_macd,
                        'signal': curr_signal,
                        'histogram': curr_hist,
                    }
                )

        return None

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters for tuning."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
        }

    def set_params(self, **kwargs):
        """Update parameters."""
        if 'fast_period' in kwargs:
            self.fast_period = int(kwargs['fast_period'])
        if 'slow_period' in kwargs:
            self.slow_period = int(kwargs['slow_period'])
        if 'signal_period' in kwargs:
            self.signal_period = int(kwargs['signal_period'])
        if 'stop_loss_pct' in kwargs:
            self.stop_loss_pct = float(kwargs['stop_loss_pct'])
        if 'take_profit_pct' in kwargs:
            self.take_profit_pct = float(kwargs['take_profit_pct'])
