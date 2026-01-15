"""
RSI Reversal Strategy
=====================
Mean reversion based on RSI overbought/oversold levels.

Signal Logic:
- BUY: RSI drops below oversold level (default 30)
- CLOSE: RSI rises above overbought level (default 70)

Default Parameters:
- RSI Period: 14
- Oversold: 30
- Overbought: 70

This is a visualization/exploration strategy, not production.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class RSIReversalStrategy(BaseStrategy):
    """
    RSI Reversal Strategy

    Buys when RSI indicates oversold conditions,
    closes when RSI indicates overbought conditions.

    Parameters can be tuned via constructor for exploration.
    """

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10):
        super().__init__("rsi_reversal")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Position tracking (internal state for multi-day backtests)
        self._in_position = {}  # symbol -> bool

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: Dict[str, pd.DataFrame],
                        current_positions: List[str] = None,
                        vix_regime: str = None) -> List[Signal]:
        """
        Generate signals based on RSI levels.

        Args:
            data: Dict mapping symbol to DataFrame with OHLCV data
            current_positions: List of symbols currently held
            vix_regime: Current VIX regime (unused by this strategy)
        """
        signals = []
        current_positions = current_positions or []

        for symbol, df in data.items():
            if df is None or len(df) < self.rsi_period + 5:
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
        """Check a single symbol for RSI signals."""
        df = df.copy()
        df['rsi'] = self._calculate_rsi(df['close'])

        # Need valid RSI values
        if df['rsi'].isna().iloc[-1]:
            return None

        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) > 1 else current_rsi
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()

        # Check if we have a position (either from backtest state or internal tracking)
        in_position = symbol in current_positions or self._in_position.get(symbol, False)

        # Buy signal: RSI crosses below oversold -> BUY
        if prev_rsi >= self.oversold and current_rsi < self.oversold:
            if not in_position:
                self._in_position[symbol] = True
                # Strength based on how oversold (deeper = stronger signal)
                strength = min(0.9, 0.5 + (self.oversold - current_rsi) / 100)
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    target_price=current_price * (1 + self.take_profit_pct),
                    reason=f'RSI oversold ({current_rsi:.1f})',
                    metadata={
                        'rsi': current_rsi,
                    }
                )

        # Exit signal: RSI crosses above overbought -> CLOSE
        elif prev_rsi <= self.overbought and current_rsi > self.overbought:
            if in_position:
                self._in_position[symbol] = False
                strength = min(0.9, 0.5 + (current_rsi - self.overbought) / 100)
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=strength,
                    price=current_price,
                    reason=f'RSI overbought ({current_rsi:.1f})',
                    metadata={
                        'rsi': current_rsi,
                    }
                )

        return None

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters for tuning."""
        return {
            'rsi_period': self.rsi_period,
            'oversold': self.oversold,
            'overbought': self.overbought,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
        }

    def set_params(self, **kwargs):
        """Update parameters."""
        if 'rsi_period' in kwargs:
            self.rsi_period = int(kwargs['rsi_period'])
        if 'oversold' in kwargs:
            self.oversold = float(kwargs['oversold'])
        if 'overbought' in kwargs:
            self.overbought = float(kwargs['overbought'])
        if 'stop_loss_pct' in kwargs:
            self.stop_loss_pct = float(kwargs['stop_loss_pct'])
        if 'take_profit_pct' in kwargs:
            self.take_profit_pct = float(kwargs['take_profit_pct'])
