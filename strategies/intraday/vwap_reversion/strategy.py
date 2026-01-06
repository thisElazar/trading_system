"""
VWAP Mean Reversion Strategy
============================
Trades price deviations from VWAP expecting reversion to the mean.

This strategy performs best in:
- Range-bound, choppy market conditions
- High-liquidity ETFs (SPY, QQQ)
- Non-news days

Avoid using on:
- Strong trend days (gap > 1%)
- Major news events (FOMC, CPI, etc.)
- Low volume days
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..base import IntradayStrategy, IntradayPosition, PositionSide
from .config import VWAPReversionConfig

logger = logging.getLogger(__name__)


class VWAPReversionStrategy(IntradayStrategy):
    """
    VWAP Mean Reversion Intraday Strategy.

    Enters positions when price deviates significantly from VWAP,
    expecting reversion back toward the volume-weighted average.
    """

    def __init__(
        self,
        config: VWAPReversionConfig = None,
        name: str = "vwap_reversion"
    ):
        self.strategy_config = config or VWAPReversionConfig()

        super().__init__(
            name=name,
            symbols=self.strategy_config.symbols,
            config=self.strategy_config.to_dict()
        )

        # Session data storage (per symbol)
        self.session_bars: Dict[str, List[dict]] = {}
        self.vwap_data: Dict[str, dict] = {}  # Cumulative VWAP tracking

        logger.info(f"VWAPReversionStrategy initialized: {self.strategy_config.symbols}")

    async def on_market_open(self) -> None:
        """Reset session data at market open."""
        self.session_bars = {symbol: [] for symbol in self.symbols}
        self.vwap_data = {
            symbol: {'cum_pv': 0.0, 'cum_vol': 0}
            for symbol in self.symbols
        }
        self.is_market_open = True
        logger.info(f"[{self.name}] Market open - session reset")

    async def on_market_close(self) -> None:
        """Force close any open positions at market close."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if position and position.current_price:
                self.close_position(
                    symbol=symbol,
                    exit_price=position.current_price,
                    reason="market_close"
                )
        self.is_market_open = False
        logger.info(f"[{self.name}] Market close - positions closed")

    async def on_bar(self, symbol: str, bar: Any) -> None:
        """
        Process each 1-minute bar.

        Args:
            symbol: Stock symbol
            bar: Bar data with OHLCV
        """
        if symbol not in self.symbols:
            return

        # Extract bar data
        bar_data = self._extract_bar_data(bar)
        if bar_data is None:
            return

        # Store bar
        if symbol not in self.session_bars:
            self.session_bars[symbol] = []
        self.session_bars[symbol].append(bar_data)

        # Update VWAP
        self._update_vwap(symbol, bar_data)

        # Check timing constraints
        minutes_in_session = len(self.session_bars[symbol])

        # Update existing position
        if self.has_position(symbol):
            await self._manage_position(symbol, bar_data, minutes_in_session)
            return

        # Check for new entry
        if self._can_enter(minutes_in_session):
            await self._check_entry(symbol, bar_data, minutes_in_session)

    def _extract_bar_data(self, bar: Any) -> Optional[dict]:
        """Extract standardized bar data."""
        try:
            if hasattr(bar, 'open'):
                return {
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'timestamp': bar.timestamp if hasattr(bar, 'timestamp') else datetime.now()
                }
            elif isinstance(bar, dict):
                return {
                    'open': float(bar.get('open', 0)),
                    'high': float(bar.get('high', 0)),
                    'low': float(bar.get('low', 0)),
                    'close': float(bar.get('close', 0)),
                    'volume': int(bar.get('volume', 0)),
                    'timestamp': bar.get('timestamp', datetime.now())
                }
        except Exception as e:
            logger.debug(f"Error extracting bar data: {e}")
        return None

    def _update_vwap(self, symbol: str, bar: dict) -> None:
        """Update cumulative VWAP calculation."""
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = {'cum_pv': 0.0, 'cum_vol': 0}

        typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
        self.vwap_data[symbol]['cum_pv'] += typical_price * bar['volume']
        self.vwap_data[symbol]['cum_vol'] += bar['volume']

    def _get_vwap(self, symbol: str) -> float:
        """Get current VWAP for symbol."""
        if symbol not in self.vwap_data:
            return 0.0

        data = self.vwap_data[symbol]
        if data['cum_vol'] == 0:
            return 0.0

        return data['cum_pv'] / data['cum_vol']

    def _get_vwap_deviation(self, symbol: str, price: float) -> float:
        """Calculate price deviation from VWAP as percentage."""
        vwap = self._get_vwap(symbol)
        if vwap == 0:
            return 0.0
        return (price - vwap) / vwap * 100

    def _calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate RSI from session bars."""
        bars = self.session_bars.get(symbol, [])
        if len(bars) < period + 1:
            return 50.0

        closes = pd.Series([b['close'] for b in bars])
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi) if not np.isnan(rsi) else 50.0

    def _calculate_relative_volume(self, symbol: str, period: int = 20) -> float:
        """Calculate relative volume vs recent average."""
        bars = self.session_bars.get(symbol, [])
        if len(bars) < period:
            return 1.0

        recent_vols = [b['volume'] for b in bars[-period:]]
        avg_vol = sum(recent_vols[:-1]) / max(len(recent_vols) - 1, 1)

        if avg_vol == 0:
            return 1.0

        return bars[-1]['volume'] / avg_vol

    def _can_enter(self, minutes_in_session: int) -> bool:
        """Check if we're in a valid trading window."""
        cfg = self.strategy_config
        return cfg.start_trading_minute <= minutes_in_session <= cfg.stop_trading_minute

    async def _check_entry(
        self,
        symbol: str,
        bar: dict,
        minutes_in_session: int
    ) -> None:
        """Check for entry conditions."""
        cfg = self.strategy_config
        price = bar['close']

        # Calculate indicators
        vwap_deviation = self._get_vwap_deviation(symbol, price)
        abs_deviation = abs(vwap_deviation)

        # Check deviation is in tradeable range
        if abs_deviation < cfg.min_vwap_deviation_pct:
            return
        if abs_deviation > cfg.max_vwap_deviation_pct:
            logger.debug(f"{symbol}: Deviation {vwap_deviation:.2f}% too large (trend day?)")
            return

        # Check volume
        rel_volume = self._calculate_relative_volume(symbol)
        if rel_volume < cfg.min_relative_volume:
            return

        # Determine direction
        rsi = self._calculate_rsi(symbol)

        if vwap_deviation < -cfg.min_vwap_deviation_pct and rsi < cfg.rsi_oversold:
            # Price below VWAP + RSI oversold -> LONG
            side = PositionSide.LONG
            entry_reason = f"vwap_oversold_{abs_deviation:.1f}pct"

        elif not cfg.long_only and vwap_deviation > cfg.min_vwap_deviation_pct and rsi > cfg.rsi_overbought:
            # Price above VWAP + RSI overbought -> SHORT
            side = PositionSide.SHORT
            entry_reason = f"vwap_overbought_{abs_deviation:.1f}pct"

        else:
            return  # No valid signal

        # Calculate position size
        position_value = cfg.max_position_value
        shares = int(position_value / price)
        if shares <= 0:
            return

        # Calculate stop loss
        vwap = self._get_vwap(symbol)
        if side == PositionSide.LONG:
            stop_loss = price * (1 - cfg.stop_loss_pct / 100)
            target_price = vwap  # Target VWAP
        else:
            stop_loss = price * (1 + cfg.stop_loss_pct / 100)
            target_price = vwap  # Target VWAP

        # Open position
        self.open_position(
            symbol=symbol,
            side=side,
            shares=shares,
            entry_price=price,
            stop_loss=stop_loss,
            target_price=target_price,
            max_hold_minutes=cfg.max_hold_minutes,
            reason=entry_reason,
            metadata={
                'entry_vwap': vwap,
                'entry_deviation': vwap_deviation,
                'entry_rsi': rsi,
                'entry_rel_volume': rel_volume,
            }
        )

    async def _manage_position(
        self,
        symbol: str,
        bar: dict,
        minutes_in_session: int
    ) -> None:
        """Manage open position."""
        cfg = self.strategy_config
        position = self.get_position(symbol)
        if position is None:
            return

        price = bar['close']
        position.update_price(price)

        # Check force exit before close
        if minutes_in_session >= cfg.force_exit_minute:
            self.close_position(symbol, price, "force_exit_eod")
            return

        # Check stop loss
        if position.check_stop_loss():
            self.close_position(symbol, price, "stop_loss")
            return

        # Check time limit
        if position.should_exit_by_time:
            self.close_position(symbol, price, "time_limit")
            return

        # Check VWAP reversion
        vwap = self._get_vwap(symbol)
        entry_deviation = position.metadata.get('entry_deviation', 0)

        if cfg.vwap_cross_exit:
            # Full reversion - price crossed VWAP
            if position.side == PositionSide.LONG and price >= vwap:
                self.close_position(symbol, price, "vwap_cross")
                return
            elif position.side == PositionSide.SHORT and price <= vwap:
                self.close_position(symbol, price, "vwap_cross")
                return

        # Partial reversion target
        current_deviation = self._get_vwap_deviation(symbol, price)
        reversion_pct = 1 - abs(current_deviation / entry_deviation) if entry_deviation != 0 else 0

        if reversion_pct >= cfg.reversion_target_pct:
            self.close_position(symbol, price, f"target_{reversion_pct*100:.0f}pct")
            return


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Test strategy initialization
    config = VWAPReversionConfig(
        symbols=['SPY', 'QQQ'],
        long_only=True
    )

    strategy = VWAPReversionStrategy(config=config)
    print(f"Strategy: {strategy.name}")
    print(f"Symbols: {strategy.symbols}")
    print(f"Config: {strategy.strategy_config.to_dict()}")
