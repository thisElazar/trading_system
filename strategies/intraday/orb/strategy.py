"""
Opening Range Breakout (ORB) Strategy
=====================================
Trades breakouts from the first N minutes' high/low range.

This strategy performs best on:
- Trend days (strong directional moves)
- Days with clear catalyst or momentum
- High volume sessions

Avoid using on:
- Choppy, range-bound days
- Very large gap days (range already extended)
- Low volume, holiday sessions
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..base import IntradayStrategy, IntradayPosition, PositionSide
from .config import ORBConfig

logger = logging.getLogger(__name__)


class ORBStrategy(IntradayStrategy):
    """
    Opening Range Breakout Intraday Strategy.

    Enters positions when price breaks out of the opening range,
    expecting momentum continuation in the breakout direction.
    """

    def __init__(
        self,
        config: ORBConfig = None,
        name: str = "orb"
    ):
        self.strategy_config = config or ORBConfig()

        super().__init__(
            name=name,
            symbols=self.strategy_config.symbols,
            config=self.strategy_config.to_dict()
        )

        # Session data storage (per symbol)
        self.session_bars: Dict[str, List[dict]] = {}

        # Opening range tracking
        self.opening_range: Dict[str, dict] = {}  # {symbol: {high, low, formed, traded}}

        # VWAP tracking
        self.vwap_data: Dict[str, dict] = {}

        logger.info(f"ORBStrategy initialized: {self.strategy_config.symbols}")

    async def on_market_open(self) -> None:
        """Reset session data at market open."""
        self.session_bars = {symbol: [] for symbol in self.symbols}
        self.opening_range = {
            symbol: {'high': 0.0, 'low': float('inf'), 'formed': False, 'traded': False}
            for symbol in self.symbols
        }
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

        minutes_in_session = len(self.session_bars[symbol])

        # Update VWAP
        self._update_vwap(symbol, bar_data)

        # Build opening range during range period
        if minutes_in_session <= self.strategy_config.range_minutes:
            self._update_opening_range(symbol, bar_data)
            if minutes_in_session == self.strategy_config.range_minutes:
                self._finalize_opening_range(symbol)
            return

        # Update existing position
        if self.has_position(symbol):
            await self._manage_position(symbol, bar_data, minutes_in_session)
            return

        # Check for new entry (only if range is valid and not yet traded)
        or_data = self.opening_range.get(symbol, {})
        if or_data.get('formed') and not or_data.get('traded'):
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

    def _update_opening_range(self, symbol: str, bar: dict) -> None:
        """Update opening range high/low."""
        if symbol not in self.opening_range:
            self.opening_range[symbol] = {
                'high': 0.0, 'low': float('inf'), 'formed': False, 'traded': False
            }

        or_data = self.opening_range[symbol]
        or_data['high'] = max(or_data['high'], bar['high'])
        or_data['low'] = min(or_data['low'], bar['low'])

    def _finalize_opening_range(self, symbol: str) -> None:
        """Finalize opening range and check if valid for trading."""
        cfg = self.strategy_config
        or_data = self.opening_range[symbol]

        or_high = or_data['high']
        or_low = or_data['low']
        range_pct = (or_high - or_low) / or_low * 100 if or_low > 0 else 0

        # Check if range is valid
        if range_pct < cfg.min_range_pct:
            logger.info(f"{symbol}: Opening range {range_pct:.2f}% too small")
            or_data['formed'] = False
            return

        if range_pct > cfg.max_range_pct:
            logger.info(f"{symbol}: Opening range {range_pct:.2f}% too large (gap day?)")
            or_data['formed'] = False
            return

        or_data['formed'] = True
        or_data['range_pct'] = range_pct
        or_data['range_dollars'] = or_high - or_low

        logger.info(
            f"{symbol}: Opening range formed - "
            f"High: ${or_high:.2f}, Low: ${or_low:.2f}, Range: {range_pct:.2f}%"
        )

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
        """Check if we're in a valid entry window."""
        cfg = self.strategy_config
        return cfg.entry_start_minute <= minutes_in_session <= cfg.entry_end_minute

    async def _check_entry(
        self,
        symbol: str,
        bar: dict,
        minutes_in_session: int
    ) -> None:
        """Check for breakout entry conditions."""
        cfg = self.strategy_config
        or_data = self.opening_range[symbol]
        price = bar['close']

        or_high = or_data['high']
        or_low = or_data['low']
        range_dollars = or_data.get('range_dollars', or_high - or_low)

        # Calculate breakout thresholds with buffer
        breakout_up = or_high * (1 + cfg.breakout_buffer_pct / 100)
        breakout_down = or_low * (1 - cfg.breakout_buffer_pct / 100)

        # Check volume confirmation
        rel_volume = self._calculate_relative_volume(symbol)
        if rel_volume < cfg.min_relative_volume:
            return

        # Check VWAP alignment if required
        vwap = self._get_vwap(symbol)

        # Check for breakout
        if price > breakout_up:
            # Upside breakout
            if cfg.require_vwap_alignment and price < vwap:
                logger.debug(f"{symbol}: Breakout up but below VWAP - skip")
                return

            side = PositionSide.LONG
            entry_reason = f"orb_breakout_up"
            stop_loss = or_low * (1 - cfg.stop_loss_buffer_pct / 100)
            target_price = price + (range_dollars * cfg.target_multiple)

        elif not cfg.long_only and price < breakout_down:
            # Downside breakout
            if cfg.require_vwap_alignment and price > vwap:
                logger.debug(f"{symbol}: Breakout down but above VWAP - skip")
                return

            side = PositionSide.SHORT
            entry_reason = f"orb_breakout_down"
            stop_loss = or_high * (1 + cfg.stop_loss_buffer_pct / 100)
            target_price = price - (range_dollars * cfg.target_multiple)

        else:
            return  # No breakout

        # Calculate position size
        position_value = cfg.max_position_value
        shares = int(position_value / price)
        if shares <= 0:
            return

        # Mark as traded (only one trade per symbol per day)
        or_data['traded'] = True

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
                'or_high': or_high,
                'or_low': or_low,
                'or_range_pct': or_data.get('range_pct', 0),
                'entry_rel_volume': rel_volume,
                'entry_vwap': vwap,
                'initial_target': target_price,
                'trailing_active': False,
            }
        )

    async def _manage_position(
        self,
        symbol: str,
        bar: dict,
        minutes_in_session: int
    ) -> None:
        """Manage open position with trailing stop."""
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

        # Check target
        if position.check_target():
            self.close_position(symbol, price, "target_reached")
            return

        # Implement trailing stop after initial profit
        if cfg.use_trailing_stop and not position.metadata.get('trailing_active'):
            or_range = position.metadata.get('or_high', 0) - position.metadata.get('or_low', 0)

            if position.side == PositionSide.LONG:
                profit = price - position.entry_price
                if profit >= or_range:  # 1x range profit
                    # Activate trailing stop
                    trail_distance = or_range * cfg.trailing_stop_pct
                    new_stop = price - trail_distance
                    position.stop_loss = max(position.stop_loss, new_stop)
                    position.metadata['trailing_active'] = True
                    logger.info(f"{symbol}: Trailing stop activated at ${new_stop:.2f}")
            else:  # SHORT
                profit = position.entry_price - price
                if profit >= or_range:
                    trail_distance = or_range * cfg.trailing_stop_pct
                    new_stop = price + trail_distance
                    position.stop_loss = min(position.stop_loss, new_stop)
                    position.metadata['trailing_active'] = True
                    logger.info(f"{symbol}: Trailing stop activated at ${new_stop:.2f}")

        # Update trailing stop if active
        elif position.metadata.get('trailing_active'):
            or_range = position.metadata.get('or_high', 0) - position.metadata.get('or_low', 0)
            trail_distance = or_range * cfg.trailing_stop_pct

            if position.side == PositionSide.LONG:
                new_stop = price - trail_distance
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:  # SHORT
                new_stop = price + trail_distance
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Test strategy initialization
    config = ORBConfig(
        symbols=['SPY', 'QQQ'],
        range_minutes=30,
        long_only=True
    )

    strategy = ORBStrategy(config=config)
    print(f"Strategy: {strategy.name}")
    print(f"Symbols: {strategy.symbols}")
    print(f"Config: {strategy.strategy_config.to_dict()}")
