"""
Gap Fill Intraday Strategy
==========================
Intraday strategy that fades overnight gaps, expecting them to fill.

Research Basis:
- Gaps in the 0.15% - 0.60% range fill 59-61% of the time
- Most gap fills occur within the first 2 hours of trading
- Strategy: Enter position fading the gap, exit when filled or time limit

Trading Logic:
1. Detect gaps at market open (9:30-9:35 AM)
2. Enter position AGAINST the gap direction (fade the gap)
   - Gap UP -> SHORT (expect price to come back down)
   - Gap DOWN -> LONG (expect price to come back up)
3. Exit when:
   - Gap fills 75% (primary exit)
   - 120 minutes elapsed (time stop)
   - Stop loss hit (risk management)
   - Market close approaching
"""

from datetime import datetime, time, timedelta
from typing import Dict, Optional, Any, Tuple
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategies.intraday.base import IntradayStrategy, IntradayPosition, PositionSide
from strategies.intraday.gap_fill.detector import Gap, GapDetector
from strategies.intraday.gap_fill.config import GapFillConfig, DEFAULT_CONFIG
from data.fetchers.intraday_bars import IntradayDataManager

logger = logging.getLogger(__name__)


class GapFillStrategy(IntradayStrategy):
    """
    Gap-fill strategy that:
    1. Detects gaps at market open (9:30-9:35am)
    2. Enters positions fading the gap
    3. Exits when gap fills 75% OR after 120 minutes

    Based on research: gaps 0.15%-0.60% fill 59-61% of the time.

    This strategy FADES the gap (trades against it):
    - Gap UP -> SHORT (expect price to fall back)
    - Gap DOWN -> LONG (expect price to rise back)

    Unlike momentum strategies that trade WITH the gap, gap-fill strategies
    exploit the mean-reversion tendency of gaps within the tradeable range.
    """

    def __init__(self, config: GapFillConfig = None):
        """
        Initialize the gap-fill strategy.

        Args:
            config: Strategy configuration (uses defaults if not provided)
        """
        self.strategy_config = config or DEFAULT_CONFIG

        super().__init__(
            name="gap_fill_intraday",
            symbols=self.strategy_config.symbols,
            config=self.strategy_config.to_dict()
        )

        # Gap detection
        self.detector = GapDetector(
            min_gap_pct=self.strategy_config.min_gap_pct,
            max_gap_pct=self.strategy_config.max_gap_pct
        )

        # Data manager for getting previous close
        self.data_manager = IntradayDataManager()

        # Gap tracking
        self.detected_gaps: Dict[str, Gap] = {}
        self.gaps_checked_today: set = set()

        # Previous close cache (updated at market open)
        self._previous_closes: Dict[str, float] = {}

        logger.info(f"Gap Fill Strategy initialized with config: {self.strategy_config.to_dict()}")

    # =========================================================================
    # Core streaming callbacks
    # =========================================================================

    async def on_bar(self, symbol: str, bar: Any) -> None:
        """
        Process each 1-minute bar.

        This is the main callback from the stream handler.
        Handles gap detection and position monitoring.

        Args:
            symbol: Stock symbol
            bar: Bar data with open, high, low, close, volume, timestamp
        """
        # Extract bar data (handle both dict and object)
        if isinstance(bar, dict):
            bar_open = bar.get('open')
            bar_high = bar.get('high')
            bar_low = bar.get('low')
            bar_close = bar.get('close')
            bar_time = bar.get('timestamp', datetime.now())
        else:
            bar_open = getattr(bar, 'open', None)
            bar_high = getattr(bar, 'high', None)
            bar_low = getattr(bar, 'low', None)
            bar_close = getattr(bar, 'close', None)
            bar_time = getattr(bar, 'timestamp', datetime.now())

        if bar_close is None:
            logger.warning(f"Invalid bar data for {symbol}")
            return

        current_price = bar_close
        minutes_since_open = self.minutes_since_open()

        # Phase 1: Gap Detection (first 5 minutes)
        if (self.strategy_config.gap_detection_start_minute <= minutes_since_open
                <= self.strategy_config.gap_detection_end_minute):
            if symbol not in self.gaps_checked_today:
                gap = await self._check_for_gap(symbol, bar_open, bar_time)
                self.gaps_checked_today.add(symbol)

                if gap:
                    # Enter position immediately
                    await self._enter_position(symbol, gap, current_price)

        # Phase 2: Position Monitoring
        if self.has_position(symbol):
            # Update position price
            self.update_position_price(symbol, current_price, bar_time)

            # Check exit conditions
            should_exit, exit_reason = await self._check_exit_conditions(
                symbol, current_price
            )

            if should_exit:
                self.close_position(symbol, current_price, exit_reason)

    async def on_market_open(self) -> None:
        """
        Called at market open. Reset daily state and cache previous closes.
        """
        self.is_market_open = True
        self.detected_gaps.clear()
        self.gaps_checked_today.clear()

        # Cache previous closes for all symbols
        await self._cache_previous_closes()

        logger.info(
            f"[{self.name}] Market open - tracking {len(self.symbols)} symbols"
        )

    async def on_market_close(self) -> None:
        """
        Called at market close. Close all remaining positions.
        """
        # Close any remaining positions
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if position:
                logger.warning(
                    f"[{self.name}] Closing {symbol} at market close"
                )
                self.close_position(
                    symbol,
                    position.current_price or position.entry_price,
                    "market_close"
                )

        self.is_market_open = False
        logger.info(
            f"[{self.name}] Market closed - "
            f"Daily P&L: ${self.daily_pnl:+.2f}"
        )

    # =========================================================================
    # Gap detection and entry
    # =========================================================================

    async def _check_for_gap(
        self,
        symbol: str,
        open_price: float,
        timestamp: datetime
    ) -> Optional[Gap]:
        """
        Check for a tradeable gap at market open.

        Args:
            symbol: Stock symbol
            open_price: Today's opening price
            timestamp: Current timestamp

        Returns:
            Gap object if tradeable, None otherwise
        """
        previous_close = self._previous_closes.get(symbol)

        if previous_close is None:
            logger.warning(f"No previous close available for {symbol}")
            return None

        gap = self.detector.detect_gap(
            symbol=symbol,
            previous_close=previous_close,
            open_price=open_price,
            timestamp=timestamp
        )

        if gap:
            self.detected_gaps[symbol] = gap
            logger.info(
                f"[{self.name}] Detected gap: {gap.symbol} "
                f"{gap.gap_direction} {abs(gap.gap_pct):.2f}%"
            )

        return gap

    async def _enter_position(
        self,
        symbol: str,
        gap: Gap,
        entry_price: float
    ) -> bool:
        """
        Enter a gap-fill position (fade the gap).

        Gap-fill strategy FADES the gap:
        - Gap UP -> SHORT (expect price to fall)
        - Gap DOWN -> LONG (expect price to rise)

        Args:
            symbol: Stock symbol
            gap: Detected gap
            entry_price: Entry price

        Returns:
            True if position was opened
        """
        # Check max positions
        if len(self.positions) >= self.strategy_config.max_positions:
            logger.info(f"Max positions reached, skipping {symbol}")
            return False

        # Check for existing position
        if self.has_position(symbol):
            logger.debug(f"Already have position in {symbol}")
            return False

        # Determine side - FADE the gap
        if gap.gap_direction == 'up':
            # Gap up: expect price to fall back -> SHORT
            side = PositionSide.SHORT
            stop_loss = entry_price * (1 + self.strategy_config.stop_loss_pct / 100)
        else:
            # Gap down: expect price to rise back -> LONG
            side = PositionSide.LONG
            stop_loss = entry_price * (1 - self.strategy_config.stop_loss_pct / 100)

        # Check long_only restriction
        if self.strategy_config.long_only and side == PositionSide.SHORT:
            logger.debug(f"Skipping {symbol} - long_only mode and gap up")
            return False

        # Calculate position size
        shares = self._calculate_position_size(symbol, entry_price)
        if shares == 0:
            logger.warning(f"Position size is 0 for {symbol}")
            return False

        # Calculate target (75% fill by default)
        target_price = gap.get_fill_target_price(self.strategy_config.fill_threshold)

        # Open position
        self.open_position(
            symbol=symbol,
            side=side,
            shares=shares,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            max_hold_minutes=self.strategy_config.max_hold_minutes,
            reason=f"gap_fill_{gap.gap_direction}",
            metadata={
                'gap': gap.to_dict(),
                'fill_threshold': self.strategy_config.fill_threshold
            }
        )

        return True

    # =========================================================================
    # Exit logic
    # =========================================================================

    async def _check_exit_conditions(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Exit conditions (in priority order):
        1. Stop loss hit
        2. Gap filled to threshold
        3. Max hold time exceeded
        4. Approaching market close

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            Tuple of (should_exit, reason)
        """
        position = self.get_position(symbol)
        if position is None:
            return False, ""

        gap = self.detected_gaps.get(symbol)

        # 1. Check stop loss
        if position.check_stop_loss():
            return True, "stop_loss"

        # 2. Check gap fill (if we have gap data)
        if gap:
            fill_pct = gap.fill_percentage(current_price)
            if fill_pct >= self.strategy_config.fill_threshold:
                return True, f"gap_filled_{fill_pct:.0%}"

        # 3. Check target price (alternative to gap fill check)
        if position.check_target():
            return True, "target_reached"

        # 4. Check time limit
        if position.should_exit_by_time:
            return True, "time_limit"

        # 5. Check approaching market close (exit 15 min before close)
        if self.minutes_until_close() <= 15:
            return True, "end_of_day"

        return False, ""

    # =========================================================================
    # Position sizing
    # =========================================================================

    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float
    ) -> int:
        """
        Calculate number of shares based on risk parameters.

        Uses max_position_pct of portfolio value.

        Args:
            symbol: Stock symbol
            entry_price: Expected entry price

        Returns:
            Number of shares to trade
        """
        max_position_value = self.strategy_config.max_position_value

        # Calculate shares (round down to avoid exceeding limit)
        shares = int(max_position_value / entry_price)

        return shares

    # =========================================================================
    # Helper methods
    # =========================================================================

    async def _cache_previous_closes(self) -> None:
        """
        Cache previous day's closing prices for all symbols.

        Called at market open to have previous close ready for gap detection.
        """
        today = datetime.now()

        for symbol in self.symbols:
            try:
                prev_close = get_previous_close(
                    self.data_manager,
                    symbol,
                    today
                )
                if prev_close:
                    self._previous_closes[symbol] = prev_close
                    logger.debug(f"Cached {symbol} prev close: ${prev_close:.2f}")
            except Exception as e:
                logger.error(f"Error getting previous close for {symbol}: {e}")

        logger.info(
            f"Cached previous closes for {len(self._previous_closes)}/{len(self.symbols)} symbols"
        )

    def get_gap_status(self) -> Dict[str, Any]:
        """
        Get current status of detected gaps and positions.

        Returns:
            Status dictionary with gap and position details
        """
        return {
            'detected_gaps': {
                symbol: gap.to_dict()
                for symbol, gap in self.detected_gaps.items()
            },
            'gaps_checked': list(self.gaps_checked_today),
            'positions': {
                symbol: {
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': pos.unrealized_pnl_pct,
                    'hold_minutes': pos.hold_time_minutes
                }
                for symbol, pos in self.positions.items()
            }
        }


# =============================================================================
# Helper functions
# =============================================================================

def get_previous_close(
    data_manager: IntradayDataManager,
    symbol: str,
    date: datetime
) -> Optional[float]:
    """
    Get previous trading day's closing price.

    Helper function that wraps the data manager's get_previous_close method.

    Args:
        data_manager: IntradayDataManager instance
        symbol: Stock symbol
        date: Current date

    Returns:
        Previous close price or None
    """
    return data_manager.get_previous_close(symbol, date)


def create_gap_fill_strategy(
    symbols: list = None,
    min_gap_pct: float = 0.15,
    max_gap_pct: float = 0.60,
    max_hold_minutes: int = 120,
    portfolio_value: float = 97000.0
) -> GapFillStrategy:
    """
    Factory function to create a configured gap-fill strategy.

    Args:
        symbols: List of symbols to trade
        min_gap_pct: Minimum gap size to trade
        max_gap_pct: Maximum gap size to trade
        max_hold_minutes: Maximum holding period
        portfolio_value: Total portfolio value

    Returns:
        Configured GapFillStrategy instance
    """
    config = GapFillConfig(
        symbols=symbols or ['SPY', 'QQQ'],
        min_gap_pct=min_gap_pct,
        max_gap_pct=max_gap_pct,
        max_hold_minutes=max_hold_minutes,
        portfolio_value=portfolio_value
    )

    return GapFillStrategy(config=config)


if __name__ == "__main__":
    # Test strategy initialization
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test_strategy():
        # Create strategy with default config
        strategy = GapFillStrategy()

        print("Gap Fill Strategy Test")
        print("=" * 50)
        print(f"Name: {strategy.name}")
        print(f"Symbols: {strategy.symbols}")
        print(f"Config: {strategy.config}")

        # Simulate market open
        await strategy.start_trading_day()
        await strategy.on_market_open()

        print(f"\nStatus after market open:")
        print(f"  Is market open: {strategy.is_market_open}")
        print(f"  Previous closes cached: {len(strategy._previous_closes)}")

        # Simulate a bar with a gap
        test_bar = {
            'open': 451.00,  # Gap up from 450.00
            'high': 451.50,
            'low': 450.50,
            'close': 450.75,
            'volume': 1000000,
            'timestamp': datetime.now()
        }

        # Manually set previous close for testing
        strategy._previous_closes['SPY'] = 450.00

        # Process bar
        await strategy.on_bar('SPY', test_bar)

        print(f"\nGap status:")
        status = strategy.get_gap_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Simulate market close
        await strategy.on_market_close()
        await strategy.end_trading_day()

        print(f"\nFinal daily P&L: ${strategy.daily_pnl:+.2f}")

    asyncio.run(test_strategy())
