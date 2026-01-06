"""
Intraday Strategy Base
======================
Abstract base class for intraday trading strategies.
Provides common functionality for real-time streaming strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
from pathlib import Path

import pandas as pd

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


# Market hours (US Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class PositionSide(Enum):
    """Position direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class IntradayTimeWindow(Enum):
    """Common intraday trading time windows."""
    OPEN_CROSS = "open_cross"      # 9:30-9:35 - first 5 minutes
    MORNING = "morning"             # 9:30-12:00
    MIDDAY = "midday"               # 12:00-14:00
    AFTERNOON = "afternoon"         # 14:00-16:00
    POWER_HOUR = "power_hour"       # 15:00-16:00


@dataclass
class IntradayPosition:
    """
    Represents an active intraday position.

    Tracks all relevant data for position management including
    entry details, risk parameters, and current state.
    """
    symbol: str
    side: PositionSide
    shares: int
    entry_price: float
    entry_time: datetime

    # Risk management
    stop_loss: float
    target_price: Optional[float] = None
    max_hold_minutes: Optional[int] = None

    # State tracking
    peak_price: Optional[float] = None
    trough_price: Optional[float] = None
    current_price: Optional[float] = None
    last_update: Optional[datetime] = None

    # Metadata
    strategy_name: str = ""
    entry_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize tracking prices."""
        if self.peak_price is None:
            self.peak_price = self.entry_price
        if self.trough_price is None:
            self.trough_price = self.entry_price
        if self.current_price is None:
            self.current_price = self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate current unrealized P&L in dollars."""
        if self.current_price is None:
            return 0.0

        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.shares
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate current unrealized P&L as percentage."""
        if self.side == PositionSide.LONG:
            return ((self.current_price / self.entry_price) - 1) * 100
        else:  # SHORT
            return ((self.entry_price / self.current_price) - 1) * 100

    @property
    def hold_time_minutes(self) -> float:
        """Calculate how long position has been held in minutes."""
        now = datetime.now()
        return (now - self.entry_time).total_seconds() / 60

    @property
    def should_exit_by_time(self) -> bool:
        """Check if position has exceeded max hold time."""
        if self.max_hold_minutes is None:
            return False
        return self.hold_time_minutes >= self.max_hold_minutes

    def update_price(self, price: float, timestamp: datetime = None):
        """Update current price and tracking levels."""
        self.current_price = price
        self.last_update = timestamp or datetime.now()

        if price > self.peak_price:
            self.peak_price = price
        if price < self.trough_price:
            self.trough_price = price

    def check_stop_loss(self) -> bool:
        """Check if stop loss has been hit."""
        if self.current_price is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss

    def check_target(self) -> bool:
        """Check if target has been reached."""
        if self.target_price is None or self.current_price is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price >= self.target_price
        else:  # SHORT
            return self.current_price <= self.target_price

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'max_hold_minutes': self.max_hold_minutes,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'hold_time_minutes': self.hold_time_minutes,
            'strategy_name': self.strategy_name,
            'entry_reason': self.entry_reason,
            'metadata': self.metadata
        }


class IntradayStrategy(ABC):
    """
    Abstract base class for intraday trading strategies.

    Designed to work with real-time streaming data via callbacks.
    Subclasses implement:
    - on_bar(): Process each 1-minute bar
    - on_market_open(): Daily initialization
    - on_market_close(): End-of-day cleanup
    """

    def __init__(
        self,
        name: str,
        symbols: List[str],
        config: Dict[str, Any] = None
    ):
        """
        Initialize intraday strategy.

        Args:
            name: Strategy identifier
            symbols: List of symbols to trade
            config: Strategy configuration parameters
        """
        self.name = name
        self.symbols = symbols
        self.config = config or {}

        # Position tracking
        self.positions: Dict[str, IntradayPosition] = {}

        # Daily state
        self.is_market_open = False
        self.trading_date: Optional[datetime] = None

        # Performance tracking
        self.daily_trades: List[dict] = []
        self.daily_pnl: float = 0.0

        # Callbacks for order execution (set by stream handler)
        self._on_entry_signal = None
        self._on_exit_signal = None

        logger.info(f"Initialized {name} strategy for symbols: {symbols}")

    def set_callbacks(
        self,
        on_entry: callable = None,
        on_exit: callable = None
    ):
        """
        Set callback functions for order signals.

        Args:
            on_entry: Called when strategy wants to enter position
            on_exit: Called when strategy wants to exit position
        """
        self._on_entry_signal = on_entry
        self._on_exit_signal = on_exit

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    async def on_bar(self, symbol: str, bar: Any) -> None:
        """
        Process each 1-minute bar.

        This is the main callback for streaming data. Implement your
        strategy logic here.

        Args:
            symbol: Stock symbol
            bar: Bar data (OHLCV + metadata)
        """
        pass

    @abstractmethod
    async def on_market_open(self) -> None:
        """
        Called at market open (9:30 AM ET).

        Use this to:
        - Reset daily state
        - Initialize tracking variables
        - Prepare for the trading day
        """
        pass

    @abstractmethod
    async def on_market_close(self) -> None:
        """
        Called at market close (4:00 PM ET).

        Use this to:
        - Close any open positions
        - Calculate daily P&L
        - Log daily summary
        """
        pass

    # =========================================================================
    # Position management
    # =========================================================================

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for symbol."""
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[IntradayPosition]:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        shares: int,
        entry_price: float,
        stop_loss: float,
        target_price: float = None,
        max_hold_minutes: int = None,
        reason: str = "",
        metadata: Dict[str, Any] = None
    ) -> IntradayPosition:
        """
        Open a new intraday position.

        Args:
            symbol: Stock symbol
            side: LONG or SHORT
            shares: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Optional target price
            max_hold_minutes: Optional max hold time
            reason: Entry reason for logging
            metadata: Additional data to track

        Returns:
            The created IntradayPosition
        """
        position = IntradayPosition(
            symbol=symbol,
            side=side,
            shares=shares,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            target_price=target_price,
            max_hold_minutes=max_hold_minutes,
            strategy_name=self.name,
            entry_reason=reason,
            metadata=metadata or {}
        )

        self.positions[symbol] = position

        logger.info(
            f"[{self.name}] OPENED {side.value} {shares} {symbol} @ ${entry_price:.2f} "
            f"(SL: ${stop_loss:.2f}, Target: ${target_price:.2f if target_price else 0:.2f})"
        )

        # Trigger callback if set
        if self._on_entry_signal:
            self._on_entry_signal(position)

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = ""
    ) -> Optional[dict]:
        """
        Close an open position.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            reason: Exit reason for logging

        Returns:
            Trade result dict or None if no position
        """
        position = self.positions.pop(symbol, None)
        if position is None:
            logger.warning(f"No position to close for {symbol}")
            return None

        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.shares
            pnl_pct = ((exit_price / position.entry_price) - 1) * 100
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.shares
            pnl_pct = ((position.entry_price / exit_price) - 1) * 100

        hold_time = position.hold_time_minutes

        trade_result = {
            'symbol': symbol,
            'side': position.side.value,
            'shares': position.shares,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'hold_time_minutes': hold_time,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'entry_reason': position.entry_reason,
            'strategy': self.name,
            'metadata': position.metadata
        }

        # Track daily results
        self.daily_trades.append(trade_result)
        self.daily_pnl += pnl

        logger.info(
            f"[{self.name}] CLOSED {position.side.value} {symbol} @ ${exit_price:.2f} "
            f"(P&L: ${pnl:+.2f} / {pnl_pct:+.2f}%, Reason: {reason})"
        )

        # Trigger callback if set
        if self._on_exit_signal:
            self._on_exit_signal(trade_result)

        return trade_result

    def update_position_price(
        self,
        symbol: str,
        price: float,
        timestamp: datetime = None
    ):
        """Update current price for a position."""
        position = self.positions.get(symbol)
        if position:
            position.update_price(price, timestamp)

    # =========================================================================
    # Daily lifecycle
    # =========================================================================

    async def start_trading_day(self, date: datetime = None):
        """
        Initialize for a new trading day.

        Called before market open.
        """
        self.trading_date = date or datetime.now()
        self.is_market_open = False
        self.daily_trades = []
        self.daily_pnl = 0.0

        logger.info(f"[{self.name}] Starting trading day: {self.trading_date.date()}")

    async def end_trading_day(self):
        """
        Finalize the trading day.

        Called after market close.
        """
        self.is_market_open = False

        # Log daily summary
        wins = sum(1 for t in self.daily_trades if t['pnl'] > 0)
        losses = sum(1 for t in self.daily_trades if t['pnl'] <= 0)

        logger.info(
            f"[{self.name}] Day complete: {len(self.daily_trades)} trades, "
            f"{wins}W/{losses}L, P&L: ${self.daily_pnl:+.2f}"
        )

    # =========================================================================
    # Utility methods
    # =========================================================================

    def is_in_trading_window(
        self,
        start_time: time = None,
        end_time: time = None
    ) -> bool:
        """
        Check if current time is within trading window.

        Args:
            start_time: Window start (default: market open)
            end_time: Window end (default: market close)

        Returns:
            True if within window
        """
        start = start_time or MARKET_OPEN
        end = end_time or MARKET_CLOSE

        now = datetime.now().time()
        return start <= now <= end

    def minutes_since_open(self) -> float:
        """Get minutes elapsed since market open."""
        now = datetime.now()
        open_time = now.replace(
            hour=MARKET_OPEN.hour,
            minute=MARKET_OPEN.minute,
            second=0,
            microsecond=0
        )
        return (now - open_time).total_seconds() / 60

    def minutes_until_close(self) -> float:
        """Get minutes remaining until market close."""
        now = datetime.now()
        close_time = now.replace(
            hour=MARKET_CLOSE.hour,
            minute=MARKET_CLOSE.minute,
            second=0,
            microsecond=0
        )
        return (close_time - now).total_seconds() / 60

    def get_status(self) -> dict:
        """Get current strategy status."""
        return {
            'name': self.name,
            'symbols': self.symbols,
            'is_market_open': self.is_market_open,
            'trading_date': self.trading_date.isoformat() if self.trading_date else None,
            'positions': {
                symbol: pos.to_dict()
                for symbol, pos in self.positions.items()
            },
            'daily_trades': len(self.daily_trades),
            'daily_pnl': self.daily_pnl
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', symbols={self.symbols})"
