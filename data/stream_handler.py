"""
Real-Time Market Data Stream Handler
====================================
Handles real-time 1-minute bar streaming from Alpaca for intraday strategies.

Features:
- Subscribe/unsubscribe to multiple symbols with async callbacks
- Market hours detection (9:30 AM - 4:00 PM ET)
- Graceful start/stop with proper cleanup
- Automatic reconnection on errors
- Thread-safe subscription management

Usage:
    from data.stream_handler import MarketDataStream
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

    stream = MarketDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    async def on_bar(bar):
        print(f"{bar.symbol}: {bar.close}")

    stream.subscribe("SPY", on_bar)
    await stream.start()
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Callable, Dict, List, Optional, Set
from threading import Lock
import pytz

from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataStream:
    """
    Real-time market data streaming handler using Alpaca's StockDataStream.

    Provides subscription management for multiple symbols with individual callbacks,
    market hours detection, and robust error handling with reconnection logic.

    Attributes:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        paper: Whether to use paper trading endpoint (default True)

    Example:
        >>> stream = MarketDataStream(api_key, secret_key)
        >>> async def handle_bar(bar):
        ...     print(f"Received: {bar.symbol} @ {bar.close}")
        >>> stream.subscribe("AAPL", handle_bar)
        >>> await stream.start()
    """

    # US Eastern timezone for market hours
    ET_TIMEZONE = pytz.timezone("America/New_York")

    # Regular market hours (9:30 AM - 4:00 PM ET)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True
    ) -> None:
        """
        Initialize the market data stream handler.

        Args:
            api_key: Alpaca API key for authentication
            secret_key: Alpaca secret key for authentication
            paper: Use paper trading endpoint if True (default), live if False
        """
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper

        # Stream connection
        self._stream: Optional[StockDataStream] = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

        # Subscription management
        self._subscriptions: Dict[str, Callable[[Bar], None]] = {}
        self._lock = Lock()

        # Task management
        self._stream_task: Optional[asyncio.Task] = None

        logger.info(
            "MarketDataStream initialized (paper=%s, api_key=%s...)",
            paper,
            api_key[:8] if api_key else "None"
        )

    def _create_stream(self) -> StockDataStream:
        """Create a new StockDataStream instance."""
        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key
        )

    async def _handle_bar(self, bar: Bar) -> None:
        """
        Internal bar handler that routes bars to registered callbacks.

        Args:
            bar: The received bar data from Alpaca
        """
        symbol = bar.symbol

        with self._lock:
            callback = self._subscriptions.get(symbol)

        if callback:
            try:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(callback):
                    await callback(bar)
                else:
                    callback(bar)
            except Exception as e:
                logger.error(
                    "Error in callback for %s: %s",
                    symbol,
                    str(e),
                    exc_info=True
                )
        else:
            logger.warning("Received bar for unsubscribed symbol: %s", symbol)

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[Bar], None]
    ) -> bool:
        """
        Subscribe to real-time bars for a symbol.

        Args:
            symbol: Stock symbol to subscribe to (e.g., "SPY", "AAPL")
            callback: Async or sync function to call when a bar is received.
                      Should accept a single Bar argument.

        Returns:
            True if subscription was added, False if already subscribed

        Example:
            >>> async def my_handler(bar):
            ...     print(f"{bar.symbol}: O={bar.open} H={bar.high} L={bar.low} C={bar.close}")
            >>> stream.subscribe("SPY", my_handler)
        """
        symbol = symbol.upper()

        with self._lock:
            if symbol in self._subscriptions:
                logger.warning("Already subscribed to %s", symbol)
                return False

            self._subscriptions[symbol] = callback
            logger.info("Subscribed to %s", symbol)

        # If stream is running, subscribe to the new symbol
        if self._stream and self._running:
            try:
                self._stream.subscribe_bars(self._handle_bar, symbol)
                logger.info("Added live subscription for %s", symbol)
            except Exception as e:
                logger.error("Failed to add live subscription for %s: %s", symbol, e)

        return True

    def unsubscribe(self, symbol: str) -> bool:
        """
        Unsubscribe from a symbol's real-time bars.

        Args:
            symbol: Stock symbol to unsubscribe from

        Returns:
            True if unsubscribed, False if was not subscribed
        """
        symbol = symbol.upper()

        with self._lock:
            if symbol not in self._subscriptions:
                logger.warning("Not subscribed to %s", symbol)
                return False

            del self._subscriptions[symbol]
            logger.info("Unsubscribed from %s", symbol)

        # If stream is running, unsubscribe from the symbol
        if self._stream and self._running:
            try:
                self._stream.unsubscribe_bars(symbol)
                logger.info("Removed live subscription for %s", symbol)
            except Exception as e:
                logger.error("Failed to remove live subscription for %s: %s", symbol, e)

        return True

    def get_subscribed_symbols(self) -> List[str]:
        """
        Get list of currently subscribed symbols.

        Returns:
            List of symbol strings currently subscribed
        """
        with self._lock:
            return list(self._subscriptions.keys())

    @classmethod
    def is_market_hours(cls) -> bool:
        """
        Check if current time is within regular market hours.

        Regular market hours are 9:30 AM - 4:00 PM Eastern Time.
        Does NOT account for holidays - only checks time of day.

        Returns:
            True if within market hours, False otherwise
        """
        now_et = datetime.now(cls.ET_TIMEZONE)
        current_time = now_et.time()

        # Check if weekday (Monday=0, Sunday=6)
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return False

        return cls.MARKET_OPEN <= current_time < cls.MARKET_CLOSE

    @classmethod
    def get_market_status(cls) -> Dict[str, any]:
        """
        Get detailed market status information.

        Returns:
            Dictionary with market status details:
            - is_open: bool indicating if market is open
            - current_time_et: current time in ET
            - market_open: market open time
            - market_close: market close time
            - is_weekend: bool indicating if it's weekend
        """
        now_et = datetime.now(cls.ET_TIMEZONE)
        current_time = now_et.time()
        is_weekend = now_et.weekday() >= 5

        return {
            "is_open": cls.is_market_hours(),
            "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "market_open": cls.MARKET_OPEN.strftime("%H:%M"),
            "market_close": cls.MARKET_CLOSE.strftime("%H:%M"),
            "is_weekend": is_weekend,
            "day_of_week": now_et.strftime("%A"),
        }

    async def start(self) -> None:
        """
        Start the real-time data stream.

        Connects to Alpaca's streaming API and begins receiving bars
        for all subscribed symbols. Handles reconnection on errors.

        Raises:
            RuntimeError: If stream is already running
            ConnectionError: If unable to connect after max retries
        """
        if self._running:
            logger.warning("Stream is already running")
            return

        with self._lock:
            symbols = list(self._subscriptions.keys())

        if not symbols:
            logger.warning("No symbols subscribed - nothing to stream")
            return

        logger.info("Starting stream for %d symbols: %s", len(symbols), symbols)

        self._running = True
        self._reconnect_attempts = 0

        while self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                # Create fresh stream connection
                self._stream = self._create_stream()

                # Subscribe to all symbols
                self._stream.subscribe_bars(self._handle_bar, *symbols)

                logger.info("Connected to Alpaca stream, subscribing to bars...")

                # Run the stream (blocks until disconnected)
                await self._stream._run_forever()

            except asyncio.CancelledError:
                logger.info("Stream cancelled")
                break

            except Exception as e:
                self._reconnect_attempts += 1
                logger.error(
                    "Stream error (attempt %d/%d): %s",
                    self._reconnect_attempts,
                    self._max_reconnect_attempts,
                    str(e),
                    exc_info=True
                )

                if self._running and self._reconnect_attempts < self._max_reconnect_attempts:
                    delay = self._reconnect_delay * self._reconnect_attempts
                    logger.info("Reconnecting in %d seconds...", delay)
                    await asyncio.sleep(delay)

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._running = False
            raise ConnectionError(
                f"Failed to connect after {self._max_reconnect_attempts} attempts"
            )

        logger.info("Stream stopped")

    async def stop(self) -> None:
        """
        Stop the real-time data stream gracefully.

        Unsubscribes from all symbols and closes the connection.
        Safe to call even if stream is not running.
        """
        if not self._running:
            logger.info("Stream is not running")
            return

        logger.info("Stopping stream...")
        self._running = False

        if self._stream:
            try:
                # Unsubscribe from all symbols
                with self._lock:
                    symbols = list(self._subscriptions.keys())

                if symbols:
                    self._stream.unsubscribe_bars(*symbols)
                    logger.info("Unsubscribed from all symbols")

                # Close the stream
                await self._stream.close()
                logger.info("Stream connection closed")

            except Exception as e:
                logger.error("Error during stream shutdown: %s", e)
            finally:
                self._stream = None

        # Cancel stream task if exists
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        logger.info("Stream stopped successfully")

    @property
    def is_running(self) -> bool:
        """Check if the stream is currently running."""
        return self._running

    @property
    def subscription_count(self) -> int:
        """Get the number of active subscriptions."""
        with self._lock:
            return len(self._subscriptions)


# Convenience function to create a pre-configured stream
def create_market_stream(paper: bool = True) -> MarketDataStream:
    """
    Create a MarketDataStream using credentials from config.

    Args:
        paper: Use paper trading endpoint if True (default)

    Returns:
        Configured MarketDataStream instance

    Raises:
        ValueError: If API credentials are not configured
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError(
            "Alpaca API credentials not configured. "
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file."
        )

    return MarketDataStream(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=paper
    )


# Test/demo code
if __name__ == "__main__":
    import sys

    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    async def demo_callback(bar: Bar) -> None:
        """Demo callback that prints bar data."""
        print(
            f"[{bar.timestamp}] {bar.symbol}: "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} "
            f"V={bar.volume:,}"
        )

    async def main():
        """Main demo function."""
        print("=" * 60)
        print("Market Data Stream Handler Demo")
        print("=" * 60)

        # Check market status
        status = MarketDataStream.get_market_status()
        print(f"\nMarket Status:")
        print(f"  Current Time (ET): {status['current_time_et']}")
        print(f"  Day of Week: {status['day_of_week']}")
        print(f"  Market Hours: {status['market_open']} - {status['market_close']}")
        print(f"  Is Weekend: {status['is_weekend']}")
        print(f"  Market Open: {status['is_open']}")
        print()

        # Check credentials
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            print("ERROR: Alpaca API credentials not configured.")
            print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file.")
            sys.exit(1)

        # Create stream
        try:
            stream = create_market_stream(paper=True)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        # Subscribe to SPY
        stream.subscribe("SPY", demo_callback)
        print(f"Subscribed symbols: {stream.get_subscribed_symbols()}")

        if not status['is_open']:
            print("\nWARNING: Market is closed. Stream will connect but no bars will arrive.")
            print("Real-time bars are only sent during market hours (9:30 AM - 4:00 PM ET)")

        print("\nStarting stream (press Ctrl+C to stop)...")
        print("-" * 60)

        try:
            await stream.start()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except ConnectionError as e:
            print(f"\nConnection failed: {e}")
        finally:
            await stream.stop()
            print("\nStream stopped. Goodbye!")

    # Run the demo
    asyncio.run(main())
