"""
Daily Bars Fetcher
==================
Fetches and stores daily OHLCV data from Alpaca.
Stores data in Parquet format for efficient reading.

Safety features:
- Request timeouts (default 30s)
- Retry with exponential backoff (3 attempts)
- Rate limiting between requests
- Graceful error handling
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import time
import random
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import pandas as pd
import requests
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError, ReadTimeout

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    DIRS, HISTORICAL_YEARS, BATCH_SIZE
)
from utils.timezone import normalize_timestamp, normalize_dataframe

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Timeout settings (seconds)
REQUEST_TIMEOUT = 30.0      # Individual request timeout
CONNECT_TIMEOUT = 10.0      # Connection timeout
SYMBOL_TIMEOUT = 60.0       # Max time per symbol fetch (includes retries)

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0      # Base delay for exponential backoff
RETRY_MAX_DELAY = 30.0      # Max delay between retries

# Rate limiting
MIN_REQUEST_DELAY = 0.2     # Minimum delay between requests (seconds)

# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(max_retries: int = MAX_RETRIES,
                       base_delay: float = RETRY_BASE_DELAY,
                       max_delay: float = RETRY_MAX_DELAY,
                       exceptions: tuple = (Exception,)):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay = delay * (0.5 + random.random())  # Add jitter
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                                      f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

# Lazy imports - only load when needed
_stock_client = None

def _get_client():
    """Lazy load Alpaca client."""
    global _stock_client
    if _stock_client is None:
        from alpaca.data.historical import StockHistoricalDataClient

        _stock_client = StockHistoricalDataClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY
        )

    return _stock_client


class DailyBarsFetcher:
    """Fetches and manages daily OHLCV data."""
    
    def __init__(self):
        self.data_dir = DIRS["daily"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = None  # Lazy load
    
    def _ensure_client(self):
        """Ensure Alpaca client is initialized."""
        if self.client is None:
            self.client = _get_client()
    
    def get_parquet_path(self, symbol: str) -> Path:
        """Get path to parquet file for a symbol."""
        return self.data_dir / f"{symbol}.parquet"
    
    def fetch_symbol(self, symbol: str, start_date: datetime = None,
                     end_date: datetime = None, force: bool = False) -> pd.DataFrame:
        """
        Fetch daily bars for a single symbol with timeout and retry protection.

        Args:
            symbol: Stock symbol
            start_date: Start date (default: HISTORICAL_YEARS ago)
            end_date: End date (default: today)
            force: If True, refetch even if data exists

        Returns:
            DataFrame with OHLCV data
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        self._ensure_client()

        parquet_path = self.get_parquet_path(symbol)

        # Check if we have recent data
        if not force and parquet_path.exists():
            try:
                existing_df = pd.read_parquet(parquet_path)
                if len(existing_df) > 0:
                    last_date = existing_df.index.max()
                    if isinstance(last_date, pd.Timestamp):
                        # Convert to naive datetime using centralized timezone utility
                        last_date = normalize_timestamp(last_date).to_pydatetime()

                    # If data is from today or yesterday, don't refetch
                    if (datetime.now() - last_date).days <= 1:
                        logger.debug(f"{symbol}: Using cached data (last: {last_date.date()})")
                        return existing_df

                    # Otherwise, fetch only new data
                    start_date = last_date + timedelta(days=1)
            except Exception as e:
                logger.warning(f"{symbol}: Could not read cache, will refetch: {e}")

        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=HISTORICAL_YEARS * 365)

        # Use timeout wrapper for the actual API call
        try:
            df = self._fetch_with_timeout(symbol, start_date, end_date)

            if df is None or df.empty:
                logger.warning(f"{symbol}: No data returned")
                return pd.DataFrame()

            # Merge with existing data if we're updating
            if parquet_path.exists() and not force:
                try:
                    existing_df = pd.read_parquet(parquet_path)
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                except Exception as e:
                    logger.warning(f"{symbol}: Could not merge with existing data: {e}")

            # Save to parquet
            df.to_parquet(parquet_path)
            logger.info(f"{symbol}: Saved {len(df)} bars to {parquet_path.name}")

            return df

        except FuturesTimeoutError:
            logger.error(f"{symbol}: Request timed out after {SYMBOL_TIMEOUT}s")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"{symbol}: Failed to fetch - {e}")
            return pd.DataFrame()

    def _fetch_with_timeout(self, symbol: str, start_date: datetime,
                            end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch bars with timeout protection using ThreadPoolExecutor.

        This wraps the API call in a thread with a timeout to prevent
        indefinite hangs on network issues.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._fetch_bars_with_retry,
                symbol, start_date, end_date
            )
            return future.result(timeout=SYMBOL_TIMEOUT)

    @retry_with_backoff(
        max_retries=MAX_RETRIES,
        base_delay=RETRY_BASE_DELAY,
        max_delay=RETRY_MAX_DELAY,
        exceptions=(Timeout, RequestsConnectionError, ReadTimeout,
                   ConnectionError, OSError, Exception)
    )
    def _fetch_bars_with_retry(self, symbol: str, start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
        """
        Fetch bars from Alpaca API with retry logic.

        This method is wrapped with retry_with_backoff decorator.
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request)

        if bars.df.empty:
            return pd.DataFrame()

        df = bars.df.reset_index()

        # Handle multi-index if present
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].drop(columns=['symbol'])

        # Set timestamp as index
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)

        return df
    
    def fetch_symbols(self, symbols: List[str], force: bool = False,
                      delay: float = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily bars for multiple symbols with rate limiting.

        Args:
            symbols: List of stock symbols
            force: If True, refetch all data
            delay: Delay between requests (default: MIN_REQUEST_DELAY)

        Returns:
            Dict mapping symbol to DataFrame
        """
        if delay is None:
            delay = MIN_REQUEST_DELAY

        results = {}
        total = len(symbols)
        failed = []
        start_time = time.time()

        for i, symbol in enumerate(symbols):
            try:
                df = self.fetch_symbol(symbol, force=force)
                if not df.empty:
                    results[symbol] = df
                else:
                    failed.append(symbol)

                # Progress logging every 50 symbols
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {i + 1}/{total} symbols "
                               f"({len(results)} success, {len(failed)} failed, "
                               f"{rate:.1f} symbols/sec)")

                # Rate limiting
                if delay > 0 and i < total - 1:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"{symbol}: Unexpected error - {e}")
                failed.append(symbol)
                continue

        elapsed = time.time() - start_time
        logger.info(f"Fetch complete: {len(results)}/{total} symbols in {elapsed:.1f}s "
                   f"({len(failed)} failed)")

        if failed and len(failed) <= 10:
            logger.warning(f"Failed symbols: {', '.join(failed)}")

        return results
    
    def fetch_batch(self, symbols: List[str], start_date: datetime = None,
                    end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily bars for multiple symbols in a single API call.
        More efficient than individual fetches.

        Args:
            symbols: List of stock symbols (max 100 per request)
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping symbol to DataFrame
        """
        self._ensure_client()

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=HISTORICAL_YEARS * 365)

        results = {}
        total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE

        # Process in batches
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            try:
                # Use timeout wrapper for batch fetch
                batch_results = self._fetch_batch_with_timeout(
                    batch, start_date, end_date
                )

                if batch_results:
                    results.update(batch_results)
                    logger.info(f"Batch {batch_num}/{total_batches}: "
                               f"Fetched {len(batch_results)}/{len(batch)} symbols")
                else:
                    logger.warning(f"Batch {batch_num}/{total_batches}: No data returned")

                # Rate limiting between batches
                if i + BATCH_SIZE < len(symbols):
                    time.sleep(MIN_REQUEST_DELAY * 2)

            except FuturesTimeoutError:
                logger.error(f"Batch {batch_num}/{total_batches}: Timed out")
                continue
            except Exception as e:
                logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
                continue

        logger.info(f"Batch fetch complete: {len(results)}/{len(symbols)} symbols")
        return results

    def _fetch_batch_with_timeout(self, symbols: List[str], start_date: datetime,
                                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch a batch of symbols with timeout protection."""
        # Longer timeout for batch operations
        batch_timeout = SYMBOL_TIMEOUT * 2

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._fetch_batch_with_retry,
                symbols, start_date, end_date
            )
            return future.result(timeout=batch_timeout)

    @retry_with_backoff(
        max_retries=MAX_RETRIES,
        base_delay=RETRY_BASE_DELAY,
        max_delay=RETRY_MAX_DELAY,
        exceptions=(Timeout, RequestsConnectionError, ReadTimeout,
                   ConnectionError, OSError, Exception)
    )
    def _fetch_batch_with_retry(self, symbols: List[str], start_date: datetime,
                                 end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch a batch of symbols with retry logic."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request)

        if bars.df.empty:
            return {}

        df = bars.df.reset_index()
        results = {}

        # Split by symbol and save
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            if symbol_df.empty:
                continue

            symbol_df = symbol_df.drop(columns=['symbol'])
            symbol_df = symbol_df.set_index('timestamp')
            symbol_df.index = pd.to_datetime(symbol_df.index)

            # Save to parquet
            parquet_path = self.get_parquet_path(symbol)
            symbol_df.to_parquet(parquet_path)

            results[symbol] = symbol_df

        return results
    
    def load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load cached data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame or None if not cached
        """
        parquet_path = self.get_parquet_path(symbol)

        if not parquet_path.exists():
            return None

        try:
            df = pd.read_parquet(parquet_path)
            return normalize_dataframe(df)
        except Exception as e:
            logger.error(f"{symbol}: Failed to load - {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with cached data."""
        return [p.stem for p in self.data_dir.glob("*.parquet")]
    
    def get_latest_date(self, symbol: str) -> Optional[datetime]:
        """Get the latest date in cached data for a symbol."""
        df = self.load_symbol(symbol)
        if df is None or df.empty:
            return None
        
        last_date = df.index.max()
        if isinstance(last_date, pd.Timestamp):
            return last_date.to_pydatetime()
        return last_date
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """
        Remove parquet files not updated in N days.
        
        Args:
            days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        
        for parquet_path in self.data_dir.glob("*.parquet"):
            try:
                mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
                if mtime < cutoff:
                    parquet_path.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove {parquet_path}: {e}")
        
        if removed > 0:
            logger.info(f"Removed {removed} old parquet files")
        
        return removed


# Convenience functions

def fetch_daily_bars(symbol: str, **kwargs) -> pd.DataFrame:
    """Convenience function to fetch daily bars for a symbol."""
    fetcher = DailyBarsFetcher()
    return fetcher.fetch_symbol(symbol, **kwargs)


def load_daily_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Convenience function to load cached daily bars."""
    fetcher = DailyBarsFetcher()
    return fetcher.load_symbol(symbol)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch daily bars from Alpaca")
    parser.add_argument("symbols", nargs="*", default=["SPY", "QQQ", "IWM"],
                        help="Symbols to fetch")
    parser.add_argument("--force", action="store_true", 
                        help="Force refetch all data")
    parser.add_argument("--list", action="store_true",
                        help="List cached symbols")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    fetcher = DailyBarsFetcher()
    
    if args.list:
        symbols = fetcher.get_available_symbols()
        print(f"\nCached symbols ({len(symbols)}):")
        for symbol in sorted(symbols):
            latest = fetcher.get_latest_date(symbol)
            print(f"  {symbol}: {latest.date() if latest else 'N/A'}")
    else:
        print(f"\nFetching: {', '.join(args.symbols)}")
        results = fetcher.fetch_symbols(args.symbols, force=args.force)
        
        print(f"\nResults:")
        for symbol, df in results.items():
            print(f"  {symbol}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
