"""
Intraday (Minute) Bars Fetcher
==============================
Downloads and manages 1-minute bar data for gap-fill strategy.

Alpaca provides up to 1 month of historical minute bars.
We store in parquet format, one file per symbol per day.

Safety features:
- Request timeouts (default 30s per request)
- Retry with exponential backoff (3 attempts)
- Rate limiting between requests
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
import numpy as np
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError, ReadTimeout

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, DIRS, INTRADAY_DAYS,
    INTRADAY_SYMBOLS, INTRADAY_DATA_CONFIG, INTRADAY_UNIVERSE
)

logger = logging.getLogger(__name__)

# =============================================================================
# TIMEOUT AND RETRY CONFIGURATION
# =============================================================================

REQUEST_TIMEOUT = 30.0      # Individual request timeout
SYMBOL_TIMEOUT = 60.0       # Max time per symbol fetch (includes retries)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0


def retry_with_backoff(max_retries: int = MAX_RETRIES,
                       base_delay: float = RETRY_BASE_DELAY,
                       max_delay: float = RETRY_MAX_DELAY,
                       exceptions: tuple = (Exception,)):
    """Decorator that retries a function with exponential backoff."""
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
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay = delay * (0.5 + random.random())
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                                      f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


class IntradayDataManager:
    """
    Manages minute-bar data for intraday strategies.

    Storage format: data/historical/intraday/{symbol}/{date}.parquet
    Each file contains one day of 1-minute bars (390 bars for full day).

    Universe is configured in config.py:INTRADAY_UNIVERSE (42 symbols across
    8 categories: broad market, sectors, mega-caps, volatility, commodities,
    bonds, international, thematic).
    """

    # Default universe from config (can be overridden per-call)
    DEFAULT_UNIVERSE = INTRADAY_SYMBOLS

    # Legacy alias for backwards compatibility
    GAP_FILL_UNIVERSE = INTRADAY_UNIVERSE.get("broad_market", ["SPY", "QQQ", "IWM", "DIA"])

    def __init__(self, symbols: List[str] = None):
        """
        Initialize the intraday data manager.

        Args:
            symbols: List of symbols to manage. Defaults to full INTRADAY_SYMBOLS
                     from config.py (42 symbols across all categories).
        """
        self.symbols = symbols or self.DEFAULT_UNIVERSE
        self.data_client = StockHistoricalDataClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY
        )
        self.intraday_dir = DIRS["intraday"]
        self.intraday_dir.mkdir(parents=True, exist_ok=True)

        # Config-driven settings
        self.retention_days = INTRADAY_DATA_CONFIG.get("retention_days", 30)
        self.refresh_days = INTRADAY_DATA_CONFIG.get("refresh_days", 5)
        self.rate_limit = INTRADAY_DATA_CONFIG.get("rate_limit_seconds", 0.25)
        
    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get directory for a symbol's intraday data."""
        sym_dir = self.intraday_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        return sym_dir
    
    def _get_file_path(self, symbol: str, date: datetime) -> Path:
        """Get path for a specific day's data file."""
        date_str = date.strftime("%Y%m%d")
        return self._get_symbol_dir(symbol) / f"{date_str}.parquet"
    
    def fetch_day(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch minute bars for a single day with timeout and retry protection.

        Args:
            symbol: Stock symbol
            date: Date to fetch (will get full trading day)

        Returns:
            DataFrame with OHLCV minute bars, or None if failed
        """
        try:
            return self._fetch_day_with_timeout(symbol, date)
        except FuturesTimeoutError:
            logger.error(f"{symbol}: Request timed out after {SYMBOL_TIMEOUT}s for {date.date()}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {symbol} for {date.date()}: {e}")
            return None

    def _fetch_day_with_timeout(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Fetch with timeout protection using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._fetch_day_with_retry, symbol, date)
            return future.result(timeout=SYMBOL_TIMEOUT)

    @retry_with_backoff(
        max_retries=MAX_RETRIES,
        base_delay=RETRY_BASE_DELAY,
        max_delay=RETRY_MAX_DELAY,
        exceptions=(Timeout, RequestsConnectionError, ReadTimeout,
                   ConnectionError, OSError, Exception)
    )
    def _fetch_day_with_retry(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Fetch minute bars with retry logic."""
        # Market hours: 9:30 AM - 4:00 PM ET
        start = date.replace(hour=9, minute=30, second=0, microsecond=0)
        end = date.replace(hour=16, minute=0, second=0, microsecond=0)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )

        bars = self.data_client.get_stock_bars(request)

        if symbol not in bars.data or len(bars.data[symbol]) == 0:
            logger.warning(f"No data for {symbol} on {date.date()}")
            return None

        # Convert to DataFrame
        records = []
        for bar in bars.data[symbol]:
            records.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
                'trade_count': bar.trade_count if hasattr(bar, 'trade_count') else None
            })

        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)

        logger.info(f"Fetched {len(df)} minute bars for {symbol} on {date.date()}")
        return df
    
    def download_recent(
        self,
        symbols: List[str] = None,
        days: int = None
    ) -> Dict[str, int]:
        """
        Download recent minute bars for specified symbols.

        Args:
            symbols: List of symbols (defaults to self.symbols, i.e., full universe)
            days: Number of days to fetch (defaults to self.refresh_days from config)

        Returns:
            Dict of {symbol: days_downloaded}
        """
        symbols = symbols or self.symbols
        days = days or self.refresh_days
        
        results = {}
        end_date = datetime.now()
        
        for symbol in symbols:
            downloaded = 0
            
            for i in range(days):
                date = end_date - timedelta(days=i)
                
                # Skip weekends
                if date.weekday() >= 5:
                    continue
                
                # Skip if we already have this day
                file_path = self._get_file_path(symbol, date)
                if file_path.exists():
                    downloaded += 1
                    continue
                
                # Fetch and save
                df = self.fetch_day(symbol, date)
                if df is not None and len(df) > 0:
                    try:
                        df.to_parquet(file_path)
                        downloaded += 1
                        logger.info(f"Saved {symbol} {date.date()} -> {file_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to save intraday data {symbol} {date.date()}: {e}")
                
                # Rate limiting
                time.sleep(self.rate_limit)
            
            results[symbol] = downloaded
            logger.info(f"{symbol}: {downloaded} days available")
        
        return results
    
    def load_day(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """
        Load minute bars for a specific day from cache.
        
        Args:
            symbol: Stock symbol
            date: Date to load
            
        Returns:
            DataFrame with minute bars, or None if not available
        """
        file_path = self._get_file_path(symbol, date)
        
        if not file_path.exists():
            return None
        
        df = pd.read_parquet(file_path)
        return df
    
    def load_range(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load minute bars for a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Combined DataFrame with all available data
        """
        frames = []
        current = start_date
        
        while current <= end_date:
            if current.weekday() < 5:  # Skip weekends
                df = self.load_day(symbol, current)
                if df is not None:
                    frames.append(df)
            current += timedelta(days=1)
        
        if not frames:
            return None
        
        return pd.concat(frames).sort_index()
    
    def get_previous_close(self, symbol: str, date: datetime) -> Optional[float]:
        """
        Get the previous day's closing price.
        
        Args:
            symbol: Stock symbol
            date: Current date (will look for previous trading day)
            
        Returns:
            Previous close price, or None
        """
        # Look back up to 5 days to find previous trading day
        for i in range(1, 6):
            prev_date = date - timedelta(days=i)
            df = self.load_day(symbol, prev_date)
            if df is not None and len(df) > 0:
                return df['close'].iloc[-1]
        
        return None
    
    def get_opening_bar(self, symbol: str, date: datetime) -> Optional[dict]:
        """
        Get the opening bar (9:30 AM) for a specific day.
        
        Args:
            symbol: Stock symbol
            date: Date
            
        Returns:
            Dict with open, high, low, close, volume or None
        """
        df = self.load_day(symbol, date)
        if df is None or len(df) == 0:
            return None
        
        first_bar = df.iloc[0]
        return {
            'timestamp': df.index[0],
            'open': first_bar['open'],
            'high': first_bar['high'],
            'low': first_bar['low'],
            'close': first_bar['close'],
            'volume': first_bar['volume']
        }
    
    def calculate_gap(self, symbol: str, date: datetime) -> Optional[dict]:
        """
        Calculate the overnight gap for a specific day.
        
        Args:
            symbol: Stock symbol
            date: Date to calculate gap for
            
        Returns:
            Dict with gap info or None
        """
        prev_close = self.get_previous_close(symbol, date)
        opening = self.get_opening_bar(symbol, date)
        
        if prev_close is None or opening is None:
            return None
        
        gap_dollars = opening['open'] - prev_close
        gap_percent = (gap_dollars / prev_close) * 100
        
        return {
            'date': date.date(),
            'symbol': symbol,
            'prev_close': prev_close,
            'open': opening['open'],
            'gap_dollars': gap_dollars,
            'gap_percent': gap_percent,
            'direction': 'up' if gap_percent > 0 else 'down'
        }
    
    def get_minute_bars(
        self,
        symbol: str,
        n_minutes: int = 120,
        end_time: datetime = None
    ) -> Optional[pd.DataFrame]:
        """
        Get recent minute bars for a symbol.

        Args:
            symbol: Stock symbol
            n_minutes: Number of minutes of data to return
            end_time: End time (defaults to now)

        Returns:
            DataFrame with minute bars or None if not available
        """
        end_time = end_time or datetime.now()

        # For intraday, we need today's data
        today = end_time.date()
        today_dt = datetime.combine(today, datetime.min.time())

        # Try to load today's data
        df = self.fetch_day(symbol, today_dt)

        if df is None or df.empty:
            # Try yesterday if today not available
            yesterday = today_dt - timedelta(days=1)
            df = self.fetch_day(symbol, yesterday)

        if df is None or df.empty:
            logger.warning(f"No minute data available for {symbol}")
            return None

        # Get last n_minutes
        if len(df) > n_minutes:
            df = df.tail(n_minutes)

        return df

    def get_gap_data(
        self,
        symbol: str,
        date: datetime = None
    ) -> Optional[tuple]:
        """
        Get gap percentage and recent bars for gap-fill strategy.

        Args:
            symbol: Stock symbol
            date: Date to check (defaults to today)

        Returns:
            Tuple of (gap_pct, recent_bars_df) or None if not available
        """
        date = date or datetime.now()

        # Calculate gap
        gap_info = self.calculate_gap(symbol, date)
        if gap_info is None:
            return None

        gap_pct = gap_info['gap_percent'] / 100  # Convert to decimal

        # Get recent minute bars (first 30 minutes after open)
        df = self.fetch_day(symbol, date)
        if df is None or df.empty:
            return gap_pct, pd.DataFrame()

        # Get first 30 minutes of trading (9:30-10:00)
        recent_bars = df.head(30)

        return gap_pct, recent_bars

    def get_realtime_bar(self, symbol: str) -> Optional[dict]:
        """
        Get the most recent bar for a symbol (pseudo-realtime).

        Uses the latest available minute bar from downloaded data.
        For true real-time, use the stream_handler instead.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with latest bar data or None
        """
        df = self.get_minute_bars(symbol, n_minutes=5)

        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        return {
            'symbol': symbol,
            'timestamp': latest.name,
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume']
        }

    def get_data_status(self, symbols: List[str] = None) -> Dict[str, dict]:
        """
        Get status of intraday data for tracked symbols.

        Args:
            symbols: List of symbols to check (defaults to self.symbols)

        Returns:
            Dict of {symbol: {days_available, oldest, newest}}
        """
        symbols = symbols or self.symbols
        status = {}

        for symbol in symbols:
            sym_dir = self._get_symbol_dir(symbol)
            files = list(sym_dir.glob("*.parquet"))
            
            if not files:
                status[symbol] = {'days_available': 0, 'oldest': None, 'newest': None}
                continue
            
            dates = sorted([f.stem for f in files])
            status[symbol] = {
                'days_available': len(files),
                'oldest': dates[0],
                'newest': dates[-1]
            }
        
        return status
    
    def cleanup_old_data(self, keep_days: int = None, symbols: List[str] = None):
        """
        Remove intraday data older than specified days.

        Args:
            keep_days: Days to keep (defaults to self.retention_days from config)
            symbols: Symbols to clean (defaults to self.symbols)
        """
        keep_days = keep_days or self.retention_days
        symbols = symbols or self.symbols
        cutoff = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff.strftime("%Y%m%d")

        removed = 0
        for symbol in symbols:
            sym_dir = self._get_symbol_dir(symbol)
            for file in sym_dir.glob("*.parquet"):
                if file.stem < cutoff_str:
                    file.unlink()
                    removed += 1
        
        logger.info(f"Removed {removed} old intraday files")
        return removed


def download_intraday_data():
    """Convenience function to download intraday data for full universe."""
    manager = IntradayDataManager()

    print("Downloading intraday data for full universe...")
    print(f"Symbols: {len(manager.symbols)} total")
    print(f"Categories: {list(INTRADAY_UNIVERSE.keys())}")
    print(f"Days: {manager.refresh_days}")
    print()
    
    results = manager.download_recent()

    # Summary by category
    print("\nResults by category:")
    for category, symbols in INTRADAY_UNIVERSE.items():
        cat_days = sum(results.get(s, 0) for s in symbols)
        print(f"  {category}: {cat_days} days across {len(symbols)} symbols")

    print(f"\nTotal: {sum(results.values())} symbol-days downloaded")

    # Show any symbols with no data
    status = manager.get_data_status()
    missing = [s for s, info in status.items() if info['days_available'] == 0]
    if missing:
        print(f"\nWarning: {len(missing)} symbols with no data: {missing}")


# Legacy alias for backwards compatibility
download_gap_fill_data = download_intraday_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_intraday_data()
