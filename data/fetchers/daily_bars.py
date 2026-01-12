"""
Daily Bars Fetcher
==================
Fetches and stores daily OHLCV data from Alpaca.
Stores data in Parquet format for efficient reading.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import time

import pandas as pd

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    DIRS, HISTORICAL_YEARS, BATCH_SIZE
)
from utils.timezone import normalize_timestamp, normalize_dataframe

logger = logging.getLogger(__name__)

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
        Fetch daily bars for a single symbol.
        
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
        
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=HISTORICAL_YEARS * 365)
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.client.get_stock_bars(request)
            
            if bars.df.empty:
                logger.warning(f"{symbol}: No data returned")
                return pd.DataFrame()
            
            df = bars.df.reset_index()
            
            # Handle multi-index if present
            if 'symbol' in df.columns:
                df = df[df['symbol'] == symbol].drop(columns=['symbol'])
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
            
            # Merge with existing data if we're updating
            if parquet_path.exists() and not force:
                existing_df = pd.read_parquet(parquet_path)
                df = pd.concat([existing_df, df])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
            
            # Save to parquet
            df.to_parquet(parquet_path)
            logger.info(f"{symbol}: Saved {len(df)} bars to {parquet_path.name}")
            
            return df
            
        except Exception as e:
            logger.error(f"{symbol}: Failed to fetch - {e}")
            return pd.DataFrame()
    
    def fetch_symbols(self, symbols: List[str], force: bool = False,
                      delay: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily bars for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            force: If True, refetch all data
            delay: Delay between requests to avoid rate limits
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.fetch_symbol(symbol, force=force)
                if not df.empty:
                    results[symbol] = df
                
                if i > 0 and i % 50 == 0:
                    logger.info(f"Progress: {i}/{total} symbols fetched")
                
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"{symbol}: Error - {e}")
                continue
        
        logger.info(f"Fetched {len(results)}/{total} symbols successfully")
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
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        self._ensure_client()
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=HISTORICAL_YEARS * 365)
        
        results = {}
        
        # Process in batches
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.client.get_stock_bars(request)
                
                if bars.df.empty:
                    continue
                
                df = bars.df.reset_index()
                
                # Split by symbol
                for symbol in batch:
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
                
                logger.info(f"Batch {i//BATCH_SIZE + 1}: Fetched {len(batch)} symbols")
                
                # Small delay between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Batch {i//BATCH_SIZE + 1} failed: {e}")
                continue
        
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
