"""
Unified Data Loader
====================
Loads and merges data from multiple sources:
1. Yahoo Finance daily (1990s - present) - Extended history
2. Alpaca daily (2016 - present) - Original source
3. Alpaca 1-minute (2016 - present) - Intraday data

Prioritizes Yahoo for daily data (longer history), falls back to Alpaca.
Handles intelligent merging when both sources exist.

Usage:
    from data.unified_data_loader import UnifiedDataLoader

    loader = UnifiedDataLoader()

    # Load daily data (auto-merges Yahoo + Alpaca)
    df = loader.load_daily("AAPL")

    # Load 1-minute data
    df_1min = loader.load_intraday("AAPL", timeframe="1min")

    # Load all symbols
    data = loader.load_all_daily()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Literal
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import DIRS

logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """
    Unified loader that merges data from multiple sources.

    Priority for daily data:
    1. Yahoo Finance (extended history back to 1990s)
    2. Alpaca daily (2016+, fallback)

    For intraday:
    1. Alpaca 1-minute (2016+)
    """

    def __init__(self):
        # Data directories
        self.yahoo_daily_dir = DIRS.get("daily_yahoo", DIRS["historical"] / "daily_yahoo")
        self.alpaca_daily_dir = DIRS.get("daily", DIRS["historical"] / "daily")
        self.intraday_1min_dir = DIRS.get("intraday_1min", DIRS["historical"] / "intraday_1min")

        # Cache
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._intraday_cache: Dict[str, pd.DataFrame] = {}
        self._available_symbols: Optional[List[str]] = None

    def get_available_daily_symbols(self) -> List[str]:
        """Get all symbols with daily data from any source."""
        if self._available_symbols is not None:
            return self._available_symbols

        symbols = set()

        # Yahoo symbols
        if self.yahoo_daily_dir.exists():
            for f in self.yahoo_daily_dir.glob("*.parquet"):
                # Handle symbols like ^GSPC -> _GSPC
                symbol = f.stem.replace('_', '^') if f.stem.startswith('_') else f.stem
                symbols.add(symbol)

        # Alpaca symbols
        if self.alpaca_daily_dir.exists():
            for f in self.alpaca_daily_dir.glob("*.parquet"):
                if not f.stem.endswith("_1min"):
                    symbols.add(f.stem)

        self._available_symbols = sorted(symbols)
        return self._available_symbols

    def get_available_intraday_symbols(self) -> List[str]:
        """Get all symbols with 1-minute data."""
        if not self.intraday_1min_dir.exists():
            return []
        return sorted([f.stem for f in self.intraday_1min_dir.glob("*.parquet")])

    def _load_parquet(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load and normalize a parquet file."""
        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)

            # Normalize column names
            df.columns = df.columns.str.lower()

            # Handle index vs column for timestamp
            if 'timestamp' not in df.columns:
                if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'timestamp'})

            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Remove timezone if present - use tz_convert(None) for aware datetimes
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(None)

            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None

    def load_daily(self, symbol: str, use_cache: bool = True,
                   source: Literal["auto", "yahoo", "alpaca"] = "auto") -> pd.DataFrame:
        """
        Load daily data for a symbol.

        Args:
            symbol: Stock symbol
            use_cache: Use cached data if available
            source: Data source - "auto" merges both, "yahoo" or "alpaca" for specific

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{source}"

        if use_cache and cache_key in self._daily_cache:
            return self._daily_cache[cache_key].copy()
        # Symbol aliases for common indices
        SYMBOL_ALIASES = {
            "VIX": "^VIX",
            "SPX": "^GSPC",
            "DJI": "^DJI",
            "IXIC": "^IXIC",
            "RUT": "^RUT",
        }
        symbol = SYMBOL_ALIASES.get(symbol.upper(), symbol)

        # Handle special symbols (^ -> _)
        yahoo_symbol = symbol.replace('^', '_')

        yahoo_path = self.yahoo_daily_dir / f"{yahoo_symbol}.parquet"
        alpaca_path = self.alpaca_daily_dir / f"{symbol}.parquet"

        yahoo_df = None
        alpaca_df = None

        if source in ("auto", "yahoo"):
            yahoo_df = self._load_parquet(yahoo_path)

        if source in ("auto", "alpaca"):
            alpaca_df = self._load_parquet(alpaca_path)

        # Merge or select
        if source == "auto":
            df = self._merge_daily_sources(yahoo_df, alpaca_df)
        elif source == "yahoo":
            df = yahoo_df if yahoo_df is not None else pd.DataFrame()
        else:
            df = alpaca_df if alpaca_df is not None else pd.DataFrame()

        if df is not None and not df.empty:
            self._daily_cache[cache_key] = df
            return df.copy()

        return pd.DataFrame()

    def _merge_daily_sources(self, yahoo_df: Optional[pd.DataFrame],
                             alpaca_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Intelligently merge Yahoo and Alpaca daily data.

        Yahoo provides extended history, Alpaca may have more recent data.
        """
        if yahoo_df is None and alpaca_df is None:
            return pd.DataFrame()

        if yahoo_df is None:
            return alpaca_df

        if alpaca_df is None:
            return yahoo_df

        # Both exist - merge them
        # Standardize columns
        common_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        yahoo_cols = [c for c in common_cols if c in yahoo_df.columns]
        alpaca_cols = [c for c in common_cols if c in alpaca_df.columns]

        yahoo_df = yahoo_df[yahoo_cols].copy()
        alpaca_df = alpaca_df[alpaca_cols].copy()

        # Use Yahoo as base (extended history)
        # Add any more recent Alpaca data - compare by DATE to avoid timezone duplicates
        yahoo_max_date = yahoo_df['timestamp'].dt.date.max()
        alpaca_new = alpaca_df[alpaca_df['timestamp'].dt.date > yahoo_max_date]

        if not alpaca_new.empty:
            merged = pd.concat([yahoo_df, alpaca_new], ignore_index=True)
            merged = merged.sort_values('timestamp').reset_index(drop=True)
            return merged

        return yahoo_df

    def load_intraday(self, symbol: str, timeframe: str = "1min",
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Load intraday data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Currently only "1min" supported
            use_cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        if timeframe != "1min":
            logger.warning(f"Only 1min timeframe supported, got {timeframe}")
            return pd.DataFrame()

        cache_key = f"{symbol}_1min"

        if use_cache and cache_key in self._intraday_cache:
            return self._intraday_cache[cache_key].copy()

        filepath = self.intraday_1min_dir / f"{symbol}.parquet"
        df = self._load_parquet(filepath)

        if df is not None and not df.empty:
            self._intraday_cache[cache_key] = df
            return df.copy()

        return pd.DataFrame()

    def load_all_daily(self, symbols: List[str] = None,
                       use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load daily data for all (or specified) symbols.

        Args:
            symbols: List of symbols to load (default: all available)
            use_cache: Use cached data

        Returns:
            Dict mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.get_available_daily_symbols()

        result = {}
        loaded = 0
        failed = 0

        for symbol in symbols:
            df = self.load_daily(symbol, use_cache=use_cache)
            if not df.empty:
                result[symbol] = df
                loaded += 1
            else:
                failed += 1

        logger.info(f"Loaded daily data: {loaded} succeeded, {failed} failed")
        return result

    def load_all_intraday(self, symbols: List[str] = None,
                          use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load 1-minute data for all (or specified) symbols.

        Args:
            symbols: List of symbols to load (default: all available)
            use_cache: Use cached data

        Returns:
            Dict mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.get_available_intraday_symbols()

        result = {}
        loaded = 0

        for symbol in symbols:
            df = self.load_intraday(symbol, use_cache=use_cache)
            if not df.empty:
                result[symbol] = df
                loaded += 1

        logger.info(f"Loaded 1-min data: {loaded} symbols")
        return result

    def get_date_range(self, symbol: str,
                       data_type: Literal["daily", "intraday"] = "daily") -> tuple:
        """
        Get the date range available for a symbol.

        Returns:
            (start_date, end_date) or (None, None) if no data
        """
        if data_type == "daily":
            df = self.load_daily(symbol)
        else:
            df = self.load_intraday(symbol)

        if df.empty or 'timestamp' not in df.columns:
            return (None, None)

        return (df['timestamp'].min(), df['timestamp'].max())

    def get_data_summary(self) -> Dict:
        """Get summary of available data."""
        daily_symbols = self.get_available_daily_symbols()
        intraday_symbols = self.get_available_intraday_symbols()

        # Sample date ranges
        daily_range = None
        intraday_range = None

        if daily_symbols:
            # Check SPY or first symbol
            test_sym = "SPY" if "SPY" in daily_symbols else daily_symbols[0]
            daily_range = self.get_date_range(test_sym, "daily")

        if intraday_symbols:
            test_sym = "SPY" if "SPY" in intraday_symbols else intraday_symbols[0]
            intraday_range = self.get_date_range(test_sym, "intraday")

        # Count files by source
        yahoo_count = len(list(self.yahoo_daily_dir.glob("*.parquet"))) if self.yahoo_daily_dir.exists() else 0
        alpaca_daily_count = len(list(self.alpaca_daily_dir.glob("*.parquet"))) if self.alpaca_daily_dir.exists() else 0
        intraday_count = len(list(self.intraday_1min_dir.glob("*.parquet"))) if self.intraday_1min_dir.exists() else 0

        return {
            "daily_symbols": len(daily_symbols),
            "intraday_symbols": len(intraday_symbols),
            "yahoo_files": yahoo_count,
            "alpaca_daily_files": alpaca_daily_count,
            "intraday_1min_files": intraday_count,
            "daily_date_range": daily_range,
            "intraday_date_range": intraday_range,
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._daily_cache.clear()
        self._intraday_cache.clear()
        self._available_symbols = None


# Convenience functions
def load_daily(symbol: str, **kwargs) -> pd.DataFrame:
    """Load daily data for a symbol."""
    return UnifiedDataLoader().load_daily(symbol, **kwargs)


def load_intraday(symbol: str, **kwargs) -> pd.DataFrame:
    """Load 1-minute data for a symbol."""
    return UnifiedDataLoader().load_intraday(symbol, **kwargs)


def get_data_summary() -> Dict:
    """Get summary of all available data."""
    return UnifiedDataLoader().get_data_summary()


# ============================================================================
# MAIN - Demo/Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n📂 Unified Data Loader")
    print("=" * 60)

    loader = UnifiedDataLoader()

    # Get summary
    summary = loader.get_data_summary()

    print(f"\nData Summary:")
    print(f"  Daily symbols:     {summary['daily_symbols']}")
    print(f"  Intraday symbols:  {summary['intraday_symbols']}")
    print(f"  Yahoo files:       {summary['yahoo_files']}")
    print(f"  Alpaca daily:      {summary['alpaca_daily_files']}")
    print(f"  Intraday 1-min:    {summary['intraday_1min_files']}")

    if summary['daily_date_range'][0]:
        print(f"\n  Daily range:    {summary['daily_date_range'][0].date()} to {summary['daily_date_range'][1].date()}")
    if summary['intraday_date_range'][0]:
        print(f"  Intraday range: {summary['intraday_date_range'][0].date()} to {summary['intraday_date_range'][1].date()}")

    # Test loading
    test_symbol = "SPY"
    print(f"\n📊 Testing {test_symbol}...")

    df = loader.load_daily(test_symbol)
    if not df.empty:
        years = (df['timestamp'].max() - df['timestamp'].min()).days / 365
        print(f"  Daily: {len(df):,} bars, {years:.1f} years")
        print(f"         {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    else:
        print(f"  Daily: No data")

    df_1min = loader.load_intraday(test_symbol)
    if not df_1min.empty:
        years = (df_1min['timestamp'].max() - df_1min['timestamp'].min()).days / 365
        print(f"  1-min: {len(df_1min):,} bars, {years:.1f} years")
    else:
        print(f"  1-min: No data")

    print("\n✅ Done!")
