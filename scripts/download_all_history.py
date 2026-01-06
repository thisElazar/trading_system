#!/usr/bin/env python3
"""
Download All Historical Data
============================
Downloads:
1. Daily candles from Yahoo Finance (1990s - present) - FREE
2. 1-minute candles from Alpaca (2016 - present) - FREE with API key

Usage:
    python scripts/download_all_history.py --daily          # Daily only
    python scripts/download_all_history.py --intraday       # 1-min only
    python scripts/download_all_history.py --all            # Both
    python scripts/download_all_history.py --symbols SPY QQQ  # Specific symbols
    python scripts/download_all_history.py --resume         # Resume interrupted download
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import threading

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

from config import (
    DIRS, DATA_ROOT,
    ALPACA_API_KEY, ALPACA_SECRET_KEY
)

# New directories for expanded data
DAILY_YAHOO_DIR = DIRS["historical"] / "daily_yahoo"
INTRADAY_1MIN_DIR = DIRS["historical"] / "intraday_1min"
PROGRESS_FILE = DATA_ROOT / "temp" / "download_progress.json"

# Ensure directories exist
DAILY_YAHOO_DIR.mkdir(parents=True, exist_ok=True)
INTRADAY_1MIN_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Download settings
YAHOO_START_DATE = "1990-01-01"  # Yahoo has data back to ~1990 for many stocks
ALPACA_1MIN_START = datetime(2016, 1, 1)  # Alpaca 1-min starts 2016
BATCH_SIZE = 50  # Symbols per batch for Alpaca
MAX_WORKERS_YAHOO = 10  # Parallel downloads for Yahoo
MAX_WORKERS_ALPACA = 3  # Lower for Alpaca rate limits

# Rate limiting
_rate_lock = threading.Lock()
_last_alpaca_call = 0


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress() -> Dict:
    """Load download progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.debug(f"Failed to load progress file: {e}")
    return {"daily_completed": [], "intraday_completed": [], "daily_failed": [], "intraday_failed": []}


def save_progress(progress: Dict):
    """Save download progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def clear_progress():
    """Clear progress file."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


# ============================================================================
# SYMBOL LOADING
# ============================================================================

def load_all_symbols() -> List[str]:
    """Load all unique symbols from reference files."""
    symbols = set()

    # Load from all constituent files
    ref_dir = DIRS["reference"]
    for json_file in ref_dir.glob("*_constituents.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if "symbols" in data:
                    symbols.update(data["symbols"])
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    # Load universe.json
    universe_file = ref_dir / "universe.json"
    if universe_file.exists():
        try:
            with open(universe_file) as f:
                data = json.load(f)
                if "symbols" in data:
                    symbols.update(data["symbols"])
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.debug(f"Failed to load universe.json: {e}")

    # Load core universe
    core_file = ref_dir / "core_universe.json"
    if core_file.exists():
        try:
            with open(core_file) as f:
                data = json.load(f)
                for key, values in data.items():
                    if isinstance(values, list):
                        symbols.update(values)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.debug(f"Failed to load core_universe.json: {e}")

    # Add essential ETFs
    essential = ["SPY", "QQQ", "IWM", "DIA", "VIX", "^VIX", "^GSPC", "^IXIC", "^DJI"]
    symbols.update(essential)

    # Clean symbols (remove any with special chars except ^)
    cleaned = []
    for s in symbols:
        s = s.strip().upper()
        if s and len(s) <= 10:
            cleaned.append(s)

    return sorted(set(cleaned))


# ============================================================================
# YAHOO FINANCE DOWNLOADER
# ============================================================================

def download_yahoo_daily(symbol: str, start_date: str = YAHOO_START_DATE,
                         delay: float = 0.5) -> Tuple[str, bool, str]:
    """
    Download daily data from Yahoo Finance.

    Returns:
        (symbol, success, message)
    """
    # Rate limiting delay
    time.sleep(delay)

    try:
        import yfinance as yf
    except ImportError:
        return (symbol, False, "yfinance not installed. Run: pip install yfinance")

    output_path = DAILY_YAHOO_DIR / f"{symbol.replace('^', '_')}.parquet"

    try:
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, auto_adjust=True)

        if df.empty:
            return (symbol, False, "No data returned")

        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Keep essential columns
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]

        # Add timestamp column from index
        df = df.reset_index()
        df = df.rename(columns={'Date': 'timestamp', 'index': 'timestamp'})
        if 'timestamp' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Remove timezone if present
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df = df.set_index('timestamp')

        # Save to parquet
        df.to_parquet(output_path)

        years = (df.index.max() - df.index.min()).days / 365
        return (symbol, True, f"{len(df):,} bars, {years:.1f} years ({df.index.min().date()} to {df.index.max().date()})")

    except Exception as e:
        return (symbol, False, str(e)[:100])


def download_all_yahoo_daily(symbols: List[str], progress: Dict, max_workers: int = MAX_WORKERS_YAHOO,
                             retry_failed: bool = False):
    """Download daily data for all symbols from Yahoo Finance."""

    # Filter out already completed
    completed = set(progress.get("daily_completed", []))

    if retry_failed:
        # Retry only failed symbols
        failed_set = set(progress.get("daily_failed", []))
        to_download = [s for s in symbols if s in failed_set]
        logger.info(f"Retrying {len(to_download)} previously failed symbols...")
    else:
        to_download = [s for s in symbols if s not in completed]

    if not to_download:
        logger.info("All daily data already downloaded!")
        return

    logger.info(f"Downloading daily data for {len(to_download)} symbols from Yahoo Finance...")
    logger.info(f"(Skipping {len(completed)} already completed)")
    logger.info(f"Using {max_workers} workers with 0.5s delay per request")

    success_count = 0
    fail_count = 0

    # Use sequential for rate limiting (parallel causes rate limits)
    if max_workers <= 1:
        for i, symbol in enumerate(to_download):
            sym, success, message = download_yahoo_daily(symbol, delay=0.5)

            if success:
                success_count += 1
                progress["daily_completed"].append(sym)
                if sym in progress.get("daily_failed", []):
                    progress["daily_failed"].remove(sym)
                logger.info(f"[{i+1}/{len(to_download)}] {sym}: {message}")
            else:
                fail_count += 1
                if sym not in progress.get("daily_failed", []):
                    progress.setdefault("daily_failed", []).append(sym)
                logger.warning(f"[{i+1}/{len(to_download)}] {sym}: FAILED - {message}")

            # Save progress every 50 symbols
            if (i + 1) % 50 == 0:
                save_progress(progress)
    else:
        # Parallel with rate limiting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_yahoo_daily, sym, YAHOO_START_DATE, 1.0): sym for sym in to_download}

            for i, future in enumerate(as_completed(futures)):
                symbol, success, message = future.result()

                if success:
                    success_count += 1
                    progress["daily_completed"].append(symbol)
                    if symbol in progress.get("daily_failed", []):
                        progress["daily_failed"].remove(symbol)
                    logger.info(f"[{i+1}/{len(to_download)}] {symbol}: {message}")
                else:
                    fail_count += 1
                    if symbol not in progress.get("daily_failed", []):
                        progress.setdefault("daily_failed", []).append(symbol)
                    logger.warning(f"[{i+1}/{len(to_download)}] {symbol}: FAILED - {message}")

                # Save progress every 50 symbols
                if (i + 1) % 50 == 0:
                    save_progress(progress)

    save_progress(progress)
    logger.info(f"\nYahoo Daily Download Complete: {success_count} succeeded, {fail_count} failed")


# ============================================================================
# ALPACA 1-MINUTE DOWNLOADER
# ============================================================================

def rate_limit_alpaca(min_interval: float = 0.2):
    """Rate limit Alpaca API calls."""
    global _last_alpaca_call
    with _rate_lock:
        elapsed = time.time() - _last_alpaca_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_alpaca_call = time.time()


def download_alpaca_1min(symbol: str, client, start_date: datetime = ALPACA_1MIN_START) -> Tuple[str, bool, str]:
    """
    Download 1-minute data from Alpaca.

    Downloads in chunks to handle large data volumes.

    Returns:
        (symbol, success, message)
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    output_path = INTRADAY_1MIN_DIR / f"{symbol}.parquet"

    try:
        end_date = datetime.now()
        all_bars = []

        # Download in 6-month chunks to avoid timeouts
        chunk_start = start_date
        chunk_size = timedelta(days=180)

        while chunk_start < end_date:
            chunk_end = min(chunk_start + chunk_size, end_date)

            rate_limit_alpaca()

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=chunk_start,
                end=chunk_end
            )

            bars = client.get_stock_bars(request)

            if not bars.df.empty:
                df = bars.df.reset_index()
                if 'symbol' in df.columns:
                    df = df[df['symbol'] == symbol].drop(columns=['symbol'])
                all_bars.append(df)

            chunk_start = chunk_end

        if not all_bars:
            return (symbol, False, "No data returned")

        # Combine all chunks
        df = pd.concat(all_bars, ignore_index=True)

        # Standardize
        df = df.rename(columns={'timestamp': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Remove timezone
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')

        # Save to parquet
        df.to_parquet(output_path)

        years = (df.index.max() - df.index.min()).days / 365
        return (symbol, True, f"{len(df):,} bars, {years:.1f} years")

    except Exception as e:
        return (symbol, False, str(e)[:100])


def download_all_alpaca_1min(symbols: List[str], progress: Dict, max_workers: int = MAX_WORKERS_ALPACA):
    """Download 1-minute data for all symbols from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        return

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # Filter out already completed and index symbols (Yahoo-only)
    completed = set(progress.get("intraday_completed", []))
    to_download = [s for s in symbols if s not in completed and not s.startswith('^')]

    if not to_download:
        logger.info("All intraday data already downloaded!")
        return

    logger.info(f"Downloading 1-minute data for {len(to_download)} symbols from Alpaca...")
    logger.info(f"(Skipping {len(completed)} already completed)")
    logger.info(f"Date range: {ALPACA_1MIN_START.date()} to {datetime.now().date()}")
    logger.info(f"Estimated time: {len(to_download) * 2} - {len(to_download) * 5} minutes")

    success_count = 0
    fail_count = 0

    # Sequential for now due to rate limits (can parallelize with care)
    for i, symbol in enumerate(to_download):
        symbol_result, success, message = download_alpaca_1min(symbol, client)

        if success:
            success_count += 1
            progress["intraday_completed"].append(symbol)
            if symbol in progress.get("intraday_failed", []):
                progress["intraday_failed"].remove(symbol)
            logger.info(f"[{i+1}/{len(to_download)}] {symbol}: {message}")
        else:
            fail_count += 1
            if symbol not in progress.get("intraday_failed", []):
                progress.setdefault("intraday_failed", []).append(symbol)
            logger.warning(f"[{i+1}/{len(to_download)}] {symbol}: FAILED - {message}")

        # Save progress every 10 symbols
        if (i + 1) % 10 == 0:
            save_progress(progress)
            logger.info(f"  Progress saved. {success_count} succeeded, {fail_count} failed so far.")

    save_progress(progress)
    logger.info(f"\nAlpaca 1-Minute Download Complete: {success_count} succeeded, {fail_count} failed")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download all historical data")
    parser.add_argument("--daily", action="store_true", help="Download daily data from Yahoo Finance")
    parser.add_argument("--intraday", action="store_true", help="Download 1-minute data from Alpaca")
    parser.add_argument("--all", action="store_true", help="Download both daily and intraday")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only previously failed symbols")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1 for rate limiting)")
    parser.add_argument("--clear-progress", action="store_true", help="Clear progress and start fresh")
    parser.add_argument("--status", action="store_true", help="Show download status")

    args = parser.parse_args()

    # Handle progress
    if args.clear_progress:
        clear_progress()
        logger.info("Progress cleared.")
        return

    # Always load progress for resume or retry-failed
    if args.resume or args.retry_failed:
        progress = load_progress()
    else:
        progress = {"daily_completed": [], "intraday_completed": [], "daily_failed": [], "intraday_failed": []}

    # Status check
    if args.status:
        print(f"\nDownload Status:")
        print(f"  Daily completed: {len(progress.get('daily_completed', []))}")
        print(f"  Daily failed: {len(progress.get('daily_failed', []))}")
        print(f"  Intraday completed: {len(progress.get('intraday_completed', []))}")
        print(f"  Intraday failed: {len(progress.get('intraday_failed', []))}")
        print(f"\nFiles on disk:")
        print(f"  Daily Yahoo: {len(list(DAILY_YAHOO_DIR.glob('*.parquet')))} files")
        print(f"  Intraday 1min: {len(list(INTRADAY_1MIN_DIR.glob('*.parquet')))} files")
        return

    # Get symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = load_all_symbols()

    logger.info(f"Loaded {len(symbols)} symbols")

    # Default to --all if nothing specified (unless retrying)
    if not (args.daily or args.intraday or args.all or args.retry_failed):
        args.all = True

    # Download daily
    if args.daily or args.all or args.retry_failed:
        try:
            import yfinance
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            logger.info("Skipping daily download...")
        else:
            download_all_yahoo_daily(symbols, progress, max_workers=args.workers,
                                    retry_failed=args.retry_failed)

    # Download intraday (skip if only retrying daily failures)
    if (args.intraday or args.all) and not args.retry_failed:
        download_all_alpaca_1min(symbols, progress)

    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Daily (Yahoo):    {len(progress.get('daily_completed', [])):,} completed, {len(progress.get('daily_failed', [])):,} failed")
    print(f"Intraday (Alpaca): {len(progress.get('intraday_completed', [])):,} completed, {len(progress.get('intraday_failed', [])):,} failed")
    print(f"\nData locations:")
    print(f"  Daily:    {DAILY_YAHOO_DIR}")
    print(f"  Intraday: {INTRADAY_1MIN_DIR}")
    print(f"\nTo resume interrupted download: python {__file__} --resume")


if __name__ == "__main__":
    main()
