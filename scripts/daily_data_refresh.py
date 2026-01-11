#!/usr/bin/env python3
"""
Daily Data Refresh Script
=========================
Runs after market close to refresh daily data for all symbols.
Designed to be run via cron at 6:00 PM ET (market close + buffer).

Usage:
    python scripts/daily_data_refresh.py              # Refresh SP500 symbols
    python scripts/daily_data_refresh.py --full       # Refresh all cached Alpaca symbols
    python scripts/daily_data_refresh.py --universe   # Refresh ALL symbols (Yahoo universe)
    python scripts/daily_data_refresh.py --test       # Test with 5 symbols
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DIRS, DATABASES
from data.fetchers.daily_bars import DailyBarsFetcher
from data.fetchers.vix import VIXFetcher

# Configure logging
LOG_FILE = DIRS["logs"] / "data_refresh.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_sp500_symbols() -> list:
    """Load SP500 symbols from reference file."""
    import json

    # Try multiple possible file names
    possible_files = [
        DIRS["reference"] / "sp500_constituents.json",
        DIRS["reference"] / "sp500.json",
        DIRS["reference"] / "universe.json",
    ]

    for sp500_path in possible_files:
        if sp500_path.exists():
            try:
                with open(sp500_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        symbols = data
                    elif isinstance(data, dict) and "symbols" in data:
                        symbols = data["symbols"]
                    elif isinstance(data, dict):
                        # Try to extract symbols from dict keys or values
                        symbols = list(data.keys()) if all(isinstance(k, str) and len(k) <= 5 for k in list(data.keys())[:10]) else []
                    else:
                        continue

                    if symbols:
                        logger.info(f"Loaded {len(symbols)} symbols from {sp500_path.name}")
                        return symbols
            except Exception as e:
                logger.warning(f"Failed to load {sp500_path}: {e}")
                continue

    # Fallback to hardcoded major symbols
    logger.warning("Using fallback symbol list - no reference file found")
    return ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN",
            "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ", "V", "UNH"]


def get_full_universe_symbols() -> list:
    """Get all symbols from Yahoo daily data directory (full universe)."""
    yahoo_dir = DIRS.get("daily_yahoo", DIRS["historical"] / "daily_yahoo")

    if not yahoo_dir.exists():
        logger.warning("Yahoo daily directory not found, falling back to SP500")
        return get_sp500_symbols()

    symbols = []
    for f in yahoo_dir.glob("*.parquet"):
        sym = f.stem
        # Skip index symbols (start with _) - Alpaca doesn't have these
        if sym.startswith('_'):
            continue
        symbols.append(sym)

    logger.info(f"Loaded {len(symbols)} symbols from Yahoo universe")
    return sorted(symbols)


def refresh_daily_data(symbols: list = None, force: bool = False, rate_limit: bool = False) -> dict:
    """
    Refresh daily data for specified symbols.

    Args:
        symbols: List of symbols to refresh
        force: Force refetch even if data is recent
        rate_limit: Add delay between requests (for large universe)

    Returns:
        dict with success/failure counts
    """
    import time

    fetcher = DailyBarsFetcher()

    if symbols is None:
        symbols = get_sp500_symbols()

    logger.info(f"Starting data refresh for {len(symbols)} symbols")
    if rate_limit:
        logger.info("Rate limiting enabled (0.1s between requests)")

    results = {
        "total": len(symbols),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }

    for i, symbol in enumerate(symbols):
        try:
            df = fetcher.fetch_symbol(symbol, force=force)
            if df is not None and not df.empty:
                results["success"] += 1
            else:
                results["skipped"] += 1

            # Progress logging
            if (i + 1) % 100 == 0:
                pct = (i + 1) / len(symbols) * 100
                logger.info(f"Progress: {i + 1}/{len(symbols)} ({pct:.1f}%) - {results['success']} success, {results['failed']} failed")

            # Rate limiting for large universe
            if rate_limit and (i + 1) % 10 == 0:
                time.sleep(0.1)

        except Exception as e:
            results["failed"] += 1
            error_msg = str(e)[:50]
            results["errors"].append(f"{symbol}: {error_msg}")
            # Only log first few errors to avoid spam
            if results["failed"] <= 10:
                logger.warning(f"Failed to refresh {symbol}: {error_msg}")
            elif results["failed"] == 11:
                logger.warning("Suppressing further error messages...")

    return results


def refresh_vix_data() -> bool:
    """Refresh VIX data for regime detection."""
    try:
        fetcher = VIXFetcher()
        # Try Yahoo first (more reliable), fall back to proxy
        df = fetcher.fetch_from_yahoo(days=365)
        if df is None or df.empty:
            df = fetcher.fetch_vix_proxy(days=365)
        if df is not None and not df.empty:
            logger.info(f"VIX data refreshed: {len(df)} rows, current: {fetcher.get_current_vix():.2f}")
            return True
    except Exception as e:
        logger.error(f"VIX refresh failed: {e}")
    return False


def log_refresh_to_db(results: dict):
    """Log refresh results to performance database."""
    import sqlite3

    try:
        conn = sqlite3.connect(DATABASES["performance"])
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_refresh_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_symbols INTEGER,
                success INTEGER,
                failed INTEGER,
                skipped INTEGER,
                duration_sec REAL,
                errors TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO data_refresh_log
            (timestamp, total_symbols, success, failed, skipped, errors)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            results["total"],
            results["success"],
            results["failed"],
            results["skipped"],
            "; ".join(results["errors"][:10])  # First 10 errors
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log to DB: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily data refresh")
    parser.add_argument("--full", action="store_true",
                        help="Refresh all cached Alpaca symbols")
    parser.add_argument("--universe", action="store_true",
                        help="Refresh ALL symbols from Yahoo universe (~2500)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode - only 5 symbols")
    parser.add_argument("--force", action="store_true",
                        help="Force refetch even if data is recent")
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("DAILY DATA REFRESH STARTING")
    logger.info(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Determine symbols to refresh and rate limiting
    rate_limit = False
    if args.test:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
        logger.info("TEST MODE: Using 5 symbols only")
    elif args.universe:
        symbols = get_full_universe_symbols()
        rate_limit = True  # Be nice to API with large universe
        logger.info(f"UNIVERSE MODE: Refreshing all {len(symbols)} symbols from Yahoo universe")
    elif args.full:
        fetcher = DailyBarsFetcher()
        symbols = fetcher.get_available_symbols()
        logger.info(f"FULL MODE: Refreshing all {len(symbols)} cached Alpaca symbols")
    else:
        symbols = get_sp500_symbols()
        logger.info(f"STANDARD MODE: Refreshing {len(symbols)} SP500 symbols")

    # Refresh daily bars
    results = refresh_daily_data(symbols, force=args.force, rate_limit=rate_limit)

    # Refresh VIX
    vix_ok = refresh_vix_data()

    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()

    # Log results
    logger.info("=" * 60)
    logger.info("REFRESH COMPLETE")
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Success: {results['success']}/{results['total']}")
    logger.info(f"Failed: {results['failed']}, Skipped: {results['skipped']}")
    logger.info(f"VIX: {'OK' if vix_ok else 'FAILED'}")
    logger.info("=" * 60)

    # Log to database
    log_refresh_to_db(results)

    # Exit with error code if too many failures
    if results["failed"] > results["total"] * 0.1:  # >10% failed
        logger.error("Too many failures - check API connection")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
