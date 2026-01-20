#!/usr/bin/env python3
"""
Download Fundamental Data for Quality Small-Cap Value Strategy
===============================================================

This script downloads fundamental data from yfinance for use in the
Quality Small-Cap Value strategy. The strategy requires real fundamental
metrics to accurately screen for quality stocks.

USAGE:
------
    # Download fundamentals for default symbol list (Russell 2000 / small-caps)
    python scripts/download_fundamentals.py

    # Download for specific symbols
    python scripts/download_fundamentals.py --symbols AAPL MSFT GOOGL

    # Download for symbols from a file (one per line)
    python scripts/download_fundamentals.py --symbols-file data/reference/small_cap_universe.txt

    # Specify output path
    python scripts/download_fundamentals.py --output data/fundamentals/fundamentals.parquet

OUTPUT:
-------
Saves a parquet file with columns:
    - symbol: Stock ticker
    - market_cap: Market capitalization in dollars
    - roa: Return on Assets (returnOnAssets from yfinance)
    - profit_margin: Net profit margin (profitMargins from yfinance)
    - debt_to_equity: Debt-to-equity ratio (debtToEquity from yfinance)
    - roe: Return on equity (returnOnEquity from yfinance)
    - book_to_market: Book value / market cap (if available)
    - download_date: When the data was downloaded

NOTES:
------
- yfinance has rate limits; downloading 1000+ symbols takes ~10-15 minutes
- Some symbols may have missing data (handled gracefully)
- Run periodically (weekly/monthly) to keep fundamentals current
- Data is point-in-time; historical fundamentals require different source

REQUIREMENTS:
-------------
    pip install yfinance pandas pyarrow
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "fundamentals" / "fundamentals.parquet"
# Use combined universe (2467 symbols) - includes Russell 2000, S&P 500, Nasdaq 100
DEFAULT_SYMBOLS_FILE = PROJECT_ROOT / "data" / "reference" / "combined_universe_constituents.json"


# Sample small-cap symbols for testing (expand this list for production)
DEFAULT_SYMBOLS = [
    # Russell 2000 sample - diversified small-caps
    "AFRM", "APPS", "AXON", "BJ", "BLDR", "BOOT", "BWA", "CAKE", "CARG",
    "CASY", "CEIX", "CHRD", "CIEN", "CIVI", "CLF", "COKE", "CROX", "CRS",
    "CVLT", "DDS", "DINO", "DKS", "DOCS", "EAT", "ELF", "ENPH", "ESNT",
    "EXLS", "FANG", "FIZZ", "FIX", "FN", "FORM", "FRPT", "FSS", "GEF",
    "GFF", "GHC", "GIII", "GLOB", "GMS", "GNTX", "GOLF", "GPI", "GTLS",
    "GVA", "HCC", "HGV", "HLI", "HQY", "HRI", "HUBG", "IBP", "ICFI",
    "IESC", "IIPR", "ITCI", "JACK", "JBT", "JJSF", "KLIC", "KMT", "KNF",
    "KWR", "LBRT", "LCII", "LECO", "LMAT", "LNTH", "LPG", "LSTR", "MATX",
    "MC", "MCRI", "MD", "MGEE", "MGRC", "MLI", "MOD", "MTDR", "MTH",
    "MUR", "MUSA", "NAVI", "NCNO", "NEO", "NMIH", "NOMD", "NVT", "OII",
    "OLN", "ONTO", "ORA", "OSCR", "PATK", "PBH", "PEBO", "PII", "PIPR",
    "PLXS", "POWI", "PRGS", "PRIM", "PRO", "PTEN", "PUMP", "PVH", "QRVO",
]


def download_fundamentals_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Download fundamental data for a single symbol using yfinance.

    Returns dict with fundamental metrics or None if download fails.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or 'symbol' not in info:
            logger.debug(f"{symbol}: No data available")
            return None

        # Extract fundamental metrics
        # yfinance returns None for missing values, which we preserve
        fundamentals = {
            'symbol': symbol,
            'market_cap': info.get('marketCap'),
            'roa': info.get('returnOnAssets'),
            'profit_margin': info.get('profitMargins'),
            'debt_to_equity': info.get('debtToEquity'),
            'roe': info.get('returnOnEquity'),
            'book_value': info.get('bookValue'),
            'price_to_book': info.get('priceToBook'),
            'enterprise_value': info.get('enterpriseValue'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'download_date': datetime.now().isoformat(),
        }

        # Calculate book-to-market if we have the data
        # B/M = 1 / P/B
        price_to_book = info.get('priceToBook')
        if price_to_book and price_to_book > 0:
            fundamentals['book_to_market'] = 1.0 / price_to_book
        else:
            fundamentals['book_to_market'] = None

        return fundamentals

    except Exception as e:
        logger.debug(f"{symbol}: Error downloading - {e}")
        return None


def download_fundamentals_batch(
    symbols: List[str],
    delay_seconds: float = 0.1,
    progress_interval: int = 50
) -> pd.DataFrame:
    """
    Download fundamentals for a batch of symbols.

    Args:
        symbols: List of stock symbols
        delay_seconds: Delay between requests to avoid rate limiting
        progress_interval: How often to log progress

    Returns:
        DataFrame with fundamental data
    """
    results = []
    total = len(symbols)
    success_count = 0
    fail_count = 0

    logger.info(f"Downloading fundamentals for {total} symbols...")
    start_time = time.time()

    for i, symbol in enumerate(symbols):
        data = download_fundamentals_for_symbol(symbol)

        if data:
            results.append(data)
            success_count += 1
        else:
            fail_count += 1

        # Progress logging
        if (i + 1) % progress_interval == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i + 1}/{total} ({success_count} success, {fail_count} failed) "
                f"- {rate:.1f} symbols/sec - ETA: {eta:.0f}s"
            )

        # Rate limiting delay
        if delay_seconds > 0 and i < total - 1:
            time.sleep(delay_seconds)

    elapsed_total = time.time() - start_time
    logger.info(
        f"Download complete: {success_count}/{total} symbols in {elapsed_total:.1f}s "
        f"({fail_count} failed/missing)"
    )

    if not results:
        logger.error("No fundamental data downloaded!")
        return pd.DataFrame()

    return pd.DataFrame(results)


def load_symbols_from_file(filepath: Path) -> List[str]:
    """Load symbols from a text file (one symbol per line) or JSON file."""
    if not filepath.exists():
        logger.warning(f"Symbols file not found: {filepath}")
        return []

    # Handle JSON files (e.g., russell2000_constituents.json)
    if filepath.suffix.lower() == '.json':
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Handle format: {"symbols": [...]} or {"index": "...", "symbols": [...]}
        if isinstance(data, dict) and 'symbols' in data:
            symbols = [s.upper() for s in data['symbols']]
        elif isinstance(data, list):
            symbols = [s.upper() for s in data]
        else:
            logger.warning(f"Unexpected JSON format in {filepath}")
            return []
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols

    # Handle text files (one symbol per line)
    with open(filepath, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]

    logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
    return symbols


def main():
    parser = argparse.ArgumentParser(
        description="Download fundamental data for Quality Small-Cap Value strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download for default symbols
    python download_fundamentals.py

    # Download for specific symbols
    python download_fundamentals.py --symbols AAPL MSFT GOOGL

    # Download from symbols file
    python download_fundamentals.py --symbols-file universe.txt

    # Custom output path
    python download_fundamentals.py --output /path/to/output.parquet
        """
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        help='List of stock symbols to download'
    )
    parser.add_argument(
        '--symbols-file',
        type=Path,
        help='File containing symbols (one per line)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output parquet file path (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between API calls in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine symbols to download
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    elif DEFAULT_SYMBOLS_FILE.exists():
        symbols = load_symbols_from_file(DEFAULT_SYMBOLS_FILE)
        if not symbols:
            logger.info("Using default symbol list")
            symbols = DEFAULT_SYMBOLS
    else:
        logger.info("Using default symbol list")
        symbols = DEFAULT_SYMBOLS

    if not symbols:
        logger.error("No symbols to download!")
        sys.exit(1)

    logger.info(f"Will download fundamentals for {len(symbols)} symbols")

    # Download fundamentals
    df = download_fundamentals_batch(symbols, delay_seconds=args.delay)

    if df.empty:
        logger.error("No data downloaded, exiting")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Clean data before saving - handle 'Infinity' strings from yfinance
    import numpy as np
    for col in df.columns:
        if col == 'symbol':
            continue
        # Replace string 'Infinity'/'-Infinity' with NaN
        if df[col].dtype == 'object':
            df[col] = df[col].replace(['Infinity', '-Infinity', 'inf', '-inf'], np.nan)
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Save to parquet
    df.to_parquet(args.output, index=False)
    logger.info(f"Saved {len(df)} records to {args.output}")

    # Summary statistics
    logger.info("\n=== Download Summary ===")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    # Check data quality
    for col in ['market_cap', 'roa', 'profit_margin', 'debt_to_equity']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"  {col}: {non_null}/{len(df)} non-null ({100*non_null/len(df):.1f}%)")

    # Market cap distribution (for small-cap filtering)
    if 'market_cap' in df.columns:
        mc = df['market_cap'].dropna() / 1e9  # Convert to billions
        logger.info(f"\nMarket cap distribution (billions):")
        logger.info(f"  Min: ${mc.min():.2f}B")
        logger.info(f"  25%: ${mc.quantile(0.25):.2f}B")
        logger.info(f"  50%: ${mc.quantile(0.50):.2f}B")
        logger.info(f"  75%: ${mc.quantile(0.75):.2f}B")
        logger.info(f"  Max: ${mc.max():.2f}B")

        # Count small-caps ($300M - $2B)
        small_cap_mask = (df['market_cap'] >= 300e6) & (df['market_cap'] <= 2e9)
        n_small_cap = small_cap_mask.sum()
        logger.info(f"\nSmall-cap range ($300M-$2B): {n_small_cap} stocks")

    logger.info(f"\nFundamentals saved to: {args.output}")
    logger.info("Run the Quality Small-Cap Value strategy to use this data.")


if __name__ == "__main__":
    main()
