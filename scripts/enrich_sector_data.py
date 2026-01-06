"""
Sector/Industry Data Enrichment
================================
Fetches sector and industry data from Yahoo Finance for stocks missing this info.

Usage:
    python scripts/enrich_sector_data.py
    python scripts/enrich_sector_data.py --batch-size 100
    python scripts/enrich_sector_data.py --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Sector type mapping
SECTOR_TYPES = {
    'Technology': 'growth',
    'Healthcare': 'defensive',
    'Financial Services': 'cyclical',
    'Financials': 'cyclical',
    'Consumer Cyclical': 'cyclical',
    'Consumer Defensive': 'defensive',
    'Industrials': 'cyclical',
    'Energy': 'cyclical',
    'Utilities': 'defensive',
    'Real Estate': 'rate_sensitive',
    'Basic Materials': 'cyclical',
    'Communication Services': 'growth',
}


def fetch_yahoo_info(symbol: str) -> Optional[Dict]:
    """Fetch sector/industry info from Yahoo Finance."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info:
            return None

        sector = info.get('sector')
        industry = info.get('industry')

        if not sector and not industry:
            return None

        return {
            'sector': sector,
            'industry': industry,
            'sector_type': SECTOR_TYPES.get(sector, 'other'),
            'market_cap': info.get('marketCap'),
            'source': 'yahoo_finance',
            'fetched_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.debug(f"Failed to fetch {symbol}: {e}")
        return None


def enrich_sector_data(batch_size: int = 50, dry_run: bool = False, delay: float = 0.2):
    """Enrich stock characteristics with sector/industry data."""

    char_path = Path("data/reference/stock_characteristics.json")

    if not char_path.exists():
        logger.error(f"Stock characteristics file not found: {char_path}")
        return

    # Load current data
    with open(char_path) as f:
        data = json.load(f)

    chars = data.get('characteristics', data)

    # Find stocks missing sector data
    missing = []
    for symbol, info in chars.items():
        if not info.get('sector'):
            missing.append(symbol)

    logger.info(f"Total stocks: {len(chars)}")
    logger.info(f"Missing sector data: {len(missing)}")

    if not missing:
        logger.info("All stocks have sector data!")
        return

    if dry_run:
        logger.info(f"DRY RUN: Would fetch data for {len(missing)} stocks")
        logger.info(f"First 20: {missing[:20]}")
        return

    # Fetch in batches
    updated = 0
    failed = 0
    batch = missing[:batch_size]

    logger.info(f"Fetching sector data for {len(batch)} stocks...")

    for i, symbol in enumerate(batch):
        if i > 0 and i % 10 == 0:
            logger.info(f"Progress: {i}/{len(batch)} (updated: {updated}, failed: {failed})")

        info = fetch_yahoo_info(symbol)

        if info:
            # Update the record
            if symbol in chars:
                chars[symbol]['sector'] = info['sector']
                chars[symbol]['industry'] = info['industry']
                chars[symbol]['sector_type'] = info['sector_type']
                chars[symbol]['sector_source'] = 'yahoo_finance'
                chars[symbol]['sector_updated'] = info['fetched_at']
                updated += 1
        else:
            failed += 1

        # Rate limiting
        time.sleep(delay)

    # Save updated data
    if updated > 0:
        # Handle both formats
        if 'characteristics' in data:
            data['characteristics'] = chars
        else:
            data = chars

        with open(char_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {updated} updates to {char_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SECTOR ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Stocks processed: {len(batch)}")
    print(f"Successfully updated: {updated}")
    print(f"Failed/skipped: {failed}")
    print(f"Remaining without sector: {len(missing) - updated}")
    print("=" * 60)

    # Show sector distribution after update
    sectors = {}
    for v in chars.values():
        s = v.get('sector') or 'Unknown'
        sectors[s] = sectors.get(s, 0) + 1

    print("\nUpdated sector distribution:")
    for s, c in sorted(sectors.items(), key=lambda x: -x[1])[:12]:
        print(f"  {s}: {c}")

    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich sector data from Yahoo Finance")
    parser.add_argument('--batch-size', type=int, default=100, help='Number of stocks to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between API calls (seconds)')

    args = parser.parse_args()

    enrich_sector_data(batch_size=args.batch_size, dry_run=args.dry_run, delay=args.delay)
