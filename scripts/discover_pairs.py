#!/usr/bin/env python3
"""
Pair Discovery Script
=====================
Runs cointegration analysis to find valid trading pairs and populates pairs.db.

Usage:
    python scripts/discover_pairs.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import sqlite3
from datetime import datetime

from strategies.pairs_trading import PairsAnalyzer, SECTOR_STOCKS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def get_pairs_db():
    """Connect to existing pairs.db."""
    db_path = Path(__file__).parent.parent / "db" / "pairs.db"
    return sqlite3.connect(db_path)


def save_pairs_to_db(conn, all_pairs: dict):
    """Save discovered pairs to database using existing schema."""
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    saved = 0
    for sector, pairs in all_pairs.items():
        for p in pairs:
            pair_id = f"{p.stock_a}_{p.stock_b}"
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO pairs
                    (pair_id, symbol_a, symbol_b, sector, correlation, coint_pvalue,
                     half_life, hedge_ratio, spread_mean, spread_std,
                     last_tested, is_active, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ''', (
                    pair_id, p.stock_a, p.stock_b, p.sector, p.correlation,
                    p.coint_pvalue, p.half_life, p.hedge_ratio,
                    p.spread_mean, p.spread_std, now, now
                ))
                saved += 1
            except Exception as e:
                logger.warning(f"Failed to save {p.stock_a}/{p.stock_b}: {e}")

    conn.commit()
    return saved


def main():
    print("=" * 60)
    print("PAIR DISCOVERY")
    print("=" * 60)

    # Show universe
    total_stocks = sum(len(stocks) for stocks in SECTOR_STOCKS.values())
    print(f"\nUniverse: {len(SECTOR_STOCKS)} sectors, {total_stocks} stocks")

    # Initialize analyzer
    print("\nLoading market data...")
    analyzer = PairsAnalyzer()

    # Load data first
    analyzer.data_mgr.load_all()

    # Find pairs
    print("\nScanning for cointegrated pairs...")
    print("Criteria: correlation > 0.8, p-value < 0.05, half-life 5-30 days")
    print("-" * 60)

    all_pairs = analyzer.find_all_pairs(max_per_sector=5)

    # Summary
    print("-" * 60)
    total_pairs = sum(len(pairs) for pairs in all_pairs.values())
    print(f"\nDISCOVERY RESULTS:")
    print(f"  Sectors with pairs: {len(all_pairs)}")
    print(f"  Total pairs found: {total_pairs}")

    if total_pairs > 0:
        print("\nPAIRS FOUND:")
        for sector, pairs in all_pairs.items():
            print(f"\n  {sector}:")
            for p in pairs:
                print(f"    {p.stock_a}/{p.stock_b}: "
                      f"corr={p.correlation:.2f}, p={p.coint_pvalue:.4f}, "
                      f"half-life={p.half_life:.1f}d")

        # Save to database
        print("\nSaving to pairs.db...")
        conn = get_pairs_db()
        saved = save_pairs_to_db(conn, all_pairs)
        conn.close()
        print(f"Saved {saved} pairs to database")
    else:
        print("\nNo pairs found with current criteria.")
        print("\nPossible actions:")
        print("  1. Loosen correlation threshold (0.7 instead of 0.8)")
        print("  2. Expand half-life range (3-60 days instead of 5-30)")
        print("  3. Add more stocks to SECTOR_STOCKS")
        print("  4. Disable pairs_trading strategy")

    print("\n" + "=" * 60)
    return total_pairs


if __name__ == "__main__":
    pairs_found = main()
    sys.exit(0 if pairs_found > 0 else 1)
