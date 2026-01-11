#!/usr/bin/env python3
"""
Backfill Promotion Pipeline
============================
One-time script to migrate existing discovered_strategies from research.db
to strategy_lifecycle in promotion_pipeline.db.

This bridges the gap between:
- Evolution Engine → research.db.discovered_strategies (94 candidates)
- Promotion Pipeline → promotion_pipeline.db.strategy_lifecycle (empty)
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
TRADING_SYSTEM = Path(__file__).parent.parent
RESEARCH_DB = TRADING_SYSTEM / "db" / "research.db"
PIPELINE_DB = TRADING_SYSTEM / "data" / "promotion_pipeline.db"


def backfill_strategies():
    """Migrate discovered strategies to promotion pipeline."""

    # Connect to both databases
    if not RESEARCH_DB.exists():
        logger.error(f"Research DB not found: {RESEARCH_DB}")
        return 0

    # Ensure pipeline DB directory exists
    PIPELINE_DB.parent.mkdir(parents=True, exist_ok=True)

    research_conn = sqlite3.connect(RESEARCH_DB)
    research_conn.row_factory = sqlite3.Row

    pipeline_conn = sqlite3.connect(PIPELINE_DB)

    # Initialize pipeline DB schema if needed
    pipeline_conn.executescript("""
        CREATE TABLE IF NOT EXISTS strategy_lifecycle (
            strategy_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            -- Discovery metrics
            discovery_generation INTEGER,
            discovery_sharpe REAL,
            discovery_sortino REAL,
            discovery_max_drawdown REAL,
            discovery_trades INTEGER,
            deflated_sharpe REAL,

            -- Validation metrics
            walk_forward_efficiency REAL,
            monte_carlo_confidence REAL,
            validation_periods_passed INTEGER,

            -- Paper trading
            paper_start_date TEXT,
            paper_days INTEGER DEFAULT 0,
            paper_trades INTEGER DEFAULT 0,
            paper_pnl REAL DEFAULT 0,
            paper_sharpe REAL DEFAULT 0,
            paper_max_drawdown REAL DEFAULT 0,
            paper_win_rate REAL DEFAULT 0,

            -- Live trading
            live_start_date TEXT,
            live_days INTEGER DEFAULT 0,
            live_trades INTEGER DEFAULT 0,
            live_pnl REAL DEFAULT 0,
            live_sharpe REAL DEFAULT 0,
            live_max_drawdown REAL DEFAULT 0,
            current_allocation REAL DEFAULT 0,

            -- Retirement
            retired_at TEXT,
            retirement_reason TEXT,

            -- Genome/Config
            genome_json TEXT,

            -- Execution config
            run_time TEXT DEFAULT '10:00'
        );

        CREATE INDEX IF NOT EXISTS idx_lifecycle_status
        ON strategy_lifecycle(status);
    """)

    # Migration: Add run_time column if it doesn't exist (for existing DBs)
    try:
        pipeline_conn.execute("SELECT run_time FROM strategy_lifecycle LIMIT 1")
    except sqlite3.OperationalError:
        pipeline_conn.execute("ALTER TABLE strategy_lifecycle ADD COLUMN run_time TEXT DEFAULT '10:00'")

    # Get all discovered strategies
    cursor = research_conn.execute("""
        SELECT
            strategy_id,
            generation_discovered,
            oos_sharpe,
            oos_sortino,
            oos_max_drawdown,
            oos_total_trades,
            genome_json,
            created_at
        FROM discovered_strategies
        WHERE status = 'candidate'
    """)

    strategies = cursor.fetchall()
    logger.info(f"Found {len(strategies)} candidates in research.db")

    # Check existing in pipeline
    existing = set()
    for row in pipeline_conn.execute("SELECT strategy_id FROM strategy_lifecycle"):
        existing.add(row[0])

    logger.info(f"Already in promotion pipeline: {len(existing)}")

    # Insert missing strategies
    migrated = 0
    now = datetime.now().isoformat()

    for strat in strategies:
        if strat['strategy_id'] in existing:
            continue

        # Parse genome_json if it's a string containing JSON
        genome_json = strat['genome_json']
        if isinstance(genome_json, str):
            try:
                genome_data = json.loads(genome_json)
                # Re-serialize as proper JSON for storage
                genome_json = json.dumps(genome_data)
            except json.JSONDecodeError:
                pass

        # Estimate deflated_sharpe from sharpe (conservative: DSR ≈ 0.8 * Sharpe)
        # This is an approximation since we don't have the original DSR
        estimated_dsr = (strat['oos_sharpe'] or 0) * 0.8

        pipeline_conn.execute("""
            INSERT OR IGNORE INTO strategy_lifecycle (
                strategy_id, status, created_at, updated_at,
                discovery_generation, discovery_sharpe, discovery_sortino,
                discovery_max_drawdown, discovery_trades, deflated_sharpe,
                genome_json, run_time
            ) VALUES (?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?, ?, '10:00')
        """, (
            strat['strategy_id'],
            strat['created_at'] or now,
            now,
            strat['generation_discovered'],
            strat['oos_sharpe'],
            strat['oos_sortino'],
            strat['oos_max_drawdown'],
            strat['oos_total_trades'],
            estimated_dsr,
            genome_json
        ))
        migrated += 1

    pipeline_conn.commit()

    # Summary
    final_count = pipeline_conn.execute(
        "SELECT COUNT(*) FROM strategy_lifecycle"
    ).fetchone()[0]

    logger.info(f"Migrated {migrated} strategies to promotion pipeline")
    logger.info(f"Total in strategy_lifecycle: {final_count}")

    # Show strategies ready for validation
    ready_for_validation = pipeline_conn.execute("""
        SELECT COUNT(*) FROM strategy_lifecycle
        WHERE status = 'candidate'
          AND discovery_sharpe >= 0.5
          AND discovery_sortino >= 0.8
          AND discovery_max_drawdown >= -30
          AND discovery_trades >= 50
    """).fetchone()[0]

    logger.info(f"Strategies ready for CANDIDATE → VALIDATED: {ready_for_validation}")

    research_conn.close()
    pipeline_conn.close()

    return migrated


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Backfilling promotion pipeline from discovered strategies")
    logger.info("=" * 60)

    migrated = backfill_strategies()

    logger.info("=" * 60)
    logger.info(f"Done! Migrated {migrated} strategies")
    logger.info("=" * 60)
