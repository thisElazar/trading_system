"""
VGP Migration
=============
Migrates from scalar GP to Vectorial GP (VGP) primitives.

Phase 0: Archive existing scalar strategies
Phase 1: VGP primitives are in gp_core.py
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.storage.db_manager import get_db

logger = logging.getLogger(__name__)


def archive_scalar_tables():
    """
    Archive existing scalar GP tables before VGP migration.

    Renames tables with _v1_scalar suffix (non-destructive).
    Creates fresh v2 tables with genome_version field.
    """
    db = get_db()
    conn = db._get_connection("research")
    cursor = conn.cursor()

    tables_to_archive = [
        'discovered_strategies',
        'evolution_checkpoints',
        'evolution_history'
    ]

    archived_count = 0

    for table in tables_to_archive:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        if not cursor.fetchone():
            logger.info(f"Table {table} does not exist, skipping")
            continue

        # Check if already archived
        archive_name = f"{table}_v1_scalar"
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (archive_name,)
        )
        if cursor.fetchone():
            logger.warning(f"Archive {archive_name} already exists, skipping")
            continue

        # Archive by renaming
        logger.info(f"Archiving {table} -> {archive_name}")
        cursor.execute(f"ALTER TABLE {table} RENAME TO {archive_name}")
        archived_count += 1

    conn.commit()
    logger.info(f"Archived {archived_count} tables")

    return archived_count


def create_vgp_tables():
    """
    Create fresh VGP tables with genome_version field.
    """
    db = get_db()
    conn = db._get_connection("research")
    cursor = conn.cursor()

    # Evolution checkpoints for VGP (with genome_version)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evolution_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_id TEXT NOT NULL UNIQUE,
            generation INTEGER NOT NULL,
            genome_version INTEGER DEFAULT 2,
            population_json TEXT NOT NULL,
            pareto_front_json TEXT,
            novelty_archive_json TEXT,
            config_json TEXT,
            regime_state TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Discovered strategies for VGP (with genome_version)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS discovered_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL UNIQUE,
            genome_version INTEGER DEFAULT 2,
            genome_json TEXT NOT NULL,
            generation_discovered INTEGER,

            -- Performance metrics (out-of-sample)
            oos_sharpe REAL,
            oos_sortino REAL,
            oos_max_drawdown REAL,
            oos_total_trades INTEGER,
            oos_win_rate REAL,

            -- Behavioral characteristics
            behavior_vector TEXT,
            novelty_score REAL,

            -- Status tracking
            status TEXT DEFAULT 'candidate',
            validation_date TEXT,
            deployment_date TEXT,
            retirement_date TEXT,
            retirement_reason TEXT,

            -- Generated code
            python_code TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Evolution history for VGP (with genome_version)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evolution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            genome_version INTEGER DEFAULT 2,

            -- Population statistics
            pop_size INTEGER,
            pareto_front_size INTEGER,
            novelty_archive_size INTEGER,

            -- Fitness statistics
            best_sortino REAL,
            avg_sortino REAL,
            best_drawdown REAL,
            avg_novelty REAL,

            -- Diversity metrics
            behavior_diversity REAL,
            genome_diversity REAL,

            -- Discoveries this generation
            new_pareto_solutions INTEGER,
            strategies_validated INTEGER,

            regime TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_generation ON evolution_checkpoints(generation)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_version ON evolution_checkpoints(genome_version)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_status ON discovered_strategies(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_version ON discovered_strategies(genome_version)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_run ON evolution_history(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_version ON evolution_history(genome_version)")

    conn.commit()
    logger.info("VGP tables created successfully")


def run_vgp_migration():
    """
    Run the complete VGP migration.

    1. Archive existing scalar tables (non-destructive)
    2. Create fresh VGP tables with genome_version field
    """
    logger.info("=" * 60)
    logger.info("VGP Migration: Scalar GP -> Vectorial GP")
    logger.info("=" * 60)

    # Phase 0: Archive
    logger.info("\n[Phase 0] Archiving scalar tables...")
    archived = archive_scalar_tables()

    # Create new tables
    logger.info("\n[Phase 0] Creating VGP tables...")
    create_vgp_tables()

    logger.info("\n" + "=" * 60)
    logger.info("VGP Migration complete!")
    logger.info(f"  - Archived {archived} scalar tables (with _v1_scalar suffix)")
    logger.info("  - Created fresh VGP tables with genome_version=2")
    logger.info("=" * 60)

    return True


def rollback_vgp_migration():
    """
    Rollback VGP migration by restoring scalar tables.

    WARNING: This drops VGP tables and restores scalar tables.
    """
    db = get_db()
    conn = db._get_connection("research")
    cursor = conn.cursor()

    tables = [
        'discovered_strategies',
        'evolution_checkpoints',
        'evolution_history'
    ]

    for table in tables:
        # Drop VGP table
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

        # Restore scalar table
        archive_name = f"{table}_v1_scalar"
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (archive_name,)
        )
        if cursor.fetchone():
            logger.info(f"Restoring {archive_name} -> {table}")
            cursor.execute(f"ALTER TABLE {archive_name} RENAME TO {table}")

    conn.commit()
    logger.info("VGP migration rolled back")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description="VGP Migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback VGP migration")
    args = parser.parse_args()

    if args.rollback:
        print("Rolling back VGP migration...")
        rollback_vgp_migration()
    else:
        print("Running VGP migration...")
        run_vgp_migration()
