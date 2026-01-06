"""
Database Schema Extensions
==========================
Extends research.db with tables for GP-based strategy discovery.
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.storage.db_manager import get_db

logger = logging.getLogger(__name__)


def migrate_discovery_tables():
    """
    Add strategy discovery tables to research.db.

    Tables:
    - evolution_checkpoints: State persistence for continuous evolution
    - discovered_strategies: Strategies promoted from evolution
    - evolution_history: Per-generation statistics
    """
    db = get_db()
    conn = db._get_connection("research")
    cursor = conn.cursor()

    # Evolution checkpoints for continuous operation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evolution_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_id TEXT NOT NULL UNIQUE,
            generation INTEGER NOT NULL,
            population_json TEXT NOT NULL,
            pareto_front_json TEXT,
            novelty_archive_json TEXT,
            config_json TEXT,
            regime_state TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Discovered strategies (promoted from evolution)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS discovered_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL UNIQUE,
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

    # Evolution history for analysis
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evolution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            generation INTEGER NOT NULL,

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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_status ON discovered_strategies(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_run ON evolution_history(run_id)")

    conn.commit()
    logger.info("Discovery tables migrated successfully")


def check_tables_exist() -> bool:
    """Check if discovery tables exist."""
    db = get_db()

    try:
        result = db.fetchone(
            "research",
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evolution_checkpoints'"
        )
        return result is not None
    except Exception:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Checking if tables exist...")
    if check_tables_exist():
        print("Discovery tables already exist")
    else:
        print("Creating discovery tables...")
        migrate_discovery_tables()
        print("Done!")
