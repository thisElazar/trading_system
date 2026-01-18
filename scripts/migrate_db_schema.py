#!/usr/bin/env python3
"""
Database Schema Migration: Variable Standardization
====================================================
Renames database columns to match canonical naming conventions.

Changes:
- trades.db:
  - trades.pnl_percent -> pnl_pct
  - trades.strategy -> strategy_id (future consideration)
  - signal_history.signal_strength -> strength
  - signal_history.signal_type -> side
  - signal_history.strategy -> strategy_id
  - strategy_performance.strategy_name -> strategy_id

- signal_scores.db:
  - signal_history.signal_strength -> strength
  - signal_history.signal_type -> side
  - signal_history.strategy -> strategy_id

Usage:
    python scripts/migrate_db_schema.py --dry-run    # Preview changes
    python scripts/migrate_db_schema.py              # Execute migration
    python scripts/migrate_db_schema.py --rollback   # Restore from backup
"""

import sqlite3
import shutil
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASES

BACKUP_DIR = Path(__file__).parent.parent / "backups" / "db_migration"


def backup_database(db_path: Path, backup_dir: Path) -> Path:
    """Create backup of database."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}_{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    print(f"  Backed up: {db_path} -> {backup_path}")
    return backup_path


def rename_column(conn: sqlite3.Connection, table: str, old_name: str, new_name: str, dry_run: bool = False):
    """Rename a column using ALTER TABLE (SQLite 3.25+)."""
    sql = f'ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}'
    if dry_run:
        print(f"  [DRY RUN] Would execute: {sql}")
    else:
        try:
            conn.execute(sql)
            print(f"  Renamed: {table}.{old_name} -> {new_name}")
        except sqlite3.OperationalError as e:
            if "no such column" in str(e).lower():
                print(f"  Skipped: {table}.{old_name} (column doesn't exist)")
            elif "duplicate column name" in str(e).lower():
                print(f"  Skipped: {table}.{old_name} (target column already exists)")
            else:
                raise


def update_index(conn: sqlite3.Connection, table: str, old_col: str, new_col: str, dry_run: bool = False):
    """Update index after column rename."""
    # Get existing indexes
    cursor = conn.execute(f"SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='{table}'")
    for idx_name, idx_sql in cursor.fetchall():
        if idx_sql and old_col in idx_sql:
            # Drop old index and create new one
            new_sql = idx_sql.replace(old_col, new_col)
            new_idx_name = idx_name.replace(old_col, new_col) if old_col in idx_name else idx_name

            if dry_run:
                print(f"  [DRY RUN] Would recreate index: {idx_name} -> {new_idx_name}")
            else:
                conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
                conn.execute(new_sql.replace(idx_name, new_idx_name))
                print(f"  Recreated index: {idx_name} -> {new_idx_name}")


def migrate_trades_db(dry_run: bool = False):
    """Migrate trades.db schema."""
    db_path = DATABASES.get("trades")
    if not db_path or not db_path.exists():
        print(f"trades.db not found at {db_path}")
        return False

    print(f"\nMigrating trades.db ({db_path})...")

    if not dry_run:
        backup_database(db_path, BACKUP_DIR)

    conn = sqlite3.connect(db_path)
    try:
        # trades table
        rename_column(conn, "trades", "pnl_percent", "pnl_pct", dry_run)

        # signal_history table (if exists in trades.db)
        rename_column(conn, "signal_history", "signal_strength", "strength", dry_run)
        rename_column(conn, "signal_history", "signal_type", "side", dry_run)
        rename_column(conn, "signal_history", "strategy", "strategy_id", dry_run)
        update_index(conn, "signal_history", "strategy", "strategy_id", dry_run)

        # strategy_performance table
        rename_column(conn, "strategy_performance", "strategy_name", "strategy_id", dry_run)

        if not dry_run:
            conn.commit()
            print("  Committed changes to trades.db")

        return True
    except Exception as e:
        print(f"  Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def migrate_signal_scores_db(dry_run: bool = False):
    """Migrate signal_scores.db schema."""
    db_path = DATABASES.get("signal_scores")
    if not db_path or not db_path.exists():
        print(f"signal_scores.db not found at {db_path}")
        return False

    print(f"\nMigrating signal_scores.db ({db_path})...")

    if not dry_run:
        backup_database(db_path, BACKUP_DIR)

    conn = sqlite3.connect(db_path)
    try:
        # signal_history table
        rename_column(conn, "signal_history", "signal_strength", "strength", dry_run)
        rename_column(conn, "signal_history", "signal_type", "side", dry_run)
        rename_column(conn, "signal_history", "strategy", "strategy_id", dry_run)

        # Update indexes
        update_index(conn, "signal_history", "strategy", "strategy_id", dry_run)

        if not dry_run:
            conn.commit()
            print("  Committed changes to signal_scores.db")

        return True
    except Exception as e:
        print(f"  Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def rollback(db_name: str):
    """Restore database from most recent backup."""
    backup_files = sorted(BACKUP_DIR.glob(f"{db_name}_*.db"), reverse=True)
    if not backup_files:
        print(f"No backup found for {db_name}")
        return False

    latest_backup = backup_files[0]
    target_path = DATABASES.get(db_name.replace(".db", ""))

    if not target_path:
        print(f"Unknown database: {db_name}")
        return False

    print(f"Restoring {target_path} from {latest_backup}")
    shutil.copy2(latest_backup, target_path)
    print("  Restored successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate database schema for variable standardization")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    parser.add_argument("--rollback", type=str, help="Rollback specific database (trades or signal_scores)")
    args = parser.parse_args()

    if args.rollback:
        return rollback(args.rollback)

    print("=" * 60)
    print("Database Schema Migration: Variable Standardization")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    success = True
    success &= migrate_trades_db(args.dry_run)
    success &= migrate_signal_scores_db(args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")
    elif success:
        print("Migration complete. Backups saved to:", BACKUP_DIR)
    else:
        print("Migration completed with errors. Check output above.")
    print("=" * 60)

    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
