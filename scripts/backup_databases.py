#!/usr/bin/env python3
"""
Database Backup Script
======================
Backs up all SQLite databases to a timestamped backup directory.

Usage:
    python scripts/backup_databases.py              # Daily backup
    python scripts/backup_databases.py --full       # Full backup with verification
    python scripts/backup_databases.py --list       # List existing backups
    python scripts/backup_databases.py --restore BACKUP_DIR  # Restore from backup

Recommended: Run daily via cron or systemd timer.
    0 5 * * * cd /home/thiselazar/trading_system && venv/bin/python scripts/backup_databases.py
"""

import sys
import shutil
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS, DATABASES

# Backup configuration
BACKUP_DIR = DIRS.get('data_root', Path('.')) / 'backups' / 'databases'
MAX_BACKUPS = 14  # Keep 2 weeks of daily backups


def get_backup_path() -> Path:
    """Get timestamped backup directory path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return BACKUP_DIR / timestamp


def backup_database(db_path: Path, backup_dir: Path, verify: bool = False) -> bool:
    """
    Backup a single database file.

    Args:
        db_path: Path to source database
        backup_dir: Destination directory
        verify: If True, verify backup integrity

    Returns:
        True if backup successful
    """
    if not db_path.exists():
        print(f"  SKIP: {db_path.name} (not found)")
        return True

    dest_path = backup_dir / db_path.name

    try:
        # Use SQLite backup API for consistent backup (handles WAL mode)
        source_conn = sqlite3.connect(db_path)
        dest_conn = sqlite3.connect(dest_path)

        with dest_conn:
            source_conn.backup(dest_conn)

        source_conn.close()
        dest_conn.close()

        # Get file sizes
        source_size = db_path.stat().st_size
        dest_size = dest_path.stat().st_size

        # Verify if requested
        if verify:
            try:
                conn = sqlite3.connect(dest_path)
                conn.execute("PRAGMA integrity_check")
                conn.close()
                print(f"  OK: {db_path.name} ({source_size:,} bytes, verified)")
            except sqlite3.Error as e:
                print(f"  WARN: {db_path.name} backup may be corrupt: {e}")
                return False
        else:
            print(f"  OK: {db_path.name} ({source_size:,} bytes)")

        return True

    except Exception as e:
        print(f"  FAIL: {db_path.name} - {e}")
        return False


def run_backup(full: bool = False) -> bool:
    """
    Run database backup.

    Args:
        full: If True, verify each backup

    Returns:
        True if all backups successful
    """
    backup_dir = get_backup_path()
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backing up databases to: {backup_dir}")
    print("-" * 50)

    success = True
    for name, db_path in DATABASES.items():
        if not backup_database(db_path, backup_dir, verify=full):
            success = False

    print("-" * 50)

    if success:
        print(f"Backup complete: {backup_dir}")
        cleanup_old_backups()
    else:
        print("Backup completed with errors")

    return success


def cleanup_old_backups():
    """Remove backups older than MAX_BACKUPS days."""
    if not BACKUP_DIR.exists():
        return

    backups = sorted(BACKUP_DIR.iterdir(), reverse=True)

    if len(backups) > MAX_BACKUPS:
        for old_backup in backups[MAX_BACKUPS:]:
            if old_backup.is_dir():
                shutil.rmtree(old_backup)
                print(f"Removed old backup: {old_backup.name}")


def list_backups():
    """List existing backups."""
    if not BACKUP_DIR.exists():
        print("No backups found")
        return

    backups = sorted(BACKUP_DIR.iterdir(), reverse=True)

    if not backups:
        print("No backups found")
        return

    print(f"Backups in {BACKUP_DIR}:")
    print("-" * 60)

    for backup in backups:
        if backup.is_dir():
            # Count files and total size
            files = list(backup.glob('*.db'))
            total_size = sum(f.stat().st_size for f in files)

            # Parse timestamp
            try:
                ts = datetime.strptime(backup.name, '%Y%m%d_%H%M%S')
                age = datetime.now() - ts
                age_str = f"{age.days}d ago" if age.days > 0 else "today"
            except ValueError:
                age_str = "unknown"

            print(f"  {backup.name}  ({len(files)} files, {total_size:,} bytes, {age_str})")

    print("-" * 60)
    print(f"Total: {len(backups)} backups")


def restore_backup(backup_name: str) -> bool:
    """
    Restore databases from a backup.

    Args:
        backup_name: Name of backup directory to restore from

    Returns:
        True if restore successful
    """
    backup_dir = BACKUP_DIR / backup_name

    if not backup_dir.exists():
        print(f"Backup not found: {backup_dir}")
        return False

    print(f"Restoring from: {backup_dir}")
    print("WARNING: This will overwrite current databases!")

    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted")
        return False

    print("-" * 50)

    success = True
    for name, db_path in DATABASES.items():
        backup_file = backup_dir / db_path.name

        if not backup_file.exists():
            print(f"  SKIP: {db_path.name} (not in backup)")
            continue

        try:
            # Verify backup integrity first
            conn = sqlite3.connect(backup_file)
            conn.execute("PRAGMA integrity_check")
            conn.close()

            # Copy to destination
            shutil.copy2(backup_file, db_path)
            print(f"  OK: {db_path.name}")

        except Exception as e:
            print(f"  FAIL: {db_path.name} - {e}")
            success = False

    print("-" * 50)

    if success:
        print("Restore complete")
    else:
        print("Restore completed with errors")

    return success


def main():
    parser = argparse.ArgumentParser(description='Backup trading system databases')
    parser.add_argument('--full', action='store_true', help='Full backup with verification')
    parser.add_argument('--list', action='store_true', help='List existing backups')
    parser.add_argument('--restore', type=str, help='Restore from backup directory')

    args = parser.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        sys.exit(0 if restore_backup(args.restore) else 1)
    else:
        sys.exit(0 if run_backup(full=args.full) else 1)


if __name__ == '__main__':
    main()
