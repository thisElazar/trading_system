#!/usr/bin/env python3
"""
Fix all table schemas in trades.db - add missing columns.
Run this once to migrate the database.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "db" / "trades.db"

# Expected schema for each table (column_name, column_definition)
EXPECTED_SCHEMAS = {
    'signals': [
        ('strategy_name', 'TEXT'),
        ('symbol', 'TEXT'),
        ('direction', 'TEXT'),
        ('signal_type', 'TEXT'),
        ('price', 'REAL'),
        ('stop_loss', 'REAL'),
        ('take_profit', 'REAL'),
        ('quantity', 'INTEGER'),
        ('confidence', 'REAL'),
        ('metadata', 'TEXT'),
        ('status', "TEXT DEFAULT 'pending'"),
        ('created_at', 'TEXT'),
        ('expires_at', 'TEXT'),
        ('executed_at', 'TEXT'),
    ],
    'executions': [
        ('signal_id', 'INTEGER'),
        ('order_id', 'TEXT'),
        ('symbol', 'TEXT'),
        ('direction', 'TEXT'),
        ('quantity', 'INTEGER'),
        ('fill_price', 'REAL'),
        ('commission', 'REAL DEFAULT 0'),
        ('slippage', 'REAL DEFAULT 0'),
        ('executed_at', 'TEXT'),
    ],
    'positions': [
        ('signal_id', 'INTEGER'),
        ('strategy_name', 'TEXT'),
        ('symbol', 'TEXT'),
        ('direction', 'TEXT'),
        ('quantity', 'INTEGER'),
        ('entry_price', 'REAL'),
        ('current_price', 'REAL'),
        ('stop_loss', 'REAL'),
        ('take_profit', 'REAL'),
        ('unrealized_pnl', 'REAL DEFAULT 0'),
        ('status', "TEXT DEFAULT 'open'"),
        ('opened_at', 'TEXT'),
        ('closed_at', 'TEXT'),
        ('exit_price', 'REAL'),
        ('exit_reason', 'TEXT'),
        ('realized_pnl', 'REAL'),
    ],
    'strategy_performance': [
        ('strategy_name', 'TEXT'),
        ('date', 'TEXT'),
        ('total_signals', 'INTEGER DEFAULT 0'),
        ('executed_signals', 'INTEGER DEFAULT 0'),
        ('winning_trades', 'INTEGER DEFAULT 0'),
        ('losing_trades', 'INTEGER DEFAULT 0'),
        ('total_pnl', 'REAL DEFAULT 0'),
        ('avg_slippage', 'REAL DEFAULT 0'),
        ('backtest_win_rate', 'REAL'),
        ('live_win_rate', 'REAL'),
    ],
}

INDEXES = [
    ('idx_signals_status', 'signals(status)'),
    ('idx_signals_strategy', 'signals(strategy_name)'),
    ('idx_positions_status', 'positions(status)'),
    ('idx_positions_symbol', 'positions(symbol)'),
]

def fix_schema():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for table_name, expected_cols in EXPECTED_SCHEMAS.items():
        print(f"\n{'='*50}")
        print(f"Checking table: {table_name}")
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            print(f"  Table doesn't exist, will be created on next run")
            continue
        
        # Get current columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        current_cols = {row[1] for row in cursor.fetchall()}
        print(f"  Current columns: {sorted(current_cols)}")
        
        # Find missing columns
        missing = [(name, defn) for name, defn in expected_cols if name not in current_cols]
        
        if not missing:
            print(f"  ✓ Schema up to date")
            continue
        
        print(f"  Adding {len(missing)} missing columns...")
        for col_name, col_def in missing:
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}")
                print(f"    ✓ Added {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    print(f"    - {col_name} already exists")
                else:
                    print(f"    ✗ Failed: {e}")
    
    conn.commit()
    
    # Create indexes
    print(f"\n{'='*50}")
    print("Creating indexes...")
    for idx_name, idx_def in INDEXES:
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}")
            print(f"  ✓ {idx_name}")
        except sqlite3.OperationalError as e:
            print(f"  - {idx_name}: {e}")
    
    conn.commit()
    conn.close()
    print(f"\n{'='*50}")
    print("✓ Schema migration complete!")
    return True

if __name__ == "__main__":
    fix_schema()
