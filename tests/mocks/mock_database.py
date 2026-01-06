"""
Mock database implementations for testing.

Provides in-memory SQLite databases and mock database managers
for testing without affecting production data.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MockSignal:
    """Mock signal record."""
    id: int
    symbol: str
    strategy: str
    signal_type: str
    strength: float
    price: float
    created_at: datetime


@dataclass
class MockPositionRecord:
    """Mock position record."""
    id: int
    symbol: str
    strategy: str
    side: str
    qty: int
    entry_price: float
    current_price: Optional[float]
    stop_loss: Optional[float]
    target: Optional[float]
    unrealized_pnl: float
    status: str
    opened_at: datetime
    closed_at: Optional[datetime]


def create_test_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Create a test database with trading system schema.

    Args:
        path: Optional path for the database. If None, uses in-memory DB.

    Returns:
        SQLite connection
    """
    db_path = str(path) if path else ':memory:'
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.executescript("""
        -- Signals table
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            strength REAL DEFAULT 0.5,
            price REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );

        -- Positions table
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('long', 'short')),
            qty INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            stop_loss REAL,
            target REAL,
            unrealized_pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'pending')),
            broker_order_id TEXT,
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP,
            close_reason TEXT
        );

        -- Trades table (completed trades)
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            pnl REAL,
            pnl_pct REAL,
            fees REAL DEFAULT 0,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            hold_days INTEGER,
            notes TEXT
        );

        -- Strategy performance table
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            date DATE NOT NULL,
            pnl REAL DEFAULT 0,
            trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            UNIQUE(strategy, date)
        );

        -- GA optimization results
        CREATE TABLE IF NOT EXISTS ga_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            generation INTEGER NOT NULL,
            params TEXT NOT NULL,
            fitness REAL NOT NULL,
            sharpe REAL,
            returns REAL,
            max_drawdown REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indices for performance
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
        CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
        CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
    """)

    conn.commit()
    return conn


class MockDatabaseManager:
    """
    Mock database manager for testing.

    Provides a simplified interface matching the real database manager
    but uses an in-memory SQLite database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.conn = create_test_db(db_path)
        self._transaction_depth = 0

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        self._transaction_depth += 1
        try:
            yield
            if self._transaction_depth == 1:
                self.conn.commit()
        except Exception:
            if self._transaction_depth == 1:
                self.conn.rollback()
            raise
        finally:
            self._transaction_depth -= 1

    # -------------------------------------------------------------------------
    # Signal Operations
    # -------------------------------------------------------------------------

    def record_signal(
        self,
        symbol: str,
        strategy: str,
        signal_type: str,
        strength: float = 0.5,
        price: Optional[float] = None,
    ) -> int:
        """Record a new trading signal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO signals (symbol, strategy, signal_type, strength, price)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, strategy, signal_type, strength, price))
        self.conn.commit()
        return cursor.lastrowid

    def get_signals(
        self,
        strategy: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get signals, optionally filtered."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM signals WHERE 1=1"
        params = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if since:
            query += " AND created_at >= ?"
            params.append(since)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Position Operations
    # -------------------------------------------------------------------------

    def create_position(
        self,
        symbol: str,
        strategy: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
    ) -> int:
        """Create a new position."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO positions
            (symbol, strategy, side, qty, entry_price, current_price, stop_loss, target)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, strategy, side, qty, entry_price, entry_price, stop_loss, target))
        self.conn.commit()
        return cursor.lastrowid

    def get_open_positions(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open positions."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM positions WHERE status = 'open'"
        params = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def update_position_price(
        self,
        position_id: int,
        current_price: float,
        unrealized_pnl: float,
    ) -> None:
        """Update position with current price."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE positions
            SET current_price = ?, unrealized_pnl = ?
            WHERE id = ?
        """, (current_price, unrealized_pnl, position_id))
        self.conn.commit()

    def close_position(
        self,
        position_id: int,
        reason: str = 'manual',
        exit_price: Optional[float] = None,
    ) -> None:
        """Close a position."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE positions
            SET status = 'closed', closed_at = CURRENT_TIMESTAMP, close_reason = ?
            WHERE id = ?
        """, (reason, position_id))
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Trade Operations
    # -------------------------------------------------------------------------

    def record_trade(
        self,
        symbol: str,
        strategy: str,
        side: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
    ) -> int:
        """Record a completed trade."""
        pnl = (exit_price - entry_price) * qty
        if side == 'short':
            pnl = -pnl
        pnl_pct = pnl / (entry_price * qty) * 100

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades
            (symbol, strategy, side, qty, entry_price, exit_price, pnl, pnl_pct, entry_time, exit_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, strategy, side, qty, entry_price, exit_price, pnl, pnl_pct, entry_time, exit_time))
        self.conn.commit()
        return cursor.lastrowid

    def get_trades(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get trades, optionally filtered."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Performance Operations
    # -------------------------------------------------------------------------

    def get_strategy_stats(self, strategy: str) -> Dict[str, Any]:
        """Get performance statistics for a strategy."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(pnl_pct) as avg_pnl_pct
            FROM trades
            WHERE strategy = ?
        """, (strategy,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_pnl_pct': 0,
        }

    # -------------------------------------------------------------------------
    # Test Utilities
    # -------------------------------------------------------------------------

    def clear_all(self):
        """Clear all data from tables (for test cleanup)."""
        cursor = self.conn.cursor()
        cursor.executescript("""
            DELETE FROM signals;
            DELETE FROM positions;
            DELETE FROM trades;
            DELETE FROM strategy_performance;
            DELETE FROM ga_results;
        """)
        self.conn.commit()

    def seed_test_data(self):
        """Seed database with sample test data."""
        # Sample signals
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            self.record_signal(symbol, 'momentum', 'buy', 0.8, 150.0)

        # Sample positions
        self.create_position('AAPL', 'momentum', 'long', 100, 150.0, 142.5, 165.0)
        self.create_position('GOOGL', 'mean_reversion', 'long', 50, 2800.0, 2660.0, 3080.0)

        # Sample trades
        self.record_trade('MSFT', 'momentum', 'long', 100, 300.0, 315.0)
        self.record_trade('AMZN', 'mean_reversion', 'long', 20, 3200.0, 3100.0)
