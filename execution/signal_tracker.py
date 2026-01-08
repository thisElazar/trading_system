"""
Signal Database & Execution Tracker
====================================
Connects research → signals → execution → validation.

Tables:
- signals: All generated signals (pending, executed, expired)
- executions: Actual fills from broker
- positions: Current open positions
- performance: Strategy performance tracking

Flow:
1. Strategy generates Signal → saved to DB with status='pending'
2. Executor picks up pending signals → submits to broker
3. Broker confirms fill → execution recorded, signal status='executed'
4. Position tracked until exit
5. Performance compared to backtest expectations
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASES
from strategies.base import Signal as StrategySignal, SignalType

logger = logging.getLogger(__name__)


class SignalStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    EXECUTED = "executed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"


@dataclass
class StoredSignal:
    """Trading signal for database persistence."""
    id: Optional[int] = None
    strategy_name: str = ""
    symbol: str = ""
    direction: str = ""  # 'long' or 'short'
    signal_type: str = ""  # 'entry' or 'exit'
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    quantity: int = 0
    confidence: float = 0.0
    metadata: str = "{}"  # JSON string
    status: str = SignalStatus.PENDING.value
    created_at: str = ""
    expires_at: str = ""
    executed_at: Optional[str] = None
    execution_id: Optional[int] = None  # Links to executions table

    @classmethod
    def from_strategy_signal(cls, signal: 'StrategySignal', direction: str = 'long', quantity: int = 0) -> 'StoredSignal':
        """Convert a strategy Signal to StoredSignal for database storage."""
        return cls(
            strategy_name=signal.strategy,
            symbol=signal.symbol,
            direction=direction,
            signal_type='entry' if signal.signal_type.value == 'BUY' else 'exit',
            price=signal.price,
            stop_loss=signal.stop_loss or 0.0,
            take_profit=signal.target_price or 0.0,
            quantity=quantity,
            confidence=signal.strength,
            metadata=json.dumps(signal.metadata) if signal.metadata else "{}",
            created_at=signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
        )

    def to_strategy_signal(self) -> 'StrategySignal':
        """Convert StoredSignal back to strategy Signal."""
        return StrategySignal(
            timestamp=datetime.fromisoformat(self.created_at) if self.created_at else datetime.now(),
            symbol=self.symbol,
            strategy=self.strategy_name,
            signal_type=SignalType.BUY if self.signal_type == 'entry' else SignalType.CLOSE,
            strength=self.confidence,
            price=self.price,
            stop_loss=self.stop_loss if self.stop_loss > 0 else None,
            target_price=self.take_profit if self.take_profit > 0 else None,
            metadata=json.loads(self.metadata) if self.metadata else {},
        )


@dataclass
class Execution:
    """Actual execution from broker."""
    id: Optional[int] = None
    signal_id: int = 0
    order_id: str = ""
    symbol: str = ""
    direction: str = ""
    quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0  # fill_price - signal_price
    executed_at: str = ""


@dataclass
class Position:
    """Open position being tracked."""
    id: Optional[int] = None
    signal_id: int = 0
    strategy_name: str = ""
    symbol: str = ""
    direction: str = ""
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = PositionStatus.OPEN.value
    opened_at: str = ""
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    scaled_at: Optional[str] = None  # Timestamp when position was scaled (for rapid gain scaler)


class SignalDatabase:
    """
    SQLite database for signals, executions, and positions.
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DATABASES.get('trades', Path('./db/trades.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
    
    def _get_conn(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {self.db_path} - {e}")
            raise
    
    def _init_tables(self):
        """Create tables if they don't exist."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    quantity INTEGER,
                    confidence REAL,
                    metadata TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    executed_at TEXT
                )
            """)

            # Executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    fill_price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    realized_pnl REAL,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)

            # Performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    executed_signals INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_slippage REAL DEFAULT 0,
                    backtest_win_rate REAL,
                    live_win_rate REAL,
                    UNIQUE(strategy_name, date)
                )
            """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")

            conn.commit()
            conn.close()
            logger.debug(f"Signal database tables initialized: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize signal database tables: {self.db_path} - {e}")
            raise
    
    # Signal operations
    def add_signal(self, signal: StoredSignal) -> int:
        """Add a new signal. Returns signal ID."""
        if not signal.created_at:
            signal.created_at = datetime.now().isoformat()
        if not signal.expires_at:
            signal.expires_at = (datetime.now() + timedelta(hours=24)).isoformat()

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO signals (
                    strategy_name, symbol, direction, signal_type, price,
                    stop_loss, take_profit, quantity, confidence, metadata,
                    status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.strategy_name, signal.symbol, signal.direction,
                signal.signal_type, signal.price, signal.stop_loss,
                signal.take_profit, signal.quantity, signal.confidence,
                signal.metadata, signal.status, signal.created_at, signal.expires_at
            ))

            signal_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Signal added: {signal.strategy_name} {signal.direction} {signal.symbol} @ {signal.price} [id={signal_id}]")
            return signal_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add signal: {signal.strategy_name} {signal.symbol} @ {signal.price} - {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_pending_signals(self, strategy_name: str = None) -> List[StoredSignal]:
        """Get all pending signals, optionally filtered by strategy."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if strategy_name:
                cursor.execute(
                    "SELECT * FROM signals WHERE status = 'pending' AND strategy_name = ?",
                    (strategy_name,)
                )
            else:
                cursor.execute("SELECT * FROM signals WHERE status = 'pending'")

            rows = cursor.fetchall()
            return [StoredSignal(**dict(row)) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get pending signals (strategy={strategy_name}): {e}")
            return []
        finally:
            if conn:
                conn.close()

    def update_signal_status(self, signal_id: int, status: SignalStatus, executed_at: str = None):
        """Update signal status."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if executed_at:
                cursor.execute(
                    "UPDATE signals SET status = ?, executed_at = ? WHERE id = ?",
                    (status.value, executed_at, signal_id)
                )
            else:
                cursor.execute(
                    "UPDATE signals SET status = ? WHERE id = ?",
                    (status.value, signal_id)
                )

            conn.commit()
            logger.debug(f"Signal {signal_id} status updated to {status.value}")
        except sqlite3.Error as e:
            logger.error(f"Failed to update signal status: signal_id={signal_id}, status={status.value} - {e}")
            raise
        finally:
            if conn:
                conn.close()

    def expire_old_signals(self) -> int:
        """Mark expired signals."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            cursor.execute("""
                UPDATE signals
                SET status = 'expired'
                WHERE status = 'pending' AND expires_at < ?
            """, (now,))

            expired = cursor.rowcount
            conn.commit()

            if expired > 0:
                logger.info(f"Expired {expired} old signals")

            return expired
        except sqlite3.Error as e:
            logger.error(f"Failed to expire old signals: {e}")
            return 0
        finally:
            if conn:
                conn.close()

    def cleanup_orphaned_signals(self, max_age_hours: int = 24) -> Dict[str, int]:
        """
        Clean up orphaned signals on startup to prevent duplicate trades.

        This should be called during system startup/recovery to handle:
        - Pending signals from crashed sessions that might re-execute
        - Submitted signals that never got confirmation
        - Old signals beyond the max age window

        Args:
            max_age_hours: Signals older than this are marked expired

        Returns:
            Dict with counts: {'expired': N, 'cancelled': M, 'orphaned_positions': P}
        """
        conn = None
        results = {'expired': 0, 'cancelled': 0, 'orphaned_positions': 0}

        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            now = datetime.now()
            cutoff = (now - timedelta(hours=max_age_hours)).isoformat()

            # 1. Expire all pending signals older than max_age_hours
            cursor.execute("""
                UPDATE signals
                SET status = 'expired'
                WHERE status = 'pending' AND created_at < ?
            """, (cutoff,))
            results['expired'] = cursor.rowcount

            # 2. Cancel submitted signals older than max_age (broker likely rejected)
            cursor.execute("""
                UPDATE signals
                SET status = 'cancelled'
                WHERE status = 'submitted' AND created_at < ?
            """, (cutoff,))
            results['cancelled'] = cursor.rowcount

            # 3. Find orphaned positions (open in DB but signal is expired/cancelled)
            cursor.execute("""
                SELECT p.id, p.symbol, p.strategy_name
                FROM positions p
                LEFT JOIN signals s ON p.signal_id = s.id
                WHERE p.status = 'open'
                  AND (s.status IN ('expired', 'cancelled') OR s.id IS NULL)
            """)
            orphaned = cursor.fetchall()
            results['orphaned_positions'] = len(orphaned)

            if orphaned:
                for pos_id, symbol, strategy in orphaned:
                    logger.warning(f"ORPHAN DETECTED: Position {symbol} ({strategy}) "
                                  f"has expired/cancelled/missing signal - needs broker reconciliation")

            conn.commit()

            if any(results.values()):
                logger.info(f"Orphan cleanup: {results['expired']} signals expired, "
                           f"{results['cancelled']} cancelled, {results['orphaned_positions']} orphaned positions")

            return results

        except sqlite3.Error as e:
            logger.error(f"Failed to cleanup orphaned signals: {e}")
            return results
        finally:
            if conn:
                conn.close()

    def get_stale_pending_signals(self, max_age_hours: int = 24) -> List[StoredSignal]:
        """
        Get pending signals that are older than max_age_hours.
        Used for recovery review before auto-expiring.
        """
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

            cursor.execute("""
                SELECT * FROM signals
                WHERE status = 'pending' AND created_at < ?
                ORDER BY created_at
            """, (cutoff,))

            rows = cursor.fetchall()
            return [StoredSignal(**dict(row)) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to get stale pending signals: {e}")
            return []
        finally:
            if conn:
                conn.close()

    # Execution operations
    def record_execution(self, execution: Execution) -> int:
        """Record an execution."""
        if not execution.executed_at:
            execution.executed_at = datetime.now().isoformat()

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO executions (
                    signal_id, order_id, symbol, direction, quantity,
                    fill_price, commission, slippage, executed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.signal_id, execution.order_id, execution.symbol,
                execution.direction, execution.quantity, execution.fill_price,
                execution.commission, execution.slippage, execution.executed_at
            ))

            exec_id = cursor.lastrowid
            conn.commit()
            conn.close()
            conn = None  # Prevent double close in finally

            # Update signal status
            self.update_signal_status(
                execution.signal_id,
                SignalStatus.EXECUTED,
                execution.executed_at
            )

            logger.info(f"Execution recorded: {execution.symbol} @ {execution.fill_price} "
                       f"(slippage: {execution.slippage:.2f}, order_id={execution.order_id}) [exec_id={exec_id}]")
            return exec_id
        except sqlite3.Error as e:
            logger.error(f"Failed to record execution: {execution.symbol} @ {execution.fill_price} "
                        f"order_id={execution.order_id} - {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # Position operations
    def open_position(self, position: Position) -> int:
        """Open a new position. Prevents duplicates by checking existing open positions."""
        if not position.opened_at:
            position.opened_at = datetime.now().isoformat()

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Check for existing open position with same symbol
            cursor.execute("""
                SELECT id, quantity, entry_price FROM positions
                WHERE symbol = ? AND status = 'open'
            """, (position.symbol,))
            existing = cursor.fetchone()

            if existing:
                existing_id, existing_qty, existing_price = existing
                logger.warning(f"Position already exists for {position.symbol} (id={existing_id}, "
                             f"qty={existing_qty}). Skipping duplicate open.")
                return existing_id

            cursor.execute("""
                INSERT INTO positions (
                    signal_id, strategy_name, symbol, direction, quantity,
                    entry_price, current_price, stop_loss, take_profit,
                    unrealized_pnl, status, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.signal_id, position.strategy_name, position.symbol,
                position.direction, position.quantity, position.entry_price,
                position.current_price or position.entry_price, position.stop_loss,
                position.take_profit, position.unrealized_pnl, position.status,
                position.opened_at
            ))

            pos_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Position opened: {position.direction} {position.quantity} {position.symbol} "
                       f"@ {position.entry_price} (stop={position.stop_loss}, target={position.take_profit}) [pos_id={pos_id}]")
            return pos_id
        except sqlite3.Error as e:
            logger.error(f"Failed to open position: {position.direction} {position.symbol} @ {position.entry_price} - {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_open_positions(self, strategy_name: str = None) -> List[Position]:
        """Get all open positions."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            if strategy_name:
                cursor.execute(
                    "SELECT * FROM positions WHERE status = 'open' AND strategy_name = ?",
                    (strategy_name,)
                )
            else:
                cursor.execute("SELECT * FROM positions WHERE status = 'open'")

            rows = cursor.fetchall()
            return [Position(**dict(row)) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get open positions (strategy={strategy_name}): {e}")
            return []
        finally:
            if conn:
                conn.close()

    def update_position_price(self, position_id: int, current_price: float):
        """Update position's current price and unrealized P&L."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Get position
            cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
            row = cursor.fetchone()

            if row:
                pos = Position(**dict(row))
                if pos.direction == 'long':
                    unrealized = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    unrealized = (pos.entry_price - current_price) / pos.entry_price * 100

                cursor.execute("""
                    UPDATE positions
                    SET current_price = ?, unrealized_pnl = ?
                    WHERE id = ?
                """, (current_price, unrealized, position_id))
                conn.commit()
            else:
                logger.warning(f"Position not found for price update: position_id={position_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to update position price: position_id={position_id}, price={current_price} - {e}")
        finally:
            if conn:
                conn.close()

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str
    ):
        """Close a position."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Get position
            cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
            row = cursor.fetchone()

            if row:
                pos = Position(**dict(row))

                if pos.direction == 'long':
                    realized_pnl = (exit_price - pos.entry_price) / pos.entry_price * 100
                else:
                    realized_pnl = (pos.entry_price - exit_price) / pos.entry_price * 100

                cursor.execute("""
                    UPDATE positions
                    SET status = 'closed', closed_at = ?, exit_price = ?,
                        exit_reason = ?, realized_pnl = ?
                    WHERE id = ?
                """, (
                    datetime.now().isoformat(), exit_price,
                    exit_reason, realized_pnl, position_id
                ))
                conn.commit()

                logger.info(f"Position closed: {pos.symbol} @ {exit_price} ({exit_reason}) "
                           f"P&L: {realized_pnl:+.2f}% [pos_id={position_id}]")
            else:
                logger.warning(f"Position not found for close: position_id={position_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to close position: position_id={position_id}, exit_price={exit_price}, "
                        f"reason={exit_reason} - {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # Performance tracking
    def update_strategy_performance(self, strategy_name: str):
        """Update daily performance stats for a strategy."""
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            today = datetime.now().strftime("%Y-%m-%d")

            # Get today's stats
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN status = 'executed' THEN 1 ELSE 0 END) as executed
                FROM signals
                WHERE strategy_name = ? AND DATE(created_at) = ?
            """, (strategy_name, today))
            signals = cursor.fetchone()

            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                       SUM(realized_pnl) as pnl
                FROM positions
                WHERE strategy_name = ? AND DATE(closed_at) = ?
            """, (strategy_name, today))
            trades = cursor.fetchone()

            cursor.execute("""
                SELECT AVG(slippage) as avg_slip
                FROM executions e
                JOIN signals s ON e.signal_id = s.id
                WHERE s.strategy_name = ? AND DATE(e.executed_at) = ?
            """, (strategy_name, today))
            slippage = cursor.fetchone()

            # Upsert performance
            cursor.execute("""
                INSERT INTO strategy_performance (
                    strategy_name, date, total_signals, executed_signals,
                    winning_trades, losing_trades, total_pnl, avg_slippage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_name, date) DO UPDATE SET
                    total_signals = excluded.total_signals,
                    executed_signals = excluded.executed_signals,
                    winning_trades = excluded.winning_trades,
                    losing_trades = excluded.losing_trades,
                    total_pnl = excluded.total_pnl,
                    avg_slippage = excluded.avg_slippage
            """, (
                strategy_name, today,
                signals['total'] or 0, signals['executed'] or 0,
                trades['wins'] or 0, trades['losses'] or 0,
                trades['pnl'] or 0, slippage['avg_slip'] or 0
            ))

            conn.commit()
            logger.debug(f"Strategy performance updated: {strategy_name} for {today}")
        except sqlite3.Error as e:
            logger.error(f"Failed to update strategy performance: {strategy_name} - {e}")
        finally:
            if conn:
                conn.close()

    def get_performance_summary(
        self,
        strategy_name: str = None,
        days: int = 30
    ) -> pd.DataFrame:
        """Get performance summary."""
        conn = None
        try:
            conn = self._get_conn()

            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            if strategy_name:
                query = """
                    SELECT * FROM strategy_performance
                    WHERE strategy_name = ? AND date >= ?
                    ORDER BY date DESC
                """
                df = pd.read_sql_query(query, conn, params=(strategy_name, cutoff))
            else:
                query = """
                    SELECT * FROM strategy_performance
                    WHERE date >= ?
                    ORDER BY strategy_name, date DESC
                """
                df = pd.read_sql_query(query, conn, params=(cutoff,))

            return df
        except sqlite3.Error as e:
            logger.error(f"Failed to get performance summary (strategy={strategy_name}, days={days}): {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def get_all_closed_positions(
        self,
        strategy_name: str = None,
        days: int = 30
    ) -> pd.DataFrame:
        """Get all closed positions."""
        conn = None
        try:
            conn = self._get_conn()

            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            if strategy_name:
                query = """
                    SELECT * FROM positions
                    WHERE strategy_name = ? AND status = 'closed' AND closed_at >= ?
                    ORDER BY closed_at DESC
                """
                df = pd.read_sql_query(query, conn, params=(strategy_name, cutoff))
            else:
                query = """
                    SELECT * FROM positions
                    WHERE status = 'closed' AND closed_at >= ?
                    ORDER BY closed_at DESC
                """
                df = pd.read_sql_query(query, conn, params=(cutoff,))

            return df
        except sqlite3.Error as e:
            logger.error(f"Failed to get closed positions (strategy={strategy_name}, days={days}): {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()


class ExecutionTracker:
    """
    Tracks execution quality and compares live vs backtest performance.
    """
    
    def __init__(self, db: SignalDatabase = None):
        self.db = db or SignalDatabase()
        
        # Backtest baselines (from research)
        self.backtest_baselines = {
            'gap_fill': {'win_rate': 0.55, 'avg_pnl': -0.04},
            'pairs_trading': {'win_rate': 0.67, 'avg_pnl': 1.97},
            'relative_volume_breakout': {'win_rate': 0.48, 'avg_pnl': -0.09},
        }
    
    def record_signal_and_execute(
        self,
        strategy_name: str = None,
        symbol: str = None,
        direction: str = None,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        quantity: int = None,
        confidence: float = 0.5,
        metadata: dict = None,
        strategy_signal: StrategySignal = None
    ) -> Tuple[int, int]:
        """
        Record a signal and simulate immediate execution.
        Returns (signal_id, position_id).

        Can be called either with individual parameters or with a StrategySignal:
        - Individual parameters: strategy_name, symbol, direction, entry_price, etc.
        - StrategySignal: Pass a StrategySignal object via strategy_signal parameter,
          along with direction and quantity which are not part of StrategySignal.
        """
        # Create StoredSignal from StrategySignal if provided
        if strategy_signal is not None:
            stored_signal = StoredSignal.from_strategy_signal(
                strategy_signal,
                direction=direction or 'long',
                quantity=quantity or 0
            )
            stored_signal.status = SignalStatus.PENDING.value
            # Override entry_price if provided separately
            if entry_price is not None:
                stored_signal.price = entry_price
            # Override stop_loss and take_profit if provided separately
            if stop_loss is not None:
                stored_signal.stop_loss = stop_loss
            if take_profit is not None:
                stored_signal.take_profit = take_profit
        else:
            # Create StoredSignal from individual parameters
            stored_signal = StoredSignal(
                strategy_name=strategy_name,
                symbol=symbol,
                direction=direction,
                signal_type='entry',
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                confidence=confidence,
                metadata=json.dumps(metadata or {}),
                status=SignalStatus.PENDING.value
            )

        # Use stored_signal for the rest of the method
        signal = stored_signal

        signal_id = self.db.add_signal(signal)

        # Simulate execution (paper trading assumes immediate fill)
        # In live trading, this would come from broker callback
        slippage = 0.0  # Paper trading = no slippage
        fill_price = signal.price

        execution = Execution(
            signal_id=signal_id,
            order_id=f"PAPER_{signal_id}",
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=signal.quantity,
            fill_price=fill_price,
            slippage=slippage
        )

        self.db.record_execution(execution)

        # Open position
        position = Position(
            signal_id=signal_id,
            strategy_name=signal.strategy_name,
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=signal.quantity,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )

        position_id = self.db.open_position(position)

        return signal_id, position_id
    
    def check_stops_and_targets(self, current_prices: Dict[str, float]):
        """
        Check open positions against current prices for stops/targets.
        
        Args:
            current_prices: Dict of {symbol: current_price}
        """
        positions = self.db.get_open_positions()
        
        for pos in positions:
            if pos.symbol not in current_prices:
                continue
            
            current = current_prices[pos.symbol]
            self.db.update_position_price(pos.id, current)
            
            # Check stop loss
            if pos.direction == 'long':
                if current <= pos.stop_loss:
                    self.db.close_position(pos.id, current, 'stop_loss')
                elif pos.take_profit and current >= pos.take_profit:
                    self.db.close_position(pos.id, current, 'target')
            else:  # short
                if current >= pos.stop_loss:
                    self.db.close_position(pos.id, current, 'stop_loss')
                elif pos.take_profit and current <= pos.take_profit:
                    self.db.close_position(pos.id, current, 'target')
    
    def compare_to_backtest(self, strategy_name: str, days: int = 30) -> dict:
        """
        Compare live performance to backtest baseline.
        """
        closed = self.db.get_all_closed_positions(strategy_name, days)
        
        if len(closed) == 0:
            return {'status': 'no_data', 'trades': 0}
        
        live_win_rate = (closed['realized_pnl'] > 0).mean()
        live_avg_pnl = closed['realized_pnl'].mean()
        
        baseline = self.backtest_baselines.get(strategy_name, {})
        bt_win_rate = baseline.get('win_rate', 0)
        bt_avg_pnl = baseline.get('avg_pnl', 0)
        
        # Calculate degradation
        win_rate_diff = live_win_rate - bt_win_rate
        pnl_diff = live_avg_pnl - bt_avg_pnl
        
        status = 'ok'
        if win_rate_diff < -0.15:  # 15% worse
            status = 'degraded'
        if win_rate_diff < -0.25:  # 25% worse
            status = 'critical'
        
        return {
            'status': status,
            'trades': len(closed),
            'live_win_rate': live_win_rate,
            'backtest_win_rate': bt_win_rate,
            'win_rate_diff': win_rate_diff,
            'live_avg_pnl': live_avg_pnl,
            'backtest_avg_pnl': bt_avg_pnl,
            'pnl_diff': pnl_diff
        }
    
    def get_pending_signals(self, strategy_name: str = None) -> List[StoredSignal]:
        """Get pending signals, optionally filtered by strategy."""
        return self.db.get_pending_signals(strategy_name)

    def get_active_strategies(self) -> List[str]:
        """Get list of strategies with open positions or recent activity."""
        try:
            # Get strategies from open positions
            open_positions = self.db.get_open_positions()
            strategies = set(p.strategy_name for p in open_positions if p.strategy_name)

            # Also include strategies from backtest baselines
            strategies.update(self.backtest_baselines.keys())

            return list(strategies)
        except Exception as e:
            logger.warning(f"Error getting active strategies: {e}")
            return list(self.backtest_baselines.keys())

    def get_strategy_pnl(self, strategy_name: str, date: str = None) -> Optional[float]:
        """Get P&L for a strategy on a given date."""
        try:
            conn = self.db._get_conn()
            cursor = conn.cursor()

            if date:
                cursor.execute("""
                    SELECT SUM(pnl) as total_pnl
                    FROM positions
                    WHERE strategy_name = ? AND DATE(closed_at) = DATE(?)
                """, (strategy_name, date))
            else:
                cursor.execute("""
                    SELECT SUM(pnl) as total_pnl
                    FROM positions
                    WHERE strategy_name = ?
                """, (strategy_name,))

            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else 0.0
        except Exception as e:
            logger.warning(f"Error getting strategy PnL: {e}")
            return None

    def generate_report(self) -> str:
        """Generate execution tracking report."""
        lines = []
        lines.append("=" * 60)
        lines.append("EXECUTION TRACKER REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Open positions
        open_pos = self.db.get_open_positions()
        lines.append(f"## Open Positions: {len(open_pos)}")
        for p in open_pos:
            lines.append(f"  {p.symbol} {p.direction} @ {p.entry_price:.2f} | "
                        f"P&L: {p.unrealized_pnl:.2f}% | Stop: {p.stop_loss:.2f}")
        lines.append("")
        
        # Strategy comparison
        lines.append("## Strategy Performance vs Backtest")
        for strategy in self.backtest_baselines.keys():
            comp = self.compare_to_backtest(strategy)
            if comp['status'] == 'no_data':
                lines.append(f"  {strategy}: No live trades yet")
            else:
                status_icon = {'ok': '✓', 'degraded': '⚠', 'critical': '✗'}[comp['status']]
                lines.append(f"  {status_icon} {strategy}:")
                lines.append(f"      Live: {comp['live_win_rate']:.1%} win, {comp['live_avg_pnl']:.2f}% avg")
                lines.append(f"      Backtest: {comp['backtest_win_rate']:.1%} win, {comp['backtest_avg_pnl']:.2f}% avg")
                lines.append(f"      Diff: {comp['win_rate_diff']:+.1%} win rate, {comp['pnl_diff']:+.2f}% P&L")
        
        return "\n".join(lines)


def test_signal_database():
    """Test the signal database."""
    print("Testing Signal Database...")
    
    db = SignalDatabase()
    tracker = ExecutionTracker(db)
    
    # Simulate a trade
    signal_id, pos_id = tracker.record_signal_and_execute(
        strategy_name='pairs_trading',
        symbol='AAPL',
        direction='long',
        entry_price=250.0,
        stop_loss=245.0,
        take_profit=260.0,
        quantity=10,
        confidence=0.75,
        metadata={'pair': 'AAPL/MSFT', 'zscore': -2.1}
    )
    
    print(f"Created signal {signal_id}, position {pos_id}")
    
    # Check positions
    positions = db.get_open_positions()
    print(f"Open positions: {len(positions)}")
    
    # Simulate price update
    tracker.check_stops_and_targets({'AAPL': 252.0})
    
    # Generate report
    print("\n" + tracker.generate_report())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_signal_database()
