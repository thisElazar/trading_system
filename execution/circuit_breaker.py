"""
Circuit Breaker & Kill Switch System
=====================================
Comprehensive protection layer for the trading system.

Circuit Breakers (Automatic):
- Daily Loss: Halt trading when daily loss exceeds threshold
- Drawdown: Reduce position sizes when drawdown from peak exceeds threshold
- Rapid Loss: Temporary halt on sudden losses (flash crash protection)
- Consecutive Loss: Pause strategy after N consecutive losing trades
- Strategy Performance: Disable underperforming strategies

Kill Switches (Manual):
- HALT: Stop all new orders, keep existing positions
- CLOSE_ALL: Emergency close all positions (requires confirmation)
- GRACEFUL: Stop new signals, let pending orders complete
- STRATEGY_X: Disable specific strategy

Usage:
    manager = CircuitBreakerManager()
    manager.inject_dependencies(broker, alert_manager)

    # Check before trading
    if manager.can_trade():
        execute_order(...)

    # Get position size multiplier (for drawdown reduction)
    multiplier = manager.get_position_multiplier()

    # CLI control
    python scripts/circuit_breaker_cli.py status
    python scripts/circuit_breaker_cli.py halt
"""

import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASES, DIRS, MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class BreakerAction(Enum):
    """Actions a circuit breaker can take."""
    NONE = "none"
    HALT = "halt"                    # Stop all new orders
    REDUCE = "reduce"                # Reduce position sizes
    PAUSE_STRATEGY = "pause_strategy"  # Pause specific strategy
    CLOSE_ALL = "close_all"          # Emergency close all


class KillSwitchType(Enum):
    """Types of manual kill switches."""
    HALT = "HALT"
    CLOSE_ALL = "CLOSE_ALL"
    GRACEFUL = "GRACEFUL"
    STRATEGY = "STRATEGY"


@dataclass
class CircuitBreakerConfig:
    """Configuration for all circuit breakers."""
    # Daily Loss
    daily_loss_pct: float = 0.02

    # Drawdown
    drawdown_pct: float = 0.15

    # Rapid Loss (flash crash protection)
    rapid_loss_pct: float = 0.01
    rapid_loss_window_min: int = 15
    rapid_loss_pause_min: int = 30

    # Consecutive Losses
    max_consecutive_losses: int = 5
    consecutive_loss_pause_hrs: int = 4

    # Strategy Performance
    strategy_loss_pct: float = 0.05
    strategy_pause_hrs: int = 24

    @classmethod
    def from_dict(cls, d: Dict) -> 'CircuitBreakerConfig':
        """Create from dictionary (for config.py integration)."""
        return cls(
            daily_loss_pct=d.get('daily_loss_pct', 0.02),
            drawdown_pct=d.get('drawdown_pct', 0.15),
            rapid_loss_pct=d.get('rapid_loss_pct', 0.01),
            rapid_loss_window_min=d.get('rapid_loss_window_min', 15),
            rapid_loss_pause_min=d.get('rapid_loss_pause_min', 30),
            max_consecutive_losses=d.get('max_consecutive_losses', 5),
            consecutive_loss_pause_hrs=d.get('consecutive_loss_pause_hrs', 4),
            strategy_loss_pct=d.get('strategy_loss_pct', 0.05),
            strategy_pause_hrs=d.get('strategy_pause_hrs', 24),
        )


@dataclass
class BreakerState:
    """State of a circuit breaker."""
    breaker_type: str
    is_triggered: bool
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    action: BreakerAction = BreakerAction.NONE
    reason: str = ""
    target: str = "all"  # "all" or specific strategy name


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class CircuitBreakerDB:
    """Database operations for circuit breaker state persistence."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASES.get("performance")
        self._ensure_tables()

    def _ensure_tables(self):
        """Create circuit breaker tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Active circuit breaker states
                CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    breaker_type TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    expires_at TEXT,
                    reason TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT DEFAULT 'all',
                    is_active INTEGER DEFAULT 1,
                    cleared_at TEXT,
                    cleared_by TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_cb_active
                ON circuit_breaker_state(is_active, breaker_type, target);

                -- Trade streak tracking for consecutive loss detection
                CREATE TABLE IF NOT EXISTS trade_streak (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL UNIQUE,
                    streak_type TEXT NOT NULL,
                    streak_count INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL
                );

                -- Equity snapshots for rapid loss detection
                CREATE TABLE IF NOT EXISTS rapid_loss_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rapid_loss_time
                ON rapid_loss_log(timestamp);

                -- Kill switch audit log
                CREATE TABLE IF NOT EXISTS kill_switch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    switch_type TEXT NOT NULL,
                    triggered_by TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    positions_affected INTEGER DEFAULT 0,
                    orders_cancelled INTEGER DEFAULT 0
                );
            """)

    def save_breaker_state(self, state: BreakerState):
        """Save a triggered circuit breaker state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO circuit_breaker_state
                (breaker_type, triggered_at, expires_at, reason, action, target, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (
                state.breaker_type,
                state.triggered_at.isoformat() if state.triggered_at else None,
                state.expires_at.isoformat() if state.expires_at else None,
                state.reason,
                state.action.value,
                state.target
            ))

    def get_active_breakers(
        self,
        action: Optional[str] = None,
        target: Optional[str] = None
    ) -> List[BreakerState]:
        """Get all active circuit breaker states."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM circuit_breaker_state WHERE is_active = 1"
            params = []

            if action:
                query += " AND action = ?"
                params.append(action)
            if target:
                query += " AND target = ?"
                params.append(target)

            rows = conn.execute(query, params).fetchall()

            states = []
            for row in rows:
                states.append(BreakerState(
                    breaker_type=row['breaker_type'],
                    is_triggered=True,
                    triggered_at=datetime.fromisoformat(row['triggered_at']) if row['triggered_at'] else None,
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                    action=BreakerAction(row['action']),
                    reason=row['reason'],
                    target=row['target']
                ))

            return states

    def clear_breaker(
        self,
        breaker_type: str,
        target: str = 'all',
        cleared_by: str = 'manual'
    ) -> bool:
        """Clear an active circuit breaker."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                UPDATE circuit_breaker_state
                SET is_active = 0, cleared_at = ?, cleared_by = ?
                WHERE breaker_type = ? AND target = ? AND is_active = 1
            """, (datetime.now().isoformat(), cleared_by, breaker_type, target))
            return result.rowcount > 0

    def clear_expired_breakers(self) -> int:
        """Clear all expired circuit breakers."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                UPDATE circuit_breaker_state
                SET is_active = 0, cleared_at = ?, cleared_by = 'timer'
                WHERE is_active = 1 AND expires_at IS NOT NULL AND expires_at < ?
            """, (now, now))
            return result.rowcount

    # Trade streak methods
    def get_streak(self, strategy: str) -> Dict:
        """Get current trade streak for a strategy."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_streak WHERE strategy_name = ?",
                (strategy,)
            ).fetchone()

            if row:
                return {
                    'strategy': row['strategy_name'],
                    'streak_type': row['streak_type'],
                    'streak_count': row['streak_count']
                }
            return {'strategy': strategy, 'streak_type': 'none', 'streak_count': 0}

    def update_streak(self, strategy: str, streak_type: str, count: int):
        """Update trade streak for a strategy."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trade_streak (strategy_name, streak_type, streak_count, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(strategy_name) DO UPDATE SET
                    streak_type = excluded.streak_type,
                    streak_count = excluded.streak_count,
                    last_updated = excluded.last_updated
            """, (strategy, streak_type, count, datetime.now().isoformat()))

    def reset_all_streaks(self):
        """Reset all trade streaks (called at daily reset)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE trade_streak SET streak_count = 0, streak_type = 'none'")

    # Rapid loss tracking
    def record_equity_snapshot(self, equity: float):
        """Record equity snapshot for rapid loss detection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO rapid_loss_log (timestamp, equity) VALUES (?, ?)",
                (datetime.now().isoformat(), equity)
            )
            # Clean up old snapshots (keep last hour)
            cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
            conn.execute("DELETE FROM rapid_loss_log WHERE timestamp < ?", (cutoff,))

    def get_equity_snapshots(self, since: datetime) -> List[Dict]:
        """Get equity snapshots since a given time."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM rapid_loss_log WHERE timestamp >= ? ORDER BY timestamp",
                (since.isoformat(),)
            ).fetchall()
            return [{'timestamp': row['timestamp'], 'equity': row['equity']} for row in rows]

    # Kill switch logging
    def log_kill_switch(self, switch_type: str, triggered_by: str, result: Dict):
        """Log a kill switch activation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO kill_switch_log
                (timestamp, switch_type, triggered_by, action_taken, positions_affected, orders_cancelled)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                switch_type,
                triggered_by,
                json.dumps(result),
                result.get('positions_affected', result.get('positions_closed', 0)),
                result.get('orders_cancelled', 0)
            ))

    def get_kill_switch_log(self, limit: int = 20) -> List[Dict]:
        """Get recent kill switch activations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM kill_switch_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]


# =============================================================================
# CIRCUIT BREAKER BASE CLASS
# =============================================================================

class CircuitBreaker(ABC):
    """Base class for all circuit breakers."""

    def __init__(self, db: Optional[CircuitBreakerDB] = None):
        self.db = db or CircuitBreakerDB()

    @abstractmethod
    def check(self, context: Dict) -> BreakerState:
        """Check if the breaker should trigger. Returns state."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Reset the breaker. Returns success."""
        pass

    def _not_triggered(self, breaker_type: str) -> BreakerState:
        """Return a non-triggered state."""
        return BreakerState(
            breaker_type=breaker_type,
            is_triggered=False,
            action=BreakerAction.NONE,
            reason=""
        )


# =============================================================================
# SPECIFIC CIRCUIT BREAKERS
# =============================================================================

class DailyLossBreaker(CircuitBreaker):
    """Halts all trading when daily loss exceeds threshold."""

    def __init__(self, threshold_pct: float = 0.02, db: Optional[CircuitBreakerDB] = None):
        super().__init__(db)
        self.threshold = threshold_pct

    def check(self, context: Dict) -> BreakerState:
        start_equity = context.get('start_of_day_equity', 0)
        current_equity = context.get('current_equity', 0)

        if start_equity <= 0:
            return self._not_triggered('daily_loss')

        daily_loss_pct = (start_equity - current_equity) / start_equity

        if daily_loss_pct >= self.threshold:
            state = BreakerState(
                breaker_type='daily_loss',
                is_triggered=True,
                triggered_at=datetime.now(),
                expires_at=self._next_market_open(),
                action=BreakerAction.HALT,
                reason=f"Daily loss {daily_loss_pct:.2%} exceeds {self.threshold:.2%} limit"
            )
            self.db.save_breaker_state(state)
            logger.warning(f"CIRCUIT BREAKER: {state.reason}")
            return state

        return self._not_triggered('daily_loss')

    def _next_market_open(self) -> datetime:
        """Calculate next market open (simplified: next day 9:30 AM ET)."""
        now = datetime.now()
        next_day = now + timedelta(days=1)
        return next_day.replace(hour=9, minute=30, second=0, microsecond=0)

    def reset(self) -> bool:
        return self.db.clear_breaker('daily_loss')


class DrawdownBreaker(CircuitBreaker):
    """Reduces position sizes when drawdown from peak exceeds threshold."""

    def __init__(self, threshold_pct: float = 0.15, db: Optional[CircuitBreakerDB] = None):
        super().__init__(db)
        self.threshold = threshold_pct

    def check(self, context: Dict) -> BreakerState:
        peak_equity = context.get('peak_equity', 0)
        current_equity = context.get('current_equity', 0)

        if peak_equity <= 0:
            return self._not_triggered('drawdown')

        drawdown_pct = (peak_equity - current_equity) / peak_equity

        if drawdown_pct >= self.threshold:
            state = BreakerState(
                breaker_type='drawdown',
                is_triggered=True,
                triggered_at=datetime.now(),
                expires_at=None,  # Until recovery
                action=BreakerAction.REDUCE,
                reason=f"Drawdown {drawdown_pct:.2%} exceeds {self.threshold:.2%} - reducing position sizes by 50%"
            )

            # Only save if not already active
            existing = self.db.get_active_breakers(action='reduce')
            if not existing:
                self.db.save_breaker_state(state)
                logger.warning(f"CIRCUIT BREAKER: {state.reason}")

            return state

        # Check if we should clear an existing drawdown breaker
        existing = self.db.get_active_breakers(action='reduce')
        if existing and drawdown_pct < self.threshold * 0.5:  # Recovered to half the threshold
            self.db.clear_breaker('drawdown', cleared_by='recovery')
            logger.info("Drawdown breaker cleared - equity recovered")

        return self._not_triggered('drawdown')

    def reset(self) -> bool:
        return self.db.clear_breaker('drawdown')


class RapidLossBreaker(CircuitBreaker):
    """Temporary halt when losses occur too quickly (flash crash protection)."""

    def __init__(
        self,
        loss_pct: float = 0.01,
        window_min: int = 15,
        pause_min: int = 30,
        db: Optional[CircuitBreakerDB] = None
    ):
        super().__init__(db)
        self.loss_threshold = loss_pct
        self.window = timedelta(minutes=window_min)
        self.pause_duration = timedelta(minutes=pause_min)

    def check(self, context: Dict) -> BreakerState:
        current_equity = context.get('current_equity', 0)
        now = datetime.now()

        if current_equity <= 0:
            return self._not_triggered('rapid_loss')

        # Record current equity
        self.db.record_equity_snapshot(current_equity)

        # Get snapshots from the window
        snapshots = self.db.get_equity_snapshots(since=now - self.window)

        if not snapshots:
            return self._not_triggered('rapid_loss')

        # Compare to oldest snapshot in window
        oldest_equity = snapshots[0]['equity']
        if oldest_equity <= 0:
            return self._not_triggered('rapid_loss')

        loss_pct = (oldest_equity - current_equity) / oldest_equity

        if loss_pct >= self.loss_threshold:
            expires_at = now + self.pause_duration
            state = BreakerState(
                breaker_type='rapid_loss',
                is_triggered=True,
                triggered_at=now,
                expires_at=expires_at,
                action=BreakerAction.HALT,
                reason=f"Rapid loss {loss_pct:.2%} in {int(self.window.total_seconds()/60)} min - pausing for {int(self.pause_duration.total_seconds()/60)} min"
            )
            self.db.save_breaker_state(state)
            logger.warning(f"CIRCUIT BREAKER: {state.reason}")
            return state

        return self._not_triggered('rapid_loss')

    def reset(self) -> bool:
        return self.db.clear_breaker('rapid_loss')


class ConsecutiveLossBreaker(CircuitBreaker):
    """Pauses strategy after N consecutive losing trades."""

    def __init__(
        self,
        max_losses: int = 5,
        pause_hrs: int = 4,
        db: Optional[CircuitBreakerDB] = None
    ):
        super().__init__(db)
        self.max_losses = max_losses
        self.pause_duration = timedelta(hours=pause_hrs)

    def record_trade_result(self, strategy: str, pnl: float):
        """Called after each trade closes."""
        is_win = pnl > 0
        streak = self.db.get_streak(strategy)

        if is_win:
            self.db.update_streak(strategy, 'win', 1)
        else:
            if streak['streak_type'] == 'loss':
                new_count = streak['streak_count'] + 1
            else:
                new_count = 1
            self.db.update_streak(strategy, 'loss', new_count)

    def check(self, context: Dict) -> BreakerState:
        strategy = context.get('strategy_name')
        if not strategy:
            return self._not_triggered('consecutive_loss')

        streak = self.db.get_streak(strategy)

        if streak['streak_type'] == 'loss' and streak['streak_count'] >= self.max_losses:
            state = BreakerState(
                breaker_type='consecutive_loss',
                is_triggered=True,
                triggered_at=datetime.now(),
                expires_at=datetime.now() + self.pause_duration,
                action=BreakerAction.PAUSE_STRATEGY,
                target=strategy,
                reason=f"Strategy {strategy} has {streak['streak_count']} consecutive losses - paused for {self.pause_duration.total_seconds()/3600:.0f} hours"
            )
            self.db.save_breaker_state(state)
            logger.warning(f"CIRCUIT BREAKER: {state.reason}")
            return state

        return self._not_triggered('consecutive_loss')

    def reset(self) -> bool:
        self.db.reset_all_streaks()
        return True


class StrategyPerformanceBreaker(CircuitBreaker):
    """Disables strategy when its individual loss exceeds threshold."""

    def __init__(
        self,
        loss_pct: float = 0.05,
        pause_hrs: int = 24,
        db: Optional[CircuitBreakerDB] = None
    ):
        super().__init__(db)
        self.loss_threshold = loss_pct
        self.pause_duration = timedelta(hours=pause_hrs)

    def check(self, context: Dict) -> BreakerState:
        strategy = context.get('strategy_name')
        strategy_allocation = context.get('strategy_allocation', 0)
        strategy_pnl = context.get('strategy_realized_pnl', 0)

        if not strategy or strategy_allocation <= 0:
            return self._not_triggered('strategy_performance')

        loss_pct = -strategy_pnl / strategy_allocation if strategy_pnl < 0 else 0

        if loss_pct >= self.loss_threshold:
            state = BreakerState(
                breaker_type='strategy_performance',
                is_triggered=True,
                triggered_at=datetime.now(),
                expires_at=datetime.now() + self.pause_duration,
                action=BreakerAction.PAUSE_STRATEGY,
                target=strategy,
                reason=f"Strategy {strategy} lost {loss_pct:.2%} of allocation - paused for {self.pause_duration.total_seconds()/3600:.0f} hours"
            )
            self.db.save_breaker_state(state)
            logger.warning(f"CIRCUIT BREAKER: {state.reason}")
            return state

        return self._not_triggered('strategy_performance')

    def reset(self) -> bool:
        return self.db.clear_breaker('strategy_performance')


# =============================================================================
# KILL SWITCH MANAGER
# =============================================================================

class KillSwitchManager:
    """
    File-based and programmatic kill switch.

    File-based triggers:
        touch killswitch/HALT               # Immediate halt
        echo CONFIRM > killswitch/CLOSE_ALL # Emergency close (requires confirmation)
        touch killswitch/GRACEFUL           # Graceful shutdown
        touch killswitch/STRATEGY_gap_fill  # Disable specific strategy
    """

    def __init__(self, kill_switch_dir: Optional[Path] = None):
        self.kill_switch_dir = kill_switch_dir or DIRS.get("killswitch", Path("killswitch"))
        self.kill_switch_dir.mkdir(parents=True, exist_ok=True)
        self.db = CircuitBreakerDB()
        self.broker = None
        self.alert_manager = None

    def check_file_triggers(self) -> List[tuple]:
        """Check for file-based kill switch triggers. Returns list of (type, target)."""
        triggered = []

        # HALT
        halt_file = self.kill_switch_dir / "HALT"
        if halt_file.exists():
            triggered.append((KillSwitchType.HALT, 'all'))

        # CLOSE_ALL (requires confirmation)
        close_file = self.kill_switch_dir / "CLOSE_ALL"
        if close_file.exists():
            try:
                content = close_file.read_text().strip()
                if content == "CONFIRM":
                    triggered.append((KillSwitchType.CLOSE_ALL, 'all'))
                else:
                    logger.warning("CLOSE_ALL file exists but missing CONFIRM - ignoring")
            except Exception as e:
                logger.error(f"Failed to read CLOSE_ALL kill switch file: {e}")

        # GRACEFUL
        graceful_file = self.kill_switch_dir / "GRACEFUL"
        if graceful_file.exists():
            triggered.append((KillSwitchType.GRACEFUL, 'all'))

        # Strategy-specific
        for f in self.kill_switch_dir.glob("STRATEGY_*"):
            strategy_name = f.name.replace("STRATEGY_", "")
            triggered.append((KillSwitchType.STRATEGY, strategy_name))

        return triggered

    def is_halt_active(self) -> bool:
        """Check if HALT kill switch is active."""
        return (self.kill_switch_dir / "HALT").exists()

    def is_graceful_active(self) -> bool:
        """Check if GRACEFUL shutdown is active."""
        return (self.kill_switch_dir / "GRACEFUL").exists()

    def is_strategy_disabled(self, strategy: str) -> bool:
        """Check if a specific strategy is disabled."""
        return (self.kill_switch_dir / f"STRATEGY_{strategy}").exists()

    def execute_halt(self, triggered_by: str = "file") -> Dict:
        """Immediate halt - stop all new orders, cancel pending."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'action': 'HALT',
            'orders_cancelled': 0,
            'positions_kept': 0
        }

        if self.broker:
            try:
                # Cancel all pending orders
                open_orders = self.broker.get_orders(status='open')
                if open_orders:
                    for order in open_orders:
                        try:
                            self.broker.cancel_order(order.get('id') or order.get('order_id'))
                            result['orders_cancelled'] += 1
                        except Exception as e:
                            logger.error(f"Failed to cancel order: {e}")

                # Count positions (keep them)
                positions = self.broker.get_positions()
                result['positions_kept'] = len(positions) if positions else 0
            except Exception as e:
                logger.error(f"Error during halt execution: {e}")

        self.db.log_kill_switch('HALT', triggered_by, result)

        if self.alert_manager:
            try:
                self.alert_manager.critical(
                    "KILL SWITCH: HALT ACTIVATED",
                    f"All new orders blocked. {result['orders_cancelled']} orders cancelled. "
                    f"{result['positions_kept']} positions held."
                )
            except Exception as e:
                logger.error(f"Failed to send HALT alert: {e}")

        logger.critical(f"KILL SWITCH HALT: {result}")
        return result

    def execute_close_all(self, triggered_by: str = "file") -> Dict:
        """Emergency - close all positions immediately."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'action': 'CLOSE_ALL',
            'orders_cancelled': 0,
            'positions_closed': 0
        }

        if self.broker:
            try:
                # Cancel all orders first
                open_orders = self.broker.get_orders(status='open')
                if open_orders:
                    for order in open_orders:
                        try:
                            self.broker.cancel_order(order.get('id') or order.get('order_id'))
                            result['orders_cancelled'] += 1
                        except Exception as e:
                            logger.error(f"Failed to cancel order during CLOSE_ALL: order_id={order.get('id') or order.get('order_id')} - {e}")

                # Close all positions
                positions = self.broker.get_positions()
                result['positions_closed'] = len(positions) if positions else 0

                if hasattr(self.broker, 'close_all_positions'):
                    self.broker.close_all_positions()
                else:
                    # Fallback: close each position individually
                    for pos in (positions or []):
                        try:
                            symbol = pos.get('symbol')
                            if symbol:
                                self.broker.close_position(symbol)
                        except Exception as e:
                            logger.error(f"Failed to close position: {e}")

            except Exception as e:
                logger.error(f"Error during close all: {e}")

        self.db.log_kill_switch('CLOSE_ALL', triggered_by, result)

        if self.alert_manager:
            try:
                self.alert_manager.critical(
                    "EMERGENCY: CLOSE ALL POSITIONS",
                    f"Emergency liquidation executed. {result['positions_closed']} positions closed. "
                    f"{result['orders_cancelled']} orders cancelled."
                )
            except Exception as e:
                logger.error(f"Failed to send CLOSE_ALL alert: {e}")

        logger.critical(f"KILL SWITCH CLOSE_ALL: {result}")
        return result

    def execute_graceful(self, triggered_by: str = "file") -> Dict:
        """Graceful - stop new signals, let pending orders complete."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'action': 'GRACEFUL',
            'pending_orders': 0
        }

        if self.broker:
            try:
                orders = self.broker.get_orders(status='open')
                result['pending_orders'] = len(orders) if orders else 0
            except Exception as e:
                logger.error(f"Failed to get open orders during GRACEFUL shutdown: {e}")

        self.db.log_kill_switch('GRACEFUL', triggered_by, result)

        if self.alert_manager:
            try:
                self.alert_manager.critical(
                    "GRACEFUL SHUTDOWN INITIATED",
                    f"No new signals will be generated. {result['pending_orders']} pending orders will complete."
                )
            except Exception as e:
                logger.error(f"Failed to send GRACEFUL alert: {e}")

        logger.warning(f"GRACEFUL SHUTDOWN: {result}")
        return result

    def disable_strategy(self, strategy: str, triggered_by: str = "file") -> Dict:
        """Disable a specific strategy."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'action': f'DISABLE_{strategy}',
            'strategy': strategy
        }

        self.db.log_kill_switch(f'STRATEGY_{strategy}', triggered_by, result)

        if self.alert_manager:
            try:
                self.alert_manager.critical(
                    f"STRATEGY DISABLED: {strategy}",
                    f"Strategy {strategy} has been disabled via kill switch."
                )
            except Exception as e:
                logger.error(f"Failed to send strategy disable alert for {strategy}: {e}")

        logger.warning(f"STRATEGY DISABLED: {strategy}")
        return result

    def clear_switch(self, switch_type: str) -> bool:
        """Clear a kill switch (remove file)."""
        if switch_type.startswith('STRATEGY_'):
            file_path = self.kill_switch_dir / switch_type
        else:
            file_path = self.kill_switch_dir / switch_type

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Kill switch cleared: {switch_type}")
            return True

        return False

    def activate_halt(self):
        """Programmatically activate HALT."""
        (self.kill_switch_dir / "HALT").touch()
        self.execute_halt(triggered_by="programmatic")

    def activate_graceful(self):
        """Programmatically activate GRACEFUL shutdown."""
        (self.kill_switch_dir / "GRACEFUL").touch()
        self.execute_graceful(triggered_by="programmatic")


# =============================================================================
# CIRCUIT BREAKER MANAGER (MAIN COORDINATOR)
# =============================================================================

class CircuitBreakerManager:
    """
    Central coordinator for all circuit breakers and kill switches.

    Usage:
        manager = CircuitBreakerManager()
        manager.inject_dependencies(broker, alert_manager)

        # Check before any order
        if manager.can_trade():
            execute_order(...)

        # Check before strategy runs
        if manager.can_run_strategy('gap_fill'):
            strategy.generate_signals()

        # Get position size multiplier
        multiplier = manager.get_position_multiplier()
        position_size *= multiplier
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.db = CircuitBreakerDB()

        # Initialize breakers
        self.breakers = {
            'daily_loss': DailyLossBreaker(self.config.daily_loss_pct, self.db),
            'drawdown': DrawdownBreaker(self.config.drawdown_pct, self.db),
            'rapid_loss': RapidLossBreaker(
                self.config.rapid_loss_pct,
                self.config.rapid_loss_window_min,
                self.config.rapid_loss_pause_min,
                self.db
            ),
            'consecutive_loss': ConsecutiveLossBreaker(
                self.config.max_consecutive_losses,
                self.config.consecutive_loss_pause_hrs,
                self.db
            ),
            'strategy_performance': StrategyPerformanceBreaker(
                self.config.strategy_loss_pct,
                self.config.strategy_pause_hrs,
                self.db
            ),
        }

        # Initialize kill switch manager
        self.kill_switch = KillSwitchManager()

        self.broker = None
        self.alert_manager = None

        # Thread safety for state checks and modifications
        # Ensures atomicity during signal evaluation
        self._state_lock = threading.Lock()

        # Load persisted state
        self._load_persisted_state()

    def _load_persisted_state(self):
        """Load and log active circuit breaker states from database."""
        # Clear expired breakers first
        cleared = self.db.clear_expired_breakers()
        if cleared:
            logger.info(f"Cleared {cleared} expired circuit breakers")

        # Log any still-active breakers
        active_states = self.db.get_active_breakers()
        for state in active_states:
            logger.warning(
                f"Active circuit breaker from previous session: "
                f"{state.breaker_type} - {state.reason}"
            )

    def inject_dependencies(self, broker=None, alert_manager=None):
        """Inject broker and alert manager dependencies."""
        self.broker = broker
        self.alert_manager = alert_manager
        self.kill_switch.broker = broker
        self.kill_switch.alert_manager = alert_manager

    def check_all(self, context: Dict) -> List[BreakerState]:
        """
        Check all circuit breakers and file triggers. Returns triggered states.

        Thread-safe: Uses lock to ensure atomic state checks during signal evaluation.
        """
        with self._state_lock:
            triggered = []

            # First check file-based kill switches
            file_triggers = self.kill_switch.check_file_triggers()
            for trigger_type, target in file_triggers:
                if trigger_type == KillSwitchType.HALT:
                    self.kill_switch.execute_halt()
                elif trigger_type == KillSwitchType.CLOSE_ALL:
                    self.kill_switch.execute_close_all()
                elif trigger_type == KillSwitchType.GRACEFUL:
                    self.kill_switch.execute_graceful()
                elif trigger_type == KillSwitchType.STRATEGY:
                    self.kill_switch.disable_strategy(target)

            # Then check automatic circuit breakers
            for name, breaker in self.breakers.items():
                try:
                    state = breaker.check(context)
                    if state.is_triggered:
                        triggered.append(state)
                except Exception as e:
                    logger.error(f"Error checking breaker {name}: {e}")

            return triggered

    def can_trade(self) -> bool:
        """
        Check if trading is allowed (no halt-type breakers active).

        Thread-safe: Uses lock for consistent state checks.
        """
        with self._state_lock:
            # Check kill switches first (fast path)
            if self.kill_switch.is_halt_active():
                return False
            if self.kill_switch.is_graceful_active():
                return False

            # Check database for active halt breakers
            now = datetime.now()
            active_halts = self.db.get_active_breakers(action='halt')

            for state in active_halts:
                if state.expires_at is None or now < state.expires_at:
                    return False

            return True

    def can_run_strategy(self, strategy_name: str) -> bool:
        """
        Check if a specific strategy can run.

        Thread-safe: Uses lock for consistent state checks.
        """
        with self._state_lock:
            # Check kill switches first (fast path)
            if self.kill_switch.is_halt_active():
                return False
            if self.kill_switch.is_graceful_active():
                return False

            # Check database for active halt breakers
            now = datetime.now()
            active_halts = self.db.get_active_breakers(action='halt')
            for state in active_halts:
                if state.expires_at is None or now < state.expires_at:
                    return False

            # Check strategy-specific kill switch
            if self.kill_switch.is_strategy_disabled(strategy_name):
                return False

            # Check strategy-specific breakers
            active_pauses = self.db.get_active_breakers(action='pause_strategy', target=strategy_name)

            for state in active_pauses:
                if state.expires_at is None or now < state.expires_at:
                    return False

            return True

    def get_position_multiplier(self) -> float:
        """Get position size multiplier (1.0 = normal, 0.5 = reduced due to drawdown)."""
        active_reduce = self.db.get_active_breakers(action='reduce')

        for state in active_reduce:
            now = datetime.now()
            if state.expires_at is None or now < state.expires_at:
                return 0.5  # Reduce by 50%

        return 1.0

    def record_trade_close(self, strategy: str, pnl: float):
        """Record a trade close for streak tracking."""
        self.breakers['consecutive_loss'].record_trade_result(strategy, pnl)

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        active = self.db.get_active_breakers()
        file_switches = list(self.kill_switch.kill_switch_dir.glob("*"))

        return {
            'trading_allowed': self.can_trade(),
            'position_multiplier': self.get_position_multiplier(),
            'active_breakers': [
                {
                    'type': s.breaker_type,
                    'reason': s.reason,
                    'action': s.action.value,
                    'triggered_at': s.triggered_at.isoformat() if s.triggered_at else None,
                    'expires_at': s.expires_at.isoformat() if s.expires_at else None,
                    'target': s.target
                }
                for s in active
            ],
            'file_kill_switches': [f.name for f in file_switches if f.is_file()]
        }

    def clear_breaker(self, breaker_type: str, target: str = 'all') -> bool:
        """Manually clear a circuit breaker."""
        success = self.db.clear_breaker(breaker_type, target, cleared_by='manual')

        if success:
            logger.info(f"Circuit breaker cleared: {breaker_type} ({target})")
            if self.alert_manager:
                try:
                    self.alert_manager.send_alert(
                        f"Circuit breaker cleared: {breaker_type} ({target})",
                        level="info"
                    )
                except Exception as e:
                    logger.error(f"Failed to send breaker cleared alert: {breaker_type} ({target}) - {e}")

        return success

    def daily_reset(self):
        """Called at start of each trading day."""
        # Clear daily loss breaker
        self.db.clear_breaker('daily_loss', 'all', cleared_by='daily_reset')

        # Don't reset streaks - they should persist across days
        logger.info("Daily circuit breaker reset complete")

    def emergency_halt(self):
        """Programmatic emergency halt."""
        self.kill_switch.activate_halt()

    def emergency_close_all(self):
        """Programmatic emergency close all."""
        (self.kill_switch.kill_switch_dir / "CLOSE_ALL").write_text("CONFIRM")
        self.kill_switch.execute_close_all(triggered_by="programmatic")


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create the global circuit breaker manager."""
    global _manager
    if _manager is None:
        _manager = CircuitBreakerManager()
    return _manager


def reset_circuit_breaker_manager():
    """Reset the global circuit breaker manager (for testing)."""
    global _manager
    _manager = None
