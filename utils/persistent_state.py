"""
Unified State Persistence Utilities
====================================
Provides persistent state management that survives restarts.
DB is the source of truth; memory is a cache.

Usage:
    from utils.persistent_state import PersistentCounter, PersistentSet, StateManager

    # Persistent counter
    counter = PersistentCounter(db, "pending_approvals", "strategy_a")
    counter.increment()
    count = counter.value

    # Persistent set
    pending = PersistentSet(db, "pending_symbols")
    pending.add("AAPL")
    if "AAPL" in pending:
        ...

Design Principles:
- DB is authority, memory is cache
- All state survives process restart
- Automatic cleanup of stale state on startup
- Clear error propagation (no silent failures)
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# PERSISTENT COUNTER
# =============================================================================

class PersistentCounter:
    """
    Counter that persists across restarts via database.

    Uses a simple key-value table in the database. Value is always
    synced to DB on modification.

    Example:
        counter = PersistentCounter(db, "execution", "pending_approvals")
        counter.increment()
        print(counter.value)  # Reads from memory (DB-backed)
        counter.reset()

    Table schema (auto-created if needed):
        persistent_counters(
            namespace TEXT,
            key TEXT,
            value INTEGER,
            updated_at TEXT,
            PRIMARY KEY(namespace, key)
        )
    """

    def __init__(
        self,
        db_manager,
        namespace: str,
        key: str,
        db_name: str = "trades"
    ):
        """
        Initialize persistent counter.

        Args:
            db_manager: DatabaseManager instance
            namespace: Grouping namespace (e.g., "execution", "research")
            key: Counter key within namespace
            db_name: Which database to use
        """
        self.db = db_manager
        self.namespace = namespace
        self.key = key
        self.db_name = db_name
        self._lock = threading.Lock()
        self._value: int = 0

        self._ensure_table()
        self._load()

    def _ensure_table(self) -> None:
        """Create table if it doesn't exist."""
        self.db.execute(
            self.db_name,
            """
            CREATE TABLE IF NOT EXISTS persistent_counters (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            )
            """
        )

    def _load(self) -> None:
        """Load current value from database."""
        row = self.db.fetchone(
            self.db_name,
            "SELECT value FROM persistent_counters WHERE namespace = ? AND key = ?",
            (self.namespace, self.key)
        )
        self._value = row['value'] if row else 0

    def _save(self) -> None:
        """Save current value to database."""
        self.db.execute(
            self.db_name,
            """
            INSERT OR REPLACE INTO persistent_counters (namespace, key, value, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (self.namespace, self.key, self._value)
        )

    @property
    def value(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value

    def increment(self, amount: int = 1) -> int:
        """Increment counter and persist. Returns new value."""
        with self._lock:
            self._value += amount
            self._save()
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and persist. Returns new value."""
        with self._lock:
            self._value = max(0, self._value - amount)
            self._save()
            return self._value

    def set(self, value: int) -> None:
        """Set counter to specific value."""
        with self._lock:
            self._value = value
            self._save()

    def reset(self) -> None:
        """Reset counter to zero."""
        self.set(0)

    def refresh(self) -> int:
        """Reload value from database. Returns current value."""
        with self._lock:
            self._load()
            return self._value


# =============================================================================
# PERSISTENT DICT
# =============================================================================

class PersistentDict(Generic[T]):
    """
    Dict that persists across restarts via database.

    Values are JSON-serialized. Updates are immediately persisted.

    Example:
        approvals = PersistentDict[int](db, "execution", "pending_by_strategy")
        approvals["momentum"] = 3
        print(approvals.get("momentum", 0))

    Table schema (auto-created if needed):
        persistent_dicts(
            namespace TEXT,
            dict_name TEXT,
            key TEXT,
            value TEXT,  -- JSON encoded
            updated_at TEXT,
            PRIMARY KEY(namespace, dict_name, key)
        )
    """

    def __init__(
        self,
        db_manager,
        namespace: str,
        dict_name: str,
        db_name: str = "trades",
        default_factory: Optional[Callable[[], T]] = None
    ):
        """
        Initialize persistent dict.

        Args:
            db_manager: DatabaseManager instance
            namespace: Grouping namespace
            dict_name: Dict name within namespace
            db_name: Which database to use
            default_factory: Optional factory for default values
        """
        self.db = db_manager
        self.namespace = namespace
        self.dict_name = dict_name
        self.db_name = db_name
        self.default_factory = default_factory
        self._lock = threading.Lock()
        self._cache: Dict[str, T] = {}

        self._ensure_table()
        self._load_all()

    def _ensure_table(self) -> None:
        """Create table if it doesn't exist."""
        self.db.execute(
            self.db_name,
            """
            CREATE TABLE IF NOT EXISTS persistent_dicts (
                namespace TEXT NOT NULL,
                dict_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, dict_name, key)
            )
            """
        )

    def _load_all(self) -> None:
        """Load all values from database."""
        rows = self.db.fetchall(
            self.db_name,
            """
            SELECT key, value FROM persistent_dicts
            WHERE namespace = ? AND dict_name = ?
            """,
            (self.namespace, self.dict_name)
        )
        self._cache = {row['key']: json.loads(row['value']) for row in rows}

    def _save(self, key: str, value: T) -> None:
        """Save single key-value pair to database."""
        self.db.execute(
            self.db_name,
            """
            INSERT OR REPLACE INTO persistent_dicts
            (namespace, dict_name, key, value, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (self.namespace, self.dict_name, key, json.dumps(value))
        )

    def _delete(self, key: str) -> None:
        """Delete key from database."""
        self.db.execute(
            self.db_name,
            """
            DELETE FROM persistent_dicts
            WHERE namespace = ? AND dict_name = ? AND key = ?
            """,
            (self.namespace, self.dict_name, key)
        )

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get value for key, returning default if not found."""
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self.default_factory and default is None:
                return self.default_factory()
            return default

    def __getitem__(self, key: str) -> T:
        """Get value (raises KeyError if not found and no default_factory)."""
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self.default_factory:
                value = self.default_factory()
                self._cache[key] = value
                self._save(key, value)
                return value
            raise KeyError(key)

    def __setitem__(self, key: str, value: T) -> None:
        """Set value for key."""
        with self._lock:
            self._cache[key] = value
            self._save(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._delete(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Return number of items."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> List[str]:
        """Return all keys."""
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[T]:
        """Return all values."""
        with self._lock:
            return list(self._cache.values())

    def items(self) -> List[tuple]:
        """Return all items."""
        with self._lock:
            return list(self._cache.items())

    def pop(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Remove and return value."""
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._delete(key)
                return value
            return default

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._cache.clear()
            self.db.execute(
                self.db_name,
                """
                DELETE FROM persistent_dicts
                WHERE namespace = ? AND dict_name = ?
                """,
                (self.namespace, self.dict_name)
            )

    def refresh(self) -> None:
        """Reload all values from database."""
        with self._lock:
            self._load_all()


# =============================================================================
# PERSISTENT SET
# =============================================================================

class PersistentSet:
    """
    Set that persists across restarts via database.

    Useful for tracking pending items (symbols, orders, etc.).

    Example:
        pending = PersistentSet(db, "pending_orders")
        pending.add("AAPL", order_id="123", created_at=datetime.now())
        if "AAPL" in pending:
            ...
        pending.remove("AAPL")

    Table schema (auto-created if needed):
        persistent_sets(
            set_name TEXT,
            item TEXT,
            metadata TEXT,  -- JSON for extra data
            created_at TEXT,
            PRIMARY KEY(set_name, item)
        )
    """

    def __init__(
        self,
        db_manager,
        set_name: str,
        db_name: str = "trades"
    ):
        """
        Initialize persistent set.

        Args:
            db_manager: DatabaseManager instance
            set_name: Name of the set
            db_name: Which database to use
        """
        self.db = db_manager
        self.set_name = set_name
        self.db_name = db_name
        self._lock = threading.Lock()
        self._cache: Set[str] = set()

        self._ensure_table()
        self._load()

    def _ensure_table(self) -> None:
        """Create table if it doesn't exist."""
        self.db.execute(
            self.db_name,
            """
            CREATE TABLE IF NOT EXISTS persistent_sets (
                set_name TEXT NOT NULL,
                item TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (set_name, item)
            )
            """
        )

    def _load(self) -> None:
        """Load all items from database."""
        rows = self.db.fetchall(
            self.db_name,
            "SELECT item FROM persistent_sets WHERE set_name = ?",
            (self.set_name,)
        )
        self._cache = {row['item'] for row in rows}

    def add(self, item: str, **metadata) -> bool:
        """
        Add item to set. Returns True if item was new.

        Args:
            item: Item to add
            **metadata: Optional metadata to store with item
        """
        with self._lock:
            if item in self._cache:
                return False

            self._cache.add(item)
            self.db.execute(
                self.db_name,
                """
                INSERT OR REPLACE INTO persistent_sets
                (set_name, item, metadata, created_at)
                VALUES (?, ?, ?, datetime('now'))
                """,
                (self.set_name, item, json.dumps(metadata) if metadata else None)
            )
            return True

    def remove(self, item: str) -> bool:
        """Remove item from set. Returns True if item was present."""
        with self._lock:
            if item not in self._cache:
                return False

            self._cache.discard(item)
            self.db.execute(
                self.db_name,
                "DELETE FROM persistent_sets WHERE set_name = ? AND item = ?",
                (self.set_name, item)
            )
            return True

    def discard(self, item: str) -> None:
        """Remove item if present (no error if not)."""
        self.remove(item)

    def __contains__(self, item: str) -> bool:
        """Check if item is in set."""
        with self._lock:
            return item in self._cache

    def __len__(self) -> int:
        """Return number of items."""
        with self._lock:
            return len(self._cache)

    def __iter__(self):
        """Iterate over items."""
        with self._lock:
            return iter(list(self._cache))

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._cache.clear()
            self.db.execute(
                self.db_name,
                "DELETE FROM persistent_sets WHERE set_name = ?",
                (self.set_name,)
            )

    def get_with_metadata(self, item: str) -> Optional[Dict[str, Any]]:
        """Get item with its metadata."""
        row = self.db.fetchone(
            self.db_name,
            """
            SELECT item, metadata, created_at FROM persistent_sets
            WHERE set_name = ? AND item = ?
            """,
            (self.set_name, item)
        )
        if row:
            return {
                'item': row['item'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'created_at': row['created_at']
            }
        return None

    def clear_stale(self, max_age_seconds: float) -> int:
        """
        Remove items older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of items removed
        """
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

        with self._lock:
            # Find stale items
            rows = self.db.fetchall(
                self.db_name,
                """
                SELECT item FROM persistent_sets
                WHERE set_name = ? AND created_at < ?
                """,
                (self.set_name, cutoff_str)
            )
            stale_items = {row['item'] for row in rows}

            if not stale_items:
                return 0

            # Remove from cache and DB
            self._cache -= stale_items
            self.db.execute(
                self.db_name,
                """
                DELETE FROM persistent_sets
                WHERE set_name = ? AND created_at < ?
                """,
                (self.set_name, cutoff_str)
            )

            logger.info(f"Cleared {len(stale_items)} stale items from {self.set_name}")
            return len(stale_items)

    def refresh(self) -> None:
        """Reload from database."""
        with self._lock:
            self._load()


# =============================================================================
# STATE MANAGER BASE CLASS
# =============================================================================

class StateManager(ABC):
    """
    Base class for components that need state persistence.

    Subclasses implement save_state() and load_state() to persist
    their specific state to DB.

    Example:
        class MyManager(StateManager):
            def __init__(self, db):
                super().__init__(db, "my_manager")
                self.counter = 0
                self.load_state()

            def save_state(self):
                self._save_json_state({"counter": self.counter})

            def load_state(self):
                state = self._load_json_state()
                if state:
                    self.counter = state.get("counter", 0)
    """

    def __init__(
        self,
        db_manager,
        state_name: str,
        db_name: str = "trades"
    ):
        """
        Initialize state manager.

        Args:
            db_manager: DatabaseManager instance
            state_name: Unique name for this manager's state
            db_name: Which database to use
        """
        self.db = db_manager
        self.state_name = state_name
        self.db_name = db_name
        self._state_lock = threading.Lock()

        self._ensure_state_table()

    def _ensure_state_table(self) -> None:
        """Create state table if it doesn't exist."""
        self.db.execute(
            self.db_name,
            """
            CREATE TABLE IF NOT EXISTS component_state (
                state_name TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _save_json_state(self, state: Dict[str, Any]) -> None:
        """Save state as JSON to database."""
        with self._state_lock:
            self.db.execute(
                self.db_name,
                """
                INSERT OR REPLACE INTO component_state (state_name, state_json, updated_at)
                VALUES (?, ?, datetime('now'))
                """,
                (self.state_name, json.dumps(state))
            )

    def _load_json_state(self) -> Optional[Dict[str, Any]]:
        """Load state from database."""
        with self._state_lock:
            row = self.db.fetchone(
                self.db_name,
                "SELECT state_json FROM component_state WHERE state_name = ?",
                (self.state_name,)
            )
            if row:
                return json.loads(row['state_json'])
            return None

    def _clear_state(self) -> None:
        """Clear persisted state."""
        with self._state_lock:
            self.db.execute(
                self.db_name,
                "DELETE FROM component_state WHERE state_name = ?",
                (self.state_name,)
            )

    @abstractmethod
    def save_state(self) -> None:
        """Save current state to database. Implement in subclass."""
        pass

    @abstractmethod
    def load_state(self) -> None:
        """Load state from database. Implement in subclass."""
        pass

    def recovery_on_startup(self) -> None:
        """
        Optional hook for recovery logic on startup.

        Default implementation just loads state. Override in subclass
        for custom recovery (e.g., clearing stale data).
        """
        self.load_state()


# =============================================================================
# STARTUP CLEANUP UTILITIES
# =============================================================================

def cleanup_stale_state(
    db_manager,
    db_name: str = "trades",
    max_age_hours: float = 1.0
) -> Dict[str, int]:
    """
    Cleanup stale persistent state on startup.

    Call this once on system startup to clear orphaned state from crashes.

    Args:
        db_manager: DatabaseManager instance
        db_name: Database to clean
        max_age_hours: Maximum age in hours before state is considered stale

    Returns:
        Dict with cleanup statistics
    """
    stats = {
        'counters_reset': 0,
        'set_items_removed': 0,
        'dict_items_removed': 0
    }

    max_age_seconds = max_age_hours * 3600
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

    # Clean stale set items
    try:
        cursor = db_manager.execute(
            db_name,
            """
            DELETE FROM persistent_sets
            WHERE created_at < ?
            """,
            (cutoff_str,)
        )
        stats['set_items_removed'] = cursor.rowcount
    except Exception as e:
        logger.warning(f"Could not cleanup persistent_sets: {e}")

    # Clean stale dict items
    try:
        cursor = db_manager.execute(
            db_name,
            """
            DELETE FROM persistent_dicts
            WHERE updated_at < ?
            """,
            (cutoff_str,)
        )
        stats['dict_items_removed'] = cursor.rowcount
    except Exception as e:
        logger.warning(f"Could not cleanup persistent_dicts: {e}")

    total = sum(stats.values())
    if total > 0:
        logger.info(f"Startup cleanup: removed {total} stale state items - {stats}")

    return stats
