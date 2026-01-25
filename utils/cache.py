"""
Unified Cache Utilities
=======================
Provides LRU caching utilities with consistent limits and monitoring
for all caching needs across the trading system.

Usage:
    from utils.cache import LRUCache, lru_cached, CacheStats

    # Standalone LRU cache
    cache = LRUCache(maxsize=200)
    cache["key"] = value
    result = cache.get("key")

    # Method decorator with LRU caching
    class MyClass:
        @lru_cached(maxsize=100)
        def expensive_method(self, arg):
            ...

Design Principles:
- Proactive eviction (evict BEFORE adding, not after)
- Explicit size limits everywhere
- Optional statistics for monitoring
- Thread-safe operations
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generic, Hashable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')
T = TypeVar('T')


# =============================================================================
# CACHE STATISTICS
# =============================================================================

@dataclass
class CacheStats:
    """Statistics for cache monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    maxsize: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def utilization(self) -> float:
        """Calculate cache utilization (0.0 to 1.0)."""
        return self.size / self.maxsize if self.maxsize > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for logging/metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "maxsize": self.maxsize,
            "hit_rate": f"{self.hit_rate:.2%}",
            "utilization": f"{self.utilization:.2%}"
        }

    def reset(self):
        """Reset statistics (keeps maxsize)."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0


# =============================================================================
# LRU CACHE CLASS
# =============================================================================

class LRUCache(Generic[K, V]):
    """
    Thread-safe Least Recently Used (LRU) cache with explicit limits.

    Features:
    - Proactive eviction (evicts before adding when at capacity)
    - Optional statistics tracking
    - Thread-safe with lock
    - Dict-like interface

    Example:
        cache = LRUCache(maxsize=200)
        cache["key"] = expensive_result
        result = cache.get("key", default=None)

        # With stats monitoring
        cache = LRUCache(maxsize=200, track_stats=True)
        print(cache.stats.hit_rate)
    """

    def __init__(
        self,
        maxsize: int = 128,
        track_stats: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize LRU cache.

        Args:
            maxsize: Maximum number of items to store
            track_stats: Whether to track cache statistics
            name: Optional name for logging
        """
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")

        self.maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self._name = name or "LRUCache"

        self._track_stats = track_stats
        self._stats = CacheStats(maxsize=maxsize) if track_stats else None

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get an item from cache, moving it to end (most recently used).

        Args:
            key: Cache key
            default: Value to return if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                if self._stats:
                    self._stats.hits += 1
                return self._cache[key]

            if self._stats:
                self._stats.misses += 1
            return default

    def set(self, key: K, value: V) -> None:
        """
        Set an item in cache, evicting oldest if at capacity.

        Eviction happens BEFORE adding to ensure we never exceed maxsize.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing - move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
                return

            # Evict BEFORE adding if at capacity (proactive eviction)
            while len(self._cache) >= self.maxsize:
                oldest_key, _ = self._cache.popitem(last=False)
                if self._stats:
                    self._stats.evictions += 1
                logger.debug(f"{self._name}: Evicted {oldest_key}")

            # Add new item
            self._cache[key] = value
            if self._stats:
                self._stats.size = len(self._cache)

    def __getitem__(self, key: K) -> V:
        """Get item (raises KeyError if not found)."""
        result = self.get(key)
        if result is None and key not in self._cache:
            raise KeyError(key)
        return result

    def __setitem__(self, key: K, value: V) -> None:
        """Set item."""
        self.set(key, value)

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Return number of items in cache."""
        with self._lock:
            return len(self._cache)

    def __delitem__(self, key: K) -> None:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self._stats:
                    self._stats.size = len(self._cache)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return item."""
        with self._lock:
            result = self._cache.pop(key, default)
            if self._stats:
                self._stats.size = len(self._cache)
            return result

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            cleared = len(self._cache)
            self._cache.clear()
            if self._stats:
                self._stats.size = 0
            logger.debug(f"{self._name}: Cleared {cleared} items")

    def keys(self):
        """Return cache keys (most recently used last)."""
        with self._lock:
            return list(self._cache.keys())

    def values(self):
        """Return cache values (most recently used last)."""
        with self._lock:
            return list(self._cache.values())

    def items(self):
        """Return cache items (most recently used last)."""
        with self._lock:
            return list(self._cache.items())

    @property
    def stats(self) -> Optional[CacheStats]:
        """Get cache statistics (None if tracking disabled)."""
        if not self._stats:
            return None

        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats

    def get_or_compute(
        self,
        key: K,
        compute_fn: Callable[[], V]
    ) -> V:
        """
        Get item from cache or compute and cache it.

        Args:
            key: Cache key
            compute_fn: Function to call if key not in cache

        Returns:
            Cached or computed value
        """
        # Try to get existing value first
        result = self.get(key)
        if result is not None:
            return result

        # Not in cache - need to check if key exists but has None value
        with self._lock:
            if key in self._cache:
                return self._cache[key]

            # Compute new value
            value = compute_fn()
            self.set(key, value)
            return value


# =============================================================================
# LRU CACHED DECORATOR
# =============================================================================

def lru_cached(
    maxsize: int = 128,
    key_fn: Optional[Callable[..., Hashable]] = None,
    track_stats: bool = False,
    cache_name: Optional[str] = None
):
    """
    Decorator to add LRU caching to a method.

    Unlike functools.lru_cache, this:
    - Works properly with instance methods (caches per-instance)
    - Has explicit maxsize enforcement
    - Supports custom key functions
    - Provides optional statistics

    Args:
        maxsize: Maximum cache size per instance
        key_fn: Optional function to compute cache key from arguments.
                If None, uses tuple of all arguments.
        track_stats: Whether to track cache statistics
        cache_name: Optional name for the cache (for logging)

    Example:
        class MyClass:
            @lru_cached(maxsize=100)
            def expensive_method(self, x, y):
                return x + y

            @lru_cached(maxsize=50, key_fn=lambda self, x: x)
            def method_with_custom_key(self, x, metadata=None):
                return process(x)

        obj = MyClass()
        obj.expensive_method(1, 2)  # Computed
        obj.expensive_method(1, 2)  # From cache
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Store cache on each instance using a unique attribute name
        cache_attr = f"_lru_cache_{func.__name__}"
        name = cache_name or f"{func.__name__}_cache"

        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            # Get or create cache for this instance
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, LRUCache(
                    maxsize=maxsize,
                    track_stats=track_stats,
                    name=name
                ))
            cache: LRUCache = getattr(self, cache_attr)

            # Compute cache key
            if key_fn is not None:
                key = key_fn(self, *args, **kwargs)
            else:
                # Default: tuple of all args
                key = (args, tuple(sorted(kwargs.items())))

            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result

            # Check if key exists but has None value
            if key in cache:
                return cache[key]

            # Compute and cache
            result = func(self, *args, **kwargs)
            cache.set(key, result)
            return result

        # Add method to clear cache
        def clear_cache(self):
            if hasattr(self, cache_attr):
                getattr(self, cache_attr).clear()

        # Add method to get cache stats
        def cache_stats(self) -> Optional[CacheStats]:
            if hasattr(self, cache_attr):
                return getattr(self, cache_attr).stats
            return None

        wrapper.clear_cache = clear_cache
        wrapper.cache_stats = cache_stats
        wrapper._cache_attr = cache_attr

        return wrapper
    return decorator


# =============================================================================
# TIME-BASED CACHE
# =============================================================================

@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with timestamp."""
    value: V
    timestamp: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.timestamp + self.ttl


class TTLCache(Generic[K, V]):
    """
    LRU cache with time-to-live (TTL) for entries.

    Items expire after TTL seconds. Expired items are evicted on access
    or during periodic cleanup.

    Example:
        cache = TTLCache(maxsize=100, ttl=60.0)  # 60 second TTL
        cache["key"] = value
        # After 60 seconds, cache["key"] will return None
    """

    def __init__(
        self,
        maxsize: int = 128,
        ttl: float = 60.0,
        track_stats: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize TTL cache.

        Args:
            maxsize: Maximum number of items
            ttl: Time-to-live in seconds for each entry
            track_stats: Whether to track statistics
            name: Optional name for logging
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._name = name or "TTLCache"

        self._track_stats = track_stats
        self._stats = CacheStats(maxsize=maxsize) if track_stats else None

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get item, returning default if expired or missing."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                if entry.is_expired:
                    # Expired - remove and return default
                    del self._cache[key]
                    if self._stats:
                        self._stats.evictions += 1
                        self._stats.misses += 1
                    return default

                # Valid - move to end and return
                self._cache.move_to_end(key)
                if self._stats:
                    self._stats.hits += 1
                return entry.value

            if self._stats:
                self._stats.misses += 1
            return default

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set item with optional custom TTL."""
        entry_ttl = ttl if ttl is not None else self.ttl

        with self._lock:
            # Proactive eviction
            while len(self._cache) >= self.maxsize:
                oldest_key, _ = self._cache.popitem(last=False)
                if self._stats:
                    self._stats.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=entry_ttl
            )

            if key in self._cache:
                self._cache.move_to_end(key)

            if self._stats:
                self._stats.size = len(self._cache)

    def __getitem__(self, key: K) -> V:
        result = self.get(key)
        if result is None and key not in self._cache:
            raise KeyError(key)
        return result

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._cache.clear()
            if self._stats:
                self._stats.size = 0

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                if self._stats:
                    self._stats.evictions += 1

            if self._stats:
                self._stats.size = len(self._cache)

            return len(expired_keys)

    def refresh(self, key: K) -> bool:
        """
        Refresh TTL on an existing entry.

        Returns True if key exists and was refreshed, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired:
                    self._cache[key] = CacheEntry(
                        value=entry.value,
                        timestamp=time.time(),
                        ttl=entry.ttl
                    )
                    self._cache.move_to_end(key)
                    return True
            return False

    @property
    def stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        if not self._stats:
            return None

        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats
