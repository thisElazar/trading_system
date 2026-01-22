"""
Shared Memory Data Manager
==========================
Manages market data in shared memory for efficient parallel processing.

Instead of copying ~500MB of data to each worker process, we store
the data in shared memory that all workers can access read-only.

Usage:
    # In main process:
    manager = SharedDataManager()
    manager.load_data(data_dict, vix_data)

    # In worker processes:
    data, vix = manager.get_data()

    # Cleanup:
    manager.cleanup()
"""

import logging
import numpy as np
import pandas as pd
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SharedArrayInfo:
    """Metadata for a shared memory array."""
    name: str           # Shared memory block name
    shape: Tuple        # Array shape
    dtype: str          # Numpy dtype string
    columns: List[str]  # DataFrame column names
    index_name: str     # Index name


class SharedDataManager:
    """
    Manages market data in shared memory for parallel workers.

    Data is stored as numpy arrays in shared memory blocks.
    Metadata (column names, dtypes) is stored separately and
    passed to workers to reconstruct DataFrames.
    """

    def __init__(self, prefix: str = "trading"):
        self.prefix = prefix
        self._shared_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._metadata: Dict[str, SharedArrayInfo] = {}
        self._is_owner = False  # True if this instance created the shared memory

    def _generate_block_name(self, symbol: str) -> str:
        """Generate unique shared memory block name."""
        # Use hash to keep names short (shm names have length limits)
        h = hashlib.md5(f"{self.prefix}_{symbol}".encode()).hexdigest()[:12]
        return f"shm_{h}"

    def load_data(self, data: Dict[str, pd.DataFrame], vix_data: Optional[pd.DataFrame] = None):
        """
        Load market data into shared memory.

        Args:
            data: Dict mapping symbol -> DataFrame with OHLCV data
            vix_data: Optional VIX DataFrame
        """
        self._is_owner = True
        logger.info(f"Loading {len(data)} symbols into shared memory...")

        # Store each symbol's data
        for symbol, df in data.items():
            self._store_dataframe(symbol, df)

        # Store VIX separately
        if vix_data is not None:
            self._store_dataframe("__VIX__", vix_data)

        logger.info(f"Shared memory initialized: {len(self._shared_blocks)} blocks, "
                   f"~{self._total_size_mb():.1f}MB")

    def _store_dataframe(self, key: str, df: pd.DataFrame):
        """Store a DataFrame in shared memory."""
        # Reset index to make it a column
        df_reset = df.reset_index()

        # Convert to numpy record array for efficient storage
        data_to_store = df_reset.copy()

        # Convert timestamp/datetime columns to float (unix timestamp)
        for col in data_to_store.columns:
            if pd.api.types.is_datetime64_any_dtype(data_to_store[col]):
                data_to_store[col] = data_to_store[col].astype(np.int64) / 1e9

        # Convert to numpy array (all columns as float64)
        try:
            arr = data_to_store.values.astype(np.float64)
            columns_to_store = list(df_reset.columns)
        except (ValueError, TypeError):
            # Fallback: only keep numeric columns
            numeric_df = data_to_store.select_dtypes(include=[np.number])
            arr = numeric_df.values.astype(np.float64)
            columns_to_store = list(numeric_df.columns)

        # Create shared memory block (cleanup any leftover from previous run)
        block_name = self._generate_block_name(key)
        try:
            # Try to unlink any existing orphaned block with same name
            existing = shared_memory.SharedMemory(name=block_name)
            existing.close()
            existing.unlink()
            logger.info(f"Cleaned up orphaned shared memory block: {block_name}")
        except FileNotFoundError:
            pass  # Block doesn't exist, which is expected
        except Exception as e:
            # Permission error, resource busy, etc. - log and continue
            logger.warning(f"Could not clean up shared memory block {block_name}: {e}")

        try:
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=block_name)
        except FileExistsError:
            # Block still exists after cleanup attempt - force unlink and retry
            logger.warning(f"Shared memory block {block_name} still exists, forcing cleanup")
            try:
                existing = shared_memory.SharedMemory(name=block_name)
                existing.close()
                existing.unlink()
            except Exception:
                pass
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=block_name)

        # Copy data to shared memory
        shared_arr = np.ndarray(arr.shape, dtype=np.float64, buffer=shm.buf)
        shared_arr[:] = arr[:]

        # Store metadata
        self._shared_blocks[key] = shm
        self._metadata[key] = SharedArrayInfo(
            name=block_name,
            shape=arr.shape,
            dtype=str(arr.dtype),
            columns=columns_to_store,
            index_name=df.index.name or 'index'
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dict for passing to workers."""
        return {
            key: {
                'name': info.name,
                'shape': info.shape,
                'dtype': info.dtype,
                'columns': info.columns,
                'index_name': info.index_name
            }
            for key, info in self._metadata.items()
        }

    def _total_size_mb(self) -> float:
        """Calculate total shared memory size in MB."""
        total = sum(shm.size for shm in self._shared_blocks.values())
        return total / (1024 * 1024)

    def cleanup(self):
        """Clean up shared memory blocks."""
        cleaned = 0
        failed = 0
        for key, shm in list(self._shared_blocks.items()):
            try:
                shm.close()
                if self._is_owner:
                    shm.unlink()
                cleaned += 1
            except FileNotFoundError:
                # Already cleaned up (possibly by another process)
                cleaned += 1
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory {key}: {e}")
                failed += 1
        self._shared_blocks.clear()
        self._metadata.clear()
        if failed > 0:
            logger.warning(f"Shared memory cleanup: {cleaned} cleaned, {failed} failed")
        else:
            logger.info(f"Shared memory cleaned up: {cleaned} blocks")

    def __del__(self):
        """Cleanup on deletion."""
        if self._shared_blocks:
            self.cleanup()


class SharedDataReader:
    """
    Reader for shared memory data - used by worker processes.

    Attaches to existing shared memory blocks created by SharedDataManager.
    """

    # Maximum cache size for LRU eviction
    MAX_CACHE_SIZE = 20

    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize reader with metadata from main process.

        Args:
            metadata: Dict from SharedDataManager.get_metadata()
        """
        self._metadata = metadata
        self._shared_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}  # Acts as LRU cache
        self._attached = False

    def attach(self):
        """Attach to shared memory blocks."""
        if self._attached:
            return

        for key, info in self._metadata.items():
            try:
                shm = shared_memory.SharedMemory(name=info['name'])
                self._shared_blocks[key] = shm
            except FileNotFoundError:
                logger.warning(f"Shared memory block not found: {info['name']}")

        self._attached = True

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Reconstruct DataFrame from shared memory.

        Uses LRU caching with MAX_CACHE_SIZE limit to prevent unbounded memory growth.

        Args:
            key: Symbol or "__VIX__" for VIX data

        Returns:
            DataFrame or None if not found
        """
        if not self._attached:
            self.attach()

        # Check cache first (with LRU move-to-end)
        if key in self._data_cache:
            # Move to end for LRU ordering
            df = self._data_cache.pop(key)
            self._data_cache[key] = df
            return df

        if key not in self._shared_blocks:
            return None

        info = self._metadata[key]
        shm = self._shared_blocks[key]

        # Read array from shared memory (zero-copy view)
        arr = np.ndarray(tuple(info['shape']), dtype=info['dtype'], buffer=shm.buf)

        # Identify timestamp column
        timestamp_col = info['index_name'] if info['index_name'] in info['columns'] else 'timestamp'
        ts_col_idx = info['columns'].index(timestamp_col) if timestamp_col in info['columns'] else None

        if ts_col_idx is not None:
            # Extract timestamp column and convert to datetime index
            ts_values = arr[:, ts_col_idx]
            datetime_idx = pd.to_datetime(ts_values, unit='s')

            # Create DataFrame from non-timestamp columns only (zero-copy slices)
            data_cols = [c for c in info['columns'] if c != timestamp_col]
            col_indices = [info['columns'].index(c) for c in data_cols]

            # Build DataFrame column by column to preserve zero-copy views
            df_data = {}
            for i, col in zip(col_indices, data_cols):
                col_arr = arr[:, i]
                col_arr.flags.writeable = False  # Mark read-only
                df_data[col] = col_arr

            df = pd.DataFrame(df_data, index=pd.DatetimeIndex(datetime_idx, name=timestamp_col), copy=False)
        else:
            # No timestamp column - just create DataFrame directly
            arr.flags.writeable = False
            df = pd.DataFrame(arr, columns=info['columns'], copy=False)

        # LRU eviction before adding new entry
        if len(self._data_cache) >= self.MAX_CACHE_SIZE:
            oldest = next(iter(self._data_cache))
            del self._data_cache[oldest]
            logger.debug(f"LRU eviction: removed {oldest} from cache")

        # Cache for reuse
        self._data_cache[key] = df
        return df

    def get_all_data(self) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get all market data and VIX data.

        Returns:
            (data_dict, vix_data)
        """
        data = {}
        vix = None

        for key in self._metadata.keys():
            if key == "__VIX__":
                vix = self.get_dataframe(key)
            else:
                df = self.get_dataframe(key)
                if df is not None:
                    data[key] = df

        return data, vix

    def detach(self):
        """Detach from shared memory (don't unlink - main process owns it)."""
        for shm in self._shared_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
        self._shared_blocks.clear()
        self._data_cache.clear()
        self._attached = False

    def __del__(self):
        self.detach()


# Global reader instance for workers (initialized once per worker process)
_worker_reader: Optional[SharedDataReader] = None


def init_worker(metadata: Dict[str, Any]):
    """
    Initialize worker process with shared data access.

    Called once when worker starts via Pool initializer.
    """
    global _worker_reader
    _worker_reader = SharedDataReader(metadata)
    _worker_reader.attach()
    logger.debug("Worker initialized with shared data access")


def get_worker_data() -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Get data in worker process."""
    global _worker_reader
    if _worker_reader is None:
        raise RuntimeError("Worker not initialized - call init_worker first")
    return _worker_reader.get_all_data()


def cleanup_worker():
    """Cleanup worker's shared memory connections."""
    global _worker_reader
    if _worker_reader is not None:
        _worker_reader.detach()
        _worker_reader = None
