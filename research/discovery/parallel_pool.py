"""
Persistent Worker Pool
======================
A worker pool that stays alive across multiple evaluations,
avoiding the overhead of spawning new processes each time.

Workers are initialized once with shared memory access to market data,
then reused for all subsequent evaluations.

Usage:
    # Create pool once
    pool = PersistentWorkerPool(n_workers=4, shared_metadata=metadata)
    pool.start()

    # Use for many evaluations
    results = pool.evaluate_batch(tasks)
    results = pool.evaluate_batch(more_tasks)

    # Cleanup when done
    pool.shutdown()
"""

import logging
import multiprocessing as mp
from multiprocessing import Pool, Queue
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import time
import signal
import os

logger = logging.getLogger(__name__)


# Worker state - initialized once per worker process
_worker_initialized = False
_worker_data = None
_worker_vix = None


def _pool_warmup_task(worker_idx: int) -> bool:
    """
    Dummy task to trigger worker initialization.

    Must be module-level (not local) to be picklable for multiprocessing.
    """
    return True


def _pool_initializer(shared_metadata: Dict[str, Any]):
    """
    Initialize worker process - called once when worker starts.

    Sets up shared memory access to market data.
    """
    global _worker_initialized, _worker_data, _worker_vix

    # CRITICAL: Workers must ignore signals - only main process handles shutdown
    # This prevents zombie workers when SIGTERM is sent to the process group
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if _worker_initialized:
        return

    try:
        from research.discovery.shared_data import SharedDataReader

        reader = SharedDataReader(shared_metadata)
        reader.attach()
        _worker_data, _worker_vix = reader.get_all_data()
        _worker_initialized = True

        logger.debug(f"Worker {os.getpid()} initialized with {len(_worker_data)} symbols")

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


def _evaluate_genome_shared(args: Tuple) -> Dict[str, Any]:
    """
    Evaluate a genome using shared memory data.

    This is the worker function that runs in the pool.
    Data is accessed from shared memory, not passed as argument.
    """
    global _worker_data, _worker_vix

    genome_data, config_dict = args

    try:
        # Import here to avoid circular imports
        from research.discovery.config import EvolutionConfig
        from research.discovery.strategy_genome import GenomeFactory
        from research.discovery.strategy_compiler import StrategyCompiler
        from research.discovery.multi_objective import calculate_fitness_vector
        from research.discovery.novelty_search import extract_behavior_vector
        from research.backtester import Backtester, BacktestResult

        config = EvolutionConfig(**config_dict)
        factory = GenomeFactory(config)
        compiler = StrategyCompiler(config)
        backtester = Backtester(initial_capital=100000, cost_model="conservative")

        # Deserialize genome
        genome = factory.deserialize_genome(genome_data)

        # Compile strategy
        strategy = compiler.compile(genome)

        # Run backtest using shared data
        try:
            result = backtester.run(
                strategy=strategy,
                data=_worker_data,
                vix_data=_worker_vix
            )
        except Exception as e:
            result = BacktestResult(
                run_id=genome.genome_id,
                strategy=strategy.name if strategy else "unknown",
                start_date="",
                end_date=""
            )

        # Extract behavior vector
        behavior = extract_behavior_vector(result)

        return {
            "success": True,
            "genome_id": genome.genome_id,
            "behavior_array": behavior.to_array().tolist(),
            "result": result,
        }

    except Exception as e:
        genome_id = genome_data.get("genome_id", "unknown") if isinstance(genome_data, dict) else "unknown"
        logger.warning(f"Evaluation failed for {genome_id}: {e}")
        return {
            "success": False,
            "genome_id": genome_id,
            "error": str(e)
        }


class PersistentWorkerPool:
    """
    A worker pool that persists across evaluations.

    Features:
    - Workers initialized once with shared memory access
    - No data copying between main process and workers
    - Automatic worker restart on failure
    - Graceful shutdown
    """

    def __init__(
        self,
        n_workers: int = 4,
        shared_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pool configuration.

        Args:
            n_workers: Number of worker processes
            shared_metadata: Metadata for shared memory data access
        """
        self.n_workers = n_workers
        self.shared_metadata = shared_metadata or {}
        self._pool: Optional[Pool] = None
        self._started = False

    def start(self, stagger_delay: float = 3.0):
        """
        Start the worker pool with staggered initialization.

        Args:
            stagger_delay: Seconds to wait between starting each worker (default 3s).
                          This prevents memory stampede from all workers importing
                          heavy libraries (numpy, pandas, sklearn) simultaneously.
        """
        if self._started:
            logger.warning("Pool already started")
            return

        logger.info(f"Starting persistent worker pool with {self.n_workers} workers (staggered {stagger_delay}s)...")

        # CRITICAL: Close database connections before fork to prevent SQLite deadlock
        from data.storage.db_manager import get_db
        get_db().close_thread_connections()

        # Create pool with initializer
        self._pool = Pool(
            processes=self.n_workers,
            initializer=_pool_initializer,
            initargs=(self.shared_metadata,)
        )

        # Stagger worker initialization by sending warmup tasks one at a time
        # Each worker runs its initializer on first task, causing heavy imports
        logger.info("Warming up workers with staggered initialization...")
        for i in range(self.n_workers):
            # Send one task to one worker and wait for it to complete
            result = self._pool.apply(_pool_warmup_task, args=(i,))
            logger.debug(f"Worker {i+1}/{self.n_workers} initialized")
            if i < self.n_workers - 1:  # Don't sleep after the last worker
                time.sleep(stagger_delay)

        self._started = True
        logger.info("Worker pool started and initialized (all workers warmed up)")

    def evaluate_batch(
        self,
        genome_data_list: List[Dict],
        config_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of genomes in parallel.

        Args:
            genome_data_list: List of serialized genome dicts
            config_dict: Evolution config as dict

        Returns:
            List of result dicts
        """
        if not self._started:
            raise RuntimeError("Pool not started - call start() first")

        # Prepare args - only genome data and config, not market data
        args_list = [(g, config_dict) for g in genome_data_list]

        # Run in parallel
        try:
            results = self._pool.map(_evaluate_genome_shared, args_list)
            return results
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            # Return failure results
            return [
                {"success": False, "genome_id": g.get("genome_id", "?"), "error": str(e)}
                for g in genome_data_list
            ]

    def evaluate_batch_async(
        self,
        genome_data_list: List[Dict],
        config_dict: Dict[str, Any],
        callback: Optional[Callable] = None
    ):
        """
        Evaluate batch asynchronously with optional callback.

        Args:
            genome_data_list: List of serialized genome dicts
            config_dict: Evolution config as dict
            callback: Called with each result as it completes

        Returns:
            AsyncResult object
        """
        if not self._started:
            raise RuntimeError("Pool not started - call start() first")

        args_list = [(g, config_dict) for g in genome_data_list]

        return self._pool.map_async(
            _evaluate_genome_shared,
            args_list,
            callback=callback
        )

    def shutdown(self, wait: bool = True):
        """
        Shutdown the worker pool.

        Args:
            wait: If True, wait for workers to finish current tasks
        """
        if not self._started:
            return

        logger.info("Shutting down worker pool...")

        if wait:
            self._pool.close()
            self._pool.join()
        else:
            self._pool.terminate()

        self._pool = None
        self._started = False
        logger.info("Worker pool shut down")

    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._started and self._pool is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


class AdaptiveWorkerPool(PersistentWorkerPool):
    """
    Worker pool that adapts based on system conditions.

    Reduces workers if memory pressure is detected,
    increases if resources are available.
    """

    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 4,
        shared_metadata: Optional[Dict[str, Any]] = None
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers

        # Start with max workers
        super().__init__(n_workers=max_workers, shared_metadata=shared_metadata)

    def adjust_workers(self):
        """Adjust worker count based on system resources."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)

            if available_mb < 300 and self.n_workers > self.min_workers:
                # Memory pressure - reduce workers
                logger.warning(f"Memory pressure ({available_mb:.0f}MB), reducing workers")
                self._reduce_workers()

            elif available_mb > 800 and self.n_workers < self.max_workers:
                # Plenty of memory - increase workers
                logger.info(f"Memory available ({available_mb:.0f}MB), increasing workers")
                self._increase_workers()

        except ImportError:
            pass  # psutil not available

    def _reduce_workers(self):
        """Reduce worker count by 1."""
        if self.n_workers <= self.min_workers:
            return

        # Need to restart pool with fewer workers
        self.shutdown(wait=True)
        self.n_workers -= 1
        self.start()

    def _increase_workers(self):
        """Increase worker count by 1."""
        if self.n_workers >= self.max_workers:
            return

        # Need to restart pool with more workers
        self.shutdown(wait=True)
        self.n_workers += 1
        self.start()
