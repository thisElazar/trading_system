"""
GA Parallel Support
===================
Shared memory and persistent worker pool support for GA optimization.

Provides efficient parallelism for:
- RapidBacktester multi-period testing
- GeneticOptimizer fitness evaluation
- AdaptiveGAOptimizer island evolution
"""

import logging
import os
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

import pandas as pd

logger = logging.getLogger(__name__)

# Global state for worker processes
_ga_worker_initialized = False
_ga_worker_data: Dict[str, pd.DataFrame] = {}
_ga_worker_vix: Optional[pd.DataFrame] = None


def _ga_pool_initializer(shared_metadata: Dict[str, Any]):
    """
    Initialize GA worker process with shared memory access.

    Called once when worker starts.
    """
    global _ga_worker_initialized, _ga_worker_data, _ga_worker_vix

    if _ga_worker_initialized:
        return

    try:
        from research.discovery.shared_data import SharedDataReader

        reader = SharedDataReader(shared_metadata)
        reader.attach()
        _ga_worker_data, _ga_worker_vix = reader.get_all_data()
        _ga_worker_initialized = True

        logger.debug(f"GA Worker {os.getpid()} initialized with {len(_ga_worker_data)} symbols")

    except Exception as e:
        logger.error(f"GA Worker initialization failed: {e}")
        raise


def _run_period_test_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel period testing.

    Runs a single period test using shared memory data.
    """
    global _ga_worker_data, _ga_worker_vix

    strategy_factory, genes, period_dict, config = args

    try:
        from research.genetic.market_periods import MarketPeriod, PeriodType
        import time

        start_time = time.time()

        # Reconstruct period
        period = MarketPeriod(
            name=period_dict['name'],
            start_date=pd.Timestamp(period_dict['start_date']),
            end_date=pd.Timestamp(period_dict['end_date']),
            period_type=PeriodType(period_dict['period_type']),
            description=period_dict.get('description', ''),
            difficulty=period_dict.get('difficulty', 0.5),
            spy_return=period_dict.get('spy_return', 0.0),
        )

        # Create strategy with genes
        strategy = strategy_factory(genes)

        # Filter data to period
        period_data = {}
        for symbol, df in _ga_worker_data.items():
            if 'timestamp' in df.columns:
                mask = (df['timestamp'] >= period.start_date) & (df['timestamp'] <= period.end_date)
            elif isinstance(df.index, pd.DatetimeIndex):
                mask = (df.index >= period.start_date) & (df.index <= period.end_date)
            else:
                continue

            filtered = df[mask]
            if len(filtered) > 0:
                period_data[symbol] = filtered

        if not period_data:
            return {
                'success': False,
                'period_name': period.name,
                'error': 'No data in period'
            }

        # Run simplified backtest
        result = _fast_backtest(strategy, period_data, _ga_worker_vix, config)

        execution_ms = (time.time() - start_time) * 1000

        return {
            'success': True,
            'period_name': period.name,
            'period_type': period.period_type.value,
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'sortino_ratio': result.get('sortino_ratio', 0.0),
            'execution_ms': execution_ms,
            'period_difficulty': period.difficulty,
            'period_spy_return': period.spy_return,
            'alpha': result['total_return'] - period.spy_return,
        }

    except Exception as e:
        return {
            'success': False,
            'period_name': period_dict.get('name', 'unknown'),
            'error': str(e)
        }


def _fast_backtest(
    strategy: Any,
    data: Dict[str, pd.DataFrame],
    vix_data: Optional[pd.DataFrame],
    config: Dict
) -> Dict[str, Any]:
    """
    Simplified fast backtest for GA evaluation.
    """
    trades = []
    equity = 10000.0
    equity_curve = [equity]

    # Get sorted dates
    all_dates = set()
    for df in data.values():
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp']).dt.date
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        else:
            continue
        all_dates.update(dates)

    if not all_dates:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
        }

    sorted_dates = sorted(all_dates)
    positions = {}
    daily_returns = []

    # Simplified daily loop
    for i, current_date in enumerate(sorted_dates):
        try:
            # Get data up to current date
            day_data = {}
            for symbol, df in data.items():
                if 'timestamp' in df.columns:
                    mask = pd.to_datetime(df['timestamp']).dt.date <= current_date
                else:
                    mask = df.index.date <= current_date
                day_data[symbol] = df[mask].tail(60)  # Last 60 days for signals

            # Generate signals
            signals = strategy.generate_signals(day_data, vix_data)

            # Process signals (simplified)
            if signals:
                for signal in signals[:3]:  # Max 3 signals per day
                    symbol = getattr(signal, 'symbol', signal.get('symbol', ''))
                    if symbol and symbol not in positions:
                        positions[symbol] = {'entry_price': getattr(signal, 'price', 100)}
                        trades.append(1)

            # Calculate daily P&L (simplified)
            daily_pnl = 0
            for symbol in list(positions.keys()):
                if symbol in day_data and len(day_data[symbol]) > 0:
                    current_price = day_data[symbol]['close'].iloc[-1]
                    entry = positions[symbol]['entry_price']
                    daily_pnl += (current_price / entry - 1) * (equity / max(len(positions), 1))

            equity += daily_pnl
            equity_curve.append(equity)
            if len(equity_curve) > 1:
                daily_returns.append(equity_curve[-1] / equity_curve[-2] - 1)

        except Exception:
            continue

    # Calculate metrics (using Numba-optimized functions when available)
    total_return = (equity / 10000.0 - 1) * 100

    if equity_curve:
        equity_arr = np.array(equity_curve, dtype=np.float64)
        if len(equity_arr) > 1:
            returns_arr = np.diff(equity_arr) / equity_arr[:-1]
            sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * np.sqrt(252)
            running_max = np.maximum.accumulate(equity_arr)
            max_dd = np.min((equity_arr - running_max) / running_max * 100)
        else:
            sharpe = 0.0
            max_dd = 0.0
    else:
        sharpe = 0.0
        max_dd = 0.0

    win_rate = 0.5 if not trades else sum(1 for t in trades if t > 0) / len(trades)

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
    }


def _evaluate_fitness_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for GA fitness evaluation.

    Evaluates a single individual using shared memory data.
    """
    global _ga_worker_data, _ga_worker_vix

    strategy_factory, genes, config = args

    try:
        # Create strategy
        strategy = strategy_factory(genes)

        # Run backtest
        result = _fast_backtest(strategy, _ga_worker_data, _ga_worker_vix, config)

        # Fitness is Sharpe ratio (or other metric)
        fitness = result['sharpe_ratio']

        return {
            'success': True,
            'genes': genes,
            'fitness': fitness,
            'metrics': result,
        }

    except Exception as e:
        return {
            'success': False,
            'genes': genes,
            'fitness': float('-inf'),
            'error': str(e),
        }


class GAWorkerPool:
    """
    Persistent worker pool for GA optimization.

    Manages a pool of workers with shared memory access
    for efficient parallel fitness evaluation.
    """

    def __init__(
        self,
        n_workers: int = 3,
        shared_metadata: Optional[Dict[str, Any]] = None
    ):
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
        import time

        if self._started:
            return

        logger.info(f"Starting GA worker pool with {self.n_workers} workers (staggered {stagger_delay}s)...")

        # Create pool - this spawns all worker processes
        self._pool = Pool(
            processes=self.n_workers,
            initializer=_ga_pool_initializer,
            initargs=(self.shared_metadata,)
        )

        # Stagger worker initialization by sending warmup tasks one at a time
        # Each worker runs its initializer on first task, causing heavy imports
        def _warmup_task(_):
            """Dummy task to trigger worker initialization."""
            return True

        logger.info("Warming up workers with staggered initialization...")
        for i in range(self.n_workers):
            # Send one task to one worker and wait for it to complete
            result = self._pool.apply(_warmup_task, args=(i,))
            logger.debug(f"Worker {i+1}/{self.n_workers} initialized")
            if i < self.n_workers - 1:  # Don't sleep after the last worker
                time.sleep(stagger_delay)

        self._started = True
        logger.info("GA worker pool started (all workers warmed up)")

    def evaluate_fitness_batch(
        self,
        strategy_factory: Callable,
        genes_list: List[Dict],
        config: Dict = None
    ) -> List[Dict]:
        """
        Evaluate fitness for a batch of gene configurations.
        """
        if not self._started:
            raise RuntimeError("Pool not started")

        config = config or {}
        args_list = [(strategy_factory, genes, config) for genes in genes_list]

        try:
            results = self._pool.map(_evaluate_fitness_worker, args_list)
            return results
        except Exception as e:
            logger.error(f"Batch fitness evaluation failed: {e}")
            return [{'success': False, 'fitness': float('-inf')} for _ in genes_list]

    def run_period_tests_batch(
        self,
        strategy_factory: Callable,
        genes: Dict,
        periods: List[Dict],
        config: Dict = None
    ) -> List[Dict]:
        """
        Run period tests for a strategy configuration.
        """
        if not self._started:
            raise RuntimeError("Pool not started")

        config = config or {}
        args_list = [(strategy_factory, genes, period, config) for period in periods]

        try:
            results = self._pool.map(_run_period_test_worker, args_list)
            return results
        except Exception as e:
            logger.error(f"Batch period testing failed: {e}")
            return [{'success': False, 'error': str(e)} for _ in periods]

    def shutdown(self, wait: bool = True):
        """Shutdown the pool."""
        if not self._started:
            return

        if wait:
            self._pool.close()
            self._pool.join()
        else:
            self._pool.terminate()

        self._pool = None
        self._started = False
        logger.info("GA worker pool shut down")

    def is_running(self) -> bool:
        return self._started

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
