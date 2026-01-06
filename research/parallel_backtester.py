"""
Parallel Backtesting Engine
===========================
Run multiple strategy backtests concurrently for faster analysis.

Features:
- Parallel execution of multiple strategies
- Parallel walk-forward validation
- Parallel parameter sweeps
- Progress tracking and result aggregation
- Memory-efficient data sharing

Usage:
    from research.parallel_backtester import ParallelBacktester

    # Backtest multiple strategies in parallel
    backtester = ParallelBacktester(n_workers=4)
    results = backtester.run_strategies(
        strategies=['mean_reversion', 'pairs_trading', 'vol_managed_momentum'],
        data=cached_data,
        vix_data=vix_df
    )

    # Parallel parameter sweep for single strategy
    sweep_results = backtester.parameter_sweep(
        strategy_class=MeanReversionStrategy,
        param_grid={'lookback': [10, 15, 20], 'entry_std': [1.5, 2.0, 2.5]},
        data=cached_data
    )
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Type, Tuple
import itertools

import pandas as pd
import numpy as np

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.backtester import Backtester, BacktestResult
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel backtesting."""
    n_workers: int = None  # None = auto-detect (cpu_count // 2)
    chunk_size: int = 1  # Tasks per worker batch
    timeout_per_task: int = 300  # 5 minutes per backtest
    show_progress: bool = True
    save_intermediate: bool = False
    results_dir: Path = None

    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = max(1, (os.cpu_count() or 4) // 2)
        if self.results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results" / "parallel"
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ParallelResult:
    """Aggregated results from parallel backtesting."""
    strategy_results: Dict[str, BacktestResult] = field(default_factory=dict)
    execution_times: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    total_time: float = 0.0
    n_workers: int = 0

    def summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        rows = []
        for name, result in self.strategy_results.items():
            rows.append({
                'strategy': name,
                'sharpe': result.sharpe_ratio,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'execution_time': self.execution_times.get(name, 0),
            })
        return pd.DataFrame(rows).sort_values('sharpe', ascending=False)

    def best_strategy(self) -> Tuple[str, BacktestResult]:
        """Get the best performing strategy by Sharpe ratio."""
        if not self.strategy_results:
            return None, None
        best_name = max(self.strategy_results, key=lambda x: self.strategy_results[x].sharpe_ratio)
        return best_name, self.strategy_results[best_name]


def _run_single_backtest(args: Tuple) -> Tuple[str, Optional[BacktestResult], float, Optional[str]]:
    """
    Worker function for running a single backtest.
    Must be at module level for pickling.

    Args:
        args: Tuple of (strategy_name, strategy_instance, data, vix_data, initial_capital)

    Returns:
        Tuple of (strategy_name, result, execution_time, error_message)
    """
    strategy_name, strategy, data, vix_data, initial_capital = args
    start_time = time.time()

    try:
        backtester = Backtester(initial_capital=initial_capital)
        result = backtester.run(strategy, data, vix_data=vix_data)
        execution_time = time.time() - start_time
        return (strategy_name, result, execution_time, None)
    except Exception as e:
        execution_time = time.time() - start_time
        return (strategy_name, None, execution_time, str(e))


def _run_parameter_combo(args: Tuple) -> Tuple[Dict, Optional[BacktestResult], float, Optional[str]]:
    """
    Worker function for running a single parameter combination.

    Args:
        args: Tuple of (params_dict, strategy_class, data, vix_data, initial_capital)

    Returns:
        Tuple of (params_dict, result, execution_time, error_message)
    """
    params, strategy_class, data, vix_data, initial_capital = args
    start_time = time.time()

    try:
        # Create strategy with these parameters
        strategy = strategy_class()
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)

        backtester = Backtester(initial_capital=initial_capital)
        result = backtester.run(strategy, data, vix_data=vix_data)
        execution_time = time.time() - start_time
        return (params, result, execution_time, None)
    except Exception as e:
        execution_time = time.time() - start_time
        return (params, None, execution_time, str(e))


class ParallelBacktester:
    """
    Parallel backtesting engine for running multiple backtests concurrently.

    Examples:
        # Run multiple strategies in parallel
        pb = ParallelBacktester(n_workers=4)
        results = pb.run_strategies(
            strategies={
                'mean_rev': MeanReversionStrategy(),
                'momentum': VolManagedMomentumV2(),
            },
            data=data_dict,
            vix_data=vix_df
        )
        print(results.summary_df())

        # Parameter sweep
        sweep = pb.parameter_sweep(
            strategy_class=MeanReversionStrategy,
            param_grid={
                'lookback_period': [10, 15, 20],
                'entry_std': [1.5, 2.0, 2.5],
            },
            data=data_dict
        )
    """

    def __init__(self, n_workers: int = None, config: ParallelConfig = None):
        """
        Initialize parallel backtester.

        Args:
            n_workers: Number of worker processes (default: auto-detect)
            config: Full configuration object (overrides n_workers)
        """
        if config:
            self.config = config
        else:
            self.config = ParallelConfig(n_workers=n_workers)

        logger.info(f"ParallelBacktester initialized with {self.config.n_workers} workers")

    def run_strategies(
        self,
        strategies: Dict[str, BaseStrategy],
        data: Dict[str, pd.DataFrame],
        vix_data: pd.DataFrame = None,
        initial_capital: float = 100000
    ) -> ParallelResult:
        """
        Run multiple strategies in parallel.

        Args:
            strategies: Dict of {name: strategy_instance}
            data: Dict of {symbol: DataFrame} with OHLCV data
            vix_data: Optional VIX DataFrame for regime filtering
            initial_capital: Starting capital for each backtest

        Returns:
            ParallelResult with all strategy results
        """
        if not strategies:
            logger.warning("No strategies provided")
            return ParallelResult()

        logger.info(f"Running {len(strategies)} strategies in parallel")
        start_time = time.time()

        result = ParallelResult(n_workers=self.config.n_workers)

        # Prepare tasks
        tasks = [
            (name, strategy, data, vix_data, initial_capital)
            for name, strategy in strategies.items()
        ]

        # Execute in parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(_run_single_backtest, task): task[0]
                for task in tasks
            }

            for future in as_completed(futures, timeout=self.config.timeout_per_task * len(tasks)):
                strategy_name = futures[future]
                try:
                    name, bt_result, exec_time, error = future.result()
                    if bt_result:
                        result.strategy_results[name] = bt_result
                        result.execution_times[name] = exec_time
                        logger.info(f"  {name}: Sharpe={bt_result.sharpe_ratio:.2f} ({exec_time:.1f}s)")
                    if error:
                        result.errors[name] = error
                        logger.error(f"  {name}: Failed - {error}")
                except Exception as e:
                    result.errors[strategy_name] = str(e)
                    logger.error(f"  {strategy_name}: Exception - {e}")

                completed += 1
                if self.config.show_progress:
                    print(f"Progress: {completed}/{len(tasks)} strategies", end='\r')

        result.total_time = time.time() - start_time

        if self.config.show_progress:
            print()  # Clear progress line

        logger.info(f"Completed {len(result.strategy_results)}/{len(strategies)} strategies in {result.total_time:.1f}s")

        return result

    def parameter_sweep(
        self,
        strategy_class: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        data: Dict[str, pd.DataFrame],
        vix_data: pd.DataFrame = None,
        initial_capital: float = 100000,
        metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Run parallel parameter sweep for a strategy.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid: Dict of {param_name: [values_to_test]}
            data: Dict of {symbol: DataFrame} with OHLCV data
            vix_data: Optional VIX DataFrame
            initial_capital: Starting capital
            metric: Metric to optimize ('sharpe_ratio', 'annual_return', etc.)

        Returns:
            DataFrame with all parameter combinations and their results
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Running parameter sweep: {len(combinations)} combinations")
        start_time = time.time()

        # Prepare tasks
        tasks = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            tasks.append((params, strategy_class, data, vix_data, initial_capital))

        # Execute in parallel
        results = []
        completed = 0

        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(_run_parameter_combo, task): task[0]
                for task in tasks
            }

            for future in as_completed(futures, timeout=self.config.timeout_per_task * len(tasks)):
                params = futures[future]
                try:
                    params_result, bt_result, exec_time, error = future.result()
                    row = params_result.copy()
                    if bt_result:
                        row['sharpe_ratio'] = bt_result.sharpe_ratio
                        row['annual_return'] = bt_result.annual_return
                        row['max_drawdown'] = bt_result.max_drawdown_pct
                        row['total_trades'] = bt_result.total_trades
                        row['win_rate'] = bt_result.win_rate
                        row['profit_factor'] = bt_result.profit_factor
                        row['execution_time'] = exec_time
                        row['error'] = None
                    else:
                        row['sharpe_ratio'] = float('-inf')
                        row['error'] = error
                    results.append(row)
                except Exception as e:
                    row = params.copy()
                    row['sharpe_ratio'] = float('-inf')
                    row['error'] = str(e)
                    results.append(row)

                completed += 1
                if self.config.show_progress:
                    print(f"Progress: {completed}/{len(tasks)} combinations", end='\r')

        if self.config.show_progress:
            print()  # Clear progress line

        total_time = time.time() - start_time
        logger.info(f"Parameter sweep completed in {total_time:.1f}s")

        # Create results DataFrame
        df = pd.DataFrame(results)

        # Sort by target metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df

    def walk_forward_parallel(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        n_splits: int = 5,
        train_ratio: float = 0.7,
        vix_data: pd.DataFrame = None,
        initial_capital: float = 100000
    ) -> pd.DataFrame:
        """
        Run walk-forward validation with parallel execution of each fold.

        Args:
            strategy: Strategy instance to test
            data: Dict of {symbol: DataFrame} with OHLCV data
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of data for training in each split
            vix_data: Optional VIX DataFrame
            initial_capital: Starting capital

        Returns:
            DataFrame with results for each fold
        """
        # Determine date range from data
        all_dates = set()
        for symbol, df in data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())
            elif 'timestamp' in df.columns:
                all_dates.update(pd.to_datetime(df['timestamp']).tolist())

        all_dates = sorted(all_dates)
        n_dates = len(all_dates)

        if n_dates < 100:
            logger.warning("Insufficient data for walk-forward validation")
            return pd.DataFrame()

        # Calculate fold boundaries
        fold_size = n_dates // n_splits
        folds = []

        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = min((i + 2) * fold_size, n_dates)  # Include overlap

            train_end_idx = start_idx + int((end_idx - start_idx) * train_ratio)

            fold_data = {}
            for symbol, df in data.items():
                if isinstance(df.index, pd.DatetimeIndex):
                    fold_df = df[(df.index >= all_dates[start_idx]) & (df.index <= all_dates[end_idx])]
                else:
                    # Assume timestamp column
                    mask = (pd.to_datetime(df['timestamp']) >= all_dates[start_idx]) & \
                           (pd.to_datetime(df['timestamp']) <= all_dates[end_idx])
                    fold_df = df[mask]

                if len(fold_df) > 0:
                    fold_data[symbol] = fold_df.copy()

            # Filter VIX data for this fold
            fold_vix = None
            if vix_data is not None:
                if isinstance(vix_data.index, pd.DatetimeIndex):
                    fold_vix = vix_data[(vix_data.index >= all_dates[start_idx]) &
                                        (vix_data.index <= all_dates[end_idx])]
                elif 'timestamp' in vix_data.columns:
                    mask = (pd.to_datetime(vix_data['timestamp']) >= all_dates[start_idx]) & \
                           (pd.to_datetime(vix_data['timestamp']) <= all_dates[end_idx])
                    fold_vix = vix_data[mask]

            folds.append({
                'fold': i + 1,
                'start_date': all_dates[start_idx],
                'end_date': all_dates[end_idx - 1],
                'data': fold_data,
                'vix_data': fold_vix,
            })

        logger.info(f"Running {n_splits}-fold walk-forward validation in parallel")

        # Create fresh strategy instances for each fold
        strategy_class = type(strategy)
        tasks = []
        for fold in folds:
            fresh_strategy = strategy_class()
            tasks.append((
                f"fold_{fold['fold']}",
                fresh_strategy,
                fold['data'],
                fold['vix_data'],
                initial_capital
            ))

        # Run in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(_run_single_backtest, task): (task[0], folds[i])
                for i, task in enumerate(tasks)
            }

            for future in as_completed(futures):
                fold_name, fold_info = futures[future]
                try:
                    name, bt_result, exec_time, error = future.result()
                    if bt_result:
                        results.append({
                            'fold': fold_info['fold'],
                            'start_date': fold_info['start_date'],
                            'end_date': fold_info['end_date'],
                            'sharpe_ratio': bt_result.sharpe_ratio,
                            'annual_return': bt_result.annual_return,
                            'max_drawdown': bt_result.max_drawdown_pct,
                            'total_trades': bt_result.total_trades,
                            'win_rate': bt_result.win_rate,
                            'execution_time': exec_time,
                            'error': None
                        })
                    else:
                        results.append({
                            'fold': fold_info['fold'],
                            'error': error
                        })
                except Exception as e:
                    results.append({
                        'fold': fold_info['fold'],
                        'error': str(e)
                    })

        df = pd.DataFrame(results).sort_values('fold')

        # Add summary statistics
        if 'sharpe_ratio' in df.columns:
            valid = df[df['error'].isna()]
            logger.info(f"Walk-forward results: Mean Sharpe={valid['sharpe_ratio'].mean():.2f}, "
                       f"Std={valid['sharpe_ratio'].std():.2f}")

        return df


def run_all_strategies_parallel(
    data: Dict[str, pd.DataFrame] = None,
    vix_data: pd.DataFrame = None,
    n_workers: int = None
) -> ParallelResult:
    """
    Convenience function to run all registered strategies in parallel.

    Args:
        data: Optional data dict (loads from cache if None)
        vix_data: Optional VIX data
        n_workers: Number of workers (auto-detect if None)

    Returns:
        ParallelResult with all strategy results
    """
    from data.cached_data_manager import CachedDataManager
    from strategies import (
        MeanReversionStrategy,
        PairsTradingStrategy,
        VolManagedMomentumV2,
        GapFillStrategy,
        SectorRotationStrategy,
        QualitySmallCapValueStrategy,
        FactorMomentumStrategy,
        VIXRegimeRotationStrategy,
        RelativeVolumeBreakout,
    )
    from config import DIRS

    # Load data if not provided
    if data is None:
        logger.info("Loading data from cache...")
        data_mgr = CachedDataManager()
        if not data_mgr.cache:
            data_mgr.load_all()

        # Get top 100 by liquidity
        metadata = data_mgr.get_all_metadata()
        sorted_symbols = sorted(
            metadata.items(),
            key=lambda x: x[1].get('dollar_volume', 0),
            reverse=True
        )[:100]

        data = {}
        for symbol, _ in sorted_symbols:
            df = data_mgr.get_bars(symbol)
            if df is not None and len(df) >= 252:
                if 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                data[symbol] = df

        logger.info(f"Loaded {len(data)} symbols")

    # Load VIX if not provided
    if vix_data is None:
        vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
        if vix_path.exists():
            vix_data = pd.read_parquet(vix_path)
            if 'timestamp' in vix_data.columns:
                vix_data = vix_data.set_index('timestamp')
            if vix_data.index.tz is not None:
                vix_data.index = vix_data.index.tz_localize(None)

    # Create strategy instances
    strategies = {
        'mean_reversion': MeanReversionStrategy(),
        'vol_managed_momentum': VolManagedMomentumV2(),
        'sector_rotation': SectorRotationStrategy(),
        'quality_smallcap_value': QualitySmallCapValueStrategy(),
        'factor_momentum': FactorMomentumStrategy(),
        'vix_regime_rotation': VIXRegimeRotationStrategy(),
        'relative_volume_breakout': RelativeVolumeBreakout(),
    }

    # Pairs trading needs special handling (uses its own data loading)
    # Gap fill needs intraday data

    pb = ParallelBacktester(n_workers=n_workers)
    return pb.run_strategies(strategies, data, vix_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Backtesting")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--strategies", nargs="*", help="Specific strategies to test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    print("=" * 60)
    print("PARALLEL BACKTESTING")
    print("=" * 60)

    result = run_all_strategies_parallel(n_workers=args.workers)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(result.summary_df().to_string(index=False))

    print(f"\nTotal execution time: {result.total_time:.1f}s")
    print(f"Workers used: {result.n_workers}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for name, error in result.errors.items():
            print(f"  {name}: {error}")

    best_name, best_result = result.best_strategy()
    if best_result:
        print(f"\nBest Strategy: {best_name}")
        print(f"  Sharpe: {best_result.sharpe_ratio:.2f}")
        print(f"  Annual Return: {best_result.annual_return:.1f}%")
        print(f"  Max Drawdown: {best_result.max_drawdown_pct:.1f}%")
