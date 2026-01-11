"""
Rapid Backtester
================
Ultra-fast backtester optimized for short period testing.

Designed for:
- 30-second test runs on specific market periods
- Parallel evaluation of many strategy configurations
- Multi-environment fitness assessment
- Rapid GA generation evaluation

Key optimizations:
- Pre-sliced data caching
- Simplified execution model for speed
- Vectorized operations where possible
- Minimal logging overhead
- Optional detailed metrics (off for speed)
- Shared memory + persistent worker pool for true multi-core parallelism

Usage:
    from research.genetic.rapid_backtester import RapidBacktester
    from research.genetic.market_periods import MarketPeriodLibrary

    rapid = RapidBacktester()
    library = MarketPeriodLibrary()

    # Test on COVID crash
    period = library.get_period("covid_crash")
    result = rapid.run_period_test(strategy, period, data)

    # Run multi-environment test
    results = rapid.run_multi_period_test(strategy, ['covid_crash', 'recovery', 'bull'])

    # For GA optimization with persistent pool:
    with RapidBacktester() as rapid:
        rapid.init_parallel_pool(data, vix_data)
        for generation in range(100):
            result = rapid.run_multi_period_test_parallel(strategy_factory, genes, periods)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from .market_periods import MarketPeriodLibrary, MarketPeriod, PeriodType
from .ga_parallel import GAWorkerPool
from research.discovery.shared_data import SharedDataManager

logger = logging.getLogger(__name__)


@dataclass
class RapidBacktestResult:
    """Lightweight backtest result for rapid testing."""
    period_name: str
    period_type: str

    # Core metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float

    # Optional extended metrics
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    profit_factor: float = 0.0

    # Execution time
    execution_ms: float = 0.0

    # Period characteristics
    period_difficulty: float = 0.5
    period_spy_return: float = 0.0

    # Relative performance
    alpha: float = 0.0  # Return vs SPY

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period_name': self.period_name,
            'period_type': self.period_type,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'avg_trade_pnl': self.avg_trade_pnl,
            'profit_factor': self.profit_factor,
            'execution_ms': self.execution_ms,
            'alpha': self.alpha,
        }

    @property
    def is_valid(self) -> bool:
        """Check if result meets minimum validity criteria."""
        return (
            self.total_trades >= 3 and
            self.max_drawdown > -100 and
            not np.isnan(self.sharpe_ratio)
        )


@dataclass
class MultiPeriodResult:
    """Aggregated results across multiple periods."""
    results: List[RapidBacktestResult]
    strategy_name: str = ""

    # Aggregated metrics
    avg_sharpe: float = 0.0
    avg_return: float = 0.0
    avg_drawdown: float = 0.0
    worst_drawdown: float = 0.0
    best_return: float = 0.0
    worst_return: float = 0.0
    total_trades: int = 0
    avg_win_rate: float = 0.0

    # Regime-specific performance
    crisis_sharpe: float = 0.0
    bull_sharpe: float = 0.0
    sideways_sharpe: float = 0.0

    # Consistency
    sharpe_std: float = 0.0  # Lower = more consistent
    return_std: float = 0.0

    # Alpha metrics
    avg_alpha: float = 0.0

    # Execution
    total_execution_ms: float = 0.0

    def __post_init__(self):
        if self.results:
            self._calculate_aggregates()

    def _calculate_aggregates(self):
        """Calculate aggregate metrics from individual results."""
        valid_results = [r for r in self.results if r.is_valid]

        if not valid_results:
            return

        # Basic averages
        self.avg_sharpe = np.mean([r.sharpe_ratio for r in valid_results])
        self.avg_return = np.mean([r.total_return for r in valid_results])
        self.avg_drawdown = np.mean([r.max_drawdown for r in valid_results])
        self.worst_drawdown = min([r.max_drawdown for r in valid_results])
        self.best_return = max([r.total_return for r in valid_results])
        self.worst_return = min([r.total_return for r in valid_results])
        self.total_trades = sum([r.total_trades for r in valid_results])
        self.avg_win_rate = np.mean([r.win_rate for r in valid_results if r.total_trades > 0])

        # Standard deviations (consistency)
        self.sharpe_std = np.std([r.sharpe_ratio for r in valid_results])
        self.return_std = np.std([r.total_return for r in valid_results])

        # Alpha
        self.avg_alpha = np.mean([r.alpha for r in valid_results])

        # Execution time
        self.total_execution_ms = sum([r.execution_ms for r in self.results])

        # Regime-specific
        crisis_results = [r for r in valid_results if r.period_type == 'crisis']
        bull_results = [r for r in valid_results if r.period_type in ['bull_run', 'recovery']]
        sideways_results = [r for r in valid_results if r.period_type == 'sideways']

        if crisis_results:
            self.crisis_sharpe = np.mean([r.sharpe_ratio for r in crisis_results])
        if bull_results:
            self.bull_sharpe = np.mean([r.sharpe_ratio for r in bull_results])
        if sideways_results:
            self.sideways_sharpe = np.mean([r.sharpe_ratio for r in sideways_results])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'num_periods': len(self.results),
            'avg_sharpe': self.avg_sharpe,
            'avg_return': self.avg_return,
            'avg_drawdown': self.avg_drawdown,
            'worst_drawdown': self.worst_drawdown,
            'best_return': self.best_return,
            'worst_return': self.worst_return,
            'total_trades': self.total_trades,
            'avg_win_rate': self.avg_win_rate,
            'sharpe_std': self.sharpe_std,
            'crisis_sharpe': self.crisis_sharpe,
            'bull_sharpe': self.bull_sharpe,
            'sideways_sharpe': self.sideways_sharpe,
            'avg_alpha': self.avg_alpha,
            'total_execution_ms': self.total_execution_ms,
            'results': [r.to_dict() for r in self.results],
        }

    @property
    def consistency_score(self) -> float:
        """Score for performance consistency across regimes (0-1, higher = better)."""
        if not self.results or self.avg_sharpe <= 0:
            return 0.0

        # Penalize high variance relative to mean
        cv = self.sharpe_std / max(self.avg_sharpe, 0.1)
        return max(0, 1 - cv)


class RapidBacktester:
    """
    Ultra-fast backtester for short-period testing.

    Optimized for:
    - Speed over detailed simulation
    - Parallel period testing
    - GA fitness evaluation
    - Shared memory for multi-core parallelism
    """

    # Simplified cost model for speed
    DEFAULT_COSTS = {
        'spread_bps': 8,
        'slippage_bps': 5,
    }

    # Cache size limit to prevent OOM during long research runs
    MAX_PERIOD_CACHE_ENTRIES = 10

    def __init__(
        self,
        period_library: MarketPeriodLibrary = None,
        costs: Dict[str, float] = None,
        parallel: bool = True,
        max_workers: int = 1
    ):
        """
        Initialize the rapid backtester.

        Args:
            period_library: MarketPeriodLibrary instance
            costs: Transaction cost assumptions
            parallel: Enable parallel execution
            max_workers: Number of parallel workers
        """
        self.library = period_library or MarketPeriodLibrary()
        self.costs = costs or self.DEFAULT_COSTS.copy()
        self.parallel = parallel
        self.max_workers = max_workers

        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._period_data_cache: Dict[str, pd.DataFrame] = {}
        self._vix_data: Optional[pd.DataFrame] = None

        # Persistent pool state
        self._shared_data_manager: Optional[SharedDataManager] = None
        self._worker_pool: Optional[GAWorkerPool] = None
        self._pool_initialized = False

        logger.info("RapidBacktester initialized")

    # =========================================================================
    # Shared Memory + Persistent Pool Support
    # =========================================================================

    def init_parallel_pool(
        self,
        data: Dict[str, pd.DataFrame],
        vix_data: pd.DataFrame = None,
        n_workers: int = None
    ):
        """
        Initialize shared memory and persistent worker pool.

        Call this once before running multiple tests for best performance.
        Workers stay alive and share memory access to data.

        Args:
            data: Market data to share
            vix_data: Optional VIX data
            n_workers: Number of workers (default: self.max_workers)
        """
        # Cache data locally too
        self.cache_data(data)
        self._vix_data = vix_data

        # Create shared memory manager
        self._shared_data_manager = SharedDataManager()
        self._shared_data_manager.load_data(data, vix_data)

        # Create persistent worker pool
        n = n_workers or self.max_workers
        self._worker_pool = GAWorkerPool(
            n_workers=n,
            shared_metadata=self._shared_data_manager.get_metadata()
        )
        self._worker_pool.start()
        self._pool_initialized = True

        logger.info(f"Parallel pool initialized with {n} workers")

    def cleanup_parallel_pool(self):
        """Clean up shared memory and worker pool."""
        if self._worker_pool is not None:
            self._worker_pool.shutdown()
            self._worker_pool = None

        if self._shared_data_manager is not None:
            self._shared_data_manager.cleanup()
            self._shared_data_manager = None

        self._pool_initialized = False
        logger.info("Parallel pool cleaned up")

    def run_multi_period_test_parallel(
        self,
        strategy_factory: Callable,
        genes: Dict,
        period_names: List[str] = None,
        period_types: List[PeriodType] = None,
    ) -> MultiPeriodResult:
        """
        Run backtest across multiple periods using persistent worker pool.

        This is much faster than run_multi_period_test for repeated calls
        because workers stay alive and share memory access to data.

        Args:
            strategy_factory: Function that creates strategy from genes dict
            genes: Strategy parameter genes
            period_names: Specific period names to test
            period_types: Or, types of periods to include

        Returns:
            MultiPeriodResult with aggregated metrics
        """
        if not self._pool_initialized:
            raise RuntimeError("Pool not initialized - call init_parallel_pool first")

        # Determine periods to test
        if period_names:
            periods = [self.library.get_period(name) for name in period_names if self.library.get_period(name)]
        elif period_types:
            periods = []
            for pt in period_types:
                periods.extend(self.library.get_periods_by_type(pt))
        else:
            periods = self.library.get_diverse_test_set()

        if not periods:
            return MultiPeriodResult(results=[], strategy_name="unknown")

        # Convert periods to dicts for serialization
        period_dicts = [
            {
                'name': p.name,
                'start_date': p.start_date.isoformat(),
                'end_date': p.end_date.isoformat(),
                'period_type': p.period_type.value,
                'description': p.description,
                'difficulty': p.difficulty,
                'spy_return': p.spy_return,
            }
            for p in periods
        ]

        # Run tests in parallel using persistent pool
        raw_results = self._worker_pool.run_period_tests_batch(
            strategy_factory=strategy_factory,
            genes=genes,
            periods=period_dicts,
            config={'costs': self.costs}
        )

        # Convert to RapidBacktestResult objects
        results = []
        for raw in raw_results:
            if raw.get('success', False):
                results.append(RapidBacktestResult(
                    period_name=raw['period_name'],
                    period_type=raw['period_type'],
                    total_return=raw['total_return'],
                    sharpe_ratio=raw['sharpe_ratio'],
                    max_drawdown=raw['max_drawdown'],
                    total_trades=raw['total_trades'],
                    win_rate=raw['win_rate'],
                    sortino_ratio=raw.get('sortino_ratio', 0.0),
                    execution_ms=raw.get('execution_ms', 0.0),
                    period_difficulty=raw.get('period_difficulty', 0.5),
                    period_spy_return=raw.get('period_spy_return', 0.0),
                    alpha=raw.get('alpha', 0.0),
                ))
            else:
                logger.warning(f"Period test failed: {raw.get('period_name', '?')}: {raw.get('error', 'unknown')}")

        return MultiPeriodResult(
            results=results,
            strategy_name="optimized"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - cleanup pool."""
        self.cleanup_parallel_pool()

    # =========================================================================
    # Original Methods (unchanged)
    # =========================================================================

    def cache_data(self, data: Dict[str, pd.DataFrame], force: bool = False):
        """
        Cache market data for rapid access.

        Args:
            data: Dict mapping symbol to OHLCV DataFrames
            force: Force re-cache even if data appears unchanged
        """
        # Skip if data already cached with same symbols (avoids redundant copies)
        if not force and self._data_cache and len(self._data_cache) == len(data):
            # Quick check: same symbols means likely same data
            if set(self._data_cache.keys()) == set(data.keys()):
                return  # Already cached

        self._data_cache = data.copy()
        logger.info(f"Cached data for {len(data)} symbols")

    def clear_cache(self):
        """
        Clear all cached data to free memory.

        Call this between strategy runs during nightly research
        to prevent unbounded memory growth.
        """
        cache_size = len(self._period_data_cache)
        self._period_data_cache.clear()
        self._data_cache.clear()
        self._vix_data = None
        if cache_size > 0:
            logger.info(f"Cleared {cache_size} cached period data entries")

    def _get_period_data(
        self,
        period: MarketPeriod,
        symbols: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get data filtered to a specific period."""
        cache_key = f"{period.name}_{len(symbols or [])}"

        if cache_key in self._period_data_cache:
            return self._period_data_cache[cache_key]

        # Evict oldest cache entries if at limit (prevents OOM during long runs)
        while len(self._period_data_cache) >= self.MAX_PERIOD_CACHE_ENTRIES:
            oldest_key = next(iter(self._period_data_cache))
            del self._period_data_cache[oldest_key]

        symbols = symbols or list(self._data_cache.keys())
        period_data = {}

        for symbol in symbols:
            if symbol not in self._data_cache:
                continue

            df = self._data_cache[symbol]
            filtered = self.library.filter_data_to_period(df, period)

            if len(filtered) >= 5:  # Minimum data requirement
                period_data[symbol] = filtered

        self._period_data_cache[cache_key] = period_data
        return period_data

    def run_period_test(
        self,
        strategy: Any,
        period: MarketPeriod,
        data: Dict[str, pd.DataFrame] = None,
        vix_data: pd.DataFrame = None,
        detailed: bool = False
    ) -> RapidBacktestResult:
        """
        Run a rapid backtest on a specific period.

        Args:
            strategy: Strategy instance with generate_signals method
            period: MarketPeriod to test on
            data: Optional data override (uses cached if None)
            vix_data: Optional VIX data
            detailed: If True, calculate extended metrics

        Returns:
            RapidBacktestResult
        """
        start_time = time.time()

        # Get period data
        if data is None:
            data = self._get_period_data(period)

        if not data:
            return RapidBacktestResult(
                period_name=period.name,
                period_type=period.period_type.value,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=-100.0,
                total_trades=0,
                win_rate=0.0,
                period_difficulty=period.difficulty,
                period_spy_return=period.spy_return,
            )

        # Filter VIX data to period if provided
        if vix_data is not None:
            vix_data = self.library.filter_data_to_period(vix_data, period)

        # Run simplified backtest
        result = self._run_fast_backtest(
            strategy=strategy,
            data=data,
            vix_data=vix_data,
            detailed=detailed
        )

        execution_ms = (time.time() - start_time) * 1000

        return RapidBacktestResult(
            period_name=period.name,
            period_type=period.period_type.value,
            total_return=result['total_return'],
            sharpe_ratio=result['sharpe_ratio'],
            max_drawdown=result['max_drawdown'],
            total_trades=result['total_trades'],
            win_rate=result['win_rate'],
            sortino_ratio=result.get('sortino_ratio', 0.0),
            calmar_ratio=result.get('calmar_ratio', 0.0),
            avg_trade_pnl=result.get('avg_trade_pnl', 0.0),
            profit_factor=result.get('profit_factor', 0.0),
            execution_ms=execution_ms,
            period_difficulty=period.difficulty,
            period_spy_return=period.spy_return,
            alpha=result['total_return'] - period.spy_return,
        )

    def _run_fast_backtest(
        self,
        strategy: Any,
        data: Dict[str, pd.DataFrame],
        vix_data: pd.DataFrame = None,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Run fast simplified backtest.

        Uses simplified execution model for speed.
        """
        trades = []
        equity_curve = [10000.0]  # Starting capital
        current_equity = 10000.0

        # Get all dates across all symbols
        all_dates = set()
        for df in data.values():
            if 'timestamp' in df.columns:
                dates = pd.to_datetime(df['timestamp']).dt.date
            elif isinstance(df.index, pd.DatetimeIndex):
                dates = df.index.date
            else:
                continue
            all_dates.update(dates)

        sorted_dates = sorted(all_dates)

        # Track open positions
        positions: Dict[str, Dict] = {}

        # Process each day
        for current_date in sorted_dates:
            # Get signals from strategy
            try:
                day_data = {}
                for symbol, df in data.items():
                    if 'timestamp' in df.columns:
                        mask = pd.to_datetime(df['timestamp']).dt.date <= current_date
                    else:
                        mask = df.index.date <= current_date
                    day_data[symbol] = df[mask]

                signals = strategy.generate_signals(day_data, vix_data)

            except Exception as e:
                logger.debug(f"Signal generation failed for {current_date}: {e}")
                continue

            if not signals:
                continue

            # Process signals
            for signal in signals:
                symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol')
                signal_type = signal.signal_type if hasattr(signal, 'signal_type') else signal.get('signal_type')
                price = signal.price if hasattr(signal, 'price') else signal.get('price', 0)

                if not symbol or not price:
                    continue

                # Apply costs
                cost_bps = self.costs['spread_bps'] + self.costs['slippage_bps']
                effective_price = price * (1 + cost_bps / 10000) if 'BUY' in str(signal_type) else price * (1 - cost_bps / 10000)

                if 'BUY' in str(signal_type) and symbol not in positions:
                    # Open position
                    position_size = current_equity * 0.02  # 2% position size
                    shares = position_size / effective_price

                    positions[symbol] = {
                        'entry_price': effective_price,
                        'shares': shares,
                        'entry_date': current_date,
                    }

                elif ('SELL' in str(signal_type) or 'CLOSE' in str(signal_type)) and symbol in positions:
                    # Close position
                    pos = positions.pop(symbol)
                    pnl = (effective_price - pos['entry_price']) * pos['shares']

                    trades.append({
                        'symbol': symbol,
                        'entry_price': pos['entry_price'],
                        'exit_price': effective_price,
                        'pnl': pnl,
                        'pnl_pct': (effective_price / pos['entry_price'] - 1) * 100,
                        'holding_days': (current_date - pos['entry_date']).days if isinstance(pos['entry_date'], date) else 0,
                    })

                    current_equity += pnl
                    equity_curve.append(current_equity)

        # Close any remaining positions at last price
        for symbol, pos in positions.items():
            if symbol in data:
                df = data[symbol]
                last_price = float(df['close'].iloc[-1]) * (1 - self.costs['spread_bps'] / 10000)
                pnl = (last_price - pos['entry_price']) * pos['shares']

                trades.append({
                    'symbol': symbol,
                    'entry_price': pos['entry_price'],
                    'exit_price': last_price,
                    'pnl': pnl,
                    'pnl_pct': (last_price / pos['entry_price'] - 1) * 100,
                })

                current_equity += pnl
                equity_curve.append(current_equity)

        # Calculate metrics
        total_return = (current_equity / 10000 - 1) * 100
        total_trades = len(trades)

        if trades:
            winning = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning) / len(trades) * 100
            avg_trade_pnl = np.mean([t['pnl'] for t in trades])

            gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
            gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            profit_factor = 0.0

        # Drawdown from equity curve
        equity_series = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = np.min(drawdown)

        # Sharpe ratio (simplified)
        if len(equity_curve) > 2:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                # Annualize assuming ~252 trading days
                periods_per_year = 252 / max(1, len(sorted_dates))
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        result = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'profit_factor': profit_factor,
        }

        # Extended metrics if requested
        if detailed and len(equity_curve) > 2:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            negative_returns = returns[returns < 0]

            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                periods_per_year = 252 / max(1, len(sorted_dates))
                sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(periods_per_year)
            else:
                sortino = sharpe_ratio

            calmar = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

            result['sortino_ratio'] = sortino
            result['calmar_ratio'] = calmar

        return result

    def run_multi_period_test(
        self,
        strategy: Any,
        period_names: List[str] = None,
        period_types: List[PeriodType] = None,
        data: Dict[str, pd.DataFrame] = None,
        vix_data: pd.DataFrame = None
    ) -> MultiPeriodResult:
        """
        Run backtest across multiple periods.

        Args:
            strategy: Strategy to test
            period_names: Specific period names to test
            period_types: Or, types of periods to include
            data: Market data
            vix_data: VIX data

        Returns:
            MultiPeriodResult with aggregated metrics
        """
        # Determine periods to test
        if period_names:
            periods = [self.library.get_period(name) for name in period_names if self.library.get_period(name)]
        elif period_types:
            periods = []
            for pt in period_types:
                periods.extend(self.library.get_periods_by_type(pt))
        else:
            # Default: diverse test set
            periods = self.library.get_diverse_test_set()

        if not periods:
            return MultiPeriodResult(results=[], strategy_name=getattr(strategy, 'name', 'unknown'))

        # Cache data if provided
        if data:
            self.cache_data(data)

        results = []

        if self.parallel and len(periods) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.run_period_test,
                        strategy,
                        period,
                        None,  # Use cached data
                        vix_data
                    ): period
                    for period in periods
                }

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        period = futures[future]
                        logger.warning(f"Period test failed for {period.name}: {e}")
        else:
            # Sequential execution
            for period in periods:
                try:
                    result = self.run_period_test(strategy, period, None, vix_data)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Period test failed for {period.name}: {e}")

        return MultiPeriodResult(
            results=results,
            strategy_name=getattr(strategy, 'name', 'unknown')
        )

    def run_rapid_test_suite(
        self,
        strategy: Any,
        data: Dict[str, pd.DataFrame] = None,
        vix_data: pd.DataFrame = None
    ) -> Dict[str, MultiPeriodResult]:
        """
        Run the standard rapid test suite.

        Tests across:
        - Crisis periods
        - Recovery periods
        - Bull markets
        - Bear markets
        - Sideways markets

        Returns:
            Dict mapping regime to MultiPeriodResult
        """
        if data:
            self.cache_data(data)

        suite = self.library.get_rapid_test_suite()
        results = {}

        for regime, periods in suite.items():
            period_results = []

            for period in periods:
                try:
                    result = self.run_period_test(strategy, period, None, vix_data)
                    period_results.append(result)
                except Exception as e:
                    logger.warning(f"Suite test failed for {period.name}: {e}")

            results[regime] = MultiPeriodResult(
                results=period_results,
                strategy_name=getattr(strategy, 'name', 'unknown')
            )

        return results

    def calculate_regime_fitness(
        self,
        strategy: Any,
        current_vix: float = 15.0,
        current_trend: float = 0.0,
        current_correlation: float = 0.5,
        data: Dict[str, pd.DataFrame] = None,
        vix_data: pd.DataFrame = None,
        weight_similar: float = 0.5,
        weight_diverse: float = 0.3,
        weight_crisis: float = 0.2
    ) -> float:
        """
        Calculate fitness with emphasis on regime-matched periods.

        Combines:
        - Performance on periods similar to current conditions
        - Performance across diverse conditions (robustness)
        - Crisis performance (stress test)

        Args:
            strategy: Strategy to evaluate
            current_vix: Current VIX level
            current_trend: Current market trend (-1 to 1)
            current_correlation: Current correlation regime (0-1)
            data: Market data
            vix_data: VIX data
            weight_similar: Weight for similar period performance
            weight_diverse: Weight for diverse period performance
            weight_crisis: Weight for crisis performance

        Returns:
            Composite fitness score
        """
        if data:
            self.cache_data(data)

        fitness_components = []

        # 1. Similar period performance
        similar_periods = self.library.find_similar_periods(
            current_vix,
            current_trend,
            current_correlation,
            top_n=3
        )

        if similar_periods:
            similar_results = []
            for period, similarity in similar_periods:
                try:
                    result = self.run_period_test(strategy, period, None, vix_data)
                    # Weight by similarity
                    weighted_sharpe = result.sharpe_ratio * similarity
                    similar_results.append(weighted_sharpe)
                except Exception as e:
                    logger.debug(f"Similar period test failed for {period.name}: {e}")
                    continue

            if similar_results:
                similar_fitness = np.mean(similar_results)
                fitness_components.append(('similar', similar_fitness, weight_similar))

        # 2. Diverse period performance
        diverse_periods = self.library.get_diverse_test_set(4)
        diverse_results = []

        for period in diverse_periods:
            try:
                result = self.run_period_test(strategy, period, None, vix_data)
                diverse_results.append(result.sharpe_ratio)
            except Exception as e:
                logger.debug(f"Diverse period test failed for {period.name}: {e}")
                continue

        if diverse_results:
            diverse_fitness = np.mean(diverse_results) * (1 - np.std(diverse_results) / 2)  # Penalize variance
            fitness_components.append(('diverse', diverse_fitness, weight_diverse))

        # 3. Crisis performance
        crisis_periods = self.library.get_crisis_periods()[:2]  # Top 2 crisis periods
        crisis_results = []

        for period in crisis_periods:
            try:
                result = self.run_period_test(strategy, period, None, vix_data)
                # For crisis, we care about not losing too much
                # Positive Sharpe during crisis is exceptional
                crisis_score = max(0, result.sharpe_ratio + 0.5)  # Shift to reward survival
                crisis_results.append(crisis_score)
            except Exception as e:
                logger.debug(f"Crisis period test failed for {period.name}: {e}")
                continue

        if crisis_results:
            crisis_fitness = np.mean(crisis_results)
            fitness_components.append(('crisis', crisis_fitness, weight_crisis))

        # Combine components
        if not fitness_components:
            return 0.0

        total_weight = sum(w for _, _, w in fitness_components)
        composite_fitness = sum(f * w for _, f, w in fitness_components) / total_weight

        return max(0, composite_fitness)


# =============================================================================
# Utility functions for GA integration
# =============================================================================

def create_rapid_fitness_function(
    backtester: RapidBacktester,
    strategy_factory: Callable[[Dict], Any],
    period_names: List[str] = None,
    weights: Dict[str, float] = None
) -> Callable[[Dict], float]:
    """
    Create a fitness function for GA optimization using rapid backtesting.

    Args:
        backtester: RapidBacktester instance
        strategy_factory: Function that creates strategy from genes
        period_names: Periods to test on (uses diverse set if None)
        weights: Metric weights for fitness calculation

    Returns:
        Fitness function that takes genes dict and returns float
    """
    import math

    weights = weights or {
        'sharpe': 0.35,
        'sortino': 0.25,
        'consistency': 0.20,
        'alpha': 0.10,
        'crisis_survival': 0.10,
    }

    # Statistical validity threshold (research recommends 30+ for basic inference)
    MIN_TRADES_THRESHOLD = 30

    # Fitness ranges:
    # - 0.0: Strategy crashed or produced no results
    # - 0.01-0.10: Infeasible (below threshold) - ranked by Deb's feasibility rules
    # - 0.10-1.0+: Feasible - actual performance with exponential trade factor

    def fitness_fn(genes: Dict[str, float]) -> float:
        try:
            # Create strategy
            strategy = strategy_factory(genes)

            # Run multi-period test
            result = backtester.run_multi_period_test(
                strategy,
                period_names=period_names
            )

            # No results at all = crashed strategy
            if not result.results:
                return 0.0

            total_trades = result.total_trades

            # =================================================================
            # Deb's Feasibility Rules for infeasible solutions
            # =================================================================
            # Instead of fitness=0 (death penalty), rank by constraint violation.
            # This gives the GA gradient information to navigate toward feasibility.
            # Range: 0.01 (0 trades) to 0.10 (threshold-1 trades)
            if total_trades < MIN_TRADES_THRESHOLD:
                # Proxy metric: small score based on how close to threshold
                # Zero trades gets 0.01, threshold-1 trades gets ~0.10
                feasibility_score = 0.01 + 0.09 * (total_trades / MIN_TRADES_THRESHOLD)
                return feasibility_score

            # =================================================================
            # Feasible solutions: Calculate actual performance fitness
            # =================================================================

            # Calculate base fitness from performance metrics
            base_fitness = 0.0

            # Sharpe (normalized to ~0-1 range for typical values)
            sharpe_score = max(0, min(1, result.avg_sharpe / 2))
            base_fitness += sharpe_score * weights['sharpe']

            # Consistency (inverse of variance)
            consistency = result.consistency_score
            base_fitness += consistency * weights['consistency']

            # Alpha (beat market by meaningful amount)
            alpha_score = max(0, min(1, (result.avg_alpha + 10) / 20))
            base_fitness += alpha_score * weights['alpha']

            # Crisis survival (not blowing up during crisis)
            if result.crisis_sharpe is not None:
                crisis_score = max(0, min(1, (result.crisis_sharpe + 1) / 2))
                base_fitness += crisis_score * weights['crisis_survival']

            # =================================================================
            # Exponential soft penalty for trade count
            # =================================================================
            # Even feasible solutions get rewarded for more trades (statistical robustness)
            # At 30 trades: factor = 0.63, at 60: 0.86, at 90: 0.95
            trade_factor = 1 - math.exp(-total_trades / MIN_TRADES_THRESHOLD)

            # Final fitness: base performance scaled by trade factor
            # Minimum 0.10 to ensure feasible always beats infeasible
            fitness = 0.10 + base_fitness * trade_factor

            return fitness

        except Exception as e:
            logger.debug(f"Fitness calculation failed: {e}")
            return 0.0

    return fitness_fn


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("RAPID BACKTESTER DEMO")
    print("=" * 60)

    # Create backtester
    backtester = RapidBacktester()

    # Show available periods
    library = backtester.library
    print(f"\nAvailable periods: {len(library.get_all_periods())}")
    print(f"Short periods (for rapid testing): {len(library.get_short_periods())}")
    print(f"Crisis periods: {len(library.get_crisis_periods())}")

    # Show test suite
    print("\n" + "-" * 40)
    print("Rapid test suite:")
    suite = library.get_rapid_test_suite()
    for regime, periods in suite.items():
        print(f"  {regime.upper()}: {len(periods)} periods")
        for p in periods:
            print(f"    - {p.name} ({p.duration_days} days)")

    print("\nTo use with actual data and strategies:")
    print("  backtester.cache_data(data)")
    print("  result = backtester.run_period_test(strategy, period)")
    print("  suite_results = backtester.run_rapid_test_suite(strategy)")
    print("\nFor parallel pool usage (much faster for GA):")
    print("  with RapidBacktester() as bt:")
    print("      bt.init_parallel_pool(data, vix)")
    print("      result = bt.run_multi_period_test_parallel(factory, genes)")
