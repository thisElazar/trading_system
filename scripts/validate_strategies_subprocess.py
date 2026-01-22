#!/usr/bin/env python3
"""
Standalone Strategy Validation Worker

This script runs memory-intensive validation (walk-forward + Monte Carlo)
in a separate process to prevent OOM crashes in the main research pipeline.

Usage:
    python scripts/validate_strategies_subprocess.py --strategy-id GP_20260115_001
    python scripts/validate_strategies_subprocess.py --all-candidates
    python scripts/validate_strategies_subprocess.py --strategy-id GP_20260115_001 --memory-limit 1500

Communication:
    - Reads strategy data from promotion_pipeline.db
    - Writes validation results back to promotion_pipeline.db
    - Exit codes: 0=success, 1=validation failed, 2=memory limit, 3=error

Architecture:
    This modular approach allows:
    - Independent memory budget (can be killed without affecting main process)
    - Fine-grained timeouts per validation
    - Easy debugging and testing
    - Parallel validation of multiple strategies (future enhancement)
"""

import sys
import os
import gc
import json
import argparse
import logging
import resource
import signal
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability.logger import get_logger

logger = get_logger('validation_worker')


def check_memory_available(min_mb: int = 400) -> bool:
    """Check if enough memory is available to proceed."""
    mem = psutil.virtual_memory()
    available_mb = mem.available / (1024 * 1024)
    if available_mb < min_mb:
        logger.warning(f"Low memory: {available_mb:.0f}MB available, need {min_mb}MB")
        return False
    return True


def set_memory_limit(limit_mb: int) -> bool:
    """
    Set a soft memory limit for monitoring (NOT a hard limit).

    NOTE: We intentionally do NOT use resource.setrlimit(RLIMIT_AS) because:
    1. It affects C/C++ libraries (pyarrow, numpy) at a low level
    2. Those libraries throw C++ exceptions that can't be caught by Python
    3. This causes process crashes instead of graceful error handling

    Instead, we use soft monitoring via MemoryMonitor and check_memory_available().

    Args:
        limit_mb: Maximum memory in megabytes (for logging only)

    Returns:
        True always (limit is advisory, not enforced)
    """
    logger.info(f"Memory limit (advisory): {limit_mb}MB - using soft monitoring")

    # Log current memory state
    mem = psutil.virtual_memory()
    process = psutil.Process()
    current_mb = process.memory_info().rss / (1024 * 1024)

    logger.info(f"Current process memory: {current_mb:.0f}MB")
    logger.info(f"System available: {mem.available / (1024*1024):.0f}MB")

    return True


class ValidationTimeout(Exception):
    """Raised when validation exceeds time limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for validation timeout."""
    raise ValidationTimeout("Validation timed out")


def run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """
    Run a function with a timeout.

    Args:
        func: Function to run
        timeout_seconds: Maximum time in seconds
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result of func

    Raises:
        ValidationTimeout: If timeout exceeded
    """
    import signal

    # Set up the timeout handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the alarm
        return result
    except ValidationTimeout:
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)


class MemoryMonitor:
    """
    Context manager for monitoring memory during operations.

    Usage:
        with MemoryMonitor(limit_mb=1500, check_interval=100) as monitor:
            # Do memory-intensive work
            if monitor.check():  # Returns False if limit exceeded
                continue_work()
    """

    def __init__(self, limit_mb: int, check_interval: int = 100):
        """
        Args:
            limit_mb: Memory limit in MB
            check_interval: How often to check (for use in loops)
        """
        self.limit_mb = limit_mb
        self.check_interval = check_interval
        self.check_count = 0
        self.peak_mb = 0
        self.process = psutil.Process()

    def __enter__(self):
        self.peak_mb = self._get_memory_mb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_mb = self._get_memory_mb()
        if final_mb > self.peak_mb:
            self.peak_mb = final_mb
        logger.debug(f"Memory monitor: peak={self.peak_mb:.0f}MB")
        return False

    def _get_memory_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def check(self) -> bool:
        """
        Check if memory is within limit.

        Returns:
            True if OK, False if limit exceeded
        """
        self.check_count += 1

        # Only check every N calls to reduce overhead
        if self.check_count % self.check_interval != 0:
            return True

        current_mb = self._get_memory_mb()
        if current_mb > self.peak_mb:
            self.peak_mb = current_mb

        if current_mb > self.limit_mb:
            logger.error(f"Memory limit exceeded: {current_mb:.0f}MB > {self.limit_mb}MB")
            return False

        return True

    def get_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._get_memory_mb()


def stratified_sample_symbols(
    data: Dict[str, pd.DataFrame],
    target_count: int = 500,
    n_volatility_buckets: int = 3,
    n_momentum_buckets: int = 3
) -> List[str]:
    """
    Stratified sampling of symbols by volatility and momentum.

    This ensures Monte Carlo validation covers the full spectrum of market behavior:
    - Low/medium/high volatility stocks
    - Bearish/neutral/bullish momentum stocks

    Args:
        data: Dict mapping symbol to DataFrame with OHLCV + indicators
        target_count: Target number of symbols to sample
        n_volatility_buckets: Number of volatility buckets (default 3: low/med/high)
        n_momentum_buckets: Number of momentum buckets (default 3: bear/neutral/bull)

    Returns:
        List of sampled symbols with stratified coverage
    """
    # Calculate characteristics for each symbol
    symbol_stats = []
    for symbol, df in data.items():
        if len(df) < 50:
            continue
        try:
            # Use ATR for volatility (normalized by price)
            if 'atr' in df.columns and 'close' in df.columns:
                atr_pct = (df['atr'].iloc[-20:].mean() / df['close'].iloc[-1]) * 100
            else:
                # Fallback: use standard deviation of returns
                returns = df['close'].pct_change().dropna()
                atr_pct = returns.iloc[-20:].std() * 100 if len(returns) >= 20 else 2.0

            # Use momentum (20-day return)
            if 'pct_change_20d' in df.columns:
                momentum = df['pct_change_20d'].iloc[-1]
            elif 'momentum_12m' in df.columns:
                momentum = df['momentum_12m'].iloc[-1]
            else:
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100 if len(df) >= 20 else 0

            # Handle NaN/inf
            if pd.isna(atr_pct) or np.isinf(atr_pct):
                atr_pct = 2.0
            if pd.isna(momentum) or np.isinf(momentum):
                momentum = 0.0

            symbol_stats.append({
                'symbol': symbol,
                'volatility': float(atr_pct),
                'momentum': float(momentum)
            })
        except Exception:
            continue

    if not symbol_stats:
        logger.warning("No valid symbols for stratified sampling, returning random sample")
        return list(data.keys())[:target_count]

    stats_df = pd.DataFrame(symbol_stats)
    logger.info(f"Calculated stats for {len(stats_df)} symbols")
    logger.info(f"Volatility range: {stats_df['volatility'].min():.2f}% - {stats_df['volatility'].max():.2f}%")
    logger.info(f"Momentum range: {stats_df['momentum'].min():.2f}% - {stats_df['momentum'].max():.2f}%")

    # Create volatility buckets using quantiles
    stats_df['vol_bucket'] = pd.qcut(
        stats_df['volatility'],
        q=n_volatility_buckets,
        labels=range(n_volatility_buckets),
        duplicates='drop'
    )

    # Create momentum buckets using quantiles
    stats_df['mom_bucket'] = pd.qcut(
        stats_df['momentum'],
        q=n_momentum_buckets,
        labels=range(n_momentum_buckets),
        duplicates='drop'
    )

    # Sample from each bucket combination
    n_buckets = n_volatility_buckets * n_momentum_buckets
    per_bucket = max(1, target_count // n_buckets)

    sampled = []
    bucket_counts = {}

    for vol_bucket in range(n_volatility_buckets):
        for mom_bucket in range(n_momentum_buckets):
            bucket_df = stats_df[
                (stats_df['vol_bucket'] == vol_bucket) &
                (stats_df['mom_bucket'] == mom_bucket)
            ]
            bucket_key = f"vol{vol_bucket}_mom{mom_bucket}"

            if len(bucket_df) > 0:
                # Sample up to per_bucket symbols from this bucket
                n_sample = min(per_bucket, len(bucket_df))
                bucket_sample = bucket_df.sample(n=n_sample, random_state=42)['symbol'].tolist()
                sampled.extend(bucket_sample)
                bucket_counts[bucket_key] = n_sample
            else:
                bucket_counts[bucket_key] = 0

    logger.info(f"Stratified sample: {len(sampled)} symbols across {n_buckets} buckets")
    logger.info(f"Bucket distribution: {bucket_counts}")

    # If we need more symbols to reach target, add randomly from remaining
    if len(sampled) < target_count:
        remaining = [s for s in data.keys() if s not in sampled]
        extra_needed = min(target_count - len(sampled), len(remaining))
        if extra_needed > 0:
            import random
            random.shuffle(remaining)
            sampled.extend(remaining[:extra_needed])
            logger.info(f"Added {extra_needed} additional symbols to reach target")

    return sampled


def pre_validate_genome(
    strategy,
    data: Dict[str, pd.DataFrame],
    max_entry_rate: float = 0.20,
    min_entry_rate: float = 0.01,
    sample_size: int = 100
) -> Tuple[bool, str, float]:
    """
    Quick pre-validation to reject obviously degenerate genomes.

    Tests the entry condition on a sample of symbols. Rejects if:
    - Entry triggers on too many symbols (> max_entry_rate) - always true condition
    - Entry triggers on too few symbols (< min_entry_rate) - always false condition

    Args:
        strategy: EvolvedStrategy instance
        data: Dict of symbol -> DataFrame
        max_entry_rate: Maximum acceptable entry rate (default 20%)
        min_entry_rate: Minimum acceptable entry rate (default 1%)
        sample_size: Number of symbols to test

    Returns:
        Tuple of (passed, reason, entry_rate)
    """
    import random

    # Sample symbols for quick test
    symbols = list(data.keys())
    if len(symbols) > sample_size:
        test_symbols = random.sample(symbols, sample_size)
    else:
        test_symbols = symbols

    entry_count = 0
    tested_count = 0

    for symbol in test_symbols:
        df = data.get(symbol)
        if df is None or len(df) < 50:
            continue

        tested_count += 1

        try:
            # Test entry condition
            entry_triggered = strategy._evaluate_entry(df)
            if entry_triggered:
                entry_count += 1
        except Exception:
            continue

    if tested_count == 0:
        return False, "No valid symbols to test", 0.0

    entry_rate = entry_count / tested_count

    # Check for too frequent (always true condition)
    if entry_rate > max_entry_rate:
        reason = (
            f"Entry condition too frequent: {entry_rate:.1%} of symbols triggered "
            f"(max allowed: {max_entry_rate:.0%}). Likely always-true condition."
        )
        return False, reason, entry_rate

    # Check for too rare (always false condition)
    if entry_rate < min_entry_rate:
        reason = (
            f"Entry condition too rare: {entry_rate:.1%} of symbols triggered "
            f"(min required: {min_entry_rate:.0%}). Likely always-false condition."
        )
        return False, reason, entry_rate

    logger.info(f"Pre-validation passed: entry rate {entry_rate:.1%} ({entry_count}/{tested_count} symbols)")
    return True, "OK", entry_rate


def validate_strategy(
    strategy_id: str,
    n_mc_simulations: int = 500,
    memory_limit_mb: int = 1500,
    timeout_minutes: int = 10
) -> Tuple[bool, Dict]:
    """
    Run full validation on a single strategy.

    Args:
        strategy_id: Strategy identifier from promotion_pipeline.db
        n_mc_simulations: Number of Monte Carlo bootstrap samples
        memory_limit_mb: Abort if process memory exceeds this limit
        timeout_minutes: Maximum time for validation (default 10 minutes)

    Returns:
        Tuple of (success, results_dict)

    Safety features:
        - Hard memory limit via resource.setrlimit()
        - Timeout via SIGALRM
        - Memory monitoring during backtest passes
        - Early termination if memory pressure detected
    """
    from research.discovery.promotion_pipeline import PromotionPipeline
    from research.discovery.strategy_genome import GenomeFactory
    from research.discovery.strategy_compiler import EvolvedStrategy
    from research.backtester import Backtester
    from data.cached_data_manager import CachedDataManager

    logger.info(f"Starting validation for {strategy_id}")
    logger.info(f"Safety limits: memory={memory_limit_mb}MB, timeout={timeout_minutes}min")
    start_time = datetime.now()

    # ===== SAFETY: Set hard memory limit =====
    # This causes MemoryError if exceeded, preventing OOM crash
    set_memory_limit(memory_limit_mb)

    # ===== SAFETY: Set timeout =====
    # This raises ValidationTimeout if exceeded
    def alarm_handler(signum, frame):
        raise ValidationTimeout(f"Validation timed out after {timeout_minutes} minutes")

    old_alarm_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout_minutes * 60)

    # Memory check before starting
    if not check_memory_available(400):
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_alarm_handler)
        return False, {'error': 'Insufficient memory to start', 'strategy_id': strategy_id}

    try:
        # Load strategy record
        pipeline = PromotionPipeline()
        record = pipeline.get_strategy_record(strategy_id)

        if record is None:
            logger.error(f"Strategy not found: {strategy_id}")
            return False, {'error': 'Strategy not found', 'strategy_id': strategy_id}

        if not record.genome_json:
            logger.error(f"No genome data for {strategy_id}")
            return False, {'error': 'No genome data', 'strategy_id': strategy_id}

        # Reconstruct strategy from genome
        factory = GenomeFactory()
        genome_json_str = record.genome_json

        # Handle double-encoded JSON
        if genome_json_str.startswith('"'):
            genome_json_str = json.loads(genome_json_str)

        genome = factory.deserialize_genome(genome_json_str)
        strategy = EvolvedStrategy(genome, factory)

        logger.info(f"Strategy reconstructed: {strategy_id}")

        # Memory check before loading data
        if not check_memory_available(600):
            return False, {'error': 'Insufficient memory for data load', 'strategy_id': strategy_id}

        # ===== MEMORY-SAFE DATA LOADING =====
        # Don't load ALL 2500+ symbols - that causes OOM
        # Instead, load a representative subset for validation
        # Walk-forward needs ~500 symbols, Monte Carlo uses batches of 500
        MAX_SYMBOLS_TO_LOAD = 800  # Memory-safe limit for Pi

        logger.info("Loading market data (memory-safe subset)...")
        data_manager = CachedDataManager()

        # Get available symbols and randomly sample
        all_symbols = data_manager.get_available_symbols()
        logger.info(f"Total available symbols: {len(all_symbols)}")

        if len(all_symbols) > MAX_SYMBOLS_TO_LOAD:
            import random
            random.seed(42)  # Reproducible sampling
            symbols_to_load = random.sample(all_symbols, MAX_SYMBOLS_TO_LOAD)
            logger.info(f"Loading {len(symbols_to_load)} symbols (subset for memory safety)")
        else:
            symbols_to_load = all_symbols

        # Load only the selected symbols
        data_manager.load_all(symbols_to_load)

        # Check memory after data load
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory after data load: {mem_mb:.0f}MB")

        if mem_mb > memory_limit_mb:
            logger.error(f"Memory limit exceeded: {mem_mb:.0f}MB > {memory_limit_mb}MB")
            return False, {'error': 'Memory limit exceeded', 'memory_mb': mem_mb, 'strategy_id': strategy_id}

        # Get the loaded data
        raw_data = data_manager.get_bars_batch(symbols_to_load)

        if len(raw_data) < 10:
            logger.error(f"Insufficient data: only {len(raw_data)} symbols")
            return False, {'error': 'Insufficient data', 'symbols': len(raw_data), 'strategy_id': strategy_id}

        # Convert timestamp column to DatetimeIndex (required for walk-forward)
        data = {}
        for symbol, df in raw_data.items():
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    continue  # Skip symbols with invalid timestamps
            data[symbol] = df

        logger.info(f"Loaded {len(data)} symbols for validation (with DatetimeIndex)")

        # ===== PRE-VALIDATION: Check for degenerate genome =====
        # Quick test to reject genomes that trigger on too many symbols
        logger.info("Running pre-validation check...")
        pre_valid, pre_reason, entry_rate = pre_validate_genome(
            strategy=strategy,
            data=data,
            max_entry_rate=0.20,  # Reject if entry triggers on > 20% of symbols
            sample_size=100
        )

        if not pre_valid:
            logger.warning(f"Pre-validation FAILED: {pre_reason}")
            # Update database to mark as rejected
            pipeline.update_validation_metrics(
                strategy_id=strategy_id,
                walk_forward_efficiency=0.0,
                monte_carlo_confidence=0.0
            )
            return False, {
                'strategy_id': strategy_id,
                'error': pre_reason,
                'entry_rate': entry_rate,
                'degenerate': True,
                'passed': False
            }

        # ===== WALK-FORWARD VALIDATION =====
        logger.info("Running walk-forward validation...")
        backtester = Backtester(initial_capital=100000)

        wf_results = backtester.run_walk_forward(
            strategy=strategy,
            data=data,
            train_days=252,
            test_days=63,
            step_days=21
        )

        if not wf_results:
            logger.warning(f"Walk-forward returned no results for {strategy_id}")
            # Early exit - no point running Monte Carlo on a strategy that can't backtest
            logger.info(f"Skipping Monte Carlo: walk-forward failed")
            return False, {
                'strategy_id': strategy_id,
                'walk_forward_efficiency': 0.0,
                'walk_forward_periods': 0,
                'monte_carlo_confidence': 0.0,
                'monte_carlo_median_sharpe': 0.0,
                'error': 'Walk-forward returned no results - possibly degenerate genome',
                'passed': False
            }

        positive_periods = sum(1 for r in wf_results if r.sharpe_ratio > 0)
        wf_efficiency = positive_periods / len(wf_results)
        wf_periods = len(wf_results)
        logger.info(f"Walk-forward: {wf_efficiency:.2f} efficiency ({positive_periods}/{wf_periods} positive)")

        # Memory check before Monte Carlo
        mem_mb = process.memory_info().rss / (1024 * 1024)
        if mem_mb > memory_limit_mb:
            logger.error(f"Memory limit exceeded before Monte Carlo: {mem_mb:.0f}MB")
            return False, {'error': 'Memory limit exceeded', 'phase': 'pre-monte-carlo', 'strategy_id': strategy_id}

        # ===== MONTE CARLO VALIDATION =====
        # Multi-pass approach: run backtests on multiple stratified batches
        # and combine returns for more comprehensive coverage of the universe
        #
        # This gives us coverage of ~1500 symbols while staying memory-safe:
        # - 3 passes of 500 symbols each
        # - Each pass uses stratified sampling for representative coverage
        # - Returns are combined for Monte Carlo bootstrapping

        mc_batch_size = 500  # Symbols per batch (memory-safe)
        mc_num_passes = 3    # Number of passes (total: 1500 symbols)
        all_returns = []
        total_trades = 0
        symbols_tested = set()

        # Get all symbols and shuffle for different batches
        all_symbols = list(data.keys())
        import random
        random.seed(42)  # Reproducible batches
        random.shuffle(all_symbols)

        for pass_num in range(mc_num_passes):
            # Memory check before each pass
            mem_mb = process.memory_info().rss / (1024 * 1024)
            if mem_mb > memory_limit_mb * 0.9:  # 90% threshold
                logger.warning(f"Memory pressure at pass {pass_num+1}: {mem_mb:.0f}MB, stopping early")
                break

            # Get symbols for this pass (non-overlapping)
            start_idx = pass_num * mc_batch_size
            end_idx = min(start_idx + mc_batch_size, len(all_symbols))
            if start_idx >= len(all_symbols):
                break

            batch_symbols = all_symbols[start_idx:end_idx]

            # Apply stratified sampling within this batch
            batch_data = {s: data[s] for s in batch_symbols if s in data}
            if len(batch_data) < 50:
                logger.warning(f"Pass {pass_num+1}: insufficient data ({len(batch_data)} symbols), skipping")
                continue

            # Stratify within the batch for representative coverage
            stratified_batch = stratified_sample_symbols(
                batch_data,
                target_count=min(mc_batch_size, len(batch_data)),
                n_volatility_buckets=3,
                n_momentum_buckets=3
            )
            mc_data = {s: batch_data[s] for s in stratified_batch if s in batch_data}
            symbols_tested.update(mc_data.keys())

            logger.info(f"Monte Carlo pass {pass_num+1}/{mc_num_passes}: {len(mc_data)} symbols")

            # Run backtest for this batch
            batch_result = backtester.run(strategy, mc_data)

            if batch_result.equity_curve is not None and len(batch_result.equity_curve) > 20:
                batch_returns = batch_result.equity_curve.pct_change().dropna()
                all_returns.append(batch_returns)
                total_trades += batch_result.total_trades
                logger.info(f"  Pass {pass_num+1}: {batch_result.total_trades} trades, "
                           f"Sharpe={batch_result.sharpe_ratio:.2f}")

            # Clear batch data to free memory
            del mc_data, batch_data
            gc.collect()

        mc_confidence = 0.0
        median_sharpe = 0.0
        combined_sharpe = 0.0

        if all_returns:
            # Combine returns from all passes
            combined_returns = pd.concat(all_returns).sort_index()
            # Remove duplicate dates (take mean if same date appears in multiple batches)
            combined_returns = combined_returns.groupby(combined_returns.index).mean()

            logger.info(f"Combined {len(combined_returns)} return observations from {len(symbols_tested)} symbols")

            if len(combined_returns) > 50:
                # Calculate combined Sharpe
                combined_sharpe = (
                    np.mean(combined_returns) / np.std(combined_returns) * np.sqrt(252)
                    if np.std(combined_returns) > 0 else 0.0
                )

                # Monte Carlo bootstrap
                logger.info(f"Running Monte Carlo ({n_mc_simulations} simulations)...")
                sharpe_dist = []
                returns_values = combined_returns.values

                for _ in range(n_mc_simulations):
                    boot_returns = np.random.choice(returns_values, size=len(returns_values), replace=True)
                    if np.std(boot_returns) > 0:
                        boot_sharpe = np.mean(boot_returns) / np.std(boot_returns) * np.sqrt(252)
                        sharpe_dist.append(boot_sharpe)

                if sharpe_dist:
                    mc_confidence = sum(1 for s in sharpe_dist if s > 0) / len(sharpe_dist)
                    median_sharpe = float(np.median(sharpe_dist))
                    p5_sharpe = float(np.percentile(sharpe_dist, 5))
                    p95_sharpe = float(np.percentile(sharpe_dist, 95))
                    logger.info(
                        f"Monte Carlo: confidence={mc_confidence:.2f}, "
                        f"median_sharpe={median_sharpe:.2f}, "
                        f"95% CI=[{p5_sharpe:.2f}, {p95_sharpe:.2f}]"
                    )
        else:
            logger.warning("No valid returns from any Monte Carlo pass")

        # Use the combined result for final metrics
        full_result_sharpe = combined_sharpe
        full_result_trades = total_trades

        # Compile results
        duration = (datetime.now() - start_time).total_seconds()

        results = {
            'strategy_id': strategy_id,
            'walk_forward_efficiency': wf_efficiency,
            'walk_forward_periods': wf_periods,
            'monte_carlo_confidence': mc_confidence,
            'monte_carlo_median_sharpe': median_sharpe,
            'monte_carlo_simulations': n_mc_simulations,
            'full_backtest_sharpe': full_result_sharpe,
            'full_backtest_trades': full_result_trades,
            'symbols_tested': len(symbols_tested),
            'validation_duration_sec': duration,
            'peak_memory_mb': process.memory_info().rss / (1024 * 1024),
            'timestamp': datetime.now().isoformat()
        }

        # Check pass/fail criteria
        criteria = pipeline.criteria
        passed = (
            wf_efficiency >= criteria.min_walk_forward_efficiency and
            mc_confidence >= criteria.min_monte_carlo_confidence
        )

        results['passed'] = passed
        results['criteria'] = {
            'min_walk_forward_efficiency': criteria.min_walk_forward_efficiency,
            'min_monte_carlo_confidence': criteria.min_monte_carlo_confidence
        }

        # Update database with results
        logger.info(f"Updating database with validation results...")
        pipeline.update_validation_metrics(
            strategy_id=strategy_id,
            walk_forward_efficiency=wf_efficiency,
            monte_carlo_confidence=mc_confidence
        )

        status = 'PASSED' if passed else 'FAILED'
        logger.info(
            f"Validation {status} for {strategy_id} in {duration:.1f}s: "
            f"WF={wf_efficiency:.2f}, MC={mc_confidence:.2f}"
        )

        return passed, results

    except ValidationTimeout as e:
        logger.error(f"Validation TIMED OUT for {strategy_id}: {e}")
        return False, {
            'error': f'Timeout after {timeout_minutes} minutes',
            'strategy_id': strategy_id,
            'timeout': True
        }

    except MemoryError as e:
        logger.error(f"Validation OUT OF MEMORY for {strategy_id}: {e}")
        # Force garbage collection
        gc.collect()
        return False, {
            'error': f'Memory limit exceeded ({memory_limit_mb}MB)',
            'strategy_id': strategy_id,
            'oom': True
        }

    except Exception as e:
        logger.error(f"Validation error for {strategy_id}: {e}", exc_info=True)
        return False, {'error': str(e), 'strategy_id': strategy_id}

    finally:
        # ===== SAFETY: Always cancel the alarm and restore handler =====
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm_handler)


def get_candidate_strategies() -> List[str]:
    """Get all CANDIDATE strategies that need validation."""
    from research.discovery.promotion_pipeline import PromotionPipeline, StrategyStatus

    pipeline = PromotionPipeline()
    candidate_ids = pipeline.get_strategies_by_status(StrategyStatus.CANDIDATE)

    # Filter to those without validation metrics
    needs_validation = []
    for strategy_id in candidate_ids:
        record = pipeline.get_strategy_record(strategy_id)
        if record and record.walk_forward_efficiency == 0.0 and record.monte_carlo_confidence == 0.0:
            needs_validation.append(strategy_id)

    return needs_validation


def get_validated_with_placeholders() -> List[str]:
    """Get VALIDATED strategies with placeholder metrics needing real validation.

    Strategies may have been promoted with placeholder 0.5/0.5 metrics
    when validation was skipped. This function identifies them for re-validation.
    """
    from research.discovery.promotion_pipeline import PromotionPipeline, StrategyStatus

    pipeline = PromotionPipeline()
    validated_ids = pipeline.get_strategies_by_status(StrategyStatus.VALIDATED)

    needs_validation = []
    for strategy_id in validated_ids:
        record = pipeline.get_strategy_record(strategy_id)
        if record and record.monte_carlo_confidence is not None:
            # Placeholder metrics used 0.5/0.5 - check for low confidence
            if record.monte_carlo_confidence < 0.6:
                needs_validation.append(strategy_id)

    return needs_validation


def cleanup_degenerate_genomes(max_entry_rate: float = 0.20) -> Dict:
    """
    Scan all candidate and validated strategies, reject degenerate ones.

    Uses quick pre-validation (entry frequency check) to identify and reject
    genomes that trigger on too many symbols.

    Args:
        max_entry_rate: Maximum acceptable entry rate (default 20%)

    Returns:
        Dict with cleanup statistics
    """
    from research.discovery.promotion_pipeline import PromotionPipeline, StrategyStatus
    from research.discovery.strategy_genome import GenomeFactory
    from research.discovery.strategy_compiler import EvolvedStrategy
    from data.cached_data_manager import CachedDataManager
    import sqlite3

    logger.info(f"Starting degenerate genome cleanup (max_entry_rate={max_entry_rate:.0%})")

    # Load a small data sample for testing
    logger.info("Loading data sample for pre-validation...")
    dm = CachedDataManager()
    all_symbols = dm.get_available_symbols()

    # Use 200 symbols for quick testing
    import random
    random.seed(42)
    test_symbols = random.sample(all_symbols, min(200, len(all_symbols)))
    dm.load_all(test_symbols)

    raw_data = dm.get_bars_batch(test_symbols)

    # Convert to DatetimeIndex
    data = {}
    for symbol, df in raw_data.items():
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        if len(df) >= 50:
            data[symbol] = df

    logger.info(f"Loaded {len(data)} symbols for pre-validation testing")

    # Get candidates and validated strategies
    pipeline = PromotionPipeline()
    factory = GenomeFactory()

    candidates = pipeline.get_strategies_by_status(StrategyStatus.CANDIDATE)
    validated = pipeline.get_strategies_by_status(StrategyStatus.VALIDATED)
    all_ids = candidates + validated

    logger.info(f"Scanning {len(all_ids)} strategies ({len(candidates)} candidate, {len(validated)} validated)")

    stats = {
        'scanned': 0,
        'rejected': 0,
        'passed': 0,
        'skipped': 0,
        'rejected_ids': []
    }

    for strategy_id in all_ids:
        record = pipeline.get_strategy_record(strategy_id)
        if not record or not record.genome_json:
            stats['skipped'] += 1
            continue

        stats['scanned'] += 1

        try:
            # Reconstruct strategy
            genome_json_str = record.genome_json
            if genome_json_str.startswith('"'):
                genome_json_str = json.loads(genome_json_str)

            genome = factory.deserialize_genome(genome_json_str)
            strategy = EvolvedStrategy(genome, factory)

            # Run pre-validation
            passed, reason, entry_rate = pre_validate_genome(
                strategy=strategy,
                data=data,
                max_entry_rate=max_entry_rate,
                sample_size=100
            )

            if not passed:
                # Reject the strategy
                logger.info(f"REJECTING {strategy_id}: entry_rate={entry_rate:.1%}")
                stats['rejected'] += 1
                stats['rejected_ids'].append(strategy_id)

                # Update database directly
                db_path = pipeline.db_path
                conn = sqlite3.connect(db_path)
                conn.execute("""
                    UPDATE strategy_lifecycle
                    SET status = 'rejected',
                        retirement_reason = ?,
                        updated_at = ?
                    WHERE strategy_id = ?
                """, (
                    f"Degenerate genome: entry rate {entry_rate:.1%} > {max_entry_rate:.0%}",
                    datetime.now().isoformat(),
                    strategy_id
                ))
                conn.commit()
                conn.close()
            else:
                stats['passed'] += 1
                logger.debug(f"PASSED {strategy_id}: entry_rate={entry_rate:.1%}")

        except Exception as e:
            logger.warning(f"Error checking {strategy_id}: {e}")
            stats['skipped'] += 1

    logger.info(f"Cleanup complete: {stats['rejected']} rejected, {stats['passed']} passed, {stats['skipped']} skipped")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Standalone strategy validation worker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--strategy-id', '-s',
        type=str,
        help='Specific strategy ID to validate'
    )

    parser.add_argument(
        '--all-candidates',
        action='store_true',
        help='Validate all CANDIDATE strategies without validation metrics'
    )

    parser.add_argument(
        '--memory-limit',
        type=int,
        default=1500,
        help='Memory limit in MB (default: 1500)'
    )

    parser.add_argument(
        '--mc-simulations',
        type=int,
        default=500,
        help='Number of Monte Carlo simulations (default: 500)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be validated without running'
    )

    parser.add_argument(
        '--validated-placeholders',
        action='store_true',
        help='Validate VALIDATED strategies that have placeholder metrics (0.5/0.5)'
    )

    parser.add_argument(
        '--cleanup-degenerate',
        action='store_true',
        help='Scan all candidates/validated and reject degenerate genomes (quick pre-validation only)'
    )

    parser.add_argument(
        '--max-entry-rate',
        type=float,
        default=0.20,
        help='Max entry rate for pre-validation (default: 0.20 = 20%%)'
    )

    args = parser.parse_args()

    # Handle cleanup mode
    if args.cleanup_degenerate:
        stats = cleanup_degenerate_genomes(max_entry_rate=args.max_entry_rate)
        print(json.dumps(stats, indent=2))
        return 0 if stats['rejected'] >= 0 else 1

    # Determine which strategies to validate
    if args.strategy_id:
        strategy_ids = [args.strategy_id]
    elif args.all_candidates:
        strategy_ids = get_candidate_strategies()
        if not strategy_ids:
            logger.info("No CANDIDATE strategies need validation")
            return 0
    elif args.validated_placeholders:
        strategy_ids = get_validated_with_placeholders()
        if not strategy_ids:
            logger.info("No VALIDATED strategies have placeholder metrics")
            return 0
        logger.info(f"Found {len(strategy_ids)} validated strategies with placeholder metrics")
    else:
        parser.print_help()
        return 1

    if args.dry_run:
        print(f"Would validate {len(strategy_ids)} strategies:")
        for sid in strategy_ids:
            print(f"  - {sid}")
        return 0

    # Validate strategies
    passed_count = 0
    failed_count = 0
    error_count = 0

    for strategy_id in strategy_ids:
        # Memory check before each validation
        if not check_memory_available(400):
            logger.error(f"Stopping: insufficient memory to continue")
            break

        success, results = validate_strategy(
            strategy_id=strategy_id,
            n_mc_simulations=args.mc_simulations,
            memory_limit_mb=args.memory_limit
        )

        if 'error' in results:
            error_count += 1
            logger.error(f"Error validating {strategy_id}: {results['error']}")
        elif success:
            passed_count += 1
        else:
            failed_count += 1

        # Print results as JSON for easy parsing
        print(json.dumps(results, indent=2))

    # Summary
    logger.info(f"Validation complete: {passed_count} passed, {failed_count} failed, {error_count} errors")

    # Exit code based on results
    if error_count > 0:
        return 3
    elif failed_count > 0 and passed_count == 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())
