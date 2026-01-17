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
import json
import argparse
import logging
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

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


def validate_strategy(
    strategy_id: str,
    n_mc_simulations: int = 500,
    memory_limit_mb: int = 1500
) -> Tuple[bool, Dict]:
    """
    Run full validation on a single strategy.

    Args:
        strategy_id: Strategy identifier from promotion_pipeline.db
        n_mc_simulations: Number of Monte Carlo bootstrap samples
        memory_limit_mb: Abort if process memory exceeds this limit

    Returns:
        Tuple of (success, results_dict)
    """
    from research.discovery.promotion_pipeline import PromotionPipeline
    from research.discovery.strategy_genome import GenomeFactory
    from research.discovery.strategy_compiler import EvolvedStrategy
    from research.backtester import Backtester
    from data.cached_data_manager import CachedDataManager

    logger.info(f"Starting validation for {strategy_id}")
    start_time = datetime.now()

    # Memory check before starting
    if not check_memory_available(400):
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

        # Load data - this is the memory-intensive part
        logger.info("Loading market data...")
        data_manager = CachedDataManager()
        data_manager.load_all()

        # Check memory after data load
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory after data load: {mem_mb:.0f}MB")

        if mem_mb > memory_limit_mb:
            logger.error(f"Memory limit exceeded: {mem_mb:.0f}MB > {memory_limit_mb}MB")
            return False, {'error': 'Memory limit exceeded', 'memory_mb': mem_mb, 'strategy_id': strategy_id}

        # Make data copies for validation (required for backtester)
        data = {s: df.copy() for s, df in data_manager.cache.items()}

        if len(data) < 10:
            logger.error(f"Insufficient data: only {len(data)} symbols")
            return False, {'error': 'Insufficient data', 'symbols': len(data), 'strategy_id': strategy_id}

        logger.info(f"Loaded {len(data)} symbols for validation")

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
            logger.warning("Walk-forward returned no results")
            wf_efficiency = 0.0
            wf_periods = 0
        else:
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
        logger.info(f"Running Monte Carlo validation ({n_mc_simulations} simulations)...")
        full_result = backtester.run(strategy, data)

        mc_confidence = 0.0
        median_sharpe = 0.0

        if full_result.equity_curve is not None and len(full_result.equity_curve) > 50:
            returns = full_result.equity_curve.pct_change().dropna()

            sharpe_dist = []
            for _ in range(n_mc_simulations):
                boot_returns = np.random.choice(returns.values, size=len(returns), replace=True)
                if np.std(boot_returns) > 0:
                    boot_sharpe = np.mean(boot_returns) / np.std(boot_returns) * np.sqrt(252)
                    sharpe_dist.append(boot_sharpe)

            if sharpe_dist:
                mc_confidence = sum(1 for s in sharpe_dist if s > 0) / len(sharpe_dist)
                median_sharpe = float(np.median(sharpe_dist))
                logger.info(f"Monte Carlo: {mc_confidence:.2f} confidence, {median_sharpe:.2f} median Sharpe")
        else:
            logger.warning("Insufficient equity curve for Monte Carlo")

        # Compile results
        duration = (datetime.now() - start_time).total_seconds()

        results = {
            'strategy_id': strategy_id,
            'walk_forward_efficiency': wf_efficiency,
            'walk_forward_periods': wf_periods,
            'monte_carlo_confidence': mc_confidence,
            'monte_carlo_median_sharpe': median_sharpe,
            'monte_carlo_simulations': n_mc_simulations,
            'full_backtest_sharpe': full_result.sharpe_ratio if full_result else 0.0,
            'full_backtest_trades': full_result.total_trades if full_result else 0,
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

    except Exception as e:
        logger.error(f"Validation error for {strategy_id}: {e}", exc_info=True)
        return False, {'error': str(e), 'strategy_id': strategy_id}


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

    args = parser.parse_args()

    # Determine which strategies to validate
    if args.strategy_id:
        strategy_ids = [args.strategy_id]
    elif args.all_candidates:
        strategy_ids = get_candidate_strategies()
        if not strategy_ids:
            logger.info("No CANDIDATE strategies need validation")
            return 0
        logger.info(f"Found {len(strategy_ids)} candidates needing validation")
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
