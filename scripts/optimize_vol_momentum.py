"""
Vol Momentum Parameter Optimization (2020-2025)
================================================
Focused optimization on recent market data for better parameter fit.

Usage:
    python scripts/optimize_vol_momentum.py
    python scripts/optimize_vol_momentum.py --quick  # Fast test run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def run_single_backtest(params: dict, data: dict, vix_data: pd.DataFrame = None) -> dict:
    """Run a single backtest with given parameters."""
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from research.backtester import Backtester

    # Create strategy with custom params
    strategy = VolManagedMomentumStrategy()
    strategy.formation_period = params['formation_period']
    strategy.skip_period = params['skip_period']
    strategy.vol_lookback = params['vol_lookback']
    strategy.target_vol = params['target_vol']
    strategy.top_percentile = params['top_percentile']
    strategy.last_rebalance_month = None

    # Run backtest
    backtester = Backtester(initial_capital=100000)

    try:
        result = backtester.run(strategy, data, vix_data=vix_data)

        return {
            **params,
            'sharpe': result.sharpe_ratio,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown_pct,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'sortino': getattr(result, 'sortino_ratio', result.sharpe_ratio * 1.2)
        }
    except Exception as e:
        return {**params, 'sharpe': -999, 'error': str(e)}


def optimize_parameters(quick: bool = False):
    """Run parameter grid search."""
    from data.unified_data_loader import UnifiedDataLoader

    print("=" * 60)
    print("VOL MOMENTUM PARAMETER OPTIMIZATION (2020-2025)")
    print("=" * 60)

    # Load data
    logger.info("Loading market data...")
    loader = UnifiedDataLoader()
    all_data = loader.load_all_daily()

    # Filter to top 50 liquid stocks
    sorted_syms = sorted(all_data.items(), key=lambda x: len(x[1]), reverse=True)[:50]

    # Prepare data with datetime index, filtered to 2020-2025
    data = {}
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)

    for symbol, df in sorted_syms:
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Filter date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(df) >= 252:  # At least 1 year
            data[symbol] = df

    logger.info(f"Using {len(data)} symbols with data from 2020-2025")

    # Parameter grid - focused on promising ranges
    if quick:
        param_grid = {
            'formation_period': [63, 126, 189],
            'skip_period': [10, 21],
            'vol_lookback': [10, 21],
            'target_vol': [0.15, 0.20],
            'top_percentile': [0.15, 0.20, 0.30]
        }
    else:
        param_grid = {
            'formation_period': [42, 63, 84, 126, 168, 189, 252],
            'skip_period': [5, 10, 15, 21, 42],
            'vol_lookback': [7, 10, 14, 21, 30],
            'target_vol': [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
            'top_percentile': [0.10, 0.15, 0.20, 0.25, 0.30]
        }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    total = len(combinations)

    logger.info(f"Testing {total} parameter combinations...")

    results = []

    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))

        if i % 10 == 0:
            logger.info(f"Progress: {i}/{total} ({100*i/total:.0f}%)")

        result = run_single_backtest(params, data)
        if result.get('sharpe', -999) > -900:
            results.append(result)

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    print("\n" + "=" * 60)
    print("TOP 15 PARAMETER COMBINATIONS BY SHARPE")
    print("=" * 60)

    display_cols = ['formation_period', 'skip_period', 'vol_lookback',
                    'target_vol', 'top_percentile', 'sharpe', 'annual_return',
                    'max_drawdown', 'win_rate', 'total_trades']

    print(df[display_cols].head(15).to_string(index=False))

    # Best parameters
    best = df.iloc[0]
    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS:")
    print("=" * 60)
    print(f"  formation_period: {int(best['formation_period'])}")
    print(f"  skip_period:      {int(best['skip_period'])}")
    print(f"  vol_lookback:     {int(best['vol_lookback'])}")
    print(f"  target_vol:       {best['target_vol']:.2f}")
    print(f"  top_percentile:   {best['top_percentile']:.2f}")
    print()
    print(f"  Sharpe:       {best['sharpe']:.3f}")
    print(f"  Annual Ret:   {best['annual_return']:.1f}%")
    print(f"  Max DD:       {best['max_drawdown']:.1f}%")
    print(f"  Win Rate:     {best['win_rate']:.1f}%")
    print(f"  Trades:       {int(best['total_trades'])}")
    print("=" * 60)

    # Save results
    output_path = Path("research/optimization_results")
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_path / f"vol_momentum_optimization_{timestamp}.csv", index=False)
    logger.info(f"Results saved to {output_path}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer combinations')
    args = parser.parse_args()

    optimize_parameters(quick=args.quick)
