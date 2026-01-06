#!/usr/bin/env python3
"""
Strategy Profiler
=================
Profiles each strategy to understand backtest timing characteristics.
Helps optimize for weekend research runs.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from data.cached_data_manager import CachedDataManager
from research.backtester import Backtester
from run_nightly_research import create_strategy_instance


def enrich_data(data: dict) -> dict:
    """Pre-compute common indicators for all symbols (same as research engine)."""
    print("  Pre-computing indicators...", end=' ', flush=True)
    start = time.time()

    for symbol, df in data.items():
        if len(df) < 20:
            continue

        # ATR (14-period) for volatility-based strategies
        if 'atr' not in df.columns:
            high = df['high']
            low = df['low']
            close = df['close']
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()

        # Relative volume (20-day) for breakout strategies
        if 'relative_volume' not in df.columns and 'volume' in df.columns:
            avg_vol = df['volume'].rolling(20).mean()
            df['relative_volume'] = df['volume'] / avg_vol.replace(0, np.nan)

        # Rolling stats for mean reversion
        if 'rolling_mean_20' not in df.columns:
            df['rolling_mean_20'] = df['close'].rolling(20).mean()
            df['rolling_std_20'] = df['close'].rolling(20).std()

        # Log returns for pairs trading
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    elapsed = time.time() - start
    print(f"done ({elapsed:.1f}s)")
    return data

# Strategies to profile
STRATEGIES = [
    'vol_managed_momentum',
    'factor_momentum',
    'pairs_trading',
    'relative_volume_breakout',
    'vix_regime_rotation',
    'sector_rotation',
    'mean_reversion',
]

def profile_strategy(strategy_name: str, data: dict, num_symbols: int, years: float) -> dict:
    """Profile a single strategy backtest."""

    # Create strategy with default params
    try:
        strategy = create_strategy_instance(strategy_name, {})
    except Exception as e:
        return {'strategy': strategy_name, 'error': str(e)}

    backtester = Backtester(initial_capital=100000)

    start = time.time()
    try:
        result = backtester.run(strategy=strategy, data=data)
        elapsed = time.time() - start

        return {
            'strategy': strategy_name,
            'symbols': num_symbols,
            'years': years,
            'elapsed_sec': round(elapsed, 1),
            'trades': result.total_trades,
            'sharpe': round(result.sharpe_ratio, 2),
            'per_symbol_sec': round(elapsed / num_symbols, 2) if num_symbols > 0 else 0,
            'per_year_sec': round(elapsed / years, 2) if years > 0 else 0,
        }
    except Exception as e:
        return {
            'strategy': strategy_name,
            'error': str(e),
            'elapsed_sec': time.time() - start
        }


def main():
    print("=" * 70)
    print("STRATEGY PROFILER")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data_mgr = CachedDataManager()
    if not data_mgr.cache:
        data_mgr.load_all()

    metadata = data_mgr.get_all_metadata()
    all_symbols = list(metadata.keys())
    print(f"Total symbols available: {len(all_symbols)}")

    # Test configurations
    configs = [
        {'symbols': 10, 'days': 252},      # 10 symbols, 1 year
        {'symbols': 50, 'days': 252},      # 50 symbols, 1 year
        {'symbols': 50, 'days': 756},      # 50 symbols, 3 years
        {'symbols': 100, 'days': 1260},    # 100 symbols, 5 years
    ]

    results = []

    for config in configs:
        num_symbols = config['symbols']
        num_days = config['days']
        years = num_days / 252

        print(f"\n{'=' * 70}")
        print(f"Testing: {num_symbols} symbols × {years:.1f} years ({num_days} days)")
        print("=" * 70)

        # Prepare data subset
        test_symbols = all_symbols[:num_symbols]
        test_data = {}
        for sym in test_symbols:
            df = data_mgr.get_bars(sym)
            if df is not None and len(df) >= num_days:
                test_data[sym] = df.tail(num_days).copy()

        actual_symbols = len(test_data)
        print(f"Loaded {actual_symbols} symbols with sufficient data")

        # Pre-compute indicators (matching research engine behavior)
        test_data = enrich_data(test_data)

        for strategy_name in STRATEGIES:
            print(f"  {strategy_name}...", end=' ', flush=True)
            result = profile_strategy(strategy_name, test_data, actual_symbols, years)
            result['config'] = f"{num_symbols}sym_{years:.0f}yr"
            results.append(result)

            if 'error' in result:
                print(f"ERROR: {result['error'][:50]}")
            else:
                print(f"{result['elapsed_sec']}s ({result['trades']} trades, Sharpe {result['sharpe']})")

    # Summary
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<25} {'10sym/1yr':>10} {'50sym/1yr':>10} {'50sym/3yr':>10} {'100sym/5yr':>10}")
    print("-" * 70)

    for strategy in STRATEGIES:
        row = [strategy]
        for config in configs:
            years = config['days'] / 252
            key = f"{config['symbols']}sym_{years:.0f}yr"
            matching = [r for r in results if r['strategy'] == strategy and r.get('config') == key]
            if matching and 'elapsed_sec' in matching[0]:
                row.append(f"{matching[0]['elapsed_sec']}s")
            else:
                row.append("ERROR")
        print(f"{row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")

    # Estimate weekend research time
    print("\n" + "=" * 70)
    print("WEEKEND RESEARCH ESTIMATES (100 symbols, 5 years, 10 individuals × 2 gens)")
    print("=" * 70)

    for strategy in STRATEGIES:
        matching = [r for r in results if r['strategy'] == strategy and '100sym_5yr' in r.get('config', '')]
        if matching and 'elapsed_sec' in matching[0]:
            per_backtest = matching[0]['elapsed_sec']
            # Walk-forward = 2 backtests per individual (train + test)
            # 10 individuals × 2 generations × 2 backtests = 40 backtests
            total_time = per_backtest * 40
            hours = total_time / 3600
            print(f"  {strategy:<25}: {per_backtest:>6.1f}s/backtest → {hours:>5.1f} hours total")
        else:
            print(f"  {strategy:<25}: ERROR")

    return results


if __name__ == "__main__":
    main()
