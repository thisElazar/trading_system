"""Trace signal generation through the backtest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from research.discovery.strategy_genome import GenomeFactory
from research.discovery.strategy_compiler import StrategyCompiler, EvolvedStrategy
from research.discovery.config import EvolutionConfig

def main():
    print("=" * 70)
    print("TRACE: Signal generation over backtest period")
    print("=" * 70)

    # Create test data
    np.random.seed(42)
    n_rows = 100
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
    close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
    
    full_data = pd.DataFrame({
        'open': close + np.random.randn(n_rows) * 0.5,
        'high': close + abs(np.random.randn(n_rows)) * 1.5,
        'low': close - abs(np.random.randn(n_rows)) * 1.5,
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n_rows)
    }, index=dates)

    config = EvolutionConfig(population_size=5, max_tree_depth=3)
    factory = GenomeFactory(config)
    compiler = StrategyCompiler(config)

    # Create one genome and compile it
    genome = factory.create_random_genome(generation=0)
    strategy = compiler.compile(genome)

    print(f"\nGenome: {genome.genome_id[:8]}")
    print(f"Entry tree: {str(genome.entry_tree)}")
    print("-" * 70)

    # Simulate backtest loop - check signal at each day from 50 onwards
    signal_days = []
    entry_results = []

    for day in range(50, n_rows):
        # Slice data up to current day (like backtester does)
        current_df = full_data.iloc[:day+1].copy()
        current_data = {'TEST': current_df}
        
        # Generate signals
        signals = strategy.generate_signals(current_data, current_positions=[], vix_regime='normal')
        
        # Also check the entry evaluation directly
        entry_result = strategy._evaluate_entry(current_df)
        entry_results.append(entry_result)
        
        if signals:
            signal_days.append(day)
            print(f"Day {day}: SIGNAL! {signals[0].signal_type.value}")

    print(f"\nSummary over days 50-{n_rows-1}:")
    print(f"  Entry evaluations True: {sum(entry_results)}/{len(entry_results)}")
    print(f"  Signals generated: {len(signal_days)}")
    
    if len(signal_days) == 0 and sum(entry_results) > 0:
        print("\n⚠️  Entry is True but no signals generated!")
        print("   Something between entry evaluation and signal creation is blocking")
    elif sum(entry_results) == 0:
        print("\n⚠️  Entry never evaluates to True on this data")
        print("   Entry tree may have incompatible comparisons")

if __name__ == "__main__":
    main()
