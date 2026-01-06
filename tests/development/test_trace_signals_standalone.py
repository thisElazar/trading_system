"""Standalone trace of signal generation - no db dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import only what we need, avoiding db_manager
from research.discovery.config import EvolutionConfig
from research.discovery.strategy_genome import GenomeFactory
from research.discovery.gp_core import set_eval_data, clear_eval_data
from deap import gp

def main():
    print("=" * 70)
    print("DEBUGGING: Why entry trees don't produce signals")
    print("=" * 70)

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

    genome = factory.create_random_genome(generation=0)
    print(f"\nGenome ID: {genome.genome_id[:8]}")
    print(f"Entry tree: {genome.entry_tree}")
    print("-" * 70)

    # Test entry at multiple days
    true_days = []
    false_days = []

    for day in range(50, n_rows):
        df = full_data.iloc[:day+1].copy()
        try:
            set_eval_data(df)
            result = gp.compile(genome.entry_tree, factory.bool_pset)
            if callable(result):
                entry = result()
            else:
                entry = result
            clear_eval_data()

            if entry:
                true_days.append(day)
            else:
                false_days.append(day)
        except Exception as e:
            print(f"Day {day}: ERROR - {e}")

    print(f"\nEntry evaluation over days 50-99:")
    print(f"  TRUE:  {len(true_days)} days ({len(true_days)/50*100:.0f}%)")
    print(f"  FALSE: {len(false_days)} days ({len(false_days)/50*100:.0f}%)")
    
    if len(true_days) > 0:
        print(f"  First True day: {true_days[0]}")
    
    # Now test 10 different genomes
    print("\n" + "=" * 70)
    print("Testing 10 different genomes (day 99 only):")
    print("-" * 70)
    
    for i in range(10):
        genome = factory.create_random_genome(generation=0)
        try:
            set_eval_data(full_data)
            result = gp.compile(genome.entry_tree, factory.bool_pset)
            if callable(result):
                entry = result()
            else:
                entry = result
            clear_eval_data()
            
            status = "TRUE" if entry else "FALSE"
            print(f"{i}: {status:5} | {str(genome.entry_tree)[:55]}...")
        except Exception as e:
            print(f"{i}: ERROR | {e}")

if __name__ == "__main__":
    main()
