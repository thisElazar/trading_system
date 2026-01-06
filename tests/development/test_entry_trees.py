"""Debug why entry trees aren't producing signals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from research.discovery.strategy_genome import GenomeFactory
from research.discovery.config import EvolutionConfig
from research.discovery.gp_core import set_eval_data, clear_eval_data
from deap import gp

def main():
    print("=" * 70)
    print("DEBUG: Entry tree evaluation")
    print("=" * 70)

    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    test_data = pd.DataFrame({
        'open': close + np.random.randn(100) * 0.5,
        'high': close + abs(np.random.randn(100)) * 1.5,
        'low': close - abs(np.random.randn(100)) * 1.5,
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    config = EvolutionConfig(population_size=20, max_tree_depth=4)
    factory = GenomeFactory(config)

    print("\nGenerating 20 random genomes and evaluating entry conditions...")
    print("-" * 70)

    true_count = 0
    false_count = 0

    for i in range(20):
        genome = factory.create_random_genome(generation=0)
        tree_str = str(genome.entry_tree)

        # Evaluate the entry tree
        try:
            set_eval_data(test_data)
            compiled = gp.compile(genome.entry_tree, factory.bool_pset)
            if callable(compiled):
                result = compiled()
            else:
                result = compiled
            clear_eval_data()

            result_bool = bool(result) if result is not None else False
            
            if result_bool:
                true_count += 1
                marker = "✓ TRUE"
            else:
                false_count += 1
                marker = "✗ FALSE"

            print(f"\n{i:2d}. {marker}")
            print(f"    Tree: {tree_str[:70]}...")
            
            # Show what values the tree is comparing
            if 'gt(' in tree_str or 'lt(' in tree_str or 'ge(' in tree_str or 'le(' in tree_str:
                # Extract comparison operands
                print(f"    (Comparing mixed types - may cause always-false)")

        except Exception as e:
            print(f"\n{i:2d}. ERROR: {e}")
            print(f"    Tree: {tree_str[:70]}...")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {true_count} TRUE, {false_count} FALSE out of 20 genomes")
    print(f"True rate: {true_count/20*100:.0f}%")
    
    if true_count < 5:
        print("\n⚠️  PROBLEM: Entry conditions rarely evaluate to True")
        print("   Most likely cause: comparing incompatible value ranges")
        print("   (e.g., returns ~0.01 vs prices ~100)")

if __name__ == "__main__":
    main()
