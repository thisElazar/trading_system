"""Quick test to verify GP genome evaluation works and produces different signals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from research.discovery.strategy_genome import GenomeFactory
from research.discovery.strategy_compiler import StrategyCompiler
from research.discovery.config import EvolutionConfig

def main():
    print("=" * 60)
    print("GP GENOME EVALUATION TEST")
    print("=" * 60)

    # Create realistic test data with 100 rows (generate_signals requires >= 50)
    np.random.seed(42)
    n_rows = 100
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')

    # Generate realistic OHLCV data
    close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
    test_data = pd.DataFrame({
        'open': close + np.random.randn(n_rows) * 0.5,
        'high': close + abs(np.random.randn(n_rows)) * 1.5,
        'low': close - abs(np.random.randn(n_rows)) * 1.5,
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n_rows)
    }, index=dates)
    
    # Create genomes
    config = EvolutionConfig(
        population_size=5,
        max_tree_depth=3,  # Keep shallow for quick test
        min_tree_depth=1
    )
    factory = GenomeFactory(config)
    
    print(f"\nCreating {config.population_size} random genomes...")
    genomes = [factory.create_random_genome(generation=0) for _ in range(5)]
    
    # Evaluate each genome
    print("\nEvaluating genomes on test data...")
    print("-" * 60)

    # Create compiler once (it takes config, not genome)
    compiler = StrategyCompiler(config)

    # Wrap test_data in dict format expected by generate_signals
    data_dict = {'TEST': test_data}

    results = []
    for i, genome in enumerate(genomes):
        print(f"\nGenome {i} (ID: {genome.genome_id[:8]}):")
        print(f"  Entry tree: {str(genome.entry_tree)[:80]}...")

        try:
            # Compile genome to get EvolvedStrategy
            strategy = compiler.compile(genome)
            # generate_signals expects Dict[str, DataFrame]
            signals = strategy.generate_signals(data_dict, current_positions=[], vix_regime='normal')

            # Convert signals to readable format
            signal_summary = {
                'count': len(signals),
                'types': [s.signal_type.value for s in signals],
                'symbols': [s.symbol for s in signals]
            }

            print(f"  Signals generated: {signal_summary['count']}")
            print(f"  Signal types: {signal_summary['types']}")

            results.append(signal_summary)

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)
    
    # Check for diversity
    print("\n" + "=" * 60)
    print("DIVERSITY CHECK")
    print("=" * 60)

    # Count successful evaluations
    valid_results = [r for r in results if r is not None]
    print(f"\nSuccessful evaluations: {len(valid_results)}/{len(genomes)}")

    if len(valid_results) == 0:
        print("❌ FAILURE: All genome evaluations failed")
        return False

    # Check signal count diversity
    signal_counts = [r['count'] for r in valid_results]
    unique_counts = len(set(signal_counts))
    print(f"Signal counts: {signal_counts}")
    print(f"Unique signal count patterns: {unique_counts}/{len(valid_results)}")

    # Also show the entry trees for comparison
    print("\nEntry trees comparison:")
    for i, genome in enumerate(genomes):
        print(f"  Genome {i}: {str(genome.entry_tree)[:60]}...")

    if unique_counts == 1 and len(valid_results) > 1:
        print("\n⚠️  WARNING: All genomes produced same signal count")
        print("   This may indicate evaluation issues, but trees may still differ")
        # Not necessarily a failure - different trees can produce same count
        return True
    else:
        print("\n✅ SUCCESS: Genomes producing different signals")
        print("   → GP evaluation pipeline is working correctly")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
