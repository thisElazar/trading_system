"""Test evolution engine with 3 generations to verify GP evolution is working."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

from research.discovery.evolution_engine import EvolutionEngine
from research.discovery.config import EvolutionConfig

def main():
    print("=" * 70)
    print("EVOLUTION ENGINE TEST - 3 GENERATIONS")
    print("=" * 70)

    # Create test data - 200 rows for proper indicator coverage
    np.random.seed(42)
    n_rows = 200
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')

    print(f"\nGenerating test data: {n_rows} days, 5 symbols...")
    test_data = {}
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
        close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
        test_data[symbol] = pd.DataFrame({
            'open': close + np.random.randn(n_rows) * 0.5,
            'high': close + abs(np.random.randn(n_rows)) * 1.5,
            'low': close - abs(np.random.randn(n_rows)) * 1.5,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n_rows)
        }, index=dates)

    # Small config for quick test
    config = EvolutionConfig(
        population_size=10,
        generations_per_session=3,
        checkpoint_frequency=10,  # Don't checkpoint during test
        max_tree_depth=4,
        min_tree_depth=2
    )

    print(f"\nConfig: population={config.population_size}, generations=3")
    print("-" * 70)

    # Create engine with regular Backtester (FastBacktester has compatibility issues)
    engine = EvolutionEngine(config=config, use_fast_backtester=False)
    engine.load_data(data=test_data)
    engine.initialize_population()

    print(f"Initial population: {len(engine.population)} genomes")

    # Track fitness across generations
    generation_stats = []

    # Run evolution with timing
    total_start = time.time()

    for gen in range(3):
        gen_start = time.time()

        engine.evolve_generation()

        gen_time = time.time() - gen_start

        # Collect stats
        fitnesses = [
            engine.fitness_cache.get(g.genome_id)
            for g in engine.population
            if g.genome_id in engine.fitness_cache
        ]

        sortinos = [f.sortino for f in fitnesses if f]

        stats = {
            'generation': engine.current_generation,
            'time_sec': gen_time,
            'min_sortino': min(sortinos) if sortinos else 0,
            'max_sortino': max(sortinos) if sortinos else 0,
            'avg_sortino': np.mean(sortinos) if sortinos else 0,
            'pareto_size': len(engine.pareto_front),
            'unique_genomes': len(set(g.genome_id for g in engine.population))
        }
        generation_stats.append(stats)

        print(f"\nGen {stats['generation']}: {stats['time_sec']:.1f}s | "
              f"Sortino [min={stats['min_sortino']:.2f}, max={stats['max_sortino']:.2f}, "
              f"avg={stats['avg_sortino']:.2f}] | Pareto={stats['pareto_size']}")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Strategies evaluated: {engine.total_strategies_evaluated}")
    print(f"Time per evaluation: {total_time/engine.total_strategies_evaluated:.2f}s")

    print(f"\nFitness progression:")
    for stats in generation_stats:
        print(f"  Gen {stats['generation']}: avg_sortino={stats['avg_sortino']:.3f}, "
              f"max_sortino={stats['max_sortino']:.3f}")

    # Check if evolution is working
    print("\n" + "=" * 70)
    print("EVOLUTION CHECK")
    print("=" * 70)

    # Check 1: Are we getting different genomes?
    all_genome_ids = set()
    for g in engine.population:
        all_genome_ids.add(g.genome_id)

    print(f"\nUnique genomes in final population: {len(all_genome_ids)}")

    # Check 2: Did fitness change across generations?
    sortino_progression = [s['avg_sortino'] for s in generation_stats]
    fitness_changed = len(set(sortino_progression)) > 1

    print(f"Fitness changed across generations: {fitness_changed}")
    print(f"  Sortino progression: {[f'{s:.3f}' for s in sortino_progression]}")

    # Check 3: Are there diverse entry trees?
    print(f"\nSample entry trees from final population:")
    for i, genome in enumerate(engine.population[:5]):
        tree_str = str(genome.entry_tree)[:60]
        print(f"  {i}: {tree_str}...")

    # Final verdict
    print("\n" + "=" * 70)
    if fitness_changed and len(all_genome_ids) == config.population_size:
        print("✅ SUCCESS: Evolution is working correctly!")
        print("   - Genomes are unique")
        print("   - Fitness values changing across generations")
        return True
    else:
        print("⚠️  WARNING: Evolution may have issues")
        if not fitness_changed:
            print("   - Fitness not changing (may need more generations)")
        if len(all_genome_ids) < config.population_size:
            print(f"   - Only {len(all_genome_ids)} unique genomes (expected {config.population_size})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
