"""Test evolution engine for 60 seconds to observe convergence behavior."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s | %(levelname)s | %(message)s'
)

from research.discovery.evolution_engine import EvolutionEngine
from research.discovery.config import EvolutionConfig

def main():
    print("=" * 70)
    print("EVOLUTION ENGINE TEST - 60 SECONDS")
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

    # Config for longer run
    config = EvolutionConfig(
        population_size=20,
        generations_per_session=1000,  # Will be limited by time
        checkpoint_frequency=100,
        max_tree_depth=5,
        min_tree_depth=2,
        mutation_rate=0.3,
        crossover_rate=0.7
    )

    print(f"\nConfig: population={config.population_size}")
    print("-" * 70)

    engine = EvolutionEngine(config=config, use_fast_backtester=False)
    engine.load_data(data=test_data)
    engine.initialize_population()

    # Track stats every 10 generations
    generation_stats = []
    start_time = time.time()
    max_runtime = 60  # seconds

    print(f"\nRunning for {max_runtime} seconds...")
    print(f"{'Gen':>5} {'Time':>6} {'Sortino':>20} {'Diversity':>10} {'Pareto':>7}")
    print("-" * 55)

    while time.time() - start_time < max_runtime:
        engine.evolve_generation()

        # Get current stats
        fitnesses = [
            engine.fitness_cache.get(g.genome_id)
            for g in engine.population
            if g.genome_id in engine.fitness_cache
        ]
        sortinos = [f.sortino for f in fitnesses if f]

        # Calculate diversity from behavior cache
        behaviors = [
            engine.behavior_cache.get(g.genome_id)
            for g in engine.population
            if g.genome_id in engine.behavior_cache
        ]
        
        from research.discovery.novelty_search import calculate_population_diversity
        diversity = calculate_population_diversity([b for b in behaviors if b])

        stats = {
            'generation': engine.current_generation,
            'time': time.time() - start_time,
            'min_sortino': min(sortinos) if sortinos else 0,
            'max_sortino': max(sortinos) if sortinos else 0,
            'avg_sortino': np.mean(sortinos) if sortinos else 0,
            'diversity': diversity,
            'pareto_size': len(engine.pareto_front)
        }
        generation_stats.append(stats)

        # Print every 10 generations
        if engine.current_generation % 10 == 0 or engine.current_generation <= 5:
            print(f"{stats['generation']:>5} {stats['time']:>5.1f}s "
                  f"[{stats['min_sortino']:>5.2f}, {stats['max_sortino']:>5.2f}] avg={stats['avg_sortino']:>5.2f} "
                  f"{stats['diversity']:>10.3f} {stats['pareto_size']:>7}")

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal generations: {engine.current_generation}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Strategies evaluated: {engine.total_strategies_evaluated}")
    print(f"Generations per second: {engine.current_generation / total_time:.1f}")

    # Show convergence pattern
    print(f"\nConvergence analysis:")
    checkpoints = [1, 5, 10, 20, 50, 100, 200, 500]
    for cp in checkpoints:
        matching = [s for s in generation_stats if s['generation'] == cp]
        if matching:
            s = matching[0]
            print(f"  Gen {cp:>3}: avg_sortino={s['avg_sortino']:.3f}, diversity={s['diversity']:.3f}")

    # Final population diversity
    print(f"\nFinal population tree samples:")
    for i, genome in enumerate(engine.population[:5]):
        tree_str = str(genome.entry_tree)[:50]
        print(f"  {i}: {tree_str}...")

    # Check if we have diversity issues
    final_stats = generation_stats[-1] if generation_stats else None
    if final_stats:
        if final_stats['diversity'] < 0.1:
            print(f"\n⚠️  LOW DIVERSITY: {final_stats['diversity']:.3f}")
            print("   Population may have converged prematurely")
        else:
            print(f"\n✅ Diversity maintained: {final_stats['diversity']:.3f}")

if __name__ == "__main__":
    main()
