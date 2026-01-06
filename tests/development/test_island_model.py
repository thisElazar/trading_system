"""
Test Island Model vs Regular Evolution
======================================
Compares diversity and fitness progression between:
1. Regular single-population evolution
2. Island-based parallel evolution
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s | %(levelname)s | %(message)s'
)

from research.discovery.evolution_engine import EvolutionEngine
from research.discovery.island_model import IslandEvolutionEngine
from research.discovery.config import EvolutionConfig, IslandConfig
from research.discovery.novelty_search import calculate_population_diversity


def create_test_data(n_rows=200, n_symbols=5):
    """Create synthetic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][:n_symbols]
    data = {}

    for symbol in symbols:
        close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
        data[symbol] = pd.DataFrame({
            'open': close + np.random.randn(n_rows) * 0.5,
            'high': close + abs(np.random.randn(n_rows)) * 1.5,
            'low': close - abs(np.random.randn(n_rows)) * 1.5,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n_rows)
        }, index=dates)

    return data


def run_regular_evolution(data, max_time_seconds=30):
    """Run regular single-population evolution."""
    config = EvolutionConfig(
        population_size=40,  # Same total as 4 islands x 10 each
        generations_per_session=1000,
        checkpoint_frequency=100,
        max_tree_depth=5,
        min_tree_depth=2
    )

    engine = EvolutionEngine(config=config, use_fast_backtester=False)
    engine.load_data(data=data)
    engine.initialize_population()

    stats = []
    start_time = time.time()

    while time.time() - start_time < max_time_seconds:
        engine.evolve_generation()

        # Collect stats
        fitnesses = [
            engine.fitness_cache.get(g.genome_id)
            for g in engine.population
            if g.genome_id in engine.fitness_cache
        ]
        sortinos = [f.sortino for f in fitnesses if f]

        behaviors = [
            engine.behavior_cache.get(g.genome_id)
            for g in engine.population
            if g.genome_id in engine.behavior_cache
        ]
        diversity = calculate_population_diversity([b for b in behaviors if b])

        stats.append({
            'generation': engine.current_generation,
            'time': time.time() - start_time,
            'min_sortino': min(sortinos) if sortinos else -999,
            'max_sortino': max(sortinos) if sortinos else -999,
            'avg_sortino': np.mean(sortinos) if sortinos else -999,
            'diversity': diversity,
            'unique_trees': len(set(str(g.entry_tree) for g in engine.population))
        })

    return stats, engine


def run_island_evolution(data, max_time_seconds=30):
    """Run island-based parallel evolution."""
    config = EvolutionConfig(
        population_size=10,  # Will be overridden
        generations_per_session=1000,
        checkpoint_frequency=100,
        max_tree_depth=5,
        min_tree_depth=2
    )

    island_config = IslandConfig(
        num_islands=4,
        population_per_island=10,  # 4 x 10 = 40 total (same as regular)
        migration_interval=5,
        migration_rate=0.2,
        topology="ring"
    )

    engine = IslandEvolutionEngine(
        config=config,
        island_config=island_config,
        use_fast_backtester=False
    )
    engine.load_data(data=data)
    engine.initialize_populations()

    stats = []
    start_time = time.time()

    while time.time() - start_time < max_time_seconds:
        engine.evolve_generation()

        # Collect stats across all islands
        all_sortinos = []
        all_trees = set()

        for island in engine.islands:
            for genome in island.population:
                if genome.genome_id in island.fitness_cache:
                    all_sortinos.append(island.fitness_cache[genome.genome_id].sortino)
                all_trees.add(str(genome.entry_tree))

        # Average diversity across islands
        diversities = [island.diversity for island in engine.islands]

        stats.append({
            'generation': engine.current_generation,
            'time': time.time() - start_time,
            'min_sortino': min(all_sortinos) if all_sortinos else -999,
            'max_sortino': max(all_sortinos) if all_sortinos else -999,
            'avg_sortino': np.mean(all_sortinos) if all_sortinos else -999,
            'diversity': np.mean(diversities),
            'min_diversity': min(diversities),
            'max_diversity': max(diversities),
            'unique_trees': len(all_trees)
        })

    return stats, engine


def main():
    print("=" * 70)
    print("ISLAND MODEL vs REGULAR EVOLUTION COMPARISON")
    print("=" * 70)

    # Create test data
    print("\nGenerating test data: 200 days, 5 symbols...")
    data = create_test_data(n_rows=200, n_symbols=5)

    test_duration = 60  # seconds each

    # Run regular evolution
    print(f"\n{'='*70}")
    print("REGULAR EVOLUTION (single population of 40)")
    print("=" * 70)
    print(f"Running for {test_duration} seconds...")

    regular_stats, regular_engine = run_regular_evolution(data, max_time_seconds=test_duration)

    # Run island evolution
    print(f"\n{'='*70}")
    print("ISLAND EVOLUTION (4 islands x 10 = 40 total)")
    print("=" * 70)
    print(f"Running for {test_duration} seconds...")

    island_stats, island_engine = run_island_evolution(data, max_time_seconds=test_duration)

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Final stats
    regular_final = regular_stats[-1] if regular_stats else {}
    island_final = island_stats[-1] if island_stats else {}

    print(f"\n{'Metric':<25} {'Regular':>15} {'Island':>15}")
    print("-" * 55)
    print(f"{'Generations':<25} {regular_final.get('generation', 0):>15} {island_final.get('generation', 0):>15}")
    print(f"{'Final Avg Sortino':<25} {regular_final.get('avg_sortino', 0):>15.3f} {island_final.get('avg_sortino', 0):>15.3f}")
    print(f"{'Final Max Sortino':<25} {regular_final.get('max_sortino', 0):>15.3f} {island_final.get('max_sortino', 0):>15.3f}")
    print(f"{'Final Diversity':<25} {regular_final.get('diversity', 0):>15.3f} {island_final.get('diversity', 0):>15.3f}")
    print(f"{'Unique Trees':<25} {regular_final.get('unique_trees', 0):>15} {island_final.get('unique_trees', 0):>15}")

    # Progression at key points
    print(f"\n{'='*70}")
    print("DIVERSITY PROGRESSION")
    print("=" * 70)

    checkpoints = [1, 5, 10, 20, 50]
    print(f"{'Generation':<12} {'Regular Div':>12} {'Island Div':>12} {'Island Min-Max':>20}")
    print("-" * 60)

    for gen in checkpoints:
        regular_at_gen = [s for s in regular_stats if s['generation'] == gen]
        island_at_gen = [s for s in island_stats if s['generation'] == gen]

        if regular_at_gen and island_at_gen:
            r = regular_at_gen[0]
            i = island_at_gen[0]
            min_max = f"[{i.get('min_diversity', 0):.3f}, {i.get('max_diversity', 0):.3f}]"
            print(f"{gen:<12} {r['diversity']:>12.3f} {i['diversity']:>12.3f} {min_max:>20}")

    # Show sample trees from each
    print(f"\n{'='*70}")
    print("SAMPLE ENTRY TREES (final generation)")
    print("=" * 70)

    print("\nRegular Evolution (5 samples):")
    for i, genome in enumerate(regular_engine.population[:5]):
        print(f"  {i}: {str(genome.entry_tree)[:60]}...")

    print("\nIsland Evolution (2 from each island):")
    for island in island_engine.islands:
        print(f"\n  Island {island.island_id} (mutation={island.mutation_rate:.2f}, depth={island.max_tree_depth}):")
        for genome in island.population[:2]:
            print(f"    {str(genome.entry_tree)[:55]}...")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    regular_div = regular_final.get('diversity', 0)
    island_div = island_final.get('diversity', 0)
    regular_trees = regular_final.get('unique_trees', 0)
    island_trees = island_final.get('unique_trees', 0)

    if island_div > regular_div * 1.5 or island_trees > regular_trees * 1.5:
        print("\n✅ ISLAND MODEL MAINTAINS BETTER DIVERSITY")
        print(f"   Diversity: {island_div:.3f} vs {regular_div:.3f}")
        print(f"   Unique trees: {island_trees} vs {regular_trees}")
    elif island_div > regular_div:
        print("\n✓ Island model shows improvement in diversity")
    else:
        print("\n⚠️  Results inconclusive - may need longer run or parameter tuning")

    return regular_stats, island_stats


if __name__ == "__main__":
    main()
