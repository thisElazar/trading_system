"""
10-Minute Anti-Convergence Test
===============================
Tests the full GA system with:
- Genotype diversity tracking
- Auto-intervention for diversity
- MAP-Elites grid
- Island model with migration
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
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

from data.unified_data_loader import UnifiedDataLoader
from research.discovery.island_model import IslandEvolutionEngine
from research.discovery.config import EvolutionConfig, IslandConfig


def load_real_data(n_symbols=15, min_history_days=400):
    """Load real market data for testing."""
    loader = UnifiedDataLoader()

    all_symbols = loader.get_available_daily_symbols()
    logger.info(f"Found {len(all_symbols)} symbols")

    priority_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'JPM', 'BAC', 'GS', 'WFC',
        'JNJ', 'PFE', 'UNH', 'ABBV',
        'XOM', 'CVX', 'COP',
        'WMT', 'COST', 'TGT',
        'DIS', 'NFLX', 'CMCSA',
    ]

    available = [s for s in priority_symbols if s in all_symbols]
    logger.info(f"Using {len(available[:n_symbols])} priority symbols")

    data = {}
    for symbol in available[:n_symbols]:
        try:
            df = loader.load_daily(symbol)
            if df is not None and len(df) >= min_history_days:
                df = df.tail(504)  # ~2 years
                data[symbol] = df
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    logger.info(f"Loaded {len(data)} symbols with sufficient history")
    return data


def main():
    print("=" * 70)
    print("10-MINUTE ANTI-CONVERGENCE TEST")
    print("=" * 70)

    # Load data
    print("\nLoading real market data...")
    start_load = time.time()
    data = load_real_data(n_symbols=12, min_history_days=400)
    load_time = time.time() - start_load
    print(f"Loaded {len(data)} symbols in {load_time:.1f}s")

    if len(data) < 3:
        print("ERROR: Not enough data loaded!")
        return None

    # Configure evolution with anti-convergence features
    config = EvolutionConfig(
        population_size=15,
        generations_per_session=1000,
        max_tree_depth=5,
        min_tree_depth=2,
        mutation_rate=0.25,
        crossover_rate=0.75,
        min_trades=20,
    )

    island_config = IslandConfig(
        num_islands=4,
        population_per_island=15,  # 4 x 15 = 60 total
        migration_interval=5,
        migration_rate=0.2,
        topology="ring",
        vary_mutation_rate=True,
        vary_tree_depth=True
    )

    # Create engine with NEW features explicitly enabled
    print(f"\nInitializing Evolution Engine...")
    print(f"  Islands: {island_config.num_islands}")
    print(f"  Population per island: {island_config.population_per_island}")
    print(f"  Total population: {island_config.num_islands * island_config.population_per_island}")

    engine = IslandEvolutionEngine(
        config=config,
        island_config=island_config,
        use_fast_backtester=False,
        enable_diversity_monitor=True,  # NEW
        enable_map_elites=True          # NEW
    )
    engine.load_data(data=data)
    engine.initialize_populations()

    print(f"\nAnti-Convergence Features:")
    print(f"  Diversity Monitor: {engine.diversity_monitor is not None}")
    print(f"  MAP-Elites Grid: {engine.map_elites is not None}")
    if engine.diversity_monitor:
        th = engine.diversity_monitor.thresholds
        print(f"  Intervention thresholds:")
        print(f"    - Min genotype diversity: {th.min_genotype_diversity}")
        print(f"    - Min phenotype diversity: {th.min_phenotype_diversity}")
        print(f"    - Stagnation generations: {th.stagnation_generations}")

    # Run for 10 minutes
    test_duration = 10 * 60  # 10 minutes
    print(f"\n{'='*70}")
    print(f"RUNNING EVOLUTION FOR 10 MINUTES")
    print("=" * 70)

    stats_history = []
    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < test_duration:
        engine.evolve_generation()

        elapsed = time.time() - start_time

        # Collect stats
        all_sortinos = []
        all_trades = []
        for island in engine.islands:
            for genome in island.population:
                if genome.genome_id in island.fitness_cache:
                    fitness = island.fitness_cache[genome.genome_id]
                    all_sortinos.append(fitness.sortino)
                    all_trades.append(fitness.trades)

        diversities = [island.diversity for island in engine.islands]

        # Get genotype diversity from monitor
        geno_div = 0.0
        if engine.diversity_monitor and engine.diversity_monitor.get_latest_metrics():
            geno_div = engine.diversity_monitor.get_latest_metrics().genotype_metrics.genotype_diversity

        # Get MAP-Elites coverage
        map_coverage = engine.map_elites.get_coverage() if engine.map_elites else 0.0

        stats = {
            'generation': engine.current_generation,
            'elapsed_sec': elapsed,
            'min_sortino': min(all_sortinos) if all_sortinos else -999,
            'max_sortino': max(all_sortinos) if all_sortinos else -999,
            'avg_sortino': np.mean(all_sortinos) if all_sortinos else -999,
            'avg_trades': np.mean(all_trades) if all_trades else 0,
            'avg_diversity': np.mean(diversities),
            'min_diversity': min(diversities),
            'max_diversity': max(diversities),
            'genotype_diversity': geno_div,
            'map_coverage': map_coverage,
            'pareto_size': len(engine.global_pareto_front),
            'interventions': engine.diversity_monitor.total_interventions if engine.diversity_monitor else 0
        }
        stats_history.append(stats)

        # Log every 30 seconds or first few generations
        if (time.time() - last_log_time >= 30) or engine.current_generation <= 3:
            last_log_time = time.time()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"\n[{mins:02d}:{secs:02d}] Generation {engine.current_generation}")
            print(f"  Sortino: [{stats['min_sortino']:.2f}, {stats['max_sortino']:.2f}] avg={stats['avg_sortino']:.2f}")
            print(f"  Phenotype Diversity: [{stats['min_diversity']:.3f}, {stats['max_diversity']:.3f}] avg={stats['avg_diversity']:.3f}")
            print(f"  Genotype Diversity: {stats['genotype_diversity']:.3f}")
            print(f"  MAP-Elites Coverage: {stats['map_coverage']:.1%} ({len(engine.map_elites.grid)} cells)")
            print(f"  Pareto front: {stats['pareto_size']} strategies")
            print(f"  Diversity interventions: {stats['interventions']}")

    total_time = time.time() - start_time

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("=" * 70)

    final_stats = stats_history[-1] if stats_history else {}

    print(f"\nEvolution Summary:")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Generations completed: {engine.current_generation}")
    print(f"  Strategies evaluated: {engine.total_strategies_evaluated}")
    print(f"  Generations per minute: {engine.current_generation / (total_time/60):.1f}")

    print(f"\nAnti-Convergence Metrics:")
    print(f"  Final Genotype Diversity: {final_stats.get('genotype_diversity', 0):.3f}")
    print(f"  Final Phenotype Diversity: {final_stats.get('avg_diversity', 0):.3f}")
    print(f"  MAP-Elites Coverage: {final_stats.get('map_coverage', 0):.1%}")
    print(f"  Total Interventions: {final_stats.get('interventions', 0)}")

    print(f"\nFitness Metrics:")
    print(f"  Best Sortino: {final_stats.get('max_sortino', 0):.3f}")
    print(f"  Avg Sortino: {final_stats.get('avg_sortino', 0):.3f}")
    print(f"  Pareto Front Size: {len(engine.global_pareto_front)}")

    # Show diversity progression
    print(f"\n{'='*70}")
    print("DIVERSITY PROGRESSION")
    print("=" * 70)

    checkpoints = [1, 5, 10, 25, 50, 100]
    print(f"\n{'Gen':>6} {'Time':>8} {'Geno Div':>10} {'Pheno Div':>10} {'MAP Cov':>10} {'Intv':>6}")
    print("-" * 60)

    for gen in checkpoints:
        matching = [s for s in stats_history if s['generation'] == gen]
        if matching:
            s = matching[0]
            mins = int(s['elapsed_sec'] // 60)
            secs = int(s['elapsed_sec'] % 60)
            print(f"{gen:>6} {mins:02d}:{secs:02d} {s['genotype_diversity']:>10.3f} {s['avg_diversity']:>10.3f} "
                  f"{s['map_coverage']:>9.1%} {s['interventions']:>6}")

    # Final row
    if stats_history:
        s = stats_history[-1]
        mins = int(s['elapsed_sec'] // 60)
        secs = int(s['elapsed_sec'] % 60)
        print(f"{s['generation']:>6} {mins:02d}:{secs:02d} {s['genotype_diversity']:>10.3f} {s['avg_diversity']:>10.3f} "
              f"{s['map_coverage']:>9.1%} {s['interventions']:>6}")

    # Show MAP-Elites quality
    if engine.map_elites and len(engine.map_elites.grid) > 0:
        print(f"\n{'='*70}")
        print("MAP-ELITES QUALITY")
        print("=" * 70)
        quality = engine.map_elites.get_quality_stats()
        print(f"  Elites: {quality['count']}")
        print(f"  Fitness range: [{quality['min']:.3f}, {quality['max']:.3f}]")
        print(f"  Mean fitness: {quality['mean']:.3f}")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print("=" * 70)

    return stats_history, engine


if __name__ == "__main__":
    main()
