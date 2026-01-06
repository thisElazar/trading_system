"""
15-Minute Island Model Test with Real Market Data
==================================================
Tests the island evolution engine on actual market data.
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


def load_real_data(n_symbols=20, min_history_days=500):
    """Load real market data for testing."""
    loader = UnifiedDataLoader()

    # Get available symbols
    all_symbols = loader.get_available_daily_symbols()
    logger.info(f"Found {len(all_symbols)} symbols")

    # Select diverse symbols - mix of large caps and sectors
    priority_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
        'JPM', 'BAC', 'GS', 'WFC',                 # Finance
        'JNJ', 'PFE', 'UNH', 'ABBV',               # Healthcare
        'XOM', 'CVX', 'COP',                       # Energy
        'WMT', 'COST', 'TGT',                      # Retail
        'DIS', 'NFLX', 'CMCSA',                    # Media
        'CAT', 'DE', 'MMM',                        # Industrial
        'PG', 'KO', 'PEP'                          # Consumer
    ]

    # Filter to available symbols
    available_priority = [s for s in priority_symbols if s in all_symbols]
    logger.info(f"Using {len(available_priority[:n_symbols])} priority symbols")

    data = {}
    for symbol in available_priority[:n_symbols]:
        try:
            df = loader.load_daily(symbol)
            if df is not None and len(df) >= min_history_days:
                # Use last 2 years of data for faster backtesting
                df = df.tail(504)  # ~2 years of trading days
                data[symbol] = df
                logger.debug(f"Loaded {symbol}: {len(df)} days")
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    logger.info(f"Loaded {len(data)} symbols with sufficient history")
    return data


def main():
    print("=" * 70)
    print("15-MINUTE ISLAND MODEL TEST WITH REAL DATA")
    print("=" * 70)

    # Load real market data
    print("\nLoading real market data...")
    start_load = time.time()
    data = load_real_data(n_symbols=15, min_history_days=400)
    load_time = time.time() - start_load
    print(f"Loaded {len(data)} symbols in {load_time:.1f}s")

    for symbol, df in list(data.items())[:5]:
        try:
            start = pd.Timestamp(df.index.min()).date()
            end = pd.Timestamp(df.index.max()).date()
            print(f"  {symbol}: {len(df)} days, {start} to {end}")
        except Exception:
            print(f"  {symbol}: {len(df)} days")

    # Configure island evolution
    config = EvolutionConfig(
        population_size=15,
        generations_per_session=1000,
        max_tree_depth=5,
        min_tree_depth=2,
        mutation_rate=0.25,
        crossover_rate=0.75,
        min_trades=20,  # Slightly relaxed for real data
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

    # Create engine
    print(f"\nInitializing Island Evolution Engine...")
    print(f"  Islands: {island_config.num_islands}")
    print(f"  Population per island: {island_config.population_per_island}")
    print(f"  Total population: {island_config.num_islands * island_config.population_per_island}")
    print(f"  Migration every {island_config.migration_interval} generations")

    engine = IslandEvolutionEngine(
        config=config,
        island_config=island_config,
        use_fast_backtester=False
    )
    engine.load_data(data=data)
    engine.initialize_populations()

    # Show island configuration
    print("\nIsland Configuration:")
    for island in engine.islands:
        print(f"  Island {island.island_id}: mutation={island.mutation_rate:.2f}, max_depth={island.max_tree_depth}")

    # Run for 15 minutes
    test_duration = 15 * 60  # 15 minutes in seconds
    print(f"\n{'='*70}")
    print(f"RUNNING EVOLUTION FOR 15 MINUTES")
    print("=" * 70)

    stats_history = []
    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < test_duration:
        engine.evolve_generation()

        # Collect stats every generation
        elapsed = time.time() - start_time

        all_sortinos = []
        all_trades = []
        for island in engine.islands:
            for genome in island.population:
                if genome.genome_id in island.fitness_cache:
                    fitness = island.fitness_cache[genome.genome_id]
                    all_sortinos.append(fitness.sortino)
                    all_trades.append(fitness.trades)

        diversities = [island.diversity for island in engine.islands]

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
            'pareto_size': len(engine.global_pareto_front)
        }
        stats_history.append(stats)

        # Log every 30 seconds or first few generations
        if (time.time() - last_log_time >= 30) or engine.current_generation <= 3:
            last_log_time = time.time()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"\n[{mins:02d}:{secs:02d}] Generation {engine.current_generation}")
            print(f"  Sortino: [{stats['min_sortino']:.2f}, {stats['max_sortino']:.2f}] avg={stats['avg_sortino']:.2f}")
            print(f"  Trades: avg={stats['avg_trades']:.0f}")
            print(f"  Diversity: [{stats['min_diversity']:.3f}, {stats['max_diversity']:.3f}] avg={stats['avg_diversity']:.3f}")
            print(f"  Pareto front: {stats['pareto_size']} strategies")

            # Show best per island
            for island in engine.islands:
                if island.best_fitness > -999:
                    print(f"    Island {island.island_id}: best={island.best_fitness:.2f}, div={island.diversity:.3f}")

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

    print(f"\nFinal Metrics:")
    print(f"  Best Sortino: {final_stats.get('max_sortino', 0):.3f}")
    print(f"  Avg Sortino: {final_stats.get('avg_sortino', 0):.3f}")
    print(f"  Avg Trades: {final_stats.get('avg_trades', 0):.0f}")
    print(f"  Final Diversity: {final_stats.get('avg_diversity', 0):.3f}")
    print(f"  Pareto Front Size: {len(engine.global_pareto_front)}")

    # Show progression
    print(f"\n{'='*70}")
    print("EVOLUTION PROGRESSION")
    print("=" * 70)

    checkpoints = [1, 5, 10, 25, 50, 100]
    print(f"\n{'Gen':>6} {'Time':>8} {'Avg Sort':>10} {'Max Sort':>10} {'Diversity':>10} {'Trades':>8}")
    print("-" * 60)

    for gen in checkpoints:
        matching = [s for s in stats_history if s['generation'] == gen]
        if matching:
            s = matching[0]
            mins = int(s['elapsed_sec'] // 60)
            secs = int(s['elapsed_sec'] % 60)
            print(f"{gen:>6} {mins:02d}:{secs:02d} {s['avg_sortino']:>10.3f} {s['max_sortino']:>10.3f} "
                  f"{s['avg_diversity']:>10.3f} {s['avg_trades']:>8.0f}")

    # Show final population samples from each island
    print(f"\n{'='*70}")
    print("BEST STRATEGIES PER ISLAND")
    print("=" * 70)

    for island in engine.islands:
        print(f"\nIsland {island.island_id} (mutation={island.mutation_rate:.2f}, depth={island.max_tree_depth}):")
        print(f"  Best fitness: {island.best_fitness:.3f}")
        print(f"  Diversity: {island.diversity:.3f}")

        # Find best genome
        best_genome = None
        best_fitness = -float('inf')
        for genome in island.population:
            if genome.genome_id in island.fitness_cache:
                fitness = island.fitness_cache[genome.genome_id]
                if fitness.sortino > best_fitness:
                    best_fitness = fitness.sortino
                    best_genome = genome

        if best_genome:
            print(f"  Entry tree: {str(best_genome.entry_tree)[:70]}...")
            if best_genome.genome_id in island.fitness_cache:
                f = island.fitness_cache[best_genome.genome_id]
                print(f"  Trades: {f.trades}, Win rate: {f.win_rate:.1f}%")

    # Show global Pareto front
    if engine.global_pareto_front:
        print(f"\n{'='*70}")
        print(f"GLOBAL PARETO FRONT ({len(engine.global_pareto_front)} strategies)")
        print("=" * 70)

        # Get fitness for pareto members
        pareto_with_fitness = []
        for genome in engine.global_pareto_front[:10]:  # Show top 10
            for island in engine.islands:
                if genome.genome_id in island.fitness_cache:
                    pareto_with_fitness.append((genome, island.fitness_cache[genome.genome_id]))
                    break

        pareto_with_fitness.sort(key=lambda x: x[1].sortino, reverse=True)

        for i, (genome, fitness) in enumerate(pareto_with_fitness[:5]):
            print(f"\n  #{i+1}: Sortino={fitness.sortino:.3f}, Trades={fitness.trades}, WinRate={fitness.win_rate:.1f}%")
            print(f"      Entry: {str(genome.entry_tree)[:60]}...")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print("=" * 70)

    return stats_history, engine


if __name__ == "__main__":
    main()
