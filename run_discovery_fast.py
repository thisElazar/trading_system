#!/usr/bin/env python3
"""
Fast Discovery Run
==================
Runs GP-based strategy discovery with a subset of symbols for faster iteration.
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger('fast_discovery')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from research.discovery import EvolutionEngine, EvolutionConfig
from data.cached_data_manager import CachedDataManager


def run_fast_discovery(
    n_symbols: int = 100,
    hours: float = 1.0,
    population: int = 20,
    generations: int = 50
):
    """
    Run discovery with a subset of symbols for faster iteration.

    Args:
        n_symbols: Number of top symbols to use
        hours: Maximum runtime in hours
        population: Population size
        generations: Max generations (will stop at time limit)
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("FAST DISCOVERY RUN")
    logger.info(f"Symbols: {n_symbols} | Hours: {hours} | Pop: {population}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading market data...")
    data_manager = CachedDataManager()
    n_loaded = data_manager.load_all()  # Returns count, populates cache
    all_data = data_manager.cache
    # Note: get_vix() returns a float (current value), not historical data
    # Pass None for vix_data since we don't have regime history
    vix_data = None
    logger.info(f"Loaded {n_loaded} symbols total")

    # Select top N symbols by data length (most history = most liquid)
    symbol_lengths = [(sym, len(df)) for sym, df in all_data.items()]
    symbol_lengths.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [sym for sym, _ in symbol_lengths[:n_symbols]]

    # Filter data
    filtered_data = {sym: all_data[sym] for sym in top_symbols}
    logger.info(f"Using top {n_symbols} symbols by history length")

    # Sample symbols for logging
    sample = top_symbols[:5]
    logger.info(f"Sample symbols: {sample}")

    # Configure evolution
    config = EvolutionConfig(
        population_size=population,
        generations_per_session=generations,
        max_runtime_hours=hours,
        min_tree_depth=1,
        max_tree_depth=5,
        min_trades=10,
        novelty_k_neighbors=5,
        elitism=2,
        checkpoint_frequency=5,
        log_frequency=1,
        parallel_enabled=False,  # Simpler debugging
    )

    # Create engine
    engine = EvolutionEngine(config=config)
    engine.load_data(data=filtered_data, vix_data=vix_data)

    # Initialize fresh population
    engine.initialize_population()
    logger.info(f"Initialized population of {len(engine.population)}")

    # Run evolution
    logger.info(f"Starting evolution (max {hours} hours, {generations} generations)")
    engine.run(generations=generations, hours=hours)

    # Report results
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info("DISCOVERY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration/60:.1f} minutes")
    logger.info(f"Generations: {engine.current_generation}")
    logger.info(f"Strategies evaluated: {engine.total_strategies_evaluated}")
    logger.info(f"Pareto front size: {len(engine.pareto_front)}")
    logger.info(f"Strategies promoted: {engine.strategies_promoted}")

    if engine.pareto_front:
        logger.info("")
        logger.info("Top Pareto solutions:")
        for i, genome in enumerate(engine.pareto_front[:5]):
            fitness = engine.fitness_cache.get(genome.genome_id)
            if fitness:
                logger.info(f"  {i+1}. {genome.genome_id}: Sortino={fitness.sortino:.2f}, DD={fitness.max_drawdown:.1f}%")
            else:
                logger.info(f"  {i+1}. {genome.genome_id}: entry={str(genome.entry_tree)[:40]}...")

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast GP Discovery Run")
    parser.add_argument("--symbols", type=int, default=100, help="Number of symbols to use")
    parser.add_argument("--hours", type=float, default=1.0, help="Max runtime in hours")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")

    args = parser.parse_args()

    run_fast_discovery(
        n_symbols=args.symbols,
        hours=args.hours,
        population=args.population,
        generations=args.generations
    )
