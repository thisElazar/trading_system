"""
Overnight Runner
================
Main entry point for autonomous overnight strategy discovery.

Uses Island-based evolution by default for better diversity and exploration.

Usage:
    python -m research.discovery.overnight_runner --hours 8
    python -m research.discovery.overnight_runner --generations 100
    python -m research.discovery.overnight_runner --validate  # Run validation test
    python -m research.discovery.overnight_runner --no-islands  # Use single population

Features:
- Island-based parallel evolution (default)
- Automatic checkpoint loading/saving
- Strategy persistence to database
- Progress monitoring and logging
- Graceful shutdown handling
- Final report generation
"""

import argparse
import logging
import sys
import os
import json
import signal
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Ensure parent is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .config import EvolutionConfig, IslandConfig, DEFAULT_CONFIG, PI_CONFIG, DEFAULT_ISLAND_CONFIG, OVERNIGHT_ISLAND_CONFIG
from .evolution_engine import EvolutionEngine
from .island_model import IslandEvolutionEngine
from .db_schema import migrate_discovery_tables, check_tables_exist
from data.storage.db_manager import get_db

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path = None, verbose: bool = False):
    """Setup logging for overnight runs."""
    log_dir = log_dir or Path("research/discovery/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=handlers
    )

    logger.info(f"Logging to {log_file}")
    return log_file


def load_market_data(symbols: list = None, start_date: str = None,
                     end_date: str = None, max_symbols: int = None) -> Dict[str, pd.DataFrame]:
    """
    Load market data for backtesting.

    Args:
        symbols: List of symbols to load (default: load priority symbols)
        start_date: Start date string (default: 2 years ago)
        end_date: End date string (default: today)
        max_symbols: Maximum number of symbols to load

    Returns:
        Dict mapping symbol to DataFrame
    """
    from data.unified_data_loader import UnifiedDataLoader

    loader = UnifiedDataLoader()

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Priority symbols for diverse market coverage
    priority_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',  # Tech
        'JPM', 'BAC', 'GS', 'WFC', 'C',                            # Finance
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',                        # Healthcare
        'XOM', 'CVX', 'COP', 'SLB',                                 # Energy
        'WMT', 'COST', 'TGT', 'HD',                                 # Retail
        'DIS', 'NFLX', 'CMCSA',                                     # Media
        'CAT', 'DE', 'MMM', 'GE',                                   # Industrial
        'PG', 'KO', 'PEP', 'MCD'                                    # Consumer
    ]

    if symbols is None:
        available = loader.get_available_daily_symbols()
        symbols = [s for s in priority_symbols if s in available]
        if max_symbols:
            symbols = symbols[:max_symbols]

    data = {}
    for symbol in symbols:
        try:
            df = loader.load_daily(symbol)
            if df is not None and len(df) >= 252:  # At least 1 year
                # Ensure we have a datetime index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Filter to date range
                df = df.loc[start_date:end_date]
                if len(df) >= 252:
                    data[symbol] = df
        except Exception as e:
            logger.debug(f"Failed to load {symbol}: {e}")

    logger.info(f"Loaded data for {len(data)} symbols")
    return data


def load_vix_data() -> Optional[pd.DataFrame]:
    """Load VIX data for regime detection."""
    try:
        from data.fetchers.vix_fetcher import VIXFetcher
        fetcher = VIXFetcher()
        vix = fetcher.load_vix()
        if vix is not None:
            logger.info(f"Loaded VIX data: {len(vix)} days")
        return vix
    except Exception as e:
        logger.warning(f"Failed to load VIX data: {e}")
        return None


def save_discovered_strategy(genome, fitness, run_id: str):
    """
    Save a discovered strategy to the database.

    Args:
        genome: The strategy genome
        fitness: FitnessVector with performance metrics
        run_id: Unique run identifier
    """
    try:
        db = get_db()

        # Serialize genome
        genome_data = {
            'genome_id': genome.genome_id,
            'generation': genome.generation,
            'entry_tree': str(genome.entry_tree),
            'exit_tree': str(genome.exit_tree) if genome.exit_tree else None,
            'position_tree': str(genome.position_tree) if genome.position_tree else None,
            'stop_loss_tree': str(genome.stop_loss_tree) if genome.stop_loss_tree else None,
            'target_tree': str(genome.target_tree) if genome.target_tree else None,
        }

        db.execute(
            "research",
            """
            INSERT OR REPLACE INTO discovered_strategies
            (strategy_id, genome_json, generation_discovered,
             oos_sharpe, oos_sortino, oos_max_drawdown, oos_total_trades, oos_win_rate,
             behavior_vector, novelty_score, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?)
            """,
            (
                genome.genome_id,
                json.dumps(genome_data),
                genome.generation,
                fitness.sharpe,
                fitness.sortino,
                fitness.max_drawdown,
                fitness.trades,
                fitness.win_rate,
                None,  # behavior_vector not available in this context
                fitness.novelty,
                datetime.now().isoformat()
            )
        )

        logger.info(f"Saved strategy {genome.genome_id[:8]} to database "
                   f"(Sharpe={fitness.sharpe:.2f}, Sortino={fitness.sortino:.2f}, Trades={fitness.trades})")

    except Exception as e:
        logger.error(f"Failed to save strategy: {e}")


def save_evolution_history(run_id: str, generation: int, stats: Dict[str, Any]):
    """Save generation statistics to database."""
    try:
        db = get_db()

        db.execute(
            "research",
            """
            INSERT INTO evolution_history
            (run_id, generation, pop_size, pareto_front_size, novelty_archive_size,
             best_sortino, avg_sortino, behavior_diversity, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                generation,
                stats.get('pop_size', 0),
                stats.get('pareto_size', 0),
                stats.get('novelty_archive_size', 0),
                stats.get('best_sortino', 0),
                stats.get('avg_sortino', 0),
                stats.get('diversity', 0),
                datetime.now().isoformat()
            )
        )
    except Exception as e:
        logger.debug(f"Failed to save history: {e}")


def run_island_discovery(
    hours: float = None,
    generations: int = None,
    config: EvolutionConfig = None,
    island_config: IslandConfig = None,
    symbols: list = None,
    max_symbols: int = 20
) -> Dict[str, Any]:
    """
    Run island-based strategy discovery.

    Args:
        hours: Maximum hours to run
        generations: Number of generations to run
        config: Evolution configuration
        island_config: Island configuration
        symbols: List of symbols to use
        max_symbols: Maximum symbols to load

    Returns:
        Results dictionary
    """
    config = config or DEFAULT_CONFIG
    island_config = island_config or DEFAULT_ISLAND_CONFIG

    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting island discovery run {run_id}")

    # Ensure database tables exist
    if not check_tables_exist():
        logger.info("Creating discovery database tables...")
        migrate_discovery_tables()

    # Create engine
    engine = IslandEvolutionEngine(
        config=config,
        island_config=island_config,
        use_fast_backtester=False
    )

    # Load data
    logger.info("Loading market data...")
    data = load_market_data(symbols=symbols, max_symbols=max_symbols)
    vix_data = load_vix_data()

    if not data:
        raise ValueError("No market data available")

    engine.load_data(data=data, vix_data=vix_data)
    engine.initialize_populations()

    # Log configuration
    logger.info(f"Island configuration:")
    logger.info(f"  Islands: {island_config.num_islands}")
    logger.info(f"  Population per island: {island_config.population_per_island}")
    logger.info(f"  Total population: {island_config.num_islands * island_config.population_per_island}")
    logger.info(f"  Migration interval: {island_config.migration_interval}")
    logger.info(f"  Symbols: {len(data)}")

    for island in engine.islands:
        logger.info(f"  Island {island.island_id}: mutation={island.mutation_rate:.2f}, depth={island.max_tree_depth}")

    # Setup shutdown handler
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info("Shutdown requested, finishing current generation...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run evolution
    start_time = datetime.now()
    max_time = hours * 3600 if hours else None
    max_gens = generations or 10000

    logger.info(f"Starting evolution at {start_time}")
    if hours:
        logger.info(f"Will run for up to {hours} hours")
    if generations:
        logger.info(f"Will run for up to {generations} generations")

    best_strategies_saved = set()
    last_log_time = time.time()

    while not shutdown_requested:
        # Check termination conditions
        elapsed = (datetime.now() - start_time).total_seconds()
        if max_time and elapsed >= max_time:
            logger.info("Time limit reached")
            break
        if engine.current_generation >= max_gens:
            logger.info("Generation limit reached")
            break

        # Evolve one generation
        engine.evolve_generation()

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

        stats = {
            'pop_size': sum(len(i.population) for i in engine.islands),
            'pareto_size': len(engine.global_pareto_front),
            'novelty_archive_size': sum(len(i.novelty_archive) for i in engine.islands),
            'best_sortino': max(all_sortinos) if all_sortinos else 0,
            'avg_sortino': np.mean(all_sortinos) if all_sortinos else 0,
            'diversity': np.mean(diversities) if diversities else 0
        }

        # Save history periodically
        if engine.current_generation % 5 == 0:
            save_evolution_history(run_id, engine.current_generation, stats)

        # Save promising strategies to database
        for island in engine.islands:
            for genome in island.population:
                if genome.genome_id in island.fitness_cache:
                    fitness = island.fitness_cache[genome.genome_id]
                    # Save strategies with good Sortino and sufficient trades
                    if (fitness.sortino >= 1.0 and
                        fitness.trades >= 20 and
                        genome.genome_id not in best_strategies_saved):
                        save_discovered_strategy(genome, fitness, run_id)
                        best_strategies_saved.add(genome.genome_id)

        # Log progress every 60 seconds
        if time.time() - last_log_time >= 60:
            last_log_time = time.time()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            logger.info(
                f"[{mins:02d}:{secs:02d}] Gen {engine.current_generation}: "
                f"Sortino=[{min(all_sortinos):.2f}, {max(all_sortinos):.2f}] avg={stats['avg_sortino']:.2f}, "
                f"Diversity={stats['diversity']:.3f}, Pareto={stats['pareto_size']}"
            )

    end_time = datetime.now()
    duration = end_time - start_time

    # Save final best strategies
    logger.info("Saving final best strategies...")
    for genome in engine.global_pareto_front[:20]:  # Top 20 from Pareto front
        for island in engine.islands:
            if genome.genome_id in island.fitness_cache:
                fitness = island.fitness_cache[genome.genome_id]
                if genome.genome_id not in best_strategies_saved:
                    save_discovered_strategy(genome, fitness, run_id)
                    best_strategies_saved.add(genome.genome_id)
                break

    # Generate results
    results = {
        'run_id': run_id,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_hours': duration.total_seconds() / 3600,
        'generations': engine.current_generation,
        'strategies_evaluated': engine.total_strategies_evaluated,
        'strategies_saved': len(best_strategies_saved),
        'pareto_front_size': len(engine.global_pareto_front),
        'final_diversity': np.mean([i.diversity for i in engine.islands]),
        'best_sortino': engine.global_best_fitness,
        'islands': [
            {
                'id': i.island_id,
                'best_fitness': i.best_fitness,
                'diversity': i.diversity,
                'mutation_rate': i.mutation_rate
            }
            for i in engine.islands
        ]
    }

    return results


def run_discovery(
    hours: float = None,
    generations: int = None,
    config: EvolutionConfig = None,
    resume: bool = True,
    symbols: list = None,
    use_islands: bool = True,
    island_config: IslandConfig = None,
    max_symbols: int = 20
) -> Dict[str, Any]:
    """
    Run strategy discovery.

    Args:
        hours: Maximum hours to run
        generations: Number of generations to run
        config: Evolution configuration
        resume: Whether to resume from checkpoint (single population only)
        symbols: List of symbols to use
        use_islands: Use island-based evolution (default True)
        island_config: Island configuration
        max_symbols: Maximum symbols to load

    Returns:
        Results dictionary
    """
    if use_islands:
        return run_island_discovery(
            hours=hours,
            generations=generations,
            config=config,
            island_config=island_config,
            symbols=symbols,
            max_symbols=max_symbols
        )

    # Original single-population mode
    config = config or DEFAULT_CONFIG

    # Create engine
    engine = EvolutionEngine(config=config)

    # Load data
    logger.info("Loading market data...")
    data = load_market_data(symbols=symbols, max_symbols=max_symbols)
    vix_data = load_vix_data()

    if not data:
        raise ValueError("No market data available")

    engine.load_data(data=data, vix_data=vix_data)

    # Resume or initialize
    if resume:
        loaded = engine.load_checkpoint()
        if not loaded:
            logger.info("No checkpoint found, initializing fresh population")
            engine.initialize_population()
    else:
        engine.initialize_population()

    # Run evolution
    start_time = datetime.now()
    logger.info(f"Starting evolution at {start_time}")

    engine.run(generations=generations, hours=hours)

    end_time = datetime.now()
    duration = end_time - start_time

    # Generate results
    results = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_hours': duration.total_seconds() / 3600,
        'generations': engine.current_generation,
        'strategies_evaluated': engine.total_strategies_evaluated,
        'strategies_promoted': engine.strategies_promoted,
        'pareto_front_size': len(engine.pareto_front),
        'novelty_archive_size': len(engine.novelty_archive),
        'final_diversity': engine.novelty_archive.get_archive_diversity()
    }

    return results


def generate_report(results: Dict[str, Any], output_path: Path = None):
    """Generate a summary report."""
    output_path = output_path or Path("research/discovery/reports")
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Build report based on whether islands were used
    if 'islands' in results:
        island_info = "\n".join([
            f"    Island {i['id']}: best={i['best_fitness']:.3f}, diversity={i['diversity']:.3f}"
            for i in results['islands']
        ])
        extra_info = f"""
Island Performance:
{island_info}
"""
    else:
        extra_info = f"""
  Strategies Promoted: {results.get('strategies_promoted', 0)}
  Novelty Archive Size: {results.get('novelty_archive_size', 0)}
"""

    report = f"""
================================================================================
AUTONOMOUS STRATEGY DISCOVERY REPORT
================================================================================

Run Information:
  Run ID: {results.get('run_id', 'N/A')}
  Start Time: {results['start_time']}
  End Time: {results['end_time']}
  Duration: {results['duration_hours']:.2f} hours

Evolution Statistics:
  Generations Completed: {results['generations']}
  Total Strategies Evaluated: {results['strategies_evaluated']}
  Strategies Saved to DB: {results.get('strategies_saved', 0)}
  Pareto Front Size: {results['pareto_front_size']}
  Final Diversity: {results.get('final_diversity', 0):.4f}
  Best Sortino: {results.get('best_sortino', 0):.3f}
{extra_info}
================================================================================
"""

    report_file.write_text(report)
    logger.info(f"Report saved to {report_file}")
    print(report)

    return report_file


def validate_overnight_run(verbose: bool = False, use_islands: bool = True) -> Dict[str, Any]:
    """
    Run a validation test of the overnight evolution system.

    Tests:
    1. Database setup
    2. Population initialization
    3. 3-generation evolution run
    4. Strategy persistence
    5. Resource usage

    Returns:
        Dict with validation results
    """
    results = {
        'passed': True,
        'tests': {},
        'memory': {},
        'errors': []
    }

    # Memory tracking
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
    else:
        process = None
        start_memory = 0

    print("=" * 60)
    print("OVERNIGHT EVOLUTION VALIDATION TEST")
    print(f"Mode: {'Island-based' if use_islands else 'Single population'}")
    print("=" * 60)
    print()

    # Test 1: Database setup
    print("[1/5] Setting up database tables...")
    try:
        if not check_tables_exist():
            migrate_discovery_tables()
        results['tests']['database'] = 'PASSED'
        print("   ✓ Database tables ready")
    except Exception as e:
        results['tests']['database'] = f'FAILED: {e}'
        results['passed'] = False
        results['errors'].append(str(e))
        print(f"   ✗ Database setup failed: {e}")

    # Test 2: Create test data
    print("[2/5] Creating test market data...")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        test_data = {}

        for symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']:
            base_price = 100 + np.random.randn() * 20
            returns = np.random.randn(300) * 0.02
            prices = base_price * np.cumprod(1 + returns)

            test_data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.randn(300) * 0.001),
                'high': prices * (1 + np.abs(np.random.randn(300)) * 0.01),
                'low': prices * (1 - np.abs(np.random.randn(300)) * 0.01),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 300)
            }, index=dates)

        results['tests']['data_creation'] = 'PASSED'
        print(f"   ✓ Created test data for {len(test_data)} symbols")
    except Exception as e:
        results['tests']['data_creation'] = f'FAILED: {e}'
        results['passed'] = False
        results['errors'].append(str(e))
        print(f"   ✗ Failed to create test data: {e}")
        return results

    # Test 3: Initialize engine
    print("[3/5] Initializing evolution engine...")
    try:
        config = EvolutionConfig(
            population_size=10,
            generations_per_session=3,
            min_trades=5,
            max_tree_depth=4
        )

        if use_islands:
            island_config = IslandConfig(
                num_islands=3,
                population_per_island=8,
                migration_interval=2
            )
            engine = IslandEvolutionEngine(
                config=config,
                island_config=island_config,
                use_fast_backtester=False
            )
            engine.load_data(data=test_data)
            engine.initialize_populations()
            pop_size = sum(len(i.population) for i in engine.islands)
        else:
            engine = EvolutionEngine(config=config, use_portfolio_fitness=False)
            engine.load_data(data=test_data)
            engine.initialize_population()
            pop_size = len(engine.population)

        results['tests']['engine_init'] = 'PASSED'
        print(f"   ✓ Engine initialized with population of {pop_size}")
    except Exception as e:
        results['tests']['engine_init'] = f'FAILED: {e}'
        results['passed'] = False
        results['errors'].append(str(e))
        print(f"   ✗ Failed to initialize engine: {e}")
        return results

    # Test 4: Run evolution
    print("[4/5] Running 3-generation evolution test...")
    try:
        start_time = time.time()

        if use_islands:
            for _ in range(3):
                engine.evolve_generation()
            strategies_evaluated = engine.total_strategies_evaluated
            pareto_size = len(engine.global_pareto_front)
        else:
            engine.run(generations=3)
            strategies_evaluated = engine.total_strategies_evaluated
            pareto_size = len(engine.pareto_front)

        elapsed = time.time() - start_time

        results['tests']['evolution_run'] = 'PASSED'
        results['evolution_time_seconds'] = elapsed
        print(f"   ✓ Completed 3 generations in {elapsed:.1f}s")
        print(f"     - Strategies evaluated: {strategies_evaluated}")
        print(f"     - Pareto front size: {pareto_size}")
    except Exception as e:
        results['tests']['evolution_run'] = f'FAILED: {e}'
        results['passed'] = False
        results['errors'].append(str(e))
        print(f"   ✗ Evolution run failed: {e}")
        return results

    # Test 5: Strategy persistence
    print("[5/5] Testing strategy persistence...")
    try:
        saved_count = 0
        if use_islands:
            for island in engine.islands:
                for genome in island.population[:2]:
                    if genome.genome_id in island.fitness_cache:
                        fitness = island.fitness_cache[genome.genome_id]
                        save_discovered_strategy(genome, fitness, "test_run")
                        saved_count += 1
        else:
            for genome in engine.population[:2]:
                if genome.genome_id in engine.fitness_cache:
                    fitness = engine.fitness_cache[genome.genome_id]
                    save_discovered_strategy(genome, fitness, "test_run")
                    saved_count += 1

        results['tests']['persistence'] = 'PASSED'
        print(f"   ✓ Saved {saved_count} strategies to database")
    except Exception as e:
        results['tests']['persistence'] = f'FAILED: {e}'
        results['errors'].append(str(e))
        print(f"   ✗ Persistence test failed: {e}")

    # Memory stats
    if HAS_PSUTIL and process:
        end_memory = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.1)

        results['memory'] = {
            'start_mb': round(start_memory, 2),
            'end_mb': round(end_memory, 2),
            'delta_mb': round(end_memory - start_memory, 2)
        }

        print()
        print("-" * 60)
        print("RESOURCE USAGE:")
        print(f"  Memory: {start_memory:.1f} MB -> {end_memory:.1f} MB (delta: {end_memory - start_memory:+.1f} MB)")
        print()

    # Summary
    passed = sum(1 for v in results['tests'].values() if 'PASSED' in str(v))
    total = len(results['tests'])
    results['passed'] = passed == total

    print("=" * 60)
    if results['passed']:
        print(f"VALIDATION PASSED ({passed}/{total} tests)")
    else:
        print(f"VALIDATION FAILED ({passed}/{total} tests passed)")
        for name, status in results['tests'].items():
            if 'FAILED' in str(status):
                print(f"  - {name}: {status}")
    print("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Strategy Discovery Engine (Island-based)"
    )

    parser.add_argument(
        '--hours', type=float, default=None,
        help='Maximum hours to run'
    )
    parser.add_argument(
        '--generations', type=int, default=None,
        help='Number of generations to run'
    )
    parser.add_argument(
        '--population', type=int, default=None,
        help='Population size per island'
    )
    parser.add_argument(
        '--islands', type=int, default=None,
        help='Number of islands (default: 4)'
    )
    parser.add_argument(
        '--no-islands', action='store_true',
        help='Use single population instead of island model'
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='Start fresh instead of resuming from checkpoint'
    )
    parser.add_argument(
        '--pi', action='store_true',
        help='Use Pi-optimized configuration'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose logging'
    )
    parser.add_argument(
        '--symbols', type=str, nargs='+',
        help='Specific symbols to use'
    )
    parser.add_argument(
        '--max-symbols', type=int, default=20,
        help='Maximum number of symbols to load (default: 20)'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run validation test'
    )

    args = parser.parse_args()

    # Handle validation mode
    if args.validate:
        setup_logging(verbose=args.verbose)
        results = validate_overnight_run(
            verbose=args.verbose,
            use_islands=not args.no_islands
        )
        return 0 if results['passed'] else 1

    # Setup logging
    log_file = setup_logging(verbose=args.verbose)

    # Select config
    if args.pi:
        config = PI_CONFIG
        island_config = IslandConfig(num_islands=3, population_per_island=15)
        logger.info("Using Pi-optimized configuration")
    else:
        config = DEFAULT_CONFIG
        island_config = OVERNIGHT_ISLAND_CONFIG

    # Override settings if specified
    if args.population:
        island_config.population_per_island = args.population
    if args.islands:
        island_config.num_islands = args.islands

    # Default to 8 hours if neither specified
    if args.hours is None and args.generations is None:
        args.hours = 8.0
        logger.info("No duration specified, defaulting to 8 hours")

    use_islands = not args.no_islands

    try:
        results = run_discovery(
            hours=args.hours,
            generations=args.generations,
            config=config,
            resume=not args.no_resume,
            symbols=args.symbols,
            use_islands=use_islands,
            island_config=island_config if use_islands else None,
            max_symbols=args.max_symbols
        )

        generate_report(results)

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1

    except Exception as e:
        logger.exception(f"Evolution failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
