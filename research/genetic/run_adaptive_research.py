#!/usr/bin/env python3
"""
Adaptive Research Runner
========================
Integration script demonstrating the adaptive GA system.

Combines all components:
- MarketPeriodLibrary for curated testing periods
- RapidBacktester for 30-second period tests
- RegimeMatchingEngine for condition-matched testing
- AdaptiveGAOptimizer for multi-scale evolution
- MultiScaleFitnessCalculator for comprehensive evaluation
- AdaptiveStrategyManager for allocation decisions

Usage:
    python research/genetic/run_adaptive_research.py

    # Or with specific options:
    python research/genetic/run_adaptive_research.py --generations 20 --rapid-first
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )


def run_adaptive_research(args):
    """Main research function."""
    print("\n" + "=" * 70)
    print("ADAPTIVE GA RESEARCH SYSTEM")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Import components
    from research.genetic.market_periods import MarketPeriodLibrary
    from research.genetic.rapid_backtester import RapidBacktester
    from research.genetic.regime_matching import RegimeMatchingEngine
    from research.genetic.adaptive_optimizer import AdaptiveGAOptimizer, AdaptiveGAConfig
    from research.genetic.multiscale_fitness import MultiScaleFitnessCalculator
    from research.genetic.adaptive_strategy_manager import AdaptiveStrategyManager
    from research.genetic.strategy_genome import GenomeFactory

    # Initialize components
    print("\n[1/6] Initializing components...")
    library = MarketPeriodLibrary()
    regime_engine = RegimeMatchingEngine(library)
    rapid_backtester = RapidBacktester(library)
    fitness_calculator = MultiScaleFitnessCalculator(rapid_backtester, regime_engine)
    strategy_manager = AdaptiveStrategyManager(regime_engine=regime_engine)
    genome_factory = GenomeFactory()

    print(f"  - Loaded {len(library.get_all_periods())} market periods")
    print(f"  - {len(library.get_crisis_periods())} crisis periods for stress testing")
    print(f"  - {len(library.get_short_periods())} short periods for rapid testing")

    # Get current market conditions
    print("\n[2/6] Analyzing current market conditions...")
    fingerprint = regime_engine.get_current_fingerprint()
    print(f"  - Current regime: {fingerprint.overall_regime.upper()}")
    print(f"  - VIX level: {fingerprint.vix_level:.1f}")
    print(f"  - Trend direction: {fingerprint.trend_direction:+.2f}")
    print(f"  - Confidence: {fingerprint.regime_confidence:.0%}")

    # Find matching periods
    print("\n[3/6] Finding similar historical periods...")
    matches = regime_engine.find_matching_periods(fingerprint, n=5)
    for match in matches:
        print(f"  - {match.period.name}: {match.similarity:.0%} similar")

    # Get recommended test periods
    print("\n[4/6] Selecting test periods for GA...")
    test_periods = regime_engine.get_ga_test_periods()
    print(f"  Similar periods: {[p.name for p in test_periods['similar']]}")
    print(f"  Stress periods: {[p.name for p in test_periods['stress']]}")
    print(f"  Diverse periods: {[p.name for p in test_periods['diverse']]}")

    # Show current strategy allocations
    print("\n[5/6] Current strategy allocations...")
    strategy_manager.print_status()

    # Show regime comparison
    print("\n[6/6] Allocation by regime...")
    strategy_manager.print_regime_comparison()

    # Demo: Create some sample genomes
    print("\n" + "-" * 70)
    print("STRATEGY GENOME EXAMPLES")
    print("-" * 70)

    # Random genome
    genome1 = genome_factory.create_random(n_signals=2)
    print(f"\nRandom genome:")
    print(f"  Archetype: {genome1.get_archetype()}")
    print(f"  Is hybrid: {genome1.is_hybrid()}")
    print(f"  Signals: {[g.signal_type.value for g in genome1.signal_genes]}")

    # Archetype-based genome
    genome2 = genome_factory.create_from_archetype('momentum_volume')
    print(f"\nMomentum-Volume hybrid:")
    print(f"  Signals: {[g.signal_type.value for g in genome2.signal_genes]}")
    print(f"  Combination: {genome2.combination_method}")

    # Crossover
    child = genome_factory.crossover(genome1, genome2)
    print(f"\nCrossover child:")
    print(f"  Signals: {[g.signal_type.value for g in child.signal_genes]}")

    # Show what a full evolution run would look like
    print("\n" + "-" * 70)
    print("ADAPTIVE GA CONFIGURATION")
    print("-" * 70)

    config = AdaptiveGAConfig(
        total_population=60,
        n_islands=4,
        generations_per_session=args.generations,
        use_rapid_testing=True,
    )

    print(f"\nIsland model configuration:")
    print(f"  Total population: {config.total_population}")
    print(f"  Number of islands: {config.n_islands}")
    print(f"  Population per island: {config.island_population}")
    print(f"  Generations per session: {config.generations_per_session}")

    print(f"\nFitness weights:")
    print(f"  Long-term: {config.long_term_weight:.0%}")
    print(f"  Regime-matched: {config.regime_weight:.0%}")
    print(f"  Crisis resilience: {config.crisis_weight:.0%}")
    print(f"  Consistency: {config.consistency_weight:.0%}")

    # If running full evolution (requires data)
    if args.run_evolution:
        print("\n" + "-" * 70)
        print("RUNNING EVOLUTION (requires market data)")
        print("-" * 70)

        try:
            from data.cached_data_manager import CachedDataManager

            print("\nLoading market data...")
            data_manager = CachedDataManager()
            data_manager.load_all()

            if not data_manager.cache:
                print("  No data available. Skipping evolution.")
            else:
                print(f"  Loaded {len(data_manager.cache)} symbols")

                # Cache data in rapid backtester
                rapid_backtester.cache_data(data_manager.cache)

                # Initialize optimizer
                optimizer = AdaptiveGAOptimizer(
                    config=config,
                    period_library=library,
                    regime_engine=regime_engine,
                )

                # Run rapid generations first if requested
                if args.rapid_first:
                    print("\nRunning rapid generations on short periods...")
                    # Would need a real strategy factory here
                    print("  (Skipped - requires strategy factory)")

                print("\nEvolution would run here with real strategy factories...")
                print("  Use: optimizer.evolve(strategy_factory, data)")

        except ImportError as e:
            print(f"  Data manager not available: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
The adaptive GA system is ready for use. Key features:

1. RAPID TESTING (30-second tests)
   - Test on specific periods: COVID crash, 2008 GFC, bull runs
   - Quick parameter exploration before full backtests

2. REGIME-MATCHED EVOLUTION
   - Prioritize testing against current market conditions
   - Find strategies that work in similar historical periods

3. MULTI-SCALE FITNESS
   - Combines long-term, regime-specific, and crisis performance
   - Penalizes inconsistency and overfitting

4. ISLAND MODEL
   - Maintains diverse sub-populations
   - Specialists for crisis, bull markets, etc.
   - Migration between islands for innovation

5. CREATIVE STRATEGY DISCOVERY
   - StrategyGenome enables novel combinations
   - Signal mixing, conditional logic, archetype blending

6. ADAPTIVE ALLOCATION
   - Real-time strategy weights based on regime
   - Automatic rebalancing on regime changes

To use with your strategies:

    from research.genetic import (
        AdaptiveGAOptimizer,
        AdaptiveGAConfig,
        AdaptiveStrategyManager
    )

    # Create optimizer
    optimizer = AdaptiveGAOptimizer()

    # Define strategy factory
    def create_strategy(genes):
        return MyStrategy(**genes)

    # Evolve
    best = optimizer.evolve(create_strategy, data)

    # Get allocations
    manager = AdaptiveStrategyManager()
    allocations = manager.get_strategy_allocations()
""")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive GA research system"
    )
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=10,
        help='Number of generations per session'
    )
    parser.add_argument(
        '--rapid-first',
        action='store_true',
        help='Run rapid generations on short periods first'
    )
    parser.add_argument(
        '--run-evolution',
        action='store_true',
        help='Actually run evolution (requires data)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    run_adaptive_research(args)


if __name__ == "__main__":
    main()
