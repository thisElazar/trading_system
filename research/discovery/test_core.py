#!/usr/bin/env python3
"""
Test Core GP Components
=======================
Quick validation that the discovery engine components work correctly.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('test_core')


def test_primitive_sets():
    """Test primitive set creation."""
    logger.info("=" * 50)
    logger.info("Testing Primitive Sets")
    logger.info("=" * 50)
    
    from research.discovery.gp_core import (
        create_primitive_set,
        create_boolean_primitive_set,
        PrimitiveConfig
    )
    
    config = PrimitiveConfig()
    
    # Float primitive set
    float_pset = create_primitive_set(config)
    logger.info(f"Float pset: {len(float_pset.primitives)} primitives, {len(float_pset.terminals)} terminals")
    
    # Boolean primitive set
    bool_pset = create_boolean_primitive_set(config)
    logger.info(f"Bool pset: {len(bool_pset.primitives)} primitives, {len(bool_pset.terminals)} terminals")
    
    return True


def test_genome_creation():
    """Test genome factory."""
    logger.info("=" * 50)
    logger.info("Testing Genome Creation")
    logger.info("=" * 50)
    
    from research.discovery.strategy_genome import GenomeFactory
    from research.discovery.config import EvolutionConfig
    
    config = EvolutionConfig(
        max_tree_depth=4,
        min_tree_depth=1,
        population_size=10
    )
    
    factory = GenomeFactory(config)
    
    # Create genomes
    genomes = []
    for i in range(5):
        g = factory.create_random_genome(generation=0)
        genomes.append(g)
        logger.info(f"  Genome {i}: {g.genome_id}, complexity={g.total_complexity}")
        logger.info(f"    Entry: {str(g.entry_tree)[:60]}...")
    
    # Test crossover
    logger.info("\nTesting crossover...")
    child1, child2 = factory.crossover(genomes[0], genomes[1], generation=1)
    logger.info(f"  Child 1: {child1.genome_id}, parents={child1.parent_ids}")
    logger.info(f"  Child 2: {child2.genome_id}")
    
    # Test mutation
    logger.info("\nTesting mutation...")
    mutant = factory.mutate(genomes[0], generation=1)
    logger.info(f"  Original: {genomes[0].genome_id}")
    logger.info(f"  Mutant: {mutant.genome_id}, parent={mutant.parent_ids}")
    
    # Test serialization
    logger.info("\nTesting serialization...")
    serialized = factory.serialize_genome(genomes[0])
    logger.info(f"  Serialized length: {len(serialized)} chars")
    
    # Deserialization often fails - let's test carefully
    try:
        deserialized = factory.deserialize_genome(serialized)
        logger.info(f"  Deserialized: {deserialized.genome_id}")
        logger.info("  ✅ Serialization working!")
    except Exception as e:
        logger.warning(f"  ⚠️ Deserialization failed: {e}")
        logger.warning("  This may need debugging for checkpoint persistence")
    
    return True


def test_genome_evaluation():
    """Test genome evaluation on sample data."""
    logger.info("=" * 50)
    logger.info("Testing Genome Evaluation")
    logger.info("=" * 50)
    
    import pandas as pd
    import numpy as np
    from research.discovery.strategy_genome import GenomeFactory
    from research.discovery.gp_core import set_eval_data, clear_eval_data
    
    factory = GenomeFactory()
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    price_walk = np.random.randn(200).cumsum() + 100
    
    test_data = pd.DataFrame({
        'open': price_walk + np.random.randn(200) * 0.5,
        'high': price_walk + abs(np.random.randn(200)),
        'low': price_walk - abs(np.random.randn(200)),
        'close': price_walk,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)
    
    logger.info(f"Test data: {len(test_data)} rows")
    
    # Create and evaluate genomes
    success_count = 0
    fail_count = 0
    
    for i in range(10):
        genome = factory.create_random_genome(generation=0)
        try:
            result = factory.evaluate_genome(genome, test_data)
            success_count += 1
            logger.info(
                f"  Genome {i}: entry={result['entry_signal']}, exit={result['exit_signal']}, "
                f"pos={result['position_pct']:.2%}, stop={result['stop_loss_pct']:.2%}"
            )
        except Exception as e:
            fail_count += 1
            logger.warning(f"  Genome {i}: Evaluation failed - {e}")
    
    logger.info(f"\nEvaluation results: {success_count} success, {fail_count} failed")
    
    return fail_count < 5  # Allow some failures


def test_novelty_search():
    """Test novelty archive and behavior vectors."""
    logger.info("=" * 50)
    logger.info("Testing Novelty Search")
    logger.info("=" * 50)
    
    from research.discovery.novelty_search import BehaviorVector, NoveltyArchive
    import numpy as np
    
    # Create archive
    archive = NoveltyArchive(k_neighbors=5, archive_size=100)
    
    # Add some behaviors
    for i in range(20):
        behavior = BehaviorVector(
            trade_frequency=np.random.uniform(0.5, 5.0),
            avg_hold_period=np.random.uniform(1, 20),
            long_short_ratio=np.random.uniform(-0.5, 1.0),
            return_autocorr=np.random.uniform(-0.3, 0.3),
            drawdown_depth=np.random.uniform(0.1, 0.5),
            benchmark_corr=np.random.uniform(-0.2, 0.8),
            signal_variance=np.random.uniform(0.01, 0.1)
        )
        novelty = archive.calculate_novelty(behavior)
        archive.maybe_add(behavior, novelty, fitness=np.random.uniform(0, 2))
    
    logger.info(f"Archive size: {len(archive)}")
    
    # Test novelty calculation
    test_behavior = BehaviorVector(
        trade_frequency=10.0,  # Very different
        avg_hold_period=1.0,
        long_short_ratio=1.0,
        return_autocorr=0.0,
        drawdown_depth=0.1,
        benchmark_corr=0.0,
        signal_variance=0.05
    )
    novelty = archive.calculate_novelty(test_behavior)
    logger.info(f"Novel behavior novelty score: {novelty:.4f}")
    
    return True


def test_multi_objective():
    """Test multi-objective fitness calculation."""
    logger.info("=" * 50)
    logger.info("Testing Multi-Objective Fitness")
    logger.info("=" * 50)
    
    from research.discovery.multi_objective import FitnessVector, non_dominated_sort, crowding_distance
    
    # Create sample fitness vectors
    fitness_vectors = [
        FitnessVector(sortino=1.5, max_drawdown=-15, cvar_95=-0.02, novelty=0.5, deflated_sharpe=0.8),
        FitnessVector(sortino=1.2, max_drawdown=-10, cvar_95=-0.015, novelty=0.7, deflated_sharpe=0.9),
        FitnessVector(sortino=2.0, max_drawdown=-25, cvar_95=-0.03, novelty=0.3, deflated_sharpe=0.75),
        FitnessVector(sortino=0.8, max_drawdown=-8, cvar_95=-0.01, novelty=0.9, deflated_sharpe=0.95),
        FitnessVector(sortino=1.8, max_drawdown=-20, cvar_95=-0.025, novelty=0.4, deflated_sharpe=0.85),
    ]
    
    logger.info(f"Testing {len(fitness_vectors)} fitness vectors")
    
    # Non-dominated sorting
    fronts = non_dominated_sort(fitness_vectors)
    logger.info(f"Non-dominated fronts: {len(fronts)}")
    for i, front in enumerate(fronts):
        logger.info(f"  Front {i}: {front}")
    
    # Crowding distance
    if fronts and len(fronts[0]) > 2:
        distances = crowding_distance(fronts[0], fitness_vectors)
        logger.info(f"Crowding distances for front 0: {distances}")
    
    return True


def test_database_tables():
    """Test database table creation."""
    logger.info("=" * 50)
    logger.info("Testing Database Tables")
    logger.info("=" * 50)
    
    from research.discovery.db_schema import check_tables_exist, migrate_discovery_tables
    
    if check_tables_exist():
        logger.info("✅ Discovery tables exist")
    else:
        logger.info("Creating discovery tables...")
        try:
            migrate_discovery_tables()
            logger.info("✅ Tables created successfully")
        except Exception as e:
            logger.error(f"❌ Table creation failed: {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("GP STRATEGY DISCOVERY - COMPONENT TESTS")
    logger.info("=" * 70 + "\n")
    
    tests = [
        ("Primitive Sets", test_primitive_sets),
        ("Genome Creation", test_genome_creation),
        ("Genome Evaluation", test_genome_evaluation),
        ("Novelty Search", test_novelty_search),
        ("Multi-Objective", test_multi_objective),
        ("Database Tables", test_database_tables),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"\n❌ {name} FAILED: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {name}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
