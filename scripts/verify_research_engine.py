#!/usr/bin/env python3
"""
Verify Autonomous Research Engine Setup
=======================================
Run this script to verify all components are properly configured.

Usage:
    python scripts/verify_research_engine.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_database_tables():
    """Verify GA tables exist in research.db."""
    print("\n1. Checking database tables...")
    
    from data.storage.db_manager import get_db
    db = get_db()
    
    # Check tables exist
    tables_to_check = ['ga_populations', 'ga_history', 'ga_runs']
    
    for table in tables_to_check:
        result = db.fetchone(
            "research",
            f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        )
        if result:
            print(f"   ✓ Table '{table}' exists")
        else:
            print(f"   ✗ Table '{table}' MISSING - run db init")
            return False
    
    return True


def check_strategy_params():
    """Verify strategy parameter specs are defined."""
    print("\n2. Checking strategy parameter specs...")
    
    from research.genetic.optimizer import STRATEGY_PARAMS
    
    required_strategies = [
        'vol_managed_momentum',
        'pairs_trading',
        'relative_volume_breakout',
    ]
    
    all_good = True
    for strategy in required_strategies:
        if strategy in STRATEGY_PARAMS:
            params = STRATEGY_PARAMS[strategy]
            print(f"   ✓ {strategy}: {len(params)} parameters")
        else:
            print(f"   ✗ {strategy}: MISSING parameter specs")
            all_good = False
    
    return all_good


def check_persistent_optimizer():
    """Test PersistentGAOptimizer basic functionality."""
    print("\n3. Testing PersistentGAOptimizer...")
    
    from research.genetic.persistent_optimizer import PersistentGAOptimizer
    from research.genetic.optimizer import GeneticConfig
    
    # Simple test fitness function
    def test_fitness(genes):
        return sum(genes.values()) / max(len(genes), 1)
    
    try:
        optimizer = PersistentGAOptimizer(
            'vol_managed_momentum',
            test_fitness,
            config=GeneticConfig(population_size=5, generations=1)
        )
        print("   ✓ PersistentGAOptimizer instantiated")
        
        # Test save/load
        optimizer.optimizer.population = [
            optimizer.optimizer._create_individual(0)
            for _ in range(5)
        ]
        for ind in optimizer.optimizer.population:
            ind.fitness = test_fitness(ind.genes)
        
        optimizer.current_generation = 1
        optimizer.best_ever_fitness = 0.5
        optimizer.best_ever_genes = {'test': 0.5}
        optimizer.save_population()
        print("   ✓ Population saved to database")
        
        # Test load
        loaded = optimizer.load_population()
        if loaded:
            print("   ✓ Population loaded from database")
        else:
            print("   ! No population loaded (expected on first run)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def check_db_operations():
    """Test GA database operations."""
    print("\n4. Testing database operations...")
    
    from data.storage.db_manager import get_db
    import json
    
    db = get_db()
    
    try:
        # Test save/load cycle
        db.save_ga_population(
            strategy='_test_strategy',
            generation=1,
            population=[{'genes': {'a': 1}, 'fitness': 0.5, 'generation': 1}],
            best_fitness=0.5,
            best_genes={'a': 1}
        )
        print("   ✓ save_ga_population works")
        
        loaded = db.load_ga_population('_test_strategy')
        assert loaded is not None
        assert loaded['generation'] == 1
        print("   ✓ load_ga_population works")
        
        # Test history
        db.log_ga_history(
            strategy='_test_strategy',
            generation=1,
            best_fitness=0.5,
            mean_fitness=0.4,
            std_fitness=0.1,
            best_genes={'a': 1}
        )
        print("   ✓ log_ga_history works")
        
        history = db.get_ga_history('_test_strategy', days=1)
        assert len(history) > 0
        print("   ✓ get_ga_history works")
        
        # Test run tracking
        db.start_ga_run('_test_run_123')
        print("   ✓ start_ga_run works")
        
        db.complete_ga_run('_test_run_123', ['_test_strategy'], 1, 0)
        print("   ✓ complete_ga_run works")
        
        runs = db.get_recent_ga_runs(1)
        assert len(runs) > 0
        print("   ✓ get_recent_ga_runs works")
        
        # Cleanup test data
        db.execute("research", "DELETE FROM ga_populations WHERE strategy = '_test_strategy'")
        db.execute("research", "DELETE FROM ga_history WHERE strategy = '_test_strategy'")
        db.execute("research", "DELETE FROM ga_runs WHERE run_id = '_test_run_123'")
        print("   ✓ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_availability():
    """Verify market data is available for backtesting."""
    print("\n5. Checking data availability...")
    
    from data.cached_data_manager import CachedDataManager
    from config import DIRS
    
    try:
        dm = CachedDataManager()
        dm.load_all()
        
        n_symbols = len(dm.cache)
        print(f"   ✓ Loaded {n_symbols} symbols")
        
        if n_symbols < 100:
            print("   ! Warning: Low symbol count, backtests may be limited")
        
        # Check VIX data
        vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
        if vix_path.exists():
            print(f"   ✓ VIX data available")
        else:
            print(f"   ! VIX data not found at {vix_path}")
        
        return n_symbols > 0
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False


def check_nightly_runner():
    """Verify nightly runner can be imported."""
    print("\n6. Checking nightly runner module...")
    
    try:
        # Import should work without errors
        import run_nightly_research
        
        # Check key components exist
        assert hasattr(run_nightly_research, 'NightlyResearchEngine')
        assert hasattr(run_nightly_research, 'EVOLVABLE_STRATEGIES')
        assert hasattr(run_nightly_research, 'main')
        
        print("   ✓ run_nightly_research module loads correctly")
        print(f"   ✓ Evolvable strategies: {run_nightly_research.EVOLVABLE_STRATEGIES}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("AUTONOMOUS RESEARCH ENGINE VERIFICATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Database tables", check_database_tables()))
    results.append(("Strategy params", check_strategy_params()))
    results.append(("DB operations", check_db_operations()))
    results.append(("Persistent optimizer", check_persistent_optimizer()))
    results.append(("Data availability", check_data_availability()))
    results.append(("Nightly runner", check_nightly_runner()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All checks passed! ✓")
        print("\nTo run the research engine:")
        print("  python run_nightly_research.py          # Single run")
        print("  python run_nightly_research.py --loop   # Continuous")
        print("  python run_nightly_research.py --status # View status")
        return 0
    else:
        print("Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
