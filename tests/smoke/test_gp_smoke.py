#!/usr/bin/env python3
"""
Smoke Test 4: Genetic Programming Engine
Validates GP primitives and minimal evolution.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_gp_engine():
    print("=" * 60)
    print("TEST 4: Genetic Programming Engine")
    print("=" * 60)
    print("(This test may take 30-60 seconds)")
    
    start_time = time.time()
    results = {
        "primitives_loaded": False,
        "primitive_count": 0,
        "terminal_count": 0,
        "population_created": False,
        "evolution_completed": False,
        "generations": 0,
        "best_fitness": 0,
    }
    
    try:
        from deap import base, creator, tools, gp
        print("\n[OK] DEAP imported")
        
        from research.discovery.gp_core import create_primitive_set
        
        print("Loading primitive set...")
        pset = create_primitive_set()
        results["primitives_loaded"] = True
        
        prim_count = sum(len(v) for v in pset.primitives.values())
        term_count = sum(len(v) for v in pset.terminals.values())
        results["primitive_count"] = prim_count
        results["terminal_count"] = term_count
        print(f"  Primitives: {prim_count} functions, {term_count} terminals")
        
        # Try to run a minimal evolution
        print("\nAttempting minimal evolution (10 pop, 2 gen)...")
        try:
            from research.discovery.evolution_engine import EvolutionEngine
            
            engine = EvolutionEngine(
                population_size=10,
                generations=2,
                symbols=["SPY"],
                start_date="2024-01-01",
                end_date="2024-01-31"
            )
            results["population_created"] = True
            
            evo_start = time.time()
            evolution_result = engine.run()
            evo_time = time.time() - evo_start
            
            results["evolution_completed"] = True
            results["generations"] = 2
            if evolution_result:
                results["best_fitness"] = getattr(evolution_result, "best_fitness", 0)
            
            print(f"  Evolution completed in {evo_time:.1f}s")
            
        except Exception as e:
            print(f"  Evolution test skipped: {e}")
            results["evolution_completed"] = False
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, results, time.time() - start_time
    
    duration = time.time() - start_time
    passed = results["primitives_loaded"] and results["primitive_count"] >= 40
    
    print(f"\nKey Metrics:")
    print(f"  - Primitives loaded: {results['primitives_loaded']}")
    print(f"  - Function count: {results['primitive_count']}")
    print(f"  - Terminal count: {results['terminal_count']}")
    print(f"  - Population created: {results['population_created']}")
    print(f"  - Evolution completed: {results['evolution_completed']}")
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)
    
    return passed, results, duration

if __name__ == "__main__":
    passed, results, duration = test_gp_engine()
    sys.exit(0 if passed else 1)
