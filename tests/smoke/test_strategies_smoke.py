#!/usr/bin/env python3
"""
Smoke Test 2: Strategy Initialization
Validates all registered strategies can be instantiated.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_strategies():
    print("=" * 60)
    print("TEST 2: Strategy Initialization")
    print("=" * 60)
    
    start_time = time.time()
    results = {"total": 0, "passed": 0, "failed": 0, "strategies": {}}
    
    try:
        from strategies import STRATEGY_REGISTRY, get_strategy
        
        print(f"\nFound {len(STRATEGY_REGISTRY)} registered strategies:")
        results["total"] = len(STRATEGY_REGISTRY)
        
        for name in STRATEGY_REGISTRY.keys():
            try:
                strategy = get_strategy(name)
                results["strategies"][name] = "OK"
                results["passed"] += 1
                print(f"  [OK] {name}")
            except Exception as e:
                results["strategies"][name] = f"FAIL: {e}"
                results["failed"] += 1
                print(f"  [FAIL] {name}: {e}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, results, time.time() - start_time
    
    duration = time.time() - start_time
    passed = results["failed"] == 0 and results["total"] >= 10
    
    print(f"\nKey Metrics:")
    print(f"  - Total strategies: {results['total']}")
    print(f"  - Initialized: {results['passed']}")
    print(f"  - Failed: {results['failed']}")
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)
    
    return passed, results, duration

if __name__ == "__main__":
    passed, results, duration = test_strategies()
    sys.exit(0 if passed else 1)
