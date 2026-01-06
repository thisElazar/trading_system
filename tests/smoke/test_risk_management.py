#!/usr/bin/env python3
"""
Smoke Test 5: Risk Management
Validates circuit breakers and risk limits.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_risk_management():
    print("=" * 60)
    print("TEST 5: Risk Management")
    print("=" * 60)
    
    start_time = time.time()
    results = {
        "circuit_breaker_exists": False,
        "position_sizer_exists": False,
        "kill_switch_types": [],
        "max_drawdown_configured": False,
        "methods_exist": [],
    }
    
    try:
        # Check circuit breaker
        print("\nChecking Circuit Breaker...")
        try:
            from execution.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, KillSwitchType
            results["circuit_breaker_exists"] = True
            print("  [OK] CircuitBreaker imported")
            
            # Check kill switch types
            results["kill_switch_types"] = [e.name for e in KillSwitchType]
            print(f"  Kill switch types: {results['kill_switch_types']}")
        except ImportError as e:
            print(f"  [WARN] CircuitBreaker: {e}")
        
        # Check position sizer
        print("\nChecking Position Sizer...")
        try:
            from execution.position_sizer import PositionSizer
            results["position_sizer_exists"] = True
            print("  [OK] PositionSizer imported")
            
            # Check for max drawdown config
            import inspect
            source = inspect.getsourcefile(PositionSizer)
            with open(source) as f:
                content = f.read()
                if "max_drawdown" in content.lower() or "MAX_DRAWDOWN" in content:
                    results["max_drawdown_configured"] = True
                    print("  [OK] Max drawdown configured")
        except ImportError as e:
            print(f"  [WARN] PositionSizer: {e}")
        
        # Check for key methods
        print("\nChecking Risk Methods...")
        try:
            from execution.circuit_breaker import CircuitBreakerManager
            mgr = CircuitBreakerManager
            methods = ["check_breakers", "trigger_breaker", "clear_breaker"]
            for method in methods:
                if hasattr(mgr, method):
                    results["methods_exist"].append(method)
                    print(f"  [OK] {method}")
        except Exception as e:
            print(f"  [WARN] {e}")
        
        # Check alerts system
        print("\nChecking Alerts...")
        try:
            from execution.alerts import AlertManager, AlertLevel
            print("  [OK] AlertManager imported")
            print(f"  Alert levels: {[e.name for e in AlertLevel]}")
        except ImportError as e:
            print(f"  [WARN] Alerts: {e}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, results, time.time() - start_time
    
    duration = time.time() - start_time
    passed = results["circuit_breaker_exists"] and results["position_sizer_exists"]
    
    print(f"\nKey Metrics:")
    print(f"  - Circuit breaker: {results['circuit_breaker_exists']}")
    print(f"  - Position sizer: {results['position_sizer_exists']}")
    print(f"  - Max drawdown: {results['max_drawdown_configured']}")
    print(f"  - Kill switch types: {len(results['kill_switch_types'])}")
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)
    
    return passed, results, duration

if __name__ == "__main__":
    passed, results, duration = test_risk_management()
    sys.exit(0 if passed else 1)
