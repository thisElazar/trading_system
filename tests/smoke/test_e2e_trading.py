#!/usr/bin/env python3
"""
Smoke Test 6: End-to-End Trading Loop
Validates orchestrator components work together.
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_e2e_trading():
    print("=" * 60)
    print("TEST 6: End-to-End Trading Loop")
    print("=" * 60)
    
    start_time = time.time()
    results = {
        "alpaca_connected": False,
        "account_value": 0,
        "strategies_loaded": 0,
        "data_loader_works": False,
        "signal_generation": False,
    }
    
    try:
        # Test Alpaca connection
        print("\nTesting Alpaca connection...")
        try:
            from execution.alpaca_connector import AlpacaConnector
            connector = AlpacaConnector(paper=True)
            account = connector.get_account()
            if account:
                results["alpaca_connected"] = True
                results["account_value"] = float(account.portfolio_value)
                print(f"  [OK] Connected - Portfolio: ${results['account_value']:,.2f}")
        except Exception as e:
            print(f"  [WARN] Alpaca: {e}")
        
        # Test data loading - using correct API
        print("\nTesting data loader...")
        try:
            from data.unified_data_loader import UnifiedDataLoader
            loader = UnifiedDataLoader()
            df = loader.load_daily("SPY")
            
            # Filter to last 60 days
            end_dt = pd.Timestamp.now()
            start_dt = end_dt - pd.Timedelta(days=60)
            df = df[df["timestamp"] >= start_dt]
            
            if df is not None and len(df) > 0:
                results["data_loader_works"] = True
                print(f"  [OK] Loaded {len(df)} rows for SPY (last 60 days)")
        except Exception as e:
            print(f"  [WARN] Data loader: {e}")
        
        # Test strategy loading
        print("\nTesting strategy loading...")
        try:
            from strategies import STRATEGY_REGISTRY
            results["strategies_loaded"] = len(STRATEGY_REGISTRY)
            print(f"  [OK] {results['strategies_loaded']} strategies available")
        except Exception as e:
            print(f"  [WARN] Strategies: {e}")
        
        # Test signal generation (dry run)
        print("\nTesting signal generation...")
        try:
            from strategies import get_strategy
            strategy = get_strategy("mean_reversion")
            
            from data.unified_data_loader import UnifiedDataLoader
            loader = UnifiedDataLoader()
            data = loader.load_daily("SPY")
            
            # Filter to last 60 days and set index
            end_dt = pd.Timestamp.now()
            start_dt = end_dt - pd.Timedelta(days=60)
            data = data[data["timestamp"] >= start_dt].copy()
            data = data.set_index("timestamp")
            
            if data is not None and len(data) > 20:
                signals = strategy.generate_signals({"SPY": data})
                results["signal_generation"] = True
                signal_count = len(signals) if signals else 0
                print(f"  [OK] Generated {signal_count} signals")
        except Exception as e:
            print(f"  [WARN] Signal generation: {e}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, results, time.time() - start_time
    
    duration = time.time() - start_time
    passed = (
        results["alpaca_connected"] and
        results["strategies_loaded"] >= 10 and
        results["data_loader_works"]
    )
    
    print(f"\nKey Metrics:")
    print(f"  - Alpaca connected: {results['alpaca_connected']}")
    print(f"  - Portfolio value: ${results['account_value']:,.2f}")
    print(f"  - Strategies loaded: {results['strategies_loaded']}")
    print(f"  - Data loader works: {results['data_loader_works']}")
    print(f"  - Signal generation: {results['signal_generation']}")
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)
    
    return passed, results, duration

if __name__ == "__main__":
    passed, results, duration = test_e2e_trading()
    sys.exit(0 if passed else 1)
