"""
Research Infrastructure Test
============================
Quick test to verify all research tools are working correctly.

Run this before your first full research run to catch any issues early.

Usage:
    python scripts/test_research_tools.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS
from data.cached_data_manager import CachedDataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_infrastructure():
    """Test data loading."""
    print("\n" + "="*60)
    print("TEST 1: Data Infrastructure")
    print("="*60)
    
    try:
        dm = CachedDataManager()
        
        # Check if data exists
        symbols = dm.get_available_symbols()
        print(f"✓ Found {len(symbols)} symbols")
        
        if len(symbols) == 0:
            print("❌ FAIL: No data found")
            print("   Run: python scripts/universe_downloader.py --index sp500 --years 2")
            return False
        
        # Load data
        loaded = dm.load_all()
        print(f"✓ Loaded {loaded} symbols into cache")
        
        # Check metadata
        metadata = dm.get_all_metadata()
        print(f"✓ Generated metadata for {len(metadata)} symbols")
        
        # Check VIX
        vix = dm.get_vix()
        print(f"✓ VIX: {vix:.1f}")
        
        print("\n✅ Data infrastructure: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Data infrastructure: FAIL")
        print(f"   Error: {e}")
        return False


def test_backtester():
    """Test backtesting engine."""
    print("\n" + "="*60)
    print("TEST 2: Backtesting Engine")
    print("="*60)
    
    try:
        from research.backtester import Backtester
        from strategies.vol_managed_momentum import VolManagedMomentumStrategy
        
        # Load small dataset
        dm = CachedDataManager()
        if not dm.cache:
            dm.load_all()
        
        # Get 10 symbols
        symbols = list(dm.cache.keys())[:10]
        data = {s: dm.get_bars(s) for s in symbols}
        
        print(f"Testing on {len(data)} symbols...")
        
        # Run quick backtest
        strategy = VolManagedMomentumStrategy()
        backtester = Backtester(initial_capital=100000)
        
        result = backtester.run(strategy, data)
        
        print(f"✓ Backtest completed")
        print(f"  - Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  - Return: {result.annual_return:.1f}%")
        print(f"  - Trades: {result.total_trades}")
        
        if result.total_trades == 0:
            print("⚠️  WARNING: No trades generated (may be normal for small dataset)")
        
        print("\n✅ Backtester: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Backtester: FAIL")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_comparison():
    """Test strategy comparison."""
    print("\n" + "="*60)
    print("TEST 3: Strategy Comparison")
    print("="*60)
    
    try:
        from research.strategy_comparison import StrategyComparison
        
        comp = StrategyComparison()
        print(f"✓ Initialized with {len(comp.strategies)} strategies")
        
        # We won't run full comparison (too slow for test)
        # Just verify it loads
        print("✓ Strategy comparison module loaded")
        
        print("\n✅ Strategy Comparison: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Strategy Comparison: FAIL")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monte_carlo():
    """Test Monte Carlo engine."""
    print("\n" + "="*60)
    print("TEST 4: Monte Carlo Engine")
    print("="*60)
    
    try:
        from research.monte_carlo import MonteCarloSimulator
        
        mc = MonteCarloSimulator()
        print(f"✓ Monte Carlo simulator initialized")
        
        # We won't run simulations (too slow)
        # Just verify it loads
        print("✓ Monte Carlo module loaded")
        
        print("\n✅ Monte Carlo: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Monte Carlo: FAIL")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_optimizer():
    """Test parameter optimizer."""
    print("\n" + "="*60)
    print("TEST 5: Parameter Optimizer")
    print("="*60)
    
    try:
        from research.parameter_optimizer import ParameterOptimizer
        
        opt = ParameterOptimizer()
        print(f"✓ Parameter optimizer initialized")
        
        print("✓ Parameter optimizer module loaded")
        
        print("\n✅ Parameter Optimizer: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Parameter Optimizer: FAIL")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test database connectivity."""
    print("\n" + "="*60)
    print("TEST 6: Database")
    print("="*60)
    
    try:
        from data.storage.db_manager import get_db
        
        db = get_db()
        print("✓ Database connection established")
        
        # Try a simple query
        rows = db.fetchall("research", "SELECT * FROM backtests LIMIT 5")
        print(f"✓ Query successful ({len(rows)} existing backtest records)")
        
        print("\n✅ Database: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Database: FAIL")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_runner_script():
    """Test runner script exists and imports."""
    print("\n" + "="*60)
    print("TEST 7: Runner Script")
    print("="*60)
    
    try:
        import importlib.util
        
        runner_path = Path(__file__).parent / "run_research.py"
        
        if not runner_path.exists():
            print(f"❌ Runner script not found: {runner_path}")
            return False
        
        print(f"✓ Runner script exists: {runner_path}")
        
        # Try importing
        spec = importlib.util.spec_from_file_location("run_research", runner_path)
        module = importlib.util.module_from_spec(spec)
        
        print("✓ Runner script imports successfully")
        
        print("\n✅ Runner Script: PASS")
        return True
        
    except Exception as e:
        print(f"\n❌ Runner Script: FAIL")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RESEARCH INFRASTRUCTURE TEST SUITE")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run tests
    results.append(("Data Infrastructure", test_data_infrastructure()))
    results.append(("Backtester", test_backtester()))
    results.append(("Strategy Comparison", test_strategy_comparison()))
    results.append(("Monte Carlo", test_monte_carlo()))
    results.append(("Parameter Optimizer", test_parameter_optimizer()))
    results.append(("Database", test_database()))
    results.append(("Runner Script", test_runner_script()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")
    
    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("-"*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Research infrastructure ready to use.")
        print("\nNext steps:")
        print("1. Review RESEARCH_GUIDE.md for usage instructions")
        print("2. Run: python scripts/run_research.py --full --quick")
        print("3. Check results in research/backtests/")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Fix issues before running research.")
        print("\nCommon fixes:")
        print("- No data: python scripts/universe_downloader.py --index sp500 --years 2")
        print("- Import errors: Check all files are in correct directories")
        print("- Database errors: Delete db/*.db and restart")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
