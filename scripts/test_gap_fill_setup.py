#!/usr/bin/env python3
"""
Gap-Fill Infrastructure Test
============================
Verify all components are working before downloading data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    try:
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DIRS
        print("  ✓ config.py")
    except Exception as e:
        print(f"  ✗ config.py: {e}")
        return False
    
    try:
        from data.fetchers.intraday_bars import IntradayDataManager
        print("  ✓ intraday_bars.py")
    except Exception as e:
        print(f"  ✗ intraday_bars.py: {e}")
        return False
    
    try:
        from strategies.gap_fill import GapFillStrategy
        print("  ✓ gap_fill.py")
    except Exception as e:
        print(f"  ✗ gap_fill.py: {e}")
        return False
    
    try:
        from data.fetchers.vix import get_current_vix
        print("  ✓ vix.py")
    except Exception as e:
        print(f"  ✗ vix.py: {e}")
        return False
    
    return True


def test_directories():
    """Test directory structure."""
    print("\nTesting directories...")
    
    from config import DIRS
    
    intraday_dir = DIRS["intraday"]
    if not intraday_dir.exists():
        intraday_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {intraday_dir}")
    else:
        print(f"  ✓ {intraday_dir}")
    
    return True


def test_api_connection():
    """Test Alpaca API connection."""
    print("\nTesting API connection...")
    
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("  ✗ API keys not configured")
        print("    Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
        return False
    
    print(f"  API Key: {ALPACA_API_KEY[:8]}...")
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        print("  ✓ Client created")
        
        # Test a simple request
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=5),
            end=datetime.now()
        )
        
        bars = client.get_stock_bars(request)
        if bars.df.empty:
            print("  ⚠ API returned empty data (market may be closed)")
        else:
            print(f"  ✓ API returned {len(bars.df)} bars")
        
        return True
        
    except Exception as e:
        print(f"  ✗ API test failed: {e}")
        return False


def test_strategy_init():
    """Test strategy initialization."""
    print("\nTesting strategy initialization...")
    
    try:
        from strategies.gap_fill import GapFillStrategy
        strategy = GapFillStrategy()
        print(f"  ✓ GapFillStrategy created")
        print(f"    Universe: {strategy.UNIVERSE}")
        print(f"    Gap range: {strategy.MIN_GAP_PCT}% - {strategy.MAX_GAP_PCT}%")
        print(f"    Hold time: {strategy.HOLD_MINUTES} minutes")
        return True
    except Exception as e:
        print(f"  ✗ Strategy init failed: {e}")
        return False


def main():
    print("=" * 60)
    print("GAP-FILL INFRASTRUCTURE TEST")
    print("=" * 60)
    
    all_pass = True
    
    all_pass &= test_imports()
    all_pass &= test_directories()
    all_pass &= test_api_connection()
    all_pass &= test_strategy_init()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED ✓")
        print("\nNext steps:")
        print("  1. Download intraday data:")
        print("     python data/fetchers/intraday_bars.py")
        print("  2. Run scanner (scan only):")
        print("     python scripts/gap_fill_scanner.py")
        print("  3. Run backtest:")
        print("     python scripts/gap_fill_scanner.py --backtest")
    else:
        print("SOME TESTS FAILED ✗")
        print("Fix issues above before proceeding")
    print("=" * 60)


if __name__ == "__main__":
    main()
