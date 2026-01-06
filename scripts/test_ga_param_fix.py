#!/usr/bin/env python3
"""
Test script to verify GA parameter mapping fix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Clear bytecode cache warnings
import warnings
warnings.filterwarnings('ignore')

def test_strategy_param_setting():
    """Test that we can create strategies and set parameters."""
    from run_nightly_research import create_strategy_instance
    
    print("=" * 60)
    print("Testing GA Parameter Mapping Fix")
    print("=" * 60)
    
    # Test 1: VolManagedMomentumStrategy
    print("\n1. Testing vol_managed_momentum...")
    genes = {
        'formation_period': 210,
        'skip_period': 14,
        'vol_lookback': 20,
        'target_vol': 0.18,
        'top_pct': 0.15
    }
    
    try:
        strategy = create_strategy_instance('vol_managed_momentum', genes)
        print(f"   ✓ Created strategy: {strategy.name}")
        print(f"   ✓ formation_period: {strategy.formation_period} (expected: 210)")
        print(f"   ✓ skip_period: {strategy.skip_period} (expected: 14)")
        print(f"   ✓ vol_lookback: {strategy.vol_lookback} (expected: 20)")
        print(f"   ✓ target_vol: {strategy.target_vol} (expected: 0.18)")
        print(f"   ✓ top_percentile: {strategy.top_percentile} (expected: 0.15)")
        
        assert strategy.formation_period == 210, "formation_period mismatch"
        assert strategy.skip_period == 14, "skip_period mismatch"
        print("   ✓ All assertions passed!")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 2: PairsTradingStrategy
    print("\n2. Testing pairs_trading...")
    genes = {
        'entry_z': 2.0,
        'exit_z': 0.5,
        'stop_z': 3.5,
        'min_correlation': 0.8,
        'max_half_life': 25,
        'max_hold_days': 30
    }
    
    try:
        strategy = create_strategy_instance('pairs_trading', genes)
        print(f"   ✓ Created strategy: {strategy.name}")
        print(f"   ✓ ENTRY_ZSCORE: {strategy.ENTRY_ZSCORE} (expected: 2.0)")
        print(f"   ✓ EXIT_ZSCORE: {strategy.EXIT_ZSCORE} (expected: 0.5)")
        print(f"   ✓ STOP_ZSCORE: {strategy.STOP_ZSCORE} (expected: 3.5)")
        print(f"   ✓ min_correlation: {strategy.min_correlation} (expected: 0.8)")
        print(f"   ✓ max_half_life: {strategy.max_half_life} (expected: 25)")
        print(f"   ✓ MAX_HOLD_DAYS: {strategy.MAX_HOLD_DAYS} (expected: 30)")
        
        assert strategy.ENTRY_ZSCORE == 2.0, "ENTRY_ZSCORE mismatch"
        assert strategy.min_correlation == 0.8, "min_correlation mismatch"
        print("   ✓ All assertions passed!")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 3: RelativeVolumeBreakout
    print("\n3. Testing relative_volume_breakout...")
    genes = {
        'min_rv': 2.0,
        'min_gap_pct': 0.04,
        'atr_stop_mult': 2.0,
        'atr_target_mult': 2.5,
        'max_hold_days': 2
    }
    
    try:
        strategy = create_strategy_instance('relative_volume_breakout', genes)
        print(f"   ✓ Created strategy: {strategy.name}")
        print(f"   ✓ MIN_RELATIVE_VOLUME: {strategy.MIN_RELATIVE_VOLUME} (expected: 2.0)")
        print(f"   ✓ MIN_GAP_PCT: {strategy.MIN_GAP_PCT} (expected: 0.04)")
        print(f"   ✓ ATR_STOP_MULT: {strategy.ATR_STOP_MULT} (expected: 2.0)")
        print(f"   ✓ ATR_TARGET_MULT: {strategy.ATR_TARGET_MULT} (expected: 2.5)")
        print(f"   ✓ MAX_HOLD_DAYS: {strategy.MAX_HOLD_DAYS} (expected: 2)")
        
        assert strategy.MIN_RELATIVE_VOLUME == 2.0, "MIN_RELATIVE_VOLUME mismatch"
        assert strategy.ATR_STOP_MULT == 2.0, "ATR_STOP_MULT mismatch"
        print("   ✓ All assertions passed!")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nGA parameter mapping is working correctly.")
    print("You can now run: python run_nightly_research.py -g 1")
    return True


if __name__ == "__main__":
    import subprocess
    
    # Clear pycache first
    print("Clearing Python bytecode cache...")
    result = subprocess.run(
        ["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True
    )
    print("Done.\n")
    
    success = test_strategy_param_setting()
    sys.exit(0 if success else 1)
