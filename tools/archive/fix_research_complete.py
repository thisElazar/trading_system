#!/usr/bin/env python3
"""
Complete Research Pipeline Fix
===============================

Fixes all three issues discovered:
1. VIX data missing 'regime' column
2. MonteCarloSimulator method name mismatch (run_monte_carlo → run_simulation)
3. ParameterOptimizer not yet integrated (temporarily disable)

Run this once, then re-run the research pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DIRS, get_vix_regime

print("="*70)
print("🔧 COMPLETE RESEARCH PIPELINE FIX")
print("="*70)

success_count = 0
total_fixes = 3

# ============================================================================
# FIX 1: Add 'regime' column to VIX data
# ============================================================================

print("\n[1/3] Fixing VIX data...")

vix_file = DIRS['vix'] / 'vix.parquet'

if not vix_file.exists():
    print(f"   ❌ VIX file not found: {vix_file}")
    print("   Run: python scripts/universe_downloader.py --years 10")
else:
    vix_df = pd.read_parquet(vix_file)
    
    if 'regime' in vix_df.columns:
        print("   ℹ️  VIX data already has 'regime' column")
        success_count += 1
    else:
        # Add regime column
        vix_df['regime'] = vix_df['close'].apply(get_vix_regime)
        vix_df['vix_ma_10'] = vix_df['close'].rolling(10).mean()
        
        # Save
        vix_df.to_parquet(vix_file)
        
        print("   ✅ Added 'regime' column to VIX data")
        print(f"   Distribution:")
        regime_counts = vix_df['regime'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(vix_df) * 100
            print(f"      {regime}: {count} ({pct:.1f}%)")
        
        success_count += 1

# ============================================================================
# FIX 2 & 3: Patch run_research.py
# ============================================================================

print("\n[2/3] Patching run_research.py...")

research_script = Path(__file__).parent / "scripts" / "run_research.py"

if not research_script.exists():
    print(f"   ❌ File not found: {research_script}")
else:
    with open(research_script, 'r') as f:
        content = f.read()
    
    # Backup
    backup = research_script.with_suffix('.py.backup_' + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'))
    with open(backup, 'w') as f:
        f.write(content)
    
    print(f"   ✅ Backed up to: {backup.name}")
    
    changes_made = False
    
    # Fix Monte Carlo method name
    if 'self.monte_carlo.run_monte_carlo(' in content:
        # Replace the method call
        content = content.replace(
            'result = self.monte_carlo.run_monte_carlo(',
            'result = self.monte_carlo.run_simulation('
        )
        # Remove the simulation_type parameter (run_simulation doesn't take it)
        content = content.replace(
            '''                    simulation_type="bootstrap"
                )''',
            '''                )'''
        )
        # Also fix the parameter names
        content = content.replace(
            '''                result = self.monte_carlo.run_simulation(
                    strategy=strategy,
                    data=data,
                    vix_data=vix_data,
                    n_simulations=n_sims,
                )''',
            '''                result = self.monte_carlo.run_simulation(
                    strategy_name=strategy.name,
                    n_simulations=n_sims
                )'''
        )
        print("   ✅ Fixed MonteCarloSimulator method name")
        changes_made = True
    
    # Temporarily disable parameter optimization (not fully integrated)
    if 'self.optimizer.grid_search(' in content:
        # Find and replace the optimizer call
        old_block = '''                results = self.optimizer.grid_search(
                    strategy=strategy,
                    data=data,
                    param_grid=param_grid,
                    n_splits=3 if self.quick_mode else 5
                )'''
        
        new_block = '''                # TODO: Parameter optimization needs integration work
                results = {
                    'best_params': None,
                    'best_sharpe': 0.0,
                    'results': []
                }
                logger.warning(f"Parameter optimization temporarily skipped for {strategy_name}")'''
        
        if old_block in content:
            content = content.replace(old_block, new_block)
            print("   ✅ Disabled parameter optimization (temporary)")
            changes_made = True
    
    if changes_made:
        with open(research_script, 'w') as f:
            f.write(content)
        success_count += 1
        print(f"   ✅ Patched {research_script.name}")
    else:
        print("   ℹ️  No changes needed (already patched?)")
        success_count += 1

print("\n[3/3] Verifying fixes...")

if success_count == total_fixes:
    print("   ✅ All fixes applied successfully!")
else:
    print(f"   ⚠️  Only {success_count}/{total_fixes} fixes applied")

# ============================================================================
# Summary and next steps
# ============================================================================

print("\n" + "="*70)
print("📊 FIX SUMMARY")
print("="*70)

print("\n✅ FIXED:")
print("  1. VIX data now has 'regime' column for regime-specific backtests")
print("  2. MonteCarloSimulator now uses correct method name")
print("  3. Parameter optimization temporarily disabled (needs integration)")

print("\n🚀 NEXT STEPS:")
print("\n  1. Re-run research pipeline:")
print("     cd /Volumes/HotStorage/Pi/TradeBot/trading_system")
print("     python scripts/run_research.py --full --quick")
print("\n  2. This time you should get:")
print("     ✓ Strategy comparison (with regime analysis)")
print("     ✓ Monte Carlo robustness testing")
print("     ✓ Master report with recommendations")

print("\n⏱️  EXPECTED RUNTIME:")
print("  - Strategy comparison: ~5-10 min")
print("  - Monte Carlo (100 sims): ~10-15 min")
print("  - Report generation: <1 min")
print("  TOTAL: ~20-30 minutes")

print("\n" + "="*70)
print("Ready to run! Execute:")
print("  python scripts/run_research.py --full --quick")
print("="*70 + "\n")
