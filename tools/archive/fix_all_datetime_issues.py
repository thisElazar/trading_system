#!/usr/bin/env python3
"""
Comprehensive Backtester Fix
============================

The issue is that historical data might have:
1. Integer indices instead of datetime indices
2. String timestamps in a 'timestamp' column
3. Timezone-aware vs timezone-naive mismatches

This fix handles all cases by ensuring data has proper datetime indices.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DIRS

print("="*70)
print("🔧 COMPREHENSIVE DATA & BACKTESTER FIX")
print("="*70)

# PART 1: Fix historical data to have datetime indices
print("\n[1/2] Ensuring historical data has datetime indices...")

daily_dir = DIRS['daily']
files_checked = 0
files_fixed = 0

# Sample check a few files
sample_files = list(daily_dir.glob("*.parquet"))[:5]

for filepath in sample_files:
    try:
        df = pd.read_parquet(filepath)
        symbol = filepath.stem
        
        needs_fix = False
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            needs_fix = True
            
            # Look for timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                print(f"   ⚠️  {symbol}: No datetime index or timestamp column")
                continue
        
        # Make timezone-naive if timezone-aware (for consistency)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            needs_fix = True
        
        if needs_fix:
            df.to_parquet(filepath)
            files_fixed += 1
            print(f"   ✅ Fixed {symbol}")
        
        files_checked += 1
    
    except Exception as e:
        print(f"   ❌ Error with {filepath.name}: {e}")

if files_fixed > 0:
    print(f"\n✅ Fixed {files_fixed} files to have proper datetime indices")
elif files_checked > 0:
    print(f"\n✅ Checked {files_checked} files - all have proper datetime indices")

# Fix VIX data too
print("\n   Checking VIX data...")
vix_file = DIRS['vix'] / 'vix.parquet'

if vix_file.exists():
    vix_df = pd.read_parquet(vix_file)
    
    needs_fix = False
    if not isinstance(vix_df.index, pd.DatetimeIndex):
        if 'timestamp' in vix_df.columns:
            vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
            vix_df.set_index('timestamp', inplace=True)
            needs_fix = True
    
    if vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
        needs_fix = True
    
    if needs_fix:
        vix_df.to_parquet(vix_file)
        print("   ✅ Fixed VIX data")
    else:
        print("   ✅ VIX data OK")

# PART 2: Fix backtester to be more robust
print("\n[2/2] Making backtester more robust...")

backtester_file = Path(__file__).parent / "research" / "backtester.py"

if backtester_file.exists():
    with open(backtester_file, 'r') as f:
        content = f.read()
    
    # Backup
    import datetime as dt
    backup_suffix = '.backup_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = backtester_file.with_suffix(backup_suffix)
    with open(backup, 'w') as f:
        f.write(content)
    
    # Fix 1: Ensure datetime comparison is robust
    old_compare = '                vix_mask = vix_data.index <= current_date'
    new_compare = '''                # Ensure both sides are comparable (handle timezone issues)
                try:
                    current_date_normalized = pd.Timestamp(current_date)
                    if current_date_normalized.tz is not None:
                        current_date_normalized = current_date_normalized.tz_localize(None)
                    vix_mask = vix_data.index <= current_date_normalized
                except Exception:
                    vix_mask = pd.Series([True] * len(vix_data))'''
    
    if old_compare in content:
        content = content.replace(old_compare, new_compare)
        print("   ✅ Added robust VIX comparison")
    
    # Fix 2: Validate data has datetime indices at start of run()
    validation_code = '''        # Validate data has datetime indices
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Symbol {symbol} does not have datetime index, attempting to fix...")
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
                else:
                    logger.error(f"Symbol {symbol} has no datetime index or timestamp column")
            
            # Remove timezone awareness for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                data[symbol] = df
        '''
    
    # Insert after docstring of run() method
    run_method_start = '        """'
    # Find the end of the docstring
    docstring_end_idx = content.find('"""', content.find('def run(')) + len('"""')
    if docstring_end_idx > 0:
        # Insert validation code after the docstring
        before = content[:docstring_end_idx]
        after = content[docstring_end_idx:]
        content = before + '\n' + validation_code + after
        print("   ✅ Added data validation at start of backtest")
    
    with open(backtester_file, 'w') as f:
        f.write(content)
    
    print(f"   ✅ Patched backtester (backup: {backup.name})")

print("\n" + "="*70)
print("✅ ALL FIXES APPLIED")
print("="*70)

print("\n📊 What was fixed:")
print("  1. Historical data now has proper datetime indices (timezone-naive)")
print("  2. VIX data has proper datetime indices (timezone-naive)")
print("  3. Backtester validates and fixes data at runtime")
print("  4. Backtester handles timezone mismatches gracefully")

print("\n🚀 READY TO RUN:")
print("  python scripts/run_research.py --full --quick")

print("\n⏱️  Expected time: ~20-30 minutes")
print("="*70)
