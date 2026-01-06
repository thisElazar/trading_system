#!/usr/bin/env python3
"""
Fix Data History Issue - Update config and re-download with 10 years

This script:
1. Updates config.py to use 10 years of history
2. Downloads data for all symbols with proper lookback
3. Validates that we have enough data for momentum strategies

Run this before running the research pipeline!
"""

import sys
from pathlib import Path
import subprocess
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("🔧 FIXING DATA HISTORY CONFIGURATION")
print("="*60)

# Step 1: Update config.py
print("\n1. Updating config.py...")
config_path = Path(__file__).parent / "config.py"

if not config_path.exists():
    print(f"❌ Config file not found: {config_path}")
    sys.exit(1)

# Read current config
with open(config_path, 'r') as f:
    content = f.read()

# Update HISTORICAL_YEARS
old_pattern = r'HISTORICAL_YEARS = 2\s+# Years of daily data to fetch'
new_value = 'HISTORICAL_YEARS = 10            # Years of daily data (need long history for momentum)'

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_value, content)
    
    # Save backup
    backup_path = config_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated HISTORICAL_YEARS = 10")
    print(f"   Backup saved to: {backup_path}")
else:
    print(f"⚠️  Pattern not found - config may already be updated or different format")

# Step 2: Show what this means
print("\n" + "="*60)
print("📊 DATA HISTORY REQUIREMENTS")
print("="*60)
print("\nWhy 10 years?")
print("  • 12-month momentum needs 252 days just for formation")
print("  • Need multiple market cycles for validation")
print("  • 10 years covers 2015-2024:")
print("    - 2018 correction")
print("    - 2020 COVID crash & recovery")
print("    - 2022 bear market")
print("    - 2023-2024 recovery")
print("  • Gives ~2,500 trading days for robust backtesting")

print("\nWhat gets downloaded:")
print("  • ~800 symbols (S&P 500 + NASDAQ-100 + Russell 2000 sample)")
print("  • ~2,500 trading days per symbol")
print("  • ~2 million total bars")
print("  • ~500 MB total storage")
print("  • Download time: ~30-45 minutes (with rate limiting)")

# Step 3: Run download
print("\n" + "="*60)
print("📥 DOWNLOADING DATA")
print("="*60)

response = input("\n⚠️  This will download ~500 MB of data and take 30-45 minutes. Continue? (yes/no): ")

if response.lower() not in ['yes', 'y']:
    print("\n❌ Download cancelled. You can run it manually with:")
    print(f"   python scripts/universe_downloader.py --years 10")
    print("\nOr run this script again when ready.")
    sys.exit(0)

# Run the downloader
print("\n🚀 Starting download...")
print("=" * 60)

downloader_script = Path(__file__).parent / "scripts" / "universe_downloader.py"

if not downloader_script.exists():
    print(f"❌ Downloader script not found: {downloader_script}")
    sys.exit(1)

try:
    # Run the downloader with 10 years
    result = subprocess.run(
        [sys.executable, str(downloader_script), "--years", "10"],
        cwd=Path(__file__).parent,
        check=True
    )
    
    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*60)
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Download failed: {e}")
    print("\nYou can run it manually:")
    print(f"   cd {Path(__file__).parent}")
    print(f"   python scripts/universe_downloader.py --years 10")
    sys.exit(1)

# Step 4: Validate data
print("\n" + "="*60)
print("🔍 VALIDATING DATA")
print("="*60)

from config import DIRS
import pandas as pd

daily_dir = DIRS['daily']

if not daily_dir.exists():
    print(f"❌ Daily data directory not found: {daily_dir}")
    sys.exit(1)

parquet_files = list(daily_dir.glob("*.parquet"))
print(f"\n📁 Found {len(parquet_files)} symbol files")

if len(parquet_files) == 0:
    print("❌ No data files found! Download may have failed.")
    sys.exit(1)

# Check a few samples
samples = ['AAPL', 'MSFT', 'SPY'] if len(parquet_files) >= 3 else parquet_files[:3]
print(f"\n📊 Sample data validation:")

all_good = True
for filename in samples:
    if isinstance(filename, Path):
        filepath = filename
        symbol = filepath.stem
    else:
        filepath = daily_dir / f"{filename}.parquet"
        symbol = filename
    
    if not filepath.exists():
        continue
    
    df = pd.read_parquet(filepath)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start = df['timestamp'].min()
        end = df['timestamp'].max()
    else:
        start = df.index.min()
        end = df.index.max()
    
    years = (end - start).days / 365.25
    
    print(f"\n  {symbol}:")
    print(f"    Bars: {len(df)}")
    print(f"    Range: {start.date()} to {end.date()}")
    print(f"    Years: {years:.1f}")
    
    # Check if enough for momentum (need 252+ days for formation)
    if len(df) < 252:
        print(f"    ⚠️  WARNING: Not enough data for 12-month momentum!")
        all_good = False
    elif len(df) < 500:
        print(f"    ⚠️  WARNING: Minimal data - backtests will be limited")
        all_good = False
    elif years < 3:
        print(f"    ⚠️  WARNING: Less than 3 years of history")
        all_good = False
    else:
        print(f"    ✅ Sufficient data for backtesting")

# Summary
print("\n" + "="*60)
if all_good:
    print("✅ DATA VALIDATION PASSED")
    print("="*60)
    print("\n🎯 Next Steps:")
    print("   1. Run research pipeline:")
    print("      python scripts/run_research.py --full --quick")
    print("\n   2. Review the master report in:")
    print("      research/backtests/00_MASTER_REPORT_*.md")
    print("\n   3. Based on recommendations, proceed to paper trading")
else:
    print("⚠️  DATA VALIDATION HAD WARNINGS")
    print("="*60)
    print("\nSome symbols don't have full 10-year history.")
    print("This is normal for:")
    print("  • Recently IPO'd companies")
    print("  • ETFs launched in the last 10 years")
    print("\nThe research pipeline will work, but may have reduced sample size.")
    print("You can still proceed with:")
    print("   python scripts/run_research.py --full --quick")

print("\n" + "="*60)

