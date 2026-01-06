#!/usr/bin/env python3
"""
Find and Fix get_symbol_data Bug
=================================
"""

import os
import sys
from pathlib import Path

root = Path("/Volumes/HotStorage/Pi/TradeBot/trading_system")
os.chdir(root)

print("=" * 70)
print("🔍 FINDING get_symbol_data BUG")
print("=" * 70)

# Search all Python files
found_files = []

for pyfile in root.rglob("*.py"):
    if "__pycache__" in str(pyfile):
        continue
    
    try:
        with open(pyfile, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'get_symbol_data' in content:
                found_files.append(pyfile)

                # Show the lines
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if 'get_symbol_data' in line:
                        print(f"\n📁 {pyfile.relative_to(root)}:{i}")
                        print(f"   {line.strip()}")
    except (IOError, OSError, UnicodeDecodeError) as e:
        print(f"  [DEBUG] Could not read {pyfile}: {e}")

if not found_files:
    print("\n❌ No files found calling get_symbol_data()")
    print("\nThe error might be in cached bytecode. Clearing...")
    
    import shutil
    for pycache in root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
        print(f"   Deleted {pycache.relative_to(root)}")
    
    print("\n✅ Cleared Python caches. Try running again.")
else:
    print(f"\n✅ Found {len(found_files)} files")
    print("\nFix: Replace get_symbol_data with get_bars")

print("\n" + "=" * 70)
