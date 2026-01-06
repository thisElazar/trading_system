"""
Check actual data ranges in the parquet files.
"""
import pandas as pd
from pathlib import Path

data_dir = Path("data/historical/daily")

print("="*60)
print("ACTUAL DATE RANGES IN PARQUET FILES")
print("="*60)

# Check key symbols
symbols = ['AAPL', 'MSFT', 'SPY', 'GOOGL', 'JPM', 'XOM', 'JNJ']

for symbol in symbols:
    filepath = data_dir / f"{symbol}.parquet"
    if filepath.exists():
        df = pd.read_parquet(filepath)
        
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        elif df.index.name == 'timestamp':
            dates = pd.to_datetime(df.index)
        else:
            dates = df.index
        
        print(f"{symbol}: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')} ({len(dates)} bars)")

# Check some newer stocks that might be limiting intersection
print("\nNewer stocks (potential intersection limiters):")
newer = ['ABNB', 'RIVN', 'LCID', 'CRWD', 'DDOG']
for symbol in newer:
    filepath = data_dir / f"{symbol}.parquet"
    if filepath.exists():
        df = pd.read_parquet(filepath)
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = df.index
        print(f"{symbol}: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')} ({len(dates)} bars)")

# Count stocks by data length
print("\n" + "="*60)
print("DISTRIBUTION OF DATA LENGTH")
print("="*60)

lengths = {}
for f in data_dir.glob("*.parquet"):
    df = pd.read_parquet(f)
    years = len(df) / 252
    bucket = f"{int(years)}+ years"
    lengths[bucket] = lengths.get(bucket, 0) + 1

for bucket in sorted(lengths.keys()):
    print(f"  {bucket}: {lengths[bucket]} stocks")
