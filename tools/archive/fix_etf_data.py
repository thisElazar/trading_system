"""
Fix truncated data for key ETFs and re-download.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Check if we have Alpaca credentials
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("ERROR: Alpaca credentials not configured")
    sys.exit(1)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

# Initialize client
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ETFs that need full history
etfs_to_fix = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 
               'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

data_dir = Path("data/historical/daily")

print("="*60)
print("FIXING TRUNCATED ETF DATA")
print("="*60)

# Target: 10 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10 + 30)  # 10 years + buffer

for symbol in etfs_to_fix:
    filepath = data_dir / f"{symbol}.parquet"
    
    # Check current data
    current_bars = 0
    if filepath.exists():
        df = pd.read_parquet(filepath)
        current_bars = len(df)
    
    print(f"\n{symbol}: Currently {current_bars} bars")
    
    if current_bars >= 2500:
        print(f"  ✓ Already has sufficient data, skipping")
        continue
    
    print(f"  Downloading 10 years of data...")
    
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = client.get_stock_bars(request)
        
        if bars.df.empty:
            print(f"  ✗ No data returned")
            continue
        
        df = bars.df.reset_index()
        
        # Clean up columns
        if 'symbol' in df.columns:
            df = df.drop(columns=['symbol'])
        
        # Rename timestamp column if needed
        if 'timestamp' not in df.columns and df.columns[0] != 'timestamp':
            df = df.rename(columns={df.columns[0]: 'timestamp'})
        
        # Save
        df.to_parquet(filepath, index=False)
        print(f"  ✓ Saved {len(df)} bars ({df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')})")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

for symbol in etfs_to_fix[:5]:  # Check first 5
    filepath = data_dir / f"{symbol}.parquet"
    if filepath.exists():
        df = pd.read_parquet(filepath)
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = df.index
        print(f"{symbol}: {len(df)} bars, {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")

print("\n✓ Done! Re-run the diagnostic to verify momentum signal.")
