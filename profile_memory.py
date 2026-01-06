#!/usr/bin/env python3
"""Quick sanity test with 1 year of data."""

import psutil
from datetime import datetime

def get_mem():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Quick sanity test (1 year, 10 symbols)")
print("="*50)

from config import DIRS
from data.cached_data_manager import CachedDataManager
import pandas as pd
from pathlib import Path

dm = CachedDataManager()
symbols = sorted(dm.get_available_symbols())[:10]
dm.load_all(symbols=symbols)
data = dm.cache.copy()

vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
vix_data = pd.read_parquet(vix_path) if vix_path.exists() else None

from research.backtester import Backtester
from strategies.vol_managed_momentum_v2 import VolManagedMomentumV2

bt = Backtester()
strategy = VolManagedMomentumV2()

result = bt.run(strategy, data, 
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
    vix_data=vix_data)

print(f"Return: {result.total_return:.2%}")
print(f"Memory: {get_mem():.0f} MB")
print("SUCCESS!")
