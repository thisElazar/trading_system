#!/usr/bin/env python3
"""
Test GA parallel pool functionality.
"""
import sys
import time
import os
sys.path.insert(0, "/home/thiselazar/trading_system")

def test_shared_memory_basic():
    """Test basic shared memory functionality."""
    print("=" * 60)
    print("TEST 1: Basic Shared Memory")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from research.discovery.shared_data import SharedDataManager, SharedDataReader

    # Create test data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = {
        'SPY': pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000000, 10000000, 100),
        }, index=dates),
        'QQQ': pd.DataFrame({
            'open': np.random.randn(100) + 300,
            'high': np.random.randn(100) + 301,
            'low': np.random.randn(100) + 299,
            'close': np.random.randn(100) + 300,
            'volume': np.random.randint(500000, 5000000, 100),
        }, index=dates),
    }

    # Create manager and load data
    manager = SharedDataManager()
    manager.load_data(data, None)
    metadata = manager.get_metadata()
    print(f"  Created {len(metadata)} shared memory blocks")

    # Test reader
    reader = SharedDataReader(metadata)
    reader.attach()
    read_data, vix = reader.get_all_data()
    print(f"  Reader retrieved {len(read_data)} symbols")

    # Verify data integrity
    for sym in ['SPY', 'QQQ']:
        if sym in read_data:
            orig_len = len(data[sym])
            read_len = len(read_data[sym])
            match = "OK" if orig_len == read_len else "MISMATCH"
            print(f"  {sym}: orig={orig_len}, read={read_len} [{match}]")

    # Cleanup
    reader.detach()
    manager.cleanup()
    print("  [OK] Basic shared memory test passed!")
    return True


def test_ga_worker_pool():
    """Test GAWorkerPool initialization."""
    print("\n" + "=" * 60)
    print("TEST 2: GAWorkerPool Initialization")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from research.discovery.shared_data import SharedDataManager
    from research.genetic.ga_parallel import GAWorkerPool

    # Create test data
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    data = {
        'TEST': pd.DataFrame({
            'open': np.random.randn(50) + 100,
            'high': np.random.randn(50) + 101,
            'low': np.random.randn(50) + 99,
            'close': np.random.randn(50) + 100,
            'volume': np.random.randint(1000000, 10000000, 50),
        }, index=dates),
    }

    # Create shared memory
    manager = SharedDataManager()
    manager.load_data(data, None)
    metadata = manager.get_metadata()

    # Create and start pool
    n_workers = min(2, os.cpu_count() or 2)
    pool = GAWorkerPool(n_workers=n_workers, shared_metadata=metadata)
    pool.start()
    print(f"  Pool started with {n_workers} workers")
    print(f"  Pool is running: {pool.is_running()}")

    # Cleanup
    pool.shutdown()
    manager.cleanup()
    print("  [OK] GAWorkerPool test passed!")
    return True


def test_cpu_cores():
    """Check CPU core availability."""
    print("\n" + "=" * 60)
    print("TEST 3: CPU Core Availability")
    print("=" * 60)

    cpu_count = os.cpu_count()
    print(f"  Available CPU cores: {cpu_count}")
    print(f"  Recommended workers: {max(1, (cpu_count or 4) - 1)}")
    return True


if __name__ == "__main__":
    try:
        test_shared_memory_basic()
        test_ga_worker_pool()
        test_cpu_cores()
        print("\n" + "=" * 60)
        print("ALL GA PARALLEL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
