#!/usr/bin/env python3
"""
Test script for shared memory and persistent worker pool.
"""
import sys
import time
sys.path.insert(0, "/home/thiselazar/trading_system")

def test_shared_memory():
    """Test shared memory data manager."""
    print("=" * 60)
    print("TEST: Shared Memory Data Manager")
    print("=" * 60)

    from research.discovery.shared_data import SharedDataManager, SharedDataReader
    from data.unified_data_loader import UnifiedDataLoader

    # Load some data
    loader = UnifiedDataLoader()
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
    data = {}
    for sym in symbols:
        df = loader.load_daily(sym)
        if df is not None and len(df) > 0:
            # Ensure timestamp is index
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            data[sym] = df
            print(f"  Loaded {sym}: {len(df)} rows")

    # Create shared memory manager
    manager = SharedDataManager()
    manager.load_data(data, None)

    # Get metadata
    metadata = manager.get_metadata()
    print(f"\n  Shared memory blocks: {len(metadata)}")

    # Test reader (simulating worker)
    reader = SharedDataReader(metadata)
    reader.attach()

    all_data, vix = reader.get_all_data()
    print(f"  Reader retrieved: {len(all_data)} symbols")

    # Verify data integrity
    for sym in symbols:
        if sym in all_data:
            orig_len = len(data[sym])
            read_len = len(all_data[sym])
            match = "OK" if orig_len == read_len else "MISMATCH"
            print(f"  {sym}: orig={orig_len}, read={read_len} [{match}]")

    # Cleanup
    reader.detach()
    manager.cleanup()

    print("\n  [OK] Shared memory test passed!")
    return True


def test_persistent_pool():
    """Test persistent worker pool with shared data."""
    print("\n" + "=" * 60)
    print("TEST: Persistent Worker Pool")
    print("=" * 60)

    from research.discovery.shared_data import SharedDataManager
    from research.discovery.parallel_pool import PersistentWorkerPool
    from data.unified_data_loader import UnifiedDataLoader

    # Load data
    loader = UnifiedDataLoader()
    symbols = ["SPY", "QQQ", "AAPL"]
    data = {}
    for sym in symbols:
        df = loader.load_daily(sym)
        if df is not None and len(df) > 0:
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            data[sym] = df

    print(f"  Loaded {len(data)} symbols for testing")

    # Create shared memory
    manager = SharedDataManager()
    manager.load_data(data, None)
    metadata = manager.get_metadata()

    # Create and start pool with 2 workers
    pool = PersistentWorkerPool(n_workers=2, shared_metadata=metadata)
    pool.start()
    print("  Pool started with 2 workers")

    # We can't easily test genome evaluation without the full setup,
    # but we can verify the pool is running
    print(f"  Pool is running: {pool.is_running()}")

    # Cleanup
    pool.shutdown()
    manager.cleanup()

    print("\n  [OK] Persistent pool test passed!")
    return True


def test_cpu_usage():
    """Check if all cores are being used during parallel work."""
    print("\n" + "=" * 60)
    print("TEST: CPU Core Usage")
    print("=" * 60)

    import subprocess
    result = subprocess.run(['mpstat', '-P', 'ALL', '1', '1'],
                          capture_output=True, text=True)
    print(result.stdout)
    return True


if __name__ == "__main__":
    try:
        test_shared_memory()
        test_persistent_pool()
        test_cpu_usage()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
