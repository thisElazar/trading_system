#!/usr/bin/env python3
"""
Smoke Test 1: Data Infrastructure
Validates database integrity, data coverage, and query performance.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_data_infrastructure():
    print("=" * 60)
    print("TEST 1: Data Infrastructure")
    print("=" * 60)
    
    start_time = time.time()
    results = {
        "database_exists": False,
        "integrity_ok": False,
        "symbol_count": 0,
        "date_range": None,
        "query_time_ms": 0,
        "parquet_count": 0,
        "spy_parquet_rows": 0,
    }
    
    try:
        import sqlite3
        import pandas as pd
        
        db_dir = PROJECT_ROOT / "db"
        data_dir = PROJECT_ROOT / "data"
        
        print(f"\nChecking database directory: {db_dir}")
        db_files = list(db_dir.glob("*.db")) if db_dir.exists() else []
        print(f"  Found {len(db_files)} database files: {[f.name for f in db_files]}")
        
        research_db = db_dir / "research.db"
        if research_db.exists():
            results["database_exists"] = True
            conn = sqlite3.connect(research_db)
            cursor = conn.execute("PRAGMA integrity_check;")
            integrity = cursor.fetchone()[0]
            results["integrity_ok"] = integrity == "ok"
            print(f"  research.db integrity: {integrity}")
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            print(f"  Tables: {tables}")
            conn.close()
        
        parquet_dir = data_dir / "historical" / "daily"
        print(f"\nChecking parquet directory: {parquet_dir}")
        
        if parquet_dir.exists():
            parquet_files = list(parquet_dir.glob("*.parquet"))
            results["parquet_count"] = len(parquet_files)
            print(f"  Found {len(parquet_files)} parquet files")
            
            spy_path = parquet_dir / "SPY.parquet"
            if spy_path.exists():
                query_start = time.time()
                df = pd.read_parquet(spy_path)
                results["query_time_ms"] = (time.time() - query_start) * 1000
                results["spy_parquet_rows"] = len(df)
                if "timestamp" in df.columns:
                    min_date = df["timestamp"].min()
                    max_date = df["timestamp"].max()
                    results["date_range"] = f"{min_date} to {max_date}"
                print(f"  SPY.parquet: {len(df)} rows")
                print(f"  Date range: {results['date_range']}")
                print(f"  Query time: {results['query_time_ms']:.1f}ms")
        
        results["symbol_count"] = results["parquet_count"]
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, results, time.time() - start_time
    
    duration = time.time() - start_time
    passed = (
        results["database_exists"] and
        results["integrity_ok"] and
        results["symbol_count"] >= 500 and
        results["query_time_ms"] < 1000
    )
    
    print(f"\nKey Metrics:")
    print(f"  - Database exists: {results['database_exists']}")
    print(f"  - Integrity OK: {results['integrity_ok']}")
    print(f"  - Symbol count: {results['symbol_count']}")
    print(f"  - Date range: {results['date_range']}")
    print(f"  - Query time: {results['query_time_ms']:.1f}ms")
    print(f"  - SPY rows: {results['spy_parquet_rows']}")
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)
    
    return passed, results, duration

if __name__ == "__main__":
    passed, results, duration = test_data_infrastructure()
    sys.exit(0 if passed else 1)
