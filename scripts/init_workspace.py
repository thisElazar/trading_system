#!/usr/bin/env python3
"""
Initialize Trading System Workspace
====================================
First-run setup script.

Usage:
    python scripts/init_workspace.py [--force]
    
This script:
1. Creates all required directories
2. Initializes all SQLite databases
3. Creates reference data files
4. Validates API credentials
5. Downloads initial historical data (optional)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_ROOT, DIRS, DATABASES, ensure_dirs,
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    STRATEGIES, get_enabled_strategies
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_directories() -> bool:
    """Create all required directories."""
    logger.info("Creating directory structure...")
    
    try:
        ensure_dirs()
        
        created = 0
        for name, path in DIRS.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created += 1
            logger.debug(f"  ✓ {name}: {path}")
        
        logger.info(f"Created {created} directories (total: {len(DIRS)})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False


def init_databases() -> bool:
    """Initialize all SQLite databases."""
    logger.info("Initializing databases...")
    
    try:
        from data.storage.db_manager import get_db
        
        db = get_db()
        
        for name, path in DATABASES.items():
            if path.exists():
                logger.debug(f"  ✓ {name}: exists")
            else:
                logger.info(f"  Creating {name}...")
        
        logger.info(f"Initialized {len(DATABASES)} databases")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        return False


def create_reference_data() -> bool:
    """Create reference data files."""
    logger.info("Creating reference data...")
    
    ref_dir = DIRS["reference"]
    
    try:
        # S&P 500 sector ETFs
        sectors = {
            "XLB": {"name": "Materials", "type": "cyclical"},
            "XLC": {"name": "Communication Services", "type": "growth"},
            "XLE": {"name": "Energy", "type": "cyclical"},
            "XLF": {"name": "Financials", "type": "cyclical"},
            "XLI": {"name": "Industrials", "type": "cyclical"},
            "XLK": {"name": "Technology", "type": "growth"},
            "XLP": {"name": "Consumer Staples", "type": "defensive"},
            "XLRE": {"name": "Real Estate", "type": "rate_sensitive"},
            "XLU": {"name": "Utilities", "type": "defensive"},
            "XLV": {"name": "Healthcare", "type": "defensive"},
            "XLY": {"name": "Consumer Discretionary", "type": "cyclical"},
        }
        
        with open(ref_dir / "sectors.json", "w") as f:
            json.dump(sectors, f, indent=2)
        
        # Core universe for initial testing
        core_universe = {
            "indices": ["SPY", "QQQ", "IWM", "DIA"],
            "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
            "sector_etfs": list(sectors.keys()),
            "bond_etfs": ["TLT", "IEF", "SHY", "BND", "AGG"],
            "volatility": ["VXX", "VIXY", "UVXY"],
            "commodities": ["GLD", "SLV", "USO", "UNG"],
        }
        
        with open(ref_dir / "core_universe.json", "w") as f:
            json.dump(core_universe, f, indent=2)
        
        # Pairs trading universe (initial candidates)
        pairs_universe = {
            "technology": [
                ["AAPL", "MSFT"],
                ["GOOGL", "META"],
                ["NVDA", "AMD"],
            ],
            "financials": [
                ["JPM", "BAC"],
                ["GS", "MS"],
                ["V", "MA"],
            ],
            "healthcare": [
                ["JNJ", "PFE"],
                ["UNH", "CVS"],
                ["ABBV", "MRK"],
            ],
            "energy": [
                ["XOM", "CVX"],
                ["COP", "EOG"],
            ],
            "consumer": [
                ["HD", "LOW"],
                ["MCD", "SBUX"],
                ["NKE", "LULU"],
            ],
        }
        
        with open(ref_dir / "pairs_universe.json", "w") as f:
            json.dump(pairs_universe, f, indent=2)
        
        logger.info(f"Created reference data files in {ref_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create reference data: {e}")
        return False


def validate_api_credentials() -> bool:
    """Validate Alpaca API credentials."""
    logger.info("Validating API credentials...")
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.warning("API credentials not set")
        logger.info("  Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        logger.info("  Or create a .env file in the project root")
        return False
    
    if ALPACA_API_KEY.startswith("your_") or len(ALPACA_API_KEY) < 10:
        logger.warning("API credentials appear to be placeholder values")
        return False
    
    try:
        from alpaca.trading.client import TradingClient
        
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = client.get_account()
        
        logger.info(f"  ✓ Connected to Alpaca (Paper Trading)")
        logger.info(f"  Account value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"  Buying power: ${float(account.buying_power):,.2f}")
        
        return True
        
    except ImportError:
        logger.warning("alpaca-py not installed - run: pip install alpaca-py")
        return False
    except Exception as e:
        logger.error(f"Failed to validate API credentials: {e}")
        return False


def download_initial_data(symbols: list = None) -> bool:
    """Download initial historical data."""
    logger.info("Downloading initial historical data...")
    
    if symbols is None:
        # Load core universe
        ref_file = DIRS["reference"] / "core_universe.json"
        if ref_file.exists():
            with open(ref_file) as f:
                universe = json.load(f)
            symbols = (
                universe.get("indices", []) +
                universe.get("mega_cap", [])[:5] +
                universe.get("sector_etfs", [])[:5]
            )
        else:
            symbols = ["SPY", "QQQ", "IWM"]
    
    try:
        from data.fetchers.daily_bars import DailyBarsFetcher
        
        fetcher = DailyBarsFetcher()
        results = fetcher.fetch_symbols(symbols, delay=0.2)
        
        logger.info(f"Downloaded data for {len(results)}/{len(symbols)} symbols")
        return len(results) > 0
        
    except ImportError as e:
        logger.warning(f"Could not import fetcher: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False


def create_env_template() -> bool:
    """Create .env template file."""
    env_path = DATA_ROOT / ".env.template"
    
    template = """# Trading System Environment Variables
# Copy this file to .env and fill in your values

# Alpaca API (Paper Trading)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Optional: Override data root (defaults to auto-detect)
# TRADING_SYSTEM_ROOT=/path/to/trading_system
"""
    
    try:
        with open(env_path, "w") as f:
            f.write(template)
        logger.info(f"Created .env template at {env_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create .env template: {e}")
        return False


def print_summary():
    """Print summary of workspace configuration."""
    print("\n" + "=" * 60)
    print("TRADING SYSTEM WORKSPACE SUMMARY")
    print("=" * 60)
    
    print(f"\nData Root: {DATA_ROOT}")
    print(f"Root exists: {DATA_ROOT.exists()}")
    
    print("\nDirectories:")
    for name, path in sorted(DIRS.items()):
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path.relative_to(DATA_ROOT) if path.is_relative_to(DATA_ROOT) else path}")
    
    print("\nDatabases:")
    for name, path in DATABASES.items():
        status = "✓" if path.exists() else "✗"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else ""
        print(f"  {status} {name}: {path.name} {size}")
    
    print("\nEnabled Strategies:")
    for name in get_enabled_strategies():
        config = STRATEGIES[name]
        print(f"  - {name} ({config['allocation_pct']*100:.0f}% allocation)")
    
    print("\nAPI Status:")
    if ALPACA_API_KEY and len(ALPACA_API_KEY) > 10:
        print(f"  ✓ Alpaca API key configured (ends with ...{ALPACA_API_KEY[-4:]})")
    else:
        print("  ✗ Alpaca API key not configured")
    
    # Check for reference data
    ref_dir = DIRS["reference"]
    ref_files = list(ref_dir.glob("*.json")) if ref_dir.exists() else []
    print(f"\nReference Data: {len(ref_files)} files")
    for f in ref_files:
        print(f"  - {f.name}")
    
    # Check for historical data
    daily_dir = DIRS["daily"]
    parquet_files = list(daily_dir.glob("*.parquet")) if daily_dir.exists() else []
    print(f"\nHistorical Data: {len(parquet_files)} symbols")
    if parquet_files:
        print(f"  Latest: {sorted(parquet_files, key=lambda x: x.stat().st_mtime)[-1].stem}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize trading system workspace"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force recreation of all files"
    )
    parser.add_argument(
        "--skip-data", "-s",
        action="store_true",
        help="Skip downloading historical data"
    )
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate existing setup"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TRADING SYSTEM WORKSPACE INITIALIZATION")
    print("=" * 60)
    print(f"\nData Root: {DATA_ROOT}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    if args.validate_only:
        print_summary()
        return 0
    
    # Track results
    results = {}
    
    # Step 1: Create directories
    results["directories"] = create_directories()
    
    # Step 2: Create .env template
    results["env_template"] = create_env_template()
    
    # Step 3: Initialize databases
    results["databases"] = init_databases()
    
    # Step 4: Create reference data
    results["reference_data"] = create_reference_data()
    
    # Step 5: Validate API credentials
    results["api_validation"] = validate_api_credentials()
    
    # Step 6: Download initial data (if API is valid and not skipped)
    if not args.skip_data and results["api_validation"]:
        results["initial_data"] = download_initial_data()
    else:
        results["initial_data"] = None
        if args.skip_data:
            logger.info("Skipping data download (--skip-data)")
        elif not results["api_validation"]:
            logger.info("Skipping data download (API not configured)")
    
    # Print summary
    print_summary()
    
    # Final status
    print("\nInitialization Results:")
    all_passed = True
    for step, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"  {status}: {step}")
    
    if all_passed:
        print("\n✓ Workspace initialization complete!")
        print("\nNext steps:")
        print("  1. Configure API keys in .env file")
        print("  2. Run: python data/fetchers/daily_bars.py SPY QQQ IWM")
        print("  3. Run: python strategies/vol_managed_momentum.py")
        return 0
    else:
        print("\n⚠ Some steps failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
