"""
Test Fixtures for the Trading System
=====================================

This directory contains static test data files and fixtures.

Contents:
    - sample_ohlcv.parquet - Sample OHLCV price data
    - sample_signals.json - Sample trading signals
    - sample_config.yaml - Sample configuration files
    - test_universe.csv - Small test universe of symbols

Usage:
    from pathlib import Path
    FIXTURES_DIR = Path(__file__).parent
    sample_data = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")

Note: For programmatically generated test data, use the fixtures
defined in conftest.py (e.g., sample_ohlcv_data, sample_multi_stock_data).
This directory is for static data files that don't change between tests.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def get_fixture_path(filename: str) -> Path:
    """Get the path to a fixture file."""
    path = FIXTURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {filename}")
    return path
