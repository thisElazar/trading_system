"""
Shared pytest fixtures for the trading system test suite.

This file is automatically loaded by pytest and provides fixtures
that can be used across all tests.

Fixture Categories:
    - Database fixtures: test_db, test_db_path, clean_db
    - Configuration fixtures: test_config, override_config
    - Market data fixtures: sample_ohlcv_data, sample_multi_stock_data, sample_vix_data
    - Alpaca fixtures: mock_alpaca_client, mock_account_info, mock_position
    - Strategy fixtures: strategy_config, sample_strategy, sample_signals
    - Backtester fixtures: sample_backtest_result, backtester_instance

Design Principles:
    - Fixtures are composable (depend on each other cleanly)
    - Named consistently: test_*, sample_*, mock_*
    - Scoped appropriately (function/module/session)
    - Clean up after themselves
"""

import os
import sys
import tempfile
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment before importing config
os.environ["TRADING_SYSTEM_ROOT"] = str(PROJECT_ROOT)


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def test_db_path(tmp_path) -> Path:
    """Create a temporary database path for testing."""
    return tmp_path / "test_trading.db"


@pytest.fixture
def test_db(test_db_path):
    """Create a temporary SQLite database with schema."""
    import sqlite3

    conn = sqlite3.connect(str(test_db_path))
    cursor = conn.cursor()

    # Create minimal schema for testing
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            strength REAL,
            price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            stop_loss REAL,
            target REAL,
            unrealized_pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            pnl REAL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP
        );
    """)

    conn.commit()
    yield conn
    conn.close()


# =============================================================================
# Market Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')

    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_multi_stock_data(sample_ohlcv_data) -> Dict[str, pd.DataFrame]:
    """Generate OHLCV data for multiple symbols."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    data = {}

    for i, symbol in enumerate(symbols):
        # Slightly different price levels and volatility per stock
        multiplier = 1 + i * 0.5
        df = sample_ohlcv_data.copy()
        df = df * multiplier
        df['volume'] = (df['volume'] * (1 + i * 0.2)).astype(int)
        data[symbol] = df

    return data


@pytest.fixture
def sample_vix_data() -> pd.DataFrame:
    """Generate sample VIX data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')

    # VIX typically mean-reverts around 15-20
    vix = 18 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    vix = np.clip(vix, 10, 50)  # Keep within realistic bounds

    return pd.DataFrame({
        'close': vix,
        'open': vix + np.random.uniform(-0.5, 0.5, len(dates)),
        'high': vix + np.random.uniform(0, 2, len(dates)),
        'low': vix - np.random.uniform(0, 2, len(dates)),
    }, index=dates)


# =============================================================================
# Alpaca Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_account_info():
    """Mock Alpaca account information."""
    return {
        'equity': 100000.0,
        'cash': 50000.0,
        'buying_power': 200000.0,
        'portfolio_value': 100000.0,
        'day_trade_count': 0,
        'pattern_day_trader': False,
        'trading_blocked': False,
    }


@pytest.fixture
def mock_position():
    """Mock Alpaca position."""
    return {
        'symbol': 'AAPL',
        'qty': 100,
        'side': 'long',
        'avg_entry_price': 150.0,
        'market_value': 15500.0,
        'unrealized_pnl': 500.0,
        'unrealized_pnl_pct': 3.33,
        'current_price': 155.0,
    }


@pytest.fixture
def mock_alpaca_client(mock_account_info, mock_position):
    """
    Create a mock Alpaca trading client.

    Note: Full implementation in tests/mocks/mock_alpaca.py (managed by mock agent).
    This fixture provides a basic MagicMock if the full mock is not available.
    """
    try:
        from tests.mocks.mock_alpaca import MockTradingClient
        return MockTradingClient(
            initial_equity=mock_account_info['equity'],
            initial_cash=mock_account_info['cash'],
        )
    except ImportError:
        # Fallback to basic MagicMock if mock module not yet implemented
        mock_client = MagicMock()
        mock_client.get_account.return_value = MagicMock(**mock_account_info)
        mock_client.get_all_positions.return_value = [MagicMock(**mock_position)]
        mock_client.submit_order.return_value = MagicMock(
            id='test-order-123',
            status='filled',
            filled_qty=100,
            filled_avg_price=150.0,
        )
        return mock_client


@pytest.fixture
def mock_alpaca_data_client():
    """Create a mock Alpaca data client for historical data."""
    mock_client = MagicMock()

    # Mock get_stock_bars to return sample data
    def mock_get_bars(request):
        result = MagicMock()
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        result.df = pd.DataFrame({
            'open': np.random.uniform(145, 155, 10),
            'high': np.random.uniform(150, 160, 10),
            'low': np.random.uniform(140, 150, 10),
            'close': np.random.uniform(145, 155, 10),
            'volume': np.random.randint(1000000, 5000000, 10),
        }, index=dates)
        return result

    mock_client.get_stock_bars = mock_get_bars
    return mock_client


# =============================================================================
# Strategy Fixtures
# =============================================================================

@pytest.fixture
def strategy_config():
    """Sample strategy configuration for testing."""
    return {
        'enabled': True,
        'tier': 1,
        'allocation_pct': 0.10,
        'max_positions': 10,
        'rebalance_frequency': 'monthly',
    }


@pytest.fixture
def sample_signals():
    """Generate sample trading signals for testing."""
    return [
        {'symbol': 'AAPL', 'signal': 1, 'strength': 0.8, 'strategy': 'momentum'},
        {'symbol': 'GOOGL', 'signal': 1, 'strength': 0.6, 'strategy': 'momentum'},
        {'symbol': 'MSFT', 'signal': -1, 'strength': 0.7, 'strategy': 'momentum'},
        {'symbol': 'AMZN', 'signal': 0, 'strength': 0.3, 'strategy': 'momentum'},
    ]


@pytest.fixture
def sample_signal_objects():
    """Generate sample Signal objects from the strategies.base module."""
    try:
        from strategies.base import Signal, SignalType
        return [
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                strategy='vol_managed_momentum',
                signal_type=SignalType.BUY,
                strength=0.85,
                price=150.0,
                stop_loss=142.5,
                target_price=165.0,
                position_size_pct=0.05,
                reason='12-1 momentum: 15.2%, vol: 18%',
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='GOOGL',
                strategy='vol_managed_momentum',
                signal_type=SignalType.BUY,
                strength=0.70,
                price=140.0,
                stop_loss=133.0,
                target_price=154.0,
                position_size_pct=0.04,
                reason='12-1 momentum: 12.1%, vol: 22%',
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='MSFT',
                strategy='vol_managed_momentum',
                signal_type=SignalType.CLOSE,
                strength=0.80,
                price=380.0,
                reason='Dropped from top momentum',
            ),
        ]
    except ImportError:
        # Return dict representations if imports fail
        return [
            {'symbol': 'AAPL', 'signal_type': 'BUY', 'strength': 0.85, 'price': 150.0},
            {'symbol': 'GOOGL', 'signal_type': 'BUY', 'strength': 0.70, 'price': 140.0},
            {'symbol': 'MSFT', 'signal_type': 'CLOSE', 'strength': 0.80, 'price': 380.0},
        ]


@pytest.fixture
def sample_momentum_strategy(sample_multi_stock_data):
    """Create a sample VolManagedMomentumStrategy for testing."""
    try:
        from strategies.vol_managed_momentum import VolManagedMomentumStrategy
        strategy = VolManagedMomentumStrategy()
        # Reset rebalance state for deterministic testing
        strategy.last_rebalance_month = None
        return strategy
    except ImportError:
        return MagicMock(name='vol_managed_momentum')


@pytest.fixture
def sample_mean_reversion_strategy():
    """Create a sample MeanReversionStrategy for testing."""
    try:
        from strategies.mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy()
    except ImportError:
        return MagicMock(name='mean_reversion')


@pytest.fixture
def all_strategies():
    """Load all available strategies for integration testing."""
    strategies = {}
    strategy_modules = [
        ('vol_managed_momentum', 'VolManagedMomentumStrategy'),
        ('mean_reversion', 'MeanReversionStrategy'),
        ('vix_regime_rotation', 'VIXRegimeRotationStrategy'),
        ('pairs_trading', 'PairsTradingStrategy'),
        ('relative_volume_breakout', 'RelativeVolumeBreakoutStrategy'),
        ('gap_fill', 'GapFillStrategy'),
    ]

    for module_name, class_name in strategy_modules:
        try:
            module = __import__(f'strategies.{module_name}', fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            strategies[module_name] = strategy_class()
        except (ImportError, AttributeError):
            pass

    return strategies


# =============================================================================
# Backtester Fixtures
# =============================================================================

@pytest.fixture
def backtester_instance():
    """Create a Backtester instance for testing."""
    try:
        from research.backtester import Backtester
        return Backtester(
            initial_capital=100000,
            commission_per_trade=0,
            slippage_model='none',
        )
    except ImportError:
        return MagicMock()


@pytest.fixture
def sample_backtest_result():
    """Sample backtest result for testing."""
    try:
        from research.backtester import BacktestResult
        return BacktestResult(
            run_id='test-123',
            strategy='vol_managed_momentum',
            start_date='2024-01-01',
            end_date='2024-12-31',
            total_return=15.5,
            annual_return=15.5,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown_pct=-8.5,
            total_trades=48,
            winning_trades=28,
            losing_trades=20,
            win_rate=58.3,
            profit_factor=1.8,
            avg_trade_pnl=322.92,
            meets_threshold=True,
            vs_research_pct=70.6,
        )
    except ImportError:
        return {
            'run_id': 'test-123',
            'strategy': 'vol_managed_momentum',
            'sharpe_ratio': 1.2,
            'total_return': 15.5,
            'max_drawdown_pct': -8.5,
        }


# =============================================================================
# Data Manager Fixtures
# =============================================================================

@pytest.fixture
def mock_cached_data_manager(sample_multi_stock_data, sample_vix_data):
    """Create a mock CachedDataManager for testing without real data files."""
    mock_mgr = MagicMock()
    mock_mgr.cache = sample_multi_stock_data
    mock_mgr.metadata = {
        symbol: {
            'price': df['close'].iloc[-1],
            'avg_volume': df['volume'].mean(),
            'dollar_volume': df['close'].iloc[-1] * df['volume'].mean(),
            'volatility': df['close'].pct_change().std() * np.sqrt(252),
            'tier': 'large',
            'bars': len(df),
        }
        for symbol, df in sample_multi_stock_data.items()
    }
    mock_mgr.get_bars.side_effect = lambda s: sample_multi_stock_data.get(s, pd.DataFrame())
    mock_mgr.get_available_symbols.return_value = list(sample_multi_stock_data.keys())
    mock_mgr.get_vix.return_value = sample_vix_data['close'].iloc[-1]
    mock_mgr.get_all_metadata.return_value = mock_mgr.metadata

    return mock_mgr


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config(tmp_path):
    """Override config paths for testing."""
    test_dirs = {
        'data_root': tmp_path / 'data',
        'historical': tmp_path / 'data' / 'historical',
        'daily': tmp_path / 'data' / 'historical' / 'daily',
        'db': tmp_path / 'db',
        'logs': tmp_path / 'logs',
    }

    # Create directories
    for path in test_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return test_dirs


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def freeze_time():
    """Fixture to freeze time for deterministic testing."""
    from unittest.mock import patch
    import datetime

    frozen_time = datetime.datetime(2024, 6, 15, 10, 30, 0)

    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = frozen_time
        mock_dt.side_effect = lambda *args, **kwargs: datetime.datetime(*args, **kwargs)
        yield frozen_time


@pytest.fixture
def caplog_info(caplog):
    """Capture INFO level logs."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# =============================================================================
# Database Manager Fixtures
# =============================================================================

@pytest.fixture
def test_db_manager(test_db_path):
    """
    Create a DatabaseManager instance with test database.

    Uses the trading system's actual DatabaseManager but pointed at test paths.
    """
    try:
        from data.storage.db_manager import DatabaseManager
        # Temporarily override DATABASES paths
        with patch('data.storage.db_manager.DATABASES', {
            'trades': test_db_path.parent / 'trades.db',
            'performance': test_db_path.parent / 'performance.db',
            'research': test_db_path.parent / 'research.db',
            'pairs': test_db_path.parent / 'pairs.db',
        }):
            mgr = DatabaseManager()
            yield mgr
            mgr.close_all()
    except ImportError:
        yield MagicMock()


# =============================================================================
# Execution Fixtures
# =============================================================================

@pytest.fixture
def sample_broker_order():
    """Sample broker order for testing."""
    return {
        'id': 'test-order-001',
        'symbol': 'AAPL',
        'side': 'buy',
        'qty': 100,
        'order_type': 'market',
        'status': 'filled',
        'filled_qty': 100,
        'filled_avg_price': 150.25,
        'submitted_at': datetime.now() - timedelta(minutes=5),
        'filled_at': datetime.now() - timedelta(minutes=4),
    }


@pytest.fixture
def sample_broker_position():
    """Sample broker position for testing."""
    return {
        'symbol': 'AAPL',
        'qty': 100,
        'side': 'long',
        'avg_entry_price': 150.0,
        'market_value': 15500.0,
        'unrealized_pnl': 500.0,
        'unrealized_pnl_pct': 3.33,
        'current_price': 155.0,
    }


# =============================================================================
# Integration Test Fixtures
# =============================================================================

@pytest.fixture
def mock_orchestrator(mock_alpaca_client, mock_cached_data_manager, test_db):
    """Create a mock orchestrator for integration testing."""
    with patch('daily_orchestrator.AlpacaConnector') as mock_connector_class, \
         patch('daily_orchestrator.CachedDataManager') as mock_data_class:

        # Configure mocks
        mock_connector_class.return_value = mock_alpaca_client
        mock_data_class.return_value = mock_cached_data_manager

        from daily_orchestrator import DailyOrchestrator
        orch = DailyOrchestrator(paper_mode=True)
        yield orch


@pytest.fixture
def mock_signal_database(test_db_path):
    """Create a mock SignalDatabase for integration testing."""
    from execution.signal_tracker import SignalDatabase

    with patch('execution.signal_tracker.DATABASES', {
        'trades': test_db_path.parent / 'trades.db',
    }):
        db = SignalDatabase(db_path=test_db_path)
        yield db


@pytest.fixture
def mock_broker_with_positions(mock_alpaca_client, sample_broker_position):
    """Create mock broker with pre-existing positions."""
    mock_alpaca_client._positions = {
        'AAPL': MagicMock(**sample_broker_position),
    }
    return mock_alpaca_client


@pytest.fixture
def integration_signal():
    """Create a signal suitable for integration testing."""
    try:
        from strategies.base import Signal, SignalType
        return Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            strategy='test_strategy',
            signal_type=SignalType.BUY,
            strength=0.85,
            price=150.0,
            stop_loss=142.5,
            target_price=165.0,
            position_size_pct=0.05,
            reason='Integration test signal',
        )
    except ImportError:
        return {
            'symbol': 'AAPL',
            'strategy': 'test_strategy',
            'signal_type': 'BUY',
            'strength': 0.85,
            'price': 150.0,
        }


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_env():
    """Clean up test environment after each test."""
    yield
    # Cleanup happens automatically via tmp_path fixture


@pytest.fixture
def clean_db(test_db):
    """Provide a clean database connection that's rolled back after test."""
    # Start a transaction
    test_db.execute("BEGIN")
    yield test_db
    # Rollback any changes
    test_db.rollback()


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "critical: Critical path tests that must pass"
    )
    config.addinivalue_line(
        "markers", "unit: Fast, isolated unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take > 5 seconds"
    )
    config.addinivalue_line(
        "markers", "live_api: Tests requiring live API connection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection - add markers based on location."""
    for item in items:
        # Auto-mark tests in unit/ as unit tests
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Auto-mark tests in integration/ as integration tests
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def pytest_runtest_setup(item):
    """Called before each test - can skip tests based on markers."""
    # Skip live_api tests unless explicitly requested
    if 'live_api' in [marker.name for marker in item.iter_markers()]:
        if not item.config.getoption("--run-live-api", default=False):
            pytest.skip("Live API tests disabled (use --run-live-api to enable)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run tests that require live API connection"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


# =============================================================================
# Test Reporting Hooks
# =============================================================================

def pytest_report_header(config):
    """Add custom header to test report."""
    return [
        "Trading System Test Suite",
        f"Project Root: {PROJECT_ROOT}",
    ]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Custom test result processing for reporting."""
    # This hook can be used by coverage agent to collect test results
    pass
