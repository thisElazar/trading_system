"""
Mock implementations for testing without live API calls.

This package provides mock implementations of:
- Alpaca Trading/Data clients (with error injection)
- Database connections (in-memory SQLite)
- Market data generators (OHLCV, VIX, corporate actions)
- External services (HTTP, email, webhooks, cache)

Usage:
    from tests.mocks import (
        MockTradingClient,
        MockDataClient,
        MockDatabaseManager,
        generate_mock_bars,
        create_mock_trading_client,
    )

    # Create a mock trading client with positions
    client = create_mock_trading_client(
        equity=100000,
        positions={'AAPL': {'qty': 100, 'avg_price': 150.0}},
    )

    # Simulate API errors
    client.simulate_error('rate_limit')
"""

# Alpaca mocks
from .mock_alpaca import (
    # Core mock classes
    MockTradingClient,
    MockDataClient,
    MockAccount,
    MockPosition,
    MockOrder,
    MockQuote,
    MockBarsResponse,
    MockOrderStatus,
    MockOrderSide,

    # Enhanced clients with error injection
    MockTradingClientV2,
    AsyncMockTradingClient,
    AsyncMockDataClient,

    # Error types
    AlpacaAPIError,
    RateLimitError,
    InsufficientFundsError,
    MarketClosedError,
    SymbolNotFoundError,

    # Factory functions
    create_mock_trading_client,
    create_mock_data_client,
)

# Database mocks
from .mock_database import (
    MockDatabaseManager,
    create_test_db,
)

# Market data mocks
from .mock_market_data import (
    MockMarketDataProvider,
    generate_mock_bars,
    generate_mock_quotes,
    generate_intraday_bars,
    generate_gap_scenario,
    generate_vix_data,
    generate_corporate_action_scenario,
    generate_halt_scenario,
    generate_momentum_scenario,
    generate_mean_reversion_scenario,
    generate_correlated_assets,
    generate_pairs_trading_data,
)

# Service mocks
from .mock_services import (
    # HTTP client
    MockHTTPClient,
    MockHTTPResponse,

    # Notification services
    MockEmailService,
    MockWebhookService,

    # Utility services
    MockTimeProvider,
    MockCacheService,
    MockMetricsService,

    # Container
    MockServiceContainer,

    # Error types
    FailureMode,
    NetworkError,
    TimeoutError as MockTimeoutError,  # Renamed to avoid conflict with builtin
    ConnectionError as MockConnectionError,  # Renamed to avoid conflict with builtin

    # Latency injection
    LatencyConfig,
    LatencyInjector,

    # Factory functions
    create_mock_http_client,
    create_service_container,
)


__all__ = [
    # =========================================================================
    # Alpaca Mocks
    # =========================================================================
    'MockTradingClient',
    'MockTradingClientV2',
    'AsyncMockTradingClient',
    'MockDataClient',
    'AsyncMockDataClient',
    'MockAccount',
    'MockPosition',
    'MockOrder',
    'MockQuote',
    'MockBarsResponse',
    'MockOrderStatus',
    'MockOrderSide',

    # Alpaca errors
    'AlpacaAPIError',
    'RateLimitError',
    'InsufficientFundsError',
    'MarketClosedError',
    'SymbolNotFoundError',

    # Alpaca factories
    'create_mock_trading_client',
    'create_mock_data_client',

    # =========================================================================
    # Database Mocks
    # =========================================================================
    'MockDatabaseManager',
    'create_test_db',

    # =========================================================================
    # Market Data Mocks
    # =========================================================================
    'MockMarketDataProvider',
    'generate_mock_bars',
    'generate_mock_quotes',
    'generate_intraday_bars',
    'generate_gap_scenario',
    'generate_vix_data',
    'generate_corporate_action_scenario',
    'generate_halt_scenario',
    'generate_momentum_scenario',
    'generate_mean_reversion_scenario',
    'generate_correlated_assets',
    'generate_pairs_trading_data',

    # =========================================================================
    # Service Mocks
    # =========================================================================
    'MockHTTPClient',
    'MockHTTPResponse',
    'MockEmailService',
    'MockWebhookService',
    'MockTimeProvider',
    'MockCacheService',
    'MockMetricsService',
    'MockServiceContainer',

    # Service errors
    'FailureMode',
    'NetworkError',
    'MockTimeoutError',
    'MockConnectionError',

    # Latency injection
    'LatencyConfig',
    'LatencyInjector',

    # Service factories
    'create_mock_http_client',
    'create_service_container',
]


# =============================================================================
# Pytest Fixture Helpers
# =============================================================================
# These functions can be imported into conftest.py to create fixtures

def mock_trading_client_fixture():
    """
    Create a mock trading client fixture.

    Usage in conftest.py:
        from tests.mocks import mock_trading_client_fixture

        @pytest.fixture
        def mock_trading_client():
            return mock_trading_client_fixture()
    """
    return create_mock_trading_client()


def mock_data_client_fixture():
    """
    Create a mock data client fixture.

    Usage in conftest.py:
        from tests.mocks import mock_data_client_fixture

        @pytest.fixture
        def mock_data_client():
            return mock_data_client_fixture()
    """
    return create_mock_data_client()


def mock_db_fixture():
    """
    Create a mock database fixture.

    Usage in conftest.py:
        from tests.mocks import mock_db_fixture

        @pytest.fixture
        def mock_db():
            return mock_db_fixture()
    """
    return MockDatabaseManager()


def mock_market_data_fixture():
    """
    Create a mock market data provider fixture.

    Usage in conftest.py:
        from tests.mocks import mock_market_data_fixture

        @pytest.fixture
        def mock_market_data():
            return mock_market_data_fixture()
    """
    return MockMarketDataProvider(seed=42)


def mock_services_fixture():
    """
    Create a mock services container fixture.

    Usage in conftest.py:
        from tests.mocks import mock_services_fixture

        @pytest.fixture
        def mock_services():
            return mock_services_fixture()
    """
    return create_service_container()


# Export fixture helpers
__all__.extend([
    'mock_trading_client_fixture',
    'mock_data_client_fixture',
    'mock_db_fixture',
    'mock_market_data_fixture',
    'mock_services_fixture',
])
