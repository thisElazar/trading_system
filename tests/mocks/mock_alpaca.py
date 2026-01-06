"""
Mock Alpaca API clients for testing.

Provides realistic mock implementations of Alpaca's Trading and Data clients
that can be configured with custom responses, error injection, and latency simulation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


class MockOrderStatus(Enum):
    """Mock order status enum."""
    NEW = 'new'
    ACCEPTED = 'accepted'
    PENDING_NEW = 'pending_new'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    DONE_FOR_DAY = 'done_for_day'
    CANCELED = 'canceled'
    EXPIRED = 'expired'
    REPLACED = 'replaced'
    REJECTED = 'rejected'


class MockOrderSide(Enum):
    """Mock order side enum."""
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class MockAccount:
    """Mock Alpaca account."""
    equity: float = 100000.0
    cash: float = 50000.0
    buying_power: float = 200000.0
    portfolio_value: float = 100000.0
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False


@dataclass
class MockPosition:
    """Mock Alpaca position."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = None
    side: str = 'long'

    def __post_init__(self):
        if self.current_price is None:
            self.current_price = self.avg_entry_price

    @property
    def market_value(self) -> float:
        return abs(self.qty) * self.current_price

    @property
    def unrealized_pl(self) -> float:
        return (self.current_price - self.avg_entry_price) * self.qty

    @property
    def unrealized_plpc(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class MockOrder:
    """Mock Alpaca order."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ''
    qty: float = 0
    side: MockOrderSide = MockOrderSide.BUY
    order_type: str = 'market'
    status: MockOrderStatus = MockOrderStatus.NEW
    filled_qty: float = 0
    filled_avg_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


class MockTradingClient:
    """
    Mock Alpaca Trading Client for testing.

    Features:
    - Configurable account balances
    - Position tracking
    - Order execution simulation
    - Error injection
    - Fill simulation modes
    """

    def __init__(
        self,
        api_key: str = 'test_key',
        secret_key: str = 'test_secret',
        paper: bool = True,
        initial_equity: float = 100000.0,
        initial_cash: float = 50000.0,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Account state
        self._account = MockAccount(
            equity=initial_equity,
            cash=initial_cash,
            buying_power=initial_cash * 4,  # 4x margin
            portfolio_value=initial_equity,
        )

        # Position tracking
        self._positions: Dict[str, MockPosition] = {}

        # Order tracking
        self._orders: Dict[str, MockOrder] = {}

        # Configuration
        self._fill_immediately = True
        self._fill_delay_seconds = 0
        self._next_error: Optional[Exception] = None
        self._price_provider: Optional[callable] = None

    # -------------------------------------------------------------------------
    # Account & Positions
    # -------------------------------------------------------------------------

    def get_account(self) -> MockAccount:
        """Get mock account info."""
        self._maybe_raise_error()
        return self._account

    def get_all_positions(self) -> List[MockPosition]:
        """Get all open positions."""
        self._maybe_raise_error()
        return list(self._positions.values())

    def get_open_position(self, symbol: str) -> MockPosition:
        """Get position for specific symbol."""
        self._maybe_raise_error()
        if symbol not in self._positions:
            raise Exception(f"No position found for {symbol}")
        return self._positions[symbol]

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    def submit_order(self, order_data: Any) -> MockOrder:
        """Submit a new order."""
        self._maybe_raise_error()

        # Extract order details from request object
        symbol = getattr(order_data, 'symbol', 'UNKNOWN')
        qty = getattr(order_data, 'qty', 0)
        side = getattr(order_data, 'side', MockOrderSide.BUY)
        limit_price = getattr(order_data, 'limit_price', None)

        # Create order
        order = MockOrder(
            symbol=symbol,
            qty=float(qty),
            side=side if isinstance(side, MockOrderSide) else MockOrderSide(side.value),
            order_type='limit' if limit_price else 'market',
            limit_price=limit_price,
            status=MockOrderStatus.ACCEPTED,
        )

        self._orders[order.id] = order

        # Auto-fill if configured
        if self._fill_immediately:
            self._fill_order(order.id)

        return order

    def cancel_order_by_id(self, order_id: str) -> None:
        """Cancel an order."""
        self._maybe_raise_error()
        if order_id not in self._orders:
            raise Exception(f"Order not found: {order_id}")

        order = self._orders[order_id]
        if order.status == MockOrderStatus.FILLED:
            raise Exception("Cannot cancel filled order")

        order.status = MockOrderStatus.CANCELED

    def get_order_by_id(self, order_id: str) -> MockOrder:
        """Get order by ID."""
        self._maybe_raise_error()
        if order_id not in self._orders:
            raise Exception(f"Order not found: {order_id}")
        return self._orders[order_id]

    def get_orders(self, request: Any = None) -> List[MockOrder]:
        """Get orders matching filter."""
        self._maybe_raise_error()
        orders = list(self._orders.values())

        # Filter by status if request specifies
        if request and hasattr(request, 'status'):
            status = request.status
            if hasattr(status, 'value'):
                status = status.value
            orders = [o for o in orders if o.status.value == status.lower()]

        return orders

    def close_position(self, symbol: str) -> None:
        """Close position for symbol."""
        self._maybe_raise_error()
        if symbol in self._positions:
            del self._positions[symbol]

    def close_all_positions(self, cancel_orders: bool = False) -> None:
        """Close all positions."""
        self._maybe_raise_error()
        self._positions.clear()
        if cancel_orders:
            for order in self._orders.values():
                if order.status not in [MockOrderStatus.FILLED, MockOrderStatus.CANCELED]:
                    order.status = MockOrderStatus.CANCELED

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _fill_order(self, order_id: str) -> None:
        """Simulate order fill."""
        order = self._orders[order_id]

        # Determine fill price
        if order.limit_price:
            fill_price = order.limit_price
        elif self._price_provider:
            fill_price = self._price_provider(order.symbol)
        else:
            # Use a default mock price
            fill_price = 150.0

        order.status = MockOrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = fill_price
        order.filled_at = datetime.now()

        # Update position
        self._update_position(order, fill_price)

        # Update account
        cost = order.qty * fill_price
        if order.side == MockOrderSide.BUY:
            self._account.cash -= cost
        else:
            self._account.cash += cost

    def _update_position(self, order: MockOrder, price: float) -> None:
        """Update position based on filled order."""
        symbol = order.symbol
        qty = order.qty if order.side == MockOrderSide.BUY else -order.qty

        if symbol in self._positions:
            pos = self._positions[symbol]
            new_qty = pos.qty + qty
            if new_qty == 0:
                del self._positions[symbol]
            else:
                # Update average price
                if qty > 0:  # Adding to position
                    total_cost = pos.qty * pos.avg_entry_price + qty * price
                    pos.avg_entry_price = total_cost / (pos.qty + qty)
                pos.qty = new_qty
        else:
            if qty != 0:
                self._positions[symbol] = MockPosition(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=price,
                    current_price=price,
                )

    def _maybe_raise_error(self) -> None:
        """Raise configured error if set."""
        if self._next_error:
            error = self._next_error
            self._next_error = None
            raise error

    # -------------------------------------------------------------------------
    # Test Configuration Methods
    # -------------------------------------------------------------------------

    def set_next_error(self, error: Exception) -> None:
        """Configure next API call to raise an error."""
        self._next_error = error

    def set_fill_mode(self, immediate: bool = True, delay_seconds: float = 0) -> None:
        """Configure order fill behavior."""
        self._fill_immediately = immediate
        self._fill_delay_seconds = delay_seconds

    def set_price_provider(self, provider: callable) -> None:
        """Set function to provide prices for fills."""
        self._price_provider = provider

    def add_position(self, symbol: str, qty: float, avg_price: float) -> None:
        """Add a position for testing."""
        self._positions[symbol] = MockPosition(
            symbol=symbol,
            qty=qty,
            avg_entry_price=avg_price,
            current_price=avg_price,
        )

    def set_account_balance(self, equity: float, cash: float) -> None:
        """Set account balances for testing."""
        self._account.equity = equity
        self._account.cash = cash
        self._account.buying_power = cash * 4


class MockDataClient:
    """
    Mock Alpaca Data Client for testing.

    Provides configurable mock market data responses.
    """

    def __init__(
        self,
        api_key: str = 'test_key',
        secret_key: str = 'test_secret',
    ):
        self.api_key = api_key
        self.secret_key = secret_key

        # Custom price data
        self._prices: Dict[str, float] = {}
        self._bars_data: Dict[str, pd.DataFrame] = {}

    def get_stock_latest_quote(self, request: Any) -> Dict[str, Any]:
        """Get latest quote for symbol(s)."""
        symbols = getattr(request, 'symbol_or_symbols', [])
        if isinstance(symbols, str):
            symbols = [symbols]

        result = {}
        for symbol in symbols:
            price = self._prices.get(symbol, 150.0)
            result[symbol] = MockQuote(
                bid_price=price * 0.999,
                ask_price=price * 1.001,
            )

        return result

    def get_stock_bars(self, request: Any) -> MockBarsResponse:
        """Get historical bars for symbol(s)."""
        symbols = getattr(request, 'symbol_or_symbols', [])
        if isinstance(symbols, str):
            symbols = [symbols]

        # Generate or return configured data
        if symbols[0] in self._bars_data:
            df = self._bars_data[symbols[0]]
        else:
            df = self._generate_bars(60)

        return MockBarsResponse(df)

    def set_price(self, symbol: str, price: float) -> None:
        """Set mock price for a symbol."""
        self._prices[symbol] = price

    def set_bars(self, symbol: str, df: pd.DataFrame) -> None:
        """Set mock bars data for a symbol."""
        self._bars_data[symbol] = df

    def _generate_bars(self, days: int) -> pd.DataFrame:
        """Generate random bars data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = 150 * np.cumprod(1 + np.random.normal(0, 0.02, days))

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, days)),
            'low': prices * (1 - np.random.uniform(0, 0.02, days)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days),
        })


@dataclass
class MockQuote:
    """Mock quote response."""
    bid_price: float
    ask_price: float


class MockBarsResponse:
    """Mock bars response with DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def reset_index(self) -> pd.DataFrame:
        return self._df.reset_index()


# =============================================================================
# Error Simulation Classes
# =============================================================================

class AlpacaAPIError(Exception):
    """Mock Alpaca API error."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


class RateLimitError(AlpacaAPIError):
    """Rate limit exceeded error."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(429, f"Rate limit exceeded. Retry after {retry_after} seconds.")


class InsufficientFundsError(AlpacaAPIError):
    """Insufficient buying power error."""
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(403, f"Insufficient funds: required ${required:.2f}, available ${available:.2f}")


class MarketClosedError(AlpacaAPIError):
    """Market is closed error."""
    def __init__(self):
        super().__init__(400, "Market is closed. Cannot submit order.")


class SymbolNotFoundError(AlpacaAPIError):
    """Symbol not found error."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(404, f"Symbol not found: {symbol}")


class TimeoutError(AlpacaAPIError):
    """Request timeout error."""
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        super().__init__(408, f"Request timed out after {timeout_seconds} seconds")


# =============================================================================
# Enhanced Mock Trading Client with Error Injection
# =============================================================================

class MockTradingClientV2(MockTradingClient):
    """
    Enhanced mock trading client with advanced error simulation.

    Features:
    - Configurable failure modes
    - Rate limit simulation
    - Latency injection
    - Market hours awareness
    - Pattern day trader simulation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Error injection configuration
        self._error_rate: float = 0.0  # Probability of random error
        self._rate_limit_remaining: int = 200
        self._rate_limit_reset: datetime = datetime.now() + timedelta(minutes=1)
        self._latency_ms: int = 0
        self._market_open: bool = True
        self._blocked_symbols: set = set()

        # Call tracking for testing
        self._call_history: List[Dict] = []

    def simulate_error(self, error_type: str, **kwargs) -> None:
        """
        Configure specific error for next call.

        Args:
            error_type: One of 'rate_limit', 'insufficient_funds',
                       'market_closed', 'timeout', 'symbol_not_found'
        """
        error_map = {
            'rate_limit': RateLimitError(kwargs.get('retry_after', 60)),
            'insufficient_funds': InsufficientFundsError(
                kwargs.get('required', 10000),
                kwargs.get('available', 100)
            ),
            'market_closed': MarketClosedError(),
            'timeout': TimeoutError(kwargs.get('timeout', 30)),
            'symbol_not_found': SymbolNotFoundError(kwargs.get('symbol', 'INVALID')),
        }

        if error_type in error_map:
            self._next_error = error_map[error_type]
        else:
            raise ValueError(f"Unknown error type: {error_type}")

    def set_error_rate(self, rate: float) -> None:
        """Set probability of random errors (0.0 to 1.0)."""
        self._error_rate = max(0.0, min(1.0, rate))

    def set_latency(self, ms: int) -> None:
        """Set simulated latency in milliseconds."""
        self._latency_ms = max(0, ms)

    def set_market_open(self, is_open: bool) -> None:
        """Set whether market is open."""
        self._market_open = is_open

    def block_symbol(self, symbol: str) -> None:
        """Block a symbol from trading."""
        self._blocked_symbols.add(symbol)

    def unblock_symbol(self, symbol: str) -> None:
        """Unblock a symbol."""
        self._blocked_symbols.discard(symbol)

    def get_call_history(self) -> List[Dict]:
        """Get history of API calls for testing assertions."""
        return self._call_history.copy()

    def clear_call_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    def _maybe_raise_error(self) -> None:
        """Enhanced error checking with random failures."""
        # Check for configured error
        if self._next_error:
            error = self._next_error
            self._next_error = None
            raise error

        # Random error based on error rate
        if self._error_rate > 0 and np.random.random() < self._error_rate:
            raise AlpacaAPIError(500, "Random simulated server error")

        # Rate limit check
        if self._rate_limit_remaining <= 0:
            if datetime.now() < self._rate_limit_reset:
                raise RateLimitError()
            else:
                self._rate_limit_remaining = 200
                self._rate_limit_reset = datetime.now() + timedelta(minutes=1)

        self._rate_limit_remaining -= 1

    def submit_order(self, order_data: Any) -> MockOrder:
        """Enhanced order submission with validation."""
        # Track call
        self._call_history.append({
            'method': 'submit_order',
            'timestamp': datetime.now(),
            'data': order_data,
        })

        # Simulate latency
        if self._latency_ms > 0:
            import time
            time.sleep(self._latency_ms / 1000.0)

        # Check market hours
        if not self._market_open:
            raise MarketClosedError()

        # Check blocked symbols
        symbol = getattr(order_data, 'symbol', 'UNKNOWN')
        if symbol in self._blocked_symbols:
            raise SymbolNotFoundError(symbol)

        # Check buying power
        qty = float(getattr(order_data, 'qty', 0))
        limit_price = getattr(order_data, 'limit_price', None)
        price = limit_price or 150.0  # Default price
        required = qty * price

        side = getattr(order_data, 'side', MockOrderSide.BUY)
        if hasattr(side, 'value'):
            side_val = side.value.lower()
        else:
            side_val = str(side).lower()

        if side_val == 'buy' and required > self._account.buying_power:
            raise InsufficientFundsError(required, self._account.buying_power)

        return super().submit_order(order_data)


# =============================================================================
# Async Mock Clients (for codebases using asyncio)
# =============================================================================

class AsyncMockTradingClient:
    """
    Async version of MockTradingClient for async codebases.

    Wraps synchronous mock in async interface.
    """

    def __init__(self, *args, **kwargs):
        self._sync_client = MockTradingClientV2(*args, **kwargs)

    async def get_account(self) -> MockAccount:
        return self._sync_client.get_account()

    async def get_all_positions(self) -> List[MockPosition]:
        return self._sync_client.get_all_positions()

    async def get_open_position(self, symbol: str) -> MockPosition:
        return self._sync_client.get_open_position(symbol)

    async def submit_order(self, order_data: Any) -> MockOrder:
        return self._sync_client.submit_order(order_data)

    async def cancel_order_by_id(self, order_id: str) -> None:
        return self._sync_client.cancel_order_by_id(order_id)

    async def get_order_by_id(self, order_id: str) -> MockOrder:
        return self._sync_client.get_order_by_id(order_id)

    async def get_orders(self, request: Any = None) -> List[MockOrder]:
        return self._sync_client.get_orders(request)

    async def close_position(self, symbol: str) -> None:
        return self._sync_client.close_position(symbol)

    async def close_all_positions(self, cancel_orders: bool = False) -> None:
        return self._sync_client.close_all_positions(cancel_orders)

    # Delegate configuration methods
    def set_next_error(self, error: Exception) -> None:
        self._sync_client.set_next_error(error)

    def simulate_error(self, error_type: str, **kwargs) -> None:
        self._sync_client.simulate_error(error_type, **kwargs)

    def set_error_rate(self, rate: float) -> None:
        self._sync_client.set_error_rate(rate)

    def set_latency(self, ms: int) -> None:
        self._sync_client.set_latency(ms)

    def add_position(self, symbol: str, qty: float, avg_price: float) -> None:
        self._sync_client.add_position(symbol, qty, avg_price)

    def set_account_balance(self, equity: float, cash: float) -> None:
        self._sync_client.set_account_balance(equity, cash)


class AsyncMockDataClient:
    """Async version of MockDataClient."""

    def __init__(self, *args, **kwargs):
        self._sync_client = MockDataClient(*args, **kwargs)

    async def get_stock_latest_quote(self, request: Any) -> Dict[str, Any]:
        return self._sync_client.get_stock_latest_quote(request)

    async def get_stock_bars(self, request: Any) -> MockBarsResponse:
        return self._sync_client.get_stock_bars(request)

    def set_price(self, symbol: str, price: float) -> None:
        self._sync_client.set_price(symbol, price)

    def set_bars(self, symbol: str, df: pd.DataFrame) -> None:
        self._sync_client.set_bars(symbol, df)


# =============================================================================
# Factory Functions
# =============================================================================

def create_mock_trading_client(
    equity: float = 100000.0,
    cash: float = 50000.0,
    positions: Optional[Dict[str, Dict]] = None,
    error_mode: Optional[str] = None,
) -> MockTradingClientV2:
    """
    Factory function to create configured mock trading client.

    Args:
        equity: Initial account equity
        cash: Initial cash balance
        positions: Dict of symbol -> {qty, avg_price} for initial positions
        error_mode: Optional error mode to configure

    Returns:
        Configured MockTradingClientV2 instance
    """
    client = MockTradingClientV2(
        initial_equity=equity,
        initial_cash=cash,
    )

    if positions:
        for symbol, pos_data in positions.items():
            client.add_position(
                symbol=symbol,
                qty=pos_data.get('qty', 0),
                avg_price=pos_data.get('avg_price', 100.0),
            )

    if error_mode:
        client.simulate_error(error_mode)

    return client


def create_mock_data_client(
    prices: Optional[Dict[str, float]] = None,
    bars: Optional[Dict[str, pd.DataFrame]] = None,
) -> MockDataClient:
    """
    Factory function to create configured mock data client.

    Args:
        prices: Dict of symbol -> price for quote responses
        bars: Dict of symbol -> DataFrame for bars responses

    Returns:
        Configured MockDataClient instance
    """
    client = MockDataClient()

    if prices:
        for symbol, price in prices.items():
            client.set_price(symbol, price)

    if bars:
        for symbol, df in bars.items():
            client.set_bars(symbol, df)

    return client
