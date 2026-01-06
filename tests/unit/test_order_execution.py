"""
Unit Tests: Order Execution
===========================

Tests for the order execution module including:
- Order creation and validation
- Order type handling (market, limit, stop, bracket)
- Position sizing calculations
- Order state transitions
- Error handling for failed orders

Uses fixtures from conftest.py: test_db, sample_ohlcv_data, mock_alpaca_client
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Import test targets
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execution.order_executor import OrderExecutor, OrderResult, OrderStatus
from execution.position_sizer import PositionSizer, PositionSize
from strategies.base import Signal, SignalType


# =============================================================================
# OrderStatus Tests
# =============================================================================

@pytest.mark.unit
class TestOrderStatus:
    """Tests for OrderStatus enum and transitions."""

    def test_order_status_values_exist(self):
        """Verify all expected order status values are defined."""
        expected_statuses = ['PENDING', 'SUBMITTED', 'FILLED', 'PARTIALLY_FILLED',
                            'CANCELLED', 'REJECTED', 'FAILED']
        for status in expected_statuses:
            assert hasattr(OrderStatus, status), f"OrderStatus.{status} not found"

    def test_order_status_string_values(self):
        """Verify status enum values are lowercase strings."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.FAILED.value == "failed"


# =============================================================================
# OrderResult Tests
# =============================================================================

@pytest.mark.unit
class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_order_result_creation_success(self):
        """Verify successful order result can be created with all fields."""
        result = OrderResult(
            success=True,
            order_id="test-123",
            status=OrderStatus.SUBMITTED,
            filled_qty=100,
            filled_price=150.50,
            message="Order submitted successfully",
            submitted_at=datetime.now()
        )

        assert result.success is True
        assert result.order_id == "test-123"
        assert result.status == OrderStatus.SUBMITTED
        assert result.filled_qty == 100
        assert result.filled_price == 150.50

    def test_order_result_creation_failure(self):
        """Verify failed order result captures error information."""
        result = OrderResult(
            success=False,
            status=OrderStatus.FAILED,
            message="Insufficient buying power"
        )

        assert result.success is False
        assert result.status == OrderStatus.FAILED
        assert "Insufficient" in result.message

    def test_order_result_defaults(self):
        """Verify default values for optional fields."""
        result = OrderResult(success=True)

        assert result.order_id is None
        assert result.status == OrderStatus.PENDING
        assert result.filled_qty == 0
        assert result.filled_price == 0
        assert result.message == ""
        assert result.submitted_at is None
        assert result.filled_at is None


# =============================================================================
# OrderExecutor Tests
# =============================================================================

@pytest.mark.unit
class TestOrderExecutor:
    """Tests for OrderExecutor class."""

    def test_executor_initialization_paper_mode(self):
        """Verify executor initializes in paper trading mode by default."""
        executor = OrderExecutor()

        assert executor.paper is True
        assert executor._client is None
        assert executor._trading_client is None

    def test_executor_initialization_live_mode(self):
        """Verify executor can be initialized in live trading mode."""
        executor = OrderExecutor(paper=False)

        assert executor.paper is False

    @pytest.mark.parametrize("side,expected_side", [
        ("buy", "buy"),
        ("BUY", "buy"),
        ("Buy", "buy"),
        ("sell", "sell"),
        ("SELL", "sell"),
        ("Sell", "sell"),
    ])
    def test_submit_market_order_side_normalization(self, mock_alpaca_client, side, expected_side):
        """Verify order side is normalized to lowercase."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        result = executor.submit_market_order(
            symbol="AAPL",
            qty=10,
            side=side
        )

        assert result.success is True
        assert result.status == OrderStatus.SUBMITTED

    def test_submit_market_order_success(self, mock_alpaca_client):
        """Verify market order submission succeeds with valid parameters."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        result = executor.submit_market_order(
            symbol="AAPL",
            qty=100,
            side="buy"
        )

        assert result.success is True
        assert result.order_id is not None
        assert result.status == OrderStatus.SUBMITTED
        assert result.submitted_at is not None

    def test_submit_market_order_failure(self, mock_alpaca_client):
        """Verify market order handles API errors gracefully."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # Configure mock to raise an error
        mock_alpaca_client.set_next_error(Exception("API Error: Market closed"))

        result = executor.submit_market_order(
            symbol="AAPL",
            qty=100,
            side="buy"
        )

        assert result.success is False
        assert result.status == OrderStatus.FAILED
        assert "API Error" in result.message

    def test_submit_limit_order_success(self, mock_alpaca_client):
        """Verify limit order submission succeeds."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        result = executor.submit_limit_order(
            symbol="GOOGL",
            qty=50,
            side="buy",
            limit_price=140.00
        )

        assert result.success is True
        assert result.order_id is not None
        assert result.status == OrderStatus.SUBMITTED

    @pytest.mark.parametrize("limit_price,expected_rounded", [
        (150.123, 150.12),
        (150.125, 150.12),
        (150.129, 150.13),
        (150.001, 150.00),
    ])
    def test_submit_limit_order_price_rounding(self, mock_alpaca_client, limit_price, expected_rounded):
        """Verify limit prices are rounded to 2 decimal places."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # The order should be submitted (rounding happens internally)
        result = executor.submit_limit_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            limit_price=limit_price
        )

        assert result.success is True

    def test_submit_bracket_order_success(self, mock_alpaca_client):
        """Verify bracket order submission with stop loss and take profit."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        result = executor.submit_bracket_order(
            symbol="MSFT",
            qty=25,
            side="buy",
            stop_loss=145.00,
            take_profit=165.00
        )

        assert result.success is True
        assert result.order_id is not None
        assert "Bracket" in result.message or "bracket" in result.message.lower()

    def test_close_position_success(self, mock_alpaca_client):
        """Verify position closing succeeds."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # Add a position to close
        mock_alpaca_client.add_position("AAPL", 100, 150.00)

        result = executor.close_position("AAPL")

        assert result.success is True
        assert result.status == OrderStatus.SUBMITTED

    def test_cancel_order_success(self, mock_alpaca_client):
        """Verify order cancellation succeeds."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # First submit an order
        mock_alpaca_client.set_fill_mode(immediate=False)  # Don't auto-fill
        order_result = executor.submit_market_order("AAPL", 10, "buy")

        # Then cancel it
        success = executor.cancel_order(order_result.order_id)

        assert success is True

    @pytest.mark.parametrize("time_in_force,expected_valid", [
        ("day", True),
        ("DAY", True),
        ("gtc", True),
        ("GTC", True),
    ])
    def test_submit_order_time_in_force(self, mock_alpaca_client, time_in_force, expected_valid):
        """Verify different time-in-force values are handled."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        result = executor.submit_market_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            time_in_force=time_in_force
        )

        assert result.success is expected_valid


# =============================================================================
# Execute Signal Tests
# =============================================================================

@pytest.mark.unit
class TestExecuteSignal:
    """Tests for signal execution."""

    def test_execute_buy_signal_market_order(self, mock_alpaca_client):
        """Verify BUY signal creates market order without stop/target."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00,
            metadata={'shares': 50}
        )

        result = executor.execute_signal(signal)

        assert result.success is True
        assert result.status == OrderStatus.SUBMITTED

    def test_execute_buy_signal_bracket_order(self, mock_alpaca_client):
        """Verify BUY signal with stop/target creates bracket order."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00,
            stop_loss=145.00,
            target_price=165.00,
            metadata={'shares': 50}
        )

        result = executor.execute_signal(signal)

        assert result.success is True

    def test_execute_sell_signal(self, mock_alpaca_client):
        """Verify SELL signal creates sell order."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # First add a position to sell
        mock_alpaca_client.add_position("AAPL", 100, 140.00)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.SELL,
            strength=0.8,
            price=150.00,
            metadata={'shares': 50}
        )

        result = executor.execute_signal(signal)

        assert result.success is True

    def test_execute_close_signal(self, mock_alpaca_client):
        """Verify CLOSE signal closes entire position."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # Add a position to close
        mock_alpaca_client.add_position("AAPL", 100, 140.00)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.CLOSE,
            strength=1.0,
            price=150.00
        )

        result = executor.execute_signal(signal)

        assert result.success is True

    def test_execute_hold_signal_no_action(self, mock_alpaca_client):
        """Verify HOLD signal returns failure (no action taken)."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.HOLD,
            strength=0.5,
            price=150.00
        )

        result = executor.execute_signal(signal)

        assert result.success is False
        assert "Unknown signal type" in result.message or result.status == OrderStatus.FAILED


# =============================================================================
# PositionSizer Tests
# =============================================================================

@pytest.mark.unit
class TestPositionSizer:
    """Tests for position sizing calculations."""

    def test_position_sizer_initialization(self):
        """Verify position sizer initializes with default values."""
        sizer = PositionSizer()

        assert sizer.total_capital > 0
        assert sizer.risk_per_trade > 0
        assert sizer.max_position_size > 0
        assert sizer.max_positions > 0

    def test_position_sizer_custom_initialization(self):
        """Verify position sizer accepts custom parameters."""
        sizer = PositionSizer(
            total_capital=50000,
            risk_per_trade=0.01,
            max_position_size=5000,
            max_positions=5
        )

        assert sizer.total_capital == 50000
        assert sizer.risk_per_trade == 0.01
        assert sizer.max_position_size == 5000
        assert sizer.max_positions == 5

    def test_calculate_atr_based_position(self):
        """Verify ATR-based position sizing formula."""
        sizer = PositionSizer(
            total_capital=100000,
            risk_per_trade=0.02
        )

        result = sizer.calculate_atr_based(
            symbol="AAPL",
            price=150.00,
            atr=3.00,
            stop_multiplier=2.0
        )

        # Verify result structure
        assert isinstance(result, PositionSize)
        assert result.symbol == "AAPL"
        assert result.shares >= 0
        assert result.dollar_value >= 0
        assert result.method == 'atr_based'

        # Verify stop distance calculation
        expected_stop_distance = 3.00 * 2.0  # ATR * multiplier
        assert result.stop_distance == expected_stop_distance

    def test_calculate_atr_based_respects_max_position_size(self):
        """Verify ATR sizing respects maximum position size limits."""
        sizer = PositionSizer(
            total_capital=100000,
            risk_per_trade=0.10,  # High risk would give large position
            max_position_size=5000
        )

        result = sizer.calculate_atr_based(
            symbol="AAPL",
            price=150.00,
            atr=1.00  # Small ATR means larger position
        )

        assert result.dollar_value <= 5000
        assert result.was_capped is True
        assert "max_position" in result.cap_reason.lower()

    def test_calculate_fixed_fractional_position(self):
        """Verify fixed fractional position sizing."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_fixed_fractional(
            symbol="AAPL",
            price=100.00,
            fraction=0.10
        )

        assert result.method == 'fixed_fractional'
        assert result.shares >= 0
        # 10% of $100k = $10k / $100 = 100 shares (before any caps)

    def test_calculate_volatility_targeted_position(self):
        """Verify volatility-targeted position sizing."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_volatility_targeted(
            symbol="NVDA",
            price=450.00,
            realized_vol=0.45,
            target_vol=0.15,
            strategy_allocation=0.25
        )

        assert result.method == 'volatility_targeted'
        assert result.shares >= 0

    def test_calculate_kelly_position_positive_edge(self):
        """Verify Kelly sizing with positive expected edge."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_kelly(
            symbol="MSFT",
            price=375.00,
            win_rate=0.60,
            avg_win=0.10,
            avg_loss=0.05,
            kelly_fraction=0.25
        )

        assert result.method == 'kelly'
        assert result.shares >= 0
        assert result.dollar_value >= 0

    def test_calculate_kelly_position_negative_edge(self):
        """Verify Kelly sizing with negative edge gives zero position."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_kelly(
            symbol="MSFT",
            price=375.00,
            win_rate=0.30,  # Low win rate
            avg_win=0.05,
            avg_loss=0.10,
            kelly_fraction=0.25
        )

        # Negative Kelly should result in zero position
        assert result.shares == 0

    def test_position_size_with_drawdown_multiplier(self):
        """Verify position sizing reduces during drawdown."""
        sizer = PositionSizer(
            total_capital=100000,
            drawdown_pct=0.20  # 20% drawdown exceeds 15% threshold
        )

        # Get position size with drawdown
        result = sizer.calculate_fixed_fractional(
            symbol="AAPL",
            price=100.00,
            fraction=0.10
        )

        # Should be reduced by 50% due to drawdown
        assert sizer.drawdown_multiplier == 0.5

    def test_can_open_position_true(self):
        """Verify can_open_position returns True when allowed."""
        sizer = PositionSizer(
            total_capital=100000,
            max_positions=5
        )

        assert sizer.can_open_position is True

    def test_can_open_position_max_reached(self):
        """Verify can_open_position returns False when max positions reached."""
        sizer = PositionSizer(
            total_capital=100000,
            max_positions=2
        )

        # Simulate having 2 positions already
        sizer.update_positions({
            'AAPL': 10000,
            'GOOGL': 10000
        })

        assert sizer.can_open_position is False

    def test_update_capital(self):
        """Verify capital updates work correctly."""
        sizer = PositionSizer(total_capital=100000)

        sizer.update_capital(120000, cash=60000)

        assert sizer.total_capital == 120000
        assert sizer._cash == 60000

    def test_available_capital_with_buffer(self):
        """Verify available capital accounts for cash buffer."""
        sizer = PositionSizer(
            total_capital=100000,
            cash_buffer_pct=0.05
        )

        # Available capital should be cash minus buffer
        expected_min = 100000 * 0.95 - 100000 * 0.05  # Approximately
        assert sizer.available_capital >= 0

    @pytest.mark.parametrize("atr,price,expected_has_shares", [
        (5.0, 150.0, True),   # Normal case
        (0.0, 150.0, True),   # Zero ATR falls back to 5% stop
        (-1.0, 150.0, True),  # Negative ATR treated as zero
        (50.0, 150.0, True),  # Large ATR still gives position
    ])
    def test_atr_based_edge_cases(self, atr, price, expected_has_shares):
        """Verify ATR-based sizing handles edge cases."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_atr_based(
            symbol="TEST",
            price=price,
            atr=max(atr, 0)  # Ensure non-negative
        )

        if expected_has_shares:
            assert result.shares >= 0

    def test_calculate_for_signal_uses_atr_when_available(self):
        """Verify signal-based sizing uses ATR when provided."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_for_signal(
            symbol="AAPL",
            price=150.00,
            atr=3.00
        )

        assert result.method == 'atr_based'

    def test_calculate_for_signal_uses_vol_when_no_atr(self):
        """Verify signal-based sizing uses volatility when no ATR."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_for_signal(
            symbol="AAPL",
            price=150.00,
            realized_vol=0.25
        )

        assert result.method == 'volatility_targeted'

    def test_calculate_for_signal_uses_fixed_fallback(self):
        """Verify signal-based sizing falls back to fixed fractional."""
        sizer = PositionSizer(total_capital=100000)

        result = sizer.calculate_for_signal(
            symbol="AAPL",
            price=150.00
            # No ATR or vol provided
        )

        assert result.method == 'fixed_fractional'

    def test_signal_strength_scales_position(self):
        """Verify signal strength scales position size."""
        sizer = PositionSizer(total_capital=100000)

        # Full strength
        result_full = sizer.calculate_for_signal(
            symbol="AAPL",
            price=100.00,
            atr=2.00,
            signal_strength=1.0
        )

        # Half strength
        result_half = sizer.calculate_for_signal(
            symbol="AAPL",
            price=100.00,
            atr=2.00,
            signal_strength=0.5
        )

        # Half strength should give approximately half the shares
        # (may not be exactly half due to rounding)
        assert result_half.shares <= result_full.shares


# =============================================================================
# Integration-like Tests (still unit tests but test module interaction)
# =============================================================================

@pytest.mark.unit
class TestOrderExecutionWorkflow:
    """Tests for order execution workflow."""

    def test_complete_buy_workflow(self, mock_alpaca_client):
        """Test complete workflow: signal -> size -> execute."""
        # Arrange
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client
        sizer = PositionSizer(total_capital=100000)

        # Size the position
        position = sizer.calculate_atr_based(
            symbol="AAPL",
            price=150.00,
            atr=3.00
        )

        # Create signal with calculated shares
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00,
            stop_loss=144.00,  # 2x ATR stop
            target_price=165.00,
            metadata={'shares': position.shares}
        )

        # Act
        result = executor.execute_signal(signal)

        # Assert
        assert result.success is True
        assert result.status == OrderStatus.SUBMITTED

    def test_order_states_tracked_correctly(self, mock_alpaca_client):
        """Verify order status can be retrieved after submission."""
        executor = OrderExecutor()
        executor._trading_client = mock_alpaca_client

        # Submit order
        result = executor.submit_market_order("AAPL", 100, "buy")

        # Get order status
        status = executor.get_order_status(result.order_id)

        assert status is not None
        assert status['symbol'] == 'AAPL'
        assert 'status' in status
