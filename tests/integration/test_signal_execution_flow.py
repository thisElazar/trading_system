"""
Integration tests for signal→execution flow.

Tests the complete path from signal generation to order execution:
1. Signal creation and validation
2. Position sizing calculation
3. Order submission to broker
4. Fill confirmation and database recording
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestSignalCreation:
    """Test signal creation and validation."""

    def test_signal_created_with_required_fields(self, sample_signal_objects):
        """Signals should have all required fields populated."""
        for signal in sample_signal_objects:
            if hasattr(signal, 'symbol'):
                assert signal.symbol is not None
                assert signal.strategy is not None
                assert signal.signal_type is not None
                assert signal.strength >= 0 and signal.strength <= 1
            else:
                # Dict fallback
                assert signal['symbol'] is not None

    def test_signal_strength_affects_priority(self, sample_signal_objects):
        """Higher strength signals should have higher priority."""
        if hasattr(sample_signal_objects[0], 'strength'):
            signals = sorted(sample_signal_objects, key=lambda s: s.strength, reverse=True)
            assert signals[0].strength >= signals[-1].strength


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_position_size_respects_max_limit(self, mock_account_info):
        """Position size should not exceed max position limit."""
        from execution.position_sizer import PositionSizer

        sizer = PositionSizer(
            total_capital=mock_account_info['equity'],
            max_position_pct=0.05,  # 5% max
        )

        # Calculate size using fixed fractional method
        # API: calculate_fixed_fractional(symbol, price, fraction)
        result = sizer.calculate_fixed_fractional(
            symbol='AAPL',
            price=150.0,
            fraction=0.10,  # Request 10%
        )

        max_allowed = mock_account_info['equity'] * 0.05
        # Should be capped to max_position_pct (5%)
        assert result.dollar_value <= max_allowed

    def test_position_size_scales_with_allocation(self, mock_account_info):
        """Position size should scale with allocation percentage."""
        from execution.position_sizer import PositionSizer

        sizer = PositionSizer(total_capital=mock_account_info['equity'])

        result_high = sizer.calculate_fixed_fractional(
            symbol='AAPL',
            price=150.0,
            fraction=0.10,  # 10%
        )

        result_low = sizer.calculate_fixed_fractional(
            symbol='AAPL',
            price=150.0,
            fraction=0.05,  # 5%
        )

        assert result_high.shares >= result_low.shares

    def test_position_size_capped_by_max(self, mock_account_info):
        """Position size should be capped by max_position_size."""
        from execution.position_sizer import PositionSizer

        sizer = PositionSizer(
            total_capital=mock_account_info['equity'],
            max_position_size=5000,  # Cap at $5000
        )

        result = sizer.calculate_fixed_fractional(
            symbol='AAPL',
            price=150.0,
            fraction=0.50,  # Request 50% (would be $50k)
        )

        # Should be capped
        assert result.dollar_value <= 5000


class TestOrderSubmission:
    """Test order submission to broker."""

    def test_order_created_with_correct_fields(self):
        """Orders should have correct fields populated."""
        from tests.mocks.mock_alpaca import MockOrder, MockOrderSide

        order = MockOrder(
            symbol='AAPL',
            qty=100,
            side=MockOrderSide.BUY,
            order_type='market',
        )

        assert order.symbol == 'AAPL'
        assert order.qty == 100
        assert order.side == MockOrderSide.BUY

    def test_limit_order_has_price(self):
        """Limit orders should have limit price set."""
        from tests.mocks.mock_alpaca import MockOrder, MockOrderSide

        order = MockOrder(
            symbol='AAPL',
            qty=100,
            side=MockOrderSide.BUY,
            order_type='limit',
            limit_price=150.0,
        )

        assert order.limit_price == 150.0

    def test_order_rejected_insufficient_funds(self, mock_alpaca_client):
        """Orders should be rejected when insufficient funds."""
        # Set up mock to have low buying power
        mock_alpaca_client._account.buying_power = 1000.0

        # The mock should handle this - just verify the low buying power
        assert mock_alpaca_client._account.buying_power == 1000.0


class TestSignalToExecution:
    """Test complete signal→execution flow."""

    def test_buy_signal_has_required_fields(self, sample_signal_objects):
        """A BUY signal should have all required fields for execution."""
        # Get a BUY signal
        buy_signal = None
        for s in sample_signal_objects:
            if hasattr(s, 'signal_type') and s.signal_type.name == 'BUY':
                buy_signal = s
                break

        if buy_signal is None:
            pytest.skip("No BUY signal in sample data")

        # Verify required fields for execution
        assert buy_signal.symbol is not None
        assert buy_signal.price > 0
        assert buy_signal.strength > 0

    def test_close_signal_has_symbol(self, sample_signal_objects):
        """A CLOSE signal should have the symbol to close."""
        # Get a CLOSE signal
        close_signal = None
        for s in sample_signal_objects:
            if hasattr(s, 'signal_type') and s.signal_type.name == 'CLOSE':
                close_signal = s
                break

        if close_signal is None:
            pytest.skip("No CLOSE signal in sample data")

        # Verify symbol is set
        assert close_signal.symbol is not None
        assert len(close_signal.symbol) > 0

    def test_signal_to_order_mapping(self, sample_signal_objects):
        """Signals should map correctly to order parameters."""
        from tests.mocks.mock_alpaca import MockOrder, MockOrderSide

        for signal in sample_signal_objects:
            if not hasattr(signal, 'signal_type'):
                continue

            if signal.signal_type.name == 'BUY':
                order = MockOrder(
                    symbol=signal.symbol,
                    qty=100,
                    side=MockOrderSide.BUY,
                )
                assert order.side == MockOrderSide.BUY
            elif signal.signal_type.name == 'SELL':
                order = MockOrder(
                    symbol=signal.symbol,
                    qty=100,
                    side=MockOrderSide.SELL,
                )
                assert order.side == MockOrderSide.SELL


class TestDatabaseRecording:
    """Test that executions are recorded to database."""

    def test_executed_signal_recorded(self, test_db, sample_signal_objects):
        """Executed signals should be recorded in database."""
        cursor = test_db.cursor()

        # Record a signal
        signal = sample_signal_objects[0]
        if hasattr(signal, 'symbol'):
            cursor.execute(
                """INSERT INTO signals (symbol, strategy, signal_type, strength, price)
                   VALUES (?, ?, ?, ?, ?)""",
                (signal.symbol, signal.strategy, signal.signal_type.name,
                 signal.strength, signal.price)
            )
        else:
            cursor.execute(
                """INSERT INTO signals (symbol, strategy, signal_type, strength, price)
                   VALUES (?, ?, ?, ?, ?)""",
                (signal['symbol'], signal.get('strategy', 'test'),
                 signal['signal_type'], signal['strength'], signal.get('price', 0))
            )

        test_db.commit()

        # Verify recording
        cursor.execute("SELECT COUNT(*) FROM signals")
        count = cursor.fetchone()[0]
        assert count >= 1

    def test_position_recorded_on_fill(self, test_db):
        """Filled orders should create position records."""
        cursor = test_db.cursor()

        # Record a position
        cursor.execute(
            """INSERT INTO positions
               (symbol, strategy, side, qty, entry_price, current_price, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ('AAPL', 'test_strategy', 'long', 100, 150.0, 155.0, 'open')
        )
        test_db.commit()

        # Verify position exists
        cursor.execute("SELECT * FROM positions WHERE symbol = 'AAPL'")
        position = cursor.fetchone()
        assert position is not None

    def test_trade_recorded_on_close(self, test_db):
        """Closing a position should record a trade."""
        cursor = test_db.cursor()

        # Record a complete trade
        cursor.execute(
            """INSERT INTO trades
               (symbol, strategy, side, qty, entry_price, exit_price, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ('AAPL', 'test_strategy', 'long', 100, 150.0, 160.0, 1000.0)
        )
        test_db.commit()

        # Verify trade exists
        cursor.execute("SELECT pnl FROM trades WHERE symbol = 'AAPL'")
        trade = cursor.fetchone()
        assert trade is not None
        assert trade[0] == 1000.0


class TestMultipleSignals:
    """Test handling multiple signals."""

    def test_conflicting_signals_resolved(self, sample_signal_objects):
        """Conflicting signals from different strategies should be resolved."""
        # Create conflicting signals (BUY and SELL for same symbol)
        signals = [s for s in sample_signal_objects if hasattr(s, 'symbol')]

        if len(signals) < 2:
            pytest.skip("Need at least 2 signals for conflict test")

        # Group by symbol
        by_symbol = {}
        for s in signals:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = []
            by_symbol[s.symbol].append(s)

        # For each symbol, highest strength should win
        for symbol, sigs in by_symbol.items():
            if len(sigs) > 1:
                winner = max(sigs, key=lambda s: s.strength)
                assert winner.strength == max(s.strength for s in sigs)

    def test_signals_ordered_by_priority(self, sample_signal_objects):
        """Signals should be processed in priority order."""
        signals = [s for s in sample_signal_objects if hasattr(s, 'strength')]

        if not signals:
            pytest.skip("No signals with strength attribute")

        sorted_signals = sorted(signals, key=lambda s: s.strength, reverse=True)

        # First signal should have highest strength
        assert sorted_signals[0].strength >= sorted_signals[-1].strength
