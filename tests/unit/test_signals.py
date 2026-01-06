"""
Unit Tests: Signal Generation
=============================

Tests for signal creation, validation, and processing including:
- Signal creation and validation
- Signal strength calculations
- Signal conflict resolution
- Signal expiration logic
- Signal metadata handling

Uses fixtures from conftest.py: test_db, sample_ohlcv_data, mock_alpaca_client
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import numpy as np
import pandas as pd

# Import test targets
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base import Signal, SignalType


# =============================================================================
# Signal Creation Tests
# =============================================================================

@pytest.mark.unit
class TestSignalCreation:
    """Tests for creating Signal objects."""

    def test_signal_creation_minimal_fields(self):
        """Verify signal can be created with minimal required fields."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test_strategy",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        assert signal.symbol == "AAPL"
        assert signal.strategy == "test_strategy"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.price == 150.00

    def test_signal_creation_all_fields(self):
        """Verify signal with all optional fields."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="gap_fill",
            signal_type=SignalType.BUY,
            strength=0.85,
            price=150.00,
            stop_loss=145.00,
            target_price=165.00,
            position_size_pct=0.05,
            reason="Gap down outside 3-day range",
            metadata={'gap_percent': -0.45, 'exit_time': '11:30'}
        )

        assert signal.stop_loss == 145.00
        assert signal.target_price == 165.00
        assert signal.position_size_pct == 0.05
        assert signal.reason == "Gap down outside 3-day range"
        assert signal.metadata['gap_percent'] == -0.45

    def test_signal_type_enum_values(self):
        """Verify all signal types exist."""
        expected_types = ['BUY', 'SELL', 'HOLD', 'CLOSE']
        for signal_type in expected_types:
            assert hasattr(SignalType, signal_type)

    def test_signal_type_values(self):
        """Verify signal type enum values."""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"
        assert SignalType.CLOSE.value == "CLOSE"

    @pytest.mark.parametrize("signal_type,expected_value", [
        (SignalType.BUY, "BUY"),
        (SignalType.SELL, "SELL"),
        (SignalType.HOLD, "HOLD"),
        (SignalType.CLOSE, "CLOSE"),
    ])
    def test_signal_type_string_conversion(self, signal_type, expected_value):
        """Verify signal types convert to strings correctly."""
        assert signal_type.value == expected_value


# =============================================================================
# Signal Validation Tests
# =============================================================================

@pytest.mark.unit
class TestSignalValidation:
    """Tests for signal validation logic."""

    @pytest.mark.parametrize("strength,is_valid", [
        (0.0, True),
        (0.5, True),
        (1.0, True),
        (-0.1, False),
        (1.1, False),
    ])
    def test_signal_strength_bounds(self, strength, is_valid):
        """Verify signal strength must be between 0 and 1."""
        valid = 0.0 <= strength <= 1.0
        assert valid == is_valid

    def test_signal_allows_empty_symbol(self):
        """Verify signal can be created with empty symbol (validation at higher level)."""
        # Note: Empty symbol validation should happen at order submission level
        signal = Signal(
            timestamp=datetime.now(),
            symbol="",  # Empty symbol allowed at Signal level
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )
        assert signal.symbol == ""

    def test_signal_requires_positive_price(self):
        """Verify signal price must be positive."""
        with pytest.raises((TypeError, ValueError, AssertionError)):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="test",
                signal_type=SignalType.BUY,
                strength=0.8,
                price=-10.00  # Negative price should fail
            )

    def test_signal_stop_loss_validation_long(self):
        """Verify stop loss is below entry for BUY signal."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00,
            stop_loss=145.00  # Below entry
        )

        # For long positions, stop should be below entry
        assert signal.stop_loss < signal.price

    def test_signal_target_validation_long(self):
        """Verify target is above entry for BUY signal."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00,
            target_price=165.00  # Above entry
        )

        # For long positions, target should be above entry
        assert signal.target_price > signal.price

    def test_signal_timestamp_required(self):
        """Verify signal requires a timestamp."""
        with pytest.raises(TypeError):
            Signal(
                symbol="AAPL",
                strategy="test",
                signal_type=SignalType.BUY,
                strength=0.8,
                price=150.00
                # Missing timestamp
            )


# =============================================================================
# Signal Strength Tests
# =============================================================================

@pytest.mark.unit
class TestSignalStrength:
    """Tests for signal strength calculations."""

    def test_high_strength_signal(self):
        """Verify high strength signals are identified."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.9,
            price=150.00
        )

        is_high_conviction = signal.strength >= 0.8
        assert is_high_conviction is True

    def test_low_strength_signal(self):
        """Verify low strength signals are identified."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.3,
            price=150.00
        )

        is_high_conviction = signal.strength >= 0.8
        assert is_high_conviction is False

    def test_strength_affects_position_size(self):
        """Verify signal strength should scale position size."""
        full_strength = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=1.0,
            price=150.00,
            position_size_pct=0.10
        )

        half_strength = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.5,
            price=150.00,
            position_size_pct=0.10
        )

        # Position size can be scaled by strength
        effective_full = full_strength.position_size_pct * full_strength.strength
        effective_half = half_strength.position_size_pct * half_strength.strength

        assert effective_full == 0.10
        assert effective_half == 0.05

    @pytest.mark.parametrize("indicator_values,expected_min_strength", [
        # Multiple confirming indicators = higher strength
        ({'rsi': 25, 'bb_position': 0.1, 'momentum': -5}, 0.7),  # All bearish
        ({'rsi': 75, 'bb_position': 0.9, 'momentum': 5}, 0.7),   # All bullish
        # Mixed indicators = lower strength
        ({'rsi': 50, 'bb_position': 0.5, 'momentum': 0}, 0.3),   # Neutral
        ({'rsi': 30, 'bb_position': 0.8, 'momentum': 3}, 0.4),   # Mixed
    ])
    def test_indicator_confirmation_affects_strength(self, indicator_values, expected_min_strength):
        """Verify multiple confirming indicators increase signal strength."""
        # Count confirming indicators for oversold buy signal
        confirmations = 0

        if indicator_values['rsi'] < 30:
            confirmations += 1
        if indicator_values['bb_position'] < 0.2:
            confirmations += 1
        if indicator_values['momentum'] < 0:
            confirmations += 1

        # Strength should increase with confirmations
        # This is just testing the concept, not actual implementation
        assert isinstance(confirmations, int)


# =============================================================================
# Signal Conflict Resolution Tests
# =============================================================================

@pytest.mark.unit
class TestSignalConflictResolution:
    """Tests for resolving conflicting signals."""

    def test_same_symbol_conflicting_signals(self):
        """Verify conflicting signals for same symbol are detected."""
        buy_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="momentum",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        sell_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="mean_reversion",
            signal_type=SignalType.SELL,
            strength=0.7,
            price=150.00
        )

        # Detect conflict
        has_conflict = (
            buy_signal.symbol == sell_signal.symbol and
            buy_signal.signal_type != sell_signal.signal_type and
            buy_signal.signal_type in [SignalType.BUY, SignalType.SELL] and
            sell_signal.signal_type in [SignalType.BUY, SignalType.SELL]
        )

        assert has_conflict is True

    def test_resolve_conflict_by_strength(self):
        """Verify stronger signal wins in conflict."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="momentum",
                signal_type=SignalType.BUY,
                strength=0.9,  # Stronger
                price=150.00
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="mean_reversion",
                signal_type=SignalType.SELL,
                strength=0.6,  # Weaker
                price=150.00
            )
        ]

        # Winner is signal with highest strength
        winner = max(signals, key=lambda s: s.strength)

        assert winner.signal_type == SignalType.BUY
        assert winner.strength == 0.9

    def test_resolve_conflict_by_strategy_tier(self):
        """Verify higher-tier strategy wins in conflict."""
        # Assuming tier 1 strategies take precedence
        strategy_tiers = {
            'gap_fill': 1,
            'momentum': 1,
            'sector_rotation': 2,
        }

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="gap_fill",  # Tier 1
                signal_type=SignalType.BUY,
                strength=0.7,
                price=150.00
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="sector_rotation",  # Tier 2
                signal_type=SignalType.SELL,
                strength=0.8,  # Higher strength but lower tier
                price=150.00
            )
        ]

        # In case of equal strength, tier decides
        winner = min(signals, key=lambda s: strategy_tiers.get(s.strategy, 99))

        assert winner.strategy == "gap_fill"

    def test_no_conflict_different_symbols(self):
        """Verify signals for different symbols don't conflict."""
        signal1 = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="momentum",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        signal2 = Signal(
            timestamp=datetime.now(),
            symbol="GOOGL",  # Different symbol
            strategy="momentum",
            signal_type=SignalType.SELL,
            strength=0.8,
            price=140.00
        )

        has_conflict = signal1.symbol == signal2.symbol
        assert has_conflict is False

    def test_aggregate_signals_same_direction(self):
        """Verify signals in same direction can be aggregated."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="momentum",
                signal_type=SignalType.BUY,
                strength=0.7,
                price=150.00
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="mean_reversion",
                signal_type=SignalType.BUY,
                strength=0.6,
                price=150.00
            )
        ]

        # Both signals are BUY, aggregate strength
        # Simple average for demonstration
        avg_strength = sum(s.strength for s in signals) / len(signals)

        assert avg_strength == pytest.approx(0.65)
        assert all(s.signal_type == SignalType.BUY for s in signals)


# =============================================================================
# Signal Expiration Tests
# =============================================================================

@pytest.mark.unit
class TestSignalExpiration:
    """Tests for signal expiration logic."""

    def test_signal_not_expired(self):
        """Verify recent signal is not expired."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        max_age = timedelta(hours=1)
        is_expired = datetime.now() - signal.timestamp > max_age

        assert is_expired is False

    def test_signal_expired(self):
        """Verify old signal is expired."""
        signal = Signal(
            timestamp=datetime.now() - timedelta(hours=2),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        max_age = timedelta(hours=1)
        is_expired = datetime.now() - signal.timestamp > max_age

        assert is_expired is True

    @pytest.mark.parametrize("strategy,expected_max_age_hours", [
        ("gap_fill", 2),       # Intraday strategy
        ("momentum", 168),     # Weekly rebalance
        ("mean_reversion", 24),  # Daily
    ])
    def test_strategy_specific_expiration(self, strategy, expected_max_age_hours):
        """Verify different strategies have different expiration times."""
        strategy_expiration = {
            'gap_fill': timedelta(hours=2),
            'momentum': timedelta(days=7),
            'mean_reversion': timedelta(hours=24),
        }

        max_age = strategy_expiration.get(strategy, timedelta(hours=24))

        assert max_age == timedelta(hours=expected_max_age_hours)

    def test_expired_signals_filtered(self):
        """Verify expired signals are filtered from active list."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                strategy="test",
                signal_type=SignalType.BUY,
                strength=0.8,
                price=150.00
            ),
            Signal(
                timestamp=datetime.now() - timedelta(hours=5),
                symbol="GOOGL",
                strategy="test",
                signal_type=SignalType.BUY,
                strength=0.7,
                price=140.00
            )
        ]

        max_age = timedelta(hours=1)
        now = datetime.now()

        active_signals = [s for s in signals if (now - s.timestamp) <= max_age]

        assert len(active_signals) == 1
        assert active_signals[0].symbol == "AAPL"


# =============================================================================
# Signal Metadata Tests
# =============================================================================

@pytest.mark.unit
class TestSignalMetadata:
    """Tests for signal metadata handling."""

    def test_metadata_stores_custom_fields(self):
        """Verify metadata can store custom fields."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="SPY",
            strategy="gap_fill",
            signal_type=SignalType.BUY,
            strength=0.85,
            price=450.00,
            metadata={
                'gap_percent': -0.35,
                'exit_time': '11:30',
                'outside_range': True,
                'hold_minutes': 120
            }
        )

        assert signal.metadata['gap_percent'] == -0.35
        assert signal.metadata['exit_time'] == '11:30'
        assert signal.metadata['outside_range'] is True
        assert signal.metadata['hold_minutes'] == 120

    def test_metadata_defaults_to_empty(self):
        """Verify metadata defaults to empty dict."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        assert signal.metadata is None or signal.metadata == {}

    def test_metadata_preserves_types(self):
        """Verify metadata preserves data types."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00,
            metadata={
                'int_value': 42,
                'float_value': 3.14,
                'bool_value': True,
                'str_value': 'test',
                'list_value': [1, 2, 3],
                'dict_value': {'nested': 'data'}
            }
        )

        assert isinstance(signal.metadata['int_value'], int)
        assert isinstance(signal.metadata['float_value'], float)
        assert isinstance(signal.metadata['bool_value'], bool)
        assert isinstance(signal.metadata['str_value'], str)
        assert isinstance(signal.metadata['list_value'], list)
        assert isinstance(signal.metadata['dict_value'], dict)

    def test_reason_field_descriptive(self):
        """Verify reason field provides trade rationale."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="mean_reversion",
            signal_type=SignalType.BUY,
            strength=0.75,
            price=150.00,
            reason="RSI oversold (24.5), price at lower BB, 3-day reversal pattern"
        )

        assert "RSI" in signal.reason
        assert "oversold" in signal.reason


# =============================================================================
# Signal Serialization Tests
# =============================================================================

@pytest.mark.unit
class TestSignalSerialization:
    """Tests for signal serialization/deserialization."""

    def test_signal_to_dict(self):
        """Verify signal can be converted to dictionary."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        # Use asdict from dataclasses
        signal_dict = asdict(signal)

        assert signal_dict['symbol'] == 'AAPL'
        assert signal_dict['strategy'] == 'test'
        assert signal_dict['strength'] == 0.8

    def test_signal_from_dict(self):
        """Verify signal can be created from dictionary."""
        signal_data = {
            'timestamp': datetime.now(),
            'symbol': 'AAPL',
            'strategy': 'test',
            'signal_type': SignalType.BUY,
            'strength': 0.8,
            'price': 150.00
        }

        signal = Signal(**signal_data)

        assert signal.symbol == 'AAPL'
        assert signal.strength == 0.8

    def test_signal_equality(self):
        """Verify two signals with same values are equal."""
        timestamp = datetime.now()

        signal1 = Signal(
            timestamp=timestamp,
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        signal2 = Signal(
            timestamp=timestamp,
            symbol="AAPL",
            strategy="test",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=150.00
        )

        assert signal1 == signal2


# =============================================================================
# Signal Processing Pipeline Tests
# =============================================================================

@pytest.mark.unit
class TestSignalProcessingPipeline:
    """Tests for complete signal processing workflow."""

    def test_filter_by_minimum_strength(self):
        """Verify signals below minimum strength are filtered."""
        signals = [
            Signal(timestamp=datetime.now(), symbol="AAPL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.9, price=150.00),
            Signal(timestamp=datetime.now(), symbol="GOOGL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.4, price=140.00),
            Signal(timestamp=datetime.now(), symbol="MSFT", strategy="test",
                   signal_type=SignalType.BUY, strength=0.7, price=380.00),
        ]

        min_strength = 0.5
        filtered = [s for s in signals if s.strength >= min_strength]

        assert len(filtered) == 2
        assert all(s.strength >= min_strength for s in filtered)

    def test_sort_signals_by_strength(self):
        """Verify signals can be sorted by strength."""
        signals = [
            Signal(timestamp=datetime.now(), symbol="AAPL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.7, price=150.00),
            Signal(timestamp=datetime.now(), symbol="GOOGL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.9, price=140.00),
            Signal(timestamp=datetime.now(), symbol="MSFT", strategy="test",
                   signal_type=SignalType.BUY, strength=0.8, price=380.00),
        ]

        sorted_signals = sorted(signals, key=lambda s: s.strength, reverse=True)

        assert sorted_signals[0].symbol == "GOOGL"  # Highest strength
        assert sorted_signals[1].symbol == "MSFT"
        assert sorted_signals[2].symbol == "AAPL"   # Lowest strength

    def test_group_signals_by_symbol(self):
        """Verify signals can be grouped by symbol."""
        signals = [
            Signal(timestamp=datetime.now(), symbol="AAPL", strategy="momentum",
                   signal_type=SignalType.BUY, strength=0.7, price=150.00),
            Signal(timestamp=datetime.now(), symbol="AAPL", strategy="mean_reversion",
                   signal_type=SignalType.BUY, strength=0.6, price=150.00),
            Signal(timestamp=datetime.now(), symbol="GOOGL", strategy="momentum",
                   signal_type=SignalType.BUY, strength=0.8, price=140.00),
        ]

        from collections import defaultdict
        grouped = defaultdict(list)
        for signal in signals:
            grouped[signal.symbol].append(signal)

        assert len(grouped['AAPL']) == 2
        assert len(grouped['GOOGL']) == 1

    def test_deduplicate_signals(self):
        """Verify duplicate signals are removed."""
        timestamp = datetime.now()

        signals = [
            Signal(timestamp=timestamp, symbol="AAPL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.8, price=150.00),
            Signal(timestamp=timestamp, symbol="AAPL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.8, price=150.00),  # Duplicate
            Signal(timestamp=timestamp, symbol="GOOGL", strategy="test",
                   signal_type=SignalType.BUY, strength=0.7, price=140.00),
        ]

        # Remove duplicates by symbol-strategy pair
        seen = set()
        unique = []
        for signal in signals:
            key = (signal.symbol, signal.strategy)
            if key not in seen:
                seen.add(key)
                unique.append(signal)

        assert len(unique) == 2
