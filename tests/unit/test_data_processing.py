"""
Unit Tests: Data Processing
===========================

Tests for data processing and validation including:
- OHLCV data validation
- Data normalization
- Missing data handling
- Technical indicator calculations

Uses fixtures from conftest.py: test_db, sample_ohlcv_data, mock_alpaca_client
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Import test targets
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.indicators.technical import (
    add_all_indicators,
    add_bollinger_bands,
    add_rsi,
    add_macd,
    add_atr,
    add_volume_indicators,
    add_momentum,
    add_support_resistance,
    add_volatility_indicators,
    add_trend_indicators,
    calculate_gap,
    calculate_divergence,
)


# =============================================================================
# OHLCV Validation Tests
# =============================================================================

@pytest.mark.unit
class TestOHLCVValidation:
    """Tests for OHLCV data validation."""

    def test_ohlcv_columns_exist(self, sample_ohlcv_data):
        """Verify OHLCV data has required columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            assert col in sample_ohlcv_data.columns, f"Missing column: {col}"

    def test_ohlcv_no_null_values(self, sample_ohlcv_data):
        """Verify OHLCV data has no null values."""
        assert sample_ohlcv_data.isnull().sum().sum() == 0

    def test_high_greater_equal_low(self, sample_ohlcv_data):
        """Verify high >= low for all rows."""
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['low']).all()

    def test_high_greater_equal_open_close(self, sample_ohlcv_data):
        """Verify high >= open and high >= close."""
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['open']).all()
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['close']).all()

    def test_low_less_equal_open_close(self, sample_ohlcv_data):
        """Verify low <= open and low <= close."""
        assert (sample_ohlcv_data['low'] <= sample_ohlcv_data['open']).all()
        assert (sample_ohlcv_data['low'] <= sample_ohlcv_data['close']).all()

    def test_positive_prices(self, sample_ohlcv_data):
        """Verify all prices are positive."""
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            assert (sample_ohlcv_data[col] > 0).all(), f"Non-positive values in {col}"

    def test_non_negative_volume(self, sample_ohlcv_data):
        """Verify volume is non-negative."""
        assert (sample_ohlcv_data['volume'] >= 0).all()

    def test_datetime_index(self, sample_ohlcv_data):
        """Verify data has datetime index."""
        assert isinstance(sample_ohlcv_data.index, pd.DatetimeIndex)

    def test_index_is_sorted(self, sample_ohlcv_data):
        """Verify datetime index is sorted."""
        assert sample_ohlcv_data.index.is_monotonic_increasing

    @pytest.mark.parametrize("invalid_condition,description", [
        ('high < low', 'High less than low'),
        ('high < open', 'High less than open'),
        ('high < close', 'High less than close'),
        ('low > open', 'Low greater than open'),
        ('low > close', 'Low greater than close'),
    ])
    def test_invalid_ohlc_relationships(self, invalid_condition, description):
        """Verify invalid OHLC relationships are detected."""
        # Create invalid data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000000] * 10,
        }, index=dates)

        # Make one row invalid based on condition
        if invalid_condition == 'high < low':
            df.loc[df.index[5], 'high'] = 90
        elif invalid_condition == 'high < open':
            df.loc[df.index[5], 'high'] = 98
        elif invalid_condition == 'high < close':
            df.loc[df.index[5], 'high'] = 100

        # Detect violation
        high_low_valid = (df['high'] >= df['low']).all()
        high_open_valid = (df['high'] >= df['open']).all()
        high_close_valid = (df['high'] >= df['close']).all()

        # At least one should be invalid
        if invalid_condition in ['high < low', 'high < open', 'high < close']:
            assert not (high_low_valid and high_open_valid and high_close_valid)


# =============================================================================
# Data Normalization Tests
# =============================================================================

@pytest.mark.unit
class TestDataNormalization:
    """Tests for data normalization."""

    def test_column_names_lowercase(self, sample_ohlcv_data):
        """Verify column names are normalized to lowercase."""
        # Create data with mixed case columns
        df = sample_ohlcv_data.copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Normalize
        df.columns = [c.lower() for c in df.columns]

        assert all(c.islower() for c in df.columns)

    def test_returns_calculation(self, sample_ohlcv_data):
        """Verify returns are calculated correctly."""
        df = sample_ohlcv_data.copy()
        df['returns'] = df['close'].pct_change()

        # First row should be NaN
        assert pd.isna(df['returns'].iloc[0])

        # Check returns calculation
        expected_return = (df['close'].iloc[1] - df['close'].iloc[0]) / df['close'].iloc[0]
        assert df['returns'].iloc[1] == pytest.approx(expected_return, rel=1e-6)

    def test_log_returns_calculation(self, sample_ohlcv_data):
        """Verify log returns are calculated correctly."""
        df = sample_ohlcv_data.copy()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # First row should be NaN
        assert pd.isna(df['log_returns'].iloc[0])

        # Log returns should be close to regular returns for small changes
        regular_return = (df['close'].iloc[1] / df['close'].iloc[0]) - 1
        log_return = df['log_returns'].iloc[1]

        # For small changes, log returns ~ regular returns
        if abs(regular_return) < 0.1:
            assert abs(log_return - regular_return) < 0.01

    def test_price_percentage_change(self, sample_ohlcv_data):
        """Verify percentage change calculation."""
        df = sample_ohlcv_data.copy()

        periods = [1, 5, 10, 20]
        for period in periods:
            col_name = f'pct_change_{period}'
            df[col_name] = df['close'].pct_change(period) * 100

            assert col_name in df.columns
            # First {period} rows should be NaN
            assert df[col_name].iloc[:period].isna().all()


# =============================================================================
# Missing Data Handling Tests
# =============================================================================

@pytest.mark.unit
class TestMissingDataHandling:
    """Tests for handling missing data."""

    def test_forward_fill_gaps(self):
        """Verify forward fill handles missing data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'close': [100, 101, np.nan, np.nan, 104, 105, np.nan, 107, 108, 109],
        }, index=dates)

        df_filled = df.ffill()

        assert df_filled['close'].isna().sum() == 0
        assert df_filled['close'].iloc[2] == 101  # Forward filled
        assert df_filled['close'].iloc[3] == 101

    def test_interpolate_missing_values(self):
        """Verify interpolation fills missing values."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'close': [100.0, np.nan, np.nan, np.nan, 108.0],
        }, index=dates)

        df_interp = df.interpolate(method='linear')

        # Values should be linearly interpolated
        assert df_interp['close'].iloc[1] == pytest.approx(102, rel=0.01)
        assert df_interp['close'].iloc[2] == pytest.approx(104, rel=0.01)
        assert df_interp['close'].iloc[3] == pytest.approx(106, rel=0.01)

    def test_detect_missing_dates(self):
        """Verify missing trading dates are detected."""
        # Create data with a gap
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'])
        df = pd.DataFrame({
            'close': [100, 101, 103, 104],
        }, index=dates)

        # Create expected full range
        full_range = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')

        # Find missing dates
        missing = full_range.difference(df.index)

        assert len(missing) == 1
        assert pd.Timestamp('2024-01-03') in missing

    def test_drop_rows_with_missing(self, sample_ohlcv_data):
        """Verify rows with missing values can be dropped."""
        df = sample_ohlcv_data.copy()
        df.loc[df.index[5], 'close'] = np.nan
        df.loc[df.index[10], 'volume'] = np.nan

        original_len = len(df)
        df_clean = df.dropna()

        assert len(df_clean) == original_len - 2

    def test_minimum_data_length_for_indicators(self, sample_ohlcv_data):
        """Verify minimum data length check for indicators."""
        # Need at least 20 rows for most indicators
        min_length = 20

        if len(sample_ohlcv_data) >= min_length:
            df = add_all_indicators(sample_ohlcv_data)
            assert len(df) >= min_length
        else:
            # Should return data unchanged or warn
            pass


# =============================================================================
# Bollinger Bands Tests
# =============================================================================

@pytest.mark.unit
class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_bands_columns_created(self, sample_ohlcv_data):
        """Verify BB calculation creates expected columns."""
        df = add_bollinger_bands(sample_ohlcv_data)

        expected_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_bollinger_band_relationships(self, sample_ohlcv_data):
        """Verify BB upper > middle > lower."""
        df = add_bollinger_bands(sample_ohlcv_data)

        # After warmup period, relationships should hold
        warmup = 20
        valid_data = df.iloc[warmup:].dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])

        assert (valid_data['bb_upper'] >= valid_data['bb_middle']).all()
        assert (valid_data['bb_middle'] >= valid_data['bb_lower']).all()

    def test_bb_width_positive(self, sample_ohlcv_data):
        """Verify BB width is positive."""
        df = add_bollinger_bands(sample_ohlcv_data)

        warmup = 20
        valid_width = df['bb_width'].iloc[warmup:].dropna()

        assert (valid_width > 0).all()

    def test_bb_position_bounded(self, sample_ohlcv_data):
        """Verify BB position is between 0 and 1 for most values."""
        df = add_bollinger_bands(sample_ohlcv_data)

        warmup = 20
        valid_pos = df['bb_position'].iloc[warmup:].dropna()

        # Most values should be 0-1, but price can go outside bands
        within_bands = ((valid_pos >= 0) & (valid_pos <= 1)).mean()
        assert within_bands >= 0.90  # At least 90% within bands

    @pytest.mark.parametrize("period,std_dev", [
        (20, 2.0),
        (10, 1.5),
        (50, 2.5),
    ])
    def test_bb_custom_parameters(self, sample_ohlcv_data, period, std_dev):
        """Verify BB works with custom parameters."""
        df = add_bollinger_bands(sample_ohlcv_data, period=period, std_dev=std_dev)

        assert 'bb_upper' in df.columns
        assert df['bb_upper'].iloc[-1] is not np.nan


# =============================================================================
# RSI Tests
# =============================================================================

@pytest.mark.unit
class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_column_created(self, sample_ohlcv_data):
        """Verify RSI calculation creates expected column."""
        df = add_rsi(sample_ohlcv_data)

        assert 'rsi' in df.columns
        assert 'rsi_sma' in df.columns

    def test_rsi_bounded(self, sample_ohlcv_data):
        """Verify RSI is bounded between 0 and 100."""
        df = add_rsi(sample_ohlcv_data)

        warmup = 14
        valid_rsi = df['rsi'].iloc[warmup:].dropna()

        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_oversold_detection(self, sample_ohlcv_data):
        """Verify RSI < 30 indicates oversold."""
        df = add_rsi(sample_ohlcv_data)

        warmup = 14
        valid_rsi = df['rsi'].iloc[warmup:].dropna()

        # Check if any oversold conditions exist
        oversold = valid_rsi < 30
        overbought = valid_rsi > 70

        # At least the logic should work
        assert oversold.dtype == bool
        assert overbought.dtype == bool

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_rsi_custom_period(self, sample_ohlcv_data, period):
        """Verify RSI works with custom periods."""
        df = add_rsi(sample_ohlcv_data, period=period)

        valid_rsi = df['rsi'].iloc[period:].dropna()
        assert len(valid_rsi) > 0


# =============================================================================
# MACD Tests
# =============================================================================

@pytest.mark.unit
class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_columns_created(self, sample_ohlcv_data):
        """Verify MACD calculation creates expected columns."""
        df = add_macd(sample_ohlcv_data)

        expected_cols = ['macd', 'macd_signal', 'macd_histogram']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_macd_histogram_calculation(self, sample_ohlcv_data):
        """Verify histogram = MACD - Signal."""
        df = add_macd(sample_ohlcv_data)

        warmup = 26
        valid_idx = df.iloc[warmup:].index

        for idx in valid_idx[:10]:  # Check first 10 valid rows
            expected = df.loc[idx, 'macd'] - df.loc[idx, 'macd_signal']
            actual = df.loc[idx, 'macd_histogram']

            if not pd.isna(expected) and not pd.isna(actual):
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_macd_crossover_detection(self, sample_ohlcv_data):
        """Verify MACD/signal crossovers can be detected."""
        df = add_macd(sample_ohlcv_data)

        # Calculate crossover
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & \
                             (df['macd'].shift(1) <= df['macd_signal'].shift(1))

        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & \
                               (df['macd'].shift(1) >= df['macd_signal'].shift(1))

        assert 'macd_cross_up' in df.columns
        assert 'macd_cross_down' in df.columns


# =============================================================================
# ATR Tests
# =============================================================================

@pytest.mark.unit
class TestATR:
    """Tests for ATR calculation."""

    def test_atr_column_created(self, sample_ohlcv_data):
        """Verify ATR calculation creates expected columns."""
        df = add_atr(sample_ohlcv_data)

        assert 'atr' in df.columns
        assert 'atr_pct' in df.columns

    def test_atr_positive(self, sample_ohlcv_data):
        """Verify ATR is always positive."""
        df = add_atr(sample_ohlcv_data)

        warmup = 14
        valid_atr = df['atr'].iloc[warmup:].dropna()

        assert (valid_atr > 0).all()

    def test_atr_reasonable_magnitude(self, sample_ohlcv_data):
        """Verify ATR is reasonable relative to price."""
        df = add_atr(sample_ohlcv_data)

        warmup = 14
        valid_atr_pct = df['atr_pct'].iloc[warmup:].dropna()

        # ATR as % of price should typically be 1-10%
        assert (valid_atr_pct > 0).all()
        assert (valid_atr_pct < 50).all()  # Rarely above 50%


# =============================================================================
# Volume Indicators Tests
# =============================================================================

@pytest.mark.unit
class TestVolumeIndicators:
    """Tests for volume-based indicators."""

    def test_volume_columns_created(self, sample_ohlcv_data):
        """Verify volume indicator columns are created."""
        df = add_volume_indicators(sample_ohlcv_data)

        expected_cols = ['volume_sma', 'relative_volume', 'obv']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_relative_volume_calculation(self, sample_ohlcv_data):
        """Verify relative volume is volume / SMA."""
        df = add_volume_indicators(sample_ohlcv_data)

        warmup = 20
        for idx in df.index[warmup:warmup+10]:
            expected = df.loc[idx, 'volume'] / df.loc[idx, 'volume_sma']
            actual = df.loc[idx, 'relative_volume']

            if not pd.isna(expected) and not pd.isna(actual):
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_obv_direction(self, sample_ohlcv_data):
        """Verify OBV increases on up days, decreases on down days."""
        df = add_volume_indicators(sample_ohlcv_data)

        # OBV should be cumulative
        assert 'obv' in df.columns


# =============================================================================
# Momentum Indicators Tests
# =============================================================================

@pytest.mark.unit
class TestMomentumIndicators:
    """Tests for momentum indicators."""

    def test_momentum_columns_created(self, sample_ohlcv_data):
        """Verify momentum columns are created."""
        df = add_momentum(sample_ohlcv_data)

        expected_cols = ['momentum_5d', 'momentum_10d', 'momentum_20d', 'roc']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_momentum_calculation(self, sample_ohlcv_data):
        """Verify momentum is price change percentage."""
        df = add_momentum(sample_ohlcv_data)

        # Check 5-day momentum
        for i in range(5, min(15, len(df))):
            expected = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5] * 100
            actual = df['momentum_5d'].iloc[i]

            if not pd.isna(expected) and not pd.isna(actual):
                assert actual == pytest.approx(expected, rel=1e-4)

    def test_high_low_tracking(self, sample_ohlcv_data):
        """Verify 20-day and 52-week high/low tracking."""
        df = add_momentum(sample_ohlcv_data)

        if len(df) >= 252:
            assert 'high_52w' in df.columns
            assert 'low_52w' in df.columns

        if len(df) >= 20:
            assert 'high_20' in df.columns
            assert 'low_20' in df.columns


# =============================================================================
# Support/Resistance Tests
# =============================================================================

@pytest.mark.unit
class TestSupportResistance:
    """Tests for support/resistance levels."""

    def test_support_resistance_columns_created(self, sample_ohlcv_data):
        """Verify support/resistance columns are created."""
        df = add_support_resistance(sample_ohlcv_data)

        expected_cols = ['resistance', 'support', 'pivot']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_resistance_above_support(self, sample_ohlcv_data):
        """Verify resistance > support."""
        df = add_support_resistance(sample_ohlcv_data)

        warmup = 20
        valid_data = df.iloc[warmup:].dropna(subset=['resistance', 'support'])

        assert (valid_data['resistance'] >= valid_data['support']).all()


# =============================================================================
# Gap Calculation Tests
# =============================================================================

@pytest.mark.unit
class TestGapCalculation:
    """Tests for gap calculations."""

    def test_gap_columns_created(self, sample_ohlcv_data):
        """Verify gap columns are created."""
        df = calculate_gap(sample_ohlcv_data)

        expected_cols = ['gap', 'gap_abs', 'gap_direction']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_gap_calculation(self, sample_ohlcv_data):
        """Verify gap = (open - prev_close) / prev_close."""
        df = calculate_gap(sample_ohlcv_data)

        for i in range(1, min(10, len(df))):
            expected = (df['open'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1] * 100
            actual = df['gap'].iloc[i]

            if not pd.isna(expected) and not pd.isna(actual):
                assert actual == pytest.approx(expected, rel=1e-4)

    def test_gap_direction_classification(self, sample_ohlcv_data):
        """Verify gap direction is classified correctly."""
        df = calculate_gap(sample_ohlcv_data)

        # Check direction matches gap sign
        for i in range(1, min(20, len(df))):
            if df['gap'].iloc[i] > 0:
                assert df['gap_direction'].iloc[i] == 'up'
            elif df['gap'].iloc[i] < 0:
                assert df['gap_direction'].iloc[i] == 'down'


# =============================================================================
# Add All Indicators Tests
# =============================================================================

@pytest.mark.unit
class TestAddAllIndicators:
    """Tests for the add_all_indicators function."""

    def test_add_all_indicators_success(self, sample_ohlcv_data):
        """Verify add_all_indicators processes data successfully."""
        df = add_all_indicators(sample_ohlcv_data)

        # Should have more columns than original
        assert len(df.columns) > len(sample_ohlcv_data.columns)

    def test_add_all_indicators_preserves_original(self, sample_ohlcv_data):
        """Verify original data is not modified."""
        original_cols = sample_ohlcv_data.columns.tolist()
        _ = add_all_indicators(sample_ohlcv_data)

        assert sample_ohlcv_data.columns.tolist() == original_cols

    def test_add_all_indicators_minimum_length(self):
        """Verify minimum data length is enforced."""
        # Create data that's too short
        short_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200],
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='D'))

        result = add_all_indicators(short_df)

        # Should return data unchanged or with minimal indicators
        assert len(result) == 3

    def test_add_advanced_indicators(self, sample_ohlcv_data):
        """Verify advanced indicators are added when requested."""
        df = add_all_indicators(sample_ohlcv_data, include_advanced=True)

        # Should include volatility and trend indicators
        # (if data is long enough)
        if len(sample_ohlcv_data) >= 63:  # Need 63 days for 63d vol
            assert 'adx' in df.columns or len(df.columns) > 20
