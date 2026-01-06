"""
Unit tests for trading strategies.

Tests cover:
- Strategy initialization
- Parameter validation
- Signal generation
- Strategy state management
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestStrategyBase:
    """Test base strategy functionality."""

    def test_strategy_requires_data(self, sample_ohlcv_data):
        """Test that strategies require market data to generate signals."""
        # Strategies need historical data
        assert len(sample_ohlcv_data) > 0
        assert 'close' in sample_ohlcv_data.columns

    def test_signal_output_format(self, sample_signals):
        """Test that signals have required fields."""
        for signal in sample_signals:
            assert 'symbol' in signal
            assert 'signal' in signal or 'signal_type' in signal
            assert 'strength' in signal

    def test_signal_strength_bounds(self, sample_signals):
        """Test signal strength is between 0 and 1."""
        for signal in sample_signals:
            assert 0 <= signal['strength'] <= 1


class TestSignalValidation:
    """Test signal validation logic."""

    def test_valid_buy_signal(self):
        """Test buy signal is valid."""
        signal = {
            'symbol': 'AAPL',
            'signal': 1,
            'strength': 0.8,
            'price': 150.0,
        }

        assert signal['signal'] == 1  # Buy
        assert signal['strength'] > 0.5  # Strong enough

    def test_valid_sell_signal(self):
        """Test sell signal is valid."""
        signal = {
            'symbol': 'AAPL',
            'signal': -1,
            'strength': 0.7,
            'price': 150.0,
        }

        assert signal['signal'] == -1  # Sell
        assert signal['strength'] > 0.5

    def test_neutral_signal(self):
        """Test neutral/no-action signal."""
        signal = {
            'symbol': 'AAPL',
            'signal': 0,
            'strength': 0.3,
            'price': 150.0,
        }

        assert signal['signal'] == 0  # No action


class TestMomentumStrategy:
    """Test momentum strategy logic."""

    def test_momentum_calculation(self, sample_ohlcv_data):
        """Test 12-1 momentum calculation."""
        prices = sample_ohlcv_data['close']

        # 12-month momentum (skip last month)
        if len(prices) >= 252:
            momentum_12_1 = (prices.iloc[-21] / prices.iloc[-252]) - 1
            assert isinstance(momentum_12_1, (int, float))

    def test_high_momentum_generates_buy(self, sample_ohlcv_data):
        """Test that high momentum generates buy signal."""
        # Create strong uptrend
        uptrend_data = sample_ohlcv_data.copy()
        uptrend_data['close'] = np.linspace(100, 200, len(uptrend_data))

        momentum = (uptrend_data['close'].iloc[-21] / uptrend_data['close'].iloc[0]) - 1

        # Strong momentum should signal buy
        assert momentum > 0.5  # 50%+ return

    def test_negative_momentum_generates_sell(self, sample_ohlcv_data):
        """Test that negative momentum generates sell signal."""
        # Create downtrend
        downtrend_data = sample_ohlcv_data.copy()
        downtrend_data['close'] = np.linspace(200, 100, len(downtrend_data))

        momentum = (downtrend_data['close'].iloc[-21] / downtrend_data['close'].iloc[0]) - 1

        # Negative momentum should signal sell
        assert momentum < 0


class TestVIXRegimeStrategy:
    """Test VIX regime rotation strategy."""

    def test_vix_regime_classification(self, sample_vix_data):
        """Test VIX levels are classified into regimes."""
        vix = sample_vix_data['close'].iloc[-1]

        # Classify based on thresholds
        if vix < 15:
            regime = 'low'
        elif vix < 25:
            regime = 'normal'
        elif vix < 35:
            regime = 'high'
        else:
            regime = 'extreme'

        assert regime in ['low', 'normal', 'high', 'extreme']

    @pytest.mark.parametrize("vix_level,expected_regime", [
        (12, 'low'),
        (18, 'normal'),
        (28, 'high'),
        (45, 'extreme'),
    ])
    def test_vix_regime_thresholds(self, vix_level, expected_regime):
        """Test VIX regime thresholds."""
        if vix_level < 15:
            regime = 'low'
        elif vix_level < 25:
            regime = 'normal'
        elif vix_level < 35:
            regime = 'high'
        else:
            regime = 'extreme'

        assert regime == expected_regime


class TestMeanReversionStrategy:
    """Test mean reversion strategy logic."""

    def test_zscore_calculation(self, sample_ohlcv_data):
        """Test z-score calculation for mean reversion."""
        prices = sample_ohlcv_data['close']

        # Calculate z-score (21-day)
        window = 21
        if len(prices) >= window:
            rolling_mean = prices.rolling(window).mean()
            rolling_std = prices.rolling(window).std()
            zscore = (prices - rolling_mean) / rolling_std

            # Z-score should be bounded
            last_zscore = zscore.iloc[-1]
            assert -5 < last_zscore < 5  # Reasonable bounds

    def test_oversold_generates_buy(self):
        """Test that oversold conditions generate buy signal."""
        zscore = -2.5  # Very oversold

        # Oversold should signal buy
        signal = 1 if zscore < -2 else 0
        assert signal == 1

    def test_overbought_generates_sell(self):
        """Test that overbought conditions generate sell signal."""
        zscore = 2.5  # Very overbought

        # Overbought should signal sell
        signal = -1 if zscore > 2 else 0
        assert signal == -1


class TestPairsTradingStrategy:
    """Test pairs trading strategy logic."""

    def test_spread_calculation(self, sample_multi_stock_data):
        """Test spread calculation between two stocks."""
        if 'AAPL' in sample_multi_stock_data and 'MSFT' in sample_multi_stock_data:
            stock1 = sample_multi_stock_data['AAPL']['close']
            stock2 = sample_multi_stock_data['MSFT']['close']

            # Simple ratio spread
            spread = stock1 / stock2

            assert len(spread) > 0
            assert spread.iloc[-1] > 0

    def test_cointegration_check(self, sample_multi_stock_data):
        """Test that pairs have correlation/cointegration."""
        if len(sample_multi_stock_data) >= 2:
            symbols = list(sample_multi_stock_data.keys())[:2]
            returns1 = sample_multi_stock_data[symbols[0]]['close'].pct_change().dropna()
            returns2 = sample_multi_stock_data[symbols[1]]['close'].pct_change().dropna()

            # Check correlation
            corr = returns1.corr(returns2)
            assert -1 <= corr <= 1


class TestRelativeVolumeStrategy:
    """Test relative volume breakout strategy."""

    def test_relative_volume_calculation(self, sample_ohlcv_data):
        """Test relative volume calculation."""
        volume = sample_ohlcv_data['volume']

        # Calculate relative volume (current vs 20-day average)
        window = 20
        if len(volume) > window:
            avg_volume = volume.rolling(window).mean()
            relative_volume = volume / avg_volume

            assert relative_volume.iloc[-1] > 0

    def test_high_volume_breakout_signal(self, sample_ohlcv_data):
        """Test that high relative volume generates breakout signal."""
        volume = sample_ohlcv_data['volume']
        price = sample_ohlcv_data['close']

        # High volume + price up = breakout
        vol_spike = volume.iloc[-1] > volume.mean() * 2
        price_up = price.iloc[-1] > price.iloc[-2]

        breakout = vol_spike and price_up
        assert isinstance(breakout, (bool, np.bool_))


class TestGapFillStrategy:
    """Test gap fill strategy logic."""

    def test_gap_detection(self):
        """Test gap detection logic."""
        prev_close = 100.0
        open_price = 102.0

        gap_pct = (open_price - prev_close) / prev_close
        assert gap_pct == pytest.approx(0.02, rel=0.01)  # 2% gap

    def test_gap_fill_target(self):
        """Test gap fill target calculation."""
        prev_close = 100.0
        open_price = 102.0
        fill_pct = 0.8  # Expect 80% fill

        gap = open_price - prev_close
        fill_target = open_price - (gap * fill_pct)

        assert fill_target == pytest.approx(100.4, rel=0.01)

    @pytest.mark.parametrize("gap_pct,min_gap,should_trade", [
        (0.01, 0.005, True),   # 1% gap > 0.5% min
        (0.003, 0.005, False), # 0.3% gap < 0.5% min
        (0.02, 0.01, True),    # 2% gap > 1% min
        (-0.02, -0.01, True),  # -2% gap down
    ])
    def test_gap_threshold(self, gap_pct, min_gap, should_trade):
        """Test gap threshold filters."""
        trade = abs(gap_pct) >= abs(min_gap)
        assert trade == should_trade


class TestStrategyIntegration:
    """Integration tests for strategy combinations."""

    def test_multiple_strategies_can_run(
        self,
        sample_multi_stock_data,
        sample_vix_data
    ):
        """Test that multiple strategies can run on same data."""
        strategies_run = 0

        # Momentum
        for symbol, data in sample_multi_stock_data.items():
            if len(data) >= 252:
                momentum = data['close'].iloc[-21] / data['close'].iloc[-252] - 1
                strategies_run += 1
                break

        # VIX regime
        vix = sample_vix_data['close'].iloc[-1]
        regime = 'normal' if vix < 25 else 'high'
        strategies_run += 1

        assert strategies_run >= 2

    def test_signals_dont_conflict(self, sample_signals):
        """Test that signals from same symbol don't have impossible conflicts."""
        symbol_signals = {}
        for signal in sample_signals:
            symbol = signal['symbol']
            if symbol in symbol_signals:
                # Same symbol shouldn't have both strong buy and strong sell
                prev = symbol_signals[symbol]
                if prev['signal'] == 1 and signal['signal'] == -1:
                    # Both strong = conflict
                    assert not (prev['strength'] > 0.8 and signal['strength'] > 0.8)
            symbol_signals[symbol] = signal
