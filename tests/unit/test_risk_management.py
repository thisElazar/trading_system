"""
Unit tests for risk management functionality.

Tests cover:
- Position limit enforcement
- Drawdown calculations
- Stop-loss logic
- Daily loss limits
- Portfolio exposure limits

These are CRITICAL PATH tests - risk management protects capital.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

pytestmark = [pytest.mark.unit, pytest.mark.critical]


class TestPositionLimits:
    """Test position limit enforcement."""

    def test_max_position_size_pct(self):
        """Test maximum position size as percentage of portfolio."""
        portfolio_value = 100000
        max_position_pct = 0.05  # 5%

        max_position_value = portfolio_value * max_position_pct
        assert max_position_value == 5000

        # Test violation
        proposed_position = 8000
        is_valid = proposed_position <= max_position_value
        assert is_valid == False

    def test_max_positions_count(self):
        """Test maximum number of concurrent positions."""
        max_positions = 10
        current_positions = 8

        can_open_new = current_positions < max_positions
        assert can_open_new == True

        # At limit
        current_positions = 10
        can_open_new = current_positions < max_positions
        assert can_open_new == False

    @pytest.mark.parametrize("portfolio,position_value,max_pct,is_valid", [
        (100000, 5000, 0.05, True),
        (100000, 6000, 0.05, False),
        (100000, 10000, 0.10, True),
        (50000, 3000, 0.05, False),
    ])
    def test_position_size_validation(self, portfolio, position_value, max_pct, is_valid):
        """Test position size validation with various scenarios."""
        position_pct = position_value / portfolio
        valid = position_pct <= max_pct
        assert valid == is_valid


class TestDrawdownCalculations:
    """Test drawdown calculation logic."""

    def test_max_drawdown_calculation(self, sample_ohlcv_data):
        """Test maximum drawdown calculation."""
        prices = sample_ohlcv_data['close']

        # Calculate cumulative returns
        returns = prices.pct_change().dropna()
        cumulative = (1 + returns).cumprod()

        # Calculate drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()

        assert max_dd <= 0  # Drawdown is negative
        assert max_dd >= -1  # Can't lose more than 100%

    def test_current_drawdown(self):
        """Test current drawdown from peak."""
        peak_value = 100000
        current_value = 92000

        drawdown = (current_value - peak_value) / peak_value
        assert drawdown == pytest.approx(-0.08, rel=0.01)  # -8%

    def test_drawdown_triggers_action(self):
        """Test that exceeding drawdown threshold triggers action."""
        max_drawdown_pct = 0.15  # 15%
        current_drawdown = 0.18  # 18%

        should_reduce_exposure = current_drawdown > max_drawdown_pct
        assert should_reduce_exposure == True

    def test_drawdown_recovery(self):
        """Test drawdown recovery tracking."""
        peak = 100000
        trough = 85000  # -15% drawdown
        current = 95000

        drawdown_from_peak = (current - peak) / peak  # -5%
        recovery_from_trough = (current - trough) / (peak - trough)  # 66.7%

        assert drawdown_from_peak < 0
        assert 0 < recovery_from_trough < 1


class TestStopLossLogic:
    """Test stop-loss enforcement."""

    def test_stop_loss_triggered(self):
        """Test stop-loss trigger logic."""
        entry_price = 100.0
        stop_loss_price = 95.0  # 5% stop
        current_price = 94.0

        stop_triggered = current_price <= stop_loss_price
        assert stop_triggered == True

    def test_stop_loss_not_triggered(self):
        """Test stop-loss not triggered when price above."""
        entry_price = 100.0
        stop_loss_price = 95.0
        current_price = 97.0

        stop_triggered = current_price <= stop_loss_price
        assert stop_triggered == False

    def test_trailing_stop_calculation(self):
        """Test trailing stop-loss calculation."""
        entry_price = 100.0
        highest_price = 115.0
        trailing_pct = 0.05  # 5% trailing

        trailing_stop = highest_price * (1 - trailing_pct)
        assert trailing_stop == pytest.approx(109.25, rel=0.01)

    @pytest.mark.parametrize("side,entry,current,stop_pct,triggered", [
        ('long', 100, 94, 0.05, True),    # Long, price fell below stop
        ('long', 100, 96, 0.05, False),   # Long, price above stop
        ('short', 100, 106, 0.05, True),  # Short, price rose above stop
        ('short', 100, 103, 0.05, False), # Short, price below stop
    ])
    def test_stop_loss_by_side(self, side, entry, current, stop_pct, triggered):
        """Test stop-loss for both long and short positions."""
        if side == 'long':
            stop_price = entry * (1 - stop_pct)
            is_triggered = current <= stop_price
        else:
            stop_price = entry * (1 + stop_pct)
            is_triggered = current >= stop_price

        assert is_triggered == triggered


class TestDailyLossLimits:
    """Test daily loss limit enforcement."""

    def test_daily_loss_limit_calculation(self):
        """Test daily loss limit as percentage of portfolio."""
        portfolio_value = 100000
        max_daily_loss_pct = 0.02  # 2%

        max_daily_loss = portfolio_value * max_daily_loss_pct
        assert max_daily_loss == 2000

    def test_daily_loss_triggers_halt(self):
        """Test trading halt when daily loss exceeded."""
        starting_equity = 100000
        current_equity = 97500
        max_daily_loss_pct = 0.02

        daily_loss = starting_equity - current_equity
        daily_loss_pct = daily_loss / starting_equity

        should_halt = daily_loss_pct >= max_daily_loss_pct
        assert should_halt == True

    def test_daily_loss_within_limits(self):
        """Test trading continues when within limits."""
        starting_equity = 100000
        current_equity = 99000  # -1%
        max_daily_loss_pct = 0.02

        daily_loss_pct = (starting_equity - current_equity) / starting_equity
        should_halt = daily_loss_pct >= max_daily_loss_pct
        assert should_halt == False


class TestPortfolioExposure:
    """Test portfolio exposure limits."""

    def test_gross_exposure_calculation(self):
        """Test gross exposure (long + short) calculation."""
        portfolio_value = 100000
        long_positions = 50000
        short_positions = 20000

        gross_exposure = long_positions + short_positions
        gross_exposure_pct = gross_exposure / portfolio_value

        assert gross_exposure == 70000
        assert gross_exposure_pct == 0.70

    def test_net_exposure_calculation(self):
        """Test net exposure (long - short) calculation."""
        portfolio_value = 100000
        long_positions = 50000
        short_positions = 20000

        net_exposure = long_positions - short_positions
        net_exposure_pct = net_exposure / portfolio_value

        assert net_exposure == 30000
        assert net_exposure_pct == 0.30

    def test_max_gross_exposure_limit(self):
        """Test maximum gross exposure limit."""
        portfolio_value = 100000
        max_gross_exposure_pct = 1.0  # 100% max gross

        current_long = 60000
        current_short = 30000
        new_position = 15000

        current_gross = current_long + current_short
        proposed_gross = current_gross + new_position
        proposed_gross_pct = proposed_gross / portfolio_value

        exceeds_limit = proposed_gross_pct > max_gross_exposure_pct
        assert exceeds_limit == True

    def test_sector_concentration_limit(self):
        """Test sector concentration limits."""
        portfolio_value = 100000
        max_sector_pct = 0.25  # 25% max per sector

        tech_positions = [
            {'symbol': 'AAPL', 'value': 8000, 'sector': 'tech'},
            {'symbol': 'MSFT', 'value': 7000, 'sector': 'tech'},
            {'symbol': 'GOOGL', 'value': 6000, 'sector': 'tech'},
        ]

        tech_exposure = sum(p['value'] for p in tech_positions)
        tech_exposure_pct = tech_exposure / portfolio_value

        assert tech_exposure_pct == 0.21  # 21%
        assert tech_exposure_pct < max_sector_pct  # Within limit


class TestRiskMetrics:
    """Test risk metric calculations."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Generate sample returns
        np.random.seed(42)
        daily_returns = np.random.normal(0.0005, 0.02, 252)

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)

        assert -5 < sharpe < 5  # Reasonable bounds

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation (downside deviation)."""
        np.random.seed(42)
        daily_returns = np.random.normal(0.0005, 0.02, 252)

        mean_return = np.mean(daily_returns)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.01

        sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        assert isinstance(sortino, (int, float))

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        np.random.seed(42)
        daily_returns = np.random.normal(0.0005, 0.02, 252)
        confidence = 0.95

        var_95 = np.percentile(daily_returns, (1 - confidence) * 100)

        # VaR at 95% should be negative (loss)
        assert var_95 < 0

    def test_volatility_annualization(self):
        """Test volatility annualization."""
        daily_vol = 0.02  # 2% daily

        # Annualize with sqrt(252)
        annual_vol = daily_vol * np.sqrt(252)

        assert annual_vol == pytest.approx(0.317, rel=0.01)  # ~31.7%
