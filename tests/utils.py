"""
Test utilities and helper functions for the trading system test suite.

Provides assertion helpers, data generators, and comparison utilities
specific to trading system testing needs.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# Price Comparison Helpers
# =============================================================================

def assert_price_equal(
    actual: float,
    expected: float,
    tolerance_pct: float = 0.01,
    msg: str = ""
):
    """
    Assert two prices are equal within a percentage tolerance.

    Args:
        actual: Actual price
        expected: Expected price
        tolerance_pct: Tolerance as decimal (0.01 = 1%)
        msg: Optional message on failure
    """
    if expected == 0:
        assert actual == 0, f"Expected 0, got {actual}. {msg}"
        return

    diff_pct = abs(actual - expected) / expected
    assert diff_pct <= tolerance_pct, (
        f"Price mismatch: {actual} vs {expected} "
        f"(diff: {diff_pct*100:.2f}% > {tolerance_pct*100}%). {msg}"
    )


def assert_prices_close(
    prices1: Union[List[float], np.ndarray],
    prices2: Union[List[float], np.ndarray],
    tolerance_pct: float = 0.01,
):
    """Assert two price series are close within tolerance."""
    arr1 = np.array(prices1)
    arr2 = np.array(prices2)

    assert len(arr1) == len(arr2), f"Length mismatch: {len(arr1)} vs {len(arr2)}"

    for i, (p1, p2) in enumerate(zip(arr1, arr2)):
        assert_price_equal(p1, p2, tolerance_pct, f"at index {i}")


# =============================================================================
# Return/Performance Helpers
# =============================================================================

def assert_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    min_sharpe: float,
    periods_per_year: int = 252,
):
    """
    Assert that returns achieve a minimum Sharpe ratio.

    Args:
        returns: Return series
        min_sharpe: Minimum required Sharpe ratio
        periods_per_year: Number of periods per year (252 for daily)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

    assert sharpe >= min_sharpe, (
        f"Sharpe ratio {sharpe:.2f} < required {min_sharpe}"
    )


def assert_max_drawdown(
    returns: Union[pd.Series, np.ndarray],
    max_allowed: float,
):
    """
    Assert that maximum drawdown doesn't exceed threshold.

    Args:
        returns: Return series
        max_allowed: Maximum allowed drawdown (e.g., 0.20 for 20%)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())

    assert max_dd <= max_allowed, (
        f"Max drawdown {max_dd*100:.1f}% exceeds allowed {max_allowed*100:.1f}%"
    )


# =============================================================================
# Position/Order Helpers
# =============================================================================

def assert_position_size_valid(
    position_value: float,
    portfolio_value: float,
    max_position_pct: float = 0.10,
):
    """Assert position size is within limits."""
    position_pct = position_value / portfolio_value
    assert position_pct <= max_position_pct, (
        f"Position {position_pct*100:.1f}% exceeds max {max_position_pct*100:.1f}%"
    )


def assert_order_valid(order: Dict[str, Any], expected_fields: List[str] = None):
    """Assert order has required fields and valid values."""
    if expected_fields is None:
        expected_fields = ['symbol', 'qty', 'side']

    for field in expected_fields:
        assert field in order, f"Order missing required field: {field}"

    if 'qty' in order:
        assert order['qty'] > 0, f"Invalid quantity: {order['qty']}"

    if 'side' in order:
        assert order['side'] in ['buy', 'sell', 'BUY', 'SELL'], (
            f"Invalid side: {order['side']}"
        )


# =============================================================================
# OHLCV Data Helpers
# =============================================================================

def assert_ohlcv_valid(df: pd.DataFrame):
    """Assert OHLCV data is valid."""
    required_cols = ['open', 'high', 'low', 'close', 'volume']

    # Check columns exist (case-insensitive)
    df_cols_lower = [c.lower() for c in df.columns]
    for col in required_cols:
        assert col in df_cols_lower, f"Missing required column: {col}"

    # Check OHLC relationships
    for col in ['open', 'high', 'low', 'close']:
        actual_col = [c for c in df.columns if c.lower() == col][0]
        assert (df[actual_col] > 0).all(), f"Non-positive values in {col}"

    # High >= max(open, close)
    high_col = [c for c in df.columns if c.lower() == 'high'][0]
    open_col = [c for c in df.columns if c.lower() == 'open'][0]
    close_col = [c for c in df.columns if c.lower() == 'close'][0]
    low_col = [c for c in df.columns if c.lower() == 'low'][0]

    assert (df[high_col] >= df[[open_col, close_col]].max(axis=1)).all(), (
        "High must be >= max(open, close)"
    )

    # Low <= min(open, close)
    assert (df[low_col] <= df[[open_col, close_col]].min(axis=1)).all(), (
        "Low must be <= min(open, close)"
    )


def assert_no_gaps(
    df: pd.DataFrame,
    max_gap_days: int = 5,
    ignore_weekends: bool = True,
):
    """Assert no unexpected gaps in time series data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return  # Can't check gaps without datetime index

    diffs = df.index.to_series().diff()

    if ignore_weekends:
        # Allow gaps up to max_gap_days plus weekends
        max_allowed = timedelta(days=max_gap_days + 2)
    else:
        max_allowed = timedelta(days=max_gap_days)

    large_gaps = diffs[diffs > max_allowed]
    assert len(large_gaps) == 0, (
        f"Found {len(large_gaps)} gaps > {max_gap_days} days"
    )


# =============================================================================
# Signal Validation Helpers
# =============================================================================

def assert_signal_valid(signal: Dict[str, Any]):
    """Assert trading signal is valid."""
    assert 'symbol' in signal, "Signal missing symbol"
    assert 'signal' in signal or 'direction' in signal, (
        "Signal missing signal/direction"
    )

    signal_value = signal.get('signal', signal.get('direction', 0))
    assert signal_value in [-1, 0, 1], f"Invalid signal value: {signal_value}"

    if 'strength' in signal:
        assert 0 <= signal['strength'] <= 1, (
            f"Signal strength must be 0-1, got {signal['strength']}"
        )


def assert_signals_balanced(signals: List[Dict], max_imbalance: float = 0.8):
    """Assert signals aren't too heavily skewed in one direction."""
    if not signals:
        return

    long_count = sum(1 for s in signals if s.get('signal', 0) > 0)
    short_count = sum(1 for s in signals if s.get('signal', 0) < 0)
    total = long_count + short_count

    if total == 0:
        return

    imbalance = abs(long_count - short_count) / total
    assert imbalance <= max_imbalance, (
        f"Signal imbalance {imbalance:.1%} exceeds {max_imbalance:.1%}"
    )


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_random_prices(
    n_days: int = 252,
    start_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005,
    seed: int = None,
) -> pd.Series:
    """Generate random price series with geometric Brownian motion."""
    if seed is not None:
        np.random.seed(seed)

    returns = np.random.normal(drift, volatility, n_days)
    prices = start_price * np.cumprod(1 + returns)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    return pd.Series(prices, index=dates, name='price')


def generate_ohlcv(
    n_days: int = 252,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = None,
) -> pd.DataFrame:
    """Generate realistic OHLCV data."""
    if seed is not None:
        np.random.seed(seed)

    close = generate_random_prices(n_days, start_price, volatility, seed=seed)

    # Generate OHLC from close
    high_mult = 1 + np.random.uniform(0, 0.02, n_days)
    low_mult = 1 - np.random.uniform(0, 0.02, n_days)
    open_mult = 1 + np.random.uniform(-0.01, 0.01, n_days)

    df = pd.DataFrame({
        'open': close.values * open_mult,
        'high': close.values * high_mult,
        'low': close.values * low_mult,
        'close': close.values,
        'volume': np.random.randint(1000000, 10000000, n_days),
    }, index=close.index)

    # Ensure OHLC constraints
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def generate_correlated_prices(
    n_symbols: int = 5,
    n_days: int = 252,
    correlation: float = 0.5,
    seed: int = None,
) -> Dict[str, pd.DataFrame]:
    """Generate correlated price data for multiple symbols."""
    if seed is not None:
        np.random.seed(seed)

    # Generate correlated returns
    mean = np.zeros(n_symbols)
    cov = np.full((n_symbols, n_symbols), correlation * 0.02**2)
    np.fill_diagonal(cov, 0.02**2)

    returns = np.random.multivariate_normal(mean, cov, n_days)

    # Convert to prices
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    result = {}
    for i, symbol in enumerate(symbols):
        prices = 100 * np.cumprod(1 + returns[:, i])
        df = pd.DataFrame({
            'close': prices,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
            'volume': np.random.randint(1000000, 10000000, n_days),
        }, index=dates)

        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        result[symbol] = df

    return result
