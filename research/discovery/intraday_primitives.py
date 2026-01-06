"""
Intraday Genetic Programming Primitives
========================================
Specialized primitives for evolving intraday trading strategies.

Key differences from daily primitives:
- VWAP (Volume Weighted Average Price)
- Opening Range (first N minutes high/low)
- Intraday momentum indicators
- Time-based indicators (minutes since open)
- Relative volume vs average

These primitives work with 1-minute bar data and require
the DataFrame to have a proper datetime index.
"""

import operator
import math
import random
import logging
import threading
from typing import Optional
from datetime import time, datetime

import numpy as np
import pandas as pd

from deap import gp

from .gp_core import (
    BoolType, FloatType, protected_div, protected_log, protected_sqrt,
    if_then_else, _safe_get_column, get_eval_data, set_eval_data, clear_eval_data
)

logger = logging.getLogger(__name__)


# ==========================================================================
# Intraday-specific indicator functions
# ==========================================================================

def ind_vwap() -> float:
    """
    Volume Weighted Average Price for the current session.

    VWAP = cumsum(price * volume) / cumsum(volume)
    """
    df = get_eval_data()
    if df is None or len(df) == 0:
        return _safe_get_column(df, 'close', 0.0)

    # Use typical price: (high + low + close) / 3
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].cumsum()
    cum_pv = (typical_price * df['volume']).cumsum()

    if cum_vol.iloc[-1] == 0:
        return df['close'].iloc[-1]

    vwap = cum_pv.iloc[-1] / cum_vol.iloc[-1]
    return float(vwap) if not np.isnan(vwap) else df['close'].iloc[-1]


def ind_vwap_distance() -> float:
    """
    Distance from VWAP as percentage of price.

    Positive = above VWAP, negative = below VWAP.
    """
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    vwap = ind_vwap()
    close = df['close'].iloc[-1]

    if vwap == 0:
        return 0.0

    distance = (close - vwap) / vwap * 100
    return float(distance) if not np.isnan(distance) else 0.0


def _make_opening_range_high(minutes: int = 30):
    """Factory for opening range high indicator."""
    def _or_high() -> float:
        df = get_eval_data()
        if df is None or len(df) == 0:
            return _safe_get_column(df, 'high', 0.0)

        # Get first N bars (opening range)
        or_bars = df.iloc[:min(minutes, len(df))]
        return float(or_bars['high'].max())

    _or_high.__name__ = f'or_high_{minutes}'
    return _or_high


def _make_opening_range_low(minutes: int = 30):
    """Factory for opening range low indicator."""
    def _or_low() -> float:
        df = get_eval_data()
        if df is None or len(df) == 0:
            return _safe_get_column(df, 'low', 0.0)

        # Get first N bars (opening range)
        or_bars = df.iloc[:min(minutes, len(df))]
        return float(or_bars['low'].min())

    _or_low.__name__ = f'or_low_{minutes}'
    return _or_low


def ind_opening_range_size() -> float:
    """
    Size of the opening range (30-min high - low) as percentage.
    """
    df = get_eval_data()
    if df is None or len(df) < 30:
        return 0.0

    or_bars = df.iloc[:30]
    or_high = or_bars['high'].max()
    or_low = or_bars['low'].min()

    if or_low == 0:
        return 0.0

    return float((or_high - or_low) / or_low * 100)


def ind_relative_volume() -> float:
    """
    Current volume relative to average volume at this time of day.

    Returns ratio (e.g., 1.5 = 50% above average).
    For simplicity, uses rolling 20-bar average.
    """
    df = get_eval_data()
    if df is None or len(df) < 20:
        return 1.0

    current_vol = df['volume'].iloc[-1]
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]

    if avg_vol == 0 or np.isnan(avg_vol):
        return 1.0

    return float(current_vol / avg_vol)


def ind_minutes_since_open() -> float:
    """
    Number of minutes elapsed since market open (bar count).
    """
    df = get_eval_data()
    if df is None:
        return 0.0
    return float(len(df))


def ind_minutes_until_close() -> float:
    """
    Approximate minutes until market close.
    Assumes 390 minute trading day.
    """
    df = get_eval_data()
    if df is None:
        return 390.0
    return float(max(0, 390 - len(df)))


def ind_day_high() -> float:
    """Session high so far."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0
    return float(df['high'].max())


def ind_day_low() -> float:
    """Session low so far."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0
    return float(df['low'].min())


def ind_day_range() -> float:
    """Current day's range (high - low) as percentage."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    day_high = df['high'].max()
    day_low = df['low'].min()

    if day_low == 0:
        return 0.0

    return float((day_high - day_low) / day_low * 100)


def ind_distance_from_day_high() -> float:
    """Distance from session high as percentage (always negative or zero)."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    day_high = df['high'].max()
    close = df['close'].iloc[-1]

    if day_high == 0:
        return 0.0

    return float((close - day_high) / day_high * 100)


def ind_distance_from_day_low() -> float:
    """Distance from session low as percentage (always positive or zero)."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    day_low = df['low'].min()
    close = df['close'].iloc[-1]

    if day_low == 0:
        return 0.0

    return float((close - day_low) / day_low * 100)


def _make_intraday_momentum(periods: int = 5):
    """Factory for intraday momentum (N-bar return)."""
    def _momentum() -> float:
        df = get_eval_data()
        if df is None or len(df) <= periods:
            return 0.0

        prev_close = df['close'].iloc[-periods - 1]
        close = df['close'].iloc[-1]

        if prev_close == 0:
            return 0.0

        return float((close - prev_close) / prev_close * 100)

    _momentum.__name__ = f'momentum_{periods}m'
    return _momentum


def _make_intraday_sma(period: int):
    """Factory for intraday SMA."""
    def _sma() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return _safe_get_column(df, 'close', 0.0)
        val = df['close'].rolling(period).mean().iloc[-1]
        return float(val) if not np.isnan(val) else df['close'].iloc[-1]
    _sma.__name__ = f'sma_{period}m'
    return _sma


def _make_intraday_ema(period: int):
    """Factory for intraday EMA."""
    def _ema() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return _safe_get_column(df, 'close', 0.0)
        val = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
        return float(val) if not np.isnan(val) else df['close'].iloc[-1]
    _ema.__name__ = f'ema_{period}m'
    return _ema


def ind_bar_range() -> float:
    """Current bar's range (high - low) as percentage."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]

    if low == 0:
        return 0.0

    return float((high - low) / low * 100)


def ind_bar_body() -> float:
    """Current bar's body (close - open) as percentage. Positive = bullish."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    open_price = df['open'].iloc[-1]
    close = df['close'].iloc[-1]

    if open_price == 0:
        return 0.0

    return float((close - open_price) / open_price * 100)


def ind_upper_wick_pct() -> float:
    """Upper wick as percentage of bar range."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    open_price = df['open'].iloc[-1]
    close = df['close'].iloc[-1]

    bar_range = high - low
    if bar_range == 0:
        return 0.0

    body_top = max(open_price, close)
    upper_wick = high - body_top

    return float(upper_wick / bar_range * 100)


def ind_lower_wick_pct() -> float:
    """Lower wick as percentage of bar range."""
    df = get_eval_data()
    if df is None or len(df) == 0:
        return 0.0

    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    open_price = df['open'].iloc[-1]
    close = df['close'].iloc[-1]

    bar_range = high - low
    if bar_range == 0:
        return 0.0

    body_bottom = min(open_price, close)
    lower_wick = body_bottom - low

    return float(lower_wick / bar_range * 100)


def _make_intraday_rsi(period: int = 14):
    """Factory for intraday RSI."""
    def _rsi() -> float:
        df = get_eval_data()
        if df is None or len(df) < period + 1:
            return 50.0
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi) if not np.isnan(rsi) else 50.0
    _rsi.__name__ = f'rsi_{period}m'
    return _rsi


def _make_intraday_atr(period: int = 14):
    """Factory for intraday ATR."""
    def _atr() -> float:
        df = get_eval_data()
        if df is None or len(df) < period + 1:
            return 0.0
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0
    _atr.__name__ = f'atr_{period}m'
    return _atr


# ==========================================================================
# Primitive Set Creation
# ==========================================================================

def create_intraday_primitive_set() -> gp.PrimitiveSetTyped:
    """
    Create the primitive set for intraday GP (float-valued trees).

    Includes VWAP, opening range, and time-based indicators
    specifically designed for intraday trading.
    """
    pset = gp.PrimitiveSetTyped("INTRADAY", [], FloatType)

    # ==========================================================================
    # ARITHMETIC FUNCTIONS
    # ==========================================================================
    pset.addPrimitive(operator.add, [FloatType, FloatType], FloatType, name='add')
    pset.addPrimitive(operator.sub, [FloatType, FloatType], FloatType, name='sub')
    pset.addPrimitive(operator.mul, [FloatType, FloatType], FloatType, name='mul')
    pset.addPrimitive(protected_div, [FloatType, FloatType], FloatType, name='div')

    # ==========================================================================
    # UNARY FUNCTIONS
    # ==========================================================================
    pset.addPrimitive(operator.neg, [FloatType], FloatType, name='neg')
    pset.addPrimitive(abs, [FloatType], FloatType, name='abs')
    pset.addPrimitive(protected_log, [FloatType], FloatType, name='log')
    pset.addPrimitive(protected_sqrt, [FloatType], FloatType, name='sqrt')

    # ==========================================================================
    # COMPARISON FUNCTIONS
    # ==========================================================================
    pset.addPrimitive(operator.gt, [FloatType, FloatType], BoolType, name='gt')
    pset.addPrimitive(operator.lt, [FloatType, FloatType], BoolType, name='lt')
    pset.addPrimitive(operator.ge, [FloatType, FloatType], BoolType, name='ge')
    pset.addPrimitive(operator.le, [FloatType, FloatType], BoolType, name='le')

    # ==========================================================================
    # LOGICAL FUNCTIONS
    # ==========================================================================
    pset.addPrimitive(operator.and_, [BoolType, BoolType], BoolType, name='and_')
    pset.addPrimitive(operator.or_, [BoolType, BoolType], BoolType, name='or_')
    pset.addPrimitive(operator.not_, [BoolType], BoolType, name='not_')

    # ==========================================================================
    # CONDITIONAL
    # ==========================================================================
    pset.addPrimitive(if_then_else, [BoolType, FloatType, FloatType], FloatType, name='if_then_else')

    # ==========================================================================
    # PRICE DATA TERMINALS
    # ==========================================================================
    from .gp_core import ind_close, ind_open, ind_high, ind_low, ind_volume
    pset.addPrimitive(ind_close, [], FloatType, name='close')
    pset.addPrimitive(ind_open, [], FloatType, name='open')
    pset.addPrimitive(ind_high, [], FloatType, name='high')
    pset.addPrimitive(ind_low, [], FloatType, name='low')
    pset.addPrimitive(ind_volume, [], FloatType, name='volume')

    # ==========================================================================
    # VWAP INDICATORS
    # ==========================================================================
    pset.addPrimitive(ind_vwap, [], FloatType, name='vwap')
    pset.addPrimitive(ind_vwap_distance, [], FloatType, name='vwap_dist')

    # ==========================================================================
    # OPENING RANGE INDICATORS
    # ==========================================================================
    # 5-minute opening range (scalping)
    pset.addPrimitive(_make_opening_range_high(5), [], FloatType, name='or_high_5m')
    pset.addPrimitive(_make_opening_range_low(5), [], FloatType, name='or_low_5m')

    # 15-minute opening range
    pset.addPrimitive(_make_opening_range_high(15), [], FloatType, name='or_high_15m')
    pset.addPrimitive(_make_opening_range_low(15), [], FloatType, name='or_low_15m')

    # 30-minute opening range (standard ORB)
    pset.addPrimitive(_make_opening_range_high(30), [], FloatType, name='or_high_30m')
    pset.addPrimitive(_make_opening_range_low(30), [], FloatType, name='or_low_30m')

    pset.addPrimitive(ind_opening_range_size, [], FloatType, name='or_size')

    # ==========================================================================
    # SESSION INDICATORS
    # ==========================================================================
    pset.addPrimitive(ind_day_high, [], FloatType, name='day_high')
    pset.addPrimitive(ind_day_low, [], FloatType, name='day_low')
    pset.addPrimitive(ind_day_range, [], FloatType, name='day_range')
    pset.addPrimitive(ind_distance_from_day_high, [], FloatType, name='dist_day_high')
    pset.addPrimitive(ind_distance_from_day_low, [], FloatType, name='dist_day_low')

    # ==========================================================================
    # TIME INDICATORS
    # ==========================================================================
    pset.addPrimitive(ind_minutes_since_open, [], FloatType, name='mins_open')
    pset.addPrimitive(ind_minutes_until_close, [], FloatType, name='mins_close')

    # ==========================================================================
    # VOLUME INDICATORS
    # ==========================================================================
    pset.addPrimitive(ind_relative_volume, [], FloatType, name='rel_volume')

    # ==========================================================================
    # MOMENTUM INDICATORS
    # ==========================================================================
    for period in [1, 3, 5, 10, 15]:
        pset.addPrimitive(_make_intraday_momentum(period), [], FloatType, name=f'mom_{period}m')

    # ==========================================================================
    # MOVING AVERAGES (shorter periods for intraday)
    # ==========================================================================
    for period in [5, 9, 20, 50]:
        pset.addPrimitive(_make_intraday_sma(period), [], FloatType, name=f'sma_{period}m')
        pset.addPrimitive(_make_intraday_ema(period), [], FloatType, name=f'ema_{period}m')

    # ==========================================================================
    # RSI & ATR
    # ==========================================================================
    pset.addPrimitive(_make_intraday_rsi(14), [], FloatType, name='rsi')
    pset.addPrimitive(_make_intraday_atr(14), [], FloatType, name='atr')

    # ==========================================================================
    # BAR STRUCTURE
    # ==========================================================================
    pset.addPrimitive(ind_bar_range, [], FloatType, name='bar_range')
    pset.addPrimitive(ind_bar_body, [], FloatType, name='bar_body')
    pset.addPrimitive(ind_upper_wick_pct, [], FloatType, name='upper_wick')
    pset.addPrimitive(ind_lower_wick_pct, [], FloatType, name='lower_wick')

    # ==========================================================================
    # CONSTANTS (adjusted for intraday scale)
    # ==========================================================================

    def gen_small_pct():
        """Small percentage (0.05% - 0.5%) for tight stops."""
        return random.uniform(0.0005, 0.005)

    def gen_medium_pct():
        """Medium percentage (0.5% - 2%) for targets."""
        return random.uniform(0.005, 0.02)

    def gen_time_const():
        """Time constant (5 - 60 minutes)."""
        return float(random.randint(5, 60))

    pset.addEphemeralConstant("const_small_pct", gen_small_pct, FloatType)
    pset.addEphemeralConstant("const_medium_pct", gen_medium_pct, FloatType)
    pset.addEphemeralConstant("const_time", gen_time_const, FloatType)

    # Boolean constants
    pset.addTerminal(True, BoolType, name='true')
    pset.addTerminal(False, BoolType, name='false')

    # Fixed constants (RSI levels, time thresholds, VWAP distance thresholds)
    for val in [0.0, 0.5, 1.0, 1.5, 2.0]:  # VWAP distance thresholds
        pset.addTerminal(val, FloatType, name=f'const_{str(val).replace(".", "_")}')

    for val in [30, 50, 70]:  # RSI levels
        pset.addTerminal(float(val), FloatType, name=f'const_{val}')

    for val in [30, 60, 120, 240]:  # Time thresholds (minutes)
        pset.addTerminal(float(val), FloatType, name=f'time_{val}')

    return pset


def create_intraday_boolean_primitive_set() -> gp.PrimitiveSetTyped:
    """
    Create primitive set for boolean expressions (entry/exit conditions).
    Same indicators but output type is BoolType.
    """
    pset = gp.PrimitiveSetTyped("INTRADAY_BOOL", [], BoolType)

    # Comparison functions (key for boolean output)
    pset.addPrimitive(operator.gt, [FloatType, FloatType], BoolType, name='gt')
    pset.addPrimitive(operator.lt, [FloatType, FloatType], BoolType, name='lt')
    pset.addPrimitive(operator.ge, [FloatType, FloatType], BoolType, name='ge')
    pset.addPrimitive(operator.le, [FloatType, FloatType], BoolType, name='le')

    # Logical functions
    pset.addPrimitive(operator.and_, [BoolType, BoolType], BoolType, name='and_')
    pset.addPrimitive(operator.or_, [BoolType, BoolType], BoolType, name='or_')
    pset.addPrimitive(operator.not_, [BoolType], BoolType, name='not_')

    # Arithmetic (for building float expressions)
    pset.addPrimitive(operator.add, [FloatType, FloatType], FloatType, name='add')
    pset.addPrimitive(operator.sub, [FloatType, FloatType], FloatType, name='sub')
    pset.addPrimitive(operator.mul, [FloatType, FloatType], FloatType, name='mul')
    pset.addPrimitive(protected_div, [FloatType, FloatType], FloatType, name='div')
    pset.addPrimitive(operator.neg, [FloatType], FloatType, name='neg')
    pset.addPrimitive(abs, [FloatType], FloatType, name='abs')

    # Price data
    from .gp_core import ind_close, ind_open, ind_high, ind_low, ind_volume
    pset.addPrimitive(ind_close, [], FloatType, name='close')
    pset.addPrimitive(ind_open, [], FloatType, name='open')
    pset.addPrimitive(ind_high, [], FloatType, name='high')
    pset.addPrimitive(ind_low, [], FloatType, name='low')
    pset.addPrimitive(ind_volume, [], FloatType, name='volume')

    # VWAP
    pset.addPrimitive(ind_vwap, [], FloatType, name='vwap')
    pset.addPrimitive(ind_vwap_distance, [], FloatType, name='vwap_dist')

    # Opening range
    pset.addPrimitive(_make_opening_range_high(5), [], FloatType, name='or_high_5m')
    pset.addPrimitive(_make_opening_range_low(5), [], FloatType, name='or_low_5m')
    pset.addPrimitive(_make_opening_range_high(15), [], FloatType, name='or_high_15m')
    pset.addPrimitive(_make_opening_range_low(15), [], FloatType, name='or_low_15m')
    pset.addPrimitive(_make_opening_range_high(30), [], FloatType, name='or_high_30m')
    pset.addPrimitive(_make_opening_range_low(30), [], FloatType, name='or_low_30m')

    # Session
    pset.addPrimitive(ind_day_high, [], FloatType, name='day_high')
    pset.addPrimitive(ind_day_low, [], FloatType, name='day_low')
    pset.addPrimitive(ind_distance_from_day_high, [], FloatType, name='dist_day_high')
    pset.addPrimitive(ind_distance_from_day_low, [], FloatType, name='dist_day_low')

    # Time
    pset.addPrimitive(ind_minutes_since_open, [], FloatType, name='mins_open')
    pset.addPrimitive(ind_minutes_until_close, [], FloatType, name='mins_close')

    # Volume
    pset.addPrimitive(ind_relative_volume, [], FloatType, name='rel_volume')

    # Momentum
    for period in [1, 3, 5, 10]:
        pset.addPrimitive(_make_intraday_momentum(period), [], FloatType, name=f'mom_{period}m')

    # Moving averages
    for period in [9, 20, 50]:
        pset.addPrimitive(_make_intraday_sma(period), [], FloatType, name=f'sma_{period}m')
        pset.addPrimitive(_make_intraday_ema(period), [], FloatType, name=f'ema_{period}m')

    # RSI
    pset.addPrimitive(_make_intraday_rsi(14), [], FloatType, name='rsi')

    # Constants
    def gen_small_pct():
        return random.uniform(0.0005, 0.005)

    def gen_medium_pct():
        return random.uniform(0.005, 0.02)

    pset.addEphemeralConstant("const_small_pct", gen_small_pct, FloatType)
    pset.addEphemeralConstant("const_medium_pct", gen_medium_pct, FloatType)

    pset.addTerminal(True, BoolType, name='true')
    pset.addTerminal(False, BoolType, name='false')

    for val in [0.0, 0.5, 1.0, 1.5, 2.0, 30.0, 50.0, 70.0]:
        pset.addTerminal(val, FloatType, name=f'const_{int(val) if val == int(val) else str(val).replace(".", "_")}')

    for val in [30, 60, 120]:
        pset.addTerminal(float(val), FloatType, name=f'time_{val}')

    return pset


if __name__ == "__main__":
    # Test intraday primitive set
    logging.basicConfig(level=logging.INFO)

    print("Creating intraday primitive set...")
    pset = create_intraday_primitive_set()
    print(f"  Primitives: {len(pset.primitives)}")
    print(f"  Terminals: {len(pset.terminals)}")

    # Test with sample data
    print("\nTesting with sample intraday data...")
    dates = pd.date_range('2024-01-02 09:30', periods=120, freq='1min')
    test_data = pd.DataFrame({
        'open': 450 + np.random.randn(120).cumsum() * 0.1,
        'high': 451 + np.random.randn(120).cumsum() * 0.1,
        'low': 449 + np.random.randn(120).cumsum() * 0.1,
        'close': 450 + np.random.randn(120).cumsum() * 0.1,
        'volume': np.random.randint(10000, 100000, 120)
    }, index=dates)

    set_eval_data(test_data)

    print(f"  VWAP: ${ind_vwap():.2f}")
    print(f"  VWAP Distance: {ind_vwap_distance():.3f}%")
    print(f"  OR High (30m): ${_make_opening_range_high(30)():.2f}")
    print(f"  OR Low (30m): ${_make_opening_range_low(30)():.2f}")
    print(f"  Relative Volume: {ind_relative_volume():.2f}x")
    print(f"  Minutes Since Open: {ind_minutes_since_open():.0f}")
    print(f"  Day Range: {ind_day_range():.3f}%")

    clear_eval_data()
    print("\nTest complete!")
