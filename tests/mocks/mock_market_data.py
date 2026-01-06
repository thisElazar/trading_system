"""
Mock market data generators for testing.

Provides synthetic market data generation for testing strategies
without requiring live data feeds.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def generate_mock_bars(
    symbol: str = 'TEST',
    n_days: int = 252,
    start_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate mock OHLCV bar data.

    Args:
        symbol: Stock symbol
        n_days: Number of trading days
        start_price: Starting price
        volatility: Daily volatility (std dev of returns)
        drift: Daily drift (mean return)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate close prices using geometric Brownian motion
    returns = np.random.normal(drift, volatility, n_days)
    close = start_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    # Intraday volatility for high/low
    intraday_vol = volatility * 0.5

    high = close * (1 + np.abs(np.random.normal(0, intraday_vol, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, intraday_vol, n_days)))

    # Open is previous close plus overnight gap
    open_prices = np.roll(close, 1) * (1 + np.random.normal(0, volatility * 0.3, n_days))
    open_prices[0] = start_price

    # Volume with some autocorrelation
    base_volume = 5_000_000
    volume = base_volume + np.random.normal(0, base_volume * 0.3, n_days)
    volume = np.maximum(volume, 100_000).astype(int)

    # Create datetime index
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    df = pd.DataFrame({
        'symbol': symbol,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)

    # Ensure OHLC constraints
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def generate_mock_quotes(
    symbols: List[str],
    base_prices: Optional[Dict[str, float]] = None,
    spread_bps: float = 5.0,
    seed: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Generate mock quote data (bid/ask).

    Args:
        symbols: List of symbols
        base_prices: Optional dict of symbol -> price
        spread_bps: Bid-ask spread in basis points
        seed: Random seed

    Returns:
        Dict of symbol -> {bid_price, ask_price, mid_price}
    """
    if seed is not None:
        np.random.seed(seed)

    if base_prices is None:
        base_prices = {s: 100.0 + np.random.uniform(-50, 150) for s in symbols}

    spread_pct = spread_bps / 10000

    quotes = {}
    for symbol in symbols:
        mid = base_prices.get(symbol, 100.0)
        half_spread = mid * spread_pct / 2

        quotes[symbol] = {
            'bid_price': mid - half_spread,
            'ask_price': mid + half_spread,
            'mid_price': mid,
        }

    return quotes


def generate_intraday_bars(
    symbol: str = 'TEST',
    date: datetime = None,
    bar_size_minutes: int = 1,
    market_open: Tuple[int, int] = (9, 30),
    market_close: Tuple[int, int] = (16, 0),
    open_price: float = 100.0,
    volatility: float = 0.001,  # Per-minute volatility
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate intraday bar data.

    Args:
        symbol: Stock symbol
        date: Trading date (defaults to today)
        bar_size_minutes: Bar size in minutes
        market_open: Market open time (hour, minute)
        market_close: Market close time (hour, minute)
        open_price: Opening price
        volatility: Per-bar volatility
        seed: Random seed

    Returns:
        DataFrame with intraday OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)

    if date is None:
        date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate number of bars
    start_time = date.replace(hour=market_open[0], minute=market_open[1])
    end_time = date.replace(hour=market_close[0], minute=market_close[1])
    n_bars = int((end_time - start_time).seconds / 60 / bar_size_minutes)

    # Generate timestamps
    timestamps = pd.date_range(
        start=start_time,
        periods=n_bars,
        freq=f'{bar_size_minutes}min'
    )

    # Generate returns with U-shaped volume pattern
    returns = np.random.normal(0, volatility, n_bars)

    # Add slight trend component
    returns += np.random.uniform(-0.0001, 0.0001)

    close = open_price * np.cumprod(1 + returns)

    # Volume follows U-shape (high at open/close)
    x = np.linspace(0, 1, n_bars)
    volume_shape = 1 + 0.5 * (4 * (x - 0.5) ** 2)
    volume = (1_000_000 * volume_shape * (1 + np.random.uniform(-0.3, 0.3, n_bars))).astype(int)

    df = pd.DataFrame({
        'symbol': symbol,
        'timestamp': timestamps,
        'open': np.roll(close, 1),
        'high': close * (1 + np.abs(np.random.normal(0, volatility * 0.5, n_bars))),
        'low': close * (1 - np.abs(np.random.normal(0, volatility * 0.5, n_bars))),
        'close': close,
        'volume': volume,
    })

    df.loc[0, 'open'] = open_price
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df.set_index('timestamp')


class MockMarketDataProvider:
    """
    Mock market data provider for testing.

    Provides a consistent interface for getting market data
    with configurable behavior.
    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self._price_cache: Dict[str, float] = {}
        self._bars_cache: Dict[str, pd.DataFrame] = {}

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol."""
        if symbol not in self._price_cache:
            np.random.seed(hash(symbol) % 2**32 if self.seed is None else self.seed)
            self._price_cache[symbol] = 100.0 + np.random.uniform(-50, 150)
        return self._price_cache[symbol]

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        return {s: self.get_latest_price(s) for s in symbols}

    def get_bars(
        self,
        symbol: str,
        n_days: int = 60,
        timeframe: str = 'day',
    ) -> pd.DataFrame:
        """Get historical bars for symbol."""
        cache_key = f"{symbol}_{n_days}_{timeframe}"

        if cache_key not in self._bars_cache:
            self._bars_cache[cache_key] = generate_mock_bars(
                symbol=symbol,
                n_days=n_days,
                start_price=self.get_latest_price(symbol) * 0.9,
                seed=hash(cache_key) % 2**32 if self.seed is None else self.seed,
            )

        return self._bars_cache[cache_key]

    def get_multi_bars(
        self,
        symbols: List[str],
        n_days: int = 60,
    ) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols."""
        return {s: self.get_bars(s, n_days) for s in symbols}

    def set_price(self, symbol: str, price: float) -> None:
        """Override price for a symbol (for testing)."""
        self._price_cache[symbol] = price

    def set_bars(self, symbol: str, df: pd.DataFrame) -> None:
        """Override bars for a symbol (for testing)."""
        self._bars_cache[f"{symbol}_60_day"] = df

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._price_cache.clear()
        self._bars_cache.clear()


def generate_gap_scenario(
    symbol: str = 'SPY',
    gap_pct: float = 0.02,
    fill_pct: float = 0.8,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate intraday data with a gap-and-fill scenario.

    Useful for testing gap-fill strategies.

    Args:
        symbol: Stock symbol
        gap_pct: Gap size as percentage (positive = gap up)
        fill_pct: How much of the gap fills (0-1)
        seed: Random seed

    Returns:
        Intraday DataFrame with gap scenario
    """
    if seed is not None:
        np.random.seed(seed)

    prev_close = 400.0  # Previous day close
    gap_open = prev_close * (1 + gap_pct)

    # Generate intraday with gap fill
    df = generate_intraday_bars(
        symbol=symbol,
        open_price=gap_open,
        volatility=0.0005,
        seed=seed,
    )

    # Modify to create gap fill pattern
    n_bars = len(df)
    fill_bars = int(n_bars * 0.4)  # Gap fills in first 40% of day

    # Create fill pattern
    fill_target = gap_open - (gap_open - prev_close) * fill_pct

    # Gradually move toward fill target
    fill_progression = np.linspace(0, fill_pct, fill_bars)
    fill_prices = gap_open - (gap_open - prev_close) * fill_progression

    # Apply to close prices (first 40% of day)
    df.iloc[:fill_bars, df.columns.get_loc('close')] = fill_prices

    # Recalculate OHLC
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.001, n_bars))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.001, n_bars))

    # Add metadata
    df.attrs['prev_close'] = prev_close
    df.attrs['gap_pct'] = gap_pct
    df.attrs['fill_pct'] = fill_pct

    return df


# =============================================================================
# Advanced Market Regime Simulation
# =============================================================================

def generate_vix_data(
    n_days: int = 252,
    mean_vix: float = 18.0,
    volatility: float = 0.05,
    regime_switch_prob: float = 0.02,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate realistic VIX data with regime changes.

    Args:
        n_days: Number of trading days
        mean_vix: Long-term mean VIX level
        volatility: VIX volatility
        regime_switch_prob: Probability of regime change each day
        seed: Random seed

    Returns:
        DataFrame with VIX data and regime classification
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Regime definitions
    regimes = {
        'low': {'mean': 12, 'std': 2},
        'normal': {'mean': 18, 'std': 3},
        'high': {'mean': 28, 'std': 5},
        'extreme': {'mean': 45, 'std': 10},
    }

    regime_names = list(regimes.keys())
    current_regime = 'normal'

    vix_values = []
    regime_history = []

    vix = mean_vix

    for _ in range(n_days):
        # Check for regime switch
        if np.random.random() < regime_switch_prob:
            # Transition probabilities based on current regime
            if current_regime == 'low':
                current_regime = np.random.choice(['low', 'normal'], p=[0.7, 0.3])
            elif current_regime == 'normal':
                current_regime = np.random.choice(['low', 'normal', 'high'], p=[0.2, 0.6, 0.2])
            elif current_regime == 'high':
                current_regime = np.random.choice(['normal', 'high', 'extreme'], p=[0.3, 0.5, 0.2])
            else:  # extreme
                current_regime = np.random.choice(['high', 'extreme'], p=[0.4, 0.6])

        # Mean reversion within regime
        regime_params = regimes[current_regime]
        target = regime_params['mean']
        reversion_rate = 0.1

        vix = vix + reversion_rate * (target - vix) + np.random.normal(0, regime_params['std'] * 0.1)
        vix = max(8, min(80, vix))  # Cap VIX between 8 and 80

        vix_values.append(vix)
        regime_history.append(current_regime)

    df = pd.DataFrame({
        'vix': vix_values,
        'regime': regime_history,
    }, index=dates)

    # Add derived columns
    df['vix_change'] = df['vix'].pct_change()
    df['vix_ma20'] = df['vix'].rolling(20).mean()

    return df


def generate_corporate_action_scenario(
    symbol: str,
    action_type: str = 'split',
    action_date: datetime = None,
    n_days: int = 60,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate price data with corporate action (split or dividend).

    Args:
        symbol: Stock symbol
        action_type: 'split' or 'dividend'
        action_date: Date of corporate action (defaults to middle of period)
        n_days: Number of trading days
        seed: Random seed

    Returns:
        DataFrame with adjusted and unadjusted prices
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base data
    df = generate_mock_bars(symbol, n_days=n_days, start_price=100.0, seed=seed)

    if action_date is None:
        action_date = df.index[n_days // 2]

    action_idx = df.index.get_loc(action_date) if action_date in df.index else n_days // 2

    if action_type == 'split':
        # 2-for-1 stock split
        split_ratio = 0.5

        # Create unadjusted column
        df['close_unadjusted'] = df['close'].copy()

        # Adjust pre-split prices
        df.loc[:action_idx, 'close'] = df.loc[:action_idx, 'close'] * split_ratio
        df.loc[:action_idx, 'open'] = df.loc[:action_idx, 'open'] * split_ratio
        df.loc[:action_idx, 'high'] = df.loc[:action_idx, 'high'] * split_ratio
        df.loc[:action_idx, 'low'] = df.loc[:action_idx, 'low'] * split_ratio

        df.attrs['split_ratio'] = split_ratio
        df.attrs['split_date'] = action_date

    elif action_type == 'dividend':
        # Ex-dividend drop
        dividend_yield = 0.01  # 1% dividend

        df['close_unadjusted'] = df['close'].copy()

        # Price drops by dividend amount on ex-date
        df.loc[action_idx:, 'close'] = df.loc[action_idx:, 'close'] * (1 - dividend_yield)
        df.loc[action_idx:, 'open'] = df.loc[action_idx:, 'open'] * (1 - dividend_yield)
        df.loc[action_idx:, 'high'] = df.loc[action_idx:, 'high'] * (1 - dividend_yield)
        df.loc[action_idx:, 'low'] = df.loc[action_idx:, 'low'] * (1 - dividend_yield)

        df.attrs['dividend_yield'] = dividend_yield
        df.attrs['ex_date'] = action_date

    return df


def generate_halt_scenario(
    symbol: str,
    halt_duration_bars: int = 30,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate intraday data with trading halt simulation.

    Args:
        symbol: Stock symbol
        halt_duration_bars: Number of bars during halt
        seed: Random seed

    Returns:
        Intraday DataFrame with halt period
    """
    if seed is not None:
        np.random.seed(seed)

    df = generate_intraday_bars(symbol=symbol, seed=seed)

    n_bars = len(df)
    halt_start = n_bars // 3  # Halt occurs 1/3 into the day
    halt_end = halt_start + halt_duration_bars

    # During halt, volume is 0 and prices are unchanged
    if halt_end <= n_bars:
        last_price = df.iloc[halt_start - 1]['close']
        df.iloc[halt_start:halt_end, df.columns.get_loc('volume')] = 0
        df.iloc[halt_start:halt_end, df.columns.get_loc('open')] = last_price
        df.iloc[halt_start:halt_end, df.columns.get_loc('high')] = last_price
        df.iloc[halt_start:halt_end, df.columns.get_loc('low')] = last_price
        df.iloc[halt_start:halt_end, df.columns.get_loc('close')] = last_price

        # Mark halt period
        df['is_halted'] = False
        df.iloc[halt_start:halt_end, df.columns.get_loc('is_halted')] = True

        df.attrs['halt_start_bar'] = halt_start
        df.attrs['halt_end_bar'] = halt_end

    return df


def generate_momentum_scenario(
    symbol: str = 'AAPL',
    n_days: int = 252,
    trend_strength: float = 0.001,  # Daily drift
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate trending price data for momentum strategy testing.

    Args:
        symbol: Stock symbol
        n_days: Number of trading days
        trend_strength: Daily drift (positive = uptrend)
        seed: Random seed

    Returns:
        DataFrame with trending price data
    """
    return generate_mock_bars(
        symbol=symbol,
        n_days=n_days,
        start_price=100.0,
        volatility=0.02,
        drift=trend_strength,
        seed=seed,
    )


def generate_mean_reversion_scenario(
    symbol: str = 'AAPL',
    n_days: int = 252,
    mean_price: float = 100.0,
    reversion_speed: float = 0.1,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate mean-reverting price data for mean reversion strategy testing.

    Args:
        symbol: Stock symbol
        n_days: Number of trading days
        mean_price: Long-term mean price
        reversion_speed: Speed of mean reversion (0-1)
        seed: Random seed

    Returns:
        DataFrame with mean-reverting price data
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Ornstein-Uhlenbeck process
    prices = [mean_price]
    volatility = 0.02

    for _ in range(n_days - 1):
        shock = np.random.normal(0, volatility * mean_price)
        new_price = prices[-1] + reversion_speed * (mean_price - prices[-1]) + shock
        prices.append(max(new_price, mean_price * 0.5))  # Floor at 50% of mean

    close = np.array(prices)

    df = pd.DataFrame({
        'symbol': symbol,
        'open': np.roll(close, 1) * (1 + np.random.normal(0, 0.005, n_days)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': close,
        'volume': np.random.randint(1_000_000, 10_000_000, n_days),
    }, index=dates)

    df.iloc[0, df.columns.get_loc('open')] = mean_price
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


# =============================================================================
# Multi-Asset Portfolio Simulation
# =============================================================================

def generate_correlated_assets(
    symbols: List[str],
    n_days: int = 252,
    correlation_matrix: np.ndarray = None,
    seed: int = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate correlated price data for multiple assets.

    Args:
        symbols: List of stock symbols
        n_days: Number of trading days
        correlation_matrix: Correlation matrix (uses default if None)
        seed: Random seed

    Returns:
        Dict of symbol -> DataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(symbols)

    # Default correlation matrix (moderate positive correlation)
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets) * 0.5 + 0.5
        np.fill_diagonal(correlation_matrix, 1.0)

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(correlation_matrix)

    # Generate uncorrelated returns
    uncorrelated_returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))

    # Apply correlation
    correlated_returns = uncorrelated_returns @ L.T

    # Generate price paths
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    result = {}
    for i, symbol in enumerate(symbols):
        start_price = 50 + np.random.uniform(0, 150)
        close = start_price * np.cumprod(1 + correlated_returns[:, i])

        df = pd.DataFrame({
            'symbol': symbol,
            'open': np.roll(close, 1) * (1 + np.random.normal(0, 0.005, n_days)),
            'high': close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': close,
            'volume': np.random.randint(500_000, 5_000_000, n_days),
        }, index=dates)

        df.iloc[0, df.columns.get_loc('open')] = start_price
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        result[symbol] = df

    return result


def generate_pairs_trading_data(
    symbol_a: str = 'XOM',
    symbol_b: str = 'CVX',
    n_days: int = 252,
    correlation: float = 0.85,
    cointegration_factor: float = 0.9,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate price data suitable for pairs trading testing.

    The prices are cointegrated with configurable correlation.

    Args:
        symbol_a: First symbol
        symbol_b: Second symbol
        n_days: Number of trading days
        correlation: Price correlation
        cointegration_factor: Strength of cointegration
        seed: Random seed

    Returns:
        Tuple of (df_a, df_b)
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Common factor (market)
    common_factor = np.cumsum(np.random.normal(0, 0.01, n_days))

    # Idiosyncratic components
    idio_a = np.cumsum(np.random.normal(0, 0.005 * (1 - correlation), n_days))
    idio_b = np.cumsum(np.random.normal(0, 0.005 * (1 - correlation), n_days))

    # Prices
    base_a, base_b = 100.0, 80.0
    prices_a = base_a * np.exp(cointegration_factor * common_factor + idio_a)
    prices_b = base_b * np.exp(cointegration_factor * common_factor + idio_b)

    def create_ohlcv(prices, symbol, dates):
        return pd.DataFrame({
            'symbol': symbol,
            'open': np.roll(prices, 1) * (1 + np.random.normal(0, 0.003, len(prices))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, len(prices)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, len(prices)))),
            'close': prices,
            'volume': np.random.randint(1_000_000, 5_000_000, len(prices)),
        }, index=dates)

    df_a = create_ohlcv(prices_a, symbol_a, dates)
    df_b = create_ohlcv(prices_b, symbol_b, dates)

    df_a.iloc[0, df_a.columns.get_loc('open')] = base_a
    df_b.iloc[0, df_b.columns.get_loc('open')] = base_b

    return df_a, df_b
