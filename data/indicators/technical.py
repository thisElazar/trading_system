"""
Technical Indicators
====================
Calculates technical indicators for trading strategies.
Uses ta library for standard indicators, adds custom ones.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_all_indicators(df: pd.DataFrame, 
                       include_advanced: bool = False) -> pd.DataFrame:
    """
    Add all standard technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns
        include_advanced: If True, include computationally expensive indicators
        
    Returns:
        DataFrame with added indicator columns
    """
    if len(df) < 20:
        logger.warning(f"DataFrame too short ({len(df)} rows) for indicators")
        return df
    
    df = df.copy()
    
    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Basic indicators
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_volume_indicators(df)
    df = add_momentum(df)
    df = add_support_resistance(df)
    
    if include_advanced:
        df = add_volatility_indicators(df)
        df = add_trend_indicators(df)
    
    return df


def add_bollinger_bands(df: pd.DataFrame, 
                        period: int = 20, 
                        std_dev: float = 2.0) -> pd.DataFrame:
    """
    Add Bollinger Bands.
    
    Research notes:
    - 71% win rates during ranging markets on liquid instruments
    - Best used with RSI confirmation
    - Lower band reversals more reliable than upper band
    """
    df = df.copy()
    
    df['bb_middle'] = df['close'].rolling(period).mean()
    rolling_std = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
    df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)
    
    # Bandwidth (volatility measure)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Position within bands (0 = lower, 1 = upper)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index.
    
    Research notes:
    - RSI < 30: Oversold, mean reversion opportunity
    - RSI > 70: Overbought, potential reversal
    - Divergence with price often precedes reversal
    """
    df = df.copy()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI momentum
    df['rsi_sma'] = df['rsi'].rolling(5).mean()
    
    return df


def add_macd(df: pd.DataFrame, 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> pd.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence).
    
    Research notes:
    - MACD histogram divergence with price often signals reversal
    - Signal line crossovers provide entry/exit timing
    """
    df = df.copy()
    
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range.
    
    Research notes:
    - Used for position sizing and stop placement
    - 2x ATR stop-loss is common
    - High ATR = high volatility regime
    """
    df = df.copy()
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(period).mean()
    
    # ATR as percentage of price
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    return df


def add_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Add volume-based indicators.
    
    Research notes:
    - Relative volume > 100% indicates institutional attention
    - Volume divergence with price often precedes reversal
    - High volume winners outperform low volume winners
    """
    df = df.copy()
    
    # Simple moving average of volume
    df['volume_sma'] = df['volume'].rolling(period).mean()
    
    # Relative volume
    df['relative_volume'] = df['volume'] / df['volume_sma']
    
    # Volume price trend
    df['vpt'] = (df['volume'] * 
                 ((df['close'] - df['close'].shift()) / df['close'].shift())).cumsum()
    
    # On-balance volume
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Volume-weighted average price (intraday proxy)
    df['vwap_proxy'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators.
    
    Research notes:
    - 12-1 momentum (12 month, skip last month) is standard
    - Momentum strongest in mid-caps
    - Scale exposure inversely to momentum volatility
    """
    df = df.copy()
    
    # Price momentum (various periods)
    for days in [5, 10, 20, 60, 252]:
        col_name = f'momentum_{days}d'
        df[col_name] = df['close'].pct_change(days) * 100
    
    # 12-1 momentum (skip most recent month)
    if len(df) >= 252:
        # 252 trading days - 21 (1 month) = 231
        df['momentum_12_1'] = (
            df['close'].shift(21).pct_change(231) * 100
        )
    
    # Rate of change
    df['roc'] = df['close'].pct_change(10) * 100
    
    # N-day high/low
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['high_52w'] = df['high'].rolling(252).max()
    df['low_52w'] = df['low'].rolling(252).min()
    
    # Distance from 52-week high (%)
    df['dist_from_high'] = (df['close'] - df['high_52w']) / df['high_52w'] * 100
    
    return df


def add_support_resistance(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Add simple support/resistance levels.
    
    Research notes:
    - Bounces off support with volume confirmation are high probability
    - Resistance breakouts with volume often continue
    """
    df = df.copy()
    
    df['resistance'] = df['high'].rolling(period).max()
    df['support'] = df['low'].rolling(period).min()
    
    # Distance from levels
    df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close'] * 100
    df['dist_to_support'] = (df['close'] - df['support']) / df['close'] * 100
    
    # Pivot points (simple daily)
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['r1'] = 2 * df['pivot'] - df['low']
    df['s1'] = 2 * df['pivot'] - df['high']
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators.
    
    Research notes:
    - Realized volatility useful for vol-managed momentum
    - Volatility clustering means high vol persists
    - VIX mean-reverts after spikes
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Realized volatility (annualized)
    for days in [5, 10, 21, 63]:
        col_name = f'realized_vol_{days}d'
        df[col_name] = df['returns'].rolling(days).std() * np.sqrt(252) * 100
    
    # Parkinson volatility (uses high-low)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        (np.log(df['high'] / df['low']) ** 2).rolling(21).mean()
    ) * np.sqrt(252) * 100
    
    # Volatility ratio (short-term vs long-term)
    df['vol_ratio'] = df['realized_vol_5d'] / df['realized_vol_63d']
    
    return df


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend-following indicators.
    
    Research notes:
    - ADX > 25 indicates trending market
    - Moving average crossovers work better in trends
    """
    df = df.copy()
    
    # Simple moving averages
    for period in [10, 20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
    
    # Exponential moving averages
    for period in [12, 26, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Trend direction (above/below SMA)
    df['above_sma20'] = (df['close'] > df['sma_20']).astype(int)
    df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)
    
    # ADX (Average Directional Index)
    period = 14
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    df['+dm'] = np.where(
        (df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
        np.maximum(df['high'] - df['high'].shift(), 0),
        0
    )
    df['-dm'] = np.where(
        (df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
        np.maximum(df['low'].shift() - df['low'], 0),
        0
    )
    
    tr_smooth = df['tr'].ewm(span=period, adjust=False).mean()
    plus_di = 100 * df['+dm'].ewm(span=period, adjust=False).mean() / tr_smooth
    minus_di = 100 * df['-dm'].ewm(span=period, adjust=False).mean() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.ewm(span=period, adjust=False).mean()
    
    # Clean up temporary columns
    df.drop(['+dm', '-dm', 'tr'], axis=1, inplace=True, errors='ignore')
    
    return df


def calculate_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overnight gaps.
    
    Research notes:
    - Gap-fill strategy: Sharpe 2.38 on S&P 500
    - Optimal gap size: -0.15% to -0.6%
    - 120-minute hold time is optimal
    - Fill probability decreases with gap size
    """
    df = df.copy()
    
    # Gap = today's open vs yesterday's close
    df['gap'] = (df['open'] - df['close'].shift()) / df['close'].shift() * 100
    df['gap_abs'] = df['gap'].abs()
    
    # Gap classification
    df['gap_direction'] = np.where(df['gap'] > 0, 'up', 
                                   np.where(df['gap'] < 0, 'down', 'none'))
    
    # Gap size category
    conditions = [
        df['gap_abs'] > 2,
        df['gap_abs'] > 1,
        df['gap_abs'] > 0.5,
        df['gap_abs'] > 0.15
    ]
    choices = ['large', 'medium', 'small', 'tiny']
    df['gap_size'] = np.select(conditions, choices, default='minimal')
    
    # Gap fill tracking (did it fill during the day?)
    df['gap_filled'] = np.where(
        df['gap'] > 0,
        df['low'] <= df['close'].shift(),  # Gap up: did low touch prev close?
        df['high'] >= df['close'].shift()   # Gap down: did high touch prev close?
    )
    
    return df


def calculate_divergence(df: pd.DataFrame, 
                         lookback: int = 5) -> pd.DataFrame:
    """
    Calculate price/indicator divergences.
    
    Research notes:
    - Bullish divergence: lower lows in price, higher lows in indicator
    - Bearish divergence: higher highs in price, lower highs in indicator
    - MACD and RSI divergences are most reliable
    """
    df = df.copy()
    
    # Price lows and highs
    df['price_low'] = df['close'].rolling(lookback).min()
    df['price_high'] = df['close'].rolling(lookback).max()
    
    # Price trend (is current low lower than previous?)
    df['price_lower_low'] = df['price_low'] < df['price_low'].shift(lookback)
    df['price_higher_high'] = df['price_high'] > df['price_high'].shift(lookback)
    
    # RSI divergence
    if 'rsi' in df.columns:
        df['rsi_low'] = df['rsi'].rolling(lookback).min()
        df['rsi_high'] = df['rsi'].rolling(lookback).max()
        
        df['rsi_higher_low'] = df['rsi_low'] > df['rsi_low'].shift(lookback)
        df['rsi_lower_high'] = df['rsi_high'] < df['rsi_high'].shift(lookback)
        
        # Bullish: price lower low, RSI higher low
        df['rsi_bullish_div'] = df['price_lower_low'] & df['rsi_higher_low']
        # Bearish: price higher high, RSI lower high
        df['rsi_bearish_div'] = df['price_higher_high'] & df['rsi_lower_high']
    
    # MACD divergence
    if 'macd' in df.columns:
        df['macd_low'] = df['macd'].rolling(lookback).min()
        df['macd_high'] = df['macd'].rolling(lookback).max()
        
        df['macd_higher_low'] = df['macd_low'] > df['macd_low'].shift(lookback)
        df['macd_lower_high'] = df['macd_high'] < df['macd_high'].shift(lookback)
        
        df['macd_bullish_div'] = df['price_lower_low'] & df['macd_higher_low']
        df['macd_bearish_div'] = df['price_higher_high'] & df['macd_lower_high']
    
    return df


if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    
    print("Testing technical indicators...")
    
    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    # Simulate price series with trend and noise
    returns = np.random.normal(0.001, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    
    # Add all indicators
    df = add_all_indicators(df, include_advanced=True)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    
    # Group columns by category
    categories = {
        'Bollinger': [c for c in df.columns if c.startswith('bb_')],
        'RSI': [c for c in df.columns if 'rsi' in c],
        'MACD': [c for c in df.columns if 'macd' in c],
        'ATR': [c for c in df.columns if 'atr' in c],
        'Volume': [c for c in df.columns if 'volume' in c or c in ['obv', 'vpt', 'vwap_proxy', 'relative_volume']],
        'Momentum': [c for c in df.columns if 'momentum' in c or c in ['roc', 'dist_from_high']],
        'Trend': [c for c in df.columns if 'sma' in c or 'ema' in c or c == 'adx'],
        'Volatility': [c for c in df.columns if 'vol' in c and 'volume' not in c],
    }
    
    for cat, cols in categories.items():
        if cols:
            print(f"\n  {cat}:")
            for col in cols:
                print(f"    - {col}")
    
    # Show last row
    print(f"\n\nLatest values:")
    latest = df.iloc[-1]
    for col in ['close', 'rsi', 'bb_position', 'atr_pct', 'relative_volume', 'momentum_20d', 'realized_vol_21d']:
        if col in latest.index:
            print(f"  {col}: {latest[col]:.2f}")
