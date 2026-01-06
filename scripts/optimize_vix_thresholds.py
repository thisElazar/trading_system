#!/usr/bin/env python3
"""
VIX Regime Rotation Threshold Optimizer
========================================
Tests different VIX threshold configurations to optimize defensive
performance during high volatility while preserving upside.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import DIRS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

# Regime portfolios - can be optimized
REGIME_PORTFOLIOS = {
    'low': {
        'QQQ': 0.30,
        'XLK': 0.25,
        'IWM': 0.20,
        'XLY': 0.15,
        'XLF': 0.10,
    },
    'normal': {
        'SPY': 0.25,
        'XLV': 0.20,
        'XLK': 0.20,
        'XLP': 0.20,
        'XLU': 0.15,
    },
    'high': {
        'XLV': 0.30,
        'XLP': 0.25,
        'XLU': 0.25,
        'TLT': 0.20,
    },
    'extreme': {
        'TLT': 0.35,
        'XLV': 0.25,
        'XLP': 0.25,
        'GLD': 0.15,
    }
}


def load_data() -> dict:
    """Load all ETF price data from parquet files."""
    daily_dir = DIRS["daily"]
    vix_dir = DIRS["vix"]

    all_symbols = set()
    for portfolio in REGIME_PORTFOLIOS.values():
        all_symbols.update(portfolio.keys())
    all_symbols.add('SPY')

    data = {}
    for symbol in all_symbols:
        path = daily_dir / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Ensure datetime index and remove timezone info
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
            # Remove timezone info and normalize to date only
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                # Normalize to date (remove time component)
                df.index = df.index.normalize()
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} bars, {df.index.min().date()} to {df.index.max().date()}")
        else:
            logger.warning(f"Missing data for {symbol}")

    # Load VIX data
    vix_path = vix_dir / "vix.parquet"
    if vix_path.exists():
        vix_df = pd.read_parquet(vix_path)
        # Ensure datetime index
        if not isinstance(vix_df.index, pd.DatetimeIndex):
            if 'timestamp' in vix_df.columns:
                vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
                vix_df = vix_df.set_index('timestamp')
            elif 'date' in vix_df.columns:
                vix_df['date'] = pd.to_datetime(vix_df['date'])
                vix_df = vix_df.set_index('date')
        if isinstance(vix_df.index, pd.DatetimeIndex):
            if vix_df.index.tz is not None:
                vix_df.index = vix_df.index.tz_localize(None)
            vix_df.index = vix_df.index.normalize()
        data['VIX'] = vix_df
        logger.info(f"Loaded VIX: {len(vix_df)} bars, {vix_df.index.min().date()} to {vix_df.index.max().date()}")

    return data


def calculate_vix_proxy(spy_data: pd.DataFrame) -> pd.Series:
    """Calculate VIX proxy from SPY realized volatility (fallback)."""
    returns = spy_data['close'].pct_change()
    realized_vol = returns.rolling(21).std() * np.sqrt(252) * 100
    vix_proxy = realized_vol * 1.2
    vix_proxy = vix_proxy.rolling(5).mean()
    return vix_proxy


def get_vix_series(data: dict) -> pd.Series:
    """Get VIX time series - use actual VIX if available, otherwise proxy."""
    if 'VIX' in data and 'close' in data['VIX'].columns:
        logger.info("Using actual VIX data")
        return data['VIX']['close']
    elif 'SPY' in data:
        logger.info("Using VIX proxy from SPY volatility")
        return calculate_vix_proxy(data['SPY'])
    else:
        raise ValueError("No VIX data or SPY data available")


def get_regime(vix: float, thresholds: dict) -> str:
    """Classify VIX level into regime based on thresholds."""
    if pd.isna(vix):
        return 'normal'
    elif vix < thresholds['low']:
        return 'low'
    elif vix < thresholds['high']:
        return 'normal'
    elif vix < thresholds['extreme']:
        return 'high'
    else:
        return 'extreme'


def get_portfolio_returns(data: dict, portfolio: dict, date: pd.Timestamp) -> float:
    """Calculate weighted portfolio return for a single day."""
    total_return = 0.0
    total_weight = 0.0

    for symbol, weight in portfolio.items():
        if symbol not in data:
            continue

        df = data[symbol]

        if date not in df.index:
            continue

        idx = df.index.get_loc(date)
        if idx == 0:
            continue

        prev_close = df.iloc[idx - 1]['close']
        curr_close = df.iloc[idx]['close']
        daily_return = (curr_close - prev_close) / prev_close

        total_return += weight * daily_return
        total_weight += weight

    if total_weight > 0 and total_weight < 1.0:
        total_return = total_return / total_weight

    return total_return


def run_backtest(data: dict, thresholds: dict, initial_capital: float = 100000) -> dict:
    """
    Run the VIX Regime Rotation backtest with given thresholds.
    """
    vix_series = get_vix_series(data)

    # Get common dates across all symbols
    common_dates = None
    for symbol, df in data.items():
        if symbol == 'VIX':
            continue
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    # Also intersect with VIX dates
    common_dates = common_dates.intersection(set(vix_series.index))

    dates = sorted(common_dates)
    dates = dates[30:]  # Skip warmup

    if len(dates) < 252:
        return {'error': 'Insufficient data'}

    # Track performance
    equity = initial_capital
    equity_curve = [equity]
    daily_returns = []

    current_regime = None
    regime_changes = 0

    # Track high vol performance specifically
    high_vol_returns = []
    low_vol_returns = []

    # Track regime distribution
    regime_days = {'low': 0, 'normal': 0, 'high': 0, 'extreme': 0}

    for date in dates:
        if date not in vix_series.index:
            continue

        vix_level = vix_series.loc[date]
        new_regime = get_regime(vix_level, thresholds)
        regime_days[new_regime] += 1

        if new_regime != current_regime:
            if current_regime is not None:
                regime_changes += 1
            current_regime = new_regime

        portfolio = REGIME_PORTFOLIOS[current_regime]
        daily_ret = get_portfolio_returns(data, portfolio, date)
        daily_returns.append(daily_ret)

        # Track by volatility
        if current_regime in ['high', 'extreme']:
            high_vol_returns.append(daily_ret)
        else:
            low_vol_returns.append(daily_ret)

        equity = equity * (1 + daily_ret)
        equity_curve.append(equity)

    # Calculate metrics
    returns_series = pd.Series(daily_returns)

    total_return = (equity - initial_capital) / initial_capital * 100
    years = len(dates) / 252
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = returns_series.std() * np.sqrt(252) * 100
    mean_return = returns_series.mean() * 252
    sharpe = mean_return / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0

    # Sortino ratio
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = mean_return / downside_std if downside_std > 0 else 0

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # High vol metrics
    high_vol_returns_series = pd.Series(high_vol_returns) if high_vol_returns else pd.Series([0])
    high_vol_sharpe = (high_vol_returns_series.mean() * 252) / (high_vol_returns_series.std() * np.sqrt(252)) if high_vol_returns_series.std() > 0 else 0
    high_vol_total = high_vol_returns_series.sum() * 100

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    return {
        'thresholds': thresholds.copy(),
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'regime_changes': regime_changes,
        'regime_days': regime_days.copy(),
        'high_vol_sharpe': high_vol_sharpe,
        'high_vol_return': high_vol_total,
        'years': years
    }


def run_buy_and_hold(data: dict, symbol: str = 'SPY', initial_capital: float = 100000) -> dict:
    """Run buy-and-hold benchmark for comparison."""
    if symbol not in data:
        return {}

    df = data[symbol]
    df = df.iloc[30:]

    returns = df['close'].pct_change().dropna()

    equity = initial_capital
    for ret in returns:
        equity = equity * (1 + ret)

    total_return = (equity - initial_capital) / initial_capital * 100
    years = len(returns) / 252
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = returns.std() * np.sqrt(252) * 100
    mean_return = returns.mean() * 252
    sharpe = mean_return / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    prices = df['close']
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'max_drawdown': max_drawdown
    }


def optimize_thresholds(data: dict) -> list:
    """
    Grid search over VIX threshold combinations.
    """
    results = []

    # Parameter ranges to test
    low_thresholds = [12, 14, 15, 16, 18]
    high_thresholds = [22, 24, 25, 27, 30]
    extreme_thresholds = [35, 40, 45, 50]

    total = len(low_thresholds) * len(high_thresholds) * len(extreme_thresholds)
    logger.info(f"Testing {total} threshold combinations...")

    count = 0
    for low, high, extreme in product(low_thresholds, high_thresholds, extreme_thresholds):
        if low >= high or high >= extreme:
            continue

        thresholds = {'low': low, 'high': high, 'extreme': extreme}
        result = run_backtest(data, thresholds)

        if 'error' not in result:
            results.append(result)
            count += 1

            if count % 20 == 0:
                logger.info(f"Progress: {count}/{total}")

    return results


def main():
    print("\n" + "=" * 70)
    print("VIX REGIME ROTATION THRESHOLD OPTIMIZER")
    print("=" * 70)

    # Load data
    data = load_data()

    if len(data) < 5:
        print("\nInsufficient data.")
        return 1

    # Current thresholds for baseline
    current_thresholds = {'low': 15, 'high': 25, 'extreme': 40}

    print("\n" + "-" * 70)
    print("BASELINE (Current Thresholds)")
    print("-" * 70)
    print(f"Thresholds: low={current_thresholds['low']}, high={current_thresholds['high']}, extreme={current_thresholds['extreme']}")

    baseline = run_backtest(data, current_thresholds)
    benchmark = run_buy_and_hold(data)

    if 'error' in baseline:
        print(f"Error: {baseline['error']}")
        return 1

    print(f"\nBaseline Results:")
    print(f"  Total Return:     {baseline['total_return']:.2f}%")
    print(f"  Annual Return:    {baseline['annual_return']:.2f}%")
    print(f"  Sharpe Ratio:     {baseline['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:    {baseline['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:     {baseline['max_drawdown']:.2f}%")
    print(f"  Regime Changes:   {baseline['regime_changes']}")
    print(f"  High Vol Sharpe:  {baseline['high_vol_sharpe']:.3f}")

    print(f"\nSPY Buy & Hold:")
    print(f"  Total Return:     {benchmark.get('total_return', 0):.2f}%")
    print(f"  Annual Return:    {benchmark.get('annual_return', 0):.2f}%")
    print(f"  Sharpe Ratio:     {benchmark.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:     {benchmark.get('max_drawdown', 0):.2f}%")

    print(f"\nRegime Distribution:")
    total_days = sum(baseline['regime_days'].values())
    for regime, days in baseline['regime_days'].items():
        pct = days / total_days * 100 if total_days > 0 else 0
        print(f"  {regime}: {days} days ({pct:.1f}%)")

    # Run optimization
    print("\n" + "-" * 70)
    print("OPTIMIZATION")
    print("-" * 70)

    results = optimize_thresholds(data)

    if not results:
        print("No valid results from optimization.")
        return 1

    # Sort by different objectives
    print(f"\nTested {len(results)} valid threshold combinations")

    # Find best by Sharpe
    best_sharpe = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)[0]

    # Find best by Calmar (return/drawdown)
    best_calmar = sorted(results, key=lambda x: x['calmar_ratio'], reverse=True)[0]

    # Find best defensive (high vol sharpe)
    best_defensive = sorted(results, key=lambda x: x['high_vol_sharpe'], reverse=True)[0]

    # Find best balanced (weighted average of sharpe + high_vol_sharpe - regime_changes/100)
    for r in results:
        r['composite_score'] = r['sharpe_ratio'] + 0.5 * r['high_vol_sharpe'] - r['regime_changes'] / 200 - r['max_drawdown'] / 100
    best_balanced = sorted(results, key=lambda x: x['composite_score'], reverse=True)[0]

    print("\n" + "=" * 70)
    print("TOP THRESHOLD CONFIGURATIONS")
    print("=" * 70)

    configs = [
        ("Best Sharpe Ratio", best_sharpe),
        ("Best Calmar Ratio", best_calmar),
        ("Best Defensive (High Vol)", best_defensive),
        ("Best Balanced Score", best_balanced),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        print(f"  Thresholds: low={config['thresholds']['low']}, high={config['thresholds']['high']}, extreme={config['thresholds']['extreme']}")
        print(f"  Sharpe:       {config['sharpe_ratio']:.3f} (vs baseline {baseline['sharpe_ratio']:.3f})")
        print(f"  Sortino:      {config['sortino_ratio']:.3f}")
        print(f"  Calmar:       {config['calmar_ratio']:.3f}")
        print(f"  Return:       {config['annual_return']:.2f}%")
        print(f"  Max DD:       {config['max_drawdown']:.2f}%")
        print(f"  High Vol:     {config['high_vol_sharpe']:.3f}")
        print(f"  Rebalances:   {config['regime_changes']}")

    # Show improvement over baseline
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Choose the best balanced configuration
    recommended = best_balanced

    print(f"\nRecommended Configuration:")
    print(f"  low threshold:     {recommended['thresholds']['low']} (current: 15)")
    print(f"  high threshold:    {recommended['thresholds']['high']} (current: 25)")
    print(f"  extreme threshold: {recommended['thresholds']['extreme']} (current: 40)")

    print(f"\nImprovements vs Baseline:")
    print(f"  Sharpe:       {recommended['sharpe_ratio'] - baseline['sharpe_ratio']:+.3f}")
    print(f"  Max DD:       {recommended['max_drawdown'] - baseline['max_drawdown']:+.2f}% (less negative is better)")
    print(f"  High Vol:     {recommended['high_vol_sharpe'] - baseline['high_vol_sharpe']:+.3f}")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
