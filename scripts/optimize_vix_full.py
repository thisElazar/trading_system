#!/usr/bin/env python3
"""
VIX Regime Rotation Full Optimization
======================================
Tests VIX thresholds, drift threshold, and portfolio allocations.
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


def load_data() -> dict:
    """Load all ETF price data from parquet files."""
    daily_dir = DIRS["daily"]
    vix_dir = DIRS["vix"]

    all_symbols = ['QQQ', 'XLK', 'IWM', 'XLY', 'XLF', 'SPY', 'XLV', 'XLP', 'XLU', 'TLT', 'GLD', 'SHY', 'USMV', 'IEF']

    data = {}
    for symbol in all_symbols:
        path = daily_dir / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} bars")

    # Load VIX data
    vix_path = vix_dir / "vix.parquet"
    if vix_path.exists():
        vix_df = pd.read_parquet(vix_path)
        if isinstance(vix_df.index, pd.DatetimeIndex):
            if vix_df.index.tz is not None:
                vix_df.index = vix_df.index.tz_localize(None)
            vix_df.index = vix_df.index.normalize()
        data['VIX'] = vix_df
        logger.info(f"Loaded VIX: {len(vix_df)} bars")

    return data


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


def run_backtest_with_drift(data: dict, thresholds: dict, portfolios: dict,
                            drift_threshold: float = 0.05,
                            initial_capital: float = 100000) -> dict:
    """
    Run backtest with drift-based rebalancing.
    """
    vix_series = data['VIX']['close']

    # Get common dates
    common_dates = None
    for symbol, df in data.items():
        if symbol == 'VIX':
            continue
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    common_dates = common_dates.intersection(set(vix_series.index))
    dates = sorted(common_dates)[30:]

    if len(dates) < 252:
        return {'error': 'Insufficient data'}

    equity = initial_capital
    equity_curve = [equity]
    daily_returns = []

    current_regime = None
    regime_changes = 0
    drift_rebalances = 0
    high_vol_returns = []
    regime_days = {'low': 0, 'normal': 0, 'high': 0, 'extreme': 0}

    # Track actual weights
    position_values = {}
    last_rebalance = None
    min_days_between_drift = 5

    for i, date in enumerate(dates):
        vix_level = vix_series.loc[date]
        new_regime = get_regime(vix_level, thresholds)
        regime_days[new_regime] += 1

        # Get target portfolio
        portfolio = portfolios.get(new_regime, portfolios['normal'])

        # Check for regime change
        if new_regime != current_regime:
            if current_regime is not None:
                regime_changes += 1
            current_regime = new_regime
            # Rebalance to new regime
            position_values = {s: w * equity for s, w in portfolio.items()}
            last_rebalance = i

        # Check for drift rebalance (only if no recent rebalance)
        elif drift_threshold > 0 and last_rebalance is not None:
            days_since = i - last_rebalance
            if days_since >= min_days_between_drift and equity > 0:
                # Calculate current weights
                total_value = sum(position_values.values())
                if total_value > 0:
                    current_weights = {s: v / total_value for s, v in position_values.items()}
                    max_drift = max(abs(current_weights.get(s, 0) - w) for s, w in portfolio.items())
                    if max_drift > drift_threshold:
                        position_values = {s: w * equity for s, w in portfolio.items()}
                        last_rebalance = i
                        drift_rebalances += 1

        # Calculate daily return
        daily_ret = get_portfolio_returns(data, portfolio, date)
        daily_returns.append(daily_ret)

        if current_regime in ['high', 'extreme']:
            high_vol_returns.append(daily_ret)

        # Update equity and positions
        equity = equity * (1 + daily_ret)
        equity_curve.append(equity)

        # Update position values with returns
        for symbol in position_values:
            if symbol in data and date in data[symbol].index:
                idx = data[symbol].index.get_loc(date)
                if idx > 0:
                    prev_close = data[symbol].iloc[idx - 1]['close']
                    curr_close = data[symbol].iloc[idx]['close']
                    position_values[symbol] *= (1 + (curr_close - prev_close) / prev_close)

    # Calculate metrics
    returns_series = pd.Series(daily_returns)
    total_return = (equity - initial_capital) / initial_capital * 100
    years = len(dates) / 252
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = returns_series.std() * np.sqrt(252) * 100
    mean_return = returns_series.mean() * 252
    sharpe = mean_return / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0

    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = mean_return / downside_std if downside_std > 0 else 0

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    high_vol_series = pd.Series(high_vol_returns) if high_vol_returns else pd.Series([0])
    high_vol_sharpe = (high_vol_series.mean() * 252) / (high_vol_series.std() * np.sqrt(252)) if high_vol_series.std() > 0 else 0

    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    return {
        'thresholds': thresholds.copy(),
        'drift_threshold': drift_threshold,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'regime_changes': regime_changes,
        'drift_rebalances': drift_rebalances,
        'total_rebalances': regime_changes + drift_rebalances,
        'regime_days': regime_days.copy(),
        'high_vol_sharpe': high_vol_sharpe,
        'years': years
    }


def main():
    print("\n" + "=" * 70)
    print("VIX REGIME ROTATION FULL OPTIMIZATION")
    print("=" * 70)

    data = load_data()

    if len(data) < 8:
        print("\nInsufficient data.")
        return 1

    # Portfolio configurations to test
    # Current portfolios
    current_portfolios = {
        'low': {'QQQ': 0.30, 'XLK': 0.25, 'IWM': 0.20, 'XLY': 0.15, 'XLF': 0.10},
        'normal': {'SPY': 0.25, 'XLV': 0.20, 'XLK': 0.20, 'XLP': 0.20, 'XLU': 0.15},
        'high': {'XLV': 0.30, 'XLP': 0.25, 'XLU': 0.25, 'TLT': 0.20},
        'extreme': {'TLT': 0.35, 'XLV': 0.25, 'XLP': 0.25, 'GLD': 0.15},
    }

    # Enhanced defensive portfolios (more TLT/bonds in high vol)
    enhanced_defensive = {
        'low': {'QQQ': 0.30, 'XLK': 0.25, 'IWM': 0.20, 'XLY': 0.15, 'XLF': 0.10},
        'normal': {'SPY': 0.25, 'XLV': 0.20, 'XLK': 0.20, 'XLP': 0.20, 'XLU': 0.15},
        'high': {'TLT': 0.35, 'XLV': 0.25, 'XLP': 0.25, 'XLU': 0.15},  # More TLT
        'extreme': {'TLT': 0.45, 'XLV': 0.20, 'XLP': 0.20, 'GLD': 0.15},  # Even more TLT
    }

    # Aggressive low vol, ultra defensive high vol
    aggressive_defensive = {
        'low': {'QQQ': 0.35, 'XLK': 0.30, 'IWM': 0.20, 'XLY': 0.15},  # More tech
        'normal': {'SPY': 0.30, 'XLV': 0.20, 'XLK': 0.20, 'XLP': 0.15, 'XLU': 0.15},
        'high': {'TLT': 0.40, 'XLV': 0.25, 'XLP': 0.20, 'XLU': 0.15},
        'extreme': {'TLT': 0.50, 'XLV': 0.20, 'XLP': 0.15, 'GLD': 0.15},
    }

    # Best threshold from previous optimization
    best_thresholds = {'low': 15, 'high': 25, 'extreme': 35}

    print("\n" + "-" * 70)
    print("TESTING DRIFT THRESHOLDS")
    print("-" * 70)

    drift_values = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    drift_results = []

    for drift in drift_values:
        result = run_backtest_with_drift(data, best_thresholds, current_portfolios, drift)
        if 'error' not in result:
            drift_results.append(result)
            print(f"Drift {drift:.0%}: Sharpe={result['sharpe_ratio']:.3f}, DD={result['max_drawdown']:.1f}%, "
                  f"Rebal={result['total_rebalances']}")

    best_drift = sorted(drift_results, key=lambda x: x['sharpe_ratio'] - x['total_rebalances']/200, reverse=True)[0]
    print(f"\nBest Drift Threshold: {best_drift['drift_threshold']:.0%}")

    print("\n" + "-" * 70)
    print("TESTING PORTFOLIO CONFIGURATIONS")
    print("-" * 70)

    portfolio_configs = [
        ("Current", current_portfolios),
        ("Enhanced Defensive", enhanced_defensive),
        ("Aggressive/Defensive", aggressive_defensive),
    ]

    portfolio_results = []
    for name, portfolios in portfolio_configs:
        result = run_backtest_with_drift(data, best_thresholds, portfolios, best_drift['drift_threshold'])
        if 'error' not in result:
            result['portfolio_name'] = name
            portfolio_results.append(result)
            print(f"\n{name}:")
            print(f"  Sharpe:       {result['sharpe_ratio']:.3f}")
            print(f"  Sortino:      {result['sortino_ratio']:.3f}")
            print(f"  Return:       {result['annual_return']:.2f}%")
            print(f"  Max DD:       {result['max_drawdown']:.2f}%")
            print(f"  High Vol:     {result['high_vol_sharpe']:.3f}")
            print(f"  Calmar:       {result['calmar_ratio']:.3f}")

    # Best portfolio by balanced score
    for r in portfolio_results:
        r['score'] = r['sharpe_ratio'] + 0.3 * r['high_vol_sharpe'] - r['max_drawdown'] / 100

    best_portfolio = sorted(portfolio_results, key=lambda x: x['score'], reverse=True)[0]

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    print(f"\nOptimized VIX Thresholds:")
    print(f"  low:     {best_thresholds['low']} (current: 15)")
    print(f"  high:    {best_thresholds['high']} (current: 25)")
    print(f"  extreme: {best_thresholds['extreme']} (current: 40)")

    print(f"\nOptimized Drift Threshold: {best_drift['drift_threshold']:.0%} (current: 5%)")

    print(f"\nBest Portfolio Configuration: {best_portfolio['portfolio_name']}")
    print(f"  Sharpe:       {best_portfolio['sharpe_ratio']:.3f}")
    print(f"  High Vol:     {best_portfolio['high_vol_sharpe']:.3f}")
    print(f"  Max DD:       {best_portfolio['max_drawdown']:.2f}%")

    # Compare to current baseline
    current_baseline = run_backtest_with_drift(data, {'low': 15, 'high': 25, 'extreme': 40},
                                                current_portfolios, 0.05)

    if 'error' not in current_baseline:
        print(f"\nImprovement vs Current Settings:")
        print(f"  Sharpe:   {best_portfolio['sharpe_ratio'] - current_baseline['sharpe_ratio']:+.3f}")
        print(f"  Max DD:   {best_portfolio['max_drawdown'] - current_baseline['max_drawdown']:+.2f}%")
        print(f"  High Vol: {best_portfolio['high_vol_sharpe'] - current_baseline['high_vol_sharpe']:+.3f}")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
