#!/usr/bin/env python3
"""
VIX Regime Rotation Backtest (Fixed)
=====================================
Simplified backtest that directly tests regime rotation returns.

Research benchmark: Sharpe 0.73
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import DIRS
from data.fetchers.daily_bars import DailyBarsFetcher
from data.indicators.technical import add_all_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

# Regime portfolios
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


def load_data():
    """Load all ETF price data."""
    fetcher = DailyBarsFetcher()
    
    # Get all needed symbols
    all_symbols = set()
    for portfolio in REGIME_PORTFOLIOS.values():
        all_symbols.update(portfolio.keys())
    all_symbols.add('SPY')  # Need for VIX proxy
    
    data = {}
    for symbol in all_symbols:
        df = fetcher.load_symbol(symbol)
        if df is not None and len(df) > 50:
            # Remove timezone info for easier comparison
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} bars")
        else:
            logger.warning(f"Missing or insufficient data for {symbol}")
    
    return data


def calculate_vix_proxy(spy_data: pd.DataFrame) -> pd.Series:
    """Calculate VIX proxy from SPY realized volatility."""
    returns = spy_data['close'].pct_change()
    
    # 21-day realized vol, annualized, scaled to VIX-like levels
    realized_vol = returns.rolling(21).std() * np.sqrt(252) * 100
    
    # VIX typically runs ~1.2x realized vol
    vix_proxy = realized_vol * 1.2
    
    # Smooth it a bit
    vix_proxy = vix_proxy.rolling(5).mean()
    
    return vix_proxy


def get_regime(vix: float) -> str:
    """Classify VIX level into regime."""
    if pd.isna(vix):
        return 'normal'
    elif vix < 15:
        return 'low'
    elif vix < 25:
        return 'normal'
    elif vix < 40:
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
        
        # Find this date in the data
        if date not in df.index:
            continue
        
        idx = df.index.get_loc(date)
        if idx == 0:
            continue
        
        # Daily return
        prev_close = df.iloc[idx - 1]['close']
        curr_close = df.iloc[idx]['close']
        daily_return = (curr_close - prev_close) / prev_close
        
        total_return += weight * daily_return
        total_weight += weight
    
    # Normalize if we don't have all symbols
    if total_weight > 0 and total_weight < 1.0:
        total_return = total_return / total_weight
    
    return total_return


def run_backtest(data: dict, initial_capital: float = 100000) -> dict:
    """
    Run the VIX Regime Rotation backtest.
    
    Returns dict with performance metrics.
    """
    if 'SPY' not in data:
        logger.error("SPY data required for VIX proxy")
        return {}
    
    spy = data['SPY']
    
    # Calculate VIX proxy
    vix_proxy = calculate_vix_proxy(spy)
    
    # Get common dates across all symbols
    common_dates = None
    for symbol, df in data.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))
    
    dates = sorted(common_dates)
    
    # Skip first 30 days for VIX calculation warmup
    dates = dates[30:]
    
    logger.info(f"Backtest period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
    
    # Track performance
    equity = initial_capital
    equity_curve = [equity]
    daily_returns = []
    
    current_regime = None
    regime_changes = []
    
    # Track regime distribution
    regime_days = {'low': 0, 'normal': 0, 'high': 0, 'extreme': 0}
    
    for i, date in enumerate(dates):
        # Get current VIX level
        if date not in vix_proxy.index:
            continue
        
        vix_level = vix_proxy.loc[date]
        new_regime = get_regime(vix_level)
        regime_days[new_regime] += 1
        
        # Check for regime change
        if new_regime != current_regime:
            if current_regime is not None:
                regime_changes.append({
                    'date': date,
                    'from': current_regime,
                    'to': new_regime,
                    'vix': vix_level
                })
            current_regime = new_regime
        
        # Get portfolio for current regime
        portfolio = REGIME_PORTFOLIOS[current_regime]
        
        # Calculate daily return
        daily_ret = get_portfolio_returns(data, portfolio, date)
        daily_returns.append(daily_ret)
        
        # Update equity
        equity = equity * (1 + daily_ret)
        equity_curve.append(equity)
    
    # Calculate metrics
    returns_series = pd.Series(daily_returns)
    
    total_return = (equity - initial_capital) / initial_capital * 100
    
    # Annualized return
    years = len(dates) / 252
    annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Volatility (annualized)
    volatility = returns_series.std() * np.sqrt(252) * 100
    
    # Sharpe ratio
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
    
    # Print regime distribution
    logger.info(f"\nRegime distribution:")
    for regime, days in regime_days.items():
        pct = days / len(dates) * 100
        logger.info(f"  {regime}: {days} days ({pct:.1f}%)")
    
    logger.info(f"\nRegime changes: {len(regime_changes)}")
    for change in regime_changes[:10]:  # Show first 10
        logger.info(f"  {change['date'].date()}: {change['from']} → {change['to']} (VIX: {change['vix']:.1f})")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'regime_changes': len(regime_changes),
        'equity_curve': equity_curve,
        'regime_days': regime_days
    }


def run_buy_and_hold(data: dict, symbol: str = 'SPY', initial_capital: float = 100000) -> dict:
    """Run buy-and-hold benchmark for comparison."""
    if symbol not in data:
        return {}
    
    df = data[symbol]
    
    # Skip first 30 days to match regime backtest
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
    
    # Max drawdown
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


def main():
    print("\n" + "=" * 70)
    print("VIX REGIME ROTATION BACKTEST")
    print("=" * 70)
    print(f"\nResearch benchmark: Sharpe 0.73")
    print(f"Minimum threshold:  Sharpe 0.40")
    
    # Load data
    data = load_data()
    
    if len(data) < 5:
        print("\n⚠️  Insufficient data.")
        return 1
    
    # Run strategy backtest
    print("\n" + "-" * 70)
    print("Running VIX Regime Rotation strategy...")
    results = run_backtest(data)
    
    # Run benchmark
    print("\n" + "-" * 70)
    print("Running SPY Buy-and-Hold benchmark...")
    benchmark = run_buy_and_hold(data, 'SPY')
    
    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Strategy':>15} {'SPY B&H':>15} {'Difference':>15}")
    print("-" * 70)
    
    metrics = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annual_return', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Volatility', 'volatility', '%'),
        ('Max Drawdown', 'max_drawdown', '%'),
    ]
    
    for name, key, suffix in metrics:
        strat_val = results.get(key, 0)
        bench_val = benchmark.get(key, 0)
        diff = strat_val - bench_val if key != 'max_drawdown' else bench_val - strat_val
        
        print(f"{name:<25} {strat_val:>14.2f}{suffix} {bench_val:>14.2f}{suffix} {diff:>+14.2f}{suffix}")
    
    print("-" * 70)
    print(f"{'Regime Changes':<25} {results.get('regime_changes', 0):>15}")
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    sharpe = results.get('sharpe_ratio', 0)
    research_sharpe = 0.73
    min_sharpe = 0.40
    
    print(f"\nResearch Sharpe:  {research_sharpe:.2f}")
    print(f"Minimum Sharpe:   {min_sharpe:.2f}")
    print(f"Achieved Sharpe:  {sharpe:.2f}")
    print(f"vs Research:      {sharpe/research_sharpe*100:.1f}%")
    
    if sharpe >= min_sharpe:
        print(f"\n✓ PASSED: Strategy meets minimum threshold ({sharpe:.2f} >= {min_sharpe:.2f})")
        status = 0
    else:
        print(f"\n✗ FAILED: Strategy below minimum threshold ({sharpe:.2f} < {min_sharpe:.2f})")
        status = 1
    
    # Alpha over benchmark
    alpha = results.get('annual_return', 0) - benchmark.get('annual_return', 0)
    print(f"\nAlpha vs SPY: {alpha:+.2f}% annual")
    
    print("\n" + "=" * 70)
    
    return status


if __name__ == "__main__":
    sys.exit(main())
