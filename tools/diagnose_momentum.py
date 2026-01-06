"""
Momentum Strategy Diagnostic - FIXED VERSION
=============================================
Uses UNION of dates and filters for stocks with 5+ years of history.
"""

import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_diagnostic():
    from data.cached_data_manager import CachedDataManager
    from config import DIRS
    
    print("="*70)
    print("MOMENTUM STRATEGY DIAGNOSTIC (FIXED)")
    print("="*70)
    
    # Load data
    data_mgr = CachedDataManager()
    if not data_mgr.cache:
        data_mgr.load_all()
    
    # Load VIX
    vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
    vix_data = None
    if vix_path.exists():
        vix_data = pd.read_parquet(vix_path)
        if 'timestamp' in vix_data.columns:
            vix_data = vix_data.set_index('timestamp')
        if vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)
    
    # Get all stocks
    metadata = data_mgr.get_all_metadata()
    sorted_symbols = sorted(metadata.items(), key=lambda x: x[1].get('dollar_volume', 0), reverse=True)[:500]
    
    data = {}
    for symbol, _ in sorted_symbols:
        df = data_mgr.get_bars(symbol)
        if df is not None and len(df) >= 300:
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data[symbol] = df
    
    print(f"\n1. UNIVERSE SIZE")
    print(f"   Total loaded: {len(data)} stocks")
    
    # FIX: Filter to stocks with 5+ years of data (1260 trading days)
    long_history_stocks = {sym: df for sym, df in data.items() if len(df) >= 1260}
    print(f"   Stocks with 5+ years: {len(long_history_stocks)} stocks")
    
    # Use these for analysis
    data = long_history_stocks
    
    if len(data) < 50:
        print("   ERROR: Not enough stocks with 5+ years of data!")
        return
    
    # FIX: Use UNION of dates (not intersection)
    all_dates = set()
    for symbol, df in data.items():
        all_dates.update(df.index.tolist())
    
    all_dates = sorted(all_dates)
    print(f"   Date range (union): {len(all_dates)} days ({all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')})")
    
    # Calculate momentum for each month
    print(f"\n2. MOMENTUM SIGNAL ANALYSIS")
    
    formation_period = 252  # 12 months
    skip_period = 21        # 1 month
    required_bars = formation_period + skip_period
    
    # Get monthly rebalance dates (every ~21 trading days)
    monthly_dates = [d for i, d in enumerate(all_dates) if i % 21 == 0 and i >= required_bars]
    
    print(f"   Rebalance dates: {len(monthly_dates)}")
    
    # Track decile returns
    decile_returns = {i: [] for i in range(1, 11)}
    winner_returns = []
    loser_returns = []
    wml_returns = []
    
    for i, rebal_date in enumerate(monthly_dates[:-1]):
        next_rebal = monthly_dates[i + 1]
        
        # Calculate momentum on rebalance date
        momentum_scores = {}
        
        for symbol, df in data.items():
            # Check if this stock has data around this date
            mask = df.index <= rebal_date
            df_subset = df.loc[mask]
            
            if len(df_subset) < required_bars:
                continue
            
            try:
                recent_price = df_subset['close'].iloc[-(skip_period + 1)]
                old_price = df_subset['close'].iloc[-required_bars]
                if old_price > 0:
                    momentum_scores[symbol] = (recent_price - old_price) / old_price
            except (IndexError, KeyError, ZeroDivisionError, TypeError) as e:
                logger.debug(f"Could not calculate momentum for {symbol}: {e}")
                continue
        
        if len(momentum_scores) < 50:
            continue
        
        # Rank into deciles
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        n_per_decile = len(ranked) // 10
        
        # Calculate forward returns for each decile
        decile_fwd_returns = {i: [] for i in range(1, 11)}
        
        for rank, (symbol, mom) in enumerate(ranked):
            decile = min((rank // n_per_decile) + 1, 10)
            
            # Get forward return to next rebalance
            df = data[symbol]
            
            # Find closest dates in this stock's data
            start_mask = df.index <= rebal_date
            end_mask = df.index <= next_rebal
            
            if start_mask.sum() > 0 and end_mask.sum() > 0:
                start_price = df.loc[start_mask, 'close'].iloc[-1]
                end_price = df.loc[end_mask, 'close'].iloc[-1]
                fwd_return = (end_price - start_price) / start_price
                decile_fwd_returns[decile].append(fwd_return)
        
        # Average forward return per decile
        for decile in range(1, 11):
            if decile_fwd_returns[decile]:
                avg_ret = np.mean(decile_fwd_returns[decile])
                decile_returns[decile].append(avg_ret)
        
        # Winner and loser portfolio returns
        if decile_fwd_returns[1]:
            winner_ret = np.mean(decile_fwd_returns[1])
            winner_returns.append(winner_ret)
        if decile_fwd_returns[10]:
            loser_ret = np.mean(decile_fwd_returns[10])
            loser_returns.append(loser_ret)
        
        if decile_fwd_returns[1] and decile_fwd_returns[10]:
            wml = np.mean(decile_fwd_returns[1]) - np.mean(decile_fwd_returns[10])
            wml_returns.append(wml)
    
    # Analyze momentum spread
    print(f"\n   DECILE RETURNS (monthly avg) - {len(wml_returns)} months of data:")
    print(f"   {'Decile':<10} {'Avg Return':>12} {'Std Dev':>12} {'Sharpe':>10}")
    print(f"   {'-'*46}")
    
    for decile in range(1, 11):
        if decile_returns[decile]:
            avg = np.mean(decile_returns[decile]) * 100
            std = np.std(decile_returns[decile]) * 100
            sharpe = (avg / std * np.sqrt(12)) if std > 0 else 0
            label = "← WINNERS" if decile == 1 else "← LOSERS" if decile == 10 else ""
            print(f"   {decile:<10} {avg:>11.2f}% {std:>11.2f}% {sharpe:>10.2f} {label}")
    
    # Winner minus Loser spread
    print(f"\n   MOMENTUM SPREAD (Winners - Losers):")
    wml_sharpe = 0
    if wml_returns:
        wml_avg = np.mean(wml_returns) * 100
        wml_std = np.std(wml_returns) * 100
        wml_sharpe = (wml_avg / wml_std * np.sqrt(12)) if wml_std > 0 else 0
        print(f"   Monthly avg: {wml_avg:.2f}%")
        print(f"   Monthly std: {wml_std:.2f}%")
        print(f"   Annualized Sharpe: {wml_sharpe:.2f}")
        print(f"   Research expects: 0.8-1.0 Sharpe for raw momentum")
    
    # Analyze crashes
    print(f"\n3. MOMENTUM CRASH ANALYSIS")
    if winner_returns:
        winner_series = pd.Series(winner_returns)
        worst_months = winner_series.nsmallest(5)
        print(f"   Worst 5 months for winners:")
        for i, (idx, ret) in enumerate(worst_months.items()):
            if idx < len(monthly_dates):
                date = monthly_dates[idx].strftime('%Y-%m')
                print(f"   {i+1}. {date}: {ret*100:.1f}%")
    
    # Analyze by year
    print(f"\n4. MOMENTUM BY YEAR")
    if wml_returns and len(monthly_dates) > 1:
        year_returns = {}
        for i, wml in enumerate(wml_returns):
            if i < len(monthly_dates):
                year = monthly_dates[i].year
                if year not in year_returns:
                    year_returns[year] = []
                year_returns[year].append(wml)
        
        print(f"   {'Year':<8} {'Months':>8} {'Avg WML':>10} {'Cum Ret':>10}")
        print(f"   {'-'*38}")
        for year in sorted(year_returns.keys()):
            rets = year_returns[year]
            avg = np.mean(rets) * 100
            cum = (np.prod([1+r for r in rets]) - 1) * 100
            print(f"   {year:<8} {len(rets):>8} {avg:>9.2f}% {cum:>9.1f}%")
    
    # Volatility clustering
    print(f"\n5. VOLATILITY CLUSTERING")
    vol_return_pairs = []
    if winner_returns and len(winner_returns) > 10:
        winner_series = pd.Series(winner_returns)
        rolling_vol = winner_series.rolling(6).std() * np.sqrt(12)
        
        for i in range(6, len(winner_returns)):
            prior_vol = rolling_vol.iloc[i-1]
            next_return = winner_returns[i]
            if not np.isnan(prior_vol):
                vol_return_pairs.append((prior_vol, next_return))
        
        if vol_return_pairs:
            vols, rets = zip(*vol_return_pairs)
            correlation = np.corrcoef(vols, rets)[0, 1]
            print(f"   Correlation (prior vol → next return): {correlation:.3f}")
            print(f"   (Negative = high vol predicts crashes = vol scaling helps)")
    
    # Summary
    print(f"\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if wml_sharpe > 0.5:
        print(f"✓ Momentum signal is present (Sharpe {wml_sharpe:.2f})")
        print(f"  → Vol-managed momentum should work in this period")
    elif wml_sharpe > 0:
        print(f"⚠️  Weak momentum signal (Sharpe {wml_sharpe:.2f})")
        print(f"  → Momentum is marginally positive but weak")
    else:
        print(f"❌ Momentum INVERTED (Sharpe {wml_sharpe:.2f})")
        print(f"  → Losers outperforming winners - momentum crash regime")
        print(f"  → Consider mean reversion or reduced momentum allocation")
    
    if vol_return_pairs:
        correlation = np.corrcoef(vols, rets)[0, 1]
        if correlation < -0.1:
            print(f"✓ Volatility clustering present (corr {correlation:.2f})")
        else:
            print(f"⚠️  Weak volatility clustering (corr {correlation:.2f})")
    
    return {
        'wml_sharpe': wml_sharpe,
        'n_months': len(wml_returns),
        'universe_size': len(data)
    }


if __name__ == "__main__":
    run_diagnostic()
