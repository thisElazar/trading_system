#!/usr/bin/env python3
"""
Baseline Backtest Runner
=========================
Run a quick backtest of ALL strategies to establish baseline performance
and verify everything works before GA optimization.

Usage:
    python scripts/run_baseline_backtest.py
    python scripts/run_baseline_backtest.py --quick  # Fewer symbols, faster
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS
from data.cached_data_manager import CachedDataManager
from research.backtester import Backtester
from strategies import (
    VolManagedMomentumStrategy,
    MeanReversionStrategy,
    VIXRegimeRotationStrategy,
    PairsTradingStrategy,
    RelativeVolumeBreakout,
    SectorRotationStrategy,
    QualitySmallCapValueStrategy,
    FactorMomentumStrategy,
)
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('baseline_backtest')


# All strategies to test
STRATEGIES = {
    # Tier 1: Core Alpha Generators
    'vol_managed_momentum': VolManagedMomentumStrategy,
    'quality_small_cap_value': QualitySmallCapValueStrategy,
    'factor_momentum': FactorMomentumStrategy,
    'pairs_trading': PairsTradingStrategy,
    'relative_volume_breakout': RelativeVolumeBreakout,
    # Tier 2: Regime & Tactical
    'vix_regime_rotation': VIXRegimeRotationStrategy,
    'sector_rotation': SectorRotationStrategy,
    'mean_reversion': MeanReversionStrategy,
    # Note: gap_fill excluded - requires intraday data
}


def run_baseline_backtest(quick_mode: bool = False):
    """Run baseline backtest for all strategies."""
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 70)
    logger.info("BASELINE BACKTEST - ALL STRATEGIES")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    logger.info(f"Strategies: {len(STRATEGIES)}")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n[1/3] Loading market data...")
    data_manager = CachedDataManager()
    data_manager.load_all()
    
    # Get symbols - filter by liquidity
    metadata = data_manager.get_all_metadata()
    sorted_symbols = sorted(
        metadata.items(),
        key=lambda x: x[1].get('dollar_volume', 0),
        reverse=True
    )
    
    # Select symbols based on mode
    if quick_mode:
        n_symbols = 50
    else:
        n_symbols = 200
    
    top_symbols = [sym for sym, _ in sorted_symbols[:n_symbols]]
    
    # Add ETFs required by rotation strategies
    ROTATION_ETFS = [
        # VIX Regime Rotation
        'QQQ', 'XLK', 'IWM', 'XLY', 'XLF', 'SPY', 'XLV', 'XLP', 'XLU', 'TLT', 'GLD',
        # Sector Rotation
        'XLI', 'XLB', 'XLE', 'XLC',
    ]
    
    # Ensure ETFs are included
    for etf in ROTATION_ETFS:
        if etf not in top_symbols:
            top_symbols.append(etf)
    
    logger.info(f"  Selected {len(top_symbols)} symbols (including {len(ROTATION_ETFS)} rotation ETFs)")
    
    # Build data dict
    data = {}
    missing_etfs = []
    for sym in top_symbols:
        df = data_manager.get_bars(sym)
        if df is not None and len(df) > 252:  # At least 1 year
            data[sym] = df
        elif sym in ROTATION_ETFS:
            missing_etfs.append(sym)
    
    logger.info(f"  Loaded {len(data)} symbols with sufficient history")
    
    if missing_etfs:
        logger.warning(f"  Missing rotation ETFs: {missing_etfs}")
        logger.warning(f"  Rotation strategies may have limited signals")
    
    # Load VIX data
    vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
    vix_data = None
    if vix_path.exists():
        vix_data = pd.read_parquet(vix_path)
        if 'timestamp' in vix_data.columns:
            vix_data = vix_data.set_index('timestamp')
        if vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)
        logger.info(f"  Loaded VIX data: {len(vix_data)} bars")
    else:
        logger.warning("  VIX data not found - some strategies may not work correctly")
    
    # Run backtests
    logger.info("\n[2/3] Running backtests...")
    
    results = {}
    errors = {}
    
    backtester = Backtester(initial_capital=100000)
    
    for strategy_name, strategy_class in STRATEGIES.items():
        logger.info(f"\n  Testing: {strategy_name}")
        
        try:
            # Create strategy instance
            strategy = strategy_class()
            
            # Run backtest
            result = backtester.run(
                strategy=strategy,
                data=data,
                vix_data=vix_data
            )
            
            results[strategy_name] = {
                'sharpe': result.sharpe_ratio,
                'total_return': result.total_return,  # Already in %
                'max_drawdown': result.max_drawdown_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate / 100 if result.win_rate > 1 else result.win_rate,
                'profit_factor': result.profit_factor,
            }
            
            status = "✅" if result.sharpe_ratio > 0 else "⚠️"
            logger.info(f"    {status} Sharpe: {result.sharpe_ratio:.2f}, "
                       f"Return: {result.total_return:.1f}%, "
                       f"Trades: {result.total_trades}, "
                       f"Win Rate: {result.win_rate:.1f}%")
            
        except Exception as e:
            errors[strategy_name] = str(e)
            logger.error(f"    ❌ FAILED: {e}")
    
    # Generate report
    logger.info("\n[3/3] Generating report...")
    
    report_lines = []
    report_lines.append("# Baseline Backtest Report")
    report_lines.append(f"\n**Run ID:** `{run_id}`")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Mode:** {'Quick' if quick_mode else 'Full'}")
    report_lines.append(f"**Symbols:** {len(data)}")
    report_lines.append("")
    
    # Summary table
    report_lines.append("## Results Summary")
    report_lines.append("")
    report_lines.append("| Strategy | Sharpe | Return | Max DD | Trades | Win Rate | Status |")
    report_lines.append("|----------|--------|--------|--------|--------|----------|--------|")
    
    for strategy_name in STRATEGIES.keys():
        if strategy_name in results:
            r = results[strategy_name]
            status = "✅" if r['sharpe'] > 0.5 else "⚠️" if r['sharpe'] > 0 else "❌"
            report_lines.append(
                f"| {strategy_name} | {r['sharpe']:.2f} | {r['total_return']:.1f}% | "
                f"{r['max_drawdown']:.1f}% | {r['total_trades']} | "
                f"{r['win_rate']:.1%} | {status} |"
            )
        elif strategy_name in errors:
            report_lines.append(f"| {strategy_name} | - | - | - | - | - | ❌ ERROR |")
    
    report_lines.append("")
    
    # Errors section
    if errors:
        report_lines.append("## Errors")
        report_lines.append("")
        for strategy_name, error in errors.items():
            report_lines.append(f"### {strategy_name}")
            report_lines.append(f"```\n{error}\n```")
            report_lines.append("")
    
    # Rankings
    report_lines.append("## Rankings")
    report_lines.append("")
    
    # By Sharpe
    sorted_by_sharpe = sorted(
        [(k, v['sharpe']) for k, v in results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    report_lines.append("### By Sharpe Ratio")
    report_lines.append("")
    for i, (name, sharpe) in enumerate(sorted_by_sharpe, 1):
        report_lines.append(f"{i}. **{name}**: {sharpe:.2f}")
    report_lines.append("")
    
    # Ready for GA
    report_lines.append("## GA Readiness")
    report_lines.append("")
    
    working_strategies = [k for k, v in results.items() if v['total_trades'] > 0]
    broken_strategies = [k for k, v in results.items() if v['total_trades'] == 0]
    
    report_lines.append(f"**Working strategies:** {len(working_strategies)}/{len(STRATEGIES)}")
    report_lines.append("")
    
    if working_strategies:
        report_lines.append("✅ Ready for GA optimization:")
        for s in working_strategies:
            report_lines.append(f"  - {s}")
        report_lines.append("")
    
    if broken_strategies:
        report_lines.append("⚠️ Need debugging (0 trades):")
        for s in broken_strategies:
            report_lines.append(f"  - {s}")
        report_lines.append("")
    
    if errors:
        report_lines.append("❌ Failed to run:")
        for s in errors.keys():
            report_lines.append(f"  - {s}")
        report_lines.append("")
    
    # Save report
    report_dir = DIRS.get('backtests', Path('./research/backtests'))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"BASELINE_BACKTEST_{run_id}.md"
    
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report: {report_path}")
    logger.info(f"Working strategies: {len(working_strategies)}/{len(STRATEGIES)}")
    
    if broken_strategies:
        logger.warning(f"Broken strategies (0 trades): {broken_strategies}")
    if errors:
        logger.error(f"Failed strategies: {list(errors.keys())}")
    
    # Print quick table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30} {'Sharpe':>8} {'Return':>10} {'Trades':>8}")
    print("-" * 70)
    
    for strategy_name in STRATEGIES.keys():
        if strategy_name in results:
            r = results[strategy_name]
            print(f"{strategy_name:<30} {r['sharpe']:>8.2f} {r['total_return']:>9.1f}% {r['total_trades']:>8}")
        else:
            print(f"{strategy_name:<30} {'ERROR':>8} {'-':>10} {'-':>8}")
    
    print("=" * 70)
    
    return results, errors


def main():
    parser = argparse.ArgumentParser(description='Run baseline backtest for all strategies')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Quick mode: fewer symbols for faster testing')
    
    args = parser.parse_args()
    
    results, errors = run_baseline_backtest(quick_mode=args.quick)
    
    # Return exit code based on results
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
