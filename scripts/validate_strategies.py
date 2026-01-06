#!/usr/bin/env python3
"""
Strategy Validation Suite
=========================
Comprehensive validation of all 9 strategies with:
- Walk-forward backtesting (70/30 train/test split)
- Deflated Sharpe Ratio calculation
- Strategy correlation matrix
- Performance attribution

Usage:
    python scripts/validate_strategies.py
    python scripts/validate_strategies.py --quick   # Fast mode (fewer symbols)
    python scripts/validate_strategies.py --output results.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS, STRATEGIES
from data.cached_data_manager import CachedDataManager
from research.backtester import Backtester, BacktestResult
from research.discovery.multi_objective import calculate_deflated_sharpe, calculate_fitness_vector
from research.strategy_correlation import StrategyCorrelationAnalyzer, calculate_strategy_correlations
from research.strategy_attribution import StrategyAttributionTracker, attribute_backtest_trades

# Strategy imports
from strategies.vol_managed_momentum_v2 import VolManagedMomentumV2
from strategies.pairs_trading import PairsTradingStrategy
from strategies.quality_small_cap_value import QualitySmallCapValueStrategy
from strategies.factor_momentum import FactorMomentumStrategy
from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
from strategies.sector_rotation import SectorRotationStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.relative_volume_breakout import RelativeVolumeBreakout
from strategies.gap_fill import GapFillStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# Strategy factory mapping
STRATEGY_FACTORIES = {
    'vol_managed_momentum_v2': VolManagedMomentumV2,
    'pairs_trading': PairsTradingStrategy,
    'quality_small_cap_value': QualitySmallCapValueStrategy,
    'factor_momentum': FactorMomentumStrategy,
    'vix_regime_rotation': VIXRegimeRotationStrategy,
    'sector_rotation': SectorRotationStrategy,
    'mean_reversion': MeanReversionStrategy,
    'relative_volume_breakout': RelativeVolumeBreakout,
    'gap_fill': GapFillStrategy,
}


def load_market_data(quick: bool = False) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Load market data for backtesting."""
    logger.info("Loading market data...")

    data_manager = CachedDataManager()
    if not data_manager.cache:
        data_manager.load_all()

    data = {symbol: df.copy() for symbol, df in data_manager.cache.items()}

    # Quick mode: limit symbols
    if quick and len(data) > 20:
        logger.info(f"Quick mode: limiting to 20 symbols (from {len(data)})")
        symbols = list(data.keys())[:20]
        data = {s: data[s] for s in symbols}

    # Load VIX data
    vix_data = None
    vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
    if vix_path.exists():
        vix_data = pd.read_parquet(vix_path)
        if 'timestamp' in vix_data.columns:
            vix_data = vix_data.set_index('timestamp')
        if vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)

        # Add regime classification
        vix_data['regime'] = 'normal'
        vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
        vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
        vix_data.loc[vix_data['close'] > 40, 'regime'] = 'extreme'

    logger.info(f"Loaded {len(data)} symbols, VIX: {vix_data is not None}")
    return data, vix_data


def split_train_test(data: Dict[str, pd.DataFrame],
                     vix_data: Optional[pd.DataFrame],
                     train_ratio: float = 0.7) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    # Get all unique dates
    all_dates = set()
    for df in data.values():
        if isinstance(df.index, pd.DatetimeIndex):
            all_dates.update(df.index.tolist())
        elif 'timestamp' in df.columns:
            all_dates.update(pd.to_datetime(df['timestamp']).tolist())

    if not all_dates:
        raise ValueError("No date information found in data")

    sorted_dates = sorted(all_dates)
    split_idx = int(len(sorted_dates) * train_ratio)
    if split_idx >= len(sorted_dates):
        split_idx = len(sorted_dates) - 1
    split_date = sorted_dates[split_idx]

    logger.info(f"Train/test split at {split_date} ({train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%)")

    # Split data
    train_data = {}
    test_data = {}
    for sym, df in data.items():
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            train_data[sym] = df[idx <= split_date].copy()
            test_data[sym] = df[idx > split_date].copy()
        elif 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            train_data[sym] = df[ts <= split_date].copy()
            test_data[sym] = df[ts > split_date].copy()

    # Filter empty
    train_data = {s: d for s, d in train_data.items() if len(d) >= 20}
    test_data = {s: d for s, d in test_data.items() if len(d) >= 20}

    # Split VIX
    train_vix = None
    test_vix = None
    if vix_data is not None:
        idx = vix_data.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        train_vix = vix_data[idx <= split_date].copy()
        test_vix = vix_data[idx > split_date].copy()

    return train_data, test_data, train_vix, test_vix


def run_walk_forward_backtest(
    strategy_name: str,
    train_data: Dict,
    test_data: Dict,
    train_vix: pd.DataFrame,
    test_vix: pd.DataFrame
) -> Dict:
    """Run walk-forward backtest for a single strategy."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Validating: {strategy_name}")
    logger.info(f"{'='*50}")

    if strategy_name not in STRATEGY_FACTORIES:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {'strategy': strategy_name, 'error': 'Unknown strategy'}

    backtester = Backtester(initial_capital=100000, cost_model='conservative')

    try:
        # Create strategy instance
        strategy_class = STRATEGY_FACTORIES[strategy_name]
        strategy = strategy_class()

        # Run on training data
        logger.info("Running on training data...")
        train_result = backtester.run(strategy, train_data, vix_data=train_vix)

        # Create fresh strategy for test
        strategy = strategy_class()

        # Run on test data
        logger.info("Running on test data...")
        test_result = backtester.run(strategy, test_data, vix_data=test_vix)

        # Calculate metrics
        train_sharpe = train_result.sharpe_ratio or 0
        test_sharpe = test_result.sharpe_ratio or 0

        # Degradation
        if train_sharpe > 0:
            degradation = (train_sharpe - test_sharpe) / train_sharpe
        else:
            degradation = 0

        # Deflated Sharpe Ratio (for test set)
        test_equity = pd.Series(test_result.equity_curve) if test_result.equity_curve else pd.Series([100000])
        test_returns = test_equity.pct_change().dropna()

        n_trials = 100  # Assume we've tested ~100 parameter combinations
        dsr = calculate_deflated_sharpe(
            sharpe=test_sharpe,
            n_returns=len(test_returns),
            n_trials=n_trials,
            skew=test_returns.skew() if len(test_returns) > 2 else 0,
            kurtosis=test_returns.kurtosis() if len(test_returns) > 2 else 3
        )

        # Walk-forward efficiency (ratio of OOS to IS performance)
        wf_efficiency = test_sharpe / train_sharpe if train_sharpe > 0.1 else 0

        result = {
            'strategy': strategy_name,
            'success': True,
            # Training metrics
            'train_sharpe': train_sharpe,
            'train_sortino': train_result.sortino_ratio or 0,
            'train_trades': train_result.total_trades,
            'train_win_rate': train_result.win_rate or 0,
            'train_max_dd': train_result.max_drawdown_pct or 0,
            'train_annual_return': train_result.annual_return or 0,
            # Test metrics (out-of-sample)
            'test_sharpe': test_sharpe,
            'test_sortino': test_result.sortino_ratio or 0,
            'test_trades': test_result.total_trades,
            'test_win_rate': test_result.win_rate or 0,
            'test_max_dd': test_result.max_drawdown_pct or 0,
            'test_annual_return': test_result.annual_return or 0,
            # Validation metrics
            'degradation': degradation,
            'deflated_sharpe': dsr,
            'walk_forward_efficiency': wf_efficiency,
            # Raw results for correlation analysis
            'train_equity': train_result.equity_curve,
            'test_equity': test_result.equity_curve,
            'test_trades_detail': test_result.trades,
        }

        # Log summary
        logger.info(f"  Train: Sharpe={train_sharpe:.2f}, Trades={train_result.total_trades}")
        logger.info(f"  Test:  Sharpe={test_sharpe:.2f}, Trades={test_result.total_trades}")
        logger.info(f"  Degradation: {degradation*100:.1f}%")
        logger.info(f"  Deflated Sharpe: {dsr:.3f}")
        logger.info(f"  Walk-Forward Efficiency: {wf_efficiency:.2f}")

        return result

    except Exception as e:
        logger.error(f"Failed to validate {strategy_name}: {e}", exc_info=True)
        return {'strategy': strategy_name, 'success': False, 'error': str(e)}


def calculate_correlation_matrix(results: List[Dict]) -> Dict:
    """Calculate correlation matrix from backtest results."""
    logger.info("\n" + "="*50)
    logger.info("Calculating Strategy Correlations")
    logger.info("="*50)

    analyzer = StrategyCorrelationAnalyzer()

    for result in results:
        if not result.get('success') or not result.get('test_equity'):
            continue

        equity = pd.Series(result['test_equity'])
        returns = equity.pct_change().dropna()

        if len(returns) > 30:
            # Create date index
            date_index = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
            returns.index = date_index
            analyzer.add_strategy_returns(result['strategy'], returns)

    corr_result = analyzer.calculate_correlations()

    if corr_result.correlation_matrix is not None and len(corr_result.correlation_matrix) > 0:
        logger.info("\nCorrelation Matrix:")
        print(corr_result.correlation_matrix.round(3).to_string())

        logger.info(f"\nDiversification Score: {corr_result.get_diversification_score():.3f}")

        highly_corr = corr_result.get_highly_correlated_pairs(0.6)
        if highly_corr:
            logger.info("\nHighly Correlated Pairs (>0.6):")
            for s1, s2, corr in highly_corr:
                logger.info(f"  {s1} vs {s2}: {corr:.3f}")

        # Get adjusted weights
        weights = analyzer.get_correlation_adjusted_weights()
        logger.info("\nCorrelation-Adjusted Weights:")
        for s, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {s}: {w:.1%}")

        return {
            'correlation_matrix': corr_result.correlation_matrix.to_dict(),
            'diversification_score': corr_result.get_diversification_score(),
            'highly_correlated_pairs': [(s1, s2, c) for s1, s2, c in highly_corr],
            'adjusted_weights': weights,
        }

    return {}


def generate_report(results: List[Dict], correlation: Dict) -> Dict:
    """Generate final validation report."""
    logger.info("\n" + "="*70)
    logger.info("STRATEGY VALIDATION REPORT")
    logger.info("="*70)

    # Summary table
    summary = []
    for r in results:
        if r.get('success'):
            summary.append({
                'Strategy': r['strategy'],
                'Train Sharpe': f"{r['train_sharpe']:.2f}",
                'Test Sharpe': f"{r['test_sharpe']:.2f}",
                'Degradation': f"{r['degradation']*100:.1f}%",
                'DSR': f"{r['deflated_sharpe']:.3f}",
                'WF Eff': f"{r['walk_forward_efficiency']:.2f}",
                'Test Trades': r['test_trades'],
                'Test WR': f"{r['test_win_rate']:.1f}%",
            })

    if summary:
        df = pd.DataFrame(summary)
        print("\n" + df.to_string(index=False))

    # Thresholds check
    logger.info("\n" + "-"*50)
    logger.info("THRESHOLD CHECKS (Pi Deployment Readiness)")
    logger.info("-"*50)

    passed = 0
    failed = 0
    warnings = 0

    for r in results:
        if not r.get('success'):
            failed += 1
            logger.warning(f"  FAIL: {r['strategy']} - {r.get('error', 'Unknown error')}")
            continue

        issues = []

        # DSR threshold (0.7 = 70% probability not due to luck)
        if r['deflated_sharpe'] < 0.7:
            issues.append(f"DSR {r['deflated_sharpe']:.3f} < 0.7")

        # Walk-forward efficiency (should be > 0.5)
        if r['walk_forward_efficiency'] < 0.5:
            issues.append(f"WF Eff {r['walk_forward_efficiency']:.2f} < 0.5")

        # Test Sharpe minimum
        if r['test_sharpe'] < 0.5:
            issues.append(f"Test Sharpe {r['test_sharpe']:.2f} < 0.5")

        # Degradation check
        if r['degradation'] > 0.5:
            issues.append(f"Degradation {r['degradation']*100:.0f}% > 50%")

        # Minimum trades
        if r['test_trades'] < 30:
            issues.append(f"Only {r['test_trades']} trades < 30")

        if issues:
            warnings += 1
            logger.warning(f"  WARN: {r['strategy']}: {', '.join(issues)}")
        else:
            passed += 1
            logger.info(f"  PASS: {r['strategy']}")

    logger.info(f"\nSummary: {passed} passed, {warnings} warnings, {failed} failed")

    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_strategies': len(results),
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
        },
        'strategies': [
            {k: v for k, v in r.items() if k not in ('train_equity', 'test_equity', 'test_trades_detail')}
            for r in results
        ],
        'correlation': correlation,
        'deployment_ready': passed >= 5 and failed == 0,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description='Validate all trading strategies')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode (fewer symbols)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON file')
    parser.add_argument('--strategies', '-s', nargs='+', default=None, help='Specific strategies to validate')
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"Strategy Validation Suite - {start_time.isoformat()}")

    # Load data
    data, vix_data = load_market_data(quick=args.quick)

    if len(data) < 5:
        logger.error("Insufficient data for validation")
        return 1

    # Split train/test
    train_data, test_data, train_vix, test_vix = split_train_test(data, vix_data)

    logger.info(f"Train: {len(train_data)} symbols")
    logger.info(f"Test: {len(test_data)} symbols")

    # Determine strategies to validate
    strategies = args.strategies if args.strategies else list(STRATEGY_FACTORIES.keys())

    # Run walk-forward backtests
    results = []
    for strategy_name in strategies:
        result = run_walk_forward_backtest(
            strategy_name,
            train_data, test_data,
            train_vix, test_vix
        )
        results.append(result)

    # Calculate correlations
    correlation = calculate_correlation_matrix(results)

    # Generate report
    report = generate_report(results, correlation)

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_path}")

    # Print deployment status
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nValidation completed in {duration:.1f}s")

    if report['deployment_ready']:
        logger.info("\n*** DEPLOYMENT READY ***")
        return 0
    else:
        logger.warning("\n*** NOT READY FOR DEPLOYMENT ***")
        logger.warning("Review warnings and fix issues before deploying to Pi")
        return 1


if __name__ == "__main__":
    sys.exit(main())
