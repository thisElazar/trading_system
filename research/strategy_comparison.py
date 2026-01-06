"""
Strategy Comparison Framework
==============================
Systematically compare multiple strategies head-to-head on identical data.

Features:
- Side-by-side performance metrics
- Regime-specific breakdowns (high/low VIX)
- Transaction cost sensitivity analysis
- Statistical significance testing
- Markdown report generation
- Full database logging

Usage:
    from research.strategy_comparison import StrategyComparison
    
    comp = StrategyComparison()
    results = comp.compare_all(start_date='2024-01-01')
    comp.generate_report(results)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import uuid
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from config import STRATEGIES, VALIDATION, DIRS
from research.backtester import Backtester, BacktestResult
from data.cached_data_manager import CachedDataManager
from data.storage.db_manager import get_db
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

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from comparing multiple strategies."""
    run_id: str
    timestamp: str
    strategies: List[str]
    period: Tuple[str, str]
    
    # Individual results
    backtest_results: Dict[str, BacktestResult] = field(default_factory=dict)
    
    # Regime-specific results
    high_vix_results: Dict[str, BacktestResult] = field(default_factory=dict)
    low_vix_results: Dict[str, BacktestResult] = field(default_factory=dict)
    
    # Statistical tests
    sharpe_rankings: List[Tuple[str, float]] = field(default_factory=list)
    statistical_significance: Dict[str, dict] = field(default_factory=dict)
    
    # Cost sensitivity
    cost_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Summary
    best_overall: Optional[str] = None
    best_high_vix: Optional[str] = None
    best_low_vix: Optional[str] = None
    best_risk_adjusted: Optional[str] = None


class StrategyComparison:
    """
    Compare multiple trading strategies systematically.
    
    Runs all strategies on identical data and generates comprehensive
    comparison reports with statistical validation.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.data_manager = CachedDataManager()
        self.db = get_db()
        
        # Available strategies - all 9 in portfolio
        self.strategies = {
            # Tier 1: Core Alpha Generators
            'vol_managed_momentum': VolManagedMomentumStrategy(),
            'quality_small_cap_value': QualitySmallCapValueStrategy(),
            'factor_momentum': FactorMomentumStrategy(),
            'pairs_trading': PairsTradingStrategy(),
            'relative_volume_breakout': RelativeVolumeBreakout(),
            # Tier 2: Regime & Tactical
            'vix_regime_rotation': VIXRegimeRotationStrategy(),
            'sector_rotation': SectorRotationStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            # Note: gap_fill excluded - requires intraday data
        }
        
        logger.info(f"Initialized comparison with {len(self.strategies)} strategies")
    
    def load_data(self, 
                  symbols: List[str] = None,
                  start_date: datetime = None,
                  end_date: datetime = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load market data for comparison.
        
        Args:
            symbols: Optional list of symbols to load
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            (data dict, vix dataframe)
        """
        logger.info("Loading data for comparison...")
        
        # Load all cached data
        if not self.data_manager.cache:
            self.data_manager.load_all()
        
        # Get data
        if symbols:
            data = {s: self.data_manager.get_bars(s) for s in symbols 
                   if s in self.data_manager.cache}
        else:
            # Use all available
            data = {s: df.copy() for s, df in self.data_manager.cache.items()}
        
        # Filter by date if specified
        if start_date or end_date:
            for symbol in list(data.keys()):
                df = data[symbol]
                if 'timestamp' in df.columns:
                    if start_date:
                        df = df[df['timestamp'] >= start_date]
                    if end_date:
                        df = df[df['timestamp'] <= end_date]
                    data[symbol] = df
        
        # Load VIX data
        vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
        vix_data = None
        
        if vix_path.exists():
            vix_data = pd.read_parquet(vix_path)
            if 'timestamp' in vix_data.columns:
                vix_data = vix_data.set_index('timestamp')
            
            # Ensure timezone-naive index
            if vix_data.index.tz is not None:
                vix_data.index = vix_data.index.tz_localize(None)
            
            # Add regime classification (always do this)
            vix_data['regime'] = 'normal'
            vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
            vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
            vix_data.loc[vix_data['close'] > 40, 'regime'] = 'extreme'
        
        logger.info(f"Loaded {len(data)} symbols, VIX: {vix_data is not None}")
        
        return data, vix_data
    
    def run_backtest(self,
                     strategy_name: str,
                     data: Dict[str, pd.DataFrame],
                     vix_data: pd.DataFrame = None,
                     start_date: datetime = None,
                     end_date: datetime = None,
                     slippage_model: str = 'volatility') -> BacktestResult:
        """Run backtest for a single strategy."""
        
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        # Reset strategy state before each backtest run
        if hasattr(strategy, 'last_rebalance_month'):
            strategy.last_rebalance_month = None
        if hasattr(strategy, 'last_regime'):
            strategy.last_regime = None
        if hasattr(strategy, 'strategy_returns'):
            strategy.strategy_returns = []
        
        logger.info(f"Running backtest: {strategy_name}")
        
        backtester = Backtester(
            initial_capital=self.initial_capital,
            slippage_model=slippage_model
        )
        
        result = backtester.run(
            strategy=strategy,
            data=data,
            start_date=start_date,
            end_date=end_date,
            vix_data=vix_data
        )
        
        return result
    
    def run_regime_specific_backtests(self,
                                       strategy_name: str,
                                       data: Dict[str, pd.DataFrame],
                                       vix_data: pd.DataFrame) -> Tuple[BacktestResult, BacktestResult]:
        """
        Run backtests for high and low VIX regimes separately.
        
        Returns:
            (high_vix_result, low_vix_result)
        """
        if vix_data is None:
            logger.warning("No VIX data for regime analysis")
            return None, None
        
        # Split data by regime
        high_vix_dates = vix_data[vix_data['regime'].isin(['high', 'extreme'])].index
        low_vix_dates = vix_data[vix_data['regime'] == 'low'].index
        
        # Filter data for each regime
        high_vix_data = {}
        low_vix_data = {}
        
        for symbol, df in data.items():
            if 'timestamp' in df.columns:
                high_vix_data[symbol] = df[df['timestamp'].isin(high_vix_dates)]
                low_vix_data[symbol] = df[df['timestamp'].isin(low_vix_dates)]
            else:
                high_vix_data[symbol] = df[df.index.isin(high_vix_dates)]
                low_vix_data[symbol] = df[df.index.isin(low_vix_dates)]
        
        # Run backtests
        high_result = None
        low_result = None
        
        if any(len(df) > 20 for df in high_vix_data.values()):
            logger.info(f"  High VIX regime backtest: {strategy_name}")
            high_result = self.run_backtest(strategy_name, high_vix_data, vix_data)
        
        if any(len(df) > 20 for df in low_vix_data.values()):
            logger.info(f"  Low VIX regime backtest: {strategy_name}")
            low_result = self.run_backtest(strategy_name, low_vix_data, vix_data)
        
        return high_result, low_result
    
    def calculate_statistical_significance(self,
                                           results: Dict[str, BacktestResult]) -> Dict[str, dict]:
        """
        Test statistical significance between strategies.
        
        Uses t-tests on daily returns to determine if differences are significant.
        
        Returns:
            Dict mapping strategy pairs to test results
        """
        significance = {}
        
        strategy_names = list(results.keys())
        
        for i, strat_a in enumerate(strategy_names):
            for strat_b in strategy_names[i+1:]:
                result_a = results[strat_a]
                result_b = results[strat_b]
                
                # Get equity curves
                equity_a = pd.Series(result_a.equity_curve)
                equity_b = pd.Series(result_b.equity_curve)
                
                if len(equity_a) < 2 or len(equity_b) < 2:
                    continue
                
                # Calculate returns
                returns_a = equity_a.pct_change().dropna()
                returns_b = equity_b.pct_change().dropna()
                
                # Align lengths
                min_len = min(len(returns_a), len(returns_b))
                returns_a = returns_a.iloc[:min_len]
                returns_b = returns_b.iloc[:min_len]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((returns_a.std()**2 + returns_b.std()**2) / 2)
                cohens_d = (returns_a.mean() - returns_b.mean()) / pooled_std if pooled_std > 0 else 0
                
                pair_key = f"{strat_a}_vs_{strat_b}"
                significance[pair_key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'cohens_d': cohens_d,
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                }
        
        return significance
    
    def cost_sensitivity_analysis(self,
                                   strategy_name: str,
                                   data: Dict[str, pd.DataFrame],
                                   vix_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Test strategy performance with different transaction cost assumptions.
        
        Returns:
            Dict mapping cost model to resulting Sharpe ratio
        """
        logger.info(f"Running cost sensitivity: {strategy_name}")
        
        results = {}
        
        # Test with different slippage models
        for model in ['none', 'volatility']:
            result = self.run_backtest(strategy_name, data, vix_data, slippage_model=model)
            results[model] = result.sharpe_ratio
        
        return results
    
    def compare_all(self,
                    symbols: List[str] = None,
                    start_date: str = None,
                    end_date: str = None) -> ComparisonResult:
        """
        Run complete comparison of all strategies.
        
        Args:
            symbols: Optional list of symbols to test on
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            ComparisonResult with full analysis
        """
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        logger.info("=" * 60)
        logger.info("STARTING STRATEGY COMPARISON")
        logger.info("=" * 60)
        
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Load data
        data, vix_data = self.load_data(symbols, start_dt, end_dt)
        
        if not data:
            raise ValueError("No data loaded")
        
        # Initialize result
        result = ComparisonResult(
            run_id=run_id,
            timestamp=timestamp,
            strategies=list(self.strategies.keys()),
            period=(start_date or 'earliest', end_date or 'latest')
        )
        
        # Run backtests for each strategy
        for strategy_name in self.strategies.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {strategy_name}")
            logger.info(f"{'='*60}")
            
            # Main backtest
            backtest_result = self.run_backtest(strategy_name, data, vix_data, start_dt, end_dt)
            result.backtest_results[strategy_name] = backtest_result
            
            # Log to database
            self.db.log_backtest(
                run_id=run_id,
                strategy=strategy_name,
                start_date=backtest_result.start_date,
                end_date=backtest_result.end_date,
                metrics=backtest_result.to_dict(),
                params=None
            )
            
            # Regime-specific tests
            if vix_data is not None:
                high_result, low_result = self.run_regime_specific_backtests(
                    strategy_name, data, vix_data
                )
                
                if high_result:
                    result.high_vix_results[strategy_name] = high_result
                if low_result:
                    result.low_vix_results[strategy_name] = low_result
            
            # Cost sensitivity
            cost_results = self.cost_sensitivity_analysis(strategy_name, data, vix_data)
            result.cost_sensitivity[strategy_name] = cost_results
        
        # Statistical analysis
        result.statistical_significance = self.calculate_statistical_significance(
            result.backtest_results
        )
        
        # Rankings
        sharpe_scores = [(name, res.sharpe_ratio) 
                         for name, res in result.backtest_results.items()]
        result.sharpe_rankings = sorted(sharpe_scores, key=lambda x: x[1], reverse=True)
        
        # Determine winners
        if result.sharpe_rankings:
            result.best_overall = result.sharpe_rankings[0][0]
            result.best_risk_adjusted = result.best_overall
        
        if result.high_vix_results:
            high_sharpe = [(n, r.sharpe_ratio) for n, r in result.high_vix_results.items()]
            result.best_high_vix = max(high_sharpe, key=lambda x: x[1])[0] if high_sharpe else None
        
        if result.low_vix_results:
            low_sharpe = [(n, r.sharpe_ratio) for n, r in result.low_vix_results.items()]
            result.best_low_vix = max(low_sharpe, key=lambda x: x[1])[0] if low_sharpe else None
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON COMPLETE")
        logger.info("=" * 60)
        
        return result
    
    def generate_report(self, result: ComparisonResult, output_path: Path = None) -> str:
        """
        Generate markdown comparison report.
        
        Args:
            result: ComparisonResult from compare_all()
            output_path: Optional path to save report
            
        Returns:
            Report content as string
        """
        lines = []
        
        # Header
        lines.append("# Strategy Comparison Report")
        lines.append(f"\n**Run ID:** `{result.run_id}`")
        lines.append(f"**Timestamp:** {result.timestamp}")
        lines.append(f"**Period:** {result.period[0]} to {result.period[1]}")
        lines.append(f"**Strategies Compared:** {len(result.strategies)}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"**Best Overall:** {result.best_overall}")
        lines.append(f"**Best High VIX:** {result.best_high_vix}")
        lines.append(f"**Best Low VIX:** {result.best_low_vix}")
        lines.append("")
        
        # Overall Performance Comparison
        lines.append("## Overall Performance")
        lines.append("")
        lines.append("| Strategy | Sharpe | Annual Return | Max DD | Win Rate | Total Trades |")
        lines.append("|----------|--------|---------------|--------|----------|--------------|")
        
        for name, sharpe in result.sharpe_rankings:
            res = result.backtest_results[name]
            lines.append(
                f"| {name:20s} | {res.sharpe_ratio:6.2f} | "
                f"{res.annual_return:6.1f}% | {res.max_drawdown_pct:6.1f}% | "
                f"{res.win_rate:5.1f}% | {res.total_trades:4d} |"
            )
        
        lines.append("")
        
        # Regime Analysis
        if result.high_vix_results or result.low_vix_results:
            lines.append("## Regime-Specific Performance")
            lines.append("")
            
            if result.high_vix_results:
                lines.append("### High VIX Regime")
                lines.append("")
                lines.append("| Strategy | Sharpe | Annual Return | Max DD |")
                lines.append("|----------|--------|---------------|--------|")
                
                for name, res in result.high_vix_results.items():
                    lines.append(
                        f"| {name:20s} | {res.sharpe_ratio:6.2f} | "
                        f"{res.annual_return:6.1f}% | {res.max_drawdown_pct:6.1f}% |"
                    )
                
                lines.append("")
            
            if result.low_vix_results:
                lines.append("### Low VIX Regime")
                lines.append("")
                lines.append("| Strategy | Sharpe | Annual Return | Max DD |")
                lines.append("|----------|--------|---------------|--------|")
                
                for name, res in result.low_vix_results.items():
                    lines.append(
                        f"| {name:20s} | {res.sharpe_ratio:6.2f} | "
                        f"{res.annual_return:6.1f}% | {res.max_drawdown_pct:6.1f}% |"
                    )
                
                lines.append("")
        
        # Statistical Significance
        if result.statistical_significance:
            lines.append("## Statistical Significance")
            lines.append("")
            lines.append("| Comparison | p-value | Significant | Effect Size |")
            lines.append("|------------|---------|-------------|-------------|")
            
            for pair, stats_result in result.statistical_significance.items():
                sig_marker = "✓" if stats_result['significant'] else "✗"
                lines.append(
                    f"| {pair:30s} | {stats_result['p_value']:7.4f} | "
                    f"{sig_marker:11s} | {stats_result['effect_size']:11s} |"
                )
            
            lines.append("")
        
        # Cost Sensitivity
        if result.cost_sensitivity:
            lines.append("## Transaction Cost Sensitivity")
            lines.append("")
            lines.append("Sharpe ratio under different cost assumptions:")
            lines.append("")
            lines.append("| Strategy | No Slippage | With Slippage | Degradation |")
            lines.append("|----------|-------------|---------------|-------------|")
            
            for name, costs in result.cost_sensitivity.items():
                no_cost = costs.get('none', 0)
                with_cost = costs.get('volatility', 0)
                degradation = ((no_cost - with_cost) / no_cost * 100) if no_cost > 0 else 0
                
                lines.append(
                    f"| {name:20s} | {no_cost:11.2f} | {with_cost:13.2f} | "
                    f"{degradation:10.1f}% |"
                )
            
            lines.append("")
        
        # Research Validation
        lines.append("## Research Validation")
        lines.append("")
        lines.append("Performance vs academic benchmarks:")
        lines.append("")
        lines.append("| Strategy | Actual Sharpe | Research Sharpe | % of Target | Meets Threshold |")
        lines.append("|----------|---------------|-----------------|-------------|-----------------|")
        
        for name, res in result.backtest_results.items():
            research_sharpe = VALIDATION.get(name, {}).get('research_sharpe', 0)
            min_sharpe = VALIDATION.get(name, {}).get('min_sharpe', 0)
            meets = "✓" if res.sharpe_ratio >= min_sharpe else "✗"
            
            lines.append(
                f"| {name:20s} | {res.sharpe_ratio:13.2f} | {research_sharpe:15.2f} | "
                f"{res.vs_research_pct:10.1f}% | {meets:15s} |"
            )
        
        lines.append("")
        
        # Detailed Metrics
        lines.append("## Detailed Metrics")
        lines.append("")
        
        for name, res in result.backtest_results.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- **Total Return:** {res.total_return:.2f}%")
            lines.append(f"- **Annual Return:** {res.annual_return:.2f}%")
            lines.append(f"- **Sharpe Ratio:** {res.sharpe_ratio:.2f}")
            lines.append(f"- **Sortino Ratio:** {res.sortino_ratio:.2f}")
            lines.append(f"- **Max Drawdown:** {res.max_drawdown_pct:.2f}%")
            lines.append(f"- **Volatility:** {res.volatility:.2f}%")
            lines.append(f"- **Total Trades:** {res.total_trades}")
            lines.append(f"- **Win Rate:** {res.win_rate:.1f}%")
            lines.append(f"- **Profit Factor:** {res.profit_factor:.2f}")
            lines.append(f"- **Avg Trade P&L:** ${res.avg_trade_pnl:.2f}")
            lines.append("")
        
        # Join and optionally save
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON FRAMEWORK")
    print("=" * 60)
    
    comp = StrategyComparison()
    
    # Run comparison
    results = comp.compare_all()
    
    # Generate report
    report_path = DIRS.get('backtests', Path('./research/backtests')) / f"comparison_{results.run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = comp.generate_report(results, report_path)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nReport saved to: {report_path}")
    print(f"\nBest Overall: {results.best_overall}")
    print(f"Best High VIX: {results.best_high_vix}")
    print(f"Best Low VIX: {results.best_low_vix}")
