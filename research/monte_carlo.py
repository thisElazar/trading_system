"""
Monte Carlo Robustness Testing
================================
Test strategy robustness through bootstrap simulation and parameter perturbation.

Features:
- Bootstrap historical returns (10,000 simulations)
- Parameter perturbation (costs, timing, slippage)
- Trade randomization (entry/exit timing variation)
- Confidence intervals for all metrics
- Stress testing under various scenarios
- Probability distributions of outcomes

Usage:
    from research.monte_carlo import MonteCarloSimulator
    
    mc = MonteCarloSimulator()
    results = mc.run_simulation(strategy_name='vol_managed_momentum', n_simulations=1000)
    mc.generate_report(results)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from config import VALIDATION, DIRS
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
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    strategy: str
    timestamp: str
    n_simulations: int
    
    # Distribution of outcomes
    sharpe_distribution: List[float] = field(default_factory=list)
    return_distribution: List[float] = field(default_factory=list)
    drawdown_distribution: List[float] = field(default_factory=list)
    
    # Confidence intervals
    sharpe_ci_95: Tuple[float, float] = (0, 0)
    sharpe_ci_68: Tuple[float, float] = (0, 0)
    return_ci_95: Tuple[float, float] = (0, 0)
    drawdown_ci_95: Tuple[float, float] = (0, 0)
    
    # Probabilities
    prob_sharpe_above_1: float = 0.0
    prob_sharpe_above_threshold: float = 0.0
    prob_positive_return: float = 0.0
    prob_drawdown_below_20: float = 0.0
    
    # Summary statistics
    median_sharpe: float = 0.0
    median_return: float = 0.0
    median_drawdown: float = 0.0
    worst_case_sharpe: float = 0.0
    worst_case_return: float = 0.0
    worst_case_drawdown: float = 0.0
    
    # Cost sensitivity
    cost_impact: Dict[str, float] = field(default_factory=dict)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.
    
    Tests:
    1. Bootstrap - Resample historical returns to generate alternate histories
    2. Parameter perturbation - Vary cost assumptions
    3. Trade randomization - Vary entry/exit timing
    4. Scenario stress testing - Test under extreme conditions
    """
    
    def __init__(self, data_manager: CachedDataManager = None, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.data_manager = data_manager if data_manager is not None else CachedDataManager()
        self.db = get_db()
        
        # Available strategies - all 8 in portfolio (gap_fill excluded - needs intraday)
        self.strategies = {
            'vol_managed_momentum': VolManagedMomentumStrategy(),
            'quality_small_cap_value': QualitySmallCapValueStrategy(),
            'factor_momentum': FactorMomentumStrategy(),
            'pairs_trading': PairsTradingStrategy(),
            'relative_volume_breakout': RelativeVolumeBreakout(),
            'vix_regime_rotation': VIXRegimeRotationStrategy(),
            'sector_rotation': SectorRotationStrategy(),
            'mean_reversion': MeanReversionStrategy(),
        }
    
    def bootstrap_returns(self,
                          equity_curve: List[float],
                          n_samples: int = 1000) -> List[List[float]]:
        """
        Bootstrap resample from historical returns.
        
        Args:
            equity_curve: Original equity curve
            n_samples: Number of bootstrap samples
            
        Returns:
            List of resampled equity curves
        """
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        bootstrapped_curves = []
        
        for _ in range(n_samples):
            # Resample returns with replacement
            sampled_returns = returns.sample(n=len(returns), replace=True)
            
            # Reconstruct equity curve
            sampled_equity = [self.initial_capital]
            for ret in sampled_returns:
                sampled_equity.append(sampled_equity[-1] * (1 + ret))
            
            bootstrapped_curves.append(sampled_equity)
        
        return bootstrapped_curves
    
    def perturb_costs(self,
                      base_slippage: float = 0.001) -> List[float]:
        """
        Generate range of cost assumptions to test.
        
        Args:
            base_slippage: Base slippage percentage
            
        Returns:
            List of slippage values to test
        """
        # Test range from -50% to +100% of base
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        return [base_slippage * m for m in multipliers]
    
    def randomize_timing(self,
                         data: Dict[str, pd.DataFrame],
                         max_shift_minutes: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Randomly shift entry/exit timing for all trades.
        
        Simulates realistic execution delays and timing variations.
        
        Args:
            data: Original market data
            max_shift_minutes: Maximum time shift in minutes
            
        Returns:
            Data with randomized timing
        """
        # For daily data, we'll shift entry signals by 0-3 bars
        max_shift_bars = 3
        randomized = {}
        
        for symbol, df in data.items():
            df_copy = df.copy()
            
            # Randomly shift price data slightly
            shift = np.random.randint(-max_shift_bars, max_shift_bars + 1)
            
            if shift != 0:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_copy.columns:
                        df_copy[col] = df_copy[col].shift(shift)
                
                df_copy = df_copy.dropna()
            
            randomized[symbol] = df_copy
        
        return randomized
    
    def calculate_metrics_from_equity(self, equity_curve: List[float]) -> Dict[str, float]:
        """Calculate performance metrics from equity curve."""
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'sharpe': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        # Sharpe
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        # Total return
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100
        
        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': abs(max_drawdown)
        }
    
    def run_bootstrap_simulation(self,
                                  strategy_name: str,
                                  data: Dict[str, pd.DataFrame],
                                  vix_data: pd.DataFrame = None,
                                  n_simulations: int = 1000) -> MonteCarloResult:
        """
        Run bootstrap Monte Carlo simulation.
        
        Args:
            strategy_name: Strategy to test
            data: Historical market data
            vix_data: VIX data
            n_simulations: Number of bootstrap samples
            
        Returns:
            MonteCarloResult with distributions and confidence intervals
        """
        logger.info(f"Running Monte Carlo simulation: {strategy_name}")
        logger.info(f"  Bootstrap samples: {n_simulations}")
        
        # Run base backtest to get original equity curve
        strategy = self.strategies[strategy_name]
        backtester = Backtester(self.initial_capital)
        
        base_result = backtester.run(strategy, data, vix_data=vix_data)
        
        if not base_result.equity_curve:
            raise ValueError("No equity curve from base backtest")
        
        # Bootstrap simulation
        logger.info("  Bootstrapping returns...")
        bootstrapped_curves = self.bootstrap_returns(base_result.equity_curve, n_simulations)
        
        # Calculate metrics for each bootstrapped curve
        sharpe_dist = []
        return_dist = []
        drawdown_dist = []
        
        for i, curve in enumerate(bootstrapped_curves):
            if i % 100 == 0:
                logger.info(f"    Processed {i}/{n_simulations} simulations...")
            
            metrics = self.calculate_metrics_from_equity(curve)
            sharpe_dist.append(metrics['sharpe'])
            return_dist.append(metrics['total_return'])
            drawdown_dist.append(metrics['max_drawdown'])
        
        # Calculate confidence intervals
        sharpe_ci_95 = (np.percentile(sharpe_dist, 2.5), np.percentile(sharpe_dist, 97.5))
        sharpe_ci_68 = (np.percentile(sharpe_dist, 16), np.percentile(sharpe_dist, 84))
        return_ci_95 = (np.percentile(return_dist, 2.5), np.percentile(return_dist, 97.5))
        drawdown_ci_95 = (np.percentile(drawdown_dist, 2.5), np.percentile(drawdown_dist, 97.5))
        
        # Calculate probabilities
        min_sharpe = VALIDATION.get(strategy_name, {}).get('min_sharpe', 0.5)
        
        prob_sharpe_above_1 = sum(1 for s in sharpe_dist if s >= 1.0) / len(sharpe_dist)
        prob_sharpe_above_threshold = sum(1 for s in sharpe_dist if s >= min_sharpe) / len(sharpe_dist)
        prob_positive_return = sum(1 for r in return_dist if r > 0) / len(return_dist)
        prob_drawdown_below_20 = sum(1 for d in drawdown_dist if d < 20) / len(drawdown_dist)
        
        # Summary statistics
        result = MonteCarloResult(
            strategy=strategy_name,
            timestamp=datetime.now().isoformat(),
            n_simulations=n_simulations,
            sharpe_distribution=sharpe_dist,
            return_distribution=return_dist,
            drawdown_distribution=drawdown_dist,
            sharpe_ci_95=sharpe_ci_95,
            sharpe_ci_68=sharpe_ci_68,
            return_ci_95=return_ci_95,
            drawdown_ci_95=drawdown_ci_95,
            prob_sharpe_above_1=prob_sharpe_above_1,
            prob_sharpe_above_threshold=prob_sharpe_above_threshold,
            prob_positive_return=prob_positive_return,
            prob_drawdown_below_20=prob_drawdown_below_20,
            median_sharpe=np.median(sharpe_dist),
            median_return=np.median(return_dist),
            median_drawdown=np.median(drawdown_dist),
            worst_case_sharpe=np.percentile(sharpe_dist, 5),
            worst_case_return=np.percentile(return_dist, 5),
            worst_case_drawdown=np.percentile(drawdown_dist, 95)
        )
        
        logger.info(f"  Bootstrap complete!")
        logger.info(f"    Median Sharpe: {result.median_sharpe:.2f}")
        logger.info(f"    95% CI: [{sharpe_ci_95[0]:.2f}, {sharpe_ci_95[1]:.2f}]")
        
        return result
    
    def run_cost_sensitivity(self,
                             strategy_name: str,
                             data: Dict[str, pd.DataFrame],
                             vix_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Test sensitivity to transaction cost assumptions.
        
        Returns:
            Dict mapping cost level to resulting Sharpe ratio
        """
        logger.info(f"Running cost sensitivity analysis: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        cost_results = {}
        
        # Test different slippage assumptions
        slippage_levels = {
            'optimistic': 'none',
            'realistic': 'volatility',
        }
        
        for level, slippage_model in slippage_levels.items():
            backtester = Backtester(
                initial_capital=self.initial_capital,
                slippage_model=slippage_model
            )
            
            result = backtester.run(strategy, data, vix_data=vix_data)
            cost_results[level] = result.sharpe_ratio
            
            logger.info(f"  {level:12s}: Sharpe = {result.sharpe_ratio:.2f}")
        
        return cost_results
    
    def run_timing_randomization(self,
                                  strategy_name: str,
                                  data: Dict[str, pd.DataFrame],
                                  vix_data: pd.DataFrame = None,
                                  n_trials: int = 100) -> List[float]:
        """
        Test robustness to entry/exit timing variations.
        
        Returns:
            List of Sharpe ratios from randomized timing
        """
        logger.info(f"Running timing randomization: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        timing_results = []
        
        for i in range(n_trials):
            if i % 20 == 0:
                logger.info(f"  Trial {i}/{n_trials}...")
            
            # Randomize timing
            randomized_data = self.randomize_timing(data)
            
            # Run backtest
            backtester = Backtester(self.initial_capital)
            result = backtester.run(strategy, randomized_data, vix_data=vix_data)
            
            timing_results.append(result.sharpe_ratio)
        
        mean_sharpe = np.mean(timing_results)
        std_sharpe = np.std(timing_results)
        
        logger.info(f"  Timing sensitivity: Mean Sharpe = {mean_sharpe:.2f} ± {std_sharpe:.2f}")
        
        return timing_results
    
    def run_simulation(self,
                       strategy_name: str,
                       symbols: List[str] = None,
                       n_simulations: int = 1000) -> MonteCarloResult:
        """
        Run complete Monte Carlo simulation.
        
        Args:
            strategy_name: Strategy to test
            symbols: Optional list of symbols
            n_simulations: Number of bootstrap samples
            
        Returns:
            MonteCarloResult with full analysis
        """
        logger.info("=" * 60)
        logger.info(f"MONTE CARLO SIMULATION: {strategy_name}")
        logger.info("=" * 60)
        
        # Load data
        if not self.data_manager.cache:
            self.data_manager.load_all()
        
        if symbols:
            data = {s: self.data_manager.get_bars(s) for s in symbols
                   if s in self.data_manager.cache}
        else:
            data = {s: df.copy() for s, df in self.data_manager.cache.items()}
        
        # Load VIX
        vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
        vix_data = None
        if vix_path.exists():
            vix_data = pd.read_parquet(vix_path)
        
        # Run bootstrap simulation
        result = self.run_bootstrap_simulation(
            strategy_name, data, vix_data, n_simulations
        )
        
        # Add cost sensitivity
        cost_sensitivity = self.run_cost_sensitivity(strategy_name, data, vix_data)
        result.cost_impact = cost_sensitivity
        
        logger.info("\n" + "=" * 60)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 60)
        
        return result
    
    def generate_report(self, result: MonteCarloResult, output_path: Path = None) -> str:
        """Generate Monte Carlo analysis report."""
        lines = []
        
        lines.append("# Monte Carlo Robustness Analysis")
        lines.append(f"\n**Strategy:** {result.strategy}")
        lines.append(f"**Timestamp:** {result.timestamp}")
        lines.append(f"**Simulations:** {result.n_simulations:,}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"**Median Sharpe Ratio:** {result.median_sharpe:.2f}")
        lines.append(f"**68% Confidence Interval:** [{result.sharpe_ci_68[0]:.2f}, {result.sharpe_ci_68[1]:.2f}]")
        lines.append(f"**95% Confidence Interval:** [{result.sharpe_ci_95[0]:.2f}, {result.sharpe_ci_95[1]:.2f}]")
        lines.append("")
        lines.append(f"**Probability of Sharpe > 1.0:** {result.prob_sharpe_above_1:.1%}")
        lines.append(f"**Probability of Positive Returns:** {result.prob_positive_return:.1%}")
        lines.append(f"**Probability of Max DD < 20%:** {result.prob_drawdown_below_20:.1%}")
        lines.append("")
        
        # Distribution Statistics
        lines.append("## Distribution Statistics")
        lines.append("")
        lines.append("### Sharpe Ratio")
        lines.append("")
        lines.append(f"- **Median:** {result.median_sharpe:.2f}")
        lines.append(f"- **Mean:** {np.mean(result.sharpe_distribution):.2f}")
        lines.append(f"- **Std Dev:** {np.std(result.sharpe_distribution):.2f}")
        lines.append(f"- **5th Percentile (Worst Case):** {result.worst_case_sharpe:.2f}")
        lines.append(f"- **95th Percentile:** {np.percentile(result.sharpe_distribution, 95):.2f}")
        lines.append("")
        
        lines.append("### Total Return")
        lines.append("")
        lines.append(f"- **Median:** {result.median_return:.1f}%")
        lines.append(f"- **95% CI:** [{result.return_ci_95[0]:.1f}%, {result.return_ci_95[1]:.1f}%]")
        lines.append(f"- **5th Percentile (Worst Case):** {result.worst_case_return:.1f}%")
        lines.append("")
        
        lines.append("### Max Drawdown")
        lines.append("")
        lines.append(f"- **Median:** {result.median_drawdown:.1f}%")
        lines.append(f"- **95% CI:** [{result.drawdown_ci_95[0]:.1f}%, {result.drawdown_ci_95[1]:.1f}%]")
        lines.append(f"- **95th Percentile (Worst Case):** {result.worst_case_drawdown:.1f}%")
        lines.append("")
        
        # Probabilities
        lines.append("## Probability Analysis")
        lines.append("")
        min_sharpe = VALIDATION.get(result.strategy, {}).get('min_sharpe', 0.5)
        lines.append(f"- **P(Sharpe ≥ {min_sharpe}):** {result.prob_sharpe_above_threshold:.1%}")
        lines.append(f"- **P(Sharpe ≥ 1.0):** {result.prob_sharpe_above_1:.1%}")
        lines.append(f"- **P(Positive Returns):** {result.prob_positive_return:.1%}")
        lines.append(f"- **P(Max DD < 20%):** {result.prob_drawdown_below_20:.1%}")
        lines.append("")
        
        # Cost Sensitivity
        if result.cost_impact:
            lines.append("## Transaction Cost Impact")
            lines.append("")
            lines.append("| Cost Assumption | Sharpe Ratio |")
            lines.append("|-----------------|--------------|")
            
            for level, sharpe in result.cost_impact.items():
                lines.append(f"| {level:15s} | {sharpe:12.2f} |")
            
            if 'optimistic' in result.cost_impact and 'realistic' in result.cost_impact:
                degradation = result.cost_impact['optimistic'] - result.cost_impact['realistic']
                degradation_pct = (degradation / result.cost_impact['optimistic'] * 100) if result.cost_impact['optimistic'] > 0 else 0
                lines.append("")
                lines.append(f"**Degradation from costs:** {degradation_pct:.1f}%")
            
            lines.append("")
        
        # Interpretation
        lines.append("## Interpretation")
        lines.append("")
        
        if result.prob_sharpe_above_1 > 0.8:
            lines.append("✅ **ROBUST** - Strategy shows strong probability of achieving Sharpe > 1.0")
        elif result.prob_sharpe_above_1 > 0.5:
            lines.append("⚠️ **MODERATE** - Strategy has reasonable chance of success but significant uncertainty")
        else:
            lines.append("❌ **WEAK** - Strategy shows low probability of achieving target performance")
        
        lines.append("")
        
        if result.worst_case_sharpe > 0:
            lines.append("✅ Even in worst-case scenarios, strategy maintains positive risk-adjusted returns")
        else:
            lines.append("⚠️ Worst-case scenarios show negative risk-adjusted returns")
        
        lines.append("")
        
        # Join and save
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("MONTE CARLO ROBUSTNESS TESTING")
    print("=" * 60)
    
    mc = MonteCarloSimulator()
    
    # Run simulation
    result = mc.run_simulation(
        strategy_name='vol_managed_momentum',
        n_simulations=1000
    )
    
    # Generate report
    report_path = DIRS.get('backtests', Path('./research/backtests')) / f"monte_carlo_{result.strategy}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = mc.generate_report(result, report_path)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nReport: {report_path}")
    print(f"\nMedian Sharpe: {result.median_sharpe:.2f}")
    print(f"95% CI: [{result.sharpe_ci_95[0]:.2f}, {result.sharpe_ci_95[1]:.2f}]")
    print(f"P(Sharpe > 1.0): {result.prob_sharpe_above_1:.1%}")
