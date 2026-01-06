"""
Research Pipeline Runner
========================
Master script to orchestrate all research tools systematically.

Runs:
1. Strategy Comparison - Head-to-head testing
2. Monte Carlo Simulation - Robustness validation
3. Parameter Optimization - Find optimal settings

Usage:
    # Run full research pipeline
    python scripts/run_research.py --full
    
    # Run specific components
    python scripts/run_research.py --comparison
    python scripts/run_research.py --monte-carlo
    python scripts/run_research.py --optimize
    
    # Quick test (fewer simulations)
    python scripts/run_research.py --full --quick
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS
from data.cached_data_manager import CachedDataManager
from research.strategy_comparison import StrategyComparison
from research.monte_carlo import MonteCarloSimulator
from research.parameter_optimizer import ParameterOptimizer
from strategies import (
    VolManagedMomentumStrategy,
    VIXRegimeRotationStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    RelativeVolumeBreakout,
    SectorRotationStrategy,
    QualitySmallCapValueStrategy,
    FactorMomentumStrategy,
)

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """
    Master research pipeline that runs all research tools.
    
    Generates comprehensive reports on:
    - Which strategies work best overall
    - Which strategies work best in each regime
    - How robust each strategy is
    - What optimal parameters are
    """
    
    def __init__(self, output_dir: Path = None, quick_mode: bool = False):
        """
        Initialize research pipeline.
        
        Args:
            output_dir: Directory for outputs
            quick_mode: If True, use fewer simulations for faster results
        """
        self.output_dir = output_dir or DIRS.get('backtests', Path('./research/backtests'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quick_mode = quick_mode
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Components
        self.data_manager = CachedDataManager()
        self.comparison = StrategyComparison()
        self.monte_carlo = MonteCarloSimulator(self.data_manager)
        self.optimizer = ParameterOptimizer(self.data_manager)
        
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
            # Note: gap_fill requires intraday data - excluded from daily research
            # 'gap_fill': GapFillStrategy(),
        }
        
        logger.info("="*60)
        logger.info("RESEARCH PIPELINE INITIALIZED")
        logger.info("="*60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Quick Mode: {self.quick_mode}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info("="*60)
    
    def run_full_pipeline(self):
        """Run complete research pipeline."""
        logger.info("\n" + "="*60)
        logger.info("STARTING FULL RESEARCH PIPELINE")
        logger.info("="*60)
        
        results = {}
        
        # 1. Strategy Comparison
        logger.info("\n### PHASE 1: STRATEGY COMPARISON ###\n")
        comparison_results = self.run_strategy_comparison()
        results['comparison'] = comparison_results
        
        # 2. Monte Carlo for each strategy
        logger.info("\n### PHASE 2: MONTE CARLO ROBUSTNESS ###\n")
        mc_results = self.run_monte_carlo_analysis()
        results['monte_carlo'] = mc_results
        
        # 3. Parameter Optimization
        logger.info("\n### PHASE 3: PARAMETER OPTIMIZATION ###\n")
        opt_results = self.run_parameter_optimization()
        results['optimization'] = opt_results
        
        # 4. Generate Master Report
        logger.info("\n### PHASE 4: MASTER REPORT ###\n")
        self.generate_master_report(results)
        
        logger.info("\n" + "="*60)
        logger.info("RESEARCH PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved to: {self.output_dir}")
        
        return results
    
    def run_strategy_comparison(self):
        """Run strategy comparison analysis."""
        logger.info("Running strategy comparison...")
        
        try:
            results = self.comparison.compare_all()
            
            # Generate report
            report_path = self.output_dir / f"01_comparison_{self.run_id}.md"
            self.comparison.generate_report(results, report_path)
            
            logger.info(f"✓ Comparison complete: {report_path}")
            return results
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}", exc_info=True)
            return None
    
    def run_monte_carlo_analysis(self):
        """Run Monte Carlo analysis on all strategies."""
        logger.info("Running Monte Carlo analysis...")
        
        # Load data
        if not self.data_manager.cache:
            self.data_manager.load_all()
        
        # Get top symbols by liquidity
        metadata = self.data_manager.get_all_metadata()
        top_symbols = sorted(
            metadata.items(),
            key=lambda x: x[1].get('dollar_volume', 0),
            reverse=True
        )[:100]  # Top 100 liquid symbols
        
        data = {sym: self.data_manager.get_bars(sym) for sym, _ in top_symbols}
        
        # Load VIX
        vix_path = DIRS.get('vix', self.data_manager.data_dir.parent / 'vix') / 'vix.parquet'
        vix_data = None
        if vix_path.exists():
            import pandas as pd
            vix_data = pd.read_parquet(vix_path)
        
        mc_results = {}
        
        # Determine simulation count
        n_sims = 100 if self.quick_mode else 1000
        
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"\nMonte Carlo: {strategy_name}")
            
            try:
                # Run bootstrap simulation
                result = self.monte_carlo.run_simulation(
                    strategy_name=strategy.name,
                    n_simulations=n_sims
                )
                
                # Generate report
                report_path = self.output_dir / f"02_monte_carlo_{strategy_name}_{self.run_id}.md"
                self.monte_carlo.generate_report(result, report_path)
                
                mc_results[strategy_name] = result
                logger.info(f"✓ {strategy_name}: Sharpe 95% CI [{result.sharpe_ci_95[0]:.2f}, {result.sharpe_ci_95[1]:.2f}]")
                
            except Exception as e:
                logger.error(f"Monte Carlo failed for {strategy_name}: {e}", exc_info=True)
        
        return mc_results
    
    def run_parameter_optimization(self):
        """Run parameter optimization for all strategies."""
        logger.info("Running parameter optimization...")
        
        # Load data
        if not self.data_manager.cache:
            self.data_manager.load_all()
        
        metadata = self.data_manager.get_all_metadata()
        top_symbols = sorted(
            metadata.items(),
            key=lambda x: x[1].get('dollar_volume', 0),
            reverse=True
        )[:100]
        
        data = {sym: self.data_manager.get_bars(sym) for sym, _ in top_symbols}
        
        opt_results = {}
        
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"\nOptimizing: {strategy_name}")
            
            try:
                # Define parameter grids based on strategy
                if strategy_name == 'vol_managed_momentum':
                    param_grid = {
                        'formation_period': [63, 126, 189, 252],  # 3-12 months
                        'vol_lookback': [14, 21, 42],
                        'target_vol': [0.15, 0.20, 0.25],
                    }
                elif strategy_name == 'vix_regime_rotation':
                    param_grid = {
                        'low_threshold': [12, 14, 16, 18],
                        'high_threshold': [23, 25, 28, 30],
                        'exposure_high_vix': [0.3, 0.5, 0.7],
                    }
                elif strategy_name == 'quality_small_cap_value':
                    param_grid = {
                        'min_roa': [-0.05, 0.0, 0.05],
                        'max_debt_to_equity': [0.5, 1.0, 1.5],
                        'value_percentile': [0.15, 0.20, 0.25, 0.30],
                    }
                elif strategy_name == 'factor_momentum':
                    param_grid = {
                        'formation_period_long': [126, 189, 252],
                        'skip_period': [0, 21, 42],
                        'max_factor_weight': [0.30, 0.40, 0.50],
                    }
                elif strategy_name == 'pairs_trading':
                    param_grid = {
                        'zscore_entry': [1.5, 2.0, 2.5],
                        'zscore_exit': [0.3, 0.5, 0.75],
                        'lookback': [60, 90, 120],
                    }
                elif strategy_name == 'sector_rotation':
                    param_grid = {
                        'momentum_period': [21, 63, 126],
                        'low_vix_threshold': [15, 18, 20],
                        'high_vix_threshold': [23, 25, 28],
                    }
                elif strategy_name == 'relative_volume_breakout':
                    param_grid = {
                        'min_relative_volume': [1.5, 2.0, 2.5, 3.0],
                        'atr_stop_mult': [1.0, 1.5, 2.0],
                        'atr_target_mult': [1.5, 2.0, 3.0],
                    }
                elif strategy_name == 'mean_reversion':
                    param_grid = {
                        'bb_period': [14, 20, 30],
                        'bb_std': [1.5, 2.0, 2.5],
                        'rsi_oversold': [25, 30, 35],
                    }
                else:
                    logger.warning(f"No parameter grid defined for {strategy_name}")
                    continue
                
                # Parameter optimization is handled by the GA/GP discovery system
                # For manual parameter tuning, use research/parameter_optimizer.py directly
                results = {
                    'best_params': None,
                    'best_sharpe': 0.0,
                    'results': []
                }
                logger.info(f"Parameter optimization skipped - use GA discovery for {strategy_name}")
                
                # Skip report generation when optimization is skipped (dict vs dataclass)
                # report_path = self.output_dir / f"03_optimization_{strategy_name}_{self.run_id}.md"
                # self.optimizer.generate_report(results, report_path)
                
                opt_results[strategy_name] = results
                
                if results['best_params']:
                    logger.info(f"✓ {strategy_name}: Best Sharpe = {results['best_sharpe']:.2f}")
                    logger.info(f"  Params: {results['best_params']}")
                
            except Exception as e:
                logger.error(f"Optimization failed for {strategy_name}: {e}", exc_info=True)
        
        return opt_results
    
    def generate_master_report(self, results: dict):
        """Generate comprehensive master report."""
        logger.info("Generating master report...")
        
        lines = []
        
        # Header
        lines.append("# Master Research Report")
        lines.append(f"\n**Run ID:** `{self.run_id}`")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Mode:** {'Quick' if self.quick_mode else 'Full'}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        
        # Strategy comparison summary
        if results.get('comparison'):
            comp = results['comparison']
            lines.append("### Best Strategies")
            lines.append(f"- **Overall:** {comp.best_overall}")
            lines.append(f"- **High VIX:** {comp.best_high_vix}")
            lines.append(f"- **Low VIX:** {comp.best_low_vix}")
            lines.append("")
        
        # Monte Carlo summary
        if results.get('monte_carlo'):
            lines.append("### Robustness Assessment")
            lines.append("")
            for strategy_name, mc_result in results['monte_carlo'].items():
                prob_good = mc_result.prob_sharpe_above_1
                status = "✅ ROBUST" if prob_good > 0.8 else "⚠️ MODERATE" if prob_good > 0.5 else "❌ WEAK"
                lines.append(f"- **{strategy_name}:** {status} (P(Sharpe>1.0) = {prob_good:.1%})")
            lines.append("")
        
        # Optimization summary
        if results.get('optimization'):
            lines.append("### Optimal Parameters")
            lines.append("")
            for strategy_name, opt_result in results['optimization'].items():
                if opt_result.get('best_params'):
                    lines.append(f"**{strategy_name}:**")
                    for param, value in opt_result['best_params'].items():
                        lines.append(f"- {param}: {value}")
                    lines.append(f"- Sharpe: {opt_result['best_sharpe']:.2f}")
                    lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        
        # Determine recommendation based on results
        if results.get('comparison') and results.get('monte_carlo'):
            comp = results['comparison']
            mc = results['monte_carlo']
            
            best_strategy = comp.best_overall
            if best_strategy in mc:
                mc_result = mc[best_strategy]
                
                if mc_result.prob_sharpe_above_1 > 0.8:
                    lines.append(f"✅ **RECOMMENDED:** Deploy {best_strategy}")
                    lines.append(f"- High confidence (>80%) in sustained performance")
                    lines.append(f"- Sharpe 95% CI: [{mc_result.sharpe_ci_95[0]:.2f}, {mc_result.sharpe_ci_95[1]:.2f}]")
                elif mc_result.prob_sharpe_above_1 > 0.5:
                    lines.append(f"⚠️ **PROCEED WITH CAUTION:** {best_strategy}")
                    lines.append(f"- Moderate confidence (50-80%) in performance")
                    lines.append(f"- Consider smaller position sizes initially")
                else:
                    lines.append(f"❌ **NOT RECOMMENDED:** Insufficient confidence in any strategy")
                    lines.append(f"- Continue research and testing")
                    lines.append(f"- Consider additional strategies or parameter tuning")
        
        lines.append("")
        
        # Links to detailed reports
        lines.append("## Detailed Reports")
        lines.append("")
        lines.append("1. [Strategy Comparison](./01_comparison_{}.md)".format(self.run_id))
        
        for strategy_name in self.strategies.keys():
            lines.append(f"2. [Monte Carlo - {strategy_name}](./02_monte_carlo_{strategy_name}_{self.run_id}.md)")
            lines.append(f"3. [Optimization - {strategy_name}](./03_optimization_{strategy_name}_{self.run_id}.md)")
        
        lines.append("")
        
        # Save
        report_path = self.output_dir / f"00_MASTER_REPORT_{self.run_id}.md"
        report_path.write_text("\n".join(lines))
        
        logger.info(f"✓ Master report: {report_path}")
        
        # Print to console
        print("\n" + "="*60)
        print("MASTER RESEARCH REPORT")
        print("="*60)
        print("\n".join(lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run research pipeline")
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--comparison', action='store_true', help='Run strategy comparison only')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo only')
    parser.add_argument('--optimize', action='store_true', help='Run optimization only')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer simulations)')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'research_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # Create pipeline
    output_dir = Path(args.output) if args.output else None
    pipeline = ResearchPipeline(output_dir=output_dir, quick_mode=args.quick)
    
    # Run components
    if args.full:
        pipeline.run_full_pipeline()
    else:
        if args.comparison:
            pipeline.run_strategy_comparison()
        if args.monte_carlo:
            pipeline.run_monte_carlo_analysis()
        if args.optimize:
            pipeline.run_parameter_optimization()
        
        if not (args.comparison or args.monte_carlo or args.optimize):
            # Default: run full pipeline
            pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
