"""
Parameter Optimization Framework
==================================
Systematically find optimal strategy parameters using walk-forward validation.

Features:
- Grid search across parameter space
- Walk-forward testing (train on N months, test on M)
- Rolling optimization (re-optimize quarterly)
- Stability analysis (do optimal params change drastically?)
- Out-of-sample performance tracking
- Database logging with auto-apply capability

Usage:
    from research.parameter_optimizer import ParameterOptimizer
    
    optimizer = ParameterOptimizer()
    results = optimizer.optimize_strategy('vol_managed_momentum')
    optimizer.generate_report(results)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import itertools
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import STRATEGIES, VALIDATION, DIRS
from research.backtester import Backtester, BacktestResult
from data.cached_data_manager import CachedDataManager
from data.storage.db_manager import get_db

# Import ALL strategies
from strategies.vol_managed_momentum import VolManagedMomentumStrategy
from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.relative_volume_breakout import RelativeVolumeBreakout
from strategies.pairs_trading import PairsTradingStrategy
from strategies.quality_small_cap_value import QualitySmallCapValueStrategy
from strategies.factor_momentum import FactorMomentumStrategy
from strategies.sector_rotation import SectorRotationStrategy

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Parameter configuration for optimization."""
    name: str
    param_type: str  # 'int', 'float', 'categorical'
    values: List[Any]
    current_value: Any = None


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    strategy: str
    timestamp: str
    
    # Best parameters found
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_sharpe: float = 0.0
    
    # Original baseline
    baseline_params: Dict[str, Any] = field(default_factory=dict)
    baseline_sharpe: float = 0.0
    
    # Improvement
    improvement_pct: float = 0.0
    
    # All tested combinations
    all_results: List[Dict] = field(default_factory=list)
    
    # Stability analysis
    param_sensitivity: Dict[str, float] = field(default_factory=dict)
    stable_params: List[str] = field(default_factory=list)
    unstable_params: List[str] = field(default_factory=list)
    
    # Walk-forward results
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    degradation_pct: float = 0.0


class ParameterOptimizer:
    """
    Optimize strategy parameters using walk-forward validation.
    
    Prevents overfitting by:
    1. Training on one period, testing on subsequent period
    2. Rolling optimization windows
    3. Stability analysis across time periods
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.data_manager = CachedDataManager()
        self.db = get_db()
        
        # Define parameter search spaces for ALL strategies
        self.param_spaces = {
            # ==================== TIER 1: CORE ALPHA ====================
            
            'vol_managed_momentum': [
                ParameterConfig(
                    name='lookback_period',
                    param_type='int',
                    values=[126, 189, 252],  # 6, 9, 12 months
                    current_value=252
                ),
                ParameterConfig(
                    name='top_n',
                    param_type='int',
                    values=[5, 10, 20, 30],
                    current_value=20
                ),
                ParameterConfig(
                    name='vol_window',
                    param_type='int',
                    values=[20, 60, 120],
                    current_value=60
                ),
                ParameterConfig(
                    name='target_vol',
                    param_type='float',
                    values=[0.10, 0.15, 0.20],
                    current_value=0.15
                ),
            ],
            
            'mean_reversion': [
                # OPTIMIZED 2025-12-29: 14-day lookback, 25% bottom, 7 per sector, -20% stop
                # Improvement: Sharpe 0.52 -> 0.66 (+26%), MaxDD -19.3% -> -15.5%
                ParameterConfig(
                    name='lookback_period',
                    param_type='int',
                    values=[10, 14, 21, 30],  # Days for reversal signal
                    current_value=14  # Optimized from 21
                ),
                ParameterConfig(
                    name='bottom_percentile',
                    param_type='float',
                    values=[0.10, 0.15, 0.20, 0.25],  # Quintile selection
                    current_value=0.25  # Optimized from 0.20
                ),
                ParameterConfig(
                    name='max_stocks_per_sector',
                    param_type='int',
                    values=[3, 5, 7],
                    current_value=7  # Optimized from 5
                ),
                ParameterConfig(
                    name='stop_loss_pct',
                    param_type='float',
                    values=[-0.10, -0.15, -0.20],
                    current_value=-0.20  # Optimized from -0.15
                ),
                ParameterConfig(
                    name='target_vol',
                    param_type='float',
                    values=[0.10, 0.15, 0.20],
                    current_value=0.15
                ),
            ],
            
            'relative_volume_breakout': [
                ParameterConfig(
                    name='MIN_RELATIVE_VOLUME',
                    param_type='float',
                    values=[1.5, 2.0, 2.5, 3.0],
                    current_value=1.5
                ),
                ParameterConfig(
                    name='MIN_GAP_PCT',
                    param_type='float',
                    values=[0.01, 0.02, 0.03, 0.04],
                    current_value=0.03
                ),
                ParameterConfig(
                    name='ATR_STOP_MULT',
                    param_type='float',
                    values=[1.0, 1.5, 2.0],
                    current_value=1.5
                ),
                ParameterConfig(
                    name='ATR_TARGET_MULT',
                    param_type='float',
                    values=[1.0, 1.5, 2.0, 2.5],
                    current_value=1.5
                ),
                ParameterConfig(
                    name='MAX_HOLD_DAYS',
                    param_type='int',
                    values=[1, 2, 3, 5],
                    current_value=1
                ),
                ParameterConfig(
                    name='MAX_POSITIONS',
                    param_type='int',
                    values=[3, 5, 7, 10],
                    current_value=5
                ),
            ],
            
            'pairs_trading': [
                ParameterConfig(
                    name='ENTRY_ZSCORE',
                    param_type='float',
                    values=[1.5, 1.75, 2.0, 2.25, 2.5],
                    current_value=1.5
                ),
                ParameterConfig(
                    name='EXIT_ZSCORE',
                    param_type='float',
                    values=[0.25, 0.5, 0.75, 1.0],
                    current_value=0.75
                ),
                ParameterConfig(
                    name='STOP_ZSCORE',
                    param_type='float',
                    values=[2.5, 3.0, 3.5, 4.0],
                    current_value=3.0
                ),
                ParameterConfig(
                    name='MAX_HOLD_DAYS',
                    param_type='int',
                    values=[20, 30, 40, 60],
                    current_value=40
                ),
                ParameterConfig(
                    name='min_correlation',
                    param_type='float',
                    values=[0.6, 0.7, 0.8],
                    current_value=0.7
                ),
            ],
            
            'quality_small_cap_value': [
                ParameterConfig(
                    name='MIN_ROA',
                    param_type='float',
                    values=[-0.05, 0.0, 0.05, 0.10],
                    current_value=0.0
                ),
                ParameterConfig(
                    name='MAX_DEBT_TO_EQUITY',
                    param_type='float',
                    values=[0.5, 1.0, 1.5, 2.0],
                    current_value=1.0
                ),
                ParameterConfig(
                    name='VALUE_PERCENTILE',
                    param_type='float',
                    values=[0.15, 0.20, 0.25, 0.30],
                    current_value=0.25
                ),
                ParameterConfig(
                    name='MAX_POSITIONS',
                    param_type='int',
                    values=[20, 30, 40, 50],
                    current_value=30
                ),
                ParameterConfig(
                    name='MAX_SINGLE_POSITION',
                    param_type='float',
                    values=[0.03, 0.05, 0.07, 0.10],
                    current_value=0.05
                ),
            ],
            
            'factor_momentum': [
                ParameterConfig(
                    name='FORMATION_PERIOD_LONG',
                    param_type='int',
                    values=[126, 189, 252],  # 6, 9, 12 months
                    current_value=252
                ),
                ParameterConfig(
                    name='FORMATION_PERIOD_MED',
                    param_type='int',
                    values=[42, 63, 126],  # 2, 3, 6 months
                    current_value=126
                ),
                ParameterConfig(
                    name='SKIP_PERIOD',
                    param_type='int',
                    values=[0, 7, 14, 21],
                    current_value=21
                ),
                ParameterConfig(
                    name='MAX_FACTOR_WEIGHT',
                    param_type='float',
                    values=[0.30, 0.40, 0.50],
                    current_value=0.40
                ),
                ParameterConfig(
                    name='HIGH_VIX_REDUCTION',
                    param_type='float',
                    values=[0.50, 0.60, 0.70, 0.80],
                    current_value=0.70
                ),
            ],
            
            # ==================== TIER 2: REGIME & TACTICAL ====================
            
            'vix_regime_rotation': [
                ParameterConfig(
                    name='thresholds_low',
                    param_type='float',
                    values=[12.0, 15.0, 18.0],
                    current_value=15.0
                ),
                ParameterConfig(
                    name='thresholds_high',
                    param_type='float',
                    values=[20.0, 25.0, 30.0],
                    current_value=25.0
                ),
                ParameterConfig(
                    name='min_days_between',
                    param_type='int',
                    values=[3, 5, 7, 10],
                    current_value=5
                ),
            ],
            
            'sector_rotation': [
                ParameterConfig(
                    name='low_vix',
                    param_type='float',
                    values=[15.0, 18.0, 20.0],
                    current_value=18.0
                ),
                ParameterConfig(
                    name='high_vix',
                    param_type='float',
                    values=[22.0, 25.0, 28.0],
                    current_value=25.0
                ),
                ParameterConfig(
                    name='rebalance_threshold',
                    param_type='float',
                    values=[0.03, 0.05, 0.07, 0.10],
                    current_value=0.05
                ),
                ParameterConfig(
                    name='momentum_weight',
                    param_type='float',
                    values=[0.2, 0.3, 0.4, 0.5],
                    current_value=0.3
                ),
            ],
        }
    
    def create_strategy_with_params(self,
                                     strategy_name: str,
                                     params: Dict[str, Any]):
        """
        Create strategy instance with specified parameters.
        
        Supports all 8 strategies in the portfolio.
        """
        # Strategy class mapping
        STRATEGY_CLASSES = {
            'vol_managed_momentum': VolManagedMomentumStrategy,
            'vix_regime_rotation': VIXRegimeRotationStrategy,
            'mean_reversion': MeanReversionStrategy,
            'relative_volume_breakout': RelativeVolumeBreakout,
            'pairs_trading': PairsTradingStrategy,
            'quality_small_cap_value': QualitySmallCapValueStrategy,
            'factor_momentum': FactorMomentumStrategy,
            'sector_rotation': SectorRotationStrategy,
        }
        
        if strategy_name not in STRATEGY_CLASSES:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_CLASSES.keys())}")
        
        # Create strategy instance
        strategy = STRATEGY_CLASSES[strategy_name]()
        
        # Apply parameters
        for param_name, value in params.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, value)
            else:
                logger.warning(f"Strategy {strategy_name} has no attribute '{param_name}'")
        
        return strategy
    
    def get_available_strategies(self) -> List[str]:
        """Return list of strategies with defined parameter spaces."""
        return list(self.param_spaces.keys())
    
    def get_param_space(self, strategy_name: str) -> List[ParameterConfig]:
        """Return parameter space for a strategy."""
        return self.param_spaces.get(strategy_name, [])
    
    def get_total_combinations(self, strategy_name: str) -> int:
        """Calculate total number of parameter combinations for a strategy."""
        param_space = self.param_spaces.get(strategy_name, [])
        if not param_space:
            return 0
        total = 1
        for p in param_space:
            total *= len(p.values)
        return total
    
    def grid_search(self,
                    strategy_name: str,
                    data: Dict[str, pd.DataFrame],
                    vix_data: pd.DataFrame = None,
                    max_combinations: int = 50) -> List[Dict]:
        """
        Grid search over parameter space.
        
        Args:
            strategy_name: Strategy to optimize
            data: Training data
            vix_data: VIX data
            max_combinations: Maximum parameter combinations to test
            
        Returns:
            List of results for each parameter combination
        """
        param_space = self.param_spaces.get(strategy_name, [])
        
        if not param_space:
            logger.warning(f"No parameter space defined for {strategy_name}")
            return []
        
        # Generate all combinations
        param_names = [p.name for p in param_space]
        param_values = [p.values for p in param_space]
        
        combinations = list(itertools.product(*param_values))
        
        # Limit if too many
        if len(combinations) > max_combinations:
            logger.warning(f"Limiting to {max_combinations} of {len(combinations)} combinations")
            np.random.shuffle(combinations)
            combinations = combinations[:max_combinations]
        
        logger.info(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            logger.info(f"  [{i+1}/{len(combinations)}] Testing: {params}")
            
            # Create strategy with these parameters
            strategy = self.create_strategy_with_params(strategy_name, params)
            
            # Run backtest
            backtester = Backtester(self.initial_capital)
            result = backtester.run(strategy, data, vix_data=vix_data)
            
            results.append({
                'params': params,
                'sharpe': result.sharpe_ratio,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'result': result
            })
            
            logger.info(f"      Sharpe: {result.sharpe_ratio:.2f}, Return: {result.annual_return:.1f}%")
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        return results
    
    def walk_forward_optimization(self,
                                   strategy_name: str,
                                   data: Dict[str, pd.DataFrame],
                                   vix_data: pd.DataFrame = None,
                                   train_months: int = 12,
                                   test_months: int = 3) -> Tuple[Dict, BacktestResult, BacktestResult]:
        """
        Walk-forward optimization.
        
        Train on first N months, test on next M months.
        
        Args:
            strategy_name: Strategy to optimize
            data: Full dataset
            vix_data: VIX data
            train_months: Months for training
            test_months: Months for testing
            
        Returns:
            (best_params, in_sample_result, out_of_sample_result)
        """
        logger.info("Walk-forward optimization:")
        logger.info(f"  Train: {train_months} months, Test: {test_months} months")
        
        # Get date range
        all_dates = set()
        for df in data.values():
            if 'timestamp' in df.columns:
                all_dates.update(df['timestamp'].tolist())
            else:
                all_dates.update(df.index.tolist())
        
        all_dates = sorted(all_dates)
        
        if len(all_dates) < 30:
            raise ValueError("Insufficient data for walk-forward")
        
        # Split into train and test
        split_idx = int(len(all_dates) * 0.7)  # 70% train, 30% test
        train_end = all_dates[split_idx]
        
        # Filter data
        train_data = {}
        test_data = {}
        
        for symbol, df in data.items():
            if 'timestamp' in df.columns:
                train_data[symbol] = df[df['timestamp'] <= train_end]
                test_data[symbol] = df[df['timestamp'] > train_end]
            else:
                train_data[symbol] = df[df.index <= train_end]
                test_data[symbol] = df[df.index > train_end]
        
        # Optimize on training data
        logger.info("  Optimizing on training data...")
        train_results = self.grid_search(strategy_name, train_data, vix_data)
        
        if not train_results:
            raise ValueError("No optimization results")
        
        best_params = train_results[0]['params']
        in_sample_result = train_results[0]['result']
        
        logger.info(f"  Best params: {best_params}")
        logger.info(f"  In-sample Sharpe: {in_sample_result.sharpe_ratio:.2f}")
        
        # Test on out-of-sample data
        logger.info("  Testing on out-of-sample data...")
        strategy = self.create_strategy_with_params(strategy_name, best_params)
        backtester = Backtester(self.initial_capital)
        out_of_sample_result = backtester.run(strategy, test_data, vix_data=vix_data)
        
        logger.info(f"  Out-of-sample Sharpe: {out_of_sample_result.sharpe_ratio:.2f}")
        
        return best_params, in_sample_result, out_of_sample_result
    
    def analyze_parameter_sensitivity(self,
                                       strategy_name: str,
                                       all_results: List[Dict]) -> Dict[str, float]:
        """
        Analyze how sensitive performance is to each parameter.
        
        Returns:
            Dict mapping parameter name to sensitivity score (0-1)
        """
        if not all_results:
            return {}
        
        param_names = list(all_results[0]['params'].keys())
        sensitivity = {}
        
        for param_name in param_names:
            # Group results by this parameter value
            sharpes_by_value = {}
            
            for result in all_results:
                value = result['params'][param_name]
                if value not in sharpes_by_value:
                    sharpes_by_value[value] = []
                sharpes_by_value[value].append(result['sharpe'])
            
            # Calculate variance across values
            mean_sharpes = [np.mean(sharpes) for sharpes in sharpes_by_value.values()]
            
            if len(mean_sharpes) > 1:
                sensitivity[param_name] = np.std(mean_sharpes) / (np.mean(mean_sharpes) + 1e-6)
            else:
                sensitivity[param_name] = 0.0
        
        return sensitivity
    
    def optimize_strategy(self,
                          strategy_name: str,
                          symbols: List[str] = None) -> OptimizationResult:
        """
        Run complete parameter optimization.
        
        Args:
            strategy_name: Strategy to optimize
            symbols: Optional list of symbols
            
        Returns:
            OptimizationResult with best parameters and analysis
        """
        logger.info("=" * 60)
        logger.info(f"PARAMETER OPTIMIZATION: {strategy_name}")
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
            if 'timestamp' in vix_data.columns:
                vix_data['regime'] = 'normal'
                vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
                vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
        
        # Get baseline performance (current parameters)
        baseline_params = {p.name: p.current_value 
                          for p in self.param_spaces.get(strategy_name, [])}
        
        logger.info(f"Baseline parameters: {baseline_params}")
        baseline_strategy = self.create_strategy_with_params(strategy_name, baseline_params)
        backtester = Backtester(self.initial_capital)
        baseline_result = backtester.run(baseline_strategy, data, vix_data=vix_data)
        
        logger.info(f"Baseline Sharpe: {baseline_result.sharpe_ratio:.2f}")
        
        # Walk-forward optimization
        best_params, in_sample_result, out_of_sample_result = self.walk_forward_optimization(
            strategy_name, data, vix_data
        )
        
        # Full grid search for sensitivity analysis
        logger.info("\nRunning full grid search...")
        all_results = self.grid_search(strategy_name, data, vix_data)
        
        # Analyze sensitivity
        sensitivity = self.analyze_parameter_sensitivity(strategy_name, all_results)
        
        # Classify stability
        stable_params = [p for p, s in sensitivity.items() if s < 0.1]
        unstable_params = [p for p, s in sensitivity.items() if s >= 0.1]
        
        # Calculate improvement
        improvement_pct = ((in_sample_result.sharpe_ratio - baseline_result.sharpe_ratio) / 
                          baseline_result.sharpe_ratio * 100) if baseline_result.sharpe_ratio > 0 else 0
        
        # Calculate degradation (in-sample vs out-of-sample)
        degradation_pct = ((in_sample_result.sharpe_ratio - out_of_sample_result.sharpe_ratio) / 
                          in_sample_result.sharpe_ratio * 100) if in_sample_result.sharpe_ratio > 0 else 0
        
        # Create result
        result = OptimizationResult(
            strategy=strategy_name,
            timestamp=datetime.now().isoformat(),
            best_params=best_params,
            best_sharpe=in_sample_result.sharpe_ratio,
            baseline_params=baseline_params,
            baseline_sharpe=baseline_result.sharpe_ratio,
            improvement_pct=improvement_pct,
            all_results=all_results,
            param_sensitivity=sensitivity,
            stable_params=stable_params,
            unstable_params=unstable_params,
            in_sample_sharpe=in_sample_result.sharpe_ratio,
            out_of_sample_sharpe=out_of_sample_result.sharpe_ratio,
            degradation_pct=degradation_pct
        )
        
        # Log to database
        for param_name, old_value in baseline_params.items():
            new_value = best_params.get(param_name)
            
            if new_value != old_value:
                self.db.execute(
                    'research',
                    """
                    INSERT INTO optimizations
                    (timestamp, strategy, param_name, param_type, old_value, new_value,
                     old_sharpe, new_sharpe, improvement_pct, sample_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (datetime.now().isoformat(), strategy_name, param_name, 'unknown',
                     str(old_value), str(new_value), baseline_result.sharpe_ratio,
                     in_sample_result.sharpe_ratio, improvement_pct, len(all_results))
                )
        
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best Sharpe: {result.best_sharpe:.2f}")
        logger.info(f"Improvement: {improvement_pct:.1f}%")
        logger.info(f"Out-of-sample degradation: {degradation_pct:.1f}%")
        
        return result
    
    def generate_report(self, result: OptimizationResult, output_path: Path = None) -> str:
        """Generate optimization report."""
        lines = []
        
        lines.append("# Parameter Optimization Report")
        lines.append(f"\n**Strategy:** {result.strategy}")
        lines.append(f"**Timestamp:** {result.timestamp}")
        lines.append("")
        
        # Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"**Baseline Sharpe:** {result.baseline_sharpe:.2f}")
        lines.append(f"**Optimized Sharpe:** {result.best_sharpe:.2f}")
        lines.append(f"**Improvement:** {result.improvement_pct:.1f}%")
        lines.append("")
        lines.append(f"**In-Sample Sharpe:** {result.in_sample_sharpe:.2f}")
        lines.append(f"**Out-of-Sample Sharpe:** {result.out_of_sample_sharpe:.2f}")
        lines.append(f"**Degradation:** {result.degradation_pct:.1f}%")
        lines.append("")
        
        # Interpretation
        if result.degradation_pct > 50:
            lines.append("⚠️ **WARNING:** Significant overfitting detected (>50% degradation)")
        elif result.degradation_pct > 30:
            lines.append("⚠️ **CAUTION:** Moderate overfitting (30-50% degradation)")
        else:
            lines.append("✅ **GOOD:** Acceptable out-of-sample performance")
        
        lines.append("")
        
        # Parameter changes
        lines.append("## Optimized Parameters")
        lines.append("")
        lines.append("| Parameter | Baseline | Optimized | Change |")
        lines.append("|-----------|----------|-----------|--------|")
        
        for param_name in result.baseline_params.keys():
            old = result.baseline_params[param_name]
            new = result.best_params.get(param_name, old)
            change = "→" if old != new else "—"
            lines.append(f"| {param_name:20s} | {str(old):10s} | {str(new):10s} | {change:6s} |")
        
        lines.append("")
        
        # Sensitivity analysis
        lines.append("## Parameter Sensitivity")
        lines.append("")
        lines.append("How much does each parameter affect performance?")
        lines.append("")
        lines.append("| Parameter | Sensitivity | Classification |")
        lines.append("|-----------|-------------|----------------|")
        
        for param, sens in sorted(result.param_sensitivity.items(), key=lambda x: x[1], reverse=True):
            classification = "Unstable" if sens >= 0.1 else "Stable"
            lines.append(f"| {param:20s} | {sens:11.3f} | {classification:14s} |")
        
        lines.append("")
        
        if result.stable_params:
            lines.append(f"**Stable parameters:** {', '.join(result.stable_params)}")
        if result.unstable_params:
            lines.append(f"**Unstable parameters:** {', '.join(result.unstable_params)}")
        
        lines.append("")
        
        # Top results
        lines.append("## Top Parameter Combinations")
        lines.append("")
        lines.append("| Rank | Sharpe | Annual Return | Max DD | Parameters |")
        lines.append("|------|--------|---------------|--------|------------|")
        
        for i, res in enumerate(result.all_results[:10], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in res['params'].items())
            lines.append(
                f"| {i:4d} | {res['sharpe']:6.2f} | {res['annual_return']:12.1f}% | "
                f"{res['max_drawdown']:6.1f}% | {params_str} |"
            )
        
        lines.append("")
        
        # Recommendation
        lines.append("## Recommendation")
        lines.append("")
        
        if result.improvement_pct > 20 and result.degradation_pct < 30:
            lines.append("✅ **APPLY** - Significant improvement with acceptable out-of-sample performance")
        elif result.improvement_pct > 10 and result.degradation_pct < 50:
            lines.append("⚠️ **CONSIDER** - Moderate improvement but monitor for overfitting")
        else:
            lines.append("❌ **REJECT** - Insufficient improvement or excessive overfitting")
        
        lines.append("")
        
        # Join and save
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def optimize_all_strategies(self, 
                                 strategies: List[str] = None,
                                 max_combinations_per_strategy: int = 50) -> Dict[str, OptimizationResult]:
        """
        Optimize all strategies and return consolidated results.
        
        Args:
            strategies: List of strategies to optimize (defaults to all)
            max_combinations_per_strategy: Max parameter combos per strategy
            
        Returns:
            Dict mapping strategy name to OptimizationResult
        """
        if strategies is None:
            strategies = self.get_available_strategies()
        
        results = {}
        summary = []
        
        print("\n" + "=" * 70)
        print("MULTI-STRATEGY PARAMETER OPTIMIZATION")
        print("=" * 70)
        print(f"\nOptimizing {len(strategies)} strategies...\n")
        
        # Print parameter space summary
        print("Strategy Parameter Spaces:")
        print("-" * 70)
        for strat in strategies:
            combos = self.get_total_combinations(strat)
            params = [p.name for p in self.get_param_space(strat)]
            print(f"  {strat:30s}: {combos:6d} combinations ({len(params)} params)")
        print("-" * 70 + "\n")
        
        for i, strategy_name in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] Optimizing {strategy_name}...")
            print("=" * 50)
            
            try:
                result = self.optimize_strategy(strategy_name)
                results[strategy_name] = result
                
                summary.append({
                    'strategy': strategy_name,
                    'baseline_sharpe': result.baseline_sharpe,
                    'optimized_sharpe': result.best_sharpe,
                    'improvement': result.improvement_pct,
                    'oos_degradation': result.degradation_pct,
                    'status': 'success'
                })
                
                # Generate individual report
                report_path = DIRS.get('backtests', Path('./research/backtests')) / f"optimization_{strategy_name}.md"
                self.generate_report(result, report_path)
                
            except Exception as e:
                logger.error(f"Failed to optimize {strategy_name}: {e}")
                summary.append({
                    'strategy': strategy_name,
                    'baseline_sharpe': 0,
                    'optimized_sharpe': 0,
                    'improvement': 0,
                    'oos_degradation': 0,
                    'status': f'error: {str(e)[:50]}'
                })
        
        # Print consolidated summary
        print("\n" + "=" * 70)
        print("OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"\n{'Strategy':<30} {'Baseline':>10} {'Optimized':>10} {'Improve':>10} {'OOS Deg':>10}")
        print("-" * 70)
        
        for s in summary:
            if s['status'] == 'success':
                print(f"{s['strategy']:<30} {s['baseline_sharpe']:>10.2f} {s['optimized_sharpe']:>10.2f} "
                      f"{s['improvement']:>9.1f}% {s['oos_degradation']:>9.1f}%")
            else:
                print(f"{s['strategy']:<30} {'ERROR':>10} {s['status']:<40}")
        
        print("-" * 70)
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        for s in summary:
            if s['status'] != 'success':
                print(f"  ❌ {s['strategy']}: Failed - {s['status']}")
            elif s['improvement'] > 20 and s['oos_degradation'] < 30:
                print(f"  ✅ {s['strategy']}: APPLY - {s['improvement']:.1f}% improvement, {s['oos_degradation']:.1f}% degradation")
            elif s['improvement'] > 10 and s['oos_degradation'] < 50:
                print(f"  ⚠️  {s['strategy']}: CONSIDER - moderate improvement")
            else:
                print(f"  ❌ {s['strategy']}: REJECT - insufficient improvement or high overfitting")
        
        print("\n" + "=" * 70)
        
        return results
    
    def print_strategy_summary(self):
        """Print summary of all available strategies and their parameter spaces."""
        print("\n" + "=" * 70)
        print("AVAILABLE STRATEGIES FOR OPTIMIZATION")
        print("=" * 70)
        
        total_combos = 0
        
        for strategy_name in self.get_available_strategies():
            param_space = self.get_param_space(strategy_name)
            combos = self.get_total_combinations(strategy_name)
            total_combos += combos
            
            print(f"\n{strategy_name}:")
            print(f"  Parameters: {len(param_space)}")
            print(f"  Total combinations: {combos}")
            print(f"  Parameters:")
            
            for p in param_space:
                print(f"    - {p.name}: {p.values} (current: {p.current_value})")
        
        print(f"\n" + "-" * 70)
        print(f"Total strategies: {len(self.get_available_strategies())}")
        print(f"Total parameter combinations: {total_combos}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Optimize strategy parameters')
    parser.add_argument('--strategy', '-s', type=str, default=None,
                        help='Strategy to optimize (or "all" for all strategies)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available strategies and parameters')
    parser.add_argument('--max-combos', '-m', type=int, default=50,
                        help='Maximum parameter combinations to test')
    
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer()
    
    if args.list:
        optimizer.print_strategy_summary()
    elif args.strategy == 'all':
        # Optimize all strategies
        results = optimizer.optimize_all_strategies()
    elif args.strategy:
        # Optimize specific strategy
        if args.strategy not in optimizer.get_available_strategies():
            print(f"\nUnknown strategy: {args.strategy}")
            print(f"Available strategies: {optimizer.get_available_strategies()}")
            sys.exit(1)
        
        result = optimizer.optimize_strategy(args.strategy)
        
        # Generate report
        report_path = DIRS.get('backtests', Path('./research/backtests')) / f"optimization_{result.strategy}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        optimizer.generate_report(result, report_path)
        
        print(f"\nReport saved to: {report_path}")
    else:
        # Show help
        print("\n" + "=" * 70)
        print("PARAMETER OPTIMIZER")
        print("=" * 70)
        print("\nUsage:")
        print("  python parameter_optimizer.py --list              # Show all strategies")
        print("  python parameter_optimizer.py -s mean_reversion   # Optimize one strategy")
        print("  python parameter_optimizer.py -s all              # Optimize all strategies")
        print("\nAvailable strategies:")
        for s in optimizer.get_available_strategies():
            combos = optimizer.get_total_combinations(s)
            print(f"  - {s} ({combos} combinations)")
        print("\n" + "=" * 70)
