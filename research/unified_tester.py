"""
Unified Strategy Testing Framework
===================================
Standardizes testing across all strategies with:
- Consistent interface adapters
- Shared backtester integration  
- Genetic optimizer integration
- Ensemble coordination

Usage:
    tester = UnifiedTester()
    results = tester.test_all_strategies()
    tester.optimize_strategy('vol_managed_momentum')
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS, VALIDATION
from data.cached_data_manager import CachedDataManager
from research.backtester import Backtester, BacktestResult
from research.genetic.optimizer import (
    GeneticOptimizer, GeneticConfig, ParameterSpec, Individual, STRATEGY_PARAMS
)

# Import all strategies
from strategies.vol_managed_momentum import VolManagedMomentumStrategy
from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
from strategies.sector_rotation import SectorRotationStrategy
from strategies.pairs_trading import PairsAnalyzer, PairsTradingStrategy
from strategies.relative_volume_breakout import RelativeVolumeBreakout

logger = logging.getLogger(__name__)


@dataclass
class StrategyTestResult:
    """Unified test result for any strategy."""
    strategy_name: str
    sharpe_ratio: float
    annual_return: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    meets_threshold: bool
    vs_research_pct: float
    
    # Optional detailed results
    backtest_result: Optional[BacktestResult] = None
    trades_df: Optional[pd.DataFrame] = None
    metadata: Dict = field(default_factory=dict)


# Extended parameter specs for all strategies
EXTENDED_STRATEGY_PARAMS = {
    'vol_managed_momentum': [
        ParameterSpec('formation_period', 126, 315, step=21, dtype=int),  # 6-15 months
        ParameterSpec('skip_period', 10, 42, step=7, dtype=int),          # 2-6 weeks skip
        ParameterSpec('vol_lookback', 10, 42, step=7, dtype=int),         # vol window
        ParameterSpec('target_vol', 0.10, 0.25, step=0.025),              # target volatility
        ParameterSpec('top_percentile', 0.10, 0.30, step=0.05),           # quintile selection
    ],
    'vix_regime_rotation': [
        ParameterSpec('low_threshold', 12, 18, step=1, dtype=int),
        ParameterSpec('high_threshold', 22, 32, step=2, dtype=int),
        ParameterSpec('confirmation_days', 1, 5, step=1, dtype=int),
        ParameterSpec('min_days_between', 3, 10, step=1, dtype=int),
    ],
    'sector_rotation': [
        ParameterSpec('low_vix_threshold', 14, 20, step=1, dtype=int),
        ParameterSpec('high_vix_threshold', 22, 30, step=2, dtype=int),
        ParameterSpec('rebalance_threshold', 0.03, 0.10, step=0.01),
        ParameterSpec('momentum_weight', 0.1, 0.5, step=0.1),
    ],
    'pairs_trading': [
        ParameterSpec('entry_z', 1.5, 3.0, step=0.25),
        ParameterSpec('exit_z', 0.25, 1.0, step=0.25),
        ParameterSpec('stop_z', 3.0, 5.0, step=0.5),
        ParameterSpec('max_hold', 15, 45, step=5, dtype=int),
    ],
    'relative_volume_breakout': [
        ParameterSpec('min_rv', 1.25, 3.0, step=0.25),
        ParameterSpec('atr_stop_mult', 0.5, 2.0, step=0.25),
        ParameterSpec('atr_target_mult', 1.5, 4.0, step=0.5),
        ParameterSpec('max_hold_days', 3, 10, step=1, dtype=int),
    ],
}


class StrategyAdapter:
    """
    Adapts different strategy interfaces to a common format.
    All strategies return signals compatible with shared backtester.
    """
    
    def __init__(self, strategy_name: str):
        self.name = strategy_name
        self.strategy = self._create_strategy()
        self.data_mgr = CachedDataManager()
        
    def _create_strategy(self):
        """Factory for strategy instances."""
        if self.name == 'vol_managed_momentum':
            return VolManagedMomentumStrategy()
        elif self.name == 'vix_regime_rotation':
            return VIXRegimeRotationStrategy()
        elif self.name == 'sector_rotation':
            return SectorRotationStrategy()
        elif self.name == 'pairs_trading':
            return PairsTradingStrategy()
        elif self.name == 'relative_volume_breakout':
            return RelativeVolumeBreakout()
        else:
            raise ValueError(f"Unknown strategy: {self.name}")
    
    def apply_params(self, params: Dict[str, Any]):
        """Apply parameter values to strategy."""
        for key, value in params.items():
            if hasattr(self.strategy, key):
                setattr(self.strategy, key, value)
    
    def is_compatible_with_backtester(self) -> bool:
        """Check if strategy works with shared backtester."""
        return self.name in [
            'vol_managed_momentum',
            'vix_regime_rotation', 
            'sector_rotation'
        ]


class UnifiedTester:
    """
    Unified testing framework for all strategies.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.data_mgr = CachedDataManager()
        self.results: Dict[str, StrategyTestResult] = {}
        
        # Load data once
        if not self.data_mgr.cache:
            logger.info("Loading market data...")
            self.data_mgr.load_all()
        
        # Load VIX
        self.vix_data = self._load_vix()
    
    def _load_vix(self) -> Optional[pd.DataFrame]:
        """Load VIX data with regime classification."""
        vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
        if not vix_path.exists():
            logger.warning("No VIX data found")
            return None
        
        vix_df = pd.read_parquet(vix_path)
        
        # Ensure datetime index
        if 'timestamp' in vix_df.columns:
            vix_df = vix_df.set_index('timestamp')
        if vix_df.index.tz is not None:
            vix_df.index = vix_df.index.tz_localize(None)
        
        # Add regime
        vix_df['regime'] = 'normal'
        vix_df.loc[vix_df['close'] < 15, 'regime'] = 'low'
        vix_df.loc[vix_df['close'] > 25, 'regime'] = 'high'
        vix_df.loc[vix_df['close'] > 40, 'regime'] = 'extreme'
        
        return vix_df
    
    def _prepare_data(self, n_symbols: int = None) -> Dict[str, pd.DataFrame]:
        """Prepare data dict with datetime index for backtester."""
        metadata = self.data_mgr.get_all_metadata()
        
        # Sort by liquidity
        sorted_symbols = sorted(
            metadata.items(),
            key=lambda x: x[1].get('dollar_volume', 0),
            reverse=True
        )
        
        if n_symbols:
            sorted_symbols = sorted_symbols[:n_symbols]
        
        data = {}
        for symbol, _ in sorted_symbols:
            df = self.data_mgr.get_bars(symbol)
            if df is not None and len(df) >= 252:
                # Convert timestamp column to index if needed
                if 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                data[symbol] = df
        
        return data
    
    def test_strategy(
        self,
        strategy_name: str,
        n_symbols: int = 100,
        params: Dict = None
    ) -> StrategyTestResult:
        """
        Test a single strategy.
        
        Args:
            strategy_name: Name of strategy to test
            n_symbols: Number of symbols to use (sorted by liquidity)
            params: Optional parameter overrides
            
        Returns:
            StrategyTestResult
        """
        logger.info(f"Testing {strategy_name}...")
        
        adapter = StrategyAdapter(strategy_name)
        
        if params:
            adapter.apply_params(params)
        
        # Use shared backtester for compatible strategies
        if adapter.is_compatible_with_backtester():
            data = self._prepare_data(n_symbols)
            backtester = Backtester(initial_capital=self.capital)
            result = backtester.run(
                strategy=adapter.strategy,
                data=data,
                vix_data=self.vix_data
            )
            
            return StrategyTestResult(
                strategy_name=strategy_name,
                sharpe_ratio=result.sharpe_ratio,
                annual_return=result.annual_return,
                total_trades=result.total_trades,
                win_rate=result.win_rate,
                max_drawdown=result.max_drawdown_pct,
                meets_threshold=result.meets_threshold,
                vs_research_pct=result.vs_research_pct,
                backtest_result=result
            )
        
        # Use strategy's own backtester for others
        elif strategy_name == 'pairs_trading':
            return self._test_pairs_trading(params)
        elif strategy_name == 'relative_volume_breakout':
            return self._test_rv_breakout(params)
        else:
            raise ValueError(f"No test method for {strategy_name}")
    
    def _test_pairs_trading(self, params: Dict = None) -> StrategyTestResult:
        """Test pairs trading with its own backtester."""
        from strategies.pairs_trading import PairsBacktester, PairsAnalyzer, PairsTradingStrategy
        
        # Use optimized defaults from strategy class
        entry_z = params.get('entry_z', PairsTradingStrategy.ENTRY_ZSCORE) if params else PairsTradingStrategy.ENTRY_ZSCORE
        exit_z = params.get('exit_z', PairsTradingStrategy.EXIT_ZSCORE) if params else PairsTradingStrategy.EXIT_ZSCORE
        stop_z = params.get('stop_z', PairsTradingStrategy.STOP_ZSCORE) if params else PairsTradingStrategy.STOP_ZSCORE
        max_hold = params.get('max_hold', PairsTradingStrategy.MAX_HOLD_DAYS) if params else PairsTradingStrategy.MAX_HOLD_DAYS
        
        analyzer = PairsAnalyzer()
        backtester = PairsBacktester()
        
        # Find pairs
        all_pairs = analyzer.find_all_pairs(max_per_sector=2)
        pairs_list = []
        for sector_pairs in all_pairs.values():
            pairs_list.extend(sector_pairs)
        
        if not pairs_list:
            return StrategyTestResult(
                strategy_name='pairs_trading',
                sharpe_ratio=0, annual_return=0, total_trades=0,
                win_rate=0, max_drawdown=0, meets_threshold=False,
                vs_research_pct=0
            )
        
        # Backtest each pair
        all_trades = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        for pair in pairs_list:
            trades = backtester.backtest_pair(
                pair, start_date, end_date,
                entry_z=entry_z, exit_z=exit_z, 
                stop_z=stop_z, max_hold=max_hold
            )
            if len(trades) > 0:
                trades['pair'] = f"{pair.stock_a}/{pair.stock_b}"
                all_trades.append(trades)
        
        if not all_trades:
            return StrategyTestResult(
                strategy_name='pairs_trading',
                sharpe_ratio=0, annual_return=0, total_trades=0,
                win_rate=0, max_drawdown=0, meets_threshold=False,
                vs_research_pct=0
            )
        
        trades_df = pd.concat(all_trades, ignore_index=True)
        
        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = trades_df['win'].mean() * 100
        total_pnl = trades_df['pnl_pct'].sum()
        days = (end_date - start_date).days
        annual_return = total_pnl * (365 / days)
        
        # Approximate Sharpe (assumes daily trades)
        if len(trades_df) > 1:
            sharpe = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252/trades_df['days_held'].mean())
        else:
            sharpe = 0
        
        research_sharpe = VALIDATION.get('pairs_trading', {}).get('research_sharpe', 2.5)
        min_sharpe = VALIDATION.get('pairs_trading', {}).get('min_sharpe', 1.5)
        
        return StrategyTestResult(
            strategy_name='pairs_trading',
            sharpe_ratio=sharpe,
            annual_return=annual_return,
            total_trades=total_trades,
            win_rate=win_rate,
            max_drawdown=0,  # Not tracked in pairs backtester
            meets_threshold=sharpe >= min_sharpe,
            vs_research_pct=(sharpe / research_sharpe * 100) if research_sharpe else 0,
            trades_df=trades_df
        )
    
    def _test_rv_breakout(self, params: Dict = None) -> StrategyTestResult:
        """Test relative volume breakout with its own backtester."""
        from strategies.relative_volume_breakout import RVBreakoutBacktester, RelativeVolumeBreakout
        
        backtester = RVBreakoutBacktester()
        
        # Use optimized defaults from strategy class
        if params:
            backtester.strategy.MIN_RELATIVE_VOLUME = params.get('min_rv', RelativeVolumeBreakout.MIN_RELATIVE_VOLUME)
            backtester.strategy.MIN_GAP_PCT = params.get('min_gap_pct', RelativeVolumeBreakout.MIN_GAP_PCT)
            backtester.strategy.ATR_STOP_MULT = params.get('atr_stop_mult', RelativeVolumeBreakout.ATR_STOP_MULT)
            backtester.strategy.ATR_TARGET_MULT = params.get('atr_target_mult', RelativeVolumeBreakout.ATR_TARGET_MULT)
            backtester.strategy.MAX_HOLD_DAYS = params.get('max_hold_days', RelativeVolumeBreakout.MAX_HOLD_DAYS)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        results = backtester.backtest_universe(start_date=start_date, end_date=end_date)
        
        if len(results) == 0:
            return StrategyTestResult(
                strategy_name='relative_volume_breakout',
                sharpe_ratio=0, annual_return=0, total_trades=0,
                win_rate=0, max_drawdown=0, meets_threshold=False,
                vs_research_pct=0
            )
        
        total_trades = len(results)
        win_rate = results['win'].mean() * 100
        total_pnl = results['pnl_pct'].sum()
        annual_return = total_pnl  # Already ~1 year
        
        if total_trades > 1:
            sharpe = results['pnl_pct'].mean() / results['pnl_pct'].std() * np.sqrt(252/results['days_held'].mean())
        else:
            sharpe = 0
        
        research_sharpe = VALIDATION.get('relative_volume_breakout', {}).get('research_sharpe', 2.81)
        min_sharpe = VALIDATION.get('relative_volume_breakout', {}).get('min_sharpe', 1.8)
        
        return StrategyTestResult(
            strategy_name='relative_volume_breakout',
            sharpe_ratio=sharpe,
            annual_return=annual_return,
            total_trades=total_trades,
            win_rate=win_rate,
            max_drawdown=0,
            meets_threshold=sharpe >= min_sharpe,
            vs_research_pct=(sharpe / research_sharpe * 100) if research_sharpe else 0,
            trades_df=results
        )
    
    def test_all_strategies(self, n_symbols: int = 100) -> Dict[str, StrategyTestResult]:
        """Test all available strategies."""
        strategies = [
            'vol_managed_momentum',
            'vix_regime_rotation',
            'sector_rotation',
            'pairs_trading',
            'relative_volume_breakout',
        ]
        
        results = {}
        for name in strategies:
            try:
                result = self.test_strategy(name, n_symbols)
                results[name] = result
                logger.info(
                    f"  {name}: Sharpe={result.sharpe_ratio:.2f}, "
                    f"Trades={result.total_trades}, Win={result.win_rate:.1f}%"
                )
            except Exception as e:
                logger.error(f"  {name}: FAILED - {e}")
        
        self.results = results
        return results
    
    def optimize_strategy(
        self,
        strategy_name: str,
        n_symbols: int = 100,
        generations: int = 15,
        population: int = 30
    ) -> Individual:
        """
        Optimize strategy parameters using genetic algorithm.
        
        Args:
            strategy_name: Strategy to optimize
            n_symbols: Symbols for backtesting
            generations: GA generations
            population: GA population size
            
        Returns:
            Best individual found
        """
        logger.info(f"Optimizing {strategy_name}...")
        
        param_specs = EXTENDED_STRATEGY_PARAMS.get(strategy_name)
        if not param_specs:
            raise ValueError(f"No parameter specs for {strategy_name}")
        
        def fitness_fn(genes: Dict) -> float:
            """Evaluate parameter set."""
            try:
                result = self.test_strategy(strategy_name, n_symbols, params=genes)
                # Optimize for Sharpe, penalize low trade count
                trade_penalty = max(0, 50 - result.total_trades) * 0.01
                return result.sharpe_ratio - trade_penalty
            except Exception as e:
                logger.warning(f"Fitness eval failed: {e}")
                return float('-inf')
        
        config = GeneticConfig(
            population_size=population,
            generations=generations,
            mutation_rate=0.15,
            crossover_rate=0.7,
            elitism=2,
            early_stop_generations=5
        )
        
        optimizer = GeneticOptimizer(param_specs, fitness_fn, config)
        best = optimizer.evolve()
        
        logger.info(f"Best params: {best.genes}")
        logger.info(f"Best fitness: {best.fitness:.3f}")
        
        return best
    
    def generate_report(self) -> str:
        """Generate markdown report of all test results."""
        lines = [
            "# Unified Strategy Test Report",
            f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Symbols:** {len(self.data_mgr.cache)}",
            "",
            "## Results Summary",
            "",
            "| Strategy | Sharpe | Annual Return | Trades | Win Rate | Meets Min | vs Research |",
            "|----------|--------|---------------|--------|----------|-----------|-------------|",
        ]
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True):
            meets = "✓" if result.meets_threshold else "✗"
            lines.append(
                f"| {name} | {result.sharpe_ratio:.2f} | {result.annual_return:.1f}% | "
                f"{result.total_trades} | {result.win_rate:.1f}% | {meets} | {result.vs_research_pct:.0f}% |"
            )
        
        lines.extend([
            "",
            "## Validation Status",
            ""
        ])
        
        for name, result in self.results.items():
            val = VALIDATION.get(name, {})
            min_sharpe = val.get('min_sharpe', 0)
            research_sharpe = val.get('research_sharpe', 0)
            
            status = "✅ PASS" if result.meets_threshold else "❌ FAIL"
            lines.append(f"- **{name}:** {status}")
            lines.append(f"  - Actual: {result.sharpe_ratio:.2f}, Min: {min_sharpe}, Research: {research_sharpe}")
        
        return "\n".join(lines)


def main():
    """Run unified testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("UNIFIED STRATEGY TESTER")
    print("="*60)
    
    tester = UnifiedTester()
    
    print("\n### Testing All Strategies ###\n")
    results = tester.test_all_strategies(n_symbols=100)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for name, result in sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True):
        status = "✓" if result.meets_threshold else "✗"
        print(f"{status} {name:30s} Sharpe={result.sharpe_ratio:5.2f}  "
              f"Trades={result.total_trades:4d}  Win={result.win_rate:5.1f}%")
    
    # Save report
    report = tester.generate_report()
    report_path = DIRS.get('backtests', Path('./research/backtests')) / 'unified_test_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
