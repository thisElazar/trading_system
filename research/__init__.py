"""
Research Module
===============
Backtesting, optimization, and strategy validation.
"""

from research.backtester import Backtester, BacktestResult
from research.monte_carlo import MonteCarloSimulator
from research.parameter_optimizer import ParameterOptimizer
from research.strategy_comparison import StrategyComparison
from research.parallel_backtester import (
    ParallelBacktester,
    ParallelConfig,
    ParallelResult,
    run_all_strategies_parallel
)
from research.strategy_correlation import (
    StrategyCorrelationAnalyzer,
    CorrelationResult,
    calculate_strategy_correlations
)
from research.strategy_attribution import (
    StrategyAttributionTracker,
    AttributedPosition,
    PortfolioAttribution,
    StrategyAttribution,
    attribute_backtest_trades
)
from research.regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeSignal,
    RegimeState,
    REGIME_ADJUSTMENTS
)

__all__ = [
    'Backtester',
    'BacktestResult',
    'MonteCarloSimulator',
    'ParameterOptimizer',
    'StrategyComparison',
    'ParallelBacktester',
    'ParallelConfig',
    'ParallelResult',
    'run_all_strategies_parallel',
    # Correlation analysis
    'StrategyCorrelationAnalyzer',
    'CorrelationResult',
    'calculate_strategy_correlations',
    # Strategy attribution
    'StrategyAttributionTracker',
    'AttributedPosition',
    'PortfolioAttribution',
    'StrategyAttribution',
    'attribute_backtest_trades',
    # Regime detection
    'RegimeDetector',
    'MarketRegime',
    'RegimeSignal',
    'RegimeState',
    'REGIME_ADJUSTMENTS',
]
