"""
Genetic Algorithm Optimization
==============================
Advanced evolutionary algorithms for strategy development.

Core GA:
- GeneticOptimizer: Core GA implementation
- PersistentGAOptimizer: GA with database persistence for cross-session evolution

Adaptive GA (NEW):
- MarketPeriodLibrary: Curated periods (COVID crash, 2008 crisis, bull runs)
- RapidBacktester: Ultra-fast backtester for 30-second tests
- RegimeMatchingEngine: Match current conditions to historical periods
- AdaptiveGAOptimizer: Multi-scale testing with regime awareness
- StrategyGenome: Creative DNA for novel strategy discovery
- MultiScaleFitnessCalculator: Combined fitness across regimes
- AdaptiveStrategyManager: Real-time rebalancing based on conditions
"""

from research.genetic.optimizer import (
    GeneticOptimizer,
    GeneticConfig,
    Individual,
    ParameterSpec,
    STRATEGY_PARAMS,
    optimize_strategy
)

from research.genetic.persistent_optimizer import (
    PersistentGAOptimizer,
    create_backtest_fitness_fn
)

# Adaptive GA components
from research.genetic.market_periods import (
    MarketPeriodLibrary,
    MarketPeriod,
    PeriodType,
)

from research.genetic.rapid_backtester import (
    RapidBacktester,
    RapidBacktestResult,
    MultiPeriodResult,
    create_rapid_fitness_function,
)

from research.genetic.regime_matching import (
    RegimeMatchingEngine,
    RegimeFingerprint,
    PeriodMatch,
)

from research.genetic.adaptive_optimizer import (
    AdaptiveGAOptimizer,
    AdaptiveGAConfig,
    Individual as AdaptiveIndividual,
    Island,
)

from research.genetic.strategy_genome import (
    StrategyGenome,
    GenomeFactory,
    SignalGene,
    SignalType,
    RiskGene,
    FilterGene,
)

from research.genetic.multiscale_fitness import (
    MultiScaleFitnessCalculator,
    MultiScaleFitnessResult,
    FitnessWeights,
    create_adaptive_fitness_function,
)

from research.genetic.adaptive_strategy_manager import (
    AdaptiveStrategyManager,
    AdaptiveManagerConfig,
    PortfolioState,
    StrategyAllocation,
    RebalanceReason,
)

__all__ = [
    # Core GA
    'GeneticOptimizer',
    'GeneticConfig',
    'Individual',
    'ParameterSpec',
    'STRATEGY_PARAMS',
    'optimize_strategy',

    # Persistent GA
    'PersistentGAOptimizer',
    'create_backtest_fitness_fn',

    # Market Periods
    'MarketPeriodLibrary',
    'MarketPeriod',
    'PeriodType',

    # Rapid Backtesting
    'RapidBacktester',
    'RapidBacktestResult',
    'MultiPeriodResult',
    'create_rapid_fitness_function',

    # Regime Matching
    'RegimeMatchingEngine',
    'RegimeFingerprint',
    'PeriodMatch',

    # Adaptive Optimizer
    'AdaptiveGAOptimizer',
    'AdaptiveGAConfig',
    'AdaptiveIndividual',
    'Island',

    # Strategy Genome
    'StrategyGenome',
    'GenomeFactory',
    'SignalGene',
    'SignalType',
    'RiskGene',
    'FilterGene',

    # Multi-Scale Fitness
    'MultiScaleFitnessCalculator',
    'MultiScaleFitnessResult',
    'FitnessWeights',
    'create_adaptive_fitness_function',

    # Adaptive Manager
    'AdaptiveStrategyManager',
    'AdaptiveManagerConfig',
    'PortfolioState',
    'StrategyAllocation',
    'RebalanceReason',
]
