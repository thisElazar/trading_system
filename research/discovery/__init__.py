"""
Autonomous Strategy Discovery Engine
=====================================
Genetic programming-based strategy discovery with multi-objective optimization
and novelty search for continuous overnight evolution.

Components:
- config: Evolution configuration
- gp_core: DEAP-based genetic programming primitives
- strategy_genome: Tree-based strategy representation
- strategy_compiler: Converts GP trees to executable strategies
- multi_objective: NSGA-II with Sortino, drawdown, CVaR, novelty
- novelty_search: Behavioral diversity maintenance
- evolution_engine: Core evolutionary loop
- island_model: Island-based parallel evolution for diversity preservation
- overnight_runner: Autonomous overnight runner
- db_schema: Database schema extensions

Usage:
    from research.discovery import EvolutionEngine, EvolutionConfig

    config = EvolutionConfig(population_size=100, generations_per_session=50)
    engine = EvolutionEngine(config=config)
    engine.load_data(data=market_data)
    engine.run(hours=8)

Island Model Usage:
    from research.discovery import IslandEvolutionEngine, IslandConfig, EvolutionConfig

    config = EvolutionConfig(population_size=20, generations_per_session=100)
    island_config = IslandConfig(num_islands=4, population_per_island=20)
    engine = IslandEvolutionEngine(config=config, island_config=island_config)
    engine.load_data(data=market_data)
    engine.initialize_populations()
    engine.evolve_generation()  # Run single generation across all islands

CLI Usage:
    python -m research.discovery.overnight_runner --hours 8
    python -m research.discovery.overnight_runner --hours 8 --islands 6
"""

from .config import (
    EvolutionConfig,
    IslandConfig,
    DEFAULT_CONFIG,
    PI_CONFIG,
    DEFAULT_ISLAND_CONFIG,
    OVERNIGHT_ISLAND_CONFIG
)
from .strategy_genome import StrategyGenome, GenomeFactory
from .strategy_compiler import StrategyCompiler, EvolvedStrategy
from .multi_objective import FitnessVector, calculate_fitness_vector
from .novelty_search import BehaviorVector, NoveltyArchive
from .evolution_engine import EvolutionEngine, EvolutionState
from .island_model import IslandEvolutionEngine
from .diversity_metrics import (
    DiversityMonitor,
    DiversityThresholds,
    GenotypeMetrics,
    calculate_genotype_diversity
)
from .map_elites import MAPElitesGrid, MapElitesConfig, create_default_grid
from .portfolio_fitness import (
    PortfolioFitnessEvaluator,
    PortfolioContribution,
    CompositeScore,
    FitnessWeights
)
from .promotion_pipeline import (
    PromotionPipeline,
    PromotionCriteria,
    StrategyStatus,
    StrategyRecord,
    RetirementReason
)

__all__ = [
    # Configuration
    'EvolutionConfig',
    'IslandConfig',
    'DEFAULT_CONFIG',
    'PI_CONFIG',
    'DEFAULT_ISLAND_CONFIG',
    'OVERNIGHT_ISLAND_CONFIG',

    # Core components
    'StrategyGenome',
    'GenomeFactory',
    'StrategyCompiler',
    'EvolvedStrategy',

    # Multi-objective
    'FitnessVector',
    'calculate_fitness_vector',

    # Novelty search
    'BehaviorVector',
    'NoveltyArchive',

    # Portfolio fitness
    'PortfolioFitnessEvaluator',
    'PortfolioContribution',
    'CompositeScore',
    'FitnessWeights',

    # Evolution
    'EvolutionEngine',
    'EvolutionState',
    'IslandEvolutionEngine',

    # Diversity Metrics
    'DiversityMonitor',
    'DiversityThresholds',
    'GenotypeMetrics',
    'calculate_genotype_diversity',

    # MAP-Elites
    'MAPElitesGrid',
    'MapElitesConfig',
    'create_default_grid',

    # Promotion Pipeline
    'PromotionPipeline',
    'PromotionCriteria',
    'StrategyStatus',
    'StrategyRecord',
    'RetirementReason',
]
