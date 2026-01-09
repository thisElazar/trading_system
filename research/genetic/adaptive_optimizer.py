"""
Adaptive GA Optimizer
=====================
Advanced genetic algorithm optimizer that balances long-term vs regime-specific testing.

Key innovations:
- Multi-scale fitness: Combines long-term robustness with regime-specific performance
- Regime-aware evolution: Prioritizes testing against current market conditions
- Adaptive generation sizing: More rapid generations for regime-matched periods
- Island model: Maintains diverse sub-populations for different market conditions
- Creative mutation: Novel parameter combinations for strategy discovery

Usage:
    from research.genetic.adaptive_optimizer import AdaptiveGAOptimizer

    optimizer = AdaptiveGAOptimizer()

    # Evolve with regime awareness
    best = optimizer.evolve(
        strategy_factory=create_strategy,
        data=market_data,
        current_conditions={'vix': 18, 'trend': 0.3}
    )
"""

import logging
import time
import random
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from research.discovery.shared_data import SharedDataManager
import sys

# Import performance config
try:
    from config import PERF
except ImportError:
    PERF = {"parallel_enabled": False, "n_workers": 1}

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .market_periods import MarketPeriodLibrary, MarketPeriod, PeriodType
from .rapid_backtester import RapidBacktester, RapidBacktestResult, MultiPeriodResult
from .regime_matching import RegimeMatchingEngine, RegimeFingerprint

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """An individual in the GA population."""
    genes: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0

    # Multi-scale fitness components
    long_term_fitness: float = 0.0
    regime_fitness: float = 0.0
    crisis_fitness: float = 0.0
    consistency_score: float = 0.0

    # Performance by period type
    period_scores: Dict[str, float] = field(default_factory=dict)

    # Evaluation tracking - prevents re-evaluation when fitness happens to be 0
    evaluated: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Unique identifier based on genes."""
        gene_str = "_".join(f"{k}:{v:.4f}" for k, v in sorted(self.genes.items()))
        return hash(gene_str) % (10 ** 8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'genes': self.genes,
            'fitness': self.fitness,
            'generation': self.generation,
            'long_term_fitness': self.long_term_fitness,
            'regime_fitness': self.regime_fitness,
            'crisis_fitness': self.crisis_fitness,
            'consistency_score': self.consistency_score,
            'period_scores': self.period_scores,
        }


@dataclass
class Island:
    """
    Sub-population specialized for a specific market regime.

    Maintains diversity by evolving separately then occasionally migrating.
    """
    name: str
    regime_type: str  # "crisis", "bull", "bear", "sideways", "current"
    population: List[Individual]
    target_periods: List[str]  # Period names this island optimizes for
    fitness_weights: Dict[str, float]  # Custom fitness weights

    # Island statistics
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    generations_evolved: int = 0
    last_improvement_gen: int = 0


@dataclass
class AdaptiveGAConfig:
    """Configuration for AdaptiveGAOptimizer."""
    # Population settings
    total_population: int = 60
    n_islands: int = 4
    island_population: int = 15  # total_population / n_islands

    # Evolution settings
    generations_per_session: int = 10
    generations_per_island: int = 3

    # Selection
    tournament_size: int = 3
    elitism: int = 2

    # Crossover and mutation
    crossover_rate: float = 0.7
    mutation_rate: float = 0.12
    mutation_sigma: float = 0.1

    # Adaptive mutation
    adaptive_mutation: bool = True
    stagnation_threshold: int = 3
    mutation_boost: float = 1.5

    # Migration
    migration_interval: int = 3  # Migrate every N generations
    migration_size: int = 2  # Number of individuals to migrate

    # Fitness weighting
    long_term_weight: float = 0.35
    regime_weight: float = 0.35
    crisis_weight: float = 0.15
    consistency_weight: float = 0.15

    # Performance
    parallel: bool = False  # Override via PERF
    n_workers: int = 1  # Override via PERF
    timeout_per_eval: int = 30  # seconds

    # Rapid testing
    use_rapid_testing: bool = True
    rapid_generations: int = 5  # Extra rapid generations
    rapid_period_limit: int = 60  # Max days for rapid period


class AdaptiveGAOptimizer:
    """
    Advanced GA optimizer with regime awareness and multi-scale fitness.

    Features:
    - Island model for diversity
    - Regime-matched period testing
    - Adaptive mutation rates
    - Multi-scale fitness combining long-term and regime-specific
    - Creative mutation for novel strategies
    """

    # Default parameter ranges (can be overridden)
    DEFAULT_PARAMS = {
        'lookback_period': (10, 252, 1),  # (min, max, step)
        'entry_threshold': (0.5, 3.0, 0.1),
        'exit_threshold': (0.1, 1.5, 0.05),
        'stop_loss_pct': (0.005, 0.03, 0.001),
        'position_size_pct': (0.01, 0.05, 0.005),
        'max_hold_days': (1, 30, 1),
    }

    def __init__(
        self,
        config: AdaptiveGAConfig = None,
        parameter_specs: Dict[str, Tuple] = None,
        period_library: MarketPeriodLibrary = None,
        regime_engine: RegimeMatchingEngine = None
    ):
        """
        Initialize the adaptive optimizer.

        Args:
            config: AdaptiveGAConfig instance
            parameter_specs: Dict of parameter name -> (min, max, step)
            period_library: MarketPeriodLibrary instance
            regime_engine: RegimeMatchingEngine instance
        """
        self.config = config or AdaptiveGAConfig()
        self.param_specs = parameter_specs or self.DEFAULT_PARAMS.copy()
        self.library = period_library or MarketPeriodLibrary()
        self.regime_engine = regime_engine or RegimeMatchingEngine(self.library)

        # Backtester for rapid evaluation
        self.rapid_backtester = RapidBacktester(
            period_library=self.library,
            parallel=self.config.parallel,
            max_workers=self.config.n_workers
        )

        # Islands (sub-populations)
        self.islands: Dict[str, Island] = {}

        # Global best
        self.global_best: Optional[Individual] = None
        self.generation_history: List[Dict] = []

        # State tracking
        self.current_generation = 0
        self.total_evaluations = 0
        self.stagnation_counter = 0

        # Current regime fingerprint
        self._current_fingerprint: Optional[RegimeFingerprint] = None

        # Persistent pool state for multi-core parallelism
        self._shared_data_manager: Optional[SharedDataManager] = None
        self._pool_initialized = False

        logger.info(f"AdaptiveGAOptimizer initialized with {len(self.param_specs)} parameters")

    def _create_random_individual(self, generation: int = 0) -> Individual:
        """Create a random individual."""
        genes = {}
        for name, (min_val, max_val, step) in self.param_specs.items():
            if step >= 1:
                # Integer parameter
                genes[name] = float(random.randint(int(min_val), int(max_val)))
            else:
                # Float parameter
                n_steps = int((max_val - min_val) / step)
                genes[name] = min_val + random.randint(0, n_steps) * step

        return Individual(genes=genes, generation=generation)

    def _initialize_islands(self):
        """Initialize island sub-populations."""
        island_configs = [
            {
                'name': 'current_regime',
                'regime_type': 'current',
                'fitness_weights': {
                    'regime': 0.5,
                    'long_term': 0.25,
                    'crisis': 0.15,
                    'consistency': 0.10,
                },
            },
            {
                'name': 'crisis_specialists',
                'regime_type': 'crisis',
                'fitness_weights': {
                    'crisis': 0.5,
                    'consistency': 0.25,
                    'long_term': 0.15,
                    'regime': 0.10,
                },
            },
            {
                'name': 'bull_optimized',
                'regime_type': 'bull',
                'fitness_weights': {
                    'long_term': 0.4,
                    'regime': 0.3,
                    'consistency': 0.2,
                    'crisis': 0.1,
                },
            },
            {
                'name': 'robustness_focused',
                'regime_type': 'diverse',
                'fitness_weights': {
                    'consistency': 0.4,
                    'long_term': 0.3,
                    'crisis': 0.2,
                    'regime': 0.1,
                },
            },
        ]

        # Get target periods for each island
        test_periods = self.regime_engine.get_ga_test_periods()

        for i, island_cfg in enumerate(island_configs[:self.config.n_islands]):
            pop_size = self.config.island_population
            population = [
                self._create_random_individual(0)
                for _ in range(pop_size)
            ]

            # Determine target periods
            regime_type = island_cfg['regime_type']
            if regime_type == 'current':
                target_periods = [p.name for p in test_periods.get('similar', [])]
            elif regime_type == 'crisis':
                target_periods = [p.name for p in test_periods.get('stress', [])]
            elif regime_type == 'bull':
                target_periods = [p.name for p in self.library.get_periods_by_type(PeriodType.BULL_RUN)[:3]]
            else:
                target_periods = [p.name for p in test_periods.get('diverse', [])]

            # Add some diverse periods to all islands
            diverse_additions = [p.name for p in self.library.get_diverse_test_set(2)]
            target_periods = list(set(target_periods + diverse_additions))

            self.islands[island_cfg['name']] = Island(
                name=island_cfg['name'],
                regime_type=regime_type,
                population=population,
                target_periods=target_periods,
                fitness_weights=island_cfg['fitness_weights'],
            )

        logger.info(f"Initialized {len(self.islands)} islands")

    def _mutate(self, individual: Individual, boost: float = 1.0) -> Individual:
        """
        Mutate an individual.

        Args:
            individual: Individual to mutate
            boost: Mutation strength multiplier

        Returns:
            Mutated individual
        """
        new_genes = individual.genes.copy()
        mutations = []

        for name, (min_val, max_val, step) in self.param_specs.items():
            if random.random() < self.config.mutation_rate * boost:
                # Gaussian mutation
                sigma = (max_val - min_val) * self.config.mutation_sigma * boost
                delta = random.gauss(0, sigma)
                new_val = new_genes[name] + delta

                # Clip to bounds
                new_val = max(min_val, min(max_val, new_val))

                # Snap to step
                if step >= 1:
                    new_val = round(new_val)
                else:
                    n_steps = round((new_val - min_val) / step)
                    new_val = min_val + n_steps * step

                new_genes[name] = new_val
                mutations.append(f"{name}:{delta:+.4f}")

        child = Individual(
            genes=new_genes,
            generation=individual.generation + 1,
            parent_ids=[str(individual.id)],
            mutation_history=mutations,
        )

        return child

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.config.crossover_rate:
            return self._mutate(parent1), self._mutate(parent2)

        # Uniform crossover
        genes1 = {}
        genes2 = {}

        for name in self.param_specs:
            if random.random() < 0.5:
                genes1[name] = parent1.genes[name]
                genes2[name] = parent2.genes[name]
            else:
                genes1[name] = parent2.genes[name]
                genes2[name] = parent1.genes[name]

        child1 = Individual(
            genes=genes1,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[str(parent1.id), str(parent2.id)],
        )
        child2 = Individual(
            genes=genes2,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[str(parent1.id), str(parent2.id)],
        )

        return self._mutate(child1), self._mutate(child2)

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Tournament selection."""
        tournament = random.sample(
            population,
            min(self.config.tournament_size, len(population))
        )
        return max(tournament, key=lambda x: x.fitness)

    def _creative_mutation(self, individual: Individual) -> Individual:
        """
        Apply creative mutation for novel strategy discovery.

        This goes beyond standard Gaussian mutation to explore
        more radical parameter combinations.
        """
        new_genes = individual.genes.copy()
        mutations = []

        # Pick a mutation type
        mutation_type = random.choice([
            'extreme_shift',      # Large shift in one parameter
            'correlated_shift',   # Shift related parameters together
            'boundary_explore',   # Push towards parameter boundaries
            'swap_roles',         # Swap entry/exit type parameters
        ])

        if mutation_type == 'extreme_shift':
            # Large shift in a random parameter
            param = random.choice(list(self.param_specs.keys()))
            min_val, max_val, step = self.param_specs[param]
            # Move by 30-50% of range
            shift = (max_val - min_val) * random.uniform(0.3, 0.5)
            if random.random() < 0.5:
                shift = -shift
            new_val = max(min_val, min(max_val, new_genes[param] + shift))
            new_genes[param] = new_val
            mutations.append(f"extreme:{param}")

        elif mutation_type == 'correlated_shift':
            # Find related parameters (e.g., entry and exit thresholds)
            related_pairs = [
                ('entry_threshold', 'exit_threshold'),
                ('lookback_period', 'max_hold_days'),
                ('stop_loss_pct', 'position_size_pct'),
            ]
            for p1, p2 in related_pairs:
                if p1 in new_genes and p2 in new_genes:
                    # Shift both in same direction
                    min1, max1, _ = self.param_specs[p1]
                    min2, max2, _ = self.param_specs[p2]
                    factor = random.uniform(0.8, 1.2)
                    new_genes[p1] = max(min1, min(max1, new_genes[p1] * factor))
                    new_genes[p2] = max(min2, min(max2, new_genes[p2] * factor))
                    mutations.append(f"correlated:{p1}+{p2}")
                    break

        elif mutation_type == 'boundary_explore':
            # Push a parameter to near-boundary
            param = random.choice(list(self.param_specs.keys()))
            min_val, max_val, step = self.param_specs[param]
            if random.random() < 0.5:
                # Near minimum
                new_genes[param] = min_val + (max_val - min_val) * random.uniform(0, 0.1)
            else:
                # Near maximum
                new_genes[param] = max_val - (max_val - min_val) * random.uniform(0, 0.1)
            mutations.append(f"boundary:{param}")

        elif mutation_type == 'swap_roles':
            # Swap entry/exit or similar paired parameters
            if 'entry_threshold' in new_genes and 'exit_threshold' in new_genes:
                new_genes['entry_threshold'], new_genes['exit_threshold'] = \
                    new_genes['exit_threshold'], new_genes['entry_threshold']
                mutations.append("swap:entry_exit")

        child = Individual(
            genes=new_genes,
            generation=individual.generation + 1,
            parent_ids=[str(individual.id)],
            mutation_history=mutations,
        )

        return child

    def evaluate_individual(
        self,
        individual: Individual,
        strategy_factory: Callable[[Dict], Any],
        data: Dict[str, Any],
        vix_data: Any = None,
        period_names: List[str] = None,
        detailed: bool = False,
        timeout_seconds: int = 60
    ) -> Individual:
        """
        Evaluate an individual's fitness with timeout protection.

        Args:
            individual: Individual to evaluate
            strategy_factory: Function to create strategy from genes
            data: Market data
            vix_data: VIX data
            period_names: Specific periods to test on
            detailed: Calculate detailed metrics
            timeout_seconds: Max time for evaluation (default 60s)

        Returns:
            Individual with fitness scores populated
        """
        def _do_evaluation():
            """Inner evaluation function that can be timed out."""
            # Get test periods
            nonlocal period_names
            if period_names is None:
                test_periods = self.regime_engine.get_ga_test_periods()
                period_names = []
                for category in ['similar', 'stress', 'diverse']:
                    period_names.extend([p.name for p in test_periods.get(category, [])])

            # Run multi-period test - use parallel pool if available
            if self._pool_initialized:
                # Use persistent worker pool for true multi-core parallelism
                result = self.rapid_backtester.run_multi_period_test_parallel(
                    strategy_factory=strategy_factory,
                    genes=individual.genes,
                    period_names=period_names
                )
            else:
                # Sequential path - create strategy in main process
                strategy = strategy_factory(individual.genes)
                result = self.rapid_backtester.run_multi_period_test(
                    strategy,
                    period_names=period_names,
                    data=data,
                    vix_data=vix_data
                )
            return result

        try:
            # Run evaluation with timeout to prevent hangs
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_evaluation)
                try:
                    result = future.result(timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(f"Evaluation timed out after {timeout_seconds}s for individual {individual.id}")
                    individual.fitness = 0.0
                    individual.evaluated = True
                    return individual

            # Extract period-specific scores
            individual.period_scores = {}
            for r in result.results:
                individual.period_scores[r.period_name] = r.sharpe_ratio

            # Calculate component fitnesses
            individual.long_term_fitness = max(0, result.avg_sharpe)
            individual.consistency_score = result.consistency_score

            # Regime fitness (similar periods)
            similar_periods = [p.name for p in self.regime_engine.get_ga_test_periods().get('similar', [])]
            similar_scores = [s for p, s in individual.period_scores.items() if p in similar_periods]
            individual.regime_fitness = np.mean(similar_scores) if similar_scores else 0.0

            # Crisis fitness
            crisis_periods = [p.name for p in self.library.get_crisis_periods()]
            crisis_scores = [s for p, s in individual.period_scores.items() if p in crisis_periods]
            # For crisis, positive sharpe is exceptional, so boost it
            individual.crisis_fitness = np.mean([max(0, s + 0.5) for s in crisis_scores]) if crisis_scores else 0.0

            # Calculate composite fitness
            individual.fitness = (
                individual.long_term_fitness * self.config.long_term_weight +
                individual.regime_fitness * self.config.regime_weight +
                individual.crisis_fitness * self.config.crisis_weight +
                individual.consistency_score * self.config.consistency_weight
            )

            self.total_evaluations += 1
            individual.evaluated = True

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            individual.fitness = 0.0
            individual.evaluated = True

        return individual

    def _evolve_island(
        self,
        island: Island,
        strategy_factory: Callable[[Dict], Any],
        data: Dict[str, Any],
        vix_data: Any = None,
        generations: int = None
    ) -> Island:
        """Evolve a single island."""
        generations = generations or self.config.generations_per_island

        for gen in range(generations):
            # Evaluate population
            for ind in island.population:
                if not ind.evaluated:  # Only evaluate unevaluated individuals
                    self.evaluate_individual(
                        ind,
                        strategy_factory,
                        data,
                        vix_data,
                        island.target_periods
                    )

            # Apply island-specific fitness weights
            for ind in island.population:
                ind.fitness = (
                    ind.long_term_fitness * island.fitness_weights.get('long_term', 0.25) +
                    ind.regime_fitness * island.fitness_weights.get('regime', 0.25) +
                    ind.crisis_fitness * island.fitness_weights.get('crisis', 0.25) +
                    ind.consistency_score * island.fitness_weights.get('consistency', 0.25)
                )

            # Sort by fitness
            island.population.sort(key=lambda x: -x.fitness)

            # Update island stats
            island.best_fitness = island.population[0].fitness
            island.avg_fitness = np.mean([ind.fitness for ind in island.population])

            # Check for improvement
            if island.population[0].fitness > island.best_fitness:
                island.last_improvement_gen = island.generations_evolved

            # Selection and reproduction
            new_population = []

            # Elitism
            new_population.extend(island.population[:self.config.elitism])

            # Generate offspring
            while len(new_population) < len(island.population):
                parent1 = self._tournament_select(island.population)
                parent2 = self._tournament_select(island.population)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([child1, child2])

            # Creative mutation on some individuals
            if random.random() < 0.1:
                idx = random.randint(self.config.elitism, len(new_population) - 1)
                new_population[idx] = self._creative_mutation(new_population[idx])

            island.population = new_population[:len(island.population)]
            island.generations_evolved += 1

        return island

    def _migrate_between_islands(self):
        """Migrate individuals between islands."""
        island_names = list(self.islands.keys())

        for i, source_name in enumerate(island_names):
            target_name = island_names[(i + 1) % len(island_names)]

            source = self.islands[source_name]
            target = self.islands[target_name]

            # Send best individuals from source
            migrants = source.population[:self.config.migration_size]

            # Replace worst in target
            for j, migrant in enumerate(migrants):
                # Create a copy for the target island
                migrant_copy = Individual(
                    genes=migrant.genes.copy(),
                    generation=migrant.generation,
                    fitness=0,  # Will be re-evaluated
                )
                target.population[-(j + 1)] = migrant_copy

        logger.debug("Migration completed between islands")

    def evolve(
        self,
        strategy_factory: Callable[[Dict], Any],
        data: Dict[str, Any],
        vix_data: Any = None,
        current_conditions: Dict[str, float] = None,
        generations: int = None,
        callback: Callable[[int, Individual], None] = None
    ) -> Individual:
        """
        Run the adaptive GA evolution.

        Args:
            strategy_factory: Function to create strategy from genes dict
            data: Market data dict {symbol: DataFrame}
            vix_data: VIX DataFrame
            current_conditions: Dict with 'vix', 'trend', 'correlation'
            generations: Number of generations (uses config default if None)
            callback: Optional callback(generation, best_individual)

        Returns:
            Best individual found
        """
        start_time = time.time()
        generations = generations or self.config.generations_per_session

        # Update current fingerprint
        if current_conditions:
            # Use provided conditions
            self._current_fingerprint = RegimeFingerprint(
                vix_level=current_conditions.get('vix', 15),
                vix_percentile=50,
                vix_trend=0,
                trend_direction=current_conditions.get('trend', 0),
                trend_strength=abs(current_conditions.get('trend', 0)),
                momentum_breadth=50,
                realized_vol=15,
                vol_regime='normal',
                vol_trend=0,
                correlation_level=current_conditions.get('correlation', 0.5),
                correlation_trend=0,
                sector_leadership='mixed',
                sector_dispersion=0.5,
                credit_spread_z=0,
                term_structure='contango',
                overall_regime='transition',
                regime_confidence=0.5,
            )
        else:
            self._current_fingerprint = self.regime_engine.get_current_fingerprint()

        # Initialize islands if needed
        if not self.islands:
            self._initialize_islands()

        # Cache data in rapid backtester
        self.rapid_backtester.cache_data(data)

        logger.info(f"Starting evolution for {generations} generations")
        logger.info(f"Current regime: {self._current_fingerprint.overall_regime}")

        # Main evolution loop
        for gen in range(generations):
            self.current_generation = gen

            # Evolve each island
            for island_name, island in self.islands.items():
                # More generations for current-regime island
                extra_gens = 2 if island.regime_type == 'current' else 0
                self._evolve_island(
                    island,
                    strategy_factory,
                    data,
                    vix_data,
                    self.config.generations_per_island + extra_gens
                )

            # Migration between islands
            if (gen + 1) % self.config.migration_interval == 0:
                self._migrate_between_islands()

            # Find global best
            all_individuals = []
            for island in self.islands.values():
                all_individuals.extend(island.population)

            current_best = max(all_individuals, key=lambda x: x.fitness)

            if self.global_best is None or current_best.fitness > self.global_best.fitness:
                self.global_best = current_best
                self.stagnation_counter = 0
                logger.info(f"Gen {gen}: New best fitness {current_best.fitness:.4f}")
            else:
                self.stagnation_counter += 1

            # Adaptive mutation if stagnating
            if self.config.adaptive_mutation and self.stagnation_counter >= self.config.stagnation_threshold:
                logger.info("Boosting mutation due to stagnation")
                # Apply extra mutation to some individuals
                for island in self.islands.values():
                    for i in range(self.config.elitism, len(island.population)):
                        if random.random() < 0.3:
                            island.population[i] = self._mutate(
                                island.population[i],
                                boost=self.config.mutation_boost
                            )

            # Record history
            self.generation_history.append({
                'generation': gen,
                'best_fitness': self.global_best.fitness,
                'avg_fitness': np.mean([ind.fitness for ind in all_individuals]),
                'island_stats': {
                    name: {'best': isl.best_fitness, 'avg': isl.avg_fitness}
                    for name, isl in self.islands.items()
                },
            })

            # Callback
            if callback:
                callback(gen, self.global_best)

        elapsed = time.time() - start_time
        logger.info(f"Evolution completed in {elapsed:.1f}s")
        logger.info(f"Best fitness: {self.global_best.fitness:.4f}")
        logger.info(f"Total evaluations: {self.total_evaluations}")

        return self.global_best

    def run_rapid_generations(
        self,
        strategy_factory: Callable[[Dict], Any],
        data: Dict[str, Any],
        vix_data: Any = None,
        n_generations: int = None,
        period_names: List[str] = None
    ) -> Individual:
        """
        Run rapid generations on short periods only.

        Useful for quick exploration of parameter space.

        Args:
            strategy_factory: Strategy factory function
            data: Market data
            vix_data: VIX data
            n_generations: Number of rapid generations
            period_names: Specific short periods to test

        Returns:
            Best individual from rapid evolution
        """
        n_generations = n_generations or self.config.rapid_generations

        # Get short periods
        if period_names is None:
            short_periods = self.library.get_short_periods(
                max_days=self.config.rapid_period_limit
            )
            period_names = [p.name for p in short_periods[:5]]

        logger.info(f"Running {n_generations} rapid generations on {len(period_names)} short periods")

        # Create a temporary small population
        rapid_population = [
            self._create_random_individual(0)
            for _ in range(20)
        ]

        # If we have existing islands, seed with their best
        if self.islands:
            for island in self.islands.values():
                if island.population:
                    best = max(island.population, key=lambda x: x.fitness)
                    rapid_population[0] = Individual(
                        genes=best.genes.copy(),
                        generation=0,
                    )

        best_rapid = None

        for gen in range(n_generations):
            # Evaluate
            for ind in rapid_population:
                if ind.fitness == 0:
                    self.evaluate_individual(
                        ind,
                        strategy_factory,
                        data,
                        vix_data,
                        period_names
                    )

            # Sort
            rapid_population.sort(key=lambda x: -x.fitness)

            if best_rapid is None or rapid_population[0].fitness > best_rapid.fitness:
                best_rapid = rapid_population[0]

            # Reproduce
            new_pop = rapid_population[:2]  # Elitism
            while len(new_pop) < len(rapid_population):
                p1 = self._tournament_select(rapid_population)
                p2 = self._tournament_select(rapid_population)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([c1, c2])

            rapid_population = new_pop[:20]

        logger.info(f"Rapid evolution complete. Best fitness: {best_rapid.fitness:.4f}")

        return best_rapid

    def get_best_for_regime(self, regime: str) -> Optional[Individual]:
        """Get best individual for a specific regime."""
        for island in self.islands.values():
            if island.regime_type == regime and island.population:
                return max(island.population, key=lambda x: x.fitness)
        return None

    def get_ensemble(self, n: int = 3) -> List[Individual]:
        """
        Get diverse ensemble of top individuals.

        Returns individuals with different strengths.
        """
        ensemble = []

        # Best from each island
        for island in self.islands.values():
            if island.population:
                best = max(island.population, key=lambda x: x.fitness)
                ensemble.append(best)

        # Sort by fitness and deduplicate by genes
        ensemble.sort(key=lambda x: -x.fitness)
        seen_genes = set()
        unique_ensemble = []
        for ind in ensemble:
            gene_key = tuple(sorted(ind.genes.items()))
            if gene_key not in seen_genes:
                unique_ensemble.append(ind)
                seen_genes.add(gene_key)

        return unique_ensemble[:n]

    def print_status(self):
        """Print current optimization status."""
        print("\n" + "=" * 60)
        print("ADAPTIVE GA OPTIMIZER STATUS")
        print("=" * 60)

        print(f"\nGeneration: {self.current_generation}")
        print(f"Total evaluations: {self.total_evaluations}")
        print(f"Stagnation counter: {self.stagnation_counter}")

        if self.global_best:
            print(f"\nGLOBAL BEST:")
            print(f"  Fitness: {self.global_best.fitness:.4f}")
            print(f"  Long-term: {self.global_best.long_term_fitness:.3f}")
            print(f"  Regime: {self.global_best.regime_fitness:.3f}")
            print(f"  Crisis: {self.global_best.crisis_fitness:.3f}")
            print(f"  Consistency: {self.global_best.consistency_score:.3f}")
            print(f"  Genes: {self.global_best.genes}")

        print("\nISLAND STATUS:")
        print("-" * 40)
        for name, island in self.islands.items():
            print(f"  {name}:")
            print(f"    Best: {island.best_fitness:.4f}")
            print(f"    Avg: {island.avg_fitness:.4f}")
            print(f"    Generations: {island.generations_evolved}")

        if self.generation_history:
            print("\nFITNESS HISTORY (last 5):")
            for entry in self.generation_history[-5:]:
                print(f"  Gen {entry['generation']}: "
                      f"best={entry['best_fitness']:.4f}, "
                      f"avg={entry['avg_fitness']:.4f}")

        print("=" * 60 + "\n")



    # =========================================================================
    # Persistent Pool Support for True Multi-Core Parallelism
    # =========================================================================

    def init_parallel_pool(
        self,
        data: Dict[str, pd.DataFrame],
        vix_data: pd.DataFrame = None
    ):
        """
        Initialize shared memory and persistent worker pool on the RapidBacktester.

        Call this once before evolve() for much faster parallel evaluation.
        Workers stay alive across all generations.

        Args:
            data: Market data dict (symbol -> DataFrame)
            vix_data: Optional VIX DataFrame
        """
        if self._pool_initialized:
            return

        # Use RapidBacktester's parallel pool
        self.rapid_backtester.init_parallel_pool(
            data=data,
            vix_data=vix_data,
            n_workers=self.config.n_workers
        )
        self._pool_initialized = True

        logger.info(f"AdaptiveGAOptimizer parallel pool initialized with {self.config.n_workers} workers")

    def cleanup_parallel_pool(self):
        """Clean up shared memory and worker pool."""
        if self._pool_initialized:
            self.rapid_backtester.cleanup_parallel_pool()
            self._pool_initialized = False
            logger.info("AdaptiveGAOptimizer parallel pool cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup_parallel_pool()

# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    print("\n" + "=" * 60)
    print("ADAPTIVE GA OPTIMIZER DEMO")
    print("=" * 60)

    # Create optimizer
    config = AdaptiveGAConfig(
        total_population=40,
        n_islands=4,
        generations_per_session=5,
    )

    optimizer = AdaptiveGAOptimizer(config=config)

    print(f"\nConfiguration:")
    print(f"  Total population: {config.total_population}")
    print(f"  Islands: {config.n_islands}")
    print(f"  Generations/session: {config.generations_per_session}")

    print(f"\nParameter specs: {list(optimizer.param_specs.keys())}")

    # Create a dummy strategy factory for demo
    def dummy_factory(genes: Dict) -> Any:
        class DummyStrategy:
            def __init__(self, g):
                self.genes = g
                self.name = "demo"

            def generate_signals(self, data, vix=None):
                return []
        return DummyStrategy(genes)

    print("\nTo run evolution:")
    print("  best = optimizer.evolve(strategy_factory, data)")
    print("  optimizer.print_status()")
