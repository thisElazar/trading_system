"""
Island Model for Genetic Programming
=====================================
Implements island-based parallel evolution to maintain population diversity
and prevent premature convergence.

Key concepts:
- Multiple independent subpopulations (islands) evolve in parallel
- Periodic migration exchanges individuals between islands
- Different islands can explore different regions of the solution space
- Best solutions from each island are preserved

Migration topologies:
- Ring: Each island sends migrants to the next island in a ring
- Random: Random pairs of islands exchange migrants
- Fully connected: All islands can exchange with all others
"""

import random
import logging
import copy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .config import EvolutionConfig, IslandConfig
from .strategy_genome import StrategyGenome, GenomeFactory
from .strategy_compiler import StrategyCompiler
from .multi_objective import (
    FitnessVector, calculate_fitness_vector,
    non_dominated_sort, crowding_distance
)
from .novelty_search import (
    BehaviorVector, NoveltyArchive, extract_behavior_vector,
    calculate_population_diversity
)
from .diversity_metrics import (
    DiversityMonitor, DiversityThresholds, calculate_genotype_diversity
)
from .map_elites import MAPElitesGrid, create_default_grid

logger = logging.getLogger(__name__)


class MigrationTopology(Enum):
    """Migration topology options."""
    RING = "ring"           # i -> i+1 (circular)
    RANDOM = "random"       # Random pairs
    FULL = "full"           # All-to-all


@dataclass
class Island:
    """Represents a single island (subpopulation)."""
    island_id: int
    population: List[StrategyGenome] = field(default_factory=list)
    fitness_cache: Dict[str, FitnessVector] = field(default_factory=dict)
    behavior_cache: Dict[str, BehaviorVector] = field(default_factory=dict)
    novelty_archive: NoveltyArchive = None

    # Island-specific parameters
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    max_tree_depth: int = 5

    # Statistics
    generations_evolved: int = 0
    best_fitness: float = -float('inf')
    best_genome_id: str = ""
    diversity: float = 0.0

    def __post_init__(self):
        if self.novelty_archive is None:
            self.novelty_archive = NoveltyArchive(k_neighbors=5, archive_size=50)


class IslandEvolutionEngine:
    """
    Island-based evolution engine for strategy discovery.

    Maintains multiple independent populations that periodically
    exchange individuals, preventing premature convergence while
    still allowing good solutions to spread.
    """

    def __init__(
        self,
        config: EvolutionConfig = None,
        island_config: IslandConfig = None,
        backtester = None,
        use_fast_backtester: bool = False,
        enable_diversity_monitor: bool = True,
        enable_map_elites: bool = True
    ):
        """
        Initialize island evolution engine.

        Args:
            config: Base evolution configuration
            island_config: Island-specific configuration
            backtester: Backtester instance
            use_fast_backtester: Use FastBacktester for speed
            enable_diversity_monitor: Enable auto-intervention for diversity
            enable_map_elites: Enable MAP-Elites grid
        """
        self.config = config or EvolutionConfig()
        self.island_config = island_config or IslandConfig()

        # Override population size in config
        self.config.population_size = self.island_config.population_per_island

        # Core components
        self.factory = GenomeFactory(self.config)
        self.compiler = StrategyCompiler(self.config)

        # Backtester
        if backtester:
            self.backtester = backtester
        elif use_fast_backtester:
            from research.backtester_fast import FastBacktester
            self.backtester = FastBacktester(initial_capital=100000, slippage_bps=15)
        else:
            from research.backtester import Backtester
            self.backtester = Backtester(initial_capital=100000, cost_model='conservative')

        # Islands
        self.islands: List[Island] = []

        # Global state
        self.current_generation = 0
        self.total_strategies_evaluated = 0
        self.global_pareto_front: List[StrategyGenome] = []
        self.global_best_fitness = -float('inf')
        self.global_best_genome: Optional[StrategyGenome] = None

        # Data
        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._vix_data: Optional[pd.DataFrame] = None

        # Diversity monitoring (global across all islands)
        self.enable_diversity_monitor = enable_diversity_monitor
        if enable_diversity_monitor:
            self.diversity_monitor = DiversityMonitor(DiversityThresholds())
            logger.info("Diversity monitoring enabled with auto-intervention")
        else:
            self.diversity_monitor = None

        # MAP-Elites grid (global, populated from all islands)
        self.enable_map_elites = enable_map_elites
        if enable_map_elites:
            self.map_elites = create_default_grid()
            logger.info(f"MAP-Elites grid enabled: {self.map_elites}")
        else:
            self.map_elites = None

        # Initialize islands
        self._initialize_islands()

    def _initialize_islands(self):
        """Create and configure islands with varied parameters."""
        logger.info(f"Initializing {self.island_config.num_islands} islands")

        for i in range(self.island_config.num_islands):
            island = Island(island_id=i)

            # Vary parameters per island for diversity
            if self.island_config.vary_mutation_rate:
                # Mutation rates from 0.1 to 0.4
                island.mutation_rate = 0.1 + (i / max(1, self.island_config.num_islands - 1)) * 0.3

            if self.island_config.vary_tree_depth:
                # Tree depths from 3 to 6
                island.max_tree_depth = 3 + (i % 4)

            island.crossover_rate = self.config.crossover_rate

            self.islands.append(island)
            logger.debug(f"Island {i}: mutation={island.mutation_rate:.2f}, depth={island.max_tree_depth}")

    def initialize_populations(self):
        """Initialize population for each island."""
        for island in self.islands:
            # Create factory with island-specific depth
            island_config = copy.copy(self.config)
            island_config.max_tree_depth = island.max_tree_depth
            island_factory = GenomeFactory(island_config)

            island.population = [
                island_factory.create_random_genome(generation=0)
                for _ in range(self.island_config.population_per_island)
            ]

            logger.debug(f"Island {island.island_id}: initialized {len(island.population)} individuals")

    def load_data(self, data: Dict[str, pd.DataFrame], vix_data: pd.DataFrame = None):
        """Load market data for backtesting."""
        self._data = data
        self._vix_data = vix_data
        logger.info(f"Loaded data for {len(data)} symbols")

    def evaluate_genome(self, genome: StrategyGenome, island: Island) -> Tuple[FitnessVector, BehaviorVector]:
        """
        Evaluate a genome within an island context.

        Uses island's caches to avoid redundant evaluation.
        """
        # Check island cache first
        if genome.genome_id in island.fitness_cache:
            return island.fitness_cache[genome.genome_id], island.behavior_cache[genome.genome_id]

        # Compile and backtest
        strategy = self.compiler.compile(genome)

        try:
            result = self.backtester.run(
                strategy=strategy,
                data=self._data,
                vix_data=self._vix_data
            )
        except Exception as e:
            logger.debug(f"Backtest failed for {genome.genome_id}: {e}")
            from research.backtester import BacktestResult
            result = BacktestResult(
                run_id=genome.genome_id,
                strategy=strategy.name,
                start_date="",
                end_date=""
            )

        # Extract behavior and calculate novelty
        behavior = extract_behavior_vector(result)
        novelty = island.novelty_archive.calculate_novelty(behavior)

        # Calculate fitness
        fitness = calculate_fitness_vector(
            result=result,
            novelty_score=novelty,
            total_trials=self.total_strategies_evaluated + 1
        )

        # Cache results
        island.fitness_cache[genome.genome_id] = fitness
        island.behavior_cache[genome.genome_id] = behavior

        # Store on genome
        genome.backtest_result = result
        genome.fitness_values = fitness.to_tuple()

        self.total_strategies_evaluated += 1

        return fitness, behavior

    def evaluate_island(self, island: Island):
        """Evaluate all individuals in an island."""
        for genome in island.population:
            if genome.genome_id not in island.fitness_cache:
                self.evaluate_genome(genome, island)

    def evolve_island(self, island: Island):
        """Evolve a single island for one generation."""
        # Evaluate current population
        self.evaluate_island(island)

        # Update island statistics
        self._update_island_stats(island)

        # Update novelty archive
        for genome in island.population:
            if genome.genome_id in island.behavior_cache:
                behavior = island.behavior_cache[genome.genome_id]
                fitness = island.fitness_cache[genome.genome_id]
                novelty = island.novelty_archive.calculate_novelty(behavior)
                island.novelty_archive.maybe_add(behavior, novelty, fitness.sortino)

        # Select parents
        parents = self._select_parents(island)

        # Create offspring
        offspring = self._create_offspring(island, parents)

        # Replace population
        island.population = offspring
        island.generations_evolved += 1

    def _update_island_stats(self, island: Island):
        """Update island statistics after evaluation."""
        fitnesses = [
            island.fitness_cache.get(g.genome_id)
            for g in island.population
            if g.genome_id in island.fitness_cache
        ]

        if fitnesses:
            best_fitness = max(f.sortino for f in fitnesses)
            if best_fitness > island.best_fitness:
                island.best_fitness = best_fitness
                best_genome = max(
                    island.population,
                    key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino
                )
                island.best_genome_id = best_genome.genome_id

        # Calculate diversity
        behaviors = [
            island.behavior_cache.get(g.genome_id)
            for g in island.population
            if g.genome_id in island.behavior_cache
        ]
        island.diversity = calculate_population_diversity([b for b in behaviors if b])

    def _select_parents(self, island: Island, n_parents: int = None) -> List[StrategyGenome]:
        """Select parents using tournament selection."""
        n_parents = n_parents or len(island.population)

        parents = []
        for _ in range(n_parents):
            # Tournament selection
            tournament_size = min(3, len(island.population))
            tournament = random.sample(island.population, tournament_size)

            winner = max(
                tournament,
                key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino
            )
            parents.append(winner)

        return parents

    def _create_offspring(self, island: Island, parents: List[StrategyGenome]) -> List[StrategyGenome]:
        """Create offspring through crossover and mutation."""
        offspring = []

        # Elitism: keep best individuals
        elite_count = max(1, int(len(island.population) * 0.1))
        elite = sorted(
            island.population,
            key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
            reverse=True
        )[:elite_count]
        offspring.extend([copy.deepcopy(g) for g in elite])

        # Get mutation rate (potentially boosted if diversity intervention active)
        mutation_rate = island.mutation_rate
        if self.diversity_monitor is not None:
            multiplier = self.diversity_monitor.get_current_mutation_rate_multiplier()
            if multiplier > 1.0:
                mutation_rate = min(0.8, mutation_rate * multiplier)  # Cap at 80%
                logger.debug(f"Island {island.island_id} mutation boosted: {island.mutation_rate:.2f} -> {mutation_rate:.2f}")

        # Fill rest with offspring
        while len(offspring) < self.island_config.population_per_island:
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            # Crossover
            if random.random() < island.crossover_rate:
                child1, child2 = self.factory.crossover(p1, p2, self.current_generation)
            else:
                child1 = copy.deepcopy(p1)
                child2 = copy.deepcopy(p2)
                child1.generation = self.current_generation
                child2.generation = self.current_generation

            # Mutation (using potentially boosted rate)
            if random.random() < mutation_rate:
                child1 = self.factory.mutate(child1, self.current_generation)
            if random.random() < mutation_rate:
                child2 = self.factory.mutate(child2, self.current_generation)

            offspring.append(child1)
            if len(offspring) < self.island_config.population_per_island:
                offspring.append(child2)

        return offspring[:self.island_config.population_per_island]

    def migrate(self):
        """Perform migration between islands based on topology."""
        n_migrants = max(1, int(self.island_config.population_per_island * self.island_config.migration_rate))

        logger.info(f"Migration: {n_migrants} individuals per island")

        topology = self.island_config.topology
        if topology == "ring" or topology == MigrationTopology.RING:
            self._migrate_ring(n_migrants)
        elif topology == "random" or topology == MigrationTopology.RANDOM:
            self._migrate_random(n_migrants)
        else:  # full
            self._migrate_full(n_migrants)

    def _migrate_ring(self, n_migrants: int):
        """Ring topology: island i sends to island (i+1) % n."""
        n_islands = len(self.islands)

        # Collect emigrants from each island
        emigrants = []
        for island in self.islands:
            # Select best individuals to emigrate
            sorted_pop = sorted(
                island.population,
                key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                reverse=True
            )
            emigrants.append([copy.deepcopy(g) for g in sorted_pop[:n_migrants]])

        # Perform migration
        for i, island in enumerate(self.islands):
            source_island = (i - 1) % n_islands
            incoming = emigrants[source_island]

            # Replace worst individuals with immigrants
            island.population = sorted(
                island.population,
                key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                reverse=True
            )
            island.population = island.population[:-n_migrants] + incoming

            # Copy fitness cache entries for immigrants
            source = self.islands[source_island]
            for immigrant in incoming:
                if immigrant.genome_id in source.fitness_cache:
                    island.fitness_cache[immigrant.genome_id] = source.fitness_cache[immigrant.genome_id]
                    island.behavior_cache[immigrant.genome_id] = source.behavior_cache[immigrant.genome_id]

    def _migrate_random(self, n_migrants: int):
        """Random topology: random pairs exchange migrants."""
        island_indices = list(range(len(self.islands)))
        random.shuffle(island_indices)

        # Pair up islands
        for i in range(0, len(island_indices) - 1, 2):
            island_a = self.islands[island_indices[i]]
            island_b = self.islands[island_indices[i + 1]]

            # Select best from each
            best_a = sorted(
                island_a.population,
                key=lambda g: island_a.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                reverse=True
            )[:n_migrants]

            best_b = sorted(
                island_b.population,
                key=lambda g: island_b.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                reverse=True
            )[:n_migrants]

            # Exchange
            for j in range(n_migrants):
                # Remove worst from each
                island_a.population = sorted(
                    island_a.population,
                    key=lambda g: island_a.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                    reverse=True
                )[:-1]
                island_b.population = sorted(
                    island_b.population,
                    key=lambda g: island_b.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                    reverse=True
                )[:-1]

                # Add immigrant
                island_a.population.append(copy.deepcopy(best_b[j]))
                island_b.population.append(copy.deepcopy(best_a[j]))

    def _migrate_full(self, n_migrants: int):
        """Fully connected: best from each island goes to all others."""
        # Collect best from each island
        best_per_island = []
        for island in self.islands:
            best = max(
                island.population,
                key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino
            )
            best_per_island.append((island.island_id, copy.deepcopy(best)))

        # Each island receives best from all other islands
        for island in self.islands:
            immigrants = [
                g for island_id, g in best_per_island
                if island_id != island.island_id
            ][:n_migrants]

            # Remove worst
            island.population = sorted(
                island.population,
                key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                reverse=True
            )[:-len(immigrants)]

            # Add immigrants
            island.population.extend(immigrants)

    def evolve_generation(self):
        """Evolve all islands for one generation."""
        self.current_generation += 1

        # Evolve each island
        for island in self.islands:
            self.evolve_island(island)

        # Periodic migration
        if self.current_generation % self.island_config.migration_interval == 0:
            self.migrate()

        # Update global best
        self._update_global_best()

        # Update global Pareto front
        self._update_global_pareto_front()

        # Update MAP-Elites grid with all evaluated genomes
        if self.map_elites is not None:
            for island in self.islands:
                for genome in island.population:
                    if genome.genome_id in island.fitness_cache and genome.genome_id in island.behavior_cache:
                        fitness = island.fitness_cache[genome.genome_id]
                        behavior = island.behavior_cache[genome.genome_id]
                        self.map_elites.maybe_add(
                            genome_id=genome.genome_id,
                            fitness=fitness.sortino,
                            fitness_vector=fitness.to_tuple(),
                            behavior=behavior.to_array(),
                            generation=self.current_generation
                        )

        # Global diversity monitoring and auto-intervention
        if self.diversity_monitor is not None:
            # Aggregate genomes and metrics across all islands
            all_genomes = []
            all_behaviors = []
            best_fitness = -float('inf')

            for island in self.islands:
                all_genomes.extend(island.population)
                for genome in island.population:
                    if genome.genome_id in island.behavior_cache:
                        all_behaviors.append(island.behavior_cache[genome.genome_id])
                    if genome.genome_id in island.fitness_cache:
                        fitness = island.fitness_cache[genome.genome_id].sortino
                        if fitness > best_fitness:
                            best_fitness = fitness

            phenotype_diversity = calculate_population_diversity(all_behaviors)
            archive_diversity = np.mean([
                island.novelty_archive.get_archive_diversity()
                for island in self.islands
            ])

            should_intervene, reason = self.diversity_monitor.update(
                generation=self.current_generation,
                genomes=all_genomes,
                phenotype_diversity=phenotype_diversity,
                archive_diversity=archive_diversity,
                best_fitness=best_fitness
            )

            if should_intervene:
                logger.warning(f"Global diversity intervention triggered: {reason}")
                self._inject_diversity_all_islands()

    def _inject_diversity_all_islands(self):
        """Inject random individuals into all islands to combat convergence."""
        ratio = self.diversity_monitor.get_injection_ratio() if self.diversity_monitor else 0.3
        n_replace_per_island = max(1, int(self.island_config.population_per_island * ratio))

        logger.info(f"Injecting {n_replace_per_island} random individuals per island")

        for island in self.islands:
            # Sort by fitness, replace worst
            sorted_pop = sorted(
                island.population,
                key=lambda g: island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino,
                reverse=True
            )

            # Create island-specific factory for depth
            island_config = copy.copy(self.config)
            island_config.max_tree_depth = island.max_tree_depth
            island_factory = GenomeFactory(island_config)

            # Keep best, replace worst with random
            island.population = sorted_pop[:-n_replace_per_island]
            island.population.extend([
                island_factory.create_random_genome(self.current_generation)
                for _ in range(n_replace_per_island)
            ])

    def _update_global_best(self):
        """Track the best genome across all islands."""
        for island in self.islands:
            if island.best_fitness > self.global_best_fitness:
                self.global_best_fitness = island.best_fitness
                # Find the actual genome
                for genome in island.population:
                    if genome.genome_id == island.best_genome_id:
                        self.global_best_genome = copy.deepcopy(genome)
                        break

    def _update_global_pareto_front(self):
        """Maintain global Pareto front across all islands."""
        # Collect all evaluated genomes
        all_genomes = []
        all_fitness = []

        for island in self.islands:
            for genome in island.population:
                if genome.genome_id in island.fitness_cache:
                    all_genomes.append(genome)
                    all_fitness.append(island.fitness_cache[genome.genome_id])

        if not all_fitness:
            return

        # Non-dominated sort
        fronts = non_dominated_sort(all_fitness)

        if fronts:
            self.global_pareto_front = [all_genomes[i] for i in fronts[0]]

    def run(self, generations: int = None, max_time_seconds: float = None):
        """
        Run island evolution.

        Args:
            generations: Number of generations to run
            max_time_seconds: Maximum time in seconds (overrides generations)
        """
        import time

        generations = generations or self.config.generations_per_session
        start_time = time.time()

        logger.info(f"Starting island evolution: {self.island_config.num_islands} islands, "
                   f"{self.island_config.population_per_island} per island")

        # Initialize if needed
        if not any(island.population for island in self.islands):
            self.initialize_populations()

        # Check data
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Evolution loop
        while True:
            # Check termination conditions
            if max_time_seconds and (time.time() - start_time) >= max_time_seconds:
                logger.info("Time limit reached")
                break
            if generations and self.current_generation >= generations:
                logger.info("Generation limit reached")
                break

            self.evolve_generation()

            # Log progress
            if self.current_generation % 5 == 0 or self.current_generation <= 3:
                self._log_progress()

        logger.info(f"Evolution complete: {self.current_generation} generations, "
                   f"{self.total_strategies_evaluated} strategies evaluated")

    def _log_progress(self):
        """Log evolution progress."""
        island_stats = []
        for island in self.islands:
            fitnesses = [
                island.fitness_cache.get(g.genome_id, FitnessVector(-999, 0, 0, 0, 0)).sortino
                for g in island.population
            ]
            island_stats.append({
                'id': island.island_id,
                'best': max(fitnesses) if fitnesses else -999,
                'avg': np.mean(fitnesses) if fitnesses else -999,
                'div': island.diversity
            })

        # Summary stats
        all_best = [s['best'] for s in island_stats]
        all_div = [s['div'] for s in island_stats]

        # Build log parts
        log_parts = [
            f"Gen {self.current_generation}:",
            f"Best=[{min(all_best):.2f}/{max(all_best):.2f}]",
            f"Div=[{min(all_div):.3f}/{max(all_div):.3f}]",
            f"Pareto={len(self.global_pareto_front)}"
        ]

        # Add MAP-Elites coverage if enabled
        if self.map_elites is not None:
            log_parts.append(f"MAP={self.map_elites.get_coverage():.1%}")

        # Add genotype diversity if monitor enabled
        if self.diversity_monitor is not None:
            latest = self.diversity_monitor.get_latest_metrics()
            if latest:
                log_parts.append(f"Geno={latest.genotype_metrics.genotype_diversity:.3f}")
                if self.diversity_monitor.total_interventions > 0:
                    log_parts.append(f"Intv={self.diversity_monitor.total_interventions}")

        logger.info(" ".join(log_parts))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evolution state."""
        return {
            'generation': self.current_generation,
            'total_evaluated': self.total_strategies_evaluated,
            'global_best_fitness': self.global_best_fitness,
            'global_pareto_size': len(self.global_pareto_front),
            'islands': [
                {
                    'id': island.island_id,
                    'best_fitness': island.best_fitness,
                    'diversity': island.diversity,
                    'generations': island.generations_evolved,
                    'mutation_rate': island.mutation_rate,
                    'tree_depth': island.max_tree_depth
                }
                for island in self.islands
            ]
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    print("Testing Island Evolution Engine...")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    test_data = {}

    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        close = 100 + np.cumsum(np.random.randn(200) * 2)
        test_data[symbol] = pd.DataFrame({
            'open': close + np.random.randn(200) * 0.5,
            'high': close + abs(np.random.randn(200)) * 1.5,
            'low': close - abs(np.random.randn(200)) * 1.5,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)

    # Create engine
    config = EvolutionConfig(
        population_size=10,  # Will be overridden by island_config
        generations_per_session=20
    )

    island_config = IslandConfig(
        num_islands=4,
        population_per_island=10,
        migration_interval=5,
        migration_rate=0.2
    )

    engine = IslandEvolutionEngine(
        config=config,
        island_config=island_config,
        use_fast_backtester=False
    )

    engine.load_data(test_data)
    engine.initialize_populations()

    print(f"\nIslands: {len(engine.islands)}")
    for island in engine.islands:
        print(f"  Island {island.island_id}: mutation={island.mutation_rate:.2f}, depth={island.max_tree_depth}")

    # Run for a few generations
    print("\nRunning evolution...")
    engine.run(generations=15)

    # Summary
    summary = engine.get_summary()
    print(f"\nFinal Summary:")
    print(f"  Generations: {summary['generation']}")
    print(f"  Strategies evaluated: {summary['total_evaluated']}")
    print(f"  Global best fitness: {summary['global_best_fitness']:.3f}")
    print(f"  Pareto front size: {summary['global_pareto_size']}")

    print("\nTest complete!")
