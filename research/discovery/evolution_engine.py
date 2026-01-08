"""
Evolution Engine
================
Core evolutionary loop for autonomous strategy discovery.

Integrates:
- Genetic programming (tree-based genomes)
- Multi-objective optimization (NSGA-II)
- Novelty search (behavioral diversity)
- Continuous checkpointing (overnight operation)
"""

import random
import logging
import json
import signal
import copy
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Memory thresholds for Pi (4GB RAM)
MEMORY_WARNING_MB = 600
MEMORY_CRITICAL_MB = 300
MAX_CACHE_SIZE = 200  # Max fitness cache entries
MAX_PARETO_FRONT_SIZE = 50
# Shared memory and persistent pool for efficient parallelism
from .shared_data import SharedDataManager
from .parallel_pool import PersistentWorkerPool

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .config import EvolutionConfig
from .strategy_genome import StrategyGenome, GenomeFactory
from .strategy_compiler import StrategyCompiler, EvolvedStrategy
from .multi_objective import (
    FitnessVector, calculate_fitness_vector,
    non_dominated_sort, crowding_distance, apply_constraints
)
from .novelty_search import (
    BehaviorVector, NoveltyArchive, extract_behavior_vector,
    calculate_population_diversity
)
from .diversity_metrics import (
    DiversityMonitor, DiversityThresholds, calculate_genotype_diversity,
    log_diversity_metrics
)
from .map_elites import MAPElitesGrid, MapElitesConfig, create_default_grid
from research.backtester import Backtester, BacktestResult
from research.backtester_fast import FastBacktester
from data.storage.db_manager import get_db

# Optional portfolio fitness (for portfolio-level optimization)
try:
    from .portfolio_fitness import PortfolioFitnessEvaluator, CompositeScore
    HAS_PORTFOLIO_FITNESS = True
except ImportError:
    PortfolioFitnessEvaluator = None
    CompositeScore = None
    HAS_PORTFOLIO_FITNESS = False

logger = logging.getLogger(__name__)


# Worker function for parallel genome evaluation (must be at module level for pickling)
def _evaluate_genome_worker(args):
    """
    Worker function to evaluate a genome in a separate process.

    Args:
        args: Tuple of (genome_data, data, vix_data, config_dict)

    Returns:
        Dict with evaluation results
    """
    genome_data, data, vix_data, config_dict = args

    try:
        # Recreate necessary objects in worker process
        from .config import EvolutionConfig
        from .strategy_genome import GenomeFactory
        from .strategy_compiler import StrategyCompiler
        from .multi_objective import calculate_fitness_vector, FitnessVector
        from .novelty_search import extract_behavior_vector
        from research.backtester import Backtester, BacktestResult

        config = EvolutionConfig(**config_dict)
        factory = GenomeFactory(config)
        compiler = StrategyCompiler(config)
        backtester = Backtester(initial_capital=100000, cost_model="conservative")

        # Deserialize genome
        genome = factory.deserialize_genome(genome_data)

        # Compile and run backtest
        strategy = compiler.compile(genome)
        try:
            result = backtester.run(strategy=strategy, data=data, vix_data=vix_data)
        except Exception as e:
            result = BacktestResult(
                run_id=genome.genome_id,
                strategy=strategy.name if strategy else "unknown",
                start_date="",
                end_date=""
            )

        # Extract behavior vector
        behavior = extract_behavior_vector(result)

        return {
            "success": True,
            "genome_id": genome.genome_id,
            "behavior_array": behavior.to_array().tolist(),
            "result": result,
        }

    except Exception as e:
        return {
            "success": False,
            "genome_id": genome_data.get("genome_id", "unknown") if isinstance(genome_data, dict) else "unknown",
            "error": str(e)
        }


@dataclass
class EvolutionState:
    """State of evolution for checkpointing."""
    generation: int
    population_size: int
    pareto_front_size: int
    best_sortino: float
    best_drawdown: float
    avg_novelty: float
    diversity: float
    total_strategies_evaluated: int
    strategies_promoted: int
    timestamp: str


class EvolutionEngine:
    """
    Autonomous evolution engine for strategy discovery.

    Features:
    - NSGA-II multi-objective optimization
    - Novelty search for behavioral diversity
    - Continuous checkpointing for overnight runs
    - Regime change detection and adaptation
    - Parallel fitness evaluation
    """

    def __init__(
        self,
        config: EvolutionConfig = None,
        data_loader=None,
        backtester: Backtester = None,
        portfolio_evaluator=None,
        use_portfolio_fitness: bool = True,
        use_fast_backtester: bool = False,
        enable_diversity_monitor: bool = True,
        enable_map_elites: bool = True
    ):
        """
        Initialize evolution engine.

        Args:
            config: Evolution configuration
            data_loader: Data loader for market data (optional)
            backtester: Backtester instance (optional, created if not provided)
            portfolio_evaluator: PortfolioFitnessEvaluator for portfolio-level fitness
            use_portfolio_fitness: Whether to use portfolio-level fitness (default True)
            use_fast_backtester: Use FastBacktester for faster GA iterations (default False)
            enable_diversity_monitor: Enable auto-intervention for diversity (default True)
            enable_map_elites: Enable MAP-Elites grid for quality-diversity (default True)
        """
        self.config = config or EvolutionConfig()
        self.data_loader = data_loader

        # Choose backtester - FastBacktester is ~2-5x faster but slightly less accurate
        if backtester:
            self.backtester = backtester
        elif use_fast_backtester:
            self.backtester = FastBacktester(
                initial_capital=100000,
                sample_rate=0.7,  # Use 70% of days for speed
                slippage_bps=15   # Conservative slippage
            )
            logger.info("Using FastBacktester for GA optimization")
        else:
            self.backtester = Backtester(
                initial_capital=100000,
                cost_model='conservative'
            )

        # Core components
        self.factory = GenomeFactory(self.config)
        self.compiler = StrategyCompiler(self.config)
        self.novelty_archive = NoveltyArchive(
            k_neighbors=self.config.novelty_k_neighbors,
            archive_size=self.config.novelty_archive_size
        )
        self.db = get_db()

        # Portfolio-level fitness evaluation
        self.use_portfolio_fitness = use_portfolio_fitness and HAS_PORTFOLIO_FITNESS
        if self.use_portfolio_fitness:
            self.portfolio_evaluator = portfolio_evaluator or PortfolioFitnessEvaluator()
            logger.info("Portfolio fitness evaluation enabled")
        else:
            self.portfolio_evaluator = None

        # Population
        self.population: List[StrategyGenome] = []
        self.fitness_cache: Dict[str, FitnessVector] = {}
        self.behavior_cache: Dict[str, BehaviorVector] = {}

        # Pareto front tracking
        self.pareto_front: List[StrategyGenome] = []

        # State
        self.current_generation = 0
        self.total_strategies_evaluated = 0
        self.strategies_promoted = 0
        self.shutdown_requested = False

        # Data (loaded on demand)
        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._vix_data: Optional[pd.DataFrame] = None

        # Shared memory and persistent worker pool for efficient parallelism
        self._shared_data_manager: Optional[SharedDataManager] = None
        self._worker_pool: Optional[PersistentWorkerPool] = None
        self._pool_initialized = False

        # Diversity monitoring and auto-intervention
        self.enable_diversity_monitor = enable_diversity_monitor
        if enable_diversity_monitor:
            self.diversity_monitor = DiversityMonitor(DiversityThresholds())
            logger.info("Diversity monitoring enabled with auto-intervention")
        else:
            self.diversity_monitor = None

        # MAP-Elites grid for quality-diversity optimization
        self.enable_map_elites = enable_map_elites
        if enable_map_elites:
            self.map_elites = create_default_grid()
            logger.info(f"MAP-Elites grid enabled: {self.map_elites}")
        else:
            self.map_elites = None

        # Setup signal handling for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def handle_shutdown(signum, frame):
            logger.warning(f"Received signal {signum}, requesting shutdown...")
            self.shutdown_requested = True

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

    def _check_memory(self) -> tuple:
        """Check available memory. Returns (is_ok, available_mb)."""
        if not HAS_PSUTIL:
            return True, 0

        try:
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)

            if available_mb < MEMORY_CRITICAL_MB:
                logger.error(f"CRITICAL: Memory very low ({available_mb:.0f}MB available)")
                return False, available_mb
            elif available_mb < MEMORY_WARNING_MB:
                logger.warning(f"Memory getting low ({available_mb:.0f}MB available)")

            return True, available_mb
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")
            return True, 0

    def _prune_caches(self):
        """Prune caches to keep only entries for current population."""
        current_ids = {g.genome_id for g in self.population}

        if len(self.fitness_cache) > MAX_CACHE_SIZE:
            self.fitness_cache = {
                gid: f for gid, f in self.fitness_cache.items()
                if gid in current_ids
            }
            self.behavior_cache = {
                gid: b for gid, b in self.behavior_cache.items()
                if gid in current_ids
            }

    def _cleanup_generation(self):
        """Perform memory cleanup after a generation."""
        self._prune_caches()

        # Limit Pareto front size
        if len(self.pareto_front) > MAX_PARETO_FRONT_SIZE:
            self.pareto_front = sorted(
                self.pareto_front,
                key=lambda g: self.fitness_cache.get(g.genome_id, FitnessVector(0, 0, 0, 0, 0)).sortino,
                reverse=True
            )[:MAX_PARETO_FRONT_SIZE]

        gc.collect()

    def initialize_population(self, size: int = None):
        """
        Initialize random population.

        Args:
            size: Population size (default from config)
        """
        size = size or self.config.population_size
        logger.info(f"Initializing population with {size} individuals")

        self.population = [
            self.factory.create_random_genome(generation=0)
            for _ in range(size)
        ]

    def load_data(self, data: Dict[str, pd.DataFrame] = None,
                  vix_data: pd.DataFrame = None):
        """
        Load market data for backtesting.

        Args:
            data: Dict mapping symbol to DataFrame
            vix_data: VIX data for regime detection
        """
        if data is not None:
            self._data = data
        elif self.data_loader is not None:
            # Load from data loader
            # Limit symbols for Pi memory
            all_syms = self.data_loader.get_available_symbols() if hasattr(self.data_loader, "get_available_symbols") else []
            limited_syms = sorted(all_syms)[:50] if len(all_syms) > 50 else all_syms
            self._data = {s: self.data_loader.load_symbol(s) for s in limited_syms} if limited_syms else self.data_loader.load_all_symbols()

        if vix_data is not None:
            self._vix_data = vix_data
        elif self.data_loader is not None:
            self._vix_data = self.data_loader.load_vix()

        if self._data:
            logger.info(f"Loaded data for {len(self._data)} symbols")

            # Initialize shared memory for parallel workers
            if self.config.parallel_enabled and not self._pool_initialized:
                self._init_parallel_pool()

    def _init_parallel_pool(self):
        """Initialize shared memory and persistent worker pool."""
        if self._pool_initialized:
            return

        logger.info("Initializing shared memory and persistent worker pool...")

        # Create shared memory manager and load data
        self._shared_data_manager = SharedDataManager()
        self._shared_data_manager.load_data(self._data, self._vix_data)

        # Create persistent worker pool
        self._worker_pool = PersistentWorkerPool(
            n_workers=self.config.n_workers,
            shared_metadata=self._shared_data_manager.get_metadata()
        )
        self._worker_pool.start()

        self._pool_initialized = True
        logger.info(f"Parallel pool ready: {self.config.n_workers} workers with shared memory")

    def evaluate_genome(self, genome: StrategyGenome) -> Tuple[FitnessVector, BehaviorVector]:
        """
        Evaluate a single genome.

        Args:
            genome: Genome to evaluate

        Returns:
            Tuple of (FitnessVector, BehaviorVector)
        """
        # Check cache
        if genome.genome_id in self.fitness_cache:
            return self.fitness_cache[genome.genome_id], self.behavior_cache[genome.genome_id]

        # Compile genome to strategy
        strategy = self.compiler.compile(genome)

        # Run backtest
        try:
            result = self.backtester.run(
                strategy=strategy,
                data=self._data,
                vix_data=self._vix_data
            )
        except Exception as e:
            logger.debug(f"Backtest failed for {genome.genome_id}: {e}")
            # Return poor fitness for failed strategies
            result = BacktestResult(
                run_id=genome.genome_id,
                strategy=strategy.name,
                start_date="",
                end_date=""
            )

        # Extract behavior
        behavior = extract_behavior_vector(result)

        # Calculate novelty
        novelty = self.novelty_archive.calculate_novelty(behavior)

        # Calculate fitness
        fitness = calculate_fitness_vector(
            result=result,
            novelty_score=novelty,
            total_trials=self.total_strategies_evaluated + 1
        )

        # Portfolio-level fitness adjustment
        portfolio_score = None
        if self.use_portfolio_fitness and self.portfolio_evaluator is not None:
            try:
                portfolio_score, contribution = self.portfolio_evaluator.evaluate_for_evolution(
                    backtest_result=result,
                    novelty_score=novelty,
                    candidate_weight=0.05  # 5% allocation for evaluation
                )

                # Store portfolio contribution on genome
                genome.portfolio_contribution = contribution
                genome.composite_score = portfolio_score.composite_fitness

                # Reject highly correlated strategies by zeroing fitness
                if portfolio_score.rejected:
                    logger.debug(f"Strategy {genome.genome_id} rejected: {portfolio_score.rejection_reason}")
                    # Reduce novelty score for rejected strategies
                    fitness = FitnessVector(
                        sortino=fitness.sortino * 0.1,  # Heavy penalty
                        max_drawdown=fitness.max_drawdown,
                        cvar_95=fitness.cvar_95,
                        novelty=fitness.novelty * 0.5,
                        deflated_sharpe=fitness.deflated_sharpe,
                        trades=fitness.trades,
                        win_rate=fitness.win_rate,
                        sharpe=fitness.sharpe
                    )
                else:
                    # Boost fitness based on portfolio contribution
                    contribution_boost = 1.0 + portfolio_score.composite_fitness * 0.5
                    fitness = FitnessVector(
                        sortino=fitness.sortino * contribution_boost,
                        max_drawdown=fitness.max_drawdown,
                        cvar_95=fitness.cvar_95,
                        novelty=fitness.novelty * (1.0 + contribution.diversification_ratio * 0.2),
                        deflated_sharpe=fitness.deflated_sharpe,
                        trades=fitness.trades,
                        win_rate=fitness.win_rate,
                        sharpe=fitness.sharpe
                    )

            except Exception as e:
                logger.debug(f"Portfolio fitness evaluation failed for {genome.genome_id}: {e}")

        # Cache results
        self.fitness_cache[genome.genome_id] = fitness
        self.behavior_cache[genome.genome_id] = behavior

        # Store only essential metrics on genome (NOT full backtest result - saves memory)
        genome.fitness_values = fitness.to_tuple()
        # Don't store: genome.backtest_result = result (equity curves consume too much memory)

        self.total_strategies_evaluated += 1

        return fitness, behavior

    def evaluate_population(self):
        """Evaluate entire population."""
        logger.info(f"Evaluating {len(self.population)} individuals")

        for genome in self.population:
            if genome.genome_id not in self.fitness_cache:
                self.evaluate_genome(genome)

    def evaluate_population_parallel(self):
        """
        Evaluate population in parallel using persistent worker pool.

        Uses shared memory for data and persistent workers to avoid
        respawning processes and copying data each generation.

        Memory-safe batching: Evaluates in batches of 10 with GC and cooldown
        between batches to prevent memory exhaustion on Pi.
        """
        import time

        if not self.config.parallel_enabled:
            self.evaluate_population()
            return

        # Ensure pool is initialized
        if not self._pool_initialized:
            self._init_parallel_pool()

        # Filter to genomes that need evaluation
        to_evaluate = [g for g in self.population if g.genome_id not in self.fitness_cache]

        if not to_evaluate:
            return

        # Batch configuration for memory safety
        BATCH_SIZE = 10  # Evaluate 10 at a time instead of all 50
        COOLDOWN_SECONDS = 1.5  # Rest between batches

        total_to_eval = len(to_evaluate)
        logger.info(f"Evaluating {total_to_eval} individuals in batches of {BATCH_SIZE} with {self.config.n_workers} workers")

        # Serialize genomes - data is in shared memory, not passed
        config_dict = asdict(self.config)

        # Split into batches
        batches = [to_evaluate[i:i + BATCH_SIZE] for i in range(0, len(to_evaluate), BATCH_SIZE)]

        success_count = 0

        try:
            for batch_idx, batch_genomes in enumerate(batches):
                batch_num = batch_idx + 1

                # Serialize this batch
                genome_data_list = [self.factory.serialize_genome(g) for g in batch_genomes]

                # Evaluate batch
                raw_results = self._worker_pool.evaluate_batch(genome_data_list, config_dict)

                # Process results in main thread (novelty calculation needs shared archive)
                for genome, result in zip(batch_genomes, raw_results):
                    if not result.get("success", False):
                        logger.warning(f"Worker failed for {genome.genome_id}: {result.get('error', 'unknown')}")
                        continue

                    success_count += 1

                    # Reconstruct behavior vector
                    behavior = BehaviorVector(*result["behavior_array"])

                    # Calculate novelty (requires access to shared novelty archive)
                    novelty = self.novelty_archive.calculate_novelty(behavior)

                    # Calculate fitness vector
                    backtest_result = result["result"]
                    fitness = calculate_fitness_vector(
                        result=backtest_result,
                        novelty_score=novelty,
                        total_trials=self.total_strategies_evaluated + 1
                    )

                    # Cache results
                    self.fitness_cache[genome.genome_id] = fitness
                    self.behavior_cache[genome.genome_id] = behavior

                    # Store only fitness values on genome (NOT backtest_result - saves memory)
                    genome.fitness_values = fitness.to_tuple()

                    self.total_strategies_evaluated += 1

                # Memory cleanup between batches
                gc.collect()

                # Log progress
                evaluated_so_far = min((batch_idx + 1) * BATCH_SIZE, total_to_eval)
                logger.info(f"  Batch {batch_num}/{len(batches)}: {evaluated_so_far}/{total_to_eval} evaluated")

                # Cooldown between batches (except after last batch)
                if batch_idx < len(batches) - 1:
                    time.sleep(COOLDOWN_SECONDS)

            logger.info(f"Parallel evaluation complete: {success_count}/{total_to_eval} succeeded")

        except Exception as e:
            logger.warning(f"Parallel execution failed ({e}), falling back to sequential")
            self.evaluate_population()

    def cleanup_parallel(self):
        """Clean up shared memory and worker pool."""
        if self._worker_pool is not None:
            self._worker_pool.shutdown()
            self._worker_pool = None

        if self._shared_data_manager is not None:
            self._shared_data_manager.cleanup()
            self._shared_data_manager = None

        self._pool_initialized = False
        logger.info("Parallel resources cleaned up")

    def select_parents(self, n_parents: int) -> List[StrategyGenome]:
        """
        Select parents using NSGA-II selection.

        Args:
            n_parents: Number of parents to select

        Returns:
            List of selected parent genomes
        """
        # Get fitness vectors for all population
        population_fitness = []
        for genome in self.population:
            if genome.genome_id in self.fitness_cache:
                fitness = self.fitness_cache[genome.genome_id]
            else:
                fitness, _ = self.evaluate_genome(genome)
            population_fitness.append((genome, fitness))

        # Non-dominated sorting
        fitness_vectors = [f for _, f in population_fitness]
        fronts = non_dominated_sort(fitness_vectors)

        # Select based on rank and crowding distance
        selected = []
        remaining = n_parents

        for front_indices in fronts:
            if remaining <= 0:
                break

            if len(front_indices) <= remaining:
                # Add entire front
                selected.extend([population_fitness[i][0] for i in front_indices])
                remaining -= len(front_indices)
            else:
                # Need to use crowding distance
                distances = crowding_distance(front_indices, fitness_vectors)
                sorted_front = sorted(front_indices, key=lambda i: distances[i], reverse=True)
                selected.extend([population_fitness[i][0] for i in sorted_front[:remaining]])
                remaining = 0

        return selected

    def evolve_generation(self) -> bool:
        """
        Evolve one generation.

        Returns:
            True if evolution should continue, False if memory critical
        """
        self.current_generation += 1
        logger.info(f"Generation {self.current_generation}")

        # Check memory before evolution
        mem_ok, available_mb = self._check_memory()
        if not mem_ok:
            logger.error(f"Aborting evolution due to critical memory ({available_mb:.0f}MB)")
            return False

        # Evaluate current population (parallel if enabled)
        self.evaluate_population_parallel()

        # Log statistics (before creating offspring, while population is still evaluated)
        self._log_generation_stats()

        # Update Pareto front
        self._update_pareto_front()

        # Update novelty archive and MAP-Elites grid
        for genome in self.population:
            if genome.genome_id in self.behavior_cache:
                behavior = self.behavior_cache[genome.genome_id]
                fitness = self.fitness_cache[genome.genome_id]
                novelty = self.novelty_archive.calculate_novelty(behavior)
                self.novelty_archive.maybe_add(behavior, novelty, fitness.sortino)

                # Update MAP-Elites grid
                if self.map_elites is not None:
                    self.map_elites.maybe_add(
                        genome_id=genome.genome_id,
                        fitness=fitness.sortino,
                        fitness_vector=fitness.to_tuple(),
                        behavior=behavior.to_array(),
                        generation=self.current_generation
                    )

        # Check diversity and auto-intervene if needed
        should_intervene = False
        if self.diversity_monitor is not None:
            # Calculate current diversity metrics
            behaviors = [
                self.behavior_cache.get(g.genome_id)
                for g in self.population
                if g.genome_id in self.behavior_cache
            ]
            phenotype_diversity = calculate_population_diversity([b for b in behaviors if b])
            archive_diversity = self.novelty_archive.get_archive_diversity()
            best_fitness = max(
                (self.fitness_cache.get(g.genome_id, FitnessVector(0, 0, 0, 0, 0)).sortino
                 for g in self.population),
                default=0.0
            )

            should_intervene, reason = self.diversity_monitor.update(
                generation=self.current_generation,
                genomes=self.population,
                phenotype_diversity=phenotype_diversity,
                archive_diversity=archive_diversity,
                best_fitness=best_fitness
            )

            if should_intervene:
                logger.warning(f"Auto-intervention triggered: {reason}")
                self.inject_diversity(ratio=self.diversity_monitor.get_injection_ratio())

        # Select parents
        parents = self.select_parents(self.config.population_size)

        # Create offspring through crossover and mutation
        offspring = []

        # Elitism: keep best individuals
        elite = sorted(
            self.population,
            key=lambda g: self.fitness_cache.get(g.genome_id, FitnessVector(0, 0, 0, 0, 0)).sortino,
            reverse=True
        )[:self.config.elitism]
        offspring.extend([copy.deepcopy(g) for g in elite])

        # Get mutation rate (potentially boosted if diversity intervention active)
        mutation_rate = self.config.mutation_rate
        if self.diversity_monitor is not None:
            mutation_rate *= self.diversity_monitor.get_current_mutation_rate_multiplier()
            if self.diversity_monitor.mutation_boost_active:
                logger.debug(f"Mutation rate boosted: {self.config.mutation_rate:.2f} -> {mutation_rate:.2f}")

        # Fill rest with offspring
        while len(offspring) < self.config.population_size:
            # Tournament selection for parents
            p1 = self._tournament_select(parents)
            p2 = self._tournament_select(parents)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.factory.crossover(
                    p1, p2, self.current_generation
                )
            else:
                child1 = copy.deepcopy(p1)
                child1.generation = self.current_generation
                child2 = copy.deepcopy(p2)
                child2.generation = self.current_generation

            # Mutation (using potentially boosted rate)
            if random.random() < mutation_rate:
                child1 = self.factory.mutate(child1, self.current_generation)
            if random.random() < mutation_rate:
                child2 = self.factory.mutate(child2, self.current_generation)

            offspring.append(child1)
            if len(offspring) < self.config.population_size:
                offspring.append(child2)

        # Replace population with offspring
        self.population = offspring[:self.config.population_size]

        # Memory cleanup at end of generation
        self._cleanup_generation()

        return True  # Continue evolution

    def _tournament_select(self, candidates: List[StrategyGenome]) -> StrategyGenome:
        """Tournament selection."""
        tournament = random.sample(
            candidates,
            min(self.config.tournament_size, len(candidates))
        )
        return max(
            tournament,
            key=lambda g: self.fitness_cache.get(g.genome_id, FitnessVector(0, 0, 0, 0, 0)).sortino
        )

    def _update_pareto_front(self):
        """Update the Pareto front with current population."""
        # Combine current front with population
        candidates = self.pareto_front + self.population

        # Get fitness for all
        candidate_fitness = []
        for genome in candidates:
            if genome.genome_id in self.fitness_cache:
                fitness = self.fitness_cache[genome.genome_id]
            else:
                fitness, _ = self.evaluate_genome(genome)
            candidate_fitness.append((genome, fitness))

        # Non-dominated sort
        fitness_vectors = [f for _, f in candidate_fitness]
        fronts = non_dominated_sort(fitness_vectors)

        # First front is the new Pareto front
        if fronts:
            self.pareto_front = [candidate_fitness[i][0] for i in fronts[0]]

    def _log_generation_stats(self):
        """Log generation statistics."""
        if not self.population:
            return

        fitnesses = [
            self.fitness_cache.get(g.genome_id, FitnessVector(0, 0, 0, 0, 0))
            for g in self.population
        ]

        sortinos = [f.sortino for f in fitnesses]
        drawdowns = [f.max_drawdown for f in fitnesses]
        novelties = [f.novelty for f in fitnesses]

        # Calculate diversity
        behaviors = [
            self.behavior_cache.get(g.genome_id)
            for g in self.population
            if g.genome_id in self.behavior_cache
        ]
        diversity = calculate_population_diversity([b for b in behaviors if b])

        # Build log message
        log_parts = [
            f"Gen {self.current_generation}:",
            f"Sortino [{min(sortinos):.2f}/{max(sortinos):.2f}/{np.mean(sortinos):.2f}]",
            f"DD [best={max(drawdowns):.1f}%]",
            f"Novelty={np.mean(novelties):.3f}",
            f"Div={diversity:.3f}",
            f"Pareto={len(self.pareto_front)}"
        ]

        # Add MAP-Elites coverage if enabled
        if self.map_elites is not None:
            log_parts.append(f"MAP={self.map_elites.get_coverage():.1%}")

        # Add diversity monitor status if enabled
        if self.diversity_monitor is not None:
            latest = self.diversity_monitor.get_latest_metrics()
            if latest:
                log_parts.append(f"Geno={latest.genotype_metrics.genotype_diversity:.3f}")

        logger.info("  " + " | ".join(log_parts))

    def run(self, generations: int = None, hours: float = None):
        """
        Run evolution for specified generations or time.

        Args:
            generations: Number of generations to run
            hours: Maximum hours to run (overrides generations if set)
        """
        generations = generations or self.config.generations_per_session
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours) if hours else None

        logger.info(f"Starting evolution: {generations} generations, "
                    f"population={self.config.population_size}")

        # Initialize if needed
        if not self.population:
            self.initialize_population()

        # Check data
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            # Main evolution loop
            for gen in range(generations):
                if self.shutdown_requested:
                    logger.warning("Shutdown requested, saving state...")
                    break

                if end_time and datetime.now() >= end_time:
                    logger.info("Time limit reached")
                    break

                # Evolve and check for memory abort
                should_continue = self.evolve_generation()
                if not should_continue:
                    logger.warning("Evolution stopped due to memory constraints")
                    break

                # Checkpoint periodically
                if self.current_generation % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint()

            # Final checkpoint
            self.save_checkpoint()

            # Promote promising strategies
            self._promote_strategies()

            logger.info(f"Evolution complete: {self.current_generation} generations, "
                        f"{self.total_strategies_evaluated} strategies evaluated, "
                        f"{self.strategies_promoted} promoted")
        finally:
            # Always clean up parallel resources (shared memory, worker pool)
            self.cleanup_parallel()

    def save_checkpoint(self):
        """Save current state to database."""
        logger.info(f"Saving checkpoint at generation {self.current_generation}")

        # Prepare state
        state = EvolutionState(
            generation=self.current_generation,
            population_size=len(self.population),
            pareto_front_size=len(self.pareto_front),
            best_sortino=max(
                f.sortino for f in self.fitness_cache.values()
            ) if self.fitness_cache else 0,
            best_drawdown=max(
                f.max_drawdown for f in self.fitness_cache.values()
            ) if self.fitness_cache else 0,
            avg_novelty=np.mean([
                f.novelty for f in self.fitness_cache.values()
            ]) if self.fitness_cache else 0,
            diversity=calculate_population_diversity([
                b for b in self.behavior_cache.values()
            ]) if self.behavior_cache else 0,
            total_strategies_evaluated=self.total_strategies_evaluated,
            strategies_promoted=self.strategies_promoted,
            timestamp=datetime.now().isoformat()
        )

        # Serialize population
        population_data = [
            self.factory.serialize_genome(g)
            for g in self.population
        ]

        # Serialize Pareto front
        pareto_data = [
            self.factory.serialize_genome(g)
            for g in self.pareto_front
        ]

        # Serialize novelty archive
        archive_data = self.novelty_archive.to_dict()

        checkpoint = {
            'state': asdict(state),
            'population': population_data,
            'pareto_front': pareto_data,
            'novelty_archive': archive_data,
            'config': asdict(self.config)
        }

        # Save to database
        try:
            # Use microseconds + generation to ensure unique checkpoint IDs
            checkpoint_id = f"evo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_g{self.current_generation}"
            self.db.execute(
                "research",
                """
                INSERT OR REPLACE INTO evolution_checkpoints
                (checkpoint_id, generation, population_json, pareto_front_json,
                 novelty_archive_json, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    self.current_generation,
                    json.dumps(population_data),
                    json.dumps(pareto_data),
                    json.dumps(archive_data),
                    json.dumps(asdict(self.config))
                )
            )
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to DB: {e}")
            # Fallback: save to file
            checkpoint_path = Path("research/discovery/checkpoints")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path / f"checkpoint_{self.current_generation}.json", "w") as f:
                json.dump(checkpoint, f)

    def load_checkpoint(self) -> bool:
        """
        Load most recent checkpoint.

        Returns:
            True if checkpoint was loaded
        """
        try:
            row = self.db.fetchone(
                "research",
                """
                SELECT * FROM evolution_checkpoints
                ORDER BY generation DESC LIMIT 1
                """
            )

            if not row:
                return False

            # Restore population
            population_data = json.loads(row['population_json'])
            self.population = [
                self.factory.deserialize_genome(g)
                for g in population_data
            ]

            # Restore Pareto front
            pareto_data = json.loads(row['pareto_front_json'])
            self.pareto_front = [
                self.factory.deserialize_genome(g)
                for g in pareto_data
            ]

            # Restore novelty archive
            archive_data = json.loads(row['novelty_archive_json'])
            self.novelty_archive = NoveltyArchive.from_dict(archive_data)

            self.current_generation = row['generation']

            logger.info(f"Loaded checkpoint from generation {self.current_generation}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False

    def _promote_strategies(self):
        """Promote promising strategies from Pareto front."""
        for genome in self.pareto_front:
            if genome.genome_id not in self.fitness_cache:
                continue

            fitness = self.fitness_cache[genome.genome_id]

            # Check promotion thresholds
            if (fitness.sortino >= self.config.promotion_threshold_sortino and
                fitness.deflated_sharpe >= self.config.promotion_threshold_dsr and
                apply_constraints(fitness, self.config)):

                # Save as discovered strategy
                try:
                    from .strategy_genome import generate_strategy_code

                    self.db.execute(
                        "research",
                        """
                        INSERT OR REPLACE INTO discovered_strategies
                        (strategy_id, genome_json, generation_discovered,
                         oos_sharpe, oos_sortino, oos_max_drawdown,
                         oos_total_trades, oos_win_rate,
                         behavior_vector, novelty_score, status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?)
                        """,
                        (
                            genome.genome_id,
                            self.factory.serialize_genome(genome),
                            self.current_generation,
                            fitness.sharpe,
                            fitness.sortino,
                            fitness.max_drawdown,
                            fitness.trades,
                            fitness.win_rate,
                            json.dumps(self.behavior_cache[genome.genome_id].to_array().tolist())
                            if genome.genome_id in self.behavior_cache else None,
                            fitness.novelty,
                            datetime.now().isoformat(),
                        )
                    )
                    self.strategies_promoted += 1
                    logger.info(f"Promoted strategy {genome.genome_id}: "
                               f"Sortino={fitness.sortino:.2f}, DSR={fitness.deflated_sharpe:.2f}")

                    # Add to portfolio evaluator for future marginal calculations
                    if self.portfolio_evaluator is not None and genome.backtest_result:
                        try:
                            equity = pd.Series(genome.backtest_result.equity_curve)
                            returns = equity.pct_change().dropna()
                            if len(returns) > 20:
                                # Add datetime index if missing
                                if not isinstance(returns.index, pd.DatetimeIndex):
                                    returns.index = pd.date_range(
                                        end=pd.Timestamp.now(),
                                        periods=len(returns),
                                        freq='D'
                                    )
                                self.portfolio_evaluator.add_portfolio_strategy(
                                    strategy_name=genome.genome_id,
                                    returns=returns,
                                    weight=0.05,  # Start with 5% allocation
                                    metrics={
                                        'sharpe': fitness.sharpe,
                                        'sortino': fitness.sortino,
                                        'max_drawdown': fitness.max_drawdown
                                    }
                                )
                        except Exception as pe:
                            logger.debug(f"Failed to add to portfolio: {pe}")

                except Exception as e:
                    logger.warning(f"Failed to promote strategy: {e}")

    def inject_diversity(self, ratio: float = 0.3):
        """
        Inject random individuals to maintain diversity.

        Args:
            ratio: Ratio of population to replace (default 30%)
        """
        n_replace = int(len(self.population) * ratio)
        logger.info(f"Injecting {n_replace} random individuals for diversity")

        # Sort by fitness and replace worst
        sorted_pop = sorted(
            self.population,
            key=lambda g: self.fitness_cache.get(g.genome_id, FitnessVector(0, 0, 0, 0, 0)).sortino,
            reverse=True
        )

        # Keep best, replace worst
        self.population = sorted_pop[:-n_replace]
        self.population.extend([
            self.factory.create_random_genome(self.current_generation)
            for _ in range(n_replace)
        ])


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    print("Testing Evolution Engine...")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    test_data = {}

    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
        test_data[symbol] = pd.DataFrame({
            'open': np.random.randn(252).cumsum() + 100,
            'high': np.random.randn(252).cumsum() + 102,
            'low': np.random.randn(252).cumsum() + 98,
            'close': np.random.randn(252).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)

    # Create engine with small config for testing
    config = EvolutionConfig(
        population_size=10,
        generations_per_session=3,
        checkpoint_frequency=1
    )

    engine = EvolutionEngine(config=config)
    engine.load_data(data=test_data)
    engine.initialize_population()

    print(f"\nPopulation size: {len(engine.population)}")

    # Run a few generations
    print("\nRunning evolution...")
    engine.run(generations=3)

    print(f"\nFinal state:")
    print(f"  Generation: {engine.current_generation}")
    print(f"  Pareto front size: {len(engine.pareto_front)}")
    print(f"  Strategies evaluated: {engine.total_strategies_evaluated}")
    print(f"  Novelty archive size: {len(engine.novelty_archive)}")

    print("\nTest complete!")
