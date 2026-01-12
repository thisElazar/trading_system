"""
Persistent Genetic Algorithm Optimizer
======================================
Wraps GeneticOptimizer with database persistence for cross-session evolution.

Key features:
- Load population from database (resume from yesterday)
- Save population after each generation
- Log evolution history for analysis
- Incremental evolution (1-5 generations per session)
- Fitness constraints to reject overfit/unrealistic strategies
- Multi-metric composite fitness to prevent single-metric overfitting:
  * Sharpe Ratio (40%) - General risk-adjusted performance
  * Sortino Ratio (30%) - Downside risk-adjusted returns
  * Calmar Ratio (20%) - Return per unit of max drawdown
  * Win Rate (10%) - Consistency and psychological tradability

Usage:
    from research.genetic.persistent_optimizer import PersistentGAOptimizer

    optimizer = PersistentGAOptimizer('vol_managed_momentum', fitness_fn)
    best = optimizer.evolve_incremental(generations=3)
"""

import logging
import multiprocessing as mp
import time
from multiprocessing import Pool
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.genetic.optimizer import (
    GeneticOptimizer, GeneticConfig, Individual, ParameterSpec, STRATEGY_PARAMS
)
from research.genetic.fitness_utils import calculate_composite_fitness
from data.storage.db_manager import get_db

logger = logging.getLogger(__name__)


# Module-level function for pool warmup (lambdas can't be pickled)
def _warmup_worker(worker_id: int) -> bool:
    """Dummy function to trigger worker initialization."""
    return True


# ============================================================================
# FITNESS CONSTRAINTS
# ============================================================================

# Constraint thresholds (configurable per strategy type)
# These are calibrated for FULL dataset (in-sample) testing
# For OOS testing, thresholds are automatically scaled by test_ratio
CONSTRAINT_THRESHOLDS = {
    'default': {
        # Hard constraints - now return small positive fitness instead of 0
        # This allows poor individuals to still be selected occasionally,
        # maintaining genetic diversity while strongly preferring good ones
        'min_trades': 30,           # Minimum trades for statistical significance
        'max_drawdown': -35,        # Maximum acceptable drawdown (%) - relaxed from -30
        'min_annual_return': -5,    # Allow slightly negative (was 0) - OOS can be rough
        'min_win_rate': 30,         # Minimum win rate (%) - relaxed from 35
        # Soft constraint thresholds
        'low_trades': 50,           # Below this gets penalized
        'moderate_drawdown': -20,   # Below this gets penalized
        'low_win_rate': 45,         # Below this gets penalized
        'suspicious_sharpe': 2.5,   # Above this gets penalized - raised from 2.0
    },
    # Strategy-specific overrides (momentum/trend strategies need fewer trades)
    'momentum': {
        'min_trades': 15,           # Reduced from 20 for OOS
        'low_trades': 30,           # Reduced from 35
    },
    'mean_reversion': {
        'min_trades': 30,           # Reduced from 40 for OOS
        'low_trades': 50,           # Reduced from 60
    },
}

# Minimum fitness for "rejected" individuals - allows some selection pressure
# while maintaining diversity. Set to small positive value instead of 0.
REJECTION_FITNESS = 0.01

# Maximum fitness for infeasible solutions (Deb's feasibility rules)
# Feasible solutions always beat infeasible, so we cap infeasible at this value
MAX_INFEASIBLE_FITNESS = 0.10

# Map strategies to constraint profiles
STRATEGY_CONSTRAINT_PROFILE = {
    'vol_managed_momentum': 'momentum',
    'factor_momentum': 'momentum',
    'sector_rotation': 'momentum',
    'mean_reversion': 'mean_reversion',
    'pairs_trading': 'mean_reversion',
    # Others use 'default'
}


def get_constraint_thresholds(strategy_name: str) -> dict:
    """Get constraint thresholds for a strategy, with profile overrides."""
    # Start with defaults
    thresholds = CONSTRAINT_THRESHOLDS['default'].copy()

    # Apply strategy-specific profile overrides
    profile = STRATEGY_CONSTRAINT_PROFILE.get(strategy_name, 'default')
    if profile in CONSTRAINT_THRESHOLDS and profile != 'default':
        thresholds.update(CONSTRAINT_THRESHOLDS[profile])

    return thresholds


def apply_fitness_constraints(
    result,
    strategy_name: str = 'default',
    is_oos: bool = False,
    test_ratio: float = 0.3
) -> Tuple[float, str]:
    """
    Apply hard and soft constraints to fitness evaluation.

    IMPORTANT: For OOS (out-of-sample) testing, constraints are automatically
    relaxed because:
    - Fewer trades expected (only test_ratio of time period)
    - Higher variance in metrics
    - More extreme drawdowns possible

    Instead of hard rejection (fitness=0), we now return REJECTION_FITNESS
    to maintain genetic diversity while still penalizing poor performers.

    Args:
        result: Backtest result object with metrics
        strategy_name: Name of strategy for profile-specific thresholds
        is_oos: True if evaluating out-of-sample results
        test_ratio: Fraction of data used for OOS test (default 0.3 = 30%)

    Returns:
        (multiplier, reason) where multiplier is applied to base fitness
    """
    thresholds = get_constraint_thresholds(strategy_name)

    # Scale trade-count thresholds for OOS testing
    # If we only have 30% of the data, expect ~30% of the trades
    if is_oos:
        oos_scale = max(0.25, test_ratio)  # Floor at 25% to avoid too-low thresholds
        min_trades_scaled = max(5, int(thresholds['min_trades'] * oos_scale))
        low_trades_scaled = max(10, int(thresholds['low_trades'] * oos_scale))
    else:
        min_trades_scaled = thresholds['min_trades']
        low_trades_scaled = thresholds['low_trades']

    # ========================================================================
    # HARD CONSTRAINTS - Apply Deb's feasibility rules
    # Instead of flat rejection, rank by constraint violation magnitude.
    # This gives the GA gradient information to navigate toward feasibility.
    # Fitness range for infeasible: 0.01 to 0.10
    # ========================================================================

    # Minimum trades for statistical significance
    # Use Deb's feasibility rules: rank infeasible by how close to threshold
    if result.total_trades < min_trades_scaled:
        # Scale from REJECTION_FITNESS (0 trades) to MAX_INFEASIBLE_FITNESS (threshold-1)
        trade_ratio = result.total_trades / min_trades_scaled
        infeasible_fitness = REJECTION_FITNESS + (MAX_INFEASIBLE_FITNESS - REJECTION_FITNESS) * trade_ratio
        return infeasible_fitness, f"Low fitness: Only {result.total_trades} trades (min {min_trades_scaled})"

    # Maximum drawdown (catastrophic loss prevention)
    if result.max_drawdown_pct < thresholds['max_drawdown']:
        return REJECTION_FITNESS, f"Low fitness: {result.max_drawdown_pct:.1f}% drawdown (max {thresholds['max_drawdown']}%)"

    # Annual return threshold (relaxed for OOS)
    min_return = thresholds['min_annual_return']
    if is_oos:
        min_return = -10  # Allow up to -10% for OOS since it's a shorter period
    if hasattr(result, 'annual_return') and result.annual_return < min_return:
        return REJECTION_FITNESS, f"Low fitness: {result.annual_return:.1f}% annual return (min {min_return}%)"

    # Minimum win rate (avoid extreme skew strategies)
    min_wr = thresholds['min_win_rate']
    if is_oos:
        min_wr = max(25, min_wr - 10)  # Relax by 10% for OOS, floor at 25%
    if hasattr(result, 'win_rate') and result.win_rate < min_wr:
        return REJECTION_FITNESS, f"Low fitness: {result.win_rate:.1f}% win rate (min {min_wr}%)"

    # ========================================================================
    # SOFT CONSTRAINTS - Reduce fitness but don't reject
    # ========================================================================
    multiplier = 1.0
    reasons = []

    # Penalize low trade count (but less severely)
    if result.total_trades < low_trades_scaled:
        # Graduated penalty: fewer trades = higher penalty
        trade_ratio = result.total_trades / low_trades_scaled
        penalty = 0.7 + (0.3 * trade_ratio)  # 0.7 to 1.0
        multiplier *= penalty
        reasons.append(f"Low trades ({result.total_trades}/{low_trades_scaled}): {penalty:.2f}x")

    # Penalize moderate drawdown (graduated)
    if result.max_drawdown_pct < thresholds['moderate_drawdown']:
        # Graduated: -20% gets 0.9x, -30% gets 0.7x
        dd_severity = (thresholds['moderate_drawdown'] - result.max_drawdown_pct) / 15
        penalty = max(0.6, 0.9 - (0.2 * dd_severity))
        multiplier *= penalty
        reasons.append(f"DD ({result.max_drawdown_pct:.1f}%): {penalty:.2f}x")

    # Penalize low win rate (graduated)
    if hasattr(result, 'win_rate') and result.win_rate < thresholds['low_win_rate']:
        wr_gap = thresholds['low_win_rate'] - result.win_rate
        penalty = max(0.7, 1.0 - (wr_gap / 30))  # 30% gap = 0.7x penalty
        multiplier *= penalty
        reasons.append(f"WR ({result.win_rate:.1f}%): {penalty:.2f}x")

    # Penalize suspiciously high Sharpe (likely overfit) - less aggressive
    if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio > thresholds['suspicious_sharpe']:
        excess = result.sharpe_ratio - thresholds['suspicious_sharpe']
        penalty = max(0.5, 0.85 - (0.1 * excess))  # Graduated penalty
        multiplier *= penalty
        reasons.append(f"High Sharpe ({result.sharpe_ratio:.2f}): {penalty:.2f}x")

    reason = "; ".join(reasons) if reasons else "Passed all constraints"
    return multiplier, reason


class PersistentGAOptimizer:
    """
    Genetic algorithm optimizer with database persistence.

    Enables continuous evolution across sessions:
    - Run a few generations each night
    - Resume from the last population the next day
    - Track improvement over weeks/months

    Anti-stagnation features:
    - Diversity injection: Replace low-fitness individuals with fresh random ones
    - Adaptive mutation: Increase mutation rate when population stagnates
    - Soft rejection: Poor individuals get low (not zero) fitness
    """

    # Diversity injection settings (fitness-based - for failing populations)
    DIVERSITY_INJECTION_THRESHOLD = 0.5   # Inject if >50% have low fitness
    DIVERSITY_INJECTION_RATIO = 0.2       # Replace 20% of population
    LOW_FITNESS_THRESHOLD = 0.05          # Fitness below this is "low"

    # Similarity-based diversity injection (for homogeneous populations)
    SIMILARITY_THRESHOLD = 0.85           # Inject if >85% of genes are identical across population
    SIMILARITY_INJECTION_RATIO = 0.3      # Replace 30% when too similar
    SIMILARITY_CHECK_INTERVAL = 2         # Check every 2 generations

    # Adaptive mutation settings
    MUTATION_INCREASE_RATE = 0.05         # Increase mutation by 5% per stagnant gen
    MAX_MUTATION_RATE = 0.4               # Cap at 40% mutation rate
    BASE_MUTATION_RATE = 0.15             # Reset to this when improvement found

    # Hard reset settings (last resort for deeply stuck populations)
    # Note: This now uses CROSS-SESSION stagnation tracking
    HARD_RESET_THRESHOLD = 50             # Reset population after 50 gens without improvement
    HARD_RESET_PRESERVE_RATIO = 0.20      # Keep top 20% of individuals during reset

    def __init__(
        self,
        strategy_name: str,
        fitness_function: Callable[[Dict[str, float]], float],
        parameter_specs: List[ParameterSpec] = None,
        config: GeneticConfig = None
    ):
        """
        Initialize persistent GA optimizer.
        
        Args:
            strategy_name: Name of strategy (used as DB key)
            fitness_function: Function(genes) -> fitness score
            parameter_specs: Optional custom parameter specs (defaults to STRATEGY_PARAMS)
            config: Optional GA configuration
        """
        self.strategy_name = strategy_name
        self.fitness_fn = fitness_function
        self.db = get_db()
        
        # Get parameter specs
        if parameter_specs:
            self.specs = parameter_specs
        elif strategy_name in STRATEGY_PARAMS:
            self.specs = STRATEGY_PARAMS[strategy_name]
        else:
            raise ValueError(
                f"No parameter specs for strategy '{strategy_name}'. "
                f"Provide parameter_specs or use known strategy: {list(STRATEGY_PARAMS.keys())}"
            )
        
        # Default config for incremental evolution
        self.config = config or GeneticConfig(
            population_size=30,      # Smaller population for nightly runs
            generations=5,           # Max generations per session
            mutation_rate=0.15,      # Slightly higher mutation for exploration
            crossover_rate=0.7,
            elitism=2,
            tournament_size=3,
            early_stop_generations=3  # Stop early if no improvement
        )
        
        # Create base optimizer
        self.optimizer = GeneticOptimizer(
            self.specs,
            self.fitness_fn,
            self.config
        )
        
        # State tracking
        self.current_generation = 0
        self.generations_without_improvement = 0
        self.best_ever_fitness = float('-inf')
        self.best_ever_genes = None

        # Cross-session stagnation tracking
        self.cross_session_stagnation = 0
        self._hard_reset_done_this_session = False  # Only reset once per session

        # Adaptive mutation tracking
        self._current_mutation_rate = self.BASE_MUTATION_RATE

        # Persistent pool management
        self._pool: Optional[Pool] = None
        self._pool_initialized: bool = False
        self._n_workers: int = self.config.n_workers if self.config.parallel else 1

    # =========================================================================
    # Persistent Pool Management
    # =========================================================================

    def init_pool(self, stagger_delay: float = 3.0) -> bool:
        """
        Initialize persistent worker pool for parallel fitness evaluation.

        Call this once before evolve_incremental() when using parallel evaluation.
        Pool will be reused across all generations in the session.

        Args:
            stagger_delay: Seconds between starting each worker (prevents memory stampede)

        Returns:
            True if pool started successfully, False otherwise
        """
        if self._pool_initialized:
            logger.debug("Pool already initialized")
            return True

        if not self.config.parallel:
            logger.debug("Parallel mode disabled, skipping pool init")
            return True

        try:
            logger.info(f"Starting persistent pool with {self._n_workers} workers...")

            # CRITICAL: Close database connections before fork to prevent SQLite deadlock
            from data.storage.db_manager import get_db
            get_db().close_thread_connections()

            # Import worker initializer from run_nightly_research
            from run_nightly_research import _init_parallel_worker

            # Use fork context for shared globals (Linux)
            ctx = mp.get_context('fork')

            self._pool = ctx.Pool(
                processes=self._n_workers,
                initializer=_init_parallel_worker
            )

            # Stagger worker initialization - each worker imports heavy libs on first task
            logger.info("Warming up workers with staggered initialization...")
            for i in range(self._n_workers):
                # Dummy task to trigger worker initialization (uses module-level function for pickling)
                self._pool.apply(_warmup_worker, args=(i,))
                logger.debug(f"Worker {i+1}/{self._n_workers} initialized")
                if i < self._n_workers - 1:
                    time.sleep(stagger_delay)

            self._pool_initialized = True
            logger.info("Persistent pool started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pool: {e}")
            self._pool = None
            self._pool_initialized = False
            return False

    def shutdown_pool(self, wait: bool = True) -> None:
        """
        Shutdown the persistent worker pool.

        Args:
            wait: If True, wait for workers to finish. If False, terminate immediately.
        """
        if not self._pool_initialized or self._pool is None:
            return

        logger.info("Shutting down persistent pool...")

        try:
            if wait:
                self._pool.close()
                self._pool.join()
            else:
                self._pool.terminate()
        except Exception as e:
            logger.warning(f"Error shutting down pool: {e}")
        finally:
            self._pool = None
            self._pool_initialized = False
            logger.info("Pool shutdown complete")

    def __enter__(self):
        """Context manager entry - initialize pool."""
        self.init_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown pool."""
        self.shutdown_pool()
        return False  # Don't suppress exceptions

    def _evaluate_population_with_pool(self, individuals: List[Individual]) -> None:
        """
        Evaluate fitness for individuals using the persistent pool.

        Uses the internal pool if initialized, otherwise falls back to
        the optimizer's built-in parallel evaluation.

        Args:
            individuals: List of individuals to evaluate
        """
        # Filter to only evaluate individuals that need evaluation
        to_evaluate = [(i, ind) for i, ind in enumerate(individuals) if ind.fitness == 0]

        if not to_evaluate:
            return

        if not self._pool_initialized or self._pool is None:
            # Fallback to optimizer's method (creates temp pool)
            logger.debug("Pool not initialized, using fallback evaluation")
            if self.config.parallel:
                self.optimizer._evaluate_population_parallel(individuals)
            else:
                self.optimizer._evaluate_population_sequential(individuals)
            return

        logger.info(f"Evaluating {len(to_evaluate)} individuals with persistent pool")

        try:
            # Import the module-level parallel evaluation function
            from run_nightly_research import evaluate_genes_parallel

            # Map genes to fitness values using persistent pool
            genes_list = [ind.genes for _, ind in to_evaluate]
            results = self._pool.map(evaluate_genes_parallel, genes_list)

            # Apply results
            for (idx, ind), fitness in zip(to_evaluate, results):
                individuals[idx].fitness = fitness if fitness is not None else float('-inf')

            logger.info(f"Persistent pool evaluation completed: {len(results)} individuals")

        except Exception as e:
            logger.warning(f"Persistent pool evaluation failed ({e}), falling back to sequential")
            self.optimizer._evaluate_population_sequential(individuals)

    # =========================================================================
    # Anti-Stagnation Methods
    # =========================================================================

    def _inject_diversity(self, population: List[Individual], generation: int) -> int:
        """
        Replace low-fitness individuals with fresh random ones.

        This breaks the stagnation cycle by introducing new genetic material
        when the population becomes dominated by clones of elite individuals.

        Args:
            population: Current population list (modified in place)
            generation: Current generation number

        Returns:
            Number of individuals replaced
        """
        # Count low-fitness individuals
        low_fitness_count = sum(
            1 for ind in population
            if ind.fitness <= self.LOW_FITNESS_THRESHOLD
        )
        low_fitness_ratio = low_fitness_count / len(population)

        if low_fitness_ratio < self.DIVERSITY_INJECTION_THRESHOLD:
            return 0  # Population is healthy, no injection needed

        # Calculate how many to replace
        n_replace = int(len(population) * self.DIVERSITY_INJECTION_RATIO)
        n_replace = max(2, min(n_replace, low_fitness_count))  # At least 2, at most all low-fitness

        # Sort by fitness to identify worst individuals
        sorted_indices = sorted(
            range(len(population)),
            key=lambda i: population[i].fitness
        )

        # Replace worst individuals with fresh random ones
        replaced = 0
        for idx in sorted_indices[:n_replace]:
            # Don't replace elites (top 2)
            if population[idx].fitness > self.LOW_FITNESS_THRESHOLD:
                continue

            # Create fresh random individual
            new_ind = self.optimizer._create_individual(generation)
            population[idx] = new_ind
            replaced += 1

        if replaced > 0:
            logger.info(
                f"  Diversity injection: Replaced {replaced} low-fitness individuals "
                f"({low_fitness_ratio:.0%} had low fitness)"
            )

        return replaced

    def _inject_similarity_diversity(self, population: List[Individual], generation: int) -> int:
        """
        Replace individuals when population becomes too homogeneous.

        Unlike fitness-based injection (for failing populations), this detects
        when all individuals have converged to similar genes - a sign of
        premature convergence to a local optimum.

        Args:
            population: Current population list (modified in place)
            generation: Current generation number

        Returns:
            Number of individuals replaced
        """
        # Only check periodically to reduce overhead
        if generation % self.SIMILARITY_CHECK_INTERVAL != 0:
            return 0

        if len(population) < 3:
            return 0

        # Calculate gene similarity across population
        similarity = self._calculate_population_similarity(population)

        if similarity < self.SIMILARITY_THRESHOLD:
            return 0  # Population has healthy diversity

        # Population is too similar - inject fresh random individuals
        n_replace = int(len(population) * self.SIMILARITY_INJECTION_RATIO)
        n_replace = max(2, n_replace)  # At least 2

        # Sort by fitness - replace from the middle (not best or worst)
        # This preserves elite individuals and exploration failures
        sorted_pop = sorted(enumerate(population), key=lambda x: x[1].fitness, reverse=True)

        # Skip top 2 (elites) and bottom 2 (might be useful exploration)
        replaceable_indices = [idx for idx, _ in sorted_pop[2:-2]] if len(sorted_pop) > 4 else []

        if not replaceable_indices:
            replaceable_indices = [idx for idx, _ in sorted_pop[2:]]  # Just skip elites

        replaced = 0
        for idx in replaceable_indices[:n_replace]:
            new_ind = self.optimizer._create_individual(generation)
            population[idx] = new_ind
            replaced += 1

        if replaced > 0:
            logger.info(
                f"  Similarity injection: Population {similarity:.0%} similar, "
                f"replaced {replaced} individuals with fresh random ones"
            )

        return replaced

    def _calculate_population_similarity(self, population: List[Individual]) -> float:
        """
        Calculate how similar the population's genes are.

        Returns a value 0-1 where 1 means all individuals are identical.
        Uses the mode (most common value) for each gene and measures
        what fraction of the population shares that mode.
        """
        if len(population) < 2:
            return 0.0

        gene_names = list(population[0].genes.keys())
        if not gene_names:
            return 0.0

        total_similarity = 0.0

        for gene_name in gene_names:
            # Get all values for this gene
            values = [ind.genes.get(gene_name) for ind in population]

            # Count occurrences of each value
            value_counts = {}
            for v in values:
                # Round floats for comparison
                key = round(v, 4) if isinstance(v, float) else v
                value_counts[key] = value_counts.get(key, 0) + 1

            # Find the mode (most common value)
            max_count = max(value_counts.values())
            mode_frequency = max_count / len(population)

            total_similarity += mode_frequency

        # Average similarity across all genes
        return total_similarity / len(gene_names)

    def _adapt_mutation_rate(self, improved: bool) -> float:
        """
        Adjust mutation rate based on whether we found improvement.

        When stagnating, increase mutation to explore more of the search space.
        When improving, reduce mutation to exploit good solutions.

        Args:
            improved: True if this generation found improvement

        Returns:
            New mutation rate
        """
        if improved:
            # Reset to base rate when we find improvement
            self._current_mutation_rate = self.BASE_MUTATION_RATE
        else:
            # Increase mutation rate when stagnating
            self._current_mutation_rate = min(
                self.MAX_MUTATION_RATE,
                self._current_mutation_rate + self.MUTATION_INCREASE_RATE
            )

        # Apply to the optimizer
        self.config.mutation_rate = self._current_mutation_rate

        return self._current_mutation_rate

    def _check_cross_session_stagnation(self) -> int:
        """
        Check how many generations this strategy has been stuck across all sessions.

        Queries the database to find:
        1. The best fitness ever achieved
        2. When that best fitness was first achieved
        3. How many generations have passed since then

        Returns:
            Number of generations without improvement
        """
        try:
            # Use raw SQL for efficiency
            conn = self.db._get_connection('research')
            cursor = conn.cursor()

            # Get the best fitness ever and max generation
            cursor.execute("""
                SELECT MAX(best_fitness), MAX(generation)
                FROM ga_history
                WHERE strategy = ? AND generation >= 0
            """, (self.strategy_name,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                return 0

            best_ever, max_gen = row

            # Find when we first achieved this best fitness
            cursor.execute("""
                SELECT MIN(generation)
                FROM ga_history
                WHERE strategy = ? AND best_fitness = ? AND generation >= 0
            """, (self.strategy_name, best_ever))
            first_best_gen = cursor.fetchone()[0] or max_gen

            generations_stuck = max_gen - first_best_gen
            self.cross_session_stagnation = generations_stuck

            return generations_stuck

        except Exception as e:
            logger.warning(f"Error checking cross-session stagnation: {e}")
            return 0

    def _hard_reset_population(self, generation: int) -> bool:
        """
        Reset the entire population except elites - last resort for deep stagnation.

        This is triggered after HARD_RESET_THRESHOLD generations without improvement
        ACROSS ALL SESSIONS, when all other anti-stagnation measures have failed.

        Args:
            generation: Current generation number

        Returns:
            True if reset was performed
        """
        # Only reset once per session (DB history doesn't change mid-session)
        if self._hard_reset_done_this_session:
            return False

        # Check cross-session stagnation (not just within-session)
        cross_session_stuck = self._check_cross_session_stagnation()

        if cross_session_stuck < self.HARD_RESET_THRESHOLD:
            return False

        logger.warning(
            f"  HARD RESET: {self.strategy_name} stuck for {cross_session_stuck} generations "
            f"(threshold: {self.HARD_RESET_THRESHOLD}) - resetting population"
        )

        # Preserve elite individuals (percentage-based)
        sorted_pop = sorted(
            self.optimizer.population,
            key=lambda x: x.fitness,
            reverse=True
        )
        n_elites = max(2, int(len(sorted_pop) * self.HARD_RESET_PRESERVE_RATIO))
        elites = sorted_pop[:n_elites]

        # Create fresh random population
        new_population = list(elites)  # Start with elites
        n_new = len(self.optimizer.population) - len(elites)

        for _ in range(n_new):
            new_ind = self.optimizer._create_individual(generation)
            new_population.append(new_ind)

        # Replace population
        self.optimizer.population = new_population

        # Reset stagnation counters (give fresh population a chance)
        self.generations_without_improvement = 0
        self.cross_session_stagnation = 0
        self._hard_reset_done_this_session = True  # Don't reset again this session

        # Reset mutation rate to base
        self._current_mutation_rate = self.BASE_MUTATION_RATE
        self.config.mutation_rate = self._current_mutation_rate

        # Log restart event to database for tracking
        self.db.log_ga_history(
            strategy=self.strategy_name,
            generation=-1,  # Special marker for restart event
            best_fitness=elites[0].fitness if elites else 0,
            mean_fitness=0.0,
            std_fitness=0.0,
            best_genes=elites[0].genes if elites else {},
            generations_without_improvement=cross_session_stuck
        )

        logger.info(
            f"  Hard reset complete: Preserved {len(elites)} elites, "
            f"created {n_new} new individuals"
        )

        return True

    def _individual_to_dict(self, ind: Individual) -> dict:
        """Convert Individual to JSON-serializable dict."""
        return {
            'genes': ind.genes,
            'fitness': ind.fitness,
            'generation': ind.generation
        }
    
    def _dict_to_individual(self, d: dict) -> Individual:
        """Convert dict back to Individual."""
        return Individual(
            genes=d['genes'],
            fitness=d['fitness'],
            generation=d['generation']
        )
    
    def load_population(self) -> bool:
        """
        Load population from database.
        
        Returns:
            True if population was loaded, False if starting fresh
        """
        saved = self.db.load_ga_population(self.strategy_name)
        
        if saved and saved['population']:
            # Restore population
            self.optimizer.population = [
                self._dict_to_individual(d) for d in saved['population']
            ]
            self.current_generation = saved['generation']
            
            # Find best individual
            if self.optimizer.population:
                self.optimizer.best_individual = max(
                    self.optimizer.population, 
                    key=lambda x: x.fitness
                )
                self.best_ever_fitness = saved['best_fitness']
                self.best_ever_genes = saved['best_genes']
            
            logger.info(
                f"Loaded population for {self.strategy_name}: "
                f"Gen {self.current_generation}, "
                f"Best fitness: {self.best_ever_fitness:.4f}, "
                f"Population size: {len(self.optimizer.population)}"
            )
            return True
        
        logger.info(f"No saved population for {self.strategy_name}, starting fresh")
        return False
    
    def save_population(self) -> None:
        """Save current population to database."""
        if not self.optimizer.population:
            return
        
        # Convert population to JSON-serializable format
        population_data = [
            self._individual_to_dict(ind) for ind in self.optimizer.population
        ]
        
        best = self.optimizer.best_individual
        self.db.save_ga_population(
            strategy=self.strategy_name,
            generation=self.current_generation,
            population=population_data,
            best_fitness=best.fitness if best else 0,
            best_genes=best.genes if best else {}
        )
        
        logger.debug(f"Saved population: Gen {self.current_generation}")
    
    def log_generation(self, stats: dict) -> None:
        """Log generation statistics to history."""
        self.db.log_ga_history(
            strategy=self.strategy_name,
            generation=self.current_generation,
            best_fitness=stats['best_fitness'],
            mean_fitness=stats['mean_fitness'],
            std_fitness=stats['std_fitness'],
            best_genes=stats['best_genes'],
            generations_without_improvement=self.generations_without_improvement
        )
    
    def evolve_incremental(self, generations: int = None) -> Individual:
        """
        Run incremental evolution.
        
        Loads existing population (or creates new), evolves for N generations,
        saves state, and returns best individual.
        
        Args:
            generations: Number of generations to run (default: config.generations)
            
        Returns:
            Best individual found
        """
        generations = generations or self.config.generations
        
        # Try to load existing population
        loaded = self.load_population()
        
        if not loaded:
            # Initialize new population
            self.optimizer.population = [
                self.optimizer._create_individual(0)
                for _ in range(self.config.population_size)
            ]
            
            # Evaluate initial population (uses persistent pool if available)
            self._evaluate_population_with_pool(self.optimizer.population)
            
            self.optimizer.best_individual = max(
                self.optimizer.population, 
                key=lambda x: x.fitness
            )
            self.best_ever_fitness = self.optimizer.best_individual.fitness
            self.best_ever_genes = self.optimizer.best_individual.genes.copy()
            
            # Log initial generation
            fitnesses = [ind.fitness for ind in self.optimizer.population]
            self.log_generation({
                'best_fitness': self.best_ever_fitness,
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'best_genes': self.best_ever_genes
            })
            
            self.save_population()
            
            logger.info(
                f"Generation 0 (init): Best = {self.best_ever_fitness:.4f}"
            )
        
        # Track if we found improvement this session
        session_improved = False
        start_fitness = self.best_ever_fitness
        
        # Evolve for specified generations
        for i in range(generations):
            self.current_generation += 1

            # Evolve one generation
            self.optimizer.population = self.optimizer._evolve_generation(
                self.current_generation
            )

            # Evaluate new individuals (uses persistent pool if available)
            self._evaluate_population_with_pool(self.optimizer.population)

            # Log fitness summary for this generation
            fitnesses = [ind.fitness for ind in self.optimizer.population]
            low_fitness_count = sum(1 for f in fitnesses if f <= self.LOW_FITNESS_THRESHOLD)
            if low_fitness_count > 0:
                logger.info(
                    f"  Generation {self.current_generation}: "
                    f"{low_fitness_count}/{len(self.optimizer.population)} have low fitness "
                    f"(mutation rate: {self._current_mutation_rate:.0%})"
                )

            # === DIVERSITY INJECTION ===
            # 1. Fitness-based: If too many individuals have low fitness, inject fresh random ones
            n_injected = self._inject_diversity(
                self.optimizer.population,
                self.current_generation
            )

            # 2. Similarity-based: If population is too homogeneous, inject variety
            n_similarity_injected = self._inject_similarity_diversity(
                self.optimizer.population,
                self.current_generation
            )
            n_injected += n_similarity_injected

            # Re-evaluate injected individuals
            if n_injected > 0:
                self._evaluate_population_with_pool(self.optimizer.population)

            # Find generation best
            gen_best = max(self.optimizer.population, key=lambda x: x.fitness)
            
            # Update global best
            improved_this_gen = False
            if gen_best.fitness > self.best_ever_fitness:
                self.best_ever_fitness = gen_best.fitness
                self.best_ever_genes = gen_best.genes.copy()
                self.optimizer.best_individual = gen_best.copy()
                self.generations_without_improvement = 0
                session_improved = True
                improved_this_gen = True

                logger.info(
                    f"  Generation {self.current_generation}: "
                    f"NEW BEST = {self.best_ever_fitness:.4f} 🎯"
                )
            else:
                self.generations_without_improvement += 1
                logger.info(
                    f"  Generation {self.current_generation}: "
                    f"Gen best = {gen_best.fitness:.4f}, "
                    f"All-time best = {self.best_ever_fitness:.4f}"
                )

            # === HARD RESET CHECK ===
            # If stagnating for too long despite all measures, reset population
            if self._hard_reset_population(self.current_generation):
                # Re-evaluate the new population
                self._evaluate_population_with_pool(self.optimizer.population)

            # === ADAPTIVE MUTATION ===
            # Increase mutation rate when stagnating, reset when improving
            new_mutation_rate = self._adapt_mutation_rate(improved_this_gen)
            if not improved_this_gen and new_mutation_rate > self.BASE_MUTATION_RATE:
                logger.info(
                    f"  Adaptive mutation: rate increased to {new_mutation_rate:.0%}"
                )

            # Log and save progress
            fitnesses = [ind.fitness for ind in self.optimizer.population]
            self.log_generation({
                'best_fitness': self.best_ever_fitness,
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'best_genes': self.best_ever_genes
            })
            self.save_population()

            # Early stopping - but with higher threshold when we have adaptive mutation
            # Give more time for the increased mutation to find improvements
            early_stop_gens = self.config.early_stop_generations
            if self._current_mutation_rate > self.BASE_MUTATION_RATE:
                early_stop_gens += 2  # Allow 2 more generations when mutation is elevated

            if self.generations_without_improvement >= early_stop_gens:
                logger.info(
                    f"  Early stopping: No improvement for "
                    f"{self.generations_without_improvement} generations"
                )
                break
        
        # Summary
        improvement = self.best_ever_fitness - start_fitness
        logger.info(
            f"\n{self.strategy_name} evolution complete:\n"
            f"  Generations run: {i + 1}\n"
            f"  Best fitness: {self.best_ever_fitness:.4f}\n"
            f"  Session improvement: {improvement:+.4f}\n"
            f"  Best genes: {self.best_ever_genes}"
        )
        
        return self.optimizer.best_individual
    
    def get_improvement_summary(self, days: int = 30) -> dict:
        """
        Get summary of improvement over time.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Summary dict with improvement statistics
        """
        history = self.db.get_ga_history(self.strategy_name, days)
        
        if not history:
            return {
                'strategy': self.strategy_name,
                'days_tracked': 0,
                'total_generations': 0,
                'improvement': 0,
                'current_best': None
            }
        
        first_fitness = history[0]['best_fitness']
        last_fitness = history[-1]['best_fitness']
        
        return {
            'strategy': self.strategy_name,
            'days_tracked': days,
            'total_generations': len(history),
            'start_fitness': first_fitness,
            'end_fitness': last_fitness,
            'improvement': last_fitness - first_fitness,
            'improvement_pct': ((last_fitness - first_fitness) / abs(first_fitness) * 100) 
                               if first_fitness != 0 else 0,
            'best_genes': self.best_ever_genes
        }


def create_backtest_fitness_fn(
    strategy_name: str,
    backtester,
    data: Dict,
    vix_data=None,
    use_composite: bool = True,
    walk_forward: bool = True,
    train_ratio: float = 0.7,
    degradation_threshold: float = 0.2
) -> Callable[[Dict], float]:
    """
    Create a fitness function that runs walk-forward backtests.

    Uses walk-forward validation to prevent overfitting:
    - Splits data into 70% train / 30% test (configurable)
    - Runs backtest on both sets
    - Uses OUT-OF-SAMPLE (test) fitness as primary metric
    - Penalizes strategies with high train-to-test degradation

    Uses multi-metric composite fitness to prevent single-metric overfitting:
    - Sharpe Ratio (40%): General risk-adjusted performance
    - Sortino Ratio (30%): Downside risk-adjusted returns
    - Calmar Ratio (20%): Return per unit of max drawdown
    - Win Rate (10%): Consistency and psychological tradability

    Args:
        strategy_name: Name of strategy
        backtester: Backtester instance
        data: Historical data dict
        vix_data: Optional VIX data
        use_composite: If True, use multi-metric fitness (default: True)
        walk_forward: Enable walk-forward validation (default: True)
        train_ratio: Ratio of data for training (default: 0.7 = 70%)
        degradation_threshold: Max allowed degradation before penalty (default: 0.2 = 20%)

    Returns:
        Fitness function: genes -> float
    """
    # Lazy import to avoid cascade failures
    strategy_class = None

    if strategy_name == 'vol_managed_momentum':
        from strategies.vol_managed_momentum import VolManagedMomentumStrategy
        strategy_class = VolManagedMomentumStrategy
    elif strategy_name == 'mean_reversion':
        from strategies.mean_reversion import MeanReversionStrategy
        strategy_class = MeanReversionStrategy
    elif strategy_name == 'pairs_trading':
        from strategies.pairs_trading import PairsTradingStrategy
        strategy_class = PairsTradingStrategy
    elif strategy_name == 'relative_volume_breakout':
        from strategies.relative_volume_breakout import RelativeVolumeBreakout
        strategy_class = RelativeVolumeBreakout
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Pre-compute train/test split if walk-forward is enabled
    train_data = None
    test_data = None
    train_vix = None
    test_vix = None
    wf_enabled = False

    if walk_forward and data:
        # Get all unique dates across all symbols
        all_dates = set()
        for sym, df in data.items():
            if df is not None and len(df) > 0:
                all_dates.update(df.index.tolist())

        if len(all_dates) >= 100:  # Need enough data points for meaningful split
            sorted_dates = sorted(all_dates)
            split_idx = int(len(sorted_dates) * train_ratio)
            split_date = sorted_dates[split_idx]

            # Split data
            train_data = {}
            test_data = {}
            for sym, df in data.items():
                if df is not None and len(df) > 0:
                    train_data[sym] = df[df.index <= split_date].copy()
                    test_data[sym] = df[df.index > split_date].copy()

            # Filter out symbols with insufficient data in either split
            train_data = {sym: df for sym, df in train_data.items() if len(df) >= 20}
            test_data = {sym: df for sym, df in test_data.items() if len(df) >= 20}

            # Split VIX data if provided
            if vix_data is not None and len(vix_data) > 0:
                train_vix = vix_data[vix_data.index <= split_date].copy()
                test_vix = vix_data[vix_data.index > split_date].copy()

            # Verify we have enough data in both splits
            if len(train_data) >= 5 and len(test_data) >= 5:
                wf_enabled = True
                logger.info(f"Walk-forward enabled: {len(sorted_dates)} dates, split at {split_date}")
                logger.info(f"  Train: {len(train_data)} symbols ({int(train_ratio*100)}%), "
                           f"Test: {len(test_data)} symbols ({int((1-train_ratio)*100)}%)")
            else:
                logger.warning(f"Insufficient symbols after split (train={len(train_data)}, "
                              f"test={len(test_data)}), using full dataset")
        else:
            logger.warning(f"Insufficient data for walk-forward ({len(all_dates)} dates), "
                          "using full dataset")

    def fitness_fn(genes: Dict) -> float:
        """Evaluate genes by running backtest with walk-forward validation and composite fitness."""
        try:
            # Create strategy with evolved parameters
            strategy = strategy_class(**genes)

            if wf_enabled:
                # === WALK-FORWARD VALIDATION ===

                # Run backtest on TRAINING data (in-sample)
                train_result = backtester.run(
                    strategy=strategy,
                    data=train_data,
                    vix_data=train_vix
                )

                # Recreate strategy to reset any internal state
                strategy = strategy_class(**genes)

                # Run backtest on TEST data (out-of-sample)
                test_result = backtester.run(
                    strategy=strategy,
                    data=test_data,
                    vix_data=test_vix
                )

                # Apply constraint checks to TEST (out-of-sample) result
                # Pass is_oos=True to use relaxed thresholds appropriate for OOS testing
                constraint_mult, constraint_reason = apply_fitness_constraints(
                    test_result,
                    strategy_name,
                    is_oos=True,
                    test_ratio=(1 - train_ratio)
                )

                # No more hard rejection - constraint_mult is always > 0 now
                if constraint_mult <= REJECTION_FITNESS:
                    logger.debug(f"OOS Low fitness: {constraint_reason}")
                    return constraint_mult  # Return small positive value, not 0

                # Calculate fitness using composite multi-metric approach
                if use_composite:
                    train_fitness = calculate_composite_fitness(train_result, verbose=False)
                    test_fitness = calculate_composite_fitness(test_result, verbose=False)
                else:
                    train_fitness = getattr(train_result, 'sharpe_ratio', 0)
                    test_fitness = getattr(test_result, 'sharpe_ratio', 0)

                # Handle edge cases
                if test_fitness == float('inf') or test_fitness == float('-inf'):
                    test_fitness = 0.0
                if train_fitness == float('inf') or train_fitness == float('-inf'):
                    train_fitness = 0.0

                # Use OUT-OF-SAMPLE as primary fitness
                fitness = test_fitness

                # Calculate degradation and apply penalty
                degradation = 0.0
                if train_fitness > 0 and test_fitness < train_fitness:
                    degradation = (train_fitness - test_fitness) / train_fitness
                    if degradation > degradation_threshold:
                        # Penalize fitness proportionally to degradation
                        penalty = max(0.5, 1.0 - degradation)
                        fitness *= penalty

                # Apply constraint penalties
                fitness *= constraint_mult

                # Log walk-forward metrics for debugging
                logger.debug(
                    f"WF Fitness: Train={train_fitness:.3f}, Test={test_fitness:.3f}, "
                    f"Deg={degradation:.0%}, Final={fitness:.3f}"
                )

                return float(fitness)

            else:
                # === FALLBACK: Full dataset (no walk-forward) ===
                result = backtester.run(
                    strategy=strategy,
                    data=data,
                    vix_data=vix_data
                )

                # Apply constraint checks first
                constraint_mult, constraint_reason = apply_fitness_constraints(
                    result, strategy_name
                )

                if constraint_mult == 0:
                    logger.debug(f"Constraint rejection: {constraint_reason}")
                    return 0.0

                # Calculate fitness using composite multi-metric approach
                if use_composite:
                    base_fitness = calculate_composite_fitness(result, verbose=False)
                else:
                    base_fitness = getattr(result, 'sharpe_ratio', 0)

                # Apply constraint penalties
                fitness = base_fitness * constraint_mult

                # Handle edge cases
                if fitness == float('inf') or fitness == float('-inf'):
                    return 0.0

                # Log component metrics for debugging
                logger.debug(
                    f"Fitness: {fitness:.3f} (Sh={result.sharpe_ratio:.2f}, "
                    f"So={result.sortino_ratio:.2f}, DD={result.max_drawdown_pct:.1f}%, "
                    f"WR={result.win_rate:.1f}%, trades={result.total_trades})"
                )

                return float(fitness)

        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0

    # Expose walk-forward status for debugging
    fitness_fn.walk_forward_enabled = wf_enabled
    return fitness_fn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    print("=" * 60)
    print("Persistent GA Optimizer Test")
    print("=" * 60)
    
    # Simple test fitness function
    def test_fitness(genes: dict) -> float:
        """Simple test function: maximize sum of normalized parameters."""
        return sum(genes.values()) / len(genes)
    
    # Test with vol_managed_momentum params
    optimizer = PersistentGAOptimizer(
        'vol_managed_momentum',
        test_fitness,
        config=GeneticConfig(
            population_size=10,
            generations=3,
            early_stop_generations=2
        )
    )
    
    print("\nRunning incremental evolution...")
    best = optimizer.evolve_incremental(generations=3)
    
    print(f"\nBest individual:")
    print(f"  Genes: {best.genes}")
    print(f"  Fitness: {best.fitness:.4f}")
    
    # Check if we can load and continue
    print("\nSimulating restart...")
    optimizer2 = PersistentGAOptimizer(
        'vol_managed_momentum',
        test_fitness,
        config=GeneticConfig(
            population_size=10,
            generations=2
        )
    )
    
    best2 = optimizer2.evolve_incremental(generations=2)
    
    print(f"\nAfter restart, best:")
    print(f"  Generation: {optimizer2.current_generation}")
    print(f"  Fitness: {best2.fitness:.4f}")
    
    # Show improvement summary
    summary = optimizer2.get_improvement_summary(days=1)
    print(f"\nImprovement summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
