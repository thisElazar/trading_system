"""
Genetic Algorithm Parameter Optimizer
=====================================
Evolves strategy parameters using genetic algorithms.

Features:
- Tournament selection
- Crossover and mutation
- Elitism
- Parallel fitness evaluation
- Early stopping on convergence
"""

import random
import logging
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Import performance config
try:
    from config import PERF
except ImportError:
    PERF = {"parallel_enabled": False, "n_workers": 1, "use_persistent_pool": False}

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """A single solution (set of parameters)."""
    genes: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0
    
    def copy(self) -> 'Individual':
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            generation=self.generation
        )


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    min_val: float
    max_val: float
    step: float = None  # If set, parameter is discrete
    dtype: type = float
    
    def random_value(self) -> float:
        if self.step:
            n_steps = int((self.max_val - self.min_val) / self.step) + 1
            value = self.min_val + random.randint(0, n_steps - 1) * self.step
        else:
            value = random.uniform(self.min_val, self.max_val)
        return self.dtype(value)
    
    def mutate(self, value: float, strength: float = 0.2) -> float:
        """Mutate value within bounds."""
        range_size = self.max_val - self.min_val
        delta = random.gauss(0, range_size * strength)
        new_val = value + delta
        new_val = max(self.min_val, min(self.max_val, new_val))
        
        if self.step:
            new_val = round((new_val - self.min_val) / self.step) * self.step + self.min_val
        
        return self.dtype(new_val)


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism: int = 2  # Keep top N individuals
    tournament_size: int = 3
    early_stop_generations: int = 5  # Stop if no improvement
    parallel: bool = False  # Override via PERF in production
    n_workers: int = 1  # Override via PERF in production (os.cpu_count() // 2 is safe)


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for trading strategy parameters.
    
    Usage:
        params = [
            ParameterSpec('stop_loss', 0.01, 0.10, step=0.01),
            ParameterSpec('take_profit', 0.02, 0.20, step=0.01),
            ParameterSpec('lookback', 5, 30, step=1, dtype=int),
        ]
        
        def fitness_fn(genes: dict) -> float:
            # Run backtest with these parameters
            result = backtest(strategy, **genes)
            return result.sharpe_ratio
        
        optimizer = GeneticOptimizer(params, fitness_fn)
        best = optimizer.evolve()
        print(f"Best params: {best.genes}, fitness: {best.fitness}")
    """
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        fitness_function: Callable[[Dict[str, float]], float],
        config: GeneticConfig = None
    ):
        self.specs = {p.name: p for p in parameter_specs}
        self.fitness_fn = fitness_function
        self.config = config or GeneticConfig()
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []
    
    def _create_individual(self, generation: int = 0) -> Individual:
        """Create random individual."""
        genes = {name: spec.random_value() for name, spec in self.specs.items()}
        return Individual(genes=genes, generation=generation)
    
    def _evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual."""
        try:
            return self.fitness_fn(individual.genes)
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return float('-inf')
    
    def _tournament_select(self) -> Individual:
        """Select individual via tournament selection."""
        contestants = random.sample(self.population, self.config.tournament_size)
        return max(contestants, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual, generation: int) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_genes = {}
        child2_genes = {}
        
        param_names = list(self.specs.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        for i, name in enumerate(param_names):
            if i < crossover_point:
                child1_genes[name] = parent1.genes[name]
                child2_genes[name] = parent2.genes[name]
            else:
                child1_genes[name] = parent2.genes[name]
                child2_genes[name] = parent1.genes[name]
        
        return (
            Individual(genes=child1_genes, generation=generation),
            Individual(genes=child2_genes, generation=generation)
        )
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutate individual's genes."""
        mutated = individual.copy()
        
        for name, spec in self.specs.items():
            if random.random() < self.config.mutation_rate:
                mutated.genes[name] = spec.mutate(mutated.genes[name])
        
        return mutated
    
    def _evolve_generation(self, generation: int) -> List[Individual]:
        """Create next generation."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elitism - keep best individuals
        new_population = [ind.copy() for ind in self.population[:self.config.elitism]]
        
        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            child1, child2 = self._crossover(parent1, parent2, generation)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.config.population_size]
    
    def _evaluate_population_parallel(self, individuals: List[Individual]) -> None:
        """Evaluate fitness for multiple individuals in parallel using module-level function."""
        # Filter to only evaluate individuals that need evaluation
        to_evaluate = [(i, ind) for i, ind in enumerate(individuals) if ind.fitness == 0]

        if not to_evaluate:
            return

        logger.info(f"Evaluating {len(to_evaluate)} individuals in parallel with {self.config.n_workers} workers")

        try:
            # Import the module-level parallel evaluation function and worker initializer
            from run_nightly_research import evaluate_genes_parallel, _init_parallel_worker

            # CRITICAL: Close database connections before fork to prevent SQLite deadlock
            # Fork inherits file descriptors; if children exit while parent has active
            # connection, the connection becomes corrupted and hangs indefinitely
            from data.storage.db_manager import get_db
            get_db().close_thread_connections()

            # Use multiprocessing Pool with fork context (Linux) for shared globals
            import multiprocessing as mp
            ctx = mp.get_context('fork')

            # Pass initializer to ensure workers ignore SIGTERM (prevents zombie workers)
            with ctx.Pool(processes=self.config.n_workers, initializer=_init_parallel_worker) as pool:
                # Map genes to fitness values
                genes_list = [ind.genes for _, ind in to_evaluate]
                results = pool.map(evaluate_genes_parallel, genes_list)
                
                # Apply results
                for (idx, ind), fitness in zip(to_evaluate, results):
                    individuals[idx].fitness = fitness if fitness is not None else float('-inf')
                
                logger.info(f"Parallel evaluation completed: {len(results)} individuals")

        except Exception as e:
            # Fall back to sequential if parallel fails
            logger.warning(f"Parallel execution failed ({e}), falling back to sequential")
            self._evaluate_population_sequential(individuals)

    def _evaluate_population_sequential(self, individuals: List[Individual]) -> None:
        """Evaluate fitness for multiple individuals sequentially."""
        for ind in individuals:
            if ind.fitness == 0:
                ind.fitness = self._evaluate_fitness(ind)

    def evolve(self) -> Individual:
        """
        Run genetic algorithm evolution.

        Returns:
            Best individual found
        """
        # Initialize population
        self.population = [
            self._create_individual(0)
            for _ in range(self.config.population_size)
        ]

        # Evaluate initial population (parallel or sequential)
        if self.config.parallel:
            self._evaluate_population_parallel(self.population)
        else:
            self._evaluate_population_sequential(self.population)

        self.best_individual = max(self.population, key=lambda x: x.fitness)
        generations_without_improvement = 0

        logger.info(f"Generation 0: Best fitness = {self.best_individual.fitness:.4f}")

        for gen in range(1, self.config.generations + 1):
            # Evolve
            self.population = self._evolve_generation(gen)

            # Evaluate fitness (parallel or sequential)
            if self.config.parallel:
                self._evaluate_population_parallel(self.population)
            else:
                self._evaluate_population_sequential(self.population)

            # Update best
            gen_best = max(self.population, key=lambda x: x.fitness)

            if gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best.copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Record history
            fitnesses = [ind.fitness for ind in self.population]
            self.history.append({
                'generation': gen,
                'best_fitness': self.best_individual.fitness,
                'gen_best': gen_best.fitness,
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses)
            })

            logger.info(
                f"Generation {gen}: Best = {self.best_individual.fitness:.4f}, "
                f"Gen best = {gen_best.fitness:.4f}, Mean = {np.mean(fitnesses):.4f}"
            )

            # Early stopping
            if generations_without_improvement >= self.config.early_stop_generations:
                logger.info(f"Early stopping: No improvement for {generations_without_improvement} generations")
                break

        return self.best_individual
    
    def get_top_n(self, n: int = 5) -> List[Individual]:
        """Get top N individuals from final population."""
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n]


# Predefined parameter specs for common strategies
# These define the search space for genetic optimization
# Ranges are tightened based on academic research to prevent overfitting
# =============================================================================
# STRATEGY PARAMETERS - WIDENED FOR GA EXPLORATION
# =============================================================================
# These ranges are intentionally wider than "optimal" to allow the GA to explore.
# The GA will find the best values within these ranges through evolution.
# Finer steps = more possible combinations = better exploration.
# =============================================================================

STRATEGY_PARAMS = {
    'gap_fill': [
        # Gap thresholds - widened for exploration
        ParameterSpec('min_gap_pct', 0.003, 0.025, step=0.002),   # 0.3-2.5% (widened, finer step)
        ParameterSpec('max_gap_pct', 0.015, 0.06, step=0.005),    # 1.5-6% (widened)
        # Stop loss - wider range
        ParameterSpec('stop_loss_pct', 0.003, 0.025, step=0.002), # 0.3-2.5% (widened)
    ],

    'pairs_trading': [
        # Entry z-score - widened
        ParameterSpec('entry_z', 1.25, 3.0, step=0.25),           # 1.25-3.0 (widened)
        ParameterSpec('exit_z', 0.0, 1.0, step=0.25),             # 0-1.0 (widened)
        # Stop z-score
        ParameterSpec('stop_z', 3.0, 6.0, step=0.5),              # 3.0-6.0 (widened)
        # Cointegration requirements
        ParameterSpec('min_correlation', 0.5, 0.95, step=0.05),   # 0.5-0.95 (widened, finer)
        ParameterSpec('max_half_life', 10, 60, step=5, dtype=int),# 10-60 days (widened)
        # Position management
        ParameterSpec('max_hold_days', 5, 45, step=5, dtype=int), # 5-45 days (widened)
    ],

    'relative_volume_breakout': [
        # Relative volume - our active strategy, explore more
        ParameterSpec('min_rv', 1.2, 3.5, step=0.1),              # 1.2-3.5x (widened, finer step)
        # Gap filter - widened
        ParameterSpec('min_gap_pct', 0.005, 0.05, step=0.005),    # 0.5-5% (widened)
        # ATR-based stops - wider exploration
        ParameterSpec('atr_stop_mult', 0.75, 2.5, step=0.25),     # 0.75-2.5x ATR (widened)
        # ATR-based targets
        ParameterSpec('atr_target_mult', 1.5, 4.0, step=0.25),    # 1.5-4x ATR (widened)
        # Hold period - explore longer holds
        ParameterSpec('max_hold_days', 1, 5, step=1, dtype=int),  # 1-5 days (widened)
    ],

    'vol_managed_momentum': [
        # Formation period - widened
        ParameterSpec('formation_period', 126, 315, step=21, dtype=int),  # 6-15 months (widened)
        ParameterSpec('skip_period', 7, 42, step=7, dtype=int),           # 1-6 weeks (widened)
        # Volatility management - wider exploration
        ParameterSpec('vol_lookback', 7, 168, step=7, dtype=int),         # 1 week - 8 months (widened)
        ParameterSpec('target_vol', 0.08, 0.35, step=0.02),               # 8-35% target (widened, finer)
        # Portfolio construction
        ParameterSpec('top_pct', 0.05, 0.40, step=0.025),                 # Top 5-40% (widened)
    ],

    'mean_reversion': [
        # Lookback period - widened
        ParameterSpec('lookback_period', 5, 30, step=2, dtype=int),       # 1-6 weeks (widened, finer)
        # Entry/exit thresholds - widened
        ParameterSpec('entry_std', 1.0, 3.0, step=0.2),                   # 1.0-3.0 std (widened, finer)
        ParameterSpec('exit_std', 0.0, 1.0, step=0.2),                    # 0-1.0 std (widened)
        # Stop - wider range
        ParameterSpec('stop_std', 2.5, 6.0, step=0.5),                    # 2.5-6.0 std (widened)
        ParameterSpec('max_hold_days', 3, 25, step=2, dtype=int),         # 3-25 days (widened, finer)
    ],

    'sector_rotation': [
        # Momentum calculation - widened significantly
        ParameterSpec('momentum_period', 21, 189, step=14, dtype=int),    # 1-9 months (widened)
        # Number of sectors - more options
        ParameterSpec('top_n_sectors', 1, 6, step=1, dtype=int),          # 1-6 sectors (widened)
        # Rebalance frequency - wider range
        ParameterSpec('rebalance_days', 7, 63, step=7, dtype=int),        # Weekly to quarterly (widened)
    ],

    'quality_smallcap_value': [
        # Quality thresholds - widened
        ParameterSpec('min_roa', 0.01, 0.10, step=0.01),                  # 1-10% ROA (widened)
        ParameterSpec('min_profit_margin', 0.01, 0.10, step=0.01),        # 1-10% margin (widened)
        ParameterSpec('max_debt_to_equity', 0.25, 1.5, step=0.125),       # 0.25-1.5 D/E (widened, finer)
        # Value factor weight - widened
        ParameterSpec('value_weight', 0.20, 0.65, step=0.05),             # 20-65% (widened)
        # Portfolio construction - widened
        ParameterSpec('max_positions', 10, 50, step=5, dtype=int),        # 10-50 positions (widened)
        ParameterSpec('max_single_position', 0.02, 0.10, step=0.01),      # 2-10% per position (widened)
    ],

    'factor_momentum': [
        # Momentum calculation - widened
        ParameterSpec('formation_period_long', 126, 315, step=21, dtype=int),  # 6-15 months (widened)
        ParameterSpec('formation_period_med', 42, 168, step=14, dtype=int),    # 2-8 months (widened)
        ParameterSpec('skip_period', 7, 42, step=7, dtype=int),                # 1-6 weeks (widened)
        # Volatility - wider range
        ParameterSpec('vol_lookback', 21, 126, step=14, dtype=int),            # 1-6 months (widened)
        # Portfolio construction - widened
        ParameterSpec('max_factor_weight', 0.20, 0.60, step=0.05),             # 20-60% (widened)
        ParameterSpec('min_factor_weight', 0.02, 0.15, step=0.01),             # 2-15% (widened)
    ],

    'vix_regime_rotation': [
        # VIX thresholds - widened for better regime detection
        ParameterSpec('low_vix_threshold', 12, 20, step=1, dtype=int),         # 12-20 (widened)
        ParameterSpec('high_vix_threshold', 20, 35, step=1, dtype=int),        # 20-35 (widened, finer)
        ParameterSpec('extreme_vix_threshold', 30, 55, step=5, dtype=int),     # 30-55 (widened)
        # Exposure adjustments - widened
        ParameterSpec('high_vix_reduction', 0.3, 0.8, step=0.1),               # 30-80% (widened)
        ParameterSpec('extreme_vix_reduction', 0.05, 0.40, step=0.05),         # 5-40% (widened)
    ],
}


def optimize_strategy(
    strategy_name: str,
    backtest_fn: Callable,
    config: GeneticConfig = None
) -> Individual:
    """
    Convenience function to optimize a strategy.
    
    Args:
        strategy_name: Name of strategy (must be in STRATEGY_PARAMS)
        backtest_fn: Function(genes) -> fitness score
        config: Optional genetic algorithm config
        
    Returns:
        Best individual found
    """
    if strategy_name not in STRATEGY_PARAMS:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    optimizer = GeneticOptimizer(
        STRATEGY_PARAMS[strategy_name],
        backtest_fn,
        config
    )
    
    return optimizer.evolve()
