"""
Evolution Configuration
=======================
Configuration for autonomous strategy discovery engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EvolutionConfig:
    """Configuration for autonomous evolution."""

    # Population settings
    # Reduced from 100 to 30 for Pi stability; batched evaluation in evolution_engine.py
    population_size: int = 30
    generations_per_session: int = 20

    # Genetic operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2

    # Mutation type probabilities (must sum to 1.0)
    subtree_mutation_prob: float = 0.70   # Replace entire subtree
    point_mutation_prob: float = 0.15     # Change single node
    hoist_mutation_prob: float = 0.10     # Promote subtree to root
    shrink_mutation_prob: float = 0.05    # Replace function with argument

    # Selection
    tournament_size: int = 3
    elitism: int = 5

    # Novelty search
    novelty_k_neighbors: int = 20
    novelty_archive_size: int = 500
    novelty_weight: float = 0.3  # Weight of novelty vs fitness in selection

    # GP-012: Novelty pulsation (adaptive novelty during plateaus)
    novelty_weight_plateau: float = 0.7   # Elevated weight during fitness plateaus
    plateau_detection_window: int = 10    # Generations to check for plateau
    plateau_improvement_threshold: float = 0.01  # Min improvement to avoid plateau

    # Fitness constraints (hard constraints - strategies failing these are rejected)
    # Relaxed for initial discovery - tighten after strategies are found
    min_trades: int = 30              # Was 50 - allows faster iteration
    max_drawdown: float = -45.0       # Was -40 - slightly more lenient
    min_deflated_sharpe: float = 0.5  # Was 0.7 - lower bar for discovery
    min_win_rate: float = 25.0        # Was 30 - allow more experimentation
    max_sharpe: float = 5.0           # Was 4.0 - slightly higher tolerance

    # Checkpointing
    checkpoint_frequency: int = 10     # Every N generations
    max_checkpoints: int = 100         # Keep last N checkpoints

    # Regime adaptation
    regime_change_injection: float = 0.3  # % of population to replace on regime change
    regime_novelty_reset: float = 0.5     # % of novelty archive to clear

    # Tree constraints
    max_tree_depth: int = 6
    min_tree_depth: int = 2
    max_tree_size: int = 50           # Maximum total nodes per tree

    # Strategy constraints
    max_position_pct: float = 0.20    # Maximum 20% per position
    min_position_pct: float = 0.01    # Minimum 1% per position
    max_stop_loss_pct: float = 0.15   # Maximum 15% stop loss
    min_stop_loss_pct: float = 0.01   # Minimum 1% stop loss
    max_target_pct: float = 0.30      # Maximum 30% target
    min_target_pct: float = 0.02      # Minimum 2% target

    # Overnight running
    max_runtime_hours: float = 8.0
    log_frequency: int = 5            # Log every N generations

    # Parallel evaluation
    # Now safe to enable: batched evaluation with GC between batches (evolution_engine.py)
    n_workers: int = 4                # Use all cores
    parallel_enabled: bool = True     # Safe with batched evaluation

    # Validation
    train_ratio: float = 0.7          # 70% train, 30% test
    validation_generations: int = 5   # Extra generations for validation
    # Relaxed promotion thresholds for initial discovery
    promotion_threshold_sortino: float = 0.6   # Was 0.8 - allow more through initially
    promotion_threshold_dsr: float = 0.75      # Was 0.85 - can tighten after seeing results

    # CPCV validation (GP-008: Combinatorial Purged Cross-Validation)
    cpcv_n_subsets: int = 16          # Number of data subsets (S parameter)
    cpcv_purge_days: int = 5          # Gap between train/test for leakage prevention
    cpcv_embargo_pct: float = 0.01    # Skip 1% of data after train end
    cpcv_max_combinations: int = 1000 # Sample for Pi efficiency (full=12870 for S=16)
    cpcv_pbo_threshold: float = 0.05  # Reject if PBO > 5%
    cpcv_n_workers: int = 2           # Parallel workers for CPCV (memory-safe)

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []

        # Check mutation probabilities sum to 1 (tight tolerance to catch config errors)
        mut_sum = (self.subtree_mutation_prob + self.point_mutation_prob +
                   self.hoist_mutation_prob + self.shrink_mutation_prob)
        if abs(mut_sum - 1.0) > 0.001:
            errors.append(f"Mutation probabilities must sum to 1.0, got {mut_sum}")

        # Check constraints
        if self.min_trades < 10:
            errors.append("min_trades should be at least 10 for statistical significance")

        if self.max_tree_depth < self.min_tree_depth:
            errors.append("max_tree_depth must be >= min_tree_depth")

        if self.novelty_weight < 0 or self.novelty_weight > 1:
            errors.append("novelty_weight must be between 0 and 1")

        return errors


@dataclass
class PrimitiveConfig:
    """Configuration for GP primitive set."""

    # Technical indicator periods (used as constants in trees)
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Lookback periods for historical extremes
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Constant ranges for ephemeral random constants
    small_const_range: tuple = (0.001, 0.05)   # 0.1% - 5%
    medium_const_range: tuple = (0.05, 0.20)   # 5% - 20%
    int_const_range: tuple = (5, 50)           # Integer periods


# Default configurations
DEFAULT_CONFIG = EvolutionConfig()
PI_CONFIG = EvolutionConfig(
    population_size=30,        # Memory-safe for Pi (batched in groups of 10)
    generations_per_session=20,
    n_workers=4,               # Use all cores; batching prevents memory issues
    max_runtime_hours=6.0
)

# Aggressive discovery (more exploration)
EXPLORATION_CONFIG = EvolutionConfig(
    population_size=150,
    novelty_weight=0.5,        # Higher novelty weight
    mutation_rate=0.3,         # More mutation
    subtree_mutation_prob=0.80,
    point_mutation_prob=0.10,
    hoist_mutation_prob=0.07,
    shrink_mutation_prob=0.03,  # Sum = 1.0
)

# Exploitation (refine existing good strategies)
EXPLOITATION_CONFIG = EvolutionConfig(
    population_size=80,
    novelty_weight=0.1,        # Lower novelty weight
    mutation_rate=0.1,
    point_mutation_prob=0.50,  # More fine-grained changes
    subtree_mutation_prob=0.30,
    hoist_mutation_prob=0.12,
    shrink_mutation_prob=0.08,  # Sum = 1.0
)


# =============================================================================
# ISLAND MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class IslandConfig:
    """Configuration for island-based parallel evolution."""

    # Island structure
    num_islands: int = 4                    # Number of independent subpopulations
    population_per_island: int = 20         # Individuals per island

    # Migration settings
    # GP-011: Reduced migration to preserve island specialization
    # Research: 2-5% every 50 gens for regime-specialist islands
    migration_interval: int = 50            # Generations between migrations (was 5)
    migration_rate: float = 0.03            # Fraction of population to migrate (was 0.15)
    topology: str = "ring"                  # ring, random, or full

    # Island diversity settings
    vary_mutation_rate: bool = True         # Each island gets different mutation rate
    vary_tree_depth: bool = True            # Each island explores different depths

    def validate(self) -> list:
        """Validate configuration."""
        errors = []
        if self.num_islands < 2:
            errors.append("num_islands must be at least 2")
        if self.migration_rate <= 0 or self.migration_rate >= 1:
            errors.append("migration_rate must be between 0 and 1")
        if self.topology not in ["ring", "random", "full"]:
            errors.append("topology must be 'ring', 'random', or 'full'")
        return errors


# Default island configuration
DEFAULT_ISLAND_CONFIG = IslandConfig()

# Small/fast island config for testing
TEST_ISLAND_CONFIG = IslandConfig(
    num_islands=3,
    population_per_island=10,
    migration_interval=3,
    migration_rate=0.2
)

# Large island config for overnight runs
OVERNIGHT_ISLAND_CONFIG = IslandConfig(
    num_islands=6,
    population_per_island=30,
    migration_interval=10,
    migration_rate=0.1
)
