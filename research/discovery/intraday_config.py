"""
Intraday Evolution Configuration
================================
Configuration for evolving intraday trading strategies.

Key differences from daily/swing strategies:
- Shorter hold times (minutes to hours, not days)
- Tighter stops (0.5-2% vs 5-15%)
- More trades required for statistical significance
- Different fitness metrics emphasis
- Must close all positions by EOD
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class IntradayEvolutionConfig:
    """
    Configuration for intraday strategy evolution.

    Optimized for minute-bar data and same-day trading.
    """

    # Population settings
    population_size: int = 80
    generations_per_session: int = 40

    # Genetic operators
    crossover_rate: float = 0.75
    mutation_rate: float = 0.25

    # Mutation type probabilities (must sum to 1.0)
    subtree_mutation_prob: float = 0.65   # Less subtree (simpler trees)
    point_mutation_prob: float = 0.20     # More fine-tuning
    hoist_mutation_prob: float = 0.10
    shrink_mutation_prob: float = 0.05

    # Selection
    tournament_size: int = 3
    elitism: int = 4

    # Novelty search
    novelty_k_neighbors: int = 15
    novelty_archive_size: int = 300
    novelty_weight: float = 0.25

    # ==========================================================================
    # INTRADAY-SPECIFIC FITNESS CONSTRAINTS
    # ==========================================================================

    # More trades required for intraday (higher frequency)
    min_trades: int = 100           # Need more trades for significance

    # Tighter drawdown (intraday should be safer)
    max_drawdown: float = -20.0     # Max 20% drawdown (stricter than swing)

    # Statistical significance
    min_deflated_sharpe: float = 0.75

    # Higher win rate expectation for intraday
    min_win_rate: float = 45.0      # Higher than swing (45% vs 30%)

    # Overfitting detection
    max_sharpe: float = 5.0         # Can be higher for intraday

    # ==========================================================================
    # INTRADAY-SPECIFIC TRADE CONSTRAINTS
    # ==========================================================================

    # Hold time limits (minutes)
    max_hold_minutes: int = 240     # 4 hours max
    min_hold_minutes: int = 1       # At least 1 minute

    # Stop loss (tighter for intraday)
    max_stop_loss_pct: float = 0.02   # 2% max stop
    min_stop_loss_pct: float = 0.002  # 0.2% min stop

    # Target (tighter for intraday)
    max_target_pct: float = 0.03      # 3% max target
    min_target_pct: float = 0.003     # 0.3% min target

    # Position size (can be larger for intraday due to tighter stops)
    max_position_pct: float = 0.25    # 25% max per position
    min_position_pct: float = 0.02    # 2% min per position

    # ==========================================================================
    # TIME CONSTRAINTS
    # ==========================================================================

    # Only trade during certain windows
    trade_start_minute: int = 5       # Start 5 min after open
    trade_end_minute: int = 360       # Stop trading 30 min before close (6 hours into day)
    force_exit_minute: int = 385      # Force close 5 min before market close

    # ==========================================================================
    # TREE CONSTRAINTS (simpler for intraday)
    # ==========================================================================

    max_tree_depth: int = 5           # Simpler trees
    min_tree_depth: int = 2
    max_tree_size: int = 35           # Smaller max size

    # ==========================================================================
    # CHECKPOINTING
    # ==========================================================================

    checkpoint_frequency: int = 8
    max_checkpoints: int = 50

    # ==========================================================================
    # RUNTIME
    # ==========================================================================

    max_runtime_hours: float = 4.0    # Shorter sessions
    log_frequency: int = 4
    n_workers: int = 4
    parallel_enabled: bool = False  # Disabled - too memory-hungry for Pi

    # Validation
    train_ratio: float = 0.7
    validation_generations: int = 5
    promotion_threshold_sortino: float = 0.9   # Higher bar for intraday
    promotion_threshold_dsr: float = 0.85

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []

        mut_sum = (self.subtree_mutation_prob + self.point_mutation_prob +
                   self.hoist_mutation_prob + self.shrink_mutation_prob)
        if abs(mut_sum - 1.0) > 0.01:
            errors.append(f"Mutation probabilities must sum to 1.0, got {mut_sum}")

        if self.min_trades < 50:
            errors.append("min_trades should be at least 50 for statistical significance")

        if self.max_tree_depth < self.min_tree_depth:
            errors.append("max_tree_depth must be >= min_tree_depth")

        if self.max_stop_loss_pct < self.min_stop_loss_pct:
            errors.append("max_stop_loss_pct must be >= min_stop_loss_pct")

        if self.trade_end_minute <= self.trade_start_minute:
            errors.append("trade_end_minute must be > trade_start_minute")

        return errors


@dataclass
class IntradayPrimitiveConfig:
    """Configuration for intraday GP primitive set."""

    # Moving average periods (minutes)
    sma_periods: List[int] = field(default_factory=lambda: [5, 9, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [9, 20, 50])

    # RSI/ATR periods
    rsi_period: int = 14
    atr_period: int = 14

    # Opening range periods (minutes)
    or_periods: List[int] = field(default_factory=lambda: [5, 15, 30])

    # Momentum lookback periods (minutes)
    momentum_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 15])

    # Constant ranges for intraday (tighter than daily)
    small_pct_range: tuple = (0.0005, 0.005)   # 0.05% - 0.5%
    medium_pct_range: tuple = (0.005, 0.02)    # 0.5% - 2%


# ==========================================================================
# Preset Configurations
# ==========================================================================

# Default intraday config
INTRADAY_DEFAULT = IntradayEvolutionConfig()

# Pi-optimized (smaller population, shorter runtime)
INTRADAY_PI_CONFIG = IntradayEvolutionConfig(
    population_size=40,
    generations_per_session=25,
    n_workers=2,
    max_runtime_hours=3.0,
    checkpoint_frequency=5
)

# Aggressive exploration (find diverse strategies)
INTRADAY_EXPLORE_CONFIG = IntradayEvolutionConfig(
    population_size=100,
    novelty_weight=0.45,
    mutation_rate=0.35,
    subtree_mutation_prob=0.75,
    point_mutation_prob=0.12,
    hoist_mutation_prob=0.08,
    shrink_mutation_prob=0.05,
)

# Exploitation (refine existing)
INTRADAY_EXPLOIT_CONFIG = IntradayEvolutionConfig(
    population_size=60,
    novelty_weight=0.1,
    mutation_rate=0.15,
    point_mutation_prob=0.45,
    subtree_mutation_prob=0.35,
    hoist_mutation_prob=0.12,
    shrink_mutation_prob=0.08,
)

# Scalping focused (very short holds, tight stops)
INTRADAY_SCALP_CONFIG = IntradayEvolutionConfig(
    population_size=60,
    min_trades=200,               # Need more trades for scalping
    min_win_rate=55.0,            # Higher win rate expected
    max_hold_minutes=60,          # 1 hour max
    max_stop_loss_pct=0.01,       # 1% max stop
    max_target_pct=0.015,         # 1.5% max target
    trade_start_minute=5,
    trade_end_minute=330,         # Stop earlier
)

# Swing-ish intraday (longer holds, larger moves)
INTRADAY_SWING_CONFIG = IntradayEvolutionConfig(
    population_size=70,
    min_trades=50,                # Fewer but larger trades
    min_win_rate=40.0,
    max_hold_minutes=360,         # Full day possible
    max_stop_loss_pct=0.025,      # 2.5% stop
    max_target_pct=0.04,          # 4% target
)


if __name__ == "__main__":
    # Test configurations
    print("Intraday Evolution Configurations")
    print("=" * 50)

    configs = [
        ("Default", INTRADAY_DEFAULT),
        ("Pi", INTRADAY_PI_CONFIG),
        ("Explore", INTRADAY_EXPLORE_CONFIG),
        ("Exploit", INTRADAY_EXPLOIT_CONFIG),
        ("Scalp", INTRADAY_SCALP_CONFIG),
        ("Swing", INTRADAY_SWING_CONFIG),
    ]

    for name, cfg in configs:
        errors = cfg.validate()
        status = "VALID" if not errors else f"INVALID: {errors}"
        print(f"\n{name} Config: {status}")
        print(f"  Population: {cfg.population_size}")
        print(f"  Min Trades: {cfg.min_trades}")
        print(f"  Max Hold: {cfg.max_hold_minutes} min")
        print(f"  Stop Range: {cfg.min_stop_loss_pct*100:.2f}% - {cfg.max_stop_loss_pct*100:.2f}%")
