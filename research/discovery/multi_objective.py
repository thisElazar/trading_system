"""
Multi-Objective Fitness
=======================
Multi-objective fitness evaluation for evolved strategies.

Implements:
- Sortino ratio (risk-adjusted returns with downside focus)
- Maximum drawdown (capital preservation)
- CVaR at 95% (tail risk)
- Deflated Sharpe Ratio (multiple testing correction)
- Novelty score (behavioral diversity)
"""

import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.backtester import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class FitnessVector:
    """
    Multi-objective fitness values for a strategy.

    Objectives:
    - sortino: Sortino ratio (maximize)
    - max_drawdown: Maximum drawdown percentage (minimize, stored as negative)
    - cvar_95: Conditional Value at Risk at 95% (minimize)
    - novelty: Behavioral novelty score (maximize)
    - deflated_sharpe: DSR corrected for multiple testing (informational)

    Constraints:
    - trades: Total number of trades (for constraint checking)
    - win_rate: Win rate percentage (for constraint checking)
    """
    sortino: float
    max_drawdown: float      # Negative percentage (e.g., -15.0 for 15% drawdown)
    cvar_95: float           # Negative (e.g., -0.02 for -2% expected loss)
    novelty: float           # 0 to infinity
    deflated_sharpe: float   # 0 to 1 (probability Sharpe is not due to luck)

    # Constraints (not optimized, used for filtering)
    trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """
        Convert to tuple for DEAP fitness.

        Returns (sortino, max_drawdown, cvar_95, novelty).
        Note: max_drawdown is stored as negative, so maximizing it
        minimizes the actual drawdown.
        """
        return (self.sortino, self.max_drawdown, self.cvar_95, self.novelty)

    def dominates(self, other: 'FitnessVector') -> bool:
        """
        Check if this solution Pareto-dominates another.

        A solution dominates if it's at least as good in all objectives
        and strictly better in at least one.

        Objectives:
        - sortino: higher is better
        - max_drawdown: higher (less negative) is better
        - cvar_95: higher (less negative) is better
        - novelty: higher is better
        """
        dominated_in_all = True
        strictly_better_in_one = False

        comparisons = [
            (self.sortino, other.sortino),       # Maximize
            (self.max_drawdown, other.max_drawdown),  # Maximize (less negative)
            (-self.cvar_95, -other.cvar_95),     # Minimize -> negate for comparison
            (self.novelty, other.novelty)        # Maximize
        ]

        for mine, theirs in comparisons:
            if mine < theirs:
                dominated_in_all = False
                break
            if mine > theirs:
                strictly_better_in_one = True

        return dominated_in_all and strictly_better_in_one

    def __repr__(self) -> str:
        return (f"FitnessVector(sortino={self.sortino:.3f}, "
                f"max_dd={self.max_drawdown:.1f}%, "
                f"cvar={self.cvar_95:.3f}, "
                f"novelty={self.novelty:.3f})")


def calculate_sortino_ratio(returns: pd.Series, target_return: float = 0.0,
                            annualize: bool = True) -> float:
    """
    Calculate Sortino ratio (downside deviation risk-adjusted return).

    Args:
        returns: Series of period returns
        target_return: Target/minimum acceptable return (default 0)
        annualize: Whether to annualize the ratio

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - target_return

    # Downside returns only
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        # No downside: return a high but not infinite value
        return 5.0

    # Downside deviation
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))

    if downside_dev < 1e-10:
        return 5.0

    # Sortino ratio
    mean_return = returns.mean()
    sortino = mean_return / downside_dev

    if annualize:
        sortino *= np.sqrt(252)

    return float(sortino) if not np.isnan(sortino) else 0.0


def calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that we're in the worst alpha% of outcomes.

    Args:
        returns: Series of period returns
        alpha: Tail probability (0.05 = worst 5%)

    Returns:
        CVaR (negative number representing expected loss)
    """
    if len(returns) < 10:
        return 0.0

    # VaR at alpha
    var = returns.quantile(alpha)

    # CVaR is expected return given return <= VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    cvar = tail_returns.mean()
    return float(cvar) if not np.isnan(cvar) else 0.0


def calculate_deflated_sharpe(sharpe: float, n_returns: int, n_trials: int,
                               skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Calculate Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects for multiple testing by estimating the probability that
    the observed Sharpe ratio could have arisen by chance given the
    number of strategies tested.

    Args:
        sharpe: Observed Sharpe ratio
        n_returns: Number of return observations
        n_trials: Number of strategy variations tested
        skew: Return skewness (default 0 = symmetric)
        kurtosis: Return kurtosis (default 3 = normal)

    Returns:
        Probability that Sharpe is significant (0 to 1)
    """
    if n_trials < 1:
        n_trials = 1
    if n_returns < 2:
        return 0.0

    # Standard error of Sharpe ratio
    se_sharpe = np.sqrt(
        (1 + 0.5 * sharpe**2 - skew * sharpe + ((kurtosis - 3) / 4) * sharpe**2) / n_returns
    )

    if se_sharpe < 1e-10:
        se_sharpe = 1e-10

    # Expected maximum Sharpe under null hypothesis (multiple testing)
    # Using Euler's constant for the extreme value distribution
    euler_gamma = 0.5772156649

    # Expected max of n_trials independent standard normals
    if n_trials > 1:
        expected_max_z = ((1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) +
                         euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    else:
        expected_max_z = 0

    # Expected maximum Sharpe
    expected_max_sharpe = expected_max_z * se_sharpe

    # Deflated Sharpe: probability that observed Sharpe exceeds expected max
    if se_sharpe > 0:
        z_score = (sharpe - expected_max_sharpe) / se_sharpe
        dsr = stats.norm.cdf(z_score)
    else:
        dsr = 0.5

    return float(dsr) if not np.isnan(dsr) else 0.0


def calculate_fitness_vector(
    result: BacktestResult,
    novelty_score: float = 0.0,
    total_trials: int = 1
) -> FitnessVector:
    """
    Calculate multi-objective fitness vector from backtest result.

    Args:
        result: Backtest result object
        novelty_score: Pre-calculated novelty score
        total_trials: Number of strategies tested (for DSR)

    Returns:
        FitnessVector with all objectives
    """
    # Extract equity curve and calculate returns
    equity = pd.Series(result.equity_curve) if result.equity_curve else pd.Series([100000])
    returns = equity.pct_change().dropna()

    # Sortino ratio
    sortino = calculate_sortino_ratio(returns)

    # CRITICAL: Penalize strategies with no/few trades
    # Without trades, Sortino edge case returns 5.0 (no downside)
    # This causes all non-trading strategies to converge at same fitness
    if result.total_trades == 0:
        sortino = -5.0  # Heavily penalize no-trade strategies
    elif result.total_trades < 10:
        # Proportional penalty for very few trades
        sortino = sortino * (result.total_trades / 10) - 2.0

    # Clamp extreme values
    sortino = max(-5.0, min(10.0, sortino))

    # Maximum drawdown (already calculated, but verify)
    max_dd = result.max_drawdown_pct
    if max_dd is None or np.isnan(max_dd):
        max_dd = 0.0

    # CVaR at 95%
    cvar = calculate_cvar(returns, alpha=0.05)

    # Deflated Sharpe Ratio
    sharpe = result.sharpe_ratio if result.sharpe_ratio else 0.0
    skew = returns.skew() if len(returns) > 2 else 0.0
    kurt = returns.kurtosis() if len(returns) > 2 else 3.0

    # Handle infinite/nan skew and kurtosis
    if np.isnan(skew) or np.isinf(skew):
        skew = 0.0
    if np.isnan(kurt) or np.isinf(kurt):
        kurt = 3.0

    dsr = calculate_deflated_sharpe(
        sharpe=sharpe,
        n_returns=len(returns),
        n_trials=total_trials,
        skew=skew,
        kurtosis=kurt
    )

    return FitnessVector(
        sortino=sortino,
        max_drawdown=max_dd,  # Already negative or zero
        cvar_95=cvar,
        novelty=novelty_score,
        deflated_sharpe=dsr,
        trades=result.total_trades,
        win_rate=result.win_rate,
        sharpe=sharpe
    )


def non_dominated_sort(fitness_vectors: List[FitnessVector]) -> List[List[int]]:
    """
    Sort solutions into non-dominated fronts (Pareto ranking).

    Front 0 = Pareto optimal solutions (no other solution dominates them)
    Front 1 = Solutions dominated only by front 0
    etc.

    Args:
        fitness_vectors: List of fitness vectors

    Returns:
        List of fronts, each front is a list of indices into fitness_vectors
    """
    n = len(fitness_vectors)
    if n == 0:
        return []

    # Count how many solutions dominate each solution
    domination_count = [0] * n

    # List of solutions dominated by each solution
    dominated_solutions = [[] for _ in range(n)]

    # Build domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            if fitness_vectors[i].dominates(fitness_vectors[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif fitness_vectors[j].dominates(fitness_vectors[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    # Build fronts
    fronts = [[]]

    # First front: solutions not dominated by anyone
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Handle edge case: if first front is empty, return empty list
    if not fronts[0]:
        return []

    # Build subsequent fronts
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    # Remove empty last front if present
    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


def crowding_distance(front_indices: List[int],
                      fitness_vectors: List[FitnessVector]) -> Dict[int, float]:
    """
    Calculate crowding distance for diversity preservation.

    Solutions at the boundary of the objective space get infinite distance.
    Solutions in crowded regions get lower distance.

    Args:
        front_indices: Indices of solutions in this front
        fitness_vectors: All fitness vectors

    Returns:
        Dict mapping index to crowding distance
    """
    n = len(front_indices)
    if n == 0:
        return {}
    if n <= 2:
        return {i: float('inf') for i in front_indices}

    distances = {i: 0.0 for i in front_indices}

    # Calculate for each objective
    objectives = ['sortino', 'max_drawdown', 'cvar_95', 'novelty']

    for obj in objectives:
        # Sort by this objective
        sorted_indices = sorted(
            front_indices,
            key=lambda i: getattr(fitness_vectors[i], obj)
        )

        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Calculate range
        f_min = getattr(fitness_vectors[sorted_indices[0]], obj)
        f_max = getattr(fitness_vectors[sorted_indices[-1]], obj)
        f_range = f_max - f_min

        if f_range > 1e-10:
            for k in range(1, n - 1):
                prev_val = getattr(fitness_vectors[sorted_indices[k - 1]], obj)
                next_val = getattr(fitness_vectors[sorted_indices[k + 1]], obj)
                distances[sorted_indices[k]] += (next_val - prev_val) / f_range

    return distances


def apply_constraints(fitness: FitnessVector, config) -> bool:
    """
    Check if a solution meets hard constraints.

    Args:
        fitness: Fitness vector to check
        config: EvolutionConfig with constraint thresholds

    Returns:
        True if constraints are satisfied
    """
    # Minimum trades
    if fitness.trades < config.min_trades:
        return False

    # Maximum drawdown
    if fitness.max_drawdown < config.max_drawdown:  # max_dd is negative
        return False

    # Minimum win rate
    if fitness.win_rate < config.min_win_rate:
        return False

    # Suspicious Sharpe (likely overfitting)
    if fitness.sharpe > config.max_sharpe:
        return False

    # Deflated Sharpe threshold (statistically significant)
    if fitness.deflated_sharpe < config.min_deflated_sharpe:
        return False

    return True


def select_nsga2(population_fitness: List[Tuple[Any, FitnessVector]],
                 n_select: int) -> List[int]:
    """
    NSGA-II selection: select best solutions based on Pareto rank
    and crowding distance.

    Args:
        population_fitness: List of (individual, fitness) tuples
        n_select: Number of solutions to select

    Returns:
        Indices of selected solutions
    """
    n = len(population_fitness)
    if n <= n_select:
        return list(range(n))

    fitness_vectors = [f for _, f in population_fitness]

    # Non-dominated sorting
    fronts = non_dominated_sort(fitness_vectors)

    selected = []
    front_idx = 0

    # Add complete fronts until we exceed n_select
    while front_idx < len(fronts) and len(selected) + len(fronts[front_idx]) <= n_select:
        selected.extend(fronts[front_idx])
        front_idx += 1

    # If we still need more, use crowding distance on the next front
    if len(selected) < n_select and front_idx < len(fronts):
        remaining = n_select - len(selected)
        current_front = fronts[front_idx]

        # Calculate crowding distances
        distances = crowding_distance(current_front, fitness_vectors)

        # Sort by crowding distance (higher is better)
        sorted_front = sorted(current_front, key=lambda i: distances[i], reverse=True)

        # Add the most spread-out solutions
        selected.extend(sorted_front[:remaining])

    return selected


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Multi-Objective Fitness...")

    # Create synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    print("\nCalculating individual metrics...")
    sortino = calculate_sortino_ratio(returns)
    print(f"  Sortino: {sortino:.3f}")

    cvar = calculate_cvar(returns)
    print(f"  CVaR 95%: {cvar:.4f}")

    dsr = calculate_deflated_sharpe(1.5, 252, 100)
    print(f"  DSR (Sharpe=1.5, 100 trials): {dsr:.3f}")

    # Create mock backtest result
    class MockResult:
        equity_curve = (100000 * np.cumprod(1 + returns)).tolist()
        sharpe_ratio = 1.2
        max_drawdown_pct = -15.5
        total_trades = 80
        win_rate = 52.0

    result = MockResult()

    print("\nCalculating fitness vector...")
    fitness = calculate_fitness_vector(result, novelty_score=0.5, total_trials=50)
    print(f"  {fitness}")

    # Test non-dominated sorting
    print("\nTesting non-dominated sorting...")
    fitness_vectors = [
        FitnessVector(sortino=1.5, max_drawdown=-10, cvar_95=-0.02, novelty=0.8, deflated_sharpe=0.9),
        FitnessVector(sortino=1.2, max_drawdown=-8, cvar_95=-0.015, novelty=0.6, deflated_sharpe=0.85),
        FitnessVector(sortino=0.8, max_drawdown=-5, cvar_95=-0.01, novelty=0.9, deflated_sharpe=0.7),
        FitnessVector(sortino=2.0, max_drawdown=-15, cvar_95=-0.025, novelty=0.3, deflated_sharpe=0.95),
        FitnessVector(sortino=0.5, max_drawdown=-20, cvar_95=-0.03, novelty=0.2, deflated_sharpe=0.5),
    ]

    fronts = non_dominated_sort(fitness_vectors)
    print(f"  Fronts: {fronts}")

    # Test crowding distance
    print("\nTesting crowding distance...")
    if fronts[0]:
        distances = crowding_distance(fronts[0], fitness_vectors)
        print(f"  Front 0 distances: {distances}")

    print("\nTest complete!")
