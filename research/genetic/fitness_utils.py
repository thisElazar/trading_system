"""
Multi-Metric Fitness Utilities
==============================
Composite fitness functions for GA optimization to prevent single-metric overfitting.

The composite fitness combines multiple performance metrics with carefully chosen
weights to ensure strategies are evaluated holistically:
- Sharpe Ratio: General risk-adjusted performance
- Sortino Ratio: Downside risk-adjusted (more relevant for trading)
- Calmar Ratio: Return per unit of max drawdown
- Omega Ratio: Full distribution capture (gains/losses ratio)
- Win Rate: Trading consistency and psychological tradability
"""

import logging
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_omega_ratio(
    returns: Union[pd.Series, np.ndarray, list],
    threshold: float = 0.0,
    annualize: bool = False,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Omega Ratio - captures all moments of the return distribution.

    The Omega ratio is superior to Sharpe for non-normal distributions because
    it considers the entire return distribution, not just mean and variance.

    Omega(r) = sum(gains above threshold) / sum(losses below threshold)

    A higher Omega means the strategy has better upside relative to downside.
    - Omega = 1.0: Gains equal losses relative to threshold
    - Omega > 1.0: More gains than losses (desirable)
    - Omega < 1.0: More losses than gains (undesirable)

    Args:
        returns: Series or array of period returns (not cumulative)
        threshold: Return threshold (default 0 = risk-free rate)
        annualize: If True, annualize the threshold
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Omega ratio (1.0 = breakeven, higher = better)

    Reference:
        Keating & Shadwick (2002) "A Universal Performance Measure"
    """
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    elif isinstance(returns, list):
        returns = np.array(returns)

    if len(returns) < 10:
        return 1.0  # Insufficient data

    # Annualize threshold if requested (convert annual rate to period rate)
    if annualize and threshold != 0:
        threshold = (1 + threshold) ** (1 / periods_per_year) - 1

    # Calculate excess returns
    excess = returns - threshold

    # Sum of gains (positive excess returns)
    gains = np.sum(excess[excess > 0])

    # Sum of losses (absolute value of negative excess returns)
    losses = np.abs(np.sum(excess[excess < 0]))

    # Handle edge cases
    if losses == 0:
        if gains > 0:
            return 10.0  # Cap at 10 (extremely good - all gains, no losses)
        else:
            return 1.0  # No gains, no losses

    omega = gains / losses

    # Cap at reasonable range to prevent outliers
    return min(max(omega, 0.1), 10.0)


def calculate_composite_fitness(result: Any, verbose: bool = False) -> float:
    """
    Calculate multi-metric fitness to prevent single-metric overfitting.

    Weights based on:
    - Sharpe (40%): General risk-adjusted performance benchmark
    - Sortino (30%): Penalizes downside volatility more (important for trading)
    - Calmar (20%): Ensures drawdowns are manageable for live trading
    - Win rate (10%): Psychological tradability and consistency

    Args:
        result: BacktestResult object with performance metrics
        verbose: If True, log component breakdown

    Returns:
        Composite fitness score (typically 0-3 range for good strategies)
    """
    # Sharpe Ratio: typically 0-2 for good strategies, cap at 3
    sharpe = max(0, getattr(result, 'sharpe_ratio', 0) or 0)
    sharpe = min(sharpe, 3)  # Cap extreme values

    # Sortino Ratio: if not available, estimate from Sharpe * 1.2 (typical ratio)
    sortino = getattr(result, 'sortino_ratio', None)
    if sortino is None or sortino == 0:
        sortino = sharpe * 1.2  # Estimate if not calculated
    sortino = max(0, sortino)
    sortino = min(sortino, 4)  # Cap at 4 (Sortino is typically higher than Sharpe)

    # Calmar Ratio: annual_return / abs(max_drawdown)
    calmar = getattr(result, 'calmar_ratio', None)
    if calmar is None:
        annual_ret = getattr(result, 'annual_return', 0) or 0
        max_dd_pct = getattr(result, 'max_drawdown_pct', 0) or 0

        # Convert from percentage if needed
        if abs(annual_ret) > 1:  # Likely percentage format
            annual_ret = annual_ret / 100
        if abs(max_dd_pct) > 1:  # Likely percentage format
            max_dd_pct = max_dd_pct / 100

        max_dd = abs(max_dd_pct)

        # Avoid division by zero; require minimum 1% drawdown
        if max_dd > 0.01:
            calmar = annual_ret / max_dd
        else:
            calmar = 0

    calmar = max(0, calmar)
    calmar = min(calmar, 3)  # Cap at 3 to prevent outliers from dominating

    # Win Rate: convert from percentage (0-100) to normalized scale (0-2)
    win_rate_raw = getattr(result, 'win_rate', 0) or 0
    if win_rate_raw > 1:  # Likely percentage format
        win_rate = (win_rate_raw / 100) * 2
    else:
        win_rate = win_rate_raw * 2
    win_rate = max(0, min(win_rate, 2))  # Cap at 2

    # Weighted combination
    fitness = (
        0.40 * sharpe +
        0.30 * sortino +
        0.20 * calmar +
        0.10 * win_rate
    )

    # Log component breakdown
    if verbose or logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Fitness components: Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
            f"Calmar={calmar:.2f}, WinRate={win_rate:.2f} -> Composite={fitness:.2f}"
        )

    return float(fitness)


def calculate_fitness_with_penalties(
    result: Any,
    min_trades: int = 30,  # Raised from 10 for statistical validity
    max_drawdown_threshold: float = -40,
    extreme_drawdown_threshold: float = -60,
    verbose: bool = False
) -> float:
    """
    Calculate composite fitness with additional penalties for edge cases.

    This version adds penalties for:
    - Too few trades (potential overfitting) - uses continuous penalty, not cliff
    - Excessive drawdowns (risk management)

    Uses Deb's feasibility rules: instead of a cliff at min_trades,
    applies a continuous penalty proportional to trade count shortfall.

    Args:
        result: BacktestResult object with performance metrics
        min_trades: Minimum trades for full fitness (below gets proportional penalty)
        max_drawdown_threshold: Drawdown level for 20% penalty
        extreme_drawdown_threshold: Drawdown level for 50% penalty
        verbose: If True, log component breakdown

    Returns:
        Penalized composite fitness score
    """
    import math

    # Calculate base composite fitness
    fitness = calculate_composite_fitness(result, verbose=False)

    original_fitness = fitness
    penalties_applied = []

    # Penalty for insufficient trades using Deb's feasibility rules
    # Instead of cliff (0.5x below threshold), use continuous scaling
    total_trades = getattr(result, 'total_trades', 0) or 0
    if total_trades < min_trades:
        # Linear scale: 0 trades = 0.1x, min_trades-1 = ~1.0x
        trade_factor = 0.1 + 0.9 * (total_trades / min_trades)
        fitness *= trade_factor
        penalties_applied.append(f"low_trades({total_trades}<{min_trades}, factor={trade_factor:.2f})")
    else:
        # Exponential soft penalty for feasible solutions (reward more trades)
        trade_factor = 1 - math.exp(-total_trades / min_trades)
        fitness *= trade_factor
        if trade_factor < 0.95:
            penalties_applied.append(f"trade_bonus(factor={trade_factor:.2f})")

    # Penalty for extreme drawdowns
    max_dd = getattr(result, 'max_drawdown_pct', 0) or 0
    if max_dd < extreme_drawdown_threshold:
        fitness *= 0.5
        penalties_applied.append(f"extreme_dd({max_dd:.1f}%)")
    elif max_dd < max_drawdown_threshold:
        fitness *= 0.8
        penalties_applied.append(f"high_dd({max_dd:.1f}%)")

    # Log if penalties were applied
    if verbose or (penalties_applied and logger.isEnabledFor(logging.DEBUG)):
        # Get component metrics for logging
        sharpe = max(0, min(getattr(result, 'sharpe_ratio', 0) or 0, 3))
        sortino = getattr(result, 'sortino_ratio', None)
        if sortino is None or sortino == 0:
            sortino = sharpe * 1.2
        sortino = max(0, min(sortino, 4))

        annual_ret = getattr(result, 'annual_return', 0) or 0
        max_dd_pct = getattr(result, 'max_drawdown_pct', 0) or 0
        if abs(annual_ret) > 1:
            annual_ret = annual_ret / 100
        if abs(max_dd_pct) > 1:
            max_dd_pct = max_dd_pct / 100
        max_dd_abs = abs(max_dd_pct)
        calmar = (annual_ret / max_dd_abs) if max_dd_abs > 0.01 else 0
        calmar = max(0, min(calmar, 3))

        win_rate_raw = getattr(result, 'win_rate', 0) or 0
        if win_rate_raw > 1:
            win_rate = (win_rate_raw / 100) * 2
        else:
            win_rate = win_rate_raw * 2
        win_rate = max(0, min(win_rate, 2))

        penalty_str = f" Penalties: {', '.join(penalties_applied)}" if penalties_applied else ""
        logger.debug(
            f"Fitness components: Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
            f"Calmar={calmar:.2f}, WinRate={win_rate:.2f} -> "
            f"Base={original_fitness:.2f}, Final={fitness:.2f}{penalty_str}"
        )

    return float(fitness)
