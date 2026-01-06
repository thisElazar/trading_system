"""
Novelty Search
==============
Behavioral diversity maintenance through novelty search.

Key concepts:
- BehaviorVector: Characterizes HOW a strategy trades (not just returns)
- NoveltyArchive: Maintains diverse archive of behaviors
- Novelty score: Distance to k-nearest neighbors in behavior space

Reference: Lehman & Stanley (2011) "Abandoning Objectives"
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.backtester import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class BehaviorVector:
    """
    Characterizes strategy behavior for novelty comparison.

    Captures HOW a strategy trades, not just its returns.
    This enables diversity in trading approaches, not just parameters.

    Dimensions:
    - trade_frequency: Trades per week (activity level)
    - avg_hold_period: Average holding time in days (timeframe)
    - long_short_ratio: Balance between longs and shorts
    - return_autocorr: Correlation of consecutive returns (momentum vs mean-reversion)
    - drawdown_depth: Normalized max drawdown (risk tolerance)
    - benchmark_corr: Correlation to market (systematic vs idiosyncratic)
    - signal_variance: Volatility of position changes (stability)
    """
    trade_frequency: float      # Trades per week
    avg_hold_period: float      # Days (log-normalized)
    long_short_ratio: float     # -1 (all short) to +1 (all long)
    return_autocorr: float      # -1 to +1
    drawdown_depth: float       # 0 to 1 (normalized)
    benchmark_corr: float       # -1 to +1
    signal_variance: float      # Normalized variance

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for distance calculations."""
        # Ensure all values are valid and clamped to expected ranges
        trade_freq = min(self.trade_frequency / 10.0, 1.0)  # Cap at 1.0
        hold_period = np.log1p(max(self.avg_hold_period, 0)) / 5.0  # Safer log
        ls_ratio = np.clip((self.long_short_ratio + 1) / 2.0, 0, 1)
        autocorr = np.clip((self.return_autocorr + 1) / 2.0, 0, 1)
        dd = np.clip(self.drawdown_depth, 0, 1)
        bench_corr = np.clip((self.benchmark_corr + 1) / 2.0, 0, 1)
        sig_var = np.clip(self.signal_variance * 10, 0, 1)

        arr = np.array([trade_freq, hold_period, ls_ratio, autocorr, dd, bench_corr, sig_var])
        # Replace any NaN with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BehaviorVector':
        """Create from numpy array."""
        return cls(
            trade_frequency=arr[0] * 10.0,
            avg_hold_period=np.expm1(arr[1] * 3.0),
            long_short_ratio=arr[2] * 2 - 1,
            return_autocorr=arr[3] * 2 - 1,
            drawdown_depth=arr[4],
            benchmark_corr=arr[5] * 2 - 1,
            signal_variance=arr[6] / 10.0
        )

    def distance(self, other: 'BehaviorVector') -> float:
        """Calculate Euclidean distance to another behavior vector."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __repr__(self) -> str:
        return (f"BehaviorVector(freq={self.trade_frequency:.1f}/wk, "
                f"hold={self.avg_hold_period:.1f}d, "
                f"ls={self.long_short_ratio:.2f}, "
                f"dd={self.drawdown_depth:.2f})")


def extract_behavior_vector(result: BacktestResult,
                             benchmark_returns: pd.Series = None) -> BehaviorVector:
    """
    Extract behavioral characteristics from a backtest result.

    Args:
        result: Backtest result
        benchmark_returns: Optional benchmark returns for correlation

    Returns:
        BehaviorVector characterizing the strategy
    """
    trades = result.trades or []
    equity = pd.Series(result.equity_curve) if result.equity_curve else pd.Series([100000])
    returns = equity.pct_change().dropna()

    # Trade frequency (trades per week over backtest period)
    days = max(len(equity), 1)
    weeks = days / 5.0
    trade_freq = len(trades) / max(weeks, 1)

    # Average holding period
    if trades:
        hold_periods = []
        for t in trades:
            try:
                entry = pd.Timestamp(t.get('entry_date', t.get('entry_timestamp')))
                exit_date = t.get('exit_date', t.get('exit_timestamp'))
                if exit_date:
                    exit_ts = pd.Timestamp(exit_date)
                    hold_days = (exit_ts - entry).days
                    hold_periods.append(max(hold_days, 0))
            except Exception as e:
                logger.debug(f"Hold period extraction failed: {e}")
                continue
        avg_hold = np.mean(hold_periods) if hold_periods else 1.0
    else:
        avg_hold = 1.0

    # Long/short ratio (currently assuming all long for simplicity)
    # In future, could track from trade sides
    long_count = sum(1 for t in trades if t.get('side', 'BUY') == 'BUY')
    short_count = len(trades) - long_count
    total = long_count + short_count
    ls_ratio = (long_count - short_count) / max(total, 1)

    # Return autocorrelation (lag-1)
    if len(returns) > 1:
        autocorr = returns.autocorr(lag=1)
        autocorr = autocorr if not np.isnan(autocorr) else 0.0
    else:
        autocorr = 0.0

    # Drawdown depth (normalized)
    cummax = equity.cummax()
    # Avoid division by zero - replace 0 with small value
    cummax_safe = cummax.replace(0, np.nan)
    drawdown = (equity - cummax) / cummax_safe
    drawdown = drawdown.fillna(0)
    dd_depth = abs(drawdown.min()) if len(drawdown) > 0 and not np.isnan(drawdown.min()) else 0.0
    dd_depth = min(dd_depth, 1.0)  # Cap at 100%

    # Benchmark correlation
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align indices
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 10:
            strat_aligned = returns.loc[common_idx]
            bench_aligned = benchmark_returns.loc[common_idx]
            bench_corr = strat_aligned.corr(bench_aligned)
            bench_corr = bench_corr if not np.isnan(bench_corr) else 0.0
        else:
            bench_corr = 0.0
    else:
        bench_corr = 0.0

    # Signal variance (volatility of position changes)
    signal_var = returns.std() if len(returns) > 1 else 0.0
    signal_var = signal_var if not np.isnan(signal_var) else 0.0

    return BehaviorVector(
        trade_frequency=trade_freq,
        avg_hold_period=avg_hold,
        long_short_ratio=ls_ratio,
        return_autocorr=autocorr,
        drawdown_depth=dd_depth,
        benchmark_corr=bench_corr,
        signal_variance=signal_var
    )


class NoveltyArchive:
    """
    Maintains an archive of novel behaviors for diversity.

    Novelty is calculated as the average distance to k-nearest neighbors
    in behavior space. Strategies with high novelty are behaviorally
    different from existing strategies.
    """

    def __init__(self, k_neighbors: int = 20, archive_size: int = 500,
                 min_fitness: float = 0.0):
        """
        Initialize novelty archive.

        Args:
            k_neighbors: Number of neighbors for novelty calculation
            archive_size: Maximum archive size
            min_fitness: Minimum fitness to consider for archive
        """
        self.k = k_neighbors
        self.max_size = archive_size
        self.min_fitness = min_fitness

        self.archive: List[BehaviorVector] = []
        self.archive_fitness: List[float] = []  # Track fitness for context

        # Precomputed distance matrix (updated on add)
        self._distance_matrix: Optional[np.ndarray] = None
        self._behavior_matrix: Optional[np.ndarray] = None

    def calculate_novelty(self, behavior: BehaviorVector) -> float:
        """
        Calculate novelty as average distance to k-nearest neighbors.

        Args:
            behavior: Behavior vector to evaluate

        Returns:
            Novelty score (higher = more novel)
        """
        if len(self.archive) < self.k:
            # Everything is novel when archive is small
            return 1.0

        # Get behavior array
        behavior_arr = behavior.to_array().reshape(1, -1)

        # Calculate distances to all archived behaviors
        distances = cdist(behavior_arr, self._behavior_matrix, metric='euclidean')[0]

        # Get k-nearest neighbors
        k_nearest = np.partition(distances, self.k - 1)[:self.k]

        # Average distance
        novelty = float(np.mean(k_nearest))

        return novelty

    def calculate_novelty_batch(self, behaviors: List[BehaviorVector]) -> List[float]:
        """
        Calculate novelty for multiple behaviors efficiently.

        Args:
            behaviors: List of behavior vectors

        Returns:
            List of novelty scores
        """
        if len(self.archive) < self.k:
            return [1.0] * len(behaviors)

        # Stack all behaviors
        query_matrix = np.vstack([b.to_array() for b in behaviors])

        # Calculate all pairwise distances
        distances = cdist(query_matrix, self._behavior_matrix, metric='euclidean')

        # Get k-nearest for each
        novelties = []
        for i in range(len(behaviors)):
            k_nearest = np.partition(distances[i], self.k - 1)[:self.k]
            novelties.append(float(np.mean(k_nearest)))

        return novelties

    def maybe_add(self, behavior: BehaviorVector, novelty: float,
                  fitness: float) -> bool:
        """
        Probabilistically add to archive based on novelty and fitness.

        Args:
            behavior: Behavior to potentially add
            novelty: Pre-calculated novelty score
            fitness: Fitness value for context

        Returns:
            True if added to archive
        """
        # Skip if fitness too low
        if fitness < self.min_fitness:
            return False

        # Always add if archive not full
        if len(self.archive) < self.max_size:
            self._add_to_archive(behavior, fitness)
            return True

        # Replace least novel if this is more novel
        if self._behavior_matrix is not None:
            # Find least novel in archive
            internal_novelties = self._calculate_internal_novelties()
            min_idx = np.argmin(internal_novelties)
            min_novelty = internal_novelties[min_idx]

            if novelty > min_novelty:
                # Replace
                self.archive[min_idx] = behavior
                self.archive_fitness[min_idx] = fitness
                self._update_behavior_matrix()
                return True

        return False

    def _add_to_archive(self, behavior: BehaviorVector, fitness: float):
        """Add behavior to archive and update distance matrix."""
        self.archive.append(behavior)
        self.archive_fitness.append(fitness)
        self._update_behavior_matrix()

    def _update_behavior_matrix(self):
        """Update the precomputed behavior matrix."""
        if self.archive:
            self._behavior_matrix = np.vstack([b.to_array() for b in self.archive])
        else:
            self._behavior_matrix = None

    def _calculate_internal_novelties(self) -> np.ndarray:
        """Calculate novelty of each archived behavior relative to others."""
        if len(self.archive) < 2:
            return np.ones(len(self.archive))

        # Pairwise distances within archive
        distances = cdist(self._behavior_matrix, self._behavior_matrix, metric='euclidean')

        # Set diagonal to inf so we don't count self-distance
        np.fill_diagonal(distances, np.inf)

        k = min(self.k, len(self.archive) - 1)
        novelties = np.zeros(len(self.archive))

        for i in range(len(self.archive)):
            k_nearest = np.partition(distances[i], k - 1)[:k]
            novelties[i] = np.mean(k_nearest)

        return novelties

    def get_archive_diversity(self) -> float:
        """Calculate overall diversity of the archive."""
        if len(self.archive) < 2:
            return 0.0

        novelties = self._calculate_internal_novelties()
        return float(np.mean(novelties))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize archive to dictionary."""
        return {
            'k_neighbors': self.k,
            'archive_size': self.max_size,
            'min_fitness': self.min_fitness,
            'behaviors': [
                {
                    'trade_frequency': b.trade_frequency,
                    'avg_hold_period': b.avg_hold_period,
                    'long_short_ratio': b.long_short_ratio,
                    'return_autocorr': b.return_autocorr,
                    'drawdown_depth': b.drawdown_depth,
                    'benchmark_corr': b.benchmark_corr,
                    'signal_variance': b.signal_variance,
                    'fitness': f
                }
                for b, f in zip(self.archive, self.archive_fitness)
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoveltyArchive':
        """Deserialize archive from dictionary."""
        archive = cls(
            k_neighbors=data['k_neighbors'],
            archive_size=data['archive_size'],
            min_fitness=data.get('min_fitness', 0.0)
        )

        for b_data in data.get('behaviors', []):
            behavior = BehaviorVector(
                trade_frequency=b_data['trade_frequency'],
                avg_hold_period=b_data['avg_hold_period'],
                long_short_ratio=b_data['long_short_ratio'],
                return_autocorr=b_data['return_autocorr'],
                drawdown_depth=b_data['drawdown_depth'],
                benchmark_corr=b_data['benchmark_corr'],
                signal_variance=b_data['signal_variance']
            )
            archive.archive.append(behavior)
            archive.archive_fitness.append(b_data.get('fitness', 0.0))

        archive._update_behavior_matrix()
        return archive

    def __len__(self) -> int:
        return len(self.archive)

    def __repr__(self) -> str:
        return f"NoveltyArchive(size={len(self.archive)}, k={self.k}, diversity={self.get_archive_diversity():.3f})"


def calculate_population_diversity(behaviors: List[BehaviorVector]) -> float:
    """
    Calculate diversity of a population of behaviors.

    Args:
        behaviors: List of behavior vectors

    Returns:
        Average pairwise distance (higher = more diverse)
    """
    if len(behaviors) < 2:
        return 0.0

    behavior_matrix = np.vstack([b.to_array() for b in behaviors])
    distances = cdist(behavior_matrix, behavior_matrix, metric='euclidean')

    # Get upper triangle (exclude diagonal and lower triangle)
    upper_tri = distances[np.triu_indices(len(behaviors), k=1)]

    return float(np.mean(upper_tri))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Novelty Search...")

    # Create sample behavior vectors
    behaviors = [
        BehaviorVector(trade_frequency=5, avg_hold_period=3, long_short_ratio=0.8,
                      return_autocorr=0.1, drawdown_depth=0.15, benchmark_corr=0.5,
                      signal_variance=0.02),
        BehaviorVector(trade_frequency=20, avg_hold_period=1, long_short_ratio=0.5,
                      return_autocorr=-0.2, drawdown_depth=0.25, benchmark_corr=0.7,
                      signal_variance=0.04),
        BehaviorVector(trade_frequency=2, avg_hold_period=20, long_short_ratio=1.0,
                      return_autocorr=0.3, drawdown_depth=0.10, benchmark_corr=0.3,
                      signal_variance=0.01),
    ]

    print("\nBehavior vectors:")
    for i, b in enumerate(behaviors):
        print(f"  {i}: {b}")
        print(f"      Array: {b.to_array()}")

    # Test distances
    print("\nPairwise distances:")
    for i, b1 in enumerate(behaviors):
        for j, b2 in enumerate(behaviors):
            if i < j:
                print(f"  {i} <-> {j}: {b1.distance(b2):.3f}")

    # Test novelty archive
    print("\nTesting NoveltyArchive...")
    archive = NoveltyArchive(k_neighbors=2, archive_size=10)

    for i, b in enumerate(behaviors):
        novelty = archive.calculate_novelty(b)
        added = archive.maybe_add(b, novelty, fitness=1.0)
        print(f"  Behavior {i}: novelty={novelty:.3f}, added={added}")

    print(f"\nArchive: {archive}")

    # Test population diversity
    print(f"\nPopulation diversity: {calculate_population_diversity(behaviors):.3f}")

    # Test serialization
    print("\nTesting serialization...")
    archive_dict = archive.to_dict()
    restored = NoveltyArchive.from_dict(archive_dict)
    print(f"  Restored: {restored}")

    print("\nTest complete!")
