"""
MAP-Elites Grid for Quality-Diversity Optimization
===================================================
Maintains a grid of elite solutions across behavioral feature space.

Each cell in the grid represents a specific behavioral niche.
Only the best-performing solution in each niche is kept.
This ensures diversity across different trading styles while
maintaining quality within each style.

Reference: Mouret & Clune (2015) "Illuminating search spaces by
mapping elites"
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import json

import numpy as np

if TYPE_CHECKING:
    from .strategy_genome import StrategyGenome
    from .novelty_search import BehaviorVector
    from .multi_objective import FitnessVector

logger = logging.getLogger(__name__)


@dataclass
class MapElitesConfig:
    """Configuration for MAP-Elites grid."""

    # Grid dimensions (behavioral features to track)
    # Each dimension will be discretized into `resolution` bins
    dimensions: List[str] = field(default_factory=lambda: [
        'trade_frequency',    # Activity level
        'avg_hold_period',    # Timeframe
        'drawdown_depth'      # Risk tolerance
    ])

    # Number of bins per dimension
    resolution: int = 10

    # Feature ranges for normalization
    # If not specified, will use adaptive ranges
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'trade_frequency': (0.0, 10.0),     # 0-10 trades per week
        'avg_hold_period': (0.0, 30.0),     # 0-30 days
        'long_short_ratio': (-1.0, 1.0),    # -1 (all short) to +1 (all long)
        'return_autocorr': (-1.0, 1.0),     # Momentum vs mean-reversion
        'drawdown_depth': (0.0, 0.5),       # 0-50% drawdown
        'benchmark_corr': (-1.0, 1.0),      # Market correlation
        'signal_variance': (0.0, 0.1)       # Signal volatility
    })


@dataclass
class EliteEntry:
    """Entry in the MAP-Elites grid."""
    genome_id: str
    fitness: float                    # Primary fitness (Sortino)
    fitness_vector: Tuple[float, ...] # All objectives
    behavior: np.ndarray              # Behavior vector
    cell_coords: Tuple[int, ...]      # Grid cell coordinates
    generation_added: int = 0
    times_replaced: int = 0           # How many times this cell was updated


class MAPElitesGrid:
    """
    MAP-Elites grid for maintaining diverse elite solutions.

    The grid discretizes the behavioral feature space into cells.
    Each cell stores the best-performing solution with that behavior.
    This guarantees diversity across different trading styles.

    Usage:
        grid = MAPElitesGrid(config)
        grid.maybe_add(genome, fitness_vector, behavior, generation)
        elites = grid.get_all_elites()
        coverage = grid.get_coverage()
    """

    def __init__(self, config: MapElitesConfig = None):
        """
        Initialize MAP-Elites grid.

        Args:
            config: Grid configuration
        """
        self.config = config or MapElitesConfig()

        # Calculate grid shape
        self.n_dims = len(self.config.dimensions)
        self.resolution = self.config.resolution
        self.grid_shape = tuple([self.resolution] * self.n_dims)
        self.total_cells = self.resolution ** self.n_dims

        # Initialize grid (None = empty cell)
        self.grid: Dict[Tuple[int, ...], EliteEntry] = {}

        # Track dimension indices in behavior vector
        self._dim_indices = self._get_dimension_indices()

        # Statistics
        self.total_attempts = 0
        self.successful_adds = 0
        self.replacements = 0

        logger.info(f"Initialized MAP-Elites grid: {self.n_dims}D, "
                    f"{self.resolution} bins/dim, {self.total_cells} total cells")

    def _get_dimension_indices(self) -> Dict[str, int]:
        """Map dimension names to behavior vector indices."""
        behavior_dims = [
            'trade_frequency',
            'avg_hold_period',
            'long_short_ratio',
            'return_autocorr',
            'drawdown_depth',
            'benchmark_corr',
            'signal_variance'
        ]

        indices = {}
        for dim in self.config.dimensions:
            if dim in behavior_dims:
                indices[dim] = behavior_dims.index(dim)
            else:
                logger.warning(f"Unknown dimension '{dim}', defaulting to 0")
                indices[dim] = 0

        return indices

    def _behavior_to_cell(self, behavior: np.ndarray) -> Tuple[int, ...]:
        """
        Convert behavior vector to grid cell coordinates.

        Args:
            behavior: Normalized behavior vector (7 dimensions)

        Returns:
            Tuple of cell indices for each grid dimension
        """
        coords = []

        for dim_name in self.config.dimensions:
            idx = self._dim_indices.get(dim_name, 0)

            # Get value and feature range
            if idx < len(behavior):
                value = behavior[idx]
            else:
                value = 0.0

            # Get range for this dimension
            low, high = self.config.feature_ranges.get(
                dim_name, (0.0, 1.0)
            )

            # Normalize to [0, 1]
            if high > low:
                normalized = (value - low) / (high - low)
            else:
                normalized = 0.5

            # Clamp and discretize
            normalized = max(0.0, min(1.0, normalized))
            bin_idx = int(normalized * (self.resolution - 1))
            bin_idx = max(0, min(self.resolution - 1, bin_idx))

            coords.append(bin_idx)

        return tuple(coords)

    def maybe_add(self,
                  genome_id: str,
                  fitness: float,
                  fitness_vector: Tuple[float, ...],
                  behavior: np.ndarray,
                  generation: int) -> bool:
        """
        Try to add a solution to the grid.

        Solution is added if:
        1. The cell is empty, OR
        2. The solution has better fitness than current occupant

        Args:
            genome_id: Unique genome identifier
            fitness: Primary fitness score (Sortino)
            fitness_vector: All objective values
            behavior: Behavior vector (7D)
            generation: Current generation number

        Returns:
            True if solution was added to grid
        """
        self.total_attempts += 1

        # Get cell coordinates
        cell = self._behavior_to_cell(behavior)

        # Check if cell is empty or solution is better
        current = self.grid.get(cell)

        if current is None:
            # Empty cell - add solution
            entry = EliteEntry(
                genome_id=genome_id,
                fitness=fitness,
                fitness_vector=fitness_vector,
                behavior=behavior.copy(),
                cell_coords=cell,
                generation_added=generation,
                times_replaced=0
            )
            self.grid[cell] = entry
            self.successful_adds += 1
            return True

        elif fitness > current.fitness:
            # Better solution - replace
            entry = EliteEntry(
                genome_id=genome_id,
                fitness=fitness,
                fitness_vector=fitness_vector,
                behavior=behavior.copy(),
                cell_coords=cell,
                generation_added=generation,
                times_replaced=current.times_replaced + 1
            )
            self.grid[cell] = entry
            self.successful_adds += 1
            self.replacements += 1
            return True

        return False

    def get_elite(self, cell: Tuple[int, ...]) -> Optional[EliteEntry]:
        """Get elite at specific cell coordinates."""
        return self.grid.get(cell)

    def get_all_elites(self) -> List[EliteEntry]:
        """Get all elites in the grid."""
        return list(self.grid.values())

    def get_elite_ids(self) -> List[str]:
        """Get genome IDs of all elites."""
        return [e.genome_id for e in self.grid.values()]

    def get_coverage(self) -> float:
        """
        Get grid coverage (fraction of cells occupied).

        Returns:
            Coverage ratio (0-1)
        """
        return len(self.grid) / self.total_cells

    def get_quality_stats(self) -> Dict[str, float]:
        """
        Get statistics about elite quality.

        Returns:
            Dictionary with min/max/mean/std of fitness across elites
        """
        if not self.grid:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}

        fitnesses = [e.fitness for e in self.grid.values()]

        return {
            'min': float(np.min(fitnesses)),
            'max': float(np.max(fitnesses)),
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
            'count': len(fitnesses)
        }

    def get_cell_distribution(self) -> np.ndarray:
        """
        Get distribution of occupied cells across dimensions.

        Returns:
            Array of shape (n_dims, resolution) with counts
        """
        distribution = np.zeros((self.n_dims, self.resolution))

        for cell in self.grid.keys():
            for dim_idx, bin_idx in enumerate(cell):
                distribution[dim_idx, bin_idx] += 1

        return distribution

    def sample_elites(self, n: int, method: str = 'uniform') -> List[EliteEntry]:
        """
        Sample elites from the grid.

        Args:
            n: Number of elites to sample
            method: 'uniform' (random), 'quality' (weighted by fitness),
                   'sparse' (prefer underrepresented cells)

        Returns:
            List of sampled elites
        """
        if not self.grid or n <= 0:
            return []

        elites = list(self.grid.values())
        n = min(n, len(elites))

        if method == 'uniform':
            indices = np.random.choice(len(elites), n, replace=False)
            return [elites[i] for i in indices]

        elif method == 'quality':
            # Weight by fitness (softmax)
            fitnesses = np.array([e.fitness for e in elites])
            fitnesses = fitnesses - fitnesses.max()  # Numerical stability
            weights = np.exp(fitnesses)
            weights = weights / weights.sum()
            indices = np.random.choice(len(elites), n, replace=False, p=weights)
            return [elites[i] for i in indices]

        elif method == 'sparse':
            # Weight inversely by times_replaced (favor stable cells)
            weights = np.array([1.0 / (e.times_replaced + 1) for e in elites])
            weights = weights / weights.sum()
            indices = np.random.choice(len(elites), n, replace=False, p=weights)
            return [elites[i] for i in indices]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def get_neighbors(self, cell: Tuple[int, ...], radius: int = 1) -> List[EliteEntry]:
        """
        Get elites from neighboring cells.

        Args:
            cell: Center cell coordinates
            radius: Manhattan distance radius

        Returns:
            List of elites in neighboring cells
        """
        neighbors = []

        # Generate all cells within Manhattan distance
        for offset in self._generate_offsets(radius):
            neighbor_cell = tuple(
                max(0, min(self.resolution - 1, c + o))
                for c, o in zip(cell, offset)
            )

            if neighbor_cell != cell and neighbor_cell in self.grid:
                neighbors.append(self.grid[neighbor_cell])

        return neighbors

    def _generate_offsets(self, radius: int) -> List[Tuple[int, ...]]:
        """Generate all offset tuples within Manhattan distance."""
        from itertools import product

        ranges = [range(-radius, radius + 1)] * self.n_dims
        offsets = []

        for offset in product(*ranges):
            if sum(abs(o) for o in offset) <= radius and offset != tuple([0] * self.n_dims):
                offsets.append(offset)

        return offsets

    def clear(self):
        """Clear all entries from the grid."""
        self.grid.clear()
        self.total_attempts = 0
        self.successful_adds = 0
        self.replacements = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grid to dictionary."""
        return {
            'config': {
                'dimensions': self.config.dimensions,
                'resolution': self.config.resolution,
                'feature_ranges': {
                    k: list(v) for k, v in self.config.feature_ranges.items()
                }
            },
            'stats': {
                'total_attempts': self.total_attempts,
                'successful_adds': self.successful_adds,
                'replacements': self.replacements,
                'coverage': self.get_coverage(),
                'occupied_cells': len(self.grid)
            },
            'elites': [
                {
                    'genome_id': e.genome_id,
                    'fitness': e.fitness,
                    'fitness_vector': list(e.fitness_vector) if e.fitness_vector else [],
                    'behavior': e.behavior.tolist(),
                    'cell': list(e.cell_coords),
                    'generation_added': e.generation_added,
                    'times_replaced': e.times_replaced
                }
                for e in self.grid.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MAPElitesGrid':
        """Deserialize grid from dictionary."""
        config = MapElitesConfig(
            dimensions=data['config']['dimensions'],
            resolution=data['config']['resolution'],
            feature_ranges={
                k: tuple(v) for k, v in data['config']['feature_ranges'].items()
            }
        )

        grid = cls(config)
        grid.total_attempts = data['stats']['total_attempts']
        grid.successful_adds = data['stats']['successful_adds']
        grid.replacements = data['stats']['replacements']

        # Restore elites
        for e_data in data.get('elites', []):
            entry = EliteEntry(
                genome_id=e_data['genome_id'],
                fitness=e_data['fitness'],
                fitness_vector=tuple(e_data.get('fitness_vector', [])),
                behavior=np.array(e_data['behavior']),
                cell_coords=tuple(e_data['cell']),
                generation_added=e_data.get('generation_added', 0),
                times_replaced=e_data.get('times_replaced', 0)
            )
            grid.grid[entry.cell_coords] = entry

        return grid

    def __len__(self) -> int:
        return len(self.grid)

    def __repr__(self) -> str:
        return (f"MAPElitesGrid({self.n_dims}D, res={self.resolution}, "
                f"coverage={self.get_coverage():.1%}, "
                f"elites={len(self.grid)})")


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_default_grid() -> MAPElitesGrid:
    """Create a default MAP-Elites grid for trading strategies."""
    config = MapElitesConfig(
        dimensions=['trade_frequency', 'avg_hold_period', 'drawdown_depth'],
        resolution=10
    )
    return MAPElitesGrid(config)


def create_high_resolution_grid() -> MAPElitesGrid:
    """Create a high-resolution grid (more cells, slower but more diverse)."""
    config = MapElitesConfig(
        dimensions=['trade_frequency', 'avg_hold_period', 'drawdown_depth'],
        resolution=15
    )
    return MAPElitesGrid(config)


def create_5d_grid() -> MAPElitesGrid:
    """Create a 5-dimensional grid for maximum diversity."""
    config = MapElitesConfig(
        dimensions=[
            'trade_frequency',
            'avg_hold_period',
            'long_short_ratio',
            'drawdown_depth',
            'benchmark_corr'
        ],
        resolution=5  # Lower resolution due to more dimensions
    )
    return MAPElitesGrid(config)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing MAP-Elites Grid...")
    print("=" * 60)

    # Create grid
    grid = create_default_grid()
    print(f"\nGrid created: {grid}")
    print(f"Total cells: {grid.total_cells}")

    # Add some test solutions
    print("\nAdding test solutions...")

    np.random.seed(42)
    for i in range(100):
        # Random behavior vector
        behavior = np.random.rand(7)

        # Random fitness
        fitness = np.random.rand() * 2 - 0.5  # Range [-0.5, 1.5]

        added = grid.maybe_add(
            genome_id=f"genome_{i:03d}",
            fitness=fitness,
            fitness_vector=(fitness, -0.1, -0.05, 0.3),
            behavior=behavior,
            generation=i // 10
        )

        if i < 10:
            print(f"  {i}: behavior={behavior[:3]}, fitness={fitness:.3f}, added={added}")

    print(f"\nAfter adding 100 solutions:")
    print(f"  Grid: {grid}")
    print(f"  Coverage: {grid.get_coverage():.1%}")
    print(f"  Quality stats: {grid.get_quality_stats()}")

    # Test sampling
    print("\nSampling elites...")
    uniform_sample = grid.sample_elites(5, method='uniform')
    print(f"  Uniform sample: {[e.genome_id for e in uniform_sample]}")

    quality_sample = grid.sample_elites(5, method='quality')
    print(f"  Quality sample: {[e.genome_id for e in quality_sample]}")

    # Test serialization
    print("\nTesting serialization...")
    grid_dict = grid.to_dict()
    print(f"  Serialized size: {len(json.dumps(grid_dict))} bytes")

    restored = MAPElitesGrid.from_dict(grid_dict)
    print(f"  Restored: {restored}")
    print(f"  Elites match: {len(restored.grid) == len(grid.grid)}")

    # Test cell distribution
    print("\nCell distribution:")
    dist = grid.get_cell_distribution()
    for i, dim in enumerate(grid.config.dimensions):
        print(f"  {dim}: {dist[i].astype(int).tolist()}")

    print("\nTest complete!")
