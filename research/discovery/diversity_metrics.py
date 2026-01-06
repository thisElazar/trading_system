"""
Diversity Metrics and Auto-Intervention
========================================
Comprehensive diversity monitoring with automatic convergence detection
and diversity injection triggers.

Key components:
- GenotypeDiversity: Tree structure similarity, operator entropy
- PhenotypeDiversity: Behavioral diversity (from novelty_search)
- DiversityMonitor: Unified tracking with auto-intervention triggers

Reference: Preventing premature convergence in genetic algorithms
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import Counter
import math

import numpy as np
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from .strategy_genome import StrategyGenome

logger = logging.getLogger(__name__)


# =============================================================================
# GENOTYPE DIVERSITY METRICS
# =============================================================================

@dataclass
class GenotypeMetrics:
    """Metrics describing genetic diversity of a population."""

    # Tree structure metrics
    avg_tree_size: float = 0.0          # Average nodes per tree
    tree_size_variance: float = 0.0     # Variance in tree sizes
    avg_tree_depth: float = 0.0         # Average tree depth
    tree_depth_variance: float = 0.0    # Variance in depths

    # Operator distribution
    operator_entropy: float = 0.0       # Shannon entropy of operator usage
    unique_operators: int = 0           # Number of distinct operators used
    operator_concentration: float = 0.0 # Gini coefficient of operator usage

    # Structural similarity
    avg_edit_distance: float = 0.0      # Average tree edit distance
    unique_tree_ratio: float = 0.0      # Fraction of unique tree strings

    # Overall genotype diversity score (0-1, higher = more diverse)
    genotype_diversity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/serialization."""
        return {
            'avg_tree_size': self.avg_tree_size,
            'tree_size_variance': self.tree_size_variance,
            'avg_tree_depth': self.avg_tree_depth,
            'tree_depth_variance': self.tree_depth_variance,
            'operator_entropy': self.operator_entropy,
            'unique_operators': self.unique_operators,
            'operator_concentration': self.operator_concentration,
            'avg_edit_distance': self.avg_edit_distance,
            'unique_tree_ratio': self.unique_tree_ratio,
            'genotype_diversity': self.genotype_diversity
        }


def calculate_operator_entropy(genomes: List['StrategyGenome']) -> Tuple[float, int, float]:
    """
    Calculate Shannon entropy of operator usage across population.

    High entropy = operators used uniformly (diverse)
    Low entropy = few operators dominate (converged)

    Returns:
        Tuple of (entropy, unique_operators, gini_coefficient)
    """
    if not genomes:
        return 0.0, 0, 1.0

    # Count all operators across all trees
    operator_counts = Counter()

    for genome in genomes:
        for tree in [genome.entry_tree, genome.exit_tree, genome.position_tree,
                     genome.stop_loss_tree, genome.target_tree]:
            for node in tree:
                # Get operator/terminal name
                if hasattr(node, 'name'):
                    operator_counts[node.name] += 1
                else:
                    # Ephemeral constant or primitive
                    operator_counts[str(type(node).__name__)] += 1

    if not operator_counts:
        return 0.0, 0, 1.0

    # Calculate Shannon entropy
    total = sum(operator_counts.values())
    probabilities = [count / total for count in operator_counts.values()]

    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    # Normalize to 0-1 (max entropy is log2(n) where n = number of unique operators)
    max_entropy = math.log2(len(operator_counts)) if len(operator_counts) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Calculate Gini coefficient (concentration)
    sorted_counts = sorted(operator_counts.values())
    n = len(sorted_counts)
    if n == 0 or total == 0:
        gini = 1.0
    else:
        cumulative = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumulative) / total) / n
        gini = max(0.0, min(1.0, gini))

    return normalized_entropy, len(operator_counts), gini


def calculate_tree_size_stats(genomes: List['StrategyGenome']) -> Tuple[float, float, float, float]:
    """
    Calculate tree size and depth statistics.

    Returns:
        Tuple of (avg_size, size_var, avg_depth, depth_var)
    """
    if not genomes:
        return 0.0, 0.0, 0.0, 0.0

    sizes = []
    depths = []

    for genome in genomes:
        for tree in [genome.entry_tree, genome.exit_tree, genome.position_tree,
                     genome.stop_loss_tree, genome.target_tree]:
            sizes.append(len(tree))
            depths.append(tree.height)

    avg_size = np.mean(sizes)
    size_var = np.var(sizes)
    avg_depth = np.mean(depths)
    depth_var = np.var(depths)

    return avg_size, size_var, avg_depth, depth_var


def calculate_tree_string_similarity(genomes: List['StrategyGenome']) -> Tuple[float, float]:
    """
    Calculate structural similarity based on tree string representations.

    Uses string-based comparison for efficiency (true edit distance is O(n^2)).

    Returns:
        Tuple of (avg_distance, unique_ratio)
    """
    if len(genomes) < 2:
        return 0.0, 1.0

    # Get string representations of entry trees (most important)
    tree_strings = [str(g.entry_tree) for g in genomes]

    # Calculate unique ratio
    unique_count = len(set(tree_strings))
    unique_ratio = unique_count / len(tree_strings)

    # Calculate average Levenshtein-like distance (simplified)
    # Using character-level differences as proxy for structure
    if len(tree_strings) > 100:
        # Sample for efficiency
        sample_idx = np.random.choice(len(tree_strings), 100, replace=False)
        tree_strings = [tree_strings[i] for i in sample_idx]

    total_distance = 0
    count = 0

    for i in range(len(tree_strings)):
        for j in range(i + 1, min(i + 20, len(tree_strings))):  # Limit pairs for efficiency
            # Jaccard distance on character trigrams
            s1, s2 = tree_strings[i], tree_strings[j]
            trigrams1 = set(s1[k:k+3] for k in range(len(s1)-2)) if len(s1) >= 3 else {s1}
            trigrams2 = set(s2[k:k+3] for k in range(len(s2)-2)) if len(s2) >= 3 else {s2}

            if trigrams1 or trigrams2:
                intersection = len(trigrams1 & trigrams2)
                union = len(trigrams1 | trigrams2)
                jaccard = 1 - (intersection / union) if union > 0 else 1.0
                total_distance += jaccard
                count += 1

    avg_distance = total_distance / count if count > 0 else 0.0

    return avg_distance, unique_ratio


def calculate_genotype_diversity(genomes: List['StrategyGenome']) -> GenotypeMetrics:
    """
    Calculate comprehensive genotype diversity metrics for a population.

    Args:
        genomes: List of strategy genomes

    Returns:
        GenotypeMetrics with all diversity measures
    """
    if not genomes:
        return GenotypeMetrics()

    # Calculate component metrics
    avg_size, size_var, avg_depth, depth_var = calculate_tree_size_stats(genomes)
    op_entropy, unique_ops, gini = calculate_operator_entropy(genomes)
    avg_dist, unique_ratio = calculate_tree_string_similarity(genomes)

    # Calculate overall genotype diversity score
    # Weighted combination of normalized metrics

    # Size variance contribution (normalized by expected variance)
    size_var_norm = min(size_var / 100, 1.0)  # Cap at 1.0

    # Depth variance contribution
    depth_var_norm = min(depth_var / 10, 1.0)

    # Overall score (weighted average)
    genotype_diversity = (
        0.30 * op_entropy +           # Operator diversity
        0.25 * unique_ratio +          # Unique tree structures
        0.20 * avg_dist +              # Structural distance
        0.15 * size_var_norm +         # Size variety
        0.10 * depth_var_norm          # Depth variety
    )

    return GenotypeMetrics(
        avg_tree_size=avg_size,
        tree_size_variance=size_var,
        avg_tree_depth=avg_depth,
        tree_depth_variance=depth_var,
        operator_entropy=op_entropy,
        unique_operators=unique_ops,
        operator_concentration=gini,
        avg_edit_distance=avg_dist,
        unique_tree_ratio=unique_ratio,
        genotype_diversity=genotype_diversity
    )


# =============================================================================
# DIVERSITY MONITOR WITH AUTO-INTERVENTION
# =============================================================================

@dataclass
class DiversityThresholds:
    """Thresholds for triggering diversity intervention."""

    # Minimum acceptable diversity levels
    min_genotype_diversity: float = 0.25      # Below this = inject diversity
    min_phenotype_diversity: float = 0.10     # Behavioral diversity minimum
    min_unique_tree_ratio: float = 0.20       # At least 20% unique trees
    min_operator_entropy: float = 0.30        # Operator usage entropy

    # Stagnation detection
    stagnation_generations: int = 10          # Generations without improvement
    stagnation_improvement_threshold: float = 0.01  # Min fitness improvement

    # Intervention parameters
    injection_ratio: float = 0.30             # Replace 30% on intervention
    mutation_rate_boost: float = 2.0          # Multiply mutation rate by this
    boost_duration: int = 5                   # Generations to maintain boost


@dataclass
class DiversitySnapshot:
    """Snapshot of diversity metrics at a point in time."""
    generation: int
    genotype_metrics: GenotypeMetrics
    phenotype_diversity: float
    archive_diversity: float
    best_fitness: float
    intervention_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'genotype': self.genotype_metrics.to_dict(),
            'phenotype_diversity': self.phenotype_diversity,
            'archive_diversity': self.archive_diversity,
            'best_fitness': self.best_fitness,
            'intervention_triggered': self.intervention_triggered
        }


class DiversityMonitor:
    """
    Monitors population diversity and triggers automatic intervention
    when convergence is detected.

    Tracks both genotype (tree structure) and phenotype (behavior) diversity.
    Triggers diversity injection when thresholds are breached.
    """

    def __init__(self, thresholds: DiversityThresholds = None):
        """
        Initialize diversity monitor.

        Args:
            thresholds: Thresholds for intervention triggers
        """
        self.thresholds = thresholds or DiversityThresholds()

        # History tracking
        self.history: List[DiversitySnapshot] = []
        self.best_fitness_ever: float = -float('inf')
        self.generations_since_improvement: int = 0

        # Intervention state
        self.mutation_boost_active: bool = False
        self.mutation_boost_remaining: int = 0
        self.total_interventions: int = 0

    def update(self,
               generation: int,
               genomes: List['StrategyGenome'],
               phenotype_diversity: float,
               archive_diversity: float,
               best_fitness: float) -> Tuple[bool, str]:
        """
        Update diversity metrics and check for intervention triggers.

        Args:
            generation: Current generation number
            genomes: Current population
            phenotype_diversity: Behavioral diversity score
            archive_diversity: Novelty archive diversity
            best_fitness: Best fitness in population

        Returns:
            Tuple of (should_intervene, reason)
        """
        # Calculate genotype metrics
        genotype_metrics = calculate_genotype_diversity(genomes)

        # Track fitness improvement
        if best_fitness > self.best_fitness_ever + self.thresholds.stagnation_improvement_threshold:
            self.best_fitness_ever = best_fitness
            self.generations_since_improvement = 0
        else:
            self.generations_since_improvement += 1

        # Check intervention triggers
        should_intervene = False
        reasons = []

        # Check genotype diversity
        if genotype_metrics.genotype_diversity < self.thresholds.min_genotype_diversity:
            should_intervene = True
            reasons.append(f"genotype_div={genotype_metrics.genotype_diversity:.3f}<{self.thresholds.min_genotype_diversity}")

        # Check phenotype diversity
        if phenotype_diversity < self.thresholds.min_phenotype_diversity:
            should_intervene = True
            reasons.append(f"phenotype_div={phenotype_diversity:.3f}<{self.thresholds.min_phenotype_diversity}")

        # Check unique tree ratio
        if genotype_metrics.unique_tree_ratio < self.thresholds.min_unique_tree_ratio:
            should_intervene = True
            reasons.append(f"unique_ratio={genotype_metrics.unique_tree_ratio:.3f}<{self.thresholds.min_unique_tree_ratio}")

        # Check operator entropy
        if genotype_metrics.operator_entropy < self.thresholds.min_operator_entropy:
            should_intervene = True
            reasons.append(f"op_entropy={genotype_metrics.operator_entropy:.3f}<{self.thresholds.min_operator_entropy}")

        # Check stagnation
        if self.generations_since_improvement >= self.thresholds.stagnation_generations:
            should_intervene = True
            reasons.append(f"stagnation={self.generations_since_improvement}gen")

        # Record snapshot
        snapshot = DiversitySnapshot(
            generation=generation,
            genotype_metrics=genotype_metrics,
            phenotype_diversity=phenotype_diversity,
            archive_diversity=archive_diversity,
            best_fitness=best_fitness,
            intervention_triggered=should_intervene
        )
        self.history.append(snapshot)

        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-500:]

        # Update intervention tracking
        if should_intervene:
            self.total_interventions += 1
            self.mutation_boost_active = True
            self.mutation_boost_remaining = self.thresholds.boost_duration
            self.generations_since_improvement = 0  # Reset stagnation counter

        # Manage mutation boost decay
        if self.mutation_boost_remaining > 0:
            self.mutation_boost_remaining -= 1
        else:
            self.mutation_boost_active = False

        reason_str = "; ".join(reasons) if reasons else "none"

        if should_intervene:
            logger.warning(f"Diversity intervention triggered at gen {generation}: {reason_str}")

        return should_intervene, reason_str

    def get_current_mutation_rate_multiplier(self) -> float:
        """Get current mutation rate multiplier (boosted if intervention active)."""
        if self.mutation_boost_active:
            return self.thresholds.mutation_rate_boost
        return 1.0

    def get_injection_ratio(self) -> float:
        """Get ratio of population to replace during intervention."""
        return self.thresholds.injection_ratio

    def get_latest_metrics(self) -> Optional[DiversitySnapshot]:
        """Get most recent diversity snapshot."""
        return self.history[-1] if self.history else None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of diversity monitoring state."""
        latest = self.get_latest_metrics()

        return {
            'total_generations_monitored': len(self.history),
            'total_interventions': self.total_interventions,
            'generations_since_improvement': self.generations_since_improvement,
            'best_fitness_ever': self.best_fitness_ever,
            'mutation_boost_active': self.mutation_boost_active,
            'current_genotype_diversity': latest.genotype_metrics.genotype_diversity if latest else 0.0,
            'current_phenotype_diversity': latest.phenotype_diversity if latest else 0.0,
            'current_unique_tree_ratio': latest.genotype_metrics.unique_tree_ratio if latest else 0.0
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize monitor state."""
        return {
            'thresholds': {
                'min_genotype_diversity': self.thresholds.min_genotype_diversity,
                'min_phenotype_diversity': self.thresholds.min_phenotype_diversity,
                'min_unique_tree_ratio': self.thresholds.min_unique_tree_ratio,
                'min_operator_entropy': self.thresholds.min_operator_entropy,
                'stagnation_generations': self.thresholds.stagnation_generations,
                'injection_ratio': self.thresholds.injection_ratio,
                'mutation_rate_boost': self.thresholds.mutation_rate_boost
            },
            'state': {
                'best_fitness_ever': self.best_fitness_ever,
                'generations_since_improvement': self.generations_since_improvement,
                'total_interventions': self.total_interventions,
                'mutation_boost_active': self.mutation_boost_active,
                'mutation_boost_remaining': self.mutation_boost_remaining
            },
            'history_length': len(self.history),
            'recent_snapshots': [s.to_dict() for s in self.history[-10:]]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiversityMonitor':
        """Deserialize monitor state."""
        thresholds = DiversityThresholds(
            min_genotype_diversity=data['thresholds']['min_genotype_diversity'],
            min_phenotype_diversity=data['thresholds']['min_phenotype_diversity'],
            min_unique_tree_ratio=data['thresholds']['min_unique_tree_ratio'],
            min_operator_entropy=data['thresholds']['min_operator_entropy'],
            stagnation_generations=data['thresholds']['stagnation_generations'],
            injection_ratio=data['thresholds']['injection_ratio'],
            mutation_rate_boost=data['thresholds']['mutation_rate_boost']
        )

        monitor = cls(thresholds=thresholds)
        monitor.best_fitness_ever = data['state']['best_fitness_ever']
        monitor.generations_since_improvement = data['state']['generations_since_improvement']
        monitor.total_interventions = data['state']['total_interventions']
        monitor.mutation_boost_active = data['state']['mutation_boost_active']
        monitor.mutation_boost_remaining = data['state']['mutation_boost_remaining']

        return monitor


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_diversity_metrics(metrics: GenotypeMetrics, phenotype_div: float,
                          archive_div: float, generation: int):
    """Log diversity metrics in a structured format."""
    logger.info(
        f"[Gen {generation}] Diversity: "
        f"genotype={metrics.genotype_diversity:.3f} "
        f"(entropy={metrics.operator_entropy:.3f}, unique={metrics.unique_tree_ratio:.3f}), "
        f"phenotype={phenotype_div:.3f}, archive={archive_div:.3f}"
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Diversity Metrics...")
    print("=" * 60)

    # Create mock genomes for testing
    from .strategy_genome import GenomeFactory

    factory = GenomeFactory()

    # Create population
    print("\nCreating test population of 20 genomes...")
    genomes = [factory.create_random_genome(generation=0) for _ in range(20)]

    # Calculate diversity
    print("\nCalculating genotype diversity metrics...")
    metrics = calculate_genotype_diversity(genomes)

    print(f"\nGenotype Metrics:")
    print(f"  Avg tree size: {metrics.avg_tree_size:.1f}")
    print(f"  Size variance: {metrics.tree_size_variance:.1f}")
    print(f"  Avg depth: {metrics.avg_tree_depth:.1f}")
    print(f"  Depth variance: {metrics.tree_depth_variance:.1f}")
    print(f"  Operator entropy: {metrics.operator_entropy:.3f}")
    print(f"  Unique operators: {metrics.unique_operators}")
    print(f"  Operator concentration: {metrics.operator_concentration:.3f}")
    print(f"  Avg edit distance: {metrics.avg_edit_distance:.3f}")
    print(f"  Unique tree ratio: {metrics.unique_tree_ratio:.3f}")
    print(f"  Overall genotype diversity: {metrics.genotype_diversity:.3f}")

    # Test DiversityMonitor
    print("\n" + "=" * 60)
    print("Testing DiversityMonitor...")

    monitor = DiversityMonitor()

    # Simulate several generations
    for gen in range(15):
        should_intervene, reason = monitor.update(
            generation=gen,
            genomes=genomes,
            phenotype_diversity=0.3 - gen * 0.02,  # Declining diversity
            archive_diversity=0.4,
            best_fitness=0.5 + gen * 0.001  # Slight improvement
        )

        if should_intervene:
            print(f"  Gen {gen}: INTERVENTION - {reason}")
        else:
            print(f"  Gen {gen}: OK (genotype={metrics.genotype_diversity:.3f})")

    print(f"\nMonitor summary: {monitor.get_summary()}")

    print("\nTest complete!")
