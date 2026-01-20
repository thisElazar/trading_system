"""
Strategy Genome
===============
Multi-tree genome representation for evolved trading strategies.

Each genome contains multiple GP trees:
- entry_tree: Boolean expression for entry conditions
- exit_tree: Boolean expression for exit conditions
- position_tree: Float expression for position sizing
- stop_loss_tree: Float expression for stop loss percentage
- target_tree: Float expression for target percentage
"""

import uuid
import json
import random
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from deap import gp, tools

from .gp_core import (
    create_primitive_set,
    create_boolean_primitive_set,
    create_toolbox,
    setup_deap_types,
    tree_to_string,
    tree_complexity,
    evaluate_tree,
    BoolType,
    FloatType
)
from .config import EvolutionConfig, PrimitiveConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyGenome:
    """
    Complete genome for an evolved trading strategy.

    Contains multiple GP trees representing different strategy components:
    - entry_tree: When to buy (boolean output)
    - exit_tree: When to sell (boolean output)
    - position_tree: How much to buy (0-1 output)
    - stop_loss_tree: Stop loss distance (% output)
    - target_tree: Target distance (% output)
    """
    entry_tree: gp.PrimitiveTree
    exit_tree: gp.PrimitiveTree
    position_tree: gp.PrimitiveTree
    stop_loss_tree: gp.PrimitiveTree
    target_tree: gp.PrimitiveTree

    # Metadata
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Self-adaptive mutation parameters (evolve alongside strategy)
    # These rates are subject to mutation themselves (meta-evolution)
    mutation_rate: float = field(default_factory=lambda: random.uniform(0.15, 0.35))
    crossover_rate: float = field(default_factory=lambda: random.uniform(0.6, 0.9))

    # Cached fitness (set by evaluator)
    fitness_values: Optional[Tuple[float, ...]] = None
    backtest_result: Optional[Any] = None

    def __post_init__(self):
        """Validate genome after creation."""
        pass

    @property
    def total_complexity(self) -> int:
        """Total number of nodes across all trees."""
        return (tree_complexity(self.entry_tree) +
                tree_complexity(self.exit_tree) +
                tree_complexity(self.position_tree) +
                tree_complexity(self.stop_loss_tree) +
                tree_complexity(self.target_tree))

    @property
    def max_depth(self) -> int:
        """Maximum depth across all trees."""
        return max(
            self.entry_tree.height,
            self.exit_tree.height,
            self.position_tree.height,
            self.stop_loss_tree.height,
            self.target_tree.height
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        return {
            'genome_id': self.genome_id,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'created_at': self.created_at.isoformat(),
            'trees': {
                'entry': str(self.entry_tree),
                'exit': str(self.exit_tree),
                'position': str(self.position_tree),
                'stop_loss': str(self.stop_loss_tree),
                'target': str(self.target_tree)
            },
            'complexity': self.total_complexity,
            'max_depth': self.max_depth,
            'fitness_values': self.fitness_values,
            # Self-adaptive parameters
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }

    def __str__(self) -> str:
        return f"StrategyGenome({self.genome_id}, gen={self.generation}, complexity={self.total_complexity})"

    def __repr__(self) -> str:
        return self.__str__()


class GenomeFactory:
    """
    Factory for creating and manipulating strategy genomes.

    Handles:
    - Random genome generation
    - Crossover between genomes
    - Mutation of genomes
    - Serialization/deserialization
    """

    def __init__(self, config: EvolutionConfig = None, prim_config: PrimitiveConfig = None):
        """
        Initialize genome factory.

        Args:
            config: Evolution configuration
            prim_config: Primitive set configuration
        """
        self.config = config or EvolutionConfig()
        self.prim_config = prim_config or PrimitiveConfig()

        # Create primitive sets
        self.float_pset = create_primitive_set(self.prim_config)
        self.bool_pset = create_boolean_primitive_set(self.prim_config)

        # Setup DEAP types
        setup_deap_types()

        # Create toolboxes
        self.float_toolbox = create_toolbox(self.float_pset, self.config)
        self.bool_toolbox = create_toolbox(self.bool_pset, self.config)

        # Failure tracking for monitoring genetic operation health
        self.crossover_failures = 0
        self.crossover_attempts = 0
        self.mutation_failures = 0
        self.mutation_attempts = 0

    def create_random_genome(self, generation: int = 0) -> StrategyGenome:
        """
        Create a random genome with all trees.

        Args:
            generation: Generation number for this genome

        Returns:
            New random StrategyGenome
        """
        # Entry and exit trees need boolean output
        entry_tree = self._create_bool_tree()
        exit_tree = self._create_bool_tree()

        # Position, stop, target trees need float output
        position_tree = self._create_float_tree()
        stop_loss_tree = self._create_float_tree()
        target_tree = self._create_float_tree()

        return StrategyGenome(
            entry_tree=entry_tree,
            exit_tree=exit_tree,
            position_tree=position_tree,
            stop_loss_tree=stop_loss_tree,
            target_tree=target_tree,
            generation=generation
        )

    def _create_bool_tree(self) -> gp.PrimitiveTree:
        """Create a random boolean tree."""
        return self.bool_toolbox.individual()

    def _create_float_tree(self) -> gp.PrimitiveTree:
        """Create a random float tree."""
        return self.float_toolbox.individual()

    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome,
                  generation: int) -> Tuple[StrategyGenome, StrategyGenome]:
        """
        Perform crossover between two parent genomes.

        Each tree is crossed over independently with probability crossover_rate.

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            generation: Generation number for offspring

        Returns:
            Tuple of two offspring genomes
        """
        import random
        import copy

        # Deep copy trees
        entry1 = copy.deepcopy(parent1.entry_tree)
        entry2 = copy.deepcopy(parent2.entry_tree)
        exit1 = copy.deepcopy(parent1.exit_tree)
        exit2 = copy.deepcopy(parent2.exit_tree)
        pos1 = copy.deepcopy(parent1.position_tree)
        pos2 = copy.deepcopy(parent2.position_tree)
        stop1 = copy.deepcopy(parent1.stop_loss_tree)
        stop2 = copy.deepcopy(parent2.stop_loss_tree)
        target1 = copy.deepcopy(parent1.target_tree)
        target2 = copy.deepcopy(parent2.target_tree)

        # Crossover each tree pair with probability
        if random.random() < self.config.crossover_rate:
            entry1, entry2 = self._crossover_trees(entry1, entry2, self.bool_pset)
        if random.random() < self.config.crossover_rate:
            exit1, exit2 = self._crossover_trees(exit1, exit2, self.bool_pset)
        if random.random() < self.config.crossover_rate:
            pos1, pos2 = self._crossover_trees(pos1, pos2, self.float_pset)
        if random.random() < self.config.crossover_rate:
            stop1, stop2 = self._crossover_trees(stop1, stop2, self.float_pset)
        if random.random() < self.config.crossover_rate:
            target1, target2 = self._crossover_trees(target1, target2, self.float_pset)

        # Create offspring genomes
        child1 = StrategyGenome(
            entry_tree=entry1,
            exit_tree=exit1,
            position_tree=pos1,
            stop_loss_tree=stop1,
            target_tree=target1,
            generation=generation,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        child2 = StrategyGenome(
            entry_tree=entry2,
            exit_tree=exit2,
            position_tree=pos2,
            stop_loss_tree=stop2,
            target_tree=target2,
            generation=generation,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        return child1, child2

    def _crossover_trees(self, tree1: gp.PrimitiveTree, tree2: gp.PrimitiveTree,
                         pset: gp.PrimitiveSetTyped) -> Tuple[gp.PrimitiveTree, gp.PrimitiveTree]:
        """
        Perform one-point crossover on two trees.

        STGP crossover only swaps subtrees with matching return types.
        Failures are logged and tracked for monitoring.
        """
        self.crossover_attempts += 1

        try:
            return gp.cxOnePoint(tree1, tree2)
        except IndexError:
            # Expected: no compatible crossover points (trees too small or no common types)
            # This is normal for small trees - don't count as failure
            logger.debug(f"Crossover skipped: no compatible crossover points "
                        f"(tree sizes: {len(tree1)}, {len(tree2)})")
            return tree1, tree2
        except TypeError as e:
            # STGP type mismatch - this should NOT happen if types are enforced
            self.crossover_failures += 1
            logger.warning(f"STGP crossover type error (possible type violation): {e} | "
                          f"tree1_root={tree1[0].ret if tree1 else 'empty'}, "
                          f"tree2_root={tree2[0].ret if tree2 else 'empty'}")
            return tree1, tree2
        except Exception as e:
            # Unexpected error - log with full context for debugging
            self.crossover_failures += 1
            logger.error(f"Unexpected crossover failure: {type(e).__name__}: {e} | "
                        f"tree1={str(tree1)[:100]}, tree2={str(tree2)[:100]}")
            return tree1, tree2

    def mutate(self, genome: StrategyGenome, generation: int) -> StrategyGenome:
        """
        Mutate a genome using its self-adaptive mutation rate.

        Each tree is mutated independently with probability based on the genome's
        own mutation_rate (self-adaptive). The mutation_rate itself also evolves.

        Args:
            genome: Genome to mutate
            generation: Current generation

        Returns:
            Mutated genome (new object)
        """
        import random
        import copy

        # Use genome's own adaptive mutation rate (falls back to config if not set)
        mutation_rate = getattr(genome, 'mutation_rate', self.config.mutation_rate)
        crossover_rate = getattr(genome, 'crossover_rate', 0.7)

        # Deep copy all trees
        entry = copy.deepcopy(genome.entry_tree)
        exit_tree = copy.deepcopy(genome.exit_tree)
        position = copy.deepcopy(genome.position_tree)
        stop_loss = copy.deepcopy(genome.stop_loss_tree)
        target = copy.deepcopy(genome.target_tree)

        # Mutate each tree with genome's own probability
        if random.random() < mutation_rate:
            entry = self._mutate_tree(entry, self.bool_pset, self.bool_toolbox)
        if random.random() < mutation_rate:
            exit_tree = self._mutate_tree(exit_tree, self.bool_pset, self.bool_toolbox)
        if random.random() < mutation_rate:
            position = self._mutate_tree(position, self.float_pset, self.float_toolbox)
        if random.random() < mutation_rate:
            stop_loss = self._mutate_tree(stop_loss, self.float_pset, self.float_toolbox)
        if random.random() < mutation_rate:
            target = self._mutate_tree(target, self.float_pset, self.float_toolbox)

        # Self-adaptation: mutate the mutation parameters themselves
        # Use log-normal distribution for multiplicative changes (standard ES approach)
        tau = 1.0 / (2 * 5**0.5)  # Learning rate (1 / sqrt(2*n) where n=5 trees)
        new_mutation_rate = mutation_rate * (1 + tau * random.gauss(0, 1))
        new_crossover_rate = crossover_rate * (1 + tau * random.gauss(0, 1))

        # Clamp to valid ranges
        new_mutation_rate = max(0.05, min(0.5, new_mutation_rate))
        new_crossover_rate = max(0.4, min(0.95, new_crossover_rate))

        mutated = StrategyGenome(
            entry_tree=entry,
            exit_tree=exit_tree,
            position_tree=position,
            stop_loss_tree=stop_loss,
            target_tree=target,
            generation=generation,
            parent_ids=[genome.genome_id]
        )

        # Set the adapted rates
        mutated.mutation_rate = new_mutation_rate
        mutated.crossover_rate = new_crossover_rate

        return mutated

    def _mutate_tree(self, tree: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped,
                     toolbox) -> gp.PrimitiveTree:
        """
        Apply weighted mutation to a single tree.

        STGP mutations preserve type constraints:
        - Subtree: replaces subtree with same-typed new subtree
        - Point: replaces node with same-type/arity alternative
        - Hoist: promotes subtree (may change root type - use carefully)
        - Shrink: reduces tree size

        Failures are logged and tracked for monitoring.
        """
        import random

        self.mutation_attempts += 1
        r = random.random()
        mutation_type = None

        try:
            if r < self.config.subtree_mutation_prob:
                mutation_type = "subtree"
                # DEAP's mutUniform calls expr(pset=pset, type_=type_) to generate replacement subtree
                # Must accept these kwargs and use type_ to maintain STGP constraints
                def gen_typed_subtree(pset, type_):
                    return gp.genFull(pset, min_=0, max_=2, type_=type_)
                result = gp.mutUniform(tree, expr=gen_typed_subtree, pset=pset)
                return result[0] if isinstance(result, tuple) else result

            elif r < self.config.subtree_mutation_prob + self.config.point_mutation_prob:
                mutation_type = "point"
                result = gp.mutNodeReplacement(tree, pset=pset)
                return result[0] if isinstance(result, tuple) else result

            elif r < (self.config.subtree_mutation_prob + self.config.point_mutation_prob +
                      self.config.hoist_mutation_prob):
                mutation_type = "hoist"
                try:
                    tree_len = len(tree)
                except (TypeError, AttributeError):
                    return tree
                if tree_len is None or tree_len <= 2:
                    return tree
                index = random.randrange(1, tree_len)
                slice_ = tree.searchSubtree(index)
                # Validate slice bounds before using (DEAP can return invalid slices)
                if (slice_ is None or slice_.start is None or slice_.stop is None or
                    slice_.start < 0 or slice_.stop > tree_len or slice_.start >= slice_.stop):
                    logger.debug(f"Hoist skipped: invalid slice {slice_} for tree length {tree_len}")
                    return tree
                subtree = tree[slice_]
                # Verify hoist preserves root type (critical for STGP)
                try:
                    subtree_len = len(subtree) if subtree is not None else 0
                except (TypeError, AttributeError):
                    subtree_len = 0
                if subtree_len > 0:
                    subtree_root = subtree[0] if isinstance(subtree, list) else subtree
                    tree_root = tree[0]
                    subtree_ret = getattr(subtree_root, 'ret', None)
                    tree_ret = getattr(tree_root, 'ret', None)
                    if subtree_ret is not None and tree_ret is not None:
                        if subtree_ret != tree_ret:
                            logger.debug(f"Hoist skipped: would change root type from "
                                        f"{tree_ret} to {subtree_ret}")
                            return tree
                    tree[:] = subtree
                return tree

            else:
                mutation_type = "shrink"
                result = gp.mutShrink(tree)
                return result[0] if isinstance(result, tuple) else result

        except IndexError:
            # Expected: tree too small for this mutation type
            logger.debug(f"{mutation_type} mutation skipped: tree too small (size={len(tree)})")
            return tree
        except TypeError as e:
            # STGP type mismatch - should NOT happen if types are enforced
            self.mutation_failures += 1
            logger.warning(f"STGP {mutation_type} mutation type error (possible type violation): {e} | "
                          f"tree_root={tree[0].ret if tree else 'empty'}, tree={str(tree)[:80]}")
            return tree
        except Exception as e:
            # Unexpected error - log with full context
            self.mutation_failures += 1
            logger.error(f"Unexpected {mutation_type} mutation failure: {type(e).__name__}: {e} | "
                        f"tree={str(tree)[:100]}")
            return tree

    def get_operation_stats(self) -> Dict[str, Any]:
        """
        Get statistics on genetic operation success/failure rates.

        Returns dict with crossover and mutation attempt/failure counts
        and failure rates. Use this to monitor STGP health.
        """
        cx_rate = (self.crossover_failures / self.crossover_attempts * 100
                   if self.crossover_attempts > 0 else 0.0)
        mut_rate = (self.mutation_failures / self.mutation_attempts * 100
                    if self.mutation_attempts > 0 else 0.0)

        return {
            'crossover_attempts': self.crossover_attempts,
            'crossover_failures': self.crossover_failures,
            'crossover_failure_rate_pct': cx_rate,
            'mutation_attempts': self.mutation_attempts,
            'mutation_failures': self.mutation_failures,
            'mutation_failure_rate_pct': mut_rate
        }

    def log_operation_stats(self):
        """Log current genetic operation statistics."""
        stats = self.get_operation_stats()
        if stats['crossover_failures'] > 0 or stats['mutation_failures'] > 0:
            logger.warning(f"Genetic operation stats: "
                          f"crossover={stats['crossover_failures']}/{stats['crossover_attempts']} "
                          f"({stats['crossover_failure_rate_pct']:.1f}% failures), "
                          f"mutation={stats['mutation_failures']}/{stats['mutation_attempts']} "
                          f"({stats['mutation_failure_rate_pct']:.1f}% failures)")
        else:
            logger.info(f"Genetic operation stats: "
                       f"{stats['crossover_attempts']} crossovers, "
                       f"{stats['mutation_attempts']} mutations, 0 failures")

    def evaluate_genome(self, genome: StrategyGenome, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a genome on market data.

        Returns dictionary with:
        - entry_signal: bool (should enter)
        - exit_signal: bool (should exit)
        - position_pct: float (position size 0-1)
        - stop_loss_pct: float (stop loss distance)
        - target_pct: float (target distance)

        Args:
            genome: Genome to evaluate
            data: Market data DataFrame

        Returns:
            Evaluation results dictionary
        """
        results = {
            'entry_signal': False,
            'exit_signal': False,
            'position_pct': 0.1,
            'stop_loss_pct': 0.05,
            'target_pct': 0.10
        }

        try:
            # Evaluate entry condition
            entry_func = gp.compile(genome.entry_tree, self.bool_pset)
            entry_result = entry_func(data)
            results['entry_signal'] = bool(entry_result) if entry_result is not None else False
        except Exception as e:
            logger.debug(f"Entry evaluation failed: {e}")

        try:
            # Evaluate exit condition
            exit_func = gp.compile(genome.exit_tree, self.bool_pset)
            exit_result = exit_func(data)
            results['exit_signal'] = bool(exit_result) if exit_result is not None else False
        except Exception as e:
            logger.debug(f"Exit evaluation failed: {e}")

        try:
            # Evaluate position size
            pos_func = gp.compile(genome.position_tree, self.float_pset)
            pos_result = pos_func(data)
            if pos_result is not None and not pd.isna(pos_result):
                # Clamp to valid range
                results['position_pct'] = max(
                    self.config.min_position_pct,
                    min(self.config.max_position_pct, abs(float(pos_result)))
                )
        except Exception as e:
            logger.debug(f"Position evaluation failed: {e}")

        try:
            # Evaluate stop loss
            stop_func = gp.compile(genome.stop_loss_tree, self.float_pset)
            stop_result = stop_func(data)
            if stop_result is not None and not pd.isna(stop_result):
                results['stop_loss_pct'] = max(
                    self.config.min_stop_loss_pct,
                    min(self.config.max_stop_loss_pct, abs(float(stop_result)))
                )
        except Exception as e:
            logger.debug(f"Stop loss evaluation failed: {e}")

        try:
            # Evaluate target
            target_func = gp.compile(genome.target_tree, self.float_pset)
            target_result = target_func(data)
            if target_result is not None and not pd.isna(target_result):
                results['target_pct'] = max(
                    self.config.min_target_pct,
                    min(self.config.max_target_pct, abs(float(target_result)))
                )
        except Exception as e:
            logger.debug(f"Target evaluation failed: {e}")

        return results

    def serialize_genome(self, genome: StrategyGenome) -> str:
        """Serialize genome to JSON string."""
        return json.dumps(genome.to_dict())

    def _safe_tree_from_string(self, tree_str: str, pset) -> gp.PrimitiveTree:
        """
        Safely parse a tree from string, handling ephemeral constants.

        DEAP's from_string fails on bare float constants. This wrapper
        catches the error and creates a minimal valid tree instead.
        """
        try:
            return gp.PrimitiveTree.from_string(tree_str, pset)
        except Exception as e:
            # If parsing fails (usually due to ephemeral constant type mismatch),
            # create a minimal valid tree
            logger.warning(f"Tree parse failed, creating minimal tree: {e}")
            if pset == self.bool_pset:
                # Return simple 'true' terminal
                return gp.PrimitiveTree([pset.mapping['true']])
            else:
                # Return a simple constant
                return gp.PrimitiveTree([pset.mapping['const_small']])

    def deserialize_genome(self, data: str) -> StrategyGenome:
        """
        Deserialize genome from JSON string.

        Note: This creates new trees from string representation,
        which may not be identical to originals but are functionally equivalent.
        """
        d = json.loads(data)

        # Parse trees from strings with fallback for type mismatches
        entry = self._safe_tree_from_string(d['trees']['entry'], self.bool_pset)
        exit_tree = self._safe_tree_from_string(d['trees']['exit'], self.bool_pset)
        position = self._safe_tree_from_string(d['trees']['position'], self.float_pset)
        stop_loss = self._safe_tree_from_string(d['trees']['stop_loss'], self.float_pset)
        target = self._safe_tree_from_string(d['trees']['target'], self.float_pset)

        genome = StrategyGenome(
            entry_tree=entry,
            exit_tree=exit_tree,
            position_tree=position,
            stop_loss_tree=stop_loss,
            target_tree=target,
            genome_id=d['genome_id'],
            generation=d['generation'],
            parent_ids=d.get('parent_ids', []),
            created_at=datetime.fromisoformat(d['created_at'])
        )

        genome.fitness_values = tuple(d['fitness_values']) if d.get('fitness_values') else None

        return genome


def generate_strategy_code(genome: StrategyGenome, factory: GenomeFactory) -> str:
    """
    Generate Python code for an evolved strategy.

    Args:
        genome: Strategy genome
        factory: GenomeFactory for primitive sets

    Returns:
        Python source code as string
    """
    code = f'''"""
Evolved Strategy: {genome.genome_id}
Generated: {genome.created_at.isoformat()}
Generation: {genome.generation}
Complexity: {genome.total_complexity} nodes
"""

from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType


class EvolvedStrategy_{genome.genome_id}(BaseStrategy):
    """
    Evolved trading strategy discovered through genetic programming.

    Entry condition:
        {str(genome.entry_tree)}

    Exit condition:
        {str(genome.exit_tree)}

    Position sizing:
        {str(genome.position_tree)}

    Stop loss:
        {str(genome.stop_loss_tree)}

    Target:
        {str(genome.target_tree)}
    """

    def __init__(self):
        super().__init__(name="evolved_{genome.genome_id}")
        self.genome_id = "{genome.genome_id}"
        self.generation = {genome.generation}

    def generate_signals(self,
                        data: Dict[str, pd.DataFrame],
                        current_positions: List[str] = None,
                        vix_regime: str = None) -> List[Signal]:
        """Generate trading signals based on evolved rules."""
        signals = []
        current_positions = current_positions or []

        for symbol, df in data.items():
            if len(df) < 50:  # Need enough history
                continue

            try:
                # Check entry condition
                if symbol not in current_positions:
                    entry_signal = self._evaluate_entry(df)
                    if entry_signal:
                        price = float(df['close'].iloc[-1])
                        position_pct = self._evaluate_position(df)
                        stop_pct = self._evaluate_stop_loss(df)
                        target_pct = self._evaluate_target(df)

                        signals.append(Signal(
                            timestamp=df.index[-1] if hasattr(df.index[-1], 'isoformat') else datetime.now(),
                            symbol=symbol,
                            strategy=self.name,
                            signal_type=SignalType.BUY,
                            strength=0.7,
                            price=price,
                            stop_loss=price * (1 - stop_pct),
                            target_price=price * (1 + target_pct),
                            position_size_pct=position_pct,
                            reason="GP evolved entry"
                        ))

                # Check exit condition for existing positions
                elif symbol in current_positions:
                    exit_signal = self._evaluate_exit(df)
                    if exit_signal:
                        signals.append(Signal(
                            timestamp=df.index[-1] if hasattr(df.index[-1], 'isoformat') else datetime.now(),
                            symbol=symbol,
                            strategy=self.name,
                            signal_type=SignalType.CLOSE,
                            strength=1.0,
                            price=float(df['close'].iloc[-1]),
                            reason="GP evolved exit"
                        ))

            except Exception as e:
                continue  # Skip symbols with evaluation errors

        return signals

    def _evaluate_entry(self, df: pd.DataFrame) -> bool:
        """Evaluate entry condition tree."""
        # Entry tree: {str(genome.entry_tree)}
        try:
            # Implement the entry tree logic here
            # This is a placeholder - actual implementation would compile the tree
            return False
        except Exception as e:
            logger.debug(f"Entry tree evaluation failed: {e}")
            return False

    def _evaluate_exit(self, df: pd.DataFrame) -> bool:
        """Evaluate exit condition tree."""
        # Exit tree: {str(genome.exit_tree)}
        try:
            return False
        except Exception as e:
            logger.debug(f"Exit tree evaluation failed: {e}")
            return False

    def _evaluate_position(self, df: pd.DataFrame) -> float:
        """Evaluate position sizing tree."""
        # Position tree: {str(genome.position_tree)}
        try:
            return 0.10  # Default 10%
        except Exception as e:
            logger.debug(f"Position sizing tree evaluation failed: {e}")
            return 0.10

    def _evaluate_stop_loss(self, df: pd.DataFrame) -> float:
        """Evaluate stop loss tree."""
        # Stop loss tree: {str(genome.stop_loss_tree)}
        try:
            return 0.05  # Default 5%
        except Exception as e:
            logger.debug(f"Stop loss tree evaluation failed: {e}")
            return 0.05

    def _evaluate_target(self, df: pd.DataFrame) -> float:
        """Evaluate target tree."""
        # Target tree: {str(genome.target_tree)}
        try:
            return 0.10  # Default 10%
        except Exception as e:
            logger.debug(f"Target tree evaluation failed: {e}")
            return 0.10


# Instantiate strategy for use
strategy = EvolvedStrategy_{genome.genome_id}()
'''
    return code


if __name__ == "__main__":
    # Test genome creation and manipulation
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Creating genome factory...")
    factory = GenomeFactory()

    print("\nCreating random genomes...")
    genomes = [factory.create_random_genome(generation=0) for _ in range(5)]

    for i, g in enumerate(genomes):
        print(f"  {i}: {g}")
        print(f"      Entry: {str(g.entry_tree)[:60]}...")

    print("\nTesting crossover...")
    child1, child2 = factory.crossover(genomes[0], genomes[1], generation=1)
    print(f"  Child 1: {child1}")
    print(f"  Child 2: {child2}")

    print("\nTesting mutation...")
    mutant = factory.mutate(genomes[0], generation=1)
    print(f"  Original: {genomes[0]}")
    print(f"  Mutant: {mutant}")

    print("\nTesting serialization...")
    serialized = factory.serialize_genome(genomes[0])
    print(f"  Serialized length: {len(serialized)} chars")

    print("\nTesting evaluation...")
    import numpy as np
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)

    result = factory.evaluate_genome(genomes[0], test_data)
    print(f"  Entry: {result['entry_signal']}")
    print(f"  Exit: {result['exit_signal']}")
    print(f"  Position: {result['position_pct']:.2%}")
    print(f"  Stop: {result['stop_loss_pct']:.2%}")
    print(f"  Target: {result['target_pct']:.2%}")

    print("\nTest complete!")
