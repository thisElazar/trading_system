"""
Genetic Programming Core
========================
DEAP-based genetic programming primitives for trading strategy evolution.

Uses Strongly-Typed GP to enforce type constraints:
- Boolean expressions for entry/exit conditions
- Float expressions for position sizing, stops, targets

Design: Indicator functions are 0-arity terminals that access market data
from a thread-local context. This avoids the need for DataFrame as an
intermediate type in the GP tree, which DEAP cannot handle during generation.
"""

import operator
import math
import random
import logging
import threading
from typing import Callable, Dict, Any, List, Optional
from functools import partial

import numpy as np
import pandas as pd

# DEAP imports
from deap import base, creator, gp, tools

from .config import PrimitiveConfig

logger = logging.getLogger(__name__)


# Thread-local storage for market data during evaluation
_eval_context = threading.local()


def set_eval_data(data: pd.DataFrame):
    """Set the market data for tree evaluation (thread-safe)."""
    _eval_context.data = data


def get_eval_data() -> Optional[pd.DataFrame]:
    """Get the market data for tree evaluation (thread-safe)."""
    return getattr(_eval_context, 'data', None)


def clear_eval_data():
    """Clear the evaluation data context."""
    if hasattr(_eval_context, 'data'):
        del _eval_context.data


# Type definitions for Strongly-Typed GP
class BoolType:
    """Boolean type for entry/exit conditions."""
    pass


# FloatType is an alias for float to ensure DEAP can parse serialized trees.
# Previously this was a separate class, but DEAP parses numeric literals as
# Python floats, causing type mismatch errors when loading checkpoints.
FloatType = float


# Protected division to avoid division by zero
def protected_div(a: float, b: float) -> float:
    """Division that returns 1.0 when dividing by near-zero."""
    if abs(b) < 1e-6:
        return 1.0
    return a / b


def protected_log(x: float) -> float:
    """Log that handles non-positive values."""
    if x <= 0:
        return 0.0
    return math.log(x)


def protected_sqrt(x: float) -> float:
    """Square root that handles negative values."""
    if x < 0:
        return 0.0
    return math.sqrt(x)


def if_then_else(cond: bool, a: float, b: float) -> float:
    """Conditional expression."""
    return a if cond else b


# ==========================================================================
# Technical indicator functions (context-based, 0-arity for GP terminals)
# ==========================================================================
# These functions access market data from thread-local context rather than
# taking DataFrame as a parameter. This allows them to be registered as
# terminals in the GP primitive set.

def _safe_get_column(df, col: str, default: float = 0.0) -> float:
    """Safely get the latest value from a DataFrame column."""
    if df is None or len(df) == 0:
        return default
    if col not in df.columns:
        return default
    val = df[col].iloc[-1]
    return float(val) if not np.isnan(val) else default


def ind_close() -> float:
    """Get latest close price."""
    return _safe_get_column(get_eval_data(), 'close', 0.0)


def ind_open() -> float:
    """Get latest open price."""
    return _safe_get_column(get_eval_data(), 'open', 0.0)


def ind_high() -> float:
    """Get latest high price."""
    return _safe_get_column(get_eval_data(), 'high', 0.0)


def ind_low() -> float:
    """Get latest low price."""
    return _safe_get_column(get_eval_data(), 'low', 0.0)


def ind_volume() -> float:
    """Get latest volume."""
    return _safe_get_column(get_eval_data(), 'volume', 0.0)


def _make_sma(period: int):
    """Factory for SMA indicator function."""
    def _sma() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_close()
        val = df['close'].rolling(period).mean().iloc[-1]
        return float(val) if not np.isnan(val) else ind_close()
    _sma.__name__ = f'sma_{period}'
    return _sma


def _make_ema(period: int):
    """Factory for EMA indicator function."""
    def _ema() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_close()
        val = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
        return float(val) if not np.isnan(val) else ind_close()
    _ema.__name__ = f'ema_{period}'
    return _ema


def _make_rsi(period: int = 14):
    """Factory for RSI indicator function."""
    def _rsi() -> float:
        df = get_eval_data()
        if df is None or len(df) < period + 1:
            return 50.0
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi) if not np.isnan(rsi) else 50.0
    _rsi.__name__ = f'rsi_{period}'
    return _rsi


def _make_atr(period: int = 14):
    """Factory for ATR indicator function."""
    def _atr() -> float:
        df = get_eval_data()
        if df is None or len(df) < period + 1:
            return 0.0
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0
    _atr.__name__ = f'atr_{period}'
    return _atr


def _make_bbands_upper(period: int = 20, std_dev: float = 2.0):
    """Factory for Bollinger Bands upper band."""
    def _bb_upper() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_close()
        sma = df['close'].rolling(period).mean().iloc[-1]
        std = df['close'].rolling(period).std().iloc[-1]
        return float(sma + std_dev * std) if not np.isnan(std) else ind_close()
    _bb_upper.__name__ = f'bb_upper_{period}'
    return _bb_upper


def _make_bbands_lower(period: int = 20, std_dev: float = 2.0):
    """Factory for Bollinger Bands lower band."""
    def _bb_lower() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_close()
        sma = df['close'].rolling(period).mean().iloc[-1]
        std = df['close'].rolling(period).std().iloc[-1]
        return float(sma - std_dev * std) if not np.isnan(std) else ind_close()
    _bb_lower.__name__ = f'bb_lower_{period}'
    return _bb_lower


def _make_macd(fast: int = 12, slow: int = 26):
    """Factory for MACD line."""
    def _macd() -> float:
        df = get_eval_data()
        if df is None or len(df) < slow:
            return 0.0
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast.iloc[-1] - ema_slow.iloc[-1]
        return float(macd) if not np.isnan(macd) else 0.0
    _macd.__name__ = f'macd_{fast}_{slow}'
    return _macd


def _make_macd_signal(fast: int = 12, slow: int = 26, signal: int = 9):
    """Factory for MACD signal line."""
    def _macd_signal() -> float:
        df = get_eval_data()
        if df is None or len(df) < slow + signal:
            return 0.0
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean().iloc[-1]
        return float(macd_signal) if not np.isnan(macd_signal) else 0.0
    _macd_signal.__name__ = f'macd_signal_{fast}_{slow}_{signal}'
    return _macd_signal


def _make_high_n(period: int):
    """Factory for highest high in N periods."""
    def _high_n() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_high()
        val = df['high'].tail(period).max()
        return float(val) if not np.isnan(val) else ind_high()
    _high_n.__name__ = f'high_{period}d'
    return _high_n


def _make_low_n(period: int):
    """Factory for lowest low in N periods."""
    def _low_n() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_low()
        val = df['low'].tail(period).min()
        return float(val) if not np.isnan(val) else ind_low()
    _low_n.__name__ = f'low_{period}d'
    return _low_n


def _make_volume_sma(period: int = 20):
    """Factory for volume SMA."""
    def _volume_sma() -> float:
        df = get_eval_data()
        if df is None or len(df) < period:
            return ind_volume()
        val = df['volume'].rolling(period).mean().iloc[-1]
        return float(val) if not np.isnan(val) else ind_volume()
    _volume_sma.__name__ = f'volume_sma_{period}'
    return _volume_sma


def _make_returns(period: int = 1):
    """Factory for returns over N periods."""
    def _returns() -> float:
        df = get_eval_data()
        if df is None or len(df) <= period:
            return 0.0
        prev_close = df['close'].iloc[-period - 1]
        if prev_close == 0 or np.isnan(prev_close):
            return 0.0
        val = (df['close'].iloc[-1] / prev_close) - 1
        return float(val) if not np.isnan(val) else 0.0
    _returns.__name__ = f'returns_{period}d'
    return _returns


def _make_volatility(period: int = 20):
    """Factory for rolling volatility."""
    def _volatility() -> float:
        df = get_eval_data()
        if df is None or len(df) < period + 1:
            return 0.0
        returns = df['close'].pct_change()
        vol = returns.rolling(period).std().iloc[-1] * np.sqrt(252)
        return float(vol) if not np.isnan(vol) else 0.0
    _volatility.__name__ = f'volatility_{period}'
    return _volatility


def create_primitive_set(config: PrimitiveConfig = None) -> gp.PrimitiveSetTyped:
    """
    Create the primitive set for GP (float-valued trees).

    Returns a strongly-typed primitive set with:
    - FloatType for numeric values
    - BoolType for conditions
    - No input arguments (indicators access data via thread-local context)
    """
    config = config or PrimitiveConfig()

    # Create primitive set with NO input arguments, FloatType output
    # Indicators access data from thread-local context during evaluation
    pset = gp.PrimitiveSetTyped("MAIN", [], FloatType)

    # ==========================================================================
    # ARITHMETIC FUNCTIONS (FloatType, FloatType) -> FloatType
    # ==========================================================================
    pset.addPrimitive(operator.add, [FloatType, FloatType], FloatType, name='add')
    pset.addPrimitive(operator.sub, [FloatType, FloatType], FloatType, name='sub')
    pset.addPrimitive(operator.mul, [FloatType, FloatType], FloatType, name='mul')
    pset.addPrimitive(protected_div, [FloatType, FloatType], FloatType, name='div')

    # ==========================================================================
    # UNARY FUNCTIONS (FloatType) -> FloatType
    # ==========================================================================
    pset.addPrimitive(operator.neg, [FloatType], FloatType, name='neg')
    pset.addPrimitive(abs, [FloatType], FloatType, name='abs')
    pset.addPrimitive(protected_log, [FloatType], FloatType, name='log')
    pset.addPrimitive(protected_sqrt, [FloatType], FloatType, name='sqrt')

    # ==========================================================================
    # COMPARISON FUNCTIONS (FloatType, FloatType) -> BoolType
    # ==========================================================================
    pset.addPrimitive(operator.gt, [FloatType, FloatType], BoolType, name='gt')
    pset.addPrimitive(operator.lt, [FloatType, FloatType], BoolType, name='lt')
    pset.addPrimitive(operator.ge, [FloatType, FloatType], BoolType, name='ge')
    pset.addPrimitive(operator.le, [FloatType, FloatType], BoolType, name='le')

    # ==========================================================================
    # LOGICAL FUNCTIONS
    # ==========================================================================
    pset.addPrimitive(operator.and_, [BoolType, BoolType], BoolType, name='and_')
    pset.addPrimitive(operator.or_, [BoolType, BoolType], BoolType, name='or_')
    pset.addPrimitive(operator.not_, [BoolType], BoolType, name='not_')

    # ==========================================================================
    # CONDITIONAL (BoolType, FloatType, FloatType) -> FloatType
    # ==========================================================================
    pset.addPrimitive(if_then_else, [BoolType, FloatType, FloatType], FloatType, name='if_then_else')

    # ==========================================================================
    # PRICE DATA TERMINALS (0-arity functions returning FloatType)
    # ==========================================================================
    pset.addPrimitive(ind_close, [], FloatType, name='close')
    pset.addPrimitive(ind_open, [], FloatType, name='open')
    pset.addPrimitive(ind_high, [], FloatType, name='high')
    pset.addPrimitive(ind_low, [], FloatType, name='low')
    pset.addPrimitive(ind_volume, [], FloatType, name='volume')

    # ==========================================================================
    # TECHNICAL INDICATORS (0-arity functions returning FloatType)
    # ==========================================================================

    # SMA with various periods
    for period in config.sma_periods:
        pset.addPrimitive(_make_sma(period), [], FloatType, name=f'sma_{period}')

    # EMA with various periods
    for period in config.ema_periods:
        pset.addPrimitive(_make_ema(period), [], FloatType, name=f'ema_{period}')

    # RSI
    pset.addPrimitive(_make_rsi(config.rsi_period), [], FloatType, name='rsi')

    # ATR
    pset.addPrimitive(_make_atr(config.atr_period), [], FloatType, name='atr')

    # Bollinger Bands
    pset.addPrimitive(_make_bbands_upper(config.bb_period), [], FloatType, name='bb_upper')
    pset.addPrimitive(_make_bbands_lower(config.bb_period), [], FloatType, name='bb_lower')

    # MACD
    pset.addPrimitive(_make_macd(config.macd_fast, config.macd_slow), [], FloatType, name='macd')
    pset.addPrimitive(
        _make_macd_signal(config.macd_fast, config.macd_slow, config.macd_signal),
        [], FloatType, name='macd_signal'
    )

    # Historical lookbacks
    for period in config.lookback_periods:
        pset.addPrimitive(_make_high_n(period), [], FloatType, name=f'high_{period}d')
        pset.addPrimitive(_make_low_n(period), [], FloatType, name=f'low_{period}d')

    # Volume SMA
    pset.addPrimitive(_make_volume_sma(20), [], FloatType, name='volume_sma')

    # Returns and Volatility
    for period in [1, 5, 20]:
        pset.addPrimitive(_make_returns(period), [], FloatType, name=f'returns_{period}d')

    pset.addPrimitive(_make_volatility(20), [], FloatType, name='volatility')

    # ==========================================================================
    # EPHEMERAL RANDOM CONSTANTS
    # ==========================================================================

    # Use proper functions instead of lambdas to avoid pickle issues
    def gen_small_const():
        return random.uniform(0.001, 0.05)

    def gen_medium_const():
        return random.uniform(0.05, 0.20)

    # Small constants (for stop losses, position sizes, etc.)
    pset.addEphemeralConstant("const_small", gen_small_const, FloatType)

    # Medium constants (for targets, thresholds, etc.)
    pset.addEphemeralConstant("const_medium", gen_medium_const, FloatType)

    # Boolean constants (rare)
    pset.addTerminal(True, BoolType, name='true')
    pset.addTerminal(False, BoolType, name='false')

    # Fixed numeric constants
    for val in [0.0, 1.0, 30.0, 50.0, 70.0, 100.0]:  # Common thresholds (RSI levels, etc.)
        pset.addTerminal(val, FloatType, name=f'const_{int(val)}')

    return pset


def create_boolean_primitive_set(config: PrimitiveConfig = None) -> gp.PrimitiveSetTyped:
    """
    Create a primitive set specifically for boolean expressions (entry/exit conditions).

    Output type is BoolType instead of FloatType.
    This primitive set includes all the same functions as the float set,
    but the tree must ultimately return a boolean value.
    No input arguments - indicators access data via thread-local context.
    """
    config = config or PrimitiveConfig()

    # Boolean output for entry/exit conditions, no input arguments
    pset = gp.PrimitiveSetTyped("BOOL", [], BoolType)

    # ==========================================================================
    # COMPARISON FUNCTIONS (FloatType, FloatType) -> BoolType
    # These are the key functions that produce boolean output
    # ==========================================================================
    pset.addPrimitive(operator.gt, [FloatType, FloatType], BoolType, name='gt')
    pset.addPrimitive(operator.lt, [FloatType, FloatType], BoolType, name='lt')
    pset.addPrimitive(operator.ge, [FloatType, FloatType], BoolType, name='ge')
    pset.addPrimitive(operator.le, [FloatType, FloatType], BoolType, name='le')

    # ==========================================================================
    # LOGICAL FUNCTIONS (BoolType, BoolType) -> BoolType
    # ==========================================================================
    pset.addPrimitive(operator.and_, [BoolType, BoolType], BoolType, name='and_')
    pset.addPrimitive(operator.or_, [BoolType, BoolType], BoolType, name='or_')
    pset.addPrimitive(operator.not_, [BoolType], BoolType, name='not_')

    # ==========================================================================
    # ARITHMETIC FUNCTIONS (FloatType, FloatType) -> FloatType
    # ==========================================================================
    pset.addPrimitive(operator.add, [FloatType, FloatType], FloatType, name='add')
    pset.addPrimitive(operator.sub, [FloatType, FloatType], FloatType, name='sub')
    pset.addPrimitive(operator.mul, [FloatType, FloatType], FloatType, name='mul')
    pset.addPrimitive(protected_div, [FloatType, FloatType], FloatType, name='div')

    # ==========================================================================
    # UNARY FUNCTIONS (FloatType) -> FloatType
    # ==========================================================================
    pset.addPrimitive(operator.neg, [FloatType], FloatType, name='neg')
    pset.addPrimitive(abs, [FloatType], FloatType, name='abs')

    # ==========================================================================
    # PRICE DATA TERMINALS (0-arity functions returning FloatType)
    # ==========================================================================
    pset.addPrimitive(ind_close, [], FloatType, name='close')
    pset.addPrimitive(ind_open, [], FloatType, name='open')
    pset.addPrimitive(ind_high, [], FloatType, name='high')
    pset.addPrimitive(ind_low, [], FloatType, name='low')
    pset.addPrimitive(ind_volume, [], FloatType, name='volume')

    # ==========================================================================
    # TECHNICAL INDICATORS (0-arity functions returning FloatType)
    # ==========================================================================

    # SMA with various periods
    for period in config.sma_periods:
        pset.addPrimitive(_make_sma(period), [], FloatType, name=f'sma_{period}')

    # EMA with various periods
    for period in config.ema_periods:
        pset.addPrimitive(_make_ema(period), [], FloatType, name=f'ema_{period}')

    # RSI
    pset.addPrimitive(_make_rsi(config.rsi_period), [], FloatType, name='rsi')

    # ATR
    pset.addPrimitive(_make_atr(config.atr_period), [], FloatType, name='atr')

    # Bollinger Bands
    pset.addPrimitive(_make_bbands_upper(config.bb_period), [], FloatType, name='bb_upper')
    pset.addPrimitive(_make_bbands_lower(config.bb_period), [], FloatType, name='bb_lower')

    # MACD
    pset.addPrimitive(_make_macd(config.macd_fast, config.macd_slow), [], FloatType, name='macd')

    # Historical lookbacks
    for period in config.lookback_periods:
        pset.addPrimitive(_make_high_n(period), [], FloatType, name=f'high_{period}d')
        pset.addPrimitive(_make_low_n(period), [], FloatType, name=f'low_{period}d')

    # Returns
    for period in [1, 5, 20]:
        pset.addPrimitive(_make_returns(period), [], FloatType, name=f'returns_{period}d')

    # ==========================================================================
    # CONSTANTS
    # ==========================================================================

    # Use proper functions instead of lambdas to avoid pickle issues
    def gen_small_const():
        return random.uniform(0.001, 0.05)

    def gen_medium_const():
        return random.uniform(0.05, 0.20)

    pset.addEphemeralConstant("const_small", gen_small_const, FloatType)
    pset.addEphemeralConstant("const_medium", gen_medium_const, FloatType)

    # Boolean constants
    pset.addTerminal(True, BoolType, name='true')
    pset.addTerminal(False, BoolType, name='false')

    # Fixed numeric constants (RSI levels, etc.)
    for val in [0.0, 1.0, 30.0, 50.0, 70.0, 100.0]:
        pset.addTerminal(val, FloatType, name=f'const_{int(val)}')

    return pset


def setup_deap_types():
    """
    Setup DEAP creator types for multi-objective optimization.

    Creates:
    - FitnessMulti: Multi-objective fitness (Sortino, -Drawdown, -CVaR, Novelty)
    - Individual: GP tree with fitness
    """
    # Check if already created (DEAP complains about recreating)
    if not hasattr(creator, 'FitnessMulti'):
        # Weights: positive = maximize, negative = minimize
        # Sortino: maximize (positive)
        # Max Drawdown: minimize (we store as negative, so maximize makes it less negative)
        # CVaR: minimize (negative weight)
        # Novelty: maximize (positive)
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0, 1.0))

    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


def create_toolbox(
    pset: gp.PrimitiveSetTyped,
    config: 'EvolutionConfig' = None
) -> base.Toolbox:
    """
    Create DEAP toolbox with genetic operators.

    Args:
        pset: Primitive set for tree generation
        config: Evolution configuration

    Returns:
        Configured DEAP toolbox
    """
    from .config import EvolutionConfig
    config = config or EvolutionConfig()

    setup_deap_types()
    toolbox = base.Toolbox()

    # Tree generation
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=config.min_tree_depth, max_=config.max_tree_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Compile function
    toolbox.register("compile", gp.compile, pset=pset)

    # Selection (NSGA-II)
    toolbox.register("select", tools.selNSGA2)

    # Crossover
    toolbox.register("mate", gp.cxOnePoint)

    # Mutation operators
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    # Subtree mutation (main mutation type)
    toolbox.register("mutate_subtree", gp.mutUniform,
                     expr=toolbox.expr_mut, pset=pset)

    # Point mutation (change single node)
    toolbox.register("mutate_point", gp.mutNodeReplacement, pset=pset)

    # Shrink mutation (simplify tree)
    toolbox.register("mutate_shrink", gp.mutShrink)

    # Hoist mutation (promote subtree)
    def hoist_mutation(individual):
        """Hoist mutation: randomly select a subtree and make it the root."""
        if len(individual) < 3:
            return individual,

        # Find a random subtree
        index = random.randrange(1, len(individual))
        slice_ = individual.searchSubtree(index)
        subtree = individual[slice_]

        # Replace individual with subtree
        individual[:] = subtree
        return individual,

    toolbox.register("mutate_hoist", hoist_mutation)

    # Combined mutation (weighted selection)
    def weighted_mutation(individual):
        """Apply one mutation type based on configured probabilities."""
        r = random.random()

        if r < config.subtree_mutation_prob:
            return toolbox.mutate_subtree(individual)
        elif r < config.subtree_mutation_prob + config.point_mutation_prob:
            return toolbox.mutate_point(individual)
        elif r < config.subtree_mutation_prob + config.point_mutation_prob + config.hoist_mutation_prob:
            return toolbox.mutate_hoist(individual)
        else:
            return toolbox.mutate_shrink(individual)

    toolbox.register("mutate", weighted_mutation)

    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_tree_depth))

    return toolbox


def evaluate_tree(tree: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped,
                  data: pd.DataFrame) -> Any:
    """
    Evaluate a GP tree on market data.

    Args:
        tree: GP tree to evaluate
        pset: Primitive set used to create the tree
        data: Market data DataFrame (set in thread-local context)

    Returns:
        Evaluation result (float or bool depending on tree type)
    """
    try:
        # Set data in thread-local context for indicator functions
        set_eval_data(data)
        func = gp.compile(tree, pset)
        # For trees with 0-arity primitives, DEAP may return the result directly
        # instead of a callable when the tree is very simple
        if callable(func):
            return func()
        else:
            return func
    except Exception as e:
        logger.warning(f"Tree evaluation failed: {e} | tree={str(tree)[:100]}")
        return None
    finally:
        clear_eval_data()


def tree_to_string(tree: gp.PrimitiveTree) -> str:
    """Convert GP tree to human-readable string."""
    return str(tree)


def tree_complexity(tree: gp.PrimitiveTree) -> int:
    """Calculate tree complexity (number of nodes)."""
    return len(tree)


if __name__ == "__main__":
    # Test primitive set creation
    logging.basicConfig(level=logging.INFO)

    print("Creating primitive set...")
    pset = create_primitive_set()
    print(f"  Primitives: {len(pset.primitives)}")
    print(f"  Terminals: {len(pset.terminals)}")

    print("\nSetting up DEAP types...")
    setup_deap_types()

    print("\nCreating toolbox...")
    toolbox = create_toolbox(pset)

    print("\nGenerating sample individual...")
    ind = toolbox.individual()
    print(f"  Tree: {ind}")
    print(f"  Height: {ind.height}")
    print(f"  Size: {len(ind)}")

    # Test evaluation with dummy data
    print("\nTesting evaluation...")
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)

    result = evaluate_tree(ind, pset, test_data)
    print(f"  Result: {result}")

    print("\nTest complete!")
