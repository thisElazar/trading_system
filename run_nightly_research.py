#!/usr/bin/env python3
"""
Nightly Research Runner
========================
Autonomous overnight research engine that evolves trading strategies.

This script:
1. Runs after market close
2. Evolves each enabled strategy for 1-5 generations (parameter optimization)
3. Optionally runs GP-based strategy discovery for novel strategies
4. Optionally runs adaptive GA with regime-matched testing (NEW)
5. Persists state to database
6. Logs improvements for analysis
7. Handles restarts gracefully (idempotent)

Usage:
    # Run once (for cron/systemd)
    python run_nightly_research.py

    # Run in continuous loop (for development)
    python run_nightly_research.py --loop

    # Query improvement history
    python run_nightly_research.py --status

    # Run with strategy discovery enabled
    python run_nightly_research.py --discovery

    # Run only strategy discovery (skip parameter optimization)
    python run_nightly_research.py --discovery-only --discovery-hours 4

    # Run with adaptive GA (regime-matched testing)
    python run_nightly_research.py --adaptive

    # Run adaptive GA with rapid testing on short periods first
    python run_nightly_research.py --adaptive --rapid-first

Design principles:
- Incremental: Few generations per night, accumulates over weeks
- Resilient: Crash recovery via database persistence
- Observable: Full logging and queryable history
- Overfitting-aware: Walk-forward validation only
- Discovery: GP-based novel strategy discovery (optional)
- Adaptive: Regime-matched multi-scale testing (optional)
"""

import argparse
import gc
import logging
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Global shutdown flag for graceful termination
_shutdown_requested = False

def _handle_shutdown_signal(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logging.getLogger('nightly_research').warning(f"Received signal {signum}, requesting shutdown...")

# Register signal handlers at module load
signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)

# ============================================================================
# RESEARCH TIME BOUNDARIES
# ============================================================================
# Research must self-enforce its allowed hours. The orchestrator may not be
# responsive enough to kill us if we're consuming too many resources.
#
# Time boundaries are defined in utils/timezone.py:
#   - Weekdays: 7:30 AM - 5:00 PM ET blocked (market hours + buffer)
#   - Sunday: After 7:30 PM ET blocked (before Monday pre-market)
#   - Saturday: Always allowed

from utils.timezone import is_research_allowed as _is_research_allowed

def _should_stop_research() -> bool:
    """
    Check if research should stop NOW.

    Returns True if:
    - Shutdown signal received (SIGTERM/SIGINT)
    - Outside allowed research hours
    """
    if _shutdown_requested:
        return True

    if not _is_research_allowed():
        logging.getLogger('nightly_research').warning(
            "Research time boundary reached - stopping to preserve system for trading"
        )
        return True

    return False

# ============================================================================
# PARALLEL FITNESS EVALUATION INFRASTRUCTURE
# ============================================================================
# Module-level globals for parallel fitness evaluation
# These get copied to worker processes via fork() on Linux

_parallel_fitness_context = {
    'strategy_name': None,
    'train_data': None,
    'test_data': None,
    'train_vix': None,
    'test_vix': None,
    'full_data': None,
    'full_vix': None,
    'wf_enabled': False,
    'degradation_threshold': 0.2,
    'backtester': None,
}

def _init_parallel_worker():
    """Initialize worker process (called once per worker)."""
    # Workers inherit the global context via fork()
    # CRITICAL: Workers must ignore signals - only main process handles shutdown
    # This prevents zombie workers when SIGTERM is sent to the process group
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def evaluate_genes_parallel(genes: dict) -> float:
    """
    Module-level fitness function for parallel evaluation.
    Accesses shared data via module globals (copied via fork).
    """
    ctx = _parallel_fitness_context
    
    if ctx['strategy_name'] is None:
        return 0.0
    
    try:
        # Import here to avoid circular imports in workers
        from research.backtester import Backtester
        
        # Create strategy with evolved parameters
        strategy = create_strategy_instance(ctx['strategy_name'], genes)
        
        # Create a fresh backtester for this evaluation (thread-safe)
        backtester = Backtester()
        
        if ctx['wf_enabled']:
            # Walk-forward evaluation
            train_result = backtester.run(
                strategy=strategy,
                data=ctx['train_data'],
                vix_data=ctx['train_vix']
            )
            
            if train_result.total_trades < 5:
                return 0.0
            
            train_fitness = calculate_composite_fitness(train_result, verbose=False)
            if train_fitness <= 0:
                return 0.0
            
            # Test on out-of-sample data
            test_result = backtester.run(
                strategy=strategy,
                data=ctx['test_data'],
                vix_data=ctx['test_vix']
            )
            
            if test_result.total_trades < 3:
                return train_fitness * 0.3
            
            test_fitness = calculate_composite_fitness(test_result, verbose=False)
            
            # Apply degradation penalty
            if train_fitness > 0:
                degradation = (train_fitness - test_fitness) / train_fitness
                if degradation > ctx['degradation_threshold']:
                    penalty = 1.0 - min((degradation - ctx['degradation_threshold']) * 2, 0.5)
                    test_fitness *= penalty
            
            # Apply constraints
            constraint_mult, _ = apply_fitness_constraints(test_result, ctx['strategy_name'])
            
            return float(test_fitness * constraint_mult)
        else:
            # Full dataset evaluation
            result = backtester.run(
                strategy=strategy,
                data=ctx['full_data'],
                vix_data=ctx['full_vix']
            )
            
            constraint_mult, _ = apply_fitness_constraints(result, ctx['strategy_name'])
            if constraint_mult == 0:
                return 0.0
            
            base_fitness = calculate_composite_fitness(result, verbose=False)
            if base_fitness == float('inf') or base_fitness == float('-inf'):
                return 0.0
            
            return float(base_fitness * constraint_mult)

    except Exception as e:
        # Log the error for debugging, but return 0.0 to not crash worker
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Fitness evaluation failed: {e}")
        return 0.0

def setup_parallel_fitness_context(strategy_name: str, backtester, data: dict,
                                   vix_data=None, train_data=None, test_data=None,
                                   train_vix=None, test_vix=None, wf_enabled=False,
                                   degradation_threshold=0.2):
    """Set up the parallel fitness context before running parallel evaluation."""
    global _parallel_fitness_context
    _parallel_fitness_context = {
        'strategy_name': strategy_name,
        'train_data': train_data,
        'test_data': test_data,
        'train_vix': train_vix,
        'test_vix': test_vix,
        'full_data': data,
        'full_vix': vix_data,
        'wf_enabled': wf_enabled,
        'degradation_threshold': degradation_threshold,
        'backtester': None,  # Don't store - create fresh in workers
    }


def clear_parallel_fitness_context():
    """
    Clear the parallel fitness context to free memory.

    Call this after each strategy evolution to allow gc to reclaim
    the train/test data copies. This prevents memory accumulation
    across multiple strategy evolutions.

    Memory savings: ~200-400MB per strategy (train/test data copies).
    """
    global _parallel_fitness_context
    _parallel_fitness_context = {
        'strategy_name': None,
        'train_data': None,
        'test_data': None,
        'train_vix': None,
        'test_vix': None,
        'full_data': None,
        'full_vix': None,
        'wf_enabled': False,
        'degradation_threshold': 0.2,
        'backtester': None,
    }
    gc.collect()


import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import STRATEGIES, DIRS, get_enabled_strategies, PERF, PERF_PROFILE_NAME
from data.cached_data_manager import CachedDataManager
from data.storage.db_manager import get_db
from research.backtester import Backtester
from research.genetic.persistent_optimizer import PersistentGAOptimizer
from research.genetic.optimizer import GeneticConfig, ParameterSpec, STRATEGY_PARAMS
from research.genetic.fitness_utils import calculate_composite_fitness

# Adaptive GA imports (lazy loaded for faster startup)
ADAPTIVE_GA_AVAILABLE = True
try:
    from research.genetic.adaptive_optimizer import AdaptiveGAOptimizer, AdaptiveGAConfig
    from research.genetic.market_periods import MarketPeriodLibrary
    from research.genetic.rapid_backtester import RapidBacktester
    from research.genetic.regime_matching import RegimeMatchingEngine
    from research.genetic.multiscale_fitness import MultiScaleFitnessCalculator
    from research.genetic.adaptive_strategy_manager import AdaptiveStrategyManager
except ImportError as e:
    ADAPTIVE_GA_AVAILABLE = False
    logger_init_msg = f"Adaptive GA not available: {e}"

# Intervention system (optional human oversight)
INTERVENTION_AVAILABLE = True
try:
    from orchestration.intervention import (
        InterventionManager,
        InterventionConfig,
        InterventionMode,
        CheckpointPriority,
    )
except ImportError as e:
    INTERVENTION_AVAILABLE = False
    InterventionManager = None
    InterventionMode = None

# Hardware LED integration (optional) - uses client to request LED states
# The main orchestrator process is the LED authority; we just send requests
LED_AVAILABLE = True
try:
    from hardware.led_authority import LEDClient
except ImportError:
    LED_AVAILABLE = False
    LEDClient = None

# Setup logging
LOG_DIR = DIRS.get('logs', Path('./logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Import database error handler for dashboard visibility
from observability.logger import DatabaseErrorHandler
from logging.handlers import RotatingFileHandler
from utils.timezone import normalize_dataframe, normalize_timestamp, normalize_index

# Log rotation: 10MB per file, keep 5 backups
LOG_MAX_BYTES = 10_000_000
LOG_BACKUP_COUNT = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        RotatingFileHandler(
            LOG_DIR / 'nightly_research.log',
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('nightly_research')

# Add database handler for errors/warnings (enables dashboard display)
db_error_handler = DatabaseErrorHandler(min_level=logging.WARNING)
db_error_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'))
logger.addHandler(db_error_handler)

# Log performance profile on module load
logger.info(f'Performance profile: {PERF_PROFILE_NAME} (max_symbols={PERF["max_symbols"]}, max_years={PERF["max_years"]}, workers={PERF["n_workers"]})')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategies to evolve (must have parameter specs defined)
EVOLVABLE_STRATEGIES = [
    # Tier 1: Core Alpha Generators
    'vol_managed_momentum',
    'quality_smallcap_value',
    'factor_momentum',
    # 'pairs_trading',  # DISABLED: Requires pair symbols in backtest data - fix data pipeline first
    'relative_volume_breakout',
    # Tier 2: Regime & Tactical
    'vix_regime_rotation',
    'sector_rotation',
    'mean_reversion',
    # 'gap_fill',  # Requires intraday data
]

# GA configuration for nightly runs (~6-8 hours for all strategies)
# At ~10 min per evaluation: pop=10 × gen=2 × 7 strategies = 140 evals = ~24 hours
# For faster runs, use fewer strategies or --quick mode
# GA configs now driven by PERF (from config.py)
NIGHTLY_GA_CONFIG = GeneticConfig(
    population_size=PERF['population_size'] * 2,  # Nightly gets 2x population
    generations=PERF['generations'],
    mutation_rate=0.15,
    crossover_rate=0.7,
    elitism=2,
    tournament_size=3,
    early_stop_generations=PERF['early_stop_generations'],
    parallel=PERF['parallel_enabled'],
    n_workers=PERF['n_workers'],
)

# Quick test uses PERF baseline directly
QUICK_GA_CONFIG = GeneticConfig(
    population_size=PERF['population_size'],
    generations=max(1, PERF['generations'] // 2),
    mutation_rate=0.15,
    crossover_rate=0.7,
    elitism=1,
    tournament_size=2,
    early_stop_generations=1,
    parallel=PERF['parallel_enabled'],
    n_workers=PERF['n_workers'],
)

# ============================================================================
# STRATEGY DISCOVERY CONFIGURATION
# ============================================================================

# Enable GP-based strategy discovery by default
ENABLE_STRATEGY_DISCOVERY = True  # Enabled for nightly runs

# Discovery configuration
# Memory-safe settings: population reduced from 50 to 30, with batched evaluation
# Batches of 10 individuals with GC between batches (see evolution_engine.py)
DISCOVERY_CONFIG = {
    'population_size': 30,        # GP population size (reduced from 50 for Pi stability)
    'generations_per_session': 20,  # Generations per nightly run
    'hours': 2.0,                 # Max hours for discovery (overrides generations)
    'novelty_weight': 0.3,        # Balance between fitness and novelty
    'min_trades': 50,             # Minimum trades for valid strategy
    'checkpoint_frequency': 5,    # Save checkpoint every N generations
}

# Pi-optimized discovery config (even smaller for limited resources / quick mode)
PI_DISCOVERY_CONFIG = {
    'population_size': 20,        # Minimal population for quick tests
    'generations_per_session': 10,
    'hours': 1.0,
    'novelty_weight': 0.3,
    'min_trades': 50,
    'checkpoint_frequency': 5,
}

# Fitness metric to optimize (now using composite multi-metric fitness)
# Individual metric kept for backward compatibility in constraint checking
FITNESS_METRIC = 'composite'  # Uses calculate_composite_fitness()


# ============================================================================
# DATA SCOPE CONFIGURATION
# ============================================================================
# Limits data to prevent extremely long backtests
# Based on profiling: 100 symbols × 5 years takes ~50s per backtest
# Full dataset (2500 symbols × 35 years) takes 1.5+ hours per backtest!

# Data scope configs now driven by PERF (from config.py)
# Override with environment: TRADING_PERF_PROFILE=workstation
DATA_SCOPE_CONFIG = {
    'max_symbols': PERF['max_symbols'],
    'max_years': PERF['max_years'],
    'min_data_points': 200,
}

# Weekend/Quick modes scale from PERF baseline
WEEKEND_DATA_SCOPE = {
    'max_symbols': min(PERF['max_symbols'] * 2, 200),
    'max_years': min(PERF['max_years'] * 3, 15),
    'min_data_points': 504,
}

QUICK_DATA_SCOPE = {
    'max_symbols': PERF['max_symbols'],
    'max_years': PERF['max_years'],
    'min_data_points': 200,
}


# ============================================================================
# ADAPTIVE GA CONFIGURATION
# ============================================================================

# Enable adaptive GA by default
ENABLE_ADAPTIVE_GA = True  # Enabled for nightly runs

# Adaptive GA configuration
ADAPTIVE_GA_CONFIG = {
    'total_population': 60,
    'n_islands': 4,
    'generations_per_session': 10,
    'generations_per_island': 3,
    'rapid_generations': 5,       # Extra rapid generations on short periods
    'use_rapid_testing': True,    # Test on short periods first
    'regime_weight': 0.35,        # Weight for regime-matched performance
    'crisis_weight': 0.15,        # Weight for crisis resilience
    'long_term_weight': 0.35,     # Weight for long-term performance
    'consistency_weight': 0.15,   # Weight for consistency
}

# Pi-optimized adaptive config (smaller for limited resources)
PI_ADAPTIVE_CONFIG = {
    'total_population': 40,
    'n_islands': 3,
    'generations_per_session': 6,
    'generations_per_island': 2,
    'rapid_generations': 3,
    'use_rapid_testing': True,
    'regime_weight': 0.35,
    'crisis_weight': 0.15,
    'long_term_weight': 0.35,
    'consistency_weight': 0.15,
}


# ============================================================================
# INTERVENTION CONFIGURATION
# ============================================================================

# Enable human intervention by default
ENABLE_INTERVENTION = False  # Set to True to enable by default

# Intervention mode: autonomous, notify_only, review_recommended, approval_required
DEFAULT_INTERVENTION_MODE = "review_recommended"

# Intervention checkpoints in nightly research
INTERVENTION_CHECKPOINTS = {
    'pre_research': {
        'enabled': True,
        'priority': 'low',
        'timeout_minutes': 5,
        'description': 'Before starting nightly research run',
    },
    'apply_ga_results': {
        'enabled': True,
        'priority': 'medium',
        'timeout_minutes': 30,
        'description': 'Before applying GA optimization results to strategies',
    },
    'apply_discovered_strategy': {
        'enabled': True,
        'priority': 'high',
        'timeout_minutes': 60,
        'description': 'Before adding a newly discovered strategy',
    },
    'regime_change_rebalance': {
        'enabled': True,
        'priority': 'medium',
        'timeout_minutes': 15,
        'description': 'Before rebalancing based on regime change',
    },
    'post_research_summary': {
        'enabled': False,  # Just informational
        'priority': 'low',
        'timeout_minutes': 5,
        'description': 'Review research results before completion',
    },
}


# ============================================================================
# FITNESS CONSTRAINTS
# ============================================================================

# Constraint thresholds (configurable per strategy type)
CONSTRAINT_THRESHOLDS = {
    'default': {
        # Hard constraints (rejection if violated)
        'min_trades': 30,           # Minimum trades for statistical significance
        'max_drawdown': -30,        # Maximum acceptable drawdown (%)
        'min_annual_return': -10,   # Allow up to -10% annual return in OOS (was 0, too strict for GA)
        'min_win_rate': 30,         # Minimum win rate (%) - relaxed from 35 for exploration
        # Soft constraint thresholds
        'low_trades': 50,           # Below this gets penalized
        'moderate_drawdown': -20,   # Below this gets penalized
        'low_win_rate': 45,         # Below this gets penalized
        'suspicious_sharpe': 2.0,   # Above this gets penalized (likely overfit)
    },
    # Strategy-specific overrides (momentum/trend strategies need fewer trades)
    'momentum': {
        'min_trades': 20,
        'low_trades': 35,
    },
    'mean_reversion': {
        'min_trades': 40,           # Mean reversion should trade more frequently
        'low_trades': 60,
    },
}

# Map strategies to constraint profiles
STRATEGY_CONSTRAINT_PROFILE = {
    'vol_managed_momentum': 'momentum',
    'factor_momentum': 'momentum',
    'sector_rotation': 'momentum',
    'mean_reversion': 'mean_reversion',
    'pairs_trading': 'mean_reversion',
    # Others use 'default'
}


def get_constraint_thresholds(strategy_name: str) -> dict:
    """Get constraint thresholds for a strategy, with profile overrides."""
    # Start with defaults
    thresholds = CONSTRAINT_THRESHOLDS['default'].copy()

    # Apply strategy-specific profile overrides
    profile = STRATEGY_CONSTRAINT_PROFILE.get(strategy_name, 'default')
    if profile in CONSTRAINT_THRESHOLDS and profile != 'default':
        thresholds.update(CONSTRAINT_THRESHOLDS[profile])

    return thresholds


# Minimum fitness for "rejected" individuals - allows some selection pressure
# while maintaining diversity. Set to small positive value instead of 0.
REJECTION_FITNESS = 0.01


def apply_fitness_constraints(
    result,
    strategy_name: str = 'default',
    is_oos: bool = False,
    test_ratio: float = 0.3
) -> tuple:
    """
    Apply hard and soft constraints to fitness evaluation.

    IMPORTANT: For OOS (out-of-sample) testing, constraints are automatically
    relaxed because:
    - Fewer trades expected (only test_ratio of time period)
    - Higher variance in metrics
    - More extreme drawdowns possible

    Instead of hard rejection (fitness=0), we now return REJECTION_FITNESS
    to maintain genetic diversity while still penalizing poor performers.

    Args:
        result: Backtest result object with metrics
        strategy_name: Name of strategy for profile-specific thresholds
        is_oos: True if evaluating out-of-sample results
        test_ratio: Fraction of data used for OOS test (default 0.3 = 30%)

    Returns:
        (multiplier, reason) where multiplier is applied to base fitness
    """
    thresholds = get_constraint_thresholds(strategy_name)

    # Scale trade-count thresholds for OOS testing
    # If we only have 30% of the data, expect ~30% of the trades
    if is_oos:
        oos_scale = max(0.25, test_ratio)  # Floor at 25% to avoid too-low thresholds
        min_trades_scaled = max(5, int(thresholds['min_trades'] * oos_scale))
        low_trades_scaled = max(10, int(thresholds['low_trades'] * oos_scale))
    else:
        min_trades_scaled = thresholds['min_trades']
        low_trades_scaled = thresholds['low_trades']

    # ========================================================================
    # HARD CONSTRAINTS - Return small positive fitness (not 0)
    # This maintains diversity while strongly penalizing poor performers
    # ========================================================================

    # Minimum trades for statistical significance
    if result.total_trades < min_trades_scaled:
        return REJECTION_FITNESS, f"Low fitness: Only {result.total_trades} trades (min {min_trades_scaled})"

    # Maximum drawdown (catastrophic loss prevention) - relaxed thresholds
    max_dd_threshold = thresholds.get('max_drawdown', -35)
    if result.max_drawdown_pct < max_dd_threshold:
        return REJECTION_FITNESS, f"Low fitness: {result.max_drawdown_pct:.1f}% drawdown (max {max_dd_threshold}%)"

    # Annual return threshold (relaxed for OOS)
    min_return = thresholds.get('min_annual_return', -5)
    if is_oos:
        min_return = -10  # Allow up to -10% for OOS since it's a shorter period
    if hasattr(result, 'annual_return') and result.annual_return < min_return:
        return REJECTION_FITNESS, f"Low fitness: {result.annual_return:.1f}% annual return (min {min_return}%)"

    # Minimum win rate (avoid extreme skew strategies) - relaxed
    min_wr = thresholds.get('min_win_rate', 30)
    if is_oos:
        min_wr = max(25, min_wr - 10)  # Relax by 10% for OOS, floor at 25%
    if hasattr(result, 'win_rate') and result.win_rate < min_wr:
        return REJECTION_FITNESS, f"Low fitness: {result.win_rate:.1f}% win rate (min {min_wr}%)"

    # ========================================================================
    # SOFT CONSTRAINTS - Reduce fitness but don't reject
    # ========================================================================
    multiplier = 1.0
    reasons = []

    # Penalize low trade count (graduated penalty)
    if result.total_trades < low_trades_scaled:
        trade_ratio = result.total_trades / low_trades_scaled
        penalty = 0.7 + (0.3 * trade_ratio)  # 0.7 to 1.0
        multiplier *= penalty
        reasons.append(f"Low trades ({result.total_trades}/{low_trades_scaled}): {penalty:.2f}x")

    # Penalize moderate drawdown (graduated)
    moderate_dd = thresholds.get('moderate_drawdown', -20)
    if result.max_drawdown_pct < moderate_dd:
        dd_severity = (moderate_dd - result.max_drawdown_pct) / 15
        penalty = max(0.6, 0.9 - (0.2 * dd_severity))
        multiplier *= penalty
        reasons.append(f"DD ({result.max_drawdown_pct:.1f}%): {penalty:.2f}x")

    # Penalize low win rate (graduated)
    low_wr = thresholds.get('low_win_rate', 45)
    if hasattr(result, 'win_rate') and result.win_rate < low_wr:
        wr_gap = low_wr - result.win_rate
        penalty = max(0.7, 1.0 - (wr_gap / 30))
        multiplier *= penalty
        reasons.append(f"WR ({result.win_rate:.1f}%): {penalty:.2f}x")

    # Penalize suspiciously high Sharpe (likely overfit) - less aggressive
    suspicious_sharpe = thresholds.get('suspicious_sharpe', 2.5)
    if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio > suspicious_sharpe:
        excess = result.sharpe_ratio - suspicious_sharpe
        penalty = max(0.5, 0.85 - (0.1 * excess))
        multiplier *= penalty
        reasons.append(f"High Sharpe ({result.sharpe_ratio:.2f}): {penalty:.2f}x")

    reason = "; ".join(reasons) if reasons else "Passed all constraints"
    return multiplier, reason

# Market close time (Eastern)
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


# ============================================================================
# STRATEGY FACTORIES
# ============================================================================

def create_strategy_instance(strategy_name: str, params: dict = None):
    """
    Create a strategy instance with optional parameter overrides.
    
    Note: Strategies use class attributes rather than __init__ params,
    so we instantiate first then set attributes from the genes dict.
    """
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.pairs_trading import PairsTradingStrategy
    from strategies.relative_volume_breakout import RelativeVolumeBreakout
    from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
    from strategies.sector_rotation import SectorRotationStrategy
    from strategies.quality_small_cap_value import QualitySmallCapValueStrategy
    from strategies.factor_momentum import FactorMomentumStrategy
    
    factories = {
        'vol_managed_momentum': VolManagedMomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'pairs_trading': PairsTradingStrategy,
        'relative_volume_breakout': RelativeVolumeBreakout,
        'vix_regime_rotation': VIXRegimeRotationStrategy,
        'sector_rotation': SectorRotationStrategy,
        'quality_small_cap_value': QualitySmallCapValueStrategy,
        'quality_smallcap_value': QualitySmallCapValueStrategy,  # alias
        'factor_momentum': FactorMomentumStrategy,
    }
    
    if strategy_name not in factories:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Create strategy instance (no params to __init__)
    strategy = factories[strategy_name]()
    
    # Apply gene parameters as attributes
    # Strategies store config as instance attributes, not constructor args
    if params:
        # Map gene names to strategy attribute names
        # Note: Some strategies use CAPS for class-level constants
        PARAM_MAPPING = {
            # vol_managed_momentum (lowercase attrs)
            'formation_period': 'formation_period',
            'skip_period': 'skip_period', 
            'vol_lookback': 'vol_lookback',
            'target_vol': 'target_vol',
            'top_pct': 'top_percentile',
            # pairs_trading (CAPS for class constants)
            'entry_z': 'ENTRY_ZSCORE',
            'exit_z': 'EXIT_ZSCORE',
            'stop_z': 'STOP_ZSCORE',
            'min_correlation': 'min_correlation',
            'max_half_life': 'max_half_life',
            'max_hold_days': 'MAX_HOLD_DAYS',
            # relative_volume_breakout (CAPS for class constants)
            'min_rv': 'MIN_RELATIVE_VOLUME',
            'min_gap_pct': 'MIN_GAP_PCT',
            'atr_stop_mult': 'ATR_STOP_MULT',
            'atr_target_mult': 'ATR_TARGET_MULT', 
            # quality_small_cap_value (CAPS for class constants)
            'min_roa': 'MIN_ROA',
            'min_profit_margin': 'MIN_PROFIT_MARGIN',
            'max_debt_to_equity': 'MAX_DEBT_TO_EQUITY',
            'value_percentile': 'VALUE_PERCENTILE',
            'max_positions': 'MAX_POSITIONS',
            'max_single_position': 'MAX_SINGLE_POSITION',
            # factor_momentum (CAPS for class constants)
            'formation_period_long': 'FORMATION_PERIOD_LONG',
            'formation_period_med': 'FORMATION_PERIOD_MED',
            'max_factor_weight': 'MAX_FACTOR_WEIGHT',
            'min_factor_weight': 'MIN_FACTOR_WEIGHT',
            # vix_regime_rotation
            'low_vix_threshold': 'LOW_VIX_THRESHOLD',
            'high_vix_threshold': 'HIGH_VIX_THRESHOLD',
            'extreme_vix_threshold': 'EXTREME_VIX_THRESHOLD',
            'high_vix_reduction': 'HIGH_VIX_REDUCTION',
            'extreme_vix_reduction': 'EXTREME_VIX_REDUCTION',
            # sector_rotation
            'momentum_period': 'MOMENTUM_PERIOD',
            'top_n_sectors': 'TOP_N_SECTORS',
            'rebalance_days': 'REBALANCE_DAYS',
            # mean_reversion
            'lookback_period': 'LOOKBACK_PERIOD',
            'entry_std': 'ENTRY_STD',
            'exit_std': 'EXIT_STD',
            'stop_std': 'STOP_STD',
        }
        
        for gene_name, value in params.items():
            attr_name = PARAM_MAPPING.get(gene_name, gene_name)
            if hasattr(strategy, attr_name):
                setattr(strategy, attr_name, value)
            elif hasattr(strategy, gene_name):
                # Fallback: try gene name directly
                setattr(strategy, gene_name, value)
            # else: silently skip unknown params
    
    return strategy


def create_fitness_function(strategy_name: str, backtester: Backtester,
                           data: dict, vix_data=None, population_size: int = 30,
                           walk_forward: bool = True, train_ratio: float = 0.7,
                           degradation_threshold: float = 0.2):
    """
    Create fitness function for GA optimization.

    Uses walk-forward backtest to prevent overfitting:
    - Splits data into 70% train / 30% test (configurable)
    - Runs backtest on both sets
    - Uses OUT-OF-SAMPLE (test) fitness as primary metric
    - Penalizes strategies with high train-to-test degradation

    Includes progress logging and constraint enforcement.

    Args:
        strategy_name: Name of strategy to create
        backtester: Backtester instance
        data: Historical data dict {symbol: DataFrame}
        vix_data: Optional VIX data DataFrame
        population_size: Population size for progress logging
        walk_forward: Enable walk-forward validation (default: True)
        train_ratio: Ratio of data for training (default: 0.7 = 70%)
        degradation_threshold: Max allowed degradation before penalty (default: 0.2 = 20%)

    Returns:
        Fitness function: genes -> float
    """
    import pandas as pd

    # Progress tracking via closure
    eval_count = [0]  # Use list to allow mutation in closure
    rejection_count = [0]  # Track rejections per generation

    # Pre-compute train/test split dates if walk-forward is enabled
    train_data = None
    test_data = None
    train_vix = None
    test_vix = None
    split_date = None
    wf_enabled = False  # Track if walk-forward is actually active

    if walk_forward and data:
        import pandas as pd

        # Helper to normalize timestamps to tz-naive
        def to_naive(ts):
            if hasattr(ts, 'tz') and ts.tz is not None:
                return ts.tz_localize(None)
            return ts

        # Get all unique dates across all symbols
        all_dates = set()
        for sym, df in data.items():
            if df is not None and len(df) > 0:
                # Handle both DatetimeIndex and 'timestamp' column
                if isinstance(df.index, pd.DatetimeIndex):
                    dates = df.index
                    dates = normalize_index(dates)
                    all_dates.update(dates.tolist())
                elif 'timestamp' in df.columns:
                    dates = pd.to_datetime(df['timestamp'])
                    if dates.dt.tz is not None:
                        dates = dates.dt.tz_localize(None)
                    all_dates.update(dates.tolist())

        if len(all_dates) >= 100:  # Need enough data points for meaningful split
            sorted_dates = sorted(all_dates)
            split_idx = int(len(sorted_dates) * train_ratio)
            split_date = sorted_dates[split_idx]

            # Split data
            train_data = {}
            test_data = {}
            for sym, df in data.items():
                if df is not None and len(df) > 0:
                    # Handle both DatetimeIndex and 'timestamp' column
                    if isinstance(df.index, pd.DatetimeIndex):
                        idx = df.index
                        idx = normalize_index(idx)
                        train_data[sym] = df[idx <= split_date].copy()
                        test_data[sym] = df[idx > split_date].copy()
                    elif 'timestamp' in df.columns:
                        ts = pd.to_datetime(df['timestamp'])
                        if ts.dt.tz is not None:
                            ts = ts.dt.tz_localize(None)
                        train_data[sym] = df[ts <= split_date].copy()
                        test_data[sym] = df[ts > split_date].copy()

            # Filter out symbols with insufficient data in either split
            train_data = {sym: df for sym, df in train_data.items() if len(df) >= 20}
            test_data = {sym: df for sym, df in test_data.items() if len(df) >= 20}

            # Split VIX data if provided
            if vix_data is not None and len(vix_data) > 0:
                if isinstance(vix_data.index, pd.DatetimeIndex):
                    idx = vix_data.index
                    idx = normalize_index(idx)
                    train_vix = vix_data[idx <= split_date].copy()
                    test_vix = vix_data[idx > split_date].copy()
                elif 'timestamp' in vix_data.columns:
                    ts = pd.to_datetime(vix_data['timestamp'])
                    if ts.dt.tz is not None:
                        ts = ts.dt.tz_localize(None)
                    train_vix = vix_data[ts <= split_date].copy()
                    test_vix = vix_data[ts > split_date].copy()
                else:
                    train_vix = None
                    test_vix = None

            # Verify we have enough data in both splits
            if len(train_data) >= 5 and len(test_data) >= 5:
                wf_enabled = True
                logger.info(f"Walk-forward enabled: {len(sorted_dates)} dates, split at {split_date}")
                logger.info(f"  Train: {len(train_data)} symbols ({int(train_ratio*100)}%), "
                           f"Test: {len(test_data)} symbols ({int((1-train_ratio)*100)}%)")
            else:
                logger.warning(f"Insufficient symbols after split (train={len(train_data)}, "
                              f"test={len(test_data)}), using full dataset")
        else:
            logger.warning(f"Insufficient data for walk-forward ({len(all_dates)} dates), "
                          "using full dataset")

    def fitness_fn(genes: dict) -> float:
        """Evaluate genes via backtest with walk-forward validation and constraint enforcement."""
        eval_count[0] += 1
        eval_num = eval_count[0]

        # Calculate generation and position within generation for clearer progress display
        current_gen = (eval_num - 1) // population_size + 1
        eval_in_gen = (eval_num - 1) % population_size + 1

        # Log progress every evaluation (compact format)
        genes_summary = ', '.join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in list(genes.items())[:3])
        logger.info(f"  [G{current_gen} {eval_in_gen}/{population_size}] Testing: {genes_summary}...")
        # CPU throttle between evaluations (prevents Pi crashes)
        sleep_time = PERF.get("sleep_between_evals", 0)
        if sleep_time > 0:
            time.sleep(sleep_time)

        start_time = time.time()

        try:
            # Create strategy with evolved parameters
            strategy = create_strategy_instance(strategy_name, genes)

            if wf_enabled:
                # === WALK-FORWARD VALIDATION ===

                # Run backtest on TRAINING data (in-sample)
                train_result = backtester.run(
                    strategy=strategy,
                    data=train_data,
                    vix_data=train_vix
                )

                # Recreate strategy to reset any internal state for test run
                strategy = create_strategy_instance(strategy_name, genes)

                # Run backtest on TEST data (out-of-sample)
                test_result = backtester.run(
                    strategy=strategy,
                    data=test_data,
                    vix_data=test_vix
                )

                # Apply constraints to TEST (out-of-sample) result
                # Pass is_oos=True to use relaxed thresholds for OOS testing
                constraint_mult, constraint_reason = apply_fitness_constraints(
                    test_result,
                    strategy_name,
                    is_oos=True,
                    test_ratio=(1 - train_ratio)
                )

                # No more hard rejection - constraint_mult is always > 0 now
                if constraint_mult <= REJECTION_FITNESS:
                    rejection_count[0] += 1
                    elapsed = time.time() - start_time
                    logger.debug(f"  [G{current_gen} {eval_in_gen}/{population_size}] OOS {constraint_reason} ({elapsed:.1f}s)")
                    return constraint_mult  # Return small positive value, not 0

                # Calculate composite fitness for both train and test
                train_fitness = calculate_composite_fitness(train_result, verbose=False)
                test_fitness = calculate_composite_fitness(test_result, verbose=False)

                # Handle edge cases
                if test_fitness == float('inf') or test_fitness == float('-inf'):
                    test_fitness = 0.0
                if train_fitness == float('inf') or train_fitness == float('-inf'):
                    train_fitness = 0.0

                # Use OUT-OF-SAMPLE as primary fitness
                fitness = test_fitness

                # Calculate degradation and apply penalty
                degradation = 0.0
                if train_fitness > 0 and test_fitness < train_fitness:
                    degradation = (train_fitness - test_fitness) / train_fitness
                    if degradation > degradation_threshold:
                        # Penalize fitness proportionally to degradation
                        # More degradation = lower multiplier, but never below 0.5
                        penalty = max(0.5, 1.0 - degradation)
                        fitness *= penalty

                # Apply soft constraint penalties
                final_fitness = fitness * constraint_mult

                elapsed = time.time() - start_time

                # Log walk-forward results
                deg_str = f"Deg={degradation:.0%}" if degradation > 0 else "Deg=0%"
                if constraint_mult < 1.0:
                    logger.info(
                        f"  [G{current_gen} {eval_in_gen}/{population_size}] WF: Train={train_fitness:.3f} ({train_result.total_trades}t), "
                        f"Test={test_fitness:.3f} ({test_result.total_trades}t), {deg_str}, "
                        f"Fit={final_fitness:.3f} ({elapsed:.1f}s) [{constraint_reason}]"
                    )
                else:
                    logger.info(
                        f"  [G{current_gen} {eval_in_gen}/{population_size}] WF: Train={train_fitness:.3f} ({train_result.total_trades}t), "
                        f"Test={test_fitness:.3f} ({test_result.total_trades}t), {deg_str}, "
                        f"Fit={final_fitness:.3f} ({elapsed:.1f}s)"
                    )

                return float(final_fitness)

            else:
                # === FALLBACK: Full dataset (no walk-forward) ===
                result = backtester.run(
                    strategy=strategy,
                    data=data,
                    vix_data=vix_data
                )

                # Apply constraints
                constraint_mult, constraint_reason = apply_fitness_constraints(
                    result, strategy_name
                )

                if constraint_mult == 0:
                    rejection_count[0] += 1
                    elapsed = time.time() - start_time
                    logger.debug(f"  [G{current_gen} {eval_in_gen}/{population_size}] {constraint_reason} ({elapsed:.1f}s)")
                    return 0.0

                # Calculate composite fitness
                base_fitness = calculate_composite_fitness(result, verbose=False)

                # Handle edge cases
                if base_fitness == float('inf') or base_fitness == float('-inf'):
                    return 0.0

                # Apply soft constraint penalties
                final_fitness = base_fitness * constraint_mult

                elapsed = time.time() - start_time

                # Get component metrics for detailed logging
                sharpe = max(0, min(result.sharpe_ratio or 0, 3))
                sortino = max(0, min(result.sortino_ratio or sharpe * 1.2, 4))
                annual_ret = (result.annual_return or 0) / 100 if abs(result.annual_return or 0) > 1 else (result.annual_return or 0)
                max_dd = abs((result.max_drawdown_pct or 0) / 100) if abs(result.max_drawdown_pct or 0) > 1 else abs(result.max_drawdown_pct or 0)
                calmar = min(annual_ret / max_dd, 3) if max_dd > 0.01 else 0
                win_rate_norm = ((result.win_rate or 0) / 100) * 2 if (result.win_rate or 0) > 1 else (result.win_rate or 0) * 2

                # Log with component breakdown
                if constraint_mult < 1.0:
                    logger.info(
                        f"  [G{current_gen} {eval_in_gen}/{population_size}] Composite: {base_fitness:.3f} * {constraint_mult:.2f} = {final_fitness:.3f} "
                        f"(Sh={sharpe:.2f} So={sortino:.2f} Ca={calmar:.2f} WR={win_rate_norm:.2f}) "
                        f"[{result.total_trades} trades, {elapsed:.1f}s] [{constraint_reason}]"
                    )
                else:
                    logger.info(
                        f"  [G{current_gen} {eval_in_gen}/{population_size}] Composite: {final_fitness:.3f} "
                        f"(Sh={sharpe:.2f} So={sortino:.2f} Ca={calmar:.2f} WR={win_rate_norm:.2f}) "
                        f"[{result.total_trades} trades, {elapsed:.1f}s]"
                    )

                return float(final_fitness)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"  [G{current_gen} {eval_in_gen}/{population_size}] Failed ({elapsed:.1f}s): {e}")
            return 0.0

    # Reset counter for each new generation
    def reset_counter():
        eval_count[0] = 0
        rejection_count[0] = 0

    def get_rejection_count():
        return rejection_count[0]

    fitness_fn.reset_counter = reset_counter
    fitness_fn.get_rejection_count = get_rejection_count
    fitness_fn.walk_forward_enabled = wf_enabled  # Expose for debugging
    
    # Set up parallel fitness context for module-level parallel evaluation
    setup_parallel_fitness_context(
        strategy_name=strategy_name,
        backtester=backtester,
        data=data,
        vix_data=vix_data,
        train_data=train_data,
        test_data=test_data,
        train_vix=train_vix,
        test_vix=test_vix,
        wf_enabled=wf_enabled,
        degradation_threshold=degradation_threshold
    )
    
    return fitness_fn


# ============================================================================
# STRATEGY DISCOVERY ENGINE
# ============================================================================

def run_strategy_discovery(data: dict, vix_data=None, config: dict = None,
                           hours: float = None, generations: int = None,
                           resume: bool = True) -> dict:
    """
    Run GP-based strategy discovery.

    Args:
        data: Historical data dict {symbol: DataFrame}
        vix_data: Optional VIX data DataFrame
        config: Discovery configuration dict
        hours: Max hours to run (overrides config)
        generations: Generations to run (overrides config)
        resume: Whether to resume from checkpoint

    Returns:
        Results dict with discovery statistics
    """
    config = config or DISCOVERY_CONFIG

    try:
        # Import discovery engine
        from research.discovery import EvolutionEngine, EvolutionConfig
        from research.discovery.db_schema import migrate_discovery_tables, check_tables_exist
        from research.discovery.promotion_pipeline import PromotionPipeline

        # Ensure database tables exist
        if not check_tables_exist():
            logger.info("Creating discovery database tables...")
            migrate_discovery_tables()

        # Create evolution config
        evo_config = EvolutionConfig(
            population_size=config.get('population_size', 50),
            generations_per_session=config.get('generations_per_session', 20),
            novelty_weight=config.get('novelty_weight', 0.3),
            min_trades=config.get('min_trades', 50),
            checkpoint_frequency=config.get('checkpoint_frequency', 5),
        )

        # Create backtester with conservative costs
        backtester = Backtester(initial_capital=100000, cost_model='conservative')

        # Create promotion pipeline for strategy lifecycle tracking
        promotion_pipeline = PromotionPipeline()

        # Create evolution engine
        engine = EvolutionEngine(
            config=evo_config,
            backtester=backtester,
            promotion_pipeline=promotion_pipeline
        )

        # Load data
        engine.load_data(data=data, vix_data=vix_data)

        # Resume or initialize
        if resume:
            loaded = engine.load_checkpoint()
            if not loaded:
                logger.info("No discovery checkpoint found, initializing fresh population")
                engine.initialize_population()
        else:
            engine.initialize_population()

        # Determine runtime
        run_hours = hours or config.get('hours')
        run_generations = generations or config.get('generations_per_session')

        # Run evolution
        start_time = datetime.now()
        logger.info(f"Starting strategy discovery: hours={run_hours}, generations={run_generations}")

        engine.run(generations=run_generations, hours=run_hours)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Process promotions after discovery completes
        # Skip heavy validation (walk-forward + Monte Carlo) to prevent OOM crashes
        # Heavy validation should be run separately via scripts/validate_strategies_subprocess.py
        logger.info("Processing strategy promotions (skipping heavy validation)...")
        try:
            promotion_results = promotion_pipeline.process_all_promotions(skip_heavy_validation=True)
            logger.info(f"Promotion results: promoted={promotion_results.get('promoted', 0)}, "
                       f"retired={promotion_results.get('retired', 0)}, "
                       f"failed={promotion_results.get('failed', 0)}")
        except Exception as promo_err:
            logger.error(f"Promotion processing failed: {promo_err}", exc_info=True)
            promotion_results = {'error': str(promo_err)}

        # Compile results
        results = {
            'success': True,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'generations_run': engine.current_generation,
            'strategies_evaluated': engine.total_strategies_evaluated,
            'strategies_promoted': engine.strategies_promoted,
            'pareto_front_size': len(engine.pareto_front),
            'novelty_archive_size': len(engine.novelty_archive),
            'diversity': engine.novelty_archive.get_archive_diversity() if engine.novelty_archive else 0,
            'promotion_pipeline': promotion_results,
        }

        logger.info(f"Strategy discovery complete: {results['strategies_evaluated']} evaluated, "
                   f"{results['strategies_promoted']} promoted")

        return results

    except ImportError as e:
        logger.error(f"Strategy discovery not available: {e}")
        logger.error("Install DEAP with: pip install deap>=1.4.1")
        return {'success': False, 'error': str(e)}

    except Exception as e:
        logger.error(f"Strategy discovery failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# ============================================================================
# ADAPTIVE GA ENGINE
# ============================================================================

def run_adaptive_ga(data: dict, vix_data=None, config: dict = None,
                    strategies: list = None, rapid_first: bool = True,
                    generations: int = None) -> dict:
    """
    Run adaptive GA with regime-matched multi-scale testing.

    Features:
    - Tests against current market regime conditions
    - Rapid 30-second tests on short periods (COVID crash, etc.)
    - Island model for diverse strategy specialists
    - Multi-scale fitness (long-term + regime + crisis)

    Args:
        data: Historical data dict {symbol: DataFrame}
        vix_data: Optional VIX data DataFrame
        config: Adaptive GA configuration dict
        strategies: List of strategies to evolve (default: all evolvable)
        rapid_first: Run rapid generations on short periods first
        generations: Override generations from config

    Returns:
        Results dict with adaptive GA statistics
    """
    if not ADAPTIVE_GA_AVAILABLE:
        return {'success': False, 'error': 'Adaptive GA components not available'}

    config = config or ADAPTIVE_GA_CONFIG
    strategies = strategies or EVOLVABLE_STRATEGIES

    try:
        start_time = datetime.now()

        # Initialize components
        logger.info("Initializing adaptive GA components...")
        library = MarketPeriodLibrary()
        regime_engine = RegimeMatchingEngine(library)
        rapid_backtester = RapidBacktester(library, parallel=True)
        strategy_manager = AdaptiveStrategyManager(regime_engine=regime_engine)

        # Get current market conditions
        fingerprint = regime_engine.get_current_fingerprint()
        logger.info(f"Current regime: {fingerprint.overall_regime.upper()}")
        logger.info(f"VIX: {fingerprint.vix_level:.1f}, Trend: {fingerprint.trend_direction:+.2f}")

        # Find similar historical periods
        matches = regime_engine.find_matching_periods(fingerprint, n=5)
        logger.info(f"Similar periods: {[m.period.name for m in matches[:3]]}")

        # Get recommended test periods
        test_periods = regime_engine.get_ga_test_periods()
        all_test_periods = []
        for category in ['similar', 'stress', 'diverse']:
            all_test_periods.extend([p.name for p in test_periods.get(category, [])])
        logger.info(f"Test periods: {all_test_periods}")

        # Cache data in rapid backtester
        rapid_backtester.cache_data(data)

        # Create adaptive GA config
        ga_config = AdaptiveGAConfig(
            total_population=config.get('total_population', 60),
            n_islands=config.get('n_islands', 4),
            generations_per_session=generations or config.get('generations_per_session', 10),
            generations_per_island=config.get('generations_per_island', 3),
            long_term_weight=config.get('long_term_weight', 0.35),
            regime_weight=config.get('regime_weight', 0.35),
            crisis_weight=config.get('crisis_weight', 0.15),
            consistency_weight=config.get('consistency_weight', 0.15),
            use_rapid_testing=config.get('use_rapid_testing', True),
            rapid_generations=config.get('rapid_generations', 5),
        )

        # Track results per strategy
        strategy_results = []
        total_improvements = 0

        for strategy_name in strategies:
            if strategy_name not in STRATEGY_PARAMS:
                logger.warning(f"Skipping {strategy_name}: no parameter specs")
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"Adaptive evolution: {strategy_name}")
            logger.info(f"{'='*50}")

            try:
                # Create optimizer with strategy-specific parameter specs
                # STRATEGY_PARAMS returns a list of ParameterSpec objects
                param_specs = {}
                for spec in STRATEGY_PARAMS.get(strategy_name, []):
                    param_specs[spec.name] = (spec.min_val, spec.max_val, spec.step)

                if not param_specs:
                    logger.warning(f"No parameters for {strategy_name}, skipping")
                    continue

                optimizer = AdaptiveGAOptimizer(
                    config=ga_config,
                    parameter_specs=param_specs,
                    period_library=library,
                    regime_engine=regime_engine,
                )

                # Create strategy factory
                def make_factory(name):
                    def factory(genes):
                        return create_strategy_instance(name, genes)
                    return factory

                strategy_factory = make_factory(strategy_name)

                # Run rapid generations first if enabled
                if rapid_first and config.get('use_rapid_testing', True):
                    logger.info(f"Running {ga_config.rapid_generations} rapid generations...")
                    rapid_best = optimizer.run_rapid_generations(
                        strategy_factory=strategy_factory,
                        data=data,
                        vix_data=vix_data,
                        n_generations=ga_config.rapid_generations,
                    )
                    logger.info(f"Rapid best fitness: {rapid_best.fitness:.4f}")

                # Run main evolution
                logger.info(f"Running {ga_config.generations_per_session} adaptive generations...")
                best = optimizer.evolve(
                    strategy_factory=strategy_factory,
                    data=data,
                    vix_data=vix_data,
                    current_conditions={
                        'vix': fingerprint.vix_level,
                        'trend': fingerprint.trend_direction,
                        'correlation': fingerprint.correlation_level,
                        'regime': fingerprint.overall_regime,
                    },
                    generations=ga_config.generations_per_session,
                )

                # Record result
                result = {
                    'strategy': strategy_name,
                    'best_fitness': best.fitness,
                    'long_term_fitness': best.long_term_fitness,
                    'regime_fitness': best.regime_fitness,
                    'crisis_fitness': best.crisis_fitness,
                    'consistency_score': best.consistency_score,
                    'best_genes': best.genes,
                    'generation': best.generation,
                    'improved': best.fitness > 0.5,  # Threshold for "good"
                }

                strategy_results.append(result)

                if result['improved']:
                    total_improvements += 1
                    logger.info(f"✅ {strategy_name}: Fitness {best.fitness:.4f}")
                else:
                    logger.info(f"➖ {strategy_name}: Fitness {best.fitness:.4f}")

                # Print breakdown
                logger.info(f"   Long-term: {best.long_term_fitness:.3f}")
                logger.info(f"   Regime: {best.regime_fitness:.3f}")
                logger.info(f"   Crisis: {best.crisis_fitness:.3f}")
                logger.info(f"   Consistency: {best.consistency_score:.3f}")

            except Exception as e:
                logger.error(f"Failed to evolve {strategy_name}: {e}", exc_info=True)
                strategy_results.append({
                    'strategy': strategy_name,
                    'error': str(e),
                    'improved': False,
                })

        # Get final strategy allocations based on regime
        allocations = strategy_manager.get_strategy_allocations()
        active_strategies = strategy_manager.get_active_strategies()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results = {
            'success': True,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'current_regime': fingerprint.overall_regime,
            'vix_level': fingerprint.vix_level,
            'strategies_evolved': len(strategy_results),
            'improvements_found': total_improvements,
            'strategy_results': strategy_results,
            'test_periods_used': all_test_periods,
            'recommended_allocations': allocations,
            'active_strategies': active_strategies,
        }

        logger.info(f"\nAdaptive GA complete: {len(strategy_results)} strategies, "
                   f"{total_improvements} improvements")

        return results

    except Exception as e:
        logger.error(f"Adaptive GA failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# ============================================================================
# CORE RESEARCH ENGINE
# ============================================================================

class NightlyResearchEngine:
    """
    Orchestrates nightly research runs across all strategies.

    Supports three modes:
    1. Parameter optimization: Evolve parameters of existing strategies
    2. Strategy discovery: GP-based discovery of novel strategies
    3. Adaptive GA: Regime-matched multi-scale testing (NEW)
    """

    def __init__(self, strategies: List[str] = None, quick_mode: bool = False,
                 enable_discovery: bool = False, discovery_config: dict = None,
                 enable_adaptive: bool = False, adaptive_config: dict = None,
                 rapid_first: bool = True,
                 enable_intervention: bool = False, intervention_mode: str = None,
                 data_scope: str = None, data_scope_config: dict = None):
        """
        Initialize research engine.

        Args:
            strategies: List of strategies to evolve (default: EVOLVABLE_STRATEGIES)
            quick_mode: Use smaller population for faster testing
            enable_discovery: Enable GP-based strategy discovery
            discovery_config: Config for strategy discovery (default: DISCOVERY_CONFIG)
            enable_adaptive: Enable adaptive GA with regime matching
            adaptive_config: Config for adaptive GA (default: ADAPTIVE_GA_CONFIG)
            rapid_first: Run rapid generations on short periods first (adaptive mode)
            enable_intervention: Enable human intervention checkpoints
            intervention_mode: Override intervention mode (autonomous, notify_only, etc.)
            data_scope: Preset data scope ('quick', 'weekend', or None for default)
            data_scope_config: Custom data scope config (overrides preset)
        """
        self.strategies = strategies or EVOLVABLE_STRATEGIES
        self.quick_mode = quick_mode
        self.ga_config = QUICK_GA_CONFIG if quick_mode else NIGHTLY_GA_CONFIG

        # Data scope configuration (controls backtest speed)
        if data_scope_config:
            self.data_scope_config = data_scope_config
        elif data_scope == 'quick':
            self.data_scope_config = QUICK_DATA_SCOPE
        elif data_scope == 'weekend':
            self.data_scope_config = WEEKEND_DATA_SCOPE
        else:
            self.data_scope_config = DATA_SCOPE_CONFIG

        # Strategy discovery settings
        self.enable_discovery = enable_discovery or ENABLE_STRATEGY_DISCOVERY
        self.discovery_config = discovery_config or (
            PI_DISCOVERY_CONFIG if quick_mode else DISCOVERY_CONFIG
        )

        # Adaptive GA settings
        self.enable_adaptive = enable_adaptive or ENABLE_ADAPTIVE_GA
        self.adaptive_config = adaptive_config or (
            PI_ADAPTIVE_CONFIG if quick_mode else ADAPTIVE_GA_CONFIG
        )
        self.rapid_first = rapid_first

        # Intervention settings
        self.enable_intervention = enable_intervention or ENABLE_INTERVENTION
        self.intervention_mode = intervention_mode or DEFAULT_INTERVENTION_MODE
        self.intervention_manager = None

        if self.enable_intervention and INTERVENTION_AVAILABLE:
            mode_map = {
                'autonomous': InterventionMode.AUTONOMOUS,
                'notify_only': InterventionMode.NOTIFY_ONLY,
                'review_recommended': InterventionMode.REVIEW_RECOMMENDED,
                'approval_required': InterventionMode.APPROVAL_REQUIRED,
            }
            config = InterventionConfig(
                mode=mode_map.get(self.intervention_mode, InterventionMode.REVIEW_RECOMMENDED),
                default_timeout_minutes=30,
                approval_dir=Path('./intervention'),
            )
            self.intervention_manager = InterventionManager(config)
            logger.info(f"Intervention enabled (mode={self.intervention_mode})")

        self.db = get_db()
        self.data_manager = CachedDataManager()
        self.backtester = Backtester(initial_capital=100000)

        # Clean up any stale "running" GA runs from previous crashes
        stale_count = self.db.cleanup_stale_ga_runs()
        if stale_count > 0:
            logger.info(f"Cleaned up {stale_count} stale GA run(s) from previous session")

        # Filter to strategies that have parameter specs
        self.strategies = [
            s for s in self.strategies
            if s in STRATEGY_PARAMS
        ]

        mode_str = "QUICK" if quick_mode else "FULL"
        discovery_str = "enabled" if self.enable_discovery else "disabled"
        adaptive_str = "enabled" if self.enable_adaptive else "disabled"
        intervention_str = f"enabled ({self.intervention_mode})" if self.enable_intervention else "disabled"
        logger.info(f"Initialized research engine ({mode_str} mode)")
        logger.info(f"  Discovery: {discovery_str}, Adaptive: {adaptive_str}")
        logger.info(f"  Intervention: {intervention_str}")
        logger.info(f"  Strategies: {self.strategies}")
        logger.info(f"  Population: {self.ga_config.population_size}, Generations: {self.ga_config.generations}")
        scope = self.data_scope_config
        scope_str = f"max {scope.get('max_symbols', 'all')} symbols, {scope.get('max_years', 'all')} years"
        logger.info(f"  Data scope: {scope_str}")
        if self.enable_discovery:
            logger.info(f"  Discovery: pop={self.discovery_config['population_size']}, "
                       f"hours={self.discovery_config.get('hours', 'N/A')}")
        if self.enable_adaptive:
            logger.info(f"  Adaptive: pop={self.adaptive_config['total_population']}, "
                       f"islands={self.adaptive_config['n_islands']}, rapid_first={rapid_first}")

        # LED client for hardware feedback (sends requests to orchestrator)
        self._leds = None
        if LED_AVAILABLE and LEDClient is not None:
            try:
                self._leds = LEDClient()
            except Exception as e:
                logger.debug(f"LED client not available: {e}")

        # State
        self.data = None
        self.vix_data = None
        self.run_id = None

    def request_intervention(self, checkpoint: str, context: dict = None) -> bool:
        """
        Request intervention at a checkpoint.

        Args:
            checkpoint: Name of the checkpoint
            context: Additional context for the decision

        Returns:
            True if approved (or intervention disabled), False if rejected
        """
        if not self.enable_intervention or self.intervention_manager is None:
            return True  # Proceed if intervention disabled

        cp_config = INTERVENTION_CHECKPOINTS.get(checkpoint, {})
        if not cp_config.get('enabled', True):
            return True  # Checkpoint disabled

        priority_map = {
            'low': CheckpointPriority.LOW,
            'medium': CheckpointPriority.MEDIUM,
            'high': CheckpointPriority.HIGH,
            'critical': CheckpointPriority.CRITICAL,
        }

        return self.intervention_manager.request_approval(
            checkpoint=checkpoint,
            context=context or {},
            timeout_minutes=cp_config.get('timeout_minutes', 30),
            priority=priority_map.get(cp_config.get('priority', 'medium')),
        )
    
    def load_data(self) -> bool:
        """Load market data for backtesting with scope limits."""
        logger.info("Loading market data...")

        try:
            import pandas as pd
            from datetime import datetime, timedelta

            if not self.data_manager.cache:
                # Limit symbols upfront to prevent memory explosion (Pi fix)
                max_symbols = PERF['max_symbols']  # From performance config
                all_symbols = self.data_manager.get_available_symbols()
                symbols_to_load = sorted(all_symbols)[:max_symbols]
                logger.info(f"Loading {len(symbols_to_load)} of {len(all_symbols)} available symbols")
                self.data_manager.load_all(symbols=symbols_to_load)

            # Get data scope config
            scope = self.data_scope_config
            max_symbols = scope.get('max_symbols')
            max_years = scope.get('max_years')
            min_data_points = scope.get('min_data_points', 200)

            # Calculate cutoff date for max_years
            cutoff_date = None
            if max_years:
                cutoff_date = datetime.now() - timedelta(days=max_years * 365)
                logger.info(f"  Data scope: last {max_years} years (from {cutoff_date.strftime('%Y-%m-%d')})")

            # Get metadata for symbol ranking
            metadata = self.data_manager.get_all_metadata()

            # Sort symbols by dollar volume (most liquid first)
            if max_symbols and metadata:
                sorted_symbols = sorted(
                    metadata.items(),
                    key=lambda x: x[1].get('dollar_volume', 0),
                    reverse=True
                )
                top_symbols = set(sym for sym, _ in sorted_symbols[:max_symbols])
                logger.info(f"  Data scope: top {max_symbols} symbols by liquidity")
            else:
                top_symbols = None

            # Filter and limit data
            self.data = {}
            skipped_count = 0

            for symbol, df in self.data_manager.cache.items():
                # Filter by top symbols
                if top_symbols and symbol not in top_symbols:
                    skipped_count += 1
                    continue

                # Note: normalize_dataframe() already does .copy() internally,
                # so we don't need to copy here - avoids ~200MB memory duplication

                # Ensure datetime index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df = normalize_dataframe(df)

                # Apply date cutoff
                if cutoff_date:
                    df = df[df.index >= cutoff_date]

                # Check minimum data points
                if len(df) < min_data_points:
                    skipped_count += 1
                    continue

                self.data[symbol] = df

            logger.info(f"  Loaded {len(self.data)} symbols ({skipped_count} skipped by scope limits)")

            # Load VIX data
            vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
            if vix_path.exists():
                self.vix_data = pd.read_parquet(vix_path)
                if 'timestamp' in self.vix_data.columns:
                    self.vix_data = self.vix_data.set_index('timestamp')
                if self.vix_data.index.tz is not None:
                    self.vix_data.index = self.vix_data.index.tz_localize(None)

                # Apply same date cutoff to VIX
                if cutoff_date:
                    self.vix_data = self.vix_data[self.vix_data.index >= cutoff_date]

                # Add regime classification
                self.vix_data['regime'] = 'normal'
                self.vix_data.loc[self.vix_data['close'] < 15, 'regime'] = 'low'
                self.vix_data.loc[self.vix_data['close'] > 25, 'regime'] = 'high'
                self.vix_data.loc[self.vix_data['close'] > 40, 'regime'] = 'extreme'

            logger.info(f"Loaded {len(self.data)} symbols, VIX: {self.vix_data is not None}")

            # Pre-compute indicators for faster backtesting
            self._enrich_data()

            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _enrich_data(self):
        """
        Pre-compute common indicators to speed up backtesting.

        Adds to each DataFrame:
        - atr: Average True Range (14-period)
        - relative_volume: Volume / 20-day average volume
        - rolling_mean_20: 20-day rolling mean of close
        - rolling_std_20: 20-day rolling std of close

        This moves O(n) calculations from per-signal-generation to once-per-backtest.
        Speedup: 10-20x for daily strategies like relative_volume_breakout.
        """
        import time
        start = time.time()

        enriched_count = 0
        for symbol, df in self.data.items():
            if len(df) < 20:
                continue

            # ATR (14-period) - used by many strategies
            if 'atr' not in df.columns:
                high = df['high']
                low = df['low']
                close = df['close'].shift(1)

                tr1 = high - low
                tr2 = (high - close).abs()
                tr3 = (low - close).abs()

                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = tr.rolling(14).mean()

            # Relative Volume (20-day) - used by relative_volume_breakout
            if 'relative_volume' not in df.columns:
                avg_volume = df['volume'].rolling(20).mean()
                df['relative_volume'] = df['volume'] / avg_volume

            # Rolling stats (20-day) - used by many mean-reversion strategies
            if 'rolling_mean_20' not in df.columns:
                df['rolling_mean_20'] = df['close'].rolling(20).mean()
                df['rolling_std_20'] = df['close'].rolling(20).std()

            # Log returns - useful for momentum/volatility calculations
            if 'log_return' not in df.columns:
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            enriched_count += 1

        elapsed = time.time() - start
        logger.info(f"  Enriched {enriched_count} symbols with indicators ({elapsed:.1f}s)")

    def evolve_strategy(self, strategy_name: str, 
                        generations: int = None) -> Optional[dict]:
        """
        Evolve a single strategy.
        
        Args:
            strategy_name: Name of strategy to evolve
            generations: Number of generations (default: NIGHTLY_GA_CONFIG.generations)
            
        Returns:
            Result dict with improvement info, or None on failure
        """
        generations = generations or self.ga_config.generations
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evolving: {strategy_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create fitness function
            fitness_fn = create_fitness_function(
                strategy_name,
                self.backtester,
                self.data,
                self.vix_data,
                population_size=self.ga_config.population_size,
                walk_forward=PERF.get("walk_forward", True)
            )

            # Use context manager for automatic pool lifecycle management
            # Pool is created in __enter__, reused across all generations, shutdown in __exit__
            with PersistentGAOptimizer(
                strategy_name,
                fitness_fn,
                config=self.ga_config
            ) as optimizer:
                # Load existing population first to get accurate starting fitness
                optimizer.load_population()
                start_best = optimizer.best_ever_fitness

                # Evolve (uses persistent pool internally)
                best = optimizer.evolve_incremental(generations=generations)

                # Calculate improvement (handle -inf start for new strategies)
                if start_best == float('-inf'):
                    improvement = 0.0  # First run, no prior to compare
                else:
                    improvement = optimizer.best_ever_fitness - start_best
                improved = improvement > 0.01  # Meaningful improvement threshold

                result = {
                    'strategy': strategy_name,
                    'generation': optimizer.current_generation,
                    'best_fitness': optimizer.best_ever_fitness,
                    'best_genes': optimizer.best_ever_genes,
                    'improvement': improvement,
                    'improved': improved,
                    'generations_run': generations
                }

            # Pool automatically cleaned up here

            if improved:
                logger.info(f"[OK] {strategy_name}: Improved by {improvement:.4f}")
            else:
                logger.info(f"[-] {strategy_name}: No significant improvement")

            return result
            
        except Exception as e:
            logger.error(f"Failed to evolve {strategy_name}: {e}", exc_info=True)
            return None
    
    def run_nightly(self, generations_per_strategy: int = None,
                    skip_param_optimization: bool = False,
                    skip_discovery: bool = False,
                    discovery_hours: float = None,
                    discovery_generations: int = None,
                    adaptive_generations: int = None,
                    skip_adaptive: bool = False,
                    resume_run_id: str = None) -> dict:
        """
        Run complete nightly research cycle.

        Args:
            generations_per_strategy: Generations to run for each strategy
            skip_param_optimization: Skip parameter optimization (discovery only)
            skip_discovery: Skip strategy discovery even if enabled
            discovery_hours: Override discovery hours
            discovery_generations: Override discovery generations
            adaptive_generations: Override adaptive GA generations
            skip_adaptive: Skip adaptive GA even if enabled
            resume_run_id: If provided, resume this existing run instead of creating new

        Returns:
            Summary dict of the run
        """
        is_resume = resume_run_id is not None
        self.run_id = resume_run_id if is_resume else str(uuid.uuid4())[:8]
        start_time = datetime.now()

        logger.info("\n" + "=" * 70)
        logger.info(f"NIGHTLY RESEARCH RUN: {self.run_id}" + (" (RESUMED)" if is_resume else ""))
        logger.info(f"Started: {start_time.isoformat()}")
        if not skip_param_optimization:
            logger.info(f"Strategies: {', '.join(self.strategies)}")
        if self.enable_discovery and not skip_discovery:
            logger.info(f"Discovery: ENABLED")
        if self.enable_adaptive and not skip_adaptive:
            logger.info(f"Adaptive GA: ENABLED")
        logger.info("=" * 70)

        # Start research LED breathing (blue = evolving)
        if self._leds:
            try:
                self._leds.breathe('research', 'blue', period=3.0, min_brightness=0.2)
            except Exception as e:
                logger.debug(f"Failed to start research LED: {e}")

        # Log run start with strategies and planned generations (skip if resuming)
        if not is_resume:
            self.db.start_ga_run(self.run_id, strategies=self.strategies,
                                 planned_generations=self.ga_config.generations)

        # Load data
        if not self.load_data():
            self.db.fail_ga_run(self.run_id, "Failed to load market data")
            if self._leds:
                try:
                    self._leds.set_color('research', 'red')
                except Exception as e:
                    logger.warning(f"Failed to set LED color: {e}")
            return {'success': False, 'error': 'Data load failed'}

        # Track results
        results = []
        errors = []
        total_generations = 0
        improvements = 0
        discovery_results = None
        adaptive_results = None

        # ====================================================================
        # INTERVENTION: Pre-research checkpoint
        # ====================================================================
        if not self.request_intervention('pre_research', {
            'run_id': self.run_id,
            'strategies': self.strategies,
            'phases': {
                'param_optimization': not skip_param_optimization,
                'discovery': self.enable_discovery and not skip_discovery,
                'adaptive': self.enable_adaptive and not skip_adaptive,
            }
        }):
            logger.warning("Pre-research intervention rejected, aborting run")
            return {'success': False, 'error': 'Intervention rejected'}

        # ====================================================================
        # PHASE 1: Parameter Optimization (existing strategies)
        # ====================================================================
        if _should_stop_research():
            logger.warning("Research stop triggered before Phase 1 (shutdown or time boundary)")
            # Time boundary stop is expected behavior, not a failure
            return {'success': True, 'stopped_early': True, 'reason': 'time_boundary'}

        if not skip_param_optimization:
            logger.info("\n" + "-" * 50)
            logger.info("PHASE 1: Parameter Optimization")
            logger.info("-" * 50)

            # Parallel strategy execution
            concurrent_strategies = PERF.get('concurrent_strategies', 1)
            if concurrent_strategies > 1:
                logger.info(f"Running {len(self.strategies)} strategies with {concurrent_strategies} concurrent")
                with ThreadPoolExecutor(max_workers=concurrent_strategies) as executor:
                    future_to_strategy = {
                        executor.submit(self.evolve_strategy, strategy, generations_per_strategy): strategy
                        for strategy in self.strategies
                    }
                    for future in as_completed(future_to_strategy):
                        strategy = future_to_strategy[future]
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                                total_generations += result.get('generations_run', 0)
                                if result.get('improved'):
                                    improvements += 1
                            else:
                                errors.append(f"{strategy}: Evolution failed")
                        except Exception as e:
                            logger.error(f"{strategy}: Exception - {e}")
                            errors.append(f"{strategy}: {e}")
                # Clear parallel context after all parallel strategies complete
                clear_parallel_fitness_context()
            else:
                # Sequential fallback
                for strategy in self.strategies:
                    result = self.evolve_strategy(
                        strategy,
                        generations=generations_per_strategy
                    )
                    if result:
                        results.append(result)
                        total_generations += result.get('generations_run', 0)
                        if result.get('improved'):
                            improvements += 1
                    else:
                        errors.append(f"{strategy}: Evolution failed")

                    # Clear parallel context after each strategy to free train/test copies
                    # This prevents memory accumulation across strategy evolutions (~200-400MB each)
                    clear_parallel_fitness_context()

            # Intervention checkpoint: Apply GA results
            if improvements > 0:
                if not self.request_intervention('apply_ga_results', {
                    'improvements': improvements,
                    'strategies_evolved': len(results),
                    'results_summary': [
                        {'strategy': r.get('strategy'), 'improved': r.get('improved'),
                         'fitness_delta': r.get('fitness_delta', 0)}
                        for r in results if r.get('improved')
                    ]
                }):
                    logger.warning("GA results intervention rejected, improvements will not be applied")
                    # Note: Results are already saved by evolve_strategy, this is informational

            # Clear memory between phases to prevent OOM during long research runs
            # Ensure parallel context is cleared (may already be done per-strategy)
            clear_parallel_fitness_context()
            logger.info("Memory cleared after Phase 1")

        # ====================================================================
        # PHASE 2: Strategy Discovery (novel strategies via GP)
        # ====================================================================
        if _should_stop_research():
            logger.warning("Research stop triggered before Phase 2 (shutdown or time boundary)")
            return {'success': True, 'partial': True, 'error': 'Research stopped after Phase 1'}

        if self.enable_discovery and not skip_discovery:
            logger.info("\n" + "-" * 50)
            logger.info("PHASE 2: Strategy Discovery (GP)")
            logger.info("-" * 50)

            discovery_results = run_strategy_discovery(
                data=self.data,
                vix_data=self.vix_data,
                config=self.discovery_config,
                hours=discovery_hours,
                generations=discovery_generations,
                resume=True
            )

            if not discovery_results.get('success'):
                errors.append(f"Discovery: {discovery_results.get('error', 'Unknown error')}")

            # Intervention checkpoint: Apply discovered strategies
            discovered_count = discovery_results.get('strategies_discovered', 0)
            if discovered_count > 0:
                if not self.request_intervention('apply_discovered_strategy', {
                    'strategies_discovered': discovered_count,
                    'best_strategy': discovery_results.get('best_strategy', {}),
                    'average_fitness': discovery_results.get('average_fitness', 0),
                }):
                    logger.warning("Discovered strategy intervention rejected")
                    # Mark discovered strategies as needing review

            # Clear memory between phases to prevent OOM during long research runs
            # Aggressively clear caches to reclaim memory before Phase 3
            clear_parallel_fitness_context()
            if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'clear_cache'):
                self.data_manager.clear_cache()
            logger.info("Memory cleared after Phase 2")

        # ====================================================================
        # PHASE 3: Adaptive GA (regime-matched multi-scale testing)
        # ====================================================================
        if _should_stop_research():
            logger.warning("Research stop triggered before Phase 3 (shutdown or time boundary)")
            return {'success': True, 'partial': True, 'error': 'Research stopped after Phase 2'}

        if self.enable_adaptive and not skip_adaptive:
            logger.info("\n" + "-" * 50)
            logger.info("PHASE 3: Adaptive GA (Regime-Matched)")
            logger.info("-" * 50)

            adaptive_results = run_adaptive_ga(
                data=self.data,
                vix_data=self.vix_data,
                config=self.adaptive_config,
                strategies=self.strategies,
                rapid_first=self.rapid_first,
                generations=adaptive_generations,
            )

            if adaptive_results.get('success'):
                # Log adaptive results
                logger.info(f"\nAdaptive GA Results:")
                logger.info(f"  Current regime: {adaptive_results.get('current_regime', 'N/A').upper()}")
                logger.info(f"  VIX level: {adaptive_results.get('vix_level', 0):.1f}")
                logger.info(f"  Strategies evolved: {adaptive_results.get('strategies_evolved', 0)}")
                logger.info(f"  Improvements: {adaptive_results.get('improvements_found', 0)}")

                # Log recommended allocations
                allocations = adaptive_results.get('recommended_allocations', {})
                if allocations:
                    logger.info(f"\n  Recommended allocations:")
                    for strategy, alloc in sorted(allocations.items(), key=lambda x: -x[1]):
                        logger.info(f"    {strategy}: {alloc:.1%}")

                # Intervention checkpoint: Regime-based rebalancing
                if adaptive_results.get('rebalance_recommended', False):
                    if not self.request_intervention('regime_change_rebalance', {
                        'current_regime': adaptive_results.get('current_regime', 'unknown'),
                        'vix_level': adaptive_results.get('vix_level', 0),
                        'recommended_allocations': allocations,
                        'rebalance_reason': adaptive_results.get('rebalance_reason', 'regime change'),
                    }):
                        logger.warning("Rebalance intervention rejected, allocations will not be applied")
                        adaptive_results['rebalance_applied'] = False
                    else:
                        adaptive_results['rebalance_applied'] = True

                # Add to improvements count
                improvements += adaptive_results.get('improvements_found', 0)
            else:
                errors.append(f"Adaptive GA: {adaptive_results.get('error', 'Unknown error')}")

            # Clear memory after Phase 3 to prevent OOM during long research runs
            clear_parallel_fitness_context()
            logger.info("Memory cleared after Phase 3")

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            'run_id': self.run_id,
            'success': True,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'strategies_evolved': len(results),
            'total_generations': total_generations,
            'improvements_found': improvements,
            'errors': errors,
            'results': results,
            'discovery': discovery_results,
            'adaptive': adaptive_results,
        }

        # Log completion
        self.db.complete_ga_run(
            self.run_id,
            strategies=[r['strategy'] for r in results],
            total_generations=total_generations,
            improvements=improvements,
            errors=errors if errors else None
        )

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("NIGHTLY RESEARCH COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

        if not skip_param_optimization:
            logger.info(f"\nParameter Optimization:")
            logger.info(f"  Strategies evolved: {len(results)}/{len(self.strategies)}")
            logger.info(f"  Total generations: {total_generations}")
            logger.info(f"  Improvements found: {improvements}")

            logger.info("\n  Results by strategy:")
            for r in results:
                status = "✅" if r['improved'] else "➖"
                logger.info(
                    f"    {status} {r['strategy']}: "
                    f"Gen {r['generation']}, "
                    f"Fitness {r['best_fitness']:.4f} "
                    f"({r['improvement']:+.4f})"
                )

        if discovery_results:
            logger.info(f"\nStrategy Discovery:")
            if discovery_results.get('success'):
                logger.info(f"  Generations: {discovery_results.get('generations_run', 0)}")
                logger.info(f"  Strategies evaluated: {discovery_results.get('strategies_evaluated', 0)}")
                logger.info(f"  Strategies promoted: {discovery_results.get('strategies_promoted', 0)}")
                logger.info(f"  Pareto front size: {discovery_results.get('pareto_front_size', 0)}")
                logger.info(f"  Novelty archive: {discovery_results.get('novelty_archive_size', 0)}")
                logger.info(f"  Diversity: {discovery_results.get('diversity', 0):.4f}")
            else:
                logger.warning(f"  Failed: {discovery_results.get('error', 'Unknown')}")

        if adaptive_results:
            logger.info(f"\nAdaptive GA (Regime-Matched):")
            if adaptive_results.get('success'):
                logger.info(f"  Current regime: {adaptive_results.get('current_regime', 'N/A').upper()}")
                logger.info(f"  VIX level: {adaptive_results.get('vix_level', 0):.1f}")
                logger.info(f"  Strategies evolved: {adaptive_results.get('strategies_evolved', 0)}")
                logger.info(f"  Improvements: {adaptive_results.get('improvements_found', 0)}")
                logger.info(f"  Test periods: {len(adaptive_results.get('test_periods_used', []))}")

                # Show strategy-specific results
                for sr in adaptive_results.get('strategy_results', []):
                    if 'error' not in sr:
                        status = "✅" if sr.get('improved') else "➖"
                        logger.info(
                            f"    {status} {sr['strategy']}: "
                            f"Fitness {sr['best_fitness']:.4f} "
                            f"(LT={sr['long_term_fitness']:.2f}, "
                            f"Reg={sr['regime_fitness']:.2f}, "
                            f"Crisis={sr['crisis_fitness']:.2f})"
                        )

                # Show active strategies
                active = adaptive_results.get('active_strategies', [])
                if active:
                    logger.info(f"  Active strategies: {', '.join(active)}")
            else:
                logger.warning(f"  Failed: {adaptive_results.get('error', 'Unknown')}")

        if errors:
            logger.warning(f"\nErrors: {len(errors)}")
            for err in errors:
                logger.warning(f"  - {err}")

        # Set research LED to final state (green=success, yellow=partial, red=error)
        if self._leds:
            try:
                if errors:
                    self._leds.set_color('research', 'yellow')  # Completed with warnings
                else:
                    self._leds.set_color('research', 'green')  # Full success
            except Exception as e:
                logger.debug(f"Failed to set research LED: {e}")

        return summary


# ============================================================================
# STATUS REPORTING
# ============================================================================

def print_status():
    """Print current GA status for all strategies."""
    db = get_db()

    print("\n" + "=" * 70)
    print("GENETIC ALGORITHM STATUS")
    print("=" * 70)

    # Recent runs
    print("\nRecent Runs:")
    print("-" * 50)
    runs = db.get_recent_ga_runs(limit=5)
    if runs:
        for run in runs:
            print(
                f"  {run['run_id']} | {run['start_time'][:16]} | "
                f"{run['status']} | "
                f"Gens: {run['total_generations']} | "
                f"Improvements: {run['improvements_found']}"
            )
    else:
        print("  No runs recorded yet")

    # Per-strategy status
    print("\nStrategy Status:")
    print("-" * 50)

    for strategy in EVOLVABLE_STRATEGIES:
        pop = db.load_ga_population(strategy)
        best = db.get_ga_best_all_time(strategy)

        if pop:
            print(f"\n  {strategy}:")
            print(f"    Current generation: {pop['generation']}")
            print(f"    Population size: {pop['population_size']}")
            print(f"    Best fitness: {pop['best_fitness']:.4f}")
            if best and best['best_genes']:
                print(f"    Best genes: {best['best_genes']}")
        else:
            print(f"\n  {strategy}: Not yet evolved")

    # Improvement trends (last 7 days)
    print("\n7-Day Improvement Trends:")
    print("-" * 50)

    for strategy in EVOLVABLE_STRATEGIES:
        history = db.get_ga_history(strategy, days=7)
        if len(history) >= 2:
            first = history[0]['best_fitness']
            last = history[-1]['best_fitness']
            improvement = last - first
            print(f"  {strategy}: {first:.4f} → {last:.4f} ({improvement:+.4f})")
        elif history:
            print(f"  {strategy}: {history[0]['best_fitness']:.4f} (insufficient history)")
        else:
            print(f"  {strategy}: No history")

    # Strategy Discovery Status
    print("\n" + "=" * 70)
    print("STRATEGY DISCOVERY STATUS")
    print("=" * 70)

    try:
        # Check for evolution checkpoints
        checkpoint = db.fetchone(
            "research",
            """
            SELECT checkpoint_id, generation, created_at
            FROM evolution_checkpoints
            ORDER BY generation DESC LIMIT 1
            """
        )

        if checkpoint:
            print(f"\nLatest Checkpoint:")
            print(f"  ID: {checkpoint['checkpoint_id']}")
            print(f"  Generation: {checkpoint['generation']}")
            print(f"  Created: {checkpoint['created_at']}")
        else:
            print("\n  No discovery checkpoints found")

        # Check for discovered strategies
        discovered = db.fetchall(
            "research",
            """
            SELECT strategy_id, status, oos_sortino, oos_max_drawdown,
                   oos_total_trades, novelty_score, created_at
            FROM discovered_strategies
            ORDER BY oos_sortino DESC
            LIMIT 10
            """
        )

        if discovered:
            print(f"\nDiscovered Strategies ({len(discovered)} shown):")
            print("-" * 50)
            for d in discovered:
                sortino = d['oos_sortino'] or 0
                dd = d['oos_max_drawdown'] or 0
                trades = d['oos_total_trades'] or 0
                novelty = d['novelty_score'] or 0
                print(
                    f"  {d['strategy_id']}: "
                    f"Sortino={sortino:.2f}, DD={dd:.1f}%, "
                    f"Trades={trades}, Novelty={novelty:.3f} "
                    f"[{d['status']}]"
                )
        else:
            print("\n  No strategies discovered yet")

    except Exception as e:
        print(f"\n  Discovery tables not initialized: {e}")
        print("  Run: python -m research.discovery.db_schema")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Nightly research runner for strategy evolution and discovery'
    )

    # Run modes
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Run in continuous loop (for development)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Print current GA status and exit'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test mode: smaller population (8 vs 30) for faster iteration'
    )

    # Parameter optimization options
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=None,
        help='Generations per strategy (default: 3 normal, 2 quick mode)'
    )
    parser.add_argument(
        '--strategies', '-s',
        nargs='+',
        default=None,
        help='Specific strategies to evolve (default: all evolvable)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='RUN_ID',
        help='Resume a previously interrupted run by its run_id'
    )

    # Data scope options (controls backtest speed)
    parser.add_argument(
        '--data-scope',
        choices=['quick', 'default', 'weekend', 'full'],
        default='default',
        help='Data scope preset: quick (50 sym, 5yr), default (150 sym, 10yr), weekend (200 sym, 15yr), full (no limits)'
    )
    parser.add_argument(
        '--max-symbols',
        type=int,
        default=None,
        help='Max symbols to use (overrides preset, None=unlimited)'
    )
    parser.add_argument(
        '--max-years',
        type=int,
        default=None,
        help='Max years of history (overrides preset, None=unlimited)'
    )

    # Strategy discovery options
    parser.add_argument(
        '--discovery',
        action='store_true',
        help='Enable GP-based strategy discovery'
    )
    parser.add_argument(
        '--discovery-only',
        action='store_true',
        help='Run only strategy discovery (skip parameter optimization)'
    )
    parser.add_argument(
        '--discovery-hours',
        type=float,
        default=None,
        help='Hours to run strategy discovery (default: 2.0)'
    )
    parser.add_argument(
        '--discovery-generations',
        type=int,
        default=None,
        help='Generations for strategy discovery (overrides --discovery-hours)'
    )
    parser.add_argument(
        '--discovery-population',
        type=int,
        default=None,
        help='Population size for strategy discovery (default: 50)'
    )

    # Adaptive GA options
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Enable adaptive GA with regime-matched testing'
    )
    parser.add_argument(
        '--adaptive-only',
        action='store_true',
        help='Run only adaptive GA (skip parameter optimization and discovery)'
    )
    parser.add_argument(
        '--rapid-first',
        action='store_true',
        help='Run rapid generations on short periods before full testing'
    )
    parser.add_argument(
        '--adaptive-generations',
        type=int,
        default=None,
        help='Generations for adaptive GA (overrides config)'
    )
    parser.add_argument(
        '--adaptive-population',
        type=int,
        default=None,
        help='Total population for adaptive GA (default: 60)'
    )
    parser.add_argument(
        '--adaptive-islands',
        type=int,
        default=None,
        help='Number of islands for adaptive GA (default: 4)'
    )

    # Intervention options
    parser.add_argument(
        '--intervention',
        action='store_true',
        help='Enable human intervention checkpoints'
    )
    parser.add_argument(
        '--intervention-mode',
        type=str,
        choices=['autonomous', 'notify_only', 'review_recommended', 'approval_required'],
        default=None,
        help='Intervention mode (default: review_recommended)'
    )
    parser.add_argument(
        '--no-intervention',
        action='store_true',
        help='Disable intervention even if enabled by default'
    )

    args = parser.parse_args()

    # Status mode
    if args.status:
        print_status()
        return 0

    # Resume mode - look up run info from database
    if args.resume:
        db = get_db()
        run_info = db.fetchone(
            "research",
            "SELECT strategies_evolved, planned_generations, status FROM ga_runs WHERE run_id = ?",
            (args.resume,)
        )
        if not run_info:
            logger.error(f"Run {args.resume} not found in database")
            return 1
        if run_info['status'] not in ('interrupted', 'paused', 'abandoned'):
            logger.warning(f"Run {args.resume} has status '{run_info['status']}' - resuming anyway")

        # Parse strategies from the interrupted run
        strategies_str = run_info['strategies_evolved'] or ''
        if strategies_str.startswith('['):
            import json as json_mod
            try:
                resume_strategies = json_mod.loads(strategies_str)
            except (json_mod.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse strategies from run {args.resume}: {e}")
                resume_strategies = []
        else:
            resume_strategies = [s.strip() for s in strategies_str.split(',') if s.strip()]

        if resume_strategies:
            args.strategies = resume_strategies
            logger.info(f"Resuming run {args.resume} with strategies: {resume_strategies}")

        # Update run status back to 'running'
        db.execute(
            "research",
            "UPDATE ga_runs SET status = 'running', end_time = NULL, errors = NULL WHERE run_id = ?",
            (args.resume,)
        )

    # Filter strategies
    # Handle special case: --strategies all means run all evolvable strategies
    if args.strategies and args.strategies == ['all']:
        strategies = EVOLVABLE_STRATEGIES
    else:
        strategies = args.strategies or EVOLVABLE_STRATEGIES
    strategies = [s for s in strategies if s in STRATEGY_PARAMS]

    if not strategies and not args.discovery_only:
        logger.error("No valid strategies to evolve")
        return 1

    # Build discovery config
    discovery_config = DISCOVERY_CONFIG.copy()
    if args.discovery_population:
        discovery_config['population_size'] = args.discovery_population
    if args.discovery_hours:
        discovery_config['hours'] = args.discovery_hours
    if args.discovery_generations:
        discovery_config['generations_per_session'] = args.discovery_generations

    # Enable discovery if requested
    enable_discovery = args.discovery or args.discovery_only

    # Build adaptive GA config
    adaptive_config = ADAPTIVE_GA_CONFIG.copy()
    if args.adaptive_population:
        adaptive_config['total_population'] = args.adaptive_population
    if args.adaptive_islands:
        adaptive_config['n_islands'] = args.adaptive_islands
    if args.adaptive_generations:
        adaptive_config['generations_per_session'] = args.adaptive_generations
    if args.rapid_first:
        adaptive_config['use_rapid_testing'] = True

    # Enable adaptive GA if requested
    enable_adaptive = args.adaptive or args.adaptive_only

    # Validate adaptive GA availability
    if enable_adaptive and not ADAPTIVE_GA_AVAILABLE:
        logger.error("Adaptive GA requested but not available. Check imports.")
        return 1

    # Determine intervention settings
    enable_intervention = args.intervention and not args.no_intervention
    intervention_mode = args.intervention_mode

    # Validate intervention availability
    if enable_intervention and not INTERVENTION_AVAILABLE:
        logger.warning("Intervention requested but not available. Running without intervention.")
        enable_intervention = False

    # Build data scope config from CLI args
    data_scope = args.data_scope if args.data_scope != 'full' else None
    data_scope_config = None

    # Override preset with explicit values
    if args.max_symbols is not None or args.max_years is not None:
        # Start from preset and override
        if data_scope == 'quick':
            data_scope_config = QUICK_DATA_SCOPE.copy()
        elif data_scope == 'weekend':
            data_scope_config = WEEKEND_DATA_SCOPE.copy()
        else:
            data_scope_config = DATA_SCOPE_CONFIG.copy()

        if args.max_symbols is not None:
            data_scope_config['max_symbols'] = args.max_symbols if args.max_symbols > 0 else None
        if args.max_years is not None:
            data_scope_config['max_years'] = args.max_years if args.max_years > 0 else None

    # Create engine
    engine = NightlyResearchEngine(
        strategies=strategies,
        quick_mode=args.quick,
        enable_discovery=enable_discovery,
        discovery_config=discovery_config,
        enable_adaptive=enable_adaptive,
        adaptive_config=adaptive_config,
        rapid_first=args.rapid_first,
        enable_intervention=enable_intervention,
        intervention_mode=intervention_mode,
        data_scope=data_scope,
        data_scope_config=data_scope_config
    )

    # Determine generations (CLI overrides config)
    generations = args.generations
    if generations is None:
        generations = engine.ga_config.generations

    # Determine what phases to skip based on mode
    skip_param_opt = args.discovery_only or args.adaptive_only
    skip_discovery = args.adaptive_only

    if args.loop:
        # Continuous loop mode (development)
        logger.info("Running in continuous loop mode (Ctrl+C to stop)")

        while True:
            try:
                # Run research
                result = engine.run_nightly(
                    generations_per_strategy=generations,
                    skip_param_optimization=skip_param_opt,
                    skip_discovery=skip_discovery,
                    discovery_hours=args.discovery_hours,
                    discovery_generations=args.discovery_generations,
                    adaptive_generations=args.adaptive_generations
                )

                if not result.get('success'):
                    logger.error("Run failed, sleeping before retry...")
                    time.sleep(300)  # 5 min on failure
                    continue

                # Sleep until next run (every 6 hours for dev)
                sleep_hours = 6
                logger.info(f"\nSleeping for {sleep_hours} hours...")
                time.sleep(sleep_hours * 3600)

            except KeyboardInterrupt:
                logger.info("\nShutdown requested")
                if engine.run_id:
                    engine.db.interrupt_ga_run(engine.run_id, "User interrupted (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                if engine.run_id:
                    engine.db.interrupt_ga_run(engine.run_id, f"Exception: {str(e)[:200]}")
                time.sleep(300)
    else:
        # Single run mode (for cron/systemd)
        try:
            result = engine.run_nightly(
                generations_per_strategy=generations,
                skip_param_optimization=skip_param_opt,
                skip_discovery=skip_discovery,
                discovery_hours=args.discovery_hours,
                discovery_generations=args.discovery_generations,
                adaptive_generations=args.adaptive_generations,
                resume_run_id=args.resume
            )
            return 0 if result.get('success') else 1
        except KeyboardInterrupt:
            logger.warning("Run interrupted by user (Ctrl+C)")
            if engine.run_id:
                engine.db.interrupt_ga_run(engine.run_id, "User interrupted (Ctrl+C)")
            return 130  # Standard exit code for Ctrl+C
        except Exception as e:
            logger.error(f"Run failed with exception: {e}", exc_info=True)
            if engine.run_id:
                engine.db.interrupt_ga_run(engine.run_id, f"Exception: {str(e)[:200]}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
