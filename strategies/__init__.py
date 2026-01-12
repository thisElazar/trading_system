"""
Trading Strategies
==================
All strategy implementations.
"""

from strategies.base import BaseStrategy, Signal, SignalType, LongOnlyStrategy
from strategies.vol_managed_momentum import VolManagedMomentumStrategy as VolManagedMomentumV1
from strategies.vol_managed_momentum_v2 import VolManagedMomentumV2

# V2 is now the default (research-aligned: 12-month formation, strategy-level vol scaling)
VolManagedMomentumStrategy = VolManagedMomentumV2
from strategies.mean_reversion import MeanReversionStrategy
from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
from strategies.gap_fill import GapFillStrategy
from strategies.pairs_trading import PairsTradingStrategy, PairsAnalyzer, PairsBacktester
from strategies.relative_volume_breakout import RelativeVolumeBreakout, RVBreakoutBacktester
from strategies.sector_rotation import SectorRotationStrategy, SectorRotationBacktester
from strategies.quality_small_cap_value import QualitySmallCapValueStrategy
from strategies.factor_momentum import FactorMomentumStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'LongOnlyStrategy',
    'VolManagedMomentumStrategy',  # Alias for V2 (default)
    'VolManagedMomentumV1',        # Original 6-month formation
    'VolManagedMomentumV2',        # Research-aligned 12-month formation
    'MeanReversionStrategy',
    'VIXRegimeRotationStrategy',
    'GapFillStrategy',
    'PairsTradingStrategy',
    'PairsAnalyzer',
    'PairsBacktester',
    'RelativeVolumeBreakout',
    'RVBreakoutBacktester',
    'SectorRotationStrategy',
    'SectorRotationBacktester',
    'QualitySmallCapValueStrategy',
    'FactorMomentumStrategy',
    # GP strategy loading
    'load_live_gp_strategies',
    'get_all_active_strategies',
]

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'vol_managed_momentum': VolManagedMomentumV2,      # Default: research-aligned
    'vol_managed_momentum_v1': VolManagedMomentumV1,  # Legacy: 6-month formation
    'mean_reversion': MeanReversionStrategy,
    'vix_regime_rotation': VIXRegimeRotationStrategy,
    'gap_fill': GapFillStrategy,
    'pairs_trading': PairsTradingStrategy,
    'relative_volume_breakout': RelativeVolumeBreakout,
    'sector_rotation': SectorRotationStrategy,
    'quality_smallcap_value': QualitySmallCapValueStrategy,
    'factor_momentum': FactorMomentumStrategy,
}

def get_strategy(name: str) -> BaseStrategy:
    """Get strategy instance by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]()


def load_live_gp_strategies() -> list:
    """
    Load LIVE GP-evolved strategies from the promotion pipeline.

    These strategies were discovered by genetic programming, validated through
    walk-forward testing, paper traded, and promoted to LIVE status.

    Returns:
        List of EvolvedStrategy instances ready for signal generation

    Usage:
        from strategies import load_live_gp_strategies

        gp_strategies = load_live_gp_strategies()
        for strat in gp_strategies:
            signals = strat.generate_signals(data, current_positions)
    """
    try:
        from research.discovery.promotion_pipeline import PromotionPipeline

        pipeline = PromotionPipeline()
        return pipeline.load_live_strategies()
    except ImportError as e:
        # DEAP or other dependencies not available
        import logging
        logging.getLogger(__name__).debug(f"GP strategies not available: {e}")
        return []
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to load GP strategies: {e}")
        return []


def get_all_active_strategies() -> list:
    """
    Get all active strategies: both hardcoded and GP-evolved LIVE strategies.

    Returns:
        List of all strategy instances (BaseStrategy and EvolvedStrategy)
    """
    strategies = []

    # Add hardcoded strategies
    for name in STRATEGY_REGISTRY:
        try:
            strategies.append(STRATEGY_REGISTRY[name]())
        except Exception:
            pass

    # Add GP-evolved LIVE strategies
    gp_strategies = load_live_gp_strategies()
    strategies.extend(gp_strategies)

    return strategies
