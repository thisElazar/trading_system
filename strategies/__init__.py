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

# Simple indicator strategies (for visualization/exploration and GP primitives)
from strategies.indicators import EMACrossoverStrategy, RSIReversalStrategy, MACDSignalStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'LongOnlyStrategy',
    # Indicator strategies
    'EMACrossoverStrategy',
    'RSIReversalStrategy',
    'MACDSignalStrategy',
    # Production strategies
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
    # Indicator strategies (single-symbol, always generate trades)
    'ema_crossover': EMACrossoverStrategy,
    'rsi_reversal': RSIReversalStrategy,
    'macd_signal': MACDSignalStrategy,
    # Production strategies
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

# Strategy metadata - describes capabilities and requirements
# single_symbol: Can generate trades with just one symbol
# category: 'portfolio' (needs universe), 'single' (one symbol), 'indicator' (simple technical)
# description: Human-readable description
STRATEGY_METADATA = {
    # Indicator strategies - always generate trades on any symbol
    'ema_crossover': {
        'single_symbol': True,
        'category': 'indicator',
        'description': 'Buy when 9 EMA crosses above 30 EMA (tunable)',
    },
    'rsi_reversal': {
        'single_symbol': True,
        'category': 'indicator',
        'description': 'Buy oversold (RSI<30), sell overbought (RSI>70)',
    },
    'macd_signal': {
        'single_symbol': True,
        'category': 'indicator',
        'description': 'Trade MACD/signal line crossovers (12/26/9)',
    },
    # Production strategies
    'vol_managed_momentum': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Ranks stocks by 12-month momentum, vol-scaled positions',
    },
    'vol_managed_momentum_v1': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Legacy 6-month momentum formation',
    },
    'mean_reversion': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Buys bottom 25% losers within each sector',
    },
    'vix_regime_rotation': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'VIX-based regime switching with stock ranking',
    },
    'gap_fill': {
        'single_symbol': True,
        'category': 'single',
        'description': 'Trades gap momentum (continuation after gaps)',
    },
    'pairs_trading': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Statistical arbitrage on correlated pairs',
    },
    'relative_volume_breakout': {
        'single_symbol': True,
        'category': 'single',
        'description': 'Breakouts on unusual volume spikes',
    },
    'sector_rotation': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Rotates between sectors based on momentum',
    },
    'quality_smallcap_value': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Quality screens on small cap value stocks',
    },
    'factor_momentum': {
        'single_symbol': False,
        'category': 'portfolio',
        'description': 'Multi-factor ranking (value, quality, momentum)',
    },
}


def get_strategy_metadata(name: str = None) -> dict:
    """Get metadata for a strategy or all strategies."""
    if name:
        return STRATEGY_METADATA.get(name, {
            'single_symbol': False,
            'category': 'unknown',
            'description': 'No description available',
        })
    return STRATEGY_METADATA


def get_single_symbol_strategies() -> list:
    """Get list of strategies that work with single symbols."""
    return [name for name, meta in STRATEGY_METADATA.items() if meta.get('single_symbol', False)]


def get_portfolio_strategies() -> list:
    """Get list of strategies that require a universe of stocks."""
    return [name for name, meta in STRATEGY_METADATA.items() if not meta.get('single_symbol', False)]

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
