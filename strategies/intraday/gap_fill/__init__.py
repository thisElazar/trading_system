"""
Gap Fill Intraday Strategy
==========================
Intraday strategy that fades overnight gaps within the 0.15%-0.60% range.

Research shows these gaps fill 59-61% of the time within 2 hours.

Usage:
    from strategies.intraday.gap_fill import GapFillStrategy, GapFillConfig

    # With default config
    strategy = GapFillStrategy()

    # With custom config
    config = GapFillConfig(
        symbols=['SPY', 'QQQ', 'IWM'],
        min_gap_pct=0.20,
        fill_threshold=0.80
    )
    strategy = GapFillStrategy(config=config)

    # Using factory function
    strategy = create_gap_fill_strategy(
        symbols=['SPY'],
        portfolio_value=50000.0
    )
"""

from .config import GapFillConfig, DEFAULT_CONFIG
from .detector import Gap, GapDetector
from .strategy import GapFillStrategy, create_gap_fill_strategy, get_previous_close

__all__ = [
    # Main strategy
    'GapFillStrategy',
    'create_gap_fill_strategy',
    
    # Configuration
    'GapFillConfig',
    'DEFAULT_CONFIG',
    
    # Gap detection
    'Gap',
    'GapDetector',
    
    # Helpers
    'get_previous_close'
]
