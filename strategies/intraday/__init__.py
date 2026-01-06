"""
Intraday Strategies Package
===========================
Contains strategies designed for intraday (same-day) trading
with real-time streaming data.

Available Strategies:
- GapFillStrategy: Fades overnight gaps expecting reversion
- VWAPReversionStrategy: Trades price deviations from VWAP
- ORBStrategy: Opening Range Breakout momentum trades
"""

from .base import (
    IntradayStrategy,
    IntradayPosition,
    IntradayTimeWindow,
    PositionSide,
    MARKET_OPEN,
    MARKET_CLOSE
)

# Strategy imports (lazy to avoid circular dependencies)
def get_gap_fill_strategy():
    from .gap_fill import GapFillStrategy, GapFillConfig
    return GapFillStrategy, GapFillConfig

def get_vwap_reversion_strategy():
    from .vwap_reversion import VWAPReversionStrategy, VWAPReversionConfig
    return VWAPReversionStrategy, VWAPReversionConfig

def get_orb_strategy():
    from .orb import ORBStrategy, ORBConfig
    return ORBStrategy, ORBConfig

__all__ = [
    # Base classes
    'IntradayStrategy',
    'IntradayPosition',
    'IntradayTimeWindow',
    'PositionSide',
    'MARKET_OPEN',
    'MARKET_CLOSE',
    # Strategy getters
    'get_gap_fill_strategy',
    'get_vwap_reversion_strategy',
    'get_orb_strategy',
]
