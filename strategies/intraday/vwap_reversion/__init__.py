"""
VWAP Mean Reversion Strategy
============================
Trades price deviations from VWAP expecting reversion to the mean.
"""

from .strategy import VWAPReversionStrategy
from .config import VWAPReversionConfig

__all__ = ['VWAPReversionStrategy', 'VWAPReversionConfig']
