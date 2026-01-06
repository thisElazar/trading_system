"""
Opening Range Breakout (ORB) Strategy
=====================================
Trades breakouts from the first N minutes' high/low range.
"""

from .strategy import ORBStrategy
from .config import ORBConfig

__all__ = ['ORBStrategy', 'ORBConfig']
