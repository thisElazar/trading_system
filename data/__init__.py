"""
Data Module
===========
Data fetching, caching, storage, and indicators.

Supports multiple data sources:
- Yahoo Finance daily (extended history back to 1990s)
- Alpaca daily (2016+)
- Alpaca 1-minute intraday (2016+)
- Real-time streaming via Alpaca WebSocket
"""

from data.cached_data_manager import CachedDataManager
from data.unified_data_loader import UnifiedDataLoader, load_daily, load_intraday, get_data_summary
from data.stream_handler import MarketDataStream, create_market_stream
from data.stock_characteristics import (
    StockCharacterizer,
    StockCharacteristics,
    MarketCapTier,
    LiquidityTier,
    VolatilityRegime,
    SectorType,
    STRATEGY_REQUIREMENTS,
    characterize_universe,
)

__all__ = [
    'CachedDataManager',
    'UnifiedDataLoader',
    'load_daily',
    'load_intraday',
    'get_data_summary',
    'MarketDataStream',
    'create_market_stream',
    # Stock Characterization
    'StockCharacterizer',
    'StockCharacteristics',
    'MarketCapTier',
    'LiquidityTier',
    'VolatilityRegime',
    'SectorType',
    'STRATEGY_REQUIREMENTS',
    'characterize_universe',
]
