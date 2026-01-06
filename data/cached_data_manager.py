"""
Cached Data Manager
===================
Fast data access using parquet files with optional live updates.

Replaces slow per-symbol API calls with:
1. Bulk loading from parquet files (downloaded by universe_downloader.py)
2. Optional daily incremental updates
3. In-memory caching for repeated access during a session

Usage:
    from data.cached_data_manager import CachedDataManager, CachedWatchlistManager
    
    data_mgr = CachedDataManager()
    data_mgr.load_all()  # Load entire universe in ~2 seconds
    
    # Get data for a symbol
    df = data_mgr.get_bars("AAPL")
    
    # Fast watchlist building
    watchlist_mgr = CachedWatchlistManager(data_mgr)
    watchlist = watchlist_mgr.build_watchlist()
"""

import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import ta

# Import from config
try:
    from config import (
        DIRS, ALPACA_API_KEY, ALPACA_SECRET_KEY,
        MARKET_CAP_TIERS, TRANSACTION_COSTS_BPS,
        MIN_STOCK_PRICE, MAX_STOCK_PRICE
    )
except ImportError:
    # Fallback
    DIRS = {"daily": Path("./data/historical/daily")}
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    MARKET_CAP_TIERS = {"large": 10e9, "mid": 2e9, "small": 300e6, "micro": 0}
    TRANSACTION_COSTS_BPS = {"large": 50, "mid": 150, "small": 300, "micro": 500}
    MIN_STOCK_PRICE = 10
    MAX_STOCK_PRICE = 500

logger = logging.getLogger(__name__)

def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast DataFrame dtypes to reduce memory usage."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        # Check if values fit in int32
        if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
            df[col] = df[col].astype('int32')
    return df


# Optional: for live updates
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class CachedDataManager:
    """
    Manages market data from parquet files with optional live updates.

    Key features:
    - Bulk loads all symbols at once (fast)
    - In-memory cache for repeated access
    - Optional daily updates from Alpaca
    - Automatic indicator calculation
    - Supports unified data loading (Yahoo + Alpaca merged)
    """

    def __init__(self, data_dir: Path = None, use_unified_loader: bool = True):
        """
        Args:
            data_dir: Directory containing parquet files (default from config)
            use_unified_loader: Use UnifiedDataLoader to merge Yahoo + Alpaca data
        """
        self.data_dir = Path(data_dir) if data_dir else DIRS["daily"]
        self.cache: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}
        self.last_load_time: Optional[datetime] = None
        self.use_unified_loader = use_unified_loader

        # Initialize unified loader if enabled
        self._unified_loader = None
        if use_unified_loader:
            try:
                from data.unified_data_loader import UnifiedDataLoader
                self._unified_loader = UnifiedDataLoader()
                logger.info("Using UnifiedDataLoader (Yahoo + Alpaca merged)")
            except ImportError:
                logger.warning("UnifiedDataLoader not available, using legacy loader")
                self.use_unified_loader = False

        # Alpaca client for live updates
        self.live_client = None
        if ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_AVAILABLE:
            try:
                self.live_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca client: {e}")

        # VIX cache
        self.current_vix: Optional[float] = None
        self.vix_updated: Optional[datetime] = None

    def clear_cache(self, symbols: List[str] = None):
        """
        Clear cached data to free memory.

        Args:
            symbols: Specific symbols to clear. If None, clears entire cache.
        """
        if symbols is None:
            self.cache.clear()
            self.metadata.clear()
            logger.debug("Cleared entire data cache")
        else:
            for symbol in symbols:
                self.cache.pop(symbol, None)
                self.metadata.pop(symbol, None)
            logger.debug(f"Cleared {len(symbols)} symbols from cache")

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with cached data"""
        # Use unified loader if available
        if self._unified_loader:
            return self._unified_loader.get_available_daily_symbols()

        # Legacy: scan data directory
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return []

        symbols = []
        for f in self.data_dir.glob("*.parquet"):
            symbol = f.stem
            if not symbol.endswith("_1min"):  # Skip intraday files
                symbols.append(symbol)

        return sorted(symbols)
    
    def load_all(self, symbols: List[str] = None, add_indicators: bool = True) -> int:
        """
        Load all symbols into memory cache.
        
        Args:
            symbols: Optional list of symbols to load (default: all available)
            add_indicators: Whether to calculate technical indicators
            
        Returns:
            Number of symbols loaded
        """
        start_time = time.time()
        
        if symbols is None:
            symbols = self.get_available_symbols()
        
        if not symbols:
            logger.warning("No symbols to load")
            return 0
        
        loaded = 0
        failed = 0
        
        for symbol in symbols:
            try:
                df = self._load_symbol(symbol, add_indicators)
                if df is not None and len(df) >= 20:
                    self.cache[symbol] = df
                    self._calculate_metadata(symbol, df)
                    loaded += 1
                else:
                    failed += 1
            except Exception as e:
                logger.debug(f"Failed to load {symbol}: {e}")
                failed += 1
        
        self.last_load_time = datetime.now()
        elapsed = time.time() - start_time
        
        logger.info(f"Loaded {loaded} symbols in {elapsed:.2f}s ({loaded/max(elapsed,0.01):.0f} symbols/sec)")
        if failed > 0:
            logger.debug(f"Failed to load {failed} symbols")
        
        return loaded
    
    def _load_symbol(self, symbol: str, add_indicators: bool = True) -> Optional[pd.DataFrame]:
        """Load a single symbol from parquet (uses unified loader if available)"""
        # Use unified loader if available (merges Yahoo + Alpaca)
        if self._unified_loader:
            df = self._unified_loader.load_daily(symbol)
            if df.empty:
                return None
        else:
            # Legacy: load from single directory
            filepath = self.data_dir / f"{symbol}.parquet"

            if not filepath.exists():
                return None

            df = pd.read_parquet(filepath)

            # Normalize column names
            df.columns = df.columns.str.lower()

            # Handle index vs column for timestamp
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                # Try to find a datetime column
                for col in df.columns:
                    if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower():
                        df = df.rename(columns={col: 'timestamp'})
                        break

            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)

        # Add indicators
        if add_indicators and len(df) >= 20:
            df = self._add_indicators(df)

        return df
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        if len(df) < 20:
            return df
        
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close']
            ).average_true_range()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            # Avoid division by zero/NaN - use safe division
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, float('nan'))
            
            # Price momentum
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['pct_change_1d'] = df['close'].pct_change()
            df['pct_change_5d'] = df['close'].pct_change(5)
            df['pct_change_20d'] = df['close'].pct_change(20)
            
            # Support/Resistance
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            
            # Momentum (12-1 for research-backed momentum)
            if len(df) >= 252:
                df['momentum_12m'] = df['close'].pct_change(252)
                df['momentum_12_1'] = df['close'].shift(21).pct_change(231)
            
        except Exception as e:
            logger.debug(f"Indicator calculation error: {e}")
        
        return df
    
    def _calculate_metadata(self, symbol: str, df: pd.DataFrame):
        """Calculate and store metadata for a symbol"""
        if df is None or df.empty or len(df) < 1:
            return

        latest = df.iloc[-1]

        # Basic metrics - with safe access
        price = float(latest.get('close', 0) or 0)
        if price <= 0:
            return  # Skip symbols with invalid price

        avg_volume = df['volume'].tail(20).mean()
        if pd.isna(avg_volume) or avg_volume <= 0:
            avg_volume = 0
        dollar_volume = price * avg_volume

        # Volatility (ATR as % of price) - safe division
        atr = latest.get('atr', None)
        if atr is None or pd.isna(atr):
            atr = df['close'].tail(20).std() if len(df) >= 20 else 0
        volatility = atr / price if price > 0 else 0
        
        # Market cap tier estimate based on dollar volume
        tier = 'micro'
        market_cap = 200_000_000
        
        if dollar_volume > 100_000_000:
            tier = 'large'
            market_cap = 15_000_000_000
        elif dollar_volume > 20_000_000:
            tier = 'mid'
            market_cap = 5_000_000_000
        elif dollar_volume > 5_000_000:
            tier = 'small'
            market_cap = 1_000_000_000
        
        # Store metadata
        self.metadata[symbol] = {
            'price': price,
            'avg_volume': avg_volume,
            'dollar_volume': dollar_volume,
            'volatility': volatility,
            'atr': atr if not pd.isna(atr) else 0,
            'market_cap': market_cap,
            'tier': tier,
            'bars': len(df),
            'last_date': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None,
        }
    
    def get_bars(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """Get bars for a symbol."""
        if use_cache and symbol in self.cache:
            return self.cache[symbol].copy()
        
        df = self._load_symbol(symbol, add_indicators=True)
        
        if df is not None:
            self.cache[symbol] = df
            self._calculate_metadata(symbol, df)
            return df.copy()
        
        return pd.DataFrame()
    
    def get_bars_batch(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols"""
        result = {}
        for symbol in symbols:
            df = self.get_bars(symbol)
            if not df.empty:
                result[symbol] = df
        return result
    
    def get_metadata(self, symbol: str) -> dict:
        """Get metadata for a symbol"""
        if symbol not in self.metadata:
            df = self.get_bars(symbol)
            if df.empty:
                return {}
        return self.metadata.get(symbol, {})
    
    def get_all_metadata(self) -> Dict[str, dict]:
        """Get metadata for all cached symbols"""
        return self.metadata.copy()
    
    def get_vix(self, force_refresh: bool = False) -> float:
        """Get current VIX level"""
        # Return cached if fresh (use total_seconds() not .seconds)
        if (not force_refresh and
            self.current_vix is not None and
            self.vix_updated is not None and
            (datetime.now() - self.vix_updated).total_seconds() < 3600):
            return self.current_vix
        
        # Try VIX parquet file first
        vix_path = DIRS.get("vix", self.data_dir.parent / "vix") / "vix.parquet"
        if vix_path.exists():
            try:
                vix_df = pd.read_parquet(vix_path)
                if not vix_df.empty:
                    self.current_vix = float(vix_df['close'].iloc[-1])
                    self.vix_updated = datetime.now()
                    return self.current_vix
            except Exception as e:
                logger.debug(f"Failed to load VIX from parquet: {e}")
        
        # Try Alpaca
        if self.live_client:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols="VIXY",
                    timeframe=TimeFrame.Day,
                    start=datetime.now() - timedelta(days=5),
                    end=datetime.now()
                )
                bars = self.live_client.get_stock_bars(request)
                if not bars.df.empty:
                    vixy_price = bars.df['close'].iloc[-1]
                    self.current_vix = vixy_price * 1.5  # VIXY proxy
                    self.vix_updated = datetime.now()
                    return self.current_vix
            except Exception as e:
                logger.debug(f"VIX fetch failed: {e}")
        
        # Fallback
        if self.current_vix is None:
            self.current_vix = 18.0
        return self.current_vix
    
    def update_daily(self, symbols: List[str] = None) -> int:
        """Fetch latest daily bar for all symbols and update cache."""
        if not self.live_client:
            logger.warning("No Alpaca client for live updates")
            return 0
        
        if symbols is None:
            symbols = list(self.cache.keys())
        
        if not symbols:
            return 0
        
        logger.info(f"Updating {len(symbols)} symbols...")
        updated = 0
        batch_size = 50
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=datetime.now() - timedelta(days=5),
                    end=datetime.now()
                )
                
                bars = self.live_client.get_stock_bars(request)
                if bars.df.empty:
                    continue
                
                df = bars.df.reset_index()
                
                for symbol in batch:
                    symbol_df = df[df['symbol'] == symbol]
                    if symbol_df.empty or symbol not in self.cache:
                        continue
                    
                    existing = self.cache[symbol]
                    
                    # Find new bars
                    if 'timestamp' in existing.columns:
                        last_ts = existing['timestamp'].max()
                        symbol_df = symbol_df[symbol_df['timestamp'] > last_ts]
                    
                    if not symbol_df.empty:
                        combined = pd.concat([existing, symbol_df.drop(columns=['symbol'])])
                        combined = combined.sort_values('timestamp').reset_index(drop=True)
                        combined = self._add_indicators(combined)
                        self.cache[symbol] = combined
                        self._calculate_metadata(symbol, combined)
                        updated += 1
                
            except Exception as e:
                logger.error(f"Batch update failed: {e}")
            
            time.sleep(0.3)
        
        logger.info(f"Updated {updated} symbols")
        return updated


class CachedWatchlistManager:
    """
    Fast watchlist building using cached data.
    
    Original: ~8000 API calls, 40+ minutes
    This: Load from parquet, instant
    """
    
    # Strategy-tier mapping from research
    STRATEGY_TIERS = {
        'gap_fill': ['large'],
        'bb_reversion': ['large', 'mid'],
        'volume_divergence': ['mid', 'large'],
        'momentum_breakout': ['mid', 'large'],
        'rsi_reversion': ['mid', 'large'],
        'macd_divergence': ['mid', 'large'],
        'volume_spike': ['mid', 'large'],
        'support_resistance': ['large'],
        'vol_managed_momentum': ['large', 'mid'],
        'vix_regime_rotation': ['large'],
        'pairs_trading': ['large'],
    }
    
    def __init__(self, data_manager: CachedDataManager, database=None):
        self.data_manager = data_manager
        self.database = database
    
    def build_watchlist(self,
                       min_price: float = None,
                       max_price: float = None,
                       min_volume: float = 500_000,
                       min_volatility: float = 0.02,
                       max_volatility: float = 0.10,
                       exclude_micro_cap: bool = True,
                       max_symbols: int = 200) -> List[dict]:
        """Build watchlist from cached data."""
        min_price = min_price or MIN_STOCK_PRICE
        max_price = max_price or MAX_STOCK_PRICE
        
        start_time = time.time()
        
        # Ensure data is loaded
        if not self.data_manager.cache:
            self.data_manager.load_all()
        
        metadata = self.data_manager.get_all_metadata()
        
        if not metadata:
            logger.warning("No metadata available")
            return []
        
        candidates = []
        
        for symbol, meta in metadata.items():
            price = meta.get('price', 0)
            volume = meta.get('avg_volume', 0)
            volatility = meta.get('volatility', 0)
            tier = meta.get('tier', 'unknown')
            
            # Apply filters
            if not (min_price <= price <= max_price):
                continue
            if volume < min_volume:
                continue
            if not (min_volatility <= volatility <= max_volatility):
                continue
            if exclude_micro_cap and tier == 'micro':
                continue
            
            candidates.append({
                'symbol': symbol,
                'price': price,
                'avg_volume': volume,
                'volatility': volatility,
                'dollar_volume': meta.get('dollar_volume', 0),
                'market_cap': meta.get('market_cap', 0),
                'market_cap_tier': tier,
                'atr': meta.get('atr', 0),
            })
        
        # Sort by volatility (more = more opportunity)
        candidates.sort(key=lambda x: x['volatility'], reverse=True)
        result = candidates[:max_symbols]
        
        elapsed = time.time() - start_time
        logger.info(f"Built watchlist: {len(result)} from {len(metadata)} in {elapsed:.3f}s")
        
        return result
    
    def build_tiered_watchlist(self, max_per_tier: int = 50) -> Dict[str, List[dict]]:
        """Build watchlist organized by market cap tier."""
        all_candidates = self.build_watchlist(max_symbols=10000)
        
        tiered = {'large': [], 'mid': [], 'small': []}
        
        for candidate in all_candidates:
            tier = candidate['market_cap_tier']
            if tier in tiered and len(tiered[tier]) < max_per_tier:
                tiered[tier].append(candidate)
        
        for tier, symbols in tiered.items():
            logger.info(f"  {tier.upper()}: {len(symbols)} symbols")
        
        return tiered
    
    def get_symbols_for_strategy(self, strategy_name: str, max_symbols: int = 100) -> List[str]:
        """Get symbols appropriate for a specific strategy."""
        target_tiers = self.STRATEGY_TIERS.get(strategy_name, ['large', 'mid'])
        tiered = self.build_tiered_watchlist(max_per_tier=max_symbols)
        
        symbols = []
        for tier in target_tiers:
            symbols.extend([s['symbol'] for s in tiered.get(tier, [])])
        
        return symbols[:max_symbols]
    
    def get_watchlist_with_metadata(self) -> Dict[str, dict]:
        """Get watchlist with full metadata (compatible with existing code)."""
        watchlist = self.build_watchlist()
        return {
            w['symbol']: {
                'tier': w['market_cap_tier'],
                'volume': w['avg_volume'],
                'volatility': w['volatility']
            } for w in watchlist
        }


# ============================================================================
# MAIN - Demo/Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n📂 Cached Data Manager Demo")
    print("=" * 50)
    
    data_mgr = CachedDataManager()
    
    # Load all data
    loaded = data_mgr.load_all()
    
    if loaded == 0:
        print("❌ No data found. Run: python scripts/universe_downloader.py")
        sys.exit(1)
    
    print(f"\n✅ Loaded {loaded} symbols")
    
    # Show sample
    print("\n📊 Sample Metadata:")
    for symbol in list(data_mgr.cache.keys())[:5]:
        meta = data_mgr.get_metadata(symbol)
        print(f"  {symbol}: ${meta['price']:.2f}, vol={meta['volatility']:.2%}, tier={meta['tier']}")
    
    # VIX
    vix = data_mgr.get_vix()
    print(f"\n📈 VIX: {vix:.1f}")
    
    # Watchlist
    print("\n📋 Building Watchlist...")
    watchlist_mgr = CachedWatchlistManager(data_mgr)
    watchlist = watchlist_mgr.build_watchlist(max_symbols=10)
    
    print(f"\nTop 10 by volatility:")
    for item in watchlist:
        print(f"  {item['symbol']:6s} ${item['price']:8.2f}  vol={item['volatility']:.2%}  tier={item['market_cap_tier']}")
    
    print("\n✅ Done!")
