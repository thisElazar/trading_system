"""
VIX Data Fetcher
================
Fetches VIX data and provides regime classification.
Uses VIXY or ^VIX depending on availability.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import time

import pandas as pd
import numpy as np

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    DIRS, VIX_REGIMES, get_vix_regime
)
from utils.timezone import normalize_timestamp, normalize_dataframe, now_naive

logger = logging.getLogger(__name__)

# VIX proxies in order of preference
VIX_SYMBOLS = ["VIXY", "VXX", "UVXY"]


class VIXFetcher:
    """Fetches and manages VIX data."""
    
    def __init__(self):
        self.data_dir = DIRS["vix"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.data_dir / "vix.parquet"
        self.client = None
        self._current_vix: Optional[float] = None
        self._current_regime: Optional[str] = None
        self._last_update: Optional[datetime] = None
    
    def _ensure_client(self):
        """Ensure Alpaca client is initialized."""
        if self.client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self.client = StockHistoricalDataClient(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY
            )
    
    def fetch_vix_proxy(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch VIX proxy data (VIXY, VXX, etc).
        
        Note: Alpaca doesn't have ^VIX directly, so we use ETF proxies
        and scale them to approximate VIX levels.
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with VIX-like data
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        self._ensure_client()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for symbol in VIX_SYMBOLS:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.client.get_stock_bars(request)
                
                if bars.df.empty:
                    logger.debug(f"{symbol}: No data")
                    continue
                
                df = bars.df.reset_index()
                if 'symbol' in df.columns:
                    df = df.drop(columns=['symbol'])
                
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index)
                
                # Calculate VIX approximation from proxy
                # VIXY and others track VIX futures, not spot VIX
                # This is an approximation - returns are more important than levels
                df['vix_proxy'] = df['close']
                df['vix_returns'] = df['close'].pct_change()
                
                # Estimate "VIX-like" level using 20-day realized volatility
                # Annualized vol * 100 approximates VIX
                df['realized_vol'] = df['vix_returns'].rolling(20).std() * np.sqrt(252) * 100
                
                # Use a blend: proxy price trend + realized vol
                # This is crude but captures regime changes
                df['vix_estimate'] = (
                    df['realized_vol'].fillna(20) * 0.7 +
                    df['vix_proxy'].pct_change().rolling(5).mean().fillna(0) * 100 + 20
                ).clip(10, 80)
                
                logger.info(f"Fetched VIX proxy from {symbol}: {len(df)} bars")
                
                # Save
                df.to_parquet(self.parquet_path)
                
                return df
                
            except Exception as e:
                logger.warning(f"{symbol}: Failed - {e}")
                continue
        
        logger.error("All VIX proxies failed")
        return pd.DataFrame()
    
    def fetch_from_yahoo(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch actual VIX data from Yahoo Finance as fallback.
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with VIX data
        """
        try:
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            
            if vix.empty:
                logger.warning("Yahoo VIX download returned empty")
                return pd.DataFrame()
            
            # Handle MultiIndex columns from newer yfinance
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = [c[0].lower() for c in vix.columns]
            else:
                vix.columns = [c.lower() for c in vix.columns]
            
            vix['vix_estimate'] = vix['close']
            
            # Save
            vix.to_parquet(self.parquet_path)
            logger.info(f"Fetched VIX from Yahoo: {len(vix)} bars")
            
            return vix
            
        except ImportError:
            logger.warning("yfinance not installed - pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Yahoo VIX fetch failed: {e}")
            return pd.DataFrame()
    
    def load(self) -> Optional[pd.DataFrame]:
        """Load cached VIX data."""
        if not self.parquet_path.exists():
            return None
        
        try:
            return pd.read_parquet(self.parquet_path)
        except Exception as e:
            logger.error(f"Failed to load VIX data: {e}")
            return None
    
    def get_current_vix(self, use_cache: bool = True) -> float:
        """
        Get current VIX level.
        
        Args:
            use_cache: If True, use cached value if recent
            
        Returns:
            Current VIX estimate
        """
        # Return cache if recent (within 30 min)
        if (use_cache and self._current_vix is not None and 
            self._last_update is not None and
            (datetime.now() - self._last_update).seconds < 1800):
            return self._current_vix
        
        # Try to load from cache
        df = self.load()
        
        # If no cache or stale, refetch
        if df is None or df.empty:
            df = self.fetch_vix_proxy(days=30)
        else:
            # Check if data is stale (> 1 day old)
            last_date = df.index.max()
            if isinstance(last_date, pd.Timestamp):
                # Normalize timezone using centralized utility
                last_date = normalize_timestamp(last_date).to_pydatetime()

            if (datetime.now() - last_date).days > 1:
                df = self.fetch_vix_proxy(days=30)
        
        if df is None or df.empty or 'vix_estimate' not in df.columns:
            logger.warning("Could not get VIX, using default 20")
            self._current_vix = 20.0
        else:
            self._current_vix = float(df['vix_estimate'].iloc[-1])
        
        self._last_update = datetime.now()
        self._current_regime = get_vix_regime(self._current_vix)
        
        return self._current_vix
    
    def get_current_regime(self) -> str:
        """
        Get current VIX regime.
        
        Returns:
            Regime string: 'low', 'normal', 'high', or 'extreme'
        """
        if self._current_regime is None:
            self.get_current_vix()
        return self._current_regime
    
    def get_vix_history(self, days: int = 60) -> pd.DataFrame:
        """
        Get VIX history with regime classification.
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with VIX and regime columns
        """
        df = self.load()
        
        if df is None or df.empty:
            df = self.fetch_vix_proxy(days=days)
        
        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize timezone using centralized utility
        df = normalize_dataframe(df)

        # Filter to requested days
        cutoff = now_naive() - timedelta(days=days)
        df = df[df.index >= cutoff]
        
        # Add regime classification
        if 'vix_estimate' in df.columns:
            df['regime'] = df['vix_estimate'].apply(get_vix_regime)
        
        return df
    
    def get_regime_changes(self, days: int = 60) -> pd.DataFrame:
        """
        Get history of regime changes.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with regime change events
        """
        df = self.get_vix_history(days=days)
        
        if df.empty or 'regime' not in df.columns:
            return pd.DataFrame()
        
        # Find regime changes
        df['prev_regime'] = df['regime'].shift(1)
        changes = df[df['regime'] != df['prev_regime']].copy()
        changes = changes[changes['prev_regime'].notna()]
        
        return changes[['vix_estimate', 'prev_regime', 'regime']]
    
    def get_regime_stats(self, days: int = 252) -> dict:
        """
        Get statistics about regime distribution.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with regime statistics
        """
        df = self.get_vix_history(days=days)
        
        if df.empty or 'regime' not in df.columns:
            return {}
        
        regime_counts = df['regime'].value_counts()
        total_days = len(df)
        
        stats = {
            'total_days': total_days,
            'current_vix': float(df['vix_estimate'].iloc[-1]) if 'vix_estimate' in df.columns else None,
            'current_regime': df['regime'].iloc[-1],
            'avg_vix': float(df['vix_estimate'].mean()) if 'vix_estimate' in df.columns else None,
            'regimes': {}
        }
        
        for regime in ['low', 'normal', 'high', 'extreme']:
            count = regime_counts.get(regime, 0)
            stats['regimes'][regime] = {
                'days': count,
                'pct': count / total_days * 100 if total_days > 0 else 0
            }
        
        return stats


# Convenience functions

def get_current_vix() -> float:
    """Get current VIX level."""
    fetcher = VIXFetcher()
    return fetcher.get_current_vix()


def get_current_regime() -> str:
    """Get current VIX regime."""
    fetcher = VIXFetcher()
    return fetcher.get_current_regime()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = VIXFetcher()
    
    print("\n" + "=" * 50)
    print("VIX Data Fetcher")
    print("=" * 50)
    
    # Get current VIX
    vix = fetcher.get_current_vix()
    regime = fetcher.get_current_regime()
    
    print(f"\nCurrent VIX: {vix:.2f}")
    print(f"Current Regime: {regime.upper()}")
    
    # Get stats
    stats = fetcher.get_regime_stats(days=252)
    
    if stats:
        print(f"\nLast 252 days:")
        print(f"  Average VIX: {stats['avg_vix']:.2f}")
        print(f"\n  Regime Distribution:")
        for regime, data in stats['regimes'].items():
            print(f"    {regime.upper():8s}: {data['days']:3d} days ({data['pct']:.1f}%)")
    
    # Get recent regime changes
    changes = fetcher.get_regime_changes(days=60)
    if not changes.empty:
        print(f"\nRecent Regime Changes:")
        for idx, row in changes.tail(5).iterrows():
            print(f"  {idx.date()}: {row['prev_regime'].upper()} → {row['regime'].upper()} (VIX: {row['vix_estimate']:.2f})")
