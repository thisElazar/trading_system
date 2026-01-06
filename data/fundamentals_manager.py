"""
Fundamentals Manager
====================
High-level manager for fundamental data with advanced caching,
scheduling, and batch operations.

This module provides:
- Scheduled refresh of stale data
- Bulk operations across the universe
- Fundamentals scoring and ranking
- Integration with strategy modules
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time

import pandas as pd
import numpy as np

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DIRS, FUNDAMENTALS_REFRESH_DAYS
from data.fetchers.fundamentals import FundamentalsFetcher, get_fundamentals_bulk

logger = logging.getLogger(__name__)


class FundamentalsManager:
    """
    High-level manager for fundamental data operations.
    
    Provides caching, scheduling, and analytics on top of the
    base FundamentalsFetcher.
    """
    
    def __init__(self, cache_days: int = None):
        """
        Initialize the fundamentals manager.
        
        Args:
            cache_days: Days before refreshing cached data.
        """
        self.fetcher = FundamentalsFetcher(cache_days=cache_days)
        self.cache_days = cache_days or FUNDAMENTALS_REFRESH_DAYS
        
        # Ensure the cache summary file directory exists
        self._cache_summary_path = DIRS["fundamentals"] / "_cache_summary.parquet"
    
    def refresh_stale(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Refresh only stale (expired cache) symbols.
        
        Args:
            symbols: List of symbols to check, or None for all cached
            
        Returns:
            Dict of refreshed symbol data
        """
        if symbols is None:
            symbols = self.fetcher.get_available_symbols()
        
        stale = [s for s in symbols if not self.fetcher._is_cache_valid(s)]
        
        if not stale:
            logger.info("No stale symbols to refresh")
            return {}
        
        logger.info(f"Refreshing {len(stale)} stale symbols")
        return get_fundamentals_bulk(stale, force=True)
    
    def ensure_coverage(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Ensure all symbols have fundamental data (fetch missing).
        
        Args:
            symbols: List of symbols that should have data
            
        Returns:
            Dict of all symbol data (cached + newly fetched)
        """
        cached = set(self.fetcher.get_available_symbols())
        missing = [s for s in symbols if s not in cached]
        
        results = {}
        
        # Load all cached data
        for symbol in symbols:
            if symbol in cached:
                df = self.fetcher.load_symbol(symbol)
                if df is not None:
                    results[symbol] = df
        
        # Fetch missing
        if missing:
            logger.info(f"Fetching {len(missing)} missing symbols")
            new_data = get_fundamentals_bulk(missing)
            results.update(new_data)
        
        return results
    
    def get_quality_scores(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Calculate quality scores for symbols based on fundamentals.
        
        Quality factors (AQR-style):
        - Profitability: ROE, ROA, profit margin
        - Growth: Revenue growth, earnings growth
        - Safety: Low debt, high current ratio
        
        Args:
            symbols: List of symbols, or None for all cached
            
        Returns:
            DataFrame with quality scores
        """
        if symbols is None:
            symbols = self.fetcher.get_available_symbols()
        
        df = self.fetcher.load_symbols(symbols)
        
        if df.empty:
            return pd.DataFrame()
        
        scores = pd.DataFrame(index=df.index)
        
        # Profitability score (higher is better)
        for metric in ["roe", "roa", "profit_margin", "operating_margin"]:
            if metric in df.columns:
                scores[f"{metric}_rank"] = df[metric].rank(pct=True, na_option="keep")
        
        # Growth score (higher is better)
        for metric in ["revenue_growth", "earnings_growth"]:
            if metric in df.columns:
                scores[f"{metric}_rank"] = df[metric].rank(pct=True, na_option="keep")
        
        # Safety score (lower debt is better, higher liquidity is better)
        if "debt_to_equity" in df.columns:
            # Invert: lower debt = higher score
            scores["debt_rank"] = 1 - df["debt_to_equity"].rank(pct=True, na_option="keep")
        
        if "current_ratio" in df.columns:
            scores["liquidity_rank"] = df["current_ratio"].rank(pct=True, na_option="keep")
        
        # Composite scores
        profitability_cols = [c for c in scores.columns if any(
            m in c for m in ["roe", "roa", "profit_margin", "operating_margin"]
        )]
        growth_cols = [c for c in scores.columns if any(
            m in c for m in ["revenue_growth", "earnings_growth"]
        )]
        safety_cols = [c for c in scores.columns if any(
            m in c for m in ["debt", "liquidity"]
        )]
        
        if profitability_cols:
            scores["profitability_score"] = scores[profitability_cols].mean(axis=1)
        if growth_cols:
            scores["growth_score"] = scores[growth_cols].mean(axis=1)
        if safety_cols:
            scores["safety_score"] = scores[safety_cols].mean(axis=1)
        
        # Overall quality score
        composite_cols = [c for c in ["profitability_score", "growth_score", "safety_score"] 
                         if c in scores.columns]
        if composite_cols:
            scores["quality_score"] = scores[composite_cols].mean(axis=1)
        
        return scores
    
    def get_value_scores(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Calculate value scores for symbols based on fundamentals.
        
        Value factors:
        - Low P/E, P/B, P/S ratios
        - High dividend yield
        
        Args:
            symbols: List of symbols, or None for all cached
            
        Returns:
            DataFrame with value scores
        """
        if symbols is None:
            symbols = self.fetcher.get_available_symbols()
        
        df = self.fetcher.load_symbols(symbols)
        
        if df.empty:
            return pd.DataFrame()
        
        scores = pd.DataFrame(index=df.index)
        
        # Value metrics (lower is better for ratios)
        for metric in ["pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda"]:
            if metric in df.columns:
                # Invert: lower ratio = higher value score
                scores[f"{metric}_rank"] = 1 - df[metric].rank(pct=True, na_option="keep")
        
        # Dividend yield (higher is better)
        if "dividend_yield" in df.columns:
            scores["dividend_rank"] = df["dividend_yield"].rank(pct=True, na_option="keep")
        
        # Composite value score
        value_cols = [c for c in scores.columns if "rank" in c]
        if value_cols:
            scores["value_score"] = scores[value_cols].mean(axis=1)
        
        return scores
    
    def get_ranked_symbols(self, symbols: List[str] = None,
                           factor: str = "quality",
                           top_n: int = None) -> List[Tuple[str, float]]:
        """
        Get symbols ranked by a factor score.
        
        Args:
            symbols: List of symbols, or None for all cached
            factor: "quality" or "value"
            top_n: Return only top N symbols (None for all)
            
        Returns:
            List of (symbol, score) tuples, sorted by score descending
        """
        if factor == "quality":
            scores = self.get_quality_scores(symbols)
            score_col = "quality_score"
        elif factor == "value":
            scores = self.get_value_scores(symbols)
            score_col = "value_score"
        else:
            raise ValueError(f"Unknown factor: {factor}")
        
        if scores.empty or score_col not in scores.columns:
            return []
        
        # Sort by score descending
        sorted_scores = scores[score_col].dropna().sort_values(ascending=False)
        
        if top_n:
            sorted_scores = sorted_scores.head(top_n)
        
        return list(sorted_scores.items())
    
    def get_sector_breakdown(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get sector and industry breakdown of symbols.
        
        Args:
            symbols: List of symbols, or None for all cached
            
        Returns:
            DataFrame with sector/industry counts
        """
        if symbols is None:
            symbols = self.fetcher.get_available_symbols()
        
        df = self.fetcher.load_symbols(symbols)
        
        if df.empty:
            return pd.DataFrame()
        
        if "sector" not in df.columns or "industry" not in df.columns:
            return pd.DataFrame()
        
        # Sector counts
        sector_counts = df["sector"].value_counts().rename("count")
        sector_pct = (sector_counts / len(df) * 100).rename("pct")
        
        return pd.DataFrame({"count": sector_counts, "pct": sector_pct})
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the fundamentals cache.
        
        Returns:
            Dict with cache statistics
        """
        symbols = self.fetcher.get_available_symbols()
        
        if not symbols:
            return {
                "total_symbols": 0,
                "valid_symbols": 0,
                "stale_symbols": 0,
                "oldest_days": None,
                "newest_days": None,
            }
        
        ages = []
        for symbol in symbols:
            age = self.fetcher.get_cache_age(symbol)
            if age is not None:
                ages.append(age)
        
        valid_count = sum(1 for a in ages if a < self.cache_days)
        stale_count = len(ages) - valid_count
        
        return {
            "total_symbols": len(symbols),
            "valid_symbols": valid_count,
            "stale_symbols": stale_count,
            "oldest_days": max(ages) if ages else None,
            "newest_days": min(ages) if ages else None,
            "avg_age_days": sum(ages) / len(ages) if ages else None,
        }
    
    def cleanup(self, max_age_days: int = None) -> int:
        """
        Clean up old cache files.
        
        Args:
            max_age_days: Remove files older than this (default: 2x cache_days)
            
        Returns:
            Number of files removed
        """
        return self.fetcher.cleanup_old_data(days=max_age_days)


# Convenience functions

def refresh_fundamentals(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Refresh stale fundamentals data."""
    manager = FundamentalsManager()
    return manager.refresh_stale(symbols)


def get_quality_ranked(top_n: int = 50) -> List[Tuple[str, float]]:
    """Get top symbols ranked by quality score."""
    manager = FundamentalsManager()
    return manager.get_ranked_symbols(factor="quality", top_n=top_n)


def get_value_ranked(top_n: int = 50) -> List[Tuple[str, float]]:
    """Get top symbols ranked by value score."""
    manager = FundamentalsManager()
    return manager.get_ranked_symbols(factor="value", top_n=top_n)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fundamentals Manager")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--refresh", action="store_true", help="Refresh stale data")
    parser.add_argument("--quality", type=int, default=0, help="Show top N by quality")
    parser.add_argument("--value", type=int, default=0, help="Show top N by value")
    parser.add_argument("--sectors", action="store_true", help="Show sector breakdown")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old files")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    manager = FundamentalsManager()
    
    if args.stats:
        stats = manager.get_cache_stats()
        print("\nCache Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.refresh:
        results = manager.refresh_stale()
        print(f"\nRefreshed {len(results)} symbols")
    
    elif args.quality > 0:
        ranked = manager.get_ranked_symbols(factor="quality", top_n=args.quality)
        print(f"\nTop {args.quality} by Quality Score:")
        for symbol, score in ranked:
            print(f"  {symbol}: {score:.3f}")
    
    elif args.value > 0:
        ranked = manager.get_ranked_symbols(factor="value", top_n=args.value)
        print(f"\nTop {args.value} by Value Score:")
        for symbol, score in ranked:
            print(f"  {symbol}: {score:.3f}")
    
    elif args.sectors:
        sectors = manager.get_sector_breakdown()
        if sectors.empty:
            print("\nNo sector data available")
        else:
            print("\nSector Breakdown:")
            print(sectors.to_string())
    
    elif args.cleanup:
        removed = manager.cleanup()
        print(f"\nRemoved {removed} old files")
    
    else:
        # Default: show stats
        stats = manager.get_cache_stats()
        print("\nFundamentals Cache Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
