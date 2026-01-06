"""
Stock Characteristics Engine
============================
Characterizes stocks for strategy-specific filtering and universe management.

Research shows significant performance differences based on stock characteristics:
- Gap-fill on SPY: 2.38 Sharpe (needs high liquidity, tight spreads)
- Within-industry mean reversion: +0.821%/month (vs -0.295% cross-industry)
- Small-cap transaction costs: 2-4% annual (filter by liquidity)

This module provides:
- Market cap, liquidity, volatility classification
- Sector/industry grouping for pairs and mean reversion
- Strategy-specific eligibility filtering
- Caching for efficient repeated access

Usage:
    from data.stock_characteristics import StockCharacterizer, MarketCapTier

    characterizer = StockCharacterizer()
    characterizer.load_universe(data_dict, fundamentals_dict)

    # Get eligible symbols for a strategy
    gap_fill_universe = characterizer.get_eligible_universe('gap_fill')

    # Get characteristics for a specific stock
    char = characterizer.get_characteristics('AAPL')
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import DIRS

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MarketCapTier(Enum):
    """Market capitalization tiers."""
    MEGA = "mega"       # >$200B
    LARGE = "large"     # $10B-$200B
    MID = "mid"         # $2B-$10B
    SMALL = "small"     # $300M-$2B
    MICRO = "micro"     # <$300M


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "low"         # <15% annualized
    NORMAL = "normal"   # 15-30%
    HIGH = "high"       # 30-50%
    EXTREME = "extreme" # >50%


class LiquidityTier(Enum):
    """Liquidity classifications based on average dollar volume."""
    HIGHLY_LIQUID = "highly_liquid"   # >$50M ADV
    LIQUID = "liquid"                  # $10M-$50M ADV
    MODERATE = "moderate"              # $5M-$10M ADV
    ILLIQUID = "illiquid"             # <$5M ADV


class SectorType(Enum):
    """Sector behavior classifications."""
    CYCLICAL = "cyclical"       # XLY, XLI, XLB, XLF, XLE
    DEFENSIVE = "defensive"     # XLU, XLP, XLV
    GROWTH = "growth"           # XLK, XLC
    RATE_SENSITIVE = "rate_sensitive"  # XLRE


@dataclass
class StockCharacteristics:
    """Complete characterization of a stock for strategy matching."""
    symbol: str

    # Size characteristics
    market_cap: float = 0
    market_cap_tier: MarketCapTier = MarketCapTier.SMALL

    # Sector/Industry
    sector: Optional[str] = None
    industry: Optional[str] = None
    sector_type: Optional[SectorType] = None

    # Liquidity characteristics
    avg_daily_volume: float = 0
    avg_dollar_volume: float = 0
    liquidity_tier: LiquidityTier = LiquidityTier.MODERATE
    avg_spread_pct: float = 0.01  # Estimated from high-low

    # Volatility characteristics
    volatility_20d: float = 0.20
    volatility_60d: float = 0.20
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL

    # Risk characteristics
    beta: float = 1.0
    correlation_to_spy: float = 0.5

    # Momentum/Mean-reversion characteristics
    momentum_20d: float = 0
    momentum_60d: float = 0
    mean_reversion_strength: float = 0  # Lag-1 return autocorrelation

    # Computed eligibility flags (set by __post_init__)
    eligible_for_gap_fill: bool = False
    eligible_for_momentum: bool = False
    eligible_for_mean_reversion: bool = False
    eligible_for_pairs: bool = False
    eligible_for_breakout: bool = False

    # Metadata
    last_updated: Optional[datetime] = None
    data_quality_score: float = 1.0  # 0-1, lower if missing data

    def __post_init__(self):
        """Compute eligibility flags based on characteristics."""
        # Gap fill: needs high liquidity, low spread, large cap
        self.eligible_for_gap_fill = (
            self.liquidity_tier in [LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID] and
            self.avg_spread_pct < 0.002 and  # <0.2% spread
            self.market_cap_tier in [MarketCapTier.MEGA, MarketCapTier.LARGE] and
            self.avg_dollar_volume >= 20_000_000
        )

        # Momentum: needs adequate liquidity, avoid extreme vol
        self.eligible_for_momentum = (
            self.liquidity_tier != LiquidityTier.ILLIQUID and
            self.volatility_regime != VolatilityRegime.EXTREME and
            self.avg_dollar_volume >= 5_000_000
        )

        # Mean reversion: best in normal-high vol, needs liquidity
        self.eligible_for_mean_reversion = (
            self.liquidity_tier in [LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID,
                                    LiquidityTier.MODERATE] and
            self.volatility_regime in [VolatilityRegime.NORMAL, VolatilityRegime.HIGH] and
            self.industry is not None  # Needs industry for within-industry
        )

        # Pairs: needs industry peers and good liquidity
        self.eligible_for_pairs = (
            self.liquidity_tier in [LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID] and
            self.industry is not None and
            self.avg_dollar_volume >= 10_000_000
        )

        # Breakout: needs volatility and volume
        self.eligible_for_breakout = (
            self.liquidity_tier != LiquidityTier.ILLIQUID and
            self.volatility_regime in [VolatilityRegime.NORMAL, VolatilityRegime.HIGH] and
            self.avg_dollar_volume >= 5_000_000
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'StockCharacteristics':
        """Create from dictionary."""
        # Convert enum strings back to enums
        if 'market_cap_tier' in data and isinstance(data['market_cap_tier'], str):
            data['market_cap_tier'] = MarketCapTier(data['market_cap_tier'])
        if 'liquidity_tier' in data and isinstance(data['liquidity_tier'], str):
            data['liquidity_tier'] = LiquidityTier(data['liquidity_tier'])
        if 'volatility_regime' in data and isinstance(data['volatility_regime'], str):
            data['volatility_regime'] = VolatilityRegime(data['volatility_regime'])
        if 'sector_type' in data and isinstance(data['sector_type'], str):
            data['sector_type'] = SectorType(data['sector_type'])
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


# =============================================================================
# STRATEGY REQUIREMENTS
# =============================================================================

STRATEGY_REQUIREMENTS = {
    'gap_fill': {
        'min_dollar_volume': 20_000_000,
        'max_spread': 0.002,
        'market_cap_tiers': [MarketCapTier.MEGA, MarketCapTier.LARGE],
        'description': 'High liquidity, tight spreads for intraday mean reversion'
    },
    'pairs_trading': {
        'min_dollar_volume': 10_000_000,
        'require_industry': True,
        'min_correlation': 0.7,
        'liquidity_tiers': [LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID],
        'description': 'Liquid stocks with industry peers for cointegration'
    },
    'momentum': {
        'min_dollar_volume': 5_000_000,
        'volatility_regimes': [VolatilityRegime.LOW, VolatilityRegime.NORMAL],
        'exclude_market_cap_tiers': [MarketCapTier.MICRO],
        'description': 'Stable volatility, avoid microcaps for trend following'
    },
    'mean_reversion': {
        'min_dollar_volume': 5_000_000,
        'volatility_regimes': [VolatilityRegime.NORMAL, VolatilityRegime.HIGH],
        'require_industry': True,  # Within-industry only
        'description': 'Higher volatility for mean reversion opportunities'
    },
    'relative_volume_breakout': {
        'min_dollar_volume': 5_000_000,
        'volatility_regimes': [VolatilityRegime.NORMAL, VolatilityRegime.HIGH],
        'description': 'Volatile stocks for breakout signals'
    },
    'sector_rotation': {
        'min_dollar_volume': 50_000_000,
        'market_cap_tiers': [MarketCapTier.MEGA, MarketCapTier.LARGE],
        'description': 'Highly liquid for sector ETF rotation'
    },
    'quality_smallcap_value': {
        'min_dollar_volume': 2_000_000,
        'market_cap_tiers': [MarketCapTier.SMALL, MarketCapTier.MID],
        'description': 'Small/mid caps for quality-value factor'
    }
}


# =============================================================================
# SECTOR MAPPINGS
# =============================================================================

# GICS Sector to SectorType mapping
SECTOR_TYPE_MAP = {
    # Cyclical
    'Consumer Discretionary': SectorType.CYCLICAL,
    'Industrials': SectorType.CYCLICAL,
    'Materials': SectorType.CYCLICAL,
    'Financials': SectorType.CYCLICAL,
    'Energy': SectorType.CYCLICAL,

    # Defensive
    'Utilities': SectorType.DEFENSIVE,
    'Consumer Staples': SectorType.DEFENSIVE,
    'Healthcare': SectorType.DEFENSIVE,
    'Health Care': SectorType.DEFENSIVE,

    # Growth
    'Technology': SectorType.GROWTH,
    'Information Technology': SectorType.GROWTH,
    'Communication Services': SectorType.GROWTH,
    'Telecommunication Services': SectorType.GROWTH,

    # Rate Sensitive
    'Real Estate': SectorType.RATE_SENSITIVE,
}


# =============================================================================
# MAIN CHARACTERIZER CLASS
# =============================================================================

class StockCharacterizer:
    """
    Engine for characterizing stocks from market data.

    Provides:
    - Individual stock characterization
    - Bulk universe processing
    - Strategy-specific filtering
    - Caching for efficiency
    """

    def __init__(self, cache_path: Path = None):
        """
        Initialize the characterizer.

        Args:
            cache_path: Path to cache characteristics (default: data/reference/)
        """
        self.cache_path = cache_path or DIRS.get('reference', Path('data/reference'))
        self._characteristics: Dict[str, StockCharacteristics] = {}
        self._industry_groups: Dict[str, List[str]] = {}  # industry -> symbols
        self._sector_groups: Dict[str, List[str]] = {}    # sector -> symbols
        self._spy_returns: Optional[pd.Series] = None
        self._last_load: Optional[datetime] = None

        # Try to load cached characteristics
        self._load_cache()

    def _load_cache(self):
        """Load cached characteristics if available."""
        cache_file = self.cache_path / 'stock_characteristics.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                for symbol, char_dict in data.get('characteristics', {}).items():
                    self._characteristics[symbol] = StockCharacteristics.from_dict(char_dict)

                self._industry_groups = data.get('industry_groups', {})
                self._sector_groups = data.get('sector_groups', {})

                logger.info(f"Loaded {len(self._characteristics)} cached characteristics")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save characteristics to cache."""
        cache_file = self.cache_path / 'stock_characteristics.json'
        try:
            self.cache_path.mkdir(parents=True, exist_ok=True)

            data = {
                'characteristics': {
                    symbol: char.to_dict()
                    for symbol, char in self._characteristics.items()
                },
                'industry_groups': self._industry_groups,
                'sector_groups': self._sector_groups,
                'last_updated': datetime.now().isoformat()
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved {len(self._characteristics)} characteristics to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def load_universe(self,
                      price_data: Dict[str, pd.DataFrame],
                      fundamentals: Dict[str, dict] = None,
                      spy_data: pd.DataFrame = None,
                      force_refresh: bool = False) -> int:
        """
        Load and characterize the entire universe.

        Args:
            price_data: Dict of symbol -> OHLCV DataFrame
            fundamentals: Dict of symbol -> fundamentals dict
            spy_data: SPY price data for beta/correlation calculation
            force_refresh: If True, recompute all characteristics

        Returns:
            Number of symbols characterized
        """
        fundamentals = fundamentals or {}

        # Store SPY returns for correlation calculation
        if spy_data is not None and len(spy_data) > 20:
            self._spy_returns = spy_data['close'].pct_change().dropna()

        # Clear if force refresh
        if force_refresh:
            self._characteristics.clear()
            self._industry_groups.clear()
            self._sector_groups.clear()

        count = 0
        for symbol, df in price_data.items():
            if len(df) < 20:
                continue

            # Skip if already cached and not forcing refresh
            if symbol in self._characteristics and not force_refresh:
                count += 1
                continue

            fund = fundamentals.get(symbol, {})
            char = self.characterize_stock(symbol, df, fund)

            if char is not None:
                self._characteristics[symbol] = char
                count += 1

                # Update group mappings
                if char.industry:
                    if char.industry not in self._industry_groups:
                        self._industry_groups[char.industry] = []
                    if symbol not in self._industry_groups[char.industry]:
                        self._industry_groups[char.industry].append(symbol)

                if char.sector:
                    if char.sector not in self._sector_groups:
                        self._sector_groups[char.sector] = []
                    if symbol not in self._sector_groups[char.sector]:
                        self._sector_groups[char.sector].append(symbol)

        self._last_load = datetime.now()
        self._save_cache()

        logger.info(f"Characterized {count} stocks, "
                   f"{len(self._industry_groups)} industries, "
                   f"{len(self._sector_groups)} sectors")

        return count

    def characterize_stock(self,
                           symbol: str,
                           data: pd.DataFrame,
                           fundamentals: dict = None) -> Optional[StockCharacteristics]:
        """
        Characterize a single stock from its price data and fundamentals.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            fundamentals: Optional fundamentals dict

        Returns:
            StockCharacteristics or None if insufficient data
        """
        fundamentals = fundamentals or {}

        if len(data) < 20:
            return None

        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()

            # === VOLATILITY ===
            vol_20d = returns.tail(20).std() * np.sqrt(252)
            vol_60d = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else vol_20d
            vol_regime = self._classify_volatility(vol_20d)

            # === MARKET CAP ===
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap == 0:
                # Estimate from price and typical shares outstanding
                market_cap = data['close'].iloc[-1] * data['volume'].tail(20).mean() * 100
            mc_tier = self._classify_market_cap(market_cap)

            # === LIQUIDITY ===
            avg_volume = data['volume'].tail(20).mean()
            avg_price = data['close'].tail(20).mean()
            dollar_volume = avg_volume * avg_price
            liq_tier = self._classify_liquidity(dollar_volume)

            # === SPREAD ESTIMATE ===
            # Use high-low range as proxy for spread
            spread_pct = ((data['high'] - data['low']) / data['close']).tail(20).median()
            # Adjust: actual spread is typically 10-20% of range for liquid stocks
            spread_pct = spread_pct * 0.15

            # === SECTOR/INDUSTRY ===
            sector = fundamentals.get('sector')
            industry = fundamentals.get('industry')
            sector_type = SECTOR_TYPE_MAP.get(sector) if sector else None

            # === BETA AND CORRELATION ===
            beta = fundamentals.get('beta', 1.0)
            corr_to_spy = 0.5  # Default

            if self._spy_returns is not None and len(returns) >= 60:
                try:
                    common_idx = returns.index.intersection(self._spy_returns.index)
                    if len(common_idx) >= 20:
                        stock_ret = returns.loc[common_idx].tail(60)
                        spy_ret = self._spy_returns.loc[common_idx].tail(60)
                        corr_to_spy = stock_ret.corr(spy_ret)

                        # Calculate beta if not in fundamentals
                        if beta == 1.0:
                            cov = stock_ret.cov(spy_ret)
                            var = spy_ret.var()
                            if var > 0:
                                beta = cov / var
                except Exception as e:
                    logger.debug(f"Beta calculation failed for {symbol}: {e}")

            # === MOMENTUM ===
            if len(data) >= 20:
                momentum_20d = (data['close'].iloc[-1] / data['close'].iloc[-20]) - 1
            else:
                momentum_20d = 0

            if len(data) >= 60:
                momentum_60d = (data['close'].iloc[-1] / data['close'].iloc[-60]) - 1
            else:
                momentum_60d = momentum_20d

            # === MEAN REVERSION STRENGTH ===
            # Lag-1 autocorrelation of returns (negative = mean reverting)
            mr_strength = returns.tail(60).autocorr(lag=1) if len(returns) >= 60 else 0

            # === DATA QUALITY ===
            quality_score = 1.0
            if market_cap == 0:
                quality_score -= 0.2
            if sector is None:
                quality_score -= 0.1
            if industry is None:
                quality_score -= 0.1

            characteristics = StockCharacteristics(
                symbol=symbol,
                market_cap=market_cap,
                market_cap_tier=mc_tier,
                sector=sector,
                industry=industry,
                sector_type=sector_type,
                avg_daily_volume=avg_volume,
                avg_dollar_volume=dollar_volume,
                liquidity_tier=liq_tier,
                avg_spread_pct=spread_pct,
                volatility_20d=vol_20d,
                volatility_60d=vol_60d,
                volatility_regime=vol_regime,
                beta=beta,
                correlation_to_spy=corr_to_spy,
                momentum_20d=momentum_20d,
                momentum_60d=momentum_60d,
                mean_reversion_strength=mr_strength,
                last_updated=datetime.now(),
                data_quality_score=quality_score
            )

            return characteristics

        except Exception as e:
            logger.debug(f"Error characterizing {symbol}: {e}")
            return None

    def _classify_market_cap(self, mc: float) -> MarketCapTier:
        """Classify market cap into tiers."""
        if mc >= 200e9:
            return MarketCapTier.MEGA
        elif mc >= 10e9:
            return MarketCapTier.LARGE
        elif mc >= 2e9:
            return MarketCapTier.MID
        elif mc >= 300e6:
            return MarketCapTier.SMALL
        else:
            return MarketCapTier.MICRO

    def _classify_liquidity(self, dollar_vol: float) -> LiquidityTier:
        """Classify liquidity based on average dollar volume."""
        if dollar_vol >= 50e6:
            return LiquidityTier.HIGHLY_LIQUID
        elif dollar_vol >= 10e6:
            return LiquidityTier.LIQUID
        elif dollar_vol >= 5e6:
            return LiquidityTier.MODERATE
        else:
            return LiquidityTier.ILLIQUID

    def _classify_volatility(self, vol: float) -> VolatilityRegime:
        """Classify volatility regime."""
        if vol < 0.15:
            return VolatilityRegime.LOW
        elif vol < 0.30:
            return VolatilityRegime.NORMAL
        elif vol < 0.50:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_characteristics(self, symbol: str) -> Optional[StockCharacteristics]:
        """Get characteristics for a specific symbol."""
        return self._characteristics.get(symbol)

    def get_eligible_universe(self,
                               strategy_name: str,
                               additional_filters: Dict[str, Any] = None
                               ) -> List[str]:
        """
        Get list of symbols eligible for a specific strategy.

        Args:
            strategy_name: Name of the strategy
            additional_filters: Optional additional filter criteria

        Returns:
            List of eligible symbols
        """
        requirements = STRATEGY_REQUIREMENTS.get(strategy_name, {})
        eligible = []

        for symbol, char in self._characteristics.items():
            if self._passes_requirements(char, requirements, additional_filters):
                eligible.append(symbol)

        return eligible

    def _passes_requirements(self,
                              char: StockCharacteristics,
                              requirements: dict,
                              additional_filters: Dict[str, Any] = None) -> bool:
        """Check if a stock passes the strategy requirements."""
        # Dollar volume
        if 'min_dollar_volume' in requirements:
            if char.avg_dollar_volume < requirements['min_dollar_volume']:
                return False

        # Spread
        if 'max_spread' in requirements:
            if char.avg_spread_pct > requirements['max_spread']:
                return False

        # Market cap tiers
        if 'market_cap_tiers' in requirements:
            if char.market_cap_tier not in requirements['market_cap_tiers']:
                return False

        # Exclude market cap tiers
        if 'exclude_market_cap_tiers' in requirements:
            if char.market_cap_tier in requirements['exclude_market_cap_tiers']:
                return False

        # Volatility regimes
        if 'volatility_regimes' in requirements:
            if char.volatility_regime not in requirements['volatility_regimes']:
                return False

        # Liquidity tiers
        if 'liquidity_tiers' in requirements:
            if char.liquidity_tier not in requirements['liquidity_tiers']:
                return False

        # Industry requirement
        if requirements.get('require_industry', False):
            if not char.industry:
                return False

        # Additional filters
        if additional_filters:
            for key, value in additional_filters.items():
                char_value = getattr(char, key, None)
                if char_value is None:
                    return False
                if isinstance(value, tuple) and len(value) == 2:
                    # Range filter (min, max)
                    if not (value[0] <= char_value <= value[1]):
                        return False
                elif char_value != value:
                    return False

        return True

    def get_industry_peers(self, symbol: str, min_liquidity: bool = True) -> List[str]:
        """
        Get stocks in the same industry as the given symbol.

        Args:
            symbol: The reference symbol
            min_liquidity: If True, filter to liquid peers only

        Returns:
            List of peer symbols (excluding the input symbol)
        """
        char = self._characteristics.get(symbol)
        if char is None or char.industry is None:
            return []

        peers = self._industry_groups.get(char.industry, [])

        # Filter out the input symbol
        peers = [s for s in peers if s != symbol]

        # Optional liquidity filter
        if min_liquidity:
            peers = [
                s for s in peers
                if self._characteristics.get(s, StockCharacteristics(s)).liquidity_tier
                   in [LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID]
            ]

        return peers

    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get all stocks in a sector."""
        return self._sector_groups.get(sector, [])

    def get_sector_type_stocks(self, sector_type: SectorType) -> List[str]:
        """Get all stocks of a given sector type (cyclical, defensive, etc.)."""
        result = []
        for symbol, char in self._characteristics.items():
            if char.sector_type == sector_type:
                result.append(symbol)
        return result

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the characterized universe."""
        if not self._characteristics:
            return {'total': 0}

        chars = list(self._characteristics.values())

        # Count by tier
        mc_counts = {}
        for tier in MarketCapTier:
            mc_counts[tier.value] = sum(1 for c in chars if c.market_cap_tier == tier)

        liq_counts = {}
        for tier in LiquidityTier:
            liq_counts[tier.value] = sum(1 for c in chars if c.liquidity_tier == tier)

        vol_counts = {}
        for regime in VolatilityRegime:
            vol_counts[regime.value] = sum(1 for c in chars if c.volatility_regime == regime)

        # Eligibility counts
        eligibility = {
            'gap_fill': sum(1 for c in chars if c.eligible_for_gap_fill),
            'momentum': sum(1 for c in chars if c.eligible_for_momentum),
            'mean_reversion': sum(1 for c in chars if c.eligible_for_mean_reversion),
            'pairs': sum(1 for c in chars if c.eligible_for_pairs),
            'breakout': sum(1 for c in chars if c.eligible_for_breakout),
        }

        return {
            'total': len(chars),
            'with_sector': sum(1 for c in chars if c.sector),
            'with_industry': sum(1 for c in chars if c.industry),
            'industries': len(self._industry_groups),
            'sectors': len(self._sector_groups),
            'market_cap_distribution': mc_counts,
            'liquidity_distribution': liq_counts,
            'volatility_distribution': vol_counts,
            'strategy_eligibility': eligibility,
            'last_updated': self._last_load.isoformat() if self._last_load else None,
        }

    def print_summary(self):
        """Print a summary of the characterized universe."""
        stats = self.get_summary_stats()

        print("\n" + "=" * 60)
        print("STOCK CHARACTERIZATION SUMMARY")
        print("=" * 60)

        print(f"\nTotal Stocks: {stats['total']}")
        print(f"With Sector: {stats['with_sector']}")
        print(f"With Industry: {stats['with_industry']}")
        print(f"Unique Industries: {stats['industries']}")
        print(f"Unique Sectors: {stats['sectors']}")

        print("\n--- Market Cap Distribution ---")
        for tier, count in stats['market_cap_distribution'].items():
            pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {tier:12s}: {count:4d} ({pct:5.1f}%)")

        print("\n--- Liquidity Distribution ---")
        for tier, count in stats['liquidity_distribution'].items():
            pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {tier:15s}: {count:4d} ({pct:5.1f}%)")

        print("\n--- Strategy Eligibility ---")
        for strategy, count in stats['strategy_eligibility'].items():
            pct = count / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {strategy:20s}: {count:4d} ({pct:5.1f}%)")

        print("=" * 60 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_characterizer_from_cache() -> StockCharacterizer:
    """Create a characterizer and load from cache."""
    return StockCharacterizer()


def characterize_universe(data_manager=None) -> StockCharacterizer:
    """
    Characterize the full universe using CachedDataManager.

    Args:
        data_manager: CachedDataManager instance (creates new if None)

    Returns:
        Configured StockCharacterizer
    """
    from data.cached_data_manager import CachedDataManager
    from data.fundamentals_manager import FundamentalsManager

    dm = data_manager or CachedDataManager()
    fm = FundamentalsManager()

    # Load price data
    if not dm.cache:
        dm.load_all()

    # Get SPY data
    spy_data = dm.cache.get('SPY')

    # Load fundamentals
    symbols = list(dm.cache.keys())
    fund_df = fm.fetcher.load_symbols(symbols)

    # Convert to dict format
    fundamentals = {}
    if not fund_df.empty:
        for symbol in fund_df.index:
            row = fund_df.loc[symbol]
            fundamentals[symbol] = {
                'market_cap': row.get('market_cap', 0),
                'sector': row.get('sector'),
                'industry': row.get('industry'),
                'beta': row.get('beta', 1.0),
            }

    # Create and load characterizer
    characterizer = StockCharacterizer()
    characterizer.load_universe(dm.cache, fundamentals, spy_data)

    return characterizer


# =============================================================================
# MAIN - Demo/Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    print("\nStock Characterization Engine")
    print("=" * 60)

    # Try to load from cache first
    characterizer = StockCharacterizer()

    if characterizer._characteristics:
        print(f"\nLoaded {len(characterizer._characteristics)} characteristics from cache")
        characterizer.print_summary()
    else:
        print("\nNo cached data. Run characterize_universe() to build.")
        print("\nExample usage:")
        print("  from data.stock_characteristics import characterize_universe")
        print("  characterizer = characterize_universe()")
        print("  gap_fill_stocks = characterizer.get_eligible_universe('gap_fill')")
