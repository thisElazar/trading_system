"""
Regime Matching Engine
======================
Matches current market conditions to historical periods for targeted testing.

Integrates with:
- RegimeDetector for current market state
- MarketPeriodLibrary for historical periods
- RapidBacktester for period-specific testing
- AdaptiveGAOptimizer for regime-aware evolution

Key capabilities:
- Real-time regime fingerprinting
- Historical period similarity scoring
- Dynamic test period selection
- Regime transition detection
- Strategy performance prediction based on regime

Usage:
    from research.genetic.regime_matching import RegimeMatchingEngine

    engine = RegimeMatchingEngine()

    # Get current regime fingerprint
    fingerprint = engine.get_current_fingerprint()

    # Find matching historical periods
    matches = engine.find_matching_periods(n=5)

    # Get recommended test periods for GA
    test_periods = engine.get_ga_test_periods()
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path
import sys

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .market_periods import MarketPeriodLibrary, MarketPeriod, PeriodType

logger = logging.getLogger(__name__)


@dataclass
class RegimeFingerprint:
    """
    Multi-dimensional fingerprint of market conditions.

    Used for matching current conditions to historical periods.
    """
    # Primary indicators
    vix_level: float              # Current VIX
    vix_percentile: float         # VIX percentile vs history (0-100)
    vix_trend: float              # VIX change over 20 days (%)

    # Trend indicators
    trend_direction: float        # -1 (bear) to 1 (bull)
    trend_strength: float         # 0-1, higher = stronger trend
    momentum_breadth: float       # % of stocks with positive momentum

    # Volatility indicators
    realized_vol: float           # 20-day realized volatility
    vol_regime: str               # "low", "normal", "high", "extreme"
    vol_trend: float              # Increasing (+) or decreasing (-)

    # Correlation indicators
    correlation_level: float      # 0-1, average cross-asset correlation
    correlation_trend: float      # Rising or falling correlations

    # Sector indicators
    sector_leadership: str        # "cyclical", "defensive", "mixed"
    sector_dispersion: float      # High dispersion = stock picking matters

    # Sentiment/flow indicators
    credit_spread_z: float        # Z-score of credit spreads
    term_structure: str           # "contango" or "backwardation"

    # Composite regime
    overall_regime: str           # "risk_on", "risk_off", "transition", "crisis"
    regime_confidence: float      # 0-1

    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert fingerprint to numerical vector for similarity comparison."""
        return np.array([
            self.vix_level / 80,  # Normalize VIX
            self.vix_percentile / 100,
            self.vix_trend / 50 + 0.5,  # Center around 0.5
            self.trend_direction / 2 + 0.5,
            self.trend_strength,
            self.momentum_breadth / 100,
            min(1, self.realized_vol / 40),  # Cap at 40% vol
            self._vol_regime_to_num(),
            self.vol_trend / 50 + 0.5,
            self.correlation_level,
            self.correlation_trend / 0.5 + 0.5,
            self._sector_to_num(),
            self.sector_dispersion,
            self.credit_spread_z / 4 + 0.5,  # Z-score normalization
            1 if self.term_structure == "contango" else 0,
            self._regime_to_num(),
        ])

    def _vol_regime_to_num(self) -> float:
        mapping = {"low": 0.2, "normal": 0.5, "high": 0.75, "extreme": 1.0}
        return mapping.get(self.vol_regime, 0.5)

    def _sector_to_num(self) -> float:
        mapping = {"cyclical": 0.8, "defensive": 0.2, "mixed": 0.5}
        return mapping.get(self.sector_leadership, 0.5)

    def _regime_to_num(self) -> float:
        mapping = {"risk_on": 0.9, "transition": 0.5, "risk_off": 0.3, "crisis": 0.1}
        return mapping.get(self.overall_regime, 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vix_level': self.vix_level,
            'vix_percentile': self.vix_percentile,
            'vix_trend': self.vix_trend,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'momentum_breadth': self.momentum_breadth,
            'realized_vol': self.realized_vol,
            'vol_regime': self.vol_regime,
            'vol_trend': self.vol_trend,
            'correlation_level': self.correlation_level,
            'correlation_trend': self.correlation_trend,
            'sector_leadership': self.sector_leadership,
            'sector_dispersion': self.sector_dispersion,
            'credit_spread_z': self.credit_spread_z,
            'term_structure': self.term_structure,
            'overall_regime': self.overall_regime,
            'regime_confidence': self.regime_confidence,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class PeriodMatch:
    """A matched historical period with similarity score."""
    period: MarketPeriod
    similarity: float           # 0-1, higher = more similar
    dimension_scores: Dict[str, float]  # Similarity per dimension
    relevance_boost: float      # Boost for recency or importance
    final_score: float          # similarity * relevance_boost

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period_name': self.period.name,
            'period_type': self.period.period_type.value,
            'similarity': self.similarity,
            'dimension_scores': self.dimension_scores,
            'relevance_boost': self.relevance_boost,
            'final_score': self.final_score,
        }


class RegimeMatchingEngine:
    """
    Engine for matching current market conditions to historical periods.

    Provides:
    - Current regime fingerprinting
    - Historical period matching
    - Dynamic test period selection for GA
    - Regime transition detection
    """

    # Weights for different dimensions in similarity calculation
    DIMENSION_WEIGHTS = {
        'vix': 0.20,
        'trend': 0.20,
        'volatility': 0.15,
        'correlation': 0.15,
        'sector': 0.10,
        'credit': 0.10,
        'overall': 0.10,
    }

    def __init__(
        self,
        period_library: MarketPeriodLibrary = None,
        data_manager: Any = None
    ):
        """
        Initialize the regime matching engine.

        Args:
            period_library: MarketPeriodLibrary instance
            data_manager: CachedDataManager for market data
        """
        self.library = period_library or MarketPeriodLibrary()
        self.data_manager = data_manager

        # Cache
        self._current_fingerprint: Optional[RegimeFingerprint] = None
        self._fingerprint_time: Optional[datetime] = None
        self._fingerprint_cache_seconds = 300  # 5 minute cache

        # Historical VIX for percentile calculation
        self._vix_history: Optional[pd.Series] = None

        logger.info("RegimeMatchingEngine initialized")

    def _ensure_data_manager(self):
        """Lazy load data manager."""
        if self.data_manager is None:
            try:
                from data.cached_data_manager import CachedDataManager
                self.data_manager = CachedDataManager()
            except ImportError:
                logger.warning("CachedDataManager not available")
                return False
        return True

    def _get_vix_percentile(self, current_vix: float) -> float:
        """Calculate current VIX percentile vs historical."""
        if self._vix_history is None:
            # Use approximate historical VIX distribution
            # Based on 1990-2024 VIX data
            historical_percentiles = {
                10: 12.0,
                20: 13.5,
                30: 15.0,
                40: 16.5,
                50: 18.0,
                60: 20.0,
                70: 23.0,
                80: 27.0,
                90: 35.0,
                95: 45.0,
            }

            # Interpolate
            for pct, vix in sorted(historical_percentiles.items()):
                if current_vix <= vix:
                    return float(pct)

            return 99.0

        # Calculate from actual history
        return float((self._vix_history < current_vix).mean() * 100)

    def get_current_fingerprint(self, force_refresh: bool = False) -> RegimeFingerprint:
        """
        Generate fingerprint of current market conditions.

        Args:
            force_refresh: Ignore cache and recalculate

        Returns:
            RegimeFingerprint
        """
        # Check cache
        if (not force_refresh and
            self._current_fingerprint is not None and
            self._fingerprint_time is not None and
            (datetime.now() - self._fingerprint_time).total_seconds() < self._fingerprint_cache_seconds):
            return self._current_fingerprint

        # Initialize with defaults
        fingerprint_data = {
            'vix_level': 15.0,
            'vix_percentile': 50.0,
            'vix_trend': 0.0,
            'trend_direction': 0.0,
            'trend_strength': 0.5,
            'momentum_breadth': 50.0,
            'realized_vol': 15.0,
            'vol_regime': 'normal',
            'vol_trend': 0.0,
            'correlation_level': 0.5,
            'correlation_trend': 0.0,
            'sector_leadership': 'mixed',
            'sector_dispersion': 0.5,
            'credit_spread_z': 0.0,
            'term_structure': 'contango',
            'overall_regime': 'transition',
            'regime_confidence': 0.5,
        }

        if self._ensure_data_manager():
            try:
                fingerprint_data = self._calculate_fingerprint()
            except Exception as e:
                logger.warning(f"Error calculating fingerprint: {e}")

        self._current_fingerprint = RegimeFingerprint(**fingerprint_data)
        self._fingerprint_time = datetime.now()

        return self._current_fingerprint

    def _calculate_fingerprint(self) -> Dict[str, Any]:
        """Calculate full fingerprint from market data."""
        result = {}

        # Get VIX
        vix = self.data_manager.get_vix()
        result['vix_level'] = vix if vix else 15.0
        result['vix_percentile'] = self._get_vix_percentile(result['vix_level'])

        # VIX trend (20-day change)
        vix_data = self.data_manager.get_bars('VIX')
        if vix_data is not None and len(vix_data) >= 20:
            vix_20d_ago = float(vix_data['close'].iloc[-20]) if len(vix_data) >= 20 else result['vix_level']
            result['vix_trend'] = (result['vix_level'] / vix_20d_ago - 1) * 100
        else:
            result['vix_trend'] = 0.0

        # Market trend (SPY)
        spy_data = self.data_manager.get_bars('SPY')
        if spy_data is not None and len(spy_data) >= 50:
            current = float(spy_data['close'].iloc[-1])
            ma_50 = spy_data['close'].tail(50).mean()
            ma_20 = spy_data['close'].tail(20).mean()

            # Trend direction
            if current > ma_50 * 1.02:
                result['trend_direction'] = min(1.0, (current / ma_50 - 1) * 5)
            elif current < ma_50 * 0.98:
                result['trend_direction'] = max(-1.0, (current / ma_50 - 1) * 5)
            else:
                result['trend_direction'] = (current / ma_50 - 1) * 5

            # Trend strength (consistency)
            returns = spy_data['close'].pct_change().tail(20)
            if len(returns) > 0:
                trend_consistency = abs(returns.mean()) / (returns.std() + 0.0001)
                result['trend_strength'] = min(1.0, trend_consistency * 2)
            else:
                result['trend_strength'] = 0.5
        else:
            result['trend_direction'] = 0.0
            result['trend_strength'] = 0.5

        # Momentum breadth
        if hasattr(self.data_manager, 'cache') and self.data_manager.cache:
            advancing = 0
            total = 0
            for symbol, df in self.data_manager.cache.items():
                if len(df) >= 20:
                    try:
                        current = float(df['close'].iloc[-1])
                        past = float(df['close'].iloc[-20])
                        total += 1
                        if current > past:
                            advancing += 1
                    except Exception:
                        continue

            result['momentum_breadth'] = (advancing / total * 100) if total > 0 else 50.0
        else:
            result['momentum_breadth'] = 50.0

        # Realized volatility
        if spy_data is not None and len(spy_data) >= 20:
            returns = spy_data['close'].pct_change().tail(20).dropna()
            result['realized_vol'] = float(returns.std() * np.sqrt(252) * 100)
        else:
            result['realized_vol'] = 15.0

        # Vol regime
        if result['realized_vol'] < 12:
            result['vol_regime'] = 'low'
        elif result['realized_vol'] < 20:
            result['vol_regime'] = 'normal'
        elif result['realized_vol'] < 30:
            result['vol_regime'] = 'high'
        else:
            result['vol_regime'] = 'extreme'

        # Vol trend
        if spy_data is not None and len(spy_data) >= 40:
            vol_recent = spy_data['close'].pct_change().tail(10).std() * np.sqrt(252) * 100
            vol_prior = spy_data['close'].pct_change().iloc[-40:-20].std() * np.sqrt(252) * 100
            result['vol_trend'] = (vol_recent - vol_prior)
        else:
            result['vol_trend'] = 0.0

        # Correlation (simplified - use sector ETFs)
        result['correlation_level'] = 0.5  # Default
        result['correlation_trend'] = 0.0

        # Sector leadership
        result['sector_leadership'] = 'mixed'
        result['sector_dispersion'] = 0.5

        # Credit spread (simplified)
        result['credit_spread_z'] = 0.0

        # Term structure (from VIX if available)
        if result['vix_level'] > result['vix_percentile'] * 0.3:
            result['term_structure'] = 'backwardation'
        else:
            result['term_structure'] = 'contango'

        # Overall regime
        if result['vix_level'] > 35:
            result['overall_regime'] = 'crisis'
            result['regime_confidence'] = 0.9
        elif result['vix_level'] > 25 or result['trend_direction'] < -0.5:
            result['overall_regime'] = 'risk_off'
            result['regime_confidence'] = 0.7
        elif result['vix_level'] < 15 and result['trend_direction'] > 0.3:
            result['overall_regime'] = 'risk_on'
            result['regime_confidence'] = 0.8
        else:
            result['overall_regime'] = 'transition'
            result['regime_confidence'] = 0.5

        return result

    def _period_to_fingerprint(self, period: MarketPeriod) -> np.ndarray:
        """Convert a MarketPeriod's characteristics to a comparable vector."""
        # Map period characteristics to fingerprint dimensions
        vix_normalized = period.avg_vix / 80
        vix_percentile = self._get_vix_percentile(period.avg_vix) / 100

        # Vol regime from VIX
        if period.avg_vix < 12:
            vol_regime = 0.2
        elif period.avg_vix < 20:
            vol_regime = 0.5
        elif period.avg_vix < 30:
            vol_regime = 0.75
        else:
            vol_regime = 1.0

        # Sector
        sector_map = {"cyclical": 0.8, "defensive": 0.2, "mixed": 0.5}
        sector_val = sector_map.get(period.sector_leadership, 0.5)

        # Overall regime from period type
        type_regime_map = {
            PeriodType.CRISIS: 0.1,
            PeriodType.BEAR_MARKET: 0.25,
            PeriodType.HIGH_VOL: 0.35,
            PeriodType.SIDEWAYS: 0.5,
            PeriodType.SECTOR_ROTATION: 0.55,
            PeriodType.RECOVERY: 0.7,
            PeriodType.BULL_RUN: 0.85,
            PeriodType.LOW_VOL: 0.9,
        }
        regime_val = type_regime_map.get(period.period_type, 0.5)

        return np.array([
            vix_normalized,
            vix_percentile,
            0.5,  # VIX trend (unknown for historical)
            period.trend_direction / 2 + 0.5,
            abs(period.trend_direction),  # Trend strength
            0.5,  # Momentum breadth (unknown)
            period.avg_daily_range / 5,  # Normalized daily range
            vol_regime,
            0.5,  # Vol trend (unknown)
            period.correlation_regime,
            0.5,  # Correlation trend
            sector_val,
            0.5,  # Sector dispersion
            0.5,  # Credit spread (unknown)
            1 if period.avg_vix < 20 else 0,  # Term structure proxy
            regime_val,
        ])

    def calculate_similarity(
        self,
        fingerprint: RegimeFingerprint,
        period: MarketPeriod
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate similarity between current fingerprint and historical period.

        Returns:
            (overall_similarity, dimension_scores)
        """
        current_vector = fingerprint.to_vector()
        period_vector = self._period_to_fingerprint(period)

        # Calculate dimension-wise similarities
        dimension_scores = {}

        # VIX dimensions (indices 0-2)
        vix_sim = 1 - np.mean(np.abs(current_vector[0:3] - period_vector[0:3]))
        dimension_scores['vix'] = vix_sim

        # Trend dimensions (indices 3-5)
        trend_sim = 1 - np.mean(np.abs(current_vector[3:6] - period_vector[3:6]))
        dimension_scores['trend'] = trend_sim

        # Volatility dimensions (indices 6-8)
        vol_sim = 1 - np.mean(np.abs(current_vector[6:9] - period_vector[6:9]))
        dimension_scores['volatility'] = vol_sim

        # Correlation dimensions (indices 9-10)
        corr_sim = 1 - np.mean(np.abs(current_vector[9:11] - period_vector[9:11]))
        dimension_scores['correlation'] = corr_sim

        # Sector dimensions (indices 11-12)
        sector_sim = 1 - np.mean(np.abs(current_vector[11:13] - period_vector[11:13]))
        dimension_scores['sector'] = sector_sim

        # Credit dimensions (indices 13-14)
        credit_sim = 1 - np.mean(np.abs(current_vector[13:15] - period_vector[13:15]))
        dimension_scores['credit'] = credit_sim

        # Overall regime (index 15)
        overall_sim = 1 - abs(current_vector[15] - period_vector[15])
        dimension_scores['overall'] = overall_sim

        # Weighted combination
        overall_similarity = sum(
            dimension_scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_WEIGHTS
        )

        return overall_similarity, dimension_scores

    def find_matching_periods(
        self,
        fingerprint: RegimeFingerprint = None,
        n: int = 5,
        include_types: List[PeriodType] = None,
        exclude_types: List[PeriodType] = None,
        min_similarity: float = 0.3,
        recency_boost: bool = True
    ) -> List[PeriodMatch]:
        """
        Find historical periods most similar to current conditions.

        Args:
            fingerprint: Current fingerprint (calculates if None)
            n: Number of matches to return
            include_types: Only include these period types
            exclude_types: Exclude these period types
            min_similarity: Minimum similarity threshold
            recency_boost: Boost more recent periods

        Returns:
            List of PeriodMatch objects sorted by final score
        """
        if fingerprint is None:
            fingerprint = self.get_current_fingerprint()

        matches = []
        all_periods = self.library.get_all_periods()

        for period in all_periods:
            # Type filtering
            if include_types and period.period_type not in include_types:
                continue
            if exclude_types and period.period_type in exclude_types:
                continue

            similarity, dimension_scores = self.calculate_similarity(fingerprint, period)

            if similarity < min_similarity:
                continue

            # Calculate relevance boost
            relevance_boost = 1.0

            if recency_boost:
                # More recent periods get a boost
                years_ago = (datetime.now().year - period.end_date.year)
                relevance_boost *= max(0.8, 1.0 - years_ago * 0.02)

            # Short periods get a slight boost for rapid testing
            if period.is_short_period:
                relevance_boost *= 1.05

            # High difficulty periods are more valuable for testing
            relevance_boost *= (0.9 + period.difficulty * 0.2)

            final_score = similarity * relevance_boost

            matches.append(PeriodMatch(
                period=period,
                similarity=similarity,
                dimension_scores=dimension_scores,
                relevance_boost=relevance_boost,
                final_score=final_score,
            ))

        # Sort by final score
        matches.sort(key=lambda m: -m.final_score)

        return matches[:n]

    def get_ga_test_periods(
        self,
        fingerprint: RegimeFingerprint = None,
        n_similar: int = 3,
        n_stress: int = 2,
        n_diverse: int = 2
    ) -> Dict[str, List[MarketPeriod]]:
        """
        Get recommended periods for GA testing.

        Returns a balanced set including:
        - Periods similar to current conditions (regime-matched)
        - Stress test periods (crisis)
        - Diverse periods (robustness)

        Args:
            fingerprint: Current fingerprint
            n_similar: Number of similar periods
            n_stress: Number of stress test periods
            n_diverse: Number of diverse periods

        Returns:
            Dict with 'similar', 'stress', 'diverse' period lists
        """
        if fingerprint is None:
            fingerprint = self.get_current_fingerprint()

        result = {
            'similar': [],
            'stress': [],
            'diverse': [],
        }

        # Similar periods (regime-matched)
        similar_matches = self.find_matching_periods(
            fingerprint,
            n=n_similar,
            exclude_types=[PeriodType.CRISIS],
            min_similarity=0.4
        )
        result['similar'] = [m.period for m in similar_matches]

        # Stress test periods
        crisis_periods = self.library.get_crisis_periods()
        # Sort by difficulty and pick top n
        crisis_sorted = sorted(crisis_periods, key=lambda p: -p.difficulty)
        result['stress'] = crisis_sorted[:n_stress]

        # Diverse periods (different from current)
        # Find periods with LOW similarity to add robustness
        all_matches = self.find_matching_periods(
            fingerprint,
            n=20,
            min_similarity=0.0
        )

        # Take from the less similar end
        diverse_candidates = sorted(all_matches, key=lambda m: m.similarity)
        used_types = set()
        for match in diverse_candidates:
            if len(result['diverse']) >= n_diverse:
                break
            if match.period.period_type not in used_types:
                result['diverse'].append(match.period)
                used_types.add(match.period.period_type)

        return result

    def get_regime_transition_probability(
        self,
        fingerprint: RegimeFingerprint = None
    ) -> Dict[str, float]:
        """
        Estimate probability of transitioning to different regimes.

        Based on current conditions and historical patterns.

        Returns:
            Dict mapping regime names to transition probabilities
        """
        if fingerprint is None:
            fingerprint = self.get_current_fingerprint()

        # Base probabilities from current regime
        current = fingerprint.overall_regime

        # Historical transition matrix (approximate)
        # Rows: current state, Cols: next state
        transition_base = {
            'risk_on': {'risk_on': 0.75, 'transition': 0.15, 'risk_off': 0.08, 'crisis': 0.02},
            'transition': {'risk_on': 0.30, 'transition': 0.40, 'risk_off': 0.25, 'crisis': 0.05},
            'risk_off': {'risk_on': 0.15, 'transition': 0.25, 'risk_off': 0.50, 'crisis': 0.10},
            'crisis': {'risk_on': 0.05, 'transition': 0.20, 'risk_off': 0.40, 'crisis': 0.35},
        }

        probs = transition_base.get(current, transition_base['transition']).copy()

        # Adjust based on current conditions
        # High VIX trend increases crisis probability
        if fingerprint.vix_trend > 20:
            probs['crisis'] = min(0.4, probs['crisis'] * 1.5)
            probs['risk_on'] *= 0.5
        elif fingerprint.vix_trend < -20:
            probs['risk_on'] = min(0.6, probs['risk_on'] * 1.3)
            probs['crisis'] *= 0.5

        # High correlation increases risk_off/crisis
        if fingerprint.correlation_level > 0.8:
            probs['risk_off'] *= 1.2
            probs['crisis'] *= 1.3
            probs['risk_on'] *= 0.7

        # Strong trend reduces transition probability
        if fingerprint.trend_strength > 0.7:
            if fingerprint.trend_direction > 0:
                probs['risk_on'] *= 1.2
            else:
                probs['risk_off'] *= 1.2
            probs['transition'] *= 0.8

        # Normalize
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def get_recommended_strategy_weights(
        self,
        fingerprint: RegimeFingerprint = None,
        strategies: List[str] = None
    ) -> Dict[str, float]:
        """
        Get recommended strategy allocation weights based on regime.

        Args:
            fingerprint: Current fingerprint
            strategies: List of strategy names

        Returns:
            Dict mapping strategy names to recommended weights
        """
        if fingerprint is None:
            fingerprint = self.get_current_fingerprint()

        # Default strategies if not provided
        if strategies is None:
            strategies = [
                'momentum', 'mean_reversion', 'pairs_trading',
                'vol_managed_momentum', 'relative_volume_breakout',
                'gap_fill', 'vix_regime_rotation', 'sector_rotation'
            ]

        # Base weights by regime
        regime_weights = {
            'risk_on': {
                'momentum': 1.2,
                'vol_managed_momentum': 1.2,
                'relative_volume_breakout': 1.1,
                'gap_fill': 1.0,
                'mean_reversion': 0.8,
                'pairs_trading': 0.7,
                'vix_regime_rotation': 0.6,
                'sector_rotation': 1.0,
            },
            'transition': {
                'momentum': 0.9,
                'vol_managed_momentum': 0.9,
                'relative_volume_breakout': 0.8,
                'gap_fill': 0.9,
                'mean_reversion': 1.0,
                'pairs_trading': 1.1,
                'vix_regime_rotation': 1.0,
                'sector_rotation': 0.9,
            },
            'risk_off': {
                'momentum': 0.5,
                'vol_managed_momentum': 0.6,
                'relative_volume_breakout': 0.5,
                'gap_fill': 0.7,
                'mean_reversion': 1.2,
                'pairs_trading': 1.3,
                'vix_regime_rotation': 1.4,
                'sector_rotation': 0.8,
            },
            'crisis': {
                'momentum': 0.2,
                'vol_managed_momentum': 0.25,
                'relative_volume_breakout': 0.2,
                'gap_fill': 0.0,
                'mean_reversion': 0.5,
                'pairs_trading': 0.8,
                'vix_regime_rotation': 1.0,
                'sector_rotation': 0.3,
            },
        }

        weights = regime_weights.get(
            fingerprint.overall_regime,
            regime_weights['transition']
        ).copy()

        # Filter to requested strategies
        result = {}
        for strategy in strategies:
            result[strategy] = weights.get(strategy, 0.5)

        return result

    def print_status(self):
        """Print current regime matching status."""
        fingerprint = self.get_current_fingerprint()

        print("\n" + "=" * 60)
        print("REGIME MATCHING ENGINE STATUS")
        print("=" * 60)

        print("\nCURRENT FINGERPRINT:")
        print("-" * 40)
        print(f"  VIX: {fingerprint.vix_level:.1f} ({fingerprint.vix_percentile:.0f}th percentile)")
        print(f"  VIX Trend: {fingerprint.vix_trend:+.1f}%")
        print(f"  Trend Direction: {fingerprint.trend_direction:+.2f}")
        print(f"  Trend Strength: {fingerprint.trend_strength:.2f}")
        print(f"  Momentum Breadth: {fingerprint.momentum_breadth:.0f}%")
        print(f"  Realized Vol: {fingerprint.realized_vol:.1f}%")
        print(f"  Vol Regime: {fingerprint.vol_regime}")
        print(f"  Correlation: {fingerprint.correlation_level:.2f}")
        print(f"  Sector Leadership: {fingerprint.sector_leadership}")
        print(f"  Overall Regime: {fingerprint.overall_regime.upper()}")
        print(f"  Confidence: {fingerprint.regime_confidence:.0%}")

        print("\nMATCHING PERIODS:")
        print("-" * 40)
        matches = self.find_matching_periods(n=5)
        for match in matches:
            print(f"  {match.period.name:25s} | sim: {match.similarity:.2f} | "
                  f"score: {match.final_score:.2f} | {match.period.period_type.value}")

        print("\nGA TEST PERIODS:")
        print("-" * 40)
        test_periods = self.get_ga_test_periods()
        for category, periods in test_periods.items():
            print(f"  {category.upper()}:")
            for p in periods:
                print(f"    - {p.name}")

        print("\nTRANSITION PROBABILITIES:")
        print("-" * 40)
        probs = self.get_regime_transition_probability()
        for regime, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            print(f"  {regime:12s} [{bar:30s}] {prob:.1%}")

        print("=" * 60 + "\n")


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("REGIME MATCHING ENGINE DEMO")
    print("=" * 60)

    engine = RegimeMatchingEngine()

    # Get current fingerprint (will use defaults without data manager)
    fingerprint = engine.get_current_fingerprint()
    print(f"\nCurrent regime: {fingerprint.overall_regime}")
    print(f"VIX level: {fingerprint.vix_level:.1f}")

    # Find matching periods
    print("\nTop matching periods:")
    matches = engine.find_matching_periods(n=5)
    for match in matches:
        print(f"  {match.period.name}: {match.final_score:.2f}")

    # Get GA test periods
    print("\nGA test period recommendations:")
    test_periods = engine.get_ga_test_periods()
    for category, periods in test_periods.items():
        print(f"  {category}: {[p.name for p in periods]}")

    # Print full status
    engine.print_status()
