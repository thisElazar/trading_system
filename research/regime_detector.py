"""
Regime Detection Engine
========================
Advanced market regime detection that goes beyond simple VIX levels.

Uses multiple signals:
- VIX level AND term structure (contango vs backwardation)
- Credit spreads proxy (HYG vs LQD)
- Market breadth (% stocks above 50-day MA)
- Momentum breadth (advancing vs declining)
- Sector rotation patterns (defensive vs cyclical leadership)
- Correlation regime (are correlations spiking?)

Integrates with the daily orchestrator and logs regime changes to performance.db.

Usage:
    from research.regime_detector import RegimeDetector, MarketRegime

    detector = RegimeDetector()
    regime = detector.detect_regime()
    confidence = detector.get_regime_confidence()
    recommendations = detector.get_strategy_recommendations(regime)
"""

import logging
import json
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from data.cached_data_manager import CachedDataManager
from config import DATABASES, VIX_REGIMES

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    RISK_ON = "risk_on"           # Bull market, low vol, risk appetite
    RISK_OFF = "risk_off"         # Defensive, high vol, flight to safety
    TRANSITION = "transition"      # Unclear, mixed signals
    CRISIS = "crisis"             # Extreme stress, correlations spike


@dataclass
class RegimeSignal:
    """Individual signal contributing to regime detection."""
    name: str
    value: float              # Raw value
    normalized: float         # Normalized to 0-1 scale (0=risk-off, 1=risk-on)
    weight: float             # Weight in final calculation
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'value': self.value,
            'normalized': self.normalized,
            'weight': self.weight,
            'description': self.description,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RegimeState:
    """Complete regime state at a point in time."""
    regime: MarketRegime
    confidence: float           # 0-1 confidence score
    composite_score: float      # Weighted average of signals (0=crisis, 1=risk-on)
    signals: List[RegimeSignal]
    timestamp: datetime = field(default_factory=datetime.now)
    previous_regime: Optional[MarketRegime] = None
    regime_changed: bool = False

    def to_dict(self) -> dict:
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'composite_score': self.composite_score,
            'signals': [s.to_dict() for s in self.signals],
            'timestamp': self.timestamp.isoformat(),
            'previous_regime': self.previous_regime.value if self.previous_regime else None,
            'regime_changed': self.regime_changed
        }


# Strategy adjustment multipliers by regime
REGIME_ADJUSTMENTS = {
    MarketRegime.RISK_ON: {
        'momentum': 1.2,
        'vol_managed_momentum': 1.2,
        'relative_volume_breakout': 1.2,
        'gap_fill': 1.0,
        'mean_reversion': 0.8,
        'pairs_trading': 0.8,
        'vix_regime_rotation': 0.5,
        'sector_rotation': 1.0,
        'quality_smallcap_value': 1.0,
        'factor_momentum': 1.0,
    },
    MarketRegime.RISK_OFF: {
        'momentum': 0.5,
        'vol_managed_momentum': 0.6,
        'relative_volume_breakout': 0.5,
        'gap_fill': 0.7,
        'mean_reversion': 1.2,
        'pairs_trading': 1.5,
        'vix_regime_rotation': 1.5,
        'sector_rotation': 0.8,
        'quality_smallcap_value': 1.2,
        'factor_momentum': 0.7,
    },
    MarketRegime.TRANSITION: {
        'momentum': 0.8,
        'vol_managed_momentum': 0.8,
        'relative_volume_breakout': 0.8,
        'gap_fill': 0.8,
        'mean_reversion': 1.0,
        'pairs_trading': 1.2,
        'vix_regime_rotation': 1.0,
        'sector_rotation': 0.9,
        'quality_smallcap_value': 1.0,
        'factor_momentum': 0.9,
    },
    MarketRegime.CRISIS: {
        'all': 0.25,  # Minimal exposure across all strategies
        'momentum': 0.25,
        'vol_managed_momentum': 0.25,
        'relative_volume_breakout': 0.25,
        'gap_fill': 0.0,  # No intraday during crisis
        'mean_reversion': 0.25,
        'pairs_trading': 0.5,  # Pairs can still work in crisis
        'vix_regime_rotation': 1.0,  # Let VIX strategy manage
        'sector_rotation': 0.25,
        'quality_smallcap_value': 0.25,
        'factor_momentum': 0.25,
    },
}

# ETF symbols for regime detection
REGIME_ETFS = {
    'vix_proxy': ['VIXY', 'VXX', 'UVXY'],
    'vix_futures': ['VXZ'],  # Mid-term VIX futures for term structure
    'credit_high_yield': ['HYG', 'JNK'],
    'credit_investment_grade': ['LQD', 'VCIT'],
    'market_index': ['SPY', 'QQQ', 'IWM'],
    'defensive': ['XLU', 'XLP', 'VNQ', 'GLD', 'TLT'],
    'cyclical': ['XLY', 'XLI', 'XLB', 'XLF', 'XLE'],
}


class RegimeDetector:
    """
    Advanced market regime detection engine.

    Combines multiple signals to determine the current market regime:
    - VIX level and term structure
    - Credit spreads
    - Market breadth
    - Sector rotation
    - Correlation regime
    """

    # Signal weights (must sum to 1.0)
    SIGNAL_WEIGHTS = {
        'vix_level': 0.20,
        'vix_term_structure': 0.15,
        'credit_spread': 0.15,
        'market_breadth': 0.15,
        'momentum_breadth': 0.10,
        'sector_rotation': 0.10,
        'correlation_regime': 0.15,
    }

    # Regime thresholds (composite score)
    REGIME_THRESHOLDS = {
        'crisis': 0.20,      # Below this = crisis
        'risk_off': 0.40,    # Below this = risk_off
        'transition': 0.60,  # Below this = transition
        # Above 0.60 = risk_on
    }

    def __init__(self, data_manager: CachedDataManager = None):
        """
        Initialize the regime detector.

        Args:
            data_manager: CachedDataManager instance (creates new if None)
        """
        self.data_manager = data_manager or CachedDataManager()
        self._db = None

        # State tracking
        self._current_state: Optional[RegimeState] = None
        self._regime_history: List[RegimeState] = []
        self._last_detection: Optional[datetime] = None

        # Cache for signal data
        self._signal_cache: Dict[str, Any] = {}
        self._cache_expiry: Optional[datetime] = None

        logger.info("RegimeDetector initialized")

    @property
    def db(self):
        """Lazy load database connection."""
        if self._db is None:
            from data.storage.db_manager import get_db
            self._db = get_db()
        return self._db

    def _get_etf_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Get ETF data from cache or data manager."""
        try:
            df = self.data_manager.get_bars(symbol)
            if df.empty:
                return None

            # Filter to requested days
            if 'timestamp' in df.columns:
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df['timestamp'] >= cutoff]
            else:
                df = df.tail(days)

            return df
        except Exception as e:
            logger.debug(f"Failed to get data for {symbol}: {e}")
            return None

    def _get_first_available_etf(self, etf_list: List[str], days: int = 60) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """Get data for the first available ETF in a list."""
        for symbol in etf_list:
            df = self._get_etf_data(symbol, days)
            if df is not None and len(df) >= 20:
                return symbol, df
        return None, None

    # =========================================================================
    # SIGNAL CALCULATION METHODS
    # =========================================================================

    def _calculate_vix_level_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate VIX level signal.

        VIX < 15: Risk-on (normalized ~0.9)
        VIX 15-25: Normal (normalized ~0.6)
        VIX 25-35: Elevated (normalized ~0.3)
        VIX > 35: Crisis (normalized ~0.1)
        """
        vix = self.data_manager.get_vix()
        if vix is None:
            return None

        # Normalize VIX to 0-1 scale (inverted - lower VIX = higher score)
        if vix < VIX_REGIMES['low']:
            normalized = 0.9
            description = f"Low VIX ({vix:.1f}) - Risk appetite"
        elif vix < VIX_REGIMES['normal']:
            # Linear interpolation between 0.6 and 0.9
            normalized = 0.6 + 0.3 * (VIX_REGIMES['normal'] - vix) / (VIX_REGIMES['normal'] - VIX_REGIMES['low'])
            description = f"Normal VIX ({vix:.1f})"
        elif vix < VIX_REGIMES['high']:
            # Linear interpolation between 0.3 and 0.6
            normalized = 0.3 + 0.3 * (VIX_REGIMES['high'] - vix) / (VIX_REGIMES['high'] - VIX_REGIMES['normal'])
            description = f"Elevated VIX ({vix:.1f}) - Caution"
        else:
            # Exponential decay below 0.3 for extreme VIX
            normalized = max(0.05, 0.3 * np.exp(-(vix - VIX_REGIMES['high']) / 20))
            description = f"Extreme VIX ({vix:.1f}) - Crisis mode"

        return RegimeSignal(
            name='vix_level',
            value=vix,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['vix_level'],
            description=description
        )

    def _calculate_vix_term_structure_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate VIX term structure signal.

        Contango (VIX < VXZ): Normal/Risk-on
        Backwardation (VIX > VXZ): Risk-off/Crisis
        """
        # Get short-term VIX proxy
        _, short_df = self._get_first_available_etf(REGIME_ETFS['vix_proxy'], days=5)
        if short_df is None or short_df.empty:
            return None

        # Get mid-term VIX futures proxy
        _, long_df = self._get_first_available_etf(REGIME_ETFS['vix_futures'], days=5)
        if long_df is None or long_df.empty:
            return None

        # Get latest prices
        short_price = float(short_df['close'].iloc[-1])
        long_price = float(long_df['close'].iloc[-1])

        # Calculate term structure (positive = contango = normal)
        term_spread_pct = (long_price - short_price) / short_price * 100

        # Normalize: contango (+5% to +10%) = risk-on, flat = neutral, backwardation = risk-off
        if term_spread_pct >= 5:
            normalized = min(0.9, 0.6 + (term_spread_pct - 5) / 15)
            description = f"Contango {term_spread_pct:.1f}% - Risk-on term structure"
        elif term_spread_pct >= 0:
            normalized = 0.5 + term_spread_pct / 10
            description = f"Mild contango {term_spread_pct:.1f}%"
        elif term_spread_pct >= -5:
            normalized = 0.5 + term_spread_pct / 10
            description = f"Mild backwardation {term_spread_pct:.1f}% - Caution"
        else:
            normalized = max(0.1, 0.3 + term_spread_pct / 20)
            description = f"Deep backwardation {term_spread_pct:.1f}% - Fear"

        return RegimeSignal(
            name='vix_term_structure',
            value=term_spread_pct,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['vix_term_structure'],
            description=description
        )

    def _calculate_credit_spread_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate credit spread signal using HYG/LQD ratio.

        Rising HYG/LQD ratio: Risk-on (investors prefer high-yield)
        Falling HYG/LQD ratio: Risk-off (flight to quality)
        """
        # Get high-yield bond ETF
        _, hyg_df = self._get_first_available_etf(REGIME_ETFS['credit_high_yield'], days=60)
        if hyg_df is None or len(hyg_df) < 20:
            return None

        # Get investment-grade bond ETF
        _, lqd_df = self._get_first_available_etf(REGIME_ETFS['credit_investment_grade'], days=60)
        if lqd_df is None or len(lqd_df) < 20:
            return None

        # Align data
        min_len = min(len(hyg_df), len(lqd_df))
        hyg_prices = hyg_df['close'].tail(min_len).values
        lqd_prices = lqd_df['close'].tail(min_len).values

        # Calculate ratio
        ratio = hyg_prices / lqd_prices

        # Current ratio vs 20-day and 60-day averages
        current_ratio = ratio[-1]
        ratio_20d = np.mean(ratio[-20:])
        ratio_60d = np.mean(ratio)

        # Z-score of current ratio vs 60-day mean
        ratio_std = np.std(ratio)
        z_score = (current_ratio - ratio_60d) / ratio_std if ratio_std > 0 else 0

        # Also check trend (rising = risk-on)
        ratio_change_20d = (ratio[-1] / ratio[-20] - 1) * 100

        # Normalize: positive z-score and rising trend = risk-on
        normalized = 0.5 + z_score * 0.15 + ratio_change_20d / 20
        normalized = max(0.1, min(0.9, normalized))

        if z_score > 1:
            description = f"Credit spreads tight (z={z_score:.2f}) - Risk appetite"
        elif z_score < -1:
            description = f"Credit spreads wide (z={z_score:.2f}) - Flight to quality"
        else:
            description = f"Credit spreads normal (z={z_score:.2f})"

        return RegimeSignal(
            name='credit_spread',
            value=z_score,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['credit_spread'],
            description=description
        )

    def _calculate_market_breadth_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate market breadth signal.

        Uses % of cached stocks above their 50-day moving average.
        >70%: Strong breadth (risk-on)
        50-70%: Normal breadth
        30-50%: Weak breadth (risk-off)
        <30%: Very weak (crisis)
        """
        # Ensure data is loaded
        if not self.data_manager.cache:
            self.data_manager.load_all()

        above_50ma = 0
        total = 0

        for symbol, df in self.data_manager.cache.items():
            if len(df) < 50:
                continue

            try:
                current_price = float(df['close'].iloc[-1])
                ma_50 = df['close'].tail(50).mean()

                total += 1
                if current_price > ma_50:
                    above_50ma += 1
            except Exception as e:
                logger.debug(f"Market breadth calculation failed for {symbol}: {e}")
                continue

        if total < 20:
            return None

        breadth_pct = above_50ma / total * 100

        # Normalize
        if breadth_pct >= 70:
            normalized = 0.7 + (breadth_pct - 70) / 100
            description = f"Strong breadth ({breadth_pct:.0f}% above 50-MA) - Risk-on"
        elif breadth_pct >= 50:
            normalized = 0.5 + (breadth_pct - 50) / 100
            description = f"Normal breadth ({breadth_pct:.0f}% above 50-MA)"
        elif breadth_pct >= 30:
            normalized = 0.3 + (breadth_pct - 30) / 100
            description = f"Weak breadth ({breadth_pct:.0f}% above 50-MA) - Caution"
        else:
            normalized = breadth_pct / 100
            description = f"Very weak breadth ({breadth_pct:.0f}% above 50-MA) - Crisis"

        normalized = max(0.1, min(0.9, normalized))

        return RegimeSignal(
            name='market_breadth',
            value=breadth_pct,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['market_breadth'],
            description=description
        )

    def _calculate_momentum_breadth_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate momentum breadth signal.

        Uses % of stocks with positive 20-day returns.
        """
        if not self.data_manager.cache:
            self.data_manager.load_all()

        advancing = 0
        total = 0

        for symbol, df in self.data_manager.cache.items():
            if len(df) < 20:
                continue

            try:
                current_price = float(df['close'].iloc[-1])
                price_20d_ago = float(df['close'].iloc[-20])

                total += 1
                if current_price > price_20d_ago:
                    advancing += 1
            except Exception as e:
                logger.debug(f"Momentum breadth calculation failed for {symbol}: {e}")
                continue

        if total < 20:
            return None

        momentum_pct = advancing / total * 100

        # Normalize
        if momentum_pct >= 65:
            normalized = 0.7 + (momentum_pct - 65) / 100
            description = f"Strong momentum breadth ({momentum_pct:.0f}% advancing)"
        elif momentum_pct >= 50:
            normalized = 0.5 + (momentum_pct - 50) / 66
            description = f"Positive momentum breadth ({momentum_pct:.0f}% advancing)"
        elif momentum_pct >= 35:
            normalized = 0.3 + (momentum_pct - 35) / 75
            description = f"Weak momentum breadth ({momentum_pct:.0f}% advancing)"
        else:
            normalized = momentum_pct / 100
            description = f"Negative momentum breadth ({momentum_pct:.0f}% advancing) - Risk-off"

        normalized = max(0.1, min(0.9, normalized))

        return RegimeSignal(
            name='momentum_breadth',
            value=momentum_pct,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['momentum_breadth'],
            description=description
        )

    def _calculate_sector_rotation_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate sector rotation signal.

        Compares performance of defensive sectors vs cyclical sectors.
        Cyclical outperformance: Risk-on
        Defensive outperformance: Risk-off
        """
        # Get defensive and cyclical sector data
        defensive_returns = []
        for symbol in REGIME_ETFS['defensive']:
            df = self._get_etf_data(symbol, days=20)
            if df is not None and len(df) >= 20:
                ret = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                defensive_returns.append(ret)

        cyclical_returns = []
        for symbol in REGIME_ETFS['cyclical']:
            df = self._get_etf_data(symbol, days=20)
            if df is not None and len(df) >= 20:
                ret = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                cyclical_returns.append(ret)

        if not defensive_returns or not cyclical_returns:
            return None

        avg_defensive = np.mean(defensive_returns)
        avg_cyclical = np.mean(cyclical_returns)

        # Relative performance (positive = cyclical outperforming)
        relative_perf = avg_cyclical - avg_defensive

        # Normalize: cyclical outperformance = risk-on
        if relative_perf > 3:
            normalized = min(0.9, 0.7 + relative_perf / 30)
            description = f"Cyclical leadership (+{relative_perf:.1f}%) - Risk-on rotation"
        elif relative_perf > 0:
            normalized = 0.5 + relative_perf / 10
            description = f"Mild cyclical tilt (+{relative_perf:.1f}%)"
        elif relative_perf > -3:
            normalized = 0.5 + relative_perf / 10
            description = f"Mild defensive tilt ({relative_perf:.1f}%)"
        else:
            normalized = max(0.1, 0.3 + relative_perf / 20)
            description = f"Defensive leadership ({relative_perf:.1f}%) - Risk-off rotation"

        return RegimeSignal(
            name='sector_rotation',
            value=relative_perf,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['sector_rotation'],
            description=description
        )

    def _calculate_correlation_signal(self) -> Optional[RegimeSignal]:
        """
        Calculate correlation regime signal.

        High correlations typically spike during stress/crisis.
        Uses rolling 20-day correlation of major ETFs.
        """
        # Get SPY as base
        spy_df = self._get_etf_data('SPY', days=60)
        if spy_df is None or len(spy_df) < 30:
            return None

        spy_returns = spy_df['close'].pct_change().dropna()

        correlations = []
        for symbol in REGIME_ETFS['market_index'][1:] + REGIME_ETFS['cyclical'][:3]:
            df = self._get_etf_data(symbol, days=60)
            if df is None or len(df) < 30:
                continue

            returns = df['close'].pct_change().dropna()

            # Align data
            min_len = min(len(spy_returns), len(returns))
            if min_len < 20:
                continue

            # 20-day rolling correlation
            corr = np.corrcoef(spy_returns.values[-20:], returns.values[-20:])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if len(correlations) < 3:
            return None

        avg_corr = np.mean(correlations)

        # Compare to historical
        # Typical avg correlation during calm markets: 0.5-0.7
        # During stress: 0.8-0.95

        if avg_corr < 0.6:
            normalized = min(0.9, 0.7 + (0.6 - avg_corr))
            description = f"Low correlations ({avg_corr:.2f}) - Diversification working"
        elif avg_corr < 0.75:
            normalized = 0.5 + (0.75 - avg_corr)
            description = f"Normal correlations ({avg_corr:.2f})"
        elif avg_corr < 0.85:
            normalized = 0.4 - (avg_corr - 0.75) * 2
            description = f"Elevated correlations ({avg_corr:.2f}) - Caution"
        else:
            normalized = max(0.1, 0.2 - (avg_corr - 0.85) * 2)
            description = f"Correlation spike ({avg_corr:.2f}) - Crisis behavior"

        normalized = max(0.1, min(0.9, normalized))

        return RegimeSignal(
            name='correlation_regime',
            value=avg_corr,
            normalized=normalized,
            weight=self.SIGNAL_WEIGHTS['correlation_regime'],
            description=description
        )

    # =========================================================================
    # MAIN DETECTION METHODS
    # =========================================================================

    def detect_regime(self, force_refresh: bool = False) -> MarketRegime:
        """
        Detect the current market regime.

        Args:
            force_refresh: If True, recalculate even if recent detection exists

        Returns:
            Current MarketRegime
        """
        # Check cache (avoid recalculating within 5 minutes)
        if (not force_refresh and
            self._current_state is not None and
            self._last_detection is not None and
            (datetime.now() - self._last_detection).total_seconds() < 300):
            return self._current_state.regime

        # Calculate all signals
        signals = []

        signal_methods = [
            self._calculate_vix_level_signal,
            self._calculate_vix_term_structure_signal,
            self._calculate_credit_spread_signal,
            self._calculate_market_breadth_signal,
            self._calculate_momentum_breadth_signal,
            self._calculate_sector_rotation_signal,
            self._calculate_correlation_signal,
        ]

        for method in signal_methods:
            try:
                signal = method()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Signal calculation failed for {method.__name__}: {e}")

        if not signals:
            logger.error("No signals could be calculated")
            return MarketRegime.TRANSITION

        # Calculate weighted composite score
        total_weight = sum(s.weight for s in signals)
        composite_score = sum(s.normalized * s.weight for s in signals) / total_weight

        # Determine regime
        if composite_score < self.REGIME_THRESHOLDS['crisis']:
            regime = MarketRegime.CRISIS
        elif composite_score < self.REGIME_THRESHOLDS['risk_off']:
            regime = MarketRegime.RISK_OFF
        elif composite_score < self.REGIME_THRESHOLDS['transition']:
            regime = MarketRegime.TRANSITION
        else:
            regime = MarketRegime.RISK_ON

        # Calculate confidence based on signal agreement
        signal_regimes = []
        for s in signals:
            if s.normalized < 0.25:
                signal_regimes.append(MarketRegime.CRISIS)
            elif s.normalized < 0.45:
                signal_regimes.append(MarketRegime.RISK_OFF)
            elif s.normalized < 0.55:
                signal_regimes.append(MarketRegime.TRANSITION)
            else:
                signal_regimes.append(MarketRegime.RISK_ON)

        # Confidence = % of signals agreeing with final regime
        agreement = sum(1 for r in signal_regimes if r == regime)
        confidence = agreement / len(signals)

        # Boost confidence if composite score is far from thresholds
        threshold_distance = min(
            abs(composite_score - self.REGIME_THRESHOLDS['crisis']),
            abs(composite_score - self.REGIME_THRESHOLDS['risk_off']),
            abs(composite_score - self.REGIME_THRESHOLDS['transition'])
        )
        confidence = min(0.95, confidence + threshold_distance * 0.5)

        # Check for regime change
        previous_regime = self._current_state.regime if self._current_state else None
        regime_changed = previous_regime is not None and previous_regime != regime

        # Create state
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            composite_score=composite_score,
            signals=signals,
            timestamp=datetime.now(),
            previous_regime=previous_regime,
            regime_changed=regime_changed
        )

        self._current_state = state
        self._last_detection = datetime.now()
        self._regime_history.append(state)

        # Trim history
        if len(self._regime_history) > 1000:
            self._regime_history = self._regime_history[-500:]

        # Log regime change to database
        if regime_changed:
            self._log_regime_change(state)
            logger.info(f"REGIME CHANGE: {previous_regime.value} -> {regime.value} "
                       f"(confidence: {confidence:.1%})")

        return regime

    def _log_regime_change(self, state: RegimeState):
        """Log regime change to database."""
        try:
            vix_signal = next((s for s in state.signals if s.name == 'vix_level'), None)
            vix_level = vix_signal.value if vix_signal else 0

            action = f"Regime transition, composite score: {state.composite_score:.2f}"

            # Add table if it doesn't exist
            self._ensure_regime_table()

            self.db.execute(
                "performance",
                """
                INSERT INTO regime_changes
                (timestamp, regime, previous_regime, confidence, composite_score,
                 vix_level, action_taken, signal_breakdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.timestamp.isoformat(),
                    state.regime.value,
                    state.previous_regime.value if state.previous_regime else None,
                    state.confidence,
                    state.composite_score,
                    vix_level,
                    action,
                    json.dumps({s.name: s.to_dict() for s in state.signals})
                )
            )
        except Exception as e:
            logger.error(f"Failed to log regime change: {e}")

    def _ensure_regime_table(self):
        """Ensure the regime_changes table exists."""
        try:
            self.db.execute(
                "performance",
                """
                CREATE TABLE IF NOT EXISTS regime_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    previous_regime TEXT,
                    confidence REAL,
                    composite_score REAL,
                    vix_level REAL,
                    action_taken TEXT,
                    signal_breakdown TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.db.execute(
                "performance",
                "CREATE INDEX IF NOT EXISTS idx_regime_changes_timestamp ON regime_changes(timestamp)"
            )
        except Exception as e:
            logger.debug(f"Regime table creation: {e}")

    def get_regime_confidence(self) -> float:
        """Get confidence score for current regime detection."""
        if self._current_state is None:
            self.detect_regime()
        return self._current_state.confidence if self._current_state else 0.5

    def get_signal_breakdown(self) -> Dict[str, dict]:
        """Get breakdown of all signals contributing to regime detection."""
        if self._current_state is None:
            self.detect_regime()

        if self._current_state is None:
            return {}

        return {s.name: s.to_dict() for s in self._current_state.signals}

    def get_regime_history(self, days: int = 60) -> List[Dict[str, Any]]:
        """
        Get history of regime changes.

        Args:
            days: Number of days to look back

        Returns:
            List of regime change dictionaries
        """
        try:
            self._ensure_regime_table()

            rows = self.db.fetchall(
                "performance",
                """
                SELECT timestamp, regime, previous_regime, confidence,
                       composite_score, vix_level, action_taken
                FROM regime_changes
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
                """,
                (days,)
            )

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get regime history: {e}")
            return []

    def get_strategy_recommendations(self, regime: MarketRegime = None) -> Dict[str, float]:
        """
        Get strategy position size multipliers for the given regime.

        Args:
            regime: MarketRegime (uses current if None)

        Returns:
            Dict mapping strategy names to position size multipliers
        """
        if regime is None:
            regime = self.detect_regime()

        adjustments = REGIME_ADJUSTMENTS.get(regime, {})

        # If crisis mode has 'all' key, apply to all strategies
        if 'all' in adjustments:
            default_mult = adjustments['all']
            result = {k: v for k, v in adjustments.items() if k != 'all'}
            # Fill in any missing strategies with the 'all' multiplier
            for strategy in REGIME_ADJUSTMENTS[MarketRegime.RISK_ON].keys():
                if strategy not in result:
                    result[strategy] = default_mult
            return result

        return adjustments.copy()

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get full current regime state as dictionary."""
        if self._current_state is None:
            self.detect_regime()

        if self._current_state is None:
            return None

        return self._current_state.to_dict()

    def get_regime_duration(self) -> Optional[timedelta]:
        """Get how long we've been in the current regime."""
        if len(self._regime_history) < 2:
            return None

        current_regime = self._current_state.regime if self._current_state else None
        if current_regime is None:
            return None

        # Find when regime started
        for state in reversed(self._regime_history[:-1]):
            if state.regime != current_regime:
                return datetime.now() - state.timestamp

        # Been in same regime for entire history
        return datetime.now() - self._regime_history[0].timestamp

    def print_status(self):
        """Print current regime status to console."""
        if self._current_state is None:
            self.detect_regime()

        if self._current_state is None:
            print("Unable to detect regime")
            return

        state = self._current_state

        print("\n" + "=" * 60)
        print("MARKET REGIME DETECTION")
        print("=" * 60)
        print(f"Regime: {state.regime.value.upper()}")
        print(f"Confidence: {state.confidence:.1%}")
        print(f"Composite Score: {state.composite_score:.2f}")
        print(f"Timestamp: {state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        duration = self.get_regime_duration()
        if duration:
            print(f"Duration: {duration}")

        print("\n" + "-" * 40)
        print("SIGNAL BREAKDOWN:")
        print("-" * 40)

        for signal in sorted(state.signals, key=lambda x: -x.weight):
            bar = "#" * int(signal.normalized * 20) + "." * (20 - int(signal.normalized * 20))
            print(f"  {signal.name:20s} [{bar}] {signal.normalized:.2f}")
            print(f"    {signal.description}")

        print("\n" + "-" * 40)
        print("STRATEGY RECOMMENDATIONS:")
        print("-" * 40)

        recommendations = self.get_strategy_recommendations()
        for strategy, mult in sorted(recommendations.items()):
            if mult < 0.5:
                indicator = "REDUCE"
            elif mult > 1.0:
                indicator = "INCREASE"
            else:
                indicator = "NORMAL"
            print(f"  {strategy:25s}: {mult:.2f}x  ({indicator})")

        print("=" * 60 + "\n")


# ============================================================================
# MAIN - Demo/Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    print("\nInitializing Regime Detector...")
    detector = RegimeDetector()

    # Load data
    print("Loading market data...")
    detector.data_manager.load_all()

    # Detect regime
    print("\nDetecting regime...")
    regime = detector.detect_regime()

    # Print status
    detector.print_status()

    # Show history
    print("\nRecent regime history:")
    history = detector.get_regime_history(days=30)
    if history:
        for entry in history[:5]:
            print(f"  {entry['timestamp']}: {entry.get('previous_regime', 'N/A')} -> {entry['regime']}")
    else:
        print("  No regime changes in last 30 days")

    print("\nDone!")
