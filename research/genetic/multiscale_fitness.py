"""
Multi-Scale Fitness
===================
Advanced fitness functions that combine performance across multiple scales:
- Long-term (years of data)
- Medium-term (specific market regimes)
- Short-term (rapid period tests)
- Crisis resilience
- Consistency across conditions

This prevents over-optimization to a single market regime while still
allowing adaptation to current conditions.

Usage:
    from research.genetic.multiscale_fitness import (
        MultiScaleFitnessCalculator,
        create_adaptive_fitness_function
    )

    calculator = MultiScaleFitnessCalculator()

    # Calculate multi-scale fitness
    fitness = calculator.calculate(
        strategy,
        data,
        current_vix=18,
        current_regime='risk_on'
    )
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .market_periods import MarketPeriodLibrary, MarketPeriod, PeriodType
from .rapid_backtester import RapidBacktester, RapidBacktestResult, MultiPeriodResult
from .regime_matching import RegimeMatchingEngine, RegimeFingerprint

logger = logging.getLogger(__name__)


@dataclass
class FitnessComponent:
    """A single component of the multi-scale fitness."""
    name: str
    value: float
    weight: float
    normalized: float  # 0-1 scale
    description: str
    period_count: int = 0

    @property
    def contribution(self) -> float:
        """Weighted contribution to total fitness."""
        return self.normalized * self.weight


@dataclass
class MultiScaleFitnessResult:
    """Complete result of multi-scale fitness calculation."""
    total_fitness: float
    components: List[FitnessComponent]

    # Individual scale scores
    long_term_score: float
    regime_match_score: float
    crisis_score: float
    consistency_score: float
    alpha_score: float

    # Period breakdown
    period_results: Dict[str, float]  # period_name -> sharpe

    # Metadata
    current_regime: str
    n_periods_tested: int
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_fitness': self.total_fitness,
            'long_term_score': self.long_term_score,
            'regime_match_score': self.regime_match_score,
            'crisis_score': self.crisis_score,
            'consistency_score': self.consistency_score,
            'alpha_score': self.alpha_score,
            'current_regime': self.current_regime,
            'n_periods_tested': self.n_periods_tested,
            'components': [
                {
                    'name': c.name,
                    'value': c.value,
                    'weight': c.weight,
                    'normalized': c.normalized,
                    'contribution': c.contribution,
                }
                for c in self.components
            ],
        }


@dataclass
class FitnessWeights:
    """Configurable weights for fitness components."""
    # Core weights
    long_term: float = 0.30
    regime_matched: float = 0.25
    crisis_resilience: float = 0.15
    consistency: float = 0.15
    alpha: float = 0.10
    trade_quality: float = 0.05

    # Regime-specific adjustments
    crisis_mode_boost: float = 0.3  # Extra weight to crisis score during crisis
    bull_mode_boost: float = 0.2   # Extra weight to alpha during bull

    # Penalties
    overfitting_penalty: float = 0.3
    low_trades_penalty: float = 0.2
    high_drawdown_penalty: float = 0.25

    def adjusted_for_regime(self, regime: str) -> 'FitnessWeights':
        """Return weights adjusted for current regime."""
        w = FitnessWeights(
            long_term=self.long_term,
            regime_matched=self.regime_matched,
            crisis_resilience=self.crisis_resilience,
            consistency=self.consistency,
            alpha=self.alpha,
            trade_quality=self.trade_quality,
        )

        if regime == 'crisis':
            # Prioritize crisis resilience and consistency
            w.crisis_resilience += self.crisis_mode_boost
            w.consistency += 0.1
            w.alpha -= 0.1
            w.regime_matched -= 0.1
        elif regime == 'risk_on':
            # Prioritize alpha and regime match
            w.alpha += self.bull_mode_boost
            w.regime_matched += 0.1
            w.crisis_resilience -= 0.1
            w.long_term -= 0.1
        elif regime == 'risk_off':
            # Prioritize long-term and consistency
            w.long_term += 0.1
            w.consistency += 0.1
            w.alpha -= 0.15

        # Normalize to sum to 1
        total = w.long_term + w.regime_matched + w.crisis_resilience + w.consistency + w.alpha + w.trade_quality
        w.long_term /= total
        w.regime_matched /= total
        w.crisis_resilience /= total
        w.consistency /= total
        w.alpha /= total
        w.trade_quality /= total

        return w


class MultiScaleFitnessCalculator:
    """
    Calculate fitness across multiple scales and conditions.

    Combines:
    1. Long-term performance (years of data)
    2. Regime-matched performance (current market conditions)
    3. Crisis resilience (stress test periods)
    4. Consistency (low variance across periods)
    5. Alpha generation (beating benchmarks)
    6. Trade quality (win rate, profit factor)
    """

    # Normalization ranges for metrics
    METRIC_RANGES = {
        'sharpe': (-1.0, 3.0),
        'sortino': (-1.0, 4.0),
        'calmar': (-1.0, 3.0),
        'max_drawdown': (-50.0, 0.0),
        'win_rate': (0.0, 100.0),
        'profit_factor': (0.0, 3.0),
        'alpha': (-20.0, 30.0),
    }

    def __init__(
        self,
        backtester: RapidBacktester = None,
        regime_engine: RegimeMatchingEngine = None,
        weights: FitnessWeights = None
    ):
        """
        Initialize the calculator.

        Args:
            backtester: RapidBacktester instance
            regime_engine: RegimeMatchingEngine instance
            weights: FitnessWeights configuration
        """
        self.library = MarketPeriodLibrary()
        self.backtester = backtester or RapidBacktester(period_library=self.library)
        self.regime_engine = regime_engine or RegimeMatchingEngine(self.library)
        self.base_weights = weights or FitnessWeights()

        logger.info("MultiScaleFitnessCalculator initialized")

    def _normalize(
        self,
        value: float,
        metric: str,
        invert: bool = False
    ) -> float:
        """Normalize a metric to 0-1 scale."""
        min_val, max_val = self.METRIC_RANGES.get(metric, (0, 1))

        # Clip to range
        clipped = max(min_val, min(max_val, value))

        # Normalize
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (clipped - min_val) / (max_val - min_val)

        if invert:
            normalized = 1 - normalized

        return normalized

    def _calculate_long_term_score(
        self,
        strategy: Any,
        data: Dict[str, Any],
        vix_data: Any = None
    ) -> Tuple[float, List[RapidBacktestResult]]:
        """Calculate long-term performance score."""
        # Use full year periods
        year_periods = self.library.get_year_periods()

        results = []
        for period in year_periods[-5:]:  # Last 5 years
            try:
                result = self.backtester.run_period_test(
                    strategy,
                    period,
                    data=data,
                    vix_data=vix_data
                )
                if result.is_valid:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Long-term test failed for {period.name}: {e}")

        if not results:
            return 0.0, []

        # Average Sharpe across years
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])

        # Bonus for consistent positive performance
        positive_years = sum(1 for r in results if r.sharpe_ratio > 0)
        consistency_bonus = positive_years / len(results) * 0.2

        score = self._normalize(avg_sharpe, 'sharpe') + consistency_bonus
        return min(1.0, score), results

    def _calculate_regime_match_score(
        self,
        strategy: Any,
        fingerprint: RegimeFingerprint,
        data: Dict[str, Any],
        vix_data: Any = None
    ) -> Tuple[float, List[RapidBacktestResult]]:
        """Calculate score on regime-matched periods."""
        # Find similar periods
        matches = self.regime_engine.find_matching_periods(
            fingerprint,
            n=5,
            min_similarity=0.4
        )

        if not matches:
            return 0.5, []  # Neutral if no matches

        results = []
        weighted_sharpes = []

        for match in matches:
            try:
                result = self.backtester.run_period_test(
                    strategy,
                    match.period,
                    data=data,
                    vix_data=vix_data
                )
                if result.is_valid:
                    results.append(result)
                    # Weight by similarity
                    weighted_sharpes.append(result.sharpe_ratio * match.similarity)
            except Exception as e:
                logger.debug(f"Regime test failed for {match.period.name}: {e}")

        if not weighted_sharpes:
            return 0.5, []

        # Average weighted Sharpe
        avg_weighted = np.mean(weighted_sharpes)

        score = self._normalize(avg_weighted, 'sharpe')
        return score, results

    def _calculate_crisis_score(
        self,
        strategy: Any,
        data: Dict[str, Any],
        vix_data: Any = None
    ) -> Tuple[float, List[RapidBacktestResult]]:
        """Calculate crisis resilience score."""
        crisis_periods = self.library.get_crisis_periods()

        results = []
        for period in crisis_periods[:4]:  # Top 4 crisis periods
            try:
                result = self.backtester.run_period_test(
                    strategy,
                    period,
                    data=data,
                    vix_data=vix_data
                )
                if result.is_valid:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Crisis test failed for {period.name}: {e}")

        if not results:
            return 0.0, []

        # For crisis, we care more about survival than profit
        # Positive Sharpe during crisis is exceptional
        crisis_sharpes = [r.sharpe_ratio for r in results]
        crisis_drawdowns = [r.max_drawdown for r in results]

        # Average Sharpe (shifted to reward survival)
        avg_sharpe = np.mean(crisis_sharpes)
        sharpe_score = self._normalize(avg_sharpe + 1.0, 'sharpe')  # Shift by +1

        # Average drawdown (less severe = better)
        avg_dd = np.mean(crisis_drawdowns)
        dd_score = self._normalize(avg_dd, 'max_drawdown', invert=True)

        # Combine (drawdown matters more in crisis)
        score = sharpe_score * 0.4 + dd_score * 0.6

        return score, results

    def _calculate_consistency_score(
        self,
        all_results: List[RapidBacktestResult]
    ) -> float:
        """Calculate consistency across all tested periods."""
        if len(all_results) < 3:
            return 0.5  # Not enough data

        sharpes = [r.sharpe_ratio for r in all_results if r.is_valid]

        if not sharpes:
            return 0.0

        # Coefficient of variation (lower = more consistent)
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)

        if mean_sharpe <= 0:
            return 0.1  # Low score for negative average

        cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else 1.0

        # Convert CV to score (lower CV = higher score)
        # CV of 0 -> score 1.0
        # CV of 1 -> score 0.5
        # CV of 2 -> score 0.0
        score = max(0, 1 - cv / 2)

        # Bonus for all positive periods
        positive_pct = sum(1 for s in sharpes if s > 0) / len(sharpes)
        bonus = positive_pct * 0.2

        return min(1.0, score + bonus)

    def _calculate_alpha_score(
        self,
        all_results: List[RapidBacktestResult]
    ) -> float:
        """Calculate alpha generation score."""
        if not all_results:
            return 0.0

        # Alpha is strategy return - benchmark return
        alphas = [r.alpha for r in all_results if r.is_valid]

        if not alphas:
            return 0.0

        avg_alpha = np.mean(alphas)

        # Normalize alpha
        score = self._normalize(avg_alpha, 'alpha')

        return score

    def _calculate_trade_quality_score(
        self,
        all_results: List[RapidBacktestResult]
    ) -> float:
        """Calculate trade quality score."""
        if not all_results:
            return 0.0

        valid_results = [r for r in all_results if r.is_valid and r.total_trades > 0]

        if not valid_results:
            return 0.0

        # Average win rate
        avg_win_rate = np.mean([r.win_rate for r in valid_results])
        win_score = self._normalize(avg_win_rate, 'win_rate')

        # Average profit factor
        profit_factors = [r.profit_factor for r in valid_results if r.profit_factor < float('inf')]
        if profit_factors:
            avg_pf = np.mean(profit_factors)
            pf_score = self._normalize(avg_pf, 'profit_factor')
        else:
            pf_score = 0.5

        # Total trades (we want enough but not too many)
        total_trades = sum(r.total_trades for r in valid_results)
        trades_per_period = total_trades / len(valid_results)

        # Ideal is 5-20 trades per short period
        if trades_per_period < 3:
            trades_score = 0.3
        elif trades_per_period < 5:
            trades_score = 0.6
        elif trades_per_period < 20:
            trades_score = 1.0
        elif trades_per_period < 50:
            trades_score = 0.8
        else:
            trades_score = 0.5  # Too many might be overtrading

        # Combine
        score = win_score * 0.4 + pf_score * 0.4 + trades_score * 0.2

        return score

    def _apply_penalties(
        self,
        base_fitness: float,
        all_results: List[RapidBacktestResult],
        weights: FitnessWeights
    ) -> float:
        """Apply penalties for undesirable characteristics."""
        import math

        fitness = base_fitness

        if not all_results:
            return 0.0

        # Statistical validity threshold (research recommends 30+ for basic inference)
        MIN_TRADES_THRESHOLD = 30

        # Low trades penalty using exponential soft penalty
        # Instead of hard thresholds, use continuous penalty that rewards more trades
        total_trades = sum(r.total_trades for r in all_results)
        if total_trades < MIN_TRADES_THRESHOLD:
            # Deb's feasibility rules: rank by constraint violation
            # Scale fitness down proportionally to how far below threshold
            trade_factor = total_trades / MIN_TRADES_THRESHOLD
            fitness *= trade_factor
        else:
            # Exponential soft penalty - still reward more trades for statistical robustness
            # At 30 trades: factor = 0.63, at 60: 0.86, at 90: 0.95
            trade_factor = 1 - math.exp(-total_trades / MIN_TRADES_THRESHOLD)
            fitness *= trade_factor

        # High drawdown penalty
        worst_dd = min(r.max_drawdown for r in all_results)
        if worst_dd < -30:
            fitness *= (1 - weights.high_drawdown_penalty)
        elif worst_dd < -20:
            fitness *= (1 - weights.high_drawdown_penalty / 2)

        # Overfitting detection
        # If performance degrades significantly in recent periods
        sorted_results = sorted(all_results, key=lambda r: r.period_name)
        if len(sorted_results) >= 4:
            early_sharpes = [r.sharpe_ratio for r in sorted_results[:len(sorted_results)//2]]
            late_sharpes = [r.sharpe_ratio for r in sorted_results[len(sorted_results)//2:]]

            if early_sharpes and late_sharpes:
                degradation = np.mean(early_sharpes) - np.mean(late_sharpes)
                if degradation > 0.5:  # Significant degradation
                    fitness *= (1 - weights.overfitting_penalty)
                elif degradation > 0.25:
                    fitness *= (1 - weights.overfitting_penalty / 2)

        return max(0, fitness)

    def calculate(
        self,
        strategy: Any,
        data: Dict[str, Any],
        vix_data: Any = None,
        current_vix: float = None,
        current_regime: str = None,
        fingerprint: RegimeFingerprint = None,
        regime_adjusted_weights: bool = True
    ) -> MultiScaleFitnessResult:
        """
        Calculate complete multi-scale fitness.

        Args:
            strategy: Strategy to evaluate
            data: Market data
            vix_data: VIX data
            current_vix: Current VIX level
            current_regime: Current market regime
            fingerprint: RegimeFingerprint (calculated if None)
            regime_adjusted_weights: Adjust weights based on current regime

        Returns:
            MultiScaleFitnessResult
        """
        import time
        start_time = time.time()

        # Get current regime fingerprint
        if fingerprint is None:
            fingerprint = self.regime_engine.get_current_fingerprint()

        if current_regime is None:
            current_regime = fingerprint.overall_regime

        # Cache data in backtester
        self.backtester.cache_data(data)

        # Get weights (possibly adjusted for regime)
        weights = self.base_weights
        if regime_adjusted_weights:
            weights = weights.adjusted_for_regime(current_regime)

        # Collect all results for cross-scale analysis
        all_results: List[RapidBacktestResult] = []
        period_results: Dict[str, float] = {}

        components: List[FitnessComponent] = []

        # 1. Long-term performance
        long_term_score, long_term_results = self._calculate_long_term_score(
            strategy, data, vix_data
        )
        all_results.extend(long_term_results)
        for r in long_term_results:
            period_results[r.period_name] = r.sharpe_ratio

        components.append(FitnessComponent(
            name='long_term',
            value=np.mean([r.sharpe_ratio for r in long_term_results]) if long_term_results else 0,
            weight=weights.long_term,
            normalized=long_term_score,
            description='Performance across multiple years',
            period_count=len(long_term_results),
        ))

        # 2. Regime-matched performance
        regime_score, regime_results = self._calculate_regime_match_score(
            strategy, fingerprint, data, vix_data
        )
        all_results.extend(regime_results)
        for r in regime_results:
            period_results[r.period_name] = r.sharpe_ratio

        components.append(FitnessComponent(
            name='regime_matched',
            value=np.mean([r.sharpe_ratio for r in regime_results]) if regime_results else 0,
            weight=weights.regime_matched,
            normalized=regime_score,
            description=f'Performance in conditions similar to current ({current_regime})',
            period_count=len(regime_results),
        ))

        # 3. Crisis resilience
        crisis_score, crisis_results = self._calculate_crisis_score(
            strategy, data, vix_data
        )
        all_results.extend(crisis_results)
        for r in crisis_results:
            period_results[r.period_name] = r.sharpe_ratio

        components.append(FitnessComponent(
            name='crisis_resilience',
            value=np.mean([r.sharpe_ratio for r in crisis_results]) if crisis_results else 0,
            weight=weights.crisis_resilience,
            normalized=crisis_score,
            description='Survival and performance during crisis periods',
            period_count=len(crisis_results),
        ))

        # 4. Consistency
        consistency_score = self._calculate_consistency_score(all_results)
        components.append(FitnessComponent(
            name='consistency',
            value=np.std([r.sharpe_ratio for r in all_results]) if all_results else 0,
            weight=weights.consistency,
            normalized=consistency_score,
            description='Consistency of performance across periods',
            period_count=len(all_results),
        ))

        # 5. Alpha generation
        alpha_score = self._calculate_alpha_score(all_results)
        components.append(FitnessComponent(
            name='alpha',
            value=np.mean([r.alpha for r in all_results]) if all_results else 0,
            weight=weights.alpha,
            normalized=alpha_score,
            description='Excess return vs benchmark',
            period_count=len(all_results),
        ))

        # 6. Trade quality
        quality_score = self._calculate_trade_quality_score(all_results)
        components.append(FitnessComponent(
            name='trade_quality',
            value=np.mean([r.win_rate for r in all_results if r.total_trades > 0]) if all_results else 0,
            weight=weights.trade_quality,
            normalized=quality_score,
            description='Win rate and profit factor',
            period_count=len(all_results),
        ))

        # Calculate weighted fitness
        total_fitness = sum(c.contribution for c in components)

        # Apply penalties
        total_fitness = self._apply_penalties(total_fitness, all_results, weights)

        execution_time_ms = (time.time() - start_time) * 1000

        return MultiScaleFitnessResult(
            total_fitness=total_fitness,
            components=components,
            long_term_score=long_term_score,
            regime_match_score=regime_score,
            crisis_score=crisis_score,
            consistency_score=consistency_score,
            alpha_score=alpha_score,
            period_results=period_results,
            current_regime=current_regime,
            n_periods_tested=len(all_results),
            execution_time_ms=execution_time_ms,
        )


def create_adaptive_fitness_function(
    calculator: MultiScaleFitnessCalculator,
    strategy_factory: Callable[[Dict], Any],
    data: Dict[str, Any],
    vix_data: Any = None,
    current_conditions: Dict[str, float] = None
) -> Callable[[Dict], float]:
    """
    Create an adaptive fitness function for GA optimization.

    The returned function:
    - Takes a gene dictionary
    - Returns a fitness score

    Automatically adjusts weights based on current market conditions.

    Args:
        calculator: MultiScaleFitnessCalculator instance
        strategy_factory: Function to create strategy from genes
        data: Market data
        vix_data: VIX data
        current_conditions: Dict with 'vix', 'regime', etc.

    Returns:
        Fitness function for GA
    """
    # Pre-calculate fingerprint if conditions provided
    fingerprint = None
    if current_conditions:
        fingerprint = RegimeFingerprint(
            vix_level=current_conditions.get('vix', 15),
            vix_percentile=50,
            vix_trend=0,
            trend_direction=current_conditions.get('trend', 0),
            trend_strength=abs(current_conditions.get('trend', 0)),
            momentum_breadth=50,
            realized_vol=15,
            vol_regime='normal',
            vol_trend=0,
            correlation_level=current_conditions.get('correlation', 0.5),
            correlation_trend=0,
            sector_leadership='mixed',
            sector_dispersion=0.5,
            credit_spread_z=0,
            term_structure='contango',
            overall_regime=current_conditions.get('regime', 'transition'),
            regime_confidence=0.5,
        )

    def fitness_fn(genes: Dict[str, float]) -> float:
        try:
            # Create strategy
            strategy = strategy_factory(genes)

            # Calculate multi-scale fitness
            result = calculator.calculate(
                strategy,
                data,
                vix_data=vix_data,
                fingerprint=fingerprint,
            )

            return result.total_fitness

        except Exception as e:
            logger.warning(f"Fitness calculation failed: {e}")
            return 0.0

    return fitness_fn


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("MULTI-SCALE FITNESS DEMO")
    print("=" * 60)

    # Create calculator
    calculator = MultiScaleFitnessCalculator()

    # Show weights
    print("\nDefault fitness weights:")
    w = calculator.base_weights
    print(f"  Long-term: {w.long_term:.2f}")
    print(f"  Regime-matched: {w.regime_matched:.2f}")
    print(f"  Crisis: {w.crisis_resilience:.2f}")
    print(f"  Consistency: {w.consistency:.2f}")
    print(f"  Alpha: {w.alpha:.2f}")
    print(f"  Trade quality: {w.trade_quality:.2f}")

    # Show adjusted weights for different regimes
    print("\nWeights adjusted for crisis:")
    w_crisis = w.adjusted_for_regime('crisis')
    print(f"  Crisis: {w_crisis.crisis_resilience:.2f} (+)")
    print(f"  Consistency: {w_crisis.consistency:.2f} (+)")

    print("\nWeights adjusted for risk_on:")
    w_bull = w.adjusted_for_regime('risk_on')
    print(f"  Alpha: {w_bull.alpha:.2f} (+)")
    print(f"  Regime-matched: {w_bull.regime_matched:.2f} (+)")

    print("\nTo use with actual data:")
    print("  result = calculator.calculate(strategy, data)")
    print("  print(f'Total fitness: {result.total_fitness}')")
