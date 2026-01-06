#!/usr/bin/env python3
"""
Strategy Ensemble Intelligence
==============================
Monitor strategy correlations and dynamically adjust allocations.

Features:
- Real-time correlation tracking between strategies
- Detect clustering (strategies betting the same way)
- Reduce exposure when correlation spikes
- Boost allocation when strategies diverge
- Anti-fragile portfolio construction
- Signal agreement/disagreement analysis

Usage:
    ensemble = StrategyEnsemble()

    # Update with daily returns
    ensemble.update_returns(strategy_returns)

    # Get allocation adjustments
    adjustments = ensemble.get_allocation_adjustments()

    # Check for clustering
    if ensemble.is_clustering():
        reduce_exposure()
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import json

import pandas as pd
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CorrelationMatrix:
    """Current correlation state between strategies."""
    matrix: pd.DataFrame
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    clustering_score: float  # 0 = diverse, 1 = all correlated
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AllocationAdjustment:
    """Recommended adjustment for a strategy."""
    strategy: str
    current_weight: float
    recommended_weight: float
    adjustment_factor: float  # multiplier (1.0 = no change)
    reason: str
    confidence: float


@dataclass
class ClusterAlert:
    """Alert when strategies are clustering."""
    timestamp: str
    avg_correlation: float
    clustering_strategies: List[str]
    risk_level: str  # low, medium, high, critical
    recommended_action: str


@dataclass
class SignalAgreement:
    """Analysis of signal agreement across strategies."""
    timestamp: str
    total_strategies: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    agreement_ratio: float  # 0 = all disagree, 1 = all agree
    consensus_direction: str  # bullish, bearish, mixed
    confidence: float


@dataclass
class EnsembleConfig:
    """Configuration for ensemble intelligence."""
    # Correlation thresholds
    high_correlation_threshold: float = 0.7
    clustering_threshold: float = 0.6
    diversification_bonus_threshold: float = 0.3

    # Rolling window for correlation
    correlation_window_days: int = 30
    min_observations: int = 10

    # Allocation adjustments
    max_correlation_penalty: float = 0.5  # Reduce allocation by up to 50%
    diversification_bonus: float = 0.25  # Increase allocation by up to 25%
    max_single_strategy_weight: float = 0.4

    # Clustering response
    cluster_exposure_reduction: float = 0.3
    cluster_alert_threshold: float = 0.65


# ============================================================================
# MAIN ENSEMBLE CLASS
# ============================================================================

class StrategyEnsemble:
    """
    Monitor and optimize strategy ensemble behavior.

    Tracks correlations, detects clustering, and recommends
    allocation adjustments for better diversification.
    """

    def __init__(self, config: EnsembleConfig = None):
        """
        Initialize ensemble intelligence.

        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()

        # Rolling returns storage
        self.returns_history: Dict[str, deque] = {}
        self.dates: deque = deque(maxlen=self.config.correlation_window_days * 2)

        # Current state
        self.current_correlations: Optional[CorrelationMatrix] = None
        self.current_signals: Dict[str, str] = {}  # strategy -> direction
        self.base_weights: Dict[str, float] = {}

        # Alert history
        self.cluster_alerts: List[ClusterAlert] = []

        logger.info("StrategyEnsemble initialized")

    # ========================================================================
    # DATA INGESTION
    # ========================================================================

    def update_returns(
        self,
        returns: Dict[str, float],
        date: datetime = None,
    ):
        """
        Update with daily returns for each strategy.

        Args:
            returns: Dict mapping strategy name to daily return
            date: Date of returns (default: today)
        """
        if date is None:
            date = datetime.now()

        self.dates.append(date)

        for strategy, ret in returns.items():
            if strategy not in self.returns_history:
                self.returns_history[strategy] = deque(
                    maxlen=self.config.correlation_window_days * 2
                )
            self.returns_history[strategy].append(ret)

        # Recalculate correlations
        self._update_correlations()

        # Check for clustering
        self._check_clustering()

    def set_base_weights(self, weights: Dict[str, float]):
        """
        Set base allocation weights for strategies.

        Args:
            weights: Dict mapping strategy to base weight (should sum to 1.0)
        """
        total = sum(weights.values())
        self.base_weights = {k: v / total for k, v in weights.items()}

    def update_signals(self, signals: Dict[str, str]):
        """
        Update current signal directions for each strategy.

        Args:
            signals: Dict mapping strategy to direction ('bullish', 'bearish', 'neutral')
        """
        self.current_signals = signals

    # ========================================================================
    # CORRELATION ANALYSIS
    # ========================================================================

    def _update_correlations(self):
        """Update correlation matrix based on recent returns."""
        strategies = list(self.returns_history.keys())

        if len(strategies) < 2:
            return

        # Check minimum observations
        min_obs = min(len(self.returns_history[s]) for s in strategies)
        if min_obs < self.config.min_observations:
            return

        # Build returns DataFrame
        returns_df = pd.DataFrame({
            s: list(self.returns_history[s])[-min_obs:]
            for s in strategies
        })

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Calculate statistics
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_triangle.values.flatten()
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) == 0:
            return

        avg_corr = float(np.mean(correlations))
        max_corr = float(np.max(correlations))
        min_corr = float(np.min(correlations))

        # Clustering score: average of absolute correlations
        clustering = float(np.mean(np.abs(correlations)))

        self.current_correlations = CorrelationMatrix(
            matrix=corr_matrix,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
            clustering_score=clustering,
        )

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get current correlation matrix."""
        if self.current_correlations is None:
            return None
        return self.current_correlations.matrix

    def get_pairwise_correlation(self, strategy1: str, strategy2: str) -> float:
        """Get correlation between two specific strategies."""
        if self.current_correlations is None:
            return 0.0

        matrix = self.current_correlations.matrix
        if strategy1 not in matrix.columns or strategy2 not in matrix.columns:
            return 0.0

        return float(matrix.loc[strategy1, strategy2])

    def get_strategy_diversification_score(self, strategy: str) -> float:
        """
        Get diversification score for a single strategy.

        High score = strategy adds diversification to portfolio.
        Low score = strategy is redundant with others.
        """
        if self.current_correlations is None:
            return 0.5

        matrix = self.current_correlations.matrix
        if strategy not in matrix.columns:
            return 0.5

        # Average correlation with other strategies
        correlations = matrix[strategy].drop(strategy)
        avg_corr = correlations.mean()

        # Score: 0 (perfectly correlated) to 1 (perfectly uncorrelated)
        return float(1.0 - abs(avg_corr))

    # ========================================================================
    # CLUSTERING DETECTION
    # ========================================================================

    def _check_clustering(self):
        """Check for strategy clustering and generate alerts."""
        if self.current_correlations is None:
            return

        clustering = self.current_correlations.clustering_score

        if clustering > self.config.cluster_alert_threshold:
            # Find highly correlated pairs
            matrix = self.current_correlations.matrix
            clustering_strategies = set()

            for i, s1 in enumerate(matrix.columns):
                for s2 in matrix.columns[i+1:]:
                    if abs(matrix.loc[s1, s2]) > self.config.high_correlation_threshold:
                        clustering_strategies.add(s1)
                        clustering_strategies.add(s2)

            # Determine risk level
            if clustering > 0.85:
                risk_level = "critical"
                action = "Immediately reduce exposure by 50%"
            elif clustering > 0.75:
                risk_level = "high"
                action = "Reduce exposure by 30% and review positions"
            elif clustering > 0.65:
                risk_level = "medium"
                action = "Monitor closely and consider reducing exposure"
            else:
                risk_level = "low"
                action = "Continue monitoring"

            alert = ClusterAlert(
                timestamp=datetime.now().isoformat(),
                avg_correlation=self.current_correlations.avg_correlation,
                clustering_strategies=list(clustering_strategies),
                risk_level=risk_level,
                recommended_action=action,
            )

            self.cluster_alerts.append(alert)
            if len(self.cluster_alerts) > 100:
                self.cluster_alerts = self.cluster_alerts[-100:]

            if risk_level in ("high", "critical"):
                logger.warning(
                    f"CLUSTERING ALERT [{risk_level}]: "
                    f"avg_corr={clustering:.2f}, strategies={list(clustering_strategies)}"
                )

    def is_clustering(self) -> bool:
        """Check if strategies are currently clustering."""
        if self.current_correlations is None:
            return False
        return self.current_correlations.clustering_score > self.config.clustering_threshold

    def get_cluster_risk_level(self) -> str:
        """Get current clustering risk level."""
        if self.current_correlations is None:
            return "unknown"

        clustering = self.current_correlations.clustering_score

        if clustering > 0.85:
            return "critical"
        elif clustering > 0.75:
            return "high"
        elif clustering > 0.65:
            return "medium"
        elif clustering > 0.5:
            return "low"
        else:
            return "none"

    # ========================================================================
    # ALLOCATION ADJUSTMENTS
    # ========================================================================

    def get_allocation_adjustments(self) -> Dict[str, AllocationAdjustment]:
        """
        Get recommended allocation adjustments based on correlations.

        Returns:
            Dict mapping strategy to AllocationAdjustment
        """
        if not self.base_weights:
            logger.warning("No base weights set. Call set_base_weights() first.")
            return {}

        adjustments = {}

        for strategy, base_weight in self.base_weights.items():
            div_score = self.get_strategy_diversification_score(strategy)

            # Calculate adjustment factor
            if div_score > 1 - self.config.diversification_bonus_threshold:
                # High diversification: bonus
                adjustment = 1.0 + (div_score - 0.7) * self.config.diversification_bonus
                reason = f"High diversification score ({div_score:.2f})"
            elif div_score < 1 - self.config.high_correlation_threshold:
                # Low diversification: penalty
                adjustment = 1.0 - (0.3 - div_score) * self.config.max_correlation_penalty
                reason = f"Low diversification score ({div_score:.2f})"
            else:
                adjustment = 1.0
                reason = "Normal diversification"

            # Apply clustering penalty
            if self.is_clustering():
                clustering_adj = 1.0 - self.config.cluster_exposure_reduction
                adjustment *= clustering_adj
                reason += f", clustering penalty ({clustering_adj:.2f})"

            # Cap maximum weight
            recommended = min(
                base_weight * adjustment,
                self.config.max_single_strategy_weight
            )

            adjustments[strategy] = AllocationAdjustment(
                strategy=strategy,
                current_weight=base_weight,
                recommended_weight=recommended,
                adjustment_factor=adjustment,
                reason=reason,
                confidence=0.8 if self.current_correlations else 0.5,
            )

        # Normalize to sum to 1.0
        total = sum(a.recommended_weight for a in adjustments.values())
        if total > 0:
            for adj in adjustments.values():
                adj.recommended_weight /= total

        return adjustments

    def get_exposure_multiplier(self) -> float:
        """
        Get overall portfolio exposure multiplier.

        Returns value between 0.5 and 1.0.
        Lower when clustering is high.
        """
        if self.current_correlations is None:
            return 1.0

        clustering = self.current_correlations.clustering_score

        if clustering > 0.85:
            return 0.5
        elif clustering > 0.75:
            return 0.65
        elif clustering > 0.65:
            return 0.8
        elif clustering > 0.55:
            return 0.9
        else:
            return 1.0

    # ========================================================================
    # SIGNAL AGREEMENT ANALYSIS
    # ========================================================================

    def analyze_signal_agreement(self) -> SignalAgreement:
        """
        Analyze agreement/disagreement of current signals.

        Returns:
            SignalAgreement with consensus analysis
        """
        if not self.current_signals:
            return SignalAgreement(
                timestamp=datetime.now().isoformat(),
                total_strategies=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                agreement_ratio=0.0,
                consensus_direction="unknown",
                confidence=0.0,
            )

        total = len(self.current_signals)
        bullish = sum(1 for s in self.current_signals.values() if s == 'bullish')
        bearish = sum(1 for s in self.current_signals.values() if s == 'bearish')
        neutral = total - bullish - bearish

        # Agreement ratio: max(bullish, bearish) / total
        max_agreement = max(bullish, bearish, neutral)
        agreement_ratio = max_agreement / total if total > 0 else 0

        # Consensus direction
        if bullish > bearish and bullish > neutral:
            direction = "bullish"
        elif bearish > bullish and bearish > neutral:
            direction = "bearish"
        else:
            direction = "mixed"

        # Confidence based on agreement
        if agreement_ratio > 0.8:
            confidence = 0.9
        elif agreement_ratio > 0.6:
            confidence = 0.7
        elif agreement_ratio > 0.4:
            confidence = 0.5
        else:
            confidence = 0.3

        return SignalAgreement(
            timestamp=datetime.now().isoformat(),
            total_strategies=total,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            agreement_ratio=agreement_ratio,
            consensus_direction=direction,
            confidence=confidence,
        )

    def should_reduce_on_consensus(self) -> Tuple[bool, str]:
        """
        Check if position should be reduced due to over-consensus.

        High consensus + high correlation = dangerous crowding.

        Returns:
            Tuple of (should_reduce, reason)
        """
        agreement = self.analyze_signal_agreement()

        if agreement.agreement_ratio > 0.8 and self.is_clustering():
            return True, (
                f"High consensus ({agreement.agreement_ratio:.0%}) "
                f"combined with clustering - reduce exposure"
            )

        if agreement.agreement_ratio > 0.9:
            return True, (
                f"Very high consensus ({agreement.agreement_ratio:.0%}) "
                f"- market may be crowded"
            )

        return False, "Normal consensus levels"

    # ========================================================================
    # PORTFOLIO OPTIMIZATION
    # ========================================================================

    def optimize_weights(
        self,
        expected_returns: Dict[str, float] = None,
        risk_aversion: float = 1.0,
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using correlation information.

        Simple mean-variance optimization with correlation penalty.

        Args:
            expected_returns: Expected return per strategy (optional)
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Optimized weights dict
        """
        if self.current_correlations is None or len(self.base_weights) < 2:
            return self.base_weights.copy()

        strategies = list(self.base_weights.keys())
        n = len(strategies)

        # Build covariance matrix from correlations and assumed volatilities
        corr_matrix = self.current_correlations.matrix
        available = [s for s in strategies if s in corr_matrix.columns]

        if len(available) < 2:
            return self.base_weights.copy()

        corr = corr_matrix.loc[available, available].values

        # Assume equal volatility for simplicity (or could use historical)
        vol = 0.15  # 15% annual volatility assumption
        cov = corr * (vol ** 2)

        # Expected returns (default: base weight * scaling factor)
        if expected_returns:
            mu = np.array([expected_returns.get(s, 0.1) for s in available])
        else:
            mu = np.array([0.1] * len(available))  # 10% expected return

        # Simple optimization: maximize Sharpe-like ratio
        # w* = (1/gamma) * Cov^-1 * mu
        try:
            cov_inv = np.linalg.inv(cov + np.eye(len(available)) * 0.01)  # Regularization
            raw_weights = cov_inv @ mu / risk_aversion

            # Normalize and enforce bounds
            raw_weights = np.maximum(raw_weights, 0.05)  # Min 5%
            raw_weights = np.minimum(raw_weights, self.config.max_single_strategy_weight)
            weights = raw_weights / raw_weights.sum()

            return {s: float(w) for s, w in zip(available, weights)}

        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using equal weights")
            return {s: 1/len(available) for s in available}

    # ========================================================================
    # REPORTING
    # ========================================================================

    def print_status(self):
        """Print ensemble status."""
        print("\n" + "=" * 70)
        print("STRATEGY ENSEMBLE STATUS")
        print("=" * 70)

        if self.current_correlations is None:
            print("\nNo correlation data available yet.")
            print("Call update_returns() with daily strategy returns.")
            print("=" * 70)
            return

        corr = self.current_correlations

        print(f"\nCorrelation Summary:")
        print(f"  Average: {corr.avg_correlation:.3f}")
        print(f"  Range: [{corr.min_correlation:.3f}, {corr.max_correlation:.3f}]")
        print(f"  Clustering Score: {corr.clustering_score:.3f}")
        print(f"  Risk Level: {self.get_cluster_risk_level().upper()}")

        print(f"\nCorrelation Matrix:")
        print(corr.matrix.round(2).to_string())

        print(f"\nDiversification Scores:")
        for strategy in self.returns_history.keys():
            score = self.get_strategy_diversification_score(strategy)
            print(f"  {strategy}: {score:.2f}")

        if self.base_weights:
            print(f"\nAllocation Adjustments:")
            adjustments = self.get_allocation_adjustments()
            for strat, adj in adjustments.items():
                print(f"  {strat}:")
                print(f"    Base: {adj.current_weight:.1%} -> Recommended: {adj.recommended_weight:.1%}")
                print(f"    Factor: {adj.adjustment_factor:.2f} ({adj.reason})")

        exposure = self.get_exposure_multiplier()
        print(f"\nOverall Exposure Multiplier: {exposure:.2f}")

        if self.current_signals:
            agreement = self.analyze_signal_agreement()
            print(f"\nSignal Agreement:")
            print(f"  Bullish: {agreement.bullish_count}/{agreement.total_strategies}")
            print(f"  Bearish: {agreement.bearish_count}/{agreement.total_strategies}")
            print(f"  Consensus: {agreement.consensus_direction} ({agreement.agreement_ratio:.0%})")

        if self.cluster_alerts:
            recent = self.cluster_alerts[-3:]
            print(f"\nRecent Cluster Alerts:")
            for alert in recent:
                print(f"  [{alert.risk_level}] {alert.timestamp[:16]}: {alert.recommended_action}")

        print("\n" + "=" * 70)

    def get_status_dict(self) -> Dict[str, Any]:
        """Get status as dictionary for API/dashboard."""
        if self.current_correlations is None:
            return {'status': 'no_data'}

        return {
            'avg_correlation': self.current_correlations.avg_correlation,
            'clustering_score': self.current_correlations.clustering_score,
            'risk_level': self.get_cluster_risk_level(),
            'exposure_multiplier': self.get_exposure_multiplier(),
            'is_clustering': self.is_clustering(),
            'diversification_scores': {
                s: self.get_strategy_diversification_score(s)
                for s in self.returns_history.keys()
            },
            'correlation_matrix': self.current_correlations.matrix.to_dict(),
        }


# ============================================================================
# FACTORY
# ============================================================================

_ensemble: Optional[StrategyEnsemble] = None

def get_ensemble() -> StrategyEnsemble:
    """Get or create global ensemble intelligence."""
    global _ensemble
    if _ensemble is None:
        _ensemble = StrategyEnsemble()
    return _ensemble


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("STRATEGY ENSEMBLE INTELLIGENCE DEMO")
    print("=" * 60)

    ensemble = StrategyEnsemble()

    # Set base weights
    ensemble.set_base_weights({
        'momentum': 0.3,
        'mean_reversion': 0.25,
        'pairs_trading': 0.25,
        'sector_rotation': 0.2,
    })

    # Simulate 30 days of returns
    np.random.seed(42)
    print("\nSimulating 30 days of returns...")

    for day in range(30):
        # Simulate correlated returns
        base_return = np.random.normal(0, 0.01)

        returns = {
            'momentum': base_return + np.random.normal(0, 0.005),
            'mean_reversion': -base_return * 0.5 + np.random.normal(0, 0.008),  # Negative correlation
            'pairs_trading': np.random.normal(0, 0.003),  # Low correlation
            'sector_rotation': base_return * 0.7 + np.random.normal(0, 0.006),  # High correlation with momentum
        }

        ensemble.update_returns(
            returns,
            date=datetime.now() - timedelta(days=30-day)
        )

    # Update signals
    ensemble.update_signals({
        'momentum': 'bullish',
        'mean_reversion': 'bearish',
        'pairs_trading': 'neutral',
        'sector_rotation': 'bullish',
    })

    # Print status
    ensemble.print_status()

    # Test specific features
    print("\nPairwise Correlations:")
    pairs = [
        ('momentum', 'sector_rotation'),
        ('momentum', 'mean_reversion'),
        ('pairs_trading', 'mean_reversion'),
    ]
    for s1, s2 in pairs:
        corr = ensemble.get_pairwise_correlation(s1, s2)
        print(f"  {s1} vs {s2}: {corr:.3f}")

    # Optimize weights
    print("\nOptimized Weights:")
    optimized = ensemble.optimize_weights(risk_aversion=2.0)
    for strat, weight in sorted(optimized.items(), key=lambda x: -x[1]):
        print(f"  {strat}: {weight:.1%}")

    # Check consensus
    reduce, reason = ensemble.should_reduce_on_consensus()
    print(f"\nShould reduce on consensus: {reduce}")
    print(f"Reason: {reason}")
