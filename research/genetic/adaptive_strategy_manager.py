"""
Adaptive Strategy Manager
=========================
Real-time strategy allocation and rebalancing based on market conditions.

Watches the market and dynamically adjusts:
- Strategy allocations based on regime
- Individual strategy parameters
- Risk exposure levels
- Which strategies are active

Integrates with:
- RegimeMatchingEngine for condition detection
- MultiScaleFitnessCalculator for strategy evaluation
- AdaptiveGAOptimizer for parameter evolution
- RapidBacktester for quick validation

Usage:
    from research.genetic.adaptive_strategy_manager import AdaptiveStrategyManager

    manager = AdaptiveStrategyManager()

    # Get current recommendations
    allocations = manager.get_strategy_allocations()

    # Rebalance based on conditions
    manager.rebalance()
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .market_periods import MarketPeriodLibrary, PeriodType
from .regime_matching import RegimeMatchingEngine, RegimeFingerprint
from .rapid_backtester import RapidBacktester
from .multiscale_fitness import MultiScaleFitnessCalculator

logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """Reasons for rebalancing."""
    SCHEDULED = "scheduled"          # Regular rebalance
    REGIME_CHANGE = "regime_change"  # Market regime changed
    VIX_SPIKE = "vix_spike"         # VIX moved significantly
    DRAWDOWN = "drawdown"           # Strategy drawdown exceeded threshold
    PERFORMANCE = "performance"      # Performance triggered review
    MANUAL = "manual"               # Manual override


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy."""
    strategy_name: str
    base_allocation: float        # Base allocation %
    adjusted_allocation: float    # After regime adjustment
    regime_multiplier: float      # Current regime multiplier
    is_active: bool               # Is strategy active?
    confidence: float             # Confidence in allocation (0-1)
    last_updated: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'base_allocation': self.base_allocation,
            'adjusted_allocation': self.adjusted_allocation,
            'regime_multiplier': self.regime_multiplier,
            'is_active': self.is_active,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat(),
            'notes': self.notes,
        }


@dataclass
class PortfolioState:
    """Current state of the strategy portfolio."""
    regime: str
    regime_confidence: float
    vix_level: float
    allocations: List[StrategyAllocation]
    total_risk_exposure: float    # 0-1
    cash_allocation: float        # Reserved cash %
    rebalance_reason: Optional[RebalanceReason] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'vix_level': self.vix_level,
            'allocations': [a.to_dict() for a in self.allocations],
            'total_risk_exposure': self.total_risk_exposure,
            'cash_allocation': self.cash_allocation,
            'rebalance_reason': self.rebalance_reason.value if self.rebalance_reason else None,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AdaptiveManagerConfig:
    """Configuration for the adaptive manager."""
    # Base strategy allocations (sum to 1.0)
    base_allocations: Dict[str, float] = field(default_factory=lambda: {
        'vol_managed_momentum': 0.15,
        'mean_reversion': 0.20,
        'pairs_trading': 0.10,
        'relative_volume_breakout': 0.10,
        'gap_fill': 0.05,
        'vix_regime_rotation': 0.15,
        'sector_rotation': 0.10,
        'factor_momentum': 0.15,
    })

    # Regime multipliers for each strategy
    regime_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'vol_managed_momentum': {
            'risk_on': 1.3, 'transition': 0.9, 'risk_off': 0.5, 'crisis': 0.2
        },
        'mean_reversion': {
            'risk_on': 0.8, 'transition': 1.0, 'risk_off': 1.2, 'crisis': 0.8
        },
        'pairs_trading': {
            'risk_on': 0.7, 'transition': 1.0, 'risk_off': 1.3, 'crisis': 0.9
        },
        'relative_volume_breakout': {
            'risk_on': 1.2, 'transition': 0.9, 'risk_off': 0.5, 'crisis': 0.1
        },
        'gap_fill': {
            'risk_on': 1.0, 'transition': 0.8, 'risk_off': 0.6, 'crisis': 0.0
        },
        'vix_regime_rotation': {
            'risk_on': 0.6, 'transition': 1.0, 'risk_off': 1.4, 'crisis': 1.5
        },
        'sector_rotation': {
            'risk_on': 1.1, 'transition': 0.9, 'risk_off': 0.7, 'crisis': 0.3
        },
        'factor_momentum': {
            'risk_on': 1.2, 'transition': 0.9, 'risk_off': 0.6, 'crisis': 0.2
        },
    })

    # VIX-based adjustments
    vix_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 15,      # VIX < 15: low vol environment
        'normal': 20,   # VIX 15-20: normal
        'elevated': 25, # VIX 20-25: elevated
        'high': 30,     # VIX 25-30: high
        'extreme': 40,  # VIX > 40: extreme
    })

    # Risk exposure limits by VIX level
    exposure_limits: Dict[str, float] = field(default_factory=lambda: {
        'low': 1.0,      # Full exposure in low vol
        'normal': 0.9,
        'elevated': 0.7,
        'high': 0.5,
        'extreme': 0.25,  # Minimal exposure in extreme vol
    })

    # Rebalance triggers
    min_rebalance_interval_hours: int = 4
    regime_change_triggers_rebalance: bool = True
    vix_spike_threshold: float = 5.0  # VIX move that triggers review
    drawdown_threshold: float = 10.0  # Strategy DD that triggers review

    # Cash reserve
    min_cash_reserve: float = 0.05  # Always keep 5% cash
    crisis_cash_reserve: float = 0.30  # 30% cash in crisis


class AdaptiveStrategyManager:
    """
    Manager for adaptive strategy allocation and rebalancing.

    Monitors market conditions and adjusts strategy allocations
    in real-time based on regime changes and VIX levels.
    """

    def __init__(
        self,
        config: AdaptiveManagerConfig = None,
        regime_engine: RegimeMatchingEngine = None,
        data_manager: Any = None
    ):
        """
        Initialize the manager.

        Args:
            config: AdaptiveManagerConfig instance
            regime_engine: RegimeMatchingEngine instance
            data_manager: CachedDataManager for market data
        """
        self.config = config or AdaptiveManagerConfig()
        self.regime_engine = regime_engine or RegimeMatchingEngine()
        self.data_manager = data_manager

        # State
        self._current_state: Optional[PortfolioState] = None
        self._previous_state: Optional[PortfolioState] = None
        self._last_rebalance: Optional[datetime] = None
        self._previous_vix: Optional[float] = None

        # History
        self._state_history: List[PortfolioState] = []
        self._rebalance_history: List[Dict] = []

        logger.info("AdaptiveStrategyManager initialized")

    def _get_vix_level_category(self, vix: float) -> str:
        """Categorize VIX level."""
        if vix < self.config.vix_thresholds['low']:
            return 'low'
        elif vix < self.config.vix_thresholds['normal']:
            return 'normal'
        elif vix < self.config.vix_thresholds['elevated']:
            return 'elevated'
        elif vix < self.config.vix_thresholds['high']:
            return 'high'
        else:
            return 'extreme'

    def _get_exposure_limit(self, vix: float) -> float:
        """Get maximum risk exposure for current VIX."""
        category = self._get_vix_level_category(vix)
        return self.config.exposure_limits.get(category, 0.5)

    def _calculate_regime_multiplier(
        self,
        strategy_name: str,
        regime: str
    ) -> float:
        """Get regime multiplier for a strategy."""
        strategy_mults = self.config.regime_multipliers.get(strategy_name, {})
        return strategy_mults.get(regime, 1.0)

    def _calculate_allocations(
        self,
        fingerprint: RegimeFingerprint
    ) -> List[StrategyAllocation]:
        """Calculate strategy allocations based on current conditions."""
        regime = fingerprint.overall_regime
        vix = fingerprint.vix_level

        allocations = []
        total_adjusted = 0.0

        for strategy_name, base_alloc in self.config.base_allocations.items():
            # Get regime multiplier
            mult = self._calculate_regime_multiplier(strategy_name, regime)

            # Apply VIX-based adjustments
            if vix > self.config.vix_thresholds['extreme']:
                # In extreme vol, reduce most strategies
                if strategy_name not in ['vix_regime_rotation', 'pairs_trading']:
                    mult *= 0.5

            # Calculate adjusted allocation
            adjusted = base_alloc * mult

            # Determine if strategy should be active
            is_active = adjusted > 0.01  # 1% minimum

            # Confidence based on regime confidence and multiplier
            confidence = fingerprint.regime_confidence * min(1.0, mult)

            allocation = StrategyAllocation(
                strategy_name=strategy_name,
                base_allocation=base_alloc,
                adjusted_allocation=adjusted,
                regime_multiplier=mult,
                is_active=is_active,
                confidence=confidence,
            )

            allocations.append(allocation)
            total_adjusted += adjusted

        # Normalize to exposure limit
        exposure_limit = self._get_exposure_limit(vix)

        if total_adjusted > 0:
            scale_factor = min(1.0, exposure_limit / total_adjusted)
            for alloc in allocations:
                alloc.adjusted_allocation *= scale_factor

        return allocations

    def _calculate_cash_allocation(
        self,
        fingerprint: RegimeFingerprint,
        allocations: List[StrategyAllocation]
    ) -> float:
        """Calculate cash reserve."""
        regime = fingerprint.overall_regime

        # Base cash reserve
        if regime == 'crisis':
            cash = self.config.crisis_cash_reserve
        elif regime == 'risk_off':
            cash = self.config.min_cash_reserve * 2
        else:
            cash = self.config.min_cash_reserve

        # Add any remaining allocation gap
        total_allocated = sum(a.adjusted_allocation for a in allocations)
        remaining = 1.0 - total_allocated

        return max(cash, remaining)

    def _should_rebalance(
        self,
        new_fingerprint: RegimeFingerprint
    ) -> Tuple[bool, Optional[RebalanceReason]]:
        """Determine if rebalancing is needed."""
        # Time-based check
        if self._last_rebalance is not None:
            hours_since = (datetime.now() - self._last_rebalance).total_seconds() / 3600
            if hours_since < self.config.min_rebalance_interval_hours:
                return False, None

        # Regime change check
        if self._current_state is not None:
            if self._current_state.regime != new_fingerprint.overall_regime:
                if self.config.regime_change_triggers_rebalance:
                    return True, RebalanceReason.REGIME_CHANGE

        # VIX spike check
        if self._previous_vix is not None:
            vix_change = abs(new_fingerprint.vix_level - self._previous_vix)
            if vix_change >= self.config.vix_spike_threshold:
                return True, RebalanceReason.VIX_SPIKE

        # Initial state
        if self._current_state is None:
            return True, RebalanceReason.SCHEDULED

        return False, None

    def update(self, force: bool = False) -> PortfolioState:
        """
        Update portfolio state based on current conditions.

        Args:
            force: Force update regardless of rebalance rules

        Returns:
            Current PortfolioState
        """
        # Get current fingerprint
        fingerprint = self.regime_engine.get_current_fingerprint()

        # Check if we should rebalance
        should_rebalance, reason = self._should_rebalance(fingerprint)

        if force:
            should_rebalance = True
            reason = RebalanceReason.MANUAL

        if should_rebalance:
            # Store previous state
            self._previous_state = self._current_state

            # Calculate new allocations
            allocations = self._calculate_allocations(fingerprint)
            cash_allocation = self._calculate_cash_allocation(fingerprint, allocations)

            # Calculate total risk exposure
            total_exposure = sum(a.adjusted_allocation for a in allocations)

            # Create new state
            self._current_state = PortfolioState(
                regime=fingerprint.overall_regime,
                regime_confidence=fingerprint.regime_confidence,
                vix_level=fingerprint.vix_level,
                allocations=allocations,
                total_risk_exposure=total_exposure,
                cash_allocation=cash_allocation,
                rebalance_reason=reason,
            )

            # Update history
            self._last_rebalance = datetime.now()
            self._previous_vix = fingerprint.vix_level
            self._state_history.append(self._current_state)

            # Trim history
            if len(self._state_history) > 1000:
                self._state_history = self._state_history[-500:]

            # Log rebalance
            self._log_rebalance(reason, fingerprint)

        return self._current_state

    def _log_rebalance(
        self,
        reason: RebalanceReason,
        fingerprint: RegimeFingerprint
    ):
        """Log a rebalance event."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason.value,
            'regime': fingerprint.overall_regime,
            'vix': fingerprint.vix_level,
            'allocations': {
                a.strategy_name: a.adjusted_allocation
                for a in self._current_state.allocations
            },
        }

        self._rebalance_history.append(log_entry)

        logger.info(
            f"Rebalanced: reason={reason.value}, "
            f"regime={fingerprint.overall_regime}, "
            f"VIX={fingerprint.vix_level:.1f}, "
            f"exposure={self._current_state.total_risk_exposure:.1%}"
        )

    def get_strategy_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations as simple dict.

        Returns:
            Dict mapping strategy name to allocation %
        """
        if self._current_state is None:
            self.update()

        return {
            a.strategy_name: a.adjusted_allocation
            for a in self._current_state.allocations
        }

    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies."""
        if self._current_state is None:
            self.update()

        return [
            a.strategy_name
            for a in self._current_state.allocations
            if a.is_active
        ]

    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyAllocation]:
        """Get detailed status for a specific strategy."""
        if self._current_state is None:
            self.update()

        for alloc in self._current_state.allocations:
            if alloc.strategy_name == strategy_name:
                return alloc

        return None

    def rebalance(self, reason: str = None) -> PortfolioState:
        """
        Force a rebalance.

        Args:
            reason: Optional reason for the rebalance

        Returns:
            New PortfolioState
        """
        return self.update(force=True)

    def simulate_regime(self, regime: str) -> PortfolioState:
        """
        Simulate allocations for a hypothetical regime.

        Useful for planning and testing.

        Args:
            regime: One of 'risk_on', 'risk_off', 'transition', 'crisis'

        Returns:
            Simulated PortfolioState
        """
        # Create simulated fingerprint
        vix_levels = {
            'risk_on': 12,
            'transition': 18,
            'risk_off': 28,
            'crisis': 45,
        }

        fingerprint = RegimeFingerprint(
            vix_level=vix_levels.get(regime, 15),
            vix_percentile=50,
            vix_trend=0,
            trend_direction=0,
            trend_strength=0.5,
            momentum_breadth=50,
            realized_vol=15,
            vol_regime='normal',
            vol_trend=0,
            correlation_level=0.5,
            correlation_trend=0,
            sector_leadership='mixed',
            sector_dispersion=0.5,
            credit_spread_z=0,
            term_structure='contango',
            overall_regime=regime,
            regime_confidence=0.8,
        )

        allocations = self._calculate_allocations(fingerprint)
        cash = self._calculate_cash_allocation(fingerprint, allocations)
        total_exposure = sum(a.adjusted_allocation for a in allocations)

        return PortfolioState(
            regime=regime,
            regime_confidence=0.8,
            vix_level=fingerprint.vix_level,
            allocations=allocations,
            total_risk_exposure=total_exposure,
            cash_allocation=cash,
        )

    def get_transition_plan(
        self,
        from_regime: str,
        to_regime: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get allocation changes for a regime transition.

        Args:
            from_regime: Current regime
            to_regime: Target regime

        Returns:
            Dict with 'increase', 'decrease', 'unchanged' strategy lists
        """
        from_state = self.simulate_regime(from_regime)
        to_state = self.simulate_regime(to_regime)

        from_allocs = {a.strategy_name: a.adjusted_allocation for a in from_state.allocations}
        to_allocs = {a.strategy_name: a.adjusted_allocation for a in to_state.allocations}

        changes = {
            'increase': {},
            'decrease': {},
            'unchanged': {},
        }

        for strategy in from_allocs:
            from_val = from_allocs[strategy]
            to_val = to_allocs.get(strategy, 0)
            diff = to_val - from_val

            if diff > 0.01:
                changes['increase'][strategy] = {'from': from_val, 'to': to_val, 'change': diff}
            elif diff < -0.01:
                changes['decrease'][strategy] = {'from': from_val, 'to': to_val, 'change': diff}
            else:
                changes['unchanged'][strategy] = {'allocation': to_val}

        return changes

    def get_regime_history(self, days: int = 30) -> List[Dict]:
        """Get recent regime and allocation history."""
        cutoff = datetime.now() - timedelta(days=days)

        recent = [
            s.to_dict() for s in self._state_history
            if s.timestamp >= cutoff
        ]

        return recent

    def print_status(self):
        """Print current allocation status."""
        if self._current_state is None:
            self.update()

        state = self._current_state

        print("\n" + "=" * 60)
        print("ADAPTIVE STRATEGY MANAGER STATUS")
        print("=" * 60)

        print(f"\nCurrent Regime: {state.regime.upper()}")
        print(f"Regime Confidence: {state.regime_confidence:.0%}")
        print(f"VIX Level: {state.vix_level:.1f}")
        print(f"Total Risk Exposure: {state.total_risk_exposure:.1%}")
        print(f"Cash Reserve: {state.cash_allocation:.1%}")

        if state.rebalance_reason:
            print(f"Last Rebalance Reason: {state.rebalance_reason.value}")

        print("\nSTRATEGY ALLOCATIONS:")
        print("-" * 50)
        print(f"{'Strategy':<30} {'Base':>8} {'Adj':>8} {'Mult':>6} {'Active'}")
        print("-" * 50)

        for alloc in sorted(state.allocations, key=lambda x: -x.adjusted_allocation):
            status = "Yes" if alloc.is_active else "No"
            print(f"{alloc.strategy_name:<30} {alloc.base_allocation:>7.1%} "
                  f"{alloc.adjusted_allocation:>7.1%} {alloc.regime_multiplier:>5.2f}x  {status}")

        print("-" * 50)
        print(f"{'Total':<30} {sum(a.base_allocation for a in state.allocations):>7.1%} "
              f"{sum(a.adjusted_allocation for a in state.allocations):>7.1%}")

        print("=" * 60 + "\n")

    def print_regime_comparison(self):
        """Print allocation comparison across all regimes."""
        print("\n" + "=" * 70)
        print("ALLOCATION COMPARISON BY REGIME")
        print("=" * 70)

        regimes = ['risk_on', 'transition', 'risk_off', 'crisis']
        states = {r: self.simulate_regime(r) for r in regimes}

        # Header
        print(f"{'Strategy':<25}", end="")
        for regime in regimes:
            print(f" {regime:>10}", end="")
        print()
        print("-" * 70)

        # Strategy rows
        strategies = list(self.config.base_allocations.keys())
        for strategy in strategies:
            print(f"{strategy:<25}", end="")
            for regime in regimes:
                alloc = next(
                    a.adjusted_allocation
                    for a in states[regime].allocations
                    if a.strategy_name == strategy
                )
                print(f" {alloc:>9.1%}", end="")
            print()

        print("-" * 70)

        # Totals
        print(f"{'Total Exposure':<25}", end="")
        for regime in regimes:
            print(f" {states[regime].total_risk_exposure:>9.1%}", end="")
        print()

        print(f"{'Cash Reserve':<25}", end="")
        for regime in regimes:
            print(f" {states[regime].cash_allocation:>9.1%}", end="")
        print()

        print("=" * 70 + "\n")


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )

    print("\n" + "=" * 60)
    print("ADAPTIVE STRATEGY MANAGER DEMO")
    print("=" * 60)

    # Create manager
    manager = AdaptiveStrategyManager()

    # Show current status
    print("\nCurrent market conditions:")
    manager.print_status()

    # Show comparison
    manager.print_regime_comparison()

    # Simulate regime transition
    print("\nTransition plan: risk_on -> crisis")
    print("-" * 40)
    plan = manager.get_transition_plan('risk_on', 'crisis')

    print("INCREASE:")
    for strategy, info in plan['increase'].items():
        print(f"  {strategy}: {info['from']:.1%} -> {info['to']:.1%} ({info['change']:+.1%})")

    print("\nDECREASE:")
    for strategy, info in plan['decrease'].items():
        print(f"  {strategy}: {info['from']:.1%} -> {info['to']:.1%} ({info['change']:+.1%})")

    print("\nUNCHANGED:")
    for strategy, info in plan['unchanged'].items():
        print(f"  {strategy}: {info['allocation']:.1%}")

    print("\n" + "=" * 60)
