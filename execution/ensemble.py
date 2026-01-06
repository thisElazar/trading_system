"""
Strategy Ensemble & Capital Allocator
=====================================
Coordinates multiple strategies into unified portfolio decisions.

Responsibilities:
- Aggregate signals from multiple strategies
- Resolve conflicts (same symbol, opposite directions)
- Allocate capital across strategies
- Enforce portfolio-level risk limits
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import pandas as pd
import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import Signal, SignalType
from config import MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT, MAX_POSITION_PCT, TOTAL_CAPITAL

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    EQUAL = "equal"              # Equal weight to all strategies
    RISK_PARITY = "risk_parity"  # Weight by inverse volatility
    PERFORMANCE = "performance"  # Weight by recent Sharpe
    FIXED = "fixed"              # User-defined weights


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy."""
    strategy_name: str
    weight: float           # 0.0 - 1.0
    capital: float          # Dollar amount
    max_positions: int      # Position limit for this strategy
    current_positions: int = 0
    realized_pnl: float = 0.0
    sharpe_30d: float = 0.0


@dataclass
class AggregatedSignal:
    """Signal after ensemble aggregation."""
    symbol: str
    direction: str          # 'long', 'short', 'close'
    strength: float         # Combined confidence 0-1
    sources: List[str]      # Contributing strategies
    capital_allocation: float
    conflicts: List[str] = field(default_factory=list)  # Conflicting strategies


class ConflictResolver:
    """
    Resolves conflicts when multiple strategies disagree.
    
    Resolution methods:
    - CONFIDENCE: Higher confidence wins
    - MAJORITY: Most strategies agreeing wins
    - HIERARCHY: Predefined strategy priority
    - CANCEL: Conflicting signals cancel out
    """
    
    def __init__(self, method: str = 'confidence'):
        self.method = method
        
        # Strategy priority (higher = more trusted)
        self.priority = {
            'pairs_trading': 5,      # Highest - statistical edge
            'gap_fill': 4,
            'vol_managed_momentum': 3,
            'relative_volume_breakout': 2,
            'vix_regime_rotation': 1,
        }
    
    def resolve(
        self, 
        symbol: str, 
        signals: List[Tuple[str, Signal]]  # [(strategy_name, signal), ...]
    ) -> Optional[AggregatedSignal]:
        """
        Resolve conflicting signals for a symbol.
        
        Returns:
            AggregatedSignal or None if signals cancel out
        """
        if not signals:
            return None
        
        # Group by direction
        longs = [(name, sig) for name, sig in signals if sig.signal_type == SignalType.BUY]
        shorts = [(name, sig) for name, sig in signals if sig.signal_type == SignalType.SELL]
        closes = [(name, sig) for name, sig in signals if sig.signal_type == SignalType.CLOSE]
        
        # Handle closes first
        if closes:
            return AggregatedSignal(
                symbol=symbol,
                direction='close',
                strength=1.0,
                sources=[name for name, _ in closes],
                capital_allocation=0,
                conflicts=[]
            )
        
        # No conflict
        if longs and not shorts:
            return self._aggregate_same_direction(symbol, 'long', longs)
        if shorts and not longs:
            return self._aggregate_same_direction(symbol, 'short', shorts)
        
        # Conflict - resolve based on method
        if self.method == 'cancel':
            logger.info(f"Conflict on {symbol}: {len(longs)} long vs {len(shorts)} short - cancelling")
            return None
        
        elif self.method == 'majority':
            if len(longs) > len(shorts):
                return self._aggregate_same_direction(
                    symbol, 'long', longs, 
                    conflicts=[name for name, _ in shorts]
                )
            elif len(shorts) > len(longs):
                return self._aggregate_same_direction(
                    symbol, 'short', shorts,
                    conflicts=[name for name, _ in longs]
                )
            else:
                return None  # Tie
        
        elif self.method == 'hierarchy':
            long_priority = max(self.priority.get(name, 0) for name, _ in longs)
            short_priority = max(self.priority.get(name, 0) for name, _ in shorts)
            
            if long_priority > short_priority:
                return self._aggregate_same_direction(
                    symbol, 'long', longs,
                    conflicts=[name for name, _ in shorts]
                )
            else:
                return self._aggregate_same_direction(
                    symbol, 'short', shorts,
                    conflicts=[name for name, _ in longs]
                )
        
        else:  # confidence
            long_conf = max(sig.strength for _, sig in longs)
            short_conf = max(sig.strength for _, sig in shorts)
            
            if long_conf > short_conf:
                return self._aggregate_same_direction(
                    symbol, 'long', longs,
                    conflicts=[name for name, _ in shorts]
                )
            else:
                return self._aggregate_same_direction(
                    symbol, 'short', shorts,
                    conflicts=[name for name, _ in longs]
                )
    
    def _aggregate_same_direction(
        self, 
        symbol: str, 
        direction: str,
        signals: List[Tuple[str, Signal]],
        conflicts: List[str] = None
    ) -> AggregatedSignal:
        """Aggregate signals in same direction."""
        strengths = [sig.strength for _, sig in signals]
        avg_strength = np.mean(strengths)
        
        # Boost strength if multiple strategies agree
        agreement_boost = min(len(signals) * 0.1, 0.3)  # Up to 30% boost
        combined_strength = min(avg_strength + agreement_boost, 1.0)
        
        return AggregatedSignal(
            symbol=symbol,
            direction=direction,
            strength=combined_strength,
            sources=[name for name, _ in signals],
            capital_allocation=0,  # Set by allocator
            conflicts=conflicts or []
        )


@dataclass
class RiskStatus:
    """Current risk status for the trading system."""
    daily_pnl_pct: float = 0.0           # Today's P&L as percentage
    current_drawdown_pct: float = 0.0    # Current drawdown from peak
    trading_allowed: bool = True          # Whether trading is allowed
    position_size_multiplier: float = 1.0 # Multiplier for position sizes
    reason: str = ""                      # Reason if trading paused


class RiskController:
    """
    Portfolio-level risk controls.

    Monitors daily P&L and drawdown to:
    - Pause trading if daily loss exceeds MAX_DAILY_LOSS_PCT
    - Reduce position sizes by 50% if drawdown exceeds MAX_DRAWDOWN_PCT
    """

    def __init__(
        self,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        # State tracking
        self.daily_pnl_pct = 0.0
        self.current_drawdown_pct = 0.0
        self.peak_capital = 0.0
        self.start_of_day_capital = 0.0

    def update_capital(self, current_capital: float, start_of_day_capital: float = None):
        """
        Update capital levels for risk calculations.

        Args:
            current_capital: Current total portfolio value
            start_of_day_capital: Portfolio value at start of day (optional)
        """
        if start_of_day_capital is not None:
            self.start_of_day_capital = start_of_day_capital

        # Update peak capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        # Calculate daily P&L
        if self.start_of_day_capital > 0:
            self.daily_pnl_pct = (current_capital - self.start_of_day_capital) / self.start_of_day_capital
        else:
            self.daily_pnl_pct = 0.0

        # Calculate drawdown from peak
        if self.peak_capital > 0:
            self.current_drawdown_pct = (self.peak_capital - current_capital) / self.peak_capital
        else:
            self.current_drawdown_pct = 0.0

    def reset_daily(self, current_capital: float):
        """Reset for new trading day."""
        self.start_of_day_capital = current_capital
        self.daily_pnl_pct = 0.0

    def check_risk_status(self) -> RiskStatus:
        """
        Check current risk status and return trading permissions.

        Returns:
            RiskStatus with trading_allowed and position_size_multiplier
        """
        status = RiskStatus(
            daily_pnl_pct=self.daily_pnl_pct,
            current_drawdown_pct=self.current_drawdown_pct,
            trading_allowed=True,
            position_size_multiplier=1.0,
            reason=""
        )

        # Check daily loss limit
        if self.daily_pnl_pct < -self.max_daily_loss_pct:
            status.trading_allowed = False
            status.reason = (
                f"Daily loss {self.daily_pnl_pct:.2%} exceeds limit "
                f"{-self.max_daily_loss_pct:.2%} - trading paused"
            )
            logger.warning(status.reason)
            return status

        # Check drawdown - reduce position sizes if exceeded
        if self.current_drawdown_pct > self.max_drawdown_pct:
            status.position_size_multiplier = 0.5
            status.reason = (
                f"Drawdown {self.current_drawdown_pct:.2%} exceeds limit "
                f"{self.max_drawdown_pct:.2%} - position sizes reduced by 50%"
            )
            logger.warning(status.reason)

        return status

    def should_skip_signals(self) -> bool:
        """Quick check if we should skip signal generation entirely."""
        return self.daily_pnl_pct < -self.max_daily_loss_pct

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on current drawdown."""
        if self.current_drawdown_pct > self.max_drawdown_pct:
            return 0.5
        return 1.0


class CapitalAllocator:
    """
    Allocates capital across strategies and positions.
    """

    def __init__(
        self,
        total_capital: float,
        method: AllocationMethod = AllocationMethod.EQUAL,
        max_position_pct: float = MAX_POSITION_PCT,  # Max 5% per position (from config)
        max_strategy_pct: float = 0.40,              # Max 40% per strategy
        reserve_pct: float = 0.10                    # Keep 10% cash reserve
    ):
        self.total_capital = total_capital
        self.method = method
        self.max_position_pct = max_position_pct
        self.max_strategy_pct = max_strategy_pct
        self.reserve_pct = reserve_pct
        
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.fixed_weights: Dict[str, float] = {}
    
    def set_fixed_weights(self, weights: Dict[str, float]):
        """Set fixed strategy weights (must sum to <= 1.0)."""
        total = sum(weights.values())
        if total > 1.0:
            raise ValueError(f"Weights sum to {total}, must be <= 1.0")
        self.fixed_weights = weights
    
    def update_performance(self, strategy_name: str, sharpe_30d: float, realized_pnl: float):
        """Update strategy performance for dynamic allocation."""
        if strategy_name in self.allocations:
            self.allocations[strategy_name].sharpe_30d = sharpe_30d
            self.allocations[strategy_name].realized_pnl = realized_pnl
    
    def calculate_allocations(self, active_strategies: List[str]) -> Dict[str, StrategyAllocation]:
        """
        Calculate capital allocation for each strategy.
        """
        deployable = self.total_capital * (1 - self.reserve_pct)
        n_strategies = len(active_strategies)
        
        if n_strategies == 0:
            return {}
        
        if self.method == AllocationMethod.FIXED:
            weights = {s: self.fixed_weights.get(s, 0) for s in active_strategies}
        
        elif self.method == AllocationMethod.EQUAL:
            equal_weight = 1.0 / n_strategies
            weights = {s: equal_weight for s in active_strategies}
        
        elif self.method == AllocationMethod.PERFORMANCE:
            # Weight by recent Sharpe (positive only)
            sharpes = {}
            for s in active_strategies:
                if s in self.allocations:
                    sharpes[s] = max(self.allocations[s].sharpe_30d, 0.1)  # Floor at 0.1
                else:
                    sharpes[s] = 0.5  # Default for new strategies
            
            total_sharpe = sum(sharpes.values())
            weights = {s: sharpe / total_sharpe for s, sharpe in sharpes.items()}
        
        else:  # RISK_PARITY - placeholder, needs volatility data
            weights = {s: 1.0 / n_strategies for s in active_strategies}
        
        # Cap at max_strategy_pct
        for s in weights:
            weights[s] = min(weights[s], self.max_strategy_pct)
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}
        
        # Create allocations
        self.allocations = {}
        for strategy in active_strategies:
            weight = weights.get(strategy, 0)
            capital = deployable * weight
            max_positions = max(1, int(capital / (self.total_capital * self.max_position_pct)))
            
            self.allocations[strategy] = StrategyAllocation(
                strategy_name=strategy,
                weight=weight,
                capital=capital,
                max_positions=max_positions
            )
        
        return self.allocations
    
    def get_position_size(self, strategy: str, signal_strength: float = 1.0) -> float:
        """Get dollar amount for a position."""
        if strategy not in self.allocations:
            return 0
        
        alloc = self.allocations[strategy]
        max_position = self.total_capital * self.max_position_pct
        strategy_available = alloc.capital / max(1, alloc.max_positions - alloc.current_positions)
        
        base_size = min(max_position, strategy_available)
        return base_size * signal_strength


class StrategyEnsemble:
    """
    Coordinates multiple strategies into unified portfolio decisions.

    Usage:
        ensemble = StrategyEnsemble()  # Uses TOTAL_CAPITAL from config
        ensemble.register_strategy('pairs_trading', pairs_strategy)
        ensemble.register_strategy('gap_fill', gap_strategy)

        # Collect signals
        ensemble.add_signals('pairs_trading', pairs_signals)
        ensemble.add_signals('gap_fill', gap_signals)

        # Get aggregated portfolio decisions
        decisions = ensemble.get_portfolio_decisions()
    """
    
    def __init__(
        self,
        total_capital: float = None,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL,
        conflict_method: str = 'confidence',
        max_portfolio_positions: int = 10
    ):
        # Use broker-provided capital or fall back to config
        if total_capital is None:
            total_capital = TOTAL_CAPITAL
        self.total_capital = total_capital
        self.max_positions = max_portfolio_positions

        self.allocator = CapitalAllocator(total_capital, allocation_method)
        self.resolver = ConflictResolver(conflict_method)
        self.risk_controller = RiskController()

        # Initialize risk controller with starting capital
        self.risk_controller.update_capital(total_capital, start_of_day_capital=total_capital)

        self.strategies: Dict[str, any] = {}  # strategy_name -> strategy instance
        self.pending_signals: Dict[str, List[Signal]] = defaultdict(list)
        self.current_positions: Dict[str, str] = {}  # symbol -> strategy
    
    def register_strategy(self, name: str, strategy=None, enabled: bool = True):
        """Register a strategy with the ensemble."""
        self.strategies[name] = {
            'instance': strategy,
            'enabled': enabled,
            'signals': []
        }
        logger.info(f"Registered strategy: {name}")
    
    def add_signals(self, strategy_name: str, signals: List[Signal]):
        """Add signals from a strategy."""
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return
        
        if not self.strategies[strategy_name]['enabled']:
            return
        
        self.pending_signals[strategy_name] = signals
        logger.debug(f"{strategy_name}: {len(signals)} signals")
    
    def update_capital(self, current_capital: float, start_of_day_capital: float = None):
        """
        Update capital levels for risk tracking.

        Should be called periodically to track P&L and drawdown.
        """
        self.total_capital = current_capital
        self.allocator.total_capital = current_capital
        self.risk_controller.update_capital(current_capital, start_of_day_capital)

    def reset_daily(self, current_capital: float):
        """Reset for new trading day."""
        self.risk_controller.reset_daily(current_capital)

    def get_risk_status(self) -> RiskStatus:
        """Get current risk status."""
        return self.risk_controller.check_risk_status()

    def get_portfolio_decisions(self) -> List[AggregatedSignal]:
        """
        Process all pending signals and return portfolio decisions.

        Applies risk controls:
        - If daily P&L < -MAX_DAILY_LOSS_PCT, returns empty list (trading paused)
        - If drawdown > MAX_DRAWDOWN_PCT, reduces position sizes by 50%

        Returns:
            List of AggregatedSignal with capital allocations
        """
        # Check risk status first
        risk_status = self.risk_controller.check_risk_status()

        if not risk_status.trading_allowed:
            logger.warning(f"Trading paused: {risk_status.reason}")
            self.pending_signals.clear()
            return []

        # Update allocations
        active = [name for name, s in self.strategies.items() if s['enabled']]
        self.allocator.calculate_allocations(active)

        # Group signals by symbol
        symbol_signals: Dict[str, List[Tuple[str, Signal]]] = defaultdict(list)

        for strategy_name, signals in self.pending_signals.items():
            for signal in signals:
                symbol_signals[signal.symbol].append((strategy_name, signal))

        # Resolve and aggregate
        decisions = []

        for symbol, signals in symbol_signals.items():
            # Skip if we're at position limit
            if (len(self.current_positions) >= self.max_positions and
                symbol not in self.current_positions):
                continue

            aggregated = self.resolver.resolve(symbol, signals)

            if aggregated:
                # Determine capital allocation
                # Use highest-priority contributing strategy
                primary_strategy = max(
                    aggregated.sources,
                    key=lambda s: self.resolver.priority.get(s, 0)
                )

                position_size = self.allocator.get_position_size(
                    primary_strategy,
                    aggregated.strength
                )

                # Apply risk-based position size reduction
                position_size *= risk_status.position_size_multiplier
                aggregated.capital_allocation = position_size

                decisions.append(aggregated)

        # Sort by strength (highest first)
        decisions.sort(key=lambda x: x.strength, reverse=True)

        # Clear pending signals
        self.pending_signals.clear()

        return decisions
    
    def update_position(self, symbol: str, strategy: str = None):
        """Track that we opened a position."""
        if strategy:
            self.current_positions[symbol] = strategy
            if strategy in self.allocator.allocations:
                self.allocator.allocations[strategy].current_positions += 1
    
    def close_position(self, symbol: str):
        """Track that we closed a position."""
        if symbol in self.current_positions:
            strategy = self.current_positions.pop(symbol)
            if strategy in self.allocator.allocations:
                self.allocator.allocations[strategy].current_positions -= 1
    
    def get_status(self) -> dict:
        """Get ensemble status."""
        risk_status = self.risk_controller.check_risk_status()
        return {
            'total_capital': self.total_capital,
            'strategies': {
                name: {
                    'enabled': s['enabled'],
                    'allocation': self.allocator.allocations.get(name, {})
                }
                for name, s in self.strategies.items()
            },
            'current_positions': len(self.current_positions),
            'max_positions': self.max_positions,
            'positions': dict(self.current_positions),
            'risk': {
                'daily_pnl_pct': risk_status.daily_pnl_pct,
                'current_drawdown_pct': risk_status.current_drawdown_pct,
                'trading_allowed': risk_status.trading_allowed,
                'position_size_multiplier': risk_status.position_size_multiplier,
                'reason': risk_status.reason
            }
        }
    
    def summary(self) -> str:
        """Print summary of ensemble state."""
        risk_status = self.risk_controller.check_risk_status()

        lines = ["=" * 50, "STRATEGY ENSEMBLE STATUS", "=" * 50]
        lines.append(f"Capital: ${self.total_capital:,.0f}")
        lines.append(f"Positions: {len(self.current_positions)}/{self.max_positions}")
        lines.append("")

        # Risk status section
        lines.append("Risk Status:")
        lines.append(f"  Daily P&L: {risk_status.daily_pnl_pct:+.2%}")
        lines.append(f"  Drawdown: {risk_status.current_drawdown_pct:.2%}")
        trading_status = "ALLOWED" if risk_status.trading_allowed else "PAUSED"
        lines.append(f"  Trading: {trading_status}")
        if risk_status.position_size_multiplier < 1.0:
            lines.append(f"  Position Size: {risk_status.position_size_multiplier:.0%} (reduced)")
        if risk_status.reason:
            lines.append(f"  Note: {risk_status.reason}")
        lines.append("")

        lines.append("Strategy Allocations:")

        for name, alloc in self.allocator.allocations.items():
            enabled = "[+]" if self.strategies.get(name, {}).get('enabled') else "[-]"
            lines.append(f"  {enabled} {name}: ${alloc.capital:,.0f} ({alloc.weight:.1%})")

        if self.current_positions:
            lines.append("")
            lines.append("Open Positions:")
            for symbol, strategy in self.current_positions.items():
                lines.append(f"  {symbol} ({strategy})")

        return "\n".join(lines)


# Convenience function
def create_ensemble(capital: float = None) -> StrategyEnsemble:
    """Create ensemble with default strategies. Uses TOTAL_CAPITAL from config if not specified."""
    ensemble = StrategyEnsemble(total_capital=capital)
    
    # Register all known strategies
    for name in ['gap_fill', 'pairs_trading', 'relative_volume_breakout',
                 'vol_managed_momentum', 'vix_regime_rotation']:
        ensemble.register_strategy(name, enabled=True)
    
    return ensemble


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo - uses TOTAL_CAPITAL from config
    ensemble = create_ensemble()
    print(ensemble.summary())
    
    # Simulate some signals
    from strategies.base import Signal, SignalType
    
    # Gap-fill wants to buy AAPL
    gap_signals = [
        Signal(timestamp=datetime.now(), symbol='AAPL', strategy='gap_fill',
               signal_type=SignalType.BUY, price=250, strength=0.7)
    ]
    
    # Pairs trading also wants to buy AAPL
    pairs_signals = [
        Signal(timestamp=datetime.now(), symbol='AAPL', strategy='pairs_trading',
               signal_type=SignalType.BUY, price=250, strength=0.8),
        Signal(timestamp=datetime.now(), symbol='MSFT', strategy='pairs_trading',
               signal_type=SignalType.SELL, price=430, strength=0.75)
    ]
    
    ensemble.add_signals('gap_fill', gap_signals)
    ensemble.add_signals('pairs_trading', pairs_signals)
    
    decisions = ensemble.get_portfolio_decisions()
    
    print("\nPortfolio Decisions:")
    for d in decisions:
        print(f"  {d.direction.upper()} {d.symbol} | strength={d.strength:.2f} | "
              f"${d.capital_allocation:,.0f} | sources={d.sources}")
