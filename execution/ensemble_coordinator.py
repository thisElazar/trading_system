"""
Ensemble Strategy Coordinator
=============================
Combines multiple strategies into a unified portfolio.

Features:
- Dynamic allocation based on strategy performance
- Regime-aware strategy weighting
- Risk budgeting across strategies
- Signal aggregation and conflict resolution
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import STRATEGIES, VALIDATION, VIX_REGIMES, TOTAL_CAPITAL
from core.types import Signal, Side, EnsembleResult

# Backward compatibility aliases
SignalType = Side
from strategies.vol_managed_momentum import VolManagedMomentumStrategy
from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
from strategies.sector_rotation import SectorRotationStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy."""
    name: str
    base_weight: float
    current_weight: float
    regime_multiplier: float = 1.0
    performance_multiplier: float = 1.0
    enabled: bool = True

    @property
    def effective_weight(self) -> float:
        if not self.enabled:
            return 0.0
        return self.current_weight * self.regime_multiplier * self.performance_multiplier


@dataclass
class EnsembleSignal:
    """
    DEPRECATED: Use EnsembleResult from core.types instead.

    Aggregated signal from ensemble.
    Kept for backward compatibility.
    """
    symbol: str
    direction: str  # 'BUY', 'SELL', 'CLOSE'
    combined_strength: float
    contributing_strategies: List[str]
    position_size_pct: float
    stop_loss: Optional[float] = None
    target: Optional[float] = None  # DEPRECATED: Use target_price
    metadata: Dict = field(default_factory=dict)

    @property
    def target_price(self) -> Optional[float]:
        """Canonical name for target."""
        return self.target

    @property
    def side(self) -> Side:
        """Canonical name for direction."""
        return Side(self.direction.upper())


class EnsembleCoordinator:
    """
    Coordinates multiple strategies into unified portfolio.
    
    Allocation approach:
    1. Base allocation from config
    2. Regime adjustment (momentum down in high VIX, etc.)
    3. Performance adjustment (reduce allocation to underperformers)
    4. Signal aggregation with conflict resolution
    """
    
    # Regime multipliers by strategy type
    REGIME_ADJUSTMENTS = {
        'vol_managed_momentum': {
            'low': 1.2,     # Momentum works well in low vol
            'normal': 1.0,
            'high': 0.6,    # Reduce in high vol (crash risk)
            'extreme': 0.3,
        },
        'vix_regime_rotation': {
            'low': 0.7,     # Less value in calm markets
            'normal': 1.0,
            'high': 1.3,    # Shines in volatility
            'extreme': 1.5,
        },
        'sector_rotation': {
            'low': 1.0,
            'normal': 1.0,
            'high': 1.2,
            'extreme': 1.0,
        },
        'pairs_trading': {
            'low': 0.8,
            'normal': 1.0,
            'high': 1.4,    # Mean reversion stronger in high vol
            'extreme': 1.0,
        },
        'relative_volume_breakout': {
            'low': 1.1,
            'normal': 1.0,
            'high': 0.8,
            'extreme': 0.4,
        },
    }
    
    def __init__(self, total_capital: float = None):
        # Use broker-provided capital or fall back to config
        if total_capital is None:
            total_capital = TOTAL_CAPITAL
        self.total_capital = total_capital
        self.current_regime = 'normal'
        
        # Strategy instances
        self.strategies = {
            'vol_managed_momentum': VolManagedMomentumStrategy(),
            'vix_regime_rotation': VIXRegimeRotationStrategy(),
            'sector_rotation': SectorRotationStrategy(),
        }
        
        # Allocations (from config or defaults)
        self.allocations = self._init_allocations()
        
        # Performance tracking for adaptive allocation
        self.strategy_performance: Dict[str, List[float]] = {
            name: [] for name in self.strategies
        }
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
    
    def _init_allocations(self) -> Dict[str, StrategyAllocation]:
        """Initialize allocations from config."""
        allocations = {}
        
        default_weights = {
            'vol_managed_momentum': 0.35,
            'vix_regime_rotation': 0.25,
            'sector_rotation': 0.20,
            'pairs_trading': 0.10,
            'relative_volume_breakout': 0.10,
        }
        
        for name in self.strategies:
            config = STRATEGIES.get(name, {})
            weight = config.get('allocation_pct', default_weights.get(name, 0.2))
            enabled = config.get('enabled', True)
            
            allocations[name] = StrategyAllocation(
                name=name,
                base_weight=weight,
                current_weight=weight,
                enabled=enabled
            )
        
        return allocations
    
    def update_regime(self, vix_value: float):
        """Update VIX regime and adjust allocations."""
        if vix_value < VIX_REGIMES['low']:
            regime = 'low'
        elif vix_value < VIX_REGIMES['high']:
            regime = 'normal'
        elif vix_value < VIX_REGIMES['extreme']:
            regime = 'high'
        else:
            regime = 'extreme'
        
        if regime != self.current_regime:
            logger.info(f"Regime change: {self.current_regime} -> {regime} (VIX={vix_value:.1f})")
            self.current_regime = regime
            self._apply_regime_adjustments()
    
    def _apply_regime_adjustments(self):
        """Apply regime-based multipliers."""
        for name, alloc in self.allocations.items():
            adjustments = self.REGIME_ADJUSTMENTS.get(name, {})
            alloc.regime_multiplier = adjustments.get(self.current_regime, 1.0)
    
    def update_performance(self, strategy_name: str, sharpe: float):
        """Update rolling performance for adaptive allocation."""
        if strategy_name not in self.strategy_performance:
            return
        
        perf_list = self.strategy_performance[strategy_name]
        perf_list.append(sharpe)
        
        # Keep last 12 periods
        if len(perf_list) > 12:
            perf_list.pop(0)
        
        # Calculate performance multiplier
        if len(perf_list) >= 3:
            avg_sharpe = np.mean(perf_list)
            
            if avg_sharpe > 1.0:
                mult = 1.2
            elif avg_sharpe > 0.5:
                mult = 1.0
            elif avg_sharpe > 0:
                mult = 0.8
            else:
                mult = 0.5
            
            self.allocations[strategy_name].performance_multiplier = mult
    
    def get_effective_allocations(self) -> Dict[str, float]:
        """Get current effective allocations after all adjustments."""
        raw = {name: alloc.effective_weight for name, alloc in self.allocations.items()}
        
        # Normalize to sum to 1.0
        total = sum(raw.values())
        if total > 0:
            return {name: w / total for name, w in raw.items()}
        return raw
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        vix_regime: str = None
    ) -> List[EnsembleSignal]:
        """
        Generate aggregated signals from all strategies.
        
        Args:
            data: Market data dict
            vix_regime: Current VIX regime
            
        Returns:
            List of aggregated EnsembleSignals
        """
        if vix_regime:
            self.current_regime = vix_regime
            self._apply_regime_adjustments()
        
        # Collect signals from each strategy
        all_signals: Dict[str, List[Tuple[str, Signal]]] = {}  # symbol -> [(strategy, signal)]
        
        allocations = self.get_effective_allocations()
        
        for name, strategy in self.strategies.items():
            if not self.allocations[name].enabled:
                continue
            if allocations.get(name, 0) < 0.01:
                continue
            
            current_positions = [
                sym for sym, pos in self.positions.items()
                if pos.get('strategy') == name
            ]
            
            try:
                signals = strategy.generate_signals(data, current_positions, vix_regime)
                
                for sig in signals:
                    if sig.symbol not in all_signals:
                        all_signals[sig.symbol] = []
                    all_signals[sig.symbol].append((name, sig))
                    
            except Exception as e:
                logger.error(f"Signal generation failed for {name}: {e}")
        
        # Aggregate signals
        return self._aggregate_signals(all_signals, allocations)
    
    def _aggregate_signals(
        self,
        signals_by_symbol: Dict[str, List[Tuple[str, Signal]]],
        allocations: Dict[str, float]
    ) -> List[EnsembleSignal]:
        """Aggregate and resolve conflicting signals."""
        ensemble_signals = []
        
        for symbol, strategy_signals in signals_by_symbol.items():
            # Group by direction
            buys = [(s, sig) for s, sig in strategy_signals if sig.signal_type == SignalType.BUY]
            sells = [(s, sig) for s, sig in strategy_signals if sig.signal_type in [SignalType.SELL, SignalType.CLOSE]]
            
            # Determine net direction
            buy_weight = sum(allocations.get(s, 0) * sig.strength for s, sig in buys)
            sell_weight = sum(allocations.get(s, 0) * sig.strength for s, sig in sells)
            
            if buy_weight > sell_weight and buy_weight > 0.1:
                direction = 'BUY'
                signals = buys
                strength = buy_weight
            elif sell_weight > buy_weight and sell_weight > 0.1:
                direction = 'SELL'
                signals = sells
                strength = sell_weight
            else:
                continue  # No clear consensus
            
            # Calculate position size (sum of strategy allocations * position sizes)
            position_pct = 0
            for strat_name, sig in signals:
                strat_alloc = allocations.get(strat_name, 0)
                sig_size = sig.position_size_pct or 0.1
                position_pct += strat_alloc * sig_size
            
            # Cap at max position
            position_pct = min(position_pct, 0.10)  # Max 10% per position
            
            # Get stops/targets from strongest signal
            strongest = max(signals, key=lambda x: x[1].strength)
            _, strongest_sig = strongest
            
            ensemble_signals.append(EnsembleSignal(
                symbol=symbol,
                direction=direction,
                combined_strength=strength,
                contributing_strategies=[s for s, _ in signals],
                position_size_pct=position_pct,
                stop_loss=strongest_sig.stop_loss,
                target=strongest_sig.target_price,
                metadata={
                    'buy_weight': buy_weight,
                    'sell_weight': sell_weight,
                    'regime': self.current_regime,
                }
            ))
        
        # Sort by strength
        ensemble_signals.sort(key=lambda x: x.combined_strength, reverse=True)
        
        return ensemble_signals
    
    def get_status(self) -> Dict:
        """Get current ensemble status."""
        allocations = self.get_effective_allocations()
        
        status = {
            'regime': self.current_regime,
            'total_capital': self.total_capital,
            'strategies': {},
            'positions': len(self.positions),
        }
        
        for name, alloc in self.allocations.items():
            status['strategies'][name] = {
                'enabled': alloc.enabled,
                'base_weight': alloc.base_weight,
                'effective_weight': allocations.get(name, 0),
                'regime_mult': alloc.regime_multiplier,
                'perf_mult': alloc.performance_multiplier,
                'capital': self.total_capital * allocations.get(name, 0),
            }
        
        return status
    
    def print_status(self):
        """Print formatted status."""
        status = self.get_status()
        
        print(f"\n{'='*60}")
        print("ENSEMBLE STATUS")
        print(f"{'='*60}")
        print(f"Regime: {status['regime']}")
        print(f"Capital: ${status['total_capital']:,.0f}")
        print(f"Positions: {status['positions']}")
        print(f"\n{'Strategy':<30} {'Weight':>8} {'Capital':>12} {'Regime':>8}")
        print("-" * 60)
        
        for name, info in status['strategies'].items():
            enabled = "✓" if info['enabled'] else "✗"
            print(f"{enabled} {name:<28} {info['effective_weight']:>7.1%} "
                  f"${info['capital']:>10,.0f} {info['regime_mult']:>7.1f}x")


def main():
    """Demo ensemble coordinator."""
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("ENSEMBLE COORDINATOR")
    print("="*60)

    # Uses TOTAL_CAPITAL from config
    ensemble = EnsembleCoordinator()
    
    # Show initial status
    ensemble.print_status()
    
    # Simulate regime changes
    print("\n--- Simulating regime changes ---\n")
    
    for vix in [12, 18, 28, 45]:
        ensemble.update_regime(vix)
        allocs = ensemble.get_effective_allocations()
        print(f"VIX={vix}: {', '.join(f'{k}={v:.0%}' for k, v in allocs.items())}")


if __name__ == "__main__":
    main()
