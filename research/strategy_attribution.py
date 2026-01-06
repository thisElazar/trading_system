"""
Strategy Attribution
=====================
Tracks positions by strategy and provides attribution analysis.

Features:
- Position-to-strategy mapping
- PnL attribution by strategy
- Factor exposure analysis (momentum, value, quality, etc.)
- Regime-based performance attribution
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy classification for factor attribution."""
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    MEAN_REVERSION = "mean_reversion"
    PAIRS = "pairs"
    REGIME = "regime"
    BREAKOUT = "breakout"
    SECTOR = "sector"
    UNKNOWN = "unknown"


# Map strategy names to types
STRATEGY_TYPE_MAP = {
    'vol_managed_momentum': StrategyType.MOMENTUM,
    'vol_managed_momentum_v2': StrategyType.MOMENTUM,
    'factor_momentum': StrategyType.MOMENTUM,
    'quality_small_cap_value': StrategyType.VALUE,
    'mean_reversion': StrategyType.MEAN_REVERSION,
    'pairs_trading': StrategyType.PAIRS,
    'vix_regime_rotation': StrategyType.REGIME,
    'sector_rotation': StrategyType.SECTOR,
    'relative_volume_breakout': StrategyType.BREAKOUT,
    'gap_fill': StrategyType.MEAN_REVERSION,
    'orb': StrategyType.BREAKOUT,
    'vwap_reversion': StrategyType.MEAN_REVERSION,
}


@dataclass
class AttributedPosition:
    """A position with full strategy attribution."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    entry_date: datetime
    strategy_name: str
    strategy_type: StrategyType

    # Optional fields
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    exit_price: float = None
    exit_date: datetime = None
    realized_pnl: float = None
    realized_pnl_pct: float = None
    exit_reason: str = None

    # Factor exposures at entry
    momentum_score: float = 0.0
    value_score: float = 0.0
    quality_score: float = 0.0
    volatility_score: float = 0.0

    # Regime at entry
    vix_regime: str = "normal"  # low, normal, high, extreme

    # Attribution metadata
    position_id: str = None
    created_at: datetime = field(default_factory=datetime.now)

    def update_price(self, current_price: float) -> None:
        """Update unrealized PnL based on current price."""
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

    def close(self, exit_price: float, exit_date: datetime, exit_reason: str = "signal") -> None:
        """Close the position and calculate realized PnL."""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_reason = exit_reason

        if self.side == 'long':
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
            self.realized_pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
            self.realized_pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type.value,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'exit_price': self.exit_price,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'realized_pnl': self.realized_pnl,
            'realized_pnl_pct': self.realized_pnl_pct,
            'exit_reason': self.exit_reason,
            'momentum_score': self.momentum_score,
            'value_score': self.value_score,
            'quality_score': self.quality_score,
            'volatility_score': self.volatility_score,
            'vix_regime': self.vix_regime,
            'position_id': self.position_id,
        }


@dataclass
class StrategyAttribution:
    """Attribution summary for a single strategy."""
    strategy_name: str
    strategy_type: StrategyType

    # Position counts
    open_positions: int = 0
    closed_positions: int = 0
    total_positions: int = 0

    # PnL
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0

    # Performance metrics
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Factor exposures (average across positions)
    avg_momentum_score: float = 0.0
    avg_value_score: float = 0.0
    avg_quality_score: float = 0.0

    # Regime breakdown
    pnl_by_regime: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioAttribution:
    """Full portfolio attribution summary."""
    timestamp: datetime
    by_strategy: Dict[str, StrategyAttribution]
    by_type: Dict[str, StrategyAttribution]
    by_regime: Dict[str, float]

    # Portfolio totals
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_open_positions: int = 0
    total_closed_positions: int = 0

    # Concentration metrics
    largest_position_pct: float = 0.0
    strategy_concentration: float = 0.0  # HHI of strategy allocation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/logging."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'total_pnl': self.total_pnl,
            'total_open_positions': self.total_open_positions,
            'total_closed_positions': self.total_closed_positions,
            'largest_position_pct': self.largest_position_pct,
            'strategy_concentration': self.strategy_concentration,
            'by_strategy': {
                name: {
                    'type': attr.strategy_type.value,
                    'open': attr.open_positions,
                    'closed': attr.closed_positions,
                    'unrealized_pnl': attr.unrealized_pnl,
                    'realized_pnl': attr.realized_pnl,
                    'win_rate': attr.win_rate,
                    'profit_factor': attr.profit_factor,
                }
                for name, attr in self.by_strategy.items()
            },
            'by_type': {
                type_name: attr.total_pnl
                for type_name, attr in self.by_type.items()
            },
            'by_regime': self.by_regime,
        }


class StrategyAttributionTracker:
    """
    Tracks positions and provides strategy attribution.

    Usage:
        tracker = StrategyAttributionTracker()

        # Open a position
        tracker.open_position(
            symbol='AAPL',
            side='long',
            quantity=100,
            entry_price=150.0,
            strategy_name='vol_managed_momentum',
            vix_regime='normal'
        )

        # Update prices
        tracker.update_prices({'AAPL': 155.0})

        # Close a position
        tracker.close_position(
            symbol='AAPL',
            strategy_name='vol_managed_momentum',
            exit_price=155.0,
            exit_reason='target'
        )

        # Get attribution
        attribution = tracker.get_attribution()
    """

    def __init__(self):
        self.open_positions: Dict[str, List[AttributedPosition]] = {}  # symbol -> positions
        self.closed_positions: List[AttributedPosition] = []
        self._position_counter = 0

    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"pos_{self._position_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def open_position(self,
                      symbol: str,
                      side: str,
                      quantity: float,
                      entry_price: float,
                      strategy_name: str,
                      entry_date: datetime = None,
                      vix_regime: str = "normal",
                      momentum_score: float = 0.0,
                      value_score: float = 0.0,
                      quality_score: float = 0.0,
                      volatility_score: float = 0.0) -> AttributedPosition:
        """
        Open a new attributed position.

        Returns:
            The created AttributedPosition object
        """
        strategy_type = STRATEGY_TYPE_MAP.get(strategy_name, StrategyType.UNKNOWN)

        position = AttributedPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_date=entry_date or datetime.now(),
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            current_price=entry_price,
            vix_regime=vix_regime,
            momentum_score=momentum_score,
            value_score=value_score,
            quality_score=quality_score,
            volatility_score=volatility_score,
            position_id=self._generate_position_id()
        )

        if symbol not in self.open_positions:
            self.open_positions[symbol] = []
        self.open_positions[symbol].append(position)

        logger.debug(f"Opened position: {symbol} {side} {quantity} @ {entry_price} ({strategy_name})")
        return position

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        for symbol, price in prices.items():
            if symbol in self.open_positions:
                for position in self.open_positions[symbol]:
                    position.update_price(price)

    def close_position(self,
                       symbol: str,
                       strategy_name: str,
                       exit_price: float,
                       exit_date: datetime = None,
                       exit_reason: str = "signal",
                       quantity: float = None) -> Optional[AttributedPosition]:
        """
        Close a position.

        If quantity is specified, partially close. Otherwise close first matching position.

        Returns:
            The closed position, or None if not found
        """
        if symbol not in self.open_positions:
            logger.warning(f"No open positions for {symbol}")
            return None

        # Find matching position
        matching = [p for p in self.open_positions[symbol] if p.strategy_name == strategy_name]
        if not matching:
            logger.warning(f"No {strategy_name} position for {symbol}")
            return None

        position = matching[0]  # FIFO
        position.close(exit_price, exit_date or datetime.now(), exit_reason)

        # Move to closed
        self.open_positions[symbol].remove(position)
        if not self.open_positions[symbol]:
            del self.open_positions[symbol]
        self.closed_positions.append(position)

        logger.debug(f"Closed position: {symbol} @ {exit_price} ({exit_reason}) -> PnL: {position.realized_pnl:.2f}")
        return position

    def close_all_for_strategy(self,
                               strategy_name: str,
                               prices: Dict[str, float],
                               exit_reason: str = "strategy_exit") -> List[AttributedPosition]:
        """Close all positions for a strategy."""
        closed = []
        symbols = list(self.open_positions.keys())

        for symbol in symbols:
            price = prices.get(symbol, 0.0)
            while any(p.strategy_name == strategy_name for p in self.open_positions.get(symbol, [])):
                position = self.close_position(symbol, strategy_name, price, exit_reason=exit_reason)
                if position:
                    closed.append(position)
                else:
                    break

        return closed

    def get_open_positions_for_strategy(self, strategy_name: str) -> List[AttributedPosition]:
        """Get all open positions for a strategy."""
        positions = []
        for symbol_positions in self.open_positions.values():
            positions.extend([p for p in symbol_positions if p.strategy_name == strategy_name])
        return positions

    def get_attribution(self) -> PortfolioAttribution:
        """
        Calculate full portfolio attribution.

        Returns:
            PortfolioAttribution with breakdown by strategy, type, and regime
        """
        by_strategy: Dict[str, StrategyAttribution] = {}
        by_type: Dict[str, StrategyAttribution] = {}
        by_regime: Dict[str, float] = {'low': 0, 'normal': 0, 'high': 0, 'extreme': 0}

        all_positions = []

        # Collect open positions
        for symbol_positions in self.open_positions.values():
            all_positions.extend(symbol_positions)

        # Add closed positions
        all_positions.extend(self.closed_positions)

        # Aggregate by strategy
        for position in all_positions:
            strategy = position.strategy_name
            if strategy not in by_strategy:
                by_strategy[strategy] = StrategyAttribution(
                    strategy_name=strategy,
                    strategy_type=position.strategy_type
                )

            attr = by_strategy[strategy]

            if position.exit_date is None:
                # Open position
                attr.open_positions += 1
                attr.unrealized_pnl += position.unrealized_pnl
            else:
                # Closed position
                attr.closed_positions += 1
                attr.realized_pnl += position.realized_pnl or 0

                if position.realized_pnl and position.realized_pnl > 0:
                    attr.win_count += 1
                else:
                    attr.loss_count += 1

            attr.total_positions += 1
            attr.total_pnl = attr.unrealized_pnl + attr.realized_pnl

            # Factor scores
            attr.avg_momentum_score += position.momentum_score
            attr.avg_value_score += position.value_score
            attr.avg_quality_score += position.quality_score

            # Regime PnL
            regime = position.vix_regime
            pnl = position.realized_pnl or position.unrealized_pnl
            if regime in attr.pnl_by_regime:
                attr.pnl_by_regime[regime] += pnl
            else:
                attr.pnl_by_regime[regime] = pnl

            by_regime[regime] = by_regime.get(regime, 0) + pnl

        # Calculate averages and derived metrics
        for strategy, attr in by_strategy.items():
            if attr.total_positions > 0:
                attr.avg_momentum_score /= attr.total_positions
                attr.avg_value_score /= attr.total_positions
                attr.avg_quality_score /= attr.total_positions

            if attr.closed_positions > 0:
                attr.win_rate = attr.win_count / attr.closed_positions * 100

                # Calculate avg win/loss from closed positions
                closed_for_strategy = [p for p in self.closed_positions if p.strategy_name == strategy]
                wins = [p.realized_pnl for p in closed_for_strategy if p.realized_pnl and p.realized_pnl > 0]
                losses = [p.realized_pnl for p in closed_for_strategy if p.realized_pnl and p.realized_pnl <= 0]

                attr.avg_win = np.mean(wins) if wins else 0
                attr.avg_loss = np.mean(losses) if losses else 0

                gross_profit = sum(wins) if wins else 0
                gross_loss = abs(sum(losses)) if losses else 0
                attr.profit_factor = min(10.0, gross_profit / gross_loss) if gross_loss > 0 else 10.0

        # Aggregate by type
        for strategy, attr in by_strategy.items():
            type_name = attr.strategy_type.value
            if type_name not in by_type:
                by_type[type_name] = StrategyAttribution(
                    strategy_name=type_name,
                    strategy_type=attr.strategy_type
                )

            type_attr = by_type[type_name]
            type_attr.open_positions += attr.open_positions
            type_attr.closed_positions += attr.closed_positions
            type_attr.total_positions += attr.total_positions
            type_attr.unrealized_pnl += attr.unrealized_pnl
            type_attr.realized_pnl += attr.realized_pnl
            type_attr.total_pnl += attr.total_pnl
            type_attr.win_count += attr.win_count
            type_attr.loss_count += attr.loss_count

        # Calculate type-level metrics
        for type_attr in by_type.values():
            if type_attr.closed_positions > 0:
                type_attr.win_rate = type_attr.win_count / type_attr.closed_positions * 100

        # Portfolio totals
        total_unrealized = sum(a.unrealized_pnl for a in by_strategy.values())
        total_realized = sum(a.realized_pnl for a in by_strategy.values())
        total_open = sum(a.open_positions for a in by_strategy.values())
        total_closed = sum(a.closed_positions for a in by_strategy.values())

        # Concentration metrics
        open_position_values = []
        for symbol_positions in self.open_positions.values():
            for p in symbol_positions:
                open_position_values.append(abs(p.current_price * p.quantity))

        total_value = sum(open_position_values) if open_position_values else 1
        largest_pct = max(open_position_values) / total_value * 100 if open_position_values else 0

        # Strategy HHI (concentration)
        strategy_values = {s: a.unrealized_pnl for s, a in by_strategy.items() if a.open_positions > 0}
        total_strategy = sum(abs(v) for v in strategy_values.values())
        if total_strategy > 0:
            shares = [abs(v) / total_strategy for v in strategy_values.values()]
            hhi = sum(s**2 for s in shares) * 10000  # HHI scale
        else:
            hhi = 0

        return PortfolioAttribution(
            timestamp=datetime.now(),
            by_strategy=by_strategy,
            by_type=by_type,
            by_regime=by_regime,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            total_pnl=total_unrealized + total_realized,
            total_open_positions=total_open,
            total_closed_positions=total_closed,
            largest_position_pct=largest_pct,
            strategy_concentration=hhi,
        )

    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of strategy performance."""
        attribution = self.get_attribution()

        rows = []
        for name, attr in attribution.by_strategy.items():
            rows.append({
                'strategy': name,
                'type': attr.strategy_type.value,
                'open': attr.open_positions,
                'closed': attr.closed_positions,
                'unrealized_pnl': attr.unrealized_pnl,
                'realized_pnl': attr.realized_pnl,
                'total_pnl': attr.total_pnl,
                'win_rate': attr.win_rate,
                'profit_factor': attr.profit_factor,
            })

        return pd.DataFrame(rows).sort_values('total_pnl', ascending=False)


def attribute_backtest_trades(trades: List[Dict], strategy_name: str) -> List[AttributedPosition]:
    """
    Convert backtest trades to attributed positions.

    Args:
        trades: List of trade dicts from BacktestResult.trades
        strategy_name: Name of the strategy that generated these trades

    Returns:
        List of AttributedPosition objects
    """
    positions = []
    strategy_type = STRATEGY_TYPE_MAP.get(strategy_name, StrategyType.UNKNOWN)

    for i, trade in enumerate(trades):
        position = AttributedPosition(
            symbol=trade.get('symbol', 'UNKNOWN'),
            side=trade.get('side', 'long'),
            quantity=trade.get('quantity', 1),
            entry_price=trade.get('entry_price', 0),
            entry_date=pd.to_datetime(trade.get('entry_date')) if trade.get('entry_date') else datetime.now(),
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            exit_price=trade.get('exit_price'),
            exit_date=pd.to_datetime(trade.get('exit_date')) if trade.get('exit_date') else None,
            realized_pnl=trade.get('pnl'),
            realized_pnl_pct=trade.get('pnl_pct'),
            exit_reason=trade.get('exit_reason', 'unknown'),
            position_id=f"bt_{strategy_name}_{i}"
        )
        positions.append(position)

    return positions


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    tracker = StrategyAttributionTracker()

    # Open some positions
    tracker.open_position('AAPL', 'long', 100, 150.0, 'vol_managed_momentum', vix_regime='normal')
    tracker.open_position('MSFT', 'long', 50, 380.0, 'quality_small_cap_value', vix_regime='low')
    tracker.open_position('GOOGL', 'short', 30, 140.0, 'pairs_trading', vix_regime='normal')
    tracker.open_position('NVDA', 'long', 40, 500.0, 'vol_managed_momentum', vix_regime='high')

    # Update prices
    tracker.update_prices({
        'AAPL': 155.0,
        'MSFT': 390.0,
        'GOOGL': 135.0,
        'NVDA': 520.0
    })

    # Close some positions
    tracker.close_position('AAPL', 'vol_managed_momentum', 155.0, exit_reason='target')
    tracker.close_position('GOOGL', 'pairs_trading', 135.0, exit_reason='signal')

    # Get attribution
    attribution = tracker.get_attribution()

    print("\nPortfolio Attribution")
    print("=" * 50)
    print(f"Total Unrealized PnL: ${attribution.total_unrealized_pnl:,.2f}")
    print(f"Total Realized PnL: ${attribution.total_realized_pnl:,.2f}")
    print(f"Total PnL: ${attribution.total_pnl:,.2f}")
    print(f"Open Positions: {attribution.total_open_positions}")
    print(f"Closed Positions: {attribution.total_closed_positions}")

    print("\nBy Strategy:")
    for name, attr in attribution.by_strategy.items():
        print(f"  {name}:")
        print(f"    Open: {attr.open_positions}, Closed: {attr.closed_positions}")
        print(f"    PnL: ${attr.total_pnl:,.2f} (Unrealized: ${attr.unrealized_pnl:,.2f})")
        if attr.closed_positions > 0:
            print(f"    Win Rate: {attr.win_rate:.1f}%")

    print("\nBy Type:")
    for type_name, attr in attribution.by_type.items():
        print(f"  {type_name}: ${attr.total_pnl:,.2f}")

    print("\nBy Regime:")
    for regime, pnl in attribution.by_regime.items():
        if pnl != 0:
            print(f"  {regime}: ${pnl:,.2f}")

    print("\nSummary DataFrame:")
    print(tracker.get_strategy_summary().to_string())
