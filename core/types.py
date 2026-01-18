"""
Core Types for Trading System
=============================
Canonical type definitions for signals, positions, and related data.

This module is the SINGLE SOURCE OF TRUTH for trading data types.
All other modules should import from here.

Strategy ID Convention
----------------------
Two valid formats exist (intentional design):

1. GP-discovered strategies: 8-character hex UUID
   - Source: research/discovery/strategy_genome.py UUID generation
   - Example: "00304020", "a3f7b291"
   - Tracked in: promotion_pipeline.db, research.db

2. Human-coded strategies: Descriptive snake_case name
   - Source: Strategy class __init__ name parameter
   - Example: "mean_reversion", "vol_managed_momentum"
   - NOT in promotion_pipeline.db

Both use strategy_id field - the value format indicates the source.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import uuid
import json


class Side(Enum):
    """
    Trade direction.

    Canonical values - use these instead of:
    - 'buy'/'sell'/'long'/'short' (lowercase strings)
    - 'entry'/'exit' (ambiguous)
    - SignalType enum (deprecated, use Side)
    """
    BUY = "BUY"      # Open long or close short
    SELL = "SELL"    # Open short or close long
    CLOSE = "CLOSE"  # Explicit position close (direction-agnostic)
    HOLD = "HOLD"    # No action (used in strategy output)


class SignalStatus(Enum):
    """Signal lifecycle status."""
    PENDING = "pending"        # Generated, not yet acted on
    SUBMITTED = "submitted"    # Order submitted to broker
    EXECUTED = "executed"      # Order filled
    PARTIAL = "partial"        # Partially filled
    CANCELLED = "cancelled"    # User cancelled
    EXPIRED = "expired"        # Time limit reached
    REJECTED = "rejected"      # Broker rejected


class Signal:
    """
    Canonical trading signal - SINGLE SOURCE OF TRUTH.

    All signal-related classes should be replaced with this one:
    - StoredSignal (signal_tracker.py) - DEPRECATED
    - EnsembleSignal (ensemble_coordinator.py) - use EnsembleResult
    - AggregatedSignal (ensemble.py) - use EnsembleResult
    - CandidateSignal (universe_scanner.py) - use Signal with status
    - SignalRecord (signal_scoring.py) - use Signal + outcome fields

    Field naming conventions (new names preferred, old names still work):
    - signal_time (old: timestamp)
    - strategy_id (old: strategy)
    - side (old: signal_type)
    - strength (old: confidence)
    - target_price (old: take_profit)
    - quantity (old: shares)
    """

    __slots__ = (
        'signal_id', 'symbol', 'strategy_id', 'side', 'strength', 'price',
        'signal_time', 'target_price', 'stop_loss', 'quantity', 'position_size_pct',
        'reason', 'metadata', 'status', 'expires_at', 'executed_at', 'execution_id',
        'outcome', 'pnl_pct', 'hold_days', 'max_drawdown_pct', 'max_profit_pct', 'exit_time'
    )

    def __init__(
        self,
        # New canonical names (preferred)
        symbol: str = None,
        strategy_id: str = None,
        side: Side = None,
        strength: float = None,
        price: float = None,
        signal_time: datetime = None,
        # Old names (deprecated but supported for backward compatibility)
        timestamp: datetime = None,      # -> signal_time
        strategy: str = None,            # -> strategy_id
        signal_type: Side = None,        # -> side
        confidence: float = None,        # -> strength
        take_profit: float = None,       # -> target_price
        shares: float = None,            # -> quantity
        # Optional fields
        signal_id: str = None,
        target_price: float = None,
        stop_loss: float = None,
        quantity: float = None,
        position_size_pct: float = None,
        reason: str = "",
        metadata: Dict[str, Any] = None,
        status: str = None,
        expires_at: datetime = None,
        executed_at: datetime = None,
        execution_id: int = None,
        outcome: str = None,
        pnl_pct: float = None,
        hold_days: int = None,
        max_drawdown_pct: float = None,
        max_profit_pct: float = None,
        exit_time: datetime = None,
    ):
        # Map old names to new names (old names take precedence if both provided)
        self.signal_time = timestamp if timestamp is not None else signal_time
        self.strategy_id = strategy if strategy is not None else strategy_id
        self.side = signal_type if signal_type is not None else side
        self.strength = confidence if confidence is not None else strength
        self.target_price = take_profit if take_profit is not None else target_price
        self.quantity = shares if shares is not None else quantity

        # Required fields
        self.symbol = symbol
        self.price = price

        # Generate signal_id if not provided
        self.signal_id = signal_id if signal_id else str(uuid.uuid4())[:12]

        # Optional fields
        self.stop_loss = stop_loss
        self.position_size_pct = position_size_pct
        self.reason = reason or ""
        self.metadata = metadata if metadata is not None else {}
        self.status = status if status else SignalStatus.PENDING.value
        self.expires_at = expires_at
        self.executed_at = executed_at
        self.execution_id = execution_id
        self.outcome = outcome
        self.pnl_pct = pnl_pct
        self.hold_days = hold_days
        self.max_drawdown_pct = max_drawdown_pct
        self.max_profit_pct = max_profit_pct
        self.exit_time = exit_time

        # Validation
        if self.strength is not None and not 0 <= self.strength <= 1:
            raise ValueError(f"Signal strength must be 0-1, got {self.strength}")
        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")

        # Convert Side enum if passed as string
        if isinstance(self.side, str):
            self.side = Side(self.side.upper())

    # Backward compatibility properties (read-only)
    @property
    def timestamp(self) -> datetime:
        """Deprecated: Use signal_time instead."""
        return self.signal_time

    @property
    def strategy(self) -> str:
        """Deprecated: Use strategy_id instead."""
        return self.strategy_id

    @property
    def signal_type(self) -> Side:
        """Deprecated: Use side instead."""
        return self.side

    @property
    def confidence(self) -> float:
        """Deprecated: Use strength instead."""
        return self.strength

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'strategy_id': self.strategy_id,
            'side': self.side.value,
            'strength': self.strength,
            'price': self.price,
            'signal_time': self.signal_time.isoformat() if self.signal_time else None,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'quantity': self.quantity,
            'position_size_pct': self.position_size_pct,
            'reason': self.reason,
            'metadata': self.metadata,
            'status': self.status,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'outcome': self.outcome,
            'pnl_pct': self.pnl_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal from dictionary."""
        # Handle datetime fields
        signal_time = data.get('signal_time')
        if isinstance(signal_time, str):
            signal_time = datetime.fromisoformat(signal_time)
        elif signal_time is None:
            signal_time = datetime.now()

        expires_at = data.get('expires_at')
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        executed_at = data.get('executed_at')
        if isinstance(executed_at, str):
            executed_at = datetime.fromisoformat(executed_at)

        exit_time = data.get('exit_time')
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)

        # Handle side field
        side = data.get('side', 'BUY')
        if isinstance(side, str):
            side = Side(side.upper())

        return cls(
            signal_id=data.get('signal_id', str(uuid.uuid4())[:12]),
            symbol=data['symbol'],
            strategy_id=data.get('strategy_id', data.get('strategy', '')),
            side=side,
            strength=data.get('strength', data.get('confidence', 0.5)),
            price=data['price'],
            signal_time=signal_time,
            target_price=data.get('target_price', data.get('take_profit')),
            stop_loss=data.get('stop_loss'),
            quantity=data.get('quantity', data.get('shares')),
            position_size_pct=data.get('position_size_pct'),
            reason=data.get('reason', ''),
            metadata=data.get('metadata', {}),
            status=data.get('status', SignalStatus.PENDING.value),
            expires_at=expires_at,
            executed_at=executed_at,
            execution_id=data.get('execution_id'),
            outcome=data.get('outcome'),
            pnl_pct=data.get('pnl_pct', data.get('profit_pct')),
            hold_days=data.get('hold_days'),
            max_drawdown_pct=data.get('max_drawdown_pct'),
            max_profit_pct=data.get('max_profit_pct'),
            exit_time=exit_time,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'Signal':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # Compatibility properties for gradual migration
    @property
    def strategy(self) -> str:
        """Deprecated: Use strategy_id instead."""
        return self.strategy_id

    @property
    def signal_type(self) -> Side:
        """Deprecated: Use side instead."""
        return self.side

    @property
    def timestamp(self) -> datetime:
        """Deprecated: Use signal_time instead."""
        return self.signal_time

    @property
    def confidence(self) -> float:
        """Deprecated: Use strength instead."""
        return self.strength


@dataclass
class SignalScore:
    """
    Scoring result for a signal.

    This is NOT a signal itself - it's the evaluation of a signal's quality
    based on historical performance of similar signals.
    """
    signal_id: str                        # Reference to Signal.signal_id
    conviction: float                     # 0.0-1.0 overall score
    win_probability: float                # Historical win rate
    expected_return: float                # Average return for similar
    expected_risk: float                  # Average max drawdown
    risk_reward_ratio: float              # expected_return / expected_risk

    # Sizing recommendation
    size_multiplier: float                # 0.0-2.0 (NOT suggested_size_multiplier)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 95% CI

    # Contributing factors
    factors: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    similar_signals_count: int = 0

    def __str__(self) -> str:
        return (
            f"SignalScore(conviction={self.conviction:.2f}, "
            f"win_prob={self.win_probability:.1%}, "
            f"exp_return={self.expected_return:.2%}, "
            f"size_mult={self.size_multiplier:.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signal_id': self.signal_id,
            'conviction': self.conviction,
            'win_probability': self.win_probability,
            'expected_return': self.expected_return,
            'expected_risk': self.expected_risk,
            'risk_reward_ratio': self.risk_reward_ratio,
            'size_multiplier': self.size_multiplier,
            'confidence_interval': list(self.confidence_interval),
            'factors': self.factors,
            'sample_size': self.sample_size,
            'similar_signals_count': self.similar_signals_count,
        }


@dataclass
class EnsembleResult:
    """
    Aggregated result from ensemble coordination.

    Replaces:
    - EnsembleSignal (ensemble_coordinator.py)
    - AggregatedSignal (ensemble.py)

    This references Signal objects rather than duplicating their fields.
    """
    symbol: str
    side: Side                            # Aggregated direction
    combined_strength: float              # Weighted average of contributing signals
    contributing_signal_ids: List[str]    # References to Signal.signal_id
    contributing_strategies: List[str]    # Strategy IDs that contributed

    # Sizing
    position_size_pct: float              # Recommended position size
    capital_allocation: float = 0.0       # Dollar amount

    # Risk
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None  # NOT target

    # Conflict info
    conflicts: List[str] = field(default_factory=list)  # Conflicting strategies

    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'combined_strength': self.combined_strength,
            'contributing_signal_ids': self.contributing_signal_ids,
            'contributing_strategies': self.contributing_strategies,
            'position_size_pct': self.position_size_pct,
            'capital_allocation': self.capital_allocation,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'conflicts': self.conflicts,
            'metadata': self.metadata,
        }


# Type aliases for clarity
StrategyId = str  # Either hex UUID ("00304020") or name ("mean_reversion")
SignalId = str    # 12-char UUID prefix
