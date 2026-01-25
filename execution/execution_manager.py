"""
ExecutionManager - Central authority for ALL trade execution decisions.

This module provides the ExecutionManager class which coordinates:
- Signal scoring and conviction checks
- Position limit enforcement with smart overrides
- Weighted rebalancing decisions
- Shadow/live routing based on strategy graduation
- Daily trade limit tracking
- Audit trail logging

All signals MUST flow through ExecutionManager before execution.

Thread Safety:
- Uses threading.Lock to prevent TOCTOU race conditions
- Position checks and order submissions are atomic
- Pending symbols tracked to prevent duplicate orders
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import logging
import sqlite3
import json
import os
import threading

from data.storage.db_manager import get_db
from utils.persistent_state import PersistentCounter, PersistentDict, PersistentSet

logger = logging.getLogger(__name__)

# Maximum age for pending approvals before cleanup (1 hour)
MAX_PENDING_AGE_SECONDS = 3600


# =============================================================================
# Enums and Constants
# =============================================================================

class ExecutionRoute(Enum):
    """Execution routing options."""
    LIVE = "live"
    SHADOW = "shadow"
    REJECTED = "rejected"


class RejectionReason(Enum):
    """Standard rejection reasons."""
    CIRCUIT_BREAKER_HALT = "circuit_breaker_halt"
    CIRCUIT_BREAKER_STRATEGY = "circuit_breaker_strategy"
    LOW_CONVICTION = "low_conviction"
    POSITION_LIMIT_GLOBAL = "position_limit_global"
    POSITION_LIMIT_STRATEGY = "position_limit_strategy"
    DAILY_TRADE_LIMIT = "daily_trade_limit"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    INVALID_SIGNAL = "invalid_signal"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExecutionManagerConfig:
    """Configuration for ExecutionManager."""

    # Position limits (increased for paper trading research)
    max_positions: int = 20
    max_positions_per_strategy: int = 8
    max_position_pct: float = 0.05  # 5% per position

    # Smart override (automatic)
    enable_smart_override: bool = True
    override_conviction_threshold: float = 0.85  # 85%+ conviction
    override_win_probability_threshold: float = 0.70  # 70%+ win prob
    max_override_positions: int = 2  # Max extra positions via override

    # Rebalancing (weighted decision)
    enable_rebalancing: bool = True
    min_improvement_for_rebalance: float = 0.15  # 15% better required
    rebalance_cooldown_minutes: int = 30
    rebalance_conviction_weight: float = 0.6  # 60% conviction
    rebalance_pnl_weight: float = 0.4  # 40% P&L

    # Daily trade limit
    max_daily_trades: int = 50

    # Routing
    # Paper trading mode: route to Alpaca paper even before graduation
    # This allows testing order flow while shadow trading builds track record
    require_graduation_for_live: bool = True
    graduation_override_strategies: List[str] = field(default_factory=lambda: [
        "mean_reversion",
        "vol_managed_momentum",
        "relative_volume_breakout",
    ])

    # Conviction
    min_conviction_to_execute: float = 0.40

    # Position Reconciliation
    enable_position_reconciliation: bool = True
    target_position_value: float = 2000.0  # Target $ value per position
    oversized_threshold_pct: float = 0.50  # Trim if 50%+ over target
    min_trim_value: float = 100.0  # Don't bother trimming less than $100

    # Logging
    log_all_decisions: bool = True
    decisions_db_path: str = ""  # Set in __post_init__

    def __post_init__(self):
        if not self.decisions_db_path:
            # Default to trades.db location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.decisions_db_path = os.path.join(base_dir, "db", "trades.db")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RebalanceResult:
    """Result of rebalancing evaluation."""
    should_rebalance: bool = False
    positions_to_close: List[str] = field(default_factory=list)
    conviction_improvement: float = 0.0
    weighted_improvement: float = 0.0
    reason: str = ""


@dataclass
class PositionInfo:
    """Information about an existing position for rebalancing decisions."""
    position_id: int
    symbol: str
    strategy_name: str
    direction: str
    entry_price: float
    current_price: float
    quantity: int
    unrealized_pnl: float
    pnl_pct: float
    opened_at: datetime
    override_entry: bool = False
    entry_conviction: float = 0.0


@dataclass
class ReconciliationResult:
    """Result of a position reconciliation/trim operation."""
    symbol: str
    action: str  # 'trimmed', 'skipped', 'error'
    current_shares: int = 0
    current_value: float = 0.0
    target_value: float = 0.0
    shares_trimmed: int = 0
    trim_value: float = 0.0
    reason: str = ""
    order_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionDecision:
    """Result of the ExecutionManager's decision process."""
    # Input signal info
    strategy_name: str
    symbol: str
    direction: str
    signal_type: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: int = 0
    confidence: float = 0.5

    # Decision
    approved: bool = False
    route: str = ExecutionRoute.REJECTED.value

    # Sizing
    final_shares: int = 0
    final_dollar_value: float = 0.0

    # Scoring
    conviction: float = 0.0
    win_probability: float = 0.0
    risk_reward_ratio: float = 0.0
    suggested_size_multiplier: float = 1.0

    # Override info
    override_applied: bool = False
    override_reason: str = ""

    # Rebalancing
    rebalance_required: bool = False
    positions_to_close: List[str] = field(default_factory=list)

    # Rejection
    rejection_reason: str = ""

    # Checks performed
    checks_performed: Dict[str, bool] = field(default_factory=dict)

    # Audit
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    decision_id: Optional[int] = None


@dataclass
class ExecutionResult:
    """Result of executing a decision."""
    decision: ExecutionDecision
    success: bool = False
    signal_id: Optional[int] = None
    position_id: Optional[int] = None
    fill_price: float = 0.0
    filled_qty: int = 0
    rebalance_closes: List[Dict] = field(default_factory=list)
    message: str = ""
    error: str = ""


# =============================================================================
# ExecutionManager
# =============================================================================

class ExecutionManager:
    """
    Central authority for ALL trade execution decisions.

    Responsibilities:
    1. Unified pre-execution gate checking ALL limits
    2. Signal scoring integration for execution decisions
    3. Smart limit override for high-conviction signals (automatic)
    4. Weighted rebalancing logic (conviction + P&L)
    5. Shadow/Live routing based on strategy graduation
    6. Daily trade limit enforcement
    7. Audit trail logging

    Usage:
        manager = ExecutionManager(config)
        manager.inject_dependencies(...)

        decision = manager.evaluate_signal(signal_data, context)
        if decision.approved:
            result = manager.execute(decision)
    """

    def __init__(self, config: ExecutionManagerConfig = None):
        self.config = config or ExecutionManagerConfig()

        # Injected dependencies (set via inject_dependencies)
        self.signal_scorer = None  # SignalScorer
        self.shadow_trader = None  # ShadowTrader
        self.circuit_breaker = None  # CircuitBreakerManager
        self.adaptive_sizer = None  # AdaptivePositionSizer
        self.execution_tracker = None  # ExecutionTracker
        self.order_executor = None  # OrderExecutor
        self.signal_database = None  # SignalDatabase
        self.promotion_pipeline = None  # PromotionPipeline (graduation authority)
        self.broker = None  # AlpacaConnector (persistent connection)

        # State tracking
        self._strategy_graduation_cache: Dict[str, bool] = {}
        self._daily_trade_count: int = 0
        self._daily_trade_reset_date: str = ""
        self._last_rebalance_times: Dict[str, datetime] = {}

        # Thread safety for position operations (prevents TOCTOU race conditions)
        self._position_lock = threading.Lock()

        # Database manager for persistent state
        self._db = get_db()

        # Persistent symbols with orders in flight (survives restarts)
        self._pending_symbols = PersistentSet(
            self._db, "pending_order_symbols", db_name="trades"
        )

        self._held_symbols_cache: Set[str] = set()  # Cached held symbols (from broker)
        self._held_symbols_cache_time: datetime = datetime.min

        # Persistent approval tracking (survives restarts)
        # These counters track signals that have been APPROVED but not yet persisted
        # to the database. This prevents multiple signals in a batch from all passing
        # the position limit check before any positions are actually created.
        self._pending_approval_count = PersistentCounter(
            self._db, "execution", "pending_approvals", db_name="trades"
        )
        self._pending_approval_by_strategy = PersistentDict[int](
            self._db, "execution", "pending_by_strategy", db_name="trades",
            default_factory=lambda: 0
        )

        # Initialize database
        self._init_decision_logging()

        # Clean up stale persistent state from crashes/restarts
        self._cleanup_stale_state()

        logger.info("ExecutionManager initialized")

    def inject_dependencies(
        self,
        signal_scorer=None,
        shadow_trader=None,
        circuit_breaker=None,
        adaptive_sizer=None,
        execution_tracker=None,
        order_executor=None,
        signal_database=None,
        promotion_pipeline=None,
        broker=None
    ):
        """Inject dependencies after construction."""
        if signal_scorer:
            self.signal_scorer = signal_scorer
        if shadow_trader:
            self.shadow_trader = shadow_trader
        if circuit_breaker:
            self.circuit_breaker = circuit_breaker
        if adaptive_sizer:
            self.adaptive_sizer = adaptive_sizer
        if execution_tracker:
            self.execution_tracker = execution_tracker
        if order_executor:
            self.order_executor = order_executor
        if signal_database:
            self.signal_database = signal_database
        if promotion_pipeline:
            self.promotion_pipeline = promotion_pipeline
        if broker:
            self.broker = broker

        logger.info("ExecutionManager dependencies injected")

        # Sync broker positions after dependencies are injected
        if broker:
            self._sync_broker_positions()

    def _cleanup_stale_state(self):
        """
        Clean up stale persistent state on startup.

        This handles recovery from crashes where pending state was not properly
        cleared. Aggressively clears anything older than 1 hour.
        """
        try:
            # Clear stale pending symbols
            stale_count = self._pending_symbols.clear_stale(MAX_PENDING_AGE_SECONDS)
            if stale_count > 0:
                logger.info(f"Cleaned up {stale_count} stale pending symbols")

            # Reset pending approval counters if they seem stale
            # (we can't know their age, but we reset on startup to be safe)
            current_count = self._pending_approval_count.value
            if current_count > 0:
                logger.warning(
                    f"Found {current_count} pending approvals from previous run, resetting"
                )
                self._pending_approval_count.reset()
                self._pending_approval_by_strategy.clear()

        except Exception as e:
            logger.error(f"Error cleaning up stale state: {e}")

    def _sync_broker_positions(self):
        """
        Sync broker positions to DB on startup.

        Compares broker positions to database and logs any discrepancies.
        This helps detect orphaned positions or sync issues.
        """
        if not self.broker:
            return

        try:
            broker_positions = self.broker.get_positions()
            broker_symbols = {pos.symbol for pos in broker_positions}

            # Update held symbols cache
            self._held_symbols_cache = broker_symbols
            self._held_symbols_cache_time = datetime.now()

            logger.info(f"Synced {len(broker_symbols)} positions from broker")

            # If we have an execution tracker, compare to DB
            if self.execution_tracker:
                try:
                    db_positions = self.execution_tracker.db.get_open_positions()
                    db_symbols = {pos['symbol'] for pos in db_positions}

                    # Find discrepancies
                    broker_only = broker_symbols - db_symbols
                    db_only = db_symbols - broker_symbols

                    if broker_only:
                        logger.warning(f"Positions in broker but not DB: {broker_only}")
                    if db_only:
                        logger.warning(f"Positions in DB but not broker: {db_only}")

                except Exception as e:
                    logger.warning(f"Could not compare DB positions: {e}")

        except Exception as e:
            logger.error(f"Failed to sync broker positions: {e}")

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def evaluate_signal(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,
        signal_type: str = "entry",
        price: float = 0.0,
        stop_loss: float = None,
        take_profit: float = None,
        quantity: int = 0,
        confidence: float = 0.5,
        context: Dict = None
    ) -> ExecutionDecision:
        """
        Evaluate a signal through all pre-execution gates.

        This is THE entry point for all signals before execution.

        Args:
            strategy_name: Name of the strategy generating the signal
            symbol: Trading symbol (e.g., 'AAPL')
            direction: 'long' or 'short'
            signal_type: 'entry' or 'exit'
            price: Signal price
            stop_loss: Stop loss price
            take_profit: Take profit price
            quantity: Requested quantity
            confidence: Signal confidence (0-1)
            context: Additional context (vix_level, current_prices, etc.)

        Returns:
            ExecutionDecision with approval status and execution parameters
        """
        context = context or {}

        # Create decision object
        decision = ExecutionDecision(
            strategy_name=strategy_name,
            symbol=symbol,
            direction=direction,
            signal_type=signal_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            confidence=confidence,
            approved=False,
            route=ExecutionRoute.REJECTED.value
        )

        logger.info(f"Evaluating signal: {strategy_name} {direction} {symbol} @ ${price:.2f}")

        # 1. Circuit breaker check (ultimate authority)
        if not self._check_circuit_breaker(strategy_name, decision):
            self._log_decision(decision)
            return decision

        # 2. Daily trade limit check
        if not self._check_daily_trade_limit(decision):
            self._log_decision(decision)
            return decision

        # 2.5. Duplicate position check (prevent buying more of symbols we already hold)
        if signal_type == "entry" and self._already_holds_position(symbol):
            decision.rejection_reason = f"Already holding position in {symbol}"
            decision.checks_performed['duplicate_position_check'] = False
            logger.info(f"Signal REJECTED: {symbol} - {decision.rejection_reason}")
            self._log_decision(decision)
            return decision
        decision.checks_performed['duplicate_position_check'] = True

        # 3. Score the signal
        self._score_signal(strategy_name, symbol, direction, confidence, context, decision)

        # 4. Conviction gate
        if not self._check_conviction(decision):
            self._log_decision(decision)
            return decision

        # 5. Determine routing (shadow vs live)
        route = self._determine_route(strategy_name)
        decision.route = route

        # 6. Position limit check (with potential override/rebalance)
        if not self._check_position_limits(strategy_name, decision, context):
            self._log_decision(decision)
            return decision

        # 7. Calculate final sizing
        self._calculate_sizing(price, stop_loss, decision, context)

        # 8. Final validation
        if decision.final_shares > 0:
            decision.approved = True
            # Increment pending approval counters to prevent batch race conditions
            self._increment_pending_approval(strategy_name)
            logger.info(f"Signal APPROVED: {symbol} {direction} x{decision.final_shares} via {route}")
        else:
            decision.rejection_reason = "Zero shares after sizing"
            logger.info(f"Signal REJECTED: {symbol} - {decision.rejection_reason}")

        # 9. Log decision
        self._log_decision(decision)

        return decision

    # =========================================================================
    # Check Methods
    # =========================================================================

    def _check_circuit_breaker(self, strategy_name: str, decision: ExecutionDecision) -> bool:
        """Check circuit breaker status."""
        if not self.circuit_breaker:
            decision.checks_performed['circuit_breaker'] = True
            return True

        # Global halt check
        if not self.circuit_breaker.can_trade():
            decision.rejection_reason = RejectionReason.CIRCUIT_BREAKER_HALT.value
            decision.checks_performed['circuit_breaker'] = False
            logger.warning(f"Signal rejected: circuit breaker halt")
            return False

        # Strategy-specific check
        if not self.circuit_breaker.can_run_strategy(strategy_name):
            decision.rejection_reason = RejectionReason.CIRCUIT_BREAKER_STRATEGY.value
            decision.checks_performed['circuit_breaker'] = False
            logger.warning(f"Signal rejected: strategy {strategy_name} paused by circuit breaker")
            return False

        decision.checks_performed['circuit_breaker'] = True
        return True

    def _check_daily_trade_limit(self, decision: ExecutionDecision) -> bool:
        """Check if daily trade limit has been reached."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Reset counter on new day
        if self._daily_trade_reset_date != today:
            self._daily_trade_count = self._get_daily_trade_count_from_db()
            self._daily_trade_reset_date = today

        if self._daily_trade_count >= self.config.max_daily_trades:
            decision.rejection_reason = RejectionReason.DAILY_TRADE_LIMIT.value
            decision.checks_performed['daily_trade_limit'] = False
            logger.warning(f"Signal rejected: daily trade limit ({self.config.max_daily_trades}) reached")
            return False

        decision.checks_performed['daily_trade_limit'] = True
        return True

    def _already_holds_position(self, symbol: str) -> bool:
        """
        Check if we already hold a position in this symbol (thread-safe).

        Uses locking and caching to prevent TOCTOU race conditions where
        multiple signals could pass the check before any order is placed.

        Checks:
        1. Pending symbols (orders in flight)
        2. Cached held symbols (refreshed every 5 seconds)
        3. Internal DB as backup
        """
        with self._position_lock:
            # 1. Check pending symbols first (orders in flight)
            if symbol in self._pending_symbols:
                logger.debug(f"Found {symbol} in pending orders")
                return True

            # 2. Check/refresh held symbols cache (reduced from 5s to 1s for critical paths)
            cache_age = (datetime.now() - self._held_symbols_cache_time).total_seconds()
            if cache_age > 1.0:  # Refresh cache every 1 second
                self._refresh_held_symbols_cache()

            if symbol in self._held_symbols_cache:
                logger.debug(f"Found existing position in {symbol}")
                return True

            # 3. Check internal DB as backup
            internal_positions = self._get_current_positions()
            internal_symbols = {pos.symbol for pos in internal_positions}
            if symbol in internal_symbols:
                logger.debug(f"Found existing DB position in {symbol}")
                return True

            return False

    def _refresh_held_symbols_cache(self) -> None:
        """Refresh the cache of held symbols from broker (called within lock)."""
        try:
            broker = self._get_broker()
            if broker:
                positions = broker.get_positions()
                self._held_symbols_cache = {pos.symbol for pos in positions}
                self._held_symbols_cache_time = datetime.now()
                logger.debug(f"Refreshed held symbols cache: {len(self._held_symbols_cache)} positions")
            else:
                logger.warning("No broker available for position check")
        except Exception as e:
            logger.warning(f"Failed to refresh held symbols cache: {e}")

    def _get_broker(self):
        """Get the broker connection, creating one if needed."""
        if self.broker is None:
            try:
                from execution.alpaca_connector import AlpacaConnector
                self.broker = AlpacaConnector(paper=True)
                logger.info("Created persistent AlpacaConnector for ExecutionManager")
            except Exception as e:
                logger.error(f"Failed to create broker connection: {e}")
                return None
        return self.broker

    def mark_symbol_pending(self, symbol: str) -> bool:
        """
        Mark a symbol as pending (order about to be placed).

        Returns False if symbol is already held or pending.
        This should be called right before placing an order.
        """
        with self._position_lock:
            # Double-check not already held
            if symbol in self._pending_symbols:
                logger.warning(f"Cannot mark {symbol} pending - already pending")
                return False

            if symbol in self._held_symbols_cache:
                logger.warning(f"Cannot mark {symbol} pending - already held")
                return False

            self._pending_symbols.add(symbol)
            logger.debug(f"Marked {symbol} as pending")
            return True

    def unmark_symbol_pending(self, symbol: str, was_filled: bool = False) -> None:
        """
        Remove a symbol from pending status after order completes.

        Args:
            symbol: The symbol to unmark
            was_filled: If True, add to held cache; if False, just remove from pending
        """
        with self._position_lock:
            self._pending_symbols.discard(symbol)
            if was_filled:
                self._held_symbols_cache.add(symbol)
                logger.debug(f"Order filled: {symbol} moved from pending to held")
            else:
                logger.debug(f"Order not filled: {symbol} removed from pending")

    def _increment_pending_approval(self, strategy_name: str) -> None:
        """
        Increment pending approval counters after a signal is approved.

        This prevents batch race conditions where multiple signals all pass
        the position limit check before any positions are actually created.
        Must be decremented when the position is persisted or signal fails.
        """
        with self._position_lock:
            new_count = self._pending_approval_count.increment()
            current = self._pending_approval_by_strategy.get(strategy_name, 0)
            self._pending_approval_by_strategy[strategy_name] = current + 1
            logger.debug(
                f"Pending approval incremented: global={new_count}, "
                f"{strategy_name}={self._pending_approval_by_strategy[strategy_name]}"
            )

    def decrement_pending_approval(self, strategy_name: str) -> None:
        """
        Decrement pending approval counters after position is persisted or signal fails.

        Call this after:
        1. Position is successfully created in the database
        2. Order fails to execute
        3. Signal is abandoned for any reason
        """
        with self._position_lock:
            new_count = self._pending_approval_count.decrement()
            current = self._pending_approval_by_strategy.get(strategy_name, 0)
            if current > 0:
                self._pending_approval_by_strategy[strategy_name] = current - 1
            logger.debug(
                f"Pending approval decremented: global={new_count}, "
                f"{strategy_name}={self._pending_approval_by_strategy.get(strategy_name, 0)}"
            )

    def reset_pending_approvals(self) -> None:
        """
        Reset all pending approval counters.

        Call this at the start of each trading day or after position sync.
        """
        with self._position_lock:
            old_count = self._pending_approval_count.value
            self._pending_approval_count.reset()
            self._pending_approval_by_strategy.clear()
            if old_count > 0:
                logger.info(f"Reset pending approvals: cleared {old_count} pending")

    def _score_signal(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,
        confidence: float,
        context: Dict,
        decision: ExecutionDecision
    ):
        """Score the signal using SignalScorer."""
        if not self.signal_scorer:
            # Use confidence as conviction if no scorer
            decision.conviction = confidence
            decision.win_probability = 0.5
            decision.risk_reward_ratio = 1.0
            decision.suggested_size_multiplier = 1.0
            return

        try:
            signal_type = 'buy' if direction == 'long' else 'sell'
            score = self.signal_scorer.score_signal(
                strategy=strategy_name,
                signal_type=signal_type,
                signal_strength=confidence,
                vix_level=context.get('vix_level', 0),
                trend_alignment=context.get('trend_alignment', 'neutral'),
                volume_confirmed=context.get('volume_confirmed', False)
            )

            decision.conviction = score.conviction
            decision.win_probability = score.win_probability
            decision.risk_reward_ratio = score.risk_reward_ratio
            decision.suggested_size_multiplier = score.suggested_size_multiplier

        except Exception as e:
            logger.warning(f"Signal scoring failed: {e}, using defaults")
            decision.conviction = confidence
            decision.win_probability = 0.5
            decision.risk_reward_ratio = 1.0
            decision.suggested_size_multiplier = 1.0

    def _check_conviction(self, decision: ExecutionDecision) -> bool:
        """Check if signal meets minimum conviction threshold."""
        if decision.conviction < self.config.min_conviction_to_execute:
            decision.rejection_reason = f"{RejectionReason.LOW_CONVICTION.value}: {decision.conviction:.2f} < {self.config.min_conviction_to_execute}"
            decision.checks_performed['conviction'] = False
            logger.info(f"Signal rejected: low conviction ({decision.conviction:.2f})")
            return False

        decision.checks_performed['conviction'] = True
        return True

    def _check_position_limits(
        self,
        strategy_name: str,
        decision: ExecutionDecision,
        context: Dict
    ) -> bool:
        """
        Check position limits with intelligent override and rebalancing.

        Flow:
        1. Check if under limits -> approve
        2. Check if high-conviction override applies -> approve with override
        3. Check if rebalancing makes sense -> approve with rebalance
        4. Otherwise -> reject
        """
        current_positions = self._get_current_positions()
        strategy_positions = [p for p in current_positions if p.strategy_name == strategy_name]

        # Get limits
        global_limit = self.config.max_positions
        strategy_limit = self.config.max_positions_per_strategy

        # Include pending approvals in the count (fixes batch processing race condition)
        # These are signals that have been approved but not yet persisted to the database
        with self._position_lock:
            pending_global = self._pending_approval_count.value
            pending_strategy = self._pending_approval_by_strategy.get(strategy_name, 0)

        effective_global_count = len(current_positions) + pending_global
        effective_strategy_count = len(strategy_positions) + pending_strategy

        # Check if at limits (including pending)
        at_global_limit = effective_global_count >= global_limit
        at_strategy_limit = effective_strategy_count >= strategy_limit

        decision.checks_performed['global_position_limit'] = not at_global_limit
        decision.checks_performed['strategy_position_limit'] = not at_strategy_limit

        # Log effective counts for debugging
        if at_global_limit or at_strategy_limit:
            logger.info(
                f"Position limit check: global={effective_global_count}/{global_limit} "
                f"(+{pending_global} pending), strategy={effective_strategy_count}/{strategy_limit} "
                f"(+{pending_strategy} pending)"
            )

        # Case 1: Under limits - approve
        if not at_global_limit and not at_strategy_limit:
            return True

        # Case 2: Check for smart override (high-conviction exception)
        if self.config.enable_smart_override:
            if self._qualifies_for_override(decision, current_positions):
                decision.override_applied = True
                decision.override_reason = (
                    f"High conviction override: conviction={decision.conviction:.2f}, "
                    f"win_prob={decision.win_probability:.2f}"
                )
                decision.checks_performed['override_applied'] = True
                logger.info(f"Override applied for {decision.symbol}: {decision.override_reason}")
                return True

        # Case 3: Check for rebalancing opportunity
        if self.config.enable_rebalancing:
            # Determine which positions to consider (strategy or global)
            positions_for_rebalance = strategy_positions if at_strategy_limit else current_positions

            rebalance_result = self._evaluate_rebalancing(
                decision, positions_for_rebalance, context
            )
            if rebalance_result.should_rebalance:
                decision.rebalance_required = True
                decision.positions_to_close = rebalance_result.positions_to_close
                decision.checks_performed['rebalance_approved'] = True
                logger.info(f"Rebalance approved: {rebalance_result.reason}")
                return True

        # Case 4: Reject due to limits
        if at_global_limit:
            decision.rejection_reason = f"{RejectionReason.POSITION_LIMIT_GLOBAL.value}: {global_limit} positions"
        else:
            decision.rejection_reason = f"{RejectionReason.POSITION_LIMIT_STRATEGY.value}: {strategy_limit} for {strategy_name}"

        logger.info(f"Signal rejected: {decision.rejection_reason}")
        return False

    def _qualifies_for_override(
        self,
        decision: ExecutionDecision,
        current_positions: List[PositionInfo]
    ) -> bool:
        """
        Check if signal qualifies for automatic limit override.

        Requirements:
        - Conviction >= override_conviction_threshold (85%)
        - Win probability >= override_win_probability_threshold (70%)
        - Risk/reward ratio > 1.0
        - Not already at max override positions
        """
        # Check thresholds
        if decision.conviction < self.config.override_conviction_threshold:
            return False
        if decision.win_probability < self.config.override_win_probability_threshold:
            return False

        # Check override count
        override_count = sum(1 for p in current_positions if p.override_entry)
        if override_count >= self.config.max_override_positions:
            return False

        # Check positive edge
        if decision.risk_reward_ratio < 1.0:
            return False

        return True

    def _evaluate_rebalancing(
        self,
        decision: ExecutionDecision,
        existing_positions: List[PositionInfo],
        context: Dict
    ) -> RebalanceResult:
        """
        Evaluate whether to close an existing position to make room for new signal.

        Uses WEIGHTED scoring: conviction * 0.6 + pnl_factor * 0.4
        """
        result = RebalanceResult()

        if not existing_positions:
            return result

        # Check cooldown
        for pos in existing_positions:
            if pos.symbol in self._last_rebalance_times:
                last_rebalance = self._last_rebalance_times[pos.symbol]
                if datetime.now() - last_rebalance < timedelta(minutes=self.config.rebalance_cooldown_minutes):
                    continue  # Skip this position due to cooldown

        # Score existing positions with weighted formula
        position_scores = []
        for pos in existing_positions:
            weighted_score = self._calculate_weighted_position_score(pos, context)
            position_scores.append({
                'position': pos,
                'weighted_score': weighted_score,
                'conviction': pos.entry_conviction or 0.5,
                'pnl_pct': pos.pnl_pct
            })

        if not position_scores:
            return result

        # Sort by weighted score (lowest first)
        position_scores.sort(key=lambda x: x['weighted_score'])

        # Calculate new signal's weighted score
        # For new signal, pnl_factor starts at 0 (neutral)
        new_signal_pnl_factor = 0.5  # Neutral starting point
        new_signal_weighted = (
            decision.conviction * self.config.rebalance_conviction_weight +
            new_signal_pnl_factor * self.config.rebalance_pnl_weight
        )

        # Check if new signal is significantly better than weakest position
        weakest = position_scores[0]
        weighted_improvement = new_signal_weighted - weakest['weighted_score']

        if weighted_improvement >= self.config.min_improvement_for_rebalance:
            result.should_rebalance = True
            result.positions_to_close = [weakest['position'].symbol]
            result.weighted_improvement = weighted_improvement
            result.conviction_improvement = decision.conviction - weakest['conviction']
            result.reason = (
                f"New signal (weighted={new_signal_weighted:.2f}) is {weighted_improvement:.0%} "
                f"better than {weakest['position'].symbol} (weighted={weakest['weighted_score']:.2f}, "
                f"P&L={weakest['pnl_pct']:.1%})"
            )

        return result

    def _calculate_weighted_position_score(
        self,
        position: PositionInfo,
        context: Dict
    ) -> float:
        """
        Calculate weighted score for existing position.

        Score = conviction * 0.6 + pnl_factor * 0.4

        pnl_factor:
        - Highly profitable (>5%): 0.8-1.0
        - Profitable (0-5%): 0.5-0.8
        - Near breakeven (-1% to 0%): 0.4-0.5
        - Losing (-5% to -1%): 0.2-0.4
        - Deep loss (<-5%): 0.0-0.2
        """
        conviction = position.entry_conviction or 0.5
        pnl_pct = position.pnl_pct

        # Convert P&L to factor (0-1 scale)
        if pnl_pct > 0.05:  # >5% profit
            pnl_factor = min(1.0, 0.8 + (pnl_pct - 0.05) * 2)
        elif pnl_pct > 0:  # 0-5% profit
            pnl_factor = 0.5 + pnl_pct * 6  # 0.5 to 0.8
        elif pnl_pct > -0.01:  # -1% to 0%
            pnl_factor = 0.4 + (pnl_pct + 0.01) * 10  # 0.4 to 0.5
        elif pnl_pct > -0.05:  # -5% to -1%
            pnl_factor = 0.2 + (pnl_pct + 0.05) * 5  # 0.2 to 0.4
        else:  # <-5% loss
            pnl_factor = max(0.0, 0.2 + (pnl_pct + 0.05) * 2)

        weighted_score = (
            conviction * self.config.rebalance_conviction_weight +
            pnl_factor * self.config.rebalance_pnl_weight
        )

        return weighted_score

    # =========================================================================
    # Routing
    # =========================================================================

    def _determine_route(self, strategy_name: str) -> str:
        """
        Determine whether signal should route to shadow or live execution.
        """
        # Check override list
        if strategy_name in self.config.graduation_override_strategies:
            return ExecutionRoute.LIVE.value

        # Check graduation status
        if self.config.require_graduation_for_live:
            if self._is_strategy_graduated(strategy_name):
                return ExecutionRoute.LIVE.value
            else:
                return ExecutionRoute.SHADOW.value

        return ExecutionRoute.LIVE.value

    def _is_strategy_graduated(self, strategy_name: str) -> bool:
        """Check if strategy has graduated to live trading.

        Uses PromotionPipeline as the single source of truth for strategy lifecycle.
        A strategy is considered 'graduated' if it has LIVE status in the pipeline.
        """
        # Check cache first
        if strategy_name in self._strategy_graduation_cache:
            return self._strategy_graduation_cache[strategy_name]

        # Query promotion pipeline (authoritative source)
        if self.promotion_pipeline:
            try:
                record = self.promotion_pipeline.get_strategy_record(strategy_name)
                if record:
                    # Import here to avoid circular imports
                    from research.discovery.promotion_pipeline import StrategyStatus
                    is_graduated = record.status == StrategyStatus.LIVE
                    self._strategy_graduation_cache[strategy_name] = is_graduated
                    return is_graduated
            except Exception as e:
                logger.warning(f"Failed to check promotion status for {strategy_name}: {e}")

        # If no promotion pipeline record, check if it's a hardcoded strategy
        # Hardcoded strategies (not from GP discovery) default to live
        return True

    def invalidate_graduation_cache(self, strategy_name: str = None):
        """Invalidate graduation cache for a strategy or all strategies."""
        if strategy_name:
            self._strategy_graduation_cache.pop(strategy_name, None)
        else:
            self._strategy_graduation_cache.clear()

    # =========================================================================
    # Sizing
    # =========================================================================

    def _calculate_sizing(
        self,
        price: float,
        stop_loss: float,
        decision: ExecutionDecision,
        context: Dict
    ):
        """Calculate final position size."""
        base_shares = 0

        if self.adaptive_sizer:
            try:
                result = self.adaptive_sizer.calculate_position_size(
                    strategy=decision.strategy_name,
                    symbol=decision.symbol,
                    price=price,
                    stop_loss=stop_loss or price * 0.95,  # Default 5% stop
                    context=context
                )
                base_shares = result.shares if hasattr(result, 'shares') else decision.quantity
            except Exception as e:
                logger.warning(f"Adaptive sizing failed: {e}")
                base_shares = decision.quantity

        # Fallback: if no adaptive sizer or quantity is 0, use simple fixed-dollar sizing
        if base_shares <= 0:
            # Default to $2,000 per position for paper trading research
            default_position_value = 2000
            if price > 0:
                base_shares = max(1, int(default_position_value / price))
                logger.debug(f"Using default sizing: ${default_position_value} / ${price:.2f} = {base_shares} shares")

        # Apply signal score multiplier (default to 1.0 if not set)
        multiplier = decision.suggested_size_multiplier if decision.suggested_size_multiplier > 0 else 1.0
        adjusted_shares = int(base_shares * multiplier)

        # Apply circuit breaker multiplier if applicable
        if self.circuit_breaker:
            cb_multiplier = self.circuit_breaker.get_position_multiplier()
            adjusted_shares = int(adjusted_shares * cb_multiplier)

        # Ensure at least 1 share if we have positive base_shares (got past all validations)
        # This prevents rounding to 0 from small multipliers
        if base_shares > 0 and adjusted_shares <= 0:
            adjusted_shares = 1
            logger.debug(f"Adjusted shares from 0 to 1 (multiplier rounding protection)")

        decision.final_shares = max(1, adjusted_shares) if adjusted_shares > 0 else 0
        decision.final_dollar_value = decision.final_shares * price

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(self, decision: ExecutionDecision) -> ExecutionResult:
        """
        Execute an approved decision.

        Handles:
        1. Rebalancing (close positions first)
        2. Shadow vs live routing
        3. Order submission
        4. Position tracking
        """
        if not decision.approved:
            return ExecutionResult(
                decision=decision,
                success=False,
                error="Cannot execute unapproved decision"
            )

        result = ExecutionResult(decision=decision)

        try:
            # Step 1: Handle rebalancing
            if decision.rebalance_required:
                for symbol in decision.positions_to_close:
                    close_result = self._close_position_for_rebalance(symbol)
                    result.rebalance_closes.append(close_result)
                    # Update rebalance cooldown
                    self._last_rebalance_times[symbol] = datetime.now()

            # Step 2: Route to appropriate executor
            if decision.route == ExecutionRoute.SHADOW.value:
                exec_result = self._execute_shadow(decision)
            elif decision.route == ExecutionRoute.LIVE.value:
                exec_result = self._execute_live(decision)
            else:
                raise ValueError(f"Unknown route: {decision.route}")

            result.signal_id = exec_result.get('signal_id')
            result.position_id = exec_result.get('position_id')
            result.fill_price = exec_result.get('fill_price', decision.price)
            result.filled_qty = exec_result.get('filled_qty', decision.final_shares)
            result.success = exec_result.get('success', False)
            result.message = exec_result.get('message', '')

            # Update daily trade count
            if result.success:
                self._daily_trade_count += 1

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            result.success = False
            result.error = str(e)

        finally:
            # Always decrement pending approval counter after execution attempt
            # This must happen regardless of success/failure to prevent counter drift
            self.decrement_pending_approval(decision.strategy_name)

        return result

    def _execute_shadow(self, decision: ExecutionDecision) -> Dict:
        """Execute signal in shadow (paper) mode."""
        if not self.shadow_trader:
            logger.warning("ShadowTrader not configured, skipping shadow execution")
            return {'success': False, 'message': 'ShadowTrader not configured'}

        try:
            # Auto-register strategy if not in shadow trading
            active_strategies = self.shadow_trader.get_active_strategies()
            # get_active_strategies returns list of strategy names (strings) or objects
            strategy_names = [s.name if hasattr(s, 'name') else s for s in active_strategies]
            if decision.strategy_name not in strategy_names:
                logger.info(f"Auto-registering strategy {decision.strategy_name} for shadow trading")
                self.shadow_trader.add_strategy(
                    name=decision.strategy_name,
                    initial_capital=10000.0,
                    min_trades=30,
                    min_win_rate=0.55,
                    min_profit_factor=1.5,
                    min_days=14
                )

            signal_type = 'buy' if decision.direction == 'long' else 'sell'
            trade_id = self.shadow_trader.process_signal(
                strategy=decision.strategy_name,
                symbol=decision.symbol,
                signal_type=signal_type,
                price=decision.price,
                shares=decision.final_shares
            )

            return {
                'success': trade_id is not None,
                'signal_id': trade_id,
                'fill_price': decision.price,
                'filled_qty': decision.final_shares,
                'message': f"Shadow executed: {trade_id}"
            }
        except Exception as e:
            logger.error(f"Shadow execution failed: {e}")
            return {'success': False, 'message': str(e)}

    def _execute_live(self, decision: ExecutionDecision) -> Dict:
        """Execute signal in live mode."""
        try:
            # Record signal and open position
            if self.execution_tracker:
                signal_id, position_id = self.execution_tracker.record_signal_and_execute(
                    strategy_name=decision.strategy_name,
                    symbol=decision.symbol,
                    direction=decision.direction,
                    entry_price=decision.price,
                    stop_loss=decision.stop_loss or 0,
                    take_profit=decision.take_profit or 0,
                    quantity=decision.final_shares,
                    confidence=decision.confidence,
                    metadata={
                        'conviction': decision.conviction,
                        'win_probability': decision.win_probability,
                        'override_applied': decision.override_applied,
                        'rebalance_required': decision.rebalance_required
                    }
                )

                return {
                    'success': True,
                    'signal_id': signal_id,
                    'position_id': position_id,
                    'fill_price': decision.price,
                    'filled_qty': decision.final_shares,
                    'message': f"Live executed: signal={signal_id}, position={position_id}"
                }
            else:
                logger.warning("ExecutionTracker not configured")
                return {'success': False, 'message': 'ExecutionTracker not configured'}

        except Exception as e:
            logger.error(f"Live execution failed: {e}")
            return {'success': False, 'message': str(e)}

    def _close_position_for_rebalance(self, symbol: str) -> Dict:
        """Close a position as part of rebalancing."""
        try:
            if self.signal_database:
                # Find the open position
                positions = self.signal_database.get_open_positions()
                for pos in positions:
                    if pos.symbol == symbol:
                        self.signal_database.close_position(
                            position_id=pos.id,
                            exit_price=pos.current_price,
                            exit_reason='rebalance'
                        )
                        logger.info(f"Closed position {symbol} for rebalancing")
                        return {'success': True, 'symbol': symbol, 'reason': 'rebalance'}

            return {'success': False, 'symbol': symbol, 'reason': 'position not found'}

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return {'success': False, 'symbol': symbol, 'error': str(e)}

    # =========================================================================
    # Position Helpers
    # =========================================================================

    def _get_current_positions(self) -> List[PositionInfo]:
        """Get current open positions."""
        positions = []

        if self.signal_database:
            try:
                db_positions = self.signal_database.get_open_positions()
                for pos in db_positions:
                    pnl_pct = 0.0
                    if pos.entry_price and pos.entry_price > 0:
                        current = pos.current_price or pos.entry_price
                        pnl_pct = (current - pos.entry_price) / pos.entry_price

                    positions.append(PositionInfo(
                        position_id=pos.id,
                        symbol=pos.symbol,
                        strategy_name=pos.strategy_name,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price or pos.entry_price,
                        quantity=pos.quantity,
                        unrealized_pnl=pos.unrealized_pnl or 0,
                        pnl_pct=pnl_pct,
                        opened_at=datetime.fromisoformat(pos.opened_at) if pos.opened_at else datetime.now(),
                        override_entry=getattr(pos, 'override_entry', False),
                        entry_conviction=getattr(pos, 'entry_conviction', 0.5)
                    ))
            except Exception as e:
                logger.warning(f"Failed to get positions: {e}")

        return positions

    # =========================================================================
    # Position Reconciliation
    # =========================================================================

    def reconcile_positions(self, dry_run: bool = False) -> List[ReconciliationResult]:
        """
        Reconcile actual Alpaca positions against target sizing.

        Identifies oversized positions and trims them to target value.
        This handles cases like duplicate orders or positions that have grown
        beyond intended allocation.

        Args:
            dry_run: If True, report what would be done without executing

        Returns:
            List of ReconciliationResult for each position evaluated
        """
        if not self.config.enable_position_reconciliation:
            logger.debug("Position reconciliation disabled")
            return []

        results = []
        target_value = self.config.target_position_value
        threshold_pct = self.config.oversized_threshold_pct
        min_trim = self.config.min_trim_value

        # Get actual positions from Alpaca (source of truth)
        try:
            broker = self._get_broker()
            if not broker:
                raise RuntimeError("Broker not available")
            positions = broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions for reconciliation: {e}")
            return [ReconciliationResult(
                symbol="N/A",
                action="error",
                error=f"Failed to get positions: {e}"
            )]

        if not positions:
            logger.debug("No positions to reconcile")
            return []

        logger.info(f"Reconciling {len(positions)} positions (target=${target_value:.0f}, threshold={threshold_pct:.0%})")

        for pos in positions:
            result = self._reconcile_single_position(
                broker=broker,
                position=pos,
                target_value=target_value,
                threshold_pct=threshold_pct,
                min_trim=min_trim,
                dry_run=dry_run
            )
            results.append(result)

        # Log summary
        trimmed = [r for r in results if r.action == 'trimmed']
        if trimmed:
            total_trim_value = sum(r.trim_value for r in trimmed)
            logger.info(f"Reconciliation complete: trimmed {len(trimmed)} positions, total ${total_trim_value:.2f}")
        else:
            logger.info("Reconciliation complete: no positions needed trimming")

        return results

    def _reconcile_single_position(
        self,
        broker,
        position,
        target_value: float,
        threshold_pct: float,
        min_trim: float,
        dry_run: bool
    ) -> ReconciliationResult:
        """Evaluate and potentially trim a single position."""
        symbol = position.symbol
        qty = int(float(position.qty))
        price = float(position.current_price)
        current_value = qty * price

        # Calculate how much we're over target
        over_target = current_value - target_value
        over_target_pct = over_target / target_value if target_value > 0 else 0

        # Check if position exceeds threshold
        if over_target_pct < threshold_pct:
            return ReconciliationResult(
                symbol=symbol,
                action='skipped',
                current_shares=qty,
                current_value=current_value,
                target_value=target_value,
                reason=f"Within threshold ({over_target_pct:.1%} over, need {threshold_pct:.0%}+)"
            )

        # Calculate shares to trim
        shares_to_trim = int(over_target / price) if price > 0 else 0
        trim_value = shares_to_trim * price

        # Check minimum trim value
        if trim_value < min_trim:
            return ReconciliationResult(
                symbol=symbol,
                action='skipped',
                current_shares=qty,
                current_value=current_value,
                target_value=target_value,
                reason=f"Trim value ${trim_value:.2f} below minimum ${min_trim:.2f}"
            )

        # Ensure we don't trim everything
        if shares_to_trim >= qty:
            shares_to_trim = qty - 1  # Keep at least 1 share
            trim_value = shares_to_trim * price

        if shares_to_trim <= 0:
            return ReconciliationResult(
                symbol=symbol,
                action='skipped',
                current_shares=qty,
                current_value=current_value,
                target_value=target_value,
                reason="No shares to trim after constraints"
            )

        # Log what we're about to do
        logger.info(
            f"RECONCILE {symbol}: {qty} shares (${current_value:.2f}) -> "
            f"trim {shares_to_trim} shares (${trim_value:.2f}) to reach ~${target_value:.0f}"
        )

        if dry_run:
            return ReconciliationResult(
                symbol=symbol,
                action='trimmed',
                current_shares=qty,
                current_value=current_value,
                target_value=target_value,
                shares_trimmed=shares_to_trim,
                trim_value=trim_value,
                reason=f"DRY RUN: Would trim {shares_to_trim} shares"
            )

        # Execute the trim
        try:
            order = broker.submit_market_order(
                symbol=symbol,
                qty=shares_to_trim,
                side='sell',
                time_in_force='day'
            )

            if order:
                order_id = str(order.id) if hasattr(order, 'id') else None
                logger.info(f"TRIM EXECUTED: {symbol} sold {shares_to_trim} shares (order={order_id})")

                return ReconciliationResult(
                    symbol=symbol,
                    action='trimmed',
                    current_shares=qty,
                    current_value=current_value,
                    target_value=target_value,
                    shares_trimmed=shares_to_trim,
                    trim_value=trim_value,
                    reason=f"Trimmed to target (was {over_target_pct:.0%} over)",
                    order_id=order_id
                )
            else:
                return ReconciliationResult(
                    symbol=symbol,
                    action='error',
                    current_shares=qty,
                    current_value=current_value,
                    target_value=target_value,
                    error="Order rejected or returned None"
                )

        except Exception as e:
            logger.error(f"Failed to trim {symbol}: {e}")
            return ReconciliationResult(
                symbol=symbol,
                action='error',
                current_shares=qty,
                current_value=current_value,
                target_value=target_value,
                error=str(e)
            )

    # =========================================================================
    # Database / Logging
    # =========================================================================

    def _init_decision_logging(self):
        """Initialize decision logging table."""
        try:
            conn = sqlite3.connect(self.config.decisions_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal_type TEXT,
                    price REAL,
                    approved INTEGER NOT NULL,
                    route TEXT,
                    final_shares INTEGER,
                    final_dollar_value REAL,
                    conviction REAL,
                    win_probability REAL,
                    risk_reward_ratio REAL,
                    override_applied INTEGER DEFAULT 0,
                    override_reason TEXT,
                    rebalance_required INTEGER DEFAULT 0,
                    positions_closed TEXT,
                    rejection_reason TEXT,
                    checks_performed TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exec_decisions_strategy
                ON execution_decisions(strategy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exec_decisions_timestamp
                ON execution_decisions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exec_decisions_approved
                ON execution_decisions(approved)
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to init decision logging: {e}")

    def _log_decision(self, decision: ExecutionDecision):
        """Log decision to database."""
        if not self.config.log_all_decisions:
            return

        try:
            conn = sqlite3.connect(self.config.decisions_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO execution_decisions (
                    timestamp, strategy, symbol, direction, signal_type, price,
                    approved, route, final_shares, final_dollar_value,
                    conviction, win_probability, risk_reward_ratio,
                    override_applied, override_reason,
                    rebalance_required, positions_closed,
                    rejection_reason, checks_performed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.timestamp,
                decision.strategy_name,
                decision.symbol,
                decision.direction,
                decision.signal_type,
                decision.price,
                1 if decision.approved else 0,
                decision.route,
                decision.final_shares,
                decision.final_dollar_value,
                decision.conviction,
                decision.win_probability,
                decision.risk_reward_ratio,
                1 if decision.override_applied else 0,
                decision.override_reason,
                1 if decision.rebalance_required else 0,
                json.dumps(decision.positions_to_close),
                decision.rejection_reason,
                json.dumps(decision.checks_performed)
            ))

            decision.decision_id = cursor.lastrowid
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def _get_daily_trade_count_from_db(self) -> int:
        """Get today's trade count from database."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            conn = sqlite3.connect(self.config.decisions_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM execution_decisions
                WHERE approved = 1 AND timestamp LIKE ?
            """, (f"{today}%",))

            count = cursor.fetchone()[0]
            conn.close()
            return count

        except Exception as e:
            logger.warning(f"Failed to get daily trade count: {e}")
            return 0

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_decision_summary(self, days: int = 1) -> Dict:
        """Get summary of recent decisions."""
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conn = sqlite3.connect(self.config.decisions_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(approved) as approved,
                    SUM(override_applied) as overrides,
                    SUM(rebalance_required) as rebalances
                FROM execution_decisions
                WHERE timestamp >= ?
            """, (cutoff,))

            row = cursor.fetchone()
            conn.close()

            return {
                'total_decisions': row[0] or 0,
                'approved': row[1] or 0,
                'rejected': (row[0] or 0) - (row[1] or 0),
                'overrides_used': row[2] or 0,
                'rebalances': row[3] or 0,
                'approval_rate': (row[1] or 0) / (row[0] or 1)
            }

        except Exception as e:
            logger.error(f"Failed to get decision summary: {e}")
            return {}
