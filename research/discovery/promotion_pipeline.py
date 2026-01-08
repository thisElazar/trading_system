"""
Strategy Promotion Pipeline
============================
Manages the lifecycle of strategies from discovery to live trading.

Stages:
1. CANDIDATE: Discovered by GA, meets basic thresholds
2. VALIDATED: Passed walk-forward and Monte Carlo validation
3. PAPER: Paper trading with real market data
4. LIVE: Deployed to live trading with real capital
5. RETIRED: Removed from active trading

Promotion Thresholds:
- Candidate → Validated: OOS Sharpe > 0.5, Sortino > 0.8, DSR > 0.80
- Validated → Paper: Walk-forward efficiency > 0.50, MC confidence > 0.90
- Paper → Live: 14+ days, 10+ trades, Sharpe > 0.3, Max DD < 20%

Usage:
    from research.discovery.promotion_pipeline import PromotionPipeline

    pipeline = PromotionPipeline()

    # Check if strategy is ready for promotion
    if pipeline.check_promotion_ready(strategy_id, StrategyStatus.PAPER):
        pipeline.promote_strategy(strategy_id, StrategyStatus.LIVE)
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.storage.db_manager import get_db

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class StrategyStatus(Enum):
    """Strategy lifecycle stages."""
    CANDIDATE = "candidate"     # Just discovered, meets basic thresholds
    VALIDATED = "validated"     # Passed validation tests
    PAPER = "paper"             # Paper trading
    LIVE = "live"               # Live trading with real capital
    PAUSED = "paused"           # Temporarily paused
    RETIRED = "retired"         # No longer active


class RetirementReason(Enum):
    """Reasons for retiring a strategy."""
    PERFORMANCE_DECAY = "performance_decay"
    MAX_DRAWDOWN = "max_drawdown"
    REGIME_CHANGE = "regime_change"
    CORRELATION_INCREASE = "correlation_increase"
    MANUAL = "manual"
    PAPER_FAILURE = "paper_failure"


@dataclass
class PromotionCriteria:
    """Thresholds for strategy promotion."""

    # Candidate → Validated
    min_oos_sharpe: float = 0.5
    min_oos_sortino: float = 0.8
    max_oos_drawdown: float = -30.0     # Percentage
    min_oos_trades: int = 50
    min_deflated_sharpe: float = 0.80   # DSR threshold

    # Validated → Paper
    min_walk_forward_efficiency: float = 0.50
    min_monte_carlo_confidence: float = 0.90
    min_validation_periods: int = 3

    # Paper → Live
    min_paper_days: int = 14
    min_paper_trades: int = 10
    max_paper_drawdown: float = -20.0   # Percentage
    min_paper_sharpe: float = 0.3
    min_paper_win_rate: float = 0.40

    # Live monitoring
    max_live_drawdown: float = -25.0    # Trigger pause
    min_rolling_sharpe: float = 0.0     # 60-day rolling
    max_correlation_increase: float = 0.3

    # Allocation
    initial_paper_allocation: float = 0.05    # 5%
    initial_live_allocation: float = 0.03     # 3%
    max_strategy_allocation: float = 0.15     # 15%


@dataclass
class StrategyRecord:
    """Complete record of a strategy's lifecycle."""
    strategy_id: str
    status: StrategyStatus
    created_at: datetime
    updated_at: datetime

    # Discovery metrics
    discovery_generation: int = 0
    discovery_sharpe: float = 0.0
    discovery_sortino: float = 0.0
    discovery_max_drawdown: float = 0.0
    discovery_trades: int = 0
    deflated_sharpe: float = 0.0

    # Validation metrics
    walk_forward_efficiency: float = 0.0
    monte_carlo_confidence: float = 0.0
    validation_periods_passed: int = 0

    # Paper trading metrics
    paper_start_date: Optional[datetime] = None
    paper_days: int = 0
    paper_trades: int = 0
    paper_pnl: float = 0.0
    paper_sharpe: float = 0.0
    paper_max_drawdown: float = 0.0
    paper_win_rate: float = 0.0

    # Live trading metrics
    live_start_date: Optional[datetime] = None
    live_days: int = 0
    live_trades: int = 0
    live_pnl: float = 0.0
    live_sharpe: float = 0.0
    live_max_drawdown: float = 0.0
    current_allocation: float = 0.0

    # Retirement info
    retired_at: Optional[datetime] = None
    retirement_reason: Optional[str] = None

    # Execution config
    genome_json: Optional[str] = None
    run_time: str = "10:00"  # Default execution time (ET)

    def to_dict(self) -> dict:
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat() if value else None
            else:
                result[key] = value
        return result


@dataclass
class PromotionResult:
    """Result of a promotion attempt."""
    success: bool
    strategy_id: str
    from_status: StrategyStatus
    to_status: StrategyStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# PROMOTION PIPELINE
# =============================================================================

class PromotionPipeline:
    """
    Manages strategy lifecycle from discovery to live trading.

    Key responsibilities:
    - Track strategy status and metrics
    - Evaluate promotion readiness
    - Execute promotions with proper allocation
    - Monitor live strategies for retirement
    """

    def __init__(
        self,
        criteria: PromotionCriteria = None,
        db_path: Optional[Path] = None,
        on_promotion: Optional[Callable[[str], None]] = None,
        on_retirement: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize promotion pipeline.

        Args:
            criteria: Promotion criteria thresholds
            db_path: Path to database
            on_promotion: Callback when strategy is promoted to LIVE
                          Signature: (strategy_id: str) -> None
            on_retirement: Callback when strategy is retired
                           Signature: (strategy_id: str) -> None
        """
        self.criteria = criteria or PromotionCriteria()
        self._on_promotion_callback = on_promotion
        self._on_retirement_callback = on_retirement
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "promotion_pipeline.db"
        self.db = get_db()

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                    strategy_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    -- Discovery metrics
                    discovery_generation INTEGER,
                    discovery_sharpe REAL,
                    discovery_sortino REAL,
                    discovery_max_drawdown REAL,
                    discovery_trades INTEGER,
                    deflated_sharpe REAL,

                    -- Validation metrics
                    walk_forward_efficiency REAL,
                    monte_carlo_confidence REAL,
                    validation_periods_passed INTEGER,

                    -- Paper trading
                    paper_start_date TEXT,
                    paper_days INTEGER DEFAULT 0,
                    paper_trades INTEGER DEFAULT 0,
                    paper_pnl REAL DEFAULT 0,
                    paper_sharpe REAL DEFAULT 0,
                    paper_max_drawdown REAL DEFAULT 0,
                    paper_win_rate REAL DEFAULT 0,

                    -- Live trading
                    live_start_date TEXT,
                    live_days INTEGER DEFAULT 0,
                    live_trades INTEGER DEFAULT 0,
                    live_pnl REAL DEFAULT 0,
                    live_sharpe REAL DEFAULT 0,
                    live_max_drawdown REAL DEFAULT 0,
                    current_allocation REAL DEFAULT 0,

                    -- Retirement
                    retired_at TEXT,
                    retirement_reason TEXT,

                    -- Genome/Config
                    genome_json TEXT,

                    -- Execution config
                    run_time TEXT DEFAULT '10:00'
                );

                CREATE TABLE IF NOT EXISTS promotion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    from_status TEXT NOT NULL,
                    to_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    message TEXT,
                    metrics_json TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_performance (
                    strategy_id TEXT,
                    date TEXT,
                    status TEXT,
                    pnl REAL,
                    trades INTEGER,
                    win_rate REAL,
                    PRIMARY KEY (strategy_id, date)
                );

                CREATE INDEX IF NOT EXISTS idx_lifecycle_status
                ON strategy_lifecycle(status);

                CREATE INDEX IF NOT EXISTS idx_history_strategy
                ON promotion_history(strategy_id);
            """)

            # Migration: Add run_time column if it doesn't exist (for existing DBs)
            try:
                conn.execute("SELECT run_time FROM strategy_lifecycle LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE strategy_lifecycle ADD COLUMN run_time TEXT DEFAULT '10:00'")

    def set_callbacks(
        self,
        on_promotion: Optional[Callable[[str], None]] = None,
        on_retirement: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Set callbacks for lifecycle events.

        Args:
            on_promotion: Called when strategy is promoted to LIVE
            on_retirement: Called when strategy is retired
        """
        if on_promotion:
            self._on_promotion_callback = on_promotion
        if on_retirement:
            self._on_retirement_callback = on_retirement

    # =========================================================================
    # STRATEGY REGISTRATION
    # =========================================================================

    def register_candidate(
        self,
        strategy_id: str,
        generation: int,
        sharpe: float,
        sortino: float,
        max_drawdown: float,
        trades: int,
        deflated_sharpe: float,
        genome_json: Optional[str] = None,
        run_time: str = "10:00"
    ) -> StrategyRecord:
        """
        Register a newly discovered strategy as candidate.

        Args:
            strategy_id: Unique strategy identifier
            generation: GA generation when discovered
            sharpe: Out-of-sample Sharpe ratio
            sortino: Out-of-sample Sortino ratio
            max_drawdown: Maximum drawdown percentage
            trades: Number of trades in backtest
            deflated_sharpe: Deflated Sharpe Ratio
            genome_json: Serialized genome (optional)
            run_time: Execution time in HH:MM format (ET), default '10:00'

        Returns:
            StrategyRecord for the new candidate
        """
        now = datetime.now()

        record = StrategyRecord(
            strategy_id=strategy_id,
            status=StrategyStatus.CANDIDATE,
            created_at=now,
            updated_at=now,
            discovery_generation=generation,
            discovery_sharpe=sharpe,
            discovery_sortino=sortino,
            discovery_max_drawdown=max_drawdown,
            discovery_trades=trades,
            deflated_sharpe=deflated_sharpe,
            genome_json=genome_json,
            run_time=run_time
        )

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO strategy_lifecycle
                (strategy_id, status, created_at, updated_at,
                 discovery_generation, discovery_sharpe, discovery_sortino,
                 discovery_max_drawdown, discovery_trades, deflated_sharpe,
                 genome_json, run_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, record.status.value,
                now.isoformat(), now.isoformat(),
                generation, sharpe, sortino, max_drawdown, trades,
                deflated_sharpe, genome_json, run_time
            ))

        logger.info(f"Registered candidate strategy: {strategy_id} (run_time={run_time})")
        return record

    # =========================================================================
    # PROMOTION CHECKS
    # =========================================================================

    def check_candidate_for_validation(self, strategy_id: str) -> Tuple[bool, str, Dict]:
        """
        Check if a candidate strategy is ready for validation.

        Criteria:
        - OOS Sharpe >= 0.5
        - OOS Sortino >= 0.8
        - Max Drawdown >= -30%
        - Trades >= 50
        - DSR >= 0.80

        Returns:
            Tuple of (ready, message, metrics)
        """
        record = self.get_strategy_record(strategy_id)
        if record is None:
            return False, "Strategy not found", {}

        if record.status != StrategyStatus.CANDIDATE:
            return False, f"Strategy is {record.status.value}, not candidate", {}

        metrics = {
            'sharpe': record.discovery_sharpe,
            'sortino': record.discovery_sortino,
            'max_drawdown': record.discovery_max_drawdown,
            'trades': record.discovery_trades,
            'deflated_sharpe': record.deflated_sharpe
        }

        # Check each criterion
        failures = []

        if record.discovery_sharpe < self.criteria.min_oos_sharpe:
            failures.append(f"Sharpe {record.discovery_sharpe:.2f} < {self.criteria.min_oos_sharpe}")

        if record.discovery_sortino < self.criteria.min_oos_sortino:
            failures.append(f"Sortino {record.discovery_sortino:.2f} < {self.criteria.min_oos_sortino}")

        if record.discovery_max_drawdown < self.criteria.max_oos_drawdown:
            failures.append(f"Drawdown {record.discovery_max_drawdown:.1f}% < {self.criteria.max_oos_drawdown}%")

        if record.discovery_trades < self.criteria.min_oos_trades:
            failures.append(f"Trades {record.discovery_trades} < {self.criteria.min_oos_trades}")

        if record.deflated_sharpe < self.criteria.min_deflated_sharpe:
            failures.append(f"DSR {record.deflated_sharpe:.2f} < {self.criteria.min_deflated_sharpe}")

        if failures:
            return False, "; ".join(failures), metrics

        return True, "Ready for validation", metrics

    def check_validated_for_paper(self, strategy_id: str) -> Tuple[bool, str, Dict]:
        """
        Check if a validated strategy is ready for paper trading.

        Criteria:
        - Walk-forward efficiency >= 0.50
        - Monte Carlo confidence >= 0.90

        Returns:
            Tuple of (ready, message, metrics)
        """
        record = self.get_strategy_record(strategy_id)
        if record is None:
            return False, "Strategy not found", {}

        if record.status != StrategyStatus.VALIDATED:
            return False, f"Strategy is {record.status.value}, not validated", {}

        metrics = {
            'walk_forward_efficiency': record.walk_forward_efficiency,
            'monte_carlo_confidence': record.monte_carlo_confidence,
            'validation_periods': record.validation_periods_passed
        }

        failures = []

        if record.walk_forward_efficiency < self.criteria.min_walk_forward_efficiency:
            failures.append(f"WF efficiency {record.walk_forward_efficiency:.2f} < {self.criteria.min_walk_forward_efficiency}")

        if record.monte_carlo_confidence < self.criteria.min_monte_carlo_confidence:
            failures.append(f"MC confidence {record.monte_carlo_confidence:.2f} < {self.criteria.min_monte_carlo_confidence}")

        if failures:
            return False, "; ".join(failures), metrics

        return True, "Ready for paper trading", metrics

    def check_paper_for_live(self, strategy_id: str) -> Tuple[bool, str, Dict]:
        """
        Check if a paper trading strategy is ready for live.

        Criteria:
        - Paper days >= 14
        - Paper trades >= 10
        - Paper Sharpe >= 0.3
        - Paper Max DD >= -20%
        - Paper Win Rate >= 40%

        Returns:
            Tuple of (ready, message, metrics)
        """
        record = self.get_strategy_record(strategy_id)
        if record is None:
            return False, "Strategy not found", {}

        if record.status != StrategyStatus.PAPER:
            return False, f"Strategy is {record.status.value}, not paper", {}

        metrics = {
            'paper_days': record.paper_days,
            'paper_trades': record.paper_trades,
            'paper_sharpe': record.paper_sharpe,
            'paper_max_drawdown': record.paper_max_drawdown,
            'paper_win_rate': record.paper_win_rate,
            'paper_pnl': record.paper_pnl
        }

        failures = []

        if record.paper_days < self.criteria.min_paper_days:
            failures.append(f"Days {record.paper_days} < {self.criteria.min_paper_days}")

        if record.paper_trades < self.criteria.min_paper_trades:
            failures.append(f"Trades {record.paper_trades} < {self.criteria.min_paper_trades}")

        if record.paper_sharpe < self.criteria.min_paper_sharpe:
            failures.append(f"Sharpe {record.paper_sharpe:.2f} < {self.criteria.min_paper_sharpe}")

        if record.paper_max_drawdown < self.criteria.max_paper_drawdown:
            failures.append(f"Drawdown {record.paper_max_drawdown:.1f}% < {self.criteria.max_paper_drawdown}%")

        if record.paper_win_rate < self.criteria.min_paper_win_rate:
            failures.append(f"Win rate {record.paper_win_rate:.1%} < {self.criteria.min_paper_win_rate:.1%}")

        if failures:
            return False, "; ".join(failures), metrics

        return True, "Ready for live trading", metrics

    def check_live_for_retirement(self, strategy_id: str) -> Tuple[bool, str, Dict]:
        """
        Check if a live strategy should be retired.

        Criteria:
        - Max DD exceeded
        - Rolling Sharpe negative
        - Correlation increased significantly

        Returns:
            Tuple of (should_retire, reason, metrics)
        """
        record = self.get_strategy_record(strategy_id)
        if record is None:
            return False, "Strategy not found", {}

        if record.status != StrategyStatus.LIVE:
            return False, f"Strategy is {record.status.value}, not live", {}

        metrics = {
            'live_days': record.live_days,
            'live_pnl': record.live_pnl,
            'live_sharpe': record.live_sharpe,
            'live_max_drawdown': record.live_max_drawdown
        }

        # Check retirement criteria
        if record.live_max_drawdown < self.criteria.max_live_drawdown:
            return True, f"Max drawdown {record.live_max_drawdown:.1f}% exceeded limit", metrics

        if record.live_sharpe < self.criteria.min_rolling_sharpe and record.live_days > 30:
            return True, f"Rolling Sharpe {record.live_sharpe:.2f} below threshold", metrics

        return False, "Strategy performing within limits", metrics

    # =========================================================================
    # PROMOTIONS
    # =========================================================================

    def promote_to_validated(
        self,
        strategy_id: str,
        walk_forward_efficiency: float,
        monte_carlo_confidence: float,
        validation_periods: int = 3
    ) -> PromotionResult:
        """
        Promote a candidate to validated status.

        Args:
            strategy_id: Strategy identifier
            walk_forward_efficiency: Walk-forward test efficiency
            monte_carlo_confidence: Monte Carlo confidence level
            validation_periods: Number of validation periods passed
        """
        # Check if ready
        ready, message, metrics = self.check_candidate_for_validation(strategy_id)
        if not ready:
            return PromotionResult(
                success=False,
                strategy_id=strategy_id,
                from_status=StrategyStatus.CANDIDATE,
                to_status=StrategyStatus.VALIDATED,
                message=f"Not ready: {message}",
                metrics=metrics
            )

        # Update record
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET status = ?,
                    updated_at = ?,
                    walk_forward_efficiency = ?,
                    monte_carlo_confidence = ?,
                    validation_periods_passed = ?
                WHERE strategy_id = ?
            """, (
                StrategyStatus.VALIDATED.value, now.isoformat(),
                walk_forward_efficiency, monte_carlo_confidence,
                validation_periods, strategy_id
            ))

            # Log promotion
            conn.execute("""
                INSERT INTO promotion_history
                (strategy_id, from_status, to_status, timestamp, success, message, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, StrategyStatus.CANDIDATE.value,
                StrategyStatus.VALIDATED.value, now.isoformat(),
                1, "Promoted to validated", json.dumps(metrics)
            ))

        logger.info(f"Promoted {strategy_id} to VALIDATED")

        return PromotionResult(
            success=True,
            strategy_id=strategy_id,
            from_status=StrategyStatus.CANDIDATE,
            to_status=StrategyStatus.VALIDATED,
            message="Successfully promoted to validated",
            metrics=metrics
        )

    def promote_to_paper(self, strategy_id: str) -> PromotionResult:
        """
        Promote a validated strategy to paper trading.

        Registers with shadow trader at initial allocation.
        """
        # Check if ready
        ready, message, metrics = self.check_validated_for_paper(strategy_id)
        if not ready:
            return PromotionResult(
                success=False,
                strategy_id=strategy_id,
                from_status=StrategyStatus.VALIDATED,
                to_status=StrategyStatus.PAPER,
                message=f"Not ready: {message}",
                metrics=metrics
            )

        now = datetime.now()

        # Update record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET status = ?,
                    updated_at = ?,
                    paper_start_date = ?,
                    current_allocation = ?
                WHERE strategy_id = ?
            """, (
                StrategyStatus.PAPER.value, now.isoformat(),
                now.isoformat(), self.criteria.initial_paper_allocation,
                strategy_id
            ))

            # Log promotion
            conn.execute("""
                INSERT INTO promotion_history
                (strategy_id, from_status, to_status, timestamp, success, message, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, StrategyStatus.VALIDATED.value,
                StrategyStatus.PAPER.value, now.isoformat(),
                1, "Promoted to paper trading", json.dumps(metrics)
            ))

        # Register with shadow trader
        self._register_with_shadow_trader(strategy_id)

        logger.info(f"Promoted {strategy_id} to PAPER trading")

        return PromotionResult(
            success=True,
            strategy_id=strategy_id,
            from_status=StrategyStatus.VALIDATED,
            to_status=StrategyStatus.PAPER,
            message=f"Started paper trading with {self.criteria.initial_paper_allocation:.1%} allocation",
            metrics=metrics
        )

    def promote_to_live(self, strategy_id: str) -> PromotionResult:
        """
        Promote a paper trading strategy to live.

        Starts with conservative allocation.
        """
        # Check if ready
        ready, message, metrics = self.check_paper_for_live(strategy_id)
        if not ready:
            return PromotionResult(
                success=False,
                strategy_id=strategy_id,
                from_status=StrategyStatus.PAPER,
                to_status=StrategyStatus.LIVE,
                message=f"Not ready: {message}",
                metrics=metrics
            )

        now = datetime.now()

        # Update record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET status = ?,
                    updated_at = ?,
                    live_start_date = ?,
                    current_allocation = ?
                WHERE strategy_id = ?
            """, (
                StrategyStatus.LIVE.value, now.isoformat(),
                now.isoformat(), self.criteria.initial_live_allocation,
                strategy_id
            ))

            # Log promotion
            conn.execute("""
                INSERT INTO promotion_history
                (strategy_id, from_status, to_status, timestamp, success, message, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, StrategyStatus.PAPER.value,
                StrategyStatus.LIVE.value, now.isoformat(),
                1, "Promoted to live trading", json.dumps(metrics)
            ))

        logger.info(f"Promoted {strategy_id} to LIVE trading with {self.criteria.initial_live_allocation:.1%} allocation")

        # Notify callback (e.g., to reload strategy into scheduler)
        if self._on_promotion_callback:
            try:
                self._on_promotion_callback(strategy_id)
            except Exception as e:
                logger.warning(f"Promotion callback failed for {strategy_id}: {e}")

        return PromotionResult(
            success=True,
            strategy_id=strategy_id,
            from_status=StrategyStatus.PAPER,
            to_status=StrategyStatus.LIVE,
            message=f"Started live trading with {self.criteria.initial_live_allocation:.1%} allocation",
            metrics=metrics
        )

    def retire_strategy(
        self,
        strategy_id: str,
        reason: RetirementReason
    ) -> PromotionResult:
        """
        Retire a strategy from active trading.

        Args:
            strategy_id: Strategy identifier
            reason: Retirement reason
        """
        record = self.get_strategy_record(strategy_id)
        if record is None:
            return PromotionResult(
                success=False,
                strategy_id=strategy_id,
                from_status=StrategyStatus.LIVE,
                to_status=StrategyStatus.RETIRED,
                message="Strategy not found"
            )

        now = datetime.now()
        old_status = record.status

        # Update record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET status = ?,
                    updated_at = ?,
                    retired_at = ?,
                    retirement_reason = ?,
                    current_allocation = 0
                WHERE strategy_id = ?
            """, (
                StrategyStatus.RETIRED.value, now.isoformat(),
                now.isoformat(), reason.value, strategy_id
            ))

            # Log retirement
            conn.execute("""
                INSERT INTO promotion_history
                (strategy_id, from_status, to_status, timestamp, success, message, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, old_status.value,
                StrategyStatus.RETIRED.value, now.isoformat(),
                1, f"Retired: {reason.value}", "{}"
            ))

        logger.warning(f"Retired strategy {strategy_id}: {reason.value}")

        # Notify callback (e.g., to unload strategy from scheduler)
        if self._on_retirement_callback:
            try:
                self._on_retirement_callback(strategy_id)
            except Exception as e:
                logger.warning(f"Retirement callback failed for {strategy_id}: {e}")

        return PromotionResult(
            success=True,
            strategy_id=strategy_id,
            from_status=old_status,
            to_status=StrategyStatus.RETIRED,
            message=f"Retired due to {reason.value}"
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _register_with_shadow_trader(self, strategy_id: str):
        """Register strategy with shadow trader for paper trading."""
        try:
            from execution.shadow_trading import ShadowTrader

            shadow_trader = ShadowTrader()
            # Calculate initial capital based on allocation
            # Assuming $100k portfolio, 5% allocation = $5000
            initial_capital = 100000 * self.criteria.initial_paper_allocation
            shadow_trader.add_strategy(
                name=strategy_id,
                initial_capital=initial_capital,
                min_trades=self.criteria.min_paper_trades,
                min_days=self.criteria.min_paper_days
            )
            logger.info(f"Registered {strategy_id} with shadow trader (capital=${initial_capital:,.0f})")
        except Exception as e:
            logger.warning(f"Failed to register with shadow trader: {e}")

    def get_strategy_record(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Get full record for a strategy."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM strategy_lifecycle WHERE strategy_id = ?
                """, (strategy_id,))
                row = cursor.fetchone()

                if row is None:
                    return None

                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))

                # Convert status string to enum
                data['status'] = StrategyStatus(data['status'])

                # Convert datetime strings
                for field in ['created_at', 'updated_at', 'paper_start_date',
                              'live_start_date', 'retired_at']:
                    if data.get(field):
                        data[field] = datetime.fromisoformat(data[field])

                # Include genome_json and run_time (needed for strategy loading)
                # run_time defaults to '10:00' if not present (migration from old DBs)
                if 'run_time' not in data or data['run_time'] is None:
                    data['run_time'] = '10:00'

                return StrategyRecord(**data)

        except Exception as e:
            logger.error(f"Failed to get strategy record: {e}")
            return None

    def get_strategies_by_status(self, status: StrategyStatus) -> List[str]:
        """Get all strategy IDs with a given status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT strategy_id FROM strategy_lifecycle WHERE status = ?
                """, (status.value,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get strategies: {e}")
            return []

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline status."""
        summary = {
            'candidates': 0,
            'validated': 0,
            'paper': 0,
            'live': 0,
            'retired': 0,
            'total_promoted': 0,
            'promotion_success_rate': 0.0
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) FROM strategy_lifecycle GROUP BY status
                """)
                for row in cursor:
                    status, count = row
                    if status in summary:
                        summary[status] = count

                # Promotion stats
                cursor = conn.execute("""
                    SELECT COUNT(*), SUM(success) FROM promotion_history
                """)
                row = cursor.fetchone()
                if row and row[0] > 0:
                    summary['total_promoted'] = row[1] or 0
                    summary['promotion_success_rate'] = (row[1] or 0) / row[0]

        except Exception as e:
            logger.error(f"Failed to get pipeline summary: {e}")

        return summary

    def update_paper_metrics(
        self,
        strategy_id: str,
        trades: int,
        pnl: float,
        sharpe: float,
        max_drawdown: float,
        win_rate: float
    ):
        """Update paper trading metrics for a strategy."""
        record = self.get_strategy_record(strategy_id)
        if record is None or record.status != StrategyStatus.PAPER:
            return

        # Calculate paper days
        paper_days = 0
        if record.paper_start_date:
            paper_days = (datetime.now() - record.paper_start_date).days

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET paper_days = ?,
                    paper_trades = ?,
                    paper_pnl = ?,
                    paper_sharpe = ?,
                    paper_max_drawdown = ?,
                    paper_win_rate = ?,
                    updated_at = ?
                WHERE strategy_id = ?
            """, (
                paper_days, trades, pnl, sharpe, max_drawdown, win_rate,
                datetime.now().isoformat(), strategy_id
            ))

    def update_live_metrics(
        self,
        strategy_id: str,
        trades: int,
        pnl: float,
        sharpe: float,
        max_drawdown: float
    ):
        """Update live trading metrics for a strategy."""
        record = self.get_strategy_record(strategy_id)
        if record is None or record.status != StrategyStatus.LIVE:
            return

        # Calculate live days
        live_days = 0
        if record.live_start_date:
            live_days = (datetime.now() - record.live_start_date).days

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET live_days = ?,
                    live_trades = ?,
                    live_pnl = ?,
                    live_sharpe = ?,
                    live_max_drawdown = ?,
                    updated_at = ?
                WHERE strategy_id = ?
            """, (
                live_days, trades, pnl, sharpe, max_drawdown,
                datetime.now().isoformat(), strategy_id
            ))


    def process_all_promotions(self) -> Dict[str, int]:
        """
        Process all strategies through the promotion pipeline.

        Iterates through strategies at each stage (LIVE, PAPER, VALIDATED, CANDIDATE)
        and checks if they should be promoted or retired. Executes promotions/retirements
        for eligible strategies.

        Processing order (reverse to avoid immediate re-evaluation):
        1. LIVE -> Check for retirement (performance decay, max drawdown)
        2. PAPER -> Check for promotion to LIVE or retirement (paper failure)
        3. VALIDATED -> Check for promotion to PAPER
        4. CANDIDATE -> Check for promotion to VALIDATED (if validation metrics exist)

        Returns:
            Dict with keys:
            - 'promoted': Number of strategies promoted to next stage
            - 'retired': Number of strategies retired
            - 'failed': Number of strategies that failed processing
        """
        promoted_count = 0
        retired_count = 0
        failed_count = 0

        # Stage 1: Check LIVE strategies for retirement
        live_strategies = self.get_strategies_by_status(StrategyStatus.LIVE)
        for strategy_id in live_strategies:
            try:
                should_retire, reason, metrics = self.check_live_for_retirement(strategy_id)
                if should_retire:
                    # Determine retirement reason from message
                    if "drawdown" in reason.lower():
                        retire_reason = RetirementReason.MAX_DRAWDOWN
                    elif "sharpe" in reason.lower() or "performance" in reason.lower():
                        retire_reason = RetirementReason.PERFORMANCE_DECAY
                    else:
                        retire_reason = RetirementReason.PERFORMANCE_DECAY

                    result = self.retire_strategy(strategy_id, retire_reason)
                    if result.success:
                        retired_count += 1
            except Exception as e:
                logger.error(f"Failed to process LIVE strategy {strategy_id}: {e}", exc_info=True)
                failed_count += 1

        # Stage 2: Check PAPER strategies for promotion to LIVE or retirement
        paper_strategies = self.get_strategies_by_status(StrategyStatus.PAPER)
        for strategy_id in paper_strategies:
            try:
                # Check if paper strategy should be retired due to poor performance
                record = self.get_strategy_record(strategy_id)
                if record and record.paper_days >= self.criteria.min_paper_days // 2:
                    # Severe negative Sharpe after observation period
                    if record.paper_sharpe < -0.5:
                        result = self.retire_strategy(strategy_id, RetirementReason.PAPER_FAILURE)
                        if result.success:
                            retired_count += 1
                        continue
                    # Severe drawdown
                    if record.paper_max_drawdown < self.criteria.max_paper_drawdown * 1.25:
                        result = self.retire_strategy(strategy_id, RetirementReason.PAPER_FAILURE)
                        if result.success:
                            retired_count += 1
                        continue

                # Check if ready for promotion to live
                ready, message, metrics = self.check_paper_for_live(strategy_id)
                if ready:
                    result = self.promote_to_live(strategy_id)
                    if result.success:
                        promoted_count += 1
            except Exception as e:
                logger.error(f"Failed to process PAPER strategy {strategy_id}: {e}", exc_info=True)
                failed_count += 1

        # Stage 3: Check VALIDATED strategies for promotion to PAPER
        validated_strategies = self.get_strategies_by_status(StrategyStatus.VALIDATED)
        for strategy_id in validated_strategies:
            try:
                ready, message, metrics = self.check_validated_for_paper(strategy_id)
                if ready:
                    result = self.promote_to_paper(strategy_id)
                    if result.success:
                        promoted_count += 1
            except Exception as e:
                logger.error(f"Failed to process VALIDATED strategy {strategy_id}: {e}", exc_info=True)
                failed_count += 1

        # Stage 4: Check CANDIDATE strategies for promotion to VALIDATED
        # Note: This requires validation metrics to be already populated
        candidate_strategies = self.get_strategies_by_status(StrategyStatus.CANDIDATE)
        for strategy_id in candidate_strategies:
            try:
                ready, message, metrics = self.check_candidate_for_validation(strategy_id)
                if ready:
                    # Get validation metrics from the record
                    record = self.get_strategy_record(strategy_id)
                    if record and record.walk_forward_efficiency > 0 and record.monte_carlo_confidence > 0:
                        result = self.promote_to_validated(
                            strategy_id,
                            walk_forward_efficiency=record.walk_forward_efficiency,
                            monte_carlo_confidence=record.monte_carlo_confidence,
                            validation_periods=record.validation_periods_passed or 3
                        )
                        if result.success:
                            promoted_count += 1
            except Exception as e:
                logger.error(f"Failed to process CANDIDATE strategy {strategy_id}: {e}", exc_info=True)
                failed_count += 1

        if failed_count > 0:
            logger.warning(f"Promotion pipeline completed with {failed_count} failures")

        return {
            'promoted': promoted_count,
            'retired': retired_count,
            'failed': failed_count
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate promotion pipeline."""
    print("=" * 60)
    print("Promotion Pipeline Demo")
    print("=" * 60)

    # Create pipeline
    pipeline = PromotionPipeline()

    # Register a test candidate
    print("\n--- Registering Candidate ---")
    record = pipeline.register_candidate(
        strategy_id="test_strategy_001",
        generation=50,
        sharpe=0.8,
        sortino=1.2,
        max_drawdown=-18.0,
        trades=75,
        deflated_sharpe=0.85
    )
    print(f"Registered: {record.strategy_id} as {record.status.value}")

    # Check for validation
    print("\n--- Checking Validation Readiness ---")
    ready, message, metrics = pipeline.check_candidate_for_validation("test_strategy_001")
    print(f"Ready: {ready}")
    print(f"Message: {message}")
    print(f"Metrics: {metrics}")

    # Promote to validated
    if ready:
        print("\n--- Promoting to Validated ---")
        result = pipeline.promote_to_validated(
            "test_strategy_001",
            walk_forward_efficiency=0.65,
            monte_carlo_confidence=0.92,
            validation_periods=3
        )
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")

    # Check for paper
    print("\n--- Checking Paper Readiness ---")
    ready, message, metrics = pipeline.check_validated_for_paper("test_strategy_001")
    print(f"Ready: {ready}")
    print(f"Message: {message}")

    # Promote to paper
    if ready:
        print("\n--- Promoting to Paper ---")
        result = pipeline.promote_to_paper("test_strategy_001")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")

    # Update paper metrics (simulating 20 days of paper trading)
    print("\n--- Simulating Paper Trading ---")
    pipeline.update_paper_metrics(
        "test_strategy_001",
        trades=15,
        pnl=2500.0,
        sharpe=0.5,
        max_drawdown=-8.0,
        win_rate=0.55
    )

    # Manually update paper_days for demo
    with sqlite3.connect(pipeline.db_path) as conn:
        conn.execute("""
            UPDATE strategy_lifecycle SET paper_days = 20 WHERE strategy_id = ?
        """, ("test_strategy_001",))

    # Check for live
    print("\n--- Checking Live Readiness ---")
    ready, message, metrics = pipeline.check_paper_for_live("test_strategy_001")
    print(f"Ready: {ready}")
    print(f"Message: {message}")
    print(f"Metrics: {metrics}")

    # Pipeline summary
    print("\n--- Pipeline Summary ---")
    summary = pipeline.get_pipeline_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
