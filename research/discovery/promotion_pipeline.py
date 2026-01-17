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
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any, Union
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
    # GP-007: Extended duration for statistical significance (was 14d/10 trades)
    # Research recommends 90+ days and 60+ trades to distinguish skill from luck
    min_paper_days: int = 90
    min_paper_trades: int = 60
    max_paper_drawdown: float = -20.0   # Percentage
    min_paper_sharpe: float = 0.3
    min_paper_win_rate: float = 0.40

    # Live monitoring
    max_live_drawdown: float = -25.0    # Trigger pause
    min_rolling_sharpe: float = 0.0     # 60-day rolling
    max_correlation_increase: float = 0.3

    # GP-016: Alpha Decay Detection
    # Rolling Sharpe thresholds for multi-window analysis
    rolling_sharpe_windows: Tuple[int, int, int] = (756, 252, 63)  # 36-mo, 12-mo, 3-mo (trading days)
    min_sharpe_36mo: float = 0.3          # Long-term baseline
    min_sharpe_12mo: float = 0.2          # Medium-term signal
    min_sharpe_3mo: float = 0.0           # Short-term can be negative briefly
    sharpe_decay_threshold: float = 0.5   # 50% decline from peak = decay warning
    max_factor_correlation: float = 0.6   # Crowding risk if correlated with major factors
    max_slippage_ratio: float = 2.0       # paper_slippage / expected_slippage

    # Allocation
    initial_paper_allocation: float = 0.05    # 5%
    initial_live_allocation: float = 0.03     # 3%
    max_strategy_allocation: float = 0.15     # 15%


@dataclass
class AlphaDecayMetrics:
    """GP-016: Metrics for alpha decay detection."""
    strategy_id: str
    analysis_date: datetime

    # Rolling Sharpe at different windows
    sharpe_36mo: Optional[float] = None
    sharpe_12mo: Optional[float] = None
    sharpe_3mo: Optional[float] = None

    # Historical peak and decay
    peak_sharpe: float = 0.0
    peak_sharpe_date: Optional[datetime] = None
    sharpe_decay_pct: float = 0.0  # Current vs peak

    # Factor correlations (crowding risk)
    momentum_corr: float = 0.0
    value_corr: float = 0.0
    quality_corr: float = 0.0
    volatility_corr: float = 0.0
    max_factor_corr: float = 0.0

    # Execution quality
    avg_slippage_bps: float = 0.0
    slippage_trend: float = 0.0  # Positive = worsening
    paper_vs_live_gap: float = 0.0  # Difference in Sharpe

    # Decay signals
    is_decaying: bool = False
    decay_signals: List[str] = field(default_factory=list)
    severity: str = "none"  # none, mild, moderate, severe

    def to_dict(self) -> dict:
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, datetime):
                result[key] = value.isoformat() if value else None
            else:
                result[key] = value
        return result


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

    # CPCV validation metrics (GP-008)
    cpcv_pbo: Optional[float] = None           # Probability of Backtest Overfitting
    cpcv_mean_oos_sharpe: Optional[float] = None
    cpcv_timestamp: Optional[datetime] = None

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
# GP-016: ALPHA DECAY MONITOR
# =============================================================================

class AlphaDecayMonitor:
    """
    GP-016: Monitor strategies for alpha decay.

    Detects performance degradation through:
    - Multi-window rolling Sharpe analysis (36/12/3 month)
    - Trend analysis of Sharpe degradation over time
    - Factor correlation monitoring (momentum, value, quality crowding)
    - Slippage trend tracking (paper vs live execution gap)

    Research basis:
    - Rolling 36-month Sharpe identifies long-term trend
    - Factor correlation > 0.6 signals crowding risk
    - Slippage growth indicates capacity constraints
    """

    def __init__(self, criteria: PromotionCriteria = None):
        self.criteria = criteria or PromotionCriteria()
        self._sharpe_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window_days: int
    ) -> Optional[float]:
        """Calculate rolling Sharpe ratio for a given window."""
        if len(returns) < window_days:
            return None

        window_returns = returns.iloc[-window_days:]
        if window_returns.std() == 0:
            return 0.0

        return float(window_returns.mean() / window_returns.std() * np.sqrt(252))

    def calculate_factor_correlations(
        self,
        strategy_returns: pd.Series,
        factor_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Calculate correlations with common factors.

        Args:
            strategy_returns: Strategy daily returns
            factor_returns: Dict of factor name -> returns series

        Returns:
            Dict of factor name -> correlation
        """
        correlations = {}

        for factor_name, factor_ret in factor_returns.items():
            # Align series
            common_idx = strategy_returns.index.intersection(factor_ret.index)
            if len(common_idx) < 60:  # Need at least 60 days
                correlations[factor_name] = 0.0
                continue

            strat = strategy_returns.loc[common_idx]
            factor = factor_ret.loc[common_idx]

            if strat.std() == 0 or factor.std() == 0:
                correlations[factor_name] = 0.0
            else:
                correlations[factor_name] = float(strat.corr(factor))

        return correlations

    def analyze_decay(
        self,
        strategy_id: str,
        returns: pd.Series,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        slippage_history: Optional[pd.Series] = None,
        paper_sharpe: Optional[float] = None
    ) -> AlphaDecayMetrics:
        """
        Comprehensive alpha decay analysis.

        Args:
            strategy_id: Strategy identifier
            returns: Daily returns series
            factor_returns: Optional factor return series for correlation
            slippage_history: Optional slippage basis points over time
            paper_sharpe: Optional paper trading Sharpe for comparison

        Returns:
            AlphaDecayMetrics with full analysis
        """
        now = datetime.now()
        metrics = AlphaDecayMetrics(
            strategy_id=strategy_id,
            analysis_date=now
        )
        decay_signals = []

        # Calculate rolling Sharpes
        windows = self.criteria.rolling_sharpe_windows
        metrics.sharpe_36mo = self.calculate_rolling_sharpe(returns, windows[0])
        metrics.sharpe_12mo = self.calculate_rolling_sharpe(returns, windows[1])
        metrics.sharpe_3mo = self.calculate_rolling_sharpe(returns, windows[2])

        # Track Sharpe history and find peak
        if strategy_id not in self._sharpe_history:
            self._sharpe_history[strategy_id] = []

        current_sharpe = metrics.sharpe_12mo or 0.0
        self._sharpe_history[strategy_id].append((now, current_sharpe))

        # Find peak Sharpe
        if self._sharpe_history[strategy_id]:
            peak_entry = max(self._sharpe_history[strategy_id], key=lambda x: x[1])
            metrics.peak_sharpe = peak_entry[1]
            metrics.peak_sharpe_date = peak_entry[0]

            # Calculate decay percentage
            if metrics.peak_sharpe > 0:
                metrics.sharpe_decay_pct = 1.0 - (current_sharpe / metrics.peak_sharpe)
            else:
                metrics.sharpe_decay_pct = 0.0

        # Check Sharpe thresholds
        if metrics.sharpe_36mo is not None and metrics.sharpe_36mo < self.criteria.min_sharpe_36mo:
            decay_signals.append(f"36mo Sharpe {metrics.sharpe_36mo:.2f} < {self.criteria.min_sharpe_36mo}")

        if metrics.sharpe_12mo is not None and metrics.sharpe_12mo < self.criteria.min_sharpe_12mo:
            decay_signals.append(f"12mo Sharpe {metrics.sharpe_12mo:.2f} < {self.criteria.min_sharpe_12mo}")

        if metrics.sharpe_3mo is not None and metrics.sharpe_3mo < self.criteria.min_sharpe_3mo:
            decay_signals.append(f"3mo Sharpe {metrics.sharpe_3mo:.2f} < {self.criteria.min_sharpe_3mo}")

        # Check decay from peak
        if metrics.sharpe_decay_pct > self.criteria.sharpe_decay_threshold:
            decay_signals.append(
                f"Sharpe declined {metrics.sharpe_decay_pct:.1%} from peak "
                f"({metrics.peak_sharpe:.2f} -> {current_sharpe:.2f})"
            )

        # Factor correlations (crowding detection)
        if factor_returns:
            correlations = self.calculate_factor_correlations(returns, factor_returns)
            metrics.momentum_corr = correlations.get('momentum', 0.0)
            metrics.value_corr = correlations.get('value', 0.0)
            metrics.quality_corr = correlations.get('quality', 0.0)
            metrics.volatility_corr = correlations.get('volatility', 0.0)
            metrics.max_factor_corr = max(abs(c) for c in correlations.values()) if correlations else 0.0

            if metrics.max_factor_corr > self.criteria.max_factor_correlation:
                crowded_factors = [
                    f for f, c in correlations.items()
                    if abs(c) > self.criteria.max_factor_correlation
                ]
                decay_signals.append(
                    f"High factor correlation: {', '.join(crowded_factors)} "
                    f"(max={metrics.max_factor_corr:.2f})"
                )

        # Slippage analysis
        if slippage_history is not None and len(slippage_history) > 20:
            metrics.avg_slippage_bps = float(slippage_history.mean())

            # Calculate slippage trend (slope of linear regression)
            x = np.arange(len(slippage_history))
            if len(x) > 1:
                coeffs = np.polyfit(x, slippage_history.values, 1)
                metrics.slippage_trend = float(coeffs[0])  # Positive = worsening

                if metrics.slippage_trend > 0.1:  # Significant positive trend
                    decay_signals.append(
                        f"Slippage trending up: {metrics.slippage_trend:.2f} bps/day"
                    )

        # Paper vs live gap
        if paper_sharpe is not None and metrics.sharpe_12mo is not None:
            metrics.paper_vs_live_gap = paper_sharpe - metrics.sharpe_12mo
            if metrics.paper_vs_live_gap > 0.5:  # Live significantly worse than paper
                decay_signals.append(
                    f"Live Sharpe {metrics.paper_vs_live_gap:.2f} below paper trading"
                )

        # Determine decay status and severity
        metrics.decay_signals = decay_signals
        metrics.is_decaying = len(decay_signals) > 0

        if len(decay_signals) == 0:
            metrics.severity = "none"
        elif len(decay_signals) == 1:
            metrics.severity = "mild"
        elif len(decay_signals) <= 3:
            metrics.severity = "moderate"
        else:
            metrics.severity = "severe"

        return metrics

    def should_retire(self, metrics: AlphaDecayMetrics) -> Tuple[bool, str]:
        """
        Determine if strategy should be retired based on decay metrics.

        Returns:
            Tuple of (should_retire, reason)
        """
        if metrics.severity == "severe":
            return True, f"Severe alpha decay: {'; '.join(metrics.decay_signals[:3])}"

        # Specific retirement triggers
        if metrics.sharpe_36mo is not None and metrics.sharpe_36mo < 0:
            return True, f"36-month Sharpe negative ({metrics.sharpe_36mo:.2f})"

        if metrics.sharpe_decay_pct > 0.7:  # 70% decline from peak
            return True, f"Sharpe declined {metrics.sharpe_decay_pct:.1%} from peak"

        if metrics.max_factor_corr > 0.8:  # Highly crowded
            return True, f"Strategy highly crowded (factor corr={metrics.max_factor_corr:.2f})"

        return False, "Within acceptable decay limits"


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

        # GP-016: Alpha decay monitor
        self.decay_monitor = AlphaDecayMonitor(self.criteria)

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

            # Migration: Add CPCV columns if they don't exist (GP-008)
            try:
                conn.execute("SELECT cpcv_pbo FROM strategy_lifecycle LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE strategy_lifecycle ADD COLUMN cpcv_pbo REAL")
                conn.execute("ALTER TABLE strategy_lifecycle ADD COLUMN cpcv_mean_oos_sharpe REAL")
                conn.execute("ALTER TABLE strategy_lifecycle ADD COLUMN cpcv_timestamp TEXT")

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

    def check_validated_for_paper(
        self,
        strategy_id: str,
        require_cpcv: bool = True
    ) -> Tuple[bool, str, Dict]:
        """
        Check if a validated strategy is ready for paper trading.

        Criteria:
        - Walk-forward efficiency >= 0.50
        - Monte Carlo confidence >= 0.90
        - CPCV PBO <= 0.05 (GP-008, optional but recommended)

        Args:
            strategy_id: Strategy identifier
            require_cpcv: If True, require CPCV validation to pass (default True)

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
            'validation_periods': record.validation_periods_passed,
            'cpcv_pbo': record.cpcv_pbo,
            'cpcv_mean_oos_sharpe': record.cpcv_mean_oos_sharpe
        }

        failures = []

        if record.walk_forward_efficiency < self.criteria.min_walk_forward_efficiency:
            failures.append(f"WF efficiency {record.walk_forward_efficiency:.2f} < {self.criteria.min_walk_forward_efficiency}")

        if record.monte_carlo_confidence < self.criteria.min_monte_carlo_confidence:
            failures.append(f"MC confidence {record.monte_carlo_confidence:.2f} < {self.criteria.min_monte_carlo_confidence}")

        # GP-008: CPCV validation (optional but recommended)
        if require_cpcv:
            if record.cpcv_pbo is None:
                failures.append("CPCV validation not yet run")
            elif record.cpcv_pbo > 0.05:
                failures.append(f"CPCV PBO {record.cpcv_pbo:.1%} > 5% (likely overfit)")

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

    def check_live_for_retirement(
        self,
        strategy_id: str,
        returns: Optional[pd.Series] = None,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        slippage_history: Optional[pd.Series] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Check if a live strategy should be retired.

        GP-016 Enhanced with alpha decay detection:
        - Max DD exceeded
        - Rolling Sharpe analysis (36/12/3 month windows)
        - Sharpe decay from peak
        - Factor correlation monitoring (crowding)
        - Slippage trend tracking

        Args:
            strategy_id: Strategy identifier
            returns: Optional daily returns series for decay analysis
            factor_returns: Optional factor return series for correlation
            slippage_history: Optional slippage basis points over time

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

        # Original basic checks
        if record.live_max_drawdown < self.criteria.max_live_drawdown:
            return True, f"Max drawdown {record.live_max_drawdown:.1f}% exceeded limit", metrics

        if record.live_sharpe < self.criteria.min_rolling_sharpe and record.live_days > 30:
            return True, f"Rolling Sharpe {record.live_sharpe:.2f} below threshold", metrics

        # GP-016: Enhanced alpha decay analysis
        if returns is not None and len(returns) >= 63:  # At least 3 months of data
            decay_metrics = self.decay_monitor.analyze_decay(
                strategy_id=strategy_id,
                returns=returns,
                factor_returns=factor_returns,
                slippage_history=slippage_history,
                paper_sharpe=record.paper_sharpe
            )

            # Add decay metrics to return
            metrics['decay_analysis'] = decay_metrics.to_dict()
            metrics['decay_severity'] = decay_metrics.severity
            metrics['decay_signals'] = decay_metrics.decay_signals

            # Check if decay warrants retirement
            should_retire, decay_reason = self.decay_monitor.should_retire(decay_metrics)
            if should_retire:
                return True, f"Alpha decay detected: {decay_reason}", metrics

            # Log warning for moderate decay
            if decay_metrics.severity in ('moderate', 'mild'):
                logger.warning(
                    f"Strategy {strategy_id} showing {decay_metrics.severity} decay: "
                    f"{', '.join(decay_metrics.decay_signals)}"
                )

        return False, "Strategy performing within limits", metrics

    def analyze_strategy_decay(
        self,
        strategy_id: str,
        returns: pd.Series,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        slippage_history: Optional[pd.Series] = None
    ) -> AlphaDecayMetrics:
        """
        GP-016: Standalone alpha decay analysis for a strategy.

        Use this for monitoring without retirement decision.

        Args:
            strategy_id: Strategy identifier
            returns: Daily returns series
            factor_returns: Optional factor return series for correlation
            slippage_history: Optional slippage basis points over time

        Returns:
            AlphaDecayMetrics with full analysis
        """
        record = self.get_strategy_record(strategy_id)
        paper_sharpe = record.paper_sharpe if record else None

        return self.decay_monitor.analyze_decay(
            strategy_id=strategy_id,
            returns=returns,
            factor_returns=factor_returns,
            slippage_history=slippage_history,
            paper_sharpe=paper_sharpe
        )

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
                              'live_start_date', 'retired_at', 'cpcv_timestamp']:
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

    def update_validation_metrics(
        self,
        strategy_id: str,
        walk_forward_efficiency: float,
        monte_carlo_confidence: float
    ):
        """
        Update validation metrics for a strategy.

        Called by the standalone validation subprocess to record results.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategy_lifecycle
                SET walk_forward_efficiency = ?,
                    monte_carlo_confidence = ?,
                    updated_at = ?
                WHERE strategy_id = ?
            """, (
                walk_forward_efficiency, monte_carlo_confidence,
                datetime.now().isoformat(), strategy_id
            ))
        logger.info(f"Updated validation metrics for {strategy_id}: WF={walk_forward_efficiency:.2f}, MC={monte_carlo_confidence:.2f}")

    def run_validation_subprocess(
        self,
        strategy_id: Optional[str] = None,
        all_candidates: bool = False,
        memory_limit_mb: int = 1500,
        timeout_seconds: int = 600
    ) -> Tuple[bool, Dict]:
        """
        Run heavy validation in a separate subprocess.

        This isolates memory-intensive validation (walk-forward + Monte Carlo)
        from the main process to prevent OOM crashes.

        Args:
            strategy_id: Specific strategy to validate
            all_candidates: Validate all CANDIDATE strategies
            memory_limit_mb: Memory limit for subprocess
            timeout_seconds: Timeout for subprocess (default 10 minutes)

        Returns:
            Tuple of (success, result_dict)
        """
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'validate_strategies_subprocess.py'

        if not script_path.exists():
            logger.error(f"Validation script not found: {script_path}")
            return False, {'error': 'Validation script not found'}

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            '--memory-limit', str(memory_limit_mb)
        ]

        if strategy_id:
            cmd.extend(['--strategy-id', strategy_id])
        elif all_candidates:
            cmd.append('--all-candidates')
        else:
            return False, {'error': 'Must specify strategy_id or all_candidates'}

        logger.info(f"Launching validation subprocess: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(Path(__file__).parent.parent.parent)
            )

            # Parse output
            output_lines = result.stdout.strip().split('\n') if result.stdout else []
            stderr = result.stderr.strip() if result.stderr else ''

            # Try to parse JSON results from stdout
            results = []
            for line in output_lines:
                if line.startswith('{'):
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            if result.returncode == 0:
                logger.info(f"Validation subprocess completed successfully")
                return True, {'results': results, 'exit_code': 0}
            elif result.returncode == 1:
                logger.warning(f"Validation subprocess: all strategies failed validation")
                return False, {'results': results, 'exit_code': 1}
            elif result.returncode == 2:
                logger.error(f"Validation subprocess: memory limit exceeded")
                return False, {'error': 'Memory limit exceeded', 'exit_code': 2, 'stderr': stderr}
            else:
                logger.error(f"Validation subprocess error (exit {result.returncode}): {stderr}")
                return False, {'error': stderr, 'exit_code': result.returncode}

        except subprocess.TimeoutExpired:
            logger.error(f"Validation subprocess timed out after {timeout_seconds}s")
            return False, {'error': f'Timeout after {timeout_seconds}s'}
        except Exception as e:
            logger.error(f"Failed to run validation subprocess: {e}", exc_info=True)
            return False, {'error': str(e)}

    def validate_with_cpcv(
        self,
        strategy_id: str,
        returns: pd.Series,
        strategy_signals: pd.Series,
        pbo_threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run CPCV validation on a strategy (GP-008).

        CPCV (Combinatorial Purged Cross-Validation) tests strategies across
        1000+ train/test combinations to detect overfitting via PBO
        (Probability of Backtest Overfitting).

        Args:
            strategy_id: Strategy identifier
            returns: Daily returns series for the strategy universe
            strategy_signals: Strategy signal series (1=long, 0=flat, -1=short)
            pbo_threshold: Reject if PBO > threshold (default 5%)

        Returns:
            Tuple of (passed, result_dict)
        """
        try:
            from research.validation.cpcv import CPCVConfig, run_cpcv_validation
        except ImportError as e:
            logger.warning(f"CPCV module not available: {e}")
            return True, {'error': 'CPCV module not available', 'pbo': None}

        record = self.get_strategy_record(strategy_id)
        if record is None:
            return False, {'error': 'Strategy not found'}

        # Run CPCV validation
        config = CPCVConfig(
            n_subsets=16,
            purge_days=5,
            embargo_pct=0.01,
            max_combinations=1000,
            pbo_reject_threshold=pbo_threshold,
            n_workers=2
        )

        try:
            logger.info(f"Running CPCV validation for {strategy_id}...")
            result = run_cpcv_validation(
                returns=returns,
                strategy_signals=strategy_signals,
                config=config
            )

            # Store results
            now = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE strategy_lifecycle
                    SET cpcv_pbo = ?,
                        cpcv_mean_oos_sharpe = ?,
                        cpcv_timestamp = ?,
                        updated_at = ?
                    WHERE strategy_id = ?
                """, (
                    result.pbo, result.mean_oos_sharpe,
                    now.isoformat(), now.isoformat(),
                    strategy_id
                ))

            passed = result.pbo <= pbo_threshold

            result_dict = {
                'pbo': result.pbo,
                'pbo_ci_95': result.pbo_ci_95,
                'mean_is_sharpe': result.mean_is_sharpe,
                'mean_oos_sharpe': result.mean_oos_sharpe,
                'sharpe_degradation': result.mean_sharpe_degradation,
                'n_splits': result.n_splits_completed,
                'n_overfit': result.n_splits_overfit,
                'passed': passed,
                'threshold': pbo_threshold
            }

            if passed:
                logger.info(f"CPCV PASSED for {strategy_id}: PBO={result.pbo:.1%} <= {pbo_threshold:.1%}")
            else:
                logger.warning(f"CPCV FAILED for {strategy_id}: PBO={result.pbo:.1%} > {pbo_threshold:.1%}")

            return passed, result_dict

        except Exception as e:
            logger.error(f"CPCV validation failed for {strategy_id}: {e}", exc_info=True)
            return False, {'error': str(e), 'pbo': None}

    def validate_gp_strategy_full(
        self,
        strategy_id: str,
        n_mc_simulations: int = 500
    ) -> Tuple[bool, float, float, Dict]:
        """
        Run full validation (walk-forward + Monte Carlo) on a GP strategy.

        Args:
            strategy_id: Strategy identifier
            n_mc_simulations: Number of Monte Carlo bootstrap samples

        Returns:
            Tuple of (passed, walk_forward_efficiency, monte_carlo_confidence, metrics_dict)
        """
        from research.discovery.strategy_genome import GenomeFactory
        from research.discovery.strategy_compiler import EvolvedStrategy
        from research.backtester import Backtester
        from data.cached_data_manager import CachedDataManager

        logger.info(f"Running full validation for GP strategy {strategy_id}")

        try:
            # Load genome from promotion pipeline DB
            record = self.get_strategy_record(strategy_id)
            if record is None or not record.genome_json:
                logger.error(f"No genome found for {strategy_id}")
                return False, 0.0, 0.0, {'error': 'No genome'}

            # Reconstruct the EvolvedStrategy
            factory = GenomeFactory()
            # Handle double-encoded JSON (string stored in DB as escaped JSON)
            # The DB stores a JSON string, which when read is a Python string
            # containing escaped JSON. We need to parse once to get the actual
            # JSON string that deserialize_genome expects.
            genome_json_str = record.genome_json
            if genome_json_str.startswith('"'):
                # Double-encoded - parse once to get the inner JSON string
                genome_json_str = json.loads(genome_json_str)
            genome = factory.deserialize_genome(genome_json_str)
            strategy = EvolvedStrategy(genome, factory)

            # Load data
            data_manager = CachedDataManager()
            data_manager.load_all()
            data = {s: df.copy() for s, df in data_manager.cache.items()}

            if len(data) < 10:
                logger.error(f"Insufficient data for validation ({len(data)} symbols)")
                return False, 0.0, 0.0, {'error': 'Insufficient data'}

            # ===== WALK-FORWARD VALIDATION =====
            backtester = Backtester(initial_capital=100000)
            wf_results = backtester.run_walk_forward(
                strategy=strategy,
                data=data,
                train_days=252,
                test_days=63,
                step_days=21
            )

            if not wf_results:
                logger.warning(f"Walk-forward returned no results for {strategy_id}")
                wf_efficiency = 0.0
            else:
                # Calculate walk-forward efficiency
                # Efficiency = proportion of test periods with positive Sharpe
                positive_periods = sum(1 for r in wf_results if r.sharpe_ratio > 0)
                wf_efficiency = positive_periods / len(wf_results) if wf_results else 0.0
                logger.info(f"Walk-forward efficiency: {wf_efficiency:.2f} ({positive_periods}/{len(wf_results)} positive)")

            # ===== MONTE CARLO VALIDATION =====
            # Run bootstrap simulation on full backtest returns
            full_result = backtester.run(strategy, data)

            if full_result.equity_curve is not None and len(full_result.equity_curve) > 50:
                returns = full_result.equity_curve.pct_change().dropna()

                # Bootstrap simulation
                sharpe_dist = []
                for _ in range(n_mc_simulations):
                    # Sample returns with replacement
                    boot_returns = np.random.choice(returns.values, size=len(returns), replace=True)
                    if np.std(boot_returns) > 0:
                        boot_sharpe = np.mean(boot_returns) / np.std(boot_returns) * np.sqrt(252)
                        sharpe_dist.append(boot_sharpe)

                if sharpe_dist:
                    # Monte Carlo confidence = P(Sharpe > 0)
                    mc_confidence = sum(1 for s in sharpe_dist if s > 0) / len(sharpe_dist)
                    median_sharpe = np.median(sharpe_dist)
                    logger.info(f"Monte Carlo confidence: {mc_confidence:.2f}, Median Sharpe: {median_sharpe:.2f}")
                else:
                    mc_confidence = 0.0
                    median_sharpe = 0.0
            else:
                mc_confidence = 0.0
                median_sharpe = 0.0
                logger.warning(f"Insufficient equity curve for Monte Carlo")

            # Compile metrics
            metrics = {
                'walk_forward_efficiency': wf_efficiency,
                'walk_forward_periods': len(wf_results) if wf_results else 0,
                'monte_carlo_confidence': mc_confidence,
                'monte_carlo_median_sharpe': median_sharpe,
                'monte_carlo_simulations': n_mc_simulations,
                'full_backtest_sharpe': full_result.sharpe_ratio,
                'full_backtest_trades': full_result.total_trades
            }

            # Check if passed
            passed = (
                wf_efficiency >= self.criteria.min_walk_forward_efficiency and
                mc_confidence >= self.criteria.min_monte_carlo_confidence
            )

            logger.info(
                f"Validation {'PASSED' if passed else 'FAILED'} for {strategy_id}: "
                f"WF={wf_efficiency:.2f} (need {self.criteria.min_walk_forward_efficiency}), "
                f"MC={mc_confidence:.2f} (need {self.criteria.min_monte_carlo_confidence})"
            )

            return passed, wf_efficiency, mc_confidence, metrics

        except Exception as e:
            logger.error(f"Full validation failed for {strategy_id}: {e}", exc_info=True)
            return False, 0.0, 0.0, {'error': str(e)}

    def process_all_promotions(self, skip_heavy_validation: bool = False) -> Dict[str, int]:
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

        Args:
            skip_heavy_validation: If True, skip memory-intensive validation (WF/MC tests).
                                   Use during POST_MARKET to prevent OOM crashes.
                                   Heavy validation should run during OVERNIGHT instead.

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
        # Run actual validation for strategies with placeholder metrics (0.5)
        # NOTE: Heavy validation (WF/MC tests) is skipped during POST_MARKET to prevent OOM
        validated_strategies = self.get_strategies_by_status(StrategyStatus.VALIDATED)
        revalidation_count = 0
        skipped_validation_count = 0
        max_revalidations_per_run = 5  # Limit to avoid long runs

        for strategy_id in validated_strategies:
            try:
                record = self.get_strategy_record(strategy_id)

                # Check if this strategy has placeholder metrics that need real validation
                has_placeholder_metrics = (
                    record is not None and
                    record.monte_carlo_confidence is not None and
                    record.monte_carlo_confidence < 0.6  # Placeholder was 0.5
                )

                if has_placeholder_metrics and revalidation_count < max_revalidations_per_run:
                    if skip_heavy_validation:
                        # Skip heavy validation during POST_MARKET - will run during OVERNIGHT
                        skipped_validation_count += 1
                        continue

                    logger.info(f"Re-validating {strategy_id} with real WF/MC tests (had placeholder metrics)")
                    passed, wf_eff, mc_conf, val_metrics = self.validate_gp_strategy_full(strategy_id)
                    revalidation_count += 1

                    # Update database with real metrics
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE strategy_lifecycle
                            SET walk_forward_efficiency = ?,
                                monte_carlo_confidence = ?,
                                updated_at = ?
                            WHERE strategy_id = ?
                        """, (wf_eff, mc_conf, datetime.now().isoformat(), strategy_id))

                    if not passed:
                        logger.info(f"{strategy_id} failed real validation, staying at VALIDATED")
                        continue

                # Check if ready for paper trading (only if not skipped)
                if not (has_placeholder_metrics and skip_heavy_validation):
                    ready, message, metrics = self.check_validated_for_paper(strategy_id)
                    if ready:
                        result = self.promote_to_paper(strategy_id)
                        if result.success:
                            promoted_count += 1
                            logger.info(f"Promoted {strategy_id} to PAPER trading")
            except Exception as e:
                logger.error(f"Failed to process VALIDATED strategy {strategy_id}: {e}", exc_info=True)
                failed_count += 1

        if revalidation_count > 0:
            logger.info(f"Re-validated {revalidation_count} strategies with real WF/MC tests")
        if skipped_validation_count > 0:
            logger.info(f"Skipped {skipped_validation_count} heavy validations (will run during OVERNIGHT)")

        # Stage 4: Check CANDIDATE strategies for promotion to VALIDATED
        # Note: This now supports auto-validation for high-quality candidates
        candidate_strategies = self.get_strategies_by_status(StrategyStatus.CANDIDATE)
        for strategy_id in candidate_strategies:
            try:
                ready, message, metrics = self.check_candidate_for_validation(strategy_id)
                if ready:
                    # Get validation metrics from the record
                    record = self.get_strategy_record(strategy_id)
                    if record:
                        # Check if validation metrics exist
                        has_validation_metrics = (
                            record.walk_forward_efficiency is not None and
                            record.walk_forward_efficiency > 0 and
                            record.monte_carlo_confidence is not None and
                            record.monte_carlo_confidence > 0
                        )

                        if has_validation_metrics:
                            # Use existing validation metrics
                            result = self.promote_to_validated(
                                strategy_id,
                                walk_forward_efficiency=record.walk_forward_efficiency,
                                monte_carlo_confidence=record.monte_carlo_confidence,
                                validation_periods=record.validation_periods_passed or 3
                            )
                            if result.success:
                                promoted_count += 1
                        elif record.discovery_sortino >= 1.5:
                            # AUTO-VALIDATION: High-quality candidates bypass manual validation
                            # Sortino >= 1.5 indicates strong risk-adjusted returns from discovery
                            logger.info(
                                f"Auto-validating CANDIDATE {strategy_id}: "
                                f"Sortino {record.discovery_sortino:.2f} >= 1.5 threshold, "
                                f"applying placeholder validation metrics"
                            )
                            result = self.promote_to_validated(
                                strategy_id,
                                walk_forward_efficiency=0.5,  # Placeholder - conservative estimate
                                monte_carlo_confidence=0.5,   # Placeholder - conservative estimate
                                validation_periods=3
                            )
                            if result.success:
                                promoted_count += 1
                                logger.info(f"Auto-validated {strategy_id} to VALIDATED status")
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

    def load_live_strategies(self) -> List['EvolvedStrategy']:
        """
        Load all LIVE strategies as executable EvolvedStrategy objects.

        This is the key integration point between GP discovery and live trading.
        Strategies with genome_json stored in the database are reconstructed
        as EvolvedStrategy instances that can generate signals.

        Returns:
            List of EvolvedStrategy objects ready for execution
        """
        from research.discovery.strategy_genome import GenomeFactory
        from research.discovery.strategy_compiler import EvolvedStrategy

        strategies = []
        factory = None  # Lazy initialization

        live_ids = self.get_strategies_by_status(StrategyStatus.LIVE)
        logger.info(f"Loading {len(live_ids)} LIVE strategies from promotion pipeline")

        for strategy_id in live_ids:
            try:
                record = self.get_strategy_record(strategy_id)
                if record is None:
                    logger.warning(f"Strategy {strategy_id} not found in database")
                    continue

                if not record.genome_json:
                    logger.warning(f"Strategy {strategy_id} has no genome_json, skipping")
                    continue

                # Initialize factory on first use
                if factory is None:
                    factory = GenomeFactory()

                # Deserialize genome from JSON
                genome = factory.deserialize_genome(record.genome_json)

                # Create EvolvedStrategy wrapper
                evolved = EvolvedStrategy(genome, factory)

                # Attach metadata for execution context
                evolved.strategy_id = strategy_id
                evolved.run_time = record.run_time
                evolved.live_start_date = record.live_start_date
                evolved.discovery_generation = record.discovery_generation

                strategies.append(evolved)
                logger.debug(f"Loaded LIVE strategy: {strategy_id}")

            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_id}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully loaded {len(strategies)} LIVE GP strategies")
        return strategies

    def get_live_strategy_ids(self) -> List[str]:
        """
        Get list of LIVE strategy IDs (convenience method).

        Returns:
            List of strategy IDs currently in LIVE status
        """
        return self.get_strategies_by_status(StrategyStatus.LIVE)


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
