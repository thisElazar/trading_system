#!/usr/bin/env python3
"""
Unified Daily Orchestrator
===========================
Master controller that coordinates all daily trading system operations:
- Pre-market: Data refresh, system checks, position review
- Market hours: Strategy scheduler, signal monitoring
- Post-market: Reconciliation, P&L calculation, daily report
- Overnight: Nightly research (parameter optimization, strategy discovery)

Usage:
    python daily_orchestrator.py              # Run continuously
    python daily_orchestrator.py --status     # Show current status
    python daily_orchestrator.py --once       # Run current phase once and exit
    python daily_orchestrator.py --phase pre  # Force run specific phase
"""

import sys
import os
import signal
import logging
import argparse
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import gc
import json
import traceback
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pytz
import pandas as pd

# Centralized timezone handling
from utils.timezone import normalize_dataframe, now_naive, safe_date_filter

# Import system components
from data.cached_data_manager import CachedDataManager
from execution.scheduler import StrategyScheduler, MarketHours, create_default_scheduler
from execution.signal_tracker import ExecutionTracker
from execution.alpaca_connector import AlpacaConnector
from execution.alerts import AlertManager
from execution.circuit_breaker import CircuitBreakerManager, CircuitBreakerConfig
from config import CIRCUIT_BREAKER, WEEKEND_CONFIG, ALPACA_API_KEY, ALPACA_SECRET_KEY, DATABASES, INTRADAY_EXIT_CONFIG

# Intelligence modules
try:
    from execution.signal_scoring import SignalScorer
    HAS_SIGNAL_SCORER = True
except ImportError:
    SignalScorer = None
    HAS_SIGNAL_SCORER = False

try:
    from execution.shadow_trading import ShadowTrader
    HAS_SHADOW_TRADER = True
except ImportError:
    ShadowTrader = None
    HAS_SHADOW_TRADER = False

try:
    from execution.execution_manager import ExecutionManager, ExecutionManagerConfig
    HAS_EXECUTION_MANAGER = True
except ImportError:
    ExecutionManager = None
    ExecutionManagerConfig = None
    HAS_EXECUTION_MANAGER = False

try:
    from research.ensemble_intelligence import StrategyEnsemble
    HAS_ENSEMBLE = True
except ImportError:
    StrategyEnsemble = None
    HAS_ENSEMBLE = False

try:
    from research.ml_regime_detector import MLRegimeDetector
    HAS_ML_REGIME = True
except ImportError:
    MLRegimeDetector = None
    HAS_ML_REGIME = False

try:
    from research.discovery import PromotionPipeline, PromotionCriteria, StrategyStatus
    HAS_PROMOTION_PIPELINE = True
except ImportError:
    PromotionPipeline = None
    PromotionCriteria = None
    StrategyStatus = None
    HAS_PROMOTION_PIPELINE = False

try:
    from execution.volatility_manager import VolatilityManager
    HAS_VOLATILITY_MANAGER = True
except ImportError:
    VolatilityManager = None
    HAS_VOLATILITY_MANAGER = False

# Optional intraday components (may not be installed)
try:
    from data.stream_handler import MarketDataStream
    HAS_STREAM_HANDLER = True
except ImportError:
    MarketDataStream = None
    HAS_STREAM_HANDLER = False

try:
    from strategies.intraday.gap_fill.strategy import GapFillStrategy
    from strategies.intraday.gap_fill.config import GapFillConfig
    HAS_GAP_FILL = True
except ImportError:
    GapFillStrategy = None
    GapFillConfig = None
    HAS_GAP_FILL = False

try:
    from execution.rapid_gain_scaler import RapidGainScaler, RapidGainConfig
    HAS_RAPID_GAIN_SCALER = True
except ImportError:
    RapidGainScaler = None
    RapidGainConfig = None
    HAS_RAPID_GAIN_SCALER = False

# Hardware integration (optional - Pi 5 status LEDs)
try:
    from hardware.integration import get_hardware_status, HardwareStatus
    HAS_HARDWARE = True
except ImportError:
    get_hardware_status = None
    HardwareStatus = None
    HAS_HARDWARE = False

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Import database error handler for dashboard visibility
from observability.logger import DatabaseErrorHandler
from logging.handlers import RotatingFileHandler

# Log rotation: 10MB per file, keep 5 backups
LOG_MAX_BYTES = 10_000_000
LOG_BACKUP_COUNT = 5

# Configure orchestrator logger (not root logger to avoid duplicate output)
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger (avoids duplicate logs)

# Add file handler with rotation
_file_handler = RotatingFileHandler(
    LOG_DIR / "orchestrator.log",
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT
)
_file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
logger.addHandler(_file_handler)

# Add console handler for systemd journal
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
logger.addHandler(_console_handler)

# Add database handler for errors/warnings (enables dashboard display)
db_error_handler = DatabaseErrorHandler(min_level=logging.WARNING)
db_error_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
logger.addHandler(db_error_handler)


class MarketPhase(Enum):
    """Trading day phases."""
    PRE_MARKET = "pre_market"           # 8:00 AM - 9:30 AM ET
    INTRADAY_OPEN = "intraday_open"     # 9:30 AM - 9:35 AM ET - gap detection window
    INTRADAY_ACTIVE = "intraday_active" # 9:35 AM - 11:30 AM ET - position monitoring
    MARKET_OPEN = "market_open"         # 9:30 AM - 4:00 PM ET (fallback when not in intraday)
    POST_MARKET = "post_market"         # 4:00 PM - 5:00 PM ET
    EVENING = "evening"                 # 5:00 PM - 9:30 PM ET
    OVERNIGHT = "overnight"             # 9:30 PM - 8:00 AM ET
    WEEKEND = "weekend"                 # Friday 4 PM - Sunday evening


class WeekendSubPhase(Enum):
    """Sub-phases within the weekend for structured task execution."""
    FRIDAY_CLEANUP = "friday_cleanup"    # Weekly report, backup, DB cleanup
    RESEARCH = "research"                # Extended optimization (Fri eve → Sat)
    DATA_REFRESH = "data_refresh"        # Sunday AM data updates
    PREWEEK_PREP = "preweek_prep"        # Sunday PM validation
    COMPLETE = "complete"                # Ready for Monday


@dataclass
class PhaseConfig:
    """Configuration for each market phase."""
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    tasks: List[str] = field(default_factory=list)
    check_interval_seconds: int = 60


@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""
    current_phase: MarketPhase = MarketPhase.EVENING
    phase_started_at: Optional[datetime] = None
    last_task_run: Dict[str, datetime] = field(default_factory=dict)
    errors_today: List[Dict[str, Any]] = field(default_factory=list)
    tasks_completed_today: List[str] = field(default_factory=list)
    daily_stats: Dict[str, Any] = field(default_factory=dict)
    is_running: bool = False
    scheduler_thread: Optional[threading.Thread] = None

    # Per-phase task completion tracking (prevents redundant execution within a phase)
    phase_tasks_completed: Dict[str, set] = field(default_factory=dict)

    # Weekend phase state
    weekend_sub_phase: Optional[WeekendSubPhase] = None
    weekend_tasks_completed: List[str] = field(default_factory=list)
    weekend_started_at: Optional[datetime] = None
    weekend_config: Dict[str, Any] = field(default_factory=dict)  # Runtime config from dashboard
    weekend_research_progress: Dict[str, Any] = field(default_factory=dict)  # Generation, strategy, etc.
    last_phase_transition: Optional[datetime] = None  # For settle period tracking


class DailyOrchestrator:
    """
    Master orchestrator for the trading system.

    Coordinates all daily operations across different market phases.
    """

    # Phase configurations (Eastern Time)
    PHASE_CONFIGS = {
        MarketPhase.PRE_MARKET: PhaseConfig(
            start_hour=8, start_minute=0,
            end_hour=9, end_minute=30,
            tasks=["refresh_premarket_data", "refresh_intraday_data", "refresh_data", "system_check", "sync_positions_from_broker", "review_positions", "cancel_stale_orders", "update_regime_detection", "calculate_position_scalars", "load_live_strategies"],
            check_interval_seconds=60
        ),
        MarketPhase.INTRADAY_OPEN: PhaseConfig(
            start_hour=9, start_minute=30,
            end_hour=9, end_minute=35,
            tasks=["start_intraday_stream", "detect_gaps"],
            check_interval_seconds=5  # Fast checking during open
        ),
        MarketPhase.INTRADAY_ACTIVE: PhaseConfig(
            start_hour=9, start_minute=35,
            end_hour=11, end_minute=30,
            tasks=["monitor_intraday_positions"],
            check_interval_seconds=10
        ),
        MarketPhase.MARKET_OPEN: PhaseConfig(
            start_hour=9, start_minute=30,
            end_hour=16, end_minute=0,
            tasks=["run_scheduler", "monitor_positions", "check_risk_limits", "score_pending_signals", "process_shadow_trades"],
            check_interval_seconds=30
        ),
        MarketPhase.POST_MARKET: PhaseConfig(
            start_hour=16, start_minute=0,
            end_hour=17, end_minute=0,
            tasks=["reconcile_positions", "calculate_pnl", "generate_daily_report", "send_alerts", "update_ensemble_correlations", "update_paper_metrics", "update_live_metrics", "run_promotion_pipeline"],
            check_interval_seconds=60
        ),
        MarketPhase.EVENING: PhaseConfig(
            start_hour=17, start_minute=0,
            end_hour=21, end_minute=30,
            tasks=["refresh_eod_data", "cleanup_logs", "backup_databases", "cleanup_databases"],
            check_interval_seconds=300
        ),
        MarketPhase.OVERNIGHT: PhaseConfig(
            start_hour=21, start_minute=30,
            end_hour=8, end_minute=0,  # Next day
            tasks=["run_nightly_research", "train_ml_regime_model"],
            check_interval_seconds=600
        ),
        MarketPhase.WEEKEND: PhaseConfig(
            start_hour=16, start_minute=0,  # Friday 4 PM start
            end_hour=20, end_minute=0,  # Sunday 8 PM end
            tasks=["run_weekend_schedule"],  # Master dispatcher handles sub-phases
            check_interval_seconds=1800  # 30 min checks
        ),
    }

    def __init__(self, paper_mode: bool = True):
        """
        Initialize the orchestrator.

        Args:
            paper_mode: If True, use paper trading. If False, live trading.
        """
        self.paper_mode = paper_mode
        self.tz = pytz.timezone('US/Eastern')
        self.state = OrchestratorState()
        self.shutdown_event = threading.Event()

        # Initialize components (lazy loading)
        self._data_manager: Optional[CachedDataManager] = None
        self._scheduler: Optional[StrategyScheduler] = None
        self._execution_tracker: Optional[ExecutionTracker] = None
        self._broker: Optional[AlpacaConnector] = None
        self._alert_manager: Optional[AlertManager] = None
        self._circuit_breaker: Optional[CircuitBreakerManager] = None

        # Intelligence components (lazy loading)
        self._signal_scorer = None
        self._shadow_trader = None
        self._execution_manager = None
        self._strategy_ensemble = None
        self._ml_regime_detector = None
        self._promotion_pipeline = None
        self._volatility_manager = None
        self._strategy_loader = None

        # Intraday components
        self._stream_handler = None
        self._intraday_strategies: List = []
        self._intraday_positions: Dict[str, Any] = {}  # Track intraday positions

        # Hardware status interface (Pi 5 LEDs)
        self._hardware: Optional[HardwareStatus] = None

        # VIX cache for efficient display updates
        self._vix_cache: float = 0.0
        self._vix_cache_time: float = 0.0
        self._vix_cache_ttl: float = 30.0  # Cache VIX for 30 seconds

        if HAS_HARDWARE:
            try:
                self._hardware = get_hardware_status()
                logger.info("Hardware status interface initialized")
            except Exception as e:
                logger.warning(f"Hardware initialization failed: {e}")

        # Task registry
        self._task_registry: Dict[str, Callable] = {
            # Pre-market tasks
            "refresh_premarket_data": self._task_refresh_premarket_data,
            "refresh_data": self._task_refresh_data,
            "system_check": self._task_system_check,
            "sync_positions_from_broker": self._task_sync_positions_from_broker,
            "review_positions": self._task_review_positions,
            "cancel_stale_orders": self._task_cancel_stale_orders,
            "refresh_intraday_data": self._task_refresh_intraday_data,

            # Market hours tasks
            "run_scheduler": self._task_run_scheduler,
            "monitor_positions": self._task_monitor_positions,
            "check_risk_limits": self._task_check_risk_limits,

            # Post-market tasks
            "reconcile_positions": self._task_reconcile_positions,
            "calculate_pnl": self._task_calculate_pnl,
            "generate_daily_report": self._task_generate_daily_report,
            "send_alerts": self._task_send_alerts,

            # Evening tasks
            "refresh_eod_data": self._task_refresh_eod_data,
            "cleanup_logs": self._task_cleanup_logs,
            "backup_databases": self._task_backup_databases,

            # Overnight tasks
            "run_nightly_research": self._task_run_nightly_research,

            # Intraday tasks
            "start_intraday_stream": self._task_start_intraday_stream,
            "detect_gaps": self._task_detect_gaps,
            "monitor_intraday_positions": self._task_monitor_intraday_positions,
            "stop_intraday_stream": self._task_stop_intraday_stream,
            "check_rapid_gains": self._task_check_rapid_gains,

            # Intelligence tasks
            "score_pending_signals": self._task_score_pending_signals,
            "process_shadow_trades": self._task_process_shadow_trades,
            "update_ensemble_correlations": self._task_update_ensemble_correlations,
            "update_regime_detection": self._task_update_regime_detection,
            "train_ml_regime_model": self._task_train_ml_regime_model,

            # Strategy lifecycle tasks
            "run_promotion_pipeline": self._task_run_promotion_pipeline,
            "calculate_position_scalars": self._task_calculate_position_scalars,
            "update_paper_metrics": self._task_update_paper_metrics,
            "update_live_metrics": self._task_update_live_metrics,
            "load_live_strategies": self._task_load_live_strategies,

            # Weekend tasks
            "run_weekend_schedule": self._task_run_weekend_schedule,
            "generate_weekly_report": self._task_generate_weekly_report,
            "vacuum_databases": self._task_vacuum_databases,
            "cleanup_databases": self._task_cleanup_databases,
            "run_weekend_research": self._task_run_weekend_research,
            "refresh_index_constituents": self._task_refresh_index_constituents,
            "refresh_fundamentals": self._task_refresh_fundamentals,
            "refresh_pairs_correlations": self._task_refresh_pairs_correlations,
            "validate_strategies": self._task_validate_strategies,
            "verify_system_readiness": self._task_verify_system_readiness,
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(f"Orchestrator initialized (paper_mode={paper_mode})")

    # =========================================================================
    # Component Access (Lazy Loading)
    # =========================================================================

    @property
    def data_manager(self) -> CachedDataManager:
        if self._data_manager is None:
            self._data_manager = CachedDataManager()
        return self._data_manager

    @property
    def scheduler(self) -> StrategyScheduler:
        if self._scheduler is None:
            # Use create_default_scheduler to register all enabled strategies
            self._scheduler = create_default_scheduler(circuit_breaker=self.circuit_breaker)
            logger.info(f"Scheduler initialized with strategies: {list(self._scheduler.strategies.keys())}")
        return self._scheduler

    @property
    def execution_tracker(self) -> ExecutionTracker:
        if self._execution_tracker is None:
            self._execution_tracker = ExecutionTracker()
        return self._execution_tracker

    @property
    def broker(self) -> AlpacaConnector:
        if self._broker is None:
            self._broker = AlpacaConnector(paper=self.paper_mode)
        return self._broker

    @property
    def alert_manager(self) -> AlertManager:
        if self._alert_manager is None:
            self._alert_manager = AlertManager()
        return self._alert_manager

    @property
    def circuit_breaker(self) -> CircuitBreakerManager:
        if self._circuit_breaker is None:
            config = CircuitBreakerConfig.from_dict(CIRCUIT_BREAKER)
            self._circuit_breaker = CircuitBreakerManager(config)
            # Inject dependencies
            self._circuit_breaker.inject_dependencies(
                broker=self.broker,
                alert_manager=self.alert_manager
            )
            logger.info("Circuit breaker manager initialized")
        return self._circuit_breaker

    @property
    def signal_scorer(self):
        """Signal scoring for trade filtering."""
        if self._signal_scorer is None and HAS_SIGNAL_SCORER:
            db_path = Path(__file__).parent / "data" / "signal_scores.db"
            self._signal_scorer = SignalScorer(db_path=db_path)
            logger.info("Signal scorer initialized")
        return self._signal_scorer

    @property
    def shadow_trader(self):
        """Shadow trading for strategy validation."""
        if self._shadow_trader is None and HAS_SHADOW_TRADER:
            db_path = Path(__file__).parent / "db" / "trades.db"
            self._shadow_trader = ShadowTrader(db_path=db_path)
            logger.info("Shadow trader initialized")
        return self._shadow_trader

    @property
    def execution_manager(self):
        """Central execution authority for all trades."""
        if self._execution_manager is None and HAS_EXECUTION_MANAGER:
            config = ExecutionManagerConfig()
            self._execution_manager = ExecutionManager(config)
            # Inject all dependencies
            self._execution_manager.inject_dependencies(
                signal_scorer=self.signal_scorer,
                shadow_trader=self.shadow_trader,
                circuit_breaker=self.circuit_breaker,
                execution_tracker=self.execution_tracker,
                signal_database=self.execution_tracker.db if self.execution_tracker else None,
                promotion_pipeline=self.promotion_pipeline
            )
            logger.info("ExecutionManager initialized with dependencies")
        return self._execution_manager

    @property
    def strategy_ensemble(self):
        """Ensemble intelligence for correlation tracking."""
        if self._strategy_ensemble is None and HAS_ENSEMBLE:
            self._strategy_ensemble = StrategyEnsemble()
            logger.info("Strategy ensemble initialized")
        return self._strategy_ensemble

    @property
    def ml_regime_detector(self):
        """ML-based regime detection."""
        if self._ml_regime_detector is None and HAS_ML_REGIME:
            self._ml_regime_detector = MLRegimeDetector()
            logger.info("ML regime detector initialized")
        return self._ml_regime_detector

    @property
    def promotion_pipeline(self):
        """Strategy promotion pipeline for lifecycle management."""
        if self._promotion_pipeline is None and HAS_PROMOTION_PIPELINE:
            self._promotion_pipeline = PromotionPipeline()

            # Set callbacks for dynamic strategy loading/unloading
            self._promotion_pipeline.set_callbacks(
                on_promotion=self._on_strategy_promoted,
                on_retirement=self._on_strategy_retired
            )

            logger.info("Promotion pipeline initialized with callbacks")
        return self._promotion_pipeline

    def _on_strategy_promoted(self, strategy_id: str) -> None:
        """Callback when a strategy is promoted to LIVE."""
        logger.info(f"Strategy {strategy_id} promoted to LIVE - reloading into scheduler")
        try:
            loader = self.strategy_loader
            if loader and loader.available:
                loader.reload_strategy(strategy_id)
        except Exception as e:
            logger.warning(f"Failed to load promoted strategy {strategy_id}: {e}")

    def _on_strategy_retired(self, strategy_id: str) -> None:
        """Callback when a strategy is retired."""
        logger.info(f"Strategy {strategy_id} retired - unloading from scheduler")
        try:
            loader = self.strategy_loader
            if loader:
                loader.unload_strategy(strategy_id)
        except Exception as e:
            logger.warning(f"Failed to unload retired strategy {strategy_id}: {e}")

    @property
    def volatility_manager(self):
        """Volatility-based position sizing manager."""
        if self._volatility_manager is None and HAS_VOLATILITY_MANAGER:
            self._volatility_manager = VolatilityManager()
            logger.info("Volatility manager initialized")
        return self._volatility_manager

    @property
    def strategy_loader(self):
        """Strategy loader for discovered GP strategies."""
        if self._strategy_loader is None and HAS_PROMOTION_PIPELINE:
            try:
                from execution.strategy_loader import StrategyLoader
                self._strategy_loader = StrategyLoader(
                    promotion_pipeline=self.promotion_pipeline,
                    scheduler=self.scheduler
                )
                logger.info("Strategy loader initialized")
            except ImportError as e:
                logger.debug(f"Strategy loader not available: {e}")
            except Exception as e:
                logger.warning(f"Strategy loader initialization failed: {e}")
        return self._strategy_loader

    # =========================================================================
    # Phase Detection
    # =========================================================================

    def get_current_phase(self) -> MarketPhase:
        """Determine the current market phase based on time.

        Note: Intraday phases (INTRADAY_OPEN, INTRADAY_ACTIVE) take precedence
        over MARKET_OPEN during their specific time windows.
        """
        now = datetime.now(self.tz)
        weekday = now.weekday()

        # Weekend check (Saturday=5, Sunday=6)
        if weekday >= 5:
            return MarketPhase.WEEKEND

        # Friday after market close until Monday pre-market
        if weekday == 4 and now.hour >= 17:
            return MarketPhase.WEEKEND

        current_time = now.hour * 60 + now.minute

        # Intraday phases take precedence during their windows
        # These must be checked BEFORE MARKET_OPEN since they overlap
        intraday_phases = {
            MarketPhase.INTRADAY_OPEN: (9 * 60 + 30, 9 * 60 + 35),    # 9:30-9:35
            MarketPhase.INTRADAY_ACTIVE: (9 * 60 + 35, 11 * 60 + 30),  # 9:35-11:30
        }

        for phase, (start, end) in intraday_phases.items():
            if start <= current_time < end:
                return phase

        # Standard phase times (checked after intraday phases)
        phase_times = {
            MarketPhase.PRE_MARKET: (8 * 60, 9 * 60 + 30),
            MarketPhase.MARKET_OPEN: (9 * 60 + 30, 16 * 60),  # 9:30-16:00 (after intraday window)
            MarketPhase.POST_MARKET: (16 * 60, 17 * 60),
            MarketPhase.EVENING: (17 * 60, 21 * 60 + 30),
            # Overnight spans midnight
        }

        for phase, (start, end) in phase_times.items():
            if start <= current_time < end:
                return phase

        # If not in any defined phase, it's overnight
        return MarketPhase.OVERNIGHT

    def get_phase_config(self, phase: MarketPhase) -> PhaseConfig:
        """Get configuration for a phase."""
        return self.PHASE_CONFIGS.get(phase, PhaseConfig(0, 0, 0, 0))

    def time_until_next_phase(self) -> timedelta:
        """Calculate time until the next phase transition."""
        now = datetime.now(self.tz)
        current_phase = self.get_current_phase()

        if current_phase == MarketPhase.WEEKEND:
            # Calculate time until Monday 8:00 AM
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_monday = now.replace(hour=8, minute=0, second=0, microsecond=0)
            next_monday += timedelta(days=days_until_monday)
            return next_monday - now

        config = self.get_phase_config(current_phase)
        next_phase_time = now.replace(
            hour=config.end_hour,
            minute=config.end_minute,
            second=0,
            microsecond=0
        )

        # Handle overnight phase (spans midnight)
        if current_phase == MarketPhase.OVERNIGHT and now.hour >= 21:
            next_phase_time += timedelta(days=1)

        if next_phase_time <= now:
            next_phase_time += timedelta(days=1)

        return next_phase_time - now

    # =========================================================================
    # Task Implementations - Pre-Market
    # =========================================================================

    def _task_refresh_data(self) -> bool:
        """Refresh market data before market open."""
        logger.info("Refreshing market data...")
        try:
            start = time.time()
            # Limit symbols for Pi memory
            max_symbols = 50
            all_syms = self.data_manager.get_available_symbols()
            symbols_to_load = sorted(all_syms)[:max_symbols]
            count = self.data_manager.load_all(symbols=symbols_to_load)
            elapsed = time.time() - start
            logger.info(f"Loaded {count} symbols in {elapsed:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            return False

    def _task_refresh_premarket_data(self) -> bool:
        """Fetch fresh daily data from Alpaca before market open.

        This ensures strategies have yesterday's closing prices and any
        pre-market moves before generating signals.
        """
        logger.info("Fetching fresh pre-market data from Alpaca...")
        try:
            from data.fetchers.daily_bars import DailyBarsFetcher
            from data.fetchers.vix import VIXFetcher

            start = time.time()
            fetcher = DailyBarsFetcher()

            # Get symbols we need to refresh - prioritize open positions + strategy universe
            symbols_to_refresh = set()

            # 1. Add symbols from open positions (critical)
            try:
                open_positions = self.execution_tracker.db.get_open_positions()
                for pos in open_positions:
                    symbols_to_refresh.add(pos.get('symbol', pos.get('ticker', '')))
                logger.info(f"Added {len(open_positions)} symbols from open positions")
            except Exception as e:
                logger.warning(f"Could not get open positions: {e}")

            # 2. Add core ETFs and major indices
            core_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'VIX', 'TLT', 'GLD', 'XLF', 'XLE', 'XLK']
            symbols_to_refresh.update(core_symbols)

            # 3. Add random sample of universe for broader coverage (limit for Pi memory)
            import random
            all_symbols = self.data_manager.get_available_symbols()
            sample_size = min(100, len(all_symbols))  # Max 100 additional symbols
            if all_symbols:
                random.seed(int(datetime.now().strftime('%Y%m%d')))  # Consistent daily sample
                sample = random.sample(all_symbols, sample_size)
                symbols_to_refresh.update(sample)

            symbols_list = list(symbols_to_refresh)
            logger.info(f"Refreshing {len(symbols_list)} symbols for pre-market")

            # Fetch fresh data
            success_count = 0
            fail_count = 0
            for i, symbol in enumerate(symbols_list):
                try:
                    df = fetcher.fetch_symbol(symbol, force=True)  # Force fresh fetch
                    if df is not None and not df.empty:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.debug(f"Failed to refresh {symbol}: {e}")

                # Progress logging every 50 symbols
                if (i + 1) % 50 == 0:
                    logger.info(f"Pre-market refresh progress: {i + 1}/{len(symbols_list)}")

            # Also refresh VIX
            try:
                vix_fetcher = VIXFetcher()
                vix_df = vix_fetcher.fetch_from_yahoo(days=30)
                if vix_df is not None and not vix_df.empty:
                    current_vix = vix_fetcher.get_current_vix()
                    logger.info(f"VIX refreshed: current={current_vix:.2f}")
            except Exception as e:
                logger.warning(f"VIX refresh failed: {e}")

            elapsed = time.time() - start
            logger.info(f"Pre-market data refresh complete: {success_count} success, {fail_count} failed in {elapsed:.1f}s")

            return success_count > 0

        except Exception as e:
            logger.error(f"Pre-market data refresh failed: {e}")
            return False

    def _task_refresh_intraday_data(self) -> bool:
        """Refresh intraday (minute bar) data for gap-fill strategy.

        Downloads recent intraday bars from Alpaca for the gap-fill universe
        (SPY, QQQ, IWM, DIA) to ensure the gap detection has fresh data.
        """
        logger.info("Refreshing intraday data for gap-fill strategy...")
        try:
            from data.fetchers.intraday_bars import IntradayDataManager

            start = time.time()
            manager = IntradayDataManager()

            # Download last 5 trading days of minute bars
            # This ensures we have yesterday's data for gap calculation
            results = manager.download_recent(days=5)

            # Log results
            total_days = sum(results.values())
            logger.info(f"Intraday data refresh: {total_days} days across {len(results)} symbols")
            for symbol, days in results.items():
                logger.debug(f"  {symbol}: {days} days")

            # Log data status
            status = manager.get_data_status()
            for symbol, info in status.items():
                if info['newest']:
                    logger.info(f"  {symbol}: {info['days_available']} days ({info['oldest']} to {info['newest']})")
                else:
                    logger.warning(f"  {symbol}: No intraday data available")

            elapsed = time.time() - start
            logger.info(f"Intraday data refresh completed in {elapsed:.1f}s")

            return True

        except Exception as e:
            logger.error(f"Intraday data refresh failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _task_refresh_eod_data(self) -> bool:
        """Fetch end-of-day data for the full universe after market close.

        This captures the complete day's trading data so overnight research
        and next day's pre-market scan have fresh EOD prices.
        """
        logger.info("Fetching EOD data for full universe...")
        try:
            from data.fetchers.daily_bars import DailyBarsFetcher
            from data.fetchers.vix import VIXFetcher

            start = time.time()
            fetcher = DailyBarsFetcher()

            # Get all available symbols for full refresh
            all_symbols = fetcher.get_available_symbols()
            if not all_symbols:
                # Fallback to cached symbols
                all_symbols = self.data_manager.get_available_symbols()

            logger.info(f"EOD refresh starting for {len(all_symbols)} symbols")

            success_count = 0
            fail_count = 0

            for i, symbol in enumerate(all_symbols):
                try:
                    df = fetcher.fetch_symbol(symbol, force=True)
                    if df is not None and not df.empty:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.debug(f"Failed to refresh {symbol}: {e}")

                # Progress logging every 100 symbols
                if (i + 1) % 100 == 0:
                    logger.info(f"EOD refresh progress: {i + 1}/{len(all_symbols)} ({success_count} success)")

            # Refresh VIX data
            try:
                vix_fetcher = VIXFetcher()
                vix_df = vix_fetcher.fetch_from_yahoo(days=365)
                if vix_df is not None and not vix_df.empty:
                    current_vix = vix_fetcher.get_current_vix()
                    logger.info(f"VIX EOD refresh: {len(vix_df)} rows, current={current_vix:.2f}")
            except Exception as e:
                logger.warning(f"VIX EOD refresh failed: {e}")

            elapsed = time.time() - start
            logger.info(f"EOD data refresh complete: {success_count}/{len(all_symbols)} symbols in {elapsed:.1f}s")

            # Log to performance database
            try:
                import sqlite3
                conn = sqlite3.connect(str(DATABASES["performance"]))
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_refresh_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        refresh_type TEXT,
                        total_symbols INTEGER,
                        success INTEGER,
                        failed INTEGER,
                        duration_sec REAL
                    )
                """)
                cursor.execute("""
                    INSERT INTO data_refresh_log (timestamp, refresh_type, total_symbols, success, failed, duration_sec)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), 'eod', len(all_symbols), success_count, fail_count, elapsed))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to log EOD refresh: {e}")

            return success_count > len(all_symbols) * 0.5  # Success if >50% fetched

        except Exception as e:
            logger.error(f"EOD data refresh failed: {e}")
            return False

    def _task_system_check(self) -> bool:
        """Run system health checks."""
        logger.info("Running system checks...")
        checks = {
            "api_credentials": False,
            "data_manager": False,
            "broker_connection": False,
            "database": False,
            "disk_space": False,
        }

        try:
            # Check API credentials are configured
            if ALPACA_API_KEY and ALPACA_SECRET_KEY:
                if len(ALPACA_API_KEY) > 10 and len(ALPACA_SECRET_KEY) > 10:
                    checks["api_credentials"] = True
                else:
                    logger.error("API credentials appear invalid (too short)")
            else:
                logger.error("ALPACA_API_KEY or ALPACA_SECRET_KEY not configured!")

            # Check data manager
            if self.data_manager.cache:
                checks["data_manager"] = True
            else:
                syms = self.data_manager.get_available_symbols()[:10]; self.data_manager.load_all(symbols=syms)  # Only 10 for check
                checks["data_manager"] = len(self.data_manager.cache) > 0

            # Check broker connection (validates credentials actually work)
            try:
                account = self.broker.get_account()
                if account is not None:
                    checks["broker_connection"] = True
                    logger.info(f"Broker account verified: ${float(account.equity):,.2f} equity")
                else:
                    logger.error("Broker returned None account - check API credentials")
            except Exception as e:
                logger.error(f"Broker check failed (likely bad credentials): {e}")

            # Check database
            try:
                from data.storage.db_manager import get_db
                db = get_db()
                checks["database"] = db is not None
            except Exception as e:
                logger.warning(f"Database check failed: {e}")

            # Check disk space
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            checks["disk_space"] = free_gb > 5  # At least 5GB free

            passed = sum(checks.values())
            total = len(checks)
            logger.info(f"System checks: {passed}/{total} passed")

            for check, status in checks.items():
                status_str = "OK" if status else "FAILED"
                logger.info(f"  {check}: {status_str}")

            # Broker connection is critical - fail if not working
            if not checks["broker_connection"]:
                logger.error("CRITICAL: Broker connection failed - trading disabled until resolved")
                return False

            return passed >= 4  # At least 4 of 5 must pass

        except Exception as e:
            logger.error(f"System check error: {e}")
            return False

    def _startup_recovery(self) -> bool:
        """
        Run startup recovery sequence to ensure clean state after restart/crash.

        This handles:
        1. Expire/cancel orphaned signals that could cause duplicate trades
        2. Validate broker positions match local database
        3. Log any discrepancies for manual review

        Returns:
            True if recovery completed successfully
        """
        logger.info("=" * 60)
        logger.info("STARTUP RECOVERY SEQUENCE")
        logger.info("=" * 60)

        recovery_success = True

        try:
            # 1. Clean up orphaned signals (prevents duplicate trades)
            logger.info("Step 1: Cleaning up orphaned signals...")
            try:
                signal_db = self.execution_tracker.db
                cleanup_results = signal_db.cleanup_orphaned_signals(max_age_hours=24)

                if cleanup_results['expired'] > 0:
                    logger.warning(f"Expired {cleanup_results['expired']} stale pending signals")
                if cleanup_results['cancelled'] > 0:
                    logger.warning(f"Cancelled {cleanup_results['cancelled']} stale submitted signals")
                if cleanup_results['orphaned_positions'] > 0:
                    logger.error(f"ATTENTION: Found {cleanup_results['orphaned_positions']} orphaned positions - need broker sync")
                    recovery_success = False
            except Exception as e:
                logger.error(f"Signal cleanup failed: {e}")
                recovery_success = False

            # 2. Validate API credentials before anything else
            logger.info("Step 2: Validating broker credentials...")
            try:
                if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                    logger.error("CRITICAL: API credentials not configured!")
                    recovery_success = False
                else:
                    account = self.broker.get_account()
                    if account:
                        logger.info(f"Broker connection OK - Account equity: ${float(account.equity):,.2f}")
                    else:
                        logger.error("CRITICAL: Broker returned no account data")
                        recovery_success = False
            except Exception as e:
                logger.error(f"CRITICAL: Broker validation failed: {e}")
                recovery_success = False

            # 3. Sync local positions with broker positions
            logger.info("Step 3: Syncing positions with broker...")
            try:
                local_positions = self.execution_tracker.db.get_open_positions()
                broker_positions = self.broker.get_positions()

                local_symbols = {p.symbol for p in local_positions}
                broker_symbols = {p.symbol for p in broker_positions}

                # Find discrepancies
                in_local_not_broker = local_symbols - broker_symbols
                in_broker_not_local = broker_symbols - local_symbols

                if in_local_not_broker or in_broker_not_local:
                    if in_local_not_broker:
                        logger.warning(f"Positions in DB but not broker: {in_local_not_broker}")
                    if in_broker_not_local:
                        logger.warning(f"Positions in broker but not DB: {in_broker_not_local}")

                    # Actually perform the sync now (don't wait for pre-market)
                    logger.info("Performing immediate position sync...")
                    new_count, updated_count, closed_count = self.broker.sync_positions(
                        self.execution_tracker.db
                    )
                    logger.info(f"Position sync complete: {new_count} added, {updated_count} updated, {closed_count} closed")
                else:
                    logger.info(f"Position reconciliation OK - {len(broker_symbols)} positions match")

            except Exception as e:
                logger.error(f"Position sync failed: {e}")
                # Don't fail recovery for this - pre-market sync task is backup

            # 4. Check for market holiday
            logger.info("Step 4: Checking market calendar...")
            try:
                is_holiday = self._is_market_holiday()
                if is_holiday:
                    logger.warning("TODAY IS A MARKET HOLIDAY - Trading will be skipped")
                else:
                    logger.info("Market is open today")
            except Exception as e:
                logger.warning(f"Could not check market calendar: {e}")

            logger.info("=" * 60)
            if recovery_success:
                logger.info("STARTUP RECOVERY COMPLETE - System ready")
            else:
                logger.error("STARTUP RECOVERY COMPLETED WITH ERRORS - Review above")
            logger.info("=" * 60)

            return recovery_success

        except Exception as e:
            logger.error(f"Startup recovery failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _is_market_holiday(self) -> bool:
        """
        Check if today is a US market holiday.

        Returns:
            True if market is closed for holiday
        """
        try:
            # Try to use Alpaca's calendar API
            from datetime import date
            from alpaca.trading.requests import GetCalendarRequest
            today = date.today()

            # Get calendar from broker's trading client
            request = GetCalendarRequest(start=today, end=today)
            calendar = self.broker.trading_client.get_calendar(request)

            if not calendar:
                # No trading day returned = holiday
                logger.info(f"No market calendar entry for {today} - likely holiday")
                return True

            # Calendar returns trading days only
            cal_date = calendar[0].date if hasattr(calendar[0], 'date') else str(calendar[0])
            if str(cal_date) != str(today):
                return True

            return False

        except Exception as e:
            logger.warning(f"Could not check market calendar via API: {e}")

            # Fallback: hardcoded 2026 NYSE holidays
            from datetime import date
            today = date.today()

            NYSE_HOLIDAYS_2026 = [
                date(2026, 1, 1),   # New Year's Day
                date(2026, 1, 19),  # MLK Day
                date(2026, 2, 16),  # Presidents Day
                date(2026, 4, 3),   # Good Friday
                date(2026, 5, 25),  # Memorial Day
                date(2026, 6, 19),  # Juneteenth
                date(2026, 7, 3),   # Independence Day (observed)
                date(2026, 9, 7),   # Labor Day
                date(2026, 11, 26), # Thanksgiving
                date(2026, 12, 25), # Christmas
            ]

            return today in NYSE_HOLIDAYS_2026

    def _task_sync_positions_from_broker(self) -> bool:
        """Sync local position database with broker's actual positions.

        This ensures our local state matches the broker's truth, catching:
        - Positions opened/closed externally
        - Fills that occurred overnight
        - Any drift between local and broker state
        """
        logger.info("Syncing positions from broker...")
        try:
            # Get the signal database from execution tracker
            signal_db = self.execution_tracker.db

            # Perform the sync
            new_count, updated_count, closed_count = self.broker.sync_positions(signal_db)

            logger.info(f"Position sync complete: {new_count} new, {updated_count} updated, {closed_count} closed")

            # Also store broker account equity for use by position sizing
            account = self.broker.get_account()
            if account:
                self.state.daily_stats['broker_equity'] = account.equity
                self.state.daily_stats['broker_buying_power'] = account.buying_power
                logger.info(f"Broker equity: ${account.equity:,.2f}, Buying power: ${account.buying_power:,.2f}")

            return True

        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return False

    def _task_review_positions(self) -> bool:
        """Review current positions before market open."""
        logger.info("Reviewing positions...")
        try:
            positions = self.broker.get_positions()
            if not positions:
                logger.info("No open positions")
                return True

            logger.info(f"Current positions: {len(positions)}")
            total_value = 0
            for pos in positions:
                # BrokerPosition is a dataclass with attributes
                symbol = pos.symbol
                qty = pos.qty
                market_value = pos.market_value
                unrealized_pl = pos.unrealized_pnl
                total_value += market_value
                logger.info(f"  {symbol}: {qty} shares, ${market_value:.2f} (P&L: ${unrealized_pl:.2f})")

            logger.info(f"Total position value: ${total_value:.2f}")
            return True

        except Exception as e:
            logger.error(f"Position review failed: {e}")
            return False

    def _task_cancel_stale_orders(self) -> bool:
        """Cancel any stale orders from previous day."""
        logger.info("Checking for stale orders...")
        try:
            orders = self.broker.get_open_orders()
            if not orders:
                logger.info("No open orders")
                return True

            now = datetime.now(self.tz)
            stale_count = 0

            for order in orders:
                # BrokerOrder is a dataclass with attributes
                order_time = order.submitted_at
                if order_time.tzinfo is None:
                    order_time = self.tz.localize(order_time)
                else:
                    order_time = order_time.astimezone(self.tz)

                age = now - order_time

                # Cancel orders older than 1 day
                if age > timedelta(days=1):
                    logger.info(f"Canceling stale order {order.id} (age: {age})")
                    try:
                        self.broker.cancel_order(order.id)
                        stale_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order.id}: {e}")

            logger.info(f"Canceled {stale_count} stale orders")
            return True

        except Exception as e:
            logger.error(f"Stale order check failed: {e}")
            return False

    # =========================================================================
    # Task Implementations - Market Hours
    # =========================================================================

    def _task_run_scheduler(self) -> bool:
        """Run the strategy scheduler during market hours."""
        logger.info("Starting strategy scheduler...")

        if self.state.scheduler_thread and self.state.scheduler_thread.is_alive():
            logger.info("Scheduler already running")
            return True

        try:
            # Setup and start scheduler in background thread
            self.scheduler.setup_schedule()

            # Check for and run any strategies that missed their scheduled time today
            # This handles late starts (e.g., starting at 11 AM when gap_fill was at 9:31)
            missed = self.scheduler.run_missed_strategies()
            if missed:
                logger.info(f"Ran {len(missed)} missed strategies on startup: {missed}")

            def scheduler_loop():
                while not self.shutdown_event.is_set() and MarketHours.is_market_open():
                    try:
                        import schedule
                        schedule.run_pending()
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Scheduler error: {e}")
                        time.sleep(5)
                logger.info("Scheduler loop ended")

            self.state.scheduler_thread = threading.Thread(
                target=scheduler_loop,
                name="StrategyScheduler",
                daemon=True
            )
            self.state.scheduler_thread.start()
            logger.info("Scheduler started in background thread")
            return True

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def _task_monitor_positions(self) -> bool:
        """Monitor positions for risk limits and TP/SL exits during market hours."""
        try:
            positions = self.broker.get_positions()
            account = self.broker.get_account()

            if not account:
                logger.warning("Could not get account info")
                return False

            # AccountInfo is a dataclass
            equity = account.equity

            # Get exit config
            exit_config = INTRADAY_EXIT_CONFIG
            tp_pct = exit_config.get("take_profit_pct", 0.10)
            sl_pct = exit_config.get("stop_loss_pct", 0.08)
            exit_enabled = exit_config.get("enabled", True)
            log_checks = exit_config.get("log_checks", False)

            exits_executed = []

            for pos in positions:
                # BrokerPosition is a dataclass
                symbol = pos.symbol
                market_value = pos.market_value
                concentration = market_value / equity if equity > 0 else 0

                # Check position concentration
                if concentration > 0.20:  # 20% max per position
                    logger.warning(f"High concentration: {symbol} is {concentration:.1%} of portfolio")
                    self.alert_manager.send_alert(
                        f"Position concentration warning: {symbol} is {concentration:.1%} of portfolio",
                        level="warning"
                    )

                # Check TP/SL exits
                if exit_enabled:
                    entry_price = float(pos.avg_entry_price)
                    current_price = float(pos.current_price)
                    qty = int(float(pos.qty))

                    if entry_price <= 0 or qty <= 0:
                        continue

                    gain_pct = (current_price - entry_price) / entry_price

                    if log_checks:
                        logger.debug(f"{symbol}: entry=${entry_price:.2f}, current=${current_price:.2f}, gain={gain_pct:.2%}")

                    # Check take-profit
                    if gain_pct >= tp_pct:
                        logger.info(f"TAKE PROFIT: {symbol} hit +{gain_pct:.2%} (threshold: +{tp_pct:.0%})")
                        exit_result = self._execute_exit(
                            symbol=symbol,
                            qty=qty,
                            entry_price=entry_price,
                            current_price=current_price,
                            reason="take_profit",
                            gain_pct=gain_pct
                        )
                        if exit_result:
                            exits_executed.append(exit_result)

                    # Check stop-loss
                    elif gain_pct <= -sl_pct:
                        logger.info(f"STOP LOSS: {symbol} hit {gain_pct:.2%} (threshold: -{sl_pct:.0%})")
                        exit_result = self._execute_exit(
                            symbol=symbol,
                            qty=qty,
                            entry_price=entry_price,
                            current_price=current_price,
                            reason="stop_loss",
                            gain_pct=gain_pct
                        )
                        if exit_result:
                            exits_executed.append(exit_result)

            # Send summary alert if any exits
            if exits_executed:
                tp_exits = [e for e in exits_executed if e.get("reason") == "take_profit"]
                sl_exits = [e for e in exits_executed if e.get("reason") == "stop_loss"]
                summary = f"Executed {len(exits_executed)} exits: {len(tp_exits)} TP, {len(sl_exits)} SL"
                logger.info(summary)
                self.alert_manager.send_alert(summary, level="info")

            return True

        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _execute_exit(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        current_price: float,
        reason: str,
        gain_pct: float
    ) -> Optional[Dict[str, Any]]:
        """Execute an exit order for a position.

        Args:
            symbol: Stock symbol
            qty: Number of shares to sell
            entry_price: Original entry price
            current_price: Current market price
            reason: Exit reason ('take_profit' or 'stop_loss')
            gain_pct: Percentage gain/loss

        Returns:
            Dict with exit details if successful, None if failed
        """
        try:
            # Submit market sell order (waits for fill by default)
            order = self.broker.submit_market_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                time_in_force='day'
            )

            if order:
                # Use ACTUAL filled quantity and price (handles partial fills)
                filled_qty = int(order.filled_qty) if order.filled_qty else 0
                filled_price = order.filled_avg_price if order.filled_avg_price else current_price

                if filled_qty == 0:
                    logger.error(f"Exit order for {symbol} not filled (status={order.status})")
                    return None

                # Warn about partial fills
                if filled_qty < qty:
                    logger.warning(
                        f"PARTIAL EXIT: {symbol} requested {qty} shares, only {filled_qty} filled. "
                        f"Remaining {qty - filled_qty} shares still open!"
                    )

                # Calculate P&L using actual filled quantity
                pnl = filled_qty * (filled_price - entry_price)
                actual_gain_pct = (filled_price - entry_price) / entry_price if entry_price > 0 else 0

                logger.info(
                    f"EXIT EXECUTED: {symbol} - Sold {filled_qty} shares @ ${filled_price:.2f} "
                    f"({reason}, P&L: ${pnl:+.2f})"
                )

                # Record in database with actual filled values
                self._record_exit_trade(
                    symbol=symbol,
                    qty=filled_qty,  # Use actual filled qty
                    entry_price=entry_price,
                    exit_price=filled_price,
                    reason=reason,
                    pnl=pnl,
                    gain_pct=actual_gain_pct
                )

                return {
                    "symbol": symbol,
                    "qty": filled_qty,  # Return actual filled qty
                    "requested_qty": qty,
                    "entry_price": entry_price,
                    "exit_price": filled_price,
                    "reason": reason,
                    "pnl": pnl,
                    "gain_pct": actual_gain_pct,
                    "partial_fill": filled_qty < qty
                }
            else:
                logger.error(f"Exit order failed for {symbol}: no order returned")
                return None

        except Exception as e:
            logger.error(f"Failed to execute exit for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _record_exit_trade(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        reason: str,
        pnl: float,
        gain_pct: float
    ) -> None:
        """Record an exit trade in the database and update strategy stats."""
        try:
            import sqlite3
            from config import DATABASES

            conn = sqlite3.connect(str(DATABASES['trades']))
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # Look up original strategy from positions table
            cursor.execute("""
                SELECT strategy_name FROM positions
                WHERE symbol = ? AND status = 'open'
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            original_strategy = row[0] if row else 'unknown'

            # Update positions table - mark as closed
            cursor.execute("""
                UPDATE positions
                SET status = 'closed',
                    closed_at = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    realized_pnl = ?
                WHERE symbol = ? AND status = 'open'
            """, (now, exit_price, reason, pnl, symbol))

            # Record in trades table with ORIGINAL strategy (not 'intraday_exit')
            cursor.execute("""
                INSERT OR IGNORE INTO trades (
                    timestamp, symbol, strategy, side, quantity,
                    entry_price, exit_price, exit_timestamp,
                    pnl, pnl_percent, status, exit_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, symbol, original_strategy, 'SELL', qty,
                entry_price, exit_price, now,
                pnl, gain_pct * 100, 'CLOSED', reason,
                now, now
            ))

            conn.commit()
            conn.close()

            # Update strategy stats in performance database
            self._update_strategy_stats(original_strategy, pnl, gain_pct, is_win=(pnl > 0))

            logger.info(f"Recorded exit for {symbol} ({original_strategy}): {reason}, P&L ${pnl:.2f}")

        except Exception as e:
            logger.warning(f"Failed to record exit trade for {symbol}: {e}")

    def _update_strategy_stats(
        self,
        strategy: str,
        pnl: float,
        pnl_pct: float,
        is_win: bool
    ) -> None:
        """Update strategy performance stats after a trade closes."""
        try:
            import sqlite3
            from config import DATABASES

            perf_db = DATABASES.get('performance')
            if not perf_db:
                return

            conn = sqlite3.connect(str(perf_db))
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # Upsert strategy stats
            cursor.execute("""
                INSERT INTO strategy_stats (strategy, total_trades, winning_trades, losing_trades,
                                           total_pnl, last_trade_date, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy) DO UPDATE SET
                    total_trades = total_trades + 1,
                    winning_trades = winning_trades + ?,
                    losing_trades = losing_trades + ?,
                    total_pnl = total_pnl + ?,
                    avg_pnl = (total_pnl + ?) / (total_trades + 1),
                    win_rate = CAST(winning_trades + ? AS REAL) / (total_trades + 1),
                    best_trade = MAX(best_trade, ?),
                    worst_trade = MIN(worst_trade, ?),
                    last_trade_date = ?,
                    updated_at = ?
            """, (
                strategy,
                1 if is_win else 0,  # winning_trades for INSERT
                0 if is_win else 1,  # losing_trades for INSERT
                pnl,                 # total_pnl for INSERT
                now, now,            # dates for INSERT
                1 if is_win else 0,  # winning_trades increment
                0 if is_win else 1,  # losing_trades increment
                pnl,                 # total_pnl increment
                pnl,                 # for avg calculation
                1 if is_win else 0,  # for win_rate calculation
                pnl if pnl > 0 else None,  # best_trade
                pnl if pnl < 0 else None,  # worst_trade
                now, now
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Updated strategy_stats for {strategy}: pnl=${pnl:.2f}, win={is_win}")

        except Exception as e:
            logger.warning(f"Failed to update strategy stats for {strategy}: {e}")

    def _task_check_risk_limits(self) -> bool:
        """Check overall portfolio risk limits and circuit breakers."""
        try:
            account = self.broker.get_account()
            if not account:
                return False

            # AccountInfo is a dataclass
            equity = account.equity
            buying_power = account.buying_power

            # Initialize daily stats if not set
            if 'start_equity' not in self.state.daily_stats:
                self.state.daily_stats['start_equity'] = equity
            if 'peak_equity' not in self.state.daily_stats:
                self.state.daily_stats['peak_equity'] = equity

            # Update peak tracking
            if equity > self.state.daily_stats.get('peak_equity', 0):
                self.state.daily_stats['peak_equity'] = equity

            # Build context for circuit breaker checks
            context = {
                'current_equity': equity,
                'start_of_day_equity': self.state.daily_stats.get('start_equity', equity),
                'peak_equity': self.state.daily_stats.get('peak_equity', equity),
            }

            # Run circuit breaker checks (includes file-based kill switches)
            triggered = self.circuit_breaker.check_all(context)
            if triggered:
                for state in triggered:
                    logger.warning(f"Circuit breaker triggered: {state.reason}")

            # Log circuit breaker status
            status = self.circuit_breaker.get_status()
            if not status['trading_allowed']:
                logger.warning("Trading is currently HALTED by circuit breaker")
            if status['position_multiplier'] < 1.0:
                logger.info(f"Position size multiplier: {status['position_multiplier']}")

            # Check margin usage
            margin_used = equity - buying_power
            margin_pct = margin_used / equity if equity > 0 else 0

            if margin_pct > 0.80:  # 80% margin usage warning
                logger.warning(f"High margin usage: {margin_pct:.1%}")
                self.alert_manager.send_alert(
                    f"High margin usage: {margin_pct:.1%}",
                    level="warning"
                )

            return True

        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return False

    # =========================================================================
    # Task Implementations - Post-Market
    # =========================================================================

    def _task_reconcile_positions(self) -> bool:
        """Reconcile positions with broker - trim oversized positions to target."""
        logger.info("Reconciling positions...")
        try:
            # 1. Save position snapshot first
            positions = self.broker.get_positions()
            snapshot = {
                'timestamp': datetime.now(self.tz).isoformat(),
                'positions': [
                    {
                        'symbol': p.symbol,
                        'qty': float(p.qty),
                        'market_value': float(p.market_value) if hasattr(p, 'market_value') else 0,
                        'current_price': float(p.current_price),
                        'unrealized_pnl': float(p.unrealized_pnl) if hasattr(p, 'unrealized_pnl') else 0,
                    }
                    for p in positions
                ]
            }
            snapshot_file = LOG_DIR / f"position_snapshot_{datetime.now().strftime('%Y%m%d')}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            logger.info(f"Position snapshot saved: {snapshot_file}")

            # 2. Run position reconciliation through ExecutionManager
            from execution.execution_manager import ExecutionManager
            exec_manager = ExecutionManager()
            results = exec_manager.reconcile_positions(dry_run=False)

            # Log results
            trimmed = [r for r in results if r.action == 'trimmed']
            if trimmed:
                for r in trimmed:
                    logger.info(f"  Trimmed {r.symbol}: {r.shares_trimmed} shares (${r.trim_value:.2f})")

            return True

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
            return False

    def _task_calculate_pnl(self) -> bool:
        """Calculate daily P&L after market close."""
        logger.info("Calculating daily P&L...")
        try:
            account = self.broker.get_account()
            if not account:
                return False

            # AccountInfo is a dataclass
            equity = account.equity
            portfolio_value = account.portfolio_value

            # Calculate unrealized P&L from positions
            positions = self.broker.get_positions()
            unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)

            self.state.daily_stats['equity'] = equity
            self.state.daily_stats['portfolio_value'] = portfolio_value
            self.state.daily_stats['unrealized_pnl'] = unrealized_pnl
            self.state.daily_stats['cash'] = account.cash

            logger.info(f"Account equity: ${equity:.2f}")
            logger.info(f"Portfolio value: ${portfolio_value:.2f}")
            logger.info(f"Unrealized P&L: ${unrealized_pnl:.2f}")
            logger.info(f"Cash: ${account.cash:.2f}")

            return True

        except Exception as e:
            logger.error(f"P&L calculation failed: {e}")
            return False

    def _task_generate_daily_report(self) -> bool:
        """Generate comprehensive end-of-day trading journal."""
        logger.info("Generating daily report...")
        try:
            import sqlite3
            from config import DATABASES

            now = datetime.now(self.tz)
            today_str = now.strftime('%Y-%m-%d')
            report_file = LOG_DIR / f"daily_report_{now.strftime('%Y%m%d')}.txt"

            # Fetch today's data from databases
            trades_today = []
            positions_opened = []
            positions_closed = []
            open_positions = []
            strategy_performance = {}

            try:
                conn = sqlite3.connect(str(DATABASES['trades']))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get today's closed trades
                cursor.execute("""
                    SELECT symbol, strategy, side, quantity, entry_price, exit_price,
                           pnl, pnl_percent, exit_reason, timestamp
                    FROM trades
                    WHERE date(timestamp) = date('now', 'localtime')
                      AND status = 'CLOSED'
                    ORDER BY timestamp DESC
                """)
                trades_today = [dict(row) for row in cursor.fetchall()]

                # Get positions opened today
                cursor.execute("""
                    SELECT symbol, strategy_name, quantity, entry_price, take_profit, stop_loss, opened_at
                    FROM positions
                    WHERE date(opened_at) = date('now', 'localtime')
                    ORDER BY opened_at DESC
                """)
                positions_opened = [dict(row) for row in cursor.fetchall()]

                # Get positions closed today
                cursor.execute("""
                    SELECT symbol, strategy_name, quantity, entry_price, exit_price,
                           realized_pnl, exit_reason, closed_at
                    FROM positions
                    WHERE date(closed_at) = date('now', 'localtime')
                      AND status = 'closed'
                    ORDER BY closed_at DESC
                """)
                positions_closed = [dict(row) for row in cursor.fetchall()]

                # Get current open positions
                cursor.execute("""
                    SELECT symbol, strategy_name, quantity, entry_price, current_price,
                           unrealized_pnl, take_profit, stop_loss
                    FROM positions
                    WHERE status = 'open'
                    ORDER BY symbol
                """)
                open_positions = [dict(row) for row in cursor.fetchall()]

                conn.close()
            except Exception as e:
                logger.warning(f"Error fetching trade data: {e}")

            # Get strategy performance from performance db
            try:
                perf_conn = sqlite3.connect(str(DATABASES['performance']))
                perf_conn.row_factory = sqlite3.Row
                cursor = perf_conn.cursor()
                cursor.execute("""
                    SELECT strategy, total_trades, winning_trades, total_pnl, win_rate
                    FROM strategy_stats
                    WHERE total_trades > 0
                    ORDER BY total_pnl DESC
                """)
                for row in cursor.fetchall():
                    strategy_performance[row['strategy']] = dict(row)
                perf_conn.close()
            except Exception as e:
                logger.warning(f"Error fetching strategy stats: {e}")

            # Get account info
            account = self.broker.get_account() if self.broker else None
            broker_positions = self.broker.get_positions() if self.broker else []

            # Calculate daily stats
            total_realized = sum(t.get('pnl', 0) or 0 for t in trades_today)
            winning_trades = [t for t in trades_today if (t.get('pnl') or 0) > 0]
            losing_trades = [t for t in trades_today if (t.get('pnl') or 0) < 0]
            best_trade = max(trades_today, key=lambda x: x.get('pnl') or 0) if trades_today else None
            worst_trade = min(trades_today, key=lambda x: x.get('pnl') or 0) if trades_today else None

            # Calculate unrealized P&L from broker positions
            total_unrealized = sum(float(p.unrealized_pnl or 0) for p in broker_positions)

            with open(report_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write(f"  DAILY TRADING JOURNAL - {today_str}\n")
                f.write("=" * 70 + "\n\n")

                # ===== DAILY P&L SUMMARY =====
                f.write("📊 DAILY P&L SUMMARY\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Realized P&L:      ${total_realized:>+10.2f}\n")
                f.write(f"  Unrealized P&L:    ${total_unrealized:>+10.2f}\n")
                f.write(f"  Combined:          ${total_realized + total_unrealized:>+10.2f}\n")
                f.write(f"\n")
                f.write(f"  Trades Today:      {len(trades_today):>10}\n")
                f.write(f"  Winners:           {len(winning_trades):>10}  (${sum(t.get('pnl', 0) or 0 for t in winning_trades):+.2f})\n")
                f.write(f"  Losers:            {len(losing_trades):>10}  (${sum(t.get('pnl', 0) or 0 for t in losing_trades):+.2f})\n")
                if trades_today:
                    win_rate = len(winning_trades) / len(trades_today) * 100
                    f.write(f"  Win Rate:          {win_rate:>9.1f}%\n")
                f.write("\n")

                # ===== ACCOUNT STATUS =====
                f.write("💰 ACCOUNT STATUS\n")
                f.write("-" * 50 + "\n")
                if account:
                    f.write(f"  Equity:            ${account.equity:>10,.2f}\n")
                    f.write(f"  Cash:              ${account.cash:>10,.2f}\n")
                    f.write(f"  Buying Power:      ${account.buying_power:>10,.2f}\n")
                f.write(f"  Open Positions:    {len(broker_positions):>10}\n")
                f.write("\n")

                # ===== TODAY'S TRADES =====
                f.write("📈 TODAY'S TRADES\n")
                f.write("-" * 50 + "\n")
                if trades_today:
                    for t in trades_today:
                        pnl = t.get('pnl') or 0
                        pnl_pct = t.get('pnl_percent') or 0
                        icon = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                        f.write(f"  {icon} {t['symbol']:6} | {t['strategy']:25} | "
                               f"${pnl:>+8.2f} ({pnl_pct:>+5.1f}%) | {t.get('exit_reason', '-')}\n")
                else:
                    f.write("  No trades executed today\n")
                f.write("\n")

                # ===== BEST & WORST TRADES =====
                if best_trade or worst_trade:
                    f.write("🏆 BEST & WORST TRADES\n")
                    f.write("-" * 50 + "\n")
                    if best_trade and (best_trade.get('pnl') or 0) > 0:
                        f.write(f"  Best:  {best_trade['symbol']:6} | ${best_trade.get('pnl', 0):>+.2f} | {best_trade.get('strategy', '-')}\n")
                    if worst_trade and (worst_trade.get('pnl') or 0) < 0:
                        f.write(f"  Worst: {worst_trade['symbol']:6} | ${worst_trade.get('pnl', 0):>+.2f} | {worst_trade.get('strategy', '-')}\n")
                    f.write("\n")

                # ===== STRATEGY PERFORMANCE =====
                f.write("📊 STRATEGY PERFORMANCE (All-Time)\n")
                f.write("-" * 50 + "\n")
                if strategy_performance:
                    for strat, stats in strategy_performance.items():
                        win_rate = (stats.get('win_rate') or 0) * 100
                        f.write(f"  {strat:25} | {stats['winning_trades']}/{stats['total_trades']} wins | "
                               f"${stats['total_pnl']:>+8.2f} | {win_rate:.0f}%\n")
                else:
                    f.write("  No strategy data available\n")
                f.write("\n")

                # ===== OPEN POSITIONS =====
                f.write("📋 OPEN POSITIONS (Overnight Holdings)\n")
                f.write("-" * 50 + "\n")
                if broker_positions:
                    for p in sorted(broker_positions, key=lambda x: float(x.unrealized_pnl or 0), reverse=True):
                        pnl = float(p.unrealized_pnl or 0)
                        icon = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                        entry = float(p.avg_entry_price)
                        current = float(p.current_price)
                        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                        f.write(f"  {icon} {p.symbol:6} | {int(float(p.qty)):>4} shares | "
                               f"${entry:>7.2f} → ${current:>7.2f} | ${pnl:>+8.2f} ({pnl_pct:>+5.1f}%)\n")
                else:
                    f.write("  No open positions\n")
                f.write("\n")

                # ===== POSITIONS OPENED TODAY =====
                if positions_opened:
                    f.write("🆕 POSITIONS OPENED TODAY\n")
                    f.write("-" * 50 + "\n")
                    for p in positions_opened:
                        f.write(f"  {p['symbol']:6} | {p.get('strategy_name', '-'):20} | "
                               f"{p.get('quantity', 0)} @ ${p.get('entry_price', 0):.2f}\n")
                    f.write("\n")

                # ===== ERRORS & ISSUES =====
                if self.state.errors_today:
                    f.write("⚠️ ERRORS & ISSUES\n")
                    f.write("-" * 50 + "\n")
                    seen = set()
                    for error in self.state.errors_today:
                        err_key = f"{error.get('task')}:{error.get('error')}"
                        if err_key not in seen:
                            f.write(f"  • {error.get('task')}: {error.get('error')[:60]}\n")
                            seen.add(err_key)
                    f.write("\n")

                # ===== FOOTER =====
                f.write("=" * 70 + "\n")
                f.write(f"  Report generated at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
                f.write("=" * 70 + "\n")

            logger.info(f"Daily report saved: {report_file}")
            return True

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _task_send_alerts(self) -> bool:
        """Send end-of-day alert summary."""
        logger.info("Sending daily alerts...")
        try:
            equity = self.state.daily_stats.get('equity', 0)
            unrealized_pnl = self.state.daily_stats.get('unrealized_pnl', 0)
            cash = self.state.daily_stats.get('cash', 0)

            message = (
                f"Daily Summary:\n"
                f"  Equity: ${equity:.2f}\n"
                f"  Unrealized P&L: ${unrealized_pnl:.2f}\n"
                f"  Cash: ${cash:.2f}\n"
                f"  Tasks completed: {len(self.state.tasks_completed_today)}\n"
                f"  Errors: {len(self.state.errors_today)}"
            )

            self.alert_manager.send_alert(message, level="info")
            logger.info("Daily alert sent")
            return True

        except Exception as e:
            logger.error(f"Alert sending failed: {e}")
            return False

    # =========================================================================
    # Task Implementations - Evening
    # =========================================================================

    def _task_cleanup_logs(self) -> bool:
        """Clean up old log files."""
        logger.info("Cleaning up old logs...")
        try:
            retention_days = 30
            cutoff = datetime.now() - timedelta(days=retention_days)

            cleaned = 0
            for log_file in LOG_DIR.glob("*.log.*"):
                if log_file.stat().st_mtime < cutoff.timestamp():
                    log_file.unlink()
                    cleaned += 1

            # Also clean old reports
            for report_file in LOG_DIR.glob("daily_report_*.txt"):
                if report_file.stat().st_mtime < cutoff.timestamp():
                    report_file.unlink()
                    cleaned += 1

            logger.info(f"Cleaned {cleaned} old files")
            return True

        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
            return False

    def _task_backup_databases(self) -> bool:
        """Backup databases."""
        logger.info("Backing up databases...")
        try:
            # This is a placeholder - implement based on your backup strategy
            # For SQLite, you might copy the .db files
            # For other databases, use appropriate backup commands

            db_dir = Path(__file__).parent / "data" / "storage"
            backup_dir = Path(__file__).parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            import shutil
            today = datetime.now().strftime('%Y%m%d')

            for db_file in db_dir.glob("*.db"):
                backup_path = backup_dir / f"{db_file.stem}_{today}.db"
                if not backup_path.exists():
                    shutil.copy2(db_file, backup_path)
                    logger.info(f"Backed up {db_file.name}")

            # Clean old backups (keep 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            for backup in backup_dir.glob("*_*.db"):
                if backup.stat().st_mtime < cutoff.timestamp():
                    backup.unlink()

            return True

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False

    def _task_cleanup_databases(self) -> bool:
        """Clean up stale/useless data from databases."""
        logger.info("Cleaning up database tables...")
        try:
            import sqlite3
            db_dir = Path(__file__).parent / "db"
            cleaned_total = 0

            # Clean research.db - remove useless GA runs
            research_db = db_dir / "research.db"
            if research_db.exists():
                conn = sqlite3.connect(research_db)
                cursor = conn.cursor()

                # Delete GA runs with 0 generations that aren't running
                cursor.execute("""
                    DELETE FROM ga_runs
                    WHERE total_generations = 0
                    AND status NOT IN ('running', 'paused')
                """)
                ga_cleaned = cursor.rowcount
                cleaned_total += ga_cleaned

                # Delete old backtest results (keep 90 days)
                cursor.execute("""
                    DELETE FROM backtests
                    WHERE timestamp < datetime('now', '-90 days')
                """)
                bt_cleaned = cursor.rowcount
                cleaned_total += bt_cleaned

                conn.commit()
                conn.close()

                if ga_cleaned > 0 or bt_cleaned > 0:
                    logger.info(f"Cleaned {ga_cleaned} useless GA runs, {bt_cleaned} old backtests")

            # Clean performance.db - resolve old errors
            perf_db = db_dir / "performance.db"
            if perf_db.exists():
                conn = sqlite3.connect(perf_db)
                cursor = conn.cursor()

                # Auto-resolve errors older than 7 days
                cursor.execute("""
                    UPDATE error_log
                    SET is_resolved = 1, resolved_by = 'auto_cleanup'
                    WHERE is_resolved = 0
                    AND timestamp < datetime('now', '-7 days')
                """)
                errors_resolved = cursor.rowcount
                cleaned_total += errors_resolved

                # Delete resolved errors older than 30 days
                cursor.execute("""
                    DELETE FROM error_log
                    WHERE is_resolved = 1
                    AND timestamp < datetime('now', '-30 days')
                """)
                errors_deleted = cursor.rowcount
                cleaned_total += errors_deleted

                conn.commit()
                conn.close()

                if errors_resolved > 0 or errors_deleted > 0:
                    logger.info(f"Resolved {errors_resolved} old errors, deleted {errors_deleted} stale entries")

            # VACUUM databases to reclaim space (weekly - on Sundays)
            if datetime.now().weekday() == 6:  # Sunday
                for db_file in db_dir.glob("*.db"):
                    try:
                        conn = sqlite3.connect(db_file)
                        conn.execute("VACUUM")
                        conn.close()
                        logger.debug(f"Vacuumed {db_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to vacuum {db_file.name}: {e}")

            logger.info(f"Database cleanup complete: {cleaned_total} items cleaned")
            return True

        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            return False

    # =========================================================================
    # Task Implementations - Overnight
    # =========================================================================

    def _task_run_nightly_research(self) -> bool:
        """Run nightly research engine."""
        logger.info("Starting nightly research...")

        # Check if already run today (success)
        last_run = self.state.last_task_run.get("run_nightly_research")
        if last_run:
            hours_since = (datetime.now(self.tz) - last_run).total_seconds() / 3600
            if hours_since < 20:  # Don't run more than once per day
                logger.info(f"Nightly research already ran {hours_since:.1f} hours ago")
                return True

        # Check if recently failed - use 1 hour cooldown to prevent retry storm
        last_attempt = self.state.last_task_run.get("run_nightly_research_attempt")
        if last_attempt:
            hours_since_attempt = (datetime.now(self.tz) - last_attempt).total_seconds() / 3600
            if hours_since_attempt < 1.0:  # 1 hour cooldown after failure
                logger.info(f"Nightly research attempted {hours_since_attempt:.1f}h ago (failed), waiting for cooldown")
                return True  # Return True to mark phase complete, retry after cooldown

        try:
            import subprocess

            research_script = Path(__file__).parent / "run_nightly_research.py"
            if not research_script.exists():
                logger.warning("Nightly research script not found")
                return False

            # Pre-flight system check - wait for memory and load to be reasonable
            self._preflight_system_check(min_memory_mb=1500, max_load=3.0)

            # Start research LED breathing and update display with full data
            if self._hardware:
                self._hardware.set_research_active(True)
                # Push full display data first
                self._update_hardware_display(MarketPhase.OVERNIGHT)
                # Then update research-specific fields
                self._hardware.update_display({
                    'research_status': 'STARTING',
                    'research_generation': 0,
                    'research_max_gen': 100,
                    'research_best_sharpe': 0,
                })

            # Track that we're attempting research (for retry cooldown)
            self.state.last_task_run["run_nightly_research_attempt"] = datetime.now(self.tz)

            # Run as subprocess so it doesn't block
            logger.info("Launching nightly research subprocess...")
            process = subprocess.Popen(
                [sys.executable, str(research_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent)
            )

            # Poll for completion with periodic display updates
            timeout_hours = 4
            timeout_secs = timeout_hours * 3600
            display_update_interval = 30  # Update display every 30 seconds
            start_time = time.time()
            last_display_update = 0

            while True:
                # Check if process finished
                if process.poll() is not None:
                    break

                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_secs:
                    logger.warning(f"Nightly research timed out after {timeout_hours} hours")
                    process.kill()
                    if self._hardware:
                        self._hardware.set_research_active(False)
                    return False

                # Check for shutdown
                if self.shutdown_event.is_set():
                    logger.info("Shutdown requested during research - terminating")
                    process.terminate()
                    if self._hardware:
                        self._hardware.set_research_active(False)
                    return False

                # Update display periodically
                if time.time() - last_display_update >= display_update_interval:
                    self._update_hardware_display(MarketPhase.OVERNIGHT)
                    # Also update research status
                    if self._hardware:
                        self._hardware.update_display({
                            'research_status': 'EVOLVING',
                        })
                    last_display_update = time.time()

                # Brief sleep before next poll
                time.sleep(1)

            # Process finished - check result
            if process.returncode == 0:
                logger.info("Nightly research completed successfully")
                if self._hardware:
                    self._hardware.set_research_complete()
                return True
            else:
                logger.error(f"Nightly research failed with code {process.returncode}")
                if self._hardware:
                    self._hardware.set_research_active(False)
                return False

        except Exception as e:
            logger.error(f"Nightly research failed: {e}")
            if self._hardware:
                self._hardware.set_research_active(False)
            return False

    # =========================================================================
    # Task Implementations - Intraday
    # =========================================================================

    def _task_start_intraday_stream(self) -> bool:
        """Start the real-time data stream for intraday strategies.

        This task initializes the market data stream handler and prepares
        intraday strategies for the trading session.
        """
        logger.info("Starting intraday data stream...")

        # Check if already running
        if self._stream_handler is not None:
            logger.info("Intraday stream already running")
            return True

        # Check if stream handler is available
        if not HAS_STREAM_HANDLER:
            logger.warning("MarketDataStream not available - intraday streaming disabled")
            return False

        # Validate API credentials before attempting to create stream
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.error("Cannot start intraday stream: ALPACA_API_KEY or ALPACA_SECRET_KEY not configured")
            return False

        try:
            # Initialize stream handler with API credentials
            self._stream_handler = MarketDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

            # Initialize intraday strategies if available
            if HAS_GAP_FILL:
                try:
                    gap_config = GapFillConfig()
                    gap_strategy = GapFillStrategy(config=gap_config)
                    self._intraday_strategies.append(gap_strategy)
                    logger.info("GapFillStrategy initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize GapFillStrategy: {e}")

            # Start the stream
            self._stream_handler.start()
            logger.info("Intraday data stream started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start intraday stream: {e}")
            logger.debug(traceback.format_exc())
            self._stream_handler = None
            return False

    def _task_detect_gaps(self) -> bool:
        """Run gap detection for intraday strategies.

        This task runs during the market open window (9:30-9:35) to detect
        gap opportunities and potentially initiate positions.
        """
        logger.info("Running gap detection...")

        if not self._intraday_strategies:
            logger.warning("No intraday strategies configured for gap detection")
            return True  # Not an error, just nothing to do

        try:
            gaps_detected = 0
            signals_generated = 0

            for strategy in self._intraday_strategies:
                # Check if strategy has gap detection capability
                if hasattr(strategy, 'detect_gaps'):
                    try:
                        gaps = strategy.detect_gaps()
                        if gaps:
                            gaps_detected += len(gaps)
                            logger.info(f"Strategy {strategy.__class__.__name__} detected {len(gaps)} gaps")

                            # Process each gap signal
                            for gap in gaps:
                                symbol = gap.get('symbol', 'UNKNOWN')
                                gap_pct = gap.get('gap_pct', 0)
                                direction = gap.get('direction', 'unknown')
                                logger.info(f"  Gap: {symbol} {direction} {gap_pct:.2f}%")

                    except Exception as e:
                        logger.error(f"Gap detection failed for {strategy.__class__.__name__}: {e}")

                # Check if strategy can generate signals
                if hasattr(strategy, 'generate_signals'):
                    try:
                        signals = strategy.generate_signals()
                        if signals:
                            signals_generated += len(signals)
                            for signal in signals:
                                self._process_intraday_signal(signal)
                    except Exception as e:
                        logger.error(f"Signal generation failed for {strategy.__class__.__name__}: {e}")

            logger.info(f"Gap detection complete: {gaps_detected} gaps, {signals_generated} signals")
            return True

        except Exception as e:
            logger.error(f"Gap detection task failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _task_monitor_intraday_positions(self) -> bool:
        """Monitor and manage intraday positions.

        This task runs during the intraday active phase (9:35-11:30) to:
        - Monitor open intraday positions
        - Check stop losses and take profits
        - Manage position exits based on strategy rules
        """
        logger.info("Monitoring intraday positions...")

        try:
            # Get current positions
            positions = self.broker.get_positions()

            # Filter for intraday positions (tracked separately)
            intraday_symbols = set(self._intraday_positions.keys())

            for pos in positions:
                symbol = pos.symbol
                if symbol not in intraday_symbols:
                    continue

                # Get position details
                qty = pos.qty
                market_value = pos.market_value
                unrealized_pnl = pos.unrealized_pnl
                entry_price = self._intraday_positions[symbol].get('entry_price', 0)
                current_price = pos.current_price if hasattr(pos, 'current_price') else (market_value / qty if qty != 0 else 0)

                # Calculate P&L percentage
                if entry_price > 0:
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = 0

                logger.debug(f"Intraday position {symbol}: {qty} shares, P&L: ${unrealized_pnl:.2f} ({pnl_pct:.2f}%)")

                # Check strategy exit conditions
                for strategy in self._intraday_strategies:
                    if hasattr(strategy, 'check_exit_conditions'):
                        try:
                            exit_signal = strategy.check_exit_conditions(
                                symbol=symbol,
                                qty=qty,
                                entry_price=entry_price,
                                current_price=current_price,
                                unrealized_pnl=unrealized_pnl
                            )
                            if exit_signal:
                                self._execute_intraday_exit(symbol, exit_signal)
                        except Exception as e:
                            logger.error(f"Exit check failed for {symbol}: {e}")

            # Run strategy-specific monitoring
            for strategy in self._intraday_strategies:
                if hasattr(strategy, 'monitor'):
                    try:
                        strategy.monitor()
                    except Exception as e:
                        logger.error(f"Strategy monitoring failed for {strategy.__class__.__name__}: {e}")

            return True

        except Exception as e:
            logger.error(f"Intraday position monitoring failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _task_stop_intraday_stream(self) -> bool:
        """Stop intraday streaming (called at 11:30am or market close).

        This task:
        - Stops the real-time data stream
        - Closes any remaining intraday positions if configured
        - Cleans up intraday strategy resources
        """
        logger.info("Stopping intraday stream...")

        try:
            # Stop stream handler
            if self._stream_handler is not None:
                try:
                    self._stream_handler.stop()
                    logger.info("Market data stream stopped")
                except Exception as e:
                    logger.error(f"Error stopping stream handler: {e}")
                finally:
                    self._stream_handler = None

            # Cleanup strategies
            for strategy in self._intraday_strategies:
                if hasattr(strategy, 'cleanup'):
                    try:
                        strategy.cleanup()
                    except Exception as e:
                        logger.warning(f"Strategy cleanup failed for {strategy.__class__.__name__}: {e}")

            # Log final intraday summary
            if self._intraday_positions:
                total_pnl = sum(
                    pos.get('realized_pnl', 0)
                    for pos in self._intraday_positions.values()
                )
                logger.info(f"Intraday session complete. Positions: {len(self._intraday_positions)}, Total P&L: ${total_pnl:.2f}")

            # Clear intraday state
            self._intraday_strategies = []
            self._intraday_positions = {}

            logger.info("Intraday stream stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop intraday stream: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _task_check_rapid_gains(self) -> bool:
        """Check positions for rapid gains and trim if threshold met.

        This task monitors positions opened within the last 24 hours.
        If a position gains >= 3% within that window, it trims 1/3
        of the position to lock in profits. Each position is only
        trimmed once to avoid squashing runners.
        """
        if not HAS_RAPID_GAIN_SCALER:
            logger.debug("RapidGainScaler not available")
            return True

        try:
            # Configure: 3% gain in 24 hours triggers 33% trim
            config = RapidGainConfig(
                gain_threshold=0.03,
                time_window_hours=24,
                trim_fraction=0.33
            )

            scaler = RapidGainScaler(broker=self.broker, config=config)

            # Get positions with current prices from broker
            broker_positions = self.broker.get_positions()
            if not broker_positions:
                return True

            # Get position metadata from database (entry time, scaled_at)
            positions = scaler.get_eligible_positions()
            if not positions:
                return True

            # Merge broker prices with database metadata
            broker_prices = {p.symbol: float(p.current_price) for p in broker_positions}
            for pos in positions:
                symbol = pos.get('symbol')
                if symbol in broker_prices:
                    pos['current_price'] = broker_prices[symbol]

            # Check and trim
            results = scaler.check_and_trim_positions(positions)

            for result in results:
                if result.success:
                    logger.info(
                        f"RAPID GAIN TRIM: {result.symbol} - "
                        f"Sold {result.shares_trimmed}/{result.shares_before} shares "
                        f"at {result.gain_pct:.1%} gain, locked ${result.profit_locked:.2f}"
                    )
                else:
                    logger.warning(
                        f"Rapid gain trim failed for {result.symbol}: {result.error}"
                    )

            return True

        except Exception as e:
            logger.error(f"Rapid gain check failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _process_intraday_signal(self, signal: Dict[str, Any]) -> None:
        """Process an intraday trading signal.

        Args:
            signal: Dictionary containing signal details (symbol, action, qty, etc.)
        """
        try:
            symbol = signal.get('symbol')
            action = signal.get('action')  # 'buy' or 'sell'
            qty = signal.get('qty', 0)
            price = signal.get('price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            strategy_name = signal.get('strategy', 'intraday')
            confidence = signal.get('confidence', 0.5)

            if not symbol or not action or qty <= 0:
                logger.warning(f"Invalid intraday signal: {signal}")
                return

            logger.info(f"Processing intraday signal: {action.upper()} {qty} {symbol} @ {price}")

            # Log signal to database for audit trail
            try:
                direction = 'long' if action == 'buy' else 'short'
                signal_id, position_id = self.execution_tracker.record_signal_and_execute(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    direction=direction,
                    entry_price=price or 0,
                    stop_loss=stop_loss or 0,
                    take_profit=take_profit or 0,
                    quantity=qty,
                    confidence=confidence,
                    metadata={'source': 'intraday', 'action': action}
                )
                logger.info(f"Signal logged to DB: signal_id={signal_id}, position_id={position_id}")
            except Exception as e:
                logger.warning(f"Failed to log signal to DB (continuing with order): {e}")

            # Execute the order
            try:
                if action == 'buy':
                    order = self.broker.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market' if price is None else 'limit',
                        limit_price=price
                    )
                elif action == 'sell':
                    order = self.broker.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market' if price is None else 'limit',
                        limit_price=price
                    )
                else:
                    logger.warning(f"Unknown action in signal: {action}")
                    return

                # Track the intraday position
                if order:
                    self._intraday_positions[symbol] = {
                        'order_id': order.id if hasattr(order, 'id') else None,
                        'action': action,
                        'qty': qty,
                        'entry_price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now(self.tz).isoformat(),
                        'realized_pnl': 0
                    }
                    logger.info(f"Intraday order submitted: {symbol}")

            except Exception as e:
                logger.error(f"Failed to execute intraday order for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error processing intraday signal: {e}")

    def _execute_intraday_exit(self, symbol: str, exit_signal: Dict[str, Any]) -> None:
        """Execute an exit for an intraday position.

        Args:
            symbol: The symbol to exit
            exit_signal: Dictionary containing exit details
        """
        try:
            position_info = self._intraday_positions.get(symbol)
            if not position_info:
                logger.warning(f"No intraday position found for {symbol}")
                return

            reason = exit_signal.get('reason', 'unknown')
            qty = exit_signal.get('qty', position_info.get('qty', 0))

            logger.info(f"Exiting intraday position: {symbol} (reason: {reason})")

            # Determine exit side (opposite of entry)
            entry_action = position_info.get('action', 'buy')
            exit_side = 'sell' if entry_action == 'buy' else 'buy'

            try:
                order = self.broker.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=exit_side,
                    type='market'
                )

                if order:
                    # Calculate realized P&L (approximate)
                    entry_price = position_info.get('entry_price', 0)
                    # Note: actual P&L will be calculated when fill is confirmed
                    logger.info(f"Intraday exit order submitted for {symbol}")

                    # Mark position as closed
                    self._intraday_positions[symbol]['closed'] = True
                    self._intraday_positions[symbol]['exit_time'] = datetime.now(self.tz).isoformat()
                    self._intraday_positions[symbol]['exit_reason'] = reason

            except Exception as e:
                logger.error(f"Failed to execute intraday exit for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error executing intraday exit: {e}")

    # =========================================================================
    # Task Implementations - Intelligence
    # =========================================================================

    def _task_score_pending_signals(self) -> bool:
        """Evaluate and execute pending signals via ExecutionManager."""
        # Use ExecutionManager if available (preferred)
        if HAS_EXECUTION_MANAGER and self.execution_manager is not None:
            return self._process_signals_via_execution_manager()

        # Fallback to legacy scoring-only flow
        if not HAS_SIGNAL_SCORER or self.signal_scorer is None:
            logger.debug("Signal scorer not available")
            return True

        try:
            pending_signals = self.execution_tracker.get_pending_signals()
            if not pending_signals:
                logger.debug("No pending signals to score")
                return True

            scored_count = 0
            for signal in pending_signals:
                try:
                    signal_type = 'buy' if signal.direction == 'long' else 'sell'
                    score = self.signal_scorer.score_signal(
                        strategy=signal.strategy_name,
                        signal_type=signal_type,
                        signal_strength=signal.confidence if signal.confidence else 0.5
                    )

                    if score is not None:
                        scored_count += 1
                        logger.info(f"Signal {signal.id} ({signal.strategy_name} {signal.direction} {signal.symbol}): "
                                    f"conviction={score.conviction:.2f}, win_prob={score.win_probability:.2f}")

                except Exception as e:
                    logger.warning(f"Failed to score signal {signal.id}: {e}")

            logger.info(f"Scored {scored_count} pending signals (legacy mode)")
            return True

        except Exception as e:
            logger.error(f"Signal scoring failed: {e}")
            return False

    def _update_signal_route(self, signal_id: int, route: str):
        """Update the execution route for a signal."""
        try:
            conn = self.execution_tracker.db._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE signals SET execution_route = ? WHERE id = ?",
                (route, signal_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update signal route: {e}")

    def _process_signals_via_execution_manager(self) -> bool:
        """Process pending signals through the central ExecutionManager."""
        try:
            pending_signals = self.execution_tracker.get_pending_signals()

            if not pending_signals:
                logger.debug("No pending signals to process")
                return True

            approved_count = 0
            rejected_count = 0
            executed_count = 0

            for signal in pending_signals:
                try:
                    # Evaluate signal through ExecutionManager
                    decision = self.execution_manager.evaluate_signal(
                        strategy_name=signal.strategy_name,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        signal_type=signal.signal_type,
                        price=signal.price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        quantity=signal.quantity or 10,  # Default qty
                        confidence=signal.confidence or 0.5,
                        context={}
                    )

                    if decision.approved:
                        approved_count += 1
                        # Execute the decision
                        result = self.execution_manager.execute(decision)
                        if result.success:
                            executed_count += 1
                            logger.info(f"Signal {signal.id} executed via {decision.route}: "
                                        f"{signal.symbol} {signal.direction} x{decision.final_shares}")
                            # Mark original signal as executed with route
                            self.execution_tracker.db.update_signal_status(
                                signal.id, 'executed', datetime.now().isoformat()
                            )
                            # Store the execution route
                            self._update_signal_route(signal.id, decision.route)

                            # Send alert for live trades
                            if decision.route == 'live':
                                self.alert_manager.signal(
                                    symbol=signal.symbol,
                                    direction=signal.direction,
                                    price=signal.price,
                                    strategy=signal.strategy_name
                                )

                            # Alert on overrides
                            if decision.override_applied:
                                self.alert_manager.info(
                                    "Override Applied",
                                    f"{signal.symbol} executed via high-conviction override"
                                )

                            # Alert on rebalancing
                            if decision.rebalance_required:
                                self.alert_manager.info(
                                    "Rebalance Executed",
                                    f"Closed {decision.positions_to_close} to open {signal.symbol}"
                                )
                        else:
                            logger.warning(f"Signal {signal.id} execution failed: {result.error}")
                    else:
                        rejected_count += 1
                        logger.info(f"Signal {signal.id} rejected: {decision.rejection_reason}")
                        # Mark as rejected
                        self.execution_tracker.db.update_signal_status(
                            signal.id, 'rejected', datetime.now().isoformat()
                        )

                except Exception as e:
                    logger.warning(f"Failed to process signal {signal.id}: {e}")

            # Log summary
            summary = self.execution_manager.get_decision_summary(days=1)
            logger.info(f"ExecutionManager: {approved_count} approved, {rejected_count} rejected, "
                        f"{executed_count} executed | Daily: {summary.get('total_decisions', 0)} decisions, "
                        f"{summary.get('overrides_used', 0)} overrides, {summary.get('rebalances', 0)} rebalances")

            return True

        except Exception as e:
            logger.error(f"ExecutionManager signal processing failed: {e}")
            return False

    def _task_process_shadow_trades(self) -> bool:
        """Process shadow (paper) trades and check for promotion readiness.

        Uses PromotionPipeline as the authority for graduation decisions.
        ShadowTrader is only used for metrics tracking.
        """
        if not HAS_SHADOW_TRADER or self.shadow_trader is None:
            logger.debug("Shadow trader not available")
            return True

        if not HAS_PROMOTION_PIPELINE or self.promotion_pipeline is None:
            logger.debug("Promotion pipeline not available")
            return True

        try:
            # Get strategies in PAPER status from promotion pipeline
            paper_strategies = self.promotion_pipeline.get_strategies_by_status(StrategyStatus.PAPER)

            if not paper_strategies:
                logger.debug("No strategies in PAPER status")
                return True

            processed = 0
            ready_for_live = []

            for strategy_id in paper_strategies:
                try:
                    # Check if ready for promotion using PromotionPipeline criteria
                    ready, message, metrics = self.promotion_pipeline.check_paper_for_live(strategy_id)

                    if ready:
                        ready_for_live.append(strategy_id)
                        logger.info(f"Strategy '{strategy_id}' ready for LIVE: {message}")

                        # Send alert for promotion readiness
                        self.alert_manager.send_alert(
                            title=f"Strategy Ready: {strategy_id}",
                            message=f"Paper strategy '{strategy_id}' meets LIVE criteria. Metrics: {metrics}",
                            level="warning"
                        )

                    processed += 1

                except Exception as e:
                    logger.warning(f"Failed to check promotion for {strategy_id}: {e}")

            if ready_for_live:
                logger.info(f"Strategies ready for LIVE promotion: {ready_for_live}")

            logger.info(f"Checked {processed} paper strategies for promotion")
            return True

        except Exception as e:
            logger.error(f"Shadow trade processing failed: {e}")
            return False

    def _task_update_ensemble_correlations(self) -> bool:
        """Update strategy correlation matrix for ensemble intelligence."""
        if not HAS_ENSEMBLE or self.strategy_ensemble is None:
            logger.debug("Strategy ensemble not available")
            return True

        try:
            # Get today's strategy returns
            today = datetime.now(self.tz).date()
            strategy_returns = {}

            # Fetch returns from execution tracker
            for strategy_name in self.execution_tracker.get_active_strategies():
                try:
                    daily_pnl = self.execution_tracker.get_strategy_pnl(strategy_name, date=today)
                    if daily_pnl is not None:
                        strategy_returns[strategy_name] = daily_pnl
                except Exception as e:
                    logger.warning(f"Failed to get PnL for {strategy_name}: {e}")

            if not strategy_returns:
                logger.debug("No strategy returns to update correlations")
                return True

            # Update ensemble with today's returns
            for strategy_name, returns in strategy_returns.items():
                self.strategy_ensemble.update_returns(strategy_name, returns)

            # Get allocation adjustments
            adjustments = self.strategy_ensemble.get_allocation_adjustments()

            if adjustments:
                logger.info(f"Ensemble allocation adjustments: {adjustments}")

                # Check for clustering (risk)
                if self.strategy_ensemble.is_clustering():
                    logger.warning("Strategy clustering detected - consider diversification")
                    self.alert_manager.send_alert(
                        title="Strategy Clustering Warning",
                        message="Multiple strategies showing correlated behavior. Review allocations.",
                        priority="medium"
                    )

            logger.info(f"Updated ensemble correlations for {len(strategy_returns)} strategies")
            return True

        except Exception as e:
            logger.error(f"Ensemble correlation update failed: {e}")
            return False

    def _task_update_regime_detection(self) -> bool:
        """Update ML regime detection before market open."""
        if not HAS_ML_REGIME or self.ml_regime_detector is None:
            logger.debug("ML regime detector not available")
            return True

        try:
            # Get current market data for regime prediction
            market_data = self._get_market_regime_features()

            if market_data is None:
                logger.debug("No market data available for regime detection")
                return True

            # Extract required parameters for predict_regime
            vix = market_data.get('vix', 0)
            sp500_price = market_data.get('sp500_price', 0)

            if vix <= 0 or sp500_price <= 0:
                logger.debug("Invalid VIX or SP500 price for regime detection")
                return True

            # Predict current regime
            regime = self.ml_regime_detector.predict_regime(vix, sp500_price)

            if regime:
                logger.info(f"ML Regime Detection: {regime.regime} (confidence: {regime.confidence:.2f})")

                # Get strategy weights for this regime
                weights = self.ml_regime_detector.get_regime_weights(regime.regime)
                if weights:
                    logger.info(f"Recommended strategy weights: {weights}")

                    # Get previous regime for comparison
                    previous_regime = self.state.daily_stats.get('current_regime')

                    # Store regime info for other tasks
                    self.state.daily_stats['current_regime'] = regime.regime
                    self.state.daily_stats['regime_confidence'] = regime.confidence
                    self.state.daily_stats['regime_weights'] = weights

                    # Log to database if regime changed or first detection of day
                    if previous_regime != regime.regime or previous_regime is None:
                        self._log_regime_change(
                            vix_level=vix,
                            vix_regime=regime.regime,
                            previous_regime=previous_regime,
                            weights=weights
                        )

            return True

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return False

    def _log_regime_change(
        self,
        vix_level: float,
        vix_regime: str,
        previous_regime: str,
        weights: dict
    ) -> None:
        """Log regime change to performance.db:regime_log table."""
        try:
            import sqlite3
            from config import DATABASES

            db_path = DATABASES.get('performance')
            if not db_path or not db_path.exists():
                logger.warning("Performance database not found for regime logging")
                return

            # Determine action taken based on regime change
            action_taken = None
            if previous_regime and previous_regime != vix_regime:
                if vix_regime in ('high', 'extreme'):
                    action_taken = "Reducing exposure due to elevated volatility"
                elif vix_regime == 'low' and previous_regime in ('high', 'extreme'):
                    action_taken = "Increasing exposure as volatility normalizes"
                else:
                    action_taken = f"Regime transition: {previous_regime} -> {vix_regime}"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO regime_log (
                    timestamp, vix_level, vix_regime, previous_regime, action_taken, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(self.tz).isoformat(),
                vix_level,
                vix_regime,
                previous_regime,
                action_taken,
                json.dumps({'weights': weights})
            ))

            conn.commit()
            conn.close()

            logger.info(f"Regime logged to DB: {vix_regime} (VIX: {vix_level:.1f})")

        except Exception as e:
            logger.warning(f"Failed to log regime to database: {e}")

    def _task_train_ml_regime_model(self) -> bool:
        """Retrain ML regime model with recent data."""
        if not HAS_ML_REGIME or self.ml_regime_detector is None:
            logger.debug("ML regime detector not available")
            return True

        try:
            # Check if already trained recently
            last_train = self.state.last_task_run.get("train_ml_regime_model")
            if last_train:
                days_since = (datetime.now(self.tz) - last_train).days
                if days_since < 7:  # Train weekly
                    logger.debug(f"ML model trained {days_since} days ago, skipping")
                    return True

            logger.info("Training ML regime detection model...")

            # Get historical market data
            historical_data = self._get_historical_regime_data()

            if historical_data is None or len(historical_data) < 100:
                logger.warning("Insufficient historical data for ML training")
                return True

            # Train the model
            metrics = self.ml_regime_detector.train(historical_data["vix"], historical_data["spy_close"])

            if metrics:
                logger.info(f"ML model trained: accuracy={metrics.get('accuracy', 0):.2%}")

            return True

        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return False

    def _get_market_regime_features(self) -> Optional[Dict[str, float]]:
        """Get current market features for regime detection."""
        try:
            # Get current VIX using existing method
            current_vix = self.data_manager.get_vix()

            # Get SPY for market breadth
            spy_data = self.data_manager.get_bars('SPY')
            if spy_data.empty:
                return None

            # Get current SPY price
            current_spy_price = float(spy_data['close'].iloc[-1])

            # Get last 20 days of SPY returns
            spy_returns = spy_data['close'].tail(20).pct_change().dropna()

            return {
                'vix': current_vix,
                'sp500_price': current_spy_price,
                'spy_return_5d': float(spy_returns.tail(5).sum()),
                'spy_return_20d': float(spy_returns.sum()),
                'spy_volatility': float(spy_returns.std() * (252 ** 0.5)),
            }

        except Exception as e:
            logger.warning(f"Failed to get market regime features: {e}")
            return None

    def _get_historical_regime_data(self) -> Optional[pd.DataFrame]:
        """Get historical data for ML model training."""
        try:
            from config import DIRS
            
            # Load VIX from its parquet file
            vix_path = DIRS.get('vix', Path(__file__).parent / 'data' / 'historical' / 'vix') / 'vix.parquet'
            if not vix_path.exists():
                logger.warning(f'VIX data not found at {vix_path}')
                return None
            vix_data = pd.read_parquet(vix_path)
            
            # Load SPY using cached data manager
            spy_data = self.data_manager.get_bars('SPY')
            if spy_data.empty:
                logger.warning('SPY data not available')
                return None

            # Normalize timezones using centralized utility (fixes datetime comparison errors)
            vix_data = normalize_dataframe(vix_data, index_col='timestamp' if 'timestamp' in vix_data.columns else None)
            spy_data = normalize_dataframe(spy_data, index_col='timestamp' if 'timestamp' in spy_data.columns else None)

            # Get last 2 years of data (now_naive() ensures consistent comparison)
            cutoff = now_naive() - pd.Timedelta(days=730)
            vix_data = vix_data[vix_data.index >= cutoff]
            spy_data = spy_data[spy_data.index >= cutoff]

            # Combine into training dataset using inner join on dates
            df = pd.DataFrame({
                'vix': vix_data['close'],
            }).join(
                pd.DataFrame({'spy_close': spy_data['close']}),
                how='inner'
            )
            
            if len(df) < 100:
                logger.warning(f'Insufficient data for ML training: {len(df)} rows')
                return None

            # Add features
            df['spy_return'] = df['spy_close'].pct_change()
            df['vix_change'] = df['vix'].pct_change()
            df['volatility_20d'] = df['spy_return'].rolling(20).std() * (252 ** 0.5)

            return df.dropna()

        except Exception as e:
            logger.warning(f'Failed to get historical regime data: {e}')
            return None

    # =========================================================================
    # Strategy Lifecycle Tasks
    # =========================================================================

    def _task_run_promotion_pipeline(self) -> bool:
        """Run strategy promotion pipeline to advance strategies through lifecycle."""
        if not HAS_PROMOTION_PIPELINE or self.promotion_pipeline is None:
            logger.debug("Promotion pipeline not available")
            return True

        try:
            logger.info("Running strategy promotion pipeline...")

            # Process promotions for strategies at each stage
            results = self.promotion_pipeline.process_all_promotions()

            promoted = results.get('promoted', 0)
            retired = results.get('retired', 0)

            if promoted > 0 or retired > 0:
                logger.info(f"Promotion pipeline: {promoted} promoted, {retired} retired")

            return True

        except Exception as e:
            logger.error(f"Promotion pipeline failed: {e}")
            return False

    def _task_update_paper_metrics(self) -> bool:
        """Update paper trading metrics in promotion pipeline from shadow trader."""
        if not HAS_PROMOTION_PIPELINE or self.promotion_pipeline is None:
            logger.debug("Promotion pipeline not available")
            return True

        if not HAS_SHADOW_TRADER or self.shadow_trader is None:
            logger.debug("Shadow trader not available")
            return True

        try:
            logger.info("Updating paper trading metrics...")

            # Get all strategies in PAPER status
            paper_strategies = self.promotion_pipeline.get_strategies_by_status(StrategyStatus.PAPER)

            if not paper_strategies:
                logger.debug("No strategies in PAPER status")
                return True

            updated = 0
            for strategy_id in paper_strategies:
                try:
                    # Get metrics from shadow trader
                    metrics = self.shadow_trader.get_strategy_metrics(strategy_id)
                    if not metrics or 'error' in metrics:
                        continue

                    # Calculate Sharpe ratio
                    sharpe = self.shadow_trader.calculate_sharpe_ratio(strategy_id)

                    # Calculate max drawdown
                    max_dd = self.shadow_trader.get_strategy_max_drawdown(strategy_id)

                    # Update promotion pipeline
                    self.promotion_pipeline.update_paper_metrics(
                        strategy_id=strategy_id,
                        trades=int(metrics.get('total_trades', 0)),
                        pnl=float(metrics.get('total_pnl', 0)),
                        sharpe=sharpe,
                        max_drawdown=max_dd,
                        win_rate=float(metrics.get('win_rate', 0))
                    )
                    updated += 1

                except Exception as e:
                    logger.warning(f"Failed to update paper metrics for {strategy_id}: {e}")

            if updated > 0:
                logger.info(f"Updated paper metrics for {updated} strategies")

            return True

        except Exception as e:
            logger.error(f"Paper metrics update failed: {e}")
            return False

    def _task_update_live_metrics(self) -> bool:
        """Update live trading metrics in promotion pipeline from execution tracker."""
        if not HAS_PROMOTION_PIPELINE or self.promotion_pipeline is None:
            logger.debug("Promotion pipeline not available")
            return True

        try:
            logger.info("Updating live trading metrics...")

            # Get all strategies in LIVE status
            live_strategies = self.promotion_pipeline.get_strategies_by_status(StrategyStatus.LIVE)

            if not live_strategies:
                logger.debug("No strategies in LIVE status")
                return True

            updated = 0
            for strategy_id in live_strategies:
                try:
                    # Get metrics from execution tracker
                    if self._tracker:
                        metrics = self._tracker.get_strategy_performance(strategy_id)
                        if not metrics:
                            continue

                        # Calculate Sharpe from daily returns if available
                        sharpe = 0.0
                        max_dd = 0.0

                        # Use shadow trader for consistent Sharpe calculation
                        if HAS_SHADOW_TRADER and self.shadow_trader:
                            sharpe = self.shadow_trader.calculate_sharpe_ratio(strategy_id)
                            max_dd = self.shadow_trader.get_strategy_max_drawdown(strategy_id)

                        # Update promotion pipeline
                        self.promotion_pipeline.update_live_metrics(
                            strategy_id=strategy_id,
                            live_days=metrics.get('days_active', 0),
                            trades=metrics.get('total_trades', 0),
                            pnl=metrics.get('total_pnl', 0),
                            sharpe=sharpe,
                            max_drawdown=max_dd
                        )
                        updated += 1

                except Exception as e:
                    logger.warning(f"Failed to update live metrics for {strategy_id}: {e}")

            if updated > 0:
                logger.info(f"Updated live metrics for {updated} strategies")

            return True

        except Exception as e:
            logger.error(f"Live metrics update failed: {e}")
            return False

    def _task_load_live_strategies(self) -> bool:
        """Load discovered GP strategies with LIVE status into scheduler.

        This bridges the gap between the promotion pipeline (which tracks
        strategy lifecycle) and the execution scheduler (which runs strategies).
        """
        if not HAS_PROMOTION_PIPELINE:
            logger.debug("Promotion pipeline not available")
            return True

        try:
            loader = self.strategy_loader
            if loader is None:
                logger.debug("Strategy loader not available")
                return True

            if not loader.available:
                logger.debug("Strategy loader not ready (missing dependencies)")
                return True

            logger.info("Loading discovered strategies into scheduler...")

            loaded = loader.load_live_strategies()

            if loaded > 0:
                logger.info(f"Loaded {loaded} discovered strategies into scheduler")

                # Update hardware display if available
                if self._hardware:
                    self._hardware.set_status('research', 'active')

            return True

        except Exception as e:
            logger.error(f"Strategy loading failed: {e}")
            return False

    def _task_calculate_position_scalars(self) -> bool:
        """Calculate volatility-adjusted position scalars for active strategies."""
        if not HAS_VOLATILITY_MANAGER or self.volatility_manager is None:
            logger.debug("Volatility manager not available")
            return True

        try:
            logger.info("Calculating position scalars...")

            # Get active strategy names from scheduler
            strategy_names = [name for name, info in self.scheduler.strategies.items() if info.get('enabled')] if self._scheduler else []

            for strategy_name in strategy_names:
                try:
                    # Get strategy returns
                    returns = self._get_strategy_returns(strategy_name)
                    if returns is None or len(returns) < 20:
                        continue

                    # Get current regime
                    regime = self.state.daily_stats.get('market_regime', 'normal')

                    # Calculate position scalar
                    scalar = self.volatility_manager.get_position_scalar(
                        returns=returns,
                        regime=regime
                    )

                    # Store for use by execution
                    if 'position_scalars' not in self.state.daily_stats:
                        self.state.daily_stats['position_scalars'] = {}
                    self.state.daily_stats['position_scalars'][strategy_name] = scalar

                    logger.debug(f"Position scalar for {strategy_name}: {scalar:.2f}")

                except Exception as e:
                    logger.warning(f"Failed to calculate scalar for {strategy_name}: {e}")

            return True

        except Exception as e:
            logger.error(f"Position scalar calculation failed: {e}")
            return False

    def _get_strategy_returns(self, strategy_name: str) -> Optional[pd.Series]:
        """Get recent returns for a strategy."""
        try:
            # Try to load from performance database
            from data.storage.db_manager import get_db
            db = get_db()

            rows = db.fetchall(
                "performance",
                """
                SELECT date, daily_pnl_pct FROM strategy_daily
                WHERE strategy = ?
                ORDER BY date DESC
                LIMIT 60
                """,
                (strategy_name,)
            )

            if not rows or len(rows) < 20:
                return None

            returns = pd.Series(
                [r['daily_pnl_pct'] for r in rows],
                index=pd.to_datetime([r['date'] for r in rows])
            ).sort_index()

            return returns

        except Exception as e:
            logger.debug(f"Failed to get returns for {strategy_name}: {e}")
            return None

    # =========================================================================
    # Weekend Tasks
    # =========================================================================

    def _check_weekend_control_file(self) -> Optional[dict]:
        """Check for and process dashboard control commands."""
        import json
        control_file = Path(__file__).parent / "logs" / "weekend_control.json"

        if not control_file.exists():
            return None

        try:
            with open(control_file, 'r') as f:
                control = json.load(f)

            # Remove the file after reading (command consumed)
            control_file.unlink()

            return control
        except Exception as e:
            logger.warning(f"Failed to read weekend control file: {e}")
            return None

    def _apply_weekend_control(self, control: dict) -> bool:
        """Apply a weekend control command from the dashboard."""
        action = control.get('action')
        logger.info(f"Received weekend control command: {action}")

        if action == 'start':
            # Update config with dashboard settings
            self.state.weekend_config['research'] = {
                'generations_default': control.get('generations', 10),
                'population_default': control.get('population', 30),
                'discovery_enabled': control.get('discovery_enabled', True),
                'adaptive_ga_enabled': control.get('adaptive_ga_enabled', True),
                'discovery_hours': control.get('discovery_hours', 4.0),
                'strategies': control.get('strategies', []),
            }
            # Start research immediately
            self.state.weekend_sub_phase = WeekendSubPhase.RESEARCH
            return self._task_run_weekend_research()

        elif action == 'pause':
            # Kill the research process
            pid = self.state.weekend_research_progress.get('pid')
            if pid:
                try:
                    import signal
                    os.kill(pid, signal.SIGTERM)
                    self.state.weekend_research_progress['status'] = 'paused'
                    logger.info(f"Paused weekend research (killed PID {pid})")
                except ProcessLookupError:
                    logger.warning(f"Research process {pid} not found")
            return True

        elif action == 'skip':
            # Skip to next sub-phase
            current = self.state.weekend_sub_phase
            next_phases = [
                WeekendSubPhase.FRIDAY_CLEANUP,
                WeekendSubPhase.RESEARCH,
                WeekendSubPhase.DATA_REFRESH,
                WeekendSubPhase.PREWEEK_PREP,
                WeekendSubPhase.COMPLETE,
            ]
            try:
                idx = next_phases.index(current)
                if idx < len(next_phases) - 1:
                    self.state.weekend_sub_phase = next_phases[idx + 1]
                    logger.info(f"Skipped to sub-phase: {self.state.weekend_sub_phase.value}")
            except ValueError:
                pass
            return True

        elif action == 'stop':
            # Kill research and mark weekend as complete
            pid = self.state.weekend_research_progress.get('pid')
            if pid:
                try:
                    import signal
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self.state.weekend_sub_phase = WeekendSubPhase.COMPLETE
            self.state.weekend_research_progress['status'] = 'stopped'
            logger.info("Weekend research stopped")
            return True

        return False

    def _task_run_weekend_schedule(self) -> bool:
        """Master dispatcher for weekend sub-phases."""
        now = datetime.now(self.tz)

        # Check for dashboard control commands
        control = self._check_weekend_control_file()
        if control:
            return self._apply_weekend_control(control)

        # Initialize weekend state if not set
        if self.state.weekend_sub_phase is None:
            self.state.weekend_sub_phase = WeekendSubPhase.FRIDAY_CLEANUP
            self.state.weekend_started_at = now
            self.state.weekend_tasks_completed = []
            self.state.last_phase_transition = None  # Track for settle period
            # Use default config or override from dashboard
            if not self.state.weekend_config:
                self.state.weekend_config = WEEKEND_CONFIG.copy()
            logger.info("Weekend phase started - beginning with Friday cleanup")

        # Determine which sub-phase we should be in based on time
        time_based_sub_phase = self._get_weekend_sub_phase(now)

        # Get actual sub-phase considering task completion requirements
        current_sub_phase = self._get_safe_weekend_sub_phase(time_based_sub_phase)

        # Check if we need to transition sub-phases
        if current_sub_phase != self.state.weekend_sub_phase:
            old_phase = self.state.weekend_sub_phase

            # Apply settle period between cleanup and research phases
            if (old_phase == WeekendSubPhase.FRIDAY_CLEANUP and
                    current_sub_phase == WeekendSubPhase.RESEARCH):
                last_transition = getattr(self.state, 'last_phase_transition', None)
                if last_transition is None:
                    # First transition attempt - mark time and wait
                    self.state.last_phase_transition = now
                    logger.info("Cleanup complete, starting 30s settle period before research...")
                    return True  # Stay in cleanup for now
                elif (now - last_transition).total_seconds() < 30:
                    # Still in settle period
                    settle_remaining = 30 - (now - last_transition).total_seconds()
                    logger.debug(f"Settle period: {settle_remaining:.0f}s remaining")
                    return True
                else:
                    # Settle period complete, clear the marker
                    self.state.last_phase_transition = None

            self.state.weekend_sub_phase = current_sub_phase
            logger.info(f"Weekend sub-phase transition: {old_phase.value} -> {current_sub_phase.value}")

        # Run tasks for current sub-phase
        return self._run_weekend_sub_phase_tasks(current_sub_phase)

    def _get_safe_weekend_sub_phase(self, time_based_phase: WeekendSubPhase) -> WeekendSubPhase:
        """
        Determine safe sub-phase considering both time AND task completion.

        Don't transition to a new phase until tasks in the current phase are complete.
        This prevents starting heavy research while cleanup is still running.
        """
        current_phase = self.state.weekend_sub_phase

        # Define required tasks for each phase to be considered "complete"
        phase_requirements = {
            WeekendSubPhase.FRIDAY_CLEANUP: [
                "generate_weekly_report",
                "backup_databases",
                "vacuum_databases",
            ],
            WeekendSubPhase.RESEARCH: [],  # No completion requirement (runs in background)
            WeekendSubPhase.DATA_REFRESH: [
                "refresh_index_constituents",
                "refresh_fundamentals",
            ],
            WeekendSubPhase.PREWEEK_PREP: [
                "train_ml_regime_model",
                "validate_strategies",
                "verify_system_readiness",
            ],
        }

        # If time says we should move to a later phase, check if current phase is complete
        if time_based_phase != current_phase:
            required_tasks = phase_requirements.get(current_phase, [])
            completed_tasks = self.state.weekend_tasks_completed

            # Check if all required tasks for current phase are done
            incomplete = [t for t in required_tasks if t not in completed_tasks]
            if incomplete:
                logger.debug(f"Cannot transition from {current_phase.value}: incomplete tasks: {incomplete}")
                return current_phase  # Stay in current phase

        return time_based_phase

    def _get_weekend_sub_phase(self, now: datetime) -> WeekendSubPhase:
        """Determine the current weekend sub-phase based on time."""
        day = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        hour = now.hour

        # Friday (after market close)
        if day == 4 and hour >= 16:
            if hour < 20:
                return WeekendSubPhase.FRIDAY_CLEANUP
            else:
                return WeekendSubPhase.RESEARCH

        # Saturday - all day research
        if day == 5:
            return WeekendSubPhase.RESEARCH

        # Sunday
        if day == 6:
            if hour < 8:
                return WeekendSubPhase.RESEARCH  # Overnight research continues
            elif hour < 14:
                return WeekendSubPhase.DATA_REFRESH
            elif hour < 18:
                return WeekendSubPhase.PREWEEK_PREP
            else:
                return WeekendSubPhase.COMPLETE

        return WeekendSubPhase.COMPLETE

    def _run_weekend_sub_phase_tasks(self, sub_phase: WeekendSubPhase) -> bool:
        """Run tasks for a specific weekend sub-phase."""
        tasks = {
            WeekendSubPhase.FRIDAY_CLEANUP: [
                "generate_weekly_report",
                "backup_databases",
                "vacuum_databases",
            ],
            WeekendSubPhase.RESEARCH: [
                "run_weekend_research",
            ],
            WeekendSubPhase.DATA_REFRESH: [
                "refresh_index_constituents",
                "refresh_fundamentals",
                # "refresh_pairs_correlations",  # Disabled - loads too much data for Pi
            ],
            WeekendSubPhase.PREWEEK_PREP: [
                "train_ml_regime_model",
                "validate_strategies",
                "verify_system_readiness",
            ],
            WeekendSubPhase.COMPLETE: [],
        }

        phase_tasks = tasks.get(sub_phase, [])
        all_succeeded = True

        for task in phase_tasks:
            # Special handling for research - check if process is still running
            if task == "run_weekend_research":
                if self._is_research_running():
                    logger.debug("Weekend research still running")
                    continue
                elif self.state.weekend_research_progress.get('status') == 'completed':
                    logger.debug("Weekend research already completed successfully")
                    continue
                else:
                    # Research not running and not completed - (re)start it
                    logger.info("Starting/restarting weekend research")
                    result = self.run_task(task)
                    if not result:
                        all_succeeded = False
                        logger.warning(f"Weekend task {task} failed to start")
                    continue

            # Standard task handling - skip if already completed
            if task in self.state.weekend_tasks_completed:
                continue

            result = self.run_task(task)
            if result:
                self.state.weekend_tasks_completed.append(task)
            else:
                all_succeeded = False
                logger.warning(f"Weekend task {task} failed")

        return all_succeeded

    def _is_research_running(self) -> bool:
        """Check if the weekend research process is still running."""
        # First check by saved PID
        pid = self.state.weekend_research_progress.get('pid')
        if pid:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                pass
            except PermissionError:
                return True  # Process exists but we don't have permission

        # Also check by process name (in case orchestrator restarted)
        import subprocess
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'run_nightly_research.py'],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                # Found a running research process
                pids = result.stdout.strip().split('\n')
                if pids:
                    # Update our state with the found PID
                    self.state.weekend_research_progress['pid'] = int(pids[0])
                    self.state.weekend_research_progress['status'] = 'running'
                    return True
        except Exception:
            pass

        return False

    def _task_generate_weekly_report(self) -> bool:
        """Generate weekly performance report."""
        try:
            logger.info("Generating weekly performance report...")

            from data.storage.db_manager import get_db
            db = get_db()

            # Get week's trades
            week_ago = (datetime.now(self.tz) - timedelta(days=7)).strftime('%Y-%m-%d')
            trades = db.fetchall(
                "trades",
                """
                SELECT * FROM trades
                WHERE created_at >= ?
                ORDER BY created_at DESC
                """,
                (week_ago,)
            )

            # Get strategy daily performance
            strategy_perf = db.fetchall(
                "performance",
                """
                SELECT strategy, SUM(net_pnl) as total_pnl,
                       COUNT(*) as trading_days,
                       AVG(daily_pnl_pct) as avg_daily_pct
                FROM strategy_daily
                WHERE date >= ?
                GROUP BY strategy
                """,
                (week_ago,)
            )

            # Generate report
            report_lines = [
                "=" * 60,
                "WEEKLY PERFORMANCE REPORT",
                f"Week ending: {datetime.now(self.tz).strftime('%Y-%m-%d')}",
                "=" * 60,
                "",
                f"Total trades this week: {len(trades) if trades else 0}",
                "",
                "Strategy Performance:",
            ]

            if strategy_perf:
                for perf in strategy_perf:
                    report_lines.append(
                        f"  {perf['strategy']}: ${perf['total_pnl']:.2f} "
                        f"({perf['avg_daily_pct']*100:.2f}% avg daily)"
                    )
            else:
                report_lines.append("  No strategy performance data available")

            report_lines.extend(["", "=" * 60])

            # Save report
            report_path = Path(__file__).parent / "logs" / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))

            logger.info(f"Weekly report saved to {report_path}")

            # Update state for dashboard
            self.state.weekend_research_progress['weekly_report'] = {
                'generated_at': datetime.now(self.tz).isoformat(),
                'trades_count': len(trades) if trades else 0,
                'path': str(report_path),
            }

            return True

        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")
            return False

    def _task_vacuum_databases(self) -> bool:
        """Run VACUUM on all SQLite databases for optimization."""
        try:
            from config import DATABASES
            import sqlite3

            logger.info("Vacuuming databases...")

            for name, path in DATABASES.items():
                if path.exists():
                    try:
                        conn = sqlite3.connect(str(path))
                        conn.execute("VACUUM")
                        conn.close()
                        logger.info(f"  Vacuumed {name}")
                    except Exception as e:
                        logger.warning(f"  Failed to vacuum {name}: {e}")

            return True

        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            return False

    def _preflight_system_check(self, min_memory_mb: int = 1500, max_load: float = 3.0) -> bool:
        """
        Perform pre-flight system check before launching heavy tasks.

        Checks memory availability and system load, waits if needed.

        Args:
            min_memory_mb: Minimum available memory required (default 1.5GB)
            max_load: Maximum 1-minute load average allowed (default 3.0)

        Returns:
            True if system is ready, False if still not ready after waiting
        """
        import os

        logger.info("Running pre-flight system check...")

        # Check system load first - wait for it to settle
        for attempt in range(6):  # Up to 3 minutes waiting for load
            load_1min = os.getloadavg()[0]
            if load_1min <= max_load:
                break
            logger.info(f"Waiting for load to settle ({load_1min:.1f} > {max_load})")
            time.sleep(30)
        else:
            logger.warning(f"Load still high ({load_1min:.1f}) after 3 min, proceeding anyway")

        # Check memory availability
        for attempt in range(10):  # Up to 5 minutes waiting for memory
            mem_status = self._check_memory_status()
            if mem_status['available_mb'] >= min_memory_mb:
                break
            logger.info(f"Waiting for memory ({mem_status['available_mb']:.0f}MB available, need {min_memory_mb}MB)...")
            gc.collect()  # Force garbage collection
            time.sleep(30)
        else:
            logger.warning(f"Memory still low ({mem_status['available_mb']:.0f}MB) after 5 min, proceeding anyway with swap")

        # Log final status
        mem_status = self._check_memory_status()
        load_1min = os.getloadavg()[0]
        logger.info(f"Pre-flight check complete: Memory={mem_status['available_mb']:.0f}MB, Load={load_1min:.1f}")

        return True  # Always proceed - swap will catch overflow

    def _task_run_weekend_research(self) -> bool:
        """Run extended weekend research with configurable parameters."""
        try:
            logger.info("Starting weekend research...")

            # Pre-flight system check - wait for memory and load to be reasonable
            self._preflight_system_check(min_memory_mb=1500, max_load=3.0)

            # Get config (from dashboard or defaults)
            config = self.state.weekend_config.get('research', WEEKEND_CONFIG['research'])
            generations = config.get('generations_default', 10)
            population = config.get('population_default', 30)
            discovery_enabled = config.get('discovery_enabled', True)
            adaptive_enabled = config.get('adaptive_ga_enabled', True)
            discovery_hours = config.get('discovery_hours', 4.0)
            strategies = config.get('strategies', [])  # Empty = all strategies

            # Build command
            research_script = Path(__file__).parent / "run_nightly_research.py"
            cmd = [sys.executable, str(research_script), "-g", str(generations)]

            # Add strategies (empty list means all evolvable strategies)
            if strategies:
                cmd.extend(["--strategies"] + strategies)
            else:
                cmd.extend(["--strategies", "all"])

            if discovery_enabled:
                cmd.extend(["--discovery", "--discovery-hours", str(discovery_hours)])
            if adaptive_enabled:
                cmd.append("--adaptive")

            logger.info(f"Launching weekend research: {' '.join(cmd)}")

            # Create log file for research output
            log_file = Path(__file__).parent / "logs" / "weekend_research.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Run in subprocess with output to log file
            import subprocess
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).parent),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )

            # Update progress state for dashboard
            self.state.weekend_research_progress['status'] = 'running'
            self.state.weekend_research_progress['started_at'] = datetime.now(self.tz).isoformat()
            self.state.weekend_research_progress['config'] = {
                'generations': generations,
                'population': population,
                'discovery': discovery_enabled,
                'adaptive': adaptive_enabled,
                'strategies': strategies,
            }
            self.state.weekend_research_progress['log_file'] = str(log_file)

            # Don't wait - let it run in background
            # Weekend schedule will check on it periodically
            self.state.weekend_research_progress['pid'] = process.pid
            logger.info(f"Weekend research started with PID {process.pid}")

            return True

        except Exception as e:
            logger.error(f"Failed to start weekend research: {e}")
            self.state.weekend_research_progress['status'] = 'failed'
            self.state.weekend_research_progress['error'] = str(e)
            return False

    def _task_refresh_index_constituents(self) -> bool:
        """Refresh index constituent lists (S&P 500, NASDAQ-100, etc.)."""
        try:
            logger.info("Refreshing index constituents...")

            script = Path(__file__).parent / "scripts" / "fetch_index_constituents.py"
            if not script.exists():
                logger.warning("fetch_index_constituents.py not found")
                return True  # Not a critical failure

            import subprocess
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                logger.info("Index constituents refreshed successfully")
                return True
            else:
                logger.warning(f"Index constituent refresh had issues: {result.stderr}")
                return True  # Partial success is OK

        except subprocess.TimeoutExpired:
            logger.error("Index constituent refresh timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to refresh index constituents: {e}")
            return False

    def _task_refresh_fundamentals(self) -> bool:
        """Refresh fundamental data for universe."""
        try:
            logger.info("Refreshing fundamentals data...")

            # Check scope from weekend config
            scope = self.state.weekend_config.get('data', {}).get('universe_scope', 'watchlist')

            script = Path(__file__).parent / "scripts" / "download_fundamentals.py"
            if not script.exists():
                logger.warning("download_fundamentals.py not found")
                return True

            import subprocess
            cmd = [sys.executable, str(script)]
            if scope == 'full':
                cmd.append("--all")

            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours max
            )

            if result.returncode == 0:
                logger.info("Fundamentals refresh completed")
                return True
            else:
                logger.warning(f"Fundamentals refresh had issues: {result.stderr[:500]}")
                return True

        except subprocess.TimeoutExpired:
            logger.error("Fundamentals refresh timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to refresh fundamentals: {e}")
            return False

    def _task_refresh_pairs_correlations(self) -> bool:
        """Recalculate pairs trading cointegration relationships."""
        try:
            logger.info("Refreshing pairs correlations...")

            script = Path(__file__).parent / "scripts" / "discover_pairs.py"
            if not script.exists():
                logger.warning("discover_pairs.py not found")
                return True

            import subprocess
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=1800  # 30 min max
            )

            if result.returncode == 0:
                logger.info("Pairs correlations refreshed")
                return True
            else:
                logger.warning(f"Pairs refresh had issues: {result.stderr[:500]}")
                return True

        except subprocess.TimeoutExpired:
            logger.error("Pairs refresh timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to refresh pairs: {e}")
            return False

    def _task_validate_strategies(self) -> bool:
        """Validate all enabled strategies can be loaded and run."""
        try:
            from config import get_enabled_strategies, STRATEGIES

            logger.info("Validating strategies...")
            issues = []

            for name in get_enabled_strategies():
                try:
                    # Try to import the strategy
                    if name == "vol_managed_momentum":
                        from strategies.vol_managed_momentum import VolManagedMomentumStrategy
                    elif name == "mean_reversion":
                        from strategies.mean_reversion import MeanReversionStrategy
                    elif name == "gap_fill":
                        from strategies.gap_fill import GapFillStrategy
                    elif name == "pairs_trading":
                        from strategies.pairs_trading import PairsTradingStrategy
                    elif name == "relative_volume_breakout":
                        from strategies.relative_volume_breakout import RelativeVolumeBreakout
                    elif name == "vix_regime_rotation":
                        from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
                    elif name == "quality_smallcap_value":
                        from strategies.quality_small_cap_value import QualitySmallCapValueStrategy
                    elif name == "factor_momentum":
                        from strategies.factor_momentum import FactorMomentumStrategy

                    logger.debug(f"  {name}: OK")

                except ImportError as e:
                    issues.append(f"{name}: Import error - {e}")
                except Exception as e:
                    issues.append(f"{name}: {e}")

            if issues:
                for issue in issues:
                    logger.warning(f"  Strategy issue: {issue}")
                return len(issues) < len(get_enabled_strategies())  # Partial success

            logger.info(f"All {len(get_enabled_strategies())} strategies validated")
            return True

        except Exception as e:
            logger.error(f"Strategy validation failed: {e}")
            return False

    def _check_memory_status(self) -> dict:
        """
        Check system memory status and return classification.

        Returns:
            dict with keys: available_mb, percent_used, level (ok/warning/critical)

        Thresholds (Pi 5 with 4GB):
        - OK: > 1024 MB available
        - Warning: 512-1024 MB available
        - Critical: < 512 MB available
        """
        try:
            # Read from /proc/meminfo (works on all Linux)
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1])  # Value in kB
                        meminfo[key] = value

            total_kb = meminfo.get('MemTotal', 0)
            available_kb = meminfo.get('MemAvailable', 0)

            total_mb = total_kb / 1024
            available_mb = available_kb / 1024
            used_mb = total_mb - available_mb
            percent_used = (used_mb / total_mb * 100) if total_mb > 0 else 0

            # Classify memory status
            if available_mb < 512:
                level = 'critical'
            elif available_mb < 1024:
                level = 'warning'
            else:
                level = 'ok'

            return {
                'available_mb': available_mb,
                'used_mb': used_mb,
                'total_mb': total_mb,
                'percent_used': percent_used,
                'level': level,
            }

        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return {
                'available_mb': 0,
                'used_mb': 0,
                'total_mb': 0,
                'percent_used': 0,
                'level': 'unknown',
            }

    def _log_memory_alert(self, mem_status: dict):
        """Log memory alert to performance database for dashboard visibility."""
        try:
            import sqlite3
            from config import DATABASES

            conn = sqlite3.connect(DATABASES['performance'])
            cursor = conn.cursor()

            # Ensure memory_alerts table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    available_mb REAL,
                    used_mb REAL,
                    total_mb REAL,
                    percent_used REAL,
                    level TEXT,
                    action_taken TEXT
                )
            """)

            cursor.execute("""
                INSERT INTO memory_alerts
                (timestamp, available_mb, used_mb, total_mb, percent_used, level, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                mem_status.get('available_mb', 0),
                mem_status.get('used_mb', 0),
                mem_status.get('total_mb', 0),
                mem_status.get('percent_used', 0),
                mem_status.get('level', 'unknown'),
                'paused_60s' if mem_status.get('level') == 'critical' else 'warning_logged'
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to log memory alert: {e}")

    def _task_verify_system_readiness(self) -> bool:
        """Verify system is ready for Monday trading."""
        try:
            logger.info("Verifying system readiness...")
            issues = []

            # Check broker connection
            try:
                acct = self.broker.get_account()
                logger.info(f"  Broker: Connected (${acct.portfolio_value:,.2f})")
            except Exception as e:
                issues.append(f"Broker connection failed: {e}")

            # Check data freshness (weekend/holiday aware)
            from config import DIRS
            daily_dir = DIRS.get('daily_yahoo')
            if daily_dir and daily_dir.exists():
                files = list(daily_dir.glob('*.parquet'))
                if files:
                    newest = max(f.stat().st_mtime for f in files)
                    age_hours = (time.time() - newest) / 3600
                    # Allow more staleness on weekends/holidays (5 days vs 3)
                    now = datetime.now(self.tz)
                    is_weekend = now.weekday() >= 5  # Sat=5, Sun=6
                    max_age = 120 if is_weekend else 72  # 5 days vs 3 days
                    if age_hours > max_age:
                        issues.append(f"Daily data is {age_hours:.0f} hours old")
                    else:
                        logger.info(f"  Data: Fresh ({age_hours:.1f} hours old, max {max_age}h for {'weekend' if is_weekend else 'weekday'})")
            else:
                issues.append("Daily data directory not found")

            # Check disk space
            import shutil
            disk = shutil.disk_usage(str(Path(__file__).parent))
            free_gb = disk.free / (1024**3)
            if free_gb < 1:
                issues.append(f"Low disk space: {free_gb:.1f} GB free")
            else:
                logger.info(f"  Disk: {free_gb:.1f} GB free")

            # Check memory usage
            mem_status = self._check_memory_status()
            if mem_status['level'] == 'critical':
                issues.append(f"Critical memory: {mem_status['available_mb']:.0f}MB available")
            elif mem_status['level'] == 'warning':
                logger.warning(f"  Memory: {mem_status['available_mb']:.0f}MB available (warning)")
            else:
                logger.info(f"  Memory: {mem_status['available_mb']:.0f}MB available ({mem_status['percent_used']:.0f}% used)")

            # Check databases
            from config import DATABASES
            for name, path in DATABASES.items():
                if not path.exists():
                    issues.append(f"Database missing: {name}")

            # Update state
            self.state.weekend_research_progress['readiness'] = {
                'checked_at': datetime.now(self.tz).isoformat(),
                'issues': issues,
                'ready': len(issues) == 0,
            }

            if issues:
                for issue in issues:
                    logger.warning(f"  Issue: {issue}")
                return False

            logger.info("System ready for Monday trading")
            return True

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return False

    # =========================================================================
    # Task Execution
    # =========================================================================

    def run_task(self, task_name: str) -> bool:
        """
        Run a specific task with error handling.

        Args:
            task_name: Name of the task to run

        Returns:
            True if task succeeded, False otherwise
        """
        if task_name not in self._task_registry:
            logger.error(f"Unknown task: {task_name}")
            return False

        task_func = self._task_registry[task_name]
        start_time = time.time()

        try:
            logger.info(f"Starting task: {task_name}")
            result = task_func()
            elapsed = time.time() - start_time

            if result:
                logger.info(f"Task {task_name} completed in {elapsed:.1f}s")
                # Only add to daily list if not already present (avoid duplicates in report)
                if task_name not in self.state.tasks_completed_today:
                    self.state.tasks_completed_today.append(task_name)
                self.state.last_task_run[task_name] = datetime.now(self.tz)
            else:
                logger.warning(f"Task {task_name} returned False after {elapsed:.1f}s")

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Task {task_name} failed after {elapsed:.1f}s: {e}")
            logger.debug(traceback.format_exc())

            self.state.errors_today.append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now(self.tz).isoformat()
            })

            return False

    # Tasks that should only run ONCE per phase (initialization tasks)
    # All other tasks run repeatedly based on check_interval_seconds
    ONCE_PER_PHASE_TASKS = {
        "run_scheduler",           # Start scheduler thread once
        "start_intraday_stream",   # Start stream once at market open
        "stop_intraday_stream",    # Stop stream once
        "detect_gaps",             # Detect gaps once at open
        "run_nightly_research",    # Run research once per night
        "train_ml_regime_model",   # Train once per night
        "load_live_strategies",    # Load strategies once at pre-market
        "refresh_data",            # Refresh data once per phase
        "refresh_premarket_data",  # Refresh premarket data once
        "refresh_eod_data",        # Refresh EOD data once
        "run_weekend_schedule",    # Weekend tasks dispatched once
    }

    def run_phase_tasks(self, phase: MarketPhase) -> Dict[str, bool]:
        """
        Run all tasks for a specific phase.

        Args:
            phase: The market phase to run tasks for

        Returns:
            Dict mapping task names to success status
        """
        config = self.get_phase_config(phase)
        results = {}
        phase_key = phase.value

        # Initialize phase completion tracking if needed
        if phase_key not in self.state.phase_tasks_completed:
            self.state.phase_tasks_completed[phase_key] = set()

        logger.info(f"Running tasks for phase: {phase.value}")

        for task_name in config.tasks:
            # Check if this is a once-per-phase task that already ran
            is_once_per_phase = task_name in self.ONCE_PER_PHASE_TASKS
            if is_once_per_phase and task_name in self.state.phase_tasks_completed[phase_key]:
                logger.debug(f"Skipping {task_name}, already completed this phase (once-per-phase)")
                results[task_name] = True
                continue

            # Check if task was already run recently (within check interval)
            # This applies to recurring tasks like monitor_positions
            last_run = self.state.last_task_run.get(task_name)
            if last_run:
                seconds_since = (datetime.now(self.tz) - last_run).total_seconds()
                if seconds_since < config.check_interval_seconds:
                    logger.debug(f"Skipping {task_name}, ran {seconds_since:.0f}s ago")
                    results[task_name] = True
                    continue

            result = self.run_task(task_name)
            results[task_name] = result

            # Only mark once-per-phase tasks as completed
            if result and is_once_per_phase:
                self.state.phase_tasks_completed[phase_key].add(task_name)

        return results

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self, once: bool = False, force_phase: Optional[str] = None):
        """
        Main orchestrator loop.

        Args:
            once: If True, run current phase once and exit
            force_phase: Force run a specific phase (pre, market, post, evening, overnight)
        """
        logger.info("=" * 60)
        logger.info("DAILY ORCHESTRATOR STARTING")
        logger.info(f"Paper mode: {self.paper_mode}")
        logger.info("=" * 60)

        self.state.is_running = True

        # Run hardware startup sequence
        if self._hardware:
            logger.info("Calling hardware startup sequence...")
            try:
                self._hardware.startup()
                logger.info("Hardware startup complete")
            except Exception as e:
                logger.error(f"Hardware startup failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Run startup recovery sequence (cleanup orphans, validate state)
        self._startup_recovery()

        try:
            while not self.shutdown_event.is_set():
                # Determine current phase
                if force_phase:
                    phase_map = {
                        'pre': MarketPhase.PRE_MARKET,
                        'intraday_open': MarketPhase.INTRADAY_OPEN,
                        'intraday_active': MarketPhase.INTRADAY_ACTIVE,
                        'market': MarketPhase.MARKET_OPEN,
                        'post': MarketPhase.POST_MARKET,
                        'evening': MarketPhase.EVENING,
                        'overnight': MarketPhase.OVERNIGHT,
                        'weekend': MarketPhase.WEEKEND,
                    }
                    current_phase = phase_map.get(force_phase, MarketPhase.EVENING)
                else:
                    current_phase = self.get_current_phase()

                # Check for phase transition
                if current_phase != self.state.current_phase:
                    previous_phase = self.state.current_phase
                    logger.info(f"Phase transition: {previous_phase.value} -> {current_phase.value}")

                    # Stop intraday stream when transitioning out of intraday phases
                    if previous_phase in (MarketPhase.INTRADAY_OPEN, MarketPhase.INTRADAY_ACTIVE):
                        if current_phase not in (MarketPhase.INTRADAY_OPEN, MarketPhase.INTRADAY_ACTIVE):
                            logger.info("Transitioning out of intraday phase - stopping intraday stream")
                            self._task_stop_intraday_stream()

                    self.state.current_phase = current_phase
                    self.state.phase_started_at = datetime.now(self.tz)

                    # Update hardware LEDs for new phase
                    if self._hardware:
                        self._hardware.set_phase(current_phase.value)

                    # Reset daily stats at start of pre-market
                    if current_phase == MarketPhase.PRE_MARKET:
                        self.state.tasks_completed_today = []
                        self.state.errors_today = []
                        self.state.daily_stats = {}
                        self.state.phase_tasks_completed = {}  # Reset per-phase tracking

                        # Reset weekend state for next weekend
                        if previous_phase == MarketPhase.WEEKEND:
                            self.state.weekend_sub_phase = None
                            self.state.weekend_tasks_completed = []
                            self.state.weekend_started_at = None
                            self.state.weekend_research_progress = {}
                            logger.info("Weekend phase completed - reset for next week")

                # Handle weekend - run weekend schedule instead of sleeping
                if current_phase == MarketPhase.WEEKEND:
                    sub_phase = self.state.weekend_sub_phase or WeekendSubPhase.FRIDAY_CLEANUP

                    # Count weekend tasks for dashboard display
                    completed = len(self.state.weekend_tasks_completed)
                    research_running = self._is_research_running()
                    # Total tasks: cleanup(3) + research(1) + refresh(3) + prep(3) = 10
                    total = 10
                    if research_running:
                        completed = max(completed, 4)  # Show progress during research
                    logger.info(f"Phase weekend: {completed}/{total} tasks succeeded")
                    logger.info(f"Weekend phase active - sub-phase: {sub_phase.value}")

                    if once:
                        # Run one iteration of weekend tasks
                        self._task_run_weekend_schedule()
                        break

                    # Run weekend schedule (handles sub-phases internally)
                    self._task_run_weekend_schedule()

                    # Update hardware display after weekend tasks
                    if self._hardware:
                        self._update_hardware_display(current_phase)

                    # If weekend is complete, sleep until Monday
                    if self.state.weekend_sub_phase == WeekendSubPhase.COMPLETE:
                        time_to_next = self.time_until_next_phase()
                        logger.info(f"Weekend tasks complete - ready for Monday ({time_to_next})")
                        sleep_seconds = min(time_to_next.total_seconds(), 3600)
                        self._interruptible_sleep(sleep_seconds)
                    else:
                        # Sleep between weekend task checks (30 min)
                        self._interruptible_sleep(1800)
                    continue

                # Check memory before running tasks
                mem_status = self._check_memory_status()
                if mem_status['level'] == 'critical':
                    logger.warning(f"MEMORY CRITICAL: {mem_status['available_mb']:.0f}MB available - pausing for 60s")
                    # Log to performance DB for dashboard visibility
                    self._log_memory_alert(mem_status)
                    self._interruptible_sleep(60)
                    continue  # Re-check memory before proceeding
                elif mem_status['level'] == 'warning':
                    logger.warning(f"Memory warning: {mem_status['available_mb']:.0f}MB available ({mem_status['percent_used']:.0f}% used)")

                # Run phase tasks
                results = self.run_phase_tasks(current_phase)

                # Log results summary
                passed = sum(results.values())
                total = len(results)
                logger.info(f"Phase {current_phase.value}: {passed}/{total} tasks succeeded")

                # Update hardware display with current data
                if self._hardware:
                    self._update_hardware_display(current_phase)

                if once:
                    break

                # Sleep until next check
                config = self.get_phase_config(current_phase)
                self._interruptible_sleep(config.check_interval_seconds)

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            logger.error(traceback.format_exc())

        finally:
            self.state.is_running = False
            self._cleanup()
            logger.info("Orchestrator stopped")

    def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by shutdown event.

        During market hours, updates display every 5 seconds for live SPY/VIX tracking.
        """
        display_interval = 5.0  # Update display every 5 seconds during market hours
        elapsed = 0.0

        while elapsed < seconds and not self.shutdown_event.is_set():
            # Determine sleep chunk (5 seconds or remaining time)
            sleep_chunk = min(display_interval, seconds - elapsed)
            self.shutdown_event.wait(sleep_chunk)
            elapsed += sleep_chunk

            # Update display during market hours for live price tracking
            if self._hardware and not self.shutdown_event.is_set():
                current_phase = self.get_current_phase()
                if current_phase in (MarketPhase.MARKET_OPEN, MarketPhase.INTRADAY_OPEN,
                                     MarketPhase.INTRADAY_ACTIVE, MarketPhase.PRE_MARKET):
                    self._update_hardware_display(current_phase)

    def _update_hardware_display(self, current_phase: MarketPhase) -> None:
        """Update LCD display with current system data."""
        try:
            # Get account info if available
            portfolio_value = 0
            daily_pnl = 0
            daily_pnl_pct = 0
            position_count = 0
            cash_pct = 0
            position_list = []

            try:
                account = self.broker.get_account()
                if account:
                    portfolio_value = float(account.portfolio_value)
                    cash = float(account.cash) if account.cash else 0
                    cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0

                positions = self.broker.get_positions()
                position_count = len(positions) if positions else 0

                # Calculate daily P&L from positions' unrealized P&L
                if positions:
                    daily_pnl = sum(p.unrealized_pnl for p in positions)
                    if portfolio_value > 0:
                        daily_pnl_pct = (daily_pnl / portfolio_value) * 100

                    # Convert positions to list of dicts for display
                    for p in positions:
                        position_list.append({
                            'symbol': p.symbol,
                            'qty': int(p.qty),
                            'unrealized_pnl': p.unrealized_pnl
                        })
            except Exception:
                pass  # Use defaults if broker unavailable

            # Get market data
            spy_price = 0
            vix = 0
            try:
                spy_data = self.broker.get_latest_price('SPY')
                if spy_data is not None:
                    spy_price = float(spy_data)

                # Get VIX with caching (30-second TTL to reduce API calls)
                now = time.time()
                if now - self._vix_cache_time < self._vix_cache_ttl and self._vix_cache > 0:
                    vix = self._vix_cache
                else:
                    # Fetch fresh VIX from yfinance with timeout (prevents hanging)
                    try:
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

                        def fetch_vix_from_yfinance():
                            import yfinance as yf
                            vix_ticker = yf.Ticker('^VIX')
                            # Try intraday first (1-minute data)
                            vix_hist = vix_ticker.history(period='1d', interval='1m')
                            if not vix_hist.empty:
                                return float(vix_hist['Close'].iloc[-1])
                            # Fallback to daily
                            vix_hist = vix_ticker.history(period='5d')
                            if not vix_hist.empty:
                                return float(vix_hist['Close'].iloc[-1])
                            return 0

                        # Use thread pool with 10 second timeout
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(fetch_vix_from_yfinance)
                            vix = future.result(timeout=10.0)

                        # Update cache
                        if vix > 0:
                            self._vix_cache = vix
                            self._vix_cache_time = now
                    except FuturesTimeout:
                        logger.debug("VIX fetch timed out after 10s, using cached value")
                        vix = self._vix_cache if self._vix_cache > 0 else 0
                    except Exception:
                        # Fallback to cached data manager
                        vix_data = self.data_manager.get_vix()
                        if vix_data is not None:
                            vix = float(vix_data)
            except Exception:
                pass

            # Get system metrics
            memory_pct = psutil.virtual_memory().percent
            cpu_pct = psutil.cpu_percent(interval=None)
            load_avg = os.getloadavg()[0]  # 1-minute load average

            # Get CPU temperature (Pi-specific)
            cpu_temp = 0
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = int(f.read().strip()) / 1000.0
            except Exception:
                pass

            # Get ZRAM usage
            zram_pct = 0
            try:
                swap = psutil.swap_memory()
                if swap.total > 0:
                    zram_pct = swap.percent
            except Exception:
                pass

            # Calculate uptime
            uptime_str = "--"
            if self.state.phase_started_at:
                uptime_secs = (datetime.now(self.tz) - self.state.phase_started_at).total_seconds()
                if uptime_secs >= 86400:  # Days
                    uptime_str = f"{int(uptime_secs // 86400)}d {int((uptime_secs % 86400) // 3600)}h"
                elif uptime_secs >= 3600:  # Hours
                    uptime_str = f"{int(uptime_secs // 3600)}h {int((uptime_secs % 3600) // 60)}m"
                else:
                    uptime_str = f"{int(uptime_secs // 60)}m"

            # Calculate time to next phase
            time_remaining = self.time_until_next_phase()
            total_secs = int(time_remaining.total_seconds())
            if total_secs >= 3600:
                phase_time_str = f"{total_secs // 3600}h {(total_secs % 3600) // 60}m"
            else:
                phase_time_str = f"{total_secs // 60}m"

            # Build display data
            display_data = {
                'phase': current_phase.value,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'position_count': position_count,
                'positions': position_list,
                'cash_pct': cash_pct,
                'spy_price': spy_price,
                'vix': vix,
                'memory_pct': memory_pct,
                'zram_pct': zram_pct,
                'cpu_pct': cpu_pct,
                'cpu_temp': cpu_temp,
                'load_avg': load_avg,
                'uptime': uptime_str,
                'phase_time_remaining': phase_time_str,
            }

            # Add research data during overnight and weekend phases
            if current_phase in (MarketPhase.OVERNIGHT, MarketPhase.WEEKEND):
                research_data = self._get_research_progress()
                display_data.update(research_data)

            self._hardware.update_display(display_data)

        except Exception as e:
            logger.debug(f"Display update failed: {e}")

    def _get_research_progress(self) -> Dict[str, Any]:
        """Get current research progress from database for LCD display."""
        result = {
            'research_status': 'IDLE',
            'research_generation': 0,
            'research_max_gen': 100,
            'research_best_sharpe': 0,
            'research_eta': 0,
        }

        try:
            import sqlite3
            db_path = Path(__file__).parent / "db" / "research.db"
            if not db_path.exists():
                return result

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Get the current running or most recent run
            cursor = conn.execute("""
                SELECT run_id, status, planned_generations, total_generations, start_time
                FROM ga_runs
                WHERE status = 'running'
                ORDER BY start_time DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

            if not row:
                # No running research - check for recent completed
                cursor = conn.execute("""
                    SELECT run_id, status, planned_generations, total_generations
                    FROM ga_runs
                    ORDER BY start_time DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row and row['status'] == 'completed':
                    result['research_status'] = 'DONE'
                    result['research_generation'] = row['total_generations'] or 0
                    result['research_max_gen'] = row['planned_generations'] or row['total_generations'] or 0

                    # Get best fitness from ga_history for today's runs
                    from datetime import date
                    today = date.today().isoformat()
                    cursor = conn.execute("""
                        SELECT MAX(best_fitness) as best_fitness
                        FROM ga_history
                        WHERE run_date = ?
                    """, (today,))
                    fitness_row = cursor.fetchone()
                    if fitness_row and fitness_row['best_fitness']:
                        result['research_best_sharpe'] = round(fitness_row['best_fitness'], 2)
                conn.close()
                return result

            # Get progress from ga_history (where generations are actually logged)
            run_id = row['run_id']
            planned_gens = row['planned_generations'] or 3

            # Get today's date for filtering
            from datetime import date
            today = date.today().isoformat()

            # Count total generations completed today and get best fitness
            cursor = conn.execute("""
                SELECT COUNT(*) as gen_count,
                       MAX(best_fitness) as best_fitness
                FROM ga_history
                WHERE run_date = ?
            """, (today,))
            progress = cursor.fetchone()

            if progress:
                completed = progress['gen_count'] or 0
                best_fitness = progress['best_fitness'] or 0

                result['research_generation'] = completed
                # Show progress relative to completed (no fixed max during active research)
                result['research_max_gen'] = max(completed, 21)  # At least 21 (7 strategies * 3 gens)
                result['research_best_sharpe'] = round(best_fitness, 2)
                result['research_status'] = 'EVOLVING'

            conn.close()

        except Exception as e:
            logger.debug(f"Failed to get research progress: {e}")

        return result

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    def _cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up resources...")

        # Stop intraday stream if running
        if self._stream_handler is not None or self._intraday_strategies:
            logger.info("Stopping intraday components...")
            self._task_stop_intraday_stream()

        # Stop scheduler thread if running
        if self.state.scheduler_thread and self.state.scheduler_thread.is_alive():
            logger.info("Stopping scheduler thread...")
            self.shutdown_event.set()
            self.state.scheduler_thread.join(timeout=30)  # Increased from 5s
            if self.state.scheduler_thread.is_alive():
                logger.warning("Scheduler thread did not stop cleanly within timeout")

        # Cancel any open orders to prevent unexpected fills after restart
        if self.broker:
            try:
                open_orders = self.broker.get_open_orders()
                if open_orders:
                    logger.info(f"Cancelling {len(open_orders)} open orders on shutdown...")
                    cancelled = 0
                    for order in open_orders:
                        try:
                            self.broker.cancel_order(order.id)
                            cancelled += 1
                        except Exception as e:
                            logger.warning(f"Failed to cancel order {order.id}: {e}")
                    logger.info(f"Cancelled {cancelled}/{len(open_orders)} open orders")
            except Exception as e:
                logger.warning(f"Failed to check/cancel open orders on shutdown: {e}")

        # Generate final report if we have data
        if self.state.daily_stats:
            self._task_generate_daily_report()

        # Shutdown hardware
        if self._hardware:
            logger.info("Shutting down hardware...")
            self._hardware.shutdown()

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        now = datetime.now(self.tz)
        current_phase = self.get_current_phase()
        time_to_next = self.time_until_next_phase()

        return {
            'timestamp': now.isoformat(),
            'is_running': self.state.is_running,
            'paper_mode': self.paper_mode,
            'current_phase': current_phase.value,
            'phase_started_at': self.state.phase_started_at.isoformat() if self.state.phase_started_at else None,
            'time_to_next_phase': str(time_to_next),
            'tasks_completed_today': len(self.state.tasks_completed_today),
            'errors_today': len(self.state.errors_today),
            'last_task_runs': {
                k: v.isoformat() for k, v in self.state.last_task_run.items()
            },
            'scheduler_running': (
                self.state.scheduler_thread.is_alive()
                if self.state.scheduler_thread else False
            ),
            'market_open': MarketHours.is_market_open(),
            # Intraday status
            'intraday_stream_active': self._stream_handler is not None,
            'intraday_strategies_count': len(self._intraday_strategies),
            'intraday_positions_count': len(self._intraday_positions),
        }

    def print_status(self):
        """Print formatted status to console."""
        status = self.status()

        print("\n" + "=" * 60)
        print("DAILY ORCHESTRATOR STATUS")
        print("=" * 60)
        print(f"Time: {status['timestamp']}")
        print(f"Running: {status['is_running']}")
        print(f"Paper Mode: {status['paper_mode']}")
        print(f"Market Open: {status['market_open']}")
        print()
        print(f"Current Phase: {status['current_phase']}")
        print(f"Phase Started: {status['phase_started_at']}")
        print(f"Time to Next Phase: {status['time_to_next_phase']}")
        print()
        print(f"Tasks Completed Today: {status['tasks_completed_today']}")
        print(f"Errors Today: {status['errors_today']}")
        print(f"Scheduler Running: {status['scheduler_running']}")
        print()

        if status['last_task_runs']:
            print("Last Task Runs:")
            for task, time_str in status['last_task_runs'].items():
                print(f"  {task}: {time_str}")

        # Intraday status
        print()
        print("Intraday Status:")
        print(f"  Stream Active: {status['intraday_stream_active']}")
        print(f"  Strategies: {status['intraday_strategies_count']}")
        print(f"  Positions: {status['intraday_positions_count']}")

        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Daily Orchestrator")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--once", action="store_true", help="Run current phase once and exit")
    parser.add_argument("--phase", type=str,
                        choices=['pre', 'intraday_open', 'intraday_active', 'market', 'post', 'evening', 'overnight', 'weekend'],
                        help="Force run a specific phase")
    parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")

    args = parser.parse_args()

    orchestrator = DailyOrchestrator(paper_mode=not args.live)

    if args.status:
        orchestrator.print_status()
        return

    orchestrator.run(once=args.once, force_phase=args.phase)


if __name__ == "__main__":
    main()
