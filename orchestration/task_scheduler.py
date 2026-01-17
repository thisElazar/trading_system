"""
TaskScheduler - Time-aware task scheduling with priorities and dependencies.

This module implements Option C from the orchestrator refactor plan:
- Priority-based task selection (CRITICAL -> BACKGROUND)
- Dependency resolution between tasks
- Resource-aware execution (memory, CPU tracking)
- Dynamic time window calculation (holidays, early close)
- Opportunistic task backfilling

Usage:
    from orchestration.task_scheduler import TaskScheduler

    scheduler = TaskScheduler(orchestrator)
    ready_tasks = scheduler.get_ready_tasks()
    result = scheduler.execute_next()
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Set, Optional, Callable, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta, date
import logging
import os
import psutil

if TYPE_CHECKING:
    from daily_orchestrator import TradingOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TaskPriority(Enum):
    """
    Task priority levels for scheduling decisions.

    Lower value = higher priority (CRITICAL executes first).
    """
    CRITICAL = 1    # Must complete: trading operations, alerts
    HIGH = 2        # Should complete: EOD reconciliation, P&L
    NORMAL = 3      # Standard tasks: data refresh, core research
    LOW = 4         # Opportunistic: extended research, validation
    BACKGROUND = 5  # Fill gaps: cleanup, maintenance


class TaskCategory(Enum):
    """Task categories for resource grouping and conflict detection."""
    TRADING = auto()      # Broker interactions, position management
    DATA = auto()         # Data fetching, refresh operations
    RESEARCH = auto()     # GA/GP optimization, model training
    MAINTENANCE = auto()  # Cleanup, backups, database operations
    MONITORING = auto()   # Health checks, alerts, status updates


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class TaskSpec:
    """
    Specification for a schedulable task.

    This defines everything the scheduler needs to know about a task:
    when it can run, what resources it needs, and what it depends on.
    """
    name: str
    handler: str  # Method name in orchestrator (e.g., "_task_refresh_data")
    priority: TaskPriority
    category: TaskCategory

    # Time estimates (minutes)
    estimated_minutes: float = 5.0
    max_runtime_minutes: float = 30.0
    can_interrupt: bool = False  # Can be gracefully stopped mid-execution

    # Resource requirements
    memory_mb: int = 200
    cpu_intensive: bool = False

    # Phase constraints - empty set means all phases
    phases: Set[str] = field(default_factory=set)

    # Dependencies and conflicts
    dependencies: Set[str] = field(default_factory=set)  # Must complete first
    conflicts: Set[str] = field(default_factory=set)     # Cannot run simultaneously

    # Execution rules
    run_once_per_phase: bool = False
    run_once_per_day: bool = False
    min_interval_minutes: float = 0  # Minimum time between runs

    # Market awareness
    requires_market_open: bool = False
    requires_market_closed: bool = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, TaskSpec):
            return self.name == other.name
        return False


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_name: str
    success: bool
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60


@dataclass
class TimeWindow:
    """
    Represents available time for scheduling decisions.

    Captures all context needed to make scheduling decisions:
    current time, phase, market status, and special conditions.
    """
    start: datetime
    end: datetime
    phase: str
    is_market_day: bool
    is_holiday: bool
    is_early_close: bool

    @property
    def duration_minutes(self) -> float:
        """Minutes remaining in this window."""
        return max(0, (self.end - self.start).total_seconds() / 60)

    @property
    def duration_hours(self) -> float:
        """Hours remaining in this window."""
        return self.duration_minutes / 60

    @property
    def is_extended(self) -> bool:
        """True if this is an extended window (holiday, weekend)."""
        return self.is_holiday or not self.is_market_day


@dataclass
class SchedulerState:
    """
    Persistent state for task scheduler.

    Tracks execution history, performance metrics, and current status.
    """
    # Execution tracking
    last_run: Dict[str, datetime] = field(default_factory=dict)
    run_counts_today: Dict[str, int] = field(default_factory=dict)
    phase_completions: Dict[str, Set[str]] = field(default_factory=dict)

    # Performance metrics (for duration estimation)
    avg_durations: Dict[str, float] = field(default_factory=dict)  # minutes
    success_rates: Dict[str, float] = field(default_factory=dict)

    # Current execution state
    running_tasks: Set[str] = field(default_factory=set)
    queued_tasks: List[str] = field(default_factory=list)

    # Resource tracking
    estimated_memory_used: int = 0  # MB currently allocated to running tasks

    def update_duration(self, task_name: str, duration_minutes: float):
        """Update rolling average duration for a task using EMA."""
        alpha = 0.3  # Weight for new observation
        if task_name in self.avg_durations:
            self.avg_durations[task_name] = (
                (1 - alpha) * self.avg_durations[task_name] +
                alpha * duration_minutes
            )
        else:
            self.avg_durations[task_name] = duration_minutes

    def update_success_rate(self, task_name: str, success: bool):
        """Update rolling success rate for a task using EMA."""
        alpha = 0.2
        value = 1.0 if success else 0.0
        if task_name in self.success_rates:
            self.success_rates[task_name] = (
                (1 - alpha) * self.success_rates[task_name] +
                alpha * value
            )
        else:
            self.success_rates[task_name] = value


# =============================================================================
# TASK SCHEDULER
# =============================================================================

class TaskScheduler:
    """
    Time-aware task scheduler with priority, dependencies, and resource management.

    Key Features:
    - Dynamic time window calculation (holidays, early close)
    - Priority-based task selection
    - Dependency resolution
    - Resource-aware execution (memory, CPU)
    - Opportunistic task backfilling

    Usage:
        scheduler = TaskScheduler(orchestrator)

        # Get tasks ready to run
        ready = scheduler.get_ready_tasks()

        # Execute highest priority task
        result = scheduler.execute_next()

        # Plan entire phase
        plan = scheduler.plan_phase()
    """

    # US Market holidays (2025-2027) - fallback when API unavailable
    HOLIDAYS = {
        # 2025
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
        # 2026
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
        # 2027
        date(2027, 1, 1),   # New Year's Day
        date(2027, 1, 18),  # MLK Day
        date(2027, 2, 15),  # Presidents Day
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth (observed)
        date(2027, 7, 5),   # Independence Day (observed)
        date(2027, 9, 6),   # Labor Day
        date(2027, 11, 25), # Thanksgiving
        date(2027, 12, 24), # Christmas (observed)
    }

    # Early close days (1 PM) - day before/after major holidays
    EARLY_CLOSE_DAYS = {
        # 2025
        date(2025, 7, 3),   # Day before Independence Day
        date(2025, 11, 28), # Day after Thanksgiving
        date(2025, 12, 24), # Christmas Eve
        # 2026
        date(2026, 7, 2),   # Day before Independence Day (observed)
        date(2026, 11, 27), # Day after Thanksgiving
        date(2026, 12, 24), # Christmas Eve
        # 2027
        date(2027, 11, 26), # Day after Thanksgiving
        date(2027, 12, 23), # Day before Christmas (observed)
    }

    def __init__(self, orchestrator: 'TradingOrchestrator'):
        """
        Initialize TaskScheduler.

        Args:
            orchestrator: The TradingOrchestrator instance to schedule tasks for.
        """
        self.orchestrator = orchestrator
        self.state = SchedulerState()
        self._task_specs: Dict[str, TaskSpec] = {}
        self._holiday_cache: Dict[date, bool] = {}
        self._early_close_cache: Dict[date, bool] = {}

        # Load task specs (will be populated by task_specs.py)
        self._load_task_specs()

        # Load persisted state
        self._load_state()

    def _load_task_specs(self):
        """Load task specifications from registry."""
        try:
            from orchestration.task_specs import TASK_SPECS
            self._task_specs = TASK_SPECS.copy()
            logger.info(f"Loaded {len(self._task_specs)} task specifications")
        except ImportError:
            logger.warning("Task specs not found, using empty registry")
            self._task_specs = {}

    # =========================================================================
    # TIME WINDOW CALCULATION
    # =========================================================================

    def get_current_window(self) -> TimeWindow:
        """
        Get the current scheduling window with all context.

        Returns:
            TimeWindow with start, end, phase, and market status.
        """
        now = datetime.now(self.orchestrator.tz)
        phase = self.orchestrator.get_current_phase()

        # Check market calendar
        is_weekend = now.weekday() >= 5
        is_holiday = self._is_holiday(now.date())
        is_early_close = self._is_early_close(now.date())

        # Calculate window end based on phase
        end = self._calculate_window_end(now, phase, is_holiday)

        return TimeWindow(
            start=now,
            end=end,
            phase=phase.value,
            is_market_day=not is_weekend and not is_holiday,
            is_holiday=is_holiday,
            is_early_close=is_early_close,
        )

    def _calculate_window_end(self, now: datetime, phase, is_holiday: bool) -> datetime:
        """Calculate when current scheduling window ends."""
        from daily_orchestrator import MarketPhase

        if phase == MarketPhase.WEEKEND:
            return self._get_next_trading_premarket(now)

        if phase == MarketPhase.OVERNIGHT:
            # Check if tomorrow is holiday - extends window
            tomorrow = now.date() + timedelta(days=1)
            if is_holiday or self._is_holiday(tomorrow):
                return self._get_next_trading_premarket(now)
            else:
                # Normal overnight ends at pre-market
                return self._get_premarket_start(now)

        # For other phases, use phase config end time
        config = self.orchestrator.get_phase_config(phase)
        end = now.replace(
            hour=config.end_hour,
            minute=config.end_minute,
            second=0,
            microsecond=0
        )

        # Handle phases that span midnight
        if end <= now:
            end += timedelta(days=1)

        return end

    def _get_premarket_start(self, now: datetime) -> datetime:
        """Get pre-market start time (8:00 AM) for today or tomorrow."""
        premarket = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if premarket <= now:
            premarket += timedelta(days=1)
        return premarket

    def _get_next_trading_premarket(self, now: datetime) -> datetime:
        """Get next trading day pre-market start, skipping weekends and holidays."""
        check_date = now.date()

        # Move to next day if past pre-market
        if now.hour >= 8:
            check_date += timedelta(days=1)

        # Skip weekends and holidays
        max_days = 10  # Safety limit
        for _ in range(max_days):
            if check_date.weekday() >= 5:  # Weekend
                check_date += timedelta(days=1)
                continue
            if self._is_holiday(check_date):
                check_date += timedelta(days=1)
                continue
            break

        return datetime.combine(
            check_date,
            datetime.min.time().replace(hour=8, minute=0),
            tzinfo=self.orchestrator.tz
        )

    def get_available_minutes(self) -> float:
        """Get minutes available until next hard deadline."""
        window = self.get_current_window()
        return window.duration_minutes

    def calculate_research_window(self) -> timedelta:
        """
        Calculate total research time available until next trading.

        Accounts for:
        - Regular overnight (10.5 hours)
        - Weekend (56+ hours)
        - Monday holidays (extends weekend)
        - Early close days (starts research earlier)

        Returns:
            timedelta of available research time, minus reserved time for
            data refresh (1h), validation (30min), and prep (30min).
        """
        now = datetime.now(self.orchestrator.tz)

        # Find next trading session start
        next_trading = self._get_next_trading_premarket(now)

        # Find when research can start
        research_start = self._get_research_start(now)

        # Calculate total window
        total = next_trading - research_start

        # Subtract reserved time (2 hours for data/validation/prep)
        reserved = timedelta(hours=2)
        available = total - reserved

        return max(available, timedelta(hours=0))

    def _get_research_start(self, now: datetime) -> datetime:
        """Get when research can start."""
        from daily_orchestrator import MarketPhase

        phase = self.orchestrator.get_current_phase()

        if phase == MarketPhase.WEEKEND:
            # Weekend: research starts Friday 8 PM (after cleanup)
            # Calculate most recent Friday
            days_since_friday = (now.weekday() - 4) % 7
            friday = now - timedelta(days=days_since_friday)
            return friday.replace(hour=20, minute=0, second=0, microsecond=0)

        if phase == MarketPhase.EVENING:
            # Check for early close
            if self._is_early_close(now.date()):
                # Research can start at 3 PM instead of 9:30 PM
                return now.replace(hour=15, minute=0, second=0, microsecond=0)

        # Normal overnight: 9:30 PM
        return now.replace(hour=21, minute=30, second=0, microsecond=0)

    # =========================================================================
    # HOLIDAY DETECTION
    # =========================================================================

    def _is_holiday(self, check_date: date) -> bool:
        """
        Check if date is a market holiday.

        First tries Alpaca calendar API, falls back to hardcoded list.
        Results are cached for performance.
        """
        if check_date in self._holiday_cache:
            return self._holiday_cache[check_date]

        # Try Alpaca API first
        try:
            is_holiday = self._check_alpaca_calendar(check_date)
        except Exception as e:
            logger.debug(f"Alpaca calendar unavailable: {e}")
            # Fallback to hardcoded list
            is_holiday = check_date in self.HOLIDAYS

        self._holiday_cache[check_date] = is_holiday
        return is_holiday

    def _is_early_close(self, check_date: date) -> bool:
        """
        Check if date has early market close (1 PM).

        First tries Alpaca calendar API, falls back to hardcoded list.
        """
        if check_date in self._early_close_cache:
            return self._early_close_cache[check_date]

        # Try Alpaca API first
        try:
            is_early = self._check_alpaca_early_close(check_date)
        except Exception:
            # Fallback to hardcoded list
            is_early = check_date in self.EARLY_CLOSE_DAYS

        self._early_close_cache[check_date] = is_early
        return is_early

    def _check_alpaca_calendar(self, check_date: date) -> bool:
        """Query Alpaca API for market calendar."""
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetCalendarRequest

        api_key = os.environ.get('ALPACA_API_KEY')
        api_secret = os.environ.get('ALPACA_SECRET_KEY')

        if not api_key or not api_secret:
            raise ValueError("Alpaca credentials not configured")

        client = TradingClient(api_key, api_secret, paper=True)
        request = GetCalendarRequest(start=check_date, end=check_date)
        calendar = client.get_calendar(request)

        # Empty calendar means market closed (holiday)
        return len(calendar) == 0

    def _check_alpaca_early_close(self, check_date: date) -> bool:
        """Query Alpaca API for early close status."""
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetCalendarRequest

        api_key = os.environ.get('ALPACA_API_KEY')
        api_secret = os.environ.get('ALPACA_SECRET_KEY')

        if not api_key or not api_secret:
            raise ValueError("Alpaca credentials not configured")

        client = TradingClient(api_key, api_secret, paper=True)
        request = GetCalendarRequest(start=check_date, end=check_date)
        calendar = client.get_calendar(request)

        if calendar:
            close_time = calendar[0].close
            return close_time.hour == 13  # 1 PM close

        return False

    # =========================================================================
    # TASK SELECTION
    # =========================================================================

    def get_ready_tasks(self) -> List[TaskSpec]:
        """
        Get all tasks ready to run, sorted by priority.

        A task is ready if:
        1. Valid for current phase
        2. Dependencies completed
        3. Not conflicting with running tasks
        4. Respects run-once rules
        5. Has sufficient resources

        Returns:
            List of TaskSpec sorted by priority (CRITICAL first),
            then by estimated duration (shorter first).
        """
        window = self.get_current_window()
        ready = []

        for spec in self._task_specs.values():
            if self._is_task_ready(spec, window):
                ready.append(spec)

        # Sort by priority (CRITICAL=1 first), then estimated duration
        ready.sort(key=lambda t: (t.priority.value, t.estimated_minutes))

        return ready

    def _is_task_ready(self, spec: TaskSpec, window: TimeWindow) -> bool:
        """Check if a task is ready to execute."""
        # Phase check (empty phases means all phases valid)
        if spec.phases and window.phase not in spec.phases:
            return False

        # Market requirements
        if spec.requires_market_open and not self._is_market_open():
            return False
        if spec.requires_market_closed and self._is_market_open():
            return False

        # Already running
        if spec.name in self.state.running_tasks:
            return False

        # Conflicts with running tasks
        if spec.conflicts & self.state.running_tasks:
            return False

        # Dependencies not met
        if spec.dependencies:
            completed = self.state.phase_completions.get(window.phase, set())
            if not spec.dependencies.issubset(completed):
                return False

        # Run-once-per-phase check
        if spec.run_once_per_phase:
            completed = self.state.phase_completions.get(window.phase, set())
            if spec.name in completed:
                return False

        # Run-once-per-day check
        if spec.run_once_per_day:
            if self.state.run_counts_today.get(spec.name, 0) > 0:
                return False

        # Minimum interval check
        if spec.min_interval_minutes > 0:
            last = self.state.last_run.get(spec.name)
            if last:
                elapsed = (datetime.now(self.orchestrator.tz) - last).total_seconds() / 60
                if elapsed < spec.min_interval_minutes:
                    return False

        # Resource check - memory
        available_memory = self._get_available_memory()
        if spec.memory_mb > available_memory:
            logger.debug(f"Task {spec.name} blocked: needs {spec.memory_mb}MB, "
                        f"only {available_memory}MB available")
            return False

        # Resource check - CPU load for heavy tasks
        if spec.memory_mb > 500:  # Heavy task threshold
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 85:
                logger.debug(f"Task {spec.name} blocked: CPU at {cpu_percent}%, "
                            f"waiting for load to decrease")
                return False

        return True

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        from daily_orchestrator import MarketPhase

        phase = self.orchestrator.get_current_phase()
        return phase in (
            MarketPhase.MARKET_OPEN,
            MarketPhase.INTRADAY_OPEN,
            MarketPhase.INTRADAY_ACTIVE
        )

    # =========================================================================
    # EXECUTION PLANNING
    # =========================================================================

    def plan_phase(self) -> List[TaskSpec]:
        """
        Plan task execution for remaining phase time.

        Returns ordered list of tasks to execute, considering:
        - Time available
        - Task priorities
        - Dependencies
        - Resource constraints

        Returns:
            List of TaskSpec in recommended execution order.
        """
        window = self.get_current_window()
        available = window.duration_minutes

        plan = []
        planned_set = set()
        estimated_time = 0

        # First pass: required tasks (CRITICAL, HIGH)
        for spec in self.get_ready_tasks():
            if spec.priority in (TaskPriority.CRITICAL, TaskPriority.HIGH):
                est = self._get_estimated_duration(spec)
                if estimated_time + est <= available:
                    plan.append(spec)
                    planned_set.add(spec.name)
                    estimated_time += est

        # Second pass: normal priority tasks
        for spec in self.get_ready_tasks():
            if spec.name in planned_set:
                continue
            if spec.priority != TaskPriority.NORMAL:
                continue

            est = self._get_estimated_duration(spec)
            if estimated_time + est <= available:
                plan.append(spec)
                planned_set.add(spec.name)
                estimated_time += est

        # Third pass: opportunistic tasks (can be interrupted)
        for spec in self.get_ready_tasks():
            if spec.name in planned_set:
                continue
            if spec.priority not in (TaskPriority.LOW, TaskPriority.BACKGROUND):
                continue

            est = self._get_estimated_duration(spec)

            # For interruptible tasks, allow even if might not finish
            if spec.can_interrupt:
                remaining = available - estimated_time
                if remaining > 10:  # At least 10 minutes remaining
                    plan.append(spec)
                    planned_set.add(spec.name)
                    estimated_time += min(est, remaining)
            elif estimated_time + est <= available:
                plan.append(spec)
                planned_set.add(spec.name)
                estimated_time += est

        logger.info(f"Phase plan: {len(plan)} tasks, ~{estimated_time:.0f}min "
                   f"of {available:.0f}min available")

        return plan

    def backfill_with_opportunistic_tasks(self, remaining_minutes: float) -> List[TaskSpec]:
        """
        Find tasks that can fill remaining time after core tasks complete.

        Called when research finishes early or during extended windows.

        Args:
            remaining_minutes: Time remaining in current window.

        Returns:
            List of LOW/BACKGROUND priority tasks that can fit.
        """
        candidates = []

        for spec in self._task_specs.values():
            # Must be LOW or BACKGROUND priority
            if spec.priority not in (TaskPriority.LOW, TaskPriority.BACKGROUND):
                continue

            # Must be interruptible (can stop if time runs out)
            if not spec.can_interrupt:
                est = self._get_estimated_duration(spec)
                if est > remaining_minutes:
                    continue

            # Must be ready
            window = self.get_current_window()
            if self._is_task_ready(spec, window):
                candidates.append(spec)

        # Sort by priority then duration (shorter = more likely to complete)
        candidates.sort(key=lambda t: (t.priority.value, t.estimated_minutes))

        return candidates

    def should_extend_task(self, task_name: str) -> bool:
        """
        Determine if a running task should be extended.

        Called when a task reaches its estimated time but hasn't completed.

        Args:
            task_name: Name of the currently running task.

        Returns:
            True if there's time to continue, False to stop.
        """
        spec = self._task_specs.get(task_name)
        if not spec:
            return False

        window = self.get_current_window()

        # Never extend past max runtime
        started = self.state.last_run.get(task_name)
        if started:
            elapsed = (datetime.now(self.orchestrator.tz) - started).total_seconds() / 60
            if elapsed >= spec.max_runtime_minutes:
                logger.info(f"Task {task_name} reached max runtime ({spec.max_runtime_minutes}min)")
                return False

        # Check remaining time before next deadline
        remaining = window.duration_minutes

        # For research tasks in extended windows, be generous
        if spec.category == TaskCategory.RESEARCH and window.is_extended:
            return remaining > 30  # At least 30 min remaining

        # For normal tasks, require more buffer
        return remaining > 60

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def execute_next(self) -> Optional[TaskResult]:
        """
        Execute the highest priority ready task.

        Returns:
            TaskResult if a task was executed, None if no ready tasks.
        """
        ready = self.get_ready_tasks()
        if not ready:
            return None

        spec = ready[0]
        return self._execute_task(spec)

    def _execute_task(self, spec: TaskSpec) -> TaskResult:
        """Execute a single task with full tracking."""
        started = datetime.now(self.orchestrator.tz)
        self.state.running_tasks.add(spec.name)
        self.state.estimated_memory_used += spec.memory_mb

        logger.info(f"Executing task: {spec.name} (priority={spec.priority.name}, "
                   f"est={spec.estimated_minutes:.1f}min)")

        try:
            # Get handler method from orchestrator
            handler = getattr(self.orchestrator, spec.handler, None)
            if not handler:
                raise ValueError(f"Handler not found: {spec.handler}")

            # Execute the task
            success = handler()

            completed = datetime.now(self.orchestrator.tz)
            duration = (completed - started).total_seconds()

            result = TaskResult(
                task_name=spec.name,
                success=success if success is not None else True,
                started_at=started,
                completed_at=completed,
                duration_seconds=duration,
            )

            # Update state
            self._record_completion(spec, result)

            logger.info(f"Task {spec.name} completed: success={result.success}, "
                       f"duration={duration:.1f}s")

            return result

        except Exception as e:
            completed = datetime.now(self.orchestrator.tz)
            duration = (completed - started).total_seconds()

            result = TaskResult(
                task_name=spec.name,
                success=False,
                started_at=started,
                completed_at=completed,
                duration_seconds=duration,
                error=str(e),
            )

            logger.error(f"Task {spec.name} failed: {e}", exc_info=True)
            self._record_completion(spec, result)

            return result

        finally:
            self.state.running_tasks.discard(spec.name)
            self.state.estimated_memory_used = max(
                0, self.state.estimated_memory_used - spec.memory_mb
            )

    def _record_completion(self, spec: TaskSpec, result: TaskResult):
        """Record task completion in state."""
        window = self.get_current_window()

        # Update last run timestamp
        self.state.last_run[spec.name] = result.started_at

        # Update run counts
        self.state.run_counts_today[spec.name] = (
            self.state.run_counts_today.get(spec.name, 0) + 1
        )

        # Update phase completions (only on success)
        if result.success:
            if window.phase not in self.state.phase_completions:
                self.state.phase_completions[window.phase] = set()
            self.state.phase_completions[window.phase].add(spec.name)

        # Update duration estimate
        self.state.update_duration(spec.name, result.duration_minutes)

        # Update success rate
        self.state.update_success_rate(spec.name, result.success)

        # Persist state
        self._save_state()

    def record_execution(self, task_name: str, success: bool, phase: str):
        """
        Record task execution from external caller (e.g., legacy execution path).

        Simple interface for the orchestrator to report task completions
        when using run_task() directly instead of execute_next().

        Args:
            task_name: Name of the completed task.
            success: Whether the task succeeded.
            phase: Current phase name.
        """
        now = datetime.now(pytz.timezone('US/Eastern'))

        # Update last run timestamp
        self.state.last_run[task_name] = now

        # Update run counts
        self.state.run_counts_today[task_name] = (
            self.state.run_counts_today.get(task_name, 0) + 1
        )

        # Update phase completions (only on success)
        if success:
            if phase not in self.state.phase_completions:
                self.state.phase_completions[phase] = set()
            self.state.phase_completions[phase].add(task_name)

        # Update success rate
        self.state.update_success_rate(task_name, success)

        # Persist state periodically (not on every call)
        if sum(self.state.run_counts_today.values()) % 5 == 0:
            self._save_state()

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    def _get_available_memory(self) -> int:
        """
        Get available memory in MB for new tasks.

        Returns:
            Available MB after reserving 400MB for system and
            accounting for currently running tasks.
        """
        mem = psutil.virtual_memory()
        available = mem.available // (1024 * 1024)

        # Reserve 400MB for system
        # Subtract memory already allocated to running tasks
        usable = available - 400 - self.state.estimated_memory_used

        return max(0, usable)

    def _get_estimated_duration(self, spec: TaskSpec) -> float:
        """
        Get estimated duration using historical data if available.

        Returns:
            Estimated minutes with 20% buffer for uncertainty.
        """
        if spec.name in self.state.avg_durations:
            # Use historical average with buffer
            return self.state.avg_durations[spec.name] * 1.2
        return spec.estimated_minutes

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _load_state(self):
        """Load scheduler state from database."""
        import sqlite3
        from config import DATABASES

        db_path = DATABASES.get('research', 'db/research.db')

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_scheduler_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Load average durations
            cursor.execute('''
                SELECT key, value FROM task_scheduler_state
                WHERE key LIKE 'avg_duration_%'
            ''')
            for key, value in cursor.fetchall():
                task_name = key.replace('avg_duration_', '')
                self.state.avg_durations[task_name] = float(value)

            # Load success rates
            cursor.execute('''
                SELECT key, value FROM task_scheduler_state
                WHERE key LIKE 'success_rate_%'
            ''')
            for key, value in cursor.fetchall():
                task_name = key.replace('success_rate_', '')
                parts = value.split('/')
                if len(parts) == 2:
                    self.state.success_rates[task_name] = {
                        'success': int(parts[0]),
                        'total': int(parts[1])
                    }

            conn.close()
            logger.debug(f"Loaded {len(self.state.avg_durations)} task duration averages")

        except Exception as e:
            logger.debug(f"Could not load scheduler state: {e}")

    def _save_state(self):
        """Save scheduler state to database."""
        import sqlite3
        from config import DATABASES

        db_path = DATABASES.get('research', 'db/research.db')

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_scheduler_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Save average durations
            for task_name, duration in self.state.avg_durations.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO task_scheduler_state (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (f'avg_duration_{task_name}', str(duration)))

            # Save success rates
            for task_name, rates in self.state.success_rates.items():
                value = f"{rates.get('success', 0)}/{rates.get('total', 0)}"
                cursor.execute('''
                    INSERT OR REPLACE INTO task_scheduler_state (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (f'success_rate_{task_name}', value))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.debug(f"Could not save scheduler state: {e}")

    def reset_daily(self):
        """Reset daily tracking. Called at PRE_MARKET start."""
        self.state.run_counts_today.clear()
        self.state.phase_completions.clear()
        logger.info("TaskScheduler daily state reset")

    def reset_phase(self, phase: str):
        """Reset phase-specific tracking. Called on phase transition."""
        if phase in self.state.phase_completions:
            self.state.phase_completions[phase].clear()
        logger.debug(f"TaskScheduler phase state reset for {phase}")
