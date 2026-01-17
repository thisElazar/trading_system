# TaskScheduler Design Document (Option C)

## Overview

This document outlines the comprehensive refactor of the trading system's orchestration layer from rigid time-based phase execution to an intelligent, time-aware task scheduling system.

**Goal:** Maximize research utilization (currently ~19% on weekends, target ~67%) while maintaining trading reliability and respecting Pi 5 resource constraints (4GB RAM, 4 cores).

---

## Current Architecture Analysis

### Phase System (Rigid Time-Based)

```
Phase           | Hours (ET)        | Check Interval | Tasks
----------------|-------------------|----------------|-------
PRE_MARKET      | 8:00-9:30         | 60s           | 11
INTRADAY_OPEN   | 9:30-9:35         | 5s            | 2
INTRADAY_ACTIVE | 9:35-11:30        | 10s           | 1
MARKET_OPEN     | 9:30-16:00        | 30s           | 5
POST_MARKET     | 16:00-17:00       | 60s           | 8
EVENING         | 17:00-21:30       | 300s          | 4
OVERNIGHT       | 21:30-8:00        | 600s          | 2
WEEKEND         | Fri 16:00-Sun 20:00| 1800s        | 9 (via sub-phases)
```

### Problems with Current System

1. **Wasted Time Windows:**
   - OVERNIGHT has 10.5 hours but research only runs 4 hours
   - Weekend has 56+ hours but research is capped
   - Early market close days (1 PM) don't extend research

2. **No Dynamic Adaptation:**
   - Monday holidays don't trigger extended research
   - Task durations aren't tracked/predicted
   - No backfilling when tasks finish early

3. **Rigid Transitions:**
   - Hard cutoff at phase boundaries (e.g., 8:00 AM kills research)
   - No grace period for nearly-complete tasks
   - No priority for time-sensitive vs opportunistic work

4. **No Resource-Aware Scheduling:**
   - Tasks compete for memory without coordination
   - Heavy validation can overlap with research
   - No memory budget allocation per task

---

## Proposed TaskScheduler Architecture

### Core Dataclasses

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Set, Optional, Callable, Dict, Any
from datetime import datetime, timedelta

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1    # Must complete (trading, alerts)
    HIGH = 2        # Should complete (EOD reconciliation)
    NORMAL = 3      # Standard tasks (data refresh)
    LOW = 4         # Opportunistic (extended research)
    BACKGROUND = 5  # Fill gaps (validation subprocess)

class TaskCategory(Enum):
    """Task categories for resource grouping."""
    TRADING = auto()      # Broker interactions
    DATA = auto()         # Data fetching/refresh
    RESEARCH = auto()     # GA/GP optimization
    MAINTENANCE = auto()  # Cleanup, backups
    MONITORING = auto()   # Health checks, alerts

@dataclass
class TaskSpec:
    """Specification for a schedulable task."""
    name: str
    handler: str  # Method name in orchestrator (e.g., "_task_refresh_data")
    priority: TaskPriority
    category: TaskCategory

    # Time constraints
    estimated_minutes: float = 5.0
    max_runtime_minutes: float = 30.0
    can_interrupt: bool = False  # Can be gracefully stopped

    # Resource requirements
    memory_mb: int = 200
    cpu_intensive: bool = False

    # Scheduling constraints
    phases: Set[str] = field(default_factory=set)  # Valid phases
    dependencies: Set[str] = field(default_factory=set)  # Must complete first
    conflicts: Set[str] = field(default_factory=set)  # Cannot run simultaneously

    # Execution rules
    run_once_per_phase: bool = False
    run_once_per_day: bool = False
    min_interval_minutes: float = 0  # Minimum time between runs

    # Market awareness
    requires_market_open: bool = False
    requires_market_closed: bool = False

    def __hash__(self):
        return hash(self.name)

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

@dataclass
class TimeWindow:
    """Represents available time for scheduling."""
    start: datetime
    end: datetime
    phase: str
    is_market_day: bool
    is_holiday: bool
    is_early_close: bool

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    @property
    def is_extended(self) -> bool:
        """True if this is an extended window (holiday, weekend)."""
        return self.is_holiday or not self.is_market_day

@dataclass
class SchedulerState:
    """Persistent state for task scheduler."""
    # Execution tracking
    last_run: Dict[str, datetime] = field(default_factory=dict)
    run_counts_today: Dict[str, int] = field(default_factory=dict)
    phase_completions: Dict[str, Set[str]] = field(default_factory=dict)

    # Performance metrics (for estimation)
    avg_durations: Dict[str, float] = field(default_factory=dict)
    success_rates: Dict[str, float] = field(default_factory=dict)

    # Current execution
    running_tasks: Set[str] = field(default_factory=set)
    queued_tasks: List[str] = field(default_factory=list)

    # Resource tracking
    estimated_memory_used: int = 0

    def update_duration(self, task_name: str, duration: float):
        """Update rolling average duration for a task."""
        if task_name in self.avg_durations:
            # Exponential moving average (alpha=0.3)
            self.avg_durations[task_name] = 0.7 * self.avg_durations[task_name] + 0.3 * duration
        else:
            self.avg_durations[task_name] = duration
```

### Task Registry

```python
# Complete task registry with Option C specifications
TASK_SPECS: Dict[str, TaskSpec] = {
    # === CRITICAL: Trading Operations ===
    "monitor_positions": TaskSpec(
        name="monitor_positions",
        handler="_task_monitor_positions",
        priority=TaskPriority.CRITICAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=100,
        phases={"MARKET_OPEN", "INTRADAY_ACTIVE"},
        min_interval_minutes=0.5,
        requires_market_open=True,
    ),

    "check_risk_limits": TaskSpec(
        name="check_risk_limits",
        handler="_task_check_risk_limits",
        priority=TaskPriority.CRITICAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.1,
        memory_mb=50,
        phases={"MARKET_OPEN", "INTRADAY_ACTIVE"},
        min_interval_minutes=0.5,
        requires_market_open=True,
    ),

    # === HIGH: EOD Operations ===
    "reconcile_positions": TaskSpec(
        name="reconcile_positions",
        handler="_task_reconcile_positions",
        priority=TaskPriority.HIGH,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        memory_mb=150,
        phases={"POST_MARKET"},
        run_once_per_day=True,
        requires_market_closed=True,
    ),

    "calculate_pnl": TaskSpec(
        name="calculate_pnl",
        handler="_task_calculate_pnl",
        priority=TaskPriority.HIGH,
        category=TaskCategory.TRADING,
        estimated_minutes=0.5,
        memory_mb=100,
        phases={"POST_MARKET"},
        dependencies={"reconcile_positions"},
        run_once_per_day=True,
    ),

    "send_alerts": TaskSpec(
        name="send_alerts",
        handler="_task_send_alerts",
        priority=TaskPriority.HIGH,
        category=TaskCategory.MONITORING,
        estimated_minutes=0.5,
        memory_mb=50,
        phases={"POST_MARKET"},
        dependencies={"calculate_pnl"},
        run_once_per_day=True,
    ),

    # === NORMAL: Data Operations ===
    "refresh_data": TaskSpec(
        name="refresh_data",
        handler="_task_refresh_data",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=2,
        memory_mb=300,
        phases={"PRE_MARKET"},
        run_once_per_phase=True,
    ),

    "refresh_eod_data": TaskSpec(
        name="refresh_eod_data",
        handler="_task_refresh_eod_data",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=30,
        max_runtime_minutes=60,
        memory_mb=400,
        cpu_intensive=True,
        phases={"EVENING", "WEEKEND"},
        run_once_per_day=True,
        conflicts={"run_nightly_research", "run_weekend_research"},
    ),

    "refresh_index_constituents": TaskSpec(
        name="refresh_index_constituents",
        handler="_task_refresh_index_constituents",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=5,
        memory_mb=100,
        phases={"WEEKEND"},
        run_once_per_day=True,
    ),

    "refresh_fundamentals": TaskSpec(
        name="refresh_fundamentals",
        handler="_task_refresh_fundamentals",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=60,
        max_runtime_minutes=120,
        memory_mb=400,
        can_interrupt=True,
        phases={"WEEKEND"},
        run_once_per_day=True,
    ),

    # === NORMAL: Research (Core) ===
    "run_nightly_research": TaskSpec(
        name="run_nightly_research",
        handler="_task_run_nightly_research",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.RESEARCH,
        estimated_minutes=240,  # 4 hours baseline
        max_runtime_minutes=480,  # Can extend to 8 hours
        memory_mb=1500,
        cpu_intensive=True,
        can_interrupt=True,
        phases={"OVERNIGHT"},
        conflicts={"refresh_eod_data", "validate_candidates"},
        requires_market_closed=True,
    ),

    "run_weekend_research": TaskSpec(
        name="run_weekend_research",
        handler="_task_run_weekend_research",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.RESEARCH,
        estimated_minutes=480,  # 8 hours baseline
        max_runtime_minutes=2160,  # Can extend to 36 hours
        memory_mb=1500,
        cpu_intensive=True,
        can_interrupt=True,
        phases={"WEEKEND"},
        conflicts={"refresh_eod_data"},
    ),

    # === LOW: Opportunistic Tasks ===
    "validate_candidates": TaskSpec(
        name="validate_candidates",
        handler="_task_validate_candidates",
        priority=TaskPriority.LOW,
        category=TaskCategory.RESEARCH,
        estimated_minutes=30,
        max_runtime_minutes=60,
        memory_mb=1500,
        can_interrupt=True,
        phases={"OVERNIGHT", "WEEKEND"},
        dependencies={"run_nightly_research"},  # After core research
        conflicts={"run_nightly_research", "run_weekend_research"},
    ),

    "train_ml_regime_model": TaskSpec(
        name="train_ml_regime_model",
        handler="_task_train_ml_regime_model",
        priority=TaskPriority.LOW,
        category=TaskCategory.RESEARCH,
        estimated_minutes=15,
        memory_mb=500,
        phases={"OVERNIGHT", "WEEKEND"},
        min_interval_minutes=7 * 24 * 60,  # Weekly
    ),

    # === BACKGROUND: Maintenance ===
    "backup_databases": TaskSpec(
        name="backup_databases",
        handler="_task_backup_databases",
        priority=TaskPriority.BACKGROUND,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=2,
        memory_mb=100,
        phases={"EVENING", "WEEKEND"},
        run_once_per_day=True,
    ),

    "cleanup_logs": TaskSpec(
        name="cleanup_logs",
        handler="_task_cleanup_logs",
        priority=TaskPriority.BACKGROUND,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=1,
        memory_mb=50,
        phases={"EVENING", "WEEKEND"},
        run_once_per_day=True,
    ),

    "vacuum_databases": TaskSpec(
        name="vacuum_databases",
        handler="_task_vacuum_databases",
        priority=TaskPriority.BACKGROUND,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=5,
        memory_mb=100,
        phases={"WEEKEND"},
        min_interval_minutes=7 * 24 * 60,  # Weekly
    ),
}
```

### TaskScheduler Class

```python
class TaskScheduler:
    """
    Time-aware task scheduler with priority, dependencies, and resource management.

    Key Features:
    - Dynamic time window calculation (holidays, early close)
    - Priority-based task selection
    - Dependency resolution
    - Resource-aware execution (memory, CPU)
    - Opportunistic task backfilling
    """

    def __init__(self, orchestrator: 'TradingOrchestrator'):
        self.orchestrator = orchestrator
        self.state = SchedulerState()
        self.task_specs = TASK_SPECS.copy()
        self._load_state()

    # === Time Window Calculation ===

    def get_current_window(self) -> TimeWindow:
        """Get the current scheduling window with all context."""
        now = datetime.now(self.orchestrator.tz)
        phase = self.orchestrator.get_current_phase()

        # Check market calendar
        is_market_day = not self._is_weekend(now)
        is_holiday = self._is_holiday(now.date())
        is_early_close = self._is_early_close(now.date())

        # Calculate window end
        if phase == MarketPhase.WEEKEND:
            end = self._get_monday_premarket(now)
        elif phase == MarketPhase.OVERNIGHT:
            end = self._get_premarket_start(now)
            # Extend if tomorrow is holiday
            if is_holiday or self._is_holiday(now.date() + timedelta(days=1)):
                end = self._get_next_trading_premarket(now)
        else:
            end = self._get_phase_end(phase, now)

        return TimeWindow(
            start=now,
            end=end,
            phase=phase.value,
            is_market_day=is_market_day,
            is_holiday=is_holiday,
            is_early_close=is_early_close,
        )

    def get_available_minutes(self) -> float:
        """Get minutes available until next hard deadline."""
        window = self.get_current_window()
        return window.duration_minutes

    # === Task Selection ===

    def get_ready_tasks(self) -> List[TaskSpec]:
        """
        Get all tasks ready to run, sorted by priority.

        A task is ready if:
        1. Valid for current phase
        2. Dependencies completed
        3. Not conflicting with running tasks
        4. Respects run-once rules
        5. Has sufficient resources
        """
        window = self.get_current_window()
        ready = []

        for spec in self.task_specs.values():
            if self._is_task_ready(spec, window):
                ready.append(spec)

        # Sort by priority (CRITICAL=1 first), then by estimated duration (shorter first)
        ready.sort(key=lambda t: (t.priority.value, t.estimated_minutes))

        return ready

    def _is_task_ready(self, spec: TaskSpec, window: TimeWindow) -> bool:
        """Check if a task is ready to execute."""
        # Phase check
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

        # Conflicts
        if spec.conflicts & self.state.running_tasks:
            return False

        # Dependencies
        if spec.dependencies:
            completed = self.state.phase_completions.get(window.phase, set())
            if not spec.dependencies.issubset(completed):
                return False

        # Run-once rules
        if spec.run_once_per_phase:
            completed = self.state.phase_completions.get(window.phase, set())
            if spec.name in completed:
                return False

        if spec.run_once_per_day:
            if spec.name in self.state.run_counts_today:
                return False

        # Interval check
        if spec.min_interval_minutes > 0:
            last = self.state.last_run.get(spec.name)
            if last:
                elapsed = (datetime.now(self.orchestrator.tz) - last).total_seconds() / 60
                if elapsed < spec.min_interval_minutes:
                    return False

        # Resource check
        available_memory = self._get_available_memory()
        if spec.memory_mb > available_memory:
            return False

        return True

    # === Execution Planning ===

    def plan_phase(self) -> List[TaskSpec]:
        """
        Plan task execution for remaining phase time.

        Returns ordered list of tasks to execute, considering:
        - Time available
        - Task priorities
        - Dependencies
        - Resource constraints
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

        # Second pass: opportunistic tasks that fit
        for spec in self.get_ready_tasks():
            if spec.name in planned_set:
                continue

            est = self._get_estimated_duration(spec)

            # For interruptible tasks, allow them even if they might not finish
            if spec.can_interrupt:
                est = min(est, available - estimated_time)

            if estimated_time + est <= available:
                plan.append(spec)
                planned_set.add(spec.name)
                estimated_time += est

        return plan

    def should_extend_task(self, task_name: str) -> bool:
        """
        Determine if a running task should be extended.

        Called when a task reaches its estimated time but hasn't completed.
        Returns True if there's time to continue.
        """
        spec = self.task_specs.get(task_name)
        if not spec:
            return False

        window = self.get_current_window()

        # Never extend past max runtime
        started = self.state.last_run.get(task_name)
        if started:
            elapsed = (datetime.now(self.orchestrator.tz) - started).total_seconds() / 60
            if elapsed >= spec.max_runtime_minutes:
                return False

        # Check if we have time before next hard deadline
        remaining = window.duration_minutes

        # For research tasks, check if extended window available
        if spec.category == TaskCategory.RESEARCH and window.is_extended:
            return remaining > 30  # At least 30 min remaining

        # For normal tasks, only extend if plenty of time
        return remaining > 60

    # === Execution ===

    def execute_next(self) -> Optional[TaskResult]:
        """Execute the highest priority ready task."""
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

        try:
            # Get handler method
            handler = getattr(self.orchestrator, spec.handler, None)
            if not handler:
                raise ValueError(f"Handler not found: {spec.handler}")

            # Execute
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

            self._record_completion(spec, result)
            return result

        finally:
            self.state.running_tasks.discard(spec.name)
            self.state.estimated_memory_used -= spec.memory_mb

    def _record_completion(self, spec: TaskSpec, result: TaskResult):
        """Record task completion in state."""
        window = self.get_current_window()

        # Update last run
        self.state.last_run[spec.name] = result.started_at

        # Update run counts
        if spec.name not in self.state.run_counts_today:
            self.state.run_counts_today[spec.name] = 0
        self.state.run_counts_today[spec.name] += 1

        # Update phase completions
        if window.phase not in self.state.phase_completions:
            self.state.phase_completions[window.phase] = set()
        if result.success:
            self.state.phase_completions[window.phase].add(spec.name)

        # Update duration estimate
        self.state.update_duration(spec.name, result.duration_seconds / 60)

        # Update success rate
        if spec.name not in self.state.success_rates:
            self.state.success_rates[spec.name] = 1.0
        alpha = 0.2
        self.state.success_rates[spec.name] = (
            (1 - alpha) * self.state.success_rates[spec.name] +
            alpha * (1.0 if result.success else 0.0)
        )

    # === Helper Methods ===

    def _get_estimated_duration(self, spec: TaskSpec) -> float:
        """Get estimated duration using historical data if available."""
        if spec.name in self.state.avg_durations:
            # Use historical average with 20% buffer
            return self.state.avg_durations[spec.name] * 1.2
        return spec.estimated_minutes

    def _get_available_memory(self) -> int:
        """Get available memory in MB."""
        import psutil
        mem = psutil.virtual_memory()
        available = mem.available // (1024 * 1024)
        # Reserve 400MB for system
        return max(0, available - 400 - self.state.estimated_memory_used)

    # === State Persistence ===

    def _load_state(self):
        """Load scheduler state from database."""
        # Implementation: Load from performance.db or dedicated scheduler.db
        pass

    def _save_state(self):
        """Save scheduler state to database."""
        # Implementation: Save to performance.db or dedicated scheduler.db
        pass

    def reset_daily(self):
        """Reset daily tracking (called at PRE_MARKET start)."""
        self.state.run_counts_today.clear()
        self.state.phase_completions.clear()
```

---

## Integration with Daily Orchestrator

### Modified Run Loop

```python
def run(self, once: bool = False):
    """Main run loop with TaskScheduler integration."""

    while not self.shutdown_event.is_set():
        current_phase = self.get_current_phase()

        # Phase transition handling (existing logic)
        if current_phase != self.state.current_phase:
            self._handle_phase_transition(current_phase)

        # === NEW: TaskScheduler integration ===

        # Get scheduling context
        window = self.scheduler.get_current_window()
        available = window.duration_minutes

        # Log context for debugging
        logger.info(
            f"Phase {current_phase.value}: {available:.0f}min available, "
            f"extended={window.is_extended}, holiday={window.is_holiday}"
        )

        # Execute tasks based on scheduler plan
        plan = self.scheduler.plan_phase()

        for spec in plan:
            # Check for interrupts
            if self.shutdown_event.is_set():
                break

            # Execute task
            result = self.scheduler._execute_task(spec)

            # Log result
            status = "OK" if result.success else "FAIL"
            logger.info(
                f"  {spec.name}: {status} ({result.duration_seconds:.1f}s)"
            )

            # Update hardware display
            if self._hardware:
                self._update_hardware_display(current_phase)

        # === END TaskScheduler integration ===

        # Sleep until next check
        config = self.get_phase_config(current_phase)
        self._interruptible_sleep(config.check_interval_seconds)
```

### Holiday Detection Enhancement

```python
def _is_holiday(self, date: datetime.date) -> bool:
    """Check if date is a market holiday using Alpaca calendar."""
    # First check local cache
    if hasattr(self, '_holiday_cache') and date in self._holiday_cache:
        return self._holiday_cache[date]

    # Try Alpaca calendar API
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.environ['ALPACA_API_KEY'],
            os.environ['ALPACA_SECRET_KEY'],
            paper=True
        )
        calendar = client.get_calendar(
            start=date.isoformat(),
            end=date.isoformat()
        )
        is_holiday = len(calendar) == 0
    except Exception:
        # Fallback to hardcoded list
        is_holiday = date in self._get_hardcoded_holidays()

    # Cache result
    if not hasattr(self, '_holiday_cache'):
        self._holiday_cache = {}
    self._holiday_cache[date] = is_holiday

    return is_holiday

def _is_early_close(self, date: datetime.date) -> bool:
    """Check if date has early market close (1 PM)."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.environ['ALPACA_API_KEY'],
            os.environ['ALPACA_SECRET_KEY'],
            paper=True
        )
        calendar = client.get_calendar(
            start=date.isoformat(),
            end=date.isoformat()
        )
        if calendar:
            close_time = calendar[0].close
            return close_time.hour == 13  # 1 PM close
    except Exception:
        pass

    # Fallback to hardcoded list
    return date in self._get_early_close_dates()
```

---

## Extended Research Window Calculation

```python
def calculate_research_window(self) -> timedelta:
    """
    Calculate total research time available until next trading.

    Accounts for:
    - Regular overnight (10.5 hours)
    - Weekend (56+ hours)
    - Monday holidays (extends weekend)
    - Early close days (starts research earlier)
    """
    now = datetime.now(self.tz)

    # Find next trading session start
    next_trading = self._get_next_trading_premarket(now)

    # Find when research can start
    research_start = self._get_research_start(now)

    # Calculate total window
    total = next_trading - research_start

    # Subtract reserved time
    # - 1 hour for data refresh (before trading)
    # - 30 min for validation
    # - 30 min for system prep
    reserved = timedelta(hours=2)

    available = total - reserved

    return max(available, timedelta(hours=0))

def _get_research_start(self, now: datetime) -> datetime:
    """Get when research can start."""
    current_phase = self.get_current_phase()

    if current_phase == MarketPhase.WEEKEND:
        # Weekend: after Friday cleanup (8 PM)
        friday = now - timedelta(days=now.weekday() - 4)
        return friday.replace(hour=20, minute=0, second=0, microsecond=0)

    elif current_phase == MarketPhase.EVENING:
        # Check for early close
        if self._is_early_close(now.date()):
            # Research can start at 3 PM instead of 9:30 PM
            return now.replace(hour=15, minute=0, second=0, microsecond=0)

    # Normal overnight: 9:30 PM
    return now.replace(hour=21, minute=30, second=0, microsecond=0)

def _get_next_trading_premarket(self, now: datetime) -> datetime:
    """Get next trading day pre-market start."""
    check_date = now.date()

    # Move to next day if past pre-market
    if now.hour >= 8:
        check_date += timedelta(days=1)

    # Skip weekends and holidays
    while True:
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
        tzinfo=self.tz
    )
```

---

## Opportunistic Task Scheduling

```python
def backfill_with_opportunistic_tasks(self, remaining_minutes: float) -> List[TaskSpec]:
    """
    Find tasks that can fill remaining time after core tasks complete.

    Called when research finishes early or during extended windows.
    """
    candidates = []

    for spec in self.task_specs.values():
        # Must be LOW or BACKGROUND priority
        if spec.priority not in (TaskPriority.LOW, TaskPriority.BACKGROUND):
            continue

        # Must be interruptible (can stop if time runs out)
        if not spec.can_interrupt:
            continue

        # Must fit in remaining time (or be interruptible)
        est = self._get_estimated_duration(spec)
        if est > remaining_minutes and not spec.can_interrupt:
            continue

        # Must be ready
        window = self.get_current_window()
        if self._is_task_ready(spec, window):
            candidates.append(spec)

    # Sort by value: priority then estimated duration (shorter = more likely to complete)
    candidates.sort(key=lambda t: (t.priority.value, t.estimated_minutes))

    return candidates
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Minimal Changes)
1. Add `TaskSpec` and `TaskResult` dataclasses to `daily_orchestrator.py`
2. Add `TaskScheduler` class with basic `get_ready_tasks()` and `execute_next()`
3. Create `TASK_SPECS` registry for top 20 tasks (highest impact)
4. Add `get_available_minutes()` using existing `time_until_next_phase()`

### Phase 2: Enhanced Time Awareness
1. Implement `_is_holiday()` with Alpaca calendar API
2. Implement `_is_early_close()` detection
3. Add `calculate_research_window()` for dynamic research time
4. Integrate holiday detection into weekend sub-phase system

### Phase 3: Opportunistic Scheduling
1. Add `backfill_with_opportunistic_tasks()`
2. Implement `validate_candidates` task (subprocess-based)
3. Add task extension logic (`should_extend_task()`)
4. Track actual durations for better estimation

### Phase 4: Resource-Aware Execution
1. Add memory tracking to `SchedulerState`
2. Implement `_get_available_memory()` check
3. Add conflict detection for memory-heavy tasks
4. Integrate with existing `_check_memory_status()`

### Phase 5: Monitoring & Polish
1. Add scheduler metrics to dashboard
2. Create scheduler state persistence
3. Add alerting for scheduling anomalies
4. Performance tuning based on metrics

---

## Migration Strategy

### Backwards Compatibility

The TaskScheduler is designed to coexist with the existing phase system:

1. **PhaseConfig still defines valid phases** - TaskSpec.phases references these
2. **ONCE_PER_PHASE_TASKS preserved** - Maps to TaskSpec.run_once_per_phase
3. **Existing handlers unchanged** - TaskSpec.handler points to `_task_*` methods
4. **Weekend sub-phases preserved** - TaskScheduler respects sub-phase boundaries

### Incremental Rollout

```python
# Feature flag for gradual rollout
USE_TASK_SCHEDULER = os.environ.get('USE_TASK_SCHEDULER', 'false').lower() == 'true'

def run(self, once: bool = False):
    while not self.shutdown_event.is_set():
        current_phase = self.get_current_phase()

        if USE_TASK_SCHEDULER:
            # New scheduler-based execution
            self._run_with_scheduler(current_phase)
        else:
            # Existing phase-based execution
            self._run_legacy(current_phase)
```

---

## Expected Improvements

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Weekend research utilization | ~19% (8h/56h) | ~67% (36h/56h) | Dynamic windows |
| Overnight research utilization | ~38% (4h/10.5h) | ~76% (8h/10.5h) | Holiday extension |
| Task completion predictability | Unknown | 90%+ | Duration tracking |
| Memory conflict rate | Occasional | Near-zero | Conflict detection |
| Research on holiday weekends | No change | +50% time | Holiday detection |

---

## File Locations

All code should be added to:
- `daily_orchestrator.py` - Main scheduler integration
- `orchestration/task_scheduler.py` (NEW) - TaskScheduler class
- `orchestration/task_specs.py` (NEW) - Task registry
- `config.py` - Feature flags and thresholds

---

## Appendix: Complete Task Spec Reference

See the `TASK_SPECS` dictionary above for the complete 44-task registry with all specifications.
