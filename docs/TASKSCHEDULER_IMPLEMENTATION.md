# TaskScheduler Implementation Plan

## Quick Reference

**Target:** Option C - Full task queue with priorities, dependencies, and time-aware scheduling
**Expected Duration:** 5 implementation phases
**Risk Level:** Medium (non-breaking, feature-flagged)

---

## Phase 1: Core Infrastructure

### Commit 1.1: Add Task Specification Dataclasses

**Files to create:**
- `orchestration/task_scheduler.py`

**Content:**
```python
# Add TaskPriority, TaskCategory, TaskSpec, TaskResult, TimeWindow, SchedulerState
# See TASKSCHEDULER_DESIGN.md for complete implementations
```

**Test:** Import without errors
```bash
python -c "from orchestration.task_scheduler import TaskSpec, TaskPriority"
```

### Commit 1.2: Create Task Registry

**Files to create:**
- `orchestration/task_specs.py`

**Content:**
- Top 20 highest-impact TaskSpecs (CRITICAL + HIGH priority)
- Start with: monitor_positions, check_risk_limits, reconcile_positions, calculate_pnl, send_alerts, refresh_data, refresh_eod_data, run_nightly_research, backup_databases, cleanup_logs

**Test:**
```bash
python -c "from orchestration.task_specs import TASK_SPECS; print(len(TASK_SPECS))"
```

### Commit 1.3: Basic TaskScheduler Class

**Files to modify:**
- `orchestration/task_scheduler.py` (add TaskScheduler class)

**Methods to implement:**
- `__init__(orchestrator)`
- `get_current_window()` - Use existing `time_until_next_phase()`
- `get_ready_tasks()` - Basic filtering without dependencies
- `execute_next()` - Single task execution

**Test:**
```python
# Manual test in Python shell
from daily_orchestrator import TradingOrchestrator
from orchestration.task_scheduler import TaskScheduler

orch = TradingOrchestrator()
sched = TaskScheduler(orch)
print(sched.get_current_window())
print(sched.get_ready_tasks())
```

### Commit 1.4: Feature Flag Integration

**Files to modify:**
- `config.py` - Add `USE_TASK_SCHEDULER = False`
- `daily_orchestrator.py` - Add conditional in `run()` loop

**Changes:**
```python
# config.py
USE_TASK_SCHEDULER = os.environ.get('USE_TASK_SCHEDULER', 'false').lower() == 'true'

# daily_orchestrator.py
from config import USE_TASK_SCHEDULER

def run(self, once: bool = False):
    while not self.shutdown_event.is_set():
        # ... phase transition handling ...

        if USE_TASK_SCHEDULER:
            self._run_with_scheduler(current_phase)
        else:
            self._run_legacy(current_phase)
```

**Test:** Run orchestrator with flag off (existing behavior), then on (new behavior)

---

## Phase 2: Enhanced Time Awareness

### Commit 2.1: Holiday Detection via Alpaca API

**Files to modify:**
- `orchestration/task_scheduler.py`

**Add methods:**
- `_is_holiday(date)` - Query Alpaca calendar, cache result
- `_is_early_close(date)` - Check for 1 PM close
- `_holiday_cache` - Dict for caching API responses

**Test:**
```python
sched = TaskScheduler(orch)
print(sched._is_holiday(date(2026, 1, 19)))  # MLK Day
print(sched._is_early_close(date(2026, 11, 27)))  # Day after Thanksgiving
```

### Commit 2.2: Dynamic Research Window Calculation

**Files to modify:**
- `orchestration/task_scheduler.py`

**Add methods:**
- `calculate_research_window()` - Total time until next trading
- `_get_research_start(now)` - When research can begin
- `_get_next_trading_premarket(now)` - Next trading day 8 AM

**Test:**
```python
# Test on Friday evening
window = sched.calculate_research_window()
print(f"Research window: {window.total_seconds() / 3600:.1f} hours")
# Should be ~56 hours for normal weekend, more for holiday
```

### Commit 2.3: Integrate Holiday Detection into Weekend System

**Files to modify:**
- `daily_orchestrator.py` - Update `_get_weekend_sub_phase()` and `_task_run_weekend_schedule()`

**Changes:**
- Check if Monday is holiday when calculating RESEARCH phase duration
- Log extended window detection
- Update hardware display to show extended mode

**Test:** Set up test with mock holiday, verify research window extends

---

## Phase 3: Opportunistic Scheduling

### Commit 3.1: Task Dependency Resolution

**Files to modify:**
- `orchestration/task_scheduler.py`

**Enhance `_is_task_ready()`:**
- Check `spec.dependencies` against `state.phase_completions`
- Check `spec.conflicts` against `state.running_tasks`

**Test:**
```python
# Verify calculate_pnl waits for reconcile_positions
ready = sched.get_ready_tasks()
assert 'calculate_pnl' not in [t.name for t in ready]  # Before reconcile
sched._record_completion(TASK_SPECS['reconcile_positions'], mock_result)
ready = sched.get_ready_tasks()
assert 'calculate_pnl' in [t.name for t in ready]  # After reconcile
```

### Commit 3.2: Add validate_candidates Task

**Files to modify:**
- `orchestration/task_specs.py` - Add TaskSpec
- `daily_orchestrator.py` - Add `_task_validate_candidates()` handler

**Handler implementation:**
```python
def _task_validate_candidates(self) -> bool:
    """Run validation subprocess for CANDIDATE strategies."""
    import subprocess

    cmd = [
        sys.executable,
        'scripts/validate_strategies_subprocess.py',
        '--all-candidates',
        '--memory-limit', '1500',
    ]

    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent),
        capture_output=True,
        timeout=3600,  # 1 hour max
    )

    return result.returncode == 0
```

**Test:** Run manually, verify subprocess executes

### Commit 3.3: Backfill with Opportunistic Tasks

**Files to modify:**
- `orchestration/task_scheduler.py`

**Add method:**
- `backfill_with_opportunistic_tasks(remaining_minutes)`

**Integration:**
- After core research completes, check for remaining time
- Execute LOW/BACKGROUND priority tasks that fit

**Test:** Mock research completion at 50% of window, verify backfill triggers

### Commit 3.4: Task Extension Logic

**Files to modify:**
- `orchestration/task_scheduler.py`

**Add method:**
- `should_extend_task(task_name)` - Check if task should continue past estimate

**Integration:**
- Research tasks check this when reaching estimated time
- Allows research to extend into available window

**Test:** Start research, verify it extends when time available

---

## Phase 4: Resource-Aware Execution

### Commit 4.1: Memory Tracking in SchedulerState

**Files to modify:**
- `orchestration/task_scheduler.py`

**Add fields to SchedulerState:**
- `estimated_memory_used: int = 0`

**Update `_execute_task()`:**
- Add memory to estimate on start
- Subtract on completion/error

**Test:** Run memory-heavy task, verify tracking updates

### Commit 4.2: Memory Check in Task Readiness

**Files to modify:**
- `orchestration/task_scheduler.py`

**Enhance `_is_task_ready()`:**
- Call `_get_available_memory()`
- Compare against `spec.memory_mb`
- Block task if insufficient memory

**Test:** Mock low memory, verify heavy tasks blocked

### Commit 4.3: Conflict Detection for Memory-Heavy Tasks

**Files to modify:**
- `orchestration/task_specs.py` - Add `conflicts` to heavy tasks

**Update specs:**
- `run_nightly_research.conflicts = {"refresh_eod_data", "validate_candidates"}`
- `refresh_eod_data.conflicts = {"run_nightly_research"}`

**Test:** Try to schedule conflicting tasks, verify rejection

### Commit 4.4: Integration with Existing Memory Monitoring

**Files to modify:**
- `daily_orchestrator.py`

**Connect:**
- Use existing `_check_memory_status()` values in scheduler
- Log memory decisions
- Add memory alert if task blocked repeatedly

**Test:** Run during low memory, verify appropriate blocking

---

## Phase 5: Monitoring & Polish

### Commit 5.1: Scheduler State Persistence

**Files to modify:**
- `orchestration/task_scheduler.py`
- Create table in `performance.db`

**Add:**
- `_load_state()` - Load from DB on init
- `_save_state()` - Save after each task completion
- Schema: `scheduler_state(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)`

**Test:** Kill orchestrator, restart, verify state restored

### Commit 5.2: Dashboard Integration

**Files to modify:**
- `observability/dashboard/app.py`

**Add components:**
- Scheduler status card (current window, running tasks)
- Task queue visualization
- Estimated vs actual duration chart

**Test:** Load dashboard, verify new components render

### Commit 5.3: Scheduler Metrics Logging

**Files to modify:**
- `orchestration/task_scheduler.py`

**Add logging:**
- Task completion with duration
- Window utilization percentage
- Blocked task reasons

**Test:** Run for full day, review logs for insights

### Commit 5.4: Duration Estimation Tuning

**Files to modify:**
- `orchestration/task_scheduler.py`

**Enhance `_get_estimated_duration()`:**
- Use exponential moving average from `state.avg_durations`
- Add 20% buffer for uncertainty
- Fall back to spec default if no history

**Test:** Run tasks multiple times, verify estimates improve

### Commit 5.5: Feature Flag Removal (Final)

**Files to modify:**
- `config.py` - Remove `USE_TASK_SCHEDULER`
- `daily_orchestrator.py` - Remove conditional

**Test:** Full day operation without flag

---

## Testing Strategy

### Unit Tests

```python
# tests/test_task_scheduler.py

def test_task_ready_phase_check():
    """Task not ready if phase doesn't match."""
    spec = TaskSpec(name="test", phases={"OVERNIGHT"})
    window = TimeWindow(phase="PRE_MARKET", ...)
    assert not scheduler._is_task_ready(spec, window)

def test_task_ready_dependency_check():
    """Task not ready if dependency incomplete."""
    spec = TaskSpec(name="child", dependencies={"parent"})
    scheduler.state.phase_completions["OVERNIGHT"] = set()
    assert not scheduler._is_task_ready(spec, window)

def test_task_ready_conflict_check():
    """Task not ready if conflict running."""
    spec = TaskSpec(name="research", conflicts={"eod_data"})
    scheduler.state.running_tasks = {"eod_data"}
    assert not scheduler._is_task_ready(spec, window)

def test_holiday_detection():
    """Holiday correctly detected from Alpaca API."""
    assert scheduler._is_holiday(date(2026, 1, 19))  # MLK Day
    assert not scheduler._is_holiday(date(2026, 1, 20))  # Regular day

def test_research_window_calculation():
    """Research window accounts for holidays."""
    # Friday evening before MLK weekend
    window = scheduler.calculate_research_window()
    # Should be ~80 hours (Fri eve -> Tue morning)
    assert window.total_seconds() / 3600 > 70
```

### Integration Tests

```bash
# Run orchestrator in test mode
USE_TASK_SCHEDULER=true python daily_orchestrator.py --once

# Verify output shows scheduler decisions
grep "TaskScheduler" logs/orchestrator.log

# Check task execution order respects dependencies
grep "execute" logs/orchestrator.log | head -20
```

### Stress Tests

```bash
# Simulate memory pressure
stress --vm 1 --vm-bytes 3G &
python daily_orchestrator.py --once
# Verify memory-heavy tasks blocked

# Simulate long-running task
# Modify task to sleep(3600)
# Verify scheduler handles gracefully
```

---

## Rollback Plan

If issues discovered:

1. **Immediate:** Set `USE_TASK_SCHEDULER=false` in environment
2. **Short-term:** Revert commits in reverse order
3. **Long-term:** Keep feature flag for toggling

All changes are additive - existing `_task_*` handlers unchanged.

---

## Checklist Before Each Commit

- [ ] No syntax errors (`python -m py_compile orchestration/task_scheduler.py`)
- [ ] Imports work (`python -c "from orchestration.task_scheduler import ..."`)
- [ ] Feature flag respects off state
- [ ] Existing tests pass
- [ ] No breaking changes to `_task_*` handlers
- [ ] Memory footprint reasonable (<50MB for scheduler itself)
- [ ] Logs are informative but not excessive

---

## Files Changed Summary

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 1 | orchestration/task_scheduler.py, orchestration/task_specs.py | config.py, daily_orchestrator.py |
| 2 | - | orchestration/task_scheduler.py, daily_orchestrator.py |
| 3 | - | orchestration/task_scheduler.py, orchestration/task_specs.py, daily_orchestrator.py |
| 4 | - | orchestration/task_scheduler.py, orchestration/task_specs.py, daily_orchestrator.py |
| 5 | tests/test_task_scheduler.py | orchestration/task_scheduler.py, observability/dashboard/app.py, config.py |

---

## Estimated Effort

| Phase | Commits | Est. Lines | Complexity |
|-------|---------|------------|------------|
| 1 | 4 | ~400 | Low |
| 2 | 3 | ~150 | Medium |
| 3 | 4 | ~200 | Medium |
| 4 | 4 | ~150 | Medium |
| 5 | 5 | ~300 | Low |
| **Total** | **20** | **~1200** | - |

---

## Success Criteria

- [ ] Research utilization > 60% on weekends (up from ~19%)
- [ ] Research utilization > 70% on overnight (up from ~38%)
- [ ] Holiday weekends automatically extend research
- [ ] Early close days start research at 3 PM
- [ ] No memory conflicts between heavy tasks
- [ ] Task completion predictability > 85%
- [ ] Zero breaking changes to existing functionality

---

## Phase 6: Unified Scheduler (January 2026)

### Overview

Phase 6 makes TaskScheduler the single authority for market status and operating mode. Replaces rigid `WeekendSubPhase` state machine with dynamic budget-aware task selection.

### Commit 6.1: Add Core Dataclasses

**Files modified:**
- `orchestration/task_scheduler.py`

**Add:**
```python
class OperatingMode(Enum):
    TRADING = "trading"    # Market hours
    RESEARCH = "research"  # Extended window
    PREP = "prep"          # Pre-market/pre-week

@dataclass
class MarketCalendar:
    is_trading_day: bool
    is_early_close: bool
    next_trading_day: date
    hours_until_trading: float

@dataclass
class ResearchBudget:
    total_hours: float
    research_hours: float
    budget_type: str  # overnight|weekend|holiday|holiday_weekend
    is_extended: bool
```

**Test:**
```bash
python -c "from orchestration.task_scheduler import OperatingMode, MarketCalendar, ResearchBudget"
```

### Commit 6.2: Add Unified Scheduler Methods

**Files modified:**
- `orchestration/task_scheduler.py`

**Add methods to TaskScheduler:**
- `get_market_calendar()` - Returns MarketCalendar
- `get_operating_mode()` - Returns OperatingMode based on calendar
- `calculate_research_budget()` - Returns ResearchBudget with actual hours
- `get_extended_window_tasks()` - Budget-aware task selection
- `get_current_mode()` - Dict interface for orchestrator

**Test:**
```python
from orchestration.task_scheduler import TaskScheduler
sched = TaskScheduler(orch)
print(sched.get_operating_mode())
print(sched.calculate_research_budget())
print(sched.get_current_mode())
```

### Commit 6.3: Add USE_UNIFIED_SCHEDULER Flag

**Files modified:**
- `config.py`

**Add:**
```python
USE_UNIFIED_SCHEDULER = os.environ.get('USE_UNIFIED_SCHEDULER', 'false').lower() == 'true'
```

**Test:** Verify flag defaults to False and can be enabled via environment.

### Commit 6.4: Orchestrator Integration

**Files modified:**
- `daily_orchestrator.py`

**Add:**
- Import `USE_UNIFIED_SCHEDULER` and `OperatingMode`
- Add `_get_operating_mode_phase()` method
- Add `_run_unified_extended_window()` method
- Modify weekend handling in `run()` to use unified scheduler when flag enabled

**Test:**
```bash
USE_UNIFIED_SCHEDULER=true python daily_orchestrator.py --once
```

### Commit 6.5: Hardware LED Integration

**Files modified:**
- `hardware/integration.py`

**Add:**
```python
OPERATING_MODE_LED_MAP = {
    'trading': {'system': 'healthy', 'trading': 'active'},
    'research': {'system': 'healthy', 'research': 'evolving'},
    'prep': {'system': 'healthy', 'trading': 'pending'},
}

def set_operating_mode(self, mode: str) -> None:
    """Update LEDs based on operating mode."""
```

**Test:** Verify LEDs update correctly for each mode.

### Commit 6.6: Unit Tests

**Files created:**
- `tests/unit/test_unified_scheduler.py`

**Test cases:**
- OperatingMode enum values
- MarketCalendar creation and is_market_open property
- ResearchBudget creation and budget_pct_remaining
- get_operating_mode() returns correct mode for each phase
- calculate_research_budget() returns correct budget types
- get_extended_window_tasks() budget-aware selection
- get_current_mode() returns complete dict
- USE_UNIFIED_SCHEDULER flag defaults and enablement
- Hardware LED mappings exist

**Test:**
```bash
python -m pytest tests/unit/test_unified_scheduler.py -v
```

### Files Changed Summary (Phase 6)

| Commit | New Files | Modified Files |
|--------|-----------|----------------|
| 6.1 | - | orchestration/task_scheduler.py |
| 6.2 | - | orchestration/task_scheduler.py |
| 6.3 | - | config.py |
| 6.4 | - | daily_orchestrator.py |
| 6.5 | - | hardware/integration.py |
| 6.6 | tests/unit/test_unified_scheduler.py | - |

### Migration Strategy

1. **Week 1**: Deploy with `USE_UNIFIED_SCHEDULER=false`
   - New code present but inactive
   - Verify no regressions

2. **Week 2**: Enable for weekend only
   - Set flag true Friday evening
   - Monitor budget calculations
   - Rollback: `export USE_UNIFIED_SCHEDULER=false`

3. **Week 3**: Enable for overnight
   - Test overnight budget handling

4. **Week 4**: Full enablement
   - Keep legacy code as thin fallback

### Verification Checklist

- [ ] `get_operating_mode()` returns correct mode for all phases
- [ ] `calculate_research_budget()` returns correct hours for:
  - Regular overnight: ~10h
  - Weekend: ~56h
  - Mid-week holiday: ~22h
  - Holiday + weekend: ~80h
- [ ] Integration: Run with `USE_UNIFIED_SCHEDULER=true` during weekend
- [ ] Rollback: Disable flag, verify legacy behavior works
- [ ] Hardware: LEDs update correctly for each operating mode
- [ ] All 51 tests pass (32 unified scheduler + 19 orchestrator)
