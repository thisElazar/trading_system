"""
Orchestration Module
====================
Central coordination for the trading system.

Components:
- TaskScheduler: Time-aware task scheduling with priorities and dependencies
- InterventionManager: Human oversight and approval system
- StrategyScheduler: Strategy execution scheduling
- ExecutionTracker: Signal and position tracking
- AlertManager: Notification system
"""

from execution.scheduler import (
    StrategyScheduler,
    MarketHours,
    create_default_scheduler
)
from execution.signal_tracker import (
    SignalDatabase,
    ExecutionTracker,
    StoredSignal,
    Position
)
from execution.alerts import (
    AlertManager,
    create_alert_manager,
    get_alerts
)
from orchestration.intervention import (
    InterventionManager,
    InterventionConfig,
    InterventionMode,
    InterventionResult,
    CheckpointPriority,
)
from orchestration.task_scheduler import (
    TaskScheduler,
    TaskSpec,
    TaskResult,
    TaskPriority,
    TaskCategory,
    TimeWindow,
    SchedulerState,
)
from orchestration.task_specs import TASK_SPECS

__all__ = [
    # Task Scheduler (Option C)
    'TaskScheduler',
    'TaskSpec',
    'TaskResult',
    'TaskPriority',
    'TaskCategory',
    'TimeWindow',
    'SchedulerState',
    'TASK_SPECS',
    # Intervention
    'InterventionManager',
    'InterventionConfig',
    'InterventionMode',
    'InterventionResult',
    'CheckpointPriority',
    # Strategy Scheduler
    'StrategyScheduler',
    'MarketHours',
    'create_default_scheduler',
    # Tracking
    'SignalDatabase',
    'ExecutionTracker',
    'StoredSignal',
    'Position',
    # Alerts
    'AlertManager',
    'create_alert_manager',
    'get_alerts'
]
