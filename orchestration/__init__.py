"""
Orchestration Module
====================
Central coordination for the trading system.

Components:
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

__all__ = [
    # Intervention
    'InterventionManager',
    'InterventionConfig',
    'InterventionMode',
    'InterventionResult',
    'CheckpointPriority',
    # Scheduler
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
