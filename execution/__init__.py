"""
Execution Module
================
Signal tracking, scheduling, alerts, order execution, strategy coordination,
and execution quality tracking.
"""

from execution.signal_tracker import (
    SignalDatabase,
    ExecutionTracker,
    StoredSignal,
    Execution,
    Position,
    SignalStatus,
    PositionStatus
)
from execution.scheduler import (
    StrategyScheduler,
    MarketHours,
    create_default_scheduler
)
from execution.alerts import (
    AlertManager,
    AlertLevel,
    AlertType,
    Alert,
    create_alert_manager,
    get_alerts
)
from execution.ensemble import (
    StrategyEnsemble,
    CapitalAllocator,
    ConflictResolver,
    AllocationMethod,
    AggregatedSignal,
    create_ensemble
)
from execution.position_sizer import (
    PositionSizer,
    PositionSize,
    calculate_position_size
)
from execution.adaptive_position_sizer import (
    AdaptivePositionSizer,
    AdaptivePositionResult,
    SizingContext,
    ADAPTIVE_SIZING,
    create_adaptive_sizer,
    calculate_adaptive_position
)
from execution.execution_quality import (
    ExecutionQualityTracker,
    ExecutionRecord,
    ExecutionReport,
    SlippageStats,
    ExecutionAlert,
    AlertSeverity,
    get_execution_tracker,
    create_tracker
)

# Alpaca connector (optional - requires alpaca-py)
try:
    from execution.alpaca_connector import (
        AlpacaConnector,
        BrokerPosition,
        BrokerOrder,
        AccountInfo,
        create_connector
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

__all__ = [
    # Signal tracking
    'SignalDatabase',
    'ExecutionTracker',
    'StoredSignal',
    'Execution',
    'Position',
    'SignalStatus',
    'PositionStatus',
    # Scheduling
    'StrategyScheduler',
    'MarketHours',
    'create_default_scheduler',
    # Alerts
    'AlertManager',
    'AlertLevel',
    'AlertType',
    'Alert',
    'create_alert_manager',
    'get_alerts',
    # Ensemble
    'StrategyEnsemble',
    'CapitalAllocator',
    'ConflictResolver',
    'AllocationMethod',
    'AggregatedSignal',
    'create_ensemble',
    # Position sizing
    'PositionSizer',
    'PositionSize',
    'calculate_position_size',
    # Adaptive position sizing
    'AdaptivePositionSizer',
    'AdaptivePositionResult',
    'SizingContext',
    'ADAPTIVE_SIZING',
    'create_adaptive_sizer',
    'calculate_adaptive_position',
    # Execution quality tracking
    'ExecutionQualityTracker',
    'ExecutionRecord',
    'ExecutionReport',
    'SlippageStats',
    'ExecutionAlert',
    'AlertSeverity',
    'get_execution_tracker',
    'create_tracker',
    # Alpaca
    'ALPACA_AVAILABLE',
]

if ALPACA_AVAILABLE:
    __all__.extend([
        'AlpacaConnector',
        'BrokerPosition',
        'BrokerOrder',
        'AccountInfo',
        'create_connector',
    ])
