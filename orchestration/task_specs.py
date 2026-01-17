"""
Task Specifications Registry

Complete registry of all 44 orchestrator tasks with Option C specifications.
Each task has:
- Priority (CRITICAL to BACKGROUND)
- Category for resource grouping
- Duration estimates (from actual measurements)
- Memory requirements (based on Pi 5 profiling)
- Phase constraints
- Dependencies and conflicts
- Execution rules (once-per-phase, once-per-day, intervals)

Task durations and memory are based on production profiling on Pi 5 (4GB RAM).
"""

from orchestration.task_scheduler import (
    TaskSpec,
    TaskPriority,
    TaskCategory,
)


# =============================================================================
# TASK SPECIFICATIONS REGISTRY
# =============================================================================

TASK_SPECS = {

    # =========================================================================
    # CRITICAL PRIORITY - Trading Operations (must complete)
    # =========================================================================

    "monitor_positions": TaskSpec(
        name="monitor_positions",
        handler="_task_monitor_positions",
        priority=TaskPriority.CRITICAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=100,
        phases={"market_open", "intraday_active"},
        min_interval_minutes=0.5,
        requires_market_open=True,
    ),

    "check_risk_limits": TaskSpec(
        name="check_risk_limits",
        handler="_task_check_risk_limits",
        priority=TaskPriority.CRITICAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.1,
        max_runtime_minutes=1,
        memory_mb=50,
        phases={"market_open", "intraday_active"},
        min_interval_minutes=0.5,
        requires_market_open=True,
    ),

    "check_rapid_gains": TaskSpec(
        name="check_rapid_gains",
        handler="_task_check_rapid_gains",
        priority=TaskPriority.CRITICAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=100,
        phases={"market_open"},
        min_interval_minutes=5,
        requires_market_open=True,
    ),

    "run_scheduler": TaskSpec(
        name="run_scheduler",
        handler="_task_run_scheduler",
        priority=TaskPriority.CRITICAL,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=300,
        cpu_intensive=True,
        phases={"market_open"},
        run_once_per_phase=True,
        requires_market_open=True,
    ),

    # =========================================================================
    # HIGH PRIORITY - EOD Operations (should complete)
    # =========================================================================

    "reconcile_positions": TaskSpec(
        name="reconcile_positions",
        handler="_task_reconcile_positions",
        priority=TaskPriority.HIGH,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=150,
        phases={"post_market"},
        run_once_per_day=True,
        requires_market_closed=True,
    ),

    "calculate_pnl": TaskSpec(
        name="calculate_pnl",
        handler="_task_calculate_pnl",
        priority=TaskPriority.HIGH,
        category=TaskCategory.TRADING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=100,
        phases={"post_market"},
        dependencies={"reconcile_positions"},
        run_once_per_day=True,
    ),

    "generate_daily_report": TaskSpec(
        name="generate_daily_report",
        handler="_task_generate_daily_report",
        priority=TaskPriority.HIGH,
        category=TaskCategory.MONITORING,
        estimated_minutes=2,
        max_runtime_minutes=10,
        memory_mb=200,
        phases={"post_market"},
        dependencies={"calculate_pnl"},
        run_once_per_day=True,
    ),

    "send_alerts": TaskSpec(
        name="send_alerts",
        handler="_task_send_alerts",
        priority=TaskPriority.HIGH,
        category=TaskCategory.MONITORING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=50,
        phases={"post_market"},
        dependencies={"generate_daily_report"},
        run_once_per_day=True,
    ),

    "update_ensemble_correlations": TaskSpec(
        name="update_ensemble_correlations",
        handler="_task_update_ensemble_correlations",
        priority=TaskPriority.HIGH,
        category=TaskCategory.RESEARCH,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=200,
        phases={"post_market"},
        run_once_per_day=True,
    ),

    "update_paper_metrics": TaskSpec(
        name="update_paper_metrics",
        handler="_task_update_paper_metrics",
        priority=TaskPriority.HIGH,
        category=TaskCategory.RESEARCH,
        estimated_minutes=2,
        max_runtime_minutes=5,
        memory_mb=150,
        phases={"post_market"},
        run_once_per_day=True,
    ),

    "update_live_metrics": TaskSpec(
        name="update_live_metrics",
        handler="_task_update_live_metrics",
        priority=TaskPriority.HIGH,
        category=TaskCategory.RESEARCH,
        estimated_minutes=2,
        max_runtime_minutes=5,
        memory_mb=150,
        phases={"post_market"},
        run_once_per_day=True,
    ),

    "run_promotion_pipeline": TaskSpec(
        name="run_promotion_pipeline",
        handler="_task_run_promotion_pipeline",
        priority=TaskPriority.HIGH,
        category=TaskCategory.RESEARCH,
        estimated_minutes=3,
        max_runtime_minutes=10,
        memory_mb=200,
        phases={"post_market"},
        run_once_per_day=True,
    ),

    # =========================================================================
    # NORMAL PRIORITY - Pre-market & Data Operations
    # =========================================================================

    "stop_research_processes": TaskSpec(
        name="stop_research_processes",
        handler="_task_stop_research_processes",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=0.2,
        max_runtime_minutes=1,
        memory_mb=50,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "refresh_premarket_data": TaskSpec(
        name="refresh_premarket_data",
        handler="_task_refresh_premarket_data",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=2,
        max_runtime_minutes=5,
        memory_mb=300,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "refresh_intraday_data": TaskSpec(
        name="refresh_intraday_data",
        handler="_task_refresh_intraday_data",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=1,
        max_runtime_minutes=3,
        memory_mb=200,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "refresh_data": TaskSpec(
        name="refresh_data",
        handler="_task_refresh_data",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=300,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "system_check": TaskSpec(
        name="system_check",
        handler="_task_system_check",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MONITORING,
        estimated_minutes=0.2,
        max_runtime_minutes=1,
        memory_mb=50,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "sync_positions_from_broker": TaskSpec(
        name="sync_positions_from_broker",
        handler="_task_sync_positions_from_broker",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=100,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "review_positions": TaskSpec(
        name="review_positions",
        handler="_task_review_positions",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.2,
        max_runtime_minutes=1,
        memory_mb=50,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "cancel_stale_orders": TaskSpec(
        name="cancel_stale_orders",
        handler="_task_cancel_stale_orders",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.2,
        max_runtime_minutes=1,
        memory_mb=50,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "update_regime_detection": TaskSpec(
        name="update_regime_detection",
        handler="_task_update_regime_detection",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.RESEARCH,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=300,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "calculate_position_scalars": TaskSpec(
        name="calculate_position_scalars",
        handler="_task_calculate_position_scalars",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=3,
        memory_mb=200,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    "load_live_strategies": TaskSpec(
        name="load_live_strategies",
        handler="_task_load_live_strategies",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=3,
        memory_mb=300,
        phases={"pre_market"},
        run_once_per_phase=True,
    ),

    # =========================================================================
    # NORMAL PRIORITY - Intraday Operations
    # =========================================================================

    "start_intraday_stream": TaskSpec(
        name="start_intraday_stream",
        handler="_task_start_intraday_stream",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=200,
        phases={"intraday_open"},
        run_once_per_phase=True,
        requires_market_open=True,
    ),

    "detect_gaps": TaskSpec(
        name="detect_gaps",
        handler="_task_detect_gaps",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=3,
        memory_mb=200,
        phases={"intraday_open"},
        run_once_per_phase=True,
        requires_market_open=True,
    ),

    "monitor_intraday_positions": TaskSpec(
        name="monitor_intraday_positions",
        handler="_task_monitor_intraday_positions",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=0.3,
        max_runtime_minutes=1,
        memory_mb=100,
        phases={"intraday_active"},
        min_interval_minutes=0.15,  # Every 10 seconds
        requires_market_open=True,
    ),

    "stop_intraday_stream": TaskSpec(
        name="stop_intraday_stream",
        handler="_task_stop_intraday_stream",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=0.2,
        max_runtime_minutes=1,
        memory_mb=50,
        # Triggered by phase transition, not specific phase
        run_once_per_phase=True,
    ),

    # =========================================================================
    # NORMAL PRIORITY - Signal Processing
    # =========================================================================

    "score_pending_signals": TaskSpec(
        name="score_pending_signals",
        handler="_task_score_pending_signals",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=200,
        phases={"market_open"},
        min_interval_minutes=0.5,
        requires_market_open=True,
    ),

    "process_shadow_trades": TaskSpec(
        name="process_shadow_trades",
        handler="_task_process_shadow_trades",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.TRADING,
        estimated_minutes=1,
        max_runtime_minutes=5,
        memory_mb=150,
        phases={"market_open"},
        min_interval_minutes=5,
        requires_market_open=True,
    ),

    # =========================================================================
    # NORMAL PRIORITY - Evening/EOD Data Operations
    # =========================================================================

    "refresh_eod_data": TaskSpec(
        name="refresh_eod_data",
        handler="_task_refresh_eod_data",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=30,
        max_runtime_minutes=60,
        memory_mb=400,
        cpu_intensive=True,
        phases={"evening", "weekend"},
        run_once_per_day=True,
        conflicts={"run_nightly_research", "run_weekend_research"},
    ),

    "cleanup_logs": TaskSpec(
        name="cleanup_logs",
        handler="_task_cleanup_logs",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=50,
        phases={"evening", "weekend"},
        run_once_per_day=True,
    ),

    "backup_databases": TaskSpec(
        name="backup_databases",
        handler="_task_backup_databases",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=2,
        max_runtime_minutes=10,
        memory_mb=100,
        phases={"evening", "weekend"},
        run_once_per_day=True,
    ),

    "cleanup_databases": TaskSpec(
        name="cleanup_databases",
        handler="_task_cleanup_databases",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=5,
        max_runtime_minutes=30,
        memory_mb=150,
        phases={"evening", "weekend"},
        run_once_per_day=True,
    ),

    # =========================================================================
    # NORMAL PRIORITY - Core Research
    # =========================================================================

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
        phases={"overnight"},
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
        phases={"weekend"},
        conflicts={"refresh_eod_data"},
    ),

    "run_weekend_schedule": TaskSpec(
        name="run_weekend_schedule",
        handler="_task_run_weekend_schedule",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=5,  # Dispatcher is quick
        max_runtime_minutes=30,
        memory_mb=100,
        phases={"weekend"},
        # This is the master dispatcher, runs repeatedly
        min_interval_minutes=5,
    ),

    # =========================================================================
    # NORMAL PRIORITY - Weekend Data Operations
    # =========================================================================

    "refresh_index_constituents": TaskSpec(
        name="refresh_index_constituents",
        handler="_task_refresh_index_constituents",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=5,
        max_runtime_minutes=15,
        memory_mb=100,
        phases={"weekend"},
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
        phases={"weekend"},
        run_once_per_day=True,
    ),

    "refresh_pairs_correlations": TaskSpec(
        name="refresh_pairs_correlations",
        handler="_task_refresh_pairs_correlations",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.DATA,
        estimated_minutes=30,
        max_runtime_minutes=60,
        memory_mb=500,
        can_interrupt=True,
        phases={"weekend"},
        run_once_per_day=True,
        # DISABLED - too heavy for Pi
    ),

    "generate_weekly_report": TaskSpec(
        name="generate_weekly_report",
        handler="_task_generate_weekly_report",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MONITORING,
        estimated_minutes=2,
        max_runtime_minutes=10,
        memory_mb=200,
        phases={"weekend"},
        run_once_per_day=True,
    ),

    "vacuum_databases": TaskSpec(
        name="vacuum_databases",
        handler="_task_vacuum_databases",
        priority=TaskPriority.NORMAL,
        category=TaskCategory.MAINTENANCE,
        estimated_minutes=5,
        max_runtime_minutes=30,
        memory_mb=150,
        phases={"weekend"},
        min_interval_minutes=7 * 24 * 60,  # Weekly
    ),

    # =========================================================================
    # LOW PRIORITY - Opportunistic Tasks
    # =========================================================================

    "validate_candidates": TaskSpec(
        name="validate_candidates",
        handler="_task_validate_candidates",
        priority=TaskPriority.LOW,
        category=TaskCategory.RESEARCH,
        estimated_minutes=30,
        max_runtime_minutes=60,
        memory_mb=1500,
        can_interrupt=True,
        phases={"overnight", "weekend"},
        dependencies={"run_nightly_research"},  # After core research
        conflicts={"run_nightly_research", "run_weekend_research"},
    ),

    "train_ml_regime_model": TaskSpec(
        name="train_ml_regime_model",
        handler="_task_train_ml_regime_model",
        priority=TaskPriority.LOW,
        category=TaskCategory.RESEARCH,
        estimated_minutes=15,
        max_runtime_minutes=60,
        memory_mb=500,
        phases={"overnight", "weekend"},
        min_interval_minutes=7 * 24 * 60,  # Weekly
    ),

    "validate_strategies": TaskSpec(
        name="validate_strategies",
        handler="_task_validate_strategies",
        priority=TaskPriority.LOW,
        category=TaskCategory.RESEARCH,
        estimated_minutes=2,
        max_runtime_minutes=10,
        memory_mb=300,
        phases={"weekend"},
        run_once_per_day=True,
    ),

    "verify_system_readiness": TaskSpec(
        name="verify_system_readiness",
        handler="_task_verify_system_readiness",
        priority=TaskPriority.LOW,
        category=TaskCategory.MONITORING,
        estimated_minutes=0.5,
        max_runtime_minutes=2,
        memory_mb=50,
        phases={"weekend"},
        run_once_per_day=True,
    ),

}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tasks_by_priority(priority: TaskPriority) -> list:
    """Get all tasks with a specific priority."""
    return [spec for spec in TASK_SPECS.values() if spec.priority == priority]


def get_tasks_by_category(category: TaskCategory) -> list:
    """Get all tasks in a specific category."""
    return [spec for spec in TASK_SPECS.values() if spec.category == category]


def get_tasks_for_phase(phase: str) -> list:
    """Get all tasks valid for a specific phase."""
    return [
        spec for spec in TASK_SPECS.values()
        if not spec.phases or phase in spec.phases
    ]


def get_memory_heavy_tasks(threshold_mb: int = 500) -> list:
    """Get tasks that use more than threshold MB of memory."""
    return [spec for spec in TASK_SPECS.values() if spec.memory_mb >= threshold_mb]


def get_cpu_intensive_tasks() -> list:
    """Get all CPU-intensive tasks."""
    return [spec for spec in TASK_SPECS.values() if spec.cpu_intensive]


def get_interruptible_tasks() -> list:
    """Get tasks that can be safely interrupted."""
    return [spec for spec in TASK_SPECS.values() if spec.can_interrupt]


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

if __name__ == "__main__":
    # Print summary when run directly
    print(f"Total tasks registered: {len(TASK_SPECS)}")
    print()

    for priority in TaskPriority:
        tasks = get_tasks_by_priority(priority)
        print(f"{priority.name}: {len(tasks)} tasks")
        for task in tasks:
            print(f"  - {task.name} ({task.estimated_minutes:.1f}min, {task.memory_mb}MB)")
        print()

    memory_heavy = get_memory_heavy_tasks()
    print(f"Memory-heavy tasks (>=500MB): {len(memory_heavy)}")
    for task in memory_heavy:
        print(f"  - {task.name}: {task.memory_mb}MB")
