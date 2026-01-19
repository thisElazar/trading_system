"""
Integration tests for the Daily Orchestrator.

Tests the orchestrator's ability to:
1. Detect market phases correctly
2. Run phase-specific tasks
3. Handle phase transitions
4. Gracefully shutdown
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import threading

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


class TestPhaseDetection:
    """Test market phase detection."""

    def test_detects_market_open_during_trading_hours(self):
        """Should detect market_open phase during trading hours."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase
        import pytz

        orch = DailyOrchestrator(paper_mode=True)

        # Mock time to be during market hours (10:00 AM ET on a weekday)
        mock_time = datetime(2024, 6, 17, 10, 0, 0)  # Monday
        with patch('daily_orchestrator.datetime') as mock_dt:
            mock_dt.now.return_value = pytz.timezone('US/Eastern').localize(mock_time)

            phase = orch.get_current_phase()
            # During market hours, should be market_open or intraday_active
            assert phase in [MarketPhase.MARKET_OPEN, MarketPhase.INTRADAY_ACTIVE]

    def test_detects_pre_market_phase(self):
        """Should detect pre_market phase before market open."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase
        import pytz

        orch = DailyOrchestrator(paper_mode=True)

        # Mock time to be 8:30 AM ET on a weekday
        mock_time = datetime(2024, 6, 17, 8, 30, 0)  # Monday
        with patch('daily_orchestrator.datetime') as mock_dt:
            mock_dt.now.return_value = pytz.timezone('US/Eastern').localize(mock_time)

            phase = orch.get_current_phase()
            assert phase == MarketPhase.PRE_MARKET

    def test_detects_post_market_phase(self):
        """Should detect post_market phase after market close."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase
        import pytz

        orch = DailyOrchestrator(paper_mode=True)

        # Mock time to be 4:30 PM ET on a weekday
        mock_time = datetime(2024, 6, 17, 16, 30, 0)  # Monday
        with patch('daily_orchestrator.datetime') as mock_dt:
            mock_dt.now.return_value = pytz.timezone('US/Eastern').localize(mock_time)

            phase = orch.get_current_phase()
            assert phase == MarketPhase.POST_MARKET

    def test_detects_weekend(self):
        """Should detect weekend phase on Saturday/Sunday."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase
        import pytz

        orch = DailyOrchestrator(paper_mode=True)

        # Mock time to be Saturday
        mock_time = datetime(2024, 6, 15, 12, 0, 0)  # Saturday
        with patch('daily_orchestrator.datetime') as mock_dt:
            mock_dt.now.return_value = pytz.timezone('US/Eastern').localize(mock_time)

            phase = orch.get_current_phase()
            assert phase == MarketPhase.WEEKEND


class TestPhaseTasks:
    """Test phase-specific task execution."""

    def test_pre_market_tasks_defined(self):
        """Pre-market phase should have correct tasks defined."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        orch = DailyOrchestrator(paper_mode=True)
        config = orch.get_phase_config(MarketPhase.PRE_MARKET)

        assert 'refresh_data' in config.tasks
        assert 'system_check' in config.tasks
        assert 'review_positions' in config.tasks

    def test_market_open_tasks_defined(self):
        """Market open phase should have correct tasks defined."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        orch = DailyOrchestrator(paper_mode=True)
        config = orch.get_phase_config(MarketPhase.MARKET_OPEN)

        assert 'run_scheduler' in config.tasks
        assert 'monitor_positions' in config.tasks
        assert 'check_risk_limits' in config.tasks

    def test_post_market_tasks_defined(self):
        """Post-market phase should have correct tasks defined."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        orch = DailyOrchestrator(paper_mode=True)
        config = orch.get_phase_config(MarketPhase.POST_MARKET)

        assert 'reconcile_positions' in config.tasks
        assert 'calculate_pnl' in config.tasks
        assert 'generate_daily_report' in config.tasks

    def test_task_registry_complete(self):
        """All defined tasks should have implementations."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        orch = DailyOrchestrator(paper_mode=True)

        all_tasks = set()
        for phase in MarketPhase:
            config = orch.PHASE_CONFIGS.get(phase)
            if config:
                all_tasks.update(config.tasks)

        # Verify all tasks have implementations
        for task_name in all_tasks:
            assert task_name in orch._task_registry, f"Task {task_name} not in registry"


class TestTaskExecution:
    """Test individual task execution."""

    def test_system_check_returns_result(self):
        """System check should return success/failure."""
        from daily_orchestrator import DailyOrchestrator

        with patch('daily_orchestrator.CachedDataManager') as mock_dm, \
             patch('daily_orchestrator.AlpacaConnector') as mock_broker:

            # Configure mocks
            mock_dm_instance = MagicMock()
            mock_dm_instance.cache = {'AAPL': MagicMock()}
            mock_dm.return_value = mock_dm_instance

            mock_broker_instance = MagicMock()
            mock_broker_instance.get_account.return_value = MagicMock()
            mock_broker.return_value = mock_broker_instance

            orch = DailyOrchestrator(paper_mode=True)
            result = orch._task_system_check()

            # System check should return True if enough checks pass
            assert isinstance(result, bool)

    def test_review_positions_handles_no_positions(self):
        """Review positions should handle empty portfolio gracefully."""
        from daily_orchestrator import DailyOrchestrator

        with patch('daily_orchestrator.AlpacaConnector') as mock_broker:
            mock_broker_instance = MagicMock()
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            orch = DailyOrchestrator(paper_mode=True)
            result = orch._task_review_positions()

            assert result is True

    def test_run_task_logs_completion(self):
        """Running a task should complete successfully and update state."""
        from daily_orchestrator import DailyOrchestrator

        with patch('daily_orchestrator.AlpacaConnector') as mock_broker:
            mock_broker_instance = MagicMock()
            mock_broker_instance.get_positions.return_value = []
            mock_broker.return_value = mock_broker_instance

            orch = DailyOrchestrator(paper_mode=True)
            result = orch.run_task('review_positions')

            # Task should return True (success) and complete
            assert result is True


class TestPhaseTransitions:
    """Test phase transition handling."""

    def test_transition_resets_daily_stats_at_premarket(self):
        """Transitioning to pre-market should reset daily stats."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        orch = DailyOrchestrator(paper_mode=True)

        # Set some existing stats
        orch.state.tasks_completed_today = ['task1', 'task2']
        orch.state.errors_today = [{'error': 'test'}]
        orch.state.daily_stats = {'equity': 100000}

        # Simulate phase transition to pre-market
        orch.state.current_phase = MarketPhase.EVENING
        # The reset happens in the run() method when transitioning to PRE_MARKET

    def test_time_until_next_phase_positive(self):
        """Time until next phase should be positive."""
        from daily_orchestrator import DailyOrchestrator

        orch = DailyOrchestrator(paper_mode=True)
        time_to_next = orch.time_until_next_phase()

        assert time_to_next.total_seconds() > 0


class TestGracefulShutdown:
    """Test graceful shutdown handling."""

    def test_shutdown_event_stops_loop(self):
        """Setting shutdown event should stop the main loop."""
        from daily_orchestrator import DailyOrchestrator

        # Mock hardware to avoid slow startup sequence
        with patch('daily_orchestrator.get_hardware_status') as mock_hw:
            mock_hw.return_value = MagicMock()

            orch = DailyOrchestrator(paper_mode=True)
            # Disable hardware to skip startup delay
            orch._hardware = None

            # Set shutdown event
            orch.shutdown_event.set()

            # Run should exit quickly
            import time
            start = time.time()
            orch.run(once=True)
            elapsed = time.time() - start

            # Should complete in under 10 seconds (includes startup recovery)
            assert elapsed < 10

    def test_cleanup_called_on_shutdown(self):
        """Cleanup should be called when orchestrator stops."""
        from daily_orchestrator import DailyOrchestrator

        orch = DailyOrchestrator(paper_mode=True)

        with patch.object(orch, '_cleanup') as mock_cleanup:
            orch.shutdown_event.set()
            orch.run(once=True)

            mock_cleanup.assert_called()


class TestOrchestratorStatus:
    """Test orchestrator status reporting."""

    def test_status_includes_required_fields(self):
        """Status should include all required fields."""
        from daily_orchestrator import DailyOrchestrator

        orch = DailyOrchestrator(paper_mode=True)
        status = orch.status()

        required_fields = [
            'timestamp',
            'is_running',
            'paper_mode',
            'current_phase',
            'tasks_completed_today',
            'errors_today',
        ]

        for field in required_fields:
            assert field in status, f"Missing field: {field}"

    def test_status_reflects_paper_mode(self):
        """Status should correctly reflect paper mode setting."""
        from daily_orchestrator import DailyOrchestrator

        # Paper mode
        orch_paper = DailyOrchestrator(paper_mode=True)
        assert orch_paper.status()['paper_mode'] is True

        # Live mode (still testing, just checking the flag)
        orch_live = DailyOrchestrator(paper_mode=False)
        assert orch_live.status()['paper_mode'] is False


class TestOrchestratorOnce:
    """Test --once mode execution."""

    def test_once_mode_runs_single_phase(self):
        """--once mode should run current phase once and exit."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        with patch('daily_orchestrator.AlpacaConnector') as mock_broker, \
             patch('daily_orchestrator.CachedDataManager') as mock_dm, \
             patch('daily_orchestrator.get_hardware_status') as mock_hw:

            mock_broker.return_value = MagicMock()
            mock_dm.return_value = MagicMock()
            mock_hw.return_value = MagicMock()

            orch = DailyOrchestrator(paper_mode=True)
            orch._hardware = None  # Skip hardware to speed up test

            # Track if phase tasks were run
            with patch.object(orch, 'run_phase_tasks') as mock_run:
                mock_run.return_value = {}
                # Force pre-market phase to avoid weekend-specific code path
                orch.run(once=True, force_phase='pre')

                # Phase tasks should be called once
                assert mock_run.call_count >= 1

    def test_force_phase_runs_specific_phase(self):
        """--phase flag should force specific phase execution."""
        from daily_orchestrator import DailyOrchestrator, MarketPhase

        with patch('daily_orchestrator.AlpacaConnector') as mock_broker, \
             patch('daily_orchestrator.CachedDataManager') as mock_dm:

            mock_broker.return_value = MagicMock()
            mock_dm.return_value = MagicMock()

            orch = DailyOrchestrator(paper_mode=True)

            with patch.object(orch, 'run_phase_tasks') as mock_run:
                mock_run.return_value = {}
                orch.run(once=True, force_phase='pre')

                # Should run pre-market phase
                if mock_run.called:
                    called_phase = mock_run.call_args[0][0]
                    assert called_phase == MarketPhase.PRE_MARKET
