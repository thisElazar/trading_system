"""
Unit tests for the Unified TaskScheduler.

Tests the new market-aware scheduling system:
- OperatingMode enum
- MarketCalendar dataclass
- ResearchBudget dataclass
- TaskScheduler methods for unified scheduling
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
import pytz


class TestOperatingMode:
    """Test OperatingMode enum."""

    def test_operating_mode_values(self):
        """OperatingMode enum should have expected values."""
        from orchestration.task_scheduler import OperatingMode

        assert OperatingMode.TRADING.value == "trading"
        assert OperatingMode.RESEARCH.value == "research"
        assert OperatingMode.PREP.value == "prep"

    def test_operating_mode_iteration(self):
        """Should be able to iterate over all modes."""
        from orchestration.task_scheduler import OperatingMode

        modes = list(OperatingMode)
        assert len(modes) == 3
        assert OperatingMode.TRADING in modes
        assert OperatingMode.RESEARCH in modes
        assert OperatingMode.PREP in modes


class TestMarketCalendar:
    """Test MarketCalendar dataclass."""

    def test_market_calendar_creation(self):
        """MarketCalendar should be creatable with all fields."""
        from orchestration.task_scheduler import MarketCalendar

        calendar = MarketCalendar(
            is_trading_day=True,
            is_early_close=False,
            next_trading_day=date(2026, 1, 20),
            hours_until_trading=2.5,
        )

        assert calendar.is_trading_day is True
        assert calendar.is_early_close is False
        assert calendar.next_trading_day == date(2026, 1, 20)
        assert calendar.hours_until_trading == 2.5

    def test_is_market_open_property(self):
        """is_market_open should return True when trading day and hours=0."""
        from orchestration.task_scheduler import MarketCalendar

        # Market is open
        calendar_open = MarketCalendar(
            is_trading_day=True,
            is_early_close=False,
            next_trading_day=date(2026, 1, 20),
            hours_until_trading=0,
        )
        assert calendar_open.is_market_open is True

        # Market not open - hours until trading > 0
        calendar_closed = MarketCalendar(
            is_trading_day=True,
            is_early_close=False,
            next_trading_day=date(2026, 1, 20),
            hours_until_trading=2.5,
        )
        assert calendar_closed.is_market_open is False

        # Market not open - not a trading day
        calendar_weekend = MarketCalendar(
            is_trading_day=False,
            is_early_close=False,
            next_trading_day=date(2026, 1, 20),
            hours_until_trading=0,
        )
        assert calendar_weekend.is_market_open is False


class TestResearchBudget:
    """Test ResearchBudget dataclass."""

    def test_research_budget_creation(self):
        """ResearchBudget should be creatable with all fields."""
        from orchestration.task_scheduler import ResearchBudget

        budget = ResearchBudget(
            total_hours=56.0,
            research_hours=52.5,
            budget_type="weekend",
            is_extended=True,
        )

        assert budget.total_hours == 56.0
        assert budget.research_hours == 52.5
        assert budget.budget_type == "weekend"
        assert budget.is_extended is True

    def test_hours_remaining_alias(self):
        """hours_remaining should be alias for research_hours."""
        from orchestration.task_scheduler import ResearchBudget

        budget = ResearchBudget(
            total_hours=56.0,
            research_hours=42.0,
            budget_type="weekend",
            is_extended=True,
        )

        assert budget.hours_remaining == budget.research_hours == 42.0

    def test_budget_pct_remaining(self):
        """budget_pct_remaining should return correct percentage."""
        from orchestration.task_scheduler import ResearchBudget

        # 50% remaining
        budget = ResearchBudget(
            total_hours=56.0,
            research_hours=28.0,
            budget_type="weekend",
            is_extended=True,
        )
        assert abs(budget.budget_pct_remaining - 0.5) < 0.01

        # 0% remaining
        budget_empty = ResearchBudget(
            total_hours=56.0,
            research_hours=0.0,
            budget_type="weekend",
            is_extended=True,
        )
        assert budget_empty.budget_pct_remaining == 0.0

        # Handle edge case: total_hours = 0
        budget_zero = ResearchBudget(
            total_hours=0.0,
            research_hours=0.0,
            budget_type="overnight",
            is_extended=False,
        )
        assert budget_zero.budget_pct_remaining == 0.0

    def test_overnight_budget(self):
        """Overnight budget should have ~10h and not be extended."""
        from orchestration.task_scheduler import ResearchBudget

        budget = ResearchBudget(
            total_hours=10.5,
            research_hours=9.0,
            budget_type="overnight",
            is_extended=False,
        )

        assert budget.budget_type == "overnight"
        assert budget.is_extended is False
        assert budget.total_hours < 12

    def test_holiday_budget(self):
        """Holiday budget should reflect ~22h window."""
        from orchestration.task_scheduler import ResearchBudget

        # Mid-week holiday (key insight: NOT 56 hours like weekend)
        budget = ResearchBudget(
            total_hours=22.0,
            research_hours=20.0,
            budget_type="holiday",
            is_extended=True,
        )

        assert budget.budget_type == "holiday"
        assert budget.is_extended is True
        assert budget.total_hours < 30  # Not a full weekend


class TestGetOperatingMode:
    """Test TaskScheduler.get_operating_mode() method."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        mock = MagicMock()
        mock.tz = pytz.timezone('US/Eastern')
        return mock

    def test_pre_market_returns_prep(self, mock_orchestrator):
        """Pre-market phase should return PREP mode."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.PRE_MARKET

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)
            scheduler._holiday_cache = {}
            scheduler._early_close_cache = {}

            # Mock get_market_calendar to avoid API calls
            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=1.0)
                mode = scheduler.get_operating_mode()

        assert mode == OperatingMode.PREP

    def test_market_open_returns_trading(self, mock_orchestrator):
        """Market open phase should return TRADING mode."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.MARKET_OPEN

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=0)
                mode = scheduler.get_operating_mode()

        assert mode == OperatingMode.TRADING

    def test_intraday_phases_return_trading(self, mock_orchestrator):
        """Intraday phases should return TRADING mode."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=0)

                for phase in [MarketPhase.INTRADAY_OPEN, MarketPhase.INTRADAY_ACTIVE]:
                    mock_orchestrator.get_current_phase.return_value = phase
                    mode = scheduler.get_operating_mode()
                    assert mode == OperatingMode.TRADING, f"Expected TRADING for {phase}"

    def test_overnight_returns_research(self, mock_orchestrator):
        """Overnight phase should return RESEARCH mode (if enough time)."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.OVERNIGHT

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            # Plenty of time until trading
            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=8.0)
                mode = scheduler.get_operating_mode()

        assert mode == OperatingMode.RESEARCH

    def test_overnight_near_premarket_returns_prep(self, mock_orchestrator):
        """Overnight phase close to pre-market should return PREP mode."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.OVERNIGHT

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            # Less than 1.5 hours until trading
            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=1.0)
                mode = scheduler.get_operating_mode()

        assert mode == OperatingMode.PREP

    def test_weekend_saturday_returns_research(self, mock_orchestrator):
        """Weekend (Saturday) should return RESEARCH mode."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.WEEKEND

        # Saturday at noon
        saturday = datetime(2026, 1, 17, 12, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'), \
             patch('orchestration.task_scheduler.datetime') as mock_dt:
            mock_dt.now.return_value = saturday

            scheduler = TaskScheduler(mock_orchestrator)

            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=40.0)
                mode = scheduler.get_operating_mode()

        assert mode == OperatingMode.RESEARCH

    def test_weekend_sunday_afternoon_returns_prep(self, mock_orchestrator):
        """Weekend (Sunday afternoon) should return PREP mode."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.WEEKEND

        # Sunday at 3 PM (14:00+)
        sunday_afternoon = datetime(2026, 1, 18, 15, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'), \
             patch('orchestration.task_scheduler.datetime') as mock_dt:
            mock_dt.now.return_value = sunday_afternoon

            scheduler = TaskScheduler(mock_orchestrator)

            with patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=18.0)
                mode = scheduler.get_operating_mode()

        assert mode == OperatingMode.PREP


class TestCalculateResearchBudget:
    """Test TaskScheduler.calculate_research_budget() method."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        mock = MagicMock()
        mock.tz = pytz.timezone('US/Eastern')
        return mock

    def test_weekend_budget_is_extended(self, mock_orchestrator):
        """Weekend budget should be marked as extended (>12h)."""
        from orchestration.task_scheduler import TaskScheduler
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.WEEKEND

        # Saturday morning
        saturday = datetime(2026, 1, 17, 10, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'), \
             patch('orchestration.task_scheduler.datetime') as mock_dt:
            mock_dt.now.return_value = saturday

            scheduler = TaskScheduler(mock_orchestrator)
            scheduler._holiday_cache = {}
            scheduler._early_close_cache = {}

            # Mock the trading day calculation
            monday = datetime(2026, 1, 19, 8, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
            with patch.object(scheduler, '_get_next_trading_premarket', return_value=monday), \
                 patch.object(scheduler, '_is_holiday', return_value=False), \
                 patch.object(scheduler, 'get_market_calendar') as mock_cal:
                mock_cal.return_value = MagicMock(hours_until_trading=45.0)

                budget = scheduler.calculate_research_budget()

        assert budget.is_extended is True
        assert budget.budget_type == "weekend"
        assert budget.total_hours > 12

    def test_overnight_budget_not_extended(self, mock_orchestrator):
        """Overnight budget should not be marked as extended (<12h)."""
        from orchestration.task_scheduler import TaskScheduler
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.OVERNIGHT

        # Tuesday night 10 PM
        tuesday_night = datetime(2026, 1, 20, 22, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'), \
             patch('orchestration.task_scheduler.datetime') as mock_dt:
            mock_dt.now.return_value = tuesday_night

            scheduler = TaskScheduler(mock_orchestrator)
            scheduler._holiday_cache = {}
            scheduler._early_close_cache = {}

            # Next day 9:30 AM
            next_morning = datetime(2026, 1, 21, 9, 30, 0, tzinfo=pytz.timezone('US/Eastern'))
            with patch.object(scheduler, '_get_next_trading_premarket') as mock_next:
                mock_next.return_value = next_morning.replace(hour=8)
                with patch.object(scheduler, '_is_holiday', return_value=False):
                    budget = scheduler.calculate_research_budget()

        assert budget.is_extended is False
        assert budget.budget_type == "overnight"
        assert budget.total_hours < 12

    def test_holiday_budget_mid_week(self, mock_orchestrator):
        """Mid-week holiday should give ~22h, not weekend-length budget."""
        from orchestration.task_scheduler import TaskScheduler
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.OVERNIGHT

        # Wednesday evening (Thursday is a holiday)
        wednesday = datetime(2026, 1, 21, 18, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'), \
             patch('orchestration.task_scheduler.datetime') as mock_dt:
            mock_dt.now.return_value = wednesday

            scheduler = TaskScheduler(mock_orchestrator)
            scheduler._holiday_cache = {}
            scheduler._early_close_cache = {}

            # Friday morning (skip Thursday holiday)
            friday = datetime(2026, 1, 23, 8, 0, 0, tzinfo=pytz.timezone('US/Eastern'))

            def mock_is_holiday(d):
                return d == date(2026, 1, 22)  # Thursday is holiday

            with patch.object(scheduler, '_get_next_trading_premarket', return_value=friday), \
                 patch.object(scheduler, '_is_holiday', side_effect=mock_is_holiday):
                budget = scheduler.calculate_research_budget()

        assert budget.budget_type == "holiday"
        assert budget.is_extended is True
        # Should be ~38h, not 56+ like weekend
        assert budget.total_hours < 50


class TestGetExtendedWindowTasks:
    """Test TaskScheduler.get_extended_window_tasks() method."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        mock = MagicMock()
        mock.tz = pytz.timezone('US/Eastern')
        return mock

    def test_high_budget_returns_cleanup_tasks(self, mock_orchestrator):
        """With >80% budget remaining, should return cleanup tasks."""
        from orchestration.task_scheduler import TaskScheduler, ResearchBudget

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            # Mock high budget (85% remaining)
            mock_budget = ResearchBudget(
                total_hours=56.0,
                research_hours=47.6,  # 85%
                budget_type="weekend",
                is_extended=True,
            )

            with patch.object(scheduler, 'calculate_research_budget', return_value=mock_budget):
                tasks = scheduler.get_extended_window_tasks()

        assert "generate_weekly_report" in tasks
        assert "backup_databases" in tasks
        assert "vacuum_databases" in tasks

    def test_mid_budget_returns_research_tasks(self, mock_orchestrator):
        """With 20%-80% budget remaining, should return research tasks."""
        from orchestration.task_scheduler import TaskScheduler, ResearchBudget

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            # Mock mid budget (50% remaining)
            mock_budget = ResearchBudget(
                total_hours=56.0,
                research_hours=28.0,  # 50%
                budget_type="weekend",
                is_extended=True,
            )

            with patch.object(scheduler, 'calculate_research_budget', return_value=mock_budget):
                tasks = scheduler.get_extended_window_tasks()

        assert "run_weekend_research" in tasks

    def test_low_budget_returns_data_refresh_tasks(self, mock_orchestrator):
        """With 1.5h-20% budget remaining, should return data refresh tasks."""
        from orchestration.task_scheduler import TaskScheduler, ResearchBudget

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            # Mock low budget (10% remaining, 5.6h)
            mock_budget = ResearchBudget(
                total_hours=56.0,
                research_hours=5.6,  # 10%
                budget_type="weekend",
                is_extended=True,
            )

            with patch.object(scheduler, 'calculate_research_budget', return_value=mock_budget):
                tasks = scheduler.get_extended_window_tasks()

        assert "refresh_index_constituents" in tasks
        assert "refresh_fundamentals" in tasks

    def test_minimal_budget_returns_prep_tasks(self, mock_orchestrator):
        """With <1.5h budget remaining, should return prep tasks."""
        from orchestration.task_scheduler import TaskScheduler, ResearchBudget

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            # Mock minimal budget (1h remaining)
            mock_budget = ResearchBudget(
                total_hours=56.0,
                research_hours=1.0,
                budget_type="weekend",
                is_extended=True,
            )

            with patch.object(scheduler, 'calculate_research_budget', return_value=mock_budget):
                tasks = scheduler.get_extended_window_tasks()

        assert "validate_strategies" in tasks
        assert "verify_system_readiness" in tasks


class TestGetCurrentMode:
    """Test TaskScheduler.get_current_mode() method."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        mock = MagicMock()
        mock.tz = pytz.timezone('US/Eastern')
        return mock

    def test_returns_mode_dict(self, mock_orchestrator):
        """get_current_mode should return dict with required fields."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode, MarketCalendar
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.MARKET_OPEN

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            mock_cal = MarketCalendar(
                is_trading_day=True,
                is_early_close=False,
                next_trading_day=date(2026, 1, 20),
                hours_until_trading=0,
            )

            with patch.object(scheduler, 'get_operating_mode', return_value=OperatingMode.TRADING), \
                 patch.object(scheduler, 'get_market_calendar', return_value=mock_cal):
                result = scheduler.get_current_mode()

        assert "mode" in result
        assert "is_trading" in result
        assert "is_research" in result
        assert "is_prep" in result
        assert "calendar" in result
        assert result["mode"] == "trading"
        assert result["is_trading"] is True

    def test_research_mode_includes_budget(self, mock_orchestrator):
        """Research mode should include budget info and tasks."""
        from orchestration.task_scheduler import TaskScheduler, OperatingMode, MarketCalendar, ResearchBudget
        from daily_orchestrator import MarketPhase

        mock_orchestrator.get_current_phase.return_value = MarketPhase.WEEKEND

        with patch.object(TaskScheduler, '_load_task_specs'), \
             patch.object(TaskScheduler, '_load_state'):
            scheduler = TaskScheduler(mock_orchestrator)

            mock_cal = MarketCalendar(
                is_trading_day=False,
                is_early_close=False,
                next_trading_day=date(2026, 1, 19),
                hours_until_trading=40.0,
            )

            mock_budget = ResearchBudget(
                total_hours=56.0,
                research_hours=40.0,
                budget_type="weekend",
                is_extended=True,
            )

            with patch.object(scheduler, 'get_operating_mode', return_value=OperatingMode.RESEARCH), \
                 patch.object(scheduler, 'get_market_calendar', return_value=mock_cal), \
                 patch.object(scheduler, 'calculate_research_budget', return_value=mock_budget), \
                 patch.object(scheduler, 'get_extended_window_tasks', return_value=["run_weekend_research"]):
                result = scheduler.get_current_mode()

        assert result["mode"] == "research"
        assert result["is_research"] is True
        assert "budget" in result
        assert result["budget"]["budget_type"] == "weekend"
        assert "tasks" in result
        assert "run_weekend_research" in result["tasks"]


class TestConfigFlag:
    """Test USE_UNIFIED_SCHEDULER config flag."""

    def test_flag_defaults_to_false(self):
        """USE_UNIFIED_SCHEDULER should default to False."""
        from config import USE_UNIFIED_SCHEDULER

        # In test environment, should be False by default
        assert USE_UNIFIED_SCHEDULER is False

    def test_flag_can_be_enabled(self):
        """Flag should be controllable via environment variable."""
        import os
        from importlib import reload

        # Save original value
        original = os.environ.get('USE_UNIFIED_SCHEDULER')

        try:
            os.environ['USE_UNIFIED_SCHEDULER'] = 'true'
            import config
            reload(config)
            assert config.USE_UNIFIED_SCHEDULER is True
        finally:
            # Restore
            if original is not None:
                os.environ['USE_UNIFIED_SCHEDULER'] = original
            else:
                os.environ.pop('USE_UNIFIED_SCHEDULER', None)
            reload(config)


class TestHardwareIntegration:
    """Test hardware integration for operating modes."""

    def test_operating_mode_led_map_exists(self):
        """OPERATING_MODE_LED_MAP should exist and have all modes."""
        from hardware.integration import HardwareStatus

        led_map = HardwareStatus.OPERATING_MODE_LED_MAP

        assert "trading" in led_map
        assert "research" in led_map
        assert "prep" in led_map

    def test_set_operating_mode_method_exists(self):
        """HardwareStatus should have set_operating_mode method."""
        from hardware.integration import HardwareStatus

        hs = HardwareStatus()
        assert hasattr(hs, 'set_operating_mode')
        assert callable(getattr(hs, 'set_operating_mode'))


class TestOrchestratorIntegration:
    """Test orchestrator integration with unified scheduler."""

    def test_run_unified_extended_window_exists(self):
        """Orchestrator should have _run_unified_extended_window method."""
        from daily_orchestrator import DailyOrchestrator

        orch = DailyOrchestrator(paper_mode=True)
        assert hasattr(orch, '_run_unified_extended_window')

    def test_get_operating_mode_phase_exists(self):
        """Orchestrator should have _get_operating_mode_phase method."""
        from daily_orchestrator import DailyOrchestrator

        orch = DailyOrchestrator(paper_mode=True)
        assert hasattr(orch, '_get_operating_mode_phase')

    def test_unified_scheduler_disabled_by_default(self):
        """With flag disabled, should use legacy weekend handling."""
        from daily_orchestrator import DailyOrchestrator
        from config import USE_UNIFIED_SCHEDULER

        assert USE_UNIFIED_SCHEDULER is False

        orch = DailyOrchestrator(paper_mode=True)
        # The orchestrator should exist and have the legacy method
        assert hasattr(orch, '_task_run_weekend_schedule')
