"""
处理感知调度器单元测试

测试H3: ProcessingAwareScheduler (Chapter 20.5)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from core.processing.processing_aware_scheduler import (
    ProcessingAwareScheduler,
    ProcessingWindow,
    ScheduleWithProcessing,
    ProcessingConstraint
)
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


class TestProcessingWindow:
    """测试处理窗口"""

    def test_processing_window_creation(self):
        """测试创建处理窗口"""
        window = ProcessingWindow(
            imaging_task_id="TASK-001",
            satellite_id="SAT-01",
            earliest_start=datetime(2024, 1, 1, 0, 0, 0),
            latest_end=datetime(2024, 1, 1, 1, 0, 0),
            required_duration_seconds=300.0
        )

        assert window.imaging_task_id == "TASK-001"
        assert window.satellite_id == "SAT-01"
        assert window.required_duration_seconds == 300.0
        assert window.assigned_start is None
        assert window.assigned_end is None


class TestProcessingConstraint:
    """测试处理约束"""

    def test_constraint_creation(self):
        """测试创建处理约束"""
        constraint = ProcessingConstraint(
            min_battery_soc=0.35,
            min_thermal_headroom_c=15.0,
            min_storage_free_gb=5.0
        )

        assert constraint.min_battery_soc == 0.35
        assert constraint.min_thermal_headroom_c == 15.0
        assert constraint.min_storage_free_gb == 5.0


class TestProcessingAwareScheduler:
    """测试处理感知调度器"""

    @pytest.fixture
    def mock_base_scheduler(self):
        """创建模拟的基础调度器"""
        scheduler = Mock()
        return scheduler

    @pytest.fixture
    def mock_processing_manager(self):
        """创建模拟的处理管理器"""
        manager = Mock()
        return manager

    @pytest.fixture
    def mock_satellite_pool(self):
        """创建模拟的卫星池"""
        pool = Mock()
        return pool

    @pytest.fixture
    def scheduler(self, mock_base_scheduler, mock_processing_manager, mock_satellite_pool):
        """创建处理感知调度器"""
        return ProcessingAwareScheduler(
            base_scheduler=mock_base_scheduler,
            processing_manager=mock_processing_manager,
            satellite_pool=mock_satellite_pool
        )

    def test_scheduler_initialization(self, scheduler):
        """测试调度器初始化"""
        assert scheduler.base_scheduler is not None
        assert scheduler.processing_manager is not None
        assert scheduler.satellite_pool is not None

    def test_plan_processing_windows(self, scheduler):
        """测试规划处理窗口"""
        # 创建模拟的调度结果
        task1 = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=datetime(2024, 1, 1, 0, 0, 0),
            imaging_end=datetime(2024, 1, 1, 0, 5, 0),
            imaging_mode="push_broom"
        )
        task2 = ScheduledTask(
            task_id="TASK-002",
            satellite_id="SAT-01",
            target_id="TARGET-02",
            imaging_start=datetime(2024, 1, 1, 0, 30, 0),
            imaging_end=datetime(2024, 1, 1, 0, 35, 0),
            imaging_mode="push_broom"
        )

        base_schedule = ScheduleResult(
            scheduled_tasks=[task1, task2],
            unscheduled_tasks={},
            makespan=2100.0,
            computation_time=1.0,
            iterations=100
        )

        # 模拟处理时间估算
        scheduler._estimate_processing_time = Mock(return_value=300.0)

        windows = scheduler._plan_processing_windows(base_schedule)

        assert "SAT-01" in windows
        assert len(windows["SAT-01"]) == 2

    def test_can_process(self, scheduler):
        """测试处理可行性检查"""
        # 创建模拟的卫星状态
        mock_state = Mock()
        mock_state.battery_soc = 0.5
        mock_state.storage_free_gb = 20.0
        mock_state.thermal_headroom_c = 20.0
        mock_state.ai_accelerator_busy = False

        window = ProcessingWindow(
            imaging_task_id="TASK-001",
            satellite_id="SAT-01",
            earliest_start=datetime(2024, 1, 1, 0, 0, 0),
            latest_end=datetime(2024, 1, 1, 1, 0, 0),
            required_duration_seconds=300.0,
            required_storage_gb=5.0
        )

        result = scheduler._can_process(mock_state, window)
        assert result is True

    def test_can_process_low_battery(self, scheduler):
        """测试低电量时无法处理"""
        mock_state = Mock()
        mock_state.battery_soc = 0.2  # 低于阈值
        mock_state.storage_free_gb = 20.0
        mock_state.thermal_headroom_c = 20.0
        mock_state.ai_accelerator_busy = False

        window = ProcessingWindow(
            imaging_task_id="TASK-001",
            satellite_id="SAT-01",
            earliest_start=datetime(2024, 1, 1, 0, 0, 0),
            latest_end=datetime(2024, 1, 1, 1, 0, 0),
            required_duration_seconds=300.0
        )

        result = scheduler._can_process(mock_state, window)
        assert result is False

    def test_resolve_processing_conflicts(self, scheduler):
        """测试处理冲突消解"""
        # 创建模拟的调度结果
        task1 = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=datetime(2024, 1, 1, 0, 0, 0),
            imaging_end=datetime(2024, 1, 1, 0, 5, 0),
            imaging_mode="push_broom"
        )

        base_schedule = ScheduleResult(
            scheduled_tasks=[task1],
            unscheduled_tasks={},
            makespan=300.0,
            computation_time=1.0,
            iterations=100
        )

        # 创建处理窗口
        windows = {
            "SAT-01": [
                ProcessingWindow(
                    imaging_task_id="TASK-001",
                    satellite_id="SAT-01",
                    earliest_start=datetime(2024, 1, 1, 0, 5, 30),
                    latest_end=datetime(2024, 1, 1, 1, 0, 0),
                    required_duration_seconds=300.0
                )
            ]
        }

        # 模拟卫星和处理可行性
        mock_satellite = Mock()
        mock_state = Mock()
        mock_state.battery_soc = 0.5
        mock_state.storage_free_gb = 20.0
        mock_state.thermal_headroom_c = 20.0
        mock_state.ai_accelerator_busy = False

        mock_satellite.predict_state.return_value = mock_state
        scheduler.satellite_pool.get_satellite.return_value = mock_satellite

        result = scheduler._resolve_processing_conflicts(base_schedule, windows)

        assert isinstance(result, ScheduleWithProcessing)
        assert result.base_schedule == base_schedule
        assert "SAT-01" in result.processing_windows

    def test_estimate_processing_time(self, scheduler):
        """测试处理时间估算"""
        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=datetime(2024, 1, 1, 0, 0, 0),
            imaging_end=datetime(2024, 1, 1, 0, 5, 0),
            imaging_mode="push_broom"
        )

        # 模拟处理管理器返回处理规格
        mock_spec = Mock()
        mock_spec.processing_time_seconds.return_value = 600.0
        scheduler.processing_manager.processing_specs = {
            Mock(): mock_spec
        }

        # 模拟推断任务类型
        scheduler._infer_task_type = Mock(return_value=list(scheduler.processing_manager.processing_specs.keys())[0])

        processing_time = scheduler._estimate_processing_time(task)

        assert processing_time > 0


class TestScheduleWithProcessing:
    """测试带处理的调度结果"""

    def test_schedule_with_processing_creation(self):
        """测试创建带处理的调度结果"""
        base_schedule = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        processing_windows = {
            "SAT-01": [
                ProcessingWindow(
                    imaging_task_id="TASK-001",
                    satellite_id="SAT-01",
                    earliest_start=datetime(2024, 1, 1, 0, 0, 0),
                    latest_end=datetime(2024, 1, 1, 1, 0, 0),
                    required_duration_seconds=300.0,
                    assigned_start=datetime(2024, 1, 1, 0, 5, 30),
                    assigned_end=datetime(2024, 1, 1, 0, 10, 30)
                )
            ]
        }

        schedule = ScheduleWithProcessing(
            base_schedule=base_schedule,
            processing_windows=processing_windows
        )

        assert schedule.base_schedule == base_schedule
        assert schedule.processing_windows == processing_windows

    def test_get_processing_decision(self):
        """测试获取处理决策"""
        base_schedule = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        window = ProcessingWindow(
            imaging_task_id="TASK-001",
            satellite_id="SAT-01",
            earliest_start=datetime(2024, 1, 1, 0, 0, 0),
            latest_end=datetime(2024, 1, 1, 1, 0, 0),
            required_duration_seconds=300.0,
            assigned_start=datetime(2024, 1, 1, 0, 5, 30),
            assigned_end=datetime(2024, 1, 1, 0, 10, 30)
        )

        schedule = ScheduleWithProcessing(
            base_schedule=base_schedule,
            processing_windows={"SAT-01": [window]}
        )

        decision = schedule.get_processing_decision("TASK-001")
        assert decision is not None
        assert decision["task_id"] == "TASK-001"
        assert decision["satellite_id"] == "SAT-01"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
