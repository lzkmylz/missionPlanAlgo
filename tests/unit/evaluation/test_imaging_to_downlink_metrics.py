"""
观测数传实效性指标测试

测试从成像完成到数传完成用时的计算逻辑
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from evaluation import PerformanceMetrics, MetricsCalculator
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


@dataclass
class MockScheduledTask:
    """模拟已调度任务"""
    task_id: str
    satellite_id: str = "sat_1"
    imaging_start: datetime = field(default_factory=lambda: datetime(2024, 1, 1, 12, 0, 0))
    imaging_end: datetime = field(default_factory=lambda: datetime(2024, 1, 1, 12, 0, 10))
    downlink_start: Optional[datetime] = None
    downlink_end: Optional[datetime] = None
    ground_station_id: Optional[str] = None


@dataclass
class MockMission:
    """模拟任务"""
    targets: List[Any] = field(default_factory=lambda: [1, 2, 3])
    start_time: datetime = field(default_factory=lambda: datetime(2024, 1, 1, 12, 0, 0))
    end_time: datetime = field(default_factory=lambda: datetime(2024, 1, 1, 14, 0, 0))
    satellites: List[Any] = field(default_factory=list)


@dataclass
class MockScheduleResult:
    """模拟调度结果"""
    scheduled_tasks: List[Any] = field(default_factory=list)
    unscheduled_tasks: Dict[str, Any] = field(default_factory=dict)
    makespan: float = 3600.0
    computation_time: float = 1.0


class TestImagingToDownlinkMetrics:
    """观测数传实效性指标测试类"""

    def test_basic_calculation(self):
        """测试基本计算"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        tasks = [
            MockScheduledTask(
                "task_1",
                imaging_end=base_time,
                downlink_end=base_time + timedelta(minutes=30)
            ),
            MockScheduledTask(
                "task_2",
                imaging_end=base_time + timedelta(minutes=10),
                downlink_end=base_time + timedelta(minutes=40)
            ),
        ]

        mission = MockMission()
        calculator = MetricsCalculator(mission)

        avg_delay = calculator._calculate_imaging_to_downlink_time(tasks)

        # 任务1: 30分钟，任务2: 30分钟，平均30分钟
        assert avg_delay == 30 * 60  # 1800秒

    def test_with_missing_downlink(self):
        """测试某些任务没有数传信息的情况"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        tasks = [
            MockScheduledTask(
                "task_1",
                imaging_end=base_time,
                downlink_end=base_time + timedelta(minutes=30)
            ),
            MockScheduledTask(
                "task_2",
                imaging_end=base_time + timedelta(minutes=10),
                downlink_end=None  # 无数传
            ),
            MockScheduledTask(
                "task_3",
                imaging_end=base_time + timedelta(minutes=20),
                downlink_end=base_time + timedelta(minutes=50)
            ),
        ]

        mission = MockMission()
        calculator = MetricsCalculator(mission)

        avg_delay = calculator._calculate_imaging_to_downlink_time(tasks)

        # 只计算有数传的任务: (30 + 30) / 2 = 30分钟
        assert avg_delay == 30 * 60

    def test_empty_tasks(self):
        """测试空任务列表"""
        mission = MockMission()
        calculator = MetricsCalculator(mission)

        avg_delay = calculator._calculate_imaging_to_downlink_time([])

        assert avg_delay == 0.0

    def test_no_downlink_tasks(self):
        """测试所有任务都没有数传信息"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        tasks = [
            MockScheduledTask(
                "task_1",
                imaging_end=base_time,
                downlink_end=None
            ),
            MockScheduledTask(
                "task_2",
                imaging_end=base_time + timedelta(minutes=10),
                downlink_end=None
            ),
        ]

        mission = MockMission()
        calculator = MetricsCalculator(mission)

        avg_delay = calculator._calculate_imaging_to_downlink_time(tasks)

        assert avg_delay == 0.0

    def test_negative_delay_ignored(self):
        """测试负数延迟被忽略（异常情况）"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        tasks = [
            MockScheduledTask(
                "task_1",
                imaging_end=base_time + timedelta(minutes=30),
                downlink_end=base_time  # 数传在成像之前（异常）
            ),
            MockScheduledTask(
                "task_2",
                imaging_end=base_time,
                downlink_end=base_time + timedelta(minutes=20)
            ),
        ]

        mission = MockMission()
        calculator = MetricsCalculator(mission)

        avg_delay = calculator._calculate_imaging_to_downlink_time(tasks)

        # 只计算正数延迟的任务
        assert avg_delay == 20 * 60

    def test_metrics_dataclass(self):
        """测试 PerformanceMetrics 数据类"""
        metrics = PerformanceMetrics(
            imaging_to_downlink_time=1800.0  # 30分钟
        )

        assert metrics.imaging_to_downlink_time == 1800.0

        # 测试 to_dict
        d = metrics.to_dict()
        assert 'imaging_to_downlink_time_seconds' in d
        assert 'imaging_to_downlink_time_minutes' in d
        assert d['imaging_to_downlink_time_seconds'] == 1800.0
        assert d['imaging_to_downlink_time_minutes'] == 30.0

    def test_calculate_all_integration(self):
        """测试 calculate_all 方法集成"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        tasks = [
            MockScheduledTask(
                "task_1",
                imaging_end=base_time,
                downlink_end=base_time + timedelta(minutes=20)
            ),
            MockScheduledTask(
                "task_2",
                imaging_end=base_time + timedelta(minutes=30),
                downlink_end=base_time + timedelta(minutes=60)
            ),
        ]

        mission = MockMission(targets=[1, 2])
        result = MockScheduleResult(scheduled_tasks=tasks)

        calculator = MetricsCalculator(mission)
        metrics = calculator.calculate_all(result)

        # 验证观测数传实效性被计算
        assert metrics.imaging_to_downlink_time > 0
        # (20 + 30) / 2 = 25分钟
        assert metrics.imaging_to_downlink_time == 25 * 60


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
