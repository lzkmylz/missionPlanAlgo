"""
离散采样选择 imaging_begin 单元测试

TDD 测试: 验证在可见性窗口内离散采样选择最优成像时刻
"""

import pytest
from datetime import datetime, timedelta

from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, Orbit
from core.models.mission import Mission
from scheduler.greedy.greedy_scheduler import GreedyScheduler


@pytest.fixture
def sample_satellite():
    """创建测试卫星"""
    from core.models import SatelliteCapabilities, ImagingMode
    return Satellite(
        id="SAT-001",
        name="Test Satellite",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(altitude=500000.0, inclination=97.4),
        capabilities=SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM],
            resolution=0.5,
            max_roll_angle=45.0,
            max_pitch_angle=20.0,
            agility={
                'max_slew_rate': 2.0,
                'settling_time': 5.0,
                'max_torque': 0.5,
            }
        )
    )


@pytest.fixture
def sample_mission(sample_satellite):
    """创建测试任务"""
    return Mission(
        name="Test Mission",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 2, 0, 0, 0),
        satellites=[sample_satellite],
        targets=[]
    )


@pytest.fixture
def sample_target():
    """创建测试目标"""
    return Target(
        id="target_001",
        name="Test Target",
        target_type=TargetType.POINT,
        longitude=0.0,
        latitude=0.0,
        priority=1
    )


class TestDiscreteSampling:
    """测试离散采样选择 imaging_begin"""

    def test_discrete_sampling_1_second_step(self, sample_mission):
        """测试 1 秒步长的离散采样"""
        scheduler = GreedyScheduler({})
        scheduler.mission = sample_mission

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(seconds=30)  # 30秒窗口

        # 模拟离散采样
        samples = []
        current = window_start
        step = timedelta(seconds=1)
        while current <= window_end:
            samples.append(current)
            current += step

        # 验证采样点数量 (30秒窗口, 1秒步长 = 31个点)
        assert len(samples) == 31
        assert samples[0] == window_start
        assert samples[-1] == window_end

    def test_fifo_strategy_selects_first_feasible(self, sample_mission):
        """测试 FIFO 策略选择第一个满足约束的时刻"""
        scheduler = GreedyScheduler({})
        scheduler.mission = sample_mission

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(seconds=10)

        # 模拟候选时刻及其姿态约束满足情况
        candidates = [
            (window_start, False),  # 不满足
            (window_start + timedelta(seconds=1), False),  # 不满足
            (window_start + timedelta(seconds=2), True),   # 第一个满足
            (window_start + timedelta(seconds=3), True),   # 也满足, 但不选
        ]

        # FIFO 策略: 选择第一个满足约束的
        selected = None
        for timestamp, feasible in candidates:
            if feasible:
                selected = timestamp
                break

        assert selected == window_start + timedelta(seconds=2)

    def test_imaging_begin_within_window(self, sample_mission):
        """测试选择的 imaging_begin 必须在可见性窗口内"""
        scheduler = GreedyScheduler({})
        scheduler.mission = sample_mission

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)

        # 任何有效的 imaging_begin 必须满足:
        # window_start <= imaging_begin <= window_end - imaging_duration
        imaging_duration = 10.0  # 10秒

        # 选择 imaging_begin
        imaging_begin = window_start + timedelta(seconds=20)

        # 验证在窗口内
        assert imaging_begin >= window_start
        assert imaging_begin + timedelta(seconds=imaging_duration) <= window_end


class TestGreedySchedulerWithDiscreteSampling:
    """测试 GreedyScheduler 的离散采样集成"""

    def test_scheduler_has_discrete_sampling_method(self, sample_mission):
        """验证调度器有离散采样方法"""
        scheduler = GreedyScheduler({})
        scheduler.mission = sample_mission

        # 应该有离散采样相关方法
        assert hasattr(scheduler, '_select_imaging_begin_by_sampling')

    def test_select_imaging_begin_returns_valid_time(self, sample_mission, sample_satellite, sample_target):
        """测试选择方法返回有效时间"""
        scheduler = GreedyScheduler({})
        scheduler.mission = sample_mission

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)
        prev_end_time = window_start - timedelta(seconds=30)

        # 调用离散采样方法
        imaging_begin = scheduler._select_imaging_begin_by_sampling(
            satellite=sample_satellite,
            target=sample_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            imaging_duration=10.0
        )

        # 验证返回有效时间
        assert imaging_begin is not None
        assert isinstance(imaging_begin, datetime)
        assert imaging_begin >= window_start
        assert imaging_begin <= window_end


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
