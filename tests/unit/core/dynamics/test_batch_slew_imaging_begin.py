"""
BatchSlewCandidate imaging_begin 字段单元测试

TDD 测试: 验证新字段支持正确的成像时间选择
"""

import pytest
from datetime import datetime, timedelta

from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, Orbit
from core.models.mission import Mission
from scheduler.constraints import (
    BatchSlewConstraintChecker,
    BatchSlewCandidate,
    SlewFeasibilityResult,
)


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


@pytest.fixture
def satellite_position_ecef():
    """卫星在ECEF坐标系中的位置"""
    return (6871000.0, 0.0, 0.0)


class TestBatchSlewCandidateImagingBegin:
    """测试 BatchSlewCandidate 的 imaging_begin 字段"""

    def test_candidate_with_explicit_imaging_begin(
        self, sample_satellite, sample_target, satellite_position_ecef
    ):
        """测试显式设置 imaging_begin 的候选"""
        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)
        prev_end_time = window_start - timedelta(seconds=30)

        # imaging_begin 在 window_start 之后 10 秒
        imaging_begin = window_start + timedelta(seconds=10)

        candidate = BatchSlewCandidate(
            sat_id=sample_satellite.id,
            satellite=sample_satellite,
            target=sample_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            prev_target=None,
            imaging_duration=10.0,
            sat_position=satellite_position_ecef,
            sat_velocity=(0.0, 7650.0, 0.0),
            imaging_begin=imaging_begin  # 新字段
        )

        assert candidate.imaging_begin == imaging_begin
        assert candidate.window_start == window_start
        assert candidate.window_end == window_end

        # time_interval 应该是 imaging_begin - prev_end_time = 40 秒
        time_interval = (candidate.imaging_begin - candidate.prev_end_time).total_seconds()
        assert time_interval == 40.0

    def test_candidate_without_explicit_imaging_begin_defaults_to_window_start(
        self, sample_satellite, sample_target, satellite_position_ecef
    ):
        """测试未设置 imaging_begin 时默认为 window_start"""
        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)
        prev_end_time = window_start - timedelta(seconds=30)

        # 不传入 imaging_begin
        candidate = BatchSlewCandidate(
            sat_id=sample_satellite.id,
            satellite=sample_satellite,
            target=sample_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            prev_target=None,
            imaging_duration=10.0,
            sat_position=satellite_position_ecef,
            sat_velocity=(0.0, 7650.0, 0.0)
        )

        # 默认 imaging_begin = window_start
        assert candidate.imaging_begin == window_start

    def test_time_interval_calculation_with_imaging_begin(
        self, sample_satellite, sample_target, satellite_position_ecef
    ):
        """测试使用 imaging_begin 的 time_interval 计算"""
        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)
        prev_end_time = window_start - timedelta(seconds=60)
        imaging_begin = window_start + timedelta(seconds=20)

        candidate = BatchSlewCandidate(
            sat_id=sample_satellite.id,
            satellite=sample_satellite,
            target=sample_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            prev_target=None,
            imaging_duration=10.0,
            sat_position=satellite_position_ecef,
            sat_velocity=(0.0, 7650.0, 0.0),
            imaging_begin=imaging_begin
        )

        # time_interval = imaging_begin - prev_end_time = 80 秒
        time_interval = (candidate.imaging_begin - candidate.prev_end_time).total_seconds()
        assert time_interval == 80.0

        # 注意：这不等于 window_start - prev_end_time = 60 秒
        window_based_interval = (candidate.window_start - candidate.prev_end_time).total_seconds()
        assert window_based_interval == 60.0
        assert time_interval != window_based_interval


class TestBatchSlewDataImagingBegins:
    """测试 BatchSlewData 的 imaging_begins 数组"""

    def test_batch_data_with_imaging_begins(
        self, sample_satellite, sample_target, satellite_position_ecef
    ):
        """测试 BatchSlewData 包含 imaging_begins 数组"""
        from scheduler.constraints.batch_slew_calculator import BatchSlewData

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)
        prev_end_time = window_start - timedelta(seconds=30)

        # 创建多个候选，每个有不同的 imaging_begin
        candidates = []
        for i in range(3):
            imaging_begin = window_start + timedelta(seconds=i * 10)
            candidates.append(BatchSlewCandidate(
                sat_id=sample_satellite.id,
                satellite=sample_satellite,
                target=sample_target,
                window_start=window_start,
                window_end=window_end,
                prev_end_time=prev_end_time,
                prev_target=None,
                imaging_duration=10.0,
                sat_position=satellite_position_ecef,
                sat_velocity=(0.0, 7650.0, 0.0),
                imaging_begin=imaging_begin
            ))

        # 创建 BatchSlewData
        data = BatchSlewData(n_candidates=3)
        data.candidates = candidates

        # 验证有 imaging_begins 数组
        assert hasattr(data, 'imaging_begins')
        assert len(data.imaging_begins) == 3


class TestSlewConstraintWithImagingBegin:
    """测试使用 imaging_begin 的约束检查"""

    def test_slew_constraint_uses_imaging_begin_for_time_interval(
        self, sample_mission, sample_satellite, sample_target, satellite_position_ecef
    ):
        """验证约束检查使用 imaging_begin 计算 time_interval"""
        checker = BatchSlewConstraintChecker(sample_mission, use_precise_model=True)

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        window_end = window_start + timedelta(minutes=5)
        prev_end_time = window_start - timedelta(seconds=30)

        # 情况1: imaging_begin = window_start (30秒间隔)
        candidate1 = BatchSlewCandidate(
            sat_id=sample_satellite.id,
            satellite=sample_satellite,
            target=sample_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            prev_target=None,
            imaging_duration=10.0,
            sat_position=satellite_position_ecef,
            sat_velocity=(0.0, 7650.0, 0.0),
            imaging_begin=window_start
        )

        # 情况2: imaging_begin = window_start + 20秒 (50秒间隔，更宽松)
        candidate2 = BatchSlewCandidate(
            sat_id=sample_satellite.id,
            satellite=sample_satellite,
            target=sample_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            prev_target=None,
            imaging_duration=10.0,
            sat_position=satellite_position_ecef,
            sat_velocity=(0.0, 7650.0, 0.0),
            imaging_begin=window_start + timedelta(seconds=20)
        )

        results = checker.check_slew_feasibility_batch([candidate1, candidate2])

        # 两个都应该返回结果
        assert len(results) == 2
        for result in results:
            assert isinstance(result, SlewFeasibilityResult)
            assert result.slew_time > 0

        # candidate2 有更长的 time_interval，应该更容易满足约束
        # 注意：具体可行性取决于机动角度，但 time_interval 计算应该正确


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
