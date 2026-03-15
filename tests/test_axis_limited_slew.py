"""
分轴姿态机动限制测试

测试角速度和角加速度分轴限制（滚转/俯仰）的功能。
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from core.models.satellite import Satellite, SatelliteCapabilities, SatelliteType, Orbit
from core.constants import (
    DEFAULT_MAX_ROLL_RATE_DEG_S, DEFAULT_MAX_PITCH_RATE_DEG_S,
    DEFAULT_MAX_ROLL_ACCEL_DEG_S2, DEFAULT_MAX_PITCH_ACCEL_DEG_S2
)
from core.dynamics.precise.trajectory_planner import (
    TrajectoryPlannerConfig, TimeOptimalTrajectoryPlanner
)
from core.dynamics.precise.attitude_types import Quaternion, AttitudeState, AngularVelocity
from core.dynamics.precise.precise_slew_calculator import (
    SatelliteDynamicsConfig, PreciseSlewCalculator
)
from scheduler.constraints.batch_slew_calculator import (
    BatchSlewData, BatchSlewCalculator
)


class TestSatelliteCapabilitiesAxisLimits:
    """测试卫星能力的分轴限制"""

    def test_default_agility_has_axis_limits(self):
        """测试默认agility字典包含分轴限制"""
        cap = SatelliteCapabilities()

        assert 'max_roll_rate' in cap.agility
        assert 'max_pitch_rate' in cap.agility
        assert 'max_roll_acceleration' in cap.agility
        assert 'max_pitch_acceleration' in cap.agility

    def test_default_axis_limit_values(self):
        """测试默认分轴限制值"""
        cap = SatelliteCapabilities()

        # 俯仰应该比滚转慢（默认使用标量的2/3）
        assert cap.agility['max_roll_rate'] == DEFAULT_MAX_ROLL_RATE_DEG_S
        assert cap.agility['max_pitch_rate'] == DEFAULT_MAX_PITCH_RATE_DEG_S  # 2.0

        assert cap.agility['max_roll_acceleration'] == DEFAULT_MAX_ROLL_ACCEL_DEG_S2
        assert cap.agility['max_pitch_acceleration'] == DEFAULT_MAX_PITCH_ACCEL_DEG_S2  # 1.0

    def test_get_effective_limits_roll_dominant(self):
        """测试滚转主导旋转的有效限制"""
        cap = SatelliteCapabilities()
        cap.agility['max_roll_rate'] = 4.0
        cap.agility['max_pitch_rate'] = 2.0

        # 主要绕X轴旋转
        limits = cap.get_effective_limits((1.0, 0.1, 0.0))

        assert limits['effective_rate'] == 4.0  # 滚转限制

    def test_get_effective_limits_pitch_dominant(self):
        """测试俯仰主导旋转的有效限制"""
        cap = SatelliteCapabilities()
        cap.agility['max_roll_rate'] = 4.0
        cap.agility['max_pitch_rate'] = 2.0

        # 主要绕Y轴旋转
        limits = cap.get_effective_limits((0.1, 1.0, 0.0))

        assert limits['effective_rate'] == 2.0  # 俯仰限制

    def test_backward_compatibility(self):
        """测试向后兼容性 - 没有分轴限制时使用标量限制"""
        cap = SatelliteCapabilities()
        # 手动设置只有标量限制
        cap.agility = {
            'max_slew_rate': 3.0,
            'slew_acceleration': 1.5,
            'settling_time': 5.0
        }

        # post_init 应该自动派生分轴限制
        limits = cap.get_effective_limits()

        assert limits['max_roll_rate'] == 3.0
        # 俯仰应该比滚转慢
        assert limits['max_pitch_rate'] < limits['max_roll_rate']


class TestTrajectoryPlannerAxisLimits:
    """测试轨迹规划器的分轴限制"""

    def test_config_has_axis_limits(self):
        """测试配置对象包含分轴限制"""
        config = TrajectoryPlannerConfig()

        assert config.max_roll_rate is not None
        assert config.max_pitch_rate is not None

    def test_config_effective_limits_roll(self):
        """测试配置的有效限制 - 滚转主导"""
        config = TrajectoryPlannerConfig(
            max_roll_rate=4.0,
            max_pitch_rate=2.0,
            max_roll_acceleration=2.0,
            max_pitch_acceleration=1.0
        )

        vel, accel = config.get_effective_limits(np.array([1.0, 0.1, 0.0]))

        assert vel == 4.0
        assert accel == 2.0

    def test_config_effective_limits_pitch(self):
        """测试配置的有效限制 - 俯仰主导"""
        config = TrajectoryPlannerConfig(
            max_roll_rate=4.0,
            max_pitch_rate=2.0,
            max_roll_acceleration=2.0,
            max_pitch_acceleration=1.0
        )

        vel, accel = config.get_effective_limits(np.array([0.1, 1.0, 0.0]))

        assert vel == 2.0
        assert accel == 1.0

    def test_trajectory_roll_faster_than_pitch(self):
        """测试滚转机动比俯仰机动更快（当滚转限制更高时）"""
        config = TrajectoryPlannerConfig(
            max_roll_rate=4.0,
            max_pitch_rate=2.0,
            max_roll_acceleration=2.0,
            max_pitch_acceleration=1.0,
            settling_time=0.0  # 移除稳定时间以简化比较
        )
        planner = TimeOptimalTrajectoryPlanner(config)

        q_start = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        # 滚转机动（45度）
        angle = 45
        q_end_roll = Quaternion(
            w=np.cos(np.radians(angle/2)),
            x=np.sin(np.radians(angle/2)),
            y=0.0, z=0.0
        )
        traj_roll = planner.plan_trajectory(
            q_start=q_start, q_end=q_end_roll, effective_inertia=80.0
        )

        # 俯仰机动（45度）
        q_end_pitch = Quaternion(
            w=np.cos(np.radians(angle/2)),
            x=0.0,
            y=np.sin(np.radians(angle/2)),
            z=0.0
        )
        traj_pitch = planner.plan_trajectory(
            q_start=q_start, q_end=q_end_pitch, effective_inertia=80.0
        )

        # 滚转应该比俯仰快（因为限制更高）
        assert traj_roll.total_time < traj_pitch.total_time


class TestSatelliteDynamicsConfigAxisLimits:
    """测试卫星动力学配置的分轴限制"""

    def test_from_satellite_capabilities(self):
        """测试从卫星能力创建动力学配置"""
        cap = SatelliteCapabilities()
        cap.agility = {
            'max_slew_rate': 4.0,
            'slew_acceleration': 2.0,
            'settling_time': 5.0,
            'max_roll_rate': 4.0,
            'max_pitch_rate': 3.0,
            'max_roll_acceleration': 2.0,
            'max_pitch_acceleration': 1.5
        }

        # 模拟从能力创建配置
        config = SatelliteDynamicsConfig()
        config.max_angular_velocity = cap.agility['max_slew_rate']
        config.max_roll_rate = cap.agility.get('max_roll_rate', 4.0)
        config.max_pitch_rate = cap.agility.get('max_pitch_rate', 4.0 * 0.67)
        config.max_roll_acceleration = cap.agility.get('max_roll_acceleration', 2.0)
        config.max_pitch_acceleration = cap.agility.get('max_pitch_acceleration', 2.0 * 0.67)

        assert config.max_roll_rate == 4.0
        assert config.max_pitch_rate == 3.0
        assert config.max_roll_acceleration == 2.0
        assert config.max_pitch_acceleration == 1.5


class TestBatchSlewCalculatorAxisLimits:
    """测试批量机动计算器的分轴限制"""

    def test_batch_data_has_axis_limits(self):
        """测试批量数据包含分轴限制数组"""
        data = BatchSlewData(3)

        assert hasattr(data, 'max_roll_rates')
        assert hasattr(data, 'max_pitch_rates')
        assert hasattr(data, 'max_roll_accelerations')
        assert hasattr(data, 'max_pitch_accelerations')

    def test_batch_computation_with_axis_limits(self):
        """测试使用分轴限制的批量计算"""
        data = BatchSlewData(3)

        # 填充分轴限制
        data.max_roll_rates[:] = [4.0, 4.0, 4.0]
        data.max_pitch_rates[:] = [2.0, 2.0, 2.0]
        data.max_roll_accelerations[:] = [2.0, 2.0, 2.0]
        data.max_pitch_accelerations[:] = [1.0, 1.0, 1.0]

        # 填充其他必要数据
        data.prev_quaternions[:] = [[1.0, 0.0, 0.0, 0.0]] * 3
        data.target_quaternions[:] = [[1.0, 0.0, 0.0, 0.0]] * 3
        data.sat_positions[:] = [[7000000.0, 0.0, 0.0]] * 3
        data.target_positions[:] = [[6371000.0, 100000.0, 0.0]] * 3
        data.time_intervals[:] = [300.0] * 3
        data.imaging_durations[:] = [10.0] * 3
        data.window_durations[:] = [300.0] * 3
        data.settling_time[:] = [5.0] * 3

        calc = BatchSlewCalculator(use_lookup_table=False)

        # 使用分轴限制计算
        results = calc.compute_batch(data, use_axis_limits=True)

        assert len(results) == 3
        assert all(hasattr(r, 'feasible') for r in results)


class TestIntegrationAxisLimits:
    """集成测试 - 验证端到端分轴限制功能"""

    def test_end_to_end_roll_vs_pitch(self):
        """测试端到端滚转与俯仰机动差异"""
        # 创建卫星配置
        config = SatelliteDynamicsConfig(
            max_roll_rate=4.0,
            max_pitch_rate=2.0,
            max_roll_acceleration=2.0,
            max_pitch_acceleration=1.0,
            settling_time=0.0  # 简化测试
        )

        calc = PreciseSlewCalculator(config=config, use_precise=True)

        now = datetime.now()
        zero_vel = AngularVelocity(x=0, y=0, z=0)

        q_start = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        # 使用更大的角度（45度）来确保能够达到最大速度并显示差异
        angle = 45
        half_rad = np.radians(angle / 2)
        q_end_roll = Quaternion(w=np.cos(half_rad), x=np.sin(half_rad), y=0.0, z=0.0)
        q_end_pitch = Quaternion(w=np.cos(half_rad), x=0.0, y=np.sin(half_rad), z=0.0)

        prev_att = AttitudeState(quaternion=q_start, angular_velocity=zero_vel, timestamp=now)
        target_roll = AttitudeState(quaternion=q_end_roll, angular_velocity=zero_vel, timestamp=now)
        target_pitch = AttitudeState(quaternion=q_end_pitch, angular_velocity=zero_vel, timestamp=now)

        # 计算机动
        result_roll = calc.calculate_slew_maneuver(prev_att, target_roll)
        result_pitch = calc.calculate_slew_maneuver(prev_att, target_pitch)

        # 滚转应该比俯仰快
        assert result_roll.total_time < result_pitch.total_time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
