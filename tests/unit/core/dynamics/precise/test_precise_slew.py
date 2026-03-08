"""
精确姿态机动计算模块单元测试

测试内容:
1. 刚体动力学模型正确性
2. 轨迹规划器时间最优性
3. 能量模型准确性
4. 动量管理约束检查
5. 与简化模型结果对比
"""

import pytest
import numpy as np
from datetime import datetime

from core.dynamics.precise import (
    RigidBodyDynamics, TimeOptimalTrajectoryPlanner, EnergyConsumptionModel,
    MomentumManagementSystem, PreciseSlewCalculator, SatelliteDynamicsConfig,
    AttitudeState, Quaternion, AngularVelocity, InertiaTensor, ReactionWheelConfig
)


class TestRigidBodyDynamics:
    """测试刚体动力学模型"""

    def test_initialization(self):
        """测试初始化"""
        inertia = InertiaTensor.diagonal(100.0, 80.0, 60.0)
        dynamics = RigidBodyDynamics(inertia)

        assert dynamics.inertia.Ixx == 100.0
        assert dynamics.inertia.Iyy == 80.0
        assert dynamics.inertia.Izz == 60.0

    def test_invalid_inertia(self):
        """测试无效惯性张量"""
        # 负特征值应该抛出异常
        with pytest.raises(ValueError):
            inertia = InertiaTensor(-100.0, 80.0, 60.0)
            RigidBodyDynamics(inertia)

    def test_equations_of_motion(self):
        """测试欧拉方程"""
        inertia = InertiaTensor.diagonal(100.0, 80.0, 60.0)
        dynamics = RigidBodyDynamics(inertia)

        # 测试状态: 单位四元数, 零角速度
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 0.0])

        state_dot = dynamics.equations_of_motion(state, control)

        # 零角速度下, 四元数导数应为零
        assert np.allclose(state_dot[0:4], np.zeros(4))

    def test_kinetic_energy(self):
        """测试动能计算"""
        inertia = InertiaTensor.diagonal(100.0, 80.0, 60.0)
        dynamics = RigidBodyDynamics(inertia)

        omega = AngularVelocity(0.1, 0.0, 0.0)  # rad/s
        energy = dynamics.compute_kinetic_energy(omega)

        # E = 0.5 * Ixx * ωx² = 0.5 * 100 * 0.01 = 0.5
        expected = 0.5 * 100.0 * 0.1**2
        assert np.isclose(energy, expected)


class TestTimeOptimalTrajectoryPlanner:
    """测试时间最优轨迹规划器"""

    def test_initialization(self):
        """测试初始化"""
        planner = TimeOptimalTrajectoryPlanner()
        assert planner.config.max_control_torque == 0.5
        assert planner.config.max_angular_velocity == 3.0

    def test_triangular_profile(self):
        """测试三角形速度剖面"""
        planner = TimeOptimalTrajectoryPlanner()

        q_start = Quaternion(1.0, 0.0, 0.0, 0.0)
        q_end = Quaternion(0.0, 1.0, 0.0, 0.0)  # 180度旋转

        trajectory = planner.plan_trajectory(
            q_start=q_start,
            q_end=q_end,
            effective_inertia=100.0
        )

        assert trajectory.total_time > 0
        assert trajectory.rotation_angle > 0
        assert trajectory.profile_type.value in ['triangular', 'trapezoidal']

    def test_bang_bang_control(self):
        """测试bang-bang控制切换"""
        planner = TimeOptimalTrajectoryPlanner()

        q_start = Quaternion(1.0, 0.0, 0.0, 0.0)
        q_end = Quaternion(np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0)  # 90度

        trajectory = planner.plan_trajectory(
            q_start=q_start,
            q_end=q_end,
            effective_inertia=100.0
        )

        # 应该有控制切换点
        assert len(trajectory.switch_times) >= 1

    def test_slerp_interpolation(self):
        """测试SLERP插值"""
        planner = TimeOptimalTrajectoryPlanner()

        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)

        # t=0时应该等于q1
        q_mid = planner._slerp(q1, q2, 0.0)
        assert np.allclose(q_mid.to_array(), q1.to_array())

        # t=1时应该等于q2
        q_end = planner._slerp(q1, q2, 1.0)
        assert np.allclose(q_end.to_array(), q2.to_array())


class TestEnergyConsumptionModel:
    """测试能量消耗模型"""

    def test_initialization(self):
        """测试初始化"""
        wheel_config = ReactionWheelConfig()
        model = EnergyConsumptionModel(wheel_config)

        assert model.wheel_config.num_wheels == 4

    def test_energy_calculation(self):
        """测试能量计算"""
        wheel_config = ReactionWheelConfig(
            num_wheels=4,
            wheel_inertia=0.01,
            motor_efficiency=0.85
        )
        model = EnergyConsumptionModel(wheel_config)

        # 创建一个虚拟轨迹
        from core.dynamics.precise import Trajectory, TrajectoryProfile, VelocityProfileType

        profile = TrajectoryProfile(
            time_points=np.linspace(0, 10, 100),
            angular_velocity=np.ones(100) * 1.0,  # 1 deg/s
            angular_acceleration=np.zeros(100),
            control_torque=np.zeros(100)
        )

        trajectory = Trajectory(
            rotation_axis=np.array([0.0, 1.0, 0.0]),
            rotation_angle=10.0,
            total_time=10.0,
            acceleration_time=2.0,
            coast_time=6.0,
            deceleration_time=2.0,
            settling_time=5.0,
            max_angular_velocity=1.0,
            profile_type=VelocityProfileType.TRAPEZOIDAL,
            velocity_profile=profile
        )

        energy_report = model.compute_energy(trajectory)

        assert energy_report.total_energy > 0
        assert energy_report.average_power > 0


class TestMomentumManagementSystem:
    """测试动量管理系统"""

    def test_initialization(self):
        """测试初始化"""
        mgr = MomentumManagementSystem(num_wheels=4)
        assert mgr.num_wheels == 4
        assert mgr.max_momentum_per_wheel > 0

    def test_pyramid_configuration(self):
        """测试金字塔构型"""
        mgr = MomentumManagementSystem(
            num_wheels=4,
            wheel_configuration='pyramid_4'
        )

        # 检查方向矩阵维度
        assert mgr.wheel_directions.shape == (3, 4)

    def test_momentum_distribution(self):
        """测试动量分配"""
        mgr = MomentumManagementSystem(num_wheels=4)

        # 给定总动量
        total_momentum = np.array([1.0, 0.0, 0.0])  # Nms
        wheel_momentum = mgr.get_wheel_momentum_distribution(total_momentum)

        # 应该分配到4个飞轮
        assert len(wheel_momentum) == 4


class TestPreciseSlewCalculator:
    """测试精确机动计算器"""

    def test_initialization_precise(self):
        """测试精确模式初始化"""
        config = SatelliteDynamicsConfig(
            max_control_torque=0.5,
            max_angular_velocity=3.0
        )
        calc = PreciseSlewCalculator(config, use_precise=True)

        assert calc.use_precise is True
        assert calc.dynamics is not None

    def test_initialization_simple(self):
        """测试简化模式初始化"""
        config = SatelliteDynamicsConfig()
        calc = PreciseSlewCalculator(config, use_precise=False)

        assert calc.use_precise is False

    def test_simple_slew_time(self):
        """测试简化机动时间计算"""
        config = SatelliteDynamicsConfig(
            max_slew_rate=3.0,
            settling_time=5.0
        )
        calc = PreciseSlewCalculator(config, use_precise=False)

        slew_angle = 30.0  # degrees
        result = calc.calculate_slew_time(slew_angle)

        expected_time = slew_angle / 3.0 + 5.0
        assert np.isclose(result.total_time, expected_time, rtol=0.1)

    def test_precise_slew_maneuver(self):
        """测试精确机动分析"""
        config = SatelliteDynamicsConfig()
        calc = PreciseSlewCalculator(config, use_precise=True)

        prev_attitude = AttitudeState(
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=datetime.now()
        )

        target_attitude = AttitudeState(
            quaternion=Quaternion(0.0, 1.0, 0.0, 0.0),
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=datetime.now()
        )

        result = calc.calculate_slew_maneuver(
            prev_attitude=prev_attitude,
            target_attitude=target_attitude
        )

        assert result.total_time > 0
        assert result.trajectory is not None
        assert result.energy_consumption >= 0


class TestComparisonWithSimpleModel:
    """测试与简化模型的对比"""

    def test_time_comparison(self):
        """比较两种模型的时间估算"""
        config = SatelliteDynamicsConfig(
            inertia_tensor=InertiaTensor.diagonal(100.0, 80.0, 60.0),
            max_control_torque=0.5,
            max_angular_velocity=3.0,
            max_slew_rate=3.0,
            settling_time=5.0
        )

        # 简化模型
        simple_calc = PreciseSlewCalculator(config, use_precise=False)

        # 精确模型
        precise_calc = PreciseSlewCalculator(config, use_precise=True)

        # 测试不同角度
        angles = [10.0, 20.0, 30.0, 45.0]

        for angle in angles:
            # 简化模型
            simple_result = simple_calc.calculate_slew_time(angle)

            # 精确模型 (需要构造姿态四元数)
            from core.dynamics.precise import Quaternion
            q_start = Quaternion(1.0, 0.0, 0.0, 0.0)
            angle_rad = np.radians(angle)
            q_end = Quaternion(
                w=np.cos(angle_rad / 2),
                x=0.0,
                y=np.sin(angle_rad / 2),
                z=0.0
            )

            precise_result = precise_calc.calculate_slew_time(
                angle, q_start=q_start, q_end=q_end
            )

            # 结果应该在合理范围内
            assert simple_result.total_time > 0
            assert precise_result.total_time > 0

            # 输出差异 (用于分析)
            diff = abs(precise_result.total_time - simple_result.total_time)
            print(f"Angle {angle}°: simple={simple_result.total_time:.2f}s, "
                  f"precise={precise_result.total_time:.2f}s, diff={diff:.2f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
