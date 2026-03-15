"""
精确姿态机动动力学模型

基于刚体动力学的卫星姿态机动精确计算模块，
包含时间最优轨迹规划和能量消耗精确建模。

模块组成:
- rigid_body_dynamics: 刚体动力学模型
- trajectory_planner: 时间最优轨迹规划
- energy_model: 能量消耗精确建模
- momentum_manager: 飞轮动量管理
- precise_slew_calculator: 统一接口适配器

使用示例:
    from core.dynamics.precise import PreciseSlewCalculator, SatelliteDynamicsConfig

    config = SatelliteDynamicsConfig(
        inertia_tensor=np.diag([100, 80, 60]),
        max_control_torque=0.5,
        max_angular_velocity=3.0
    )

    calc = PreciseSlewCalculator(config)
    result = calc.calculate_slew_maneuver(prev_attitude, target_attitude)
"""

from .rigid_body_dynamics import RigidBodyDynamics, InertiaTensor
from .trajectory_planner import TimeOptimalTrajectoryPlanner, Trajectory, TrajectoryProfile, VelocityProfileType
from .energy_model import EnergyConsumptionModel, EnergyReport, ReactionWheelConfig
from .momentum_manager import MomentumManagementSystem, MomentumFeasibilityResult
from .precise_slew_calculator import PreciseSlewCalculator, SatelliteDynamicsConfig
from .attitude_types import (
    AttitudeState, Quaternion, AngularVelocity,
    ControlTorque, MomentumState
)
from .lookup_table import (
    SlewLookupTable, SlewLookupEntry, SlewLookupResult
)

__all__ = [
    # 刚体动力学
    'RigidBodyDynamics',
    'InertiaTensor',

    # 轨迹规划
    'TimeOptimalTrajectoryPlanner',
    'Trajectory',
    'TrajectoryProfile',
    'VelocityProfileType',

    # 能量模型
    'EnergyConsumptionModel',
    'EnergyReport',
    'ReactionWheelConfig',

    # 动量管理
    'MomentumManagementSystem',
    'MomentumFeasibilityResult',

    # 主接口
    'PreciseSlewCalculator',
    'SatelliteDynamicsConfig',

    # 查找表（高性能）
    'SlewLookupTable',
    'SlewLookupEntry',
    'SlewLookupResult',

    # 类型定义
    'AttitudeState',
    'Quaternion',
    'AngularVelocity',
    'ControlTorque',
    'MomentumState',
]
