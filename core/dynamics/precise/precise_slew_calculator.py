"""
精确姿态机动计算器 - 适配器

提供与现有 SlewCalculator 兼容的接口，内部使用精确模型计算。
支持通过配置切换简化/精确模型。

主要功能:
1. 保持与 SlewCalculator 的接口兼容
2. 基于刚体动力学计算机动时间和轨迹
3. 能量消耗精确建模
4. 飞轮动量管理
5. 自动回退到简化模型

使用示例:
    # 使用精确模型
    calc = PreciseSlewCalculator(config, use_precise=True)
    result = calc.calculate_slew_maneuver(prev_attitude, target_attitude)

    # 回退到简化模型
    calc = PreciseSlewCalculator(config, use_precise=False)
    slew_time = calc.calculate_slew_time(slew_angle)
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .attitude_types import (
    AttitudeState, Quaternion, AngularVelocity,
    InertiaTensor, MomentumState
)
from .rigid_body_dynamics import RigidBodyDynamics, DynamicsConfig
from .trajectory_planner import (
    TimeOptimalTrajectoryPlanner, TrajectoryPlannerConfig, Trajectory
)
from .energy_model import EnergyConsumptionModel, EnergyReport, ReactionWheelConfig
from .momentum_manager import MomentumManagementSystem, MomentumFeasibilityResult

logger = logging.getLogger(__name__)


@dataclass
class SatelliteDynamicsConfig:
    """卫星动力学配置

    包含精确姿态机动计算所需的所有物理参数。
    """
    # 惯性特性
    inertia_tensor: InertiaTensor = InertiaTensor.diagonal(100.0, 80.0, 60.0)

    # 控制约束
    max_control_torque: float = 0.5  # Nm
    max_angular_velocity: float = 3.0  # deg/s

    # 飞轮配置
    reaction_wheels: ReactionWheelConfig = ReactionWheelConfig()

    # 动力学配置
    dynamics_config: DynamicsConfig = DynamicsConfig()
    trajectory_config: TrajectoryPlannerConfig = TrajectoryPlannerConfig()

    # 简化模型参数 (用于回退)
    max_slew_rate: float = 3.0  # deg/s
    max_slew_angle: float = 45.0  # deg
    settling_time: float = 5.0  # s

    @property
    def effective_inertia(self) -> float:
        """计算有效惯性矩 (简化计算用)"""
        return np.mean([
            self.inertia_tensor.Ixx,
            self.inertia_tensor.Iyy,
            self.inertia_tensor.Izz
        ])


@dataclass
class SlewTimeResult:
    """机动时间结果

    与简化模型兼容的返回格式。
    """
    total_time: float
    acceleration_time: float = 0.0
    coast_time: float = 0.0
    deceleration_time: float = 0.0
    settling_time: float = 5.0
    max_angular_velocity: float = 0.0
    trajectory_profile: Optional[Trajectory] = None


@dataclass
class PreciseManeuverResult:
    """精确机动分析结果

    包含完整的机动分析信息。
    """
    feasible: bool
    total_time: float
    trajectory: Trajectory
    energy_consumption: float
    momentum_margin: float
    final_momentum: np.ndarray
    torque_profile: np.ndarray

    # 详细结果
    momentum_check: Optional[MomentumFeasibilityResult] = None
    energy_report: Optional[EnergyReport] = None


class PreciseSlewCalculator:
    """精确姿态机动计算器

    与现有 SlewCalculator 接口兼容的精确姿态机动计算器。
    支持通过 use_precise 参数切换模型。
    """

    def __init__(
        self,
        config: SatelliteDynamicsConfig,
        use_precise: bool = True
    ):
        """初始化精确机动计算器

        Args:
            config: 卫星动力学配置
            use_precise: 是否使用精确模型 (False则使用简化模型)
        """
        self.config = config
        self.use_precise = use_precise

        if use_precise:
            # 初始化精确模型组件
            self.dynamics = RigidBodyDynamics(
                inertia_tensor=config.inertia_tensor,
                config=config.dynamics_config
            )
            self.trajectory_planner = TimeOptimalTrajectoryPlanner(
                config=config.trajectory_config
            )
            self.momentum_mgr = MomentumManagementSystem(
                num_wheels=config.reaction_wheels.num_wheels,
                wheel_inertia=config.reaction_wheels.wheel_inertia,
                max_wheel_speed=config.reaction_wheels.max_speed
            )
            self.energy_model = EnergyConsumptionModel(
                wheel_config=config.reaction_wheels
            )
            logger.debug("PreciseSlewCalculator initialized with precise model")
        else:
            logger.debug("PreciseSlewCalculator initialized with simple model")

    def calculate_slew_time(
        self,
        slew_angle: float,
        q_start: Optional[Quaternion] = None,
        q_end: Optional[Quaternion] = None
    ) -> SlewTimeResult:
        """计算机动时间 - 与 SlewCalculator 兼容的接口

        Args:
            slew_angle: 机动角度 (度)
            q_start: 起始姿态 (精确模型需要)
            q_end: 目标姿态 (精确模型需要)

        Returns:
            机动时间结果
        """
        if not self.use_precise or q_start is None or q_end is None:
            # 回退到简化计算
            return self._simple_slew_time(slew_angle)

        # 使用精确模型
        trajectory = self.trajectory_planner.plan_trajectory(
            q_start=q_start,
            q_end=q_end,
            effective_inertia=self.config.effective_inertia
        )

        return SlewTimeResult(
            total_time=trajectory.total_time,
            acceleration_time=trajectory.acceleration_time,
            coast_time=trajectory.coast_time,
            deceleration_time=trajectory.deceleration_time,
            settling_time=trajectory.settling_time,
            max_angular_velocity=trajectory.max_angular_velocity,
            trajectory_profile=trajectory
        )

    def _simple_slew_time(self, slew_angle: float) -> SlewTimeResult:
        """简化模型计算机动时间

        与原有 SlewCalculator 一致的计算方式。

        Args:
            slew_angle: 机动角度 (度)

        Returns:
            机动时间结果
        """
        # 限制在最大角度内
        effective_angle = min(slew_angle, self.config.max_slew_angle)

        # 简化公式
        slew_time = effective_angle / self.config.max_slew_rate
        total_time = slew_time + self.config.settling_time

        return SlewTimeResult(
            total_time=total_time,
            settling_time=self.config.settling_time,
            max_angular_velocity=self.config.max_slew_rate
        )

    def calculate_slew_maneuver(
        self,
        prev_attitude: AttitudeState,
        target_attitude: AttitudeState,
        current_time: Optional[datetime] = None
    ) -> PreciseManeuverResult:
        """完整机动分析 - 精确模型的核心方法

        执行完整的机动分析，包括轨迹规划、动量检查、能量计算。

        Args:
            prev_attitude: 前一姿态状态
            target_attitude: 目标姿态状态
            current_time: 当前时间

        Returns:
            精确机动分析结果
        """
        if not self.use_precise:
            raise ValueError(
                "calculate_slew_maneuver requires precise model. "
                "Set use_precise=True when initializing."
            )

        # 1. 规划最优轨迹
        trajectory = self.trajectory_planner.plan_trajectory(
            q_start=prev_attitude.quaternion,
            q_end=target_attitude.quaternion,
            effective_inertia=self.config.effective_inertia
        )

        # 2. 检查动量可行性
        momentum_check = self.momentum_mgr.check_momentum_feasibility(
            trajectory=trajectory,
            current_momentum=prev_attitude.momentum or MomentumState(
                wheel_speeds=np.zeros(self.config.reaction_wheels.num_wheels),
                wheel_inertias=np.full(
                    self.config.reaction_wheels.num_wheels,
                    self.config.reaction_wheels.wheel_inertia
                ),
                total_momentum=np.zeros(3)
            )
        )

        # 3. 计算能量消耗
        energy_report = self.energy_model.compute_energy(trajectory)

        # 4. 计算机控制力矩曲线
        torque_profile = self._compute_torque_profile(trajectory)

        # 5. 判断可行性
        feasible = momentum_check.feasible

        return PreciseManeuverResult(
            feasible=feasible,
            total_time=trajectory.total_time,
            trajectory=trajectory,
            energy_consumption=energy_report.total_energy,
            momentum_margin=momentum_check.margin,
            final_momentum=momentum_check.final_momentum,
            torque_profile=torque_profile,
            momentum_check=momentum_check,
            energy_report=energy_report
        )

    def _compute_torque_profile(
        self,
        trajectory: Trajectory,
        num_points: int = 100
    ) -> np.ndarray:
        """计算机控制力矩曲线

        Args:
            trajectory: 机动轨迹
            num_points: 采样点数

        Returns:
            力矩数组 (Nm)
        """
        time_points = np.linspace(0, trajectory.total_time, num_points)
        torque_profile = np.zeros(num_points)

        for i, t in enumerate(time_points):
            torque = self.trajectory_planner.compute_control_torque(
                trajectory=trajectory,
                effective_inertia=self.config.effective_inertia,
                time=t
            )
            torque_profile[i] = np.linalg.norm([
                torque.x, torque.y, torque.z
            ])

        return torque_profile

    def check_momentum_feasibility(
        self,
        trajectory: Trajectory,
        current_momentum: MomentumState
    ) -> MomentumFeasibilityResult:
        """检查动量可行性

        Args:
            trajectory: 机动轨迹
            current_momentum: 当前动量状态

        Returns:
            可行性结果
        """
        if not self.use_precise:
            raise ValueError("Requires precise model")

        return self.momentum_mgr.check_momentum_feasibility(
            trajectory, current_momentum
        )

    def compute_energy_consumption(
        self,
        trajectory: Trajectory,
        initial_momentum: Optional[MomentumState] = None
    ) -> EnergyReport:
        """计算能量消耗

        Args:
            trajectory: 机动轨迹
            initial_momentum: 初始动量状态

        Returns:
            能量消耗报告
        """
        if not self.use_precise:
            raise ValueError("Requires precise model")

        return self.energy_model.compute_energy(trajectory, initial_momentum)

    def interpolate_attitude(
        self,
        trajectory: Trajectory,
        time: float
    ) -> Quaternion:
        """插值计算中间姿态

        Args:
            trajectory: 机动轨迹
            time: 时间点

        Returns:
            姿态四元数
        """
        if not self.use_precise:
            raise ValueError("Requires precise model")

        return self.trajectory_planner.interpolate_attitude(trajectory, time)

    @property
    def max_slew_angle(self) -> float:
        """最大机动角度 (与简化模型兼容)"""
        return self.config.max_slew_angle

    @property
    def max_slew_rate(self) -> float:
        """最大机动速率 (与简化模型兼容)"""
        return self.config.max_slew_rate

    @property
    def settling_time(self) -> float:
        """稳定时间 (与简化模型兼容)"""
        return self.config.settling_time
