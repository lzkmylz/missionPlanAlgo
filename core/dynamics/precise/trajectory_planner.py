"""
时间最优轨迹规划器

基于Pontryagin极大值原理的时间最优姿态机动轨迹规划。
采用bang-bang控制策略，考虑卫星刚体动力学约束。

算法:
1. 计算姿态误差四元数 (最短路径)
2. 确定特征轴旋转 (eigenaxis rotation)
3. bang-bang控制参数计算
4. 生成最优速度/力矩剖面

参考: "Time-Optimal Reorientation of a Rigid Body" by Bilimoria & Wie
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

from .attitude_types import Quaternion, AngularVelocity, ControlTorque

logger = logging.getLogger(__name__)


class VelocityProfileType(Enum):
    """速度剖面类型"""
    TRIANGULAR = "triangular"  # 纯加速-减速，无匀速
    TRAPEZOIDAL = "trapezoidal"  # 加速-匀速-减速


@dataclass
class TrajectoryProfile:
    """轨迹剖面参数"""
    time_points: np.ndarray      # 时间点数组
    angular_velocity: np.ndarray  # 角速度剖面 (deg/s)
    angular_acceleration: np.ndarray  # 角加速度剖面 (deg/s²)
    control_torque: np.ndarray   # 控制力矩剖面 (Nm)


@dataclass
class Trajectory:
    """时间最优轨迹"""
    # 旋转参数
    rotation_axis: np.ndarray    # 旋转轴 (单位向量)
    rotation_angle: float        # 旋转角度 (度)

    # 时间参数
    total_time: float           # 总机动时间 (秒)
    acceleration_time: float    # 加速时间 (秒)
    coast_time: float          # 匀速时间 (秒)
    deceleration_time: float   # 减速时间 (秒)
    settling_time: float       # 稳定时间 (秒)

    # 速度参数
    max_angular_velocity: float  # 最大角速度 (deg/s)
    profile_type: VelocityProfileType

    # 轨迹剖面
    velocity_profile: Optional[TrajectoryProfile] = None

    # 边界条件
    initial_quaternion: Optional[Quaternion] = None
    final_quaternion: Optional[Quaternion] = None

    @property
    def switch_times(self) -> List[float]:
        """控制切换时间点"""
        times = [self.acceleration_time]
        if self.coast_time > 0:
            times.append(self.acceleration_time + self.coast_time)
        return times


@dataclass
class TrajectoryPlannerConfig:
    """轨迹规划器配置"""
    # 控制约束
    max_control_torque: float = 0.5  # 最大控制力矩 (Nm)

    # 速度约束
    max_angular_velocity: float = 3.0  # 最大角速度 (deg/s)

    # 加速度约束
    max_angular_acceleration: float = 0.5  # 最大角加速度 (deg/s²)

    # 稳定时间
    settling_time: float = 5.0  # 姿态稳定时间 (秒)

    # 数值精度
    angle_tolerance: float = 1e-6  # 角度容差 (度)

    def compute_max_acceleration(self, inertia: float) -> float:
        """根据力矩和惯性计算最大角加速度

        α_max = T_max / I

        Args:
            inertia: 绕旋转轴的惯性矩 (kg·m²)

        Returns:
            最大角加速度 (rad/s²)
        """
        return self.max_control_torque / inertia


class TimeOptimalTrajectoryPlanner:
    """时间最优轨迹规划器

    规划满足刚体动力学约束的时间最优机动轨迹。
    采用bang-bang控制：最大加速度 → 可能匀速 → 最大减速度
    """

    def __init__(self, config: Optional[TrajectoryPlannerConfig] = None):
        """初始化轨迹规划器

        Args:
            config: 规划器配置
        """
        self.config = config or TrajectoryPlannerConfig()

    def plan_trajectory(
        self,
        q_start: Quaternion,
        q_end: Quaternion,
        effective_inertia: float,
        current_angular_velocity: Optional[np.ndarray] = None
    ) -> Trajectory:
        """规划时间最优轨迹

        Args:
            q_start: 起始姿态四元数
            q_end: 目标姿态四元数
            effective_inertia: 绕旋转轴的有效惯性矩 (kg·m²)
            current_angular_velocity: 当前角速度 (rad/s)

        Returns:
            最优轨迹对象
        """
        # 1. 计算姿态误差
        q_error = self._compute_attitude_error(q_start, q_end)

        # 2. 提取旋转轴和角度
        rotation_axis, rotation_angle = self._extract_axis_angle(q_error)

        # 3. 计算最大角加速度
        alpha_max = self.config.compute_max_acceleration(effective_inertia)
        alpha_max_deg = np.degrees(alpha_max)  # 转为度/s²

        # 4. bang-bang控制参数计算
        trajectory_params = self._compute_bang_bang_parameters(
            rotation_angle=rotation_angle,
            max_angular_velocity=self.config.max_angular_velocity,
            max_acceleration=alpha_max_deg
        )

        # 5. 生成速度剖面
        velocity_profile = self._generate_velocity_profile(
            trajectory_params, alpha_max_deg
        )

        # 6. 构造轨迹对象
        trajectory = Trajectory(
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            total_time=trajectory_params['total_time'] + self.config.settling_time,
            acceleration_time=trajectory_params['acceleration_time'],
            coast_time=trajectory_params['coast_time'],
            deceleration_time=trajectory_params['deceleration_time'],
            settling_time=self.config.settling_time,
            max_angular_velocity=trajectory_params['max_velocity'],
            profile_type=trajectory_params['profile_type'],
            velocity_profile=velocity_profile,
            initial_quaternion=q_start,
            final_quaternion=q_end
        )

        logger.debug(
            f"Trajectory planned: angle={rotation_angle:.2f}°, "
            f"time={trajectory.total_time:.2f}s, "
            f"type={trajectory.profile_type.value}"
        )

        return trajectory

    def _compute_attitude_error(
        self,
        q_start: Quaternion,
        q_end: Quaternion
    ) -> Quaternion:
        """计算姿态误差四元数

        q_error = q_end ⊗ q_start⁻¹

        选择最短路径（如果旋转角 > 180°，取共轭）

        Args:
            q_start: 起始姿态
            q_end: 目标姿态

        Returns:
            误差四元数
        """
        # 起始四元数共轭
        q_start_conj = q_start.conjugate()

        # 误差四元数
        q_error = q_end * q_start_conj

        # 归一化
        q_error = q_error.normalize()

        # 选择最短路径（如果w < 0，取共轭）
        if q_error.w < 0:
            q_error = Quaternion(
                w=-q_error.w,
                x=-q_error.x,
                y=-q_error.y,
                z=-q_error.z
            )

        return q_error

    def _extract_axis_angle(self, q: Quaternion) -> Tuple[np.ndarray, float]:
        """从四元数提取旋转轴和角度

        Args:
            q: 姿态四元数

        Returns:
            (rotation_axis, rotation_angle_deg)
        """
        # 归一化
        q = q.normalize()

        # 旋转角度
        angle_rad = 2 * np.arccos(np.clip(q.w, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # 旋转轴
        sin_half_angle = np.sqrt(1 - q.w**2)

        if sin_half_angle < self.config.angle_tolerance:
            # 接近零旋转，任意轴
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([q.x, q.y, q.z]) / sin_half_angle
            axis = axis / np.linalg.norm(axis)

        return axis, angle_deg

    def _compute_bang_bang_parameters(
        self,
        rotation_angle: float,
        max_angular_velocity: float,
        max_acceleration: float
    ) -> dict:
        """计算bang-bang控制参数

        确定是三角形还是梯形速度剖面。

        Args:
            rotation_angle: 总旋转角度 (度)
            max_angular_velocity: 最大角速度 (度/s)
            max_acceleration: 最大角加速度 (度/s²)

        Returns:
            轨迹参数字典
        """
        # 计算达到最大速度所需的转角
        # θ_accel = 0.5 * α * t², t = ω_max / α
        # θ_accel = 0.5 * ω_max² / α
        theta_to_max_vel = 0.5 * max_angular_velocity**2 / max_acceleration

        # 判断速度剖面类型
        if 2 * theta_to_max_vel >= rotation_angle:
            # 三角形剖面：达不到最大速度
            profile_type = VelocityProfileType.TRIANGULAR

            # 切换时间
            t_accel = np.sqrt(rotation_angle / max_acceleration)
            t_coast = 0.0
            t_decel = t_accel

            # 实际最大速度
            actual_max_vel = max_acceleration * t_accel

        else:
            # 梯形剖面：加速-匀速-减速
            profile_type = VelocityProfileType.TRAPEZOIDAL

            t_accel = max_angular_velocity / max_acceleration
            theta_coast = rotation_angle - 2 * theta_to_max_vel
            t_coast = theta_coast / max_angular_velocity
            t_decel = t_accel

            actual_max_vel = max_angular_velocity

        return {
            'profile_type': profile_type,
            'acceleration_time': t_accel,
            'coast_time': t_coast,
            'deceleration_time': t_decel,
            'total_time': t_accel + t_coast + t_decel,
            'max_velocity': actual_max_vel
        }

    def _generate_velocity_profile(
        self,
        params: dict,
        max_acceleration: float,
        num_points: int = 100
    ) -> TrajectoryProfile:
        """生成速度剖面

        Args:
            params: 轨迹参数
            max_acceleration: 最大角加速度
            num_points: 采样点数

        Returns:
            轨迹剖面对象
        """
        total_time = params['total_time']
        t_accel = params['acceleration_time']
        t_coast = params['coast_time']
        t_decel = params['deceleration_time']
        max_vel = params['max_velocity']

        # 时间点
        time_points = np.linspace(0, total_time, num_points)

        velocity = np.zeros(num_points)
        acceleration = np.zeros(num_points)
        torque = np.zeros(num_points)

        for i, t in enumerate(time_points):
            if t <= t_accel:
                # 加速阶段
                velocity[i] = max_acceleration * t
                acceleration[i] = max_acceleration
                torque[i] = 1.0  # 归一化力矩方向
            elif t <= t_accel + t_coast:
                # 匀速阶段
                velocity[i] = max_vel
                acceleration[i] = 0.0
                torque[i] = 0.0
            elif t <= total_time:
                # 减速阶段
                t_decel_elapsed = t - t_accel - t_coast
                velocity[i] = max_vel - max_acceleration * t_decel_elapsed
                acceleration[i] = -max_acceleration
                torque[i] = -1.0
            else:
                velocity[i] = 0.0
                acceleration[i] = 0.0
                torque[i] = 0.0

        return TrajectoryProfile(
            time_points=time_points,
            angular_velocity=velocity,
            angular_acceleration=acceleration,
            control_torque=torque
        )

    def compute_control_torque(
        self,
        trajectory: Trajectory,
        effective_inertia: float,
        time: float
    ) -> ControlTorque:
        """计算指定时刻的控制力矩

        Args:
            trajectory: 轨迹对象
            effective_inertia: 有效惯性矩
            time: 时间 (秒)

        Returns:
            控制力矩
        """
        t = time
        t_accel = trajectory.acceleration_time
        t_coast = trajectory.coast_time

        # 确定控制阶段
        if t < t_accel:
            # 加速阶段
            torque_magnitude = self.config.max_control_torque
        elif t < t_accel + t_coast:
            # 匀速阶段
            torque_magnitude = 0.0
        elif t < trajectory.total_time - trajectory.settling_time:
            # 减速阶段
            torque_magnitude = -self.config.max_control_torque
        else:
            # 稳定阶段
            torque_magnitude = 0.0

        # 沿旋转轴施加力矩
        torque_vector = torque_magnitude * trajectory.rotation_axis

        return ControlTorque(
            x=torque_vector[0],
            y=torque_vector[1],
            z=torque_vector[2]
        )

    def interpolate_attitude(
        self,
        trajectory: Trajectory,
        time: float
    ) -> Quaternion:
        """插值计算指定时刻的姿态

        使用球面线性插值 (SLERP)

        Args:
            trajectory: 轨迹对象
            time: 时间 (秒)

        Returns:
            姿态四元数
        """
        if time <= 0:
            return trajectory.initial_quaternion

        if time >= trajectory.total_time:
            return trajectory.final_quaternion

        # 计算完成的旋转比例
        total_maneuver_time = (
            trajectory.total_time - trajectory.settling_time
        )

        if time >= total_maneuver_time:
            return trajectory.final_quaternion

        ratio = time / total_maneuver_time

        # SLERP插值
        return self._slerp(
            trajectory.initial_quaternion,
            trajectory.final_quaternion,
            ratio
        )

    def _slerp(
        self,
        q1: Quaternion,
        q2: Quaternion,
        t: float
    ) -> Quaternion:
        """球面线性插值

        Args:
            q1: 起始四元数
            q2: 结束四元数
            t: 插值参数 [0, 1]

        Returns:
            插值结果
        """
        # 转换为数组
        v1 = q1.to_array()
        v2 = q2.to_array()

        # 计算点积
        dot = np.dot(v1, v2)

        # 如果点积为负，反转一个四元数以取最短路径
        if dot < 0.0:
            v2 = -v2
            dot = -dot

        # 阈值
        DOT_THRESHOLD = 0.9995

        if dot > DOT_THRESHOLD:
            # 线性插值
            result = v1 + t * (v2 - v1)
            result = result / np.linalg.norm(result)
        else:
            # SLERP
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)

            theta = theta_0 * t
            sin_theta = np.sin(theta)

            s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0

            result = s1 * v1 + s2 * v2

        return Quaternion.from_array(result)
