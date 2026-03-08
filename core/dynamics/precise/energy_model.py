"""
能量消耗精确模型

实现姿态机动的能量消耗精确计算，包括：
1. 反作用飞轮加/减速能耗
2. 电机效率损失
3. 电子设备功耗
4. 热损耗

电机模型:
    P_motor = τ·ω / η  (加速)
    P_motor = τ·ω·η    (减速, 能量回收)

其中:
    τ: 电机力矩
    ω: 飞轮转速
    η: 电机效率
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import logging

from .attitude_types import MomentumState
from .trajectory_planner import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class ReactionWheelConfig:
    """反作用飞轮配置

    Attributes:
        num_wheels: 飞轮数量
        wheel_inertia: 单个飞轮转动惯量 (kg·m²)
        max_speed: 最大转速 (rad/s)
        max_torque: 最大输出力矩 (Nm)
        motor_efficiency: 电机效率 [0, 1]
        static_friction: 静摩擦力矩 (Nm)
        viscous_damping: 粘性阻尼系数 (Nms/rad)
    """
    num_wheels: int = 4
    wheel_inertia: float = 0.01  # kg·m²
    max_speed: float = 600.0  # rad/s (~5730 RPM)
    max_torque: float = 0.2  # Nm
    motor_efficiency: float = 0.85
    static_friction: float = 0.001  # Nm
    viscous_damping: float = 1e-5  # Nms/rad

    def compute_max_angular_momentum(self) -> float:
        """计算单个飞轮最大角动量"""
        return self.wheel_inertia * self.max_speed


@dataclass
class EnergyReport:
    """能量消耗报告"""
    # 总能量
    total_energy: float  # Joules

    # 飞轮储能变化
    kinetic_change: float  # J (正值表示储能增加)

    # 能量损失
    motor_loss: float  # J (电机发热)
    friction_loss: float  # J (摩擦损失)

    # 净电能消耗
    electrical_energy: float  # J

    # 功率统计
    average_power: float  # W
    peak_power: float  # W

    # 效率
    effective_efficiency: float  # 有效效率

    def __post_init__(self):
        """计算派生指标"""
        if abs(self.electrical_energy) > 1e-10:
            self.effective_efficiency = abs(self.kinetic_change) / self.electrical_energy
        else:
            self.effective_efficiency = 1.0


class EnergyConsumptionModel:
    """姿态机动能量消耗精确模型

    基于反作用飞轮的动力学模型，精确计算机动过程中的电能消耗。
    考虑电机效率、摩擦损耗和飞轮储能变化。
    """

    def __init__(
        self,
        wheel_config: ReactionWheelConfig,
        electronics_power: float = 10.0  # 电子设备功耗 (W)
    ):
        """初始化能量模型

        Args:
            wheel_config: 飞轮配置
            electronics_power: 姿态控制系统电子功耗 (W)
        """
        self.wheel_config = wheel_config
        self.electronics_power = electronics_power

    def compute_energy(
        self,
        trajectory: Trajectory,
        initial_momentum: Optional[MomentumState] = None
    ) -> EnergyReport:
        """计算机动的能量消耗

        Args:
            trajectory: 机动轨迹
            initial_momentum: 初始动量状态

        Returns:
            能量消耗报告
        """
        if trajectory.velocity_profile is None:
            raise ValueError("Trajectory must have velocity profile")

        profile = trajectory.velocity_profile

        # 1. 计算飞轮速度变化
        delta_omega_rw = self._compute_wheel_speed_change(trajectory)

        # 2. 计算动能变化
        delta_kinetic = self._compute_kinetic_energy_change(delta_omega_rw)

        # 3. 计算电机损耗
        motor_loss = self._compute_motor_losses(
            trajectory, delta_omega_rw
        )

        # 4. 计算摩擦损耗
        friction_loss = self._compute_friction_loss(trajectory)

        # 5. 计算总电能消耗
        electrical_energy = abs(delta_kinetic) + motor_loss + friction_loss

        # 6. 计算功率统计
        avg_power = electrical_energy / trajectory.total_time
        peak_power = self._estimate_peak_power(trajectory)

        # 7. 加电子设备功耗
        electronics_energy = self.electronics_power * trajectory.total_time
        total_energy = electrical_energy + electronics_energy

        return EnergyReport(
            total_energy=total_energy,
            kinetic_change=delta_kinetic,
            motor_loss=motor_loss,
            friction_loss=friction_loss,
            electrical_energy=electrical_energy,
            average_power=avg_power,
            peak_power=peak_power,
            effective_efficiency=0.0  # 将在__post_init__中计算
        )

    def _compute_wheel_speed_change(
        self,
        trajectory: Trajectory
    ) -> float:
        """计算飞轮转速变化量

        基于角动量交换原理:
        Δh_satellite = -Δh_wheels

        Args:
            trajectory: 机动轨迹

        Returns:
            飞轮转速变化 (rad/s)
        """
        # 机动引起的卫星角动量变化
        delta_h_sat = self._compute_satellite_momentum_change(trajectory)

        # 假设动量均匀分配到所有飞轮
        num_wheels = self.wheel_config.num_wheels

        # 每个飞轮的角动量变化
        delta_h_per_wheel = delta_h_sat / num_wheels

        # 转换为转速变化
        delta_omega = delta_h_per_wheel / self.wheel_config.wheel_inertia

        return delta_omega

    def _compute_satellite_momentum_change(
        self,
        trajectory: Trajectory
    ) -> float:
        """计算卫星角动量变化

        基于刚体动力学:
        h = I · ω

        Args:
            trajectory: 机动轨迹

        Returns:
            角动量变化大小 (Nms)
        """
        # 使用最大角速度估算角动量变化
        # 简化为标量计算，实际应用中应为矢量
        max_omega_rad = np.radians(trajectory.max_angular_velocity)

        # 假设有效惯性 (需要根据实际卫星惯性张量计算)
        effective_inertia = 100.0  # kg·m² (示例值)

        delta_h = effective_inertia * max_omega_rad

        return delta_h

    def _compute_kinetic_energy_change(
        self,
        delta_omega_rw: float
    ) -> float:
        """计算飞轮动能变化

        E_k = 0.5 · J · ω²

        Args:
            delta_omega_rw: 飞轮转速变化

        Returns:
            动能变化 (J), 正值表示储能增加
        """
        J = self.wheel_config.wheel_inertia
        n = self.wheel_config.num_wheels

        # 假设初始转速为平均转速的50%
        omega_initial = self.wheel_config.max_speed * 0.5
        omega_final = omega_initial + delta_omega_rw

        # 动能变化
        delta_E_per_wheel = 0.5 * J * (omega_final**2 - omega_initial**2)
        delta_E_total = n * delta_E_per_wheel

        return delta_E_total

    def _compute_motor_losses(
        self,
        trajectory: Trajectory,
        delta_omega_rw: float
    ) -> float:
        """计算电机损耗

        电机损耗包括:
        1. 铜损 (I²R)
        2. 铁损 (磁滞和涡流)
        3. 开关损耗

        简化为效率模型:
        P_loss = P_in - P_out = P_out · (1-η)/η

        Args:
            trajectory: 机动轨迹
            delta_omega_rw: 飞轮转速变化

        Returns:
            电机损耗 (J)
        """
        eta = self.wheel_config.motor_efficiency

        # 飞轮动能变化
        delta_kinetic = abs(self._compute_kinetic_energy_change(delta_omega_rw))

        # 电机输入电能
        if delta_omega_rw > 0:
            # 加速: 电机输出给飞轮
            P_out = delta_kinetic
            P_in = P_out / eta
        else:
            # 减速: 飞轮回馈给电机 (能量回收)
            P_in = delta_kinetic * eta
            P_out = delta_kinetic

        motor_loss = abs(P_in - P_out)

        return motor_loss

    def _compute_friction_loss(
        self,
        trajectory: Trajectory
    ) -> float:
        """计算摩擦损耗

        包括:
        1. 轴承静摩擦
        2. 轴承粘性摩擦
        3. 风阻 (如果有)

        Args:
            trajectory: 机动轨迹

        Returns:
            摩擦损耗 (J)
        """
        t_total = trajectory.total_time
        omega_avg = np.radians(trajectory.max_angular_velocity) / 2

        # 粘性摩擦功率
        P_viscous = self.wheel_config.viscous_damping * omega_avg**2

        # 静摩擦功率 (简化模型)
        P_static = self.wheel_config.static_friction * omega_avg

        # 总摩擦损耗
        friction_loss = (P_viscous + P_static) * t_total
        friction_loss *= self.wheel_config.num_wheels

        return friction_loss

    def _estimate_peak_power(
        self,
        trajectory: Trajectory
    ) -> float:
        """估算峰值功率

        Args:
            trajectory: 机动轨迹

        Returns:
            峰值功率 (W)
        """
        # 峰值功率发生在最大角加速度时刻
        max_alpha_rad = np.radians(
            trajectory.velocity_profile.angular_acceleration.max()
        )

        # 假设有效惯性
        effective_inertia = 100.0  # kg·m²

        # 峰值力矩
        max_torque = effective_inertia * max_alpha_rad

        # 峰值角速度
        max_omega = np.radians(trajectory.max_angular_velocity)

        # 峰值机械功率
        peak_mech_power = max_torque * max_omega

        # 考虑电机效率
        peak_elec_power = peak_mech_power / self.wheel_config.motor_efficiency

        # 加上电子功耗
        total_peak_power = peak_elec_power + self.electronics_power

        return total_peak_power

    def compute_power_profile(
        self,
        trajectory: Trajectory,
        num_points: int = 100
    ) -> np.ndarray:
        """计算功率曲线

        Args:
            trajectory: 机动轨迹
            num_points: 采样点数

        Returns:
            功率数组 (W)
        """
        profile = trajectory.velocity_profile
        dt = np.diff(profile.time_points)

        power = np.zeros(len(profile.time_points))

        for i in range(len(profile.time_points)):
            omega = np.radians(profile.angular_velocity[i])
            alpha = np.radians(profile.angular_acceleration[i])

            # 机械功率
            if abs(alpha) > 1e-6:
                # 加速或减速
                P_mech = 100.0 * abs(alpha * omega)  # 简化模型
                P_elec = P_mech / self.wheel_config.motor_efficiency
            else:
                # 匀速, 只需克服摩擦
                P_elec = self.wheel_config.static_friction * omega

            power[i] = P_elec + self.electronics_power

        return power
