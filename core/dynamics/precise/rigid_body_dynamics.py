"""
刚体动力学模型

实现卫星刚体动力学方程，包括：
1. 四元数运动学方程
2. 欧拉动力学方程
3. 状态积分器
4. 干扰力矩模型

参考: "Spacecraft Dynamics and Control" by Sidi
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import logging

from .attitude_types import (
    Quaternion, AngularVelocity, ControlTorque,
    AttitudeState, InertiaTensor
)

logger = logging.getLogger(__name__)


@dataclass
class DynamicsConfig:
    """动力学模型配置"""
    # 积分器参数
    integration_step: float = 0.1  # 积分步长 (秒)
    max_integration_time: float = 300.0  # 最大积分时间 (秒)

    # 数值精度
    position_tolerance: float = 1e-9
    velocity_tolerance: float = 1e-9

    # 干扰力矩开关
    include_gravity_gradient: bool = False
    include_aerodynamic: bool = False
    include_magnetic: bool = False
    include_solar_pressure: bool = False


class RigidBodyDynamics:
    """卫星刚体动力学模型

    状态方程:
        q̇ = 0.5 * q ⊗ ω
        I·ω̇ + ω×(I·ω) = T_ctrl + T_dist

    其中:
        q: 姿态四元数 [w, x, y, z]
        ω: 角速度矢量 [ωx, ωy, ωz] (rad/s)
        I: 惯性张量矩阵 (3×3)
        T_ctrl: 控制力矩 (Nm)
        T_dist: 干扰力矩 (Nm)
    """

    def __init__(
        self,
        inertia_tensor: InertiaTensor,
        config: Optional[DynamicsConfig] = None
    ):
        """初始化刚体动力学模型

        Args:
            inertia_tensor: 卫星惯性张量
            config: 动力学配置参数
        """
        self.inertia = inertia_tensor
        self.I_matrix = inertia_tensor.to_matrix()
        self.I_inv = np.linalg.inv(self.I_matrix)

        self.config = config or DynamicsConfig()

        # 验证惯性矩阵
        self._validate_inertia()

        logger.debug(
            f"RigidBodyDynamics initialized: "
            f"Ixx={inertia_tensor.Ixx:.2f}, "
            f"Iyy={inertia_tensor.Iyy:.2f}, "
            f"Izz={inertia_tensor.Izz:.2f} kg·m²"
        )

    def _validate_inertia(self) -> None:
        """验证惯性张量的有效性"""
        # 检查正定性
        eigenvalues = np.linalg.eigvals(self.I_matrix)
        if np.any(eigenvalues <= 0):
            raise ValueError(
                f"Inertia tensor must be positive definite. "
                f"Eigenvalues: {eigenvalues}"
            )

        # 三角不等式检查
        Ixx, Iyy, Izz = self.inertia.Ixx, self.inertia.Iyy, self.inertia.Izz
        if not (Ixx + Iyy >= Izz and Ixx + Izz >= Iyy and Iyy + Izz >= Ixx):
            logger.warning(
                "Inertia tensor may violate triangle inequality"
            )

    def equations_of_motion(
        self,
        state: np.ndarray,
        control_torque: np.ndarray,
        disturbance_torque: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """刚体动力学方程

        Args:
            state: 状态向量 [q_w, q_x, q_y, q_z, ω_x, ω_y, ω_z]
            control_torque: 控制力矩 [Tx, Ty, Tz] (Nm)
            disturbance_torque: 干扰力矩 [Tx, Ty, Tz] (Nm)

        Returns:
            状态导数 [q̇_w, q̇_x, q̇_y, q̇_z, ω̇_x, ω̇_y, ω̇_z]
        """
        # 提取四元数和角速度
        q = state[0:4]  # [w, x, y, z]
        omega = state[4:7]  # [ωx, ωy, ωz]

        # 四元数运动学: q̇ = 0.5 * q ⊗ [0, ω]
        q_dot = 0.5 * self._quaternion_multiply(q, np.array([0, *omega]))

        # 欧拉动力学: I·ω̇ = T - ω×(I·ω)
        I_omega = self.I_matrix @ omega
        gyroscopic_term = np.cross(omega, I_omega)

        total_torque = control_torque
        if disturbance_torque is not None:
            total_torque = total_torque + disturbance_torque

        omega_dot = self.I_inv @ (total_torque - gyroscopic_term)

        return np.concatenate([q_dot, omega_dot])

    def _quaternion_multiply(
        self,
        q1: np.ndarray,
        q2: np.ndarray
    ) -> np.ndarray:
        """四元数乘法

        Args:
            q1: 四元数 [w, x, y, z]
            q2: 四元数 [w, x, y, z]

        Returns:
            乘积四元数
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def integrate_state(
        self,
        initial_state: AttitudeState,
        control_profile: Callable[[float], ControlTorque],
        duration: float,
        disturbance_torque: Optional[Callable[[float], np.ndarray]] = None
    ) -> Tuple[AttitudeState, np.ndarray]:
        """数值积分计算状态演化

        Args:
            initial_state: 初始姿态状态
            control_profile: 控制力矩函数 T(t)
            duration: 积分时长 (秒)
            disturbance_torque: 干扰力矩函数 T_dist(t)

        Returns:
            (final_state, state_history)
        """
        # 初始状态向量
        q0 = initial_state.quaternion.to_array()
        omega0 = initial_state.angular_velocity.to_array()
        state = np.concatenate([q0, omega0])

        # 积分参数
        dt = self.config.integration_step
        num_steps = int(duration / dt) + 1

        # 状态历史记录
        state_history = np.zeros((num_steps, 7))
        state_history[0] = state

        # Runge-Kutta 4阶积分
        for i in range(1, num_steps):
            t = (i - 1) * dt

            # 获取控制力矩
            T_ctrl = control_profile(t).to_array()

            # 获取干扰力矩
            T_dist = None
            if disturbance_torque is not None:
                T_dist = disturbance_torque(t)

            # RK4积分
            k1 = self.equations_of_motion(state, T_ctrl, T_dist)
            k2 = self.equations_of_motion(
                state + 0.5*dt*k1, T_ctrl, T_dist
            )
            k3 = self.equations_of_motion(
                state + 0.5*dt*k2, T_ctrl, T_dist
            )
            k4 = self.equations_of_motion(
                state + dt*k3, T_ctrl, T_dist
            )

            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # 四元数归一化
            state[0:4] = state[0:4] / np.linalg.norm(state[0:4])

            state_history[i] = state

        # 构造最终状态
        final_state = AttitudeState(
            quaternion=Quaternion.from_array(state[0:4]),
            angular_velocity=AngularVelocity.from_array(state[4:7]),
            timestamp=initial_state.timestamp,  # 需要在外部更新
            momentum=initial_state.momentum
        )

        return final_state, state_history

    def compute_gravity_gradient_torque(
        self,
        attitude: Quaternion,
        orbital_position: np.ndarray,
        orbital_angular_velocity: float
    ) -> np.ndarray:
        """计算重力梯度力矩

        T_gg = 3·ω₀² · r̂ × (I · r̂)

        Args:
            attitude: 卫星姿态四元数
            orbital_position: 卫星位置矢量 (m)
            orbital_angular_velocity: 轨道角速度 (rad/s)

        Returns:
            重力梯度力矩 (Nm)
        """
        if not self.config.include_gravity_gradient:
            return np.zeros(3)

        # 轨道半径方向单位矢量
        r_unit = orbital_position / np.linalg.norm(orbital_position)

        # 转换到本体坐标系
        r_body = self._rotate_vector_by_quaternion(r_unit, attitude)

        # 重力梯度力矩
        T_gg = 3 * orbital_angular_velocity**2 * np.cross(
            r_body,
            self.I_matrix @ r_body
        )

        return T_gg

    def _rotate_vector_by_quaternion(
        self,
        vector: np.ndarray,
        q: Quaternion
    ) -> np.ndarray:
        """使用四元数旋转向量

        v' = q ⊗ [0, v] ⊗ q*
        """
        q_arr = q.to_array()
        q_conj = np.array([q_arr[0], -q_arr[1], -q_arr[2], -q_arr[3]])

        v_quat = np.array([0, *vector])
        temp = self._quaternion_multiply(q_arr, v_quat)
        result = self._quaternion_multiply(temp, q_conj)

        return result[1:4]

    def compute_kinetic_energy(self, angular_velocity: AngularVelocity) -> float:
        """计算旋转动能

        E = 0.5 · ωᵀ · I · ω

        Args:
            angular_velocity: 角速度

        Returns:
            动能 (Joules)
        """
        omega = angular_velocity.to_array()
        return 0.5 * omega @ self.I_matrix @ omega

    def compute_angular_momentum(
        self,
        angular_velocity: AngularVelocity
    ) -> np.ndarray:
        """计算角动量

        h = I · ω

        Args:
            angular_velocity: 角速度

        Returns:
            角动量矢量 [hx, hy, hz] (Nms)
        """
        return self.I_matrix @ angular_velocity.to_array()
