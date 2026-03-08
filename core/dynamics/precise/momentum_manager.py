
"""
飞轮动量管理系统

管理反作用飞轮系统的角动量状态，确保机动后不会饱和。

功能:
1. 计算机动所需角动量交换
2. 预测机动后各飞轮转速
3. 检查动量包络约束
4. 动量卸载策略建议

飞轮构型支持:
- 金字塔构型 (4个飞轮)
- 正交构型 (3个飞轮)
- 五轮冗余构型
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

from .attitude_types import MomentumState
from .trajectory_planner import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class MomentumFeasibilityResult:
    """动量可行性检查结果"""
    feasible: bool  # 是否可行
    final_momentum: np.ndarray  # 机动后角动量矢量 [hx, hy, hz]
    final_wheel_speeds: np.ndarray  # 机动后各飞轮转速
    margin: float  # 最小动量裕度 (Nms)
    saturated_wheels: List[int]  # 饱和的飞轮索引
    recommendation: Optional[str] = None  # 建议


class MomentumManagementSystem:
    """飞轮动量管理系统

    跟踪和管理卫星反作用飞轮系统的角动量状态，
    确保姿态机动不会导致飞轮饱和。
    """

    # 标准飞轮构型矩阵
    # 每列代表一个飞轮的安装方向 (在卫星本体坐标系中)
    WHEEL_CONFIGURATIONS = {
        'pyramid_4': np.array([
            [1, -1, 0, 0],   # x分量
            [0, 0, 1, -1],   # y分量
            [1, 1, 1, 1]     # z分量 (都朝z方向倾斜)
        ]) / np.sqrt(2),

        'orthogonal_3': np.array([
            [1, 0, 0],   # x轴飞轮
            [0, 1, 0],   # y轴飞轮
            [0, 0, 1]    # z轴飞轮
        ]),

        'tetrahedron_4': np.array([
            [1, -1, -1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1]
        ]) / np.sqrt(3),
    }

    def __init__(
        self,
        num_wheels: int = 4,
        wheel_inertia: float = 0.01,  # kg·m²
        max_wheel_speed: float = 600.0,  # rad/s
        wheel_configuration: str = 'pyramid_4',
        momentum_envelope: Optional[float] = None
    ):
        """初始化动量管理系统

        Args:
            num_wheels: 飞轮数量
            wheel_inertia: 单个飞轮转动惯量
            max_wheel_speed: 最大转速
            wheel_configuration: 飞轮构型名称
            momentum_envelope: 总动量包络 (Nms)
        """
        self.num_wheels = num_wheels
        self.wheel_inertia = wheel_inertia
        self.max_wheel_speed = max_wheel_speed
        self.max_momentum_per_wheel = wheel_inertia * max_wheel_speed

        # 动量包络 (默认所有飞轮总和的80%)
        self.momentum_envelope = (
            momentum_envelope or
            num_wheels * self.max_momentum_per_wheel * 0.8
        )

        # 飞轮方向矩阵 (3 x num_wheels)
        self.wheel_directions = self._get_wheel_directions(
            wheel_configuration, num_wheels
        )

        logger.debug(
            f"MomentumManagementSystem initialized: "
            f"{num_wheels} wheels, max momentum={self.max_momentum_per_wheel:.3f} Nms"
        )

    def _get_wheel_directions(
        self,
        config_name: str,
        num_wheels: int
    ) -> np.ndarray:
        """获取飞轮方向矩阵

        Args:
            config_name: 构型名称
            num_wheels: 飞轮数量

        Returns:
            3 x num_wheels 方向矩阵
        """
        if config_name in self.WHEEL_CONFIGURATIONS:
            matrix = self.WHEEL_CONFIGURATIONS[config_name]
            if matrix.shape[1] == num_wheels:
                return matrix

        # 默认金字塔构型
        if num_wheels == 4:
            return self.WHEEL_CONFIGURATIONS['pyramid_4']
        elif num_wheels == 3:
            return self.WHEEL_CONFIGURATIONS['orthogonal_3']
        else:
            # 自定义构型: 均匀分布在球面上
            return self._generate_uniform_configuration(num_wheels)

    def _generate_uniform_configuration(self, n: int) -> np.ndarray:
        """生成均匀分布的飞轮构型"""
        # 简化的均匀分布算法
        directions = np.zeros((3, n))
        phi = np.pi * (3 - np.sqrt(5))  # 黄金角

        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i

            directions[0, i] = np.cos(theta) * radius
            directions[1, i] = np.sin(theta) * radius
            directions[2, i] = y

        return directions

    def check_momentum_feasibility(
        self,
        trajectory: Trajectory,
        current_momentum: MomentumState
    ) -> MomentumFeasibilityResult:
        """检查机动后的动量可行性

        Args:
            trajectory: 机动轨迹
            current_momentum: 当前动量状态

        Returns:
            可行性结果
        """
        # 1. 计算机动所需的角动量交换
        required_momentum = self._compute_required_momentum(trajectory)

        # 2. 预测机动后各飞轮转速
        final_speeds = self._predict_wheel_speeds(
            current_momentum, required_momentum
        )

        # 3. 检查转速约束
        saturated_wheels = []
        min_margin = float('inf')

        for i, speed in enumerate(final_speeds):
            margin = self.max_wheel_speed - abs(speed)
            min_margin = min(min_margin, margin)

            if abs(speed) > self.max_wheel_speed:
                saturated_wheels.append(i)

        # 4. 检查总动量包络
        final_momentum_vector = self.wheel_directions @ (
            final_speeds * self.wheel_inertia
        )
        total_momentum = np.linalg.norm(final_momentum_vector)
        envelope_ok = total_momentum < self.momentum_envelope

        # 5. 生成建议
        feasible = len(saturated_wheels) == 0 and envelope_ok
        recommendation = None

        if not feasible:
            if saturated_wheels:
                recommendation = (
                    f"飞轮 {saturated_wheels} 将达到饱和. "
                    "建议: 1) 延长机动时间降低峰值速度; "
                    "2) 执行动量卸载; 3) 调整机动顺序"
                )
            elif not envelope_ok:
                recommendation = (
                    f"总动量 {total_momentum:.2f} Nms 超过包络 "
                    f"{self.momentum_envelope:.2f} Nms. "
                    "建议执行动量卸载"
                )

        return MomentumFeasibilityResult(
            feasible=feasible,
            final_momentum=final_momentum_vector,
            final_wheel_speeds=final_speeds,
            margin=min_margin * self.wheel_inertia,  # 转换为动量单位
            saturated_wheels=saturated_wheels,
            recommendation=recommendation
        )

    def _compute_required_momentum(
        self,
        trajectory: Trajectory
    ) -> np.ndarray:
        """计算机动所需的角动量

        基于刚体动力学:
        h_required = ∫(I · α) dt ≈ I · Δω

        Args:
            trajectory: 机动轨迹

        Returns:
            所需角动量矢量 [hx, hy, hz] (Nms)
        """
        # 假设有效惯性 (实际应根据卫星惯性张量计算)
        effective_inertia = 100.0  # kg·m²

        # 最大角速度变化
        max_omega = np.radians(trajectory.max_angular_velocity)

        # 旋转轴方向
        rotation_axis = trajectory.rotation_axis

        # 角动量变化矢量
        delta_h = effective_inertia * max_omega * rotation_axis

        return delta_h

    def _predict_wheel_speeds(
        self,
        current_momentum: MomentumState,
        required_momentum: np.ndarray
    ) -> np.ndarray:
        """预测机动后的飞轮转速

        Args:
            current_momentum: 当前动量状态
            required_momentum: 所需角动量

        Returns:
            机动后各飞轮转速 (rad/s)
        """
        # 当前飞轮转速
        current_speeds = current_momentum.wheel_speeds

        # 需要的飞轮角动量变化 (与卫星角动量变化相反)
        wheel_momentum_change = -required_momentum

        # 分配到各飞轮 (伪逆分配)
        # Δh = D · J · Δω
        # Δω = (D · J)^+ · Δh
        D = self.wheel_directions
        J = self.wheel_inertia

        # 伪逆
        DJ = D @ np.diag([J] * self.num_wheels)
        DJ_pinv = np.linalg.pinv(DJ)

        delta_speeds = DJ_pinv @ wheel_momentum_change

        # 预测最终转速
        final_speeds = current_speeds + delta_speeds

        return final_speeds

    def compute_momentum_dumping_strategy(
        self,
        current_momentum: MomentumState,
        target_momentum: Optional[np.ndarray] = None
    ) -> dict:
        """计算动量卸载策略

        Args:
            current_momentum: 当前动量状态
            target_momentum: 目标动量状态 (默认零动量)

        Returns:
            卸载策略字典
        """
        if target_momentum is None:
            target_momentum = np.zeros(3)

        # 当前总动量
        current_h = self.wheel_directions @ (
            current_momentum.wheel_speeds * self.wheel_inertia
        )

        # 需要卸载的动量
        h_dump = current_h - target_momentum

        # 卸载时间估算 (假设使用磁力矩器)
        # τ_mag = m × B
        # 典型磁力矩器能产生约 0.1 Nms/s 的卸载速率
        dumping_rate = 0.1  # Nms/s
        estimated_time = np.linalg.norm(h_dump) / dumping_rate

        return {
            'momentum_to_dump': h_dump,
            'dumping_magnitude': np.linalg.norm(h_dump),
            'estimated_time': estimated_time,
            'recommended_action': 'execute_momentum_dumping'
        }

    def get_wheel_momentum_distribution(
        self,
        total_momentum: np.ndarray
    ) -> np.ndarray:
        """将总动量分配到各飞轮

        使用最小范数解:
        h_wheels = D^+ · h_total

        Args:
            total_momentum: 总动量矢量 [hx, hy, hz]

        Returns:
            各飞轮角动量数组
        """
        D = self.wheel_directions
        D_pinv = np.linalg.pinv(D)

        wheel_momentum = D_pinv @ total_momentum

        return wheel_momentum

    def compute_saturation_margin(
        self,
        momentum_state: MomentumState
    ) -> dict:
        """计算饱和度裕度

        Args:
            momentum_state: 动量状态

        Returns:
            裕度信息字典
        """
        wheel_speeds = momentum_state.wheel_speeds

        # 各飞轮裕度
        speed_margins = self.max_wheel_speed - np.abs(wheel_speeds)
        momentum_margins = speed_margins * self.wheel_inertia

        # 总动量裕度
        current_momentum = self.wheel_directions @ (
            wheel_speeds * self.wheel_inertia
        )
        total_momentum = np.linalg.norm(current_momentum)
        envelope_margin = self.momentum_envelope - total_momentum

        return {
            'speed_margins': speed_margins,
            'momentum_margins': momentum_margins,
            'min_wheel_margin': np.min(speed_margins),
            'total_momentum': total_momentum,
            'envelope_margin': envelope_margin,
            'saturation_ratio': total_momentum / self.momentum_envelope
        }
