"""
姿态动力学类型定义

定义姿态机动计算中使用的核心数据类型。
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
from datetime import datetime
import numpy as np


@dataclass
class Quaternion:
    """四元数表示姿态

    Attributes:
        w: 实部 (标量)
        x, y, z: 虚部 (向量)
    """
    w: float
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        """转换为numpy数组 [w, x, y, z]"""
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """从numpy数组创建"""
        return cls(w=arr[0], x=arr[1], y=arr[2], z=arr[3])

    def normalize(self) -> 'Quaternion':
        """归一化四元数"""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(
            w=self.w / norm,
            x=self.x / norm,
            y=self.y / norm,
            z=self.z / norm
        )

    def conjugate(self) -> 'Quaternion':
        """共轭四元数"""
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """四元数乘法"""
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        )


@dataclass
class AngularVelocity:
    """角速度矢量 (rad/s)"""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AngularVelocity':
        return cls(x=arr[0], y=arr[1], z=arr[2])

    def magnitude(self) -> float:
        """角速度大小 (rad/s)"""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def magnitude_deg(self) -> float:
        """角速度大小 (deg/s)"""
        return np.degrees(self.magnitude())


@dataclass
class ControlTorque:
    """控制力矩矢量 (Nm)"""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControlTorque':
        return cls(x=arr[0], y=arr[1], z=arr[2])

    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class MomentumState:
    """飞轮角动量状态"""
    wheel_speeds: np.ndarray  # 各飞轮转速 (rad/s)
    wheel_inertias: np.ndarray  # 各飞轮转动惯量 (kg·m²)
    total_momentum: np.ndarray  # 总角动量矢量 [hx, hy, hz] (Nms)

    def compute_total_momentum(self) -> np.ndarray:
        """计算总角动量"""
        return self.wheel_speeds * self.wheel_inertias


@dataclass
class AttitudeState:
    """完整的卫星姿态状态"""
    quaternion: Quaternion      # 姿态四元数
    angular_velocity: AngularVelocity  # 角速度
    timestamp: datetime         # 时间戳
    momentum: Optional[MomentumState] = None  # 飞轮动量状态

    def is_valid(self) -> bool:
        """检查状态是否有效"""
        if self.quaternion is None:
            return False
        # 检查四元数是否归一化
        q_norm = np.sqrt(
            self.quaternion.w**2 +
            self.quaternion.x**2 +
            self.quaternion.y**2 +
            self.quaternion.z**2
        )
        return abs(q_norm - 1.0) < 1e-6


@dataclass
class InertiaTensor:
    """惯性张量矩阵 (kg·m²)"""
    # 对角元素
    Ixx: float
    Iyy: float
    Izz: float
    # 非对角元素
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """转换为3x3矩阵"""
        return np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Ixy, self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz]
        ])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'InertiaTensor':
        """从矩阵创建"""
        return cls(
            Ixx=matrix[0, 0],
            Iyy=matrix[1, 1],
            Izz=matrix[2, 2],
            Ixy=matrix[0, 1],
            Ixz=matrix[0, 2],
            Iyz=matrix[1, 2]
        )

    @classmethod
    def diagonal(cls, Ixx: float, Iyy: float, Izz: float) -> 'InertiaTensor':
        """创建对角惯性张量"""
        return cls(Ixx=Ixx, Iyy=Iyy, Izz=Izz)
