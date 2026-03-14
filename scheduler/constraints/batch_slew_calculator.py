"""
批量姿态机动计算器 - Numba向量化优化

核心功能：
1. 将Python对象转换为Numba兼容的NumPy数组
2. 使用@njit(parallel=True)批量计算机动可行性
3. 将结果转换回Python对象

性能目标：相比逐个Python调用，加速5-15倍
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Numba JIT优化支持
try:
    from numba import jit, njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)

from core.models.satellite import Satellite
from core.models.target import Target
from core.dynamics.precise.attitude_types import AttitudeState, Quaternion

logger = logging.getLogger(__name__)


@dataclass
class BatchSlewCandidate:
    """批量机动计算候选数据结构"""
    sat_id: str
    satellite: Satellite
    target: Target
    window_start: datetime
    window_end: datetime
    prev_end_time: datetime
    prev_target: Optional[Target]
    imaging_duration: float
    sat_position: Tuple[float, float, float]  # ECEF (m)
    sat_velocity: Tuple[float, float, float]  # ECEF (m/s)


@dataclass
class BatchSlewResult:
    """批量机动计算结果"""
    feasible: bool
    slew_angle: float  # degrees
    slew_time: float   # seconds
    reset_time: float  # seconds
    actual_start: datetime
    energy_consumption: float
    momentum_margin: float
    reason: Optional[str] = None


class BatchSlewData:
    """批量数据容器 - 将Python对象转换为NumPy数组

    使用NumPy structured arrays或plain arrays以兼容Numba
    """

    def __init__(self, n_candidates: int):
        self.n = n_candidates

        # 输入数据数组
        # 前一姿态四元数 [n x 4]
        self.prev_quaternions = np.zeros((n_candidates, 4), dtype=np.float64)
        self.prev_quaternions[:, 0] = 1.0  # 默认单位四元数

        # 目标姿态四元数 [n x 4]
        self.target_quaternions = np.zeros((n_candidates, 4), dtype=np.float64)
        self.target_quaternions[:, 0] = 1.0

        # 卫星位置 [n x 3] (ECEF, m)
        self.sat_positions = np.zeros((n_candidates, 3), dtype=np.float64)

        # 目标位置 [n x 3] (ECEF, m)
        self.target_positions = np.zeros((n_candidates, 3), dtype=np.float64)

        # 时间间隔 [n] (seconds between prev_end and window_start)
        self.time_intervals = np.zeros(n_candidates, dtype=np.float64)

        # 成像持续时间 [n]
        self.imaging_durations = np.zeros(n_candidates, dtype=np.float64)

        # 动力学参数 [n] (每候选可不同)
        self.max_control_torque = np.zeros(n_candidates, dtype=np.float64)
        self.max_angular_velocity = np.zeros(n_candidates, dtype=np.float64)
        self.effective_inertia = np.full(n_candidates, 100.0, dtype=np.float64)
        self.settling_time = np.zeros(n_candidates, dtype=np.float64)
        self.motor_efficiency = np.full(n_candidates, 0.85, dtype=np.float64)

        # 窗口约束
        self.window_durations = np.zeros(n_candidates, dtype=np.float64)

        # 输出数组 (预分配)
        self.out_feasible = np.zeros(n_candidates, dtype=np.bool_)
        self.out_slew_angle = np.zeros(n_candidates, dtype=np.float64)
        self.out_slew_time = np.zeros(n_candidates, dtype=np.float64)
        self.out_reset_time = np.zeros(n_candidates, dtype=np.float64)
        self.out_energy = np.zeros(n_candidates, dtype=np.float64)
        self.out_momentum_margin = np.zeros(n_candidates, dtype=np.float64)

        # 元数据（Numba不直接处理，Python层使用）
        self.candidates: List[BatchSlewCandidate] = []
        self.satellite_configs: Dict[str, Any] = {}


# ============================================================================
# Numba JIT优化的核心计算函数
# ============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _target_to_ecef_numba(lat_rad: float, lon_rad: float, earth_radius: float = 6371000.0) -> Tuple[float, float, float]:
        """将地理坐标转换为ECEF坐标"""
        x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = earth_radius * np.sin(lat_rad)
        return x, y, z

    @njit(cache=True)
    def _compute_rotation_between_vectors_numba(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, float, float]:
        """计算从向量v1旋转到v2的四元数

        返回: (w, x, y, z) 四元数分量
        """
        # 确保单位向量
        norm1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        norm2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 1.0, 0.0, 0.0, 0.0

        v1_norm = v1 / norm1
        v2_norm = v2 / norm2

        # 点积
        dot = v1_norm[0]*v2_norm[0] + v1_norm[1]*v2_norm[1] + v1_norm[2]*v2_norm[2]

        # 如果向量相同，返回单位四元数
        if dot > 0.999999:
            return 1.0, 0.0, 0.0, 0.0

        # 如果向量相反
        if dot < -0.999999:
            # 找一个与v1不平行的向量
            if abs(v1_norm[0]) < 0.9:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([0.0, 1.0, 0.0])

            # axis = axis - v1_norm * dot_product
            axis_dot = axis[0]*v1_norm[0] + axis[1]*v1_norm[1] + axis[2]*v1_norm[2]
            axis[0] = axis[0] - v1_norm[0] * axis_dot
            axis[1] = axis[1] - v1_norm[1] * axis_dot
            axis[2] = axis[2] - v1_norm[2] * axis_dot

            # 归一化
            axis_norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
            if axis_norm > 1e-10:
                axis = axis / axis_norm

            return 0.0, axis[0], axis[1], axis[2]

        # 计算旋转轴 (叉积)
        axis = np.array([
            v1_norm[1]*v2_norm[2] - v1_norm[2]*v2_norm[1],
            v1_norm[2]*v2_norm[0] - v1_norm[0]*v2_norm[2],
            v1_norm[0]*v2_norm[1] - v1_norm[1]*v2_norm[0]
        ])
        axis_norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)

        if axis_norm < 1e-10:
            return 1.0, 0.0, 0.0, 0.0

        axis = axis / axis_norm

        # 旋转角度 (使用min/max代替np.clip以兼容Numba)
        dot_clipped = dot
        if dot_clipped > 1.0:
            dot_clipped = 1.0
        elif dot_clipped < -1.0:
            dot_clipped = -1.0
        angle = np.arccos(dot_clipped)

        # 构造四元数 (半角公式)
        half_angle = angle / 2
        sin_half = np.sin(half_angle)

        return np.cos(half_angle), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half

    @njit(cache=True)
    def _quaternion_rotation_angle_numba(q1_w: float, q1_x: float, q1_y: float, q1_z: float,
                                        q2_w: float, q2_x: float, q2_y: float, q2_z: float) -> float:
        """计算两个四元数之间的旋转角度（度）"""
        # q1的共轭
        q1_conj_w = q1_w
        q1_conj_x = -q1_x
        q1_conj_y = -q1_y
        q1_conj_z = -q1_z

        # 四元数乘法: q2 * q1_conj
        w = q2_w * q1_conj_w - q2_x * q1_conj_x - q2_y * q1_conj_y - q2_z * q1_conj_z

        # 旋转角度 = 2 * acos(|w|)
        w_abs = abs(w)
        if w_abs > 1.0:
            w_abs = 1.0

        angle_rad = 2.0 * np.arccos(w_abs)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    @njit(cache=True)
    def _compute_bang_bang_time_numba(
        rotation_angle_deg: float,
        max_angular_velocity: float,
        max_acceleration: float
    ) -> float:
        """计算bang-bang机动时间

        简化模型，假设时间最优轨迹
        """
        if rotation_angle_deg <= 0:
            return 0.0

        # 转换为弧度
        theta = np.radians(rotation_angle_deg)
        omega_max = np.radians(max_angular_velocity)
        alpha_max = np.radians(max_acceleration)

        if omega_max <= 0 or alpha_max <= 0:
            return 0.0

        # 达到最大速度所需时间
        t_accel = omega_max / alpha_max

        # 在最大速度下所需角度
        theta_at_max_speed = omega_max * t_accel  # = omega_max^2 / alpha_max

        if theta <= theta_at_max_speed:
            # 三角形速度曲线 (无法达到最大速度)
            # theta = 0.5 * alpha * t^2 * 2 = alpha * t^2
            t_total = 2.0 * np.sqrt(theta / alpha_max)
        else:
            # 梯形速度曲线
            # 加速段 + 匀速段 + 减速段
            theta_accel = 0.5 * alpha_max * t_accel**2
            theta_remaining = theta - 2.0 * theta_accel
            t_coast = theta_remaining / omega_max
            t_total = 2.0 * t_accel + t_coast

        return t_total

    @njit(cache=True)
    def _compute_single_slew_feasibility(
        prev_q: np.ndarray,  # [4] (w, x, y, z)
        target_q: np.ndarray,  # [4]
        sat_pos: np.ndarray,  # [3]
        target_pos: np.ndarray,  # [3]
        time_interval: float,
        imaging_duration: float,
        window_duration: float,
        max_angular_velocity: float,
        max_torque: float,
        settling_time: float,
        motor_efficiency: float
    ) -> Tuple[bool, float, float, float, float, float]:
        """计算单个候选的机动可行性

        返回: (feasible, slew_angle, slew_time, reset_time, energy, momentum_margin)
        """
        # 1. 计算从对地定向到目标姿态的旋转角度
        # 对地定向 = 单位四元数 (1, 0, 0, 0)
        nadir_q = np.array([1.0, 0.0, 0.0, 0.0])

        # 计算目标姿态（从卫星位置到目标位置）
        # 对地指向向量
        sat_norm = np.sqrt(sat_pos[0]**2 + sat_pos[1]**2 + sat_pos[2]**2)
        if sat_norm < 1e-10:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0

        nadir_vec = -sat_pos / sat_norm

        # 视线向量
        los_vec = target_pos - sat_pos
        los_norm = np.sqrt(los_vec[0]**2 + los_vec[1]**2 + los_vec[2]**2)
        if los_norm < 1e-10:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0

        los_vec = los_vec / los_norm

        # 计算目标四元数
        target_q_computed = _compute_rotation_between_vectors_numba(nadir_vec, los_vec)

        # 任务机动角度（从对地定向到目标）
        task_angle = _quaternion_rotation_angle_numba(
            nadir_q[0], nadir_q[1], nadir_q[2], nadir_q[3],
            target_q_computed[0], target_q_computed[1], target_q_computed[2], target_q_computed[3]
        )

        # 2. 检查时间间隔决定是否需要姿态复位
        assume_nadir = time_interval > 300.0  # 5分钟阈值

        total_reset_time = 0.0
        total_slew_angle = task_angle

        if not assume_nadir:
            # 需要计算姿态复位：从前一姿态到对地定向
            reset_angle = _quaternion_rotation_angle_numba(
                prev_q[0], prev_q[1], prev_q[2], prev_q[3],
                nadir_q[0], nadir_q[1], nadir_q[2], nadir_q[3]
            )

            # 复位机动时间
            if reset_angle > 0.1:  # 只有角度足够大才需要复位
                reset_time = _compute_bang_bang_time_numba(
                    reset_angle, max_angular_velocity, max_torque * 2.0  # 假设加速度与力矩成正比
                )
                total_reset_time = reset_time + settling_time

        # 3. 计算任务机动时间
        if task_angle > 0.1:
            task_time = _compute_bang_bang_time_numba(
                task_angle, max_angular_velocity, max_torque * 2.0
            )
            total_task_time = task_time + settling_time
        else:
            total_task_time = settling_time

        total_slew_time = total_reset_time + total_task_time

        # 4. 检查窗口约束
        feasible = total_slew_time <= window_duration

        # 5. 估算能量消耗 (简化模型)
        # E = P * t, 假设平均功率与角速度成正比
        avg_power = max_angular_velocity * 0.1 * motor_efficiency  # 简化估算
        energy = avg_power * total_slew_time

        # 6. 动量裕度 (简化估算)
        momentum_margin = 100.0 - total_slew_angle * 0.5  # 简化估算
        if momentum_margin < 0:
            momentum_margin = 0.0

        return feasible, total_slew_angle, total_slew_time, total_reset_time, energy, momentum_margin

    @njit(parallel=True, cache=True)
    def batch_compute_slew_feasibility_numba(
        # 输入数组
        prev_quaternions: np.ndarray,  # [n x 4]
        target_quaternions: np.ndarray,  # [n x 4]
        sat_positions: np.ndarray,  # [n x 3]
        target_positions: np.ndarray,  # [n x 3]
        time_intervals: np.ndarray,  # [n]
        imaging_durations: np.ndarray,  # [n]
        window_durations: np.ndarray,  # [n]
        max_angular_velocities: np.ndarray,  # [n]
        max_torques: np.ndarray,  # [n]
        settling_times: np.ndarray,  # [n]
        motor_efficiencies: np.ndarray,  # [n]

        # 输出数组 (预分配)
        out_feasible: np.ndarray,
        out_slew_angle: np.ndarray,
        out_slew_time: np.ndarray,
        out_reset_time: np.ndarray,
        out_energy: np.ndarray,
        out_momentum_margin: np.ndarray
    ):
        """批量计算机动可行性 - Numba并行版本"""
        n = len(prev_quaternions)

        for i in prange(n):
            # 提取单个候选的数据
            prev_q = prev_quaternions[i]
            target_q = target_quaternions[i]
            sat_pos = sat_positions[i]
            target_pos = target_positions[i]

            # 计算单个候选的可行性
            feasible, slew_angle, slew_time, reset_time, energy, margin = \
                _compute_single_slew_feasibility(
                    prev_q, target_q, sat_pos, target_pos,
                    time_intervals[i],
                    imaging_durations[i],
                    window_durations[i],
                    max_angular_velocities[i],
                    max_torques[i],
                    settling_times[i],
                    motor_efficiencies[i]
                )

            # 写入输出
            out_feasible[i] = feasible
            out_slew_angle[i] = slew_angle
            out_slew_time[i] = slew_time
            out_reset_time[i] = reset_time
            out_energy[i] = energy
            out_momentum_margin[i] = margin


# ============================================================================
# Python层API
# ============================================================================

class BatchSlewCalculator:
    """批量姿态机动计算器

    将Python对象转换为NumPy数组，调用Numba加速计算，再转回Python对象。
    支持两种模式：
    1. Bang-Bang快速计算（默认）
    2. 刚体动力学查表计算（高精度，use_lookup_table=True）
    """

    def __init__(self, use_lookup_table: bool = True):
        """初始化批量计算器

        Args:
            use_lookup_table: 是否使用刚体动力学预计算查找表
                            True: 使用查表（高精度，性能与Bang-Bang相同）
                            False: 使用Bang-Bang简化计算
        """
        self.use_numba = HAS_NUMBA
        self.use_lookup_table = use_lookup_table

        if use_lookup_table:
            # 使用刚体动力学查表
            from core.dynamics.precise import SlewLookupTable
            self._lookup_table = SlewLookupTable.get_instance()
            logger.info("BatchSlewCalculator initialized with rigid body dynamics lookup table")
        elif not self.use_numba:
            # 高精度要求：Numba是必需的
            raise RuntimeError(
                "Numba is required for high-precision batch calculations. "
                "Please install: pip install numba"
            )
        else:
            # 预热Numba JIT编译器
            self._warmup_numba()

    def _warmup_numba(self):
        """预热Numba JIT编译器

        通过执行一次小规模计算来触发Numba函数的编译，
        避免在真实调度中遇到编译延迟。
        """
        try:
            logger.debug("Warming up Numba JIT compiler...")
            # 创建最小规模的测试数据
            n = 2
            data = BatchSlewData(n)

            # 填充简单的测试数据
            for i in range(n):
                data.prev_quaternions[i] = [1.0, 0.0, 0.0, 0.0]
                data.target_quaternions[i] = [1.0, 0.0, 0.0, 0.0]
                data.sat_positions[i] = [7000000.0, 0.0, 0.0]
                data.target_positions[i] = [6371000.0, 0.0, 0.0]
                data.time_intervals[i] = 600.0
                data.imaging_durations[i] = 30.0
                data.window_durations[i] = 300.0
                data.max_angular_velocity[i] = 3.0
                data.max_control_torque[i] = 0.5
                data.settling_time[i] = 5.0
                data.motor_efficiency[i] = 0.85

            # 调用Numba函数触发编译
            batch_compute_slew_feasibility_numba(
                data.prev_quaternions,
                data.target_quaternions,
                data.sat_positions,
                data.target_positions,
                data.time_intervals,
                data.imaging_durations,
                data.window_durations,
                data.max_angular_velocity,
                data.max_control_torque,
                data.settling_time,
                data.motor_efficiency,
                data.out_feasible,
                data.out_slew_angle,
                data.out_slew_time,
                data.out_reset_time,
                data.out_energy,
                data.out_momentum_margin
            )
            logger.debug("Numba warmup completed")
        except Exception as e:
            logger.warning(f"Numba warmup failed: {e}")

    def prepare_batch_data(
        self,
        candidates: List[BatchSlewCandidate],
        sat_attitudes: Dict[str, AttitudeState]
    ) -> BatchSlewData:
        """准备批量计算数据

        Args:
            candidates: 候选列表
            sat_attitudes: 当前卫星姿态状态 {sat_id: AttitudeState}

        Returns:
            BatchSlewData: 批量数据容器
        """
        import math

        n = len(candidates)
        data = BatchSlewData(n)
        data.candidates = candidates

        for i, cand in enumerate(candidates):
            # 获取前一姿态
            prev_attitude = sat_attitudes.get(cand.sat_id)
            if prev_attitude is not None:
                q = prev_attitude.quaternion
                data.prev_quaternions[i] = [q.w, q.x, q.y, q.z]
            else:
                # 默认对地定向
                data.prev_quaternions[i] = [1.0, 0.0, 0.0, 0.0]

            # 目标位置
            if hasattr(cand.target, 'latitude') and hasattr(cand.target, 'longitude'):
                lat_rad = math.radians(cand.target.latitude)
                lon_rad = math.radians(cand.target.longitude)
                # 使用简化ECEF计算
                R_earth = 6371000.0
                data.target_positions[i] = [
                    R_earth * math.cos(lat_rad) * math.cos(lon_rad),
                    R_earth * math.cos(lat_rad) * math.sin(lon_rad),
                    R_earth * math.sin(lat_rad)
                ]
            else:
                data.target_positions[i] = [0.0, 0.0, 0.0]

            # 卫星位置
            data.sat_positions[i] = list(cand.sat_position)

            # 时间间隔
            time_diff = (cand.window_start - cand.prev_end_time).total_seconds()
            data.time_intervals[i] = time_diff

            # 成像持续时间
            data.imaging_durations[i] = cand.imaging_duration

            # 窗口持续时间
            window_duration = (cand.window_end - cand.window_start).total_seconds()
            data.window_durations[i] = window_duration

            # 动力学参数
            agility = getattr(cand.satellite.capabilities, 'agility', {}) or {}
            data.max_angular_velocity[i] = agility.get('max_slew_rate', 3.0)
            data.max_control_torque[i] = agility.get('max_torque', 0.5)
            data.settling_time[i] = agility.get('settling_time', 5.0)

        return data

    def compute_batch(
        self,
        data: BatchSlewData
    ) -> List[BatchSlewResult]:
        """执行批量计算

        Args:
            data: 批量数据容器

        Returns:
            批量计算结果列表
        """
        if self.use_lookup_table:
            # 使用刚体动力学查表（高精度）
            return self._compute_batch_lookup(data)

        if self.use_numba:
            try:
                # 调用Numba加速版本（Bang-Bang简化计算）
                batch_compute_slew_feasibility_numba(
                    data.prev_quaternions,
                    data.target_quaternions,
                    data.sat_positions,
                    data.target_positions,
                    data.time_intervals,
                    data.imaging_durations,
                    data.window_durations,
                    data.max_angular_velocity,
                    data.max_control_torque,
                    data.settling_time,
                    data.motor_efficiency,
                    data.out_feasible,
                    data.out_slew_angle,
                    data.out_slew_time,
                    data.out_reset_time,
                    data.out_energy,
                    data.out_momentum_margin
                )
            except Exception as e:
                # 高精度要求：Numba批量计算失败应抛出错误
                raise RuntimeError(f"批量机动计算失败: {e}") from e
        else:
            self._compute_batch_python(data)

        # 转换结果为Python对象
        results = []
        base_time = datetime.now()  # 如果没有候选数据，使用当前时间

        for i in range(data.n):
            # 获取候选的时间信息（如果可用）
            if i < len(data.candidates):
                cand = data.candidates[i]
                window_start = cand.window_start
            else:
                # 测试模式：使用默认时间
                window_start = base_time

            # 计算actual_start
            actual_start = window_start + timedelta(seconds=data.out_slew_time[i])

            results.append(BatchSlewResult(
                feasible=data.out_feasible[i],
                slew_angle=data.out_slew_angle[i],
                slew_time=data.out_slew_time[i],
                reset_time=data.out_reset_time[i],
                actual_start=actual_start,
                energy_consumption=data.out_energy[i],
                momentum_margin=data.out_momentum_margin[i],
                reason=None if data.out_feasible[i] else "Slew time exceeds window"
            ))

        return results

    def _compute_batch_lookup(self, data: BatchSlewData) -> List[BatchSlewResult]:
        """使用刚体动力学查找表进行批量计算

        通过查表获取预计算的刚体动力学结果，实现O(1)查询时间。
        对于角度在预计算点之间的查询，使用线性插值。

        Args:
            data: 批量数据容器

        Returns:
            批量计算结果列表
        """
        import math

        results = []

        for i in range(data.n):
            # 获取候选信息
            if i < len(data.candidates):
                cand = data.candidates[i]
                satellite = cand.satellite
            else:
                # 测试模式
                satellite = None

            # 计算机动角度
            sat_pos = data.sat_positions[i]
            target_pos = data.target_positions[i]

            # 计算从卫星到目标的视线角度
            sat_norm = np.linalg.norm(sat_pos)
            target_norm = np.linalg.norm(target_pos)

            if sat_norm < 1e-10 or target_norm < 1e-10:
                # 无效位置，返回不可行
                results.append(BatchSlewResult(
                    feasible=False,
                    slew_angle=0.0,
                    slew_time=0.0,
                    reset_time=0.0,
                    actual_start=datetime.now(),
                    energy_consumption=0.0,
                    momentum_margin=0.0,
                    reason="Invalid satellite or target position"
                ))
                continue

            # 计算地心角
            nadir_vec = -np.array(sat_pos) / sat_norm
            los_vec = (np.array(target_pos) - np.array(sat_pos))
            los_norm = np.linalg.norm(los_vec)

            if los_norm < 1e-10:
                slew_angle = 0.0
            else:
                los_vec = los_vec / los_norm
                dot = np.clip(np.dot(nadir_vec, los_vec), -1.0, 1.0)
                slew_angle = math.degrees(math.acos(dot))

            # 检查时间间隔决定是否包含复位时间
            time_interval = data.time_intervals[i]
            include_reset = time_interval < 300.0  # 小于5分钟需要复位

            # 通过查找表获取刚体动力学结果
            if satellite is not None:
                lookup_result = self._lookup_table.query(satellite, slew_angle)

                # 基础机动时间
                slew_time = lookup_result.time
                energy = lookup_result.energy
                momentum_margin = lookup_result.momentum_margin
                feasible = lookup_result.feasible

                # 如果需要复位，加上复位时间（从当前姿态回到对地定向）
                reset_time = 0.0
                if include_reset:
                    # 计算前一姿态到对地定向的角度
                    prev_q = data.prev_quaternions[i]
                    # 简化：假设复位角度为平均姿态偏离（约10度）
                    # 更精确的做法是计算前一姿态四元数到单位四元数的角度
                    reset_angle = self._estimate_reset_angle(prev_q)
                    reset_result = self._lookup_table.query(satellite, reset_angle)
                    reset_time = reset_result.time

                total_slew_time = slew_time + reset_time
            else:
                # 测试模式：使用Bang-Bang估算
                max_angular_velocity = data.max_angular_velocity[i]
                settling_time = data.settling_time[i]
                slew_time = slew_angle / max_angular_velocity + settling_time
                reset_time = settling_time if include_reset else 0.0
                total_slew_time = slew_time + reset_time
                energy = slew_time * 10.0
                momentum_margin = max(0, 100.0 - slew_angle)
                feasible = slew_angle <= 45.0

            # 检查窗口约束
            window_duration = data.window_durations[i]
            feasible = feasible and (total_slew_time <= window_duration)

            # 计算actual_start
            if i < len(data.candidates):
                window_start = data.candidates[i].window_start
            else:
                window_start = datetime.now()

            actual_start = window_start + timedelta(seconds=total_slew_time)

            results.append(BatchSlewResult(
                feasible=feasible,
                slew_angle=slew_angle,
                slew_time=total_slew_time,
                reset_time=reset_time,
                actual_start=actual_start,
                energy_consumption=energy,
                momentum_margin=momentum_margin,
                reason=None if feasible else "Slew time exceeds window or angle not feasible"
            ))

        return results

    def _estimate_reset_angle(self, prev_q: np.ndarray) -> float:
        """估算从当前姿态复位到对地定向所需角度

        Args:
            prev_q: 前一姿态四元数 [w, x, y, z]

        Returns:
            复位角度（度）
        """
        # 对地定向 = 单位四元数
        # 两四元数夹角 = 2 * acos(|dot(q1, q2)|)
        w = prev_q[0]
        # 取绝对值确保得到最小旋转角度
        w_abs = abs(w)
        if w_abs > 1.0:
            w_abs = 1.0

        # 旋转角度 = 2 * acos(|w|)
        angle_rad = 2.0 * np.arccos(w_abs)
        angle_deg = np.degrees(angle_rad)

        # 如果角度很小，返回最小值
        if angle_deg < 0.1:
            return 0.0
        return float(angle_deg)

    def _compute_batch_python(self, data: BatchSlewData):
        """Python回退版本（当Numba不可用时）"""
        n = data.n

        for i in range(n):
            feasible, slew_angle, slew_time, reset_time, energy, margin = \
                _compute_single_slew_feasibility(
                    data.prev_quaternions[i],
                    data.target_quaternions[i],
                    data.sat_positions[i],
                    data.target_positions[i],
                    data.time_intervals[i],
                    data.imaging_durations[i],
                    data.window_durations[i],
                    data.max_angular_velocity[i],
                    data.max_control_torque[i],
                    data.settling_time[i],
                    data.motor_efficiency[i]
                )

            data.out_feasible[i] = feasible
            data.out_slew_angle[i] = slew_angle
            data.out_slew_time[i] = slew_time
            data.out_reset_time[i] = reset_time
            data.out_energy[i] = energy
            data.out_momentum_margin[i] = margin


# 供非Numba环境使用的Python版本
if not HAS_NUMBA:
    def _compute_single_slew_feasibility(
        prev_q, target_q, sat_pos, target_pos,
        time_interval, imaging_duration, window_duration,
        max_angular_velocity, max_torque, settling_time, motor_efficiency
    ):
        """Python版本的单个候选计算（回退使用）"""
        import math

        # 简化的姿态计算
        # 1. 计算任务角度（简化）
        # 计算视线角度
        sat_norm = math.sqrt(sat_pos[0]**2 + sat_pos[1]**2 + sat_pos[2]**2)
        target_norm = math.sqrt(target_pos[0]**2 + target_pos[1]**2 + target_pos[2]**2)

        if sat_norm < 1e-10 or target_norm < 1e-10:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0

        # 点积计算角度
        dot = (sat_pos[0]*target_pos[0] + sat_pos[1]*target_pos[1] + sat_pos[2]*target_pos[2]) / (sat_norm * target_norm)
        dot = max(-1.0, min(1.0, dot))

        # 地心角
        angle_rad = math.acos(dot)
        task_angle = math.degrees(angle_rad)

        # 2. 简化机动时间计算
        if task_angle < 1.0:
            task_time = settling_time
        else:
            # 简化的slew time估算
            omega_max = max_angular_velocity * (math.pi / 180.0)  # rad/s
            alpha = max_torque * 2.0 / 100.0  # 简化假设

            if alpha > 0 and omega_max > 0:
                t_accel = omega_max / alpha
                theta_accel = 0.5 * alpha * t_accel**2
                theta_target = angle_rad

                if theta_target <= 2 * theta_accel:
                    task_time = 2.0 * math.sqrt(theta_target / alpha) + settling_time
                else:
                    t_coast = (theta_target - 2 * theta_accel) / omega_max
                    task_time = 2.0 * t_accel + t_coast + settling_time
            else:
                task_time = settling_time + task_angle / max_angular_velocity

        # 3. 姿态复位时间
        reset_time = 0.0
        if time_interval < 300.0:  # 小于5分钟
            # 简化的复位角度估算
            reset_angle = 15.0  # 假设平均复位角度
            reset_time = settling_time + reset_angle / max_angular_velocity

        total_time = task_time + reset_time

        # 4. 可行性检查
        feasible = total_time < window_duration

        # 5. 简化的能量和动量估算
        energy = total_time * 10.0  # 简化估算
        margin = 100.0 - task_angle

        return feasible, task_angle, total_time, reset_time, energy, margin
