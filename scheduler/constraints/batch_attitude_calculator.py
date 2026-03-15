"""
批量姿态计算器 - Numba向量化优化

将姿态角计算批量化和并行化，解决调度器性能瓶颈。

核心优化：
1. Numba JIT编译核心计算函数
2. 并行计算多个候选的姿态角
3. 直接使用批量传播器数据，避免逐个查询

性能目标：将姿态预计算从 ~1000ms 降低到 ~10ms (100x加速)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

# Numba JIT优化支持
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)

# ============================================================================
# Numba加速的核心计算函数
# ============================================================================

@njit(cache=True)
def _normalize_vector_numba(v):
    """归一化向量"""
    norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm < 1e-10:
        return v
    return v / norm

@njit(cache=True)
def _cross_product_numba(a, b):
    """计算叉积"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@njit(cache=True)
def _compute_attitude_angles_numba(
    sat_pos: np.ndarray,      # [3] 卫星位置 (ECEF, m)
    sat_vel: np.ndarray,      # [3] 卫星速度 (ECEF, m/s)
    target_lat: float,        # 目标纬度 (度)
    target_lon: float         # 目标经度 (度)
) -> Tuple[float, float]:
    """
    计算姿态角 (Numba加速版本)

    Returns:
        (roll, pitch) in degrees
    """
    # 地球半径
    R_earth = 6371000.0

    # 将目标地理坐标转换为ECEF
    lat_rad = np.radians(target_lat)
    lon_rad = np.radians(target_lon)

    target_pos = np.array([
        R_earth * np.cos(lat_rad) * np.cos(lon_rad),
        R_earth * np.cos(lat_rad) * np.sin(lon_rad),
        R_earth * np.sin(lat_rad)
    ])

    # 构建LVLH坐标系
    # Z轴：指向地心（负位置方向）
    r_norm = np.sqrt(sat_pos[0]**2 + sat_pos[1]**2 + sat_pos[2]**2)
    if r_norm < 1e-10:
        return 0.0, 0.0

    Z = -sat_pos / r_norm

    # X轴：沿飞行方向（速度在垂直于Z方向的分量）
    v_dot_z = sat_vel[0]*Z[0] + sat_vel[1]*Z[1] + sat_vel[2]*Z[2]
    vx_perp = sat_vel[0] - v_dot_z * Z[0]
    vy_perp = sat_vel[1] - v_dot_z * Z[1]
    vz_perp = sat_vel[2] - v_dot_z * Z[2]

    v_perp_norm = np.sqrt(vx_perp**2 + vy_perp**2 + vz_perp**2)
    if v_perp_norm < 1e-10:
        # 速度几乎平行于Z轴，使用位置与极轴的叉积
        pole = np.array([0.0, 0.0, 1.0])
        X = _cross_product_numba(Z, pole)
        x_norm = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
        if x_norm < 1e-10:
            X = np.array([1.0, 0.0, 0.0])
        else:
            X = X / x_norm
    else:
        X = np.array([vx_perp, vy_perp, vz_perp]) / v_perp_norm

    # Y轴：完成右手系 (Z × X)
    Y = _cross_product_numba(Z, X)
    y_norm = np.sqrt(Y[0]**2 + Y[1]**2 + Y[2]**2)
    if y_norm > 1e-10:
        Y = Y / y_norm

    # 计算视线向量（从卫星指向目标）
    los_vec = target_pos - sat_pos
    los_norm = np.sqrt(los_vec[0]**2 + los_vec[1]**2 + los_vec[2]**2)
    if los_norm < 1e-10:
        return 0.0, 0.0

    los_vec = los_vec / los_norm

    # 反转视线向量（卫星看向目标）
    pointing_vec = -los_vec

    # 将视线向量转换到LVLH坐标系
    # 在LVLH系中的分量 = 点积
    x_lvlh = pointing_vec[0]*X[0] + pointing_vec[1]*X[1] + pointing_vec[2]*X[2]
    y_lvlh = pointing_vec[0]*Y[0] + pointing_vec[1]*Y[1] + pointing_vec[2]*Y[2]
    z_lvlh = pointing_vec[0]*Z[0] + pointing_vec[1]*Z[1] + pointing_vec[2]*Z[2]

    # 计算滚转和俯仰角
    # 滚转 = atan2(Y分量, Z分量)
    # 俯仰 = atan2(X分量, Z分量)
    # 注意：当Z分量为负时（指向地心），需要特殊处理

    if abs(z_lvlh) < 1e-10:
        roll = 0.0
        pitch = 0.0
    else:
        roll = np.arctan2(y_lvlh, z_lvlh)
        pitch = np.arctan2(x_lvlh, z_lvlh)

    # 转换为角度
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)

    return roll_deg, pitch_deg


@njit(parallel=True, cache=True)
def _batch_compute_attitudes_numba(
    sat_positions: np.ndarray,    # [n x 3]
    sat_velocities: np.ndarray,   # [n x 3]
    target_lats: np.ndarray,      # [n]
    target_lons: np.ndarray       # [n]
) -> Tuple[np.ndarray, np.ndarray]:  # (rolls, pitches)
    """
    批量计算姿态角 (Numba并行版本)

    Args:
        sat_positions: 卫星位置数组 [n x 3] (ECEF, m)
        sat_velocities: 卫星速度数组 [n x 3] (ECEF, m/s)
        target_lats: 目标纬度数组 [n] (度)
        target_lons: 目标经度数组 [n] (度)

    Returns:
        (rolls, pitches) 数组，单位为度
    """
    n = len(sat_positions)
    rolls = np.zeros(n, dtype=np.float64)
    pitches = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        roll, pitch = _compute_attitude_angles_numba(
            sat_positions[i],
            sat_velocities[i],
            target_lats[i],
            target_lons[i]
        )
        rolls[i] = roll
        pitches[i] = pitch

    return rolls, pitches


# ============================================================================
# Python层API
# ============================================================================

@dataclass
class BatchAttitudeCandidate:
    """批量姿态计算候选"""
    sat_id: str
    target_id: str
    target_lat: float
    target_lon: float
    imaging_time: datetime


@dataclass
class BatchAttitudeResult:
    """批量姿态计算结果"""
    sat_id: str
    target_id: str
    imaging_time: datetime
    roll: float
    pitch: float
    feasible: bool


class BatchAttitudeCalculator:
    """批量姿态计算器

    使用Numba并行计算多个候选的姿态角，大幅提升调度器性能。
    """

    def __init__(self):
        self.use_numba = HAS_NUMBA
        if self.use_numba:
            # 预热Numba编译器
            self._warmup()

    def _warmup(self):
        """预热Numba编译器"""
        # 创建小的测试数据集触发编译
        n = 10
        sat_positions = np.random.randn(n, 3) * 7000000
        sat_velocities = np.random.randn(n, 3) * 7000
        target_lats = np.random.uniform(-90, 90, n)
        target_lons = np.random.uniform(-180, 180, n)

        try:
            _batch_compute_attitudes_numba(
                sat_positions, sat_velocities, target_lats, target_lons
            )
        except Exception:
            pass

    def compute_batch(
        self,
        candidates: List[BatchAttitudeCandidate],
        sat_positions: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    ) -> List[BatchAttitudeResult]:
        """
        批量计算姿态角

        Args:
            candidates: 候选列表
            sat_positions: 卫星位置字典 {sat_id: (position, velocity)}

        Returns:
            姿态计算结果列表
        """
        if not candidates:
            return []

        n = len(candidates)

        # 准备输入数组
        sat_pos_array = np.zeros((n, 3), dtype=np.float64)
        sat_vel_array = np.zeros((n, 3), dtype=np.float64)
        target_lats = np.zeros(n, dtype=np.float64)
        target_lons = np.zeros(n, dtype=np.float64)

        # 填充数据
        valid_mask = np.ones(n, dtype=np.bool_)
        for i, cand in enumerate(candidates):
            # 支持两种键格式: (sat_id, imaging_time) 或 sat_id
            pos_vel = sat_positions.get((cand.sat_id, cand.imaging_time))
            if pos_vel is None:
                pos_vel = sat_positions.get(cand.sat_id)
            if pos_vel is None:
                valid_mask[i] = False
                continue

            position, velocity = pos_vel
            sat_pos_array[i] = position
            sat_vel_array[i] = velocity
            target_lats[i] = cand.target_lat
            target_lons[i] = cand.target_lon

        if self.use_numba and np.any(valid_mask):
            # 使用Numba并行计算
            rolls, pitches = _batch_compute_attitudes_numba(
                sat_pos_array, sat_vel_array, target_lats, target_lons
            )
        else:
            # Python回退版本
            rolls = np.zeros(n)
            pitches = np.zeros(n)
            for i in range(n):
                if valid_mask[i]:
                    roll, pitch = _compute_attitude_angles_numba(
                        sat_pos_array[i],
                        sat_vel_array[i],
                        target_lats[i],
                        target_lons[i]
                    )
                    rolls[i] = roll
                    pitches[i] = pitch

        # 构建结果
        results = []
        for i, cand in enumerate(candidates):
            results.append(BatchAttitudeResult(
                sat_id=cand.sat_id,
                target_id=cand.target_id,
                imaging_time=cand.imaging_time,
                roll=rolls[i],
                pitch=pitches[i],
                feasible=valid_mask[i]
            ))

        return results


# ============================================================================
# Numba加速的姿态过滤函数
# ============================================================================

@njit(parallel=True, cache=True)
def _batch_filter_by_attitude_numba(
    rolls: np.ndarray,          # [n] roll angles in degrees
    pitches: np.ndarray,        # [n] pitch angles in degrees
    max_roll_angles: np.ndarray,    # [n] max roll angles in degrees
    max_pitch_angles: np.ndarray    # [n] max pitch angles in degrees
) -> np.ndarray:
    """
    批量姿态过滤 - Numba并行版本

    分别检查滚转角和俯仰角是否超过各自的限制。
    这是正确的姿态约束检查方式（替代原来的合成角度检查）。

    Args:
        rolls: 滚转角数组（度）
        pitches: 俯仰角数组（度）
        max_roll_angles: 最大滚转角数组（度）
        max_pitch_angles: 最大俯仰角数组（度）

    Returns:
        feasible_mask: 可行标记数组 [n]
    """
    n = len(rolls)
    feasible_mask = np.ones(n, dtype=np.bool_)

    for i in prange(n):
        # 分别检查滚转角和俯仰角（严格检查，不允许超出名义约束）
        if abs(rolls[i]) > max_roll_angles[i]:
            feasible_mask[i] = False
        elif abs(pitches[i]) > max_pitch_angles[i]:
            feasible_mask[i] = False

    return feasible_mask


@dataclass
class BatchAttitudeFilterResult:
    """批量姿态过滤结果"""
    feasible_mask: np.ndarray  # 可行标记
    filtered_indices: List[int]  # 通过过滤的索引
    filtered_count: int  # 通过过滤的数量
    roll_violations: List[int]  # 滚转角超限的索引
    pitch_violations: List[int]  # 俯仰角超限的索引


def batch_filter_by_attitude(
    rolls: List[float],
    pitches: List[float],
    max_roll_angles: List[float],
    max_pitch_angles: List[float]
) -> BatchAttitudeFilterResult:
    """
    批量姿态过滤 - 根据姿态角筛选候选

    分别检查滚转角和俯仰角是否超过各自的限制。

    Args:
        rolls: 滚转角列表（度）
        pitches: 俯仰角列表（度）
        max_roll_angles: 最大滚转角列表（度）
        max_pitch_angles: 最大俯仰角列表（度）

    Returns:
        BatchAttitudeFilterResult: 过滤结果
    """
    if not rolls:
        return BatchAttitudeFilterResult(
            feasible_mask=np.array([], dtype=np.bool_),
            filtered_indices=[],
            filtered_count=0,
            roll_violations=[],
            pitch_violations=[]
        )

    # 转换为numpy数组
    rolls_arr = np.array(rolls, dtype=np.float64)
    pitches_arr = np.array(pitches, dtype=np.float64)
    max_roll_arr = np.array(max_roll_angles, dtype=np.float64)
    max_pitch_arr = np.array(max_pitch_angles, dtype=np.float64)

    # Numba批量过滤
    if HAS_NUMBA:
        feasible_mask = _batch_filter_by_attitude_numba(
            rolls_arr, pitches_arr, max_roll_arr, max_pitch_arr
        )
    else:
        # Python回退
        feasible_mask = np.ones(len(rolls), dtype=np.bool_)
        for i in range(len(rolls)):
            if abs(rolls[i]) > max_roll_angles[i]:
                feasible_mask[i] = False
            elif abs(pitches[i]) > max_pitch_angles[i]:
                feasible_mask[i] = False

    # 获取通过的索引
    filtered_indices = np.where(feasible_mask)[0].tolist()
    roll_violations = [i for i in range(len(rolls)) if abs(rolls[i]) > max_roll_angles[i]]
    pitch_violations = [i for i in range(len(pitches)) if abs(pitches[i]) > max_pitch_angles[i]]

    return BatchAttitudeFilterResult(
        feasible_mask=feasible_mask,
        filtered_indices=filtered_indices,
        filtered_count=len(filtered_indices),
        roll_violations=roll_violations,
        pitch_violations=pitch_violations
    )


# 全局计算器实例
_batch_attitude_calculator: Optional[BatchAttitudeCalculator] = None


def get_batch_attitude_calculator() -> BatchAttitudeCalculator:
    """获取全局批量姿态计算器实例"""
    global _batch_attitude_calculator
    if _batch_attitude_calculator is None:
        _batch_attitude_calculator = BatchAttitudeCalculator()
    return _batch_attitude_calculator
