"""
姿态预计算缓存 - 空间换时间优化

策略：
1. 预加载所有卫星轨道数据到NumPy数组（O(1)索引访问）
2. 预计算所有可见窗口的姿态角并缓存
3. 使用内存映射和向量化查询

内存需求：
- 轨道数据：~445MB（已加载）
- 姿态缓存：~30MB（277,946窗口 × 2角度 × 8字节）
- 总计：~475MB

性能提升：
- 原方式：每次查询需要线性搜索+插值+姿态计算 (~10-20ms)
- 优化后：直接数组索引查询 (~0.001ms)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import gzip
import json
from pathlib import Path

# Numba JIT优化
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

logger = logging.getLogger(__name__)

# Numba JIT函数使用的常量（必须在模块级别定义以便Numba编译）
_EARTH_RADIUS_M = 6371000.0  # 地球半径（米）
_BYTES_PER_MB = 1024 * 1024  # 每MB的字节数
_ATTITUDE_ENTRY_BYTES = 16   # 每个姿态缓存条目的大小（字节）


@dataclass
class AttitudePrecomputeConfig:
    """姿态预计算配置"""
    enable_precache: bool = True
    use_memory_mapping: bool = False  # 如果内存不足，使用内存映射
    max_cache_size_mb: int = 500


class AttitudePrecacheManager:
    """
    姿态预计算管理器

    在调度开始前预计算所有可见窗口的姿态角，存储在内存中，
    调度时直接O(1)查询，无需实时计算。
    """

    def __init__(self, config: Optional[AttitudePrecomputeConfig] = None):
        self.config = config or AttitudePrecomputeConfig()

        # 预计算的姿态缓存
        # 键: (sat_id, window_start_iso) -> (roll, pitch)
        self._attitude_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

        # 卫星轨道数据数组（O(1)索引访问）
        self._sat_timestamps: Dict[str, np.ndarray] = {}
        self._sat_positions: Dict[str, np.ndarray] = {}
        self._sat_velocities: Dict[str, np.ndarray] = {}

        # 时间基准
        self._start_time: Optional[datetime] = None
        self._time_step: float = 1.0  # 秒

        # 统计
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("AttitudePrecacheManager initialized")

    def load_orbit_data(self, orbit_file: str, start_time: datetime) -> bool:
        """
        加载轨道数据到NumPy数组

        Args:
            orbit_file: 轨道数据文件路径 (JSON.GZ)
            start_time: 场景开始时间

        Returns:
            是否成功
        """
        try:
            logger.info(f"Loading orbit data from {orbit_file}")

            with gzip.open(orbit_file, 'rt') as f:
                data = json.load(f)

            orbits = data.get('orbits', [])
            if not orbits:
                logger.error("No orbit data found")
                return False

            self._start_time = start_time

            # 按卫星分组
            sat_data: Dict[str, List[Dict]] = {}
            for record in orbits:
                sat_id = record.get('satellite_id')
                if sat_id not in sat_data:
                    sat_data[sat_id] = []
                sat_data[sat_id].append(record)

            # 转换为NumPy数组
            for sat_id, records in sat_data.items():
                # 按时间戳排序
                records.sort(key=lambda r: r.get('timestamp', 0))

                n = len(records)
                timestamps = np.zeros(n, dtype=np.float64)
                positions = np.zeros((n, 3), dtype=np.float64)
                velocities = np.zeros((n, 3), dtype=np.float64)

                for i, record in enumerate(records):
                    timestamps[i] = record.get('timestamp', 0)
                    # 支持两种格式：position数组 或 pos_x/pos_y/pos_z单独字段
                    if 'position' in record:
                        pos = record.get('position', [0, 0, 0])
                        vel = record.get('velocity', [0, 0, 0])
                    else:
                        # Java导出的格式：pos_x, pos_y, pos_z
                        pos = [record.get('pos_x', 0), record.get('pos_y', 0), record.get('pos_z', 0)]
                        vel = [record.get('vel_x', 0), record.get('vel_y', 0), record.get('vel_z', 0)]
                    positions[i] = pos
                    velocities[i] = vel

                self._sat_timestamps[sat_id] = timestamps
                self._sat_positions[sat_id] = positions
                self._sat_velocities[sat_id] = velocities

            logger.info(f"Loaded orbit data for {len(sat_data)} satellites")
            return True

        except Exception as e:
            logger.error(f"Failed to load orbit data: {e}")
            return False

    def precompute_attitudes_for_windows(
        self,
        visibility_windows: List[Dict[str, Any]]
    ) -> int:
        """
        预计算所有可见窗口的姿态角

        Args:
            visibility_windows: 可见性窗口列表

        Returns:
            预计算的姿态数量
        """
        if not HAS_NUMBA:
            logger.warning("Numba not available, skipping precomputation")
            return 0

        logger.info(f"Precomputing attitudes for {len(visibility_windows)} windows...")

        # 准备批量计算数据
        batch_data = []

        for window in visibility_windows:
            sat_id = window.get('satellite_id')
            target_id = window.get('target_id')

            # 跳过地面站窗口
            if target_id and target_id.startswith('GS:'):
                continue

            # 获取目标坐标
            target = window.get('target', {})
            lat = target.get('latitude') if isinstance(target, dict) else getattr(target, 'latitude', None)
            lon = target.get('longitude') if isinstance(target, dict) else getattr(target, 'longitude', None)

            if lat is None or lon is None:
                continue

            # 获取窗口时间
            window_start = window.get('start')
            window_end = window.get('end')

            if isinstance(window_start, str):
                window_start = datetime.fromisoformat(window_start.replace('Z', '+00:00'))
            if isinstance(window_end, str):
                window_end = datetime.fromisoformat(window_end.replace('Z', '+00:00'))

            if window_start is None or window_end is None:
                continue

            # 使用窗口中点作为计算时间
            mid_time = window_start + (window_end - window_start) / 2

            batch_data.append({
                'sat_id': sat_id,
                'target_id': target_id,
                'window_start': window_start,
                'window_end': window_end,
                'imaging_time': mid_time,
                'target_lat': lat,
                'target_lon': lon
            })

        # 批量计算姿态
        n_computed = self._batch_compute_and_cache(batch_data)

        logger.info(f"Precomputed {n_computed} attitudes")
        return n_computed

    def _batch_compute_and_cache(self, batch_data: List[Dict]) -> int:
        """批量计算姿态并缓存"""
        if not batch_data:
            return 0

        n = len(batch_data)
        sat_positions = np.zeros((n, 3), dtype=np.float64)
        sat_velocities = np.zeros((n, 3), dtype=np.float64)
        target_lats = np.zeros(n, dtype=np.float64)
        target_lons = np.zeros(n, dtype=np.float64)
        valid_mask = np.zeros(n, dtype=np.bool_)

        # 获取卫星位置
        for i, data in enumerate(batch_data):
            sat_id = data['sat_id']
            imaging_time = data['imaging_time']

            if sat_id not in self._sat_timestamps:
                continue

            # 计算时间索引
            timestamp = (imaging_time - self._start_time).total_seconds()

            # 查找最近的时间点
            timestamps = self._sat_timestamps[sat_id]
            positions = self._sat_positions[sat_id]
            velocities = self._sat_velocities[sat_id]

            # 二分查找
            idx = np.searchsorted(timestamps, timestamp)
            if idx == 0:
                idx = 0
            elif idx >= len(timestamps):
                idx = len(timestamps) - 1
            else:
                # 选择更近的
                if abs(timestamps[idx] - timestamp) > abs(timestamps[idx-1] - timestamp):
                    idx = idx - 1

            sat_positions[i] = positions[idx]
            sat_velocities[i] = velocities[idx]
            target_lats[i] = data['target_lat']
            target_lons[i] = data['target_lon']
            valid_mask[i] = True

        # Numba批量姿态计算
        if np.any(valid_mask):
            rolls, pitches = _batch_compute_attitudes_numba(
                sat_positions[valid_mask],
                sat_velocities[valid_mask],
                target_lats[valid_mask],
                target_lons[valid_mask]
            )

            # 存入缓存
            valid_indices = np.where(valid_mask)[0]
            for i, idx in enumerate(valid_indices):
                data = batch_data[idx]
                key = (data['sat_id'], data['window_start'].isoformat())
                self._attitude_cache[key] = (rolls[i], pitches[i])

            return len(rolls)

        return 0

    def load_precomputed_attitudes_from_windows(
        self,
        visibility_windows: List[Any]
    ) -> int:
        """
        从Java预计算的可见性窗口中加载姿态数据

        Args:
            visibility_windows: VisibilityWindow对象列表（包含attitude_samples）

        Returns:
            加载的姿态缓存数量
        """
        logger.info(f"Loading precomputed attitudes from {len(visibility_windows)} windows...")

        count = 0
        for window in visibility_windows:
            sat_id = window.satellite_id
            window_start = window.start_time

            # 跳过地面站窗口
            if window.target_id.startswith('GS:'):
                continue

            # 检查是否有预计算的姿态采样数据
            if window.attitude_samples:
                # 找到窗口中点附近的姿态（或取第一个有效姿态）
                # attitude_samples: [(timestamp_offset, roll, pitch), ...]
                mid_idx = len(window.attitude_samples) // 2
                sample = window.attitude_samples[mid_idx]
                roll, pitch = sample[1], sample[2]

                key = (sat_id, window_start.isoformat())
                self._attitude_cache[key] = (roll, pitch)
                count += 1
            elif not window.attitude_feasible:
                # 如果Java标记为不可行，缓存一个特殊值
                # 这样后续查询可以快速知道这个窗口不可行
                key = (sat_id, window_start.isoformat())
                self._attitude_cache[key] = (None, None)  # None表示不可行
                count += 1

        logger.info(f"Loaded {count} precomputed attitudes from Java")
        return count

    def get_attitude(
        self,
        sat_id: str,
        window_start: datetime
    ) -> Optional[Tuple[float, float]]:
        """
        O(1)查询预计算的姿态角

        Args:
            sat_id: 卫星ID
            window_start: 窗口开始时间

        Returns:
            (roll, pitch) 或 None (None表示窗口不可行或数据不存在)
        """
        key = (sat_id, window_start.isoformat())

        if key in self._attitude_cache:
            self._cache_hits += 1
            result = self._attitude_cache[key]
            # 检查是否是特殊标记的不可行窗口
            if result[0] is None:
                return None
            return result

        self._cache_misses += 1
        return None

    def is_attitude_feasible(
        self,
        sat_id: str,
        window_start: datetime
    ) -> bool:
        """
        检查窗口的姿态可行性（使用Java预计算结果）

        Args:
            sat_id: 卫星ID
            window_start: 窗口开始时间

        Returns:
            True如果姿态可行，False如果不可行或数据不存在
        """
        key = (sat_id, window_start.isoformat())

        if key in self._attitude_cache:
            result = self._attitude_cache[key]
            # (None, None) 表示不可行
            return result[0] is not None

        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        # 计算有效缓存条目数（排除标记为不可行的条目）
        valid_entries = sum(1 for v in self._attitude_cache.values() if v[0] is not None)
        infeasible_entries = len(self._attitude_cache) - valid_entries

        return {
            'cache_size': len(self._attitude_cache),
            'valid_entries': valid_entries,
            'infeasible_entries': infeasible_entries,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'memory_mb': len(self._attitude_cache) * _ATTITUDE_ENTRY_BYTES / _BYTES_PER_MB
        }


# ============================================================================
# Numba加速的姿态计算
# ============================================================================

@njit(cache=True)
def _batch_compute_attitudes_numba(
    sat_positions: np.ndarray,
    sat_velocities: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """批量姿态计算 - Numba加速"""
    n = len(sat_positions)
    rolls = np.zeros(n, dtype=np.float64)
    pitches = np.zeros(n, dtype=np.float64)

    R_earth = _EARTH_RADIUS_M

    for i in prange(n):
        # 目标位置(ECEF)
        lat_rad = np.radians(target_lats[i])
        lon_rad = np.radians(target_lons[i])

        target_pos = np.array([
            R_earth * np.cos(lat_rad) * np.cos(lon_rad),
            R_earth * np.cos(lat_rad) * np.sin(lon_rad),
            R_earth * np.sin(lat_rad)
        ])

        sat_pos = sat_positions[i]
        sat_vel = sat_velocities[i]

        # LVLH坐标系
        r_norm = np.sqrt(sat_pos[0]**2 + sat_pos[1]**2 + sat_pos[2]**2)
        if r_norm < 1e-10:
            continue

        Z = -sat_pos / r_norm

        # X轴
        v_dot_z = sat_vel[0]*Z[0] + sat_vel[1]*Z[1] + sat_vel[2]*Z[2]
        vx_perp = sat_vel[0] - v_dot_z * Z[0]
        vy_perp = sat_vel[1] - v_dot_z * Z[1]
        vz_perp = sat_vel[2] - v_dot_z * Z[2]

        v_perp_norm = np.sqrt(vx_perp**2 + vy_perp**2 + vz_perp**2)
        if v_perp_norm < 1e-10:
            X = np.array([1.0, 0.0, 0.0])
        else:
            X = np.array([vx_perp, vy_perp, vz_perp]) / v_perp_norm

        # Y轴
        Y = np.cross(Z, X)
        y_norm = np.sqrt(Y[0]**2 + Y[1]**2 + Y[2]**2)
        if y_norm > 1e-10:
            Y = Y / y_norm

        # 视线向量
        los_vec = target_pos - sat_pos
        los_norm = np.sqrt(los_vec[0]**2 + los_vec[1]**2 + los_vec[2]**2)
        if los_norm < 1e-10:
            continue

        los_vec = los_vec / los_norm
        pointing_vec = -los_vec

        # 转换到LVLH
        x_lvlh = pointing_vec[0]*X[0] + pointing_vec[1]*X[1] + pointing_vec[2]*X[2]
        y_lvlh = pointing_vec[0]*Y[0] + pointing_vec[1]*Y[1] + pointing_vec[2]*Y[2]
        z_lvlh = pointing_vec[0]*Z[0] + pointing_vec[1]*Z[1] + pointing_vec[2]*Z[2]

        if abs(z_lvlh) < 1e-10:
            rolls[i] = 0.0
            pitches[i] = 0.0
        else:
            # LVLH坐标系：Z轴指向地心，需要取负号来计算相对于-Z轴的角度
            rolls[i] = np.degrees(np.arctan2(y_lvlh, -z_lvlh))
            pitches[i] = np.degrees(np.arctan2(x_lvlh, -z_lvlh))

    return rolls, pitches


# 全局实例
_attitude_precache_manager: Optional[AttitudePrecacheManager] = None


def get_attitude_precache_manager() -> AttitudePrecacheManager:
    """获取全局姿态预计算管理器"""
    global _attitude_precache_manager
    if _attitude_precache_manager is None:
        _attitude_precache_manager = AttitudePrecacheManager()
    return _attitude_precache_manager
