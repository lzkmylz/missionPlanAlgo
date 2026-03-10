"""
批量SAA约束计算器 - Numba向量化优化

功能: 批量处理多个候选的SAA检查，使用Numba JIT加速
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

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

logger = logging.getLogger(__name__)


@dataclass
class BatchSAACandidate:
    """SAA批量检查候选"""
    sat_id: str
    window_start: datetime
    window_end: datetime
    sample_interval: float = 60.0  # 秒


@dataclass
class BatchSAAResult:
    """SAA批量检查结果"""
    feasible: bool
    violation_count: int
    max_separation: float
    sample_count: int
    violation_times: List[datetime] = None


class BatchSAAData:
    """批量SAA数据容器"""

    def __init__(self, n_candidates: int, max_samples: int = 20):
        self.n = n_candidates
        self.max_samples = max_samples

        # 输入数组 [n_candidates x max_samples]
        # 存储每个候选的采样点经纬度（度）
        self.lons = np.zeros((n_candidates, max_samples), dtype=np.float64)
        self.lats = np.zeros((n_candidates, max_samples), dtype=np.float64)
        self.sample_counts = np.zeros(n_candidates, dtype=np.int32)

        # 采样时间（用于记录违规时间）
        self.sample_times = np.zeros((n_candidates, max_samples), dtype=np.float64)

        # SAA椭圆参数（南大西洋异常区）
        # 参考值：中心在西经45度，南纬25度
        self.saa_center_lon = -45.0
        self.saa_center_lat = -25.0
        self.saa_semi_major = 40.0  # 半长轴（度）
        self.saa_semi_minor = 30.0  # 半短轴（度）

        # 输出数组
        self.out_feasible = np.ones(n_candidates, dtype=np.bool_)
        self.out_violation_count = np.zeros(n_candidates, dtype=np.int32)
        self.out_max_separation = np.zeros(n_candidates, dtype=np.float64)


if HAS_NUMBA:
    @njit(cache=True)
    def _check_point_in_saa(lon: float, lat: float,
                           center_lon: float, center_lat: float,
                           semi_major: float, semi_minor: float) -> Tuple[bool, float]:
        """检查单点是否在SAA区域内

        使用归一化椭圆方程: ((lon - c_lon)/a)^2 + ((lat - c_lat)/b)^2 <= 1

        Returns:
            (是否在SAA内, 归一化距离)
        """
        # 处理经度环绕（-180到180）
        dlon = lon - center_lon
        if dlon > 180.0:
            dlon -= 360.0
        elif dlon < -180.0:
            dlon += 360.0

        # 归一化椭圆方程
        norm_lon = dlon / semi_major
        norm_lat = (lat - center_lat) / semi_minor
        separation = np.sqrt(norm_lon ** 2 + norm_lat ** 2)

        return separation <= 1.0, separation

    @njit(parallel=True, cache=True)
    def batch_check_saa_numba(
        lons: np.ndarray,  # [n x max_samples]
        lats: np.ndarray,  # [n x max_samples]
        sample_counts: np.ndarray,  # [n]
        center_lon: float,
        center_lat: float,
        semi_major: float,
        semi_minor: float,
        out_feasible: np.ndarray,
        out_violation_count: np.ndarray,
        out_max_separation: np.ndarray
    ):
        """批量SAA检查 - Numba并行版本

        Args:
            lons: 采样点经度数组 [n_candidates x max_samples]
            lats: 采样点纬度数组 [n_candidates x max_samples]
            sample_counts: 每个候选的实际采样点数 [n_candidates]
            center_lon: SAA中心经度
            center_lat: SAA中心纬度
            semi_major: 椭圆半长轴
            semi_minor: 椭圆半短轴
            out_feasible: 输出可行性数组 [n_candidates]
            out_violation_count: 输出违规次数数组 [n_candidates]
            out_max_separation: 输出最大归一化距离 [n_candidates]
        """
        n = len(sample_counts)

        for i in prange(n):
            n_samples = sample_counts[i]
            violations = 0
            max_sep = 0.0

            for j in range(n_samples):
                lon = lons[i, j]
                lat = lats[i, j]

                # 处理经度环绕
                dlon = lon - center_lon
                if dlon > 180.0:
                    dlon -= 360.0
                elif dlon < -180.0:
                    dlon += 360.0

                # 归一化距离
                norm_lon = dlon / semi_major
                norm_lat = (lat - center_lat) / semi_minor
                separation = np.sqrt(norm_lon ** 2 + norm_lat ** 2)

                if separation > max_sep:
                    max_sep = separation

                # 检查是否在SAA内（separation <= 1.0）
                if separation <= 1.0:
                    violations += 1

            out_violation_count[i] = violations
            out_max_separation[i] = max_sep
            out_feasible[i] = (violations == 0)


class BatchSAACalculator:
    """批量SAA计算器"""

    def __init__(self):
        self.use_numba = HAS_NUMBA

    def prepare_batch_data(
        self,
        candidates: List[BatchSAACandidate],
        position_cache=None,
        orbit_propagator=None
    ) -> BatchSAAData:
        """准备批量SAA检查数据

        Args:
            candidates: SAA检查候选列表
            position_cache: 预计算位置缓存（可选）
            orbit_propagator: 轨道传播器（可选，用于计算位置）

        Returns:
            BatchSAAData: 批量数据容器
        """
        data = BatchSAAData(len(candidates))

        for i, cand in enumerate(candidates):
            # 生成采样时间点
            duration = (cand.window_end - cand.window_start).total_seconds()
            n_samples = min(data.max_samples, max(1, int(duration / cand.sample_interval) + 1))

            data.sample_counts[i] = n_samples

            for j in range(n_samples):
                # 计算采样时间
                t_offset = j * cand.sample_interval
                sample_time = cand.window_start + timedelta(seconds=t_offset)

                # 存储相对时间（秒）- 使用window_start作为参考
                data.sample_times[i, j] = t_offset

                # 获取卫星位置
                position = None
                if position_cache is not None:
                    position = position_cache.get_position(cand.sat_id, sample_time)
                elif orbit_propagator is not None:
                    try:
                        position = orbit_propagator.get_state_at_time(cand.sat_id, sample_time)
                    except:
                        pass

                if position is not None:
                    # 将ECEF位置转换为地理坐标
                    lon, lat = self._ecef_to_geodetic(position[0], position[1], position[2])
                    data.lons[i, j] = lon
                    data.lats[i, j] = lat
                else:
                    # 无位置数据，使用默认值（不会通过SAA检查）
                    data.lons[i, j] = 0.0
                    data.lats[i, j] = 0.0

        return data

    def _ecef_to_geodetic(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """将ECEF坐标转换为地理坐标（简化WGS84）

        Returns:
            (经度, 纬度) 单位：度
        """
        import math

        # 计算经度
        lon = math.degrees(math.atan2(y, x))

        # 简化纬度计算（假设地球为球体）
        r = math.sqrt(x**2 + y**2 + z**2)
        lat = math.degrees(math.asin(z / r)) if r > 0 else 0.0

        return lon, lat

    def compute_batch(self, data: BatchSAAData) -> List[BatchSAAResult]:
        """执行批量SAA计算

        Args:
            data: 批量数据容器

        Returns:
            BatchSAAResult列表
        """
        if self.use_numba and data.n > 0:
            try:
                batch_check_saa_numba(
                    data.lons, data.lats, data.sample_counts,
                    data.saa_center_lon, data.saa_center_lat,
                    data.saa_semi_major, data.saa_semi_minor,
                    data.out_feasible, data.out_violation_count,
                    data.out_max_separation
                )
            except Exception as e:
                logger.warning(f"Numba batch SAA computation failed: {e}, falling back to Python")
                self._compute_batch_python(data)
        else:
            self._compute_batch_python(data)

        # 转换结果为Python对象
        results = []
        base_time = datetime(2024, 1, 1)

        for i in range(data.n):
            # 收集违规时间（如果有）
            violation_times = []
            if data.out_violation_count[i] > 0:
                for j in range(data.sample_counts[i]):
                    lon = data.lons[i, j]
                    lat = data.lats[i, j]

                    # 快速检查是否违规
                    dlon = lon - data.saa_center_lon
                    if dlon > 180.0:
                        dlon -= 360.0
                    elif dlon < -180.0:
                        dlon += 360.0

                    norm_lon = dlon / data.saa_semi_major
                    norm_lat = (lat - data.saa_center_lat) / data.saa_semi_minor
                    separation = (norm_lon ** 2 + norm_lat ** 2) ** 0.5

                    if separation <= 1.0:
                        t_seconds = data.sample_times[i, j]
                        violation_times.append(base_time + timedelta(seconds=t_seconds))

            results.append(BatchSAAResult(
                feasible=data.out_feasible[i],
                violation_count=data.out_violation_count[i],
                max_separation=data.out_max_separation[i],
                sample_count=data.sample_counts[i],
                violation_times=violation_times if violation_times else None
            ))

        return results

    def _compute_batch_python(self, data: BatchSAAData):
        """Python回退版本（当Numba不可用时）"""
        for i in range(data.n):
            n_samples = data.sample_counts[i]
            violations = 0
            max_sep = 0.0

            for j in range(n_samples):
                lon = data.lons[i, j]
                lat = data.lats[i, j]

                # 处理经度环绕
                dlon = lon - data.saa_center_lon
                if dlon > 180.0:
                    dlon -= 360.0
                elif dlon < -180.0:
                    dlon += 360.0

                # 归一化距离
                norm_lon = dlon / data.saa_semi_major
                norm_lat = (lat - data.saa_center_lat) / data.saa_semi_minor
                separation = (norm_lon ** 2 + norm_lat ** 2) ** 0.5

                if separation > max_sep:
                    max_sep = separation

                if separation <= 1.0:
                    violations += 1

            data.out_violation_count[i] = violations
            data.out_max_separation[i] = max_sep
            data.out_feasible[i] = (violations == 0)
