"""
SAA (南大西洋异常区) 约束检查器

检查卫星在指定时间窗口内是否处于南大西洋异常区。
使用自适应多采样策略，确保窗口任何部分都不在 SAA 中。
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import logging

from core.models.mission import Mission
from core.models.satellite import Satellite
from core.models.saa_boundary import SAABoundaryModel
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType

logger = logging.getLogger(__name__)


@dataclass
class SAAFeasibilityResult:
    """SAA 约束检查结果

    Attributes:
        feasible: 是否可行（任何采样点都不在 SAA 中）
        violation_count: 采样点中违规的数量
        violation_times: 违规时间点列表
        sample_count: 总采样点数
        max_separation: 最大偏离椭圆中心的归一化距离
    """

    feasible: bool
    violation_count: int = 0
    violation_times: List[datetime] = field(default_factory=list)
    sample_count: int = 0
    max_separation: float = 0.0


class SAAConstraintChecker:
    """SAA (南大西洋异常区) 约束检查器

    检查卫星在指定时间窗口内是否处于南大西洋异常区。
    使用自适应多采样策略，确保任务执行期间卫星不会进入 SAA。

    SAA 边界使用简化 NASA 椭圆模型：
    - 中心: 西经 45°, 南纬 25° (巴西海岸附近)
    - 半长轴: 40° (东西向)
    - 半短轴: 30° (南北向)

    Attributes:
        mission: 任务对象
        _attitude_calc: 姿态计算器（用于获取卫星位置）
        _position_cache: 卫星位置缓存
        _saa_model: SAA 边界模型
    """

    # 地球参数
    EARTH_RADIUS = 6371000.0  # 地球平均半径（米）

    def __init__(
        self,
        mission: Mission,
        attitude_calculator: Optional[AttitudeCalculator] = None,
        saa_model: Optional[SAABoundaryModel] = None,
    ):
        """初始化 SAA 约束检查器

        Args:
            mission: 任务对象
            attitude_calculator: 姿态计算器（可选，默认创建新实例）
            saa_model: SAA 边界模型（可选，默认使用标准参数）
        """
        self.mission = mission
        self._attitude_calc = attitude_calculator or AttitudeCalculator(
            propagator_type=PropagatorType.SGP4
        )
        self._saa_model = saa_model or SAABoundaryModel()
        self._position_cache: Dict[str, Dict[datetime, Tuple[float, float]]] = {}
        self._precomputed_position_cache = None  # 预计算位置缓存

    def set_position_cache(self, cache) -> None:
        """设置预计算位置缓存"""
        self._precomputed_position_cache = cache

    def initialize_satellite(self, satellite: Satellite) -> None:
        """初始化卫星的位置缓存

        Args:
            satellite: 卫星对象
        """
        self._position_cache[satellite.id] = {}

    def check_window_feasibility(
        self,
        satellite_id: str,
        window_start: datetime,
        window_end: datetime,
        min_samples: int = 3,
        sample_interval: timedelta = timedelta(seconds=60),
    ) -> SAAFeasibilityResult:
        """检查时间窗口是否可行（任何采样点都不在 SAA 中）

        使用自适应多采样策略：
        - 始终包含窗口起止点
        - 根据窗口长度和 sample_interval 增加中间采样点
        - 确保至少 min_samples 个采样点

        Args:
            satellite_id: 卫星 ID
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            min_samples: 最小采样点数，默认 3（起、止、中点）
            sample_interval: 采样间隔，默认 60 秒

        Returns:
            SAAFeasibilityResult: 检查结果
                - feasible=True: 窗口内所有采样点都不在 SAA 中
                - feasible=False: 至少一个采样点在 SAA 中
        """
        # 获取卫星
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            # 找不到卫星，安全优先：视为不可行
            logger.error(f"Satellite {satellite_id} not found in mission")
            return SAAFeasibilityResult(
                feasible=False,
                violation_count=1,
                violation_times=[window_start],
                sample_count=1,
            )

        # 生成采样时间点
        sample_times = self._generate_sample_times(
            window_start, window_end, min_samples, sample_interval
        )

        # 检查每个采样点
        violations = []
        max_separation = 0.0

        for t in sample_times:
            subpoint = self._get_satellite_subpoint(satellite, t)
            if subpoint:
                lon, lat = subpoint
                is_in_saa, separation = self._is_in_saa(lon, lat)
                max_separation = max(max_separation, separation)
                if is_in_saa:
                    violations.append(t)

        # 构建结果
        feasible = len(violations) == 0
        return SAAFeasibilityResult(
            feasible=feasible,
            violation_count=len(violations),
            violation_times=violations,
            sample_count=len(sample_times),
            max_separation=max_separation,
        )

    def check_single_time(
        self, satellite_id: str, timestamp: datetime
    ) -> Tuple[bool, float]:
        """检查单个时间点卫星是否在 SAA 中

        Args:
            satellite_id: 卫星 ID
            timestamp: 检查时间

        Returns:
            Tuple[bool, float]: (是否在 SAA 中, 归一化偏离距离)
                - (False, 0.0): 不在 SAA 中
                - (True, d): 在 SAA 中，d 为偏离中心的归一化距离
        """
        # 获取卫星
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            # 找不到卫星，安全优先：视为在 SAA 中（阻止任务调度）
            logger.error(f"Satellite {satellite_id} not found in mission")
            return (True, float('inf'))

        # 获取卫星星下点
        subpoint = self._get_satellite_subpoint(satellite, timestamp)
        if not subpoint:
            # 计算失败，保守处理为不在 SAA 中
            return (False, 0.0)

        lon, lat = subpoint
        return self._is_in_saa(lon, lat)

    def _generate_sample_times(
        self,
        start: datetime,
        end: datetime,
        min_samples: int,
        sample_interval: timedelta,
    ) -> List[datetime]:
        """生成自适应采样时间点

        策略：
        - 始终包含起止点
        - 根据窗口长度和 sample_interval 增加中间采样点
        - 确保至少 min_samples 个点
        - 采样点均匀分布

        Args:
            start: 开始时间
            end: 结束时间
            min_samples: 最小采样点数
            sample_interval: 采样间隔

        Returns:
            List[datetime]: 采样时间点列表（已排序）
        """
        duration = (end - start).total_seconds()

        # 起止相同，只返回一个点
        if duration <= 0:
            return [start]

        # 计算需要的采样点数
        num_intervals = max(min_samples - 1, int(duration / sample_interval.total_seconds()))
        step = duration / num_intervals

        # 生成均匀分布的采样点
        samples = []
        for i in range(num_intervals + 1):
            samples.append(start + timedelta(seconds=i * step))

        # 确保包含终点（处理浮点精度问题）
        if samples[-1] != end:
            samples.append(end)

        # 去重并排序
        unique_samples = sorted(set(samples))

        return unique_samples

    def _get_satellite_subpoint(
        self, satellite: Satellite, timestamp: datetime
    ) -> Optional[Tuple[float, float]]:
        """获取卫星星下点位置（经度、纬度）

        首先检查预计算缓存，然后检查本地缓存，最后计算并缓存。

        Args:
            satellite: 卫星对象
            timestamp: 时间戳

        Returns:
            Optional[Tuple[float, float]]: (经度, 纬度) 或 None（计算失败）
        """
        # 1. 优先使用预计算位置缓存（ECEF坐标）
        if self._precomputed_position_cache is not None:
            result = self._precomputed_position_cache.get_position(satellite.id, timestamp)
            if result is not None:
                position, _ = result
                # 将 ECEF 位置转换为地理坐标
                lon, lat, _ = self._ecef_to_geodetic(position[0], position[1], position[2])
                return (lon, lat)

        # 2. 检查本地缓存（已经是经纬度）
        cache = self._position_cache.get(satellite.id, {})

        # 尝试精确匹配
        if timestamp in cache:
            return cache[timestamp]

        # 尝试邻近时间（±1秒）
        for delta in range(-1, 2):
            near_time = timestamp + timedelta(seconds=delta)
            if near_time in cache:
                return cache[near_time]

        # 3. 实时计算卫星状态
        try:
            position, _ = self._attitude_calc._get_satellite_state(satellite, timestamp)
            if position:
                # 将 ECEF 位置转换为地理坐标
                lon, lat, _ = self._ecef_to_geodetic(position[0], position[1], position[2])
                result = (lon, lat)
                # 缓存结果
                if satellite.id not in self._position_cache:
                    self._position_cache[satellite.id] = {}
                self._position_cache[satellite.id][timestamp] = result
                return result
        except Exception as e:
            logger.warning(f"Failed to get satellite subpoint for {satellite.id} at {timestamp}: {e}")

        return None

    def _ecef_to_geodetic(
        self, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """将 ECEF 坐标转换为地理坐标（经度、纬度、高度）

        使用简化算法：球体近似 + 高度修正。
        对于卫星星下点计算（高度通常 < 1000km），精度足够。

        Args:
            x: ECEF X 坐标（米）
            y: ECEF Y 坐标（米）
            z: ECEF Z 坐标（米）

        Returns:
            Tuple[float, float, float]: (经度, 纬度, 高度)
        """
        # 计算经度
        lon = math.degrees(math.atan2(y, x))

        # 计算到地心的距离
        r = math.sqrt(x * x + y * y + z * z)

        # 计算纬度（球体近似）
        lat = math.degrees(math.asin(z / r))

        # 计算高度（相对于平均地球半径）
        h = r - self.EARTH_RADIUS

        return (lon, lat, h)

    def _is_in_saa(self, lon: float, lat: float) -> Tuple[bool, float]:
        """检查坐标是否在 SAA 椭圆区域内

        委托给 SAABoundaryModel 进行计算，确保一致性。

        Args:
            lon: 经度（度）
            lat: 纬度（度）

        Returns:
            Tuple[bool, float]: (是否在 SAA 中, 归一化偏离距离)
        """
        is_inside = self._saa_model.is_inside(lon, lat)

        # 计算归一化偏离距离（用于诊断信息）
        normalized_lon = (lon - self._saa_model.center_lon) / self._saa_model.semi_major
        normalized_lat = (lat - self._saa_model.center_lat) / self._saa_model.semi_minor
        separation = math.sqrt(normalized_lon ** 2 + normalized_lat ** 2)

        return (is_inside, separation)
