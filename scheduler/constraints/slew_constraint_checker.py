"""
姿态机动约束检查器

将姿态机动约束作为调度器的核心约束条件。
使用正确的 ECEF 坐标系计算机动角度。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import math

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.mission import Mission
from core.dynamics.slew_calculator import SlewCalculator
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType


@dataclass
class SlewFeasibilityResult:
    """机动可行性检查结果

    Attributes:
        feasible: 是否可行
        slew_angle: 机动角度（度）
        slew_time: 机动时间（秒）
        actual_start: 实际可开始时间
        reason: 不可行原因（如果不可行）
    """
    feasible: bool
    slew_angle: float
    slew_time: float
    actual_start: datetime
    reason: Optional[str] = None


class SlewConstraintChecker:
    """姿态机动约束检查器

    将姿态机动约束作为调度器的核心约束条件。
    使用 ECEF 坐标系正确计算机动角度。

    Attributes:
        mission: 任务对象
        _attitude_calc: 姿态计算器
        _slew_calculators: 每个卫星的 SlewCalculator
        _satellite_cache: 卫星位置缓存
    """

    # 地球半径（米）
    EARTH_RADIUS = 6371000.0

    def __init__(
        self,
        mission: Mission,
        attitude_calculator: Optional[AttitudeCalculator] = None
    ):
        """初始化机动约束检查器

        Args:
            mission: 任务对象
            attitude_calculator: 姿态计算器（可选）
        """
        self.mission = mission
        self._attitude_calc = attitude_calculator or AttitudeCalculator(
            propagator_type=PropagatorType.SGP4
        )
        self._slew_calculators: Dict[str, SlewCalculator] = {}
        self._satellite_cache: Dict[str, Dict[datetime, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = {}

    def initialize_satellite(self, satellite: Satellite) -> None:
        """初始化卫星的 SlewCalculator

        Args:
            satellite: 卫星对象
        """
        agility = getattr(satellite.capabilities, 'agility', {}) or {}

        self._slew_calculators[satellite.id] = SlewCalculator(
            max_slew_rate=agility.get('max_slew_rate', 3.0),
            max_slew_angle=satellite.capabilities.max_off_nadir,
            settling_time=agility.get('settling_time', 5.0)
        )

        # 初始化缓存
        self._satellite_cache[satellite.id] = {}

    def check_slew_feasibility(
        self,
        satellite_id: str,
        prev_target: Optional[Target],
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime,
        imaging_duration: float = 0.0
    ) -> SlewFeasibilityResult:
        """检查机动可行性

        这是核心约束检查方法。检查从上一个目标机动到当前目标是否可行。

        Args:
            satellite_id: 卫星ID
            prev_target: 上一个目标（None表示这是第一个任务）
            current_target: 当前目标
            prev_end_time: 上一个任务结束时间
            window_start: 当前窗口开始时间
            imaging_duration: 成像持续时间（秒）

        Returns:
            SlewFeasibilityResult: 可行性结果
        """
        # 获取卫星
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=0.0,
                slew_time=0.0,
                actual_start=window_start,
                reason=f"Satellite {satellite_id} not found"
            )

        # 获取 SlewCalculator
        slew_calc = self._slew_calculators.get(satellite_id)
        if not slew_calc:
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=0.0,
                slew_time=0.0,
                actual_start=window_start,
                reason=f"SlewCalculator not initialized for {satellite_id}"
            )

        # 如果没有上一个目标，不需要机动
        if prev_target is None:
            return SlewFeasibilityResult(
                feasible=True,
                slew_angle=0.0,
                slew_time=slew_calc.settling_time,  # 只需要稳定时间
                actual_start=max(window_start, prev_end_time + timedelta(seconds=slew_calc.settling_time)),
                reason=None
            )

        # 检查目标是否有位置信息
        if not hasattr(current_target, 'latitude') or not hasattr(current_target, 'longitude'):
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=0.0,
                slew_time=0.0,
                actual_start=window_start,
                reason="Current target missing position information"
            )

        if not hasattr(prev_target, 'latitude') or not hasattr(prev_target, 'longitude'):
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=0.0,
                slew_time=0.0,
                actual_start=window_start,
                reason="Previous target missing position information"
            )

        # 获取卫星位置（用于正确计算机动角度）
        sat_position, _ = self._get_satellite_position(satellite, prev_end_time)
        if sat_position is None:
            # 如果无法获取卫星位置，使用简化计算
            slew_angle = self._calculate_simplified_slew_angle(prev_target, current_target)
        else:
            # 使用 ECEF 坐标正确计算机动角度
            slew_angle = self._calculate_slew_angle_ecef(
                sat_position, prev_target, current_target
            )

        # 检查机动角度是否超过限制（使用原始角度）
        if slew_angle > slew_calc.max_slew_angle:
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=slew_angle,
                slew_time=0.0,
                actual_start=window_start,
                reason=f"Slew angle {slew_angle:.2f}° exceeds max {slew_calc.max_slew_angle}°"
            )

        # 限制在最大侧摆角内（仅用于后续计算）
        slew_angle = min(slew_angle, slew_calc.max_slew_angle)

        # 计算机动时间
        slew_time = slew_calc.calculate_slew_time(slew_angle)

        # 计算实际开始时间
        earliest_start = prev_end_time + timedelta(seconds=slew_time)
        actual_start = max(window_start, earliest_start)

        # 检查是否有足够时间（如果提供了成像持续时间）
        if imaging_duration > 0 and window_start is not None:
            window_end = getattr(current_target, 'time_window_end', None)
            if window_end:
                actual_end = actual_start + timedelta(seconds=imaging_duration)
                if actual_end > window_end:
                    return SlewFeasibilityResult(
                        feasible=False,
                        slew_angle=slew_angle,
                        slew_time=slew_time,
                        actual_start=actual_start,
                        reason="Not enough time for imaging after slew"
                    )

        return SlewFeasibilityResult(
            feasible=True,
            slew_angle=slew_angle,
            slew_time=slew_time,
            actual_start=actual_start,
            reason=None
        )

    def _get_satellite_position(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """获取卫星位置和速度

        首先检查缓存，如果未缓存则计算并缓存。

        Args:
            satellite: 卫星对象
            timestamp: 时间戳

        Returns:
            (position, velocity) 或 (None, None)
        """
        # 检查缓存
        cache = self._satellite_cache.get(satellite.id, {})

        # 尝试精确匹配
        if timestamp in cache:
            return cache[timestamp]

        # 尝试邻近时间（±1秒）
        for delta in range(-1, 2):
            near_time = timestamp + timedelta(seconds=delta)
            if near_time in cache:
                return cache[near_time]

        # 计算卫星状态
        try:
            position, velocity = self._attitude_calc._get_satellite_state(satellite, timestamp)
            if position:
                cache[timestamp] = (position, velocity)
            return position, velocity
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to get satellite position for {satellite.id} at {timestamp}: {e}"
            )
            return None, None

    def _calculate_slew_angle_ecef(
        self,
        satellite_position: Tuple[float, float, float],
        target1: Target,
        target2: Target
    ) -> float:
        """使用 ECEF 坐标正确计算机动角度

        这是几何上正确的方法，考虑卫星实际位置。

        Args:
            satellite_position: 卫星 ECEF 位置 (x, y, z) in meters
            target1: 起始目标
            target2: 结束目标

        Returns:
            机动角度（度）
        """
        # 转换目标位置到 ECEF
        target1_ecef = self._geodetic_to_ecef(target1.longitude, target1.latitude)
        target2_ecef = self._geodetic_to_ecef(target2.longitude, target2.latitude)

        # 计算视线向量（从卫星到目标）
        los1 = self._subtract_vectors(target1_ecef, satellite_position)
        los2 = self._subtract_vectors(target2_ecef, satellite_position)

        # 归一化
        los1_norm = self._normalize_vector(los1)
        los2_norm = self._normalize_vector(los2)

        # 计算点积
        dot = self._dot_product(los1_norm, los2_norm)

        # 数值稳定处理
        dot = max(-1.0, min(1.0, dot))

        # 计算角度
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _calculate_simplified_slew_angle(
        self,
        target1: Target,
        target2: Target
    ) -> float:
        """简化机动角度计算（当无法获取卫星位置时）

        使用经纬度差估算角度。注意：这是近似方法，在高纬度不准确。

        Args:
            target1: 起始目标
            target2: 结束目标

        Returns:
            估算的机动角度（度）
        """
        lon_diff = target2.longitude - target1.longitude
        lat_diff = target2.latitude - target1.latitude

        # 考虑经度随纬度收缩
        lat_rad = math.radians((target1.latitude + target2.latitude) / 2)
        lon_diff_corrected = lon_diff * math.cos(lat_rad)

        return math.sqrt(lon_diff_corrected**2 + lat_diff**2)

    def _geodetic_to_ecef(
        self,
        lon: float,
        lat: float,
        alt: float = 0.0
    ) -> Tuple[float, float, float]:
        """将地理坐标转换为 ECEF 坐标

        Args:
            lon: 经度（度）
            lat: 纬度（度）
            alt: 海拔高度（米），默认0

        Returns:
            (x, y, z) in ECEF meters
        """
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        r = self.EARTH_RADIUS + alt

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    @staticmethod
    def _subtract_vectors(
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """向量减法"""
        return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

    @staticmethod
    def _normalize_vector(
        v: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """归一化向量"""
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length < 1e-10:
            return (0.0, 0.0, 0.0)
        return (v[0] / length, v[1] / length, v[2] / length)

    @staticmethod
    def _dot_product(
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float]
    ) -> float:
        """向量点积"""
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
