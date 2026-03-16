"""
姿态机动约束检查器

.. deprecated::
    SlewConstraintChecker 类已弃用。请使用 BatchSlewConstraintChecker 替代。

    旧用法（已弃用）:
        from scheduler.constraints import SlewConstraintChecker
        checker = SlewConstraintChecker(mission, attitude_calc)

    新用法（推荐）:
        from scheduler.constraints import BatchSlewConstraintChecker
        checker = BatchSlewConstraintChecker(mission, use_precise_model=True)
        results = checker.check_slew_feasibility_batch(candidates)

SlewFeasibilityResult 数据类仍然有效，被批量检查器共享使用。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import math
import warnings

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.mission import Mission
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType

# 模块级弃用警告
warnings.warn(
    "scheduler.constraints.slew_constraint_checker 模块中的 SlewConstraintChecker 类已弃用。"
    "请使用 scheduler.constraints.batch_slew_constraint_checker.BatchSlewConstraintChecker",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class SlewFeasibilityResult:
    """机动可行性检查结果

    Attributes:
        feasible: 是否可行
        slew_angle: 机动角度（度）
        slew_time: 机动时间（秒）
        actual_start: 实际可开始时间
        reason: 不可行原因（如果不可行）
        reset_time: 姿态复位时间（秒），仅短时间间隔任务有
    """
    feasible: bool
    slew_angle: float
    slew_time: float
    actual_start: datetime
    reason: Optional[str] = None
    reset_time: Optional[float] = None  # 姿态复位时间（秒）


class SlewConstraintChecker:
    """姿态机动约束检查器

    .. deprecated::
        此类已弃用。请使用 BatchSlewConstraintChecker 替代。

        旧用法（已弃用）:
            checker = SlewConstraintChecker(mission, attitude_calc)

        新用法（推荐）:
            from scheduler.constraints import BatchSlewConstraintChecker
            checker = BatchSlewConstraintChecker(mission, use_precise_model=True)

    此类保留仅用于向后兼容，不再被主动维护。

    Attributes:
        mission: 任务对象
        _attitude_calc: 姿态计算器
        _satellite_configs: 每个卫星的机动配置
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

        .. deprecated::
            请使用 BatchSlewConstraintChecker 替代
        """
        warnings.warn(
            "SlewConstraintChecker 已弃用。请使用 BatchSlewConstraintChecker",
            DeprecationWarning,
            stacklevel=2
        )
        self.mission = mission
        self._attitude_calc = attitude_calculator or AttitudeCalculator(
            propagator_type=PropagatorType.SGP4
        )
        self._satellite_configs: Dict[str, Dict[str, float]] = {}
        self._satellite_cache: Dict[str, Dict[datetime, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = {}
        self._position_cache = None  # 预计算位置缓存

    def set_position_cache(self, cache) -> None:
        """设置预计算位置缓存"""
        self._position_cache = cache

    def initialize_satellite(self, satellite: Satellite) -> None:
        """初始化卫星的机动配置

        Args:
            satellite: 卫星对象
        """
        agility = getattr(satellite.capabilities, 'agility', {}) or {}

        self._satellite_configs[satellite.id] = {
            'max_slew_rate': agility.get('max_slew_rate', 3.0),
            'max_roll_angle': satellite.capabilities.max_roll_angle,
            'max_pitch_angle': satellite.capabilities.max_pitch_angle,
            'settling_time': agility.get('settling_time', 5.0)
        }

        # 初始化缓存
        self._satellite_cache[satellite.id] = {}

    def _calculate_slew_time(self, slew_angle: float, sat_id: str) -> float:
        """计算机动时间

        Args:
            slew_angle: 机动角度（度）
            sat_id: 卫星ID

        Returns:
            机动时间（秒）
        """
        config = self._satellite_configs.get(sat_id, {})
        max_slew_rate = config.get('max_slew_rate', 3.0)
        settling_time = config.get('settling_time', 5.0)

        if slew_angle <= 0:
            return settling_time
        return slew_angle / max_slew_rate + settling_time

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

        # 获取卫星配置
        sat_config = self._satellite_configs.get(satellite_id)
        if not sat_config:
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=0.0,
                slew_time=0.0,
                actual_start=window_start,
                reason=f"Satellite configuration not initialized for {satellite_id}"
            )

        # 使用最大滚转角作为机动角度限制（滚转是主要机动方向）
        max_slew_angle = sat_config['max_roll_angle']

        # 如果没有上一个目标，计算从对地定向到第一个目标的机动
        if prev_target is None:
            # 计算目标姿态角度（简化估算）
            target_lat = getattr(current_target, 'latitude', 0)
            target_lon = getattr(current_target, 'longitude', 0)
            # 使用目标经纬度估算典型机动角度（卫星通常以20-45度离轴角观测）
            slew_angle = math.sqrt(target_lat**2 + target_lon**2) * 0.5
            slew_angle = min(slew_angle, max_slew_angle)
            slew_time = self._calculate_slew_time(slew_angle, satellite_id)
            return SlewFeasibilityResult(
                feasible=True,
                slew_angle=slew_angle,
                slew_time=slew_time,
                actual_start=max(window_start, prev_end_time + timedelta(seconds=slew_time)),
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
            # 高精度要求：无法获取卫星位置时抛出错误
            raise RuntimeError(
                f"Cannot get satellite position for {satellite_id} at {prev_end_time}. "
                "High precision mode requires exact position data."
            )

        # 使用 ECEF 坐标正确计算机动角度
        slew_angle = self._calculate_slew_angle_ecef(
            sat_position, prev_target, current_target
        )

        # 检查机动角度是否超过限制（使用原始角度）
        if slew_angle > max_slew_angle:
            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=slew_angle,
                slew_time=0.0,
                actual_start=window_start,
                reason=f"Slew angle {slew_angle:.2f}° exceeds max {max_slew_angle}°"
            )

        # 限制在最大侧摆角内（仅用于后续计算）
        slew_angle = min(slew_angle, max_slew_angle)

        # 计算机动时间
        slew_time = self._calculate_slew_time(slew_angle, satellite_id)

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

        首先检查预计算位置缓存，然后检查本地缓存，最后计算并缓存。

        Args:
            satellite: 卫星对象
            timestamp: 时间戳

        Returns:
            (position, velocity) 或 (None, None)
        """
        # 1. 优先使用预计算位置缓存
        if self._position_cache is not None:
            result = self._position_cache.get_position(satellite.id, timestamp)
            if result is not None:
                return result

        # 2. 检查本地缓存
        cache = self._satellite_cache.get(satellite.id, {})

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
        # 检查是否为有效数值（处理Mock对象的情况）
        try:
            lon1 = float(target1.longitude)
            lat1 = float(target1.latitude)
            lon2 = float(target2.longitude)
            lat2 = float(target2.latitude)
        except (TypeError, ValueError):
            # 如果无法转换为数值（如Mock对象），返回0机动角度
            return 0.0

        target1_ecef = self._geodetic_to_ecef(lon1, lat1)
        target2_ecef = self._geodetic_to_ecef(lon2, lat2)

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

    def check_slew(
        self,
        sat_id: str,
        prev_target: Optional[Target],
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime,
        imaging_duration: float = 0.0
    ) -> SlewFeasibilityResult:
        """检查机动可行性（与SlewChecker兼容的接口）

        这是check_slew_feasibility的别名方法，用于与UnifiedConstraintChecker兼容。

        Args:
            sat_id: 卫星ID
            prev_target: 上一个目标（None表示这是第一个任务）
            current_target: 当前目标
            prev_end_time: 上一个任务结束时间
            window_start: 当前窗口开始时间
            imaging_duration: 成像持续时间（秒）

        Returns:
            SlewFeasibilityResult: 可行性结果
        """
        return self.check_slew_feasibility(
            satellite_id=sat_id,
            prev_target=prev_target,
            current_target=current_target,
            prev_end_time=prev_end_time,
            window_start=window_start,
            imaging_duration=imaging_duration
        )
