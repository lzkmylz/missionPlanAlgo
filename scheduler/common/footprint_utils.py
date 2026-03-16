"""
Footprint calculation utilities for schedulers.

This module provides shared footprint calculation functions
that can be used by all scheduler implementations.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import math
import logging

logger = logging.getLogger(__name__)


def calculate_task_footprint(
    mission,
    attitude_calculator,
    sat_id: str,
    imaging_start: datetime,
    roll_angle: float,
    pitch_angle: float,
    imaging_mode: Any
) -> Dict[str, Any]:
    """
    计算任务成像足迹

    这是一个共享工具函数，供所有调度器使用来计算成像足迹。

    Args:
        mission: 任务对象
        attitude_calculator: 姿态计算器实例
        sat_id: 卫星ID
        imaging_start: 成像开始时间
        roll_angle: 滚转角（度）
        pitch_angle: 俯仰角（度）
        imaging_mode: 成像模式

    Returns:
        Dict: 包含footprint信息的字典
            - corners: 四脚点坐标列表 [(lon, lat), ...]
            - center: 中心点坐标 (lon, lat)
            - swath_width_km: 幅宽（公里）
            - fov_config: FOV配置
    """
    from core.coverage.footprint_calculator import FootprintCalculator
    from core.models import ImagingMode

    sat = mission.get_satellite_by_id(sat_id)
    if not sat:
        return {'corners': [], 'center': None, 'swath_width_km': 0.0, 'fov_config': {}}

    # 获取卫星位置和星下点
    position = None
    velocity = None

    # 首先尝试从Java预计算轨道数据获取
    try:
        from core.dynamics.orbit_batch_propagator import get_batch_propagator
        propagator = get_batch_propagator()
        if propagator is not None:
            state = propagator.get_state_at_time(sat_id, imaging_start)
            if state is not None:
                # state is a tuple of (position, velocity)
                position, velocity = state
    except Exception:
        pass

    # 如果预计算数据不可用，回退到姿态计算器
    if position is None:
        try:
            position, velocity = attitude_calculator._get_satellite_state(
                sat, imaging_start
            )
        except Exception as e:
            logger.debug(f"Failed to get satellite state for footprint calculation: {e}")
            return {'corners': [], 'center': None, 'swath_width_km': 0.0, 'fov_config': {}}

    # 计算星下点
    sat_radius = math.sqrt(sum(x**2 for x in position))
    altitude_km = (sat_radius - 6371000) / 1000.0

    # 获取FOV配置（优先从imager配置）
    imager_config = sat.capabilities.imager if hasattr(sat.capabilities, 'imager') else {}
    fov_config = imager_config.get('fov_config', {}) if isinstance(imager_config, dict) else {}

    # 如果没有FOV配置，基于swath_width构建
    if not fov_config and hasattr(sat.capabilities, 'swath_width'):
        swath_m = sat.capabilities.swath_width
        fov_config = {
            'fov_type': 'cone',
            'half_angle': math.degrees(math.atan2(swath_m / 2000, altitude_km)),
            'swath_width_km': swath_m / 1000.0
        }

    # 计算星下点经纬度
    try:
        nadir = attitude_calculator._calculate_nadir(position)
        nadir_lon, nadir_lat = nadir.longitude, nadir.latitude
    except Exception:
        # 简化计算
        x, y, z = position
        r = math.sqrt(x*x + y*y + z*z)
        lat = math.degrees(math.asin(z / r))
        lon = math.degrees(math.atan2(y, x))
        nadir_lon, nadir_lat = lon, lat

    # 计算观测角度和方向
    look_angle = math.sqrt(roll_angle**2 + pitch_angle**2)
    # 观测方向：从姿态角计算（0=北，90=东）
    look_direction = math.degrees(math.atan2(roll_angle, pitch_angle))

    # 计算footprint
    calculator = FootprintCalculator(satellite_altitude_km=altitude_km)

    try:
        # 判断使用哪种计算方法
        if fov_config:
            footprint = calculator.calculate_footprint_from_fov(
                satellite_position=position,
                nadir_position=(nadir_lon, nadir_lat),
                look_angle=look_angle,
                look_direction=look_direction,
                fov_config=fov_config,
                imaging_mode=imaging_mode if isinstance(imaging_mode, ImagingMode) else ImagingMode.PUSH_BROOM
            )
        else:
            # 回退到原始方法
            swath_width_km = getattr(sat.capabilities, 'swath_width', 10000) / 1000.0
            footprint = calculator.calculate_footprint(
                satellite_position=position,
                nadir_position=(nadir_lon, nadir_lat),
                look_angle=look_angle,
                swath_width_km=swath_width_km,
                imaging_mode=imaging_mode if isinstance(imaging_mode, ImagingMode) else ImagingMode.PUSH_BROOM
            )

        return {
            'corners': footprint.polygon,
            'center': footprint.center,
            'swath_width_km': footprint.width_km,
            'fov_config': fov_config or footprint.fov_config or {}
        }
    except Exception as e:
        logger.warning(f"Failed to calculate footprint: {e}")
        return {'corners': [], 'center': None, 'swath_width_km': 0.0, 'fov_config': {}}


def fill_footprint_to_task(
    mission,
    attitude_calculator,
    scheduled_task,
    sat_id: str,
    imaging_start: datetime,
    roll_angle: float,
    pitch_angle: float,
    imaging_mode: Any
) -> None:
    """
    为已创建的ScheduledTask填充足迹信息

    Args:
        mission: 任务对象
        attitude_calculator: 姿态计算器实例
        scheduled_task: 已调度的任务对象
        sat_id: 卫星ID
        imaging_start: 成像开始时间
        roll_angle: 滚转角（度）
        pitch_angle: 俯仰角（度）
        imaging_mode: 成像模式
    """
    try:
        footprint_result = calculate_task_footprint(
            mission=mission,
            attitude_calculator=attitude_calculator,
            sat_id=sat_id,
            imaging_start=imaging_start,
            roll_angle=roll_angle,
            pitch_angle=pitch_angle,
            imaging_mode=imaging_mode
        )
        scheduled_task.footprint_corners = footprint_result.get('corners', [])
        scheduled_task.footprint_center = footprint_result.get('center')
        scheduled_task.swath_width_km = footprint_result.get('swath_width_km', 0.0)
        scheduled_task.fov_config = footprint_result.get('fov_config', {})
    except Exception as e:
        logger.warning(f"Failed to fill footprint for task: {e}")
        scheduled_task.footprint_corners = []
        scheduled_task.footprint_center = None
        scheduled_task.swath_width_km = 0.0
        scheduled_task.fov_config = {}


def calculate_haversine_distance(
    lon1: float, lat1: float,
    lon2: float, lat2: float
) -> float:
    """
    使用Haversine公式计算两点间的大地线距离

    Args:
        lon1, lat1: 第一点经纬度（度）
        lon2, lat2: 第二点经纬度（度）

    Returns:
        float: 距离（公里）
    """
    R = 6371.0  # 地球半径（公里）

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2)**2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_footprint_center_from_attitude(
    satellite_position: Tuple[float, float, float],
    roll_angle: float,
    pitch_angle: float
) -> Optional[Tuple[float, float]]:
    """
    根据卫星位置和姿态角计算成像中心点

    Args:
        satellite_position: 卫星ECEF坐标（米）
        roll_angle: 滚转角（度）
        pitch_angle: 俯仰角（度）

    Returns:
        Optional[Tuple[float, float]]: 成像中心点经纬度 (lon, lat)，失败返回None
    """
    try:
        # 计算卫星高度
        sat_radius = math.sqrt(sum(x**2 for x in satellite_position))
        altitude_km = (sat_radius - 6371000) / 1000.0

        # 计算星下点
        x, y, z = satellite_position
        r = math.sqrt(x*x + y*y + z*z)
        nadir_lat = math.degrees(math.asin(z / r))
        nadir_lon = math.degrees(math.atan2(y, x))

        # 计算观测角度和方向
        look_angle = math.sqrt(roll_angle**2 + pitch_angle**2)

        # 如果观测角度接近0，成像中心就是星下点
        if abs(look_angle) < 0.001:
            return (nadir_lon, nadir_lat)

        # 观测方向：从姿态角计算（0=北，90=东）
        look_direction = math.degrees(math.atan2(roll_angle, pitch_angle))

        # 计算地面位移距离
        # 简化模型：displacement = altitude * tan(look_angle)
        displacement_km = altitude_km * math.tan(math.radians(look_angle))

        # 将距离转换为经纬度偏移
        # 考虑纬度影响：1度经度 = 111km * cos(lat)
        lat_rad = math.radians(nadir_lat)
        km_per_deg_lon = 111.32 * math.cos(lat_rad)
        if km_per_deg_lon < 0.001:
            km_per_deg_lon = 0.001

        # 根据观测方向计算偏移
        direction_rad = math.radians(look_direction)
        dlon = (displacement_km * math.sin(direction_rad)) / km_per_deg_lon
        dlat = (displacement_km * math.cos(direction_rad)) / 111.32

        center_lon = nadir_lon + dlon
        center_lat = nadir_lat + dlat

        return (center_lon, center_lat)

    except Exception as e:
        logger.debug(f"Failed to calculate footprint center: {e}")
        return None


def calculate_center_distance_score(
    satellite_position: Tuple[float, float, float],
    roll_angle: float,
    pitch_angle: float,
    target_lon: float,
    target_lat: float,
    max_distance: float = 10.0,
    scale: float = 3.0
) -> float:
    """
    计算成像中心点与目标坐标的距离评分

    距离越近评分越高。使用指数衰减模型：
    - 0度偏差：满分（1.0）
    - 每增加1度，评分指数衰减
    - 超过max_distance后评分为0

    Args:
        satellite_position: 卫星ECEF坐标（米）
        roll_angle: 滚转角（度）
        pitch_angle: 俯仰角（度）
        target_lon: 目标经度（度）
        target_lat: 目标纬度（度）
        max_distance: 最大考虑距离（度），默认10度
        scale: 衰减尺度（度），默认3度

    Returns:
        float: 评分值（0.0 - 1.0）
    """
    # 防御性检查：参数有效性
    if max_distance <= 0:
        logger.debug(f"Invalid max_distance: {max_distance}, using default 10.0")
        max_distance = 10.0
    if scale <= 0:
        logger.debug(f"Invalid scale: {scale}, using default 3.0")
        scale = 3.0
    if not satellite_position or len(satellite_position) != 3:
        logger.debug("Invalid satellite_position")
        return 0.5

    try:
        # 计算成像中心
        footprint_center = calculate_footprint_center_from_attitude(
            satellite_position, roll_angle, pitch_angle
        )

        if footprint_center is None:
            return 0.5  # 默认中等评分

        center_lon, center_lat = footprint_center

        # 计算角度距离（简化计算，不考虑地球曲率）
        dlon = abs(center_lon - target_lon)
        dlat = abs(center_lat - target_lat)

        # 考虑经度随纬度变化
        lat_factor = math.cos(math.radians(target_lat))
        if lat_factor < 0.01:
            lat_factor = 0.01

        angular_distance = math.sqrt((dlon * lat_factor)**2 + dlat**2)

        # 超过最大距离返回0
        if angular_distance >= max_distance:
            return 0.0

        # 指数衰减评分
        score = math.exp(-angular_distance / scale)

        return score

    except Exception as e:
        logger.debug(f"Failed to calculate center distance score: {e}")
        return 0.5  # 默认中等评分
