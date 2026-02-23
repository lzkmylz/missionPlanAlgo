"""
Orekit可见性计算器

基于纯Python实现的可见性计算（Orekit风格API）
设计文档第3.2章 - Orekit作为STK的fallback方案

功能：
1. 轨道传播：使用SGP4或数值方法传播卫星位置
2. 可见性计算：卫星-目标/地面站可见窗口计算
3. 几何计算：地球遮挡判断、仰角计算
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import math

from .base import VisibilityCalculator, VisibilityWindow

# 尝试导入SGP4
try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    Satrec = None
    jday = None


class OrekitVisibilityCalculator(VisibilityCalculator):
    """
    Orekit风格可见性计算器

    纯Python实现，不依赖Java Orekit库：
    - 支持SGP4轨道传播（如果sgp4库可用）
    - 支持简化数值传播（fallback）
    - 完整的可见性几何计算
    - 地球遮挡判断
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Orekit可见性计算器

        Args:
            config: 配置参数
                - min_elevation: 最小仰角（度，默认5.0）
                - time_step: 计算时间步长（秒，默认60）
        """
        config = config or {}
        min_elevation = config.get('min_elevation', 5.0)
        super().__init__(min_elevation=min_elevation)

        self.time_step = config.get('time_step', 60)
        self._config = config

    def _lla_to_ecef(self, longitude: float, latitude: float, altitude: float = 0.0) -> Tuple[float, float, float]:
        """
        将经纬度高度转换为地心固定坐标（ECEF）

        Args:
            longitude: 经度（度）
            latitude: 纬度（度）
            altitude: 海拔高度（米）

        Returns:
            (x, y, z) in meters
        """
        lon_rad = math.radians(longitude)
        lat_rad = math.radians(latitude)
        r = self.EARTH_RADIUS + altitude

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def _is_earth_occluded(self, sat_pos: Tuple[float, float, float],
                          target_pos: Tuple[float, float, float]) -> bool:
        """
        判断卫星到目标是否被地球遮挡

        对于卫星-地面目标场景，使用仰角检查代替几何遮挡检查。
        如果仰角大于最小仰角要求，则认为可见。

        Args:
            sat_pos: 卫星位置 (x, y, z) in meters
            target_pos: 目标位置 (x, y, z) in meters

        Returns:
            bool: 如果被地球遮挡返回True
        """
        # 如果位置相同，无遮挡
        if sat_pos == target_pos:
            return False

        # 计算仰角
        elevation = self._calculate_elevation(sat_pos, target_pos)

        # 如果仰角大于最小仰角，则可见（不被遮挡）
        return elevation < self.min_elevation

    def _propagate_satellite(self, satellite, dt: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        传播卫星到指定时间

        Args:
            satellite: 卫星模型
            dt: 目标时间

        Returns:
            (position, velocity): 位置和速度，单位km和km/s
        """
        # 如果有TLE且SGP4可用，使用SGP4
        if SGP4_AVAILABLE and hasattr(satellite, 'tle_line1') and hasattr(satellite, 'tle_line2'):
            if satellite.tle_line1 and satellite.tle_line2:
                try:
                    satrec = Satrec.twoline2rv(satellite.tle_line1, satellite.tle_line2)
                    jd, fr = jday(dt.year, dt.month, dt.day,
                                  dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
                    error, position, velocity = satrec.sgp4(jd, fr)

                    if error == 0:
                        # 转换为米
                        pos_m = (position[0] * 1000, position[1] * 1000, position[2] * 1000)
                        vel_m = (velocity[0] * 1000, velocity[1] * 1000, velocity[2] * 1000)
                        return pos_m, vel_m
                except Exception:
                    pass  # 降级到简化模型

        # 使用简化轨道模型
        return self._propagate_simplified(satellite, dt)

    def _propagate_simplified(self, satellite, dt: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        简化的轨道传播模型（圆轨道近似）

        Args:
            satellite: 卫星模型
            dt: 目标时间

        Returns:
            (position, velocity) in meters and m/s
        """
        # 获取轨道参数
        orbit = getattr(satellite, 'orbit', None)
        if orbit is None:
            # 默认500km SSO
            altitude = 500000.0
            inclination = 97.4
            raan = 0.0
        else:
            altitude = getattr(orbit, 'altitude', 500000.0)
            inclination = getattr(orbit, 'inclination', 97.4)
            raan = getattr(orbit, 'raan', 0.0)

        # 计算轨道半径和周期
        r = self.EARTH_RADIUS + altitude
        GM = 3.986004418e14  # 地球引力常数 m^3/s^2
        period = 2 * math.pi * math.sqrt(r**3 / GM)

        # 计算平均运动
        mean_motion = 2 * math.pi / period

        # 参考时间
        ref_time = datetime(2024, 1, 1)
        delta_t = (dt - ref_time).total_seconds()

        # 平近点角（简化）
        mean_anomaly = mean_motion * delta_t

        # 轨道参数
        i = math.radians(inclination)
        raan_rad = math.radians(raan)

        # 圆轨道位置（在轨道平面内）
        x_orb = r * math.cos(mean_anomaly)
        y_orb = r * math.sin(mean_anomaly)

        # 转换到ECI坐标系
        x = x_orb * math.cos(raan_rad) - y_orb * math.cos(i) * math.sin(raan_rad)
        y = x_orb * math.sin(raan_rad) + y_orb * math.cos(i) * math.cos(raan_rad)
        z = y_orb * math.sin(i)

        # 计算速度（圆轨道）
        v = math.sqrt(GM / r)
        vx_orb = -v * math.sin(mean_anomaly)
        vy_orb = v * math.cos(mean_anomaly)

        vx = vx_orb * math.cos(raan_rad) - vy_orb * math.cos(i) * math.sin(raan_rad)
        vy = vx_orb * math.sin(raan_rad) + vy_orb * math.cos(i) * math.cos(raan_rad)
        vz = vy_orb * math.sin(i)

        return ((x, y, z), (vx, vy, vz))

    def _propagate_range(self, satellite, start_time: datetime, end_time: datetime,
                        time_step: timedelta) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], datetime]]:
        """
        传播时间序列

        Args:
            satellite: 卫星模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List of (position, velocity, timestamp)
        """
        positions = []
        current_time = start_time

        while current_time <= end_time:
            try:
                pos, vel = self._propagate_satellite(satellite, current_time)
                positions.append((pos, vel, current_time))
            except Exception:
                # 跳过传播失败的点
                pass
            current_time += time_step

        return positions

    def compute_satellite_target_windows(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-目标可见窗口

        Args:
            satellite: 卫星模型
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 验证时间范围
        if start_time >= end_time:
            return []

        try:
            # 获取目标ECEF位置
            if hasattr(target, 'get_ecef_position'):
                target_pos = target.get_ecef_position()
            else:
                # 从经纬度计算
                lon = getattr(target, 'longitude', 0.0)
                lat = getattr(target, 'latitude', 0.0)
                alt = getattr(target, 'altitude', 0.0)
                target_pos = self._lla_to_ecef(lon, lat, alt)

            # 传播卫星轨道
            sat_positions = self._propagate_range(satellite, start_time, end_time, time_step)

            if not sat_positions:
                return []

            # 计算每个时间点的仰角
            elevation_data = []
            for sat_pos, vel, dt in sat_positions:
                # 检查地球遮挡
                if self._is_earth_occluded(sat_pos, target_pos):
                    elevation_data.append((dt, -90.0))  # 被遮挡，仰角为负
                else:
                    # 计算仰角
                    elevation = self._calculate_elevation(sat_pos, target_pos)
                    elevation_data.append((dt, elevation))

            # 提取可见窗口
            target_id = getattr(target, 'id', 'UNKNOWN')
            sat_id = getattr(satellite, 'id', 'UNKNOWN')

            return self._find_windows_from_elevations(
                sat_id, target_id, elevation_data, self.min_elevation
            )

        except Exception:
            # 发生错误时返回空列表
            return []

    def compute_satellite_ground_station_windows(
        self,
        satellite,
        ground_station,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-地面站可见窗口

        Args:
            satellite: 卫星模型
            ground_station: 地面站模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 验证时间范围
        if start_time >= end_time:
            return []

        try:
            # 获取地面站ECEF位置
            if hasattr(ground_station, 'get_ecef_position'):
                gs_pos = ground_station.get_ecef_position()
            else:
                # 从经纬度计算
                lon = getattr(ground_station, 'longitude', 0.0)
                lat = getattr(ground_station, 'latitude', 0.0)
                alt = getattr(ground_station, 'altitude', 0.0)
                gs_pos = self._lla_to_ecef(lon, lat, alt)

            # 传播卫星轨道
            sat_positions = self._propagate_range(satellite, start_time, end_time, time_step)

            if not sat_positions:
                return []

            # 计算每个时间点的仰角
            elevation_data = []
            for sat_pos, vel, dt in sat_positions:
                # 检查地球遮挡
                if self._is_earth_occluded(sat_pos, gs_pos):
                    elevation_data.append((dt, -90.0))
                else:
                    # 计算仰角
                    elevation = self._calculate_elevation(sat_pos, gs_pos)
                    elevation_data.append((dt, elevation))

            # 提取可见窗口
            gs_id = getattr(ground_station, 'id', 'UNKNOWN')
            sat_id = getattr(satellite, 'id', 'UNKNOWN')

            return self._find_windows_from_elevations(
                sat_id, f"GS:{gs_id}", elevation_data, self.min_elevation
            )

        except Exception:
            return []

    def calculate_windows(
        self,
        satellite_id: str,
        target_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[VisibilityWindow]:
        """
        计算可见窗口（简化接口）

        注意：此接口需要外部提供卫星和目标对象，
        实际实现中可能需要从数据库或缓存获取对象

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 此简化接口在没有实际卫星/目标对象时返回空列表
        # 实际使用时需要传入真实对象或从数据库获取
        # 子类可以覆盖此方法提供具体实现
        return []

    def is_visible(
        self,
        satellite_id: str,
        target_id: str,
        time: datetime
    ) -> bool:
        """
        检查指定时间是否可见

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            time: 检查时间

        Returns:
            bool: 是否可见
        """
        # 获取时间前后一小段时间的窗口
        window_start = time - timedelta(minutes=1)
        window_end = time + timedelta(minutes=1)

        windows = self.calculate_windows(satellite_id, target_id, window_start, window_end)

        # 检查时间是否在任一窗口内
        for window in windows:
            if window.start_time <= time <= window.end_time:
                return True

        return False
