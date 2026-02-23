"""
SGP4轨道传播器

基于sgp4库实现高精度的卫星轨道传播
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import math

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    Satrec = None
    jday = None


@dataclass
class SatelliteState:
    """卫星状态"""
    timestamp: datetime
    position_eci: Tuple[float, float, float]  # km
    velocity_eci: Tuple[float, float, float]  # km/s
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # km


class SGP4Propagator:
    """
    SGP4轨道传播器

    使用两行轨道根数(TLE)进行高精度轨道传播
    """

    EARTH_RADIUS = 6371.0  # km
    EARTH_ROTATION_RATE = 7.2921158553e-5  # rad/s

    def __init__(self, satrec: Satrec, satellite_id: str = ""):
        """
        初始化传播器

        Args:
            satrec: SGP4卫星记录
            satellite_id: 卫星标识
        """
        self.satrec = satrec
        self.satellite_id = satellite_id

    @classmethod
    def from_tle(cls, line1: str, line2: str, satellite_id: str = "") -> 'SGP4Propagator':
        """从TLE创建传播器"""
        satrec = Satrec.twoline2rv(line1, line2)
        return cls(satrec, satellite_id)

    @classmethod
    def from_satellite(cls, satellite) -> Optional['SGP4Propagator']:
        """从卫星模型创建传播器"""
        if satellite.tle_line1 and satellite.tle_line2:
            return cls.from_tle(satellite.tle_line1, satellite.tle_line2, satellite.id)
        return None

    def propagate(self, dt: datetime) -> SatelliteState:
        """
        传播到指定时间

        Args:
            dt: 目标时间

        Returns:
            SatelliteState: 卫星状态
        """
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

        error, position, velocity = self.satrec.sgp4(jd, fr)

        if error != 0:
            raise RuntimeError(f"SGP4 propagation error code: {error}")

        # 转换为地心坐标
        lat, lon, alt = self._eci_to_lla(position, dt)

        return SatelliteState(
            timestamp=dt,
            position_eci=position,
            velocity_eci=velocity,
            latitude=lat,
            longitude=lon,
            altitude=alt
        )

    def propagate_range(self, start_time: datetime, end_time: datetime,
                       time_step: timedelta = timedelta(seconds=60)) -> List[SatelliteState]:
        """
        传播时间序列

        Args:
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[SatelliteState]: 状态序列
        """
        states = []
        current_time = start_time

        while current_time <= end_time:
            try:
                state = self.propagate(current_time)
                states.append(state)
            except RuntimeError:
                # 跳过传播失败的点
                pass
            current_time += time_step

        return states

    def _eci_to_lla(self, position: Tuple[float, float, float],
                    dt: datetime) -> Tuple[float, float, float]:
        """
        将ECI坐标转换为LLA

        Args:
            position: ECI坐标 (km)
            dt: 时间

        Returns:
            (latitude, longitude, altitude) in degrees and km
        """
        x, y, z = position
        r = math.sqrt(x**2 + y**2 + z**2)

        # 纬度
        lat = math.degrees(math.asin(z / r))

        # 计算格林尼治恒星时
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

        # 简化GMST计算
        d = jd - 2451545.0 + fr
        gmst = (18.697374558 + 24.06570982441908 * d) % 24
        gmst_deg = gmst * 15.0  # 转换为度

        # 经度
        lon = (math.degrees(math.atan2(y, x)) - gmst_deg) % 360
        if lon > 180:
            lon -= 360

        # 高度
        alt = r - self.EARTH_RADIUS

        return (lat, lon, alt)

    def get_subpoint_path(self, start_time: datetime, end_time: datetime,
                          time_step: timedelta = timedelta(seconds=60)) -> List[Tuple[datetime, float, float]]:
        """
        获取星下点轨迹

        Returns:
            List of (timestamp, latitude, longitude)
        """
        states = self.propagate_range(start_time, end_time, time_step)
        return [(s.timestamp, s.latitude, s.longitude) for s in states]
