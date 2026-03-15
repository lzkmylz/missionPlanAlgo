"""
地面站模型 - 定义地面站属性和天线配置

支持多天线、多地理位置的地面站资源管理
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import math

from core.constants import (
    EARTH_RADIUS_M,
    DEFAULT_DATA_RATE_MBPS,
    DEFAULT_ACQUISITION_TIME_SECONDS,
    DEFAULT_MIN_SWITCH_TIME_SECONDS,
    FAST_SWITCH_TIME_SECONDS,
)


@dataclass
class Antenna:
    """
    天线模型

    Attributes:
        id: 天线唯一标识
        name: 天线名称
        elevation_min: 最小仰角（度）
        elevation_max: 最大仰角（度）
        data_rate: 数据传输速率（Mbps）
        slew_rate: 天线转动速率（度/秒）
        acquisition_time_seconds: 建链耗时（秒）- 指向、捕获、同步
        min_switch_time_seconds: 最小任务切换时间（秒）- 连续任务间缓冲
    """
    id: str
    name: str = ""
    elevation_min: float = 5.0  # 度
    elevation_max: float = 90.0  # 度
    data_rate: float = DEFAULT_DATA_RATE_MBPS  # Mbps
    slew_rate: float = 10.0  # 度/秒
    acquisition_time_seconds: float = DEFAULT_ACQUISITION_TIME_SECONDS  # 秒 - 建链耗时
    min_switch_time_seconds: float = DEFAULT_MIN_SWITCH_TIME_SECONDS  # 秒 - 任务切换缓冲

    def can_track(self, elevation: float) -> bool:
        """检查是否可以在给定仰角跟踪"""
        return self.elevation_min <= elevation <= self.elevation_max

    def calculate_tracking_time(self, azimuth_change: float) -> float:
        """计算转台时间（秒）"""
        return abs(azimuth_change) / self.slew_rate

    def get_effective_switch_time(self, same_satellite: bool = False) -> float:
        """
        获取有效切换时间

        Args:
            same_satellite: 是否为同一卫星的连续任务

        Returns:
            切换时间（秒）
        """
        if same_satellite:
            # 同一卫星连续任务，切换更快
            return min(self.min_switch_time_seconds, FAST_SWITCH_TIME_SECONDS)
        return self.min_switch_time_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'elevation_min': self.elevation_min,
            'elevation_max': self.elevation_max,
            'data_rate': self.data_rate,
            'slew_rate': self.slew_rate,
            'acquisition_time_seconds': self.acquisition_time_seconds,
            'min_switch_time_seconds': self.min_switch_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Antenna':
        return cls(
            id=data['id'],
            name=data.get('name', ''),
            elevation_min=data.get('elevation_min', 5.0),
            elevation_max=data.get('elevation_max', 90.0),
            data_rate=data.get('data_rate', DEFAULT_DATA_RATE_MBPS),
            slew_rate=data.get('slew_rate', 10.0),
            acquisition_time_seconds=data.get('acquisition_time_seconds', DEFAULT_ACQUISITION_TIME_SECONDS),
            min_switch_time_seconds=data.get('min_switch_time_seconds', DEFAULT_MIN_SWITCH_TIME_SECONDS),
        )


@dataclass
class GroundStation:
    """
    地面站模型

    Attributes:
        id: 地面站唯一标识
        name: 地面站名称
        longitude: 经度（度）
        latitude: 纬度（度）
        altitude: 海拔高度（米）
        antennas: 天线列表
    """
    id: str
    name: str = ""
    longitude: float = 0.0  # 度
    latitude: float = 0.0  # 度
    altitude: float = 0.0  # 米
    antennas: List[Antenna] = field(default_factory=list)

    def __post_init__(self):
        """初始化后验证"""
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")

    def add_antenna(self, antenna: Antenna) -> None:
        """添加天线"""
        self.antennas.append(antenna)

    def get_ecef_position(self) -> tuple:
        """
        获取地心固定坐标系(ECEF)中的位置

        Returns:
            (x, y, z) in meters
        """
        lon_rad = math.radians(self.longitude)
        lat_rad = math.radians(self.latitude)

        r = EARTH_RADIUS_M + self.altitude
        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def get_total_data_rate(self) -> float:
        """获取总数据速率（Mbps）"""
        return sum(ant.data_rate for ant in self.antennas)

    def get_available_antennas(self, elevation: float) -> List[Antenna]:
        """获取在指定仰角可用的天线"""
        return [ant for ant in self.antennas if ant.can_track(elevation)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'longitude': self.longitude,
            'latitude': self.latitude,
            'altitude': self.altitude,
            'antennas': [ant.to_dict() for ant in self.antennas],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundStation':
        """从字典创建，支持多种格式"""
        antennas = [Antenna.from_dict(ant_data) for ant_data in data.get('antennas', [])]

        # 支持 location 数组格式 [longitude, latitude, altitude?]
        location = data.get('location')
        if location and len(location) >= 2:
            longitude = location[0]
            latitude = location[1]
            altitude = location[2] if len(location) > 2 else 0.0
        else:
            longitude = data['longitude']
            latitude = data['latitude']
            altitude = data.get('altitude', 0.0)

        return cls(
            id=data['id'],
            name=data.get('name', ''),
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            antennas=antennas,
        )
