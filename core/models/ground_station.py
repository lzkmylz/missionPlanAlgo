"""
地面站模型 - 定义地面站属性和天线配置

支持多天线、多地理位置的地面站资源管理
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import math


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
    """
    id: str
    name: str = ""
    elevation_min: float = 5.0  # 度
    elevation_max: float = 90.0  # 度
    data_rate: float = 300.0  # Mbps
    slew_rate: float = 10.0  # 度/秒

    def can_track(self, elevation: float) -> bool:
        """检查是否可以在给定仰角跟踪"""
        return self.elevation_min <= elevation <= self.elevation_max

    def calculate_tracking_time(self, azimuth_change: float) -> float:
        """计算转台时间（秒）"""
        return abs(azimuth_change) / self.slew_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'elevation_min': self.elevation_min,
            'elevation_max': self.elevation_max,
            'data_rate': self.data_rate,
            'slew_rate': self.slew_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Antenna':
        return cls(
            id=data['id'],
            name=data.get('name', ''),
            elevation_min=data.get('elevation_min', 5.0),
            elevation_max=data.get('elevation_max', 90.0),
            data_rate=data.get('data_rate', 300.0),
            slew_rate=data.get('slew_rate', 10.0),
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
        EARTH_RADIUS = 6371000.0  # 米
        lon_rad = math.radians(self.longitude)
        lat_rad = math.radians(self.latitude)

        r = EARTH_RADIUS + self.altitude
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
        antennas = [Antenna.from_dict(ant_data) for ant_data in data.get('antennas', [])]
        return cls(
            id=data['id'],
            name=data.get('name', ''),
            longitude=data['longitude'],
            latitude=data['latitude'],
            altitude=data.get('altitude', 0.0),
            antennas=antennas,
        )
