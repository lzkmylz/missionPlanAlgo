"""
目标模型 - 定义观测目标的属性和约束

支持两类目标：
- 点群目标：分散的点目标集合
- 大区域目标：不规则多边形区域
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math


class TargetType(Enum):
    """目标类型枚举"""
    POINT = "point"      # 点目标
    AREA = "area"        # 区域目标


@dataclass
class GeoPoint:
    """地理坐标点"""
    longitude: float    # 经度（度）
    latitude: float     # 纬度（度）
    altitude: float = 0.0  # 海拔（米）

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.longitude, self.latitude, self.altitude)

    def to_ecef(self) -> Tuple[float, float, float]:
        """转换为ECEF坐标（米）"""
        EARTH_RADIUS = 6371000.0
        lon_rad = math.radians(self.longitude)
        lat_rad = math.radians(self.latitude)
        r = EARTH_RADIUS + self.altitude
        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)
        return (x, y, z)


@dataclass
class Target:
    """
    目标模型

    Attributes:
        id: 目标唯一标识
        name: 目标名称或描述
        target_type: 目标类型（点/区域）
        position: 点目标位置（点目标时必填）
        vertices: 区域目标顶点列表（区域目标时必填）
        priority: 优先级（1-10，10最高）
        required_observations: 需要观测次数
        time_window_start: 观测时间窗口开始
        time_window_end: 观测时间窗口结束
        resolution_required: 分辨率要求（米）
        immediate_downlink: 是否立即回传
    """
    id: str
    name: str = ""
    target_type: TargetType = TargetType.POINT

    # 点目标位置
    position: Optional[GeoPoint] = None

    # 区域目标边界（多边形顶点）
    vertices: List[GeoPoint] = field(default_factory=list)

    # 任务约束
    priority: int = 5                   # 优先级 1-10
    required_observations: int = 1      # 需要观测次数

    # 时间窗口
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    # 其他约束
    resolution_required: float = 10.0   # 分辨率要求（米）
    immediate_downlink: bool = False    # 是否立即回传
    cloud_max_percentage: float = 20.0  # 最大云覆盖率（光学卫星）

    def __post_init__(self):
        """验证和初始化"""
        if self.target_type == TargetType.POINT and self.position is None:
            self.position = GeoPoint(longitude=0.0, latitude=0.0)

    def get_center(self) -> GeoPoint:
        """获取目标中心点"""
        if self.target_type == TargetType.POINT:
            return self.position

        if not self.vertices:
            return GeoPoint(0, 0)

        avg_lon = sum(v.longitude for v in self.vertices) / len(self.vertices)
        avg_lat = sum(v.latitude for v in self.vertices) / len(self.vertices)
        return GeoPoint(avg_lon, avg_lat)

    def get_ecef_position(self) -> Tuple[float, float, float]:
        """获取ECEF坐标（米）"""
        center = self.get_center()
        return center.to_ecef()

    def is_in_time_window(self, t: datetime) -> bool:
        """检查时间是否在观测窗口内"""
        if self.time_window_start and t < self.time_window_start:
            return False
        if self.time_window_end and t > self.time_window_end:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'target_type': self.target_type.value,
            'position': {
                'longitude': self.position.longitude,
                'latitude': self.position.latitude,
                'altitude': self.position.altitude,
            } if self.position else None,
            'priority': self.priority,
            'required_observations': self.required_observations,
            'resolution_required': self.resolution_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Target':
        """从字典创建"""
        target_type = TargetType(data['target_type'])

        position = None
        if data.get('position'):
            pos_data = data['position']
            position = GeoPoint(
                longitude=pos_data['longitude'],
                latitude=pos_data['latitude'],
                altitude=pos_data.get('altitude', 0.0)
            )

        return cls(
            id=data['id'],
            name=data.get('name', ''),
            target_type=target_type,
            position=position,
            priority=data.get('priority', 5),
            required_observations=data.get('required_observations', 1),
            resolution_required=data.get('resolution_required', 10.0),
        )
