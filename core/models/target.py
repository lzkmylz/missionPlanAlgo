"""
目标模型 - 定义观测目标的属性和约束

支持两种目标类型：
- 点目标：特定地理位置的点
- 区域目标：多边形区域，需要分解为条带或网格
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math


class TargetType(Enum):
    """目标类型枚举"""
    POINT = "point"
    AREA = "area"


@dataclass
class GeoPoint:
    """
    地理坐标点

    Attributes:
        longitude: 经度（度）
        latitude: 纬度（度）
        altitude: 海拔高度（米）
    """
    longitude: float
    latitude: float
    altitude: float = 0.0

    def to_ecef(self) -> Tuple[float, float, float]:
        """转换为地心固定坐标（ECEF，米）"""
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
    观测目标模型

    Attributes:
        id: 目标唯一标识
        name: 目标名称
        target_type: 目标类型（点/区域）
        longitude: 经度（点目标）
        latitude: 纬度（点目标）
        area_vertices: 区域顶点列表（区域目标）
        priority: 优先级（1-10）
        required_observations: 需要观测次数
        time_window_start: 时间窗口开始
        time_window_end: 时间窗口结束
        resolution_required: 所需分辨率（米）
        immediate_downlink: 是否需要立即回传
    """
    id: str
    name: str = ""
    target_type: TargetType = TargetType.POINT

    # 点目标参数
    longitude: Optional[float] = None
    latitude: Optional[float] = None

    # 区域目标参数
    area_vertices: List[Tuple[float, float]] = field(default_factory=list)  # [(lon, lat), ...]

    # 观测要求
    priority: int = 1
    required_observations: int = 1
    resolution_required: float = 10.0  # 米

    # 时间约束
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    # 数据回传要求
    immediate_downlink: bool = False

    # 状态（运行时被更新）
    completed_observations: int = 0

    def __post_init__(self):
        """验证数据一致性"""
        if self.target_type == TargetType.POINT:
            if self.longitude is None or self.latitude is None:
                raise ValueError("Point target must have longitude and latitude")
        elif self.target_type == TargetType.AREA:
            if len(self.area_vertices) < 3:
                raise ValueError("Area target must have at least 3 vertices")

    def get_center(self) -> Tuple[float, float]:
        """获取目标中心坐标"""
        if self.target_type == TargetType.POINT:
            return (self.longitude, self.latitude)
        else:
            # 计算多边形中心
            lons = [v[0] for v in self.area_vertices]
            lats = [v[1] for v in self.area_vertices]
            return (sum(lons) / len(lons), sum(lats) / len(lats))

    def get_area(self) -> float:
        """获取区域面积（平方公里）"""
        if self.target_type == TargetType.POINT:
            return 0.0

        # 使用鞋带公式计算多边形面积
        n = len(self.area_vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.area_vertices[i][0] * self.area_vertices[j][1]
            area -= self.area_vertices[j][0] * self.area_vertices[i][1]

        # 转换为平方公里（简化，假设在小范围内）
        km_per_deg = 111.0
        return abs(area) * 0.5 * km_per_deg * km_per_deg

    def is_time_valid(self, dt: datetime) -> bool:
        """检查时间是否在允许窗口内"""
        if self.time_window_start and dt < self.time_window_start:
            return False
        if self.time_window_end and dt > self.time_window_end:
            return False
        return True

    def get_ecef_position(self) -> Tuple[float, float, float]:
        """
        获取地心固定坐标系(ECEF)中的位置

        Returns:
            (x, y, z) in meters
        """
        if self.target_type != TargetType.POINT:
            raise ValueError("Only point targets have a single ECEF position")

        EARTH_RADIUS = 6371000.0  # 米
        lon_rad = math.radians(self.longitude)
        lat_rad = math.radians(self.latitude)

        x = EARTH_RADIUS * math.cos(lat_rad) * math.cos(lon_rad)
        y = EARTH_RADIUS * math.cos(lat_rad) * math.sin(lon_rad)
        z = EARTH_RADIUS * math.sin(lat_rad)

        return (x, y, z)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'target_type': self.target_type.value,
            'longitude': self.longitude,
            'latitude': self.latitude,
            'area_vertices': self.area_vertices,
            'priority': self.priority,
            'required_observations': self.required_observations,
            'resolution_required': self.resolution_required,
            'time_window_start': self.time_window_start.isoformat() if self.time_window_start else None,
            'time_window_end': self.time_window_end.isoformat() if self.time_window_end else None,
            'immediate_downlink': self.immediate_downlink,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Target':
        """从字典创建"""
        target_type = TargetType(data['target_type'])

        return cls(
            id=data['id'],
            name=data.get('name', ''),
            target_type=target_type,
            longitude=data.get('longitude'),
            latitude=data.get('latitude'),
            area_vertices=data.get('area_vertices', []),
            priority=data.get('priority', 1),
            required_observations=data.get('required_observations', 1),
            resolution_required=data.get('resolution_required', 10.0),
            time_window_start=datetime.fromisoformat(data['time_window_start']) if data.get('time_window_start') else None,
            time_window_end=datetime.fromisoformat(data['time_window_end']) if data.get('time_window_end') else None,
            immediate_downlink=data.get('immediate_downlink', False),
        )
