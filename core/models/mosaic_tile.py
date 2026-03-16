"""
拼幅瓦片模型 - 定义区域目标分解后的单个瓦片

用于区域目标的拼幅覆盖规划，每个瓦片代表一个需要观测的子区域
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math

from core.constants import EARTH_RADIUS_M

__all__ = [
    'TileStatus',
    'TilePriorityMode',
    'MosaicTile',
    'TileVisibilityInfo',
]


class TileStatus(Enum):
    """瓦片覆盖状态"""
    PENDING = "pending"      # 待覆盖
    SCHEDULED = "scheduled"  # 已调度
    COMPLETED = "completed"  # 已完成
    SKIPPED = "skipped"      # 跳过（因约束不满足）


class TilePriorityMode(Enum):
    """瓦片优先级模式"""
    UNIFORM = "uniform"           # 统一优先级
    CENTER_FIRST = "center_first" # 中心优先（从区域中心向外）
    EDGE_FIRST = "edge_first"     # 边缘优先（从区域边缘向内）
    CORNER_FIRST = "corner_first" # 角落优先


@dataclass
class MosaicTile:
    """
    拼幅瓦片 - 区域目标分解后的单个观测单元

    Attributes:
        tile_id: 瓦片唯一标识，格式 "{target_id}-T{序号:03d}"
        parent_target_id: 所属区域目标ID
        vertices: 多边形顶点列表 [(lon, lat), ...]
        center: 中心点坐标 (lon, lat)
        area_km2: 面积（平方公里）
        priority: 优先级（1-100，数字越小优先级越高）
        required_observations: 需要观测次数
        coverage_status: 覆盖状态
        covering_task_id: 覆盖此瓦片的任务ID
        coverage_timestamp: 覆盖时间戳
        metadata: 附加元数据
    """
    tile_id: str
    parent_target_id: str
    vertices: List[Tuple[float, float]]  # [(lon, lat), ...]
    center: Tuple[float, float]  # (lon, lat)
    area_km2: float

    # 观测要求
    priority: int = 1
    required_observations: int = 1

    # 覆盖状态
    coverage_status: TileStatus = TileStatus.PENDING
    covering_task_id: Optional[str] = None
    coverage_timestamp: Optional[datetime] = None

    # 覆盖贡献（运行时被更新）
    effective_coverage_km2: float = 0.0  # 实际有效覆盖面积（去除重叠）
    overlap_with_existing: float = 0.0   # 与已覆盖区域的重叠比例

    # 动态计算参数（用于动态瓦片大小）
    optimal_roll_angle: float = 0.0      # 最佳滚转角（度）
    optimal_pitch_angle: float = 0.0     # 最佳俯仰角（度）
    recommended_satellite_id: Optional[str] = None  # 推荐卫星ID

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证数据一致性"""
        if len(self.vertices) < 3:
            raise ValueError("Tile must have at least 3 vertices")

        # 验证中心点是否在顶点范围内（简化检查）
        lons = [v[0] for v in self.vertices]
        lats = [v[1] for v in self.vertices]
        if not (min(lons) <= self.center[0] <= max(lons)):
            # 允许跨越经度边界的情况
            pass
        if not (min(lats) <= self.center[1] <= max(lats)):
            raise ValueError("Tile center latitude outside vertex bounds")

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        获取瓦片的边界框

        Returns:
            (min_lon, min_lat, max_lon, max_lat)
        """
        lons = [v[0] for v in self.vertices]
        lats = [v[1] for v in self.vertices]
        return (min(lons), min(lats), max(lons), max(lats))

    def get_centroid(self) -> Tuple[float, float]:
        """
        计算多边形质心（如果中心点未设置）
        使用多边形质心公式
        """
        if self.center and self.center != (0.0, 0.0):
            return self.center

        n = len(self.vertices)
        if n == 0:
            return (0.0, 0.0)

        cx = sum(v[0] for v in self.vertices) / n
        cy = sum(v[1] for v in self.vertices) / n
        return (cx, cy)

    def get_ecef_center(self) -> Tuple[float, float, float]:
        """获取中心点的ECEF坐标"""
        lon_rad = math.radians(self.center[0])
        lat_rad = math.radians(self.center[1])

        x = EARTH_RADIUS_M * math.cos(lat_rad) * math.cos(lon_rad)
        y = EARTH_RADIUS_M * math.cos(lat_rad) * math.sin(lon_rad)
        z = EARTH_RADIUS_M * math.sin(lat_rad)

        return (x, y, z)

    def mark_as_scheduled(self, task_id: str, timestamp: Optional[datetime] = None) -> None:
        """标记为已调度"""
        self.coverage_status = TileStatus.SCHEDULED
        self.covering_task_id = task_id
        self.coverage_timestamp = timestamp or datetime.now()

    def mark_as_completed(self, effective_coverage: Optional[float] = None) -> None:
        """标记为已完成"""
        self.coverage_status = TileStatus.COMPLETED
        if effective_coverage is not None:
            self.effective_coverage_km2 = effective_coverage

    def is_pending(self) -> bool:
        """检查是否待覆盖"""
        return self.coverage_status == TileStatus.PENDING

    def is_covered(self) -> bool:
        """检查是否已被覆盖"""
        return self.coverage_status in (TileStatus.SCHEDULED, TileStatus.COMPLETED)

    def calculate_distance_to(self, other_lon: float, other_lat: float) -> float:
        """
        计算瓦片中心到另一点的距离（公里）
        使用Haversine公式
        """
        lon1, lat1 = math.radians(self.center[0]), math.radians(self.center[1])
        lon2, lat2 = math.radians(other_lon), math.radians(other_lat)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_M / 1000 * c  # 返回公里

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'tile_id': self.tile_id,
            'parent_target_id': self.parent_target_id,
            'vertices': self.vertices,
            'center': self.center,
            'area_km2': self.area_km2,
            'priority': self.priority,
            'required_observations': self.required_observations,
            'coverage_status': self.coverage_status.value,
            'covering_task_id': self.covering_task_id,
            'coverage_timestamp': self.coverage_timestamp.isoformat() if self.coverage_timestamp else None,
            'effective_coverage_km2': self.effective_coverage_km2,
            'overlap_with_existing': self.overlap_with_existing,
            'optimal_roll_angle': self.optimal_roll_angle,
            'optimal_pitch_angle': self.optimal_pitch_angle,
            'recommended_satellite_id': self.recommended_satellite_id,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MosaicTile':
        """从字典创建"""
        return cls(
            tile_id=data['tile_id'],
            parent_target_id=data['parent_target_id'],
            vertices=[(v[0], v[1]) for v in data['vertices']],
            center=(data['center'][0], data['center'][1]),
            area_km2=data['area_km2'],
            priority=data.get('priority', 1),
            required_observations=data.get('required_observations', 1),
            coverage_status=TileStatus(data.get('coverage_status', 'pending')),
            covering_task_id=data.get('covering_task_id'),
            coverage_timestamp=datetime.fromisoformat(data['coverage_timestamp']) if data.get('coverage_timestamp') else None,
            effective_coverage_km2=data.get('effective_coverage_km2', 0.0),
            overlap_with_existing=data.get('overlap_with_existing', 0.0),
            optimal_roll_angle=data.get('optimal_roll_angle', 0.0),
            optimal_pitch_angle=data.get('optimal_pitch_angle', 0.0),
            recommended_satellite_id=data.get('recommended_satellite_id'),
            metadata=data.get('metadata', {}),
        )


@dataclass
class TileVisibilityInfo:
    """
    瓦片可见性信息 - 存储瓦片与卫星的可见性关系

    Attributes:
        tile_id: 瓦片ID
        satellite_id: 卫星ID
        visibility_windows: 可见性窗口列表
        optimal_roll_at_center: 中心点最优滚转角
        optimal_pitch_at_center: 中心点最优俯仰角
        coverage_fraction: 足迹对瓦片的覆盖比例估计
    """
    tile_id: str
    satellite_id: str
    visibility_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)
    optimal_roll_at_center: float = 0.0
    optimal_pitch_at_center: float = 0.0
    coverage_fraction: float = 1.0  # 假设完全覆盖

    def has_visibility(self) -> bool:
        """检查是否有可见性窗口"""
        return len(self.visibility_windows) > 0

    def get_total_visible_duration(self) -> float:
        """获取总可见时长（秒）"""
        total = 0.0
        for start, end in self.visibility_windows:
            total += (end - start).total_seconds()
        return total
