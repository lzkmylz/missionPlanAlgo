"""
区域覆盖规划模型 - 管理区域目标的完整拼幅覆盖计划

支持两种覆盖策略：
- 最大覆盖：尽可能覆盖更多面积
- 最高收益：在覆盖面积和任务收益间取得平衡
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import math

from .mosaic_tile import MosaicTile, TileStatus, TilePriorityMode

__all__ = [
    'CoverageStrategy',
    'OverlapHandling',
    'CoverageStatistics',
    'AreaCoveragePlan',
    'MultiAreaCoveragePlan',
]


class CoverageStrategy(Enum):
    """区域覆盖策略"""
    MAX_COVERAGE = "max_coverage"   # 最大覆盖：尽可能覆盖更多面积
    MAX_PROFIT = "max_profit"       # 最高收益：考虑优先级和资源的平衡


class OverlapHandling(Enum):
    """重叠处理方式"""
    STRICT = "strict"       # 严格：不允许重叠
    MODERATE = "moderate"   # 适中：允许10%重叠
    RELAXED = "relaxed"     # 宽松：允许20%重叠


@dataclass
class CoverageStatistics:
    """覆盖统计信息"""
    total_tiles: int = 0
    covered_tiles: int = 0
    pending_tiles: int = 0
    total_area_km2: float = 0.0
    covered_area_km2: float = 0.0
    coverage_ratio: float = 0.0
    effective_coverage_km2: float = 0.0  # 去除重叠的实际覆盖
    average_overlap_ratio: float = 0.0
    estimated_completion_time: Optional[datetime] = None


@dataclass
class AreaCoveragePlan:
    """
    区域覆盖规划 - 区域目标的完整拼幅覆盖计划

    Attributes:
        target_id: 区域目标ID
        target_name: 目标名称
        tiles: 瓦片列表
        strategy: 覆盖策略
        overlap_handling: 重叠处理方式
        min_coverage_ratio: 最小覆盖率要求（0-1）
        max_overlap_ratio: 最大允许重叠比例（0-1）
        tile_priority_mode: 瓦片优先级模式
        creation_time: 计划创建时间
        statistics: 覆盖统计
    """
    target_id: str
    target_name: str = ""
    tiles: List[MosaicTile] = field(default_factory=list)

    # 策略配置
    strategy: CoverageStrategy = CoverageStrategy.MAX_COVERAGE
    overlap_handling: OverlapHandling = OverlapHandling.MODERATE
    min_coverage_ratio: float = 0.95
    max_overlap_ratio: float = 0.15  # 默认15%重叠
    tile_priority_mode: TilePriorityMode = TilePriorityMode.CENTER_FIRST

    # 元数据
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 统计信息（运行时被更新）
    statistics: CoverageStatistics = field(default_factory=CoverageStatistics)

    # 动态瓦片大小参数
    dynamic_sizing_enabled: bool = True
    reference_footprint_size_km: Optional[float] = None  # 参考足迹大小

    def __post_init__(self):
        """初始化统计信息"""
        self._update_statistics()

    def _update_statistics(self):
        """更新统计信息"""
        total_area = sum(t.area_km2 for t in self.tiles)
        covered_tiles = [t for t in self.tiles if t.is_covered()]
        covered_area = sum(t.effective_coverage_km2 for t in covered_tiles)

        self.statistics = CoverageStatistics(
            total_tiles=len(self.tiles),
            covered_tiles=len(covered_tiles),
            pending_tiles=len([t for t in self.tiles if t.is_pending()]),
            total_area_km2=total_area,
            covered_area_km2=covered_area,
            coverage_ratio=covered_area / total_area if total_area > 0 else 0.0,
            effective_coverage_km2=covered_area,
        )

    def get_uncovered_tiles(self) -> List[MosaicTile]:
        """获取所有待覆盖的瓦片"""
        return [t for t in self.tiles if t.is_pending()]

    def get_covered_tiles(self) -> List[MosaicTile]:
        """获取已覆盖的瓦片"""
        return [t for t in self.tiles if t.is_covered()]

    def get_tile_by_id(self, tile_id: str) -> Optional[MosaicTile]:
        """根据ID获取瓦片"""
        for tile in self.tiles:
            if tile.tile_id == tile_id:
                return tile
        return None

    def register_tile_coverage(self, tile_id: str, task_id: str,
                                effective_coverage: Optional[float] = None,
                                timestamp: Optional[datetime] = None) -> None:
        """
        注册瓦片覆盖

        Args:
            tile_id: 瓦片ID
            task_id: 任务ID
            effective_coverage: 有效覆盖面积
            timestamp: 覆盖时间戳
        """
        tile = self.get_tile_by_id(tile_id)
        if tile:
            tile.mark_as_scheduled(task_id, timestamp)
            if effective_coverage is not None:
                tile.effective_coverage_km2 = effective_coverage
            self._update_statistics()

    def mark_tile_completed(self, tile_id: str, effective_coverage: Optional[float] = None) -> None:
        """标记瓦片为已完成"""
        tile = self.get_tile_by_id(tile_id)
        if tile:
            tile.mark_as_completed(effective_coverage)
            self._update_statistics()

    def is_fully_covered(self) -> bool:
        """检查是否达到最小覆盖率要求"""
        return self.statistics.coverage_ratio >= self.min_coverage_ratio

    def get_coverage_progress(self) -> float:
        """获取覆盖进度（0-1）"""
        return self.statistics.coverage_ratio

    def get_remaining_area_km2(self) -> float:
        """获取剩余待覆盖面积"""
        return self.statistics.total_area_km2 - self.statistics.covered_area_km2

    def calculate_coverage_score(self) -> float:
        """
        计算覆盖得分（用于评估计划质量）

        最大覆盖策略：覆盖率本身
        最高收益策略：加权覆盖率（考虑优先级）
        """
        if self.strategy == CoverageStrategy.MAX_COVERAGE:
            return self.statistics.coverage_ratio

        elif self.strategy == CoverageStrategy.MAX_PROFIT:
            # 加权覆盖率
            total_weight = sum(1.0 / t.priority for t in self.tiles)
            covered_weight = sum(1.0 / t.priority for t in self.tiles if t.is_covered())
            return covered_weight / total_weight if total_weight > 0 else 0.0

        return 0.0

    def estimate_required_tasks(self, average_footprint_km2: float) -> int:
        """
        估算所需任务数

        Args:
            average_footprint_km2: 平均足迹面积（平方公里）

        Returns:
            估算的任务数
        """
        if average_footprint_km2 <= 0:
            return len(self.tiles)

        # 考虑重叠
        effective_footprint = average_footprint_km2 * (1 - self.max_overlap_ratio)
        required_area = self.statistics.total_area_km2 * self.min_coverage_ratio

        return math.ceil(required_area / effective_footprint)

    def get_priority_sorted_tiles(self) -> List[MosaicTile]:
        """获取按优先级排序的待覆盖瓦片"""
        pending = self.get_uncovered_tiles()
        return sorted(pending, key=lambda t: (t.priority, t.tile_id))

    def get_neighbors(self, tile_id: str) -> List[MosaicTile]:
        """
        获取相邻瓦片（基于空间邻近性）

        简化实现：通过中心点距离判断
        """
        tile = self.get_tile_by_id(tile_id)
        if not tile:
            return []

        neighbors = []
        for other in self.tiles:
            if other.tile_id != tile_id:
                distance = tile.calculate_distance_to(other.center[0], other.center[1])
                # 假设瓦片大小相近，距离小于2倍瓦片对角线视为相邻
                # 简化：使用面积估算边长
                tile_size_km = math.sqrt(tile.area_km2)
                if distance < tile_size_km * 2:
                    neighbors.append(other)

        return neighbors

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        获取整个区域的边界框

        Returns:
            (min_lon, min_lat, max_lon, max_lat)
        """
        if not self.tiles:
            return (0.0, 0.0, 0.0, 0.0)

        all_lons = []
        all_lats = []
        for tile in self.tiles:
            bbox = tile.get_bounding_box()
            all_lons.extend([bbox[0], bbox[2]])
            all_lats.extend([bbox[1], bbox[3]])

        return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))

    def calculate_optimal_tile_size(self, satellite_footprint_km: float,
                                     altitude_km: float = 500.0) -> float:
        """
        计算最优瓦片大小

        Args:
            satellite_footprint_km: 卫星足迹大小（公里）
            altitude_km: 轨道高度（公里）

        Returns:
            建议的瓦片边长（公里）
        """
        if not self.dynamic_sizing_enabled:
            return satellite_footprint_km

        # 动态计算：考虑重叠余量
        overlap_factor = 1 - self.max_overlap_ratio
        optimal_size = satellite_footprint_km * overlap_factor

        # 根据策略调整
        if self.strategy == CoverageStrategy.MAX_COVERAGE:
            # 最大覆盖：稍小的瓦片以确保完整覆盖
            optimal_size *= 0.95
        elif self.strategy == CoverageStrategy.MAX_PROFIT:
            # 最高收益：稍大的瓦片以减少任务数
            optimal_size *= 1.05

        return max(optimal_size, satellite_footprint_km * 0.5)  # 最小为足迹的一半

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'target_id': self.target_id,
            'target_name': self.target_name,
            'tiles': [t.to_dict() for t in self.tiles],
            'strategy': self.strategy.value,
            'overlap_handling': self.overlap_handling.value,
            'min_coverage_ratio': self.min_coverage_ratio,
            'max_overlap_ratio': self.max_overlap_ratio,
            'tile_priority_mode': self.tile_priority_mode.value,
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata,
            'statistics': {
                'total_tiles': self.statistics.total_tiles,
                'covered_tiles': self.statistics.covered_tiles,
                'pending_tiles': self.statistics.pending_tiles,
                'total_area_km2': self.statistics.total_area_km2,
                'covered_area_km2': self.statistics.covered_area_km2,
                'coverage_ratio': self.statistics.coverage_ratio,
                'effective_coverage_km2': self.statistics.effective_coverage_km2,
            },
            'dynamic_sizing_enabled': self.dynamic_sizing_enabled,
            'reference_footprint_size_km': self.reference_footprint_size_km,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AreaCoveragePlan':
        """从字典创建"""
        tiles = [MosaicTile.from_dict(t) for t in data.get('tiles', [])]

        return cls(
            target_id=data['target_id'],
            target_name=data.get('target_name', ''),
            tiles=tiles,
            strategy=CoverageStrategy(data.get('strategy', 'max_coverage')),
            overlap_handling=OverlapHandling(data.get('overlap_handling', 'moderate')),
            min_coverage_ratio=data.get('min_coverage_ratio', 0.95),
            max_overlap_ratio=data.get('max_overlap_ratio', 0.15),
            tile_priority_mode=TilePriorityMode(data.get('tile_priority_mode', 'center_first')),
            creation_time=datetime.fromisoformat(data['creation_time']) if data.get('creation_time') else datetime.now(),
            metadata=data.get('metadata', {}),
            dynamic_sizing_enabled=data.get('dynamic_sizing_enabled', True),
            reference_footprint_size_km=data.get('reference_footprint_size_km'),
        )


@dataclass
class MultiAreaCoveragePlan:
    """
    多区域覆盖规划 - 管理多个区域目标的覆盖计划

    用于同时规划多个区域目标的场景
    """
    plans: Dict[str, AreaCoveragePlan] = field(default_factory=dict)
    global_strategy: CoverageStrategy = CoverageStrategy.MAX_COVERAGE

    def add_plan(self, plan: AreaCoveragePlan):
        """添加区域覆盖计划"""
        self.plans[plan.target_id] = plan

    def get_plan(self, target_id: str) -> Optional[AreaCoveragePlan]:
        """获取区域覆盖计划"""
        return self.plans.get(target_id)

    def get_all_uncovered_tiles(self) -> List[MosaicTile]:
        """获取所有待覆盖的瓦片"""
        tiles = []
        for plan in self.plans.values():
            tiles.extend(plan.get_uncovered_tiles())
        return tiles

    def get_overall_coverage_ratio(self) -> float:
        """获取整体覆盖率"""
        if not self.plans:
            return 0.0

        total_area = sum(p.statistics.total_area_km2 for p in self.plans.values())
        covered_area = sum(p.statistics.covered_area_km2 for p in self.plans.values())

        return covered_area / total_area if total_area > 0 else 0.0

    def are_all_areas_covered(self) -> bool:
        """检查所有区域是否都达到覆盖要求"""
        return all(plan.is_fully_covered() for plan in self.plans.values())

    def get_total_estimated_tasks(self, average_footprint_km2: float) -> int:
        """获取所有区域的估算任务总数"""
        return sum(p.estimate_required_tasks(average_footprint_km2) for p in self.plans.values())
