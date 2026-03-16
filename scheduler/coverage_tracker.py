"""
覆盖追踪器 - 追踪区域目标拼幅覆盖的进度

在调度过程中实时追踪每个区域的覆盖情况，支持：
- 注册已调度的瓦片
- 计算当前覆盖率
- 检查覆盖约束
- 提供未覆盖瓦片列表
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
import logging

from core.models import MosaicTile, AreaCoveragePlan, CoverageStrategy
from scheduler.area_task_utils import AreaObservationTask

__all__ = [
    'TileCoverageRecord',
    'CoverageState',
    'CoverageTracker',
]

logger = logging.getLogger(__name__)


@dataclass
class TileCoverageRecord:
    """瓦片覆盖记录"""
    tile_id: str
    task_id: str
    coverage_timestamp: datetime
    effective_coverage_km2: float
    overlap_with_existing: float
    satellite_id: str


@dataclass
class CoverageState:
    """覆盖状态快照"""
    target_id: str
    covered_tile_ids: Set[str] = field(default_factory=set)
    covered_area_km2: float = 0.0
    coverage_ratio: float = 0.0
    effective_coverage_km2: float = 0.0  # 去除重叠
    overlap_area_km2: float = 0.0
    coverage_records: List[TileCoverageRecord] = field(default_factory=list)


class CoverageTracker:
    """
    覆盖追踪器

    在调度过程中追踪区域目标的覆盖进度。
    支持多区域并行追踪。
    """

    def __init__(self, coverage_plans: Dict[str, AreaCoveragePlan],
                 max_overlap_ratio: float = 0.15):
        """
        初始化覆盖追踪器

        Args:
            coverage_plans: 区域覆盖计划字典 {target_id: plan}
            max_overlap_ratio: 最大允许重叠比例（默认15%）
        """
        self.plans = coverage_plans
        self.max_overlap_ratio = max_overlap_ratio

        # 初始化覆盖状态
        self._coverage_states: Dict[str, CoverageState] = {}
        for target_id, plan in coverage_plans.items():
            self._coverage_states[target_id] = CoverageState(target_id=target_id)

        # 瓦片覆盖记录 {tile_id: TileCoverageRecord}
        self._tile_coverage: Dict[str, TileCoverageRecord] = {}

        # 空间索引（简化：按目标ID分组）
        self._spatial_index: Dict[str, List[MosaicTile]] = {}
        for target_id, plan in coverage_plans.items():
            self._spatial_index[target_id] = list(plan.tiles)

    def register_scheduled_tile(self,
                                tile: MosaicTile,
                                task_id: str,
                                satellite_id: str,
                                timestamp: Optional[datetime] = None,
                                footprint_area_km2: Optional[float] = None) -> float:
        """
        注册已调度的瓦片

        Args:
            tile: 被覆盖的瓦片
            task_id: 覆盖任务ID
            satellite_id: 执行任务的卫星ID
            timestamp: 覆盖时间戳
            footprint_area_km2: 足迹面积（可选，用于精确计算重叠）

        Returns:
            float: 有效覆盖面积（去除重叠）
        """
        target_id = tile.parent_target_id
        timestamp = timestamp or datetime.now()

        # 计算与已覆盖瓦片的重叠
        overlap_area = self._calculate_overlap_with_existing(
            tile, target_id, footprint_area_km2
        )

        # 计算有效覆盖面积
        max_allowed_overlap = tile.area_km2 * self.max_overlap_ratio
        actual_overlap = min(overlap_area, max_allowed_overlap)
        effective_coverage = tile.area_km2 - actual_overlap

        # 创建覆盖记录
        record = TileCoverageRecord(
            tile_id=tile.tile_id,
            task_id=task_id,
            coverage_timestamp=timestamp,
            effective_coverage_km2=effective_coverage,
            overlap_with_existing=actual_overlap / tile.area_km2 if tile.area_km2 > 0 else 0,
            satellite_id=satellite_id
        )

        # 更新追踪状态
        self._tile_coverage[tile.tile_id] = record

        state = self._coverage_states.get(target_id)
        if state:
            state.covered_tile_ids.add(tile.tile_id)
            state.coverage_records.append(record)
            state.covered_area_km2 += tile.area_km2
            state.effective_coverage_km2 += effective_coverage
            state.overlap_area_km2 += actual_overlap

            # 更新覆盖率
            plan = self.plans.get(target_id)
            if plan:
                state.coverage_ratio = state.effective_coverage_km2 / plan.statistics.total_area_km2

        # 更新瓦片状态
        tile.mark_as_scheduled(task_id, timestamp)
        tile.effective_coverage_km2 = effective_coverage
        tile.overlap_with_existing = actual_overlap / tile.area_km2 if tile.area_km2 > 0 else 0

        logger.debug(f"注册瓦片覆盖: {tile.tile_id} 有效覆盖={effective_coverage:.2f}km² "
                    f"(重叠={actual_overlap:.2f}km²)")

        return effective_coverage

    def _calculate_overlap_with_existing(self,
                                         tile: MosaicTile,
                                         target_id: str,
                                         footprint_area_km2: Optional[float] = None) -> float:
        """
        计算与已覆盖瓦片的重叠面积

        简化实现：基于空间邻近性估算
        """
        overlap_area = 0.0
        state = self._coverage_states.get(target_id)

        if not state or not state.covered_tile_ids:
            return 0.0

        tile_size = math.sqrt(tile.area_km2)

        for covered_tile_id in state.covered_tile_ids:
            covered_record = self._tile_coverage.get(covered_tile_id)
            if not covered_record:
                continue

            # 获取已覆盖瓦片
            plan = self.plans.get(target_id)
            if not plan:
                continue

            covered_tile = plan.get_tile_by_id(covered_tile_id)
            if not covered_tile:
                continue

            # 计算距离
            distance = tile.calculate_distance_to(
                covered_tile.center[0], covered_tile.center[1]
            )

            # 如果距离小于瓦片尺寸，认为有重叠
            if distance < tile_size * 1.2:  # 1.2倍作为相邻阈值
                # 估算重叠面积（简化：假设固定比例）
                estimated_overlap = min(tile.area_km2, covered_tile.area_km2) * 0.15
                overlap_area += estimated_overlap

        return overlap_area

    def get_coverage_state(self, target_id: str) -> Optional[CoverageState]:
        """获取区域的覆盖状态"""
        return self._coverage_states.get(target_id)

    def get_coverage_ratio(self, target_id: str) -> float:
        """获取区域的当前覆盖率"""
        state = self._coverage_states.get(target_id)
        return state.coverage_ratio if state else 0.0

    def get_effective_coverage_ratio(self, target_id: str) -> float:
        """获取有效覆盖率（去除重叠）"""
        state = self._coverage_states.get(target_id)
        if not state:
            return 0.0

        plan = self.plans.get(target_id)
        if not plan or plan.statistics.total_area_km2 == 0:
            return 0.0

        return state.effective_coverage_km2 / plan.statistics.total_area_km2

    def get_uncovered_tiles(self, target_id: str) -> List[MosaicTile]:
        """获取区域中未覆盖的瓦片"""
        plan = self.plans.get(target_id)
        if not plan:
            return []

        state = self._coverage_states.get(target_id)
        if not state:
            return list(plan.tiles)

        return [t for t in plan.tiles if t.tile_id not in state.covered_tile_ids]

    def get_covered_tiles(self, target_id: str) -> List[MosaicTile]:
        """获取区域中已覆盖的瓦片"""
        plan = self.plans.get(target_id)
        if not plan:
            return []

        state = self._coverage_states.get(target_id)
        if not state:
            return []

        return [t for t in plan.tiles if t.tile_id in state.covered_tile_ids]

    def is_area_fully_covered(self, target_id: str) -> bool:
        """检查区域是否达到完全覆盖要求"""
        plan = self.plans.get(target_id)
        if not plan:
            return False

        effective_ratio = self.get_effective_coverage_ratio(target_id)
        return effective_ratio >= plan.min_coverage_ratio

    def get_overall_coverage_ratio(self) -> float:
        """获取所有区域的整体覆盖率"""
        if not self.plans:
            return 0.0

        total_area = sum(
            p.statistics.total_area_km2 for p in self.plans.values()
        )
        total_effective = sum(
            s.effective_coverage_km2 for s in self._coverage_states.values()
        )

        return total_effective / total_area if total_area > 0 else 0.0

    def are_all_areas_covered(self) -> bool:
        """检查所有区域是否都达到覆盖要求"""
        return all(
            self.is_area_fully_covered(target_id)
            for target_id in self.plans.keys()
        )

    def get_remaining_required_coverage(self, target_id: str) -> float:
        """
        获取区域还需要覆盖的面积

        Returns:
            还需要覆盖的平方公里数
        """
        plan = self.plans.get(target_id)
        if not plan:
            return 0.0

        required_area = plan.statistics.total_area_km2 * plan.min_coverage_ratio
        state = self._coverage_states.get(target_id)
        covered = state.effective_coverage_km2 if state else 0.0

        return max(0.0, required_area - covered)

    def get_priority_sorted_uncovered_tiles(self, target_id: str) -> List[MosaicTile]:
        """获取按优先级排序的未覆盖瓦片"""
        tiles = self.get_uncovered_tiles(target_id)
        return sorted(tiles, key=lambda t: (t.priority, t.tile_id))

    def estimate_remaining_tasks(self, target_id: str,
                                  average_coverage_per_task: float = 10.0) -> int:
        """
        估算还需要多少任务才能完成覆盖

        Args:
            target_id: 区域目标ID
            average_coverage_per_task: 每个任务的平均覆盖面积（km²）

        Returns:
            估算的任务数
        """
        remaining_area = self.get_remaining_required_coverage(target_id)
        return math.ceil(remaining_area / average_coverage_per_task) if average_coverage_per_task > 0 else 0

    def check_coverage_constraint(self, target_id: str) -> Tuple[bool, str]:
        """
        检查覆盖约束是否满足

        Returns:
            (是否满足, 原因)
        """
        plan = self.plans.get(target_id)
        if not plan:
            return False, "未找到覆盖计划"

        effective_ratio = self.get_effective_coverage_ratio(target_id)

        if effective_ratio >= plan.min_coverage_ratio:
            return True, f"覆盖率 {effective_ratio:.1%} 满足要求 ({plan.min_coverage_ratio:.1%})"
        else:
            return False, f"覆盖率 {effective_ratio:.1%} 未达到要求 ({plan.min_coverage_ratio:.1%})"

    def get_coverage_statistics(self, target_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取覆盖统计信息

        Args:
            target_id: 特定区域ID，None则返回整体统计

        Returns:
            统计信息字典
        """
        if target_id:
            plan = self.plans.get(target_id)
            state = self._coverage_states.get(target_id)

            if not plan or not state:
                return {}

            return {
                'target_id': target_id,
                'total_tiles': len(plan.tiles),
                'covered_tiles': len(state.covered_tile_ids),
                'pending_tiles': len(plan.tiles) - len(state.covered_tile_ids),
                'total_area_km2': plan.statistics.total_area_km2,
                'covered_area_km2': state.covered_area_km2,
                'effective_coverage_km2': state.effective_coverage_km2,
                'coverage_ratio': state.coverage_ratio,
                'effective_coverage_ratio': self.get_effective_coverage_ratio(target_id),
                'min_coverage_required': plan.min_coverage_ratio,
                'is_fully_covered': self.is_area_fully_covered(target_id),
                'overlap_area_km2': state.overlap_area_km2,
                'average_overlap_ratio': (state.overlap_area_km2 / state.covered_area_km2
                                          if state.covered_area_km2 > 0 else 0),
            }
        else:
            # 整体统计
            return {
                'total_areas': len(self.plans),
                'fully_covered_areas': sum(1 for tid in self.plans.keys()
                                          if self.is_area_fully_covered(tid)),
                'overall_coverage_ratio': self.get_overall_coverage_ratio(),
                'total_tiles': sum(len(p.tiles) for p in self.plans.values()),
                'total_covered_tiles': len(self._tile_coverage),
                'total_overlap_km2': sum(s.overlap_area_km2 for s in self._coverage_states.values()),
            }

    def get_best_next_tile(self, target_id: str,
                           strategy: CoverageStrategy = CoverageStrategy.MAX_COVERAGE) -> Optional[MosaicTile]:
        """
        获取下一个最佳覆盖瓦片

        基于策略选择最优的未覆盖瓦片。

        Args:
            target_id: 区域目标ID
            strategy: 覆盖策略

        Returns:
            MosaicTile 或 None（如果已全部覆盖）
        """
        pending = self.get_priority_sorted_uncovered_tiles(target_id)

        if not pending:
            return None

        if strategy == CoverageStrategy.MAX_COVERAGE:
            # 最大覆盖：优先选择大面积瓦片
            return max(pending, key=lambda t: t.area_km2)

        elif strategy == CoverageStrategy.MAX_PROFIT:
            # 最高收益：优先选择高优先级且大面积的瓦片
            return max(pending, key=lambda t: t.area_km2 / t.priority)

        return pending[0]  # 默认返回第一个（按优先级排序）

    def reset(self, target_id: Optional[str] = None):
        """
        重置覆盖状态

        Args:
            target_id: 特定区域ID，None则重置所有
        """
        if target_id:
            if target_id in self._coverage_states:
                self._coverage_states[target_id] = CoverageState(target_id=target_id)

            # 清除该目标的瓦片覆盖记录
            plan = self.plans.get(target_id)
            if plan:
                for tile in plan.tiles:
                    self._tile_coverage.pop(tile.tile_id, None)
                    tile.coverage_status = MosaicTile().coverage_status  # 重置为PENDING
        else:
            # 重置所有
            for tid in self.plans.keys():
                self._coverage_states[tid] = CoverageState(target_id=tid)
            self._tile_coverage.clear()

    def export_coverage_map(self, target_id: str) -> Dict[str, Any]:
        """
        导出覆盖地图数据

        用于可视化和分析。

        Returns:
            覆盖地图数据字典
        """
        plan = self.plans.get(target_id)
        state = self._coverage_states.get(target_id)

        if not plan or not state:
            return {}

        return {
            'target_id': target_id,
            'target_boundary': plan.get_bounding_box(),
            'tiles': [
                {
                    'tile_id': t.tile_id,
                    'center': t.center,
                    'vertices': t.vertices,
                    'area_km2': t.area_km2,
                    'is_covered': t.tile_id in state.covered_tile_ids,
                    'priority': t.priority,
                }
                for t in plan.tiles
            ],
            'coverage_ratio': state.coverage_ratio,
            'statistics': self.get_coverage_statistics(target_id),
        }
