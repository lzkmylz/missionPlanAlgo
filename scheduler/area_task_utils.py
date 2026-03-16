"""
区域目标任务工具模块

为调度算法提供区域目标（拼幅覆盖）的支持。
支持与点目标混合规划。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

from core.models import (
    Target, TargetType, MosaicTile, AreaCoveragePlan,
    CoverageStrategy
)
from core.orbit.visibility import VisibilityWindow
from core.decomposer import MosaicPlanner

__all__ = [
    'AreaObservationTask',
    'MixedTaskList',
    'create_area_observation_tasks',
    'create_mixed_task_list',
    'calculate_area_coverage_score',
    'calculate_coverage_fitness',
    'get_uncovered_tiles',
    'update_coverage_contribution',
    'merge_area_tasks_into_schedule',
]

logger = logging.getLogger(__name__)


@dataclass
class AreaObservationTask:
    """
    区域观测任务 - 代表对单个瓦片的观测任务

    这是 ObservationTask 的扩展，专门用于区域目标的拼幅覆盖。
    每个 AreaObservationTask 对应一个 MosaicTile 的一次观测。

    Attributes:
        task_id: 任务唯一标识
        target_id: 原始区域目标ID
        tile_id: 瓦片ID
        tile: 对应的 MosaicTile 对象
        observation_idx: 第几次观测（0-based）
        required_observations: 需要观测次数
        priority: 优先级
        center_lon: 中心点经度
        center_lat: 中心点纬度
        area_km2: 瓦片面积
        coverage_contribution: 覆盖贡献（去除重叠的有效面积）
        is_mosaic_task: 是否为拼幅任务（总是True）
    """
    task_id: str
    target_id: str
    tile_id: str
    tile: MosaicTile
    observation_idx: int
    required_observations: int
    priority: int
    center_lon: float
    center_lat: float
    area_km2: float
    coverage_contribution: float = 0.0  # 有效覆盖贡献
    is_mosaic_task: bool = True

    # 可选的可见性窗口（预计算）
    visibility_windows: List[VisibilityWindow] = field(default_factory=list)

    # 兼容imaging_time_calculator的target_type属性
    target_type: TargetType = field(default=TargetType.AREA)

    # 兼容调度器的时间窗口属性
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    @property
    def id(self) -> str:
        """兼容原始Target的id属性"""
        return self.task_id

    @property
    def longitude(self) -> float:
        """兼容点目标的longitude属性"""
        return self.center_lon

    @property
    def latitude(self) -> float:
        """兼容点目标的latitude属性"""
        return self.center_lat

    def to_observation_task_dict(self) -> Dict[str, Any]:
        """转换为 ObservationTask 兼容的字典"""
        return {
            'task_id': self.task_id,
            'target_id': self.target_id,
            'target_name': f"{self.target_id} - {self.tile_id}",
            'observation_idx': self.observation_idx,
            'required_observations': self.required_observations,
            'priority': self.priority,
            'longitude': self.center_lon,
            'latitude': self.center_lat,
            'target_type': TargetType.AREA,
            'is_mosaic_task': True,
            'tile_id': self.tile_id,
            'area_km2': self.area_km2,
            'coverage_contribution': self.coverage_contribution,
        }


@dataclass
class MixedTaskList:
    """
    混合任务列表 - 包含点目标和区域目标的任务

    用于统一处理点目标和区域目标的混合规划场景。
    """
    point_tasks: List[Any] = field(default_factory=list)  # ObservationTask 或其他点任务
    area_tasks: List[AreaObservationTask] = field(default_factory=list)

    # 区域覆盖计划字典 {target_id: AreaCoveragePlan}
    area_coverage_plans: Dict[str, AreaCoveragePlan] = field(default_factory=dict)

    def get_all_tasks(self) -> List[Any]:
        """获取所有任务（点目标 + 区域目标）"""
        return self.point_tasks + self.area_tasks

    def get_area_tasks_for_target(self, target_id: str) -> List[AreaObservationTask]:
        """获取特定区域目标的所有任务"""
        return [t for t in self.area_tasks if t.target_id == target_id]

    def has_area_targets(self) -> bool:
        """检查是否有区域目标"""
        return len(self.area_tasks) > 0

    def has_point_targets(self) -> bool:
        """检查是否有点目标"""
        return len(self.point_tasks) > 0

    def get_total_task_count(self) -> int:
        """获取总任务数"""
        return len(self.point_tasks) + len(self.area_tasks)


def create_area_observation_tasks(
    coverage_plan: AreaCoveragePlan,
    visibility_cache: Optional[Dict[str, List[VisibilityWindow]]] = None
) -> List[AreaObservationTask]:
    """
    根据区域覆盖计划创建观测任务列表

    Args:
        coverage_plan: 区域覆盖计划
        visibility_cache: 可选的可见性窗口缓存 {tile_id: [windows]}

    Returns:
        List[AreaObservationTask]: 区域观测任务列表
    """
    tasks = []

    for tile in coverage_plan.tiles:
        required = tile.required_observations

        for idx in range(required):
            task_id = f"{tile.tile_id}-OBS{idx+1}"

            # 获取该瓦片的可见性窗口（如果有缓存）
            tile_windows = visibility_cache.get(tile.tile_id, []) if visibility_cache else []

            task = AreaObservationTask(
                task_id=task_id,
                target_id=coverage_plan.target_id,
                tile_id=tile.tile_id,
                tile=tile,
                observation_idx=idx,
                required_observations=required,
                priority=tile.priority,
                center_lon=tile.center[0],
                center_lat=tile.center[1],
                area_km2=tile.area_km2,
                coverage_contribution=tile.area_km2,  # 初始为完整面积，后续计算去除重叠
                visibility_windows=tile_windows,
            )
            tasks.append(task)

    logger.info(f"为区域目标 {coverage_plan.target_id} 创建了 {len(tasks)} 个观测任务"
                f"（来自 {len(coverage_plan.tiles)} 个瓦片）")

    return tasks


def create_mixed_task_list(
    targets: List[Target],
    satellites: List[Any],
    mosaic_planner: Optional[MosaicPlanner] = None,
    enable_area_coverage: bool = True,
    area_coverage_config: Optional[Dict[str, Any]] = None
) -> MixedTaskList:
    """
    为目标列表创建混合任务列表

    自动识别点目标和区域目标，为区域目标生成拼幅覆盖计划。

    Args:
        targets: 目标列表（点目标或区域目标）
        satellites: 可用卫星列表
        mosaic_planner: 拼幅规划器（可选，默认创建新实例）
        enable_area_coverage: 是否启用区域覆盖
        area_coverage_config: 区域覆盖配置

    Returns:
        MixedTaskList: 混合任务列表
    """
    from .frequency_utils import ObservationTask, create_observation_tasks

    result = MixedTaskList()

    if mosaic_planner is None and enable_area_coverage:
        mosaic_planner = MosaicPlanner()

    config = area_coverage_config or {}

    # 分离点目标和区域目标
    point_targets = []
    area_targets = []

    for target in targets:
        if target.target_type == TargetType.AREA and enable_area_coverage:
            # 检查是否需要拼幅覆盖
            if getattr(target, 'mosaic_required', False) or config.get('force_mosaic', False):
                area_targets.append(target)
            else:
                # 区域目标但不启用拼幅，转换为点目标（使用中心点）
                point_targets.append(target)
        else:
            point_targets.append(target)

    # 为点目标创建观测任务
    if point_targets:
        # 将区域目标（非拼幅模式）转换为点目标
        converted_point_targets = []
        for t in point_targets:
            if t.target_type == TargetType.AREA:
                # 创建临时点目标
                center = t.get_center()
                temp_target = Target(
                    id=t.id,
                    name=t.name,
                    target_type=TargetType.POINT,
                    longitude=center[0],
                    latitude=center[1],
                    priority=t.priority,
                    required_observations=t.required_observations,
                )
                converted_point_targets.append(temp_target)
            else:
                converted_point_targets.append(t)

        result.point_tasks = create_observation_tasks(converted_point_targets)

    # 为区域目标创建拼幅覆盖计划和观测任务
    for area_target in area_targets:
        try:
            coverage_plan = mosaic_planner.create_coverage_plan(
                target=area_target,
                satellites=satellites,
                strategy=config.get('strategy'),
                overlap_ratio=config.get('overlap_ratio'),
                priority_mode=config.get('priority_mode'),
                min_coverage_ratio=config.get('min_coverage_ratio', 0.95),
            )

            result.area_coverage_plans[area_target.id] = coverage_plan

            area_tasks = create_area_observation_tasks(coverage_plan)
            result.area_tasks.extend(area_tasks)

        except Exception as e:
            logger.error(f"为区域目标 {area_target.id} 创建覆盖计划失败: {e}")
            # 失败时退化为点目标处理
            center = area_target.get_center()
            temp_target = Target(
                id=area_target.id,
                name=area_target.name,
                target_type=TargetType.POINT,
                longitude=center[0],
                latitude=center[1],
                priority=area_target.priority,
                required_observations=area_target.required_observations,
            )
            result.point_tasks.extend(create_observation_tasks([temp_target]))

    logger.info(f"混合任务列表创建完成: {len(result.point_tasks)} 点任务, "
                f"{len(result.area_tasks)} 区域任务")

    return result


def calculate_area_coverage_score(
    scheduled_area_tasks: List[AreaObservationTask],
    coverage_plan: AreaCoveragePlan
) -> float:
    """
    计算区域覆盖得分

    用于评估调度结果的区域覆盖质量。

    Args:
        scheduled_area_tasks: 已调度的区域任务
        coverage_plan: 原始覆盖计划

    Returns:
        float: 覆盖得分（0-1）
    """
    if not coverage_plan.tiles:
        return 0.0

    # 统计已覆盖的瓦片
    covered_tile_ids = set(t.tile_id for t in scheduled_area_tasks)

    if coverage_plan.strategy == CoverageStrategy.MAX_COVERAGE:
        # 最大覆盖：覆盖率本身
        coverage_ratio = len(covered_tile_ids) / len(coverage_plan.tiles)
        return coverage_ratio

    elif coverage_plan.strategy == CoverageStrategy.MAX_PROFIT:
        # 最高收益：考虑优先级的加权覆盖率
        total_weight = sum(1.0 / t.priority for t in coverage_plan.tiles)
        covered_weight = sum(
            1.0 / t.priority
            for t in coverage_plan.tiles
            if t.tile_id in covered_tile_ids
        )
        return covered_weight / total_weight if total_weight > 0 else 0.0

    return 0.0


def calculate_coverage_fitness(
    coverage_plans: Dict[str, AreaCoveragePlan],
    scheduled_tasks: List[Any],
    base_score: float = 0.0
) -> float:
    """
    计算覆盖适应度（用于元启发式算法）

    Args:
        coverage_plans: 所有区域覆盖计划 {target_id: plan}
        scheduled_tasks: 所有已调度任务
        base_score: 基础分数

    Returns:
        float: 总适应度分数
    """
    score = base_score

    # 分离区域任务
    area_tasks = [t for t in scheduled_tasks if isinstance(t, AreaObservationTask)]

    for target_id, plan in coverage_plans.items():
        target_area_tasks = [t for t in area_tasks if t.target_id == target_id]

        coverage_score = calculate_area_coverage_score(target_area_tasks, plan)

        # 根据覆盖率计算奖励/惩罚
        if coverage_score >= plan.min_coverage_ratio:
            # 达到最小覆盖率要求，给予奖励
            score += coverage_score * 100.0
        else:
            # 未达到要求，给予较小奖励但仍鼓励覆盖
            score += coverage_score * 50.0

    return score


def get_uncovered_tiles(
    coverage_plan: AreaCoveragePlan,
    scheduled_tasks: List[AreaObservationTask]
) -> List[MosaicTile]:
    """
    获取未覆盖的瓦片列表

    Args:
        coverage_plan: 覆盖计划
        scheduled_tasks: 已调度任务

    Returns:
        List[MosaicTile]: 未覆盖的瓦片列表
    """
    covered_tile_ids = set(t.tile_id for t in scheduled_tasks)
    return [t for t in coverage_plan.tiles if t.tile_id not in covered_tile_ids]


def update_coverage_contribution(
    area_tasks: List[AreaObservationTask],
    coverage_plan: AreaCoveragePlan,
    max_overlap_ratio: float = 0.15
) -> None:
    """
    更新任务的覆盖贡献（考虑重叠）

    当多个任务覆盖同一区域时，计算每个任务的有效贡献。

    Args:
        area_tasks: 区域任务列表
        coverage_plan: 覆盖计划
        max_overlap_ratio: 最大允许重叠比例
    """
    # 按优先级排序（高优先级先计算）
    sorted_tasks = sorted(area_tasks, key=lambda t: t.priority)

    covered_area_map: Dict[str, float] = {}  # tile_id -> 已被覆盖的面积

    for task in sorted_tasks:
        tile = task.tile

        # 计算与该瓦片相邻且已被覆盖的区域
        overlap_area = 0.0
        for other_tile_id, covered_area in covered_area_map.items():
            other_tile = coverage_plan.get_tile_by_id(other_tile_id)
            if other_tile:
                # 简化：假设相邻瓦片有固定比例重叠
                distance = tile.calculate_distance_to(
                    other_tile.center[0], other_tile.center[1]
                )
                tile_size = (tile.area_km2 ** 0.5)  # 估算边长
                if distance < tile_size * 1.5:  # 相邻
                    overlap_area += covered_area * 0.1  # 假设10%重叠

        # 计算有效覆盖
        max_allowed_overlap = tile.area_km2 * max_overlap_ratio
        actual_overlap = min(overlap_area, max_allowed_overlap)
        task.coverage_contribution = tile.area_km2 - actual_overlap

        # 记录该瓦片已被覆盖
        covered_area_map[tile.tile_id] = task.coverage_contribution


def merge_area_tasks_into_schedule(
    scheduled_area_tasks: List[AreaObservationTask],
    coverage_plan: AreaCoveragePlan,
    base_schedule: List[Any]
) -> List[Any]:
    """
    将区域任务合并到基础调度结果中

    Args:
        scheduled_area_tasks: 已调度的区域任务
        coverage_plan: 覆盖计划
        base_schedule: 基础调度结果（点目标等）

    Returns:
        List[Any]: 合并后的调度结果
    """
    merged = list(base_schedule)

    # 更新瓦片覆盖状态
    for task in scheduled_area_tasks:
        coverage_plan.register_tile_coverage(
            task.tile_id,
            task.task_id,
            effective_coverage=task.coverage_contribution
        )

    # 添加区域任务到合并结果
    merged.extend(scheduled_area_tasks)

    return merged
