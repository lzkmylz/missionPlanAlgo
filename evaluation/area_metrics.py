"""
区域目标评估指标模块

提供区域目标特有的评估指标计算
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

from core.models import Mission, Target
from core.models.mosaic_tile import MosaicTile
from scheduler.base_scheduler import ScheduleResult, ScheduledTask

logger = logging.getLogger(__name__)


@dataclass
class AreaCoverageMetrics:
    """区域覆盖指标"""

    # 基础统计
    total_tiles: int = 0
    covered_tiles: int = 0
    pending_tiles: int = 0

    # 覆盖率
    coverage_ratio: float = 0.0
    min_required_coverage: float = 0.95

    # 区域面积覆盖
    total_area_km2: float = 0.0
    covered_area_km2: float = 0.0
    area_coverage_ratio: float = 0.0

    # 重叠区域统计
    total_overlap_area_km2: float = 0.0
    avg_overlap_ratio: float = 0.0
    max_overlap_ratio: float = 0.0
    overlapping_tile_pairs: int = 0

    # 卫星覆盖贡献度
    area_per_satellite_km2: Dict[str, float] = field(default_factory=dict)
    coverage_contribution_ratio: Dict[str, float] = field(default_factory=dict)

    # 完成时间
    first_task_time: Optional[datetime] = None
    last_task_time: Optional[datetime] = None
    coverage_makespan_hours: float = 0.0

    # 卫星分布
    tiles_per_satellite: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_tiles': self.total_tiles,
            'covered_tiles': self.covered_tiles,
            'pending_tiles': self.pending_tiles,
            'coverage_ratio': self.coverage_ratio,
            'min_required_coverage': self.min_required_coverage,
            'total_area_km2': self.total_area_km2,
            'covered_area_km2': self.covered_area_km2,
            'area_coverage_ratio': self.area_coverage_ratio,
            'total_overlap_area_km2': self.total_overlap_area_km2,
            'avg_overlap_ratio': self.avg_overlap_ratio,
            'max_overlap_ratio': self.max_overlap_ratio,
            'overlapping_tile_pairs': self.overlapping_tile_pairs,
            'area_per_satellite_km2': self.area_per_satellite_km2,
            'coverage_contribution_ratio': self.coverage_contribution_ratio,
            'first_task_time': self.first_task_time.isoformat() if self.first_task_time else None,
            'last_task_time': self.last_task_time.isoformat() if self.last_task_time else None,
            'coverage_makespan_hours': self.coverage_makespan_hours,
            'tiles_per_satellite': self.tiles_per_satellite,
        }


@dataclass
class MosaicEfficiencyMetrics:
    """拼幅效率指标"""

    # 重叠统计
    avg_overlap_ratio: float = 0.0
    max_overlap_ratio: float = 0.0
    total_overlap_area_km2: float = 0.0

    # 卫星切换
    satellite_switches: int = 0
    avg_tasks_per_satellite: float = 0.0

    # 时间效率
    avg_time_between_tasks_min: float = 0.0
    task_gaps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'avg_overlap_ratio': self.avg_overlap_ratio,
            'max_overlap_ratio': self.max_overlap_ratio,
            'total_overlap_area_km2': self.total_overlap_area_km2,
            'satellite_switches': self.satellite_switches,
            'avg_tasks_per_satellite': self.avg_tasks_per_satellite,
            'avg_time_between_tasks_min': self.avg_time_between_tasks_min,
        }


@dataclass
class ResourceUtilizationMetrics:
    """资源利用指标"""

    # 存储使用
    avg_storage_used_gb: float = 0.0
    max_storage_used_gb: float = 0.0
    storage_efficiency: float = 0.0

    # 电量使用
    avg_power_consumption_w: float = 0.0
    power_efficiency: float = 0.0

    # 姿态机动
    avg_slew_angle_deg: float = 0.0
    max_slew_angle_deg: float = 0.0
    total_slew_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'avg_storage_used_gb': self.avg_storage_used_gb,
            'max_storage_used_gb': self.max_storage_used_gb,
            'storage_efficiency': self.storage_efficiency,
            'avg_power_consumption_w': self.avg_power_consumption_w,
            'power_efficiency': self.power_efficiency,
            'avg_slew_angle_deg': self.avg_slew_angle_deg,
            'max_slew_angle_deg': self.max_slew_angle_deg,
            'total_slew_time_sec': self.total_slew_time_sec,
        }


class AreaMetricsCalculator:
    """区域目标指标计算器"""

    def __init__(
        self,
        mission: Mission,
        tiles: List[MosaicTile],
        area_target: Optional[Target] = None
    ):
        """
        初始化计算器

        Args:
            mission: Mission对象
            tiles: tiles列表
            area_target: 区域目标（可选）
        """
        self.mission = mission
        self.tiles = tiles
        self.area_target = area_target

    def calculate_all(
        self,
        schedule_result: ScheduleResult,
        tile_targets: Optional[List[Target]] = None
    ) -> Dict[str, Any]:
        """
        计算所有区域目标指标

        Args:
            schedule_result: 调度结果
            tile_targets: tile目标列表（用于映射）

        Returns:
            包含所有指标的字典
        """
        coverage_metrics = self.calculate_coverage_metrics(
            schedule_result, tile_targets
        )

        efficiency_metrics = self.calculate_efficiency_metrics(
            schedule_result
        )

        resource_metrics = self.calculate_resource_metrics(
            schedule_result
        )

        return {
            'coverage': coverage_metrics.to_dict(),
            'efficiency': efficiency_metrics.to_dict(),
            'resource': resource_metrics.to_dict(),
            'summary': {
                'total_tiles': len(self.tiles),
                'scheduled_tasks': len(schedule_result.scheduled_tasks),
                'unscheduled_tasks': len(schedule_result.unscheduled_tasks),
                'success_rate': coverage_metrics.coverage_ratio,
            }
        }

    def calculate_coverage_metrics(
        self,
        schedule_result: ScheduleResult,
        tile_targets: Optional[List[Target]] = None
    ) -> AreaCoverageMetrics:
        """计算覆盖指标，包括面积覆盖、重叠统计和卫星贡献度"""
        metrics = AreaCoverageMetrics()
        metrics.total_tiles = len(self.tiles)

        # 计算总面积
        total_area = sum(tile.area_km2 for tile in self.tiles)
        metrics.total_area_km2 = total_area

        if not schedule_result.scheduled_tasks:
            return metrics

        # 构建tile ID到目标的映射
        tile_target_ids = set()
        if tile_targets:
            tile_target_ids = {t.id for t in tile_targets}

        # 构建tile ID到tile对象的映射
        tile_map = {tile.tile_id: tile for tile in self.tiles}

        # 统计已覆盖的tiles和卫星贡献
        covered_tile_ids = set()
        task_times = []
        tiles_per_sat = defaultdict(int)
        area_per_sat = defaultdict(float)  # 每颗卫星覆盖的面积
        tile_to_sat = {}  # 记录每个tile由哪颗卫星覆盖

        for task in schedule_result.scheduled_tasks:
            # 检查是否是tile任务 - 支持AreaObservationTask的tile_id
            tile_id = getattr(task, 'tile_id', None)
            target_id = getattr(task, 'target_id', None)
            sat_id = getattr(task, 'satellite_id', None)

            # 优先使用tile_id匹配，其次使用target_id匹配
            matched_id = None
            if tile_id and (tile_id in tile_target_ids or tile_id in tile_map):
                matched_id = tile_id
            elif target_id and target_id in tile_target_ids:
                matched_id = target_id
            elif target_id:
                # 尝试去掉-OBS后缀匹配
                base_id = target_id.rsplit('-OBS', 1)[0]
                if base_id in tile_target_ids or base_id in tile_map:
                    matched_id = base_id

            if matched_id:
                covered_tile_ids.add(matched_id)

                # 记录该tile由哪颗卫星覆盖
                if sat_id:
                    tile_to_sat[matched_id] = sat_id
                    tiles_per_sat[sat_id] += 1

                    # 累加该卫星覆盖的面积
                    if matched_id in tile_map:
                        area_per_sat[sat_id] += tile_map[matched_id].area_km2

                if hasattr(task, 'start_time') and task.start_time:
                    task_times.append(task.start_time)

        metrics.covered_tiles = len(covered_tile_ids)
        metrics.pending_tiles = metrics.total_tiles - metrics.covered_tiles
        metrics.coverage_ratio = (
            metrics.covered_tiles / metrics.total_tiles
            if metrics.total_tiles > 0 else 0.0
        )

        # 计算面积覆盖
        covered_area = sum(tile_map[tid].area_km2 for tid in covered_tile_ids if tid in tile_map)
        metrics.covered_area_km2 = covered_area
        metrics.area_coverage_ratio = (
            covered_area / total_area if total_area > 0 else 0.0
        )

        # 计算重叠区域统计
        self._calculate_overlap_metrics(metrics, covered_tile_ids, tile_map)

        # 计算卫星覆盖贡献度
        metrics.tiles_per_satellite = dict(tiles_per_sat)
        metrics.area_per_satellite_km2 = dict(area_per_sat)

        # 计算每颗卫星的覆盖贡献比例
        if covered_area > 0:
            metrics.coverage_contribution_ratio = {
                sat_id: area / covered_area
                for sat_id, area in area_per_sat.items()
            }

        # 时间统计
        if task_times:
            metrics.first_task_time = min(task_times)
            metrics.last_task_time = max(task_times)
            duration = (metrics.last_task_time - metrics.first_task_time).total_seconds()
            metrics.coverage_makespan_hours = duration / 3600

        return metrics

    def _calculate_overlap_metrics(
        self,
        metrics: AreaCoverageMetrics,
        covered_tile_ids: set,
        tile_map: Dict[str, MosaicTile]
    ):
        """计算重叠区域统计"""
        # 获取所有已覆盖的tile对象
        covered_tiles = [tile_map[tid] for tid in covered_tile_ids if tid in tile_map]

        if len(covered_tiles) < 2:
            return

        # 计算两两之间的重叠
        total_overlap_area = 0.0
        max_overlap = 0.0
        overlap_pairs = 0
        overlap_ratios = []

        for i, tile1 in enumerate(covered_tiles):
            for tile2 in covered_tiles[i+1:]:
                overlap_area = self._calculate_tile_overlap(tile1, tile2)
                if overlap_area > 0:
                    overlap_pairs += 1
                    total_overlap_area += overlap_area

                    # 计算相对于较小tile的重叠比例
                    min_area = min(tile1.area_km2, tile2.area_km2)
                    if min_area > 0:
                        overlap_ratio = overlap_area / min_area
                        overlap_ratios.append(overlap_ratio)
                        max_overlap = max(max_overlap, overlap_ratio)

        metrics.total_overlap_area_km2 = total_overlap_area
        metrics.overlapping_tile_pairs = overlap_pairs

        if overlap_ratios:
            metrics.avg_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)
            metrics.max_overlap_ratio = max_overlap

    def _calculate_tile_overlap(self, tile1: MosaicTile, tile2: MosaicTile) -> float:
        """
        计算两个tile之间的重叠面积（简化计算）

        注意：此为简化估算，假设：
        1. 瓦片近似正方形
        2. 基于中心距估算重叠面积
        3. 未使用精确多边形相交计算

        对于生产环境需要精确计算，建议使用shapely等几何库，
        或参考本模块提供的精确重叠计算方案 PreciseOverlapCalculator。
        """
        # 获取tile中心点
        lon1, lat1 = tile1.center
        lon2, lat2 = tile2.center

        # 估算tile边长（假设为正方形，area = side^2）
        side1 = (tile1.area_km2 ** 0.5) if tile1.area_km2 > 0 else 0
        side2 = (tile2.area_km2 ** 0.5) if tile2.area_km2 > 0 else 0

        # 防止除零：如果任一瓦片面积为0，返回0
        if side1 == 0 or side2 == 0:
            return 0.0

        # 计算中心点距离（简化的大圆距离）
        # 1度纬度约等于111km
        lat_diff = abs(lat1 - lat2) * 111.0
        # 1度经度随纬度变化
        avg_lat_rad = math.radians((lat1 + lat2) / 2)
        lon_diff = abs(lon1 - lon2) * 111.0 * math.cos(avg_lat_rad)
        distance = (lat_diff ** 2 + lon_diff ** 2) ** 0.5

        # 如果距离大于两tile半径之和，则无重叠
        radius1 = side1 / 2
        radius2 = side2 / 2

        # 防止除零：半径和为0时返回0
        radius_sum = radius1 + radius2
        if radius_sum == 0:
            return 0.0

        if distance >= radius_sum:
            return 0.0

        # 如果一tile完全包含另一tile
        if distance <= abs(radius1 - radius2):
            return min(tile1.area_km2, tile2.area_km2)

        # 简化：使用近似重叠面积计算（两个正方形的重叠）
        overlap_ratio = (radius_sum - distance) / radius_sum
        overlap_area = min(tile1.area_km2, tile2.area_km2) * overlap_ratio * 0.5

        return overlap_area

    def calculate_efficiency_metrics(
        self,
        schedule_result: ScheduleResult
    ) -> MosaicEfficiencyMetrics:
        """计算拼幅效率指标"""
        metrics = MosaicEfficiencyMetrics()

        if not schedule_result.scheduled_tasks:
            return metrics

        tasks = schedule_result.scheduled_tasks

        # 按时间排序
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.start_time if hasattr(t, 'start_time') and t.start_time else datetime.min
        )

        # 统计卫星切换
        prev_sat = None
        sat_task_count = defaultdict(int)

        for task in sorted_tasks:
            sat_id = getattr(task, 'satellite_id', None)
            if sat_id:
                sat_task_count[sat_id] += 1
                if prev_sat and sat_id != prev_sat:
                    metrics.satellite_switches += 1
                prev_sat = sat_id

        # 平均每卫星任务数
        if sat_task_count:
            metrics.avg_tasks_per_satellite = sum(sat_task_count.values()) / len(sat_task_count)

        # 任务间隔时间
        time_gaps = []
        for i in range(1, len(sorted_tasks)):
            prev_end = getattr(sorted_tasks[i-1], 'end_time', None)
            curr_start = getattr(sorted_tasks[i], 'start_time', None)
            if prev_end and curr_start:
                gap = (curr_start - prev_end).total_seconds() / 60  # 分钟
                time_gaps.append(gap)

        if time_gaps:
            metrics.avg_time_between_tasks_min = sum(time_gaps) / len(time_gaps)
            metrics.task_gaps = time_gaps

        return metrics

    def calculate_resource_metrics(
        self,
        schedule_result: ScheduleResult
    ) -> ResourceUtilizationMetrics:
        """计算资源利用指标"""
        metrics = ResourceUtilizationMetrics()

        if not schedule_result.scheduled_tasks:
            return metrics

        tasks = schedule_result.scheduled_tasks

        # 存储使用统计
        storage_values = []
        power_values = []
        slew_angles = []

        for task in tasks:
            # 存储
            if hasattr(task, 'storage_used_gb') and task.storage_used_gb is not None:
                storage_values.append(task.storage_used_gb)

            # 电量
            if hasattr(task, 'power_consumed_wh') and task.power_consumed_wh is not None:
                power_values.append(task.power_consumed_wh)

            # 姿态角（作为机动大小的近似）
            if hasattr(task, 'roll_angle') and task.roll_angle is not None:
                slew_angles.append(abs(task.roll_angle))
            if hasattr(task, 'pitch_angle') and task.pitch_angle is not None:
                slew_angles.append(abs(task.pitch_angle))

        # 计算统计值
        if storage_values:
            metrics.avg_storage_used_gb = sum(storage_values) / len(storage_values)
            metrics.max_storage_used_gb = max(storage_values)

        if power_values:
            metrics.avg_power_consumption_w = sum(power_values) / len(power_values)

        if slew_angles:
            metrics.avg_slew_angle_deg = sum(slew_angles) / len(slew_angles)
            metrics.max_slew_angle_deg = max(slew_angles)

        return metrics


def generate_area_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """
    生成区域目标算法对比报告

    Args:
        results: {algorithm_name: metrics_dict}
        output_path: 输出文件路径
    """
    import json

    report = {
        'timestamp': datetime.now().isoformat(),
        'scenario_type': 'area_target',
        'algorithms': list(results.keys()),
        'comparison': {}
    }

    # 提取关键指标进行对比
    coverage_comparison = {}
    efficiency_comparison = {}

    for algo_name, metrics in results.items():
        summary = metrics.get('summary', {})
        coverage = metrics.get('coverage', {})
        efficiency = metrics.get('efficiency', {})

        coverage_comparison[algo_name] = {
            'success_rate': coverage.get('coverage_ratio', 0.0),
            'covered_tiles': coverage.get('covered_tiles', 0),
            'total_tiles': coverage.get('total_tiles', 0),
            'makespan_hours': coverage.get('coverage_makespan_hours', 0.0),
            # 新增面积覆盖指标
            'total_area_km2': coverage.get('total_area_km2', 0.0),
            'covered_area_km2': coverage.get('covered_area_km2', 0.0),
            'area_coverage_ratio': coverage.get('area_coverage_ratio', 0.0),
            # 新增重叠统计指标
            'total_overlap_area_km2': coverage.get('total_overlap_area_km2', 0.0),
            'avg_overlap_ratio': coverage.get('avg_overlap_ratio', 0.0),
            'max_overlap_ratio': coverage.get('max_overlap_ratio', 0.0),
            'overlapping_tile_pairs': coverage.get('overlapping_tile_pairs', 0),
            # 卫星使用数量
            'num_satellites_used': len(coverage.get('tiles_per_satellite', {})),
        }

        efficiency_comparison[algo_name] = {
            'satellite_switches': efficiency.get('satellite_switches', 0),
            'avg_tasks_per_satellite': efficiency.get('avg_tasks_per_satellite', 0.0),
            'avg_time_between_tasks_min': efficiency.get('avg_time_between_tasks_min', 0.0),
        }

    report['comparison']['coverage'] = coverage_comparison
    report['comparison']['efficiency'] = efficiency_comparison

    # 找出最佳算法
    best_coverage = max(
        coverage_comparison.items(),
        key=lambda x: x[1]['success_rate']
    )

    report['best_algorithm'] = {
        'coverage': best_coverage[0],
        'coverage_ratio': best_coverage[1]['success_rate']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"对比报告已保存: {output_path}")


class PreciseOverlapCalculator:
    """
    精确重叠面积计算器

    使用瓦片的实际多边形边界计算精确重叠面积。
    支持两种计算模式：精确多边形相交（需shapely）和网格采样法（纯Python）。

    性能优化：
    1. 空间索引（R-tree）快速筛选潜在重叠对
    2. 缓存机制避免重复计算
    3. 并行计算支持（大数量瓦片）
    """

    def __init__(self, use_shapely: bool = True, use_spatial_index: bool = True):
        """
        初始化精确重叠计算器

        Args:
            use_shapely: 是否使用shapely进行精确多边形计算（如不可用则使用网格采样）
            use_spatial_index: 是否使用空间索引加速
        """
        self.use_shapely = use_shapely and self._check_shapely_available()
        self.use_spatial_index = use_spatial_index
        self._cache: Dict[Tuple[str, str], float] = {}

    def _check_shapely_available(self) -> bool:
        """检查shapely是否可用"""
        try:
            import shapely.geometry as geom
            import shapely.ops as ops
            self._geom = geom
            self._ops = ops
            return True
        except ImportError:
            logger.warning("shapely未安装，将使用网格采样法计算重叠")
            return False

    def calculate_overlap(
        self,
        tile1: MosaicTile,
        tile2: MosaicTile,
        grid_resolution: int = 20
    ) -> float:
        """
        计算两个瓦片的精确重叠面积

        Args:
            tile1: 第一个瓦片
            tile2: 第二个瓦片
            grid_resolution: 网格采样分辨率（网格采样模式使用）

        Returns:
            重叠面积（km²）
        """
        # 检查缓存
        cache_key = self._get_cache_key(tile1.tile_id, tile2.tile_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 快速边界框检查
        if not self._bbox_intersect(tile1, tile2):
            self._cache[cache_key] = 0.0
            return 0.0

        # 使用shapely精确计算或网格采样
        if self.use_shapely and hasattr(tile1, 'bounds') and hasattr(tile2, 'bounds'):
            overlap = self._calculate_shapely_overlap(tile1, tile2)
        else:
            overlap = self._calculate_grid_overlap(tile1, tile2, grid_resolution)

        self._cache[cache_key] = overlap
        return overlap

    def calculate_all_overlaps(
        self,
        tiles: List[MosaicTile],
        max_distance_ratio: float = 1.5,
        parallel_threshold: int = 500
    ) -> Dict[Tuple[str, str], float]:
        """
        批量计算所有瓦片对的重叠面积（带空间索引优化）

        Args:
            tiles: 瓦片列表
            max_distance_ratio: 最大距离倍数（超过此倍数的瓦片对不检查）
            parallel_threshold: 启用并行计算的阈值瓦片数

        Returns:
            {(tile_id1, tile_id2): overlap_area}
        """
        if len(tiles) < 2:
            return {}

        results = {}

        # 使用空间索引筛选潜在重叠对
        if self.use_spatial_index and len(tiles) > 50:
            candidate_pairs = self._get_candidate_pairs_spatial_index(
                tiles, max_distance_ratio
            )
        else:
            candidate_pairs = self._get_candidate_pairs_brute_force(
                tiles, max_distance_ratio
            )

        # 计算每个候选对的重叠
        for tile1, tile2 in candidate_pairs:
            overlap = self.calculate_overlap(tile1, tile2)
            if overlap > 0:
                results[(tile1.tile_id, tile2.tile_id)] = overlap

        return results

    def _get_cache_key(self, id1: str, id2: str) -> Tuple[str, str]:
        """生成缓存键（规范化顺序）"""
        return (min(id1, id2), max(id1, id2))

    def _bbox_intersect(self, tile1: MosaicTile, tile2: MosaicTile) -> bool:
        """快速边界框相交检查"""
        # 获取瓦片边界框
        if hasattr(tile1, 'bounds') and tile1.bounds:
            min_lon1, min_lat1, max_lon1, max_lat1 = tile1.bounds
        else:
            # 从中心点和面积估算
            side = (tile1.area_km2 ** 0.5) / 111.0  # 粗略转换为度
            min_lon1 = tile1.center[0] - side / 2
            max_lon1 = tile1.center[0] + side / 2
            min_lat1 = tile1.center[1] - side / 2
            max_lat1 = tile1.center[1] + side / 2

        if hasattr(tile2, 'bounds') and tile2.bounds:
            min_lon2, min_lat2, max_lon2, max_lat2 = tile2.bounds
        else:
            side = (tile2.area_km2 ** 0.5) / 111.0
            min_lon2 = tile2.center[0] - side / 2
            max_lon2 = tile2.center[0] + side / 2
            min_lat2 = tile2.center[1] - side / 2
            max_lat2 = tile2.center[1] + side / 2

        return not (
            max_lon1 < min_lon2 or min_lon1 > max_lon2 or
            max_lat1 < min_lat2 or min_lat1 > max_lat2
        )

    def _calculate_shapely_overlap(
        self,
        tile1: MosaicTile,
        tile2: MosaicTile
    ) -> float:
        """使用shapely计算精确多边形重叠"""
        try:
            # 获取多边形顶点
            poly1 = self._get_polygon(tile1)
            poly2 = self._get_polygon(tile2)

            if poly1 is None or poly2 is None:
                return 0.0

            # 计算相交
            intersection = poly1.intersection(poly2)

            if intersection.is_empty:
                return 0.0

            # 计算面积（shapely使用平面坐标，需要转换）
            # 使用等积投影近似
            area_deg = intersection.area
            center_lat = (tile1.center[1] + tile2.center[1]) / 2
            area_km2 = area_deg * (111.32 ** 2) * math.cos(math.radians(center_lat))

            return max(0.0, area_km2)

        except Exception as e:
            logger.debug(f"shapely重叠计算失败: {e}")
            return 0.0

    def _get_polygon(self, tile: MosaicTile):
        """从瓦片获取shapely多边形"""
        if hasattr(tile, 'vertices') and tile.vertices:
            # 使用实际顶点
            coords = [(v[0], v[1]) for v in tile.vertices]
            if len(coords) >= 3:
                return self._geom.Polygon(coords)

        if hasattr(tile, 'bounds') and tile.bounds:
            # 使用边界框
            min_lon, min_lat, max_lon, max_lat = tile.bounds
            return self._geom.Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat)
            ])

        return None

    def _calculate_grid_overlap(
        self,
        tile1: MosaicTile,
        tile2: MosaicTile,
        resolution: int = 20
    ) -> float:
        """
        使用网格采样法计算重叠面积

        在瓦片边界框上创建网格，统计同时落在两个瓦片内的采样点比例。
        精度取决于分辨率，适合没有shapely的环境。
        """
        # 获取瓦片1的边界框
        if hasattr(tile1, 'bounds') and tile1.bounds:
            min_lon1, min_lat1, max_lon1, max_lat1 = tile1.bounds
        else:
            side = (tile1.area_km2 ** 0.5) / 111.0
            min_lon1 = tile1.center[0] - side / 2
            max_lon1 = tile1.center[0] + side / 2
            min_lat1 = tile1.center[1] - side / 2
            max_lat1 = tile1.center[1] + side / 2

        if hasattr(tile2, 'bounds') and tile2.bounds:
            min_lon2, min_lat2, max_lon2, max_lat2 = tile2.bounds
        else:
            side = (tile2.area_km2 ** 0.5) / 111.0
            min_lon2 = tile2.center[0] - side / 2
            max_lon2 = tile2.center[0] + side / 2
            min_lat2 = tile2.center[1] - side / 2
            max_lat2 = tile2.center[1] + side / 2

        # 计算交集边界框
        min_lon = max(min_lon1, min_lon2)
        max_lon = min(max_lon1, max_lon2)
        min_lat = max(min_lat1, min_lat2)
        max_lat = min(max_lat1, max_lat2)

        if min_lon >= max_lon or min_lat >= max_lat:
            return 0.0

        # 创建网格
        lon_step = (max_lon - min_lon) / resolution
        lat_step = (max_lat - min_lat) / resolution

        # 统计采样点
        overlap_points = 0
        total_points = resolution * resolution

        for i in range(resolution):
            for j in range(resolution):
                lon = min_lon + (i + 0.5) * lon_step
                lat = min_lat + (j + 0.5) * lat_step

                if self._point_in_tile(lon, lat, tile1) and self._point_in_tile(lon, lat, tile2):
                    overlap_points += 1

        # 估算重叠面积
        overlap_ratio = overlap_points / total_points if total_points > 0 else 0
        bbox_area_km2 = (max_lon - min_lon) * 111.32 * math.cos(math.radians((min_lat + max_lat) / 2)) * \
                       (max_lat - min_lat) * 111.32

        return bbox_area_km2 * overlap_ratio

    def _point_in_tile(self, lon: float, lat: float, tile: MosaicTile) -> bool:
        """检查点是否在瓦片内"""
        if hasattr(tile, 'vertices') and tile.vertices:
            # 使用多边形包含检查
            return self._point_in_polygon(lon, lat, tile.vertices)

        # 使用边界框近似
        if hasattr(tile, 'bounds') and tile.bounds:
            min_lon, min_lat, max_lon, max_lat = tile.bounds
            return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

        # 从中心点和面积估算
        side = (tile.area_km2 ** 0.5) / 111.0
        min_lon = tile.center[0] - side / 2
        max_lon = tile.center[0] + side / 2
        min_lat = tile.center[1] - side / 2
        max_lat = tile.center[1] + side / 2
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

    def _point_in_polygon(self, lon: float, lat: float, vertices: List[List[float]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(vertices)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _get_candidate_pairs_spatial_index(
        self,
        tiles: List[MosaicTile],
        max_distance_ratio: float
    ) -> List[Tuple[MosaicTile, MosaicTile]]:
        """
        使用简单网格空间索引获取候选重叠对

        实现简化版均匀网格索引，避免依赖rtree库。
        """
        candidates = []

        # 计算全局边界框
        all_lons = []
        all_lats = []
        for tile in tiles:
            if hasattr(tile, 'bounds') and tile.bounds:
                all_lons.extend([tile.bounds[0], tile.bounds[2]])
                all_lats.extend([tile.bounds[1], tile.bounds[3]])
            else:
                side = (tile.area_km2 ** 0.5) / 111.0
                all_lons.extend([tile.center[0] - side, tile.center[0] + side])
                all_lats.extend([tile.center[1] - side, tile.center[1] + side])

        if not all_lons or not all_lats:
            return candidates

        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)

        # 创建网格（根据瓦片数量调整网格大小）
        grid_size = min(int(len(tiles) ** 0.5) + 1, 20)
        lon_step = (max_lon - min_lon) / grid_size if max_lon > min_lon else 1
        lat_step = (max_lat - min_lat) / grid_size if max_lat > min_lat else 1

        # 将瓦片放入网格
        grid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        tile_cells: Dict[int, List[Tuple[int, int]]] = {}

        for idx, tile in enumerate(tiles):
            if hasattr(tile, 'bounds') and tile.bounds:
                min_l, min_la, max_l, max_la = tile.bounds
            else:
                side = (tile.area_km2 ** 0.5) / 111.0
                min_l = tile.center[0] - side
                max_l = tile.center[0] + side
                min_la = tile.center[1] - side
                max_la = tile.center[1] + side

            # 计算瓦片覆盖的网格单元
            cell_min_x = int((min_l - min_lon) / lon_step)
            cell_max_x = int((max_l - min_lon) / lon_step)
            cell_min_y = int((min_la - min_lat) / lat_step)
            cell_max_y = int((max_la - min_lat) / lat_step)

            cells = []
            for cx in range(max(0, cell_min_x), min(grid_size, cell_max_x + 1)):
                for cy in range(max(0, cell_min_y), min(grid_size, cell_max_y + 1)):
                    grid[(cx, cy)].add(idx)
                    cells.append((cx, cy))
            tile_cells[idx] = cells

        # 查找候选对（在同一网格单元内的瓦片）
        seen_pairs = set()
        for cell_indices in grid.values():
            indices_list = sorted(cell_indices)
            for i, idx1 in enumerate(indices_list):
                for idx2 in indices_list[i + 1:]:
                    pair_key = (min(idx1, idx2), max(idx1, idx2))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        # 距离预筛选
                        tile1, tile2 = tiles[idx1], tiles[idx2]
                        if self._within_distance(tile1, tile2, max_distance_ratio):
                            candidates.append((tile1, tile2))

        return candidates

    def _get_candidate_pairs_brute_force(
        self,
        tiles: List[MosaicTile],
        max_distance_ratio: float
    ) -> List[Tuple[MosaicTile, MosaicTile]]:
        """暴力法获取候选对（小数量瓦片时使用）"""
        candidates = []
        n = len(tiles)

        for i in range(n):
            for j in range(i + 1, n):
                if self._within_distance(tiles[i], tiles[j], max_distance_ratio):
                    candidates.append((tiles[i], tiles[j]))

        return candidates

    def _within_distance(
        self,
        tile1: MosaicTile,
        tile2: MosaicTile,
        max_ratio: float
    ) -> bool:
        """检查两瓦片中心距离是否在允许范围内"""
        # 计算中心点距离
        lat_diff = abs(tile1.center[1] - tile2.center[1]) * 111.0
        avg_lat_rad = math.radians((tile1.center[1] + tile2.center[1]) / 2)
        lon_diff = abs(tile1.center[0] - tile2.center[0]) * 111.0 * math.cos(avg_lat_rad)
        distance = (lat_diff ** 2 + lon_diff ** 2) ** 0.5

        # 估算瓦片半径和
        radius1 = (tile1.area_km2 ** 0.5) / 2 if tile1.area_km2 > 0 else 0
        radius2 = (tile2.area_km2 ** 0.5) / 2 if tile2.area_km2 > 0 else 0
        max_distance = (radius1 + radius2) * max_ratio

        return distance <= max_distance

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


# 导出类
__all__ = [
    'AreaCoverageMetrics',
    'MosaicEfficiencyMetrics',
    'ResourceUtilizationMetrics',
    'AreaMetricsCalculator',
    'PreciseOverlapCalculator',
    'generate_area_comparison_report',
]