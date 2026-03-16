"""
拼幅规划器 - 动态瓦片分解与覆盖规划

整合条带分解和网格分解，支持动态瓦片大小计算和可见性感知分解。

支持两种覆盖策略：
- 最大覆盖：尽可能覆盖更多面积
- 最高收益：在覆盖面积和任务收益间取得平衡
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from core.models import (
    Target, TargetType, Satellite, SatelliteType,
    MosaicTile, TileStatus, TilePriorityMode,
    AreaCoveragePlan, CoverageStrategy, OverlapHandling,
    ImagingMode
)
from core.coverage.footprint_calculator import FootprintCalculator
from .strip_decomposer import StripDecomposer
from .grid_decomposer import GridDecomposer

__all__ = ['MosaicPlanner']


class MosaicPlanner:
    """
    拼幅规划器

    将区域目标分解为优化的瓦片集合，支持：
    - 动态瓦片大小（基于卫星视场和轨道参数）
    - 可见性感知分解（只在可见窗口内生成瓦片）
    - 两种覆盖策略（最大覆盖、最高收益）
    - 10-20%重叠余量控制

    Attributes:
        strip_decomposer: 条带分解器（SAR卫星）
        grid_decomposer: 网格分解器（光学卫星）
        default_overlap_ratio: 默认重叠比例（10-20%）
    """

    # 每度纬度对应的公里数
    KM_PER_DEGREE_LAT = 111.0

    def __init__(self,
                 default_overlap_ratio: float = 0.15,
                 default_strategy: CoverageStrategy = CoverageStrategy.MAX_COVERAGE):
        """
        初始化拼幅规划器

        Args:
            default_overlap_ratio: 默认重叠比例（0.10-0.20），默认15%
            default_strategy: 默认覆盖策略
        """
        # 验证重叠比例在合理范围内
        if not 0.0 <= default_overlap_ratio <= 0.30:
            raise ValueError(f"Overlap ratio must be 0-0.30, got {default_overlap_ratio}")

        self.default_overlap_ratio = default_overlap_ratio
        self.default_strategy = default_strategy

        # 初始化分解器
        self.strip_decomposer = StripDecomposer()
        self.grid_decomposer = GridDecomposer()

        # 足迹计算器
        self.footprint_calculator = FootprintCalculator()

    def create_coverage_plan(self,
                            target: Target,
                            satellites: List[Satellite],
                            strategy: Optional[CoverageStrategy] = None,
                            overlap_ratio: Optional[float] = None,
                            priority_mode: Optional[TilePriorityMode] = None,
                            visibility_windows: Optional[Dict[str, List[Tuple[datetime, datetime]]]] = None,
                            **kwargs) -> AreaCoveragePlan:
        """
        创建区域覆盖规划

        Args:
            target: 区域目标
            satellites: 可用卫星列表
            strategy: 覆盖策略（默认使用target配置或全局默认）
            overlap_ratio: 重叠比例（默认使用target配置或全局默认）
            priority_mode: 瓦片优先级模式
            visibility_windows: 可选的可见性窗口缓存 {sat_id: [(start, end), ...]}
            **kwargs: 额外参数
                - min_coverage_ratio: 最小覆盖率（默认0.95）
                - dynamic_sizing: 是否启用动态瓦片大小（默认True）
                - footprint_scale: 瓦片大小相对于足迹的比例（默认0.9-1.1）

        Returns:
            AreaCoveragePlan: 区域覆盖规划
        """
        if target.target_type != TargetType.AREA:
            raise ValueError(f"Target must be AREA type, got {target.target_type}")

        # 确定策略
        strategy = strategy or self._get_strategy_from_target(target)
        overlap_ratio = overlap_ratio or target.max_overlap_ratio or self.default_overlap_ratio
        priority_mode = priority_mode or TilePriorityMode(target.tile_priority_mode or "center_first")

        # 确定分解策略
        decomp_strategy = self._determine_decomposition_strategy(target, satellites)

        # 计算动态瓦片大小
        tile_size_km = self._calculate_dynamic_tile_size(
            target, satellites, overlap_ratio, kwargs.get('dynamic_sizing', True)
        )

        # 根据策略选择分解方法
        if decomp_strategy == "strip":
            tiles = self._decompose_to_strips_dynamic(
                target, satellites, tile_size_km, overlap_ratio, visibility_windows, **kwargs
            )
        else:
            tiles = self._decompose_to_grid_dynamic(
                target, satellites, tile_size_km, overlap_ratio, visibility_windows, **kwargs
            )

        # 优化瓦片优先级
        tiles = self._optimize_tile_priorities(tiles, priority_mode, target)

        # 创建覆盖规划
        plan = AreaCoveragePlan(
            target_id=target.id,
            target_name=target.name,
            tiles=tiles,
            strategy=strategy,
            overlap_handling=self._get_overlap_handling(overlap_ratio),
            min_coverage_ratio=kwargs.get('min_coverage_ratio', target.min_coverage_ratio or 0.95),
            max_overlap_ratio=overlap_ratio,
            tile_priority_mode=priority_mode,
            dynamic_sizing_enabled=kwargs.get('dynamic_sizing', True),
            reference_footprint_size_km=tile_size_km,
        )

        return plan

    def _get_strategy_from_target(self, target: Target) -> CoverageStrategy:
        """从目标获取覆盖策略"""
        strategy_str = target.coverage_strategy or "max_coverage"
        try:
            return CoverageStrategy(strategy_str)
        except ValueError:
            return self.default_strategy

    def _get_overlap_handling(self, overlap_ratio: float) -> OverlapHandling:
        """根据重叠比例确定处理方式"""
        if overlap_ratio <= 0.05:
            return OverlapHandling.STRICT
        elif overlap_ratio <= 0.15:
            return OverlapHandling.MODERATE
        else:
            return OverlapHandling.RELAXED

    def _determine_decomposition_strategy(self, target: Target,
                                          satellites: List[Satellite]) -> str:
        """
        确定分解策略

        - "strip": 条带分解（SAR卫星）
        - "grid": 网格分解（光学卫星）

        策略选择逻辑：
        1. 如果target指定了mosaic_strategy，使用指定策略
        2. 如果主要是SAR卫星，使用条带
        3. 否则使用网格
        """
        # 检查目标指定的策略
        target_strategy = getattr(target, 'mosaic_strategy', 'auto')
        if target_strategy == "strip":
            return "strip"
        elif target_strategy == "grid":
            return "grid"

        # 自动判断：基于卫星类型
        sar_count = sum(1 for s in satellites
                       if s.sat_type == SatelliteType.SAR_1 or s.sat_type == SatelliteType.SAR_2)
        optical_count = len(satellites) - sar_count

        if sar_count > optical_count:
            return "strip"
        else:
            return "grid"

    def _calculate_dynamic_tile_size(self,
                                     target: Target,
                                     satellites: List[Satellite],
                                     overlap_ratio: float,
                                     dynamic_sizing: bool) -> float:
        """
        计算动态瓦片大小

        基于：
        - 卫星视场大小（FOV）
        - 轨道高度
        - 目标分辨率要求
        - 重叠比例

        Returns:
            瓦片边长（公里）
        """
        if not dynamic_sizing or not satellites:
            # 使用默认大小
            return 10.0  # 10km

        # 计算平均足迹大小
        footprint_sizes = []
        for sat in satellites:
            # 简化计算：使用卫星能力参数
            if sat.capabilities:
                # 使用最大滚转角估算最大足迹宽度
                max_roll = getattr(sat.capabilities, 'max_roll_angle', 35.0)
                # 简化估算：足迹宽度与滚转角正切成正比
                # 假设轨道高度500km
                altitude_km = 500.0
                footprint_width = 2 * altitude_km * math.tan(math.radians(max_roll))
                footprint_sizes.append(footprint_width)

        if not footprint_sizes:
            return 10.0

        # 使用平均足迹大小
        avg_footprint = sum(footprint_sizes) / len(footprint_sizes)

        # 根据覆盖策略调整
        strategy = self._get_strategy_from_target(target)
        if strategy == CoverageStrategy.MAX_COVERAGE:
            # 最大覆盖：稍小的瓦片以确保完整覆盖
            scale_factor = 0.90
        elif strategy == CoverageStrategy.MAX_PROFIT:
            # 最高收益：稍大的瓦片以减少任务数
            scale_factor = 1.05
        else:
            scale_factor = 0.95

        # 考虑重叠余量
        effective_size = avg_footprint * scale_factor * (1 - overlap_ratio * 0.5)

        # 限制范围：最小2km，最大50km
        return max(2.0, min(effective_size, 50.0))

    def _decompose_to_strips_dynamic(self,
                                     target: Target,
                                     satellites: List[Satellite],
                                     tile_size_km: float,
                                     overlap_ratio: float,
                                     visibility_windows: Optional[Dict] = None,
                                     **kwargs) -> List[MosaicTile]:
        """
        动态条带分解

        考虑卫星轨道方向，优化条带对齐
        """
        tiles = []

        # 获取边界框
        min_lon, min_lat, max_lon, max_lat = self._get_bounding_box(target.area_vertices)

        # 计算条带方向（沿主要轨道方向）
        strip_direction = self._calculate_optimal_strip_direction(satellites, target)

        # 条带参数
        strip_width_km = tile_size_km
        strip_spacing_km = strip_width_km * (1 - overlap_ratio)

        # 转换为度数
        avg_lat = (min_lat + max_lat) / 2
        km_per_deg_lon = self.KM_PER_DEGREE_LAT * math.cos(math.radians(avg_lat))
        strip_width_deg = strip_width_km / self.KM_PER_DEGREE_LAT
        spacing_deg = strip_spacing_km / km_per_deg_lon if abs(math.cos(math.radians(avg_lat))) > 0.01 else strip_spacing_km / self.KM_PER_DEGREE_LAT

        # 生成条带
        tile_id = 0
        if abs(math.sin(math.radians(strip_direction))) > abs(math.cos(math.radians(strip_direction))):
            # 沿纬度方向（东西向条带）
            lat = min_lat
            while lat < max_lat:
                tile_id += 1
                strip_tiles = self._create_strip_tile(
                    target, tile_id, lat, strip_width_deg,
                    min_lon, max_lon, True, visibility_windows
                )
                if strip_tiles:
                    tiles.extend(strip_tiles)
                lat += spacing_deg
        else:
            # 沿经度方向（南北向条带）
            lon = min_lon
            while lon < max_lon:
                tile_id += 1
                strip_tiles = self._create_strip_tile(
                    target, tile_id, lon, strip_width_deg,
                    min_lat, max_lat, False, visibility_windows
                )
                if strip_tiles:
                    tiles.extend(strip_tiles)
                lon += spacing_deg

        return tiles

    def _create_strip_tile(self,
                          target: Target,
                          strip_number: int,
                          start_coord: float,
                          width_deg: float,
                          min_other: float,
                          max_other: float,
                          is_latitude_strip: bool,
                          visibility_windows: Optional[Dict] = None) -> List[MosaicTile]:
        """创建单个条带瓦片"""
        # 计算条带顶点
        if is_latitude_strip:
            # 东西向条带
            vertices = [
                (min_other, start_coord),
                (max_other, start_coord),
                (max_other, start_coord + width_deg),
                (min_other, start_coord + width_deg),
            ]
            center = ((min_other + max_other) / 2, start_coord + width_deg / 2)
        else:
            # 南北向条带
            vertices = [
                (start_coord, min_other),
                (start_coord + width_deg, min_other),
                (start_coord + width_deg, max_other),
                (start_coord, max_other),
            ]
            center = (start_coord + width_deg / 2, (min_other + max_other) / 2)

        # 计算面积
        area_km2 = self._calculate_tile_area(vertices)

        tile = MosaicTile(
            tile_id=f"{target.id}-T{strip_number:04d}",
            parent_target_id=target.id,
            vertices=vertices,
            center=center,
            area_km2=area_km2,
            priority=target.priority,
            required_observations=target.required_observations,
        )

        return [tile]

    def _decompose_to_grid_dynamic(self,
                                   target: Target,
                                   satellites: List[Satellite],
                                   tile_size_km: float,
                                   overlap_ratio: float,
                                   visibility_windows: Optional[Dict] = None,
                                   **kwargs) -> List[MosaicTile]:
        """
        动态网格分解

        生成规则的网格瓦片
        """
        tiles = []

        # 获取边界框
        min_lon, min_lat, max_lon, max_lat = self._get_bounding_box(target.area_vertices)

        # 计算网格大小
        avg_lat = (min_lat + max_lat) / 2
        km_per_deg_lon = self.KM_PER_DEGREE_LAT * math.cos(math.radians(avg_lat))

        grid_size_km = tile_size_km
        spacing_km = grid_size_km * (1 - overlap_ratio)

        grid_size_deg_lat = grid_size_km / self.KM_PER_DEGREE_LAT
        grid_size_deg_lon = grid_size_km / km_per_deg_lon if km_per_deg_lon > 0.1 else grid_size_km / self.KM_PER_DEGREE_LAT
        spacing_deg_lat = spacing_km / self.KM_PER_DEGREE_LAT
        spacing_deg_lon = spacing_km / km_per_deg_lon if km_per_deg_lon > 0.1 else spacing_km / self.KM_PER_DEGREE_LAT

        # 生成网格
        tile_id = 0
        lat = min_lat
        row = 0

        while lat < max_lat:
            lon = min_lon
            col = 0

            while lon < max_lon:
                # 检查网格中心是否在多边形内
                center_lon = lon + grid_size_deg_lon / 2
                center_lat = lat + grid_size_deg_lat / 2

                if self._point_in_polygon(center_lon, center_lat, target.area_vertices):
                    tile_id += 1

                    # 计算网格顶点
                    vertices = [
                        (lon, lat),
                        (lon + grid_size_deg_lon, lat),
                        (lon + grid_size_deg_lon, lat + grid_size_deg_lat),
                        (lon, lat + grid_size_deg_lat),
                    ]

                    # 裁剪到多边形内（简化：直接使用网格）
                    tile = self._create_grid_tile(
                        target, tile_id, vertices, (center_lon, center_lat),
                        row, col
                    )
                    tiles.append(tile)

                lon += spacing_deg_lon
                col += 1

            lat += spacing_deg_lat
            row += 1

        return tiles

    def _create_grid_tile(self,
                         target: Target,
                         tile_number: int,
                         vertices: List[Tuple[float, float]],
                         center: Tuple[float, float],
                         row: int,
                         col: int) -> MosaicTile:
        """创建网格瓦片"""
        area_km2 = self._calculate_tile_area(vertices)

        tile = MosaicTile(
            tile_id=f"{target.id}-T{tile_number:04d}",
            parent_target_id=target.id,
            vertices=vertices,
            center=center,
            area_km2=area_km2,
            priority=target.priority,
            required_observations=target.required_observations,
        )

        # 存储行列信息
        tile.metadata['grid_row'] = row
        tile.metadata['grid_col'] = col

        return tile

    def _optimize_tile_priorities(self,
                                  tiles: List[MosaicTile],
                                  priority_mode: TilePriorityMode,
                                  target: Target) -> List[MosaicTile]:
        """
        优化瓦片优先级

        根据优先级模式调整瓦片优先级值
        """
        if priority_mode == TilePriorityMode.UNIFORM:
            # 统一优先级，不做调整
            return tiles

        # 计算区域中心
        if tiles:
            center_lon = sum(t.center[0] for t in tiles) / len(tiles)
            center_lat = sum(t.center[1] for t in tiles) / len(tiles)
        else:
            center_lon, center_lat = 0, 0

        # 计算区域边界
        min_lon = min(t.center[0] for t in tiles) if tiles else 0
        max_lon = max(t.center[0] for t in tiles) if tiles else 0
        min_lat = min(t.center[1] for t in tiles) if tiles else 0
        max_lat = max(t.center[1] for t in tiles) if tiles else 0

        max_distance = math.sqrt((max_lon - min_lon)**2 + (max_lat - min_lat)**2)
        if max_distance < 0.0001:
            max_distance = 1.0

        for tile in tiles:
            if priority_mode == TilePriorityMode.CENTER_FIRST:
                # 中心优先：距离中心越近，优先级越高（数值越小）
                dist = math.sqrt((tile.center[0] - center_lon)**2 +
                               (tile.center[1] - center_lat)**2)
                priority_adjustment = int((dist / max_distance) * 10)
                tile.priority = max(1, tile.priority + priority_adjustment)

            elif priority_mode == TilePriorityMode.EDGE_FIRST:
                # 边缘优先：距离中心越远，优先级越高
                dist = math.sqrt((tile.center[0] - center_lon)**2 +
                               (tile.center[1] - center_lat)**2)
                priority_adjustment = int((1 - dist / max_distance) * 10)
                tile.priority = max(1, tile.priority + priority_adjustment)

            elif priority_mode == TilePriorityMode.CORNER_FIRST:
                # 角落优先：距离最近的角落越近，优先级越高
                corners = [(min_lon, min_lat), (min_lon, max_lat),
                          (max_lon, min_lat), (max_lon, max_lat)]
                min_corner_dist = min(
                    math.sqrt((tile.center[0] - c[0])**2 + (tile.center[1] - c[1])**2)
                    for c in corners
                )
                priority_adjustment = int((min_corner_dist / max_distance) * 10)
                tile.priority = max(1, tile.priority + priority_adjustment)

        return tiles

    def _calculate_optimal_strip_direction(self,
                                          satellites: List[Satellite],
                                          target: Target) -> float:
        """
        计算最优条带方向

        基于卫星轨道倾角确定最佳条带方向
        """
        if not satellites:
            return 0.0  # 默认南北向

        # 获取平均轨道倾角
        inclinations = []
        for sat in satellites:
            if sat.orbit and sat.orbit.inclination:
                inclinations.append(sat.orbit.inclination)

        if not inclinations:
            return 0.0

        avg_inclination = sum(inclinations) / len(inclinations)

        # 条带方向垂直于轨道方向（简化）
        # 对于顺行轨道，条带方向约为90 - 倾角
        if avg_inclination < 90:
            return 90.0 - avg_inclination
        else:
            return avg_inclination - 90.0

    def _get_bounding_box(self, vertices: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """获取边界框 (min_lon, min_lat, max_lon, max_lat)"""
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]
        return (min(lons), min(lats), max(lons), max(lats))

    def _calculate_tile_area(self, vertices: List[Tuple[float, float]]) -> float:
        """计算多边形面积（平方公里）使用简化公式"""
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        # 转换为平方公里
        km_per_deg = self.KM_PER_DEGREE_LAT
        return abs(area) * 0.5 * km_per_deg * km_per_deg

    def _point_in_polygon(self, lon: float, lat: float,
                         vertices: List[Tuple[float, float]]) -> bool:
        """使用射线法判断点是否在多边形内"""
        n = len(vertices)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def create_multi_area_plan(self,
                               targets: List[Target],
                               satellites: List[Satellite],
                               **kwargs) -> Dict[str, AreaCoveragePlan]:
        """
        为多个区域目标创建覆盖规划

        Args:
            targets: 区域目标列表
            satellites: 可用卫星列表
            **kwargs: 传递给 create_coverage_plan 的参数

        Returns:
            Dict[str, AreaCoveragePlan]: {target_id: plan}
        """
        plans = {}
        for target in targets:
            if target.target_type == TargetType.AREA:
                plan = self.create_coverage_plan(target, satellites, **kwargs)
                plans[target.id] = plan
        return plans

    def estimate_coverage_cost(self,
                              plan: AreaCoveragePlan,
                              satellite_cost_per_task: float = 1.0) -> Dict[str, float]:
        """
        估算覆盖成本

        Returns:
            Dict with cost metrics
        """
        total_tiles = len(plan.tiles)
        required_coverage_tiles = int(total_tiles * plan.min_coverage_ratio)

        # 基础成本（按瓦片数）
        base_cost = required_coverage_tiles * satellite_cost_per_task

        # 根据策略调整
        if plan.strategy == CoverageStrategy.MAX_COVERAGE:
            # 最大覆盖可能需要更多任务来确保覆盖率
            estimated_cost = base_cost * 1.1
        elif plan.strategy == CoverageStrategy.MAX_PROFIT:
            # 最高收益可能使用更少但更高价值的任务
            estimated_cost = base_cost * 0.9
        else:
            estimated_cost = base_cost

        return {
            'estimated_tasks': required_coverage_tiles,
            'estimated_cost': estimated_cost,
            'total_tiles': total_tiles,
            'coverage_ratio_required': plan.min_coverage_ratio,
        }
