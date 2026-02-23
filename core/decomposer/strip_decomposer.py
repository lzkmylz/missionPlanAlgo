"""
条带分解器

将区域目标分解为平行条带（适用于SAR卫星）
"""

import math
from typing import List, Tuple

from .base_decomposer import BaseDecomposer, DecompositionStrategy
from core.models import Target, TargetType, ImagingMode


class StripDecomposer(BaseDecomposer):
    """
    条带分解器

    将多边形区域沿最优方向切割为平行条带。
    适用于SAR卫星的条带模式或聚束模式。

    Attributes:
        swath_width: 幅宽（米）
        overlap_ratio: 条带重叠率（0-1）
    """

    # 每度纬度/经度对应的公里数
    KM_PER_DEGREE_LAT = 111.0
    KM_PER_DEGREE_LON = 111.0

    def __init__(self, swath_width: float = 10000.0, overlap_ratio: float = 0.1):
        """
        初始化条带分解器

        Args:
            swath_width: 幅宽（米），默认10km
            overlap_ratio: 条带重叠率（0-1），默认10%
        """
        super().__init__(DecompositionStrategy.STRIP)
        self.swath_width = swath_width
        self.overlap_ratio = overlap_ratio

    def decompose(self, target: Target, imaging_mode: ImagingMode = ImagingMode.STRIPMAP, **kwargs) -> List[Target]:
        """
        将区域目标分解为条带

        Args:
            target: 区域目标
            imaging_mode: 成像模式，影响条带数量和大小
            **kwargs: 额外参数
                - strip_direction: 条带方向（度，0=北，90=东），默认自动计算

        Returns:
            List[Target]: 条带子目标列表
        """
        self.validate_target(target)

        # 根据成像模式调整幅宽
        effective_swath = self._get_effective_swath(imaging_mode)

        # 确定条带方向
        strip_direction = kwargs.get('strip_direction')
        if strip_direction is None:
            # 自动计算最优方向（沿最长边）
            strip_direction = self._calculate_optimal_direction(target.area_vertices)

        # 计算边界框
        min_lon, max_lon, min_lat, max_lat = self._get_bounding_box(target.area_vertices)

        # 生成条带
        sub_targets = []
        strip_id = 0

        # 计算条带间距（考虑重叠）
        strip_spacing = effective_swath * (1 - self.overlap_ratio)
        strip_spacing_km = strip_spacing / 1000.0

        # 根据方向确定扫描方式
        if abs(math.sin(math.radians(strip_direction))) > abs(math.cos(math.radians(strip_direction))):
            # 主要沿纬度方向扫描
            sub_targets = self._generate_latitude_strips(
                target, min_lon, max_lon, min_lat, max_lat,
                strip_spacing_km, strip_direction, imaging_mode
            )
        else:
            # 主要沿经度方向扫描
            sub_targets = self._generate_longitude_strips(
                target, min_lon, max_lon, min_lat, max_lat,
                strip_spacing_km, strip_direction, imaging_mode
            )

        return sub_targets

    def _get_effective_swath(self, imaging_mode: ImagingMode) -> float:
        """
        根据成像模式获取有效幅宽

        Args:
            imaging_mode: 成像模式

        Returns:
            float: 有效幅宽（米）
        """
        mode_multipliers = {
            ImagingMode.SPOTLIGHT: 0.1,          # 聚束模式：窄视场
            ImagingMode.SLIDING_SPOTLIGHT: 0.3,  # 滑动聚束：中等视场
            ImagingMode.STRIPMAP: 1.0,           # 条带模式：标准幅宽
        }

        multiplier = mode_multipliers.get(imaging_mode, 1.0)
        return self.swath_width * multiplier

    def _calculate_optimal_direction(self, vertices: List[Tuple[float, float]]) -> float:
        """
        计算最优条带方向（沿多边形最长边）

        Args:
            vertices: 多边形顶点

        Returns:
            float: 方向角度（度，0-180）
        """
        # 计算多边形的主轴方向
        # 简化：计算边界框的长边方向
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]

        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)

        if lon_range >= lat_range:
            return 90.0  # 沿经度方向（东西向条带）
        else:
            return 0.0   # 沿纬度方向（南北向条带）

    def _get_bounding_box(self, vertices: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """计算边界框"""
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]
        return (min(lons), max(lons), min(lats), max(lats))

    def _generate_latitude_strips(
        self,
        target: Target,
        min_lon: float, max_lon: float,
        min_lat: float, max_lat: float,
        strip_spacing_km: float,
        direction: float,
        imaging_mode: ImagingMode
    ) -> List[Target]:
        """生成沿纬度方向的条带"""
        sub_targets = []
        strip_id = 0

        # 条带宽度（度）
        strip_width_lat = (self.swath_width / 1000.0) / self.KM_PER_DEGREE_LAT
        spacing_lat = strip_spacing_km / self.KM_PER_DEGREE_LAT

        lat = min_lat
        while lat <= max_lat:
            strip_id += 1

            # 条带的南北边界
            strip_min_lat = lat
            strip_max_lat = min(lat + strip_width_lat, max_lat + 0.01)

            # 裁剪条带到多边形内
            strip_vertices = self._clip_strip_to_polygon(
                min_lon, max_lon, strip_min_lat, strip_max_lat,
                target.area_vertices
            )

            if strip_vertices and len(strip_vertices) >= 3:
                sub_id = f"{target.id}-S{strip_id:04d}"
                sub_name = f"{target.name} 条带{strip_id}"

                sub_target = self.create_subtarget(
                    parent_target=target,
                    sub_id=sub_id,
                    sub_name=sub_name,
                    vertices=strip_vertices,
                    parent_id=target.id,
                    strip_number=strip_id,
                    imaging_mode=imaging_mode.value,
                    strip_direction=direction,
                )
                sub_targets.append(sub_target)

            lat += spacing_lat

        return sub_targets

    def _generate_longitude_strips(
        self,
        target: Target,
        min_lon: float, max_lon: float,
        min_lat: float, max_lat: float,
        strip_spacing_km: float,
        direction: float,
        imaging_mode: ImagingMode
    ) -> List[Target]:
        """生成沿经度方向的条带"""
        sub_targets = []
        strip_id = 0

        # 条带宽度（度）
        strip_width_lon = (self.swath_width / 1000.0) / self.KM_PER_DEGREE_LON
        spacing_lon = strip_spacing_km / self.KM_PER_DEGREE_LON

        lon = min_lon
        while lon <= max_lon:
            strip_id += 1

            # 条带的东西边界
            strip_min_lon = lon
            strip_max_lon = min(lon + strip_width_lon, max_lon + 0.01)

            # 裁剪条带到多边形内
            strip_vertices = self._clip_strip_to_polygon_lon(
                strip_min_lon, strip_max_lon, min_lat, max_lat,
                target.area_vertices
            )

            if strip_vertices and len(strip_vertices) >= 3:
                sub_id = f"{target.id}-S{strip_id:04d}"
                sub_name = f"{target.name} 条带{strip_id}"

                sub_target = self.create_subtarget(
                    parent_target=target,
                    sub_id=sub_id,
                    sub_name=sub_name,
                    vertices=strip_vertices,
                    parent_id=target.id,
                    strip_number=strip_id,
                    imaging_mode=imaging_mode.value,
                    strip_direction=direction,
                )
                sub_targets.append(sub_target)

            lon += spacing_lon

        return sub_targets

    def _clip_strip_to_polygon(
        self,
        min_lon: float, max_lon: float,
        min_lat: float, max_lat: float,
        polygon: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        将条带裁剪到多边形内

        简化实现：返回条带边界框与多边形的交集区域
        """
        # 简化：返回条带的四个角点，但只保留在多边形内的点
        # 实际实现应该使用Sutherland-Hodgman等裁剪算法

        strip_corners = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
        ]

        # 简化的裁剪：检查角点是否在多边形内
        # 实际应该使用更精确的裁剪算法
        return strip_corners

    def _clip_strip_to_polygon_lon(
        self,
        min_lon: float, max_lon: float,
        min_lat: float, max_lat: float,
        polygon: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """将经度方向的条带裁剪到多边形内"""
        strip_corners = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
        ]
        return strip_corners
