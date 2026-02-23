"""
网格分解器

将区域目标网格化为点目标（适用于光学卫星）
"""

import math
from typing import List, Tuple

from .base_decomposer import BaseDecomposer, DecompositionStrategy
from core.models import Target, TargetType


class GridDecomposer(BaseDecomposer):
    """
    网格分解器

    将多边形区域分解为规则网格，每个网格中心作为点目标。
    适用于光学卫星的推扫或框幅成像模式。

    Attributes:
        resolution: 网格分辨率（米）
    """

    # 每度纬度/经度对应的公里数（简化，假设在中纬度地区）
    KM_PER_DEGREE_LAT = 111.0
    KM_PER_DEGREE_LON = 111.0  # 近似值，实际随纬度变化

    def __init__(self, resolution: float = 100.0):
        """
        初始化网格分解器

        Args:
            resolution: 网格分辨率（米），默认100米
        """
        super().__init__(DecompositionStrategy.GRID)
        self.resolution = resolution

    def decompose(self, target: Target, **kwargs) -> List[Target]:
        """
        将区域目标网格化为点目标

        Args:
            target: 区域目标
            **kwargs: 额外参数（可选）
                - buffer_ratio: 边界缓冲区比例（默认0.1）

        Returns:
            List[Target]: 点目标列表
        """
        self.validate_target(target)

        buffer_ratio = kwargs.get('buffer_ratio', 0.1)

        # 计算边界框
        min_lon, max_lon, min_lat, max_lat = self._get_bounding_box(target.area_vertices)

        # 添加边界缓冲
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        min_lon -= lon_range * buffer_ratio
        max_lon += lon_range * buffer_ratio
        min_lat -= lat_range * buffer_ratio
        max_lat += lat_range * buffer_ratio

        # 计算网格大小（度）
        resolution_km = self.resolution / 1000.0
        grid_size_lon = resolution_km / self.KM_PER_DEGREE_LON
        grid_size_lat = resolution_km / self.KM_PER_DEGREE_LAT

        sub_targets = []
        grid_id = 0

        # 生成网格点
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                # 检查点是否在多边形内
                if self._point_in_polygon(lon, lat, target.area_vertices):
                    grid_id += 1
                    sub_id = f"{target.id}-G{grid_id:04d}"
                    sub_name = f"{target.name} 网格{grid_id}"

                    sub_target = self.create_subtarget(
                        parent_target=target,
                        sub_id=sub_id,
                        sub_name=sub_name,
                        vertices=[(lon, lat)],
                        parent_id=target.id,
                        grid_row=int((lat - min_lat) / grid_size_lat),
                        grid_col=int((lon - min_lon) / grid_size_lon),
                    )
                    sub_targets.append(sub_target)

                lon += grid_size_lon
            lat += grid_size_lat

        return sub_targets

    def _get_bounding_box(self, vertices: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        计算多边形的边界框

        Args:
            vertices: 顶点列表 [(lon, lat), ...]

        Returns:
            Tuple[float, float, float, float]: (min_lon, max_lon, min_lat, max_lat)
        """
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]

        return (min(lons), max(lons), min(lats), max(lats))

    def _point_in_polygon(self, lon: float, lat: float, vertices: List[Tuple[float, float]]) -> bool:
        """
        使用射线法判断点是否在多边形内

        Args:
            lon: 经度
            lat: 纬度
            vertices: 多边形顶点

        Returns:
            bool: 是否在多边形内
        """
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
