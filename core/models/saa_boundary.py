"""
南大西洋异常区 (SAA) 边界模型

使用简化 NASA 椭圆模型定义 SAA 区域边界。
卫星进入此区域时，电子设备可能受高能粒子影响。
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class SAABoundaryModel:
    """
    南大西洋异常区(SAA)边界模型 - 简化NASA椭圆模型

    基于经验数据拟合的椭圆区域：
    - 中心: 西经45°, 南纬25° (巴西海岸附近)
    - 长轴: 80° (东西向)
    - 短轴: 60° (南北向)

    数学表示: ((lon + 45)/40)² + ((lat + 25)/30)² <= 1

    Attributes:
        center_lon: 中心经度（度），默认-45.0
        center_lat: 中心纬度（度），默认-25.0
        semi_major: 半长轴（度），默认40.0
        semi_minor: 半短轴（度），默认30.0
    """

    center_lon: float = -45.0  # 中心经度
    center_lat: float = -25.0  # 中心纬度
    semi_major: float = 40.0  # 半长轴（度）
    semi_minor: float = 30.0  # 半短轴（度）

    def is_inside(self, lon: float, lat: float) -> bool:
        """
        检查坐标是否在 SAA 椭圆区域内

        使用归一化椭圆方程：
        ((lon - center_lon) / semi_major)² + ((lat - center_lat) / semi_minor)² <= 1

        Args:
            lon: 经度（度），范围 -180 ~ +180
            lat: 纬度（度），范围 -90 ~ +90

        Returns:
            bool: 如果在 SAA 区域内返回 True，否则返回 False

        Raises:
            ValueError: 如果输入超出有效范围
        """
        # 参数验证
        if not -180 <= lon <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {lon}")
        if not -90 <= lat <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {lat}")

        # 归一化坐标
        normalized_lon = (lon - self.center_lon) / self.semi_major
        normalized_lat = (lat - self.center_lat) / self.semi_minor

        # 椭圆方程: x²/a² + y²/b² <= 1
        # 使用小容差处理浮点数精度问题
        value = normalized_lon ** 2 + normalized_lat ** 2
        return value <= 1.0 + 1e-10

    def get_boundary_points(self, num_points: int = 36) -> List[Tuple[float, float]]:
        """
        获取 SAA 边界上的采样点（用于可视化）

        使用参数方程生成椭圆上的点：
        lon = center_lon + semi_major * cos(theta)
        lat = center_lat + semi_minor * sin(theta)

        Args:
            num_points: 采样点数量，默认36个（每10度一个点）

        Returns:
            List[Tuple[float, float]]: 边界点列表 [(lon, lat), ...]
        """
        points = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            lon = self.center_lon + self.semi_major * math.cos(theta)
            lat = self.center_lat + self.semi_minor * math.sin(theta)
            points.append((lon, lat))
        return points
