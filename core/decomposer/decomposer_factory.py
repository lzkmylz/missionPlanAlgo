"""
分解器工厂

根据卫星类型和成像模式创建合适的分解器
"""

from typing import Optional

from .base_decomposer import BaseDecomposer, DecompositionStrategy
from .grid_decomposer import GridDecomposer
from .strip_decomposer import StripDecomposer
from core.models import SatelliteType


class DecomposerFactory:
    """
    分解器工厂

    根据目标类型、卫星类型和成像模式自动选择合适的分解策略
    """

    @classmethod
    def create(
        cls,
        strategy: DecompositionStrategy,
        **kwargs
    ) -> BaseDecomposer:
        """
        创建分解器

        Args:
            strategy: 分解策略
            **kwargs: 分解器特定参数
                - GridDecomposer: resolution (float)
                - StripDecomposer: swath_width (float), overlap_ratio (float)

        Returns:
            BaseDecomposer: 分解器实例

        Raises:
            ValueError: 如果策略无效
        """
        if strategy == DecompositionStrategy.GRID:
            resolution = kwargs.get('resolution', 100.0)
            return GridDecomposer(resolution=resolution)

        elif strategy == DecompositionStrategy.STRIP:
            swath_width = kwargs.get('swath_width', 10000.0)
            overlap_ratio = kwargs.get('overlap_ratio', 0.1)
            return StripDecomposer(
                swath_width=swath_width,
                overlap_ratio=overlap_ratio
            )

        else:
            raise ValueError(f"Unknown decomposition strategy: {strategy}")

    @classmethod
    def create_for_satellite_type(
        cls,
        satellite_type: SatelliteType,
        **kwargs
    ) -> BaseDecomposer:
        """
        根据卫星类型创建合适的分解器

        Args:
            satellite_type: 卫星类型
            **kwargs: 分解器参数

        Returns:
            BaseDecomposer: 分解器实例
        """
        if satellite_type in [SatelliteType.OPTICAL_1, SatelliteType.OPTICAL_2]:
            # 光学卫星使用网格分解
            resolution = kwargs.get('resolution', 100.0)
            return GridDecomposer(resolution=resolution)

        elif satellite_type in [SatelliteType.SAR_1, SatelliteType.SAR_2]:
            # SAR卫星使用条带分解
            swath_width = kwargs.get('swath_width', 10000.0)
            overlap_ratio = kwargs.get('overlap_ratio', 0.1)
            return StripDecomposer(
                swath_width=swath_width,
                overlap_ratio=overlap_ratio
            )

        else:
            # 默认使用网格分解
            resolution = kwargs.get('resolution', 100.0)
            return GridDecomposer(resolution=resolution)

    @classmethod
    def get_recommended_resolution(
        cls,
        satellite_type: SatelliteType,
        required_resolution: float
    ) -> float:
        """
        获取推荐的网格分辨率

        Args:
            satellite_type: 卫星类型
            required_resolution: 所需分辨率（米）

        Returns:
            float: 推荐的网格分辨率（米）
        """
        # 网格分辨率通常是所需分辨率的1-2倍
        # 这样可以确保覆盖同时不过度采样
        return max(required_resolution * 1.5, 10.0)

    @classmethod
    def get_recommended_swath(
        cls,
        satellite_type: SatelliteType,
        altitude: float = 500000.0
    ) -> float:
        """
        获取推荐的幅宽

        Args:
            satellite_type: 卫星类型
            altitude: 轨道高度（米）

        Returns:
            float: 推荐的幅宽（米）
        """
        # 基于卫星类型的默认幅宽
        default_swaths = {
            SatelliteType.SAR_1: 10000.0,   # 10km
            SatelliteType.SAR_2: 20000.0,   # 20km（增强型SAR）
        }

        return default_swaths.get(satellite_type, 10000.0)
