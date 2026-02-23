"""
分解器基类

定义目标分解的通用接口
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any, Optional

from core.models import Target, TargetType


class DecompositionStrategy(Enum):
    """分解策略枚举"""
    GRID = "grid"      # 网格化分解（光学卫星）
    STRIP = "strip"    # 条带化分解（SAR卫星）


class BaseDecomposer(ABC):
    """
    目标分解器基类

    将大区域目标分解为可执行的子任务
    """

    def __init__(self, strategy: DecompositionStrategy):
        """
        初始化分解器

        Args:
            strategy: 分解策略
        """
        self.strategy = strategy

    @abstractmethod
    def decompose(self, target: Target, **kwargs) -> List[Target]:
        """
        分解目标

        Args:
            target: 要分解的目标（必须是区域目标）
            **kwargs: 额外的分解参数

        Returns:
            List[Target]: 分解后的子目标列表

        Raises:
            ValueError: 如果目标类型不支持或参数无效
        """
        pass

    def validate_target(self, target: Target) -> None:
        """
        验证目标是否可以分解

        Args:
            target: 要验证的目标

        Raises:
            ValueError: 如果目标无效
        """
        if target.target_type != TargetType.AREA:
            raise ValueError(
                f"Cannot decompose target of type {target.target_type}. "
                "Only AREA targets can be decomposed."
            )

        if len(target.area_vertices) < 3:
            raise ValueError(
                f"Area target must have at least 3 vertices, "
                f"got {len(target.area_vertices)}"
            )

    def create_subtarget(
        self,
        parent_target: Target,
        sub_id: str,
        sub_name: str,
        vertices: List[tuple],
        **additional_attrs
    ) -> Target:
        """
        创建子目标

        Args:
            parent_target: 父目标
            sub_id: 子目标ID
            sub_name: 子目标名称
            vertices: 子目标顶点（区域）或中心坐标（点）
            **additional_attrs: 额外属性

        Returns:
            Target: 创建的子目标
        """
        # 确定子目标类型
        if len(vertices) == 1:
            # 点目标
            target_type = TargetType.POINT
            longitude, latitude = vertices[0]
            area_vertices = []
        else:
            # 区域目标
            target_type = TargetType.AREA
            longitude = None
            latitude = None
            area_vertices = vertices

        # 继承父目标的属性
        subtarget = Target(
            id=sub_id,
            name=sub_name,
            target_type=target_type,
            longitude=longitude,
            latitude=latitude,
            area_vertices=area_vertices,
            priority=parent_target.priority,
            required_observations=parent_target.required_observations,
            resolution_required=parent_target.resolution_required,
            time_window_start=parent_target.time_window_start,
            time_window_end=parent_target.time_window_end,
            immediate_downlink=parent_target.immediate_downlink,
        )

        # 添加额外属性
        for attr, value in additional_attrs.items():
            setattr(subtarget, attr, value)

        return subtarget
