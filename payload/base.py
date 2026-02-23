"""
载荷基类

定义成像器的抽象基类和通用接口
设计文档第3章 - 载荷模块设计
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum


class ImagingMode(Enum):
    """成像模式枚举"""
    # 光学模式
    PUSH_BROOM = "push_broom"
    FRAME = "frame"

    # SAR模式
    SPOTLIGHT = "spotlight"
    SLIDING_SPOTLIGHT = "sliding_spotlight"
    STRIPMAP = "stripmap"


@dataclass
class ImagerSpecs:
    """成像器规格"""
    imager_id: str
    imager_type: str  # "optical" or "sar"
    resolution: float  # 分辨率（米）
    swath_width: float  # 幅宽（千米）
    supported_modes: List[ImagingMode]
    focal_length: Optional[float] = None  # 焦距（米）- 光学
    aperture: Optional[float] = None  # 孔径（米）- 光学
    band: Optional[str] = None  # 频段 - SAR
    polarization: Optional[str] = None  # 极化方式 - SAR
    min_look_angle: Optional[float] = None  # 最小视角（度）- SAR
    max_look_angle: Optional[float] = None  # 最大视角（度）- SAR


class Imager(ABC):
    """
    成像器抽象基类

    所有成像器（光学、SAR）必须继承此类
    """

    def __init__(
        self,
        imager_id: str,
        resolution: float = 1.0,
        swath_width: float = 10.0,
        supported_modes: Optional[List[ImagingMode]] = None
    ):
        """
        初始化成像器

        Args:
            imager_id: 成像器唯一标识
            resolution: 分辨率（米）
            swath_width: 幅宽（千米）
            supported_modes: 支持的成像模式列表

        Raises:
            ValueError: 如果参数无效
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        if swath_width <= 0:
            raise ValueError(f"Swath width must be positive, got {swath_width}")

        self.imager_id = imager_id
        self.resolution = resolution
        self.swath_width = swath_width
        self.supported_modes = supported_modes or []

        if not self.supported_modes:
            raise ValueError("At least one imaging mode must be supported")

    @abstractmethod
    def calculate_imaging_time(
        self,
        target_size: Tuple[float, float],
        mode: ImagingMode,
        **kwargs
    ) -> float:
        """
        计算成像所需时间

        Args:
            target_size: 目标尺寸 (宽度, 长度) 单位：米
            mode: 成像模式
            **kwargs: 额外参数

        Returns:
            float: 成像时间（秒）
        """
        pass

    @abstractmethod
    def get_specs(self) -> Dict[str, Any]:
        """
        获取成像器规格

        Returns:
            Dict[str, Any]: 规格字典
        """
        pass

    def supports_mode(self, mode: ImagingMode) -> bool:
        """
        检查是否支持指定成像模式

        Args:
            mode: 成像模式

        Returns:
            bool: 是否支持
        """
        return mode in self.supported_modes

    def validate_mode(self, mode: ImagingMode) -> None:
        """
        验证成像模式是否支持

        Args:
            mode: 成像模式

        Raises:
            ValueError: 如果不支持该模式
        """
        if not self.supports_mode(mode):
            raise ValueError(
                f"Mode {mode.value} not supported by imager {self.imager_id}. "
                f"Supported modes: {[m.value for m in self.supported_modes]}"
            )
