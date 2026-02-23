"""
光学成像器

实现光学成像器，支持推扫和框幅模式
设计文档第3章 - 载荷模块设计
"""

from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

from .base import Imager, ImagingMode


class OpticalImagingMode(Enum):
    """光学成像模式枚举"""
    PUSH_BROOM = "push_broom"
    FRAME = "frame"


class OpticalImager(Imager):
    """
    光学成像器

    支持推扫（push_broom）和框幅（frame）成像模式
    """

    def __init__(
        self,
        imager_id: str,
        resolution: float = 1.0,
        swath_width: float = 10.0,
        focal_length: Optional[float] = None,
        aperture: Optional[float] = None,
        supported_modes: Optional[List[ImagingMode]] = None
    ):
        """
        初始化光学成像器

        Args:
            imager_id: 成像器唯一标识
            resolution: 分辨率（米）
            swath_width: 幅宽（千米）
            focal_length: 焦距（米）
            aperture: 孔径（米）
            supported_modes: 支持的成像模式列表，默认[PUSH_BROOM, FRAME]
        """
        if supported_modes is None:
            supported_modes = [ImagingMode.PUSH_BROOM, ImagingMode.FRAME]

        super().__init__(
            imager_id=imager_id,
            resolution=resolution,
            swath_width=swath_width,
            supported_modes=supported_modes
        )

        self.focal_length = focal_length
        self.aperture = aperture

        # 推扫速度系数（米/秒）- 典型卫星轨道速度
        self._scan_velocity = 7000.0  # 约7km/s

    def calculate_imaging_time(
        self,
        target_size: Tuple[float, float],
        mode: ImagingMode,
        **kwargs
    ) -> float:
        """
        计算光学成像所需时间

        Args:
            target_size: 目标尺寸 (宽度, 长度) 单位：米
            mode: 成像模式（PUSH_BROOM或FRAME）
            **kwargs: 额外参数
                - integration_time: 积分时间（秒）

        Returns:
            float: 成像时间（秒）

        Raises:
            ValueError: 如果不支持该模式
        """
        self.validate_mode(mode)

        width, length = target_size
        integration_time = kwargs.get('integration_time', 0.001)  # 默认1ms

        if mode == ImagingMode.PUSH_BROOM:
            # 推扫模式：成像时间 = 目标长度 / 扫描速度 + 积分时间
            # 考虑幅宽覆盖
            num_strips = max(1, int(width / (self.swath_width * 1000)))
            scan_time = length / self._scan_velocity
            total_time = scan_time * num_strips + integration_time
            return max(total_time, integration_time)

        elif mode == ImagingMode.FRAME:
            # 框幅模式：单次曝光
            # 成像时间 = 积分时间 + 读出时间（假设读出时间=积分时间）
            readout_time = kwargs.get('readout_time', integration_time)
            return integration_time + readout_time

        else:
            raise ValueError(f"Unsupported optical imaging mode: {mode}")

    def get_specs(self) -> Dict[str, Any]:
        """
        获取光学成像器规格

        Returns:
            Dict[str, Any]: 规格字典
        """
        return {
            'imager_id': self.imager_id,
            'imager_type': 'optical',
            'resolution': self.resolution,
            'swath_width': self.swath_width,
            'focal_length': self.focal_length,
            'aperture': self.aperture,
            'supported_modes': [m.value for m in self.supported_modes]
        }

    def calculate_ground_sample_distance(
        self,
        altitude: float
    ) -> float:
        """
        计算地面采样距离（GSD）

        Args:
            altitude: 轨道高度（米）

        Returns:
            float: GSD（米）
        """
        if self.focal_length and self.focal_length > 0:
            # GSD = (像素尺寸 * 轨道高度) / 焦距
            # 假设像素尺寸 = 分辨率 / (轨道高度/焦距) 的简化计算
            return (self.resolution * altitude) / (self.focal_length * 1000)
        return self.resolution
