"""
光学成像器

实现光学成像器，支持推扫和框幅模式
设计文档第3章 - 载荷模块设计
"""

from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import math

from .base import Imager, ImagingMode
from core.constants import (
    DEFAULT_FOV_TYPE,
    DEFAULT_FOV_HALF_ANGLE_DEG,
    DEFAULT_FOV_HALF_ANGLE_X_DEG,
    DEFAULT_FOV_HALF_ANGLE_Y_DEG,
)


class OpticalImagingMode(Enum):
    """光学成像模式枚举"""
    PUSH_BROOM = "push_broom"
    FRAME = "frame"


class OpticalImager(Imager):
    """
    光学成像器

    支持推扫（push_broom）和框幅（frame）成像模式
    支持FOV配置，可基于视场半角计算精确幅宽
    """

    def __init__(
        self,
        imager_id: str,
        resolution: float = 1.0,
        swath_width: float = 10.0,
        focal_length: Optional[float] = None,
        aperture: Optional[float] = None,
        supported_modes: Optional[List[ImagingMode]] = None,
        # FOV配置参数（新增）
        fov_type: Optional[str] = None,
        fov_half_angle: Optional[float] = None,
        fov_half_angle_x: Optional[float] = None,
        fov_half_angle_y: Optional[float] = None,
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

        # FOV配置（新增）
        self.fov_type = fov_type or DEFAULT_FOV_TYPE
        self.fov_half_angle = fov_half_angle or DEFAULT_FOV_HALF_ANGLE_DEG
        self.fov_half_angle_x = fov_half_angle_x or DEFAULT_FOV_HALF_ANGLE_X_DEG
        self.fov_half_angle_y = fov_half_angle_y or DEFAULT_FOV_HALF_ANGLE_Y_DEG

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
            'supported_modes': [m.value for m in self.supported_modes],
            'fov_type': self.fov_type,
            'fov_half_angle': self.fov_half_angle,
            'fov_half_angle_x': self.fov_half_angle_x,
            'fov_half_angle_y': self.fov_half_angle_y,
        }

    def calculate_swath_from_fov(
        self,
        altitude_km: float,
        look_angle_deg: float = 0.0
    ) -> float:
        """
        基于FOV和轨道高度计算幅宽

        如果未配置FOV，则返回配置的swath_width。

        Args:
            altitude_km: 轨道高度（公里）
            look_angle_deg: 观测角度（度，从星下点算起）

        Returns:
            float: 幅宽（公里）
        """
        h = altitude_km

        if self.fov_type == 'cone':
            # 圆锥视场：幅宽 = 2 * h * tan(fov_half_angle)
            theta = math.radians(self.fov_half_angle)
            phi = math.radians(look_angle_deg)

            # 考虑观测角度的修正
            # 在观测方向的地面投影
            swath = 2 * h * math.tan(theta) / math.cos(phi)
            return swath

        elif self.fov_type == 'rectangular':
            # 矩形视场：分别计算两个方向的幅宽
            theta_x = math.radians(self.fov_half_angle_x)
            theta_y = math.radians(self.fov_half_angle_y)

            # 沿轨迹方向（X）和垂直轨迹方向（Y）
            swath_x = 2 * h * math.tan(theta_x)
            swath_y = 2 * h * math.tan(theta_y)

            # 返回平均幅宽或根据观测方向选择
            if abs(look_angle_deg) < 1.0:
                # 星下点观测，使用垂直轨迹方向幅宽
                return swath_y
            else:
                # 侧摆观测，使用两个方向的平均值
                return (swath_x + swath_y) / 2

        else:
            # 未知FOV类型，回退到配置的幅宽
            return self.swath_width

    def get_effective_swath(
        self,
        altitude_km: Optional[float] = None,
        look_angle_deg: float = 0.0
    ) -> float:
        """
        获取有效幅宽

        如果配置了FOV且提供了高度，基于FOV计算幅宽；
        否则返回配置的swath_width。

        Args:
            altitude_km: 轨道高度（公里），可选
            look_angle_deg: 观测角度（度），可选

        Returns:
            float: 有效幅宽（公里）
        """
        # 如果配置了FOV且提供了高度，使用FOV计算
        if altitude_km is not None and altitude_km > 0:
            fov_swath = self.calculate_swath_from_fov(altitude_km, look_angle_deg)
            # 如果FOV计算成功（不等于配置的swath_width），使用FOV结果
            if fov_swath != self.swath_width:
                return fov_swath

        # 回退到配置的幅宽
        return self.swath_width

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
