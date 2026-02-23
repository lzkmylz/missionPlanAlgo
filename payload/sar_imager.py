"""
SAR成像器

实现SAR成像器，支持聚束、滑动聚束、条带模式
设计文档第3章 - 载荷模块设计
"""

from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

from .base import Imager, ImagingMode


class SARImagingMode(Enum):
    """SAR成像模式枚举"""
    SPOTLIGHT = "spotlight"
    SLIDING_SPOTLIGHT = "sliding_spotlight"
    STRIPMAP = "stripmap"


class SARImager(Imager):
    """
    SAR成像器

    支持三种成像模式：
    - 聚束模式（Spotlight）：高分辨率，小覆盖区域
    - 滑动聚束模式（Sliding Spotlight）：中等分辨率，中等覆盖区域
    - 条带模式（Stripmap）：标准分辨率，大覆盖区域
    """

    def __init__(
        self,
        imager_id: str,
        resolution: float = 1.0,
        swath_width: float = 20.0,
        band: str = "X",
        polarization: str = "VV",
        min_look_angle: float = 20.0,
        max_look_angle: float = 50.0,
        supported_modes: Optional[List[SARImagingMode]] = None
    ):
        """
        初始化SAR成像器

        Args:
            imager_id: 成像器唯一标识
            resolution: 分辨率（米）
            swath_width: 幅宽（千米）
            band: 频段（X, C, L等）
            polarization: 极化方式（VV, VH, HH, HV等）
            min_look_angle: 最小视角（度）
            max_look_angle: 最大视角（度）
            supported_modes: 支持的成像模式列表，默认全部三种

        Raises:
            ValueError: 如果视角参数无效
        """
        if min_look_angle >= max_look_angle:
            raise ValueError(
                f"min_look_angle ({min_look_angle}) must be less than "
                f"max_look_angle ({max_look_angle})"
            )

        if supported_modes is None:
            supported_modes = [
                SARImagingMode.SPOTLIGHT,
                SARImagingMode.SLIDING_SPOTLIGHT,
                SARImagingMode.STRIPMAP
            ]

        # 将SARImagingMode转换为ImagingMode
        imaging_modes = []
        for mode in supported_modes:
            if mode == SARImagingMode.SPOTLIGHT:
                imaging_modes.append(ImagingMode.SPOTLIGHT)
            elif mode == SARImagingMode.SLIDING_SPOTLIGHT:
                imaging_modes.append(ImagingMode.SLIDING_SPOTLIGHT)
            elif mode == SARImagingMode.STRIPMAP:
                imaging_modes.append(ImagingMode.STRIPMAP)

        super().__init__(
            imager_id=imager_id,
            resolution=resolution,
            swath_width=swath_width,
            supported_modes=imaging_modes
        )

        self.band = band
        self.polarization = polarization
        self.min_look_angle = min_look_angle
        self.max_look_angle = max_look_angle

        # 模式特定的参数
        self._mode_params = {
            SARImagingMode.SPOTLIGHT: {
                'integration_factor': 3.0,  # 聚束模式积分时间更长
                'max_scene_size': (10.0, 10.0)  # 最大场景尺寸（km）
            },
            SARImagingMode.SLIDING_SPOTLIGHT: {
                'integration_factor': 2.0,
                'max_scene_size': (20.0, 50.0)
            },
            SARImagingMode.STRIPMAP: {
                'integration_factor': 1.0,
                'max_scene_size': (self.swath_width, float('inf'))
            }
        }

    def calculate_imaging_time(
        self,
        target_size: Tuple[float, float],
        mode: ImagingMode,
        **kwargs
    ) -> float:
        """
        计算SAR成像所需时间

        Args:
            target_size: 目标尺寸 (宽度, 长度) 单位：米
            mode: 成像模式（SPOTLIGHT, SLIDING_SPOTLIGHT, STRIPMAP）
            **kwargs: 额外参数
                - prf: 脉冲重复频率（Hz）
                - look_angle: 观测视角（度）

        Returns:
            float: 成像时间（秒）

        Raises:
            ValueError: 如果不支持该模式或参数无效
        """
        self.validate_mode(mode)

        # 将ImagingMode或SARImagingMode转换为SARImagingMode
        sar_mode = self._to_sar_mode(mode)

        width, length = target_size
        params = self._mode_params[sar_mode]

        # 基础脉冲重复频率（Hz）
        prf = kwargs.get('prf', 1000.0)

        # 计算合成孔径长度
        wavelength = self._get_wavelength()
        satellite_velocity = 7000.0  # m/s

        if sar_mode == SARImagingMode.SPOTLIGHT:
            # 聚束模式：天线始终指向目标区域中心
            # 成像时间由合成孔径长度决定
            synthetic_aperture_length = (wavelength * length) / (2 * self.resolution)
            integration_time = synthetic_aperture_length / satellite_velocity
            return integration_time * params['integration_factor']

        elif sar_mode == SARImagingMode.SLIDING_SPOTLIGHT:
            # 滑动聚束模式：天线在成像过程中缓慢移动
            # 成像时间介于聚束和条带之间
            synthetic_aperture_length = (wavelength * length) / (2 * self.resolution * 1.5)
            integration_time = synthetic_aperture_length / satellite_velocity
            return integration_time * params['integration_factor']

        elif sar_mode == SARImagingMode.STRIPMAP:
            # 条带模式：标准SAR成像
            # 成像时间 = 目标长度 / 卫星速度
            return length / satellite_velocity

        else:
            raise ValueError(f"Unsupported SAR imaging mode: {mode}")

    def _to_sar_mode(self, mode) -> SARImagingMode:
        """将ImagingMode或SARImagingMode转换为SARImagingMode"""
        if isinstance(mode, SARImagingMode):
            return mode
        mapping = {
            ImagingMode.SPOTLIGHT: SARImagingMode.SPOTLIGHT,
            ImagingMode.SLIDING_SPOTLIGHT: SARImagingMode.SLIDING_SPOTLIGHT,
            ImagingMode.STRIPMAP: SARImagingMode.STRIPMAP
        }
        return mapping.get(mode)

    def _to_imaging_mode(self, mode) -> ImagingMode:
        """将SARImagingMode转换为ImagingMode"""
        if isinstance(mode, ImagingMode):
            return mode
        mapping = {
            SARImagingMode.SPOTLIGHT: ImagingMode.SPOTLIGHT,
            SARImagingMode.SLIDING_SPOTLIGHT: ImagingMode.SLIDING_SPOTLIGHT,
            SARImagingMode.STRIPMAP: ImagingMode.STRIPMAP
        }
        result = mapping.get(mode)
        if result is None:
            raise ValueError(f"Cannot convert {mode} to ImagingMode")
        return result

    def supports_mode(self, mode) -> bool:
        """
        检查是否支持指定成像模式

        Args:
            mode: 成像模式（SARImagingMode或ImagingMode）

        Returns:
            bool: 是否支持
        """
        # 统一转换为ImagingMode进行比较
        try:
            imaging_mode = self._to_imaging_mode(mode)
            return imaging_mode in self.supported_modes
        except ValueError:
            return False

    def validate_mode(self, mode) -> None:
        """
        验证成像模式是否支持

        Args:
            mode: 成像模式

        Raises:
            ValueError: 如果不支持该模式
        """
        if not self.supports_mode(mode):
            raise ValueError(
                f"Mode {mode.value if hasattr(mode, 'value') else mode} not supported by imager {self.imager_id}. "
                f"Supported modes: {[m.value for m in self.supported_modes]}"
            )

    def _get_wavelength(self) -> float:
        """根据频段获取波长（米）"""
        # 典型SAR频段波长
        wavelengths = {
            'X': 0.031,  # ~9.6 GHz
            'C': 0.056,  # ~5.3 GHz
            'L': 0.235,  # ~1.275 GHz
            'S': 0.096,  # ~3.1 GHz
            'Ku': 0.022,  # ~13.6 GHz
            'Ka': 0.008,  # ~35 GHz
        }
        return wavelengths.get(self.band.upper(), 0.031)  # 默认X波段

    def get_specs(self) -> Dict[str, Any]:
        """
        获取SAR成像器规格

        Returns:
            Dict[str, Any]: 规格字典
        """
        return {
            'imager_id': self.imager_id,
            'imager_type': 'sar',
            'resolution': self.resolution,
            'swath_width': self.swath_width,
            'band': self.band,
            'polarization': self.polarization,
            'min_look_angle': self.min_look_angle,
            'max_look_angle': self.max_look_angle,
            'supported_modes': [m.value for m in self.supported_modes]
        }

    def calculate_ground_range_resolution(self, look_angle: float) -> float:
        """
        计算地距分辨率

        Args:
            look_angle: 观测视角（度）

        Returns:
            float: 地距分辨率（米）
        """
        import math
        # 地距分辨率 = 斜距分辨率 / sin(视角)
        return self.resolution / math.sin(math.radians(look_angle))

    def is_look_angle_valid(self, look_angle: float) -> bool:
        """
        检查观测视角是否有效

        Args:
            look_angle: 观测视角（度）

        Returns:
            bool: 是否有效
        """
        return self.min_look_angle <= look_angle <= self.max_look_angle
