"""
载荷配置模块

定义载荷配置容器类，管理卫星的多种成像模式配置。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import copy

from .imaging_mode import ImagingModeConfig, create_pmc_mode_config
from .pmc_config import PitchMotionCompensationConfig


@dataclass
class PayloadConfiguration:
    """
    载荷配置

    管理卫星载荷的成像模式配置，支持多种成像模式，
    每种模式有独立的分辨率、幅宽、功耗、数据率等参数。

    Attributes:
        payload_type: 载荷类型（"optical" 或 "sar"）
        default_mode: 默认成像模式名称
        modes: 成像模式配置字典 {mode_name: ImagingModeConfig}
        common_fov: 共享的FOV配置（可选）
        payload_id: 载荷标识符（可选）
        description: 载荷描述（可选）
    """
    payload_type: str  # "optical" 或 "sar"
    default_mode: str
    modes: Dict[str, ImagingModeConfig] = field(default_factory=dict)
    common_fov: Optional[Dict[str, Any]] = None
    payload_id: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """验证配置有效性"""
        if self.payload_type not in ('optical', 'sar'):
            raise ValueError(f"payload_type must be 'optical' or 'sar', got '{self.payload_type}'")

        if not self.modes:
            raise ValueError("At least one imaging mode must be defined")

        if self.default_mode not in self.modes:
            # 如果默认模式不存在，使用第一个模式作为默认
            self.default_mode = list(self.modes.keys())[0]

        # 验证所有模式类型与payload_type一致
        for mode_name, mode_config in self.modes.items():
            if mode_config.mode_type != self.payload_type:
                raise ValueError(
                    f"Mode '{mode_name}' has type '{mode_config.mode_type}' "
                    f"but payload type is '{self.payload_type}'"
                )

    def get_mode_config(self, mode: Optional[str] = None) -> ImagingModeConfig:
        """
        获取指定成像模式的配置

        Args:
            mode: 成像模式名称，None则使用默认模式

        Returns:
            ImagingModeConfig

        Raises:
            ValueError: 如果模式不存在
        """
        mode_name = mode if mode is not None else self.default_mode
        if mode_name not in self.modes:
            raise ValueError(f"Imaging mode '{mode_name}' not found in payload configuration. "
                           f"Available modes: {list(self.modes.keys())}")
        return self.modes[mode_name]

    def get_mode_names(self) -> List[str]:
        """获取所有可用的成像模式名称"""
        return list(self.modes.keys())

    def has_mode(self, mode: str) -> bool:
        """检查是否支持指定成像模式"""
        return mode in self.modes

    def add_mode(self, mode_name: str, config: ImagingModeConfig) -> None:
        """
        添加新的成像模式

        Args:
            mode_name: 模式名称
            config: 模式配置

        Raises:
            ValueError: 如果模式已存在或类型不匹配
        """
        if mode_name in self.modes:
            raise ValueError(f"Imaging mode '{mode_name}' already exists")

        if config.mode_type != self.payload_type:
            raise ValueError(
                f"Mode type '{config.mode_type}' does not match payload type '{self.payload_type}'"
            )

        self.modes[mode_name] = config

    def remove_mode(self, mode_name: str) -> None:
        """
        移除成像模式

        Args:
            mode_name: 模式名称

        Raises:
            ValueError: 如果模式不存在或是唯一模式
        """
        if mode_name not in self.modes:
            raise ValueError(f"Imaging mode '{mode_name}' not found")

        if len(self.modes) <= 1:
            raise ValueError("Cannot remove the only imaging mode")

        del self.modes[mode_name]

        # 如果删除的是默认模式，更新默认模式
        if self.default_mode == mode_name:
            self.default_mode = list(self.modes.keys())[0]

    def get_resolution(self, mode: Optional[str] = None) -> float:
        """获取指定模式的分辨率（米）"""
        return self.get_mode_config(mode).resolution_m

    def get_swath_width(self, mode: Optional[str] = None) -> float:
        """获取指定模式的幅宽（米）"""
        return self.get_mode_config(mode).swath_width_m

    def get_power_consumption(self, mode: Optional[str] = None) -> float:
        """获取指定模式的功耗（瓦特）"""
        return self.get_mode_config(mode).power_consumption_w

    def get_data_rate(self, mode: Optional[str] = None) -> float:
        """获取指定模式的数据率（Mbps）"""
        return self.get_mode_config(mode).data_rate_mbps

    def get_min_duration(self, mode: Optional[str] = None) -> float:
        """获取指定模式的最短成像时间（秒）"""
        return self.get_mode_config(mode).min_duration_s

    def get_max_duration(self, mode: Optional[str] = None) -> float:
        """获取指定模式的最长成像时间（秒）"""
        return self.get_mode_config(mode).max_duration_s

    def get_fov_config(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定模式的FOV配置

        如果模式有独立FOV配置则使用，否则使用common_fov
        """
        mode_config = self.get_mode_config(mode)
        if mode_config.fov_config:
            return mode_config.fov_config
        return self.common_fov or {}

    def get_best_resolution_mode(self) -> str:
        """
        获取分辨率最高的成像模式名称

        Returns:
            模式名称
        """
        return min(self.modes.keys(), key=lambda m: self.modes[m].resolution_m)

    def get_best_swath_mode(self) -> str:
        """
        获取幅宽最大的成像模式名称

        Returns:
            模式名称
        """
        return max(self.modes.keys(), key=lambda m: self.modes[m].swath_width_m)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'payload_type': self.payload_type,
            'default_mode': self.default_mode,
            'modes': {
                name: config.to_dict()
                for name, config in self.modes.items()
            },
            'common_fov': self.common_fov,
            'payload_id': self.payload_id,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PayloadConfiguration':
        """从字典创建"""
        modes_data = data.get('modes', {})
        modes = {
            name: ImagingModeConfig.from_dict(config)
            for name, config in modes_data.items()
        }

        return cls(
            payload_type=data['payload_type'],
            default_mode=data.get('default_mode', list(modes.keys())[0] if modes else ''),
            modes=modes,
            common_fov=data.get('common_fov'),
            payload_id=data.get('payload_id'),
            description=data.get('description'),
        )

    def validate(self) -> bool:
        """
        验证配置完整性

        Returns:
            True if valid

        Raises:
            ValueError: 如果配置无效
        """
        if not self.modes:
            raise ValueError("No imaging modes defined")

        if self.default_mode not in self.modes:
            raise ValueError(f"Default mode '{self.default_mode}' not found in modes")

        for mode_name, mode_config in self.modes.items():
            if mode_config.mode_type != self.payload_type:
                raise ValueError(
                    f"Mode '{mode_name}' type mismatch: "
                    f"expected '{self.payload_type}', got '{mode_config.mode_type}'"
                )

        return True

    def copy(self) -> 'PayloadConfiguration':
        """创建深度拷贝"""
        return copy.deepcopy(self)

    def get_pmc_modes(self) -> List[str]:
        """
        获取所有PMC模式名称

        Returns:
            PMC模式名称列表
        """
        return [name for name, config in self.modes.items() if config.is_pmc_mode()]

    def has_pmc_mode(self) -> bool:
        """检查是否有PMC模式"""
        return any(config.is_pmc_mode() for config in self.modes.values())

    def get_pmc_config(self, mode: Optional[str] = None) -> Optional[PitchMotionCompensationConfig]:
        """
        获取指定模式的PMC配置

        Args:
            mode: 模式名称，None则检查默认模式

        Returns:
            PitchMotionCompensationConfig 或 None（如果不是PMC模式）
        """
        mode_config = self.get_mode_config(mode)
        if not mode_config.is_pmc_mode():
            return None

        pmc_params = mode_config.get_pmc_params()
        return PitchMotionCompensationConfig(
            speed_reduction_ratio=pmc_params.get('speed_reduction_ratio', 0.25),
            pitch_rate_dps=pmc_params.get('pitch_rate_dps'),
            min_altitude_m=pmc_params.get('min_altitude_m', 400000.0),
            max_roll_angle_deg=pmc_params.get('max_roll_angle_deg', 30.0),
        )

    def add_pmc_mode(
        self,
        mode_name: str,
        speed_reduction_ratio: float,
        base_resolution_m: float = 0.5,
        base_swath_width_m: float = 15000.0,
        **kwargs
    ) -> None:
        """
        添加PMC模式

        Args:
            mode_name: 模式名称
            speed_reduction_ratio: 降速比（0.1-0.75）
            base_resolution_m: 基础分辨率
            base_swath_width_m: 基础幅宽
            **kwargs: 其他参数
        """
        pmc_config = create_pmc_mode_config(
            base_resolution_m=base_resolution_m,
            base_swath_width_m=base_swath_width_m,
            speed_reduction_ratio=speed_reduction_ratio,
            mode_type=self.payload_type,
            **kwargs
        )
        self.add_mode(mode_name, pmc_config)


# 预定义的载荷配置模板

def create_optical_payload_config(
    resolution_m: float = 0.5,
    swath_width_m: float = 15000,
    power_consumption_w: float = 150.0,
    data_rate_mbps: float = 200.0,
    min_duration_s: float = 6.0,
    max_duration_s: float = 12.0,
    spectral_bands: Optional[List[str]] = None,
) -> PayloadConfiguration:
    """
    创建光学载荷配置（被动推扫模式）

    Args:
        resolution_m: 分辨率（米）
        swath_width_m: 幅宽（米）
        power_consumption_w: 功耗（瓦特）
        data_rate_mbps: 数据率（Mbps）
        min_duration_s: 最短成像时间（秒）
        max_duration_s: 最长成像时间（秒）
        spectral_bands: 光谱波段列表

    Returns:
        PayloadConfiguration
    """
    return PayloadConfiguration(
        payload_type='optical',
        default_mode='push_broom',
        modes={
            'push_broom': ImagingModeConfig(
                resolution_m=resolution_m,
                swath_width_m=swath_width_m,
                power_consumption_w=power_consumption_w,
                data_rate_mbps=data_rate_mbps,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
                mode_type='optical',
                fov_config={
                    'cross_track_fov_deg': 2.5,
                    'along_track_fov_deg': 0.5,
                },
                characteristics={
                    'spectral_bands': spectral_bands or ['PAN', 'RGB', 'NIR'],
                }
            )
        },
        description=f'光学载荷，分辨率{resolution_m}m，幅宽{swath_width_m}m'
    )


def create_sar_payload_config(
    modes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> PayloadConfiguration:
    """
    创建SAR载荷配置

    Args:
        modes: 模式配置字典，格式为 {mode_name: config_dict}
               如果为None，使用默认的条带/聚束/扫描模式

    Returns:
        PayloadConfiguration
    """
    if modes is None:
        # 使用默认模式
        from .imaging_mode import (
            SAR_STRIPMAP_MODE,
            SAR_SPOTLIGHT_MODE,
            SAR_SCAN_MODE,
            SAR_SLIDING_SPOTLIGHT_MODE,
        )
        modes = {
            'stripmap': SAR_STRIPMAP_MODE,
            'spotlight': SAR_SPOTLIGHT_MODE,
            'scan': SAR_SCAN_MODE,
            'sliding_spotlight': SAR_SLIDING_SPOTLIGHT_MODE,
        }
    else:
        # 从字典创建模式配置
        modes = {
            name: ImagingModeConfig.from_dict(config)
            for name, config in modes.items()
        }

    return PayloadConfiguration(
        payload_type='sar',
        default_mode='stripmap',
        modes=modes,
        description='SAR载荷，支持条带/聚束/扫描/滑动聚束模式'
    )
