"""
载荷配置模块

定义载荷配置容器类，管理卫星的多种成像模式配置。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import copy

from .imaging_mode import ImagingModeConfig, create_pmc_mode_config
from .pmc_config import PitchMotionCompensationConfig
from .sar_spotlight_config import SARSpotlightConfig
from .sar_sliding_spotlight_config import SARSlidingSpotlightConfig
from .sar_stripmap_config import SARStripmapConfig
from .sar_topsar_config import SARTOPSARConfig
from .sar_scansar_config import SARScanSARConfig


# SAR配置类型默认解析优先级
_DEFAULT_SAR_CONFIG_PRIORITY = ('stripmap_config', 'sliding_spotlight_config', 'spotlight_config', 'topsar_config', 'scansar_config')


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
        sar_spotlight_configs: SAR聚束模式物理参数，key 为模式名
        sar_sliding_spotlight_configs: SAR滑动聚束模式物理参数，key 为模式名
        sar_stripmap_configs: SAR条带模式物理参数，key 为模式名
        sar_topsar_configs: SAR TOPSAR模式物理参数，key 为模式名
        sar_scansar_configs: SAR ScanSAR模式物理参数，key 为模式名

    SAR配置解析优先级:
    -------------------
    当from_dict解析模式配置时，按以下优先级识别SAR物理参数类型：
    1. stripmap_config (条带模式配置)
    2. sliding_spotlight_config (滑动聚束配置)
    3. spotlight_config (聚束模式配置)
    4. topsar_config (TOPSAR模式配置)
    5. scansar_config (ScanSAR模式配置)

    可通过类属性或 from_dict 的 config_priority 参数自定义优先级。
    """
    payload_type: str  # "optical" 或 "sar"
    default_mode: str
    modes: Dict[str, ImagingModeConfig] = field(default_factory=dict)
    common_fov: Optional[Dict[str, Any]] = None
    payload_id: Optional[str] = None
    description: Optional[str] = None
    sar_spotlight_configs: Dict[str, SARSpotlightConfig] = field(default_factory=dict)
    sar_sliding_spotlight_configs: Dict[str, SARSlidingSpotlightConfig] = field(default_factory=dict)
    sar_stripmap_configs: Dict[str, SARStripmapConfig] = field(default_factory=dict)
    sar_topsar_configs: Dict[str, SARTOPSARConfig] = field(default_factory=dict)
    sar_scansar_configs: Dict[str, SARScanSARConfig] = field(default_factory=dict)

    # SAR计算器实例缓存（非持久化，实例级别）
    _sar_calc_cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

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
    def from_dict(
        cls,
        data: Dict[str, Any],
        config_priority: Optional[List[str]] = None
    ) -> 'PayloadConfiguration':
        """
        从字典创建

        Args:
            data: 配置数据字典
            config_priority: SAR配置类型解析优先级列表，覆盖默认值。
                例如: ['spotlight_config', 'sliding_spotlight_config', 'stripmap_config']
                默认优先级: ['stripmap_config', 'sliding_spotlight_config', 'spotlight_config']

        Returns:
            PayloadConfiguration 实例
        """
        modes_data = data.get('modes', {})
        modes = {
            name: ImagingModeConfig.from_dict(config)
            for name, config in modes_data.items()
        }

        # 解析各聚束模式的物理参数配置（spotlight_config）
        sar_spotlight_configs: Dict[str, SARSpotlightConfig] = {}
        # 解析滑动聚束模式的物理参数配置（sliding_spotlight_config）
        sar_sliding_spotlight_configs: Dict[str, SARSlidingSpotlightConfig] = {}
        # 解析条带模式的物理参数配置（stripmap_config）
        sar_stripmap_configs: Dict[str, SARStripmapConfig] = {}
        # 解析TOPSAR模式的物理参数配置（topsar_config）
        sar_topsar_configs: Dict[str, SARTOPSARConfig] = {}
        # 解析ScanSAR模式的物理参数配置（scansar_config）
        sar_scansar_configs: Dict[str, SARScanSARConfig] = {}

        # 配置类型到配置字典和解析函数的映射
        config_mapping = {
            'spotlight_config': (sar_spotlight_configs, SARSpotlightConfig),
            'sliding_spotlight_config': (sar_sliding_spotlight_configs, SARSlidingSpotlightConfig),
            'stripmap_config': (sar_stripmap_configs, SARStripmapConfig),
            'topsar_config': (sar_topsar_configs, SARTOPSARConfig),
            'scansar_config': (sar_scansar_configs, SARScanSARConfig),
        }

        # 使用提供的优先级或默认优先级
        priority = config_priority or list(_DEFAULT_SAR_CONFIG_PRIORITY)

        for mode_name, mode_raw in modes_data.items():
            # 按优先级顺序检查配置类型
            for config_key in priority:
                if config_key in mode_raw and config_key in config_mapping:
                    config_dict, config_class = config_mapping[config_key]
                    try:
                        config_dict[mode_name] = config_class.from_dict(mode_raw[config_key])
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Failed to parse {config_key} for mode '{mode_name}': {e}"
                        )
                    # 找到第一个匹配的配置类型后即停止（优先级顺序）
                    break

        return cls(
            payload_type=data['payload_type'],
            default_mode=data.get('default_mode', list(modes.keys())[0] if modes else ''),
            modes=modes,
            common_fov=data.get('common_fov'),
            payload_id=data.get('payload_id'),
            description=data.get('description'),
            sar_spotlight_configs=sar_spotlight_configs,
            sar_sliding_spotlight_configs=sar_sliding_spotlight_configs,
            sar_stripmap_configs=sar_stripmap_configs,
            sar_topsar_configs=sar_topsar_configs,
            sar_scansar_configs=sar_scansar_configs,
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

    def _get_cached_calculator(self, cache_key: str, factory_fn) -> Any:
        """
        获取缓存的计算器实例，如果不存在则创建并缓存。

        Args:
            cache_key: 缓存键
            factory_fn: 创建计算器实例的工厂函数

        Returns:
            缓存的或新创建的计算器实例
        """
        if cache_key not in self._sar_calc_cache:
            self._sar_calc_cache[cache_key] = factory_fn()
        return self._sar_calc_cache[cache_key]

    def clear_calculator_cache(self) -> None:
        """清除SAR计算器实例缓存"""
        self._sar_calc_cache.clear()

    def get_spotlight_calculator(self, mode: str = 'spotlight', use_cache: bool = True):
        """
        获取指定聚束模式的物理计算器实例。

        Args:
            mode: 模式名称（如 "spotlight"）
            use_cache: 是否使用缓存（默认True）

        Returns:
            SARSpotlightCalculator，若该模式无 spotlight_config 则返回 None
        """
        cfg = self.sar_spotlight_configs.get(mode)
        if cfg is None:
            return None

        from core.dynamics.sar_spotlight_calculator import SARSpotlightCalculator

        if not use_cache:
            return SARSpotlightCalculator(cfg)

        cache_key = f"spotlight:{mode}"
        return self._get_cached_calculator(cache_key, lambda: SARSpotlightCalculator(cfg))

    def has_spotlight_config(self, mode: str = 'spotlight') -> bool:
        """检查指定模式是否配置了聚束物理参数"""
        return mode in self.sar_spotlight_configs

    def get_sliding_spotlight_calculator(self, mode: str = 'sliding_spotlight', use_cache: bool = True):
        """
        获取指定滑动聚束模式的物理计算器实例。

        Args:
            mode: 模式名称（如 "sliding_spotlight"）
            use_cache: 是否使用缓存（默认True）

        Returns:
            SARSlidingSpotlightCalculator，若该模式无 sliding_spotlight_config 则返回 None
        """
        cfg = self.sar_sliding_spotlight_configs.get(mode)
        if cfg is None:
            return None

        from core.dynamics.sar_sliding_spotlight_calculator import SARSlidingSpotlightCalculator

        if not use_cache:
            return SARSlidingSpotlightCalculator(cfg)

        cache_key = f"sliding_spotlight:{mode}"
        return self._get_cached_calculator(cache_key, lambda: SARSlidingSpotlightCalculator(cfg))

    def has_sliding_spotlight_config(self, mode: str = 'sliding_spotlight') -> bool:
        """检查指定模式是否配置了滑动聚束物理参数"""
        return mode in self.sar_sliding_spotlight_configs

    def get_stripmap_calculator(self, mode: str = 'stripmap', use_cache: bool = True):
        """
        获取指定条带模式的物理计算器实例。

        Args:
            mode: 模式名称（如 "stripmap"）
            use_cache: 是否使用缓存（默认True）

        Returns:
            SARStripmapCalculator，若该模式无 stripmap_config 则返回 None
        """
        cfg = self.sar_stripmap_configs.get(mode)
        if cfg is None:
            return None

        from core.dynamics.sar_stripmap_calculator import SARStripmapCalculator

        if not use_cache:
            return SARStripmapCalculator(cfg)

        cache_key = f"stripmap:{mode}"
        return self._get_cached_calculator(cache_key, lambda: SARStripmapCalculator(cfg))

    def has_stripmap_config(self, mode: str = 'stripmap') -> bool:
        """检查指定模式是否配置了条带模式物理参数"""
        return mode in self.sar_stripmap_configs

    def get_topsar_calculator(self, mode: str = 'topsar', use_cache: bool = True):
        """
        获取指定TOPSAR模式的物理计算器实例。

        Args:
            mode: 模式名称（如 \"topsar\"）
            use_cache: 是否使用缓存（默认True）

        Returns:
            SARTOPSARCalculator，若该模式无 topsar_config 则返回 None
        """
        cfg = self.sar_topsar_configs.get(mode)
        if cfg is None:
            return None

        from core.dynamics.sar_topsar_calculator import SARTOPSARCalculator

        if not use_cache:
            return SARTOPSARCalculator(cfg)

        cache_key = f"topsar:{mode}"
        return self._get_cached_calculator(cache_key, lambda: SARTOPSARCalculator(cfg))

    def has_topsar_config(self, mode: str = 'topsar') -> bool:
        """检查指定模式是否配置了TOPSAR物理参数"""
        return mode in self.sar_topsar_configs

    def get_scansar_calculator(self, mode: str = 'scansar', use_cache: bool = True):
        """
        获取指定ScanSAR模式的物理计算器实例。

        Args:
            mode: 模式名称（如 "scansar"）
            use_cache: 是否使用缓存（默认True）

        Returns:
            SARScanSARCalculator，若该模式无 scansar_config 则返回 None
        """
        cfg = self.sar_scansar_configs.get(mode)
        if cfg is None:
            return None

        from core.dynamics.sar_scansar_calculator import SARScanSARCalculator

        if not use_cache:
            return SARScanSARCalculator(cfg)

        cache_key = f"scansar:{mode}"
        return self._get_cached_calculator(cache_key, lambda: SARScanSARCalculator(cfg))

    def has_scansar_config(self, mode: str = 'scansar') -> bool:
        """检查指定模式是否配置了ScanSAR物理参数"""
        return mode in self.sar_scansar_configs

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
            direction=pmc_params.get('direction', 'forward'),
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
