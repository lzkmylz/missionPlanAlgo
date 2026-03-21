"""
成像模式配置模块

定义标准化的成像模式配置类，支持不同成像模式有不同的分辨率、
幅宽、功耗、数据率等参数。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import math


class ImagingModeType(Enum):
    """成像模式类型"""
    OPTICAL = "optical"
    SAR = "sar"


class ImagingMode(Enum):
    """
    具体成像模式枚举

    定义卫星支持的各种成像工作模式。
    """
    # 光学被动推扫模式
    PUSH_BROOM = "push_broom"
    FRAME = "frame"

    # 主动前向推扫模式（Pitch Motion Compensation）
    FORWARD_PUSHBROOM_PMC = "forward_pushbroom_pmc"

    # 主动反向推扫模式（Reverse Pitch Motion Compensation）
    REVERSE_PUSHBROOM_PMC = "reverse_pushbroom_pmc"

    # SAR模式
    STRIPMAP = "stripmap"
    SPOTLIGHT = "spotlight"
    SCAN = "scan"
    SLIDING_SPOTLIGHT = "sliding_spotlight"
    TOPSAR = "topsar"


@dataclass
class ImagingModeConfig:
    """
    成像模式配置

    定义特定成像模式的完整参数集，包括分辨率、幅宽、功耗、数据率等。

    Attributes:
        resolution_m: 分辨率（米）
        swath_width_m: 幅宽（米）
        fov_config: 视场配置（FOV参数）
        power_consumption_w: 功耗（瓦特）
        data_rate_mbps: 数据率（Mbps）
        min_duration_s: 最短成像时间（秒）
        max_duration_s: 最长成像时间（秒）
        mode_type: 模式类型（optical/sar）
        characteristics: 模式特定参数字典
    """
    resolution_m: float
    swath_width_m: float
    power_consumption_w: float
    data_rate_mbps: float
    min_duration_s: float
    max_duration_s: float
    mode_type: str = "optical"  # "optical" 或 "sar"
    fov_config: Dict[str, Any] = field(default_factory=dict)
    characteristics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证参数有效性"""
        if self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")
        if self.swath_width_m <= 0:
            raise ValueError(f"swath_width_m must be positive, got {self.swath_width_m}")
        if self.power_consumption_w < 0:
            raise ValueError(f"power_consumption_w must be non-negative, got {self.power_consumption_w}")
        if self.data_rate_mbps < 0:
            raise ValueError(f"data_rate_mbps must be non-negative, got {self.data_rate_mbps}")
        if self.min_duration_s <= 0:
            raise ValueError(f"min_duration_s must be positive, got {self.min_duration_s}")
        if self.max_duration_s < self.min_duration_s:
            raise ValueError(
                f"max_duration_s ({self.max_duration_s}) must be >= min_duration_s ({self.min_duration_s})"
            )

        # 验证PMC模式参数
        self._validate_pmc_params()

    def _validate_pmc_params(self):
        """验证PMC模式参数"""
        chars = self.characteristics
        if chars.get('motion_compensation', False):
            # 验证速度变化比
            reduction = chars.get('speed_reduction_ratio')
            if reduction is not None:
                if not 0.1 <= reduction <= 0.75:
                    raise ValueError(
                        f"PMC speed_reduction_ratio must be in [0.1, 0.75], got {reduction}"
                    )
            # 验证俯仰角速度
            pitch_rate = chars.get('pitch_rate_dps')
            direction = chars.get('direction', 'forward')
            if pitch_rate is not None:
                # 前向模式：俯仰角速度为正（相机后摆）
                # 反向模式：俯仰角速度为负（相机前摆）
                if direction == 'forward' and pitch_rate < 0:
                    raise ValueError(f"Forward PMC pitch_rate_dps must be non-negative, got {pitch_rate}")
                if direction == 'reverse' and pitch_rate > 0:
                    raise ValueError(f"Reverse PMC pitch_rate_dps must be non-positive, got {pitch_rate}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'resolution_m': self.resolution_m,
            'swath_width_m': self.swath_width_m,
            'power_consumption_w': self.power_consumption_w,
            'data_rate_mbps': self.data_rate_mbps,
            'min_duration_s': self.min_duration_s,
            'max_duration_s': self.max_duration_s,
            'mode_type': self.mode_type,
            'fov_config': self.fov_config,
            'characteristics': self.characteristics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImagingModeConfig':
        """从字典创建"""
        return cls(
            resolution_m=float(data['resolution_m']),
            swath_width_m=float(data['swath_width_m']),
            power_consumption_w=float(data.get('power_consumption_w', 150.0)),
            data_rate_mbps=float(data.get('data_rate_mbps', 200.0)),
            min_duration_s=float(data.get('min_duration_s', 5.0)),
            max_duration_s=float(data.get('max_duration_s', 15.0)),
            mode_type=data.get('mode_type', 'optical'),
            fov_config=data.get('fov_config', {}),
            characteristics=data.get('characteristics', {}),
        )

    def get_coverage_area_km2(self, duration_s: float, satellite_velocity_m_s: float = 7500.0) -> float:
        """
        计算给定时长的覆盖面积

        Args:
            duration_s: 成像时长（秒）
            satellite_velocity_m_s: 卫星速度（米/秒），默认7500

        Returns:
            覆盖面积（平方公里）
        """
        along_track_distance_km = (satellite_velocity_m_s * duration_s) / 1000.0
        swath_width_km = self.swath_width_m / 1000.0
        return along_track_distance_km * swath_width_km

    def get_data_volume_gb(self, duration_s: float) -> float:
        """
        计算给定时长的数据量

        Args:
            duration_s: 成像时长（秒）

        Returns:
            数据量（GB）
        """
        # data_rate_mbps -> GB = Mbps * 秒 / 8000
        return (self.data_rate_mbps * duration_s) / 8000.0

    def get_energy_consumption_wh(self, duration_s: float) -> float:
        """
        计算给定时长的能耗

        Args:
            duration_s: 成像时长（秒）

        Returns:
            能耗（Wh）
        """
        # W * s / 3600 = Wh
        return (self.power_consumption_w * duration_s) / 3600.0

    def is_pmc_mode(self) -> bool:
        """检查是否为PMC模式"""
        return self.characteristics.get('motion_compensation', False)

    def get_pmc_params(self) -> Dict[str, Any]:
        """
        获取PMC模式参数

        Returns:
            PMC参数字典，非PMC模式返回空字典
        """
        if not self.is_pmc_mode():
            return {}
        return {
            'speed_reduction_ratio': self.characteristics.get('speed_reduction_ratio', 0.25),
            'pitch_rate_dps': self.characteristics.get('pitch_rate_dps'),
            'direction': self.characteristics.get('direction', 'forward'),
            'min_altitude_m': self.characteristics.get('min_altitude_m', 400000),
            'max_roll_angle_deg': self.characteristics.get('max_roll_angle_deg', 30.0),
            'integration_time_gain': self.characteristics.get('integration_time_gain', 1.33),
        }

    def get_effective_integration_time(self, duration_s: float) -> float:
        """
        计算有效积分时间（考虑PMC增益）

        前向和反向PMC都是降速成像，积分时间延长:
        t_effective = t_physical / (1 - R)

        Args:
            duration_s: 物理成像时长（秒）

        Returns:
            等效积分时间（秒）
        """
        if not self.is_pmc_mode():
            return duration_s

        reduction_ratio = self.characteristics.get('speed_reduction_ratio', 0.25)

        # 前向和反向都是降速成像，积分时间延长
        # 公式: t_effective = t_physical / (1 - reduction_ratio)
        return duration_s / max(0.1, (1 - reduction_ratio))


# 预定义的成像模式配置模板

# 光学 - 被动推扫模式（高分辨率）
OPTICAL_PUSH_BROOM_HIGH_RES = ImagingModeConfig(
    resolution_m=0.5,
    swath_width_m=15000,
    power_consumption_w=150.0,
    data_rate_mbps=200.0,
    min_duration_s=6.0,
    max_duration_s=12.0,
    mode_type="optical",
    fov_config={
        'cross_track_fov_deg': 2.5,
        'along_track_fov_deg': 0.5,
    },
    characteristics={
        'spectral_bands': ['PAN', 'RGB', 'NIR'],
        'description': '高分辨率被动推扫成像',
    }
)

# 光学 - 被动推扫模式（中分辨率）
OPTICAL_PUSH_BROOM_MEDIUM_RES = ImagingModeConfig(
    resolution_m=2.0,
    swath_width_m=30000,
    power_consumption_w=120.0,
    data_rate_mbps=150.0,
    min_duration_s=5.0,
    max_duration_s=15.0,
    mode_type="optical",
    fov_config={
        'cross_track_fov_deg': 5.0,
        'along_track_fov_deg': 1.0,
    },
    characteristics={
        'spectral_bands': ['RGB', 'NIR'],
        'description': '中分辨率被动推扫成像',
    }
)

# SAR - 条带模式
SAR_STRIPMAP_MODE = ImagingModeConfig(
    resolution_m=3.0,
    swath_width_m=30000,
    power_consumption_w=300.0,
    data_rate_mbps=400.0,
    min_duration_s=5.0,
    max_duration_s=15.0,
    mode_type="sar",
    fov_config={
        'range_half_angle_deg': 3.0,
        'azimuth_half_angle_deg': 1.0,
    },
    characteristics={
        'polarization': 'HH+HV',
        'incidence_angle_range': [20, 45],
        'description': '条带模式成像',
    }
)

# SAR - 聚束模式（高分辨率）
SAR_SPOTLIGHT_MODE = ImagingModeConfig(
    resolution_m=1.0,
    swath_width_m=10000,
    power_consumption_w=500.0,
    data_rate_mbps=800.0,
    min_duration_s=10.0,
    max_duration_s=25.0,
    mode_type="sar",
    fov_config={
        'range_half_angle_deg': 1.0,
        'azimuth_half_angle_deg': 0.5,
    },
    characteristics={
        'polarization': 'HH+HV+VV+VH',
        'incidence_angle_range': [25, 50],
        'description': '聚束模式高分辨率成像',
    }
)

# SAR - 扫描模式（宽幅）
SAR_SCAN_MODE = ImagingModeConfig(
    resolution_m=10.0,
    swath_width_m=100000,
    power_consumption_w=400.0,
    data_rate_mbps=600.0,
    min_duration_s=8.0,
    max_duration_s=20.0,
    mode_type="sar",
    fov_config={
        'range_half_angle_deg': 10.0,
        'azimuth_half_angle_deg': 1.5,
    },
    characteristics={
        'polarization': 'HH+HV',
        'incidence_angle_range': [15, 55],
        'description': '扫描模式宽幅成像',
    }
)

# SAR - 滑动聚束模式
SAR_SLIDING_SPOTLIGHT_MODE = ImagingModeConfig(
    resolution_m=1.5,
    swath_width_m=20000,
    power_consumption_w=450.0,
    data_rate_mbps=700.0,
    min_duration_s=10.0,
    max_duration_s=30.0,
    mode_type="sar",
    fov_config={
        'range_half_angle_deg': 2.0,
        'azimuth_half_angle_deg': 0.8,
    },
    characteristics={
        'polarization': 'HH+HV+VV',
        'incidence_angle_range': [22, 48],
        'description': '滑动聚束模式成像',
    }
)

# SAR - TOPSAR模式（宽幅扫描）
SAR_TOPSAR_MODE = ImagingModeConfig(
    resolution_m=5.0,
    swath_width_m=100000,   # 总幅宽100km（3子条带×约33km）
    power_consumption_w=500.0,
    data_rate_mbps=600.0,
    min_duration_s=5.0,     # 至少完成一个突发循环
    max_duration_s=60.0,
    mode_type="sar",
    fov_config={
        'range_half_angle_deg': 7.27,   # 距离向大视场（与sar_2.json一致）
        'azimuth_half_angle_deg': 0.5,
    },
    characteristics={
        'polarization': 'VV+VH',
        'incidence_angle_range': [20, 50],
        'num_subswaths': 3,
        'description': 'TOPSAR宽幅扫描模式，多子条带电子扫描，覆盖宽',
    }
)

# 光学 - 主动前向推扫模式（PMC 25%降速）
OPTICAL_PMC_25PERCENT = ImagingModeConfig(
    resolution_m=0.5,
    swath_width_m=15000,
    power_consumption_w=180.0,  # PMC模式功耗略高（姿态机动）
    data_rate_mbps=200.0,
    min_duration_s=8.0,  # PMC需要更长最小时间
    max_duration_s=30.0,  # 可配置的最大时长
    mode_type="optical",
    fov_config={
        'cross_track_fov_deg': 2.5,
        'along_track_fov_deg': 0.5,
    },
    characteristics={
        'spectral_bands': ['PAN', 'RGB', 'NIR'],
        'description': '主动前向推扫模式，25%降速比，提高SNR',
        'motion_compensation': True,
        'speed_reduction_ratio': 0.25,
        'pitch_rate_dps': 0.35,  # 典型值，实际根据轨道高度计算
        'min_altitude_m': 400000,
        'max_roll_angle_deg': 30.0,
        'integration_time_gain': 1.33,  # 1/(1-0.25)
    }
)

# 光学 - 主动前向推扫模式（PMC 50%降速）
OPTICAL_PMC_50PERCENT = ImagingModeConfig(
    resolution_m=0.5,
    swath_width_m=15000,
    power_consumption_w=200.0,  # 更高降速比需要更多机动
    data_rate_mbps=200.0,
    min_duration_s=10.0,
    max_duration_s=30.0,
    mode_type="optical",
    fov_config={
        'cross_track_fov_deg': 2.5,
        'along_track_fov_deg': 0.5,
    },
    characteristics={
        'spectral_bands': ['PAN', 'RGB', 'NIR'],
        'description': '主动前向推扫模式，50%降速比，显著提高SNR',
        'motion_compensation': True,
        'speed_reduction_ratio': 0.50,
        'pitch_rate_dps': 0.70,
        'min_altitude_m': 400000,
        'max_roll_angle_deg': 25.0,  # 更高降速比限制滚转角
        'integration_time_gain': 2.0,  # 1/(1-0.5)
    }
)

# 光学 - 主动反向推扫模式（Reverse PMC 25%降速）
# 说明：反向推扫也是降速成像，只是推扫方向相反
# 初始时刻：相机处于大后俯仰角（如+25°），指向后方远处目标
# 成像过程：向前俯仰机动（角度减小），相机光轴相对地面向后推扫
OPTICAL_REVERSE_PMC_25PERCENT = ImagingModeConfig(
    resolution_m=0.6,  # 反向推扫分辨率略低于正向（推扫方向不同导致）
    swath_width_m=15000,
    power_consumption_w=170.0,
    data_rate_mbps=200.0,
    min_duration_s=5.0,
    max_duration_s=20.0,
    mode_type="optical",
    fov_config={
        'cross_track_fov_deg': 2.5,
        'along_track_fov_deg': 0.5,
    },
    characteristics={
        'spectral_bands': ['PAN', 'RGB', 'NIR'],
        'description': '主动反向推扫模式，25%降速，从后向前推扫成像',
        'motion_compensation': True,
        'direction': 'reverse',  # 反向推扫
        'speed_reduction_ratio': 0.25,  # 降速比
        'pitch_rate_dps': -0.35,  # 负值表示向前俯仰（角度减小）
        'min_altitude_m': 400000,
        'max_roll_angle_deg': 30.0,
        'integration_time_gain': 1.33,  # 1/(1-0.25)，积分时间延长
    }
)

# 光学 - 主动反向推扫模式（Reverse PMC 50%降速）
OPTICAL_REVERSE_PMC_50PERCENT = ImagingModeConfig(
    resolution_m=0.8,  # 更高降速比，分辨率提高但幅宽可能受影响
    swath_width_m=12000,
    power_consumption_w=190.0,
    data_rate_mbps=200.0,
    min_duration_s=4.0,
    max_duration_s=15.0,
    mode_type="optical",
    fov_config={
        'cross_track_fov_deg': 2.5,
        'along_track_fov_deg': 0.5,
    },
    characteristics={
        'spectral_bands': ['PAN', 'RGB', 'NIR'],
        'description': '主动反向推扫模式，50%降速，从后向前推扫成像',
        'motion_compensation': True,
        'direction': 'reverse',
        'speed_reduction_ratio': 0.50,
        'pitch_rate_dps': -0.70,  # 负值表示向前俯仰
        'min_altitude_m': 400000,
        'max_roll_angle_deg': 25.0,
        'integration_time_gain': 2.0,  # 1/(1-0.5)，积分时间延长
    }
)

# SAR - 主动前向推扫模式（PMC 25%降速）
SAR_PMC_25PERCENT = ImagingModeConfig(
    resolution_m=3.0,
    swath_width_m=30000,
    power_consumption_w=350.0,
    data_rate_mbps=400.0,
    min_duration_s=8.0,
    max_duration_s=30.0,
    mode_type="sar",
    fov_config={
        'range_half_angle_deg': 3.0,
        'azimuth_half_angle_deg': 1.0,
    },
    characteristics={
        'polarization': 'HH+HV',
        'incidence_angle_range': [20, 45],
        'description': 'SAR主动前向推扫模式，25%降速比',
        'motion_compensation': True,
        'speed_reduction_ratio': 0.25,
        'pitch_rate_dps': 0.35,
        'min_altitude_m': 500000,
        'max_roll_angle_deg': 35.0,
        'integration_time_gain': 1.33,
    }
)

# 模式模板映射（用于快速查找）
MODE_TEMPLATES = {
    'optical_push_broom_high': OPTICAL_PUSH_BROOM_HIGH_RES,
    'optical_push_broom_medium': OPTICAL_PUSH_BROOM_MEDIUM_RES,
    'optical_pmc_25percent': OPTICAL_PMC_25PERCENT,
    'optical_pmc_50percent': OPTICAL_PMC_50PERCENT,
    'optical_reverse_pmc_25percent': OPTICAL_REVERSE_PMC_25PERCENT,
    'optical_reverse_pmc_50percent': OPTICAL_REVERSE_PMC_50PERCENT,
    'sar_stripmap': SAR_STRIPMAP_MODE,
    'sar_spotlight': SAR_SPOTLIGHT_MODE,
    'sar_scan': SAR_SCAN_MODE,
    'sar_sliding_spotlight': SAR_SLIDING_SPOTLIGHT_MODE,
    'sar_topsar': SAR_TOPSAR_MODE,
    'sar_pmc_25percent': SAR_PMC_25PERCENT,
}


def get_mode_template(template_name: str) -> Optional[ImagingModeConfig]:
    """
    获取预定义的成像模式模板

    Args:
        template_name: 模板名称

    Returns:
        ImagingModeConfig 或 None
    """
    return MODE_TEMPLATES.get(template_name)


def create_pmc_mode_config(
    base_resolution_m: float,
    base_swath_width_m: float,
    speed_reduction_ratio: float,
    mode_type: str = "optical",
    max_duration_s: float = 30.0,
    **kwargs
) -> ImagingModeConfig:
    """
    创建自定义PMC模式配置

    Args:
        base_resolution_m: 基础分辨率（米）
        base_swath_width_m: 基础幅宽（米）
        speed_reduction_ratio: 降速比（0.1-0.75）
        mode_type: 模式类型（"optical"或"sar"）
        max_duration_s: 最大成像时长（秒）
        **kwargs: 其他配置参数

    Returns:
        ImagingModeConfig
    """
    if not 0.1 <= speed_reduction_ratio <= 0.75:
        raise ValueError(f"speed_reduction_ratio must be in [0.1, 0.75]")

    # 计算等效积分时间增益
    integration_gain = 1.0 / (1.0 - speed_reduction_ratio)

    # 基础俯仰角速度估算（度/秒），基于典型500km轨道
    typical_altitude = 500000  # 米
    v_ground = 7500  # m/s，近似地面速度
    pitch_rate = math.degrees(v_ground * speed_reduction_ratio / typical_altitude)

    # PMC模式功耗增加（姿态机动成本）
    base_power = kwargs.get('base_power_w', 150.0 if mode_type == 'optical' else 300.0)
    power_consumption = base_power * (1.0 + 0.2 * speed_reduction_ratio)

    characteristics = {
        'description': f'主动前向推扫模式，{int(speed_reduction_ratio*100)}%降速比',
        'motion_compensation': True,
        'speed_reduction_ratio': speed_reduction_ratio,
        'pitch_rate_dps': pitch_rate,
        'min_altitude_m': kwargs.get('min_altitude_m', 400000),
        'max_roll_angle_deg': kwargs.get('max_roll_angle_deg', 30.0),
        'integration_time_gain': integration_gain,
    }

    # 添加额外的characteristics
    if 'spectral_bands' in kwargs:
        characteristics['spectral_bands'] = kwargs['spectral_bands']
    if 'polarization' in kwargs:
        characteristics['polarization'] = kwargs['polarization']

    return ImagingModeConfig(
        resolution_m=base_resolution_m,
        swath_width_m=base_swath_width_m,
        power_consumption_w=power_consumption,
        data_rate_mbps=kwargs.get('data_rate_mbps', 200.0 if mode_type == 'optical' else 400.0),
        min_duration_s=kwargs.get('min_duration_s', 8.0),
        max_duration_s=max_duration_s,
        mode_type=mode_type,
        fov_config=kwargs.get('fov_config', {}),
        characteristics=characteristics,
    )
