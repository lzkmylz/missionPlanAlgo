"""
成像模式配置模块

定义标准化的成像模式配置类，支持不同成像模式有不同的分辨率、
幅宽、功耗、数据率等参数。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ImagingModeType(Enum):
    """成像模式类型"""
    OPTICAL = "optical"
    SAR = "sar"


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

# 模式模板映射（用于快速查找）
MODE_TEMPLATES = {
    'optical_push_broom_high': OPTICAL_PUSH_BROOM_HIGH_RES,
    'optical_push_broom_medium': OPTICAL_PUSH_BROOM_MEDIUM_RES,
    'sar_stripmap': SAR_STRIPMAP_MODE,
    'sar_spotlight': SAR_SPOTLIGHT_MODE,
    'sar_scan': SAR_SCAN_MODE,
    'sar_sliding_spotlight': SAR_SLIDING_SPOTLIGHT_MODE,
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
