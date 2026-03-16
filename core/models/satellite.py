"""
卫星模型 - 定义异构卫星的属性和能力

支持四种卫星类型：
- 光学1型：基础光学成像
- 光学2型：增强光学成像
- SAR-1型：支持聚束/滑动聚束/条带模式
- SAR-2型：增强SAR成像
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import math
import warnings

from core.constants import (
    EARTH_RADIUS_M,
    EARTH_GM,
    DEFAULT_ORBIT_ALTITUDE_M,
    DEFAULT_ORBIT_INCLINATION_DEG,
    DEFAULT_ORBIT_ECCENTRICITY,
    DEFAULT_RAAN_DEG,
    DEFAULT_ARG_OF_PERIGEE_DEG,
    DEFAULT_MEAN_ANOMALY_DEG,
    DEFAULT_STORAGE_CAPACITY_GB,
    DEFAULT_POWER_CAPACITY_WH,
    DEFAULT_DATA_RATE_MBPS,
    DEFAULT_RESOLUTION_M,
    DEFAULT_SWATH_WIDTH_M,
    DEFAULT_MAX_ROLL_ANGLE_DEG,
    DEFAULT_MAX_PITCH_ANGLE_DEG,
    OPTICAL_MAX_ROLL_ANGLE_DEG,
    OPTICAL_MAX_PITCH_ANGLE_DEG,
    SAR_MAX_ROLL_ANGLE_DEG,
    SAR_MAX_PITCH_ANGLE_DEG,
    OPTICAL_1_POWER_CAPACITY_WH,
    OPTICAL_1_STORAGE_CAPACITY_GB,
    OPTICAL_2_POWER_CAPACITY_WH,
    OPTICAL_2_STORAGE_CAPACITY_GB,
    SAR_1_POWER_CAPACITY_WH,
    SAR_1_STORAGE_CAPACITY_GB,
    SAR_2_POWER_CAPACITY_WH,
    SAR_2_STORAGE_CAPACITY_GB,
    OPTICAL_IMAGING_MIN_DURATION_S,
    OPTICAL_IMAGING_MAX_DURATION_S,
    SAR_1_SPOTLIGHT_MIN_DURATION_S,
    SAR_1_SPOTLIGHT_MAX_DURATION_S,
    SAR_1_SLIDING_SPOTLIGHT_MIN_DURATION_S,
    SAR_1_SLIDING_SPOTLIGHT_MAX_DURATION_S,
    SAR_1_STRIPMAP_MIN_DURATION_S,
    SAR_1_STRIPMAP_MAX_DURATION_S,
    SAR_2_SPOTLIGHT_MIN_DURATION_S,
    SAR_2_SPOTLIGHT_MAX_DURATION_S,
    SAR_2_SLIDING_SPOTLIGHT_MIN_DURATION_S,
    SAR_2_SLIDING_SPOTLIGHT_MAX_DURATION_S,
    SAR_2_STRIPMAP_MIN_DURATION_S,
    SAR_2_STRIPMAP_MAX_DURATION_S,
    REFERENCE_EPOCH,
    J2000_JULIAN_DATE,
    GMST_CONSTANT_1,
    GMST_CONSTANT_2,
    METERS_TO_KM,
    KM_TO_METERS,
    # 分轴角速度/加速度限制
    DEFAULT_MAX_ROLL_RATE_DEG_S,
    DEFAULT_MAX_PITCH_RATE_DEG_S,
    DEFAULT_MAX_ROLL_ACCEL_DEG_S2,
    DEFAULT_MAX_PITCH_ACCEL_DEG_S2,
    OPTICAL_MAX_ROLL_RATE_DEG_S,
    OPTICAL_MAX_PITCH_RATE_DEG_S,
    OPTICAL_MAX_ROLL_ACCEL_DEG_S2,
    OPTICAL_MAX_PITCH_ACCEL_DEG_S2,
    SAR_MAX_ROLL_RATE_DEG_S,
    SAR_MAX_PITCH_RATE_DEG_S,
    SAR_MAX_ROLL_ACCEL_DEG_S2,
    SAR_MAX_PITCH_ACCEL_DEG_S2,
    # 成像模式特定功耗和数据率
    OPTICAL_PUSH_BROOM_POWER_W,
    OPTICAL_PUSH_BROOM_DATA_RATE_MBPS,
    SAR_STRIPMAP_POWER_W,
    SAR_STRIPMAP_DATA_RATE_MBPS,
    SAR_SPOTLIGHT_POWER_W,
    SAR_SPOTLIGHT_DATA_RATE_MBPS,
    SAR_SCAN_POWER_W,
    SAR_SCAN_DATA_RATE_MBPS,
    SAR_SLIDING_SPOTLIGHT_POWER_W,
    SAR_SLIDING_SPOTLIGHT_DATA_RATE_MBPS,
)


class SatelliteType(Enum):
    """卫星类型枚举"""
    OPTICAL_1 = "optical_1"
    OPTICAL_2 = "optical_2"
    SAR_1 = "sar_1"
    SAR_2 = "sar_2"


class ImagingMode(Enum):
    """成像模式枚举"""
    # SAR模式
    SPOTLIGHT = "spotlight"
    SLIDING_SPOTLIGHT = "sliding_spotlight"
    STRIPMAP = "stripmap"
    SCAN = "scan"
    # 光学模式
    PUSH_BROOM = "push_broom"
    FRAME = "frame"


class OrbitType(Enum):
    """轨道类型"""
    SSO = "SSO"  # 太阳同步轨道
    LEO = "LEO"  # 近地轨道
    MEO = "MEO"  # 中地球轨道
    GEO = "GEO"  # 地球静止轨道


class OrbitSource(Enum):
    """轨道数据来源"""
    ELEMENTS = "elements"  # 轨道六根数
    TLE = "tle"  # TLE两行根数
    SIMPLIFIED = "simplified"  # 简化参数（仅高度和倾角）


# =============================================================================
# 时间处理工具函数
# =============================================================================

def ensure_utc_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """
    确保datetime是UTC时区感知的

    Args:
        dt: 输入datetime（可以是naive或timezone-aware）

    Returns:
        UTC timezone-aware datetime，或None如果输入为None
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_epoch_string(epoch_str: str) -> datetime:
    """
    解析epoch字符串，返回UTC timezone-aware datetime

    支持的格式：
    - ISO 8601带时区: "2024-01-15T12:00:00Z"
    - ISO 8601带微秒: "2024-01-15T12:00:00.123456Z"
    - ISO 8601带偏移: "2024-01-15T20:00:00+08:00"
    - ISO 8601无时区: "2024-01-15T12:00:00"（假设为UTC）
    - 日期only: "2024-01-15"（假设为UTC 00:00:00）

    Args:
        epoch_str: 时间字符串

    Returns:
        UTC timezone-aware datetime

    Raises:
        ValueError: 如果无法解析字符串
    """
    # 尝试多种格式
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f%z',  # 带微秒和时区偏移
        '%Y-%m-%dT%H:%M:%S%z',      # 带时区偏移
        '%Y-%m-%dT%H:%M:%S.%fZ',    # 带微秒和Z
        '%Y-%m-%dT%H:%M:%SZ',       # 带Z
        '%Y-%m-%dT%H:%M:%S.%f',     # 带微秒无TZ
        '%Y-%m-%dT%H:%M:%S',        # 无TZ
        '%Y-%m-%d %H:%M:%S',        # 空格分隔
        '%Y-%m-%d',                 # 日期only
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(epoch_str, fmt)
            # 如果没有时区信息，假设为UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                # 转换为UTC
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            continue

    raise ValueError(f"Cannot parse epoch string: {epoch_str}")


# =============================================================================
# Orbit数据类
# =============================================================================

@dataclass
class Orbit:
    """
    轨道参数

    支持三种配置方式：
    1. 轨道六根数：semi_major_axis, eccentricity, inclination, raan, arg_of_perigee, mean_anomaly
    2. TLE两行根数：tle_line1, tle_line2（历元从TLE自动解析）
    3. 简化参数：altitude, inclination（用于快速配置）

    优先级：TLE > 六根数 > 简化参数

    重要：所有datetime必须是UTC时区感知的（timezone-aware）
    """
    orbit_type: OrbitType = OrbitType.SSO

    # 轨道六根数（经典开普勒轨道元素）
    semi_major_axis: Optional[float] = None  # 半长轴（米），None时从altitude计算
    eccentricity: float = DEFAULT_ORBIT_ECCENTRICITY  # 偏心率（0-1）
    inclination: float = DEFAULT_ORBIT_INCLINATION_DEG  # 轨道倾角（度）
    raan: float = DEFAULT_RAAN_DEG  # 升交点赤经/RAAN（度）
    arg_of_perigee: float = DEFAULT_ARG_OF_PERIGEE_DEG  # 近地点幅角（度）
    mean_anomaly: float = DEFAULT_MEAN_ANOMALY_DEG  # 平近点角（度）

    # ★ 新增：轨道历元时间（用于六根数和简化参数）
    # TLE格式自动解析历元，不需要此字段
    # 必须是UTC timezone-aware datetime
    epoch: Optional[datetime] = None

    # 简化参数（向后兼容）
    altitude: float = DEFAULT_ORBIT_ALTITUDE_M  # 轨道高度（米）

    # TLE两行根数
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    # 轨道数据来源标志
    source: OrbitSource = field(default=OrbitSource.SIMPLIFIED)

    def __post_init__(self):
        """初始化后确定轨道数据来源并处理epoch时区"""
        # ★ 处理epoch时区：强制转换为UTC
        if self.epoch is not None:
            if self.epoch.tzinfo is None:
                # naive datetime假设为UTC并发出警告
                warnings.warn(
                    f"Orbit epoch is naive datetime {self.epoch}, "
                    "assuming UTC. Please provide timezone-aware datetime.",
                    UserWarning,
                    stacklevel=2
                )
                self.epoch = self.epoch.replace(tzinfo=timezone.utc)
            else:
                # 转换为UTC
                self.epoch = self.epoch.astimezone(timezone.utc)

        # 确定轨道数据来源
        """初始化后确定轨道数据来源"""
        if self.tle_line1 and self.tle_line2:
            self.source = OrbitSource.TLE
        elif self.semi_major_axis is not None:
            self.source = OrbitSource.ELEMENTS
            # 从半长轴计算高度
            self.altitude = self.semi_major_axis - EARTH_RADIUS_M
        elif self.altitude > 0:
            self.source = OrbitSource.SIMPLIFIED
            # 从高度计算半长轴
            self.semi_major_axis = EARTH_RADIUS_M + self.altitude
        else:
            # 默认值
            self.source = OrbitSource.SIMPLIFIED
            self.altitude = DEFAULT_ORBIT_ALTITUDE_M
            self.semi_major_axis = EARTH_RADIUS_M + self.altitude

    def get_semi_major_axis(self) -> float:
        """获取半长轴（米）"""
        if self.semi_major_axis is not None:
            return self.semi_major_axis
        return EARTH_RADIUS_M + self.altitude

    def get_period(self) -> float:
        """获取轨道周期（秒）"""
        a = self.get_semi_major_axis()
        return 2 * math.pi * math.sqrt(a**3 / EARTH_GM)


@dataclass
class SatelliteCapabilities:
    """卫星能力配置"""
    # 成像能力
    imaging_modes: List[ImagingMode] = field(default_factory=list)
    max_roll_angle: float = DEFAULT_MAX_ROLL_ANGLE_DEG  # 最大滚转角（度）- 绕X轴侧摆
    max_pitch_angle: float = DEFAULT_MAX_PITCH_ANGLE_DEG  # 最大俯仰角（度）- 绕Y轴前后斜视

    # 机动能力配置（支持标量和分轴限制）
    # 注意：agility字典包含标量限制（向后兼容）和分轴限制（新增）
    # 使用 get_effective_limits() 方法获取有效的分轴限制
    agility: Dict[str, float] = field(default_factory=lambda: {
        # 标量限制（向后兼容）
        'max_slew_rate': DEFAULT_MAX_ROLL_RATE_DEG_S,  # 度/秒
        'slew_acceleration': DEFAULT_MAX_ROLL_ACCEL_DEG_S2,  # 度/秒²
        'settling_time': 5.0,  # 稳定时间（秒）
        # 分轴角速度限制（新增）
        'max_roll_rate': DEFAULT_MAX_ROLL_RATE_DEG_S,  # 滚转轴最大角速度（度/秒）
        'max_pitch_rate': DEFAULT_MAX_PITCH_RATE_DEG_S,  # 俯仰轴最大角速度（度/秒）
        # 分轴角加速度限制（新增）
        'max_roll_acceleration': DEFAULT_MAX_ROLL_ACCEL_DEG_S2,  # 滚转轴最大角加速度（度/秒²）
        'max_pitch_acceleration': DEFAULT_MAX_PITCH_ACCEL_DEG_S2,  # 俯仰轴最大角加速度（度/秒²）
    })

    # 单圈约束配置（新增）
    max_starts_per_orbit: int = 5  # 单圈最大开机次数
    max_work_time_per_orbit: float = 600.0  # 单圈最大工作时长（秒）

    # 存储和能源
    storage_capacity: float = DEFAULT_STORAGE_CAPACITY_GB  # GB
    power_capacity: float = DEFAULT_POWER_CAPACITY_WH  # Wh
    data_rate: float = DEFAULT_DATA_RATE_MBPS  # Mbps

    # 成像参数（deprecated: 请使用 payload_config 获取模式特定参数）
    resolution: float = DEFAULT_RESOLUTION_M  # 分辨率（米）
    swath_width: float = DEFAULT_SWATH_WIDTH_M  # 幅宽（米）

    # 详细成像器配置（JSON模板中的详细参数）
    imager: Dict[str, Any] = field(default_factory=dict)

    # 详细成像模式参数（JSON模板中的模式详细配置）
    imaging_mode_details: List[Dict[str, Any]] = field(default_factory=list)

    # 每个成像模式的最小/最大成像时长约束
    # Format: {ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0}}
    imaging_mode_constraints: Dict[ImagingMode, Dict[str, float]] = field(
        default_factory=dict
    )

    # 载荷配置（新增）- 支持多成像模式
    payload_config: Optional[Any] = None  # PayloadConfiguration 类型（避免循环导入）

    def __post_init__(self):
        """初始化后确保分轴限制存在（向后兼容）并初始化payload_config"""
        # 如果agility中缺少分轴限制字段，从标量限制派生
        scalar_rate = self.agility.get('max_slew_rate', DEFAULT_MAX_ROLL_RATE_DEG_S)
        scalar_accel = self.agility.get('slew_acceleration', DEFAULT_MAX_ROLL_ACCEL_DEG_S2)

        # 确保分轴角速度限制存在
        if 'max_roll_rate' not in self.agility:
            self.agility['max_roll_rate'] = scalar_rate
        if 'max_pitch_rate' not in self.agility:
            # 俯仰通常比滚转慢（默认使用标量的2/3）
            self.agility['max_pitch_rate'] = scalar_rate * 0.67

        # 确保分轴角加速度限制存在
        if 'max_roll_acceleration' not in self.agility:
            self.agility['max_roll_acceleration'] = scalar_accel
        if 'max_pitch_acceleration' not in self.agility:
            # 俯仰通常比滚转慢（默认使用标量的2/3）
            self.agility['max_pitch_acceleration'] = scalar_accel * 0.67

        # 初始化payload_config（如果未设置）
        if self.payload_config is None:
            self._initialize_default_payload_config()

    def get_effective_limits(self, rotation_axis: Optional[tuple] = None) -> Dict[str, float]:
        """
        获取有效的角速度和角加速度限制

        如果提供了旋转轴，根据轴的方向计算投影后的有效限制。
        如果未提供，返回分轴限制。

        Args:
            rotation_axis: 旋转轴方向向量 (x, y, z)，单位向量或None

        Returns:
            包含以下键的字典：
            - max_roll_rate: 滚转角速度限制 (deg/s)
            - max_pitch_rate: 俯仰角速度限制 (deg/s)
            - max_roll_acceleration: 滚转角加速度限制 (deg/s²)
            - max_pitch_acceleration: 俯仰角加速度限制 (deg/s²)
            - effective_rate: 根据旋转轴计算的有效角速度 (deg/s)
            - effective_acceleration: 根据旋转轴计算的有效角加速度 (deg/s²)
        """
        # 获取分轴限制（确保存在）
        max_roll_rate = self.agility.get('max_roll_rate', DEFAULT_MAX_ROLL_RATE_DEG_S)
        max_pitch_rate = self.agility.get('max_pitch_rate', DEFAULT_MAX_PITCH_RATE_DEG_S)
        max_roll_accel = self.agility.get('max_roll_acceleration', DEFAULT_MAX_ROLL_ACCEL_DEG_S2)
        max_pitch_accel = self.agility.get('max_pitch_acceleration', DEFAULT_MAX_PITCH_ACCEL_DEG_S2)

        result = {
            'max_roll_rate': max_roll_rate,
            'max_pitch_rate': max_pitch_rate,
            'max_roll_acceleration': max_roll_accel,
            'max_pitch_acceleration': max_pitch_accel,
        }

        # 如果提供了旋转轴，计算有效限制
        if rotation_axis is not None:
            x, y, z = rotation_axis
            # 归一化
            norm = math.sqrt(x*x + y*y + z*z)
            if norm > 0:
                x, y, z = x/norm, y/norm, z/norm

            # 计算旋转轴在滚转(X)和俯仰(Y)方向上的投影
            roll_component = abs(x)
            pitch_component = abs(y)

            # 使用保守估计：根据主要旋转方向选择对应轴的限制
            if roll_component > pitch_component:
                effective_rate = max_roll_rate
                effective_accel = max_roll_accel
            else:
                effective_rate = max_pitch_rate
                effective_accel = max_pitch_accel

            result['effective_rate'] = effective_rate
            result['effective_acceleration'] = effective_accel

        return result

    def _initialize_default_payload_config(self) -> None:
        """根据现有配置初始化默认的payload_config"""
        from .imaging_mode import ImagingModeConfig
        from .payload_config import PayloadConfiguration

        # 根据成像模式确定payload_type
        if self.imaging_modes:
            first_mode = self.imaging_modes[0]
            if first_mode in [ImagingMode.PUSH_BROOM, ImagingMode.FRAME]:
                payload_type = 'optical'
            else:
                payload_type = 'sar'
        else:
            payload_type = 'optical'  # 默认

        # 创建默认模式配置
        if payload_type == 'optical':
            default_mode = 'push_broom'
            modes = {
                'push_broom': ImagingModeConfig(
                    resolution_m=self.resolution,
                    swath_width_m=self.swath_width,
                    power_consumption_w=OPTICAL_PUSH_BROOM_POWER_W,
                    data_rate_mbps=OPTICAL_PUSH_BROOM_DATA_RATE_MBPS,
                    min_duration_s=OPTICAL_IMAGING_MIN_DURATION_S,
                    max_duration_s=OPTICAL_IMAGING_MAX_DURATION_S,
                    mode_type='optical',
                    fov_config=self.imager.get('fov', {}),
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        else:  # sar
            default_mode = 'stripmap'
            modes = {}
            for mode in self.imaging_modes:
                if mode == ImagingMode.STRIPMAP:
                    modes['stripmap'] = ImagingModeConfig(
                        resolution_m=3.0,
                        swath_width_m=30000,
                        power_consumption_w=SAR_STRIPMAP_POWER_W,
                        data_rate_mbps=SAR_STRIPMAP_DATA_RATE_MBPS,
                        min_duration_s=SAR_1_STRIPMAP_MIN_DURATION_S,
                        max_duration_s=SAR_1_STRIPMAP_MAX_DURATION_S,
                        mode_type='sar',
                        fov_config={},
                        characteristics={}
                    )
                elif mode == ImagingMode.SPOTLIGHT:
                    modes['spotlight'] = ImagingModeConfig(
                        resolution_m=1.0,
                        swath_width_m=10000,
                        power_consumption_w=SAR_SPOTLIGHT_POWER_W,
                        data_rate_mbps=SAR_SPOTLIGHT_DATA_RATE_MBPS,
                        min_duration_s=SAR_1_SPOTLIGHT_MIN_DURATION_S,
                        max_duration_s=SAR_1_SPOTLIGHT_MAX_DURATION_S,
                        mode_type='sar',
                        fov_config={},
                        characteristics={}
                    )
                elif mode == ImagingMode.SCAN:
                    modes['scan'] = ImagingModeConfig(
                        resolution_m=10.0,
                        swath_width_m=100000,
                        power_consumption_w=SAR_SCAN_POWER_W,
                        data_rate_mbps=SAR_SCAN_DATA_RATE_MBPS,
                        min_duration_s=8.0,
                        max_duration_s=20.0,
                        mode_type='sar',
                        fov_config={},
                        characteristics={}
                    )
                elif mode == ImagingMode.SLIDING_SPOTLIGHT:
                    modes['sliding_spotlight'] = ImagingModeConfig(
                        resolution_m=1.5,
                        swath_width_m=20000,
                        power_consumption_w=SAR_SLIDING_SPOTLIGHT_POWER_W,
                        data_rate_mbps=SAR_SLIDING_SPOTLIGHT_DATA_RATE_MBPS,
                        min_duration_s=SAR_1_SLIDING_SPOTLIGHT_MIN_DURATION_S,
                        max_duration_s=SAR_1_SLIDING_SPOTLIGHT_MAX_DURATION_S,
                        mode_type='sar',
                        fov_config={},
                        characteristics={}
                    )

            if not modes:
                # 如果没有匹配的模式，创建默认条带模式
                modes = {
                    'stripmap': ImagingModeConfig(
                        resolution_m=self.resolution,
                        swath_width_m=self.swath_width,
                        power_consumption_w=SAR_STRIPMAP_POWER_W,
                        data_rate_mbps=SAR_STRIPMAP_DATA_RATE_MBPS,
                        min_duration_s=5.0,
                        max_duration_s=15.0,
                        mode_type='sar',
                        fov_config={},
                        characteristics={}
                    )
                }

        self.payload_config = PayloadConfiguration(
            payload_type=payload_type,
            default_mode=default_mode,
            modes=modes,
            common_fov=self.imager.get('fov') if self.imager else None
        )

    # ==========================================================================
    # 新增：通过 payload_config 访问成像模式特定参数
    # ==========================================================================

    def get_mode_config(self, mode: Optional[str] = None) -> Any:
        """
        获取指定成像模式的配置

        Args:
            mode: 成像模式名称，None则使用默认模式

        Returns:
            ImagingModeConfig
        """
        if self.payload_config is None:
            raise RuntimeError("payload_config is not initialized")
        return self.payload_config.get_mode_config(mode)

    def get_mode_resolution(self, mode: Optional[str] = None) -> float:
        """
        获取指定成像模式的分辨率

        Args:
            mode: 成像模式名称

        Returns:
            分辨率（米）
        """
        if self.payload_config:
            return self.payload_config.get_resolution(mode)
        # Fallback to legacy method
        return self._get_legacy_mode_resolution(mode)

    def get_mode_swath_width(self, mode: Optional[str] = None) -> float:
        """
        获取指定成像模式的幅宽

        Args:
            mode: 成像模式名称

        Returns:
            幅宽（米）
        """
        if self.payload_config:
            return self.payload_config.get_swath_width(mode)
        return self.swath_width

    def get_mode_power_consumption(self, mode: Optional[str] = None) -> float:
        """
        获取指定成像模式的功耗

        Args:
            mode: 成像模式名称

        Returns:
            功耗（瓦特）
        """
        if self.payload_config:
            return self.payload_config.get_power_consumption(mode)
        # 默认值
        return OPTICAL_PUSH_BROOM_POWER_W if self.payload_config and self.payload_config.payload_type == 'optical' else SAR_STRIPMAP_POWER_W

    def get_mode_data_rate(self, mode: Optional[str] = None) -> float:
        """
        获取指定成像模式的数据率

        Args:
            mode: 成像模式名称

        Returns:
            数据率（Mbps）
        """
        if self.payload_config:
            return self.payload_config.get_data_rate(mode)
        return self.data_rate

    def get_mode_duration_constraints(self, mode: Optional[str] = None) -> Dict[str, float]:
        """
        获取指定成像模式的时长约束

        Args:
            mode: 成像模式名称

        Returns:
            {'min_duration': float, 'max_duration': float}
        """
        if self.payload_config:
            config = self.payload_config.get_mode_config(mode)
            return {'min_duration': config.min_duration_s, 'max_duration': config.max_duration_s}
        # Fallback to legacy constraints
        return self._get_legacy_mode_constraints(mode)

    def _get_legacy_mode_resolution(self, mode: Optional[Any] = None) -> Optional[float]:
        """遗留方法：从imaging_mode_details获取分辨率"""
        mode_value = None
        if isinstance(mode, ImagingMode):
            mode_value = mode.value
        elif isinstance(mode, str):
            mode_value = mode
        else:
            return self.resolution

        for detail in self.imaging_mode_details:
            detail_mode_id = detail.get('mode_id')
            if detail_mode_id == mode_value:
                return float(detail.get('resolution', self.resolution))

        return self.resolution

    def _get_legacy_mode_constraints(self, mode: Optional[Any] = None) -> Dict[str, float]:
        """遗留方法：从imaging_mode_constraints获取时长约束"""
        if isinstance(mode, str):
            try:
                mode = ImagingMode(mode)
            except ValueError:
                return {'min_duration': 5.0, 'max_duration': 15.0}

        if mode and mode in self.imaging_mode_constraints:
            constraints = self.imaging_mode_constraints[mode]
            return {
                'min_duration': constraints.get('min_duration', 5.0),
                'max_duration': constraints.get('max_duration', 15.0)
            }

        # 返回默认约束
        return {'min_duration': 5.0, 'max_duration': 15.0}

    def supports_mode(self, mode: ImagingMode) -> bool:
        """检查是否支持指定成像模式"""
        return mode in self.imaging_modes

    def get_imaging_constraints(self, mode: ImagingMode) -> Optional[Dict[str, float]]:
        """
        获取指定成像模式的时长约束

        Args:
            mode: 成像模式

        Returns:
            约束字典 {'min_duration': float, 'max_duration': float} 或 None
        """
        return self.imaging_mode_constraints.get(mode)

    def validate_constraints(self) -> bool:
        """
        验证所有成像模式约束的有效性

        Returns:
            True if all constraints are valid

        Raises:
            ValueError: if any constraint is invalid
        """
        for mode, constraints in self.imaging_mode_constraints.items():
            if 'min_duration' not in constraints:
                raise ValueError(
                    f"Missing 'min_duration' for mode {mode.value}"
                )
            if 'max_duration' not in constraints:
                raise ValueError(
                    f"Missing 'max_duration' for mode {mode.value}"
                )

            min_dur = constraints['min_duration']
            max_dur = constraints['max_duration']

            if min_dur > max_dur:
                raise ValueError(
                    f"min_duration ({min_dur}) greater than max_duration ({max_dur}) "
                    f"for mode {mode.value}"
                )
            if min_dur == max_dur:
                raise ValueError(
                    f"min_duration ({min_dur}) equal to max_duration ({max_dur}) "
                    f"for mode {mode.value}"
                )

        return True

    def get_mode_resolution(self, mode: Any) -> Optional[float]:
        """
        获取指定成像模式的分辨率

        Args:
            mode: 成像模式（ImagingMode枚举或字符串）

        Returns:
            分辨率（米），如果模式不存在则返回None
        """
        # 将mode转换为字符串进行比较
        mode_value = None
        if isinstance(mode, ImagingMode):
            mode_value = mode.value
        elif isinstance(mode, str):
            mode_value = mode
        else:
            return None

        # 在imaging_mode_details中查找对应模式的分辨率
        for detail in self.imaging_mode_details:
            detail_mode_id = detail.get('mode_id')
            if detail_mode_id == mode_value:
                return float(detail.get('resolution', self.resolution))

        return None

    def get_best_resolution(self) -> float:
        """
        获取卫星的最佳（最高）分辨率

        Returns:
            最高分辨率（米）
        """
        resolutions = [self.resolution]  # 默认分辨率

        # 收集所有成像模式的分辨率
        for detail in self.imaging_mode_details:
            res = detail.get('resolution')
            if res is not None:
                resolutions.append(float(res))

        return min(resolutions) if resolutions else self.resolution

    def can_satisfy_resolution(self, required_resolution: float) -> bool:
        """
        检查卫星是否能满足指定的分辨率要求

        逻辑：卫星有任意成像模式的分辨率优于或等于要求（数值更小或相等）

        Args:
            required_resolution: 所需分辨率（米）

        Returns:
            True if satellite can satisfy the resolution requirement
        """
        if required_resolution is None:
            return True

        # 检查所有可用的分辨率
        available_resolutions = [self.resolution]

        # 添加所有成像模式的分辨率
        for detail in self.imaging_mode_details:
            res = detail.get('resolution')
            if res is not None:
                available_resolutions.append(float(res))

        # 检查是否有分辨率优于或等于要求（数值更小或相等）
        return any(r <= required_resolution for r in available_resolutions)


@dataclass
class Satellite:
    """
    卫星模型

    Attributes:
        id: 卫星唯一标识
        name: 卫星名称
        sat_type: 卫星类型
        orbit: 轨道参数
        capabilities: 能力配置
        tle_line1: TLE第一行（可选）
        tle_line2: TLE第二行（可选）
    """
    id: str
    name: str
    sat_type: SatelliteType
    orbit: Orbit = field(default_factory=Orbit)
    capabilities: SatelliteCapabilities = field(default_factory=SatelliteCapabilities)

    # TLE（用于SGP4传播）
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    # 当前状态（运行时被更新）
    current_power: float = field(default=0.0)  # 当前电量
    current_storage: float = field(default=0.0)  # 当前存储使用

    def __post_init__(self):
        """初始化后设置默认值"""
        if not self.capabilities.imaging_modes:
            self._set_default_capabilities()
        if self.current_power == 0.0:
            self.current_power = self.capabilities.power_capacity

    def _set_default_capabilities(self):
        """根据卫星类型设置默认能力"""
        if self.sat_type == SatelliteType.OPTICAL_1:
            self.capabilities.imaging_modes = [ImagingMode.PUSH_BROOM]
            self.capabilities.max_roll_angle = OPTICAL_MAX_ROLL_ANGLE_DEG
            self.capabilities.max_pitch_angle = OPTICAL_MAX_PITCH_ANGLE_DEG
            self.capabilities.storage_capacity = OPTICAL_1_STORAGE_CAPACITY_GB
            self.capabilities.power_capacity = OPTICAL_1_POWER_CAPACITY_WH
            self.capabilities.resolution = DEFAULT_RESOLUTION_M
            # 光学卫星成像时长约束
            self.capabilities.imaging_mode_constraints = {
                ImagingMode.PUSH_BROOM: {
                    'min_duration': OPTICAL_IMAGING_MIN_DURATION_S,
                    'max_duration': OPTICAL_IMAGING_MAX_DURATION_S
                }
            }
        elif self.sat_type == SatelliteType.OPTICAL_2:
            self.capabilities.imaging_modes = [ImagingMode.PUSH_BROOM, ImagingMode.FRAME]
            self.capabilities.max_roll_angle = OPTICAL_MAX_ROLL_ANGLE_DEG
            self.capabilities.max_pitch_angle = OPTICAL_MAX_PITCH_ANGLE_DEG
            self.capabilities.storage_capacity = OPTICAL_2_STORAGE_CAPACITY_GB
            self.capabilities.power_capacity = OPTICAL_2_POWER_CAPACITY_WH
            self.capabilities.resolution = 5.0
            # 光学卫星成像时长约束
            self.capabilities.imaging_mode_constraints = {
                ImagingMode.PUSH_BROOM: {
                    'min_duration': OPTICAL_IMAGING_MIN_DURATION_S,
                    'max_duration': OPTICAL_IMAGING_MAX_DURATION_S
                },
                ImagingMode.FRAME: {
                    'min_duration': OPTICAL_IMAGING_MIN_DURATION_S,
                    'max_duration': OPTICAL_IMAGING_MAX_DURATION_S
                }
            }
        elif self.sat_type == SatelliteType.SAR_1:
            self.capabilities.imaging_modes = [
                ImagingMode.SPOTLIGHT,
                ImagingMode.SLIDING_SPOTLIGHT,
                ImagingMode.STRIPMAP
            ]
            self.capabilities.max_roll_angle = SAR_MAX_ROLL_ANGLE_DEG
            self.capabilities.max_pitch_angle = SAR_MAX_PITCH_ANGLE_DEG
            self.capabilities.storage_capacity = SAR_1_STORAGE_CAPACITY_GB
            self.capabilities.power_capacity = SAR_1_POWER_CAPACITY_WH
            self.capabilities.resolution = 3.0
            # SAR-1成像时长约束
            self.capabilities.imaging_mode_constraints = {
                ImagingMode.SPOTLIGHT: {
                    'min_duration': SAR_1_SPOTLIGHT_MIN_DURATION_S,
                    'max_duration': SAR_1_SPOTLIGHT_MAX_DURATION_S
                },
                ImagingMode.SLIDING_SPOTLIGHT: {
                    'min_duration': SAR_1_SLIDING_SPOTLIGHT_MIN_DURATION_S,
                    'max_duration': SAR_1_SLIDING_SPOTLIGHT_MAX_DURATION_S
                },
                ImagingMode.STRIPMAP: {
                    'min_duration': SAR_1_STRIPMAP_MIN_DURATION_S,
                    'max_duration': SAR_1_STRIPMAP_MAX_DURATION_S
                }
            }
        elif self.sat_type == SatelliteType.SAR_2:
            self.capabilities.imaging_modes = [
                ImagingMode.SPOTLIGHT,
                ImagingMode.SLIDING_SPOTLIGHT,
                ImagingMode.STRIPMAP
            ]
            self.capabilities.max_roll_angle = SAR_MAX_ROLL_ANGLE_DEG
            self.capabilities.max_pitch_angle = SAR_MAX_PITCH_ANGLE_DEG
            self.capabilities.storage_capacity = SAR_2_STORAGE_CAPACITY_GB
            self.capabilities.power_capacity = SAR_2_POWER_CAPACITY_WH
            self.capabilities.resolution = 1.0
            # SAR-2成像时长约束
            self.capabilities.imaging_mode_constraints = {
                ImagingMode.SPOTLIGHT: {
                    'min_duration': SAR_2_SPOTLIGHT_MIN_DURATION_S,
                    'max_duration': SAR_2_SPOTLIGHT_MAX_DURATION_S
                },
                ImagingMode.SLIDING_SPOTLIGHT: {
                    'min_duration': SAR_2_SLIDING_SPOTLIGHT_MIN_DURATION_S,
                    'max_duration': SAR_2_SLIDING_SPOTLIGHT_MAX_DURATION_S
                },
                ImagingMode.STRIPMAP: {
                    'min_duration': SAR_2_STRIPMAP_MIN_DURATION_S,
                    'max_duration': SAR_2_STRIPMAP_MAX_DURATION_S
                }
            }

    def get_position_sgp4(self, dt: datetime) -> tuple:
        """
        使用SGP4计算卫星位置

        Returns:
            (x, y, z) in ECI coordinates (meters)
        """
        from sgp4.api import Satrec, jday

        if not self.tle_line1 or not self.tle_line2:
            # 如果没有TLE，使用简化轨道模型
            return self._get_position_simplified(dt)

        sat = Satrec.twoline2rv(self.tle_line1, self.tle_line2)
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            raise RuntimeError(f"SGP4 propagation error: {e}")

        return r  # (x, y, z) in km

    def _get_position_simplified(self, dt: datetime) -> tuple:
        """简化的轨道位置计算（用于没有TLE的情况）"""
        # 处理 naive datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # 这是一个简化的圆轨道模型
        period = self.orbit.get_period()
        mean_motion = 2 * math.pi / period

        # 计算从参考时间开始的秒数
        delta_t = (dt - REFERENCE_EPOCH).total_seconds()

        # 平近点角
        M = self.orbit.mean_anomaly + math.radians(mean_motion * delta_t)

        # 简化假设：圆轨道
        a = self.orbit.get_semi_major_axis()
        i = math.radians(self.orbit.inclination)
        raan = math.radians(self.orbit.raan)

        # 在轨道平面内的位置
        x_orb = a * math.cos(M)
        y_orb = a * math.sin(M)

        # 转换到ECI坐标系（简化，假设轨道倾角为i，升交点赤经为raan）
        x = x_orb * math.cos(raan) - y_orb * math.cos(i) * math.sin(raan)
        y = x_orb * math.sin(raan) + y_orb * math.cos(i) * math.cos(raan)
        z = y_orb * math.sin(i)

        return (x * METERS_TO_KM, y * METERS_TO_KM, z * METERS_TO_KM)  # 转换为km以与SGP4一致

    def get_subpoint(self, dt: datetime) -> tuple:
        """
        获取星下点坐标

        Returns:
            (latitude, longitude, altitude) in degrees and meters
        """
        from sgp4.api import jday

        r = self.get_position_sgp4(dt)

        # ECI to LLA conversion (simplified)
        x, y, z = r
        r_norm = math.sqrt(x**2 + y**2 + z**2)

        # 纬度
        lat = math.degrees(math.asin(z / r_norm))

        # 经度（需要计算GMST，这里简化）
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        # 简化格林尼治恒星时计算
        gmst = (GMST_CONSTANT_1 + GMST_CONSTANT_2 * (jd - J2000_JULIAN_DATE)) % 360
        lon = (math.degrees(math.atan2(y, x)) - gmst) % 360
        if lon > 180:
            lon -= 360

        alt = r_norm * KM_TO_METERS - EARTH_RADIUS_M  # 转换为米

        return (lat, lon, alt)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 根据轨道来源选择序列化格式
        orbit_dict: Dict[str, Any] = {
            'orbit_type': self.orbit.orbit_type.value,
            'source': self.orbit.source.value,
        }

        # ★ 添加epoch字段（ISO格式带Z表示UTC）
        if self.orbit.epoch:
            epoch_utc = self.orbit.epoch.astimezone(timezone.utc)
            orbit_dict['epoch'] = epoch_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        if self.orbit.source == OrbitSource.TLE and self.orbit.tle_line1 and self.orbit.tle_line2:
            # TLE格式
            orbit_dict['tle_line1'] = self.orbit.tle_line1
            orbit_dict['tle_line2'] = self.orbit.tle_line2
        elif self.orbit.source == OrbitSource.ELEMENTS:
            # 六根数格式
            orbit_dict['semi_major_axis'] = self.orbit.semi_major_axis
            orbit_dict['eccentricity'] = self.orbit.eccentricity
            orbit_dict['inclination'] = self.orbit.inclination
            orbit_dict['raan'] = self.orbit.raan
            orbit_dict['arg_of_perigee'] = self.orbit.arg_of_perigee
            orbit_dict['mean_anomaly'] = self.orbit.mean_anomaly
        else:
            # 简化格式
            orbit_dict['altitude'] = self.orbit.altitude
            orbit_dict['inclination'] = self.orbit.inclination

        # Convert imaging_mode_constraints to serializable format
        constraints_dict = {
            mode.value: constraints
            for mode, constraints in self.capabilities.imaging_mode_constraints.items()
        }

        # 准备agility字典（确保包含分轴限制）
        agility_dict = dict(self.capabilities.agility)
        # 确保分轴限制字段存在
        scalar_rate = agility_dict.get('max_slew_rate', DEFAULT_MAX_ROLL_RATE_DEG_S)
        scalar_accel = agility_dict.get('slew_acceleration', DEFAULT_MAX_ROLL_ACCEL_DEG_S2)
        if 'max_roll_rate' not in agility_dict:
            agility_dict['max_roll_rate'] = scalar_rate
        if 'max_pitch_rate' not in agility_dict:
            agility_dict['max_pitch_rate'] = scalar_rate * 0.67
        if 'max_roll_acceleration' not in agility_dict:
            agility_dict['max_roll_acceleration'] = scalar_accel
        if 'max_pitch_acceleration' not in agility_dict:
            agility_dict['max_pitch_acceleration'] = scalar_accel * 0.67

        # 准备capabilities字典
        capabilities_dict = {
            'imaging_modes': [m.value for m in self.capabilities.imaging_modes],
            'max_roll_angle': self.capabilities.max_roll_angle,
            'max_pitch_angle': self.capabilities.max_pitch_angle,
            'max_starts_per_orbit': self.capabilities.max_starts_per_orbit,
            'max_work_time_per_orbit': self.capabilities.max_work_time_per_orbit,
            'storage_capacity': self.capabilities.storage_capacity,
            'power_capacity': self.capabilities.power_capacity,
            'data_rate': self.capabilities.data_rate,
            'imager': self.capabilities.imager,
            'imaging_mode_details': self.capabilities.imaging_mode_details,
            'imaging_mode_constraints': constraints_dict,
            'agility': agility_dict,
        }

        # 添加payload_config（如果存在）
        if self.capabilities.payload_config is not None:
            capabilities_dict['payload_config'] = self.capabilities.payload_config.to_dict()

        return {
            'id': self.id,
            'name': self.name,
            'sat_type': self.sat_type.value,
            'orbit': orbit_dict,
            'capabilities': capabilities_dict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Satellite':
        """从字典创建，支持多种轨道配置格式"""
        # 映射简化的sat_type到枚举值
        sat_type_mapping = {
            'optical': SatelliteType.OPTICAL_1,
            'sar': SatelliteType.SAR_1,
            'optical_1': SatelliteType.OPTICAL_1,
            'optical_2': SatelliteType.OPTICAL_2,
            'sar_1': SatelliteType.SAR_1,
            'sar_2': SatelliteType.SAR_2,
        }
        sat_type_str = data['sat_type']
        if sat_type_str in sat_type_mapping:
            sat_type = sat_type_mapping[sat_type_str]
        else:
            sat_type = SatelliteType(sat_type_str)

        orbit_data = data.get('orbit', {})

        # 优先级1: 检查是否有TLE（支持orbit.tle_line1/2或orbit.tle数组）
        tle_line1 = orbit_data.get('tle_line1')
        tle_line2 = orbit_data.get('tle_line2')
        if not tle_line1 and not tle_line2:
            tle_array = orbit_data.get('tle', [])
            if len(tle_array) >= 2:
                tle_line1 = tle_array[0]
                tle_line2 = tle_array[1]

        # 优先级2: 检查是否有完整的六根数
        semi_major_axis = orbit_data.get('semi_major_axis')

        # ★ 解析epoch字段（强制UTC）
        epoch_str = orbit_data.get('epoch')
        epoch = None
        if epoch_str:
            epoch = parse_epoch_string(epoch_str)

        # ★ TLE模式下epoch冲突检测和警告
        if tle_line1 and tle_line2 and epoch_str:
            warnings.warn(
                f"Orbit configured with both TLE and explicit epoch ({epoch_str}). "
                f"TLE内置历元将覆盖指定epoch. "
                f"Remove 'epoch' field when using TLE to suppress this warning.",
                UserWarning,
                stacklevel=2
            )

        # 创建Orbit对象
        if tle_line1 and tle_line2:
            # TLE格式
            orbit = Orbit(
                orbit_type=OrbitType(orbit_data.get('orbit_type', 'SSO')),
                tle_line1=tle_line1,
                tle_line2=tle_line2,
                inclination=orbit_data.get('inclination', 97.4),
                raan=orbit_data.get('raan', 0.0),
                epoch=epoch,  # ★ TLE格式也保存epoch（虽然不会被使用）
            )
        elif semi_major_axis is not None:
            # 六根数格式
            orbit = Orbit(
                orbit_type=OrbitType(orbit_data.get('orbit_type', 'SSO')),
                semi_major_axis=semi_major_axis,
                eccentricity=orbit_data.get('eccentricity', 0.0),
                inclination=orbit_data.get('inclination', 97.4),
                raan=orbit_data.get('raan', 0.0),
                arg_of_perigee=orbit_data.get('arg_of_perigee', 0.0),
                mean_anomaly=orbit_data.get('mean_anomaly', 0.0),
                epoch=epoch,  # ★ 六根数格式的epoch
            )
        else:
            # 简化格式（向后兼容）
            orbit = Orbit(
                orbit_type=OrbitType(orbit_data.get('orbit_type', 'SSO')),
                altitude=orbit_data.get('altitude', DEFAULT_ORBIT_ALTITUDE_M),
                inclination=orbit_data.get('inclination', DEFAULT_ORBIT_INCLINATION_DEG),
                eccentricity=orbit_data.get('eccentricity', DEFAULT_ORBIT_ECCENTRICITY),
                raan=orbit_data.get('raan', DEFAULT_RAAN_DEG),
                arg_of_perigee=orbit_data.get('arg_of_perigee', DEFAULT_ARG_OF_PERIGEE_DEG),
                mean_anomaly=orbit_data.get('mean_anomaly', DEFAULT_MEAN_ANOMALY_DEG),
                epoch=epoch,  # ★ 简化参数格式的epoch
            )

        cap_data = data.get('capabilities', {})

        # 读取详细成像器配置（支持多种可能的字段名）
        imager_data = cap_data.get('imager', {})
        if not imager_data:
            # 尝试从旧格式的字段构建imager配置
            imager_data = {}

        # 读取详细成像模式配置（支持多种可能的字段名）
        imaging_mode_details = cap_data.get('imaging_mode_details', [])
        if not imaging_mode_details:
            imaging_mode_details = cap_data.get('imaging_modes_details', [])

        # 解析成像模式列表（支持字符串列表或对象列表）
        raw_imaging_modes = cap_data.get('imaging_modes', [])
        imaging_modes = []
        for m in raw_imaging_modes:
            if isinstance(m, str):
                # 旧格式：字符串列表
                imaging_modes.append(ImagingMode(m))
            elif isinstance(m, dict) and 'mode_id' in m:
                # 新格式：对象列表，提取mode_id
                imaging_modes.append(ImagingMode(m['mode_id']))
                # 同时将对象添加到imaging_mode_details（如果还没有的话）
                if m not in imaging_mode_details:
                    imaging_mode_details.append(m)

        # 解析成像模式约束
        raw_constraints = cap_data.get('imaging_mode_constraints', {})
        imaging_mode_constraints: Dict[ImagingMode, Dict[str, float]] = {}
        for mode_str, constraints in raw_constraints.items():
            try:
                mode = ImagingMode(mode_str)
                imaging_mode_constraints[mode] = {
                    'min_duration': float(constraints['min_duration']),
                    'max_duration': float(constraints['max_duration']),
                }
            except (ValueError, KeyError):
                # 跳过无效的约束配置
                continue

        # 解析agility配置
        raw_agility = cap_data.get('agility', {})
        agility_dict = {
            # 标量限制（向后兼容）
            'max_slew_rate': raw_agility.get('max_slew_rate', DEFAULT_MAX_ROLL_RATE_DEG_S),
            'slew_acceleration': raw_agility.get('slew_acceleration', DEFAULT_MAX_ROLL_ACCEL_DEG_S2),
            'settling_time': raw_agility.get('settling_time', 5.0),
        }
        # 分轴角速度限制
        agility_dict['max_roll_rate'] = raw_agility.get('max_roll_rate', agility_dict['max_slew_rate'])
        agility_dict['max_pitch_rate'] = raw_agility.get('max_pitch_rate', agility_dict['max_slew_rate'] * 0.67)
        # 分轴角加速度限制
        agility_dict['max_roll_acceleration'] = raw_agility.get('max_roll_acceleration', agility_dict['slew_acceleration'])
        agility_dict['max_pitch_acceleration'] = raw_agility.get('max_pitch_acceleration', agility_dict['slew_acceleration'] * 0.67)

        # 解析payload_config（如果存在）
        payload_config = None
        if 'payload_config' in cap_data:
            from .payload_config import PayloadConfiguration
            try:
                payload_config = PayloadConfiguration.from_dict(cap_data['payload_config'])
            except (ValueError, KeyError) as e:
                warnings.warn(f"Failed to parse payload_config: {e}", UserWarning)

        capabilities = SatelliteCapabilities(
            imaging_modes=imaging_modes,
            max_roll_angle=cap_data.get('max_roll_angle', DEFAULT_MAX_ROLL_ANGLE_DEG),
            max_pitch_angle=cap_data.get('max_pitch_angle', DEFAULT_MAX_PITCH_ANGLE_DEG),
            max_starts_per_orbit=cap_data.get('max_starts_per_orbit', 5),
            max_work_time_per_orbit=cap_data.get('max_work_time_per_orbit', 600.0),
            storage_capacity=cap_data.get('storage_capacity', DEFAULT_STORAGE_CAPACITY_GB),
            power_capacity=cap_data.get('power_capacity', DEFAULT_POWER_CAPACITY_WH),
            data_rate=cap_data.get('data_rate', DEFAULT_DATA_RATE_MBPS),
            resolution=cap_data.get('resolution', DEFAULT_RESOLUTION_M),
            swath_width=cap_data.get('swath_width', DEFAULT_SWATH_WIDTH_M),
            imager=imager_data,
            imaging_mode_details=imaging_mode_details,
            imaging_mode_constraints=imaging_mode_constraints,
            agility=agility_dict,
            payload_config=payload_config,
        )

        # 读取TLE（支持多种格式）
        tle_line1 = data.get('tle_line1')
        tle_line2 = data.get('tle_line2')
        # 如果根级别没有，尝试介orbit.tle数组
        if not tle_line1 and not tle_line2:
            tle_array = orbit_data.get('tle', [])
            if len(tle_array) >= 2:
                tle_line1 = tle_array[0]
                tle_line2 = tle_array[1]

        # 读取TLE用于Satellite对象（优先级：orbit.tle_line1/2 > orbit.tle数组 > 根级别tle）
        sat_tle_line1 = orbit.tle_line1 if orbit.tle_line1 else data.get('tle_line1')
        sat_tle_line2 = orbit.tle_line2 if orbit.tle_line2 else data.get('tle_line2')

        return cls(
            id=data['id'],
            name=data['name'],
            sat_type=sat_type,
            orbit=orbit,
            capabilities=capabilities,
            tle_line1=sat_tle_line1,
            tle_line2=sat_tle_line2,
        )
