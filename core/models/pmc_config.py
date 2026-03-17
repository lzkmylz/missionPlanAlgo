"""
Pitch Motion Compensation (PMC) 配置模块

定义主动前向推扫模式的配置参数和计算工具。

PMC原理：
- 卫星在成像过程中以固定俯仰角速度机动
- 使相机光轴的地面投影速度与卫星飞行速度部分抵消
- 等效降低地面推扫速度，延长积分时间，提高SNR

技术参数：
- 俯仰角速度 θ_dot = v_satellite * R / h
  其中 R为降速比，h为轨道高度
- 等效积分时间增益 = 1 / (1 - R)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import math


@dataclass
class PitchMotionCompensationConfig:
    """
    PMC模式配置

    Attributes:
        speed_reduction_ratio: 降速比（0.1-0.75），表示地速降低百分比
        pitch_rate_dps: 俯仰角速度（度/秒），None表示自动计算
        min_altitude_m: 最小适用轨道高度（米）
        max_roll_angle_deg: PMC模式下最大允许滚转角（度）
        max_pitch_angle_deg: PMC模式下最大允许俯仰角（度）
        integration_time_gain: 积分时间增益因子
        enter_exit_time_s: 进入/退出PMC状态的机动时间（秒）
        power_overhead_factor: 功耗增加系数（相对于普通模式）
        max_continuous_duration_s: 最大连续PMC成像时长（秒）
    """
    speed_reduction_ratio: float = 0.25
    pitch_rate_dps: Optional[float] = None  # None时自动计算
    min_altitude_m: float = 400000.0
    max_roll_angle_deg: float = 30.0
    max_pitch_angle_deg: float = 20.0
    integration_time_gain: float = 1.33
    enter_exit_time_s: float = 3.0  # 进入和退出各需的时间
    power_overhead_factor: float = 1.2
    max_continuous_duration_s: float = 30.0

    def __post_init__(self):
        """验证参数有效性"""
        if not 0.1 <= self.speed_reduction_ratio <= 0.75:
            raise ValueError(
                f"speed_reduction_ratio must be in [0.1, 0.75], got {self.speed_reduction_ratio}"
            )
        if self.pitch_rate_dps is not None and self.pitch_rate_dps < 0:
            raise ValueError(f"pitch_rate_dps must be non-negative")
        if self.min_altitude_m < 200000:
            raise ValueError(f"min_altitude_m must be >= 200km")
        if self.max_continuous_duration_s <= 0:
            raise ValueError(f"max_continuous_duration_s must be positive")

        # 自动计算积分时间增益
        self.integration_time_gain = 1.0 / (1.0 - self.speed_reduction_ratio)

    def calculate_pitch_rate(self, orbit_altitude_m: float) -> float:
        """
        计算所需俯仰角速度

        公式: θ_dot = v_ground * R / h
        其中:
        - v_ground: 卫星地面速度（约7.5km/s）
        - R: 降速比
        - h: 轨道高度

        Args:
            orbit_altitude_m: 轨道高度（米）

        Returns:
            俯仰角速度（度/秒）
        """
        if orbit_altitude_m < self.min_altitude_m:
            raise ValueError(
                f"Orbit altitude {orbit_altitude_m}m below minimum {self.min_altitude_m}m for PMC"
            )

        # 计算地面速度（近似）
        earth_radius = 6371000.0  # 米
        r = earth_radius + orbit_altitude_m
        mu = 3.986004418e14  # m^3/s^2
        v_orbital = math.sqrt(mu / r)  # m/s

        # 俯仰角速度（弧度/秒）
        pitch_rate_rad_s = v_orbital * self.speed_reduction_ratio / orbit_altitude_m

        return math.degrees(pitch_rate_rad_s)

    def get_pitch_rate(self, orbit_altitude_m: float) -> float:
        """
        获取俯仰角速度（优先使用配置值，否则计算）

        Args:
            orbit_altitude_m: 轨道高度（米）

        Returns:
            俯仰角速度（度/秒）
        """
        if self.pitch_rate_dps is not None:
            return self.pitch_rate_dps
        return self.calculate_pitch_rate(orbit_altitude_m)

    def calculate_effective_integration_time(self, physical_duration_s: float) -> float:
        """
        计算等效积分时间

        Args:
            physical_duration_s: 物理成像时长（秒）

        Returns:
            等效积分时间（秒）
        """
        return physical_duration_s * self.integration_time_gain

    def calculate_snr_gain_db(self) -> float:
        """
        估算SNR增益（分贝）

        SNR与积分时间的平方根成正比
        增益 = 10 * log10(integration_gain)

        Returns:
            SNR增益（dB）
        """
        return 10.0 * math.log10(self.integration_time_gain)

    def calculate_ground_velocity(self, orbit_altitude_m: float) -> float:
        """
        计算PMC模式下的等效地面速度

        Args:
            orbit_altitude_m: 轨道高度（米）

        Returns:
            等效地面速度（米/秒）
        """
        earth_radius = 6371000.0
        r = earth_radius + orbit_altitude_m
        mu = 3.986004418e14
        v_orbital = math.sqrt(mu / r)

        return v_orbital * (1.0 - self.speed_reduction_ratio)

    def check_altitude_feasibility(self, orbit_altitude_m: float) -> Tuple[bool, str]:
        """
        检查轨道高度是否适合PMC模式

        Args:
            orbit_altitude_m: 轨道高度（米）

        Returns:
            (是否可行, 原因说明)
        """
        if orbit_altitude_m < self.min_altitude_m:
            return False, f"Altitude {orbit_altitude_m}m below minimum {self.min_altitude_m}m"

        # 计算所需俯仰角速度
        pitch_rate = self.calculate_pitch_rate(orbit_altitude_m)

        # 检查是否在合理范围内（通常<2度/秒）
        if pitch_rate > 2.0:
            return False, f"Required pitch rate {pitch_rate:.2f}°/s exceeds 2°/s limit"

        return True, f"Feasible with pitch rate {pitch_rate:.3f}°/s"

    def get_total_maneuver_time(self, include_imaging: bool = True, imaging_duration_s: float = 0.0) -> float:
        """
        获取PMC任务总机动时间

        Args:
            include_imaging: 是否包含成像时间
            imaging_duration_s: 成像时长（秒）

        Returns:
            总时间（秒）
        """
        # 进入 + 退出时间
        total = 2.0 * self.enter_exit_time_s
        if include_imaging:
            total += imaging_duration_s
        return total

    def calculate_power_consumption(self, base_power_w: float, duration_s: float) -> float:
        """
        计算PMC模式功耗

        Args:
            base_power_w: 基础功耗（瓦特）
            duration_s: 成像时长（秒）

        Returns:
            总能耗（Wh）
        """
        # 考虑进入和退出机动的额外功耗
        maneuver_time = 2.0 * self.enter_exit_time_s
        effective_duration = duration_s + maneuver_time * 0.5  # 机动期间功耗略低

        power = base_power_w * self.power_overhead_factor
        return (power * effective_duration) / 3600.0  # Wh

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'speed_reduction_ratio': self.speed_reduction_ratio,
            'pitch_rate_dps': self.pitch_rate_dps,
            'min_altitude_m': self.min_altitude_m,
            'max_roll_angle_deg': self.max_roll_angle_deg,
            'max_pitch_angle_deg': self.max_pitch_angle_deg,
            'integration_time_gain': self.integration_time_gain,
            'enter_exit_time_s': self.enter_exit_time_s,
            'power_overhead_factor': self.power_overhead_factor,
            'max_continuous_duration_s': self.max_continuous_duration_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PitchMotionCompensationConfig':
        """从字典创建"""
        return cls(
            speed_reduction_ratio=data.get('speed_reduction_ratio', 0.25),
            pitch_rate_dps=data.get('pitch_rate_dps'),
            min_altitude_m=data.get('min_altitude_m', 400000.0),
            max_roll_angle_deg=data.get('max_roll_angle_deg', 30.0),
            max_pitch_angle_deg=data.get('max_pitch_angle_deg', 20.0),
            integration_time_gain=data.get('integration_time_gain', 1.33),
            enter_exit_time_s=data.get('enter_exit_time_s', 3.0),
            power_overhead_factor=data.get('power_overhead_factor', 1.2),
            max_continuous_duration_s=data.get('max_continuous_duration_s', 30.0),
        )


# 预定义PMC配置模板

PMC_CONFIG_10PERCENT = PitchMotionCompensationConfig(
    speed_reduction_ratio=0.10,
    max_roll_angle_deg=35.0,
    max_continuous_duration_s=30.0,
)

PMC_CONFIG_25PERCENT = PitchMotionCompensationConfig(
    speed_reduction_ratio=0.25,
    max_roll_angle_deg=30.0,
    max_continuous_duration_s=30.0,
)

PMC_CONFIG_50PERCENT = PitchMotionCompensationConfig(
    speed_reduction_ratio=0.50,
    max_roll_angle_deg=25.0,
    max_continuous_duration_s=25.0,
)

PMC_CONFIG_75PERCENT = PitchMotionCompensationConfig(
    speed_reduction_ratio=0.75,
    max_roll_angle_deg=15.0,
    max_continuous_duration_s=20.0,
    power_overhead_factor=1.3,
)

PMC_CONFIG_TEMPLATES = {
    '10percent': PMC_CONFIG_10PERCENT,
    '25percent': PMC_CONFIG_25PERCENT,
    '50percent': PMC_CONFIG_50PERCENT,
    '75percent': PMC_CONFIG_75PERCENT,
}


def get_pmc_config_template(template_name: str) -> Optional[PitchMotionCompensationConfig]:
    """
    获取预定义的PMC配置模板

    Args:
        template_name: 模板名称（如 '25percent'）

    Returns:
        PitchMotionCompensationConfig 或 None
    """
    return PMC_CONFIG_TEMPLATES.get(template_name)


def create_pmc_config_for_altitude(
    orbit_altitude_m: float,
    speed_reduction_ratio: float = 0.25,
    **kwargs
) -> PitchMotionCompensationConfig:
    """
    根据轨道高度创建PMC配置

    Args:
        orbit_altitude_m: 轨道高度（米）
        speed_reduction_ratio: 降速比
        **kwargs: 其他配置参数

    Returns:
        PitchMotionCompensationConfig
    """
    config = PitchMotionCompensationConfig(
        speed_reduction_ratio=speed_reduction_ratio,
        **kwargs
    )

    # 自动计算俯仰角速度
    pitch_rate = config.calculate_pitch_rate(orbit_altitude_m)
    config.pitch_rate_dps = pitch_rate

    return config
