"""
成像时间计算器

根据目标和成像模式计算成像时长，替代EDD调度器中的硬编码值
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from core.models import Target, TargetType, ImagingMode


@dataclass
class PowerProfile:
    """
    功率配置文件

    定义不同活动类型的功率系数（相对于总功率容量的比例）
    """

    # 默认功率系数
    DEFAULT_COEFFICIENTS: Dict[str, float] = None

    def __init__(self, custom_coefficients: Dict[str, float] = None):
        """
        初始化功率配置文件

        Args:
            custom_coefficients: 自定义功率系数字典
        """
        self.DEFAULT_COEFFICIENTS = {
            'imaging': 0.6,          # 成像功耗60%
            'downlink': 0.4,         # 数传功耗40%
            'slew': 0.3,             # 姿态调整功耗30%
            'idle': 0.05,            # 空闲功耗5%
            'charging': 0.0,         # 充电时净功耗（由光照决定）
        }

        self.coefficients = custom_coefficients or self.DEFAULT_COEFFICIENTS.copy()

    def get_coefficient(self, activity: str) -> float:
        """
        获取指定活动的功率系数

        Args:
            activity: 活动类型 ('imaging', 'downlink', 'slew', 'idle')

        Returns:
            float: 功率系数（0-1之间）
        """
        return self.coefficients.get(activity, 0.1)

    def get_coefficient_for_mode(self, mode: ImagingMode) -> float:
        """
        获取特定成像模式的功率系数

        不同成像模式可能有不同的功耗特征：
        - SPOTLIGHT: 高功耗（持续高功率发射/接收）
        - SLIDING_SPOTLIGHT: 中等功耗
        - STRIPMAP: 中等功耗
        - PUSH_BROOM: 较低功耗
        - FRAME: 低功耗（单次曝光）

        Args:
            mode: 成像模式

        Returns:
            float: 功率系数
        """
        base_coefficient = self.coefficients.get('imaging', 0.6)

        # 根据成像模式调整
        mode_multipliers = {
            ImagingMode.SPOTLIGHT: 1.2,          # 聚束模式功耗高20%
            ImagingMode.SLIDING_SPOTLIGHT: 1.1,  # 滑动聚束功耗高10%
            ImagingMode.STRIPMAP: 1.0,           # 条带模式基准功耗
            ImagingMode.PUSH_BROOM: 0.9,         # 推扫模式功耗低10%
            ImagingMode.FRAME: 0.7,              # 框幅模式功耗低30%
        }

        multiplier = mode_multipliers.get(mode, 1.0)
        return min(base_coefficient * multiplier, 1.0)


class ImagingTimeCalculator:
    """
    成像时间计算器

    根据目标类型、大小和成像模式计算所需的成像时长
    """

    # 默认成像参数
    DEFAULT_MIN_DURATION = 60       # 最小成像时长（秒）
    DEFAULT_MAX_DURATION = 1800     # 最大成像时长（秒）
    DEFAULT_DEFAULT_DURATION = 300  # 默认成像时长（秒）

    # 区域目标成像参数
    SQKM_TO_DURATION_FACTOR = 5.0   # 每平方公里需要的成像秒数
    SWATH_WIDTH_M = 10000.0         # 默认幅宽（米）
    SATELLITE_VELOCITY_MPS = 7500.0 # 卫星速度（米/秒）

    def __init__(self,
                 min_duration: float = None,
                 max_duration: float = None,
                 default_duration: float = None):
        """
        初始化成像时间计算器

        Args:
            min_duration: 最小成像时长（秒）
            max_duration: 最大成像时长（秒）
            default_duration: 默认成像时长（秒）
        """
        self.min_duration = min_duration or self.DEFAULT_MIN_DURATION
        self.max_duration = max_duration or self.DEFAULT_MAX_DURATION
        self.default_duration = default_duration or self.DEFAULT_DEFAULT_DURATION

    def calculate(self, target: Target, mode: ImagingMode) -> float:
        """
        计算成像时长

        Args:
            target: 目标对象
            mode: 成像模式

        Returns:
            float: 成像时长（秒）

        Raises:
            ValueError: 如果成像模式无效
        """
        if not isinstance(mode, ImagingMode):
            raise ValueError(f"Invalid imaging mode: {mode}")

        # 根据目标类型和成像模式计算时长
        if target.target_type == TargetType.POINT:
            duration = self._calculate_point_target(target, mode)
        elif target.target_type == TargetType.AREA:
            duration = self._calculate_area_target(target, mode)
        else:
            duration = self.default_duration

        # 应用限制
        return max(self.min_duration, min(duration, self.max_duration))

    def _calculate_point_target(self, target: Target, mode: ImagingMode) -> float:
        """
        计算点目标成像时长

        点目标通常成像时间较短，主要取决于：
        - 成像模式（框幅模式最快，聚束模式较慢）
        - 卫星过境时间

        Args:
            target: 点目标
            mode: 成像模式

        Returns:
            float: 成像时长（秒）
        """
        # 基础时长
        base_duration = self.default_duration

        # 根据成像模式调整
        mode_factors = {
            ImagingMode.FRAME: 0.5,              # 框幅模式：快速单次曝光
            ImagingMode.PUSH_BROOM: 1.0,         # 推扫模式：标准时长
            ImagingMode.STRIPMAP: 1.2,           # 条带模式：稍长
            ImagingMode.SLIDING_SPOTLIGHT: 1.5,  # 滑动聚束：较长
            ImagingMode.SPOTLIGHT: 2.0,          # 聚束模式：最长（需要更多曝光时间）
        }

        factor = mode_factors.get(mode, 1.0)
        return base_duration * factor

    def _calculate_area_target(self, target: Target, mode: ImagingMode) -> float:
        """
        计算区域目标成像时长

        区域目标成像时长主要取决于：
        - 区域面积
        - 成像模式效率
        - 卫星幅宽

        Args:
            target: 区域目标
            mode: 成像模式

        Returns:
            float: 成像时长（秒）
        """
        # 获取区域面积（平方公里）
        area_sqkm = target.get_area() if hasattr(target, 'get_area') else 100.0

        # 根据成像模式选择扫描策略
        if mode in [ImagingMode.SPOTLIGHT, ImagingMode.SLIDING_SPOTLIGHT]:
            # 聚束模式：逐点扫描，效率较低
            duration = self._calculate_spotlight_duration(area_sqkm)
        elif mode == ImagingMode.STRIPMAP:
            # 条带模式：连续扫描，效率最高
            duration = self._calculate_stripmap_duration(area_sqkm)
        elif mode == ImagingMode.PUSH_BROOM:
            # 推扫模式：适合大范围区域
            duration = self._calculate_pushbroom_duration(area_sqkm)
        elif mode == ImagingMode.FRAME:
            # 框幅模式：需要多次拍摄拼接
            duration = self._calculate_frame_duration(area_sqkm)
        else:
            duration = self.default_duration

        return duration

    def _calculate_spotlight_duration(self, area_sqkm: float) -> float:
        """
        计算聚束模式成像时长

        聚束模式通过控制天线波束指向来获取高分辨率图像，
        但覆盖效率较低。

        Args:
            area_sqkm: 区域面积（平方公里）

        Returns:
            float: 成像时长（秒）
        """
        # 聚束模式效率较低，每平方公里需要更多时间
        # 假设每次聚束观测覆盖约10平方公里
        coverage_per_shot = 10.0  # 平方公里
        shots_needed = max(1, area_sqkm / coverage_per_shot)

        # 每次观测约60秒（包括姿态调整）
        time_per_shot = 60.0
        total_time = shots_needed * time_per_shot

        return total_time

    def _calculate_stripmap_duration(self, area_sqkm: float) -> float:
        """
        计算条带模式成像时长

        条带模式是最高效的扫描方式，卫星沿飞行方向连续成像。

        Args:
            area_sqkm: 区域面积（平方公里）

        Returns:
            float: 成像时长（秒）
        """
        # 简化为线性扫描模型
        # 幅宽10km，速度7.5km/s
        # 每平方公里需要的时间 = 面积 / (幅宽 * 速度)
        swath_km = self.SWATH_WIDTH_M / 1000.0  # 转换为km
        velocity_kmps = self.SATELLITE_VELOCITY_MPS / 1000.0

        # 扫描长度 = 面积 / 幅宽
        scan_length_km = area_sqkm / swath_km

        # 扫描时间 = 长度 / 速度
        duration = scan_length_km / velocity_kmps

        # 添加姿态调整时间（每次约30秒）
        num_passes = max(1, int(area_sqkm / 500))  # 假设每次过境最多覆盖500平方公里
        setup_time = num_passes * 30.0

        return duration + setup_time

    def _calculate_pushbroom_duration(self, area_sqkm: float) -> float:
        """
        计算推扫模式成像时长

        推扫模式是光学卫星常用的扫描方式。

        Args:
            area_sqkm: 区域面积（平方公里）

        Returns:
            float: 成像时长（秒）
        """
        # 推扫模式与条带模式类似，但可能稍慢
        stripmap_time = self._calculate_stripmap_duration(area_sqkm)
        return stripmap_time * 1.1  # 增加10%时间

    def _calculate_frame_duration(self, area_sqkm: float) -> float:
        """
        计算框幅模式成像时长

        框幅模式需要多次拍摄并拼接，效率最低。

        Args:
            area_sqkm: 区域面积（平方公里）

        Returns:
            float: 成像时长（秒）
        """
        # 假设每帧覆盖约5平方公里（包含重叠）
        coverage_per_frame = 5.0
        frames_needed = max(1, area_sqkm / coverage_per_frame)

        # 每帧约10秒（包括稳定时间）
        time_per_frame = 10.0
        total_time = frames_needed * time_per_frame

        return total_time

    def get_power_consumption(self,
                              target: Target,
                              mode: ImagingMode,
                              power_capacity: float,
                              profile: PowerProfile = None) -> float:
        """
        计算成像功率消耗

        Args:
            target: 目标对象
            mode: 成像模式
            power_capacity: 卫星功率容量（Wh）
            profile: 功率配置文件（可选）

        Returns:
            float: 功率消耗（Wh）
        """
        if profile is None:
            profile = PowerProfile()

        duration_hours = self.calculate(target, mode) / 3600.0
        coefficient = profile.get_coefficient_for_mode(mode)

        return power_capacity * coefficient * duration_hours
