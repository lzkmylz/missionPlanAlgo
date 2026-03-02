"""
发电功率计算器

基于余弦衰减模型计算卫星发电功率。

功能：
1. 根据姿态模式计算发电功率
2. 根据地影判断调整功率
3. 计算太阳帆板与太阳光的夹角余弦

功率计算公式：
    P = P_max × cos(θ) × eclipse_factor

其中：
- P_max: 帆板正对太阳时的最大功率（W）
- θ: 太阳矢量与帆板法向的夹角
- cos(θ): 点积(panel_normal, sun_direction)
- eclipse_factor: 地影因子（地影期为0，日照期为1）
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Optional

from core.dynamics.attitude_mode import AttitudeMode

logger = logging.getLogger(__name__)

# 地球半径（米）
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class PowerConfig:
    """
    功率配置参数

    Attributes:
        max_power: 帆板正对太阳时的最大功率（W）
        eclipse_power: 地影期间的功率（W）
    """
    max_power: float = 1000.0
    eclipse_power: float = 0.0


class PowerGenerationCalculator:
    """
    发电功率计算器

    根据卫星姿态模式和太阳位置计算发电功率。
    使用余弦衰减模型：当帆板不直接对准时，功率随角度余弦衰减。

    Attributes:
        sun_calculator: 太阳位置计算器
        config: 功率配置参数
    """

    def __init__(self, sun_calculator, config: Optional[PowerConfig] = None):
        """
        初始化发电功率计算器

        Args:
            sun_calculator: 太阳位置计算器（必须提供）
            config: 功率配置参数（可选，默认使用PowerConfig()）

        Raises:
            TypeError: 如果sun_calculator为None
        """
        if sun_calculator is None:
            raise TypeError("sun_calculator cannot be None")

        self.sun_calculator = sun_calculator
        self.config = config if config is not None else PowerConfig()

    def _validate_timestamp(self, timestamp: datetime) -> datetime:
        """
        验证时间戳有效性

        Args:
            timestamp: 输入时间

        Returns:
            datetime: 验证后的UTC时间

        Raises:
            TypeError: 时间戳类型无效
            ValueError: 时间戳无时区信息
        """
        if timestamp is None:
            raise TypeError("Timestamp cannot be None")

        if not isinstance(timestamp, datetime):
            raise TypeError(f"Expected datetime, got {type(timestamp).__name__}")

        if timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")

        return timestamp.astimezone(timezone.utc)

    def _validate_satellite_position(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        验证卫星位置有效性

        Args:
            position: 卫星位置坐标

        Returns:
            Tuple[float, float, float]: 验证后的位置

        Raises:
            TypeError: 位置类型无效
            ValueError: 位置格式错误
        """
        if not isinstance(position, (tuple, list)):
            raise TypeError(f"Expected tuple or list, got {type(position).__name__}")

        if len(position) != 3:
            raise ValueError(f"Expected 3 coordinates, got {len(position)}")

        try:
            return (float(position[0]), float(position[1]), float(position[2]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid coordinate values: {e}")

    def _validate_sun_direction(self, sun_direction: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        验证太阳方向向量有效性

        Args:
            sun_direction: 太阳方向向量

        Returns:
            Tuple[float, float, float]: 验证后的单位方向向量

        Raises:
            TypeError: 类型无效
            ValueError: 格式错误或零向量
        """
        if not isinstance(sun_direction, (tuple, list)):
            raise TypeError(f"Expected tuple or list, got {type(sun_direction).__name__}")

        if len(sun_direction) != 3:
            raise ValueError(f"Expected 3 components, got {len(sun_direction)}")

        try:
            x, y, z = float(sun_direction[0]), float(sun_direction[1]), float(sun_direction[2])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid direction values: {e}")

        magnitude = math.sqrt(x*x + y*y + z*z)
        if magnitude < 1e-10:
            raise ValueError("Sun direction vector cannot be zero")

        return (x/magnitude, y/magnitude, z/magnitude)

    def _validate_attitude_mode(self, attitude_mode) -> AttitudeMode:
        """
        验证姿态模式有效性

        Args:
            attitude_mode: 姿态模式

        Returns:
            AttitudeMode: 验证后的姿态模式

        Raises:
            TypeError: 类型无效
        """
        if not isinstance(attitude_mode, AttitudeMode):
            raise TypeError(f"Expected AttitudeMode, got {type(attitude_mode).__name__}")
        return attitude_mode

    def is_in_eclipse(self, satellite_position: Tuple[float, float, float],
                     timestamp: datetime) -> bool:
        """
        判断卫星是否在地影中

        地影判断基于卫星、地球和太阳的相对几何关系。
        当卫星位于地球背向太阳的一侧时，处于地影中。

        Args:
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间（必须带时区信息）

        Returns:
            bool: 如果卫星在地影中返回True

        Raises:
            TypeError: 输入类型无效
            ValueError: 输入格式错误
        """
        # 验证输入
        sat_pos = self._validate_satellite_position(satellite_position)
        utc_time = self._validate_timestamp(timestamp)

        # 获取太阳位置
        sun_pos = self.sun_calculator.get_sun_position(utc_time)

        # 计算从卫星指向太阳和指向地心的向量
        # 卫星到太阳向量
        to_sun = (sun_pos[0] - sat_pos[0], sun_pos[1] - sat_pos[1], sun_pos[2] - sat_pos[2])

        # 卫星到地心向量（即位置向量的反方向）
        to_earth_center = (-sat_pos[0], -sat_pos[1], -sat_pos[2])

        # 归一化
        to_sun_mag = math.sqrt(to_sun[0]**2 + to_sun[1]**2 + to_sun[2]**2)
        to_earth_mag = math.sqrt(to_earth_center[0]**2 + to_earth_center[1]**2 + to_earth_center[2]**2)

        if to_sun_mag < 1e-10 or to_earth_mag < 1e-10:
            return False

        # 计算点积：如果为正，说明太阳和地心在卫星的同一侧
        dot_product = (to_sun[0] * to_earth_center[0] +
                      to_sun[1] * to_earth_center[1] +
                      to_sun[2] * to_earth_center[2])

        cos_angle = dot_product / (to_sun_mag * to_earth_mag)

        # 如果cos_angle > 0，太阳和地心在卫星同一侧
        # 进一步检查卫星是否被地球遮挡
        if cos_angle > 0:
            # 计算卫星到太阳-地心连线的垂直距离
            # 使用叉积计算距离
            cross_x = to_sun[1] * to_earth_center[2] - to_sun[2] * to_earth_center[1]
            cross_y = to_sun[2] * to_earth_center[0] - to_sun[0] * to_earth_center[2]
            cross_z = to_sun[0] * to_earth_center[1] - to_sun[1] * to_earth_center[0]
            cross_mag = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

            # 距离 = |cross| / |to_sun|
            distance_to_line = cross_mag / to_sun_mag

            # 如果距离小于地球半径，卫星在地影中
            if distance_to_line < EARTH_RADIUS_M:
                return True

        return False

    def calculate_cosine_factor(self, sun_direction: Tuple[float, float, float],
                               attitude_mode: AttitudeMode,
                               roll_angle: float = 0.0,
                               pitch_angle: float = 0.0) -> float:
        """
        计算余弦因子（太阳方向与帆板法向的夹角余弦）

        不同姿态模式下，帆板法向的计算方式不同：
        - SUN_POINTING: 帆板法向与太阳方向对齐（cos = 1.0）
        - NADIR_POINTING: 帆板法向垂直于星下点方向
        - IMAGING/DOWNLINK/REALTIME: 帆板法向根据滚转/俯仰角调整
        - MOMENTUM_DUMP: 通常采用对日定向

        Args:
            sun_direction: 太阳方向单位向量（指向太阳）
            attitude_mode: 姿态模式
            roll_angle: 滚转角（度），用于IMAGING/DOWNLINK/REALTIME模式
            pitch_angle: 俯仰角（度），用于IMAGING/DOWNLINK/REALTIME模式

        Returns:
            float: 余弦因子 [0, 1]

        Raises:
            TypeError: 输入类型无效
            ValueError: 输入格式错误
        """
        # 验证输入
        sun_dir = self._validate_sun_direction(sun_direction)
        mode = self._validate_attitude_mode(attitude_mode)

        # 根据姿态模式计算帆板法向
        if mode == AttitudeMode.SUN_POINTING:
            # 对日定向：帆板法向与太阳方向对齐
            panel_normal = sun_dir

        elif mode == AttitudeMode.MOMENTUM_DUMP:
            # 动量卸载：通常采用对日定向
            panel_normal = sun_dir

        elif mode == AttitudeMode.NADIR_POINTING:
            # 对地定向：帆板法向在轨道面内，垂直于径向
            # 假设帆板法向可以指向太阳在轨道面内的投影方向
            # 简化处理：假设卫星在x轴上，帆板法向在y-z平面内
            # 实际应该根据轨道参数计算，这里简化为取太阳方向在y-z平面的投影
            # 对于对地定向，帆板通常沿轨道速度方向展开
            # 这里我们假设帆板法向是太阳方向在垂直于径向的平面上的投影
            # 简化模型：假设卫星位置在x方向，帆板法向可以绕x轴旋转
            # 最优情况是帆板法向指向太阳在y-z平面的投影

            # 对于对地定向，我们假设帆板法向可以调整以最大化发电
            # 实际上卫星会绕径向轴旋转，使帆板对准太阳
            # 这里简化为取太阳方向矢量的绝对值（假设帆板可以双面发电或旋转）
            # 更准确的做法是计算太阳方向在垂直于径向平面上的投影

            # 简化处理：假设帆板法向可以指向太阳方向的y-z分量
            # 计算太阳方向在垂直于x轴平面上的投影
            # 假设卫星在x轴上（对地定向）
            x, y, z = sun_dir
            # 投影到y-z平面并归一化
            proj_mag = math.sqrt(y*y + z*z)
            if proj_mag > 1e-10:
                panel_normal = (0.0, y/proj_mag, z/proj_mag)
            else:
                # 太阳正好在径向方向，帆板边缘对日
                panel_normal = (0.0, 1.0, 0.0)

        elif mode in (AttitudeMode.IMAGING, AttitudeMode.DOWNLINK, AttitudeMode.REALTIME):
            # 成像/数传/实传模式：根据滚转和俯仰角调整
            # 基础姿态是对地定向，然后根据角度调整

            # 将角度转换为弧度
            roll_rad = math.radians(roll_angle)
            pitch_rad = math.radians(pitch_angle)

            # 假设基础对地定向时，帆板法向在y方向（轨道速度方向）
            # 滚转绕x轴，俯仰绕y轴
            # 基础帆板法向（对地定向时）
            base_normal = (0.0, 1.0, 0.0)

            # 应用滚转（绕x轴）
            # [1, 0, 0]
            # [0, cos(roll), -sin(roll)]
            # [0, sin(roll), cos(roll)]
            x1 = base_normal[0]
            y1 = base_normal[1] * math.cos(roll_rad) - base_normal[2] * math.sin(roll_rad)
            z1 = base_normal[1] * math.sin(roll_rad) + base_normal[2] * math.cos(roll_rad)

            # 应用俯仰（绕y轴）
            # [cos(pitch), 0, sin(pitch)]
            # [0, 1, 0]
            # [-sin(pitch), 0, cos(pitch)]
            x2 = x1 * math.cos(pitch_rad) + z1 * math.sin(pitch_rad)
            y2 = y1
            z2 = -x1 * math.sin(pitch_rad) + z1 * math.cos(pitch_rad)

            # 归一化
            mag = math.sqrt(x2*x2 + y2*y2 + z2*z2)
            if mag > 1e-10:
                panel_normal = (x2/mag, y2/mag, z2/mag)
            else:
                panel_normal = (0.0, 1.0, 0.0)

        else:
            # 未知模式，默认对日定向
            logger.warning(f"Unknown attitude mode: {mode}, defaulting to sun pointing")
            panel_normal = sun_dir

        # 计算余弦因子（点积）
        cosine = (panel_normal[0] * sun_dir[0] +
                 panel_normal[1] * sun_dir[1] +
                 panel_normal[2] * sun_dir[2])

        # 限制在[0, 1]范围内（帆板通常不能从背面发电）
        return max(0.0, min(1.0, cosine))

    def calculate_power(self, attitude_mode: AttitudeMode,
                       satellite_position: Tuple[float, float, float],
                       timestamp: datetime,
                       roll_angle: float = 0.0,
                       pitch_angle: float = 0.0) -> float:
        """
        计算发电功率

        计算公式：P = P_max × cos(θ) × eclipse_factor

        Args:
            attitude_mode: 姿态模式
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间（必须带时区信息）
            roll_angle: 滚转角（度），用于IMAGING/DOWNLINK/REALTIME模式
            pitch_angle: 俯仰角（度），用于IMAGING/DOWNLINK/REALTIME模式

        Returns:
            float: 发电功率（W）

        Raises:
            TypeError: 输入类型无效
            ValueError: 输入格式错误
        """
        # 验证输入
        mode = self._validate_attitude_mode(attitude_mode)
        sat_pos = self._validate_satellite_position(satellite_position)
        utc_time = self._validate_timestamp(timestamp)

        # 检查是否在地影中
        if self.is_in_eclipse(sat_pos, utc_time):
            return self.config.eclipse_power

        # 获取太阳方向
        sun_dir = self.sun_calculator.get_sun_direction(sat_pos, utc_time)

        # 计算余弦因子
        cosine_factor = self.calculate_cosine_factor(
            sun_dir, mode, roll_angle, pitch_angle
        )

        # 计算功率
        power = self.config.max_power * cosine_factor

        return power
