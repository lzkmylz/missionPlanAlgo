"""
姿态切换计算器

计算卫星在不同姿态模式之间的切换时间、角度和机动参数。

功能：
1. 计算对日定向姿态角（滚转、俯仰）
2. 计算对地定向姿态角（始终为0, 0）
3. 计算成像姿态角（基于目标位置）
4. 计算姿态切换的机动时间和角度

机动时间公式：
    slew_time = max(|Δroll|, |Δpitch|) / max_slew_rate + settling_time

机动角度公式：
    slew_angle = sqrt(Δroll² + Δpitch²)
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition, TransitionResult
from core.dynamics.sun_position_calculator import SunPositionCalculator

logger = logging.getLogger(__name__)


@dataclass
class TransitionConfig:
    """
    姿态切换配置参数

    Attributes:
        max_slew_rate: 最大机动角速度（度/秒）
        settling_time: 姿态稳定时间（秒）
    """
    max_slew_rate: float = 3.0  # deg/s
    settling_time: float = 5.0  # seconds


class AttitudeTransitionCalculator:
    """
    姿态切换计算器

    计算不同姿态模式之间的切换参数，包括机动时间、角度等。

    Attributes:
        sun_calculator: 太阳位置计算器
        config: 切换配置参数
    """

    # 地球半径（米）
    EARTH_RADIUS = 6371000.0

    def __init__(
        self,
        sun_calculator: SunPositionCalculator,
        config: Optional[TransitionConfig] = None
    ):
        """
        初始化姿态切换计算器

        Args:
            sun_calculator: 太阳位置计算器，用于计算对日定向姿态
            config: 切换配置参数，如果为None则使用默认配置

        Raises:
            TypeError: 如果sun_calculator为None
        """
        if sun_calculator is None:
            raise TypeError("sun_calculator cannot be None")

        self.sun_calculator = sun_calculator
        self.config = config if config is not None else TransitionConfig()

    def _validate_satellite_position(
        self,
        position: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
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

    def _validate_target_position(
        self,
        position: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        验证目标位置有效性

        Args:
            position: 目标位置（纬度、经度）

        Returns:
            Tuple[float, float]: 验证后的位置

        Raises:
            TypeError: 位置类型无效
            ValueError: 位置格式错误
        """
        if position is None:
            raise ValueError("Target position cannot be None")

        if not isinstance(position, (tuple, list)):
            raise TypeError(f"Expected tuple or list, got {type(position).__name__}")

        if len(position) != 2:
            raise ValueError(f"Expected 2 coordinates (lat, lon), got {len(position)}")

        try:
            return (float(position[0]), float(position[1]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid coordinate values: {e}")

    def calculate_nadir_pointing_angles(self) -> Tuple[float, float]:
        """
        计算对地定向姿态角

        对地定向时，卫星Z轴指向地心，滚转角和俯仰角均为0。

        Returns:
            Tuple[float, float]: (roll, pitch) in degrees，始终为 (0, 0)
        """
        return (0.0, 0.0)

    def calculate_sun_pointing_angles(
        self,
        sat_pos: Tuple[float, float, float],
        timestamp: datetime
    ) -> Tuple[float, float]:
        """
        计算对日定向姿态角

        基于太阳相对于卫星的位置，计算需要的滚转角和俯仰角。

        Args:
            sat_pos: 卫星ECEF位置（米）
            timestamp: UTC时间

        Returns:
            Tuple[float, float]: (roll, pitch) in degrees

        Raises:
            TypeError: 输入类型无效
            ValueError: 输入格式错误
        """
        # 验证输入
        sat_pos = self._validate_satellite_position(sat_pos)

        # 获取从卫星指向太阳的单位方向向量
        sun_direction = self.sun_calculator.get_sun_direction(sat_pos, timestamp)

        # 计算姿态角
        # 在LVLH坐标系中：
        # - Z轴指向地心（负位置方向）
        # - X轴沿飞行方向
        # - Y轴完成右手系
        #
        # 对日定向时，需要计算太阳方向与Z轴的夹角
        # Roll: 绕X轴旋转，由Y和Z分量决定
        # Pitch: 绕Y轴旋转，由X和Z分量决定

        sx, sy, sz = sun_direction

        # 卫星位置单位向量（指向地心的反方向，即天顶方向）
        px, py, pz = sat_pos
        r = math.sqrt(px*px + py*py + pz*pz)
        if r < 1e-10:
            raise ValueError("Invalid satellite position (near origin)")

        # 天顶方向（负位置方向归一化）
        zx, zy, zz = -px/r, -py/r, -pz/r

        # 计算太阳方向与天顶方向的夹角
        # 使用球面几何计算滚转和俯仰角

        # 在轨道平面坐标系中，计算太阳的方位
        # 简化计算：假设太阳方向在卫星本体坐标系中的投影

        # 构建一个简化的本体坐标系
        # Z轴：指向地心（负位置方向）
        # X轴：沿速度方向（这里简化处理，使用位置与极轴的叉积）
        # Y轴：完成右手系

        # 简化计算：使用太阳方向与卫星位置的关系
        # 当太阳方向与卫星位置垂直时，需要90度侧摆

        # 太阳方向与位置向量的点积（太阳在天顶方向的投影）
        dot_product = sx * zx + sy * zy + sz * zz

        # 太阳在垂直于天顶方向的平面上的投影
        # 计算垂直于Z轴的分量
        perp_x = sx - dot_product * zx
        perp_y = sy - dot_product * zy
        perp_z = sz - dot_product * zz

        perp_magnitude = math.sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z)

        if perp_magnitude < 1e-10:
            # 太阳在天顶或天底方向
            roll = 0.0
            pitch = math.degrees(math.acos(max(-1.0, min(1.0, dot_product))))
            if dot_product < 0:
                pitch = 180.0 - pitch
            return (roll, pitch)

        # 归一化垂直分量
        perp_x /= perp_magnitude
        perp_y /= perp_magnitude
        perp_z /= perp_magnitude

        # 计算滚转角（绕X轴）
        # 简化：使用Y-Z平面投影
        roll = math.degrees(math.atan2(perp_y, -dot_product))

        # 计算俯仰角（绕Y轴）
        pitch = math.degrees(math.atan2(perp_x, -dot_product))

        # 限制角度范围
        roll = max(-180.0, min(180.0, roll))
        pitch = max(-180.0, min(180.0, pitch))

        return (roll, pitch)

    def calculate_imaging_angles(
        self,
        sat_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        计算成像姿态角

        基于卫星位置和目标位置，计算成像所需的滚转角和俯仰角。
        使用与AttitudeCalculator类似的逻辑。

        Args:
            sat_pos: 卫星ECEF位置（米）
            target_pos: 目标位置（纬度、经度）in degrees

        Returns:
            Tuple[float, float]: (roll, pitch) in degrees

        Raises:
            TypeError: 输入类型无效
            ValueError: 输入格式错误
        """
        # 验证输入
        sat_pos = self._validate_satellite_position(sat_pos)
        target_pos = self._validate_target_position(target_pos)

        # 将目标地理坐标转换为ECEF
        target_ecef = self._geodetic_to_ecef(target_pos[0], target_pos[1], 0.0)

        # 计算从卫星到目标的视线向量
        los = (
            target_ecef[0] - sat_pos[0],
            target_ecef[1] - sat_pos[1],
            target_ecef[2] - sat_pos[2]
        )

        # 归一化视线向量
        los_norm = math.sqrt(los[0]**2 + los[1]**2 + los[2]**2)
        if los_norm < 1e-10:
            raise ValueError("Target is too close to satellite")

        los = (los[0]/los_norm, los[1]/los_norm, los[2]/los_norm)

        # 计算卫星位置单位向量（指向地心的反方向，即天顶）
        px, py, pz = sat_pos
        r = math.sqrt(px*px + py*py + pz*pz)
        if r < 1e-10:
            raise ValueError("Invalid satellite position")

        # 天顶方向（卫星本体Z轴指向）
        zx, zy, zz = -px/r, -py/r, -pz/r

        # 计算视线与天顶方向的夹角
        dot_product = los[0]*zx + los[1]*zy + los[2]*zz

        # 视线在垂直于天顶方向的平面上的投影
        perp_x = los[0] - dot_product * zx
        perp_y = los[1] - dot_product * zy
        perp_z = los[2] - dot_product * zz

        perp_magnitude = math.sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z)

        if perp_magnitude < 1e-10:
            # 目标在天顶或天底方向
            roll = 0.0
            pitch = 0.0 if dot_product > 0 else 180.0
            return (roll, pitch)

        # 归一化
        perp_x /= perp_magnitude
        perp_y /= perp_magnitude
        perp_z /= perp_magnitude

        # 构建LVLH坐标系的简化版本
        # 计算滚转角（绕X轴）和俯仰角（绕Y轴）
        # 使用简化几何计算

        # 计算方位角（在水平面上的投影）
        # 使用叉积确定方向

        # 简化计算：使用视线向量与天顶方向的夹角
        elevation = math.acos(max(-1.0, min(1.0, dot_product)))

        # 计算方位（在水平面上的方向）
        # 使用位置向量与视线垂直分量的关系
        # 叉积给出水平面上的方向
        cross_x = zy * perp_z - zz * perp_y
        cross_y = zz * perp_x - zx * perp_z
        cross_z = zx * perp_y - zy * perp_x

        cross_magnitude = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        if cross_magnitude > 1e-10:
            # 有水平分量，计算方位角
            azimuth = math.atan2(cross_y, cross_x)

            # 从方位角和高度角计算滚转和俯仰
            # Roll: 侧摆角（左右）
            # Pitch: 俯仰角（前后）
            roll = math.degrees(math.sin(azimuth) * elevation)
            pitch = math.degrees(math.cos(azimuth) * elevation)
        else:
            # 无水平分量，目标在天顶
            roll = 0.0
            pitch = 0.0

        return (roll, pitch)

    def _geodetic_to_ecef(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> Tuple[float, float, float]:
        """
        地理坐标转换为ECEF坐标

        使用简化球体模型。

        Args:
            lat: 纬度（度）
            lon: 经度（度）
            alt: 海拔高度（米）

        Returns:
            ECEF坐标 (x, y, z) in meters
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # 简化：使用地球平均半径
        r = self.EARTH_RADIUS + alt

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def _calculate_slew_parameters(
        self,
        from_angles: Tuple[float, float],
        to_angles: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """
        计算机动参数

        Args:
            from_angles: 起始姿态角 (roll, pitch)
            to_angles: 目标姿态角 (roll, pitch)

        Returns:
            Tuple[float, float, float, float]: (slew_time, slew_angle, delta_roll, delta_pitch)
        """
        delta_roll = to_angles[0] - from_angles[0]
        delta_pitch = to_angles[1] - from_angles[1]

        # 机动角度 = sqrt(Δroll² + Δpitch²)
        slew_angle = math.sqrt(delta_roll**2 + delta_pitch**2)

        # 机动时间 = max(|Δroll|, |Δpitch|) / max_slew_rate + settling_time
        # NOTE: 使用max()假设卫星可以同时绕滚转和俯仰轴旋转
        # （敏捷卫星具有多轴机动能力）。如果只能顺序机动，应使用sum()或欧几里得范数。
        max_delta = max(abs(delta_roll), abs(delta_pitch))
        slew_time = max_delta / self.config.max_slew_rate + self.config.settling_time

        return (slew_time, slew_angle, delta_roll, delta_pitch)

    def calculate_transition(self, transition: AttitudeTransition) -> TransitionResult:
        """
        计算姿态切换参数

        根据切换请求，计算机动时间、角度等参数。

        Args:
            transition: 姿态切换请求

        Returns:
            TransitionResult: 切换计算结果
        """
        from_mode = transition.from_mode
        to_mode = transition.to_mode

        # 相同模式，无需机动
        if from_mode == to_mode:
            return TransitionResult(
                slew_time=0.0,
                slew_angle=0.0,
                roll_angle=0.0,
                pitch_angle=0.0,
                power_generation=0.0,  # TODO: 实现功率计算
                feasible=True
            )

        try:
            # 计算起始姿态角
            if from_mode == AttitudeMode.NADIR_POINTING:
                from_angles = self.calculate_nadir_pointing_angles()
            elif from_mode == AttitudeMode.SUN_POINTING:
                from_angles = self.calculate_sun_pointing_angles(
                    transition.satellite_position,
                    transition.timestamp
                )
            elif from_mode == AttitudeMode.IMAGING:
                if transition.target_position is None:
                    return TransitionResult(
                        slew_time=0.0,
                        slew_angle=0.0,
                        roll_angle=0.0,
                        pitch_angle=0.0,
                        power_generation=0.0,
                        feasible=False,
                        reason="Target position required for IMAGING mode"
                    )
                from_angles = self.calculate_imaging_angles(
                    transition.satellite_position,
                    transition.target_position
                )
            else:
                # 其他模式暂不支持，返回不可行
                return TransitionResult(
                    slew_time=0.0,
                    slew_angle=0.0,
                    roll_angle=0.0,
                    pitch_angle=0.0,
                    power_generation=0.0,
                    feasible=False,
                    reason=f"Mode {from_mode} not yet supported"
                )

            # 计算目标姿态角
            if to_mode == AttitudeMode.NADIR_POINTING:
                to_angles = self.calculate_nadir_pointing_angles()
            elif to_mode == AttitudeMode.SUN_POINTING:
                to_angles = self.calculate_sun_pointing_angles(
                    transition.satellite_position,
                    transition.timestamp
                )
            elif to_mode == AttitudeMode.IMAGING:
                if transition.target_position is None:
                    return TransitionResult(
                        slew_time=0.0,
                        slew_angle=0.0,
                        roll_angle=0.0,
                        pitch_angle=0.0,
                        power_generation=0.0,
                        feasible=False,
                        reason="Target position required for IMAGING mode"
                    )
                to_angles = self.calculate_imaging_angles(
                    transition.satellite_position,
                    transition.target_position
                )
            else:
                # 其他模式暂不支持
                return TransitionResult(
                    slew_time=0.0,
                    slew_angle=0.0,
                    roll_angle=0.0,
                    pitch_angle=0.0,
                    power_generation=0.0,
                    feasible=False,
                    reason=f"Mode {to_mode} not yet supported"
                )

            # 计算机动参数
            slew_time, slew_angle, delta_roll, delta_pitch = self._calculate_slew_parameters(
                from_angles, to_angles
            )

            return TransitionResult(
                slew_time=slew_time,
                slew_angle=slew_angle,
                roll_angle=to_angles[0],  # 目标滚转角
                pitch_angle=to_angles[1],  # 目标俯仰角
                power_generation=0.0,  # TODO: 实现功率计算
                feasible=True
            )

        except Exception as e:
            logger.error(f"Error calculating transition: {e}")
            return TransitionResult(
                slew_time=0.0,
                slew_angle=0.0,
                roll_angle=0.0,
                pitch_angle=0.0,
                power_generation=0.0,
                feasible=False,
                reason=str(e)
            )
