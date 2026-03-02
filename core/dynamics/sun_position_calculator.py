"""
太阳位置计算器

基于Orekit集成或简化模型计算太阳在ECEF坐标系中的位置。
用于姿态管理系统中的对日定向计算。

功能：
1. 计算太阳在ECEF坐标系中的位置
2. 计算从卫星位置指向太阳的方向向量
3. 支持Orekit高精度计算和简化模型回退
"""

import math
import logging
from datetime import datetime, timezone
from typing import Tuple, Optional

# 尝试导入OrekitJavaBridge
try:
    from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge
    OREKIT_BRIDGE_AVAILABLE = True
except ImportError:
    OREKIT_BRIDGE_AVAILABLE = False
    OrekitJavaBridge = None

logger = logging.getLogger(__name__)

# 天文常数
AU_METERS = 149_597_870_700.0  # 1天文单位（米）
EARTH_RADIUS_M = 6_371_000.0  # 地球半径（米）

# 简化模型常数（基于VSOP87简化）
# 太阳平均运动（弧度/天）
MEAN_MOTION_SUN = 2 * math.pi / 365.25
# J2000.0历元
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class SunPositionCalculator:
    """
    太阳位置计算器

    计算太阳在ECEF坐标系中的位置和方向向量。
    支持Orekit高精度计算，当Orekit不可用时自动回退到简化模型。

    Attributes:
        use_orekit: 是否尝试使用Orekit
        _orekit_bridge: OrekitJavaBridge实例（如果可用）
    """

    def __init__(self, use_orekit: bool = True):
        """
        初始化太阳位置计算器

        Args:
            use_orekit: 是否尝试使用Orekit进行计算（默认True）
        """
        self.use_orekit = use_orekit
        self._orekit_bridge: Optional[OrekitJavaBridge] = None

        # 如果启用Orekit且可用，尝试初始化桥接器
        if use_orekit and OREKIT_BRIDGE_AVAILABLE:
            try:
                self._orekit_bridge = OrekitJavaBridge()
                logger.info("OrekitJavaBridge initialized for SunPositionCalculator")
            except Exception as e:
                logger.warning(f"Failed to initialize OrekitJavaBridge: {e}")
                self._orekit_bridge = None

    def is_available(self) -> bool:
        """
        检查计算器是否可用

        Returns:
            bool: 如果可以使用Orekit或简化模型返回True
        """
        # 如果不使用Orekit，总是可用（简化模型）
        if not self.use_orekit:
            return True

        # 如果使用Orekit，检查桥接器是否可用
        if OREKIT_BRIDGE_AVAILABLE and self._orekit_bridge is not None:
            try:
                return self._orekit_bridge.is_jvm_running()
            except Exception:
                return False

        return False

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

        # 检查时区信息
        if timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")

        # 转换为UTC
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

    def _calculate_sun_position_simplified(self, timestamp: datetime) -> Tuple[float, float, float]:
        """
        使用简化模型计算太阳位置

        基于开普勒轨道近似，精度约0.1度。
        适用于对日定向等不需要极高精度的场景。

        Args:
            timestamp: UTC时间

        Returns:
            Tuple[float, float, float]: 太阳ECEF位置（米）
        """
        # 计算从J2000起的天数
        days_since_j2000 = (timestamp - J2000_EPOCH).total_seconds() / 86400.0

        # 太阳平黄经（弧度）
        # L = 280.460 + 0.9856474 * n （度）
        mean_longitude = math.radians(280.460 + 0.9856474 * days_since_j2000)

        # 太阳平近点角（弧度）
        # g = 357.528 + 0.9856003 * n （度）
        mean_anomaly = math.radians(357.528 + 0.9856003 * days_since_j2000)

        # 太阳黄经（弧度）
        # lambda = L + 1.915 * sin(g) + 0.020 * sin(2g) （度）
        ecliptic_longitude = mean_longitude + \
                            math.radians(1.915) * math.sin(mean_anomaly) + \
                            math.radians(0.020) * math.sin(2 * mean_anomaly)

        # 太阳黄纬（弧度，非常接近0）
        ecliptic_latitude = 0.0

        # 日地距离（天文单位）
        # r = 1.00014 - 0.01671 * cos(g) - 0.00014 * cos(2g)
        distance_au = 1.00014 - 0.01671 * math.cos(mean_anomaly) - 0.00014 * math.cos(2 * mean_anomaly)
        distance_m = distance_au * AU_METERS

        # 黄赤交角（弧度）
        obliquity = math.radians(23.439 - 0.0000004 * days_since_j2000)

        # 太阳在黄道坐标系中的位置
        x_ecliptic = distance_m * math.cos(ecliptic_longitude) * math.cos(ecliptic_latitude)
        y_ecliptic = distance_m * math.sin(ecliptic_longitude) * math.cos(ecliptic_latitude)
        z_ecliptic = distance_m * math.sin(ecliptic_latitude)

        # 转换到赤道坐标系（ECI）
        x_eci = x_ecliptic
        y_eci = y_ecliptic * math.cos(obliquity) - z_ecliptic * math.sin(obliquity)
        z_eci = y_ecliptic * math.sin(obliquity) + z_ecliptic * math.cos(obliquity)

        # 计算格林尼治平恒星时（GMST）
        # GMST = 280.46061837 + 360.98564736629 * days_since_j2000 （度）
        gmst = math.radians(280.46061837 + 360.98564736629 * days_since_j2000)

        # 从ECI转换到ECEF
        x_ecef = x_eci * math.cos(gmst) + y_eci * math.sin(gmst)
        y_ecef = -x_eci * math.sin(gmst) + y_eci * math.cos(gmst)
        z_ecef = z_eci

        return (x_ecef, y_ecef, z_ecef)

    def _calculate_sun_position_orekit(self, timestamp: datetime) -> Tuple[float, float, float]:
        """
        使用Orekit计算太阳位置

        Args:
            timestamp: UTC时间

        Returns:
            Tuple[float, float, float]: 太阳ECEF位置（米）

        Raises:
            RuntimeError: Orekit计算失败
        """
        if not OREKIT_BRIDGE_AVAILABLE or self._orekit_bridge is None:
            raise RuntimeError("Orekit not available")

        try:
            # 确保JVM已启动
            self._orekit_bridge._ensure_jvm_started()

            # 获取Orekit类
            AbsoluteDate = self._orekit_bridge._get_java_class("org.orekit.time.AbsoluteDate")
            TimeScalesFactory = self._orekit_bridge._get_java_class("org.orekit.time.TimeScalesFactory")
            FramesFactory = self._orekit_bridge._get_java_class("org.orekit.frames.FramesFactory")
            CelestialBodyFactory = self._orekit_bridge._get_java_class("org.orekit.bodies.CelestialBodyFactory")

            # 创建时间尺度
            utc = TimeScalesFactory.getUTC()

            # 创建AbsoluteDate
            date = AbsoluteDate(
                timestamp.year, timestamp.month, timestamp.day,
                timestamp.hour, timestamp.minute,
                timestamp.second + timestamp.microsecond / 1e6,
                utc
            )

            # 获取太阳
            sun = CelestialBodyFactory.getSun()

            # 获取ECEF坐标系(ITRF)
            itrf = FramesFactory.getITRF(None, None)  # 不使用EOP，简化处理

            # 获取太阳在ECEF中的位置
            sun_pv = sun.getPVCoordinates(date, itrf)
            sun_pos = sun_pv.getPosition()

            return (float(sun_pos.getX()), float(sun_pos.getY()), float(sun_pos.getZ()))

        except Exception as e:
            logger.error(f"Orekit sun position calculation failed: {e}")
            raise RuntimeError(f"Orekit calculation failed: {e}") from e

    def get_sun_position(self, timestamp: datetime) -> Tuple[float, float, float]:
        """
        获取太阳在ECEF坐标系中的位置

        Args:
            timestamp: UTC时间（必须带时区信息）

        Returns:
            Tuple[float, float, float]: 太阳ECEF位置（米）

        Raises:
            TypeError: 时间戳类型无效
            ValueError: 时间戳格式错误
        """
        # 验证时间戳
        utc_time = self._validate_timestamp(timestamp)

        # 如果启用Orekit且可用，使用Orekit计算
        if self.use_orekit and OREKIT_BRIDGE_AVAILABLE and self._orekit_bridge is not None:
            try:
                if self._orekit_bridge.is_jvm_running():
                    return self._calculate_sun_position_orekit(utc_time)
            except (RuntimeError, ConnectionError) as e:
                logger.warning(f"Orekit calculation failed, falling back to simplified model: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in Orekit calculation: {e}")
                raise

        # 使用简化模型
        return self._calculate_sun_position_simplified(utc_time)

    def get_sun_direction(
        self,
        satellite_position: Tuple[float, float, float],
        timestamp: datetime
    ) -> Tuple[float, float, float]:
        """
        获取从卫星位置指向太阳的单位方向向量

        Args:
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间

        Returns:
            Tuple[float, float, float]: 单位方向向量（指向太阳）

        Raises:
            TypeError: 输入类型无效
            ValueError: 输入格式错误
        """
        # 验证输入
        sat_pos = self._validate_satellite_position(satellite_position)
        utc_time = self._validate_timestamp(timestamp)

        # 获取太阳位置
        sun_pos = self.get_sun_position(utc_time)

        # 计算从卫星到太阳的向量
        dx = sun_pos[0] - sat_pos[0]
        dy = sun_pos[1] - sat_pos[1]
        dz = sun_pos[2] - sat_pos[2]

        # 归一化
        magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)

        if magnitude < 1e-10:
            # 如果卫星在太阳位置（理论上不可能），返回默认方向
            logger.warning("Satellite position too close to sun, returning default direction")
            return (1.0, 0.0, 0.0)

        return (dx/magnitude, dy/magnitude, dz/magnitude)
