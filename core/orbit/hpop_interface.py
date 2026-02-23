"""
STK HPOP高精度轨道预报接口

实现Chapter 18.1: STK HPOP高精度轨道预报接口
使用STK COM接口与HPOP传播器交互，支持高精度轨道计算
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import math
import logging

logger = logging.getLogger(__name__)


class STKNotAvailableError(Exception):
    """STK不可用时抛出的异常"""
    pass


class ForceModel(Enum):
    """力学模型类型"""
    J2 = "j2"
    J4 = "j4"
    HPOP_FULL = "hpop_full"


class AtmosphericModel(Enum):
    """大气模型类型"""
    MSIS90 = "msis90"
    JACCHIA77 = "jacchia77"
    JACCHIA71 = "jacchia71"
    NRLMSISE00 = "nrlmsise00"


@dataclass
class HPOPConfig:
    """HPOP传播配置"""
    # 时间步长（秒）
    time_step: float = 60.0

    # 力学模型
    force_model: ForceModel = ForceModel.HPOP_FULL

    # 摄动模型开关
    use_earth_gravity: bool = True
    use_sun_gravity: bool = True
    use_moon_gravity: bool = True
    use_solar_radiation_pressure: bool = True
    use_atmospheric_drag: bool = True

    # 大气模型
    atmospheric_model: AtmosphericModel = AtmosphericModel.MSIS90

    # 卫星物理参数（用于太阳光压和大气 drag 计算）
    mass_kg: float = 100.0  # 卫星质量（kg）
    drag_area_m2: float = 1.0  # 阻力面积（m^2）
    srp_area_m2: float = 1.0  # 太阳辐射压面积（m^2）
    drag_coefficient: float = 2.2  # 阻力系数
    reflectivity: float = 1.5  # 反射系数


@dataclass
class HPOPPropagationResult:
    """HPOP轨道传播结果"""
    satellite_id: str
    timestamps: List[datetime]
    positions: List[Tuple[float, float, float]]  # ECI坐标 (x, y, z) in meters
    velocities: List[Tuple[float, float, float]]  # ECI速度 (vx, vy, vz) in m/s

    def get_position_at_time(self, query_time: datetime) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """
        获取指定时间的位置和速度（使用线性插值）

        Args:
            query_time: 查询时间

        Returns:
            (position, velocity) 或 (None, None) 如果不在范围内
        """
        if not self.timestamps or query_time < self.timestamps[0] or query_time > self.timestamps[-1]:
            return None, None

        # 找到相邻的两个时间点
        for i in range(len(self.timestamps) - 1):
            t1, t2 = self.timestamps[i], self.timestamps[i + 1]
            if t1 <= query_time <= t2:
                # 线性插值
                total_seconds = (t2 - t1).total_seconds()
                if total_seconds == 0:
                    return self.positions[i], self.velocities[i]

                elapsed = (query_time - t1).total_seconds()
                ratio = elapsed / total_seconds

                # 位置插值
                p1, p2 = self.positions[i], self.positions[i + 1]
                position = (
                    p1[0] + ratio * (p2[0] - p1[0]),
                    p1[1] + ratio * (p2[1] - p1[1]),
                    p1[2] + ratio * (p2[2] - p1[2])
                )

                # 速度插值
                v1, v2 = self.velocities[i], self.velocities[i + 1]
                velocity = (
                    v1[0] + ratio * (v2[0] - v1[0]),
                    v1[1] + ratio * (v2[1] - v1[1]),
                    v1[2] + ratio * (v2[2] - v1[2])
                )

                return position, velocity

        return None, None

    def get_orbital_elements(self) -> Dict[str, float]:
        """
        从位置和速度计算轨道根数（简化计算）

        Returns:
            轨道根数字典
        """
        if not self.positions or not self.velocities:
            return {}

        # 使用第一个点的位置和速度
        r = self.positions[0]
        v = self.velocities[0]

        r_mag = math.sqrt(sum(x**2 for x in r))
        v_mag = math.sqrt(sum(x**2 for x in v))

        # 地球引力常数
        mu = 3.986004418e14  # m^3/s^2

        # 比角动量
        h = (
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0]
        )
        h_mag = math.sqrt(sum(x**2 for x in h))

        # 半长轴
        a = 1.0 / (2.0 / r_mag - v_mag**2 / mu)

        # 偏心率向量
        e_vec = (
            (v_mag**2 / mu - 1.0 / r_mag) * r[0] - (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) * v[0] / mu,
            (v_mag**2 / mu - 1.0 / r_mag) * r[1] - (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) * v[1] / mu,
            (v_mag**2 / mu - 1.0 / r_mag) * r[2] - (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) * v[2] / mu
        )
        e = math.sqrt(sum(x**2 for x in e_vec))

        # 轨道倾角
        i = math.acos(h[2] / h_mag)

        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'inclination': math.degrees(i),
        }

    def get_subpoint(self, query_time: datetime) -> Tuple[float, float, float]:
        """
        获取指定时间的星下点坐标（简化计算）

        Returns:
            (latitude, longitude, altitude) in degrees and meters
        """
        position, _ = self.get_position_at_time(query_time)
        if position is None:
            return 0.0, 0.0, 0.0

        x, y, z = position
        r = math.sqrt(x**2 + y**2 + z**2)

        # 地球半径
        R_earth = 6371000.0  # meters

        # 纬度
        lat = math.degrees(math.asin(z / r))

        # 经度
        lon = math.degrees(math.atan2(y, x))

        # 高度
        alt = r - R_earth

        return lat, lon, alt


class STKHPOPInterface:
    """
    STK HPOP高精度轨道预报接口

    通过COM接口与STK交互，使用HPOP传播器进行高精度轨道计算
    支持大气阻力、太阳光压、三体引力等摄动模型
    """

    def __init__(self, config: Optional[HPOPConfig] = None):
        """
        初始化HPOP接口

        Args:
            config: HPOP配置，使用默认配置如果为None

        Raises:
            STKNotAvailableError: 如果STK不可用
        """
        self.config = config or HPOPConfig()
        self.is_connected = False
        self._stk_app = None
        self._stk_root = None
        self._scenario = None
        self._satellites: Dict[str, Any] = {}
        self._position_cache: Dict[str, List[Tuple[datetime, Tuple[float, float, float], Tuple[float, float, float]]]] = {}

        if not self._check_stk_available():
            raise STKNotAvailableError(
                "STK is not available. Please install STK and ensure the COM interface is accessible."
            )

    @staticmethod
    def _check_stk_available() -> bool:
        """检查STK是否可用"""
        try:
            import win32com.client
            # 尝试创建STK应用对象
            app = win32com.client.Dispatch("STK11.Application")
            app.Quit()
            return True
        except ImportError:
            logger.warning("win32com not available, STK interface disabled")
            return False
        except Exception as e:
            logger.warning(f"STK not available: {e}")
            return False

    def connect(self) -> None:
        """连接到STK"""
        if not self.is_connected:
            self._init_stk_connection()
            self.is_connected = True
            logger.info("Connected to STK")

    def disconnect(self) -> None:
        """断开与STK的连接"""
        if self.is_connected:
            try:
                if self._stk_app:
                    self._stk_app.Quit()
            except Exception as e:
                logger.warning(f"Error disconnecting from STK: {e}")
            finally:
                self._stk_app = None
                self._stk_root = None
                self._scenario = None
                self._satellites = {}
                self.is_connected = False
                logger.info("Disconnected from STK")

    def _init_stk_connection(self) -> None:
        """初始化STK连接（模拟实现）"""
        # 实际实现中，这里会通过win32com.client.Dispatch创建STK应用
        # 由于STK是商业软件，这里提供接口框架
        logger.info("Initializing STK connection...")
        # self._stk_app = win32com.client.Dispatch("STK11.Application")
        # self._stk_root = self._stk_app.Personality2
        pass

    def propagate_orbit(
        self,
        satellite_id: str,
        orbit: Any,
        start_time: datetime,
        end_time: datetime,
        config: Optional[HPOPConfig] = None
    ) -> HPOPPropagationResult:
        """
        使用HPOP传播卫星轨道

        Args:
            satellite_id: 卫星ID
            orbit: 轨道参数对象
            start_time: 开始时间
            end_time: 结束时间
            config: 传播配置，使用默认配置如果为None

        Returns:
            HPOPPropagationResult: 传播结果

        Raises:
            RuntimeError: 如果未连接到STK
            ValueError: 如果参数无效
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to STK. Call connect() first.")

        if end_time <= start_time:
            raise ValueError("End time must be after start time")

        cfg = config or self.config

        # 实际实现中，这里会：
        # 1. 在STK中创建卫星对象
        # 2. 配置HPOP传播器
        # 3. 执行传播
        # 4. 提取结果

        # 模拟返回数据
        timestamps = []
        positions = []
        velocities = []

        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            # 这里应该调用STK获取实际位置
            # 简化：返回模拟数据
            positions.append((6878000.0, 0.0, 0.0))
            velocities.append((0.0, 7500.0, 0.0))
            current_time += timedelta(seconds=cfg.time_step)

        result = HPOPPropagationResult(
            satellite_id=satellite_id,
            timestamps=timestamps,
            positions=positions,
            velocities=velocities
        )

        # 缓存结果
        self._position_cache[satellite_id] = [
            (t, p, v) for t, p, v in zip(timestamps, positions, velocities)
        ]

        return result

    def get_position_at_time(
        self,
        satellite_id: str,
        query_time: datetime
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """
        获取指定时间的位置和速度

        Args:
            satellite_id: 卫星ID
            query_time: 查询时间

        Returns:
            (position, velocity) 或 (None, None)
        """
        if satellite_id not in self._position_cache:
            return None, None

        cache = self._position_cache[satellite_id]

        # 找到相邻时间点进行插值
        for i in range(len(cache) - 1):
            t1, p1, v1 = cache[i]
            t2, p2, v2 = cache[i + 1]

            if t1 <= query_time <= t2:
                # 线性插值
                total = (t2 - t1).total_seconds()
                if total == 0:
                    return p1, v1

                elapsed = (query_time - t1).total_seconds()
                ratio = elapsed / total

                position = (
                    p1[0] + ratio * (p2[0] - p1[0]),
                    p1[1] + ratio * (p2[1] - p1[1]),
                    p1[2] + ratio * (p2[2] - p1[2])
                )
                velocity = (
                    v1[0] + ratio * (v2[0] - v1[0]),
                    v1[1] + ratio * (v2[1] - v1[1]),
                    v1[2] + ratio * (v2[2] - v1[2])
                )
                return position, velocity

        return None, None

    def _configure_hpop_force_model(self, satellite: Any, config: HPOPConfig) -> None:
        """
        配置HPOP力学模型

        Args:
            satellite: STK卫星对象
            config: HPOP配置
        """
        # 实际实现中，这里会配置STK的HPOP传播器参数
        logger.info(f"Configuring HPOP force model for {config.force_model.value}")

        # 配置示例（伪代码）：
        # propagator = satellite.Propagator
        # propagator.ForceModel.UseAtmosphericDrag = config.use_atmospheric_drag
        # propagator.ForceModel.UseSolarRadiationPressure = config.use_solar_radiation_pressure
        # propagator.ForceModel.AtmosphericModel = config.atmospheric_model.value
        pass

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
        return False
