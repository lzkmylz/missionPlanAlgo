"""
卫星姿态角计算器

基于卫星-目标几何关系，计算卫星成像时刻的姿态角（滚转、俯仰、偏航）。

坐标系：LVLH (Local Vertical Local Horizontal)
- X轴：沿飞行方向（速度方向）
- Y轴：垂直于轨道平面（与角动量方向相反）
- Z轴：指向地心（负位置方向）

姿态角定义：
- Roll (滚转角)：绕X轴旋转，控制侧摆（左右）
- Pitch (俯仰角)：绕Y轴旋转，控制前后斜视
- Yaw (偏航角)：绕Z轴旋转，本实现使用零偏航模式

传播器支持：
- SGP4：使用TLE，含J2摄动
- HPOP：高精度轨道传播（通过配置启用）

示例:
    calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)
    attitude = calculator.calculate_attitude(satellite, target, imaging_time)
    print(f"Roll: {attitude.roll:.2f}°, Pitch: {attitude.pitch:.2f}°, Yaw: {attitude.yaw:.2f}°")
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Tuple, Optional, Dict, Any
import math
import logging

from core.models.satellite import Satellite
from core.models.target import Target

logger = logging.getLogger(__name__)


class PropagatorType(Enum):
    """轨道传播器类型"""
    SGP4 = "sgp4"       # 使用TLE和SGP4模型（含J2摄动）
    HPOP = "hpop"       # 使用STK HPOP高精度传播


@dataclass
class AttitudeAngles:
    """卫星姿态角

    Attributes:
        roll: 滚转角（度），绕X轴，控制侧摆
        pitch: 俯仰角（度），绕Y轴，控制前后斜视
        yaw: 偏航角（度），绕Z轴，本实现固定为0
        coordinate_system: 坐标系名称，默认为"LVLH"
        timestamp: 计算时刻（UTC）
    """
    roll: float
    pitch: float
    yaw: float = 0.0
    coordinate_system: str = "LVLH"
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "coordinate_system": self.coordinate_system,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def __post_init__(self):
        """验证姿态角范围"""
        # 滚转角通常限制在 ±45° 或 ±30°，但特殊情况下可能更大
        # 只在角度超过170°时警告（可能指向错误方向）
        if abs(self.roll) > 170:
            logger.warning(f"Roll angle > 170° indicates possible pointing error: {self.roll}°")

        # 俯仰角通常较小，但在大斜视时可能较大
        if abs(self.pitch) > 170:
            logger.warning(f"Pitch angle > 170° indicates possible pointing error: {self.pitch}°")


class AttitudeCalculator:
    """卫星姿态角计算器

    计算卫星成像时刻在LVLH坐标系下的姿态角。

    Attributes:
        propagator_type: 使用的轨道传播器类型
        earth_radius: 地球半径（米）
    """

    EARTH_RADIUS = 6371000.0  # 地球平均半径（米）
    EARTH_GM = 3.986004418e14  # 地球引力常数 (m^3/s^2)

    def __init__(self, propagator_type: PropagatorType = PropagatorType.SGP4):
        """初始化姿态角计算器

        Args:
            propagator_type: 轨道传播器类型，默认为SGP4
        """
        self.propagator_type = propagator_type
        self._sgp4_available = self._check_sgp4_available()
        self._hpop_available = self._check_hpop_available()

    def _check_sgp4_available(self) -> bool:
        """检查SGP4库是否可用"""
        try:
            from sgp4.api import Satrec, jday
            return True
        except ImportError:
            logger.warning("SGP4 library not available")
            return False

    def _check_hpop_available(self) -> bool:
        """检查HPOP接口是否可用"""
        try:
            from core.orbit.hpop_interface import STKHPOPInterface
            return True
        except ImportError:
            return False

    def calculate_attitude(
        self,
        satellite: Satellite,
        target: Target,
        imaging_time: datetime
    ) -> AttitudeAngles:
        """计算卫星成像时刻的姿态角

        Args:
            satellite: 卫星对象，包含轨道信息
            target: 目标对象，包含地理坐标
            imaging_time: 成像时刻（UTC）

        Returns:
            AttitudeAngles: 姿态角对象

        Raises:
            ValueError: 如果输入参数无效
        """
        if satellite is None:
            raise ValueError("Satellite cannot be None")
        if target is None:
            raise ValueError("Target cannot be None")
        if imaging_time is None:
            raise ValueError("Imaging time cannot be None")

        # 获取卫星在成像时刻的位置和速度
        position, velocity = self._get_satellite_state(satellite, imaging_time)

        # 构建LVLH坐标系
        lvlh_frame = self._construct_lvlh_frame(position, velocity)

        # 计算目标视线向量
        los_vector = self._calculate_los_vector(position, target)

        # 将视线向量转换到LVLH坐标系
        los_in_lvlh = self._transform_to_lvlh(los_vector, lvlh_frame)

        # 计算滚转和俯仰角
        roll, pitch = self._calculate_roll_pitch(los_in_lvlh)

        # 零偏航模式
        yaw = 0.0

        return AttitudeAngles(
            roll=math.degrees(roll),
            pitch=math.degrees(pitch),
            yaw=yaw,
            coordinate_system="LVLH",
            timestamp=imaging_time
        )

    def _get_satellite_state(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """获取卫星在给定时刻的位置和速度

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻（UTC）

        Returns:
            (position, velocity) in ECEF coordinates (meters, m/s)
        """
        if self.propagator_type == PropagatorType.HPOP and self._hpop_available:
            return self._propagate_with_hpop(satellite, timestamp)
        else:
            return self._propagate_with_sgp4(satellite, timestamp)

    def _propagate_with_sgp4(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用SGP4传播卫星轨道

        SGP4模型包含J2摄动项，精度适合大多数对地成像任务。

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻

        Returns:
            (position, velocity) in ECEF (meters, m/s)
        """
        if not self._sgp4_available:
            raise RuntimeError("SGP4 library not available")

        from sgp4.api import Satrec, jday

        # 创建SGP4卫星记录
        if satellite.tle_line1 and satellite.tle_line2:
            satrec = Satrec.twoline2rv(satellite.tle_line1, satellite.tle_line2)
        else:
            # 没有TLE时使用轨道参数生成简化的TLE
            satrec = self._create_satrec_from_elements(satellite, timestamp)

        # 计算儒略日
        jd, fr = jday(
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute,
            timestamp.second + timestamp.microsecond / 1e6
        )

        # SGP4传播
        error, position_eci, velocity_eci = satrec.sgp4(jd, fr)

        if error != 0:
            raise RuntimeError(f"SGP4 propagation error code: {error}")

        # ECI到ECEF转换（简化，不考虑岁差章动）
        position_ecef = self._eci_to_ecef(position_eci, timestamp)
        velocity_ecef = self._eci_to_ecef_velocity(velocity_eci, position_eci, timestamp)

        # 转换为米
        position_ecef = tuple(p * 1000 for p in position_ecef)
        velocity_ecef = tuple(v * 1000 for v in velocity_ecef)

        return position_ecef, velocity_ecef

    def _create_satrec_from_elements(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Any:
        """从轨道参数创建SGP4卫星记录（简化实现）

        当没有TLE时使用轨道六根数创建简化的卫星记录。
        """
        from sgp4.api import Satrec

        orbit = satellite.orbit

        # 使用简化参数
        a = orbit.get_semi_major_axis() / 1000.0  # 转换为km
        ecc = orbit.eccentricity
        inc = math.radians(orbit.inclination)
        raan = math.radians(orbit.raan)
        argp = math.radians(orbit.arg_of_perigee)

        # 计算平近点角（从历元时间推算）
        if orbit.epoch:
            delta_t = (timestamp - orbit.epoch).total_seconds()
            period = orbit.get_period()
            mean_motion = 2 * math.pi / period
            ma = orbit.mean_anomaly + math.degrees(mean_motion * delta_t)
        else:
            ma = orbit.mean_anomaly

        ma_rad = math.radians(ma)

        # 创建Satrec对象
        satrec = Satrec()

        # sgp4init 使用位置参数
        # whichconst, opsmode, satnum, epochdays, ndot, nddot, bstar,
        # inclo, nodeo, ecco, argpo, mo, no_kozai
        no_kozai = math.sqrt(self.EARTH_GM / (a * 1000) ** 3) * 60  # 平均运动 (rad/min)

        satrec.sgp4init(
            1,  # whichconst: 1 = WGS84
            'i',  # opsmode: 'i' = 改进模式 (IMPROVED)
            99999,  # satnum: 卫星编号
            self._datetime_to_epoch_days(timestamp),  # epochdays
            0.0,  # ndot: 平均运动的一阶时间导数
            0.0,  # nddot: 平均运动的二阶时间导数
            0.0,  # bstar: B*拖曳系数
            inc,  # inclo: 轨道倾角 (rad)
            raan,  # nodeo: 升交点赤经 (rad)
            ecc,  # ecco: 偏心率
            argp,  # argpo: 近地点幅角 (rad)
            ma_rad,  # mo: 平近点角 (rad)
            no_kozai,  # no_kozai: 平均运动 (rad/min)
        )

        return satrec

    def _datetime_to_epoch_days(self, dt: datetime) -> float:
        """将datetime转换为SGP4历元天数"""
        # 参考历元：1950年1月1日
        ref_date = datetime(1950, 1, 1, tzinfo=dt.tzinfo)
        delta = dt - ref_date
        return delta.total_seconds() / 86400.0

    def _eci_to_ecef(
        self,
        position_eci: Tuple[float, float, float],
        timestamp: datetime
    ) -> Tuple[float, float, float]:
        """ECI坐标转换到ECEF坐标（简化计算）

        使用格林尼治恒星时进行坐标旋转，忽略岁差章动。
        """
        from sgp4.api import jday

        x, y, z = position_eci

        # 计算格林尼治恒星时（GMST）
        jd, fr = jday(
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute,
            timestamp.second + timestamp.microsecond / 1e6
        )

        # 简化GMST计算（度）
        d = jd - 2451545.0 + fr
        gmst = (18.697374558 + 24.06570982441908 * d) % 24
        gmst_deg = gmst * 15.0
        gmst_rad = math.radians(gmst_deg)

        # 绕Z轴旋转
        x_ecef = x * math.cos(gmst_rad) + y * math.sin(gmst_rad)
        y_ecef = -x * math.sin(gmst_rad) + y * math.cos(gmst_rad)
        z_ecef = z

        return (x_ecef, y_ecef, z_ecef)

    def _eci_to_ecef_velocity(
        self,
        velocity_eci: Tuple[float, float, float],
        position_eci: Tuple[float, float, float],
        timestamp: datetime
    ) -> Tuple[float, float, float]:
        """ECI速度转换到ECEF速度（包含地球自转效应）"""
        from sgp4.api import jday

        vx, vy, vz = velocity_eci
        x, y, z = position_eci

        # 计算GMST变化率（地球自转角速度）
        omega_earth = 7.2921158553e-5  # rad/s

        # 计算GMST
        jd, fr = jday(
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute,
            timestamp.second + timestamp.microsecond / 1e6
        )
        d = jd - 2451545.0 + fr
        gmst = (18.697374558 + 24.06570982441908 * d) % 24
        gmst_deg = gmst * 15.0
        gmst_rad = math.radians(gmst_deg)

        # ECEF速度 = 旋转后的ECI速度 - 地球自转效应
        # 公式来源: ECEF = R * ECI - omega × r
        # 其中 R 是旋转矩阵，omega 是地球自转角速度向量，r 是位置向量
        # 注意：omega × r 在X方向是 -omega*y，在Y方向是 +omega*x
        vx_ecef = vx * math.cos(gmst_rad) + vy * math.sin(gmst_rad) - omega_earth * y
        vy_ecef = -vx * math.sin(gmst_rad) + vy * math.cos(gmst_rad) + omega_earth * x
        vz_ecef = vz

        return (vx_ecef, vy_ecef, vz_ecef)

    def _propagate_with_hpop(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用HPOP传播卫星轨道（高精度）

        需要STK HPOP接口可用。

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻

        Returns:
            (position, velocity) in ECEF (meters, m/s)

        Raises:
            RuntimeError: 如果HPOP不可用或传播失败
        """
        from core.orbit.hpop_interface import STKHPOPInterface, HPOPConfig

        if not self._hpop_available:
            logger.warning("HPOP not available, falling back to SGP4")
            return self._propagate_with_sgp4(satellite, timestamp)

        try:
            with STKHPOPInterface() as hpop:
                # 简化的HPOP传播实现
                # 实际实现需要缓存传播结果以提高性能
                result = hpop.propagate_orbit(
                    satellite_id=satellite.id,
                    orbit=satellite.orbit,
                    start_time=timestamp,
                    end_time=timestamp,
                    config=HPOPConfig()
                )

                position, velocity = result.get_position_at_time(timestamp)

                if position is None:
                    raise RuntimeError("HPOP propagation returned None")

                return position, velocity

        except Exception as e:
            logger.warning(f"HPOP propagation failed: {e}, falling back to SGP4")
            return self._propagate_with_sgp4(satellite, timestamp)

    def _construct_lvlh_frame(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float]
    ) -> Dict[str, Tuple[float, float, float]]:
        """构建LVLH坐标系

        LVLH (Local Vertical Local Horizontal):
        - X: 沿飞行方向（速度单位向量）
        - Z: 指向地心（负位置单位向量）
        - Y: 完成右手系 (Z × X)

        Args:
            position: 卫星位置 (ECEF, meters)
            velocity: 卫星速度 (ECEF, m/s)

        Returns:
            LVLH坐标系的基向量字典 {'X': (x,y,z), 'Y': (x,y,z), 'Z': (x,y,z)}
        """
        px, py, pz = position
        vx, vy, vz = velocity

        # Z轴：指向地心（负位置方向）
        r_norm = math.sqrt(px**2 + py**2 + pz**2)
        if r_norm < 1e-10:
            raise ValueError("Invalid satellite position (near origin)")
        Z = (-px / r_norm, -py / r_norm, -pz / r_norm)

        # X轴：沿飞行方向（速度在垂直于Z方向的分量）
        # 先计算速度在Z方向的分量
        v_dot_z = vx * Z[0] + vy * Z[1] + vz * Z[2]

        # 速度垂直于Z的分量
        vx_perp = vx - v_dot_z * Z[0]
        vy_perp = vy - v_dot_z * Z[1]
        vz_perp = vz - v_dot_z * Z[2]

        v_perp_norm = math.sqrt(vx_perp**2 + vy_perp**2 + vz_perp**2)
        if v_perp_norm < 1e-10:
            # 速度几乎平行于Z轴，使用位置与极轴的叉积作为X
            # 这种情况在极轨道经过极点时可能发生
            pole = (0, 0, 1)
            x_temp = (
                Z[1] * pole[2] - Z[2] * pole[1],
                Z[2] * pole[0] - Z[0] * pole[2],
                Z[0] * pole[1] - Z[1] * pole[0]
            )
            x_norm = math.sqrt(sum(c**2 for c in x_temp))
            X = tuple(c / x_norm for c in x_temp)
        else:
            X = (vx_perp / v_perp_norm, vy_perp / v_perp_norm, vz_perp / v_perp_norm)

        # Y轴：Z × X（完成右手系）
        Y = (
            Z[1] * X[2] - Z[2] * X[1],
            Z[2] * X[0] - Z[0] * X[2],
            Z[0] * X[1] - Z[1] * X[0]
        )

        # 归一化Y（理论上已经归一化，但确保数值稳定性）
        y_norm = math.sqrt(sum(c**2 for c in Y))
        Y = tuple(c / y_norm for c in Y)

        return {'X': X, 'Y': Y, 'Z': Z}

    def _calculate_los_vector(
        self,
        satellite_position: Tuple[float, float, float],
        target: Target
    ) -> Tuple[float, float, float]:
        """计算卫星到目标的视线向量

        Args:
            satellite_position: 卫星位置 (ECEF, meters)
            target: 目标对象

        Returns:
            归一化的视线向量 (ECEF)

        Raises:
            ValueError: 如果目标缺少必要的坐标属性
        """
        # 验证目标具有必要的坐标属性
        if not hasattr(target, 'latitude') or not hasattr(target, 'longitude'):
            raise ValueError(f"Target {getattr(target, 'id', 'unknown')} missing latitude/longitude attributes")

        # 验证坐标值有效
        if target.latitude is None or target.longitude is None:
            raise ValueError(f"Target {getattr(target, 'id', 'unknown')} has None coordinates")

        # 目标ECEF位置
        target_ecef = self._geodetic_to_ecef(target.latitude, target.longitude, 0.0)

        # 视线向量
        los = (
            target_ecef[0] - satellite_position[0],
            target_ecef[1] - satellite_position[1],
            target_ecef[2] - satellite_position[2]
        )

        # 归一化
        los_norm = math.sqrt(sum(c**2 for c in los))
        if los_norm < 1e-10:
            raise ValueError("Target is too close to satellite")

        return tuple(c / los_norm for c in los)

    def _geodetic_to_ecef(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> Tuple[float, float, float]:
        """地理坐标转换为ECEF坐标

        使用WGS84椭球模型简化计算（假设地球为球体）。

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

    def _transform_to_lvlh(
        self,
        vector: Tuple[float, float, float],
        lvlh_frame: Dict[str, Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """将向量从ECEF转换到LVLH坐标系

        Args:
            vector: ECEF坐标系中的向量
            lvlh_frame: LVLH坐标系的基向量

        Returns:
            LVLH坐标系中的向量
        """
        X, Y, Z = lvlh_frame['X'], lvlh_frame['Y'], lvlh_frame['Z']

        # 向量在各轴上的投影
        vx = vector[0] * X[0] + vector[1] * X[1] + vector[2] * X[2]
        vy = vector[0] * Y[0] + vector[1] * Y[1] + vector[2] * Y[2]
        vz = vector[0] * Z[0] + vector[1] * Z[1] + vector[2] * Z[2]

        return (vx, vy, vz)

    def _calculate_roll_pitch(
        self,
        los_in_lvlh: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """计算滚转和俯仰角

        在LVLH坐标系中：
        - Roll (绕X轴): atan2(Y, -Z)
        - Pitch (绕Y轴): atan2(X, -Z)

        注意：-Z是因为Z轴指向地心，而我们需要指向目标。

        Args:
            los_in_lvlh: 视线向量在LVLH坐标系中的表示

        Returns:
            (roll, pitch) in radians
        """
        vx, vy, vz = los_in_lvlh

        # 计算滚转角（绕X轴）
        # roll = atan2(vy, -vz)
        roll = math.atan2(vy, -vz)

        # 计算俯仰角（绕Y轴）
        # pitch = atan2(vx, -vz)
        pitch = math.atan2(vx, -vz)

        return roll, pitch
