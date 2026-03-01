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
from datetime import datetime, timedelta
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

    传播策略（按精度从高到低）:
    1. Orekit: Java高精度轨道传播（如果启用且可用）
    2. SGP4: 使用TLE（含J2摄动）
    3. HPOP: STK高精度轨道传播（当无TLE但HPOP可用时）
    4. J2 Model: 使用J2摄动模型（当无TLE且无HPOP时）
    5. Two-Body: 简化二体模型（降级方案）

    Attributes:
        propagator_type: 使用的轨道传播器类型
        earth_radius: 地球半径（米）
    """

    EARTH_RADIUS = 6371000.0  # 地球平均半径（米）
    EARTH_GM = 3.986004418e14  # 地球引力常数 (m^3/s^2)
    EARTH_OMEGA = 7.2921158553e-5  # 地球自转角速度 (rad/s)

    # J2摄动系数 (WGS84)
    J2 = 1.08263e-3
    EARTH_RADIUS_KM = 6378.137  # km

    def __init__(self, propagator_type: PropagatorType = PropagatorType.SGP4, use_orekit: bool = True):
        """初始化姿态角计算器

        Args:
            propagator_type: 轨道传播器类型，默认为SGP4
            use_orekit: 是否优先使用Orekit（如果可用），默认True
        """
        self.propagator_type = propagator_type
        self._use_orekit = use_orekit
        self._sgp4_available = self._check_sgp4_available()
        self._hpop_available = self._check_hpop_available()
        self._orekit_available = self._check_orekit_available()

    def _check_orekit_available(self) -> bool:
        """检查Orekit Java桥接是否可用"""
        try:
            from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge, JPYPE_AVAILABLE
            return JPYPE_AVAILABLE
        except ImportError:
            return False

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
        # 1. 如果启用Orekit批量传播，优先使用（最高精度+高性能）
        if self._use_orekit and self._orekit_available:
            try:
                return self._get_state_from_batch_propagator(satellite, timestamp)
            except Exception as e:
                logger.warning(f"Orekit batch propagation failed: {e}, falling back to other methods")

        # 2. 优先使用TLE（如果可用）
        if satellite.tle_line1 and satellite.tle_line2:
            if self.propagator_type == PropagatorType.HPOP and self._hpop_available:
                return self._propagate_with_hpop(satellite, timestamp)
            else:
                return self._propagate_with_sgp4(satellite, timestamp)

        # 3. 没有TLE时，尝试其他方法
        # 优先使用HPOP，其次使用J2模型，最后使用二体模型
        if self._hpop_available:
            try:
                return self._propagate_with_hpop(satellite, timestamp)
            except Exception as e:
                logger.warning(f"HPOP propagation failed: {e}, falling back to J2 model")

        # 使用J2摄动模型（比二体更精确）
        return self._propagate_with_j2_model(satellite, timestamp)

    def _get_state_from_batch_propagator(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """从批量传播器获取卫星状态

        使用 OrekitBatchPropagator 进行高性能批量传播，
        通过缓存和插值快速获取任意时刻状态。

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻

        Returns:
            (position, velocity) in ECEF (meters, m/s)

        Raises:
            RuntimeError: 如果批量传播器不可用或获取失败
        """
        from .orbit_batch_propagator import get_batch_propagator

        propagator = get_batch_propagator()
        if propagator is None:
            raise RuntimeError("Orekit batch propagator not available")

        # 尝试从缓存获取
        state = propagator.get_state_at_time(satellite.id, timestamp)

        if state is None:
            # 缓存未命中，需要预计算
            # 计算时间窗口（前后各扩展30分钟）
            start_time = timestamp - timedelta(minutes=30)
            end_time = timestamp + timedelta(minutes=30)

            # 预计算轨道（使用1秒步长保证精度）
            success = propagator.precompute_satellite_orbit(
                satellite=satellite,
                start_time=start_time,
                end_time=end_time,
                time_step=timedelta(seconds=1)  # 1秒步长
            )

            if not success:
                raise RuntimeError(f"Failed to precompute orbit for {satellite.id}")

            # 再次尝试获取
            state = propagator.get_state_at_time(satellite.id, timestamp)

            if state is None:
                raise RuntimeError(f"Still cannot get state for {satellite.id} at {timestamp}")

        return state

    def _propagate_with_j2_model(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用J2摄动模型传播卫星轨道

        J2摄动是地球扁率引起的主要摄动，影响：
        1. 升交点赤经的长期漂移 (RAAN regression)
        2. 近地点幅角的长期漂移 (Argument of perigee precession)

        公式来源: Vallado, Fundamentals of Astrodynamics and Applications

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻

        Returns:
            (position, velocity) in ECEF (meters, m/s)
        """
        orbit = satellite.orbit
        if orbit is None:
            raise ValueError(f"Satellite {satellite.id} has no orbit data")

        # 获取轨道参数
        a = orbit.get_semi_major_axis() / 1000.0  # 转换为 km
        ecc = orbit.eccentricity
        inc = math.radians(orbit.inclination)
        raan = math.radians(orbit.raan)
        argp = math.radians(orbit.arg_of_perigee)

        # 历元时间和当前时间
        epoch = orbit.epoch if orbit.epoch else timestamp
        delta_t = (timestamp - epoch).total_seconds()

        # 平均运动 (rad/s)
        n = math.sqrt(self.EARTH_GM / ((a * 1000) ** 3))

        # 计算J2摄动引起的轨道参数变化
        # J2对升交点赤经的影响 (RAAN regression)
        cos_inc = math.cos(inc)
        j2_factor = 1.5 * self.J2 * ((self.EARTH_RADIUS_KM / (a * (1 - ecc**2))) ** 2) * n

        # RAAN变化率 (rad/s)
        raan_dot = -j2_factor * cos_inc

        # 近地点幅角变化率 (rad/s)
        argp_dot = j2_factor * (2.5 * cos_inc**2 - 0.5)

        # 更新轨道参数
        updated_raan = raan + raan_dot * delta_t
        updated_argp = argp + argp_dot * delta_t

        # 计算平近点角
        initial_mean_anomaly = math.radians(orbit.mean_anomaly)
        mean_anomaly = (initial_mean_anomaly + n * delta_t) % (2 * math.pi)

        # 从平近点角解算真近点角 (使用牛顿迭代法)
        # 对于小偏心率，可以使用级数展开
        if ecc < 0.1:
            # 小偏心率近似
            true_anomaly = mean_anomaly + 2 * ecc * math.sin(mean_anomaly)
        else:
            # 牛顿迭代法解开普勒方程
            E = mean_anomaly  # 初始猜测
            for _ in range(10):
                delta = (E - ecc * math.sin(E) - mean_anomaly) / (1 - ecc * math.cos(E))
                E -= delta
                if abs(delta) < 1e-10:
                    break
            # 从偏近点角计算真近点角
            true_anomaly = 2 * math.atan2(
                math.sqrt(1 + ecc) * math.sin(E / 2),
                math.sqrt(1 - ecc) * math.cos(E / 2)
            )

        # 考虑J2对平近点角的影响（长期项）
        mean_anomaly_dot = n + j2_factor * (1 - 1.5 * math.sin(inc)**2)
        mean_anomaly = (initial_mean_anomaly + mean_anomaly_dot * delta_t) % (2 * math.pi)

        # 重新计算真近点角（使用更新后的平近点角）
        if ecc < 0.1:
            true_anomaly = mean_anomaly + 2 * ecc * math.sin(mean_anomaly)
        else:
            E = mean_anomaly
            for _ in range(10):
                delta = (E - ecc * math.sin(E) - mean_anomaly) / (1 - ecc * math.cos(E))
                E -= delta
                if abs(delta) < 1e-10:
                    break
            true_anomaly = 2 * math.atan2(
                math.sqrt(1 + ecc) * math.sin(E / 2),
                math.sqrt(1 - ecc) * math.cos(E / 2)
            )

        # 轨道半径
        r = a * (1 - ecc**2) / (1 + ecc * math.cos(true_anomaly))

        # 在轨道平面中的位置和速度
        x_orb = r * math.cos(true_anomaly)
        y_orb = r * math.sin(true_anomaly)
        z_orb = 0.0

        # 轨道速度
        h = math.sqrt(self.EARTH_GM * a * 1000 * (1 - ecc**2))  # 比角动量
        v_orb = math.sqrt(self.EARTH_GM / (r * 1000))  # 速度大小

        vx_orb = -v_orb * math.sin(true_anomaly) / math.sqrt(1 + ecc**2 + 2*ecc*math.cos(true_anomaly))
        vy_orb = v_orb * (ecc + math.cos(true_anomaly)) / math.sqrt(1 + ecc**2 + 2*ecc*math.cos(true_anomaly))
        vz_orb = 0.0

        # 归一化
        v_mag = math.sqrt(vx_orb**2 + vy_orb**2)
        if v_mag > 0:
            vx_orb = vx_orb / v_mag * v_orb
            vy_orb = vy_orb / v_mag * v_orb

        # 从轨道平面转换到惯性系 (3-1-3旋转)
        cos_raan, sin_raan = math.cos(updated_raan), math.sin(updated_raan)
        cos_inc, sin_inc = math.cos(inc), math.sin(inc)
        cos_argp, sin_argp = math.cos(updated_argp), math.sin(updated_argp)

        # 位置转换
        x_eci = (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * x_orb + \
                (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * y_orb
        y_eci = (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * x_orb + \
                (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * y_orb
        z_eci = (sin_argp * sin_inc) * x_orb + \
                (cos_argp * sin_inc) * y_orb

        # 速度转换
        vx_eci = (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * vx_orb + \
                 (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * vy_orb
        vy_eci = (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * vx_orb + \
                 (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * vy_orb
        vz_eci = (sin_argp * sin_inc) * vx_orb + \
                 (cos_argp * sin_inc) * vy_orb

        # ECI到ECEF转换
        from sgp4.api import jday
        jd, fr = jday(
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute,
            timestamp.second + timestamp.microsecond / 1e6
        )
        d = jd - 2451545.0 + fr
        gmst = (18.697374558 + 24.06570982441908 * d) % 24
        gmst_deg = gmst * 15.0
        gmst_rad = math.radians(gmst_deg)

        cos_gmst, sin_gmst = math.cos(gmst_rad), math.sin(gmst_rad)

        x_ecef = x_eci * cos_gmst + y_eci * sin_gmst
        y_ecef = -x_eci * sin_gmst + y_eci * cos_gmst
        z_ecef = z_eci

        vx_ecef = vx_eci * cos_gmst + vy_eci * sin_gmst - self.EARTH_OMEGA * y_eci
        vy_ecef = -vx_eci * sin_gmst + vy_eci * cos_gmst + self.EARTH_OMEGA * x_eci
        vz_ecef = vz_eci

        # 转换为米
        return (
            (x_ecef * 1000, y_ecef * 1000, z_ecef * 1000),
            (vx_ecef * 1000, vy_ecef * 1000, vz_ecef * 1000)
        )

    def _propagate_with_simple_twobody(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用简化二体模型传播卫星轨道

        当没有TLE时使用轨道六根数进行简化计算。
        假设圆形轨道，使用简单的开普勒定律。

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻

        Returns:
            (position, velocity) in ECEF (meters, m/s)
        """
        orbit = satellite.orbit
        if orbit is None:
            raise ValueError(f"Satellite {satellite.id} has no orbit data")

        # 获取轨道参数
        a = orbit.get_semi_major_axis()  # 半长轴 (m)
        ecc = orbit.eccentricity
        inc = math.radians(orbit.inclination)
        raan = math.radians(orbit.raan)
        argp = math.radians(orbit.arg_of_perigee)

        # 历元时间和当前时间
        epoch = orbit.epoch if orbit.epoch else timestamp
        delta_t = (timestamp - epoch).total_seconds()

        # 计算平均运动 (rad/s)
        n = math.sqrt(self.EARTH_GM / (a ** 3))

        # 计算平近点角随时间变化
        initial_mean_anomaly = math.radians(orbit.mean_anomaly)
        mean_anomaly = (initial_mean_anomaly + n * delta_t) % (2 * math.pi)

        # 简化：假设圆形轨道 (ecc = 0)，则真近点角 = 平近点角
        true_anomaly = mean_anomaly

        # 在轨道平面中的位置 (km)
        r = a / 1000.0  # 转换为 km
        x_orb = r * math.cos(true_anomaly)
        y_orb = r * math.sin(true_anomaly)
        z_orb = 0.0

        # 轨道平面速度 (km/s) - 圆形轨道
        v = math.sqrt(self.EARTH_GM / (r * 1000)) / 1000  # km/s
        vx_orb = -v * math.sin(true_anomaly)
        vy_orb = v * math.cos(true_anomaly)
        vz_orb = 0.0

        # 从轨道平面转换到惯性系 (3-1-3旋转)
        # 1. 绕Z轴旋转 -argp (近地点幅角)
        # 2. 绕X轴旋转 -inc (倾角)
        # 3. 绕Z轴旋转 -raan (升交点赤经)

        # 组合旋转矩阵
        cos_raan, sin_raan = math.cos(raan), math.sin(raan)
        cos_inc, sin_inc = math.cos(inc), math.sin(inc)
        cos_argp, sin_argp = math.cos(argp), math.sin(argp)

        # 位置转换
        x_eci = (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * x_orb + \
                (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * y_orb
        y_eci = (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * x_orb + \
                (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * y_orb
        z_eci = (sin_argp * sin_inc) * x_orb + \
                (cos_argp * sin_inc) * y_orb

        # 速度转换
        vx_eci = (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * vx_orb + \
                 (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * vy_orb
        vy_eci = (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * vx_orb + \
                 (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * vy_orb
        vz_eci = (sin_argp * sin_inc) * vx_orb + \
                 (cos_argp * sin_inc) * vy_orb

        # ECI到ECEF转换
        from sgp4.api import jday
        jd, fr = jday(
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute,
            timestamp.second + timestamp.microsecond / 1e6
        )
        d = jd - 2451545.0 + fr
        gmst = (18.697374558 + 24.06570982441908 * d) % 24
        gmst_deg = gmst * 15.0
        gmst_rad = math.radians(gmst_deg)

        cos_gmst, sin_gmst = math.cos(gmst_rad), math.sin(gmst_rad)

        x_ecef = x_eci * cos_gmst + y_eci * sin_gmst
        y_ecef = -x_eci * sin_gmst + y_eci * cos_gmst
        z_ecef = z_eci

        vx_ecef = vx_eci * cos_gmst + vy_eci * sin_gmst - self.EARTH_OMEGA * y_eci
        vy_ecef = -vx_eci * sin_gmst + vy_eci * cos_gmst + self.EARTH_OMEGA * x_eci
        vz_ecef = vz_eci

        # 转换为米
        return (
            (x_ecef * 1000, y_ecef * 1000, z_ecef * 1000),
            (vx_ecef * 1000, vy_ecef * 1000, vz_ecef * 1000)
        )

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

    def _propagate_with_orekit(
        self,
        satellite: Satellite,
        timestamp: datetime
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用Orekit Java后端传播卫星轨道（最高精度）

        通过JPype桥接Java Orekit库，提供与STK相当的精度。
        支持完整的摄动模型：J2/J3/J4、大气阻力、太阳光压等。

        Args:
            satellite: 卫星对象
            timestamp: 目标时刻

        Returns:
            (position, velocity) in ECEF (meters, m/s)

        Raises:
            RuntimeError: 如果Orekit计算失败
        """
        from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge, ensure_jvm_attached
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        try:
            # 获取Orekit桥接器实例（单例）
            bridge = OrekitJavaBridge(config=DEFAULT_OREKIT_CONFIG)

            # 确保JVM已启动
            bridge._ensure_jvm_started()

            # 挂载当前线程到JVM
            ensure_jvm_attached(lambda: None)()

            # 使用Orekit计算卫星状态
            # 注意：这里使用简化的接口，实际使用时可能需要扩展orekit_java_bridge
            # 以支持从六根数创建卫星并传播到指定时刻

            # 临时方案：使用Orekit的SGP4或数值传播器
            # 如果卫星有TLE，使用SGP4；否则使用数值传播
            if satellite.tle_line1 and satellite.tle_line2:
                # 使用Orekit的SGP4
                return self._propagate_with_orekit_sgp4(satellite, timestamp, bridge)
            else:
                # 使用Orekit数值传播器
                return self._propagate_with_orekit_numerical(satellite, timestamp, bridge)

        except Exception as e:
            raise RuntimeError(f"Orekit propagation failed: {e}")

    def _propagate_with_orekit_sgp4(
        self,
        satellite: Satellite,
        timestamp: datetime,
        bridge: 'OrekitJavaBridge'
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用Orekit的SGP4传播器"""
        from jpype import JClass

        try:
            # 获取Orekit类
            TimeScalesFactory = JClass('org.orekit.time.TimeScalesFactory')
            TLE = JClass('org.orekit.propagation.analytical.tle.TLE')
            SGP4 = JClass('org.orekit.propagation.analytical.tle.SGP4')
            AbsoluteDate = JClass('org.orekit.time.AbsoluteDate')
            FramesFactory = JClass('org.orekit.frames.FramesFactory')
            Transform = JClass('org.orekit.frames.Transform')

            # 解析TLE
            tle = TLE(satellite.tle_line1, satellite.tle_line2)

            # 创建SGP4传播器
            sgp4 = SGP4(tle)

            # 转换时间
            utc = TimeScalesFactory.getUTC()
            orekit_date = AbsoluteDate(
                timestamp.year, timestamp.month, timestamp.day,
                timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond / 1e6,
                utc
            )

            # 传播到目标时刻
            state = sgp4.propagate(orekit_date)

            # 获取位置和速度（在惯性系）
            position_itrf = state.getPVCoordinates().getPosition()
            velocity_itrf = state.getPVCoordinates().getVelocity()

            # 转换为米
            position = (position_itrf.getX(), position_itrf.getY(), position_itrf.getZ())
            velocity = (velocity_itrf.getX(), velocity_itrf.getY(), velocity_itrf.getZ())

            return position, velocity

        except Exception as e:
            raise RuntimeError(f"Orekit SGP4 propagation failed: {e}")

    def _propagate_with_orekit_numerical(
        self,
        satellite: Satellite,
        timestamp: datetime,
        bridge: 'OrekitJavaBridge'
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """使用Orekit数值传播器（从六根数）"""
        from jpype import JClass
        from core.orbit.visibility.orekit_config import IERSConventions
        import math

        try:
            # 获取Orekit类
            TimeScalesFactory = JClass('org.orekit.time.TimeScalesFactory')
            AbsoluteDate = JClass('org.orekit.time.AbsoluteDate')
            FramesFactory = JClass('org.orekit.frames.FramesFactory')
            KeplerianOrbit = JClass('org.orekit.orbits.KeplerianOrbit')
            PositionAngle = JClass('org.orekit.orbits.PositionAngle')
            NumericalPropagator = JClass('org.orekit.propagation.numerical.NumericalPropagator')
            DormandPrince853Integrator = JClass('org.hipparchus.ode.nonstiff.DormandPrince853Integrator')
            HolmesFeatherstoneAttractionModel = JClass('org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel')
            GravityFieldFactory = JClass('org.orekit.forces.gravity.potential.GravityFieldFactory')
            Constants = JClass('org.orekit.utils.Constants')

            orbit = satellite.orbit

            # 创建Orekit轨道
            a = orbit.get_semi_major_axis()  # 半长轴 (m)
            ecc = orbit.eccentricity
            inc = math.radians(orbit.inclination)
            raan = math.radians(orbit.raan)
            argp = math.radians(orbit.arg_of_perigee)
            mean_anomaly = math.radians(orbit.mean_anomaly)

            # 历元时间
            epoch = orbit.epoch if orbit.epoch else timestamp
            utc = TimeScalesFactory.getUTC()
            epoch_date = AbsoluteDate(
                epoch.year, epoch.month, epoch.day,
                epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6,
                utc
            )

            # 惯性系
            inertial_frame = FramesFactory.getEME2000()

            # 创建开普勒轨道
            keplerian_orbit = KeplerianOrbit(
                a,  # 半长轴 (m)
                ecc,  # 偏心率
                inc,  # 倾角 (rad)
                argp,  # 近地点幅角 (rad)
                raan,  # 升交点赤经 (rad)
                mean_anomaly,  # 平近点角 (rad)
                PositionAngle.MEAN,
                inertial_frame,
                epoch_date,
                Constants.WGS84_EARTH_MU
            )

            # 创建数值传播器
            min_step = 0.001
            max_step = 1000.0
            init_step = 60.0
            position_tolerance = 1.0

            integrator = DormandPrince853Integrator(
                min_step, max_step, init_step,
                position_tolerance, position_tolerance
            )

            propagator = NumericalPropagator(integrator)
            propagator.setInitialState(keplerian_orbit)

            # 添加J2摄动（使用重力场模型）
            gravity_field = GravityFieldFactory.getNormalizedProvider(10, 10)  # 10x10重力场
            gravity_model = HolmesFeatherstoneAttractionModel(inertial_frame, gravity_field)
            propagator.addForceModel(gravity_model)

            # 目标时间
            target_date = AbsoluteDate(
                timestamp.year, timestamp.month, timestamp.day,
                timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond / 1e6,
                utc
            )

            # 传播
            final_state = propagator.propagate(target_date)

            # 转换到ITRF（地固系）
            itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
            transform = inertial_frame.getTransformTo(itrf, target_date)

            pv_itrf = transform.transformPVCoordinates(final_state.getPVCoordinates())

            position = (pv_itrf.getPosition().getX(), pv_itrf.getPosition().getY(), pv_itrf.getPosition().getZ())
            velocity = (pv_itrf.getVelocity().getX(), pv_itrf.getVelocity().getY(), pv_itrf.getVelocity().getZ())

            return position, velocity

        except Exception as e:
            raise RuntimeError(f"Orekit numerical propagation failed: {e}")

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
