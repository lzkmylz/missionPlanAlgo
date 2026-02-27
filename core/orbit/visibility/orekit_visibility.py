"""
Orekit可见性计算器

基于纯Python实现的可见性计算（Orekit风格API）
设计文档第3.2章 - Orekit作为STK的fallback方案

功能：
1. 轨道传播：使用SGP4或数值方法传播卫星位置
2. 可见性计算：卫星-目标/地面站可见窗口计算
3. 几何计算：地球遮挡判断、仰角计算
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import math
import logging
from functools import lru_cache
import threading

from .base import VisibilityCalculator, VisibilityWindow

# 尝试导入SGP4
try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    Satrec = None
    jday = None

# 尝试导入OrekitJavaBridge
try:
    from .orekit_java_bridge import OrekitJavaBridge, OrbitPropagationError
    OREKIT_BRIDGE_AVAILABLE = True
except ImportError:
    OREKIT_BRIDGE_AVAILABLE = False
    OrekitJavaBridge = None
    OrbitPropagationError = Exception

# 导入配置
from .orekit_config import merge_config

logger = logging.getLogger(__name__)


class OrekitVisibilityCalculator(VisibilityCalculator):
    """
    Orekit风格可见性计算器

    纯Python实现，不依赖Java Orekit库：
    - 支持SGP4轨道传播（如果sgp4库可用）
    - 支持简化数值传播（fallback）
    - 完整的可见性几何计算
    - 地球遮挡判断
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Orekit可见性计算器

        Args:
            config: 配置参数
                - min_elevation: 最小仰角（度，默认5.0）
                - time_step: 计算时间步长（秒，默认60）
                - use_java_orekit: 是否使用Java Orekit（默认False）
                - orekit: Orekit配置字典
        """
        config = config or {}
        min_elevation = config.get('min_elevation', 5.0)
        super().__init__(min_elevation=min_elevation)

        self.time_step = config.get('time_step', 60)
        self._config = config

        # Phase 1: 自适应时间步长配置
        self.use_adaptive_step = config.get('use_adaptive_step', True)  # 默认启用
        self.coarse_step = config.get('coarse_step_seconds', 300)  # 粗扫描步长（秒）
        self.fine_step = config.get('fine_step_seconds', 60)  # 精化步长（秒）

        # 创建自适应步长计算器（如启用）
        if self.use_adaptive_step:
            from .adaptive_step_calculator import AdaptiveStepCalculator
            self._adaptive_calculator = AdaptiveStepCalculator(
                coarse_step=self.coarse_step,
                fine_step=self.fine_step
            )
        else:
            self._adaptive_calculator = None

        # Phase 3: 多线程并行配置
        self.use_parallel = config.get('use_parallel', True)  # 默认启用
        self.max_workers = config.get('max_workers', None)  # 线程数

        # 创建并行计算器（如启用）
        if self.use_parallel:
            from .parallel_calculator import ParallelVisibilityCalculator
            self._parallel_calculator = ParallelVisibilityCalculator(
                max_workers=self.max_workers
            )
        else:
            self._parallel_calculator = None

        # Phase 3: Java Orekit集成配置
        self.use_java_orekit = config.get('use_java_orekit', False)
        self.orekit_config = merge_config(config.get('orekit')) if config.get('orekit') else merge_config(None)

        # 缓存OrekitJavaBridge实例
        self._orekit_bridge: Optional[OrekitJavaBridge] = None
        if self.use_java_orekit and OREKIT_BRIDGE_AVAILABLE:
            try:
                self._orekit_bridge = OrekitJavaBridge(self.orekit_config)
                # 启动JVM（OrekitJavaBridge是延迟初始化，需要显式启动）
                self._orekit_bridge._ensure_jvm_started()
                logger.info("OrekitJavaBridge initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OrekitJavaBridge: {e}. Will fallback to simplified model.")
                self._orekit_bridge = None

        # Phase 2: 轨道和propagator缓存
        self._orbit_cache: Dict[str, Any] = {}
        self._propagator_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

    def _lla_to_ecef(self, longitude: float, latitude: float, altitude: float = 0.0) -> Tuple[float, float, float]:
        """
        将经纬度高度转换为地心固定坐标（ECEF）

        Args:
            longitude: 经度（度）
            latitude: 纬度（度）
            altitude: 海拔高度（米）

        Returns:
            (x, y, z) in meters
        """
        lon_rad = math.radians(longitude)
        lat_rad = math.radians(latitude)
        r = self.EARTH_RADIUS + altitude

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def _is_earth_occluded(self, sat_pos: Tuple[float, float, float],
                          target_pos: Tuple[float, float, float]) -> bool:
        """
        判断卫星到目标是否被地球遮挡

        几何方法：检查目标-卫星连线是否与地球相交（除目标点外）。

        Args:
            sat_pos: 卫星位置 (x, y, z) in meters
            target_pos: 目标位置 (x, y, z) in meters

        Returns:
            bool: 如果被地球遮挡返回True
        """
        # 如果位置相同，无遮挡
        if sat_pos == target_pos:
            return False

        # 检查卫星是否在地球内部
        r_sat = math.sqrt(sat_pos[0]**2 + sat_pos[1]**2 + sat_pos[2]**2)
        if r_sat <= self.EARTH_RADIUS:
            return True

        # 检查目标是否在地球表面或外部
        r_target = math.sqrt(target_pos[0]**2 + target_pos[1]**2 + target_pos[2]**2)
        if r_target < self.EARTH_RADIUS * 0.999:
            # 目标在地球内部，视为被遮挡
            return True

        # 几何遮挡判断
        # 向量：从目标指向卫星
        dx = sat_pos[0] - target_pos[0]
        dy = sat_pos[1] - target_pos[1]
        dz = sat_pos[2] - target_pos[2]

        # 目标位置向量（从地心指向目标）
        tx, ty, tz = target_pos

        # 计算目标-卫星连线到地心的最近距离
        # 连线参数方程: P = target + t * (sat - target)
        # 最近点满足: (target + t*(sat-target)) · (sat-target) = 0
        # 解得: t = -target · (sat-target) / |sat-target|^2

        dot_t_d = tx*dx + ty*dy + tz*dz
        d_squared = dx*dx + dy*dy + dz*dz

        if d_squared < 1e-6:  # 避免除零
            return False

        t = -dot_t_d / d_squared

        # 如果最近点不在线段内（不在目标和卫星之间），则连线不穿过地球内部
        if t < 0 or t > 1:
            # 最近点在目标后方（t < 0）或卫星前方（t > 1）
            # 当 t < 0 时，卫星和目标在同一方向（都从地心向外）
            # 当 t > 1 时，卫星在目标相反方向
            # 这两种情况连线都不穿过地球内部
            return False

        # 计算最近点坐标
        closest_x = tx + t * dx
        closest_y = ty + t * dy
        closest_z = tz + t * dz

        # 最近点到地心的距离
        closest_to_center = math.sqrt(closest_x**2 + closest_y**2 + closest_z**2)

        # 如果最近点在地球内部，则连线穿过地球
        # 使用0.999因子避免数值误差
        return closest_to_center < self.EARTH_RADIUS * 0.999

    def _propagate_satellite(self, satellite, dt: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        传播卫星到指定时间

        Phase 3: 根据配置自动选择传播器
        - 如果 use_java_orekit=True 且 JVM 可用，使用 Java Orekit
        - 否则使用简化模型

        Args:
            satellite: 卫星模型
            dt: 目标时间

        Returns:
            (position, velocity): 位置和速度，单位米和米/秒
        """
        # 如果有TLE且SGP4可用，使用SGP4（最高优先级）
        if SGP4_AVAILABLE and hasattr(satellite, 'tle_line1') and hasattr(satellite, 'tle_line2'):
            if satellite.tle_line1 and satellite.tle_line2:
                try:
                    satrec = Satrec.twoline2rv(satellite.tle_line1, satellite.tle_line2)
                    jd, fr = jday(dt.year, dt.month, dt.day,
                                  dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
                    error, position, velocity = satrec.sgp4(jd, fr)

                    if error == 0:
                        # 转换为米
                        pos_m = (position[0] * 1000, position[1] * 1000, position[2] * 1000)
                        vel_m = (velocity[0] * 1000, velocity[1] * 1000, velocity[2] * 1000)
                        return pos_m, vel_m
                except Exception:
                    pass  # 降级到简化模型

        # Phase 3: 如果启用Java Orekit，尝试使用
        if self.use_java_orekit:
            try:
                return self._propagate_with_java_orekit(satellite, dt)
            except Exception as e:
                logger.warning(f"Java Orekit propagation failed: {e}. Falling back to simplified model.")
                # 回退到简化模型
                return self._propagate_simplified(satellite, dt)

        # 使用简化轨道模型
        return self._propagate_simplified(satellite, dt)

    def _propagate_simplified(self, satellite, dt: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        简化的轨道传播模型（圆轨道近似）

        Args:
            satellite: 卫星模型
            dt: 目标时间

        Returns:
            (position, velocity) in meters and m/s
        """
        # 获取轨道参数
        orbit = getattr(satellite, 'orbit', None)
        if orbit is None:
            # 默认500km SSO
            altitude = 500000.0
            inclination = 97.4
            raan = 0.0
            mean_anomaly_offset = 0.0
        else:
            altitude = getattr(orbit, 'altitude', 500000.0)
            inclination = getattr(orbit, 'inclination', 97.4)
            raan = getattr(orbit, 'raan', 0.0)
            mean_anomaly_offset = getattr(orbit, 'mean_anomaly', 0.0)

        # 计算轨道半径和周期
        r = self.EARTH_RADIUS + altitude
        GM = 3.986004418e14  # 地球引力常数 m^3/s^2
        period = 2 * math.pi * math.sqrt(r**3 / GM)

        # 计算平均运动
        mean_motion = 2 * math.pi / period

        # 参考时间
        ref_time = datetime(2024, 1, 1)
        delta_t = (dt - ref_time).total_seconds()

        # 平近点角（考虑初始mean_anomaly）
        mean_anomaly = math.radians(mean_anomaly_offset) + mean_motion * delta_t

        # 轨道参数
        i = math.radians(inclination)
        raan_rad = math.radians(raan)

        # 圆轨道位置（在轨道平面内）
        x_orb = r * math.cos(mean_anomaly)
        y_orb = r * math.sin(mean_anomaly)

        # 转换到ECI坐标系
        x_eci = x_orb * math.cos(raan_rad) - y_orb * math.cos(i) * math.sin(raan_rad)
        y_eci = x_orb * math.sin(raan_rad) + y_orb * math.cos(i) * math.cos(raan_rad)
        z_eci = y_orb * math.sin(i)

        # 地球自转角速度 (rad/s)
        omega_earth = 7.2921159e-5
        # 地球自转角度 (从参考时间开始)
        # 添加初始偏移量使简化模型与Orekit的ITRF坐标对齐
        # 在2024-01-01 00:00:00 UTC时刻，Orekit的经度约为-100°
        # 但简化模型的初始经度为0°，所以需要反向偏移
        theta_0 = math.radians(100.0)  # 初始偏移量（反向）
        theta = theta_0 + omega_earth * delta_t

        # 将ECI坐标转换为ECEF坐标（考虑地球自转）
        x = x_eci * math.cos(theta) + y_eci * math.sin(theta)
        y = -x_eci * math.sin(theta) + y_eci * math.cos(theta)
        z = z_eci

        # 计算速度（圆轨道，包含地球自转影响）
        v = math.sqrt(GM / r)
        vx_orb = -v * math.sin(mean_anomaly)
        vy_orb = v * math.cos(mean_anomaly)

        vx_eci = vx_orb * math.cos(raan_rad) - vy_orb * math.cos(i) * math.sin(raan_rad)
        vy_eci = vx_orb * math.sin(raan_rad) + vy_orb * math.cos(i) * math.cos(raan_rad)
        vz_eci = vy_orb * math.sin(i)

        # 转换速度到ECEF坐标系
        vx = vx_eci * math.cos(theta) + vy_eci * math.sin(theta) - omega_earth * y
        vy = -vx_eci * math.sin(theta) + vy_eci * math.cos(theta) + omega_earth * x
        vz = vz_eci

        return ((x, y, z), (vx, vy, vz))

    def _ensure_jvm_and_bridge(self) -> None:
        """确保JVM和桥接器已初始化

        提取重复的JVM检查逻辑，用于Java Orekit方法。

        Raises:
            RuntimeError: 如果Java Orekit不可用或JVM启动失败
        """
        if not OREKIT_BRIDGE_AVAILABLE:
            raise RuntimeError("OrekitJavaBridge not available")

        if self._orekit_bridge is None:
            raise RuntimeError("OrekitJavaBridge not initialized")

        if not self._orekit_bridge.is_jvm_running():
            # 尝试启动JVM
            try:
                self._orekit_bridge._ensure_jvm_started()
            except Exception as e:
                raise RuntimeError(f"JVM not running and failed to start: {e}")

    def _propagate_with_java_orekit(self, satellite, dt: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        使用Java Orekit进行单点轨道传播

        Phase 3: 调用真正的Java Orekit进行高精度轨道传播

        Args:
            satellite: 卫星模型
            dt: 目标时间

        Returns:
            (position, velocity): 位置和速度，单位米和米/秒

        Raises:
            RuntimeError: 如果Java Orekit不可用或传播失败
        """
        self._ensure_jvm_and_bridge()

        # 使用批量传播获取单个时间点的数据
        # 为了获取单个时间点的精确位置，我们传播一个很短的时间范围
        start_time = dt - timedelta(seconds=1)
        end_time = dt + timedelta(seconds=1)

        results = self._propagate_range_with_java_orekit(
            satellite, start_time, end_time, timedelta(seconds=1)
        )

        if not results:
            raise RuntimeError("Java Orekit propagation returned no results")

        # 返回中间时间点（最接近目标时间）的结果
        # results是[(pos, vel, timestamp), ...]列表
        target_idx = len(results) // 2
        pos, vel, _ = results[target_idx]
        return pos, vel

    def _propagate_range_with_java_orekit(
        self,
        satellite,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], datetime]]:
        """
        使用Java Orekit进行批量轨道传播（高性能方案）

        Phase 3: 调用OrekitJavaBridge.propagate_batch()进行批量传播，
        避免Python for循环中多次JNI调用的开销。

        性能优化：使用BatchStepHandler，让Java引擎负责循环，
        将86,400次JNI调用降为1次（24小时/1秒步长）。

        Args:
            satellite: 卫星模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List of (position, velocity, timestamp)

        Raises:
            RuntimeError: 如果Java Orekit不可用或传播失败
            ValueError: 如果时间范围无效
        """
        # 验证时间范围
        if start_time > end_time:
            return []

        if start_time == end_time:
            return []

        self._ensure_jvm_and_bridge()

        try:
            # 计算步长（秒）
            step_seconds = time_step.total_seconds()
            if step_seconds <= 0:
                raise ValueError("time_step must be positive")

            # 计算总步数
            duration = (end_time - start_time).total_seconds()
            n_steps = int(duration / step_seconds) + 1

            # 获取卫星轨道参数
            orbit = getattr(satellite, 'orbit', None)
            if orbit is None:
                # 使用默认轨道参数
                altitude = 500000.0
                inclination = 97.4
                raan = 0.0
                mean_anomaly = 0.0
            else:
                altitude = getattr(orbit, 'altitude', 500000.0)
                inclination = getattr(orbit, 'inclination', 97.4)
                raan = getattr(orbit, 'raan', 0.0)
                mean_anomaly = getattr(orbit, 'mean_anomaly', 0.0)

            # 创建初始轨道（简化实现，实际应该使用完整的轨道元素）
            # 这里使用近似的轨道参数创建初始状态

            # 获取Orekit类
            try:
                AbsoluteDate = self._orekit_bridge._get_java_class("org.orekit.time.AbsoluteDate")
                TimeScalesFactory = self._orekit_bridge._get_java_class("org.orekit.time.TimeScalesFactory")
                FramesFactory = self._orekit_bridge._get_java_class("org.orekit.frames.FramesFactory")
                KeplerianOrbit = self._orekit_bridge._get_java_class("org.orekit.orbits.KeplerianOrbit")
                PVCoordinates = self._orekit_bridge._get_java_class("org.orekit.utils.PVCoordinates")
                Vector3D = self._orekit_bridge._get_java_class("org.hipparchus.geometry.euclidean.threed.Vector3D")
                SpacecraftState = self._orekit_bridge._get_java_class("org.orekit.propagation.SpacecraftState")
            except Exception as e:
                raise RuntimeError(f"Failed to get Orekit classes: {e}")

            # 创建时间尺度
            utc = TimeScalesFactory.getUTC()

            # 使用固定历元（任务开始时间）创建轨道
            # 这样轨道参数（平近点角）是相对于固定时间的
            # 使用传播开始时间作为历元，避免硬编码
            epoch_date = AbsoluteDate(
                start_time.year, start_time.month, start_time.day,
                start_time.hour, start_time.minute, start_time.second,
                utc
            )

            # 创建传播起止时间的AbsoluteDate
            start_date = AbsoluteDate(
                start_time.year, start_time.month, start_time.day,
                start_time.hour, start_time.minute, start_time.second + start_time.microsecond / 1e6,
                utc
            )

            end_date = AbsoluteDate(
                end_time.year, end_time.month, end_time.day,
                end_time.hour, end_time.minute, end_time.second + end_time.microsecond / 1e6,
                utc
            )

            # 获取坐标系
            frame = FramesFactory.getEME2000()

            # 计算轨道参数
            r = self.EARTH_RADIUS + altitude
            GM = 3.986004418e14  # 地球引力常数 m^3/s^2
            import math

            # 使用轨道六根数直接创建轨道（避免PVCoordinates反推的参数偏差）
            # Orekit KeplerianOrbit: (a, e, i, pa, raan, anomaly, type, frame, date, mu)
            a = r  # 半长轴（圆轨道，等于半径）
            e = 0.0  # 偏心率（圆轨道）
            i = math.radians(inclination)  # 倾角
            pa = 0.0  # 近地点幅角（圆轨道任意）
            raan_rad = math.radians(raan)  # 升交点赤经
            ma_rad = math.radians(mean_anomaly)  # 平近点角

            # 创建开普勒轨道（使用固定历元）
            try:
                PositionAngleType = self._orekit_bridge._get_java_class(
                    "org.orekit.orbits.PositionAngleType"
                )
                mean_angle_type = PositionAngleType.MEAN
            except Exception as e:
                logger.warning(f"Failed to get PositionAngleType: {e}. Falling back to simplified model.")
                raise RuntimeError(f"Failed to create Keplerian orbit: {e}")

            orbit = KeplerianOrbit(
                a, e, i, pa, raan_rad, ma_rad,
                mean_angle_type,  # 指定anomaly类型为平近点角
                frame, epoch_date, GM
            )

            # 创建数值传播器
            propagator = self._orekit_bridge.create_numerical_propagator(
                orbit, epoch_date, frame, self.orekit_config
            )

            # 使用批量传播
            results_array = self._orekit_bridge.propagate_batch(
                propagator, start_date, end_date, step_seconds
            )

            # 获取ECEF坐标系(ITRF)用于坐标转换
            itrf = self._orekit_bridge.get_frame("ITRF")

            # 获取J2000_EPOCH常量
            J2000_EPOCH = self._orekit_bridge._get_java_class(
                "org.orekit.time.AbsoluteDate"
            ).J2000_EPOCH

            # 转换结果为Python对象（从EME2000转换到ITRF/ECEF）
            results = []
            for i, row in enumerate(results_array):
                # row格式: [seconds_since_j2000, px, py, pz, vx, vy,vz]
                # 位置和速度在EME2000坐标系中
                eme2000_pos = Vector3D(float(row[1]), float(row[2]), float(row[3]))
                eme2000_vel = Vector3D(float(row[4]), float(row[5]), float(row[6]))
                eme2000_pv = PVCoordinates(eme2000_pos, eme2000_vel)

                # 创建当前时间的AbsoluteDate
                current_seconds = float(row[0])
                current_date = AbsoluteDate(J2000_EPOCH, current_seconds, utc)

                # 从EME2000转换到ITRF
                transform = frame.getTransformTo(itrf, current_date)
                itrf_pv = transform.transformPVCoordinates(eme2000_pv)
                itrf_pos = itrf_pv.getPosition()
                itrf_vel = itrf_pv.getVelocity()

                pos = (itrf_pos.getX(), itrf_pos.getY(), itrf_pos.getZ())
                vel = (itrf_vel.getX(), itrf_vel.getY(), itrf_vel.getZ())

                # 计算时间戳
                timestamp = start_time + timedelta(seconds=i * step_seconds)
                results.append((pos, vel, timestamp))

            return results

        except Exception as e:
            logger.error(f"Java Orekit range propagation failed: {e}")
            raise RuntimeError(f"Java Orekit propagation failed: {e}") from e

    def _propagate_range(self, satellite, start_time: datetime, end_time: datetime,
                        time_step: timedelta) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], datetime]]:
        """
        传播时间序列

        Phase 1 Optimization: 当 use_java_orekit=True 时，直接使用
        _propagate_range_with_java_orekit 进行批量传播，避免循环调用
        _propagate_satellite 导致的重复JNI开销。

        Args:
            satellite: 卫星模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List of (position, velocity, timestamp)
        """
        # 验证卫星对象有效性
        if satellite is None:
            return []

        # Phase 1: 批量传播优化
        # 当启用Java Orekit时，直接使用批量传播API
        if self.use_java_orekit and OREKIT_BRIDGE_AVAILABLE and self._orekit_bridge is not None:
            try:
                return self._propagate_range_with_java_orekit(
                    satellite, start_time, end_time, time_step
                )
            except Exception as e:
                logger.warning(f"Java Orekit batch propagation failed: {e}. Falling back to simplified model.")
                # 回退到简化模型（继续执行下面的代码）

        # 原始实现：逐点传播（简化模型或SGP4）
        # 验证时间步长为正数，避免无限循环
        if time_step.total_seconds() <= 0:
            return []

        positions = []
        current_time = start_time

        while current_time <= end_time:
            try:
                pos, vel = self._propagate_satellite(satellite, current_time)
                positions.append((pos, vel, current_time))
            except Exception:
                # 跳过传播失败的点
                pass
            current_time += time_step

        return positions

    def compute_satellite_target_windows(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime,
        time_step: Optional[timedelta] = None
    ) -> List[VisibilityWindow]:
        """
        计算卫星-目标可见窗口

        Args:
            satellite: 卫星模型
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长（默认使用config中的time_step）

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 验证时间范围
        if start_time >= end_time:
            return []

        # Phase 1: 使用自适应步长（如果启用）
        if self.use_adaptive_step and self._adaptive_calculator is not None:
            return self._compute_with_adaptive_step(
                satellite, target, start_time, end_time
            )

        # 使用传入的time_step或config中的time_step
        if time_step is None:
            time_step = timedelta(seconds=self.time_step)

        try:
            # 获取目标ECEF位置
            if hasattr(target, 'get_ecef_position'):
                target_pos = target.get_ecef_position()
            else:
                # 从经纬度计算
                lon = getattr(target, 'longitude', 0.0)
                lat = getattr(target, 'latitude', 0.0)
                alt = getattr(target, 'altitude', 0.0)
                target_pos = self._lla_to_ecef(lon, lat, alt)

            # 传播卫星轨道
            sat_positions = self._propagate_range(satellite, start_time, end_time, time_step)

            if not sat_positions:
                return []

            # 计算每个时间点的仰角
            elevation_data = []
            for sat_pos, vel, dt in sat_positions:
                # 检查地球遮挡
                if self._is_earth_occluded(sat_pos, target_pos):
                    elevation_data.append((dt, -90.0))  # 被遮挡，仰角为负
                else:
                    # 计算仰角
                    elevation = self._calculate_elevation(sat_pos, target_pos)
                    elevation_data.append((dt, elevation))

            # 提取可见窗口
            target_id = getattr(target, 'id', 'UNKNOWN')
            sat_id = getattr(satellite, 'id', 'UNKNOWN')

            return self._find_windows_from_elevations(
                sat_id, target_id, elevation_data, self.min_elevation
            )

        except Exception:
            # 发生错误时返回空列表
            return []

    def _compute_with_adaptive_step(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime
    ) -> List[VisibilityWindow]:
        """
        使用自适应步长计算可见窗口

        Phase 1优化：先用粗步长定位，再用细步长精化

        Args:
            satellite: 卫星模型
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        try:
            # 获取目标ECEF位置
            if hasattr(target, 'get_ecef_position'):
                target_pos = target.get_ecef_position()
            else:
                lon = getattr(target, 'longitude', 0.0)
                lat = getattr(target, 'latitude', 0.0)
                alt = getattr(target, 'altitude', 0.0)
                target_pos = self._lla_to_ecef(lon, lat, alt)

            target_id = getattr(target, 'id', 'UNKNOWN')
            sat_id = getattr(satellite, 'id', 'UNKNOWN')

            # 定义可见性检查函数
            def is_visible_at_time(dt: datetime) -> bool:
                try:
                    sat_pos, _ = self._propagate_satellite(satellite, dt)
                    if self._is_earth_occluded(sat_pos, target_pos):
                        return False
                    elevation = self._calculate_elevation(sat_pos, target_pos)
                    return elevation >= self.min_elevation
                except Exception:
                    return False

            # 使用自适应步长计算器
            window_times = self._adaptive_calculator.compute_windows(
                is_visible_at_time,
                start_time,
                end_time
            )

            # 转换为VisibilityWindow对象
            windows = []
            for window_start, window_end in window_times:
                # 计算窗口中点的仰角作为质量评分
                mid_time = window_start + (window_end - window_start) / 2
                try:
                    sat_pos, _ = self._propagate_satellite(satellite, mid_time)
                    elevation = self._calculate_elevation(sat_pos, target_pos)
                    quality_score = min(1.0, elevation / 90.0)
                except Exception:
                    quality_score = 0.5

                windows.append(VisibilityWindow(
                    satellite_id=sat_id,
                    target_id=target_id,
                    start_time=window_start,
                    end_time=window_end,
                    duration_seconds=(window_end - window_start).total_seconds(),
                    max_elevation=elevation if 'elevation' in dir() else 0.0,
                    quality_score=quality_score
                ))

            return windows

        except Exception as e:
            logger.warning(f"Adaptive step calculation failed: {e}. Falling back to fixed step.")
            # 回退到固定步长
            return self.compute_satellite_target_windows(
                satellite, target, start_time, end_time,
                time_step=timedelta(seconds=self.time_step)
            )

    def compute_satellite_ground_station_windows(
        self,
        satellite,
        ground_station,
        start_time: datetime,
        end_time: datetime,
        time_step: Optional[timedelta] = None
    ) -> List[VisibilityWindow]:
        """
        计算卫星-地面站可见窗口

        Args:
            satellite: 卫星模型
            ground_station: 地面站模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长（默认使用config中的time_step）

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 验证时间范围
        if start_time >= end_time:
            return []

        # 使用传入的time_step或config中的time_step
        if time_step is None:
            time_step = timedelta(seconds=self.time_step)

        try:
            # 获取地面站ECEF位置
            if hasattr(ground_station, 'get_ecef_position'):
                gs_pos = ground_station.get_ecef_position()
            else:
                # 从经纬度计算
                lon = getattr(ground_station, 'longitude', 0.0)
                lat = getattr(ground_station, 'latitude', 0.0)
                alt = getattr(ground_station, 'altitude', 0.0)
                gs_pos = self._lla_to_ecef(lon, lat, alt)

            # 传播卫星轨道
            sat_positions = self._propagate_range(satellite, start_time, end_time, time_step)

            if not sat_positions:
                return []

            # 计算每个时间点的仰角
            elevation_data = []
            for sat_pos, vel, dt in sat_positions:
                # 检查地球遮挡
                if self._is_earth_occluded(sat_pos, gs_pos):
                    elevation_data.append((dt, -90.0))
                else:
                    # 计算仰角
                    elevation = self._calculate_elevation(sat_pos, gs_pos)
                    elevation_data.append((dt, elevation))

            # 提取可见窗口
            gs_id = getattr(ground_station, 'id', 'UNKNOWN')
            sat_id = getattr(satellite, 'id', 'UNKNOWN')

            return self._find_windows_from_elevations(
                sat_id, f"GS:{gs_id}", elevation_data, self.min_elevation
            )

        except Exception:
            return []

    def calculate_windows(
        self,
        satellite_id: str,
        target_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[VisibilityWindow]:
        """
        计算可见窗口（简化接口）

        注意：此接口需要外部提供卫星和目标对象，
        实际实现中可能需要从数据库或缓存获取对象

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 此简化接口在没有实际卫星/目标对象时返回空列表
        # 实际使用时需要传入真实对象或从数据库获取
        # 子类可以覆盖此方法提供具体实现
        return []

    def is_visible(
        self,
        satellite_id: str,
        target_id: str,
        time: datetime
    ) -> bool:
        """
        检查指定时间是否可见

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            time: 检查时间

        Returns:
            bool: 是否可见
        """
        # 获取时间前后一小段时间的窗口
        window_start = time - timedelta(minutes=1)
        window_end = time + timedelta(minutes=1)

        windows = self.calculate_windows(satellite_id, target_id, window_start, window_end)

        # 检查时间是否在任一窗口内
        for window in windows:
            if window.start_time <= time <= window.end_time:
                return True

        return False

    # =========================================================================
    # Phase 2: Caching Methods
    # =========================================================================

    def _get_cached_orbit(self, satellite_id: str, orbit_hash: str) -> Optional[Any]:
        """从缓存获取轨道对象

        Args:
            satellite_id: 卫星ID
            orbit_hash: 轨道参数哈希

        Returns:
            缓存的轨道对象或None
        """
        cache_key = f"{satellite_id}:{orbit_hash}"
        with self._cache_lock:
            return self._orbit_cache.get(cache_key)

    def _set_cached_orbit(self, satellite_id: str, orbit_hash: str, orbit: Any) -> None:
        """缓存轨道对象

        Args:
            satellite_id: 卫星ID
            orbit_hash: 轨道参数哈希
            orbit: 轨道对象
        """
        cache_key = f"{satellite_id}:{orbit_hash}"
        with self._cache_lock:
            self._orbit_cache[cache_key] = orbit

    def _get_cached_propagator(self, satellite_id: str, orbit_hash: str) -> Optional[Any]:
        """从缓存获取propagator对象

        Args:
            satellite_id: 卫星ID
            orbit_hash: 轨道参数哈希

        Returns:
            缓存的propagator对象或None
        """
        cache_key = f"{satellite_id}:{orbit_hash}"
        with self._cache_lock:
            return self._propagator_cache.get(cache_key)

    def _set_cached_propagator(self, satellite_id: str, orbit_hash: str, propagator: Any) -> None:
        """缓存propagator对象

        Args:
            satellite_id: 卫星ID
            orbit_hash: 轨道参数哈希
            propagator: propagator对象
        """
        cache_key = f"{satellite_id}:{orbit_hash}"
        with self._cache_lock:
            self._propagator_cache[cache_key] = propagator

    def clear_cache(self) -> None:
        """清除所有缓存"""
        with self._cache_lock:
            self._orbit_cache.clear()
            self._propagator_cache.clear()
            logger.info("Orbit and propagator cache cleared")

    # =========================================================================
    # Phase 3: Parallel Computation Methods
    # =========================================================================

    def compute_visibility_parallel(
        self,
        satellites: List[Any],
        target: Any,
        start_time: datetime,
        end_time: datetime,
        time_step: Optional[timedelta] = None,
        max_workers: int = 4
    ) -> Dict[str, List[VisibilityWindow]]:
        """并行计算多卫星可见性

        Args:
            satellites: 卫星列表
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长
            max_workers: 最大并行工作线程数

        Returns:
            Dict[str, List[VisibilityWindow]]: 每个卫星的可见窗口列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: Dict[str, List[VisibilityWindow]] = {}

        def compute_single(satellite: Any) -> Tuple[str, List[VisibilityWindow]]:
            """计算单个卫星的可见性"""
            sat_id = getattr(satellite, 'id', 'UNKNOWN')
            try:
                windows = self.compute_satellite_target_windows(
                    satellite, target, start_time, end_time, time_step
                )
                return sat_id, windows
            except Exception as e:
                logger.error(f"Error computing visibility for {sat_id}: {e}")
                return sat_id, []

        # 使用线程池并行计算
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sat = {
                executor.submit(compute_single, sat): sat
                for sat in satellites
            }

            # 收集结果
            for future in as_completed(future_to_sat):
                sat_id, windows = future.result()
                results[sat_id] = windows

        return results
