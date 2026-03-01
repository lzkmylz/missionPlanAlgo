"""
Orekit 批量轨道传播器 - 高性能姿态角计算

优化策略：
1. 单例 JVM：整个应用生命周期只启动一次
2. 时间窗口批量传播：一次传播整个任务周期
3. 插值获取：从批量结果插值获取特定时刻
4. LRU 缓存：缓存卫星传播结果

使用示例:
    propagator = OrekitBatchPropagator()

    # 预计算卫星在整个任务周期的轨道
    propagator.precompute_satellite_orbit(satellite, start_time, end_time, time_step=60)

    # 快速获取任意时刻状态（插值）
    position, velocity = propagator.get_state_at_time(satellite.id, imaging_time)
"""

import logging
import threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from functools import lru_cache

from core.models.satellite import Satellite

try:
    from core.orbit.visibility.orekit_java_bridge import (
        OrekitJavaBridge, ensure_jvm_attached, JPYPE_AVAILABLE
    )
    from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG
    OREKIT_AVAILABLE = JPYPE_AVAILABLE
except ImportError:
    OREKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SatelliteOrbitCache:
    """卫星轨道缓存

    存储单个卫星在预计算时间窗口内的轨道状态，
    支持快速插值查询。
    """

    def __init__(self,
                 satellite_id: str,
                 timestamps: List[datetime],
                 positions: List[Tuple[float, float, float]],
                 velocities: List[Tuple[float, float, float]]):
        """
        Args:
            satellite_id: 卫星ID
            timestamps: 时间戳列表（已排序）
            positions: 位置列表 (ECEF, meters)
            velocities: 速度列表 (ECEF, m/s)
        """
        self.satellite_id = satellite_id
        self.timestamps = timestamps
        self.positions = positions
        self.velocities = velocities
        self.start_time = timestamps[0] if timestamps else None
        self.end_time = timestamps[-1] if timestamps else None

    def contains_time(self, query_time: datetime) -> bool:
        """检查时间是否在缓存范围内"""
        if self.start_time is None or self.end_time is None:
            return False
        return self.start_time <= query_time <= self.end_time

    def get_state_at_time(self, query_time: datetime) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """使用线性插值获取指定时刻的状态

        Args:
            query_time: 查询时间

        Returns:
            (position, velocity) 或 None
        """
        if not self.contains_time(query_time):
            return None

        if query_time in self.timestamps:
            idx = self.timestamps.index(query_time)
            return self.positions[idx], self.velocities[idx]

        # 找到相邻时间点
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

        return None


class OrekitBatchPropagator:
    """Orekit 批量轨道传播器

    高性能批量计算卫星轨道，支持缓存和插值。
    单例模式管理 JVM 生命周期。
    """

    _instance: Optional['OrekitBatchPropagator'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, config: Optional[Dict] = None) -> 'OrekitBatchPropagator':
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, config: Optional[Dict] = None) -> None:
        """初始化传播器"""
        self._config = config or DEFAULT_OREKIT_CONFIG
        self._orbit_cache: Dict[str, SatelliteOrbitCache] = {}
        self._bridge: Optional[OrekitJavaBridge] = None
        self._jvm_initialized: bool = False

        if not OREKIT_AVAILABLE:
            logger.warning("Orekit not available, batch propagator disabled")

    def _ensure_jvm(self) -> bool:
        """确保 JVM 已启动"""
        if not OREKIT_AVAILABLE:
            return False

        if not self._jvm_initialized:
            try:
                self._bridge = OrekitJavaBridge(config=self._config)
                self._bridge._ensure_jvm_started()
                self._jvm_initialized = True
                logger.info("JVM initialized for batch propagator")
            except Exception as e:
                logger.error(f"Failed to initialize JVM: {e}")
                return False

        return True

    def precompute_satellite_orbit(
        self,
        satellite: Satellite,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=1),
        force_recompute: bool = False
    ) -> bool:
        """预计算卫星轨道

        批量传播卫星轨道并缓存结果，后续查询使用插值。

        Args:
            satellite: 卫星对象
            start_time: 开始时间
            end_time: 结束时间
            time_step: 传播时间步长（默认1秒）
            force_recompute: 强制重新计算（忽略缓存）

        Returns:
            bool: 计算是否成功
        """
        if not self._ensure_jvm():
            return False

        sat_id = satellite.id

        # 检查缓存
        if not force_recompute and sat_id in self._orbit_cache:
            cache = self._orbit_cache[sat_id]
            if cache.start_time <= start_time and cache.end_time >= end_time:
                logger.debug(f"Using cached orbit for {sat_id}")
                return True

        try:
            # 挂载线程到 JVM
            ensure_jvm_attached(lambda: None)()

            # 批量传播
            logger.info(f"Precomputing orbit for {sat_id} from {start_time} to {end_time}")
            timestamps, positions, velocities = self._batch_propagate(
                satellite, start_time, end_time, time_step
            )

            # 缓存结果
            self._orbit_cache[sat_id] = SatelliteOrbitCache(
                satellite_id=sat_id,
                timestamps=timestamps,
                positions=positions,
                velocities=velocities
            )

            logger.info(f"Cached {len(timestamps)} orbit points for {sat_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to precompute orbit for {sat_id}: {e}")
            return False

    def _batch_propagate(
        self,
        satellite: Satellite,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta
    ) -> Tuple[List[datetime], List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """批量传播卫星轨道

        使用 Orekit 数值传播器批量计算轨道状态。

        Returns:
            (timestamps, positions, velocities)
        """
        from jpype import JClass
        import math

        # 获取 Orekit 类
        TimeScalesFactory = JClass('org.orekit.time.TimeScalesFactory')
        AbsoluteDate = JClass('org.orekit.time.AbsoluteDate')
        FramesFactory = JClass('org.orekit.frames.FramesFactory')
        KeplerianOrbit = JClass('org.orekit.orbits.KeplerianOrbit')
        NumericalPropagator = JClass('org.orekit.propagation.numerical.NumericalPropagator')
        DormandPrince853Integrator = JClass('org.hipparchus.ode.nonstiff.DormandPrince853Integrator')
        HolmesFeatherstoneAttractionModel = JClass('org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel')
        GravityFieldFactory = JClass('org.orekit.forces.gravity.potential.GravityFieldFactory')
        Constants = JClass('org.orekit.utils.Constants')
        IERSConventions = JClass('org.orekit.utils.IERSConventions')

        orbit = satellite.orbit
        if orbit is None:
            raise ValueError(f"Satellite {satellite.id} has no orbit data")

        # 创建 Orekit 轨道
        a = orbit.get_semi_major_axis()
        ecc = orbit.eccentricity
        inc = math.radians(orbit.inclination)
        raan = math.radians(orbit.raan)
        argp = math.radians(orbit.arg_of_perigee)
        mean_anomaly = math.radians(orbit.mean_anomaly)

        # 从平近点角计算真近点角（用于 KeplerianOrbit 构造函数）
        # 使用简化公式：nu = M + 2*e*sin(M) （适用于小偏心率）
        if ecc < 0.1:
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

        utc = TimeScalesFactory.getUTC()
        epoch = orbit.epoch if orbit.epoch else start_time
        epoch_date = AbsoluteDate(
            epoch.year, epoch.month, epoch.day,
            epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6,
            utc
        )

        inertial_frame = FramesFactory.getEME2000()

        # 获取 PositionAngleType 枚举
        PositionAngleType = JClass('org.orekit.orbits.PositionAngleType')

        # 使用真近点角创建轨道
        keplerian_orbit = KeplerianOrbit(
            float(a), float(ecc), float(inc), float(argp), float(raan), float(true_anomaly),
            PositionAngleType.TRUE,
            inertial_frame, epoch_date, float(Constants.WGS84_EARTH_MU)
        )

        # 创建SpacecraftState作为初始状态
        SpacecraftState = JClass('org.orekit.propagation.SpacecraftState')
        initial_state = SpacecraftState(keplerian_orbit)

        # 创建数值传播器
        min_step = 0.001
        max_step = time_step.total_seconds()
        init_step = time_step.total_seconds() / 2
        position_tolerance = 1.0

        # 创建积分器 - 使用正确的构造函数
        integrator = DormandPrince853Integrator(
            float(min_step), float(max_step),
            float(init_step), float(position_tolerance)
        )

        propagator = NumericalPropagator(integrator)
        propagator.setInitialState(initial_state)

        # 添加重力场模型（J2/J3/J4）
        gravity_field = GravityFieldFactory.getNormalizedProvider(10, 10)
        gravity_model = HolmesFeatherstoneAttractionModel(inertial_frame, gravity_field)
        propagator.addForceModel(gravity_model)

        # 批量传播
        timestamps = []
        positions = []
        velocities = []

        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

        current_time = start_time
        while current_time <= end_time:
            target_date = AbsoluteDate(
                current_time.year, current_time.month, current_time.day,
                current_time.hour, current_time.minute,
                current_time.second + current_time.microsecond / 1e6,
                utc
            )

            # 传播
            state = propagator.propagate(target_date)

            # 转换到 ITRF
            transform = inertial_frame.getTransformTo(itrf, target_date)
            pv_itrf = transform.transformPVCoordinates(state.getPVCoordinates())

            pos = pv_itrf.getPosition()
            vel = pv_itrf.getVelocity()

            timestamps.append(current_time)
            positions.append((pos.getX(), pos.getY(), pos.getZ()))
            velocities.append((vel.getX(), vel.getY(), vel.getZ()))

            current_time += time_step

        return timestamps, positions, velocities

    def get_state_at_time(
        self,
        satellite_id: str,
        query_time: datetime
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """获取指定时刻的卫星状态

        从缓存中使用插值获取状态。

        Args:
            satellite_id: 卫星ID
            query_time: 查询时间

        Returns:
            (position, velocity) 或 None
        """
        if satellite_id not in self._orbit_cache:
            return None

        cache = self._orbit_cache[satellite_id]
        return cache.get_state_at_time(query_time)

    def clear_cache(self, satellite_id: Optional[str] = None) -> None:
        """清除缓存

        Args:
            satellite_id: 指定卫星ID，None表示清除所有
        """
        if satellite_id is None:
            self._orbit_cache.clear()
            logger.info("Cleared all orbit cache")
        elif satellite_id in self._orbit_cache:
            del self._orbit_cache[satellite_id]
            logger.info(f"Cleared orbit cache for {satellite_id}")

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计

        Returns:
            缓存统计字典
        """
        return {
            'cached_satellites': len(self._orbit_cache),
            'total_points': sum(len(cache.timestamps) for cache in self._orbit_cache.values())
        }


# 全局传播器实例
_batch_propagator: Optional[OrekitBatchPropagator] = None


def get_batch_propagator() -> Optional[OrekitBatchPropagator]:
    """获取全局批量传播器实例"""
    global _batch_propagator
    if _batch_propagator is None and OREKIT_AVAILABLE:
        _batch_propagator = OrekitBatchPropagator()
    return _batch_propagator
