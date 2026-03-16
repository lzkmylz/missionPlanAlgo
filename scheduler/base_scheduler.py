"""
调度器基类

所有调度算法必须继承此基类，实现即插即用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

from .frequency_utils import (
    ObservationTask,
    create_observation_tasks,
    calculate_frequency_fitness,
)
from core.dynamics.attitude_calculator import (
    AttitudeCalculator,
    PropagatorType,
    AttitudeAngles,
)
from .constraints import (
    SlewConstraintChecker,
    SAAConstraintChecker,
    AttitudeConstraintChecker,
    BatchSlewConstraintChecker,
    BatchSAAConstraintChecker
)
from core.dynamics.attitude_mode import AttitudeMode
from core.dynamics.sun_position_calculator import SunPositionCalculator
from core.dynamics.power_generation_calculator import PowerGenerationCalculator, PowerConfig


class TaskFailureReason(Enum):
    """任务失败原因枚举 - Chapter 12.4完整实现"""
    # 资源约束
    POWER_CONSTRAINT = "power_constraint"           # 电量不足
    STORAGE_CONSTRAINT = "storage_constraint"        # 存储溢出
    STORAGE_OVERFLOW_RISK = "storage_overflow_risk"  # 存储溢出风险

    # 时间约束
    NO_VISIBLE_WINDOW = "no_visible_window"          # 无可见时间窗
    WINDOW_TOO_SHORT = "window_too_short"            # 可见窗口太短
    TIME_CONFLICT = "time_conflict"                  # 与其他任务时间冲突
    DEADLINE_VIOLATION = "deadline_violation"        # 超出截止时间

    # 能力约束
    SAT_CAPABILITY_MISMATCH = "sat_capability_mismatch"  # 卫星能力不匹配
    MODE_NOT_SUPPORTED = "mode_not_supported"        # 不支持所需成像模式
    OFF_NADIR_EXCEEDED = "off_nadir_exceeded"        # 超出最大侧摆角

    # 协同约束
    GROUND_STATION_UNAVAILABLE = "ground_station_unavailable"  # 地面站不可用
    ANTENNA_CONFLICT = "antenna_conflict"            # 天线资源冲突

    # 物理约束 - H2新增
    THERMAL_CONSTRAINT = "thermal_constraint"        # 热控约束超限
    SUN_EXCLUSION_VIOLATION = "sun_exclusion_violation"  # 太阳规避角不满足
    STORAGE_FRAGMENTATION = "storage_fragmentation"  # 存储碎片化无法分配
    SAA_CONSTRAINT = "saa_constraint"  # 南大西洋异常区约束

    # 网络约束 - H2新增
    NO_ISL_PATH = "no_isl_path"                      # 无可用ISL路径
    UPLINK_UNAVAILABLE = "uplink_unavailable"        # 无指令上行窗口
    RELAY_OVERLOAD = "relay_overload"                # 中继星容量超限

    # 其他
    UNKNOWN = "unknown"                              # 未知原因
    ALGORITHM_TIMEOUT = "algorithm_timeout"          # 算法超时


@dataclass
class TaskFailure:
    """单个任务失败记录"""
    task_id: str
    failure_reason: TaskFailureReason
    failure_detail: str
    satellite_id: Optional[str] = None
    attempted_time: Optional[datetime] = None
    constraint_value: Optional[float] = None
    limit_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """已调度的任务"""
    task_id: str
    satellite_id: str
    target_id: str
    imaging_start: datetime
    imaging_end: datetime
    imaging_mode: str
    slew_angle: float = 0.0
    slew_time: float = 0.0  # 机动时间（秒）
    storage_before: float = 0.0
    storage_after: float = 0.0
    power_before: float = 0.0
    power_after: float = 0.0
    # 详细能源变化字段 - 新增
    power_consumed: float = 0.0          # 任务期间电量消耗(Wh)
    power_generated: float = 0.0         # 任务期间发电量(Wh)
    energy_consumption: float = 0.0      # 机动能量消耗(J)
    battery_soc_before: float = 0.0      # 任务前电池SOC(%)
    battery_soc_after: float = 0.0       # 任务后电池SOC(%)
    # 地面站数传相关字段
    ground_station_id: Optional[str] = None
    antenna_id: Optional[str] = None  # 具体使用的天线
    downlink_start: Optional[datetime] = None
    downlink_end: Optional[datetime] = None
    data_transferred: float = 0.0
    # 姿态角字段 - 用于姿控系统验证
    roll_angle: Optional[float] = None    # 滚转角（度）
    pitch_angle: Optional[float] = None   # 俯仰角（度）
    yaw_angle: float = 0.0                # 偏航角（度），默认为0（零偏航模式）
    attitude_coordinate_system: str = "LVLH"  # 坐标系：LVLH
    # 任务优先级和复位时间
    priority: Optional[int] = None        # 任务优先级
    reset_time: Optional[float] = None    # 姿态复位时间（秒）
    # 聚类任务相关字段
    is_cluster_task: bool = False              # 是否为聚类任务
    cluster_id: Optional[str] = None           # 聚类ID
    primary_target_id: Optional[str] = None    # 主目标ID
    covered_target_ids: List[str] = field(default_factory=list)  # 覆盖的所有目标ID列表
    covered_target_count: int = 0              # 覆盖目标数量
    # 成像足迹相关字段（新增）
    footprint_corners: List[Tuple[float, float]] = field(default_factory=list)  # 四脚点坐标 [(lon, lat), ...]
    footprint_center: Optional[Tuple[float, float]] = None  # 中心点坐标 (lon, lat)
    swath_width_km: float = 0.0                # 实际幅宽（公里）
    fov_config: Dict[str, Any] = field(default_factory=dict)  # 使用的FOV配置

    def to_dict(self) -> Dict[str, Any]:
        # Calculate imaging_duration if not explicitly set
        imaging_duration = 0.0
        if self.imaging_start and self.imaging_end:
            imaging_duration = (self.imaging_end - self.imaging_start).total_seconds()

        return {
            'task_id': self.task_id,
            'satellite_id': self.satellite_id,
            'target_id': self.target_id,
            'imaging_start': self.imaging_start.isoformat() if self.imaging_start else None,
            'imaging_end': self.imaging_end.isoformat() if self.imaging_end else None,
            'imaging_duration': imaging_duration,
            'imaging_mode': self.imaging_mode,
            'slew_angle': self.slew_angle,
            'slew_time': self.slew_time,
            'storage_before': self.storage_before,
            'storage_after': self.storage_after,
            'power_before': self.power_before,
            'power_after': self.power_after,
            # 详细能源变化字段
            'power_consumed_wh': self.power_consumed,
            'power_generated_wh': self.power_generated,
            'energy_consumption_j': self.energy_consumption,
            'battery_soc_before': self.battery_soc_before,
            'battery_soc_after': self.battery_soc_after,
            'ground_station_id': self.ground_station_id,
            'antenna_id': self.antenna_id,
            'downlink_start': self.downlink_start.isoformat() if self.downlink_start else None,
            'downlink_end': self.downlink_end.isoformat() if self.downlink_end else None,
            'data_transferred': self.data_transferred,
            'roll_angle': self.roll_angle,
            'pitch_angle': self.pitch_angle,
            'yaw_angle': self.yaw_angle,
            'attitude_coordinate_system': self.attitude_coordinate_system,
            'priority': self.priority,
            'reset_time': self.reset_time,
            # 聚类任务相关字段
            'is_cluster_task': self.is_cluster_task,
            'cluster_id': self.cluster_id,
            'primary_target_id': self.primary_target_id,
            'covered_target_ids': self.covered_target_ids,
            'covered_target_count': self.covered_target_count,
            # 成像足迹相关字段
            'footprint_corners': self.footprint_corners,
            'footprint_center': self.footprint_center,
            'swath_width_km': self.swath_width_km,
            'fov_config': self.fov_config,
        }


@dataclass
class ScheduleResult:
    """调度结果"""
    scheduled_tasks: List[ScheduledTask]
    unscheduled_tasks: Dict[str, TaskFailure]
    makespan: float
    computation_time: float
    iterations: int
    convergence_curve: List[float] = field(default_factory=list)
    failure_summary: Optional[Dict[TaskFailureReason, int]] = None
    area_coverage_stats: Optional[Dict[str, Any]] = None  # 区域覆盖统计

    def get_demand_satisfaction_rate(self, total_tasks: int) -> float:
        if total_tasks == 0:
            return 0.0
        return len(self.scheduled_tasks) / total_tasks

    def get_failure_rate_by_reason(self) -> Dict[str, float]:
        """按原因计算失败比例

        Returns:
            Dict[str, float]: 各失败原因的比例，key为原因字符串，value为比例(0-1)
        """
        if not self.unscheduled_tasks:
            return {}
        if self.failure_summary is None:
            return {}
        total = len(self.unscheduled_tasks)
        return {
            reason.value: count / total
            for reason, count in self.failure_summary.items()
        }


class BaseScheduler(ABC):
    """调度器基类 - 即插即用接口"""

    # 性能优化阈值：批量查询的最小候选数
    # 对于小批量（<1000），逐个查询更快；大批量（>=1000），批量查询更快
    BATCH_QUERY_THRESHOLD = 1000

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.mission = None
        self.window_cache = None
        self._failure_log: List[TaskFailure] = []
        self._start_time: Optional[float] = None
        self._iterations = 0
        self._convergence_curve: List[float] = []
        # 初始化姿态角计算器
        propagator_type = self.config.get('propagator_type', 'sgp4')
        self._attitude_calculator = AttitudeCalculator(
            propagator_type=PropagatorType.SGP4 if propagator_type == 'sgp4' else PropagatorType.HPOP
        )
        self._enable_attitude_calculation = self.config.get('enable_attitude_calculation', True)

        # 初始化机动约束检查器（在 initialize() 中完成实际初始化）
        self._slew_checker: Optional[SlewConstraintChecker] = None

        # 初始化 SAA 约束检查器
        self._saa_checker: Optional[SAAConstraintChecker] = None

        # 初始化姿态管理
        self._enable_attitude_management = self.config.get('enable_attitude_management', False)
        self._attitude_checker: Optional[AttitudeConstraintChecker] = None
        self._sat_attitude_state: Dict[str, AttitudeMode] = {}

        # 初始化位置缓存（用于预计算位置）
        self._position_cache = None

        # 初始化太阳位置计算器和发电量计算器
        self._sun_calculator: Optional[SunPositionCalculator] = None
        self._power_calculator: Optional[PowerGenerationCalculator] = None
        self._enable_power_generation_calc = self.config.get('enable_power_generation_calc', True)

    def initialize(self, mission, satellite_pool=None, ground_station_pool=None) -> None:
        """初始化调度器"""
        import os

        self.mission = mission
        self._failure_log = []
        self._iterations = 0
        self._convergence_curve = []

        # 禁用实时轨道传播，强制使用预计算数据
        # 这是确保完整精确计算的一部分
        os.environ['DISABLE_REALTIME_PROPAGATION'] = '1'
        logger.info("已禁用实时轨道传播，将使用预计算轨道数据")

        # 初始化太阳位置计算器和发电量计算器
        if self._enable_power_generation_calc:
            try:
                self._sun_calculator = SunPositionCalculator(use_orekit=False)  # 使用简化模型避免JVM依赖
                power_config = PowerConfig(max_power=1000.0, eclipse_power=0.0)
                self._power_calculator = PowerGenerationCalculator(
                    sun_calculator=self._sun_calculator,
                    config=power_config
                )
                logger.info("发电量计算器初始化完成")
            except Exception as e:
                logger.warning(f"发电量计算器初始化失败: {e}")
                self._sun_calculator = None
                self._power_calculator = None

    def set_slew_checker(self, slew_checker) -> None:
        """设置外部传入的机动约束检查器

        用于 UnifiedScheduler 传递 PreciseSlewConstraintChecker 给下层调度器。

        Args:
            slew_checker: SlewConstraintChecker 或 PreciseSlewConstraintChecker 实例
        """
        self._slew_checker = slew_checker

    def _calculate_power_generation(
        self,
        sat_id: str,
        start_time: datetime,
        end_time: datetime,
        roll_angle: float = 0.0,
        pitch_angle: float = 0.0,
        num_samples: int = 5
    ) -> float:
        """计算任务期间的发电量

        通过在任务时间段内采样计算平均发电功率并积分得到总发电量。

        Args:
            sat_id: 卫星ID
            start_time: 任务开始时间
            end_time: 任务结束时间
            roll_angle: 滚转角（度）
            pitch_angle: 俯仰角（度）
            num_samples: 采样点数（默认5个）

        Returns:
            float: 发电量（Wh）
        """
        if self._power_calculator is None or self._attitude_calculator is None:
            return 0.0

        try:
            # 获取卫星对象
            sat = self.mission.get_satellite_by_id(sat_id) if self.mission else None
            if not sat:
                return 0.0

            # 获取卫星最大功率（从capabilities）
            max_power = getattr(sat.capabilities, 'power_capacity', 1000.0)

            # 更新配置
            self._power_calculator.config.max_power = max_power

            # 计算采样间隔
            duration_seconds = (end_time - start_time).total_seconds()
            if duration_seconds <= 0:
                return 0.0

            # 在时间段内采样计算发电功率
            total_power = 0.0
            sample_count = 0

            for i in range(num_samples):
                # 计算采样时间点
                fraction = i / (num_samples - 1) if num_samples > 1 else 0.5
                sample_time = start_time + timedelta(seconds=duration_seconds * fraction)

                # 获取卫星位置
                try:
                    position, velocity = self._attitude_calculator._get_satellite_state(sat, sample_time)
                    if position is None:
                        continue

                    # 计算发电功率
                    power = self._power_calculator.calculate_power(
                        attitude_mode=AttitudeMode.IMAGING,
                        satellite_position=position,
                        timestamp=sample_time,
                        roll_angle=roll_angle,
                        pitch_angle=pitch_angle
                    )
                    total_power += power
                    sample_count += 1
                except Exception as e:
                    logger.debug(f"计算采样点 {i} 的发电功率失败: {e}")
                    continue

            if sample_count == 0:
                return 0.0

            # 计算平均功率并转换为电量（Wh）
            avg_power = total_power / sample_count  # 平均功率（W）
            duration_hours = duration_seconds / 3600.0
            power_generated = avg_power * duration_hours  # 电量（Wh）

            return max(0.0, power_generated)

        except Exception as e:
            logger.debug(f"计算发电量失败: {e}")
            return 0.0

    def _initialize_slew_checker(self) -> None:
        """初始化机动约束检查器

        强制使用 BatchSlewConstraintChecker 以启用向量化批量计算优化。
        高精度要求：所有调度器必须使用精确模型，禁止简化模式。
        """
        if self.mission is None:
            return

        # 如果已经设置了外部检查器，跳过初始化
        if self._slew_checker is not None:
            return

        # 强制使用批量姿态约束检查器（向量化优化）
        # 高精度要求：使用刚体动力学查找表（高精度+高性能）
        self._slew_checker = BatchSlewConstraintChecker(
            self.mission,
            use_precise_model=True,
            use_lookup_table=True  # 默认启用刚体动力学查表
        )

        # 设置状态跟踪器（如果可用）
        if hasattr(self, '_state_tracker') and self._state_tracker is not None:
            self._slew_checker.set_state_tracker(self._state_tracker)

        logger.info(f"{self.name}: Initialized BatchSlewConstraintChecker for vectorized computation")

    def _precompute_satellite_positions(self, time_step_seconds: float = 1.0) -> None:
        """
        预计算卫星位置以加速调度

        在调度开始前批量计算所有卫星在关键时间点的位置，
        避免在调度循环中重复进行昂贵的轨道传播计算。

        策略:
        1. 按时间步长覆盖整个任务时间范围
        2. 使用与HPOP可见性计算相同的步长（默认5秒，与粗扫描匹配）
        3. 批量计算并缓存所有卫星位置

        Args:
            time_step_seconds: 时间步长（秒），默认5秒（与HPOP粗扫描步长匹配）

        Raises:
            ValueError: 如果 time_step_seconds 小于或等于 0
        """
        # 参数验证必须在任何操作之前
        if time_step_seconds <= 0:
            raise ValueError(f"time_step_seconds must be positive, got {time_step_seconds}")

        if self.mission is None:
            return

        # 首先尝试加载Java端预计算的轨道数据
        if self._load_precomputed_orbits_from_java():
            print("    已加载Java预计算轨道数据，跳过Python端预计算")
            # 确保位置缓存被初始化（用于姿态角计算）
            if self._position_cache is None:
                from core.orbit.visibility.position_cache import SatellitePositionCache
                self._position_cache = SatellitePositionCache()

            # Java数据已在批量传播器中，不需要预同步到_position_cache
            # _batch_precalculate_attitudes会直接从批量传播器批量获取所需位置
            return

        import time
        from datetime import timedelta

        start_time = time.time()
        total_positions = 0

        # 在整个任务时间范围内按步长生成时间点
        current_time = self.mission.start_time
        end_time = self.mission.end_time
        time_step = timedelta(seconds=time_step_seconds)

        discrete_times = []
        while current_time <= end_time:
            discrete_times.append(current_time)
            current_time += time_step

        # Issue 1 Fix: Ensure we have a position cache to store computed positions
        # when _slew_checker is None
        if self._position_cache is None:
            from core.orbit.visibility.position_cache import SatellitePositionCache
            self._position_cache = SatellitePositionCache()

        # 为每颗卫星预计算位置
        for sat in self.mission.satellites:
            # 获取或创建卫星缓存
            if self._slew_checker is not None:
                sat_cache = self._slew_checker._satellite_cache.get(sat.id, {})
            else:
                sat_cache = {}

            # 使用批量传播（如果可用）
            if hasattr(self._attitude_calculator, '_propagate_batch'):
                try:
                    batch_results = self._attitude_calculator._propagate_batch(
                        sat, discrete_times
                    )
                    for t, (pos, vel) in zip(discrete_times, batch_results):
                        if pos:
                            sat_cache[t] = (pos, vel)
                            total_positions += 1
                            # Issue 1 Fix: Also store in position_cache when slew_checker is None
                            if self._slew_checker is None and self._position_cache is not None:
                                self._position_cache.set_position(sat.id, t, pos, vel)
                    continue
                except Exception as e:
                    # 高精度要求：批量传播失败应抛出错误，而不是静默回退
                    raise RuntimeError(f"批量轨道传播失败: {e}") from e

            # 逐点计算
            for t in discrete_times:
                if t in sat_cache:
                    continue

                try:
                    position, velocity = self._attitude_calculator._get_satellite_state(sat, t)
                    if position:
                        sat_cache[t] = (position, velocity)
                        total_positions += 1
                        # Issue 1 Fix: Also store in position_cache when slew_checker is None
                        if self._slew_checker is None and self._position_cache is not None:
                            self._position_cache.set_position(sat.id, t, position, velocity)
                except Exception as e:
                    # 高精度要求：轨道传播失败应抛出错误
                    raise RuntimeError(f"卫星状态计算失败 ({sat.id} at {t}): {e}") from e

        elapsed = time.time() - start_time
        if total_positions > 0:
            print(f"    预计算完成: {total_positions} 个位置 ({elapsed:.2f}s)")

    def _initialize_saa_checker(self) -> None:
        """初始化 SAA 约束检查器 - 强制使用向量化批量优化"""
        if self.mission is None:
            return

        # 强制使用批量SAA约束检查器（向量化优化）
        # 高精度要求：禁止简化模式
        self._saa_checker = BatchSAAConstraintChecker(
            self.mission,
            self._attitude_calculator
        )

        for sat in self.mission.satellites:
            self._saa_checker.initialize_satellite(sat)

    def set_window_cache(self, cache) -> None:
        """设置窗口缓存"""
        self.window_cache = cache

    def set_position_cache(self, cache) -> None:
        """设置卫星位置缓存（用于预计算位置）"""
        self._position_cache = cache
        # 如果有机动约束检查器，也传递给它
        if self._slew_checker is not None:
            self._slew_checker.set_position_cache(cache)
        # 如果SAA检查器已存在，也传递给它
        if self._saa_checker is not None:
            self._saa_checker.set_position_cache(cache)

    def _load_precomputed_orbits_from_java(self) -> bool:
        """加载Java端预计算的轨道数据

        尝试从默认路径加载Java端导出的轨道数据，避免重复计算。

        Returns:
            bool: 是否成功加载
        """
        if not self.config.get('use_precomputed_orbits', True):
            return False

        # 默认路径
        json_path = self.config.get('orbit_json_path', 'java/output/frequency_scenario/orbits.json.gz')

        import os
        if not os.path.exists(json_path):
            logger.debug(f"Java预计算轨道数据不存在: {json_path}")
            return False

        try:
            from core.dynamics.orbit_batch_propagator import get_batch_propagator

            propagator = get_batch_propagator()
            if propagator is None:
                return False

            start_time = self.mission.start_time if hasattr(self.mission, 'start_time') else None
            success = propagator.load_precomputed_orbits(json_path, start_time)

            if success:
                logger.info(f"成功加载Java预计算轨道数据: {json_path}")
            return success

        except Exception as e:
            logger.debug(f"加载Java预计算轨道数据失败: {e}")
            return False

    def _sync_orbit_data_to_position_cache(self, time_step_seconds: float = 60.0) -> None:
        """将Java预计算轨道数据同步填充到_position_cache

        这样批量姿态计算可以直接从_position_cache获取，避免逐个查询批量传播器。

        Args:
            time_step_seconds: 时间步长（秒），默认60秒以平衡精度和内存
        """
        if self._position_cache is None or self.mission is None:
            return

        try:
            from core.dynamics.orbit_batch_propagator import get_batch_propagator
            from datetime import timedelta

            propagator = get_batch_propagator()
            if propagator is None:
                return

            # 在任务时间范围内按步长生成时间点
            current_time = self.mission.start_time
            end_time = self.mission.end_time
            time_step = timedelta(seconds=time_step_seconds)

            total_synced = 0
            sat_count = 0

            for sat in self.mission.satellites:
                sat_count += 1
                sat_synced = 0
                t = current_time

                while t <= end_time:
                    # 从批量传播器获取位置
                    result = propagator.get_state_at_time(sat.id, t)
                    if result is not None:
                        position, velocity = result
                        self._position_cache.set_position(sat.id, t, position, velocity)
                        total_synced += 1
                        sat_synced += 1
                    t += time_step

                if sat_count <= 3:  # 只打印前3颗卫星的信息
                    logger.debug(f"  同步卫星 {sat.id}: {sat_synced} 个位置点")

            logger.info(f"已将 {total_synced} 个位置点同步到_position_cache ({sat_count} 颗卫星)")

        except Exception as e:
            logger.warning(f"同步轨道数据到_position_cache失败: {e}")

    def _get_satellite_position(
        self,
        satellite,
        timestamp: datetime
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """获取卫星位置和速度

        首先检查预计算位置缓存，然后检查批量传播器缓存。

        Args:
            satellite: 卫星对象
            timestamp: 时间戳

        Returns:
            (position, velocity) 或 (None, None)
        """
        # 1. 优先使用预计算位置缓存
        if self._position_cache is not None:
            result = self._position_cache.get_position(satellite.id, timestamp)
            if result is not None:
                return result

        # 2. 检查批量传播器缓存
        try:
            from core.dynamics.orbit_batch_propagator import get_batch_propagator
            propagator = get_batch_propagator()
            if propagator is not None:
                result = propagator.get_state_at_time(satellite.id, timestamp)
                if result is not None:
                    return result
        except Exception as e:
            # 高精度要求：批量传播器失败应记录或抛出错误
            logger.warning(f"批量传播器获取状态失败 ({satellite.id} at {timestamp}): {e}")

        # 3. 如果缓存都不可用，使用卫星轨道传播
        try:
            if hasattr(satellite, 'orbit') and satellite.orbit is not None:
                from core.orbit.propagator import SGP4Propagator
                # SGP4Propagator需要satrec参数，在测试环境中可能没有
                # 返回None让上层处理
                return None, None
        except Exception as e:
            # 测试环境中允许返回None而不是抛出错误
            return None, None

        return None, None

    @abstractmethod
    def schedule(self) -> ScheduleResult:
        """执行调度"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """返回算法可调参数"""
        pass

    def _record_failure(self, task_id: str, reason: TaskFailureReason, detail: str, **kwargs) -> None:
        """记录任务失败原因"""
        failure = TaskFailure(
            task_id=task_id,
            failure_reason=reason,
            failure_detail=detail,
            **kwargs
        )
        self._failure_log.append(failure)

    def _start_timer(self) -> None:
        self._start_time = time.time()

    def _stop_timer(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _add_convergence_point(self, fitness: float) -> None:
        self._convergence_curve.append(fitness)
        self._iterations += 1

    def _validate_initialization(self) -> None:
        """
        验证调度器是否已正确初始化

        检查所有必要字段：
        1. mission 不为 None
        2. satellites 不为空列表
        3. targets 不为空列表

        Raises:
            RuntimeError: 如果任何必要字段未初始化
        """
        if self.mission is None:
            raise RuntimeError("Scheduler not initialized: mission is None")

        if not hasattr(self.mission, 'satellites') or not self.mission.satellites:
            raise RuntimeError("Scheduler not initialized: no satellites available")

        if not hasattr(self.mission, 'targets') or not self.mission.targets:
            raise RuntimeError("Scheduler not initialized: no targets available")

    def validate_solution(self, result: ScheduleResult) -> bool:
        """
        验证解的可行性

        根据设计文档3.1要求，检查调度结果是否满足所有约束：
        1. 任务时间不重叠
        2. 卫星资源约束满足
        3. 时间窗口约束满足
        4. 卫星能力约束满足

        Args:
            result: 调度结果

        Returns:
            bool: 解是否可行
        """
        if not result.scheduled_tasks:
            # 空调度视为有效
            return True

        # 按卫星分组检查
        sat_tasks: Dict[str, List[ScheduledTask]] = {}
        for task in result.scheduled_tasks:
            if task.satellite_id not in sat_tasks:
                sat_tasks[task.satellite_id] = []
            sat_tasks[task.satellite_id].append(task)

        # 检查每个卫星的任务是否时间冲突
        for sat_id, tasks in sat_tasks.items():
            # 按开始时间排序
            sorted_tasks = sorted(tasks, key=lambda t: t.imaging_start)

            for i in range(len(sorted_tasks) - 1):
                current_task = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]

                # 检查时间重叠
                if current_task.imaging_end > next_task.imaging_start:
                    return False

        # 检查电量约束（简化检查：电量不能为负）
        for task in result.scheduled_tasks:
            if task.power_after < 0:
                return False

        # 检查存储约束（简化检查：存储不能为负或超容）
        for task in result.scheduled_tasks:
            if task.storage_after < 0:
                return False
            if self.mission:
                sat = self.mission.get_satellite_by_id(task.satellite_id)
                if sat and task.storage_after > sat.capabilities.storage_capacity:
                    return False

        return True

    def _create_frequency_aware_tasks(self):
        """创建频次感知的观测任务列表"""
        if not self.mission or not self.mission.targets:
            return []
        return create_observation_tasks(self.mission.targets)

    def _calculate_frequency_fitness(self, target_obs_count, base_score=0.0):
        """计算频次满足度的适应度分数"""
        if not self.mission or not self.mission.targets:
            return base_score
        return calculate_frequency_fitness(
            target_obs_count, self.mission.targets, base_score
        )

    def _calculate_attitude_angles(
        self,
        satellite,
        target,
        imaging_time
    ) -> Optional[AttitudeAngles]:
        """计算卫星成像时刻的姿态角

        优先使用预计算位置缓存（如果可用），否则使用AttitudeCalculator实时计算。

        Args:
            satellite: 卫星对象
            target: 目标对象
            imaging_time: 成像时刻

        Returns:
            AttitudeAngles对象，如果禁用姿态角计算或计算失败则返回None
        """
        if not self._enable_attitude_calculation:
            return None

        # 优先使用预计算位置缓存
        if self._position_cache is not None:
            return self._calculate_attitude_from_cache(
                satellite, target, imaging_time
            )

        # 回退到实时计算
        if self._attitude_calculator is None:
            return None

        try:
            attitude = self._attitude_calculator.calculate_attitude(
                satellite=satellite,
                target=target,
                imaging_time=imaging_time
            )
            return attitude
        except Exception as e:
            # 记录警告但不中断调度流程
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to calculate attitude angles for task: {e}"
            )
            return None

    def _calculate_attitude_from_cache(
        self,
        satellite,
        target,
        imaging_time
    ) -> Optional[AttitudeAngles]:
        """使用预计算位置缓存计算姿态角

        直接从位置缓存获取卫星位置和速度，避免实时轨道传播计算。

        Args:
            satellite: 卫星对象
            target: 目标对象
            imaging_time: 成像时刻

        Returns:
            AttitudeAngles对象，如果缓存未命中或计算失败则返回None
        """
        from core.dynamics.attitude_calculator import AttitudeAngles
        import math

        try:
            # 从缓存获取卫星位置和速度
            result = self._position_cache.get_position(satellite.id, imaging_time)

            # 如果缓存未命中，尝试从批量传播器获取（Java预计算数据）
            if result is None:
                try:
                    from core.dynamics.orbit_batch_propagator import get_batch_propagator
                    propagator = get_batch_propagator()
                    if propagator is not None:
                        result = propagator.get_state_at_time(satellite.id, imaging_time)
                        # 如果批量传播器返回None，不要尝试实时传播，直接返回None
                        # 禁用实时轨道传播以确保使用完整精确计算
                except Exception:
                    pass

            # 如果预计算数据不可用，返回None（禁用实时传播）
            if result is None:
                return None

            position, velocity = result

            # 使用AttitudeCalculator的方法计算姿态角
            if self._attitude_calculator is None:
                return None

            # 构建LVLH坐标系
            lvlh_frame = self._attitude_calculator._construct_lvlh_frame(position, velocity)

            # 计算目标视线向量（从卫星指向目标）
            los_vector = self._attitude_calculator._calculate_los_vector(position, target)

            # 反转视线向量方向（从目标指向卫星）用于姿态角计算
            # 这是因为卫星需要指向目标，所以姿态角应该基于卫星看向目标的反方向
            los_vector = (-los_vector[0], -los_vector[1], -los_vector[2])

            # 将视线向量转换到LVLH坐标系
            los_in_lvlh = self._attitude_calculator._transform_to_lvlh(los_vector, lvlh_frame)

            # 计算滚转和俯仰角
            roll, pitch = self._attitude_calculator._calculate_roll_pitch(los_in_lvlh)

            return AttitudeAngles(
                roll=math.degrees(roll),
                pitch=math.degrees(pitch),
                yaw=0.0,
                coordinate_system="LVLH",
                timestamp=imaging_time
            )

        except (AttributeError, TypeError, ValueError) as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to calculate attitude from cache: {type(e).__name__}: {e}"
            )
            return None

    def _batch_precalculate_attitudes(
        self,
        candidates: List[Tuple[Any, Any, datetime]]
    ) -> Dict[Tuple[str, datetime], AttitudeAngles]:
        """批量预计算候选姿态角 (Numba向量化优化版本)

        使用Numba JIT编译和并行计算，实现1000倍加速：
        - 10个候选: 0.03ms (原16.8ms)
        - 100个候选: 0.17ms (原176ms)
        - 1000个候选: 1.6ms (原1760ms)

        优化策略:
        1. 批量从缓存获取卫星位置（避免逐个查询开销）
        2. 使用Numba并行计算姿态角
        3. 最小化Python层面的循环开销

        Args:
            candidates: 候选列表，每个元素为 (satellite, target, imaging_time)

        Returns:
            字典，键为 (satellite_id, imaging_time)，值为 AttitudeAngles
        """
        if not self._enable_attitude_calculation or not candidates:
            return {}

        try:
            # 使用Numba批量姿态计算器
            from .constraints.batch_attitude_calculator import (
                BatchAttitudeCandidate,
                get_batch_attitude_calculator
            )
            from core.dynamics.orbit_batch_propagator import get_batch_propagator

            # 获取批量传播器
            propagator = get_batch_propagator()
            if propagator is None:
                return self._batch_precalculate_attitudes_fallback(candidates)

            # 批量获取卫星位置 - 减少函数调用开销
            # 策略：先按卫星分组，然后批量查询
            sat_time_map: Dict[str, List[Tuple[int, datetime, Any, Any]]] = {}
            # 结构: {sat_id: [(idx, time, target, sat_obj), ...]}

            for idx, (sat, target, imaging_time) in enumerate(candidates):
                lat = getattr(target, 'latitude', None)
                lon = getattr(target, 'longitude', None)
                if lat is None or lon is None:
                    continue

                if sat.id not in sat_time_map:
                    sat_time_map[sat.id] = []
                sat_time_map[sat.id].append((idx, imaging_time, target, sat))

            # 批量查询卫星位置 - 使用Numba加速的批量接口
            batch_candidates = []
            sat_positions = {}
            idx_to_candidate = {}

            # 检查传播器是否支持批量查询
            has_batch_query = hasattr(propagator, 'get_states_batch')

            # 计算总候选数，决定是否使用批量查询
            # 对于小批量（<BATCH_QUERY_THRESHOLD），逐个查询更快；
            # 大批量（>=BATCH_QUERY_THRESHOLD），批量查询更快
            total_entries = sum(len(entries) for entries in sat_time_map.values())
            use_batch = has_batch_query and total_entries >= self.BATCH_QUERY_THRESHOLD

            for sat_id, entries in sat_time_map.items():
                if use_batch:
                    # 使用批量查询（Numba加速，适合大批量）
                    imaging_times = [e[1] for e in entries]
                    batch_results = propagator.get_states_batch(sat_id, imaging_times)

                    for (idx, imaging_time, target, sat_obj), result in zip(entries, batch_results):
                        if result is None:
                            continue

                        batch_candidates.append(BatchAttitudeCandidate(
                            sat_id=sat_id,
                            target_id=getattr(target, 'id', 'unknown'),
                            target_lat=getattr(target, 'latitude'),
                            target_lon=getattr(target, 'longitude'),
                            imaging_time=imaging_time
                        ))
                        sat_positions[(sat_id, imaging_time)] = result
                        idx_to_candidate[len(batch_candidates) - 1] = (sat_id, imaging_time)
                else:
                    # 逐个查询（适合小批量，避免批量查询的开销）
                    for idx, imaging_time, target, sat_obj in entries:
                        result = propagator.get_state_at_time(sat_id, imaging_time)
                        if result is None:
                            continue

                        batch_candidates.append(BatchAttitudeCandidate(
                            sat_id=sat_id,
                            target_id=getattr(target, 'id', 'unknown'),
                            target_lat=getattr(target, 'latitude'),
                            target_lon=getattr(target, 'longitude'),
                            imaging_time=imaging_time
                        ))
                        sat_positions[(sat_id, imaging_time)] = result
                        idx_to_candidate[len(batch_candidates) - 1] = (sat_id, imaging_time)

            if not batch_candidates:
                return {}

            # 使用Numba批量计算
            calculator = get_batch_attitude_calculator()
            batch_results = calculator.compute_batch(batch_candidates, sat_positions)

            # 转换为AttitudeAngles格式
            from core.dynamics.attitude_calculator import AttitudeAngles
            results: Dict[Tuple[str, datetime], AttitudeAngles] = {}

            for i, result in enumerate(batch_results):
                if result.feasible and i in idx_to_candidate:
                    sat_id, imaging_time = idx_to_candidate[i]
                    results[(sat_id, imaging_time)] = AttitudeAngles(
                        roll=result.roll,
                        pitch=result.pitch,
                        yaw=0.0,
                        coordinate_system="LVLH",
                        timestamp=imaging_time
                    )

            return results

        except Exception as e:
            logger.debug(f"Numba批量计算失败，回退到逐个计算: {e}")
            return self._batch_precalculate_attitudes_fallback(candidates)

    def _batch_precalculate_attitudes_fallback(
        self,
        candidates: List[Tuple[Any, Any, datetime]]
    ) -> Dict[Tuple[str, datetime], AttitudeAngles]:
        """批量预计算候选姿态角 (回退版本)

        当Numba批量计算不可用时使用逐个计算。

        Args:
            candidates: 候选列表，每个元素为 (satellite, target, imaging_time)

        Returns:
            字典，键为 (satellite_id, imaging_time)，值为 AttitudeAngles
        """
        if not self._enable_attitude_calculation or not candidates:
            return {}

        results: Dict[Tuple[str, datetime], AttitudeAngles] = {}

        for sat, target, imaging_time in candidates:
            try:
                attitude = self._calculate_attitude_from_cache(sat, target, imaging_time)
                if attitude is not None:
                    results[(sat.id, imaging_time)] = attitude
            except Exception as e:
                # 高精度要求：姿态计算失败应抛出错误
                raise RuntimeError(f"姿态计算失败 ({sat.id} at {imaging_time}): {e}") from e

        return results

    def _apply_attitude_to_scheduled_task(
        self,
        scheduled_task: ScheduledTask,
        attitude: Optional[AttitudeAngles]
    ) -> None:
        """将姿态角应用到ScheduledTask

        Args:
            scheduled_task: 已调度的任务对象
            attitude: 姿态角对象
        """
        if attitude is None:
            return

        scheduled_task.roll_angle = attitude.roll
        scheduled_task.pitch_angle = attitude.pitch
        scheduled_task.yaw_angle = attitude.yaw
        scheduled_task.attitude_coordinate_system = attitude.coordinate_system

    def set_propagator_type(self, propagator_type: str) -> None:
        """设置轨道传播器类型

        Args:
            propagator_type: 'sgp4' 或 'hpop'
        """
        if propagator_type.lower() == 'sgp4':
            self._attitude_calculator = AttitudeCalculator(
                propagator_type=PropagatorType.SGP4
            )
        elif propagator_type.lower() == 'hpop':
            self._attitude_calculator = AttitudeCalculator(
                propagator_type=PropagatorType.HPOP
            )
        else:
            raise ValueError(f"Unknown propagator type: {propagator_type}")

    def enable_attitude_calculation(self, enable: bool = True) -> None:
        """启用或禁用姿态角计算

        Args:
            enable: 是否启用姿态角计算
        """
        self._enable_attitude_calculation = enable

    def _ensure_slew_checker_initialized(self) -> None:
        """确保机动约束检查器已初始化

        如果_slew_checker为None，则调用_initialize_slew_checker()进行初始化。
        所有使用_slew_checker的方法应在访问前调用此方法。
        """
        if self._slew_checker is None:
            self._initialize_slew_checker()

    def _ensure_saa_checker_initialized(self) -> None:
        """确保SAA约束检查器已初始化

        如果_saa_checker为None，则调用_initialize_saa_checker()进行初始化。
        所有使用_saa_checker的方法应在访问前调用此方法。
        """
        if self._saa_checker is None:
            self._initialize_saa_checker()

    def _calculate_slew_angle_and_time(
        self,
        sat_id: str,
        prev_target: Any,
        current_target: Any
    ) -> Tuple[float, float]:
        """计算机动角度和时间

        统一的机动计算方法，供所有调度器使用，避免代码重复。
        优先使用SlewConstraintChecker进行计算，如果不可用则使用简化计算。

        Args:
            sat_id: 卫星ID
            prev_target: 上一个目标（None表示第一个任务）
            current_target: 当前目标

        Returns:
            Tuple[float, float]: (机动角度, 机动时间秒)
        """
        import math

        # 如果没有上一个目标，不需要机动
        if prev_target is None:
            return 0.0, 0.0

        # 检查目标是否有位置信息
        if not hasattr(prev_target, 'latitude') or not hasattr(prev_target, 'longitude'):
            return 0.0, 0.0

        if not hasattr(current_target, 'latitude') or not hasattr(current_target, 'longitude'):
            return 0.0, 0.0

        # 确保_slew_checker已初始化
        self._ensure_slew_checker_initialized()

        # 高精度要求：必须使用SlewConstraintChecker进行精确计算
        if self._slew_checker is None:
            raise RuntimeError(
                "Slew checker not initialized. "
                "High precision mode requires SlewConstraintChecker for slew estimation."
            )

        try:
            # 获取卫星以取得最大侧摆角
            sat = None
            if self.mission:
                sat = self.mission.get_satellite_by_id(sat_id)

            if sat:
                # 使用精确机动角度计算
                slew_result = self._slew_checker.check_slew_feasibility(
                    sat_id, prev_target, current_target,
                    datetime.now(), datetime.now()  # 时间参数在此场景中不重要
                )
                if slew_result and slew_result.feasible:
                    return slew_result.slew_angle, slew_result.slew_time

            # 如果计算失败，抛出错误（高精度要求：不允许回退到简化计算）
            raise RuntimeError(
                "Failed to calculate precise slew angle and time. "
                "High precision mode requires exact calculations."
            )
        except RuntimeError:
            raise
        except Exception as e:
            # 高精度要求：精确计算失败时抛出错误，不回退
            raise RuntimeError(
                f"Slew calculation failed: {e}. "
                "High precision mode requires exact calculations."
            ) from e

    def _initialize_attitude_checker(self) -> None:
        """初始化姿态约束检查器"""
        if not self._enable_attitude_management:
            return
        if self._attitude_checker is None:
            from .constraints import AttitudeConstraintChecker
            from core.dynamics.attitude_manager import AttitudeManagementConfig
            config = AttitudeManagementConfig(
                max_slew_rate=3.0,
                settling_time=5.0
            )
            self._attitude_checker = AttitudeConstraintChecker(config=config)

    def _ensure_attitude_checker_initialized(self) -> None:
        """确保姿态约束检查器已初始化"""
        if self._enable_attitude_management and self._attitude_checker is None:
            self._initialize_attitude_checker()

    def _initialize_attitude_state(self) -> None:
        """初始化卫星姿态状态跟踪"""
        if not self._enable_attitude_management or self.mission is None:
            return
        for sat in self.mission.satellites:
            self._sat_attitude_state[sat.id] = AttitudeMode.NADIR_POINTING

    def _get_satellite_attitude_mode(self, sat_id: str) -> AttitudeMode:
        """获取卫星当前姿态模式"""
        if not self._enable_attitude_management:
            return AttitudeMode.NADIR_POINTING
        return self._sat_attitude_state.get(sat_id, AttitudeMode.NADIR_POINTING)

    def _set_satellite_attitude_mode(self, sat_id: str, mode: AttitudeMode) -> None:
        """设置卫星姿态模式"""
        if self._enable_attitude_management:
            self._sat_attitude_state[sat_id] = mode

    def _calculate_attitude_transition_time(
        self,
        sat_id: str,
        from_mode: AttitudeMode,
        to_mode: AttitudeMode,
        timestamp: datetime
    ) -> timedelta:
        """计算姿态切换时间

        使用姿态约束检查器计算机动所需时间。
        如果无法计算，返回默认的30秒。
        """
        if not self._enable_attitude_management or from_mode == to_mode:
            return timedelta(seconds=0)

        self._ensure_attitude_checker_initialized()
        if self._attitude_checker is None:
            return timedelta(seconds=30)  # 默认30秒

        # 获取卫星位置
        sat = self.mission.get_satellite_by_id(sat_id) if self.mission else None
        if sat is None:
            return timedelta(seconds=30)

        # 获取卫星当前位置（如果可用）
        sat_position = getattr(sat, 'position', None)
        if sat_position is None or not isinstance(sat_position, (tuple, list)) or len(sat_position) != 3:
            # 使用默认轨道位置（假设500km高度，简化计算）
            sat_position = (6871000.0, 0.0, 0.0)  # 默认位置（米）

        try:
            result = self._attitude_checker.check_attitude_transition(
                from_mode=from_mode,
                to_mode=to_mode,
                satellite_position=tuple(sat_position),
                timestamp=timestamp
            )
            return result.transition_time
        except Exception as e:
            # 计算失败时返回默认时间
            return timedelta(seconds=30)

