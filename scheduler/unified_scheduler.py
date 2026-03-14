"""
统一调度器 - 整合完整约束、频次需求和地面站数传

功能：
1. 完整约束检查（机动、SAA、电量、存储）
2. 观测频次需求处理
3. 地面站数传规划

Usage:
    from scheduler.unified_scheduler import UnifiedScheduler
    scheduler = UnifiedScheduler(mission, window_cache, ground_station_pool)
    result = scheduler.schedule()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
import os

from core.models import Mission, Satellite, Target
from core.orbit.visibility.window_cache import VisibilityWindowCache
from core.resources.ground_station_pool import GroundStationPool
from scheduler.base_scheduler import (
    BaseScheduler, ScheduleResult, ScheduledTask,
    TaskFailure, TaskFailureReason
)
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler
from scheduler.metaheuristic.aco_scheduler import ACOScheduler
from scheduler.metaheuristic.pso_scheduler import PSOScheduler
from scheduler.metaheuristic.tabu_scheduler import TabuScheduler
from scheduler.ground_station.scheduler import (
    GroundStationScheduler, GroundStationScheduleResult
)
from scheduler.frequency_utils import (
    ObservationTask, create_observation_tasks, calculate_frequency_fitness
)
from evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class UnifiedScheduleResult:
    """统一调度结果"""
    # 成像调度结果
    imaging_result: ScheduleResult = field(default_factory=lambda: ScheduleResult(
        scheduled_tasks=[], unscheduled_tasks={}, makespan=0.0,
        computation_time=0.0, iterations=0, convergence_curve=[]
    ))
    # 数传调度结果
    downlink_result: Optional[GroundStationScheduleResult] = None
    # 频次满足度统计
    target_observations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 总体统计
    total_computation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'imaging': {
                'scheduled_count': len(self.imaging_result.scheduled_tasks),
                'unscheduled_count': len(self.imaging_result.unscheduled_tasks),
                'makespan_hours': self.imaging_result.makespan / 3600,
                'computation_time': self.imaging_result.computation_time,
            },
            'downlink': {
                'scheduled_count': len(self.downlink_result.downlink_tasks) if self.downlink_result else 0,
                'failed_count': len(self.downlink_result.failed_tasks) if self.downlink_result else 0,
            } if self.downlink_result else None,
            'target_observations': self.target_observations,
            'total_computation_time': self.total_computation_time,
        }


class UnifiedScheduler:
    """
    统一调度器

    整合成像调度、频次需求和地面站数传规划的一站式解决方案。
    """

    # 支持的成像调度算法
    IMAGING_ALGORITHMS = {
        'greedy': GreedyScheduler,
        'ga': GAScheduler,
        'edd': EDDScheduler,
        'spt': SPTScheduler,
        'sa': SAScheduler,
        'aco': ACOScheduler,
        'pso': PSOScheduler,
        'tabu': TabuScheduler,
    }

    def __init__(
        self,
        mission: Mission,
        window_cache: VisibilityWindowCache,
        ground_station_pool: Optional[GroundStationPool] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化统一调度器

        Args:
            mission: 任务场景对象
            window_cache: 可见性窗口缓存
            ground_station_pool: 地面站资源池（可选，如果需要进行数传规划）
            config: 配置参数
                - imaging_algorithm: 成像调度算法 ('greedy', 'ga', 'edd')
                - imaging_config: 成像调度器配置
                - enable_downlink: 是否启用数传规划
                - downlink_config: 数传调度器配置
                - consider_frequency: 是否考虑频次需求
        """
        self.mission = mission
        self.window_cache = window_cache
        self.ground_station_pool = ground_station_pool
        self.config = config or {}

        # 成像调度配置
        self.imaging_algorithm = self.config.get('imaging_algorithm', 'greedy')
        self.imaging_config = self.config.get('imaging_config', {})

        # 数传配置（默认启用，如果提供了地面站资源池）
        if 'enable_downlink' in self.config:
            self.enable_downlink = self.config['enable_downlink']
        else:
            self.enable_downlink = ground_station_pool is not None
        # 全局默认配置（会被每颗卫星的独立配置覆盖）
        self.downlink_config = self.config.get('downlink_config', {})

        # 频次需求配置
        self.consider_frequency = self.config.get('consider_frequency', True)

        # 内部状态
        self._imaging_scheduler: Optional[BaseScheduler] = None
        self._ground_station_scheduler: Optional[GroundStationScheduler] = None
        self._target_obs_count: Dict[str, int] = {}

        # 初始化精确姿态机动约束检查器
        self._initialize_constraint_checkers()

        # 加载预计算轨道数据（如果配置启用）
        self._load_precomputed_orbits()

    def _initialize_constraint_checkers(self) -> None:
        """初始化精确姿态机动约束检查器

        使用 BatchSlewConstraintChecker 以启用向量化批量计算优化。
        """
        from .constraints import BatchSlewConstraintChecker

        self.slew_checker = BatchSlewConstraintChecker(
            mission=self.mission,
            use_precise_model=True,
            use_lookup_table=True  # 默认启用刚体动力学查表
        )

        # 设置状态跟踪器（如果可用）
        if hasattr(self, '_state_tracker') and self._state_tracker is not None:
            self.slew_checker.set_state_tracker(self._state_tracker)

        logger.info("使用批量姿态机动模型（向量化优化）")

    def _load_precomputed_orbits(self) -> None:
        """加载预计算的轨道数据"""
        # 检查是否启用
        if not self.config.get('use_precomputed_orbits', True):
            logger.info("跳过加载预计算轨道数据（已禁用）")
            return

        # 获取JSON文件路径
        json_path = self.config.get('orbit_json_path')
        if json_path is None:
            # 使用默认路径
            json_path = 'java/output/frequency_scenario/orbits.json.gz'
            logger.debug(f"未指定orbit_json_path，使用默认路径: {json_path}")

        if not os.path.exists(json_path):
            logger.warning(f"预计算轨道数据文件不存在: {json_path}")
            return

        try:
            from core.dynamics.orbit_batch_propagator import get_batch_propagator

            propagator = get_batch_propagator()
            if propagator is None:
                logger.warning("无法获取批量传播器，跳过加载预计算轨道数据")
                return

            # 获取场景开始时间
            start_time = self.mission.start_time if hasattr(self.mission, 'start_time') else None

            # 加载预计算数据
            success = propagator.load_precomputed_orbits(json_path, start_time)
            if success:
                logger.info(f"成功加载预计算轨道数据: {json_path}")
            else:
                logger.warning(f"加载预计算轨道数据失败: {json_path}")

        except Exception as e:
            logger.error(f"加载预计算轨道数据时出错: {e}")
            # 不抛出异常，允许回退到HPOP计算

    def schedule(self) -> UnifiedScheduleResult:
        """
        执行统一调度

        流程：
        1. 执行成像调度（考虑完整约束和频次需求）
        2. 执行地面站数传规划
        3. 计算综合指标

        Returns:
            UnifiedScheduleResult: 统一调度结果
        """
        start_time = time.time()
        result = UnifiedScheduleResult()

        logger.info("=" * 70)
        logger.info("开始统一调度")
        logger.info("=" * 70)

        # 步骤1: 成像调度
        logger.info("\n[1/3] 成像任务调度...")
        imaging_result = self._schedule_imaging()
        result.imaging_result = imaging_result
        logger.info(f"  成功调度: {len(imaging_result.scheduled_tasks)} 个任务")
        logger.info(f"  未调度: {len(imaging_result.unscheduled_tasks)} 个任务")

        # 步骤2: 地面站数传规划（如果启用）
        if self.enable_downlink and self.ground_station_pool and imaging_result.scheduled_tasks:
            logger.info("\n[2/3] 地面站数传规划...")
            downlink_result = self._schedule_downlinks(imaging_result.scheduled_tasks)
            result.downlink_result = downlink_result
            logger.info(f"  成功规划: {len(downlink_result.downlink_tasks)} 个数传任务")
            logger.info(f"  失败: {len(downlink_result.failed_tasks)} 个")
        else:
            logger.info("\n[2/3] 跳过数传规划")

        # 步骤3: 计算频次满足度
        logger.info("\n[3/3] 计算频次满足度...")
        result.target_observations = self._calculate_target_observations(
            imaging_result.scheduled_tasks
        )
        satisfaction_rate = self._calculate_satisfaction_rate(result.target_observations)
        logger.info(f"  需求满足率: {satisfaction_rate:.2%}")

        result.total_computation_time = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("统一调度完成")
        logger.info(f"总耗时: {result.total_computation_time:.2f} 秒")
        logger.info("=" * 70)

        return result

    def _schedule_imaging(self) -> ScheduleResult:
        """
        执行成像任务调度

        根据配置的算法创建相应的调度器，并执行调度。
        """
        # 获取调度器类
        scheduler_class = self.IMAGING_ALGORITHMS.get(self.imaging_algorithm)
        if scheduler_class is None:
            raise ValueError(f"不支持的成像调度算法: {self.imaging_algorithm}")

        logger.info(f"  使用算法: {self.imaging_algorithm.upper()}")

        # 创建调度器实例
        scheduler = scheduler_class(self.imaging_config)

        # 初始化
        scheduler.initialize(self.mission)
        scheduler.set_window_cache(self.window_cache)

        # 传递精确机动约束检查器（如果已初始化）
        if hasattr(self, 'slew_checker') and self.slew_checker is not None:
            scheduler.set_slew_checker(self.slew_checker)
            logger.info("  使用精确姿态机动模型")

        # 执行调度
        return scheduler.schedule()

    def _schedule_downlinks(
        self,
        scheduled_tasks: List[ScheduledTask]
    ) -> GroundStationScheduleResult:
        """
        执行地面站数传规划

        从每颗卫星的配置中读取数传速率和固存容量。

        Args:
            scheduled_tasks: 已调度的成像任务列表

        Returns:
            GroundStationScheduleResult: 数传调度结果
        """
        if self.ground_station_pool is None:
            raise ValueError("未提供地面站资源池，无法进行数传规划")

        # 创建数传调度器（使用全局默认值，但会根据每颗卫星配置覆盖）
        gs_scheduler = GroundStationScheduler(
            ground_station_pool=self.ground_station_pool,
            data_rate_mbps=self.downlink_config.get('data_rate_mbps', 300.0),
            storage_capacity_gb=self.downlink_config.get('storage_capacity_gb', 100.0),
            overflow_threshold=self.downlink_config.get('overflow_threshold', 0.95),
            link_setup_time_seconds=self.downlink_config.get('link_setup_time_seconds', 60.0)
        )

        # 根据每颗卫星的独立配置初始化固存状态
        for sat in self.mission.satellites:
            # 从卫星能力配置中读取固存容量和数据率
            storage_capacity = getattr(sat.capabilities, 'storage_capacity', 500.0)
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)

            gs_scheduler.initialize_satellite_storage(
                satellite_id=sat.id,
                current_gb=0.0,
                capacity_gb=storage_capacity
            )
            # 存储该卫星的独立数据率配置（供后续使用）
            gs_scheduler._satellite_data_rates = getattr(
                gs_scheduler, '_satellite_data_rates', {}
            )
            gs_scheduler._satellite_data_rates[sat.id] = data_rate

            logger.debug(f"卫星 {sat.id}: 固存={storage_capacity}GB, 数传速率={data_rate}Mbps")

        # 获取卫星-地面站可见窗口
        gs_windows = self._get_ground_station_windows()

        # 执行数传调度
        return gs_scheduler.schedule_downlinks_for_tasks(
            scheduled_tasks=scheduled_tasks,
            visibility_windows=gs_windows
        )

    def _get_ground_station_windows(self) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """
        获取卫星-地面站可见窗口

        Returns:
            Dict[satellite_id, List[(start, end)]]: 每个卫星的地面站可见窗口列表
        """
        windows: Dict[str, List[Tuple[datetime, datetime]]] = {}

        if self.window_cache is None:
            return windows

        for sat in self.mission.satellites:
            sat_windows = []
            for gs in self.mission.ground_stations:
                # 从缓存获取卫星-地面站窗口
                key = (sat.id, f"GS:{gs.id}")
                if hasattr(self.window_cache, '_windows') and key in self.window_cache._windows:
                    for window in self.window_cache._windows[key]:
                        sat_windows.append((window.start_time, window.end_time))

            if sat_windows:
                windows[sat.id] = sorted(sat_windows, key=lambda x: x[0])

        return windows

    def _calculate_target_observations(
        self,
        scheduled_tasks: List[ScheduledTask]
    ) -> Dict[str, Dict[str, Any]]:
        """
        计算每个目标的观测完成情况

        Args:
            scheduled_tasks: 已调度的任务列表

        Returns:
            Dict[target_id, {required, actual, satisfied}]
        """
        # 统计实际观测次数
        # 注意: task.target_id 可能是频次感知格式（如 TGT-0001-OBS1），需要提取基础目标ID
        actual_obs: Dict[str, int] = {}
        for task in scheduled_tasks:
            # 提取基础目标ID（去掉 -OBSx 后缀）
            full_target_id = task.target_id
            base_target_id = full_target_id.split('-OBS')[0] if '-OBS' in full_target_id else full_target_id
            actual_obs[base_target_id] = actual_obs.get(base_target_id, 0) + 1

        # 构建结果
        result = {}
        for target in self.mission.targets:
            required = getattr(target, 'required_observations', 1)
            actual = actual_obs.get(target.id, 0)

            if required == -1:
                # 不限频次
                satisfied = actual > 0
                status = f"{actual}次（不限）"
            else:
                satisfied = actual >= required
                status = f"{actual}/{required}次"

            result[target.id] = {
                'target_name': target.name,
                'required': required,
                'actual': actual,
                'satisfied': satisfied,
                'status': status,
                'satisfaction_rate': actual / required if required > 0 else (1.0 if actual > 0 else 0.0)
            }

        return result

    def _calculate_satisfaction_rate(
        self,
        target_observations: Dict[str, Dict[str, Any]]
    ) -> float:
        """计算总体需求满足率"""
        if not target_observations:
            return 0.0

        satisfied_count = sum(
            1 for info in target_observations.values() if info['satisfied']
        )
        return satisfied_count / len(target_observations)

    def get_metrics(self, result: UnifiedScheduleResult) -> Dict[str, Any]:
        """
        计算综合性能指标

        Args:
            result: 统一调度结果

        Returns:
            Dict: 性能指标字典
        """
        metrics_calc = MetricsCalculator(self.mission)
        imaging_metrics = metrics_calc.calculate_all(result.imaging_result)

        metrics = {
            # 成像指标
            'imaging_scheduled': len(result.imaging_result.scheduled_tasks),
            'imaging_unscheduled': len(result.imaging_result.unscheduled_tasks),
            'demand_satisfaction_rate': imaging_metrics.demand_satisfaction_rate,
            'makespan_hours': imaging_metrics.makespan / 3600,
            'satellite_utilization': imaging_metrics.satellite_utilization,

            # 频次指标
            'target_satisfaction': result.target_observations,
            'overall_satisfaction_rate': self._calculate_satisfaction_rate(
                result.target_observations
            ),

            # 数传指标
            'downlink_scheduled': (
                len(result.downlink_result.downlink_tasks)
                if result.downlink_result else 0
            ),
            'downlink_failed': (
                len(result.downlink_result.failed_tasks)
                if result.downlink_result else 0
            ),

            # 效率指标
            'imaging_time': result.imaging_result.computation_time,
            'total_time': result.total_computation_time,
        }

        return metrics
