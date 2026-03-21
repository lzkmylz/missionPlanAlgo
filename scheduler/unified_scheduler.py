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
from scheduler.ground_station.downlink_task import DownlinkTask
from scheduler.relay.scheduler import (
    RelayScheduler, RelayScheduleResult
)
from scheduler.relay.downlink_task import RelayDownlinkTask
from core.network.relay_satellite import RelayNetwork
from scheduler.isl import ISLDownlinkScheduler, ISLDownlinkTask
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
        relay_network: Optional[RelayNetwork] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化统一调度器

        Args:
            mission: 任务场景对象
            window_cache: 可见性窗口缓存
            ground_station_pool: 地面站资源池（可选，如果需要进行数传规划）
            relay_network: 中继卫星网络（可选）
            config: 配置参数
                - imaging_algorithm: 成像调度算法 ('greedy', 'ga', 'edd')
                - imaging_config: 成像调度器配置
                - enable_downlink: 是否启用数传规划
                - downlink_strategy: 数传策略 ('ground_station_first', 'relay_first', 'best_effort')
                - downlink_config: 数传调度器配置
                - consider_frequency: 是否考虑频次需求
        """
        self.mission = mission
        self.window_cache = window_cache
        self.ground_station_pool = ground_station_pool
        self.relay_network = relay_network
        self.config = config or {}

        # 成像调度配置
        self.imaging_algorithm = self.config.get('imaging_algorithm', 'greedy')
        self.imaging_config = self.config.get('imaging_config', {})

        # 调度策略配置: coverage_first (覆盖优先) 或 interleaved (交织调度)
        # 优先级: 传入配置 > Mission配置 > imaging_config > 默认值
        self.scheduling_strategy = self.config.get(
            'scheduling_strategy',
            getattr(mission, 'scheduling_config', {}).get(
                'scheduling_strategy',
                self.imaging_config.get('scheduling_strategy', 'coverage_first')
            )
        )
        # 将策略传递给成像调度器配置
        self.imaging_config['scheduling_strategy'] = self.scheduling_strategy
        self.imaging_config['downlink_lookahead_seconds'] = self.config.get(
            'downlink_lookahead_seconds', 3600)
        self.imaging_config['max_concurrent_roll_deg'] = self.config.get(
            'max_concurrent_roll_deg', 30.0)

        # 数传配置（默认启用，如果提供了地面站资源池或中继网络）
        if 'enable_downlink' in self.config:
            self.enable_downlink = self.config['enable_downlink']
        else:
            self.enable_downlink = (ground_station_pool is not None) or (relay_network is not None)

        # 数传策略: 'ground_station_first', 'relay_first', 'best_effort'
        self.downlink_strategy = self.config.get('downlink_strategy', 'best_effort')

        # 全局默认配置（会被每颗卫星的独立配置覆盖）
        self.downlink_config = self.config.get('downlink_config', {})

        # 频次需求配置
        self.consider_frequency = self.config.get('consider_frequency', True)

        # ISL router and scheduler (optional, initialised lazily)
        self._isl_router = None            # TimeVaryingISLRouter | None
        self._isl_downlink_scheduler: Optional[ISLDownlinkScheduler] = None

        # 内部状态
        self._imaging_scheduler: Optional[BaseScheduler] = None
        self._ground_station_scheduler: Optional[GroundStationScheduler] = None
        self._relay_scheduler: Optional[RelayScheduler] = None
        self._target_obs_count: Dict[str, int] = {}

        # 初始化精确姿态机动约束检查器
        self._initialize_constraint_checkers()

        # 加载预计算轨道数据（如果配置启用）
        self._load_precomputed_orbits()

        # ISL网络构建（如果配置启用且场景中有ISL卫星）
        if self.config.get('enable_isl', False):
            self._isl_router = self._build_isl_network_from_config()
            if self._isl_router is not None:
                logger.info("ISL多跳路由已启用")

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

        # 步骤2: 数传规划（如果启用）
        if self.enable_downlink and imaging_result.scheduled_tasks:
            if self.downlink_strategy == 'best_effort':
                # 混合策略：优先地面站，失败则尝试中继
                logger.info("\n[2/3] 混合数传规划（优先地面站）...")
                downlink_result = self._schedule_hybrid_downlinks(imaging_result.scheduled_tasks)
                result.downlink_result = downlink_result
            elif self.downlink_strategy == 'ground_station_first':
                # 仅地面站
                if self.ground_station_pool:
                    logger.info("\n[2/3] 地面站数传规划...")
                    downlink_result = self._schedule_downlinks(imaging_result.scheduled_tasks)
                    result.downlink_result = downlink_result
                else:
                    logger.warning("\n[2/3] 未提供地面站资源池，跳过数传规划")
            elif self.downlink_strategy == 'relay_first':
                # 仅中继
                if self.relay_network:
                    logger.info("\n[2/3] 中继卫星数传规划...")
                    downlink_result = self._schedule_relay_downlinks(imaging_result.scheduled_tasks)
                    result.downlink_result = downlink_result
                else:
                    logger.warning("\n[2/3] 未提供中继网络，跳过数传规划")
            else:
                logger.warning(f"\n[2/3] 未知的数传策略: {self.downlink_strategy}")
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
        if result.downlink_result:
            logger.info(f"数传成功: {len(result.downlink_result.downlink_tasks)} 个任务")
            logger.info(f"数传失败: {len(result.downlink_result.failed_tasks)} 个任务")
        logger.info("=" * 70)

        return result

    def _build_isl_network_from_config(self) -> Optional[Any]:
        """构建 ISL 网络路由器，使用 UnifiedScheduler 持有的窗口缓存。

        在 UnifiedScheduler 中，窗口缓存（``self.window_cache``）包含所有
        卫星-目标、卫星-地面站以及卫星-ISL 对等星的可见性窗口。本方法从中
        提取 ISL 相关窗口并构建 ``TimeVaryingISLRouter``。

        Returns:
            ``TimeVaryingISLRouter`` 实例，若不可用则返回 ``None``。
        """
        try:
            from core.network.isl_router import ISLRouterWindowCache, TimeVaryingISLRouter
            from scheduler.base_scheduler import _DuckISLLink

            satellite_isl_configs: Dict[str, Any] = {}
            for sat in self.mission.satellites:
                isl_cfg = getattr(getattr(sat, 'capabilities', None), 'isl', None)
                if isl_cfg is None:
                    isl_cfg = getattr(sat, 'isl_config', None)
                if isl_cfg is not None and getattr(isl_cfg, 'enabled', False):
                    satellite_isl_configs[sat.id] = isl_cfg

            if not satellite_isl_configs:
                logger.debug("没有卫星启用了ISL，跳过ISL网络构建")
                return None

            isl_cache = ISLRouterWindowCache()
            gs_windows: Dict[Tuple[str, str], List[Any]] = {}

            if self.window_cache is not None and hasattr(self.window_cache, '_windows'):
                for key, windows in self.window_cache._windows.items():
                    sat_id, target_id = key
                    if not isinstance(target_id, str):
                        continue

                    if target_id.startswith('ISL:'):
                        peer_sat_id = target_id[4:]
                        for win in windows:
                            lnk = _DuckISLLink(
                                satellite_a_id=sat_id,
                                satellite_b_id=peer_sat_id,
                                start_time=win.start_time,
                                end_time=win.end_time,
                                link_type=getattr(win, 'link_type', 'laser'),
                                max_data_rate=getattr(win, 'max_data_rate', 10000.0),
                                atp_setup_time_s=getattr(win, 'atp_setup_time_s', 37.0),
                                link_quality=getattr(win, 'link_quality', 1.0),
                                is_viable=getattr(win, 'is_viable', True),
                            )
                            isl_cache.add_window(lnk)
                    elif target_id.startswith('GS:'):
                        gs_id = target_id[3:]
                        gs_windows[(sat_id, gs_id)] = list(windows)

            if not gs_windows:
                logger.debug("ISL网络构建：无GS可见窗口，无法路由到地面站")
                return None

            router = TimeVaryingISLRouter(
                isl_window_cache=isl_cache,
                satellite_isl_configs=satellite_isl_configs,
                gs_visibility_windows=gs_windows,
            )
            logger.info(
                "ISL网络构建完成：%d 颗卫星启用ISL，%d 条GS可见窗口对",
                len(satellite_isl_configs),
                len(gs_windows),
            )
            return router

        except Exception as exc:
            logger.warning("ISL网络构建失败: %s", exc, exc_info=True)
            return None

    def _schedule_relay_downlinks(
        self,
        scheduled_tasks: List[ScheduledTask]
    ) -> RelayScheduleResult:
        """
        执行中继卫星数传规划

        Args:
            scheduled_tasks: 已调度的成像任务列表

        Returns:
            RelayScheduleResult: 中继数传调度结果
        """
        if self.relay_network is None:
            raise ValueError("未提供中继网络，无法进行中继数传规划")

        # 创建中继数传调度器
        relay_scheduler = RelayScheduler(
            relay_network=self.relay_network,
            default_data_rate_mbps=self.downlink_config.get('relay_data_rate_mbps', 450.0),
            link_setup_time_seconds=self.downlink_config.get('relay_link_setup_time_seconds', 10.0)
        )

        # 初始化固存状态
        for sat in self.mission.satellites:
            storage_capacity = getattr(sat.capabilities, 'storage_capacity', 500.0)
            relay_scheduler.initialize_satellite_storage(
                satellite_id=sat.id,
                current_gb=0.0,
                capacity_gb=storage_capacity
            )

        # 获取卫星-中继可见窗口
        relay_windows = self._get_relay_windows()

        # 执行数传调度
        return relay_scheduler.schedule_downlinks_for_tasks(
            scheduled_tasks=scheduled_tasks,
            visibility_windows=relay_windows
        )

    def _get_relay_windows(self) -> Dict[str, List[Tuple[str, datetime, datetime]]]:
        """
        获取卫星-中继卫星可见窗口。

        必须从Java后端预计算的缓存加载中继可见性窗口。
        如果缓存中不存在中继窗口，将抛出错误，要求重新计算可见性窗口。

        Returns:
            Dict[satellite_id, List[(relay_id, start, end)]]: 每个卫星的中继可见窗口列表，
            包含中继卫星ID，按开始时间排序

        Raises:
            RuntimeError: 如果缓存中没有找到中继卫星可见性窗口
        """
        windows: Dict[str, List[Tuple[str, datetime, datetime]]] = {}

        if self.relay_network is None:
            return windows

        # 从缓存加载中继窗口
        cache_hit = False
        missing_relays = set()

        if self.window_cache is not None:
            for sat in self.mission.satellites:
                sat_windows = []
                for relay_id in self.relay_network.relays:
                    key = (sat.id, f"RELAY:{relay_id}")
                    if hasattr(self.window_cache, '_windows') and key in self.window_cache._windows:
                        for window in self.window_cache._windows[key]:
                            sat_windows.append((relay_id, window.start_time, window.end_time))
                            # 同步注入 RelayNetwork 可见性（供 can_relay_data 使用）
                            self.relay_network.add_visibility_window(
                                sat.id, relay_id, window.start_time, window.end_time
                            )
                        cache_hit = True
                    else:
                        missing_relays.add(relay_id)

                if sat_windows:
                    windows[sat.id] = sorted(sat_windows, key=lambda x: x[1])

        # 检查是否找到任何中继窗口
        if not cache_hit:
            raise RuntimeError(
                f"缓存中未找到中继卫星可见性窗口。\n"
                f"请使用Java后端重新计算可见性窗口，确保包含中继卫星。\n"
                f"命令: cd java && java -cp \"classes:lib/*\" "
                f"orekit.visibility.LargeScaleFrequencyTest "
                f"--scenario {self.mission.name}.json "
                f"--output output/frequency_scenario"
            )

        # 记录缺失的中继窗口信息（部分中继无窗口是正常的物理现象）
        if missing_relays:
            logger.info(
                f"部分中继卫星在场景期间无可见窗口（正常）: {missing_relays}"
            )

        logger.info(f"从缓存加载了 {len(windows)} 颗卫星的中继可见性窗口")
        return windows

    def _schedule_hybrid_downlinks(
        self,
        scheduled_tasks: List[ScheduledTask]
    ) -> GroundStationScheduleResult:
        """
        执行混合数传规划（优先地面站，失败则尝试中继）

        Args:
            scheduled_tasks: 已调度的成像任务列表

        Returns:
            GroundStationScheduleResult: 数传调度结果（可能包含中继数传任务）
        """
        # 首先尝试地面站数传
        gs_tasks = []
        relay_tasks = []

        if self.ground_station_pool:
            gs_result = self._schedule_downlinks(scheduled_tasks)
            gs_scheduled_ids = {t.related_imaging_task_id for t in gs_result.downlink_tasks}
            gs_tasks = gs_result.downlink_tasks

            # 找出地面站数传失败的任务
            failed_tasks = [t for t in scheduled_tasks if t.task_id not in gs_scheduled_ids]
        else:
            failed_tasks = scheduled_tasks

        # 对于地面站失败的，尝试中继
        relay_failed_tasks = failed_tasks
        if failed_tasks and self.relay_network:
            relay_result = self._schedule_relay_downlinks(failed_tasks)
            relay_tasks = relay_result.downlink_tasks
            relay_succeeded_ids = {t.related_imaging_task_id for t in relay_tasks}
            relay_failed_tasks = [t for t in failed_tasks if t.task_id not in relay_succeeded_ids]

            logger.info(f"  地面站成功: {len(gs_tasks)} 个")
            logger.info(f"  中继成功: {len(relay_tasks)} 个")

        # 对于中继也失败的，尝试ISL多跳路由
        isl_tasks: List = []
        if relay_failed_tasks and self._isl_router is not None:
            isl_tasks_created, isl_failed = self._schedule_isl_downlinks(relay_failed_tasks)
            isl_tasks = isl_tasks_created
            logger.info(f"  ISL多跳路由成功: {len(isl_tasks)} 个")
            logger.info(f"  ISL路由失败: {len(isl_failed)} 个")

        # 合并结果
        combined_result = GroundStationScheduleResult()
        converted_relay = [self._convert_relay_to_gs_task(t) for t in relay_tasks]
        converted_isl = [self._convert_isl_to_gs_task(t) for t in isl_tasks]
        combined_result.downlink_tasks = gs_tasks + converted_relay + converted_isl
        combined_result.failed_tasks = [
            t.task_id for t in scheduled_tasks
            if t.task_id not in {dt.related_imaging_task_id for dt in combined_result.downlink_tasks}
        ]

        return combined_result

    def _convert_relay_to_gs_task(self, relay_task: RelayDownlinkTask) -> DownlinkTask:
        """将中继任务转换为地面站任务格式（用于结果合并）"""
        return DownlinkTask(
            task_id=relay_task.task_id,
            satellite_id=relay_task.satellite_id,
            ground_station_id=f"RELAY:{relay_task.relay_id}",
            start_time=relay_task.start_time,
            end_time=relay_task.end_time,
            data_size_gb=relay_task.data_size_gb,
            related_imaging_task_id=relay_task.related_imaging_task_id,
            effective_data_rate=relay_task.effective_data_rate,
            acquisition_time_seconds=relay_task.acquisition_time_seconds
        )

    def _convert_isl_to_gs_task(self, isl_task: ISLDownlinkTask) -> DownlinkTask:
        """将ISL多跳任务转换为地面站任务格式（用于结果合并）。

        路由路径编码在 ground_station_id 字段中，格式为
        ``ISL:<source>->...-><gs_id>``，便于后续统计和日志。
        """
        path_str = "->".join(isl_task.route_path_nodes) if isl_task.route_path_nodes else isl_task.destination_gs_id
        return DownlinkTask(
            task_id=isl_task.task_id,
            satellite_id=isl_task.source_satellite_id,
            ground_station_id=f"ISL:{path_str}",
            start_time=isl_task.start_time,
            end_time=isl_task.end_time,
            data_size_gb=isl_task.data_size_gb,
            related_imaging_task_id=isl_task.related_imaging_task_id,
            effective_data_rate=isl_task.effective_bandwidth_mbps,
            acquisition_time_seconds=isl_task.atp_setup_time_s,
        )

    def _schedule_isl_downlinks(
        self,
        imaging_tasks: List[ScheduledTask],
    ) -> Tuple[List[ISLDownlinkTask], List[str]]:
        """执行ISL多跳数传规划（第三层兜底策略）。

        本方法在地面站直传和GEO中继都无法满足的情况下，通过ISL星间链路网络为
        成像任务找到多跳数传路径。

        Args:
            imaging_tasks: 地面站直传和GEO中继均失败的成像任务列表。

        Returns:
            (isl_tasks_created, failed_task_ids):
                isl_tasks_created — 成功规划的 ISLDownlinkTask 列表。
                failed_task_ids — 无法通过ISL路由完成数传的任务ID列表。
        """
        if self._isl_router is None:
            logger.debug("ISL路由器未初始化，跳过ISL数传规划")
            return [], [t.task_id for t in imaging_tasks]

        # 懒加载 ISLDownlinkScheduler
        if self._isl_downlink_scheduler is None:
            isl_configs = self.config.get('satellite_isl_configs', {})
            self._isl_downlink_scheduler = ISLDownlinkScheduler(
                isl_router=self._isl_router,
                satellite_isl_configs=isl_configs,
                max_relay_hops=self.config.get('isl_max_relay_hops', 3),
                deadline_buffer_s=self.config.get('isl_deadline_buffer_s', 3600.0),
            )

        isl_tasks, failed_ids = self._isl_downlink_scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=imaging_tasks,
            satellite_states={},
            existing_tasks=[],
        )

        if isl_tasks:
            stats = self._isl_downlink_scheduler.get_statistics(isl_tasks)
            logger.info(
                "ISL数传统计: 成功=%d, 平均跳数=%.1f, 平均带宽=%.0f Mbps, "
                "总数据量=%.2f GB",
                stats['total_isl_tasks'],
                stats.get('avg_hop_count', 0),
                stats.get('avg_bandwidth_mbps', 0),
                stats.get('total_data_relayed_gb', 0),
            )

        return isl_tasks, failed_ids

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

        # 传递地面站/中继窗口用于交织调度（仅当使用interleaved策略时）
        if (hasattr(scheduler, 'set_downlink_windows') and
            self.scheduling_strategy == 'interleaved'):
            gs_windows = self._get_ground_station_windows()
            relay_windows = self._get_relay_windows()
            scheduler.set_downlink_windows(gs_windows, relay_windows)

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
            # 注册该卫星的独立数据率配置
            gs_scheduler.register_satellite_data_rate(sat.id, data_rate)

            logger.debug(f"卫星 {sat.id}: 固存={storage_capacity}GB, 数传速率={data_rate}Mbps")

        # 获取卫星-地面站可见窗口
        gs_windows = self._get_ground_station_windows()

        # 过滤掉已在交织调度阶段安排数传的任务
        pending_downlink_tasks = [t for t in scheduled_tasks if t.downlink_start is None]
        already_scheduled_count = len(scheduled_tasks) - len(pending_downlink_tasks)
        if already_scheduled_count > 0:
            logger.info(f"  交织调度已安排 {already_scheduled_count} 个任务的数传，Phase2 处理剩余 {len(pending_downlink_tasks)} 个")

        # 执行数传调度
        return gs_scheduler.schedule_downlinks_for_tasks(
            scheduled_tasks=pending_downlink_tasks,
            visibility_windows=gs_windows
        )

    def _get_ground_station_windows(self) -> Dict[str, List[Tuple[str, datetime, datetime]]]:
        """
        获取卫星-地面站可见窗口

        Returns:
            Dict[satellite_id, List[(gs_id, start, end)]]: 每个卫星的地面站可见窗口列表，
            包含地面站ID，按开始时间排序
        """
        windows: Dict[str, List[Tuple[str, datetime, datetime]]] = {}

        if self.window_cache is None:
            return windows

        for sat in self.mission.satellites:
            sat_windows = []
            for gs in self.mission.ground_stations:
                # 从缓存获取卫星-地面站窗口
                key = (sat.id, f"GS:{gs.id}")
                if hasattr(self.window_cache, '_windows') and key in self.window_cache._windows:
                    for window in self.window_cache._windows[key]:
                        sat_windows.append((gs.id, window.start_time, window.end_time))

            if sat_windows:
                windows[sat.id] = sorted(sat_windows, key=lambda x: x[1])

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
