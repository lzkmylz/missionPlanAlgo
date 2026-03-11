"""
高性能启发式调度器基类

整合 GreedyScheduler 的所有性能优化:
1. 姿态预计算缓存 (O(1)查询)
2. 批量约束检查 (向量化)
3. 候选数量限制 (剪枝)
4. 分层过滤策略 (快速淘汰)

子类只需实现:
- _sort_tasks() - 任务排序策略
- _select_best_assignment() - 候选选择策略 (可选)
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import time
import logging

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from ..frequency_utils import ObservationTask
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile
from core.dynamics.slew_calculator import SlewCalculator
from ..constraints import (
    SlewConstraintChecker,
    SlewFeasibilityResult,
    BatchSlewConstraintChecker,
    UnifiedBatchConstraintChecker,
    UnifiedBatchCandidate,
    UnifiedBatchResult
)
from scheduler.common.constraint_checker import ConstraintChecker, ConstraintContext
from scheduler.common.config import ConstraintConfig
from scheduler.common.clustering_mixin import ClusteringMixin, ClusterTask
from scheduler.constraints.batch_slew_calculator import BatchSlewCandidate

logger = logging.getLogger(__name__)


class HeuristicScheduler(BaseScheduler, ClusteringMixin):
    """
    高性能启发式调度器基类

    子类只需实现:
    1. _sort_tasks() - 任务排序策略
    2. _select_best_assignment() - 选择策略 (可选)
    """

    # ========== 性能优化配置 ==========
    MAX_PHASE4_CANDIDATES = 50  # 限制精确检查的候选数
    MIN_WINDOW_RATIO = 0.8      # 最小窗口比例
    ENABLE_ATTITUDE_PRECACHE = True
    ENABLE_BATCH_CONSTRAINT_CHECK = True

    def __init__(self, name: str, heuristic: str, config: Dict[str, Any] = None):
        """
        初始化高性能启发式调度器

        Args:
            name: 调度器名称
            heuristic: 启发式策略名称
            config: 配置参数
        """
        super().__init__(name, config)
        ClusteringMixin.__init__(self, config)
        self.heuristic = heuristic
        self.config = config or {}

        # 性能优化开关
        self._enable_attitude_precache = self.config.get(
            'enable_attitude_precache', self.ENABLE_ATTITUDE_PRECACHE
        )
        self._enable_batch_constraint_check = self.config.get(
            'enable_batch_constraint_check', self.ENABLE_BATCH_CONSTRAINT_CHECK
        )

        # 基础配置
        self.consider_power = self.config.get('consider_power', True)
        self.consider_storage = self.config.get('consider_storage', True)

        # 高性能约束检查器
        self._slew_checker: Optional[BatchSlewConstraintChecker] = None
        self._unified_checker: Optional[UnifiedBatchConstraintChecker] = None
        self._attitude_precache_manager = None
        self._constraint_checker: Optional[ConstraintChecker] = None

        # 初始化成像时间计算器和功率配置文件
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=self.config.get('min_imaging_duration'),
            max_duration=self.config.get('max_imaging_duration'),
            default_duration=self.config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(self.config.get('power_coefficients'))

        # Slew calculators per satellite
        self._slew_calculators: Dict[str, SlewCalculator] = {}
        self._last_task_target: Dict[str, Any] = {}
        self._sat_resource_usage: Dict[str, Dict[str, Any]] = {}

        # 性能统计
        self._perf_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'max_time': 0.0})

    def get_parameters(self) -> Dict[str, Any]:
        """返回算法可调参数"""
        return {
            'heuristic': self.heuristic,
            'consider_power': self.consider_power,
            'consider_storage': self.consider_storage,
        }

    def _get_scheduler_specific_params(self) -> Dict[str, Any]:
        """子类可覆盖以添加特定参数"""
        return {}

    def schedule(self) -> ScheduleResult:
        """
        执行启发式调度

        Returns:
            ScheduleResult: 调度结果
        """
        self._start_timer()
        self._validate_initialization()

        # 初始化高性能检查器
        self._initialize_high_performance_checkers()

        # 获取任务列表并排序
        if self.enable_clustering:
            pending_tasks = self._sort_tasks(self._get_clustered_tasks())
        else:
            pending_tasks = self._sort_tasks(self._create_frequency_aware_tasks())

        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, Any] = {}
        target_obs_count: Dict[str, int] = {}

        # 初始化卫星资源使用
        self._sat_resource_usage = {
            sat.id: {
                'power': sat.current_power if hasattr(sat, 'current_power') else sat.capabilities.power_capacity,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': [],
            }
            for sat in self.mission.satellites
        }

        # 初始化 slew calculators
        for sat in self.mission.satellites:
            agility = sat.capabilities.agility or {}
            self._slew_calculators[sat.id] = SlewCalculator(
                max_slew_rate=agility.get('max_slew_rate', 3.0),
                max_slew_angle=sat.capabilities.max_off_nadir,
                settling_time=agility.get('settling_time', 5.0)
            )

        # 主调度循环
        total_tasks = len(pending_tasks)
        logger.info(f"开始调度主循环，共 {total_tasks} 个任务")

        for task_idx, task in enumerate(pending_tasks):
            # 每100个任务报告进度
            if (task_idx + 1) % 100 == 0:
                elapsed = time.time() - self._start_time if self._start_time else 0
                logger.info(f"调度进度: {task_idx + 1}/{total_tasks} 任务, 已调度 {len(scheduled_tasks)}, 耗时 {elapsed:.1f}s")

            best_assignment = self._find_best_assignment(task)

            if best_assignment:
                sat_id, window, imaging_mode, slew_result = best_assignment

                scheduled_task = self._create_scheduled_task(
                    task, sat_id, window, imaging_mode, slew_result
                )
                scheduled_tasks.append(scheduled_task)

                self._update_resource_usage(sat_id, task, window, scheduled_task)
                self._last_task_target[sat_id] = task

                task_id = task.task_id if isinstance(task, ObservationTask) else task.id
                target_id = task.target_id if isinstance(task, ObservationTask) else task.id
                target_obs_count[target_id] = target_obs_count.get(target_id, 0) + 1

                self._add_convergence_point(len(scheduled_tasks))
            else:
                reason = self._determine_failure_reason(task)
                task_id = task.task_id if isinstance(task, ObservationTask) else task.id
                self._record_failure(
                    task_id=task_id,
                    reason=reason,
                    detail=f"No feasible assignment found for task {task_id}"
                )
                unscheduled[task_id] = self._failure_log[-1]

        makespan = 0.0
        if scheduled_tasks:
            last_end = max(t.imaging_end for t in scheduled_tasks)
            makespan = (last_end - self.mission.start_time).total_seconds()

        computation_time = self._stop_timer()

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self._iterations,
            convergence_curve=self._convergence_curve
        )

    def _initialize_high_performance_checkers(self) -> None:
        """初始化高性能约束检查器"""
        # 1. 批量姿态约束检查器
        if self._enable_batch_constraint_check and not self._use_simplified_slew:
            self._slew_checker = BatchSlewConstraintChecker(
                self.mission,
                use_precise_model=True
            )
            logger.info(f"{self.name}: 使用 BatchSlewConstraintChecker (向量化优化)")

        # 2. 统一批量约束检查器
        if self._enable_batch_constraint_check:
            self._unified_checker = UnifiedBatchConstraintChecker(
                mission=self.mission,
                use_precise_model=not self._use_simplified_slew,
                consider_power=self.consider_power,
                consider_storage=self.consider_storage
            )

        # 3. 姿态预计算缓存
        if self._enable_attitude_precache and not self._use_simplified_slew:
            self._initialize_attitude_precache()

        # 4. 初始化基类检查器（兼容性）
        constraint_config = ConstraintConfig(
            consider_power=self.consider_power,
            consider_storage=self.consider_storage
        )
        self._constraint_checker = ConstraintChecker(self.mission, constraint_config)
        self._constraint_checker.initialize()
        if self._position_cache is not None:
            self._constraint_checker.set_position_cache(self._position_cache)

        self._initialize_slew_checker()
        self._initialize_saa_checker()

    def _initialize_attitude_precache(self) -> None:
        """初始化姿态预计算缓存"""
        logger.info(f"{self.name}: 初始化姿态预计算缓存...")
        t_start = time.perf_counter()

        try:
            from core.dynamics.attitude_precache import get_attitude_precache_manager

            precache_manager = get_attitude_precache_manager()
            orbit_file = self.config.get(
                'orbit_json_path',
                'java/output/frequency_scenario/orbits.json.gz'
            )

            loaded = precache_manager.load_orbit_data(orbit_file, self.mission.start_time)

            if loaded:
                # 加载可见窗口
                windows = self._load_visibility_windows()

                # 预计算姿态
                n_computed = precache_manager.precompute_attitudes_for_windows(windows)
                stats = precache_manager.get_stats()

                logger.info(f"预计算了 {n_computed} 个姿态角，缓存: {stats['memory_mb']:.1f}MB")
                self._attitude_precache_manager = precache_manager
            else:
                logger.warning("姿态预计算缓存加载失败")
                self._attitude_precache_manager = None

        except Exception as e:
            logger.warning(f"姿态预计算缓存初始化失败: {e}")
            self._attitude_precache_manager = None

        elapsed = time.perf_counter() - t_start
        self._perf_stats['attitude_precache_init']['total_time'] = elapsed

    def _load_visibility_windows(self) -> List[Dict]:
        """加载可见性窗口数据"""
        windows = []

        try:
            import json
            cache_file = self.config.get('cache_path') or 'java/output/frequency_scenario/visibility_windows_with_gs.json'

            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            windows_dict = cache_data.get('windows', {})
            target_windows = windows_dict.get('target_windows', [])

            # 构建目标ID到对象的映射
            target_map = {getattr(t, 'id', None): t for t in self.mission.targets}

            for window in target_windows:
                if isinstance(window, dict):
                    sat_id = window.get('satelliteId')
                    target_id = window.get('targetId')

                    # 跳过地面站窗口
                    if target_id and str(target_id).startswith('GS:'):
                        continue

                    target_obj = target_map.get(target_id)

                    if target_obj and sat_id:
                        start_str = window.get('startTime', '').replace('Z', '+00:00')
                        end_str = window.get('endTime', '').replace('Z', '+00:00')
                        try:
                            start_time = datetime.fromisoformat(start_str)
                            end_time = datetime.fromisoformat(end_str)

                            windows.append({
                                'satellite_id': sat_id,
                                'target_id': target_id,
                                'target': target_obj,
                                'start': start_time,
                                'end': end_time
                            })
                        except:
                            pass

        except Exception as e:
            logger.warning(f"从缓存文件加载窗口失败: {e}")

        return windows

    def _find_best_assignment(self, task: Any) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """
        为任务找到最佳卫星-窗口组合 - 高性能版本

        四阶段分层过滤:
        1. 候选收集 (快速资源检查)
        2. 姿态预筛选 (O(1)缓存查询)
        3. 候选数量限制 (剪枝)
        4. 批量精确约束检查 (向量化)
        """
        # ========== Phase 1: 候选收集 ==========
        candidates = self._collect_candidates(task)
        if not candidates:
            return None

        # ========== Phase 2: 姿态预筛选 (O(1)缓存) ==========
        filtered_candidates = self._filter_by_attitude(candidates)

        # ========== Phase 3: 候选数量限制 ==========
        limited_candidates = self._limit_candidates(filtered_candidates)

        # ========== Phase 4: 批量精确约束检查 ==========
        best_assignment = self._batch_constraint_check(limited_candidates, task)

        return best_assignment

    def _collect_candidates(self, task: Any) -> List[Tuple]:
        """Phase 1: 收集候选卫星-窗口组合"""
        candidates = []

        for sat in self.mission.satellites:
            if not self._can_satellite_perform_task(sat, task):
                continue

            windows = self._get_feasible_windows(sat, task)

            for window in windows:
                window_start, window_end = self._extract_window_times(window)
                if not window_start or not window_end:
                    continue

                # 计算成像时长
                imaging_mode = self._select_imaging_mode(sat, task)
                imaging_duration = self._imaging_calculator.calculate(task, imaging_mode, sat)

                # 快速检查窗口是否足够长
                window_duration = (window_end - window_start).total_seconds()
                if window_duration < imaging_duration * self.MIN_WINDOW_RATIO:
                    continue

                # 快速资源检查
                if not self._check_resource_constraints(sat, task, imaging_mode):
                    continue

                candidates.append((sat, window, imaging_mode, imaging_duration, window_start, window_end))

        return candidates

    def _filter_by_attitude(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 2: 使用O(1)姿态缓存快速过滤"""
        if not self._attitude_precache_manager:
            return candidates

        filtered = []
        for sat, window, imaging_mode, imaging_duration, window_start, window_end in candidates:
            attitude = self._attitude_precache_manager.get_attitude(sat.id, window_start)

            if attitude:
                roll, pitch = attitude
                total_angle = (roll ** 2 + pitch ** 2) ** 0.5
                if total_angle > sat.capabilities.max_off_nadir * 1.2:
                    continue

            filtered.append((sat, window, imaging_mode, imaging_duration, window_start, window_end))

        return filtered

    def _limit_candidates(self, candidates: List[Tuple]) -> List[Tuple]:
        """Phase 3: 限制进入精确检查的候选数量"""
        if len(candidates) <= self.MAX_PHASE4_CANDIDATES:
            return candidates

        return sorted(candidates, key=lambda x: x[4])[:self.MAX_PHASE4_CANDIDATES]

    def _batch_constraint_check(
        self,
        candidates: List[Tuple],
        task: Any
    ) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """Phase 4: 批量约束检查（向量化优化）"""
        if not candidates:
            return None

        # 使用批量姿态约束检查
        use_batch = (
            isinstance(self._slew_checker, BatchSlewConstraintChecker) and
            len(candidates) > 1
        )

        if use_batch:
            # 批量姿态约束检查
            batch_candidates = []
            for sat, window, imaging_mode, imaging_duration, window_start, window_end in candidates:
                usage = self._sat_resource_usage.get(sat.id, {})
                last_task_end = usage.get('last_task_end', self.mission.start_time)
                sat_position, sat_velocity = self._get_satellite_position(sat, last_task_end)

                candidate = BatchSlewCandidate(
                    sat_id=sat.id,
                    satellite=sat,
                    target=task,
                    window_start=window_start,
                    window_end=window_end,
                    prev_end_time=last_task_end,
                    prev_target=self._get_previous_task_target(sat.id),
                    imaging_duration=imaging_duration,
                    sat_position=sat_position,
                    sat_velocity=sat_velocity
                )
                batch_candidates.append(candidate)

            batch_results = self._slew_checker.check_slew_feasibility_batch(batch_candidates)

            slew_feasible = []
            for candidate, slew_result in zip(candidates, batch_results):
                if slew_result.feasible:
                    slew_feasible.append((candidate, slew_result))
        else:
            # 逐个检查
            slew_feasible = []
            for candidate_data in candidates:
                sat, window, imaging_mode, imaging_duration, window_start, window_end = candidate_data
                usage = self._sat_resource_usage.get(sat.id, {})
                last_task_end = usage.get('last_task_end', self.mission.start_time)
                prev_target = self._get_previous_task_target(sat.id)

                self._ensure_slew_checker_initialized()
                if self._slew_checker is None:
                    continue

                slew_result = self._slew_checker.check_slew_feasibility(
                    sat.id, prev_target, task, last_task_end, window_start, imaging_duration,
                    window_end=window_end
                )

                if slew_result.feasible:
                    slew_feasible.append((candidate_data, slew_result))

        if not slew_feasible:
            return None

        # 使用统一批量约束检查（SAA、时间、资源）
        if self._unified_checker:
            unified_candidates = []
            for candidate_data, slew_result in slew_feasible:
                sat, window, imaging_mode, imaging_duration, window_start, window_end = candidate_data
                usage = self._sat_resource_usage.get(sat.id, {})
                actual_start = slew_result.actual_start
                actual_end = actual_start + timedelta(seconds=imaging_duration)

                power_needed = 0.0
                storage_produced = 0.0
                if imaging_mode is not None:
                    try:
                        power_profile = imaging_mode.get_power_profile(imaging_duration)
                        power_needed = getattr(power_profile, 'total_energy', 0.0)
                        storage_produced = getattr(imaging_mode, 'data_rate', 0.0) * imaging_duration
                    except:
                        pass

                sat_position, sat_velocity = self._get_satellite_position(sat, actual_start)

                unified_candidate = UnifiedBatchCandidate(
                    sat_id=sat.id,
                    satellite=sat,
                    target=task,
                    window_start=actual_start,
                    window_end=actual_end,
                    prev_end_time=usage.get('last_task_end', self.mission.start_time),
                    imaging_duration=imaging_duration,
                    prev_target=self._get_previous_task_target(sat.id),
                    sat_position=sat_position,
                    sat_velocity=sat_velocity,
                    power_needed=power_needed,
                    storage_produced=storage_produced
                )
                unified_candidates.append(unified_candidate)

            existing_tasks = []
            for sat in self.mission.satellites:
                usage = self._sat_resource_usage.get(sat.id, {})
                for task_info in usage.get('scheduled_tasks', []):
                    existing_tasks.append({
                        'sat_id': sat.id,
                        'satellite_id': sat.id,
                        'start': task_info.get('start'),
                        'end': task_info.get('end')
                    })

            satellite_states = {}
            for sat in self.mission.satellites:
                usage = self._sat_resource_usage.get(sat.id, {})
                satellite_states[sat.id] = {
                    'power': usage.get('power', sat.capabilities.power_capacity),
                    'storage': usage.get('storage', 0.0),
                    'power_capacity': sat.capabilities.power_capacity,
                    'storage_capacity': sat.capabilities.storage_capacity
                }

            unified_results = self._unified_checker.check_fast_phase_batch(
                candidates=unified_candidates,
                existing_tasks=existing_tasks,
                satellite_states=satellite_states,
                early_termination=True
            )

            return self._select_best_assignment(slew_feasible, unified_results)

        # 回退：简单选择第一个可行的
        for candidate_data, slew_result in slew_feasible:
            sat, window, imaging_mode, _, _, _ = candidate_data
            return (sat.id, window, imaging_mode, slew_result)

        return None

    def _select_best_assignment(
        self,
        candidates: List[Tuple],
        results: List[UnifiedBatchResult]
    ) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """
        选择最佳任务分配 - 子类可覆盖

        默认策略: 选择最早开始的分配
        """
        best = None
        best_start = None

        for (candidate, slew_result), unified_result in zip(candidates, results):
            if not unified_result.feasible:
                continue

            actual_start = slew_result.actual_start

            if best_start is None or actual_start < best_start:
                best_start = actual_start
                sat, window, imaging_mode, _, _, _ = candidate
                best = (sat.id, window, imaging_mode, slew_result)

        return best

    @abstractmethod
    def _sort_tasks(self, tasks: List[Any]) -> List[Any]:
        """
        任务排序策略 - 子类必须实现

        Args:
            tasks: 任务列表

        Returns:
            排序后的任务列表
        """
        pass

    # ========== 辅助方法 ==========

    def _get_feasible_windows(self, sat: Any, task: Any) -> List[Any]:
        """获取可行的时间窗口"""
        if self.window_cache:
            target_id = task.target_id if isinstance(task, ObservationTask) else task.id
            return self.window_cache.get_windows(sat.id, target_id)
        return []

    def _extract_window_times(self, window: Any) -> Tuple[Optional[datetime], Optional[datetime]]:
        """从窗口对象提取开始和结束时间"""
        if isinstance(window, dict):
            return window.get('start'), window.get('end')
        else:
            return getattr(window, 'start_time', None), getattr(window, 'end_time', None)

    def _can_satellite_perform_task(self, sat: Any, task: Any) -> bool:
        """检查卫星是否能执行任务"""
        if not sat.capabilities.imaging_modes:
            return False

        required_resolution = getattr(task, 'resolution_required', None)
        if required_resolution is not None:
            sat_resolution = getattr(sat.capabilities, 'resolution', None)
            try:
                if sat_resolution is not None and required_resolution is not None:
                    if sat_resolution > required_resolution:
                        return False
            except TypeError:
                pass

        return True

    def _select_imaging_mode(self, sat: Any, task: Any):
        """选择成像模式"""
        from core.models import ImagingMode

        modes = sat.capabilities.imaging_modes
        if not modes:
            return ImagingMode.PUSH_BROOM

        mode = modes[0]
        if hasattr(mode, '_mock_name') or not isinstance(mode, (ImagingMode, str)):
            return ImagingMode.PUSH_BROOM
        return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)

    def _check_resource_constraints(self, sat: Any, task: Any, imaging_mode: Any = None) -> bool:
        """检查资源约束

        Args:
            sat: 卫星对象
            task: 任务对象
            imaging_mode: 成像模式（可选，默认自动选择）

        Returns:
            bool: 资源是否充足
        """
        # 如果没有提供成像模式，自动选择
        if imaging_mode is None:
            imaging_mode = self._select_imaging_mode(sat, task)

        usage = self._sat_resource_usage.get(sat.id, {})

        if self.consider_power:
            duration = self._imaging_calculator.calculate(task, imaging_mode, sat)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_needed = sat.capabilities.power_capacity * power_coefficient * (duration / 3600)

            if usage.get('power', 0) < power_needed:
                return False

        if self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_needed = self._imaging_calculator.get_storage_consumption(task, imaging_mode, data_rate)
            current_storage = usage.get('storage', 0)
            capacity = sat.capabilities.storage_capacity

            if current_storage + storage_needed > capacity:
                return False

        return True

    def _get_previous_task_target(self, sat_id: str) -> Optional[Any]:
        """获取卫星上一个已调度任务的目标"""
        return self._last_task_target.get(sat_id)

    def _create_scheduled_task(
        self, task: Any, sat_id: str, window: Any, imaging_mode: Any,
        slew_result: Optional[SlewFeasibilityResult] = None
    ) -> ScheduledTask:
        """创建ScheduledTask对象"""
        window_start, window_end = self._extract_window_times(window)
        sat = self.mission.get_satellite_by_id(sat_id)
        imaging_duration = self._imaging_calculator.calculate(task, imaging_mode, sat)

        usage = self._sat_resource_usage.get(sat_id, {})
        power_before = usage.get('power', 0)
        storage_before = usage.get('storage', 0)

        power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
        power_consumed = 0.0
        if sat and self.consider_power:
            power_consumed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)

        storage_used = 0.0
        if sat and self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_used = self._imaging_calculator.get_storage_consumption(task, imaging_mode, data_rate)

        if slew_result:
            slew_angle = slew_result.slew_angle
            slew_time_seconds = slew_result.slew_time
            actual_start = slew_result.actual_start
        else:
            slew_angle = 0.0
            slew_time_seconds = 30.0
            actual_start = window_start

        actual_end = actual_start + timedelta(seconds=imaging_duration)

        task_id = task.task_id if isinstance(task, ObservationTask) else task.id
        target_id = task.target_id if isinstance(task, ObservationTask) else task.id

        scheduled_task = ScheduledTask(
            task_id=task_id,
            satellite_id=sat_id,
            target_id=target_id,
            imaging_start=actual_start,
            imaging_end=actual_end,
            imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
            slew_angle=slew_angle,
            slew_time=slew_time_seconds,
            storage_before=storage_before,
            storage_after=storage_before + storage_used,
            power_before=power_before,
            power_after=power_before - power_consumed,
        )

        return scheduled_task

    def _update_resource_usage(
        self, sat_id: str, task: Any, window: Any, scheduled_task: ScheduledTask
    ) -> None:
        """更新资源使用"""
        usage = self._sat_resource_usage.get(sat_id)
        if usage is None:
            return

        if self.consider_power:
            usage['power'] = scheduled_task.power_after

        if self.consider_storage:
            usage['storage'] = scheduled_task.storage_after

        usage['last_task_end'] = scheduled_task.imaging_end

        if 'scheduled_tasks' not in usage:
            usage['scheduled_tasks'] = []

        task_id = task.task_id if isinstance(task, ObservationTask) else task.id
        target_id = task.target_id if isinstance(task, ObservationTask) else task.id
        usage['scheduled_tasks'].append({
            'start': scheduled_task.imaging_start,
            'end': scheduled_task.imaging_end,
            'task_id': task_id,
            'target_id': target_id
        })

    def _determine_failure_reason(self, task: Any) -> TaskFailureReason:
        """确定任务失败原因"""
        for sat_id, usage in self._sat_resource_usage.items():
            sat = self.mission.get_satellite_by_id(sat_id)
            if sat:
                imaging_mode = self._select_imaging_mode(sat, task)

                # 检查电量约束
                if self.consider_power:
                    power_ratio = usage.get('power', sat.capabilities.power_capacity) / sat.capabilities.power_capacity
                    if power_ratio < 0.1:  # 电量低于10%
                        return TaskFailureReason.POWER_CONSTRAINT

                # 检查存储约束
                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                storage_needed = self._imaging_calculator.get_storage_consumption(task, imaging_mode, data_rate)
                if usage['storage'] + storage_needed > sat.capabilities.storage_capacity:
                    return TaskFailureReason.STORAGE_CONSTRAINT

        has_visible_window = False
        for sat in self.mission.satellites:
            windows = self._get_feasible_windows(sat, task)
            if windows:
                has_visible_window = True
                break

        if not has_visible_window:
            return TaskFailureReason.NO_VISIBLE_WINDOW

        return TaskFailureReason.UNKNOWN

    def _perf_start(self):
        """开始性能计时"""
        return time.perf_counter()

    def _perf_end(self, name: str, t_start: float) -> None:
        """结束性能计时"""
        elapsed = time.perf_counter() - t_start
        self._perf_stats[name]['count'] += 1
        self._perf_stats[name]['total_time'] += elapsed
        if elapsed > self._perf_stats[name]['max_time']:
            self._perf_stats[name]['max_time'] = elapsed
