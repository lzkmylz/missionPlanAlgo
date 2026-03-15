"""
Real Greedy Scheduler Implementation

基于启发式规则的贪心调度算法
策略：每次选择当前最优的卫星-目标组合，考虑完整约束检查

Features:
- Visibility window calculation
- Resource constraint checking (storage, power)
- Imaging time calculation
- Time conflict detection
- Support for mixed satellite types (optical + SAR)
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math
import logging
import time
from collections import defaultdict

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from ..frequency_utils import ObservationTask, create_observation_tasks
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile
from ..constraints import SlewConstraintChecker, SlewFeasibilityResult
from scheduler.common.constraint_checker import ConstraintChecker, ConstraintContext
from scheduler.common.config import ConstraintConfig
from scheduler.common.clustering_mixin import ClusteringMixin, ClusterTask
from scheduler.common.quality_aware_mixin import QualityAwareMixin
from scheduler.constraints.batch_slew_calculator import BatchSlewCandidate, BatchSlewCalculator
from scheduler.constraints.batch_slew_constraint_checker import BatchSlewConstraintChecker
from scheduler.constraints.unified_batch_constraint_checker import (
    UnifiedBatchConstraintChecker, UnifiedBatchCandidate
)
from core.dynamics.attitude_precache import get_attitude_precache_manager
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
from core.models.target import Target
from core.quality.quality_config import QualityScoreConfig

# 模块级别logger定义
logger = logging.getLogger(__name__)


class GreedyScheduler(BaseScheduler, ClusteringMixin, QualityAwareMixin):
    """
    Real Greedy Scheduler with full constraint checking

    调度策略：
    1. 按优先级排序任务（或其他启发式）
    2. 对每个任务，检查所有卫星的可见窗口
    3. 验证资源约束（存储、电量）
    4. 检查时间冲突
    5. 选择最佳卫星-窗口组合
    6. 更新资源状态并继续

    Attributes:
        heuristic: 排序启发式 ('priority', 'earliest_window', 'deadline')
        consider_power: 是否考虑电量约束
        consider_storage: 是否考虑存储约束
        consider_time_conflicts: 是否考虑时间冲突
        enable_quality_filtering: 是否启用质量筛选
        quality_score_weight: 质量评分在启发式中的权重
    """

    # Default minimum time gap between tasks (slew time + settling time)
    DEFAULT_SLEW_TIME = timedelta(seconds=30)

    # Minimum window duration as ratio of required imaging time
    MIN_WINDOW_RATIO = 1.0

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize GreedyScheduler

        Args:
            config: Configuration dictionary
                - heuristic: Sorting heuristic ('priority', 'earliest_window', 'deadline')
                - consider_power: Whether to check power constraints (default True)
                - consider_storage: Whether to check storage constraints (default True)
                - consider_time_conflicts: Whether to check time conflicts (default True)
                - min_imaging_duration: Minimum imaging duration in seconds
                - max_imaging_duration: Maximum imaging duration in seconds
                - enable_clustering: Enable target clustering (default False)
                - cluster_radius_km: Cluster radius in km (default 10.0)
                - min_cluster_size: Minimum targets per cluster (default 2)
                - quality_config: Quality scoring configuration (dict or QualityScoreConfig)
                - enable_quality_filtering: Enable quality-based window filtering (default True)
                - min_quality_threshold: Minimum quality threshold for window filtering (default 0.3)
                - quality_score_weight: Weight of quality score in assignment scoring (default 20.0)
        """
        super().__init__("Greedy", config)
        ClusteringMixin.__init__(self, config)
        config = config or {}
        self.heuristic = config.get('heuristic', 'priority')
        self.consider_power = config.get('consider_power', True)
        self.consider_storage = config.get('consider_storage', True)
        self.consider_time_conflicts = config.get('consider_time_conflicts', True)

        # Initialize quality-aware mixin
        quality_config = config.get('quality_config')
        if isinstance(quality_config, dict):
            quality_config = QualityScoreConfig.from_dict(quality_config)
        self.__init_quality_aware__(
            quality_config=quality_config,
            enable_quality_filtering=config.get('enable_quality_filtering', True),
            min_quality_threshold=config.get('min_quality_threshold', 0.3),
        )

        # 质量评分在启发式中的权重
        self.quality_score_weight = config.get('quality_score_weight', 20.0)

        # Initialize imaging time calculator
        # 使用ImagingTimeCalculator的默认值（基于实际卫星数据）
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=config.get('min_imaging_duration'),
            max_duration=config.get('max_imaging_duration'),
            default_duration=config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(config.get('power_coefficients'))

        # Track satellite resource usage during scheduling
        self._sat_resource_usage: Dict[str, Dict[str, Any]] = {}

        # 统一批量约束检查器（延迟初始化，只创建一次）
        self._unified_checker = None

        # 性能分析数据
        self._perf_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'max_time': 0.0, 'min_time': float('inf')})

        # 详细的拒绝原因统计
        self._rejection_stats = {
            'phase1_no_windows': 0,           # 没有可见窗口
            'phase1_window_too_short': 0,     # 窗口太短
            'phase1_time_constraint': 0,      # 时间约束不满足
            'phase1_resource_power': 0,       # 电量约束不满足
            'phase1_resource_storage': 0,     # 存储约束不满足
            'phase2_attitude_roll': 0,        # 滚转角超限
            'phase2_attitude_pitch': 0,       # 俯仰角超限
            'phase2_attitude_missing': 0,     # 无法获取姿态角
            'phase3_slew_infeasible': 0,      # 机动不可行
            'phase4_saa': 0,                  # SAA约束
            'phase4_time_conflict': 0,        # 时间冲突
            'phase4_resource': 0,             # 资源约束（阶段4）
            'phase4_other': 0,                # 其他约束
            'total_checked': 0,               # 总检查次数
            'total_passed': 0,                # 总通过次数
        }
        self._last_task_rejection_reason = None  # 最后一个任务的拒绝原因

    def get_parameters(self) -> Dict[str, Any]:
        """Return algorithm configurable parameters"""
        params = {
            'heuristic': self.heuristic,
            'consider_power': self.consider_power,
            'consider_storage': self.consider_storage,
            'consider_time_conflicts': self.consider_time_conflicts,
            'enable_quality_filtering': self.enable_quality_filtering,
            'min_quality_threshold': self.min_quality_threshold,
            'quality_score_weight': self.quality_score_weight,
        }
        # Add clustering parameters
        params.update(self.get_clustering_config())
        return params

    def _perf_start(self) -> float:
        """开始计时"""
        return time.perf_counter()

    def _perf_end(self, name: str, start_time: float) -> None:
        """结束计时并记录"""
        elapsed = time.perf_counter() - start_time
        stats = self._perf_stats[name]
        stats['count'] += 1
        stats['total_time'] += elapsed
        stats['max_time'] = max(stats['max_time'], elapsed)
        stats['min_time'] = min(stats['min_time'], elapsed)

    def _print_perf_report(self) -> None:
        """打印性能分析报告"""
        print("\n" + "="*80)
        print("Greedy调度器性能分析报告")
        print("="*80)
        print(f"{'环节':<40} {'调用次数':>10} {'总耗时(s)':>12} {'平均(ms)':>10} {'最大(ms)':>10}")
        print("-"*80)

        # 按总耗时排序
        sorted_stats = sorted(self._perf_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)

        total_time = sum(s['total_time'] for _, s in sorted_stats)

        for name, stats in sorted_stats:
            avg_ms = (stats['total_time'] / stats['count']) * 1000 if stats['count'] > 0 else 0
            max_ms = stats['max_time'] * 1000
            print(f"{name:<40} {stats['count']:>10} {stats['total_time']:>12.3f} {avg_ms:>10.3f} {max_ms:>10.3f}")

        print("-"*80)
        print(f"{'总计':<40} {'':>10} {total_time:>12.3f} {'':>10} {'':>10}")
        print("="*80)

        # 计算各环节占比
        print("\n各环节耗时占比:")
        for name, stats in sorted_stats[:10]:  # 只显示前10
            percentage = (stats['total_time'] / total_time) * 100 if total_time > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  {name:<38} {percentage:>6.2f}% {bar}")
        print("="*80 + "\n")

    def _print_rejection_stats(self) -> None:
        """打印详细的任务拒绝原因统计"""
        stats = self._rejection_stats
        total_checked = stats['total_checked']
        total_passed = stats['total_passed']

        if total_checked == 0:
            return

        print("\n" + "="*80)
        print("调度失败原因详细分析")
        print("="*80)

        print(f"\n总体统计:")
        print(f"  总检查候选数: {total_checked}")
        print(f"  通过候选数: {total_passed}")
        print(f"  总体通过率: {(total_passed/total_checked)*100:.2f}%")

        print(f"\n阶段1 - 候选收集:")
        print(f"  无可见窗口: {stats['phase1_no_windows']}")
        print(f"  窗口太短: {stats['phase1_window_too_short']}")
        print(f"  时间约束不满足: {stats['phase1_time_constraint']}")
        print(f"  电量约束不满足: {stats['phase1_resource_power']}")
        print(f"  存储约束不满足: {stats['phase1_resource_storage']}")

        print(f"\n阶段2/3 - 姿态约束:")
        print(f"  滚转角超限: {stats['phase2_attitude_roll']}")
        print(f"  俯仰角超限: {stats['phase2_attitude_pitch']}")
        print(f"  无法获取姿态角: {stats['phase2_attitude_missing']}")

        print(f"\n阶段4 - 精确约束检查:")
        print(f"  机动不可行: {stats['phase3_slew_infeasible']}")
        print(f"  SAA约束: {stats['phase4_saa']}")
        print(f"  时间冲突: {stats['phase4_time_conflict']}")
        print(f"  资源约束: {stats['phase4_resource']}")
        print(f"  其他约束: {stats['phase4_other']}")

        # 计算各阶段的过滤率
        print(f"\n各阶段过滤率:")
        phase1_filtered = stats['phase1_no_windows'] + stats['phase1_window_too_short'] + \
                         stats['phase1_time_constraint'] + stats['phase1_resource_power'] + \
                         stats['phase1_resource_storage']
        phase2_filtered = stats['phase2_attitude_roll'] + stats['phase2_attitude_pitch'] + stats['phase2_attitude_missing']
        phase4_filtered = stats['phase3_slew_infeasible'] + stats['phase4_saa'] + stats['phase4_time_conflict'] + \
                         stats['phase4_resource'] + stats['phase4_other']

        if total_checked > 0:
            print(f"  阶段1: {(phase1_filtered/total_checked)*100:.1f}%")
            remaining_after_phase1 = total_checked - phase1_filtered
            if remaining_after_phase1 > 0:
                print(f"  阶段2/3: {(phase2_filtered/remaining_after_phase1)*100:.1f}%")
                remaining_after_phase2 = remaining_after_phase1 - phase2_filtered
                if remaining_after_phase2 > 0:
                    print(f"  阶段4: {(phase4_filtered/remaining_after_phase2)*100:.1f}%")

        print("="*80 + "\n")

    def _initialize_slew_checker(self) -> None:
        """初始化批量姿态机动约束检查器

        覆盖基类方法，使用 BatchSlewConstraintChecker 替代普通检查器
        以启用向量化批量计算优化。
        """
        if self.mission is None:
            return

        # 如果已经设置了外部检查器，跳过初始化
        if self._slew_checker is not None:
            return

        # 强制使用批量姿态机动约束检查器
        # 高精度要求：禁止简化模式
        self._slew_checker = BatchSlewConstraintChecker(
            self.mission,
            use_precise_model=True
        )

        # 设置状态跟踪器（如果可用）
        if hasattr(self, '_state_tracker') and self._state_tracker is not None:
            self._slew_checker.set_state_tracker(self._state_tracker)

        logger.info("GreedyScheduler: Initialized BatchSlewConstraintChecker for vectorized computation")

    def _get_previous_task_target(self, sat_id: str) -> Optional[Target]:
        """获取卫星上一个任务的目标

        Args:
            sat_id: 卫星ID

        Returns:
            上一个目标或None
        """
        usage = self._sat_resource_usage.get(sat_id, {})
        scheduled_tasks = usage.get('scheduled_tasks', [])

        if scheduled_tasks:
            last_task = scheduled_tasks[-1]
            if hasattr(last_task, 'target'):
                return last_task.target

        return None

    def schedule(self) -> ScheduleResult:
        """
        Execute greedy scheduling with full constraint checking

        Returns:
            ScheduleResult: Scheduling result with all scheduled tasks and failures
        """
        self._start_timer()
        self._validate_initialization()

        # Initialize resource tracking for each satellite
        self._sat_resource_usage = {}
        for sat in self.mission.satellites:
            current_power = getattr(sat, 'current_power', None)
            # Handle Mock objects and None values
            try:
                if current_power is not None and current_power > 0:
                    power = current_power
                else:
                    power = sat.capabilities.power_capacity
            except TypeError:
                # Handle Mock objects or other non-comparable types
                power = sat.capabilities.power_capacity

            self._sat_resource_usage[sat.id] = {
                'power': power,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': []  # Track scheduled tasks for conflict detection
            }

        # Initialize slew constraint checker first (required by unified constraint checker)
        self._initialize_slew_checker()

        # Initialize unified constraint checker
        # 强制使用完整精确模式，禁用所有简化计算
        constraint_config = ConstraintConfig(
            consider_power=self.consider_power,
            consider_storage=self.consider_storage,
            enable_saa_check=True,
            mode='full'  # 使用完整精确模式，禁用简化计算
        )
        self._constraint_checker = ConstraintChecker(self.mission, constraint_config)

        # Use the precise slew checker if available (passed from UnifiedScheduler)
        if self._slew_checker is not None:
            self._constraint_checker.set_slew_checker(self._slew_checker)

        self._constraint_checker.initialize()

        # Initialize SAA constraint checker
        self._initialize_saa_checker()

        # Initialize attitude state tracking
        self._initialize_attitude_state()

        # Pass position cache to unified constraint checker if available
        if self._position_cache is not None and self._constraint_checker is not None:
            self._constraint_checker.set_position_cache(self._position_cache)

        # Precompute satellite positions to accelerate scheduling
        # 默认启用预计算以加速姿态角计算，除非显式禁用
        if self.config.get('precompute_positions', True):
            print("    Precomputing satellite positions...")
            self._precompute_satellite_positions(time_step_seconds=self.config.get('precompute_step_seconds', 1.0))

        # ========== 空间换时间：姿态预计算缓存 ==========
        # 使用Java预计算的姿态采样数据，避免实时计算的性能开销
        # Java在可见性计算阶段已经计算了所有窗口的完整姿态采样（1秒步长）
        if self.config.get('enable_attitude_precache', False):
            logger = logging.getLogger(__name__)
            logger.info("初始化姿态预计算缓存（使用Java预计算数据）...")
            t_precache = self._perf_start()

            try:
                precache_manager = get_attitude_precache_manager()

                # 收集所有窗口对象（从window_cache）
                all_visibility_windows = []
                try:
                    for (sat_id, target_id), windows in self.window_cache._windows.items():
                        # 跳过地面站窗口
                        if target_id and str(target_id).startswith('GS:'):
                            continue
                        all_visibility_windows.extend(windows)
                except Exception as e:
                    logger.warning(f"从 window_cache 加载窗口失败: {e}")

                # 从Java预计算数据加载姿态缓存
                n_loaded = precache_manager.load_precomputed_attitudes_from_windows(all_visibility_windows)
                stats = precache_manager.get_stats()
                logger.info(f"加载了 {n_loaded} 个Java预计算姿态，缓存大小: {stats['memory_mb']:.1f}MB "
                           f"(有效: {stats['valid_entries']}, 不可行: {stats['infeasible_entries']})")

                # 存储引用供后续使用
                self._attitude_precache_manager = precache_manager

                self._perf_end('attitude_precache_init', t_precache)
            except Exception as e:
                logger.warning(f"姿态预计算缓存初始化异常: {e}，将使用实时计算")
                self._attitude_precache_manager = None

        # Reset clustering state if needed
        if self.enable_clustering:
            t = self._perf_start()
            self.reset_clustering_state()
            self._perf_end('reset_clustering_state', t)

        # Get tasks based on whether clustering is enabled
        t = self._perf_start()
        if self.enable_clustering:
            pending_tasks = self._sort_tasks(self._get_clustered_tasks())
        else:
            pending_tasks = self._sort_tasks(self._create_frequency_aware_tasks())
        self._perf_end('create_and_sort_tasks', t)

        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, Any] = {}

        # 禁用快速筛选模式，使用完整精确计算
        if self._slew_checker is not None and hasattr(self._slew_checker, 'set_skip_reset_calculation'):
            self._slew_checker.set_skip_reset_calculation(False)
            logger = logging.getLogger(__name__)
            logger.info("禁用姿态机动快速筛选模式，使用完整精确计算")

        # Main scheduling loop
        loop_t = self._perf_start()
        total_tasks = len(pending_tasks)
        logger.info(f"开始调度主循环，共 {total_tasks} 个任务")

        for task_idx, task in enumerate(pending_tasks):
            # 每100个任务报告进度
            if (task_idx + 1) % 100 == 0:
                elapsed = time.perf_counter() - self._start_time if hasattr(self, '_start_time') and self._start_time else 0
                logger.info(f"调度进度: {task_idx + 1}/{total_tasks} 任务, 已调度 {len(scheduled_tasks)}, 耗时 {elapsed:.1f}s")

            task_t = self._perf_start()
            best_assignment = self._find_best_assignment(task)
            self._perf_end('find_best_assignment_per_task', task_t)

            if best_assignment:
                sat_id, window, imaging_mode, slew_result = best_assignment

                # Create scheduled task with slew information
                t = self._perf_start()
                scheduled_task = self._create_scheduled_task(
                    task, sat_id, window, imaging_mode, slew_result
                )
                self._perf_end('create_scheduled_task', t)

                # 如果创建任务失败（如姿态约束检查失败），跳过此任务
                if scheduled_task is None:
                    task_id = task.task_id if hasattr(task, 'task_id') else task.id
                    self._record_failure(
                        task_id=task_id,
                        reason="attitude_constraint_violation",
                        detail=f"Task {task_id} violates attitude constraints at actual imaging time"
                    )
                    continue

                scheduled_tasks.append(scheduled_task)

                # Update resource usage
                t = self._perf_start()
                self._update_resource_usage(sat_id, task, window, scheduled_task)
                self._perf_end('update_resource_usage', t)

                self._add_convergence_point(len(scheduled_tasks))
            else:
                # Record failure reason
                reason = self._determine_failure_reason(task)
                task_id = task.task_id if isinstance(task, ObservationTask) else task.id
                self._record_failure(
                    task_id=task_id,
                    reason=reason,
                    detail=f"No feasible assignment found for task {task_id}"
                )
                unscheduled[task_id] = self._failure_log[-1]
        self._perf_end('main_scheduling_loop', loop_t)

        # Calculate makespan
        t = self._perf_start()
        makespan = self._calculate_makespan(scheduled_tasks)
        self._perf_end('calculate_makespan', t)
        computation_time = self._stop_timer()

        # Print performance report
        self._print_perf_report()

        # Print detailed rejection statistics
        self._print_rejection_stats()

        # Build failure summary
        failure_summary = self._build_failure_summary()

        # Calculate target observation counts for frequency fitness
        target_obs_count = self._calculate_target_obs_count(scheduled_tasks)

        # Calculate frequency-aware fitness score
        frequency_fitness = self._calculate_frequency_fitness(target_obs_count, base_score=len(scheduled_tasks))

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self._iterations,
            convergence_curve=self._convergence_curve,
            failure_summary=failure_summary
        )

    def _sort_tasks(self, tasks: List[Any]) -> List[Any]:
        """
        Sort tasks based on the configured heuristic

        Args:
            tasks: List of tasks to sort

        Returns:
            Sorted list of tasks
        """
        if self.heuristic == 'priority':
            return self._sort_tasks_by_priority(tasks)
        elif self.heuristic == 'earliest_window':
            return self._sort_tasks_by_earliest_window(tasks)
        elif self.heuristic == 'deadline':
            return self._sort_tasks_by_deadline(tasks)
        else:
            # Default to priority sorting
            return self._sort_tasks_by_priority(tasks)

    def _sort_tasks_by_priority(self, tasks: List[Any]) -> List[Any]:
        """
        Sort tasks by priority (lower value = higher priority)

        优先级范围1-100，数字越小优先级越高。
        例如：priority=1 比 priority=100 优先级更高。

        Args:
            tasks: List of tasks

        Returns:
            Sorted list (highest priority first, i.e., lowest value first)
        """
        def priority_key(task):
            # Lower priority value = higher priority (1 is highest, 100 is lowest)
            priority = getattr(task, 'priority', 50) or 50
            return priority  # Ascending order: lower value = higher priority

        return sorted(tasks, key=priority_key)

    def _sort_tasks_by_earliest_window(self, tasks: List[Any]) -> List[Any]:
        """
        Sort tasks by earliest visibility window

        Args:
            tasks: List of tasks

        Returns:
            Sorted list (earliest window first)
        """
        def window_key(task):
            earliest_start = None
            for sat in self.mission.satellites:
                windows = self._get_windows(sat, task)
                if windows:
                    window_start = windows[0]['start'] if isinstance(windows[0], dict) else windows[0].start_time
                    if earliest_start is None or window_start < earliest_start:
                        earliest_start = window_start

            if earliest_start is None:
                return datetime.max
            return earliest_start

        return sorted(tasks, key=window_key)

    def _sort_tasks_by_deadline(self, tasks: List[Any]) -> List[Any]:
        """
        Sort tasks by deadline (earliest first) - EDD style

        Args:
            tasks: List of tasks

        Returns:
            Sorted list (earliest deadline first)
        """
        def deadline_key(task):
            if task.time_window_end is None:
                return (float('inf'), 0)
            priority = getattr(task, 'priority', 5) or 5
            return (task.time_window_end.timestamp(), -priority)

        return sorted(tasks, key=deadline_key)

    def _find_best_assignment(self, task: Any) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """
        Find the best satellite-window assignment for a task using unified constraint checker.

        采用批量预筛选优化：
        1. 阶段1：候选收集 - 收集所有（卫星，窗口）候选，进行快速预筛选
        2. 阶段2：批量姿态预计算 - 一次性计算所有候选的姿态角
        3. 阶段3：姿态过滤 - 基于侧摆角限制快速过滤
        4. 阶段4：精确检查 - 仅对通过预筛选的候选进行详细约束检查

        Args:
            task: Target task to schedule

        Returns:
            Tuple of (satellite_id, window, imaging_mode, slew_result) or None if no valid assignment
        """
        t_total = self._perf_start()

        # ========== 阶段1：候选收集（快速预筛选）==========
        t = self._perf_start()
        logger = logging.getLogger(__name__)
        logger.debug(f"[Task {getattr(task, 'task_id', getattr(task, 'id', '?'))}] Finding best assignment...")
        candidates = []  # List of (sat, window, imaging_mode, imaging_duration, window_start, window_end)

        # 记录阶段1的拒绝原因
        phase1_no_windows = 0
        phase1_window_too_short = 0
        phase1_time_constraint = 0
        phase1_resource_reject = 0

        for sat in self.mission.satellites:
            # Get visibility windows
            windows = self._get_windows(sat, task)
            if not windows:
                phase1_no_windows += 1
                continue

            # 快速能力检查（每个卫星只检查一次）
            if self._constraint_checker is None and not self._can_satellite_perform_task(sat, task):
                continue

            for window in windows:
                self._rejection_stats['total_checked'] += 1

                # Extract window times
                window_start, window_end = self._extract_window_times(window)
                if window_start is None or window_end is None:
                    continue

                # Check if window is within target's time window constraints
                if not self._is_window_within_target_constraints(task, window_start, window_end):
                    phase1_time_constraint += 1
                    continue

                # Calculate required imaging time
                imaging_mode = self._select_imaging_mode(sat, task)
                imaging_duration = self._calculate_imaging_time(task, imaging_mode, sat)

                # Check if window is long enough
                window_duration = (window_end - window_start).total_seconds()
                if window_duration < imaging_duration * self.MIN_WINDOW_RATIO:
                    phase1_window_too_short += 1
                    continue

                # 快速资源检查
                if self._constraint_checker is None:
                    if not self._check_resource_constraints(sat, task, imaging_mode):
                        phase1_resource_reject += 1
                        continue

                candidates.append((sat, window, imaging_mode, imaging_duration, window_start, window_end))
                self._rejection_stats['total_passed'] += 1

        # 如果没有候选，记录拒绝原因
        if not candidates:
            if phase1_no_windows >= len(self.mission.satellites):
                self._rejection_stats['phase1_no_windows'] += 1
                self._last_task_rejection_reason = 'no_visible_window'
            elif phase1_window_too_short > 0:
                self._rejection_stats['phase1_window_too_short'] += 1
                self._last_task_rejection_reason = 'window_too_short'
            elif phase1_time_constraint > 0:
                self._rejection_stats['phase1_time_constraint'] += 1
                self._last_task_rejection_reason = 'time_constraint'
            elif phase1_resource_reject > 0:
                # 区分电量和存储
                self._rejection_stats['phase1_resource_power'] += 1
                self._last_task_rejection_reason = 'resource_power'

        self._perf_end('phase1_candidate_collection', t)
        logger.debug(f"  Phase 1: Found {len(candidates)} candidates")

        if not candidates:
            return None

        # ========== 阶段2&3：O(1)姿态预计算与过滤（空间换时间优化）==========
        t = self._perf_start()

        # 记录阶段2/3的拒绝原因
        phase2_attitude_roll = 0
        phase2_attitude_pitch = 0
        phase2_attitude_missing = 0

        # 优先使用预计算缓存，如果没有则实时计算
        if hasattr(self, '_attitude_precache_manager') and self._attitude_precache_manager is not None:
            # 使用预计算缓存 - O(1)查询
            # 注意：姿态缓存只存储了窗口开始时刻的姿态，所以只检查窗口开始时刻
            filtered_candidates = []
            for sat, window, imaging_mode, imaging_duration, window_start, window_end in candidates:
                # 直接从缓存获取预计算的姿态角（只检查窗口开始时刻）
                attitude = self._attitude_precache_manager.get_attitude(sat.id, window_start)

                if attitude is not None:
                    roll, pitch = attitude
                    # 分别检查滚转角和俯仰角（严格检查，不允许超出名义约束）
                    if abs(roll) > sat.capabilities.max_roll_angle:
                        phase2_attitude_roll += 1
                        continue
                    if abs(pitch) > sat.capabilities.max_pitch_angle:
                        phase2_attitude_pitch += 1
                        continue
                    filtered_candidates.append((sat, window, imaging_mode, imaging_duration, window_start, window_end))
                else:
                    # 如果无法获取姿态角，跳过此候选（保守策略）
                    phase2_attitude_missing += 1
                    continue

            # 记录缓存命中率统计（每100个任务记录一次）
            if hasattr(self, '_task_count'):
                self._task_count += 1
                if self._task_count % 100 == 0:
                    stats = self._attitude_precache_manager.get_stats()
                    logger.debug(f"姿态缓存命中率: {stats['hit_rate']*100:.1f}% ({stats['cache_hits']}/{stats['cache_hits']+stats['cache_misses']})")
            else:
                self._task_count = 1
        else:
            # 回退到实时批量计算（原逻辑）
            attitude_cache = {}
            if self._attitude_calculator is not None:
                attitude_candidates = []
                for sat, window, imaging_mode, imaging_duration, window_start, window_end in candidates:
                    imaging_time = window_start + timedelta(seconds=(window_end - window_start).total_seconds() / 2)
                    attitude_candidates.append((sat, task, imaging_time))

                if attitude_candidates:
                    attitude_cache = self._batch_precalculate_attitudes(attitude_candidates)

            # 姿态预筛选
            filtered_candidates = []
            for sat, window, imaging_mode, imaging_duration, window_start, window_end in candidates:
                imaging_time = window_start + timedelta(seconds=(window_end - window_start).total_seconds() / 2)
                attitude = attitude_cache.get((sat.id, imaging_time))

                if attitude is not None:
                    # 分别检查滚转角和俯仰角（严格检查，不允许超出名义约束）
                    if abs(attitude.roll) > sat.capabilities.max_roll_angle:
                        continue
                    if abs(attitude.pitch) > sat.capabilities.max_pitch_angle:
                        continue
                    filtered_candidates.append((sat, window, imaging_mode, imaging_duration, window_start, window_end))
                else:
                    # 如果无法获取姿态角，保留候选（而不是跳过）
                    # 在测试环境或没有预计算数据时，让后续约束检查来处理
                    filtered_candidates.append((sat, window, imaging_mode, imaging_duration, window_start, window_end))

        # 记录阶段2的拒绝统计（无论是否全部过滤）
        self._rejection_stats['phase2_attitude_roll'] += phase2_attitude_roll
        self._rejection_stats['phase2_attitude_pitch'] += phase2_attitude_pitch
        self._rejection_stats['phase2_attitude_missing'] += phase2_attitude_missing

        # 注意：如果姿态预缓存过滤掉了所有候选，不要回退到原始候选
        # 姿态约束是硬性约束，不能绕过
        if not filtered_candidates:
            logger.debug(f"  Phase 3: All {len(candidates)} candidates filtered out by attitude constraints")
            if phase2_attitude_roll > 0 or phase2_attitude_pitch > 0 or phase2_attitude_missing > 0:
                self._last_task_rejection_reason = 'attitude_constraint'
            # 保持 filtered_candidates 为空，不要回退到原始候选

        self._perf_end('phase2_and_phase3_precompute_and_filter', t)
        logger.debug(f"  Phase 3: {len(filtered_candidates)} candidates after attitude filtering")

        # ========== 优化：限制Phase4候选数量 ==========
        # 限制进入精确约束检查的候选数量，减少批量检查开销
        MAX_PHASE4_CANDIDATES = 50
        if len(filtered_candidates) > MAX_PHASE4_CANDIDATES:
            # 按窗口开始时间排序，选择最早开始的候选（通常更有可能被调度）
            filtered_candidates = sorted(
                filtered_candidates,
                key=lambda x: x[4]  # window_start
            )[:MAX_PHASE4_CANDIDATES]
            logger.debug(f"  Phase 3.5: Limited to {MAX_PHASE4_CANDIDATES} candidates for Phase4")

        # ========== 阶段4：精确约束检查（批量优化版本）==========
        t = self._perf_start()
        best_assignment = None
        best_score = None
        constraint_check_count = 0

        # 检查是否可以使用批量姿态约束检查
        use_batch_slew = (
            isinstance(self._slew_checker, BatchSlewConstraintChecker) and
            len(filtered_candidates) > 1  # 只有多个候选时才使用批量
        )
        logger.debug(f"  Phase 4: Using batch slew check: {use_batch_slew}, candidates: {len(filtered_candidates)}")

        # 初始化阶段3和阶段4拒绝计数
        phase3_slew_infeasible = 0
        phase4_saa = 0
        phase4_time_conflict = 0
        phase4_resource = 0
        phase4_other = 0

        if use_batch_slew:
            # ===== 批量姿态约束检查 =====
            batch_candidates = []
            for sat, window, imaging_mode, imaging_duration, window_start, window_end in filtered_candidates:
                usage = self._sat_resource_usage.get(sat.id, {})
                last_task_end = usage.get('last_task_end', self.mission.start_time)

                # 获取卫星位置
                sat_position, sat_velocity = self._get_satellite_position(sat, last_task_end)

                # 离散采样选择最优 imaging_begin
                imaging_begin = self._select_imaging_begin_by_sampling(
                    satellite=sat,
                    target=task,
                    window_start=window_start,
                    window_end=window_end,
                    prev_end_time=last_task_end,
                    imaging_duration=imaging_duration,
                    step_seconds=1  # 1秒步长
                )

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
                    sat_velocity=sat_velocity,
                    imaging_begin=imaging_begin  # 使用离散采样选择的时间
                )
                batch_candidates.append(candidate)

            # 执行批量姿态约束检查
            t_batch = self._perf_start()
            batch_results = self._slew_checker.check_slew_feasibility_batch(batch_candidates)
            self._perf_end('batch_slew_check', t_batch)

            # 筛选通过姿态约束的候选
            slew_feasible_candidates = []
            phase3_slew_infeasible = 0
            for candidate, slew_result in zip(filtered_candidates, batch_results):
                if slew_result.feasible:
                    slew_feasible_candidates.append((candidate, slew_result))
                else:
                    phase3_slew_infeasible += 1

            constraint_check_count = len(batch_candidates)
        else:
            # ===== 回退到逐个检查 =====
            slew_feasible_candidates = []
            for candidate_data in filtered_candidates:
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
                constraint_check_count += 1

                if slew_result.feasible:
                    slew_feasible_candidates.append((candidate_data, slew_result))

        logger.debug(f"    Entering unified batch constraint check with {len(slew_feasible_candidates)} candidates")

        # ===== 批量检查其他约束（SAA、时间冲突、资源）=====
        if slew_feasible_candidates:
            # 准备批量检查数据
            unified_candidates = []
            candidate_data_map = []  # 保存原始候选数据与 slew_result 的映射

            for candidate_data, slew_result in slew_feasible_candidates:
                sat, window, imaging_mode, imaging_duration, window_start, window_end = candidate_data
                usage = self._sat_resource_usage.get(sat.id, {})
                scheduled_tasks = usage.get('scheduled_tasks', [])
                actual_start = slew_result.actual_start
                actual_end = actual_start + timedelta(seconds=imaging_duration)

                # 计算资源需求
                power_needed = 0.0
                storage_produced = 0.0
                if self._constraint_checker is not None and imaging_mode is not None:
                    try:
                        power_profile = imaging_mode.get_power_profile(imaging_duration)
                        power_needed = power_profile.total_energy if hasattr(power_profile, 'total_energy') else 0.0
                        storage_produced = getattr(imaging_mode, 'data_rate', 0.0) * imaging_duration
                    except:
                        pass

                # 获取卫星位置
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
                candidate_data_map.append((candidate_data, slew_result))

            # 获取已调度任务列表
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

            # 获取卫星资源状态
            satellite_states = {}
            for sat in self.mission.satellites:
                usage = self._sat_resource_usage.get(sat.id, {})
                satellite_states[sat.id] = {
                    'power': usage.get('power', sat.capabilities.power_capacity),
                    'storage': usage.get('storage', 0.0),
                    'power_capacity': sat.capabilities.power_capacity,
                    'storage_capacity': sat.capabilities.storage_capacity
                }

            # 延迟初始化统一批量约束检查器（只创建一次）
            t_unified = self._perf_start()
            if self._unified_checker is None:
                self._unified_checker = UnifiedBatchConstraintChecker(
                    mission=self.mission,
                    use_precise_model=True,
                    consider_power=self.consider_power,
                    consider_storage=self.consider_storage
                )

            # 执行批量快速阶段检查（姿态已检查，检查SAA、时间、资源）
            unified_results = self._unified_checker.check_fast_phase_batch(
                candidates=unified_candidates,
                existing_tasks=existing_tasks,
                satellite_states=satellite_states,
                early_termination=True
            )
            self._perf_end('unified_batch_constraint_check', t_unified)

            # 筛选通过所有约束的候选（姿态、SAA、时间、资源已全部批量检查完成）
            for i, (unified_result, (candidate_data, slew_result)) in enumerate(zip(unified_results, candidate_data_map)):
                if not unified_result.feasible:
                    # 记录具体的拒绝原因
                    reason = getattr(unified_result, 'reason', '') or ''
                    if 'SAA' in reason or 'saa' in reason.lower():
                        phase4_saa += 1
                    elif 'time' in reason.lower() or 'conflict' in reason.lower():
                        phase4_time_conflict += 1
                    elif 'power' in reason.lower() or 'storage' in reason.lower() or 'resource' in reason.lower():
                        phase4_resource += 1
                    else:
                        phase4_other += 1
                    continue

                sat, window, imaging_mode, imaging_duration, window_start, window_end = candidate_data
                actual_start = slew_result.actual_start

                # 资源约束已在批量检查中完成，无需逐个检查
                # unified_result.resource_result 包含批量资源检查结果

                # Calculate score for this assignment
                score = self._calculate_assignment_score(
                    sat, task, window, imaging_mode, actual_start
                )

                # Update best assignment if this is better
                if best_score is None or score > best_score:
                    best_score = score
                    best_assignment = (sat.id, window, imaging_mode, slew_result)

            # 记录阶段4的拒绝统计
            self._rejection_stats['phase3_slew_infeasible'] += phase3_slew_infeasible
            self._rejection_stats['phase4_saa'] += phase4_saa
            self._rejection_stats['phase4_time_conflict'] += phase4_time_conflict
            self._rejection_stats['phase4_resource'] += phase4_resource
            self._rejection_stats['phase4_other'] += phase4_other

        # 如果没有找到最佳分配，记录最后一个阶段的拒绝原因
        if best_assignment is None:
            if phase3_slew_infeasible > 0:
                self._last_task_rejection_reason = 'slew_infeasible'
            elif phase4_saa > 0:
                self._last_task_rejection_reason = 'saa_constraint'
            elif phase4_time_conflict > 0:
                self._last_task_rejection_reason = 'time_conflict'
            elif phase4_resource > 0:
                self._last_task_rejection_reason = 'resource_constraint'
            elif phase4_other > 0:
                self._last_task_rejection_reason = 'other_constraint'

        self._perf_end('phase4_precise_constraint_check', t)
        self._perf_stats['constraint_check_count_per_task']['count'] += 1
        self._perf_stats['constraint_check_count_per_task']['total_time'] += constraint_check_count
        self._perf_end('_find_best_assignment_total', t_total)
        return best_assignment

    def _can_satellite_perform_task(self, sat: Any, task: Any) -> bool:
        """
        Check if satellite has capability to perform the task

        Args:
            sat: Satellite to check
            task: Task to perform

        Returns:
            True if satellite can perform the task
        """
        # Check if satellite has any imaging modes
        if not sat.capabilities.imaging_modes:
            return False

        # Check resolution requirement
        required_resolution = getattr(task, 'resolution_required', None)
        if required_resolution is not None:
            # Use the correct method: check all imaging mode resolutions
            if not sat.capabilities.can_satisfy_resolution(required_resolution):
                return False

        return True

    def _get_windows(self, sat: Any, task: Any) -> List[Any]:
        """
        Get visibility windows for satellite-task pair

        Args:
            sat: Satellite
            task: Target task (ObservationTask or Target)

        Returns:
            List of visibility windows (filtered by Java precomputed attitude feasibility)
        """
        if self.window_cache:
            target_id = self._get_target_id(task)
            windows = self.window_cache.get_windows(sat.id, target_id)
            # Filter out windows that Java marked as attitude infeasible
            return [w for w in windows if getattr(w, 'attitude_feasible', True)]
        return []

    def _extract_window_times(self, window: Any) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Extract start and end times from a window object

        Args:
            window: Window object (dict or object)

        Returns:
            Tuple of (start_time, end_time)
        """
        if isinstance(window, dict):
            return window.get('start'), window.get('end')
        else:
            return getattr(window, 'start_time', None), getattr(window, 'end_time', None)

    def _select_imaging_begin_by_sampling(
        self,
        satellite: Any,
        target: Any,
        window_start: datetime,
        window_end: datetime,
        prev_end_time: datetime,
        imaging_duration: float,
        step_seconds: int = 1
    ) -> datetime:
        """
        通过离散采样选择最优成像开始时间 (imaging_begin)

        策略: FIFO - 选择第一个满足约束的采样点
        步长: 默认 1 秒

        约束条件:
        1. imaging_begin 必须在可见性窗口内
        2. imaging_begin + imaging_duration 必须 <= window_end
        3. (imaging_begin - prev_end_time) 必须足够完成机动+稳定

        Args:
            satellite: 卫星对象
            target: 目标对象
            window_start: 可见性窗口开始时间
            window_end: 可见性窗口结束时间
            prev_end_time: 前一任务结束时间
            imaging_duration: 成像持续时间(秒)
            step_seconds: 采样步长(秒), 默认 1 秒

        Returns:
            选择的 imaging_begin (datetime)
            如果没有找到满足约束的时刻, 返回 window_start (默认值)
        """
        # 计算最晚可能的成像开始时间
        latest_imaging_begin = window_end - timedelta(seconds=imaging_duration)

        if latest_imaging_begin < window_start:
            # 窗口太短, 无法完成成像
            return window_start

        # 离散采样 - 以 step_seconds 为步长
        current_time = window_start
        while current_time <= latest_imaging_begin:
            # 检查此采样点是否满足约束
            time_interval = (current_time - prev_end_time).total_seconds()

            # 获取此采样点的姿态角
            attitude = None
            if hasattr(self, '_attitude_precache_manager') and self._attitude_precache_manager is not None:
                attitude = self._attitude_precache_manager.get_attitude(
                    satellite.id, current_time
                )

            # 如果缓存未命中，实时计算姿态
            if attitude is None and self._attitude_calculator is not None:
                attitude = self._calculate_attitude_at_time(
                    satellite, target, current_time
                )

            # 检查姿态约束
            if attitude is not None:
                roll, pitch = attitude
                max_roll = getattr(satellite.capabilities, 'max_roll_angle', 45.0)
                max_pitch = getattr(satellite.capabilities, 'max_pitch_angle', 20.0)

                if abs(roll) <= max_roll and abs(pitch) <= max_pitch:
                    # FIFO 策略: 第一个满足姿态约束的采样点
                    return current_time

            # 下一个采样点
            current_time += timedelta(seconds=step_seconds)

        # 如果没有找到满足约束的时刻, 默认使用 window_start
        return window_start

    def _calculate_attitude_at_time(self, satellite, target, time):
        """在指定时间计算姿态角（实时计算）

        Args:
            satellite: 卫星对象
            target: 目标对象
            time: 计算时间

        Returns:
            (roll, pitch) 或 None
        """
        try:
            from core.dynamics.attitude_calculator import AttitudeCalculator

            # 获取卫星位置
            sat_position = None
            if hasattr(self, '_orbit_cache') and self._orbit_cache is not None:
                sat_position = self._orbit_cache.get_satellite_position(satellite.id, time)

            if sat_position is None:
                return None

            # 计算姿态
            target_position = (target.latitude, target.longitude, getattr(target, 'altitude', 0.0))
            roll, pitch = AttitudeCalculator.calculate_required_attitude(
                sat_position, target_position
            )

            return (roll, pitch)
        except Exception:
            return None

    def _is_window_within_target_constraints(
        self, task: Any, window_start: datetime, window_end: datetime
    ) -> bool:
        """
        Check if window is within target's time window constraints

        Args:
            task: Target task
            window_start: Window start time
            window_end: Window end time

        Returns:
            True if window satisfies target constraints
        """
        # Check target's time window start
        if task.time_window_start is not None:
            if window_end < task.time_window_start:
                return False

        # Check target's time window end
        if task.time_window_end is not None:
            if window_start > task.time_window_end:
                return False

        return True

    def _calculate_imaging_time(self, task: Any, imaging_mode: Any, sat: Any = None) -> float:
        """
        Calculate imaging time for a task

        Args:
            task: Target task
            imaging_mode: Imaging mode to use
            sat: Optional satellite for satellite-specific constraints

        Returns:
            Imaging duration in seconds
        """
        return self._imaging_calculator.calculate(task, imaging_mode, satellite=sat)

    def _select_imaging_mode(self, sat: Any, task: Any) -> Any:
        """
        Select appropriate imaging mode for satellite-task pair

        Args:
            sat: Satellite
            task: Target task

        Returns:
            Selected imaging mode
        """
        from core.models import ImagingMode

        modes = sat.capabilities.imaging_modes
        if not modes:
            return ImagingMode.PUSH_BROOM

        # Select first available mode
        mode = modes[0]
        # Handle Mock objects in tests (Mock objects have special attributes)
        if hasattr(mode, '_mock_name') or not isinstance(mode, (ImagingMode, str)):
            return ImagingMode.PUSH_BROOM
        return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)

    def _check_resource_constraints(self, sat: Any, task: Any, imaging_mode: Any) -> bool:
        """
        Check if satellite has sufficient resources for the task

        Args:
            sat: Satellite
            task: Target task
            imaging_mode: Imaging mode to use

        Returns:
            True if resources are sufficient
        """
        usage = self._sat_resource_usage.get(sat.id, {})

        # Check power constraint
        if self.consider_power:
            duration = self._calculate_imaging_time(task, imaging_mode, sat)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_needed = sat.capabilities.power_capacity * power_coefficient * (duration / 3600)

            if usage.get('power', 0) < power_needed:
                return False

        # Check storage constraint - 动态计算基于成像时长
        if self.consider_storage:
            # 获取卫星数据率，默认300 Mbps
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            # 动态计算固存消耗
            storage_needed = self._imaging_calculator.get_storage_consumption(
                task, imaging_mode, data_rate
            )
            current_storage = usage.get('storage', 0)
            capacity = sat.capabilities.storage_capacity

            if current_storage + storage_needed > capacity:
                return False

        return True

    def _calculate_task_time(
        self, sat_id: str, window_start: datetime, imaging_duration: float
    ) -> Tuple[datetime, datetime]:
        """
        Calculate actual start and end times for a task

        Args:
            sat_id: Satellite ID
            window_start: Visibility window start
            imaging_duration: Required imaging duration in seconds

        Returns:
            Tuple of (actual_start, actual_end)
        """
        usage = self._sat_resource_usage.get(sat_id, {})
        last_task_end = usage.get('last_task_end', self.mission.start_time)

        # Start after last task + slew time
        earliest_start = last_task_end + self.DEFAULT_SLEW_TIME

        # Actual start is the later of window start or earliest available time
        actual_start = max(window_start, earliest_start)
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        return actual_start, actual_end

    def _has_time_conflict(self, sat_id: str, start: datetime, end: datetime) -> bool:
        """
        Check if task time conflicts with existing scheduled tasks

        Args:
            sat_id: Satellite ID
            start: Task start time
            end: Task end time

        Returns:
            True if there is a conflict
        """
        usage = self._sat_resource_usage.get(sat_id, {})
        scheduled_tasks = usage.get('scheduled_tasks', [])

        for task in scheduled_tasks:
            # Check overlap
            if not (end <= task['start'] or start >= task['end']):
                return True

        return False

    def _calculate_assignment_score(
        self, sat: Any, task: Any, window: Any, imaging_mode: Any, actual_start: datetime
    ) -> float:
        """
        Calculate a score for an assignment (higher is better)

        Args:
            sat: Satellite
            task: Target task
            window: Visibility window
            imaging_mode: Selected imaging mode
            actual_start: Actual scheduled start time

        Returns:
            Score value (higher is better)
        """
        score = 0.0

        # Prefer earlier start times
        time_from_start = (actual_start - self.mission.start_time).total_seconds()
        score -= time_from_start / 3600.0  # Penalty for later starts

        # Prefer higher priority satellites (if applicable)
        score += getattr(task, 'priority', 5) * 10

        # Prefer satellites with more remaining resources
        usage = self._sat_resource_usage.get(sat.id, {})
        power_capacity = sat.capabilities.power_capacity
        storage_capacity = sat.capabilities.storage_capacity

        # Guard against division by zero
        if power_capacity > 0:
            power_ratio = usage.get('power', 0) / power_capacity
        else:
            power_ratio = 0.0

        if storage_capacity > 0:
            storage_ratio = 1.0 - (usage.get('storage', 0) / storage_capacity)
        else:
            storage_ratio = 0.0

        score += (power_ratio + storage_ratio) * 5

        # 窗口质量评分（新增）
        try:
            # 直接使用窗口的质量评分（VisibilityWindow总是有quality_score，默认1.0）
            # 如果窗口是新生成的没有详细评分，使用evaluate_window_quality进行计算（带缓存）
            window_quality = window.quality_score
            if window_quality == 1.0 and not window.detailed_scores:
                # 可能是默认评分，使用质量计算器获得更准确的评分
                window_quality = self.calculate_quality_score_for_assignment(
                    window, sat, task, self.mission
                )

            # 将质量评分加入总分（高质量获得加分）
            score += window_quality * self.quality_score_weight

        except Exception as e:
            logger.debug(f"Failed to calculate quality score: {e}")
            # 如果质量评分失败，不影响其他评分

        return score

    def _get_previous_task_target(self, sat_id: str) -> Optional[Any]:
        """获取卫星上一个已调度任务的目标

        Args:
            sat_id: 卫星ID

        Returns:
            前一个任务的目标对象，如果没有则返回None
        """
        usage = self._sat_resource_usage.get(sat_id, {})
        scheduled_tasks = usage.get('scheduled_tasks', [])

        if not scheduled_tasks:
            return None

        # 获取最后一个已调度任务
        last_task_info = scheduled_tasks[-1]
        # 优先使用 target_id，如果不存在则回退到 task_id
        prev_target_id = last_task_info.get('target_id') or last_task_info.get('task_id')

        if not prev_target_id or not self.mission:
            return None

        # 从mission中找到对应的目标
        return self.mission.get_target_by_id(prev_target_id)

    def _create_scheduled_task(
        self, task: Any, sat_id: str, window: Any, imaging_mode: Any,
        slew_result: Optional[SlewFeasibilityResult] = None
    ) -> ScheduledTask:
        """
        Create a ScheduledTask object

        Args:
            task: Target task
            sat_id: Satellite ID
            window: Visibility window
            imaging_mode: Imaging mode
            slew_result: 机动可行性检查结果（可选）

        Returns:
            ScheduledTask object
        """
        window_start, window_end = self._extract_window_times(window)

        # Get satellite for satellite-specific constraints
        sat = self.mission.get_satellite_by_id(sat_id)
        imaging_duration = self._calculate_imaging_time(task, imaging_mode, sat)

        # Get current resource levels
        usage = self._sat_resource_usage.get(sat_id, {})
        power_before = usage.get('power', 0)
        storage_before = usage.get('storage', 0)

        # Calculate resource consumption
        power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
        sat = self.mission.get_satellite_by_id(sat_id)
        power_consumed = 0.0
        if sat and self.consider_power:
            power_consumed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)

        # 动态计算固存消耗
        storage_used = 0.0
        if sat and self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            # 处理聚类任务的存储消耗（累加所有目标）
            if isinstance(task, ClusterTask):
                for target in task.targets:
                    storage_used += self._imaging_calculator.get_storage_consumption(
                        target, imaging_mode, data_rate
                    )
            else:
                storage_used = self._imaging_calculator.get_storage_consumption(
                    task, imaging_mode, data_rate
                )

        # 获取前一任务目标（用于后续计算）
        prev_target = self._get_previous_task_target(sat_id)

        # 使用 slew_result 中的机动信息（如果提供），否则重新计算
        reset_time = None
        if slew_result:
            slew_angle = slew_result.slew_angle
            slew_time_seconds = slew_result.slew_time
            actual_start = slew_result.actual_start
            actual_end = actual_start + timedelta(seconds=imaging_duration)
            reset_time = getattr(slew_result, 'reset_time', None)

            # 如果 reset_time 为 None，说明是快速筛选阶段，需要补充计算复位时间
            if reset_time is None and sat and self._slew_checker is not None:
                reset_time, conflict_resolved = self._calculate_reset_time_and_resolve_conflict(
                    sat_id=sat_id,
                    prev_target=prev_target,
                    window_start=window_start,
                    window_end=window_end,
                    current_slew_time=slew_time_seconds,
                    imaging_duration=imaging_duration
                )
                if not conflict_resolved:
                    # 冲突无法消解，标记为不可行（这里可能需要特殊处理）
                    logger.warning(f"Task {task.id} on {sat_id}: 姿态复位时间冲突无法消解")
                else:
                    # 更新实际开始时间（包含复位时间）
                    total_slew = slew_time_seconds + reset_time
                    usage = self._sat_resource_usage.get(sat_id, {})
                    last_task_end = usage.get('last_task_end')
                    if last_task_end:
                        earliest_start = last_task_end + timedelta(seconds=total_slew)
                        actual_start = max(window_start, earliest_start)
                        actual_end = actual_start + timedelta(seconds=imaging_duration)
        else:
            # 计算动态机动时间和角度（回退逻辑 - 简化估算）
            slew_angle = 0.0
            slew_time_seconds = self.DEFAULT_SLEW_TIME.total_seconds()

            if sat and hasattr(task, 'latitude') and hasattr(task, 'longitude'):
                prev_target = self._get_previous_task_target(sat_id)
                if prev_target and hasattr(prev_target, 'latitude') and hasattr(prev_target, 'longitude'):
                    # 简化估算：使用经纬度差估算机动角度
                    lon_diff = task.longitude - prev_target.longitude
                    lat_diff = task.latitude - prev_target.latitude
                    slew_angle = math.sqrt(lon_diff**2 + lat_diff**2)
                    slew_angle = min(slew_angle, sat.capabilities.max_roll_angle)
                    # 简化时间计算：角度 / 最大角速度 + 稳定时间
                    agility = getattr(sat.capabilities, 'agility', {}) or {}
                    max_slew_rate = agility.get('max_slew_rate', 3.0)
                    settling_time = agility.get('settling_time', 5.0)
                    slew_time_seconds = slew_angle / max_slew_rate + settling_time

            # 计算实际开始时间
            usage = self._sat_resource_usage.get(sat_id, {})
            last_task_end = usage.get('last_task_end')
            if last_task_end:
                earliest_start = last_task_end + timedelta(seconds=slew_time_seconds)
                actual_start = max(window_start, earliest_start)
            else:
                actual_start = window_start
            actual_end = actual_start + timedelta(seconds=imaging_duration)

        # 获取任务优先级
        task_priority = getattr(task, 'priority', None)

        # 计算机动能量消耗（从slew_result获取，如果可用）
        energy_consumption = 0.0
        if slew_result:
            energy_consumption = getattr(slew_result, 'energy_consumption', 0.0)

        # 计算电池SOC百分比
        battery_soc_before = 0.0
        battery_soc_after = 0.0
        if sat and hasattr(sat.capabilities, 'power_capacity') and sat.capabilities.power_capacity > 0:
            battery_soc_before = (power_before / sat.capabilities.power_capacity) * 100.0
            battery_soc_after = ((power_before - power_consumed) / sat.capabilities.power_capacity) * 100.0

        # 计算姿态角（用于发电量计算和约束检查）
        roll_angle = 0.0
        pitch_angle = 0.0
        if self._enable_attitude_calculation and sat and hasattr(task, 'latitude') and hasattr(task, 'longitude'):
            # 使用窗口开始时间查询姿态预缓存（姿态约束在候选过滤阶段已检查）
            window_start_attitude = None
            if hasattr(self, '_attitude_precache_manager') and self._attitude_precache_manager is not None:
                window_start_attitude = self._attitude_precache_manager.get_attitude(sat_id, window_start)
                if window_start_attitude is None:
                    logger.info(f"[Attitude] Cache miss for {sat_id} @ {window_start}, fallback to realtime calculation")
                else:
                    logger.info(f"[Attitude] Cache hit for {sat_id} @ {window_start}: roll={window_start_attitude[0]:.2f}, pitch={window_start_attitude[1]:.2f}")

            if window_start_attitude is not None:
                roll_angle, pitch_angle = window_start_attitude
            else:
                # 回退到实时计算
                logger.info(f"[Attitude] Using realtime calculation for {sat_id}")
                attitude = self._calculate_attitude_angles(sat, task, actual_start)
                if attitude:
                    roll_angle = attitude.roll
                    pitch_angle = attitude.pitch
                    logger.info(f"[Attitude] Realtime result: roll={roll_angle:.2f}, pitch={pitch_angle:.2f}")

        # 计算任务期间发电量
        power_generated = 0.0
        if self._enable_power_generation_calc:
            power_generated = self._calculate_power_generation(
                sat_id=sat_id,
                start_time=actual_start,
                end_time=actual_end,
                roll_angle=roll_angle,
                pitch_angle=pitch_angle
            )

        # 计算成像足迹（新增）
        footprint_corners = []
        footprint_center = None
        swath_width_km = 0.0
        fov_config = {}
        if sat:
            try:
                footprint_result = self._calculate_task_footprint(
                    sat_id=sat_id,
                    task=task,
                    imaging_start=actual_start,
                    roll_angle=roll_angle,
                    pitch_angle=pitch_angle,
                    imaging_mode=imaging_mode
                )
                footprint_corners = footprint_result.get('corners', [])
                footprint_center = footprint_result.get('center')
                swath_width_km = footprint_result.get('swath_width_km', 0.0)
                fov_config = footprint_result.get('fov_config', {})
            except Exception as e:
                logger.warning(f"Failed to calculate footprint for task {task.id}: {e}")

        # 创建ScheduledTask对象
        scheduled_task = ScheduledTask(
            task_id=task.id,
            satellite_id=sat_id,
            target_id=task.id,
            imaging_start=actual_start,
            imaging_end=actual_end,
            imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
            slew_angle=slew_angle,
            slew_time=slew_time_seconds,
            storage_before=storage_before,
            storage_after=storage_before + storage_used,
            power_before=power_before,
            power_after=power_before - power_consumed,
            # 详细能源变化字段
            power_consumed=power_consumed,
            power_generated=power_generated,  # 使用计算的发电量
            energy_consumption=energy_consumption,
            battery_soc_before=battery_soc_before,
            battery_soc_after=battery_soc_after,
            priority=task_priority,
            reset_time=reset_time,
            # 姿态角字段
            roll_angle=roll_angle,
            pitch_angle=pitch_angle,
            # 成像足迹字段
            footprint_corners=footprint_corners,
            footprint_center=footprint_center,
            swath_width_km=swath_width_km,
            fov_config=fov_config,
        )

        # 如果是聚类任务，记录聚类调度信息
        if isinstance(task, ClusterTask) and self.enable_clustering:
            self._record_cluster_schedule(
                task=task,
                satellite_id=sat_id,
                imaging_start=actual_start,
                imaging_end=actual_end,
                look_angle=slew_angle
            )

        # 填充聚类信息到 ScheduledTask
        self._populate_cluster_info(scheduled_task, task)

        return scheduled_task

    def _update_resource_usage(
        self, sat_id: str, task: Any, window: Any, scheduled_task: ScheduledTask
    ) -> None:
        """
        Update resource usage tracking after scheduling a task

        Args:
            sat_id: Satellite ID
            task: Target task
            window: Visibility window
            scheduled_task: Created scheduled task
        """
        usage = self._sat_resource_usage.get(sat_id)
        if usage is None:
            return

        # Update power
        if self.consider_power:
            usage['power'] = scheduled_task.power_after

        # Update storage
        if self.consider_storage:
            usage['storage'] = scheduled_task.storage_after

        # Update last task end time
        usage['last_task_end'] = scheduled_task.imaging_end

        # Track scheduled task for conflict detection
        if 'scheduled_tasks' not in usage:
            usage['scheduled_tasks'] = []
        # Use target_id if available (for ObservationTask), otherwise use task.id
        target_id = getattr(task, 'target_id', task.id)
        usage['scheduled_tasks'].append({
            'start': scheduled_task.imaging_start,
            'end': scheduled_task.imaging_end,
            'task_id': task.id,
            'target_id': target_id
        })

        # Update attitude state after task (set to IMAGING mode)
        if self._enable_attitude_management:
            from core.dynamics.attitude_mode import AttitudeMode
            self._set_satellite_attitude_mode(sat_id, AttitudeMode.IMAGING)

    def _calculate_task_footprint(
        self,
        sat_id: str,
        task: Any,
        imaging_start: datetime,
        roll_angle: float,
        pitch_angle: float,
        imaging_mode: Any
    ) -> Dict[str, Any]:
        """
        计算任务成像足迹

        Args:
            sat_id: 卫星ID
            task: 目标任务
            imaging_start: 成像开始时间
            roll_angle: 滚转角（度）
            pitch_angle: 俯仰角（度）
            imaging_mode: 成像模式

        Returns:
            Dict: 包含footprint信息的字典
                - corners: 四脚点坐标列表 [(lon, lat), ...]
                - center: 中心点坐标 (lon, lat)
                - swath_width_km: 幅宽（公里）
                - fov_config: FOV配置
        """
        from core.coverage.footprint_calculator import FootprintCalculator
        from core.models import ImagingMode

        sat = self.mission.get_satellite_by_id(sat_id)
        if not sat:
            return {'corners': [], 'center': None, 'swath_width_km': 0.0, 'fov_config': {}}

        # 获取卫星位置和星下点
        try:
            position, velocity = self._attitude_calculator._get_satellite_state(
                sat, imaging_start
            )
        except Exception:
            return {'corners': [], 'center': None, 'swath_width_km': 0.0, 'fov_config': {}}

        # 计算星下点
        sat_radius = math.sqrt(sum(x**2 for x in position))
        altitude_km = (sat_radius - 6371000) / 1000.0

        # 获取FOV配置（优先从imager配置）
        imager_config = sat.capabilities.imager if hasattr(sat.capabilities, 'imager') else {}
        fov_config = imager_config.get('fov_config', {}) if isinstance(imager_config, dict) else {}

        # 如果没有FOV配置，基于swath_width构建
        if not fov_config and hasattr(sat.capabilities, 'swath_width'):
            swath_m = sat.capabilities.swath_width
            fov_config = {
                'fov_type': 'cone',
                'half_angle': math.degrees(math.atan2(swath_m / 2000, altitude_km)),
                'swath_width_km': swath_m / 1000.0
            }

        # 计算星下点经纬度（简化计算）
        # 使用姿态计算器的方法
        try:
            nadir = self._attitude_calculator._calculate_nadir(position)
            nadir_lon, nadir_lat = nadir.longitude, nadir.latitude
        except Exception:
            # 简化计算
            x, y, z = position
            r = math.sqrt(x*x + y*y + z*z)
            lat = math.degrees(math.asin(z / r))
            lon = math.degrees(math.atan2(y, x))
            nadir_lon, nadir_lat = lon, lat

        # 计算观测角度和方向
        look_angle = math.sqrt(roll_angle**2 + pitch_angle**2)
        # 观测方向：从姿态角计算（0=北，90=东）
        look_direction = math.degrees(math.atan2(roll_angle, pitch_angle))

        # 计算footprint
        calculator = FootprintCalculator(satellite_altitude_km=altitude_km)

        # 判断使用哪种计算方法
        if fov_config:
            footprint = calculator.calculate_footprint_from_fov(
                satellite_position=position,
                nadir_position=(nadir_lon, nadir_lat),
                look_angle=look_angle,
                look_direction=look_direction,
                fov_config=fov_config,
                imaging_mode=imaging_mode if isinstance(imaging_mode, ImagingMode) else ImagingMode.PUSH_BROOM
            )
        else:
            # 回退到原始方法
            swath_width_km = getattr(sat.capabilities, 'swath_width', 10000) / 1000.0
            footprint = calculator.calculate_footprint(
                satellite_position=position,
                nadir_position=(nadir_lon, nadir_lat),
                look_angle=look_angle,
                swath_width_km=swath_width_km,
                imaging_mode=imaging_mode if isinstance(imaging_mode, ImagingMode) else ImagingMode.PUSH_BROOM
            )

        return {
            'corners': footprint.polygon,
            'center': footprint.center,
            'swath_width_km': footprint.width_km,
            'fov_config': fov_config or footprint.fov_config or {}
        }

    def _determine_failure_reason(self, task: Any) -> TaskFailureReason:
        """
        Determine why a task could not be scheduled

        Args:
            task: Task that failed to schedule

        Returns:
            TaskFailureReason enum value
        """
        # Check storage constraint - 使用动态固存消耗
        if self.consider_storage:
            for sat in self.mission.satellites:
                usage = self._sat_resource_usage.get(sat.id, {})
                imaging_mode = self._select_imaging_mode(sat, task)
                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                storage_needed = self._imaging_calculator.get_storage_consumption(
                    task, imaging_mode, data_rate
                )
                current_storage = usage.get('storage', 0)
                if current_storage + storage_needed > sat.capabilities.storage_capacity:
                    return TaskFailureReason.STORAGE_CONSTRAINT

        # Check power constraint
        if self.consider_power:
            for sat in self.mission.satellites:
                usage = self._sat_resource_usage.get(sat.id, {})
                power_ratio = usage.get('power', 0) / sat.capabilities.power_capacity
                if power_ratio < 0.1:  # Less than 10% power remaining
                    return TaskFailureReason.POWER_CONSTRAINT

        # Check visibility windows
        has_visible_window = False
        for sat in self.mission.satellites:
            windows = self._get_windows(sat, task)
            if windows:
                has_visible_window = True
                break

        if not has_visible_window:
            return TaskFailureReason.NO_VISIBLE_WINDOW

        # Check if windows were too short
        for sat in self.mission.satellites:
            windows = self._get_windows(sat, task)
            for window in windows:
                window_start, window_end = self._extract_window_times(window)
                if window_start and window_end:
                    window_duration = (window_end - window_start).total_seconds()
                    imaging_mode = self._select_imaging_mode(sat, task)
                    imaging_duration = self._calculate_imaging_time(task, imaging_mode, sat)
                    if window_duration < imaging_duration:
                        return TaskFailureReason.WINDOW_TOO_SHORT

        # Check time conflicts
        if self.consider_time_conflicts:
            return TaskFailureReason.TIME_CONFLICT

        # Check capability mismatch
        for sat in self.mission.satellites:
            if not self._can_satellite_perform_task(sat, task):
                return TaskFailureReason.SAT_CAPABILITY_MISMATCH

        return TaskFailureReason.UNKNOWN

    def _calculate_makespan(self, scheduled_tasks: List[ScheduledTask]) -> float:
        """
        Calculate makespan from scheduled tasks

        Args:
            scheduled_tasks: List of scheduled tasks

        Returns:
            Makespan in seconds
        """
        if not scheduled_tasks:
            return 0.0

        last_end = max(t.imaging_end for t in scheduled_tasks)
        return (last_end - self.mission.start_time).total_seconds()

    def _build_failure_summary(self) -> Dict[TaskFailureReason, int]:
        """
        Build summary of failure reasons

        Returns:
            Dictionary mapping failure reasons to counts
        """
        summary: Dict[TaskFailureReason, int] = {}
        for failure in self._failure_log:
            reason = failure.failure_reason
            summary[reason] = summary.get(reason, 0) + 1
        return summary

    def _calculate_target_obs_count(self, scheduled_tasks: List[ScheduledTask]) -> Dict[str, int]:
        """
        Calculate observation count per target

        Args:
            scheduled_tasks: List of scheduled tasks

        Returns:
            Dictionary mapping target_id to observation count
        """
        target_obs_count: Dict[str, int] = {}
        for task in scheduled_tasks:
            target_id = task.target_id
            target_obs_count[target_id] = target_obs_count.get(target_id, 0) + 1
        return target_obs_count

    def _calculate_reset_time_and_resolve_conflict(
        self,
        sat_id: str,
        prev_target: Optional[Any],
        window_start: datetime,
        window_end: datetime,
        current_slew_time: float,
        imaging_duration: float
    ) -> Tuple[float, bool]:
        """计算姿态复位时间并检查冲突

        在快速筛选阶段跳过了复位计算，这里补充计算并检查是否有时间冲突。

        Args:
            sat_id: 卫星ID
            prev_target: 前一任务目标（None表示第一个任务）
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            current_slew_time: 当前任务机动时间（秒）
            imaging_duration: 成像持续时间（秒）

        Returns:
            Tuple[reset_time, conflict_resolved]: 复位时间和是否成功消解冲突
        """
        # 如果没有前一任务或没有姿态检查器，不需要复位
        if prev_target is None or self._slew_checker is None:
            return 0.0, True

        # 获取卫星资源使用情况
        usage = self._sat_resource_usage.get(sat_id, {})
        last_task_end = usage.get('last_task_end')

        if last_task_end is None:
            return 0.0, True

        # 检查时间间隔，如果超过5分钟，假设已回到对地定向
        time_diff = (window_start - last_task_end).total_seconds()
        if time_diff > 300:  # 5分钟
            return 0.0, True

        try:
            # 使用精确姿态检查器计算复位时间
            # 获取前一任务结束时的姿态状态
            if hasattr(self._slew_checker, '_get_satellite_attitude_state'):
                from core.dynamics.precise import AttitudeState, Quaternion, AngularVelocity

                current_state = self._slew_checker._get_satellite_attitude_state(sat_id, last_task_end)

                # 对地定向姿态
                nadir_attitude = AttitudeState(
                    quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
                    angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
                    timestamp=last_task_end,
                    momentum=None
                )

                # 获取精确计算器
                if hasattr(self._slew_checker, '_precise_calcs') and sat_id in self._slew_checker._precise_calcs:
                    calc = self._slew_checker._precise_calcs[sat_id]

                    # 计算姿态复位机动
                    reset_maneuver = calc.calculate_slew_maneuver(
                        prev_attitude=current_state,
                        target_attitude=nadir_attitude,
                        current_time=last_task_end
                    )

                    if reset_maneuver.feasible:
                        reset_time = reset_maneuver.total_time

                        # 检查总时间是否有冲突
                        total_time_needed = reset_time + current_slew_time + imaging_duration
                        time_available = (window_end - last_task_end).total_seconds()

                        if total_time_needed <= time_available:
                            return reset_time, True
                        else:
                            # 冲突：时间不够
                            # 尝试消解：如果可以部分使用窗口时间
                            if current_slew_time + imaging_duration <= (window_end - window_start).total_seconds():
                                # 至少任务机动可以在窗口内完成，但可能开始时间会延迟
                                return reset_time, True
                            else:
                                return reset_time, False
                    else:
                        # 复位不可行，尝试不复位直接机动
                        return 0.0, True

            # 回退到简化估算
            # 估算复位角度（从前一目标回到对地定向的角度）
            if hasattr(prev_target, 'latitude') and hasattr(prev_target, 'longitude'):
                # 简化估算：假设复位角度约等于前一任务的侧摆角度
                reset_angle = math.sqrt(
                    prev_target.latitude**2 + prev_target.longitude**2
                ) * 0.5
                reset_angle = min(reset_angle, 45.0)  # 限制最大角度

                # 简化计算复位时间
                agility = getattr(self.mission.get_satellite_by_id(sat_id).capabilities, 'agility', {}) or {}
                max_slew_rate = agility.get('max_slew_rate', 3.0)
                settling_time = agility.get('settling_time', 5.0)
                reset_time = (reset_angle / max_slew_rate) + settling_time if max_slew_rate > 0 else settling_time

                # 检查冲突
                total_time_needed = reset_time + current_slew_time + imaging_duration
                time_available = (window_end - last_task_end).total_seconds()

                return reset_time, total_time_needed <= time_available

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"计算复位时间失败 {sat_id}: {e}")

        # 默认不复位
        return 0.0, True

    def _get_task_id(self, task: Any) -> str:
        """
        Get task ID from either ObservationTask or Target

        Args:
            task: Task object (ObservationTask or Target)

        Returns:
            Task ID string
        """
        if isinstance(task, ObservationTask):
            return task.task_id
        return getattr(task, 'id', '')

    def _get_target_id(self, task: Any) -> str:
        """
        Get target ID from either ObservationTask or Target

        Args:
            task: Task object (ObservationTask or Target)

        Returns:
            Target ID string
        """
        if isinstance(task, ObservationTask):
            return task.target_id
        return getattr(task, 'id', '')
