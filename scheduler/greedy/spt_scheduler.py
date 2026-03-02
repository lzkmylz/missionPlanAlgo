"""
SPT调度器（最短处理时间优先）

实现设计文档第8章设计：
- SPT（Shortest Processing Time）启发式调度
- 按处理时间排序任务
- 处理时间相同的按优先级排序
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from ..frequency_utils import ObservationTask
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile
from core.dynamics.slew_calculator import SlewCalculator
from ..constraints import SlewConstraintChecker, SlewFeasibilityResult


class SPTScheduler(BaseScheduler):
    """
    SPT（最短处理时间优先）调度器

    调度策略：
    1. 计算每个任务的预计处理时间
    2. 按处理时间升序排序（短的优先）
    3. 处理时间相同的，按优先级降序排序
    4. 依次尝试为每个任务分配最早的可用卫星-窗口组合

    特点：
    - 最小化平均流程时间（minimize average flow time）
    - 适合需要快速周转的任务场景
    """

    # 默认转移时间
    DEFAULT_SLEW_TIME = timedelta(seconds=30)

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化SPT调度器

        Args:
            config: 配置参数
                - consider_power: 是否考虑电量约束（默认True）
                - consider_storage: 是否考虑存储约束（默认True）
                - allow_tardiness: 是否允许延迟执行（默认False）
        """
        super().__init__("SPT", config)
        config = config or {}
        self.heuristic = "processing_time"
        self.consider_power = config.get('consider_power', True)
        self.consider_storage = config.get('consider_storage', True)
        self.allow_tardiness = config.get('allow_tardiness', False)

        # 初始化成像时间计算器和功率配置文件
        # 使用ImagingTimeCalculator的默认值（基于实际卫星数据）
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=config.get('min_imaging_duration'),
            max_duration=config.get('max_imaging_duration'),
            default_duration=config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(config.get('power_coefficients'))

        # Slew calculators per satellite (initialized in schedule())
        self._slew_calculators: Dict[str, SlewCalculator] = {}
        self._last_task_target: Dict[str, Any] = {}  # Track last scheduled target per satellite
        self._sat_resource_usage: Dict[str, Dict[str, Any]] = {}

    def get_parameters(self) -> Dict[str, Any]:
        """返回算法可调参数"""
        return {
            'heuristic': self.heuristic,
            'consider_power': self.consider_power,
            'consider_storage': self.consider_storage,
            'allow_tardiness': self.allow_tardiness,
        }

    def schedule(self) -> ScheduleResult:
        """
        执行SPT调度

        Returns:
            ScheduleResult: 调度结果
        """
        self._start_timer()

        self._validate_initialization()

        # 获取任务列表并按SPT规则排序（使用频次感知的任务创建）
        pending_tasks = self._sort_tasks_by_processing_time(self._create_frequency_aware_tasks())
        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, Any] = {}
        target_obs_count: Dict[str, int] = {}

        # Initialize resource tracking for each satellite
        self._sat_resource_usage = {
            sat.id: {
                'power': sat.current_power if hasattr(sat, 'current_power') else sat.capabilities.power_capacity,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': [],  # 跟踪已调度任务用于冲突检测
            }
            for sat in self.mission.satellites
        }
        # Initialize slew constraint checker (replaces individual SlewCalculator initialization)
        self._initialize_slew_checker()

        # Initialize SAA constraint checker
        self._initialize_saa_checker()

        # 预计算卫星位置以加速调度（仅在非简化模式下且明确启用时）
        if not self._use_simplified_slew and self.config.get('precompute_positions', False):
            print("    预计算卫星位置...")
            self._precompute_satellite_positions(time_step_minutes=self.config.get('precompute_step_minutes', 30))

        # Keep _slew_calculators for backward compatibility
        self._slew_calculators = {}
        self._last_task_target = {}
        for sat in self.mission.satellites:
            agility = getattr(sat.capabilities, 'agility', {})
            self._slew_calculators[sat.id] = SlewCalculator(
                max_slew_rate=agility.get('max_slew_rate', 3.0) if agility else 3.0,
                max_slew_angle=sat.capabilities.max_off_nadir,
                settling_time=agility.get('settling_time', 5.0) if agility else 5.0
            )

        # SPT调度主循环
        for task in pending_tasks:
            best_assignment = self._find_best_assignment(task)

            if best_assignment:
                sat_id, window, imaging_mode, slew_result = best_assignment

                # Create scheduled task with slew information
                scheduled_task = self._create_scheduled_task(
                    task, sat_id, window, imaging_mode, slew_result
                )
                scheduled_tasks.append(scheduled_task)

                # Update resource usage
                self._update_resource_usage(sat_id, task, window, scheduled_task)

                # Update last task target tracking
                self._last_task_target[sat_id] = task

                # 更新目标观测计数
                task_id = task.task_id if isinstance(task, ObservationTask) else task.id
                target_id = task.target_id if isinstance(task, ObservationTask) else task.id
                target_obs_count[target_id] = target_obs_count.get(target_id, 0) + 1

                self._add_convergence_point(len(scheduled_tasks))
            else:
                # 记录失败原因
                reason = self._determine_failure_reason(task)
                task_id = task.task_id if isinstance(task, ObservationTask) else task.id
                self._record_failure(
                    task_id=task_id,
                    reason=reason,
                    detail=f"No feasible assignment found for task {task_id}"
                )
                unscheduled[task_id] = self._failure_log[-1]

        # 计算makespan
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

    def _sort_tasks_by_processing_time(self, tasks: List[Any]) -> List[Any]:
        """
        按处理时间排序任务（SPT规则）

        排序规则：
        1. 按estimated_processing_time升序（短的优先）
        2. 处理时间相同的，按priority降序（优先级高的优先）

        Args:
            tasks: 任务列表

        Returns:
            List: 排序后的任务列表
        """
        def spt_key(task):
            # 估计处理时间（使用默认成像时长作为代理）
            processing_time = self._estimate_processing_time(task)

            # 优先级作为次键（越高越好，所以取负）
            priority = getattr(task, 'priority', 5)

            return (processing_time, -priority)

        return sorted(tasks, key=spt_key)

    def _estimate_processing_time(self, task: Any) -> float:
        """
        估计任务处理时间

        Args:
            task: 目标任务

        Returns:
            float: 估计处理时间（秒）
        """
        # 简化的处理时间估计：基于目标类型和成像模式
        from core.models import TargetType, ImagingMode

        if task.target_type == TargetType.POINT:
            # 点目标：使用默认框幅模式
            return self._imaging_calculator.calculate(task, ImagingMode.FRAME)
        else:
            # 区域目标：使用条带模式
            return self._imaging_calculator.calculate(task, ImagingMode.STRIPMAP)

    def _find_best_assignment(self, task: Any) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """
        为任务找到最佳卫星-窗口组合

        SPT策略：选择最早的可用窗口，使用SlewConstraintChecker检查机动约束

        Args:
            task: 目标任务

        Returns:
            Optional[Tuple]: (satellite_id, window, imaging_mode, slew_result) 或 None
        """
        best_assignment = None
        best_start = None

        for sat in self.mission.satellites:
            # 检查卫星能力匹配
            if not self._can_satellite_perform_task(sat, task):
                continue

            # 获取可见窗口
            windows = self._get_feasible_windows(sat, task)
            if not windows:
                continue

            # SPT策略：选择最早的窗口
            for window in windows:
                window_start = window['start'] if isinstance(window, dict) else window.start_time
                window_end = window['end'] if isinstance(window, dict) else window.end_time

                # 计算成像时长
                imaging_mode = self._select_imaging_mode(sat, task)
                imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)

                # 检查资源约束
                if not self._check_resource_constraints(sat, task):
                    continue

                # 检查 slew 约束（使用简化或精确计算）
                usage = self._sat_resource_usage.get(sat.id, {})
                last_task_end = usage.get('last_task_end', self.mission.start_time)

                if self._use_simplified_slew:
                    # 使用简化的机动检查（性能优化）
                    slew_result = self._get_slew_result_simple(sat.id, last_task_end, window_start)
                else:
                    # 使用精确的机动约束检查
                    prev_target = self._get_previous_task_target(sat.id)
                    slew_result = self._slew_checker.check_slew_feasibility(
                        sat.id, prev_target, task, last_task_end, window_start, imaging_duration
                    )

                if not slew_result.feasible:
                    continue

                # Check SAA constraints
                self._ensure_saa_checker_initialized()
                if self._saa_checker is not None:
                    saa_result = self._saa_checker.check_window_feasibility(
                        sat.id, window_start, window_end
                    )
                    if not saa_result.feasible:
                        continue

                # 使用实际开始时间从 slew 计算
                actual_start = slew_result.actual_start
                actual_end = actual_start + timedelta(seconds=imaging_duration)

                # 检查是否超出窗口结束时间
                if actual_end > window_end:
                    continue

                # 检查是否在截止时间之前
                if task.time_window_end and actual_start > task.time_window_end:
                    if not self.allow_tardiness:
                        continue

                # 检查时间冲突
                if self._has_time_conflict(sat.id, actual_start, actual_end):
                    continue

                # 选择最早的窗口
                if best_start is None or actual_start < best_start:
                    best_start = actual_start
                    best_assignment = (sat.id, window, imaging_mode, slew_result)

        return best_assignment

    def _has_time_conflict(self, sat_id: str, start: datetime, end: datetime) -> bool:
        """检查是否与已调度任务有时间冲突"""
        usage = self._sat_resource_usage.get(sat_id, {})
        scheduled_tasks = usage.get('scheduled_tasks', [])

        for task in scheduled_tasks:
            # 检查重叠
            if not (end <= task['start'] or start >= task['end']):
                return True
        return False

    def _get_feasible_windows(self, sat: Any, task: Any) -> List[Any]:
        """获取可行的时间窗口"""
        if self.window_cache:
            # 支持ObservationTask和原始Target
            target_id = task.target_id if isinstance(task, ObservationTask) else task.id
            return self.window_cache.get_windows(sat.id, target_id)
        return []

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
        prev_task_id = last_task_info.get('task_id')

        if not prev_task_id or not self.mission:
            return None

        # 从mission中找到对应的目标
        return self.mission.get_target_by_id(prev_task_id)

    def _can_satellite_perform_task(self, sat: Any, task: Any) -> bool:
        """检查卫星是否能执行任务"""
        if not sat.capabilities.imaging_modes:
            return False
        return True

    def _check_resource_constraints(self, sat: Any, task: Any) -> bool:
        """检查资源约束"""
        usage = self._sat_resource_usage.get(sat.id, {})

        # 电量约束
        if self.consider_power:
            imaging_mode = self._select_imaging_mode(sat, task)
            duration = self._imaging_calculator.calculate(task, imaging_mode)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_needed = sat.capabilities.power_capacity * power_coefficient * (duration / 3600)
            if usage.get('power', 0) < power_needed:
                return False

        # 存储约束 - 动态计算基于成像时长
        if self.consider_storage:
            imaging_mode = self._select_imaging_mode(sat, task)
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_needed = self._imaging_calculator.get_storage_consumption(
                task, imaging_mode, data_rate
            )
            if usage.get('storage', 0) + storage_needed > sat.capabilities.storage_capacity:
                return False

        return True

    def _select_imaging_mode(self, sat: Any, task: Any):
        """选择成像模式"""
        from core.models import ImagingMode

        modes = sat.capabilities.imaging_modes
        if not modes:
            return ImagingMode.PUSH_BROOM

        mode = modes[0]
        return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)

    def _update_resource_usage(
        self, sat_id: str, task: Any, window: Any, scheduled_task: ScheduledTask
    ) -> None:
        """更新资源使用状态

        Args:
            sat_id: 卫星ID
            task: 目标任务
            window: 可见窗口
            scheduled_task: 已调度的任务对象
        """
        usage = self._sat_resource_usage.get(sat_id)
        if usage is None:
            return

        # 更新电量
        if self.consider_power:
            usage['power'] = scheduled_task.power_after

        # 更新存储
        if self.consider_storage:
            usage['storage'] = scheduled_task.storage_after

        # 更新最后任务结束时间
        usage['last_task_end'] = scheduled_task.imaging_end

        # 记录已调度任务用于冲突检测
        if 'scheduled_tasks' not in usage:
            usage['scheduled_tasks'] = []
        task_id = task.task_id if isinstance(task, ObservationTask) else task.id
        usage['scheduled_tasks'].append({
            'start': scheduled_task.imaging_start,
            'end': scheduled_task.imaging_end,
            'task_id': task_id
        })

    def _determine_failure_reason(self, task: Any) -> TaskFailureReason:
        """确定任务失败原因"""
        # 检查是否是资源约束 - 使用动态固存消耗
        for sat_id, usage in self._sat_resource_usage.items():
            sat = self.mission.get_satellite_by_id(sat_id)
            if sat:
                imaging_mode = self._select_imaging_mode(sat, task)
                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                storage_needed = self._imaging_calculator.get_storage_consumption(
                    task, imaging_mode, data_rate
                )
                if usage['storage'] + storage_needed > sat.capabilities.storage_capacity:
                    return TaskFailureReason.STORAGE_CONSTRAINT

        return TaskFailureReason.NO_VISIBLE_WINDOW

    def _create_scheduled_task(
        self, task: Any, sat_id: str, window: Any, imaging_mode: Any,
        slew_result: Optional[SlewFeasibilityResult] = None
    ) -> ScheduledTask:
        """
        创建ScheduledTask对象

        Args:
            task: 目标任务
            sat_id: 卫星ID
            window: 可见窗口
            imaging_mode: 成像模式
            slew_result: 机动可行性检查结果（可选）

        Returns:
            ScheduledTask对象
        """
        window_start = window['start'] if isinstance(window, dict) else window.start_time

        # 获取卫星
        sat = self.mission.get_satellite_by_id(sat_id)
        imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)

        # 获取当前资源水平
        usage = self._sat_resource_usage.get(sat_id, {})
        power_before = usage.get('power', 0)
        storage_before = usage.get('storage', 0)

        # 计算资源消耗
        power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
        power_consumed = 0.0
        if sat and self.consider_power:
            power_consumed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)

        # 动态计算固存消耗
        storage_used = 0.0
        if sat and self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_used = self._imaging_calculator.get_storage_consumption(
                task, imaging_mode, data_rate
            )

        # 使用 slew_result 中的机动信息
        if slew_result:
            slew_angle = slew_result.slew_angle
            slew_time_seconds = slew_result.slew_time
            actual_start = slew_result.actual_start
            actual_end = actual_start + timedelta(seconds=imaging_duration)
        else:
            # 回退逻辑
            slew_angle = 0.0
            slew_time_seconds = self.DEFAULT_SLEW_TIME.total_seconds()
            actual_start = window_start
            actual_end = actual_start + timedelta(seconds=imaging_duration)

        # 支持ObservationTask和原始Target
        task_id = task.task_id if isinstance(task, ObservationTask) else task.id
        target_id = task.target_id if isinstance(task, ObservationTask) else task.id

        # 创建ScheduledTask对象
        scheduled_task = ScheduledTask(
            task_id=task_id,
            satellite_id=sat_id,
            target_id=target_id,
            imaging_start=actual_start,
            imaging_end=actual_end,
            imaging_mode=imaging_mode if isinstance(imaging_mode, str) else str(imaging_mode),
            slew_angle=slew_angle,
            slew_time=slew_time_seconds,
            storage_before=storage_before,
            storage_after=storage_before + storage_used,
            power_before=power_before,
            power_after=power_before - power_consumed
        )

        # 计算并应用姿态角（当有预计算位置缓存时，即使简化模式也计算）
        should_calculate_attitude = (
            (not self._use_simplified_slew) or  # 非简化模式
            (self._position_cache is not None)   # 有预计算位置缓存
        )
        if should_calculate_attitude and sat and hasattr(task, 'latitude') and hasattr(task, 'longitude'):
            attitude = self._calculate_attitude_angles(sat, task, actual_start)
            self._apply_attitude_to_scheduled_task(scheduled_task, attitude)

        return scheduled_task
