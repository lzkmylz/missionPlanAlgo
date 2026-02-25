"""
SPT调度器（最短处理时间优先）

实现设计文档第8章设计：
- SPT（Shortest Processing Time）启发式调度
- 按处理时间排序任务
- 处理时间相同的按优先级排序
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile


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
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=config.get('min_imaging_duration', 60),
            max_duration=config.get('max_imaging_duration', 1800),
            default_duration=config.get('default_imaging_duration', 300)
        )
        self._power_profile = PowerProfile(config.get('power_coefficients'))

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

        # 获取任务列表并按SPT规则排序
        pending_tasks = self._sort_tasks_by_processing_time(list(self.mission.targets))
        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, Any] = {}

        # 卫星资源状态跟踪
        sat_resource_usage = {
            sat.id: {
                'power': sat.current_power if hasattr(sat, 'current_power') else sat.capabilities.power_capacity,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': [],  # 跟踪已调度任务用于冲突检测
            }
            for sat in self.mission.satellites
        }

        # SPT调度主循环
        for task in pending_tasks:
            best_assignment = self._find_best_assignment(task, sat_resource_usage)

            if best_assignment:
                sat_id, window, imaging_mode = best_assignment

                # 计算实际开始和结束时间
                window_start = window['start'] if isinstance(window, dict) else window.start_time
                window_end = window['end'] if isinstance(window, dict) else window.end_time
                usage = sat_resource_usage[sat_id]
                last_task_end = usage.get('last_task_end', self.mission.start_time)
                earliest_start = last_task_end + self.DEFAULT_SLEW_TIME
                actual_start = max(window_start, earliest_start)
                imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)
                actual_end = actual_start + timedelta(seconds=imaging_duration)

                # 更新资源状态
                self._update_resource_usage(sat_id, task, actual_start, actual_end, sat_resource_usage)

                # 创建调度任务
                usage = sat_resource_usage[sat_id]
                sat = self.mission.get_satellite_by_id(sat_id)
                power_consumed = 0
                if self.consider_power and sat:
                    power_consumed = sat.capabilities.power_capacity * self._power_profile.get_coefficient_for_mode(imaging_mode) * (self._imaging_calculator.calculate(task, imaging_mode) / 3600)
                scheduled_task = ScheduledTask(
                    task_id=task.id,
                    satellite_id=sat_id,
                    target_id=task.id,
                    imaging_start=actual_start,
                    imaging_end=actual_end,
                    imaging_mode=imaging_mode if isinstance(imaging_mode, str) else str(imaging_mode),
                    storage_before=usage['storage'] - (getattr(task, 'data_size_gb', 1.0) if self.consider_storage else 0),
                    storage_after=usage['storage'],
                    power_before=usage['power'] + power_consumed,
                    power_after=usage['power'],
                )
                scheduled_tasks.append(scheduled_task)

                self._add_convergence_point(len(scheduled_tasks))
            else:
                # 记录失败原因
                reason = self._determine_failure_reason(task, sat_resource_usage)
                self._record_failure(
                    task_id=task.id,
                    reason=reason,
                    detail=f"No feasible assignment found for task {task.id}"
                )
                unscheduled[task.id] = self._failure_log[-1]

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

    def _find_best_assignment(
        self,
        task: Any,
        sat_resource_usage: Dict
    ) -> Optional[Tuple[str, Any, Any]]:
        """
        为任务找到最佳卫星-窗口组合

        SPT策略：选择最早的可用窗口（与EDD类似，但排序规则不同）

        Args:
            task: 目标任务
            sat_resource_usage: 卫星资源使用情况

        Returns:
            Optional[Tuple]: (satellite_id, window, imaging_mode) 或 None
        """
        best_window = None
        best_sat_id = None
        best_start = None
        best_imaging_mode = None

        for sat in self.mission.satellites:
            # 检查卫星能力匹配
            if not self._can_satellite_perform_task(sat, task):
                continue

            # 获取可见窗口
            windows = self._get_feasible_windows(sat, task, sat_resource_usage)
            if not windows:
                continue

            # SPT策略：选择最早的窗口（与EDD相同）
            for window in windows:
                window_start = window['start'] if isinstance(window, dict) else window.start_time
                window_end = window['end'] if isinstance(window, dict) else window.end_time

                # 计算成像时长
                imaging_mode = self._select_imaging_mode(sat, task)
                imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)

                # 计算实际开始时间（考虑转移时间）
                usage = sat_resource_usage[sat.id]
                last_task_end = usage.get('last_task_end', self.mission.start_time)
                earliest_start = last_task_end + self.DEFAULT_SLEW_TIME
                actual_start = max(window_start, earliest_start)
                actual_end = actual_start + timedelta(seconds=imaging_duration)

                # 检查是否超出窗口结束时间
                if actual_end > window_end:
                    continue

                # 检查是否在截止时间之前
                if task.time_window_end and actual_start > task.time_window_end:
                    if not self.allow_tardiness:
                        continue

                # 检查时间冲突
                if self._has_time_conflict(sat.id, actual_start, actual_end, sat_resource_usage):
                    continue

                # 检查资源约束
                if not self._check_resource_constraints(sat, task, window, sat_resource_usage):
                    continue

                # 选择最早的窗口
                if best_start is None or actual_start < best_start:
                    best_start = actual_start
                    best_window = window
                    best_sat_id = sat.id
                    best_imaging_mode = imaging_mode

        if best_sat_id and best_window:
            return (best_sat_id, best_window, best_imaging_mode)
        return None

    def _has_time_conflict(self, sat_id: str, start: datetime, end: datetime, sat_resource_usage: Dict) -> bool:
        """检查是否与已调度任务有时间冲突"""
        usage = sat_resource_usage.get(sat_id, {})
        scheduled_tasks = usage.get('scheduled_tasks', [])

        for task in scheduled_tasks:
            # 检查重叠
            if not (end <= task['start'] or start >= task['end']):
                return True
        return False

    def _get_feasible_windows(self, sat: Any, task: Any, sat_resource_usage: Dict) -> List[Any]:
        """获取可行的时间窗口"""
        if self.window_cache:
            return self.window_cache.get_windows(sat.id, task.id)
        return []

    def _can_satellite_perform_task(self, sat: Any, task: Any) -> bool:
        """检查卫星是否能执行任务"""
        if not sat.capabilities.imaging_modes:
            return False
        return True

    def _check_resource_constraints(
        self,
        sat: Any,
        task: Any,
        window: Any,
        sat_resource_usage: Dict
    ) -> bool:
        """检查资源约束"""
        usage = sat_resource_usage[sat.id]

        # 电量约束
        if self.consider_power:
            imaging_mode = self._select_imaging_mode(sat, task)
            duration = self._imaging_calculator.calculate(task, imaging_mode)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_needed = sat.capabilities.power_capacity * power_coefficient * (duration / 3600)
            if usage['power'] < power_needed:
                return False

        # 存储约束
        if self.consider_storage:
            storage_needed = getattr(task, 'data_size_gb', 1.0)
            if usage['storage'] + storage_needed > sat.capabilities.storage_capacity:
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
        self,
        sat_id: str,
        task: Any,
        actual_start,
        actual_end=None,
        sat_resource_usage: Dict = None
    ) -> None:
        """更新资源使用状态

        Args:
            sat_id: 卫星ID
            task: 目标任务
            actual_start: 实际开始时间（datetime）或 window 字典（测试兼容）
            actual_end: 实际结束时间（datetime）或 sat_resource_usage 字典（测试兼容）
            sat_resource_usage: 卫星资源使用情况字典
        """
        # 处理测试用例的调用方式: _update_resource_usage(sat_id, task, window, sat_resource_usage)
        # 其中 window 是 {'start': ..., 'end': ...} 字典
        if sat_resource_usage is None and isinstance(actual_end, dict):
            sat_resource_usage = actual_end
            window = actual_start
            actual_start = window['start'] if isinstance(window, dict) else window.start_time
            actual_end = window['end'] if isinstance(window, dict) else window.end_time

        usage = sat_resource_usage[sat_id]
        sat = self.mission.get_satellite_by_id(sat_id)

        if sat:
            # 更新电量
            if self.consider_power:
                imaging_mode = self._select_imaging_mode(sat, task)
                duration = self._imaging_calculator.calculate(task, imaging_mode)
                power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
                power_consumed = sat.capabilities.power_capacity * power_coefficient * (duration / 3600)
                usage['power'] -= power_consumed

            # 更新存储
            if self.consider_storage:
                storage_used = getattr(task, 'data_size_gb', 1.0)
                usage['storage'] += storage_used

            # 更新时间
            usage['last_task_end'] = actual_end

            # 记录已调度任务用于冲突检测
            if 'scheduled_tasks' not in usage:
                usage['scheduled_tasks'] = []
            usage['scheduled_tasks'].append({
                'start': actual_start,
                'end': actual_end,
                'task_id': task.id
            })

    def _determine_failure_reason(
        self,
        task: Any,
        sat_resource_usage: Dict
    ) -> TaskFailureReason:
        """确定任务失败原因"""
        # 检查是否是资源约束
        for sat_id, usage in sat_resource_usage.items():
            sat = self.mission.get_satellite_by_id(sat_id)
            if sat:
                storage_needed = getattr(task, 'data_size_gb', 1.0)
                if usage['storage'] + storage_needed > sat.capabilities.storage_capacity:
                    return TaskFailureReason.STORAGE_CONSTRAINT

        return TaskFailureReason.NO_VISIBLE_WINDOW
