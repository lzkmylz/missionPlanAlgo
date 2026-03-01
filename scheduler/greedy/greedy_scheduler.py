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

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from ..frequency_utils import ObservationTask, create_observation_tasks
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile


class GreedyScheduler(BaseScheduler):
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
        """
        super().__init__("Greedy", config)
        config = config or {}
        self.heuristic = config.get('heuristic', 'priority')
        self.consider_power = config.get('consider_power', True)
        self.consider_storage = config.get('consider_storage', True)
        self.consider_time_conflicts = config.get('consider_time_conflicts', True)

        # Initialize imaging time calculator
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=config.get('min_imaging_duration', 60),
            max_duration=config.get('max_imaging_duration', 1800),
            default_duration=config.get('default_imaging_duration', 300)
        )
        self._power_profile = PowerProfile(config.get('power_coefficients'))

        # Track satellite resource usage during scheduling
        self._sat_resource_usage: Dict[str, Dict[str, Any]] = {}

    def get_parameters(self) -> Dict[str, Any]:
        """Return algorithm configurable parameters"""
        return {
            'heuristic': self.heuristic,
            'consider_power': self.consider_power,
            'consider_storage': self.consider_storage,
            'consider_time_conflicts': self.consider_time_conflicts,
        }

    def schedule(self) -> ScheduleResult:
        """
        Execute greedy scheduling with full constraint checking

        Returns:
            ScheduleResult: Scheduling result with all scheduled tasks and failures
        """
        self._start_timer()
        self._validate_initialization()

        # Initialize resource tracking for each satellite
        self._sat_resource_usage = {
            sat.id: {
                'power': sat.current_power if hasattr(sat, 'current_power') and sat.current_power > 0
                        else sat.capabilities.power_capacity,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': []  # Track scheduled tasks for conflict detection
            }
            for sat in self.mission.satellites
        }

        # Sort tasks based on heuristic (using frequency-aware tasks)
        pending_tasks = self._sort_tasks(self._create_frequency_aware_tasks())
        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, Any] = {}

        # Main scheduling loop
        for task in pending_tasks:
            best_assignment = self._find_best_assignment(task)

            if best_assignment:
                sat_id, window, imaging_mode = best_assignment

                # Create scheduled task
                scheduled_task = self._create_scheduled_task(
                    task, sat_id, window, imaging_mode
                )
                scheduled_tasks.append(scheduled_task)

                # Update resource usage
                self._update_resource_usage(sat_id, task, window, scheduled_task)

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

        # Calculate makespan
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

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
        Sort tasks by priority (highest first)

        Args:
            tasks: List of tasks

        Returns:
            Sorted list (highest priority first)
        """
        def priority_key(task):
            # Higher priority value = higher priority
            priority = getattr(task, 'priority', 5) or 5
            return -priority  # Negative for descending order

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

    def _find_best_assignment(self, task: Any) -> Optional[Tuple[str, Any, Any]]:
        """
        Find the best satellite-window assignment for a task

        Args:
            task: Target task to schedule

        Returns:
            Tuple of (satellite_id, window, imaging_mode) or None if no valid assignment
        """
        best_assignment = None
        best_score = None

        for sat in self.mission.satellites:
            # Check if satellite can perform this task
            if not self._can_satellite_perform_task(sat, task):
                continue

            # Get visibility windows
            windows = self._get_windows(sat, task)
            if not windows:
                continue

            for window in windows:
                # Extract window times
                window_start, window_end = self._extract_window_times(window)

                if window_start is None or window_end is None:
                    continue

                # Check if window is within target's time window constraints
                if not self._is_window_within_target_constraints(task, window_start, window_end):
                    continue

                # Calculate required imaging time
                imaging_mode = self._select_imaging_mode(sat, task)
                imaging_duration = self._calculate_imaging_time(task, imaging_mode, sat)

                # Check if window is long enough
                window_duration = (window_end - window_start).total_seconds()
                if window_duration < imaging_duration * self.MIN_WINDOW_RATIO:
                    continue

                # Check resource constraints
                if not self._check_resource_constraints(sat, task, imaging_mode):
                    continue

                # Check time conflicts
                if self.consider_time_conflicts:
                    actual_start, actual_end = self._calculate_task_time(
                        sat.id, window_start, imaging_duration
                    )
                    if self._has_time_conflict(sat.id, actual_start, actual_end):
                        continue
                else:
                    actual_start = window_start
                    actual_end = window_start + timedelta(seconds=imaging_duration)

                # Calculate score for this assignment
                score = self._calculate_assignment_score(
                    sat, task, window, imaging_mode, actual_start
                )

                # Update best assignment if this is better
                if best_score is None or score > best_score:
                    best_score = score
                    best_assignment = (sat.id, window, imaging_mode)

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
            if sat.capabilities.resolution > required_resolution:
                return False

        return True

    def _get_windows(self, sat: Any, task: Any) -> List[Any]:
        """
        Get visibility windows for satellite-task pair

        Args:
            sat: Satellite
            task: Target task (ObservationTask or Target)

        Returns:
            List of visibility windows
        """
        if self.window_cache:
            target_id = self._get_target_id(task)
            return self.window_cache.get_windows(sat.id, target_id)
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
        power_ratio = usage.get('power', 0) / sat.capabilities.power_capacity
        storage_ratio = 1.0 - (usage.get('storage', 0) / sat.capabilities.storage_capacity)
        score += (power_ratio + storage_ratio) * 5

        return score

    def _create_scheduled_task(
        self, task: Any, sat_id: str, window: Any, imaging_mode: Any
    ) -> ScheduledTask:
        """
        Create a ScheduledTask object

        Args:
            task: Target task
            sat_id: Satellite ID
            window: Visibility window
            imaging_mode: Imaging mode

        Returns:
            ScheduledTask object
        """
        window_start, window_end = self._extract_window_times(window)

        # Get satellite for satellite-specific constraints
        sat = self.mission.get_satellite_by_id(sat_id)
        imaging_duration = self._calculate_imaging_time(task, imaging_mode, sat)

        # Calculate actual timing
        actual_start, actual_end = self._calculate_task_time(
            sat_id, window_start, imaging_duration
        )

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
            storage_used = self._imaging_calculator.get_storage_consumption(
                task, imaging_mode, data_rate
            )

        # 创建ScheduledTask对象
        scheduled_task = ScheduledTask(
            task_id=task.id,
            satellite_id=sat_id,
            target_id=task.id,
            imaging_start=actual_start,
            imaging_end=actual_end,
            imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
            slew_angle=0.0,  # Could be calculated based on target positions
            storage_before=storage_before,
            storage_after=storage_before + storage_used,
            power_before=power_before,
            power_after=power_before - power_consumed
        )

        # 计算并应用姿态角（用于姿控系统验证）
        if sat and hasattr(task, 'latitude') and hasattr(task, 'longitude'):
            attitude = self._calculate_attitude_angles(sat, task, actual_start)
            self._apply_attitude_to_scheduled_task(scheduled_task, attitude)

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
        usage['scheduled_tasks'].append({
            'start': scheduled_task.imaging_start,
            'end': scheduled_task.imaging_end,
            'task_id': task.id
        })

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
