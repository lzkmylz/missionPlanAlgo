"""
保守策略增量规划器

策略逻辑：
1. 仅使用当前调度方案未占用的资源（时间、电量、存储）
2. 不修改、不抢占任何已规划任务
3. 对于资源不足的目标，标记为失败

适用场景：
- 补充规划（低优先级任务追加）
- 避免影响已有高优先级任务
- 快速增量调度（无需复杂的抢占分析）
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import logging

from ..base_incremental import (
    BaseIncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalPlanResult,
    IncrementalStrategyType,
    ResourceDelta,
    PriorityRule
)
from ..incremental_state import IncrementalState, ResourceWindow
from ..resource_reclaimer import ResourceReclaimer
from ...base_scheduler import ScheduledTask
from payload.imaging_time_calculator import ImagingTimeCalculator

if TYPE_CHECKING:
    from core.models.target import Target

logger = logging.getLogger(__name__)


class ConservativeStrategy(BaseIncrementalPlanner):
    """
    保守策略增量规划器

    仅使用剩余资源进行规划，不修改已有任务
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化保守策略规划器

        Args:
            config: 配置字典
                - min_window_gap: 任务间最小间隔（秒），默认60
                - enable_quality_filtering: 是否启用质量筛选，默认True
                - max_candidates_per_target: 每个目标最大候选数，默认10
        """
        # 处理 config 为 None 的情况
        config = config or {}
        super().__init__(IncrementalStrategyType.CONSERVATIVE, config)
        self.min_window_gap = timedelta(seconds=config.get('min_window_gap', 60))
        self.enable_quality_filtering = config.get('enable_quality_filtering', True)
        self.max_candidates = config.get('max_candidates_per_target', 10)

        # 初始化成像时间计算器
        self._imaging_calculator = ImagingTimeCalculator()

    def plan(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """
        执行保守策略增量规划

        Args:
            request: 增量规划请求

        Returns:
            IncrementalPlanResult: 规划结果
        """
        # 验证请求
        if not self._validate_request(request):
            return self._create_empty_result(request)

        # 设置mission
        if request.mission:
            self.mission = request.mission

        # 初始化状态
        state = IncrementalState(self.mission)
        state.load_from_schedule(request.existing_schedule)

        reclaimer = ResourceReclaimer(state)

        logger.info(f"Starting conservative planning: {len(request.new_targets)} new targets")
        logger.debug(f"Available satellites: {len(state.get_all_satellite_ids())}")

        # 规划结果
        new_tasks: List[ScheduledTask] = []
        failed_targets: List[Tuple[Any, str]] = []
        scheduled_target_ids = set()

        # 按优先级排序新目标
        sorted_targets = self._sort_targets_by_priority(
            request.new_targets,
            request.priority_rules
        )

        # 对每个目标尝试规划
        for target in sorted_targets:
            target_id = getattr(target, 'id', str(target))

            if target_id in scheduled_target_ids:
                continue

            # 尝试为当前目标找到可行的调度方案
            task = self._try_schedule_target(
                target, state, reclaimer
            )

            if task:
                # 添加到新任务列表
                new_tasks.append(task)
                scheduled_target_ids.add(target_id)

                # 更新状态（标记资源已使用）
                self._update_state_with_task(state, task)

                logger.debug(f"Scheduled target {target_id} on {task.satellite_id}")
            else:
                # 记录失败
                failed_targets.append((target, "No available resource window"))
                logger.debug(f"Failed to schedule target {target_id}")

        # 生成合并后的调度结果
        merged_schedule = self._merge_schedules(
            request.existing_schedule,
            new_tasks,
            [],  # 无抢占任务
            []   # 无重调度任务
        )

        # 计算资源变化
        resource_delta = self._calculate_resource_delta(new_tasks)

        success_rate = len(new_tasks) / len(request.new_targets) if request.new_targets else 0.0
        logger.info(f"Conservative planning completed: {len(new_tasks)} scheduled, "
                   f"{len(failed_targets)} failed ({success_rate:.1%})")

        return IncrementalPlanResult(
            merged_schedule=merged_schedule,
            new_tasks=new_tasks,
            preempted_tasks=[],
            rescheduled_tasks=[],
            failed_targets=failed_targets,
            resource_usage_delta=resource_delta,
            strategy_used=IncrementalStrategyType.CONSERVATIVE,
            statistics={
                'total_targets': len(request.new_targets),
                'scheduled_count': len(new_tasks),
                'failed_count': len(failed_targets),
                'success_rate': len(new_tasks) / len(request.new_targets) if request.new_targets else 0.0,
                'strategy': 'conservative'
            }
        )

    def _sort_targets_by_priority(self, targets: List[Target],
                                   rules: Optional[PriorityRule] = None) -> List[Any]:
        """
        按优先级排序目标

        Args:
            targets: 目标列表
            rules: 优先级规则

        Returns:
            List: 排序后的目标列表（高优先级在前）
        """
        rules = rules or PriorityRule()

        def get_priority(target):
            return self._calculate_task_priority(target, rules)

        return sorted(targets, key=get_priority, reverse=True)

    def _try_schedule_target(self, target: Any,
                             state: IncrementalState,
                             reclaimer: ResourceReclaimer) -> Optional[ScheduledTask]:
        """
        尝试为单个目标找到调度方案

        Args:
            target: 目标对象
            state: 增量规划状态
            reclaimer: 资源回收计算器

        Returns:
            Optional[ScheduledTask]: 调度成功的任务，失败返回None
        """
        target_id = getattr(target, 'id', str(target))

        # 获取成像需求
        imaging_duration = self._get_imaging_duration(target)
        required_storage = self._estimate_storage(target)
        required_power = self._estimate_power(target)

        # 查找所有可能的卫星-窗口组合
        candidates = []

        for sat_id in state.get_all_satellite_ids():
            # 获取可见窗口
            visibility_windows = self._find_visibility_windows(sat_id, target)

            if not visibility_windows:
                continue

            # 查找匹配的资源窗口
            matching_windows = reclaimer.find_matching_windows(
                sat_id, visibility_windows,
                min_duration=imaging_duration + self.min_window_gap.total_seconds(),
                required_power=required_power,
                required_storage=required_storage
            )

            for window in matching_windows:
                candidates.append((sat_id, window))

        if not candidates:
            return None

        # 按窗口质量排序
        candidates.sort(key=lambda x: x[1].quality_score, reverse=True)

        # 尝试最好的候选
        for sat_id, window in candidates[:self.max_candidates]:
            task = self._create_scheduled_task(
                target, sat_id, window,
                imaging_duration, required_power, required_storage
            )

            # 验证任务可行性
            if self._validate_task_feasibility(task, state):
                return task

        return None

    def _find_visibility_windows(self, satellite_id: str, target: Any) -> List[Tuple[datetime, datetime]]:
        """
        查找卫星对目标的可见窗口

        Args:
            satellite_id: 卫星ID
            target: 目标对象

        Returns:
            List[Tuple[datetime, datetime]]: 可见窗口列表
        """
        if not self.window_cache:
            # 如果没有窗口缓存，使用简化的假设：全天可见
            if self.mission:
                return [(self.mission.start_time, self.mission.end_time)]
            return []

        cache_key = f"{satellite_id}:{getattr(target, 'id', str(target))}"
        windows = self.window_cache.get(cache_key, [])

        # 转换为datetime元组列表
        result = []
        for window in windows:
            if isinstance(window, dict):
                start = window.get('start') or window.get('start_time')
                end = window.get('end') or window.get('end_time')
                if start and end:
                    result.append((start, end))
            elif isinstance(window, (tuple, list)) and len(window) == 2:
                result.append((window[0], window[1]))

        return result

    def _get_imaging_duration(self, target: Any) -> float:
        """获取目标成像时长（秒）"""
        # 尝试从目标获取
        duration = getattr(target, 'imaging_duration', None)
        if duration:
            return float(duration)

        # 根据成像模式估算
        mode = getattr(target, 'imaging_mode', 'standard')
        try:
            return self._imaging_calculator.calculate(mode)
        except Exception:
            return self._imaging_calculator.default_duration

    def _estimate_storage(self, target: Any) -> float:
        """估算所需存储（GB）"""
        # 基于成像时长估算
        duration = self._get_imaging_duration(target)
        mode = getattr(target, 'imaging_mode', 'standard')

        # 简化的存储估算
        storage_rates = {
            'low_resolution': 0.01,    # 10MB/s
            'standard': 0.05,           # 50MB/s
            'high_resolution': 0.2,     # 200MB/s
        }
        rate = storage_rates.get(mode, 0.05)

        return duration * rate

    def _estimate_power(self, target: Any) -> float:
        """估算所需电量（Wh）"""
        # 基于成像时长估算
        duration = self._get_imaging_duration(target)

        # 简化的电量估算：成像功耗约100W
        imaging_power = 100.0  # Watts
        return (duration / 3600) * imaging_power

    def _validate_task_feasibility(self, task: ScheduledTask,
                                   state: IncrementalState) -> bool:
        """
        验证任务可行性（时间冲突、资源等）

        Args:
            task: 待验证任务
            state: 当前状态

        Returns:
            bool: 是否可行
        """
        sat_state = state.get_satellite_state(task.satellite_id)
        if not sat_state:
            return False

        # 检查时间冲突
        for existing_task in sat_state.scheduled_tasks:
            # 检查时间重叠
            if (task.imaging_start < existing_task['imaging_end'] and
                task.imaging_end > existing_task['imaging_start']):
                return False

            # 检查最小间隔
            if task.imaging_end <= existing_task['imaging_start']:
                gap = (existing_task['imaging_start'] - task.imaging_end).total_seconds()
                if gap < self.min_window_gap.total_seconds():
                    return False
            elif existing_task['imaging_end'] <= task.imaging_start:
                gap = (task.imaging_start - existing_task['imaging_end']).total_seconds()
                if gap < self.min_window_gap.total_seconds():
                    return False

        return True

    def _create_scheduled_task(self, target: Any,
                               satellite_id: str,
                               window: ResourceWindow,
                               duration: float,
                               power: float,
                               storage: float) -> ScheduledTask:
        """
        创建ScheduledTask对象

        Args:
            target: 目标对象
            satellite_id: 卫星ID
            window: 资源窗口
            duration: 成像时长
            power: 电量需求
            storage: 存储需求

        Returns:
            ScheduledTask: 调度任务
        """
        target_id = getattr(target, 'id', str(target))
        task_id = f"incremental_{target_id}_{satellite_id}"

        # 在窗口内居中安排
        window_duration = (window.end_time - window.start_time).total_seconds()
        if window_duration > duration:
            offset = (window_duration - duration) / 2
            imaging_start = window.start_time + timedelta(seconds=offset)
        else:
            imaging_start = window.start_time

        imaging_end = imaging_start + timedelta(seconds=duration)

        return ScheduledTask(
            task_id=task_id,
            satellite_id=satellite_id,
            target_id=target_id,
            imaging_start=imaging_start,
            imaging_end=imaging_end,
            imaging_mode=getattr(target, 'imaging_mode', 'standard'),
            slew_angle=0.0,
            slew_time=0.0,
            storage_before=window.available_storage,
            storage_after=window.available_storage + storage,
            power_before=window.available_power,
            power_after=window.available_power - power,
            power_consumed=power,
            priority=getattr(target, 'priority', 0)
        )

    def _update_state_with_task(self, state: IncrementalState, task: ScheduledTask) -> None:
        """
        更新状态以反映新任务

        Args:
            state: 增量规划状态
            task: 新任务
        """
        state.add_task(task.satellite_id, {
            'task_id': task.task_id,
            'target_id': task.target_id,
            'imaging_start': task.imaging_start,
            'imaging_end': task.imaging_end,
            'imaging_mode': task.imaging_mode,
            'power_consumed': task.power_consumed,
            'storage_produced': task.storage_after - task.storage_before,
            'priority': task.priority or 0
        })

    def _calculate_resource_delta(self, new_tasks: List[ScheduledTask]) -> ResourceDelta:
        """
        计算资源变化量

        Args:
            new_tasks: 新增任务列表

        Returns:
            ResourceDelta: 资源变化
        """
        total_power = sum(t.power_consumed for t in new_tasks)
        total_storage = sum(t.storage_after - t.storage_before for t in new_tasks)
        total_time = sum((t.imaging_end - t.imaging_start).total_seconds() for t in new_tasks)

        return ResourceDelta(
            power_delta=total_power,
            storage_delta=total_storage,
            time_delta=total_time,
            task_count_delta=len(new_tasks)
        )

    def _create_empty_result(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """创建空结果"""
        return IncrementalPlanResult(
            merged_schedule=request.existing_schedule,
            new_tasks=[],
            preempted_tasks=[],
            rescheduled_tasks=[],
            failed_targets=[(t, "Invalid request") for t in request.new_targets],
            resource_usage_delta=ResourceDelta(),
            strategy_used=IncrementalStrategyType.CONSERVATIVE,
            statistics={'error': 'Invalid request'}
        )
