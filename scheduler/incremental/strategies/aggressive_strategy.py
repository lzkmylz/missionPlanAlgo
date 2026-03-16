"""
激进策略增量规划器

策略逻辑：
1. 优先使用剩余资源进行规划
2. 剩余资源不足时，尝试抢占已有任务资源
3. 被抢占的任务尝试重新调度到其他位置
4. 无法重调度的任务标记为失败

抢占决策考虑因素：
- 任务优先级差
- 资源释放收益
- 重调度难度
- 级联影响范围

适用场景：
- 紧急任务插入
- 高优先级目标覆盖
- 资源紧张但需要最大化任务完成率
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging

# 阶段结果类型定义（放在类之前以便类型注解使用）
class _Phase1Result:
    """阶段1结果"""
    def __init__(self, new_tasks: List[ScheduledTask], remaining_targets: List[Target]):
        self.new_tasks = new_tasks
        self.remaining_targets = remaining_targets


class _Phase2Result:
    """阶段2结果"""
    def __init__(
        self,
        new_tasks: List[ScheduledTask],
        preempted_tasks: List[ScheduledTask],
        rescheduled_tasks: List[ScheduledTask],
        failed_targets: List[Tuple[Any, str]]
    ):
        self.new_tasks = new_tasks
        self.preempted_tasks = preempted_tasks
        self.rescheduled_tasks = rescheduled_tasks
        self.failed_targets = failed_targets

from ..base_incremental import (
    BaseIncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalPlanResult,
    IncrementalStrategyType,
    ResourceDelta,
    PriorityRule,
    PreemptionRule,
    PreemptionCandidate
)
from ..incremental_state import IncrementalState, ResourceWindow
from ..resource_reclaimer import ResourceReclaimer
from ...base_scheduler import ScheduleResult, ScheduledTask, TaskFailure, TaskFailureReason
from payload.imaging_time_calculator import ImagingTimeCalculator

logger = logging.getLogger(__name__)


class AggressiveStrategy(BaseIncrementalPlanner):
    """
    激进策略增量规划器

    允许抢占已有任务资源，被抢占任务尝试重调度
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化激进策略规划器

        Args:
            config: 配置字典
                - min_window_gap: 任务间最小间隔（秒），默认60
                - enable_quality_filtering: 是否启用质量筛选，默认True
                - max_preemption_ratio: 最大抢占比例，默认0.2
                - max_cascade_depth: 最大级联深度，默认3
                - min_priority_difference: 最小优先级差才允许抢占，默认2
        """
        # 处理 config 为 None 的情况
        config = config or {}
        super().__init__(IncrementalStrategyType.AGGRESSIVE, config)
        self.min_window_gap = timedelta(seconds=config.get('min_window_gap', 60))
        self.enable_quality_filtering = config.get('enable_quality_filtering', True)
        self.max_preemption_ratio = config.get('max_preemption_ratio', 0.2)
        self.max_cascade_depth = config.get('max_cascade_depth', 3)
        self.min_priority_diff = config.get('min_priority_difference', 2)

        # 初始化成像时间计算器
        self._imaging_calculator = ImagingTimeCalculator()

    def plan(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """
        执行激进策略增量规划

        流程：
        1. 验证请求并初始化
        2. 阶段1：保守策略调度
        3. 阶段2：激进策略调度（抢占）
        4. 编译结果并返回

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

        # 获取抢占规则
        preemption_rules = request.preemption_rules or PreemptionRule(
            max_preemption_ratio=request.max_preemption_ratio,
            max_cascade_depth=self.max_cascade_depth,
            min_priority_difference=self.min_priority_diff
        )

        # 初始化状态
        state = IncrementalState(self.mission)
        state.load_from_schedule(request.existing_schedule)
        reclaimer = ResourceReclaimer(state)

        # 按优先级排序目标
        sorted_targets = self._sort_targets_by_priority(
            request.new_targets,
            request.priority_rules
        )

        logger.info(f"Starting aggressive planning: {len(request.new_targets)} new targets")
        logger.debug(f"Max preemption ratio: {preemption_rules.max_preemption_ratio}, "
                    f"Max cascade depth: {preemption_rules.max_cascade_depth}")

        # 阶段1：保守策略
        phase1_result = self._run_phase1_conservative(
            sorted_targets, state, reclaimer
        )

        # 阶段2：激进策略（抢占）
        phase2_result = self._run_phase2_aggressive(
            phase1_result.remaining_targets, state, reclaimer, preemption_rules
        )

        # 编译结果
        return self._compile_results(
            request, phase1_result, phase2_result
        )

    def _try_schedule_target_conservative(self, target: Any,
                                          state: IncrementalState,
                                          reclaimer: ResourceReclaimer) -> Optional[ScheduledTask]:
        """
        保守方式尝试调度目标（仅使用剩余资源）

        Returns:
            Optional[ScheduledTask]: 调度成功的任务
        """
        target_id = getattr(target, 'id', str(target))

        # 获取成像需求
        imaging_duration = self._get_imaging_duration(target)
        required_storage = self._estimate_storage(target)
        required_power = self._estimate_power(target)

        # 查找所有可能的候选
        for sat_id in state.get_all_satellite_ids():
            visibility_windows = self._find_visibility_windows(sat_id, target)

            if not visibility_windows:
                continue

            matching_windows = reclaimer.find_matching_windows(
                sat_id, visibility_windows,
                min_duration=imaging_duration + self.min_window_gap.total_seconds(),
                required_power=required_power,
                required_storage=required_storage
            )

            for window in matching_windows:
                task = self._create_scheduled_task(
                    target, sat_id, window,
                    imaging_duration, required_power, required_storage
                )

                if self._validate_task_feasibility(task, state):
                    return task

        return None

    def _try_schedule_with_preemption(self, target: Any,
                                     state: IncrementalState,
                                     reclaimer: ResourceReclaimer,
                                     rules: PreemptionRule,
                                     depth: int = 0) -> Optional[Tuple[ScheduledTask, List[ScheduledTask], List[ScheduledTask]]]:
        """
        尝试通过抢占调度目标

        Returns:
            Optional[Tuple]: (新任务, 被抢占任务列表, 重调度任务列表)
        """
        if depth >= rules.max_cascade_depth:
            return None

        target_id = getattr(target, 'id', str(target))
        target_priority = getattr(target, 'priority', 0)

        # 获取成像需求
        imaging_duration = self._get_imaging_duration(target)
        required_storage = self._estimate_storage(target)
        required_power = self._estimate_power(target)

        # 收集所有可能的抢占候选
        candidates: List[Tuple[str, ResourceWindow, List[PreemptionCandidate]]] = []

        for sat_id in state.get_all_satellite_ids():
            # 查找可见窗口
            visibility_windows = self._find_visibility_windows(sat_id, target)
            if not visibility_windows:
                continue

            # 查找需要抢占才能满足的窗口
            sat_state = state.get_satellite_state(sat_id)
            if not sat_state:
                continue

            for vis_start, vis_end in visibility_windows:
                # 计算交集时段
                for task_info in sat_state.scheduled_tasks:
                    task_start = task_info['imaging_start']
                    task_end = task_info['imaging_end']

                    # 检查是否有重叠
                    if task_start < vis_end and task_end > vis_start:
                        # 创建抢占候选
                        candidate = PreemptionCandidate(
                            task=None,  # 稍后填充
                            satellite_id=sat_id,
                            reclaimable_power=task_info.get('power_consumed', 0),
                            reclaimable_storage=task_info.get('storage_produced', 0),
                            reschedule_difficulty=0.5,  # 默认中等难度
                            priority_score=task_info.get('priority', 0),
                            preemption_benefit=0.0
                        )

                        # 计算抢占评分
                        if target_priority - candidate.priority_score < rules.min_priority_difference:
                            continue  # 优先级差不够

                        # 找到可抢占的窗口
                        window = ResourceWindow(
                            start_time=max(vis_start, self.mission.start_time),
                            end_time=min(vis_end, self.mission.end_time),
                            available_power=sat_state.power_capacity,
                            available_storage=sat_state.storage_capacity,
                            satellite_id=sat_id
                        )

                        candidates.append((sat_id, window, [candidate]))

        if not candidates:
            return None

        # 按质量排序候选
        candidates.sort(key=lambda x: x[1].quality_score, reverse=True)

        # 尝试抢占
        for sat_id, window, preempt_candidates in candidates[:5]:
            # 收集被抢占任务
            preempted: List[ScheduledTask] = []

            for pc in preempt_candidates:
                # 查找实际任务
                sat_state = state.get_satellite_state(sat_id)
                for task_info in sat_state.scheduled_tasks:
                    if task_info.get('task_id') == pc.task.task_id if pc.task else False:
                        # 重建ScheduledTask
                        preempted_task = self._reconstruct_task_from_info(task_info, sat_id)
                        if preempted_task:
                            preempted.append(preempted_task)

            if not preempted:
                continue

            # 尝试创建新任务
            new_task = self._create_scheduled_task(
                target, sat_id, window,
                imaging_duration, required_power, required_storage
            )

            # 尝试重调度被抢占任务
            rescheduled: List[ScheduledTask] = []
            can_reschedule_all = True

            for preempted_task in preempted:
                # 尝试在其他位置重调度
                rescheduled_task = self._try_reschedule_task(
                    preempted_task, state, reclaimer, {t.task_id for t in preempted}
                )

                if rescheduled_task:
                    rescheduled.append(rescheduled_task)
                elif not rules.allow_cascade:
                    can_reschedule_all = False
                    break
                else:
                    # 尝试级联抢占
                    cascade_result = self._try_schedule_with_preemption(
                        self._get_target_from_task(preempted_task),
                        state, reclaimer, rules, depth + 1
                    )

                    if cascade_result:
                        cascade_count += 1
                        rescheduled.append(cascade_result[0])
                    else:
                        can_reschedule_all = False
                        break

            if can_reschedule_all:
                return new_task, preempted, rescheduled

        return None

    def _try_reschedule_task(self, task: ScheduledTask,
                            state: IncrementalState,
                            reclaimer: ResourceReclaimer,
                            excluded_task_ids: Set[str]) -> Optional[ScheduledTask]:
        """
        尝试在其他位置重调度任务

        Args:
            task: 需要重调度的任务
            state: 当前状态
            reclaimer: 资源回收计算器
            excluded_task_ids: 需要排除的任务ID（被抢占的）

        Returns:
            Optional[ScheduledTask]: 重调度成功的任务
        """
        duration = (task.imaging_end - task.imaging_start).total_seconds()
        required_power = task.power_consumed
        required_storage = task.storage_after - task.storage_before

        # 尝试其他卫星
        for sat_id in state.get_all_satellite_ids():
            if sat_id == task.satellite_id:
                continue  # 稍后处理原卫星

            windows = reclaimer.find_matching_windows(
                sat_id, [(self.mission.start_time, self.mission.end_time)],
                min_duration=duration + self.min_window_gap.total_seconds(),
                required_power=required_power,
                required_storage=required_storage
            )

            for window in windows:
                new_task = ScheduledTask(
                    task_id=task.task_id,
                    satellite_id=sat_id,
                    target_id=task.target_id,
                    imaging_start=window.start_time,
                    imaging_end=window.start_time + timedelta(seconds=duration),
                    imaging_mode=task.imaging_mode,
                    priority=task.priority
                )

                if self._validate_task_feasibility(new_task, state):
                    return new_task

        # 尝试原卫星的其他时间
        sat_state = state.get_satellite_state(task.satellite_id)
        if sat_state:
            # 临时移除该任务检查其他窗口
            original_windows = sat_state.find_resource_windows(
                self.mission.start_time, self.mission.end_time
            )

            for window in original_windows:
                if window.duration() >= duration + self.min_window_gap.total_seconds():
                    new_task = ScheduledTask(
                        task_id=task.task_id,
                        satellite_id=task.satellite_id,
                        target_id=task.target_id,
                        imaging_start=window.start_time,
                        imaging_end=window.start_time + timedelta(seconds=duration),
                        imaging_mode=task.imaging_mode,
                        priority=task.priority
                    )

                    if self._validate_task_feasibility(new_task, state):
                        return new_task

        return None

    def _reconstruct_task_from_info(self, task_info: Dict, satellite_id: str) -> Optional[ScheduledTask]:
        """从任务信息字典重建ScheduledTask"""
        try:
            return ScheduledTask(
                task_id=task_info['task_id'],
                satellite_id=satellite_id,
                target_id=task_info['target_id'],
                imaging_start=task_info['imaging_start'],
                imaging_end=task_info['imaging_end'],
                imaging_mode=task_info.get('imaging_mode', 'standard'),
                priority=task_info.get('priority', 0)
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct task: {e}")
            return None

    def _get_target_from_task(self, task: ScheduledTask) -> Target:
        """从任务获取目标对象（简化实现）"""
        # 创建一个模拟目标对象
        class MockTarget:
            def __init__(self, task):
                self.id = task.target_id
                self.priority = task.priority or 0
                self.imaging_mode = task.imaging_mode
                self.imaging_duration = (task.imaging_end - task.imaging_start).total_seconds()

        return MockTarget(task)

    def _sort_targets_by_priority(self, targets: List[Target],
                                   rules: Optional[PriorityRule] = None) -> List[Any]:
        """按优先级排序目标"""
        rules = rules or PriorityRule()

        def get_priority(target):
            return self._calculate_task_priority(target, rules)

        return sorted(targets, key=get_priority, reverse=True)

    def _find_visibility_windows(self, satellite_id: str, target: Any) -> List[Tuple[datetime, datetime]]:
        """查找可见窗口"""
        if not self.window_cache:
            if self.mission:
                return [(self.mission.start_time, self.mission.end_time)]
            return []

        cache_key = f"{satellite_id}:{getattr(target, 'id', str(target))}"
        windows = self.window_cache.get(cache_key, [])

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
        """获取成像时长"""
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
        """估算存储需求"""
        duration = self._get_imaging_duration(target)
        mode = getattr(target, 'imaging_mode', 'standard')

        storage_rates = {
            'low_resolution': 0.01,
            'standard': 0.05,
            'high_resolution': 0.2,
        }
        return duration * storage_rates.get(mode, 0.05)

    def _estimate_power(self, target: Any) -> float:
        """估算电量需求"""
        duration = self._get_imaging_duration(target)
        return (duration / 3600) * 100.0  # 100W imaging power

    def _validate_task_feasibility(self, task: ScheduledTask,
                                   state: IncrementalState) -> bool:
        """验证任务可行性"""
        sat_state = state.get_satellite_state(task.satellite_id)
        if not sat_state:
            return False

        for existing_task in sat_state.scheduled_tasks:
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
        """创建ScheduledTask"""
        target_id = getattr(target, 'id', str(target))
        task_id = f"incremental_{target_id}_{satellite_id}"

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
        """更新状态"""
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

    def _calculate_resource_delta(self, new_tasks: List[ScheduledTask],
                                  preempted_tasks: List[ScheduledTask],
                                  rescheduled_tasks: List[ScheduledTask]) -> ResourceDelta:
        """计算资源变化"""
        new_power = sum(t.power_consumed for t in new_tasks)
        new_storage = sum(t.storage_after - t.storage_before for t in new_tasks)
        new_time = sum((t.imaging_end - t.imaging_start).total_seconds() for t in new_tasks)

        preempted_power = sum(t.power_consumed for t in preempted_tasks)
        preempted_storage = sum(t.storage_after - t.storage_before for t in preempted_tasks)

        return ResourceDelta(
            power_delta=new_power - preempted_power,
            storage_delta=new_storage - preempted_storage,
            time_delta=new_time,
            task_count_delta=len(new_tasks) - len(preempted_tasks)
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
            strategy_used=IncrementalStrategyType.AGGRESSIVE,
            statistics={'error': 'Invalid request'}
        )

    def _run_phase1_conservative(
        self,
        sorted_targets: List[Target],
        state: IncrementalState,
        reclaimer: ResourceReclaimer
    ) -> '_Phase1Result':
        """
        阶段1：使用保守策略调度目标

        Args:
            sorted_targets: 按优先级排序的目标列表
            state: 增量规划状态
            reclaimer: 资源回收计算器

        Returns:
            _Phase1Result: 阶段1结果（已调度任务+剩余目标）
        """
        new_tasks: List[ScheduledTask] = []
        remaining_targets: List[Target] = []

        for target in sorted_targets:
            target_id = getattr(target, 'id', str(target))
            task = self._try_schedule_target_conservative(target, state, reclaimer)

            if task:
                new_tasks.append(task)
                self._update_state_with_task(state, task)
                logger.debug(f"Conservatively scheduled target {target_id}")
            else:
                remaining_targets.append(target)

        logger.info(f"Phase 1 (conservative): {len(new_tasks)} scheduled, "
                   f"{len(remaining_targets)} remaining")

        return _Phase1Result(new_tasks=new_tasks, remaining_targets=remaining_targets)

    def _run_phase2_aggressive(
        self,
        remaining_targets: List[Target],
        state: IncrementalState,
        reclaimer: ResourceReclaimer,
        preemption_rules: PreemptionRule
    ) -> '_Phase2Result':
        """
        阶段2：使用激进策略调度目标（允许抢占）

        Args:
            remaining_targets: 阶段1未能调度的目标
            state: 增量规划状态
            reclaimer: 资源回收计算器
            preemption_rules: 抢占规则

        Returns:
            _Phase2Result: 阶段2结果
        """
        new_tasks: List[ScheduledTask] = []
        preempted_tasks: List[ScheduledTask] = []
        rescheduled_tasks: List[ScheduledTask] = []
        failed_targets: List[Tuple[Any, str]] = []

        # 计算可抢占任务上限
        total_existing = sum(
            len(s.scheduled_tasks) for s in state.satellite_states.values()
        )
        max_preempt = int(total_existing * preemption_rules.max_preemption_ratio)
        preemption_count = 0

        for target in remaining_targets:
            target_id = getattr(target, 'id', str(target))

            if preemption_count >= max_preempt:
                failed_targets.append((target, "Preemption limit reached"))
                continue

            result = self._try_schedule_with_preemption(
                target, state, reclaimer, preemption_rules
            )

            if result:
                new_task, preempted, rescheduled = result
                new_tasks.append(new_task)
                preempted_tasks.extend(preempted)
                rescheduled_tasks.extend(rescheduled)
                preemption_count += len(preempted)

                # 更新状态
                self._update_state_with_task(state, new_task)
                for task in preempted:
                    state.remove_task(task.satellite_id, task.task_id)
                for task in rescheduled:
                    state.add_task(task.satellite_id, {
                        'task_id': task.task_id,
                        'target_id': task.target_id,
                        'imaging_start': task.imaging_start,
                        'imaging_end': task.imaging_end,
                        'power_consumed': task.power_consumed,
                        'storage_produced': task.storage_after - task.storage_before,
                        'priority': task.priority or 0
                    })

                logger.debug(f"Scheduled target {target_id} with preemption "
                           f"({len(preempted)} tasks preempted)")
            else:
                failed_targets.append((target, "No feasible preemption found"))
                logger.debug(f"Failed to schedule target {target_id} even with preemption")

        return _Phase2Result(
            new_tasks=new_tasks,
            preempted_tasks=preempted_tasks,
            rescheduled_tasks=rescheduled_tasks,
            failed_targets=failed_targets
        )

    def _compile_results(
        self,
        request: IncrementalPlanRequest,
        phase1: '_Phase1Result',
        phase2: '_Phase2Result'
    ) -> IncrementalPlanResult:
        """
        编译阶段结果，生成最终的增量规划结果

        Args:
            request: 原始请求
            phase1: 阶段1结果
            phase2: 阶段2结果

        Returns:
            IncrementalPlanResult: 最终结果
        """
        # 合并任务
        all_new_tasks = phase1.new_tasks + phase2.new_tasks

        # 生成合并后的调度结果
        merged_schedule = self._merge_schedules(
            request.existing_schedule,
            all_new_tasks,
            phase2.preempted_tasks,
            phase2.rescheduled_tasks
        )

        # 计算资源变化
        resource_delta = self._calculate_resource_delta(
            all_new_tasks, phase2.preempted_tasks, phase2.rescheduled_tasks
        )

        total_new = len(all_new_tasks)
        total_failed = len(phase2.failed_targets)

        logger.info(f"Aggressive planning completed: "
                   f"{total_new} new, {len(phase2.preempted_tasks)} preempted, "
                   f"{len(phase2.rescheduled_tasks)} rescheduled, {total_failed} failed")

        return IncrementalPlanResult(
            merged_schedule=merged_schedule,
            new_tasks=all_new_tasks,
            preempted_tasks=phase2.preempted_tasks,
            rescheduled_tasks=phase2.rescheduled_tasks,
            failed_targets=phase2.failed_targets,
            resource_usage_delta=resource_delta,
            strategy_used=IncrementalStrategyType.AGGRESSIVE,
            statistics={
                'total_targets': len(request.new_targets),
                'scheduled_count': total_new,
                'preempted_count': len(phase2.preempted_tasks),
                'rescheduled_count': len(phase2.rescheduled_tasks),
                'failed_count': total_failed,
                'success_rate': total_new / len(request.new_targets) if request.new_targets else 0.0,
                'preemption_count': len(phase2.preempted_tasks),
                'cascade_count': 0,
                'strategy': 'aggressive'
            }
        )
