"""
事件驱动调度器

实现第19章设计的事件驱动调度功能，用于响应实时事件并触发动态重调度。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Set, Tuple
import copy

from scheduler.base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask


class EventType(Enum):
    """
    调度事件类型

    Attributes:
        NEW_URGENT_TASK: 新的紧急任务（如灾害响应）
        TASK_CANCELLED: 任务取消
        SATELLITE_FAILURE: 卫星故障
        RESOURCE_DEGRADATION: 资源降级（电量、存储等）
        ROLLING_HORIZON_TRIGGER: 滚动时间窗触发
    """
    NEW_URGENT_TASK = "new_urgent_task"
    TASK_CANCELLED = "task_cancelled"
    SATELLITE_FAILURE = "satellite_failure"
    RESOURCE_DEGRADATION = "resource_degradation"
    ROLLING_HORIZON_TRIGGER = "rolling_horizon_trigger"


@dataclass
class ScheduleEvent:
    """
    调度事件

    Attributes:
        event_type: 事件类型
        timestamp: 事件发生时间戳
        priority: 优先级（1-10，10为最高）
        description: 事件描述
        payload: 事件详细信息（字典）
        affected_satellites: 受影响的卫星ID列表
        affected_tasks: 受影响的任务ID列表
    """
    event_type: EventType
    timestamp: datetime
    priority: int
    description: str
    payload: Dict[str, Any]
    affected_satellites: Optional[List[str]] = None
    affected_tasks: Optional[List[str]] = None

    def __post_init__(self):
        """验证优先级范围"""
        if not 1 <= self.priority <= 10:
            raise ValueError(f"Priority must be between 1 and 10, got {self.priority}")

    def __eq__(self, other):
        """比较两个事件是否相等"""
        if not isinstance(other, ScheduleEvent):
            return False
        return (
            self.event_type == other.event_type and
            self.timestamp == other.timestamp and
            self.priority == other.priority and
            self.description == other.description and
            self.payload == other.payload and
            self.affected_satellites == other.affected_satellites and
            self.affected_tasks == other.affected_tasks
        )


@dataclass
class DisruptionImpact:
    """
    扰动影响

    Attributes:
        severity: 严重程度（'minor', 'moderate', 'major'）
        affected_satellites: 受影响的卫星ID列表
        affected_tasks: 受影响的任务ID列表
        affected_ratio: 受影响任务比例（0-1）
    """
    severity: str
    affected_satellites: List[str]
    affected_tasks: List[str]
    affected_ratio: float

    def __post_init__(self):
        """验证严重程度值"""
        valid_severities = ['minor', 'moderate', 'major']
        if self.severity not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}, got {self.severity}")


class EventDrivenScheduler:
    """
    事件驱动调度器 - 响应实时事件，触发动态重调度

    基于当前计划和新事件，选择适当的重调度策略：
    - 轻微影响：局部修复
    - 中等影响：滚动优化
    - 严重影响：全局重调度

    Example:
        >>> from scheduler.greedy.greedy_scheduler import GreedyScheduler
        >>> base_scheduler = GreedyScheduler()
        >>> event_scheduler = EventDrivenScheduler(base_scheduler)
        >>>
        >>> event = ScheduleEvent(
        ...     event_type=EventType.NEW_URGENT_TASK,
        ...     timestamp=datetime.now(timezone.utc),
        ...     priority=9,
        ...     description="紧急灾害响应",
        ...     payload={"task_id": "URGENT-001"}
        ... )
        >>> event_scheduler.submit_event(event)
        >>>
        >>> new_plan = event_scheduler.reschedule(current_plan, [event])
    """

    # 高优先级阈值（优先级大于等于此值立即触发重调度）
    HIGH_PRIORITY_THRESHOLD = 8

    # 影响程度判定阈值
    MINOR_IMPACT_RATIO = 0.1
    MODERATE_IMPACT_RATIO = 0.3
    MINOR_SATELLITE_COUNT = 1
    MODERATE_SATELLITE_COUNT = 3

    # 滚动优化窗口大小
    ROLLING_HORIZON_HOURS = 2

    def __init__(self, base_scheduler: BaseScheduler):
        """
        初始化事件驱动调度器

        Args:
            base_scheduler: 基础调度器，用于全局重调度

        Raises:
            TypeError: 如果没有提供base_scheduler
        """
        if base_scheduler is None:
            raise TypeError("base_scheduler is required")

        self.base_scheduler: BaseScheduler = base_scheduler
        self.current_plan: Optional[ScheduleResult] = None
        self.event_queue: List[ScheduleEvent] = []
        self.plan_history: List[Tuple[datetime, ScheduleResult, List[ScheduleEvent]]] = []

    def submit_event(self, event: ScheduleEvent) -> bool:
        """
        提交调度事件

        将事件添加到事件队列，并根据优先级决定是否立即触发重调度。
        事件队列按优先级降序排列，相同优先级按时间戳升序排列。

        Args:
            event: 调度事件

        Returns:
            如果高优先级事件立即触发重调度返回True，否则返回False
        """
        self.event_queue.append(event)

        # 按优先级降序，相同优先级按时间戳升序
        self.event_queue.sort(key=lambda e: (-e.priority, e.timestamp))

        # 高优先级事件立即触发
        if event.priority >= self.HIGH_PRIORITY_THRESHOLD:
            return True
        return False

    def reschedule(
        self,
        current_plan: ScheduleResult,
        new_events: List[ScheduleEvent]
    ) -> ScheduleResult:
        """
        重调度接口（核心方法）

        基于当前计划和新事件，进行局部修复而非全盘重算。
        根据事件影响的严重程度选择不同的重调度策略。

        Args:
            current_plan: 当前调度计划
            new_events: 新发生的调度事件列表

        Returns:
            重调度后的新计划
        """
        if not new_events:
            # 没有事件，返回原计划
            return copy.deepcopy(current_plan)

        # 1. 分析事件影响
        impact = self._analyze_impact(current_plan, new_events)

        # 2. 决定重调度策略
        if impact.severity == 'minor':
            return self._local_repair(current_plan, impact)
        elif impact.severity == 'moderate':
            return self._rolling_reoptimize(current_plan, impact)
        else:
            return self._global_reschedule(current_plan, impact)

    def _analyze_impact(
        self,
        current_plan: ScheduleResult,
        events: List[ScheduleEvent]
    ) -> DisruptionImpact:
        """
        分析事件对当前计划的影响

        根据受影响的卫星数量和任务比例判断严重程度。

        Args:
            current_plan: 当前调度计划
            events: 事件列表

        Returns:
            扰动影响分析结果
        """
        affected_satellites: Set[str] = set()
        affected_tasks: Set[str] = set()

        for event in events:
            if event.affected_satellites:
                affected_satellites.update(event.affected_satellites)
            if event.affected_tasks:
                affected_tasks.update(event.affected_tasks)

        # 计算受影响比例
        total_tasks = len(current_plan.scheduled_tasks)
        affected_ratio = len(affected_tasks) / max(total_tasks, 1)

        # 判断严重程度
        # 使用 <= 确保边界值被正确分类
        if affected_ratio <= self.MINOR_IMPACT_RATIO and len(affected_satellites) <= self.MINOR_SATELLITE_COUNT:
            severity = 'minor'
        elif affected_ratio <= self.MODERATE_IMPACT_RATIO and len(affected_satellites) <= self.MODERATE_SATELLITE_COUNT:
            severity = 'moderate'
        else:
            severity = 'major'

        return DisruptionImpact(
            severity=severity,
            affected_satellites=list(affected_satellites),
            affected_tasks=list(affected_tasks),
            affected_ratio=affected_ratio
        )

    def _local_repair(
        self,
        current_plan: ScheduleResult,
        impact: DisruptionImpact
    ) -> ScheduleResult:
        """
        局部修复：只调整受影响的部分任务

        适用于轻微影响的情况，通过移除受影响任务并尝试重新插入来修复计划。

        Args:
            current_plan: 当前调度计划
            impact: 扰动影响分析结果

        Returns:
            修复后的计划
        """
        # 1. 移除受影响的任务
        repaired_plan = self._remove_affected_tasks(current_plan, impact.affected_tasks)

        # 2. 尝试重新调度这些任务
        for task_id in impact.affected_tasks:
            task = self._get_task_by_id_from_plan(current_plan, task_id)
            if task:
                success = self._insert_task_greedily(repaired_plan, task)
                if not success:
                    self._record_repair_failure(task_id, "Local repair failed")

        return repaired_plan

    def _rolling_reoptimize(
        self,
        current_plan: ScheduleResult,
        impact: DisruptionImpact
    ) -> ScheduleResult:
        """
        滚动优化：在滚动时间窗内重新优化

        适用于中等影响的情况，冻结即将执行的任务，在窗口内重新优化。

        Args:
            current_plan: 当前调度计划
            impact: 扰动影响分析结果

        Returns:
            重新优化后的计划
        """
        now = datetime.now(timezone.utc)
        horizon_end = now + timedelta(hours=self.ROLLING_HORIZON_HOURS)

        # 1. 冻结已执行或即将执行的任务（不可更改）
        freeze_until = now + timedelta(minutes=5)
        frozen_tasks = self._get_frozen_tasks(current_plan, freeze_until)

        # 2. 获取窗口内的任务
        window_tasks = self._get_tasks_in_window(current_plan, now, horizon_end)
        window_tasks = [t for t in window_tasks if t.task_id not in frozen_tasks]

        # 3. 创建新计划，保留冻结任务和窗外任务
        new_plan = self._remove_tasks(current_plan, [t.task_id for t in window_tasks])

        # 4. 对窗口内任务进行贪心插入
        for task in window_tasks:
            self._insert_task_greedily(new_plan, task)

        return new_plan

    def _global_reschedule(
        self,
        current_plan: ScheduleResult,
        impact: DisruptionImpact
    ) -> ScheduleResult:
        """
        全局重调度：使用基础调度器重新计算

        适用于严重影响的情况，调用基础调度器进行全局重新调度。

        Args:
            current_plan: 当前调度计划（用于参考）
            impact: 扰动影响分析结果

        Returns:
            全局重调度后的新计划
        """
        # 调用基础调度器进行全局重调度
        return self.base_scheduler.schedule()

    def _remove_affected_tasks(
        self,
        current_plan: ScheduleResult,
        affected_task_ids: List[str]
    ) -> ScheduleResult:
        """
        移除受影响的任务

        从当前计划中移除指定的任务，返回新的计划。

        Args:
            current_plan: 当前调度计划
            affected_task_ids: 受影响任务的ID列表

        Returns:
            移除任务后的新计划
        """
        affected_set = set(affected_task_ids)

        # 过滤掉受影响的任务
        new_scheduled_tasks = [
            task for task in current_plan.scheduled_tasks
            if task.task_id not in affected_set
        ]

        # 创建新的ScheduleResult
        return ScheduleResult(
            scheduled_tasks=new_scheduled_tasks,
            unscheduled_tasks=copy.deepcopy(current_plan.unscheduled_tasks),
            makespan=current_plan.makespan,
            computation_time=current_plan.computation_time,
            iterations=current_plan.iterations,
            convergence_curve=copy.deepcopy(current_plan.convergence_curve),
            failure_summary=copy.deepcopy(current_plan.failure_summary)
        )

    def _get_frozen_tasks(
        self,
        current_plan: ScheduleResult,
        freeze_until: datetime
    ) -> Set[str]:
        """
        获取冻结任务

        获取在指定时间之前开始或正在执行的任务ID集合，这些任务不可更改。

        Args:
            current_plan: 当前调度计划
            freeze_until: 冻结截止时间

        Returns:
            冻结任务的ID集合
        """
        frozen_tasks: Set[str] = set()

        for task in current_plan.scheduled_tasks:
            # 任务开始时间在冻结时间之前，或者任务正在进行中
            if task.imaging_start <= freeze_until:
                frozen_tasks.add(task.task_id)

        return frozen_tasks

    def _get_tasks_in_window(
        self,
        current_plan: ScheduleResult,
        window_start: datetime,
        window_end: datetime
    ) -> List[ScheduledTask]:
        """
        获取窗口内的任务

        获取在指定时间窗口内的所有任务。

        Args:
            current_plan: 当前调度计划
            window_start: 窗口开始时间
            window_end: 窗口结束时间

        Returns:
            窗口内的任务列表
        """
        return [
            task for task in current_plan.scheduled_tasks
            if window_start <= task.imaging_start < window_end
        ]

    def _remove_tasks(
        self,
        current_plan: ScheduleResult,
        task_ids: List[str]
    ) -> ScheduleResult:
        """
        移除指定任务

        Args:
            current_plan: 当前调度计划
            task_ids: 要移除的任务ID列表

        Returns:
            移除任务后的新计划
        """
        return self._remove_affected_tasks(current_plan, task_ids)

    def _insert_task_greedily(
        self,
        plan: ScheduleResult,
        task: ScheduledTask
    ) -> bool:
        """
        贪心插入任务

        尝试将任务插入到计划中，如果不与其他任务冲突则插入成功。

        Args:
            plan: 当前调度计划
            task: 要插入的任务

        Returns:
            插入成功返回True，否则返回False
        """
        # 检查是否与现有任务冲突
        for existing_task in plan.scheduled_tasks:
            if self._has_time_conflict(task, existing_task):
                return False

        # 无冲突，添加到计划
        plan.scheduled_tasks.append(task)
        return True

    def _has_time_conflict(
        self,
        task1: ScheduledTask,
        task2: ScheduledTask
    ) -> bool:
        """
        检查两个任务是否有时间冲突

        只有同一卫星上的任务才可能冲突。

        Args:
            task1: 任务1
            task2: 任务2

        Returns:
            有时间冲突返回True，否则返回False
        """
        # 不同卫星的任务不会冲突
        if task1.satellite_id != task2.satellite_id:
            return False

        # 检查时间重叠
        return (
            task1.imaging_start < task2.imaging_end and
            task1.imaging_end > task2.imaging_start
        )

    def _get_task_by_id_from_plan(
        self,
        plan: ScheduleResult,
        task_id: str
    ) -> Optional[ScheduledTask]:
        """
        从计划中根据ID获取任务

        Args:
            plan: 调度计划
            task_id: 任务ID

        Returns:
            找到的任务，未找到返回None
        """
        for task in plan.scheduled_tasks:
            if task.task_id == task_id:
                return task
        return None

    def _record_repair_failure(self, task_id: str, reason: str) -> None:
        """
        记录修复失败

        Args:
            task_id: 任务ID
            reason: 失败原因
        """
        # 可以扩展为记录到日志或失败列表
        pass
