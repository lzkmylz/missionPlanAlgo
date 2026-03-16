"""
增量规划状态管理器

负责：
1. 从ScheduleResult恢复调度状态
2. 计算和管理剩余资源窗口
3. 保存和恢复增量规划状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from copy import deepcopy
import logging

if TYPE_CHECKING:
    from ..base_scheduler import ScheduledTask

logger = logging.getLogger(__name__)

# 默认卫星容量常量
DEFAULT_POWER_CAPACITY = 2800.0  # Wh
DEFAULT_STORAGE_CAPACITY = 128.0  # GB
DEFAULT_INITIAL_POWER_RATIO = 0.9  # 初始电量占容量比例


@dataclass
class ResourceWindow:
    """资源可用窗口"""
    start_time: datetime
    end_time: datetime
    available_power: float          # 可用电量 (Wh)
    available_storage: float        # 可用存储 (GB)
    satellite_id: str
    quality_score: float = 1.0      # 窗口质量评分 (0-1)

    def duration(self) -> float:
        """窗口持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()

    def can_accommodate(self, required_duration: float,
                       required_power: float,
                       required_storage: float) -> bool:
        """检查窗口是否能容纳任务"""
        return (self.duration() >= required_duration and
                self.available_power >= required_power and
                self.available_storage >= required_storage)

    def split_for_task(self, task_start: datetime, task_end: datetime) -> Tuple[Optional['ResourceWindow'], Optional['ResourceWindow']]:
        """任务占用后，分割窗口为前后两部分"""
        before = None
        after = None

        if task_start > self.start_time:
            before = ResourceWindow(
                start_time=self.start_time,
                end_time=task_start,
                available_power=self.available_power,
                available_storage=self.available_storage,
                satellite_id=self.satellite_id,
                quality_score=self.quality_score
            )

        if task_end < self.end_time:
            after = ResourceWindow(
                start_time=task_end,
                end_time=self.end_time,
                available_power=self.available_power,
                available_storage=self.available_storage,
                satellite_id=self.satellite_id,
                quality_score=self.quality_score
            )

        return before, after


@dataclass
class SatelliteState:
    """单个卫星的调度状态"""
    satellite_id: str
    scheduled_tasks: List[Dict[str, Any]]  # 已规划任务列表（按时间排序）
    current_power: float                   # 当前电量
    current_storage: float                 # 当前存储占用
    power_capacity: float                  # 电量容量
    storage_capacity: float                # 存储容量

    def __post_init__(self):
        """初始化后按时间排序任务"""
        self.scheduled_tasks.sort(key=lambda t: t['imaging_start'])

    def get_resource_at_time(self, time: datetime) -> Tuple[float, float]:
        """获取指定时刻的资源状态"""
        power = self.current_power
        storage = self.current_storage

        for task in self.scheduled_tasks:
            if task['imaging_end'] <= time:
                # 任务已完成，考虑资源恢复
                power = min(self.power_capacity,
                          power + task.get('power_generated', 0) - task.get('power_consumed', 0))
                storage = max(0, storage - task.get('storage_released', 0))
            elif task['imaging_start'] <= time < task['imaging_end']:
                # 任务进行中，线性插值
                progress = (time - task['imaging_start']).total_seconds() / \
                          (task['imaging_end'] - task['imaging_start']).total_seconds()
                power -= task.get('power_consumed', 0) * progress
                storage += task.get('storage_produced', 0) * progress

        return power, storage

    def find_resource_windows(self, start_time: datetime, end_time: datetime) -> List[ResourceWindow]:
        """查找指定时间段内的资源可用窗口"""
        windows = []
        current_time = start_time

        # 遍历任务间隙
        for i, task in enumerate(self.scheduled_tasks):
            task_start = task['imaging_start']
            task_end = task['imaging_end']

            if task_start > current_time:
                # 发现间隙
                window_end = min(task_start, end_time)
                if current_time < window_end:
                    power, storage = self.get_resource_at_time(current_time)
                    windows.append(ResourceWindow(
                        start_time=current_time,
                        end_time=window_end,
                        available_power=power,
                        available_storage=self.storage_capacity - storage,
                        satellite_id=self.satellite_id
                    ))

            current_time = max(current_time, task_end)

            if current_time >= end_time:
                break

        # 最后一个任务之后到end_time的窗口
        if current_time < end_time:
            power, storage = self.get_resource_at_time(current_time)
            windows.append(ResourceWindow(
                start_time=current_time,
                end_time=end_time,
                available_power=power,
                available_storage=self.storage_capacity - storage,
                satellite_id=self.satellite_id
            ))

        return windows


class IncrementalState:
    """
    增量规划状态管理器

    负责从ScheduleResult重建调度状态，并提供状态查询接口
    """

    def __init__(self, mission: Any):
        """
        初始化增量状态管理器

        Args:
            mission: 任务场景对象
        """
        self.mission = mission
        self.satellite_states: Dict[str, SatelliteState] = {}
        self._original_schedule: Optional[Any] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    def load_from_schedule(self, schedule_result: Any) -> None:
        """
        从调度结果加载状态

        Args:
            schedule_result: ScheduleResult对象
        """
        from ..base_scheduler import ScheduleResult

        if not isinstance(schedule_result, ScheduleResult):
            raise TypeError(f"Expected ScheduleResult, got {type(schedule_result)}")

        self._original_schedule = schedule_result
        self.satellite_states = {}

        # 按卫星分组任务
        tasks_by_sat: Dict[str, List[Dict]] = {}
        for task in schedule_result.scheduled_tasks:
            sat_id = task.satellite_id
            if sat_id not in tasks_by_sat:
                tasks_by_sat[sat_id] = []
            tasks_by_sat[sat_id].append({
                'task_id': task.task_id,
                'target_id': task.target_id,
                'imaging_start': task.imaging_start,
                'imaging_end': task.imaging_end,
                'imaging_mode': task.imaging_mode,
                'power_consumed': task.power_consumed,
                'power_generated': task.power_generated,
                'storage_produced': task.storage_after - task.storage_before,
                'storage_released': 0.0,  # 数传后释放的存储
                'priority': getattr(task, 'priority', 0),
            })

        # 获取卫星容量信息
        sat_capacity = {}
        if self.mission and hasattr(self.mission, 'satellites'):
            for sat in self.mission.satellites:
                caps = getattr(sat, 'capabilities', None)
                if caps:
                    sat_capacity[sat.id] = {
                        'power_capacity': getattr(caps, 'power_capacity', DEFAULT_POWER_CAPACITY),
                        'storage_capacity': getattr(caps, 'storage_capacity', DEFAULT_STORAGE_CAPACITY),
                    }

        # 创建卫星状态对象
        for sat_id, tasks in tasks_by_sat.items():
            capacity = sat_capacity.get(sat_id, {
                'power_capacity': DEFAULT_POWER_CAPACITY,
                'storage_capacity': DEFAULT_STORAGE_CAPACITY
            })

            # 计算初始电量和存储（从第一个任务前状态推断）
            if tasks:
                first_task = min(tasks, key=lambda t: t['imaging_start'])
                initial_power = first_task.get('power_before', capacity['power_capacity'] * DEFAULT_INITIAL_POWER_RATIO)
                initial_storage = first_task.get('storage_before', 0.0)
            else:
                initial_power = capacity['power_capacity'] * 0.9
                initial_storage = 0.0

            self.satellite_states[sat_id] = SatelliteState(
                satellite_id=sat_id,
                scheduled_tasks=tasks,
                current_power=initial_power,
                current_storage=initial_storage,
                power_capacity=capacity['power_capacity'],
                storage_capacity=capacity['storage_capacity']
            )

        # 记录时间范围
        if schedule_result.scheduled_tasks:
            starts = [t.imaging_start for t in schedule_result.scheduled_tasks]
            ends = [t.imaging_end for t in schedule_result.scheduled_tasks]
            self._start_time = min(starts)
            self._end_time = max(ends)

        logger.info(f"Loaded state from schedule: {len(schedule_result.scheduled_tasks)} tasks, "
                   f"{len(self.satellite_states)} satellites")

    def get_satellite_state(self, satellite_id: str) -> Optional[SatelliteState]:
        """获取指定卫星的状态"""
        return self.satellite_states.get(satellite_id)

    def get_all_satellite_ids(self) -> List[str]:
        """获取所有卫星ID"""
        return list(self.satellite_states.keys())

    def get_available_windows(self, satellite_id: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[ResourceWindow]:
        """
        获取指定卫星的资源可用窗口

        Args:
            satellite_id: 卫星ID
            start_time: 查询起始时间（默认场景开始）
            end_time: 查询结束时间（默认场景结束）

        Returns:
            List[ResourceWindow]: 可用窗口列表
        """
        sat_state = self.satellite_states.get(satellite_id)
        if not sat_state:
            return []

        start = start_time or self._start_time or self.mission.start_time
        end = end_time or self._end_time or self.mission.end_time

        return sat_state.find_resource_windows(start, end)

    def get_all_available_windows(self, start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Dict[str, List[ResourceWindow]]:
        """获取所有卫星的可用窗口"""
        return {
            sat_id: self.get_available_windows(sat_id, start_time, end_time)
            for sat_id in self.satellite_states.keys()
        }

    def add_task(self, satellite_id: str, task_info: Dict[str, Any]) -> bool:
        """
        添加新任务到状态（用于增量规划）

        Args:
            satellite_id: 卫星ID
            task_info: 任务信息字典

        Returns:
            bool: 是否添加成功
        """
        sat_state = self.satellite_states.get(satellite_id)
        if not sat_state:
            logger.warning(f"Satellite {satellite_id} not found in state")
            return False

        sat_state.scheduled_tasks.append(task_info)
        sat_state.scheduled_tasks.sort(key=lambda t: t['imaging_start'])

        # 更新资源状态
        sat_state.current_power -= task_info.get('power_consumed', 0)
        sat_state.current_storage += task_info.get('storage_produced', 0)

        return True

    def remove_task(self, satellite_id: str, task_id: str) -> bool:
        """
        从状态中移除任务（用于抢占）

        Args:
            satellite_id: 卫星ID
            task_id: 任务ID

        Returns:
            bool: 是否移除成功
        """
        sat_state = self.satellite_states.get(satellite_id)
        if not sat_state:
            return False

        for i, task in enumerate(sat_state.scheduled_tasks):
            if task['task_id'] == task_id:
                removed_task = sat_state.scheduled_tasks.pop(i)
                # 恢复资源
                sat_state.current_power += removed_task.get('power_consumed', 0)
                sat_state.current_storage -= removed_task.get('storage_produced', 0)
                return True

        return False

    def to_schedule_result(self, original_result: Any,
                          new_tasks: List[ScheduledTask],
                          preempted_tasks: List[ScheduledTask]) -> ScheduleResult:
        """
        将当前状态转换为ScheduleResult

        从当前卫星状态重建所有任务，合并新增任务，移除被抢占任务

        Args:
            original_result: 原始调度结果
            new_tasks: 新增任务列表
            preempted_tasks: 被抢占任务列表

        Returns:
            ScheduleResult: 合并后的结果
        """
        from ..base_scheduler import ScheduleResult, ScheduledTask

        # 被抢占任务ID集合
        preempted_ids = {t.task_id for t in preempted_tasks}

        # 从当前状态重建所有任务
        all_tasks: List[ScheduledTask] = []
        for sat_id, sat_state in self.satellite_states.items():
            for task_info in sat_state.scheduled_tasks:
                task_id = task_info['task_id']

                # 跳过被抢占的任务
                if task_id in preempted_ids:
                    continue

                # 重建 ScheduledTask
                task = ScheduledTask(
                    task_id=task_id,
                    satellite_id=sat_id,
                    target_id=task_info['target_id'],
                    imaging_start=task_info['imaging_start'],
                    imaging_end=task_info['imaging_end'],
                    imaging_mode=task_info.get('imaging_mode', 'standard'),
                    power_consumed=task_info.get('power_consumed', 0.0),
                    priority=task_info.get('priority', 0)
                )
                all_tasks.append(task)

        # 添加新任务
        all_tasks.extend(new_tasks)

        # 按时间排序
        all_tasks.sort(key=lambda t: t.imaging_start)

        # 计算合并后的makespan
        makespan = 0.0
        if all_tasks:
            start = min(t.imaging_start for t in all_tasks)
            end = max(t.imaging_end for t in all_tasks)
            makespan = (end - start).total_seconds()

        return ScheduleResult(
            scheduled_tasks=all_tasks,
            unscheduled_tasks=original_result.unscheduled_tasks,
            makespan=makespan,
            computation_time=original_result.computation_time,
            iterations=original_result.iterations,
            convergence_curve=getattr(original_result, 'convergence_curve', [])
        )

    def create_snapshot(self) -> Dict[str, Any]:
        """创建状态快照（用于回滚）"""
        return {
            'satellite_states': deepcopy(self.satellite_states),
            'start_time': self._start_time,
            'end_time': self._end_time
        }

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """从快照恢复状态"""
        self.satellite_states = deepcopy(snapshot['satellite_states'])
        self._start_time = snapshot['start_time']
        self._end_time = snapshot['end_time']
