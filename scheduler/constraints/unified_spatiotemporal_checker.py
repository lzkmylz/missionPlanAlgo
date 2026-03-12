"""
统一时空约束检查器

将时间窗口冲突、机动能力、机动时间三个约束合并为统一的时空约束检查。

核心思想：
- 卫星执行任务是连续的时空过程
- 任务A结束后需要机动时间才能开始任务B
- 实际开始时间 = max(窗口开始, 前一任务结束 + 机动时间)
- 实际结束时间 = 实际开始 + 成像时长
- 冲突检查：实际结束时间必须 <= 窗口结束时间

使用统一的时空约束检查可以：
1. 避免分别检查三个约束导致的不一致
2. 简化调度逻辑
3. 提供更准确的冲突诊断信息
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import logging

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.mission import Mission
from scheduler.constraints import SlewConstraintChecker, SlewFeasibilityResult

logger = logging.getLogger(__name__)


@dataclass
class SpatiotemporalCheckResult:
    """时空约束检查结果

    Attributes:
        feasible: 是否可行
        actual_start: 实际可开始时间（考虑了机动时间）
        actual_end: 实际结束时间
        slew_angle: 机动角度（度）
        slew_time: 机动时间（秒）
        window_available: 时间窗口是否足够
        slew_feasible: 机动是否可行
        conflict_reason: 冲突原因（如果不可行）
        conflict_with: 与哪个任务冲突（任务ID）
    """
    feasible: bool
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    slew_angle: float = 0.0
    slew_time: float = 0.0
    window_available: bool = True
    slew_feasible: bool = True
    conflict_reason: Optional[str] = None
    conflict_with: Optional[str] = None


@dataclass
class ScheduledTaskInfo:
    """已调度任务信息（用于冲突检测）"""
    task_id: str
    target_id: str
    start_time: datetime
    end_time: datetime
    satellite_id: str


class UnifiedSpatiotemporalChecker:
    """统一时空约束检查器

    将时间冲突、机动能力、机动时间约束统一处理。

    时空约束检查流程：
    1. 计算机动需求（角度和时间）
    2. 计算实际可开始时间
    3. 计算实际结束时间
    4. 检查是否超出窗口边界
    5. 检查与已调度任务的时间重叠
    """

    def __init__(
        self,
        mission: Mission,
        slew_checker: Optional[SlewConstraintChecker] = None
    ):
        """初始化统一时空约束检查器

        Args:
            mission: 任务对象
            slew_checker: 机动约束检查器（可选）
        """
        self.mission = mission
        self._slew_checker = slew_checker

        # 每个卫星的已调度任务列表
        self._scheduled_tasks: Dict[str, List[ScheduledTaskInfo]] = {}

        # 每个卫星的最后一个任务结束时间和目标
        self._sat_last_info: Dict[str, Dict[str, Any]] = {}

        # 初始化
        for sat in mission.satellites:
            self._scheduled_tasks[sat.id] = []
            self._sat_last_info[sat.id] = {
                'last_end_time': mission.start_time,
                'last_target': None
            }

    def set_slew_checker(self, slew_checker: SlewConstraintChecker) -> None:
        """设置机动约束检查器"""
        self._slew_checker = slew_checker

    def reset(self) -> None:
        """重置所有卫星的调度状态"""
        for sat_id in self._scheduled_tasks:
            self._scheduled_tasks[sat_id] = []
            self._sat_last_info[sat_id] = {
                'last_end_time': self.mission.start_time,
                'last_target': None
            }

    def reset_satellite(self, satellite_id: str) -> None:
        """重置指定卫星的调度状态"""
        if satellite_id in self._scheduled_tasks:
            self._scheduled_tasks[satellite_id] = []
            self._sat_last_info[satellite_id] = {
                'last_end_time': self.mission.start_time,
                'last_target': None
            }

    def check_task_placement(
        self,
        satellite_id: str,
        target: Target,
        window_start: datetime,
        window_end: datetime,
        imaging_duration: float,
        task_id: Optional[str] = None
    ) -> SpatiotemporalCheckResult:
        """检查任务放置的时空可行性

        这是统一的约束检查入口，同时考虑：
        - 机动时间和能力
        - 时间窗口边界
        - 与已调度任务的冲突

        Args:
            satellite_id: 卫星ID
            target: 目标对象
            window_start: 可见窗口开始时间
            window_end: 可见窗口结束时间
            imaging_duration: 成像时长（秒）
            task_id: 任务ID（用于冲突诊断）

        Returns:
            SpatiotemporalCheckResult: 检查结果
        """
        # 1. 获取卫星信息
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            return SpatiotemporalCheckResult(
                feasible=False,
                conflict_reason=f"Satellite {satellite_id} not found"
            )

        # 2. 获取前一个任务信息
        last_info = self._sat_last_info.get(satellite_id, {})
        prev_target = last_info.get('last_target')
        prev_end_time = last_info.get('last_end_time', self.mission.start_time)

        # 3. 检查机动可行性（计算机动角度和时间）
        slew_result = self._check_slew(
            satellite_id, prev_target, target, prev_end_time, window_start
        )

        if not slew_result.feasible:
            return SpatiotemporalCheckResult(
                feasible=False,
                slew_angle=slew_result.slew_angle,
                slew_time=slew_result.slew_time,
                slew_feasible=False,
                conflict_reason=f"Slew not feasible: {slew_result.reason}"
            )

        # 4. 计算实际可开始时间
        # 实际开始 = max(窗口开始, 前一任务结束 + 机动时间)
        actual_start = max(window_start, slew_result.actual_start)

        # 5. 计算实际结束时间
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        # 6. 检查窗口是否足够
        if actual_end > window_end:
            return SpatiotemporalCheckResult(
                feasible=False,
                actual_start=actual_start,
                actual_end=actual_end,
                slew_angle=slew_result.slew_angle,
                slew_time=slew_result.slew_time,
                window_available=False,
                conflict_reason=f"Window too short: need {imaging_duration}s, "
                               f"but only {(window_end - actual_start).total_seconds():.1f}s available"
            )

        # 7. 检查与已调度任务的时间重叠
        conflict_task = self._find_time_conflict(satellite_id, actual_start, actual_end)
        if conflict_task:
            return SpatiotemporalCheckResult(
                feasible=False,
                actual_start=actual_start,
                actual_end=actual_end,
                slew_angle=slew_result.slew_angle,
                slew_time=slew_result.slew_time,
                window_available=True,
                conflict_reason=f"Time conflict with task {conflict_task.task_id}",
                conflict_with=conflict_task.task_id
            )

        # 8. 所有检查通过
        return SpatiotemporalCheckResult(
            feasible=True,
            actual_start=actual_start,
            actual_end=actual_end,
            slew_angle=slew_result.slew_angle,
            slew_time=slew_result.slew_time,
            window_available=True,
            slew_feasible=True
        )

    def commit_task(
        self,
        satellite_id: str,
        task_id: str,
        target_id: str,
        actual_start: datetime,
        actual_end: datetime,
        target: Optional[Target] = None
    ) -> None:
        """提交任务，更新卫星调度状态

        Args:
            satellite_id: 卫星ID
            task_id: 任务ID
            target_id: 目标ID
            actual_start: 实际开始时间
            actual_end: 实际结束时间
            target: 目标对象（用于更新last_target）
        """
        # 添加到已调度任务列表
        task_info = ScheduledTaskInfo(
            task_id=task_id,
            target_id=target_id,
            start_time=actual_start,
            end_time=actual_end,
            satellite_id=satellite_id
        )
        self._scheduled_tasks[satellite_id].append(task_info)

        # 按时间排序
        self._scheduled_tasks[satellite_id].sort(key=lambda x: x.start_time)

        # 更新最后任务信息
        if satellite_id in self._sat_last_info:
            self._sat_last_info[satellite_id]['last_end_time'] = actual_end
            if target:
                self._sat_last_info[satellite_id]['last_target'] = target

    def remove_task(self, satellite_id: str, task_id: str) -> bool:
        """移除已调度任务

        Args:
            satellite_id: 卫星ID
            task_id: 任务ID

        Returns:
            bool: 是否成功移除
        """
        if satellite_id not in self._scheduled_tasks:
            return False

        tasks = self._scheduled_tasks[satellite_id]
        for i, task in enumerate(tasks):
            if task.task_id == task_id:
                tasks.pop(i)
                # 更新最后任务信息
                if tasks:
                    last_task = tasks[-1]
                    self._sat_last_info[satellite_id]['last_end_time'] = last_task.end_time
                else:
                    self._sat_last_info[satellite_id]['last_end_time'] = self.mission.start_time
                    self._sat_last_info[satellite_id]['last_target'] = None
                return True
        return False

    def get_scheduled_tasks(self, satellite_id: str) -> List[ScheduledTaskInfo]:
        """获取卫星的已调度任务列表"""
        return self._scheduled_tasks.get(satellite_id, [])

    def get_satellite_timeline(self, satellite_id: str) -> List[Tuple[datetime, datetime, str]]:
        """获取卫星的时间线

        Returns:
            List[Tuple[start, end, task_id]]: 时间线列表
        """
        return [
            (t.start_time, t.end_time, t.task_id)
            for t in self._scheduled_tasks.get(satellite_id, [])
        ]

    def _check_slew(
        self,
        satellite_id: str,
        prev_target: Optional[Target],
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime
    ) -> SlewFeasibilityResult:
        """检查机动可行性（高精度要求：始终使用精确检查器）"""
        if self._slew_checker:
            return self._slew_checker.check_slew_feasibility(
                satellite_id, prev_target, current_target,
                prev_end_time, window_start
            )

        # 高精度要求：如果没有精确检查器则报错
        raise RuntimeError(
            "Slew checker not initialized. "
            "High precision mode requires SlewConstraintChecker."
        )

    def _find_time_conflict(
        self,
        satellite_id: str,
        start: datetime,
        end: datetime
    ) -> Optional[ScheduledTaskInfo]:
        """查找时间冲突的任务

        Args:
            satellite_id: 卫星ID
            start: 新任务开始时间
            end: 新任务结束时间

        Returns:
            冲突的任务信息，如果没有冲突则返回None
        """
        for task in self._scheduled_tasks.get(satellite_id, []):
            # 检查是否有重叠
            # 不重叠的条件：新任务在旧任务之前结束，或在旧任务之后开始
            if not (end <= task.start_time or start >= task.end_time):
                return task
        return None

    def check_batch_placement(
        self,
        placements: List[Dict[str, Any]]
    ) -> Tuple[bool, List[SpatiotemporalCheckResult]]:
        """批量检查任务放置

        用于元启发式算法评估多个任务的可行性

        Args:
            placements: 放置列表，每个元素包含：
                - satellite_id: 卫星ID
                - target: 目标对象
                - window_start: 窗口开始
                - window_end: 窗口结束
                - imaging_duration: 成像时长
                - task_id: 任务ID（可选）

        Returns:
            Tuple[全部可行, 各任务检查结果]
        """
        results = []
        all_feasible = True

        # 保存当前状态
        saved_tasks = {k: v.copy() for k, v in self._scheduled_tasks.items()}
        saved_last_info = {k: v.copy() for k, v in self._sat_last_info.items()}

        try:
            for placement in placements:
                result = self.check_task_placement(
                    satellite_id=placement['satellite_id'],
                    target=placement['target'],
                    window_start=placement['window_start'],
                    window_end=placement['window_end'],
                    imaging_duration=placement['imaging_duration'],
                    task_id=placement.get('task_id')
                )
                results.append(result)

                if result.feasible:
                    # 临时提交，供后续任务检查
                    self.commit_task(
                        satellite_id=placement['satellite_id'],
                        task_id=placement.get('task_id', 'temp'),
                        target_id=placement['target'].id,
                        actual_start=result.actual_start,
                        actual_end=result.actual_end,
                        target=placement['target']
                    )
                else:
                    all_feasible = False
        finally:
            # 恢复状态
            self._scheduled_tasks = saved_tasks
            self._sat_last_info = saved_last_info

        return all_feasible, results
