"""
处理感知调度器

实现Chapter 20.5: ProcessingAwareScheduler
将成像任务与在轨处理任务联合调度
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from scheduler.base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask
from core.processing.onboard_processing_manager import (
    OnboardProcessingManager,
    ProcessingDecision,
    SatelliteResourceState,
    DecisionContext
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingWindow:
    """处理窗口"""
    imaging_task_id: str
    satellite_id: str
    earliest_start: datetime
    latest_end: datetime
    required_duration_seconds: float
    required_storage_gb: float = 0.0
    priority: int = 5
    assigned_start: Optional[datetime] = None
    assigned_end: Optional[datetime] = None
    force_raw_downlink: bool = False


@dataclass
class ProcessingConstraint:
    """处理约束条件"""
    min_battery_soc: float = 0.35  # 最小电量SOC
    min_thermal_headroom_c: float = 15.0  # 最小热余量(℃)
    min_storage_free_gb: float = 5.0  # 最小空闲存储(GB)


@dataclass
class ScheduleWithProcessing:
    """
    带在轨处理的调度结果

    扩展ScheduleResult，添加处理窗口信息
    """
    base_schedule: ScheduleResult
    processing_windows: Dict[str, List[ProcessingWindow]] = field(default_factory=dict)
    processing_decisions: Dict[str, ProcessingDecision] = field(default_factory=dict)

    def get_processing_decision(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取指定任务的处理决策"""
        for sat_id, windows in self.processing_windows.items():
            for window in windows:
                if window.imaging_task_id == task_id:
                    return {
                        "task_id": task_id,
                        "satellite_id": sat_id,
                        "processing_start": window.assigned_start,
                        "processing_end": window.assigned_end,
                        "force_raw_downlink": window.force_raw_downlink
                    }
        return None

    def get_total_processing_time(self) -> float:
        """获取总处理时间"""
        total = 0.0
        for windows in self.processing_windows.values():
            for window in windows:
                if window.assigned_start and window.assigned_end:
                    duration = (window.assigned_end - window.assigned_start).total_seconds()
                    total += duration
        return total

    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要统计"""
        total_tasks = 0
        onboard_count = 0
        downlink_count = 0

        for windows in self.processing_windows.values():
            for window in windows:
                total_tasks += 1
                if window.force_raw_downlink:
                    downlink_count += 1
                else:
                    onboard_count += 1

        return {
            "total_tasks": total_tasks,
            "onboard_processing": onboard_count,
            "raw_downlink": downlink_count,
            "onboard_ratio": onboard_count / total_tasks if total_tasks > 0 else 0.0
        }


class ProcessingAwareScheduler:
    """
    处理感知调度器

    将成像任务与在轨处理任务联合调度

    关键约束：
    1. AI处理必须在成像完成后开始
    2. AI处理窗口不能与后续成像冲突
    3. 处理任务需要占用卫星存储
    4. 处理功耗影响后续任务电量
    """

    def __init__(self,
                 base_scheduler: BaseScheduler,
                 processing_manager: OnboardProcessingManager,
                 satellite_pool: Any):
        self.base_scheduler = base_scheduler
        self.processing_manager = processing_manager
        self.satellite_pool = satellite_pool
        self.constraints = ProcessingConstraint()

    def schedule_with_processing(
        self,
        imaging_tasks: List[Any],
        time_horizon: timedelta
    ) -> ScheduleWithProcessing:
        """
        联合调度成像任务与处理任务

        Args:
            imaging_tasks: 成像任务列表
            time_horizon: 规划时间范围

        Returns:
            ScheduleWithProcessing: 带在轨处理的调度结果
        """
        # 第一阶段：基础成像调度
        logger.info("Phase 1: Base imaging scheduling")
        base_schedule = self.base_scheduler.schedule()

        if not base_schedule.scheduled_tasks:
            logger.warning("No tasks scheduled in base scheduler")
            return ScheduleWithProcessing(base_schedule=base_schedule)

        # 第二阶段：为每个成像任务规划处理窗口
        logger.info("Phase 2: Planning processing windows")
        processing_windows = self._plan_processing_windows(base_schedule)

        # 第三阶段：资源冲突消解
        logger.info("Phase 3: Resolving processing conflicts")
        resolved_schedule = self._resolve_processing_conflicts(
            base_schedule, processing_windows
        )

        return resolved_schedule

    def _plan_processing_windows(
        self,
        schedule: ScheduleResult
    ) -> Dict[str, List[ProcessingWindow]]:
        """
        规划处理窗口

        对每个成像任务，规划AI处理的时间窗口
        """
        windows: Dict[str, List[ProcessingWindow]] = {}

        # 按卫星分组任务
        sat_tasks: Dict[str, List[ScheduledTask]] = {}
        for task in schedule.scheduled_tasks:
            sat_id = task.satellite_id
            if sat_id not in sat_tasks:
                sat_tasks[sat_id] = []
            sat_tasks[sat_id].append(task)

        # 为每个卫星规划处理窗口
        for sat_id, tasks in sat_tasks.items():
            windows[sat_id] = []

            # 按成像结束时间排序
            sorted_tasks = sorted(tasks, key=lambda t: t.imaging_end)

            for i, task in enumerate(sorted_tasks):
                # 获取处理时间需求
                processing_time = self._estimate_processing_time(task)

                # 处理窗口必须在成像完成后
                earliest_start = task.imaging_end + timedelta(seconds=30)

                # 查找下一个成像任务（作为处理窗口的结束边界）
                next_task = sorted_tasks[i + 1] if i + 1 < len(sorted_tasks) else None
                if next_task:
                    latest_end = next_task.imaging_start
                else:
                    # 使用makespan作为最后边界
                    latest_end = datetime.fromtimestamp(
                        datetime.now().timestamp() + schedule.makespan
                    )

                available_duration = (latest_end - earliest_start).total_seconds()

                if available_duration >= processing_time:
                    windows[sat_id].append(ProcessingWindow(
                        imaging_task_id=task.task_id,
                        satellite_id=sat_id,
                        earliest_start=earliest_start,
                        latest_end=latest_end,
                        required_duration_seconds=processing_time
                    ))
                else:
                    # 时间不足，标记为需下传原始数据
                    logger.warning(
                        f"Insufficient time for processing task {task.task_id} on {sat_id}, "
                        f"forcing raw downlink"
                    )
                    windows[sat_id].append(ProcessingWindow(
                        imaging_task_id=task.task_id,
                        satellite_id=sat_id,
                        earliest_start=earliest_start,
                        latest_end=latest_end,
                        required_duration_seconds=processing_time,
                        force_raw_downlink=True
                    ))

        return windows

    def _resolve_processing_conflicts(
        self,
        schedule: ScheduleResult,
        processing_windows: Dict[str, List[ProcessingWindow]]
    ) -> ScheduleWithProcessing:
        """
        消解处理窗口冲突

        考虑功耗、存储、热约束
        """
        resolved_windows: Dict[str, List[ProcessingWindow]] = {}

        for sat_id, windows in processing_windows.items():
            resolved_windows[sat_id] = []

            satellite = self.satellite_pool.get_satellite(sat_id)
            if not satellite:
                logger.warning(f"Satellite {sat_id} not found in pool")
                continue

            # 按优先级排序处理窗口
            sorted_windows = sorted(windows, key=lambda w: w.priority, reverse=True)

            for window in sorted_windows:
                if window.force_raw_downlink:
                    resolved_windows[sat_id].append(window)
                    continue

                # 获取窗口开始时的卫星状态
                state_at_window = satellite.predict_state(window.earliest_start)

                if not self._can_process(state_at_window, window):
                    # 无法处理，改为原始数据下传
                    logger.info(
                        f"Cannot process task {window.imaging_task_id} on {sat_id}, "
                        f"converting to raw downlink"
                    )
                    window.force_raw_downlink = True
                    resolved_windows[sat_id].append(window)
                    continue

                # 分配具体处理时间
                assigned_start = window.earliest_start
                assigned_end = assigned_start + timedelta(
                    seconds=window.required_duration_seconds
                )

                window.assigned_start = assigned_start
                window.assigned_end = assigned_end
                resolved_windows[sat_id].append(window)

                # 更新卫星状态预测（如果卫星支持）
                if hasattr(satellite, 'simulate_processing'):
                    satellite.simulate_processing(
                        start=assigned_start,
                        duration=window.required_duration_seconds
                    )

        return ScheduleWithProcessing(
            base_schedule=schedule,
            processing_windows=resolved_windows
        )

    def _can_process(self, state: Any, window: ProcessingWindow) -> bool:
        """
        检查卫星状态是否支持处理

        Args:
            state: 卫星状态
            window: 处理窗口

        Returns:
            bool: 是否可以处理
        """
        # 检查电量
        if hasattr(state, 'battery_soc'):
            if state.battery_soc < self.constraints.min_battery_soc:
                return False

        # 检查存储
        if hasattr(state, 'storage_free_gb'):
            if state.storage_free_gb < max(window.required_storage_gb, self.constraints.min_storage_free_gb):
                return False

        # 检查热余量
        if hasattr(state, 'thermal_headroom_c'):
            if state.thermal_headroom_c < self.constraints.min_thermal_headroom_c:
                return False

        # 检查AI加速器是否空闲
        if hasattr(state, 'ai_accelerator_busy'):
            if state.ai_accelerator_busy:
                return False

        return True

    def _estimate_processing_time(self, task: ScheduledTask) -> float:
        """
        估算任务处理时间

        Args:
            task: 成像任务

        Returns:
            float: 预计处理时间（秒）
        """
        # 推断任务类型
        task_type = self._infer_task_type(task)

        # 获取处理规格
        if task_type in self.processing_manager.processing_specs:
            spec = self.processing_manager.processing_specs[task_type]

            # 获取卫星加速器规格
            accelerator = self.processing_manager.accelerator_specs.get(task.satellite_id)
            if accelerator:
                return spec.processing_time_seconds(accelerator)

        # 默认处理时间
        return 600.0  # 10分钟

    def _infer_task_type(self, task: ScheduledTask) -> Any:
        """
        从成像任务推断处理任务类型

        Args:
            task: 成像任务

        Returns:
            处理任务类型
        """
        from core.processing.onboard_processing_manager import ProcessingTaskType

        # 从成像模式推断
        imaging_mode = task.imaging_mode.lower() if task.imaging_mode else ""

        if any(kw in imaging_mode for kw in ['vessel', 'ship', 'maritime']):
            return ProcessingTaskType.VESSEL_DETECTION
        elif any(kw in imaging_mode for kw in ['vehicle', 'traffic']):
            return ProcessingTaskType.VEHICLE_DETECTION
        elif any(kw in imaging_mode for kw in ['change', 'difference']):
            return ProcessingTaskType.CHANGE_DETECTION
        elif any(kw in imaging_mode for kw in ['cloud', 'weather']):
            return ProcessingTaskType.CLOUD_DETECTION

        # 默认特征提取
        return ProcessingTaskType.FEATURE_EXTRACTION

    def _find_next_task(
        self,
        schedule: ScheduleResult,
        satellite_id: str,
        after_time: datetime
    ) -> Optional[ScheduledTask]:
        """
        查找指定时间之后的下一个任务

        Args:
            schedule: 调度结果
            satellite_id: 卫星ID
            after_time: 参考时间

        Returns:
            Optional[ScheduledTask]: 下一个任务或None
        """
        sat_tasks = [
            t for t in schedule.scheduled_tasks
            if t.satellite_id == satellite_id and t.imaging_start > after_time
        ]

        if not sat_tasks:
            return None

        return min(sat_tasks, key=lambda t: t.imaging_start)
