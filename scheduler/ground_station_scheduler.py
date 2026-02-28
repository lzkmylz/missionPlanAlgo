"""
地面站调度器 - 为成像任务安排数传计划

核心功能：
1. 为每个已调度的成像任务安排数传任务
2. 考虑固存状态，在固存满前必须安排数传
3. 考虑地面站可见性窗口（已预计算）
4. 考虑地面站天线资源冲突（同一时间只能服务一个卫星）
5. 数传完成后释放固存
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.models.ground_station import GroundStation, Antenna
from core.resources.ground_station_pool import GroundStationPool
from scheduler.base_scheduler import ScheduledTask

logger = logging.getLogger(__name__)


def calculate_downlink_duration(data_size_gb: float, data_rate_mbps: float) -> float:
    """
    计算数传所需时长

    公式: duration = data_size_gb / (data_rate_mbps / 8 / 1024)
    其中:
        - data_size_gb: 数据量 (GB)
        - data_rate_mbps: 数据传输速率 (Mbps)
        - 转换: Mbps -> MB/s = Mbps / 8 (bits to bytes)
        - 转换: MB/s -> GB/s = MB/s / 1024

    Args:
        data_size_gb: 数据量 (GB)
        data_rate_mbps: 数据传输速率 (Mbps)

    Returns:
        数传时长 (秒)

    Raises:
        ValueError: 如果 data_rate_mbps 为 0
    """
    if data_rate_mbps <= 0:
        raise ValueError("Data rate must be positive")

    if data_size_gb <= 0:
        return 0.0

    # 转换 Mbps 到 GB/s: Mbps / 8 = MB/s, MB/s / 1024 = GB/s
    data_rate_gb_per_sec = data_rate_mbps / 8 / 1024

    return data_size_gb / data_rate_gb_per_sec


@dataclass
class StorageState:
    """卫星固存状态"""
    capacity_gb: float  # 总容量 (GB)
    current_gb: float   # 当前使用量 (GB)
    overflow_threshold: float = 1.0  # 溢出阈值 (默认 100%)

    def get_available_space(self) -> float:
        """获取可用空间"""
        return self.capacity_gb - self.current_gb

    def add_data(self, data_gb: float) -> None:
        """添加数据到固存"""
        self.current_gb = min(self.capacity_gb, self.current_gb + data_gb)

    def remove_data(self, data_gb: float) -> None:
        """从固存移除数据"""
        self.current_gb = max(0.0, self.current_gb - data_gb)

    def will_overflow(self, data_gb: float) -> bool:
        """检查添加数据后是否会溢出"""
        threshold_capacity = self.capacity_gb * self.overflow_threshold
        return (self.current_gb + data_gb) > threshold_capacity

    def get_usage_ratio(self) -> float:
        """获取使用率"""
        if self.capacity_gb <= 0:
            return 0.0
        return self.current_gb / self.capacity_gb


@dataclass
class DownlinkTask:
    """数传任务"""
    task_id: str
    satellite_id: str
    ground_station_id: str
    start_time: datetime
    end_time: datetime
    data_size_gb: float
    antenna_id: Optional[str] = None
    related_imaging_task_id: Optional[str] = None

    def get_duration_seconds(self) -> float:
        """获取数传时长（秒）"""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class GroundStationScheduleResult:
    """地面站调度结果"""
    downlink_tasks: List[DownlinkTask] = field(default_factory=list)
    updated_scheduled_tasks: List[ScheduledTask] = field(default_factory=list)
    storage_states: Dict[str, StorageState] = field(default_factory=dict)
    failed_tasks: List[str] = field(default_factory=list)


class GroundStationScheduler:
    """
    地面站调度器 - 支持天线级别独立调度

    为成像任务安排数传任务，考虑以下约束：
    1. 数传必须在成像后（数据生成后）才能开始
    2. 数传时长根据数据量计算
    3. 每个天线独立工作，同一地面站的不同天线可同时服务不同卫星
    4. 必须在固存溢出前安排数传

    关键特性：
    - 天线级别资源分配，不同天线之间无冲突
    - 同一地面站的多天线可以并行处理多个数传任务
    - 自动选择最优天线进行分配
    """

    def __init__(
        self,
        ground_station_pool: GroundStationPool,
        data_rate_mbps: float = 300.0,
        storage_capacity_gb: float = 100.0,
        overflow_threshold: float = 0.95
    ):
        """
        初始化地面站调度器

        Args:
            ground_station_pool: 地面站资源池
            data_rate_mbps: 默认数据传输速率 (Mbps)
            storage_capacity_gb: 默认固存容量 (GB)
            overflow_threshold: 固存溢出阈值 (0-1)
        """
        self.ground_station_pool = ground_station_pool
        self.data_rate_mbps = data_rate_mbps
        self.storage_capacity_gb = storage_capacity_gb
        self.overflow_threshold = overflow_threshold

        # 跟踪每个卫星的固存状态
        self._storage_states: Dict[str, StorageState] = {}

        # 跟踪已分配的数传窗口: (ground_station_id, antenna_id) -> list of (start, end, satellite_id)
        # 每个天线独立工作，可以同时进行数传
        self._downlink_allocations: Dict[Tuple[str, str], List[Tuple[datetime, datetime, str]]] = {}

        # 初始化所有天线的分配记录
        for gs_id, gs in ground_station_pool.stations.items():
            for antenna in gs.antennas:
                key = (gs_id, antenna.id)
                self._downlink_allocations[key] = []

    def initialize_satellite_storage(
        self,
        satellite_id: str,
        current_gb: float = 0.0,
        capacity_gb: Optional[float] = None
    ) -> None:
        """
        初始化卫星固存状态

        Args:
            satellite_id: 卫星ID
            current_gb: 当前使用量 (GB)
            capacity_gb: 容量 (GB), 默认使用调度器默认值
        """
        capacity = capacity_gb if capacity_gb is not None else self.storage_capacity_gb
        self._storage_states[satellite_id] = StorageState(
            capacity_gb=capacity,
            current_gb=current_gb,
            overflow_threshold=self.overflow_threshold
        )

    def get_satellite_storage(self, satellite_id: str) -> Optional[StorageState]:
        """获取卫星固存状态"""
        return self._storage_states.get(satellite_id)

    def will_storage_overflow(self, satellite_id: str, data_gb: float) -> bool:
        """检查添加数据后是否会溢出"""
        storage = self._storage_states.get(satellite_id)
        if not storage:
            return True
        return storage.will_overflow(data_gb)

    def has_antenna_conflict(
        self,
        ground_station_id: str,
        antenna_id: str,
        time_window: Tuple[datetime, datetime]
    ) -> bool:
        """
        检查指定天线在指定时间窗口是否有冲突

        每个天线独立工作，不同天线之间没有冲突。
        同一地面站的不同天线可以同时服务不同卫星。

        Args:
            ground_station_id: 地面站ID
            antenna_id: 天线ID
            time_window: (start_time, end_time) 时间窗口

        Returns:
            True 如果有冲突，False 如果没有冲突
        """
        key = (ground_station_id, antenna_id)
        if key not in self._downlink_allocations:
            return False

        start_time, end_time = time_window

        # 检查与现有分配的冲突（同一特定天线）
        for alloc_start, alloc_end, _ in self._downlink_allocations[key]:
            # 检查时间重叠
            if start_time < alloc_end and end_time > alloc_start:
                return True

        return False

    def find_available_antenna(
        self,
        ground_station_id: str,
        time_window: Tuple[datetime, datetime]
    ) -> Optional[str]:
        """
        查找地面站中可用的天线

        遍历地面站的所有天线，找到在指定时间窗口内没有冲突的天线。

        Args:
            ground_station_id: 地面站ID
            time_window: (start_time, end_time) 时间窗口

        Returns:
            可用天线的ID，如果没有可用天线则返回None
        """
        gs = self.ground_station_pool.get_station(ground_station_id)
        if not gs:
            return None

        for antenna in gs.antennas:
            if not self.has_antenna_conflict(ground_station_id, antenna.id, time_window):
                return antenna.id

        return None

    def get_total_available_antennas(self, ground_station_id: str) -> int:
        """获取地面站的总天线数量"""
        gs = self.ground_station_pool.get_station(ground_station_id)
        if not gs:
            return 0
        return len(gs.antennas)

    def allocate_downlink_window(
        self,
        satellite_id: str,
        ground_station_id: str,
        antenna_id: str,
        time_window: Tuple[datetime, datetime]
    ) -> bool:
        """
        分配数传窗口到指定天线上

        Args:
            satellite_id: 卫星ID
            ground_station_id: 地面站ID
            antenna_id: 天线ID
            time_window: (start_time, end_time) 时间窗口

        Returns:
            True 如果分配成功
        """
        key = (ground_station_id, antenna_id)
        if key not in self._downlink_allocations:
            self._downlink_allocations[key] = []

        start_time, end_time = time_window
        self._downlink_allocations[key].append(
            (start_time, end_time, satellite_id)
        )
        return True

    def plan_downlink_for_task(
        self,
        imaging_task: Optional[ScheduledTask],
        visibility_window: Tuple[datetime, datetime],
        ground_station_id: Optional[str] = None
    ) -> Optional[DownlinkTask]:
        """
        为单个成像任务计划数传

        Args:
            imaging_task: 已调度的成像任务
            visibility_window: 地面站可见性窗口 (start, end)
            ground_station_id: 指定的地面站ID，如果不指定则自动选择

        Returns:
            DownlinkTask 如果成功计划，None 如果失败
        """
        if imaging_task is None:
            return None

        vis_start, vis_end = visibility_window

        # 数传必须在成像后才能开始
        if vis_end <= imaging_task.imaging_end:
            return None

        # 计算需要传输的数据量
        data_size_gb = imaging_task.storage_after - imaging_task.storage_before
        if data_size_gb <= 0:
            return None

        # 计算数传所需时长
        downlink_duration = calculate_downlink_duration(
            data_size_gb, self.data_rate_mbps
        )

        # 检查可见性窗口是否足够长
        visibility_duration = (vis_end - vis_start).total_seconds()
        if visibility_duration < downlink_duration:
            return None

        # 确定数传开始时间（必须在成像后）
        downlink_start = max(vis_start, imaging_task.imaging_end)
        downlink_end = downlink_start + timedelta(seconds=downlink_duration)

        # 如果计算出的结束时间超过可见性窗口，调整开始时间
        if downlink_end > vis_end:
            downlink_end = vis_end
            downlink_start = downlink_end - timedelta(seconds=downlink_duration)

            # 再次检查是否在成像后
            if downlink_start < imaging_task.imaging_end:
                return None

        # 选择地面站和可用天线
        selected_gs = ground_station_id
        selected_antenna = None

        if selected_gs:
            # 在指定地面站查找可用天线
            selected_antenna = self.find_available_antenna(selected_gs, (downlink_start, downlink_end))
        else:
            # 自动选择第一个有可用的地面站和天线
            for gs_id in self.ground_station_pool.stations:
                antenna_id = self.find_available_antenna(gs_id, (downlink_start, downlink_end))
                if antenna_id:
                    selected_gs = gs_id
                    selected_antenna = antenna_id
                    break

        if not selected_gs or not selected_antenna:
            return None

        # 分配数传窗口到具体天线
        self.allocate_downlink_window(
            imaging_task.satellite_id,
            selected_gs,
            selected_antenna,
            (downlink_start, downlink_end)
        )

        # 创建数传任务（包含天线信息）
        downlink_task = DownlinkTask(
            task_id=f"DL-{imaging_task.task_id}",
            satellite_id=imaging_task.satellite_id,
            ground_station_id=selected_gs,
            start_time=downlink_start,
            end_time=downlink_end,
            data_size_gb=data_size_gb,
            antenna_id=selected_antenna,  # 记录具体使用的天线
            related_imaging_task_id=imaging_task.task_id
        )

        return downlink_task

    def schedule_downlink_to_prevent_overflow(
        self,
        imaging_task: ScheduledTask,
        visibility_windows: List[Tuple[datetime, datetime]],
        overflow_threshold: float = 0.9
    ) -> Optional[DownlinkTask]:
        """
        在固存溢出前安排数传

        Args:
            imaging_task: 成像任务
            visibility_windows: 可用的地面站可见性窗口列表
            overflow_threshold: 溢出阈值

        Returns:
            DownlinkTask 如果成功安排，None 如果失败
        """
        satellite_id = imaging_task.satellite_id

        # 初始化固存状态（如果未初始化）
        if satellite_id not in self._storage_states:
            self.initialize_satellite_storage(satellite_id)

        storage = self._storage_states[satellite_id]

        # 计算成像后的数据量
        data_after_imaging = imaging_task.storage_after

        # 检查是否会超过阈值
        threshold_capacity = storage.capacity_gb * overflow_threshold
        if data_after_imaging <= threshold_capacity:
            # 不会溢出，不需要立即数传
            return None

        # 需要安排数传，尝试每个可见性窗口
        for vis_window in visibility_windows:
            downlink_task = self.plan_downlink_for_task(imaging_task, vis_window)
            if downlink_task:
                return downlink_task

        return None

    def release_storage_after_downlink(self, downlink_task: DownlinkTask) -> None:
        """
        数传完成后释放固存

        Args:
            downlink_task: 完成的数传任务
        """
        satellite_id = downlink_task.satellite_id
        storage = self._storage_states.get(satellite_id)

        if storage:
            storage.remove_data(downlink_task.data_size_gb)
            logger.info(
                f"Released {downlink_task.data_size_gb} GB from satellite {satellite_id} storage. "
                f"Current usage: {storage.current_gb} GB"
            )

    def update_tasks_with_downlink_info(
        self,
        scheduled_tasks: List[ScheduledTask],
        downlink_tasks: List[DownlinkTask]
    ) -> List[ScheduledTask]:
        """
        更新已调度任务为数传信息

        Args:
            scheduled_tasks: 已调度的成像任务列表
            downlink_tasks: 数传任务列表

        Returns:
            更新后的成像任务列表
        """
        # 创建数传任务映射: imaging_task_id -> downlink_task
        downlink_map: Dict[str, DownlinkTask] = {}
        for dl in downlink_tasks:
            if dl.related_imaging_task_id:
                downlink_map[dl.related_imaging_task_id] = dl

        # 更新每个成像任务
        updated_tasks = []
        for task in scheduled_tasks:
            updated_task = ScheduledTask(
                task_id=task.task_id,
                satellite_id=task.satellite_id,
                target_id=task.target_id,
                imaging_start=task.imaging_start,
                imaging_end=task.imaging_end,
                imaging_mode=task.imaging_mode,
                slew_angle=task.slew_angle,
                storage_before=task.storage_before,
                storage_after=task.storage_after,
                power_before=task.power_before,
                power_after=task.power_after,
            )

            # 如果有对应的数传任务，更新数传信息
            if task.task_id in downlink_map:
                dl = downlink_map[task.task_id]
                updated_task.ground_station_id = dl.ground_station_id
                updated_task.antenna_id = dl.antenna_id  # 记录天线ID
                updated_task.downlink_start = dl.start_time
                updated_task.downlink_end = dl.end_time
                updated_task.data_transferred = dl.data_size_gb

            updated_tasks.append(updated_task)

        return updated_tasks

    def schedule_downlinks_for_tasks(
        self,
        scheduled_tasks: List[ScheduledTask],
        visibility_windows: Dict[str, List[Tuple[datetime, datetime]]]
    ) -> GroundStationScheduleResult:
        """
        为多个成像任务安排数传

        Args:
            scheduled_tasks: 已调度的成像任务列表
            visibility_windows: 卫星ID -> 地面站可见性窗口列表

        Returns:
            GroundStationScheduleResult 包含数传任务和更新后的成像任务
        """
        result = GroundStationScheduleResult()

        if not scheduled_tasks:
            return result

        # 按卫星分组任务
        tasks_by_satellite: Dict[str, List[ScheduledTask]] = {}
        for task in scheduled_tasks:
            if task.satellite_id not in tasks_by_satellite:
                tasks_by_satellite[task.satellite_id] = []
            tasks_by_satellite[task.satellite_id].append(task)

        # 初始化每个卫星的固存状态
        for sat_id in tasks_by_satellite:
            if sat_id not in self._storage_states:
                self.initialize_satellite_storage(sat_id)

        # 为每个卫星安排数传
        all_downlink_tasks: List[DownlinkTask] = []

        for sat_id, tasks in tasks_by_satellite.items():
            sat_windows = visibility_windows.get(sat_id, [])

            # 按成像时间排序任务
            sorted_tasks = sorted(tasks, key=lambda t: t.imaging_start)

            for task in sorted_tasks:
                # 更新固存状态（添加成像数据）
                storage = self._storage_states[sat_id]
                data_generated = task.storage_after - task.storage_before
                storage.add_data(data_generated)

                # 尝试为任务安排数传
                downlink_task = None

                for window in sat_windows:
                    downlink_task = self.plan_downlink_for_task(task, window)
                    if downlink_task:
                        break

                if downlink_task:
                    all_downlink_tasks.append(downlink_task)
                    # 数传完成后释放固存
                    self.release_storage_after_downlink(downlink_task)
                else:
                    # 记录失败的任务
                    result.failed_tasks.append(task.task_id)

        # 更新成像任务的数传信息
        updated_tasks = self.update_tasks_with_downlink_info(
            scheduled_tasks,
            all_downlink_tasks
        )

        result.downlink_tasks = all_downlink_tasks
        result.updated_scheduled_tasks = updated_tasks
        result.storage_states = dict(self._storage_states)

        return result
