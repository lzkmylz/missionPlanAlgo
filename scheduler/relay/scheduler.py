"""
中继卫星调度器

管理卫星成像任务通过中继卫星的数传计划。

与地面站调度器的主要区别：
1. 中继卫星是GEO轨道，可见窗口特征不同
2. 不需要天线级别的资源分配（中继卫星通常有专门的数据收发能力）
3. 可能有多个中继卫星，需要选择最优的
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.network.relay_satellite import RelayNetwork, RelaySatellite
from scheduler.base_scheduler import ScheduledTask
from scheduler.ground_station.storage import StorageState

from .downlink_task import RelayDownlinkTask

logger = logging.getLogger(__name__)


@dataclass
class RelayScheduleResult:
    """中继卫星调度结果"""
    downlink_tasks: List[RelayDownlinkTask] = field(default_factory=list)
    updated_scheduled_tasks: List[ScheduledTask] = field(default_factory=list)
    storage_states: Dict[str, StorageState] = field(default_factory=dict)
    failed_tasks: List[str] = field(default_factory=list)


class RelayScheduler:
    """中继卫星调度器

    管理通过中继卫星的数据回传计划。
    """

    def __init__(
        self,
        relay_network: RelayNetwork,
        default_data_rate_mbps: float = 450.0,
        link_setup_time_seconds: float = 10.0
    ):
        """初始化中继卫星调度器

        Args:
            relay_network: 中继卫星网络
            default_data_rate_mbps: 默认数据率 (Mbps)
            link_setup_time_seconds: 链路建立时间（秒）
        """
        self.relay_network = relay_network
        self.default_data_rate_mbps = default_data_rate_mbps
        self.link_setup_time_seconds = link_setup_time_seconds

        # 跟踪每个卫星的固存状态
        self._storage_states: Dict[str, StorageState] = {}

        # 跟踪已分配的数传窗口
        self._downlink_allocations: Dict[str, List[Tuple[datetime, datetime]]] = {}

    def initialize_satellite_storage(
        self,
        satellite_id: str,
        current_gb: float = 0.0,
        capacity_gb: float = 128.0
    ) -> None:
        """初始化卫星固存状态"""
        self._storage_states[satellite_id] = StorageState(
            capacity_gb=capacity_gb,
            current_gb=current_gb,
            overflow_threshold=0.95
        )

    def get_satellite_storage(self, satellite_id: str) -> Optional[StorageState]:
        """获取卫星固存状态"""
        return self._storage_states.get(satellite_id)

    def plan_downlink_for_task(
        self,
        imaging_task: Optional[ScheduledTask],
        visibility_windows: List[Tuple[datetime, datetime]],
        relay_id: Optional[str] = None
    ) -> Optional[RelayDownlinkTask]:
        """为单个成像任务计划中继数传

        Args:
            imaging_task: 已调度的成像任务
            visibility_windows: 中继卫星可见性窗口列表 [(start, end), ...]
            relay_id: 指定中继卫星ID，如果不指定则自动选择

        Returns:
            RelayDownlinkTask 如果成功计划，None 如果失败
        """
        if imaging_task is None:
            return None

        # 计算数据量
        data_size_gb = self._calculate_data_size(imaging_task)
        if data_size_gb <= 0:
            return None

        # 如果没有指定中继卫星，尝试找到可用的
        if relay_id is None:
            relay_id = self._find_best_relay(
                imaging_task.satellite_id,
                data_size_gb,
                imaging_task.imaging_end,
                visibility_windows
            )

        if relay_id is None:
            return None

        # 获取中继卫星配置
        relay = self.relay_network.relays.get(relay_id)
        if relay is None:
            return None

        # 计算数传时长
        data_rate = min(relay.uplink_capacity, relay.downlink_capacity)
        downlink_duration = self._calculate_downlink_duration(data_size_gb, data_rate)

        # 查找合适的窗口
        for window_start, window_end in visibility_windows:
            # 检查冲突
            if self._has_time_conflict(imaging_task.satellite_id, window_start, window_end):
                continue

            # 计算数传时间
            downlink_timing = self._calculate_downlink_timing(
                window_start, window_end,
                imaging_task.imaging_end, downlink_duration
            )

            if downlink_timing is None:
                continue

            downlink_start, downlink_end = downlink_timing

            # 分配窗口
            self._allocate_downlink_window(
                imaging_task.satellite_id, relay_id,
                downlink_start, downlink_end
            )

            # 创建数传任务
            return RelayDownlinkTask(
                task_id=f"RL-{imaging_task.task_id}",
                satellite_id=imaging_task.satellite_id,
                relay_id=relay_id,
                start_time=downlink_start,
                end_time=downlink_end,
                data_size_gb=data_size_gb,
                related_imaging_task_id=imaging_task.task_id,
                effective_data_rate=data_rate,
                acquisition_time_seconds=self.link_setup_time_seconds
            )

        return None

    def _calculate_data_size(self, imaging_task: ScheduledTask) -> float:
        """计算需要传输的数据量"""
        return imaging_task.storage_after - imaging_task.storage_before

    def _find_best_relay(
        self,
        satellite_id: str,
        data_size: float,
        start_time: datetime,
        visibility_windows: List[Tuple[datetime, datetime]]
    ) -> Optional[str]:
        """找到最佳中继卫星"""
        if not visibility_windows:
            return None

        best_relay = None
        min_latency = float('inf')

        for relay_id in self.relay_network.relays:
            # 检查该中继卫星是否有可见窗口
            has_window = False
            for window_start, window_end in visibility_windows:
                if window_start <= start_time <= window_end:
                    has_window = True
                    break

            if not has_window:
                continue

            # 检查是否可以中继数据
            can_relay, latency = self.relay_network.can_relay_data(
                satellite_id, relay_id, data_size, start_time
            )

            if can_relay and latency < min_latency:
                min_latency = latency
                best_relay = relay_id

        return best_relay

    def _calculate_downlink_duration(
        self,
        data_size_gb: float,
        data_rate_mbps: float
    ) -> float:
        """计算数传所需时长（含建链时间）"""
        if data_rate_mbps <= 0:
            return float('inf')

        # 传输时间 = 数据量(GB) * 8000 / 数据率(Mbps)
        transmission_time = (data_size_gb * 8000) / data_rate_mbps
        return transmission_time + self.link_setup_time_seconds

    def _has_time_conflict(
        self,
        satellite_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """检查是否存在时间冲突"""
        if satellite_id not in self._downlink_allocations:
            return False

        for alloc_start, alloc_end in self._downlink_allocations[satellite_id]:
            if start_time < alloc_end and end_time > alloc_start:
                return True

        return False

    def _calculate_downlink_timing(
        self,
        vis_start: datetime,
        vis_end: datetime,
        imaging_end: datetime,
        downlink_duration: float
    ) -> Optional[Tuple[datetime, datetime]]:
        """计算数传开始和结束时间"""
        downlink_start = max(vis_start, imaging_end)
        downlink_end = downlink_start + timedelta(seconds=downlink_duration)

        if downlink_end > vis_end:
            downlink_end = vis_end
            downlink_start = downlink_end - timedelta(seconds=downlink_duration)

            if downlink_start < imaging_end:
                return None

        return downlink_start, downlink_end

    def _allocate_downlink_window(
        self,
        satellite_id: str,
        relay_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """分配数传窗口"""
        if satellite_id not in self._downlink_allocations:
            self._downlink_allocations[satellite_id] = []

        self._downlink_allocations[satellite_id].append((start_time, end_time))

    def schedule_downlinks_for_tasks(
        self,
        scheduled_tasks: List[ScheduledTask],
        visibility_windows: Dict[str, List[Tuple[datetime, datetime]]]
    ) -> RelayScheduleResult:
        """为多个成像任务安排中继数传计划

        Args:
            scheduled_tasks: 已调度的成像任务列表
            visibility_windows: 每个卫星的中继可见窗口
                Dict[satellite_id, List[(start, end)]]

        Returns:
            RelayScheduleResult: 中继数传调度结果
        """
        result = RelayScheduleResult()

        # 自动初始化固存
        for task in scheduled_tasks:
            sat_id = task.satellite_id
            if sat_id not in self._storage_states:
                self.initialize_satellite_storage(
                    satellite_id=sat_id,
                    current_gb=getattr(task, 'storage_before', 0.0)
                )

        for task in scheduled_tasks:
            sat_id = task.satellite_id
            windows = visibility_windows.get(sat_id, [])

            downlink_task = self._try_schedule_single_task(task, windows)

            if downlink_task:
                result.downlink_tasks.append(downlink_task)
                self._release_storage_after_downlink(downlink_task)
            else:
                result.failed_tasks.append(task.task_id)

        result.updated_scheduled_tasks = self.update_tasks_with_downlink_info(
            scheduled_tasks, result.downlink_tasks
        )
        result.storage_states = dict(self._storage_states)

        return result

    def _try_schedule_single_task(
        self,
        task: ScheduledTask,
        windows: List[Tuple[datetime, datetime]]
    ) -> Optional[RelayDownlinkTask]:
        """尝试为单个任务调度中继数传"""
        return self.plan_downlink_for_task(task, windows)

    def _release_storage_after_downlink(self, downlink_task: RelayDownlinkTask) -> None:
        """数传完成后释放固存"""
        storage = self._storage_states.get(downlink_task.satellite_id)
        if storage:
            storage.remove_data(downlink_task.data_size_gb)

    def update_tasks_with_downlink_info(
        self,
        scheduled_tasks: List[ScheduledTask],
        downlink_tasks: List[RelayDownlinkTask]
    ) -> List[ScheduledTask]:
        """更新任务信息，添加中继数传详情"""
        downlink_map = {dl.related_imaging_task_id: dl for dl in downlink_tasks}

        updated_tasks = []
        for task in scheduled_tasks:
            if task.task_id in downlink_map:
                dl_task = downlink_map[task.task_id]
                # 更新任务的中继数传信息
                task.downlink_start = dl_task.start_time
                task.downlink_end = dl_task.end_time
                task.data_transferred = dl_task.data_size_gb
                # 标记为中继数传
                task.ground_station_id = f"RELAY:{dl_task.relay_id}"
                updated_tasks.append(task)
            else:
                updated_tasks.append(task)

        return updated_tasks

    def get_relay_utilization(self, relay_id: str) -> Dict[str, Any]:
        """获取中继卫星利用率统计"""
        total_tasks = 0
        total_time = 0.0
        total_data = 0.0

        for task in self._downlink_allocations.values():
            for start, end in task:
                total_tasks += 1
                total_time += (end - start).total_seconds()

        # 计算数据量
        for sat_id, allocations in self._downlink_allocations.items():
            for start, end in allocations:
                duration = (end - start).total_seconds()
                # 估算数据量（简化）
                data_gb = (self.default_data_rate_mbps * duration) / 8000
                total_data += data_gb

        return {
            'relay_id': relay_id,
            'total_tasks': total_tasks,
            'total_downlink_time': total_time,
            'total_data_transferred': total_data,
        }
