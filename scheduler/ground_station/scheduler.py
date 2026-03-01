"""
地面站调度器主模块

管理卫星成像任务的数传计划，包括：
- 天线资源分配
- 固存状态跟踪
- 数传任务调度
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.models.ground_station import GroundStation, Antenna
from core.resources.ground_station_pool import GroundStationPool
from scheduler.base_scheduler import ScheduledTask

from .storage import StorageState
from .downlink_task import DownlinkTask
from .utils import calculate_downlink_duration, DEFAULT_LINK_SETUP_TIME_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class GroundStationScheduleResult:
    """地面站调度结果"""
    downlink_tasks: List[DownlinkTask] = field(default_factory=list)
    updated_scheduled_tasks: List[ScheduledTask] = field(default_factory=list)
    storage_states: Dict[str, StorageState] = field(default_factory=dict)
    failed_tasks: List[str] = field(default_factory=list)


class GroundStationScheduler:
    """地面站调度器 - 支持天线级别独立调度"""

    # 默认链路建立时间（秒）：指向 + 捕获 + 同步
    DEFAULT_LINK_SETUP_TIME_SECONDS = DEFAULT_LINK_SETUP_TIME_SECONDS

    def __init__(
        self,
        ground_station_pool: GroundStationPool,
        data_rate_mbps: float = 300.0,
        storage_capacity_gb: float = 100.0,
        overflow_threshold: float = 0.95,
        link_setup_time_seconds: float = DEFAULT_LINK_SETUP_TIME_SECONDS
    ):
        """初始化地面站调度器

        Args:
            ground_station_pool: 地面站资源池
            data_rate_mbps: 默认数据传输速率 (Mbps)
            storage_capacity_gb: 默认固存容量 (GB)
            overflow_threshold: 固存溢出阈值 (0-1)
            link_setup_time_seconds: 链路建立时间（秒）
        """
        self.ground_station_pool = ground_station_pool
        self.data_rate_mbps = data_rate_mbps
        self.storage_capacity_gb = storage_capacity_gb
        self.overflow_threshold = overflow_threshold
        self.link_setup_time_seconds = link_setup_time_seconds

        # 跟踪每个卫星的固存状态
        self._storage_states: Dict[str, StorageState] = {}

        # 跟踪已分配的数传窗口
        self._downlink_allocations: Dict[Tuple[str, str], List[Tuple[datetime, datetime, str, float]]] = {}

        # 初始化所有天线的分配记录
        for gs_id, gs in ground_station_pool.stations.items():
            for antenna in gs.antennas:
                key = (gs_id, antenna.id)
                self._downlink_allocations[key] = []

    # ==================== 固存管理 ====================

    def initialize_satellite_storage(
        self,
        satellite_id: str,
        current_gb: float = 0.0,
        capacity_gb: Optional[float] = None
    ) -> None:
        """初始化卫星固存状态"""
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

    # ==================== 天线冲突检测 ====================

    def has_antenna_conflict(
        self,
        ground_station_id: str,
        antenna_id: str,
        time_window: Tuple[datetime, datetime]
    ) -> bool:
        """检查指定天线在指定时间窗口是否有冲突"""
        key = (ground_station_id, antenna_id)
        if key not in self._downlink_allocations:
            return False

        start_time, end_time = time_window

        for alloc_start, alloc_end, _, _ in self._downlink_allocations[key]:
            if start_time < alloc_end and end_time > alloc_start:
                return True

        return False

    def find_available_antenna(
        self,
        ground_station_id: str,
        time_window: Tuple[datetime, datetime]
    ) -> Optional[Tuple[str, float]]:
        """查找地面站中可用的天线"""
        gs = self.ground_station_pool.get_station(ground_station_id)
        if not gs:
            return None

        for antenna in gs.antennas:
            if not self.has_antenna_conflict(ground_station_id, antenna.id, time_window):
                effective_data_rate = antenna.data_rate if antenna.data_rate > 0 else self.data_rate_mbps
                return (antenna.id, effective_data_rate)

        return None

    # ==================== 数传计划核心方法 ====================

    def plan_downlink_for_task(
        self,
        imaging_task: Optional[ScheduledTask],
        visibility_window: Tuple[datetime, datetime],
        ground_station_id: Optional[str] = None
    ) -> Optional[DownlinkTask]:
        """为单个成像任务计划数传

        Args:
            imaging_task: 已调度的成像任务
            visibility_window: 地面站可见性窗口 (start, end)
            ground_station_id: 指定的地面站ID，如果不指定则自动选择

        Returns:
            DownlinkTask 如果成功计划，None 如果失败
        """
        if imaging_task is None:
            return None

        # 步骤1: 验证时间窗口
        vis_start, vis_end = visibility_window
        if not self._validate_visibility_window(vis_start, vis_end, imaging_task.imaging_end):
            return None

        # 步骤2: 计算数传参数
        data_size_gb = self._calculate_data_size(imaging_task)
        if data_size_gb <= 0:
            return None

        downlink_duration = self._calculate_downlink_duration_with_setup(data_size_gb)
        if not self._check_visibility_duration(vis_start, vis_end, downlink_duration):
            return None

        # 步骤3: 计算数传时间窗口
        downlink_timing = self._calculate_downlink_timing(
            vis_start, vis_end, imaging_task.imaging_end, downlink_duration
        )
        if downlink_timing is None:
            return None
        downlink_start, downlink_end = downlink_timing

        # 步骤4: 选择地面站和天线
        antenna_selection = self._select_antenna(
            ground_station_id, downlink_start, downlink_end
        )
        if antenna_selection is None:
            return None
        selected_gs, selected_antenna, selected_data_rate = antenna_selection

        # 步骤5: 重新计算数传时长并验证
        final_timing = self._recalculate_and_validate_timing(
            downlink_start, downlink_end, vis_end,
            imaging_task.imaging_end, data_size_gb, selected_data_rate
        )
        if final_timing is None:
            return None
        downlink_start, downlink_end = final_timing

        # 步骤6: 分配数传窗口
        self._allocate_downlink_window(
            imaging_task.satellite_id, selected_gs, selected_antenna,
            downlink_start, downlink_end, data_size_gb
        )

        # 步骤7: 创建数传任务
        return self._create_downlink_task(
            imaging_task, selected_gs, selected_antenna,
            downlink_start, downlink_end, data_size_gb, selected_data_rate
        )

    # ==================== 辅助方法 ====================

    def _validate_visibility_window(
        self,
        vis_start: datetime,
        vis_end: datetime,
        imaging_end: datetime
    ) -> bool:
        """验证可见性窗口有效性"""
        return vis_end > imaging_end

    def _calculate_data_size(self, imaging_task: ScheduledTask) -> float:
        """计算需要传输的数据量"""
        return imaging_task.storage_after - imaging_task.storage_before

    def _calculate_downlink_duration_with_setup(self, data_size_gb: float) -> float:
        """计算数传所需时长（含链路建立时间）"""
        return calculate_downlink_duration(
            data_size_gb, self.data_rate_mbps, self.link_setup_time_seconds
        )

    def _check_visibility_duration(
        self,
        vis_start: datetime,
        vis_end: datetime,
        required_duration: float
    ) -> bool:
        """检查可见性窗口是否足够长"""
        visibility_duration = (vis_end - vis_start).total_seconds()
        return visibility_duration >= required_duration

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

    def _select_antenna(
        self,
        ground_station_id: Optional[str],
        downlink_start: datetime,
        downlink_end: datetime
    ) -> Optional[Tuple[str, str, float]]:
        """选择地面站和可用天线

        Returns:
            (ground_station_id, antenna_id, data_rate) 元组
        """
        time_window = (downlink_start, downlink_end)

        if ground_station_id:
            antenna_info = self.find_available_antenna(ground_station_id, time_window)
            if antenna_info:
                return (ground_station_id, antenna_info[0], antenna_info[1])
            return None

        for gs_id in self.ground_station_pool.stations:
            antenna_info = self.find_available_antenna(gs_id, time_window)
            if antenna_info:
                return (gs_id, antenna_info[0], antenna_info[1])

        return None

    def _recalculate_and_validate_timing(
        self,
        downlink_start: datetime,
        downlink_end: datetime,
        vis_end: datetime,
        imaging_end: datetime,
        data_size_gb: float,
        data_rate: float
    ) -> Optional[Tuple[datetime, datetime]]:
        """使用实际数据率重新计算数传时长并验证"""
        downlink_duration = calculate_downlink_duration(
            data_size_gb, data_rate, self.link_setup_time_seconds
        )
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
        ground_station_id: str,
        antenna_id: str,
        start_time: datetime,
        end_time: datetime,
        data_size_gb: float
    ) -> None:
        """分配数传窗口到具体天线"""
        key = (ground_station_id, antenna_id)
        self._downlink_allocations[key].append(
            (start_time, end_time, satellite_id, data_size_gb)
        )

    def _create_downlink_task(
        self,
        imaging_task: ScheduledTask,
        ground_station_id: str,
        antenna_id: str,
        start_time: datetime,
        end_time: datetime,
        data_size_gb: float,
        data_rate: float
    ) -> DownlinkTask:
        """创建数传任务"""
        return DownlinkTask(
            task_id=f"DL-{imaging_task.task_id}",
            satellite_id=imaging_task.satellite_id,
            ground_station_id=ground_station_id,
            start_time=start_time,
            end_time=end_time,
            data_size_gb=data_size_gb,
            antenna_id=antenna_id,
            related_imaging_task_id=imaging_task.task_id,
            effective_data_rate=data_rate,
            link_setup_time_seconds=self.link_setup_time_seconds
        )

    # ==================== 批量调度方法 ====================

    def schedule_downlinks_for_tasks(
        self,
        scheduled_tasks: List[ScheduledTask],
        visibility_windows: Dict[str, List[Tuple[datetime, datetime]]]
    ) -> GroundStationScheduleResult:
        """为多个成像任务安排数传计划"""
        result = GroundStationScheduleResult()

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
    ) -> Optional[DownlinkTask]:
        """尝试为单个任务调度数传"""
        for window in windows:
            downlink_task = self.plan_downlink_for_task(task, window)
            if downlink_task:
                return downlink_task
        return None

    def _release_storage_after_downlink(self, downlink_task: DownlinkTask) -> None:
        """数传完成后释放固存"""
        storage = self._storage_states.get(downlink_task.satellite_id)
        if storage:
            storage.remove_data(downlink_task.data_size_gb)

    def update_tasks_with_downlink_info(
        self,
        scheduled_tasks: List[ScheduledTask],
        downlink_tasks: List[DownlinkTask]
    ) -> List[ScheduledTask]:
        """更新任务信息，添加数传详情"""
        downlink_map = {dl.related_imaging_task_id: dl for dl in downlink_tasks}

        updated_tasks = []
        for task in scheduled_tasks:
            if task.task_id in downlink_map:
                dl_task = downlink_map[task.task_id]
                updated_task = self._update_single_task(task, dl_task)
                updated_tasks.append(updated_task)
            else:
                updated_tasks.append(task)

        return updated_tasks

    def _update_single_task(
        self,
        task: ScheduledTask,
        downlink_task: DownlinkTask
    ) -> ScheduledTask:
        """更新单个任务的数传信息"""
        task.ground_station_id = downlink_task.ground_station_id
        task.downlink_start = downlink_task.start_time
        task.downlink_end = downlink_task.end_time
        task.data_transferred = downlink_task.data_size_gb
        return task

    # ==================== 统计和报告方法 ====================

    def get_antenna_utilization(
        self,
        ground_station_id: str,
        antenna_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """计算指定天线的利用率统计"""
        key = (ground_station_id, antenna_id)
        if key not in self._downlink_allocations:
            return self._empty_utilization_stats()

        allocations = self._downlink_allocations[key]
        return self._calculate_utilization_stats(allocations, time_range)

    def _empty_utilization_stats(self) -> Dict[str, Any]:
        """返回空的利用率统计"""
        return {
            'total_tasks': 0,
            'total_downlink_time': 0.0,
            'total_data_transferred': 0.0,
            'utilization_ratio': 0.0,
            'average_task_duration': 0.0,
        }

    def _calculate_utilization_stats(
        self,
        allocations: List[Tuple[datetime, datetime, str, float]],
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> Dict[str, Any]:
        """计算利用率统计"""
        total_tasks = len(allocations)
        total_downlink_time = 0.0
        total_data_transferred = 0.0

        for start_time, end_time, _, data_size_gb in allocations:
            duration, data = self._process_single_allocation(
                start_time, end_time, data_size_gb, time_range
            )
            total_downlink_time += duration
            total_data_transferred += data

        utilization_ratio = self._calculate_utilization_ratio(
            total_downlink_time, time_range, total_tasks
        )
        average_task_duration = total_downlink_time / total_tasks if total_tasks > 0 else 0.0

        return {
            'total_tasks': total_tasks,
            'total_downlink_time': total_downlink_time,
            'total_data_transferred': total_data_transferred,
            'utilization_ratio': min(utilization_ratio, 1.0),
            'average_task_duration': average_task_duration,
        }

    def _process_single_allocation(
        self,
        start_time: datetime,
        end_time: datetime,
        data_size_gb: float,
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> Tuple[float, float]:
        """处理单个分配记录，返回时长和数据量"""
        if time_range:
            return self._process_allocation_in_range(
                start_time, end_time, data_size_gb, time_range
            )
        duration = (end_time - start_time).total_seconds()
        return duration, data_size_gb

    def _process_allocation_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        data_size_gb: float,
        time_range: Tuple[datetime, datetime]
    ) -> Tuple[float, float]:
        """处理在时间范围内的分配记录"""
        range_start, range_end = time_range
        effective_start = max(start_time, range_start)
        effective_end = min(end_time, range_end)

        if effective_start >= effective_end:
            return 0.0, 0.0

        duration = (effective_end - effective_start).total_seconds()
        full_duration = (end_time - start_time).total_seconds()

        if full_duration <= 0:
            return duration, 0.0

        ratio = duration / full_duration
        return duration, data_size_gb * ratio

    def _calculate_utilization_ratio(
        self,
        total_downlink_time: float,
        time_range: Optional[Tuple[datetime, datetime]],
        total_tasks: int
    ) -> float:
        """计算利用率比例"""
        if time_range:
            range_duration = (time_range[1] - time_range[0]).total_seconds()
            return total_downlink_time / range_duration if range_duration > 0 else 0.0

        # 如果没有指定时间范围，基于24小时估算
        if total_tasks > 0:
            return total_downlink_time / (24 * 3600)
        return 0.0

    # ==================== 向后兼容的公共方法 ====================

    def allocate_downlink_window(
        self,
        satellite_id: str,
        ground_station_id: str,
        antenna_id: str,
        time_window: Tuple[datetime, datetime],
        data_size_gb: float = 0.0
    ) -> None:
        """分配数传窗口到具体天线（公共接口）"""
        start_time, end_time = time_window
        self._allocate_downlink_window(
            satellite_id, ground_station_id, antenna_id,
            start_time, end_time, data_size_gb
        )

    def get_total_available_antennas(self, ground_station_id: str) -> int:
        """获取地面站的总天线数量"""
        gs = self.ground_station_pool.get_station(ground_station_id)
        if not gs:
            return 0
        return len(gs.antennas)

    def get_antenna_data_rate(self, ground_station_id: str, antenna_id: str) -> float:
        """获取指定天线的数据率"""
        gs = self.ground_station_pool.get_station(ground_station_id)
        if not gs:
            return self.data_rate_mbps

        for antenna in gs.antennas:
            if antenna.id == antenna_id:
                return antenna.data_rate if antenna.data_rate > 0 else self.data_rate_mbps

        return self.data_rate_mbps

    def get_all_antennas_utilization_report(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """获取所有天线的利用率报告"""
        report = {}
        for (gs_id, antenna_id) in self._downlink_allocations:
            if gs_id not in report:
                report[gs_id] = {}
            report[gs_id][antenna_id] = self.get_antenna_utilization(gs_id, antenna_id, time_range)
        return report

    def get_ground_station_summary(self, ground_station_id: str) -> Dict[str, Any]:
        """获取地面站汇总统计"""
        gs = self.ground_station_pool.get_station(ground_station_id)
        if not gs:
            return {
                'ground_station_id': ground_station_id,
                'exists': False,
                'total_antennas': 0,
                'active_antennas': 0,
                'total_tasks': 0,
                'total_data_transferred': 0.0,
            }

        total_tasks = 0
        total_data = 0.0
        active_antennas = 0

        for antenna in gs.antennas:
            stats = self.get_antenna_utilization(ground_station_id, antenna.id)
            total_tasks += stats['total_tasks']
            total_data += stats['total_data_transferred']
            if stats['total_tasks'] > 0:
                active_antennas += 1

        return {
            'ground_station_id': ground_station_id,
            'exists': True,
            'total_antennas': len(gs.antennas),
            'active_antennas': active_antennas,
            'total_tasks': total_tasks,
            'total_data_transferred': total_data,
        }

    def release_storage_after_downlink(self, downlink_task: DownlinkTask) -> None:
        """数传完成后释放固存（公共接口）"""
        self._release_storage_after_downlink(downlink_task)

    def schedule_downlink_to_prevent_overflow(
        self,
        imaging_task: ScheduledTask,
        visibility_windows: List[Tuple[datetime, datetime]],
        overflow_threshold: float = 0.9
    ) -> Optional[DownlinkTask]:
        """在固存溢出前安排数传"""
        # 检查成像后固存使用率是否会超过阈值
        storage = self._storage_states.get(imaging_task.satellite_id)
        if not storage:
            return None

        # 计算成像后的使用率
        data_added = imaging_task.storage_after - imaging_task.storage_before
        projected_usage = (storage.current_gb + data_added) / storage.capacity_gb

        # 如果投影使用率低于阈值，不需要预防性数传
        if projected_usage < overflow_threshold:
            return None

        # 尝试安排数传
        for window in visibility_windows:
            downlink_task = self.plan_downlink_for_task(imaging_task, window)
            if downlink_task:
                return downlink_task

        return None
