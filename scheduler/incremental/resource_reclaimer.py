"""
资源回收计算器

负责计算现有调度结果中的剩余资源，包括：
- 时间窗口间隙
- 剩余电量
- 剩余存储容量
- 可见性窗口匹配
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .incremental_state import IncrementalState, ResourceWindow

logger = logging.getLogger(__name__)


@dataclass
class ResourceProfile:
    """资源概况"""
    total_available_time: float      # 总可用时间（秒）
    total_available_power: float     # 总可用电量（Wh）
    total_available_storage: float   # 总可用存储（GB）
    window_count: int                # 可用窗口数量
    average_window_duration: float   # 平均窗口时长（秒）
    utilization_rate: float          # 资源利用率（0-1）


@dataclass
class TimeResourceSlot:
    """时间资源槽（特定时间段的资源状态）"""
    start_time: datetime
    end_time: datetime
    power_available: float           # 可用电量
    storage_available: float         # 可用存储
    utilization_ratio: float         # 该时段的利用率


class ResourceReclaimer:
    """
    资源回收计算器

    分析现有调度结果，计算可用于增量规划的资源
    """

    def __init__(self, incremental_state: IncrementalState):
        """
        初始化资源回收计算器

        Args:
            incremental_state: 增量规划状态管理器
        """
        self.state = incremental_state
        self.mission = incremental_state.mission

    def calculate_remaining_resources(self, satellite_id: str) -> ResourceProfile:
        """
        计算指定卫星的剩余资源概况

        Args:
            satellite_id: 卫星ID

        Returns:
            ResourceProfile: 资源概况
        """
        windows = self.state.get_available_windows(satellite_id)

        if not windows:
            return ResourceProfile(
                total_available_time=0.0,
                total_available_power=0.0,
                total_available_storage=0.0,
                window_count=0,
                average_window_duration=0.0,
                utilization_rate=1.0
            )

        total_time = sum(w.duration() for w in windows)
        avg_power = sum(w.available_power for w in windows) / len(windows)
        avg_storage = sum(w.available_storage for w in windows) / len(windows)

        # 计算利用率
        sat_state = self.state.get_satellite_state(satellite_id)
        if sat_state and self.mission:
            mission_duration = (self.mission.end_time - self.mission.start_time).total_seconds()
            utilization = 1.0 - (total_time / mission_duration) if mission_duration > 0 else 0.0
        else:
            utilization = 0.0

        return ResourceProfile(
            total_available_time=total_time,
            total_available_power=avg_power * len(windows),
            total_available_storage=avg_storage,
            window_count=len(windows),
            average_window_duration=total_time / len(windows) if windows else 0.0,
            utilization_rate=utilization
        )

    def calculate_all_remaining_resources(self) -> Dict[str, ResourceProfile]:
        """计算所有卫星的剩余资源"""
        return {
            sat_id: self.calculate_remaining_resources(sat_id)
            for sat_id in self.state.get_all_satellite_ids()
        }

    def find_matching_windows(self, satellite_id: str,
                             target_visibility_windows: List[Tuple[datetime, datetime]],
                             min_duration: float = 60.0,
                             required_power: float = 0.0,
                             required_storage: float = 0.0) -> List[ResourceWindow]:
        """
        查找与目标可见窗口匹配的资源可用窗口

        Args:
            satellite_id: 卫星ID
            target_visibility_windows: 目标可见窗口列表 [(start, end), ...]
            min_duration: 最小持续时间（秒）
            required_power: 所需电量
            required_storage: 所需存储

        Returns:
            List[ResourceWindow]: 匹配的资源窗口
        """
        sat_state = self.state.get_satellite_state(satellite_id)
        if not sat_state:
            return []

        matching_windows = []

        for vis_start, vis_end in target_visibility_windows:
            # 查找资源窗口与可见窗口的交集
            resource_windows = sat_state.find_resource_windows(vis_start, vis_end)

            for rw in resource_windows:
                # 计算交集
                inter_start = max(rw.start_time, vis_start)
                inter_end = min(rw.end_time, vis_end)

                if inter_start < inter_end:
                    duration = (inter_end - inter_start).total_seconds()

                    if duration >= min_duration:
                        if rw.available_power >= required_power and \
                           rw.available_storage >= required_storage:
                            matching_windows.append(ResourceWindow(
                                start_time=inter_start,
                                end_time=inter_end,
                                available_power=rw.available_power,
                                available_storage=rw.available_storage,
                                satellite_id=satellite_id,
                                quality_score=duration / (vis_end - vis_start).total_seconds()
                            ))

        # 按质量排序
        matching_windows.sort(key=lambda w: w.quality_score, reverse=True)
        return matching_windows

    def get_resource_timeline(self, satellite_id: str,
                             time_step: timedelta = timedelta(minutes=10)) -> List[TimeResourceSlot]:
        """
        获取资源时间线（用于可视化）

        Args:
            satellite_id: 卫星ID
            time_step: 时间步长

        Returns:
            List[TimeResourceSlot]: 资源时间线
        """
        sat_state = self.state.get_satellite_state(satellite_id)
        if not sat_state or not self.mission:
            return []

        timeline = []
        current_time = self.mission.start_time

        while current_time < self.mission.end_time:
            slot_end = min(current_time + time_step, self.mission.end_time)

            power, storage = sat_state.get_resource_at_time(current_time)

            # 计算该时段的利用率
            scheduled_in_slot = [
                t for t in sat_state.scheduled_tasks
                if t['imaging_start'] < slot_end and t['imaging_end'] > current_time
            ]
            busy_time = sum(
                (min(t['imaging_end'], slot_end) - max(t['imaging_start'], current_time)).total_seconds()
                for t in scheduled_in_slot
            )
            utilization = busy_time / (slot_end - current_time).total_seconds()

            timeline.append(TimeResourceSlot(
                start_time=current_time,
                end_time=slot_end,
                power_available=power,
                storage_available=sat_state.storage_capacity - storage,
                utilization_ratio=utilization
            ))

            current_time = slot_end

        return timeline

    def estimate_task_capacity(self, satellite_id: str,
                               avg_task_duration: float = 300.0,
                               avg_power_per_task: float = 50.0,
                               avg_storage_per_task: float = 5.0) -> int:
        """
        估计卫星还可容纳的任务数量

        Args:
            satellite_id: 卫星ID
            avg_task_duration: 平均任务时长（秒）
            avg_power_per_task: 平均每任务电量消耗（Wh）
            avg_storage_per_task: 平均每任务存储产生（GB）

        Returns:
            int: 估计可容纳的任务数
        """
        profile = self.calculate_remaining_resources(satellite_id)

        if profile.total_available_time <= 0:
            return 0

        # 基于不同资源的估算
        time_capacity = int(profile.total_available_time / avg_task_duration)
        power_capacity = int(profile.total_available_power / avg_power_per_task) if avg_power_per_task > 0 else float('inf')
        storage_capacity = int(profile.total_available_storage / avg_storage_per_task) if avg_storage_per_task > 0 else float('inf')

        # 取最小值
        return min(time_capacity, power_capacity, storage_capacity)

    def generate_resource_report(self) -> Dict[str, Any]:
        """生成资源使用报告"""
        all_resources = self.calculate_all_remaining_resources()

        total_time = sum(r.total_available_time for r in all_resources.values())
        total_windows = sum(r.window_count for r in all_resources.values())
        avg_utilization = sum(r.utilization_rate for r in all_resources.values()) / len(all_resources) if all_resources else 0.0

        sat_details = {
            sat_id: {
                'available_time_hours': r.total_available_time / 3600,
                'available_power_wh': r.total_available_power,
                'available_storage_gb': r.total_available_storage,
                'window_count': r.window_count,
                'utilization_rate': r.utilization_rate
            }
            for sat_id, r in all_resources.items()
        }

        return {
            'summary': {
                'total_satellites': len(all_resources),
                'total_available_time_hours': total_time / 3600,
                'total_windows': total_windows,
                'average_utilization_rate': avg_utilization
            },
            'satellite_details': sat_details
        }
