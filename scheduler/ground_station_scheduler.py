"""
地面站调度器 - 向后兼容的导入入口

此模块已重构为子模块结构，保持向后兼容：
- scheduler.ground_station.storage: StorageState
- scheduler.ground_station.downlink_task: DownlinkTask
- scheduler.ground_station.utils: calculate_downlink_duration
- scheduler.ground_station.scheduler: GroundStationScheduler

原有导入方式仍然有效：
    from scheduler.ground_station_scheduler import GroundStationScheduler
"""

# 从子模块导入，保持向后兼容
from scheduler.ground_station.storage import StorageState
from scheduler.ground_station.downlink_task import DownlinkTask
from scheduler.ground_station.utils import calculate_downlink_duration, DEFAULT_LINK_SETUP_TIME_SECONDS
from scheduler.ground_station.scheduler import GroundStationScheduler, GroundStationScheduleResult

__all__ = [
    'StorageState',
    'DownlinkTask',
    'calculate_downlink_duration',
    'DEFAULT_LINK_SETUP_TIME_SECONDS',
    'GroundStationScheduler',
    'GroundStationScheduleResult',
]
