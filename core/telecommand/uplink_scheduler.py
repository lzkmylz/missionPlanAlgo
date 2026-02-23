"""
指令上行调度器

实现Chapter 17.4: 指令上行调度
确保任务执行前卫星已接收必要指令
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UplinkLinkType(Enum):
    """上行链路类型"""
    DIRECT = "direct"
    RELAY = "relay"


@dataclass
class UplinkWindow:
    """指令上行窗口"""
    satellite_id: str
    ground_station_id: str
    start_time: datetime
    end_time: datetime
    max_commands: int
    link_type: str  # 'direct', 'relay'


class UplinkScheduler:
    """
    指令上行调度器

    确保任务执行前卫星已接收必要指令
    """

    def __init__(self,
                 command_prep_time: int = 300,
                 command_execution_delay: int = 10):
        """
        初始化上行调度器

        Args:
            command_prep_time: 指令准备时间（秒）
            command_execution_delay: 指令执行延迟（秒）
        """
        self.command_prep_time = command_prep_time
        self.command_execution_delay = command_execution_delay
        self.uplink_windows: Dict[str, List[UplinkWindow]] = {}

    def add_uplink_window(self, window: UplinkWindow) -> None:
        """添加上行窗口"""
        sat_id = window.satellite_id
        if sat_id not in self.uplink_windows:
            self.uplink_windows[sat_id] = []
        self.uplink_windows[sat_id].append(window)

    def check_uplink_feasibility(self,
                                  satellite_id: str,
                                  task_start_time: datetime,
                                  command_complexity: str = 'standard') -> Tuple[bool, Optional[datetime]]:
        """
        检查任务开始前是否能完成指令上行

        Args:
            satellite_id: 卫星ID
            task_start_time: 任务开始时间
            command_complexity: 指令复杂度 ('simple', 'standard', 'complex')

        Returns:
            (是否可行, 最晚指令发送时间)
        """
        # 计算所需准备时间
        prep_times = {
            'simple': 60,
            'standard': 300,
            'complex': 900
        }
        required_prep = prep_times.get(command_complexity, 300)

        # 最晚指令到达卫星的时间
        latest_command_arrival = task_start_time - timedelta(
            seconds=self.command_execution_delay
        )

        # 最晚指令发送时间
        latest_uplink_time = latest_command_arrival - timedelta(seconds=required_prep)

        # 查找可用的上行窗口
        windows = self.uplink_windows.get(satellite_id, [])

        for window in windows:
            if window.end_time >= latest_uplink_time:
                if window.start_time <= latest_uplink_time:
                    return True, latest_uplink_time

        return False, None

    def schedule_uplink(self,
                       satellite_id: str,
                       task_id: str,
                       task_start_time: datetime,
                       command_complexity: str = 'standard') -> Optional[UplinkWindow]:
        """
        为任务调度上行窗口

        Args:
            satellite_id: 卫星ID
            task_id: 任务ID
            task_start_time: 任务开始时间
            command_complexity: 指令复杂度

        Returns:
            分配的上行窗口或None
        """
        feasible, _ = self.check_uplink_feasibility(
            satellite_id, task_start_time, command_complexity
        )

        if not feasible:
            logger.warning(f"Cannot find uplink window for task {task_id} on {satellite_id}")
            return None

        # 找到合适的窗口
        windows = self.uplink_windows.get(satellite_id, [])
        prep_times = {
            'simple': 60,
            'standard': 300,
            'complex': 900
        }
        required_prep = prep_times.get(command_complexity, 300)
        latest_uplink_time = task_start_time - timedelta(
            seconds=self.command_execution_delay + required_prep
        )

        for window in windows:
            if window.end_time >= latest_uplink_time:
                logger.info(f"Scheduled uplink for task {task_id} using window at {window.start_time}")
                return window

        return None

    def get_uplink_schedule(self, satellite_id: str) -> List[UplinkWindow]:
        """获取卫星的上行调度"""
        return self.uplink_windows.get(satellite_id, [])
