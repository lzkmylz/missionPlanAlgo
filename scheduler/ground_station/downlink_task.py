"""
数传任务数据模型

定义数传任务的数据结构和相关方法。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DownlinkTask:
    """数传任务

    Attributes:
        task_id: 任务ID
        satellite_id: 卫星ID
        ground_station_id: 地面站ID
        start_time: 开始时间
        end_time: 结束时间
        data_size_gb: 数据量 (GB)
        antenna_id: 天线ID
        related_imaging_task_id: 关联的成像任务ID
        effective_data_rate: 实际使用的数据率 (Mbps)
        link_setup_time_seconds: 链路建立时间（秒），包含在总时长中
    """
    task_id: str
    satellite_id: str
    ground_station_id: str
    start_time: datetime
    end_time: datetime
    data_size_gb: float
    antenna_id: Optional[str] = None
    related_imaging_task_id: Optional[str] = None
    effective_data_rate: float = 300.0
    link_setup_time_seconds: float = 15.0

    def get_duration_seconds(self) -> float:
        """获取数传总时长（秒），包含链路建立时间"""
        return (self.end_time - self.start_time).total_seconds()

    def get_transmission_time_seconds(self) -> float:
        """获取纯数据传输时长（秒），不含链路建立时间"""
        total_duration = self.get_duration_seconds()
        return max(0.0, total_duration - self.link_setup_time_seconds)
