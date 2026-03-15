"""
数传任务数据模型

定义数传任务的数据结构和相关方法。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class DownlinkTask:
    """数传任务

    Attributes:
        task_id: 任务ID
        satellite_id: 卫星ID
        ground_station_id: 地面站ID
        start_time: 开始时间（包含建链时间）
        end_time: 结束时间
        data_size_gb: 数据量 (GB)
        antenna_id: 天线ID
        related_imaging_task_id: 关联的成像任务ID
        effective_data_rate: 实际使用的数据率 (Mbps)
        acquisition_time_seconds: 天线建链时间（秒），包含在总时长中
        switch_time_seconds: 与前任务的切换时间（秒），用于资源冲突检查
        actual_transmission_start: 实际数据传输开始时间（建链完成后）
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
    acquisition_time_seconds: float = 15.0
    switch_time_seconds: float = 0.0  # 与前任务的切换时间
    actual_transmission_start: Optional[datetime] = None  # 实际传输开始

    def __post_init__(self):
        """初始化后计算实际传输开始时间"""
        if self.actual_transmission_start is None:
            self.actual_transmission_start = self.start_time + timedelta(
                seconds=self.acquisition_time_seconds
            )

    def get_duration_seconds(self) -> float:
        """获取数传总时长（秒），包含建链时间和切换时间"""
        return (self.end_time - self.start_time).total_seconds()

    def get_transmission_time_seconds(self) -> float:
        """获取纯数据传输时长（秒），不含建链时间"""
        if self.actual_transmission_start:
            return (self.end_time - self.actual_transmission_start).total_seconds()
        total_duration = self.get_duration_seconds()
        return max(0.0, total_duration - self.acquisition_time_seconds)

    def get_total_reserved_time(self) -> float:
        """
        获取总预留时间（秒）- 包含切换缓冲、建链和数据传输
        用于资源冲突检查
        """
        return self.get_duration_seconds() + self.switch_time_seconds
