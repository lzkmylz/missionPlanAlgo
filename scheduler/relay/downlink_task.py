"""
中继卫星数传任务数据模型

定义通过中继卫星传输数据的数据结构和相关方法。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class RelayDownlinkTask:
    """中继卫星数传任务

    Attributes:
        task_id: 任务ID
        satellite_id: 卫星ID（LEO卫星）
        relay_id: 中继卫星ID
        start_time: 开始时间
        end_time: 结束时间
        data_size_gb: 数据量 (GB)
        related_imaging_task_id: 关联的成像任务ID
        effective_data_rate: 实际使用的数据率 (Mbps)
        acquisition_time_seconds: 建链时间（秒）
    """
    task_id: str
    satellite_id: str
    relay_id: str
    start_time: datetime
    end_time: datetime
    data_size_gb: float
    related_imaging_task_id: Optional[str] = None
    effective_data_rate: float = 450.0  # 中继通常有更高带宽
    acquisition_time_seconds: float = 10.0  # 中继建链通常更快

    def __post_init__(self):
        """初始化后验证"""
        if self.data_size_gb <= 0:
            raise ValueError("数据量必须大于0")

    def get_duration_seconds(self) -> float:
        """获取数传总时长（秒）"""
        return (self.end_time - self.start_time).total_seconds()

    def get_transmission_time_seconds(self) -> float:
        """获取纯数据传输时长（秒），不含建链时间"""
        total_duration = self.get_duration_seconds()
        return max(0.0, total_duration - self.acquisition_time_seconds)

    def get_actual_data_transferred(self) -> float:
        """计算实际可传输的数据量（GB）"""
        transmission_time = self.get_transmission_time_seconds()
        # Mbps * seconds / 8000 = GB
        return (self.effective_data_rate * transmission_time) / 8000.0

    def is_sufficient_for_data(self) -> bool:
        """检查窗口是否足够传输所有数据"""
        return self.get_actual_data_transferred() >= self.data_size_gb

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'satellite_id': self.satellite_id,
            'relay_id': self.relay_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'data_size_gb': self.data_size_gb,
            'related_imaging_task_id': self.related_imaging_task_id,
            'effective_data_rate': self.effective_data_rate,
            'acquisition_time_seconds': self.acquisition_time_seconds,
        }
