"""
卫星固存状态管理模块

管理卫星固存容量、使用情况和溢出检测。
"""

from dataclasses import dataclass


@dataclass
class StorageState:
    """卫星固存状态

    Attributes:
        capacity_gb: 总容量 (GB)
        current_gb: 当前使用量 (GB)
        overflow_threshold: 溢出阈值 (0-1), 默认100%
    """
    capacity_gb: float
    current_gb: float
    overflow_threshold: float = 1.0

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
