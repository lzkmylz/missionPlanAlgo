"""
地面站调度器辅助函数

数传时长计算等通用工具函数。
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.models.ground_station import Antenna

# 默认链路建立时间（秒）：指向 + 捕获 + 同步
# 天线指向：约5秒
# 信号捕获：约5秒
# 链路同步：约5秒
DEFAULT_LINK_SETUP_TIME_SECONDS = 15.0


def calculate_downlink_duration(
    data_size_gb: float,
    data_rate_mbps: float,
    link_setup_time_seconds: float = 0.0
) -> float:
    """计算数传所需时长（含链路建立时间）

    公式: duration = link_setup_time + data_transmission_time
    其中:
        - link_setup_time: 链路建立时间（指向、捕获、同步）
        - data_transmission_time = data_size_gb / (data_rate_mbps / 8 / 1024)
        - data_size_gb: 数据量 (GB)
        - data_rate_mbps: 数据传输速率 (Mbps)
        - 转换: Mbps -> MB/s = Mbps / 8 (bits to bytes)
        - 转换: MB/s -> GB/s = MB/s / 1024

    Args:
        data_size_gb: 数据量 (GB)
        data_rate_mbps: 数据传输速率 (Mbps)
        link_setup_time_seconds: 链路建立时间（秒），默认0秒

    Returns:
        总时长 (秒)，包含链路建立和数据传输

    Raises:
        ValueError: 如果 data_rate_mbps 为 0
    """
    if data_rate_mbps <= 0:
        raise ValueError("Data rate must be positive")

    if data_size_gb <= 0:
        return link_setup_time_seconds

    # 转换 Mbps 到 GB/s: Mbps / 8 = MB/s, MB/s / 1024 = GB/s
    data_rate_gb_per_sec = data_rate_mbps / 8 / 1024

    # 数据传输时间 + 链路建立时间
    transmission_time = data_size_gb / data_rate_gb_per_sec
    return link_setup_time_seconds + transmission_time


def calculate_downlink_duration_with_antenna(
    data_size_gb: float,
    data_rate_mbps: float,
    antenna: Optional['Antenna'] = None,
    use_antenna_acquisition_time: bool = True
) -> float:
    """计算数传所需时长（使用天线特定的建链时间）

    Args:
        data_size_gb: 数据量 (GB)
        data_rate_mbps: 数据传输速率 (Mbps)
        antenna: 天线对象（可选），用于获取特定建链时间
        use_antenna_acquisition_time: 是否使用天线的建链时间

    Returns:
        总时长 (秒)，包含建链和数据传输
    """
    if data_rate_mbps <= 0:
        raise ValueError("Data rate must be positive")

    # 获取建链时间
    acquisition_time = 0.0
    if use_antenna_acquisition_time and antenna is not None:
        acquisition_time = getattr(antenna, 'acquisition_time_seconds', 15.0)

    if data_size_gb <= 0:
        return acquisition_time

    # 转换 Mbps 到 GB/s
    data_rate_gb_per_sec = data_rate_mbps / 8 / 1024

    # 数据传输时间 + 建链时间
    transmission_time = data_size_gb / data_rate_gb_per_sec
    return acquisition_time + transmission_time


def calculate_required_time_window(
    data_size_gb: float,
    data_rate_mbps: float,
    acquisition_time_seconds: float,
    switch_time_seconds: float = 0.0
) -> float:
    """计算所需的总时间窗口（包含所有缓冲时间）

    Args:
        data_size_gb: 数据量 (GB)
        data_rate_mbps: 数据传输速率 (Mbps)
        acquisition_time_seconds: 建链时间（秒）
        switch_time_seconds: 切换缓冲时间（秒）

    Returns:
        总所需时间（秒）
    """
    # 数据传输时间
    if data_rate_mbps <= 0:
        raise ValueError("Data rate must be positive")

    data_rate_gb_per_sec = data_rate_mbps / 8 / 1024
    transmission_time = data_size_gb / data_rate_gb_per_sec if data_size_gb > 0 else 0.0

    # 总时间 = 切换缓冲 + 建链 + 数据传输
    return switch_time_seconds + acquisition_time_seconds + transmission_time
