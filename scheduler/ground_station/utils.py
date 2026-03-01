"""
地面站调度器辅助函数

数传时长计算等通用工具函数。
"""


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
