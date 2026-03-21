"""
指令上注渠道类型定义

定义三种指令上注渠道及统一的上注弧段数据类：
  - GROUND_STATION: 通过地面站 (GS: 前缀窗口)
  - RELAY_SATELLITE: 通过中继卫星 (RELAY: 前缀窗口)
  - ISL: 通过星间链路 (ISL: 前缀窗口)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class UplinkChannelType(Enum):
    """指令上注渠道类型"""
    GROUND_STATION = "ground_station"
    RELAY_SATELLITE = "relay_satellite"
    ISL = "isl"


# 各渠道默认切换开销（秒）
_DEFAULT_SWITCHING_OVERHEAD = {
    UplinkChannelType.GROUND_STATION: 30.0,
    UplinkChannelType.RELAY_SATELLITE: 30.0,
    UplinkChannelType.ISL: 5.0,  # ISL无机械指向，切换更快
}


@dataclass
class UplinkPass:
    """单条指令上注弧段

    统一表示来自地面站、中继卫星或星间链路的可用上注窗口。

    Attributes:
        channel_type: 上注渠道类型
        channel_id: 渠道标识（地面站ID / 中继星ID / 邻星ID）
        satellite_id: 被服务的卫星ID
        start_time: 弧段开始时间
        end_time: 弧段结束时间
        max_data_rate_mbps: 最大数据率（Mbps）
        switching_overhead_s: 切换开销（秒），同一天线切换至下一颗卫星所需最少间隔
    """
    channel_type: UplinkChannelType
    channel_id: str
    satellite_id: str
    start_time: datetime
    end_time: datetime
    max_data_rate_mbps: float = 10.0
    switching_overhead_s: float = 30.0

    @property
    def duration_s(self) -> float:
        """弧段持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def usable_duration_s(self) -> float:
        """可用时长（扣除切换开销后）"""
        return max(0.0, self.duration_s - self.switching_overhead_s)

    @property
    def capacity_mb(self) -> float:
        """可传输数据容量（MB）"""
        return self.max_data_rate_mbps * self.usable_duration_s / 8.0
