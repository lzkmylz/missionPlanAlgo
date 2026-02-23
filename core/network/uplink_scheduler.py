"""
上行调度器

实现第17章设计：
- UplinkWindow数据类
- UplinkScheduler类（check_uplink_feasibility方法）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class UplinkWindow:
    """
    上行窗口数据类

    表示卫星与地面站之间的指令上行时间窗口
    """
    ground_station_id: str
    start_time: datetime
    end_time: datetime
    max_data_rate_mbps: float = 10.0
    elevation_angle: float = 45.0
    range_km: float = 1000.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """窗口持续时间"""
        return self.end_time - self.start_time

    def get_capacity_mb(self) -> float:
        """
        计算窗口可传输的数据容量

        Returns:
            容量（MB）
        """
        duration_seconds = self.duration.total_seconds()
        # Mbps * seconds / 8 = MB
        return self.max_data_rate_mbps * duration_seconds / 8.0

    def overlaps(self, other: 'UplinkWindow') -> bool:
        """
        检查是否与另一个窗口重叠

        Args:
            other: 另一个上行窗口

        Returns:
            是否重叠
        """
        return (
            self.ground_station_id == other.ground_station_id and
            self.start_time < other.end_time and
            self.end_time > other.start_time
        )


class UplinkScheduler:
    """
    指令上行调度器

    管理卫星指令上行窗口，检查上行可行性
    """

    # 默认配置
    DEFAULT_PREPARATION_LEAD_TIME = timedelta(minutes=30)
    DEFAULT_MIN_ELEVATION_ANGLE = 10.0
    DEFAULT_MAX_RANGE_KM = 2000.0

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化上行调度器

        Args:
            config: 配置参数
                - preparation_lead_time_minutes: 指令准备提前时间（分钟）
                - min_elevation_angle: 最小仰角（度）
                - max_range_km: 最大通信距离（公里）
        """
        self.config = config or {}
        self.ground_station_windows: Dict[str, List[UplinkWindow]] = {}

        # 配置参数
        prep_minutes = self.config.get('preparation_lead_time_minutes', 30)
        self.preparation_lead_time = timedelta(minutes=prep_minutes)
        self.min_elevation_angle = self.config.get(
            'min_elevation_angle', self.DEFAULT_MIN_ELEVATION_ANGLE
        )
        self.max_range_km = self.config.get(
            'max_range_km', self.DEFAULT_MAX_RANGE_KM
        )

    def add_window(self, window: UplinkWindow) -> None:
        """
        添加上行窗口

        Args:
            window: 上行窗口
        """
        gs_id = window.ground_station_id
        if gs_id not in self.ground_station_windows:
            self.ground_station_windows[gs_id] = []
        self.ground_station_windows[gs_id].append(window)

        # 按开始时间排序
        self.ground_station_windows[gs_id].sort(key=lambda w: w.start_time)

    def get_windows_for_ground_station(self, ground_station_id: str) -> List[UplinkWindow]:
        """
        获取指定地面站的所有窗口

        Args:
            ground_station_id: 地面站ID

        Returns:
            上行窗口列表
        """
        return self.ground_station_windows.get(ground_station_id, []).copy()

    def get_windows_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        ground_station_id: Optional[str] = None
    ) -> List[UplinkWindow]:
        """
        获取时间范围内的上行窗口

        Args:
            start_time: 开始时间
            end_time: 结束时间
            ground_station_id: 可选的地面站ID过滤

        Returns:
            上行窗口列表
        """
        results = []

        gs_ids = [ground_station_id] if ground_station_id else self.ground_station_windows.keys()

        for gs_id in gs_ids:
            for window in self.ground_station_windows.get(gs_id, []):
                # 检查窗口是否在时间范围内
                if window.start_time < end_time and window.end_time > start_time:
                    results.append(window)

        return sorted(results, key=lambda w: w.start_time)

    def check_uplink_feasibility(
        self,
        ground_station_id: str,
        command_size_mb: float,
        earliest_transmission_time: datetime
    ) -> Tuple[bool, Optional[UplinkWindow], str]:
        """
        检查指令上行可行性

        Args:
            ground_station_id: 地面站ID
            command_size_mb: 指令数据大小（MB）
            earliest_transmission_time: 最早可传输时间

        Returns:
            (是否可行, 可用窗口, 原因)
        """
        # 检查地面站是否存在
        if ground_station_id not in self.ground_station_windows:
            return False, None, f"Ground station {ground_station_id} not found"

        # 计算最早可用时间（考虑准备时间）
        earliest_available = earliest_transmission_time + self.preparation_lead_time

        # 查找合适的窗口
        for window in self.ground_station_windows[ground_station_id]:
            # 检查时间是否足够（考虑准备时间）
            if window.start_time < earliest_available:
                continue

            # 检查仰角约束
            if window.elevation_angle < self.min_elevation_angle:
                continue

            # 检查距离约束
            if window.range_km > self.max_range_km:
                continue

            # 检查容量
            if command_size_mb > 0 and window.get_capacity_mb() < command_size_mb:
                continue

            # 找到合适的窗口
            return True, window, ""

        # 没有找到合适的窗口
        if not self.ground_station_windows[ground_station_id]:
            return False, None, "No uplink windows available for this ground station"

        # 检查具体原因
        latest_window = self.ground_station_windows[ground_station_id][-1]
        if latest_window.end_time < earliest_available:
            return False, None, "Insufficient preparation time before next window"

        return False, None, "No suitable uplink window found (check elevation, range, or capacity)"

    def schedule_uplink(
        self,
        ground_station_id: str,
        command: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        调度单条指令上行

        Args:
            ground_station_id: 地面站ID
            command: 指令字典
                - id: 指令ID
                - size_mb: 数据大小
                - earliest_transmission: 最早传输时间

        Returns:
            调度结果或None
        """
        earliest_time = command.get('earliest_transmission', datetime.now())
        size_mb = command.get('size_mb', 0.0)

        is_feasible, window, reason = self.check_uplink_feasibility(
            ground_station_id, size_mb, earliest_time
        )

        if not is_feasible:
            return None

        return {
            'command_id': command.get('id'),
            'ground_station_id': ground_station_id,
            'start_time': window.start_time,
            'scheduled_time': window.start_time,
            'end_time': window.end_time,
            'window': window,
            'transmission_duration': self._calculate_transmission_time(size_mb, window)
        }

    def schedule_multiple_uplinks(
        self,
        ground_station_id: str,
        commands: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        调度多条指令上行

        Args:
            ground_station_id: 地面站ID
            commands: 指令列表

        Returns:
            调度结果列表
        """
        scheduled = []
        last_end_time = datetime.min

        for command in commands:
            # 确保指令按顺序调度
            earliest_time = max(
                command.get('earliest_transmission', datetime.now()),
                last_end_time
            )

            result = self.schedule_uplink(ground_station_id, {
                **command,
                'earliest_transmission': earliest_time
            })

            if result:
                scheduled.append(result)
                last_end_time = result['end_time']

        return scheduled

    def find_best_ground_station(
        self,
        command: Dict[str, Any],
        candidate_gs: List[str]
    ) -> Tuple[Optional[str], Optional[UplinkWindow]]:
        """
        查找最佳地面站

        Args:
            command: 指令
            candidate_gs: 候选地面站ID列表

        Returns:
            (最佳地面站ID, 窗口)
        """
        earliest_time = command.get('earliest_transmission', datetime.now())
        size_mb = command.get('size_mb', 0.0)

        best_gs = None
        best_window = None
        earliest_start = datetime.max

        for gs_id in candidate_gs:
            is_feasible, window, _ = self.check_uplink_feasibility(
                gs_id, size_mb, earliest_time
            )

            if is_feasible and window.start_time < earliest_start:
                earliest_start = window.start_time
                best_gs = gs_id
                best_window = window

        return best_gs, best_window

    def _calculate_transmission_time(
        self,
        data_size_mb: float,
        window: UplinkWindow
    ) -> timedelta:
        """
        计算传输时间

        Args:
            data_size_mb: 数据大小（MB）
            window: 上行窗口

        Returns:
            传输时间
        """
        if window.max_data_rate_mbps <= 0:
            return timedelta(seconds=0)

        # MB * 8 = Mbits, Mbits / Mbps = seconds
        seconds = (data_size_mb * 8.0) / window.max_data_rate_mbps
        return timedelta(seconds=seconds)
