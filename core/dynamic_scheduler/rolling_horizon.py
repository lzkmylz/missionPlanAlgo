"""
滚动时间窗管理器

实现第19章设计的滚动时间窗管理功能，用于动态调度中的周期性局部优化。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


@dataclass
class RollingHorizonConfig:
    """
    滚动时间窗配置

    Attributes:
        window_size: 优化窗口大小，默认为2小时
        shift_interval: 窗口滚动间隔，默认为15分钟
        freeze_duration: 冻结时间（已执行或即将执行的任务不可更改），默认为5分钟
        optimization_method: 优化方法，'fast_heuristic'（快速启发式）或'metaheuristic'（元启发式）
        max_optimization_time: 最大优化时间（秒），默认为30秒
    """
    window_size: timedelta = timedelta(hours=2)
    shift_interval: timedelta = timedelta(minutes=15)
    freeze_duration: timedelta = timedelta(minutes=5)
    optimization_method: str = 'fast_heuristic'
    max_optimization_time: int = 30

    def __post_init__(self):
        """验证配置值的有效性"""
        if self.optimization_method not in ('fast_heuristic', 'metaheuristic'):
            raise ValueError(
                f"Invalid optimization_method: {self.optimization_method}. "
                "Must be 'fast_heuristic' or 'metaheuristic'"
            )
        if self.max_optimization_time <= 0:
            raise ValueError("max_optimization_time must be positive")


class RollingHorizonManager:
    """
    滚动时间窗管理器 - 周期性触发重优化

    负责管理动态调度中的滚动时间窗，决定何时触发优化以及计算当前优化窗口。

    Example:
        >>> config = RollingHorizonConfig(
        ...     window_size=timedelta(hours=2),
        ...     shift_interval=timedelta(minutes=15)
        ... )
        >>> manager = RollingHorizonManager(config)
        >>> current_time = datetime.now(timezone.utc)
        >>> if manager.should_trigger_optimization(current_time):
        ...     window_start, window_end, freeze_until = manager.get_optimization_window(current_time)
        ...     # 执行优化...
    """

    def __init__(self, config: Optional[RollingHorizonConfig] = None):
        """
        初始化滚动时间窗管理器

        Args:
            config: 滚动时间窗配置，如果为None则使用默认配置
        """
        self.config = config or RollingHorizonConfig()
        self.last_optimization_time: Optional[datetime] = None

    def should_trigger_optimization(self, current_time: datetime) -> bool:
        """
        检查是否应该触发优化

        根据上次优化时间和滚动间隔决定是否触发新的优化。

        Args:
            current_time: 当前时间（必须是timezone-aware datetime）

        Returns:
            如果应该触发优化返回True，否则返回False

        Raises:
            TypeError: 如果current_time不是datetime类型
        """
        if not isinstance(current_time, datetime):
            raise TypeError("current_time must be a datetime instance")

        if self.last_optimization_time is None:
            return True

        # 处理当前时间早于上次优化时间的情况（时钟回拨等）
        if current_time < self.last_optimization_time:
            return True

        elapsed = current_time - self.last_optimization_time
        return elapsed >= self.config.shift_interval

    def get_optimization_window(
        self,
        current_time: datetime
    ) -> Tuple[datetime, datetime, datetime]:
        """
        获取当前优化窗口

        计算优化窗口的开始时间、结束时间和冻结截止时间。
        冻结时间内的任务不可更改，窗口从冻结结束后开始。

        Args:
            current_time: 当前时间（必须是timezone-aware datetime）

        Returns:
            三元组 (window_start, window_end, freeze_until)
            - window_start: 优化窗口开始时间（冻结结束后）
            - window_end: 优化窗口结束时间
            - freeze_until: 冻结截止时间（此时间前的任务不可更改）

        Raises:
            TypeError: 如果current_time不是datetime类型
            ValueError: 如果current_time没有时区信息
        """
        if not isinstance(current_time, datetime):
            raise TypeError("current_time must be a datetime instance")

        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")

        freeze_until = current_time + self.config.freeze_duration
        window_start = freeze_until
        window_end = current_time + self.config.window_size

        return window_start, window_end, freeze_until

    def record_optimization(self, optimization_time: datetime) -> None:
        """
        记录优化完成时间

        在优化完成后调用，更新last_optimization_time。

        Args:
            optimization_time: 优化完成时间
        """
        if not isinstance(optimization_time, datetime):
            raise TypeError("optimization_time must be a datetime instance")

        self.last_optimization_time = optimization_time

    def get_time_until_next_optimization(self, current_time: datetime) -> Optional[timedelta]:
        """
        获取距离下次优化的时间

        Args:
            current_time: 当前时间

        Returns:
            距离下次优化的时间，如果从未优化过则返回None
        """
        if self.last_optimization_time is None:
            return None

        if current_time < self.last_optimization_time:
            return timedelta(0)

        elapsed = current_time - self.last_optimization_time
        remaining = self.config.shift_interval - elapsed

        return max(remaining, timedelta(0))
