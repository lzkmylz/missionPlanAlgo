"""
自适应时间步长计算器

Phase 1优化：通过粗扫描+精化减少计算点数量
- 粗扫描：300秒步长快速定位潜在窗口
- 精化：60秒步长在窗口边界精确计算

预期性能提升：4-5倍
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoarseWindow:
    """粗扫描发现的潜在窗口"""
    start_time: datetime
    end_time: datetime
    is_potentially_visible: bool
    max_elevation: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """窗口持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()


class AdaptiveStepCalculator:
    """
    自适应时间步长计算器

    使用两阶段算法减少计算点：
    1. 粗扫描：大步长快速定位
    2. 精化：小步长精确边界
    """

    def __init__(
        self,
        coarse_step: int = 300,  # 5分钟
        fine_step: int = 60,     # 1分钟
        min_window_duration: int = 60  # 最小窗口持续时间（秒）
    ):
        """
        初始化自适应步长计算器

        Args:
            coarse_step: 粗扫描步长（秒，默认300=5分钟）
            fine_step: 精化步长（秒，默认60=1分钟）
            min_window_duration: 最小窗口持续时间（秒）
        """
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.min_window_duration = min_window_duration

        # 统计信息
        self._coarse_points_computed = 0
        self._fine_points_computed = 0

    def compute_windows(
        self,
        is_visible_func: Callable[[datetime], bool],
        start_time: datetime,
        end_time: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        使用自适应步长计算可见窗口

        Args:
            is_visible_func: 可见性检查函数，接收时间返回bool
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[(start, end)]: 可见窗口列表
        """
        # 阶段1: 粗扫描
        coarse_timestamps = self._generate_coarse_positions(start_time, end_time)
        # 对每个时间点进行可见性检查
        coarse_positions = [
            (timestamp, is_visible_func(timestamp))
            for timestamp, _ in coarse_timestamps
        ]
        coarse_windows = self._coarse_scan_from_positions(coarse_positions)

        # 阶段2: 精化
        refined_windows = []
        for window in coarse_windows:
            if window.is_potentially_visible:
                start, end = self._refine_window_boundaries(
                    is_visible_func,
                    window.start_time,
                    window.end_time,
                    start_time,
                    end_time
                )

                # 检查最小持续时间
                if (end - start).total_seconds() >= self.min_window_duration:
                    refined_windows.append((start, end))

        logger.debug(
            f"Adaptive step: {len(coarse_positions)} coarse points, "
            f"{self._fine_points_computed} fine points, "
            f"{len(refined_windows)} windows found"
        )

        return refined_windows

    def _generate_coarse_positions(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Tuple[datetime, bool]]:
        """
        生成粗扫描时间点和可见性

        注意：这个方法只生成时间点，实际的可见性检查需要外部提供
        这里返回一个占位符，用于接口兼容性

        Returns:
            List[(datetime, bool)]: 时间点和占位可见性
        """
        positions = []
        current = start_time

        while current <= end_time:
            positions.append((current, False))  # 占位符
            current += timedelta(seconds=self.coarse_step)

        self._coarse_points_computed = len(positions)
        return positions

    def _coarse_scan_from_positions(
        self,
        positions: List[Tuple[datetime, bool]]
    ) -> List[CoarseWindow]:
        """
        从位置数据执行粗扫描，识别潜在窗口

        Args:
            positions: List[(timestamp, is_visible)]

        Returns:
            List[CoarseWindow]: 粗扫描窗口列表（只包含可见窗口）
        """
        if not positions:
            return []

        windows = []
        current_start = positions[0][0]
        current_visible = positions[0][1]
        max_elevation = 0.0

        for i, (timestamp, is_visible) in enumerate(positions[1:], 1):
            if is_visible != current_visible:
                # 状态变化，结束当前窗口
                # 只添加可见窗口
                if current_visible:
                    window = CoarseWindow(
                        start_time=current_start,
                        end_time=timestamp,
                        is_potentially_visible=True,
                        max_elevation=max_elevation
                    )
                    windows.append(window)

                # 开始新窗口
                current_start = timestamp
                current_visible = is_visible
                max_elevation = 0.0

        # 处理最后一个窗口
        if positions and current_visible:
            window = CoarseWindow(
                start_time=current_start,
                end_time=positions[-1][0],
                is_potentially_visible=True,
                max_elevation=max_elevation
            )
            windows.append(window)

        return windows

    def _refine_window_boundaries(
        self,
        is_visible_func: Callable[[datetime], bool],
        coarse_start: datetime,
        coarse_end: datetime,
        mission_start: datetime,
        mission_end: datetime
    ) -> Tuple[datetime, datetime]:
        """
        精化窗口边界

        在粗略窗口边界附近使用小步长搜索精确边界

        Args:
            is_visible_func: 可见性检查函数
            coarse_start: 粗略开始时间
            coarse_end: 粗略结束时间
            mission_start: 任务开始时间边界
            mission_end: 任务结束时间边界

        Returns:
            (precise_start, precise_end): 精化后的时间边界
        """
        fine_step = timedelta(seconds=self.fine_step)

        # 向前扩展找精确开始时间
        precise_start = coarse_start
        while precise_start > mission_start:
            prev_time = precise_start - fine_step
            if prev_time < mission_start:
                break
            try:
                if not is_visible_func(prev_time):
                    break
                precise_start = prev_time
                self._fine_points_computed += 1
            except Exception:
                break

        # 向后扩展找精确结束时间
        precise_end = coarse_end
        while precise_end < mission_end:
            next_time = precise_end + fine_step
            if next_time > mission_end:
                break
            try:
                if not is_visible_func(next_time):
                    break
                precise_end = next_time
                self._fine_points_computed += 1
            except Exception:
                break

        return precise_start, precise_end

    def _estimate_coarse_points(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """
        估算粗扫描需要的计算点数量

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            int: 估算的计算点数量
        """
        total_seconds = (end_time - start_time).total_seconds()
        return int(total_seconds / self.coarse_step) + 1

    def get_statistics(self) -> dict:
        """
        获取计算统计信息

        Returns:
            dict: 统计信息
        """
        return {
            'coarse_points_computed': self._coarse_points_computed,
            'fine_points_computed': self._fine_points_computed,
            'total_points_computed': (
                self._coarse_points_computed + self._fine_points_computed
            ),
            'coarse_step': self.coarse_step,
            'fine_step': self.fine_step,
        }

    def reset_statistics(self):
        """重置统计信息"""
        self._coarse_points_computed = 0
        self._fine_points_computed = 0
