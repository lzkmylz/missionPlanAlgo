"""
质量感知调度mixin

为调度器提供质量评分感知能力，支持：
- 窗口质量评估
- 质量筛选
- 质量排序
- 质量缓存
"""

import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime

from core.quality.window_quality import (
    WindowQualityCalculator,
    WindowQualityScore,
    QualityScoreConfig,
)
from core.quality.quality_config import DEFAULT_QUALITY_CONFIG, QualityTier

if TYPE_CHECKING:
    from core.orbit.visibility.base import VisibilityWindow
    from core.models.satellite import Satellite
    from core.models.mission import Mission

logger = logging.getLogger(__name__)


class QualityAwareMixin:
    """
    质量感知调度mixin

    为调度器添加质量评分感知能力
    """

    def __init_quality_aware__(
        self,
        quality_config: Optional[QualityScoreConfig] = None,
        enable_quality_filtering: bool = True,
        min_quality_threshold: float = 0.3,
    ):
        """
        初始化质量感知组件

        Args:
            quality_config: 质量评分配置
            enable_quality_filtering: 是否启用质量筛选
            min_quality_threshold: 最低质量阈值
        """
        self.quality_config = quality_config or DEFAULT_QUALITY_CONFIG
        self.quality_calculator = WindowQualityCalculator(self.quality_config)
        self.enable_quality_filtering = enable_quality_filtering
        self.min_quality_threshold = min_quality_threshold

        # 缓存
        self._window_quality_cache: Dict[str, WindowQualityScore] = {}
        self._quality_stats = {
            'evaluated': 0,
            'filtered': 0,
            'high_quality': 0,
        }

    def evaluate_window_quality(
        self,
        window: 'VisibilityWindow',
        satellite: 'Satellite',
        mission: Optional['Mission'] = None,
        ground_station_windows: Optional[List['VisibilityWindow']] = None,
    ) -> WindowQualityScore:
        """
        评估窗口质量（带缓存）

        Args:
            window: 可见性窗口
            satellite: 卫星
            mission: 任务（可选）
            ground_station_windows: 地面站窗口（可选）

        Returns:
            WindowQualityScore
        """
        # 生成缓存键
        cache_key = f"{satellite.id}:{window.start_time.isoformat()}:{window.target_id}"

        # 检查缓存
        if cache_key in self._window_quality_cache:
            return self._window_quality_cache[cache_key]

        # 计算质量
        score = self.quality_calculator.calculate_quality(
            window=window,
            satellite=satellite,
            mission=mission,
            config=self.quality_config,
            ground_station_windows=ground_station_windows,
        )

        # 添加到缓存
        self._window_quality_cache[cache_key] = score
        self._quality_stats['evaluated'] += 1

        if score.quality_tier == QualityTier.HIGH:
            self._quality_stats['high_quality'] += 1

        return score

    def filter_windows_by_quality(
        self,
        windows: List['VisibilityWindow'],
        satellite: 'Satellite',
        min_quality: Optional[float] = None,
        mission: Optional['Mission'] = None,
    ) -> List['VisibilityWindow']:
        """
        按质量筛选窗口

        Args:
            windows: 窗口列表
            satellite: 卫星
            min_quality: 最低质量阈值（None则使用配置值）
            mission: 任务（可选）

        Returns:
            质量合格的窗口列表
        """
        if not self.enable_quality_filtering:
            return windows

        threshold = min_quality if min_quality is not None else self.min_quality_threshold

        result = []
        for window in windows:
            score = self.evaluate_window_quality(window, satellite, mission)
            if score.overall_score >= threshold:
                result.append(window)
            else:
                self._quality_stats['filtered'] += 1

        return result

    def sort_windows_by_quality(
        self,
        windows: List['VisibilityWindow'],
        satellite: 'Satellite',
        mission: Optional['Mission'] = None,
        reverse: bool = True,
    ) -> List[tuple]:
        """
        按质量排序窗口

        Args:
            windows: 窗口列表
            satellite: 卫星
            mission: 任务（可选）
            reverse: 是否降序（高质量在前）

        Returns:
            (窗口, 评分)元组列表
        """
        scored_windows = [
            (window, self.evaluate_window_quality(window, satellite, mission))
            for window in windows
        ]

        scored_windows.sort(
            key=lambda x: x[1].overall_score,
            reverse=reverse
        )

        return scored_windows

    def select_best_quality_window(
        self,
        windows: List['VisibilityWindow'],
        satellite: 'Satellite',
        mission: Optional['Mission'] = None,
    ) -> Optional[tuple]:
        """
        选择质量最高的窗口

        Args:
            windows: 窗口列表
            satellite: 卫星
            mission: 任务（可选）

        Returns:
            (窗口, 评分)或None
        """
        if not windows:
            return None

        scored = self.sort_windows_by_quality(windows, satellite, mission, reverse=True)
        return scored[0] if scored else None

    def get_quality_stats(self) -> Dict[str, Any]:
        """获取质量统计"""
        stats = self._quality_stats.copy()
        stats['cache_size'] = len(self._window_quality_cache)
        stats['cache_hit_rate'] = self.quality_calculator.get_cache_stats().get('hit_rate', 0)
        return stats

    def clear_quality_cache(self) -> None:
        """清空质量缓存"""
        self._window_quality_cache.clear()
        self.quality_calculator.clear_cache()

    def calculate_quality_score_for_assignment(
        self,
        window: 'VisibilityWindow',
        satellite: 'Satellite',
        task: Any,
        mission: Optional['Mission'] = None,
    ) -> float:
        """
        计算任务分配的质量评分（用于启发式函数）

        Args:
            window: 可见性窗口
            satellite: 卫星
            task: 任务
            mission: 任务（可选）

        Returns:
            质量评分（0-1）
        """
        score = self.evaluate_window_quality(window, satellite, mission)
        return score.overall_score
