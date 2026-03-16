"""
区域目标可见性计算器

计算瓦片级可见性和区域目标的可见性聚合
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from core.models import (
    Satellite, Target, TargetType, MosaicTile,
    TileVisibilityInfo
)
from .base import VisibilityWindow
from .base import VisibilityCalculator

__all__ = [
    'AreaVisibilityCalculator',
    'TileVisibilityCache',
]

logger = logging.getLogger(__name__)


class AreaVisibilityCalculator:
    """
    区域目标可见性计算器

    专门用于计算区域目标（拼幅瓦片）的可见性窗口
    支持批量计算和可见性聚合

    Attributes:
        base_calculator: 基础可见性计算器
        propagator: 轨道传播器
        min_elevation: 最小仰角（度）
        max_workers: 并行计算工作线程数
    """

    def __init__(self,
                 base_calculator: Optional[VisibilityCalculator] = None,
                 propagator: Optional[Any] = None,
                 min_elevation: float = 5.0,
                 max_workers: int = 4):
        """
        初始化区域可见性计算器

        Args:
            base_calculator: 基础可见性计算器（可选）
            propagator: 轨道传播器（可选）
            min_elevation: 最小仰角（度）
            max_workers: 并行计算工作线程数
        """
        self.base_calculator = base_calculator
        self.propagator = propagator
        self.min_elevation = min_elevation
        self.max_workers = max_workers

    def compute_tile_visibility(self,
                                satellite: Satellite,
                                tile: MosaicTile,
                                start_time: datetime,
                                end_time: datetime,
                                time_step: timedelta = timedelta(seconds=60)
                                ) -> List[VisibilityWindow]:
        """
        计算单个瓦片的可见性窗口

        与点目标不同，瓦片需要检查整个区域是否在视场内

        Args:
            satellite: 卫星
            tile: 瓦片
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 简化为点目标计算：使用瓦片中心
        # 实际实现应该检查整个瓦片是否在FOV内
        center_lon, center_lat = tile.center

        # 创建临时点目标用于计算
        temp_target = Target(
            id=tile.tile_id,
            name=f"Tile {tile.tile_id}",
            target_type=TargetType.POINT,
            longitude=center_lon,
            latitude=center_lat,
            priority=tile.priority,
        )

        # 使用基础计算器计算可见性
        if self.base_calculator:
            windows = self.base_calculator.compute_satellite_target_windows(
                satellite, temp_target, start_time, end_time, time_step
            )
        else:
            # 简化的可见性计算
            windows = self._compute_simple_visibility(
                satellite, temp_target, start_time, end_time, time_step
            )

        # 标记为拼幅窗口
        for window in windows:
            window = self._mark_as_mosaic_window(window, tile)

        return windows

    def _mark_as_mosaic_window(self, window: VisibilityWindow,
                               tile: MosaicTile) -> VisibilityWindow:
        """标记窗口为拼幅窗口"""
        from dataclasses import replace
        return replace(
            window,
            tile_id=tile.tile_id,
            is_mosaic_window=True,
            area_coverage_fraction=1.0,  # 默认完全覆盖
        )

    def compute_area_aggregate_visibility(self,
                                          satellite: Satellite,
                                          target: Target,
                                          tiles: List[MosaicTile],
                                          start_time: datetime,
                                          end_time: datetime,
                                          parallel: bool = True
                                          ) -> Dict[str, List[VisibilityWindow]]:
        """
        计算区域内所有瓦片的可见性（聚合计算）

        Args:
            satellite: 卫星
            target: 区域目标
            tiles: 瓦片列表
            start_time: 开始时间
            end_time: 结束时间
            parallel: 是否并行计算

        Returns:
            Dict[str, List[VisibilityWindow]]: {tile_id: [windows]}
        """
        results = {}

        if parallel and len(tiles) > 1:
            # 并行计算
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.compute_tile_visibility,
                        satellite, tile, start_time, end_time
                    ): tile.tile_id
                    for tile in tiles
                }

                for future in as_completed(futures):
                    tile_id = futures[future]
                    try:
                        windows = future.result()
                        results[tile_id] = windows
                    except Exception as e:
                        logger.error(f"Error computing visibility for tile {tile_id}: {e}")
                        results[tile_id] = []
        else:
            # 串行计算
            for tile in tiles:
                windows = self.compute_tile_visibility(
                    satellite, tile, start_time, end_time
                )
                results[tile.tile_id] = windows

        return results

    def compute_multi_satellite_visibility(self,
                                           satellites: List[Satellite],
                                           tiles: List[MosaicTile],
                                           start_time: datetime,
                                           end_time: datetime
                                           ) -> Dict[str, Dict[str, List[VisibilityWindow]]]:
        """
        计算多颗卫星对多个瓦片的可见性

        Returns:
            Dict[sat_id, Dict[tile_id, List[VisibilityWindow]]]
        """
        results = {}

        for sat in satellites:
            sat_results = self.compute_area_aggregate_visibility(
                sat, None, tiles, start_time, end_time, parallel=True
            )
            results[sat.id] = sat_results

        return results

    def find_best_observation_windows(self,
                                      satellite: Satellite,
                                      tiles: List[MosaicTile],
                                      start_time: datetime,
                                      end_time: datetime,
                                      min_quality: float = 0.5
                                      ) -> List[Tuple[MosaicTile, VisibilityWindow]]:
        """
        查找最佳观测窗口

        返回按质量排序的（瓦片，窗口）列表

        Args:
            satellite: 卫星
            tiles: 瓦片列表
            start_time: 开始时间
            end_time: 结束时间
            min_quality: 最小质量要求

        Returns:
            List[Tuple[MosaicTile, VisibilityWindow]]: 最佳窗口列表
        """
        best_windows = []

        for tile in tiles:
            windows = self.compute_tile_visibility(
                satellite, tile, start_time, end_time
            )

            # 筛选高质量窗口
            for window in windows:
                if window.quality_score >= min_quality:
                    best_windows.append((tile, window))

        # 按质量排序
        best_windows.sort(key=lambda x: x[1].quality_score, reverse=True)

        return best_windows

    def calculate_tile_visibility_score(self,
                                        tile: MosaicTile,
                                        windows: List[VisibilityWindow]) -> float:
        """
        计算瓦片的可见性得分

        基于：
        - 可见窗口数量
        - 窗口质量
        - 总可见时长

        Returns:
            可见性得分（0-1）
        """
        if not windows:
            return 0.0

        # 窗口数量得分
        count_score = min(1.0, len(windows) / 5.0)

        # 质量得分
        quality_score = sum(w.quality_score for w in windows) / len(windows)

        # 时长得分（假设最优时长为300秒）
        total_duration = sum(w.duration() for w in windows)
        duration_score = min(1.0, total_duration / 300.0)

        # 综合得分
        return (count_score * 0.3 + quality_score * 0.5 + duration_score * 0.2)

    def select_optimal_tiles_for_satellite(self,
                                          satellite: Satellite,
                                          tiles: List[MosaicTile],
                                          start_time: datetime,
                                          end_time: datetime,
                                          max_tiles: int = 10,
                                          min_visibility_score: float = 0.3
                                          ) -> List[MosaicTile]:
        """
        为卫星选择最优瓦片集合

        基于可见性评分选择最值得观测的瓦片

        Args:
            satellite: 卫星
            tiles: 候选瓦片列表
            start_time: 开始时间
            end_time: 结束时间
            max_tiles: 最大选择瓦片数
            min_visibility_score: 最小可见性得分

        Returns:
            List[MosaicTile]: 选中的瓦片列表
        """
        tile_scores = []

        for tile in tiles:
            windows = self.compute_tile_visibility(
                satellite, tile, start_time, end_time
            )
            score = self.calculate_tile_visibility_score(tile, windows)

            if score >= min_visibility_score:
                tile_scores.append((tile, score))

        # 按得分排序
        tile_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前N个
        return [tile for tile, _ in tile_scores[:max_tiles]]

    def check_tile_footprint_coverage(self,
                                      satellite: Satellite,
                                      tile: MosaicTile,
                                      window: VisibilityWindow,
                                      at_time: Optional[datetime] = None) -> float:
        """
        检查足迹对瓦片的覆盖比例

        Args:
            satellite: 卫星
            tile: 瓦片
            window: 可见窗口
            at_time: 特定时间点（默认窗口中点）

        Returns:
            覆盖比例（0-1）
        """
        if at_time is None:
            # 使用窗口中点
            mid_seconds = (window.end_time - window.start_time).total_seconds() / 2
            at_time = window.start_time + timedelta(seconds=mid_seconds)

        # 简化计算：假设足迹中心在瓦片中心则完全覆盖
        # 实际应该使用足迹几何计算与瓦片的交集
        tile_center = tile.center

        # 获取卫星位置（需要传播器）
        if self.propagator:
            try:
                sat_pos = self.propagator.get_position_at(satellite.id, at_time)
                # 计算卫星到瓦片中心的距离
                # 简化：如果仰角足够高，认为完全覆盖
                # 实际应该计算足迹多边形与瓦片多边形的交集
                return 1.0  # 简化返回
            except Exception as e:
                logger.warning(f"Failed to calculate footprint coverage for {satellite.id}: {e}")

        return 0.8  # 默认80%覆盖

    def _compute_simple_visibility(self,
                                   satellite: Satellite,
                                   target: Target,
                                   start_time: datetime,
                                   end_time: datetime,
                                   time_step: timedelta
                                   ) -> List[VisibilityWindow]:
        """
        简化的可见性计算（在没有基础计算器时使用）

        使用简化几何计算
        """
        windows = []
        # 这是一个占位实现
        # 实际应该使用轨道传播计算仰角
        # 返回空列表，表示需要完整计算器
        return windows

    def batch_compute_with_cache(self,
                                 satellites: List[Satellite],
                                 tiles: List[MosaicTile],
                                 start_time: datetime,
                                 end_time: datetime,
                                 cache: Optional[Dict] = None
                                 ) -> Dict[str, Dict[str, List[VisibilityWindow]]]:
        """
        带缓存的批量可见性计算

        Args:
            satellites: 卫星列表
            tiles: 瓦片列表
            start_time: 开始时间
            end_time: 结束时间
            cache: 可选的缓存字典

        Returns:
            可见性结果字典
        """
        if cache is None:
            cache = {}

        results = {}

        for sat in satellites:
            sat_id = sat.id
            results[sat_id] = {}

            for tile in tiles:
                tile_id = tile.tile_id
                cache_key = f"{sat_id}:{tile_id}"

                if cache_key in cache:
                    results[sat_id][tile_id] = cache[cache_key]
                else:
                    windows = self.compute_tile_visibility(
                        sat, tile, start_time, end_time
                    )
                    results[sat_id][tile_id] = windows
                    cache[cache_key] = windows

        return results


class TileVisibilityCache:
    """
    瓦片可见性缓存

    缓存瓦片的可见性窗口，避免重复计算
    """

    def __init__(self):
        self._cache: Dict[str, List[VisibilityWindow]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def get_key(self, satellite_id: str, tile_id: str) -> str:
        """生成缓存键"""
        return f"{satellite_id}:{tile_id}"

    def get(self, satellite_id: str, tile_id: str) -> Optional[List[VisibilityWindow]]:
        """获取缓存的可见性窗口"""
        key = self.get_key(satellite_id, tile_id)
        return self._cache.get(key)

    def set(self, satellite_id: str, tile_id: str,
            windows: List[VisibilityWindow],
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """设置缓存"""
        key = self.get_key(satellite_id, tile_id)
        self._cache[key] = windows
        if metadata:
            self._metadata[key] = metadata

    def invalidate(self, satellite_id: Optional[str] = None,
                   tile_id: Optional[str] = None):
        """使缓存失效"""
        if satellite_id is None and tile_id is None:
            # 清空所有缓存
            self._cache.clear()
            self._metadata.clear()
        elif tile_id is None:
            # 使特定卫星的缓存失效
            keys_to_remove = [k for k in self._cache.keys()
                            if k.startswith(f"{satellite_id}:")]
            for key in keys_to_remove:
                del self._cache[key]
                self._metadata.pop(key, None)
        else:
            # 使特定条目失效
            key = self.get_key(satellite_id, tile_id)
            self._cache.pop(key, None)
            self._metadata.pop(key, None)

    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_entries = len(self._cache)
        total_windows = sum(len(w) for w in self._cache.values())

        return {
            'total_entries': total_entries,
            'total_windows': total_windows,
            'avg_windows_per_entry': total_windows / total_entries if total_entries > 0 else 0,
        }
