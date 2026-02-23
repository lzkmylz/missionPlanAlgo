"""
时间窗口缓存管理器

性能优化关键组件：预计算所有可见性窗口并在内存中缓存，
使调度算法迭代时只做O(1)的内存访问，而非实时计算。
"""

import bisect
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
from .base import VisibilityWindow, VisibilityCalculator


class VisibilityWindowCache:
    """
    可见性窗口内存缓存管理器

    核心特性：
    1. O(1)时间复杂度查询
    2. 预计算阶段只执行一次
    3. 支持多维度索引（卫星、目标、时间）
    4. 内存占用优化
    """

    def __init__(self):
        # 主索引: (sat_id, target_id) -> List[VisibilityWindow]
        self._windows: Dict[Tuple[str, str], List[VisibilityWindow]] = {}

        # 辅助索引: sat_id -> Set[target_id]
        self._sat_to_targets: Dict[str, Set[str]] = {}

        # 辅助索引: target_id -> Set[sat_id]
        self._target_to_sats: Dict[str, Set[str]] = {}

        # 时间索引: (sat_id, target_id) -> List[start_time]（用于二分查找）
        self._time_index: Dict[Tuple[str, str], List[datetime]] = {}

    def precompute_all_windows(
        self,
        satellites,
        targets,
        ground_stations,
        start_time: datetime,
        end_time: datetime,
        calculator: VisibilityCalculator
    ) -> None:
        """
        预计算所有可见性窗口（实验初始化阶段调用一次）
        这是唯一会调用SGP4/Orekit的地方！

        Args:
            satellites: 卫星列表
            targets: 目标列表
            ground_stations: 地面站列表
            start_time: 开始时间
            end_time: 结束时间
            calculator: 可见性计算器
        """
        # 尝试导入tqdm，如果不存在则使用简单循环
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # 计算卫星-目标可见窗口
        total_pairs = len(satellites) * len(targets)

        def compute_sat_target_windows():
            for sat in satellites:
                for target in targets:
                    windows = calculator.compute_satellite_target_windows(
                        sat, target, start_time, end_time
                    )
                    if windows:
                        key = (sat.id, target.id)
                        self._windows[key] = sorted(windows)
                        self._time_index[key] = [w.start_time for w in windows]

                        # 更新辅助索引
                        if sat.id not in self._sat_to_targets:
                            self._sat_to_targets[sat.id] = set()
                        self._sat_to_targets[sat.id].add(target.id)

                        if target.id not in self._target_to_sats:
                            self._target_to_sats[target.id] = set()
                        self._target_to_sats[target.id].add(sat.id)

                    yield

        if has_tqdm:
            with tqdm(total=total_pairs, desc="预计算卫星-目标可见窗口") as pbar:
                for _ in compute_sat_target_windows():
                    pbar.update(1)
        else:
            for _ in compute_sat_target_windows():
                pass

        # 计算卫星-地面站可见窗口
        total_gs_pairs = len(satellites) * len(ground_stations)

        def compute_sat_gs_windows():
            for sat in satellites:
                for gs in ground_stations:
                    windows = calculator.compute_satellite_ground_station_windows(
                        sat, gs, start_time, end_time
                    )
                    if windows:
                        key = (sat.id, f"GS:{gs.id}")
                        self._windows[key] = sorted(windows)
                        self._time_index[key] = [w.start_time for w in windows]
                    yield

        if has_tqdm:
            with tqdm(total=total_gs_pairs, desc="预计算卫星-地面站可见窗口") as pbar:
                for _ in compute_sat_gs_windows():
                    pbar.update(1)
        else:
            for _ in compute_sat_gs_windows():
                pass

    def get_windows(self, satellite_id: str, target_id: str) -> List[VisibilityWindow]:
        """
        获取指定卫星-目标对的所有可见窗口 - O(1)复杂度

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        return self._windows.get((satellite_id, target_id), [])

    def get_windows_in_range(
        self,
        satellite_id: str,
        target_id: str,
        start: datetime,
        end: datetime
    ) -> List[VisibilityWindow]:
        """
        获取指定时间范围内的可见窗口
        时间复杂度: O(log n + k)

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            start: 开始时间
            end: 结束时间

        Returns:
            List[VisibilityWindow]: 时间范围内的可见窗口
        """
        all_windows = self.get_windows(satellite_id, target_id)
        if not all_windows:
            return []

        time_list = self._time_index.get((satellite_id, target_id), [])

        # 二分查找起始位置
        left = bisect.bisect_left(time_list, start)
        right = bisect.bisect_right(time_list, end)

        # 过滤出时间范围内的窗口
        result = []
        for i in range(left, min(right, len(all_windows))):
            window = all_windows[i]
            if window.start_time < end and window.end_time > start:
                result.append(window)

        return result

    def get_visible_satellites(self, target_id: str) -> Set[str]:
        """
        获取可以看到指定目标的所有卫星

        Args:
            target_id: 目标ID

        Returns:
            Set[str]: 卫星ID集合
        """
        return self._target_to_sats.get(target_id, set())

    def get_visible_targets(self, satellite_id: str) -> Set[str]:
        """
        获取指定卫星可以看到的目标

        Args:
            satellite_id: 卫星ID

        Returns:
            Set[str]: 目标ID集合
        """
        return self._sat_to_targets.get(satellite_id, set())

    def get_ground_station_windows(self, satellite_id: str, ground_station_id: str) -> List[VisibilityWindow]:
        """
        获取卫星-地面站可见窗口

        Args:
            satellite_id: 卫星ID
            ground_station_id: 地面站ID

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        return self._windows.get((satellite_id, f"GS:{ground_station_id}"), [])

    def has_windows(self, satellite_id: str, target_id: str) -> bool:
        """
        检查是否存在可见窗口

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID

        Returns:
            bool: 是否存在可见窗口
        """
        return (satellite_id, target_id) in self._windows

    def get_statistics(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            dict: 统计信息
        """
        total_windows = sum(len(windows) for windows in self._windows.values())
        total_pairs = len(self._windows)
        sat_target_pairs = sum(len(targets) for targets in self._sat_to_targets.values())

        return {
            'total_window_pairs': total_pairs,
            'total_windows': total_windows,
            'avg_windows_per_pair': total_windows / total_pairs if total_pairs > 0 else 0,
            'satellite_count': len(self._sat_to_targets),
            'target_count': len(self._target_to_sats),
            'sat_target_pairs': sat_target_pairs,
        }

    def clear(self) -> None:
        """清空缓存"""
        self._windows.clear()
        self._sat_to_targets.clear()
        self._target_to_sats.clear()
        self._time_index.clear()

    def size_bytes(self) -> int:
        """
        估算内存占用（字节）

        Returns:
            int: 估算的内存占用
        """
        import sys

        # 粗略估算
        window_size = sys.getsizeof(VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=datetime.now(),
            end_time=datetime.now()
        ))

        total_windows = sum(len(windows) for windows in self._windows.values())
        return total_windows * window_size + len(self._windows) * 100  # 额外开销
