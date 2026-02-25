"""
可见性计算基类

提供卫星-目标和卫星-地面站可见性计算的抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import math


@dataclass(frozen=True, slots=True)
class VisibilityWindow:
    """
    可见时间窗口

    Attributes:
        satellite_id: 卫星ID
        target_id: 目标ID（或地面站ID，格式为"GS:{gs_id}"）
        start_time: 窗口开始时间
        end_time: 窗口结束时间
        max_elevation: 最大仰角（度）
        quality_score: 窗口质量评分（0-1）
    """
    satellite_id: str
    target_id: str
    start_time: datetime
    end_time: datetime
    max_elevation: float = 90.0
    quality_score: float = 1.0

    def duration(self) -> float:
        """窗口持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()

    def __lt__(self, other):
        """用于排序"""
        return self.start_time < other.start_time


class VisibilityCalculator(ABC):
    """
    可见性计算器基类

    抽象基类，定义可见性计算的接口
    """

    def __init__(self, min_elevation: float = 5.0):
        """
        初始化

        Args:
            min_elevation: 最小仰角（度）
        """
        self.min_elevation = min_elevation
        self.EARTH_RADIUS = 6371000.0  # 米

    @abstractmethod
    def compute_satellite_target_windows(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-目标可见窗口

        Args:
            satellite: 卫星模型
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        pass

    @abstractmethod
    def compute_satellite_ground_station_windows(
        self,
        satellite,
        ground_station,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-地面站可见窗口

        Args:
            satellite: 卫星模型
            ground_station: 地面站模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        pass

    def _calculate_elevation(
        self,
        sat_pos: Tuple[float, float, float],
        ground_pos: Tuple[float, float, float]
    ) -> float:
        """
        计算仰角

        Args:
            sat_pos: 卫星位置 (x, y, z) in meters
            ground_pos: 地面位置 (x, y, z) in meters

        Returns:
            仰角（度），如果卫星在地球背面返回负值
        """
        # 计算地面站到卫星的向量
        dx = sat_pos[0] - ground_pos[0]
        dy = sat_pos[1] - ground_pos[1]
        dz = sat_pos[2] - ground_pos[2]

        # 地面站位置向量
        gx, gy, gz = ground_pos

        # 首先检查卫星是否被地球遮挡
        # 计算地面站-卫星连线到地心的最短距离
        dot_t_d = gx*dx + gy*dy + gz*dz
        d_squared = dx*dx + dy*dy + dz*dz

        if d_squared > 1e-6:
            t = -dot_t_d / d_squared

            # 如果最近点在线段内，检查是否穿过地球
            if 0 < t < 1:
                closest_x = gx + t * dx
                closest_y = gy + t * dy
                closest_z = gz + t * dz
                closest_dist = math.sqrt(closest_x**2 + closest_y**2 + closest_z**2)

                if closest_dist < self.EARTH_RADIUS * 0.999:
                    # 连线穿过地球内部，卫星不可见
                    return -90.0

        # 地面站到天顶的单位向量（地心反方向）
        g_norm = math.sqrt(gx**2 + gy**2 + gz**2)
        up_x = -gx / g_norm
        up_y = -gy / g_norm
        up_z = -gz / g_norm

        # 归一化卫星方向向量
        d_norm = math.sqrt(dx**2 + dy**2 + dz**2)
        dx_norm = dx / d_norm
        dy_norm = dy / d_norm
        dz_norm = dz / d_norm

        # 计算与天顶方向的夹角余弦
        cos_zenith = dx_norm * up_x + dy_norm * up_y + dz_norm * up_z

        # 仰角 = 90° - 天顶角
        elevation = math.degrees(math.asin(max(-1, min(1, cos_zenith))))

        return elevation

    def _find_windows_from_elevations(
        self,
        satellite_id: str,
        target_id: str,
        elevation_data: List[Tuple[datetime, float]],
        min_elevation: float
    ) -> List[VisibilityWindow]:
        """
        从仰角数据中提取可见窗口

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            elevation_data: [(timestamp, elevation), ...]
            min_elevation: 最小仰角

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        windows = []
        in_window = False
        window_start = None
        max_elev_in_window = 0.0

        for timestamp, elevation in elevation_data:
            if elevation >= min_elevation:
                if not in_window:
                    # 窗口开始
                    in_window = True
                    window_start = timestamp
                    max_elev_in_window = elevation
                else:
                    # 窗口内，更新最大仰角
                    max_elev_in_window = max(max_elev_in_window, elevation)
            else:
                if in_window:
                    # 窗口结束
                    in_window = False
                    # 计算质量评分（基于最大仰角）
                    quality = min(1.0, max_elev_in_window / 90.0)

                    windows.append(VisibilityWindow(
                        satellite_id=satellite_id,
                        target_id=target_id,
                        start_time=window_start,
                        end_time=timestamp,
                        max_elevation=max_elev_in_window,
                        quality_score=quality
                    ))
                    max_elev_in_window = 0.0

        # 处理最后一个窗口（如果还在窗口内）
        if in_window and elevation_data:
            quality = min(1.0, max_elev_in_window / 90.0)
            windows.append(VisibilityWindow(
                satellite_id=satellite_id,
                target_id=target_id,
                start_time=window_start,
                end_time=elevation_data[-1][0],
                max_elevation=max_elev_in_window,
                quality_score=quality
            ))

        return windows
