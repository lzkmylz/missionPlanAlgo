"""
星下点轨迹可视化器

显示卫星轨道地面轨迹，标记观测时刻位置
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class GroundTrackVisualizer:
    """星下点轨迹可视化器"""

    # 预定义颜色方案
    COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
    ]

    def __init__(self, figsize: tuple = (14, 8)):
        """
        初始化星下点轨迹可视化器

        Args:
            figsize: 图表尺寸 (宽, 高)
        """
        self.figsize = figsize

    def plot(
        self,
        ground_tracks: Dict[str, Dict[str, List]],
        observation_points: Optional[Dict[str, List[Dict]]] = None,
        title: str = "卫星星下点轨迹图",
        show_grid: bool = True
    ) -> plt.Figure:
        """
        绘制星下点轨迹图

        Args:
            ground_tracks: 星下点轨迹数据
                {
                    'SAT-01': {
                        'lon': [116.0, 117.0, ...],
                        'lat': [39.0, 40.0, ...],
                        'times': [datetime, ...]
                    }
                }
            observation_points: 观测点位置
                {
                    'SAT-01': [
                        {'lon': 117.5, 'lat': 39.5, 'time': datetime},
                        ...
                    ]
                }
            title: 图表标题
            show_grid: 是否显示网格

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        observation_points = observation_points or {}

        # 绘制每个卫星的轨迹
        for idx, (sat_id, track_data) in enumerate(ground_tracks.items()):
            color = self.COLORS[idx % len(self.COLORS)]

            lons = track_data.get('lon', [])
            lats = track_data.get('lat', [])
            times = track_data.get('times', [])

            # 过滤None值
            valid_points = [
                (lon, lat, time) for lon, lat, time in zip(lons, lats, times)
                if lon is not None and lat is not None
            ]

            if not valid_points:
                continue

            valid_lons = [p[0] for p in valid_points]
            valid_lats = [p[1] for p in valid_points]

            # 处理跨越日界线的情况
            valid_lons = self._unwrap_longitude(valid_lons)

            # 绘制轨迹线
            ax.plot(valid_lons, valid_lats, '-', color=color,
                   linewidth=1.5, alpha=0.7, label=f'{sat_id} 轨迹')

            # 绘制起点和终点
            ax.plot(valid_lons[0], valid_lats[0], 'o', color=color,
                   markersize=6, markerfacecolor='white', markeredgewidth=2)
            ax.plot(valid_lons[-1], valid_lats[-1], 's', color=color,
                   markersize=6, markerfacecolor='white', markeredgewidth=2)

            # 绘制观测点
            if sat_id in observation_points:
                obs_list = observation_points[sat_id]
                for obs in obs_list:
                    obs_lon = obs.get('lon')
                    obs_lat = obs.get('lat')
                    if obs_lon is not None and obs_lat is not None:
                        ax.plot(obs_lon, obs_lat, '*', color=color,
                               markersize=12, markeredgecolor='black',
                               markeredgewidth=1, zorder=5)

        # 设置坐标轴
        ax.set_xlabel('经度 (°)', fontsize=11)
        ax.set_ylabel('纬度 (°)', fontsize=11)
        ax.set_title(title, fontsize=14, weight='bold')

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5)

        # 设置坐标范围（如果数据不为空）
        if ground_tracks:
            all_lons = []
            all_lats = []
            for track_data in ground_tracks.values():
                lons = [lon for lon in track_data.get('lon', []) if lon is not None]
                lats = [lat for lat in track_data.get('lat', []) if lat is not None]
                all_lons.extend(lons)
                all_lats.extend(lats)

            if all_lons and all_lats:
                lon_margin = (max(all_lons) - min(all_lons)) * 0.1
                lat_margin = (max(all_lats) - min(all_lats)) * 0.1
                ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
                ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

        # 添加图例
        if ground_tracks:
            ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig

    def _unwrap_longitude(self, lons: List[float]) -> List[float]:
        """
        处理跨越日界线的经度数据

        当轨迹跨越180°经线时，避免绘制不必要的横贯线

        Args:
            lons: 经度列表

        Returns:
            处理后的经度列表
        """
        if not lons:
            return lons

        result = [lons[0]]
        for i in range(1, len(lons)):
            diff = lons[i] - lons[i-1]
            # 如果差值大于180度，说明跨越了日界线
            if diff > 180:
                result.append(lons[i] - 360)
            elif diff < -180:
                result.append(lons[i] + 360)
            else:
                result.append(lons[i])
        return result

    def save(self, fig: plt.Figure, filepath: str, dpi: int = 150):
        """
        保存图表

        Args:
            fig: matplotlib Figure对象
            filepath: 保存路径
            dpi: 分辨率
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


class GroundTrackPlotter:
    """星下点轨迹图绘制器（兼容旧接口）"""

    def __init__(self, figsize: tuple = (14, 8)):
        self.visualizer = GroundTrackVisualizer(figsize=figsize)

    def plot(self, ground_tracks: Dict, observation_points: Dict = None,
             title: str = "卫星星下点轨迹图") -> plt.Figure:
        """绘制星下点轨迹图（兼容接口）"""
        return self.visualizer.plot(ground_tracks, observation_points, title)

    def save(self, fig: plt.Figure, filepath: str, dpi: int = 150):
        """保存图表"""
        self.visualizer.save(fig, filepath, dpi)
