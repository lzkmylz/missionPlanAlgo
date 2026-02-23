"""
甘特图生成器

生成调度结果的甘特图
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import List, Any
from datetime import datetime


class GanttChart:
    """甘特图生成器"""

    COLORS = {
        'imaging_optical': '#FF6B6B',
        'imaging_sar': '#4ECDC4',
        'downlink': '#45B7D1',
        'idle': '#F7F7F7',
    }

    def __init__(self, figsize: tuple = (16, 10)):
        self.figsize = figsize

    def plot(
        self,
        scheduled_tasks: List[Any],
        satellites: List[Any],
        ground_stations: List[Any],
        start_time: datetime,
        end_time: datetime,
        title: str = "卫星任务调度甘特图"
    ) -> plt.Figure:
        """绘制甘特图"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # 收集资源
        resources = [sat.id for sat in satellites]
        y_positions = {res: i for i, res in enumerate(resources)}

        # 绘制任务条
        duration = (end_time - start_time).total_seconds()

        for task in scheduled_tasks:
            sat_id = task.satellite_id
            if sat_id not in y_positions:
                continue

            task_start = (task.imaging_start - start_time).total_seconds()
            task_duration = (task.imaging_end - task.imaging_start).total_seconds()
            y = y_positions[sat_id]

            ax.barh(
                y,
                task_duration / 3600,
                left=task_start / 3600,
                height=0.6,
                color=self.COLORS['imaging_optical'],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )

        # 设置坐标轴
        ax.set_yticks(range(len(resources)))
        ax.set_yticklabels(resources)
        ax.set_ylabel('卫星')
        ax.set_xlabel('时间 (小时)')
        ax.set_xlim(0, duration / 3600)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        return fig

    def save(self, fig: plt.Figure, filepath: str, dpi: int = 150):
        """保存图表"""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"甘特图已保存至: {filepath}")
