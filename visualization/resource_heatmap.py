"""
资源利用率热力图可视化器

显示卫星资源使用热力图，时间-资源二维热力展示
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


class ResourceHeatmapVisualizer:
    """资源利用率热力图可视化器"""

    def __init__(self, figsize: tuple = (12, 8)):
        """
        初始化资源利用率热力图可视化器

        Args:
            figsize: 图表尺寸 (宽, 高)
        """
        self.figsize = figsize

    def plot(
        self,
        utilization: np.ndarray,
        resources: List[str],
        time_slots: List[str],
        title: str = "资源利用率热力图",
        cmap: str = "YlOrRd",
        show_values: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> plt.Figure:
        """
        绘制资源利用率热力图

        Args:
            utilization: 利用率矩阵 (资源 x 时间)，值范围0-100
            resources: 资源名称列表
            time_slots: 时间槽标签列表
            title: 图表标题
            cmap: 颜色映射
            show_values: 是否在单元格内显示数值
            vmin: 颜色映射最小值
            vmax: 颜色映射最大值

        Returns:
            matplotlib Figure对象

        Raises:
            ValueError: 当维度不匹配时
        """
        # 验证维度
        if utilization.shape != (len(resources), len(time_slots)):
            raise ValueError(
                f"维度不匹配: utilization形状{utilization.shape}，"
                f"但resources有{len(resources)}个，time_slots有{len(time_slots)}个"
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        # 处理NaN值，替换为0
        plot_data = np.nan_to_num(utilization, nan=0.0)

        # 设置颜色范围
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 100

        # 绘制热力图
        im = ax.imshow(
            plot_data,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        # 设置坐标轴
        ax.set_xticks(np.arange(len(time_slots)))
        ax.set_yticks(np.arange(len(resources)))
        ax.set_xticklabels(time_slots, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(resources, fontsize=9)

        ax.set_xlabel('时间', fontsize=11)
        ax.set_ylabel('资源', fontsize=11)
        ax.set_title(title, fontsize=14, weight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('利用率 (%)', fontsize=10)

        # 在单元格内显示数值
        if show_values:
            for i in range(len(resources)):
                for j in range(len(time_slots)):
                    value = plot_data[i, j]
                    text_color = 'white' if value > 50 else 'black'
                    ax.text(j, i, f'{value:.0f}',
                           ha='center', va='center',
                           color=text_color, fontsize=8)

        plt.tight_layout()
        return fig

    def plot_multi_resource(
        self,
        utilization_dict: dict,
        time_slots: List[str],
        title: str = "多资源利用率热力图",
        cmap: str = "YlOrRd"
    ) -> plt.Figure:
        """
        绘制多类型资源利用率热力图

        Args:
            utilization_dict: 各类资源的利用率矩阵字典
                {
                    '存储': np.array([[...], [...]]),
                    '电量': np.array([[...], [...]])
                }
            time_slots: 时间槽标签列表
            title: 图表标题
            cmap: 颜色映射

        Returns:
            matplotlib Figure对象
        """
        resource_types = list(utilization_dict.keys())
        n_types = len(resource_types)

        fig, axes = plt.subplots(
            n_types, 1,
            figsize=(self.figsize[0], self.figsize[1] * n_types / 2),
            sharex=True
        )

        if n_types == 1:
            axes = [axes]

        for idx, (res_type, data) in enumerate(utilization_dict.items()):
            ax = axes[idx]

            # 处理NaN值
            plot_data = np.nan_to_num(data, nan=0.0)

            im = ax.imshow(
                plot_data,
                aspect='auto',
                cmap=cmap,
                vmin=0,
                vmax=100,
                interpolation='nearest'
            )

            ax.set_ylabel(res_type, fontsize=10)
            ax.set_yticks([])

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('%', fontsize=8)

        axes[-1].set_xticks(np.arange(len(time_slots)))
        axes[-1].set_xticklabels(time_slots, rotation=45, ha='right', fontsize=9)
        axes[-1].set_xlabel('时间', fontsize=11)

        fig.suptitle(title, fontsize=14, weight='bold')
        plt.tight_layout()

        return fig

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
