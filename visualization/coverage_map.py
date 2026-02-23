"""
覆盖分析图可视化器

显示区域目标覆盖次数，热力图形式展示覆盖密度
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional


class CoverageMap:
    """覆盖分析图可视化器"""

    def __init__(self, figsize: tuple = (12, 10)):
        """
        初始化覆盖分析图可视化器

        Args:
            figsize: 图表尺寸 (宽, 高)
        """
        self.figsize = figsize

    def plot(
        self,
        coverage_count: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        targets: Optional[List[Dict[str, Any]]] = None,
        title: str = "区域覆盖分析图",
        cmap: str = "YlOrRd",
        show_contours: bool = False,
        show_colorbar: bool = True
    ) -> plt.Figure:
        """
        绘制覆盖分析图

        Args:
            coverage_count: 覆盖次数矩阵 (纬度 x 经度)
            lon_grid: 经度网格
            lat_grid: 纬度网格
            targets: 目标位置列表 [{'lon': x, 'lat': y, 'id': 'T001'}, ...]
            title: 图表标题
            cmap: 颜色映射
            show_contours: 是否显示等高线
            show_colorbar: 是否显示颜色条

        Returns:
            matplotlib Figure对象

        Raises:
            ValueError: 当网格维度不匹配时
        """
        # 验证维度
        if coverage_count.shape != (len(lat_grid), len(lon_grid)):
            raise ValueError(
                f"网格维度不匹配: coverage_count形状{coverage_count.shape}，"
                f"但lat_grid有{len(lat_grid)}个，lon_grid有{len(lon_grid)}个"
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        # 确保覆盖次数非负
        plot_data = np.maximum(coverage_count, 0)

        # 创建网格
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # 绘制热力图
        im = ax.pcolormesh(
            lon_mesh, lat_mesh, plot_data,
            cmap=cmap,
            shading='auto',
            vmin=0
        )

        # 添加等高线
        if show_contours and plot_data.max() > plot_data.min():
            levels = np.linspace(plot_data.min(), plot_data.max(), 5)
            ax.contour(lon_mesh, lat_mesh, plot_data, levels=levels,
                      colors='black', linewidths=0.5, alpha=0.5)

        # 绘制目标位置
        if targets:
            for target in targets:
                lon = target.get('lon')
                lat = target.get('lat')
                target_id = target.get('id', '')

                if lon is not None and lat is not None:
                    ax.plot(lon, lat, 'b*', markersize=12,
                           markeredgecolor='white', markeredgewidth=1.5)

                    # 添加目标标签
                    if target_id:
                        ax.annotate(
                            target_id,
                            (lon, lat),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            color='blue',
                            bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='white', alpha=0.7)
                        )

        # 设置坐标轴
        ax.set_xlabel('经度 (°)', fontsize=11)
        ax.set_ylabel('纬度 (°)', fontsize=11)
        ax.set_title(title, fontsize=14, weight='bold')

        ax.set_xlim(lon_grid.min(), lon_grid.max())
        ax.set_ylim(lat_grid.min(), lat_grid.max())

        # 添加颜色条
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('覆盖次数', fontsize=10)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_coverage_statistics(
        self,
        coverage_count: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        title: str = "覆盖统计分析"
    ) -> plt.Figure:
        """
        绘制覆盖统计分析图（包含热力图和直方图）

        Args:
            coverage_count: 覆盖次数矩阵
            lon_grid: 经度网格
            lat_grid: 纬度网格
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.2))

        # 创建子图布局
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        ax_map = fig.add_subplot(gs[0, :])
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_stats = fig.add_subplot(gs[1, 1])

        # 确保覆盖次数非负
        plot_data = np.maximum(coverage_count, 0)

        # 1. 覆盖热力图
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        im = ax_map.pcolormesh(
            lon_mesh, lat_mesh, plot_data,
            cmap='YlOrRd',
            shading='auto',
            vmin=0
        )
        ax_map.set_xlabel('经度 (°)', fontsize=10)
        ax_map.set_ylabel('纬度 (°)', fontsize=10)
        ax_map.set_title('覆盖热力图', fontsize=12, weight='bold')
        plt.colorbar(im, ax=ax_map, label='覆盖次数')

        # 2. 覆盖次数分布直方图
        flat_data = plot_data.flatten()
        ax_hist.hist(flat_data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax_hist.set_xlabel('覆盖次数', fontsize=10)
        ax_hist.set_ylabel('网格数量', fontsize=10)
        ax_hist.set_title('覆盖分布', fontsize=11)
        ax_hist.grid(True, linestyle='--', alpha=0.3)

        # 3. 统计信息
        stats_text = (
            f"覆盖统计:\n"
            f"  平均覆盖: {flat_data.mean():.2f}\n"
            f"  最大覆盖: {flat_data.max():.0f}\n"
            f"  最小覆盖: {flat_data.min():.0f}\n"
            f"  标准差: {flat_data.std():.2f}\n"
            f"  零覆盖网格: {(flat_data == 0).sum()} "
            f"({(flat_data == 0).mean()*100:.1f}%)"
        )
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='center',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        ax_stats.set_title('统计信息', fontsize=11)

        fig.suptitle(title, fontsize=14, weight='bold')

        return fig

    def plot_coverage_gap_analysis(
        self,
        coverage_count: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        threshold: int = 1,
        title: str = "覆盖缺口分析"
    ) -> plt.Figure:
        """
        绘制覆盖缺口分析图

        Args:
            coverage_count: 覆盖次数矩阵
            lon_grid: 经度网格
            lat_grid: 纬度网格
            threshold: 覆盖缺口阈值（低于此值视为缺口）
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 识别覆盖缺口
        gap_mask = coverage_count < threshold

        # 创建网格
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # 绘制背景（有覆盖区域）
        background = np.where(coverage_count >= threshold, coverage_count, np.nan)
        im = ax.pcolormesh(
            lon_mesh, lat_mesh, background,
            cmap='YlGn',
            shading='auto',
            vmin=0
        )

        # 绘制缺口区域
        gaps = np.where(gap_mask, 1, np.nan)
        ax.pcolormesh(
            lon_mesh, lat_mesh, gaps,
            cmap='Reds',
            shading='auto',
            alpha=0.7
        )

        # 设置坐标轴
        ax.set_xlabel('经度 (°)', fontsize=11)
        ax.set_ylabel('纬度 (°)', fontsize=11)
        ax.set_title(f"{title} (阈值={threshold})", fontsize=14, weight='bold')

        ax.set_xlim(lon_grid.min(), lon_grid.max())
        ax.set_ylim(lat_grid.min(), lat_grid.max())

        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='满足覆盖要求'),
            Patch(facecolor='red', alpha=0.7, label='覆盖缺口')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.grid(True, linestyle='--', alpha=0.3)

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
