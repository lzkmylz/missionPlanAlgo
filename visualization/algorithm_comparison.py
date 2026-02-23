"""
多算法对比图表可视化器

算法性能对比（箱线图）、收敛曲线展示
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any


class AlgorithmComparisonVisualizer:
    """多算法对比图表可视化器"""

    # 预定义颜色方案
    COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
    ]

    def __init__(self, figsize: tuple = (14, 10)):
        """
        初始化算法对比可视化器

        Args:
            figsize: 图表尺寸 (宽, 高)
        """
        self.figsize = figsize

    def plot_box_comparison(
        self,
        results: Dict[str, List[float]],
        metric_name: str = "性能指标",
        title: str = "算法性能对比",
        show_mean: bool = True
    ) -> plt.Figure:
        """
        绘制算法性能箱线图对比

        Args:
            results: 算法结果字典
                {
                    'Greedy': [85.2, 86.1, 84.9, ...],
                    'GA': [92.3, 93.1, 91.8, ...],
                    ...
                }
            metric_name: 指标名称（Y轴标签）
            title: 图表标题
            show_mean: 是否显示均值标记

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not results:
            ax.set_title(title, fontsize=14, weight='bold')
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            return fig

        algorithms = list(results.keys())
        data = [results[algo] for algo in algorithms]

        # 创建箱线图
        bp = ax.boxplot(
            data,
            labels=algorithms,
            patch_artist=True,
            showmeans=show_mean,
            meanline=show_mean,
            notch=True
        )

        # 设置箱体颜色
        for patch, color in zip(bp['boxes'], self.COLORS[:len(algorithms)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 设置均值线颜色
        if show_mean:
            for mean in bp['means']:
                mean.set_color('red')
                mean.set_linewidth(2)

        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xlabel('算法', fontsize=11)
        ax.set_title(title, fontsize=14, weight='bold')

        # 添加网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def plot_convergence_curves(
        self,
        convergence_data: Dict[str, List[float]],
        title: str = "算法收敛曲线",
        xlabel: str = "迭代次数",
        ylabel: str = "目标函数值"
    ) -> plt.Figure:
        """
        绘制算法收敛曲线

        Args:
            convergence_data: 收敛曲线数据
                {
                    'GA': [75.0, 82.0, 87.0, ...],
                    'SA': [80.0, 85.0, 88.0, ...],
                    ...
                }
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not convergence_data:
            ax.set_title(title, fontsize=14, weight='bold')
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            return fig

        for idx, (algo, values) in enumerate(convergence_data.items()):
            color = self.COLORS[idx % len(self.COLORS)]
            iterations = list(range(1, len(values) + 1))

            ax.plot(iterations, values, '-', color=color,
                   linewidth=2, label=algo, marker='o', markersize=4)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=14, weight='bold')

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_radar_chart(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "算法多维度性能对比"
    ) -> plt.Figure:
        """
        绘制雷达图展示多维度性能指标

        Args:
            metrics: 多维度性能指标
                {
                    'Greedy': {'完成率': 85, '速度': 95, '稳定性': 90},
                    'GA': {'完成率': 93, '速度': 60, '稳定性': 85},
                    ...
                }
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, polar=True)

        if not metrics:
            ax.set_title(title, fontsize=14, weight='bold', pad=20)
            return fig

        # 获取指标维度
        first_algo = list(metrics.keys())[0]
        dimensions = list(metrics[first_algo].keys())
        num_dims = len(dimensions)

        if num_dims == 0:
            ax.set_title(title, fontsize=14, weight='bold', pad=20)
            return fig

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 绘制每个算法
        for idx, (algo, values) in enumerate(metrics.items()):
            color = self.COLORS[idx % len(self.COLORS)]

            # 获取该算法在各维度的值
            algo_values = [values.get(dim, 0) for dim in dimensions]
            algo_values += algo_values[:1]  # 闭合

            ax.plot(angles, algo_values, 'o-', color=color,
                   linewidth=2, label=algo)
            ax.fill(angles, algo_values, alpha=0.15, color=color)

        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=10)

        # 设置径向范围
        ax.set_ylim(0, 100)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        ax.set_title(title, fontsize=14, weight='bold', pad=20)

        plt.tight_layout()
        return fig

    def plot_pareto_front(
        self,
        pareto_data: Dict[str, List[List[float]]],
        objective_names: List[str] = None,
        title: str = "Pareto前沿"
    ) -> plt.Figure:
        """
        绘制多目标优化的Pareto前沿

        Args:
            pareto_data: Pareto前沿数据
                {
                    'GA': [[obj1, obj2], [obj1, obj2], ...],
                    'NSGA-II': [[obj1, obj2], ...],
                    ...
                }
            objective_names: 目标名称列表 [obj1_name, obj2_name]
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        if objective_names is None:
            objective_names = ['目标1', '目标2']

        fig, ax = plt.subplots(figsize=self.figsize)

        if not pareto_data:
            ax.set_title(title, fontsize=14, weight='bold')
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            return fig

        for idx, (algo, points) in enumerate(pareto_data.items()):
            color = self.COLORS[idx % len(self.COLORS)]

            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            ax.scatter(xs, ys, c=color, s=80, alpha=0.7, label=algo, edgecolors='black')

            # 绘制Pareto前沿线（按第一个目标排序）
            sorted_points = sorted(points, key=lambda x: x[0])
            sorted_xs = [p[0] for p in sorted_points]
            sorted_ys = [p[1] for p in sorted_points]
            ax.plot(sorted_xs, sorted_ys, '--', color=color, alpha=0.5, linewidth=1.5)

        ax.set_xlabel(objective_names[0], fontsize=11)
        ax.set_ylabel(objective_names[1], fontsize=11)
        ax.set_title(title, fontsize=14, weight='bold')

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

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
