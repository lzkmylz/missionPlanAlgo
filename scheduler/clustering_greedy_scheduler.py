"""
ClusteringGreedyScheduler - 支持目标聚类的贪心调度器

向后兼容的包装类，现在 GreedyScheduler 已内置聚类支持。

推荐使用方式：
    # 使用 GreedyScheduler 并启用聚类
    from scheduler.greedy.greedy_scheduler import GreedyScheduler

    scheduler = GreedyScheduler(config={
        'enable_clustering': True,
        'cluster_radius_km': 10.0,  # 10公里范围内的目标聚类
        'min_cluster_size': 2,
    })

本类保留作为向后兼容的别名：
    from scheduler.clustering_greedy_scheduler import ClusteringGreedyScheduler

    # 等效于 GreedyScheduler(enable_clustering=True)
    scheduler = ClusteringGreedyScheduler(config={
        'cluster_radius_km': 10.0,
        'min_cluster_size': 2,
    })
"""

from typing import Dict, Any
from scheduler.greedy.greedy_scheduler import GreedyScheduler


class ClusteringGreedyScheduler(GreedyScheduler):
    """
    支持目标聚类的贪心调度器（向后兼容）

    此类现在继承自 GreedyScheduler，并自动启用聚类功能。
    所有聚类配置参数与 GreedyScheduler 相同。

    Attributes:
        enable_clustering: 始终为 True
        cluster_radius_km: 聚类半径（公里）
        min_cluster_size: 最小聚类大小
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 ClusteringGreedyScheduler

        Args:
            config: 配置字典
                - cluster_radius_km: 聚类半径，公里 (默认 10.0)
                - min_cluster_size: 最小聚类大小 (默认 2)
                - 其他 GreedyScheduler 支持的参数
        """
        config = config or {}
        # 强制启用聚类
        config['enable_clustering'] = True
        super().__init__(config)
        self.name = "ClusteringGreedy"


# 向后兼容的导出
__all__ = ['ClusteringGreedyScheduler']
