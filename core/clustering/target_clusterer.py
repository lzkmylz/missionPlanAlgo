"""
目标聚类器 - 基于DBSCAN的空间聚类

功能:
- 使用DBSCAN算法对地理目标进行空间聚类
- 基于卫星幅宽计算聚类半径
- 计算聚类质心、边界框和总优先级
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import uuid

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster

from core.models.target import Target


@dataclass
class TargetCluster:
    """
    目标聚类数据类

    Attributes:
        cluster_id: 聚类唯一标识
        targets: 聚类中的目标列表
        centroid: 质心坐标 (经度, 纬度)
        total_priority: 聚类中所有目标的总优先级
        bounding_box: 边界框 (min_lon, max_lon, min_lat, max_lat)
    """
    cluster_id: str
    targets: List[Target]
    centroid: Tuple[float, float]
    total_priority: int
    bounding_box: Tuple[float, float, float, float]


class TargetClusterer:
    """
    目标聚类器

    使用层次聚类算法对地理目标进行空间聚类,基于Haversine距离计算。
    聚类半径由卫星幅宽决定。

    Attributes:
        swath_width_km: 卫星幅宽(公里),作为聚类距离阈值
        min_cluster_size: 最小聚类大小
    """

    def __init__(self, swath_width_km: float = 10.0, min_cluster_size: int = 2):
        """
        初始化聚类器

        Args:
            swath_width_km: 卫星幅宽(公里),用于确定聚类半径
            min_cluster_size: 形成聚类所需的最小目标数
        """
        self.swath_width_km = swath_width_km
        self.min_cluster_size = min_cluster_size

    def cluster_targets(self, targets: List[Target]) -> List[TargetCluster]:
        """
        对目标列表进行空间聚类

        Args:
            targets: 目标列表

        Returns:
            聚类列表,每个聚类包含多个目标
        """
        if not targets:
            return []

        if len(targets) < self.min_cluster_size:
            return []

        # 提取目标坐标
        coordinates = np.array([
            [t.longitude, t.latitude] for t in targets
        ])

        # 计算Haversine距离矩阵
        distance_matrix = self._compute_haversine_distance_matrix(coordinates)

        # 使用层次聚类
        clusters = self._hierarchical_cluster(targets, coordinates, distance_matrix)

        return clusters

    def _compute_haversine_distance_matrix(
        self, coordinates: np.ndarray
    ) -> np.ndarray:
        """
        计算Haversine距离矩阵

        Args:
            coordinates: 坐标数组,形状为 (n, 2),每行为 [lon, lat]

        Returns:
            距离矩阵(公里),形状为 (n, n)
        """
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    lon1, lat1 = coordinates[i]
                    lon2, lat2 = coordinates[j]
                    distance_matrix[i, j] = self._haversine_distance_km(
                        lon1, lat1, lon2, lat2
                    )

        return distance_matrix

    def _haversine_distance_km(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """
        计算两点间的Haversine距离(公里)

        Args:
            lon1: 点1经度
            lat1: 点1纬度
            lon2: 点2经度
            lat2: 点2纬度

        Returns:
            两点间距离(公里)
        """
        R = 6371.0  # 地球半径(公里)

        lon1_rad = math.radians(lon1)
        lat1_rad = math.radians(lat1)
        lon2_rad = math.radians(lon2)
        lat2_rad = math.radians(lat2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _hierarchical_cluster(
        self, targets: List[Target], coordinates: np.ndarray,
        distance_matrix: np.ndarray
    ) -> List[TargetCluster]:
        """
        使用层次聚类算法进行聚类

        使用single linkage方法,将距离小于swath_width_km的目标聚为一类。

        Args:
            targets: 目标列表
            coordinates: 坐标数组
            distance_matrix: 距离矩阵

        Returns:
            聚类列表
        """
        n = len(targets)

        # 将距离矩阵转换为condensed形式(上三角)
        condensed_dist = squareform(distance_matrix)

        # 使用single linkage进行层次聚类
        linkage_matrix = linkage(condensed_dist, method='single')

        # 根据距离阈值形成聚类
        # 阈值设为swath_width_km,确保距离小于阈值的目标在同一聚类
        labels = fcluster(linkage_matrix, t=self.swath_width_km, criterion='distance')

        # 构建聚类结果
        return self._build_clusters(targets, labels)

    def _build_clusters(
        self, targets: List[Target], labels: np.ndarray
    ) -> List[TargetCluster]:
        """
        根据聚类标签构建聚类对象

        Args:
            targets: 目标列表
            labels: 聚类标签(从1开始)

        Returns:
            聚类列表
        """
        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            # 获取该聚类的所有目标
            cluster_targets = [
                targets[i] for i in range(len(targets)) if labels[i] == label
            ]

            # 过滤掉小于最小聚类大小的聚类
            if len(cluster_targets) < self.min_cluster_size:
                continue

            # 计算聚类属性
            centroid = self._calculate_centroid(cluster_targets)
            total_priority = sum(t.priority for t in cluster_targets)
            bounding_box = self._calculate_bounding_box(cluster_targets)

            cluster = TargetCluster(
                cluster_id=f"cluster_{label}_{uuid.uuid4().hex[:8]}",
                targets=cluster_targets,
                centroid=centroid,
                total_priority=total_priority,
                bounding_box=bounding_box
            )
            clusters.append(cluster)

        return clusters

    def _calculate_centroid(self, targets: List[Target]) -> Tuple[float, float]:
        """
        计算聚类质心(简单平均)

        Args:
            targets: 目标列表

        Returns:
            质心坐标 (经度, 纬度)
        """
        lons = [t.longitude for t in targets]
        lats = [t.latitude for t in targets]

        centroid_lon = sum(lons) / len(lons)
        centroid_lat = sum(lats) / len(lats)

        return (centroid_lon, centroid_lat)

    def _calculate_bounding_box(
        self, targets: List[Target]
    ) -> Tuple[float, float, float, float]:
        """
        计算聚类边界框

        Args:
            targets: 目标列表

        Returns:
            边界框 (min_lon, max_lon, min_lat, max_lat)
        """
        lons = [t.longitude for t in targets]
        lats = [t.latitude for t in targets]

        return (min(lons), max(lons), min(lats), max(lats))
