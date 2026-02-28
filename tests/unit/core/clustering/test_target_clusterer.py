"""
目标聚类模块单元测试 - TDD方式编写

测试覆盖:
1. 空目标列表
2. 单目标(无聚类)
3. 两个接近目标(应聚类)
4. 两个远离目标(不应聚类)
5. 多聚类场景
6. 优先级聚合
7. 质心计算
8. 边界框计算
"""

import pytest
import math
from typing import List, Tuple
from dataclasses import dataclass

from core.models.target import Target, TargetType


# 测试数据坐标
# 聚类1 (接近目标, 约1km间距): (116.3, 39.9), (116.31, 39.91), (116.29, 39.89)
# 聚类2 (远离): (116.5, 40.1) - 距离聚类1约30km

CLUSTER_1_COORDS = [
    (116.3, 39.9),    # 目标1
    (116.31, 39.91),  # 目标2 - 约1.3km远离目标1
    (116.29, 39.89),  # 目标3 - 约1.4km远离目标1
]

CLUSTER_2_COORDS = [
    (116.5, 40.1),    # 目标4 - 距离聚类1约30km
]

CLUSTER_3_COORDS = [
    (117.0, 40.5),    # 目标5 - 距离其他约70km
    (117.02, 40.52),  # 目标6 - 约2.5km远离目标5
]


def create_target(target_id: str, lon: float, lat: float, priority: int = 1) -> Target:
    """辅助函数: 创建测试目标"""
    return Target(
        id=target_id,
        name=f"Target_{target_id}",
        target_type=TargetType.POINT,
        longitude=lon,
        latitude=lat,
        priority=priority
    )


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """辅助函数: 计算两点间球面距离(公里)"""
    R = 6371.0  # 地球半径(公里)

    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class TestTargetCluster:
    """测试TargetCluster数据类"""

    def test_cluster_creation(self):
        """测试聚类对象创建"""
        from core.clustering.target_clusterer import TargetCluster

        target = create_target("T1", 116.3, 39.9, priority=5)
        cluster = TargetCluster(
            cluster_id="cluster_1",
            targets=[target],
            centroid=(116.3, 39.9),
            total_priority=5,
            bounding_box=(116.3, 116.3, 39.9, 39.9)
        )

        assert cluster.cluster_id == "cluster_1"
        assert len(cluster.targets) == 1
        assert cluster.centroid == (116.3, 39.9)
        assert cluster.total_priority == 5
        assert cluster.bounding_box == (116.3, 116.3, 39.9, 39.9)


class TestTargetClustererInitialization:
    """测试TargetClusterer初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        from core.clustering.target_clusterer import TargetClusterer

        clusterer = TargetClusterer()

        assert clusterer.swath_width_km == 10.0
        assert clusterer.min_cluster_size == 2

    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        from core.clustering.target_clusterer import TargetClusterer

        clusterer = TargetClusterer(swath_width_km=20.0, min_cluster_size=3)

        assert clusterer.swath_width_km == 20.0
        assert clusterer.min_cluster_size == 3


class TestTargetClustererEmptyCases:
    """测试空输入场景"""

    def test_empty_target_list(self):
        """测试空目标列表返回空聚类列表"""
        from core.clustering.target_clusterer import TargetClusterer

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets([])

        assert clusters == []
        assert isinstance(clusters, list)


class TestTargetClustererSingleTarget:
    """测试单目标场景"""

    def test_single_target_no_cluster(self):
        """测试单目标无法形成聚类(少于min_cluster_size)"""
        from core.clustering.target_clusterer import TargetClusterer

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        target = create_target("T1", 116.3, 39.9)

        clusters = clusterer.cluster_targets([target])

        # 单目标不满足最小聚类大小,应返回空列表
        assert clusters == []


class TestTargetClustererCloseTargets:
    """测试接近目标聚类"""

    def test_two_close_targets_form_cluster(self):
        """测试两个接近目标应形成聚类"""
        from core.clustering.target_clusterer import TargetClusterer

        # 使用约1km间距的目标
        target1 = create_target("T1", 116.3, 39.9)
        target2 = create_target("T2", 116.31, 39.91)

        distance = haversine_distance_km(116.3, 39.9, 116.31, 39.91)
        assert distance < 2.0  # 确认距离约1.3km

        # 使用10km幅宽,应能聚类
        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets([target1, target2])

        assert len(clusters) == 1
        assert len(clusters[0].targets) == 2

    def test_three_close_targets_form_single_cluster(self):
        """测试三个接近目标形成单一聚类"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target(f"T{i}", lon, lat)
            for i, (lon, lat) in enumerate(CLUSTER_1_COORDS)
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1
        assert len(clusters[0].targets) == 3


class TestTargetClustererDistantTargets:
    """测试远离目标不聚类"""

    def test_two_distant_targets_no_cluster(self):
        """测试两个远离目标不形成聚类"""
        from core.clustering.target_clusterer import TargetClusterer

        # 使用约30km间距的目标
        target1 = create_target("T1", 116.3, 39.9)
        target2 = create_target("T2", 116.5, 40.1)

        distance = haversine_distance_km(116.3, 39.9, 116.5, 40.1)
        assert distance > 20.0  # 确认距离约30km

        # 使用10km幅宽,不应聚类
        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets([target1, target2])

        # 两个目标都标记为噪声点,不形成聚类
        assert len(clusters) == 0


class TestTargetClustererMultipleClusters:
    """测试多聚类场景"""

    def test_two_separate_clusters(self):
        """测试两个独立聚类"""
        from core.clustering.target_clusterer import TargetClusterer

        # 聚类1: 3个接近目标
        cluster_1_targets = [
            create_target(f"C1_{i}", lon, lat)
            for i, (lon, lat) in enumerate(CLUSTER_1_COORDS)
        ]

        # 聚类2: 2个接近目标(远离聚类1)
        cluster_2_targets = [
            create_target(f"C2_{i}", lon, lat)
            for i, (lon, lat) in enumerate(CLUSTER_3_COORDS)
        ]

        all_targets = cluster_1_targets + cluster_2_targets

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(all_targets)

        assert len(clusters) == 2

        # 验证每个聚类的目标数量
        cluster_sizes = sorted([len(c.targets) for c in clusters])
        assert cluster_sizes == [2, 3]

    def test_clusters_with_noise(self):
        """测试聚类与噪声点共存"""
        from core.clustering.target_clusterer import TargetClusterer

        # 聚类: 3个接近目标
        cluster_targets = [
            create_target(f"C_{i}", lon, lat)
            for i, (lon, lat) in enumerate(CLUSTER_1_COORDS)
        ]

        # 噪声点: 远离的目标(不形成聚类)
        noise_target = create_target("NOISE", 117.0, 40.5)

        all_targets = cluster_targets + [noise_target]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(all_targets)

        # 只有一个有效聚类
        assert len(clusters) == 1
        assert len(clusters[0].targets) == 3


class TestTargetClustererPriorityAggregation:
    """测试优先级聚合"""

    def test_total_priority_calculation(self):
        """测试聚类总优先级计算"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.3, 39.9, priority=3),
            create_target("T2", 116.31, 39.91, priority=5),
            create_target("T3", 116.29, 39.89, priority=2),
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1
        assert clusters[0].total_priority == 10  # 3 + 5 + 2

    def test_priority_with_single_cluster(self):
        """测试单聚类优先级"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.3, 39.9, priority=1),
            create_target("T2", 116.31, 39.91, priority=1),
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert clusters[0].total_priority == 2


class TestTargetClustererCentroid:
    """测试质心计算"""

    def test_centroid_calculation_two_points(self):
        """测试两点质心计算"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.0, 39.0),
            create_target("T2", 118.0, 41.0),
        ]

        clusterer = TargetClusterer(swath_width_km=300.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1
        centroid_lon, centroid_lat = clusters[0].centroid

        # 质心应为中点
        assert pytest.approx(centroid_lon, abs=0.01) == 117.0
        assert pytest.approx(centroid_lat, abs=0.01) == 40.0

    def test_centroid_calculation_three_points(self):
        """测试三点质心计算"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.3, 39.9),
            create_target("T2", 116.31, 39.91),
            create_target("T3", 116.29, 39.89),
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1
        centroid_lon, centroid_lat = clusters[0].centroid

        # 质心应为平均值
        expected_lon = (116.3 + 116.31 + 116.29) / 3
        expected_lat = (39.9 + 39.91 + 39.89) / 3

        assert pytest.approx(centroid_lon, abs=0.001) == expected_lon
        assert pytest.approx(centroid_lat, abs=0.001) == expected_lat


class TestTargetClustererBoundingBox:
    """测试边界框计算"""

    def test_bounding_box_calculation(self):
        """测试边界框计算"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.3, 39.9),
            create_target("T2", 116.31, 39.91),
            create_target("T3", 116.29, 39.89),
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1
        min_lon, max_lon, min_lat, max_lat = clusters[0].bounding_box

        assert min_lon == 116.29
        assert max_lon == 116.31
        assert min_lat == 39.89
        assert max_lat == 39.91

    def test_bounding_box_single_point(self):
        """测试单点边界框"""
        from core.clustering.target_clusterer import TargetClusterer, TargetCluster

        # 直接测试边界框计算逻辑
        target = create_target("T1", 116.3, 39.9)

        # 由于min_cluster_size=2,单目标不会形成聚类
        # 所以我们测试聚类对象的创建
        cluster = TargetCluster(
            cluster_id="test",
            targets=[target],
            centroid=(116.3, 39.9),
            total_priority=1,
            bounding_box=(116.3, 116.3, 39.9, 39.9)
        )

        min_lon, max_lon, min_lat, max_lat = cluster.bounding_box
        assert min_lon == max_lon == 116.3
        assert min_lat == max_lat == 39.9


class TestTargetClustererEdgeCases:
    """测试边界情况"""

    def test_all_targets_noise(self):
        """测试所有目标都是噪声点的情况"""
        from core.clustering.target_clusterer import TargetClusterer

        # 创建分散的目标(距离都超过幅宽)
        targets = [
            create_target("T1", 116.0, 39.0),
            create_target("T2", 117.0, 40.0),
            create_target("T3", 118.0, 41.0),
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        # 没有聚类形成
        assert clusters == []

    def test_targets_at_same_location(self):
        """测试同一位置的目标"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.3, 39.9),
            create_target("T2", 116.3, 39.9),
            create_target("T3", 116.3, 39.9),
        ]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1
        assert len(clusters[0].targets) == 3

    def test_large_number_of_targets(self):
        """测试大量目标性能"""
        from core.clustering.target_clusterer import TargetClusterer

        # 创建100个目标,分布在10x10网格上
        targets = []
        for i in range(10):
            for j in range(10):
                lon = 116.0 + i * 0.01  # 约1km间距
                lat = 39.0 + j * 0.01
                targets.append(create_target(f"T{i}_{j}", lon, lat))

        clusterer = TargetClusterer(swath_width_km=5.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        # 应该形成多个聚类
        assert len(clusters) > 0
        total_clustered = sum(len(c.targets) for c in clusters)
        assert total_clustered > 0

    def test_min_cluster_size_boundary(self):
        """测试最小聚类大小边界"""
        from core.clustering.target_clusterer import TargetClusterer

        targets = [
            create_target("T1", 116.3, 39.9),
            create_target("T2", 116.31, 39.91),
        ]

        # min_cluster_size=3,两个目标不应形成聚类
        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=3)
        clusters = clusterer.cluster_targets(targets)

        assert clusters == []

        # min_cluster_size=2,应形成聚类
        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)

        assert len(clusters) == 1

    def test_swath_width_boundary(self):
        """测试幅宽边界"""
        from core.clustering.target_clusterer import TargetClusterer

        # 两个目标相距约1.3km
        targets = [
            create_target("T1", 116.3, 39.9),
            create_target("T2", 116.31, 39.91),
        ]

        # 幅宽1km,不应聚类
        clusterer = TargetClusterer(swath_width_km=1.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)
        assert len(clusters) == 0

        # 幅宽2km,应聚类
        clusterer = TargetClusterer(swath_width_km=2.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(targets)
        assert len(clusters) == 1


class TestTargetClustererIntegration:
    """集成测试场景"""

    def test_realistic_scenario(self):
        """测试真实场景: 多个聚类加噪声"""
        from core.clustering.target_clusterer import TargetClusterer

        # 聚类1: 北京区域3个目标
        beijing_targets = [
            create_target(f"BJ_{i}", lon, lat, priority=i+1)
            for i, (lon, lat) in enumerate(CLUSTER_1_COORDS)
        ]

        # 聚类2: 天津区域2个目标
        tianjin_targets = [
            create_target(f"TJ_{i}", lon, lat, priority=i+1)
            for i, (lon, lat) in enumerate(CLUSTER_3_COORDS)
        ]

        # 孤立目标: 上海(远离)
        shanghai_target = create_target("SH_1", 121.47, 31.23, priority=5)

        all_targets = beijing_targets + tianjin_targets + [shanghai_target]

        clusterer = TargetClusterer(swath_width_km=10.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(all_targets)

        # 应形成2个聚类
        assert len(clusters) == 2

        # 验证聚类属性
        for cluster in clusters:
            assert cluster.cluster_id is not None
            assert len(cluster.targets) >= 2
            assert cluster.total_priority > 0
            assert cluster.centroid is not None
            assert cluster.bounding_box is not None

        # 验证总优先级
        total_priority = sum(c.total_priority for c in clusters)
        assert total_priority == (1+2+3) + (1+2)  # 北京聚类 + 天津聚类
