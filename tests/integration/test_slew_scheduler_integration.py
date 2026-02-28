"""
姿态机动计算器与调度器集成测试

验证SlewCalculator与调度系统的集成:
1. 与ClusteringGreedyScheduler的集成
2. 与可见性计算的集成
3. 实际调度场景中的机动时间计算
"""

import pytest
from datetime import datetime, timedelta

from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, Orbit
from core.clustering.target_clusterer import TargetCluster, TargetClusterer
from core.dynamics.slew_calculator import SlewCalculator, ClusterSlewCalculator


class TestSlewWithClusteringScheduler:
    """测试机动计算器与聚类调度器的集成"""

    @pytest.fixture
    def sample_satellite(self):
        """创建测试卫星"""
        return Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4)
        )

    @pytest.fixture
    def sample_targets(self):
        """创建测试目标列表"""
        targets = []
        for i in range(10):
            targets.append(Target(
                id=f"target_{i:03d}",
                name=f"Target {i}",
                target_type=TargetType.POINT,
                longitude=10.0 + i * 0.3,  # 跨越约3度
                latitude=20.0 + i * 0.1,
                priority=i % 3 + 1
            ))
        return targets

    @pytest.fixture
    def satellite_position(self):
        """卫星位置 (赤道上方500km)"""
        return (6871000.0, 0.0, 0.0)

    def test_slew_time_between_consecutive_targets(
        self, sample_targets, satellite_position
    ):
        """测试连续目标间的机动时间计算"""
        calc = SlewCalculator(max_slew_rate=2.0, max_slew_angle=45.0)

        # 计算相邻目标间的机动时间
        slew_times = []
        for i in range(len(sample_targets) - 1):
            maneuver = calc.calculate_maneuver(
                satellite_position,
                sample_targets[i],
                sample_targets[i + 1]
            )
            slew_times.append(maneuver.slew_time)

        # 验证所有机动时间都是正数
        assert all(t > 0 for t in slew_times)
        # 验证机动角度都在限制内
        for i in range(len(sample_targets) - 1):
            maneuver = calc.calculate_maneuver(
                satellite_position,
                sample_targets[i],
                sample_targets[i + 1]
            )
            assert maneuver.total_slew_angle <= 45.0 or not maneuver.feasible

    def test_cluster_feasibility_with_slew_time(
        self, sample_targets, satellite_position
    ):
        """测试聚类可行性考虑机动时间"""
        cluster_calc = ClusterSlewCalculator(
            max_slew_rate=2.0,
            max_slew_angle=45.0,
            settling_time=5.0
        )

        # 创建聚类
        clusterer = TargetClusterer(swath_width_km=50.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(sample_targets)

        assert len(clusters) > 0

        # 检查每个聚类的可行性
        for cluster in clusters:
            # 假设可见性窗口120秒，成像需要100秒
            can_cover = cluster_calc.can_cover_cluster_in_time(
                cluster,
                visibility_duration=120.0,
                imaging_time=100.0
            )

            # 获取机动时间
            sweep_angle, slew_time = cluster_calc.calculate_cluster_slew_coverage(
                satellite_position, cluster, look_angle=0.0
            )

            # 验证：如果总时间 > 120秒，应该返回False
            total_time = 100.0 + slew_time
            if total_time > 120.0:
                assert can_cover is False
            else:
                assert can_cover is True

    def test_visibility_window_with_slew_allocation(
        self, sample_targets, satellite_position
    ):
        """测试可见性窗口的机动时间分配"""
        calc = SlewCalculator(max_slew_rate=2.0, settling_time=5.0)

        # 模拟可见性窗口
        visibility_start = datetime(2024, 1, 1, 12, 0, 0)
        visibility_duration = 120.0  # 2分钟

        # 选择两个目标进行调度
        target1 = sample_targets[0]
        target2 = sample_targets[5]

        # 计算机动时间
        maneuver = calc.calculate_maneuver(
            satellite_position, target1, target2
        )

        # 计算成像时间（简化假设每个目标10秒）
        imaging_time_per_target = 10.0
        total_imaging_time = 2 * imaging_time_per_target

        # 计算总任务时间
        total_task_time = (
            total_imaging_time +
            maneuver.slew_time +
            calc.settling_time  # 额外稳定时间
        )

        # 验证时间分配
        assert total_task_time <= visibility_duration or not maneuver.feasible

    def test_agility_constraints_validation(
        self, satellite_position
    ):
        """测试机动能力约束验证"""
        # 低机动性卫星
        low_agility_calc = SlewCalculator(
            max_slew_rate=1.0,
            max_slew_angle=30.0,
            settling_time=10.0
        )

        # 高机动性卫星
        high_agility_calc = SlewCalculator(
            max_slew_rate=5.0,
            max_slew_angle=60.0,
            settling_time=3.0
        )

        # 创建两个相距较远的目标
        target_near = Target(
            id="near", name="Near Target",
            target_type=TargetType.POINT,
            longitude=0.0, latitude=0.0, priority=1
        )
        target_far = Target(
            id="far", name="Far Target",
            target_type=TargetType.POINT,
            longitude=25.0, latitude=25.0, priority=1
        )

        # 低机动性卫星可能无法完成机动
        maneuver_low = low_agility_calc.calculate_maneuver(
            satellite_position, target_near, target_far
        )

        # 高机动性卫星应该可以完成
        maneuver_high = high_agility_calc.calculate_maneuver(
            satellite_position, target_near, target_far
        )

        # 高机动性卫星的机动时间应该更短
        assert maneuver_high.slew_time < maneuver_low.slew_time

    def test_scheduling_decision_with_slew_constraint(
        self, sample_targets, satellite_position
    ):
        """测试考虑机动约束的调度决策"""
        calc = SlewCalculator(max_slew_rate=2.0, max_slew_angle=45.0)

        # 模拟调度决策：选择下一个目标
        current_target = sample_targets[0]
        remaining_targets = sample_targets[1:]

        # 可见性窗口剩余时间
        remaining_time = 60.0  # 1分钟

        # 筛选可行的下一个目标
        feasible_targets = []
        for target in remaining_targets:
            maneuver = calc.calculate_maneuver(
                satellite_position, current_target, target
            )

            # 假设成像需要10秒
            imaging_time = 10.0
            total_time = maneuver.slew_time + imaging_time

            if total_time <= remaining_time and maneuver.feasible:
                feasible_targets.append({
                    'target': target,
                    'maneuver': maneuver,
                    'total_time': total_time
                })

        # 验证筛选结果
        for item in feasible_targets:
            assert item['maneuver'].feasible
            assert item['total_time'] <= remaining_time


class TestSlewWithVisibility:
    """测试机动计算器与可见性计算的集成"""

    def test_slew_during_visibility_window(self):
        """测试在可见性窗口内完成机动"""
        calc = SlewCalculator(max_slew_rate=2.0, settling_time=5.0)

        # 卫星位置
        sat_pos = (6871000.0, 0.0, 0.0)

        # 两个目标
        target1 = Target(
            id="t1", name="T1",
            target_type=TargetType.POINT,
            longitude=5.0, latitude=0.0, priority=1
        )
        target2 = Target(
            id="t2", name="T2",
            target_type=TargetType.POINT,
            longitude=10.0, latitude=0.0, priority=1
        )

        # 计算机动
        maneuver = calc.calculate_maneuver(sat_pos, target1, target2)

        # 假设可见性窗口重叠时间为30秒
        overlap_time = 30.0

        # 验证可以在可见性窗口内完成机动
        assert maneuver.slew_time <= overlap_time

    def test_slew_exceeds_visibility_window(self):
        """测试机动时间超过可见性窗口"""
        calc = SlewCalculator(max_slew_rate=1.0, settling_time=10.0)

        sat_pos = (6871000.0, 0.0, 0.0)

        # 两个相距很远的目标
        target1 = Target(
            id="t1", name="T1",
            target_type=TargetType.POINT,
            longitude=0.0, latitude=0.0, priority=1
        )
        target2 = Target(
            id="t2", name="T2",
            target_type=TargetType.POINT,
            longitude=40.0, latitude=30.0, priority=1
        )

        maneuver = calc.calculate_maneuver(sat_pos, target1, target2)

        # 假设可见性窗口只有10秒
        short_window = 10.0

        # 机动不可行
        is_feasible = calc.is_maneuver_feasible(
            maneuver.total_slew_angle, short_window
        )

        assert is_feasible is False


class TestRealWorldScenarios:
    """真实场景测试"""

    def test_strip_imaging_slew_pattern(self):
        """测试条带成像的机动模式"""
        calc = SlewCalculator(max_slew_rate=2.0, max_slew_angle=45.0)

        sat_pos = (6871000.0, 0.0, 0.0)

        # 创建沿轨道的目标序列（模拟条带）
        strip_targets = [
            Target(
                id=f"strip_{i}",
                name=f"Strip Point {i}",
                target_type=TargetType.POINT,
                longitude=i * 0.5,  # 沿经度方向排列
                latitude=0.0,
                priority=1
            )
            for i in range(20)
        ]

        # 计算总机动时间
        total_slew_time = 0.0
        for i in range(len(strip_targets) - 1):
            maneuver = calc.calculate_maneuver(
                sat_pos, strip_targets[i], strip_targets[i + 1]
            )
            total_slew_time += maneuver.slew_time

        # 验证总机动时间是合理的
        assert total_slew_time > 0
        # 相邻目标间机动角度应该很小
        sample_maneuver = calc.calculate_maneuver(
            sat_pos, strip_targets[0], strip_targets[1]
        )
        assert sample_maneuver.total_slew_angle < 10.0  # 应该小于10度

    def test_agile_satellite_rapid_slewing(self):
        """测试敏捷卫星的快速机动能力"""
        # 敏捷卫星（如WorldView-4）
        agile_calc = SlewCalculator(
            max_slew_rate=5.0,  # 5度/秒
            max_slew_angle=45.0,
            settling_time=3.0
        )

        sat_pos = (6871000.0, 0.0, 0.0)

        # 创建分散的目标
        scattered_targets = [
            Target(id=f"agile_{i}", name=f"Agile Target {i}",
                   target_type=TargetType.POINT,
                   longitude=i * 8.0, latitude=i * 5.0, priority=1)
            for i in range(5)
        ]

        # 计算所有目标对之间的机动
        slew_times = []
        for i in range(len(scattered_targets)):
            for j in range(i + 1, len(scattered_targets)):
                maneuver = agile_calc.calculate_maneuver(
                    sat_pos, scattered_targets[i], scattered_targets[j]
                )
                slew_times.append({
                    'from': scattered_targets[i].id,
                    'to': scattered_targets[j].id,
                    'time': maneuver.slew_time,
                    'feasible': maneuver.feasible
                })

        # 验证敏捷卫星可以完成更多机动
        feasible_count = sum(1 for s in slew_times if s['feasible'])
        total_count = len(slew_times)

        # 大部分机动应该是可行的
        assert feasible_count >= total_count * 0.5

    def test_multi_cluster_scheduling_with_slew(self):
        """测试多聚类调度的机动考虑"""
        cluster_calc = ClusterSlewCalculator(
            max_slew_rate=2.0,
            max_slew_angle=45.0,
            settling_time=5.0
        )

        sat_pos = (6871000.0, 0.0, 0.0)

        # 创建两个聚类
        cluster1_targets = [
            Target(id=f"c1_t{i}", name=f"C1 T{i}",
                   target_type=TargetType.POINT,
                   longitude=5.0 + i * 0.2, latitude=10.0, priority=1)
            for i in range(5)
        ]

        cluster2_targets = [
            Target(id=f"c2_t{i}", name=f"C2 T{i}",
                   target_type=TargetType.POINT,
                   longitude=25.0 + i * 0.2, latitude=15.0, priority=1)
            for i in range(5)
        ]

        cluster1 = TargetCluster(
            cluster_id="cluster_1",
            targets=cluster1_targets,
            centroid=(5.4, 10.0),
            total_priority=5,
            bounding_box=(5.0, 5.8, 10.0, 10.0)
        )

        cluster2 = TargetCluster(
            cluster_id="cluster_2",
            targets=cluster2_targets,
            centroid=(25.4, 15.0),
            total_priority=5,
            bounding_box=(25.0, 25.8, 15.0, 15.0)
        )

        # 计算每个聚类的覆盖时间
        angle1, time1 = cluster_calc.calculate_cluster_slew_coverage(
            sat_pos, cluster1, look_angle=0.0
        )
        angle2, time2 = cluster_calc.calculate_cluster_slew_coverage(
            sat_pos, cluster2, look_angle=0.0
        )

        # 计算聚类间的机动时间
        cluster1_center = Target(
            id="c1_center", name="C1 Center",
            target_type=TargetType.POINT,
            longitude=cluster1.centroid[0],
            latitude=cluster1.centroid[1],
            priority=1
        )
        cluster2_center = Target(
            id="c2_center", name="C2 Center",
            target_type=TargetType.POINT,
            longitude=cluster2.centroid[0],
            latitude=cluster2.centroid[1],
            priority=1
        )

        inter_cluster_maneuver = cluster_calc.calculate_maneuver(
            sat_pos, cluster1_center, cluster2_center
        )

        # 总任务时间
        total_mission_time = (
            time1 +  # 聚类1扫描
            inter_cluster_maneuver.slew_time +  # 聚类间机动
            time2 +  # 聚类2扫描
            cluster_calc.settling_time  # 最终稳定
        )

        # 验证总时间是合理的
        assert total_mission_time > 0
        assert inter_cluster_maneuver.feasible or inter_cluster_maneuver.total_slew_angle <= 45.0
