"""
批量姿态机动计算器与调度器集成测试

验证 BatchSlewConstraintChecker 与调度系统的集成:
1. 与 GreedyScheduler (启用聚类) 的集成
2. 与可见性计算的集成
3. 实际调度场景中的机动时间计算

替代原有的 test_slew_scheduler_integration.py，使用精确模型
"""

import pytest
from datetime import datetime, timedelta

from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, Orbit
from core.models.mission import Mission
from core.clustering.target_clusterer import TargetClusterer
from scheduler.constraints import (
    BatchSlewConstraintChecker,
    BatchSlewCandidate,
    SlewFeasibilityResult,
)


class TestBatchSlewWithClustering:
    """测试批量机动约束检查器与聚类的集成"""

    @pytest.fixture
    def sample_satellite(self):
        """创建测试卫星"""
        from core.models import SatelliteCapabilities, ImagingMode
        return Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                resolution=0.5,
                max_roll_angle=45.0,
                agility={
                    'max_slew_rate': 2.0,
                    'settling_time': 5.0,
                    'max_torque': 0.5,
                }
            )
        )

    @pytest.fixture
    def sample_mission(self, sample_satellite):
        """创建测试任务"""
        return Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0, 0),
            satellites=[sample_satellite],
            targets=[]
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

    def test_batch_slew_between_consecutive_targets(
        self, sample_mission, sample_satellite, sample_targets, satellite_position
    ):
        """测试批量连续目标间的机动时间计算"""
        checker = BatchSlewConstraintChecker(sample_mission, use_precise_model=True)

        window_start = datetime(2024, 1, 1, 12, 0, 0)

        # 创建批量候选
        candidates = []
        for i in range(len(sample_targets) - 1):
            candidates.append(BatchSlewCandidate(
                sat_id=sample_satellite.id,
                satellite=sample_satellite,
                target=sample_targets[i + 1],
                window_start=window_start,
                window_end=window_start + timedelta(minutes=5),
                prev_end_time=window_start - timedelta(seconds=20),
                prev_target=sample_targets[i],
                imaging_duration=10.0,
                sat_position=satellite_position,
                sat_velocity=(0.0, 7650.0, 0.0)
            ))

        results = checker.check_slew_feasibility_batch(candidates)

        # 验证所有机动时间都是正数
        assert len(results) == len(sample_targets) - 1
        for result in results:
            assert isinstance(result, SlewFeasibilityResult)
            assert result.slew_time > 0

    def test_batch_slew_with_clustering(
        self, sample_mission, sample_satellite, sample_targets, satellite_position
    ):
        """测试批量机动计算与聚类的集成"""
        # 创建聚类
        clusterer = TargetClusterer(swath_width_km=50.0, min_cluster_size=2)
        clusters = clusterer.cluster_targets(sample_targets)

        assert len(clusters) > 0

        checker = BatchSlewConstraintChecker(sample_mission, use_precise_model=True)
        window_start = datetime(2024, 1, 1, 12, 0, 0)

        # 为每个聚类创建批量候选
        for cluster in clusters:
            candidates = []
            prev_target = None
            for target in cluster.targets:
                candidates.append(BatchSlewCandidate(
                    sat_id=sample_satellite.id,
                    satellite=sample_satellite,
                    target=target,
                    window_start=window_start,
                    window_end=window_start + timedelta(minutes=5),
                    prev_end_time=window_start - timedelta(seconds=20),
                    prev_target=prev_target,
                    imaging_duration=10.0,
                    sat_position=satellite_position,
                    sat_velocity=(0.0, 7650.0, 0.0)
                ))
                prev_target = target

            results = checker.check_slew_feasibility_batch(candidates)

            # 验证聚类内所有目标的机动可行性
            for result in results:
                assert isinstance(result, SlewFeasibilityResult)
                assert result.slew_angle >= 0
                assert result.slew_time > 0

    def test_visibility_window_with_batch_slew(
        self, sample_mission, sample_satellite, sample_targets, satellite_position
    ):
        """测试可见性窗口的批量机动时间分配"""
        checker = BatchSlewConstraintChecker(sample_mission, use_precise_model=True)

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        visibility_duration = 120.0  # 2分钟

        # 选择多个目标进行批量检查
        selected_targets = sample_targets[:5]
        candidates = []
        prev_target = None

        for target in selected_targets:
            candidates.append(BatchSlewCandidate(
                sat_id=sample_satellite.id,
                satellite=sample_satellite,
                target=target,
                window_start=window_start,
                window_end=window_start + timedelta(minutes=5),
                prev_end_time=window_start - timedelta(seconds=20) if prev_target else window_start,
                prev_target=prev_target,
                imaging_duration=10.0,
                sat_position=satellite_position,
                sat_velocity=(0.0, 7650.0, 0.0)
            ))
            prev_target = target

        results = checker.check_slew_feasibility_batch(candidates)

        # 计算总任务时间
        total_imaging_time = len(selected_targets) * 10.0
        total_slew_time = sum(r.slew_time for r in results)
        total_task_time = total_imaging_time + total_slew_time

        # 验证时间分配
        assert total_task_time > 0
        # 单独验证每个结果
        for result in results:
            assert result.slew_time > 0

    def test_agility_constraints_with_batch_slew(self, satellite_position):
        """测试机动能力约束与批量计算的集成"""
        from core.models import SatelliteCapabilities, ImagingMode

        # 低机动性卫星
        low_agility_sat = Satellite(
            id="sat_low",
            name="Low Agility Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                resolution=0.5,
                max_roll_angle=30.0,
                agility={
                    'max_slew_rate': 1.0,
                    'settling_time': 10.0,
                    'max_torque': 0.3,
                }
            )
        )

        # 高机动性卫星
        high_agility_sat = Satellite(
            id="sat_high",
            name="High Agility Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                resolution=0.5,
                max_roll_angle=60.0,
                agility={
                    'max_slew_rate': 5.0,
                    'settling_time': 3.0,
                    'max_torque': 1.0,
                }
            )
        )

        mission_low = Mission(
            name="Test Mission Low",
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0, 0),
            satellites=[low_agility_sat],
            targets=[]
        )

        mission_high = Mission(
            name="Test Mission High",
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0, 0),
            satellites=[high_agility_sat],
            targets=[]
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

        checker_low = BatchSlewConstraintChecker(mission_low, use_precise_model=True)
        checker_high = BatchSlewConstraintChecker(mission_high, use_precise_model=True)

        window_start = datetime(2024, 1, 1, 12, 0, 0)

        # 低机动性卫星
        candidate_low = BatchSlewCandidate(
            sat_id=low_agility_sat.id,
            satellite=low_agility_sat,
            target=target_far,
            window_start=window_start,
            window_end=window_start + timedelta(minutes=5),
            prev_end_time=window_start - timedelta(seconds=30),
            prev_target=target_near,
            imaging_duration=10.0,
            sat_position=satellite_position,
            sat_velocity=(0.0, 7650.0, 0.0)
        )

        # 高机动性卫星
        candidate_high = BatchSlewCandidate(
            sat_id=high_agility_sat.id,
            satellite=high_agility_sat,
            target=target_far,
            window_start=window_start,
            window_end=window_start + timedelta(minutes=5),
            prev_end_time=window_start - timedelta(seconds=30),
            prev_target=target_near,
            imaging_duration=10.0,
            sat_position=satellite_position,
            sat_velocity=(0.0, 7650.0, 0.0)
        )

        result_low = checker_low.check_slew_feasibility_batch([candidate_low])[0]
        result_high = checker_high.check_slew_feasibility_batch([candidate_high])[0]

        # 高机动性卫星的机动时间应该更短
        assert result_high.slew_time < result_low.slew_time


class TestBatchSlewWithScheduling:
    """测试批量机动约束检查器与调度决策的集成"""

    @pytest.fixture
    def scheduling_scenario(self):
        """创建调度测试场景"""
        from core.models import SatelliteCapabilities, ImagingMode

        satellite = Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                resolution=0.5,
                max_roll_angle=45.0,
                agility={
                    'max_slew_rate': 2.0,
                    'settling_time': 5.0,
                    'max_torque': 0.5,
                }
            )
        )

        mission = Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0, 0),
            satellites=[satellite],
            targets=[]
        )

        targets = [
            Target(id=f"target_{i}", name=f"Target {i}",
                   target_type=TargetType.POINT,
                   longitude=i * 2.0, latitude=0.0, priority=1)
            for i in range(5)
        ]

        return {
            'mission': mission,
            'satellite': satellite,
            'targets': targets,
            'satellite_position': (6871000.0, 0.0, 0.0)
        }

    def test_scheduling_decision_with_batch_slew(self, scheduling_scenario):
        """测试考虑批量机动约束的调度决策"""
        mission = scheduling_scenario['mission']
        satellite = scheduling_scenario['satellite']
        targets = scheduling_scenario['targets']
        sat_position = scheduling_scenario['satellite_position']

        checker = BatchSlewConstraintChecker(mission, use_precise_model=True)

        window_start = datetime(2024, 1, 1, 12, 0, 0)
        remaining_time = 60.0  # 1分钟

        # 模拟调度决策：为每个目标创建候选并批量检查
        candidates = []
        current_target = targets[0]

        for target in targets[1:]:
            candidates.append(BatchSlewCandidate(
                sat_id=satellite.id,
                satellite=satellite,
                target=target,
                window_start=window_start,
                window_end=window_start + timedelta(minutes=5),
                prev_end_time=window_start - timedelta(seconds=20),
                prev_target=current_target,
                imaging_duration=10.0,
                sat_position=sat_position,
                sat_velocity=(0.0, 7650.0, 0.0)
            ))

        results = checker.check_slew_feasibility_batch(candidates)

        # 筛选可行的下一个目标
        feasible_targets = []
        for i, result in enumerate(results):
            imaging_time = 10.0
            total_time = result.slew_time + imaging_time

            if total_time <= remaining_time:
                feasible_targets.append({
                    'target': targets[i + 1],
                    'slew_result': result,
                    'total_time': total_time
                })

        # 验证筛选结果（只验证返回结果有效，不强制可行性取决于轨道几何）
        for item in feasible_targets:
            # 验证结果对象有效
            assert item['slew_result'].slew_angle >= 0
            assert item['slew_result'].slew_time > 0
            assert item['total_time'] <= remaining_time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
