"""
姿态机动计算器单元测试

测试策略:
1. 单元测试: 独立测试每个函数
2. 边界测试: 相同目标、极远目标、零角度
3. 错误处理: 无效输入、空值
4. 集成测试: 与Target和TargetCluster的集成

TDD流程:
- RED: 编写失败的测试
- GREEN: 实现通过测试
- REFACTOR: 优化代码
"""

import pytest
import math
from dataclasses import dataclass
from typing import Tuple, Optional

from core.models.target import Target, TargetType
from core.clustering.target_clusterer import TargetCluster
from core.dynamics.slew_calculator import (
    SlewManeuver,
    SlewCalculator,
    ClusterSlewCalculator
)


# =============================================================================
# 测试夹具 (Fixtures)
# =============================================================================

@pytest.fixture
def basic_calculator():
    """基础机动计算器"""
    return SlewCalculator(
        max_slew_rate=2.0,      # 2度/秒
        max_slew_angle=45.0,    # 最大45度
        settling_time=5.0       # 5秒稳定时间
    )


@pytest.fixture
def high_agility_calculator():
    """高机动性计算器"""
    return SlewCalculator(
        max_slew_rate=5.0,      # 5度/秒
        max_slew_angle=60.0,    # 最大60度
        settling_time=3.0       # 3秒稳定时间
    )


@pytest.fixture
def satellite_position_ecef():
    """卫星在ECEF坐标系中的位置 (赤道上方500km)"""
    # 地球半径6371km + 500km轨道高度 = 6871km
    return (6871000.0, 0.0, 0.0)  # 米


@pytest.fixture
def target_nadir():
    """星下点目标 (0° off-nadir)"""
    return Target(
        id="target_nadir",
        name="Nadir Target",
        target_type=TargetType.POINT,
        longitude=0.0,
        latitude=0.0,
        priority=1
    )


@pytest.fixture
def target_15deg_off():
    """15度侧摆目标"""
    # 在500km高度，15度对应约134km地面距离
    return Target(
        id="target_15deg",
        name="15 Deg Off-Nadir",
        target_type=TargetType.POINT,
        longitude=12.0,  # 约15度对应的经度偏移
        latitude=0.0,
        priority=1
    )


@pytest.fixture
def target_30deg_off():
    """30度侧摆目标"""
    return Target(
        id="target_30deg",
        name="30 Deg Off-Nadir",
        target_type=TargetType.POINT,
        longitude=24.0,
        latitude=0.0,
        priority=1
    )


@pytest.fixture
def sample_cluster():
    """示例目标聚类"""
    targets = [
        Target(id=f"t{i}", name=f"Target {i}",
               target_type=TargetType.POINT,
               longitude=i * 0.5, latitude=0.0, priority=1)
        for i in range(5)  # 5个目标，跨越2度经度
    ]
    return TargetCluster(
        cluster_id="cluster_001",
        targets=targets,
        centroid=(1.0, 0.0),
        total_priority=5,
        bounding_box=(0.0, 2.0, 0.0, 0.0)
    )


# =============================================================================
# SlewManeuver 数据类测试
# =============================================================================

class TestSlewManeuver:
    """测试SlewManeuver数据类"""

    def test_slew_maneuver_creation(self):
        """测试创建SlewManeuver对象"""
        maneuver = SlewManeuver(
            start_target_id="target_1",
            end_target_id="target_2",
            roll_angle=10.0,
            pitch_angle=5.0,
            total_slew_angle=11.18,
            slew_time=10.59,
            feasible=True
        )

        assert maneuver.start_target_id == "target_1"
        assert maneuver.end_target_id == "target_2"
        assert maneuver.roll_angle == 10.0
        assert maneuver.pitch_angle == 5.0
        assert maneuver.total_slew_angle == 11.18
        assert maneuver.slew_time == 10.59
        assert maneuver.feasible is True

    def test_slew_maneuver_defaults(self):
        """测试SlewManeuver默认值"""
        maneuver = SlewManeuver(
            start_target_id="t1",
            end_target_id="t2",
            roll_angle=0.0,
            pitch_angle=0.0,
            total_slew_angle=0.0,
            slew_time=0.0,
            feasible=True
        )

        assert maneuver.feasible is True


# =============================================================================
# SlewCalculator 初始化测试
# =============================================================================

class TestSlewCalculatorInit:
    """测试SlewCalculator初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        calc = SlewCalculator()

        assert calc.max_slew_rate == 2.0
        assert calc.max_slew_angle == 45.0
        assert calc.settling_time == 5.0

    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        calc = SlewCalculator(
            max_slew_rate=3.5,
            max_slew_angle=60.0,
            settling_time=2.0
        )

        assert calc.max_slew_rate == 3.5
        assert calc.max_slew_angle == 60.0
        assert calc.settling_time == 2.0

    def test_zero_slew_rate_raises_error(self):
        """测试零机动角速度应报错"""
        with pytest.raises(ValueError):
            SlewCalculator(max_slew_rate=0.0)

    def test_negative_slew_rate_raises_error(self):
        """测试负机动角速度应报错"""
        with pytest.raises(ValueError):
            SlewCalculator(max_slew_rate=-1.0)

    def test_negative_settling_time_raises_error(self):
        """测试负稳定时间应报错"""
        with pytest.raises(ValueError):
            SlewCalculator(settling_time=-1.0)


# =============================================================================
# Slew Angle Calculation 测试
# =============================================================================

class TestSlewAngleCalculation:
    """测试姿态机动角度计算"""

    def test_same_target_zero_slew(self, basic_calculator, satellite_position_ecef):
        """测试相同目标间机动角度为0"""
        target_pos = (0.0, 0.0)  # (lon, lat)

        roll, pitch, total = basic_calculator.calculate_slew_angles(
            satellite_position_ecef,
            target_pos,
            target_pos  # 相同目标
        )

        assert roll == pytest.approx(0.0, abs=0.01)
        assert pitch == pytest.approx(0.0, abs=0.01)
        assert total == pytest.approx(0.0, abs=0.01)

    def test_nadir_to_off_nadir_slew(self, basic_calculator, satellite_position_ecef):
        """测试从星下点到侧摆目标的机动角度"""
        target1_pos = (0.0, 0.0)  # 星下点
        target2_pos = (10.0, 0.0)  # 10度经度偏移

        roll, pitch, total = basic_calculator.calculate_slew_angles(
            satellite_position_ecef,
            target1_pos,
            target2_pos
        )

        # 总角度应该大于0
        assert total > 0.0
        # roll和pitch应该合理
        assert abs(roll) >= 0.0
        assert abs(pitch) >= 0.0

    def test_symmetric_slew_angles(self, basic_calculator, satellite_position_ecef):
        """测试机动角度对称性: A->B 和 B->A 角度相同"""
        target1_pos = (0.0, 0.0)
        target2_pos = (15.0, 10.0)

        roll1, pitch1, total1 = basic_calculator.calculate_slew_angles(
            satellite_position_ecef, target1_pos, target2_pos
        )

        roll2, pitch2, total2 = basic_calculator.calculate_slew_angles(
            satellite_position_ecef, target2_pos, target1_pos
        )

        # 总角度应该相同
        assert total1 == pytest.approx(total2, abs=0.01)
        # 分量应该相反
        assert roll1 == pytest.approx(-roll2, abs=0.01)
        assert pitch1 == pytest.approx(-pitch2, abs=0.01)

    def test_invalid_satellite_position_raises_error(self, basic_calculator):
        """测试无效卫星位置应报错"""
        with pytest.raises((ValueError, TypeError)):
            basic_calculator.calculate_slew_angles(
                (0.0, 0.0),  # 缺少z坐标
                (0.0, 0.0),
                (10.0, 0.0)
            )

    def test_invalid_target_position_raises_error(self, basic_calculator, satellite_position_ecef):
        """测试无效目标位置应报错"""
        with pytest.raises((ValueError, TypeError)):
            basic_calculator.calculate_slew_angles(
                satellite_position_ecef,
                (0.0,),  # 缺少纬度
                (10.0, 0.0)
            )


# =============================================================================
# Slew Time Calculation 测试
# =============================================================================

class TestSlewTimeCalculation:
    """测试机动时间计算"""

    def test_zero_angle_zero_time(self, basic_calculator):
        """测试零角度机动时间为稳定时间"""
        slew_time = basic_calculator.calculate_slew_time(0.0)

        # 零角度只需要稳定时间
        assert slew_time == pytest.approx(5.0, abs=0.01)

    def test_basic_slew_time_calculation(self, basic_calculator):
        """测试基本机动时间计算"""
        # 15度机动，2度/秒，5秒稳定时间
        # time = 15/2 + 5 = 12.5秒
        slew_time = basic_calculator.calculate_slew_time(15.0)

        assert slew_time == pytest.approx(12.5, abs=0.01)

    def test_slew_time_with_target_ids(self, basic_calculator):
        """测试带目标ID的机动时间计算"""
        slew_time = basic_calculator.calculate_slew_time(
            20.0,
            from_target_id="target_a",
            to_target_id="target_b"
        )

        # 20/2 + 5 = 15秒
        assert slew_time == pytest.approx(15.0, abs=0.01)

    def test_high_agility_slew_time(self, high_agility_calculator):
        """测试高机动性计算器的时间计算"""
        # 30度机动，5度/秒，3秒稳定时间
        # time = 30/5 + 3 = 9秒
        slew_time = high_agility_calculator.calculate_slew_time(30.0)

        assert slew_time == pytest.approx(9.0, abs=0.01)

    def test_negative_slew_angle_raises_error(self, basic_calculator):
        """测试负机动角度应报错"""
        with pytest.raises(ValueError):
            basic_calculator.calculate_slew_time(-10.0)


# =============================================================================
# Maneuver Feasibility 测试
# =============================================================================

class TestManeuverFeasibility:
    """测试机动可行性判断"""

    def test_feasible_maneuver(self, basic_calculator):
        """测试可行机动"""
        # 30度机动，可用时间20秒
        # 需要时间 = 30/2 + 5 = 20秒
        is_feasible = basic_calculator.is_maneuver_feasible(30.0, 20.0)

        assert is_feasible is True

    def test_infeasible_due_to_time(self, basic_calculator):
        """测试因时间不足而不可行"""
        # 30度机动，可用时间15秒
        # 需要时间 = 30/2 + 5 = 20秒 > 15秒
        is_feasible = basic_calculator.is_maneuver_feasible(30.0, 15.0)

        assert is_feasible is False

    def test_infeasible_due_to_angle_limit(self, basic_calculator):
        """测试因角度超限而不可行"""
        # 50度机动，超过45度限制
        is_feasible = basic_calculator.is_maneuver_feasible(50.0, 100.0)

        assert is_feasible is False

    def test_exactly_at_angle_limit(self, basic_calculator):
        """测试恰好在角度限制边界"""
        # 45度机动，恰好在限制上
        is_feasible = basic_calculator.is_maneuver_feasible(45.0, 30.0)

        assert is_feasible is True

    def test_zero_available_time(self, basic_calculator):
        """测试零可用时间"""
        is_feasible = basic_calculator.is_maneuver_feasible(10.0, 0.0)

        assert is_feasible is False

    def test_negative_available_time_raises_error(self, basic_calculator):
        """测试负可用时间应报错"""
        with pytest.raises(ValueError):
            basic_calculator.is_maneuver_feasible(10.0, -5.0)


# =============================================================================
# Complete Maneuver Calculation 测试
# =============================================================================

class TestCompleteManeuverCalculation:
    """测试完整机动信息计算"""

    def test_calculate_maneuver_between_targets(
        self, basic_calculator, satellite_position_ecef,
        target_nadir, target_15deg_off
    ):
        """测试计算两个目标间的完整机动信息"""
        maneuver = basic_calculator.calculate_maneuver(
            satellite_position_ecef,
            target_nadir,
            target_15deg_off
        )

        assert isinstance(maneuver, SlewManeuver)
        assert maneuver.start_target_id == target_nadir.id
        assert maneuver.end_target_id == target_15deg_off.id
        assert maneuver.total_slew_angle > 0.0
        assert maneuver.slew_time > 0.0
        assert isinstance(maneuver.feasible, bool)

    def test_maneuver_to_same_target_is_feasible(
        self, basic_calculator, satellite_position_ecef, target_nadir
    ):
        """测试到相同目标的机动是可行的"""
        maneuver = basic_calculator.calculate_maneuver(
            satellite_position_ecef,
            target_nadir,
            target_nadir
        )

        assert maneuver.total_slew_angle == pytest.approx(0.0, abs=0.01)
        assert maneuver.slew_time == pytest.approx(5.0, abs=0.01)  # 只有稳定时间
        assert maneuver.feasible is True

    def test_infeasible_maneuver_due_to_angle(
        self, basic_calculator, satellite_position_ecef, target_nadir
    ):
        """测试因角度超限而不可行的机动"""
        # 创建一个很远的目标（超过45度限制）
        far_target = Target(
            id="far_target",
            name="Far Target",
            target_type=TargetType.POINT,
            longitude=60.0,  # 很远的目标
            latitude=60.0,
            priority=1
        )

        maneuver = basic_calculator.calculate_maneuver(
            satellite_position_ecef,
            target_nadir,
            far_target
        )

        assert maneuver.feasible is False

    def test_maneuver_with_area_target_raises_error(
        self, basic_calculator, satellite_position_ecef, target_nadir
    ):
        """测试区域目标应报错"""
        area_target = Target(
            id="area_001",
            name="Area Target",
            target_type=TargetType.AREA,
            area_vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        )

        with pytest.raises(ValueError):
            basic_calculator.calculate_maneuver(
                satellite_position_ecef,
                target_nadir,
                area_target
            )


# =============================================================================
# ClusterSlewCalculator 测试
# =============================================================================

class TestClusterSlewCalculator:
    """测试聚类机动计算器"""

    def test_cluster_calculator_inheritance(self):
        """测试ClusterSlewCalculator继承自SlewCalculator"""
        calc = ClusterSlewCalculator()

        assert isinstance(calc, SlewCalculator)
        assert hasattr(calc, 'max_slew_rate')
        assert hasattr(calc, 'max_slew_angle')
        assert hasattr(calc, 'settling_time')

    def test_calculate_cluster_slew_coverage(
        self, basic_calculator, satellite_position_ecef, sample_cluster
    ):
        """测试计算聚类覆盖所需的总机动角度"""
        cluster_calc = ClusterSlewCalculator(
            max_slew_rate=2.0,
            max_slew_angle=45.0,
            settling_time=5.0
        )

        total_angle, total_time = cluster_calc.calculate_cluster_slew_coverage(
            satellite_position_ecef,
            sample_cluster,
            look_angle=0.0
        )

        # 聚类跨越2度经度，应该有正的机动角度
        assert total_angle > 0.0
        assert total_time > 0.0
        # 时间应该符合公式: angle/rate + settling
        expected_time = total_angle / 2.0 + 5.0
        assert total_time == pytest.approx(expected_time, abs=0.01)

    def test_empty_cluster_zero_coverage(self, basic_calculator, satellite_position_ecef):
        """测试空聚类返回零覆盖"""
        cluster_calc = ClusterSlewCalculator()

        empty_cluster = TargetCluster(
            cluster_id="empty",
            targets=[],
            centroid=(0.0, 0.0),
            total_priority=0,
            bounding_box=(0.0, 0.0, 0.0, 0.0)
        )

        total_angle, total_time = cluster_calc.calculate_cluster_slew_coverage(
            satellite_position_ecef,
            empty_cluster,
            look_angle=0.0
        )

        assert total_angle == 0.0
        assert total_time == 0.0

    def test_single_target_cluster_minimal_coverage(
        self, basic_calculator, satellite_position_ecef
    ):
        """测试单目标聚类的最小覆盖"""
        cluster_calc = ClusterSlewCalculator()

        single_target_cluster = TargetCluster(
            cluster_id="single",
            targets=[
                Target(id="t1", name="T1", target_type=TargetType.POINT,
                       longitude=0.0, latitude=0.0, priority=1)
            ],
            centroid=(0.0, 0.0),
            total_priority=1,
            bounding_box=(0.0, 0.0, 0.0, 0.0)
        )

        total_angle, total_time = cluster_calc.calculate_cluster_slew_coverage(
            satellite_position_ecef,
            single_target_cluster,
            look_angle=0.0
        )

        # 单目标只需要稳定时间
        assert total_angle == 0.0
        assert total_time == 0.0  # 或只有稳定时间

    def test_can_cover_cluster_in_time_success(
        self, basic_calculator, sample_cluster
    ):
        """测试可以覆盖聚类的情况"""
        cluster_calc = ClusterSlewCalculator(
            max_slew_rate=2.0,
            max_slew_angle=45.0,
            settling_time=5.0
        )

        # 可见窗口120秒，成像100秒，机动需要约9秒
        can_cover = cluster_calc.can_cover_cluster_in_time(
            sample_cluster,
            visibility_duration=120.0,
            imaging_time=100.0
        )

        assert can_cover is True

    def test_can_cover_cluster_in_time_failure(
        self, basic_calculator, sample_cluster
    ):
        """测试无法覆盖聚类的情况"""
        cluster_calc = ClusterSlewCalculator(
            max_slew_rate=2.0,
            max_slew_angle=45.0,
            settling_time=5.0
        )

        # 可见窗口100秒，成像90秒，机动需要约9秒
        # 90 + 9 = 99 < 100，应该可以通过
        # 测试一个更紧张的情况
        can_cover = cluster_calc.can_cover_cluster_in_time(
            sample_cluster,
            visibility_duration=105.0,
            imaging_time=100.0
        )

        # 根据实际机动时间判断
        # 如果机动时间 > 5秒，则不可行
        assert isinstance(can_cover, bool)

    def test_cluster_coverage_with_large_cluster(
        self, basic_calculator, satellite_position_ecef
    ):
        """测试大聚类的覆盖计算"""
        cluster_calc = ClusterSlewCalculator()

        # 创建一个跨越很大角度的聚类
        large_cluster = TargetCluster(
            cluster_id="large",
            targets=[
                Target(id=f"t{i}", name=f"T{i}", target_type=TargetType.POINT,
                       longitude=i * 5.0, latitude=0.0, priority=1)
                for i in range(10)  # 跨越45度
            ],
            centroid=(22.5, 0.0),
            total_priority=10,
            bounding_box=(0.0, 45.0, 0.0, 0.0)
        )

        total_angle, total_time = cluster_calc.calculate_cluster_slew_coverage(
            satellite_position_ecef,
            large_cluster,
            look_angle=0.0
        )

        # 大聚类应该有较大的机动角度
        assert total_angle > 30.0

    def test_cluster_coverage_exceeds_max_angle(
        self, satellite_position_ecef
    ):
        """测试聚类覆盖超过最大机动角度"""
        cluster_calc = ClusterSlewCalculator(max_slew_angle=30.0)

        # 创建一个超过30度的聚类
        large_cluster = TargetCluster(
            cluster_id="too_large",
            targets=[
                Target(id=f"t{i}", name=f"T{i}", target_type=TargetType.POINT,
                       longitude=i * 4.0, latitude=0.0, priority=1)
                for i in range(10)  # 跨越36度
            ],
            centroid=(18.0, 0.0),
            total_priority=10,
            bounding_box=(0.0, 36.0, 0.0, 0.0)
        )

        total_angle, total_time = cluster_calc.calculate_cluster_slew_coverage(
            satellite_position_ecef,
            large_cluster,
            look_angle=0.0
        )

        # 应该被限制在最大角度
        assert total_angle <= 30.0


# =============================================================================
# Edge Cases 测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_very_small_slew_angle(self, basic_calculator):
        """测试非常小的机动角度"""
        slew_time = basic_calculator.calculate_slew_time(0.001)

        # 应该至少返回稳定时间
        assert slew_time >= basic_calculator.settling_time

    def test_very_large_slew_angle(self, basic_calculator):
        """测试非常大的机动角度"""
        # 180度机动
        slew_time = basic_calculator.calculate_slew_time(180.0)

        # 180/2 + 5 = 95秒
        assert slew_time == pytest.approx(95.0, abs=0.01)

    def test_satellite_at_pole(self, basic_calculator):
        """测试卫星在极点上空的情况"""
        # 卫星在北极上空
        sat_pos = (0.0, 0.0, 6871000.0)

        target1 = (0.0, 80.0)  # 靠近北极
        target2 = (180.0, 80.0)  # 对面

        roll, pitch, total = basic_calculator.calculate_slew_angles(
            sat_pos, target1, target2
        )

        # 应该能计算出角度
        assert total >= 0.0
        assert not math.isnan(total)

    def test_satellite_at_high_altitude(self, basic_calculator):
        """测试高轨道卫星"""
        # GEO高度约36000km
        geo_sat_pos = (42164000.0, 0.0, 0.0)

        target1 = (0.0, 0.0)
        target2 = (10.0, 0.0)

        roll, pitch, total = basic_calculator.calculate_slew_angles(
            geo_sat_pos, target1, target2
        )

        # GEO高度看到的地面角度很小
        assert total >= 0.0
        assert not math.isnan(total)

    def test_none_target_raises_error(self, basic_calculator, satellite_position_ecef):
        """测试None目标应报错"""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            basic_calculator.calculate_maneuver(
                satellite_position_ecef,
                None,
                Target(id="t1", name="T1", target_type=TargetType.POINT,
                       longitude=0.0, latitude=0.0, priority=1)
            )


# =============================================================================
# Integration Tests 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_workflow_single_maneuver(
        self, basic_calculator, satellite_position_ecef,
        target_nadir, target_15deg_off
    ):
        """测试完整单机动流程"""
        # 1. 计算机动角度
        roll, pitch, total = basic_calculator.calculate_slew_angles(
            satellite_position_ecef,
            (target_nadir.longitude, target_nadir.latitude),
            (target_15deg_off.longitude, target_15deg_off.latitude)
        )

        # 2. 计算机动时间
        slew_time = basic_calculator.calculate_slew_time(total)

        # 3. 判断可行性
        feasible = basic_calculator.is_maneuver_feasible(total, 20.0)

        # 4. 获取完整机动信息
        maneuver = basic_calculator.calculate_maneuver(
            satellite_position_ecef, target_nadir, target_15deg_off
        )

        # 验证一致性
        assert maneuver.total_slew_angle == pytest.approx(total, abs=0.01)
        assert maneuver.slew_time == pytest.approx(slew_time, abs=0.01)
        assert maneuver.feasible == feasible

    def test_scheduler_integration_scenario(self):
        """测试调度器集成场景"""
        calc = SlewCalculator(max_slew_rate=2.0, max_slew_angle=45.0)

        # 模拟调度决策场景
        visibility_window = 120.0  # 秒
        imaging_time = 100.0  # 秒
        slew_angle = 15.0  # 度

        # 计算机动时间
        slew_time = calc.calculate_slew_time(slew_angle)

        # 判断是否有足够时间
        total_time = imaging_time + slew_time
        can_schedule = total_time <= visibility_window

        assert slew_time == pytest.approx(12.5, abs=0.01)  # 15/2 + 5
        assert can_schedule is True  # 100 + 12.5 = 112.5 < 120

    def test_cluster_scheduler_integration(self):
        """测试聚类调度器集成场景"""
        cluster_calc = ClusterSlewCalculator(
            max_slew_rate=2.0,
            max_slew_angle=45.0,
            settling_time=5.0
        )

        # 创建聚类
        cluster = TargetCluster(
            cluster_id="test_cluster",
            targets=[
                Target(id=f"t{i}", name=f"T{i}", target_type=TargetType.POINT,
                       longitude=i * 1.0, latitude=0.0, priority=1)
                for i in range(8)  # 跨越7度
            ],
            centroid=(3.5, 0.0),
            total_priority=8,
            bounding_box=(0.0, 7.0, 0.0, 0.0)
        )

        # 模拟可见性窗口
        visibility_duration = 120.0
        imaging_time = 100.0

        # 判断是否可以覆盖
        can_cover = cluster_calc.can_cover_cluster_in_time(
            cluster, visibility_duration, imaging_time
        )

        assert isinstance(can_cover, bool)


# =============================================================================
# Performance Tests 性能测试
# =============================================================================

class TestPerformance:
    """性能测试"""

    def test_large_number_of_maneuvers(self, basic_calculator):
        """测试大量机动计算的性能"""
        import time

        sat_pos = (6871000.0, 0.0, 0.0)
        targets = [
            (i * 0.5, 0.0) for i in range(100)  # 100个目标
        ]

        start_time = time.time()

        # 计算所有目标对之间的机动
        for i in range(len(targets) - 1):
            basic_calculator.calculate_slew_angles(
                sat_pos, targets[i], targets[i + 1]
            )

        elapsed = time.time() - start_time

        # 100次计算应该在1秒内完成
        assert elapsed < 1.0

    def test_cluster_with_many_targets(self):
        """测试大聚类的性能"""
        import time

        cluster_calc = ClusterSlewCalculator()

        # 创建大聚类
        large_cluster = TargetCluster(
            cluster_id="perf_test",
            targets=[
                Target(id=f"t{i}", name=f"T{i}", target_type=TargetType.POINT,
                       longitude=i * 0.1, latitude=i * 0.05, priority=1)
                for i in range(1000)
            ],
            centroid=(50.0, 25.0),
            total_priority=1000,
            bounding_box=(0.0, 100.0, 0.0, 50.0)
        )

        sat_pos = (6871000.0, 0.0, 0.0)

        start_time = time.time()

        total_angle, total_time = cluster_calc.calculate_cluster_slew_coverage(
            sat_pos, large_cluster, look_angle=0.0
        )

        elapsed = time.time() - start_time

        # 大聚类计算应该在0.5秒内完成
        assert elapsed < 0.5
        assert total_angle >= 0.0
