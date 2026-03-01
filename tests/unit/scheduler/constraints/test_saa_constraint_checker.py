"""
SAA 约束检查器单元测试

TDD 流程：先写测试，再实现
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from scheduler.constraints.saa_constraint_checker import (
    SAAConstraintChecker,
    SAAFeasibilityResult,
)
from core.models.mission import Mission
from core.models.satellite import Satellite


class TestSAAFeasibilityResult:
    """测试 SAAFeasibilityResult 数据类"""

    def test_default_values(self):
        """测试默认值"""
        result = SAAFeasibilityResult(feasible=True)

        assert result.feasible is True
        assert result.violation_count == 0
        assert result.violation_times == []
        assert result.sample_count == 0
        assert result.max_separation == 0.0

    def test_with_violations(self):
        """测试有违规的情况"""
        now = datetime.utcnow()
        result = SAAFeasibilityResult(
            feasible=False,
            violation_count=2,
            violation_times=[now, now + timedelta(seconds=30)],
            sample_count=5,
            max_separation=1.5,
        )

        assert result.feasible is False
        assert result.violation_count == 2
        assert len(result.violation_times) == 2
        assert result.sample_count == 5
        assert result.max_separation == 1.5


class TestSAAConstraintCheckerInitialization:
    """测试 SAA 约束检查器初始化"""

    def test_init_with_mission(self):
        """测试使用 mission 初始化"""
        mission = Mock(spec=Mission)
        checker = SAAConstraintChecker(mission)

        assert checker.mission is mission
        assert checker._attitude_calc is not None
        assert checker._position_cache == {}

    def test_init_with_custom_attitude_calculator(self):
        """测试使用自定义姿态计算器初始化"""
        mission = Mock(spec=Mission)
        attitude_calc = Mock()
        checker = SAAConstraintChecker(mission, attitude_calc)

        assert checker._attitude_calc is attitude_calc

    def test_uses_saa_boundary_model(self):
        """测试使用 SAABoundaryModel 进行 SAA 检查"""
        from core.models.saa_boundary import SAABoundaryModel

        mission = Mock(spec=Mission)
        # 使用自定义 SAA 模型
        custom_model = SAABoundaryModel(center_lon=-50.0, center_lat=-30.0, semi_major=35.0, semi_minor=25.0)
        checker = SAAConstraintChecker(mission, saa_model=custom_model)

        # 验证使用了自定义模型
        assert checker._saa_model is custom_model
        assert checker._saa_model.center_lon == -50.0


class TestSAAConstraintCheckerGenerateSampleTimes:
    """测试采样时间生成"""

    @pytest.fixture
    def checker(self):
        mission = Mock(spec=Mission)
        return SAAConstraintChecker(mission)

    def test_short_window_minimum_samples(self, checker):
        """测试短窗口至少有最小采样点数"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 5)  # 5 秒窗口

        samples = checker._generate_sample_times(start, end, min_samples=3, sample_interval=timedelta(seconds=60))

        assert len(samples) >= 3
        assert samples[0] == start
        assert samples[-1] == end

    def test_long_window_interval_respected(self, checker):
        """测试长窗口按间隔采样"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 5, 0)  # 5 分钟窗口

        samples = checker._generate_sample_times(start, end, min_samples=3, sample_interval=timedelta(seconds=60))

        # 5分钟窗口，60秒间隔，应该有约 5+1=6 个采样点
        assert len(samples) >= 5
        assert samples[0] == start
        assert samples[-1] == end

    def test_samples_sorted_and_unique(self, checker):
        """测试采样点有序且不重复"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 0)  # 0 长度窗口

        samples = checker._generate_sample_times(start, end, min_samples=3, sample_interval=timedelta(seconds=60))

        # 起止相同，应该只有一个点
        assert len(samples) == 1
        assert samples[0] == start

    def test_even_distribution(self, checker):
        """测试采样点均匀分布"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 2, 0)  # 2 分钟窗口

        samples = checker._generate_sample_times(start, end, min_samples=5, sample_interval=timedelta(seconds=30))

        # 检查均匀分布
        intervals = [(samples[i+1] - samples[i]).total_seconds() for i in range(len(samples)-1)]
        if len(intervals) > 1:
            # 间隔应该大致相等
            avg_interval = sum(intervals) / len(intervals)
            for interval in intervals:
                assert abs(interval - avg_interval) < 5  # 容差 5 秒


class TestSAAConstraintCheckerIsInSAA:
    """测试 SAA 区域判断"""

    @pytest.fixture
    def checker(self):
        mission = Mock(spec=Mission)
        return SAAConstraintChecker(mission)

    def test_center_in_saa(self, checker):
        """测试中心点在 SAA 内"""
        in_saa, separation = checker._is_in_saa(-45.0, -25.0)
        assert in_saa is True
        assert separation == pytest.approx(0.0, abs=1e-10)

    def test_boundary_in_saa(self, checker):
        """测试边界上的点在 SAA 内"""
        # 东边界
        in_saa, separation = checker._is_in_saa(-5.0, -25.0)  # -45 + 40
        assert in_saa is True
        assert separation == pytest.approx(1.0, abs=1e-6)

    def test_outside_saa(self, checker):
        """测试 SAA 外的点"""
        # 中国
        in_saa, separation = checker._is_in_saa(116.0, 40.0)
        assert in_saa is False
        assert separation > 1.0

        # 欧洲
        in_saa, separation = checker._is_in_saa(10.0, 50.0)
        assert in_saa is False

    def test_separation_calculation(self, checker):
        """测试偏离距离计算"""
        # 中心点偏离为 0
        _, sep1 = checker._is_in_saa(-45.0, -25.0)
        assert sep1 == pytest.approx(0.0, abs=1e-10)

        # 半轴处的偏离为 1
        _, sep2 = checker._is_in_saa(-45.0, 5.0)  # 北边界
        assert sep2 == pytest.approx(1.0, abs=1e-6)

        # 两倍距离的偏离约为 2
        _, sep3 = checker._is_in_saa(-45.0, 35.0)  # 北边界外 30 度
        assert sep3 == pytest.approx(2.0, abs=1e-6)


class TestSAAConstraintCheckerECEFToGeodetic:
    """测试 ECEF 到地理坐标转换"""

    @pytest.fixture
    def checker(self):
        mission = Mock(spec=Mission)
        return SAAConstraintChecker(mission)

    def test_equator_prime_meridian(self, checker):
        """测试赤道与本初子午线交点"""
        # (R, 0, 0) 应该在 (0, 0, 0)
        lon, lat, alt = checker._ecef_to_geodetic(checker.EARTH_RADIUS, 0.0, 0.0)

        assert lon == pytest.approx(0.0, abs=1e-6)
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert alt == pytest.approx(0.0, abs=1e-3)

    def test_north_pole(self, checker):
        """测试北极点"""
        # (0, 0, R) 应该在 (任意经度, 90, 0)
        lon, lat, alt = checker._ecef_to_geodetic(0.0, 0.0, checker.EARTH_RADIUS)

        assert lat == pytest.approx(90.0, abs=1e-6)
        assert alt == pytest.approx(0.0, abs=1e-3)

    def test_south_pole(self, checker):
        """测试南极点"""
        lon, lat, alt = checker._ecef_to_geodetic(0.0, 0.0, -checker.EARTH_RADIUS)

        assert lat == pytest.approx(-90.0, abs=1e-6)
        assert alt == pytest.approx(0.0, abs=1e-3)

    def test_known_location_brazil(self, checker):
        """测试已知位置（巴西附近）"""
        # 巴西利亚: 西经 47.9°, 南纬 15.8°
        # 转换为 ECEF 再转回
        import math
        lon_deg = -47.9
        lat_deg = -15.8
        lon_rad = math.radians(lon_deg)
        lat_rad = math.radians(lat_deg)
        r = checker.EARTH_RADIUS

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        lon_out, lat_out, alt_out = checker._ecef_to_geodetic(x, y, z)

        assert lon_out == pytest.approx(lon_deg, abs=1e-4)
        assert lat_out == pytest.approx(lat_deg, abs=1e-4)
        assert alt_out == pytest.approx(0.0, abs=1e-3)

    def test_with_altitude(self, checker):
        """测试带高度的坐标"""
        import math
        lon_deg = 0.0
        lat_deg = 0.0
        alt_m = 100000.0  # 100 km

        lon_rad = math.radians(lon_deg)
        lat_rad = math.radians(lat_deg)
        r = checker.EARTH_RADIUS + alt_m

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        lon_out, lat_out, alt_out = checker._ecef_to_geodetic(x, y, z)

        assert lon_out == pytest.approx(lon_deg, abs=1e-6)
        assert lat_out == pytest.approx(lat_deg, abs=1e-6)
        assert alt_out == pytest.approx(alt_m, abs=1e-1)


class TestSAAConstraintCheckerSingleTime:
    """测试单时间点检查"""

    @pytest.fixture
    def checker(self):
        mission = Mock(spec=Mission)
        checker = SAAConstraintChecker(mission)

        # 模拟卫星
        satellite = Mock(spec=Satellite)
        satellite.id = "SAT-001"
        mission.get_satellite_by_id.return_value = satellite

        return checker, mission, satellite

    def test_satellite_in_saa(self, checker):
        """测试卫星在 SAA 内的情况"""
        checker_obj, mission, satellite = checker

        # 模拟卫星在 SAA 中心位置
        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=(-45.0, -25.0)
        ):
            in_saa, separation = checker_obj.check_single_time("SAT-001", datetime.utcnow())

            assert in_saa is True
            assert separation == pytest.approx(0.0, abs=1e-10)

    def test_satellite_outside_saa(self, checker):
        """测试卫星在 SAA 外的情况"""
        checker_obj, mission, satellite = checker

        # 模拟卫星在中国上空
        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=(116.0, 40.0)
        ):
            in_saa, separation = checker_obj.check_single_time("SAT-001", datetime.utcnow())

            assert in_saa is False
            assert separation > 1.0

    def test_satellite_not_found(self, checker):
        """测试找不到卫星的情况 - 安全优先：视为在 SAA 中"""
        checker_obj, mission, satellite = checker
        mission.get_satellite_by_id.return_value = None

        in_saa, separation = checker_obj.check_single_time("NON-EXISTENT", datetime.utcnow())

        # 安全优先：找不到卫星视为在 SAA 中（阻止任务调度）
        assert in_saa is True
        assert separation == float('inf')

    def test_position_calculation_failed(self, checker):
        """测试位置计算失败的情况"""
        checker_obj, mission, satellite = checker

        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=None
        ):
            in_saa, separation = checker_obj.check_single_time("SAT-001", datetime.utcnow())

            # 计算失败视为不在 SAA 中（保守处理）
            assert in_saa is False
            assert separation == 0.0


class TestSAAConstraintCheckerWindowFeasibility:
    """测试窗口可行性检查"""

    @pytest.fixture
    def checker(self):
        mission = Mock(spec=Mission)
        checker = SAAConstraintChecker(mission)

        satellite = Mock(spec=Satellite)
        satellite.id = "SAT-001"
        mission.get_satellite_by_id.return_value = satellite

        return checker, mission, satellite

    def test_window_all_outside_saa(self, checker):
        """测试窗口全部在 SAA 外（可行）"""
        checker_obj, mission, satellite = checker

        # 模拟卫星始终在中国上空
        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=(116.0, 40.0)
        ):
            result = checker_obj.check_window_feasibility(
                "SAT-001",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 1, 0),
                min_samples=3,
                sample_interval=timedelta(seconds=30),
            )

            assert result.feasible is True
            assert result.violation_count == 0
            assert len(result.violation_times) == 0
            assert result.sample_count >= 3

    def test_window_partially_in_saa(self, checker):
        """测试窗口部分在 SAA 内（不可行）"""
        checker_obj, mission, satellite = checker

        # 模拟卫星路径：从 SAA 外进入 SAA 内
        call_count = [0]
        def mock_subpoint(sat, dt):
            call_count[0] += 1
            if call_count[0] <= 2:
                return (116.0, 40.0)  # 中国，SAA 外
            else:
                return (-45.0, -25.0)  # SAA 中心

        with patch.object(checker_obj, '_get_satellite_subpoint', side_effect=mock_subpoint):
            result = checker_obj.check_window_feasibility(
                "SAT-001",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 1, 0),
                min_samples=4,
                sample_interval=timedelta(seconds=20),
            )

            assert result.feasible is False
            assert result.violation_count > 0
            assert len(result.violation_times) > 0

    def test_window_all_in_saa(self, checker):
        """测试窗口全部在 SAA 内（不可行）"""
        checker_obj, mission, satellite = checker

        # 模拟卫星始终在 SAA 内
        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=(-45.0, -25.0)
        ):
            result = checker_obj.check_window_feasibility(
                "SAT-001",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 1, 0),
                min_samples=3,
                sample_interval=timedelta(seconds=30),
            )

            assert result.feasible is False
            assert result.violation_count == result.sample_count

    def test_satellite_not_found_window_check(self, checker):
        """测试窗口检查时找不到卫星 - 安全优先"""
        checker_obj, mission, satellite = checker
        mission.get_satellite_by_id.return_value = None

        result = checker_obj.check_window_feasibility(
            "NON-EXISTENT",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 1, 0),
        )

        # 安全优先：找不到卫星视为不可行
        assert result.feasible is False
        assert result.violation_count == 1

    def test_max_separation_tracking(self, checker):
        """测试最大偏离距离跟踪"""
        checker_obj, mission, satellite = checker

        # 模拟逐渐接近 SAA
        positions = [
            (116.0, 40.0),   # 远离
            (-20.0, -10.0),  # 接近
            (-45.0, -25.0),  # 中心
        ]
        position_iter = iter(positions)

        with patch.object(
            checker_obj, '_get_satellite_subpoint', side_effect=lambda sat, dt: next(position_iter, positions[-1])
        ):
            result = checker_obj.check_window_feasibility(
                "SAT-001",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 0, 6),
                min_samples=3,
                sample_interval=timedelta(seconds=2),
            )

            assert result.max_separation >= 0.0


class TestSAAConstraintCheckerEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def checker(self):
        mission = Mock(spec=Mission)
        checker = SAAConstraintChecker(mission)

        satellite = Mock(spec=Satellite)
        satellite.id = "SAT-001"
        mission.get_satellite_by_id.return_value = satellite

        return checker, mission, satellite

    def test_longitude_wraparound_near_180(self, checker):
        """测试经度环绕 -180/180 附近"""
        checker_obj, mission, satellite = checker

        # 模拟卫星在 -179° 经度（接近 180°）
        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=(-179.0, -25.0)
        ):
            in_saa, separation = checker_obj.check_single_time("SAT-001", datetime.utcnow())
            # -179° 经度远离 SAA 中心（-45°），应该不在 SAA 内
            assert in_saa is False

        # 模拟卫星在 179° 经度（接近 180°，另一侧）
        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=(179.0, -25.0)
        ):
            in_saa, separation = checker_obj.check_single_time("SAT-001", datetime.utcnow())
            # 179° 经度远离 SAA 中心（-45°），应该不在 SAA 内
            assert in_saa is False

    def test_cache_management(self, checker):
        """测试位置缓存管理"""
        checker_obj, mission, satellite = checker

        # 初始化缓存
        checker_obj.initialize_satellite(satellite)
        assert satellite.id in checker_obj._position_cache
        assert checker_obj._position_cache[satellite.id] == {}

        # 模拟多次检查，验证缓存被使用
        call_count = 0
        def mock_get_state(sat, dt):
            nonlocal call_count
            call_count += 1
            # 返回固定位置（ECEF坐标）
            return ((6371000.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        with patch.object(checker_obj._attitude_calc, '_get_satellite_state', side_effect=mock_get_state):
            # 第一次调用
            checker_obj._get_satellite_subpoint(satellite, datetime(2024, 1, 1, 12, 0, 0))
            assert call_count == 1

            # 相同时间第二次调用 - 应该从缓存获取
            checker_obj._get_satellite_subpoint(satellite, datetime(2024, 1, 1, 12, 0, 0))
            assert call_count == 1  # 不应该增加

            # 接近时间（±1秒）也应该使用缓存
            checker_obj._get_satellite_subpoint(satellite, datetime(2024, 1, 1, 12, 0, 1))
            assert call_count == 1  # 不应该增加

    def test_satellite_not_found_window_check(self, checker):
        """测试窗口检查时找不到卫星 - 安全优先"""
        checker_obj, mission, satellite = checker
        mission.get_satellite_by_id.return_value = None

        result = checker_obj.check_window_feasibility(
            "NON-EXISTENT",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 1, 0),
        )

        # 安全优先：找不到卫星视为不可行
        assert result.feasible is False
        assert result.violation_count == 1

    def test_position_calculation_failed(self, checker):
        """测试位置计算失败的情况"""
        checker_obj, mission, satellite = checker

        with patch.object(
            checker_obj, '_get_satellite_subpoint', return_value=None
        ):
            in_saa, separation = checker_obj.check_single_time("SAT-001", datetime.utcnow())

            # 计算失败视为不在 SAA 中（允许调度继续，但记录警告）
            assert in_saa is False
            assert separation == 0.0


class TestSAAConstraintCheckerIntegration:
    """集成测试"""

    @pytest.fixture
    def checker_with_real_calculator(self):
        """使用真实姿态计算器的检查器"""
        mission = Mock(spec=Mission)
        satellite = Mock(spec=Satellite)
        satellite.id = "SAT-001"
        satellite.tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        satellite.tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"
        mission.get_satellite_by_id.return_value = satellite

        checker = SAAConstraintChecker(mission)
        return checker, mission, satellite

    def test_real_satellite_position(self, checker_with_real_calculator):
        """测试使用真实卫星位置计算"""
        checker_obj, mission, satellite = checker_with_real_calculator

        # 获取一个真实位置
        subpoint = checker_obj._get_satellite_subpoint(satellite, datetime.utcnow())

        if subpoint:
            lon, lat = subpoint
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90

    def test_real_window_check(self, checker_with_real_calculator):
        """测试真实窗口检查"""
        checker_obj, mission, satellite = checker_with_real_calculator

        result = checker_obj.check_window_feasibility(
            "SAT-001",
            datetime.utcnow(),
            datetime.utcnow() + timedelta(minutes=5),
            min_samples=5,
            sample_interval=timedelta(minutes=1),
        )

        # 应该能完成检查并返回结果
        assert isinstance(result.feasible, bool)
        assert result.sample_count >= 5
