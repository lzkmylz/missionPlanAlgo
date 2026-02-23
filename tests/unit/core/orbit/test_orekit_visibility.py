"""
Orekit可见性计算器测试

TDD测试套件 - 测试OrekitVisibilityCalculator的完整功能
"""

import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.orbit.visibility.base import VisibilityWindow


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def calculator():
    """创建默认计算器实例"""
    return OrekitVisibilityCalculator()


@pytest.fixture
def custom_calculator():
    """创建自定义配置的计算器"""
    config = {
        'min_elevation': 10.0,
        'time_step': 30
    }
    return OrekitVisibilityCalculator(config=config)


@pytest.fixture
def mock_satellite():
    """创建模拟卫星"""
    sat = Mock()
    sat.id = "SAT-001"
    sat.name = "Test Satellite"
    sat.orbit.altitude = 500000.0  # 500km
    sat.orbit.inclination = 97.4
    sat.tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    sat.tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"
    return sat


@pytest.fixture
def mock_target():
    """创建模拟点目标"""
    target = Mock()
    target.id = "TARGET-001"
    target.name = "Beijing"
    target.target_type.value = "point"
    target.longitude = 116.4074
    target.latitude = 39.9042
    target.get_ecef_position.return_value = (
        -2171419.0, 4387557.0, 4070234.0  # Beijing ECEF (meters)
    )
    return target


@pytest.fixture
def mock_ground_station():
    """创建模拟地面站"""
    gs = Mock()
    gs.id = "GS-001"
    gs.name = "Beijing Ground Station"
    gs.longitude = 116.4074
    gs.latitude = 39.9042
    gs.altitude = 0.0
    gs.get_ecef_position.return_value = (
        -2171419.0, 4387557.0, 4070234.0  # Beijing ECEF (meters)
    )
    return gs


@pytest.fixture
def time_range():
    """创建测试时间范围"""
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 12, 0, 0)
    return start, end


# =============================================================================
# 初始化测试
# =============================================================================

class TestInitialization:
    """测试初始化功能"""

    def test_default_initialization(self, calculator):
        """测试默认初始化"""
        assert calculator.min_elevation == 5.0
        assert calculator.time_step == 60
        assert calculator.EARTH_RADIUS == 6371000.0

    def test_custom_initialization(self, custom_calculator):
        """测试自定义配置初始化"""
        assert custom_calculator.min_elevation == 10.0
        assert custom_calculator.time_step == 30

    def test_empty_config_initialization(self):
        """测试空配置初始化"""
        calc = OrekitVisibilityCalculator(config={})
        assert calc.min_elevation == 5.0
        assert calc.time_step == 60

    def test_none_config_initialization(self):
        """测试None配置初始化"""
        calc = OrekitVisibilityCalculator(config=None)
        assert calc.min_elevation == 5.0
        assert calc.time_step == 60


# =============================================================================
# 几何计算测试
# =============================================================================

class TestGeometricCalculations:
    """测试几何计算功能"""

    def test_lla_to_ecef_conversion(self, calculator):
        """测试LLA到ECEF坐标转换"""
        # 测试赤道上的点
        x, y, z = calculator._lla_to_ecef(0.0, 0.0, 0.0)
        assert abs(x - 6371000.0) < 1.0  # 地球半径
        assert abs(y) < 1.0
        assert abs(z) < 1.0

        # 测试北极点
        x, y, z = calculator._lla_to_ecef(0.0, 90.0, 0.0)
        assert abs(x) < 1.0
        assert abs(y) < 1.0
        assert abs(z - 6371000.0) < 1.0

        # 测试东经90度
        x, y, z = calculator._lla_to_ecef(90.0, 0.0, 0.0)
        assert abs(x) < 1.0
        assert abs(y - 6371000.0) < 1.0
        assert abs(z) < 1.0

    def test_earth_occlusion_check(self, calculator):
        """测试地球遮挡判断"""
        # 卫星在地球表面上方
        sat_pos = (7000000.0, 0.0, 0.0)  # 约100km高度
        target_pos = (6371000.0, 0.0, 0.0)  # 地球表面
        assert not calculator._is_earth_occluded(sat_pos, target_pos)

        # 卫星和点在地球两侧（被遮挡）
        sat_pos = (-7000000.0, 0.0, 0.0)
        target_pos = (6371000.0, 0.0, 0.0)
        assert calculator._is_earth_occluded(sat_pos, target_pos)

        # 卫星正好在目标正上方（无遮挡）
        sat_pos = (6371000.0, 0.0, 1000000.0)
        target_pos = (6371000.0, 0.0, 0.0)
        assert not calculator._is_earth_occluded(sat_pos, target_pos)

    def test_earth_occlusion_edge_cases(self, calculator):
        """测试地球遮挡边界情况"""
        # 同一位置（应该无遮挡）
        pos = (6371000.0, 0.0, 0.0)
        assert not calculator._is_earth_occluded(pos, pos)

        # 极低轨道卫星
        sat_pos = (6371100.0, 0.0, 0.0)  # 仅100m高度
        target_pos = (6371000.0, 0.0, 0.0)
        assert not calculator._is_earth_occluded(sat_pos, target_pos)

    def test_calculate_elevation_inherited(self, calculator):
        """测试继承的仰角计算"""
        # 卫星在地面点正上方（z轴方向）
        ground_pos = (6371000.0, 0.0, 0.0)  # 赤道表面
        sat_pos = (6371000.0, 0.0, 7000000.0)  # 正上方约1000km
        elevation = calculator._calculate_elevation(sat_pos, ground_pos)
        # 仰角应该在0到90度之间
        assert 0.0 <= elevation <= 90.0

        # 卫星在地平线上（x轴方向延伸）
        sat_pos = (13000000.0, 0.0, 0.0)  # 水平方向
        elevation = calculator._calculate_elevation(sat_pos, ground_pos)
        # 仰角应该在-90到90度之间
        assert -90.0 <= elevation <= 90.0


# =============================================================================
# 轨道传播测试
# =============================================================================

class TestOrbitPropagation:
    """测试轨道传播功能"""

    @patch('core.orbit.visibility.orekit_visibility.SGP4_AVAILABLE', True)
    @patch('core.orbit.visibility.orekit_visibility.Satrec')
    @patch('core.orbit.visibility.orekit_visibility.jday')
    def test_propagate_satellite_with_tle(self, mock_jday, mock_satrec, calculator, mock_satellite):
        """测试使用TLE传播卫星"""
        # 设置mock
        mock_jday.return_value = (2451545.0, 0.5)
        mock_satrec.twoline2rv.return_value = Mock()
        mock_satrec.twoline2rv.return_value.sgp4.return_value = (
            0, (7000.0, 0.0, 0.0), (0.0, 7.0, 0.0)
        )

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_satellite(mock_satellite, dt)

        assert pos is not None
        assert vel is not None
        assert len(pos) == 3
        assert len(vel) == 3

    def test_propagate_without_sgp4(self, calculator):
        """测试SGP4不可用时的传播（使用简化模型）"""
        # 创建一个简单的卫星对象（非mock）
        from unittest.mock import MagicMock
        sat = MagicMock()
        sat.orbit = MagicMock()
        sat.orbit.altitude = 500000.0
        sat.orbit.inclination = 97.4
        sat.orbit.raan = 0.0

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(sat, dt)

        assert pos is not None
        assert len(pos) == 3
        assert vel is not None
        assert len(vel) == 3

    def test_propagate_without_tle(self, calculator):
        """测试无TLE时的传播"""
        # 创建一个简单的卫星对象（非mock）
        from unittest.mock import MagicMock
        sat = MagicMock()
        sat.tle_line1 = None
        sat.tle_line2 = None
        sat.orbit = MagicMock()
        sat.orbit.altitude = 500000.0
        sat.orbit.inclination = 97.4
        sat.orbit.raan = 0.0

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_satellite(sat, dt)

        assert pos is not None
        assert len(pos) == 3

    def test_propagate_range(self, calculator, mock_satellite):
        """测试时间序列传播"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)
        time_step = timedelta(minutes=10)

        with patch.object(calculator, '_propagate_satellite') as mock_prop:
            mock_prop.return_value = ((7000.0, 0.0, 0.0), (0.0, 7.0, 0.0))
            positions = calculator._propagate_range(mock_satellite, start, end, time_step)

            assert len(positions) == 7  # 0, 10, 20, 30, 40, 50, 60 minutes
            for pos, vel, dt in positions:
                assert len(pos) == 3
                assert len(vel) == 3


# =============================================================================
# 可见性窗口计算测试
# =============================================================================

class TestVisibilityWindowComputation:
    """测试可见性窗口计算"""

    def test_compute_satellite_target_windows_empty(self, calculator, mock_satellite, mock_target, time_range):
        """测试卫星-目标窗口计算（无可见窗口）"""
        start, end = time_range

        # Mock传播返回地球另一侧的位置
        with patch.object(calculator, '_propagate_range') as mock_prop:
            mock_prop.return_value = [
                ((-7000.0, 0.0, 0.0), (0.0, 0.0, 0.0), start + timedelta(minutes=i))
                for i in range(10)
            ]

            windows = calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end
            )

            assert isinstance(windows, list)
            assert len(windows) == 0

    def test_compute_satellite_target_windows_with_visibility(self, calculator, mock_satellite, mock_target, time_range):
        """测试卫星-目标窗口计算（有可见窗口）"""
        start, end = time_range

        # Mock传播返回目标上方的位置
        with patch.object(calculator, '_propagate_range') as mock_prop:
            # 创建一系列位置，中间部分在目标上方
            positions = []
            base_time = start
            for i in range(20):
                dt = base_time + timedelta(minutes=i * 5)
                if 5 <= i <= 15:  # 中间10个点在目标上方
                    pos = (-2171419.0, 4387557.0, 10000000.0)  # 目标上方
                else:
                    pos = ((-7000.0, 0.0, 0.0))  # 地球另一侧
                positions.append((pos, (0.0, 0.0, 0.0), dt))

            mock_prop.return_value = positions

            windows = calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end
            )

            assert isinstance(windows, list)

    def test_compute_satellite_ground_station_windows(self, calculator, mock_satellite, mock_ground_station, time_range):
        """测试卫星-地面站窗口计算"""
        start, end = time_range

        with patch.object(calculator, '_propagate_range') as mock_prop:
            mock_prop.return_value = [
                ((-7000.0, 0.0, 0.0), (0.0, 0.0, 0.0), start + timedelta(minutes=i))
                for i in range(10)
            ]

            windows = calculator.compute_satellite_ground_station_windows(
                mock_satellite, mock_ground_station, start, end
            )

            assert isinstance(windows, list)

    def test_window_computation_with_custom_timestep(self, custom_calculator, mock_satellite, mock_target, time_range):
        """测试自定义时间步长的窗口计算"""
        start, end = time_range

        with patch.object(custom_calculator, '_propagate_range') as mock_prop:
            mock_prop.return_value = []

            custom_calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end, timedelta(seconds=30)
            )

            # 验证使用了正确的时间步长
            mock_prop.assert_called_once()
            call_args = mock_prop.call_args
            # call_args is a tuple (args, kwargs) - 检查位置参数
            # _propagate_range(satellite, start_time, end_time, time_step)
            assert call_args[0][3].total_seconds() == 30


# =============================================================================
# 简化接口测试
# =============================================================================

class TestSimplifiedInterface:
    """测试简化接口"""

    def test_calculate_windows(self, calculator, time_range):
        """测试calculate_windows简化接口 - 默认实现返回空列表"""
        start, end = time_range

        # 默认实现返回空列表（因为没有提供卫星/目标对象获取机制）
        windows = calculator.calculate_windows("SAT-001", "TARGET-001", start, end)

        assert isinstance(windows, list)
        assert len(windows) == 0  # 默认实现返回空列表

    def test_is_visible_true(self, calculator):
        """测试is_visible返回True"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(calculator, 'calculate_windows') as mock_calc:
            mock_calc.return_value = [
                VisibilityWindow(
                    satellite_id="SAT-001",
                    target_id="TARGET-001",
                    start_time=dt - timedelta(minutes=5),
                    end_time=dt + timedelta(minutes=5),
                    max_elevation=45.0,
                    quality_score=0.5
                )
            ]

            result = calculator.is_visible("SAT-001", "TARGET-001", dt)
            assert result is True

    def test_is_visible_false(self, calculator):
        """测试is_visible返回False"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(calculator, 'calculate_windows') as mock_calc:
            mock_calc.return_value = [
                VisibilityWindow(
                    satellite_id="SAT-001",
                    target_id="TARGET-001",
                    start_time=dt - timedelta(minutes=15),
                    end_time=dt - timedelta(minutes=10),
                    max_elevation=45.0,
                    quality_score=0.5
                )
            ]

            result = calculator.is_visible("SAT-001", "TARGET-001", dt)
            assert result is False

    def test_is_visible_no_windows(self, calculator):
        """测试is_visible无窗口时返回False"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(calculator, 'calculate_windows') as mock_calc:
            mock_calc.return_value = []

            result = calculator.is_visible("SAT-001", "TARGET-001", dt)
            assert result is False


# =============================================================================
# 边界条件和错误处理测试
# =============================================================================

class TestEdgeCasesAndErrors:
    """测试边界条件和错误处理"""

    def test_invalid_time_range(self, calculator, mock_satellite, mock_target):
        """测试无效时间范围"""
        start = datetime(2024, 1, 2, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 0, 0)

        windows = calculator.compute_satellite_target_windows(
            mock_satellite, mock_target, start, end
        )
        assert windows == []

    def test_zero_duration_time_range(self, calculator, mock_satellite, mock_target):
        """测试零持续时间"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = start

        windows = calculator.compute_satellite_target_windows(
            mock_satellite, mock_target, start, end
        )
        assert windows == []

    def test_very_small_time_step(self, calculator, mock_satellite, mock_target):
        """测试非常小的时间步长"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)
        time_step = timedelta(seconds=1)

        with patch.object(calculator, '_propagate_range') as mock_prop:
            mock_prop.return_value = []

            windows = calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end, time_step
            )
            assert isinstance(windows, list)

    def test_propagation_error_handling(self, calculator, mock_satellite, mock_target):
        """测试传播错误处理"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        with patch.object(calculator, '_propagate_range') as mock_prop:
            mock_prop.side_effect = Exception("Propagation error")

            windows = calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end
            )
            assert windows == []

    def test_null_satellite(self, calculator, mock_target):
        """测试空卫星对象 - 实现返回空列表而不是抛出异常"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        # 实现捕获所有异常并返回空列表
        windows = calculator.compute_satellite_target_windows(
            None, mock_target, start, end
        )
        assert windows == []

    def test_null_target(self, calculator, mock_satellite):
        """测试空目标对象 - 实现返回空列表而不是抛出异常"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        # 实现捕获所有异常并返回空列表
        windows = calculator.compute_satellite_target_windows(
            mock_satellite, None, start, end
        )
        assert windows == []


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能相关测试"""

    def test_large_time_range(self, calculator, mock_satellite, mock_target):
        """测试大时间范围"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 7, 0, 0, 0)  # 一周

        with patch.object(calculator, '_propagate_range') as mock_prop:
            # 返回大量数据点
            mock_prop.return_value = [
                ((7000.0, 0.0, 0.0), (0.0, 0.0, 0.0), start + timedelta(minutes=i))
                for i in range(0, 7 * 24 * 60, 60)  # 每天一个点
            ]

            windows = calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end, timedelta(hours=1)
            )
            assert isinstance(windows, list)


# =============================================================================
# 积分测试
# =============================================================================

class TestIntegration:
    """积分测试 - 测试完整流程"""

    def test_full_visibility_calculation_workflow(self, calculator):
        """测试完整的可见性计算流程"""
        # 创建真实模拟对象
        satellite = Mock()
        satellite.id = "SAT-TEST"
        satellite.tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        satellite.tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"

        target = Mock()
        target.id = "TARGET-TEST"
        target.get_ecef_position.return_value = (6371000.0, 0.0, 0.0)  # 赤道

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 2, 0, 0)

        # 使用简化的数值传播而不是SGP4
        with patch('core.orbit.visibility.orekit_visibility.SGP4_AVAILABLE', False):
            windows = calculator.compute_satellite_target_windows(
                satellite, target, start, end, timedelta(minutes=5)
            )

            assert isinstance(windows, list)


# =============================================================================
# 额外覆盖测试
# =============================================================================

class TestAdditionalCoverage:
    """额外测试以达到更高覆盖率"""

    def test_propagate_satellite_sgp4_error(self, calculator):
        """测试SGP4传播错误处理"""
        from unittest.mock import MagicMock, patch

        sat = MagicMock()
        sat.tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        sat.tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"
        sat.orbit = MagicMock()
        sat.orbit.altitude = 500000.0
        sat.orbit.inclination = 97.4
        sat.orbit.raan = 0.0

        dt = datetime(2024, 1, 1, 12, 0, 0)

        # Mock SGP4返回错误
        with patch('core.orbit.visibility.orekit_visibility.SGP4_AVAILABLE', True):
            with patch('core.orbit.visibility.orekit_visibility.Satrec') as mock_satrec:
                mock_satrec.twoline2rv.return_value.sgp4.return_value = (1, None, None)  # 错误代码
                pos, vel = calculator._propagate_satellite(sat, dt)

                assert pos is not None
                assert len(pos) == 3

    def test_target_without_get_ecef_position(self, calculator, mock_satellite, time_range):
        """测试目标没有get_ecef_position方法的情况"""
        start, end = time_range

        # 创建没有get_ecef_position的目标
        target = Mock()
        target.id = "TARGET-NO-ECEF"
        target.longitude = 116.4074
        target.latitude = 39.9042
        target.altitude = 0.0
        del target.get_ecef_position  # 删除该方法

        with patch.object(calculator, '_propagate_range') as mock_prop:
            mock_prop.return_value = [
                ((-7000.0, 0.0, 0.0), (0.0, 0.0, 0.0), start + timedelta(minutes=i))
                for i in range(10)
            ]

            windows = calculator.compute_satellite_target_windows(
                mock_satellite, target, start, end
            )

            assert isinstance(windows, list)

    def test_ground_station_without_get_ecef_position(self, calculator, mock_satellite, time_range):
        """测试地面站没有get_ecef_position方法的情况"""
        start, end = time_range

        # 创建没有get_ecef_position的地面站
        gs = Mock()
        gs.id = "GS-NO-ECEF"
        gs.longitude = 116.4074
        gs.latitude = 39.9042
        gs.altitude = 0.0
        del gs.get_ecef_position  # 删除该方法

        with patch.object(calculator, '_propagate_range') as mock_prop:
            mock_prop.return_value = [
                ((-7000.0, 0.0, 0.0), (0.0, 0.0, 0.0), start + timedelta(minutes=i))
                for i in range(10)
            ]

            windows = calculator.compute_satellite_ground_station_windows(
                mock_satellite, gs, start, end
            )

            assert isinstance(windows, list)

    def test_satellite_with_none_orbit(self, calculator):
        """测试卫星orbit为None的情况"""
        from unittest.mock import MagicMock

        sat = MagicMock()
        sat.orbit = None

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(sat, dt)

        assert pos is not None
        assert len(pos) == 3
        assert vel is not None
        assert len(vel) == 3

    def test_is_earth_occluded_same_position(self, calculator):
        """测试地球遮挡判断 - 相同位置"""
        pos = (6371000.0, 0.0, 0.0)
        # 当位置相同时应该返回False（无遮挡）
        result = calculator._is_earth_occluded(pos, pos)
        assert result is False

    def test_is_earth_occluded_zero_distance(self, calculator):
        """测试地球遮挡判断 - 零距离向量"""
        sat_pos = (6371000.0, 0.0, 0.0)
        target_pos = (6371000.0, 0.0, 0.0)
        # 当距离为0时应该返回False
        result = calculator._is_earth_occluded(sat_pos, target_pos)
        assert result is False

    def test_compute_satellite_target_windows_exception(self, calculator, mock_satellite, time_range):
        """测试卫星-目标窗口计算的异常处理"""
        start, end = time_range

        # 创建一个会引发异常的目标
        bad_target = Mock()
        bad_target.id = "BAD-TARGET"
        bad_target.get_ecef_position.side_effect = Exception("ECEF error")

        windows = calculator.compute_satellite_target_windows(
            mock_satellite, bad_target, start, end
        )
        assert windows == []

    def test_compute_satellite_ground_station_windows_exception(self, calculator, mock_satellite, time_range):
        """测试卫星-地面站窗口计算的异常处理"""
        start, end = time_range

        # 创建一个会引发异常的地面站
        bad_gs = Mock()
        bad_gs.id = "BAD-GS"
        bad_gs.get_ecef_position.side_effect = Exception("ECEF error")

        windows = calculator.compute_satellite_ground_station_windows(
            mock_satellite, bad_gs, start, end
        )
        assert windows == []

    def test_propagate_range_exception_handling(self, calculator, mock_satellite):
        """测试传播范围中的异常处理"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 50, 0)  # 50分钟，产生5个点

        with patch.object(calculator, '_propagate_satellite') as mock_prop:
            # 前几个点成功，后面的失败
            call_count = [0]
            def side_effect(sat, dt):
                call_count[0] += 1
                if call_count[0] <= 3:  # 前3次成功
                    return ((7000.0, 0.0, 0.0), (0.0, 7.0, 0.0))
                raise Exception("Propagation failed")  # 后面的失败

            mock_prop.side_effect = side_effect
            positions = calculator._propagate_range(mock_satellite, start, end, timedelta(minutes=10))

            # 应该只有前3个点（0, 10, 20分钟）
            assert len(positions) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
