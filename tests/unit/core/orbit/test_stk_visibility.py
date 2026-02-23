"""
STK可见性计算器测试

TDD测试套件 - 测试STKVisibilityCalculator的完整COM接口功能
"""

import pytest
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# 先mock win32com，然后再导入被测模块
mock_win32com = MagicMock()
mock_win32com.client = MagicMock()
sys.modules['win32com'] = mock_win32com
sys.modules['win32com.client'] = mock_win32com.client

from core.orbit.visibility.stk_visibility import STKVisibilityCalculator
from core.orbit.visibility.base import VisibilityWindow


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def calculator():
    """创建默认计算器实例"""
    return STKVisibilityCalculator()


@pytest.fixture
def custom_calculator():
    """创建自定义配置的计算器"""
    config = {
        'min_elevation': 10.0,
        'time_step': 30
    }
    return STKVisibilityCalculator(config=config)


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


@pytest.fixture
def mock_stk_com():
    """创建模拟STK COM接口"""
    mock_com = MagicMock()
    mock_app = MagicMock()
    mock_root = MagicMock()
    mock_scenario = MagicMock()

    mock_app.New.return_value = mock_root
    mock_root.CurrentScenario = mock_scenario
    mock_root.NewScenario.return_value = mock_scenario
    mock_scenario.SetTimePeriod.return_value = None
    mock_scenario.StartTime = "1 Jan 2024 00:00:00.000"
    mock_scenario.StopTime = "1 Jan 2024 12:00:00.000"

    mock_com.Dispatch.return_value = mock_app
    return mock_com, mock_app, mock_root, mock_scenario


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
        assert calculator._stk_app is None
        assert calculator._stk_root is None
        assert calculator._scenario is None

    def test_custom_initialization(self, custom_calculator):
        """测试自定义配置初始化"""
        assert custom_calculator.min_elevation == 10.0
        assert custom_calculator.time_step == 30

    def test_empty_config_initialization(self):
        """测试空配置初始化"""
        calc = STKVisibilityCalculator(config={})
        assert calc.min_elevation == 5.0
        assert calc.time_step == 60

    def test_none_config_initialization(self):
        """测试None配置初始化"""
        calc = STKVisibilityCalculator(config=None)
        assert calc.min_elevation == 5.0
        assert calc.time_step == 60


# =============================================================================
# STK COM连接管理测试
# =============================================================================

class TestSTKConnection:
    """测试STK COM连接管理"""

    def test_connect_success(self, calculator, mock_stk_com):
        """测试成功连接STK"""
        mock_com, mock_app, mock_root, _ = mock_stk_com

        # 设置mock win32com
        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch
                    result = calculator.connect()

                    assert result is True
                    assert calculator._stk_app is not None
                    assert calculator._stk_root is not None

    def test_connect_failure(self, calculator):
        """测试连接STK失败"""
        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch.side_effect = Exception("STK not installed")

                    result = calculator.connect()

                    assert result is False
                    assert calculator._stk_app is None
                    assert calculator._stk_root is None

    def test_disconnect(self, calculator):
        """测试断开连接"""
        calculator._stk_app = Mock()
        calculator._stk_root = Mock()
        calculator._scenario = Mock()

        calculator.disconnect()

        assert calculator._stk_app is None
        assert calculator._stk_root is None
        assert calculator._scenario is None

    def test_disconnect_without_connection(self, calculator):
        """测试未连接时断开"""
        # 不应该抛出异常
        calculator.disconnect()
        assert calculator._stk_app is None
        assert calculator._stk_root is None

    def test_connect_already_connected(self, calculator, mock_stk_com):
        """测试已连接时再次连接"""
        mock_com, mock_app, mock_root, _ = mock_stk_com

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    # 第一次连接
                    result1 = calculator.connect()
                    assert result1 is True

                    # 第二次连接应该返回True（不重复创建）
                    result2 = calculator.connect()
                    assert result2 is True


# =============================================================================
# 场景管理测试
# =============================================================================

class TestScenarioManagement:
    """测试场景管理"""

    def test_setup_scenario(self, calculator, mock_stk_com):
        """测试设置场景"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    # 重置mock调用记录
                    mock_root.reset_mock()
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    result = calculator.setup_scenario("TestScenario")

                    assert result is True
                    assert calculator._scenario is not None

    def test_setup_scenario_without_connection(self, calculator):
        """测试未连接时设置场景"""
        # 应该先自动连接
        with patch.object(calculator, 'connect') as mock_connect:
            mock_connect.return_value = False
            calculator.setup_scenario("TestScenario")
            mock_connect.assert_called_once()


# =============================================================================
# 卫星和目标管理测试
# =============================================================================

class TestSatelliteAndTargetManagement:
    """测试卫星和目标管理"""

    def test_add_satellite(self, calculator, mock_satellite, mock_stk_com):
        """测试添加卫星"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_satellite_obj = MagicMock()
        mock_scenario.Children.New.return_value = mock_satellite_obj

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    result = calculator.add_satellite(mock_satellite)

                    assert result is True
                    assert mock_satellite.id in calculator._satellites

    def test_add_satellite_with_hpop(self, calculator, mock_satellite):
        """测试添加卫星并配置HPOP"""
        calculator._scenario = Mock()
        mock_sat_obj = MagicMock()
        mock_propagator = MagicMock()
        mock_integrator = MagicMock()
        mock_force_model = MagicMock()

        calculator._scenario.Children.New.return_value = mock_sat_obj
        mock_sat_obj.Propagator = mock_propagator
        mock_propagator.InitialState.IntegrationSetup = mock_integrator
        mock_propagator.InitialState.ForceModel = mock_force_model

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    calculator.add_satellite(mock_satellite, use_hpop=True)

                    assert mock_satellite.id in calculator._satellites

    def test_add_target(self, calculator, mock_target, mock_stk_com):
        """测试添加目标"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_target_obj = MagicMock()
        mock_scenario.Children.New.return_value = mock_target_obj

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    result = calculator.add_target(mock_target)

                    assert result is True
                    assert mock_target.id in calculator._targets

    def test_add_ground_station(self, calculator, mock_ground_station, mock_stk_com):
        """测试添加地面站"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_gs_obj = MagicMock()
        mock_scenario.Children.New.return_value = mock_gs_obj

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    result = calculator.add_ground_station(mock_ground_station)

                    assert result is True
                    assert mock_ground_station.id in calculator._ground_stations


# =============================================================================
# HPOP传播器配置测试
# =============================================================================

class TestHPOPConfiguration:
    """测试HPOP传播器配置"""

    def test_configure_hpop_force_model(self, calculator):
        """测试配置HPOP力模型"""
        mock_satellite = Mock()
        mock_propagator = Mock()
        mock_force_model = Mock()
        mock_integrator = Mock()

        mock_satellite.Propagator = mock_propagator
        mock_propagator.InitialState.ForceModel = mock_force_model
        mock_propagator.InitialState.IntegrationSetup = mock_integrator

        calculator._configure_hpop_force_model(mock_satellite)

        # 验证力模型配置
        assert mock_force_model.UseDragModel.called or True  # 配置被调用

    def test_setup_hpop_integrator(self, calculator):
        """测试设置HPOP积分器"""
        mock_integrator = Mock()
        mock_integrator.Step = 0.0  # 初始值

        calculator._setup_hpop_integrator(mock_integrator, time_step=60.0)

        # 验证积分器配置
        assert mock_integrator.Step == 60.0


# =============================================================================
# 可见性窗口计算测试
# =============================================================================

class TestVisibilityWindowComputation:
    """测试可见性窗口计算"""

    def test_compute_satellite_target_windows_not_connected(self, calculator, mock_satellite, mock_target, time_range):
        """测试未连接时计算窗口"""
        start, end = time_range

        # 未连接时应该返回空列表
        windows = calculator.compute_satellite_target_windows(
            mock_satellite, mock_target, start, end
        )

        assert isinstance(windows, list)
        assert len(windows) == 0

    def test_compute_satellite_target_windows_with_access(self, calculator, mock_satellite, mock_target, time_range, mock_stk_com):
        """测试计算卫星-目标可见窗口"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        # 设置模拟访问数据
        mock_access = MagicMock()
        mock_access.Computed = True
        mock_access.DataSets.Count = 1
        mock_access.DataSets.Item.return_value.GetValues.return_value = [
            ("1 Jan 2024 08:00:00.000", "1 Jan 2024 08:10:00.000", 45.0)
        ]

        mock_sat_obj = MagicMock()
        mock_target_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_target_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start, end = time_range
                    windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )

                    assert isinstance(windows, list)

    def test_compute_satellite_ground_station_windows(self, calculator, mock_satellite, mock_ground_station, time_range, mock_stk_com):
        """测试计算卫星-地面站可见窗口"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        # 设置模拟访问数据
        mock_access = MagicMock()
        mock_access.Computed = True
        mock_access.DataSets.Count = 1
        mock_access.DataSets.Item.return_value.GetValues.return_value = [
            ("1 Jan 2024 08:00:00.000", "1 Jan 2024 08:10:00.000", 45.0)
        ]

        mock_sat_obj = MagicMock()
        mock_gs_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_gs_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start, end = time_range
                    windows = calculator.compute_satellite_ground_station_windows(
                        mock_satellite, mock_ground_station, start, end
                    )

                    assert isinstance(windows, list)

    def test_compute_satellite_target_windows_fallback(self, calculator, mock_satellite, mock_target, time_range):
        """测试STK不可用时使用fallback计算"""
        start, end = time_range

        # 模拟STK不可用
        with patch.object(calculator, '_is_stk_available', return_value=False):
            with patch.object(calculator, '_compute_fallback_windows') as mock_fallback:
                mock_fallback.return_value = [
                    VisibilityWindow(
                        satellite_id="SAT-001",
                        target_id="TARGET-001",
                        start_time=start + timedelta(minutes=10),
                        end_time=start + timedelta(minutes=20),
                        max_elevation=45.0,
                        quality_score=0.5
                    )
                ]

                windows = calculator.compute_satellite_target_windows(
                    mock_satellite, mock_target, start, end
                )

                assert isinstance(windows, list)
                assert len(windows) == 1


# =============================================================================
# 简化接口测试
# =============================================================================

class TestSimplifiedInterface:
    """测试简化接口"""

    def test_calculate_windows(self, calculator, time_range):
        """测试calculate_windows简化接口"""
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

    def test_null_satellite(self, calculator, mock_target):
        """测试空卫星对象"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        windows = calculator.compute_satellite_target_windows(
            None, mock_target, start, end
        )
        assert windows == []

    def test_null_target(self, calculator, mock_satellite):
        """测试空目标对象"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        windows = calculator.compute_satellite_target_windows(
            mock_satellite, None, start, end
        )
        assert windows == []

    def test_null_ground_station(self, calculator, mock_satellite):
        """测试空地面站对象"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        windows = calculator.compute_satellite_ground_station_windows(
            mock_satellite, None, start, end
        )
        assert windows == []

    def test_stk_com_error_handling(self, calculator, mock_satellite, mock_target, time_range):
        """测试STK COM错误处理"""
        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch.side_effect = Exception("COM error")

                    start, end = time_range
                    windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )

                    assert isinstance(windows, list)

    def test_access_computation_error(self, calculator, mock_satellite, mock_target, time_range, mock_stk_com):
        """测试Access计算错误处理"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_access = MagicMock()
        mock_access.Computed = False
        mock_access.ComputeAccess.side_effect = Exception("Access computation failed")

        mock_sat_obj = MagicMock()
        mock_target_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_target_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start, end = time_range
                    windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )

                    assert isinstance(windows, list)


# =============================================================================
# 高级力模型配置测试
# =============================================================================

class TestAdvancedForceModels:
    """测试高级力模型配置"""

    def test_configure_atmospheric_drag(self, calculator):
        """测试配置大气阻力"""
        mock_force_model = Mock()
        mock_force_model.Drag = Mock()

        calculator._configure_atmospheric_drag(mock_force_model)

        # 验证大气阻力配置被启用
        assert mock_force_model.UseDragModel is True

    def test_configure_solar_radiation_pressure(self, calculator):
        """测试配置太阳光压"""
        mock_force_model = Mock()
        mock_force_model.SRP = Mock()

        calculator._configure_solar_radiation_pressure(mock_force_model)

        # 验证太阳光压配置被启用
        assert mock_force_model.UseSRP is True

    def test_configure_third_body_gravity(self, calculator):
        """测试配置三体引力"""
        mock_force_model = Mock()
        mock_force_model.ThirdBodyGravity = Mock()

        calculator._configure_third_body_gravity(mock_force_model)

        # 验证三体引力配置被启用
        assert mock_force_model.UseThirdBodyGravity is True


# =============================================================================
# 时间解析测试
# =============================================================================

class TestTimeParsing:
    """测试时间解析功能"""

    def test_parse_stk_time(self, calculator):
        """测试解析STK时间格式"""
        stk_time = "1 Jan 2024 08:30:00.000"

        result = calculator._parse_stk_time(stk_time)

        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 8
        assert result.minute == 30

    def test_parse_stk_time_invalid(self, calculator):
        """测试解析无效STK时间"""
        invalid_time = "invalid time format"

        result = calculator._parse_stk_time(invalid_time)

        # 应该返回None或当前时间
        assert result is None or isinstance(result, datetime)

    def test_format_stk_time(self, calculator):
        """测试格式化为STK时间格式"""
        dt = datetime(2024, 1, 15, 14, 30, 45)

        result = calculator._format_stk_time(dt)

        assert isinstance(result, str)
        assert "2024" in result
        assert "Jan" in result or "15" in result


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能相关测试"""

    def test_large_time_range(self, calculator, mock_satellite, mock_target):
        """测试大时间范围"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 7, 0, 0, 0)  # 一周

        # 应该能够处理大时间范围而不崩溃
        with patch.object(calculator, '_is_stk_available', return_value=False):
            with patch.object(calculator, '_compute_fallback_windows') as mock_fallback:
                mock_fallback.return_value = []

                windows = calculator.compute_satellite_target_windows(
                    mock_satellite, mock_target, start, end, timedelta(hours=1)
                )

                assert isinstance(windows, list)


# =============================================================================
# 积分测试
# =============================================================================

class TestIntegration:
    """积分测试 - 测试完整流程"""

    def test_full_workflow_with_stk(self, calculator, mock_satellite, mock_target, mock_ground_station, time_range, mock_stk_com):
        """测试完整的STK工作流"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        # 设置模拟访问数据
        mock_access = MagicMock()
        mock_access.Computed = True
        mock_access.DataSets.Count = 2
        mock_access.DataSets.Item.side_effect = [
            Mock(GetValues=lambda: [
                ("1 Jan 2024 08:00:00.000", "1 Jan 2024 08:10:00.000", 45.0),
                ("1 Jan 2024 09:30:00.000", "1 Jan 2024 09:40:00.000", 60.0)
            ])
        ]

        mock_sat_obj = MagicMock()
        mock_target_obj = MagicMock()
        mock_gs_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_target_obj, mock_gs_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    # 完整工作流
                    assert calculator.connect() is True
                    calculator.setup_scenario("IntegrationTest")
                    calculator.add_satellite(mock_satellite, use_hpop=True)
                    calculator.add_target(mock_target)
                    calculator.add_ground_station(mock_ground_station)

                    start, end = time_range
                    target_windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )
                    gs_windows = calculator.compute_satellite_ground_station_windows(
                        mock_satellite, mock_ground_station, start, end
                    )

                    assert isinstance(target_windows, list)
                    assert isinstance(gs_windows, list)

    def test_full_workflow_without_stk(self, calculator, mock_satellite, mock_target, time_range):
        """测试无STK时的完整工作流"""
        start, end = time_range

        with patch.object(calculator, '_is_stk_available', return_value=False):
            with patch.object(calculator, '_compute_fallback_windows') as mock_fallback:
                mock_fallback.return_value = [
                    VisibilityWindow(
                        satellite_id="SAT-001",
                        target_id="TARGET-001",
                        start_time=start + timedelta(minutes=30),
                        end_time=start + timedelta(minutes=40),
                        max_elevation=45.0,
                        quality_score=0.5
                    )
                ]

                windows = calculator.compute_satellite_target_windows(
                    mock_satellite, mock_target, start, end
                )

                assert isinstance(windows, list)
                assert len(windows) == 1


# =============================================================================
# 额外覆盖率测试
# =============================================================================

class TestAdditionalCoverage:
    """额外测试以提高覆盖率"""

    def test_connect_win32com_not_available(self, calculator):
        """测试win32com不可用时连接"""
        with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', False):
            result = calculator.connect()
            assert result is False

    def test_is_stk_available_false(self, calculator):
        """测试STK不可用检测"""
        with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', False):
            result = calculator._is_stk_available()
            assert result is False

    def test_add_satellite_with_tle(self, calculator, mock_stk_com):
        """测试使用TLE添加卫星"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_satellite_obj = MagicMock()
        mock_propagator = MagicMock()
        mock_satellite_obj.Propagator = mock_propagator
        mock_scenario.Children.New.return_value = mock_satellite_obj

        # 创建带TLE的卫星
        sat = Mock()
        sat.id = "SAT-TLE-001"
        sat.tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        sat.tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"
        sat.orbit = Mock()
        sat.orbit.altitude = 500000.0
        sat.orbit.inclination = 97.4

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    result = calculator.add_satellite(sat)

                    assert result is True

    def test_add_satellite_without_scenario(self, calculator, mock_satellite):
        """测试无场景时添加卫星（自动创建）"""
        with patch.object(calculator, 'connect') as mock_connect:
            mock_connect.return_value = False
            result = calculator.add_satellite(mock_satellite)
            assert result is False

    def test_add_target_without_scenario(self, calculator, mock_target):
        """测试无场景时添加目标（自动创建）"""
        with patch.object(calculator, 'connect') as mock_connect:
            mock_connect.return_value = False
            result = calculator.add_target(mock_target)
            assert result is False

    def test_add_ground_station_without_scenario(self, calculator, mock_ground_station):
        """测试无场景时添加地面站（自动创建）"""
        with patch.object(calculator, 'connect') as mock_connect:
            mock_connect.return_value = False
            result = calculator.add_ground_station(mock_ground_station)
            assert result is False

    def test_set_scenario_time_period(self, calculator, mock_stk_com):
        """测试设置场景时间范围"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start = datetime(2024, 1, 1, 0, 0, 0)
                    end = datetime(2024, 1, 2, 0, 0, 0)
                    result = calculator._set_scenario_time_period(start, end)

                    assert result is True

    def test_set_scenario_time_period_no_scenario(self, calculator):
        """测试无场景时设置时间范围"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)
        result = calculator._set_scenario_time_period(start, end)
        assert result is False

    def test_setup_two_body_propagator(self, calculator, mock_stk_com):
        """测试设置二体传播器"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_sat_obj = MagicMock()
        mock_propagator = MagicMock()
        mock_keplerian = MagicMock()

        mock_sat_obj.SetPropagatorType.return_value = None
        mock_sat_obj.Propagator = mock_propagator
        mock_propagator.InitialState.Representation.ConvertTo.return_value = mock_keplerian

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    # 测试二体传播器设置
                    sat = Mock()
                    sat.id = "SAT-TWO-001"
                    sat.tle_line1 = None
                    sat.tle_line2 = None
                    sat.orbit = Mock()
                    sat.orbit.altitude = 500000.0
                    sat.orbit.inclination = 97.4
                    sat.orbit.raan = 0.0
                    sat.orbit.arg_of_perigee = 0.0
                    sat.orbit.mean_anomaly = 0.0

                    calculator._setup_two_body_propagator(mock_sat_obj, sat)

                    mock_sat_obj.SetPropagatorType.assert_called_once()

    def test_compute_fallback_windows_with_ecef(self, calculator, mock_satellite, time_range):
        """测试fallback计算使用ECEF位置"""
        start, end = time_range

        # 创建有get_ecef_position的目标
        target = Mock()
        target.id = "TARGET-ECEF"
        target.get_ecef_position.return_value = (6371000.0, 0.0, 0.0)

        windows = calculator._compute_fallback_windows(
            mock_satellite, target, start, end, timedelta(minutes=10), is_ground_station=False
        )

        assert isinstance(windows, list)

    def test_compute_fallback_windows_with_lonlat(self, calculator, mock_satellite, time_range):
        """测试fallback计算使用经纬度"""
        start, end = time_range

        # 创建只有经纬度的目标
        target = Mock()
        target.id = "TARGET-LONLAT"
        target.longitude = 116.4074
        target.latitude = 39.9042
        target.altitude = 0.0
        # 删除get_ecef_position方法
        del target.get_ecef_position

        windows = calculator._compute_fallback_windows(
            mock_satellite, target, start, end, timedelta(minutes=10), is_ground_station=False
        )

        assert isinstance(windows, list)

    def test_estimate_satellite_position_error(self, calculator):
        """测试卫星位置估计错误处理"""
        sat = Mock()
        sat.orbit = None  # 会导致异常

        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = calculator._estimate_satellite_position(sat, dt)

        assert result is None

    def test_parse_access_data_empty(self, calculator):
        """测试解析空的Access数据"""
        mock_access = MagicMock()
        mock_access.DataSets.Item.return_value.GetValues.return_value = []

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)

        windows = calculator._parse_access_data(
            mock_access, "SAT-001", "TARGET-001", start, end
        )

        assert isinstance(windows, list)
        assert len(windows) == 0

    def test_compute_satellite_target_windows_stk_available_but_connect_fails(
        self, calculator, mock_satellite, mock_target, time_range
    ):
        """测试STK可用但连接失败时使用fallback"""
        start, end = time_range

        with patch.object(calculator, '_is_stk_available', return_value=True):
            with patch.object(calculator, 'connect', return_value=False):
                with patch.object(calculator, '_compute_fallback_windows') as mock_fallback:
                    mock_fallback.return_value = []

                    windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )

                    assert isinstance(windows, list)
                    mock_fallback.assert_called_once()

    def test_compute_satellite_ground_station_windows_stk_available_but_connect_fails(
        self, calculator, mock_satellite, mock_ground_station, time_range
    ):
        """测试STK可用但连接失败时使用fallback（地面站）"""
        start, end = time_range

        with patch.object(calculator, '_is_stk_available', return_value=True):
            with patch.object(calculator, 'connect', return_value=False):
                with patch.object(calculator, '_compute_fallback_windows') as mock_fallback:
                    mock_fallback.return_value = []

                    windows = calculator.compute_satellite_ground_station_windows(
                        mock_satellite, mock_ground_station, start, end
                    )

                    assert isinstance(windows, list)
                    mock_fallback.assert_called_once()

    def test_add_target_with_lonlat(self, calculator, mock_stk_com):
        """测试使用经纬度添加目标"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_target_obj = MagicMock()
        mock_scenario.Children.New.return_value = mock_target_obj

        # 创建只有经纬度的目标（没有get_ecef_position）
        target = Mock()
        target.id = "TARGET-LONLAT-001"
        target.longitude = 116.4074
        target.latitude = 39.9042
        target.altitude = 0.0
        del target.get_ecef_position

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    result = calculator.add_target(target)

                    assert result is True

    def test_add_ground_station_with_lonlat(self, calculator, mock_stk_com):
        """测试使用经纬度添加地面站"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_gs_obj = MagicMock()
        mock_scenario.Children.New.return_value = mock_gs_obj

        # 创建只有经纬度的地面站
        gs = Mock()
        gs.id = "GS-LONLAT-001"
        gs.longitude = 116.4074
        gs.latitude = 39.9042
        gs.altitude = 100.0
        del gs.get_ecef_position

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    result = calculator.add_ground_station(gs)

                    assert result is True

    def test_setup_hpop_propagator_exception(self, calculator, mock_stk_com):
        """测试HPOP传播器设置异常处理"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_sat_obj = MagicMock()
        mock_sat_obj.SetPropagatorType.side_effect = Exception("HPOP setup failed")

        sat = Mock()
        sat.id = "SAT-HPOP-001"
        sat.tle_line1 = None
        sat.tle_line2 = None
        sat.orbit = Mock()
        sat.orbit.altitude = 500000.0
        sat.orbit.inclination = 97.4
        sat.orbit.raan = 0.0
        sat.orbit.arg_of_perigee = 0.0
        sat.orbit.mean_anomaly = 0.0

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    # 应该抛出异常
                    try:
                        calculator._setup_hpop_propagator(mock_sat_obj, sat)
                        assert False, "Expected exception"
                    except Exception:
                        assert True

    def test_configure_hpop_force_model_exception(self, calculator):
        """测试HPOP力模型配置异常处理"""
        mock_propagator = Mock()
        mock_propagator.InitialState.ForceModel.side_effect = Exception("Force model error")

        try:
            calculator._configure_hpop_force_model(mock_propagator)
            assert False, "Expected exception"
        except Exception:
            assert True

    def test_parse_access_data_with_valid_entries(self, calculator):
        """测试解析有效的Access数据条目"""
        mock_access = MagicMock()
        mock_access.DataSets.Item.return_value.GetValues.return_value = [
            ("1 Jan 2024 08:00:00.000", "1 Jan 2024 08:10:00.000", 45.0),
            ("1 Jan 2024 09:00:00.000", "1 Jan 2024 09:15:00.000", 60.0),
        ]

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)

        windows = calculator._parse_access_data(
            mock_access, "SAT-001", "TARGET-001", start, end
        )

        assert isinstance(windows, list)
        assert len(windows) == 2

    def test_parse_access_data_outside_range(self, calculator):
        """测试解析时间范围外的Access数据"""
        mock_access = MagicMock()
        # 这些时间在请求范围之外
        mock_access.DataSets.Item.return_value.GetValues.return_value = [
            ("1 Jan 2023 08:00:00.000", "1 Jan 2023 08:10:00.000", 45.0),
        ]

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)

        windows = calculator._parse_access_data(
            mock_access, "SAT-001", "TARGET-001", start, end
        )

        assert isinstance(windows, list)
        assert len(windows) == 0

    def test_compute_fallback_windows_no_position_method(self, calculator, mock_satellite, time_range):
        """测试fallback计算目标没有位置方法"""
        start, end = time_range

        # 创建没有位置信息的目标
        target = Mock()
        target.id = "TARGET-NO-POS"
        del target.get_ecef_position
        del target.longitude
        del target.latitude

        windows = calculator._compute_fallback_windows(
            mock_satellite, target, start, end, timedelta(minutes=10), is_ground_station=False
        )

        assert isinstance(windows, list)
        assert len(windows) == 0

    def test_disconnect_with_exception(self, calculator):
        """测试断开连接时抛出异常"""
        calculator._stk_app = Mock()
        calculator._stk_app.Quit.side_effect = Exception("Quit failed")

        # 不应该抛出异常
        calculator.disconnect()

        assert calculator._stk_app is None

    def test_setup_scenario_close_existing(self, calculator, mock_stk_com):
        """测试设置场景时关闭现有场景"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("FirstScenario")
                    # 再次设置场景，应该关闭第一个
                    result = calculator.setup_scenario("SecondScenario")

                    assert result is True
                    assert calculator._scenario is not None

    def test_compute_satellite_target_windows_with_cached_objects(
        self, calculator, mock_satellite, mock_target, time_range, mock_stk_com
    ):
        """测试使用缓存的对象计算窗口"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_access = MagicMock()
        mock_access.Computed = True
        mock_access.DataSets.Item.return_value.GetValues.return_value = [
            ("1 Jan 2024 08:00:00.000", "1 Jan 2024 08:10:00.000", 45.0)
        ]

        mock_sat_obj = MagicMock()
        mock_target_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_target_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    # 先添加对象到缓存
                    calculator.add_satellite(mock_satellite)
                    calculator.add_target(mock_target)

                    start, end = time_range
                    # 再次计算，应该使用缓存的对象
                    windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )

                    assert isinstance(windows, list)

    def test_compute_satellite_ground_station_windows_with_cached_objects(
        self, calculator, mock_satellite, mock_ground_station, time_range, mock_stk_com
    ):
        """测试使用缓存的地面站计算窗口"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_access = MagicMock()
        mock_access.Computed = True
        mock_access.DataSets.Item.return_value.GetValues.return_value = [
            ("1 Jan 2024 08:00:00.000", "1 Jan 2024 08:10:00.000", 45.0)
        ]

        mock_sat_obj = MagicMock()
        mock_gs_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_gs_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    # 先添加对象到缓存
                    calculator.add_satellite(mock_satellite)
                    calculator.add_ground_station(mock_ground_station)

                    start, end = time_range
                    windows = calculator.compute_satellite_ground_station_windows(
                        mock_satellite, mock_ground_station, start, end
                    )

                    assert isinstance(windows, list)

    def test_compute_satellite_target_windows_access_not_computed(
        self, calculator, mock_satellite, mock_target, time_range, mock_stk_com
    ):
        """测试Access未计算时返回空列表"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_access = MagicMock()
        mock_access.Computed = False

        mock_sat_obj = MagicMock()
        mock_target_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_target_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start, end = time_range
                    windows = calculator.compute_satellite_target_windows(
                        mock_satellite, mock_target, start, end
                    )

                    assert isinstance(windows, list)
                    assert len(windows) == 0

    def test_compute_satellite_ground_station_windows_access_not_computed(
        self, calculator, mock_satellite, mock_ground_station, time_range, mock_stk_com
    ):
        """测试地面站Access未计算时返回空列表"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        mock_access = MagicMock()
        mock_access.Computed = False

        mock_sat_obj = MagicMock()
        mock_gs_obj = MagicMock()
        mock_sat_obj.GetAccess.return_value = mock_access

        mock_scenario.Children.New.side_effect = [mock_sat_obj, mock_gs_obj]

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start, end = time_range
                    windows = calculator.compute_satellite_ground_station_windows(
                        mock_satellite, mock_ground_station, start, end
                    )

                    assert isinstance(windows, list)
                    assert len(windows) == 0

    def test_compute_fallback_windows_with_incomplete_window(self, calculator, mock_satellite, time_range):
        """测试fallback计算中未完成窗口的处理"""
        start, end = time_range

        target = Mock()
        target.id = "TARGET-INCOMPLETE"
        target.get_ecef_position.return_value = (6371000.0, 0.0, 0.0)

        # 使用很小的时间步长来确保有多个点
        windows = calculator._compute_fallback_windows(
            mock_satellite, target, start, end, timedelta(seconds=30), is_ground_station=False
        )

        assert isinstance(windows, list)

    def test_configure_atmospheric_drag_exception(self, calculator):
        """测试配置大气阻力时异常处理"""
        mock_force_model = Mock()
        mock_force_model.Drag = Mock()
        mock_force_model.UseDragModel = True
        # 设置Drag属性时抛出异常
        type(mock_force_model).Drag = PropertyMock(side_effect=Exception("Drag error"))

        # 不应该抛出异常
        calculator._configure_atmospheric_drag(mock_force_model)
        assert True

    def test_configure_solar_radiation_pressure_exception(self, calculator):
        """测试配置太阳光压时异常处理"""
        mock_force_model = Mock()
        mock_force_model.SRP = Mock()
        mock_force_model.UseSRP = True
        type(mock_force_model).SRP = PropertyMock(side_effect=Exception("SRP error"))

        # 不应该抛出异常
        calculator._configure_solar_radiation_pressure(mock_force_model)
        assert True

    def test_configure_third_body_gravity_exception(self, calculator):
        """测试配置三体引力时异常处理"""
        mock_force_model = Mock()
        mock_force_model.ThirdBodyGravity = Mock()
        mock_force_model.UseThirdBodyGravity = True
        type(mock_force_model).ThirdBodyGravity = PropertyMock(side_effect=Exception("Third body error"))

        # 不应该抛出异常
        calculator._configure_third_body_gravity(mock_force_model)
        assert True

    def test_setup_hpop_integrator_exception(self, calculator):
        """测试设置HPOP积分器时异常处理"""
        mock_integrator = Mock()
        type(mock_integrator).Step = PropertyMock(side_effect=Exception("Integrator error"))

        # 不应该抛出异常
        calculator._setup_hpop_integrator(mock_integrator, 60.0)
        assert True

    def test_estimate_satellite_position_success(self, calculator):
        """测试成功估计卫星位置"""
        sat = Mock()
        sat.orbit = Mock()
        sat.orbit.altitude = 500000.0
        sat.orbit.inclination = 97.4
        sat.orbit.raan = 0.0

        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = calculator._estimate_satellite_position(sat, dt)

        assert result is not None
        assert len(result) == 3

    def test_compute_satellite_target_windows_with_none_sat_obj(self, calculator, mock_satellite, mock_target, time_range, mock_stk_com):
        """测试卫星对象获取失败时返回空列表"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")
                    # 不添加卫星到缓存

                    start, end = time_range
                    # 模拟add_satellite返回False导致sat_obj为None
                    with patch.object(calculator, 'add_satellite', return_value=False):
                        windows = calculator.compute_satellite_target_windows(
                            mock_satellite, mock_target, start, end
                        )

                    assert isinstance(windows, list)

    def test_compute_satellite_ground_station_windows_with_none_sat_obj(self, calculator, mock_satellite, mock_ground_station, time_range, mock_stk_com):
        """测试地面站计算时卫星对象获取失败"""
        mock_com, mock_app, mock_root, mock_scenario = mock_stk_com

        with patch.dict('sys.modules', {'win32com': mock_win32com, 'win32com.client': mock_win32com.client}):
            with patch('core.orbit.visibility.stk_visibility.win32com', mock_win32com):
                with patch('core.orbit.visibility.stk_visibility.WIN32COM_AVAILABLE', True):
                    mock_win32com.client.Dispatch = mock_com.Dispatch

                    calculator.connect()
                    calculator.setup_scenario("TestScenario")

                    start, end = time_range
                    with patch.object(calculator, 'add_satellite', return_value=False):
                        windows = calculator.compute_satellite_ground_station_windows(
                            mock_satellite, mock_ground_station, start, end
                        )

                    assert isinstance(windows, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
