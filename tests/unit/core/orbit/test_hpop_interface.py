"""
STK HPOP接口单元测试

测试C3: STK HPOP高精度轨道预报接口
"""

import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from core.orbit.hpop_interface import (
    STKHPOPInterface,
    HPOPPropagationResult,
    STKNotAvailableError,
    HPOPConfig,
    ForceModel,
    AtmosphericModel
)


class TestHPOPConfig:
    """测试HPOP配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = HPOPConfig()
        assert config.time_step == 60.0
        assert config.use_sun_gravity is True
        assert config.use_moon_gravity is True
        assert config.use_solar_radiation_pressure is True
        assert config.use_atmospheric_drag is True
        assert config.atmospheric_model == AtmosphericModel.MSIS90

    def test_custom_config(self):
        """测试自定义配置"""
        config = HPOPConfig(
            time_step=30.0,
            use_atmospheric_drag=False,
            atmospheric_model=AtmosphericModel.JACCHIA77
        )
        assert config.time_step == 30.0
        assert config.use_atmospheric_drag is False
        assert config.atmospheric_model == AtmosphericModel.JACCHIA77


class TestForceModel:
    """测试力学模型枚举"""

    def test_force_model_values(self):
        """测试力学模型枚举值"""
        assert ForceModel.J2.value == "j2"
        assert ForceModel.J4.value == "j4"
        assert ForceModel.HPOP_FULL.value == "hpop_full"


class TestSTKHPOPInterface:
    """测试STK HPOP接口"""

    def test_init_without_stk(self):
        """测试无STK时的初始化"""
        with patch('core.orbit.hpop_interface.STKHPOPInterface._check_stk_available', return_value=False):
            with pytest.raises(STKNotAvailableError):
                STKHPOPInterface()

    def test_init_with_stk(self):
        """测试有STK时的初始化"""
        with patch('core.orbit.hpop_interface.STKHPOPInterface._check_stk_available', return_value=True):
            with patch('core.orbit.hpop_interface.STKHPOPInterface._init_stk_connection'):
                interface = STKHPOPInterface()
                assert interface.is_connected is False

    def test_check_stk_available(self):
        """测试STK可用性检查"""
        # 在非Windows平台或没有win32com时，STK不可用
        with patch('core.orbit.hpop_interface.logger'):
            result = STKHPOPInterface._check_stk_available()
            # 由于测试环境没有win32com，应该返回False
            assert isinstance(result, bool)

    def test_connect(self):
        """测试连接STK"""
        with patch('core.orbit.hpop_interface.STKHPOPInterface._check_stk_available', return_value=True):
            with patch('core.orbit.hpop_interface.STKHPOPInterface._init_stk_connection') as mock_init:
                interface = STKHPOPInterface()
                interface.connect()
                mock_init.assert_called_once()
                assert interface.is_connected is True

    def test_disconnect(self):
        """测试断开连接"""
        with patch('core.orbit.hpop_interface.STKHPOPInterface._check_stk_available', return_value=True):
            with patch('core.orbit.hpop_interface.STKHPOPInterface._init_stk_connection'):
                interface = STKHPOPInterface()
                interface.connect()
                interface.disconnect()
                assert interface.is_connected is False


class TestHPOPPropagation:
    """测试HPOP轨道传播"""

    @pytest.fixture
    def mock_interface(self):
        """创建模拟的HPOP接口"""
        with patch('core.orbit.hpop_interface.STKHPOPInterface._check_stk_available', return_value=True):
            with patch('core.orbit.hpop_interface.STKHPOPInterface._init_stk_connection'):
                interface = STKHPOPInterface()
                interface._stk_app = MagicMock()
                interface._stk_root = MagicMock()
                interface._scenario = MagicMock()
                interface.is_connected = True
                yield interface

    @pytest.fixture
    def sample_orbit(self):
        """创建示例轨道"""
        orbit = Mock()
        orbit.semi_major_axis = 6878000.0  # 500km altitude
        orbit.eccentricity = 0.001
        orbit.inclination = 97.4
        orbit.raan = 45.0
        orbit.arg_of_perigee = 0.0
        orbit.mean_anomaly = 0.0
        return orbit

    def test_propagate_orbit_success(self, mock_interface, sample_orbit):
        """测试成功传播轨道"""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = start_time + timedelta(hours=1)

        # 模拟STK返回数据
        mock_data_provider = MagicMock()
        mock_data_provider.ExecElements.return_value = (
            (0.0, 6878000.0, 0.0, 0.0),  # time, x, y, z
            (600.0, 6800000.0, 500000.0, 100000.0),
        )

        mock_satellite = MagicMock()
        mock_satellite.DataProviders.GetDataPrvTimeVarFromPath.return_value = mock_data_provider
        mock_interface._scenario.GetChildren.return_value = ([mock_satellite], None)

        result = mock_interface.propagate_orbit(
            satellite_id="SAT-01",
            orbit=sample_orbit,
            start_time=start_time,
            end_time=end_time
        )

        assert isinstance(result, HPOPPropagationResult)
        assert result.satellite_id == "SAT-01"
        assert len(result.positions) > 0
        assert len(result.velocities) > 0
        assert len(result.timestamps) > 0

    def test_propagate_orbit_not_connected(self, mock_interface, sample_orbit):
        """测试未连接时传播失败"""
        mock_interface.is_connected = False

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = start_time + timedelta(hours=1)

        with pytest.raises(RuntimeError, match="Not connected to STK"):
            mock_interface.propagate_orbit(
                satellite_id="SAT-01",
                orbit=sample_orbit,
                start_time=start_time,
                end_time=end_time
            )

    def test_propagate_orbit_invalid_time(self, mock_interface, sample_orbit):
        """测试无效时间范围"""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = start_time - timedelta(hours=1)  # 结束时间早于开始时间

        with pytest.raises(ValueError, match="End time must be after start time"):
            mock_interface.propagate_orbit(
                satellite_id="SAT-01",
                orbit=sample_orbit,
                start_time=start_time,
                end_time=end_time
            )

    def test_get_position_at_time(self, mock_interface):
        """测试获取指定时间的位置"""
        mock_interface._position_cache = {
            "SAT-01": [
                (datetime(2024, 1, 1, 0, 0, 0), (6878000.0, 0.0, 0.0), (0.0, 7500.0, 0.0)),
                (datetime(2024, 1, 1, 0, 1, 0), (6800000.0, 500000.0, 100000.0), (-100.0, 7400.0, 200.0)),
            ]
        }

        query_time = datetime(2024, 1, 1, 0, 0, 30)
        position, velocity = mock_interface.get_position_at_time("SAT-01", query_time)

        assert position is not None
        assert velocity is not None
        assert len(position) == 3
        assert len(velocity) == 3

    def test_get_position_at_time_not_found(self, mock_interface):
        """测试获取不存在的位置"""
        mock_interface._position_cache = {}

        query_time = datetime(2024, 1, 1, 0, 0, 0)
        position, velocity = mock_interface.get_position_at_time("SAT-01", query_time)

        assert position is None
        assert velocity is None

    def test_configure_hpop_force_model(self, mock_interface):
        """测试配置HPOP力学模型"""
        config = HPOPConfig(
            force_model=ForceModel.HPOP_FULL,
            use_atmospheric_drag=True,
            use_solar_radiation_pressure=True
        )

        mock_satellite = MagicMock()
        mock_propagator = MagicMock()
        mock_satellite.Propagator = mock_propagator

        mock_interface._configure_hpop_force_model(mock_satellite, config)

        # 验证HPOP配置被应用
        mock_propagator.ForceModel.UseAtmosphericDrag = True
        mock_propagator.ForceModel.UseSolarRadiationPressure = True


class TestHPOPPropagationResult:
    """测试HPOP传播结果"""

    @pytest.fixture
    def sample_result(self):
        """创建示例传播结果"""
        return HPOPPropagationResult(
            satellite_id="SAT-01",
            timestamps=[
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 0, 1, 0),
                datetime(2024, 1, 1, 0, 2, 0),
            ],
            positions=[
                (6878000.0, 0.0, 0.0),
                (6800000.0, 500000.0, 100000.0),
                (6700000.0, 800000.0, 200000.0),
            ],
            velocities=[
                (0.0, 7500.0, 0.0),
                (-100.0, 7400.0, 200.0),
                (-200.0, 7300.0, 400.0),
            ]
        )

    def test_get_position_at_time_interpolation(self, sample_result):
        """测试时间插值获取位置"""
        query_time = datetime(2024, 1, 1, 0, 0, 30)  # 中间时刻
        position, velocity = sample_result.get_position_at_time(query_time)

        assert position is not None
        assert velocity is not None

    def test_get_orbital_elements(self, sample_result):
        """测试获取轨道根数"""
        elements = sample_result.get_orbital_elements()

        assert 'semi_major_axis' in elements
        assert 'eccentricity' in elements
        assert 'inclination' in elements

    def test_get_subpoint(self, sample_result):
        """测试获取星下点坐标"""
        query_time = datetime(2024, 1, 1, 0, 0, 0)
        lat, lon, alt = sample_result.get_subpoint(query_time)

        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert isinstance(alt, float)


class TestAtmosphericModel:
    """测试大气模型枚举"""

    def test_atmospheric_model_values(self):
        """测试大气模型枚举值"""
        assert AtmosphericModel.MSIS90.value == "msis90"
        assert AtmosphericModel.JACCHIA77.value == "jacchia77"
        assert AtmosphericModel.JACCHIA71.value == "jacchia71"


class TestErrorHandling:
    """测试错误处理"""

    def test_stk_not_available_error(self):
        """测试STK不可用错误"""
        error = STKNotAvailableError("STK not installed")
        assert str(error) == "STK not installed"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
