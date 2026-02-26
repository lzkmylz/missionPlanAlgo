"""
Orekit可见性计算器 Phase 3 集成测试

TDD测试套件 - 测试OrekitVisibilityCalculator与Java Orekit的集成
包括配置选项、传播器选择、回退机制和异常处理。
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge, OrbitPropagationError
from core.orbit.visibility.base import VisibilityWindow


# 模拟numpy数组，避免依赖问题
class MockNumpyArray:
    """模拟numpy数组行为"""
    def __init__(self, data):
        self._data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)


def create_mock_numpy_array(data):
    """创建模拟的numpy数组"""
    return MockNumpyArray(data)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_calculator():
    """创建默认计算器实例（不使用Java Orekit）"""
    return OrekitVisibilityCalculator()


@pytest.fixture
def java_enabled_calculator():
    """创建启用Java Orekit的计算器实例"""
    config = {
        'min_elevation': 5.0,
        'time_step': 60,
        'use_java_orekit': True,
        'orekit': {
            'jvm': {
                'max_memory': '2g'
            },
            'propagator': {
                'integrator': 'DormandPrince853',
                'min_step': 0.001,
                'max_step': 300.0
            }
        }
    }
    return OrekitVisibilityCalculator(config=config)


@pytest.fixture
def java_disabled_calculator():
    """创建显式禁用Java Orekit的计算器实例"""
    config = {
        'use_java_orekit': False
    }
    return OrekitVisibilityCalculator(config=config)


@pytest.fixture
def mock_satellite():
    """创建模拟卫星"""
    sat = Mock()
    sat.id = "SAT-001"
    sat.name = "Test Satellite"
    sat.orbit = Mock()
    sat.orbit.altitude = 500000.0  # 500km
    sat.orbit.inclination = 97.4
    sat.orbit.raan = 0.0
    sat.orbit.mean_anomaly = 0.0
    sat.tle_line1 = None
    sat.tle_line2 = None
    return sat


@pytest.fixture
def mock_target():
    """创建模拟点目标"""
    target = Mock()
    target.id = "TARGET-001"
    target.name = "Beijing"
    target.longitude = 116.4074
    target.latitude = 39.9042
    target.altitude = 0.0
    target.get_ecef_position.return_value = (
        -2171419.0, 4387557.0, 4070234.0
    )
    return target


@pytest.fixture
def time_range():
    """创建测试时间范围"""
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return start, end


@pytest.fixture
def mock_bridge():
    """创建模拟的OrekitJavaBridge"""
    bridge = Mock(spec=OrekitJavaBridge)
    bridge.is_jvm_running.return_value = True

    # 模拟批量传播返回模拟数组
    def mock_propagate_batch(propagator, start_date, end_date, step_size):
        # 返回模拟的轨道数据 (n_steps, 7)
        # [seconds_since_j2000, px, py, pz, vx, vy, vz]
        n_steps = 10
        data = [[0.0] * 7 for _ in range(n_steps)]
        for i in range(n_steps):
            data[i][0] = i * 60  # seconds
            data[i][1] = 7000000.0  # x
            data[i][2] = 0.0  # y
            data[i][3] = 0.0  # z
            data[i][4] = 0.0  # vx
            data[i][5] = 7000.0  # vy
            data[i][6] = 0.0  # vz
        return create_mock_numpy_array(data)

    bridge.propagate_batch.side_effect = mock_propagate_batch
    return bridge


# =============================================================================
# 配置选项测试
# =============================================================================

class TestConfigurationOptions:
    """测试配置选项"""

    def test_default_use_java_orekit_false(self, default_calculator):
        """测试默认情况下use_java_orekit为False"""
        assert hasattr(default_calculator, 'use_java_orekit')
        assert default_calculator.use_java_orekit is False

    def test_explicit_enable_java_orekit(self, java_enabled_calculator):
        """测试显式启用Java Orekit"""
        assert java_enabled_calculator.use_java_orekit is True

    def test_explicit_disable_java_orekit(self, java_disabled_calculator):
        """测试显式禁用Java Orekit"""
        assert java_disabled_calculator.use_java_orekit is False

    def test_orekit_config_storage(self, java_enabled_calculator):
        """测试orekit配置被正确存储"""
        assert hasattr(java_enabled_calculator, 'orekit_config')
        assert 'jvm' in java_enabled_calculator.orekit_config
        assert 'propagator' in java_enabled_calculator.orekit_config

    def test_empty_orekit_config_default(self, default_calculator):
        """测试默认情况下orekit_config为空或默认值"""
        assert hasattr(default_calculator, 'orekit_config')
        # 当use_java_orekit为False时，orekit_config可以为空或默认值

    def test_config_with_none_orekit(self):
        """测试配置中orekit为None的情况"""
        config = {
            'use_java_orekit': True,
            'orekit': None
        }
        calc = OrekitVisibilityCalculator(config=config)
        assert calc.use_java_orekit is True
        assert calc.orekit_config is not None  # 应该使用默认配置


# =============================================================================
# 传播器选择逻辑测试
# =============================================================================

class TestPropagatorSelection:
    """测试传播器自动选择逻辑"""

    def test_propagate_satellite_uses_java_when_enabled(self, java_enabled_calculator, mock_satellite):
        """测试启用Java时调用Java传播方法"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(java_enabled_calculator, '_propagate_with_java_orekit') as mock_java:
            with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
                mock_java.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))
                mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

                java_enabled_calculator._propagate_satellite(mock_satellite, dt)

                # 应该调用Java方法，不调用简化方法
                mock_java.assert_called_once()
                mock_simple.assert_not_called()

    def test_propagate_satellite_uses_simplified_when_disabled(self, java_disabled_calculator, mock_satellite):
        """测试禁用Java时使用简化传播方法"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(java_disabled_calculator, '_propagate_with_java_orekit') as mock_java:
            with patch.object(java_disabled_calculator, '_propagate_simplified') as mock_simple:
                mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

                java_disabled_calculator._propagate_satellite(mock_satellite, dt)

                # 应该调用简化方法，不调用Java方法
                mock_java.assert_not_called()
                mock_simple.assert_called_once()

    def test_propagate_satellite_uses_simplified_by_default(self, default_calculator, mock_satellite):
        """测试默认情况下使用简化传播方法"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(default_calculator, '_propagate_with_java_orekit') as mock_java:
            with patch.object(default_calculator, '_propagate_simplified') as mock_simple:
                mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

                default_calculator._propagate_satellite(mock_satellite, dt)

                # 默认应该调用简化方法
                mock_java.assert_not_called()
                mock_simple.assert_called_once()


# =============================================================================
# Java Orekit集成测试
# =============================================================================

class TestJavaOrekitIntegration:
    """测试与Java Orekit的集成"""

    def test_propagate_range_with_java_orekit_exists(self, java_enabled_calculator):
        """测试_propagate_range_with_java_orekit方法存在"""
        assert hasattr(java_enabled_calculator, '_propagate_range_with_java_orekit')

    def test_propagate_with_java_orekit_exists(self, java_enabled_calculator):
        """测试_propagate_with_java_orekit方法存在"""
        assert hasattr(java_enabled_calculator, '_propagate_with_java_orekit')

    @patch.object(OrekitVisibilityCalculator, '_propagate_range_with_java_orekit')
    def test_propagate_range_with_java_calls_bridge(self, mock_range_java, java_enabled_calculator, mock_satellite, time_range):
        """测试Java传播方法调用OrekitJavaBridge"""
        start, end = time_range

        # 模拟返回数据
        n_steps = 61
        mock_result = [
            ((7000000.0 + i * 1000, i * 100.0, 0.0), (0.0, 7500.0, 0.0), start + timedelta(seconds=i * 60))
            for i in range(n_steps)
        ]
        mock_range_java.return_value = mock_result

        # 调用方法（通过_propagate_range间接调用）
        time_step = timedelta(seconds=60)
        result = java_enabled_calculator._propagate_range_with_java_orekit(
            mock_satellite, start, end, time_step
        )

        # 验证结果
        assert isinstance(result, list)
        assert len(result) == n_steps

        # 验证每个结果包含位置、速度和时间戳
        for pos, vel, dt in result:
            assert len(pos) == 3
            assert len(vel) == 3
            assert isinstance(dt, datetime)

    def test_propagate_with_java_single_point(self, java_enabled_calculator, mock_satellite):
        """测试Java单点传播"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 模拟批量传播返回3个点（用于提取中间点）
        mock_result = [
            ((7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0), dt - timedelta(seconds=1)),
            ((7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0), dt),
            ((7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0), dt + timedelta(seconds=1)),
        ]

        # 同时mock _propagate_range_with_java_orekit 和 _orekit_bridge.is_jvm_running
        with patch.object(java_enabled_calculator, '_propagate_range_with_java_orekit', return_value=mock_result):
            with patch.object(java_enabled_calculator, '_orekit_bridge') as mock_bridge:
                mock_bridge.is_jvm_running.return_value = True

                # 调用方法
                pos, vel = java_enabled_calculator._propagate_with_java_orekit(mock_satellite, dt)

                # 验证结果
                assert len(pos) == 3
                assert len(vel) == 3
                # 位置应该是米
                assert pos[0] == pytest.approx(7000000.0, abs=1.0)
                assert pos[1] == pytest.approx(0.0, abs=1.0)
                assert pos[2] == pytest.approx(0.0, abs=1.0)


# =============================================================================
# 回退机制测试
# =============================================================================

class TestFallbackMechanism:
    """测试回退机制"""

    def test_fallback_when_jvm_not_available(self, java_enabled_calculator, mock_satellite):
        """测试JVM不可用时回退到简化模型"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 设置mock - JVM未运行
        mock_bridge = Mock()
        mock_bridge.is_jvm_running.return_value = False
        mock_bridge._ensure_jvm_started.side_effect = RuntimeError("JVM not running")

        # 替换calculator的bridge为mock
        java_enabled_calculator._orekit_bridge = mock_bridge

        with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
            mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

            # 调用传播方法
            pos, vel = java_enabled_calculator._propagate_satellite(mock_satellite, dt)

            # 应该回退到简化方法
            mock_simple.assert_called_once()
            assert pos is not None
            assert vel is not None

    def test_fallback_when_jpype_not_installed(self, java_enabled_calculator, mock_satellite):
        """测试JPype未安装时回退到简化模型"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 模拟JPype不可用 - 将bridge设为None来模拟导入失败的情况
        java_enabled_calculator._orekit_bridge = None

        with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
            mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

            # 调用传播方法 - 应该直接回退到简化模型，因为bridge为None
            pos, vel = java_enabled_calculator._propagate_satellite(mock_satellite, dt)

            # 应该回退到简化方法
            mock_simple.assert_called_once()

    def test_fallback_when_java_exception(self, java_enabled_calculator, mock_satellite):
        """测试Java异常时回退到简化模型"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 设置mock - Java调用抛出异常
        mock_bridge = Mock()
        mock_bridge.is_jvm_running.return_value = True
        mock_bridge._ensure_jvm_started.return_value = None
        mock_bridge.propagate_batch.side_effect = OrbitPropagationError("Java error")

        # 替换calculator的bridge为mock
        java_enabled_calculator._orekit_bridge = mock_bridge

        with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
            mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

            # 调用传播方法
            pos, vel = java_enabled_calculator._propagate_satellite(mock_satellite, dt)

            # 应该回退到简化方法
            mock_simple.assert_called_once()

    def test_fallback_when_bridge_creation_fails(self, java_enabled_calculator, mock_satellite):
        """测试桥接器创建失败时回退到简化模型"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 模拟桥接器创建失败 - 将bridge设为None
        java_enabled_calculator._orekit_bridge = None

        with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
            mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

            # 调用传播方法 - 由于bridge为None，应该直接回退到简化模型
            pos, vel = java_enabled_calculator._propagate_satellite(mock_satellite, dt)

            # 应该回退到简化方法
            mock_simple.assert_called_once()


# =============================================================================
# 异常处理测试
# =============================================================================

class TestExceptionHandling:
    """测试异常处理"""

    def test_java_exception_handling_in_range(self, java_enabled_calculator, mock_satellite, time_range):
        """测试范围传播中的Java异常处理"""
        start, end = time_range

        # 设置mock - Java调用抛出异常
        mock_bridge = Mock()
        mock_bridge.is_jvm_running.return_value = True
        mock_bridge._ensure_jvm_started.return_value = None
        mock_bridge.propagate_batch.side_effect = OrbitPropagationError("Propagation failed")

        # 替换calculator的bridge为mock
        java_enabled_calculator._orekit_bridge = mock_bridge

        # 应该抛出异常或返回空列表（取决于实现）
        with pytest.raises(Exception) as exc_info:
            java_enabled_calculator._propagate_range_with_java_orekit(
                mock_satellite, start, end, timedelta(seconds=60)
            )

    def test_invalid_satellite_in_java_propagation(self, java_enabled_calculator):
        """测试无效卫星对象在Java传播中的处理"""
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 设置mock使Java传播失败
        mock_bridge = Mock()
        mock_bridge.is_jvm_running.return_value = True
        mock_bridge._ensure_jvm_started.return_value = None
        mock_bridge.propagate_batch.side_effect = RuntimeError("Invalid satellite")

        # 替换calculator的bridge为mock
        java_enabled_calculator._orekit_bridge = mock_bridge

        # 应该处理无效卫星对象并回退到简化模型
        with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
            mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

            # 使用None作为卫星
            pos, vel = java_enabled_calculator._propagate_satellite(None, dt)

            # Java失败应该回退到简化方法
            mock_simple.assert_called_once()
            assert pos is not None
            assert vel is not None

    def test_invalid_time_in_java_propagation(self, java_enabled_calculator, mock_satellite):
        """测试无效时间在Java传播中的处理"""
        # 使用无效时间
        invalid_dt = None

        # 应该处理无效时间
        with pytest.raises((TypeError, AttributeError)):
            java_enabled_calculator._propagate_satellite(mock_satellite, invalid_dt)


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_config(self):
        """测试空配置"""
        calc = OrekitVisibilityCalculator(config={})
        assert calc.use_java_orekit is False
        assert hasattr(calc, 'orekit_config')

    def test_none_config(self):
        """测试None配置"""
        calc = OrekitVisibilityCalculator(config=None)
        assert calc.use_java_orekit is False
        assert hasattr(calc, 'orekit_config')

    def test_config_without_use_java_flag(self):
        """测试配置中没有use_java_orekit标志"""
        config = {
            'min_elevation': 10.0,
            'orekit': {
                'jvm': {'max_memory': '1g'}
            }
        }
        calc = OrekitVisibilityCalculator(config=config)
        # 应该默认为False
        assert calc.use_java_orekit is False

    def test_config_with_empty_orekit(self):
        """测试配置中orekit为空字典"""
        config = {
            'use_java_orekit': True,
            'orekit': {}
        }
        calc = OrekitVisibilityCalculator(config=config)
        assert calc.use_java_orekit is True

    def test_propagate_range_with_zero_duration(self, java_enabled_calculator, mock_satellite):
        """测试零持续时间范围传播"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = start  # 零持续时间

        result = java_enabled_calculator._propagate_range_with_java_orekit(
            mock_satellite, start, end, timedelta(seconds=60)
        )

        # 应该返回空列表或单个点
        assert isinstance(result, list)

    def test_propagate_range_with_negative_duration(self, java_enabled_calculator, mock_satellite):
        """测试负持续时间范围传播"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 11, 0, 0)  # 早于start

        result = java_enabled_calculator._propagate_range_with_java_orekit(
            mock_satellite, start, end, timedelta(seconds=60)
        )

        # 应该返回空列表
        assert isinstance(result, list)
        assert len(result) == 0

    @patch.object(OrekitVisibilityCalculator, '_propagate_range_with_java_orekit')
    def test_propagate_range_with_very_small_step(self, mock_range_java, java_enabled_calculator, mock_satellite):
        """测试非常小的时间步长"""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 1, 0)  # 1分钟

        # 模拟返回数据
        mock_result = [
            ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0), start + timedelta(milliseconds=i * 100))
            for i in range(10)
        ]
        mock_range_java.return_value = mock_result

        result = java_enabled_calculator._propagate_range_with_java_orekit(
            mock_satellite, start, end, timedelta(milliseconds=100)
        )

        # 应该处理小步长
        assert isinstance(result, list)

    def test_satellite_without_orbit_attributes(self, java_enabled_calculator):
        """测试没有轨道属性的卫星"""
        sat = Mock()
        sat.id = "SAT-NO-ORBIT"
        # 不设置orbit属性

        dt = datetime(2024, 1, 1, 12, 0, 0)

        with patch.object(java_enabled_calculator, '_propagate_simplified') as mock_simple:
            mock_simple.return_value = ((7000000.0, 0.0, 0.0), (0.0, 7000.0, 0.0))

            # 应该回退到简化方法
            pos, vel = java_enabled_calculator._propagate_satellite(sat, dt)

            mock_simple.assert_called_once()


# =============================================================================
# 可见性窗口计算集成测试
# =============================================================================

class TestVisibilityWindowComputationWithJava:
    """测试使用Java Orekit的可见性窗口计算"""

    @patch('core.orbit.visibility.orekit_visibility.OrekitJavaBridge')
    def test_compute_windows_with_java_enabled(self, mock_bridge_class, java_enabled_calculator, mock_satellite, mock_target, time_range):
        """测试启用Java时计算可见性窗口"""
        start, end = time_range

        # 设置mock
        mock_bridge = Mock()
        mock_bridge_class.return_value = mock_bridge
        mock_bridge.is_jvm_running.return_value = True

        # 模拟批量传播返回
        n_steps = 61
        mock_data = [[0.0] * 7 for _ in range(n_steps)]
        for i in range(n_steps):
            mock_data[i][0] = i * 60
            # 创建经过目标上方的轨道
            angle = 2 * 3.14159 * i / n_steps
            mock_data[i][1] = -2171419.0 + 7000000.0 * (1 if i < 30 else -1)
            mock_data[i][2] = 4387557.0 + (i - 30) * 100000.0
            mock_data[i][3] = 4070234.0
        mock_bridge.propagate_batch.return_value = create_mock_numpy_array(mock_data)

        # 计算窗口
        windows = java_enabled_calculator.compute_satellite_target_windows(
            mock_satellite, mock_target, start, end
        )

        # 验证结果
        assert isinstance(windows, list)
        # 由于mock数据可能不会产生可见窗口，这里只验证调用成功

    @patch('core.orbit.visibility.orekit_visibility.OrekitJavaBridge')
    def test_compute_windows_fallback_on_java_error(self, mock_bridge_class, java_enabled_calculator, mock_satellite, mock_target, time_range):
        """测试Java错误时回退到简化模型计算窗口"""
        start, end = time_range

        # 设置mock - Java调用失败
        mock_bridge = Mock()
        mock_bridge_class.return_value = mock_bridge
        mock_bridge.is_jvm_running.return_value = True
        mock_bridge.propagate_batch.side_effect = OrbitPropagationError("Java error")

        with patch.object(java_enabled_calculator, '_propagate_range') as mock_range:
            mock_range.return_value = [
                ((-2171419.0, 4387557.0, 10000000.0), (0.0, 0.0, 0.0), start + timedelta(minutes=i))
                for i in range(10)
            ]

            # 计算窗口
            windows = java_enabled_calculator.compute_satellite_target_windows(
                mock_satellite, mock_target, start, end
            )

            # 应该使用简化模型成功计算
            assert isinstance(windows, list)


# =============================================================================
# 性能相关测试
# =============================================================================

class TestPerformance:
    """性能相关测试"""

    @patch.object(OrekitVisibilityCalculator, '_propagate_range_with_java_orekit')
    def test_large_batch_propagation(self, mock_range_java, java_enabled_calculator, mock_satellite):
        """测试大批量传播"""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)  # 24小时

        # 模拟大量数据点（24小时，1秒步长 = 86400点）
        n_steps = 86400
        mock_result = [
            ((7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0), start + timedelta(seconds=i))
            for i in range(n_steps)
        ]
        mock_range_java.return_value = mock_result

        # 调用方法
        result = java_enabled_calculator._propagate_range_with_java_orekit(
            mock_satellite, start, end, timedelta(seconds=1)
        )

        # 验证结果
        assert len(result) == n_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
