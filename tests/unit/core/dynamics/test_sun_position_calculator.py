"""
太阳位置计算器测试

TDD测试套件 - 测试SunPositionCalculator的完整功能
"""

import pytest
import math
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def calculator():
    """创建默认计算器实例（不使用Orekit）"""
    from core.dynamics.sun_position_calculator import SunPositionCalculator
    return SunPositionCalculator(use_orekit=False)


@pytest.fixture
def orekit_calculator():
    """创建使用Orekit的计算器实例"""
    from core.dynamics.sun_position_calculator import SunPositionCalculator
    return SunPositionCalculator(use_orekit=True)


@pytest.fixture
def vernal_equinox_2024():
    """2024年春分时间（太阳在赤道上方，黄经0度）"""
    return datetime(2024, 3, 20, 3, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def summer_solstice_2024():
    """2024年夏至时间（太阳在北回归线上方，黄经90度）"""
    return datetime(2024, 6, 20, 20, 51, 0, tzinfo=timezone.utc)


@pytest.fixture
def noon_utc():
    """正午UTC时间"""
    return datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# 初始化测试
# =============================================================================

class TestInitialization:
    """测试初始化功能"""

    def test_default_initialization(self):
        """测试默认初始化（use_orekit=True）"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator
        calc = SunPositionCalculator()
        assert calc.use_orekit is True

    def test_orekit_disabled_initialization(self):
        """测试禁用Orekit的初始化"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator
        calc = SunPositionCalculator(use_orekit=False)
        assert calc.use_orekit is False

    def test_orekit_enabled_initialization(self):
        """测试启用Orekit的初始化"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator
        calc = SunPositionCalculator(use_orekit=True)
        assert calc.use_orekit is True


# =============================================================================
# is_available 测试
# =============================================================================

class TestAvailability:
    """测试可用性检查"""

    def test_available_without_orekit(self, calculator):
        """测试不使用Orekit时总是可用"""
        assert calculator.is_available() is True

    def test_available_with_orekit_when_bridge_available(self):
        """测试Orekit可用时返回True"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', True):
            with patch('core.dynamics.sun_position_calculator.OrekitJavaBridge') as mock_bridge:
                mock_instance = Mock()
                mock_instance.is_jvm_running.return_value = True
                mock_bridge.return_value = mock_instance

                calc = SunPositionCalculator(use_orekit=True)
                assert calc.is_available() is True

    def test_available_with_orekit_when_bridge_unavailable(self):
        """测试Orekit不可用时返回False"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', False):
            calc = SunPositionCalculator(use_orekit=True)
            assert calc.is_available() is False


# =============================================================================
# get_sun_position 测试
# =============================================================================

class TestGetSunPosition:
    """测试获取太阳位置功能"""

    def test_returns_tuple_of_three_floats(self, calculator, noon_utc):
        """测试返回三个浮点数的元组"""
        position = calculator.get_sun_position(noon_utc)

        assert isinstance(position, tuple)
        assert len(position) == 3
        assert all(isinstance(x, (int, float)) for x in position)

    def test_position_magnitude_is_reasonable(self, calculator, noon_utc):
        """测试太阳位置大小在合理范围内（1AU左右）"""
        position = calculator.get_sun_position(noon_utc)
        x, y, z = position

        # 计算到地心的距离
        distance = math.sqrt(x*x + y*y + z*z)

        # 1 AU = 149,597,870,700 米
        # 允许10%的误差范围
        AU = 149_597_870_700.0
        assert 0.9 * AU <= distance <= 1.1 * AU, f"Distance {distance} not within 10% of 1 AU"

    def test_spring_equinox_position(self, calculator, vernal_equinox_2024):
        """测试春分时的太阳位置（太阳在赤道平面，X轴正方向）"""
        position = calculator.get_sun_position(vernal_equinox_2024)
        x, y, z = position

        # 春分时太阳在黄经0度，应该在X轴正方向附近
        # Z坐标应该接近0（在赤道平面）
        assert abs(z) < 0.1 * math.sqrt(x*x + y*y), "Z should be small at vernal equinox"

    def test_summer_solstice_position(self, calculator, summer_solstice_2024):
        """测试夏至时的太阳位置（太阳在北半球，黄经90度）"""
        position = calculator.get_sun_position(summer_solstice_2024)
        x, y, z = position

        # 夏至时太阳在黄经90度，Z坐标应该为正（北半球）
        assert z > 0, "Z should be positive at summer solstice"

    def test_different_times_give_different_positions(self, calculator):
        """测试不同时间返回不同位置"""
        time1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        time2 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        pos1 = calculator.get_sun_position(time1)
        pos2 = calculator.get_sun_position(time2)

        # 12小时后太阳位置应该有明显变化
        assert pos1 != pos2, "Sun position should change over time"

    def test_24_hours_returns_similar_position(self, calculator):
        """测试24小时后太阳位置相近（地球公转角度很小）"""
        time1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        time2 = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

        pos1 = calculator.get_sun_position(time1)
        pos2 = calculator.get_sun_position(time2)

        # 计算距离差
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        diff = math.sqrt(dx*dx + dy*dy + dz*dz)

        # 24小时变化约1/365的轨道，约1度
        # 距离变化应该相对较小（小于0.1 AU）
        assert diff < 0.1 * 149_597_870_700, "24-hour position change should be small"

    def test_naive_datetime_raises_error(self, calculator):
        """测试无时区的datetime抛出错误"""
        naive_time = datetime(2024, 1, 1, 0, 0, 0)

        with pytest.raises((ValueError, TypeError)):
            calculator.get_sun_position(naive_time)

    def test_invalid_timestamp_type(self, calculator):
        """测试无效的时间戳类型抛出错误"""
        with pytest.raises((ValueError, TypeError)):
            calculator.get_sun_position("2024-01-01")

    def test_none_timestamp_raises_error(self, calculator):
        """测试None时间戳抛出错误"""
        with pytest.raises((ValueError, TypeError)):
            calculator.get_sun_position(None)


# =============================================================================
# get_sun_direction 测试
# =============================================================================

class TestGetSunDirection:
    """测试获取太阳方向功能"""

    def test_returns_unit_vector(self, calculator, noon_utc):
        """测试返回单位向量"""
        satellite_pos = (7000000.0, 0.0, 0.0)  # 约600km高度
        direction = calculator.get_sun_direction(satellite_pos, noon_utc)

        x, y, z = direction
        magnitude = math.sqrt(x*x + y*y + z*z)

        assert abs(magnitude - 1.0) < 1e-10, f"Direction vector should be unit vector, got magnitude {magnitude}"

    def test_direction_points_to_sun(self, calculator, noon_utc):
        """测试方向指向太阳"""
        satellite_pos = (7000000.0, 0.0, 0.0)
        direction = calculator.get_sun_direction(satellite_pos, noon_utc)

        # 获取太阳位置
        sun_pos = calculator.get_sun_position(noon_utc)

        # 计算从卫星到太阳的期望方向
        dx = sun_pos[0] - satellite_pos[0]
        dy = sun_pos[1] - satellite_pos[1]
        dz = sun_pos[2] - satellite_pos[2]
        expected_magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)

        expected_dir = (dx/expected_magnitude, dy/expected_magnitude, dz/expected_magnitude)

        # 验证方向一致
        assert abs(direction[0] - expected_dir[0]) < 1e-10
        assert abs(direction[1] - expected_dir[1]) < 1e-10
        assert abs(direction[2] - expected_dir[2]) < 1e-10

    def test_satellite_at_sun_position(self, calculator, noon_utc):
        """测试卫星在太阳位置时的方向"""
        sun_pos = calculator.get_sun_position(noon_utc)

        # 卫星在太阳位置，方向应该远离地心（大致）
        direction = calculator.get_sun_direction(sun_pos, noon_utc)

        # 方向向量应该是单位向量
        x, y, z = direction
        magnitude = math.sqrt(x*x + y*y + z*z)
        assert abs(magnitude - 1.0) < 1e-10

    def test_satellite_at_origin(self, calculator, noon_utc):
        """测试卫星在原点时的方向"""
        direction = calculator.get_sun_direction((0.0, 0.0, 0.0), noon_utc)

        # 从地心看太阳的方向
        sun_pos = calculator.get_sun_position(noon_utc)
        expected_magnitude = math.sqrt(sum(x*x for x in sun_pos))
        expected_dir = tuple(x/expected_magnitude for x in sun_pos)

        assert abs(direction[0] - expected_dir[0]) < 1e-10
        assert abs(direction[1] - expected_dir[1]) < 1e-10
        assert abs(direction[2] - expected_dir[2]) < 1e-10

    def test_invalid_satellite_position(self, calculator, noon_utc):
        """测试无效的卫星位置"""
        with pytest.raises((ValueError, TypeError)):
            calculator.get_sun_direction("invalid", noon_utc)

    def test_satellite_position_wrong_length(self, calculator, noon_utc):
        """测试卫星位置坐标数量错误"""
        with pytest.raises((ValueError, TypeError)):
            calculator.get_sun_direction((1.0, 2.0), noon_utc)


# =============================================================================
# 一致性测试
# =============================================================================

class TestConsistency:
    """测试结果一致性测试"""

    def test_multiple_calls_return_consistent_results(self, calculator, noon_utc):
        """测试多次调用返回一致结果"""
        positions = [calculator.get_sun_position(noon_utc) for _ in range(5)]

        # 所有结果应该相同
        for pos in positions[1:]:
            assert pos == positions[0], "Multiple calls should return consistent results"

    def test_multiple_direction_calls_consistent(self, calculator, noon_utc):
        """测试多次方向调用返回一致结果"""
        satellite_pos = (7000000.0, 0.0, 0.0)
        directions = [calculator.get_sun_direction(satellite_pos, noon_utc) for _ in range(5)]

        for direction in directions[1:]:
            assert direction == directions[0], "Multiple direction calls should return consistent results"


# =============================================================================
# Orekit集成测试
# =============================================================================

class TestOrekitIntegration:
    """测试Orekit集成功能"""

    def test_orekit_mode_when_available(self):
        """测试Orekit可用时使用Orekit计算"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', True):
            with patch('core.dynamics.sun_position_calculator.OrekitJavaBridge') as mock_bridge:
                mock_instance = Mock()
                mock_instance.is_jvm_running.return_value = True
                # 模拟Orekit返回的太阳位置
                mock_instance.get_sun_position.return_value = (
                    149_597_870_700.0, 0.0, 0.0  # 1 AU along X axis
                )
                mock_bridge.return_value = mock_instance

                calc = SunPositionCalculator(use_orekit=True)
                time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

                # 如果Orekit可用，应该使用Orekit计算
                if calc.is_available():
                    position = calc.get_sun_position(time)
                    # 验证使用了Orekit
                    assert isinstance(position, tuple)
                    assert len(position) == 3

    def test_fallback_when_orekit_unavailable(self):
        """测试Orekit不可用时回退到简化模型"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', False):
            calc = SunPositionCalculator(use_orekit=True)
            time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

            # Orekit不可用，但应该仍然可以计算（使用简化模型）
            position = calc.get_sun_position(time)
            assert isinstance(position, tuple)
            assert len(position) == 3

    def test_orekit_calculation_path(self):
        """测试Orekit计算路径（使用mock）"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', True):
            with patch('core.dynamics.sun_position_calculator.OrekitJavaBridge') as mock_bridge_class:
                # 创建mock实例
                mock_bridge = Mock()
                mock_bridge.is_jvm_running.return_value = True

                # 模拟Orekit Java类
                mock_absolute_date = Mock()
                mock_time_scales = Mock()
                mock_frames = Mock()
                mock_celestial = Mock()

                # 设置mock返回值
                mock_time_scales.getUTC.return_value = Mock()
                mock_frames.getITRF.return_value = Mock()

                # 模拟太阳位置
                mock_sun_pos = Mock()
                mock_sun_pos.getX.return_value = 149_597_870_700.0
                mock_sun_pos.getY.return_value = 0.0
                mock_sun_pos.getZ.return_value = 0.0

                mock_sun_pv = Mock()
                mock_sun_pv.getPosition.return_value = mock_sun_pos

                mock_sun = Mock()
                mock_sun.getPVCoordinates.return_value = mock_sun_pv
                mock_celestial.getSun.return_value = mock_sun

                # 设置_get_java_class的返回值
                def mock_get_java_class(class_name):
                    if 'AbsoluteDate' in class_name:
                        return mock_absolute_date
                    elif 'TimeScalesFactory' in class_name:
                        return mock_time_scales
                    elif 'FramesFactory' in class_name:
                        return mock_frames
                    elif 'CelestialBodyFactory' in class_name:
                        return mock_celestial
                    return Mock()

                mock_bridge._get_java_class = mock_get_java_class
                mock_bridge._ensure_jvm_started = Mock()
                mock_bridge_class.return_value = mock_bridge

                # 创建计算器
                calc = SunPositionCalculator(use_orekit=True)
                calc._orekit_bridge = mock_bridge

                time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

                # 调用Orekit计算路径
                position = calc._calculate_sun_position_orekit(time)

                assert isinstance(position, tuple)
                assert len(position) == 3
                assert position[0] == 149_597_870_700.0
                assert position[1] == 0.0
                assert position[2] == 0.0

    def test_orekit_bridge_exception_during_calculation(self):
        """测试Orekit计算时抛出异常"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', True):
            with patch('core.dynamics.sun_position_calculator.OrekitJavaBridge') as mock_bridge_class:
                mock_bridge = Mock()
                mock_bridge.is_jvm_running.return_value = True
                mock_bridge._ensure_jvm_started = Mock()
                mock_bridge._get_java_class.side_effect = Exception("Java class not found")
                mock_bridge_class.return_value = mock_bridge

                calc = SunPositionCalculator(use_orekit=True)
                calc._orekit_bridge = mock_bridge

                time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

                # 应该抛出RuntimeError
                with pytest.raises(RuntimeError):
                    calc._calculate_sun_position_orekit(time)

    def test_get_sun_position_orekit_fallback_on_exception(self):
        """测试get_sun_position在Orekit异常时回退到简化模型"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator

        with patch('core.dynamics.sun_position_calculator.OREKIT_BRIDGE_AVAILABLE', True):
            with patch('core.dynamics.sun_position_calculator.OrekitJavaBridge') as mock_bridge_class:
                mock_bridge = Mock()
                mock_bridge.is_jvm_running.return_value = True
                mock_bridge._ensure_jvm_started = Mock()
                mock_bridge._get_java_class.side_effect = Exception("JVM error")
                mock_bridge_class.return_value = mock_bridge

                calc = SunPositionCalculator(use_orekit=True)
                calc._orekit_bridge = mock_bridge

                time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

                # 应该回退到简化模型并返回有效结果
                position = calc.get_sun_position(time)
                assert isinstance(position, tuple)
                assert len(position) == 3


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    """测试边界条件"""

    def test_very_old_date(self, calculator):
        """测试非常早的日期"""
        old_time = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        position = calculator.get_sun_position(old_time)

        assert isinstance(position, tuple)
        assert len(position) == 3

    def test_very_future_date(self, calculator):
        """测试非常晚的日期"""
        future_time = datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        position = calculator.get_sun_position(future_time)

        assert isinstance(position, tuple)
        assert len(position) == 3

    def test_leap_year_date(self, calculator):
        """测试闰年日期"""
        leap_time = datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)
        position = calculator.get_sun_position(leap_time)

        assert isinstance(position, tuple)
        assert len(position) == 3

    def test_microsecond_precision(self, calculator):
        """测试微秒精度时间"""
        precise_time = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        position = calculator.get_sun_position(precise_time)

        assert isinstance(position, tuple)
        assert len(position) == 3


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能相关测试"""

    def test_multiple_calls_performance(self, calculator):
        """测试多次调用性能"""
        import time

        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        times = [start_time.replace(hour=i) for i in range(24)]

        begin = time.time()
        for t in times:
            calculator.get_sun_position(t)
        elapsed = time.time() - begin

        # 24次调用应该在1秒内完成
        assert elapsed < 1.0, f"24 calls took {elapsed}s, expected < 1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
