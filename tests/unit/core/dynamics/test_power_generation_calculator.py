"""
发电功率计算器测试

TDD测试套件 - 测试PowerGenerationCalculator的完整功能
基于余弦衰减模型计算卫星发电功率
"""

import pytest
import math
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Tuple

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_sun_calculator():
    """创建模拟的太阳位置计算器"""
    mock = Mock()
    # 默认返回从原点指向太阳的单位向量 (1, 0, 0)
    mock.get_sun_direction.return_value = (1.0, 0.0, 0.0)
    mock.get_sun_position.return_value = (1.496e11, 0.0, 0.0)  # 1 AU in meters
    return mock


@pytest.fixture
def power_config():
    """创建默认功率配置"""
    from core.dynamics.power_generation_calculator import PowerConfig
    return PowerConfig(max_power=1000.0, eclipse_power=0.0)


@pytest.fixture
def calculator(mock_sun_calculator, power_config):
    """创建默认功率计算器实例"""
    from core.dynamics.power_generation_calculator import PowerGenerationCalculator
    return PowerGenerationCalculator(
        sun_calculator=mock_sun_calculator,
        config=power_config
    )


@pytest.fixture
def sample_time():
    """样本时间"""
    return datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_satellite_position():
    """样本卫星位置 (LEO orbit, approximately 7000km from center)"""
    return (7000e3, 0.0, 0.0)  # 7000 km in ECEF


# =============================================================================
# PowerConfig 测试
# =============================================================================

class TestPowerConfig:
    """测试功率配置数据类"""

    def test_default_values(self):
        """测试默认配置值"""
        from core.dynamics.power_generation_calculator import PowerConfig
        config = PowerConfig()
        assert config.max_power == 1000.0
        assert config.eclipse_power == 0.0

    def test_custom_values(self):
        """测试自定义配置值"""
        from core.dynamics.power_generation_calculator import PowerConfig
        config = PowerConfig(max_power=2000.0, eclipse_power=50.0)
        assert config.max_power == 2000.0
        assert config.eclipse_power == 50.0

    def test_partial_custom_values(self):
        """测试部分自定义配置值"""
        from core.dynamics.power_generation_calculator import PowerConfig
        config = PowerConfig(max_power=1500.0)
        assert config.max_power == 1500.0
        assert config.eclipse_power == 0.0


# =============================================================================
# 初始化测试
# =============================================================================

class TestInitialization:
    """测试初始化功能"""

    def test_initialization_with_sun_calculator(self, mock_sun_calculator):
        """测试使用太阳计算器初始化"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator
        calc = PowerGenerationCalculator(sun_calculator=mock_sun_calculator)
        assert calc.sun_calculator is mock_sun_calculator

    def test_initialization_with_config(self, mock_sun_calculator, power_config):
        """测试使用配置初始化"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator
        calc = PowerGenerationCalculator(
            sun_calculator=mock_sun_calculator,
            config=power_config
        )
        assert calc.config.max_power == 1000.0
        assert calc.config.eclipse_power == 0.0

    def test_initialization_with_default_config(self, mock_sun_calculator):
        """测试使用默认配置初始化"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator
        calc = PowerGenerationCalculator(sun_calculator=mock_sun_calculator)
        assert calc.config.max_power == 1000.0
        assert calc.config.eclipse_power == 0.0

    def test_initialization_without_sun_calculator_raises_error(self):
        """测试没有太阳计算器时抛出错误"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator
        with pytest.raises(TypeError):
            PowerGenerationCalculator(sun_calculator=None)


# =============================================================================
# is_in_eclipse 测试
# =============================================================================

class TestIsInEclipse:
    """测试地影判断功能"""

    def test_not_in_eclipse_when_sun_visible(self, calculator, sample_time, sample_satellite_position):
        """测试太阳可见时不在地影中"""
        # 太阳在卫星+x方向，地球在原点
        # 卫星在+x方向，所以太阳可见
        result = calculator.is_in_eclipse(sample_satellite_position, sample_time)
        assert result is False

    def test_in_eclipse_when_behind_earth(self, calculator, sample_time):
        """测试卫星在地球背面时在地影中"""
        # 卫星在-x方向（地球背面），太阳在+x方向
        satellite_pos = (-7000e3, 0.0, 0.0)
        result = calculator.is_in_eclipse(satellite_pos, sample_time)
        assert result is True

    def test_not_in_eclipse_at_terminator(self, calculator, sample_time):
        """测试在晨昏线位置"""
        # 卫星在y方向，太阳在+x方向，应该可见
        satellite_pos = (0.0, 7000e3, 0.0)
        result = calculator.is_in_eclipse(satellite_pos, sample_time)
        assert result is False

    def test_eclipse_with_high_orbit(self, calculator, sample_time):
        """测试高轨道卫星（GEO高度）"""
        # GEO高度约42164km，通常不会被地球遮挡
        geo_pos = (42164e3, 0.0, 0.0)
        result = calculator.is_in_eclipse(geo_pos, sample_time)
        assert result is False

    def test_eclipse_calculation_calls_sun_calculator(self, calculator, sample_time, sample_satellite_position):
        """测试地影计算调用太阳计算器"""
        calculator.is_in_eclipse(sample_satellite_position, sample_time)
        calculator.sun_calculator.get_sun_position.assert_called_once()


# =============================================================================
# calculate_cosine_factor 测试
# =============================================================================

class TestCalculateCosineFactor:
    """测试余弦因子计算"""

    def test_sun_pointing_mode(self, calculator):
        """测试对日定向模式 - 应该接近100%"""
        from core.dynamics.attitude_mode import AttitudeMode
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(sun_direction, AttitudeMode.SUN_POINTING)
        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_nadir_pointing_with_sun_aligned(self, calculator):
        """测试对地定向模式 - 太阳与卫星-地心线对齐"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 太阳在+x方向，卫星在+x方向（面向太阳）
        # 对地定向时，帆板法向垂直于径向
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(sun_direction, AttitudeMode.NADIR_POINTING)
        # 对地定向时，帆板法向在轨道面内垂直于径向
        # 当太阳与径向对齐时，cos(90°) = 0
        assert factor == pytest.approx(0.0, abs=1e-6)

    def test_nadir_pointing_with_sun_perpendicular(self, calculator):
        """测试对地定向模式 - 太阳垂直于卫星-地心线"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 太阳在+y方向，卫星在+x方向
        # 对地定向时，帆板法向可以指向+y方向
        sun_direction = (0.0, 1.0, 0.0)
        factor = calculator.calculate_cosine_factor(sun_direction, AttitudeMode.NADIR_POINTING)
        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_imaging_mode_with_zero_angles(self, calculator):
        """测试成像模式 - 无侧摆角"""
        from core.dynamics.attitude_mode import AttitudeMode
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(
            sun_direction, AttitudeMode.IMAGING, roll_angle=0.0, pitch_angle=0.0
        )
        # 无侧摆时，成像模式等同于对地定向
        assert factor >= 0.0
        assert factor <= 1.0

    def test_imaging_mode_with_roll_angle(self, calculator):
        """测试成像模式 - 有滚转角"""
        from core.dynamics.attitude_mode import AttitudeMode
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(
            sun_direction, AttitudeMode.IMAGING, roll_angle=30.0, pitch_angle=0.0
        )
        # 有侧摆时，功率应该降低
        assert factor >= 0.0
        assert factor <= 1.0

    def test_downlink_mode(self, calculator):
        """测试数传模式"""
        from core.dynamics.attitude_mode import AttitudeMode
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(
            sun_direction, AttitudeMode.DOWNLINK, roll_angle=0.0, pitch_angle=0.0
        )
        assert factor >= 0.0
        assert factor <= 1.0

    def test_realtime_mode(self, calculator):
        """测试实传模式"""
        from core.dynamics.attitude_mode import AttitudeMode
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(
            sun_direction, AttitudeMode.REALTIME, roll_angle=0.0, pitch_angle=0.0
        )
        assert factor >= 0.0
        assert factor <= 1.0

    def test_momentum_dump_mode(self, calculator):
        """测试动量卸载模式 - 应该接近对日定向"""
        from core.dynamics.attitude_mode import AttitudeMode
        sun_direction = (1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(sun_direction, AttitudeMode.MOMENTUM_DUMP)
        # 动量卸载通常采用对日定向
        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_invalid_sun_direction_type(self, calculator):
        """测试无效的太阳方向类型"""
        from core.dynamics.attitude_mode import AttitudeMode
        with pytest.raises((TypeError, ValueError)):
            calculator.calculate_cosine_factor("invalid", AttitudeMode.SUN_POINTING)

    def test_invalid_sun_direction_length(self, calculator):
        """测试无效的太阳方向长度"""
        from core.dynamics.attitude_mode import AttitudeMode
        with pytest.raises((TypeError, ValueError)):
            calculator.calculate_cosine_factor((1.0, 0.0), AttitudeMode.SUN_POINTING)

    def test_zero_sun_direction_magnitude(self, calculator):
        """测试太阳方向向量为零向量"""
        from core.dynamics.attitude_mode import AttitudeMode
        with pytest.raises(ValueError):
            calculator.calculate_cosine_factor((0.0, 0.0, 0.0), AttitudeMode.SUN_POINTING)


# =============================================================================
# calculate_power 测试
# =============================================================================

class TestCalculatePower:
    """测试功率计算功能"""

    def test_sun_pointing_full_power(self, calculator, sample_time, sample_satellite_position):
        """测试对日定向模式 - 满功率"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            sample_satellite_position,
            sample_time
        )
        assert power == pytest.approx(1000.0, rel=0.01)

    def test_nadir_pointing_reduced_power(self, calculator, sample_time, sample_satellite_position):
        """测试对地定向模式 - 降低功率"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 设置太阳方向使其与径向垂直，获得最大功率
        calculator.sun_calculator.get_sun_direction.return_value = (0.0, 1.0, 0.0)
        power = calculator.calculate_power(
            AttitudeMode.NADIR_POINTING,
            sample_satellite_position,
            sample_time
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_eclipse_zero_power(self, calculator, sample_time):
        """测试地影期间零功率"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 卫星在地球背面
        satellite_pos = (-7000e3, 0.0, 0.0)
        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            satellite_pos,
            sample_time
        )
        assert power == 0.0

    def test_eclipse_with_custom_eclipse_power(self, mock_sun_calculator, sample_time):
        """测试地影期间使用自定义地影功率"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator, PowerConfig
        from core.dynamics.attitude_mode import AttitudeMode

        config = PowerConfig(max_power=1000.0, eclipse_power=50.0)
        calculator = PowerGenerationCalculator(
            sun_calculator=mock_sun_calculator,
            config=config
        )

        satellite_pos = (-7000e3, 0.0, 0.0)
        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            satellite_pos,
            sample_time
        )
        assert power == 50.0

    def test_imaging_mode_with_angles(self, calculator, sample_time, sample_satellite_position):
        """测试成像模式带角度"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.IMAGING,
            sample_satellite_position,
            sample_time,
            roll_angle=30.0,
            pitch_angle=15.0
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_downlink_mode_with_angles(self, calculator, sample_time, sample_satellite_position):
        """测试数传模式带角度"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.DOWNLINK,
            sample_satellite_position,
            sample_time,
            roll_angle=45.0,
            pitch_angle=0.0
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_realtime_mode(self, calculator, sample_time, sample_satellite_position):
        """测试实传模式"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.REALTIME,
            sample_satellite_position,
            sample_time,
            roll_angle=20.0,
            pitch_angle=10.0
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_momentum_dump_full_power(self, calculator, sample_time, sample_satellite_position):
        """测试动量卸载模式 - 接近满功率"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.MOMENTUM_DUMP,
            sample_satellite_position,
            sample_time
        )
        # 动量卸载通常采用对日定向，应该接近满功率
        assert power >= 900.0
        assert power <= 1000.0

    def test_invalid_attitude_mode(self, calculator, sample_time, sample_satellite_position):
        """测试无效姿态模式"""
        with pytest.raises((TypeError, ValueError)):
            calculator.calculate_power(
                "invalid_mode",
                sample_satellite_position,
                sample_time
            )

    def test_invalid_satellite_position_type(self, calculator, sample_time):
        """测试无效卫星位置类型"""
        from core.dynamics.attitude_mode import AttitudeMode
        with pytest.raises((TypeError, ValueError)):
            calculator.calculate_power(
                AttitudeMode.SUN_POINTING,
                "invalid_position",
                sample_time
            )

    def test_invalid_satellite_position_length(self, calculator, sample_time):
        """测试无效卫星位置长度"""
        from core.dynamics.attitude_mode import AttitudeMode
        with pytest.raises((TypeError, ValueError)):
            calculator.calculate_power(
                AttitudeMode.SUN_POINTING,
                (1.0, 2.0),
                sample_time
            )

    def test_invalid_timestamp_type(self, calculator, sample_satellite_position):
        """测试无效时间戳类型"""
        from core.dynamics.attitude_mode import AttitudeMode
        with pytest.raises((TypeError, ValueError)):
            calculator.calculate_power(
                AttitudeMode.SUN_POINTING,
                sample_satellite_position,
                "invalid_time"
            )

    def test_naive_timestamp_raises_error(self, calculator, sample_satellite_position):
        """测试无时区时间戳抛出错误"""
        from core.dynamics.attitude_mode import AttitudeMode
        naive_time = datetime(2024, 3, 20, 12, 0, 0)
        with pytest.raises(ValueError):
            calculator.calculate_power(
                AttitudeMode.SUN_POINTING,
                sample_satellite_position,
                naive_time
            )

    def test_power_with_different_max_power(self, mock_sun_calculator):
        """测试不同最大功率配置"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator, PowerConfig
        from core.dynamics.attitude_mode import AttitudeMode

        config = PowerConfig(max_power=2000.0)
        calculator = PowerGenerationCalculator(
            sun_calculator=mock_sun_calculator,
            config=config
        )

        sample_time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        satellite_pos = (7000e3, 0.0, 0.0)

        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            satellite_pos,
            sample_time
        )
        assert power == pytest.approx(2000.0, rel=0.01)


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    """测试边界条件"""

    def test_very_small_cosine_factor(self, calculator):
        """测试非常小的余弦因子"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 太阳方向几乎垂直于帆板法向
        sun_direction = (0.01, 0.9999, 0.0)
        factor = calculator.calculate_cosine_factor(
            sun_direction, AttitudeMode.NADIR_POINTING
        )
        assert factor >= 0.0
        assert factor <= 1.0

    def test_negative_cosine_factor_clamped(self, calculator):
        """测试负余弦因子被限制为0"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 太阳在帆板背面
        sun_direction = (-1.0, 0.0, 0.0)
        factor = calculator.calculate_cosine_factor(
            sun_direction, AttitudeMode.SUN_POINTING
        )
        # 余弦因子应该被限制在[0, 1]范围内
        assert factor >= 0.0

    def test_large_roll_angle(self, calculator, sample_time, sample_satellite_position):
        """测试大滚转角"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.IMAGING,
            sample_satellite_position,
            sample_time,
            roll_angle=85.0,
            pitch_angle=0.0
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_large_pitch_angle(self, calculator, sample_time, sample_satellite_position):
        """测试大俯仰角"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.IMAGING,
            sample_satellite_position,
            sample_time,
            roll_angle=0.0,
            pitch_angle=85.0
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_combined_large_angles(self, calculator, sample_time, sample_satellite_position):
        """测试组合大角度"""
        from core.dynamics.attitude_mode import AttitudeMode
        power = calculator.calculate_power(
            AttitudeMode.IMAGING,
            sample_satellite_position,
            sample_time,
            roll_angle=45.0,
            pitch_angle=45.0
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_extreme_satellite_altitude_low(self, calculator, sample_time):
        """测试极低轨道高度"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 近地轨道，约200km高度
        low_orbit_pos = (6578e3, 0.0, 0.0)  # 地球半径 + 200km
        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            low_orbit_pos,
            sample_time
        )
        assert power >= 0.0
        assert power <= 1000.0

    def test_extreme_satellite_altitude_high(self, calculator, sample_time):
        """测试极高轨道高度"""
        from core.dynamics.attitude_mode import AttitudeMode
        # 高椭圆轨道
        high_orbit_pos = (50000e3, 0.0, 0.0)
        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            high_orbit_pos,
            sample_time
        )
        assert power >= 0.0
        assert power <= 1000.0


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""

    def test_integration_with_real_sun_calculator(self):
        """测试与真实太阳计算器集成"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator, PowerConfig
        from core.dynamics.attitude_mode import AttitudeMode

        sun_calc = SunPositionCalculator(use_orekit=False)
        config = PowerConfig(max_power=1000.0)
        calculator = PowerGenerationCalculator(
            sun_calculator=sun_calc,
            config=config
        )

        sample_time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        satellite_pos = (7000e3, 0.0, 0.0)

        power = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            satellite_pos,
            sample_time
        )

        assert power >= 0.0
        assert power <= 1000.0

    def test_full_power_calculation_chain(self, mock_sun_calculator):
        """测试完整功率计算链"""
        from core.dynamics.power_generation_calculator import PowerGenerationCalculator, PowerConfig
        from core.dynamics.attitude_mode import AttitudeMode

        config = PowerConfig(max_power=1000.0, eclipse_power=0.0)
        calculator = PowerGenerationCalculator(
            sun_calculator=mock_sun_calculator,
            config=config
        )

        sample_time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        satellite_pos = (7000e3, 0.0, 0.0)

        # 测试所有姿态模式
        modes = [
            AttitudeMode.SUN_POINTING,
            AttitudeMode.NADIR_POINTING,
            AttitudeMode.IMAGING,
            AttitudeMode.DOWNLINK,
            AttitudeMode.REALTIME,
            AttitudeMode.MOMENTUM_DUMP,
        ]

        for mode in modes:
            power = calculator.calculate_power(mode, satellite_pos, sample_time)
            assert isinstance(power, float)
            assert power >= 0.0
            assert power <= 1000.0

    def test_sequential_calculations_consistency(self, calculator, sample_time, sample_satellite_position):
        """测试连续计算的一致性"""
        from core.dynamics.attitude_mode import AttitudeMode

        power1 = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            sample_satellite_position,
            sample_time
        )
        power2 = calculator.calculate_power(
            AttitudeMode.SUN_POINTING,
            sample_satellite_position,
            sample_time
        )

        assert power1 == pytest.approx(power2, abs=1e-6)


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能测试"""

    def test_calculation_performance(self, calculator, sample_time, sample_satellite_position):
        """测试计算性能"""
        import time
        from core.dynamics.attitude_mode import AttitudeMode

        start = time.time()
        for _ in range(1000):
            calculator.calculate_power(
                AttitudeMode.SUN_POINTING,
                sample_satellite_position,
                sample_time
            )
        elapsed = time.time() - start

        # 1000次计算应该在1秒内完成
        assert elapsed < 1.0
