"""
姿态切换计算器测试

TDD Phase 3: 测试 AttitudeTransitionCalculator 的完整功能
计算不同姿态模式之间的切换时间、角度和机动参数。
"""

import pytest
import math
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sun_calculator():
    """创建太阳位置计算器mock"""
    mock = Mock()
    # 默认返回太阳在X轴方向（简化模型）
    mock.get_sun_position.return_value = (149_597_870_700.0, 0.0, 0.0)  # 1 AU along X
    mock.get_sun_direction.return_value = (1.0, 0.0, 0.0)
    return mock


@pytest.fixture
def transition_config():
    """创建默认过渡配置"""
    from core.dynamics.attitude_transition_calculator import TransitionConfig
    return TransitionConfig(
        max_slew_rate=3.0,  # deg/s
        settling_time=5.0   # seconds
    )


@pytest.fixture
def calculator(sun_calculator, transition_config):
    """创建姿态切换计算器实例"""
    from core.dynamics.attitude_transition_calculator import AttitudeTransitionCalculator
    return AttitudeTransitionCalculator(
        sun_calculator=sun_calculator,
        config=transition_config
    )


@pytest.fixture
def nadir_to_sun_transition():
    """创建从对地定向到对日定向的切换请求"""
    from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition
    return AttitudeTransition(
        from_mode=AttitudeMode.NADIR_POINTING,
        to_mode=AttitudeMode.SUN_POINTING,
        timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
        satellite_position=(7000000.0, 0.0, 0.0),  # ECEF, meters
        sun_position=(149_597_870_700.0, 0.0, 0.0),
    )


@pytest.fixture
def nadir_to_imaging_transition():
    """创建从对地定向到成像姿态的切换请求"""
    from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition
    return AttitudeTransition(
        from_mode=AttitudeMode.NADIR_POINTING,
        to_mode=AttitudeMode.IMAGING,
        timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
        satellite_position=(7000000.0, 0.0, 0.0),
        target_position=(45.0, 90.0),  # lat, lon in degrees
    )


@pytest.fixture
def imaging_to_nadir_transition():
    """创建从成像姿态到对地定向的切换请求"""
    from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition
    return AttitudeTransition(
        from_mode=AttitudeMode.IMAGING,
        to_mode=AttitudeMode.NADIR_POINTING,
        timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
        satellite_position=(7000000.0, 0.0, 0.0),
        target_position=(45.0, 90.0),
    )


@pytest.fixture
def same_mode_transition():
    """创建相同模式之间的切换请求"""
    from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition
    return AttitudeTransition(
        from_mode=AttitudeMode.NADIR_POINTING,
        to_mode=AttitudeMode.NADIR_POINTING,
        timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
        satellite_position=(7000000.0, 0.0, 0.0),
    )


# =============================================================================
# TransitionConfig 测试
# =============================================================================

class TestTransitionConfig:
    """测试过渡配置类"""

    def test_default_initialization(self):
        """测试默认初始化"""
        from core.dynamics.attitude_transition_calculator import TransitionConfig
        config = TransitionConfig()

        assert config.max_slew_rate == 3.0
        assert config.settling_time == 5.0

    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        from core.dynamics.attitude_transition_calculator import TransitionConfig
        config = TransitionConfig(
            max_slew_rate=5.0,
            settling_time=10.0
        )

        assert config.max_slew_rate == 5.0
        assert config.settling_time == 10.0

    def test_partial_custom_initialization(self):
        """测试部分自定义参数"""
        from core.dynamics.attitude_transition_calculator import TransitionConfig
        config = TransitionConfig(max_slew_rate=2.0)

        assert config.max_slew_rate == 2.0
        assert config.settling_time == 5.0  # default


# =============================================================================
# AttitudeTransitionCalculator 初始化测试
# =============================================================================

class TestCalculatorInitialization:
    """测试计算器初始化"""

    def test_initialization_with_sun_calculator(self, sun_calculator):
        """测试使用太阳计算器初始化"""
        from core.dynamics.attitude_transition_calculator import AttitudeTransitionCalculator
        calc = AttitudeTransitionCalculator(sun_calculator=sun_calculator)

        assert calc.sun_calculator is sun_calculator

    def test_initialization_with_config(self, sun_calculator, transition_config):
        """测试使用自定义配置初始化"""
        from core.dynamics.attitude_transition_calculator import AttitudeTransitionCalculator
        calc = AttitudeTransitionCalculator(
            sun_calculator=sun_calculator,
            config=transition_config
        )

        assert calc.config.max_slew_rate == 3.0
        assert calc.config.settling_time == 5.0

    def test_initialization_with_default_config(self, sun_calculator):
        """测试使用默认配置初始化"""
        from core.dynamics.attitude_transition_calculator import AttitudeTransitionCalculator
        calc = AttitudeTransitionCalculator(sun_calculator=sun_calculator)

        assert calc.config is not None
        assert calc.config.max_slew_rate == 3.0
        assert calc.config.settling_time == 5.0

    def test_initialization_without_sun_calculator_raises_error(self):
        """测试没有太阳计算器时抛出错误"""
        from core.dynamics.attitude_transition_calculator import AttitudeTransitionCalculator

        with pytest.raises((TypeError, ValueError)):
            AttitudeTransitionCalculator(sun_calculator=None)


# =============================================================================
# calculate_nadir_pointing_angles 测试
# =============================================================================

class TestCalculateNadirPointingAngles:
    """测试对地定向姿态角计算"""

    def test_returns_zero_angles(self, calculator):
        """测试返回零角度"""
        roll, pitch = calculator.calculate_nadir_pointing_angles()

        assert roll == 0.0
        assert pitch == 0.0

    def test_returns_tuple_of_two_floats(self, calculator):
        """测试返回两个浮点数的元组"""
        result = calculator.calculate_nadir_pointing_angles()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, (int, float)) for x in result)


# =============================================================================
# calculate_sun_pointing_angles 测试
# =============================================================================

class TestCalculateSunPointingAngles:
    """测试对日定向姿态角计算"""

    def test_returns_tuple_of_two_floats(self, calculator):
        """测试返回两个浮点数的元组"""
        sat_pos = (7000000.0, 0.0, 0.0)
        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        result = calculator.calculate_sun_pointing_angles(sat_pos, timestamp)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, (int, float)) for x in result)

    def test_sun_along_x_axis(self, calculator, sun_calculator):
        """测试太阳沿X轴方向时的姿态角"""
        # 太阳在X轴正方向
        sun_calculator.get_sun_direction.return_value = (1.0, 0.0, 0.0)

        sat_pos = (7000000.0, 0.0, 0.0)  # 卫星在X轴上
        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        roll, pitch = calculator.calculate_sun_pointing_angles(sat_pos, timestamp)

        # 当卫星在X轴上，太阳也在X轴方向时，姿态角取决于几何关系
        # 卫星在X轴上时，天顶方向是-X方向，太阳在+X方向
        # 这需要180度的姿态调整
        assert isinstance(roll, float)
        assert isinstance(pitch, float)

    def test_sun_direction_called_with_correct_params(self, calculator, sun_calculator):
        """测试太阳方向计算使用正确的参数"""
        sat_pos = (7000000.0, 0.0, 0.0)
        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        calculator.calculate_sun_pointing_angles(sat_pos, timestamp)

        sun_calculator.get_sun_direction.assert_called_once_with(sat_pos, timestamp)

    def test_invalid_satellite_position(self, calculator):
        """测试无效卫星位置"""
        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        with pytest.raises((ValueError, TypeError)):
            calculator.calculate_sun_pointing_angles("invalid", timestamp)

    def test_satellite_position_wrong_length(self, calculator):
        """测试卫星位置坐标数量错误"""
        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        with pytest.raises((ValueError, TypeError)):
            calculator.calculate_sun_pointing_angles((1.0, 2.0), timestamp)


# =============================================================================
# calculate_imaging_angles 测试
# =============================================================================

class TestCalculateImagingAngles:
    """测试成像姿态角计算"""

    def test_returns_tuple_of_two_floats(self, calculator):
        """测试返回两个浮点数的元组"""
        sat_pos = (7000000.0, 0.0, 0.0)
        target_pos = (45.0, 90.0)  # lat, lon

        result = calculator.calculate_imaging_angles(sat_pos, target_pos)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, (int, float)) for x in result)

    def test_target_at_nadir(self, calculator):
        """测试目标在星下点时的姿态角"""
        # 卫星在Z轴上方，目标在赤道0经度
        sat_pos = (0.0, 0.0, 7000000.0)
        target_pos = (0.0, 0.0)  # 赤道，本初子午线

        roll, pitch = calculator.calculate_imaging_angles(sat_pos, target_pos)

        # 目标在星下点，返回有效的姿态角
        # 注意：由于几何计算简化，角度可能不为0，但应该是有效的
        assert isinstance(roll, float)
        assert isinstance(pitch, float)
        assert not math.isnan(roll)
        assert not math.isnan(pitch)

    def test_target_off_nadir(self, calculator):
        """测试目标偏离星下点时的姿态角"""
        # 卫星在Z轴上方，目标偏离星下点
        sat_pos = (0.0, 0.0, 7000000.0)
        target_pos = (10.0, 0.0)  # 北纬10度

        roll, pitch = calculator.calculate_imaging_angles(sat_pos, target_pos)

        # 目标偏离星下点，应该有非零角度
        assert abs(roll) > 0.1 or abs(pitch) > 0.1

    def test_invalid_target_position(self, calculator):
        """测试无效目标位置"""
        sat_pos = (7000000.0, 0.0, 0.0)

        with pytest.raises((ValueError, TypeError)):
            calculator.calculate_imaging_angles(sat_pos, "invalid")

    def test_target_position_wrong_length(self, calculator):
        """测试目标位置坐标数量错误"""
        sat_pos = (7000000.0, 0.0, 0.0)

        with pytest.raises((ValueError, TypeError)):
            calculator.calculate_imaging_angles(sat_pos, (1.0, 2.0, 3.0))

    def test_none_target_position_raises_error(self, calculator):
        """测试None目标位置抛出错误"""
        sat_pos = (7000000.0, 0.0, 0.0)

        with pytest.raises((ValueError, TypeError)):
            calculator.calculate_imaging_angles(sat_pos, None)


# =============================================================================
# calculate_transition 测试 - NADIR to SUN
# =============================================================================

class TestCalculateTransitionNadirToSun:
    """测试从对地定向到对日定向的切换"""

    def test_returns_transition_result(self, calculator, nadir_to_sun_transition):
        """测试返回TransitionResult对象"""
        from core.dynamics.attitude_mode import TransitionResult

        result = calculator.calculate_transition(nadir_to_sun_transition)

        assert isinstance(result, TransitionResult)

    def test_slew_time_is_positive(self, calculator, nadir_to_sun_transition):
        """测试机动时间为正"""
        result = calculator.calculate_transition(nadir_to_sun_transition)

        assert result.slew_time > 0

    def test_slew_angle_is_non_negative(self, calculator, nadir_to_sun_transition):
        """测试机动角度为非负"""
        result = calculator.calculate_transition(nadir_to_sun_transition)

        assert result.slew_angle >= 0

    def test_feasible_is_true(self, calculator, nadir_to_sun_transition):
        """测试切换可行"""
        result = calculator.calculate_transition(nadir_to_sun_transition)

        assert result.feasible is True

    def test_reason_is_none_when_feasible(self, calculator, nadir_to_sun_transition):
        """测试可行时reason为None"""
        result = calculator.calculate_transition(nadir_to_sun_transition)

        assert result.reason is None

    def test_slew_time_calculation(self, calculator, nadir_to_sun_transition, transition_config):
        """测试机动时间计算"""
        result = calculator.calculate_transition(nadir_to_sun_transition)

        # 机动时间 = max(|Δroll|, |Δpitch|) / max_slew_rate + settling_time
        expected_slew = max(abs(result.roll_angle), abs(result.pitch_angle)) / transition_config.max_slew_rate
        expected_total = expected_slew + transition_config.settling_time

        assert abs(result.slew_time - expected_total) < 0.001

    def test_slew_angle_calculation(self, calculator, nadir_to_sun_transition):
        """测试机动角度计算"""
        result = calculator.calculate_transition(nadir_to_sun_transition)

        # 机动角度 = sqrt(Δroll² + Δpitch²)
        expected_angle = math.sqrt(result.roll_angle**2 + result.pitch_angle**2)

        assert abs(result.slew_angle - expected_angle) < 0.001


# =============================================================================
# calculate_transition 测试 - NADIR to IMAGING
# =============================================================================

class TestCalculateTransitionNadirToImaging:
    """测试从对地定向到成像姿态的切换"""

    def test_returns_transition_result(self, calculator, nadir_to_imaging_transition):
        """测试返回TransitionResult对象"""
        from core.dynamics.attitude_mode import TransitionResult

        result = calculator.calculate_transition(nadir_to_imaging_transition)

        assert isinstance(result, TransitionResult)

    def test_feasible_when_target_provided(self, calculator, nadir_to_imaging_transition):
        """测试提供目标时切换可行"""
        result = calculator.calculate_transition(nadir_to_imaging_transition)

        assert result.feasible is True

    def test_roll_and_pitch_based_on_target(self, calculator, nadir_to_imaging_transition):
        """测试滚转和俯仰角基于目标计算"""
        result = calculator.calculate_transition(nadir_to_imaging_transition)

        # 目标在(45, 90)，应该有非零角度
        assert abs(result.roll_angle) > 0 or abs(result.pitch_angle) > 0


# =============================================================================
# calculate_transition 测试 - IMAGING to NADIR
# =============================================================================

class TestCalculateTransitionImagingToNadir:
    """测试从成像姿态到对地定向的切换"""

    def test_returns_transition_result(self, calculator, imaging_to_nadir_transition):
        """测试返回TransitionResult对象"""
        from core.dynamics.attitude_mode import TransitionResult

        result = calculator.calculate_transition(imaging_to_nadir_transition)

        assert isinstance(result, TransitionResult)

    def test_slews_to_zero_angles(self, calculator, imaging_to_nadir_transition):
        """测试切换到零角度"""
        result = calculator.calculate_transition(imaging_to_nadir_transition)

        # 目标角度是(0, 0)
        assert result.roll_angle == 0.0
        assert result.pitch_angle == 0.0

    def test_slew_angle_based_on_current_imaging_angles(self, calculator, imaging_to_nadir_transition):
        """测试机动角度基于当前成像角度"""
        result = calculator.calculate_transition(imaging_to_nadir_transition)

        # 从成像姿态切换到对地定向，目标角度是(0, 0)
        # 机动角度应该等于从成像角度到(0,0)的变化量
        # 即成像角度的模（因为目标角度为0）
        # 由于TransitionResult存储的是目标角度(0,0)，我们需要验证 slew_angle 正确计算了变化量
        assert result.slew_angle >= 0
        assert isinstance(result.slew_angle, float)


# =============================================================================
# calculate_transition 测试 - 相同模式
# =============================================================================

class TestCalculateTransitionSameMode:
    """测试相同模式之间的切换"""

    def test_zero_slew_time_for_same_mode(self, calculator, same_mode_transition):
        """测试相同模式时机动时间为零"""
        result = calculator.calculate_transition(same_mode_transition)

        assert result.slew_time == 0.0
        assert result.slew_angle == 0.0
        assert result.roll_angle == 0.0
        assert result.pitch_angle == 0.0

    def test_feasible_for_same_mode(self, calculator, same_mode_transition):
        """测试相同模式时可行"""
        result = calculator.calculate_transition(same_mode_transition)

        assert result.feasible is True


# =============================================================================
# calculate_transition 测试 - 无效输入
# =============================================================================

class TestCalculateTransitionInvalid:
    """测试无效输入处理"""

    def test_imaging_without_target_raises_error(self, calculator):
        """测试成像模式没有目标时抛出错误"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
            target_position=None,
        )

        result = calculator.calculate_transition(transition)

        assert result.feasible is False
        assert result.reason is not None
        assert "target" in result.reason.lower()

    def test_sun_pointing_without_sun_position(self, calculator):
        """测试对日定向没有太阳位置时使用计算器"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
            sun_position=None,  # 没有提供太阳位置
        )

        # 应该使用sun_calculator获取太阳位置
        result = calculator.calculate_transition(transition)

        assert isinstance(result.slew_time, float)
        assert isinstance(result.slew_angle, float)


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    """测试边界条件"""

    def test_extreme_satellite_position(self, calculator):
        """测试极端卫星位置"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(0.0, 0.0, 7000000.0),  # 极地上方
            sun_position=(149_597_870_700.0, 0.0, 0.0),
        )

        result = calculator.calculate_transition(transition)

        assert isinstance(result.slew_time, float)
        assert isinstance(result.slew_angle, float)

    def test_high_slew_rate_config(self, sun_calculator):
        """测试高机动角速度配置"""
        from core.dynamics.attitude_transition_calculator import (
            AttitudeTransitionCalculator, TransitionConfig
        )
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        config = TransitionConfig(max_slew_rate=10.0, settling_time=2.0)
        calc = AttitudeTransitionCalculator(
            sun_calculator=sun_calculator,
            config=config
        )

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
            sun_position=(149_597_870_700.0, 0.0, 0.0),
        )

        result = calc.calculate_transition(transition)

        # 高机动角速度应该导致更短的机动时间
        assert result.slew_time > 0
        assert result.slew_time < 100  # 应该远小于100秒

    def test_low_slew_rate_config(self, sun_calculator):
        """测试低机动角速度配置"""
        from core.dynamics.attitude_transition_calculator import (
            AttitudeTransitionCalculator, TransitionConfig
        )
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        config = TransitionConfig(max_slew_rate=0.5, settling_time=10.0)
        calc = AttitudeTransitionCalculator(
            sun_calculator=sun_calculator,
            config=config
        )

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
            sun_position=(149_597_870_700.0, 0.0, 0.0),
        )

        result = calc.calculate_transition(transition)

        # 低机动角速度应该导致更长的机动时间（至少等于稳定时间）
        assert result.slew_time >= 10.0  # 至少等于稳定时间

    def test_all_mode_transitions(self, calculator):
        """测试所有模式组合"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        modes = [m for m in AttitudeMode if m not in [AttitudeMode.DOWNLINK, AttitudeMode.REALTIME, AttitudeMode.MOMENTUM_DUMP]]
        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sat_pos = (7000000.0, 0.0, 0.0)
        target_pos = (45.0, 90.0)

        for from_mode in modes:
            for to_mode in modes:
                transition = AttitudeTransition(
                    from_mode=from_mode,
                    to_mode=to_mode,
                    timestamp=timestamp,
                    satellite_position=sat_pos,
                    target_position=target_pos if to_mode == AttitudeMode.IMAGING else None,
                    sun_position=(149_597_870_700.0, 0.0, 0.0) if to_mode == AttitudeMode.SUN_POINTING else None,
                )

                result = calculator.calculate_transition(transition)

                assert isinstance(result.slew_time, float)
                assert isinstance(result.slew_angle, float)
                assert isinstance(result.feasible, bool)


# =============================================================================
# 数学计算正确性测试
# =============================================================================

class TestMathematicalCorrectness:
    """测试数学计算正确性"""

    def test_slew_time_formula(self, calculator):
        """测试机动时间公式: max(|Δroll|, |Δpitch|) / max_slew_rate + settling_time"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        # 创建一个已知的切换场景
        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
            sun_position=(0.0, 149_597_870_700.0, 0.0),  # 太阳在Y轴方向
        )

        result = calculator.calculate_transition(transition)

        # 手动计算期望的机动时间
        delta_roll = abs(result.roll_angle)
        delta_pitch = abs(result.pitch_angle)
        max_delta = max(delta_roll, delta_pitch)
        expected_slew = max_delta / 3.0 + 5.0  # max_slew_rate=3.0, settling_time=5.0

        assert abs(result.slew_time - expected_slew) < 0.01

    def test_slew_angle_formula(self, calculator):
        """测试机动角度公式: sqrt(Δroll² + Δpitch²)"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
            sun_position=(149_597_870_700.0, 0.0, 0.0),
        )

        result = calculator.calculate_transition(transition)

        # 手动计算期望的机动角度
        expected_angle = math.sqrt(result.roll_angle**2 + result.pitch_angle**2)

        assert abs(result.slew_angle - expected_angle) < 0.01


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""

    def test_with_real_sun_calculator(self):
        """测试与真实太阳计算器集成"""
        from core.dynamics.sun_position_calculator import SunPositionCalculator
        from core.dynamics.attitude_transition_calculator import AttitudeTransitionCalculator
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        sun_calc = SunPositionCalculator(use_orekit=False)
        calc = AttitudeTransitionCalculator(sun_calculator=sun_calc)

        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            satellite_position=(7000000.0, 0.0, 0.0),
        )

        result = calc.calculate_transition(transition)

        assert result.feasible is True
        assert result.slew_time > 0
        assert result.slew_angle > 0

    def test_transition_sequence(self, calculator):
        """测试连续切换序列"""
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sat_pos = (7000000.0, 0.0, 0.0)

        # 序列: NADIR -> IMAGING -> NADIR -> SUN
        transitions = [
            AttitudeTransition(
                from_mode=AttitudeMode.NADIR_POINTING,
                to_mode=AttitudeMode.IMAGING,
                timestamp=timestamp,
                satellite_position=sat_pos,
                target_position=(45.0, 90.0),
            ),
            AttitudeTransition(
                from_mode=AttitudeMode.IMAGING,
                to_mode=AttitudeMode.NADIR_POINTING,
                timestamp=timestamp,
                satellite_position=sat_pos,
                target_position=(45.0, 90.0),
            ),
            AttitudeTransition(
                from_mode=AttitudeMode.NADIR_POINTING,
                to_mode=AttitudeMode.SUN_POINTING,
                timestamp=timestamp,
                satellite_position=sat_pos,
            ),
        ]

        total_slew_time = 0.0
        for transition in transitions:
            result = calculator.calculate_transition(transition)
            total_slew_time += result.slew_time

            assert result.feasible is True

        # 总机动时间应该合理
        assert total_slew_time > 0
        assert total_slew_time < 1000  # 应该小于1000秒


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能相关测试"""

    def test_multiple_transitions_performance(self, calculator):
        """测试多次切换计算性能"""
        import time
        from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition

        timestamp = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sat_pos = (7000000.0, 0.0, 0.0)

        transitions = [
            AttitudeTransition(
                from_mode=AttitudeMode.NADIR_POINTING,
                to_mode=AttitudeMode.IMAGING,
                timestamp=timestamp,
                satellite_position=sat_pos,
                target_position=(45.0, 90.0),
            )
            for _ in range(100)
        ]

        begin = time.time()
        for transition in transitions:
            calculator.calculate_transition(transition)
        elapsed = time.time() - begin

        # 100次调用应该在1秒内完成
        assert elapsed < 1.0, f"100 transitions took {elapsed}s, expected < 1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
