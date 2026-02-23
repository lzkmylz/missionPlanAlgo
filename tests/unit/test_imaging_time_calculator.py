"""
成像时间计算器测试

TDD测试文件 - 修复EDD调度器硬编码成像时长问题
"""

import pytest
from datetime import datetime, timedelta

from core.models import Target, TargetType, ImagingMode
from payload.imaging_time_calculator import ImagingTimeCalculator


class TestImagingTimeCalculator:
    """测试成像时间计算器"""

    def test_calculator_initialization(self):
        """测试计算器初始化"""
        calculator = ImagingTimeCalculator()
        assert calculator is not None
        assert calculator.default_duration == 300  # 默认5分钟

    def test_calculate_point_target_duration(self):
        """测试点目标成像时长计算"""
        calculator = ImagingTimeCalculator()

        target = Target(
            id="TARGET-01",
            name="点目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        # 光学推扫模式
        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM)
        assert duration > 0
        assert duration <= 600  # 应该小于10分钟

    def test_calculate_area_target_duration(self):
        """测试区域目标成像时长计算"""
        calculator = ImagingTimeCalculator()

        # 创建100平方公里的区域目标（使用顶点定义）
        target = Target(
            id="AREA-01",
            name="区域目标",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (117.0, 39.0),
                (117.0, 40.0),
                (116.0, 40.0),
            ]
        )

        # SAR条带模式
        duration = calculator.calculate(target, ImagingMode.STRIPMAP)
        assert duration > 0
        # 区域越大，时间越长
        assert duration >= 300  # 至少5分钟

    def test_different_modes_have_different_durations(self):
        """测试不同成像模式有不同的时长"""
        calculator = ImagingTimeCalculator()

        target = Target(
            id="TARGET-01",
            name="目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        durations = {}
        for mode in [ImagingMode.PUSH_BROOM, ImagingMode.FRAME,
                     ImagingMode.SPOTLIGHT, ImagingMode.STRIPMAP]:
            durations[mode] = calculator.calculate(target, mode)

        # 聚束模式通常比条带模式时间长
        assert durations[ImagingMode.SPOTLIGHT] >= durations[ImagingMode.STRIPMAP]

    def test_area_size_affects_duration(self):
        """测试区域大小影响成像时长"""
        calculator = ImagingTimeCalculator()

        # 小区域目标（约10平方公里）
        small_target = Target(
            id="SMALL-01",
            name="小区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (116.3, 39.0),
                (116.3, 39.3),
                (116.0, 39.3),
            ]
        )

        # 大区域目标（约1000平方公里）
        large_target = Target(
            id="LARGE-01",
            name="大区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (119.0, 39.0),
                (119.0, 42.0),
                (116.0, 42.0),
            ]
        )

        small_duration = calculator.calculate(small_target, ImagingMode.STRIPMAP)
        large_duration = calculator.calculate(large_target, ImagingMode.STRIPMAP)

        # 大区域应该需要更长时间
        assert large_duration > small_duration

    def test_minimum_duration(self):
        """测试最小成像时长"""
        calculator = ImagingTimeCalculator(min_duration=60)

        target = Target(
            id="TARGET-01",
            name="小目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM)
        assert duration >= 60  # 不小于最小值

    def test_maximum_duration(self):
        """测试最大成像时长限制"""
        calculator = ImagingTimeCalculator(max_duration=1800)

        # 超大区域目标
        target = Target(
            id="LARGE-01",
            name="超大区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (126.0, 39.0),  # 1000km宽度
                (126.0, 49.0),  # 1000km高度
                (116.0, 49.0),
            ]
        )

        duration = calculator.calculate(target, ImagingMode.STRIPMAP)
        assert duration <= 1800  # 不超过最大值（30分钟）

    def test_invalid_mode_raises_error(self):
        """测试无效成像模式抛出错误"""
        calculator = ImagingTimeCalculator()

        target = Target(
            id="TARGET-01",
            name="目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        with pytest.raises(ValueError):
            calculator.calculate(target, "invalid_mode")


class TestPowerProfile:
    """测试功率配置文件"""

    def test_power_profile_initialization(self):
        """测试功率配置文件初始化"""
        from payload.imaging_time_calculator import PowerProfile

        profile = PowerProfile()
        assert profile is not None
        assert 'imaging' in profile.coefficients
        assert 'downlink' in profile.coefficients
        assert 'slew' in profile.coefficients
        assert 'idle' in profile.coefficients

    def test_get_power_coefficient(self):
        """测试获取功率系数"""
        from payload.imaging_time_calculator import PowerProfile

        profile = PowerProfile()

        # 成像功率系数应该在0-1之间
        imaging_coef = profile.get_coefficient('imaging')
        assert 0 < imaging_coef <= 1.0

        # 空闲功率系数应该较低
        idle_coef = profile.get_coefficient('idle')
        assert 0 < idle_coef < imaging_coef

    def test_mode_specific_coefficients(self):
        """测试不同成像模式的功率系数"""
        from payload.imaging_time_calculator import PowerProfile
        from core.models import ImagingMode

        profile = PowerProfile()

        # SAR聚束模式通常功耗更高
        spotlight_coef = profile.get_coefficient_for_mode(ImagingMode.SPOTLIGHT)
        stripmap_coef = profile.get_coefficient_for_mode(ImagingMode.STRIPMAP)

        assert spotlight_coef >= stripmap_coef

    def test_custom_coefficients(self):
        """测试自定义功率系数"""
        from payload.imaging_time_calculator import PowerProfile

        custom_coefs = {
            'imaging': 0.5,
            'downlink': 0.3,
            'slew': 0.2,
            'idle': 0.05
        }

        profile = PowerProfile(custom_coefs)
        assert profile.get_coefficient('imaging') == 0.5
        assert profile.get_coefficient('idle') == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
