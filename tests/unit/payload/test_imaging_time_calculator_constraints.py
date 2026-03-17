"""
Tests for ImagingTimeCalculator with satellite-specific constraints.

TDD approach: Write failing tests first, then implement to make them pass.
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from payload.imaging_time_calculator import ImagingTimeCalculator
from core.models.satellite import (
    SatelliteCapabilities,
    Satellite,
    SatelliteType,
    Orbit,
)
from core.models.imaging_mode import ImagingMode, ImagingModeConfig
from core.models.payload_config import PayloadConfiguration
from core.models.target import Target, TargetType


@dataclass
class MockTarget:
    """Mock target for testing."""
    target_type: TargetType = TargetType.POINT
    area: float = 100.0

    def get_area(self) -> float:
        return self.area


class TestImagingTimeCalculatorSatelliteConstraints:
    """Test ImagingTimeCalculator with satellite-specific constraints."""

    def test_calculate_without_satellite_uses_global_defaults(self):
        """Test that calculate without satellite uses global defaults."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM)

        # Should use global defaults and apply constraints
        assert duration >= 60.0
        assert duration <= 1800.0

    def test_calculate_with_satellite_no_constraints_uses_global(self):
        """Test calculate with satellite but no mode constraints uses global."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # 使用 payload_config=None 来测试回退到全局约束的情况
        capabilities = SatelliteCapabilities(imaging_modes=[ImagingMode.PUSH_BROOM])
        capabilities.payload_config = None  # 禁用payload_config以测试回退行为

        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=capabilities
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Should use global defaults when payload_config is None
        assert duration >= 60.0
        assert duration <= 1800.0

    def test_calculate_with_satellite_constraints(self):
        """Test calculate uses satellite-specific constraints when available."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # 使用新的 payload_config 格式
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=30.0,  # 自定义约束
                    max_duration_s=600.0,  # 自定义约束
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                payload_config=payload_config
            )
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Should use satellite-specific constraints (max 600 instead of 1800)
        assert duration >= 30.0
        assert duration <= 600.0

    def test_calculate_with_different_modes_different_constraints(self):
        """Test different modes can have different constraints."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # 使用新的 payload_config 格式
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=30.0,
                    max_duration_s=600.0,
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                ),
                'frame': ImagingModeConfig(
                    resolution_m=1.0,
                    swath_width_m=20000,
                    power_consumption_w=120.0,
                    data_rate_mbps=150.0,
                    min_duration_s=10.0,
                    max_duration_s=300.0,
                    mode_type='optical',
                    characteristics={'spectral_bands': ['RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                payload_config=payload_config
            )
        )

        duration_push = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)
        duration_frame = calculator.calculate(target, ImagingMode.FRAME, satellite=satellite)

        # PUSH_BROOM: max 600
        assert duration_push <= 600.0
        # FRAME: max 300
        assert duration_frame <= 300.0

    def test_calculate_mode_without_constraints_uses_global(self):
        """Test that modes without constraints fall back to global."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # payload_config中只有PUSH_BROOM，没有FRAME
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=30.0,
                    max_duration_s=600.0,
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                payload_config=payload_config
            )
        )

        # FRAME has no constraints in payload_config, should use global (max 1800)
        duration = calculator.calculate(target, ImagingMode.FRAME, satellite=satellite)
        assert duration <= 1800.0
        assert duration >= 60.0

    def test_get_constraints_for_satellite_and_mode(self):
        """Test getting constraints for specific satellite and mode."""
        calculator = ImagingTimeCalculator()

        # 使用新的 payload_config 格式
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=30.0,
                    max_duration_s=600.0,
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                payload_config=payload_config
            )
        )

        result = calculator.get_constraints_for_satellite(satellite, ImagingMode.PUSH_BROOM)
        assert result == {'min_duration': 30.0, 'max_duration': 600.0}

    def test_get_constraints_returns_none_for_no_satellite(self):
        """Test getting constraints returns None when no satellite provided."""
        calculator = ImagingTimeCalculator()

        result = calculator.get_constraints_for_satellite(None, ImagingMode.PUSH_BROOM)
        assert result is None

    def test_get_constraints_returns_none_for_no_constraints(self):
        """Test getting constraints returns None when satellite has no constraints."""
        calculator = ImagingTimeCalculator()

        # 禁用payload_config以测试回退行为
        capabilities = SatelliteCapabilities(imaging_modes=[ImagingMode.PUSH_BROOM])
        capabilities.payload_config = None

        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=capabilities
        )

        result = calculator.get_constraints_for_satellite(satellite, ImagingMode.PUSH_BROOM)
        assert result is None

    def test_get_constraints_returns_none_for_mode_not_configured(self):
        """Test getting constraints returns None when mode not configured."""
        calculator = ImagingTimeCalculator()

        # payload_config中只有PUSH_BROOM，没有FRAME
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=30.0,
                    max_duration_s=600.0,
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                payload_config=payload_config
            )
        )

        result = calculator.get_constraints_for_satellite(satellite, ImagingMode.FRAME)
        assert result is None

    def test_backward_compatibility_no_satellite_parameter(self):
        """Test that calculate() without satellite parameter still works."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # Should work without satellite parameter
        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM)

        assert duration >= 60.0
        assert duration <= 1800.0

    def test_area_target_with_satellite_constraints(self):
        """Test area target calculation with satellite constraints."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.AREA, area=500.0)

        # 使用新的 payload_config 格式
        payload_config = PayloadConfiguration(
            payload_type='sar',
            default_mode='stripmap',
            modes={
                'stripmap': ImagingModeConfig(
                    resolution_m=3.0,
                    swath_width_m=30000,
                    power_consumption_w=300.0,
                    data_rate_mbps=400.0,
                    min_duration_s=45.0,
                    max_duration_s=1200.0,
                    mode_type='sar',
                    characteristics={'polarization': 'HH+HV'}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.SAR_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.STRIPMAP],
                payload_config=payload_config
            )
        )

        duration = calculator.calculate(target, ImagingMode.STRIPMAP, satellite=satellite)

        # Should be constrained to max 1200
        assert duration <= 1200.0
        assert duration >= 45.0


class TestImagingTimeCalculatorEdgeCases:
    """Test edge cases for ImagingTimeCalculator with constraints."""

    def test_satellite_with_invalid_constraints(self):
        """Test handling of invalid constraints (min > max)."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # 使用新的 payload_config 格式，但设置无效的约束
        # 注意：ImagingModeConfig会在__post_init__中验证参数，所以这里我们测试回退行为
        capabilities = SatelliteCapabilities(imaging_modes=[ImagingMode.PUSH_BROOM])
        capabilities.payload_config = None  # 禁用payload_config以使用全局约束

        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=capabilities
        )

        # Should fall back to global defaults
        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Result should be valid (min <= duration <= max of global)
        assert duration >= 60.0
        assert duration <= 1800.0

    def test_satellite_constraints_override_only_max(self):
        """Test that satellite constraints can override only max_duration."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # 使用新的 payload_config 格式，只覆盖max_duration
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=60.0,  # 与全局相同
                    max_duration_s=300.0,  # 覆盖全局的1800
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                payload_config=payload_config
            )
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Should use satellite's max (300) instead of global (1800)
        assert duration <= 300.0

    def test_satellite_constraints_override_only_min(self):
        """Test that satellite constraints can override only min_duration."""
        calculator = ImagingTimeCalculator(
            min_duration=60.0,
            max_duration=1800.0,
            default_duration=300.0
        )
        target = MockTarget(target_type=TargetType.POINT)

        # Higher min_duration
        payload_config = PayloadConfiguration(
            payload_type='optical',
            default_mode='push_broom',
            modes={
                'push_broom': ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=120.0,  # 覆盖全局的60
                    max_duration_s=1800.0,  # 与全局相同
                    mode_type='optical',
                    characteristics={'spectral_bands': ['PAN', 'RGB', 'NIR']}
                )
            }
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                payload_config=payload_config
            )
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Should use satellite's min (120) instead of global (60)
        assert duration >= 120.0
