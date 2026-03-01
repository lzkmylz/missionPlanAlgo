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
    ImagingMode,
    Satellite,
    SatelliteType,
    Orbit,
)
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

        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(imaging_modes=[ImagingMode.PUSH_BROOM])
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Should use global defaults
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

        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=constraints
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

        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
            ImagingMode.FRAME: {'min_duration': 10.0, 'max_duration': 300.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                imaging_mode_constraints=constraints
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

        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                imaging_mode_constraints=constraints
            )
        )

        # FRAME has no constraints, should use global (max 1800)
        duration = calculator.calculate(target, ImagingMode.FRAME, satellite=satellite)
        assert duration <= 1800.0
        assert duration >= 60.0

    def test_get_constraints_for_satellite_and_mode(self):
        """Test getting constraints for specific satellite and mode."""
        calculator = ImagingTimeCalculator()

        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=constraints
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

        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(imaging_modes=[ImagingMode.PUSH_BROOM])
        )

        result = calculator.get_constraints_for_satellite(satellite, ImagingMode.PUSH_BROOM)
        assert result is None

    def test_get_constraints_returns_none_for_mode_not_configured(self):
        """Test getting constraints returns None when mode not configured."""
        calculator = ImagingTimeCalculator()

        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                imaging_mode_constraints=constraints
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

        constraints = {
            ImagingMode.STRIPMAP: {'min_duration': 45.0, 'max_duration': 1200.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.SAR_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.STRIPMAP],
                imaging_mode_constraints=constraints
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

        # Invalid constraints: min > max
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 600.0, 'max_duration': 30.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=constraints
            )
        )

        # Should fall back to global defaults or swap values
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

        # Only specify max_duration
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 60.0, 'max_duration': 300.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=constraints
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
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 120.0, 'max_duration': 1800.0},
        }
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=constraints
            )
        )

        duration = calculator.calculate(target, ImagingMode.PUSH_BROOM, satellite=satellite)

        # Should use satellite's min (120) instead of global (60)
        assert duration >= 120.0
