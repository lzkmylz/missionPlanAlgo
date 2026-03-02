"""
Unit tests for attitude management core data structures.

TDD Phase 1: Test AttitudeMode enum, AttitudeTransition, and TransitionResult dataclasses.
"""
import pytest
from datetime import datetime
from typing import Tuple

from core.dynamics.attitude_mode import (
    AttitudeMode,
    AttitudeTransition,
    TransitionResult,
)


class TestAttitudeMode:
    """Tests for AttitudeMode enum."""

    def test_all_modes_exist(self):
        """Test that all 6 required attitude modes exist."""
        expected_modes = [
            "SUN_POINTING",
            "NADIR_POINTING",
            "IMAGING",
            "DOWNLINK",
            "REALTIME",
            "MOMENTUM_DUMP",
        ]
        for mode_name in expected_modes:
            assert hasattr(AttitudeMode, mode_name)
            mode = getattr(AttitudeMode, mode_name)
            assert isinstance(mode, AttitudeMode)

    def test_mode_values_are_unique(self):
        """Test that all mode values are unique."""
        modes = list(AttitudeMode)
        values = [m.value for m in modes]
        assert len(values) == len(set(values)), "Mode values must be unique"

    def test_mode_count(self):
        """Test that there are exactly 6 modes."""
        assert len(AttitudeMode) == 6

    def test_mode_comparison(self):
        """Test that modes can be compared."""
        assert AttitudeMode.SUN_POINTING == AttitudeMode.SUN_POINTING
        assert AttitudeMode.SUN_POINTING != AttitudeMode.NADIR_POINTING


class TestAttitudeTransition:
    """Tests for AttitudeTransition dataclass."""

    def test_basic_creation(self):
        """Test creating AttitudeTransition with required fields."""
        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            satellite_position=(7000.0, 0.0, 0.0),
        )
        assert transition.from_mode == AttitudeMode.NADIR_POINTING
        assert transition.to_mode == AttitudeMode.IMAGING
        assert transition.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert transition.satellite_position == (7000.0, 0.0, 0.0)

    def test_creation_with_all_optional_fields(self):
        """Test creating AttitudeTransition with all optional fields."""
        transition = AttitudeTransition(
            from_mode=AttitudeMode.SUN_POINTING,
            to_mode=AttitudeMode.DOWNLINK,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            satellite_position=(7000.0, 0.0, 0.0),
            sun_position=(1.5e11, 0.0, 0.0),
            target_position=(45.0, 90.0),
            ground_station_position=(40.0, 116.0),
        )
        assert transition.sun_position == (1.5e11, 0.0, 0.0)
        assert transition.target_position == (45.0, 90.0)
        assert transition.ground_station_position == (40.0, 116.0)

    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None."""
        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            satellite_position=(7000.0, 0.0, 0.0),
        )
        assert transition.sun_position is None
        assert transition.target_position is None
        assert transition.ground_station_position is None

    def test_immutability(self):
        """Test that AttitudeTransition is immutable (frozen dataclass)."""
        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            satellite_position=(7000.0, 0.0, 0.0),
        )
        with pytest.raises(AttributeError):
            transition.to_mode = AttitudeMode.SUN_POINTING

    def test_invalid_satellite_position_type(self):
        """Test validation of satellite_position (must be 3-tuple)."""
        # This should ideally raise a validation error
        # For now, we test that the field exists and accepts tuples
        with pytest.raises((TypeError, ValueError)):
            AttitudeTransition(
                from_mode=AttitudeMode.NADIR_POINTING,
                to_mode=AttitudeMode.IMAGING,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                satellite_position=(7000.0, 0.0),  # Wrong length
            )

    def test_edge_case_empty_optional_fields(self):
        """Test edge case with explicitly None optional fields."""
        transition = AttitudeTransition(
            from_mode=AttitudeMode.MOMENTUM_DUMP,
            to_mode=AttitudeMode.SUN_POINTING,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            satellite_position=(0.0, 7000.0, 0.0),
            sun_position=None,
            target_position=None,
            ground_station_position=None,
        )
        assert transition.sun_position is None
        assert transition.target_position is None
        assert transition.ground_station_position is None


class TestTransitionResult:
    """Tests for TransitionResult dataclass."""

    def test_basic_creation(self):
        """Test creating TransitionResult with all required fields."""
        result = TransitionResult(
            slew_time=30.0,
            slew_angle=45.0,
            roll_angle=10.0,
            pitch_angle=5.0,
            power_generation=150.0,
            feasible=True,
        )
        assert result.slew_time == 30.0
        assert result.slew_angle == 45.0
        assert result.roll_angle == 10.0
        assert result.pitch_angle == 5.0
        assert result.power_generation == 150.0
        assert result.feasible is True
        assert result.reason is None

    def test_creation_with_reason(self):
        """Test creating TransitionResult with reason field."""
        result = TransitionResult(
            slew_time=0.0,
            slew_angle=0.0,
            roll_angle=0.0,
            pitch_angle=0.0,
            power_generation=0.0,
            feasible=False,
            reason="Insufficient power for maneuver",
        )
        assert result.feasible is False
        assert result.reason == "Insufficient power for maneuver"

    def test_reason_defaults_to_none(self):
        """Test that reason defaults to None when feasible is True."""
        result = TransitionResult(
            slew_time=30.0,
            slew_angle=45.0,
            roll_angle=10.0,
            pitch_angle=5.0,
            power_generation=150.0,
            feasible=True,
        )
        assert result.reason is None

    def test_immutability(self):
        """Test that TransitionResult is immutable (frozen dataclass)."""
        result = TransitionResult(
            slew_time=30.0,
            slew_angle=45.0,
            roll_angle=10.0,
            pitch_angle=5.0,
            power_generation=150.0,
            feasible=True,
        )
        with pytest.raises(AttributeError):
            result.slew_time = 60.0

    def test_edge_case_zero_values(self):
        """Test edge case with zero values."""
        result = TransitionResult(
            slew_time=0.0,
            slew_angle=0.0,
            roll_angle=0.0,
            pitch_angle=0.0,
            power_generation=0.0,
            feasible=True,
        )
        assert result.slew_time == 0.0
        assert result.feasible is True

    def test_edge_case_negative_angles(self):
        """Test edge case with negative angles."""
        result = TransitionResult(
            slew_time=30.0,
            slew_angle=-45.0,
            roll_angle=-10.0,
            pitch_angle=-5.0,
            power_generation=150.0,
            feasible=True,
        )
        assert result.slew_angle == -45.0
        assert result.roll_angle == -10.0
        assert result.pitch_angle == -5.0

    def test_edge_case_large_values(self):
        """Test edge case with large values."""
        result = TransitionResult(
            slew_time=3600.0,
            slew_angle=180.0,
            roll_angle=90.0,
            pitch_angle=90.0,
            power_generation=10000.0,
            feasible=True,
        )
        assert result.slew_time == 3600.0
        assert result.slew_angle == 180.0


class TestIntegration:
    """Integration tests between attitude components."""

    def test_transition_to_result_workflow(self):
        """Test typical workflow from transition request to result."""
        # Create a transition request
        transition = AttitudeTransition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            satellite_position=(7000.0, 0.0, 0.0),
            target_position=(45.0, 90.0),
        )

        # Create a corresponding result
        result = TransitionResult(
            slew_time=25.0,
            slew_angle=35.0,
            roll_angle=15.0,
            pitch_angle=8.0,
            power_generation=120.0,
            feasible=True,
        )

        assert transition.to_mode == AttitudeMode.IMAGING
        assert result.feasible is True
        assert result.slew_time > 0

    def test_all_mode_transitions(self):
        """Test that all mode combinations can be used in transitions."""
        modes = list(AttitudeMode)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        position = (7000.0, 0.0, 0.0)

        for from_mode in modes:
            for to_mode in modes:
                transition = AttitudeTransition(
                    from_mode=from_mode,
                    to_mode=to_mode,
                    timestamp=timestamp,
                    satellite_position=position,
                )
                assert transition.from_mode == from_mode
                assert transition.to_mode == to_mode
