"""
Unit tests for AttitudeConstraintChecker.

TDD Phase 6: AttitudeConstraintChecker validates attitude transition feasibility
for scheduler integration.

Test scenarios:
- Attitude transition feasibility checking
- Slew angle validation against max_slew_angle
- Transition time calculation
- Integration with AttitudeManager
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock

from scheduler.constraints.attitude_constraint_checker import (
    AttitudeConstraintChecker,
    AttitudeFeasibilityResult,
)
from core.dynamics.attitude_mode import AttitudeMode, TransitionResult


class TestAttitudeFeasibilityResult:
    """Tests for AttitudeFeasibilityResult dataclass."""

    def test_result_creation(self):
        """Test creating a feasibility result."""
        result = AttitudeFeasibilityResult(
            feasible=True,
            slew_time=30.0,
            slew_angle=15.0,
            transition_time=timedelta(seconds=30),
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            reason=None
        )

        assert result.feasible is True
        assert result.slew_time == 30.0
        assert result.slew_angle == 15.0
        assert result.transition_time.total_seconds() == 30.0
        assert result.from_mode == AttitudeMode.NADIR_POINTING
        assert result.to_mode == AttitudeMode.IMAGING
        assert result.reason is None

    def test_infeasible_result(self):
        """Test creating an infeasible result."""
        result = AttitudeFeasibilityResult(
            feasible=False,
            slew_time=0.0,
            slew_angle=50.0,
            transition_time=timedelta(seconds=0),
            from_mode=AttitudeMode.SUN_POINTING,
            to_mode=AttitudeMode.IMAGING,
            reason="Slew angle 50.0 exceeds max 30.0"
        )

        assert result.feasible is False
        assert result.slew_angle == 50.0
        assert "exceeds max" in result.reason


class TestAttitudeConstraintCheckerInitialization:
    """Tests for AttitudeConstraintChecker initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        checker = AttitudeConstraintChecker()

        assert checker.config is not None
        assert checker.config.max_slew_rate == 3.0
        assert checker.config.settling_time == 5.0
        assert checker._attitude_manager is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        from core.dynamics.attitude_manager import AttitudeManagementConfig

        config = AttitudeManagementConfig(
            max_slew_rate=2.0,
            settling_time=10.0
        )
        checker = AttitudeConstraintChecker(config)

        assert checker.config.max_slew_rate == 2.0
        assert checker.config.settling_time == 10.0

    def test_init_creates_attitude_manager(self):
        """Test that initialization creates AttitudeManager."""
        checker = AttitudeConstraintChecker()

        assert checker._attitude_manager is not None
        assert hasattr(checker._attitude_manager, 'plan_transition')
        assert hasattr(checker._attitude_manager, 'decide_post_task_attitude')


class TestCheckAttitudeTransition:
    """Tests for check_attitude_transition method."""

    @pytest.fixture
    def checker(self):
        return AttitudeConstraintChecker()

    @pytest.fixture
    def base_time(self):
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def satellite_position(self):
        # LEO satellite position in ECEF (meters)
        return (7000000.0, 0.0, 0.0)

    def test_same_mode_transition(self, checker, base_time, satellite_position):
        """Test transition between same modes is always feasible."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.NADIR_POINTING,
            satellite_position=satellite_position,
            timestamp=base_time
        )

        assert result.feasible is True
        assert result.slew_time == 0.0
        assert result.slew_angle == 0.0
        assert result.transition_time.total_seconds() == 0.0

    def test_nadir_to_imaging_feasible(self, checker, base_time, satellite_position):
        """Test feasible transition from nadir to imaging."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            target_position=(39.9, 116.4)  # Beijing
        )

        assert isinstance(result, AttitudeFeasibilityResult)
        assert result.feasible is True
        assert result.slew_time > 0.0
        assert result.slew_angle > 0.0
        assert result.from_mode == AttitudeMode.NADIR_POINTING
        assert result.to_mode == AttitudeMode.IMAGING

    def test_imaging_to_nadir_feasible(self, checker, base_time, satellite_position):
        """Test feasible transition from imaging to nadir."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.IMAGING,
            to_mode=AttitudeMode.NADIR_POINTING,
            satellite_position=satellite_position,
            timestamp=base_time,
            target_position=(39.9, 116.4)
        )

        assert result.feasible is True
        assert result.slew_time > 0.0

    def test_nadir_to_sun_feasible(self, checker, base_time, satellite_position):
        """Test feasible transition from nadir to sun pointing."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            satellite_position=satellite_position,
            timestamp=base_time
        )

        assert result.feasible is True
        assert result.slew_time > 0.0

    def test_imaging_without_target_infeasible(self, checker, base_time, satellite_position):
        """Test that imaging transition without target is infeasible."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            target_position=None
        )

        assert result.feasible is False
        assert result.reason is not None

    def test_all_supported_mode_combinations(self, checker, base_time, satellite_position):
        """Test all supported mode combinations produce valid results."""
        supported_modes = [
            AttitudeMode.NADIR_POINTING,
            AttitudeMode.SUN_POINTING,
            AttitudeMode.IMAGING,
        ]

        for from_mode in supported_modes:
            for to_mode in supported_modes:
                result = checker.check_attitude_transition(
                    from_mode=from_mode,
                    to_mode=to_mode,
                    satellite_position=satellite_position,
                    timestamp=base_time,
                    target_position=(39.9, 116.4) if from_mode == AttitudeMode.IMAGING or to_mode == AttitudeMode.IMAGING else None
                )

                assert isinstance(result, AttitudeFeasibilityResult)
                assert hasattr(result, 'feasible')
                assert hasattr(result, 'slew_time')
                assert hasattr(result, 'slew_angle')


class TestCheckSlewFeasibility:
    """Tests for check_slew_feasibility method with max_slew_angle constraint."""

    @pytest.fixture
    def checker(self):
        return AttitudeConstraintChecker()

    @pytest.fixture
    def base_time(self):
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def satellite_position(self):
        return (7000000.0, 0.0, 0.0)

    def test_slew_within_limit(self, checker, base_time, satellite_position):
        """Test slew within max_slew_angle is feasible."""
        # Small angle should be feasible
        result = checker.check_slew_feasibility(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            max_slew_angle=45.0,  # Generous limit
            target_position=(39.9, 116.4)
        )

        assert result.feasible is True
        assert result.slew_angle <= 45.0

    def test_slew_exceeds_limit(self, checker, base_time, satellite_position):
        """Test slew exceeding max_slew_angle is infeasible."""
        # Create a scenario with large slew angle by using very restrictive limit
        result = checker.check_slew_feasibility(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            max_slew_angle=1.0,  # Very restrictive (1 degree)
            target_position=(39.9, 116.4)
        )

        # Should be infeasible due to exceeding max_slew_angle
        assert result.feasible is False
        assert result.reason is not None
        assert "exceeds" in result.reason.lower() or "angle" in result.reason.lower()

    def test_zero_max_slew_angle(self, checker, base_time, satellite_position):
        """Test with zero max_slew_angle - only same mode is feasible."""
        result = checker.check_slew_feasibility(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            max_slew_angle=0.0,
            target_position=(39.9, 116.4)
        )

        assert result.feasible is False

    def test_same_mode_always_feasible(self, checker, base_time, satellite_position):
        """Test that same mode transition is always feasible regardless of limit."""
        result = checker.check_slew_feasibility(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.NADIR_POINTING,
            satellite_position=satellite_position,
            timestamp=base_time,
            max_slew_angle=0.0  # Even with zero limit
        )

        assert result.feasible is True
        assert result.slew_angle == 0.0


class TestCalculateTransitionTime:
    """Tests for calculate_transition_time method."""

    @pytest.fixture
    def checker(self):
        return AttitudeConstraintChecker()

    @pytest.fixture
    def base_time(self):
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def satellite_position(self):
        return (7000000.0, 0.0, 0.0)

    def test_same_mode_zero_time(self, checker, base_time, satellite_position):
        """Test same mode transition has zero time."""
        transition_time = checker.calculate_transition_time(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.NADIR_POINTING,
            satellite_position=satellite_position,
            timestamp=base_time
        )

        assert transition_time.total_seconds() == 0.0

    def test_nadir_to_imaging_time(self, checker, base_time, satellite_position):
        """Test transition time from nadir to imaging."""
        transition_time = checker.calculate_transition_time(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            target_position=(39.9, 116.4)
        )

        assert transition_time.total_seconds() > 0.0

    def test_returns_timedelta(self, checker, base_time, satellite_position):
        """Test that method returns timedelta object."""
        transition_time = checker.calculate_transition_time(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            satellite_position=satellite_position,
            timestamp=base_time
        )

        assert isinstance(transition_time, timedelta)


class TestAttitudeConstraintCheckerEdgeCases:
    """Edge case tests for AttitudeConstraintChecker."""

    @pytest.fixture
    def checker(self):
        return AttitudeConstraintChecker()

    @pytest.fixture
    def base_time(self):
        return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_none_satellite_position(self, checker, base_time):
        """Test behavior with None satellite position."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            satellite_position=None,
            timestamp=base_time
        )

        # Should handle gracefully
        assert isinstance(result, AttitudeFeasibilityResult)
        assert result.feasible is False

    def test_invalid_satellite_position(self, checker, base_time):
        """Test behavior with invalid satellite position."""
        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.SUN_POINTING,
            satellite_position=(0.0, 0.0),  # Invalid: only 2 coordinates
            timestamp=base_time
        )

        # Should handle gracefully
        assert isinstance(result, AttitudeFeasibilityResult)

    def test_very_distant_target(self, checker, base_time):
        """Test with target very far from nadir."""
        satellite_position = (7000000.0, 0.0, 0.0)

        result = checker.check_attitude_transition(
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            satellite_position=satellite_position,
            timestamp=base_time,
            target_position=(-80.0, 170.0)  # Very distant target
        )

        # Should still produce a result
        assert isinstance(result, AttitudeFeasibilityResult)
        assert hasattr(result, 'slew_angle')
        assert hasattr(result, 'slew_time')

    def test_all_attitude_modes(self, checker, base_time):
        """Test that all attitude modes are handled."""
        satellite_position = (7000000.0, 0.0, 0.0)

        all_modes = [
            AttitudeMode.SUN_POINTING,
            AttitudeMode.NADIR_POINTING,
            AttitudeMode.IMAGING,
            AttitudeMode.DOWNLINK,
            AttitudeMode.REALTIME,
            AttitudeMode.MOMENTUM_DUMP,
        ]

        for mode in all_modes:
            result = checker.check_attitude_transition(
                from_mode=AttitudeMode.NADIR_POINTING,
                to_mode=mode,
                satellite_position=satellite_position,
                timestamp=base_time,
                target_position=(39.9, 116.4) if mode == AttitudeMode.IMAGING else None
            )

            assert isinstance(result, AttitudeFeasibilityResult)
