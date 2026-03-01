"""
Integration tests for satellite-specific imaging constraints with scheduler.

Tests that the scheduler correctly uses satellite-specific constraints when
available and falls back to global defaults otherwise.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from core.models.satellite import (
    Satellite,
    SatelliteCapabilities,
    SatelliteType,
    Orbit,
    ImagingMode,
)
from core.models.target import Target, TargetType
from core.models.mission import Mission
from scheduler.greedy.greedy_scheduler import GreedyScheduler


class TestSchedulerWithSatelliteConstraints:
    """Test scheduler integration with satellite-specific constraints."""

    def test_scheduler_uses_satellite_constraints(self):
        """Test that scheduler uses satellite-specific constraints."""
        # Create satellite with custom constraints (min=30, max=120)
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 120.0},
        }
        sat = Satellite(
            id="CONSTRAINED-SAT",
            name="Constrained Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=constraints
            )
        )

        target = Target(
            id="TARGET-1",
            name="Test Target",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        mission = Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[sat],
            targets=[target]
        )

        scheduler = GreedyScheduler(config={
            'min_imaging_duration': 60,  # Global: 60s
            'max_imaging_duration': 1800,  # Global: 1800s
            'default_imaging_duration': 300,  # Global: 300s
        })
        scheduler.initialize(mission)

        # Mock window cache with a window that's long enough
        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [{
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 30),  # 30 min window
            'max_elevation': 45.0
        }]
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should schedule the task
        assert len(result.scheduled_tasks) == 1

        # The duration should be constrained by satellite-specific max (120s)
        # not the global max (1800s)
        task = result.scheduled_tasks[0]
        duration = (task.imaging_end - task.imaging_start).total_seconds()
        assert duration <= 120.0, f"Duration {duration} should be <= 120 (satellite constraint)"
        assert duration >= 30.0, f"Duration {duration} should be >= 30 (satellite constraint)"

    def test_scheduler_fallback_to_global_constraints(self):
        """Test that scheduler falls back to global constraints when no satellite constraints."""
        # Create satellite WITHOUT custom constraints
        sat = Satellite(
            id="DEFAULT-SAT",
            name="Default Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM]
            )
        )

        target = Target(
            id="TARGET-1",
            name="Test Target",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        mission = Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[sat],
            targets=[target]
        )

        scheduler = GreedyScheduler(config={
            'min_imaging_duration': 60,  # Global: 60s
            'max_imaging_duration': 1800,  # Global: 1800s
            'default_imaging_duration': 300,  # Global: 300s
        })
        scheduler.initialize(mission)

        # Mock window cache
        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [{
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 30),
            'max_elevation': 45.0
        }]
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should schedule the task
        assert len(result.scheduled_tasks) == 1

        # The duration should use global constraints (60-1800s)
        task = result.scheduled_tasks[0]
        duration = (task.imaging_end - task.imaging_start).total_seconds()
        assert duration <= 1800.0, f"Duration {duration} should be <= 1800 (global constraint)"
        assert duration >= 60.0, f"Duration {duration} should be >= 60 (global constraint)"

    def test_scheduler_different_satellites_different_constraints(self):
        """Test that different satellites can have different constraints."""
        # Satellite 1: Strict constraints (max 120s)
        sat1_constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 120.0},
        }
        sat1 = Satellite(
            id="STRICT-SAT",
            name="Strict Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=sat1_constraints
            )
        )

        # Satellite 2: Lenient constraints (max 600s)
        sat2_constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        sat2 = Satellite(
            id="LENIENT-SAT",
            name="Lenient Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM],
                imaging_mode_constraints=sat2_constraints
            )
        )

        target = Target(
            id="TARGET-1",
            name="Test Target",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        mission = Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[sat1, sat2],
            targets=[target]
        )

        scheduler = GreedyScheduler(config={
            'min_imaging_duration': 60,
            'max_imaging_duration': 1800,
            'default_imaging_duration': 300,
        })
        scheduler.initialize(mission)

        # Mock window cache - both satellites have windows
        mock_cache = MagicMock()
        mock_cache.get_windows.side_effect = lambda sat_id, target_id: [{
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 30),
            'max_elevation': 45.0
        }]
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should schedule the task
        assert len(result.scheduled_tasks) == 1

        # Verify the scheduled satellite and duration
        task = result.scheduled_tasks[0]
        duration = (task.imaging_end - task.imaging_start).total_seconds()

        # The scheduler should pick the satellite that gives the best score
        # Both have the same priority, but lenient sat allows longer duration
        # which might be preferred based on the scoring function
        if task.satellite_id == "STRICT-SAT":
            assert duration <= 120.0, f"Strict sat duration {duration} should be <= 120"
        else:
            assert duration <= 600.0, f"Lenient sat duration {duration} should be <= 600"

    def test_scheduler_multiple_modes_with_constraints(self):
        """Test scheduler with multiple imaging modes having different constraints."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 300.0},
            ImagingMode.FRAME: {'min_duration': 10.0, 'max_duration': 120.0},
        }
        sat = Satellite(
            id="MULTI-MODE-SAT",
            name="Multi-Mode Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
                imaging_mode_constraints=constraints
            )
        )

        target = Target(
            id="TARGET-1",
            name="Test Target",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        mission = Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[sat],
            targets=[target]
        )

        scheduler = GreedyScheduler(config={
            'min_imaging_duration': 60,
            'max_imaging_duration': 1800,
            'default_imaging_duration': 300,
        })
        scheduler.initialize(mission)

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [{
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 30),
            'max_elevation': 45.0
        }]
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should schedule the task
        assert len(result.scheduled_tasks) == 1

        # The duration should be constrained based on the selected mode
        task = result.scheduled_tasks[0]
        duration = (task.imaging_end - task.imaging_start).total_seconds()

        # The scheduler selects the first available mode (PUSH_BROOM)
        # which has max_duration of 300s
        assert duration <= 300.0, f"Duration {duration} should be <= 300 (PUSH_BROOM constraint)"

    def test_backward_compatibility_no_constraints(self):
        """Test backward compatibility when no constraints are configured."""
        sat = Satellite(
            id="LEGACY-SAT",
            name="Legacy Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.PUSH_BROOM]
            )
        )

        target = Target(
            id="TARGET-1",
            name="Test Target",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        )

        mission = Mission(
            name="Test Mission",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[sat],
            targets=[target]
        )

        # Use default scheduler config
        scheduler = GreedyScheduler()
        scheduler.initialize(mission)

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [{
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 30),
            'max_elevation': 45.0
        }]
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should work without errors
        assert len(result.scheduled_tasks) == 1
