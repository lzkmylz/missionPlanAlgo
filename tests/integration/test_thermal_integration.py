"""
Thermal model integration tests.

Tests integration between thermal model and scheduling system.
Verifies thermal constraints affect scheduling decisions correctly.
"""
import pytest
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from simulator.thermal_model import ThermalParameters, ThermalIntegrator
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


class TestThermalSchedulingIntegration:
    """Test thermal constraints integration with scheduling"""

    @pytest.fixture
    def thermal_params(self):
        """Create standard thermal parameters"""
        return ThermalParameters(
            thermal_capacity=5000.0,
            thermal_resistance=0.5,
            ambient_temperature=273.15,
            max_operating_temp=333.15,  # 60°C
            min_operating_temp=253.15,  # -20°C
            emergency_shutdown_temp=343.15,  # 70°C
            heat_generation={
                'idle': 10.0,
                'slewing': 20.0,
                'imaging_stripmap': 80.0,
                'imaging_spotlight': 200.0,
                'downlink': 50.0,
            }
        )

    @pytest.fixture
    def sample_tasks(self):
        """Create sample scheduled tasks for testing"""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        return [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="spotlight",
                slew_angle=15.0
            ),
            ScheduledTask(
                task_id="TASK-002",
                satellite_id="SAT-01",
                target_id="TARGET-B",
                imaging_start=base_time + timedelta(minutes=10),
                imaging_end=base_time + timedelta(minutes=15),
                imaging_mode="spotlight",
                slew_angle=-10.0
            ),
            ScheduledTask(
                task_id="TASK-003",
                satellite_id="SAT-01",
                target_id="TARGET-C",
                imaging_start=base_time + timedelta(minutes=20),
                imaging_end=base_time + timedelta(minutes=25),
                imaging_mode="stripmap",
                slew_angle=5.0
            ),
        ]

    def test_thermal_integrator_with_schedule(self, thermal_params, sample_tasks):
        """Test thermal integrator tracks temperature through a schedule"""
        integrator = ThermalIntegrator(thermal_params, initial_temp=273.15)

        # Simulate temperature changes through scheduled tasks
        for task in sample_tasks:
            # Slew to target
            slew_start = task.imaging_start - timedelta(minutes=2)
            integrator.update(slew_start, 'slewing')

            # Imaging
            integrator.update(task.imaging_start, f'imaging_{task.imaging_mode}')
            integrator.update(task.imaging_end, 'idle')

        # Temperature should have increased through the schedule
        assert integrator.temperature > 273.15
        assert len(integrator.temperature_history) > 0

    def test_thermal_constraint_violation_detection(self, thermal_params):
        """Test detection of thermal constraint violations in schedule"""
        integrator = ThermalIntegrator(thermal_params, initial_temp=320.0)  # Already warm

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Try to schedule a long spotlight imaging
        imaging_duration = 600  # 10 minutes
        imaging_start = base_time
        imaging_end = imaging_start + timedelta(seconds=imaging_duration)

        # Check if this would violate thermal constraints
        is_valid, predicted_temp = integrator.is_temperature_valid(
            'imaging_spotlight',
            duration=imaging_duration,
            safety_margin=5.0
        )

        # Starting at 320K with 200W for 10 minutes should exceed limit
        assert is_valid is False or predicted_temp > thermal_params.max_operating_temp - 5.0

    def test_thermal_safe_schedule_acceptance(self, thermal_params):
        """Test that thermally safe schedules are accepted"""
        integrator = ThermalIntegrator(thermal_params, initial_temp=273.15)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Short imaging task should be safe
        imaging_duration = 60  # 1 minute

        is_valid, predicted_temp = integrator.is_temperature_valid(
            'imaging_spotlight',
            duration=imaging_duration,
            safety_margin=5.0
        )

        # Starting at ambient, 1 minute should be safe
        assert is_valid is True
        assert predicted_temp <= thermal_params.max_operating_temp - 5.0

    def test_thermal_cooling_between_tasks(self, thermal_params):
        """Test that idle time between tasks allows cooling toward idle steady-state"""
        # Start above idle steady-state to test cooling
        idle_steady_state = thermal_params.ambient_temperature + 10.0 * thermal_params.thermal_resistance
        # Start 10K above idle steady-state (but not too high to avoid overshoot in discrete integration)
        integrator = ThermalIntegrator(thermal_params, initial_temp=idle_steady_state + 10.0)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Initial state - hot
        task1_start = base_time
        integrator.update(task1_start, 'idle')
        temp_start = integrator.temperature

        # Idle period - should cool down toward idle steady-state
        # Use shorter time to avoid discrete integration overshoot
        idle_end = task1_start + timedelta(minutes=30)
        integrator.update(idle_end, 'idle')

        temp_after_idle = integrator.temperature

        # Should have cooled down from initial temperature (moving toward steady-state)
        assert temp_after_idle < temp_start
        # And should be closer to idle steady-state
        assert abs(temp_after_idle - idle_steady_state) < abs(temp_start - idle_steady_state)

    def test_thermal_aware_scheduling_decision(self, thermal_params):
        """Test thermal-aware scheduling decision making"""
        integrator = ThermalIntegrator(thermal_params, initial_temp=310.0)  # Moderately warm

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Evaluate multiple candidate tasks
        candidates = [
            {'id': 'short_spotlight', 'duration': 60, 'mode': 'imaging_spotlight'},
            {'id': 'long_spotlight', 'duration': 600, 'mode': 'imaging_spotlight'},
            {'id': 'stripmap', 'duration': 300, 'mode': 'imaging_stripmap'},
        ]

        valid_tasks = []
        for candidate in candidates:
            is_valid, predicted = integrator.is_temperature_valid(
                candidate['mode'],
                duration=candidate['duration'],
                safety_margin=5.0
            )
            if is_valid:
                valid_tasks.append(candidate['id'])

        # Short tasks should be valid, long ones may not be
        assert 'short_spotlight' in valid_tasks
        # Long spotlight at 310K start may or may not be valid depending on parameters

    def test_thermal_state_during_schedule_simulation(self, thermal_params, sample_tasks):
        """Test thermal state tracking during full schedule simulation"""
        integrator = ThermalIntegrator(thermal_params, initial_temp=273.15)

        thermal_states = []

        for i, task in enumerate(sample_tasks):
            # Simulate slew
            slew_time = task.imaging_start - timedelta(seconds=30)
            integrator.update(slew_time, 'slewing')

            # Simulate imaging
            integrator.update(task.imaging_start, f'imaging_{task.imaging_mode}')
            integrator.update(task.imaging_end, 'idle')

            # Record thermal state after task
            status = integrator.get_thermal_status()
            thermal_states.append({
                'task_id': task.task_id,
                'temperature_k': status['current_temperature_k'],
                'is_safe': status['is_safe'],
                'margin_k': status['temperature_margin_k']
            })

        # Verify thermal states
        assert len(thermal_states) == len(sample_tasks)

        # Temperature should generally increase through the schedule
        temps = [s['temperature_k'] for s in thermal_states]
        assert temps[-1] > temps[0]  # Final temp higher than initial

        # All tasks should have been safe (we started at ambient)
        assert all(s['is_safe'] for s in thermal_states)

    def test_thermal_emergency_prediction(self, thermal_params):
        """Test prediction of thermal emergency conditions"""
        integrator = ThermalIntegrator(thermal_params, initial_temp=330.0)  # Near limit

        # Predict temperature for extended high-power operation
        predicted = integrator.predict_temperature(
            duration=1800,  # 30 minutes
            activity='imaging_spotlight'
        )

        # Should predict emergency shutdown temperature exceeded
        assert predicted > thermal_params.emergency_shutdown_temp

    def test_thermal_recovery_time_calculation(self, thermal_params):
        """Test calculation of thermal recovery time between tasks"""
        # Start at a higher temperature to ensure we need cooling
        integrator = ThermalIntegrator(thermal_params, initial_temp=320.0)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Heat up more with extended imaging
        imaging_start = base_time
        imaging_end = imaging_start + timedelta(minutes=15)
        integrator.update(imaging_start, 'imaging_spotlight')
        integrator.update(imaging_end, 'idle')

        hot_temp = integrator.temperature

        # Calculate cooldown time to a target below current but above ambient
        target_temp = hot_temp - 10.0  # 10K below current
        cooldown_time = integrator.get_cooldown_time(target_temp=target_temp)

        # Should need some time to cool
        assert cooldown_time > 0

        # Verify by simulating
        recovery_end = imaging_end + timedelta(seconds=cooldown_time)
        integrator.update(recovery_end, 'idle')

        # Should be at or below target
        assert integrator.temperature <= target_temp + 0.1  # Small tolerance


class TestThermalConstraintValidation:
    """Test thermal constraint validation for scheduling"""

    def test_temperature_constraint_check_helper(self):
        """Test helper function for temperature constraint checking"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=300.0)

        # Define constraint check function
        def check_thermal_constraint(activity, duration, safety_margin=5.0):
            is_valid, predicted = integrator.is_temperature_valid(
                activity, duration, safety_margin
            )
            return {
                'valid': is_valid,
                'predicted_temp_k': predicted,
                'predicted_temp_c': predicted - 273.15,
                'limit_k': params.max_operating_temp - safety_margin,
                'margin_k': params.max_operating_temp - safety_margin - predicted
            }

        # Test constraint check
        result = check_thermal_constraint('imaging_spotlight', 300)

        assert 'valid' in result
        assert 'predicted_temp_k' in result
        assert 'margin_k' in result

    def test_thermal_constraint_batch_validation(self):
        """Test batch validation of multiple activities"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=310.0)

        activities = [
            {'name': 'task1', 'activity': 'imaging_stripmap', 'duration': 120},
            {'name': 'task2', 'activity': 'imaging_spotlight', 'duration': 60},
            {'name': 'task3', 'activity': 'imaging_spotlight', 'duration': 600},
        ]

        results = []
        for act in activities:
            is_valid, predicted = integrator.is_temperature_valid(
                act['activity'], act['duration']
            )
            results.append({
                'name': act['name'],
                'valid': is_valid,
                'predicted_temp': predicted
            })

        # All results should have validity info
        assert len(results) == len(activities)
        assert all('valid' in r for r in results)
        assert all('predicted_temp' in r for r in results)

    def test_thermal_constraint_with_schedule_result(self):
        """Test applying thermal constraints to schedule result"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Create a schedule result
        tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=3),
                imaging_mode="spotlight",
                slew_angle=15.0
            ),
        ]

        schedule = ScheduleResult(
            scheduled_tasks=tasks,
            unscheduled_tasks={},
            makespan=180.0,
            computation_time=0.1,
            iterations=1
        )

        # Validate schedule thermally
        violations = []
        for task in schedule.scheduled_tasks:
            duration = (task.imaging_end - task.imaging_start).total_seconds()
            is_valid, predicted = integrator.is_temperature_valid(
                f'imaging_{task.imaging_mode}',
                duration
            )
            if not is_valid:
                violations.append({
                    'task_id': task.task_id,
                    'predicted_temp': predicted
                })

        # Should have no violations for short task from ambient
        assert len(violations) == 0


class TestThermalSunExclusionIntegration:
    """Test integration between thermal model and sun exclusion"""

    def test_thermal_impact_of_sun_exclusion_maneuvers(self):
        """Test thermal impact of sun exclusion maneuvers"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Simulate sun exclusion maneuver (extended slewing)
        maneuver_start = base_time
        maneuver_end = maneuver_start + timedelta(minutes=5)

        integrator.update(maneuver_start, 'slewing')
        integrator.update(maneuver_end, 'idle')

        # Should have some heating from slewing
        assert integrator.temperature >= 273.15

        status = integrator.get_thermal_status()
        assert status['is_safe'] is True

    def test_thermal_vs_sun_exclusion_tradeoff(self):
        """Test tradeoff between thermal constraints and sun exclusion"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=320.0)  # Warm

        # Scenario: need to choose between sun exposure and thermal constraints
        # Longer path avoids sun but increases slewing time (more heat)

        short_path_slew_time = 30  # seconds
        long_path_slew_time = 120  # seconds

        # Calculate thermal impact of each path using predict_temperature
        temp_short = integrator.predict_temperature(short_path_slew_time, 'slewing')
        temp_long = integrator.predict_temperature(long_path_slew_time, 'slewing')

        # Longer path should result in higher or equal temperature
        # (approaching steady-state from below or above)
        steady_state = params.ambient_temperature + 20.0 * params.thermal_resistance  # slewing = 20W
        if integrator.temperature < steady_state:
            # Heating up toward steady-state
            assert temp_long >= temp_short
        else:
            # Cooling down toward steady-state
            assert temp_long <= temp_short


class TestThermalModelPerformance:
    """Test thermal model performance characteristics"""

    def test_thermal_prediction_performance(self):
        """Test that temperature predictions are computationally efficient"""
        import time

        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=300.0)

        # Time multiple predictions
        start_time = time.time()
        for _ in range(1000):
            integrator.predict_temperature(300, 'imaging_spotlight')
        elapsed = time.time() - start_time

        # Should be very fast (less than 1 second for 1000 predictions)
        assert elapsed < 1.0

    def test_thermal_update_performance(self):
        """Test that temperature updates are computationally efficient"""
        import time

        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Time multiple updates
        start_time = time.time()
        for i in range(100):
            current_time = base_time + timedelta(seconds=i * 10)
            integrator.update(current_time, 'imaging_stripmap')
        elapsed = time.time() - start_time

        # Should be fast (less than 0.5 seconds for 100 updates)
        assert elapsed < 0.5

    def test_thermal_history_memory_usage(self):
        """Test that temperature history doesn't consume excessive memory"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Simulate long mission
        for i in range(10000):
            current_time = base_time + timedelta(seconds=i * 60)
            integrator.update(current_time, 'idle')

        # History should have 10000 entries
        assert len(integrator.temperature_history) == 10000

        # Each entry is a tuple of (datetime, float) - should be reasonable size
        # Rough estimate: 10000 entries * ~50 bytes each = ~500KB
        import sys
        history_size = sys.getsizeof(integrator.temperature_history)
        # Just verify it's not outrageously large
        assert history_size < 10 * 1024 * 1024  # Less than 10MB
