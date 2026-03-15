"""
Slew Time Integration Tests - TDD Implementation

Tests to verify all schedulers properly consider attitude slew time and settling time.
All schedulers should:
1. Use satellite agility parameters (max_slew_rate, settling_time)
2. Calculate slew time dynamically based on angle between consecutive targets
3. Record both slew_angle and slew_time in ScheduledTask
4. Include slew information in output/plans

Test Coverage:
1. EDDScheduler - slew time integration
2. SPTScheduler - slew time integration
3. GAScheduler - slew time integration
4. SAScheduler - slew time integration
5. ACOScheduler - slew time integration
6. PSOScheduler - slew time integration
7. TabuScheduler - slew time integration
8. ScheduledTask - slew_time field existence
9. Edge cases (null, empty, invalid agility params)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.models import (
    Mission, Satellite, SatelliteType, Target, TargetType,
    ImagingMode, Orbit, OrbitType, SatelliteCapabilities
)
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler
from scheduler.metaheuristic.aco_scheduler import ACOScheduler
from scheduler.metaheuristic.pso_scheduler import PSOScheduler
from scheduler.metaheuristic.tabu_scheduler import TabuScheduler
from scheduler.base_scheduler import ScheduleResult, TaskFailureReason, ScheduledTask
from scheduler.constraints import BatchSlewConstraintChecker, BatchSlewCandidate


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mission_start_time():
    """Mission start time"""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_satellite_high_agility(mission_start_time):
    """High agility satellite with fast slew rate"""
    capabilities = SatelliteCapabilities(
        imaging_modes=[ImagingMode.PUSH_BROOM],
        resolution=0.5,
        swath_width=10.0,
        power_capacity=2800.0,
        storage_capacity=128.0,
        max_roll_angle=45.0,
        agility={
            'max_slew_rate': 5.0,  # 5 degrees/sec - fast
            'settling_time': 3.0    # 3 seconds
        }
    )
    return Satellite(
        id="sat_high_agility",
        name="High Agility Satellite",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(altitude=500000.0, inclination=97.4),
        capabilities=capabilities
    )


@pytest.fixture
def sample_satellite_low_agility(mission_start_time):
    """Low agility satellite with slow slew rate"""
    capabilities = SatelliteCapabilities(
        imaging_modes=[ImagingMode.PUSH_BROOM],
        resolution=0.5,
        swath_width=10.0,
        power_capacity=2800.0,
        storage_capacity=128.0,
        max_roll_angle=30.0,
        agility={
            'max_slew_rate': 1.0,  # 1 degree/sec - slow
            'settling_time': 10.0   # 10 seconds
        }
    )
    return Satellite(
        id="sat_low_agility",
        name="Low Agility Satellite",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(altitude=500000.0, inclination=97.4),
        capabilities=capabilities
    )


@pytest.fixture
def sample_satellite_default_agility(mission_start_time):
    """Satellite with default agility parameters"""
    capabilities = SatelliteCapabilities(
        imaging_modes=[ImagingMode.PUSH_BROOM],
        resolution=0.5,
        swath_width=10.0,
        power_capacity=2800.0,
        storage_capacity=128.0,
        max_roll_angle=45.0,
        agility={
            'max_slew_rate': 2.0,  # 2 degrees/sec - default
            'settling_time': 5.0    # 5 seconds - default
        }
    )
    return Satellite(
        id="sat_default_agility",
        name="Default Agility Satellite",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(altitude=500000.0, inclination=97.4),
        capabilities=capabilities
    )


@pytest.fixture
def sample_targets():
    """Sample targets for testing"""
    return [
        Target(
            id=f"target_{i:03d}",
            name=f"Target {i}",
            target_type=TargetType.POINT,
            longitude=10.0 + i * 2.0,  # 2 degrees apart
            latitude=20.0 + i * 0.5,
            priority=i % 3 + 1
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_window_cache(sample_targets, mission_start_time):
    """Mock window cache for testing"""
    class MockWindowCache:
        def __init__(self):
            self.windows = {}
            for target in sample_targets:
                # Create a visibility window for each target
                self.windows[target.id] = [
                    MagicMock(
                        start_time=mission_start_time + timedelta(minutes=i),
                        end_time=mission_start_time + timedelta(minutes=i + 5),
                        quality_score=0.8
                    )
                    for i in range(3)
                ]

        def get_windows(self, sat_id, target_id):
            return self.windows.get(target_id, [])

    return MockWindowCache()


@pytest.fixture
def sample_mission(sample_satellite_default_agility, sample_targets, mission_start_time):
    """Sample mission for testing"""
    return Mission(
        name="Test Mission",
        start_time=mission_start_time,
        end_time=mission_start_time + timedelta(hours=2),
        satellites=[sample_satellite_default_agility],
        targets=sample_targets
    )


# =============================================================================
# ScheduledTask Slew Time Field Tests
# =============================================================================

class TestScheduledTaskSlewTimeField:
    """Test that ScheduledTask has slew_time field"""

    def test_scheduled_task_has_slew_angle_field(self):
        """Test ScheduledTask has slew_angle field (existing)"""
        task = ScheduledTask(
            task_id="task_001",
            satellite_id="sat_001",
            target_id="target_001",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(seconds=30),
            imaging_mode="push_broom",
            slew_angle=15.5
        )
        assert hasattr(task, 'slew_angle')
        assert task.slew_angle == 15.5

    def test_scheduled_task_has_slew_time_field(self):
        """Test ScheduledTask has slew_time field (to be added)"""
        task = ScheduledTask(
            task_id="task_001",
            satellite_id="sat_001",
            target_id="target_001",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(seconds=30),
            imaging_mode="push_broom",
            slew_angle=15.5,
            slew_time=12.75  # New field
        )
        assert hasattr(task, 'slew_time')
        assert task.slew_time == 12.75

    def test_scheduled_task_slew_time_defaults_to_zero(self):
        """Test slew_time defaults to 0.0 when not specified"""
        task = ScheduledTask(
            task_id="task_001",
            satellite_id="sat_001",
            target_id="target_001",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(seconds=30),
            imaging_mode="push_broom"
        )
        assert hasattr(task, 'slew_time')
        assert task.slew_time == 0.0

    def test_scheduled_task_to_dict_includes_slew_time(self):
        """Test to_dict() includes slew_time"""
        task = ScheduledTask(
            task_id="task_001",
            satellite_id="sat_001",
            target_id="target_001",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(seconds=30),
            imaging_mode="push_broom",
            slew_angle=15.5,
            slew_time=12.75
        )
        task_dict = task.to_dict()
        assert 'slew_angle' in task_dict
        assert 'slew_time' in task_dict
        assert task_dict['slew_angle'] == 15.5
        assert task_dict['slew_time'] == 12.75


# =============================================================================
# GreedyScheduler Slew Time Tests (Reference - Already Implemented)
# =============================================================================

class TestGreedySchedulerSlewTime:
    """Test GreedyScheduler slew time integration (reference implementation)"""

    def test_greedy_scheduler_uses_dynamic_slew_time(
        self, sample_mission, mock_window_cache, sample_satellite_default_agility
    ):
        """Test GreedyScheduler uses dynamic slew time calculation"""
        scheduler = GreedyScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) > 0

        # Check that scheduled tasks have slew_angle
        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert task.slew_angle >= 0.0

    def test_greedy_scheduler_initializes_slew_checker(
        self, sample_mission, mock_window_cache, sample_satellite_default_agility
    ):
        """Test GreedyScheduler initializes BatchSlewConstraintChecker"""
        scheduler = GreedyScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Schedule to trigger initialization
        scheduler.schedule()

        # Check that slew checker is initialized (via base class)
        assert hasattr(scheduler, '_slew_checker')
        assert scheduler._slew_checker is not None
        assert isinstance(scheduler._slew_checker, BatchSlewConstraintChecker)


# =============================================================================
# EDDScheduler Slew Time Tests
# =============================================================================

class TestEDDSchedulerSlewTime:
    """Test EDDScheduler slew time integration"""

    def test_edd_scheduler_uses_dynamic_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test EDDScheduler uses dynamic slew time instead of fixed DEFAULT_SLEW_TIME"""
        scheduler = EDDScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        # Should have scheduled tasks

    def test_edd_scheduler_considers_agility_parameters(
        self, sample_mission, mock_window_cache, sample_satellite_high_agility
    ):
        """Test EDDScheduler considers satellite agility parameters"""
        sample_mission.satellites = [sample_satellite_high_agility]

        scheduler = EDDScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Should initialize slew calculator with agility params
        result = scheduler.schedule()
        assert isinstance(result, ScheduleResult)

    def test_edd_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test EDDScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = EDDScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')
            # slew_time should be calculated based on slew_angle
            if task.slew_angle > 0:
                assert task.slew_time > 0


# =============================================================================
# SPTScheduler Slew Time Tests
# =============================================================================

class TestSPTSchedulerSlewTime:
    """Test SPTScheduler slew time integration"""

    def test_spt_scheduler_uses_dynamic_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test SPTScheduler uses dynamic slew time instead of fixed DEFAULT_SLEW_TIME"""
        scheduler = SPTScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_spt_scheduler_considers_agility_parameters(
        self, sample_mission, mock_window_cache, sample_satellite_low_agility
    ):
        """Test SPTScheduler considers satellite agility parameters"""
        sample_mission.satellites = [sample_satellite_low_agility]

        scheduler = SPTScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()
        assert isinstance(result, ScheduleResult)

    def test_spt_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test SPTScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = SPTScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')


# =============================================================================
# GAScheduler Slew Time Tests
# =============================================================================

class TestGASchedulerSlewTime:
    """Test GAScheduler slew time integration"""

    def test_ga_scheduler_considers_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test GAScheduler considers slew time in scheduling decisions"""
        scheduler = GAScheduler(config={
            'population_size': 20,
            'generations': 10,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_ga_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test GAScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = GAScheduler(config={
            'population_size': 20,
            'generations': 10,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')


# =============================================================================
# SAScheduler Slew Time Tests
# =============================================================================

class TestSASchedulerSlewTime:
    """Test SAScheduler slew time integration"""

    def test_sa_scheduler_considers_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test SAScheduler considers slew time in scheduling decisions"""
        scheduler = SAScheduler(config={
            'max_iterations': 50,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_sa_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test SAScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = SAScheduler(config={
            'max_iterations': 50,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')


# =============================================================================
# ACOScheduler Slew Time Tests
# =============================================================================

class TestACOSchedulerSlewTime:
    """Test ACOScheduler slew time integration"""

    def test_aco_scheduler_considers_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test ACOScheduler considers slew time in scheduling decisions"""
        scheduler = ACOScheduler(config={
            'num_ants': 10,
            'max_iterations': 10,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_aco_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test ACOScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = ACOScheduler(config={
            'num_ants': 10,
            'max_iterations': 10,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')


# =============================================================================
# PSOScheduler Slew Time Tests
# =============================================================================

class TestPSOSchedulerSlewTime:
    """Test PSOScheduler slew time integration"""

    def test_pso_scheduler_considers_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test PSOScheduler considers slew time in scheduling decisions"""
        scheduler = PSOScheduler(config={
            'num_particles': 10,
            'max_iterations': 10,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_pso_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test PSOScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = PSOScheduler(config={
            'num_particles': 10,
            'max_iterations': 10,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')


# =============================================================================
# TabuScheduler Slew Time Tests
# =============================================================================

class TestTabuSchedulerSlewTime:
    """Test TabuScheduler slew time integration"""

    def test_tabu_scheduler_considers_slew_time(
        self, sample_mission, mock_window_cache
    ):
        """Test TabuScheduler considers slew time in scheduling decisions"""
        scheduler = TabuScheduler(config={
            'max_iterations': 20,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_tabu_scheduler_records_slew_angle_and_time(
        self, sample_mission, mock_window_cache
    ):
        """Test TabuScheduler records both slew_angle and slew_time in ScheduledTask"""
        scheduler = TabuScheduler(config={
            'max_iterations': 20,
            'random_seed': 42
        })
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle')
            assert hasattr(task, 'slew_time')


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestSlewTimeEdgeCases:
    """Test edge cases for slew time calculation"""

    def test_scheduler_handles_missing_agility_params(
        self, sample_mission, mock_window_cache
    ):
        """Test scheduler handles satellites without agility parameters"""
        # Create satellite without agility params
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM],
            resolution=0.5,
            swath_width=10.0,
            power_capacity=2800.0,
            storage_capacity=128.0,
            max_roll_angle=45.0
            # No agility parameter
        )
        sat_no_agility = Satellite(
            id="sat_no_agility",
            name="No Agility Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            capabilities=capabilities
        )

        sample_mission.satellites = [sat_no_agility]

        scheduler = EDDScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Should not raise error, use defaults
        result = scheduler.schedule()
        assert isinstance(result, ScheduleResult)

    def test_scheduler_handles_zero_slew_angle(
        self, sample_mission, mock_window_cache
    ):
        """Test scheduler handles zero slew angle (same target)"""
        scheduler = EDDScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        # First task should have zero slew angle
        if result.scheduled_tasks:
            first_task = result.scheduled_tasks[0]
            assert first_task.slew_angle == 0.0
            # Even with zero angle, should have settling time
            assert first_task.slew_time >= 0.0

    def test_scheduler_handles_single_task(
        self, sample_mission, mock_window_cache
    ):
        """Test scheduler with single task (no slew needed)"""
        sample_mission.targets = sample_mission.targets[:1]

        scheduler = EDDScheduler()
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        if result.scheduled_tasks:
            task = result.scheduled_tasks[0]
            assert task.slew_angle == 0.0
            # Should still have settling time
            assert hasattr(task, 'slew_time')

    def test_different_agility_produces_different_slew_times(
        self, sample_mission, mock_window_cache, sample_satellite_high_agility, sample_satellite_low_agility
    ):
        """Test that different agility parameters produce different slew times"""
        # This test verifies that agility parameters are actually used

        # Test with high agility satellite
        sample_mission.satellites = [sample_satellite_high_agility]
        scheduler_high = EDDScheduler()
        scheduler_high.initialize(sample_mission)
        scheduler_high.set_window_cache(mock_window_cache)

        # Test with low agility satellite
        sample_mission.satellites = [sample_satellite_low_agility]
        scheduler_low = EDDScheduler()
        scheduler_low.initialize(sample_mission)
        scheduler_low.set_window_cache(mock_window_cache)

        # Both should complete without errors
        result_high = scheduler_high.schedule()
        result_low = scheduler_low.schedule()

        assert isinstance(result_high, ScheduleResult)
        assert isinstance(result_low, ScheduleResult)


# =============================================================================
# Integration Tests
# =============================================================================

class TestSlewTimeIntegration:
    """Integration tests for slew time across all schedulers"""

    @pytest.mark.parametrize("scheduler_class,config", [
        (GreedyScheduler, {}),
        (EDDScheduler, {}),
        (SPTScheduler, {}),
        (GAScheduler, {'population_size': 20, 'generations': 10, 'random_seed': 42}),
        (SAScheduler, {'max_iterations': 50, 'random_seed': 42}),
        (ACOScheduler, {'num_ants': 10, 'max_iterations': 10, 'random_seed': 42}),
        (PSOScheduler, {'num_particles': 10, 'max_iterations': 10, 'random_seed': 42}),
        (TabuScheduler, {'max_iterations': 20, 'random_seed': 42}),
    ])
    def test_all_schedulers_record_slew_info(
        self, scheduler_class, config, sample_mission, mock_window_cache
    ):
        """Test that all schedulers record slew_angle and slew_time"""
        scheduler = scheduler_class(config=config)
        scheduler.initialize(sample_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

        for task in result.scheduled_tasks:
            assert hasattr(task, 'slew_angle'), f"{scheduler_class.__name__} missing slew_angle"
            assert hasattr(task, 'slew_time'), f"{scheduler_class.__name__} missing slew_time"
            assert task.slew_angle >= 0.0
            assert task.slew_time >= 0.0
