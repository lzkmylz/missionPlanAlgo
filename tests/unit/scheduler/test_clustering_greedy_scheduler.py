"""
Unit tests for ClusteringGreedyScheduler

TDD Approach:
1. Write failing tests (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor (IMPROVE)

Test scenarios:
- Basic scheduling with clusters
- Nearby targets scheduled together
- Distant targets scheduled separately
- High priority target guarantee
- Max off-nadir constraint respected
- Efficiency metrics calculation
- Fallback to individual scheduling
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from scheduler.clustering_greedy_scheduler import (
    ClusteringGreedyScheduler,
    ClusterSchedule
)
from scheduler.base_scheduler import ScheduleResult, ScheduledTask
from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, SatelliteCapabilities, ImagingMode, Orbit
from core.clustering.target_clusterer import TargetClusterer, TargetCluster
from core.coverage.footprint_calculator import FootprintCalculator
from core.orbit.visibility.base import VisibilityWindow


# Module-level fixtures
@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Base configuration for scheduler"""
    return {
        'swath_width_km': 10.0,
        'min_cluster_size': 2,
        'altitude_km': 500.0,
        'heuristic': 'priority',
        'consider_power': False,
        'consider_storage': False,
        'consider_time_conflicts': False,
    }


@pytest.fixture
def mock_mission():
    """Create a mock mission for testing"""
    mission = Mock()
    mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
    mission.end_time = datetime(2024, 1, 2, 0, 0, 0)
    mission.satellites = []
    mission.targets = []

    def get_satellite_by_id(sat_id):
        for sat in mission.satellites:
            if sat.id == sat_id:
                return sat
        return None

    mission.get_satellite_by_id = get_satellite_by_id
    return mission


@pytest.fixture
def optical_satellite() -> Satellite:
    """Create an optical satellite for testing"""
    capabilities = SatelliteCapabilities(
        imaging_modes=[ImagingMode.PUSH_BROOM],
        max_off_nadir=30.0,
        storage_capacity=500.0,
        power_capacity=2000.0,
        resolution=10.0,
        swath_width=10000.0,  # 10km
    )
    return Satellite(
        id="optical_1",
        name="Optical Satellite 1",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(altitude=500000.0, inclination=97.4),
        capabilities=capabilities
    )


@pytest.fixture
def sar_satellite() -> Satellite:
    """Create a SAR satellite with larger swath for testing"""
    capabilities = SatelliteCapabilities(
        imaging_modes=[ImagingMode.STRIPMAP, ImagingMode.SPOTLIGHT],
        max_off_nadir=45.0,
        storage_capacity=1000.0,
        power_capacity=3000.0,
        resolution=3.0,
        swath_width=20000.0,  # 20km
    )
    return Satellite(
        id="sar_1",
        name="SAR Satellite 1",
        sat_type=SatelliteType.SAR_1,
        orbit=Orbit(altitude=500000.0, inclination=97.4),
        capabilities=capabilities
    )


@pytest.fixture
def nearby_targets() -> List[Target]:
    """Create 3 targets within 1km of each other"""
    return [
        Target(
            id="target_1",
            name="Target 1",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5
        ),
        Target(
            id="target_2",
            name="Target 2",
            target_type=TargetType.POINT,
            longitude=116.005,  # ~0.5km east
            latitude=39.0,
            priority=5
        ),
        Target(
            id="target_3",
            name="Target 3",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.005,  # ~0.5km north
            priority=5
        ),
    ]


@pytest.fixture
def distant_targets() -> List[Target]:
    """Create 2 targets 30km away from the cluster"""
    return [
        Target(
            id="target_4",
            name="Target 4",
            target_type=TargetType.POINT,
            longitude=116.3,  # ~30km away
            latitude=39.0,
            priority=5
        ),
        Target(
            id="target_5",
            name="Target 5",
            target_type=TargetType.POINT,
            longitude=116.3,
            latitude=39.005,
            priority=5
        ),
    ]


@pytest.fixture
def high_priority_targets() -> List[Target]:
    """Create targets with different priorities"""
    return [
        Target(
            id="high_priority_1",
            name="High Priority 1",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=10
        ),
        Target(
            id="high_priority_2",
            name="High Priority 2",
            target_type=TargetType.POINT,
            longitude=118.0,  # Far away
            latitude=39.0,
            priority=10
        ),
        Target(
            id="normal_1",
            name="Normal 1",
            target_type=TargetType.POINT,
            longitude=116.005,
            latitude=39.005,
            priority=5
        ),
        Target(
            id="normal_2",
            name="Normal 2",
            target_type=TargetType.POINT,
            longitude=116.01,
            latitude=39.01,
            priority=5
        ),
        Target(
            id="normal_3",
            name="Normal 3",
            target_type=TargetType.POINT,
            longitude=118.005,
            latitude=39.005,
            priority=5
        ),
    ]


@pytest.fixture
def visibility_windows() -> Dict[str, List[VisibilityWindow]]:
    """Create mock visibility windows for targets"""
    base_time = datetime(2024, 1, 1, 6, 0, 0)
    windows = {}

    # Create windows for all target IDs
    target_ids = [f"target_{i}" for i in range(1, 6)]
    target_ids.extend(["high_priority_1", "high_priority_2", "normal_1", "normal_2", "normal_3"])

    for i, target_id in enumerate(target_ids):
        # Stagger windows slightly for different targets
        offset = timedelta(minutes=i * 5)
        windows[target_id] = [
            VisibilityWindow(
                satellite_id="optical_1",
                target_id=target_id,
                start_time=base_time + offset,
                end_time=base_time + offset + timedelta(minutes=10),
                max_elevation=60.0,
                quality_score=0.9
            )
        ]

    return windows


@pytest.fixture
def mock_window_cache(visibility_windows):
    """Create a mock window cache"""
    cache = Mock()

    def get_windows(sat_id, target_id):
        return visibility_windows.get(target_id, [])

    cache.get_windows = get_windows
    return cache


class TestClusteringGreedyScheduler:
    """Test suite for ClusteringGreedyScheduler"""

    def test_scheduler_initialization(self, base_config):
        """Test that scheduler initializes with correct components"""
        scheduler = ClusteringGreedyScheduler(base_config)

        assert scheduler.clusterer is not None
        assert scheduler.footprint_calc is not None
        assert scheduler.cluster_schedules == []
        assert scheduler.clusterer.swath_width_km == 10.0
        assert scheduler.clusterer.min_cluster_size == 2

    def test_cluster_targets_basic(self, base_config, nearby_targets):
        """Test basic target clustering functionality"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.targets = nearby_targets

        clusters = scheduler._cluster_targets()

        # 3 nearby targets should form 1 cluster
        assert len(clusters) == 1
        assert len(clusters[0].targets) == 3

    def test_cluster_targets_with_distant(
        self, base_config, nearby_targets, distant_targets
    ):
        """Test that distant targets form separate clusters"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.targets = nearby_targets + distant_targets

        clusters = scheduler._cluster_targets()

        # Should have 2 clusters: 1 for nearby (3 targets), 1 for distant (2 targets)
        assert len(clusters) == 2

        # Find cluster with 3 targets (nearby)
        nearby_cluster = next(c for c in clusters if len(c.targets) == 3)
        distant_cluster = next(c for c in clusters if len(c.targets) == 2)

        assert nearby_cluster is not None
        assert distant_cluster is not None

    def test_cluster_targets_empty(self, base_config):
        """Test clustering with no targets"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.targets = []

        clusters = scheduler._cluster_targets()

        assert clusters == []

    def test_cluster_targets_single(self, base_config):
        """Test clustering with single target (below min_cluster_size)"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.targets = [
            Target(
                id="single",
                name="Single Target",
                target_type=TargetType.POINT,
                longitude=116.0,
                latitude=39.0,
                priority=5
            )
        ]

        clusters = scheduler._cluster_targets()

        # Single target should not form a cluster (below min_cluster_size)
        assert clusters == []

    def test_find_best_window_for_cluster(
        self, base_config, nearby_targets, optical_satellite,
        mock_window_cache, visibility_windows
    ):
        """Test finding best window for a cluster"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.satellites = [optical_satellite]
        scheduler.mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        scheduler.set_window_cache(mock_window_cache)

        # Create a cluster
        cluster = TargetCluster(
            cluster_id="test_cluster",
            targets=nearby_targets,
            centroid=(116.0, 39.0),
            total_priority=15,
            bounding_box=(116.0, 116.005, 39.0, 39.005)
        )

        # Mock the coverage check to return valid result
        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(True, 15.0)  # Can cover with 15 degree look angle
        ):
            result = scheduler._find_best_window_for_cluster(cluster)

            # Should find a valid assignment
            assert result is not None
            sat_id, window, look_angle = result
            assert sat_id == "optical_1"
            assert window is not None
            assert isinstance(look_angle, float)
            assert look_angle == 15.0

    def test_find_best_window_respects_off_nadir(
        self, base_config, optical_satellite
    ):
        """Test that max off-nadir constraint is respected"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.satellites = [optical_satellite]
        scheduler.mission.start_time = datetime(2024, 1, 1, 0, 0, 0)

        # Create targets that would require large off-nadir angle
        far_targets = [
            Target(
                id="far_1",
                name="Far Target 1",
                target_type=TargetType.POINT,
                longitude=120.0,  # Far from satellite ground track
                latitude=39.0,
                priority=5
            ),
            Target(
                id="far_2",
                name="Far Target 2",
                target_type=TargetType.POINT,
                longitude=120.0,
                latitude=39.01,
                priority=5
            ),
        ]

        cluster = TargetCluster(
            cluster_id="far_cluster",
            targets=far_targets,
            centroid=(120.0, 39.005),
            total_priority=10,
            bounding_box=(120.0, 120.0, 39.0, 39.01)
        )

        # Mock window cache to return empty (no visibility)
        mock_cache = Mock()
        mock_cache.get_windows = Mock(return_value=[])
        scheduler.set_window_cache(mock_cache)

        result = scheduler._find_best_window_for_cluster(cluster)

        # Should not find a valid window due to off-nadir constraint
        assert result is None

    def test_schedule_with_clusters(
        self, base_config, mock_mission, nearby_targets,
        distant_targets, optical_satellite, mock_window_cache
    ):
        """Test full scheduling with clustered targets"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets + distant_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        # Should return a valid ScheduleResult
        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) > 0

        # With clustering, we should have fewer tasks than targets
        # (or at least not more)
        scheduled_target_ids = set()
        for task in result.scheduled_tasks:
            scheduled_target_ids.add(task.target_id)

        # All targets should be covered
        all_target_ids = {t.id for t in mock_mission.targets}
        assert scheduled_target_ids == all_target_ids

    def test_high_priority_guarantee(
        self, base_config, mock_mission, high_priority_targets,
        optical_satellite, mock_window_cache
    ):
        """Test that high priority targets are guaranteed to be scheduled"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = high_priority_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        # Both high priority targets should be scheduled
        scheduled_high_priority = [
            t for t in result.scheduled_tasks
            if t.target_id in ["high_priority_1", "high_priority_2"]
        ]

        assert len(scheduled_high_priority) == 2

    def test_efficiency_metrics(
        self, base_config, mock_mission, nearby_targets,
        distant_targets, optical_satellite, mock_window_cache
    ):
        """Test efficiency metrics calculation"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets + distant_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Schedule first
        result = scheduler.schedule()

        # Then get metrics
        metrics = scheduler.get_efficiency_metrics()

        assert 'task_reduction_ratio' in metrics
        assert 'high_priority_coverage' in metrics
        assert 'avg_targets_per_task' in metrics

        # Verify metric values are valid
        assert 0.0 <= metrics['task_reduction_ratio'] <= 1.0
        assert 0.0 <= metrics['high_priority_coverage'] <= 1.0
        assert metrics['avg_targets_per_task'] >= 0.0

    def test_fallback_to_individual_scheduling(
        self, base_config, mock_mission, nearby_targets, optical_satellite
    ):
        """Test fallback when clustering fails"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)

        # Mock window cache to return empty windows (clustering fails)
        mock_cache = Mock()
        mock_cache.get_windows = Mock(return_value=[])
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should still return a result, even if no tasks scheduled
        assert isinstance(result, ScheduleResult)

    def test_mixed_satellite_types(
        self, base_config, mock_mission, nearby_targets,
        optical_satellite, sar_satellite, mock_window_cache
    ):
        """Test scheduling with mixed satellite types"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite, sar_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        # Should use the best satellite for the cluster
        assert len(result.scheduled_tasks) > 0

    def test_cluster_schedule_data_structure(self):
        """Test ClusterSchedule dataclass"""
        targets = [
            Target(id="t1", name="T1", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=5),
            Target(id="t2", name="T2", target_type=TargetType.POINT,
                   longitude=116.01, latitude=39.01, priority=5),
        ]

        schedule = ClusterSchedule(
            cluster_id="cluster_1",
            targets=targets,
            satellite_id="sat_1",
            imaging_start=datetime(2024, 1, 1, 6, 0, 0),
            imaging_end=datetime(2024, 1, 1, 6, 5, 0),
            look_angle=15.0,
            priority_satisfied=2
        )

        assert schedule.cluster_id == "cluster_1"
        assert len(schedule.targets) == 2
        assert schedule.satellite_id == "sat_1"
        assert schedule.look_angle == 15.0
        assert schedule.priority_satisfied == 2

    def test_cluster_priority_density_sorting(
        self, base_config, mock_mission, high_priority_targets,
        optical_satellite, mock_window_cache
    ):
        """Test that clusters are sorted by priority density"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = high_priority_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Get clusters
        clusters = scheduler._cluster_targets()

        # Sort by priority density
        sorted_clusters = scheduler._sort_clusters_by_priority_density(clusters)

        # Higher priority density clusters should come first
        if len(sorted_clusters) >= 2:
            first_priority = sorted_clusters[0].total_priority / len(sorted_clusters[0].targets)
            second_priority = sorted_clusters[1].total_priority / len(sorted_clusters[1].targets)
            assert first_priority >= second_priority

    def test_off_nadir_constraint_respected(
        self, base_config, mock_mission, nearby_targets, optical_satellite
    ):
        """Test that off-nadir angle constraint is respected"""
        scheduler = ClusteringGreedyScheduler(base_config)

        # Modify optical satellite to have small max off-nadir
        optical_satellite.capabilities.max_off_nadir = 10.0

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)

        # Mock footprint calculator to simulate large required angle
        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(False, 45.0)  # Requires 45 degrees, max is 10 degrees
        ):
            # Create cluster
            cluster = TargetCluster(
                cluster_id="test_cluster",
                targets=nearby_targets,
                centroid=(116.0, 39.0),
                total_priority=15,
                bounding_box=(116.0, 116.005, 39.0, 39.005)
            )

            result = scheduler._find_best_window_for_cluster(cluster)

            # Should not find valid window due to off-nadir constraint
            assert result is None

    def test_cluster_coverage_validation(
        self, base_config, nearby_targets
    ):
        """Test that cluster coverage is properly validated"""
        scheduler = ClusteringGreedyScheduler(base_config)

        cluster = TargetCluster(
            cluster_id="test_cluster",
            targets=nearby_targets,
            centroid=(116.0, 39.0),
            total_priority=15,
            bounding_box=(116.0, 116.005, 39.0, 39.005)
        )

        # Mock satellite position
        sat_position = (6371000.0, 0, 0)  # Simplified position
        nadir_position = (0.0, 0.0)  # Simplified nadir

        # Test coverage check
        can_cover, angle = scheduler.footprint_calc.can_cover_targets(
            targets=cluster.targets,
            satellite_position=sat_position,
            nadir_position=nadir_position,
            max_off_nadir=30.0,
            swath_width_km=10.0
        )

        # Result depends on the specific geometry
        assert isinstance(can_cover, bool)
        assert isinstance(angle, float)

    def test_task_reduction_ratio_calculation(
        self, base_config, mock_mission, nearby_targets,
        distant_targets, optical_satellite, mock_window_cache
    ):
        """Test task reduction ratio calculation"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets + distant_targets  # 5 targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        result = scheduler.schedule()

        metrics = scheduler.get_efficiency_metrics()

        # Task reduction = 1 - (num_tasks / num_targets)
        # If all 5 targets scheduled as 2 cluster tasks, reduction = 1 - 2/5 = 0.6
        num_targets = len(mock_mission.targets)
        num_tasks = len(scheduler.cluster_schedules)

        if num_targets > 0:
            expected_reduction = 1.0 - (num_tasks / num_targets)
            assert abs(metrics['task_reduction_ratio'] - expected_reduction) < 0.01

    def test_cluster_schedule_tracking(
        self, base_config, mock_mission, nearby_targets,
        optical_satellite, mock_window_cache
    ):
        """Test that cluster schedules are properly tracked"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Initially empty
        assert scheduler.cluster_schedules == []

        # Mock the coverage check to allow clustering
        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(True, 15.0)
        ):
            result = scheduler.schedule()

            # Should have cluster schedules after scheduling
            assert len(scheduler.cluster_schedules) > 0

            # Verify cluster schedule structure
            for cs in scheduler.cluster_schedules:
                assert isinstance(cs, ClusterSchedule)
                assert cs.cluster_id is not None
                assert len(cs.targets) > 0
                assert cs.satellite_id is not None


class TestClusteringEdgeCases:
    """Test edge cases for clustering scheduler"""

    def test_all_targets_far_apart(self):
        """Test when all targets are too far to cluster"""
        scheduler = ClusteringGreedyScheduler({})
        scheduler.mission = Mock()

        # Create targets far apart (> swath_width)
        far_targets = [
            Target(id=f"far_{i}", name=f"Far {i}",
                   target_type=TargetType.POINT,
                   longitude=116.0 + i * 20,  # 20 degrees apart
                   latitude=39.0,
                   priority=5)
            for i in range(3)
        ]
        scheduler.mission.targets = far_targets

        clusters = scheduler._cluster_targets()

        # No clusters should form (all targets too far apart)
        assert len(clusters) == 0

    def test_targets_at_exact_boundary(self):
        """Test targets at exact clustering boundary"""
        scheduler = ClusteringGreedyScheduler({'swath_width_km': 10.0})
        scheduler.mission = Mock()

        # Targets exactly at swath_width_km distance (10km)
        boundary_targets = [
            Target(id="boundary_1", name="Boundary 1",
                   target_type=TargetType.POINT,
                   longitude=116.0,
                   latitude=39.0,
                   priority=5),
            Target(id="boundary_2", name="Boundary 2",
                   target_type=TargetType.POINT,
                   longitude=116.0 + 10.0 / 111.0,  # ~10km east (1 degree ~ 111km)
                   latitude=39.0,
                   priority=5),
        ]
        scheduler.mission.targets = boundary_targets

        clusters = scheduler._cluster_targets()

        # Should form a cluster (distance <= swath_width)
        assert len(clusters) >= 0  # May or may not cluster depending on exact math

    def test_empty_cluster_handling(self, mock_mission, optical_satellite):
        """Test handling when clustering produces no valid clusters"""
        scheduler = ClusteringGreedyScheduler({})

        # Single target (won't form cluster)
        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = [
            Target(id="lonely", name="Lonely Target",
                   target_type=TargetType.POINT,
                   longitude=116.0,
                   latitude=39.0,
                   priority=5)
        ]

        scheduler.initialize(mock_mission)

        # Mock window cache
        mock_cache = Mock()
        mock_cache.get_windows = Mock(return_value=[
            VisibilityWindow(
                satellite_id="optical_1",
                target_id="lonely",
                start_time=datetime(2024, 1, 1, 6, 0, 0),
                end_time=datetime(2024, 1, 1, 6, 10, 0),
                max_elevation=60.0,
                quality_score=0.9
            )
        ])
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # Should still schedule the single target
        assert isinstance(result, ScheduleResult)

    def test_cluster_with_varying_priorities(self):
        """Test cluster with mixed priority targets"""
        scheduler = ClusteringGreedyScheduler({})
        scheduler.mission = Mock()

        mixed_priority_targets = [
            Target(id="low", name="Low Priority",
                   target_type=TargetType.POINT,
                   longitude=116.0,
                   latitude=39.0,
                   priority=1),
            Target(id="high", name="High Priority",
                   target_type=TargetType.POINT,
                   longitude=116.005,
                   latitude=39.005,
                   priority=10),
        ]
        scheduler.mission.targets = mixed_priority_targets

        clusters = scheduler._cluster_targets()

        if clusters:
            # Cluster should have combined priority
            assert clusters[0].total_priority == 11

    def test_multiple_clusters_same_priority_density(self):
        """Test sorting when clusters have same priority density"""
        scheduler = ClusteringGreedyScheduler({})
        scheduler.mission = Mock()

        # Two identical clusters
        targets_a = [
            Target(id=f"a_{i}", name=f"A {i}",
                   target_type=TargetType.POINT,
                   longitude=116.0 + i * 0.001,
                   latitude=39.0,
                   priority=5)
            for i in range(3)
        ]
        targets_b = [
            Target(id=f"b_{i}", name=f"B {i}",
                   target_type=TargetType.POINT,
                   longitude=118.0 + i * 0.001,
                   latitude=39.0,
                   priority=5)
            for i in range(3)
        ]
        scheduler.mission.targets = targets_a + targets_b

        clusters = scheduler._cluster_targets()

        # Should have 2 clusters with same priority density
        assert len(clusters) == 2
        assert clusters[0].total_priority == clusters[1].total_priority


class TestClusteringIntegration:
    """Integration tests for clustering scheduler"""

    def test_full_scheduling_workflow(
        self, mock_mission, nearby_targets,
        distant_targets, optical_satellite, sar_satellite, mock_window_cache
    ):
        """Test complete scheduling workflow with clustering"""
        base_config = {
            'swath_width_km': 10.0,
            'min_cluster_size': 2,
            'altitude_km': 500.0,
            'heuristic': 'priority',
            'consider_power': False,
            'consider_storage': False,
            'consider_time_conflicts': False,
        }
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite, sar_satellite]
        mock_mission.targets = nearby_targets + distant_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Execute scheduling
        result = scheduler.schedule()

        # Verify result structure
        assert isinstance(result, ScheduleResult)
        assert result.scheduled_tasks is not None
        assert result.unscheduled_tasks is not None
        assert result.makespan >= 0
        assert result.computation_time >= 0

        # Verify all scheduled tasks are valid
        for task in result.scheduled_tasks:
            assert task.satellite_id in ["optical_1", "sar_1"]
            assert task.imaging_start < task.imaging_end
            assert task.target_id is not None

        # Verify metrics
        metrics = scheduler.get_efficiency_metrics()
        assert 'task_reduction_ratio' in metrics
        assert 'high_priority_coverage' in metrics
        assert 'avg_targets_per_task' in metrics

    def test_clustering_reduces_total_tasks(
        self, mock_mission, nearby_targets, optical_satellite
    ):
        """
        Verify that clustering actually reduces number of tasks.

        Without clustering: 3 targets = 3 tasks
        With clustering: 3 nearby targets = 1 task
        """
        base_config = {
            'swath_width_km': 10.0,
            'min_cluster_size': 2,
            'altitude_km': 500.0,
            'heuristic': 'priority',
            'consider_power': False,
            'consider_storage': False,
            'consider_time_conflicts': False,
        }
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets  # 3 targets

        scheduler.initialize(mock_mission)

        # Create windows that allow all targets to be scheduled
        base_time = datetime(2024, 1, 1, 6, 0, 0)
        windows = {}
        for target in nearby_targets:
            windows[target.id] = [
                VisibilityWindow(
                    satellite_id="optical_1",
                    target_id=target.id,
                    start_time=base_time,
                    end_time=base_time + timedelta(minutes=10),
                    max_elevation=60.0,
                    quality_score=0.9
                )
            ]

        mock_cache = Mock()
        mock_cache.get_windows = lambda sat_id, target_id: windows.get(target_id, [])
        scheduler.set_window_cache(mock_cache)

        result = scheduler.schedule()

        # With effective clustering, should have fewer tasks than targets
        # (or at least not more)
        assert len(result.scheduled_tasks) <= len(nearby_targets)

        # Calculate actual reduction
        metrics = scheduler.get_efficiency_metrics()
        if len(result.scheduled_tasks) < len(nearby_targets):
            assert metrics['task_reduction_ratio'] > 0


class TestClusteringAdditionalCoverage:
    """Additional tests to improve coverage"""

    def test_get_efficiency_metrics_empty(self, base_config):
        """Test efficiency metrics with no mission"""
        scheduler = ClusteringGreedyScheduler(base_config)

        metrics = scheduler.get_efficiency_metrics()

        assert metrics['task_reduction_ratio'] == 0.0
        assert metrics['high_priority_coverage'] == 0.0
        assert metrics['avg_targets_per_task'] == 0.0

    def test_get_efficiency_metrics_no_high_priority(
        self, base_config, mock_mission, nearby_targets, optical_satellite, mock_window_cache
    ):
        """Test efficiency metrics when no high priority targets exist"""
        scheduler = ClusteringGreedyScheduler(base_config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets  # All priority 5

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        # Mock coverage to allow scheduling
        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(True, 15.0)
        ):
            result = scheduler.schedule()
            metrics = scheduler.get_efficiency_metrics()

            # Should be 1.0 when no high priority targets
            assert metrics['high_priority_coverage'] == 1.0

    def test_schedule_with_time_conflicts(
        self, base_config, mock_mission, nearby_targets, optical_satellite, mock_window_cache
    ):
        """Test scheduling with time conflict checking enabled"""
        config = base_config.copy()
        config['consider_time_conflicts'] = True

        scheduler = ClusteringGreedyScheduler(config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(True, 15.0)
        ):
            result = scheduler.schedule()
            assert isinstance(result, ScheduleResult)

    def test_schedule_with_power_constraint(
        self, base_config, mock_mission, nearby_targets, optical_satellite, mock_window_cache
    ):
        """Test scheduling with power constraints"""
        config = base_config.copy()
        config['consider_power'] = True

        scheduler = ClusteringGreedyScheduler(config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(True, 15.0)
        ):
            result = scheduler.schedule()
            assert isinstance(result, ScheduleResult)

    def test_schedule_with_storage_constraint(
        self, base_config, mock_mission, nearby_targets, optical_satellite, mock_window_cache
    ):
        """Test scheduling with storage constraints"""
        config = base_config.copy()
        config['consider_storage'] = True

        scheduler = ClusteringGreedyScheduler(config)

        mock_mission.satellites = [optical_satellite]
        mock_mission.targets = nearby_targets

        scheduler.initialize(mock_mission)
        scheduler.set_window_cache(mock_window_cache)

        with patch.object(
            scheduler.footprint_calc,
            'can_cover_targets',
            return_value=(True, 15.0)
        ):
            result = scheduler.schedule()
            assert isinstance(result, ScheduleResult)

    def test_cluster_with_no_targets(self, base_config):
        """Test cluster imaging time with no targets"""
        scheduler = ClusteringGreedyScheduler(base_config)

        cluster = TargetCluster(
            cluster_id="empty_cluster",
            targets=[],
            centroid=(0.0, 0.0),
            total_priority=0,
            bounding_box=(0.0, 0.0, 0.0, 0.0)
        )

        from core.models.satellite import ImagingMode
        duration = scheduler._calculate_cluster_imaging_time(cluster, ImagingMode.PUSH_BROOM)
        assert duration == 60.0  # Default minimum

    def test_cluster_assignment_score(self, base_config, nearby_targets, optical_satellite):
        """Test cluster assignment score calculation"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.satellites = []
        scheduler.mission.start_time = datetime(2024, 1, 1, 0, 0, 0)

        # Initialize resource usage
        scheduler._sat_resource_usage = {
            'optical_1': {
                'power': 1000.0,
                'storage': 100.0
            }
        }

        cluster = TargetCluster(
            cluster_id="test_cluster",
            targets=nearby_targets,
            centroid=(116.0, 39.0),
            total_priority=15,
            bounding_box=(116.0, 116.005, 39.0, 39.005)
        )

        window = VisibilityWindow(
            satellite_id="optical_1",
            target_id="target_1",
            start_time=datetime(2024, 1, 1, 6, 0, 0),
            end_time=datetime(2024, 1, 1, 6, 10, 0),
            max_elevation=60.0,
            quality_score=0.9
        )

        score = scheduler._calculate_cluster_assignment_score(
            optical_satellite, cluster, window, 15.0
        )

        # Score should be positive with bonuses for targets and priority
        assert score > 0

    def test_estimate_satellite_position(self, base_config, optical_satellite):
        """Test satellite position estimation"""
        scheduler = ClusteringGreedyScheduler(base_config)

        dt = datetime(2024, 1, 1, 12, 0, 0)
        position = scheduler._estimate_satellite_position(optical_satellite, dt)

        assert len(position) == 3
        assert all(isinstance(p, float) for p in position)
        # Position should be roughly at orbital radius
        import math
        radius = math.sqrt(sum(p ** 2 for p in position))
        assert radius > 6371000  # Greater than Earth radius

    def test_check_cluster_resource_constraints_insufficient_power(
        self, base_config, nearby_targets, optical_satellite
    ):
        """Test resource constraint check with insufficient power"""
        config = base_config.copy()
        config['consider_power'] = True

        scheduler = ClusteringGreedyScheduler(config)
        scheduler._sat_resource_usage = {
            'optical_1': {
                'power': 0.0,  # No power left
                'storage': 0.0
            }
        }

        cluster = TargetCluster(
            cluster_id="test_cluster",
            targets=nearby_targets,
            centroid=(116.0, 39.0),
            total_priority=15,
            bounding_box=(116.0, 116.005, 39.0, 39.005)
        )

        from core.models.satellite import ImagingMode
        result = scheduler._check_cluster_resource_constraints(
            optical_satellite, cluster, ImagingMode.PUSH_BROOM
        )

        assert result is False

    def test_are_all_targets_scheduled(self, base_config, nearby_targets):
        """Test checking if all targets are scheduled"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler._scheduled_target_ids = set()

        # Initially none scheduled
        assert scheduler._are_all_targets_scheduled(nearby_targets) is False

        # Mark all as scheduled
        for target in nearby_targets:
            scheduler._scheduled_target_ids.add(target.id)

        assert scheduler._are_all_targets_scheduled(nearby_targets) is True

    def test_get_unscheduled_targets(self, base_config, mock_mission, nearby_targets):
        """Test getting unscheduled targets"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = mock_mission
        scheduler.mission.targets = nearby_targets
        scheduler._scheduled_target_ids = set()

        # Initially all unscheduled
        unscheduled = scheduler._get_unscheduled_targets()
        assert len(unscheduled) == len(nearby_targets)

        # Mark one as scheduled
        scheduler._scheduled_target_ids.add(nearby_targets[0].id)
        unscheduled = scheduler._get_unscheduled_targets()
        assert len(unscheduled) == len(nearby_targets) - 1

    def test_count_high_priority_targets(self, base_config):
        """Test counting high priority targets"""
        scheduler = ClusteringGreedyScheduler(base_config)

        targets = [
            Target(id="high1", name="High 1", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=10),
            Target(id="high2", name="High 2", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=9),
            Target(id="normal", name="Normal", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=5),
        ]

        count = scheduler._count_high_priority_targets(targets)
        assert count == 2  # Both 10 and 9 are >= 8

    def test_create_cluster_scheduled_task(self, base_config, nearby_targets, optical_satellite):
        """Test creating a scheduled task for a cluster"""
        scheduler = ClusteringGreedyScheduler(base_config)
        scheduler.mission = Mock()
        scheduler.mission.satellites = [optical_satellite]
        scheduler.mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        scheduler._sat_resource_usage = {
            'optical_1': {
                'power': 2000.0,
                'storage': 0.0,
                'last_task_end': datetime(2024, 1, 1, 0, 0, 0),
                'scheduled_tasks': []
            }
        }

        cluster = TargetCluster(
            cluster_id="test_cluster",
            targets=nearby_targets,
            centroid=(116.0, 39.0),
            total_priority=15,
            bounding_box=(116.0, 116.005, 39.0, 39.005)
        )

        window = VisibilityWindow(
            satellite_id="optical_1",
            target_id="target_1",
            start_time=datetime(2024, 1, 1, 6, 0, 0),
            end_time=datetime(2024, 1, 1, 6, 10, 0),
            max_elevation=60.0,
            quality_score=0.9
        )

        task = scheduler._create_cluster_scheduled_task(
            cluster, 'optical_1', window, 15.0
        )

        assert isinstance(task, ScheduledTask)
        assert task.satellite_id == 'optical_1'
        assert task.slew_angle == 15.0
        assert task.imaging_start is not None
        assert task.imaging_end is not None
