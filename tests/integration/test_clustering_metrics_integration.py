"""
Integration tests for ClusteringMetrics with ClusteringGreedyScheduler

Tests the integration between the metrics module and the actual scheduler.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from scheduler.clustering_greedy_scheduler import ClusteringGreedyScheduler
from scheduler.metrics.clustering_metrics import (
    ClusteringEfficiencyMetrics,
    ClusteringCoverageMetrics,
    ClusteringQualityScore,
    ClusteringMetricsCollector,
    ClusteringVisualizer,
)
from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, SatelliteCapabilities, ImagingMode, Orbit


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
def clustered_targets() -> List[Target]:
    """Create targets that will form clusters"""
    return [
        # Cluster 1: 3 nearby targets
        Target(id="t1", name="Target 1", target_type=TargetType.POINT,
               longitude=116.0, latitude=39.0, priority=5),
        Target(id="t2", name="Target 2", target_type=TargetType.POINT,
               longitude=116.005, latitude=39.0, priority=8),
        Target(id="t3", name="Target 3", target_type=TargetType.POINT,
               longitude=116.0, latitude=39.005, priority=5),
        # Cluster 2: 2 nearby targets
        Target(id="t4", name="Target 4", target_type=TargetType.POINT,
               longitude=116.3, latitude=39.0, priority=9),
        Target(id="t5", name="Target 5", target_type=TargetType.POINT,
               longitude=116.305, latitude=39.0, priority=5),
        # Distant target (unclustered)
        Target(id="t6", name="Target 6", target_type=TargetType.POINT,
               longitude=117.0, latitude=40.0, priority=3),
    ]


@pytest.fixture
def mock_mission(optical_satellite, clustered_targets):
    """Create a mock mission"""
    mission = Mock()
    mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
    mission.end_time = datetime(2024, 1, 2, 0, 0, 0)
    mission.satellites = [optical_satellite]
    mission.targets = clustered_targets
    return mission


class TestClusteringMetricsIntegration:
    """Integration tests for clustering metrics"""

    def test_metrics_collector_with_scheduler(self, mock_mission):
        """Test metrics collector works with a scheduler mock"""
        # Create scheduler mock with cluster schedules
        scheduler = Mock(spec=ClusteringGreedyScheduler)
        scheduler.mission = mock_mission

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        from scheduler.clustering_greedy_scheduler import ClusterSchedule
        scheduler.cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=mock_mission.targets[:3],
                satellite_id="optical_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=1
            ),
            ClusterSchedule(
                cluster_id="cluster_2",
                targets=mock_mission.targets[3:5],
                satellite_id="optical_1",
                imaging_start=base_time + timedelta(minutes=10),
                imaging_end=base_time + timedelta(minutes=14),
                look_angle=3.0,
                priority_satisfied=1
            ),
        ]

        # Create metrics collector
        collector = ClusteringMetricsCollector(scheduler)

        # Collect all metrics
        efficiency = collector.collect_efficiency_metrics()
        coverage = collector.collect_coverage_metrics()
        score = collector.calculate_quality_score()
        comparison = collector.compare_with_traditional()
        report = collector.generate_report()

        # Verify metrics
        assert efficiency.task_reduction_ratio > 0
        assert coverage.targets_covered == 5  # 3 + 2 clustered
        assert 0 <= score.overall_score <= 100
        assert comparison['improvement_ratio'] > 0
        assert 'efficiency' in report
        assert 'coverage' in report
        assert 'quality_score' in report

    def test_visualizer_with_scheduler(self, mock_mission):
        """Test visualizer works with a scheduler mock"""
        scheduler = Mock(spec=ClusteringGreedyScheduler)
        scheduler.mission = mock_mission

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        from scheduler.clustering_greedy_scheduler import ClusterSchedule
        scheduler.cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=mock_mission.targets[:3],
                satellite_id="optical_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=1
            ),
        ]

        # Create visualizer
        visualizer = ClusteringVisualizer(scheduler)

        # Get visualization data
        map_data = visualizer.prepare_cluster_map_data()
        heatmap_data = visualizer.prepare_coverage_heatmap_data()
        chart_data = visualizer.prepare_efficiency_chart_data()

        # Verify data structure
        assert len(map_data['clusters']) == 1
        assert len(map_data['unclustered_targets']) == 3  # 6 total - 3 clustered
        assert len(heatmap_data['coverage_points']) > 0
        assert len(chart_data['labels']) == 3
        assert len(chart_data['traditional_values']) == 3
        assert len(chart_data['clustering_values']) == 3

    def test_end_to_end_metrics_workflow(self, mock_mission):
        """Test complete metrics workflow"""
        scheduler = Mock(spec=ClusteringGreedyScheduler)
        scheduler.mission = mock_mission

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        from scheduler.clustering_greedy_scheduler import ClusterSchedule
        scheduler.cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=mock_mission.targets[:3],
                satellite_id="optical_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=1
            ),
            ClusterSchedule(
                cluster_id="cluster_2",
                targets=mock_mission.targets[3:5],
                satellite_id="optical_1",
                imaging_start=base_time + timedelta(minutes=10),
                imaging_end=base_time + timedelta(minutes=14),
                look_angle=3.0,
                priority_satisfied=1
            ),
        ]

        # Create collector and visualizer
        collector = ClusteringMetricsCollector(scheduler)
        visualizer = ClusteringVisualizer(scheduler)

        # Generate full report
        report = collector.generate_report()

        # Verify report structure
        assert 'efficiency' in report
        assert 'coverage' in report
        assert 'quality_score' in report
        assert 'comparison' in report
        assert 'summary' in report
        assert 'timestamp' in report

        # Verify summary
        summary = report['summary']
        assert summary['total_targets'] == 6
        assert summary['total_clustered_targets'] == 5
        assert summary['total_individual_targets'] == 1

        # Verify comparison
        comparison = report['comparison']
        assert comparison['traditional_task_count'] == 6
        assert comparison['clustering_task_count'] == 3  # 2 clusters + 1 individual
        assert comparison['improvement_ratio'] == pytest.approx(0.5, abs=0.01)

        # Get visualization data
        map_data = visualizer.prepare_cluster_map_data()
        assert len(map_data['clusters']) == 2

    def test_metrics_with_empty_scheduler(self):
        """Test metrics with empty scheduler"""
        scheduler = Mock(spec=ClusteringGreedyScheduler)
        scheduler.mission = Mock()
        scheduler.mission.targets = []
        scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(scheduler)
        visualizer = ClusteringVisualizer(scheduler)

        # All metrics should handle empty data gracefully
        efficiency = collector.collect_efficiency_metrics()
        coverage = collector.collect_coverage_metrics()
        score = collector.calculate_quality_score()
        comparison = collector.compare_with_traditional()
        report = collector.generate_report()

        assert efficiency.task_reduction_ratio == 0.0
        assert coverage.target_coverage_ratio == 0.0
        assert score.overall_score == 0.0
        assert comparison['improvement_ratio'] == 0.0

        map_data = visualizer.prepare_cluster_map_data()
        assert map_data['clusters'] == []
        assert map_data['unclustered_targets'] == []

    def test_metrics_with_all_targets_clustered(self, optical_satellite):
        """Test metrics when all targets are in clusters"""
        targets = [
            Target(id=f"t{i}", name=f"Target {i}", target_type=TargetType.POINT,
                   longitude=116.0 + i*0.001, latitude=39.0, priority=5)
            for i in range(5)
        ]

        scheduler = Mock(spec=ClusteringGreedyScheduler)
        scheduler.mission = Mock()
        scheduler.mission.targets = targets

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        from scheduler.clustering_greedy_scheduler import ClusterSchedule
        scheduler.cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=targets,
                satellite_id="optical_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=0
            ),
        ]

        collector = ClusteringMetricsCollector(scheduler)

        efficiency = collector.collect_efficiency_metrics()
        coverage = collector.collect_coverage_metrics()
        comparison = collector.compare_with_traditional()

        # All targets clustered
        assert efficiency.cluster_utilization_ratio == 1.0
        assert coverage.targets_covered == 5
        assert coverage.target_coverage_ratio == 1.0

        # Traditional: 5 tasks, Clustering: 1 task
        assert comparison['traditional_task_count'] == 5
        assert comparison['clustering_task_count'] == 1
        assert comparison['improvement_ratio'] == 0.8  # (5-1)/5 = 0.8
