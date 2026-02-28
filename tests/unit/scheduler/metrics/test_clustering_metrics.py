"""
Unit tests for ClusteringMetricsCollector and ClusteringVisualizer

TDD Approach:
1. Write failing tests (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor (IMPROVE)

Test scenarios:
- Efficiency metrics calculation
- Coverage metrics calculation
- Quality score calculation
- Comparison with traditional scheduling
- Report generation
- Visualization data preparation
- Edge cases (empty schedule, all targets clustered, etc.)
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

from scheduler.metrics.clustering_metrics import (
    ClusteringEfficiencyMetrics,
    ClusteringCoverageMetrics,
    ClusteringQualityScore,
    ClusteringMetricsCollector,
    ClusteringVisualizer,
)
from scheduler.clustering_greedy_scheduler import (
    ClusteringGreedyScheduler,
    ClusterSchedule
)
from core.models.target import Target, TargetType
from core.models.satellite import Satellite, SatelliteType, SatelliteCapabilities, ImagingMode, Orbit


# =============================================================================
# Fixtures
# =============================================================================

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
def mock_scheduler():
    """Create a mock scheduler for testing"""
    scheduler = Mock(spec=ClusteringGreedyScheduler)
    scheduler.cluster_schedules = []
    scheduler.mission = Mock()
    scheduler.mission.targets = []
    return scheduler


@pytest.fixture
def sample_targets() -> List[Target]:
    """Create sample targets for testing"""
    return [
        Target(id="t1", name="Target 1", target_type=TargetType.POINT,
               longitude=116.0, latitude=39.0, priority=5),
        Target(id="t2", name="Target 2", target_type=TargetType.POINT,
               longitude=116.005, latitude=39.0, priority=8),  # High priority
        Target(id="t3", name="Target 3", target_type=TargetType.POINT,
               longitude=116.0, latitude=39.005, priority=5),
        Target(id="t4", name="Target 4", target_type=TargetType.POINT,
               longitude=116.3, latitude=39.0, priority=9),   # High priority
        Target(id="t5", name="Target 5", target_type=TargetType.POINT,
               longitude=116.305, latitude=39.0, priority=5),
        Target(id="t6", name="Target 6", target_type=TargetType.POINT,
               longitude=117.0, latitude=40.0, priority=3),   # Distant, unclustered
        Target(id="t7", name="Target 7", target_type=TargetType.POINT,
               longitude=117.1, latitude=40.1, priority=8),   # High priority, distant
    ]


@pytest.fixture
def cluster_schedules(sample_targets) -> List[ClusterSchedule]:
    """Create sample cluster schedules for testing"""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    return [
        ClusterSchedule(
            cluster_id="cluster_1",
            targets=[sample_targets[0], sample_targets[1], sample_targets[2]],  # 3 targets
            satellite_id="sat_1",
            imaging_start=base_time,
            imaging_end=base_time + timedelta(minutes=5),
            look_angle=5.0,
            priority_satisfied=1  # 1 high priority (t2)
        ),
        ClusterSchedule(
            cluster_id="cluster_2",
            targets=[sample_targets[3], sample_targets[4]],  # 2 targets
            satellite_id="sat_1",
            imaging_start=base_time + timedelta(minutes=10),
            imaging_end=base_time + timedelta(minutes=14),
            look_angle=3.0,
            priority_satisfied=1  # 1 high priority (t4)
        ),
    ]


@pytest.fixture
def metrics_collector(mock_scheduler, sample_targets, cluster_schedules):
    """Create a metrics collector with sample data"""
    mock_scheduler.mission.targets = sample_targets
    mock_scheduler.cluster_schedules = cluster_schedules
    return ClusteringMetricsCollector(mock_scheduler)


# =============================================================================
# Test ClusteringEfficiencyMetrics Dataclass
# =============================================================================

class TestClusteringEfficiencyMetrics:
    """Test ClusteringEfficiencyMetrics dataclass"""

    def test_efficiency_metrics_creation(self):
        """Test creating efficiency metrics"""
        metrics = ClusteringEfficiencyMetrics(
            task_reduction_ratio=0.4,
            task_reduction_count=4,
            time_savings_seconds=1200.0,
            avg_targets_per_task=2.5,
            max_targets_in_single_task=5,
            cluster_utilization_ratio=0.8
        )

        assert metrics.task_reduction_ratio == 0.4
        assert metrics.task_reduction_count == 4
        assert metrics.time_savings_seconds == 1200.0
        assert metrics.avg_targets_per_task == 2.5
        assert metrics.max_targets_in_single_task == 5
        assert metrics.cluster_utilization_ratio == 0.8

    def test_efficiency_metrics_defaults(self):
        """Test efficiency metrics with default values"""
        metrics = ClusteringEfficiencyMetrics(
            task_reduction_ratio=0.0,
            task_reduction_count=0,
            time_savings_seconds=0.0,
            avg_targets_per_task=0.0,
            max_targets_in_single_task=0,
            cluster_utilization_ratio=0.0
        )

        assert metrics.task_reduction_ratio == 0.0
        assert metrics.task_reduction_count == 0


# =============================================================================
# Test ClusteringCoverageMetrics Dataclass
# =============================================================================

class TestClusteringCoverageMetrics:
    """Test ClusteringCoverageMetrics dataclass"""

    def test_coverage_metrics_creation(self):
        """Test creating coverage metrics"""
        metrics = ClusteringCoverageMetrics(
            target_coverage_ratio=0.7,
            targets_covered=7,
            targets_total=10,
            high_priority_coverage=0.8,
            high_priority_covered=4,
            high_priority_total=5,
            area_coverage_km2=150.5
        )

        assert metrics.target_coverage_ratio == 0.7
        assert metrics.targets_covered == 7
        assert metrics.targets_total == 10
        assert metrics.high_priority_coverage == 0.8
        assert metrics.high_priority_covered == 4
        assert metrics.high_priority_total == 5
        assert metrics.area_coverage_km2 == 150.5


# =============================================================================
# Test ClusteringQualityScore Dataclass
# =============================================================================

class TestClusteringQualityScore:
    """Test ClusteringQualityScore dataclass"""

    def test_quality_score_creation(self):
        """Test creating quality score"""
        score = ClusteringQualityScore(
            overall_score=75.5,
            efficiency_score=80.0,
            coverage_score=70.0,
            priority_score=85.0,
            balance_score=75.0
        )

        assert score.overall_score == 75.5
        assert score.efficiency_score == 80.0
        assert score.coverage_score == 70.0
        assert score.priority_score == 85.0
        assert score.balance_score == 75.0


# =============================================================================
# Test ClusteringMetricsCollector - Efficiency Metrics
# =============================================================================

class TestClusteringMetricsCollectorEfficiency:
    """Test efficiency metrics collection"""

    def test_collect_efficiency_metrics_basic(self, metrics_collector, sample_targets, cluster_schedules):
        """Test basic efficiency metrics calculation

        Scenario: 7 total targets, 2 cluster schedules (3 + 2 = 5 targets clustered)
        Traditional: 7 tasks
        Clustering: 2 cluster tasks + 2 individual tasks = 4 tasks
        Reduction: (7-4)/7 = 0.428... ~ 43%
        """
        metrics = metrics_collector.collect_efficiency_metrics()

        assert isinstance(metrics, ClusteringEfficiencyMetrics)
        assert metrics.task_reduction_ratio == pytest.approx(0.428, abs=0.01)
        assert metrics.task_reduction_count == 3  # 7 - 4 = 3
        assert metrics.avg_targets_per_task == 2.5  # (3+2)/2 = 2.5
        assert metrics.max_targets_in_single_task == 3

    def test_collect_efficiency_metrics_empty_schedule(self, mock_scheduler):
        """Test efficiency metrics with empty schedule"""
        mock_scheduler.mission.targets = []
        mock_scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(mock_scheduler)
        metrics = collector.collect_efficiency_metrics()

        assert metrics.task_reduction_ratio == 0.0
        assert metrics.task_reduction_count == 0
        assert metrics.avg_targets_per_task == 0.0
        assert metrics.max_targets_in_single_task == 0
        assert metrics.time_savings_seconds == 0.0

    def test_collect_efficiency_metrics_all_clustered(self, mock_scheduler):
        """Test efficiency metrics when all targets are clustered"""
        targets = [
            Target(id=f"t{i}", name=f"Target {i}", target_type=TargetType.POINT,
                   longitude=116.0 + i*0.001, latitude=39.0, priority=5)
            for i in range(5)
        ]

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=targets,
                satellite_id="sat_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=0
            )
        ]

        mock_scheduler.mission.targets = targets
        mock_scheduler.cluster_schedules = cluster_schedules

        collector = ClusteringMetricsCollector(mock_scheduler)
        metrics = collector.collect_efficiency_metrics()

        # Traditional: 5 tasks, Clustering: 1 task
        assert metrics.task_reduction_ratio == pytest.approx(0.8, abs=0.01)  # (5-1)/5 = 0.8
        assert metrics.task_reduction_count == 4  # 5 - 1 = 4
        assert metrics.avg_targets_per_task == 5.0
        assert metrics.max_targets_in_single_task == 5

    def test_collect_efficiency_metrics_time_savings(self, metrics_collector):
        """Test time savings calculation"""
        metrics = metrics_collector.collect_efficiency_metrics()

        # Time savings should be positive when there's task reduction
        assert metrics.time_savings_seconds > 0
        # Time savings = setup_savings + imaging_savings
        # setup_savings = task_reduction_count * 120
        # imaging_savings = task_reduction_count * 60 * 0.5 (50% overlap efficiency)
        # Total = task_reduction_count * (120 + 30) = task_reduction_count * 150
        expected_savings = metrics.task_reduction_count * 150
        assert metrics.time_savings_seconds == pytest.approx(expected_savings, abs=10)


# =============================================================================
# Test ClusteringMetricsCollector - Coverage Metrics
# =============================================================================

class TestClusteringMetricsCollectorCoverage:
    """Test coverage metrics collection"""

    def test_collect_coverage_metrics_basic(self, metrics_collector, sample_targets, cluster_schedules):
        """Test basic coverage metrics calculation

        Scenario: 7 total targets, 3 high priority (priority >= 8: t2, t4, t7)
        Covered: 5 targets (t1, t2, t3, t4, t5)
        High priority covered: 2 (t2, t4)
        """
        metrics = metrics_collector.collect_coverage_metrics()

        assert isinstance(metrics, ClusteringCoverageMetrics)
        assert metrics.targets_total == 7
        assert metrics.targets_covered == 5
        assert metrics.target_coverage_ratio == pytest.approx(5/7, abs=0.01)

        assert metrics.high_priority_total == 3  # t2, t4, t7
        assert metrics.high_priority_covered == 2  # t2, t4
        assert metrics.high_priority_coverage == pytest.approx(2/3, abs=0.01)

    def test_collect_coverage_metrics_empty_schedule(self, mock_scheduler):
        """Test coverage metrics with empty schedule"""
        mock_scheduler.mission.targets = []
        mock_scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(mock_scheduler)
        metrics = collector.collect_coverage_metrics()

        assert metrics.targets_total == 0
        assert metrics.targets_covered == 0
        assert metrics.target_coverage_ratio == 0.0
        assert metrics.high_priority_total == 0
        assert metrics.high_priority_covered == 0
        assert metrics.high_priority_coverage == 0.0
        assert metrics.area_coverage_km2 == 0.0

    def test_collect_coverage_metrics_no_high_priority(self, mock_scheduler):
        """Test coverage metrics with no high priority targets"""
        targets = [
            Target(id=f"t{i}", name=f"Target {i}", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=5)
            for i in range(5)
        ]

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=targets[:3],
                satellite_id="sat_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=0
            )
        ]

        mock_scheduler.mission.targets = targets
        mock_scheduler.cluster_schedules = cluster_schedules

        collector = ClusteringMetricsCollector(mock_scheduler)
        metrics = collector.collect_coverage_metrics()

        assert metrics.high_priority_total == 0
        assert metrics.high_priority_covered == 0
        assert metrics.high_priority_coverage == 1.0  # No high priority = full coverage

    def test_collect_coverage_metrics_area_calculation(self, mock_scheduler):
        """Test area coverage calculation"""
        # Create area targets
        area_target = Target(
            id="area_1",
            name="Area Target 1",
            target_type=TargetType.AREA,
            area_vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
            priority=5
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=[area_target],
                satellite_id="sat_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=0
            )
        ]

        mock_scheduler.mission.targets = [area_target]
        mock_scheduler.cluster_schedules = cluster_schedules

        collector = ClusteringMetricsCollector(mock_scheduler)
        metrics = collector.collect_coverage_metrics()

        # Area should be approximately 1 degree x 1 degree
        # ~111km x 111km = ~12321 km2 (rough approximation)
        assert metrics.area_coverage_km2 > 0


# =============================================================================
# Test ClusteringMetricsCollector - Quality Score
# =============================================================================

class TestClusteringMetricsCollectorQualityScore:
    """Test quality score calculation"""

    def test_calculate_quality_score_basic(self, metrics_collector):
        """Test basic quality score calculation"""
        score = metrics_collector.calculate_quality_score()

        assert isinstance(score, ClusteringQualityScore)
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.efficiency_score <= 100
        assert 0 <= score.coverage_score <= 100
        assert 0 <= score.priority_score <= 100
        assert 0 <= score.balance_score <= 100

    def test_calculate_quality_score_formula(self, metrics_collector):
        """Test that overall score is weighted average of components"""
        score = metrics_collector.calculate_quality_score()

        # Overall score should be weighted average:
        # efficiency(30%) + coverage(25%) + priority(30%) + balance(15%)
        expected_overall = (
            score.efficiency_score * 0.30 +
            score.coverage_score * 0.25 +
            score.priority_score * 0.30 +
            score.balance_score * 0.15
        )

        assert score.overall_score == pytest.approx(expected_overall, abs=0.1)

    def test_calculate_quality_score_empty_schedule(self, mock_scheduler):
        """Test quality score with empty schedule"""
        mock_scheduler.mission.targets = []
        mock_scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(mock_scheduler)
        score = collector.calculate_quality_score()

        assert score.overall_score == 0.0
        assert score.efficiency_score == 0.0
        assert score.coverage_score == 0.0
        assert score.priority_score == 0.0
        assert score.balance_score == 0.0

    def test_efficiency_score_calculation(self, metrics_collector):
        """Test efficiency score calculation"""
        score = metrics_collector.calculate_quality_score()

        # Efficiency score = task_reduction_ratio * 100 + bonus
        # With ~43% reduction, base score = 43
        # Bonus for avg_targets_per_task > 2: +10
        assert score.efficiency_score > 40
        assert score.efficiency_score <= 100

    def test_priority_score_penalty(self, mock_scheduler):
        """Test priority score penalty for low high-priority coverage"""
        # Create scenario with low high-priority coverage
        targets = [
            Target(id="t1", name="Target 1", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=9),
            Target(id="t2", name="Target 2", target_type=TargetType.POINT,
                   longitude=116.1, latitude=39.0, priority=9),
            Target(id="t3", name="Target 3", target_type=TargetType.POINT,
                   longitude=117.0, latitude=40.0, priority=9),
        ]

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=[targets[0]],  # Only 1 of 3 high priority covered
                satellite_id="sat_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=1
            )
        ]

        mock_scheduler.mission.targets = targets
        mock_scheduler.cluster_schedules = cluster_schedules

        collector = ClusteringMetricsCollector(mock_scheduler)
        score = collector.calculate_quality_score()

        # High priority coverage = 1/3 = 0.333
        # Score = 33.3 - 10 (penalty for < 0.8) = 23.3
        assert score.priority_score < 35


# =============================================================================
# Test ClusteringMetricsCollector - Comparison
# =============================================================================

class TestClusteringMetricsCollectorComparison:
    """Test comparison with traditional scheduling"""

    def test_compare_with_traditional_basic(self, metrics_collector):
        """Test basic comparison calculation"""
        comparison = metrics_collector.compare_with_traditional()

        assert isinstance(comparison, dict)
        assert 'traditional_task_count' in comparison
        assert 'clustering_task_count' in comparison
        assert 'improvement_ratio' in comparison
        assert 'time_saved_minutes' in comparison
        assert 'fuel_saved_estimate_kg' in comparison

    def test_compare_with_traditional_calculations(self, metrics_collector, sample_targets):
        """Test comparison calculations

        7 total targets:
        - Traditional: 7 tasks
        - Clustering: 2 cluster schedules + 2 individual = 4 tasks
        - Improvement: (7-4)/7 = 0.428...
        """
        comparison = metrics_collector.compare_with_traditional()

        assert comparison['traditional_task_count'] == 7
        assert comparison['clustering_task_count'] == 4  # 2 clusters + 2 individual
        assert comparison['improvement_ratio'] == pytest.approx(0.428, abs=0.01)
        assert comparison['time_saved_minutes'] > 0
        assert comparison['fuel_saved_estimate_kg'] > 0

    def test_compare_with_traditional_empty_schedule(self, mock_scheduler):
        """Test comparison with empty schedule"""
        mock_scheduler.mission.targets = []
        mock_scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(mock_scheduler)
        comparison = collector.compare_with_traditional()

        assert comparison['traditional_task_count'] == 0
        assert comparison['clustering_task_count'] == 0
        assert comparison['improvement_ratio'] == 0.0
        assert comparison['time_saved_minutes'] == 0.0
        assert comparison['fuel_saved_estimate_kg'] == 0.0


# =============================================================================
# Test ClusteringMetricsCollector - Report Generation
# =============================================================================

class TestClusteringMetricsCollectorReport:
    """Test report generation"""

    def test_generate_report_structure(self, metrics_collector):
        """Test report structure"""
        report = metrics_collector.generate_report()

        assert isinstance(report, dict)
        assert 'efficiency' in report
        assert 'coverage' in report
        assert 'quality_score' in report
        assert 'comparison' in report
        assert 'summary' in report
        assert 'timestamp' in report

    def test_generate_report_summary(self, metrics_collector):
        """Test report summary"""
        report = metrics_collector.generate_report()

        summary = report['summary']
        assert 'total_targets' in summary
        assert 'total_clustered_targets' in summary
        assert 'total_individual_targets' in summary
        assert 'total_tasks' in summary
        assert 'cluster_task_count' in summary
        assert 'individual_task_count' in summary

    def test_generate_report_timestamp(self, metrics_collector):
        """Test report timestamp"""
        report = metrics_collector.generate_report()

        assert isinstance(report['timestamp'], datetime)


# =============================================================================
# Test ClusteringVisualizer
# =============================================================================

class TestClusteringVisualizer:
    """Test visualization data preparation"""

    @pytest.fixture
    def visualizer(self, mock_scheduler, sample_targets, cluster_schedules):
        """Create a visualizer with sample data"""
        mock_scheduler.mission.targets = sample_targets
        mock_scheduler.cluster_schedules = cluster_schedules
        return ClusteringVisualizer(mock_scheduler)

    def test_prepare_cluster_map_data_structure(self, visualizer):
        """Test cluster map data structure"""
        data = visualizer.prepare_cluster_map_data()

        assert isinstance(data, dict)
        assert 'clusters' in data
        assert 'unclustered_targets' in data
        assert isinstance(data['clusters'], list)
        assert isinstance(data['unclustered_targets'], list)

    def test_prepare_cluster_map_data_clusters(self, visualizer, cluster_schedules):
        """Test cluster map data for clusters"""
        data = visualizer.prepare_cluster_map_data()

        assert len(data['clusters']) == len(cluster_schedules)

        for i, cluster_data in enumerate(data['clusters']):
            assert 'cluster_id' in cluster_data
            assert 'center' in cluster_data
            assert 'targets' in cluster_data
            assert 'color' in cluster_data
            assert 'scheduled' in cluster_data

            assert isinstance(cluster_data['center'], tuple)
            assert len(cluster_data['center']) == 2  # (lon, lat)
            assert isinstance(cluster_data['targets'], list)

    def test_prepare_cluster_map_data_unclustered(self, visualizer, sample_targets):
        """Test cluster map data for unclustered targets"""
        data = visualizer.prepare_cluster_map_data()

        # 7 total targets, 5 clustered = 2 unclustered (t6, t7)
        assert len(data['unclustered_targets']) == 2

        for target_data in data['unclustered_targets']:
            assert 'target_id' in target_data
            assert 'longitude' in target_data
            assert 'latitude' in target_data
            assert 'priority' in target_data

    def test_prepare_coverage_heatmap_data(self, visualizer):
        """Test coverage heatmap data"""
        data = visualizer.prepare_coverage_heatmap_data()

        assert isinstance(data, dict)
        assert 'coverage_points' in data
        assert isinstance(data['coverage_points'], list)

        for point in data['coverage_points']:
            assert 'longitude' in point
            assert 'latitude' in point
            assert 'intensity' in point
            assert 0 <= point['intensity'] <= 1

    def test_prepare_efficiency_chart_data(self, visualizer):
        """Test efficiency chart data"""
        data = visualizer.prepare_efficiency_chart_data()

        assert isinstance(data, dict)
        assert 'labels' in data
        assert 'traditional_values' in data
        assert 'clustering_values' in data

        assert isinstance(data['labels'], list)
        assert isinstance(data['traditional_values'], list)
        assert isinstance(data['clustering_values'], list)

        # Should have metrics for: tasks, time, fuel
        assert len(data['labels']) == 3
        assert len(data['traditional_values']) == 3
        assert len(data['clustering_values']) == 3

    def test_visualizer_empty_schedule(self, mock_scheduler):
        """Test visualizer with empty schedule"""
        mock_scheduler.mission.targets = []
        mock_scheduler.cluster_schedules = []

        visualizer = ClusteringVisualizer(mock_scheduler)

        map_data = visualizer.prepare_cluster_map_data()
        assert map_data['clusters'] == []
        assert map_data['unclustered_targets'] == []

        heatmap_data = visualizer.prepare_coverage_heatmap_data()
        assert heatmap_data['coverage_points'] == []


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_no_targets(self, mock_scheduler):
        """Test with no targets"""
        mock_scheduler.mission.targets = []
        mock_scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(mock_scheduler)

        efficiency = collector.collect_efficiency_metrics()
        assert efficiency.task_reduction_ratio == 0.0

        coverage = collector.collect_coverage_metrics()
        assert coverage.target_coverage_ratio == 0.0

        score = collector.calculate_quality_score()
        assert score.overall_score == 0.0

    def test_no_clusters(self, mock_scheduler):
        """Test with targets but no clusters"""
        targets = [
            Target(id="t1", name="Target 1", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=5),
        ]

        mock_scheduler.mission.targets = targets
        mock_scheduler.cluster_schedules = []

        collector = ClusteringMetricsCollector(mock_scheduler)

        efficiency = collector.collect_efficiency_metrics()
        assert efficiency.task_reduction_ratio == 0.0
        assert efficiency.avg_targets_per_task == 0.0

    def test_all_high_priority_covered(self, mock_scheduler):
        """Test when all high priority targets are covered"""
        targets = [
            Target(id="t1", name="Target 1", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=9),
            Target(id="t2", name="Target 2", target_type=TargetType.POINT,
                   longitude=116.005, latitude=39.0, priority=9),
        ]

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=targets,
                satellite_id="sat_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=2
            )
        ]

        mock_scheduler.mission.targets = targets
        mock_scheduler.cluster_schedules = cluster_schedules

        collector = ClusteringMetricsCollector(mock_scheduler)
        coverage = collector.collect_coverage_metrics()

        assert coverage.high_priority_coverage == 1.0

        score = collector.calculate_quality_score()
        # Should get bonus for high priority coverage > 0.9
        assert score.priority_score > 90

    def test_single_target_clusters(self, mock_scheduler):
        """Test with single-target clusters (edge case)"""
        targets = [
            Target(id="t1", name="Target 1", target_type=TargetType.POINT,
                   longitude=116.0, latitude=39.0, priority=5),
            Target(id="t2", name="Target 2", target_type=TargetType.POINT,
                   longitude=117.0, latitude=40.0, priority=5),
        ]

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cluster_schedules = [
            ClusterSchedule(
                cluster_id="cluster_1",
                targets=[targets[0]],
                satellite_id="sat_1",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                look_angle=5.0,
                priority_satisfied=0
            ),
            ClusterSchedule(
                cluster_id="cluster_2",
                targets=[targets[1]],
                satellite_id="sat_1",
                imaging_start=base_time + timedelta(minutes=10),
                imaging_end=base_time + timedelta(minutes=15),
                look_angle=5.0,
                priority_satisfied=0
            ),
        ]

        mock_scheduler.mission.targets = targets
        mock_scheduler.cluster_schedules = cluster_schedules

        collector = ClusteringMetricsCollector(mock_scheduler)
        efficiency = collector.collect_efficiency_metrics()

        # No task reduction since each cluster has 1 target
        assert efficiency.task_reduction_ratio == 0.0
        assert efficiency.avg_targets_per_task == 1.0
