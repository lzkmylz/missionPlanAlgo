"""
Clustering Metrics - Quality evaluation metrics for cluster-aware scheduling

This module provides comprehensive quality metrics for evaluating the effectiveness
of cluster-based target scheduling compared to traditional individual target scheduling.

Key Features:
1. Efficiency Metrics: Task reduction, time savings, resource utilization
2. Coverage Metrics: Target coverage ratio, priority satisfaction, area coverage
3. Quality Scoring: Overall scheduling quality score with weighted components
4. Comparison Analysis: Compare clustering vs traditional scheduling
5. Visualization Support: Data preparation for visualization tools
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import math

from scheduler.clustering_greedy_scheduler import (
    ClusteringGreedyScheduler,
    ClusterSchedule
)
from core.models.target import Target, TargetType


@dataclass
class ClusteringEfficiencyMetrics:
    """聚类效率指标

    Attributes:
        task_reduction_ratio: 任务减少比例 (0-1)
        task_reduction_count: 减少的任务数
        time_savings_seconds: 节省的时间（秒）
        avg_targets_per_task: 平均每任务目标数
        max_targets_in_single_task: 单任务最大目标数
        cluster_utilization_ratio: 聚类利用率
    """
    task_reduction_ratio: float
    task_reduction_count: int
    time_savings_seconds: float
    avg_targets_per_task: float
    max_targets_in_single_task: int
    cluster_utilization_ratio: float


@dataclass
class ClusteringCoverageMetrics:
    """聚类覆盖指标

    Attributes:
        target_coverage_ratio: 目标覆盖率 (0-1)
        targets_covered: 被覆盖的目标数
        targets_total: 总目标数
        high_priority_coverage: 高优先级覆盖率 (0-1)
        high_priority_covered: 高优先级已覆盖
        high_priority_total: 高优先级总数
        area_coverage_km2: 覆盖面积（平方公里）
    """
    target_coverage_ratio: float
    targets_covered: int
    targets_total: int
    high_priority_coverage: float
    high_priority_covered: int
    high_priority_total: int
    area_coverage_km2: float


@dataclass
class ClusteringQualityScore:
    """聚类质量评分

    Attributes:
        overall_score: 总体评分 (0-100)
        efficiency_score: 效率评分
        coverage_score: 覆盖评分
        priority_score: 优先级评分
        balance_score: 均衡性评分
    """
    overall_score: float
    efficiency_score: float
    coverage_score: float
    priority_score: float
    balance_score: float


class ClusteringMetricsCollector:
    """
    聚类调度指标收集器

    收集和计算聚类调度的各项质量指标。

    Example:
        scheduler = ClusteringGreedyScheduler(config)
        result = scheduler.schedule()

        collector = ClusteringMetricsCollector(scheduler)
        efficiency = collector.collect_efficiency_metrics()
        coverage = collector.collect_coverage_metrics()
        score = collector.calculate_quality_score()
        report = collector.generate_report()
    """

    # Scoring weights for overall quality score
    EFFICIENCY_WEIGHT = 0.30
    COVERAGE_WEIGHT = 0.25
    PRIORITY_WEIGHT = 0.30
    BALANCE_WEIGHT = 0.15

    # Constants for time and fuel estimation
    SETUP_TIME_SECONDS = 120  # Time for task setup/slewing
    IMAGING_TIME_PER_TARGET_SECONDS = 60  # Base imaging time per target
    FUEL_PER_SETUP_KG = 0.05  # Estimated fuel per setup
    FUEL_PER_IMAGING_KG = 0.02  # Estimated fuel per imaging second

    def __init__(self, scheduler: ClusteringGreedyScheduler):
        """
        Initialize the metrics collector

        Args:
            scheduler: ClusteringGreedyScheduler instance with completed schedule
        """
        self.scheduler = scheduler
        self.cluster_schedules = scheduler.cluster_schedules
        self.all_targets = scheduler.mission.targets if scheduler.mission else []

    def collect_efficiency_metrics(self) -> ClusteringEfficiencyMetrics:
        """
        收集效率指标

        Calculates efficiency gains from clustering including task reduction,
        time savings, and cluster utilization.

        Returns:
            ClusteringEfficiencyMetrics with calculated values
        """
        if not self.all_targets or not self.cluster_schedules:
            return self._empty_efficiency_metrics()

        total_targets = len(self.all_targets)
        cluster_task_count = len(self.cluster_schedules)

        clustered_target_ids = self._get_clustered_target_ids()
        clustered_count = len(clustered_target_ids)
        individual_count = total_targets - clustered_count

        total_clustering_tasks = cluster_task_count + individual_count
        traditional_task_count = total_targets

        task_reduction_count = traditional_task_count - total_clustering_tasks
        task_reduction_ratio = self._calculate_ratio(task_reduction_count, traditional_task_count)
        avg_targets_per_task = self._calculate_ratio(clustered_count, cluster_task_count)

        return ClusteringEfficiencyMetrics(
            task_reduction_ratio=max(0.0, task_reduction_ratio),
            task_reduction_count=max(0, task_reduction_count),
            time_savings_seconds=self._calculate_time_savings(task_reduction_count, avg_targets_per_task),
            avg_targets_per_task=avg_targets_per_task,
            max_targets_in_single_task=self._get_max_targets_in_task(),
            cluster_utilization_ratio=self._calculate_ratio(clustered_count, total_targets)
        )

    def _empty_efficiency_metrics(self) -> ClusteringEfficiencyMetrics:
        """Return empty efficiency metrics."""
        return ClusteringEfficiencyMetrics(
            task_reduction_ratio=0.0,
            task_reduction_count=0,
            time_savings_seconds=0.0,
            avg_targets_per_task=0.0,
            max_targets_in_single_task=0,
            cluster_utilization_ratio=0.0
        )

    def _get_clustered_target_ids(self) -> Set[str]:
        """Get set of all target IDs covered by clusters."""
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)
        return clustered_target_ids

    def _get_max_targets_in_task(self) -> int:
        """Get maximum number of targets in any single cluster task."""
        return max(
            (len(cs.targets) for cs in self.cluster_schedules),
            default=0
        )

    def _calculate_ratio(self, numerator: int, denominator: int) -> float:
        """Calculate ratio with zero check."""
        return numerator / denominator if denominator > 0 else 0.0

    def _calculate_time_savings(
        self,
        task_reduction_count: int,
        avg_targets_per_task: float
    ) -> float:
        """
        Calculate time savings from clustering

        Each saved task saves:
        - Setup/slewing time
        - Reduced imaging time due to overlap efficiency

        Args:
            task_reduction_count: Number of tasks saved
            avg_targets_per_task: Average targets per cluster task

        Returns:
            Time savings in seconds
        """
        if task_reduction_count <= 0:
            return 0.0

        # Each saved task saves setup time
        setup_savings = task_reduction_count * self.SETUP_TIME_SECONDS

        # Imaging time savings due to cluster overlap efficiency
        # Clustered imaging is more efficient (50% overlap assumed)
        imaging_savings = task_reduction_count * self.IMAGING_TIME_PER_TARGET_SECONDS * 0.5

        return setup_savings + imaging_savings

    def collect_coverage_metrics(self) -> ClusteringCoverageMetrics:
        """
        收集覆盖指标

        Calculates coverage statistics including target coverage ratio,
        high priority coverage, and area coverage.

        Returns:
            ClusteringCoverageMetrics with calculated values
        """
        if not self.all_targets:
            return self._empty_coverage_metrics()

        covered_target_ids = self._get_covered_target_ids()
        targets_covered = len(covered_target_ids)
        total_targets = len(self.all_targets)

        target_coverage_ratio = self._calculate_ratio(targets_covered, total_targets)
        high_priority_coverage, high_priority_covered, high_priority_total = \
            self._calculate_high_priority_coverage(covered_target_ids)
        area_coverage_km2 = self._calculate_area_coverage(covered_target_ids)

        return ClusteringCoverageMetrics(
            target_coverage_ratio=target_coverage_ratio,
            targets_covered=targets_covered,
            targets_total=total_targets,
            high_priority_coverage=high_priority_coverage,
            high_priority_covered=high_priority_covered,
            high_priority_total=high_priority_total,
            area_coverage_km2=area_coverage_km2
        )

    def _empty_coverage_metrics(self) -> ClusteringCoverageMetrics:
        """Return empty coverage metrics when no targets."""
        return ClusteringCoverageMetrics(
            target_coverage_ratio=0.0,
            targets_covered=0,
            targets_total=0,
            high_priority_coverage=0.0,
            high_priority_covered=0,
            high_priority_total=0,
            area_coverage_km2=0.0
        )

    def _get_covered_target_ids(self) -> Set[str]:
        """Get set of all covered target IDs from cluster schedules."""
        covered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                covered_target_ids.add(target.id)
        return covered_target_ids

    def _calculate_high_priority_coverage(
        self, covered_target_ids: Set[str]
    ) -> Tuple[float, int, int]:
        """
        Calculate high priority target coverage.

        Returns:
            Tuple of (coverage_ratio, covered_count, total_count)
        """
        high_priority_targets = [t for t in self.all_targets if t.priority >= 8]
        high_priority_total = len(high_priority_targets)

        high_priority_covered = sum(
            1 for t in high_priority_targets if t.id in covered_target_ids
        )

        if high_priority_total > 0:
            coverage = high_priority_covered / high_priority_total
        else:
            coverage = 1.0  # No high priority targets = full coverage

        return coverage, high_priority_covered, high_priority_total

    def _calculate_area_coverage(self, covered_target_ids: Set[str]) -> float:
        """
        Calculate total area covered

        Args:
            covered_target_ids: Set of covered target IDs

        Returns:
            Total area in square kilometers
        """
        total_area = 0.0

        for target in self.all_targets:
            if target.id in covered_target_ids:
                if target.target_type == TargetType.AREA:
                    total_area += target.get_area()
                else:
                    # Point targets: assume small coverage area
                    # ~1 km2 per point target (simplified)
                    total_area += 1.0

        return total_area

    def calculate_quality_score(self) -> ClusteringQualityScore:
        """
        计算综合质量评分

        Calculates overall quality score based on efficiency, coverage,
        priority satisfaction, and cluster balance.

        Scoring Formulas:
        - Efficiency Score: task_reduction_ratio * 100 + bonus
        - Coverage Score: target_coverage_ratio * 100 + bonus
        - Priority Score: high_priority_coverage * 100 + penalty/bonus
        - Balance Score: Based on target distribution across clusters
        - Overall Score: Weighted average of component scores

        Returns:
            ClusteringQualityScore with all component scores
        """
        if not self.all_targets:
            return self._empty_quality_score()

        efficiency_metrics = self.collect_efficiency_metrics()
        coverage_metrics = self.collect_coverage_metrics()

        efficiency_score = self._calculate_efficiency_score(efficiency_metrics)
        coverage_score = self._calculate_coverage_score(coverage_metrics)
        priority_score = self._calculate_priority_score(coverage_metrics)
        balance_score = self._calculate_balance_score()

        overall_score = (
            efficiency_score * self.EFFICIENCY_WEIGHT +
            coverage_score * self.COVERAGE_WEIGHT +
            priority_score * self.PRIORITY_WEIGHT +
            balance_score * self.BALANCE_WEIGHT
        )

        return ClusteringQualityScore(
            overall_score=round(overall_score, 1),
            efficiency_score=round(efficiency_score, 1),
            coverage_score=round(coverage_score, 1),
            priority_score=round(priority_score, 1),
            balance_score=round(balance_score, 1)
        )

    def _empty_quality_score(self) -> ClusteringQualityScore:
        """Return empty quality score."""
        return ClusteringQualityScore(
            overall_score=0.0,
            efficiency_score=0.0,
            coverage_score=0.0,
            priority_score=0.0,
            balance_score=0.0
        )

    def _calculate_efficiency_score(self, metrics: ClusteringEfficiencyMetrics) -> float:
        """Calculate efficiency score with bonus."""
        score = metrics.task_reduction_ratio * 100
        if metrics.avg_targets_per_task > 2:
            score += 10
        return min(100, score)

    def _calculate_coverage_score(self, metrics: ClusteringCoverageMetrics) -> float:
        """Calculate coverage score with bonus."""
        score = metrics.target_coverage_ratio * 100
        if metrics.high_priority_coverage > 0.9:
            score += 5
        return min(100, score)

    def _calculate_priority_score(self, metrics: ClusteringCoverageMetrics) -> float:
        """Calculate priority score with penalties and bonuses."""
        score = metrics.high_priority_coverage * 100
        if metrics.high_priority_coverage < 0.8:
            score -= 10
        if metrics.high_priority_coverage == 1.0:
            score += 5
        return max(0, min(100, score))

    def _calculate_balance_score(self) -> float:
        """
        Calculate balance score based on cluster size distribution

        Penalizes very large or very small clusters.
        Optimal cluster size is between 2 and 10 targets.

        Returns:
            Balance score (0-100)
        """
        if not self.cluster_schedules:
            return 50.0  # Neutral score for no clusters

        cluster_sizes = [len(cs.targets) for cs in self.cluster_schedules]

        if not cluster_sizes:
            return 50.0

        # Ideal cluster size is 3-5 targets
        ideal_min = 2
        ideal_max = 8

        score = 100.0

        for size in cluster_sizes:
            if size < ideal_min:
                # Penalty for very small clusters
                score -= (ideal_min - size) * 5
            elif size > ideal_max:
                # Penalty for very large clusters
                score -= (size - ideal_max) * 3

        # Penalty for high variance in cluster sizes
        if len(cluster_sizes) > 1:
            mean_size = sum(cluster_sizes) / len(cluster_sizes)
            variance = sum((s - mean_size) ** 2 for s in cluster_sizes) / len(cluster_sizes)
            std_dev = variance ** 0.5

            # Penalty for high standard deviation
            if std_dev > 2:
                score -= (std_dev - 2) * 5

        return max(0, min(100, score))

    def compare_with_traditional(self) -> Dict[str, Any]:
        """
        与传统调度方式对比

        Compares clustering-based scheduling with traditional individual
        target scheduling approach.

        Returns:
            Dictionary with comparison metrics:
                - traditional_task_count: Number of tasks in traditional approach
                - clustering_task_count: Number of tasks with clustering
                - improvement_ratio: Relative improvement (0-1)
                - time_saved_minutes: Time saved in minutes
                - fuel_saved_estimate_kg: Estimated fuel saved in kg
        """
        if not self.all_targets:
            return self._empty_comparison()

        total_targets = len(self.all_targets)
        traditional_task_count = total_targets

        clustered_count = self._get_clustered_target_count()
        individual_count = total_targets - clustered_count
        cluster_task_count = len(self.cluster_schedules)

        clustering_task_count = cluster_task_count + individual_count
        tasks_saved = traditional_task_count - clustering_task_count

        return {
            'traditional_task_count': traditional_task_count,
            'clustering_task_count': clustering_task_count,
            'improvement_ratio': self._calculate_improvement_ratio(traditional_task_count, clustering_task_count),
            'time_saved_minutes': self._calculate_time_saved(tasks_saved),
            'fuel_saved_estimate_kg': self._calculate_fuel_saved(tasks_saved),
        }

    def _empty_comparison(self) -> Dict[str, Any]:
        """Return empty comparison result."""
        return {
            'traditional_task_count': 0,
            'clustering_task_count': 0,
            'improvement_ratio': 0.0,
            'time_saved_minutes': 0.0,
            'fuel_saved_estimate_kg': 0.0,
        }

    def _get_clustered_target_count(self) -> int:
        """Get count of targets covered by clusters."""
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)
        return len(clustered_target_ids)

    def _calculate_improvement_ratio(self, traditional: int, clustering: int) -> float:
        """Calculate improvement ratio."""
        if traditional > 0:
            ratio = (traditional - clustering) / traditional
            return round(max(0.0, ratio), 3)
        return 0.0

    def _calculate_time_saved(self, tasks_saved: int) -> float:
        """Calculate time saved in minutes."""
        time_saved_seconds = tasks_saved * (self.SETUP_TIME_SECONDS + self.IMAGING_TIME_PER_TARGET_SECONDS)
        return round(max(0.0, time_saved_seconds / 60.0), 2)

    def _calculate_fuel_saved(self, tasks_saved: int) -> float:
        """Calculate estimated fuel saved in kg."""
        fuel_per_task_saved = (
            self.FUEL_PER_SETUP_KG +
            self.FUEL_PER_IMAGING_KG * self.IMAGING_TIME_PER_TARGET_SECONDS
        )
        return round(max(0.0, tasks_saved * fuel_per_task_saved), 3)

    def generate_report(self) -> Dict[str, Any]:
        """
        生成完整报告

        Generates a comprehensive report with all metrics.

        Returns:
            Dictionary containing:
                - efficiency: Efficiency metrics
                - coverage: Coverage metrics
                - quality_score: Quality scores
                - comparison: Comparison with traditional scheduling
                - summary: Summary statistics
                - timestamp: Report generation time
        """
        efficiency = self.collect_efficiency_metrics()
        coverage = self.collect_coverage_metrics()
        quality_score = self.calculate_quality_score()
        comparison = self.compare_with_traditional()

        return {
            'efficiency': self._format_efficiency_metrics(efficiency),
            'coverage': self._format_coverage_metrics(coverage),
            'quality_score': self._format_quality_score(quality_score),
            'comparison': comparison,
            'summary': self._generate_summary(comparison),
            'timestamp': datetime.now(),
        }

    def _format_efficiency_metrics(self, metrics: ClusteringEfficiencyMetrics) -> Dict[str, Any]:
        """Format efficiency metrics for report."""
        return {
            'task_reduction_ratio': metrics.task_reduction_ratio,
            'task_reduction_count': metrics.task_reduction_count,
            'time_savings_seconds': metrics.time_savings_seconds,
            'avg_targets_per_task': metrics.avg_targets_per_task,
            'max_targets_in_single_task': metrics.max_targets_in_single_task,
            'cluster_utilization_ratio': metrics.cluster_utilization_ratio,
        }

    def _format_coverage_metrics(self, metrics: ClusteringCoverageMetrics) -> Dict[str, Any]:
        """Format coverage metrics for report."""
        return {
            'target_coverage_ratio': metrics.target_coverage_ratio,
            'targets_covered': metrics.targets_covered,
            'targets_total': metrics.targets_total,
            'high_priority_coverage': metrics.high_priority_coverage,
            'high_priority_covered': metrics.high_priority_covered,
            'high_priority_total': metrics.high_priority_total,
            'area_coverage_km2': metrics.area_coverage_km2,
        }

    def _format_quality_score(self, score: ClusteringQualityScore) -> Dict[str, float]:
        """Format quality score for report."""
        return {
            'overall_score': score.overall_score,
            'efficiency_score': score.efficiency_score,
            'coverage_score': score.coverage_score,
            'priority_score': score.priority_score,
            'balance_score': score.balance_score,
        }

    def _generate_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_clustered = self._get_clustered_target_count()
        total_individual = len(self.all_targets) - total_clustered

        return {
            'total_targets': len(self.all_targets),
            'total_clustered_targets': total_clustered,
            'total_individual_targets': total_individual,
            'total_tasks': comparison['clustering_task_count'],
            'cluster_task_count': len(self.cluster_schedules),
            'individual_task_count': total_individual,
        }


class ClusteringVisualizer:
    """
    聚类可视化数据准备器

    Prepares data for visualization tools including cluster maps,
    coverage heatmaps, and efficiency charts.

    Example:
        visualizer = ClusteringVisualizer(scheduler)

        # Get map data
        map_data = visualizer.prepare_cluster_map_data()

        # Get heatmap data
        heatmap_data = visualizer.prepare_coverage_heatmap_data()

        # Get efficiency chart data
        chart_data = visualizer.prepare_efficiency_chart_data()
    """

    # Color palette for clusters
    CLUSTER_COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    ]

    def __init__(self, scheduler: ClusteringGreedyScheduler):
        """
        Initialize the visualizer

        Args:
            scheduler: ClusteringGreedyScheduler instance with completed schedule
        """
        self.scheduler = scheduler
        self.cluster_schedules = scheduler.cluster_schedules
        self.all_targets = scheduler.mission.targets if scheduler.mission else []

    def prepare_cluster_map_data(self) -> Dict[str, Any]:
        """
        准备聚类地图数据

        Prepares data for visualizing clusters on a map.

        Returns:
            Dictionary with:
                - clusters: List of cluster data with center, targets, color
                - unclustered_targets: List of targets not in any cluster
        """
        clusters_data = [
            self._format_cluster_for_map(i, cs)
            for i, cs in enumerate(self.cluster_schedules)
            if cs.targets
        ]

        return {
            'clusters': clusters_data,
            'unclustered_targets': self._get_unclustered_targets_for_map(),
        }

    def _format_cluster_for_map(self, index: int, cs) -> Dict[str, Any]:
        """Format a cluster schedule for map visualization."""
        center_lon = sum(t.longitude or 0 for t in cs.targets) / len(cs.targets)
        center_lat = sum(t.latitude or 0 for t in cs.targets) / len(cs.targets)

        return {
            'cluster_id': cs.cluster_id,
            'center': (center_lon, center_lat),
            'targets': [self._format_target_for_map(t) for t in cs.targets],
            'color': self.CLUSTER_COLORS[index % len(self.CLUSTER_COLORS)],
            'scheduled': True,
            'satellite_id': cs.satellite_id,
            'imaging_time': cs.imaging_start.isoformat() if cs.imaging_start else None,
        }

    def _format_target_for_map(self, target) -> Dict[str, Any]:
        """Format a target for map visualization."""
        return {
            'target_id': target.id,
            'name': target.name,
            'longitude': target.longitude,
            'latitude': target.latitude,
            'priority': target.priority,
        }

    def _get_clustered_target_ids(self) -> Set[str]:
        """Get IDs of all targets in clusters."""
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)
        return clustered_target_ids

    def _get_unclustered_targets_for_map(self) -> List[Dict[str, Any]]:
        """Get list of unclustered targets formatted for map."""
        clustered_ids = self._get_clustered_target_ids()
        return [
            self._format_target_for_map(t)
            for t in self.all_targets
            if t.id not in clustered_ids
        ]

    def prepare_coverage_heatmap_data(self) -> Dict[str, Any]:
        """
        准备覆盖热力图数据

        Prepares data for a coverage heatmap showing target density
        and coverage intensity.

        Returns:
            Dictionary with coverage points and intensity values
        """
        coverage_points = []

        for cs in self.cluster_schedules:
            if not cs.targets:
                continue
            coverage_points.extend(self._get_cluster_heatmap_points(cs))

        clustered_ids = self._get_clustered_target_ids()
        coverage_points.extend(self._get_unclustered_heatmap_points(clustered_ids))

        return {'coverage_points': coverage_points}

    def _get_cluster_heatmap_points(self, cs) -> List[Dict[str, Any]]:
        """Generate heatmap points for a cluster."""
        points = []
        center_lon = sum(t.longitude or 0 for t in cs.targets) / len(cs.targets)
        center_lat = sum(t.latitude or 0 for t in cs.targets) / len(cs.targets)

        avg_priority = sum(t.priority for t in cs.targets) / len(cs.targets)
        intensity = min(1.0, (len(cs.targets) / 5.0) * (avg_priority / 10.0))

        points.append({
            'longitude': center_lon,
            'latitude': center_lat,
            'intensity': round(intensity, 2),
            'type': 'cluster',
            'cluster_id': cs.cluster_id,
        })

        for target in cs.targets:
            points.append({
                'longitude': target.longitude or 0,
                'latitude': target.latitude or 0,
                'intensity': round(0.3 + (target.priority / 20.0), 2),
                'type': 'target',
                'target_id': target.id,
            })

        return points

    def _get_unclustered_heatmap_points(self, clustered_ids: Set[str]) -> List[Dict[str, Any]]:
        """Generate heatmap points for unclustered targets."""
        return [
            {
                'longitude': target.longitude or 0,
                'latitude': target.latitude or 0,
                'intensity': round(0.2 + (target.priority / 20.0), 2),
                'type': 'unclustered_target',
                'target_id': target.id,
            }
            for target in self.all_targets
            if target.id not in clustered_ids
        ]

    def prepare_efficiency_chart_data(self) -> Dict[str, Any]:
        """
        准备效率对比图表数据

        Prepares data for efficiency comparison charts showing
        clustering vs traditional scheduling.

        Returns:
            Dictionary with chart data for tasks, time, and fuel comparison
        """
        # Calculate metrics
        total_targets = len(self.all_targets)

        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)

        clustered_count = len(clustered_target_ids)
        individual_count = total_targets - clustered_count

        traditional_task_count = total_targets
        clustering_task_count = len(self.cluster_schedules) + individual_count

        # Time calculation (minutes)
        setup_time_min = 2  # 120 seconds
        imaging_time_min = 1  # 60 seconds

        traditional_time = traditional_task_count * (setup_time_min + imaging_time_min)
        clustering_time = clustering_task_count * (setup_time_min + imaging_time_min)

        # Fuel calculation (kg)
        fuel_per_task_kg = 0.1  # Simplified estimate
        traditional_fuel = traditional_task_count * fuel_per_task_kg
        clustering_fuel = clustering_task_count * fuel_per_task_kg

        return {
            'labels': ['Tasks', 'Time (min)', 'Fuel (kg)'],
            'traditional_values': [
                traditional_task_count,
                traditional_time,
                round(traditional_fuel, 2),
            ],
            'clustering_values': [
                clustering_task_count,
                clustering_time,
                round(clustering_fuel, 2),
            ],
        }
