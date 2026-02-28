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
            return ClusteringEfficiencyMetrics(
                task_reduction_ratio=0.0,
                task_reduction_count=0,
                time_savings_seconds=0.0,
                avg_targets_per_task=0.0,
                max_targets_in_single_task=0,
                cluster_utilization_ratio=0.0
            )

        total_targets = len(self.all_targets)
        cluster_task_count = len(self.cluster_schedules)

        # Count targets covered by clusters
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)

        clustered_count = len(clustered_target_ids)

        # Individual tasks = targets not in any cluster
        individual_count = total_targets - clustered_count

        # Total tasks with clustering
        total_clustering_tasks = cluster_task_count + individual_count

        # Traditional approach: one task per target
        traditional_task_count = total_targets

        # Task reduction calculations
        task_reduction_count = traditional_task_count - total_clustering_tasks
        task_reduction_ratio = task_reduction_count / traditional_task_count if traditional_task_count > 0 else 0.0

        # Average targets per cluster task
        avg_targets_per_task = clustered_count / cluster_task_count if cluster_task_count > 0 else 0.0

        # Max targets in a single task
        max_targets_in_single_task = max(
            (len(cs.targets) for cs in self.cluster_schedules),
            default=0
        )

        # Cluster utilization ratio
        cluster_utilization_ratio = clustered_count / total_targets if total_targets > 0 else 0.0

        # Time savings calculation
        # Each saved task saves setup time + imaging time difference
        time_savings_seconds = self._calculate_time_savings(
            task_reduction_count, avg_targets_per_task
        )

        return ClusteringEfficiencyMetrics(
            task_reduction_ratio=max(0.0, task_reduction_ratio),
            task_reduction_count=max(0, task_reduction_count),
            time_savings_seconds=time_savings_seconds,
            avg_targets_per_task=avg_targets_per_task,
            max_targets_in_single_task=max_targets_in_single_task,
            cluster_utilization_ratio=cluster_utilization_ratio
        )

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
            return ClusteringCoverageMetrics(
                target_coverage_ratio=0.0,
                targets_covered=0,
                targets_total=0,
                high_priority_coverage=0.0,
                high_priority_covered=0,
                high_priority_total=0,
                area_coverage_km2=0.0
            )

        total_targets = len(self.all_targets)

        # Get covered target IDs from cluster schedules
        covered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                covered_target_ids.add(target.id)

        targets_covered = len(covered_target_ids)
        target_coverage_ratio = targets_covered / total_targets if total_targets > 0 else 0.0

        # High priority targets (priority >= 8)
        high_priority_targets = [t for t in self.all_targets if t.priority >= 8]
        high_priority_total = len(high_priority_targets)

        high_priority_covered = sum(
            1 for t in high_priority_targets
            if t.id in covered_target_ids
        )

        if high_priority_total > 0:
            high_priority_coverage = high_priority_covered / high_priority_total
        else:
            high_priority_coverage = 1.0  # No high priority targets = full coverage

        # Area coverage calculation
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
            return ClusteringQualityScore(
                overall_score=0.0,
                efficiency_score=0.0,
                coverage_score=0.0,
                priority_score=0.0,
                balance_score=0.0
            )

        efficiency_metrics = self.collect_efficiency_metrics()
        coverage_metrics = self.collect_coverage_metrics()

        # Efficiency Score: base on task reduction ratio
        efficiency_score = efficiency_metrics.task_reduction_ratio * 100
        # Bonus for good average targets per task (> 2)
        if efficiency_metrics.avg_targets_per_task > 2:
            efficiency_score += 10
        efficiency_score = min(100, efficiency_score)

        # Coverage Score: base on target coverage
        coverage_score = coverage_metrics.target_coverage_ratio * 100
        # Bonus for high high-priority coverage (> 0.9)
        if coverage_metrics.high_priority_coverage > 0.9:
            coverage_score += 5
        coverage_score = min(100, coverage_score)

        # Priority Score: based on high priority coverage
        priority_score = coverage_metrics.high_priority_coverage * 100
        # Penalty for low high-priority coverage (< 0.8)
        if coverage_metrics.high_priority_coverage < 0.8:
            priority_score -= 10
        # Bonus for complete high-priority coverage
        if coverage_metrics.high_priority_coverage == 1.0:
            priority_score += 5
        priority_score = max(0, min(100, priority_score))

        # Balance Score: based on cluster size distribution
        balance_score = self._calculate_balance_score()

        # Overall Score: weighted average
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
            return {
                'traditional_task_count': 0,
                'clustering_task_count': 0,
                'improvement_ratio': 0.0,
                'time_saved_minutes': 0.0,
                'fuel_saved_estimate_kg': 0.0,
            }

        total_targets = len(self.all_targets)

        # Traditional: one task per target
        traditional_task_count = total_targets

        # Clustering: cluster tasks + individual tasks
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)

        clustered_count = len(clustered_target_ids)
        individual_count = total_targets - clustered_count
        cluster_task_count = len(self.cluster_schedules)

        clustering_task_count = cluster_task_count + individual_count

        # Improvement ratio
        if traditional_task_count > 0:
            improvement_ratio = (traditional_task_count - clustering_task_count) / traditional_task_count
        else:
            improvement_ratio = 0.0

        # Time saved
        tasks_saved = traditional_task_count - clustering_task_count
        time_saved_seconds = tasks_saved * (self.SETUP_TIME_SECONDS + self.IMAGING_TIME_PER_TARGET_SECONDS)
        time_saved_minutes = time_saved_seconds / 60.0

        # Fuel saved estimate
        # Each saved task saves fuel for setup and reduced imaging
        fuel_per_task_saved = (
            self.FUEL_PER_SETUP_KG +
            self.FUEL_PER_IMAGING_KG * self.IMAGING_TIME_PER_TARGET_SECONDS
        )
        fuel_saved_estimate_kg = tasks_saved * fuel_per_task_saved

        return {
            'traditional_task_count': traditional_task_count,
            'clustering_task_count': clustering_task_count,
            'improvement_ratio': round(max(0.0, improvement_ratio), 3),
            'time_saved_minutes': round(max(0.0, time_saved_minutes), 2),
            'fuel_saved_estimate_kg': round(max(0.0, fuel_saved_estimate_kg), 3),
        }

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

        # Calculate summary statistics
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)

        total_clustered = len(clustered_target_ids)
        total_individual = len(self.all_targets) - total_clustered

        summary = {
            'total_targets': len(self.all_targets),
            'total_clustered_targets': total_clustered,
            'total_individual_targets': total_individual,
            'total_tasks': comparison['clustering_task_count'],
            'cluster_task_count': len(self.cluster_schedules),
            'individual_task_count': total_individual,
        }

        return {
            'efficiency': {
                'task_reduction_ratio': efficiency.task_reduction_ratio,
                'task_reduction_count': efficiency.task_reduction_count,
                'time_savings_seconds': efficiency.time_savings_seconds,
                'avg_targets_per_task': efficiency.avg_targets_per_task,
                'max_targets_in_single_task': efficiency.max_targets_in_single_task,
                'cluster_utilization_ratio': efficiency.cluster_utilization_ratio,
            },
            'coverage': {
                'target_coverage_ratio': coverage.target_coverage_ratio,
                'targets_covered': coverage.targets_covered,
                'targets_total': coverage.targets_total,
                'high_priority_coverage': coverage.high_priority_coverage,
                'high_priority_covered': coverage.high_priority_covered,
                'high_priority_total': coverage.high_priority_total,
                'area_coverage_km2': coverage.area_coverage_km2,
            },
            'quality_score': {
                'overall_score': quality_score.overall_score,
                'efficiency_score': quality_score.efficiency_score,
                'coverage_score': quality_score.coverage_score,
                'priority_score': quality_score.priority_score,
                'balance_score': quality_score.balance_score,
            },
            'comparison': comparison,
            'summary': summary,
            'timestamp': datetime.now(),
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
        clusters_data = []

        for i, cs in enumerate(self.cluster_schedules):
            if not cs.targets:
                continue

            # Calculate cluster center
            center_lon = sum(t.longitude or 0 for t in cs.targets) / len(cs.targets)
            center_lat = sum(t.latitude or 0 for t in cs.targets) / len(cs.targets)

            cluster_data = {
                'cluster_id': cs.cluster_id,
                'center': (center_lon, center_lat),
                'targets': [
                    {
                        'target_id': t.id,
                        'name': t.name,
                        'longitude': t.longitude,
                        'latitude': t.latitude,
                        'priority': t.priority,
                    }
                    for t in cs.targets
                ],
                'color': self.CLUSTER_COLORS[i % len(self.CLUSTER_COLORS)],
                'scheduled': True,
                'satellite_id': cs.satellite_id,
                'imaging_time': cs.imaging_start.isoformat() if cs.imaging_start else None,
            }
            clusters_data.append(cluster_data)

        # Find unclustered targets
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)

        unclustered_targets = [
            {
                'target_id': t.id,
                'name': t.name,
                'longitude': t.longitude,
                'latitude': t.latitude,
                'priority': t.priority,
            }
            for t in self.all_targets
            if t.id not in clustered_target_ids
        ]

        return {
            'clusters': clusters_data,
            'unclustered_targets': unclustered_targets,
        }

    def prepare_coverage_heatmap_data(self) -> Dict[str, Any]:
        """
        准备覆盖热力图数据

        Prepares data for a coverage heatmap showing target density
        and coverage intensity.

        Returns:
            Dictionary with coverage points and intensity values
        """
        coverage_points = []

        # Add points for clustered targets (higher intensity)
        for cs in self.cluster_schedules:
            if not cs.targets:
                continue

            center_lon = sum(t.longitude or 0 for t in cs.targets) / len(cs.targets)
            center_lat = sum(t.latitude or 0 for t in cs.targets) / len(cs.targets)

            # Intensity based on cluster size and priority
            avg_priority = sum(t.priority for t in cs.targets) / len(cs.targets)
            intensity = min(1.0, (len(cs.targets) / 5.0) * (avg_priority / 10.0))

            coverage_points.append({
                'longitude': center_lon,
                'latitude': center_lat,
                'intensity': round(intensity, 2),
                'type': 'cluster',
                'cluster_id': cs.cluster_id,
            })

            # Also add individual target points
            for target in cs.targets:
                coverage_points.append({
                    'longitude': target.longitude or 0,
                    'latitude': target.latitude or 0,
                    'intensity': round(0.3 + (target.priority / 20.0), 2),
                    'type': 'target',
                    'target_id': target.id,
                })

        # Add points for unclustered targets
        clustered_target_ids: Set[str] = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_target_ids.add(target.id)

        for target in self.all_targets:
            if target.id not in clustered_target_ids:
                coverage_points.append({
                    'longitude': target.longitude or 0,
                    'latitude': target.latitude or 0,
                    'intensity': round(0.2 + (target.priority / 20.0), 2),
                    'type': 'unclustered_target',
                    'target_id': target.id,
                })

        return {
            'coverage_points': coverage_points,
        }

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
