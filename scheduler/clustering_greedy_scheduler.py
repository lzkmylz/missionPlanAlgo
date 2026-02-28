"""
ClusteringGreedyScheduler - 支持目标聚类的贪心调度器

扩展GreedyScheduler，在调度前进行目标聚类，一次成像覆盖多个邻近目标。

Key Features:
1. Target Clustering: Before scheduling, cluster nearby targets
2. Cluster-Aware Assignment: Find visibility windows for clusters (not individual targets)
3. Multi-Target Tasks: Schedule one imaging task to cover multiple targets
4. Quality Metrics: Track efficiency gains, coverage ratios, priority satisfaction
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.base_scheduler import ScheduleResult, ScheduledTask, TaskFailureReason
from core.clustering.target_clusterer import TargetClusterer, TargetCluster
from core.coverage.footprint_calculator import FootprintCalculator
from core.orbit.visibility.base import VisibilityWindow
from core.models.target import Target


@dataclass
class ClusterSchedule:
    """聚类调度结果

    Attributes:
        cluster_id: 聚类唯一标识
        targets: 被覆盖的所有目标
        satellite_id: 执行成像的卫星ID
        imaging_start: 成像开始时间
        imaging_end: 成像结束时间
        look_angle: 使用的侧摆角（度）
        priority_satisfied: 满足的高优先级目标数
    """
    cluster_id: str
    targets: List[Target]
    satellite_id: str
    imaging_start: datetime
    imaging_end: datetime
    look_angle: float
    priority_satisfied: int


class ClusteringGreedyScheduler(GreedyScheduler):
    """
    支持目标聚类的贪心调度器

    扩展GreedyScheduler，在调度前进行目标聚类，
    一次成像覆盖多个邻近目标。

    Attributes:
        clusterer: TargetClusterer instance for spatial clustering
        footprint_calc: FootprintCalculator for coverage analysis
        cluster_schedules: List of ClusterSchedule tracking cluster assignments
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ClusteringGreedyScheduler

        Args:
            config: Configuration dictionary
                - swath_width_km: Satellite swath width in km (default 10.0)
                - min_cluster_size: Minimum targets to form a cluster (default 2)
                - altitude_km: Satellite altitude in km (default 500.0)
                - heuristic: Task sorting heuristic (default 'priority')
                - consider_power: Whether to check power constraints (default True)
                - consider_storage: Whether to check storage constraints (default True)
                - consider_time_conflicts: Whether to check time conflicts (default True)
        """
        super().__init__(config)
        config = config or {}

        self.clusterer = TargetClusterer(
            swath_width_km=config.get('swath_width_km', 10.0),
            min_cluster_size=config.get('min_cluster_size', 2)
        )
        self.footprint_calc = FootprintCalculator(
            satellite_altitude_km=config.get('altitude_km', 500.0)
        )
        self.cluster_schedules: List[ClusterSchedule] = []

        # Track scheduled targets to avoid duplicates
        self._scheduled_target_ids: set = set()

    def schedule(self) -> ScheduleResult:
        """
        Execute cluster-aware greedy scheduling

        Strategy:
        1. Cluster targets spatially
        2. Sort clusters by priority density
        3. For each cluster, find best satellite-window
        4. Fall back to individual scheduling for unclustered targets
        5. Record efficiency metrics

        Returns:
            ScheduleResult with all scheduled tasks and cluster information
        """
        from scheduler.frequency_utils import ObservationTask

        self._start_timer()
        self._validate_initialization()

        # Reset tracking
        self.cluster_schedules = []
        self._scheduled_target_ids = set()

        # Initialize resource tracking for each satellite
        self._sat_resource_usage = {
            sat.id: {
                'power': sat.current_power if hasattr(sat, 'current_power') and sat.current_power > 0
                        else sat.capabilities.power_capacity,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': []
            }
            for sat in self.mission.satellites
        }

        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, Any] = {}

        # Step 1: Cluster targets
        clusters = self._cluster_targets()

        # Step 2: Sort clusters by priority density (higher first)
        sorted_clusters = self._sort_clusters_by_priority_density(clusters)

        # Step 3: Schedule clusters
        for cluster in sorted_clusters:
            if self._are_all_targets_scheduled(cluster.targets):
                continue

            best_assignment = self._find_best_window_for_cluster(cluster)

            if best_assignment:
                sat_id, window, look_angle = best_assignment

                # Create scheduled task for the cluster
                scheduled_task = self._create_cluster_scheduled_task(
                    cluster, sat_id, window, look_angle
                )
                scheduled_tasks.append(scheduled_task)

                # Update resource usage
                self._update_resource_usage_for_cluster(sat_id, cluster, scheduled_task)

                # Track cluster schedule
                priority_satisfied = self._count_high_priority_targets(cluster.targets)
                cluster_schedule = ClusterSchedule(
                    cluster_id=cluster.cluster_id,
                    targets=cluster.targets.copy(),
                    satellite_id=sat_id,
                    imaging_start=scheduled_task.imaging_start,
                    imaging_end=scheduled_task.imaging_end,
                    look_angle=look_angle,
                    priority_satisfied=priority_satisfied
                )
                self.cluster_schedules.append(cluster_schedule)

                # Mark targets as scheduled
                for target in cluster.targets:
                    self._scheduled_target_ids.add(target.id)

                self._add_convergence_point(len(scheduled_tasks))

        # Step 4: Fall back to individual scheduling for remaining targets
        remaining_targets = self._get_unscheduled_targets()
        if remaining_targets:
            individual_tasks = self._schedule_individual_targets(remaining_targets)
            scheduled_tasks.extend(individual_tasks)

        # Calculate makespan
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

        # Build failure summary
        failure_summary = self._build_failure_summary()

        # Calculate target observation counts
        target_obs_count = self._calculate_target_obs_count(scheduled_tasks)

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self._iterations,
            convergence_curve=self._convergence_curve,
            failure_summary=failure_summary
        )

    def _cluster_targets(self) -> List[TargetCluster]:
        """
        对所有目标进行空间聚类

        Returns:
            List of TargetCluster objects
        """
        if not self.mission or not self.mission.targets:
            return []

        return self.clusterer.cluster_targets(self.mission.targets)

    def _sort_clusters_by_priority_density(
        self, clusters: List[TargetCluster]
    ) -> List[TargetCluster]:
        """
        按优先级密度排序聚类（高优先级密度优先）

        Args:
            clusters: List of clusters to sort

        Returns:
            Sorted list of clusters
        """
        def priority_density(cluster: TargetCluster) -> float:
            if not cluster.targets:
                return 0.0
            return cluster.total_priority / len(cluster.targets)

        return sorted(clusters, key=priority_density, reverse=True)

    def _find_best_window_for_cluster(
        self,
        cluster: TargetCluster
    ) -> Optional[Tuple[str, VisibilityWindow, float]]:
        """
        为聚类寻找最佳成像窗口

        Args:
            cluster: Target cluster to schedule

        Returns:
            Tuple of (satellite_id, visibility_window, look_angle) or None
        """
        best_assignment = None
        best_score = None

        # Calculate cluster centroid
        centroid = cluster.centroid

        for sat in self.mission.satellites:
            # Check if satellite can perform imaging
            if not sat.capabilities.imaging_modes:
                continue

            # Get visibility windows for the cluster centroid
            # We use the centroid as a representative point
            windows = self._get_windows_for_cluster(sat, cluster)

            if not windows:
                continue

            for window in windows:
                # Check if window is valid for all targets in cluster
                window_start, window_end = self._extract_window_times(window)
                if window_start is None or window_end is None:
                    continue

                # Check off-nadir constraint for the cluster
                can_cover, look_angle = self._check_cluster_coverage(
                    sat, cluster, window
                )

                if not can_cover:
                    continue

                # Check if look angle is within satellite's max off-nadir
                if abs(look_angle) > sat.capabilities.max_off_nadir:
                    continue

                # Check resource constraints
                imaging_mode = self._select_imaging_mode(sat, None)
                if not self._check_cluster_resource_constraints(sat, cluster, imaging_mode):
                    continue

                # Check time conflicts
                if self.consider_time_conflicts:
                    imaging_duration = self._calculate_cluster_imaging_time(cluster, imaging_mode)
                    actual_start, actual_end = self._calculate_task_time(
                        sat.id, window_start, imaging_duration
                    )
                    if self._has_time_conflict(sat.id, actual_start, actual_end):
                        continue

                # Calculate score
                score = self._calculate_cluster_assignment_score(
                    sat, cluster, window, look_angle
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_assignment = (sat.id, window, look_angle)

        return best_assignment

    def _get_windows_for_cluster(
        self, sat, cluster: TargetCluster
    ) -> List[VisibilityWindow]:
        """
        Get visibility windows for a cluster

        Uses the cluster centroid as the representative point.

        Args:
            sat: Satellite
            cluster: Target cluster

        Returns:
            List of visibility windows
        """
        if not self.window_cache:
            return []

        # Use centroid target ID pattern for cluster
        # In practice, we check windows for the centroid position
        # For simplicity, we use the first target's windows as representative
        if not cluster.targets:
            return []

        # Get windows for the first target as representative
        # In a full implementation, we would compute windows for the centroid
        representative_target = cluster.targets[0]
        return self.window_cache.get_windows(sat.id, representative_target.id)

    def _check_cluster_coverage(
        self, sat, cluster: TargetCluster, window
    ) -> Tuple[bool, float]:
        """
        Check if satellite can cover all targets in cluster

        Args:
            sat: Satellite
            cluster: Target cluster
            window: Visibility window

        Returns:
            Tuple of (can_cover, required_look_angle)
        """
        # Get satellite position at window start
        window_start = window.start_time if hasattr(window, 'start_time') else window.get('start')

        if window_start is None:
            return False, 0.0

        # Get satellite position (simplified - use subpoint as nadir)
        try:
            sat_position = sat.get_position_sgp4(window_start)
            # Convert to meters if in km
            if abs(sat_position[0]) < 10000:  # Likely in km
                sat_position = (p * 1000 for p in sat_position)
                sat_position = tuple(sat_position)
        except Exception:
            # Fallback: estimate position from orbit
            sat_position = self._estimate_satellite_position(sat, window_start)

        # Get nadir position
        try:
            nadir_lat, nadir_lon, _ = sat.get_subpoint(window_start)
            nadir_position = (nadir_lon, nadir_lat)
        except Exception:
            # Fallback to cluster centroid
            nadir_position = cluster.centroid

        # Check coverage using footprint calculator
        swath_width_km = sat.capabilities.swath_width / 1000.0  # Convert m to km

        can_cover, look_angle = self.footprint_calc.can_cover_targets(
            targets=cluster.targets,
            satellite_position=sat_position,
            nadir_position=nadir_position,
            max_off_nadir=sat.capabilities.max_off_nadir,
            swath_width_km=swath_width_km
        )

        return can_cover, look_angle

    def _estimate_satellite_position(
        self, sat, dt: datetime
    ) -> Tuple[float, float, float]:
        """
        Estimate satellite ECEF position

        Simplified estimation when precise propagation is not available.

        Args:
            sat: Satellite
            dt: Datetime

        Returns:
            ECEF position (x, y, z) in meters
        """
        # Simplified: assume circular orbit
        orbit_radius = 6371000.0 + sat.orbit.altitude  # Earth radius + altitude

        # Estimate position based on orbit parameters
        # This is a very simplified model
        import math
        period = sat.orbit.get_period()
        mean_motion = 2 * math.pi / period

        ref_time = datetime(2024, 1, 1)
        delta_t = (dt - ref_time).total_seconds()

        M = math.radians(sat.orbit.mean_anomaly) + mean_motion * delta_t
        i = math.radians(sat.orbit.inclination)
        raan = math.radians(sat.orbit.raan)

        x_orb = orbit_radius * math.cos(M)
        y_orb = orbit_radius * math.sin(M)

        x = x_orb * math.cos(raan) - y_orb * math.cos(i) * math.sin(raan)
        y = x_orb * math.sin(raan) + y_orb * math.cos(i) * math.cos(raan)
        z = y_orb * math.sin(i)

        return (x, y, z)

    def _check_cluster_resource_constraints(
        self, sat, cluster: TargetCluster, imaging_mode
    ) -> bool:
        """
        Check if satellite has sufficient resources for the cluster

        Args:
            sat: Satellite
            cluster: Target cluster
            imaging_mode: Imaging mode to use

        Returns:
            True if resources are sufficient
        """
        usage = self._sat_resource_usage.get(sat.id, {})

        # Calculate imaging time for cluster (longer for multiple targets)
        imaging_duration = self._calculate_cluster_imaging_time(cluster, imaging_mode)

        # Check power constraint
        if self.consider_power:
            from payload.imaging_time_calculator import PowerProfile
            power_profile = PowerProfile()
            power_coefficient = power_profile.get_coefficient_for_mode(imaging_mode)
            power_needed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)

            if usage.get('power', 0) < power_needed:
                return False

        # Check storage constraint
        if self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            # Estimate storage for cluster (sum of individual targets)
            storage_needed = 0.0
            for target in cluster.targets:
                storage_needed += self._imaging_calculator.get_storage_consumption(
                    target, imaging_mode, data_rate
                )

            current_storage = usage.get('storage', 0)
            capacity = sat.capabilities.storage_capacity

            if current_storage + storage_needed > capacity:
                return False

        return True

    def _calculate_cluster_imaging_time(
        self, cluster: TargetCluster, imaging_mode
    ) -> float:
        """
        Calculate imaging time for a cluster

        Longer than single target to ensure all targets are covered.

        Args:
            cluster: Target cluster
            imaging_mode: Imaging mode

        Returns:
            Imaging duration in seconds
        """
        # Base imaging time for first target
        if not cluster.targets:
            return 60.0

        base_time = self._imaging_calculator.calculate(cluster.targets[0], imaging_mode)

        # Add time for additional targets (reduced overhead for clustering)
        additional_targets = len(cluster.targets) - 1
        if additional_targets > 0:
            # Each additional target adds 50% of base time (overlap efficiency)
            base_time += additional_targets * base_time * 0.5

        return min(base_time, 1800.0)  # Cap at 30 minutes

    def _calculate_cluster_assignment_score(
        self, sat, cluster: TargetCluster, window, look_angle: float
    ) -> float:
        """
        Calculate score for cluster assignment

        Args:
            sat: Satellite
            cluster: Target cluster
            window: Visibility window
            look_angle: Required look angle

        Returns:
            Score value (higher is better)
        """
        score = 0.0

        # Prefer clusters with more targets
        score += len(cluster.targets) * 100

        # Prefer higher total priority
        score += cluster.total_priority * 10

        # Prefer smaller look angles (less slewing)
        score -= abs(look_angle) * 2

        # Prefer satellites with more remaining resources
        usage = self._sat_resource_usage.get(sat.id, {})
        power_ratio = usage.get('power', 0) / sat.capabilities.power_capacity
        storage_ratio = 1.0 - (usage.get('storage', 0) / sat.capabilities.storage_capacity)
        score += (power_ratio + storage_ratio) * 5

        return score

    def _create_cluster_scheduled_task(
        self, cluster: TargetCluster, sat_id: str,
        window, look_angle: float
    ) -> ScheduledTask:
        """
        Create a ScheduledTask for a cluster

        Args:
            cluster: Target cluster
            sat_id: Satellite ID
            window: Visibility window
            look_angle: Look angle used

        Returns:
            ScheduledTask object
        """
        window_start, window_end = self._extract_window_times(window)

        # Get satellite
        sat = None
        for s in self.mission.satellites:
            if s.id == sat_id:
                sat = s
                break

        imaging_mode = self._select_imaging_mode(sat, None) if sat else None
        imaging_duration = self._calculate_cluster_imaging_time(cluster, imaging_mode)

        # Calculate actual timing
        actual_start, actual_end = self._calculate_task_time(
            sat_id, window_start, imaging_duration
        )

        # Get current resource levels
        usage = self._sat_resource_usage.get(sat_id, {})
        power_before = usage.get('power', 0)
        storage_before = usage.get('storage', 0)

        # Calculate resource consumption
        power_consumed = 0.0
        storage_used = 0.0

        if sat:
            if self.consider_power:
                from payload.imaging_time_calculator import PowerProfile
                power_profile = PowerProfile()
                power_coefficient = power_profile.get_coefficient_for_mode(imaging_mode)
                power_consumed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)

            if self.consider_storage:
                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                for target in cluster.targets:
                    storage_used += self._imaging_calculator.get_storage_consumption(
                        target, imaging_mode, data_rate
                    )

        # Use first target ID as primary, but include all in context
        primary_target_id = cluster.targets[0].id if cluster.targets else "cluster"

        return ScheduledTask(
            task_id=f"cluster_{cluster.cluster_id}",
            satellite_id=sat_id,
            target_id=primary_target_id,
            imaging_start=actual_start,
            imaging_end=actual_end,
            imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
            slew_angle=look_angle,
            storage_before=storage_before,
            storage_after=storage_before + storage_used,
            power_before=power_before,
            power_after=power_before - power_consumed
        )

    def _update_resource_usage_for_cluster(
        self, sat_id: str, cluster: TargetCluster, scheduled_task: ScheduledTask
    ) -> None:
        """
        Update resource usage after scheduling a cluster

        Args:
            sat_id: Satellite ID
            cluster: Target cluster
            scheduled_task: Created scheduled task
        """
        usage = self._sat_resource_usage.get(sat_id)
        if usage is None:
            return

        # Update power
        if self.consider_power:
            usage['power'] = scheduled_task.power_after

        # Update storage
        if self.consider_storage:
            usage['storage'] = scheduled_task.storage_after

        # Update last task end time
        usage['last_task_end'] = scheduled_task.imaging_end

        # Track scheduled task for conflict detection
        if 'scheduled_tasks' not in usage:
            usage['scheduled_tasks'] = []
        usage['scheduled_tasks'].append({
            'start': scheduled_task.imaging_start,
            'end': scheduled_task.imaging_end,
            'task_id': scheduled_task.task_id,
            'cluster_id': cluster.cluster_id
        })

    def _are_all_targets_scheduled(self, targets: List[Target]) -> bool:
        """
        Check if all targets in list are already scheduled

        Args:
            targets: List of targets

        Returns:
            True if all targets are scheduled
        """
        return all(t.id in self._scheduled_target_ids for t in targets)

    def _get_unscheduled_targets(self) -> List[Target]:
        """
        Get targets that haven't been scheduled yet

        Returns:
            List of unscheduled targets
        """
        if not self.mission or not self.mission.targets:
            return []

        return [
            t for t in self.mission.targets
            if t.id not in self._scheduled_target_ids
        ]

    def _schedule_individual_targets(self, targets: List[Target]) -> List[ScheduledTask]:
        """
        Schedule remaining targets individually

        Falls back to parent class scheduling for individual targets.

        Args:
            targets: List of targets to schedule

        Returns:
            List of scheduled tasks
        """
        from scheduler.frequency_utils import ObservationTask

        scheduled_tasks = []

        # Create observation tasks for remaining targets
        observation_tasks = []
        for target in targets:
            task = ObservationTask(
                task_id=target.id,
                target_id=target.id,
                target_name=target.name,
                observation_idx=0,
                required_observations=1,
                priority=target.priority,
                longitude=target.longitude or 0.0,
                latitude=target.latitude or 0.0,
                target_type=target.target_type,
                time_window_start=target.time_window_start,
                time_window_end=target.time_window_end,
                resolution_required=target.resolution_required,
                data_size_gb=1.0
            )
            observation_tasks.append(task)

        # Sort by priority
        observation_tasks = self._sort_tasks(observation_tasks)

        # Schedule each task
        for task in observation_tasks:
            best_assignment = self._find_best_assignment(task)

            if best_assignment:
                sat_id, window, imaging_mode = best_assignment

                scheduled_task = self._create_scheduled_task(
                    task, sat_id, window, imaging_mode
                )
                scheduled_tasks.append(scheduled_task)

                self._update_resource_usage(sat_id, task, window, scheduled_task)
                self._scheduled_target_ids.add(task.target_id)
                self._add_convergence_point(len(scheduled_tasks))
            else:
                reason = self._determine_failure_reason(task)
                self._record_failure(
                    task_id=task.task_id,
                    reason=reason,
                    detail=f"No feasible assignment found for task {task.task_id}"
                )

        return scheduled_tasks

    def _count_high_priority_targets(self, targets: List[Target]) -> int:
        """
        Count high priority targets in list

        Args:
            targets: List of targets

        Returns:
            Number of high priority targets (priority >= 8)
        """
        return sum(1 for t in targets if t.priority >= 8)

    def get_efficiency_metrics(self) -> Dict[str, float]:
        """
        获取效率提升指标

        Returns:
            Dictionary with efficiency metrics:
                - task_reduction_ratio: 任务减少比例 (0-1)
                - high_priority_coverage: 高优先级目标覆盖率 (0-1)
                - avg_targets_per_task: 平均每任务目标数
        """
        if not self.mission or not self.mission.targets:
            return {
                'task_reduction_ratio': 0.0,
                'high_priority_coverage': 0.0,
                'avg_targets_per_task': 0.0
            }

        total_targets = len(self.mission.targets)
        total_tasks = len(self.cluster_schedules)

        # Count targets covered by clusters
        clustered_targets = set()
        for cs in self.cluster_schedules:
            for target in cs.targets:
                clustered_targets.add(target.id)

        # Task reduction ratio
        # If we have N targets and M cluster tasks, reduction = 1 - M/N
        if total_targets > 0:
            task_reduction_ratio = 1.0 - (total_tasks / total_targets)
        else:
            task_reduction_ratio = 0.0

        # High priority coverage
        high_priority_targets = [
            t for t in self.mission.targets if t.priority >= 8
        ]
        covered_high_priority = [
            t for t in high_priority_targets
            if t.id in clustered_targets
        ]

        if high_priority_targets:
            high_priority_coverage = len(covered_high_priority) / len(high_priority_targets)
        else:
            high_priority_coverage = 1.0  # No high priority targets = full coverage

        # Average targets per task
        if total_tasks > 0:
            total_clustered = sum(len(cs.targets) for cs in self.cluster_schedules)
            avg_targets_per_task = total_clustered / total_tasks
        else:
            avg_targets_per_task = 0.0

        return {
            'task_reduction_ratio': max(0.0, task_reduction_ratio),
            'high_priority_coverage': high_priority_coverage,
            'avg_targets_per_task': avg_targets_per_task
        }
