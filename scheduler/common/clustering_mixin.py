"""
聚类功能混入类 - 为所有调度器提供目标聚类支持

使用方式:
    class MyScheduler(BaseScheduler, ClusteringMixin):
        def __init__(self, config=None):
            super().__init__("MyScheduler", config)
            ClusteringMixin.__init__(self, config)

        def schedule(self):
            # 如果启用聚类，获取聚类后的任务
            if self.enable_clustering:
                tasks = self._get_clustered_tasks()
            else:
                tasks = self._get_regular_tasks()
            # ... 继续调度逻辑

配置参数:
    - enable_clustering: 是否启用聚类 (默认 False)
    - cluster_radius_km: 聚类半径，公里 (默认 10.0)
    - min_cluster_size: 最小聚类大小 (默认 2)
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from core.models.target import Target


@dataclass
class ClusterTask:
    """聚类任务 - 代表一组邻近目标的成像任务

    作为 Target 的包装器，保持与 Target 接口兼容，
    同时包含聚类信息和多个目标。

    Attributes:
        task_id: 任务唯一标识
        cluster_id: 聚类ID
        targets: 聚类中的所有目标
        centroid: 质心坐标 (经度, 纬度)
        total_priority: 总优先级
        is_cluster: 是否为聚类任务
        primary_target: 主要目标（用于兼容）
    """
    task_id: str
    cluster_id: str
    targets: List[Target]
    centroid: Tuple[float, float]
    total_priority: int
    is_cluster: bool = True

    # 兼容 Target 接口的字段
    @property
    def id(self) -> str:
        return self.task_id

    @property
    def target_id(self) -> str:
        return self.task_id

    @property
    def name(self) -> str:
        if len(self.targets) == 1:
            return self.targets[0].name
        return f"Cluster_{self.cluster_id}({len(self.targets)} targets)"

    @property
    def longitude(self) -> float:
        return self.centroid[0]

    @property
    def latitude(self) -> float:
        return self.centroid[1]

    @property
    def priority(self) -> int:
        """返回平均优先级"""
        if self.targets:
            return self.total_priority // len(self.targets)
        return 5

    @property
    def time_window_start(self) -> Optional[datetime]:
        """返回最早的时间窗口开始"""
        starts = [t.time_window_start for t in self.targets if t.time_window_start]
        return min(starts) if starts else None

    @property
    def time_window_end(self) -> Optional[datetime]:
        """返回最晚的时间窗口结束"""
        ends = [t.time_window_end for t in self.targets if t.time_window_end]
        return max(ends) if ends else None

    @property
    def resolution_required(self) -> Optional[float]:
        """返回最严格的解析度要求"""
        resolutions = [t.resolution_required for t in self.targets if t.resolution_required]
        return min(resolutions) if resolutions else None

    @property
    def target_type(self):
        """返回主要目标类型"""
        return self.targets[0].target_type if self.targets else None

    def get_total_area(self) -> float:
        """获取聚类覆盖的总面积"""
        return sum(t.get_area() for t in self.targets if hasattr(t, 'get_area'))


@dataclass
class ClusterScheduleInfo:
    """聚类调度信息 - 记录聚类调度结果"""
    cluster_id: str
    task_id: str
    targets: List[Target]
    satellite_id: str
    imaging_start: datetime
    imaging_end: datetime
    look_angle: float = 0.0
    priority_satisfied: int = 0


@dataclass
class ClusterSchedule:
    """聚类调度结果 - 向后兼容的简化版本

    与 ClusterScheduleInfo 类似，但不包含 task_id 字段。
    用于测试和向后兼容。
    """
    cluster_id: str
    targets: List[Target]
    satellite_id: str
    imaging_start: datetime
    imaging_end: datetime
    look_angle: float = 0.0
    priority_satisfied: int = 0
    task_id: str = ""  # 可选，向后兼容


class ClusteringMixin:
    """聚类功能混入类

    为任何调度器添加目标聚类能力。
    通过配置参数控制聚类行为。

    Example:
        class MyScheduler(BaseScheduler, ClusteringMixin):
            def __init__(self, config=None):
                super().__init__("MyScheduler", config)
                ClusteringMixin.__init__(self, config)

            def schedule(self):
                if self.enable_clustering:
                    tasks = self._get_clustered_tasks()
                else:
                    tasks = self._create_frequency_aware_tasks()
                # ... 调度逻辑
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化聚类混入类

        Args:
            config: 配置字典
                - enable_clustering: 是否启用聚类 (默认 False)
                - cluster_radius_km: 聚类半径，公里 (默认 10.0)
                - min_cluster_size: 最小聚类大小 (默认 2)
                - altitude_km: 卫星高度，用于 footprint 计算 (默认 500.0)
        """
        config = config or {}
        self.enable_clustering = config.get('enable_clustering', False)
        self.cluster_radius_km = config.get('cluster_radius_km', 10.0)
        self.min_cluster_size = config.get('min_cluster_size', 2)
        self.altitude_km = config.get('altitude_km', 500.0)

        # 内部状态
        self._clusterer = None
        self._cluster_schedules: List[ClusterScheduleInfo] = []
        self._cluster_map: Dict[str, ClusterTask] = {}  # task_id -> ClusterTask
        self._scheduled_cluster_ids: Set[str] = set()

    def _initialize_clustering(self) -> None:
        """初始化聚类器（延迟初始化）"""
        if not self.enable_clustering:
            return

        try:
            from core.clustering.target_clusterer import TargetClusterer
            self._clusterer = TargetClusterer(
                swath_width_km=self.cluster_radius_km,
                min_cluster_size=self.min_cluster_size
            )
        except ImportError as e:
            raise RuntimeError(f"无法导入聚类模块: {e}")

    def _get_clustered_tasks(self) -> List[ClusterTask]:
        """获取聚类后的任务列表

        将目标进行空间聚类，返回 ClusterTask 列表。
        只返回真正的聚类（is_cluster=True），未聚类的单个目标不返回。

        Returns:
            List[ClusterTask]: 聚类任务列表（仅包含真正的聚类）
        """
        if not self.enable_clustering or not hasattr(self, 'mission') or not self.mission:
            return []

        if self._clusterer is None:
            self._initialize_clustering()

        targets = self.mission.targets
        if not targets:
            return []

        # 执行聚类
        clusters = self._clusterer.cluster_targets(targets)

        # 转换为目标列表（只返回真正的聚类）
        tasks = []

        # 添加聚类任务
        for cluster in clusters:
            task = ClusterTask(
                task_id=f"cluster_{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                targets=cluster.targets,
                centroid=cluster.centroid,
                total_priority=cluster.total_priority
            )
            tasks.append(task)
            self._cluster_map[task.task_id] = task

        return tasks

    def _cluster_targets(self) -> List[ClusterTask]:
        """获取聚类后的任务列表（向后兼容别名）

        与 _get_clustered_tasks 相同，为向后兼容提供。
        """
        return self._get_clustered_tasks()

    def _record_cluster_schedule(
        self,
        task: ClusterTask,
        satellite_id: str,
        imaging_start: datetime,
        imaging_end: datetime,
        look_angle: float = 0.0
    ) -> None:
        """记录聚类调度信息

        Args:
            task: 聚类任务
            satellite_id: 卫星ID
            imaging_start: 成像开始时间
            imaging_end: 成像结束时间
            look_angle: 侧摆角
        """
        if not self.enable_clustering:
            return

        priority_satisfied = sum(1 for t in task.targets if t.priority >= 8)

        info = ClusterScheduleInfo(
            cluster_id=task.cluster_id,
            task_id=task.task_id,
            targets=task.targets,
            satellite_id=satellite_id,
            imaging_start=imaging_start,
            imaging_end=imaging_end,
            look_angle=look_angle,
            priority_satisfied=priority_satisfied
        )
        self._cluster_schedules.append(info)
        self._scheduled_cluster_ids.add(task.task_id)

    def get_clustering_metrics(self) -> Dict[str, Any]:
        """获取聚类指标

        Returns:
            聚类相关指标字典
        """
        if not self.enable_clustering:
            return {
                'enabled': False,
                'total_clusters': 0,
                'total_clustered_targets': 0,
                'avg_targets_per_cluster': 0.0,
            }

        total_targets = len(self._cluster_map) if hasattr(self, 'mission') and self.mission else 0
        clustered_count = sum(
            1 for t in self._cluster_map.values() if t.is_cluster
        )
        total_clustered_targets = sum(
            len(t.targets) for t in self._cluster_map.values() if t.is_cluster
        )

        avg_targets = 0.0
        if clustered_count > 0:
            avg_targets = total_clustered_targets / clustered_count

        return {
            'enabled': True,
            'cluster_radius_km': self.cluster_radius_km,
            'min_cluster_size': self.min_cluster_size,
            'total_clusters': clustered_count,
            'total_clustered_targets': total_clustered_targets,
            'avg_targets_per_cluster': round(avg_targets, 2),
            'scheduled_clusters': len(self._cluster_schedules),
        }

    def reset_clustering_state(self) -> None:
        """重置聚类状态（用于重新调度）"""
        self._cluster_schedules = []
        self._cluster_map = {}
        self._scheduled_cluster_ids = set()

    def is_cluster_task(self, task_id: str) -> bool:
        """检查任务是否为聚类任务"""
        task = self._cluster_map.get(task_id)
        return task.is_cluster if task else False

    def get_cluster_task(self, task_id: str) -> Optional[ClusterTask]:
        """获取聚类任务对象"""
        return self._cluster_map.get(task_id)

    @property
    def clusterer(self):
        """获取聚类器实例（延迟初始化，向后兼容）"""
        if self._clusterer is None and self.enable_clustering:
            self._initialize_clustering()
        return self._clusterer

    @property
    def cluster_schedules(self) -> List[ClusterScheduleInfo]:
        """获取聚类调度列表（向后兼容）"""
        return self._cluster_schedules

    def get_clustering_config(self) -> Dict[str, Any]:
        """获取聚类配置"""
        return {
            'enable_clustering': self.enable_clustering,
            'cluster_radius_km': self.cluster_radius_km,
            'min_cluster_size': self.min_cluster_size,
            'altitude_km': self.altitude_km,
        }

    def _populate_cluster_info(self, scheduled_task, task) -> None:
        """填充聚类信息到 ScheduledTask

        如果源任务是 ClusterTask，则填充聚类相关字段。

        Args:
            scheduled_task: ScheduledTask 实例
            task: 源任务对象（可能是 ClusterTask 或普通 Target）
        """
        if isinstance(task, ClusterTask):
            scheduled_task.is_cluster_task = task.is_cluster
            scheduled_task.cluster_id = task.cluster_id
            scheduled_task.covered_target_ids = [t.id for t in task.targets]
            scheduled_task.covered_target_count = len(task.targets)

            # 主目标ID：聚类任务取第一个目标，单个目标取自身
            if task.targets:
                scheduled_task.primary_target_id = task.targets[0].id
            else:
                scheduled_task.primary_target_id = None
        else:
            # 普通目标，不是聚类任务
            scheduled_task.is_cluster_task = False
            scheduled_task.cluster_id = None
            scheduled_task.primary_target_id = getattr(task, 'id', None)
            scheduled_task.covered_target_ids = [getattr(task, 'id', None)] if getattr(task, 'id', None) else []
            scheduled_task.covered_target_count = 1

    def get_cluster_schedule_summary(self) -> Dict[str, Any]:
        """获取聚类调度摘要

        Returns:
            聚类调度统计信息
        """
        if not self.enable_clustering or not self._cluster_schedules:
            return {
                'clustering_enabled': self.enable_clustering,
                'total_cluster_tasks': 0,
                'total_covered_targets': 0,
                'avg_targets_per_cluster': 0.0,
            }

        total_tasks = len(self._cluster_schedules)
        total_covered = sum(len(cs.targets) for cs in self._cluster_schedules)

        return {
            'clustering_enabled': True,
            'total_cluster_tasks': total_tasks,
            'total_covered_targets': total_covered,
            'avg_targets_per_cluster': total_covered / total_tasks if total_tasks > 0 else 0.0,
            'cluster_task_ids': [cs.task_id for cs in self._cluster_schedules],
        }
