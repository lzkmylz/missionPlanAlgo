"""
单圈约束批量检查器

检查：
1. 单圈最大开机次数（max_starts_per_orbit）
2. 单圈载荷最大工作时长（max_work_time_per_orbit）

使用滑动圈定义：
- 以候选任务开始时间为基准，圈范围 = [task_start, task_start + orbit_period]
- 统计该滑动圈内所有已调度任务的开机次数和工作时长
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from core.models.mission import Mission

logger = logging.getLogger(__name__)


@dataclass
class BatchOrbitConstraintCandidate:
    """单圈约束检查候选"""
    sat_id: str
    window_start: datetime  # 任务开始时间
    window_end: datetime    # 任务结束时间
    imaging_duration: float  # 成像持续时间（秒）


@dataclass
class BatchOrbitConstraintResult:
    """单圈约束检查结果"""
    feasible: bool
    reason: Optional[str] = None
    orbit_start_time: Optional[datetime] = None  # 滑动圈开始时间
    orbit_end_time: Optional[datetime] = None    # 滑动圈结束时间
    current_starts_in_orbit: int = 0      # 圈内已开机次数（含候选）
    current_work_time_in_orbit: float = 0.0  # 圈内已工作时长（含候选，秒）
    max_starts_allowed: int = 5           # 最大允许开机次数
    max_work_time_allowed: float = 600.0  # 最大允许工作时长（秒）


class BatchOrbitConstraintChecker:
    """
    单圈约束批量检查器

    配置参数（从SatelliteCapabilities读取）:
    - max_starts_per_orbit: 单圈最大开机次数（默认5）
    - max_work_time_per_orbit: 单圈最大工作时长（秒，默认600）
    """

    # 默认约束值
    DEFAULT_MAX_STARTS = 5
    DEFAULT_MAX_WORK_TIME = 600.0  # 10分钟

    def __init__(self, mission: Mission):
        """
        初始化单圈约束检查器

        Args:
            mission: 任务对象，包含所有卫星信息
        """
        self.mission = mission
        self._orbit_periods: Dict[str, float] = {}  # 缓存轨道周期
        self._satellite_configs: Dict[str, Dict[str, Any]] = {}  # 缓存卫星约束配置

        self._initialize_satellite_data()

        # 性能统计
        self._stats = {
            'total_checks': 0,
            'batch_checks': 0,
            'starts_violations': 0,
            'work_time_violations': 0
        }

    def _initialize_satellite_data(self):
        """初始化卫星数据（轨道周期和约束配置）"""
        for sat in self.mission.satellites:
            # 缓存轨道周期（秒）
            period = sat.orbit.get_period()
            if period <= 0:
                raise ValueError(
                    f"Invalid orbit period for satellite {sat.id}: {period}. "
                    f"Period must be positive."
                )
            self._orbit_periods[sat.id] = period

            # 读取约束配置（从capabilities）
            caps = sat.capabilities

            # 类型转换和验证
            max_starts = int(getattr(caps, 'max_starts_per_orbit', self.DEFAULT_MAX_STARTS))
            max_work = float(getattr(caps, 'max_work_time_per_orbit', self.DEFAULT_MAX_WORK_TIME))

            # 验证约束值为正
            if max_starts <= 0:
                logger.warning(
                    f"Invalid max_starts_per_orbit ({max_starts}) for satellite {sat.id}, "
                    f"using default {self.DEFAULT_MAX_STARTS}"
                )
                max_starts = self.DEFAULT_MAX_STARTS

            if max_work <= 0:
                logger.warning(
                    f"Invalid max_work_time_per_orbit ({max_work}) for satellite {sat.id}, "
                    f"using default {self.DEFAULT_MAX_WORK_TIME}"
                )
                max_work = self.DEFAULT_MAX_WORK_TIME

            self._satellite_configs[sat.id] = {
                'max_starts_per_orbit': max_starts,
                'max_work_time_per_orbit': max_work,
            }

    def check_batch(
        self,
        candidates: List[BatchOrbitConstraintCandidate],
        existing_tasks: List[Dict[str, Any]]
    ) -> List[BatchOrbitConstraintResult]:
        """
        批量检查单圈约束

        使用滑动圈定义：
        - 对每个候选，定义滑动圈 = [candidate.window_start, candidate.window_start + orbit_period]
        - 统计该滑动圈内所有已调度任务的开机次数和工作时长

        Args:
            candidates: 候选任务列表
            existing_tasks: 已调度任务列表，每个任务包含:
                - satellite_id: str
                - imaging_start: datetime
                - imaging_end: datetime

        Returns:
            检查结果列表
        """
        if not candidates:
            return []

        self._stats['total_checks'] += len(candidates)
        self._stats['batch_checks'] += 1

        # 构建已调度任务的快速查询结构
        # 按卫星ID分组
        tasks_by_sat: Dict[str, List[Dict[str, Any]]] = {}
        for task in existing_tasks:
            sat_id = task.get('satellite_id')
            if sat_id:
                if sat_id not in tasks_by_sat:
                    tasks_by_sat[sat_id] = []
                tasks_by_sat[sat_id].append(task)

        # 批量检查
        results = []
        for candidate in candidates:
            result = self._check_single_candidate(candidate, tasks_by_sat)
            results.append(result)

        return results

    def _check_single_candidate(
        self,
        candidate: BatchOrbitConstraintCandidate,
        tasks_by_sat: Dict[str, List[Dict[str, Any]]]
    ) -> BatchOrbitConstraintResult:
        """
        检查单个候选的单圈约束

        Args:
            candidate: 候选任务
            tasks_by_sat: 按卫星分组的已调度任务

        Returns:
            检查结果
        """
        sat_id = candidate.sat_id
        config = self._satellite_configs[sat_id]
        orbit_period = self._orbit_periods[sat_id]

        # 定义滑动圈范围
        orbit_start = candidate.window_start
        orbit_end = orbit_start + timedelta(seconds=orbit_period)

        # 获取该卫星的已调度任务
        sat_tasks = tasks_by_sat.get(sat_id, [])

        # 统计滑动圈内的任务
        starts_in_orbit = 0  # 开机次数
        work_time_in_orbit = 0.0  # 工作时长（秒）

        for task in sat_tasks:
            task_start = task.get('imaging_start')
            task_end = task.get('imaging_end')

            if task_start is None or task_end is None:
                continue

            # 检查任务是否与滑动圈重叠
            # 任务在圈内当: task_start < orbit_end AND task_end > orbit_start
            if task_start < orbit_end and task_end > orbit_start:
                starts_in_orbit += 1

                # 计算在滑动圈内的工作时长
                # 重叠区间 = [max(task_start, orbit_start), min(task_end, orbit_end)]
                overlap_start = max(task_start, orbit_start)
                overlap_end = min(task_end, orbit_end)
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                work_time_in_orbit += max(0, overlap_duration)

        # 加上候选任务本身
        starts_in_orbit += 1
        work_time_in_orbit += candidate.imaging_duration

        # 检查开机次数约束
        if starts_in_orbit > config['max_starts_per_orbit']:
            self._stats['starts_violations'] += 1
            return BatchOrbitConstraintResult(
                feasible=False,
                reason=f"Max starts per orbit exceeded: {starts_in_orbit} > {config['max_starts_per_orbit']}",
                orbit_start_time=orbit_start,
                orbit_end_time=orbit_end,
                current_starts_in_orbit=starts_in_orbit,
                current_work_time_in_orbit=work_time_in_orbit,
                max_starts_allowed=config['max_starts_per_orbit'],
                max_work_time_allowed=config['max_work_time_per_orbit']
            )

        # 检查工作时长约束
        if work_time_in_orbit > config['max_work_time_per_orbit']:
            self._stats['work_time_violations'] += 1
            return BatchOrbitConstraintResult(
                feasible=False,
                reason=f"Max work time per orbit exceeded: {work_time_in_orbit:.1f}s > {config['max_work_time_per_orbit']:.1f}s",
                orbit_start_time=orbit_start,
                orbit_end_time=orbit_end,
                current_starts_in_orbit=starts_in_orbit,
                current_work_time_in_orbit=work_time_in_orbit,
                max_starts_allowed=config['max_starts_per_orbit'],
                max_work_time_allowed=config['max_work_time_per_orbit']
            )

        # 约束满足
        return BatchOrbitConstraintResult(
            feasible=True,
            orbit_start_time=orbit_start,
            orbit_end_time=orbit_end,
            current_starts_in_orbit=starts_in_orbit,
            current_work_time_in_orbit=work_time_in_orbit,
            max_starts_allowed=config['max_starts_per_orbit'],
            max_work_time_allowed=config['max_work_time_per_orbit']
        )

    def get_batch_stats(self) -> Dict[str, Any]:
        """获取批量检查统计信息"""
        return self._stats.copy()

    def reset_batch_stats(self):
        """重置统计信息"""
        self._stats = {
            'total_checks': 0,
            'batch_checks': 0,
            'starts_violations': 0,
            'work_time_violations': 0
        }
