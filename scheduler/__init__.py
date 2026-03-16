"""
调度器模块

提供卫星任务调度算法的实现，包括：
- 贪心调度器 (GreedyScheduler)
- 元启发式调度器 (GA, ACO, SA, PSO, Tabu)
- 区域目标拼幅覆盖支持
"""

from .base_scheduler import (
    BaseScheduler,
    ScheduleResult,
    ScheduledTask,
    TaskFailure,
    TaskFailureReason,
)
from .frequency_utils import ObservationTask, create_observation_tasks
from .area_task_utils import (
    AreaObservationTask,
    MixedTaskList,
    create_area_observation_tasks,
    create_mixed_task_list,
    calculate_area_coverage_score,
    calculate_coverage_fitness,
)
from .coverage_tracker import CoverageTracker, CoverageState, TileCoverageRecord

__all__ = [
    # Base
    'BaseScheduler',
    'ScheduleResult',
    'ScheduledTask',
    'TaskFailure',
    'TaskFailureReason',
    # Frequency
    'ObservationTask',
    'create_observation_tasks',
    # Area Coverage
    'AreaObservationTask',
    'MixedTaskList',
    'create_area_observation_tasks',
    'create_mixed_task_list',
    'calculate_area_coverage_score',
    'calculate_coverage_fitness',
    'CoverageTracker',
    'CoverageState',
    'TileCoverageRecord',
]
