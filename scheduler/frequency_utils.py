"""
频次约束工具模块

为调度算法提供观测频次约束的支持。
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime


@dataclass
class ObservationTask:
    """观测任务 - 支持多次观测"""
    task_id: str
    target_id: str
    target_name: str
    observation_idx: int  # 第几次观测（0-based）
    required_observations: int  # -1表示不限
    priority: int
    longitude: float
    latitude: float
    # 兼容Target的属性
    target_type: Any = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    resolution_required: float = 10.0
    data_size_gb: float = 1.0

    @property
    def id(self) -> str:
        """兼容原始Target的id属性"""
        return self.task_id


def create_observation_tasks(targets) -> List[ObservationTask]:
    """
    根据目标的观测需求创建任务列表

    Args:
        targets: 目标列表

    Returns:
        List[ObservationTask]: 观测任务列表
    """
    tasks = []

    for target in targets:
        required = getattr(target, 'required_observations', 1)

        if required == -1:
            # 不限频次 - 最多创建10个任务（足够多的观测机会）
            max_obs = 10
        else:
            max_obs = required

        for idx in range(max_obs):
            task_id = f"{target.id}-OBS{idx+1}"
            tasks.append(ObservationTask(
                task_id=task_id,
                target_id=target.id,
                target_name=target.name,
                observation_idx=idx,
                required_observations=required,
                priority=target.priority,
                longitude=target.longitude,
                latitude=target.latitude,
                target_type=getattr(target, 'target_type', None),
                time_window_start=getattr(target, 'time_window_start', None),
                time_window_end=getattr(target, 'time_window_end', None),
                resolution_required=getattr(target, 'resolution_required', 10.0),
                data_size_gb=getattr(target, 'data_size_gb', 1.0)
            ))

    return tasks


def calculate_frequency_fitness(
    target_obs_count: Dict[str, int],
    targets,
    base_score: float = 0.0
) -> float:
    """
    计算频次满足度的适应度分数

    Args:
        target_obs_count: 每个目标的实际观测次数
        targets: 目标列表
        base_score: 基础分数

    Returns:
        float: 总适应度分数
    """
    score = base_score

    for target in targets:
        required = getattr(target, 'required_observations', 1)
        actual = target_obs_count.get(target.id, 0)

        if required == -1:
            # 不限频次，每次观测都给予奖励
            score += actual * 5.0
        else:
            # 指定频次，满足需求给予奖励
            if actual >= required:
                score += 20.0  # 完成奖励
                score += min(actual - required, 2) * 2.0  # 少量超额奖励
            else:
                score += (actual / required) * 15.0  # 部分完成按比例奖励

    return score


def get_target_observation_requirement(target) -> int:
    """
    获取目标的观测需求次数

    Args:
        target: 目标对象

    Returns:
        int: 观测需求次数，-1表示不限
    """
    return getattr(target, 'required_observations', 1)


def is_target_fully_satisfied(target, actual_count: int) -> bool:
    """
    检查目标是否已满足观测需求

    Args:
        target: 目标对象
        actual_count: 实际观测次数

    Returns:
        bool: 是否已满足
    """
    required = getattr(target, 'required_observations', 1)

    if required == -1:
        # 不限频次，只要有观测就算满足（但不阻止更多观测）
        return actual_count > 0
    else:
        return actual_count >= required


def get_observation_gap(target_id: str, sat_task_times: Dict[int, List[Tuple[datetime, datetime]]],
                       sat_idx: int, gap_seconds: int = 60) -> bool:
    """
    检查同一目标的不同观测之间是否有足够时间间隔

    Args:
        target_id: 目标ID
        sat_task_times: 卫星任务时间记录
        sat_idx: 卫星索引
        gap_seconds: 最小间隔秒数

    Returns:
        bool: 是否有足够间隔
    """
    from datetime import timedelta

    min_gap = timedelta(seconds=gap_seconds)

    if sat_idx not in sat_task_times:
        return True

    # 这个函数需要在具体调度器中实现详细逻辑
    # 这里只是一个占位符
    return True
