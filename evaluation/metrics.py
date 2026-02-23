"""
性能指标计算模块

计算调度算法的各项性能指标
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类

    Attributes:
        demand_satisfaction_rate: 需求满足率 (0-1)
        makespan: 总完成时间 (秒)
        avg_revisit_time: 平均观测间隔 (秒)
        data_delivery_time: 数据回传用时 (秒)
        computation_time: 算法求解用时 (秒)
        solution_quality: 解质量 (0-1)
        satellite_utilization: 卫星利用率 (0-1)
        scheduled_task_count: 成功调度任务数
        unscheduled_task_count: 未调度任务数
    """
    demand_satisfaction_rate: float = 0.0
    makespan: float = 0.0
    avg_revisit_time: float = 0.0
    data_delivery_time: float = 0.0
    computation_time: float = 0.0
    solution_quality: float = 0.0
    satellite_utilization: float = 0.0
    scheduled_task_count: int = 0
    unscheduled_task_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'demand_satisfaction_rate': self.demand_satisfaction_rate,
            'makespan_hours': self.makespan / 3600,
            'avg_revisit_time_hours': self.avg_revisit_time / 3600,
            'computation_time_seconds': self.computation_time,
            'solution_quality': self.solution_quality,
            'satellite_utilization': self.satellite_utilization,
            'scheduled_task_count': self.scheduled_task_count,
            'unscheduled_task_count': self.unscheduled_task_count,
        }

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"PerformanceMetrics(\n"
            f"  需求满足率: {self.demand_satisfaction_rate:.2%}\n"
            f"  总完成时间: {self.makespan/3600:.2f} 小时\n"
            f"  算法求解用时: {self.computation_time:.2f} 秒\n"
            f"  解质量: {self.solution_quality:.4f}\n"
            f"  卫星利用率: {self.satellite_utilization:.2%}\n"
            f"  成功调度任务: {self.scheduled_task_count}\n"
            f"  未调度任务: {self.unscheduled_task_count}\n"
            f")"
        )


class MetricsCalculator:
    """性能指标计算器"""

    def __init__(self, mission: Any):
        """
        初始化

        Args:
            mission: 任务场景
        """
        self.mission = mission
        self.total_tasks = len(mission.targets)
        self.duration = (mission.end_time - mission.start_time).total_seconds()

    def calculate_all(self, schedule_result: Any) -> PerformanceMetrics:
        """
        计算所有性能指标

        Args:
            schedule_result: 调度结果

        Returns:
            PerformanceMetrics: 性能指标
        """
        metrics = PerformanceMetrics()

        # 基础指标
        metrics.scheduled_task_count = len(schedule_result.scheduled_tasks)
        metrics.unscheduled_task_count = len(schedule_result.unscheduled_tasks)
        metrics.computation_time = schedule_result.computation_time
        metrics.makespan = schedule_result.makespan

        # 需求满足率
        metrics.demand_satisfaction_rate = metrics.scheduled_task_count / max(self.total_tasks, 1)

        # 卫星利用率
        metrics.satellite_utilization = self._calculate_satellite_utilization(
            schedule_result.scheduled_tasks
        )

        # 解质量
        metrics.solution_quality = self._calculate_solution_quality(
            schedule_result, metrics.demand_satisfaction_rate
        )

        return metrics

    def _calculate_satellite_utilization(self, scheduled_tasks: List[Any]) -> float:
        """计算卫星利用率"""
        if not scheduled_tasks or not self.mission.satellites:
            return 0.0

        sat_work_time = {sat.id: 0.0 for sat in self.mission.satellites}

        for task in scheduled_tasks:
            duration = (task.imaging_end - task.imaging_start).total_seconds()
            sat_work_time[task.satellite_id] = sat_work_time.get(task.satellite_id, 0) + duration

        total_work_time = sum(sat_work_time.values())
        total_available_time = len(self.mission.satellites) * self.duration

        return total_work_time / total_available_time if total_available_time > 0 else 0.0

    def _calculate_solution_quality(self, schedule_result: Any, dsr: float) -> float:
        """计算解质量"""
        time_efficiency = 1.0 - (schedule_result.makespan / max(self.duration, 1))
        time_efficiency = max(0.0, time_efficiency)
        quality = 0.6 * dsr + 0.4 * time_efficiency
        return quality
