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
        """计算卫星利用率

        基于资源约束（电量、固存）和成像时长计算实际利用率：
        - 电量利用率 = 实际消耗电量 / 可用电量
        - 固存利用率 = 实际占用固存 / 固存容量
        - 时间利用率 = 实际成像时长 / (轨道圈数 × 每圈最大成像时长)
        最终取三个指标的平均值
        """
        if not scheduled_tasks or not self.mission.satellites:
            return 0.0

        # 按卫星分组统计
        sat_stats = {}
        for sat in self.mission.satellites:
            sat_stats[sat.id] = {
                'power_capacity': getattr(sat.capabilities, 'power_capacity', 2800.0),
                'storage_capacity': getattr(sat.capabilities, 'storage_capacity', 128.0),
                'orbit_period': self._calculate_orbit_period(sat),
                'total_imaging_time': 0.0,
                'power_consumed': 0.0,
                'storage_used': 0.0,
            }

        # 统计每颗卫星的实际消耗
        for task in scheduled_tasks:
            sat_id = task.satellite_id
            if sat_id not in sat_stats:
                continue

            # 成像时长
            imaging_duration = (task.imaging_end - task.imaging_start).total_seconds()
            sat_stats[sat_id]['total_imaging_time'] += imaging_duration

            # 电量消耗 (从task记录中获取)
            power_before = getattr(task, 'power_before', 0)
            power_after = getattr(task, 'power_after', 0)
            power_consumed = power_before - power_after
            if power_consumed > 0:
                sat_stats[sat_id]['power_consumed'] += power_consumed

            # 固存占用 (从task记录中获取)
            storage_after = getattr(task, 'storage_after', 0)
            if storage_after > sat_stats[sat_id]['storage_used']:
                sat_stats[sat_id]['storage_used'] = storage_after

        # 计算每颗卫星的利用率
        utilizations = []
        for sat_id, stats in sat_stats.items():
            # 1. 电量利用率
            power_util = stats['power_consumed'] / stats['power_capacity'] if stats['power_capacity'] > 0 else 0

            # 2. 固存利用率
            storage_util = stats['storage_used'] / stats['storage_capacity'] if stats['storage_capacity'] > 0 else 0

            # 3. 时间利用率（基于每圈最大成像时长，假设每圈最多成像5分钟）
            max_imaging_per_orbit = 300  # 5分钟 = 300秒
            num_orbits = self.duration / stats['orbit_period'] if stats['orbit_period'] > 0 else 15
            max_possible_imaging_time = num_orbits * max_imaging_per_orbit
            time_util = stats['total_imaging_time'] / max_possible_imaging_time if max_possible_imaging_time > 0 else 0

            # 综合利用率（取三个指标的平均值）
            sat_util = (power_util + storage_util + time_util) / 3.0
            utilizations.append(min(sat_util, 1.0))  # 上限100%

        return sum(utilizations) / len(utilizations) if utilizations else 0.0

    def _calculate_orbit_period(self, satellite: Any) -> float:
        """计算卫星轨道周期（秒）

        使用开普勒第三定律计算轨道周期。
        对于LEO卫星，典型周期约90-100分钟。
        """
        import math

        # 地球引力常数 (m^3/s^2)
        GM = 3.986004418e14

        # 获取轨道半长轴
        semi_major_axis = None
        if hasattr(satellite, 'orbit'):
            orbit = satellite.orbit
            if hasattr(orbit, 'semi_major_axis') and orbit.semi_major_axis:
                semi_major_axis = orbit.semi_major_axis
            elif hasattr(orbit, 'altitude') and orbit.altitude:
                # 地球半径 + 高度
                semi_major_axis = 6371000.0 + orbit.altitude

        if semi_major_axis is None:
            # 默认值：500km高度的LEO
            semi_major_axis = 6871000.0  # 6371000 + 500000

        # 开普勒第三定律：T = 2π × √(a³/GM)
        try:
            period = 2 * math.pi * math.sqrt(semi_major_axis ** 3 / GM)
            return period
        except (ValueError, OverflowError):
            # 默认LEO周期约90分钟
            return 5400.0

    def _calculate_solution_quality(self, schedule_result: Any, dsr: float) -> float:
        """计算解质量"""
        time_efficiency = 1.0 - (schedule_result.makespan / max(self.duration, 1))
        time_efficiency = max(0.0, time_efficiency)
        quality = 0.6 * dsr + 0.4 * time_efficiency
        return quality
