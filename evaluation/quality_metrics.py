"""
调度结果质量指标

提供详细的调度结果质量分析功能。
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

from core.quality.window_quality import WindowQualityCalculator, WindowQualityScore

if TYPE_CHECKING:
    from scheduler.base_scheduler import ScheduleResult, ScheduledTask
    from core.models.mission import Mission
    from core.models.satellite import Satellite
    from core.orbit.visibility.base import VisibilityWindow

logger = logging.getLogger(__name__)


@dataclass
class WindowQualityAnalysis:
    """窗口质量分析结果"""
    avg_overall_quality: float = 0.0
    quality_by_dimension: Dict[str, float] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    high_quality_tasks: List[str] = field(default_factory=list)
    low_quality_tasks: List[str] = field(default_factory=list)
    quality_variance: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'avg_overall_quality': round(self.avg_overall_quality, 4),
            'quality_by_dimension': {k: round(v, 4) for k, v in self.quality_by_dimension.items()},
            'quality_distribution': self.quality_distribution,
            'high_quality_count': len(self.high_quality_tasks),
            'low_quality_count': len(self.low_quality_tasks),
            'quality_variance': round(self.quality_variance, 4),
            'recommendations': self.recommendations,
        }


@dataclass
class TemporalAnalysis:
    """时间分布分析"""
    avg_task_interval_minutes: float = 0.0
    task_distribution_by_hour: Dict[int, int] = field(default_factory=dict)
    peak_hours: List[int] = field(default_factory=list)
    idle_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    load_balance_score: float = 0.0  # 0-1, 越接近1越均匀


@dataclass
class ResourceEfficiencyAnalysis:
    """资源使用效率分析"""
    power_efficiency: float = 0.0  # 电量使用效率
    storage_efficiency: float = 0.0  # 存储使用效率
    slew_efficiency: float = 0.0  # 姿态机动效率
    total_power_consumed: float = 0.0
    total_storage_used: float = 0.0
    avg_slew_angle: float = 0.0


class ScheduleQualityAnalyzer:
    """
    调度结果质量分析器

    提供调度结果的全面质量分析。
    """

    def __init__(
        self,
        schedule_result: 'ScheduleResult',
        mission: 'Mission',
    ):
        """
        初始化质量分析器

        Args:
            schedule_result: 调度结果
            mission: 任务
        """
        self.result = schedule_result
        self.mission = mission
        self.quality_calculator = WindowQualityCalculator()

    def analyze_window_quality(self) -> WindowQualityAnalysis:
        """
        分析已调度任务的窗口质量

        Returns:
            WindowQualityAnalysis
        """
        analysis = WindowQualityAnalysis()

        # 获取已调度任务
        imaging_result = getattr(self.result, 'imaging_result', None)
        if not imaging_result:
            return analysis

        scheduled_tasks = getattr(imaging_result, 'scheduled_tasks', [])
        if not scheduled_tasks:
            return analysis

        # 收集质量评分
        overall_scores = []
        dimension_scores = defaultdict(list)

        for task in scheduled_tasks:
            task_id = getattr(task, 'task_id', str(id(task)))
            quality = getattr(task, 'quality_score', 0.5)

            overall_scores.append(quality)

            # 分类任务质量
            if quality >= 0.7:
                analysis.high_quality_tasks.append(task_id)
            elif quality < 0.3:
                analysis.low_quality_tasks.append(task_id)

            # 详细维度评分（如果有）
            detailed_scores = getattr(task, 'detailed_scores', None)
            if detailed_scores:
                for dim, score in detailed_scores.items():
                    dimension_scores[dim].append(score)

        # 计算平均值
        if overall_scores:
            analysis.avg_overall_quality = sum(overall_scores) / len(overall_scores)

            # 质量分布
            for score in overall_scores:
                if score >= 0.7:
                    analysis.quality_distribution['high'] = analysis.quality_distribution.get('high', 0) + 1
                elif score >= 0.4:
                    analysis.quality_distribution['medium'] = analysis.quality_distribution.get('medium', 0) + 1
                elif score >= 0.3:
                    analysis.quality_distribution['low'] = analysis.quality_distribution.get('low', 0) + 1
                else:
                    analysis.quality_distribution['unacceptable'] = analysis.quality_distribution.get('unacceptable', 0) + 1

            # 计算方差
            if len(overall_scores) > 1:
                mean = analysis.avg_overall_quality
                variance = sum((x - mean) ** 2 for x in overall_scores) / len(overall_scores)
                analysis.quality_variance = variance

        # 各维度平均评分
        for dim, scores in dimension_scores.items():
            analysis.quality_by_dimension[dim] = sum(scores) / len(scores)

        # 生成建议
        analysis.recommendations = self._generate_quality_recommendations(analysis)

        return analysis

    def _generate_quality_recommendations(self, analysis: WindowQualityAnalysis) -> List[str]:
        """生成质量改进建议"""
        recommendations = []

        if analysis.avg_overall_quality < 0.5:
            recommendations.append(
                "整体窗口质量较低，建议：1) 增加卫星数量；2) 调整任务时间窗；3) 降低姿态约束要求"
            )

        if analysis.quality_distribution.get('unacceptable', 0) > len(analysis.quality_distribution) * 0.1:
            recommendations.append(
                "不可接受质量任务占比超过10%，建议检查约束设置是否合理"
            )

        if analysis.quality_variance > 0.1:
            recommendations.append(
                "任务质量差异较大，建议优化任务分配策略以获得更均衡的质量分布"
            )

        # 检查各维度
        for dim, score in analysis.quality_by_dimension.items():
            if score < 0.5:
                recommendations.append(
                    f"{dim}维度评分较低({score:.2f})，建议针对该维度优化调度策略"
                )

        if not recommendations:
            recommendations.append("窗口质量良好，无需特殊优化")

        return recommendations

    def analyze_temporal_distribution(self) -> TemporalAnalysis:
        """
        分析任务时间分布质量

        Returns:
            TemporalAnalysis
        """
        analysis = TemporalAnalysis()

        imaging_result = getattr(self.result, 'imaging_result', None)
        if not imaging_result:
            return analysis

        scheduled_tasks = getattr(imaging_result, 'scheduled_tasks', [])
        if len(scheduled_tasks) < 2:
            return analysis

        # 按时间排序
        sorted_tasks = sorted(
            scheduled_tasks,
            key=lambda t: getattr(t, 'start_time', datetime.min)
        )

        # 计算任务间隔
        intervals = []
        for i in range(1, len(sorted_tasks)):
            prev_end = getattr(sorted_tasks[i-1], 'end_time', sorted_tasks[i-1].start_time)
            curr_start = getattr(sorted_tasks[i], 'start_time', datetime.min)
            interval = (curr_start - prev_end).total_seconds() / 60  # 分钟
            intervals.append(interval)

        if intervals:
            analysis.avg_task_interval_minutes = sum(intervals) / len(intervals)

        # 按小时统计
        hour_counts = defaultdict(int)
        for task in sorted_tasks:
            start_time = getattr(task, 'start_time', None)
            if start_time:
                hour_counts[start_time.hour] += 1

        analysis.task_distribution_by_hour = dict(hour_counts)

        # 找出高峰时段（超过平均值1.5倍）
        if hour_counts:
            avg_count = sum(hour_counts.values()) / len(hour_counts)
            analysis.peak_hours = [
                h for h, c in hour_counts.items()
                if c > avg_count * 1.5
            ]

        # 计算负载均衡分数
        if hour_counts:
            counts = list(hour_counts.values())
            if len(counts) > 1:
                mean = sum(counts) / len(counts)
                variance = sum((x - mean) ** 2 for x in counts) / len(counts)
                std_dev = variance ** 0.5
                # 变异系数越小越均匀
                cv = std_dev / mean if mean > 0 else 0
                analysis.load_balance_score = max(0, 1 - cv)

        return analysis

    def analyze_resource_efficiency(self) -> ResourceEfficiencyAnalysis:
        """
        分析资源使用效率

        Returns:
            ResourceEfficiencyAnalysis
        """
        analysis = ResourceEfficiencyAnalysis()

        imaging_result = getattr(self.result, 'imaging_result', None)
        if not imaging_result:
            return analysis

        scheduled_tasks = getattr(imaging_result, 'scheduled_tasks', [])

        # 收集资源使用数据
        power_consumed = []
        storage_used = []
        slew_angles = []

        for task in scheduled_tasks:
            # 电量消耗
            power_delta = getattr(task, 'power_delta', None)
            if power_delta is not None:
                power_consumed.append(power_delta)

            # 存储使用
            storage_delta = getattr(task, 'storage_delta', None)
            if storage_delta is not None:
                storage_used.append(storage_delta)

            # 机动角度
            slew_angle = getattr(task, 'slew_angle', None)
            if slew_angle is not None:
                slew_angles.append(slew_angle)

        # 计算效率指标
        if power_consumed:
            analysis.total_power_consumed = sum(power_consumed)
            # 效率 = 实际使用 / 理论最大（简化计算）
            analysis.power_efficiency = 0.8  # 默认中等效率

        if storage_used:
            analysis.total_storage_used = sum(storage_used)
            analysis.storage_efficiency = 0.8

        if slew_angles:
            analysis.avg_slew_angle = sum(slew_angles) / len(slew_angles)
            # 机动效率：小角度机动更高效
            avg_angle = analysis.avg_slew_angle
            analysis.slew_efficiency = max(0, 1 - avg_angle / 90)

        return analysis

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        生成完整质量报告

        Returns:
            包含所有分析结果的字典
        """
        window_analysis = self.analyze_window_quality()
        temporal_analysis = self.analyze_temporal_distribution()
        resource_analysis = self.analyze_resource_efficiency()

        # 获取基础统计
        imaging_result = getattr(self.result, 'imaging_result', None)
        scheduled_count = 0
        unscheduled_count = 0

        if imaging_result:
            scheduled_count = len(getattr(imaging_result, 'scheduled_tasks', []))
            unscheduled_count = len(getattr(imaging_result, 'unscheduled_tasks', []))

        total_count = scheduled_count + unscheduled_count
        success_rate = scheduled_count / total_count if total_count > 0 else 0

        report = {
            'summary': {
                'scheduled_tasks': scheduled_count,
                'unscheduled_tasks': unscheduled_count,
                'total_tasks': total_count,
                'success_rate': round(success_rate, 4),
                'avg_window_quality': round(window_analysis.avg_overall_quality, 4),
            },
            'window_quality': window_analysis.to_dict(),
            'temporal_distribution': {
                'avg_interval_minutes': round(temporal_analysis.avg_task_interval_minutes, 2),
                'peak_hours': temporal_analysis.peak_hours,
                'load_balance_score': round(temporal_analysis.load_balance_score, 4),
                'hourly_distribution': temporal_analysis.task_distribution_by_hour,
            },
            'resource_efficiency': {
                'power_efficiency': round(resource_analysis.power_efficiency, 4),
                'storage_efficiency': round(resource_analysis.storage_efficiency, 4),
                'slew_efficiency': round(resource_analysis.slew_efficiency, 4),
                'total_power_consumed': round(resource_analysis.total_power_consumed, 2),
                'total_storage_used': round(resource_analysis.total_storage_used, 2),
                'avg_slew_angle': round(resource_analysis.avg_slew_angle, 2),
            },
            'recommendations': window_analysis.recommendations,
        }

        return report


class QualityMetricsCollector:
    """
    质量指标收集器

    用于在多次调度运行中收集和聚合质量指标。
    """

    def __init__(self):
        """初始化收集器"""
        self.runs: List[Dict[str, Any]] = []

    def add_run(self, algorithm: str, quality_report: Dict[str, Any]) -> None:
        """
        添加一次运行的质量报告

        Args:
            algorithm: 算法名称
            quality_report: 质量报告
        """
        self.runs.append({
            'algorithm': algorithm,
            'timestamp': datetime.now().isoformat(),
            'report': quality_report,
        })

    def get_average_metrics(self) -> Dict[str, float]:
        """
        获取平均指标

        Returns:
            平均指标字典
        """
        if not self.runs:
            return {}

        avg_quality = []
        success_rates = []

        for run in self.runs:
            report = run['report']
            summary = report.get('summary', {})
            avg_quality.append(summary.get('avg_window_quality', 0))
            success_rates.append(summary.get('success_rate', 0))

        return {
            'avg_quality': sum(avg_quality) / len(avg_quality),
            'avg_success_rate': sum(success_rates) / len(success_rates),
            'run_count': len(self.runs),
        }

    def compare_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        对比不同算法的质量指标

        Returns:
            算法对比结果
        """
        by_algorithm = defaultdict(list)

        for run in self.runs:
            algo = run['algorithm']
            report = run['report']
            by_algorithm[algo].append(report)

        comparison = {}
        for algo, reports in by_algorithm.items():
            qualities = [r['summary']['avg_window_quality'] for r in reports]
            success_rates = [r['summary']['success_rate'] for r in reports]

            comparison[algo] = {
                'avg_quality': sum(qualities) / len(qualities),
                'min_quality': min(qualities),
                'max_quality': max(qualities),
                'quality_std': self._calculate_std(qualities),
                'avg_success_rate': sum(success_rates) / len(success_rates),
                'run_count': len(reports),
            }

        return comparison

    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'run_count': len(self.runs),
            'average_metrics': self.get_average_metrics(),
            'algorithm_comparison': self.compare_algorithms(),
            'runs': self.runs,
        }
