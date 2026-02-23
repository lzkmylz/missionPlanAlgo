"""
失败原因分析器

用于分析调度失败模式并生成改进建议
"""

from typing import List, Dict, Any, Optional
from scheduler.base_scheduler import ScheduleResult, TaskFailure, TaskFailureReason


class FailureAnalyzer:
    """失败原因分析器，用于分析调度失败模式"""

    def __init__(self, results: List[ScheduleResult]):
        """
        初始化分析器

        Args:
            results: 调度结果列表
        """
        self.results = results

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        分析失败模式并生成改进建议

        Returns:
            Dict[str, Any]: 包含以下字段的字典:
                - most_common_reason: 最常见的失败原因（字符串）
                - reason_distribution: 各失败原因的分布统计
                - recommendations: 改进建议列表
        """
        # 收集所有失败
        all_failures: List[TaskFailure] = []
        for result in self.results:
            all_failures.extend(result.unscheduled_tasks.values())

        # 如果没有失败，返回空结果
        if not all_failures:
            return {
                'most_common_reason': None,
                'reason_distribution': {},
                'recommendations': []
            }

        # 统计各原因数量
        reason_counts = {}
        for failure in all_failures:
            reason = failure.failure_reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # 生成改进建议
        recommendations = []
        if reason_counts:
            top_reason = max(reason_counts, key=reason_counts.get)

            if top_reason == 'power_constraint':
                recommendations.append("建议增加卫星电池容量或优化能源调度策略")
            elif top_reason == 'storage_constraint':
                recommendations.append("建议增加星上存储或优化数传调度")
            elif top_reason == 'no_visible_window':
                recommendations.append("建议调整卫星轨道或增加卫星数量")

        return {
            'most_common_reason': max(reason_counts, key=reason_counts.get) if reason_counts else None,
            'reason_distribution': reason_counts,
            'recommendations': recommendations
        }
