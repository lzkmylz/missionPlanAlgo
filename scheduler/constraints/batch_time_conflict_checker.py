"""
批量时间冲突检查器

提供批量时间冲突检查功能，使用Numba JIT加速。
与现有代码保持API兼容，可无缝替换使用。
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from .batch_time_conflict_calculator import (
    BatchTimeConflictCalculator,
    BatchTimeConflictCandidate,
    BatchTimeConflictResult,
    BatchTimeConflictData
)

logger = logging.getLogger(__name__)


class BatchTimeConflictChecker:
    """批量时间冲突检查器

    使用批量计算优化时间冲突检查，特别适用于需要检查大量候选任务的场景。

    使用方法:
        # 创建检查器
        checker = BatchTimeConflictChecker()

        # 单任务检查（向后兼容）
        has_conflict = checker.check_time_conflict(
            sat_id='SAT-001',
            start=datetime(...),
            end=datetime(...),
            scheduled_tasks=existing_tasks
        )

        # 批量检查（优化性能）
        batch_results = checker.check_time_conflict_batch(
            candidates=candidate_list,
            existing_tasks=scheduled_tasks
        )
    """

    def __init__(self):
        """初始化批量时间冲突检查器"""
        self._batch_calculator = BatchTimeConflictCalculator()

        # 性能统计
        self._batch_stats = {
            'batch_calls': 0,
            'total_candidates': 0,
            'avg_batch_size': 0.0
        }

        logger.info("BatchTimeConflictChecker initialized with batch optimization")

    def check_time_conflict_batch(
        self,
        candidates: List[BatchTimeConflictCandidate],
        existing_tasks: List[Dict[str, Any]],
        sat_id_to_idx: Optional[Dict[str, int]] = None
    ) -> List[BatchTimeConflictResult]:
        """批量检查时间冲突

        这是优化的批量版本，使用Numba加速计算。

        Args:
            candidates: 候选任务列表
            existing_tasks: 已调度任务列表
            sat_id_to_idx: 卫星ID到索引的映射（可选，自动创建）

        Returns:
            BatchTimeConflictResult列表，与输入候选顺序一致
        """
        if not candidates:
            return []

        # 更新统计
        self._batch_stats['batch_calls'] += 1
        self._batch_stats['total_candidates'] += len(candidates)
        self._batch_stats['avg_batch_size'] = (
            self._batch_stats['total_candidates'] / self._batch_stats['batch_calls']
        )

        # 准备批量数据
        data = self._batch_calculator.prepare_batch_data(
            candidates, existing_tasks, sat_id_to_idx
        )

        # 执行批量计算
        results = self._batch_calculator.compute_batch(data)

        return results

    def check_time_conflict(
        self,
        sat_id: str,
        start: datetime,
        end: datetime,
        scheduled_tasks: List[Dict[str, Any]],
    ) -> bool:
        """单任务时间冲突检查（向后兼容）

        内部调用批量版本以保持一致性。

        Args:
            sat_id: 卫星ID
            start: 提议的开始时间
            end: 提议的结束时间
            scheduled_tasks: 已调度任务列表

        Returns:
            True 如果有冲突，False 如果没有冲突
        """
        # 创建单候选列表
        candidate = BatchTimeConflictCandidate(
            sat_id=sat_id,
            window_start=start,
            window_end=end
        )

        # 调用批量版本
        results = self.check_time_conflict_batch([candidate], scheduled_tasks)

        if results:
            return not results[0].feasible  # feasible=False 表示有冲突
        return True  # 默认假设有冲突（安全优先）

    def check_any_conflict(
        self,
        candidates: List[BatchTimeConflictCandidate],
        existing_tasks: List[Dict[str, Any]]
    ) -> bool:
        """快速检查是否有任何冲突

        用于快速筛选，一旦发现任何冲突立即返回True。

        Args:
            candidates: 候选任务列表
            existing_tasks: 已调度任务列表

        Returns:
            True 如果有任何冲突，False 如果全部无冲突
        """
        results = self.check_time_conflict_batch(candidates, existing_tasks)
        return any(not r.feasible for r in results)

    def get_batch_stats(self) -> Dict[str, Any]:
        """获取批量计算统计信息

        Returns:
            统计字典
        """
        return {
            **self._batch_stats,
            'use_numba': self._batch_calculator.use_numba
        }

    def reset_batch_stats(self):
        """重置批量计算统计"""
        self._batch_stats = {
            'batch_calls': 0,
            'total_candidates': 0,
            'avg_batch_size': 0.0
        }
