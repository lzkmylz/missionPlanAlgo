"""
批量资源约束检查器

提供批量资源约束检查功能，使用Numba JIT加速。
与现有ResourceManager保持API兼容，可无缝替换使用。
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .batch_resource_calculator import (
    BatchResourceCalculator,
    BatchResourceCandidate,
    BatchResourceResult,
    BatchResourceData
)

logger = logging.getLogger(__name__)


class BatchResourceChecker:
    """批量资源约束检查器

    使用批量计算优化资源约束检查，特别适用于需要检查大量候选任务的场景。

    使用方法:
        # 创建检查器
        checker = BatchResourceChecker()

        # 单任务检查（向后兼容）
        feasible = checker.check_resources(
            sat_id='SAT-001',
            power_needed=100.0,
            storage_needed=0.0,
            satellite_states=states
        )

        # 批量检查（优化性能）
        batch_results = checker.check_resources_batch(
            candidates=candidate_list,
            satellite_states=satellite_states
        )
    """

    def __init__(
        self,
        consider_power: bool = True,
        consider_storage: bool = True
    ):
        """初始化批量资源约束检查器

        Args:
            consider_power: 是否检查电量约束
            consider_storage: 是否检查存储约束
        """
        self._batch_calculator = BatchResourceCalculator()
        self.consider_power = consider_power
        self.consider_storage = consider_storage

        # 性能统计
        self._batch_stats = {
            'batch_calls': 0,
            'total_candidates': 0,
            'avg_batch_size': 0.0
        }

        logger.info("BatchResourceChecker initialized with batch optimization")

    def check_resources_batch(
        self,
        candidates: List[BatchResourceCandidate],
        satellite_states: Dict[str, Dict[str, float]],
        sat_id_to_idx: Optional[Dict[str, int]] = None
    ) -> List[BatchResourceResult]:
        """批量检查资源约束

        这是优化的批量版本，使用Numba加速计算。

        Args:
            candidates: 候选任务列表
            satellite_states: 卫星当前资源状态
            sat_id_to_idx: 卫星ID到索引的映射（可选，自动创建）

        Returns:
            BatchResourceResult列表，与输入候选顺序一致
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
            candidates, satellite_states, sat_id_to_idx
        )

        # 执行批量计算
        results = self._batch_calculator.compute_batch(data)

        return results

    def check_resources(
        self,
        sat_id: str,
        satellite_states: Dict[str, Dict[str, float]],
        power_needed: float = 0.0,
        storage_needed: float = 0.0,
        storage_produced: float = 0.0
    ) -> bool:
        """单任务资源约束检查（向后兼容）

        内部调用批量版本以保持一致性。

        Args:
            sat_id: 卫星ID
            satellite_states: 卫星当前资源状态
            power_needed: 所需电量
            storage_needed: 所需存储
            storage_produced: 产生数据量

        Returns:
            True 如果资源充足，False 如果不足
        """
        # 创建单候选列表
        candidate = BatchResourceCandidate(
            sat_id=sat_id,
            power_needed=power_needed,
            storage_needed=storage_needed,
            storage_produced=storage_produced
        )

        # 调用批量版本
        results = self.check_resources_batch([candidate], satellite_states)

        if results:
            return results[0].feasible
        return False  # 默认假设不可行（安全优先）

    def check_all_feasible(
        self,
        candidates: List[BatchResourceCandidate],
        satellite_states: Dict[str, Dict[str, float]]
    ) -> bool:
        """快速检查是否所有候选都可行

        用于快速筛选，一旦发现任何不可行立即返回False。

        Args:
            candidates: 候选任务列表
            satellite_states: 卫星当前资源状态

        Returns:
            True 如果全部可行，False 如果有任何不可行
        """
        results = self.check_resources_batch(candidates, satellite_states)
        return all(r.feasible for r in results)

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
