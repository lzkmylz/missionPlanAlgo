"""
批量SAA约束检查器

继承 SAAConstraintChecker 的接口，但使用批量计算优化。
与基类保持API兼容，可无缝替换使用。
"""

import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from core.models.mission import Mission
from core.models.satellite import Satellite
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType

from .saa_constraint_checker import (
    SAAConstraintChecker, SAAFeasibilityResult
)
from .batch_saa_calculator import (
    BatchSAACandidate, BatchSAAResult, BatchSAACalculator, BatchSAAData
)

logger = logging.getLogger(__name__)


class BatchSAAConstraintChecker(SAAConstraintChecker):
    """批量SAA约束检查器

    继承 SAAConstraintChecker 以复用配置提取和基础逻辑，
    但使用批量计算优化核心性能瓶颈。

    使用方法:
        # 创建检查器
        checker = BatchSAAConstraintChecker(mission)

        # 单窗口检查（向后兼容）
        result = checker.check_window_feasibility(...)

        # 批量检查（优化性能）
        batch_results = checker.check_window_feasibility_batch(candidates)
    """

    def __init__(
        self,
        mission: Mission,
        attitude_calculator: Optional[AttitudeCalculator] = None,
        saa_model=None,
    ):
        """初始化批量SAA约束检查器

        Args:
            mission: 任务对象
            attitude_calculator: 姿态计算器（可选）
            saa_model: SAA边界模型（可选）
        """
        # 调用父类初始化
        super().__init__(
            mission=mission,
            attitude_calculator=attitude_calculator,
            saa_model=saa_model
        )

        # 创建批量计算器
        self._batch_calculator = BatchSAACalculator()

        # 性能统计
        self._batch_stats = {
            'batch_calls': 0,
            'total_candidates': 0,
            'avg_batch_size': 0.0
        }

        logger.info("BatchSAAConstraintChecker initialized with batch optimization")

    def check_window_feasibility_batch(
        self,
        candidates: List[BatchSAACandidate],
    ) -> List[SAAFeasibilityResult]:
        """批量检查SAA约束

        这是优化的批量版本，使用Numba加速计算。

        Args:
            candidates: SAA检查候选列表

        Returns:
            SAAFeasibilityResult列表，与输入候选顺序一致
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
            candidates,
            position_cache=self._precomputed_position_cache,
            orbit_propagator=None
        )

        # 执行批量计算
        batch_results = self._batch_calculator.compute_batch(data)

        # 转换结果格式
        results = []
        for i, (cand, batch_result) in enumerate(zip(candidates, batch_results)):
            result = self._convert_batch_result(cand, batch_result)
            results.append(result)

        return results

    def check_window_feasibility(
        self,
        satellite_id: str,
        window_start: datetime,
        window_end: datetime,
        min_samples: int = 3,
        sample_interval: timedelta = timedelta(seconds=60),
    ) -> SAAFeasibilityResult:
        """单窗口检查（向后兼容）

        内部调用批量版本以保持一致性。

        Args:
            satellite_id: 卫星ID
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            min_samples: 最小采样点数
            sample_interval: 采样间隔

        Returns:
            SAAFeasibilityResult
        """
        # 创建单候选列表，调用批量版本
        candidate = BatchSAACandidate(
            sat_id=satellite_id,
            window_start=window_start,
            window_end=window_end,
            sample_interval=sample_interval.total_seconds()
        )

        # 调用批量版本
        results = self.check_window_feasibility_batch([candidate])

        return results[0] if results else self._error_result(
            "Batch computation failed", window_start
        )

    def _error_result(self, reason: str, window_start: datetime) -> SAAFeasibilityResult:
        """创建错误结果"""
        return SAAFeasibilityResult(
            feasible=False,
            violation_count=1,
            violation_times=[window_start],
            sample_count=1,
            max_separation=float('inf')
        )

    def _convert_batch_result(
        self,
        candidate: BatchSAACandidate,
        batch_result: BatchSAAResult
    ) -> SAAFeasibilityResult:
        """将批量结果转换为SAAFeasibilityResult格式

        Args:
            candidate: 原始候选
            batch_result: 批量计算结果

        Returns:
            SAAFeasibilityResult
        """
        return SAAFeasibilityResult(
            feasible=batch_result.feasible,
            violation_count=batch_result.violation_count,
            violation_times=batch_result.violation_times or [],
            sample_count=batch_result.sample_count,
            max_separation=batch_result.max_separation
        )

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
