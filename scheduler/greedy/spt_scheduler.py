"""
SPT调度器（最短处理时间优先）- 高性能版本

特性:
- 按处理时间排序任务
- 完整约束检查优化（批量+缓存）
- 与GreedyScheduler相同的性能水平

实现设计文档第8章设计：
- SPT（Shortest Processing Time）启发式调度
- 按处理时间排序任务
- 处理时间相同的按优先级排序
"""

from typing import List, Any, Optional, Tuple

from .heuristic_scheduler import HeuristicScheduler, SlewFeasibilityResult
from scheduler.constraints import UnifiedBatchResult


class SPTScheduler(HeuristicScheduler):
    """
    SPT（最短处理时间优先）调度器 - 高性能版本

    调度策略：
    1. 计算每个任务的预计处理时间
    2. 按处理时间升序排序（短的优先）
    3. 处理时间相同的，按优先级降序排序
    4. 依次尝试为每个任务分配最早的可用卫星-窗口组合

    特点：
    - 最小化平均流程时间（minimize average flow time）
    - 适合需要快速周转的任务场景
    - 使用批量约束检查和姿态预计算缓存，性能与GreedyScheduler相当

    配置参数:
        - consider_power: 是否考虑电量约束（默认True）
        - consider_storage: 是否考虑存储约束（默认True）
        - enable_clustering: 是否启用聚类（默认False）
        - enable_attitude_precache: 是否启用姿态预计算缓存（默认True）
        - enable_batch_constraint_check: 是否启用批量约束检查（默认True）
    """

    def __init__(self, config: dict = None):
        """
        初始化SPT调度器

        Args:
            config: 配置参数
        """
        super().__init__("SPT", "processing_time", config)
        # 兼容性属性
        self.allow_tardiness = self.config.get('allow_tardiness', False)

    def get_parameters(self) -> dict:
        """返回算法可调参数"""
        params = super().get_parameters()
        params['allow_tardiness'] = self.allow_tardiness
        return params

    def _sort_tasks_by_processing_time(self, tasks: List[Any]) -> List[Any]:
        """
        向后兼容的方法名 - 调用 _sort_tasks

        Args:
            tasks: 任务列表

        Returns:
            排序后的任务列表
        """
        return self._sort_tasks(tasks)

    def _sort_tasks(self, tasks: List[Any]) -> List[Any]:
        """
        按处理时间排序任务（SPT规则）

        排序规则：
        1. 按estimated_processing_time升序（短的优先）
        2. 处理时间相同的，按priority降序（优先级高的优先）

        Args:
            tasks: 任务列表

        Returns:
            List: 排序后的任务列表
        """
        def spt_key(task):
            # 估计处理时间（使用默认成像时长作为代理）
            processing_time = self._estimate_processing_time(task)

            # 优先级作为次键（越高越好，所以取负）
            priority = getattr(task, 'priority', 5)

            return (processing_time, -priority)

        return sorted(tasks, key=spt_key)

    def _estimate_processing_time(self, task: Any) -> float:
        """
        估计任务处理时间

        使用成像时间计算器估算任务所需时间。
        根据目标类型（点目标 vs 区域目标）估算不同的处理时间。

        Args:
            task: 任务对象

        Returns:
            float: 预估处理时间（秒）
        """
        from core.models import TargetType

        # 获取目标类型
        target_type = getattr(task, 'target_type', None)

        # 根据目标类型返回不同的估算时间
        # 点目标处理时间短，区域目标处理时间长
        if target_type == TargetType.POINT:
            return 5.0  # 点目标: 5秒
        elif target_type == TargetType.AREA:
            return 30.0  # 区域目标: 30秒

        # 获取任务的成像模式
        imaging_mode = getattr(task, 'preferred_mode', None)

        # 使用成像时间计算器计算
        try:
            duration = self._imaging_calculator.calculate(task, imaging_mode)
            return duration
        except Exception:
            # 回退到默认值
            return self._imaging_calculator.default_duration

    def _select_best_assignment(
        self,
        candidates: List[Tuple],
        results: List[UnifiedBatchResult]
    ) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """
        选择最佳任务分配 - SPT策略

        SPT策略: 选择最早开始的分配（以最小化流程时间）

        Args:
            candidates: 候选列表 [(candidate_data, slew_result), ...]
            results: 统一批量约束检查结果

        Returns:
            最佳分配或None
        """
        best = None
        best_start = None

        for (candidate, slew_result), unified_result in zip(candidates, results):
            if not unified_result.feasible:
                continue

            sat, window, imaging_mode, _, _, _ = candidate
            actual_start = slew_result.actual_start

            # SPT策略: 优先选择最早的分配（最小化流程时间）
            if best_start is None or actual_start < best_start:
                best_start = actual_start
                best = (sat.id, window, imaging_mode, slew_result)

        return best
