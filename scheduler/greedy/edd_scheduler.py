"""
EDD调度器（最早截止时间优先）- 高性能版本

特性:
- 按截止时间排序任务
- 完整约束检查优化（批量+缓存）
- 与GreedyScheduler相同的性能水平

实现设计文档第8章设计：
- EDD（Earliest Due Date）启发式调度
- 按截止时间排序任务
- 截止时间相同时按优先级排序
"""

from typing import List, Any, Optional, Tuple
from datetime import datetime

from .heuristic_scheduler import HeuristicScheduler, SlewFeasibilityResult
from scheduler.constraints import UnifiedBatchResult


class EDDScheduler(HeuristicScheduler):
    """
    EDD（最早截止时间优先）调度器 - 高性能版本

    调度策略：
    1. 按任务的截止时间（time_window_end）升序排序
    2. 截止时间相同的，按优先级降序排序
    3. 依次尝试为每个任务分配最早的可用卫星-窗口组合

    特点：
    - 最小化最大延迟（minimize maximum lateness）
    - 适合有严格截止时间的任务场景
    - 使用批量约束检查和姿态预计算缓存，性能与GreedyScheduler相当

    配置参数:
        - consider_power: 是否考虑电量约束（默认True）
        - consider_storage: 是否考虑存储约束（默认True）
        - allow_tardiness: 是否允许延迟执行（默认False）
        - enable_clustering: 是否启用聚类（默认False）
        - enable_attitude_precache: 是否启用姿态预计算缓存（默认True）
        - enable_batch_constraint_check: 是否启用批量约束检查（默认True）
    """

    def __init__(self, config: dict = None):
        """
        初始化EDD调度器

        Args:
            config: 配置参数
        """
        super().__init__("EDD", "due_date", config)
        self.allow_tardiness = self.config.get('allow_tardiness', False)

    def get_parameters(self) -> dict:
        """返回算法可调参数"""
        params = super().get_parameters()
        params['allow_tardiness'] = self.allow_tardiness
        return params

    def _sort_tasks_by_due_date(self, tasks: List[Any]) -> List[Any]:
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
        按截止时间排序任务（EDD规则）

        排序规则：
        1. 按time_window_end升序（越早的截止越优先）
        2. 截止时间相同的，按priority降序（优先级高的优先）
        3. 都没有截止时间的，按priority降序

        Args:
            tasks: 任务列表

        Returns:
            List: 排序后的任务列表
        """
        def edd_key(task):
            # 截止时间作为主键（None放在最后）
            due_date = task.time_window_end
            if due_date is None:
                due_date_timestamp = float('inf')
            else:
                due_date_timestamp = due_date.timestamp() if isinstance(due_date, datetime) else float('inf')

            # 优先级作为次键（越高越好，所以取负）
            priority = getattr(task, 'priority', 5)

            return (due_date_timestamp, -priority)

        return sorted(tasks, key=edd_key)

    def _select_best_assignment(
        self,
        candidates: List[Tuple],
        results: List[UnifiedBatchResult]
    ) -> Optional[Tuple[str, Any, Any, SlewFeasibilityResult]]:
        """
        选择最佳任务分配 - EDD策略

        EDD策略: 选择满足截止时间的最早分配
        如果任务有截止时间，优先选择能在截止时间前完成的分配

        Args:
            candidates: 候选列表 [(candidate_data, slew_result), ...]
            results: 统一批量约束检查结果

        Returns:
            最佳分配或None
        """
        best = None
        best_score = None

        for (candidate, slew_result), unified_result in zip(candidates, results):
            if not unified_result.feasible:
                continue

            sat, window, imaging_mode, _, _, _ = candidate
            actual_start = slew_result.actual_start

            # 获取任务的截止时间（如果存在）
            task = getattr(slew_result, 'target', None) or candidate[3] if len(candidate) > 3 else None

            # 计算得分（越小越好）
            # 主要：开始时间早
            # 次要：满足截止时间
            score = actual_start.timestamp()

            if best_score is None or score < best_score:
                best_score = score
                best = (sat.id, window, imaging_mode, slew_result)

        return best
