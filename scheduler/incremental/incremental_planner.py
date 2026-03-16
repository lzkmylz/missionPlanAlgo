"""
增量规划器主入口

根据请求的策略类型，自动选择并执行相应的增量规划策略
"""

from typing import List, Dict, Any, Optional
import logging

from .base_incremental import (
    BaseIncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalPlanResult,
    IncrementalStrategyType
)
from .incremental_state import IncrementalState
from .resource_reclaimer import ResourceReclaimer

logger = logging.getLogger(__name__)


class IncrementalPlanner:
    """
    增量规划器主入口类

    使用示例：
        planner = IncrementalPlanner()
        result = planner.plan(request)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化增量规划器

        Args:
            config: 全局配置字典
        """
        self.config = config or {}
        self._strategy_cache: Dict[IncrementalStrategyType, BaseIncrementalPlanner] = {}

    def plan(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """
        执行增量规划

        Args:
            request: 增量规划请求

        Returns:
            IncrementalPlanResult: 规划结果
        """
        # 获取或创建策略实例
        strategy = self._get_strategy(request.strategy)

        # 执行规划
        logger.info(f"Starting incremental planning with {request.strategy.value} strategy")
        logger.debug(f"  New targets: {len(request.new_targets)}")
        logger.debug(f"  Existing tasks: {len(request.existing_schedule.scheduled_tasks)}")

        result = strategy.plan(request)

        logger.info(f"Incremental planning completed: "
                   f"{len(result.new_tasks)} new, {len(result.preempted_tasks)} preempted, "
                   f"{len(result.failed_targets)} failed")

        return result

    def _get_strategy(self, strategy_type: IncrementalStrategyType) -> BaseIncrementalPlanner:
        """获取或创建策略实例"""
        if strategy_type not in self._strategy_cache:
            if strategy_type == IncrementalStrategyType.CONSERVATIVE:
                from .strategies.conservative_strategy import ConservativeStrategy
                self._strategy_cache[strategy_type] = ConservativeStrategy(self.config)
            elif strategy_type == IncrementalStrategyType.AGGRESSIVE:
                from .strategies.aggressive_strategy import AggressiveStrategy
                self._strategy_cache[strategy_type] = AggressiveStrategy(self.config)
            elif strategy_type == IncrementalStrategyType.HYBRID:
                from .strategies.hybrid_strategy import HybridStrategy
                self._strategy_cache[strategy_type] = HybridStrategy(self.config)
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")

        return self._strategy_cache[strategy_type]

    def analyze_resources(self, schedule_result: Any, mission: Any) -> Dict[str, Any]:
        """
        分析现有调度结果的资源情况

        Args:
            schedule_result: 现有调度结果
            mission: 任务场景

        Returns:
            Dict: 资源分析报告
        """
        # 创建状态管理器
        state = IncrementalState(mission)
        state.load_from_schedule(schedule_result)

        # 创建资源回收计算器
        reclaimer = ResourceReclaimer(state)

        return reclaimer.generate_resource_report()

    def estimate_capacity(self, schedule_result: Any, mission: Any,
                         avg_task_duration: float = 300.0) -> Dict[str, int]:
        """
        估计各卫星还可容纳的任务数量

        Args:
            schedule_result: 现有调度结果
            mission: 任务场景
            avg_task_duration: 平均任务时长

        Returns:
            Dict[str, int]: 各卫星可容纳任务数
        """
        state = IncrementalState(mission)
        state.load_from_schedule(schedule_result)
        reclaimer = ResourceReclaimer(state)

        return {
            sat_id: reclaimer.estimate_task_capacity(sat_id, avg_task_duration)
            for sat_id in state.get_all_satellite_ids()
        }
