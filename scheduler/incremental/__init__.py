"""
增量任务规划模块

支持在现有调度结果基础上，对新目标进行增量规划。
提供两种策略：
- 保守策略：仅使用剩余资源
- 激进策略：允许抢占已有任务资源

主要组件：
- IncrementalPlanner: 主入口
- ConservativeStrategy: 保守策略实现
- AggressiveStrategy: 激进策略实现
- HybridStrategy: 混合策略实现
"""

from .base_incremental import (
    BaseIncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalPlanResult,
    IncrementalStrategyType,
    PriorityRule,
    PreemptionRule,
    PreemptionCandidate,
    ResourceDelta,
)
from .incremental_planner import IncrementalPlanner
from .strategies.conservative_strategy import ConservativeStrategy
from .strategies.aggressive_strategy import AggressiveStrategy
from .strategies.hybrid_strategy import HybridStrategy
from .incremental_state import IncrementalState, ResourceWindow
from .resource_reclaimer import ResourceReclaimer, ResourceProfile

__all__ = [
    'BaseIncrementalPlanner',
    'IncrementalPlanner',
    'IncrementalPlanRequest',
    'IncrementalPlanResult',
    'IncrementalStrategyType',
    'PriorityRule',
    'PreemptionRule',
    'PreemptionCandidate',
    'ResourceDelta',
    'ConservativeStrategy',
    'AggressiveStrategy',
    'HybridStrategy',
    'IncrementalState',
    'ResourceWindow',
    'ResourceReclaimer',
    'ResourceProfile',
]