"""
增量规划基础抽象类

定义增量规划器的统一接口和共享功能
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import logging

# 运行时需要的导入
from ..base_scheduler import ScheduleResult, ScheduledTask, TaskFailure

if TYPE_CHECKING:
    from core.models.target import Target

logger = logging.getLogger(__name__)


class IncrementalStrategyType(Enum):
    """增量规划策略类型"""
    CONSERVATIVE = "conservative"   # 保守策略：仅使用剩余资源
    AGGRESSIVE = "aggressive"       # 激进策略：允许抢占
    HYBRID = "hybrid"               # 混合策略：动态选择


@dataclass
class PreemptionRule:
    """抢占规则配置"""
    min_priority_difference: int = 2      # 最小优先级差才允许抢占
    max_preemption_ratio: float = 0.2     # 最大可抢占任务比例
    allow_cascade: bool = True            # 是否允许级联抢占
    max_cascade_depth: int = 3            # 最大级联深度
    respect_deadline: bool = True         # 是否尊重截止时间


@dataclass
class PriorityRule:
    """优先级规则配置"""
    priority_weight: float = 1.0          # 优先级权重
    deadline_weight: float = 0.8          # 截止时间权重
    resource_efficiency_weight: float = 0.5  # 资源效率权重
    earliest_start_weight: float = 0.3    # 最早开始时间权重


@dataclass
class IncrementalPlanRequest:
    """增量规划请求"""
    new_targets: List[Target]                        # 新增目标
    existing_schedule: ScheduleResult                # 现有调度结果
    strategy: IncrementalStrategyType = IncrementalStrategyType.CONSERVATIVE
    priority_rules: Optional[PriorityRule] = None    # 优先级规则
    preemption_rules: Optional[PreemptionRule] = None  # 抢占规则
    max_preemption_ratio: float = 0.2                # 最大抢占比例
    min_task_duration: float = 30.0                  # 最小任务持续时间（秒）
    mission: Optional[Any] = None                    # 任务场景（如未提供则尝试从context推断）


@dataclass
class PreemptionCandidate:
    """抢占候选任务"""
    task: ScheduledTask
    satellite_id: str
    reclaimable_power: float
    reclaimable_storage: float
    reschedule_difficulty: float      # 0-1, 越高越难重调度
    priority_score: float             # 任务优先级评分
    preemption_benefit: float         # 抢占收益评分
    cascade_impact: int = 0           # 级联影响任务数

    def calculate_score(self, new_target_priority: int) -> float:
        """计算抢占评分（越高越适合被抢占）"""
        priority_diff = new_target_priority - (self.task.priority or 0)
        return (priority_diff * 0.4 +
                self.preemption_benefit * 0.3 -
                self.reschedule_difficulty * 0.2 -
                self.cascade_impact * 0.1)


@dataclass
class ResourceDelta:
    """资源变化量"""
    power_delta: float = 0.0          # 电量变化
    storage_delta: float = 0.0        # 存储变化
    time_delta: float = 0.0           # 时间变化（秒）
    task_count_delta: int = 0         # 任务数变化


@dataclass
class IncrementalPlanResult:
    """增量规划结果"""
    merged_schedule: ScheduleResult                  # 合并后的调度结果
    new_tasks: List[ScheduledTask]                   # 新增任务
    preempted_tasks: List[ScheduledTask]             # 被抢占的任务
    rescheduled_tasks: List[ScheduledTask]           # 重调度的任务
    failed_targets: List[Tuple[Target, str]]         # 失败的目标及原因
    resource_usage_delta: ResourceDelta              # 资源使用变化
    strategy_used: IncrementalStrategyType           # 实际使用的策略
    statistics: Dict[str, Any] = field(default_factory=dict)  # 详细统计

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_used': self.strategy_used.value,
            'new_tasks_count': len(self.new_tasks),
            'preempted_tasks_count': len(self.preempted_tasks),
            'rescheduled_tasks_count': len(self.rescheduled_tasks),
            'failed_targets_count': len(self.failed_targets),
            'resource_delta': {
                'power_delta': self.resource_usage_delta.power_delta,
                'storage_delta': self.resource_usage_delta.storage_delta,
                'time_delta': self.resource_usage_delta.time_delta,
                'task_count_delta': self.resource_usage_delta.task_count_delta,
            },
            'statistics': self.statistics,
            'new_tasks': [t.to_dict() for t in self.new_tasks],
            'preempted_tasks': [t.to_dict() for t in self.preempted_tasks],
            'failed_targets': [
                {'target_id': t.id if hasattr(t, 'id') else str(t), 'reason': r}
                for t, r in self.failed_targets
            ]
        }


class BaseIncrementalPlanner(ABC):
    """
    增量规划器基类

    所有增量规划策略（保守/激进/混合）必须继承此类
    """

    def __init__(self, strategy_type: IncrementalStrategyType, config: Dict[str, Any] = None):
        """
        初始化增量规划器

        Args:
            strategy_type: 策略类型
            config: 配置字典
        """
        self.strategy_type = strategy_type
        self.config = config or {}
        self.mission = None
        self.window_cache = None

    @abstractmethod
    def plan(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """
        执行增量规划

        Args:
            request: 增量规划请求

        Returns:
            IncrementalPlanResult: 规划结果
        """
        pass

    def initialize(self, mission: Any, window_cache: Any = None) -> None:
        """
        初始化调度器

        Args:
            mission: 任务场景对象
            window_cache: 可见性窗口缓存
        """
        self.mission = mission
        self.window_cache = window_cache
        logger.info(f"Initialized {self.strategy_type.value} incremental planner")

    def _calculate_task_priority(self, target: Target,
                                  rules: Optional[PriorityRule] = None) -> float:
        """
        计算任务优先级分数

        Args:
            target: 目标对象
            rules: 优先级规则

        Returns:
            float: 优先级分数
        """
        rules = rules or PriorityRule()
        score = 0.0

        # 基础优先级
        base_priority = getattr(target, 'priority', 0)
        score += base_priority * rules.priority_weight

        # 截止时间因素
        deadline = getattr(target, 'deadline', None)
        if deadline and self.mission:
            time_to_deadline = (deadline - self.mission.start_time).total_seconds()
            urgency = max(0, 1.0 - time_to_deadline / 86400)  # 24小时内归一化
            score += urgency * 100 * rules.deadline_weight

        return score

    def _find_visibility_windows(self, satellite_id: str, target: Target) -> List[Tuple[datetime, datetime]]:
        """
        查找卫星对目标的可见窗口

        Args:
            satellite_id: 卫星ID
            target: 目标对象

        Returns:
            List[Tuple[datetime, datetime]]: 可见窗口列表
        """
        if not self.window_cache:
            return []

        cache_key = f"{satellite_id}:{getattr(target, 'id', str(target))}"
        return self.window_cache.get(cache_key, [])

    def _validate_request(self, request: IncrementalPlanRequest) -> bool:
        """
        验证增量规划请求

        Args:
            request: 增量规划请求

        Returns:
            bool: 是否有效
        """
        if not request.new_targets:
            logger.warning("No new targets in request")
            return False

        if not request.existing_schedule:
            logger.error("No existing schedule in request")
            return False

        if request.strategy != self.strategy_type:
            logger.error(f"Strategy mismatch: request={request.strategy.value}, "
                        f"planner={self.strategy_type.value}")
            return False

        # 验证抢占比例
        if hasattr(request, 'max_preemption_ratio'):
            if not 0.0 <= request.max_preemption_ratio <= 1.0:
                logger.error(f"Invalid max_preemption_ratio: {request.max_preemption_ratio}, "
                            "must be in [0.0, 1.0]")
                return False

        logger.debug(f"Request validation passed: {len(request.new_targets)} targets, "
                    f"strategy={request.strategy.value}")
        return True

    def _merge_schedules(self, original: ScheduleResult,
                        new_tasks: List[ScheduledTask],
                        preempted: List[ScheduledTask],
                        rescheduled: List[ScheduledTask]) -> ScheduleResult:
        """
        合并调度结果

        Args:
            original: 原始调度结果
            new_tasks: 新增任务
            preempted: 被抢占任务（需要移除）
            rescheduled: 重调度任务（更新）

        Returns:
            ScheduleResult: 合并后的结果
        """
        # 创建任务ID集合便于查找
        preempted_ids = {t.task_id for t in preempted}
        rescheduled_map = {t.task_id: t for t in rescheduled}

        # 过滤原始任务
        filtered_tasks = [
            t for t in original.scheduled_tasks
            if t.task_id not in preempted_ids
        ]

        # 更新重调度任务
        final_tasks = []
        for t in filtered_tasks:
            if t.task_id in rescheduled_map:
                final_tasks.append(rescheduled_map[t.task_id])
            else:
                final_tasks.append(t)

        # 添加新任务
        final_tasks.extend(new_tasks)

        # 按时间排序
        final_tasks.sort(key=lambda t: t.imaging_start)

        return ScheduleResult(
            scheduled_tasks=final_tasks,
            unscheduled_tasks=original.unscheduled_tasks,
            makespan=original.makespan,
            computation_time=original.computation_time,
            iterations=original.iterations,
            convergence_curve=original.convergence_curve
        )

    def get_name(self) -> str:
        """获取规划器名称"""
        return f"IncrementalPlanner({self.strategy_type.value})"

    def get_parameters(self) -> Dict[str, Any]:
        """获取可配置参数"""
        return {
            'strategy': self.strategy_type.value,
            **self.config
        }
