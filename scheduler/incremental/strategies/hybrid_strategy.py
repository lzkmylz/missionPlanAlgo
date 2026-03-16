"""
混合策略增量规划器

策略逻辑：
1. 根据场景特征和请求参数动态选择策略
2. 高优先级目标自动使用激进策略
3. 普通目标使用保守策略
4. 支持策略切换阈值配置

决策因素：
- 目标优先级
- 剩余资源充足程度
- 时间紧迫性
- 已有任务重要性

适用场景：
- 复杂场景需要灵活调度
- 混合优先级目标批次
- 资源状况动态变化
"""

from typing import List, Dict, Any, Optional
import logging

from ..base_incremental import (
    BaseIncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalPlanResult,
    IncrementalStrategyType,
    ResourceDelta,
    PreemptionRule,
    PriorityRule
)
from ..incremental_state import IncrementalState
from ..resource_reclaimer import ResourceReclaimer
from ...base_scheduler import ScheduleResult, ScheduledTask

logger = logging.getLogger(__name__)


class HybridStrategy(BaseIncrementalPlanner):
    """
    混合策略增量规划器

    动态选择保守或激进策略，平衡资源利用和任务完成率
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化混合策略规划器

        Args:
            config: 配置字典
                - high_priority_threshold: 高优先级阈值，默认8
                - resource_scarcity_threshold: 资源稀缺阈值（利用率），默认0.85
                - aggressive_target_ratio: 激进策略目标比例，默认0.3
                - enable_dynamic_switch: 启用动态策略切换，默认True
        """
        # 处理 config 为 None 的情况
        config = config or {}
        super().__init__(IncrementalStrategyType.HYBRID, config)

        self.high_priority_threshold = config.get('high_priority_threshold', 8)
        self.resource_scarcity_threshold = config.get('resource_scarcity_threshold', 0.85)
        self.aggressive_target_ratio = config.get('aggressive_target_ratio', 0.3)
        self.enable_dynamic_switch = config.get('enable_dynamic_switch', True)

        # 子策略实例
        self._conservative = None
        self._aggressive = None

    def _ensure_substrategies(self):
        """确保子策略实例已创建"""
        if self._conservative is None:
            from .conservative_strategy import ConservativeStrategy
            self._conservative = ConservativeStrategy(self.config)

        if self._aggressive is None:
            from .aggressive_strategy import AggressiveStrategy
            self._aggressive = AggressiveStrategy(self.config)

    def plan(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """
        执行混合策略增量规划

        Args:
            request: 增量规划请求

        Returns:
            IncrementalPlanResult: 规划结果
        """
        # 验证请求
        if not self._validate_request(request):
            return self._create_empty_result(request)

        # 设置mission
        if request.mission:
            self.mission = request.mission

        # 确保子策略实例
        self._ensure_substrategies()

        # 初始化状态
        state = IncrementalState(self.mission)
        state.load_from_schedule(request.existing_schedule)

        reclaimer = ResourceReclaimer(state)

        logger.info(f"Hybrid planning: {len(request.new_targets)} new targets")

        # 分析场景特征
        scene_analysis = self._analyze_scene(request, state, reclaimer)
        logger.info(f"Scene analysis: {scene_analysis}")

        # 决策：使用哪种策略模式
        strategy_mode = self._decide_strategy_mode(request, scene_analysis)
        logger.info(f"Selected strategy mode: {strategy_mode}")

        # 执行相应的策略
        if strategy_mode == 'conservative':
            return self._execute_conservative(request, state)
        elif strategy_mode == 'aggressive':
            return self._execute_aggressive(request, state)
        else:  # mixed
            return self._execute_mixed(request, state, scene_analysis)

    def _analyze_scene(self, request: IncrementalPlanRequest,
                      state: IncrementalState,
                      reclaimer: ResourceReclaimer) -> Dict[str, Any]:
        """
        分析场景特征

        Returns:
            Dict: 场景分析结果
        """
        # 分析目标优先级分布
        priorities = [getattr(t, 'priority', 0) for t in request.new_targets]
        high_priority_count = sum(1 for p in priorities if p >= self.high_priority_threshold)

        # 分析资源状况
        all_resources = reclaimer.calculate_all_remaining_resources()
        avg_utilization = sum(r.utilization_rate for r in all_resources.values()) / len(all_resources) if all_resources else 0.0

        # 计算剩余任务容量
        total_capacity = sum(
            reclaimer.estimate_task_capacity(sat_id)
            for sat_id in state.get_all_satellite_ids()
        )

        # 计算时间紧迫性
        urgent_count = 0
        if self.mission:
            for target in request.new_targets:
                deadline = getattr(target, 'deadline', None)
                if deadline:
                    time_remaining = (deadline - self.mission.start_time).total_seconds()
                    if time_remaining < 3600:  # 1小时内
                        urgent_count += 1

        return {
            'high_priority_count': high_priority_count,
            'high_priority_ratio': high_priority_count / len(request.new_targets) if request.new_targets else 0,
            'average_utilization': avg_utilization,
            'resource_scarcity': avg_utilization > self.resource_scarcity_threshold,
            'total_capacity': total_capacity,
            'capacity_ratio': total_capacity / len(request.new_targets) if request.new_targets else 0,
            'urgent_count': urgent_count,
            'target_count': len(request.new_targets)
        }

    def _decide_strategy_mode(self, request: IncrementalPlanRequest,
                             analysis: Dict[str, Any]) -> str:
        """
        决策使用哪种策略模式

        Returns:
            str: 'conservative', 'aggressive', 或 'mixed'
        """
        if not self.enable_dynamic_switch:
            return 'mixed'

        # 如果高优先级目标比例高，使用激进策略
        if analysis['high_priority_ratio'] > self.aggressive_target_ratio:
            logger.info(f"High priority ratio {analysis['high_priority_ratio']:.2f} > threshold, using aggressive")
            return 'aggressive'

        # 如果资源稀缺，使用激进策略
        if analysis['resource_scarcity']:
            logger.info("Resource scarcity detected, using aggressive")
            return 'aggressive'

        # 如果容量充足，使用保守策略
        if analysis['capacity_ratio'] > 1.5:
            logger.info(f"Sufficient capacity {analysis['capacity_ratio']:.2f}, using conservative")
            return 'conservative'

        # 默认使用混合策略
        return 'mixed'

    def _execute_conservative(self, request: IncrementalPlanRequest,
                             state: IncrementalState) -> IncrementalPlanResult:
        """执行纯保守策略"""
        logger.info("Executing pure conservative strategy")

        # 使用保守策略规划所有目标
        conservative_request = IncrementalPlanRequest(
            new_targets=request.new_targets,
            existing_schedule=request.existing_schedule,
            strategy=IncrementalStrategyType.CONSERVATIVE,
            priority_rules=request.priority_rules,
            mission=request.mission
        )

        result = self._conservative.plan(conservative_request)
        result.strategy_used = IncrementalStrategyType.HYBRID
        result.statistics['hybrid_mode'] = 'conservative'

        return result

    def _execute_aggressive(self, request: IncrementalPlanRequest,
                           state: IncrementalState) -> IncrementalPlanResult:
        """执行纯激进策略"""
        logger.info("Executing pure aggressive strategy")

        # 使用激进策略规划所有目标
        aggressive_request = IncrementalPlanRequest(
            new_targets=request.new_targets,
            existing_schedule=request.existing_schedule,
            strategy=IncrementalStrategyType.AGGRESSIVE,
            priority_rules=request.priority_rules,
            preemption_rules=request.preemption_rules,
            max_preemption_ratio=request.max_preemption_ratio,
            mission=request.mission
        )

        result = self._aggressive.plan(aggressive_request)
        result.strategy_used = IncrementalStrategyType.HYBRID
        result.statistics['hybrid_mode'] = 'aggressive'

        return result

    def _execute_mixed(self, request: IncrementalPlanRequest,
                      state: IncrementalState,
                      analysis: Dict[str, Any]) -> IncrementalPlanResult:
        """
        执行混合策略

        高优先级目标使用激进策略，普通目标使用保守策略
        """
        logger.info("Executing mixed strategy")

        # 分割目标列表
        high_priority_targets = []
        normal_targets = []

        for target in request.new_targets:
            priority = getattr(target, 'priority', 0)
            deadline = getattr(target, 'deadline', None)

            # 高优先级或紧急的目标使用激进策略
            if priority >= self.high_priority_threshold:
                high_priority_targets.append(target)
            elif deadline and self.mission:
                time_to_deadline = (deadline - self.mission.start_time).total_seconds()
                if time_to_deadline < 7200:  # 2小时内截止
                    high_priority_targets.append(target)
                else:
                    normal_targets.append(target)
            else:
                normal_targets.append(target)

        logger.info(f"Split targets: {len(high_priority_targets)} high-priority, "
                   f"{len(normal_targets)} normal")

        # 第一阶段：保守策略规划普通目标
        phase1_result = None
        if normal_targets:
            conservative_request = IncrementalPlanRequest(
                new_targets=normal_targets,
                existing_schedule=request.existing_schedule,
                strategy=IncrementalStrategyType.CONSERVATIVE,
                priority_rules=request.priority_rules,
                mission=request.mission
            )

            phase1_result = self._conservative.plan(conservative_request)
            logger.info(f"Phase 1 (conservative): {len(phase1_result.new_tasks)} scheduled")

        # 第二阶段：激进策略规划高优先级目标
        base_schedule = phase1_result.merged_schedule if phase1_result else request.existing_schedule

        phase2_result = None
        if high_priority_targets:
            aggressive_request = IncrementalPlanRequest(
                new_targets=high_priority_targets,
                existing_schedule=base_schedule,
                strategy=IncrementalStrategyType.AGGRESSIVE,
                priority_rules=request.priority_rules,
                preemption_rules=request.preemption_rules,
                max_preemption_ratio=request.max_preemption_ratio * 0.5,  # 混合模式下限制抢占
                mission=request.mission
            )

            phase2_result = self._aggressive.plan(aggressive_request)
            logger.info(f"Phase 2 (aggressive): {len(phase2_result.new_tasks)} scheduled, "
                       f"{len(phase2_result.preempted_tasks)} preempted")

        # 合并结果
        return self._merge_mixed_results(
            request, phase1_result, phase2_result
        )

    def _merge_mixed_results(self, request: IncrementalPlanRequest,
                            phase1: Optional[IncrementalPlanResult],
                            phase2: Optional[IncrementalPlanResult]) -> IncrementalPlanResult:
        """合并混合策略的两阶段结果"""

        # 如果没有阶段2，直接返回阶段1
        if phase2 is None:
            if phase1:
                phase1.strategy_used = IncrementalStrategyType.HYBRID
                phase1.statistics['hybrid_mode'] = 'mixed'
                return phase1
            else:
                return self._create_empty_result(request)

        # 如果没有阶段1，返回阶段2
        if phase1 is None:
            phase2.strategy_used = IncrementalStrategyType.HYBRID
            phase2.statistics['hybrid_mode'] = 'mixed'
            return phase2

        # 合并两个阶段的任务
        all_new_tasks = phase1.new_tasks + phase2.new_tasks
        all_preempted = phase1.preempted_tasks + phase2.preempted_tasks
        all_rescheduled = phase1.rescheduled_tasks + phase2.rescheduled_tasks
        all_failed = phase1.failed_targets + phase2.failed_targets

        # 计算资源变化
        resource_delta = ResourceDelta(
            power_delta=phase1.resource_usage_delta.power_delta + phase2.resource_usage_delta.power_delta,
            storage_delta=phase1.resource_usage_delta.storage_delta + phase2.resource_usage_delta.storage_delta,
            time_delta=phase1.resource_usage_delta.time_delta + phase2.resource_usage_delta.time_delta,
            task_count_delta=phase1.resource_usage_delta.task_count_delta + phase2.resource_usage_delta.task_count_delta
        )

        return IncrementalPlanResult(
            merged_schedule=phase2.merged_schedule,  # 阶段2包含阶段1的更新
            new_tasks=all_new_tasks,
            preempted_tasks=all_preempted,
            rescheduled_tasks=all_rescheduled,
            failed_targets=all_failed,
            resource_usage_delta=resource_delta,
            strategy_used=IncrementalStrategyType.HYBRID,
            statistics={
                'total_targets': len(request.new_targets),
                'scheduled_count': len(all_new_tasks),
                'preempted_count': len(all_preempted),
                'rescheduled_count': len(all_rescheduled),
                'failed_count': len(all_failed),
                'success_rate': len(all_new_tasks) / len(request.new_targets) if request.new_targets else 0.0,
                'phase1_scheduled': len(phase1.new_tasks),
                'phase2_scheduled': len(phase2.new_tasks),
                'phase2_preempted': len(phase2.preempted_tasks),
                'hybrid_mode': 'mixed'
            }
        )

    def _create_empty_result(self, request: IncrementalPlanRequest) -> IncrementalPlanResult:
        """创建空结果"""
        return IncrementalPlanResult(
            merged_schedule=request.existing_schedule,
            new_tasks=[],
            preempted_tasks=[],
            rescheduled_tasks=[],
            failed_targets=[(t, "Invalid request") for t in request.new_targets],
            resource_usage_delta=ResourceDelta(),
            strategy_used=IncrementalStrategyType.HYBRID,
            statistics={'error': 'Invalid request', 'hybrid_mode': 'none'}
        )
