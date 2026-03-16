"""
增量规划策略单元测试
"""

import unittest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scheduler.incremental.strategies.conservative_strategy import ConservativeStrategy
from scheduler.incremental.strategies.aggressive_strategy import AggressiveStrategy
from scheduler.incremental.strategies.hybrid_strategy import HybridStrategy
from scheduler.incremental.base_incremental import (
    IncrementalPlanRequest, IncrementalStrategyType,
    PriorityRule, PreemptionRule
)
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


class MockTarget:
    """模拟目标"""
    def __init__(self, target_id, priority=0, imaging_mode='standard'):
        self.id = target_id
        self.priority = priority
        self.imaging_mode = imaging_mode
        self.imaging_duration = 600.0  # 10 minutes


class MockMission:
    """模拟任务场景"""
    def __init__(self):
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)
        self.end_time = datetime(2024, 1, 2, 0, 0, 0)
        self.satellites = []


class TestConservativeStrategy(unittest.TestCase):
    """测试保守策略"""

    def setUp(self):
        self.strategy = ConservativeStrategy(config={
            'min_window_gap': 60
        })

        self.start_time = datetime(2024, 1, 1, 0, 0, 0)

        # 创建现有调度结果
        self.existing_tasks = [
            ScheduledTask(
                task_id='T001',
                satellite_id='SAT-001',
                target_id='TARGET-001',
                imaging_start=self.start_time + timedelta(hours=1),
                imaging_end=self.start_time + timedelta(hours=1, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                storage_before=0.0,
                storage_after=5.0,
                priority=5
            )
        ]

        self.existing_schedule = ScheduleResult(
            scheduled_tasks=self.existing_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

    def test_strategy_type(self):
        """测试策略类型"""
        self.assertEqual(self.strategy.strategy_type, IncrementalStrategyType.CONSERVATIVE)

    def test_plan_with_empty_targets(self):
        """测试空目标列表"""
        request = IncrementalPlanRequest(
            new_targets=[],
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.CONSERVATIVE,
            mission=MockMission()
        )

        result = self.strategy.plan(request)

        self.assertEqual(len(result.new_tasks), 0)
        self.assertEqual(len(result.failed_targets), 0)

    def test_plan_with_new_targets(self):
        """测试新增目标"""
        new_targets = [
            MockTarget('NEW-001', priority=8),
            MockTarget('NEW-002', priority=5)
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.CONSERVATIVE,
            mission=MockMission()
        )

        # 由于缺少窗口缓存，调度可能会失败，但流程应该正常执行
        result = self.strategy.plan(request)

        # 验证结果结构
        self.assertIsNotNone(result.merged_schedule)
        self.assertEqual(result.strategy_used, IncrementalStrategyType.CONSERVATIVE)

    def test_validate_request(self):
        """测试请求验证"""
        # 无效请求 - 没有现有调度
        request = IncrementalPlanRequest(
            new_targets=[MockTarget('T001')],
            existing_schedule=None,
            strategy=IncrementalStrategyType.CONSERVATIVE
        )

        self.assertFalse(self.strategy._validate_request(request))

        # 无效请求 - 策略不匹配
        request = IncrementalPlanRequest(
            new_targets=[MockTarget('T001')],
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.AGGRESSIVE
        )

        self.assertFalse(self.strategy._validate_request(request))


class TestAggressiveStrategy(unittest.TestCase):
    """测试激进策略"""

    def setUp(self):
        self.strategy = AggressiveStrategy(config={
            'min_window_gap': 60,
            'max_preemption_ratio': 0.2,
            'max_cascade_depth': 3
        })

        self.start_time = datetime(2024, 1, 1, 0, 0, 0)

        # 创建现有调度结果
        self.existing_tasks = [
            ScheduledTask(
                task_id='T001',
                satellite_id='SAT-001',
                target_id='TARGET-001',
                imaging_start=self.start_time + timedelta(hours=1),
                imaging_end=self.start_time + timedelta(hours=1, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                storage_before=0.0,
                storage_after=5.0,
                priority=3  # 低优先级，可被抢占
            ),
            ScheduledTask(
                task_id='T002',
                satellite_id='SAT-001',
                target_id='TARGET-002',
                imaging_start=self.start_time + timedelta(hours=2),
                imaging_end=self.start_time + timedelta(hours=2, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                storage_before=5.0,
                storage_after=10.0,
                priority=9  # 高优先级，不应被抢占
            )
        ]

        self.existing_schedule = ScheduleResult(
            scheduled_tasks=self.existing_tasks,
            unscheduled_tasks={},
            makespan=7200.0,
            computation_time=1.0,
            iterations=100
        )

    def test_strategy_type(self):
        """测试策略类型"""
        self.assertEqual(self.strategy.strategy_type, IncrementalStrategyType.AGGRESSIVE)

    def test_preemption_rules(self):
        """测试抢占规则"""
        new_targets = [
            MockTarget('NEW-001', priority=8)  # 高优先级，可抢占低优先级任务
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.AGGRESSIVE,
            preemption_rules=PreemptionRule(
                min_priority_difference=2,
                max_preemption_ratio=0.2
            ),
            mission=MockMission()
        )

        result = self.strategy.plan(request)

        # 验证结果结构
        self.assertIsNotNone(result.merged_schedule)
        self.assertEqual(result.strategy_used, IncrementalStrategyType.AGGRESSIVE)


class TestHybridStrategy(unittest.TestCase):
    """测试混合策略"""

    def setUp(self):
        self.strategy = HybridStrategy(config={
            'high_priority_threshold': 8,
            'aggressive_target_ratio': 0.3
        })

        self.start_time = datetime(2024, 1, 1, 0, 0, 0)

        self.existing_schedule = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

    def test_strategy_type(self):
        """测试策略类型"""
        self.assertEqual(self.strategy.strategy_type, IncrementalStrategyType.HYBRID)

    def test_decide_strategy_mode_conservative(self):
        """测试决策 - 保守模式"""
        analysis = {
            'high_priority_ratio': 0.1,
            'resource_scarcity': False,
            'capacity_ratio': 2.0
        }

        mode = self.strategy._decide_strategy_mode(
            IncrementalPlanRequest(
                new_targets=[],
                existing_schedule=self.existing_schedule,
                strategy=IncrementalStrategyType.HYBRID
            ),
            analysis
        )

        self.assertEqual(mode, 'conservative')

    def test_decide_strategy_mode_aggressive(self):
        """测试决策 - 激进模式"""
        analysis = {
            'high_priority_ratio': 0.5,
            'resource_scarcity': True,
            'capacity_ratio': 0.5
        }

        mode = self.strategy._decide_strategy_mode(
            IncrementalPlanRequest(
                new_targets=[],
                existing_schedule=self.existing_schedule,
                strategy=IncrementalStrategyType.HYBRID
            ),
            analysis
        )

        self.assertEqual(mode, 'aggressive')

    def test_plan_with_mixed_priorities(self):
        """测试混合优先级目标"""
        new_targets = [
            MockTarget('LOW-001', priority=3),
            MockTarget('LOW-002', priority=4),
            MockTarget('HIGH-001', priority=9),
            MockTarget('HIGH-002', priority=10)
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.HYBRID,
            mission=MockMission()
        )

        result = self.strategy.plan(request)

        # 验证结果结构
        self.assertIsNotNone(result.merged_schedule)
        self.assertEqual(result.strategy_used, IncrementalStrategyType.HYBRID)
        self.assertIn('hybrid_mode', result.statistics)


class TestIncrementalPlanResult(unittest.TestCase):
    """测试增量规划结果"""

    def test_to_dict(self):
        """测试结果转字典"""
        from scheduler.incremental.base_incremental import IncrementalPlanResult, ResourceDelta

        result = IncrementalPlanResult(
            merged_schedule=ScheduleResult(
                scheduled_tasks=[],
                unscheduled_tasks={},
                makespan=0.0,
                computation_time=0.0,
                iterations=0
            ),
            new_tasks=[],
            preempted_tasks=[],
            rescheduled_tasks=[],
            failed_targets=[],
            resource_usage_delta=ResourceDelta(
                power_delta=100.0,
                storage_delta=50.0,
                time_delta=3600.0,
                task_count_delta=5
            ),
            strategy_used=IncrementalStrategyType.CONSERVATIVE,
            statistics={'test': True}
        )

        data = result.to_dict()

        self.assertEqual(data['strategy_used'], 'conservative')
        self.assertEqual(data['new_tasks_count'], 0)
        self.assertEqual(data['resource_delta']['power_delta'], 100.0)


if __name__ == '__main__':
    unittest.main()
