"""
增量规划集成测试

测试完整的增量规划流程，包括：
- 从调度结果恢复状态
- 资源分析
- 增量规划执行
- 结果合并
"""

import unittest
from datetime import datetime, timedelta

from scheduler.incremental import (
    IncrementalPlanner, IncrementalPlanRequest, IncrementalStrategyType,
    IncrementalState, ResourceReclaimer
)
from scheduler.base_scheduler import ScheduleResult, ScheduledTask
from scheduler.greedy.greedy_scheduler import GreedyScheduler


class MockCapabilities:
    """模拟卫星能力"""
    def __init__(self, power_capacity=2800.0, storage_capacity=128.0):
        self.power_capacity = power_capacity
        self.storage_capacity = storage_capacity


class MockSatellite:
    """模拟卫星"""
    def __init__(self, sat_id):
        self.id = sat_id
        self.capabilities = MockCapabilities()


class MockMission:
    """模拟任务场景"""
    def __init__(self):
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)
        self.end_time = datetime(2024, 1, 2, 0, 0, 0)
        self.satellites = [
            MockSatellite('SAT-001'),
            MockSatellite('SAT-002')
        ]

    def get_satellite_by_id(self, sat_id):
        for sat in self.satellites:
            if sat.id == sat_id:
                return sat
        return None


class MockTarget:
    """模拟目标"""
    def __init__(self, target_id, priority=0, imaging_mode='standard'):
        self.id = target_id
        self.priority = priority
        self.imaging_mode = imaging_mode
        self.latitude = 39.9
        self.longitude = 116.4


class TestIncrementalPlanner(unittest.TestCase):
    """测试增量规划器主入口"""

    def setUp(self):
        self.planner = IncrementalPlanner()
        self.mission = MockMission()
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
                power_before=2800.0,
                power_after=2790.0,
                storage_before=0.0,
                storage_after=5.0,
                priority=5
            ),
            ScheduledTask(
                task_id='T002',
                satellite_id='SAT-001',
                target_id='TARGET-002',
                imaging_start=self.start_time + timedelta(hours=3),
                imaging_end=self.start_time + timedelta(hours=3, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                power_before=2790.0,
                power_after=2780.0,
                storage_before=5.0,
                storage_after=10.0,
                priority=3
            ),
            ScheduledTask(
                task_id='T003',
                satellite_id='SAT-002',
                target_id='TARGET-003',
                imaging_start=self.start_time + timedelta(hours=2),
                imaging_end=self.start_time + timedelta(hours=2, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                power_before=2800.0,
                power_after=2790.0,
                storage_before=0.0,
                storage_after=5.0,
                priority=7
            )
        ]

        self.existing_schedule = ScheduleResult(
            scheduled_tasks=self.existing_tasks,
            unscheduled_tasks={},
            makespan=10800.0,
            computation_time=1.0,
            iterations=100
        )

    def test_analyze_resources(self):
        """测试资源分析"""
        report = self.planner.analyze_resources(
            self.existing_schedule, self.mission
        )

        self.assertIn('summary', report)
        self.assertIn('satellite_details', report)
        self.assertEqual(report['summary']['total_satellites'], 2)

    def test_estimate_capacity(self):
        """测试容量估算"""
        capacities = self.planner.estimate_capacity(
            self.existing_schedule, self.mission
        )

        self.assertIn('SAT-001', capacities)
        self.assertIn('SAT-002', capacities)

    def test_plan_conservative(self):
        """测试保守策略规划"""
        new_targets = [
            MockTarget('NEW-001', priority=8),
            MockTarget('NEW-002', priority=5)
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.CONSERVATIVE,
            mission=self.mission
        )

        result = self.planner.plan(request)

        # 验证结果
        self.assertEqual(result.strategy_used, IncrementalStrategyType.CONSERVATIVE)
        self.assertIsNotNone(result.merged_schedule)
        self.assertGreaterEqual(len(result.new_tasks) + len(result.failed_targets), 0)

    def test_plan_aggressive(self):
        """测试激进策略规划"""
        new_targets = [
            MockTarget('URGENT-001', priority=10),
            MockTarget('URGENT-002', priority=9)
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.AGGRESSIVE,
            max_preemption_ratio=0.3,
            mission=self.mission
        )

        result = self.planner.plan(request)

        # 验证结果
        self.assertEqual(result.strategy_used, IncrementalStrategyType.AGGRESSIVE)
        self.assertIsNotNone(result.merged_schedule)

    def test_plan_hybrid(self):
        """测试混合策略规划"""
        new_targets = [
            MockTarget('LOW-001', priority=3),
            MockTarget('HIGH-001', priority=9),
            MockTarget('HIGH-002', priority=10)
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=self.existing_schedule,
            strategy=IncrementalStrategyType.HYBRID,
            mission=self.mission
        )

        result = self.planner.plan(request)

        # 验证结果
        self.assertEqual(result.strategy_used, IncrementalStrategyType.HYBRID)
        self.assertIsNotNone(result.merged_schedule)
        self.assertIn('hybrid_mode', result.statistics)


class TestBaseSchedulerIncremental(unittest.TestCase):
    """测试BaseScheduler的增量规划接口"""

    def setUp(self):
        self.scheduler = GreedyScheduler()
        self.mission = MockMission()
        self.scheduler.initialize(self.mission)
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)

    def test_resume_from(self):
        """测试从调度结果恢复"""
        existing_tasks = [
            ScheduledTask(
                task_id='T001',
                satellite_id='SAT-001',
                target_id='TARGET-001',
                imaging_start=self.start_time + timedelta(hours=1),
                imaging_end=self.start_time + timedelta(hours=1, minutes=10),
                imaging_mode='standard'
            )
        ]

        existing_schedule = ScheduleResult(
            scheduled_tasks=existing_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        new_targets = [MockTarget('NEW-001')]

        # 恢复状态
        result = self.scheduler.resume_from(existing_schedule, new_targets, 'conservative')

        # 验证返回self（链式调用）
        self.assertEqual(result, self.scheduler)

        # 验证内部状态
        self.assertTrue(hasattr(self.scheduler, '_incremental_state'))
        self.assertTrue(hasattr(self.scheduler, '_incremental_targets'))
        self.assertEqual(self.scheduler._incremental_strategy, 'conservative')

    def test_get_remaining_capacity(self):
        """测试获取剩余容量"""
        existing_tasks = [
            ScheduledTask(
                task_id='T001',
                satellite_id='SAT-001',
                target_id='TARGET-001',
                imaging_start=self.start_time + timedelta(hours=1),
                imaging_end=self.start_time + timedelta(hours=1, minutes=10),
                imaging_mode='standard'
            )
        ]

        existing_schedule = ScheduleResult(
            scheduled_tasks=existing_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        self.scheduler.resume_from(existing_schedule, [], 'conservative')

        capacities = self.scheduler.get_remaining_capacity()

        self.assertIn('SAT-001', capacities)
        self.assertIn('available_time', capacities['SAT-001'])
        self.assertIn('available_power', capacities['SAT-001'])
        self.assertIn('utilization_rate', capacities['SAT-001'])


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def test_full_incremental_workflow(self):
        """测试完整的增量规划工作流"""
        mission = MockMission()
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        # 1. 创建初始调度结果
        initial_tasks = [
            ScheduledTask(
                task_id=f'T{i:03d}',
                satellite_id='SAT-001',
                target_id=f'TARGET-{i:03d}',
                imaging_start=start_time + timedelta(hours=i*2),
                imaging_end=start_time + timedelta(hours=i*2, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                priority=i
            )
            for i in range(5)
        ]

        initial_schedule = ScheduleResult(
            scheduled_tasks=initial_tasks,
            unscheduled_tasks={},
            makespan=36000.0,
            computation_time=5.0,
            iterations=500
        )

        # 2. 分析资源
        planner = IncrementalPlanner()
        report = planner.analyze_resources(initial_schedule, mission)

        self.assertEqual(report['summary']['total_satellites'], 1)

        # 3. 创建增量规划请求
        new_targets = [
            MockTarget(f'NEW-{i:03d}', priority=8 if i < 3 else 3)
            for i in range(6)
        ]

        request = IncrementalPlanRequest(
            new_targets=new_targets,
            existing_schedule=initial_schedule,
            strategy=IncrementalStrategyType.HYBRID,
            mission=mission
        )

        # 4. 执行增量规划
        result = planner.plan(request)

        # 5. 验证结果
        self.assertEqual(result.strategy_used, IncrementalStrategyType.HYBRID)
        self.assertIsNotNone(result.merged_schedule)
        self.assertEqual(len(result.merged_schedule.scheduled_tasks),
                        len(initial_tasks) + len(result.new_tasks) - len(result.preempted_tasks))

        # 6. 验证结果可序列化
        result_dict = result.to_dict()
        self.assertIn('strategy_used', result_dict)
        self.assertIn('new_tasks_count', result_dict)
        self.assertIn('statistics', result_dict)


if __name__ == '__main__':
    unittest.main()
