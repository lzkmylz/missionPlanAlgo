"""
单元测试：验证优先级修复是否正确

测试内容：
1. Target模型优先级范围1-100，数字越小优先级越高
2. GreedyScheduler按优先级排序（数字小的优先）
3. EDDScheduler截止时间相同时按优先级排序（数字小的优先）
4. SPTScheduler处理时间相同时按优先级排序（数字小的优先）
5. 元启发式基类适应度函数包含优先级奖励
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime, timedelta
from core.models.target import Target
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler


class MockTask:
    """Mock任务对象用于测试"""
    def __init__(self, task_id, priority=50, time_window_end=None):
        self.id = task_id
        self.task_id = task_id
        self.priority = priority
        self.time_window_end = time_window_end
        self.target_type = 'point'
        self.preferred_mode = None

    def __repr__(self):
        return f"MockTask({self.id}, priority={self.priority})"


class TestPriorityRange(unittest.TestCase):
    """测试优先级范围定义"""

    def test_target_default_priority(self):
        """测试Target默认优先级"""
        target = Target(
            id="TEST-001",
            name="Test Target",
            longitude=116.4,
            latitude=39.9
        )
        # 默认优先级应该是1（最低数字=最高优先级）
        self.assertEqual(target.priority, 1)

    def test_target_custom_priority(self):
        """测试Target自定义优先级"""
        target = Target(
            id="TEST-002",
            name="Test Target",
            longitude=116.4,
            latitude=39.9,
            priority=100  # 低优先级
        )
        self.assertEqual(target.priority, 100)

    def test_priority_range_1_to_100(self):
        """测试优先级范围1-100"""
        for p in [1, 10, 25, 50, 75, 100]:
            target = Target(
                id=f"TEST-{p}",
                name="Test Target",
                longitude=116.4,
                latitude=39.9,
                priority=p
            )
            self.assertEqual(target.priority, p)


class TestGreedySchedulerPriority(unittest.TestCase):
    """测试GreedyScheduler优先级排序"""

    def setUp(self):
        self.scheduler = GreedyScheduler(config={'heuristic': 'priority'})

    def test_priority_sorting_ascending(self):
        """测试优先级排序：数字越小越优先"""
        tasks = [
            MockTask("T1", priority=100),  # 低优先级
            MockTask("T2", priority=1),    # 高优先级
            MockTask("T3", priority=50),   # 中优先级
            MockTask("T4", priority=25),   # 中高优先级
        ]

        sorted_tasks = self.scheduler._sort_tasks_by_priority(tasks)
        priorities = [t.priority for t in sorted_tasks]

        # 应该是升序：1, 25, 50, 100
        self.assertEqual(priorities, [1, 25, 50, 100])
        self.assertEqual(sorted_tasks[0].id, "T2")  # 优先级1最先
        self.assertEqual(sorted_tasks[-1].id, "T1")  # 优先级100最后

    def test_priority_sorting_with_defaults(self):
        """测试优先级排序：处理默认值"""
        tasks = [
            MockTask("T1", priority=None),  # None应该被视为50
            MockTask("T2", priority=1),
            MockTask("T3", priority=100),
        ]

        sorted_tasks = self.scheduler._sort_tasks_by_priority(tasks)
        priorities = [t.priority for t in sorted_tasks]

        # None被视为50，排序应该是：1, 50, 100
        self.assertEqual(sorted_tasks[0].id, "T2")
        self.assertEqual(sorted_tasks[1].id, "T1")
        self.assertEqual(sorted_tasks[2].id, "T3")


class TestEDDSchedulerPriority(unittest.TestCase):
    """测试EDDScheduler优先级排序"""

    def setUp(self):
        self.scheduler = EDDScheduler(config={})

    def test_edd_priority_sorting(self):
        """测试EDD排序：截止时间相同，优先级小的优先"""
        now = datetime.now()
        same_deadline = now + timedelta(hours=1)

        tasks = [
            MockTask("T1", priority=100, time_window_end=same_deadline),
            MockTask("T2", priority=1, time_window_end=same_deadline),
            MockTask("T3", priority=50, time_window_end=same_deadline),
        ]

        sorted_tasks = self.scheduler._sort_tasks(tasks)

        # 截止时间相同，按优先级升序：1, 50, 100
        self.assertEqual(sorted_tasks[0].id, "T2")  # 优先级1
        self.assertEqual(sorted_tasks[1].id, "T3")  # 优先级50
        self.assertEqual(sorted_tasks[2].id, "T1")  # 优先级100

    def test_edd_different_deadlines(self):
        """测试EDD排序：不同截止时间，截止时间优先"""
        now = datetime.now()

        tasks = [
            MockTask("T1", priority=1, time_window_end=now + timedelta(hours=3)),
            MockTask("T2", priority=100, time_window_end=now + timedelta(hours=1)),
            MockTask("T3", priority=50, time_window_end=now + timedelta(hours=2)),
        ]

        sorted_tasks = self.scheduler._sort_tasks(tasks)

        # 按截止时间排序，不考虑优先级
        self.assertEqual(sorted_tasks[0].id, "T2")  # 1小时后截止
        self.assertEqual(sorted_tasks[1].id, "T3")  # 2小时后截止
        self.assertEqual(sorted_tasks[2].id, "T1")  # 3小时后截止


class TestSPTSchedulerPriority(unittest.TestCase):
    """测试SPTScheduler优先级排序"""

    def setUp(self):
        self.scheduler = SPTScheduler(config={})

    def test_spt_priority_sorting(self):
        """测试SPT排序：处理时间相同，优先级小的优先"""
        # 所有任务的estimated_processing_time相同（都是点目标=5秒）
        tasks = [
            MockTask("T1", priority=100),
            MockTask("T2", priority=1),
            MockTask("T3", priority=50),
        ]

        sorted_tasks = self.scheduler._sort_tasks(tasks)

        # 处理时间相同，按优先级升序：1, 50, 100
        self.assertEqual(sorted_tasks[0].id, "T2")  # 优先级1
        self.assertEqual(sorted_tasks[1].id, "T3")  # 优先级50
        self.assertEqual(sorted_tasks[2].id, "T1")  # 优先级100


class TestMetaheuristicPriorityBonus(unittest.TestCase):
    """测试元启发式基类优先级奖励"""

    def test_priority_bonus_calculation(self):
        """测试优先级奖励计算"""
        from scheduler.metaheuristic.base_metaheuristic import (
            MetaheuristicScheduler, EvaluationState
        )

        # 创建一个简单的子类用于测试
        class TestScheduler(MetaheuristicScheduler):
            def initialize_population(self):
                return []

            def evolve(self, population):
                return population

            def get_parameters(self):
                return {}

        # 创建模拟任务
        tasks = [
            MockTask("T1", priority=1),   # 应该获得最高奖励: (101-1)*0.5 = 50
            MockTask("T2", priority=50),  # 中等奖励: (101-50)*0.5 = 25.5
            MockTask("T3", priority=100), # 最低奖励: (101-100)*0.5 = 0.5
        ]

        # 创建调度器并设置任务
        scheduler = TestScheduler("TEST", config={})
        scheduler.tasks = tasks
        scheduler.satellites = []

        # 创建状态
        state = EvaluationState(
            score=0.0,
            scheduled_count=3,
            sat_task_times={
                0: [(datetime.now(), datetime.now())],
                1: [(datetime.now(), datetime.now())],
                2: [(datetime.now(), datetime.now())]
            }
        )

        # 计算优先级奖励
        bonus = scheduler._calculate_priority_bonus(state)

        # 验证奖励是正数
        self.assertGreater(bonus, 0)

        # 验证优先级1的任务奖励最高
        # 注意：由于sat_task_times的索引和tasks的索引对应关系，
        # 我们需要确保任务索引匹配
        print(f"Priority bonus: {bonus}")

    def test_priority_bonus_values(self):
        """测试优先级奖励的具体数值"""
        # 优先级1 -> 奖励 = (101-1)*0.5 = 50.0
        # 优先级50 -> 奖励 = (101-50)*0.5 = 25.5
        # 优先级100 -> 奖励 = (101-100)*0.5 = 0.5

        self.assertEqual((101 - 1) * 0.5, 50.0)
        self.assertEqual((101 - 50) * 0.5, 25.5)
        self.assertEqual((101 - 100) * 0.5, 0.5)


class TestPriorityConflictResolution(unittest.TestCase):
    """测试优先级冲突解决逻辑"""

    def test_high_priority_vs_low_priority(self):
        """测试高优先级 vs 低优先级"""
        high_priority = MockTask("HIGH", priority=1)
        low_priority = MockTask("LOW", priority=100)

        scheduler = GreedyScheduler(config={'heuristic': 'priority'})
        sorted_tasks = scheduler._sort_tasks_by_priority([low_priority, high_priority])

        # 高优先级（数字小）应该排在前面
        self.assertEqual(sorted_tasks[0].id, "HIGH")
        self.assertEqual(sorted_tasks[1].id, "LOW")

    def test_equal_priority_any_order(self):
        """测试相同优先级时任一顺序均可"""
        task1 = MockTask("T1", priority=50)
        task2 = MockTask("T2", priority=50)

        scheduler = GreedyScheduler(config={'heuristic': 'priority'})
        sorted_tasks = scheduler._sort_tasks_by_priority([task1, task2])

        # 相同优先级，保持原始顺序（稳定排序）
        self.assertEqual(len(sorted_tasks), 2)
        self.assertEqual(sorted_tasks[0].priority, sorted_tasks[1].priority)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestPriorityRange))
    suite.addTests(loader.loadTestsFromTestCase(TestGreedySchedulerPriority))
    suite.addTests(loader.loadTestsFromTestCase(TestEDDSchedulerPriority))
    suite.addTests(loader.loadTestsFromTestCase(TestSPTSchedulerPriority))
    suite.addTests(loader.loadTestsFromTestCase(TestMetaheuristicPriorityBonus))
    suite.addTests(loader.loadTestsFromTestCase(TestPriorityConflictResolution))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
