"""
增量规划状态管理器单元测试
"""

import unittest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from scheduler.incremental.incremental_state import (
    IncrementalState, SatelliteState, ResourceWindow
)
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


class MockMission:
    """模拟任务场景"""
    def __init__(self):
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)
        self.end_time = datetime(2024, 1, 2, 0, 0, 0)
        self.satellites = []


class MockCapabilities:
    """模拟卫星能力"""
    def __init__(self, power_capacity=2800.0, storage_capacity=128.0):
        self.power_capacity = power_capacity
        self.storage_capacity = storage_capacity


class MockSatellite:
    """模拟卫星"""
    def __init__(self, sat_id, power_capacity=2800.0, storage_capacity=128.0):
        self.id = sat_id
        self.capabilities = MockCapabilities(power_capacity, storage_capacity)


class TestResourceWindow(unittest.TestCase):
    """测试ResourceWindow类"""

    def test_duration_calculation(self):
        """测试持续时间计算"""
        window = ResourceWindow(
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 1, 1, 0, 0),
            available_power=1000.0,
            available_storage=50.0,
            satellite_id='SAT-001'
        )
        self.assertEqual(window.duration(), 3600.0)

    def test_can_accommodate(self):
        """测试容量检查"""
        window = ResourceWindow(
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 1, 1, 0, 0),
            available_power=100.0,
            available_storage=50.0,
            satellite_id='SAT-001'
        )

        # 可以容纳
        self.assertTrue(window.can_accommodate(1800, 50.0, 30.0))

        # 时间不足
        self.assertFalse(window.can_accommodate(7200, 50.0, 30.0))

        # 电量不足
        self.assertFalse(window.can_accommodate(1800, 200.0, 30.0))

        # 存储不足
        self.assertFalse(window.can_accommodate(1800, 50.0, 100.0))

    def test_split_for_task(self):
        """测试窗口分割"""
        window = ResourceWindow(
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 1, 2, 0, 0),
            available_power=1000.0,
            available_storage=50.0,
            satellite_id='SAT-001'
        )

        task_start = datetime(2024, 1, 1, 0, 30, 0)
        task_end = datetime(2024, 1, 1, 1, 30, 0)

        before, after = window.split_for_task(task_start, task_end)

        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertEqual(before.end_time, task_start)
        self.assertEqual(after.start_time, task_end)


class TestSatelliteState(unittest.TestCase):
    """测试SatelliteState类"""

    def setUp(self):
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)
        self.tasks = [
            {
                'task_id': 'T001',
                'target_id': 'TARGET-001',
                'imaging_start': self.start_time + timedelta(hours=1),
                'imaging_end': self.start_time + timedelta(hours=1, minutes=10),
                'power_consumed': 10.0,
                'storage_produced': 5.0,
                'priority': 5
            },
            {
                'task_id': 'T002',
                'target_id': 'TARGET-002',
                'imaging_start': self.start_time + timedelta(hours=2),
                'imaging_end': self.start_time + timedelta(hours=2, minutes=10),
                'power_consumed': 10.0,
                'storage_produced': 5.0,
                'priority': 3
            }
        ]

    def test_get_resource_at_time(self):
        """测试获取指定时刻资源"""
        state = SatelliteState(
            satellite_id='SAT-001',
            scheduled_tasks=self.tasks.copy(),
            current_power=2800.0,
            current_storage=0.0,
            power_capacity=2800.0,
            storage_capacity=128.0
        )

        # 任务前
        power, storage = state.get_resource_at_time(self.start_time)
        self.assertEqual(power, 2800.0)
        self.assertEqual(storage, 0.0)

    def test_find_resource_windows(self):
        """测试查找资源窗口"""
        state = SatelliteState(
            satellite_id='SAT-001',
            scheduled_tasks=self.tasks.copy(),
            current_power=2800.0,
            current_storage=0.0,
            power_capacity=2800.0,
            storage_capacity=128.0
        )

        windows = state.find_resource_windows(
            self.start_time,
            self.start_time + timedelta(hours=4)
        )

        self.assertGreaterEqual(len(windows), 1)
        # 应该找到任务之间的间隙


class TestIncrementalState(unittest.TestCase):
    """测试IncrementalState类"""

    def setUp(self):
        self.mission = MockMission()
        self.mission.satellites = [
            MockSatellite('SAT-001'),
            MockSatellite('SAT-002')
        ]

        self.start_time = datetime(2024, 1, 1, 0, 0, 0)

        # 创建测试用的ScheduleResult
        self.tasks = [
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
                imaging_start=self.start_time + timedelta(hours=2),
                imaging_end=self.start_time + timedelta(hours=2, minutes=10),
                imaging_mode='standard',
                power_consumed=10.0,
                power_before=2790.0,
                power_after=2780.0,
                storage_before=5.0,
                storage_after=10.0,
                priority=3
            )
        ]

        self.schedule_result = ScheduleResult(
            scheduled_tasks=self.tasks,
            unscheduled_tasks={},
            makespan=7200.0,
            computation_time=1.0,
            iterations=100
        )

    def test_load_from_schedule(self):
        """测试从调度结果加载"""
        state = IncrementalState(self.mission)
        state.load_from_schedule(self.schedule_result)

        self.assertEqual(len(state.satellite_states), 1)  # SAT-001有任务
        self.assertIn('SAT-001', state.satellite_states)

        sat_state = state.get_satellite_state('SAT-001')
        self.assertEqual(len(sat_state.scheduled_tasks), 2)

    def test_get_available_windows(self):
        """测试获取可用窗口"""
        state = IncrementalState(self.mission)
        state.load_from_schedule(self.schedule_result)

        windows = state.get_available_windows('SAT-001')
        self.assertGreaterEqual(len(windows), 1)

    def test_add_task(self):
        """测试添加任务"""
        state = IncrementalState(self.mission)
        state.load_from_schedule(self.schedule_result)

        new_task = {
            'task_id': 'T003',
            'target_id': 'TARGET-003',
            'imaging_start': self.start_time + timedelta(hours=3),
            'imaging_end': self.start_time + timedelta(hours=3, minutes=10),
            'power_consumed': 10.0,
            'storage_produced': 5.0,
            'priority': 7
        }

        result = state.add_task('SAT-001', new_task)
        self.assertTrue(result)

        sat_state = state.get_satellite_state('SAT-001')
        self.assertEqual(len(sat_state.scheduled_tasks), 3)

    def test_remove_task(self):
        """测试移除任务"""
        state = IncrementalState(self.mission)
        state.load_from_schedule(self.schedule_result)

        result = state.remove_task('SAT-001', 'T001')
        self.assertTrue(result)

        sat_state = state.get_satellite_state('SAT-001')
        self.assertEqual(len(sat_state.scheduled_tasks), 1)

    def test_snapshot_and_restore(self):
        """测试快照和恢复"""
        state = IncrementalState(self.mission)
        state.load_from_schedule(self.schedule_result)

        # 创建快照
        snapshot = state.create_snapshot()

        # 添加任务
        new_task = {
            'task_id': 'T003',
            'target_id': 'TARGET-003',
            'imaging_start': self.start_time + timedelta(hours=3),
            'imaging_end': self.start_time + timedelta(hours=3, minutes=10),
            'power_consumed': 10.0,
            'storage_produced': 5.0,
            'priority': 7
        }
        state.add_task('SAT-001', new_task)

        # 恢复
        state.restore_from_snapshot(snapshot)

        sat_state = state.get_satellite_state('SAT-001')
        self.assertEqual(len(sat_state.scheduled_tasks), 2)


if __name__ == '__main__':
    unittest.main()
