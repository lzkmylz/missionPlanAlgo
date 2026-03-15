"""
单圈约束检查器单元测试
"""

import unittest
from datetime import datetime, timedelta

from scheduler.constraints.batch_orbit_constraint_checker import (
    BatchOrbitConstraintChecker,
    BatchOrbitConstraintCandidate,
    BatchOrbitConstraintResult
)
from core.models.mission import Mission
from core.models.satellite import Satellite, SatelliteType, Orbit


class TestBatchOrbitConstraintChecker(unittest.TestCase):
    """测试单圈约束批量检查器"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试卫星（500km轨道，周期约94.6分钟）
        self.sat1 = Satellite(
            id='SAT-001',
            name='Test Satellite 1',
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000)  # 500km
        )
        # 设置单圈约束
        self.sat1.capabilities.max_starts_per_orbit = 3
        self.sat1.capabilities.max_work_time_per_orbit = 300.0  # 5分钟

        self.sat2 = Satellite(
            id='SAT-002',
            name='Test Satellite 2',
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000)
        )
        self.sat2.capabilities.max_starts_per_orbit = 5
        self.sat2.capabilities.max_work_time_per_orbit = 600.0  # 10分钟

        # 创建任务
        self.ref_time = datetime(2024, 1, 1, 0, 0, 0)
        self.mission = Mission(
            name='Test Mission',
            start_time=self.ref_time,
            end_time=self.ref_time + timedelta(hours=24),
            satellites=[self.sat1, self.sat2],
            targets=[],
            ground_stations=[]
        )

        self.checker = BatchOrbitConstraintChecker(self.mission)

    def test_single_candidate_no_existing_tasks(self):
        """测试无已调度任务时的单个候选"""
        candidate = BatchOrbitConstraintCandidate(
            sat_id='SAT-001',
            window_start=self.ref_time,
            window_end=self.ref_time + timedelta(seconds=60),
            imaging_duration=60.0
        )

        results = self.checker.check_batch([candidate], [])

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].feasible)
        self.assertEqual(results[0].current_starts_in_orbit, 1)
        self.assertEqual(results[0].current_work_time_in_orbit, 60.0)

    def test_max_starts_exceeded(self):
        """测试超过单圈最大开机次数"""
        orbit_period = self.sat1.orbit.get_period()  # ~5676秒

        # 候选任务开始时间
        candidate_start = self.ref_time + timedelta(seconds=500)

        # 创建3个已调度任务，都在候选的滑动圈内 [500, 500+5676]
        existing_tasks = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': self.ref_time + timedelta(seconds=500 + i * 200),
                'imaging_end': self.ref_time + timedelta(seconds=500 + i * 200 + 60)
            }
            for i in range(3)
        ]

        # 第4个候选任务（应该失败，因为已经有3个任务在圈内）
        candidate = BatchOrbitConstraintCandidate(
            sat_id='SAT-001',
            window_start=candidate_start,
            window_end=candidate_start + timedelta(seconds=60),
            imaging_duration=60.0
        )

        results = self.checker.check_batch([candidate], existing_tasks)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].feasible)
        self.assertIn('Max starts per orbit exceeded', results[0].reason)
        self.assertEqual(results[0].current_starts_in_orbit, 4)  # 3 existing + 1 candidate

    def test_max_work_time_exceeded(self):
        """测试超过单圈最大工作时长"""
        # 候选任务开始时间（也是滑动圈起点）
        candidate_start = self.ref_time + timedelta(seconds=500)

        # 创建2个已调度任务，每个150秒，都与候选的滑动圈重叠
        # 任务1: [500, 650]，任务2: [600, 750]
        # 与候选滑动圈 [candidate_start, candidate_start+5676] 都重叠
        # 总计300秒（已达到 max_work_time=300 限制）
        existing_tasks = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': candidate_start + timedelta(seconds=0),
                'imaging_end': candidate_start + timedelta(seconds=150)
            },
            {
                'satellite_id': 'SAT-001',
                'imaging_start': candidate_start + timedelta(seconds=100),
                'imaging_end': candidate_start + timedelta(seconds=250)
            }
        ]

        # 候选任务在滑动圈起点，与上述任务重叠
        # 总计360秒 > max_work_time=300，应该失败
        candidate = BatchOrbitConstraintCandidate(
            sat_id='SAT-001',
            window_start=candidate_start,  # 与滑动圈起点相同
            window_end=candidate_start + timedelta(seconds=60),
            imaging_duration=60.0
        )

        results = self.checker.check_batch([candidate], existing_tasks)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].feasible)
        self.assertIn('Max work time per orbit exceeded', results[0].reason)

    def test_sliding_orbit_boundary(self):
        """测试滑动圈边界"""
        orbit_period = int(self.sat1.orbit.get_period())  # ~5676秒

        # 在第一个圈末尾创建任务
        existing_tasks = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': self.ref_time + timedelta(seconds=orbit_period - 100),
                'imaging_end': self.ref_time + timedelta(seconds=orbit_period - 40)
            }
        ]

        # 候选任务在第二个圈开始（应该成功，因为滑动圈不同）
        candidate = BatchOrbitConstraintCandidate(
            sat_id='SAT-001',
            window_start=self.ref_time + timedelta(seconds=orbit_period + 10),
            window_end=self.ref_time + timedelta(seconds=orbit_period + 70),
            imaging_duration=60.0
        )

        results = self.checker.check_batch([candidate], existing_tasks)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].feasible)
        # 候选任务的滑动圈从 orbit_period+10 开始
        # 已调度任务在 orbit_period-100 到 orbit_period-40，不在滑动圈内
        self.assertEqual(results[0].current_starts_in_orbit, 1)

    def test_cross_orbit_task(self):
        """测试跨圈任务的统计（滑动圈定义）"""
        orbit_period = int(self.sat1.orbit.get_period())

        # 候选任务在第二圈开始
        candidate_start = self.ref_time + timedelta(seconds=orbit_period + 100)
        candidate = BatchOrbitConstraintCandidate(
            sat_id='SAT-001',
            window_start=candidate_start,
            window_end=candidate_start + timedelta(seconds=60),
            imaging_duration=60.0
        )

        # 情况1：跨圈任务与候选滑动圈重叠
        # 跨圈任务从第一圈末尾开始，延伸到第二圈
        existing_tasks_overlap = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': self.ref_time + timedelta(seconds=orbit_period - 50),
                'imaging_end': self.ref_time + timedelta(seconds=orbit_period + 150)
            }
        ]

        results = self.checker.check_batch([candidate], existing_tasks_overlap)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].feasible)
        # 跨圈任务与候选滑动圈 [candidate_start, candidate_start+period] 重叠
        self.assertEqual(results[0].current_starts_in_orbit, 2)  # 1 existing + 1 candidate

        # 情况2：跨圈任务不与候选滑动圈重叠
        existing_tasks_no_overlap = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': self.ref_time + timedelta(seconds=orbit_period - 200),
                'imaging_end': self.ref_time + timedelta(seconds=orbit_period - 100)
            }
        ]

        results = self.checker.check_batch([candidate], existing_tasks_no_overlap)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].feasible)
        # 跨圈任务不与滑动圈重叠
        self.assertEqual(results[0].current_starts_in_orbit, 1)  # only candidate

    def test_different_satellites(self):
        """测试不同卫星独立统计"""
        # SAT-001有3个任务
        existing_tasks = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': self.ref_time + timedelta(seconds=i * 100),
                'imaging_end': self.ref_time + timedelta(seconds=i * 100 + 60)
            }
            for i in range(3)
        ]

        # SAT-002的候选任务（应该成功，因为约束独立）
        candidate = BatchOrbitConstraintCandidate(
            sat_id='SAT-002',
            window_start=self.ref_time,
            window_end=self.ref_time + timedelta(seconds=60),
            imaging_duration=60.0
        )

        results = self.checker.check_batch([candidate], existing_tasks)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].feasible)
        self.assertEqual(results[0].current_starts_in_orbit, 1)

    def test_batch_multiple_candidates(self):
        """测试批量检查多个候选（候选之间相互独立）"""
        # 候选任务开始时间（所有候选共享）
        candidate_start = self.ref_time + timedelta(seconds=500)

        # 已有2个任务在候选的滑动圈内 [500, 500+5676]
        existing_tasks = [
            {
                'satellite_id': 'SAT-001',
                'imaging_start': candidate_start + timedelta(seconds=100),
                'imaging_end': candidate_start + timedelta(seconds=160)
            },
            {
                'satellite_id': 'SAT-001',
                'imaging_start': candidate_start + timedelta(seconds=200),
                'imaging_end': candidate_start + timedelta(seconds=260)
            }
        ]

        # 批量检查5个候选（max_starts=3，已有2个，加上自己正好是3个）
        candidates = [
            BatchOrbitConstraintCandidate(
                sat_id='SAT-001',
                window_start=candidate_start + timedelta(seconds=i * 10),
                window_end=candidate_start + timedelta(seconds=i * 10 + 60),
                imaging_duration=60.0
            )
            for i in range(5)
        ]

        results = self.checker.check_batch(candidates, existing_tasks)

        self.assertEqual(len(results), 5)
        # 所有候选在同一个滑动圈内（因为开始时间接近），每个都看到2个existing任务
        # 所以每个候选加上自己是3个，应该都通过（max_starts=3）
        for i, result in enumerate(results):
            self.assertTrue(result.feasible, f"Candidate {i} should be feasible")
            self.assertEqual(result.current_starts_in_orbit, 3)  # 2 existing + 1 self


if __name__ == '__main__':
    unittest.main()
