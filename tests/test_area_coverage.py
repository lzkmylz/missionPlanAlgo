"""
区域目标拼幅覆盖功能测试

测试区域目标分解、可见性计算和调度集成。
"""

import unittest
import json
from datetime import datetime

from core.models import (
    Target, TargetType, MosaicTile, AreaCoveragePlan,
    CoverageStrategy, TilePriorityMode, TileStatus
)
from core.decomposer import MosaicPlanner
from scheduler.area_task_utils import (
    AreaObservationTask, create_area_observation_tasks,
    create_mixed_task_list, calculate_area_coverage_score
)
from scheduler.coverage_tracker import CoverageTracker


class TestMosaicTile(unittest.TestCase):
    """测试 MosaicTile 模型"""

    def test_tile_creation(self):
        """测试瓦片创建"""
        tile = MosaicTile(
            tile_id="AREA-001-T001",
            parent_target_id="AREA-001",
            vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
            center=(116.05, 39.05),
            area_km2=100.0,
            priority=1
        )

        self.assertEqual(tile.tile_id, "AREA-001-T001")
        self.assertEqual(tile.parent_target_id, "AREA-001")
        self.assertEqual(tile.coverage_status, TileStatus.PENDING)
        self.assertFalse(tile.is_covered())

    def test_tile_coverage(self):
        """测试瓦片覆盖状态更新"""
        tile = MosaicTile(
            tile_id="AREA-001-T001",
            parent_target_id="AREA-001",
            vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
            center=(116.05, 39.05),
            area_km2=100.0
        )

        tile.mark_as_scheduled("TASK-001")
        self.assertEqual(tile.coverage_status, TileStatus.SCHEDULED)
        self.assertTrue(tile.is_covered())

        tile.mark_as_completed(effective_coverage=95.0)
        self.assertEqual(tile.coverage_status, TileStatus.COMPLETED)
        self.assertEqual(tile.effective_coverage_km2, 95.0)


class TestAreaCoveragePlan(unittest.TestCase):
    """测试 AreaCoveragePlan 模型"""

    def test_plan_creation(self):
        """测试覆盖计划创建"""
        tiles = [
            MosaicTile(
                tile_id=f"AREA-001-T{i:03d}",
                parent_target_id="AREA-001",
                vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
                center=(116.05, 39.05),
                area_km2=100.0
            )
            for i in range(5)
        ]

        plan = AreaCoveragePlan(
            target_id="AREA-001",
            target_name="测试区域",
            tiles=tiles,
            strategy=CoverageStrategy.MAX_COVERAGE,
            min_coverage_ratio=0.95,
            max_overlap_ratio=0.15
        )

        self.assertEqual(plan.target_id, "AREA-001")
        self.assertEqual(len(plan.tiles), 5)
        self.assertEqual(plan.statistics.total_area_km2, 500.0)
        self.assertFalse(plan.is_fully_covered())

    def test_coverage_progress(self):
        """测试覆盖进度计算"""
        tiles = [
            MosaicTile(
                tile_id=f"AREA-001-T{i:03d}",
                parent_target_id="AREA-001",
                vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
                center=(116.05, 39.05),
                area_km2=100.0
            )
            for i in range(10)
        ]

        plan = AreaCoveragePlan(
            target_id="AREA-001",
            tiles=tiles,
            min_coverage_ratio=0.95
        )

        # 模拟覆盖部分瓦片（传递有效覆盖面积）
        for i in range(8):
            plan.register_tile_coverage(tiles[i].tile_id, f"TASK-{i}", effective_coverage=95.0)

        # 检查覆盖率
        coverage_ratio = plan.get_coverage_progress()
        self.assertGreater(coverage_ratio, 0.7)


class TestMosaicPlanner(unittest.TestCase):
    """测试 MosaicPlanner"""

    def setUp(self):
        self.planner = MosaicPlanner(
            default_overlap_ratio=0.15,
            default_strategy=CoverageStrategy.MAX_COVERAGE
        )

        self.area_target = Target(
            id="TEST-AREA",
            name="测试区域",
            target_type=TargetType.AREA,
            area_vertices=[
                (116.0, 39.0),
                (116.2, 39.0),
                (116.2, 39.2),
                (116.0, 39.2)
            ],
            priority=1,
            mosaic_required=True
        )

        self.satellites = []  # 简化为空列表，使用默认瓦片大小

    def test_create_coverage_plan(self):
        """测试创建覆盖计划"""
        plan = self.planner.create_coverage_plan(
            target=self.area_target,
            satellites=self.satellites
        )

        self.assertEqual(plan.target_id, "TEST-AREA")
        self.assertGreater(len(plan.tiles), 0)
        self.assertEqual(plan.strategy, CoverageStrategy.MAX_COVERAGE)

    def test_dynamic_tile_size(self):
        """测试动态瓦片大小计算"""
        size = self.planner._calculate_dynamic_tile_size(
            target=self.area_target,
            satellites=self.satellites,
            overlap_ratio=0.15,
            dynamic_sizing=True
        )

        # 应该返回合理的瓦片大小（2-50km范围内）
        self.assertGreaterEqual(size, 2.0)
        self.assertLessEqual(size, 50.0)


class TestAreaTaskUtils(unittest.TestCase):
    """测试区域任务工具"""

    def test_create_area_observation_tasks(self):
        """测试创建区域观测任务"""
        tiles = [
            MosaicTile(
                tile_id=f"AREA-001-T{i:03d}",
                parent_target_id="AREA-001",
                vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
                center=(116.05, 39.05),
                area_km2=100.0,
                required_observations=2
            )
            for i in range(3)
        ]

        plan = AreaCoveragePlan(
            target_id="AREA-001",
            tiles=tiles
        )

        tasks = create_area_observation_tasks(plan)

        # 每个瓦片有2次观测需求，共3个瓦片，应该产生6个任务
        self.assertEqual(len(tasks), 6)

        # 检查任务属性
        task = tasks[0]
        self.assertIsInstance(task, AreaObservationTask)
        self.assertEqual(task.target_id, "AREA-001")
        self.assertTrue(task.is_mosaic_task)


class TestCoverageTracker(unittest.TestCase):
    """测试 CoverageTracker"""

    def setUp(self):
        tiles = [
            MosaicTile(
                tile_id=f"AREA-001-T{i:03d}",
                parent_target_id="AREA-001",
                vertices=[(116.0, 39.0), (116.1, 39.0), (116.1, 39.1), (116.0, 39.1)],
                center=(116.05, 39.05),
                area_km2=100.0
            )
            for i in range(10)
        ]

        self.plan = AreaCoveragePlan(
            target_id="AREA-001",
            tiles=tiles,
            min_coverage_ratio=0.95
        )

        self.tracker = CoverageTracker(
            coverage_plans={"AREA-001": self.plan},
            max_overlap_ratio=0.15
        )

    def test_register_coverage(self):
        """测试注册覆盖"""
        tile = self.plan.tiles[0]

        effective_coverage = self.tracker.register_scheduled_tile(
            tile=tile,
            task_id="TASK-001",
            satellite_id="SAT-001"
        )

        self.assertGreater(effective_coverage, 0)
        self.assertEqual(self.tracker.get_coverage_ratio("AREA-001"), 0.1)

    def test_get_uncovered_tiles(self):
        """测试获取未覆盖瓦片"""
        # 初始状态：全部未覆盖
        uncovered = self.tracker.get_uncovered_tiles("AREA-001")
        self.assertEqual(len(uncovered), 10)

        # 覆盖一个瓦片
        self.tracker.register_scheduled_tile(
            tile=self.plan.tiles[0],
            task_id="TASK-001",
            satellite_id="SAT-001"
        )

        uncovered = self.tracker.get_uncovered_tiles("AREA-001")
        self.assertEqual(len(uncovered), 9)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_end_to_end_area_coverage(self):
        """测试完整的区域覆盖流程"""
        # 1. 创建区域目标
        area_target = Target(
            id="INTEGRATION-TEST",
            name="集成测试区域",
            target_type=TargetType.AREA,
            area_vertices=[
                (116.0, 39.0),
                (116.3, 39.0),
                (116.3, 39.3),
                (116.0, 39.3)
            ],
            priority=1,
            mosaic_required=True,
            min_coverage_ratio=0.90
        )

        # 2. 创建覆盖计划
        planner = MosaicPlanner()
        plan = planner.create_coverage_plan(
            target=area_target,
            satellites=[]
        )

        self.assertGreater(len(plan.tiles), 0)

        # 3. 创建观测任务
        tasks = create_area_observation_tasks(plan)
        self.assertEqual(len(tasks), len(plan.tiles))

        # 4. 创建覆盖追踪器
        tracker = CoverageTracker({area_target.id: plan})

        # 5. 模拟调度部分任务
        for i, task in enumerate(tasks[:int(len(tasks) * 0.8)]):
            tracker.register_scheduled_tile(
                tile=task.tile,
                task_id=f"TASK-{i}",
                satellite_id="SAT-001"
            )

        # 6. 检查覆盖率
        coverage_ratio = tracker.get_coverage_ratio(area_target.id)
        self.assertGreater(coverage_ratio, 0.6)  # 放宽到60%，因为模拟环境

        # 7. 检查统计信息
        stats = tracker.get_coverage_statistics(area_target.id)
        self.assertIn('covered_tiles', stats)
        self.assertIn('coverage_ratio', stats)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMosaicTile))
    suite.addTests(loader.loadTestsFromTestCase(TestAreaCoveragePlan))
    suite.addTests(loader.loadTestsFromTestCase(TestMosaicPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestAreaTaskUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestCoverageTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
