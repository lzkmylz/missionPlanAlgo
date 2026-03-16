"""
区域目标评估指标模块单元测试

测试覆盖:
1. AreaCoverageMetrics 数据类
2. MosaicEfficiencyMetrics 数据类
3. ResourceUtilizationMetrics 数据类
4. AreaMetricsCalculator 计算方法
5. PreciseOverlapCalculator 精确重叠计算
6. generate_area_comparison_report 报告生成
"""

import unittest
import math
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import Mock, MagicMock, patch

# 被测模块
from evaluation.area_metrics import (
    AreaCoverageMetrics,
    MosaicEfficiencyMetrics,
    ResourceUtilizationMetrics,
    AreaMetricsCalculator,
    PreciseOverlapCalculator,
    generate_area_comparison_report
)
from core.models.mosaic_tile import MosaicTile
from core.models import Mission, Target
from scheduler.base_scheduler import ScheduleResult


class TestAreaCoverageMetrics(unittest.TestCase):
    """测试区域覆盖指标数据类"""

    def test_default_values(self):
        """测试默认值"""
        metrics = AreaCoverageMetrics()
        self.assertEqual(metrics.total_tiles, 0)
        self.assertEqual(metrics.covered_tiles, 0)
        self.assertEqual(metrics.coverage_ratio, 0.0)
        self.assertEqual(metrics.min_required_coverage, 0.95)
        self.assertEqual(metrics.total_area_km2, 0.0)

    def test_to_dict(self):
        """测试转换为字典"""
        metrics = AreaCoverageMetrics(
            total_tiles=100,
            covered_tiles=95,
            coverage_ratio=0.95,
            total_area_km2=1000.0,
            covered_area_km2=950.0,
            first_task_time=datetime(2024, 3, 15, 10, 0, 0),
            last_task_time=datetime(2024, 3, 15, 14, 0, 0)
        )
        result = metrics.to_dict()

        self.assertEqual(result['total_tiles'], 100)
        self.assertEqual(result['covered_tiles'], 95)
        self.assertEqual(result['coverage_ratio'], 0.95)
        self.assertEqual(result['first_task_time'], '2024-03-15T10:00:00')

    def test_to_dict_with_none_datetime(self):
        """测试None时间字段的字典转换"""
        metrics = AreaCoverageMetrics()
        result = metrics.to_dict()

        self.assertIsNone(result['first_task_time'])
        self.assertIsNone(result['last_task_time'])


class TestMosaicEfficiencyMetrics(unittest.TestCase):
    """测试拼幅效率指标数据类"""

    def test_default_values(self):
        """测试默认值"""
        metrics = MosaicEfficiencyMetrics()
        self.assertEqual(metrics.avg_overlap_ratio, 0.0)
        self.assertEqual(metrics.satellite_switches, 0)
        self.assertEqual(metrics.avg_time_between_tasks_min, 0.0)

    def test_to_dict(self):
        """测试转换为字典"""
        metrics = MosaicEfficiencyMetrics(
            avg_overlap_ratio=0.15,
            satellite_switches=5,
            avg_tasks_per_satellite=12.5
        )
        result = metrics.to_dict()

        self.assertEqual(result['avg_overlap_ratio'], 0.15)
        self.assertEqual(result['satellite_switches'], 5)
        self.assertEqual(result['avg_tasks_per_satellite'], 12.5)


class TestResourceUtilizationMetrics(unittest.TestCase):
    """测试资源利用指标数据类"""

    def test_default_values(self):
        """测试默认值"""
        metrics = ResourceUtilizationMetrics()
        self.assertEqual(metrics.avg_storage_used_gb, 0.0)
        self.assertEqual(metrics.avg_power_consumption_w, 0.0)
        self.assertEqual(metrics.avg_slew_angle_deg, 0.0)

    def test_to_dict(self):
        """测试转换为字典"""
        metrics = ResourceUtilizationMetrics(
            avg_storage_used_gb=50.0,
            max_storage_used_gb=100.0,
            avg_slew_angle_deg=15.5
        )
        result = metrics.to_dict()

        self.assertEqual(result['avg_storage_used_gb'], 50.0)
        self.assertEqual(result['max_storage_used_gb'], 100.0)
        self.assertEqual(result['avg_slew_angle_deg'], 15.5)


class MockMosaicTile:
    """模拟MosaicTile对象"""

    def __init__(
        self,
        tile_id: str,
        center: tuple,
        area_km2: float,
        bounds: Optional[tuple] = None,
        vertices: Optional[List[List[float]]] = None
    ):
        self.tile_id = tile_id
        self.center = center
        self.area_km2 = area_km2
        self.bounds = bounds
        self.vertices = vertices or []


class MockTask:
    """模拟任务对象"""

    def __init__(
        self,
        task_id: str,
        tile_id: Optional[str] = None,
        target_id: Optional[str] = None,
        satellite_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        self.task_id = task_id
        self.tile_id = tile_id
        self.target_id = target_id
        self.satellite_id = satellite_id
        self.start_time = start_time
        self.end_time = end_time


class TestAreaMetricsCalculator(unittest.TestCase):
    """测试区域指标计算器"""

    def setUp(self):
        """测试前准备"""
        self.mission = Mock(spec=Mission)
        self.tiles = [
            MockMosaicTile('T001', (120.0, 23.0), 100.0),
            MockMosaicTile('T002', (120.1, 23.0), 100.0),
            MockMosaicTile('T003', (120.2, 23.0), 100.0),
        ]
        self.calculator = AreaMetricsCalculator(self.mission, self.tiles)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.calculator.mission, self.mission)
        self.assertEqual(len(self.calculator.tiles), 3)

    def test_calculate_coverage_metrics_empty_schedule(self):
        """测试空调度结果的覆盖指标"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = []

        metrics = self.calculator.calculate_coverage_metrics(schedule_result)

        self.assertEqual(metrics.total_tiles, 3)
        self.assertEqual(metrics.covered_tiles, 0)
        self.assertEqual(metrics.coverage_ratio, 0.0)
        self.assertEqual(metrics.total_area_km2, 300.0)

    def test_calculate_coverage_metrics_full_coverage(self):
        """测试完全覆盖的指标计算"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', tile_id='T001', satellite_id='SAT-01'),
            MockTask('task2', tile_id='T002', satellite_id='SAT-01'),
            MockTask('task3', tile_id='T003', satellite_id='SAT-02'),
        ]
        schedule_result.unscheduled_tasks = {}

        metrics = self.calculator.calculate_coverage_metrics(schedule_result)

        self.assertEqual(metrics.total_tiles, 3)
        self.assertEqual(metrics.covered_tiles, 3)
        self.assertEqual(metrics.coverage_ratio, 1.0)
        self.assertEqual(metrics.covered_area_km2, 300.0)
        self.assertEqual(metrics.area_coverage_ratio, 1.0)

    def test_calculate_coverage_metrics_partial_coverage(self):
        """测试部分覆盖的指标计算"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', tile_id='T001', satellite_id='SAT-01'),
        ]
        schedule_result.unscheduled_tasks = {}

        metrics = self.calculator.calculate_coverage_metrics(schedule_result)

        self.assertEqual(metrics.total_tiles, 3)
        self.assertEqual(metrics.covered_tiles, 1)
        self.assertAlmostEqual(metrics.coverage_ratio, 1/3, places=5)
        self.assertEqual(metrics.covered_area_km2, 100.0)
        self.assertAlmostEqual(metrics.area_coverage_ratio, 1/3, places=5)

    def test_calculate_coverage_metrics_with_target_id_matching(self):
        """测试使用target_id匹配tile"""
        schedule_result = Mock(spec=ScheduleResult)
        # 使用target_id而不是tile_id
        schedule_result.scheduled_tasks = [
            MockTask('task1', target_id='T001', satellite_id='SAT-01'),
        ]
        schedule_result.unscheduled_tasks = {}

        tile_targets = [Mock(id='T001'), Mock(id='T002'), Mock(id='T003')]

        metrics = self.calculator.calculate_coverage_metrics(
            schedule_result, tile_targets
        )

        self.assertEqual(metrics.covered_tiles, 1)

    def test_calculate_coverage_metrics_with_obs_suffix(self):
        """测试带-OBS后缀的任务ID匹配"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', target_id='T001-OBS1', satellite_id='SAT-01'),
        ]
        schedule_result.unscheduled_tasks = {}

        tile_targets = [Mock(id='T001'), Mock(id='T002')]

        metrics = self.calculator.calculate_coverage_metrics(
            schedule_result, tile_targets
        )

        self.assertEqual(metrics.covered_tiles, 1)

    def test_satellite_contribution_calculation(self):
        """测试卫星贡献度计算"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', tile_id='T001', satellite_id='SAT-01'),
            MockTask('task2', tile_id='T002', satellite_id='SAT-01'),
            MockTask('task3', tile_id='T003', satellite_id='SAT-02'),
        ]
        schedule_result.unscheduled_tasks = {}

        metrics = self.calculator.calculate_coverage_metrics(schedule_result)

        # SAT-01: 2 tiles, SAT-02: 1 tile
        self.assertEqual(metrics.tiles_per_satellite.get('SAT-01'), 2)
        self.assertEqual(metrics.tiles_per_satellite.get('SAT-02'), 1)

        # Area contribution
        self.assertEqual(metrics.area_per_satellite_km2.get('SAT-01'), 200.0)
        self.assertEqual(metrics.area_per_satellite_km2.get('SAT-02'), 100.0)

        # Contribution ratio
        self.assertAlmostEqual(
            metrics.coverage_contribution_ratio.get('SAT-01'), 2/3, places=5
        )
        self.assertAlmostEqual(
            metrics.coverage_contribution_ratio.get('SAT-02'), 1/3, places=5
        )

    def test_coverage_makespan_calculation(self):
        """测试覆盖时间跨度计算"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', tile_id='T001', start_time=datetime(2024, 3, 15, 10, 0, 0)),
            MockTask('task2', tile_id='T002', start_time=datetime(2024, 3, 15, 12, 0, 0)),
            MockTask('task3', tile_id='T003', start_time=datetime(2024, 3, 15, 14, 0, 0)),
        ]
        schedule_result.unscheduled_tasks = {}

        metrics = self.calculator.calculate_coverage_metrics(schedule_result)

        self.assertEqual(metrics.first_task_time, datetime(2024, 3, 15, 10, 0, 0))
        self.assertEqual(metrics.last_task_time, datetime(2024, 3, 15, 14, 0, 0))
        self.assertEqual(metrics.coverage_makespan_hours, 4.0)

    def test_calculate_efficiency_metrics(self):
        """测试效率指标计算"""
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', satellite_id='SAT-01',
                    start_time=datetime(2024, 3, 15, 10, 0, 0),
                    end_time=datetime(2024, 3, 15, 10, 5, 0)),
            MockTask('task2', satellite_id='SAT-02',
                    start_time=datetime(2024, 3, 15, 10, 10, 0),
                    end_time=datetime(2024, 3, 15, 10, 15, 0)),
            MockTask('task3', satellite_id='SAT-01',
                    start_time=datetime(2024, 3, 15, 10, 20, 0),
                    end_time=datetime(2024, 3, 15, 10, 25, 0)),
        ]

        metrics = self.calculator.calculate_efficiency_metrics(schedule_result)

        self.assertEqual(metrics.satellite_switches, 2)  # SAT-01 -> SAT-02 -> SAT-01
        self.assertEqual(metrics.avg_tasks_per_satellite, 1.5)  # 3 tasks / 2 sats

    def test_calculate_resource_metrics(self):
        """测试资源指标计算"""
        class ResourceTask:
            def __init__(self, storage, power, roll, pitch):
                self.storage_used_gb = storage
                self.power_consumed_wh = power
                self.roll_angle = roll
                self.pitch_angle = pitch

        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            ResourceTask(50.0, 100.0, 15.0, 10.0),
            ResourceTask(75.0, 150.0, 20.0, 5.0),
            ResourceTask(100.0, 200.0, 25.0, 15.0),
        ]

        metrics = self.calculator.calculate_resource_metrics(schedule_result)

        self.assertEqual(metrics.avg_storage_used_gb, 75.0)
        self.assertEqual(metrics.max_storage_used_gb, 100.0)
        self.assertEqual(metrics.avg_power_consumption_w, 150.0)
        self.assertEqual(metrics.avg_slew_angle_deg, 15.0)  # Average of all angles
        self.assertEqual(metrics.max_slew_angle_deg, 25.0)


class TestPreciseOverlapCalculator(unittest.TestCase):
    """测试精确重叠计算器"""

    def setUp(self):
        """测试前准备"""
        self.calculator = PreciseOverlapCalculator(use_shapely=False)

    def test_no_overlap_far_apart(self):
        """测试远离的瓦片无重叠"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (130.0, 33.0), 100.0)

        overlap = self.calculator.calculate_overlap(tile1, tile2)
        self.assertEqual(overlap, 0.0)

    def test_no_overlap_touching_edge(self):
        """测试边缘接触无重叠"""
        # 10km x 10km tiles, side = 10km
        # 10km = 0.09 degrees approximately
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (120.18, 23.0), 100.0)  # ~20km apart

        overlap = self.calculator.calculate_overlap(tile1, tile2)
        self.assertEqual(overlap, 0.0)

    def test_full_containment(self):
        """测试完全包含的情况"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (120.0, 23.0), 25.0)  # Smaller, same center

        overlap = self.calculator.calculate_overlap(tile1, tile2)
        # Smaller tile fully contained, overlap = smaller area
        self.assertAlmostEqual(overlap, 25.0, delta=5.0)

    def test_partial_overlap(self):
        """测试部分重叠"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        # 5km offset (half of side length ~10km)
        tile2 = MockMosaicTile('T002', (120.045, 23.0), 100.0)

        overlap = self.calculator.calculate_overlap(tile1, tile2)
        # Should have some overlap
        self.assertGreater(overlap, 0.0)
        self.assertLess(overlap, 100.0)

    def test_zero_area_tile(self):
        """测试零面积瓦片"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (120.0, 23.0), 0.0)

        overlap = self.calculator.calculate_overlap(tile1, tile2)
        self.assertEqual(overlap, 0.0)

    def test_cache_mechanism(self):
        """测试缓存机制"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (120.0, 23.0), 50.0)

        # First calculation
        overlap1 = self.calculator.calculate_overlap(tile1, tile2)

        # Second calculation (should use cache)
        overlap2 = self.calculator.calculate_overlap(tile1, tile2)

        self.assertEqual(overlap1, overlap2)

    def test_cache_key_order_independence(self):
        """测试缓存键顺序无关"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (120.0, 23.0), 50.0)

        overlap1 = self.calculator.calculate_overlap(tile1, tile2)
        overlap2 = self.calculator.calculate_overlap(tile2, tile1)

        self.assertEqual(overlap1, overlap2)

    def test_clear_cache(self):
        """测试清除缓存"""
        tile1 = MockMosaicTile('T001', (120.0, 23.0), 100.0)
        tile2 = MockMosaicTile('T002', (120.0, 23.0), 50.0)

        self.calculator.calculate_overlap(tile1, tile2)
        self.assertGreater(len(self.calculator._cache), 0)

        self.calculator.clear_cache()
        self.assertEqual(len(self.calculator._cache), 0)

    def test_calculate_all_overlaps(self):
        """测试批量计算重叠"""
        tiles = [
            MockMosaicTile('T001', (120.0, 23.0), 100.0),
            MockMosaicTile('T002', (120.0, 23.0), 50.0),
            MockMosaicTile('T003', (125.0, 28.0), 100.0),  # Far away
        ]

        results = self.calculator.calculate_all_overlaps(tiles)

        # T001 and T002 should overlap (same center) - cache key is normalized
        self.assertTrue(
            ('T001', 'T002') in results or ('T002', 'T001') in results,
            "Expected overlap between T001 and T002"
        )

        # T003 should not overlap with others
        for key in results or {}:
            self.assertNotIn('T003', key)

    def test_spatial_index_optimization(self):
        """测试空间索引优化"""
        # Create many tiles
        tiles = []
        for i in range(100):
            tiles.append(MockMosaicTile(
                f'T{i:03d}',
                (120.0 + i * 0.1, 23.0),
                100.0
            ))

        results = self.calculator.calculate_all_overlaps(tiles)

        # Should complete without error
        self.assertIsInstance(results, dict)


class TestTileOverlapEdgeCases(unittest.TestCase):
    """测试瓦片重叠的边界情况"""

    def test_tile_overlap_division_by_zero(self):
        """测试除零保护"""
        from evaluation.area_metrics import AreaMetricsCalculator

        mission = Mock(spec=Mission)
        tiles = [
            MockMosaicTile('T001', (120.0, 23.0), 0.0),
            MockMosaicTile('T002', (120.0, 23.0), 0.0),
        ]
        calculator = AreaMetricsCalculator(mission, tiles)

        # Should not raise exception
        overlap = calculator._calculate_tile_overlap(tiles[0], tiles[1])
        self.assertEqual(overlap, 0.0)

    def test_tile_overlap_with_negative_area(self):
        """测试负面积处理"""
        from evaluation.area_metrics import AreaMetricsCalculator

        mission = Mock(spec=Mission)
        tiles = [
            MockMosaicTile('T001', (120.0, 23.0), -10.0),
            MockMosaicTile('T002', (120.0, 23.0), 100.0),
        ]
        calculator = AreaMetricsCalculator(mission, tiles)

        overlap = calculator._calculate_tile_overlap(tiles[0], tiles[1])
        self.assertEqual(overlap, 0.0)


class TestGenerateAreaComparisonReport(unittest.TestCase):
    """测试报告生成"""

    def test_generate_report(self):
        """测试报告生成"""
        results = {
            'Greedy': {
                'coverage': {
                    'coverage_ratio': 0.95,
                    'covered_tiles': 95,
                    'total_tiles': 100,
                    'area_coverage_ratio': 0.94,
                },
                'efficiency': {
                    'satellite_switches': 5,
                    'avg_tasks_per_satellite': 10.0,
                }
            },
            'GA': {
                'coverage': {
                    'coverage_ratio': 0.98,
                    'covered_tiles': 98,
                    'total_tiles': 100,
                    'area_coverage_ratio': 0.97,
                },
                'efficiency': {
                    'satellite_switches': 3,
                    'avg_tasks_per_satellite': 12.0,
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            generate_area_comparison_report(results, temp_path)

            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))

            # Verify content
            import json
            with open(temp_path, 'r') as f:
                report = json.load(f)

            self.assertEqual(report['scenario_type'], 'area_target')
            self.assertIn('Greedy', report['comparison']['coverage'])
            self.assertIn('GA', report['comparison']['coverage'])
            self.assertEqual(report['best_algorithm']['coverage'], 'GA')

        finally:
            os.unlink(temp_path)


class TestIntegrationMetricsCalculation(unittest.TestCase):
    """集成测试：完整指标计算流程"""

    def test_full_calculation_pipeline(self):
        """测试完整计算流程"""
        mission = Mock(spec=Mission)
        tiles = [
            MockMosaicTile('T001', (120.0, 23.0), 100.0),
            MockMosaicTile('T002', (120.05, 23.0), 100.0),
            MockMosaicTile('T003', (120.1, 23.0), 100.0),
        ]

        calculator = AreaMetricsCalculator(mission, tiles)

        # Create schedule result
        schedule_result = Mock(spec=ScheduleResult)
        schedule_result.scheduled_tasks = [
            MockTask('task1', tile_id='T001', satellite_id='SAT-01',
                    start_time=datetime(2024, 3, 15, 10, 0, 0)),
            MockTask('task2', tile_id='T002', satellite_id='SAT-01',
                    start_time=datetime(2024, 3, 15, 11, 0, 0)),
            MockTask('task3', tile_id='T003', satellite_id='SAT-02',
                    start_time=datetime(2024, 3, 15, 12, 0, 0)),
        ]
        schedule_result.unscheduled_tasks = {}

        # Calculate all metrics
        result = calculator.calculate_all(schedule_result)

        # Verify structure
        self.assertIn('coverage', result)
        self.assertIn('efficiency', result)
        self.assertIn('resource', result)
        self.assertIn('summary', result)

        # Verify coverage metrics
        coverage = result['coverage']
        self.assertEqual(coverage['total_tiles'], 3)
        self.assertEqual(coverage['covered_tiles'], 3)
        self.assertEqual(coverage['coverage_ratio'], 1.0)

        # Verify summary
        summary = result['summary']
        self.assertEqual(summary['total_tiles'], 3)
        self.assertEqual(summary['scheduled_tasks'], 3)


if __name__ == '__main__':
    unittest.main()
