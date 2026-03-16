"""
区域目标Benchmark工具模块单元测试

测试覆盖:
1. load_area_scenario 场景加载
2. create_tile_targets tile目标创建
3. calculate_visibility_windows 可见性计算
4. 区域边界提取功能
"""

import unittest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from scripts.area_benchmark_utils import (
    load_area_scenario,
    create_tile_targets,
    calculate_visibility_windows,
    _extract_scenario_bounds,
    get_scenario_bounds,
    clear_scenario_bounds,
    _scenario_bounds_cache
)
from core.models import Mission, Target, Satellite
from core.models.target import TargetType


class TestExtractScenarioBounds(unittest.TestCase):
    """测试场景边界提取功能"""

    def test_extract_bounds_from_area_target(self):
        """测试从区域目标提取边界"""
        scenario_data = {
            'targets': [
                {
                    'target_type': 'area',
                    'area_vertices': [
                        [120.0, 22.0],
                        [122.0, 22.0],
                        [122.0, 25.0],
                        [120.0, 25.0]
                    ]
                }
            ]
        }

        bounds = _extract_scenario_bounds(scenario_data)

        # 边界应该包含所有顶点（含10%缓冲区）
        self.assertLess(bounds['min_lon'], 120.0)
        self.assertGreater(bounds['max_lon'], 122.0)
        self.assertLess(bounds['min_lat'], 22.0)
        self.assertGreater(bounds['max_lat'], 25.0)

    def test_extract_bounds_no_area_target(self):
        """测试无区域目标时返回全球范围"""
        scenario_data = {
            'targets': [
                {'target_type': 'point', 'longitude': 120.0, 'latitude': 23.0}
            ]
        }

        bounds = _extract_scenario_bounds(scenario_data)

        self.assertEqual(bounds['min_lon'], -180.0)
        self.assertEqual(bounds['max_lon'], 180.0)
        self.assertEqual(bounds['min_lat'], -90.0)
        self.assertEqual(bounds['max_lat'], 90.0)

    def test_extract_bounds_empty_vertices(self):
        """测试空顶点列表"""
        scenario_data = {
            'targets': [
                {'target_type': 'area', 'area_vertices': []}
            ]
        }

        bounds = _extract_scenario_bounds(scenario_data)

        # 应该返回全球范围
        self.assertEqual(bounds['min_lon'], -180.0)

    def test_extract_bounds_buffer_calculation(self):
        """测试边界缓冲区计算"""
        scenario_data = {
            'targets': [
                {
                    'target_type': 'area',
                    'area_vertices': [
                        [120.0, 22.0],
                        [121.0, 22.0],  # 1度跨度
                        [121.0, 23.0],
                        [120.0, 23.0]
                    ]
                }
            ]
        }

        bounds = _extract_scenario_bounds(scenario_data)

        # 10%缓冲区 = 0.1度
        self.assertAlmostEqual(bounds['min_lon'], 119.9, places=5)
        self.assertAlmostEqual(bounds['max_lon'], 121.1, places=5)


class TestScenarioBoundsCache(unittest.TestCase):
    """测试场景边界缓存"""

    def setUp(self):
        """清除缓存"""
        clear_scenario_bounds()

    def tearDown(self):
        """清理"""
        clear_scenario_bounds()

    def test_get_bounds_without_cache(self):
        """测试无缓存时返回全球范围"""
        bounds = get_scenario_bounds()

        self.assertEqual(bounds['min_lon'], -180.0)
        self.assertEqual(bounds['max_lon'], 180.0)

    def test_clear_scenario_bounds(self):
        """测试清除缓存"""
        # 先设置一些状态
        global _scenario_bounds_cache
        import scripts.area_benchmark_utils as abu
        abu._scenario_bounds_cache = {'min_lon': 100.0}

        clear_scenario_bounds()

        self.assertIsNone(abu._scenario_bounds_cache)


class TestLoadAreaScenario(unittest.TestCase):
    """测试场景加载功能"""

    def _create_test_scenario(self):
        """创建测试场景数据"""
        return {
            'name': 'Test Scenario',
            'description': 'Test',
            'duration': {
                'start': '2024-03-15T00:00:00Z',
                'end': '2024-03-16T00:00:00Z'
            },
            'satellites': [
                {
                    'id': 'SAT-01',
                    'name': 'Test Sat',
                    'sat_type': 'optical',
                    'orbit': {
                        'orbit_type': 'LEO',
                        'semi_major_axis': 6871000.0,
                        'eccentricity': 0.0,
                        'inclination': 55.0,
                        'raan': 0.0,
                        'arg_of_perigee': 0.0,
                        'mean_anomaly': 0.0,
                        'epoch': '2024-03-15T00:00:00Z'
                    },
                    'capabilities': {
                        'max_roll_angle': 35.0,
                        'max_pitch_angle': 20.0,
                        'storage_capacity': 128.0,
                        'power_capacity': 2000.0,
                        'data_rate': 300.0
                    }
                }
            ],
            'targets': [
                {
                    'id': 'AREA-001',
                    'name': 'Test Area',
                    'target_type': 'area',
                    'area_vertices': [
                        [120.0, 22.0],
                        [121.0, 22.0],
                        [121.0, 23.0],
                        [120.0, 23.0]
                    ],
                    'priority': 1,
                    'required_observations': 1,
                    'resolution_required': 10.0,
                    'mosaic_required': True,
                    'min_coverage_ratio': 0.95,
                    'max_overlap_ratio': 0.15
                }
            ]
        }

    def test_load_area_scenario(self):
        """测试场景加载"""
        scenario_data = self._create_test_scenario()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scenario_data, f)
            temp_path = f.name

        try:
            mission, area_target, tiles = load_area_scenario(temp_path)

            # 验证Mission对象
            self.assertIsInstance(mission, Mission)
            self.assertEqual(mission.name, 'Test Scenario')
            self.assertEqual(len(mission.satellites), 1)

            # 验证区域目标
            self.assertIsInstance(area_target, Target)
            self.assertEqual(area_target.id, 'AREA-001')
            self.assertEqual(area_target.target_type, TargetType.AREA)

            # 验证tiles
            self.assertGreater(len(tiles), 0)

            # 验证边界缓存被设置
            bounds = get_scenario_bounds()
            self.assertNotEqual(bounds['min_lon'], -180.0)  # 应该被更新

        finally:
            os.unlink(temp_path)
            clear_scenario_bounds()

    def test_load_area_scenario_no_area_target(self):
        """测试无区域目标时抛出异常"""
        scenario_data = self._create_test_scenario()
        scenario_data['targets'][0]['target_type'] = 'point'
        scenario_data['targets'][0]['longitude'] = 120.0
        scenario_data['targets'][0]['latitude'] = 23.0
        del scenario_data['targets'][0]['area_vertices']

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scenario_data, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                load_area_scenario(temp_path)

            self.assertIn('未找到区域目标', str(context.exception))

        finally:
            os.unlink(temp_path)


class TestCreateTileTargets(unittest.TestCase):
    """测试创建tile目标"""

    def setUp(self):
        """准备测试数据"""
        self.mission = Mock(spec=Mission)
        self.mission.add_target = Mock()

        # 创建模拟tiles
        self.tiles = [
            Mock(tile_id='T001', center=(120.0, 23.0)),
            Mock(tile_id='T002', center=(120.1, 23.0)),
        ]

        self.area_target = Mock(spec=Target)
        self.area_target.priority = 1
        self.area_target.resolution_required = 10.0

    def test_create_tile_targets(self):
        """测试创建tile目标"""
        tile_targets = create_tile_targets(self.mission, self.tiles, self.area_target)

        # 验证返回的目标数量
        self.assertEqual(len(tile_targets), 2)

        # 验证目标属性
        self.assertEqual(tile_targets[0].id, 'T001')
        self.assertEqual(tile_targets[0].longitude, 120.0)
        self.assertEqual(tile_targets[0].latitude, 23.0)
        self.assertEqual(tile_targets[0].target_type, TargetType.POINT)

        # 验证添加到mission
        self.assertEqual(self.mission.add_target.call_count, 2)


class TestCalculateVisibilityWindows(unittest.TestCase):
    """测试可见性窗口计算"""

    def setUp(self):
        """准备测试数据"""
        self.mission = Mock(spec=Mission)
        self.mission.start_time = datetime(2024, 3, 15, 0, 0, 0)
        self.mission.end_time = datetime(2024, 3, 16, 0, 0, 0)

        # 创建模拟卫星
        sat = Mock(spec=Satellite)
        sat.id = 'SAT-01'
        sat.orbit = Mock()
        sat.orbit.semi_major_axis = 6871000.0
        sat.orbit.mean_anomaly = 0.0
        sat.orbit.raan = 0.0
        self.mission.satellites = [sat]

        # 创建模拟目标 - 在区域范围内
        target1 = Mock(spec=Target)
        target1.id = 'T001'
        target1.longitude = 120.5  # 在默认测试区域内
        target1.latitude = 22.5

        # 创建模拟目标 - 在区域范围外
        target2 = Mock(spec=Target)
        target2.id = 'T002'
        target2.longitude = 150.0  # 在区域外
        target2.latitude = 50.0

        self.tile_targets = [target1, target2]

    def test_calculate_visibility_windows(self):
        """测试可见性窗口计算"""
        # 设置全局边界（模拟已加载场景）
        import scripts.area_benchmark_utils as abu
        abu._scenario_bounds_cache = {
            'min_lon': 119.0,
            'max_lon': 122.0,
            'min_lat': 21.0,
            'max_lat': 24.0
        }

        try:
            cache = calculate_visibility_windows(self.mission, self.tile_targets)

            # 验证返回了缓存对象
            self.assertIsNotNone(cache)

            # T001在范围内，应该有窗口
            # T002在范围外，应该没有窗口

        finally:
            clear_scenario_bounds()

    def test_calculate_visibility_windows_no_orbit(self):
        """测试卫星无轨道数据"""
        self.mission.satellites[0].orbit = None

        cache = calculate_visibility_windows(self.mission, self.tile_targets)

        # 应该正常返回，但没有窗口
        self.assertIsNotNone(cache)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_end_to_end_workflow(self):
        """测试完整工作流程"""
        scenario_data = {
            'name': 'Integration Test',
            'description': 'Test',
            'duration': {
                'start': '2024-03-15T00:00:00Z',
                'end': '2024-03-15T06:00:00Z'  # 6小时场景
            },
            'satellites': [
                {
                    'id': 'SAT-01',
                    'name': 'Test Sat',
                    'sat_type': 'optical',
                    'orbit': {
                        'orbit_type': 'LEO',
                        'semi_major_axis': 6871000.0,
                        'eccentricity': 0.0,
                        'inclination': 55.0,
                        'raan': 0.0,
                        'arg_of_perigee': 0.0,
                        'mean_anomaly': 0.0,
                        'epoch': '2024-03-15T00:00:00Z'
                    },
                    'capabilities': {
                        'max_roll_angle': 35.0,
                        'max_pitch_angle': 20.0,
                        'storage_capacity': 128.0,
                        'power_capacity': 2000.0,
                        'data_rate': 300.0,
                        'agility': {
                            'max_slew_rate': 3.0,
                            'max_roll_rate': 3.0,
                            'max_pitch_rate': 2.0,
                            'settling_time': 5.0
                        }
                    }
                }
            ],
            'targets': [
                {
                    'id': 'AREA-001',
                    'name': 'Test Area',
                    'target_type': 'area',
                    'area_vertices': [
                        [120.0, 22.0],
                        [120.5, 22.0],
                        [120.5, 22.5],
                        [120.0, 22.5]
                    ],
                    'priority': 1,
                    'required_observations': 1,
                    'resolution_required': 10.0,
                    'mosaic_required': True,
                    'min_coverage_ratio': 0.95,
                    'max_overlap_ratio': 0.15
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scenario_data, f)
            temp_path = f.name

        try:
            # 步骤1: 加载场景
            mission, area_target, tiles = load_area_scenario(temp_path)
            self.assertIsNotNone(mission)
            self.assertIsNotNone(area_target)
            self.assertGreater(len(tiles), 0)

            # 步骤2: 创建tile目标
            tile_targets = create_tile_targets(mission, tiles, area_target)
            self.assertEqual(len(tile_targets), len(tiles))

            # 步骤3: 计算可见性窗口
            cache = calculate_visibility_windows(mission, tile_targets)
            self.assertIsNotNone(cache)

            # 验证边界被正确提取
            bounds = get_scenario_bounds()
            self.assertLess(bounds['min_lon'], 120.0)
            self.assertGreater(bounds['max_lon'], 120.5)

        finally:
            os.unlink(temp_path)
            clear_scenario_bounds()


if __name__ == '__main__':
    unittest.main()
