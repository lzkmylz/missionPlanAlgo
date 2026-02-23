"""
可视化扩展功能测试

TDD流程：
1. 先写测试（RED）
2. 实现代码（GREEN）
3. 重构优化（REFACTOR）

测试覆盖：
- GroundTrackVisualizer - 轨道星下点轨迹图
- ResourceHeatmapVisualizer - 资源利用率热力图
- AlgorithmComparisonVisualizer - 多算法对比图表
- CoverageMap - 覆盖分析图
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


# =============================================================================
# GroundTrackVisualizer Tests
# =============================================================================

class TestGroundTrackVisualizer:
    """星下点轨迹可视化器测试"""

    def test_visualizer_creation(self):
        """测试可视化器对象创建"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()
        assert visualizer is not None
        assert visualizer.figsize == (14, 8)

    def test_visualizer_creation_with_custom_figsize(self):
        """测试使用自定义figsize创建可视化器"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer(figsize=(10, 6))
        assert visualizer.figsize == (10, 6)

    def test_plot_with_valid_data(self):
        """测试使用有效数据绘制星下点轨迹"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        # 准备测试数据
        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, 117.0, 118.0, 119.0],
                'lat': [39.0, 40.0, 41.0, 42.0],
                'times': [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i*10)
                          for i in range(4)]
            }
        }

        observation_points = {
            'SAT-01': [
                {'lon': 117.5, 'lat': 39.5, 'time': datetime(2024, 1, 1, 0, 20)}
            ]
        }

        fig = visualizer.plot(ground_tracks, observation_points)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_multiple_satellites(self):
        """测试多卫星轨迹绘制"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, 117.0, 118.0],
                'lat': [39.0, 40.0, 41.0],
                'times': [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i*10)
                          for i in range(3)]
            },
            'SAT-02': {
                'lon': [120.0, 121.0, 122.0],
                'lat': [35.0, 36.0, 37.0],
                'times': [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i*10)
                          for i in range(3)]
            }
        }

        fig = visualizer.plot(ground_tracks, {})
        assert fig is not None
        plt.close(fig)

    def test_plot_with_empty_data(self):
        """测试空数据处理"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        # 空轨迹数据
        fig = visualizer.plot({}, {})
        assert fig is not None
        plt.close(fig)

    def test_plot_with_single_point(self):
        """测试单点数据处理"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0],
                'lat': [39.0],
                'times': [datetime(2024, 1, 1, 0, 0)]
            }
        }

        fig = visualizer.plot(ground_tracks, {})
        assert fig is not None
        plt.close(fig)

    def test_save_method(self, tmp_path):
        """测试保存功能"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, 117.0],
                'lat': [39.0, 40.0],
                'times': [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)]
            }
        }

        fig = visualizer.plot(ground_tracks, {})
        output_path = tmp_path / "ground_track.png"

        visualizer.save(fig, str(output_path))
        assert output_path.exists()
        plt.close(fig)

    def test_plot_with_crossing_antimeridian(self):
        """测试跨越日界线（经度180度）的轨迹"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [179.0, 179.5, -179.5, -179.0],
                'lat': [39.0, 40.0, 41.0, 42.0],
                'times': [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i*10)
                          for i in range(4)]
            }
        }

        fig = visualizer.plot(ground_tracks, {})
        assert fig is not None
        plt.close(fig)


# =============================================================================
# ResourceHeatmapVisualizer Tests
# =============================================================================

class TestResourceHeatmapVisualizer:
    """资源利用率热力图可视化器测试"""

    def test_visualizer_creation(self):
        """测试可视化器对象创建"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()
        assert visualizer is not None
        assert visualizer.figsize == (12, 8)

    def test_visualizer_creation_with_custom_figsize(self):
        """测试使用自定义figsize创建"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer(figsize=(10, 6))
        assert visualizer.figsize == (10, 6)

    def test_plot_with_valid_data(self):
        """测试使用有效数据绘制热力图"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        # 准备测试数据：时间-资源二维数据
        time_slots = [f"T{i:02d}" for i in range(24)]
        resources = ['SAT-01', 'SAT-02', 'SAT-03']

        # 利用率矩阵 (资源 x 时间)
        utilization = np.random.rand(3, 24) * 100

        fig = visualizer.plot(utilization, resources, time_slots)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_empty_data(self):
        """测试空数据处理"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        fig = visualizer.plot(
            np.array([]).reshape(0, 0),
            [],
            []
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_single_resource(self):
        """测试单资源数据处理"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(12)]
        resources = ['SAT-01']
        utilization = np.random.rand(1, 12) * 100

        fig = visualizer.plot(utilization, resources, time_slots)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_full_utilization(self):
        """测试100%利用率情况"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(6)]
        resources = ['SAT-01', 'SAT-02']
        utilization = np.ones((2, 6)) * 100

        fig = visualizer.plot(utilization, resources, time_slots)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_zero_utilization(self):
        """测试0%利用率情况"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(6)]
        resources = ['SAT-01', 'SAT-02']
        utilization = np.zeros((2, 6))

        fig = visualizer.plot(utilization, resources, time_slots)
        assert fig is not None
        plt.close(fig)

    def test_save_method(self, tmp_path):
        """测试保存功能"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(6)]
        resources = ['SAT-01']
        utilization = np.random.rand(1, 6) * 100

        fig = visualizer.plot(utilization, resources, time_slots)
        output_path = tmp_path / "resource_heatmap.png"

        visualizer.save(fig, str(output_path))
        assert output_path.exists()
        plt.close(fig)

    def test_plot_with_mismatched_dimensions(self):
        """测试维度不匹配处理"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(10)]  # 10个时间槽
        resources = ['SAT-01', 'SAT-02']  # 2个资源
        utilization = np.random.rand(2, 8)  # 但数据是8个时间槽

        with pytest.raises(ValueError):
            visualizer.plot(utilization, resources, time_slots)


# =============================================================================
# AlgorithmComparisonVisualizer Tests
# =============================================================================

class TestAlgorithmComparisonVisualizer:
    """多算法对比图表可视化器测试"""

    def test_visualizer_creation(self):
        """测试可视化器对象创建"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()
        assert visualizer is not None
        assert visualizer.figsize == (14, 10)

    def test_plot_box_comparison(self):
        """测试箱线图对比"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        # 准备测试数据：多算法多次运行的结果
        results = {
            'Greedy': [85.2, 86.1, 84.9, 85.5, 85.8],
            'GA': [92.3, 93.1, 91.8, 92.7, 92.9],
            'SA': [90.5, 91.2, 89.8, 90.9, 91.1],
            'ACO': [93.5, 94.2, 93.1, 93.8, 94.0]
        }

        fig = visualizer.plot_box_comparison(results, metric_name="完成率(%)")
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_convergence_curves(self):
        """测试收敛曲线展示"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        # 准备收敛曲线数据
        convergence_data = {
            'GA': [75.0, 82.0, 87.0, 90.0, 92.0, 92.5, 92.8, 93.0],
            'SA': [80.0, 85.0, 88.0, 89.5, 90.0, 90.2, 90.3, 90.3],
            'ACO': [70.0, 80.0, 88.0, 91.0, 93.0, 94.0, 94.2, 94.2]
        }

        fig = visualizer.plot_convergence_curves(convergence_data)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_box_comparison_with_empty_data(self):
        """测试空数据箱线图"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        fig = visualizer.plot_box_comparison({}, metric_name="完成率(%)")
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_with_single_point(self):
        """测试单点收敛曲线"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        convergence_data = {
            'GA': [92.0],
            'SA': [90.0]
        }

        fig = visualizer.plot_convergence_curves(convergence_data)
        assert fig is not None
        plt.close(fig)

    def test_plot_box_comparison_with_single_algorithm(self):
        """测试单算法箱线图"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        results = {
            'Greedy': [85.2, 86.1, 84.9, 85.5, 85.8]
        }

        fig = visualizer.plot_box_comparison(results, metric_name="完成率(%)")
        assert fig is not None
        plt.close(fig)

    def test_save_method(self, tmp_path):
        """测试保存功能"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        results = {
            'Greedy': [85.2, 86.1],
            'GA': [92.3, 93.1]
        }

        fig = visualizer.plot_box_comparison(results, metric_name="完成率(%)")
        output_path = tmp_path / "algorithm_comparison.png"

        visualizer.save(fig, str(output_path))
        assert output_path.exists()
        plt.close(fig)

    def test_plot_radar_chart(self):
        """测试雷达图展示"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        # 多维度性能指标
        metrics = {
            'Greedy': {'完成率': 85, '速度': 95, '稳定性': 90},
            'GA': {'完成率': 93, '速度': 60, '稳定性': 85},
            'SA': {'完成率': 91, '速度': 70, '稳定性': 88}
        }

        fig = visualizer.plot_radar_chart(metrics)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_radar_chart_with_empty_data(self):
        """测试空数据雷达图"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        fig = visualizer.plot_radar_chart({})
        assert fig is not None
        plt.close(fig)


# =============================================================================
# CoverageMap Tests
# =============================================================================

class TestCoverageMap:
    """覆盖分析图测试"""

    def test_visualizer_creation(self):
        """测试可视化器对象创建"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()
        assert visualizer is not None
        assert visualizer.figsize == (12, 10)

    def test_visualizer_creation_with_custom_figsize(self):
        """测试使用自定义figsize创建"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap(figsize=(10, 8))
        assert visualizer.figsize == (10, 8)

    def test_plot_with_valid_data(self):
        """测试使用有效数据绘制覆盖图"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        # 准备测试数据：网格覆盖次数
        lon_grid = np.linspace(115, 125, 20)  # 经度网格
        lat_grid = np.linspace(35, 45, 20)    # 纬度网格

        # 覆盖次数矩阵
        coverage_count = np.random.randint(0, 10, (20, 20))

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_target_markers(self):
        """测试带目标标记的覆盖图"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 20)
        lat_grid = np.linspace(35, 45, 20)
        coverage_count = np.random.randint(0, 10, (20, 20))

        # 目标位置
        targets = [
            {'lon': 118.0, 'lat': 39.0, 'id': 'T001'},
            {'lon': 120.0, 'lat': 40.0, 'id': 'T002'}
        ]

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid, targets=targets)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_empty_coverage(self):
        """测试零覆盖情况"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 10)
        lat_grid = np.linspace(35, 45, 10)
        coverage_count = np.zeros((10, 10))

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_full_coverage(self):
        """测试全覆盖情况"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 10)
        lat_grid = np.linspace(35, 45, 10)
        coverage_count = np.ones((10, 10)) * 20  # 全部20次覆盖

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_single_cell(self):
        """测试单网格覆盖"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.array([120.0])
        lat_grid = np.array([40.0])
        coverage_count = np.array([[5]])

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        plt.close(fig)

    def test_save_method(self, tmp_path):
        """测试保存功能"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 10)
        lat_grid = np.linspace(35, 45, 10)
        coverage_count = np.random.randint(0, 5, (10, 10))

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        output_path = tmp_path / "coverage_map.png"

        visualizer.save(fig, str(output_path))
        assert output_path.exists()
        plt.close(fig)

    def test_plot_with_mismatched_grid(self):
        """测试网格维度不匹配处理"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 10)
        lat_grid = np.linspace(35, 45, 10)
        coverage_count = np.random.randint(0, 5, (8, 12))  # 维度不匹配

        with pytest.raises(ValueError):
            visualizer.plot(coverage_count, lon_grid, lat_grid)

    def test_plot_with_contour_lines(self):
        """测试带等高线的覆盖图"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 30)
        lat_grid = np.linspace(35, 45, 30)

        # 创建梯度覆盖数据
        x, y = np.meshgrid(lon_grid, lat_grid)
        coverage_count = np.exp(-((x - 120)**2 + (y - 40)**2) / 50) * 10

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid, show_contours=True)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Integration Tests
# =============================================================================

class TestVisualizerIntegration:
    """可视化器集成测试"""

    def test_all_visualizers_importable(self):
        """测试所有可视化器可导入"""
        from visualization.ground_track import GroundTrackVisualizer
        from visualization.resource_heatmap import ResourceHeatmapVisualizer
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer
        from visualization.coverage_map import CoverageMap

        assert GroundTrackVisualizer is not None
        assert ResourceHeatmapVisualizer is not None
        assert AlgorithmComparisonVisualizer is not None
        assert CoverageMap is not None

    def test_visualizer_consistent_interface(self):
        """测试可视化器接口一致性"""
        from visualization.ground_track import GroundTrackVisualizer
        from visualization.resource_heatmap import ResourceHeatmapVisualizer
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer
        from visualization.coverage_map import CoverageMap

        visualizers = [
            GroundTrackVisualizer(),
            ResourceHeatmapVisualizer(),
            AlgorithmComparisonVisualizer(),
            CoverageMap()
        ]

        # 所有可视化器应该有save方法
        for viz in visualizers:
            assert hasattr(viz, 'save')
            assert callable(getattr(viz, 'save'))

    def test_save_with_invalid_path(self):
        """测试无效路径保存处理"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, 117.0],
                'lat': [39.0, 40.0],
                'times': [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)]
            }
        }

        fig = visualizer.plot(ground_tracks, {})

        with pytest.raises((IOError, OSError)):
            visualizer.save(fig, "/nonexistent/directory/file.png")

        plt.close(fig)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_ground_track_with_none_values(self):
        """测试包含None值的星下点数据"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, None, 118.0],
                'lat': [39.0, 40.0, None],
                'times': [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 10),
                    datetime(2024, 1, 1, 0, 20)
                ]
            }
        }

        # 应该处理None值而不崩溃
        fig = visualizer.plot(ground_tracks, {})
        assert fig is not None
        plt.close(fig)

    def test_resource_heatmap_with_nan_values(self):
        """测试包含NaN值的热力图数据"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(6)]
        resources = ['SAT-01', 'SAT-02']
        utilization = np.array([
            [50.0, np.nan, 70.0, 80.0, 90.0, 100.0],
            [60.0, 70.0, np.nan, 85.0, 95.0, 100.0]
        ])

        # 应该处理NaN值而不崩溃
        fig = visualizer.plot(utilization, resources, time_slots)
        assert fig is not None
        plt.close(fig)

    def test_coverage_map_with_negative_values(self):
        """测试覆盖图负值处理"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 10)
        lat_grid = np.linspace(35, 45, 10)
        coverage_count = np.random.randint(-5, 5, (10, 10))

        # 负值应该被处理（覆盖次数不能为负）
        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        plt.close(fig)

    def test_algorithm_comparison_with_different_lengths(self):
        """测试不同长度收敛曲线"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        convergence_data = {
            'GA': [75.0, 82.0, 87.0, 90.0, 92.0],
            'SA': [80.0, 85.0, 88.0],  # 更短的曲线
            'ACO': [70.0, 80.0, 88.0, 91.0, 93.0, 94.0, 94.5]  # 更长的曲线
        }

        fig = visualizer.plot_convergence_curves(convergence_data)
        assert fig is not None
        plt.close(fig)

    def test_ground_track_with_extreme_coordinates(self):
        """测试极值坐标"""
        from visualization.ground_track import GroundTrackVisualizer

        visualizer = GroundTrackVisualizer()

        ground_tracks = {
            'SAT-01': {
                'lon': [-180.0, 0.0, 180.0],
                'lat': [-90.0, 0.0, 90.0],
                'times': [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 0, 10),
                    datetime(2024, 1, 1, 0, 20)
                ]
            }
        }

        fig = visualizer.plot(ground_tracks, {})
        assert fig is not None
        plt.close(fig)

    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        # 大型网格
        lon_grid = np.linspace(70, 140, 200)  # 中国范围
        lat_grid = np.linspace(15, 55, 200)
        coverage_count = np.random.randint(0, 50, (200, 200))

        fig = visualizer.plot(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Extended Tests for Higher Coverage
# =============================================================================

class TestResourceHeatmapExtended:
    """资源热力图扩展测试"""

    def test_plot_multi_resource(self):
        """测试多资源类型热力图"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(12)]

        utilization_dict = {
            '存储': np.random.rand(2, 12) * 100,
            '电量': np.random.rand(2, 12) * 100
        }

        fig = visualizer.plot_multi_resource(utilization_dict, time_slots)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_multi_resource_single_type(self):
        """测试单资源类型多资源热力图"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(6)]

        utilization_dict = {
            '存储': np.random.rand(1, 6) * 100
        }

        fig = visualizer.plot_multi_resource(utilization_dict, time_slots)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_show_values(self):
        """测试显示数值的热力图"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(4)]
        resources = ['SAT-01', 'SAT-02']
        utilization = np.array([[50.0, 60.0, 70.0, 80.0],
                                [55.0, 65.0, 75.0, 85.0]])

        fig = visualizer.plot(utilization, resources, time_slots, show_values=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_vmin_vmax(self):
        """测试自定义颜色范围"""
        from visualization.resource_heatmap import ResourceHeatmapVisualizer

        visualizer = ResourceHeatmapVisualizer()

        time_slots = [f"T{i:02d}" for i in range(4)]
        resources = ['SAT-01']
        utilization = np.array([[50.0, 60.0, 70.0, 80.0]])

        fig = visualizer.plot(utilization, resources, time_slots, vmin=0, vmax=200)
        assert fig is not None
        plt.close(fig)


class TestAlgorithmComparisonExtended:
    """算法对比扩展测试"""

    def test_plot_pareto_front(self):
        """测试Pareto前沿图"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        pareto_data = {
            'GA': [[10, 20], [15, 15], [20, 10]],
            'NSGA-II': [[8, 22], [12, 18], [18, 12], [22, 8]]
        }

        fig = visualizer.plot_pareto_front(
            pareto_data,
            objective_names=['成本', '时间']
        )
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pareto_front_empty(self):
        """测试空Pareto数据"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        fig = visualizer.plot_pareto_front({}, objective_names=['成本', '时间'])
        assert fig is not None
        plt.close(fig)

    def test_plot_pareto_front_with_empty_points(self):
        """测试包含空点的Pareto数据"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        pareto_data = {
            'GA': [[10, 20], [15, 15]],
            'Empty': []
        }

        fig = visualizer.plot_pareto_front(pareto_data)
        assert fig is not None
        plt.close(fig)

    def test_plot_radar_chart_single_dimension(self):
        """测试单维度雷达图"""
        from visualization.algorithm_comparison import AlgorithmComparisonVisualizer

        visualizer = AlgorithmComparisonVisualizer()

        metrics = {
            'Greedy': {'完成率': 85},
            'GA': {'完成率': 93}
        }

        fig = visualizer.plot_radar_chart(metrics)
        assert fig is not None
        plt.close(fig)


class TestCoverageMapExtended:
    """覆盖分析图扩展测试"""

    def test_plot_coverage_statistics(self):
        """测试覆盖统计分析图"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 20)
        lat_grid = np.linspace(35, 45, 20)
        coverage_count = np.random.randint(0, 10, (20, 20))

        fig = visualizer.plot_coverage_statistics(coverage_count, lon_grid, lat_grid)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_coverage_gap_analysis(self):
        """测试覆盖缺口分析图"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 20)
        lat_grid = np.linspace(35, 45, 20)
        coverage_count = np.random.randint(0, 5, (20, 20))

        fig = visualizer.plot_coverage_gap_analysis(
            coverage_count, lon_grid, lat_grid, threshold=2
        )
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_coverage_gap_no_gaps(self):
        """测试无缺口情况"""
        from visualization.coverage_map import CoverageMap

        visualizer = CoverageMap()

        lon_grid = np.linspace(115, 125, 10)
        lat_grid = np.linspace(35, 45, 10)
        coverage_count = np.ones((10, 10)) * 10  # 全部高覆盖

        fig = visualizer.plot_coverage_gap_analysis(
            coverage_count, lon_grid, lat_grid, threshold=1
        )
        assert fig is not None
        plt.close(fig)


class TestGroundTrackPlotterCompatibility:
    """GroundTrackPlotter兼容性测试"""

    def test_plotter_compatibility(self):
        """测试旧版GroundTrackPlotter兼容接口"""
        from visualization.ground_track import GroundTrackPlotter

        plotter = GroundTrackPlotter()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, 117.0, 118.0],
                'lat': [39.0, 40.0, 41.0],
                'times': [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i*10)
                          for i in range(3)]
            }
        }

        fig = plotter.plot(ground_tracks, {})
        assert fig is not None
        plt.close(fig)

    def test_plotter_save(self):
        """测试旧版Plotter保存功能"""
        from visualization.ground_track import GroundTrackPlotter
        import tempfile
        import os

        plotter = GroundTrackPlotter()

        ground_tracks = {
            'SAT-01': {
                'lon': [116.0, 117.0],
                'lat': [39.0, 40.0],
                'times': [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)]
            }
        }

        fig = plotter.plot(ground_tracks, {})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.png")
            plotter.save(fig, output_path)
            assert os.path.exists(output_path)
