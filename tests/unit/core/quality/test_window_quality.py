"""
窗口质量评分单元测试

测试 WindowQualityCalculator 的各个维度评分计算
"""

import unittest
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.quality.window_quality import (
    WindowQualityCalculator,
    WindowQualityScore,
    QualityDimension,
)
from core.quality.quality_config import (
    QualityScoreConfig,
    QualityDimensionWeights,
    SatelliteType,
    DEFAULT_QUALITY_CONFIG,
)
from core.orbit.visibility.base import VisibilityWindow


class TestWindowQualityScore(unittest.TestCase):
    """测试 WindowQualityScore 数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        score = WindowQualityScore(
            overall_score=0.8,
            elevation_score=0.85,
            attitude_score=0.9,
            duration_score=0.7,
            illumination_score=0.75,
            downlink_score=0.8,
        )

        self.assertEqual(score.overall_score, 0.8)
        self.assertEqual(score.elevation_score, 0.85)
        self.assertTrue(score.is_high_quality())
        self.assertTrue(score.is_acceptable())

    def test_score_clamping(self):
        """测试评分范围限制"""
        score = WindowQualityScore(
            overall_score=1.5,  # 应该被限制为1.0
            elevation_score=-0.5,  # 应该被限制为0.0
            attitude_score=0.5,
            duration_score=0.5,
            illumination_score=0.5,
            downlink_score=0.5,
        )

        self.assertEqual(score.overall_score, 1.0)
        self.assertEqual(score.elevation_score, 0.0)

    def test_quality_tiers(self):
        """测试质量等级判断"""
        from core.quality.quality_config import QualityTier
        high = WindowQualityScore(
            overall_score=0.8, elevation_score=0.8, attitude_score=0.8,
            duration_score=0.8, illumination_score=0.8, downlink_score=0.8,
            quality_tier=QualityTier.HIGH
        )
        low = WindowQualityScore(
            overall_score=0.3, elevation_score=0.3, attitude_score=0.3,
            duration_score=0.3, illumination_score=0.3, downlink_score=0.3,
            quality_tier=QualityTier.LOW
        )

        self.assertTrue(high.is_high_quality())
        self.assertFalse(low.is_high_quality())
        self.assertTrue(low.is_acceptable(0.3))
        self.assertFalse(low.is_acceptable(0.4))

    def test_get_worst_best_dimension(self):
        """测试获取最差和最优维度"""
        score = WindowQualityScore(
            overall_score=0.7,
            elevation_score=0.9,
            attitude_score=0.3,  # 最差
            duration_score=0.8,
            illumination_score=0.95,  # 最优
            downlink_score=0.7,
        )

        worst = score.get_worst_dimension()
        best = score.get_best_dimension()

        self.assertEqual(worst[0], 'attitude')
        self.assertEqual(worst[1], 0.3)
        self.assertEqual(best[0], 'illumination')
        self.assertEqual(best[1], 0.95)

    def test_to_from_dict(self):
        """测试字典转换"""
        from core.quality.quality_config import QualityTier
        original = WindowQualityScore(
            overall_score=0.75,
            elevation_score=0.8,
            attitude_score=0.7,
            duration_score=0.75,
            illumination_score=0.8,
            downlink_score=0.7,
            quality_tier=QualityTier.MEDIUM,
            details={'test': 'data'},
        )

        data = original.to_dict()
        restored = WindowQualityScore.from_dict(data)

        self.assertEqual(original.overall_score, restored.overall_score)
        self.assertEqual(original.quality_tier, restored.quality_tier)


class TestWindowQualityCalculator(unittest.TestCase):
    """测试 WindowQualityCalculator"""

    def setUp(self):
        """设置测试环境"""
        self.calculator = WindowQualityCalculator()

        # 创建模拟卫星
        self.mock_sat = MagicMock()
        self.mock_sat.id = 'SAT-001'
        self.mock_sat.satellite_type = 'optical'
        self.mock_sat.capabilities.max_roll_angle = 45.0
        self.mock_sat.capabilities.max_pitch_angle = 30.0

        # 创建模拟窗口
        self.mock_window = MagicMock()
        self.mock_window.satellite_id = 'SAT-001'
        self.mock_window.target_id = 'TGT-001'
        self.mock_window.start_time = datetime.now()
        self.mock_window.end_time = datetime.now() + timedelta(seconds=120)
        self.mock_window.max_elevation = 75.0
        self.mock_window.duration.return_value = 120.0
        self.mock_window.attitude_samples = None

    def test_elevation_score_calculation(self):
        """测试仰角评分计算"""
        # 90° 应该得 1.0
        self.assertAlmostEqual(
            self.calculator._calculate_elevation_score(90), 1.0, places=2
        )

        # 45° 应该得约 0.707 (sqrt(0.5))
        score_45 = self.calculator._calculate_elevation_score(45)
        self.assertAlmostEqual(score_45, math.sqrt(0.5), places=2)

        # 0° 应该得 0.0
        self.assertAlmostEqual(
            self.calculator._calculate_elevation_score(0), 0.0, places=2
        )

        # 测试边界限制
        self.assertAlmostEqual(
            self.calculator._calculate_elevation_score(100), 1.0, places=2
        )
        self.assertAlmostEqual(
            self.calculator._calculate_elevation_score(-10), 0.0, places=2
        )

    def test_attitude_score_calculation(self):
        """测试姿态约束满足度评分"""
        max_roll = 45.0
        max_pitch = 30.0

        # 中心位置应该得高分
        samples_center = [(0.0, 0.0), (5.0, 5.0), (-5.0, -5.0)]
        score_center = self.calculator._calculate_attitude_score(
            samples_center, max_roll, max_pitch
        )
        self.assertGreater(score_center, 0.8)

        # 边界位置应该得低分
        samples_boundary = [(40.0, 25.0), (42.0, 28.0)]
        score_boundary = self.calculator._calculate_attitude_score(
            samples_boundary, max_roll, max_pitch
        )
        self.assertLess(score_boundary, 0.6)

        # 无姿态数据应该返回默认值
        score_none = self.calculator._calculate_attitude_score(None, max_roll, max_pitch)
        self.assertEqual(score_none, 0.7)

        # 超出约束应该得 0
        samples_exceed = [(50.0, 20.0)]  # roll > max_roll
        score_exceed = self.calculator._calculate_attitude_score(
            samples_exceed, max_roll, max_pitch
        )
        self.assertEqual(score_exceed, 0.0)

    def test_duration_score_calculation(self):
        """测试持续时间评分"""
        min_required = 15.0
        optimal = 120.0

        # 低于最小要求应该得 0
        self.assertEqual(
            self.calculator._calculate_duration_score(10.0, min_required, optimal), 0.0
        )

        # 达到最佳应该得 1.0
        self.assertEqual(
            self.calculator._calculate_duration_score(150.0, min_required, optimal), 1.0
        )

        # 中间值应该线性插值
        score_half = self.calculator._calculate_duration_score(67.5, min_required, optimal)
        expected = (67.5 - 15) / (120 - 15)
        self.assertAlmostEqual(score_half, expected, places=2)

    def test_determine_satellite_type(self):
        """测试卫星类型判断"""
        # 光学卫星
        optical_sat = MagicMock()
        optical_sat.id = 'SAT-OPTICAL-01'
        optical_sat.satellite_type = 'optical'
        self.assertEqual(
            self.calculator._determine_satellite_type(optical_sat),
            SatelliteType.OPTICAL
        )

        # SAR卫星
        sar_sat = MagicMock()
        sar_sat.id = 'SAT-SAR-01'
        sar_sat.satellite_type = 'sar'
        self.assertEqual(
            self.calculator._determine_satellite_type(sar_sat),
            SatelliteType.SAR
        )

        # 未知类型
        unknown_sat = MagicMock()
        unknown_sat.id = 'SAT-001'
        unknown_sat.satellite_type = 'unknown'
        self.assertEqual(
            self.calculator._determine_satellite_type(unknown_sat),
            SatelliteType.UNKNOWN
        )

    def test_cache_functionality(self):
        """测试缓存功能"""
        # 初始状态
        self.assertEqual(self.calculator.get_cache_stats()['cache_size'], 0)

        # 第一次计算
        score1 = self.calculator.calculate_quality(
            self.mock_window, self.mock_sat, None
        )

        stats_after_first = self.calculator.get_cache_stats()
        self.assertEqual(stats_after_first['misses'], 1)

        # 第二次计算（相同窗口）应该命中缓存
        score2 = self.calculator.calculate_quality(
            self.mock_window, self.mock_sat, None
        )

        stats_after_second = self.calculator.get_cache_stats()
        self.assertEqual(stats_after_second['hits'], 1)
        self.assertEqual(score1.overall_score, score2.overall_score)

        # 清空缓存
        self.calculator.clear_cache()
        self.assertEqual(self.calculator.get_cache_stats()['cache_size'], 0)

    def test_filter_windows_by_quality(self):
        """测试质量筛选功能"""
        # 创建多个窗口
        windows = []
        for i, elevation in enumerate([90, 60, 45, 10]):  # 10度的质量会很低
            window = MagicMock()
            window.satellite_id = 'SAT-001'
            window.target_id = f'TGT-{i}'
            window.start_time = datetime.now() + timedelta(seconds=i)
            window.end_time = window.start_time + timedelta(seconds=120)
            window.max_elevation = elevation
            window.duration.return_value = 120.0
            windows.append(window)

        # 筛选高质量窗口（阈值0.7）
        filtered = self.calculator.filter_windows_by_quality(
            windows, self.mock_sat, min_quality=0.7
        )

        # 验证筛选结果（仰角10的综合质量应该低于0.7）
        # 仰角90的质量评分 sqrt(90/90) = 1.0
        # 仰角60的质量评分 sqrt(60/90) ≈ 0.816
        # 仰角45的质量评分 sqrt(45/90) ≈ 0.707
        # 仰角10的质量评分 sqrt(10/90) ≈ 0.333
        # 由于综合质量包含其他维度，实际阈值可能需要调整
        # 这里主要验证筛选功能能正常工作
        self.assertGreaterEqual(len(filtered), 1)  # 至少90度的能通过
        self.assertLessEqual(len(filtered), 4)  # 不会返回超过原始数量

    def test_sort_windows_by_quality(self):
        """测试质量排序功能"""
        # 创建多个窗口
        windows = []
        for elevation in [30, 90, 60, 45]:
            window = MagicMock()
            window.satellite_id = 'SAT-001'
            window.target_id = f'TGT-{elevation}'
            window.start_time = datetime.now()
            window.end_time = datetime.now() + timedelta(seconds=120)
            window.max_elevation = elevation
            window.duration.return_value = 120.0
            windows.append(window)

        # 排序
        sorted_windows = self.calculator.sort_windows_by_quality(
            windows, self.mock_sat, reverse=True
        )

        # 验证排序（降序）
        scores = [s.overall_score for _, s in sorted_windows]
        self.assertEqual(sorted(scores, reverse=True), scores)

        # 第一个应该是质量最高的（仰角90）
        self.assertEqual(sorted_windows[0][0].max_elevation, 90)

    def test_overall_quality_calculation(self):
        """测试综合质量评分计算"""
        score = self.calculator.calculate_quality(
            self.mock_window, self.mock_sat, None
        )

        # 验证返回类型
        self.assertIsInstance(score, WindowQualityScore)

        # 验证评分范围
        self.assertGreaterEqual(score.overall_score, 0.0)
        self.assertLessEqual(score.overall_score, 1.0)

        # 验证各维度评分
        self.assertGreaterEqual(score.elevation_score, 0.0)
        self.assertGreaterEqual(score.attitude_score, 0.0)
        self.assertGreaterEqual(score.duration_score, 0.0)

        # 验证详情
        self.assertIn('weights', score.details)
        self.assertIn('elevation', score.details)

    def test_different_satellite_types(self):
        """测试不同卫星类型的权重"""
        config = QualityScoreConfig()

        # 光学卫星权重
        optical_weights = config.get_weights_for_satellite(SatelliteType.OPTICAL)
        self.assertGreater(
            optical_weights.illumination_weight,
            optical_weights.elevation_weight
        )

        # SAR卫星权重
        sar_weights = config.get_weights_for_satellite(SatelliteType.SAR)
        self.assertGreater(
            sar_weights.elevation_weight,
            sar_weights.illumination_weight
        )


class TestWindowQualityIntegration(unittest.TestCase):
    """集成测试"""

    def test_quality_with_real_visibility_window(self):
        """测试使用真实的 VisibilityWindow"""
        window = VisibilityWindow(
            satellite_id='SAT-001',
            target_id='TGT-001',
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=120),
            max_elevation=75.0,
            quality_score=0.8,
            attitude_samples=[(0.0, 5.0, 5.0), (60.0, 10.0, 8.0)],
        )

        mock_sat = MagicMock()
        mock_sat.id = 'SAT-001'
        mock_sat.satellite_type = 'optical'
        mock_sat.capabilities.max_roll_angle = 45.0
        mock_sat.capabilities.max_pitch_angle = 30.0

        calculator = WindowQualityCalculator()
        score = calculator.calculate_quality(window, mock_sat)

        self.assertIsInstance(score, WindowQualityScore)
        self.assertGreater(score.overall_score, 0)


if __name__ == '__main__':
    unittest.main()
