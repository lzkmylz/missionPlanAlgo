"""
质量评分配置单元测试

测试 QualityScoreConfig 和相关配置类
"""

import unittest

from core.quality.quality_config import (
    QualityDimensionWeights,
    QualityThresholds,
    QualityScoreConfig,
    SatelliteTypeWeights,
    SatelliteType,
    DEFAULT_QUALITY_CONFIG,
    STRICT_QUALITY_CONFIG,
    LENIENT_QUALITY_CONFIG,
)


class TestQualityDimensionWeights(unittest.TestCase):
    """测试 QualityDimensionWeights"""

    def test_default_weights(self):
        """测试默认权重"""
        weights = QualityDimensionWeights()

        self.assertEqual(weights.elevation_weight, 0.25)
        self.assertEqual(weights.attitude_weight, 0.25)
        self.assertEqual(weights.duration_weight, 0.15)
        self.assertEqual(weights.illumination_weight, 0.20)
        self.assertEqual(weights.downlink_weight, 0.15)

    def test_weight_normalization(self):
        """测试权重自动归一化"""
        # 创建权重和不等于1.0的配置
        weights = QualityDimensionWeights(
            elevation_weight=1.0,
            attitude_weight=1.0,
            duration_weight=1.0,
            illumination_weight=1.0,
            downlink_weight=1.0,
        )

        # 验证归一化（总和应为1.0）
        total = (weights.elevation_weight + weights.attitude_weight +
                 weights.duration_weight + weights.illumination_weight +
                 weights.downlink_weight)
        self.assertAlmostEqual(total, 1.0, places=3)

        # 每个权重应为0.2
        self.assertAlmostEqual(weights.elevation_weight, 0.2, places=3)

    def test_to_from_dict(self):
        """测试字典转换"""
        original = QualityDimensionWeights(
            elevation_weight=0.3,
            attitude_weight=0.3,
            duration_weight=0.2,
            illumination_weight=0.1,
            downlink_weight=0.1,
        )

        data = original.to_dict()
        restored = QualityDimensionWeights.from_dict(data)

        self.assertEqual(original.elevation_weight, restored.elevation_weight)
        self.assertEqual(original.attitude_weight, restored.attitude_weight)


class TestQualityThresholds(unittest.TestCase):
    """测试 QualityThresholds"""

    def test_default_thresholds(self):
        """测试默认阈值"""
        thresholds = QualityThresholds()

        self.assertEqual(thresholds.high_quality, 0.7)
        self.assertEqual(thresholds.medium_quality, 0.4)
        self.assertEqual(thresholds.low_quality, 0.3)

    def test_quality_tier_classification(self):
        """测试质量等级分类"""
        thresholds = QualityThresholds()

        # 高质量
        self.assertEqual(thresholds.get_quality_tier(0.8), 'high')
        self.assertEqual(thresholds.get_quality_tier(0.7), 'high')

        # 中等质量
        self.assertEqual(thresholds.get_quality_tier(0.5), 'medium')
        self.assertEqual(thresholds.get_quality_tier(0.4), 'medium')

        # 低质量
        self.assertEqual(thresholds.get_quality_tier(0.35), 'low')
        self.assertEqual(thresholds.get_quality_tier(0.3), 'low')

        # 不可接受
        self.assertEqual(thresholds.get_quality_tier(0.2), 'unacceptable')
        self.assertEqual(thresholds.get_quality_tier(0.0), 'unacceptable')


class TestQualityScoreConfig(unittest.TestCase):
    """测试 QualityScoreConfig"""

    def test_default_config(self):
        """测试默认配置"""
        config = QualityScoreConfig()

        self.assertIsNotNone(config.weights)
        self.assertIsNotNone(config.optical_weights)
        self.assertIsNotNone(config.sar_weights)
        self.assertIsNotNone(config.thresholds)
        self.assertEqual(config.min_quality_threshold, 0.3)
        self.assertTrue(config.enable_caching)

    def test_satellite_type_weights(self):
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

        # 未知类型使用默认权重
        unknown_weights = config.get_weights_for_satellite(SatelliteType.UNKNOWN)
        self.assertEqual(unknown_weights.elevation_weight, config.weights.elevation_weight)

    def test_to_from_dict(self):
        """测试字典转换"""
        original = QualityScoreConfig(
            min_quality_threshold=0.4,
            enable_caching=False,
            cache_ttl_seconds=7200,
        )

        data = original.to_dict()
        restored = QualityScoreConfig.from_dict(data)

        self.assertEqual(original.min_quality_threshold, restored.min_quality_threshold)
        self.assertEqual(original.enable_caching, restored.enable_caching)
        self.assertEqual(original.cache_ttl_seconds, restored.cache_ttl_seconds)


class TestSatelliteTypeWeights(unittest.TestCase):
    """测试 SatelliteTypeWeights"""

    def test_get_weights_by_type(self):
        """测试按类型获取权重"""
        optical_w = QualityDimensionWeights(illumination_weight=0.4)
        sar_w = QualityDimensionWeights(illumination_weight=0.1)

        type_weights = SatelliteTypeWeights(
            optical_weights=optical_w,
            sar_weights=sar_w,
        )

        # 光学卫星
        self.assertEqual(type_weights.get_weights('optical'), optical_w)
        self.assertEqual(type_weights.get_weights('OPTICAL'), optical_w)
        self.assertEqual(type_weights.get_weights('SAT-OPTICAL-01'), optical_w)

        # SAR卫星
        self.assertEqual(type_weights.get_weights('sar'), sar_w)
        self.assertEqual(type_weights.get_weights('SAR'), sar_w)
        self.assertEqual(type_weights.get_weights('SAT-SAR-01'), sar_w)

        # 未知类型使用默认
        default = type_weights.get_weights('unknown')
        self.assertEqual(default, type_weights.default_weights)


class TestPresetConfigs(unittest.TestCase):
    """测试预设配置"""

    def test_default_quality_config(self):
        """测试默认配置实例"""
        self.assertIsInstance(DEFAULT_QUALITY_CONFIG, QualityScoreConfig)
        self.assertEqual(DEFAULT_QUALITY_CONFIG.min_quality_threshold, 0.3)

    def test_strict_quality_config(self):
        """测试严格配置"""
        self.assertIsInstance(STRICT_QUALITY_CONFIG, QualityScoreConfig)
        self.assertEqual(STRICT_QUALITY_CONFIG.min_quality_threshold, 0.5)
        self.assertEqual(STRICT_QUALITY_CONFIG.thresholds.high_quality, 0.8)

    def test_lenient_quality_config(self):
        """测试宽松配置"""
        self.assertIsInstance(LENIENT_QUALITY_CONFIG, QualityScoreConfig)
        self.assertEqual(LENIENT_QUALITY_CONFIG.min_quality_threshold, 0.2)
        self.assertEqual(LENIENT_QUALITY_CONFIG.thresholds.high_quality, 0.6)


if __name__ == '__main__':
    unittest.main()
