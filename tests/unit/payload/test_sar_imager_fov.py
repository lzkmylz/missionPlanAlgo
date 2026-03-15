"""
SAR成像器FOV配置测试

测试SARImager的FOV配置功能
"""

import pytest
import math
from payload.sar_imager import SARImager, SARImagingMode


class TestSARImagerFOVConfiguration:
    """测试SAR成像器FOV配置"""

    def test_fov_initialization(self):
        """测试FOV参数初始化"""
        imager = SARImager(
            imager_id='SAR-FOV-TEST',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36,
            azimuth_half_angle=0.91,
            azimuth_exclusion_angle=0.3
        )

        assert imager.range_half_angle == 1.36
        assert imager.azimuth_half_angle == 0.91
        assert imager.azimuth_exclusion_angle == 0.3

    def test_fov_optional_parameters(self):
        """测试FOV参数可选"""
        imager = SARImager(
            imager_id='SAR-NO-FOV',
            resolution=3.0,
            swath_width=30.0
        )

        assert imager.range_half_angle is None
        assert imager.azimuth_half_angle is None
        assert imager.azimuth_exclusion_angle is None
        assert not imager.is_fov_configured()

    def test_is_fov_configured(self):
        """测试FOV配置状态检查"""
        # 无FOV配置
        imager_no_fov = SARImager('SAR-1', 3.0, 30.0)
        assert not imager_no_fov.is_fov_configured()

        # 有距离向配置
        imager_range = SARImager('SAR-2', 3.0, 30.0, range_half_angle=1.36)
        assert imager_range.is_fov_configured()

        # 有方位向配置
        imager_azimuth = SARImager('SAR-3', 3.0, 30.0, azimuth_half_angle=0.91)
        assert imager_azimuth.is_fov_configured()

        # 有排除角配置
        imager_exclusion = SARImager('SAR-4', 3.0, 30.0, azimuth_exclusion_angle=0.3)
        assert imager_exclusion.is_fov_configured()


class TestSARImagerFOVCalculation:
    """测试SAR FOV计算"""

    def test_calculate_swath_from_fov(self):
        """测试基于FOV的幅宽计算"""
        imager = SARImager(
            imager_id='SAR-CALC',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36,  # 约30km幅宽在631km高度
            azimuth_half_angle=0.91
        )

        # 在631km高度计算
        swath_range, scene_length = imager.calculate_swath_from_fov(
            altitude_km=631.0,
            look_angle_deg=0.0
        )

        # 验证计算结果（允许一定误差）
        expected_swath = 2 * 631.0 * math.tan(math.radians(1.36))
        expected_length = 2 * 631.0 * math.tan(math.radians(0.91))

        assert abs(swath_range - expected_swath) < 0.1
        assert abs(scene_length - expected_length) < 0.1

    def test_calculate_swath_with_look_angle(self):
        """测试带观测角度的幅宽计算"""
        imager = SARImager(
            imager_id='SAR-LOOK',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36
        )

        # 天底点观测
        swath_nadir, _ = imager.calculate_swath_from_fov(
            altitude_km=631.0,
            look_angle_deg=0.0
        )

        # 30度观测角
        swath_30deg, _ = imager.calculate_swath_from_fov(
            altitude_km=631.0,
            look_angle_deg=30.0
        )

        # 观测角越大，幅宽越大（1/cos(look_angle)因子）
        assert swath_30deg > swath_nadir

    def test_calculate_swath_without_fov(self):
        """测试无FOV配置时回退到swath_width"""
        imager = SARImager(
            imager_id='SAR-NO-FOV',
            resolution=3.0,
            swath_width=25.0
        )

        swath_range, scene_length = imager.calculate_swath_from_fov(
            altitude_km=500.0
        )

        assert swath_range == 25.0  # 使用直接配置的幅宽
        assert scene_length == 25.0  # 默认长度等于幅宽

    def test_azimuth_exclusion_parameter(self):
        """测试方位向排除角参数设置"""
        imager = SARImager(
            imager_id='SAR-EXCL',
            resolution=3.0,
            swath_width=30.0,
            azimuth_half_angle=1.0,
            azimuth_exclusion_angle=0.3
        )

        # 验证排除角参数被正确设置
        assert imager.azimuth_exclusion_angle == 0.3

        # calculate_swath_from_fov 不考虑排除角（影响在足迹计算器层处理）
        _, length = imager.calculate_swath_from_fov(altitude_km=500.0)
        expected = 2 * 500.0 * math.tan(math.radians(1.0))
        assert abs(length - expected) < 0.1


class TestSARImagerEffectiveSwath:
    """测试有效幅宽获取"""

    def test_get_effective_swath_with_fov(self):
        """测试有FOV配置时获取有效幅宽"""
        imager = SARImager(
            imager_id='SAR-EFF',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36
        )

        effective = imager.get_effective_swath(altitude_km=631.0)
        expected = 2 * 631.0 * math.tan(math.radians(1.36))

        assert abs(effective - expected) < 0.1

    def test_get_effective_swath_without_fov(self):
        """测试无FOV配置时回退到配置值"""
        imager = SARImager(
            imager_id='SAR-NO-FOV',
            resolution=3.0,
            swath_width=30.0
        )

        effective = imager.get_effective_swath(altitude_km=500.0)

        assert effective == 30.0


class TestSARImagerFOVConfig:
    """测试FOV配置获取"""

    def test_get_fov_config(self):
        """测试获取FOV配置字典"""
        imager = SARImager(
            imager_id='SAR-CONFIG',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36,
            azimuth_half_angle=0.91,
            azimuth_exclusion_angle=0.3
        )

        config = imager.get_fov_config()

        assert config['fov_type'] == 'sar'
        assert config['range_half_angle'] == 1.36
        assert config['azimuth_half_angle'] == 0.91
        assert config['azimuth_exclusion_angle'] == 0.3

    def test_get_fov_config_with_none_values(self):
        """测试部分FOV配置为None的情况"""
        imager = SARImager(
            imager_id='SAR-PARTIAL',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36
        )

        config = imager.get_fov_config()

        assert config['fov_type'] == 'sar'
        assert config['range_half_angle'] == 1.36
        assert config['azimuth_half_angle'] is None
        assert config['azimuth_exclusion_angle'] is None


class TestSARImagerSpecs:
    """测试规格输出"""

    def test_get_specs_with_fov(self):
        """测试规格输出包含FOV配置"""
        imager = SARImager(
            imager_id='SAR-SPECS',
            resolution=3.0,
            swath_width=30.0,
            band='X',
            polarization='VV',
            range_half_angle=1.36,
            azimuth_half_angle=0.91,
            azimuth_exclusion_angle=0.3
        )

        specs = imager.get_specs()

        assert 'range_half_angle' in specs
        assert 'azimuth_half_angle' in specs
        assert 'azimuth_exclusion_angle' in specs
        assert specs['range_half_angle'] == 1.36
        assert specs['azimuth_half_angle'] == 0.91
        assert specs['azimuth_exclusion_angle'] == 0.3

    def test_get_specs_without_fov(self):
        """测试规格输出不包含None的FOV配置"""
        imager = SARImager(
            imager_id='SAR-NO-FOV',
            resolution=3.0,
            swath_width=30.0
        )

        specs = imager.get_specs()

        assert 'range_half_angle' not in specs
        assert 'azimuth_half_angle' not in specs
        assert 'azimuth_exclusion_angle' not in specs


class TestSARImagerBackwardCompatibility:
    """测试向后兼容性"""

    def test_backward_compatible_initialization(self):
        """测试向后兼容的初始化方式"""
        # 旧式初始化（无FOV参数）
        imager = SARImager(
            imager_id='SAR-OLD',
            resolution=3.0,
            swath_width=30.0,
            band='X',
            polarization='VV',
            min_look_angle=20.0,
            max_look_angle=50.0
        )

        assert imager.resolution == 3.0
        assert imager.swath_width == 30.0
        assert imager.band == 'X'
        assert imager.range_half_angle is None

    def test_swath_width_priority(self):
        """测试直接配置swath_width优先于FOV计算（当不需要计算时）"""
        imager = SARImager(
            imager_id='SAR-PRIORITY',
            resolution=3.0,
            swath_width=25.0,  # 直接配置25km
            range_half_angle=1.36  # FOV计算约30km
        )

        # get_effective_swath使用FOV计算
        effective_fov = imager.get_effective_swath(altitude_km=631.0)
        assert effective_fov > 25.0  # FOV计算值更大

        # 但直接属性仍然是配置值
        assert imager.swath_width == 25.0


class TestSARImagerFOVIntegration:
    """集成测试"""

    def test_fov_with_different_satellites(self):
        """测试不同SAR卫星配置"""
        # SAR-1型配置
        sar1 = SARImager(
            imager_id='SAR-1',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36,
            azimuth_half_angle=0.91,
            azimuth_exclusion_angle=0.3
        )

        # SAR-2型配置
        sar2 = SARImager(
            imager_id='SAR-2',
            resolution=1.0,
            swath_width=20.0,
            range_half_angle=0.91,
            azimuth_half_angle=0.68,
            azimuth_exclusion_angle=0.25
        )

        # 在相同高度计算
        sar1_swath, sar1_length = sar1.calculate_swath_from_fov(altitude_km=631.0)
        sar2_swath, sar2_length = sar2.calculate_swath_from_fov(altitude_km=631.0)

        # SAR-1幅宽更大
        assert sar1_swath > sar2_swath

    def test_fov_geometric_consistency(self):
        """测试FOV几何一致性"""
        # 验证不同高度下的幅宽比例
        imager = SARImager(
            imager_id='SAR-GEO',
            resolution=3.0,
            swath_width=30.0,
            range_half_angle=1.36
        )

        swath_500, _ = imager.calculate_swath_from_fov(altitude_km=500.0)
        swath_631, _ = imager.calculate_swath_from_fov(altitude_km=631.0)
        swath_800, _ = imager.calculate_swath_from_fov(altitude_km=800.0)

        # 高度越高，幅宽越大（线性关系）
        assert swath_500 < swath_631 < swath_800

        # 验证比例关系
        ratio_500_631 = swath_631 / swath_500
        expected_ratio = 631.0 / 500.0
        assert abs(ratio_500_631 - expected_ratio) < 0.01
