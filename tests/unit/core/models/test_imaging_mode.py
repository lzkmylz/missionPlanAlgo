"""
测试成像模式配置类
"""

import pytest
from core.models.imaging_mode import (
    ImagingModeConfig,
    ImagingModeType,
    OPTICAL_PUSH_BROOM_HIGH_RES,
    OPTICAL_PUSH_BROOM_MEDIUM_RES,
    SAR_STRIPMAP_MODE,
    SAR_SPOTLIGHT_MODE,
    SAR_SCAN_MODE,
    SAR_SLIDING_SPOTLIGHT_MODE,
    MODE_TEMPLATES,
    get_mode_template,
)


class TestImagingModeConfig:
    """测试 ImagingModeConfig 类"""

    def test_basic_creation(self):
        """测试基本创建"""
        config = ImagingModeConfig(
            resolution_m=0.5,
            swath_width_m=15000,
            power_consumption_w=150.0,
            data_rate_mbps=200.0,
            min_duration_s=6.0,
            max_duration_s=12.0,
            mode_type="optical",
        )

        assert config.resolution_m == 0.5
        assert config.swath_width_m == 15000
        assert config.power_consumption_w == 150.0
        assert config.data_rate_mbps == 200.0
        assert config.min_duration_s == 6.0
        assert config.max_duration_s == 12.0
        assert config.mode_type == "optical"

    def test_validation_resolution_positive(self):
        """测试分辨率必须为正"""
        with pytest.raises(ValueError, match="resolution_m must be positive"):
            ImagingModeConfig(
                resolution_m=-1.0,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
            )

    def test_validation_swath_width_positive(self):
        """测试幅宽必须为正"""
        with pytest.raises(ValueError, match="swath_width_m must be positive"):
            ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=-1000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
            )

    def test_validation_power_non_negative(self):
        """测试功耗必须非负"""
        with pytest.raises(ValueError, match="power_consumption_w must be non-negative"):
            ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=-10.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
            )

    def test_validation_data_rate_non_negative(self):
        """测试数据率必须非负"""
        with pytest.raises(ValueError, match="data_rate_mbps must be non-negative"):
            ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=-100.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
            )

    def test_validation_duration_positive(self):
        """测试时长必须为正"""
        with pytest.raises(ValueError, match="min_duration_s must be positive"):
            ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=0.0,
                max_duration_s=12.0,
            )

    def test_validation_max_greater_than_min(self):
        """测试最大时长必须大于最小时长"""
        with pytest.raises(ValueError, match="max_duration_s .* must be >= min_duration_s"):
            ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=10.0,
                max_duration_s=5.0,
            )

    def test_to_dict(self):
        """测试转换为字典"""
        config = ImagingModeConfig(
            resolution_m=0.5,
            swath_width_m=15000,
            power_consumption_w=150.0,
            data_rate_mbps=200.0,
            min_duration_s=6.0,
            max_duration_s=12.0,
            mode_type="optical",
            fov_config={"cross_track": 2.5},
            characteristics={"bands": ["RGB"]},
        )

        data = config.to_dict()

        assert data["resolution_m"] == 0.5
        assert data["swath_width_m"] == 15000
        assert data["power_consumption_w"] == 150.0
        assert data["data_rate_mbps"] == 200.0
        assert data["fov_config"] == {"cross_track": 2.5}
        assert data["characteristics"] == {"bands": ["RGB"]}

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "resolution_m": 1.0,
            "swath_width_m": 20000,
            "power_consumption_w": 180.0,
            "data_rate_mbps": 250.0,
            "min_duration_s": 5.0,
            "max_duration_s": 15.0,
            "mode_type": "sar",
            "fov_config": {"range": 1.5},
            "characteristics": {"polarization": "HH"},
        }

        config = ImagingModeConfig.from_dict(data)

        assert config.resolution_m == 1.0
        assert config.swath_width_m == 20000
        assert config.power_consumption_w == 180.0
        assert config.data_rate_mbps == 250.0
        assert config.min_duration_s == 5.0
        assert config.max_duration_s == 15.0
        assert config.mode_type == "sar"
        assert config.fov_config == {"range": 1.5}

    def test_get_coverage_area(self):
        """测试覆盖面积计算"""
        config = ImagingModeConfig(
            resolution_m=0.5,
            swath_width_m=15000,  # 15km
            power_consumption_w=150.0,
            data_rate_mbps=200.0,
            min_duration_s=6.0,
            max_duration_s=12.0,
        )

        # 6秒成像，7500m/s速度，幅宽15km
        # 沿轨距离 = 7500 * 6 / 1000 = 45km
        # 面积 = 45 * 15 = 675 km2
        area = config.get_coverage_area_km2(6.0, 7500.0)
        assert area == pytest.approx(675.0, abs=0.1)

    def test_get_data_volume(self):
        """测试数据量计算"""
        config = ImagingModeConfig(
            resolution_m=0.5,
            swath_width_m=15000,
            power_consumption_w=150.0,
            data_rate_mbps=200.0,  # 200 Mbps
            min_duration_s=6.0,
            max_duration_s=12.0,
        )

        # 6秒成像，200 Mbps
        # 数据量 = 200 * 6 / 8000 = 0.15 GB
        volume = config.get_data_volume_gb(6.0)
        assert volume == pytest.approx(0.15, abs=0.001)

    def test_get_energy_consumption(self):
        """测试能耗计算"""
        config = ImagingModeConfig(
            resolution_m=0.5,
            swath_width_m=15000,
            power_consumption_w=150.0,  # 150W
            data_rate_mbps=200.0,
            min_duration_s=6.0,
            max_duration_s=12.0,
        )

        # 6秒成像，150W
        # 能耗 = 150 * 6 / 3600 = 0.25 Wh
        energy = config.get_energy_consumption_wh(6.0)
        assert energy == pytest.approx(0.25, abs=0.001)


class TestImagingModeTemplates:
    """测试预定义的成像模式模板"""

    def test_optical_push_broom_high_res(self):
        """测试高分辨率光学推扫模式模板"""
        config = OPTICAL_PUSH_BROOM_HIGH_RES

        assert config.resolution_m == 0.5
        assert config.swath_width_m == 15000
        assert config.power_consumption_w == 150.0
        assert config.data_rate_mbps == 200.0
        assert config.mode_type == "optical"
        assert "PAN" in config.characteristics.get("spectral_bands", [])

    def test_optical_push_broom_medium_res(self):
        """测试中分辨率光学推扫模式模板"""
        config = OPTICAL_PUSH_BROOM_MEDIUM_RES

        assert config.resolution_m == 2.0
        assert config.swath_width_m == 30000
        assert config.mode_type == "optical"

    def test_sar_stripmap_mode(self):
        """测试SAR条带模式模板"""
        config = SAR_STRIPMAP_MODE

        assert config.resolution_m == 3.0
        assert config.swath_width_m == 30000
        assert config.power_consumption_w == 300.0
        assert config.data_rate_mbps == 400.0
        assert config.mode_type == "sar"

    def test_sar_spotlight_mode(self):
        """测试SAR聚束模式模板"""
        config = SAR_SPOTLIGHT_MODE

        assert config.resolution_m == 1.0
        assert config.swath_width_m == 10000
        assert config.power_consumption_w == 500.0
        assert config.data_rate_mbps == 800.0
        assert config.mode_type == "sar"

    def test_sar_scan_mode(self):
        """测试SAR扫描模式模板"""
        config = SAR_SCAN_MODE

        assert config.resolution_m == 10.0
        assert config.swath_width_m == 100000
        assert config.power_consumption_w == 400.0
        assert config.data_rate_mbps == 600.0
        assert config.mode_type == "sar"

    def test_sar_sliding_spotlight_mode(self):
        """测试SAR滑动聚束模式模板"""
        config = SAR_SLIDING_SPOTLIGHT_MODE

        assert config.resolution_m == 1.5
        assert config.swath_width_m == 20000
        assert config.power_consumption_w == 450.0
        assert config.data_rate_mbps == 700.0
        assert config.mode_type == "sar"

    def test_mode_templates_dict(self):
        """测试模式模板字典"""
        assert "optical_push_broom_high" in MODE_TEMPLATES
        assert "sar_stripmap" in MODE_TEMPLATES
        assert "sar_spotlight" in MODE_TEMPLATES

    def test_get_mode_template(self):
        """测试获取模式模板函数"""
        config = get_mode_template("optical_push_broom_high")
        assert config is not None
        assert config.resolution_m == 0.5

        # 不存在的模板返回None
        assert get_mode_template("non_existent") is None
