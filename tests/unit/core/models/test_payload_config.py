"""
测试载荷配置类
"""

import pytest
import copy
from core.models.payload_config import (
    PayloadConfiguration,
    create_optical_payload_config,
    create_sar_payload_config,
)
from core.models.imaging_mode import ImagingModeConfig


class TestPayloadConfiguration:
    """测试 PayloadConfiguration 类"""

    def test_basic_creation(self):
        """测试基本创建"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.payload_type == "optical"
        assert config.default_mode == "push_broom"
        assert len(config.modes) == 1
        assert "push_broom" in config.modes

    def test_validation_invalid_payload_type(self):
        """测试无效的载荷类型"""
        with pytest.raises(ValueError, match="payload_type must be 'optical' or 'sar'"):
            PayloadConfiguration(
                payload_type="invalid",
                default_mode="push_broom",
                modes={},
            )

    def test_validation_empty_modes(self):
        """测试空模式列表"""
        with pytest.raises(ValueError, match="At least one imaging mode must be defined"):
            PayloadConfiguration(
                payload_type="optical",
                default_mode="push_broom",
                modes={},
            )

    def test_validation_default_mode_not_found(self):
        """测试默认模式不存在时自动选择第一个"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="non_existent",  # 不存在的模式
            modes=modes,
        )

        # 应该自动选择第一个模式作为默认
        assert config.default_mode == "push_broom"

    def test_validation_mode_type_mismatch(self):
        """测试模式类型与载荷类型不匹配"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="sar",  # 错误的类型
            )
        }

        with pytest.raises(ValueError, match="Mode 'push_broom' has type 'sar' but payload type is 'optical'"):
            PayloadConfiguration(
                payload_type="optical",
                default_mode="push_broom",
                modes=modes,
            )

    def test_get_mode_config(self):
        """测试获取模式配置"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            ),
            "frame": ImagingModeConfig(
                resolution_m=1.0,
                swath_width_m=10000,
                power_consumption_w=120.0,
                data_rate_mbps=150.0,
                min_duration_s=5.0,
                max_duration_s=10.0,
                mode_type="optical",
            ),
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        # 获取指定模式
        mode_config = config.get_mode_config("frame")
        assert mode_config.resolution_m == 1.0

        # 获取默认模式
        default_config = config.get_mode_config(None)
        assert default_config.resolution_m == 0.5

    def test_get_mode_config_not_found(self):
        """测试获取不存在的模式配置"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        with pytest.raises(ValueError, match="Imaging mode 'non_existent' not found"):
            config.get_mode_config("non_existent")

    def test_get_mode_names(self):
        """测试获取所有模式名称"""
        modes = {
            "stripmap": ImagingModeConfig(
                resolution_m=3.0,
                swath_width_m=30000,
                power_consumption_w=300.0,
                data_rate_mbps=400.0,
                min_duration_s=5.0,
                max_duration_s=15.0,
                mode_type="sar",
            ),
            "spotlight": ImagingModeConfig(
                resolution_m=1.0,
                swath_width_m=10000,
                power_consumption_w=500.0,
                data_rate_mbps=800.0,
                min_duration_s=10.0,
                max_duration_s=25.0,
                mode_type="sar",
            ),
        }

        config = PayloadConfiguration(
            payload_type="sar",
            default_mode="stripmap",
            modes=modes,
        )

        names = config.get_mode_names()
        assert len(names) == 2
        assert "stripmap" in names
        assert "spotlight" in names

    def test_has_mode(self):
        """测试检查是否支持某模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.has_mode("push_broom") is True
        assert config.has_mode("frame") is False

    def test_add_mode(self):
        """测试添加新模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        new_mode = ImagingModeConfig(
            resolution_m=1.0,
            swath_width_m=10000,
            power_consumption_w=120.0,
            data_rate_mbps=150.0,
            min_duration_s=5.0,
            max_duration_s=10.0,
            mode_type="optical",
        )

        config.add_mode("frame", new_mode)

        assert config.has_mode("frame") is True
        assert config.get_mode_config("frame").resolution_m == 1.0

    def test_add_mode_already_exists(self):
        """测试添加已存在的模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        new_mode = ImagingModeConfig(
            resolution_m=1.0,
            swath_width_m=10000,
            power_consumption_w=120.0,
            data_rate_mbps=150.0,
            min_duration_s=5.0,
            max_duration_s=10.0,
            mode_type="optical",
        )

        with pytest.raises(ValueError, match="Imaging mode 'push_broom' already exists"):
            config.add_mode("push_broom", new_mode)

    def test_add_mode_type_mismatch(self):
        """测试添加类型不匹配的模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        sar_mode = ImagingModeConfig(
            resolution_m=3.0,
            swath_width_m=30000,
            power_consumption_w=300.0,
            data_rate_mbps=400.0,
            min_duration_s=5.0,
            max_duration_s=15.0,
            mode_type="sar",
        )

        with pytest.raises(ValueError, match="Mode type 'sar' does not match payload type 'optical'"):
            config.add_mode("stripmap", sar_mode)

    def test_remove_mode(self):
        """测试移除模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            ),
            "frame": ImagingModeConfig(
                resolution_m=1.0,
                swath_width_m=10000,
                power_consumption_w=120.0,
                data_rate_mbps=150.0,
                min_duration_s=5.0,
                max_duration_s=10.0,
                mode_type="optical",
            ),
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        config.remove_mode("frame")

        assert config.has_mode("frame") is False
        assert len(config.modes) == 1

    def test_remove_mode_not_found(self):
        """测试移除不存在的模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        with pytest.raises(ValueError, match="Imaging mode 'non_existent' not found"):
            config.remove_mode("non_existent")

    def test_remove_only_mode(self):
        """测试移除唯一的模式"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        with pytest.raises(ValueError, match="Cannot remove the only imaging mode"):
            config.remove_mode("push_broom")

    def test_get_resolution(self):
        """测试获取分辨率"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.get_resolution("push_broom") == 0.5
        assert config.get_resolution() == 0.5  # 默认模式

    def test_get_swath_width(self):
        """测试获取幅宽"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.get_swath_width() == 15000

    def test_get_power_consumption(self):
        """测试获取功耗"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.get_power_consumption() == 150.0

    def test_get_data_rate(self):
        """测试获取数据率"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.get_data_rate() == 200.0

    def test_get_best_resolution_mode(self):
        """测试获取最佳分辨率模式"""
        modes = {
            "stripmap": ImagingModeConfig(
                resolution_m=3.0,
                swath_width_m=30000,
                power_consumption_w=300.0,
                data_rate_mbps=400.0,
                min_duration_s=5.0,
                max_duration_s=15.0,
                mode_type="sar",
            ),
            "spotlight": ImagingModeConfig(
                resolution_m=1.0,
                swath_width_m=10000,
                power_consumption_w=500.0,
                data_rate_mbps=800.0,
                min_duration_s=10.0,
                max_duration_s=25.0,
                mode_type="sar",
            ),
        }

        config = PayloadConfiguration(
            payload_type="sar",
            default_mode="stripmap",
            modes=modes,
        )

        # spotlight 有更高的分辨率（1.0 < 3.0）
        assert config.get_best_resolution_mode() == "spotlight"

    def test_get_best_swath_mode(self):
        """测试获取最佳幅宽模式"""
        modes = {
            "stripmap": ImagingModeConfig(
                resolution_m=3.0,
                swath_width_m=30000,
                power_consumption_w=300.0,
                data_rate_mbps=400.0,
                min_duration_s=5.0,
                max_duration_s=15.0,
                mode_type="sar",
            ),
            "spotlight": ImagingModeConfig(
                resolution_m=1.0,
                swath_width_m=10000,
                power_consumption_w=500.0,
                data_rate_mbps=800.0,
                min_duration_s=10.0,
                max_duration_s=25.0,
                mode_type="sar",
            ),
        }

        config = PayloadConfiguration(
            payload_type="sar",
            default_mode="stripmap",
            modes=modes,
        )

        # stripmap 有更大的幅宽（30000 > 10000）
        assert config.get_best_swath_mode() == "stripmap"

    def test_to_dict(self):
        """测试转换为字典"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
            payload_id="test_payload",
            description="Test optical payload",
        )

        data = config.to_dict()

        assert data["payload_type"] == "optical"
        assert data["default_mode"] == "push_broom"
        assert data["payload_id"] == "test_payload"
        assert data["description"] == "Test optical payload"
        assert "push_broom" in data["modes"]

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "payload_type": "sar",
            "default_mode": "stripmap",
            "modes": {
                "stripmap": {
                    "resolution_m": 3.0,
                    "swath_width_m": 30000,
                    "power_consumption_w": 300.0,
                    "data_rate_mbps": 400.0,
                    "min_duration_s": 5.0,
                    "max_duration_s": 15.0,
                    "mode_type": "sar",
                }
            },
            "payload_id": "sar_1",
            "description": "Test SAR payload",
        }

        config = PayloadConfiguration.from_dict(data)

        assert config.payload_type == "sar"
        assert config.default_mode == "stripmap"
        assert config.payload_id == "sar_1"
        assert config.get_resolution("stripmap") == 3.0

    def test_validate_success(self):
        """测试验证通过"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        assert config.validate() is True

    def test_validate_no_modes(self):
        """测试验证失败 - 无模式"""
        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes={
                "push_broom": ImagingModeConfig(
                    resolution_m=0.5,
                    swath_width_m=15000,
                    power_consumption_w=150.0,
                    data_rate_mbps=200.0,
                    min_duration_s=6.0,
                    max_duration_s=12.0,
                    mode_type="optical",
                )
            },
        )

        # 直接清空模式字典来绕过 remove_mode 的检查
        config.modes = {}

        with pytest.raises(ValueError, match="No imaging modes defined"):
            config.validate()

    def test_copy(self):
        """测试复制"""
        modes = {
            "push_broom": ImagingModeConfig(
                resolution_m=0.5,
                swath_width_m=15000,
                power_consumption_w=150.0,
                data_rate_mbps=200.0,
                min_duration_s=6.0,
                max_duration_s=12.0,
                mode_type="optical",
            )
        }

        config = PayloadConfiguration(
            payload_type="optical",
            default_mode="push_broom",
            modes=modes,
        )

        config_copy = config.copy()

        # 验证是独立的拷贝
        assert config_copy is not config
        assert config_copy.modes is not config.modes
        assert config_copy.get_resolution() == config.get_resolution()


class TestCreatePayloadConfig:
    """测试创建载荷配置辅助函数"""

    def test_create_optical_payload_config(self):
        """测试创建光学载荷配置"""
        config = create_optical_payload_config(
            resolution_m=0.8,
            swath_width_m=20000,
            power_consumption_w=180.0,
            data_rate_mbps=250.0,
            spectral_bands=["PAN", "RGB"],
        )

        assert config.payload_type == "optical"
        assert config.default_mode == "push_broom"
        assert config.get_resolution() == 0.8
        assert config.get_swath_width() == 20000
        assert config.get_power_consumption() == 180.0
        assert config.get_data_rate() == 250.0

        mode_config = config.get_mode_config("push_broom")
        assert "PAN" in mode_config.characteristics.get("spectral_bands", [])

    def test_create_sar_payload_config_default(self):
        """测试创建SAR载荷配置（使用默认模式）"""
        config = create_sar_payload_config()

        assert config.payload_type == "sar"
        assert config.default_mode == "stripmap"
        # 默认应该有4个模式
        assert len(config.modes) == 4
        assert config.has_mode("stripmap")
        assert config.has_mode("spotlight")
        assert config.has_mode("scan")
        assert config.has_mode("sliding_spotlight")

    def test_create_sar_payload_config_custom(self):
        """测试创建SAR载荷配置（自定义模式）"""
        custom_modes = {
            "stripmap": {
                "resolution_m": 5.0,
                "swath_width_m": 40000,
                "power_consumption_w": 350.0,
                "data_rate_mbps": 450.0,
                "min_duration_s": 6.0,
                "max_duration_s": 20.0,
                "mode_type": "sar",
            }
        }

        config = create_sar_payload_config(modes=custom_modes)

        assert config.payload_type == "sar"
        assert len(config.modes) == 1
        assert config.get_resolution("stripmap") == 5.0
