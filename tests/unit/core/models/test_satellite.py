"""
卫星模型单元测试

测试卫星配置加载，包括详细的成像器和成像模式参数
遵循TDD原则：先写测试，再实现功能
"""

import pytest
from typing import Dict, Any, List

from core.models.satellite import (
    Satellite,
    SatelliteCapabilities,
    SatelliteType,
    ImagingMode,
    Orbit,
    OrbitType,
)


class TestSatelliteCapabilitiesExtended:
    """测试扩展的卫星能力配置 - 包含详细成像器和成像模式参数"""

    def test_capabilities_has_imager_field(self):
        """测试SatelliteCapabilities有imager字段"""
        capabilities = SatelliteCapabilities()
        assert hasattr(capabilities, 'imager')

    def test_capabilities_has_imaging_mode_details_field(self):
        """测试SatelliteCapabilities有imaging_mode_details字段"""
        capabilities = SatelliteCapabilities()
        assert hasattr(capabilities, 'imaging_mode_details')

    def test_capabilities_default_imager_is_empty_dict(self):
        """测试默认imager为空字典"""
        capabilities = SatelliteCapabilities()
        assert capabilities.imager == {}

    def test_capabilities_default_imaging_mode_details_is_empty_list(self):
        """测试默认imaging_mode_details为空列表"""
        capabilities = SatelliteCapabilities()
        assert capabilities.imaging_mode_details == []

    def test_capabilities_with_imager_data(self):
        """测试使用imager数据创建Capabilities"""
        imager_data = {
            "imager_type": "optical",
            "resolution": 2.0,
            "swath_width": 12.0,
            "focal_length": 5.8,
            "aperture": 0.6,
            "pixel_size": 0.00001,
            "quantization": 10
        }
        capabilities = SatelliteCapabilities(
            imager=imager_data
        )
        assert capabilities.imager == imager_data
        assert capabilities.imager["resolution"] == 2.0
        assert capabilities.imager["focal_length"] == 5.8

    def test_capabilities_with_imaging_mode_details(self):
        """测试使用详细成像模式数据创建Capabilities"""
        mode_details = [
            {
                "mode_id": "push_broom",
                "mode_name": "推扫模式",
                "integration_time": 0.0002,
                "readout_time": 0.0001,
                "snr_target": 100,
                "min_imaging_duration": 17,
                "max_imaging_duration": 143,
                "typical_velocity": 7000
            }
        ]
        capabilities = SatelliteCapabilities(
            imaging_mode_details=mode_details
        )
        assert len(capabilities.imaging_mode_details) == 1
        assert capabilities.imaging_mode_details[0]["mode_id"] == "push_broom"
        assert capabilities.imaging_mode_details[0]["integration_time"] == 0.0002


class TestSatelliteFromDictExtended:
    """测试从字典加载卫星配置，包含详细参数"""

    def test_from_dict_loads_optical_imager(self):
        """测试从字典加载光学成像器详细配置"""
        data = {
            "id": "sat_001",
            "name": "测试光学卫星",
            "sat_type": "optical_1",
            "orbit": {
                "orbit_type": "SSO",
                "altitude": 645000,
                "inclination": 97.9
            },
            "capabilities": {
                "imaging_modes": ["push_broom"],
                "max_off_nadir": 30.0,
                "storage_capacity": 500.0,
                "power_capacity": 2000.0,
                "data_rate": 300.0,
                "imager": {
                    "imager_type": "optical",
                    "resolution": 2.0,
                    "swath_width": 12.0,
                    "focal_length": 5.8,
                    "aperture": 0.6,
                    "pixel_size": 0.00001,
                    "quantization": 10
                },
                "imaging_modes": [
                    {
                        "_comment": "推扫式成像模式",
                        "mode_id": "push_broom",
                        "mode_name": "推扫模式",
                        "description": "推扫式成像，卫星飞行方向连续扫描",
                        "integration_time": 0.0002,
                        "readout_time": 0.0001,
                        "snr_target": 100,
                        "min_imaging_duration": 17,
                        "max_imaging_duration": 143,
                        "typical_velocity": 7000
                    }
                ]
            }
        }

        satellite = Satellite.from_dict(data)

        # 验证基本字段
        assert satellite.id == "sat_001"
        assert satellite.name == "测试光学卫星"
        assert satellite.sat_type == SatelliteType.OPTICAL_1

        # 验证imager配置被加载
        assert satellite.capabilities.imager is not None
        assert satellite.capabilities.imager["imager_type"] == "optical"
        assert satellite.capabilities.imager["resolution"] == 2.0
        assert satellite.capabilities.imager["focal_length"] == 5.8
        assert satellite.capabilities.imager["aperture"] == 0.6

        # 验证详细成像模式被加载
        assert len(satellite.capabilities.imaging_mode_details) == 1
        mode = satellite.capabilities.imaging_mode_details[0]
        assert mode["mode_id"] == "push_broom"
        assert mode["integration_time"] == 0.0002
        assert mode["readout_time"] == 0.0001
        assert mode["snr_target"] == 100

    def test_from_dict_loads_sar_imager(self):
        """测试从字典加载SAR成像器详细配置"""
        data = {
            "id": "sar_001",
            "name": "测试SAR卫星",
            "sat_type": "sar_1",
            "orbit": {
                "orbit_type": "SSO",
                "altitude": 631000,
                "inclination": 98.0
            },
            "capabilities": {
                "imaging_modes": [
                    {
                        "mode_id": "stripmap",
                        "mode_name": "条带模式",
                        "resolution": 3.0,
                        "azimuth_resolution": 3.0,
                        "swath_width": 30.0,
                        "integration_factor": 1.0,
                        "min_imaging_duration": 4,
                        "max_imaging_duration": 72,
                        "typical_velocity": 7000
                    },
                    {
                        "mode_id": "spotlight",
                        "mode_name": "聚束模式",
                        "resolution": 1.0,
                        "azimuth_resolution": 1.0,
                        "swath_width": 10.0,
                        "integration_factor": 2.0,
                        "min_imaging_duration": 1,
                        "max_imaging_duration": 2
                    }
                ],
                "max_off_nadir": 35.0,
                "storage_capacity": 1000.0,
                "power_capacity": 3000.0,
                "data_rate": 300.0,
                "imager": {
                    "imager_type": "sar",
                    "resolution": 3.0,
                    "swath_width": 30.0,
                    "band": "X",
                    "wavelength": 0.031,
                    "polarization": "VV",
                    "antenna_length": 6.0,
                    "antenna_width": 1.5,
                    "min_look_angle": 20.0,
                    "max_look_angle": 50.0,
                    "prf_min": 1000,
                    "prf_max": 6000
                }
            }
        }

        satellite = Satellite.from_dict(data)

        # 验证SAR imager配置
        assert satellite.capabilities.imager["imager_type"] == "sar"
        assert satellite.capabilities.imager["band"] == "X"
        assert satellite.capabilities.imager["polarization"] == "VV"
        assert satellite.capabilities.imager["antenna_length"] == 6.0
        assert satellite.capabilities.imager["min_look_angle"] == 20.0
        assert satellite.capabilities.imager["prf_max"] == 6000

        # 验证多个成像模式
        assert len(satellite.capabilities.imaging_mode_details) == 2
        stripmap = satellite.capabilities.imaging_mode_details[0]
        assert stripmap["mode_id"] == "stripmap"
        assert stripmap["integration_factor"] == 1.0

        spotlight = satellite.capabilities.imaging_mode_details[1]
        assert spotlight["mode_id"] == "spotlight"
        assert spotlight["integration_factor"] == 2.0

    def test_from_dict_backwards_compatibility_missing_imager(self):
        """测试向后兼容：旧JSON没有imager字段时使用默认值"""
        data = {
            "id": "sat_old",
            "name": "旧配置卫星",
            "sat_type": "optical_1",
            "orbit": {
                "orbit_type": "SSO",
                "altitude": 500000,
                "inclination": 97.4
            },
            "capabilities": {
                "imaging_modes": ["push_broom"],
                "max_off_nadir": 30.0,
                "storage_capacity": 500.0,
                "power_capacity": 2000.0,
                "data_rate": 300.0
                # 注意：没有imager和imaging_modes_details字段
            }
        }

        satellite = Satellite.from_dict(data)

        # 验证基本功能正常
        assert satellite.id == "sat_old"
        assert satellite.capabilities.max_off_nadir == 30.0

        # 验证imager默认为空字典
        assert satellite.capabilities.imager == {}

        # 验证imaging_mode_details默认为空列表
        assert satellite.capabilities.imaging_mode_details == []

    def test_from_dict_backwards_compatibility_empty_capabilities(self):
        """测试向后兼容：capabilities完全为空时使用默认值"""
        data = {
            "id": "sat_empty",
            "name": "空配置卫星",
            "sat_type": "optical_2",
            "orbit": {},
            "capabilities": {}
        }

        satellite = Satellite.from_dict(data)

        # 验证卫星创建成功
        assert satellite.id == "sat_empty"
        assert satellite.sat_type == SatelliteType.OPTICAL_2

        # 验证默认imager为空字典
        assert satellite.capabilities.imager == {}

        # 验证默认imaging_mode_details为空列表
        assert satellite.capabilities.imaging_mode_details == []


class TestSatelliteToDictExtended:
    """测试卫星序列化为字典，包含详细参数"""

    def test_to_dict_includes_imager(self):
        """测试to_dict包含imager配置"""
        satellite = Satellite(
            id="sat_001",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1,
            capabilities=SatelliteCapabilities(
                imager={
                    "imager_type": "optical",
                    "resolution": 2.0,
                    "focal_length": 5.8
                }
            )
        )

        data = satellite.to_dict()

        # 验证imager被序列化
        assert "imager" in data["capabilities"]
        assert data["capabilities"]["imager"]["imager_type"] == "optical"
        assert data["capabilities"]["imager"]["resolution"] == 2.0
        assert data["capabilities"]["imager"]["focal_length"] == 5.8

    def test_to_dict_includes_imaging_mode_details(self):
        """测试to_dict包含详细成像模式配置"""
        satellite = Satellite(
            id="sat_001",
            name="测试卫星",
            sat_type=SatelliteType.SAR_1,
            capabilities=SatelliteCapabilities(
                imaging_mode_details=[
                    {
                        "mode_id": "stripmap",
                        "integration_time": 0.001,
                        "snr_target": 100
                    }
                ]
            )
        )

        data = satellite.to_dict()

        # 验证imaging_mode_details被序列化
        assert "imaging_mode_details" in data["capabilities"]
        assert len(data["capabilities"]["imaging_mode_details"]) == 1
        assert data["capabilities"]["imaging_mode_details"][0]["mode_id"] == "stripmap"

    def test_round_trip_serialization(self):
        """测试往返序列化：to_dict -> from_dict 保留所有数据"""
        original = Satellite(
            id="sat_roundtrip",
            name="往返测试卫星",
            sat_type=SatelliteType.SAR_2,
            orbit=Orbit(
                orbit_type=OrbitType.SSO,
                altitude=600000,
                inclination=98.0
            ),
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.SPOTLIGHT, ImagingMode.STRIPMAP],
                max_off_nadir=45.0,
                storage_capacity=1500.0,
                power_capacity=4000.0,
                data_rate=500.0,
                imager={
                    "imager_type": "sar",
                    "resolution": 1.0,
                    "band": "X",
                    "antenna_length": 8.0
                },
                imaging_mode_details=[
                    {
                        "mode_id": "spotlight",
                        "integration_factor": 2.0,
                        "min_imaging_duration": 1,
                        "max_imaging_duration": 1
                    },
                    {
                        "mode_id": "stripmap",
                        "integration_factor": 1.0,
                        "min_imaging_duration": 4,
                        "max_imaging_duration": 72
                    }
                ]
            )
        )

        # 序列化
        data = original.to_dict()

        # 反序列化
        restored = Satellite.from_dict(data)

        # 验证基本字段
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.sat_type == original.sat_type

        # 验证轨道
        assert restored.orbit.altitude == original.orbit.altitude
        assert restored.orbit.inclination == original.orbit.inclination

        # 验证基本能力
        assert restored.capabilities.max_off_nadir == original.capabilities.max_off_nadir
        assert restored.capabilities.storage_capacity == original.capabilities.storage_capacity

        # 验证imager完全保留
        assert restored.capabilities.imager == original.capabilities.imager

        # 验证imaging_mode_details完全保留
        assert len(restored.capabilities.imaging_mode_details) == len(original.capabilities.imaging_mode_details)
        assert restored.capabilities.imaging_mode_details[0]["mode_id"] == "spotlight"
        assert restored.capabilities.imaging_mode_details[1]["mode_id"] == "stripmap"


class TestSatelliteJsonFileLoading:
    """测试从实际JSON文件加载卫星配置"""

    def test_load_optical_1_json_template(self):
        """测试加载optical_1.json模板文件"""
        import json
        import os

        json_path = "/Users/zhaolin/Documents/职称论文/missionPlanAlgo/data/entity_lib/satellites/optical_1.json"

        # 确保文件存在
        assert os.path.exists(json_path), f"JSON文件不存在: {json_path}"

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # JSON模板使用template_id而不是id，需要映射
        data['id'] = data.get('template_id', 'optical_1')

        # 转换为Satellite对象
        satellite = Satellite.from_dict(data)

        # 验证基本字段
        assert satellite.id == "optical_1"
        assert satellite.name == "光学卫星1型"
        assert satellite.sat_type == SatelliteType.OPTICAL_1

        # 验证imager配置被加载
        assert satellite.capabilities.imager is not None
        assert satellite.capabilities.imager.get("imager_type") == "optical"
        assert satellite.capabilities.imager.get("resolution") == 0.5
        assert satellite.capabilities.imager.get("focal_length") == 5.8
        assert satellite.capabilities.imager.get("aperture") == 0.6

        # 验证详细成像模式
        assert len(satellite.capabilities.imaging_mode_details) >= 1
        mode = satellite.capabilities.imaging_mode_details[0]
        assert mode.get("mode_id") == "push_broom"
        assert "integration_time" in mode
        assert "readout_time" in mode
        assert "snr_target" in mode

    def test_load_sar_1_json_template(self):
        """测试加载sar_1.json模板文件"""
        import json
        import os

        json_path = "/Users/zhaolin/Documents/职称论文/missionPlanAlgo/data/entity_lib/satellites/sar_1.json"

        # 确保文件存在
        assert os.path.exists(json_path), f"JSON文件不存在: {json_path}"

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # JSON模板使用template_id而不是id，需要映射
        data['id'] = data.get('template_id', 'sar_1')

        # 转换为Satellite对象
        satellite = Satellite.from_dict(data)

        # 验证imager配置被加载
        assert satellite.capabilities.imager.get("imager_type") == "sar"
        assert satellite.capabilities.imager.get("band") == "X"
        assert satellite.capabilities.imager.get("polarization") == "VV"
        assert satellite.capabilities.imager.get("antenna_length") == 6.0

        # 验证多个成像模式
        assert len(satellite.capabilities.imaging_mode_details) >= 2


class TestOrbit:
    """测试轨道类"""

    def test_orbit_default_values(self):
        """测试轨道默认值"""
        orbit = Orbit()
        assert orbit.orbit_type == OrbitType.SSO
        assert orbit.altitude == 500000.0
        assert orbit.inclination == 97.4
        assert orbit.eccentricity == 0.0

    def test_orbit_custom_values(self):
        """测试轨道自定义值"""
        orbit = Orbit(
            orbit_type=OrbitType.LEO,
            altitude=600000.0,
            inclination=98.0,
            eccentricity=0.001
        )
        assert orbit.orbit_type == OrbitType.LEO
        assert orbit.altitude == 600000.0
        assert orbit.inclination == 98.0
        assert orbit.eccentricity == 0.001

    def test_get_semi_major_axis(self):
        """测试获取半长轴"""
        orbit = Orbit(altitude=500000.0)
        semi_major = orbit.get_semi_major_axis()
        assert semi_major == 6871000.0  # 6371000 + 500000

    def test_get_period(self):
        """测试获取轨道周期"""
        orbit = Orbit(altitude=500000.0)
        period = orbit.get_period()
        # 轨道周期应该约为5660秒（约94分钟）
        assert 5000 < period < 6000


class TestSatelliteCapabilities:
    """测试卫星能力配置"""

    def test_supports_mode_true(self):
        """测试支持模式检查 - 支持"""
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME]
        )
        assert capabilities.supports_mode(ImagingMode.PUSH_BROOM) is True
        assert capabilities.supports_mode(ImagingMode.FRAME) is True

    def test_supports_mode_false(self):
        """测试支持模式检查 - 不支持"""
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM]
        )
        assert capabilities.supports_mode(ImagingMode.FRAME) is False
        assert capabilities.supports_mode(ImagingMode.SPOTLIGHT) is False


class TestSatelliteInitialization:
    """测试卫星初始化"""

    def test_default_capabilities_set_for_optical_1(self):
        """测试光学1型卫星默认能力设置"""
        satellite = Satellite(
            id="opt1",
            name="光学1型",
            sat_type=SatelliteType.OPTICAL_1
        )
        assert ImagingMode.PUSH_BROOM in satellite.capabilities.imaging_modes
        assert satellite.capabilities.max_off_nadir == 30.0
        assert satellite.capabilities.resolution == 10.0
        assert satellite.capabilities.storage_capacity == 500.0
        assert satellite.capabilities.power_capacity == 2000.0
        assert satellite.current_power == 2000.0  # 默认初始化为满电量

    def test_default_capabilities_set_for_optical_2(self):
        """测试光学2型卫星默认能力设置"""
        satellite = Satellite(
            id="opt2",
            name="光学2型",
            sat_type=SatelliteType.OPTICAL_2
        )
        assert ImagingMode.PUSH_BROOM in satellite.capabilities.imaging_modes
        assert ImagingMode.FRAME in satellite.capabilities.imaging_modes
        assert satellite.capabilities.max_off_nadir == 45.0
        assert satellite.capabilities.resolution == 5.0
        assert satellite.capabilities.storage_capacity == 800.0
        assert satellite.capabilities.power_capacity == 2500.0

    def test_default_capabilities_set_for_sar_1(self):
        """测试SAR1型卫星默认能力设置"""
        satellite = Satellite(
            id="sar1",
            name="SAR1型",
            sat_type=SatelliteType.SAR_1
        )
        assert ImagingMode.SPOTLIGHT in satellite.capabilities.imaging_modes
        assert ImagingMode.SLIDING_SPOTLIGHT in satellite.capabilities.imaging_modes
        assert ImagingMode.STRIPMAP in satellite.capabilities.imaging_modes
        assert satellite.capabilities.max_off_nadir == 35.0
        assert satellite.capabilities.resolution == 3.0
        assert satellite.capabilities.storage_capacity == 1000.0
        assert satellite.capabilities.power_capacity == 3000.0

    def test_default_capabilities_set_for_sar_2(self):
        """测试SAR2型卫星默认能力设置"""
        satellite = Satellite(
            id="sar2",
            name="SAR2型",
            sat_type=SatelliteType.SAR_2
        )
        assert ImagingMode.SPOTLIGHT in satellite.capabilities.imaging_modes
        assert ImagingMode.SLIDING_SPOTLIGHT in satellite.capabilities.imaging_modes
        assert ImagingMode.STRIPMAP in satellite.capabilities.imaging_modes
        assert satellite.capabilities.max_off_nadir == 50.0
        assert satellite.capabilities.resolution == 1.0
        assert satellite.capabilities.storage_capacity == 1500.0
        assert satellite.capabilities.power_capacity == 4000.0

    def test_custom_capabilities_not_overridden(self):
        """测试自定义能力不会被覆盖"""
        satellite = Satellite(
            id="custom",
            name="自定义卫星",
            sat_type=SatelliteType.OPTICAL_1,
            capabilities=SatelliteCapabilities(
                imaging_modes=[ImagingMode.FRAME],  # 自定义模式
                max_off_nadir=60.0,  # 自定义侧摆角
                resolution=0.5  # 自定义分辨率
            )
        )
        # 验证自定义值被保留
        assert satellite.capabilities.imaging_modes == [ImagingMode.FRAME]
        assert satellite.capabilities.max_off_nadir == 60.0
        assert satellite.capabilities.resolution == 0.5


class TestSatelliteFromDictEdgeCases:
    """测试卫星从字典加载的边界情况"""

    def test_from_dict_with_tle(self):
        """测试从字典加载包含TLE的卫星"""
        data = {
            "id": "sat_tle",
            "name": "TLE卫星",
            "sat_type": "optical_1",
            "orbit": {},
            "capabilities": {},
            "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"
        }

        satellite = Satellite.from_dict(data)
        assert satellite.tle_line1 == data["tle_line1"]
        assert satellite.tle_line2 == data["tle_line2"]

    def test_from_dict_with_leo_orbit(self):
        """测试从字典加载LEO轨道"""
        data = {
            "id": "sat_leo",
            "name": "LEO卫星",
            "sat_type": "optical_1",
            "orbit": {
                "orbit_type": "LEO",
                "altitude": 400000,
                "inclination": 51.6
            },
            "capabilities": {}
        }

        satellite = Satellite.from_dict(data)
        assert satellite.orbit.orbit_type == OrbitType.LEO
        assert satellite.orbit.altitude == 400000
        assert satellite.orbit.inclination == 51.6

    def test_from_dict_with_imaging_modes_details_alias(self):
        """测试从字典加载使用imaging_modes_details别名的配置"""
        data = {
            "id": "sat_alias",
            "name": "别名测试卫星",
            "sat_type": "optical_1",
            "orbit": {},
            "capabilities": {
                "imaging_modes": ["push_broom"],
                "imager": {"resolution": 2.0},
                "imaging_modes_details": [  # 使用别名
                    {
                        "mode_id": "push_broom",
                        "integration_time": 0.001
                    }
                ]
            }
        }

        satellite = Satellite.from_dict(data)
        assert len(satellite.capabilities.imaging_mode_details) == 1
        assert satellite.capabilities.imaging_mode_details[0]["mode_id"] == "push_broom"


class TestSatellitePositionCalculation:
    """测试卫星位置计算"""

    def test_get_position_simplified(self):
        """测试简化轨道位置计算"""
        from datetime import datetime

        satellite = Satellite(
            id="pos_test",
            name="位置测试卫星",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                altitude=500000,
                inclination=97.4,
                raan=0.0,
                mean_anomaly=0.0
            )
        )

        # 使用简化模型计算位置（无TLE）
        dt = datetime(2024, 6, 1, 12, 0, 0)
        position = satellite._get_position_simplified(dt)

        # 验证返回的是3D坐标
        assert len(position) == 3
        x, y, z = position
        # 验证位置在合理范围内（地球半径 + 轨道高度）
        r = (x**2 + y**2 + z**2) ** 0.5
        assert 6000 < r < 7000  # km

    def test_get_subpoint(self):
        """测试获取星下点坐标"""
        from datetime import datetime

        satellite = Satellite(
            id="subpoint_test",
            name="星下点测试卫星",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                altitude=500000,
                inclination=97.4
            )
        )

        dt = datetime(2024, 6, 1, 12, 0, 0)
        lat, lon, alt = satellite.get_subpoint(dt)

        # 验证纬度在有效范围内
        assert -90 <= lat <= 90
        # 验证经度在有效范围内
        assert -180 <= lon <= 180
        # 验证高度接近轨道高度
        assert 400000 < alt < 600000

    def test_get_position_with_tle(self):
        """测试使用TLE的SGP4位置计算"""
        from datetime import datetime

        # 使用ISS的真实TLE数据
        tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537"

        satellite = Satellite(
            id="iss",
            name="国际空间站",
            sat_type=SatelliteType.OPTICAL_1,
            tle_line1=tle_line1,
            tle_line2=tle_line2
        )

        dt = datetime(2008, 9, 20, 12, 0, 0)
        position = satellite.get_position_sgp4(dt)

        # 验证返回的是3D坐标
        assert len(position) == 3
        x, y, z = position
        # 验证位置在合理范围内（地球半径 + 轨道高度，单位km）
        r = (x**2 + y**2 + z**2) ** 0.5
        assert 6000 < r < 7000  # km (ISS orbit ~400km altitude)

class TestImagingModeEnum:
    """测试成像模式枚举"""

    def test_imaging_mode_values(self):
        """测试成像模式值"""
        assert ImagingMode.SPOTLIGHT.value == "spotlight"
        assert ImagingMode.SLIDING_SPOTLIGHT.value == "sliding_spotlight"
        assert ImagingMode.STRIPMAP.value == "stripmap"
        assert ImagingMode.SCAN.value == "scan"
        assert ImagingMode.PUSH_BROOM.value == "push_broom"
        assert ImagingMode.FRAME.value == "frame"


class TestSatelliteTypeEnum:
    """测试卫星类型枚举"""

    def test_satellite_type_values(self):
        """测试卫星类型值"""
        assert SatelliteType.OPTICAL_1.value == "optical_1"
        assert SatelliteType.OPTICAL_2.value == "optical_2"
        assert SatelliteType.SAR_1.value == "sar_1"
        assert SatelliteType.SAR_2.value == "sar_2"


class TestOrbitTypeEnum:
    """测试轨道类型枚举"""

    def test_orbit_type_values(self):
        """测试轨道类型值"""
        assert OrbitType.SSO.value == "SSO"
        assert OrbitType.LEO.value == "LEO"
        assert OrbitType.MEO.value == "MEO"
        assert OrbitType.GEO.value == "GEO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
