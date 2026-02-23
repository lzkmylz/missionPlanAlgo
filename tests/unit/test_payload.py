"""
载荷模块测试

TDD测试文件 - 实现Imager基类及光学/SAR成像器
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from payload.base import Imager, ImagingMode
from payload.optical_imager import OpticalImager
from payload.sar_imager import SARImager, SARImagingMode


# Helper function to check if a SAR mode is supported (handles both enum types)
def _sar_mode_supported(imager, mode: SARImagingMode) -> bool:
    """Check if SAR mode is supported (works with both enum types in supported_modes)"""
    mode_value = mode.value
    return any(m.value == mode_value for m in imager.supported_modes)


class TestImagingMode:
    """测试成像模式枚举"""

    def test_imaging_mode_values(self):
        """测试成像模式枚举值"""
        assert ImagingMode.PUSH_BROOM.value == "push_broom"
        assert ImagingMode.FRAME.value == "frame"
        assert ImagingMode.SPOTLIGHT.value == "spotlight"
        assert ImagingMode.SLIDING_SPOTLIGHT.value == "sliding_spotlight"
        assert ImagingMode.STRIPMAP.value == "stripmap"


class TestSARImagingMode:
    """测试SAR成像模式枚举"""

    def test_sar_imaging_mode_values(self):
        """测试SAR成像模式枚举值"""
        assert SARImagingMode.SPOTLIGHT.value == "spotlight"
        assert SARImagingMode.SLIDING_SPOTLIGHT.value == "sliding_spotlight"
        assert SARImagingMode.STRIPMAP.value == "stripmap"


class TestImagerBase:
    """测试Imager基类"""

    def test_imager_is_abstract(self):
        """测试Imager是抽象基类"""
        with pytest.raises(TypeError):
            Imager(imager_id="TEST-01")

    def test_concrete_imager_requires_methods(self):
        """测试具体实现类必须实现抽象方法"""
        class IncompleteImager(Imager):
            pass

        with pytest.raises(TypeError):
            IncompleteImager(imager_id="TEST-01")


class TestOpticalImager:
    """测试光学成像器"""

    def test_optical_imager_initialization(self):
        """测试光学成像器初始化"""
        imager = OpticalImager(
            imager_id="OPT-01",
            resolution=0.5,
            swath_width=10.0,
            supported_modes=[ImagingMode.PUSH_BROOM]
        )

        assert imager.imager_id == "OPT-01"
        assert imager.resolution == 0.5
        assert imager.swath_width == 10.0
        assert ImagingMode.PUSH_BROOM in imager.supported_modes

    def test_optical_imager_default_modes(self):
        """测试光学成像器默认模式"""
        imager = OpticalImager(imager_id="OPT-01")

        assert ImagingMode.PUSH_BROOM in imager.supported_modes
        assert ImagingMode.FRAME in imager.supported_modes

    def test_optical_imager_mode_support(self):
        """测试光学成像器模式支持检查"""
        imager = OpticalImager(
            imager_id="OPT-01",
            supported_modes=[ImagingMode.PUSH_BROOM]
        )

        assert imager.supports_mode(ImagingMode.PUSH_BROOM) == True
        assert imager.supports_mode(ImagingMode.FRAME) == False
        assert imager.supports_mode(ImagingMode.SPOTLIGHT) == False

    def test_optical_imager_calculate_imaging_time(self):
        """测试光学成像器计算成像时间"""
        imager = OpticalImager(
            imager_id="OPT-01",
            resolution=1.0,
            swath_width=10.0
        )

        # 测试点目标成像时间
        duration = imager.calculate_imaging_time(
            target_size=(1000, 1000),  # 1km x 1km
            mode=ImagingMode.PUSH_BROOM
        )

        assert duration > 0
        assert isinstance(duration, (int, float))

    def test_optical_imager_get_specs(self):
        """测试光学成像器获取规格"""
        imager = OpticalImager(
            imager_id="OPT-01",
            resolution=0.5,
            swath_width=10.0,
            focal_length=1.0,
            aperture=0.3
        )

        specs = imager.get_specs()

        assert specs['imager_id'] == "OPT-01"
        assert specs['resolution'] == 0.5
        assert specs['swath_width'] == 10.0
        assert specs['focal_length'] == 1.0
        assert specs['aperture'] == 0.3
        assert specs['imager_type'] == "optical"

    def test_optical_imager_invalid_mode(self):
        """测试光学成像器无效模式处理"""
        imager = OpticalImager(imager_id="OPT-01")

        with pytest.raises(ValueError):
            imager.calculate_imaging_time(
                target_size=(1000, 1000),
                mode=ImagingMode.SPOTLIGHT  # SAR模式不支持
            )


class TestSARImager:
    """测试SAR成像器"""

    def test_sar_imager_initialization(self):
        """测试SAR成像器初始化"""
        imager = SARImager(
            imager_id="SAR-01",
            resolution=1.0,
            swath_width=20.0,
            supported_modes=[SARImagingMode.STRIPMAP]
        )

        assert imager.imager_id == "SAR-01"
        assert imager.resolution == 1.0
        assert imager.swath_width == 20.0
        assert _sar_mode_supported(imager, SARImagingMode.STRIPMAP)

    def test_sar_imager_default_modes(self):
        """测试SAR成像器默认模式"""
        imager = SARImager(imager_id="SAR-01")

        assert _sar_mode_supported(imager, SARImagingMode.SPOTLIGHT)
        assert _sar_mode_supported(imager, SARImagingMode.SLIDING_SPOTLIGHT)
        assert _sar_mode_supported(imager, SARImagingMode.STRIPMAP)

    def test_sar_imager_mode_support(self):
        """测试SAR成像器模式支持检查"""
        imager = SARImager(
            imager_id="SAR-01",
            supported_modes=[SARImagingMode.SPOTLIGHT, SARImagingMode.STRIPMAP]
        )

        assert imager.supports_mode(SARImagingMode.SPOTLIGHT) == True
        assert imager.supports_mode(SARImagingMode.STRIPMAP) == True
        assert imager.supports_mode(SARImagingMode.SLIDING_SPOTLIGHT) == False

    def test_sar_imager_spotlight_mode(self):
        """测试SAR聚束模式"""
        imager = SARImager(imager_id="SAR-01")

        duration = imager.calculate_imaging_time(
            target_size=(5000, 5000),  # 5km x 5km
            mode=SARImagingMode.SPOTLIGHT
        )

        assert duration > 0
        # 聚束模式通常需要更长时间

    def test_sar_imager_stripmap_mode(self):
        """测试SAR条带模式"""
        imager = SARImager(imager_id="SAR-01")

        duration = imager.calculate_imaging_time(
            target_size=(10000, 100000),  # 10km x 100km
            mode=SARImagingMode.STRIPMAP
        )

        assert duration > 0

    def test_sar_imager_sliding_spotlight_mode(self):
        """测试SAR滑动聚束模式"""
        imager = SARImager(imager_id="SAR-01")

        duration = imager.calculate_imaging_time(
            target_size=(10000, 20000),  # 10km x 20km
            mode=SARImagingMode.SLIDING_SPOTLIGHT
        )

        assert duration > 0

    def test_sar_imager_get_specs(self):
        """测试SAR成像器获取规格"""
        imager = SARImager(
            imager_id="SAR-01",
            resolution=1.0,
            swath_width=20.0,
            band="X",
            polarization="VV"
        )

        specs = imager.get_specs()

        assert specs['imager_id'] == "SAR-01"
        assert specs['resolution'] == 1.0
        assert specs['swath_width'] == 20.0
        assert specs['band'] == "X"
        assert specs['polarization'] == "VV"
        assert specs['imager_type'] == "sar"

    def test_sar_imager_invalid_mode(self):
        """测试SAR成像器无效模式处理"""
        imager = SARImager(imager_id="SAR-01")

        with pytest.raises(ValueError):
            imager.calculate_imaging_time(
                target_size=(1000, 1000),
                mode=ImagingMode.PUSH_BROOM  # 光学模式不支持
            )

    def test_sar_imager_look_angle(self):
        """测试SAR成像器视角设置"""
        imager = SARImager(
            imager_id="SAR-01",
            min_look_angle=20.0,
            max_look_angle=50.0
        )

        assert imager.min_look_angle == 20.0
        assert imager.max_look_angle == 50.0


class TestImagerEdgeCases:
    """测试边界情况"""

    def test_imager_with_zero_resolution(self):
        """测试分辨率为零的情况"""
        with pytest.raises(ValueError):
            OpticalImager(imager_id="OPT-01", resolution=0)

    def test_imager_with_negative_swath(self):
        """测试负幅宽的情况"""
        with pytest.raises(ValueError):
            OpticalImager(imager_id="OPT-01", swath_width=-10.0)

    def test_imager_with_empty_modes(self):
        """测试空模式列表"""
        with pytest.raises(ValueError):
            OpticalImager(imager_id="OPT-01", supported_modes=[])

    def test_sar_look_angle_validation(self):
        """测试SAR视角验证"""
        with pytest.raises(ValueError):
            SARImager(imager_id="SAR-01", min_look_angle=60.0, max_look_angle=30.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
