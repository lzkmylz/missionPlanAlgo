"""
分辨率约束检查测试

TDD测试用例，验证卫星分辨率是否能满足目标需求
"""

import pytest
from datetime import datetime

from core.models.satellite import Satellite, SatelliteType, SatelliteCapabilities, ImagingMode
from core.models.target import Target, TargetType
from scheduler.common.constraint_checker import CapabilityChecker, ConstraintType


class TestResolutionConstraint:
    """分辨率约束检查测试类"""

    @pytest.fixture
    def high_res_optical_satellite(self):
        """创建高分辨率光学卫星（支持0.5m）"""
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
            resolution=1.0,  # 默认分辨率
            imaging_mode_details=[
                {
                    "mode_id": "push_broom",
                    "resolution": 1.0,
                    "swath_width": 15000.0
                },
                {
                    "mode_id": "push_broom_high_res",
                    "resolution": 0.5,
                    "swath_width": 7500.0
                }
            ]
        )
        return Satellite(
            id="OPT-HIGH-01",
            name="高分辨率光学卫星",
            sat_type=SatelliteType.OPTICAL_2,
            capabilities=capabilities
        )

    @pytest.fixture
    def standard_optical_satellite(self):
        """创建标准分辨率光学卫星（仅1.0m）"""
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM],
            resolution=1.0,
            imaging_mode_details=[
                {
                    "mode_id": "push_broom",
                    "resolution": 1.0,
                    "swath_width": 15000.0
                }
            ]
        )
        return Satellite(
            id="OPT-STD-01",
            name="标准光学卫星",
            sat_type=SatelliteType.OPTICAL_1,
            capabilities=capabilities
        )

    @pytest.fixture
    def sar_satellite(self):
        """创建SAR卫星（多模式）"""
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.STRIPMAP, ImagingMode.SPOTLIGHT, ImagingMode.SLIDING_SPOTLIGHT],
            resolution=3.0,
            imaging_mode_details=[
                {
                    "mode_id": "stripmap",
                    "resolution": 3.0,
                    "swath_width": 30000.0
                },
                {
                    "mode_id": "spotlight",
                    "resolution": 1.0,
                    "swath_width": 10000.0
                },
                {
                    "mode_id": "sliding_spotlight",
                    "resolution": 2.0,
                    "swath_width": 20000.0
                }
            ]
        )
        return Satellite(
            id="SAR-01",
            name="SAR卫星",
            sat_type=SatelliteType.SAR_1,
            capabilities=capabilities
        )

    @pytest.fixture
    def high_res_target(self):
        """创建高分辨率需求目标（0.5m）"""
        return Target(
            id="TGT-HIGH-01",
            name="高分辨率目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=40.0,
            priority=1,
            resolution_required=0.5
        )

    @pytest.fixture
    def standard_res_target(self):
        """创建标准分辨率需求目标（1.0m）"""
        return Target(
            id="TGT-STD-01",
            name="标准分辨率目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=40.0,
            priority=4,
            resolution_required=1.0
        )

    @pytest.fixture
    def low_res_target(self):
        """创建低分辨率需求目标（3.0m）"""
        return Target(
            id="TGT-LOW-01",
            name="低分辨率目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=40.0,
            priority=8,
            resolution_required=3.0
        )

    @pytest.fixture
    def no_res_target(self):
        """创建无分辨率需求目标"""
        return Target(
            id="TGT-NO-01",
            name="无分辨率要求目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=40.0,
            priority=5
            # resolution_required defaults to 10.0
        )

    # ==================== 测试用例 ====================

    def test_high_res_satellite_can_satisfy_high_res_target(
        self, high_res_optical_satellite, high_res_target
    ):
        """测试：高分辨率卫星可以满足高分辨率目标（0.5m）"""
        checker = CapabilityChecker()
        result = checker.check_capability(high_res_optical_satellite, high_res_target)

        assert result.feasible is True
        assert ConstraintType.CAPABILITY not in result.violations

    def test_standard_satellite_cannot_satisfy_high_res_target(
        self, standard_optical_satellite, high_res_target
    ):
        """测试：标准分辨率卫星不能满足高分辨率目标（0.5m）"""
        checker = CapabilityChecker()
        result = checker.check_capability(standard_optical_satellite, high_res_target)

        assert result.feasible is False
        assert ConstraintType.CAPABILITY in result.violations
        assert "Resolution insufficient" in str(result.details)

    def test_high_res_satellite_can_satisfy_standard_res_target(
        self, high_res_optical_satellite, standard_res_target
    ):
        """测试：高分辨率卫星可以满足标准分辨率目标（1.0m）"""
        checker = CapabilityChecker()
        result = checker.check_capability(high_res_optical_satellite, standard_res_target)

        assert result.feasible is True
        assert ConstraintType.CAPABILITY not in result.violations

    def test_standard_satellite_can_satisfy_standard_res_target(
        self, standard_optical_satellite, standard_res_target
    ):
        """测试：标准分辨率卫星可以满足标准分辨率目标（1.0m）"""
        checker = CapabilityChecker()
        result = checker.check_capability(standard_optical_satellite, standard_res_target)

        assert result.feasible is True
        assert ConstraintType.CAPABILITY not in result.violations

    def test_satellite_can_satisfy_lower_res_target(
        self, standard_optical_satellite, low_res_target
    ):
        """测试：卫星分辨率优于目标需求时可以通过（1.0m vs 3.0m）"""
        checker = CapabilityChecker()
        result = checker.check_capability(standard_optical_satellite, low_res_target)

        assert result.feasible is True
        assert ConstraintType.CAPABILITY not in result.violations

    def test_sar_spotlight_mode_can_satisfy_high_res(
        self, sar_satellite, high_res_target
    ):
        """测试：SAR聚束模式（1.0m）不能满足0.5m需求"""
        checker = CapabilityChecker()
        result = checker.check_capability(sar_satellite, high_res_target)

        # SAR spotlight是1.0m，不能满足0.5m需求
        assert result.feasible is False
        assert ConstraintType.CAPABILITY in result.violations

    def test_sar_spotlight_mode_can_satisfy_standard_res(
        self, sar_satellite, standard_res_target
    ):
        """测试：SAR聚束模式（1.0m）可以满足标准分辨率需求"""
        checker = CapabilityChecker()
        result = checker.check_capability(sar_satellite, standard_res_target)

        assert result.feasible is True
        assert ConstraintType.CAPABILITY not in result.violations

    def test_no_resolution_requirement_always_passes(
        self, standard_optical_satellite, no_res_target
    ):
        """测试：目标无分辨率要求时，任何卫星都可以通过"""
        checker = CapabilityChecker()
        result = checker.check_capability(standard_optical_satellite, no_res_target)

        assert result.feasible is True
        assert ConstraintType.CAPABILITY not in result.violations

    def test_satellite_capabilities_get_mode_resolution(self, high_res_optical_satellite):
        """测试：SatelliteCapabilities.get_mode_resolution()方法"""
        caps = high_res_optical_satellite.capabilities

        # 应该能够通过方法获取模式分辨率
        assert hasattr(caps, 'get_mode_resolution')
        assert caps.get_mode_resolution(ImagingMode.PUSH_BROOM) == 1.0
        assert caps.get_mode_resolution("push_broom_high_res") == 0.5
        assert caps.get_mode_resolution("non_existent") is None

    def test_satellite_capabilities_can_satisfy_resolution(self, high_res_optical_satellite):
        """测试：SatelliteCapabilities.can_satisfy_resolution()方法"""
        caps = high_res_optical_satellite.capabilities

        # 应该能够通过方法检查分辨率满足能力
        assert hasattr(caps, 'can_satisfy_resolution')
        assert caps.can_satisfy_resolution(0.5) is True   # 有0.5m模式，恰好满足
        assert caps.can_satisfy_resolution(0.3) is False  # 最高0.5m，不能满足0.3m需求
        assert caps.can_satisfy_resolution(1.0) is True   # 有0.5m和1.0m模式，都满足1.0m需求
        assert caps.can_satisfy_resolution(2.0) is True   # 有1.0m模式，优于2.0m需求

    def test_standard_satellite_capabilities_cannot_satisfy_high_res(
        self, standard_optical_satellite
    ):
        """测试：标准卫星不能提供高分辨率"""
        caps = standard_optical_satellite.capabilities

        assert caps.can_satisfy_resolution(0.5) is False  # 没有0.5m模式
        assert caps.can_satisfy_resolution(1.0) is True   # 有1.0m模式

    def test_sar_satellite_capabilities_resolution_modes(self, sar_satellite):
        """测试：SAR卫星的多模式分辨率检查"""
        caps = sar_satellite.capabilities

        assert caps.can_satisfy_resolution(0.5) is False  # 最低1.0m
        assert caps.can_satisfy_resolution(1.0) is True   # spotlight模式
        assert caps.can_satisfy_resolution(2.0) is True   # spotlight或sliding_spotlight
        assert caps.can_satisfy_resolution(3.0) is True   # 所有模式都满足


class TestResolutionConstraintWithSpecificMode:
    """测试指定成像模式时的分辨率约束检查"""

    @pytest.fixture
    def multi_mode_satellite(self):
        """创建多模式卫星"""
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
            resolution=1.0,
            imaging_mode_details=[
                {
                    "mode_id": "push_broom",
                    "resolution": 1.0,
                    "swath_width": 15000.0
                },
                {
                    "mode_id": "frame",
                    "resolution": 0.5,
                    "swath_width": 10000.0
                }
            ]
        )
        return Satellite(
            id="OPT-MULTI-01",
            name="多模式光学卫星",
            sat_type=SatelliteType.OPTICAL_2,
            capabilities=capabilities
        )

    @pytest.fixture
    def high_res_target(self):
        """创建高分辨率需求目标（0.5m）"""
        return Target(
            id="TGT-HIGH-01",
            name="高分辨率目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=40.0,
            priority=1,
            resolution_required=0.5
        )

    def test_specific_mode_meets_requirement(
        self, multi_mode_satellite, high_res_target
    ):
        """测试：指定满足分辨率需求的成像模式可以通过"""
        checker = CapabilityChecker()

        # 指定frame模式（0.5m）应该可以通过
        result = checker.check_capability(
            multi_mode_satellite, high_res_target, imaging_mode=ImagingMode.FRAME
        )

        assert result.feasible is True

    def test_specific_mode_does_not_meet_requirement(
        self, multi_mode_satellite, high_res_target
    ):
        """测试：指定不满足分辨率需求的成像模式应该失败"""
        checker = CapabilityChecker()

        # 指定push_broom模式（1.0m）不应该通过0.5m需求
        result = checker.check_capability(
            multi_mode_satellite, high_res_target, imaging_mode=ImagingMode.PUSH_BROOM
        )

        assert result.feasible is False
        assert ConstraintType.CAPABILITY in result.violations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
