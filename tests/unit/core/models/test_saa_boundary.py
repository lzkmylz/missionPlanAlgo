"""
SAA 边界模型单元测试

TDD 流程：先写测试，再实现
"""

import pytest
import math

from core.models.saa_boundary import SAABoundaryModel


class TestSAABoundaryModel:
    """测试 SAA 边界模型"""

    def test_default_parameters(self):
        """测试默认参数"""
        model = SAABoundaryModel()

        assert model.center_lon == -45.0
        assert model.center_lat == -25.0
        assert model.semi_major == 40.0
        assert model.semi_minor == 30.0

    def test_custom_parameters(self):
        """测试自定义参数"""
        model = SAABoundaryModel(
            center_lon=-40.0,
            center_lat=-20.0,
            semi_major=35.0,
            semi_minor=25.0
        )

        assert model.center_lon == -40.0
        assert model.center_lat == -20.0
        assert model.semi_major == 35.0
        assert model.semi_minor == 25.0

    def test_is_inside_center(self):
        """测试中心点肯定在 SAA 内"""
        model = SAABoundaryModel()

        # 中心点
        assert model.is_inside(-45.0, -25.0) is True

    def test_is_inside_boundary(self):
        """测试边界上的点"""
        model = SAABoundaryModel()

        # 东边界点 (中心 + 半长轴, 同纬度)
        assert model.is_inside(-5.0, -25.0) is True  # -45 + 40

        # 西边界点 (中心 - 半长轴, 同纬度)
        assert model.is_inside(-85.0, -25.0) is True  # -45 - 40

        # 北边界点 (同经度, 中心 + 半短轴)
        assert model.is_inside(-45.0, 5.0) is True  # -25 + 30

        # 南边界点 (同经度, 中心 - 半短轴)
        assert model.is_inside(-45.0, -55.0) is True  # -25 - 30

    def test_is_outside(self):
        """测试 SAA 外的点"""
        model = SAABoundaryModel()

        # 远东（超出东边界）
        assert model.is_inside(0.0, -25.0) is False

        # 远西（超出西边界）
        assert model.is_inside(-100.0, -25.0) is False

        # 远北（超出北边界）
        assert model.is_inside(-45.0, 20.0) is False

        # 远南（超出南边界）
        assert model.is_inside(-45.0, -70.0) is False

        # 中国区域（肯定不在 SAA）
        assert model.is_inside(116.0, 40.0) is False

        # 欧洲区域（肯定不在 SAA）
        assert model.is_inside(10.0, 50.0) is False

    def test_is_inside_diagonal(self):
        """测试对角线上的内部点"""
        model = SAABoundaryModel()

        # 东北方向内部点（约0.7倍半径处）
        assert model.is_inside(-45.0 + 28.0, -25.0 + 21.0) is True  # 0.7 * 40, 0.7 * 30

        # 西南方向内部点
        assert model.is_inside(-45.0 - 28.0, -25.0 - 21.0) is True

    def test_is_inside_just_outside_diagonal(self):
        """测试刚好在对角线外的点"""
        model = SAABoundaryModel()

        # 东北方向刚好外部点（1.1倍半径处）
        assert model.is_inside(-45.0 + 44.0, -25.0 + 33.0) is False  # 1.1 * 40, 1.1 * 30

    def test_just_outside_boundary(self):
        """测试边界外侧紧挨着的点"""
        model = SAABoundaryModel()

        # 东边界外侧（ epsilon > 0）
        assert model.is_inside(-4.9, -25.0) is False  # 稍微超出东边界

        # 西边界外侧
        assert model.is_inside(-85.1, -25.0) is False

        # 北边界外侧
        assert model.is_inside(-45.0, 5.1) is False

        # 南边界外侧
        assert model.is_inside(-45.0, -55.1) is False

    def test_invalid_longitude(self):
        """测试无效经度"""
        model = SAABoundaryModel()

        with pytest.raises(ValueError, match="longitude"):
            model.is_inside(181.0, 0.0)

        with pytest.raises(ValueError, match="longitude"):
            model.is_inside(-181.0, 0.0)

    def test_invalid_latitude(self):
        """测试无效纬度"""
        model = SAABoundaryModel()

        with pytest.raises(ValueError, match="latitude"):
            model.is_inside(0.0, 91.0)

        with pytest.raises(ValueError, match="latitude"):
            model.is_inside(0.0, -91.0)

    def test_boundary_extreme_values(self):
        """测试边界极值"""
        model = SAABoundaryModel()

        # 有效范围边界
        assert model.is_inside(180.0, 0.0) is not None  # 不应该抛出异常
        assert model.is_inside(-180.0, 0.0) is not None
        assert model.is_inside(0.0, 90.0) is not None
        assert model.is_inside(0.0, -90.0) is not None

    def test_get_boundary_points_default(self):
        """测试获取默认边界点"""
        model = SAABoundaryModel()

        points = model.get_boundary_points()

        assert len(points) == 36

        # 检查返回格式
        for lon, lat in points:
            assert isinstance(lon, float)
            assert isinstance(lat, float)
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90

    def test_get_boundary_points_custom(self):
        """测试获取自定义数量边界点"""
        model = SAABoundaryModel()

        points = model.get_boundary_points(num_points=72)

        assert len(points) == 72

    def test_boundary_points_on_boundary(self):
        """测试边界点确实在边界上"""
        model = SAABoundaryModel()

        points = model.get_boundary_points(num_points=72)

        # 所有边界点都应该满足 is_inside == True
        for lon, lat in points:
            assert model.is_inside(lon, lat) is True

    def test_frozen_dataclass(self):
        """测试 dataclass 是不可变的"""
        model = SAABoundaryModel()

        with pytest.raises(AttributeError):
            model.center_lon = -50.0


class TestSAABoundaryRealWorld:
    """真实世界场景测试"""

    def test_brazil_inside(self):
        """测试巴西区域（应在 SAA 内）"""
        model = SAABoundaryModel()

        # 巴西利亚（应该在边界内或附近）
        result = model.is_inside(-47.9, -15.8)
        # 巴西利亚大约在西经48度，南纬16度，应该在SAA内部

        # 圣保罗（西经47度，南纬24度）应该在SAA内
        assert model.is_inside(-47.0, -24.0) is True

        # 里约热内卢（西经43度，南纬23度）应该在SAA内
        assert model.is_inside(-43.0, -23.0) is True

    def test_south_africa_varies(self):
        """测试南非（部分可能在SAA边缘）"""
        model = SAABoundaryModel()

        # 开普敦（东经18度，南纬34度）肯定不在SAA
        assert model.is_inside(18.0, -34.0) is False

        # 布宜诺斯艾利斯（西经58度，南纬35度）应该在SAA内
        assert model.is_inside(-58.0, -35.0) is True
