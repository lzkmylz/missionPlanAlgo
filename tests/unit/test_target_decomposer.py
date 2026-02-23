"""
目标分解器测试

TDD测试文件 - 实现设计文档第11章的目标分解模块
"""

import pytest
from datetime import datetime

from core.decomposer.base_decomposer import BaseDecomposer, DecompositionStrategy
from core.decomposer.grid_decomposer import GridDecomposer
from core.decomposer.strip_decomposer import StripDecomposer
from core.decomposer.decomposer_factory import DecomposerFactory
from core.models import Target, TargetType, SatelliteType, ImagingMode


class TestDecompositionStrategy:
    """测试分解策略枚举"""

    def test_strategy_values(self):
        """测试策略枚举值"""
        assert DecompositionStrategy.GRID.value == "grid"
        assert DecompositionStrategy.STRIP.value == "strip"


class TestBaseDecomposer:
    """测试分解器基类"""

    def test_base_decomposer_is_abstract(self):
        """测试基类是抽象类"""
        with pytest.raises(TypeError):
            BaseDecomposer()


class TestGridDecomposer:
    """测试网格分解器（适用于光学卫星）"""

    def test_grid_decomposer_initialization(self):
        """测试网格分解器初始化"""
        decomposer = GridDecomposer(resolution=10.0)
        assert decomposer.resolution == 10.0
        assert decomposer.strategy == DecompositionStrategy.GRID

    def test_decompose_square_area(self):
        """测试分解正方形区域"""
        # 使用1km分辨率以提高测试速度
        decomposer = GridDecomposer(resolution=1000.0)

        # 创建100平方公里的正方形区域（1度 x 1度）
        target = Target(
            id="AREA-01",
            name="正方形区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (117.0, 39.0),
                (117.0, 40.0),
                (116.0, 40.0),
            ]
        )

        sub_targets = decomposer.decompose(target)

        # 应该生成多个子目标（1km分辨率大约生成100个）
        assert len(sub_targets) > 0
        # 每个子目标应该是点目标
        for st in sub_targets:
            assert st.target_type == TargetType.POINT

    def test_decompose_rectangular_area(self):
        """测试分解矩形区域"""
        decomposer = GridDecomposer(resolution=1000.0)

        # 创建长方形区域（2度 x 1度 = 200平方公里）
        target = Target(
            id="AREA-02",
            name="矩形区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (118.0, 39.0),  # 更宽
                (118.0, 40.0),
                (116.0, 40.0),
            ]
        )

        sub_targets = decomposer.decompose(target)

        # 矩形应该生成更多子目标
        assert len(sub_targets) > 0

    def test_grid_size_affects_subtarget_count(self):
        """测试网格大小影响子目标数量"""
        # 使用较小的区域以提高速度
        target = Target(
            id="AREA-01",
            name="测试区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (116.1, 39.0),
                (116.1, 39.1),
                (116.0, 39.1),
            ]
        )

        # 粗网格（大分辨率）
        coarse_decomposer = GridDecomposer(resolution=1000.0)
        coarse_subtargets = coarse_decomposer.decompose(target)

        # 细网格（小分辨率）
        fine_decomposer = GridDecomposer(resolution=500.0)
        fine_subtargets = fine_decomposer.decompose(target)

        # 细网格应该生成更多子目标
        assert len(fine_subtargets) >= len(coarse_subtargets)

    def test_decompose_point_target_raises_error(self):
        """测试分解点目标应该报错"""
        decomposer = GridDecomposer(resolution=10.0)

        point_target = Target(
            id="POINT-01",
            name="点目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5,
        )

        with pytest.raises(ValueError):
            decomposer.decompose(point_target)

    def test_subtarget_inherits_properties(self):
        """测试子目标继承父目标属性"""
        decomposer = GridDecomposer(resolution=500.0)  # 使用较大分辨率

        target = Target(
            id="AREA-01",
            name="测试区域",
            target_type=TargetType.AREA,
            priority=8,
            required_observations=3,
            area_vertices=[
                (116.0, 39.0),
                (116.1, 39.0),
                (116.1, 39.1),
                (116.0, 39.1),
            ]
        )

        sub_targets = decomposer.decompose(target)

        # 子目标应该继承父目标的优先级和观测次数
        for st in sub_targets:
            assert st.priority == target.priority
            assert st.required_observations == target.required_observations


class TestStripDecomposer:
    """测试条带分解器（适用于SAR卫星）"""

    def test_strip_decomposer_initialization(self):
        """测试条带分解器初始化"""
        decomposer = StripDecomposer(swath_width=10000.0)
        assert decomposer.swath_width == 10000.0
        assert decomposer.strategy == DecompositionStrategy.STRIP

    def test_decompose_for_stripmap(self):
        """测试为条带模式分解区域"""
        decomposer = StripDecomposer(swath_width=20000.0)  # 增大幅宽以减少条带数量

        # 使用较小的区域
        target = Target(
            id="AREA-01",
            name="测试区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (116.2, 39.0),
                (116.2, 39.2),
                (116.0, 39.2),
            ]
        )

        sub_targets = decomposer.decompose(target, ImagingMode.STRIPMAP)

        # 应该生成条带形状的子目标
        assert len(sub_targets) > 0
        # 条带子目标应该是区域类型（线观测任务）
        for st in sub_targets:
            assert st.target_type == TargetType.AREA
            assert len(st.area_vertices) >= 3

    def test_decompose_for_spotlight(self):
        """测试为聚束模式分解区域"""
        decomposer = StripDecomposer(swath_width=10000.0)

        target = Target(
            id="AREA-01",
            name="测试区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (117.0, 39.0),
                (117.0, 40.0),
                (116.0, 40.0),
            ]
        )

        sub_targets = decomposer.decompose(target, ImagingMode.SPOTLIGHT)

        # 聚束模式应该生成更多小块（逐点扫描）
        assert len(sub_targets) > 0

    def test_swath_width_affects_strip_count(self):
        """测试幅宽影响条带数量"""
        target = Target(
            id="AREA-01",
            name="测试区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (117.0, 39.0),
                (117.0, 40.0),
                (116.0, 40.0),
            ]
        )

        # 大幅宽（10km）
        wide_decomposer = StripDecomposer(swath_width=10000.0)
        wide_strips = wide_decomposer.decompose(target, ImagingMode.STRIPMAP)

        # 小幅宽（3km）
        narrow_decomposer = StripDecomposer(swath_width=3000.0)
        narrow_strips = narrow_decomposer.decompose(target, ImagingMode.STRIPMAP)

        # 小幅宽应该生成更多条带
        assert len(narrow_strips) >= len(wide_strips)

    def test_overlap_ratio(self):
        """测试条带重叠率"""
        decomposer = StripDecomposer(swath_width=10000.0, overlap_ratio=0.2)

        target = Target(
            id="AREA-01",
            name="测试区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (117.0, 39.0),
                (117.0, 40.0),
                (116.0, 40.0),
            ]
        )

        sub_targets = decomposer.decompose(target, ImagingMode.STRIPMAP)

        # 有重叠应该生成更多条带
        assert len(sub_targets) > 0


class TestDecomposerFactory:
    """测试分解器工厂"""

    def test_create_grid_decomposer(self):
        """测试创建网格分解器"""
        decomposer = DecomposerFactory.create(
            DecompositionStrategy.GRID,
            resolution=10.0
        )
        assert isinstance(decomposer, GridDecomposer)

    def test_create_strip_decomposer(self):
        """测试创建条带分解器"""
        decomposer = DecomposerFactory.create(
            DecompositionStrategy.STRIP,
            swath_width=10000.0
        )
        assert isinstance(decomposer, StripDecomposer)

    def test_create_for_satellite_type(self):
        """测试根据卫星类型创建分解器"""
        # 光学卫星应该使用网格分解
        optical_decomposer = DecomposerFactory.create_for_satellite_type(
            SatelliteType.OPTICAL_1,
            resolution=10.0
        )
        assert isinstance(optical_decomposer, GridDecomposer)

        # SAR卫星应该使用条带分解
        sar_decomposer = DecomposerFactory.create_for_satellite_type(
            SatelliteType.SAR_1,
            swath_width=10000.0
        )
        assert isinstance(sar_decomposer, StripDecomposer)

    def test_invalid_strategy_raises_error(self):
        """测试无效策略应该报错"""
        with pytest.raises(ValueError):
            DecomposerFactory.create("invalid_strategy")


class TestDecomposerIntegration:
    """测试分解器集成场景"""

    def test_large_area_decomposition(self):
        """测试大区域分解"""
        # 创建一个100平方公里的大区域（1度 x 1度）
        target = Target(
            id="LARGE-AREA",
            name="大区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (117.0, 39.0),
                (117.0, 40.0),
                (116.0, 40.0),
            ]
        )

        # 使用网格分解（使用较大分辨率以提高速度）
        grid_decomposer = GridDecomposer(resolution=1000.0)
        grid_subtargets = grid_decomposer.decompose(target)

        # 使用条带分解（使用较大幅宽以减少条带数量）
        strip_decomposer = StripDecomposer(swath_width=20000.0)
        strip_subtargets = strip_decomposer.decompose(target, ImagingMode.STRIPMAP)

        # 两种方法都应该生成多个子目标
        assert len(grid_subtargets) > 0
        assert len(strip_subtargets) > 0

    def test_complex_polygon_decomposition(self):
        """测试复杂多边形分解"""
        # 创建L形区域
        target = Target(
            id="L-SHAPE",
            name="L形区域",
            target_type=TargetType.AREA,
            priority=5,
            area_vertices=[
                (116.0, 39.0),
                (116.5, 39.0),
                (116.5, 39.3),
                (117.0, 39.3),
                (117.0, 39.5),
                (116.0, 39.5),
            ]
        )

        decomposer = GridDecomposer(resolution=200.0)
        sub_targets = decomposer.decompose(target)

        # 应该成功分解
        assert len(sub_targets) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
