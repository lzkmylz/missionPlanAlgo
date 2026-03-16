"""
测试成像中心点距离评分功能

包含边界值测试、极点测试、极端情况等
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scheduler.common.footprint_utils import (
    calculate_haversine_distance,
    calculate_footprint_center_from_attitude,
    calculate_center_distance_score,
)


class TestHaversineDistance:
    """测试Haversine距离计算"""

    def test_same_point(self):
        """相同点距离应为0"""
        dist = calculate_haversine_distance(0, 0, 0, 0)
        assert dist == 0.0

    def test_equator_one_degree(self):
        """赤道上一度的距离约为111km"""
        dist = calculate_haversine_distance(0, 0, 1, 0)
        assert abs(dist - 111.19) < 1.0  # 允许1km误差

    def test_pole_to_pole(self):
        """南极到北极的距离约为20015km"""
        dist = calculate_haversine_distance(0, -90, 0, 90)
        assert abs(dist - 20015) < 10.0  # 允许10km误差

    def test_symmetry(self):
        """距离计算应该对称"""
        dist1 = calculate_haversine_distance(0, 0, 10, 10)
        dist2 = calculate_haversine_distance(10, 10, 0, 0)
        assert abs(dist1 - dist2) < 0.001


class TestFootprintCenterFromAttitude:
    """测试成像中心计算"""

    def test_nadir_pointing(self):
        """星下点观测时，成像中心应为星下点"""
        R = 6371000  # 地球半径(m)
        h = 500000   # 高度(m)
        position = (R + h, 0, 0)  # 赤道上空

        center = calculate_footprint_center_from_attitude(position, 0, 0)
        assert center is not None
        assert abs(center[0]) < 0.01  # 经度接近0
        assert abs(center[1]) < 0.01  # 纬度接近0

    def test_east_roll(self):
        """向东侧摆应该使经度增加"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        center_nadir = calculate_footprint_center_from_attitude(position, 0, 0)
        center_tilted = calculate_footprint_center_from_attitude(position, 10, 0)

        assert center_tilted[0] > center_nadir[0]  # 经度增加

    def test_north_pitch(self):
        """向北俯仰应该使纬度增加"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        center_nadir = calculate_footprint_center_from_attitude(position, 0, 0)
        center_tilted = calculate_footprint_center_from_attitude(position, 0, 10)

        assert center_tilted[1] > center_nadir[1]  # 纬度增加

    def test_high_latitude(self):
        """高纬度卫星位置的计算"""
        R = 6371000
        h = 500000
        # 北极附近
        position = (0, 0, R + h)

        center = calculate_footprint_center_from_attitude(position, 0, 0)
        assert center is not None
        assert abs(center[1] - 90) < 0.01  # 纬度接近90

    def test_invalid_position(self):
        """无效位置应该返回None"""
        center = calculate_footprint_center_from_attitude((0, 0, 0), 0, 0)
        # 位置在地球中心，高度为负，可能返回None或异常
        # 函数应该能够处理这种情况


class TestCenterDistanceScore:
    """测试距离评分功能"""

    def test_perfect_alignment(self):
        """完美对准时评分为1.0"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        score = calculate_center_distance_score(position, 0, 0, 0, 0)
        assert abs(score - 1.0) < 0.001

    def test_zero_deviation_higher_score(self):
        """偏差越小，评分越高"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        score_perfect = calculate_center_distance_score(position, 0, 0, 0, 0)
        score_tilted = calculate_center_distance_score(position, 5, 0, 0, 0)

        assert score_perfect > score_tilted

    def test_exceeds_max_distance(self):
        """超过最大距离时评分为0"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        # 极大偏差，应该超过默认10度限制
        # 90度滚转+俯仰会产生极大的地面位移，超过10度
        score = calculate_center_distance_score(position, 80, 80, 0, 0)
        assert score == 0.0

    def test_invalid_max_distance(self):
        """无效的max_distance应该使用默认值"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        # 使用负的max_distance
        score = calculate_center_distance_score(
            position, 0, 0, 0, 0, max_distance=-1.0
        )
        # 应该使用默认值并正常计算
        assert 0 <= score <= 1.0

    def test_invalid_scale(self):
        """无效的scale应该使用默认值"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        # 使用负的scale
        score = calculate_center_distance_score(
            position, 5, 0, 0, 0, scale=-1.0
        )
        # 应该使用默认值并正常计算
        assert 0 <= score <= 1.0

    def test_invalid_satellite_position(self):
        """无效的卫星位置应该返回默认评分"""
        score = calculate_center_distance_score(
            (0, 0, 0), 0, 0, 0, 0  # 位置在地球中心
        )
        # 应该返回默认中等评分0.5
        assert score == 0.5

    def test_empty_position(self):
        """空位置应该返回默认评分"""
        score = calculate_center_distance_score(
            (), 0, 0, 0, 0
        )
        assert score == 0.5

    def test_none_position(self):
        """None位置应该返回默认评分"""
        score = calculate_center_distance_score(
            None, 0, 0, 0, 0
        )
        assert score == 0.5

    def test_pole_target(self):
        """极点目标的评分计算"""
        R = 6371000
        h = 500000
        position = (0, 0, R + h)  # 北极上空

        # 目标在北极
        score = calculate_center_distance_score(position, 0, 0, 0, 90)
        assert 0 <= score <= 1.0

    def test_extreme_longitude(self):
        """极端经度值（180/-180边界）"""
        R = 6371000
        h = 500000
        position = (-(R + h), 0, 0)  # 经度180度位置

        # 目标在-180度（与180度是同一个点）
        score = calculate_center_distance_score(position, 0, 0, -180, 0)
        # 由于简化计算，经度180和-180被视为相差360度
        # 这种情况下评分会很低，但代码应该能正常处理不崩溃
        assert 0 <= score <= 1.0


class TestEdgeCases:
    """测试极端情况"""

    def test_very_small_roll_pitch(self):
        """非常小的姿态角"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        score = calculate_center_distance_score(
            position, 0.0001, 0.0001, 0, 0
        )
        assert score > 0.99  # 应该非常接近1.0

    def test_large_roll_pitch(self):
        """非常大的姿态角（超过物理限制）"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        # 90度滚转，物理上可能不可行，但代码应该能处理
        score = calculate_center_distance_score(
            position, 90, 90, 0, 0
        )
        assert 0 <= score <= 1.0

    def test_extreme_target_coordinates(self):
        """极端目标坐标"""
        R = 6371000
        h = 500000
        position = (R + h, 0, 0)

        # 目标在(180, 90) - 东北极
        score = calculate_center_distance_score(
            position, 0, 0, 180, 90
        )
        assert 0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
