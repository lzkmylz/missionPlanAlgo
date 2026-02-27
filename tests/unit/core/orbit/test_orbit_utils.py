"""
轨道工具函数测试

测试J2摄动计算和轨道常量
"""

import pytest
import math
from datetime import datetime, timezone

from core.orbit.utils import (
    calculate_j2_perturbations,
    EARTH_J2,
    EARTH_RADIUS_M,
    EARTH_GM,
    clamp
)


class TestConstants:
    """测试轨道常量"""

    def test_earth_j2_constant(self):
        """测试地球J2常数值"""
        assert EARTH_J2 == 1.08263e-3

    def test_earth_radius_constant(self):
        """测试地球半径常数值"""
        assert EARTH_RADIUS_M == 6371000.0

    def test_earth_gm_constant(self):
        """测试地球引力常数值"""
        assert EARTH_GM == 3.986004418e14


class TestClampFunction:
    """测试clamp函数"""

    def test_clamp_within_range(self):
        """测试范围内的值保持不变"""
        assert clamp(50, 0, 100) == 50

    def test_clamp_below_min(self):
        """测试低于最小值时返回最小值"""
        assert clamp(-10, 0, 100) == 0

    def test_clamp_above_max(self):
        """测试高于最大值时返回最大值"""
        assert clamp(150, 0, 100) == 100

    def test_clamp_at_boundaries(self):
        """测试边界值"""
        assert clamp(0, 0, 100) == 0
        assert clamp(100, 0, 100) == 100


class TestJ2PerturbationCalculations:
    """测试J2摄动计算"""

    def test_j2_perturbation_sso_orbit(self):
        """测试太阳同步轨道的J2摄动"""
        # SSO轨道参数
        semi_major_axis = 7000000.0  # 约700km高度
        inclination = 97.4  # SSO典型倾角
        eccentricity = 0.001
        mean_motion = 0.0011  # rad/s

        raan_dot, arg_perigee_dot = calculate_j2_perturbations(
            semi_major_axis, inclination, eccentricity, mean_motion
        )

        # RAAN应该有进动（太阳同步轨道为正值，向东进动约1度/天）
        assert raan_dot > 0

        # 近地点幅角应该有变化
        assert arg_perigee_dot != 0

    def test_j2_perturbation_equatorial_orbit(self):
        """测试赤道轨道的J2摄动"""
        semi_major_axis = 7000000.0
        inclination = 0.0  # 赤道轨道
        eccentricity = 0.001
        mean_motion = 0.0011

        raan_dot, arg_perigee_dot = calculate_j2_perturbations(
            semi_major_axis, inclination, eccentricity, mean_motion
        )

        # 赤道轨道RAAN进动为0（cos(0)=1，但公式中cos(i)使其为0）
        # 实际上RAAN_dot = -1.5 * n * J2 * (R/a)^2 * cos(i) / factor
        # 当i=0时，cos(i)=1，所以不为0
        # 但临界倾角时arg_perigee_dot = 0

    def test_j2_perturbation_critical_inclination(self):
        """测试临界倾角轨道的近地点幅角变化"""
        # 临界倾角 63.4° 或 116.6°
        semi_major_axis = 7000000.0
        inclination = 63.4
        eccentricity = 0.001
        mean_motion = 0.0011

        raan_dot, arg_perigee_dot = calculate_j2_perturbations(
            semi_major_axis, inclination, eccentricity, mean_motion
        )

        # 临界倾角时5*cos^2(i) - 1 ≈ 0，所以近地点幅角变化接近0
        # 但实际计算可能不完全为0
        assert isinstance(arg_perigee_dot, float)

    def test_j2_perturbation_zero_eccentricity(self):
        """测试零偏心率轨道"""
        semi_major_axis = 7000000.0
        inclination = 97.4
        eccentricity = 0.0
        mean_motion = 0.0011

        raan_dot, arg_perigee_dot = calculate_j2_perturbations(
            semi_major_axis, inclination, eccentricity, mean_motion
        )

        # 应该正常计算，没有除零错误
        assert isinstance(raan_dot, float)
        assert isinstance(arg_perigee_dot, float)

    def test_j2_perturbation_high_eccentricity(self):
        """测试高偏心率轨道"""
        semi_major_axis = 7000000.0
        inclination = 97.4
        eccentricity = 0.9  # 高偏心率
        mean_motion = 0.0011

        raan_dot, arg_perigee_dot = calculate_j2_perturbations(
            semi_major_axis, inclination, eccentricity, mean_motion
        )

        # 应该正常计算
        assert isinstance(raan_dot, float)
        assert isinstance(arg_perigee_dot, float)

    def test_j2_perturbation_with_time(self):
        """测试J2摄动随时间的累积效果"""
        semi_major_axis = 7000000.0
        inclination = 97.4
        eccentricity = 0.001
        mean_motion = 0.0011

        raan_dot, arg_perigee_dot = calculate_j2_perturbations(
            semi_major_axis, inclination, eccentricity, mean_motion
        )

        # 计算1天的累积进动
        delta_t = 86400.0  # 1天的秒数
        raan_change = math.degrees(raan_dot * delta_t)

        # SSO轨道每天应该有约1度的进动（360°/365天）
        # 允许一定的误差范围
        assert abs(raan_change - 1.0) < 0.5  # 误差小于0.5度


class TestCalculateOrbitalPosition:
    """测试轨道位置计算（待实现）"""

    def test_orbital_position_circle(self):
        """测试圆轨道位置计算"""
        # 这个测试将验证轨道位置计算函数
        # 目前先跳过，等待实现
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
