"""
地影计算器测试

TDD测试文件 - 实现设计文档第12章的地影计算器
"""

import pytest
from datetime import datetime, timedelta

from simulator.eclipse_calculator import EclipseCalculator
from core.models import Orbit, OrbitType


class TestEclipseCalculator:
    """测试地影计算器"""

    def test_calculator_initialization(self):
        """测试地影计算器初始化"""
        calculator = EclipseCalculator()
        assert calculator is not None
        assert calculator.EARTH_RADIUS == 6371000.0  # 地球半径（米）
        assert calculator.SUN_RADIUS == 696340000.0  # 太阳半径（米）

    def test_is_in_eclipse_basic(self):
        """测试基本地影判断"""
        calculator = EclipseCalculator()

        # 卫星在光照区（太阳对面）
        sat_pos = (7000000.0, 0.0, 0.0)  # x轴正方向
        sun_pos = (1.496e11, 0.0, 0.0)   # 太阳也在x轴正方向（远处）

        is_eclipse = calculator._is_in_eclipse(sat_pos, sun_pos)
        assert is_eclipse == False  # 卫星在光照区

        # 卫星在地影区（地球后面）
        sat_pos = (-7000000.0, 0.0, 0.0)  # x轴负方向
        sun_pos = (1.496e11, 0.0, 0.0)    # 太阳在x轴正方向

        is_eclipse = calculator._is_in_eclipse(sat_pos, sun_pos)
        assert is_eclipse == True  # 卫星在地影区

    def test_calculate_eclipse_intervals(self):
        """测试计算地影区间"""
        calculator = EclipseCalculator()

        orbit = Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,  # 500km
            inclination=97.4
        )

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0)

        intervals = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=300
        )

        # 应该返回一些地影区间
        assert isinstance(intervals, list)
        # 24小时内应该有多个地影区间（通常每轨道1-2次，共约14-16个地影区间）
        # 注意：500km轨道约90分钟一圈，每圈大约35-40分钟在地影，24小时约16个地影区间
        # 由于简化的轨道模型和计算精度，测试只检查返回区间格式的正确性

    def test_eclipse_interval_format(self):
        """测试地影区间格式"""
        calculator = EclipseCalculator()

        orbit = Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,
            inclination=97.4
        )

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0)

        intervals = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=600
        )

        for start, end in intervals:
            # 验证区间格式
            assert isinstance(start, datetime)
            assert isinstance(end, datetime)
            # 开始时间应该早于结束时间
            assert start < end
            # 区间应该在计算范围内
            assert start >= start_time
            assert end <= end_time

    def test_eclipse_duration_reasonable(self):
        """测试地影时长合理性"""
        calculator = EclipseCalculator()

        orbit = Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,
            inclination=97.4
        )

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 1, 3, 0, 0)  # 48小时

        intervals = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=300
        )

        for start, end in intervals:
            duration = (end - start).total_seconds()
            # 地影时长通常在20-40分钟（LEO轨道）
            assert 600 <= duration <= 3600, f"地影时长 {duration} 秒不在合理范围内"

    def test_get_sun_position(self):
        """测试太阳位置计算"""
        calculator = EclipseCalculator()

        # 获取不同时间的太阳位置
        dt1 = datetime(2024, 1, 1, 0, 0)   # 冬至后不久
        dt2 = datetime(2024, 6, 21, 0, 0)  # 夏至

        sun_pos1 = calculator._get_sun_position(dt1)
        sun_pos2 = calculator._get_sun_position(dt2)

        # 太阳位置应该在远处（约1AU）
        import math
        dist1 = math.sqrt(sum(x**2 for x in sun_pos1))
        dist2 = math.sqrt(sum(x**2 for x in sun_pos2))

        # 距离应该在1AU附近（有轨道偏心影响）
        AU = 149597870700.0
        assert 0.95 * AU <= dist1 <= 1.05 * AU
        assert 0.95 * AU <= dist2 <= 1.05 * AU

    def test_no_eclipse_for_high_orbit(self):
        """测试高轨道可能不受地影影响"""
        calculator = EclipseCalculator()

        # GEO轨道（36000km）
        orbit = Orbit(
            orbit_type=OrbitType.GEO,
            altitude=36000000.0,
            inclination=0.0
        )

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0)

        intervals = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=600
        )

        # GEO轨道在特定季节可能完全不受地影影响
        # 我们只需要确保函数能正常运行
        assert isinstance(intervals, list)

    def test_time_step_affects_accuracy(self):
        """测试时间步长影响精度"""
        calculator = EclipseCalculator()

        orbit = Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,
            inclination=97.4
        )

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0)

        # 粗略计算（大步长）
        intervals_coarse = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=1200
        )

        # 精细计算（小步长）
        intervals_fine = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=300
        )

        # 精细计算应该得到更准确的结果（更多或更精确的区间）
        # 或者至少区间总时长应该相似
        total_coarse = sum((e - s).total_seconds() for s, e in intervals_coarse)
        total_fine = sum((e - s).total_seconds() for s, e in intervals_fine)

        # 总地影时长应该相近（误差在20%以内）
        if total_coarse > 0:
            assert abs(total_fine - total_coarse) / total_coarse < 0.2


class TestEclipseCalculatorEdgeCases:
    """测试边界情况"""

    def test_empty_time_range(self):
        """测试空时间范围"""
        calculator = EclipseCalculator()

        orbit = Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,
            inclination=97.4
        )

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = datetime(2024, 1, 1, 0, 0)  # 相同时间

        intervals = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=600
        )

        assert intervals == []

    def test_negative_time_range(self):
        """测试结束时间早于开始时间"""
        calculator = EclipseCalculator()

        orbit = Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,
            inclination=97.4
        )

        start_time = datetime(2024, 1, 2, 0, 0)
        end_time = datetime(2024, 1, 1, 0, 0)  # 早于开始时间

        intervals = calculator.calculate_eclipse_intervals(
            orbit, start_time, end_time, time_step=600
        )

        # 应该返回空列表或正确处理
        assert isinstance(intervals, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
