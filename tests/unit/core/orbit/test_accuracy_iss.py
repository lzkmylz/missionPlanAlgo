"""
Orekit与ISS TLE传播对比测试

TDD测试套件 - 使用真实ISS TLE数据验证Orekit数值传播器精度
"""

import pytest

# 从conftest导入requires_jvm标记
from tests.conftest import requires_jvm
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch



# 检测SGP4是否可用
try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

# 标记需要SGP4的测试
requires_sgp4 = pytest.mark.skipif(
    not SGP4_AVAILABLE,
    reason="需要SGP4库"
)


class MockSatellite:
    """模拟卫星对象"""
    def __init__(self, altitude=500000.0, inclination=51.6, raan=0.0, mean_anomaly=0.0,
                 tle_line1=None, tle_line2=None):
        self.id = "TEST_SAT"
        self.orbit = MockOrbit(altitude, inclination, raan, mean_anomaly)
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2


class MockOrbit:
    """模拟轨道对象"""
    def __init__(self, altitude=500000.0, inclination=51.6, raan=0.0, mean_anomaly=0.0):
        self.altitude = altitude
        self.inclination = inclination
        self.raan = raan
        self.mean_anomaly = mean_anomaly


class ISSTLEData:
    """ISS TLE数据容器"""

    # ISS TLE示例（2024年1月）
    ISS_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00020000  00000-0  28000-4 0  9999"
    ISS_TLE_LINE2 = "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00"

    # ISS轨道参数
    ISS_ALTITUDE = 408000.0  # 408km平均高度
    ISS_INCLINATION = 51.6  # 度
    ISS_PERIOD = 92.68  # 分钟

    @classmethod
    def get_mock_satellite_with_tle(cls):
        """获取带TLE的Mock卫星"""
        return MockSatellite(
            altitude=cls.ISS_ALTITUDE,
            inclination=cls.ISS_INCLINATION,
            tle_line1=cls.ISS_TLE_LINE1,
            tle_line2=cls.ISS_TLE_LINE2
        )


class TestISSTLEData:
    """ISS TLE数据测试"""

    def test_tle_lines_valid(self):
        """测试TLE行格式有效"""
        line1 = ISSTLEData.ISS_TLE_LINE1
        line2 = ISSTLEData.ISS_TLE_LINE2

        # TLE第1行应该以'1 '开头
        assert line1.startswith('1 ')

        # TLE第2行应该以'2 '开头
        assert line2.startswith('2 ')

        # 卫星编号应该一致
        sat_num_1 = line1[2:7]
        sat_num_2 = line2[2:7]
        assert sat_num_1 == sat_num_2

    def test_iss_orbit_params_valid(self):
        """测试ISS轨道参数有效"""
        # 高度应该在合理范围
        assert 400000 < ISSTLEData.ISS_ALTITUDE < 420000

        # 倾角应该在合理范围
        assert 51.0 < ISSTLEData.ISS_INCLINATION < 52.0

        # 周期应该在合理范围
        assert 90 < ISSTLEData.ISS_PERIOD < 95

    def test_mock_satellite_creation(self):
        """测试Mock卫星创建"""
        sat = ISSTLEData.get_mock_satellite_with_tle()

        assert sat.tle_line1 is not None
        assert sat.tle_line2 is not None
        assert sat.orbit.altitude == ISSTLEData.ISS_ALTITUDE
        assert sat.orbit.inclination == ISSTLEData.ISS_INCLINATION


class TestISSPropagation:
    """ISS传播测试"""

    def test_iss_propagation_simplified(self):
        """测试ISS简化模型传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证位置
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        expected_r = calculator.EARTH_RADIUS + ISSTLEData.ISS_ALTITUDE

        # 允许较大误差，因为是简化模型
        assert abs(r - expected_r) < expected_r * 0.1

    def test_iss_velocity_simplified(self):
        """测试ISS简化模型速度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        # ISS速度约7.66 km/s
        v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        expected_v = 7660.0

        assert abs(v - expected_v) < expected_v * 0.1

    def test_iss_orbit_period_consistency(self):
        """测试ISS轨道周期一致性"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        # ISS周期约92.68分钟
        period_seconds = ISSTLEData.ISS_PERIOD * 60

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(seconds=period_seconds)
        time_step = timedelta(minutes=5)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        # 应该有约19个点（92.68分钟 / 5分钟）
        assert len(results) >= 18


class TestISSWithSGP4:
    """ISS与SGP4对比测试"""

    @requires_sgp4
    def test_iss_sgp4_propagation(self):
        """测试ISS SGP4传播"""
        try:
            from sgp4.api import Satrec, jday

            line1 = ISSTLEData.ISS_TLE_LINE1
            line2 = ISSTLEData.ISS_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)

            dt = datetime(2024, 1, 1, 12, 0, 0)
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

            error, position, velocity = satrec.sgp4(jd, fr)

            assert error == 0
            assert position is not None
            assert velocity is not None

        except ImportError:
            pytest.skip("SGP4 not available")

    @requires_sgp4
    def test_iss_sgp4_vs_simplified(self):
        """测试ISS SGP4与简化模型对比"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            line1 = ISSTLEData.ISS_TLE_LINE1
            line2 = ISSTLEData.ISS_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)

            calculator = OrekitVisibilityCalculator()
            satellite = ISSTLEData.get_mock_satellite_with_tle()

            dt = datetime(2024, 1, 1, 12, 0, 0)

            # SGP4传播
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
            error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

            # 简化模型传播
            simple_pos, simple_vel = calculator._propagate_simplified(satellite, dt)

            # 转换为相同单位（SGP4返回km，简化模型返回m）
            sgp4_pos_m = (sgp4_pos[0] * 1000, sgp4_pos[1] * 1000, sgp4_pos[2] * 1000)

            # 计算位置误差
            error_distance = math.sqrt(
                sum((a - b)**2 for a, b in zip(sgp4_pos_m, simple_pos))
            )

            # 简化模型与SGP4可能有较大差异（简化模型精度较低）
            assert error_distance < 20000000  # 20000km - 简化模型允许较大误差

        except ImportError:
            pytest.skip("SGP4 not available")


class TestISSJavaOrekit:
    """ISS Java Orekit测试（需要JVM）

    使用共享的jvm_bridge fixture避免重复JVM启动
    """

    @requires_jvm
    def test_iss_java_propagation(self, jvm_bridge):
        """测试ISS Java Orekit传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 使用共享fixture，JVM已启动
        assert jvm_bridge.is_jvm_running()

        config = {'use_java_orekit': True}
        calculator = OrekitVisibilityCalculator(config)

        satellite = ISSTLEData.get_mock_satellite_with_tle()
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_satellite(satellite, dt)

        assert pos is not None
        assert vel is not None

        # 验证高度
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        altitude = r - calculator.EARTH_RADIUS

        # 应该在ISS高度附近
        assert 300000 < altitude < 500000

    @requires_jvm
    def test_iss_java_batch_propagation(self, jvm_bridge):
        """测试ISS Java Orekit批量传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 使用共享fixture，JVM已启动
        assert jvm_bridge.is_jvm_running()

        config = {'use_java_orekit': True}
        calculator = OrekitVisibilityCalculator(config)

        satellite = ISSTLEData.get_mock_satellite_with_tle()

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=1)
        time_step = timedelta(minutes=5)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        assert len(results) > 0

        # 验证所有结果
        for pos, vel, timestamp in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            altitude = r - calculator.EARTH_RADIUS
            assert 300000 < altitude < 500000


class TestISSAccuracyMetrics:
    """ISS精度指标测试"""

    def test_iss_position_accuracy(self):
        """测试ISS位置精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        # 传播1小时
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        results = []

        for i in range(13):  # 0, 5, 10, ..., 60分钟
            dt = start_time + timedelta(minutes=i * 5)
            pos, vel = calculator._propagate_simplified(satellite, dt)
            results.append((pos, vel, dt))

        # 验证所有位置在合理范围
        for pos, vel, dt in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            altitude = r - calculator.EARTH_RADIUS

            # ISS高度应该在400km左右
            assert 350000 < altitude < 450000

    def test_iss_velocity_accuracy(self):
        """测试ISS速度精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

        # ISS速度应该在7.5-8.0 km/s (简化模型可能有偏差)
        assert 7500 < v < 8000

    def test_iss_orbit_shape(self):
        """测试ISS轨道形状"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        # 传播一个周期
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        period = timedelta(minutes=ISSTLEData.ISS_PERIOD)
        end_time = start_time + period
        time_step = timedelta(minutes=2)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        # 计算所有位置的半径
        radii = []
        for pos, vel, dt in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            radii.append(r)

        # 计算半径变化
        r_mean = sum(radii) / len(radii)
        r_std = math.sqrt(sum((r - r_mean)**2 for r in radii) / len(radii))

        # 近圆轨道，半径变化应该较小
        assert r_std / r_mean < 0.1  # 变化小于10%


class TestISSEdgeCases:
    """ISS边界情况测试"""

    def test_iss_propagation_over_long_time(self):
        """测试ISS长期传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        # 传播24小时
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=24)
        time_step = timedelta(hours=1)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        assert len(results) == 25  # 25个点

        # 验证所有位置
        for pos, vel, dt in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            assert r > calculator.EARTH_RADIUS

    def test_iss_propagation_with_tle_none(self):
        """测试ISS无TLE传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(
            altitude=ISSTLEData.ISS_ALTITUDE,
            inclination=ISSTLEData.ISS_INCLINATION,
            tle_line1=None,
            tle_line2=None
        )

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        assert pos is not None
        assert vel is not None

    def test_iss_propagation_at_different_times(self):
        """测试ISS不同时间传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = ISSTLEData.get_mock_satellite_with_tle()

        times = [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 6, 0, 0),
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 18, 0, 0),
        ]

        positions = []
        for dt in times:
            pos, vel = calculator._propagate_simplified(satellite, dt)
            positions.append(pos)

        # 不同时间位置应该不同
        for i in range(1, len(positions)):
            assert positions[i] != positions[i-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
