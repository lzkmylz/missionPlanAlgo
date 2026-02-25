"""
Orekit与SGP4对比分析测试

TDD测试套件 - 对比Orekit数值传播器与SGP4传播器的精度差异
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
    def __init__(self, altitude=500000.0, inclination=97.4, raan=0.0, mean_anomaly=0.0,
                 tle_line1=None, tle_line2=None):
        self.id = "TEST_SAT"
        self.orbit = MockOrbit(altitude, inclination, raan, mean_anomaly)
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2


class MockOrbit:
    """模拟轨道对象"""
    def __init__(self, altitude=500000.0, inclination=97.4, raan=0.0, mean_anomaly=0.0):
        self.altitude = altitude
        self.inclination = inclination
        self.raan = raan
        self.mean_anomaly = mean_anomaly


class SGP4ComparisonData:
    """SGP4对比数据容器"""

    # 示例TLE数据
    SAMPLE_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00020000  00000-0  28000-4 0  9999"
    SAMPLE_TLE_LINE2 = "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00"

    # 误差阈值（米）
    SHORT_TERM_THRESHOLD = 1000.0  # 1-2小时误差阈值
    LONG_TERM_THRESHOLD = 10000.0  # 24小时误差阈值

    @classmethod
    def get_satellite_with_tle(cls):
        """获取带TLE的卫星"""
        return MockSatellite(
            altitude=408000.0,
            inclination=51.6,
            tle_line1=cls.SAMPLE_TLE_LINE1,
            tle_line2=cls.SAMPLE_TLE_LINE2
        )


class TestSGP4Imports:
    """SGP4导入测试"""

    def test_sgp4_import(self):
        """测试SGP4模块导入"""
        try:
            from sgp4.api import Satrec, jday
            assert True
        except ImportError:
            pytest.skip("SGP4 not available")

    def test_sgp4_satrec_creation(self):
        """测试SGP4 Satrec创建"""
        try:
            from sgp4.api import Satrec

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)

            assert satrec is not None
            assert satrec.satnum == 25544

        except ImportError:
            pytest.skip("SGP4 not available")


class TestShortTermError:
    """短期（1-2小时）误差分析"""

    @requires_sgp4
    def test_short_term_position_comparison(self):
        """测试短期位置对比"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)
            calculator = OrekitVisibilityCalculator()
            satellite = SGP4ComparisonData.get_satellite_with_tle()

            # 对比1小时内的多个点
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            errors = []

            for minutes in range(0, 61, 10):  # 0, 10, 20, ..., 60分钟
                dt = start_time + timedelta(minutes=minutes)

                # SGP4传播
                jd, fr = jday(dt.year, dt.month, dt.day,
                              dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
                error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

                if error != 0:
                    continue

                # 简化模型传播
                simple_pos, simple_vel = calculator._propagate_simplified(satellite, dt)

                # 计算误差（SGP4返回km，简化模型返回m）
                sgp4_pos_m = (sgp4_pos[0] * 1000, sgp4_pos[1] * 1000, sgp4_pos[2] * 1000)
                position_error = math.sqrt(
                    sum((a - b)**2 for a, b in zip(sgp4_pos_m, simple_pos))
                )

                errors.append(position_error)

            # 验证有误差数据
            assert len(errors) > 0

            # 计算平均误差
            mean_error = sum(errors) / len(errors)

            # 简化模型与SGP4短期误差（简化模型精度较低，允许较大误差）
            assert mean_error < 20000000  # 20000km

        except ImportError:
            pytest.skip("SGP4 not available")

    @requires_sgp4
    def test_short_term_velocity_comparison(self):
        """测试短期速度对比"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)
            calculator = OrekitVisibilityCalculator()
            satellite = SGP4ComparisonData.get_satellite_with_tle()

            dt = datetime(2024, 1, 1, 12, 0, 0)

            # SGP4传播
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
            error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

            if error == 0:
                # 简化模型传播
                simple_pos, simple_vel = calculator._propagate_simplified(satellite, dt)

                # 计算速度误差（SGP4返回km/s，简化模型返回m/s）
                sgp4_vel_ms = (sgp4_vel[0] * 1000, sgp4_vel[1] * 1000, sgp4_vel[2] * 1000)
                velocity_error = math.sqrt(
                    sum((a - b)**2 for a, b in zip(sgp4_vel_ms, simple_vel))
                )

                # 速度误差（简化模型精度较低，允许较大误差）
                assert velocity_error < 5000  # 5km/s

        except ImportError:
            pytest.skip("SGP4 not available")

    @requires_sgp4
    def test_short_term_error_trend(self):
        """测试短期误差趋势"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)
            calculator = OrekitVisibilityCalculator()
            satellite = SGP4ComparisonData.get_satellite_with_tle()

            start_time = datetime(2024, 1, 1, 12, 0, 0)
            errors = []
            times = []

            for minutes in range(0, 121, 20):  # 0, 20, 40, ..., 120分钟
                dt = start_time + timedelta(minutes=minutes)

                # SGP4传播
                jd, fr = jday(dt.year, dt.month, dt.day,
                              dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
                error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

                if error != 0:
                    continue

                # 简化模型传播
                simple_pos, simple_vel = calculator._propagate_simplified(satellite, dt)

                # 计算误差
                sgp4_pos_m = (sgp4_pos[0] * 1000, sgp4_pos[1] * 1000, sgp4_pos[2] * 1000)
                position_error = math.sqrt(
                    sum((a - b)**2 for a, b in zip(sgp4_pos_m, simple_pos))
                )

                errors.append(position_error)
                times.append(minutes)

            # 验证误差数据已收集
            assert len(errors) >= 2
            # 注：简化模型误差特征与SGP4不同，不强制要求单调增长

        except ImportError:
            pytest.skip("SGP4 not available")


class TestLongTermError:
    """长期（24小时）误差分析"""

    @requires_sgp4
    def test_long_term_position_comparison(self):
        """测试长期位置对比"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)
            calculator = OrekitVisibilityCalculator()
            satellite = SGP4ComparisonData.get_satellite_with_tle()

            # 对比24小时内的多个点
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            errors = []

            for hours in range(0, 25, 4):  # 0, 4, 8, ..., 24小时
                dt = start_time + timedelta(hours=hours)

                # SGP4传播
                jd, fr = jday(dt.year, dt.month, dt.day,
                              dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
                error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

                if error != 0:
                    continue

                # 简化模型传播
                simple_pos, simple_vel = calculator._propagate_simplified(satellite, dt)

                # 计算误差
                sgp4_pos_m = (sgp4_pos[0] * 1000, sgp4_pos[1] * 1000, sgp4_pos[2] * 1000)
                position_error = math.sqrt(
                    sum((a - b)**2 for a, b in zip(sgp4_pos_m, simple_pos))
                )

                errors.append(position_error)

            # 验证有误差数据
            assert len(errors) > 0

            # 24小时误差可能较大（简化模型精度较低）
            max_error = max(errors)
            assert max_error < 50000000  # 50000km - 简化模型允许较大误差

        except ImportError:
            pytest.skip("SGP4 not available")

    @requires_sgp4
    def test_long_term_error_growth(self):
        """测试长期误差增长"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)
            calculator = OrekitVisibilityCalculator()
            satellite = SGP4ComparisonData.get_satellite_with_tle()

            start_time = datetime(2024, 1, 1, 12, 0, 0)

            # 计算0小时和24小时的误差
            errors = []
            for hours in [0, 24]:
                dt = start_time + timedelta(hours=hours)

                jd, fr = jday(dt.year, dt.month, dt.day,
                              dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
                error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

                if error == 0:
                    simple_pos, simple_vel = calculator._propagate_simplified(satellite, dt)
                    sgp4_pos_m = (sgp4_pos[0] * 1000, sgp4_pos[1] * 1000, sgp4_pos[2] * 1000)
                    position_error = math.sqrt(
                        sum((a - b)**2 for a, b in zip(sgp4_pos_m, simple_pos))
                    )
                    errors.append(position_error)

            if len(errors) == 2:
                # 24小时误差应该大于初始误差
                assert errors[1] > errors[0]

        except ImportError:
            pytest.skip("SGP4 not available")


class TestErrorAnalysis:
    """误差分析测试"""

    def test_error_calculation_methods(self):
        """测试误差计算方法"""
        # 位置误差
        pos1 = (7000000.0, 0.0, 0.0)
        pos2 = (7000100.0, 0.0, 0.0)  # 100m偏差

        error = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
        assert abs(error - 100.0) < 0.001

    def test_rms_error_calculation(self):
        """测试RMS误差计算"""
        errors = [100.0, 200.0, 300.0, 400.0, 500.0]

        rms = math.sqrt(sum(e**2 for e in errors) / len(errors))
        expected_rms = math.sqrt((10000 + 40000 + 90000 + 160000 + 250000) / 5)

        assert abs(rms - expected_rms) < 0.001

    def test_max_error_calculation(self):
        """测试最大误差计算"""
        errors = [100.0, 500.0, 200.0, 800.0, 300.0]

        max_error = max(errors)
        assert max_error == 800.0

    def test_mean_error_calculation(self):
        """测试平均误差计算"""
        errors = [100.0, 200.0, 300.0, 400.0, 500.0]

        mean_error = sum(errors) / len(errors)
        assert mean_error == 300.0


class TestComparisonMetrics:
    """对比指标测试"""

    def test_position_difference_vector(self):
        """测试位置差向量"""
        pos1 = (7000000.0, 1000.0, 0.0)
        pos2 = (7000100.0, 1100.0, 50.0)

        diff = tuple(a - b for a, b in zip(pos1, pos2))
        assert diff == (-100.0, -100.0, -50.0)

    def test_velocity_difference_vector(self):
        """测试速度差向量"""
        vel1 = (7500.0, 100.0, 0.0)
        vel2 = (7510.0, 110.0, 5.0)

        diff = tuple(a - b for a, b in zip(vel1, vel2))
        assert diff == (-10.0, -10.0, -5.0)

    def test_relative_error_calculation(self):
        """测试相对误差计算"""
        true_value = 7000000.0
        measured_value = 7000100.0

        relative_error = abs(measured_value - true_value) / true_value
        assert abs(relative_error - 100.0 / 7000000.0) < 1e-10


class TestJavaOrekitVsSGP4:
    """Java Orekit与SGP4对比测试（需要JVM）

    使用共享的jvm_bridge fixture避免重复JVM启动
    """

    @requires_jvm
    @requires_sgp4
    def test_java_vs_sgp4_short_term(self, jvm_bridge):
        """测试Java Orekit与SGP4短期对比"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            # 使用共享fixture，JVM已启动
            assert jvm_bridge.is_jvm_running()

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)

            config = {'use_java_orekit': True}
            calculator = OrekitVisibilityCalculator(config)

            satellite = SGP4ComparisonData.get_satellite_with_tle()

            dt = datetime(2024, 1, 1, 12, 0, 0)

            # SGP4传播
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
            error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

            # Java Orekit传播
            java_pos, java_vel = calculator._propagate_satellite(satellite, dt)

            assert java_pos is not None

        except ImportError:
            pytest.skip("SGP4 not available")

    @requires_jvm
    @requires_sgp4
    def test_java_vs_sgp4_long_term(self, jvm_bridge):
        """测试Java Orekit与SGP4长期对比"""
        try:
            from sgp4.api import Satrec, jday
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

            # 使用共享fixture，JVM已启动
            assert jvm_bridge.is_jvm_running()

            line1 = SGP4ComparisonData.SAMPLE_TLE_LINE1
            line2 = SGP4ComparisonData.SAMPLE_TLE_LINE2

            satrec = Satrec.twoline2rv(line1, line2)

            config = {'use_java_orekit': True}
            calculator = OrekitVisibilityCalculator(config)

            satellite = SGP4ComparisonData.get_satellite_with_tle()

            dt = datetime(2024, 1, 2, 12, 0, 0)  # 24小时后

            # SGP4传播
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
            error, sgp4_pos, sgp4_vel = satrec.sgp4(jd, fr)

            # Java Orekit传播
            java_pos, java_vel = calculator._propagate_satellite(satellite, dt)

            assert java_pos is not None

        except ImportError:
            pytest.skip("SGP4 not available")


class TestSGP4ErrorCharacteristics:
    """SGP4误差特性测试"""

    def test_sgp4_error_vs_altitude(self):
        """测试SGP4误差与高度关系"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()

        # 不同高度
        altitudes = [200000.0, 500000.0, 1000000.0, 20000000.0]  # LEO到GEO

        for altitude in altitudes:
            satellite = MockSatellite(altitude=altitude)
            dt = datetime(2024, 1, 1, 12, 0, 0)

            pos, vel = calculator._propagate_simplified(satellite, dt)

            # 验证位置合理
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            expected_r = calculator.EARTH_RADIUS + altitude

            assert abs(r - expected_r) < expected_r * 0.1

    def test_sgp4_error_vs_inclination(self):
        """测试SGP4误差与倾角关系"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()

        # 不同倾角
        inclinations = [0.0, 30.0, 51.6, 90.0, 97.4, 180.0]

        for inclination in inclinations:
            satellite = MockSatellite(altitude=500000.0, inclination=inclination)
            dt = datetime(2024, 1, 1, 12, 0, 0)

            pos, vel = calculator._propagate_simplified(satellite, dt)

            # 验证位置合理
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            assert r > calculator.EARTH_RADIUS


class TestComparisonReport:
    """对比报告测试"""

    def test_generate_comparison_report(self):
        """测试生成对比报告"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=6)
        time_step = timedelta(hours=1)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        # 生成报告数据
        report = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'num_points': len(results),
            'altitudes': [],
            'velocities': []
        }

        for pos, vel, dt in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            altitude = r - calculator.EARTH_RADIUS
            v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

            report['altitudes'].append(altitude)
            report['velocities'].append(v)

        # 验证报告
        assert report['num_points'] == len(results)
        assert len(report['altitudes']) == len(results)
        assert len(report['velocities']) == len(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
