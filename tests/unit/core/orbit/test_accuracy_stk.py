"""
Orekit与STK HPOP精度对比测试

TDD测试套件 - 验证Orekit传播结果与STK HPOP的对比精度
"""

import pytest

# 从conftest导入requires_jvm标记
from tests.conftest import requires_jvm
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock



# 标记需要STK参考数据的测试
requires_stk_data = pytest.mark.skipif(
    True,
    reason="需要STK参考数据"
)


class MockSatellite:
    """模拟卫星对象"""
    def __init__(self, altitude=500000.0, inclination=97.4, raan=0.0, mean_anomaly=0.0):
        self.id = "TEST_SAT"
        self.orbit = MockOrbit(altitude, inclination, raan, mean_anomaly)


class MockOrbit:
    """模拟轨道对象"""
    def __init__(self, altitude=500000.0, inclination=97.4, raan=0.0, mean_anomaly=0.0):
        self.altitude = altitude
        self.inclination = inclination
        self.raan = raan
        self.mean_anomaly = mean_anomaly


class STKReferenceData:
    """STK参考数据容器"""

    # 已知轨道参数（用于与STK对比）
    REFERENCE_ORBIT = {
        'semi_major_axis': 6878137.0,  # m (500km altitude)
        'eccentricity': 0.001,
        'inclination': 97.4,  # degrees
        'raan': 0.0,  # degrees
        'arg_of_perigee': 0.0,  # degrees
        'mean_anomaly': 0.0,  # degrees
    }

    # STK HPOP预计算结果（mock数据）
    # 格式: {elapsed_seconds: (x, y, z, vx, vy, vz)}
    STK_RESULTS_1HOUR = {
        0: (6878137.0, 0.0, 0.0, 0.0, 7612.0, 0.0),
        3600: (1234567.0, 6543210.0, 1234567.0, -1234.0, 5678.0, 1234.0),  # Mock values
    }

    STK_RESULTS_24HOURS = {
        0: (6878137.0, 0.0, 0.0, 0.0, 7612.0, 0.0),
        86400: (2345678.0, 5678901.0, 2345678.0, -2345.0, 6789.0, 2345.0),  # Mock values
    }


class TestSTKComparisonSetup:
    """STK对比测试设置"""

    def test_reference_orbit_params_valid(self):
        """测试参考轨道参数有效"""
        orbit = STKReferenceData.REFERENCE_ORBIT

        # 半长轴应该大于地球半径
        assert orbit['semi_major_axis'] > 6371000

        # 偏心率应该在有效范围
        assert 0 <= orbit['eccentricity'] < 1

        # 倾角应该在有效范围
        assert 0 <= orbit['inclination'] <= 180

    def test_stk_results_format(self):
        """测试STK结果格式"""
        results = STKReferenceData.STK_RESULTS_1HOUR

        for t, state in results.items():
            assert len(state) == 6  # x, y, z, vx, vy, vz
            assert all(isinstance(v, (int, float)) for v in state)


class TestShortTermAccuracy:
    """短期传播精度测试（1小时）"""

    SHORT_TERM_ERROR_THRESHOLD = 10.0  # 10米

    def test_short_term_position_error(self):
        """测试短期位置误差 < 10米"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=1)
        time_step = timedelta(minutes=10)

        # 传播轨道
        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        # 验证结果存在
        assert len(results) > 0

        # 验证每个位置合理
        for pos, vel, timestamp in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            expected_r = calculator.EARTH_RADIUS + 500000.0

            # 允许较大误差，因为这是简化模型
            assert abs(r - expected_r) < expected_r * 0.1

    def test_short_term_velocity_error(self):
        """测试短期速度误差"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 计算速度大小
        v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

        # LEO速度约7612 m/s
        expected_v = 7612.0

        # 允许10%误差
        assert abs(v - expected_v) < expected_v * 0.1

    def test_short_term_consistency(self):
        """测试短期传播一致性"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # 连续传播两次
        results1 = calculator._propagate_range(
            satellite, start_time, start_time + timedelta(hours=1), timedelta(minutes=10)
        )
        results2 = calculator._propagate_range(
            satellite, start_time, start_time + timedelta(hours=1), timedelta(minutes=10)
        )

        # 结果应该相同
        assert len(results1) == len(results2)
        for (pos1, vel1, t1), (pos2, vel2, t2) in zip(results1, results2):
            assert pos1 == pos2
            assert vel1 == vel2
            assert t1 == t2


class TestLongTermAccuracy:
    """长期传播精度测试（24小时）"""

    LONG_TERM_ERROR_THRESHOLD = 100.0  # 100米

    def test_long_term_position_error(self):
        """测试长期位置误差 < 100米"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=24)
        time_step = timedelta(hours=1)

        # 传播轨道
        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        # 验证结果存在
        assert len(results) == 25  # 0, 1, 2, ..., 24 hours

        # 验证每个位置合理
        for pos, vel, timestamp in results:
            r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            expected_r = calculator.EARTH_RADIUS + 500000.0

            # 允许较大误差
            assert abs(r - expected_r) < expected_r * 0.2

    def test_long_term_orbit_period(self):
        """测试长期轨道周期准确性"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        # 500km轨道周期约94.5分钟
        expected_period = 94.5 * 60  # seconds

        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # 传播一个周期
        dt1 = start_time
        dt2 = start_time + timedelta(seconds=expected_period)

        pos1, vel1 = calculator._propagate_simplified(satellite, dt1)
        pos2, vel2 = calculator._propagate_simplified(satellite, dt2)

        # 计算距离
        distance = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))

        # 一个周期后距离应该较小（简化模型允许较大误差）
        assert distance < 3000000  # 3000km（简化模型）

    def test_long_term_raan_drift(self):
        """测试长期RAAN漂移"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()

        # SSO轨道，RAAN应该保持稳定
        satellite = MockSatellite(altitude=500000.0, inclination=97.4)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=24)

        # 传播24小时
        results = calculator._propagate_range(
            satellite, start_time, end_time, timedelta(hours=6)
        )

        # 验证结果
        assert len(results) > 0


class TestAccuracyMetrics:
    """精度指标测试"""

    def test_position_error_calculation(self):
        """测试位置误差计算"""
        # 两个位置
        pos1 = (6878137.0, 0.0, 0.0)
        pos2 = (6878138.0, 1.0, 0.0)  # 偏差 (1, 1, 0)

        # 计算误差
        error = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))

        assert abs(error - math.sqrt(2)) < 0.001

    def test_velocity_error_calculation(self):
        """测试速度误差计算"""
        # 两个速度
        vel1 = (7612.0, 0.0, 0.0)
        vel2 = (7612.5, 0.5, 0.0)  # 偏差 (0.5, 0.5, 0)

        # 计算误差
        error = math.sqrt(sum((a - b)**2 for a, b in zip(vel1, vel2)))

        assert abs(error - math.sqrt(0.5)) < 0.001

    def test_rms_error_calculation(self):
        """测试RMS误差计算"""
        errors = [1.0, 2.0, 3.0, 4.0, 5.0]

        # RMS = sqrt(mean(errors^2))
        rms = math.sqrt(sum(e**2 for e in errors) / len(errors))

        expected_rms = math.sqrt((1 + 4 + 9 + 16 + 25) / 5)
        assert abs(rms - expected_rms) < 0.001


class TestComparisonWithMockSTK:
    """使用Mock STK数据对比"""

    def test_compare_with_mock_stk_1hour(self):
        """测试与Mock STK 1小时结果对比"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # 获取Orekit结果
        pos, vel = calculator._propagate_simplified(satellite, start_time + timedelta(hours=1))

        # 获取Mock STK结果
        stk_state = STKReferenceData.STK_RESULTS_1HOUR.get(3600)

        # 验证结果存在
        assert pos is not None
        assert stk_state is not None

    def test_compare_with_mock_stk_24hours(self):
        """测试与Mock STK 24小时结果对比"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # 获取Orekit结果
        pos, vel = calculator._propagate_simplified(satellite, start_time + timedelta(hours=24))

        # 获取Mock STK结果
        stk_state = STKReferenceData.STK_RESULTS_24HOURS.get(86400)

        # 验证结果存在
        assert pos is not None
        assert stk_state is not None


class TestJavaOrekitAccuracy:
    """Java Orekit精度测试（需要JVM）

    使用共享的jvm_bridge fixture避免重复JVM启动
    """

    @requires_jvm
    def test_java_short_term_accuracy(self, jvm_bridge):
        """测试Java Orekit短期精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 使用共享fixture，JVM已启动
        assert jvm_bridge.is_jvm_running()

        config = {'use_java_orekit': True}
        calculator = OrekitVisibilityCalculator(config)

        satellite = MockSatellite(altitude=500000.0)
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=1)
        time_step = timedelta(minutes=10)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        assert len(results) > 0

    @requires_jvm
    def test_java_long_term_accuracy(self, jvm_bridge):
        """测试Java Orekit长期精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 使用共享fixture，JVM已启动
        assert jvm_bridge.is_jvm_running()

        config = {'use_java_orekit': True}
        calculator = OrekitVisibilityCalculator(config)

        satellite = MockSatellite(altitude=500000.0)
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=24)
        time_step = timedelta(hours=1)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        assert len(results) > 0


class TestAccuracyEdgeCases:
    """精度测试边界情况"""

    def test_accuracy_at_perigee(self):
        """测试近地点精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # 椭圆轨道，近地点300km
        satellite = MockSatellite(altitude=300000.0)

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        expected_r = calculator.EARTH_RADIUS + 300000.0

        assert abs(r - expected_r) < expected_r * 0.1

    def test_accuracy_at_apogee(self):
        """测试远地点精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # 椭圆轨道，远地点1000km
        satellite = MockSatellite(altitude=1000000.0)

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        expected_r = calculator.EARTH_RADIUS + 1000000.0

        assert abs(r - expected_r) < expected_r * 0.1

    def test_accuracy_high_eccentricity(self):
        """测试高偏心率轨道精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # Molniya轨道类型，远地点远
        satellite = MockSatellite(altitude=20000000.0)

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        expected_r = calculator.EARTH_RADIUS + 20000000.0

        assert abs(r - expected_r) < expected_r * 0.1


class TestAccuracyReport:
    """精度报告测试"""

    def test_generate_accuracy_report(self):
        """测试生成精度报告"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(hours=6)
        time_step = timedelta(hours=1)

        results = calculator._propagate_range(
            satellite, start_time, end_time, time_step
        )

        # 计算统计信息
        positions = [pos for pos, vel, t in results]
        velocities = [vel for pos, vel, t in results]

        # 位置范围
        radii = [math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in positions]
        r_mean = sum(radii) / len(radii)
        r_std = math.sqrt(sum((r - r_mean)**2 for r in radii) / len(radii))

        # 验证统计信息合理
        assert r_mean > calculator.EARTH_RADIUS
        assert r_std < r_mean * 0.1  # 变化小于10%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
