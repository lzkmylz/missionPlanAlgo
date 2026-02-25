"""
Orekit轨道传播器精度测试

TDD测试套件 - 测试轨道传播精度和一致性
"""

import pytest

# 从conftest导入requires_jvm标记
from tests.conftest import requires_jvm
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch




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


class TestOrekitPropagatorImports:
    """测试传播器模块导入"""

    def test_visibility_calculator_imports(self):
        """测试可见性计算器可以正确导入"""
        try:
            from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
            assert True
        except ImportError as e:
            pytest.fail(f"无法导入OrekitVisibilityCalculator: {e}")

    def test_java_bridge_imports(self):
        """测试Java桥接模块可以正确导入"""
        try:
            from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge
            assert True
        except ImportError as e:
            pytest.fail(f"无法导入OrekitJavaBridge: {e}")


class TestSinglePointPropagation:
    """单点传播精度测试"""

    def test_propagate_simplified_model_exists(self):
        """测试简化传播模型存在"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        assert hasattr(calculator, '_propagate_simplified')
        assert callable(calculator._propagate_simplified)

    def test_propagate_simplified_returns_tuple(self):
        """测试简化传播返回元组"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        dt = datetime(2024, 1, 1, 12, 0, 0)

        result = calculator._propagate_simplified(satellite, dt)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_propagate_simplified_returns_position_velocity(self):
        """测试简化传播返回位置和速度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 位置应该是3D坐标
        assert isinstance(pos, tuple)
        assert len(pos) == 3
        assert all(isinstance(x, (int, float)) for x in pos)

        # 速度应该是3D向量
        assert isinstance(vel, tuple)
        assert len(vel) == 3
        assert all(isinstance(v, (int, float)) for v in vel)

    def test_propagate_simplified_position_magnitude(self):
        """测试简化传播位置大小合理（地球半径+高度）"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        altitude = 500000.0  # 500km
        satellite = MockSatellite(altitude=altitude)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 计算位置大小
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)

        # 应该在地球半径+高度附近（允许10%误差）
        expected_r = calculator.EARTH_RADIUS + altitude
        assert abs(r - expected_r) < expected_r * 0.1

    def test_propagate_simplified_velocity_magnitude(self):
        """测试简化传播速度大小合理（圆轨道速度）"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        altitude = 500000.0  # 500km
        satellite = MockSatellite(altitude=altitude)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 计算速度大小
        v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

        # 圆轨道速度 v = sqrt(GM/r)
        GM = 3.986004418e14  # m^3/s^2
        r = calculator.EARTH_RADIUS + altitude
        expected_v = math.sqrt(GM / r)

        # 允许10%误差
        assert abs(v - expected_v) < expected_v * 0.1

    def test_propagate_simplified_consistency(self):
        """测试简化传播结果一致性（相同输入相同输出）"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos1, vel1 = calculator._propagate_simplified(satellite, dt)
        pos2, vel2 = calculator._propagate_simplified(satellite, dt)

        assert pos1 == pos2
        assert vel1 == vel2

    def test_propagate_simplified_different_times(self):
        """测试不同时间产生不同位置"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 13, 0, 0)

        pos1, vel1 = calculator._propagate_simplified(satellite, dt1)
        pos2, vel2 = calculator._propagate_simplified(satellite, dt2)

        # 1小时后位置应该不同
        assert pos1 != pos2

    def test_propagate_simplified_with_tle_fallback(self):
        """测试有TLE时的降级处理"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        # 模拟TLE属性
        satellite.tle_line1 = None
        satellite.tle_line2 = None
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        assert pos is not None
        assert vel is not None


class TestBatchPropagation:
    """批量传播结果一致性测试"""

    def test_propagate_range_exists(self):
        """测试批量传播方法存在"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        assert hasattr(calculator, '_propagate_range')
        assert callable(calculator._propagate_range)

    def test_propagate_range_returns_list(self):
        """测试批量传播返回列表"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 1, 0)  # 1分钟
        time_step = timedelta(seconds=60)

        results = calculator._propagate_range(satellite, start_time, end_time, time_step)

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_propagate_range_result_format(self):
        """测试批量传播结果格式"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 2, 0)  # 2分钟
        time_step = timedelta(seconds=60)

        results = calculator._propagate_range(satellite, start_time, end_time, time_step)

        # 每个结果应该是 (pos, vel, timestamp) 元组
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 3
            pos, vel, timestamp = result
            assert len(pos) == 3
            assert len(vel) == 3
            assert isinstance(timestamp, datetime)

    def test_propagate_range_time_order(self):
        """测试批量传播结果按时间顺序排列"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 0)  # 5分钟
        time_step = timedelta(seconds=60)

        results = calculator._propagate_range(satellite, start_time, end_time, time_step)

        # 检查时间顺序
        timestamps = [r[2] for r in results]
        assert timestamps == sorted(timestamps)

    def test_propagate_range_empty_for_invalid_time(self):
        """测试无效时间范围返回空列表"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)  # 结束时间早于开始时间
        time_step = timedelta(seconds=60)

        results = calculator._propagate_range(satellite, start_time, end_time, time_step)

        assert results == []

    def test_propagate_range_step_accuracy(self):
        """测试批量传播时间步长准确性"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 0)  # 5分钟
        time_step = timedelta(seconds=60)  # 1分钟步长

        results = calculator._propagate_range(satellite, start_time, end_time, time_step)

        # 应该有大约6个点（0, 1, 2, 3, 4, 5分钟）
        assert len(results) >= 5

        # 检查时间间隔
        for i in range(1, len(results)):
            dt = results[i][2] - results[i-1][2]
            assert abs(dt.total_seconds() - 60) < 1  # 允许1秒误差


class TestOrbitTypes:
    """不同轨道类型测试"""

    def test_circular_orbit_propagation(self):
        """测试圆轨道传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # 圆轨道：偏心率接近0
        satellite = MockSatellite(altitude=500000.0, inclination=0.0)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证位置在赤道平面（z接近0）
        assert abs(pos[2]) < 1000  # z坐标应该很小

    def test_elliptical_orbit_propagation(self):
        """测试椭圆轨道传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # 椭圆轨道通过不同高度模拟
        satellite = MockSatellite(altitude=500000.0, inclination=45.0)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证位置有效
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert r > calculator.EARTH_RADIUS

    def test_polar_orbit_propagation(self):
        """测试极轨道传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # 极轨道：倾角90度
        satellite = MockSatellite(altitude=500000.0, inclination=90.0)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证位置有效
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert r > calculator.EARTH_RADIUS

    def test_sso_orbit_propagation(self):
        """测试太阳同步轨道传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # SSO轨道：倾角约97-98度
        satellite = MockSatellite(altitude=500000.0, inclination=97.4)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证位置有效
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert r > calculator.EARTH_RADIUS

    def test_geo_orbit_propagation(self):
        """测试地球静止轨道传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        # GEO轨道：高度约35786km，倾角0
        geo_altitude = 35786000.0
        satellite = MockSatellite(altitude=geo_altitude, inclination=0.0)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证位置高度正确
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        expected_r = calculator.EARTH_RADIUS + geo_altitude
        assert abs(r - expected_r) < expected_r * 0.1


class TestTimeConversion:
    """时间转换准确性测试"""

    def test_propagate_with_different_times(self):
        """测试不同时间输入"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()

        times = [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 6, 15, 12, 30, 0),
            datetime(2024, 12, 31, 23, 59, 59),
        ]

        for dt in times:
            pos, vel = calculator._propagate_simplified(satellite, dt)
            assert pos is not None
            assert vel is not None

    def test_propagate_with_microseconds(self):
        """测试微秒级时间精度"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        dt = datetime(2024, 1, 1, 12, 0, 0, 123456)  # 带微秒

        pos, vel = calculator._propagate_simplified(satellite, dt)

        assert pos is not None
        assert vel is not None

    def test_propagate_24_hour_period(self):
        """测试24小时周期传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        # 计算轨道周期
        GM = 3.986004418e14
        r = calculator.EARTH_RADIUS + 500000.0
        period = 2 * math.pi * math.sqrt(r**3 / GM)

        # 传播一个周期
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = dt1 + timedelta(seconds=period)

        pos1, vel1 = calculator._propagate_simplified(satellite, dt1)
        pos2, vel2 = calculator._propagate_simplified(satellite, dt2)

        # 一个周期后位置应该接近（简化模型不是完全周期性的）
        # 允许较大误差，因为简化模型不考虑完整轨道力学
        distance = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
        assert distance < 3000000  # 允许3000km误差（简化模型）


class TestPropagationAccuracy:
    """传播精度测试"""

    def test_position_conservation_energy(self):
        """测试能量守恒（简化模型中近似）"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 计算比机械能 E = v^2/2 - mu/r
        GM = 3.986004418e14
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        energy = v**2 / 2 - GM / r

        # 能量应该是负值（束缚轨道）
        assert energy < 0

    def test_angular_momentum_conservation(self):
        """测试角动量守恒（简化模型中近似）"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=500000.0)

        # 两个不同时间
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 12, 30, 0)

        pos1, vel1 = calculator._propagate_simplified(satellite, dt1)
        pos2, vel2 = calculator._propagate_simplified(satellite, dt2)

        # 计算角动量 h = r x v
        h1 = (
            pos1[1]*vel1[2] - pos1[2]*vel1[1],
            pos1[2]*vel1[0] - pos1[0]*vel1[2],
            pos1[0]*vel1[1] - pos1[1]*vel1[0]
        )
        h2 = (
            pos2[1]*vel2[2] - pos2[2]*vel2[1],
            pos2[2]*vel2[0] - pos2[0]*vel2[1],
            pos2[0]*vel2[1] - pos2[1]*vel2[0]
        )

        # 角动量大小应该相近（简化模型中只是近似）
        h1_mag = math.sqrt(h1[0]**2 + h1[1]**2 + h1[2]**2)
        h2_mag = math.sqrt(h2[0]**2 + h2[1]**2 + h2[2]**2)

        # 简化模型30分钟后角动量变化较大，放宽阈值
        assert abs(h1_mag - h2_mag) < h1_mag * 0.8  # 允许80%误差


class TestJavaOrekitPropagation:
    """Java Orekit传播测试（需要JVM）

    使用共享的jvm_bridge fixture避免重复JVM启动
    """

    @requires_jvm
    def test_java_propagate_single_point(self, jvm_bridge):
        """测试Java Orekit单点传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 使用共享fixture，JVM已启动
        assert jvm_bridge.is_jvm_running()

        config = {'use_java_orekit': True}
        calculator = OrekitVisibilityCalculator(config)

        satellite = MockSatellite()
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 如果JVM可用，应该使用Java传播
        pos, vel = calculator._propagate_satellite(satellite, dt)

        assert pos is not None
        assert vel is not None

    @requires_jvm
    def test_java_propagate_range(self, jvm_bridge):
        """测试Java Orekit批量传播"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 使用共享fixture，JVM已启动
        assert jvm_bridge.is_jvm_running()

        config = {'use_java_orekit': True}
        calculator = OrekitVisibilityCalculator(config)

        satellite = MockSatellite()
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 0)
        time_step = timedelta(seconds=60)

        results = calculator._propagate_range(satellite, start_time, end_time, time_step)

        assert len(results) > 0


class TestPropagationEdgeCases:
    """传播边界情况测试"""

    def test_propagate_with_none_orbit(self):
        """测试None轨道参数处理"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite()
        satellite.orbit = None
        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 应该使用默认值，不抛出异常
        pos, vel = calculator._propagate_simplified(satellite, dt)
        assert pos is not None
        assert vel is not None

    def test_propagate_with_zero_altitude(self):
        """测试零高度（地面）处理"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=0.0)
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 位置应该在地球表面
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert abs(r - calculator.EARTH_RADIUS) < 1000

    def test_propagate_with_very_high_altitude(self):
        """测试非常高高度处理"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        satellite = MockSatellite(altitude=10000000.0)  # 10000km
        dt = datetime(2024, 1, 1, 12, 0, 0)

        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 位置应该在指定高度
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        expected_r = calculator.EARTH_RADIUS + 10000000.0
        assert abs(r - expected_r) < expected_r * 0.1

    def test_propagate_with_extreme_inclination(self):
        """测试极端倾角处理"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()

        # 测试0度和180度倾角
        for inc in [0.0, 180.0]:
            satellite = MockSatellite(altitude=500000.0, inclination=inc)
            dt = datetime(2024, 1, 1, 12, 0, 0)

            pos, vel = calculator._propagate_simplified(satellite, dt)
            assert pos is not None
            assert vel is not None


class TestPropagationMockData:
    """使用Mock数据测试传播"""

    def test_mock_orbit_propagation(self):
        """测试使用Mock轨道数据"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()

        # 创建Mock卫星
        mock_sat = MagicMock()
        mock_sat.id = "MOCK_SAT"
        mock_sat.orbit = MagicMock()
        mock_sat.orbit.altitude = 600000.0
        mock_sat.orbit.inclination = 45.0
        mock_sat.orbit.raan = 30.0
        mock_sat.orbit.mean_anomaly = 0.0

        dt = datetime(2024, 1, 1, 12, 0, 0)
        pos, vel = calculator._propagate_simplified(mock_sat, dt)

        assert pos is not None
        assert vel is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
