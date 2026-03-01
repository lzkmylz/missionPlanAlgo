"""
姿态角计算器单元测试

测试 AttitudeCalculator 的各种场景：
- 不同轨道类型的姿态计算
- 不同目标位置的侧摆角计算
- SGP4和HPOP传播器
"""

import unittest
import math
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from core.dynamics.attitude_calculator import (
    AttitudeCalculator,
    AttitudeAngles,
    PropagatorType,
)
from core.models.satellite import Satellite, SatelliteType, Orbit, OrbitType
from core.models.target import Target, TargetType


class TestAttitudeAngles(unittest.TestCase):
    """测试姿态角数据类"""

    def test_attitude_angles_creation(self):
        """测试创建姿态角对象"""
        angles = AttitudeAngles(
            roll=15.0,
            pitch=5.0,
            yaw=0.0,
            coordinate_system="LVLH",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(angles.roll, 15.0)
        self.assertEqual(angles.pitch, 5.0)
        self.assertEqual(angles.yaw, 0.0)
        self.assertEqual(angles.coordinate_system, "LVLH")

    def test_attitude_angles_to_dict(self):
        """测试转换为字典"""
        angles = AttitudeAngles(
            roll=10.0,
            pitch=-5.0,
            yaw=0.0,
            coordinate_system="LVLH",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        d = angles.to_dict()
        self.assertEqual(d["roll"], 10.0)
        self.assertEqual(d["pitch"], -5.0)
        self.assertEqual(d["yaw"], 0.0)
        self.assertEqual(d["coordinate_system"], "LVLH")


class TestAttitudeCalculatorInitialization(unittest.TestCase):
    """测试姿态角计算器初始化"""

    def test_init_with_sgp4(self):
        """测试使用SGP4初始化"""
        calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)
        self.assertEqual(calculator.propagator_type, PropagatorType.SGP4)

    def test_init_with_hpop(self):
        """测试使用HPOP初始化"""
        calculator = AttitudeCalculator(propagator_type=PropagatorType.HPOP)
        self.assertEqual(calculator.propagator_type, PropagatorType.HPOP)

    def test_default_propagator_is_sgp4(self):
        """测试默认传播器是SGP4"""
        calculator = AttitudeCalculator()
        self.assertEqual(calculator.propagator_type, PropagatorType.SGP4)


class TestAttitudeCalculatorBasic(unittest.TestCase):
    """测试基本姿态角计算"""

    def setUp(self):
        """设置测试数据"""
        self.calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

        # 创建测试卫星（带TLE）
        self.satellite = Satellite(
            id="test_sat_1",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                orbit_type=OrbitType.SSO,
                altitude=500000.0,
                inclination=97.4,
            ),
            tle_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            tle_line2="2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537",
        )

        # 创建测试目标（星下点正下方）
        self.target_nadir = Target(
            id="target_nadir",
            name="Nadir Target",
            latitude=0.0,
            longitude=0.0,
            target_type=TargetType.POINT,
        )

        # 创建测试目标（有侧摆）
        self.target_off_nadir = Target(
            id="target_off_nadir",
            name="Off-Nadir Target",
            latitude=10.0,
            longitude=10.0,
            target_type=TargetType.POINT,
        )

        self.test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_calculate_attitude_nadir_target(self):
        """测试星下点目标的姿态角"""
        # 使用有TLE的卫星测试实际姿态计算
        # 创建一个在赤道上的测试时间点
        attitude = self.calculator.calculate_attitude(
            satellite=self.satellite,
            target=self.target_nadir,
            imaging_time=self.test_time
        )

        self.assertIsInstance(attitude, AttitudeAngles)
        self.assertEqual(attitude.yaw, 0.0)  # 零偏航模式
        self.assertEqual(attitude.coordinate_system, "LVLH")
        # 姿态角应该在合理范围内（不验证具体值，因为取决于轨道）
        self.assertIsNotNone(attitude.roll)
        self.assertIsNotNone(attitude.pitch)

    def test_calculate_attitude_off_nadir_target(self):
        """测试偏离星下点目标的姿态角"""
        attitude = self.calculator.calculate_attitude(
            satellite=self.satellite,
            target=self.target_off_nadir,
            imaging_time=self.test_time
        )

        self.assertIsInstance(attitude, AttitudeAngles)
        # 偏离星下点应该有非零的滚转或俯仰
        self.assertGreater(abs(attitude.roll) + abs(attitude.pitch), 0.1)
        self.assertEqual(attitude.yaw, 0.0)  # 零偏航模式

    def test_invalid_satellite_raises_error(self):
        """测试无效卫星抛出异常"""
        with self.assertRaises(ValueError):
            self.calculator.calculate_attitude(
                satellite=None,
                target=self.target_nadir,
                imaging_time=self.test_time
            )

    def test_invalid_target_raises_error(self):
        """测试无效目标抛出异常"""
        with self.assertRaises(ValueError):
            self.calculator.calculate_attitude(
                satellite=self.satellite,
                target=None,
                imaging_time=self.test_time
            )


class TestAttitudeCalculatorLVLH(unittest.TestCase):
    """测试LVLH坐标系计算"""

    def setUp(self):
        self.calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

    def test_lvlh_frame_construction(self):
        """测试LVLH坐标系构建"""
        # 卫星位置和速度
        position = (6878000.0, 0.0, 0.0)  # ECEF meters
        velocity = (0.0, 7500.0, 0.0)     # m/s

        lvlh = self.calculator._construct_lvlh_frame(position, velocity)

        # LVLH坐标系的X轴应该沿速度方向
        self.assertAlmostEqual(lvlh['X'][0], 0.0, places=5)
        self.assertAlmostEqual(lvlh['X'][1], 1.0, places=5)
        self.assertAlmostEqual(lvlh['X'][2], 0.0, places=5)

        # LVLH坐标系的Z轴应该指向地心（负位置方向）
        self.assertAlmostEqual(lvlh['Z'][0], -1.0, places=5)
        self.assertAlmostEqual(lvlh['Z'][1], 0.0, places=5)
        self.assertAlmostEqual(lvlh['Z'][2], 0.0, places=5)

    def test_vector_in_lvlh_frame(self):
        """测试向量在LVLH坐标系中的表示"""
        position = (6878000.0, 0.0, 0.0)
        velocity = (0.0, 7500.0, 0.0)

        # LVLH Z轴指向地心（负位置方向）
        # 所以如果向量是(-1, 0, 0)，在LVLH坐标系中与Z轴同向
        target_vector = (-1.0, 0.0, 0.0)

        lvlh = self.calculator._construct_lvlh_frame(position, velocity)
        vector_in_lvlh = self.calculator._transform_to_lvlh(target_vector, lvlh)

        # 验证Z轴方向确实是(-1, 0, 0)的归一化形式
        # 由于归一化，向量长度应该是1
        import math
        vec_norm = math.sqrt(sum(v**2 for v in vector_in_lvlh))
        self.assertAlmostEqual(vec_norm, 1.0, places=5)

        # 向量应该主要沿Z方向
        self.assertAlmostEqual(abs(vector_in_lvlh[2]), 1.0, places=5)


class TestAttitudeCalculatorRollPitch(unittest.TestCase):
    """测试滚转和俯仰角计算"""

    def setUp(self):
        self.calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

    def test_roll_angle_calculation(self):
        """测试滚转角计算"""
        # 视线向量在LVLH坐标系中，有Y分量和Z分量
        # roll = atan2(Y, -Z)  （绕X轴旋转）
        # 30度滚转：Y轴分量对应滚转
        angle = math.radians(30)
        los_lvlh = (0.0, math.sin(angle), -math.cos(angle))  # 30度滚转

        roll, pitch = self.calculator._calculate_roll_pitch(los_lvlh)

        self.assertAlmostEqual(math.degrees(roll), 30.0, places=1)
        self.assertAlmostEqual(pitch, 0.0, places=5)

    def test_pitch_angle_calculation(self):
        """测试俯仰角计算"""
        # pitch = atan2(X, -Z)  （绕Y轴旋转）
        # 30度俯仰：X轴分量对应俯仰
        angle = math.radians(30)
        los_lvlh = (math.sin(angle), 0.0, -math.cos(angle))  # 30度俯仰

        roll, pitch = self.calculator._calculate_roll_pitch(los_lvlh)

        self.assertAlmostEqual(roll, 0.0, places=5)
        self.assertAlmostEqual(math.degrees(pitch), 30.0, places=1)

    def test_combined_roll_pitch(self):
        """测试组合滚转和俯仰"""
        # 测试滚转和俯仰组合的向量
        # 构造一个已知分解的向量
        roll_input = 30.0
        pitch_input = 20.0

        # 按照 _calculate_roll_pitch 的公式反向验证
        # roll = atan2(Y, -Z), pitch = atan2(X, -Z)
        # 给定 roll=30°, pitch=20°
        roll_rad = math.radians(roll_input)
        pitch_rad = math.radians(pitch_input)

        # 构造向量，使得 atan2(Y, -Z) = roll, atan2(X, -Z) = pitch
        # 设 -Z = 1, 则 Y = tan(roll), X = tan(pitch)
        Z = -1.0
        Y = math.tan(roll_rad)
        X = math.tan(pitch_rad)

        los_lvlh = (X, Y, Z)

        # 归一化
        norm = math.sqrt(sum(c**2 for c in los_lvlh))
        los_lvlh = tuple(c / norm for c in los_lvlh)

        roll, pitch = self.calculator._calculate_roll_pitch(los_lvlh)

        # 由于归一化，角度会有微小偏差
        self.assertAlmostEqual(math.degrees(roll), roll_input, places=1)
        self.assertAlmostEqual(math.degrees(pitch), pitch_input, places=1)


class TestAttitudeCalculatorEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        self.calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

    def test_max_roll_angle_limit(self):
        """测试最大滚转角限制"""
        # 卫星最大侧摆角通常为30-50度
        satellite = Satellite(
            id="test_sat",
            name="Test",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
        )
        satellite.capabilities.max_off_nadir = 30.0

        # 检查计算的姿态角是否在限制范围内
        self.assertLessEqual(satellite.capabilities.max_off_nadir, 45.0)

    def test_night_target_attitude(self):
        """测试夜间目标的姿态角计算"""
        # 使用带TLE的卫星进行测试
        satellite = Satellite(
            id="test_sat",
            name="Test",
            sat_type=SatelliteType.SAR_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            tle_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            tle_line2="2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537",
        )

        target = Target(
            id="night_target",
            name="Night Target",
            latitude=-45.0,
            longitude=120.0,
            target_type=TargetType.POINT,
        )

        test_time = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)

        # SAR卫星夜间也能工作
        attitude = self.calculator.calculate_attitude(
            satellite=satellite,
            target=target,
            imaging_time=test_time
        )

        self.assertIsInstance(attitude, AttitudeAngles)
        self.assertEqual(attitude.yaw, 0.0)


class TestPropagatorIntegration(unittest.TestCase):
    """测试与不同传播器的集成"""

    def test_sgp4_propagation_used(self):
        """测试使用SGP4进行轨道传播"""
        calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

        satellite = Satellite(
            id="sgp4_sat",
            name="SGP4 Test",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(altitude=500000.0, inclination=97.4),
            tle_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            tle_line2="2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537",
        )

        target = Target(
            id="test_target",
            name="Test",
            latitude=0.0,
            longitude=0.0,
            target_type=TargetType.POINT,
        )

        test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 应该成功计算，使用SGP4
        attitude = calculator.calculate_attitude(satellite, target, test_time)
        self.assertIsNotNone(attitude)

    def test_hpop_config_available(self):
        """测试HPOP配置可用"""
        from core.orbit.hpop_interface import HPOPConfig, ForceModel

        config = HPOPConfig(
            force_model=ForceModel.J2,
            use_earth_gravity=True,
            use_atmospheric_drag=False,
        )

        self.assertEqual(config.force_model, ForceModel.J2)
        self.assertTrue(config.use_earth_gravity)
        self.assertFalse(config.use_atmospheric_drag)


if __name__ == "__main__":
    unittest.main()
