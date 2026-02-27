"""
传播器历元时间支持测试

测试简化模型使用epoch和scenario_start_time
"""

import pytest
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.models.satellite import Orbit, Satellite, SatelliteType


class TestPropagationWithEpoch:
    """测试传播器使用epoch时间"""

    def test_propagate_simplified_with_satellite_epoch(self):
        """测试使用卫星自己的epoch进行传播"""
        calculator = OrekitVisibilityCalculator()

        # 创建卫星，设置特定历元
        epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=epoch,
            altitude=500000,
            inclination=97.4,
            mean_anomaly=0.0
        )

        # 传播到历元后1小时
        dt = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 验证返回了位置
        assert len(pos) == 3
        assert len(vel) == 3
        assert all(isinstance(x, float) for x in pos)
        assert all(isinstance(v, float) for v in vel)

    def test_propagate_simplified_with_scenario_start_time_as_default(self):
        """测试使用scenario_start_time作为默认历元"""
        calculator = OrekitVisibilityCalculator()

        # 创建卫星，没有设置epoch
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=None,
            altitude=500000,
            inclination=97.4,
            mean_anomaly=0.0
        )

        # 场景开始时间作为默认历元
        scenario_start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2024, 6, 1, 1, 0, 0, tzinfo=timezone.utc)

        pos, vel = calculator._propagate_simplified(
            satellite, dt, scenario_start_time=scenario_start
        )

        assert len(pos) == 3
        assert len(vel) == 3

    def test_propagate_simplified_satellite_epoch_takes_precedence(self):
        """测试卫星epoch优先于scenario_start_time"""
        calculator = OrekitVisibilityCalculator()

        # 卫星有自己的epoch
        sat_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=sat_epoch,
            altitude=500000,
            inclination=97.4,
            mean_anomaly=0.0
        )

        # 场景开始时间不同
        scenario_start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        # 传播到卫星历元后1小时
        dt = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

        pos, vel = calculator._propagate_simplified(
            satellite, dt, scenario_start_time=scenario_start
        )

        # 应该成功传播（使用卫星自己的epoch）
        assert len(pos) == 3

    def test_propagate_simplified_warns_on_naive_datetime(self):
        """测试naive datetime发出警告并自动转换"""
        import warnings
        calculator = OrekitVisibilityCalculator()

        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            altitude=500000
        )

        # naive datetime应该发出警告
        naive_dt = datetime(2024, 1, 1, 1, 0, 0)

        with pytest.warns(UserWarning, match="naive datetime"):
            pos, vel = calculator._propagate_simplified(satellite, naive_dt)

        # 但应该仍然成功传播
        assert len(pos) == 3
        assert len(vel) == 3


class TestJ2Perturbation:
    """测试J2摄动修正"""

    def test_raan_precession_for_sso(self):
        """测试太阳同步轨道的RAAN进动"""
        calculator = OrekitVisibilityCalculator()

        # SSO轨道参数
        epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=epoch,
            altitude=500000,  # 约500km SSO
            inclination=97.4,  # SSO典型倾角
            raan=0.0,
            mean_anomaly=0.0
        )

        # 传播1天
        dt = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        pos1, _ = calculator._propagate_simplified(satellite, dt)

        # 传播2天
        dt2 = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        pos2, _ = calculator._propagate_simplified(satellite, dt2)

        # 位置应该不同（RAAN进动导致轨道面旋转）
        assert pos1 != pos2

    def test_zero_inclination_no_raan_precession(self):
        """测试零倾角轨道没有RAAN进动"""
        calculator = OrekitVisibilityCalculator()

        # 赤道轨道
        epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=epoch,
            altitude=500000,
            inclination=0.0,  # 赤道轨道
            raan=0.0,
            mean_anomaly=0.0
        )

        dt = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        pos, vel = calculator._propagate_simplified(satellite, dt)

        # 应该正常传播（没有除零错误）
        assert len(pos) == 3
        assert all(isinstance(x, float) for x in pos)


class TestDifferentEpochScenarios:
    """测试不同历元场景"""

    def test_epoch_before_scenario_start(self):
        """测试历元早于场景开始时间"""
        calculator = OrekitVisibilityCalculator()

        # 历元在1月1日
        epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=epoch,
            altitude=500000,
            mean_anomaly=0.0
        )

        # 场景开始和计算时间在1月15日
        scenario_start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        pos, vel = calculator._propagate_simplified(
            satellite, dt, scenario_start_time=scenario_start
        )

        assert len(pos) == 3

    def test_epoch_after_scenario_start(self):
        """测试历元晚于场景开始时间"""
        calculator = OrekitVisibilityCalculator()

        # 历元在2月1日
        epoch = datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        satellite = Mock()
        satellite.orbit = Orbit(
            epoch=epoch,
            altitude=500000,
            mean_anomaly=0.0
        )

        # 场景开始和计算时间在1月15日（历元之前）
        scenario_start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # 应该能处理负的delta_t
        pos, vel = calculator._propagate_simplified(
            satellite, dt, scenario_start_time=scenario_start
        )

        assert len(pos) == 3

    def test_multi_satellite_different_epochs(self):
        """测试多颗卫星不同历元"""
        calculator = OrekitVisibilityCalculator()

        # 卫星A：历元1月1日
        sat_a = Mock()
        sat_a.orbit = Orbit(
            epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            altitude=500000,
            mean_anomaly=0.0
        )

        # 卫星B：历元1月15日
        sat_b = Mock()
        sat_b.orbit = Orbit(
            epoch=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            altitude=600000,
            mean_anomaly=90.0
        )

        # 场景开始1月10日
        scenario_start = datetime(2024, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2024, 1, 20, 0, 0, 0, tzinfo=timezone.utc)

        # 两颗卫星都能正确传播
        pos_a, _ = calculator._propagate_simplified(
            sat_a, dt, scenario_start_time=scenario_start
        )
        pos_b, _ = calculator._propagate_simplified(
            sat_b, dt, scenario_start_time=scenario_start
        )

        assert len(pos_a) == 3
        assert len(pos_b) == 3
        # 不同卫星位置应该不同
        assert pos_a != pos_b
