"""
卫星历元时间支持 - 集成测试

验证多历元卫星场景、时区处理、向后兼容性
"""

import pytest
import json
import warnings
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.models.satellite import (
    Satellite, Orbit, SatelliteType, OrbitType, OrbitSource
)
from core.models.target import Target, TargetType
from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.orbit.visibility.stk_visibility import STKVisibilityCalculator


class TestMultiEpochScenario:
    """测试多历元卫星场景"""

    def test_create_multi_epoch_satellites(self):
        """测试创建包含多个不同历元的卫星"""
        # 卫星A：历元早于场景开始（1月1日）
        sat_a = Satellite(
            id="sat-a",
            name="Sat-A-Early-Epoch",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                altitude=500000,
                inclination=97.4,
                mean_anomaly=0.0
            )
        )

        # 卫星B：历元晚于场景开始（2月1日）
        sat_b = Satellite(
            id="sat-b",
            name="Sat-B-Late-Epoch",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
                altitude=600000,
                inclination=97.9,
                mean_anomaly=90.0
            )
        )

        # 卫星C：没有设置历元（使用场景开始时间）
        sat_c = Satellite(
            id="sat-c",
            name="Sat-C-No-Epoch",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                altitude=550000,
                inclination=97.6,
                mean_anomaly=180.0
            )
        )

        satellites = [sat_a, sat_b, sat_c]

        # 验证卫星创建成功
        assert len(satellites) == 3
        assert satellites[0].orbit.epoch is not None
        assert satellites[1].orbit.epoch is not None
        assert satellites[2].orbit.epoch is None  # 未设置

    def test_propagate_multi_epoch_satellites(self):
        """测试传播不同历元的卫星"""
        calculator = OrekitVisibilityCalculator()

        # 创建不同历元的卫星
        satellites = [
            Satellite(
                id="sat-1",
                name="Sat 1",
                sat_type=SatelliteType.OPTICAL_1,
                orbit=Orbit(
                    epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    altitude=500000,
                    mean_anomaly=0.0
                )
            ),
            Satellite(
                id="sat-2",
                name="Sat 2",
                sat_type=SatelliteType.OPTICAL_1,
                orbit=Orbit(
                    epoch=datetime(2024, 1, 10, 0, 0, 0, tzinfo=timezone.utc),
                    altitude=500000,
                    mean_anomaly=45.0
                )
            ),
            Satellite(
                id="sat-3",
                name="Sat 3",
                sat_type=SatelliteType.OPTICAL_1,
                orbit=Orbit(
                    epoch=datetime(2024, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
                    altitude=500000,
                    mean_anomaly=90.0
                )
            )
        ]

        # 场景时间：1月15日
        scenario_start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # 传播所有卫星
        positions = []
        for sat in satellites:
            pos, vel = calculator._propagate_simplified(
                sat, dt, scenario_start_time=scenario_start
            )
            positions.append(pos)

        # 验证所有卫星都成功传播
        assert len(positions) == 3
        for pos in positions:
            assert len(pos) == 3
            assert all(isinstance(x, (int, float)) for x in pos)

        # 不同历元的卫星位置应该不同
        assert positions[0] != positions[1]
        assert positions[1] != positions[2]


class TestTimezoneHandling:
    """测试时区处理"""

    def test_naive_datetime_warning(self):
        """测试使用naive datetime时发出警告"""
        calculator = OrekitVisibilityCalculator()

        sat = Satellite(
            id="sat-1",
            name="Test Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                altitude=500000
            )
        )

        # naive datetime应该发出警告
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)

        with pytest.warns(UserWarning, match="naive datetime"):
            pos, vel = calculator._propagate_simplified(sat, naive_dt)

        # 但应该成功传播
        assert pos is not None
        assert len(pos) == 3

    def test_different_timezones_converted_to_utc(self):
        """测试不同时区被正确转换为UTC"""
        # 创建东八区时间
        tz_plus8 = timezone(timedelta(hours=8))
        epoch_cst = datetime(2024, 1, 15, 20, 0, 0, tzinfo=tz_plus8)  # 北京时间20:00

        orbit = Orbit(epoch=epoch_cst, altitude=500000)

        # 应该被转换为UTC 12:00
        assert orbit.epoch.tzinfo == timezone.utc
        assert orbit.epoch.hour == 12

    def test_round_trip_timezone_preservation(self):
        """测试时区在序列化和反序列化后保持一致"""
        # 创建带时区的卫星
        original = Satellite(
            id="sat-1",
            name="Test Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                altitude=500000
            )
        )

        # 序列化
        data = original.to_dict()

        # 反序列化
        restored = Satellite.from_dict(data)

        # 验证时区保持UTC
        assert restored.orbit.epoch.tzinfo == timezone.utc
        assert restored.orbit.epoch.hour == 12


class TestTleEpochConflict:
    """测试TLE与epoch冲突"""

    def test_tle_with_epoch_emits_warning(self):
        """测试同时提供TLE和epoch时发出警告"""
        with pytest.warns(UserWarning, match="TLE内置历元将覆盖"):
            sat = Satellite.from_dict({
                "id": "sat-1",
                "name": "Test Sat",
                "sat_type": "optical_1",
                "orbit": {
                    "source": "tle",
                    "tle_line1": "1 25544U 98067A   24015.50000000  .00020000  00000-0  28000-4 0  9999",
                    "tle_line2": "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00",
                    "epoch": "2024-01-15T12:00:00Z"
                }
            })

        # 应该成功创建
        assert sat.orbit.tle_line1 is not None
        assert sat.orbit.source == OrbitSource.TLE

    def test_tle_without_epoch_no_warning(self):
        """测试只提供TLE时不发出警告"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误

            sat = Satellite.from_dict({
                "id": "sat-1",
                "name": "Test Sat",
                "sat_type": "optical_1",
                "orbit": {
                    "source": "tle",
                    "tle_line1": "1 25544U 98067A   24015.50000000  .00020000  00000-0  28000-4 0  9999",
                    "tle_line2": "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00"
                }
            })

            assert sat.orbit.tle_line1 is not None


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_old_scenario_without_epoch(self):
        """测试没有epoch的旧场景文件"""
        old_scenario_data = {
            "name": "Old Scenario",
            "start_time": "2024-01-15T00:00:00Z",
            "end_time": "2024-01-16T00:00:00Z",
            "satellites": [
                {
                    "id": "sat-1",
                    "name": "Old Sat",
                    "sat_type": "optical_1",
                    "orbit": {
                        "source": "simplified",
                        "altitude": 500000,
                        "inclination": 97.4
                    }
                }
            ]
        }

        # 创建场景
        sat = Satellite.from_dict(old_scenario_data["satellites"][0])

        # 验证卫星创建成功，epoch为None
        assert sat.orbit.epoch is None
        assert sat.orbit.altitude == 500000.0

    def test_propagate_without_epoch_uses_scenario_start(self):
        """测试没有epoch时使用场景开始时间"""
        calculator = OrekitVisibilityCalculator()

        # 没有epoch的卫星
        sat = Satellite(
            id="sat-1",
            name="No Epoch Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                altitude=500000,
                mean_anomaly=0.0
            )
        )

        assert sat.orbit.epoch is None

        # 场景开始时间
        scenario_start = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 传播
        pos, vel = calculator._propagate_simplified(
            sat, dt, scenario_start_time=scenario_start
        )

        # 应该成功传播
        assert pos is not None
        assert len(pos) == 3

    def test_old_serialization_format(self):
        """测试旧的序列化格式（不包含epoch）"""
        sat = Satellite(
            id="sat-1",
            name="Test Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                altitude=500000,
                inclination=97.4
            )
        )

        data = sat.to_dict()

        # 验证旧格式字段仍然存在
        assert "orbit_type" in data["orbit"]
        assert "source" in data["orbit"]
        assert "altitude" in data["orbit"]
        assert "inclination" in data["orbit"]

        # epoch为None时不应包含在输出中
        assert "epoch" not in data["orbit"] or data["orbit"]["epoch"] is None


class TestJ2PerturbationIntegration:
    """测试J2摄动集成"""

    def test_raan_precession_over_time(self):
        """测试RAAN随时间进动"""
        calculator = OrekitVisibilityCalculator()

        sat = Satellite(
            id="sat-1",
            name="SSO Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                altitude=500000,
                inclination=97.4,  # SSO倾角
                raan=0.0,
                mean_anomaly=0.0
            )
        )

        # 计算1天后的位置
        dt1 = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        pos1, _ = calculator._propagate_simplified(sat, dt1)

        # 计算10天后的位置
        dt2 = datetime(2024, 1, 11, 0, 0, 0, tzinfo=timezone.utc)
        pos2, _ = calculator._propagate_simplified(sat, dt2)

        # 位置应该不同（RAAN进动导致）
        assert pos1 != pos2

        # 距离应该合理（ shouldn't be too far apart）
        import math
        dist = math.sqrt(sum((p2 - p1)**2 for p1, p2 in zip(pos1, pos2)))
        assert dist > 1000000  # 大于1000km（因为RAAN进动）

    def test_equatorial_orbit_no_raan_precession(self):
        """测试赤道轨道没有RAAN进动"""
        calculator = OrekitVisibilityCalculator()

        sat = Satellite(
            id="sat-1",
            name="Equatorial Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                altitude=500000,
                inclination=0.0,  # 赤道轨道
                raan=0.0,
                mean_anomaly=0.0
            )
        )

        # 传播1天
        dt = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        pos, vel = calculator._propagate_simplified(sat, dt)

        # 应该正常传播（没有除零错误）
        assert pos is not None
        assert len(pos) == 3


class TestScenarioFileLoading:
    """测试场景文件加载"""

    def test_load_multi_epoch_satellites_from_json(self):
        """测试从JSON加载多历元卫星"""
        scenario_json = {
            "name": "Multi-Epoch Scenario",
            "start_time": "2024-01-15T00:00:00Z",
            "end_time": "2024-01-16T00:00:00Z",
            "satellites": [
                {
                    "id": "sat-a",
                    "name": "Sat A",
                    "sat_type": "optical_1",
                    "orbit": {
                        "source": "elements",
                        "semi_major_axis": 7016000,
                        "inclination": 97.4,
                        "mean_anomaly": 0.0,
                        "epoch": "2024-01-01T00:00:00Z"
                    }
                },
                {
                    "id": "sat-b",
                    "name": "Sat B",
                    "sat_type": "optical_1",
                    "orbit": {
                        "source": "elements",
                        "semi_major_axis": 7016000,
                        "inclination": 97.4,
                        "mean_anomaly": 90.0,
                        "epoch": "2024-02-01T00:00:00Z"
                    }
                }
            ],
            "targets": [
                {
                    "id": "target-1",
                    "name": "Beijing",
                    "target_type": "point",
                    "longitude": 116.4,
                    "latitude": 39.9
                }
            ]
        }

        # 创建卫星和目标
        satellites = [Satellite.from_dict(s) for s in scenario_json["satellites"]]
        targets = [Target.from_dict(t) for t in scenario_json["targets"]]

        # 验证
        assert len(satellites) == 2
        assert satellites[0].orbit.epoch is not None
        assert satellites[1].orbit.epoch is not None

    def test_save_and_load_satellites_with_epoch(self):
        """测试保存和加载带历元的卫星列表"""
        # 创建卫星列表
        satellites = [
            Satellite(
                id="sat-1",
                name="Sat 1",
                sat_type=SatelliteType.OPTICAL_1,
                orbit=Orbit(
                    epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    altitude=500000,
                    mean_anomaly=45.0
                )
            )
        ]

        # 序列化所有卫星
        data = {"satellites": [sat.to_dict() for sat in satellites]}

        # 验证epoch被正确序列化
        assert "epoch" in data["satellites"][0]["orbit"]
        assert data["satellites"][0]["orbit"]["epoch"] == "2024-01-01T00:00:00Z"

        # 从字典恢复
        restored_satellites = [Satellite.from_dict(s) for s in data["satellites"]]

        # 验证历元被正确恢复
        assert restored_satellites[0].orbit.epoch is not None
        assert restored_satellites[0].orbit.epoch.year == 2024


class TestVisibilityCalculationWithEpoch:
    """测试带历元的可见性计算"""

    def test_compute_visibility_different_epochs(self):
        """测试计算不同历元卫星的可见性"""
        calculator = OrekitVisibilityCalculator()

        satellites = [
            Satellite(
                id="sat-1",
                name="Early Epoch",
                sat_type=SatelliteType.OPTICAL_1,
                orbit=Orbit(
                    epoch=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    altitude=500000,
                    mean_anomaly=0.0
                )
            ),
            Satellite(
                id="sat-2",
                name="Late Epoch",
                sat_type=SatelliteType.OPTICAL_1,
                orbit=Orbit(
                    epoch=datetime(2024, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
                    altitude=500000,
                    mean_anomaly=180.0
                )
            )
        ]

        target = Target(
            id="beijing",
            name="Beijing",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9
        )

        start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # 计算可见性
        for sat in satellites:
            windows = calculator.compute_satellite_target_windows(
                sat, target, start_time, end_time
            )

            # 验证返回窗口列表（可能为空，但不应报错）
            assert isinstance(windows, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
