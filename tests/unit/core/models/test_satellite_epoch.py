"""
卫星历元时间支持测试

测试Orbit类的epoch字段功能、UTC时区处理、序列化/反序列化
"""

import pytest
import warnings
from datetime import datetime, timezone, timedelta
from core.models.satellite import Orbit, Satellite, OrbitType, OrbitSource, SatelliteType


class TestOrbitEpochField:
    """测试Orbit类的epoch字段"""

    def test_orbit_has_epoch_field(self):
        """测试Orbit类有epoch字段"""
        orbit = Orbit()
        assert hasattr(orbit, 'epoch')
        assert orbit.epoch is None

    def test_orbit_epoch_with_timezone_aware_datetime(self):
        """测试使用带时区的datetime创建Orbit"""
        epoch = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        orbit = Orbit(epoch=epoch)
        assert orbit.epoch == epoch
        assert orbit.epoch.tzinfo == timezone.utc

    def test_orbit_epoch_with_different_timezone_converts_to_utc(self):
        """测试不同时区的datetime会被转换为UTC"""
        # 东八区时间
        tz_plus8 = timezone(timedelta(hours=8))
        epoch_cst = datetime(2024, 1, 15, 20, 0, 0, tzinfo=tz_plus8)  # 北京时间20:00

        orbit = Orbit(epoch=epoch_cst)

        # 应该被转换为UTC时间12:00
        assert orbit.epoch.tzinfo == timezone.utc
        assert orbit.epoch.hour == 12
        assert orbit.epoch.day == 15

    def test_orbit_epoch_naive_datetime_emits_warning(self):
        """测试使用naive datetime会发出警告并假设为UTC"""
        naive_epoch = datetime(2024, 1, 15, 12, 0, 0)  # 无时区

        with pytest.warns(UserWarning, match="naive datetime"):
            orbit = Orbit(epoch=naive_epoch)

        # 应该被设置为UTC时区
        assert orbit.epoch.tzinfo == timezone.utc
        assert orbit.epoch.hour == 12

    def test_orbit_epoch_none_no_warning(self):
        """测试epoch为None时不会发出警告"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误
            orbit = Orbit(epoch=None)
            assert orbit.epoch is None


class TestEnsureUtcDatetime:
    """测试ensure_utc_datetime工具函数"""

    def test_ensure_utc_datetime_with_none(self):
        """测试None返回None"""
        from core.models.satellite import ensure_utc_datetime
        result = ensure_utc_datetime(None)
        assert result is None

    def test_ensure_utc_datetime_with_aware_datetime(self):
        """测试带时区的datetime返回UTC"""
        from core.models.satellite import ensure_utc_datetime
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc_datetime(dt)
        assert result == dt
        assert result.tzinfo == timezone.utc

    def test_ensure_utc_datetime_with_naive_datetime(self):
        """测试naive datetime被添加UTC时区"""
        from core.models.satellite import ensure_utc_datetime
        dt = datetime(2024, 1, 15, 12, 0, 0)
        result = ensure_utc_datetime(dt)
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.hour == 12

    def test_ensure_utc_datetime_converts_to_utc(self):
        """测试不同时区被转换为UTC"""
        from core.models.satellite import ensure_utc_datetime
        tz_plus8 = timezone(timedelta(hours=8))
        dt_cst = datetime(2024, 1, 15, 20, 0, 0, tzinfo=tz_plus8)
        result = ensure_utc_datetime(dt_cst)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12  # 20:00 CST = 12:00 UTC


class TestParseEpochString:
    """测试parse_epoch_string工具函数"""

    def test_parse_iso8601_with_z(self):
        """测试解析ISO 8601带Z格式"""
        from core.models.satellite import parse_epoch_string
        result = parse_epoch_string("2024-01-15T12:00:00Z")
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 12

    def test_parse_iso8601_with_microseconds_and_z(self):
        """测试解析带微秒的ISO 8601格式"""
        from core.models.satellite import parse_epoch_string
        result = parse_epoch_string("2024-01-15T12:00:00.123456Z")
        assert result.tzinfo == timezone.utc
        assert result.microsecond == 123456

    def test_parse_iso8601_with_timezone_offset(self):
        """测试解析带时区偏移的ISO 8601格式"""
        from core.models.satellite import parse_epoch_string
        result = parse_epoch_string("2024-01-15T20:00:00+08:00")
        assert result.tzinfo == timezone.utc
        assert result.hour == 12  # 转换为UTC

    def test_parse_iso8601_without_timezone(self):
        """测试解析无时区的ISO 8601格式（假设为UTC）"""
        from core.models.satellite import parse_epoch_string
        result = parse_epoch_string("2024-01-15T12:00:00")
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_parse_date_only(self):
        """测试解析日期only格式"""
        from core.models.satellite import parse_epoch_string
        result = parse_epoch_string("2024-01-15")
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0

    def test_parse_invalid_format_raises_error(self):
        """测试无效格式抛出ValueError"""
        from core.models.satellite import parse_epoch_string
        with pytest.raises(ValueError, match="Cannot parse epoch string"):
            parse_epoch_string("invalid-date-string")


class TestOrbitSerialization:
    """测试Orbit序列化/反序列化"""

    def test_to_dict_includes_epoch(self):
        """测试to_dict包含epoch字段"""
        epoch = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        orbit = Orbit(epoch=epoch, altitude=500000)

        satellite = Satellite(
            id="test-1",
            name="Test Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=orbit
        )

        result = satellite.to_dict()
        assert "epoch" in result["orbit"]
        assert result["orbit"]["epoch"] == "2024-01-15T12:00:00Z"

    def test_to_dict_epoch_none_not_included(self):
        """测试epoch为None时不包含在dict中"""
        orbit = Orbit(epoch=None, altitude=500000)

        satellite = Satellite(
            id="test-1",
            name="Test Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=orbit
        )

        result = satellite.to_dict()
        assert "epoch" not in result["orbit"] or result["orbit"]["epoch"] is None

    def test_from_dict_parses_epoch(self):
        """测试from_dict解析epoch字段"""
        data = {
            "id": "test-1",
            "name": "Test Sat",
            "sat_type": "optical_1",
            "orbit": {
                "source": "simplified",
                "altitude": 500000,
                "inclination": 97.4,
                "epoch": "2024-01-15T12:00:00Z"
            }
        }

        satellite = Satellite.from_dict(data)
        assert satellite.orbit.epoch is not None
        assert satellite.orbit.epoch.tzinfo == timezone.utc
        assert satellite.orbit.epoch.year == 2024

    def test_from_dict_epoch_none(self):
        """测试from_dict处理epoch为None的情况"""
        data = {
            "id": "test-1",
            "name": "Test Sat",
            "sat_type": "optical_1",
            "orbit": {
                "source": "simplified",
                "altitude": 500000,
                "inclination": 97.4
            }
        }

        satellite = Satellite.from_dict(data)
        assert satellite.orbit.epoch is None

    def test_round_trip_serialization(self):
        """测试序列化和反序列化的往返一致性"""
        epoch = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)
        original = Satellite(
            id="test-1",
            name="Test Sat",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                epoch=epoch,
                altitude=500000,
                inclination=97.4,
                mean_anomaly=45.0
            )
        )

        # 序列化
        data = original.to_dict()

        # 反序列化
        restored = Satellite.from_dict(data)

        # 验证epoch一致
        assert restored.orbit.epoch is not None
        assert restored.orbit.epoch.tzinfo == timezone.utc
        assert restored.orbit.epoch.year == 2024
        assert restored.orbit.epoch.hour == 12


class TestTleEpochConflictWarning:
    """测试TLE与epoch冲突警告"""

    def test_tle_with_epoch_emits_warning(self):
        """测试同时提供TLE和epoch时发出警告"""
        data = {
            "id": "test-1",
            "name": "Test Sat",
            "sat_type": "optical_1",
            "orbit": {
                "source": "tle",
                "tle_line1": "1 25544U 98067A   24015.50000000  .00020000  00000-0  28000-4 0  9999",
                "tle_line2": "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00",
                "epoch": "2024-01-15T12:00:00Z"
            }
        }

        with pytest.warns(UserWarning, match="TLE内置历元将覆盖"):
            satellite = Satellite.from_dict(data)

        # 应该仍然成功创建
        assert satellite.orbit.tle_line1 is not None

    def test_tle_without_epoch_no_warning(self):
        """测试只提供TLE时不发出警告"""
        data = {
            "id": "test-1",
            "name": "Test Sat",
            "sat_type": "optical_1",
            "orbit": {
                "source": "tle",
                "tle_line1": "1 25544U 98067A   24015.50000000  .00020000  00000-0  28000-4 0  9999",
                "tle_line2": "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00"
            }
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误
            satellite = Satellite.from_dict(data)
            assert satellite.orbit.tle_line1 is not None


class TestOrbitSourcePriority:
    """测试轨道数据来源优先级"""

    def test_tle_has_priority(self):
        """测试TLE优先于六根数和简化参数"""
        data = {
            "id": "test-1",
            "name": "Test Sat",
            "sat_type": "optical_1",
            "orbit": {
                "tle_line1": "1 25544U 98067A   24015.50000000  .00020000  00000-0  28000-4 0  9999",
                "tle_line2": "2 25544  51.6416  30.0000 0005000  45.0000  15.0000 15.50000000    00",
                "semi_major_axis": 7000000,  # 六根数参数应该被忽略
                "altitude": 600000  # 简化参数应该被忽略
            }
        }

        satellite = Satellite.from_dict(data)
        assert satellite.orbit.source == OrbitSource.TLE

    def test_elements_have_priority_over_simplified(self):
        """测试六根数优先于简化参数"""
        data = {
            "id": "test-1",
            "name": "Test Sat",
            "sat_type": "optical_1",
            "orbit": {
                "semi_major_axis": 7000000,
                "inclination": 97.9,
                "altitude": 600000  # 应该被忽略，从semi_major_axis计算
            }
        }

        satellite = Satellite.from_dict(data)
        assert satellite.orbit.source == OrbitSource.ELEMENTS
