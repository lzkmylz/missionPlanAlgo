"""
任务失败原因枚举单元测试

测试H2: 完整的TaskFailureReason枚举 (Chapter 12.4)
"""

import pytest
from scheduler.base_scheduler import TaskFailureReason


class TestTaskFailureReason:
    """测试任务失败原因枚举"""

    def test_resource_constraints(self):
        """测试资源约束失败原因"""
        assert TaskFailureReason.POWER_CONSTRAINT.value == "power_constraint"
        assert TaskFailureReason.STORAGE_CONSTRAINT.value == "storage_constraint"
        assert TaskFailureReason.STORAGE_OVERFLOW_RISK.value == "storage_overflow_risk"

    def test_time_constraints(self):
        """测试时间约束失败原因"""
        assert TaskFailureReason.NO_VISIBLE_WINDOW.value == "no_visible_window"
        assert TaskFailureReason.WINDOW_TOO_SHORT.value == "window_too_short"
        assert TaskFailureReason.TIME_CONFLICT.value == "time_conflict"
        assert TaskFailureReason.DEADLINE_VIOLATION.value == "deadline_violation"

    def test_capability_constraints(self):
        """测试能力约束失败原因"""
        assert TaskFailureReason.SAT_CAPABILITY_MISMATCH.value == "sat_capability_mismatch"
        assert TaskFailureReason.MODE_NOT_SUPPORTED.value == "mode_not_supported"
        assert TaskFailureReason.OFF_NADIR_EXCEEDED.value == "off_nadir_exceeded"

    def test_collaboration_constraints(self):
        """测试协同约束失败原因"""
        assert TaskFailureReason.GROUND_STATION_UNAVAILABLE.value == "ground_station_unavailable"
        assert TaskFailureReason.ANTENNA_CONFLICT.value == "antenna_conflict"

    def test_physical_constraints(self):
        """测试物理约束失败原因 - H2新增"""
        assert TaskFailureReason.THERMAL_CONSTRAINT.value == "thermal_constraint"
        assert TaskFailureReason.SUN_EXCLUSION_VIOLATION.value == "sun_exclusion_violation"
        assert TaskFailureReason.STORAGE_FRAGMENTATION.value == "storage_fragmentation"

    def test_network_constraints(self):
        """测试网络约束失败原因 - H2新增"""
        assert TaskFailureReason.NO_ISL_PATH.value == "no_isl_path"
        assert TaskFailureReason.UPLINK_UNAVAILABLE.value == "uplink_unavailable"
        assert TaskFailureReason.RELAY_OVERLOAD.value == "relay_overload"

    def test_other_reasons(self):
        """测试其他失败原因"""
        assert TaskFailureReason.UNKNOWN.value == "unknown"
        assert TaskFailureReason.ALGORITHM_TIMEOUT.value == "algorithm_timeout"

    def test_total_reason_count(self):
        """测试失败原因总数"""
        # 根据Chapter 12.4设计，应该有15+种失败原因
        reasons = list(TaskFailureReason)
        assert len(reasons) >= 15

        # 验证所有原因都有值
        for reason in reasons:
            assert reason.value is not None
            assert isinstance(reason.value, str)

    def test_reason_uniqueness(self):
        """测试失败原因值唯一性"""
        reasons = list(TaskFailureReason)
        values = [r.value for r in reasons]
        assert len(values) == len(set(values)), "失败原因值必须唯一"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
