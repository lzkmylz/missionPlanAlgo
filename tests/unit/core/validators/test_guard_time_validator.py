"""
测试GuardTimeValidator独立模块

M3: GuardTimeValidator独立模块测试
"""

import pytest
from datetime import datetime, timedelta

from core.validators.guard_time_validator import (
    GuardTimeValidator,
    GuardTimeRule,
    GuardTimeViolationError
)
from core.telecommand.soe_generator import SOEActionType, SOEEntry


class TestGuardTimeRule:
    """测试GuardTimeRule数据类"""

    def test_rule_creation(self):
        """测试规则创建"""
        rule = GuardTimeRule(
            action_a=SOEActionType.PAYLOAD_POWER_ON,
            action_b=SOEActionType.PAYLOAD_WARMUP,
            min_interval=timedelta(seconds=5),
            reason="Power stabilization"
        )

        assert rule.action_a == SOEActionType.PAYLOAD_POWER_ON
        assert rule.action_b == SOEActionType.PAYLOAD_WARMUP
        assert rule.min_interval == timedelta(seconds=5)
        assert rule.reason == "Power stabilization"


class TestGuardTimeValidatorInit:
    """测试GuardTimeValidator初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        validator = GuardTimeValidator()

        # 验证默认规则存在
        assert len(validator.rules) > 0

        # 验证默认规则包含关键规则
        rule_pairs = [(r.action_a, r.action_b) for r in validator.rules]
        assert (SOEActionType.PAYLOAD_POWER_ON, SOEActionType.PAYLOAD_WARMUP) in rule_pairs
        assert (SOEActionType.SLEW_COMPLETE, SOEActionType.IMAGING_START) in rule_pairs

    def test_custom_rules(self):
        """测试自定义规则"""
        custom_rules = [
            GuardTimeRule(
                SOEActionType.IMAGING_COMPLETE,
                SOEActionType.DOWNLINK_START,
                timedelta(seconds=20),
                "Custom delay"
            )
        ]

        validator = GuardTimeValidator(custom_rules)
        assert len(validator.rules) == 1
        assert validator.rules[0].min_interval == timedelta(seconds=20)

    def test_add_rule(self):
        """测试动态添加规则"""
        validator = GuardTimeValidator()
        initial_count = len(validator.rules)

        new_rule = GuardTimeRule(
            SOEActionType.DOWNLINK_START,
            SOEActionType.DOWNLINK_COMPLETE,
            timedelta(seconds=1),
            "Minimum downlink duration"
        )

        validator.add_rule(new_rule)
        assert len(validator.rules) == initial_count + 1


class TestGuardTimeValidatorValidation:
    """测试验证功能"""

    def create_soe_entry(self, timestamp, action_type, duration=None, sat_id="SAT-01"):
        """辅助方法：创建SOE条目"""
        return SOEEntry(
            timestamp=timestamp,
            action_type=action_type,
            satellite_id=sat_id,
            task_id="TASK-01",
            duration=duration or timedelta(seconds=0),
            parameters={}
        )

    def test_valid_soe_no_violations(self):
        """测试无违规的有效SOE"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 10),  # 10秒后，满足5秒间隔
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        violations = validator.validate_soe(entries)
        assert len(violations) == 0

    def test_violation_detected(self):
        """测试检测到违规"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON,
                duration=timedelta(seconds=1)
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 2),  # 只有1秒间隔，需要5秒
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        violations = validator.validate_soe(entries)
        assert len(violations) > 0

    def test_violation_details(self):
        """测试违规详情"""
        validator = GuardTimeValidator()

        entry_a = self.create_soe_entry(
            datetime(2024, 1, 1, 10, 0, 0),
            SOEActionType.PAYLOAD_POWER_ON,
            duration=timedelta(seconds=1)
        )
        entry_b = self.create_soe_entry(
            datetime(2024, 1, 1, 10, 0, 2),
            SOEActionType.PAYLOAD_WARMUP
        )

        violations = validator.validate_soe([entry_a, entry_b])

        assert len(violations) == 1
        violation = violations[0]

        assert 'entry_a' in violation
        assert 'entry_b' in violation
        assert 'rule' in violation
        assert 'required_interval' in violation
        assert 'actual_interval' in violation

        assert violation['required_interval'] == timedelta(seconds=5)
        assert violation['actual_interval'] == timedelta(seconds=1)

    def test_multi_satellite_validation(self):
        """测试多卫星验证"""
        validator = GuardTimeValidator()

        # SAT-01的条目
        entries_sat1 = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON,
                sat_id="SAT-01"
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 2),  # 违规
                SOEActionType.PAYLOAD_WARMUP,
                sat_id="SAT-01"
            ),
        ]

        # SAT-02的条目（无违规）
        entries_sat2 = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON,
                sat_id="SAT-02"
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 10),  # 满足间隔
                SOEActionType.PAYLOAD_WARMUP,
                sat_id="SAT-02"
            ),
        ]

        all_entries = entries_sat1 + entries_sat2
        violations = validator.validate_soe(all_entries)

        # 只应该检测到SAT-01的违规
        assert len(violations) == 1
        assert violations[0]['entry_a'].satellite_id == "SAT-01"

    def test_empty_soe(self):
        """测试空SOE序列"""
        validator = GuardTimeValidator()

        violations = validator.validate_soe([])
        assert len(violations) == 0

    def test_single_entry(self):
        """测试单一条目"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.IMAGING_START
            )
        ]

        violations = validator.validate_soe(entries)
        assert len(violations) == 0


class TestGuardTimeValidatorAutoFix:
    """测试自动修复功能"""

    def create_soe_entry(self, timestamp, action_type, duration=None, sat_id="SAT-01"):
        """辅助方法：创建SOE条目"""
        return SOEEntry(
            timestamp=timestamp,
            action_type=action_type,
            satellite_id=sat_id,
            task_id="TASK-01",
            duration=duration or timedelta(seconds=0),
            parameters={}
        )

    def test_auto_fix_violations(self):
        """测试自动修复违规"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON,
                duration=timedelta(seconds=1)
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 2),  # 违规
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        fixed = validator.auto_fix(entries)

        # 验证修复后无违规
        violations = validator.validate_soe(fixed)
        assert len(violations) == 0

    def test_auto_fix_preserves_order(self):
        """测试自动修复保持顺序"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 1),  # 违规
                SOEActionType.PAYLOAD_WARMUP
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 10),
                SOEActionType.IMAGING_START
            ),
        ]

        fixed = validator.auto_fix(entries)

        # 验证时间顺序
        for i in range(len(fixed) - 1):
            assert fixed[i].timestamp <= fixed[i + 1].timestamp

    def test_auto_fix_no_violations(self):
        """测试无违规时的自动修复"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 10),  # 满足间隔
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        fixed = validator.auto_fix(entries)

        # 无违规时不应改变
        assert len(fixed) == len(entries)
        assert fixed[0].timestamp == entries[0].timestamp
        assert fixed[1].timestamp == entries[1].timestamp

    def test_auto_fix_does_not_modify_original(self):
        """测试自动修复不修改原始列表"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 1),  # 违规
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        original_timestamps = [e.timestamp for e in entries]
        fixed = validator.auto_fix(entries)

        # 验证原始列表未修改
        assert entries[0].timestamp == original_timestamps[0]
        assert entries[1].timestamp == original_timestamps[1]

        # 修复后的列表应该不同
        assert fixed[1].timestamp != entries[1].timestamp


class TestGuardTimeValidatorEdgeCases:
    """测试边界情况"""

    def create_soe_entry(self, timestamp, action_type, duration=None, sat_id="SAT-01"):
        """辅助方法：创建SOE条目"""
        return SOEEntry(
            timestamp=timestamp,
            action_type=action_type,
            satellite_id=sat_id,
            task_id="TASK-01",
            duration=duration or timedelta(seconds=0),
            parameters={}
        )

    def test_overlapping_entries(self):
        """测试重叠条目"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON,
                duration=timedelta(seconds=10)
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 5),  # 在第一个条目的持续时间内
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        violations = validator.validate_soe(entries)
        # 应该检测到违规（负间隔）
        assert len(violations) > 0
        assert violations[0]['actual_interval'] < timedelta(0)

    def test_same_timestamp_different_satellites(self):
        """测试不同卫星同一时间戳"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON,
                sat_id="SAT-01"
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),  # 同一时间，不同卫星
                SOEActionType.PAYLOAD_WARMUP,
                sat_id="SAT-02"
            ),
        ]

        violations = validator.validate_soe(entries)
        # 不同卫星之间不应该有违规
        assert len(violations) == 0

    def test_unordered_entries(self):
        """测试无序条目"""
        validator = GuardTimeValidator()

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 10),
                SOEActionType.PAYLOAD_WARMUP
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),  # 时间更早
                SOEActionType.PAYLOAD_POWER_ON
            ),
        ]

        violations = validator.validate_soe(entries)
        # 验证器应该能处理无序条目
        assert isinstance(violations, list)

    def test_no_applicable_rules(self):
        """测试无适用规则的情况"""
        validator = GuardTimeValidator([])  # 空规则

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 1),
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        violations = validator.validate_soe(entries)
        # 无规则则无违规
        assert len(violations) == 0

    def test_multiple_violations_same_pair(self):
        """测试同一对条目的多个违规"""
        validator = GuardTimeValidator([
            GuardTimeRule(
                SOEActionType.PAYLOAD_POWER_ON,
                SOEActionType.PAYLOAD_WARMUP,
                timedelta(seconds=5),
                "Rule 1"
            ),
            GuardTimeRule(
                SOEActionType.PAYLOAD_POWER_ON,
                SOEActionType.PAYLOAD_WARMUP,
                timedelta(seconds=10),
                "Rule 2"
            ),
        ])

        entries = [
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 0),
                SOEActionType.PAYLOAD_POWER_ON
            ),
            self.create_soe_entry(
                datetime(2024, 1, 1, 10, 0, 3),  # 违反两条规则
                SOEActionType.PAYLOAD_WARMUP
            ),
        ]

        violations = validator.validate_soe(entries)
        # 应该检测到两个违规
        assert len(violations) == 2
