"""
SOE生成器模块的单元测试

TDD流程:
1. RED: 编写测试 - 验证SOE生成器的行为
2. GREEN: 实现代码使测试通过
3. REFACTOR: 优化代码结构
"""

import pytest
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional


# ==========================================
# 测试用例: SOEActionType Enum
# ==========================================
class TestSOEActionType:
    """SOE动作类型枚举测试"""

    def test_action_type_values(self):
        """测试所有动作类型值是否正确"""
        from core.telecommand.soe_generator import SOEActionType

        assert SOEActionType.PAYLOAD_POWER_ON.value == "PAYLOAD_POWER_ON"
        assert SOEActionType.PAYLOAD_WARMUP.value == "PAYLOAD_WARMUP"
        assert SOEActionType.SLEW_START.value == "SLEW_START"
        assert SOEActionType.SLEW_COMPLETE.value == "SLEW_COMPLETE"
        assert SOEActionType.SHUTTER_OPEN.value == "SHUTTER_OPEN"
        assert SOEActionType.IMAGING_START.value == "IMAGING_START"
        assert SOEActionType.IMAGING_COMPLETE.value == "IMAGING_COMPLETE"
        assert SOEActionType.SHUTTER_CLOSE.value == "SHUTTER_CLOSE"
        assert SOEActionType.DOWNLINK_START.value == "DOWNLINK_START"
        assert SOEActionType.DOWNLINK_COMPLETE.value == "DOWNLINK_COMPLETE"

    def test_action_type_comparison(self):
        """测试动作类型可以正确比较"""
        from core.telecommand.soe_generator import SOEActionType

        assert SOEActionType.PAYLOAD_POWER_ON == SOEActionType.PAYLOAD_POWER_ON
        assert SOEActionType.PAYLOAD_POWER_ON != SOEActionType.PAYLOAD_WARMUP

    def test_action_type_iteration(self):
        """测试可以遍历所有动作类型"""
        from core.telecommand.soe_generator import SOEActionType

        action_types = list(SOEActionType)
        assert len(action_types) == 10


# ==========================================
# 测试用例: SOEEntry dataclass
# ==========================================
class TestSOEEntry:
    """SOE条目数据类测试"""

    def test_soe_entry_creation(self):
        """测试SOEEntry可以正确创建"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        timestamp = datetime(2026, 1, 1, 12, 0, 0)
        entry = SOEEntry(
            timestamp=timestamp,
            action_type=SOEActionType.IMAGING_START,
            satellite_id="SAT-001",
            task_id="TASK-001",
            duration=timedelta(seconds=60),
            parameters={"target_id": "TGT-001"},
            guard_time_before=timedelta(seconds=5),
            guard_time_after=timedelta(seconds=5)
        )

        assert entry.timestamp == timestamp
        assert entry.action_type == SOEActionType.IMAGING_START
        assert entry.satellite_id == "SAT-001"
        assert entry.task_id == "TASK-001"
        assert entry.duration == timedelta(seconds=60)
        assert entry.parameters == {"target_id": "TGT-001"}
        assert entry.guard_time_before == timedelta(seconds=5)
        assert entry.guard_time_after == timedelta(seconds=5)

    def test_soe_entry_default_guard_times(self):
        """测试SOEEntry默认保护时间为0"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        entry = SOEEntry(
            timestamp=datetime.now(),
            action_type=SOEActionType.IMAGING_START,
            satellite_id="SAT-001",
            task_id="TASK-001",
            duration=timedelta(seconds=60),
            parameters={}
        )

        assert entry.guard_time_before == timedelta(seconds=0)
        assert entry.guard_time_after == timedelta(seconds=0)

    def test_soe_entry_optional_task_id(self):
        """测试SOEEntry的task_id可以为None"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        entry = SOEEntry(
            timestamp=datetime.now(),
            action_type=SOEActionType.PAYLOAD_POWER_ON,
            satellite_id="SAT-001",
            task_id=None,
            duration=timedelta(seconds=30),
            parameters={}
        )

        assert entry.task_id is None

    def test_soe_entry_optional_duration(self):
        """测试SOEEntry的duration可以为None"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        entry = SOEEntry(
            timestamp=datetime.now(),
            action_type=SOEActionType.IMAGING_START,
            satellite_id="SAT-001",
            task_id="TASK-001",
            duration=None,
            parameters={}
        )

        assert entry.duration is None


# ==========================================
# 测试用例: SOEGenerator
# ==========================================
class TestSOEGenerator:
    """SOE生成器测试"""

    @pytest.fixture
    def sample_schedule(self):
        """创建示例调度结果"""
        from scheduler.base_scheduler import ScheduleResult, ScheduledTask

        base_time = datetime(2026, 1, 1, 12, 0, 0)
        return ScheduleResult(
            scheduled_tasks=[
                ScheduledTask(
                    task_id="TASK-001",
                    satellite_id="SAT-001",
                    target_id="TGT-001",
                    imaging_start=base_time,
                    imaging_end=base_time + timedelta(seconds=60),
                    imaging_mode="optical_high",
                    slew_angle=15.5
                )
            ],
            unscheduled_tasks={},
            makespan=60.0,
            computation_time=1.0,
            iterations=10
        )

    @pytest.fixture
    def sample_optical_task(self):
        """创建示例光学成像任务"""
        return {
            'task_id': 'TASK-001',
            'satellite_id': 'SAT-001',
            'target_id': 'TGT-001',
            'imaging_start': datetime(2026, 1, 1, 12, 0, 0),
            'imaging_end': datetime(2026, 1, 1, 12, 1, 0),
            'imaging_mode': 'optical_high',
            'slew_angle': 15.5,
            'imaging_type': 'optical'
        }

    @pytest.fixture
    def sample_sar_task(self):
        """创建示例SAR成像任务"""
        return {
            'task_id': 'TASK-002',
            'satellite_id': 'SAT-002',
            'target_id': 'TGT-002',
            'imaging_start': datetime(2026, 1, 1, 12, 0, 0),
            'imaging_end': datetime(2026, 1, 1, 12, 0, 30),
            'imaging_mode': 'stripmap',
            'slew_angle': 10.0,
            'imaging_type': 'sar'
        }

    def test_soe_generator_creation(self):
        """测试SOEGenerator可以正确创建"""
        from core.telecommand.soe_generator import SOEGenerator

        generator = SOEGenerator()
        assert generator is not None
        assert hasattr(generator, 'ACTION_TEMPLATES')
        assert hasattr(generator, 'GUARD_TIMES')

    def test_action_templates_structure(self):
        """测试动作模板结构正确"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()

        # 检查光学成像模板
        assert 'optical_imaging' in generator.ACTION_TEMPLATES
        optical_template = generator.ACTION_TEMPLATES['optical_imaging']
        assert len(optical_template) > 0

        # 检查SAR成像模板
        assert 'sar_imaging' in generator.ACTION_TEMPLATES
        sar_template = generator.ACTION_TEMPLATES['sar_imaging']
        assert len(sar_template) > 0

    def test_guard_times_structure(self):
        """测试保护时间配置结构正确"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()

        # 检查保护时间配置
        assert SOEActionType.PAYLOAD_POWER_ON in generator.GUARD_TIMES
        assert SOEActionType.SLEW_START in generator.GUARD_TIMES
        assert SOEActionType.SHUTTER_OPEN in generator.GUARD_TIMES

        # 检查保护时间格式 (before, after)
        guard_time = generator.GUARD_TIMES[SOEActionType.PAYLOAD_POWER_ON]
        assert len(guard_time) == 2
        assert isinstance(guard_time[0], timedelta)
        assert isinstance(guard_time[1], timedelta)

    def test_generate_soe_empty_schedule(self):
        """测试空调度计划生成空SOE"""
        from core.telecommand.soe_generator import SOEGenerator
        from scheduler.base_scheduler import ScheduleResult

        generator = SOEGenerator()
        empty_schedule = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        soe_entries = generator.generate_soe(empty_schedule)
        assert soe_entries == []

    def test_generate_soe_single_optical_task(self, sample_schedule):
        """测试单个光学成像任务生成SOE"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()
        soe_entries = generator.generate_soe(sample_schedule)

        # 应该生成多个SOE条目
        assert len(soe_entries) > 0

        # 检查包含关键动作
        action_types = [entry.action_type for entry in soe_entries]
        assert SOEActionType.IMAGING_START in action_types

        # 检查按时间排序
        for i in range(len(soe_entries) - 1):
            assert soe_entries[i].timestamp <= soe_entries[i + 1].timestamp

    def test_generate_task_soe_optical(self, sample_optical_task):
        """测试为单个光学任务生成SOE"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()
        entries = generator._generate_task_soe(sample_optical_task)

        # 检查生成了条目
        assert len(entries) > 0

        # 检查包含光学成像特有动作
        action_types = [entry.action_type for entry in entries]
        assert SOEActionType.PAYLOAD_POWER_ON in action_types
        assert SOEActionType.PAYLOAD_WARMUP in action_types
        assert SOEActionType.SLEW_START in action_types
        assert SOEActionType.SHUTTER_OPEN in action_types
        assert SOEActionType.IMAGING_START in action_types

        # 检查卫星ID和任务ID正确
        for entry in entries:
            assert entry.satellite_id == "SAT-001"
            assert entry.task_id == "TASK-001"

    def test_generate_task_soe_sar(self, sample_sar_task):
        """测试为单个SAR任务生成SOE"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()
        entries = generator._generate_task_soe(sample_sar_task)

        # 检查生成了条目
        assert len(entries) > 0

        # 检查包含SAR成像特有动作
        action_types = [entry.action_type for entry in entries]
        assert SOEActionType.PAYLOAD_POWER_ON in action_types
        assert SOEActionType.PAYLOAD_WARMUP in action_types
        assert SOEActionType.SLEW_START in action_types
        assert SOEActionType.IMAGING_START in action_types

        # SAR成像没有快门动作
        assert SOEActionType.SHUTTER_OPEN not in action_types
        assert SOEActionType.SHUTTER_CLOSE not in action_types

    def test_generate_task_soe_timing(self, sample_optical_task):
        """测试SOE条目时间戳正确"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()
        entries = generator._generate_task_soe(sample_optical_task)

        imaging_start = sample_optical_task['imaging_start']

        # 找到IMAGING_START条目
        imaging_entry = next(
            e for e in entries if e.action_type == SOEActionType.IMAGING_START
        )
        assert imaging_entry.timestamp == imaging_start

        # 找到PAYLOAD_POWER_ON条目 (应该在成像前300秒)
        power_entry = next(
            e for e in entries if e.action_type == SOEActionType.PAYLOAD_POWER_ON
        )
        assert power_entry.timestamp == imaging_start - timedelta(seconds=300)

    def test_generate_task_soe_with_guard_times(self, sample_optical_task):
        """测试SOE条目包含保护时间"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()
        entries = generator._generate_task_soe(sample_optical_task)

        # 检查保护时间设置
        for entry in entries:
            if entry.action_type in generator.GUARD_TIMES:
                before, after = generator.GUARD_TIMES[entry.action_type]
                assert entry.guard_time_before == before
                assert entry.guard_time_after == after

    def test_generate_task_soe_parameters(self, sample_optical_task):
        """测试SOE条目包含正确的参数"""
        from core.telecommand.soe_generator import SOEGenerator

        generator = SOEGenerator()
        entries = generator._generate_task_soe(sample_optical_task)

        for entry in entries:
            assert 'target_id' in entry.parameters
            assert entry.parameters['target_id'] == 'TGT-001'
            assert 'imaging_mode' in entry.parameters
            assert entry.parameters['imaging_mode'] == 'optical_high'
            assert 'slew_angle' in entry.parameters
            assert entry.parameters['slew_angle'] == 15.5

    def test_generate_task_soe_unknown_type(self):
        """测试未知任务类型返回空列表"""
        from core.telecommand.soe_generator import SOEGenerator

        generator = SOEGenerator()
        unknown_task = {
            'task_id': 'TASK-003',
            'satellite_id': 'SAT-003',
            'target_id': 'TGT-003',
            'imaging_start': datetime.now(),
            'imaging_end': datetime.now() + timedelta(seconds=60),
            'imaging_mode': 'unknown',
            'imaging_type': 'unknown_type'
        }

        entries = generator._generate_task_soe(unknown_task)
        assert entries == []

    def test_generate_task_soe_completion_actions(self, sample_optical_task):
        """测试生成完成动作"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()
        entries = generator._generate_task_soe(sample_optical_task)

        # 检查包含完成动作
        action_types = [entry.action_type for entry in entries]
        assert SOEActionType.IMAGING_COMPLETE in action_types
        assert SOEActionType.SLEW_COMPLETE in action_types
        assert SOEActionType.SHUTTER_CLOSE in action_types

    def test_validate_guard_times_no_violation(self):
        """测试保护时间验证 - 无违规"""
        from core.telecommand.soe_generator import SOEGenerator, SOEEntry, SOEActionType

        generator = SOEGenerator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        # 创建间隔足够的条目
        entries = [
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.PAYLOAD_POWER_ON,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={},
                guard_time_after=timedelta(seconds=10)
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=80),
                action_type=SOEActionType.SLEW_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=30),
                parameters={},
                guard_time_before=timedelta(seconds=5)
            )
        ]

        result = generator._validate_guard_times(entries)
        assert result is True

    def test_validate_guard_times_with_violation(self):
        """测试保护时间验证 - 有违规时抛出异常"""
        from core.telecommand.soe_generator import (
            SOEGenerator, SOEEntry, SOEActionType, GuardTimeViolationError
        )

        generator = SOEGenerator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        # 创建间隔不足的条目
        entries = [
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.PAYLOAD_POWER_ON,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={},
                guard_time_after=timedelta(seconds=10)
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=65),  # 间隔只有5秒，需要15秒
                action_type=SOEActionType.SLEW_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=30),
                parameters={},
                guard_time_before=timedelta(seconds=5)
            )
        ]

        with pytest.raises(GuardTimeViolationError) as exc_info:
            generator._validate_guard_times(entries)

        assert "guard time violations" in str(exc_info.value).lower()


# ==========================================
# 测试用例: GuardTimeRule
# ==========================================
class TestGuardTimeRule:
    """保护时间规则测试"""

    def test_guard_time_rule_creation(self):
        """测试GuardTimeRule可以正确创建"""
        from core.telecommand.soe_generator import GuardTimeRule, SOEActionType

        rule = GuardTimeRule(
            action_a=SOEActionType.SLEW_COMPLETE,
            action_b=SOEActionType.IMAGING_START,
            min_interval=timedelta(seconds=5),
            reason="Attitude stabilization time"
        )

        assert rule.action_a == SOEActionType.SLEW_COMPLETE
        assert rule.action_b == SOEActionType.IMAGING_START
        assert rule.min_interval == timedelta(seconds=5)
        assert rule.reason == "Attitude stabilization time"


# ==========================================
# 测试用例: GuardTimeValidator
# ==========================================
class TestGuardTimeValidator:
    """保护时间验证器测试"""

    @pytest.fixture
    def sample_soe_entries(self):
        """创建示例SOE条目"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        base_time = datetime(2026, 1, 1, 12, 0, 0)
        return [
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.PAYLOAD_POWER_ON,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=120),
                action_type=SOEActionType.SLEW_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=30),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=180),
                action_type=SOEActionType.IMAGING_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={}
            )
        ]

    def test_guard_time_validator_creation(self):
        """测试GuardTimeValidator可以正确创建"""
        from core.telecommand.soe_generator import GuardTimeValidator

        validator = GuardTimeValidator()
        assert validator is not None
        assert hasattr(validator, 'DEFAULT_RULES')

    def test_default_rules_structure(self):
        """测试默认规则结构正确"""
        from core.telecommand.soe_generator import GuardTimeValidator, GuardTimeRule

        validator = GuardTimeValidator()

        assert len(validator.DEFAULT_RULES) > 0
        for rule in validator.DEFAULT_RULES:
            assert isinstance(rule, GuardTimeRule)
            assert rule.min_interval > timedelta(0)

    def test_validate_soe_no_violations(self, sample_soe_entries):
        """测试验证无违规的SOE"""
        from core.telecommand.soe_generator import GuardTimeValidator

        validator = GuardTimeValidator()
        violations = validator.validate_soe(sample_soe_entries)

        assert violations == []

    def test_validate_soe_with_violations(self):
        """测试验证有违规的SOE"""
        from core.telecommand.soe_generator import (
            GuardTimeValidator, SOEEntry, SOEActionType
        )

        validator = GuardTimeValidator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        # 创建违反SLEW_COMPLETE -> IMAGING_START规则的条目
        entries = [
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.SLEW_COMPLETE,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=5),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=2),  # 间隔只有2秒，需要5秒
                action_type=SOEActionType.IMAGING_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={}
            )
        ]

        violations = validator.validate_soe(entries)

        assert len(violations) > 0
        assert 'entry_a' in violations[0]
        assert 'entry_b' in violations[0]
        assert 'required_interval' in violations[0]
        assert 'actual_interval' in violations[0]

    def test_validate_soe_by_satellite(self):
        """测试按卫星分组验证SOE"""
        from core.telecommand.soe_generator import (
            GuardTimeValidator, SOEEntry, SOEActionType
        )

        validator = GuardTimeValidator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        # 创建两个卫星的条目
        entries = [
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.SLEW_COMPLETE,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=5),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=10),
                action_type=SOEActionType.IMAGING_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.SLEW_COMPLETE,
                satellite_id="SAT-002",
                task_id="TASK-002",
                duration=timedelta(seconds=5),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=2),  # 违规
                action_type=SOEActionType.IMAGING_START,
                satellite_id="SAT-002",
                task_id="TASK-002",
                duration=timedelta(seconds=60),
                parameters={}
            )
        ]

        violations = validator.validate_soe(entries)

        # 只有SAT-002有违规
        assert len(violations) == 1
        assert violations[0]['entry_a'].satellite_id == "SAT-002"

    def test_auto_fix_no_violations(self, sample_soe_entries):
        """测试无违规时的自动修复"""
        from core.telecommand.soe_generator import GuardTimeValidator

        validator = GuardTimeValidator()
        fixed_entries = validator.auto_fix(sample_soe_entries)

        # 无违规时返回原列表的副本
        assert len(fixed_entries) == len(sample_soe_entries)
        for i, entry in enumerate(fixed_entries):
            assert entry.timestamp == sample_soe_entries[i].timestamp

    def test_auto_fix_with_violations(self):
        """测试有违规时的自动修复"""
        from core.telecommand.soe_generator import (
            GuardTimeValidator, SOEEntry, SOEActionType
        )

        validator = GuardTimeValidator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        # 创建违反规则的条目
        entries = [
            SOEEntry(
                timestamp=base_time,
                action_type=SOEActionType.SLEW_COMPLETE,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=5),
                parameters={}
            ),
            SOEEntry(
                timestamp=base_time + timedelta(seconds=2),  # 间隔只有2秒，需要5秒
                action_type=SOEActionType.IMAGING_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={}
            )
        ]

        fixed_entries = validator.auto_fix(entries)

        # 修复后应该没有违规
        violations = validator.validate_soe(fixed_entries)
        assert len(violations) == 0

        # 第二个条目的时间应该被推迟
        assert fixed_entries[1].timestamp > entries[1].timestamp


# ==========================================
# 测试用例: GuardTimeViolationError
# ==========================================
class TestGuardTimeViolationError:
    """保护时间违规异常测试"""

    def test_error_creation(self):
        """测试异常可以正确创建"""
        from core.telecommand.soe_generator import GuardTimeViolationError

        violations = [{'entry_a': 'a', 'entry_b': 'b'}]
        error = GuardTimeViolationError("Test error", violations)

        assert str(error) == "Test error"
        assert error.violations == violations

    def test_error_inheritance(self):
        """测试异常继承自Exception"""
        from core.telecommand.soe_generator import GuardTimeViolationError

        assert issubclass(GuardTimeViolationError, Exception)


# ==========================================
# 边缘情况测试
# ==========================================
class TestEdgeCases:
    """边缘情况测试"""

    def test_soe_entry_with_unicode_parameters(self):
        """测试SOEEntry支持Unicode参数"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        entry = SOEEntry(
            timestamp=datetime.now(),
            action_type=SOEActionType.IMAGING_START,
            satellite_id="SAT-001",
            task_id="TASK-001",
            duration=timedelta(seconds=60),
            parameters={"description": "测试中文", "emoji": "🛰️"}
        )

        assert entry.parameters["description"] == "测试中文"
        assert entry.parameters["emoji"] == "🛰️"

    def test_soe_generator_with_multiple_tasks(self):
        """测试多个任务的SOE生成"""
        from core.telecommand.soe_generator import SOEGenerator
        from scheduler.base_scheduler import ScheduleResult, ScheduledTask

        generator = SOEGenerator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        schedule = ScheduleResult(
            scheduled_tasks=[
                ScheduledTask(
                    task_id=f"TASK-{i:03d}",
                    satellite_id=f"SAT-{i % 2 + 1:03d}",
                    target_id=f"TGT-{i:03d}",
                    imaging_start=base_time + timedelta(minutes=i * 10),
                    imaging_end=base_time + timedelta(minutes=i * 10 + 1),
                    imaging_mode="optical_high",
                    slew_angle=10.0
                )
                for i in range(5)
            ],
            unscheduled_tasks={},
            makespan=600.0,
            computation_time=1.0,
            iterations=10
        )

        soe_entries = generator.generate_soe(schedule)

        # 应该为每个任务生成条目
        assert len(soe_entries) > 0

        # 检查按时间排序
        for i in range(len(soe_entries) - 1):
            assert soe_entries[i].timestamp <= soe_entries[i + 1].timestamp

    def test_soe_entry_with_zero_duration(self):
        """测试持续时间为0的SOEEntry"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        entry = SOEEntry(
            timestamp=datetime.now(),
            action_type=SOEActionType.SHUTTER_OPEN,
            satellite_id="SAT-001",
            task_id="TASK-001",
            duration=timedelta(seconds=0),
            parameters={}
        )

        assert entry.duration == timedelta(seconds=0)

    def test_soe_entry_with_negative_offset(self):
        """测试负偏移时间的SOEEntry"""
        from core.telecommand.soe_generator import SOEEntry, SOEActionType

        base_time = datetime(2026, 1, 1, 12, 0, 0)
        entry = SOEEntry(
            timestamp=base_time - timedelta(seconds=300),
            action_type=SOEActionType.PAYLOAD_POWER_ON,
            satellite_id="SAT-001",
            task_id="TASK-001",
            duration=timedelta(seconds=60),
            parameters={}
        )

        assert entry.timestamp == base_time - timedelta(seconds=300)

    def test_validate_guard_times_empty_list(self):
        """测试验证空SOE列表"""
        from core.telecommand.soe_generator import SOEGenerator

        generator = SOEGenerator()
        result = generator._validate_guard_times([])
        assert result is True

    def test_validate_guard_times_single_entry(self):
        """测试验证单个SOE条目"""
        from core.telecommand.soe_generator import SOEGenerator, SOEEntry, SOEActionType

        generator = SOEGenerator()
        entries = [
            SOEEntry(
                timestamp=datetime.now(),
                action_type=SOEActionType.IMAGING_START,
                satellite_id="SAT-001",
                task_id="TASK-001",
                duration=timedelta(seconds=60),
                parameters={}
            )
        ]

        result = generator._validate_guard_times(entries)
        assert result is True

    def test_guard_validator_empty_soe(self):
        """测试验证器处理空SOE"""
        from core.telecommand.soe_generator import GuardTimeValidator

        validator = GuardTimeValidator()
        violations = validator.validate_soe([])
        assert violations == []

        fixed = validator.auto_fix([])
        assert fixed == []

    def test_generate_soe_with_validate_true(self):
        """测试generate_soe的validate=True参数"""
        from core.telecommand.soe_generator import (
            SOEGenerator, SOEEntry, SOEActionType, GuardTimeViolationError
        )
        from scheduler.base_scheduler import ScheduleResult, ScheduledTask

        generator = SOEGenerator()
        base_time = datetime(2026, 1, 1, 12, 0, 0)

        # 创建会产生保护时间违规的调度
        schedule = ScheduleResult(
            scheduled_tasks=[
                ScheduledTask(
                    task_id="TASK-001",
                    satellite_id="SAT-001",
                    target_id="TGT-001",
                    imaging_start=base_time,
                    imaging_end=base_time + timedelta(seconds=60),
                    imaging_mode="optical_high",
                    slew_angle=15.5
                )
            ],
            unscheduled_tasks={},
            makespan=60.0,
            computation_time=1.0,
            iterations=10
        )

        # validate=False时不抛出异常
        soe_entries = generator.generate_soe(schedule, validate=False)
        assert len(soe_entries) > 0

    def test_determine_task_type_with_object(self):
        """测试使用对象类型任务确定任务类型"""
        from core.telecommand.soe_generator import SOEGenerator

        generator = SOEGenerator()

        # 创建模拟任务对象
        class MockTask:
            def __init__(self, imaging_mode, imaging_type=None):
                self.imaging_mode = imaging_mode
                self.imaging_type = imaging_type

        optical_task = MockTask("optical_high", "optical")
        assert generator._determine_task_type(optical_task) == "optical_imaging"

        sar_task = MockTask("stripmap", "sar")
        assert generator._determine_task_type(sar_task) == "sar_imaging"

        # 测试非字典、非对象类型
        class UnknownTask:
            pass

        unknown = UnknownTask()
        assert generator._determine_task_type(unknown) == ""

    def test_generate_completion_actions_with_object(self):
        """测试使用对象类型任务生成完成动作"""
        from core.telecommand.soe_generator import SOEGenerator, SOEActionType

        generator = SOEGenerator()

        class MockTask:
            def __init__(self):
                self.task_id = "TASK-001"
                self.satellite_id = "SAT-001"
                self.target_id = "TGT-001"
                self.imaging_mode = "optical_high"
                self.imaging_end = datetime(2026, 1, 1, 12, 1, 0)
                self.slew_angle = 15.5

        task = MockTask()
        entries = generator._generate_completion_actions(task)

        # 检查生成了完成动作
        action_types = [entry.action_type for entry in entries]
        assert SOEActionType.IMAGING_COMPLETE in action_types
        assert SOEActionType.SHUTTER_CLOSE in action_types
        assert SOEActionType.SLEW_COMPLETE in action_types
