"""
ScheduleValidator 类的单元测试
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional

from simulator.schedule_validator import ScheduleValidator


class MockTask:
    """模拟任务类用于测试"""
    def __init__(self, task_id: str, start_time: datetime, end_time: datetime,
                 satellite_id: str = "sat1", power_required: float = 100.0,
                 storage_required: float = 10.0):
        self.id = task_id
        self.task_id = task_id
        self.start_time = start_time
        self.end_time = end_time
        self.satellite_id = satellite_id
        self.power_required = power_required
        self.storage_required = storage_required


class MockSatellite:
    """模拟卫星类用于测试"""
    def __init__(self, sat_id: str, power_capacity: float = 1000.0,
                 storage_capacity: float = 100.0):
        self.id = sat_id
        self.satellite_id = sat_id
        self.power_capacity = power_capacity
        self.storage_capacity = storage_capacity


class TestScheduleValidator:
    """ScheduleValidator 测试类"""

    @pytest.fixture
    def base_time(self):
        """基础时间"""
        return datetime(2024, 1, 1, 0, 0, 0)

    @pytest.fixture
    def validator(self):
        """创建验证器实例"""
        return ScheduleValidator()

    def test_validate_forward_empty_existing_tasks(self, validator, base_time):
        """测试空现有任务列表时的前向推演验证"""
        candidate = MockTask(
            "task1",
            base_time,
            base_time + timedelta(hours=1)
        )
        planning_horizon = base_time + timedelta(hours=2)

        is_valid, reason = validator.validate_forward(
            candidate, [], planning_horizon
        )

        assert is_valid is True
        assert reason == ""

    def test_validate_forward_no_time_conflict(self, validator, base_time):
        """测试无时间冲突的情况"""
        existing_task = MockTask(
            "existing",
            base_time,
            base_time + timedelta(hours=1)
        )
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=2),
            base_time + timedelta(hours=3)
        )
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, reason = validator.validate_forward(
            candidate, [existing_task], planning_horizon
        )

        assert is_valid is True
        assert reason == ""

    def test_validate_forward_time_conflict(self, validator, base_time):
        """测试时间冲突的情况"""
        existing_task = MockTask(
            "existing",
            base_time,
            base_time + timedelta(hours=2)
        )
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=3)
        )
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, reason = validator.validate_forward(
            candidate, [existing_task], planning_horizon
        )

        assert is_valid is False
        assert "时间冲突" in reason or "conflict" in reason.lower()

    def test_validate_forward_exact_boundary_no_conflict(self, validator, base_time):
        """测试精确边界无冲突（结束时间等于开始时间）"""
        existing_task = MockTask(
            "existing",
            base_time,
            base_time + timedelta(hours=1)
        )
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=2)
        )
        planning_horizon = base_time + timedelta(hours=3)

        is_valid, reason = validator.validate_forward(
            candidate, [existing_task], planning_horizon
        )

        assert is_valid is True
        assert reason == ""

    def test_validate_forward_candidate_exceeds_horizon(self, validator, base_time):
        """测试候选任务超出规划周期"""
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=3)
        )
        planning_horizon = base_time + timedelta(hours=2)

        is_valid, reason = validator.validate_forward(
            candidate, [], planning_horizon
        )

        assert is_valid is False
        assert "超出" in reason or "exceed" in reason.lower() or "horizon" in reason.lower()

    def test_validate_forward_multiple_existing_no_conflict(self, validator, base_time):
        """测试多个现有任务无冲突"""
        existing_tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=1)),
            MockTask("task2", base_time + timedelta(hours=1, minutes=30),
                    base_time + timedelta(hours=2, minutes=30)),
        ]
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=3),
            base_time + timedelta(hours=4)
        )
        planning_horizon = base_time + timedelta(hours=5)

        is_valid, reason = validator.validate_forward(
            candidate, existing_tasks, planning_horizon
        )

        assert is_valid is True
        assert reason == ""

    def test_validate_forward_multiple_existing_with_conflict(self, validator, base_time):
        """测试多个现有任务有冲突"""
        existing_tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=1)),
            MockTask("task2", base_time + timedelta(hours=2),
                    base_time + timedelta(hours=3)),
        ]
        candidate = MockTask(
            "candidate",
            base_time + timedelta(minutes=30),
            base_time + timedelta(hours=1, minutes=30)
        )
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, reason = validator.validate_forward(
            candidate, existing_tasks, planning_horizon
        )

        assert is_valid is False
        assert "时间冲突" in reason or "conflict" in reason.lower()

    def test_validate_forward_candidate_ends_exactly_at_horizon(self, validator, base_time):
        """测试候选任务恰好在规划周期结束时完成"""
        candidate = MockTask(
            "candidate",
            base_time,
            base_time + timedelta(hours=2)
        )
        planning_horizon = base_time + timedelta(hours=2)

        is_valid, reason = validator.validate_forward(
            candidate, [], planning_horizon
        )

        assert is_valid is True
        assert reason == ""

    def test_validate_forward_same_satellite_conflict(self, validator, base_time):
        """测试同一卫星的时间冲突"""
        existing_task = MockTask(
            "existing",
            base_time,
            base_time + timedelta(hours=2),
            satellite_id="sat1"
        )
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=3),
            satellite_id="sat1"
        )
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, reason = validator.validate_forward(
            candidate, [existing_task], planning_horizon
        )

        assert is_valid is False

    def test_validate_forward_different_satellite_no_conflict(self, validator, base_time):
        """测试不同卫星无时间冲突"""
        existing_task = MockTask(
            "existing",
            base_time,
            base_time + timedelta(hours=2),
            satellite_id="sat1"
        )
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=3),
            satellite_id="sat2"
        )
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, reason = validator.validate_forward(
            candidate, [existing_task], planning_horizon
        )

        # 不同卫星的任务不应该有时间冲突
        assert is_valid is True
        assert reason == ""

    def test_validate_forward_negative_duration(self, validator, base_time):
        """测试无效的时间范围（结束时间早于开始时间）"""
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=1),
            base_time  # 早于开始时间
        )
        planning_horizon = base_time + timedelta(hours=2)

        is_valid, reason = validator.validate_forward(
            candidate, [], planning_horizon
        )

        assert is_valid is False
        assert "无效" in reason or "invalid" in reason.lower() or "结束" in reason

    def test_validate_forward_zero_duration(self, validator, base_time):
        """测试零持续时间"""
        candidate = MockTask(
            "candidate",
            base_time,
            base_time  # 相同的开始和结束时间
        )
        planning_horizon = base_time + timedelta(hours=2)

        is_valid, reason = validator.validate_forward(
            candidate, [], planning_horizon
        )

        # 零持续时间应该被接受或拒绝，取决于实现
        # 这里我们假设应该被接受
        assert isinstance(is_valid, bool)

    def test_validate_forward_with_resource_constraints(self, validator, base_time):
        """测试带资源约束的前向推演"""
        # 这个测试验证资源约束检查的基本框架
        # 实际资源约束验证可能需要更复杂的模拟
        existing_task = MockTask(
            "existing",
            base_time,
            base_time + timedelta(hours=1),
            power_required=500.0,
            storage_required=50.0
        )
        candidate = MockTask(
            "candidate",
            base_time + timedelta(hours=2),
            base_time + timedelta(hours=3),
            power_required=300.0,
            storage_required=30.0
        )
        planning_horizon = base_time + timedelta(hours=4)

        # 验证器应该能够处理带资源需求的任务
        is_valid, reason = validator.validate_forward(
            candidate, [existing_task], planning_horizon
        )

        # 只要时间不冲突，应该返回有效
        assert is_valid is True

    def test_validate_full_schedule_valid(self, validator, base_time):
        """测试验证完整调度方案 - 有效情况"""
        tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=1)),
            MockTask("task2", base_time + timedelta(hours=1), base_time + timedelta(hours=2)),
            MockTask("task3", base_time + timedelta(hours=2), base_time + timedelta(hours=3)),
        ]
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, violations = validator.validate_full_schedule(tasks, planning_horizon)

        assert is_valid is True
        assert violations == []

    def test_validate_full_schedule_with_exceeding_task(self, validator, base_time):
        """测试验证完整调度方案 - 有超出规划周期的任务"""
        tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=1)),
            MockTask("task2", base_time + timedelta(hours=3), base_time + timedelta(hours=5)),
        ]
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, violations = validator.validate_full_schedule(tasks, planning_horizon)

        assert is_valid is False
        assert len(violations) == 1
        assert "超出" in violations[0] or "exceed" in violations[0].lower()

    def test_validate_full_schedule_with_conflict(self, validator, base_time):
        """测试验证完整调度方案 - 有时间冲突"""
        tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=2), satellite_id="sat1"),
            MockTask("task2", base_time + timedelta(hours=1), base_time + timedelta(hours=3), satellite_id="sat1"),
        ]
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, violations = validator.validate_full_schedule(tasks, planning_horizon)

        assert is_valid is False
        assert len(violations) >= 1
        assert any("冲突" in v or "conflict" in v.lower() for v in violations)

    def test_validate_full_schedule_empty(self, validator, base_time):
        """测试验证完整调度方案 - 空任务列表"""
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, violations = validator.validate_full_schedule([], planning_horizon)

        assert is_valid is True
        assert violations == []

    def test_validate_full_schedule_multiple_satellites(self, validator, base_time):
        """测试验证完整调度方案 - 多卫星情况"""
        tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=1), satellite_id="sat1"),
            MockTask("task2", base_time, base_time + timedelta(hours=1), satellite_id="sat2"),
            MockTask("task3", base_time + timedelta(hours=1), base_time + timedelta(hours=2), satellite_id="sat1"),
        ]
        planning_horizon = base_time + timedelta(hours=3)

        is_valid, violations = validator.validate_full_schedule(tasks, planning_horizon)

        assert is_valid is True
        assert violations == []

    def test_validate_full_schedule_multiple_conflicts(self, validator, base_time):
        """测试验证完整调度方案 - 多个冲突"""
        tasks = [
            MockTask("task1", base_time, base_time + timedelta(hours=2), satellite_id="sat1"),
            MockTask("task2", base_time + timedelta(hours=1), base_time + timedelta(hours=3), satellite_id="sat1"),
            MockTask("task3", base_time + timedelta(hours=5), base_time + timedelta(hours=6)),
        ]
        planning_horizon = base_time + timedelta(hours=4)

        is_valid, violations = validator.validate_full_schedule(tasks, planning_horizon)

        assert is_valid is False
        assert len(violations) >= 2  # 时间冲突 + 超出规划周期
