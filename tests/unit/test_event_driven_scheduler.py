"""
事件驱动调度器测试

TDD测试文件 - 第19章设计实现
"""

import pytest
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock

from scheduler.base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailure, TaskFailureReason
from core.dynamic_scheduler.event_driven_scheduler import (
    EventType,
    ScheduleEvent,
    DisruptionImpact,
    EventDrivenScheduler
)


class TestEventType:
    """测试事件类型枚举"""

    def test_event_type_values(self):
        """测试事件类型枚举值"""
        assert EventType.NEW_URGENT_TASK.value == "new_urgent_task"
        assert EventType.TASK_CANCELLED.value == "task_cancelled"
        assert EventType.SATELLITE_FAILURE.value == "satellite_failure"
        assert EventType.RESOURCE_DEGRADATION.value == "resource_degradation"
        assert EventType.ROLLING_HORIZON_TRIGGER.value == "rolling_horizon_trigger"

    def test_event_type_is_enum(self):
        """测试EventType是Enum类型"""
        assert issubclass(EventType, Enum)


class TestScheduleEvent:
    """测试调度事件数据类"""

    def test_create_schedule_event(self):
        """测试创建调度事件"""
        timestamp = datetime.now(timezone.utc)
        event = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=timestamp,
            priority=9,
            description="紧急灾害响应任务",
            payload={"task_id": "URGENT-001", "location": [116.4, 39.9]}
        )

        assert event.event_type == EventType.NEW_URGENT_TASK
        assert event.timestamp == timestamp
        assert event.priority == 9
        assert event.description == "紧急灾害响应任务"
        assert event.payload["task_id"] == "URGENT-001"
        assert event.affected_satellites is None
        assert event.affected_tasks is None

    def test_create_schedule_event_with_affected_entities(self):
        """测试创建带受影响实体的事件"""
        timestamp = datetime.now(timezone.utc)
        event = ScheduleEvent(
            event_type=EventType.SATELLITE_FAILURE,
            timestamp=timestamp,
            priority=10,
            description="卫星故障",
            payload={"satellite_id": "SAT-01", "failure_type": "power"},
            affected_satellites=["SAT-01"],
            affected_tasks=["TASK-001", "TASK-002"]
        )

        assert event.affected_satellites == ["SAT-01"]
        assert event.affected_tasks == ["TASK-001", "TASK-002"]

    def test_priority_range_validation(self):
        """测试优先级范围"""
        timestamp = datetime.now(timezone.utc)

        # 优先级1（最低）
        event_low = ScheduleEvent(
            event_type=EventType.ROLLING_HORIZON_TRIGGER,
            timestamp=timestamp,
            priority=1,
            description="滚动触发",
            payload={}
        )
        assert event_low.priority == 1

        # 优先级10（最高）
        event_high = ScheduleEvent(
            event_type=EventType.SATELLITE_FAILURE,
            timestamp=timestamp,
            priority=10,
            description="卫星故障",
            payload={}
        )
        assert event_high.priority == 10

    def test_event_equality(self):
        """测试事件相等性比较"""
        timestamp = datetime.now(timezone.utc)
        event1 = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=timestamp,
            priority=9,
            description="紧急任务",
            payload={"task_id": "T001"}
        )
        event2 = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=timestamp,
            priority=9,
            description="紧急任务",
            payload={"task_id": "T001"}
        )

        assert event1 == event2


class TestDisruptionImpact:
    """测试扰动影响数据类"""

    def test_create_disruption_impact(self):
        """测试创建扰动影响"""
        impact = DisruptionImpact(
            severity="moderate",
            affected_satellites=["SAT-01", "SAT-02"],
            affected_tasks=["TASK-001", "TASK-002", "TASK-003"],
            affected_ratio=0.25
        )

        assert impact.severity == "moderate"
        assert impact.affected_satellites == ["SAT-01", "SAT-02"]
        assert impact.affected_tasks == ["TASK-001", "TASK-002", "TASK-003"]
        assert impact.affected_ratio == 0.25

    def test_severity_values(self):
        """测试严重程度值"""
        minor_impact = DisruptionImpact(
            severity="minor",
            affected_satellites=[],
            affected_tasks=[],
            affected_ratio=0.0
        )
        assert minor_impact.severity == "minor"

        major_impact = DisruptionImpact(
            severity="major",
            affected_satellites=["SAT-01", "SAT-02", "SAT-03", "SAT-04"],
            affected_tasks=["TASK-001"] * 50,
            affected_ratio=0.5
        )
        assert major_impact.severity == "major"


class TestEventDrivenSchedulerInit:
    """测试事件驱动调度器初始化"""

    def test_init_with_base_scheduler(self):
        """测试使用基础调度器初始化"""
        mock_scheduler = Mock(spec=BaseScheduler)
        scheduler = EventDrivenScheduler(base_scheduler=mock_scheduler)

        assert scheduler.base_scheduler == mock_scheduler
        assert scheduler.current_plan is None
        assert scheduler.event_queue == []
        assert scheduler.plan_history == []

    def test_init_without_scheduler_raises_error(self):
        """测试没有提供调度器时应该报错"""
        with pytest.raises(TypeError):
            EventDrivenScheduler()


class TestSubmitEvent:
    """测试提交事件"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_submit_event_adds_to_queue(self):
        """测试提交事件添加到队列"""
        event = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=datetime.now(timezone.utc),
            priority=5,
            description="普通任务",
            payload={}
        )

        result = self.event_scheduler.submit_event(event)

        assert len(self.event_scheduler.event_queue) == 1
        assert self.event_scheduler.event_queue[0] == event
        assert result is False  # 低优先级不立即触发

    def test_submit_high_priority_event_triggers_reschedule(self):
        """测试提交高优先级事件触发重调度"""
        event = ScheduleEvent(
            event_type=EventType.SATELLITE_FAILURE,
            timestamp=datetime.now(timezone.utc),
            priority=10,
            description="卫星故障",
            payload={"satellite_id": "SAT-01"}
        )

        result = self.event_scheduler.submit_event(event)

        assert result is True  # 高优先级立即触发

    def test_event_queue_sorted_by_priority(self):
        """测试事件队列按优先级排序"""
        event_low = ScheduleEvent(
            event_type=EventType.ROLLING_HORIZON_TRIGGER,
            timestamp=datetime.now(timezone.utc),
            priority=3,
            description="低优先级",
            payload={}
        )
        event_high = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=datetime.now(timezone.utc) + timedelta(minutes=1),
            priority=9,
            description="高优先级",
            payload={}
        )
        event_medium = ScheduleEvent(
            event_type=EventType.TASK_CANCELLED,
            timestamp=datetime.now(timezone.utc) + timedelta(minutes=2),
            priority=6,
            description="中优先级",
            payload={}
        )

        self.event_scheduler.submit_event(event_low)
        self.event_scheduler.submit_event(event_medium)
        self.event_scheduler.submit_event(event_high)

        # 应该按优先级降序排列
        assert self.event_scheduler.event_queue[0].priority == 9
        assert self.event_scheduler.event_queue[1].priority == 6
        assert self.event_scheduler.event_queue[2].priority == 3

    def test_submit_multiple_events_same_priority_sorted_by_timestamp(self):
        """测试相同优先级事件按时间戳排序"""
        base_time = datetime.now(timezone.utc)
        event1 = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=base_time,
            priority=8,
            description="事件1",
            payload={}
        )
        event2 = ScheduleEvent(
            event_type=EventType.TASK_CANCELLED,
            timestamp=base_time + timedelta(minutes=5),
            priority=8,
            description="事件2",
            payload={}
        )

        self.event_scheduler.submit_event(event2)
        self.event_scheduler.submit_event(event1)

        # 相同优先级，先发生的在前
        assert self.event_scheduler.event_queue[0] == event1
        assert self.event_scheduler.event_queue[1] == event2


class TestAnalyzeImpact:
    """测试影响分析"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_analyze_minor_impact(self):
        """测试分析轻微影响"""
        # 创建包含10个任务的计划
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 只影响1个任务（10% = 0.1，边界值）
        events = [
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=datetime.now(timezone.utc),
                priority=5,
                description="任务取消",
                payload={},
                affected_satellites=["SAT-01"],
                affected_tasks=["TASK-000"]
            )
        ]

        impact = self.event_scheduler._analyze_impact(current_plan, events)

        assert impact.severity == "minor"
        assert impact.affected_tasks == ["TASK-000"]
        assert impact.affected_satellites == ["SAT-01"]
        assert impact.affected_ratio == 0.1

    def test_analyze_moderate_impact(self):
        """测试分析中等影响"""
        # 创建包含10个任务的计划
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id=f"SAT-0{(i % 3) + 1}",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 影响2个任务（20%，3颗卫星以内）
        events = [
            ScheduleEvent(
                event_type=EventType.SATELLITE_FAILURE,
                timestamp=datetime.now(timezone.utc),
                priority=8,
                description="卫星故障",
                payload={},
                affected_satellites=["SAT-01", "SAT-02"],
                affected_tasks=["TASK-001", "TASK-004"]
            )
        ]

        impact = self.event_scheduler._analyze_impact(current_plan, events)

        assert impact.severity == "moderate"
        assert set(impact.affected_satellites) == {"SAT-01", "SAT-02"}
        assert impact.affected_ratio == 0.2

    def test_analyze_major_impact(self):
        """测试分析严重影响"""
        # 创建包含10个任务的计划
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id=f"SAT-0{(i % 5) + 1}",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 影响5个任务（50%，超过30%）
        events = [
            ScheduleEvent(
                event_type=EventType.SATELLITE_FAILURE,
                timestamp=datetime.now(timezone.utc),
                priority=10,
                description="多卫星故障",
                payload={},
                affected_satellites=["SAT-01", "SAT-02", "SAT-03", "SAT-04"],
                affected_tasks=[f"TASK-{i:03d}" for i in range(5)]
            )
        ]

        impact = self.event_scheduler._analyze_impact(current_plan, events)

        assert impact.severity == "major"
        assert len(impact.affected_satellites) == 4
        assert impact.affected_ratio == 0.5

    def test_analyze_impact_with_multiple_events(self):
        """测试分析多个事件的累积影响"""
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        events = [
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=datetime.now(timezone.utc),
                priority=5,
                description="取消任务1",
                payload={},
                affected_tasks=["TASK-001"]
            ),
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=datetime.now(timezone.utc),
                priority=5,
                description="取消任务2",
                payload={},
                affected_tasks=["TASK-002"]
            )
        ]

        impact = self.event_scheduler._analyze_impact(current_plan, events)

        assert set(impact.affected_tasks) == {"TASK-001", "TASK-002"}
        assert impact.affected_ratio == 0.2

    def test_analyze_impact_empty_plan(self):
        """测试分析空计划的影响"""
        current_plan = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        events = [
            ScheduleEvent(
                event_type=EventType.SATELLITE_FAILURE,
                timestamp=datetime.now(timezone.utc),
                priority=10,
                description="卫星故障",
                payload={},
                affected_satellites=["SAT-01"],
                affected_tasks=["TASK-001"]
            )
        ]

        impact = self.event_scheduler._analyze_impact(current_plan, events)

        # 空计划但有受影响任务时，比例为1.0（1/max(0,1)=1），应该是major
        assert impact.severity == "major"


class TestRemoveAffectedTasks:
    """测试移除受影响任务"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_remove_affected_tasks(self):
        """测试移除受影响任务"""
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc) + timedelta(minutes=i*10),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=i*10+5),
                imaging_mode="push_broom"
            )
            for i in range(5)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        result = self.event_scheduler._remove_affected_tasks(
            current_plan,
            ["TASK-001", "TASK-003"]
        )

        assert len(result.scheduled_tasks) == 3
        task_ids = [t.task_id for t in result.scheduled_tasks]
        assert "TASK-001" not in task_ids
        assert "TASK-003" not in task_ids
        assert "TASK-000" in task_ids
        assert "TASK-002" in task_ids
        assert "TASK-004" in task_ids

    def test_remove_nonexistent_tasks(self):
        """测试移除不存在的任务"""
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        result = self.event_scheduler._remove_affected_tasks(
            current_plan,
            ["NONEXISTENT"]
        )

        # 应该保持不变
        assert len(result.scheduled_tasks) == 1
        assert result.scheduled_tasks[0].task_id == "TASK-001"

    def test_remove_all_tasks(self):
        """测试移除所有任务"""
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=datetime.now(timezone.utc),
                imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        result = self.event_scheduler._remove_affected_tasks(
            current_plan,
            ["TASK-001"]
        )

        assert len(result.scheduled_tasks) == 0


class TestGetFrozenTasks:
    """测试获取冻结任务"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_get_frozen_tasks(self):
        """测试获取冻结任务"""
        now = datetime.now(timezone.utc)
        freeze_until = now + timedelta(minutes=5)

        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",  # 已开始，应该冻结
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now - timedelta(minutes=2),
                imaging_end=now + timedelta(minutes=3),
                imaging_mode="push_broom"
            ),
            ScheduledTask(
                task_id="TASK-002",  # 在冻结期内开始，应该冻结
                satellite_id="SAT-01",
                target_id="TGT-002",
                imaging_start=now + timedelta(minutes=3),
                imaging_end=now + timedelta(minutes=8),
                imaging_mode="push_broom"
            ),
            ScheduledTask(
                task_id="TASK-003",  # 在冻结期后开始，不应该冻结
                satellite_id="SAT-01",
                target_id="TGT-003",
                imaging_start=now + timedelta(minutes=10),
                imaging_end=now + timedelta(minutes=15),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        frozen_tasks = self.event_scheduler._get_frozen_tasks(current_plan, freeze_until)

        assert "TASK-001" in frozen_tasks
        assert "TASK-002" in frozen_tasks
        assert "TASK-003" not in frozen_tasks

    def test_get_frozen_tasks_empty_plan(self):
        """测试空计划的冻结任务"""
        current_plan = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        frozen_tasks = self.event_scheduler._get_frozen_tasks(
            current_plan,
            datetime.now(timezone.utc)
        )

        assert frozen_tasks == set()


class TestInsertTaskGreedily:
    """测试贪心插入任务"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_insert_task_into_empty_plan(self):
        """测试向空计划插入任务"""
        current_plan = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TGT-001",
            imaging_start=datetime.now(timezone.utc),
            imaging_end=datetime.now(timezone.utc) + timedelta(minutes=5),
            imaging_mode="push_broom"
        )

        result = self.event_scheduler._insert_task_greedily(current_plan, task)

        assert result is True
        assert len(current_plan.scheduled_tasks) == 1
        assert current_plan.scheduled_tasks[0].task_id == "TASK-001"

    def test_insert_task_no_conflict(self):
        """测试无冲突插入任务"""
        now = datetime.now(timezone.utc)
        existing_task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TGT-001",
            imaging_start=now,
            imaging_end=now + timedelta(minutes=5),
            imaging_mode="push_broom"
        )
        current_plan = ScheduleResult(
            scheduled_tasks=[existing_task],
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        new_task = ScheduledTask(
            task_id="TASK-002",
            satellite_id="SAT-01",
            target_id="TGT-002",
            imaging_start=now + timedelta(minutes=10),
            imaging_end=now + timedelta(minutes=15),
            imaging_mode="push_broom"
        )

        result = self.event_scheduler._insert_task_greedily(current_plan, new_task)

        assert result is True
        assert len(current_plan.scheduled_tasks) == 2

    def test_insert_task_with_conflict(self):
        """测试有冲突时插入失败"""
        now = datetime.now(timezone.utc)
        existing_task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TGT-001",
            imaging_start=now,
            imaging_end=now + timedelta(minutes=10),
            imaging_mode="push_broom"
        )
        current_plan = ScheduleResult(
            scheduled_tasks=[existing_task],
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 时间冲突的新任务
        conflicting_task = ScheduledTask(
            task_id="TASK-002",
            satellite_id="SAT-01",
            target_id="TGT-002",
            imaging_start=now + timedelta(minutes=5),
            imaging_end=now + timedelta(minutes=15),
            imaging_mode="push_broom"
        )

        result = self.event_scheduler._insert_task_greedily(current_plan, conflicting_task)

        assert result is False
        assert len(current_plan.scheduled_tasks) == 1  # 没有插入

    def test_insert_task_different_satellite_no_conflict(self):
        """测试不同卫星的任务无冲突"""
        now = datetime.now(timezone.utc)
        existing_task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TGT-001",
            imaging_start=now,
            imaging_end=now + timedelta(minutes=10),
            imaging_mode="push_broom"
        )
        current_plan = ScheduleResult(
            scheduled_tasks=[existing_task],
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 不同卫星，时间重叠也没关系
        new_task = ScheduledTask(
            task_id="TASK-002",
            satellite_id="SAT-02",
            target_id="TGT-002",
            imaging_start=now + timedelta(minutes=5),
            imaging_end=now + timedelta(minutes=15),
            imaging_mode="push_broom"
        )

        result = self.event_scheduler._insert_task_greedily(current_plan, new_task)

        assert result is True
        assert len(current_plan.scheduled_tasks) == 2


class TestLocalRepair:
    """测试局部修复"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_local_repair_removes_and_reinserts_tasks(self):
        """测试局部修复移除并重新插入任务"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now,
                imaging_end=now + timedelta(minutes=5),
                imaging_mode="push_broom"
            ),
            ScheduledTask(
                task_id="TASK-002",
                satellite_id="SAT-01",
                target_id="TGT-002",
                imaging_start=now + timedelta(minutes=10),
                imaging_end=now + timedelta(minutes=15),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        impact = DisruptionImpact(
            severity="minor",
            affected_satellites=["SAT-01"],
            affected_tasks=["TASK-001"],
            affected_ratio=0.5
        )

        result = self.event_scheduler._local_repair(current_plan, impact)

        # 应该移除TASK-001，尝试重新插入
        assert isinstance(result, ScheduleResult)


class TestReschedule:
    """测试重调度接口"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_reschedule_minor_impact(self):
        """测试轻微影响使用局部修复"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now + timedelta(minutes=i*10),
                imaging_end=now + timedelta(minutes=i*10+5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        events = [
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=now,
                priority=5,
                description="取消1个任务",
                payload={},
                affected_satellites=["SAT-01"],
                affected_tasks=["TASK-001"]
            )
        ]

        result = self.event_scheduler.reschedule(current_plan, events)

        assert isinstance(result, ScheduleResult)

    def test_reschedule_moderate_impact(self):
        """测试中等影响使用滚动优化"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id=f"SAT-0{(i % 3) + 1}",
                target_id="TGT-001",
                imaging_start=now + timedelta(minutes=i*10),
                imaging_end=now + timedelta(minutes=i*10+5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        events = [
            ScheduleEvent(
                event_type=EventType.SATELLITE_FAILURE,
                timestamp=now,
                priority=8,
                description="卫星故障",
                payload={},
                affected_satellites=["SAT-01", "SAT-02"],
                affected_tasks=["TASK-000", "TASK-003"]
            )
        ]

        result = self.event_scheduler.reschedule(current_plan, events)

        assert isinstance(result, ScheduleResult)

    def test_reschedule_major_impact(self):
        """测试严重影响使用全局重调度"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id=f"SAT-0{(i % 5) + 1}",
                target_id="TGT-001",
                imaging_start=now + timedelta(minutes=i*10),
                imaging_end=now + timedelta(minutes=i*10+5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 模拟全局重调度返回的结果
        self.mock_scheduler.schedule.return_value = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.5,
            iterations=50
        )

        events = [
            ScheduleEvent(
                event_type=EventType.SATELLITE_FAILURE,
                timestamp=now,
                priority=10,
                description="多卫星故障",
                payload={},
                affected_satellites=["SAT-01", "SAT-02", "SAT-03", "SAT-04"],
                affected_tasks=[f"TASK-{i:03d}" for i in range(5)]
            )
        ]

        result = self.event_scheduler.reschedule(current_plan, events)

        assert isinstance(result, ScheduleResult)
        # 全局重调度应该调用base_scheduler.schedule
        self.mock_scheduler.schedule.assert_called_once()

    def test_reschedule_empty_events(self):
        """测试空事件列表"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now,
                imaging_end=now + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        result = self.event_scheduler.reschedule(current_plan, [])

        # 没有事件，应该返回原计划
        assert isinstance(result, ScheduleResult)


class TestRollingReoptimize:
    """测试滚动优化"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_rolling_reoptimize_with_frozen_tasks(self):
        """测试滚动优化考虑冻结任务"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",  # 冻结任务
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now - timedelta(minutes=2),
                imaging_end=now + timedelta(minutes=3),
                imaging_mode="push_broom"
            ),
            ScheduledTask(
                task_id="TASK-002",  # 窗内任务
                satellite_id="SAT-01",
                target_id="TGT-002",
                imaging_start=now + timedelta(minutes=30),
                imaging_end=now + timedelta(minutes=35),
                imaging_mode="push_broom"
            ),
            ScheduledTask(
                task_id="TASK-003",  # 窗外任务
                satellite_id="SAT-01",
                target_id="TGT-003",
                imaging_start=now + timedelta(hours=3),
                imaging_end=now + timedelta(hours=3, minutes=5),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        impact = DisruptionImpact(
            severity="moderate",
            affected_satellites=["SAT-01"],
            affected_tasks=["TASK-002"],
            affected_ratio=0.33
        )

        result = self.event_scheduler._rolling_reoptimize(current_plan, impact)

        assert isinstance(result, ScheduleResult)
        # 冻结任务应该保留
        task_ids = [t.task_id for t in result.scheduled_tasks]
        assert "TASK-001" in task_ids


class TestGlobalReschedule:
    """测试全局重调度"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_global_reschedule_calls_base_scheduler(self):
        """测试全局重调度调用基础调度器"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now + timedelta(minutes=i*10),
                imaging_end=now + timedelta(minutes=i*10+5),
                imaging_mode="push_broom"
            )
            for i in range(5)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 模拟全局重调度返回新计划
        new_plan = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=2.0,
            iterations=200
        )
        self.mock_scheduler.schedule.return_value = new_plan

        impact = DisruptionImpact(
            severity="major",
            affected_satellites=["SAT-01", "SAT-02"],
            affected_tasks=["TASK-000", "TASK-001", "TASK-002"],
            affected_ratio=0.6
        )

        result = self.event_scheduler._global_reschedule(current_plan, impact)

        assert result == new_plan
        self.mock_scheduler.schedule.assert_called_once()


class TestIntegration:
    """集成测试"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_full_workflow_submit_and_reschedule(self):
        """测试完整工作流程：提交事件并重调度"""
        now = datetime.now(timezone.utc)

        # 创建初始计划
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now + timedelta(minutes=i*15),
                imaging_end=now + timedelta(minutes=i*15+5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        initial_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )
        self.event_scheduler.current_plan = initial_plan

        # 提交一个高优先级事件
        event = ScheduleEvent(
            event_type=EventType.SATELLITE_FAILURE,
            timestamp=now,
            priority=10,
            description="SAT-01故障",
            payload={"satellite_id": "SAT-01"},
            affected_satellites=["SAT-01"],
            affected_tasks=[f"TASK-{i:03d}" for i in range(10)]
        )

        # 模拟全局重调度结果
        self.mock_scheduler.schedule.return_value = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=1.5,
            iterations=150
        )

        # 提交事件（高优先级应该触发重调度）
        triggered = self.event_scheduler.submit_event(event)

        assert triggered is True
        assert len(self.event_scheduler.event_queue) == 1

    def test_multiple_events_accumulation(self):
        """测试多个事件累积"""
        now = datetime.now(timezone.utc)

        events = [
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=now + timedelta(minutes=i),
                priority=5,
                description=f"取消任务{i}",
                payload={},
                affected_tasks=[f"TASK-{i:03d}"]
            )
            for i in range(3)
        ]

        for event in events:
            self.event_scheduler.submit_event(event)

        assert len(self.event_scheduler.event_queue) == 3

    def test_plan_history_tracking(self):
        """测试计划历史记录"""
        now = datetime.now(timezone.utc)

        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now,
                imaging_end=now + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
        ]
        plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 手动添加到历史记录
        self.event_scheduler.plan_history.append((now, plan, []))

        assert len(self.event_scheduler.plan_history) == 1
        assert self.event_scheduler.plan_history[0][1] == plan


class TestEdgeCases:
    """测试边界情况"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.mock_scheduler = Mock(spec=BaseScheduler)
        self.event_scheduler = EventDrivenScheduler(base_scheduler=self.mock_scheduler)

    def test_event_with_none_affected_entities(self):
        """测试受影响实体为None的事件"""
        event = ScheduleEvent(
            event_type=EventType.RESOURCE_DEGRADATION,
            timestamp=datetime.now(timezone.utc),
            priority=7,
            description="资源降级",
            payload={"resource_type": "power"},
            affected_satellites=None,
            affected_tasks=None
        )

        result = self.event_scheduler.submit_event(event)

        assert result is False  # 不是高优先级
        assert len(self.event_scheduler.event_queue) == 1

    def test_reschedule_with_no_scheduled_tasks(self):
        """测试对空计划进行重调度"""
        current_plan = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )

        events = [
            ScheduleEvent(
                event_type=EventType.NEW_URGENT_TASK,
                timestamp=datetime.now(timezone.utc),
                priority=9,
                description="紧急任务",
                payload={"task_id": "URGENT-001"},
                affected_tasks=["URGENT-001"]
            )
        ]

        # 空计划但有新任务时，会触发全局重调度
        self.mock_scheduler.schedule.return_value = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.5,
            iterations=50
        )

        result = self.event_scheduler.reschedule(current_plan, events)

        assert isinstance(result, ScheduleResult)

    def test_priority_boundary_values(self):
        """测试优先级边界值"""
        now = datetime.now(timezone.utc)

        # 优先级8应该触发
        event_priority_8 = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=now,
            priority=8,
            description="优先级8",
            payload={}
        )

        # 优先级7不应该触发
        event_priority_7 = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=now,
            priority=7,
            description="优先级7",
            payload={}
        )

        result_8 = self.event_scheduler.submit_event(event_priority_8)
        result_7 = self.event_scheduler.submit_event(event_priority_7)

        assert result_8 is True
        assert result_7 is False

    def test_analyze_impact_with_duplicate_tasks(self):
        """测试分析有重复任务的事件"""
        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id=f"TASK-{i:03d}",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now + timedelta(minutes=i*10),
                imaging_end=now + timedelta(minutes=i*10+5),
                imaging_mode="push_broom"
            )
            for i in range(10)
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 两个事件影响同一个任务
        events = [
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=now,
                priority=5,
                description="取消1",
                payload={},
                affected_tasks=["TASK-001"]
            ),
            ScheduleEvent(
                event_type=EventType.TASK_CANCELLED,
                timestamp=now + timedelta(minutes=1),
                priority=5,
                description="取消2",
                payload={},
                affected_tasks=["TASK-001", "TASK-002"]
            )
        ]

        impact = self.event_scheduler._analyze_impact(current_plan, events)

        # 应该去重
        assert len(impact.affected_tasks) == 2
        assert set(impact.affected_tasks) == {"TASK-001", "TASK-002"}


class TestAdditionalCoverage:
    """额外覆盖率测试"""

    def test_invalid_priority_raises_error(self):
        """测试无效优先级应该报错"""
        timestamp = datetime.now(timezone.utc)

        with pytest.raises(ValueError) as exc_info:
            ScheduleEvent(
                event_type=EventType.NEW_URGENT_TASK,
                timestamp=timestamp,
                priority=0,  # 无效
                description="无效优先级",
                payload={}
            )
        assert "Priority must be between 1 and 10" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            ScheduleEvent(
                event_type=EventType.NEW_URGENT_TASK,
                timestamp=timestamp,
                priority=11,  # 无效
                description="无效优先级",
                payload={}
            )
        assert "Priority must be between 1 and 10" in str(exc_info.value)

    def test_invalid_severity_raises_error(self):
        """测试无效严重程度应该报错"""
        with pytest.raises(ValueError) as exc_info:
            DisruptionImpact(
                severity="invalid",
                affected_satellites=[],
                affected_tasks=[],
                affected_ratio=0.0
            )
        assert "Severity must be one of" in str(exc_info.value)

    def test_event_equality_with_non_schedule_event(self):
        """测试事件与非ScheduleEvent比较"""
        timestamp = datetime.now(timezone.utc)
        event = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=timestamp,
            priority=5,
            description="测试",
            payload={}
        )

        assert event != "not an event"
        assert event != 123
        assert event != None

    def test_local_repair_with_failed_insertion(self):
        """测试局部修复时任务插入失败"""
        mock_scheduler = Mock(spec=BaseScheduler)
        event_scheduler = EventDrivenScheduler(base_scheduler=mock_scheduler)

        now = datetime.now(timezone.utc)
        # 创建两个时间冲突的任务
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now,
                imaging_end=now + timedelta(minutes=10),
                imaging_mode="push_broom"
            ),
            ScheduledTask(
                task_id="TASK-002",
                satellite_id="SAT-01",
                target_id="TGT-002",
                imaging_start=now + timedelta(minutes=15),
                imaging_end=now + timedelta(minutes=25),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 影响TASK-001，但尝试重新插入时会与TASK-002冲突
        impact = DisruptionImpact(
            severity="minor",
            affected_satellites=["SAT-01"],
            affected_tasks=["TASK-001"],
            affected_ratio=0.5
        )

        result = event_scheduler._local_repair(current_plan, impact)

        # 应该尝试重新插入但失败
        assert isinstance(result, ScheduleResult)

    def test_get_task_by_id_not_found(self):
        """测试获取不存在的任务"""
        mock_scheduler = Mock(spec=BaseScheduler)
        event_scheduler = EventDrivenScheduler(base_scheduler=mock_scheduler)

        now = datetime.now(timezone.utc)
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TGT-001",
                imaging_start=now,
                imaging_end=now + timedelta(minutes=5),
                imaging_mode="push_broom"
            )
        ]
        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        result = event_scheduler._get_task_by_id_from_plan(current_plan, "NONEXISTENT")

        assert result is None
