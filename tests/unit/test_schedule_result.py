"""
ScheduleResult 类的单元测试
"""

import pytest
from datetime import datetime
from scheduler.base_scheduler import ScheduleResult, ScheduledTask, TaskFailure, TaskFailureReason


class TestScheduleResult:
    """ScheduleResult 测试类"""

    def test_get_failure_rate_by_reason_empty(self):
        """测试空失败列表时返回空字典"""
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0
        )
        rates = result.get_failure_rate_by_reason()
        assert rates == {}

    def test_get_failure_rate_by_reason_single_reason(self):
        """测试单一失败原因的比例计算"""
        failure = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.POWER_CONSTRAINT: 1}
        )
        rates = result.get_failure_rate_by_reason()
        assert rates == {"power_constraint": 1.0}

    def test_get_failure_rate_by_reason_multiple_same_reason(self):
        """测试多个相同失败原因的比例计算"""
        failure1 = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        failure2 = TaskFailure(
            task_id="task2",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure1, "task2": failure2},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.POWER_CONSTRAINT: 2}
        )
        rates = result.get_failure_rate_by_reason()
        assert rates == {"power_constraint": 1.0}

    def test_get_failure_rate_by_reason_multiple_different_reasons(self):
        """测试多个不同失败原因的比例计算"""
        failure1 = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        failure2 = TaskFailure(
            task_id="task2",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage full"
        )
        failure3 = TaskFailure(
            task_id="task3",
            failure_reason=TaskFailureReason.NO_VISIBLE_WINDOW,
            failure_detail="No window"
        )
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={
                "task1": failure1,
                "task2": failure2,
                "task3": failure3
            },
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={
                TaskFailureReason.POWER_CONSTRAINT: 1,
                TaskFailureReason.STORAGE_CONSTRAINT: 1,
                TaskFailureReason.NO_VISIBLE_WINDOW: 1
            }
        )
        rates = result.get_failure_rate_by_reason()
        assert len(rates) == 3
        assert rates["power_constraint"] == pytest.approx(1/3)
        assert rates["storage_constraint"] == pytest.approx(1/3)
        assert rates["no_visible_window"] == pytest.approx(1/3)

    def test_get_failure_rate_by_reason_mixed_counts(self):
        """测试混合数量失败原因的比例计算"""
        failure1 = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        failure2 = TaskFailure(
            task_id="task2",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        failure3 = TaskFailure(
            task_id="task3",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage full"
        )
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={
                "task1": failure1,
                "task2": failure2,
                "task3": failure3
            },
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={
                TaskFailureReason.POWER_CONSTRAINT: 2,
                TaskFailureReason.STORAGE_CONSTRAINT: 1
            }
        )
        rates = result.get_failure_rate_by_reason()
        assert len(rates) == 2
        assert rates["power_constraint"] == pytest.approx(2/3)
        assert rates["storage_constraint"] == pytest.approx(1/3)

    def test_get_failure_rate_by_reason_with_scheduled_tasks(self):
        """测试有已调度任务时的失败比例计算"""
        failure1 = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        failure2 = TaskFailure(
            task_id="task2",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage full"
        )
        scheduled_task = ScheduledTask(
            task_id="task3",
            satellite_id="sat1",
            target_id="target1",
            imaging_start=datetime.now(),
            imaging_end=datetime.now(),
            imaging_mode="high"
        )
        result = ScheduleResult(
            scheduled_tasks=[scheduled_task],
            unscheduled_tasks={
                "task1": failure1,
                "task2": failure2
            },
            makespan=100.0,
            computation_time=1.0,
            iterations=10,
            failure_summary={
                TaskFailureReason.POWER_CONSTRAINT: 1,
                TaskFailureReason.STORAGE_CONSTRAINT: 1
            }
        )
        rates = result.get_failure_rate_by_reason()
        assert len(rates) == 2
        assert rates["power_constraint"] == pytest.approx(0.5)
        assert rates["storage_constraint"] == pytest.approx(0.5)

    def test_get_failure_rate_by_reason_no_failure_summary(self):
        """测试没有 failure_summary 时返回空字典"""
        failure = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary=None
        )
        rates = result.get_failure_rate_by_reason()
        assert rates == {}
