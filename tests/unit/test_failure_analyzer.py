"""
FailureAnalyzer 类的单元测试
"""

import pytest
from datetime import datetime
from scheduler.base_scheduler import ScheduleResult, TaskFailure, TaskFailureReason, ScheduledTask
from evaluation.failure_analyzer import FailureAnalyzer


class TestFailureAnalyzer:
    """FailureAnalyzer 测试类"""

    def test_analyze_empty_results(self):
        """测试空结果列表"""
        analyzer = FailureAnalyzer([])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] is None
        assert result['reason_distribution'] == {}
        assert result['recommendations'] == []

    def test_analyze_single_result_single_failure(self):
        """测试单个结果单个失败"""
        failure = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power insufficient"
        )
        schedule_result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.POWER_CONSTRAINT: 1}
        )
        analyzer = FailureAnalyzer([schedule_result])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] == 'power_constraint'
        assert result['reason_distribution'] == {'power_constraint': 1}
        assert len(result['recommendations']) == 1
        assert "电池容量" in result['recommendations'][0]

    def test_analyze_multiple_results_same_reason(self):
        """测试多个结果相同失败原因"""
        failure1 = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage full"
        )
        failure2 = TaskFailure(
            task_id="task2",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage full"
        )
        result1 = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure1},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.STORAGE_CONSTRAINT: 1}
        )
        result2 = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task2": failure2},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.STORAGE_CONSTRAINT: 1}
        )
        analyzer = FailureAnalyzer([result1, result2])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] == 'storage_constraint'
        assert result['reason_distribution'] == {'storage_constraint': 2}
        assert len(result['recommendations']) == 1
        assert "存储" in result['recommendations'][0]

    def test_analyze_mixed_reasons(self):
        """测试混合失败原因"""
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
            failure_reason=TaskFailureReason.NO_VISIBLE_WINDOW,
            failure_detail="No window"
        )
        schedule_result = ScheduleResult(
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
                TaskFailureReason.NO_VISIBLE_WINDOW: 1
            }
        )
        analyzer = FailureAnalyzer([schedule_result])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] == 'power_constraint'
        assert result['reason_distribution']['power_constraint'] == 2
        assert result['reason_distribution']['no_visible_window'] == 1
        assert len(result['recommendations']) == 1
        assert "电池" in result['recommendations'][0]

    def test_analyze_no_visible_window_recommendation(self):
        """测试无可见窗口的改进建议"""
        failure = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.NO_VISIBLE_WINDOW,
            failure_detail="No visible window available"
        )
        schedule_result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.NO_VISIBLE_WINDOW: 1}
        )
        analyzer = FailureAnalyzer([schedule_result])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] == 'no_visible_window'
        assert len(result['recommendations']) == 1
        assert "轨道" in result['recommendations'][0] or "卫星数量" in result['recommendations'][0]

    def test_analyze_unknown_reason_no_recommendation(self):
        """测试未知原因没有改进建议"""
        failure = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.UNKNOWN,
            failure_detail="Unknown error"
        )
        schedule_result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={TaskFailureReason.UNKNOWN: 1}
        )
        analyzer = FailureAnalyzer([schedule_result])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] == 'unknown'
        assert result['recommendations'] == []

    def test_analyze_with_scheduled_and_unscheduled(self):
        """测试同时有已调度和未调度任务的情况"""
        scheduled_task = ScheduledTask(
            task_id="task_scheduled",
            satellite_id="sat1",
            target_id="target1",
            imaging_start=datetime.now(),
            imaging_end=datetime.now(),
            imaging_mode="high"
        )
        failure = TaskFailure(
            task_id="task_failed",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage full"
        )
        schedule_result = ScheduleResult(
            scheduled_tasks=[scheduled_task],
            unscheduled_tasks={"task_failed": failure},
            makespan=100.0,
            computation_time=1.0,
            iterations=10,
            failure_summary={TaskFailureReason.STORAGE_CONSTRAINT: 1}
        )
        analyzer = FailureAnalyzer([schedule_result])
        result = analyzer.analyze_failure_patterns()

        assert result['most_common_reason'] == 'storage_constraint'
        assert result['reason_distribution'] == {'storage_constraint': 1}

    def test_analyze_multiple_results_aggregation(self):
        """测试多个结果的失败聚合"""
        failure1 = TaskFailure(
            task_id="task1",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power"
        )
        failure2 = TaskFailure(
            task_id="task2",
            failure_reason=TaskFailureReason.STORAGE_CONSTRAINT,
            failure_detail="Storage"
        )
        failure3 = TaskFailure(
            task_id="task3",
            failure_reason=TaskFailureReason.POWER_CONSTRAINT,
            failure_detail="Power"
        )
        failure4 = TaskFailure(
            task_id="task4",
            failure_reason=TaskFailureReason.TIME_CONFLICT,
            failure_detail="Conflict"
        )

        result1 = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task1": failure1, "task2": failure2},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={
                TaskFailureReason.POWER_CONSTRAINT: 1,
                TaskFailureReason.STORAGE_CONSTRAINT: 1
            }
        )
        result2 = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={"task3": failure3, "task4": failure4},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            failure_summary={
                TaskFailureReason.POWER_CONSTRAINT: 1,
                TaskFailureReason.TIME_CONFLICT: 1
            }
        )

        analyzer = FailureAnalyzer([result1, result2])
        analysis = analyzer.analyze_failure_patterns()

        assert analysis['most_common_reason'] == 'power_constraint'
        assert analysis['reason_distribution']['power_constraint'] == 2
        assert analysis['reason_distribution']['storage_constraint'] == 1
        assert analysis['reason_distribution']['time_conflict'] == 1
