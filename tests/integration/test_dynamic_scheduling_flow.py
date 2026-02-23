"""
动态调度核心流程集成测试

验证第19章和第20章模块的集成:
- SOE生成器与保护时间验证
- 事件驱动调度器
- 滚动时间窗管理器
- 在轨处理管理器
- 帕累托优化器
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from scheduler.base_scheduler import ScheduleResult, ScheduledTask
from core.telecommand.soe_generator import (
    SOEGenerator, SOEEntry, SOEActionType, GuardTimeValidator
)
from core.dynamic_scheduler.rolling_horizon import (
    RollingHorizonManager, RollingHorizonConfig
)
from core.dynamic_scheduler.event_driven_scheduler import (
    EventDrivenScheduler, EventType, ScheduleEvent, DisruptionImpact
)
from core.processing.onboard_processing_manager import (
    OnboardProcessingManager, AIAcceleratorSpec, AIAcceleratorType,
    ProcessingTaskSpec, ProcessingTaskType, ProcessingDecision,
    SatelliteResourceState, DecisionContext
)
from core.processing.pareto_optimizer import ParetoOptimizer, ObjectiveFunction


class TestDynamicSchedulingIntegration:
    """动态调度集成测试"""

    def test_full_scheduling_pipeline(self):
        """测试完整的调度流程"""
        # 1. 创建模拟调度结果
        # 注意：任务之间需要足够间隔以确保保护时间合规
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                imaging_end=datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                imaging_mode="push_broom",
                slew_angle=15.0
            ),
            ScheduledTask(
                task_id="TASK-002",
                satellite_id="SAT-02",  # 不同卫星，避免保护时间冲突
                target_id="TARGET-B",
                imaging_start=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                imaging_end=datetime(2024, 1, 1, 11, 3, 0, tzinfo=timezone.utc),
                imaging_mode="spotlight",
                slew_angle=-10.0
            )
        ]

        schedule = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=7200.0,
            computation_time=1.5,
            iterations=100
        )

        # 2. 生成SOE序列
        soe_gen = SOEGenerator()
        soe_entries = soe_gen.generate_soe(schedule)

        assert len(soe_entries) > 0
        assert all(isinstance(e, SOEEntry) for e in soe_entries)

        # 3. 验证保护时间
        validator = GuardTimeValidator()
        violations = validator.validate_soe(soe_entries)

        # 如果有违规，使用auto_fix修复
        if violations:
            fixed_entries = validator.auto_fix(soe_entries)
            violations_after_fix = validator.validate_soe(fixed_entries)
            assert len(violations_after_fix) == 0, f"保护时间修复后仍有违规: {violations_after_fix}"
        else:
            assert len(violations) == 0

    def test_event_driven_reschedule_pipeline(self):
        """测试事件驱动重调度流程"""
        from scheduler.greedy.edd_scheduler import EDDScheduler

        # 创建基础调度器
        base_scheduler = EDDScheduler()

        # 创建事件驱动调度器
        event_scheduler = EventDrivenScheduler(base_scheduler)

        # 创建模拟调度结果
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                imaging_end=datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                imaging_mode="push_broom",
                slew_angle=15.0
            )
        ]

        current_plan = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=0.5,
            iterations=50
        )

        # 创建新紧急任务事件
        urgent_event = ScheduleEvent(
            event_type=EventType.NEW_URGENT_TASK,
            timestamp=datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc),
            priority=9,  # 高优先级
            description="Emergency observation request",
            payload={"task_id": "URGENT-001", "priority": 10},
            affected_satellites=["SAT-01"],
            affected_tasks=["TASK-001"]
        )

        # 提交事件（高优先级应立即触发）
        triggered = event_scheduler.submit_event(urgent_event)
        assert triggered is True  # 优先级>=8应立即触发

        # 测试影响分析
        impact = event_scheduler._analyze_impact(current_plan, [urgent_event])
        assert isinstance(impact, DisruptionImpact)
        assert impact.severity in ['minor', 'moderate', 'major']

    def test_rolling_horizon_integration(self):
        """测试滚动时间窗集成"""
        config = RollingHorizonConfig(
            window_size=timedelta(hours=2),
            shift_interval=timedelta(minutes=15),
            freeze_duration=timedelta(minutes=5)
        )

        manager = RollingHorizonManager(config)

        # 第一次应该触发优化
        current_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert manager.should_trigger_optimization(current_time) is True

        # 记录优化时间
        manager.record_optimization(current_time)

        # 15分钟内不应再次触发
        later = current_time + timedelta(minutes=10)
        assert manager.should_trigger_optimization(later) is False

        # 获取优化窗口
        window_start, window_end, freeze_until = manager.get_optimization_window(current_time)

        # 验证窗口关系
        assert freeze_until == current_time + config.freeze_duration
        assert window_start == freeze_until
        assert window_end == current_time + config.window_size

    def test_onboard_processing_decision_pipeline(self):
        """测试在轨处理决策流程"""
        # 配置AI加速器
        accelerator_specs = {
            "SAT-01": AIAcceleratorSpec(
                accelerator_type=AIAcceleratorType.NVIDIA_JETSON_AGX,
                compute_tops=32.0,
                power_consumption_w=30.0,
                power_idle_w=5.0,
                memory_gb=32.0,
                radiation_hardened=True,
                operational_temp_range=(-25.0, 85.0),
                tid_tolerance_krad=100.0,
                see_immune=True
            )
        }

        # 配置处理任务规格
        processing_specs = {
            ProcessingTaskType.VESSEL_DETECTION: ProcessingTaskSpec(
                task_type=ProcessingTaskType.VESSEL_DETECTION,
                input_data_size_gb=5.0,
                output_data_size_kb=10.0,
                compute_requirement_tops=100.0,
                min_confidence=0.85
            )
        }

        manager = OnboardProcessingManager(accelerator_specs, processing_specs)

        # 创建决策上下文
        imaging_task = {
            "task_id": "TASK-001",
            "satellite_id": "SAT-01",
            "target_type": "maritime",
            "data_size_gb": 5.0
        }

        satellite_state = SatelliteResourceState(
            battery_soc=0.8,  # 80%电量
            storage_free_gb=100.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )

        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=5,
            latency_requirement=timedelta(hours=1),
            accuracy_requirement=0.9
        )

        # 做出处理决策
        decision, metadata = manager.make_processing_decision(context)

        assert isinstance(decision, ProcessingDecision)
        assert decision in [
            ProcessingDecision.PROCESS_ONBOARD,
            ProcessingDecision.DOWNLINK_RAW,
            ProcessingDecision.HYBRID
        ]

    def test_pareto_optimization_integration(self):
        """测试帕累托优化集成"""
        # 定义5个目标函数
        objectives = [
            ObjectiveFunction(name="energy", weight=0.3, minimize=True),
            ObjectiveFunction(name="time", weight=0.2, minimize=True),
            ObjectiveFunction(name="storage", weight=0.2, minimize=True),
            ObjectiveFunction(name="bandwidth", weight=0.2, minimize=True),
            ObjectiveFunction(name="thermal", weight=0.1, minimize=True),
        ]

        optimizer = ParetoOptimizer(objectives)

        # 创建成像任务
        imaging_tasks = [
            {"task_id": f"TASK-{i:03d}", "satellite_id": "SAT-01", "data_size_gb": 5.0}
            for i in range(10)
        ]

        # 创建卫星配置
        satellites = [
            {
                "id": "SAT-01",
                "has_ai_accelerator": True,
                "ai_power_wh": 50.0,
                "ai_processing_time": 300.0
            }
        ]

        # 运行优化
        pareto_solutions = optimizer.optimize(
            imaging_tasks=imaging_tasks,
            satellites=satellites,
            population_size=20,
            generations=10
        )

        # 验证结果
        assert len(pareto_solutions) > 0
        assert all("decision_vector" in sol for sol in pareto_solutions)
        assert all("objectives" in sol for sol in pareto_solutions)

    def test_end_to_end_dynamic_scheduling(self):
        """端到端动态调度测试"""
        # 1. 初始化所有组件
        soe_gen = SOEGenerator()
        validator = GuardTimeValidator()

        horizon_config = RollingHorizonConfig()
        horizon_manager = RollingHorizonManager(horizon_config)

        # 2. 创建模拟调度结果
        scheduled_tasks = [
            ScheduledTask(
                task_id="TASK-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                imaging_end=datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                imaging_mode="push_broom",
                slew_angle=15.0
            )
        ]

        schedule = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=1.0,
            iterations=100
        )

        # 3. 生成并验证SOE
        soe_entries = soe_gen.generate_soe(schedule)
        violations = validator.validate_soe(soe_entries)
        assert len(violations) == 0

        # 4. 检查是否应该触发滚动优化
        current_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        should_optimize = horizon_manager.should_trigger_optimization(current_time)
        assert should_optimize is True

        # 5. 获取优化窗口
        window_start, window_end, freeze_until = horizon_manager.get_optimization_window(current_time)
        assert window_start < window_end
        assert freeze_until <= window_start

    def test_state_constraints_integration(self):
        """测试状态约束集成"""
        accelerator_specs = {
            "SAT-01": AIAcceleratorSpec(
                accelerator_type=AIAcceleratorType.NVIDIA_JETSON_AGX,
                compute_tops=32.0,
                power_consumption_w=30.0,
                power_idle_w=5.0,
                memory_gb=32.0,
                radiation_hardened=True,
                operational_temp_range=(-25.0, 85.0),
                tid_tolerance_krad=100.0,
                see_immune=True
            )
        }

        processing_specs = {
            ProcessingTaskType.VESSEL_DETECTION: ProcessingTaskSpec(
                task_type=ProcessingTaskType.VESSEL_DETECTION,
                input_data_size_gb=5.0,
                output_data_size_kb=10.0,
                compute_requirement_tops=100.0,
                min_confidence=0.85
            )
        }

        manager = OnboardProcessingManager(accelerator_specs, processing_specs)

        # 测试低电量约束
        imaging_task = {
            "task_id": "TASK-001",
            "satellite_id": "SAT-01",
            "target_type": "maritime",
            "data_size_gb": 5.0
        }

        low_battery_state = SatelliteResourceState(
            battery_soc=0.2,  # 20%电量（低于30%阈值）
            storage_free_gb=100.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )

        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=low_battery_state,
            mission_priority=5,
            latency_requirement=timedelta(hours=1),
            accuracy_requirement=0.9
        )

        decision, metadata = manager.make_processing_decision(context)

        # 低电量应强制下传原始数据
        assert decision == ProcessingDecision.DOWNLINK_RAW
        assert "override_reason" in metadata
        assert "Low battery" in metadata["override_reason"]

    def test_event_priority_handling(self):
        """测试事件优先级处理"""
        from scheduler.greedy.edd_scheduler import EDDScheduler

        base_scheduler = EDDScheduler()
        event_scheduler = EventDrivenScheduler(base_scheduler)

        # 低优先级事件（不立即触发）
        low_priority_event = ScheduleEvent(
            event_type=EventType.ROLLING_HORIZON_TRIGGER,
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            priority=5,
            description="Regular rolling horizon trigger",
            payload={}
        )

        triggered = event_scheduler.submit_event(low_priority_event)
        assert triggered is False  # 优先级<8不立即触发

        # 高优先级事件（立即触发）
        high_priority_event = ScheduleEvent(
            event_type=EventType.SATELLITE_FAILURE,
            timestamp=datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
            priority=10,
            description="Satellite failure detected",
            payload={"satellite_id": "SAT-01"},
            affected_satellites=["SAT-01"]
        )

        triggered = event_scheduler.submit_event(high_priority_event)
        assert triggered is True  # 优先级>=8立即触发

        # 验证事件队列排序（高优先级在前）
        assert len(event_scheduler.event_queue) == 2
        assert event_scheduler.event_queue[0].priority == 10
        assert event_scheduler.event_queue[1].priority == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
