#!/usr/bin/env python3
"""
增量任务规划示例

演示如何使用增量规划功能在现有调度结果基础上规划新目标
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from scheduler.incremental import (
    IncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalStrategyType,
    PriorityRule,
    PreemptionRule
)
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


class MockTarget:
    """模拟目标"""
    def __init__(self, target_id, priority=0, imaging_mode='standard'):
        self.id = target_id
        self.priority = priority
        self.imaging_mode = imaging_mode
        self.latitude = 39.9
        self.longitude = 116.4


class MockSatellite:
    """模拟卫星"""
    def __init__(self, sat_id):
        self.id = sat_id
        class Caps:
            power_capacity = 2800.0
            storage_capacity = 128.0
        self.capabilities = Caps()


class MockMission:
    """模拟任务场景"""
    def __init__(self):
        self.start_time = datetime(2024, 1, 1, 0, 0, 0)
        self.end_time = datetime(2024, 1, 2, 0, 0, 0)
        self.satellites = [MockSatellite(f'SAT-{i:03d}') for i in range(1, 4)]


def create_sample_schedule():
    """创建示例调度结果"""
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    tasks = [
        ScheduledTask(
            task_id=f'T{i:03d}',
            satellite_id=f'SAT-{(i % 3) + 1:03d}',
            target_id=f'TARGET-{i:03d}',
            imaging_start=start_time + timedelta(hours=i*2),
            imaging_end=start_time + timedelta(hours=i*2, minutes=10),
            imaging_mode='standard',
            power_consumed=10.0,
            power_before=2800.0,
            power_after=2790.0,
            storage_before=0.0,
            storage_after=5.0,
            priority=i % 10
        )
        for i in range(10)
    ]

    return ScheduleResult(
        scheduled_tasks=tasks,
        unscheduled_tasks={},
        makespan=72000.0,
        computation_time=5.0,
        iterations=500
    )


def example_1_basic_usage():
    """示例1：基本使用"""
    print("=" * 60)
    print("示例1：基本使用 - 保守策略")
    print("=" * 60)

    # 创建示例数据
    existing_schedule = create_sample_schedule()
    mission = MockMission()

    # 创建新目标
    new_targets = [
        MockTarget('NEW-001', priority=8),
        MockTarget('NEW-002', priority=5),
        MockTarget('NEW-003', priority=3)
    ]

    # 创建增量规划请求
    request = IncrementalPlanRequest(
        new_targets=new_targets,
        existing_schedule=existing_schedule,
        strategy=IncrementalStrategyType.CONSERVATIVE,
        mission=mission
    )

    # 执行增量规划
    planner = IncrementalPlanner()
    result = planner.plan(request)

    # 输出结果
    print(f"策略: {result.strategy_used.value}")
    print(f"新增任务: {len(result.new_tasks)}")
    print(f"被抢占任务: {len(result.preempted_tasks)}")
    print(f"重调度任务: {len(result.rescheduled_tasks)}")
    print(f"失败目标: {len(result.failed_targets)}")
    print(f"成功率: {result.statistics.get('success_rate', 0):.2%}")
    print()


def example_2_aggressive_strategy():
    """示例2：激进策略"""
    print("=" * 60)
    print("示例2：激进策略 - 允许抢占")
    print("=" * 60)

    existing_schedule = create_sample_schedule()
    mission = MockMission()

    # 创建高优先级新目标
    new_targets = [
        MockTarget('URGENT-001', priority=10),  # 最高优先级
        MockTarget('URGENT-002', priority=9),
        MockTarget('NORMAL-001', priority=4)
    ]

    # 使用激进策略
    request = IncrementalPlanRequest(
        new_targets=new_targets,
        existing_schedule=existing_schedule,
        strategy=IncrementalStrategyType.AGGRESSIVE,
        preemption_rules=PreemptionRule(
            min_priority_difference=2,
            max_preemption_ratio=0.2,
            max_cascade_depth=2
        ),
        max_preemption_ratio=0.2,
        mission=mission
    )

    planner = IncrementalPlanner()
    result = planner.plan(request)

    print(f"策略: {result.strategy_used.value}")
    print(f"新增任务: {len(result.new_tasks)}")
    print(f"被抢占任务: {len(result.preempted_tasks)}")
    print(f"重调度任务: {len(result.rescheduled_tasks)}")
    print(f"失败目标: {len(result.failed_targets)}")
    print(f"抢占次数: {result.statistics.get('preemption_count', 0)}")
    print()


def example_3_resource_analysis():
    """示例3：资源分析"""
    print("=" * 60)
    print("示例3：资源分析")
    print("=" * 60)

    existing_schedule = create_sample_schedule()
    mission = MockMission()

    planner = IncrementalPlanner()

    # 生成资源报告
    report = planner.analyze_resources(existing_schedule, mission)

    print("资源使用汇总:")
    print(f"  卫星总数: {report['summary']['total_satellites']}")
    print(f"  总可用时间: {report['summary']['total_available_time_hours']:.2f} 小时")
    print(f"  可用窗口数: {report['summary']['total_windows']}")
    print(f"  平均利用率: {report['summary']['average_utilization_rate']:.2%}")
    print()

    # 估算容量
    capacities = planner.estimate_capacity(
        existing_schedule, mission, avg_task_duration=300.0
    )

    print("各卫星剩余容量:")
    for sat_id, capacity in capacities.items():
        print(f"  {sat_id}: 约 {capacity} 个任务")
    print()


def example_4_scheduler_api():
    """示例4：使用调度器API"""
    print("=" * 60)
    print("示例4：使用调度器API")
    print("=" * 60)

    mission = MockMission()
    existing_schedule = create_sample_schedule()
    new_targets = [MockTarget(f'EXTRA-{i:03d}', priority=6) for i in range(5)]

    # 创建调度器
    scheduler = GreedyScheduler()
    scheduler.initialize(mission)

    # 从现有调度恢复状态
    scheduler.resume_from(existing_schedule, new_targets, strategy='conservative')

    # 获取剩余容量
    capacities = scheduler.get_remaining_capacity()
    print("剩余资源容量:")
    for sat_id, cap in capacities.items():
        print(f"  {sat_id}:")
        print(f"    可用时间: {cap['available_time']:.1f} 秒")
        print(f"    可用电量: {cap['available_power']:.1f} Wh")
        print(f"    可用存储: {cap['available_storage']:.1f} GB")
        print(f"    利用率: {cap['utilization_rate']:.2%}")
    print()

    # 执行增量规划（注意：实际使用时需要完善schedule_incremental方法）
    print("注意: schedule_incremental方法需要在完整环境中使用")
    print()


def example_5_mixed_strategy():
    """示例5：混合策略"""
    print("=" * 60)
    print("示例5：混合策略 - 自动选择")
    print("=" * 60)

    existing_schedule = create_sample_schedule()
    mission = MockMission()

    # 创建混合优先级目标
    new_targets = [
        MockTarget('LOW-001', priority=2),
        MockTarget('LOW-002', priority=3),
        MockTarget('LOW-003', priority=4),
        MockTarget('HIGH-001', priority=9),
        MockTarget('HIGH-002', priority=10),
        MockTarget('HIGH-003', priority=8)
    ]

    # 使用混合策略
    request = IncrementalPlanRequest(
        new_targets=new_targets,
        existing_schedule=existing_schedule,
        strategy=IncrementalStrategyType.HYBRID,
        mission=mission
    )

    planner = IncrementalPlanner()
    result = planner.plan(request)

    print(f"策略: {result.strategy_used.value}")
    print(f"混合模式: {result.statistics.get('hybrid_mode', 'unknown')}")
    print(f"新增任务: {len(result.new_tasks)}")
    print(f"被抢占任务: {len(result.preempted_tasks)}")
    print(f"失败目标: {len(result.failed_targets)}")

    # 显示阶段统计
    if 'phase1_scheduled' in result.statistics:
        print(f"阶段1调度: {result.statistics['phase1_scheduled']}")
    if 'phase2_scheduled' in result.statistics:
        print(f"阶段2调度: {result.statistics['phase2_scheduled']}")
    print()


if __name__ == '__main__':
    print("\n增量任务规划示例\n")

    example_1_basic_usage()
    example_2_aggressive_strategy()
    example_3_resource_analysis()
    example_4_scheduler_api()
    example_5_mixed_strategy()

    print("=" * 60)
    print("所有示例执行完成")
    print("=" * 60)
