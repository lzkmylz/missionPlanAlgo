"""
统一时空约束检查集成示例

演示如何在元启发式算法中使用 MetaheuristicConstraintChecker 的统一约束检查功能。

对比两种使用方式：
1. 传统方式：分别调用各个约束检查方法
2. 统一方式：使用 check_task_placement 一次性检查所有时空约束
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from core.models import Mission
from scheduler.metaheuristic.constraints_utils import MetaheuristicConstraintChecker


def example_traditional_vs_unified():
    """对比传统方式和统一方式"""
    print("=" * 70)
    print("示例: 传统方式 vs 统一方式约束检查")
    print("=" * 70)

    # 加载场景
    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)
    print(f"\n场景加载成功: {len(mission.satellites)}颗卫星, {len(mission.targets)}个目标")

    # 创建传统约束检查器（不使用统一检查）
    traditional_checker = MetaheuristicConstraintChecker(
        mission,
        config={
            'consider_power': True,
            'consider_storage': True,
            # 高精度要求：始终使用精确模式
            'use_unified_constraints': False,  # 禁用统一约束
        }
    )
    traditional_checker.initialize()

    # 创建统一约束检查器
    unified_checker = MetaheuristicConstraintChecker(
        mission,
        config={
            'consider_power': True,
            'consider_storage': True,
            # 高精度要求：始终使用精确模式
            'use_unified_constraints': True,  # 启用统一约束
        }
    )
    unified_checker.initialize()

    # 测试任务
    sat = mission.satellites[0]
    target1 = mission.targets[0]
    target2 = mission.targets[1]

    window_start = mission.start_time + timedelta(minutes=10)
    window_end = window_start + timedelta(minutes=5)
    imaging_duration = 30.0

    print("\n--- 传统方式（分别检查）---")
    # 1. 检查卫星能力
    can_observe = traditional_checker._check_satellite_capability(sat, target1)
    print(f"1. 卫星能力检查: {'通过' if can_observe else '失败'}")

    # 2. 检查资源约束
    imaging_mode = traditional_checker._select_imaging_mode(sat, target1)
    has_resources = traditional_checker._check_resource_constraints(sat, target1, imaging_mode)
    print(f"2. 资源约束检查: {'通过' if has_resources else '失败'}")

    # 3. 检查机动约束
    slew_result = traditional_checker._check_slew_constraint(sat.id, target1, window_start, imaging_duration)
    if slew_result:
        print(f"3. 机动约束检查: {'通过' if slew_result.feasible else '失败'}")
        print(f"   机动角度: {slew_result.slew_angle:.2f}°")
        print(f"   机动时间: {slew_result.slew_time:.2f}s")
        print(f"   实际开始: {slew_result.actual_start}")
    else:
        print("3. 机动约束检查: 未初始化")

    # 4. 检查时间冲突
    actual_start = slew_result.actual_start if slew_result else window_start
    actual_end = actual_start + timedelta(seconds=imaging_duration)
    has_conflict = traditional_checker._has_time_conflict(sat.id, actual_start, actual_end)
    print(f"4. 时间冲突检查: {'通过' if not has_conflict else '失败'}")

    print("\n--- 统一方式（一次调用）---")
    result = unified_checker.check_task_placement(
        sat_id=sat.id,
        target=target1,
        window_start=window_start,
        window_end=window_end,
        imaging_duration=imaging_duration,
        task_id="TASK-001"
    )

    print(f"统一检查结果:")
    print(f"  可行: {result.feasible}")
    if result.feasible:
        print(f"  实际开始: {result.actual_start}")
        print(f"  实际结束: {result.actual_end}")
        print(f"  机动角度: {result.slew_angle:.2f}°")
        print(f"  机动时间: {result.slew_time:.2f}s")
    else:
        print(f"  冲突原因: {result.conflict_reason}")

    # 提交第一个任务
    if result.feasible:
        unified_checker.commit_task_placement(
            sat_id=sat.id,
            task_id="TASK-001",
            target_id=target1.id,
            actual_start=result.actual_start,
            actual_end=result.actual_end,
            target=target1
        )
        print(f"\n任务 TASK-001 已提交")

    # 检查第二个任务（应该有时间冲突）
    print("\n--- 检查第二个任务（应与第一个冲突）---")
    window2_start = result.actual_start + timedelta(seconds=10)  # 在第一个任务开始后10秒
    window2_end = window2_start + timedelta(minutes=5)

    result2 = unified_checker.check_task_placement(
        sat_id=sat.id,
        target=target2,
        window_start=window2_start,
        window_end=window2_end,
        imaging_duration=imaging_duration,
        task_id="TASK-002"
    )

    print(f"任务 TASK-002 检查结果:")
    print(f"  可行: {result2.feasible}")
    if not result2.feasible:
        print(f"  冲突原因: {result2.conflict_reason}")
        print(f"  与任务冲突: {result2.conflict_with}")


def example_batch_check():
    """演示批量约束检查"""
    print("\n" + "=" * 70)
    print("示例: 批量任务约束检查（用于元启发式算法评估）")
    print("=" * 70)

    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)

    checker = MetaheuristicConstraintChecker(
        mission,
        config={
            # 高精度要求：始终使用精确模式
            'use_unified_constraints': True,
        }
    )
    checker.initialize()

    sat = mission.satellites[0]
    targets = mission.targets[:5]  # 测试5个目标

    # 准备批量放置列表
    placements = []
    base_time = mission.start_time + timedelta(minutes=10)

    for i, target in enumerate(targets):
        window_start = base_time + timedelta(minutes=i * 10)  # 间隔10分钟
        window_end = window_start + timedelta(minutes=5)

        placements.append({
            'satellite_id': sat.id,
            'target': target,
            'window_start': window_start,
            'window_end': window_end,
            'imaging_duration': 30.0,
            'task_id': f"BATCH-TASK-{i+1}"
        })

    print(f"\n批量检查 {len(placements)} 个任务...")

    # 批量检查
    all_feasible, results = checker._unified_checker.check_batch_placement(placements)

    print(f"\n批量检查结果:")
    print(f"  全部可行: {all_feasible}")
    print(f"\n  详细结果:")
    for i, (placement, result) in enumerate(zip(placements, results)):
        status = "✓" if result.feasible else "✗"
        print(f"    {status} {placement['task_id']}: {target.id if target else 'N/A'}")
        if result.feasible:
            print(f"        开始: {result.actual_start.strftime('%H:%M:%S')}, "
                  f"机动: {result.slew_time:.1f}s")
        else:
            print(f"        原因: {result.conflict_reason}")


def main():
    """主函数"""
    print("统一时空约束检查集成示例")
    print("=" * 70)

    example_traditional_vs_unified()
    example_batch_check()

    print("\n" + "=" * 70)
    print("总结:")
    print("  1. 传统方式需要4步分别检查不同约束")
    print("  2. 统一方式只需1步即可获得完整结果")
    print("  3. 统一方式自动管理任务状态，避免冲突")
    print("  4. 批量检查功能适合元启发式算法快速评估")
    print("=" * 70)


if __name__ == "__main__":
    main()
