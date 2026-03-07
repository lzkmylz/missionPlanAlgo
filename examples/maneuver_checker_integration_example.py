"""
统一机动约束检查器集成示例

演示如何在元启发式算法中使用 MetaheuristicConstraintChecker 的统一机动约束检查功能。

对比两种使用方式：
1. 传统方式：分别调用各个约束检查方法
2. 统一机动方式：使用 check_maneuver_placement 一次性检查所有机动相关约束
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from core.models import Mission
from core.dynamics.attitude_mode import AttitudeMode
from scheduler.metaheuristic.constraints_utils import MetaheuristicConstraintChecker


def example_traditional_vs_maneuver():
    """对比传统方式和统一机动方式"""
    print("=" * 70)
    print("示例: 传统方式 vs 统一机动方式约束检查")
    print("=" * 70)

    # 加载场景
    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)
    print(f"\n场景加载成功: {len(mission.satellites)}颗卫星, {len(mission.targets)}个目标")

    # 创建传统约束检查器（不使用统一机动检查）
    traditional_checker = MetaheuristicConstraintChecker(
        mission,
        config={
            'consider_power': True,
            'consider_storage': True,
            'use_simplified_slew': True,
            'use_unified_maneuver': False,
        }
    )
    traditional_checker.initialize()

    # 创建统一机动约束检查器
    maneuver_checker = MetaheuristicConstraintChecker(
        mission,
        config={
            'consider_power': True,
            'consider_storage': True,
            'use_simplified_slew': False,
            'use_unified_maneuver': True,
            'max_slew_rate': 3.0,
            'settling_time': 5.0,
            'enable_sun_pointing_optimization': True,
        }
    )
    maneuver_checker.initialize()

    # 测试任务
    sat = mission.satellites[0]
    target = mission.targets[0]

    window_start = mission.start_time + timedelta(minutes=10)
    window_end = window_start + timedelta(minutes=5)
    imaging_duration = 30.0
    satellite_position = (7000000.0, 0.0, 0.0)

    print("\n--- 传统方式（分别检查）---")
    # 1. 检查卫星能力
    can_observe = traditional_checker._check_satellite_capability(sat, target)
    print(f"1. 卫星能力检查: {'通过' if can_observe else '失败'}")

    # 2. 检查资源约束
    imaging_mode = traditional_checker._select_imaging_mode(sat, target)
    has_resources = traditional_checker._check_resource_constraints(sat, target, imaging_mode)
    print(f"2. 资源约束检查: {'通过' if has_resources else '失败'}")

    # 3. 检查机动约束（简化）
    slew_result = traditional_checker._check_slew_constraint(sat.id, target, window_start, imaging_duration)
    if slew_result:
        print(f"3. 机动约束检查: {'通过' if slew_result.feasible else '失败'}")
        print(f"   机动时间: {slew_result.slew_time:.2f}s")
    else:
        print("3. 机动约束检查: 未初始化")

    print("\n--- 统一机动方式（一次调用）---")
    result = maneuver_checker.check_maneuver_placement(
        sat_id=sat.id,
        target=target,
        window_start=window_start,
        window_end=window_end,
        imaging_duration=imaging_duration,
        satellite_position=satellite_position,
        task_id="TASK-001",
        from_mode=AttitudeMode.NADIR_POINTING,
        to_mode=AttitudeMode.IMAGING
    )

    print(f"统一机动检查结果:")
    print(f"  可行: {result.feasible}")
    if result.feasible:
        print(f"  实际开始: {result.actual_start}")
        print(f"  实际结束: {result.actual_end}")
        print(f"  机动角度: {result.slew_angle:.2f}°")
        print(f"  机动时间: {result.slew_time:.2f}秒")
        print(f"  目标滚转角: {result.roll_angle:.2f}°")
        print(f"  目标俯仰角: {result.pitch_angle:.2f}°")
        print(f"  切换前功率: {result.power_before:.2f}W")
        print(f"  切换后功率: {result.power_after:.2f}W")
    else:
        print(f"  冲突原因: {result.conflict_reason}")

    # 提交第一个任务
    if result.feasible:
        maneuver_checker.commit_maneuver_task(
            sat_id=sat.id,
            task_id="TASK-001",
            target_id=target.id,
            actual_start=result.actual_start,
            actual_end=result.actual_end,
            target=target,
            end_mode=result.to_mode
        )
        print(f"\n任务 TASK-001 已提交")


def example_batch_check():
    """演示批量约束检查"""
    print("\n" + "=" * 70)
    print("示例: 批量任务机动约束检查（用于元启发式算法评估）")
    print("=" * 70)

    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)

    checker = MetaheuristicConstraintChecker(
        mission,
        config={
            'use_simplified_slew': True,
            'use_unified_maneuver': True,
        }
    )
    checker.initialize()

    sat = mission.satellites[0]
    targets = mission.targets[:3]

    # 准备批量放置列表
    placements = []
    base_time = mission.start_time + timedelta(minutes=10)

    for i, target in enumerate(targets):
        window_start = base_time + timedelta(minutes=i * 10)
        window_end = window_start + timedelta(minutes=5)

        placements.append({
            'satellite_id': sat.id,
            'target': target,
            'window_start': window_start,
            'window_end': window_end,
            'imaging_duration': 30.0,
            'satellite_position': (7000000.0, 0.0, 0.0),
            'task_id': f"BATCH-TASK-{i+1}"
        })

    print(f"\n批量检查 {len(placements)} 个任务...")

    # 通过 maneuver_checker 批量检查
    if checker._maneuver_checker:
        all_feasible, results = checker._maneuver_checker.check_batch_placement(placements)

        print(f"\n批量检查结果:")
        print(f"  全部可行: {all_feasible}")
        print(f"\n  详细结果:")
        for i, (placement, result) in enumerate(zip(placements, results)):
            status = "✓" if result.feasible else "✗"
            print(f"    {status} {placement['task_id']}")
            if result.feasible:
                print(f"        开始: {result.actual_start.strftime('%H:%M:%S')}, "
                      f"机动: {result.slew_time:.1f}s")
            else:
                print(f"        原因: {result.conflict_reason}")


def main():
    """主函数"""
    print("统一机动约束检查器集成示例")
    print("=" * 70)

    example_traditional_vs_maneuver()
    example_batch_check()

    print("\n" + "=" * 70)
    print("总结:")
    print("  1. 传统方式需要分别检查多个约束")
    print("  2. 统一机动方式只需1步即可获得完整结果")
    print("  3. 包含姿态切换、机动能力、时间窗口、发电功率评估")
    print("  4. 支持任务后姿态决策（对日/对地/动量卸载）")
    print("  5. 批量检查功能适合元启发式算法快速评估")
    print("=" * 70)


if __name__ == "__main__":
    main()
