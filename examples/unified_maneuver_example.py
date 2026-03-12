"""
统一机动约束检查器使用示例

演示如何使用 UnifiedManeuverChecker 合并所有机动相关约束：
1. 姿态切换约束（对日/对地/成像姿态间的切换）
2. 机动能力约束（最大机动角度、角速度限制）
3. 时间窗口约束（与已调度任务的时间冲突）

与 UnifiedSpatiotemporalChecker 的区别：
- UnifiedManeuverChecker: 专注于机动相关约束（姿态切换、机动时间、时间冲突）
- UnifiedSpatiotemporalChecker: 包含所有约束（包括SAA）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from core.models import Mission
from core.dynamics.attitude_mode import AttitudeMode
from scheduler.constraints import (
    UnifiedManeuverChecker,
    ManeuverCheckResult,
    AttitudeManagementConfig
)


def example_basic_usage():
    """基本使用示例"""
    print("=" * 70)
    print("示例1: 统一机动约束检查的基本使用")
    print("=" * 70)

    # 加载场景
    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)
    print(f"\n场景加载成功: {len(mission.satellites)}颗卫星, {len(mission.targets)}个目标")

    # 创建姿态管理配置
    config = AttitudeManagementConfig(
        max_slew_rate=3.0,          # 最大角速度 3度/秒
        settling_time=5.0,          # 稳定时间 5秒
        idle_time_threshold=300.0,   # 空闲阈值 5分钟
        soc_threshold=0.30          # 电量阈值 30%
    )

    # 创建统一机动约束检查器（高精度要求：始终使用精确模式）
    checker = UnifiedManeuverChecker(
        mission=mission,
        config=config
    )

    # 示例：检查一个成像任务
    satellite = mission.satellites[0]
    target = mission.targets[0]
    window_start = mission.start_time + timedelta(minutes=10)
    window_end = window_start + timedelta(minutes=5)
    imaging_duration = 30.0  # 30秒成像

    # 获取卫星位置（简化：使用模拟位置）
    satellite_position = (7000000.0, 0.0, 0.0)  # ECEF位置（米）

    print(f"\n检查成像任务:")
    print(f"  卫星: {satellite.id}")
    print(f"  目标: {target.id} ({getattr(target, 'latitude', 0):.2f}°, {getattr(target, 'longitude', 0):.2f}°)")
    print(f"  窗口: {window_start} ~ {window_end}")
    print(f"  成像时长: {imaging_duration}秒")
    print(f"  起始姿态: {AttitudeMode.NADIR_POINTING.name}")
    print(f"  目标姿态: {AttitudeMode.IMAGING.name}")

    result = checker.check_maneuver_placement(
        satellite_id=satellite.id,
        target=target,
        window_start=window_start,
        window_end=window_end,
        imaging_duration=imaging_duration,
        satellite_position=satellite_position,
        task_id="TASK-001",
        from_mode=AttitudeMode.NADIR_POINTING,
        to_mode=AttitudeMode.IMAGING
    )

    print(f"\n检查结果:")
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

    # 如果可行，提交任务
    if result.feasible:
        checker.commit_task(
            satellite_id=satellite.id,
            task_id="TASK-001",
            target_id=target.id,
            actual_start=result.actual_start,
            actual_end=result.actual_end,
            target=target,
            end_mode=AttitudeMode.IMAGING
        )
        print(f"\n任务 TASK-001 已提交到卫星 {satellite.id}")

        # 决定任务后姿态
        next_task_time = window_end + timedelta(minutes=10)
        post_mode = checker.decide_post_task_attitude(
            satellite_id=satellite.id,
            next_task_time=next_task_time,
            current_time=result.actual_end,
            soc=0.8,  # 80%电量
            time_since_last_dump=3600  # 距离上次动量卸载1小时
        )
        print(f"任务后建议姿态: {post_mode.name}")


def example_time_conflict():
    """时间冲突示例"""
    print("\n" + "=" * 70)
    print("示例2: 检测时间冲突")
    print("=" * 70)

    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)

    # 高精度要求：始终使用精确模式
    config = AttitudeManagementConfig()
    checker = UnifiedManeuverChecker(
        mission=mission,
        config=config
    )

    satellite = mission.satellites[0]
    target1 = mission.targets[0]
    target2 = mission.targets[1]
    satellite_position = (7000000.0, 0.0, 0.0)

    # 第一个任务
    window1_start = mission.start_time + timedelta(minutes=10)
    window1_end = window1_start + timedelta(minutes=5)

    result1 = checker.check_maneuver_placement(
        satellite_id=satellite.id,
        target=target1,
        window_start=window1_start,
        window_end=window1_end,
        imaging_duration=60.0,
        satellite_position=satellite_position,
        task_id="TASK-001"
    )

    if result1.feasible:
        checker.commit_task(
            satellite_id=satellite.id,
            task_id="TASK-001",
            target_id=target1.id,
            actual_start=result1.actual_start,
            actual_end=result1.actual_end,
            target=target1
        )
        print(f"任务1已调度: {result1.actual_start.strftime('%H:%M:%S')} ~ {result1.actual_end.strftime('%H:%M:%S')}")

    # 第二个任务（与第一个冲突）
    window2_start = mission.start_time + timedelta(minutes=10, seconds=30)
    window2_end = window2_start + timedelta(minutes=5)

    result2 = checker.check_maneuver_placement(
        satellite_id=satellite.id,
        target=target2,
        window_start=window2_start,
        window_end=window2_end,
        imaging_duration=60.0,
        satellite_position=satellite_position,
        task_id="TASK-002"
    )

    print(f"\n任务2检查结果:")
    print(f"  可行: {result2.feasible}")
    if not result2.feasible:
        print(f"  冲突原因: {result2.conflict_reason}")
        print(f"  与任务冲突: {result2.conflict_with}")


def example_post_task_attitude():
    """任务后姿态决策示例"""
    print("\n" + "=" * 70)
    print("示例3: 任务后姿态决策")
    print("=" * 70)

    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)

    config = AttitudeManagementConfig(
        idle_time_threshold=300.0,   # 5分钟空闲阈值
        soc_threshold=0.30,          # 30%电量阈值
        momentum_dump_interval=14400.0  # 4小时动量卸载间隔
    )

    checker = UnifiedManeuverChecker(mission, config)
    satellite = mission.satellites[0]

    current_time = mission.start_time + timedelta(hours=1)

    test_cases = [
        {
            "name": "低电量场景",
            "next_task": current_time + timedelta(minutes=20),
            "soc": 0.25,
            "dump_time": 3600
        },
        {
            "name": "长空闲场景",
            "next_task": current_time + timedelta(minutes=10),
            "soc": 0.8,
            "dump_time": 3600
        },
        {
            "name": "需要动量卸载",
            "next_task": current_time + timedelta(minutes=20),
            "soc": 0.8,
            "dump_time": 14400  # 正好4小时
        },
        {
            "name": "短空闲场景",
            "next_task": current_time + timedelta(minutes=2),
            "soc": 0.8,
            "dump_time": 3600
        },
        {
            "name": "无后续任务",
            "next_task": None,
            "soc": 0.8,
            "dump_time": 3600
        }
    ]

    print(f"\n当前时间: {current_time}")
    print("-" * 50)

    for case in test_cases:
        post_mode = checker.decide_post_task_attitude(
            satellite_id=satellite.id,
            next_task_time=case["next_task"],
            current_time=current_time,
            soc=case["soc"],
            time_since_last_dump=case["dump_time"]
        )

        next_str = case["next_task"].strftime("%H:%M:%S") if case["next_task"] else "None"
        print(f"{case['name']:<15} | SOC:{case['soc']*100:>3.0f}% | 下次:{next_str:<8} | 姿态:{post_mode.name}")


def example_comparison():
    """对比：传统方式 vs 统一方式"""
    print("\n" + "=" * 70)
    print("示例4: 传统方式 vs 统一方式对比")
    print("=" * 70)

    print("""
【传统方式】分别处理各个约束:
```python
# 1. 计算机动角度（使用AttitudeCalculator）
attitude = attitude_calc.calculate_attitude(sat, target, time)
roll, pitch = attitude.roll, attitude.pitch

# 2. 检查机动角度限制
slew_angle = sqrt(roll^2 + pitch^2)
if slew_angle > max_slew_angle:
    return False, "Slew angle exceeded"

# 3. 计算机动时间
slew_time = max(abs(roll), abs(pitch)) / max_slew_rate + settling_time

# 4. 计算实际开始时间
actual_start = max(window_start, prev_end + slew_time)
actual_end = actual_start + imaging_duration

# 5. 检查窗口边界
if actual_end > window_end:
    return False, "Window too short"

# 6. 检查时间冲突
if has_time_conflict(sat_id, actual_start, actual_end):
    return False, "Time conflict"

# 7. 计算发电功率
power = power_calc.calculate_power(mode, sat_pos, time)
```

【统一方式】使用 UnifiedManeuverChecker:
```python
result = checker.check_maneuver_placement(
    satellite_id=sat_id,
    target=target,
    window_start=window_start,
    window_end=window_end,
    imaging_duration=imaging_duration,
    satellite_position=sat_pos,
    from_mode=AttitudeMode.NADIR_POINTING,
    to_mode=AttitudeMode.IMAGING
)

if not result.feasible:
    return False, result.conflict_reason

# 一次调用获得所有信息
actual_start = result.actual_start
actual_end = result.actual_end
slew_angle = result.slew_angle
slew_time = result.slew_time
roll_angle = result.roll_angle
pitch_angle = result.pitch_angle
power_before = result.power_before
power_after = result.power_after

# 提交任务
checker.commit_task(...)

# 决定任务后姿态
post_mode = checker.decide_post_task_attitude(...)
```

【优势】
1. 代码简洁：从7步减少到1步主要调用
2. 姿态感知：显式处理姿态模式切换
3. 功率评估：自动计算机动对发电的影响
4. 决策支持：内置任务后姿态决策逻辑
5. 维护容易：约束逻辑集中管理
""")


def main():
    """主函数"""
    print("统一机动约束检查器使用示例")
    print("=" * 70)

    example_basic_usage()
    example_time_conflict()
    example_post_task_attitude()
    example_comparison()

    print("\n" + "=" * 70)
    print("提示: UnifiedManeuverChecker 合并了所有机动相关约束：")
    print("      - 姿态切换约束（使用 AttitudeManager）")
    print("      - 机动能力约束（最大角度、角速度）")
    print("      - 时间窗口约束（窗口边界、任务冲突）")
    print("      - 发电功率评估（机动对功率的影响）")
    print("=" * 70)


if __name__ == "__main__":
    main()
