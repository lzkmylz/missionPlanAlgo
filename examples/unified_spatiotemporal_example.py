"""
统一时空约束检查器使用示例

演示如何使用 UnifiedSpatiotemporalChecker 替代原来的分散约束检查。

对比：
- 旧方式：分别调用 _check_slew、_has_time_conflict、手动检查窗口边界
- 新方式：一次调用 check_task_placement，统一返回所有约束结果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from core.models import Mission
from scheduler.constraints import (
    UnifiedSpatiotemporalChecker,
    SpatiotemporalCheckResult,
    SlewConstraintChecker
)


def example_basic_usage():
    """基本使用示例"""
    print("=" * 70)
    print("示例1: 统一时空约束检查的基本使用")
    print("=" * 70)

    # 加载场景
    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)
    print(f"\n场景加载成功: {len(mission.satellites)}颗卫星, {len(mission.targets)}个目标")

    # 创建统一时空约束检查器
    slew_checker = SlewConstraintChecker(mission)
    for sat in mission.satellites:
        slew_checker.initialize_satellite(sat)

    checker = UnifiedSpatiotemporalChecker(
        mission=mission,
        slew_checker=slew_checker,
        use_simplified_slew=False
    )

    # 示例：检查一个任务
    satellite = mission.satellites[0]
    target = mission.targets[0]
    window_start = mission.start_time + timedelta(minutes=10)
    window_end = window_start + timedelta(minutes=5)
    imaging_duration = 30.0  # 30秒成像

    print(f"\n检查任务:")
    print(f"  卫星: {satellite.id}")
    print(f"  目标: {target.id}")
    print(f"  窗口: {window_start} ~ {window_end}")
    print(f"  成像时长: {imaging_duration}秒")

    result = checker.check_task_placement(
        satellite_id=satellite.id,
        target=target,
        window_start=window_start,
        window_end=window_end,
        imaging_duration=imaging_duration,
        task_id="TASK-001"
    )

    print(f"\n检查结果:")
    print(f"  可行: {result.feasible}")
    print(f"  实际开始: {result.actual_start}")
    print(f"  实际结束: {result.actual_end}")
    print(f"  机动角度: {result.slew_angle}度")
    print(f"  机动时间: {result.slew_time}秒")
    if result.conflict_reason:
        print(f"  冲突原因: {result.conflict_reason}")

    # 如果可行，提交任务
    if result.feasible:
        checker.commit_task(
            satellite_id=satellite.id,
            task_id="TASK-001",
            target_id=target.id,
            actual_start=result.actual_start,
            actual_end=result.actual_end,
            target=target
        )
        print(f"\n任务已提交到卫星 {satellite.id}")


def example_time_conflict():
    """时间冲突示例"""
    print("\n" + "=" * 70)
    print("示例2: 检测时间冲突")
    print("=" * 70)

    scenario_path = "scenarios/large_scale_frequency.json"
    mission = Mission.load(scenario_path)

    checker = UnifiedSpatiotemporalChecker(
        mission=mission,
        use_simplified_slew=True
    )

    satellite = mission.satellites[0]
    target1 = mission.targets[0]
    target2 = mission.targets[1]

    # 第一个任务
    window1_start = mission.start_time + timedelta(minutes=10)
    window1_end = window1_start + timedelta(minutes=5)

    result1 = checker.check_task_placement(
        satellite_id=satellite.id,
        target=target1,
        window_start=window1_start,
        window_end=window1_end,
        imaging_duration=60.0,
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
        print(f"任务1已调度: {result1.actual_start} ~ {result1.actual_end}")

    # 第二个任务（与第一个冲突）
    window2_start = mission.start_time + timedelta(minutes=10, seconds=30)
    window2_end = window2_start + timedelta(minutes=5)

    result2 = checker.check_task_placement(
        satellite_id=satellite.id,
        target=target2,
        window_start=window2_start,
        window_end=window2_end,
        imaging_duration=60.0,
        task_id="TASK-002"
    )

    print(f"\n任务2检查结果:")
    print(f"  可行: {result2.feasible}")
    print(f"  实际开始: {result2.actual_start}")
    print(f"  实际结束: {result2.actual_end}")
    if result2.conflict_reason:
        print(f"  冲突原因: {result2.conflict_reason}")
        print(f"  与任务冲突: {result2.conflict_with}")


def example_comparison_old_vs_new():
    """对比旧方式和新方式"""
    print("\n" + "=" * 70)
    print("示例3: 旧方式 vs 新方式对比")
    print("=" * 70)

    print("""
【旧方式】分别检查三个约束:
```python
# 1. 检查机动约束
slew_result = _slew_checker.check_slew_feasibility(...)
if not slew_result.feasible:
    return False, "Slew not feasible"

# 2. 计算实际开始时间
actual_start = max(window_start, slew_result.actual_start)
actual_end = actual_start + timedelta(seconds=imaging_duration)

# 3. 检查窗口边界
if actual_end > window_end:
    return False, "Window too short"

# 4. 检查时间冲突
if _has_time_conflict(sat_id, actual_start, actual_end):
    return False, "Time conflict"
```

【新方式】统一检查:
```python
result = checker.check_task_placement(
    satellite_id=sat_id,
    target=target,
    window_start=window_start,
    window_end=window_end,
    imaging_duration=imaging_duration
)

if not result.feasible:
    return False, result.conflict_reason

# 一次调用获得所有信息
actual_start = result.actual_start
actual_end = result.actual_end
slew_angle = result.slew_angle
slew_time = result.slew_time
```

【优势】
1. 代码更简洁：从4步减少到1步
2. 逻辑更清晰：统一入口，统一返回
3. 诊断更精确：返回具体冲突原因和冲突任务
4. 维护更容易：约束逻辑集中管理
""")


def main():
    """主函数"""
    print("统一时空约束检查器使用示例")
    print("=" * 70)

    example_basic_usage()
    example_time_conflict()
    example_comparison_old_vs_new()

    print("\n" + "=" * 70)
    print("提示: 在实际调度器中，可以使用 UnifiedSpatiotemporalChecker")
    print("      替代分散的 _check_slew、_has_time_conflict 等方法")
    print("=" * 70)


if __name__ == "__main__":
    main()
