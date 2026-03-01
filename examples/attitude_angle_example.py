"""
姿态角计算示例

演示如何在任务规划输出中获取俯仰角、滚转角和偏航角。

使用场景:
- 姿控系统验证
- 地面站指令生成
- 可视化展示
"""

from datetime import datetime, timezone
from core.models.satellite import Satellite, SatelliteType, Orbit, OrbitType
from core.models.target import Target, TargetType
from core.models.mission import Mission
from scheduler.greedy.greedy_scheduler import GreedyScheduler


def main():
    """运行姿态角计算示例"""

    print("=" * 60)
    print("卫星姿态角计算示例")
    print("=" * 60)

    # 1. 创建测试卫星（带TLE，用于SGP4传播）
    satellite = Satellite(
        id="sat_001",
        name="Test Optical Satellite",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,  # 500km
            inclination=97.4,
            epoch=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ),
        # 使用示例TLE（实际使用时替换为真实TLE）
        tle_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        tle_line2="2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537",
    )

    # 2. 创建测试目标
    targets = [
        Target(
            id="target_001",
            name="Beijing",
            latitude=39.9042,
            longitude=116.4074,
            target_type=TargetType.POINT,
            priority=1,
            time_window_start=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            time_window_end=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
        ),
        Target(
            id="target_002",
            name="Shanghai",
            latitude=31.2304,
            longitude=121.4737,
            target_type=TargetType.POINT,
            priority=2,
            time_window_start=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            time_window_end=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
        ),
    ]

    # 3. 创建任务
    mission = Mission(
        name="姿态角测试任务",
        satellites=[satellite],
        targets=targets,
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
    )

    # 4. 创建调度器（启用姿态角计算）
    scheduler = GreedyScheduler(config={
        'enable_attitude_calculation': True,  # 启用姿态角计算
        'propagator_type': 'sgp4',  # 使用SGP4传播器（含J2摄动）
    })

    # 5. 运行调度
    print("\n运行任务调度...")
    scheduler.initialize(mission)
    result = scheduler.schedule()

    # 6. 输出调度结果和姿态角
    print(f"\n调度完成:")
    print(f"  - 调度任务数: {len(result.scheduled_tasks)}")
    print(f"  - 未调度任务数: {len(result.unscheduled_tasks)}")

    if result.scheduled_tasks:
        print("\n" + "-" * 60)
        print("已调度任务姿态角（LVLH坐标系）:")
        print("-" * 60)
        print(f"{'任务ID':<15} {'卫星':<12} {'滚转(°)':<10} {'俯仰(°)':<10} {'偏航(°)':<10}")
        print("-" * 60)

        for task in result.scheduled_tasks:
            print(f"{task.task_id:<15} {task.satellite_id:<12} "
                  f"{task.roll_angle:>9.2f}  "
                  f"{task.pitch_angle:>9.2f}  "
                  f"{task.yaw_angle:>9.2f}")

        # 7. 输出JSON格式（可用于姿控系统）
        print("\n" + "-" * 60)
        print("JSON格式输出（用于姿控系统验证）:")
        print("-" * 60)

        import json
        task_data = []
        for task in result.scheduled_tasks:
            task_data.append({
                'task_id': task.task_id,
                'satellite_id': task.satellite_id,
                'target_id': task.target_id,
                'imaging_start': task.imaging_start.isoformat() if task.imaging_start else None,
                'imaging_end': task.imaging_end.isoformat() if task.imaging_end else None,
                'imaging_mode': task.imaging_mode,
                'attitude': {
                    'roll_angle': round(task.roll_angle, 4) if task.roll_angle is not None else None,
                    'pitch_angle': round(task.pitch_angle, 4) if task.pitch_angle is not None else None,
                    'yaw_angle': round(task.yaw_angle, 4) if task.yaw_angle is not None else None,
                    'coordinate_system': task.attitude_coordinate_system,
                }
            })

        print(json.dumps(task_data, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)

    # 8. 说明
    print("""
说明:
- 滚转角 (Roll): 绕X轴（飞行方向），正值为右侧摆
- 俯仰角 (Pitch): 绕Y轴（垂直轨道平面），正值为向前看
- 偏航角 (Yaw): 绕Z轴（指向地心），固定为0（零偏航模式）
- 坐标系: LVLH (Local Vertical Local Horizontal)
""")


if __name__ == "__main__":
    main()
