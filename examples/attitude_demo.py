"""
姿态角计算演示

直接演示 AttitudeCalculator 的使用
"""

from datetime import datetime, timezone
from core.models.satellite import Satellite, SatelliteType, Orbit, OrbitType
from core.models.target import Target, TargetType
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType


def main():
    print("=" * 60)
    print("姿态角计算器演示")
    print("=" * 60)

    # 创建姿态角计算器（使用SGP4传播器）
    calculator = AttitudeCalculator(propagator_type=PropagatorType.SGP4)
    print(f"\n使用传播器: {calculator.propagator_type.value}")

    # 创建卫星（带TLE）
    satellite = Satellite(
        id="sat_001",
        name="Test Satellite",
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(
            orbit_type=OrbitType.SSO,
            altitude=500000.0,
            inclination=97.4,
        ),
        tle_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        tle_line2="2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537",
    )
    print(f"卫星: {satellite.name}")

    # 定义多个目标
    targets = [
        Target(id="t1", name="Target 1", latitude=0.0, longitude=0.0, target_type=TargetType.POINT),
        Target(id="t2", name="Target 2", latitude=30.0, longitude=30.0, target_type=TargetType.POINT),
        Target(id="t3", name="Target 3", latitude=45.0, longitude=-90.0, target_type=TargetType.POINT),
        Target(id="t4", name="Target 4", latitude=-30.0, longitude=120.0, target_type=TargetType.POINT),
    ]

    # 计算姿态角
    test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    print(f"\n成像时刻: {test_time.isoformat()}")
    print(f"坐标系: LVLH (Local Vertical Local Horizontal)")
    print(f"偏航模式: 零偏航")

    print("\n" + "-" * 60)
    print(f"{'目标':<15} {'纬度':<8} {'经度':<8} {'滚转(°)':<10} {'俯仰(°)':<10} {'偏航(°)':<10}")
    print("-" * 60)

    for target in targets:
        try:
            attitude = calculator.calculate_attitude(satellite, target, test_time)
            print(f"{target.name:<15} {target.latitude:>7.1f}  {target.longitude:>7.1f}  "
                  f"{attitude.roll:>9.2f}  {attitude.pitch:>9.2f}  {attitude.yaw:>9.2f}")
        except Exception as e:
            print(f"{target.name:<15} {target.latitude:>7.1f}  {target.longitude:>7.1f}  计算失败: {e}")

    print("-" * 60)

    # 展示JSON输出
    print("\n姿态角JSON输出示例:")
    print("-" * 60)

    target = targets[0]
    attitude = calculator.calculate_attitude(satellite, target, test_time)

    import json
    output = {
        "task_id": "task_001",
        "satellite_id": satellite.id,
        "target_id": target.id,
        "imaging_time": test_time.isoformat(),
        "attitude": attitude.to_dict()
    }
    print(json.dumps(output, indent=2))

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)

    print("""
说明:
- 滚转角 (Roll): 绕X轴（飞行方向），控制侧摆
  正值: 向右侧摆, 负值: 向左侧摆

- 俯仰角 (Pitch): 绕Y轴（垂直轨道平面），控制前后斜视
  正值: 向前看, 负值: 向后看

- 偏航角 (Yaw): 绕Z轴（指向地心）
  本实现使用零偏航模式（固定为0）

- 坐标系: LVLH
  X: 飞行方向, Y: 轨道法向, Z: 指向地心
""")


if __name__ == "__main__":
    main()
