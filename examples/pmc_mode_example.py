"""
主动前向推扫模式（PMC）使用示例

展示如何使用PMC模式进行高分辨率成像任务规划。

PMC模式特点：
- 通过俯仰机动降低等效地面速度
- 延长积分时间，提高信噪比（SNR）
- 适用于低光照条件下的精细成像

主动反向推扫模式特点：
- 通过俯仰机动增加等效地面速度
- 缩短积分时间，降低信噪比，但增加覆盖范围
- 适用于大范围快速扫描场景
"""

from datetime import datetime
from core.models.satellite import Satellite, SatelliteType, Orbit
from core.models.target import Target, TargetType
from core.models.imaging_mode import create_pmc_mode_config, ImagingMode
from core.models.pmc_config import (
    PitchMotionCompensationConfig,
    PMC_CONFIG_25PERCENT,
    PMC_REVERSE_CONFIG_25PERCENT,
    create_reverse_pmc_config,
)
from core.models.payload_config import PayloadConfiguration
from scheduler.constraints.pmc_constraint_checker import PMCConstraintChecker, PMCCandidate
from core.dynamics.pmc_calculator import PMCCalculator, calculate_pmc_parameters


def demo_pmc_config():
    """演示PMC配置创建"""
    print("=" * 60)
    print("PMC模式配置示例")
    print("=" * 60)

    # 创建自定义前向PMC模式配置
    pmc_mode = create_pmc_mode_config(
        base_resolution_m=0.5,
        base_swath_width_m=15000,
        speed_reduction_ratio=0.25,  # 25%降速
        mode_type='optical',
        max_duration_s=30.0,
    )

    print(f"\n创建的前向PMC模式:")
    print(f"  分辨率: {pmc_mode.resolution_m}m")
    print(f"  幅宽: {pmc_mode.swath_width_m}m")
    print(f"  功耗: {pmc_mode.power_consumption_w}W")
    print(f"  是否PMC模式: {pmc_mode.is_pmc_mode()}")

    pmc_params = pmc_mode.get_pmc_params()
    print(f"\nPMC参数:")
    print(f"  方向: {pmc_params.get('direction', 'forward')}")
    print(f"  降速比: {pmc_params['speed_reduction_ratio']}")
    print(f"  俯仰角速度: {pmc_params['pitch_rate_dps']:.3f}°/s")
    print(f"  积分时间增益: {pmc_params['integration_time_gain']:.2f}x")


def demo_reverse_pmc_config():
    """演示反向PMC配置创建"""
    print("\n" + "=" * 60)
    print("反向PMC模式配置示例")
    print("=" * 60)

    # 创建反向PMC模式配置
    reverse_pmc_config = create_reverse_pmc_config(
        speed_ratio=0.25,  # 25%降速
        max_roll_angle_deg=35.0,
        max_pitch_angle_deg=20.0,
    )

    print(f"\n创建的反向PMC配置:")
    print(f"  方向: {reverse_pmc_config.direction}")
    print(f"  降速比: {reverse_pmc_config.speed_reduction_ratio}")
    print(f"  积分时间增益: {reverse_pmc_config.integration_time_gain:.2f}x (>1表示延长)")

    # 使用模板创建
    reverse_pmc_50 = PitchMotionCompensationConfig(
        speed_reduction_ratio=0.50,
        direction='reverse',
        max_roll_angle_deg=30.0,
    )

    print(f"\n50%降速反向PMC配置:")
    print(f"  方向: {reverse_pmc_50.direction}")
    print(f"  降速比: {reverse_pmc_50.speed_reduction_ratio}")
    print(f"  积分时间增益: {reverse_pmc_50.integration_time_gain:.2f}x")


def demo_pmc_performance():
    """演示PMC性能计算"""
    print("\n" + "=" * 60)
    print("前向PMC性能计算示例")
    print("=" * 60)

    orbit_altitude = 500000  # 500km

    print(f"\n轨道高度: {orbit_altitude/1000:.0f}km")
    print("\n不同降速比的性能对比:")
    print("-" * 70)
    print(f"{'降速比':<10} {'俯仰率(°/s)':<15} {'SNR增益(dB)':<15} {'地速(m/s)':<15}")
    print("-" * 70)

    for reduction in [0.10, 0.25, 0.50, 0.75]:
        params = calculate_pmc_parameters(orbit_altitude, reduction, direction='forward')
        print(f"{reduction*100:.0f}%{'':<7} "
              f"{params['pitch_rate_dps']:.3f}{'':<10} "
              f"{params['snr_change_db']:+.2f}{'':<10} "
              f"{params['ground_velocity_m_s']:.1f}")


def demo_reverse_pmc_performance():
    """演示反向PMC性能计算"""
    print("\n" + "=" * 60)
    print("反向PMC性能计算示例")
    print("=" * 60)

    orbit_altitude = 500000  # 500km

    print(f"\n轨道高度: {orbit_altitude/1000:.0f}km")
    print("\n不同降速比的性能对比（前向和反向都是降速成像）:")
    print("-" * 70)
    print(f"{'降速比':<10} {'俯仰率(°/s)':<15} {'SNR增益(dB)':<15} {'地速(m/s)':<15}")
    print("-" * 70)

    for speed_ratio in [0.10, 0.25, 0.50]:
        params = calculate_pmc_parameters(orbit_altitude, speed_ratio, direction='reverse')
        print(f"{speed_ratio*100:.0f}%{'':<7} "
              f"{params['pitch_rate_dps']:.3f}{'':<10} "
              f"{params['snr_change_db']:+.2f}{'':<10} "
              f"{params['ground_velocity_m_s']:.1f}")


def demo_pmc_payload_config():
    """演示载荷PMC配置"""
    print("\n" + "=" * 60)
    print("载荷PMC配置示例")
    print("=" * 60)

    # 创建支持PMC的载荷配置
    from core.models.imaging_mode import (
        OPTICAL_PUSH_BROOM_HIGH_RES,
        OPTICAL_PMC_25PERCENT,
        OPTICAL_REVERSE_PMC_25PERCENT,
    )

    payload = PayloadConfiguration(
        payload_type='optical',
        default_mode='push_broom',
        modes={
            'push_broom': OPTICAL_PUSH_BROOM_HIGH_RES,
            'pmc_25percent': OPTICAL_PMC_25PERCENT,
            'reverse_pmc_25percent': OPTICAL_REVERSE_PMC_25PERCENT,
        },
        description='支持前向和反向PMC模式的光学载荷'
    )

    print(f"\n载荷配置:")
    print(f"  类型: {payload.payload_type}")
    print(f"  支持模式: {payload.get_mode_names()}")
    print(f"  PMC模式: {payload.get_pmc_modes()}")
    print(f"  是否有PMC: {payload.has_pmc_mode()}")

    # 获取前向PMC配置
    pmc_config = payload.get_pmc_config('pmc_25percent')
    if pmc_config:
        print(f"\n前向PMC配置详情:")
        print(f"  降速比: {pmc_config.speed_reduction_ratio}")
        print(f"  方向: {pmc_config.direction}")
        print(f"  最大俯仰角: {pmc_config.max_pitch_angle_deg}°")

    # 获取反向PMC配置
    reverse_pmc_config = payload.get_pmc_config('reverse_pmc_25percent')
    if reverse_pmc_config:
        print(f"\n反向PMC配置详情:")
        print(f"  降速比: {reverse_pmc_config.speed_reduction_ratio}")
        print(f"  方向: {reverse_pmc_config.direction}")
        print(f"  最大俯仰角: {reverse_pmc_config.max_pitch_angle_deg}°")
        print(f"  积分时间增益: {reverse_pmc_config.integration_time_gain:.2f}x")


def demo_target_requirements():
    """演示目标任务需求"""
    print("\n" + "=" * 60)
    print("目标任务需求示例")
    print("=" * 60)

    # 创建需要前向PMC模式的目标
    target_pmc = Target(
        id='TGT-PMC-001',
        name='高分辨率暗弱目标',
        target_type=TargetType.POINT,
        longitude=116.4,
        latitude=39.9,
        required_imaging_mode='forward_pushbroom_pmc',
        pmc_priority=2,
        pmc_speed_reduction_range=(0.2, 0.5),
        resolution_required=0.5
    )

    # 创建需要反向PMC模式的目标（大范围快速扫描）
    target_reverse_pmc = Target(
        id='TGT-RPMC-001',
        name='大范围快速扫描目标',
        target_type=TargetType.POINT,
        longitude=117.0,
        latitude=40.0,
        required_imaging_mode='reverse_pushbroom_pmc',
        pmc_priority=1,
        pmc_speed_reduction_range=(0.2, 0.5),
        resolution_required=2.0
    )

    # 创建只需要光学卫星的目标
    target_optical = Target(
        id='TGT-OPT-001',
        name='普通光学目标',
        target_type=TargetType.POINT,
        longitude=116.5,
        latitude=40.0,
        required_satellite_type='optical',
        resolution_required=2.0
    )

    print(f"\n目标1 (前向PMC需求):")
    print(f"  名称: {target_pmc.name}")
    print(f"  需要PMC: {target_pmc.requires_pmc()}")
    print(f"  PMC优先级: {target_pmc.pmc_priority}")
    print(f"  要求模式: {target_pmc.required_imaging_mode}")
    print(f"  降速比范围: {target_pmc.pmc_speed_reduction_range}")

    print(f"\n目标2 (反向PMC需求):")
    print(f"  名称: {target_reverse_pmc.name}")
    print(f"  需要PMC: {target_reverse_pmc.requires_pmc()}")
    print(f"  PMC优先级: {target_reverse_pmc.pmc_priority}")
    print(f"  要求模式: {target_reverse_pmc.required_imaging_mode}")
    print(f"  降速比范围: {target_reverse_pmc.pmc_speed_reduction_range}")

    print(f"\n目标3 (光学卫星需求):")
    print(f"  名称: {target_optical.name}")
    print(f"  要求卫星类型: {target_optical.required_satellite_type}")
    print(f"  兼容光学: {target_optical.check_satellite_compatibility('optical', 'push_broom')}")
    print(f"  兼容SAR: {target_optical.check_satellite_compatibility('sar', 'stripmap')}")


def demo_pmc_constraint_check():
    """演示前向PMC约束检查"""
    print("\n" + "=" * 60)
    print("前向PMC约束检查示例")
    print("=" * 60)

    # 创建卫星（简化配置）
    sat = Satellite(
        id='SAT-OPT-001',
        name='光学卫星',
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(semi_major_axis=6871000.0)  # 500km
    )

    # 创建目标
    target = Target(
        id='TGT-001',
        name='测试目标',
        target_type=TargetType.POINT,
        longitude=116.4,
        latitude=39.9
    )

    # 创建前向PMC配置
    pmc_config = PitchMotionCompensationConfig(
        speed_reduction_ratio=0.25,
        direction='forward',
        max_pitch_angle_deg=20.0,
        max_roll_angle_deg=30.0
    )

    # 创建候选任务
    candidate = PMCCandidate(
        sat_id=sat.id,
        satellite=sat,
        target=target,
        imaging_start=datetime.now(),
        imaging_duration_s=10.0,
        pmc_config=pmc_config
    )

    # 执行约束检查
    checker = PMCConstraintChecker()
    result = checker.check_pmc_feasibility(candidate)

    print(f"\n前向PMC约束检查结果:")
    print(f"  可行: {result.feasible}")
    print(f"  SNR增益: {result.snr_change_db:+.2f}dB")
    print(f"  等效积分时间: {result.effective_integration_time_s:.2f}s")
    print(f"  估算能耗: {result.estimated_energy_wh:.2f}Wh")

    if not result.feasible:
        print(f"  原因: {result.reason}")
        if result.pitch_violations:
            print(f"  俯仰角违反: {len(result.pitch_violations)}处")
        if result.roll_violations:
            print(f"  滚转角违反: {len(result.roll_violations)}处")


def demo_reverse_pmc_constraint_check():
    """演示反向PMC约束检查"""
    print("\n" + "=" * 60)
    print("反向PMC约束检查示例")
    print("=" * 60)

    # 创建卫星（简化配置）
    sat = Satellite(
        id='SAT-OPT-002',
        name='光学卫星2',
        sat_type=SatelliteType.OPTICAL_1,
        orbit=Orbit(semi_major_axis=6871000.0)  # 500km
    )

    # 创建大范围扫描目标
    target = Target(
        id='TGT-002',
        name='大范围扫描目标',
        target_type=TargetType.POINT,
        longitude=117.0,
        latitude=40.0
    )

    # 创建反向PMC配置
    reverse_pmc_config = PitchMotionCompensationConfig(
        speed_reduction_ratio=0.25,
        direction='reverse',
        max_pitch_angle_deg=20.0,
        max_roll_angle_deg=30.0
    )

    # 创建候选任务
    candidate = PMCCandidate(
        sat_id=sat.id,
        satellite=sat,
        target=target,
        imaging_start=datetime.now(),
        imaging_duration_s=10.0,
        pmc_config=reverse_pmc_config
    )

    # 执行约束检查
    checker = PMCConstraintChecker()
    result = checker.check_pmc_feasibility(candidate)

    print(f"\n反向PMC约束检查结果:")
    print(f"  可行: {result.feasible}")
    print(f"  SNR增益: {result.snr_change_db:+.2f}dB")
    print(f"  等效积分时间: {result.effective_integration_time_s:.2f}s")
    print(f"  估算能耗: {result.estimated_energy_wh:.2f}Wh")

    if not result.feasible:
        print(f"  原因: {result.reason}")
        if result.pitch_violations:
            print(f"  俯仰角违反: {len(result.pitch_violations)}处")
        if result.roll_violations:
            print(f"  滚转角违反: {len(result.roll_violations)}处")


if __name__ == '__main__':
    # 运行所有演示
    demo_pmc_config()
    demo_reverse_pmc_config()
    demo_pmc_performance()
    demo_reverse_pmc_performance()
    demo_pmc_payload_config()
    demo_target_requirements()
    demo_pmc_constraint_check()
    demo_reverse_pmc_constraint_check()

    print("\n" + "=" * 60)
    print("PMC模式演示完成（包含前向和反向模式）")
    print("=" * 60)
