"""
多成像模式卫星配置示例

演示如何使用新的 payload_config 格式配置和使用多成像模式卫星。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    Satellite, SatelliteType, SatelliteCapabilities,
    ImagingModeConfig, PayloadConfiguration,
    create_optical_payload_config, create_sar_payload_config,
)
from core.constants import (
    OPTICAL_PUSH_BROOM_POWER_W,
    SAR_STRIPMAP_POWER_W,
    SAR_SPOTLIGHT_POWER_W,
)


def example_optical_satellite():
    """示例：配置光学卫星（被动推扫模式）"""
    print("=" * 60)
    print("示例1: 光学卫星配置")
    print("=" * 60)

    # 使用辅助函数创建光学载荷配置
    payload_config = create_optical_payload_config(
        resolution_m=0.5,
        swath_width_m=15000,
        power_consumption_w=150.0,
        data_rate_mbps=200.0,
        spectral_bands=["PAN", "RGB", "NIR"],
    )

    print(f"载荷类型: {payload_config.payload_type}")
    print(f"默认模式: {payload_config.default_mode}")
    print(f"描述: {payload_config.description}")
    print()

    # 获取推扫模式配置
    push_broom = payload_config.get_mode_config("push_broom")
    print("推扫模式配置:")
    print(f"  - 分辨率: {push_broom.resolution_m}m")
    print(f"  - 幅宽: {push_broom.swath_width_m/1000:.1f}km")
    print(f"  - 功耗: {push_broom.power_consumption_w}W")
    print(f"  - 数据率: {push_broom.data_rate_mbps}Mbps")
    print(f"  - 成像时长: {push_broom.min_duration_s}s - {push_broom.max_duration_s}s")
    print()

    # 计算10秒成像的资源消耗
    duration = 10.0
    energy = push_broom.get_energy_consumption_wh(duration)
    data = push_broom.get_data_volume_gb(duration)
    print(f"10秒成像资源消耗:")
    print(f"  - 能耗: {energy:.4f}Wh")
    print(f"  - 数据量: {data:.4f}GB")
    print()


def example_sar_satellite():
    """示例：配置SAR卫星（多模式）"""
    print("=" * 60)
    print("示例2: SAR卫星多模式配置")
    print("=" * 60)

    # 使用辅助函数创建SAR载荷配置（使用默认模式）
    payload_config = create_sar_payload_config()

    print(f"载荷类型: {payload_config.payload_type}")
    print(f"默认模式: {payload_config.default_mode}")
    print(f"可用模式: {', '.join(payload_config.get_mode_names())}")
    print()

    # 比较不同模式的功耗和数据率
    print("各模式功耗和数据率对比:")
    for mode_name in payload_config.get_mode_names():
        mode = payload_config.get_mode_config(mode_name)
        print(f"  {mode_name:20s}: {mode.power_consumption_w:6.1f}W, {mode.data_rate_mbps:6.1f}Mbps, "
              f"分辨率{mode.resolution_m}m")
    print()

    # 展示不同模式的适用场景
    print("模式选择建议:")
    print(f"  - 最佳分辨率: {payload_config.get_best_resolution_mode()}")
    print(f"  - 最大幅宽: {payload_config.get_best_swath_mode()}")
    print()


def example_mixed_satellite_capabilities():
    """示例：在 SatelliteCapabilities 中使用 payload_config"""
    print("=" * 60)
    print("示例3: 卫星能力配置中使用 payload_config")
    print("=" * 60)

    # 创建载荷配置
    payload_config = create_sar_payload_config()

    # 创建卫星能力配置
    capabilities = SatelliteCapabilities(
        payload_config=payload_config,
        storage_capacity=1000,  # GB
        power_capacity=3000,    # Wh
        max_roll_angle=45.0,
        max_pitch_angle=30.0,
    )

    # 使用 SatelliteCapabilities 的新方法获取模式特定参数
    print("通过 SatelliteCapabilities 访问成像模式参数:")
    # 获取默认模式名称
    default_mode = capabilities.payload_config.default_mode
    print(f"  - 默认模式({default_mode})分辨率: {capabilities.get_mode_resolution(default_mode)}m")
    print(f"  - 默认模式({default_mode})幅宽: {capabilities.get_mode_swath_width(default_mode)/1000:.1f}km")
    print(f"  - 聚束模式功耗: {capabilities.get_mode_power_consumption('spotlight')}W")
    print(f"  - 聚束模式数据率: {capabilities.get_mode_data_rate('spotlight')}Mbps")
    print()

    # 获取时长约束
    constraints = capabilities.get_mode_duration_constraints('spotlight')
    print(f"聚束模式时长约束:")
    print(f"  - 最短: {constraints['min_duration']}s")
    print(f"  - 最长: {constraints['max_duration']}s")
    print()


def example_migration_from_legacy():
    """示例：从旧格式迁移到新格式"""
    print("=" * 60)
    print("示例4: 旧格式自动迁移到新格式")
    print("=" * 60)

    # 模拟旧格式的 capabilities
    old_capabilities = SatelliteCapabilities(
        resolution=0.5,
        swath_width=15000,
        data_rate=200.0,
        imaging_modes=["push_broom"],
    )

    print("旧格式配置:")
    print(f"  - resolution: {old_capabilities.resolution}m (已废弃)")
    print(f"  - swath_width: {old_capabilities.swath_width}m (已废弃)")
    print()

    # __post_init__ 会自动初始化 payload_config
    print("自动生成的 payload_config:")
    print(f"  - payload_type: {old_capabilities.payload_config.payload_type}")
    print(f"  - modes: {', '.join(old_capabilities.payload_config.get_mode_names())}")
    print()

    # 新方法自动使用 payload_config
    print("通过新方法访问:")
    default_mode = old_capabilities.payload_config.default_mode
    print(f"  - get_mode_resolution('{default_mode}'): {old_capabilities.get_mode_resolution(default_mode)}m")
    print(f"  - get_mode_swath_width('{default_mode}'): {old_capabilities.get_mode_swath_width(default_mode)/1000:.1f}km")
    print(f"  - get_mode_power_consumption('{default_mode}'): {old_capabilities.get_mode_power_consumption(default_mode)}W")
    print()


def example_json_serialization():
    """示例：JSON序列化和反序列化"""
    print("=" * 60)
    print("示例5: JSON序列化和反序列化")
    print("=" * 60)

    import json

    # 创建卫星
    satellite = Satellite(
        id="SAT-001",
        name="测试卫星",
        sat_type=SatelliteType.SAR_1,
        capabilities=SatelliteCapabilities(
            payload_config=create_sar_payload_config(),
        ),
    )

    # 序列化为JSON
    data = satellite.to_dict()
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    print("序列化后的JSON片段:")
    print(json_str[:500] + "...")
    print()

    # 从JSON反序列化
    restored = Satellite.from_dict(data)
    print("反序列化验证:")
    print(f"  - ID: {restored.id}")
    print(f"  - 载荷类型: {restored.capabilities.payload_config.payload_type}")
    print(f"  - 模式数: {len(restored.capabilities.payload_config.modes)}")
    print()


if __name__ == "__main__":
    print("\n多成像模式卫星配置示例\n")
    print("本示例演示新的 payload_config 格式的使用方法\n")

    example_optical_satellite()
    example_sar_satellite()
    example_mixed_satellite_capabilities()
    example_migration_from_legacy()
    example_json_serialization()

    print("=" * 60)
    print("所有示例执行完毕!")
    print("=" * 60)
