#!/usr/bin/env python3
"""
更新场景文件，为卫星和目标添加分辨率约束配置

策略：
1. 光学卫星：
   - push_broom: 1.0m分辨率，15000m幅宽
   - push_broom_high_res: 0.5m分辨率，7500m幅宽

2. SAR卫星：
   - stripmap: 3.0m分辨率，30000m幅宽
   - spotlight: 1.0m分辨率，10000m幅宽
   - sliding_spotlight: 2.0m分辨率，20000m幅宽

3. 目标分辨率需求（基于优先级）：
   - 优先级1-3: 0.5m（高分辨率需求）
   - 优先级4-6: 1.0m（标准分辨率需求）
   - 优先级7+: 3.0m或无要求（低分辨率需求）
"""

import json
import sys
from pathlib import Path

# 卫星成像模式分辨率配置
OPTICAL_MODE_DETAILS = [
    {
        "mode_id": "push_broom",
        "resolution": 1.0,
        "swath_width": 15000.0
    },
    {
        "mode_id": "push_broom_high_res",
        "resolution": 0.5,
        "swath_width": 7500.0
    }
]

SAR_MODE_DETAILS = [
    {
        "mode_id": "stripmap",
        "resolution": 3.0,
        "swath_width": 30000.0
    },
    {
        "mode_id": "spotlight",
        "resolution": 1.0,
        "swath_width": 10000.0
    },
    {
        "mode_id": "sliding_spotlight",
        "resolution": 2.0,
        "swath_width": 20000.0
    }
]


def get_target_resolution_required(priority: int) -> float:
    """
    根据目标优先级确定所需分辨率

    Args:
        priority: 目标优先级（1-10，数字越小优先级越高）

    Returns:
        所需分辨率（米）
    """
    if priority <= 3:
        return 0.5  # 高优先级：需要高分辨率
    elif priority <= 6:
        return 1.0  # 中优先级：标准分辨率
    else:
        return 3.0  # 低优先级：低分辨率即可


def update_satellite_capabilities(satellite: dict) -> dict:
    """
    更新卫星能力配置，添加imaging_mode_details

    Args:
        satellite: 卫星配置字典

    Returns:
        更新后的卫星配置
    """
    sat_type = satellite.get("sat_type", "").lower()
    capabilities = satellite.get("capabilities", {})

    if sat_type == "optical":
        capabilities["imaging_mode_details"] = OPTICAL_MODE_DETAILS
    elif sat_type == "sar":
        capabilities["imaging_mode_details"] = SAR_MODE_DETAILS

    satellite["capabilities"] = capabilities
    return satellite


def update_target_resolution(target: dict) -> dict:
    """
    更新目标配置，添加resolution_required

    Args:
        target: 目标配置字典

    Returns:
        更新后的目标配置
    """
    priority = target.get("priority", 5)
    target["resolution_required"] = get_target_resolution_required(priority)
    return target


def update_scenario(input_path: str, output_path: str) -> dict:
    """
    更新场景文件，添加分辨率约束配置

    Args:
        input_path: 输入场景文件路径
        output_path: 输出场景文件路径

    Returns:
        更新后的场景数据
    """
    # 读取场景文件
    with open(input_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)

    # 更新版本和描述
    scenario["version"] = "2.1"
    scenario["description"] = scenario.get("description", "") + "，含分辨率约束"

    # 更新卫星配置
    satellites = scenario.get("satellites", [])
    for sat in satellites:
        update_satellite_capabilities(sat)

    print(f"已更新 {len(satellites)} 颗卫星的成像模式分辨率配置")

    # 更新目标配置
    targets = scenario.get("targets", [])
    resolution_stats = {0.5: 0, 1.0: 0, 3.0: 0}

    for target in targets:
        update_target_resolution(target)
        resolution_stats[target["resolution_required"]] += 1

    print(f"已更新 {len(targets)} 个目标的分辨率需求配置")
    print(f"  - 0.5m分辨率需求: {resolution_stats[0.5]} 个目标")
    print(f"  - 1.0m分辨率需求: {resolution_stats[1.0]} 个目标")
    print(f"  - 3.0m分辨率需求: {resolution_stats[3.0]} 个目标")

    # 保存更新后的场景文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    print(f"\n场景文件已保存到: {output_path}")
    return scenario


def main():
    """主函数"""
    # 默认路径
    default_input = "scenarios/large_scale_frequency.json"
    default_output = "scenarios/large_scale_frequency.json"

    # 解析命令行参数
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    # 确保输入文件存在
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    print(f"正在更新场景文件: {input_path}")
    print("-" * 50)

    try:
        update_scenario(input_path, output_path)
        print("\n✅ 场景文件更新成功!")
    except Exception as e:
        print(f"\n❌ 更新失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
