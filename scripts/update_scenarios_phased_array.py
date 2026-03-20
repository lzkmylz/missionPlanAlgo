#!/usr/bin/env python3
"""
批量更新场景文件，为所有卫星添加相控阵天线配置
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


def add_phased_array_to_satellite(satellite: Dict[str, Any]) -> bool:
    """为单个卫星添加相控阵配置，如果已存在则跳过"""
    capabilities = satellite.get('capabilities', {})

    # 检查是否已有 phased_array 配置
    if 'phased_array' in capabilities:
        return False  # 已存在，跳过

    # 添加默认相控阵配置
    capabilities['phased_array'] = {
        "_comment": "相控阵天线配置 - 用于并发成像+数传",
        "max_steering_angle_deg": 60.0,
        "max_steering_angle_comment": "相控阵天线最大扫描角 ±60度",
        "min_elevation_for_concurrent_deg": 30.0,
        "min_elevation_comment": "并发数传要求地面站最小仰角 30度",
        "max_concurrent_roll_deg": 30.0,
        "max_concurrent_roll_comment": "允许并发数传的最大滚转角 30度（60-30=30度余量用于地面站跟踪）",
        "data_rate_mbps": 300.0,
        "data_rate_comment": "数传速率 300Mbps - X波段相控阵对地数传"
    }

    satellite['capabilities'] = capabilities
    return True


def update_scenario_file(filepath: Path) -> int:
    """更新单个场景文件，返回更新的卫星数量"""
    print(f"\n处理文件: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        scenario = json.load(f)

    satellites = scenario.get('satellites', [])
    updated_count = 0

    for satellite in satellites:
        if add_phased_array_to_satellite(satellite):
            updated_count += 1
            print(f"  - 更新卫星: {satellite.get('id', 'unknown')}")

    # 保存更新后的文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    print(f"  完成: 更新了 {updated_count}/{len(satellites)} 颗卫星")
    return updated_count


def main():
    """主函数"""
    # 场景文件目录
    scenarios_dir = Path('/home/lz/missionPlanAlgo/scenarios')

    # 获取所有JSON场景文件
    scenario_files = list(scenarios_dir.glob('*.json'))

    print(f"找到 {len(scenario_files)} 个场景文件")

    total_updated = 0
    for filepath in scenario_files:
        try:
            count = update_scenario_file(filepath)
            total_updated += count
        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n总共更新了 {total_updated} 颗卫星的相控阵配置")


if __name__ == '__main__':
    main()
