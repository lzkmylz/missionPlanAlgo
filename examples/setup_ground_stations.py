"""
为场景中的地面站添加默认天线配置

Usage:
    python examples/setup_ground_stations.py \
        --scenario scenarios/large_scale_frequency.json \
        --output scenarios/large_scale_frequency_with_antennas.json
"""

import json
import argparse
from pathlib import Path


def add_default_antennas_to_ground_stations(scenario_path: str, output_path: str):
    """为场景中的地面站添加默认天线配置"""

    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)

    # 为每个地面站添加默认天线
    for gs in scenario.get('ground_stations', []):
        if not gs.get('antennas'):
            # 添加默认天线配置
            gs['antennas'] = [
                {
                    'id': f"{gs['id']}-ANT-01",
                    'name': '主天线',
                    'data_rate': 300.0,  # Mbps
                    'frequency_band': 'X',
                    'max_elevation': 90.0,
                    'min_elevation': 5.0
                },
                {
                    'id': f"{gs['id']}-ANT-02",
                    'name': '备用天线',
                    'data_rate': 150.0,  # Mbps
                    'frequency_band': 'S',
                    'max_elevation': 90.0,
                    'min_elevation': 5.0
                }
            ]
            print(f"为地面站 {gs['id']} 添加了 {len(gs['antennas'])} 个默认天线")

    # 保存修改后的场景
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    print(f"\n修改后的场景已保存: {output_path}")

    # 打印统计
    total_antennas = sum(len(gs.get('antennas', [])) for gs in scenario.get('ground_stations', []))
    print(f"地面站总数: {len(scenario.get('ground_stations', []))}")
    print(f"天线总数: {total_antennas}")


def main():
    parser = argparse.ArgumentParser(description='为地面站添加默认天线配置')
    parser.add_argument('--scenario', '-s', required=True, help='输入场景文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出场景文件路径')

    args = parser.parse_args()
    add_default_antennas_to_ground_stations(args.scenario, args.output)


if __name__ == '__main__':
    main()
