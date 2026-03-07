"""
计算卫星-地面站可见窗口并添加到缓存

此脚本演示如何将卫星-地面站可见窗口添加到现有缓存中，
以便进行完整的数传规划。

Usage:
    python examples/compute_gs_visibility.py \
        --scenario scenarios/large_scale_frequency_with_antennas.json \
        --input-cache java/output/frequency_scenario/visibility_windows.json \
        --output-cache java/output/frequency_scenario/visibility_windows_with_gs.json
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def compute_gs_visibility_simple(scenario_path: str, input_cache_path: str, output_cache_path: str):
    """
    简化版卫星-地面站可见窗口计算

    实际应用中应该使用Orekit进行精确计算。
    这里使用简化的模型生成示例数据。
    """

    # 加载场景
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)

    # 加载现有缓存
    with open(input_cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    satellites = scenario.get('satellites', [])
    ground_stations = scenario.get('ground_stations', [])

    print(f"卫星数量: {len(satellites)}")
    print(f"地面站数量: {len(ground_stations)}")

    # 简化的可见窗口生成（每颗卫星每个地面站每天约2-3个窗口）
    gs_windows = []
    base_time = datetime(2024, 3, 15, 0, 0, 0)

    for sat in satellites:
        sat_id = sat['id']
        # 为每颗卫星生成与地面站的可见窗口
        for i, gs in enumerate(ground_stations):
            gs_id = gs['id']
            # 简化：每颗卫星每天经过每个地面站约2-3次
            for pass_num in range(2):
                # 基于卫星ID和地面站索引生成伪随机时间
                hour_offset = (hash(sat_id) + i * 3 + pass_num * 8) % 24
                minute_offset = (hash(sat_id + gs_id) % 60)

                start_time = base_time + timedelta(hours=hour_offset, minutes=minute_offset)
                duration = 8 + (hash(sat_id) % 7)  # 8-15分钟
                end_time = start_time + timedelta(minutes=duration)

                # 最大仰角（随机生成，实际应该计算）
                max_el = 20 + (hash(gs_id + sat_id) % 70)

                gs_windows.append({
                    "sat": sat_id,
                    "tgt": f"GS:{gs_id}",
                    "start": start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "end": end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "dur": duration * 60,
                    "el": max_el
                })

    # 合并到缓存
    if 'windows' not in cache:
        cache['windows'] = []

    cache['windows'].extend(gs_windows)

    # 保存结果
    with open(output_cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    print(f"\n添加了 {len(gs_windows)} 个卫星-地面站窗口")
    print(f"总窗口数: {len(cache['windows'])}")
    print(f"输出文件: {output_cache_path}")


def main():
    parser = argparse.ArgumentParser(description='计算卫星-地面站可见窗口')
    parser.add_argument('--scenario', '-s', required=True, help='场景文件路径')
    parser.add_argument('--input-cache', '-i', required=True, help='输入缓存文件路径')
    parser.add_argument('--output-cache', '-o', required=True, help='输出缓存文件路径')

    args = parser.parse_args()
    compute_gs_visibility_simple(args.scenario, args.input_cache, args.output_cache)


if __name__ == '__main__':
    main()
