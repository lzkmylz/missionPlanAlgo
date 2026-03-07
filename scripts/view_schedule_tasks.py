#!/usr/bin/env python3
"""
查看调度任务详细信息的工具脚本

Usage:
    python scripts/view_schedule_tasks.py --algorithm edd
    python scripts/view_schedule_tasks.py --algorithm edd --target TGT-0001
    python scripts/view_schedule_tasks.py --algorithm edd --satellite OPT-01
    python scripts/view_schedule_tasks.py --algorithm edd --limit 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tasks(result_file: str):
    """加载调度结果文件"""
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('imaging', {}).get('scheduled_tasks', [])


def print_task(task: dict, index: int = None):
    """打印单个任务详情"""
    prefix = f"[{index}] " if index is not None else ""
    print(f"{prefix}任务ID: {task.get('task_id', 'N/A')}")
    print(f"    卫星: {task.get('satellite_id', 'N/A')}")
    print(f"    目标: {task.get('target_id', 'N/A')}")
    print(f"    成像时间: {task.get('imaging_start', 'N/A')} ~ {task.get('imaging_end', 'N/A')}")
    print(f"    成像时长: {task.get('imaging_duration', 'N/A')}秒")
    print(f"    成像模式: {task.get('imaging_mode', 'N/A')}")
    print(f"    侧摆角: {task.get('slew_angle', 'N/A')}度")
    print(f"    机动时间: {task.get('slew_time', 'N/A')}秒")

    # 数传信息
    gs_id = task.get('ground_station_id')
    if gs_id:
        print(f"    地面站: {gs_id}")
        print(f"    数传时间: {task.get('downlink_start', 'N/A')} ~ {task.get('downlink_end', 'N/A')}")
        print(f"    数传数据: {task.get('data_transferred', 0):.2f} GB")
    print()


def main():
    parser = argparse.ArgumentParser(description='查看调度任务详细信息')
    parser.add_argument('--algorithm', '-a', required=True,
                        choices=['greedy', 'edd', 'ga', 'sa', 'aco', 'pso', 'tabu'],
                        help='算法名称')
    parser.add_argument('--target', '-t', help='筛选特定目标ID')
    parser.add_argument('--satellite', '-s', help='筛选特定卫星ID')
    parser.add_argument('--limit', '-l', type=int, default=20, help='显示任务数量限制 (默认: 20)')
    parser.add_argument('--output', '-o', help='导出到CSV文件')

    args = parser.parse_args()

    # 加载结果文件
    result_file = f'results/benchmark_final/result_{args.algorithm}.json'

    try:
        tasks = load_tasks(result_file)
    except FileNotFoundError:
        print(f"错误: 结果文件不存在: {result_file}")
        print(f"请先运行: python scripts/benchmark_all_algorithms_complete.py --algorithms {args.algorithm}")
        return 1

    # 筛选
    filtered_tasks = tasks
    if args.target:
        filtered_tasks = [t for t in filtered_tasks if t.get('target_id') == args.target]
    if args.satellite:
        filtered_tasks = [t for t in filtered_tasks if t.get('satellite_id') == args.satellite]

    # 显示结果
    print("="*80)
    print(f"算法: {args.algorithm.upper()}")
    print(f"总任务数: {len(tasks)}")
    if args.target:
        print(f"目标筛选: {args.target}")
    if args.satellite:
        print(f"卫星筛选: {args.satellite}")
    print(f"显示数量: {min(args.limit, len(filtered_tasks))}/{len(filtered_tasks)}")
    print("="*80)
    print()

    # 打印任务
    for i, task in enumerate(filtered_tasks[:args.limit], 1):
        print_task(task, i)

    # 导出CSV
    if args.output:
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            if filtered_tasks:
                writer = csv.DictWriter(f, fieldnames=filtered_tasks[0].keys())
                writer.writeheader()
                writer.writerows(filtered_tasks)
        print(f"已导出到: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
