#!/usr/bin/env python3
"""
查看调度任务详细信息的工具脚本

Usage:
    # 查看算法结果
    python scripts/view_schedule_tasks.py --algorithm edd

    # 筛选特定目标
    python scripts/view_schedule_tasks.py --algorithm edd --target TGT-0001

    # 筛选特定卫星
    python scripts/view_schedule_tasks.py --algorithm edd --satellite OPT-01

    # 限制显示数量
    python scripts/view_schedule_tasks.py --algorithm edd --limit 50

    # 按时间范围筛选
    python scripts/view_schedule_tasks.py --algorithm edd --time-range 2024-03-15T00:00:00,2024-03-15T12:00:00

    # 只显示有数传的任务
    python scripts/view_schedule_tasks.py --algorithm edd --has-downlink

    # 显示统计信息
    python scripts/view_schedule_tasks.py --algorithm edd --stats

    # 导出到CSV
    python scripts/view_schedule_tasks.py --algorithm edd --output tasks.csv
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tasks(result_file: str) -> List[Dict[str, Any]]:
    """加载调度结果文件"""
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('imaging', {}).get('scheduled_tasks', [])


def parse_time_range(time_range_str: str) -> tuple:
    """解析时间范围字符串"""
    parts = time_range_str.split(',')
    if len(parts) != 2:
        raise ValueError("时间范围格式错误，应为: start,end")

    start = datetime.fromisoformat(parts[0].replace('Z', '+00:00'))
    end = datetime.fromisoformat(parts[1].replace('Z', '+00:00'))
    return start, end


def filter_tasks(
    tasks: List[Dict[str, Any]],
    target: Optional[str] = None,
    satellite: Optional[str] = None,
    time_range: Optional[tuple] = None,
    has_downlink: bool = False
) -> List[Dict[str, Any]]:
    """
    筛选任务

    Args:
        tasks: 任务列表
        target: 目标ID筛选
        satellite: 卫星ID筛选
        time_range: 时间范围元组 (start, end)
        has_downlink: 是否只显示有数传的任务

    Returns:
        筛选后的任务列表
    """
    filtered = tasks

    if target:
        filtered = [t for t in filtered if t.get('target_id') == target]

    if satellite:
        filtered = [t for t in filtered if t.get('satellite_id') == satellite]

    if time_range:
        start, end = time_range
        filtered = [
            t for t in filtered
            if t.get('imaging_start')
            and start <= datetime.fromisoformat(t['imaging_start'].replace('Z', '+00:00')) <= end
        ]

    if has_downlink:
        filtered = [t for t in filtered if t.get('ground_station_id')]

    return filtered


def print_task(task: Dict[str, Any], index: int = None):
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


def print_statistics(tasks: List[Dict[str, Any]]):
    """打印统计信息"""
    if not tasks:
        print("没有任务数据")
        return

    print("\n" + "=" * 70)
    print("任务统计")
    print("=" * 70)

    # 基本统计
    total_tasks = len(tasks)
    print(f"\n总任务数: {total_tasks}")

    # 按卫星统计
    sat_counts = {}
    sat_data = {}
    for task in tasks:
        sat_id = task.get('satellite_id', 'unknown')
        sat_counts[sat_id] = sat_counts.get(sat_id, 0) + 1

        if sat_id not in sat_data:
            sat_data[sat_id] = {'count': 0, 'data': 0.0}
        sat_data[sat_id]['count'] += 1
        sat_data[sat_id]['data'] += task.get('data_transferred', 0)

    print(f"\n按卫星统计:")
    for sat_id in sorted(sat_counts.keys()):
        print(f"  {sat_id}: {sat_counts[sat_id]} 个任务, {sat_data[sat_id]['data']:.2f} GB")

    # 按目标统计 (只显示前10个)
    target_counts = {}
    for task in tasks:
        tgt_id = task.get('target_id', 'unknown')
        target_counts[tgt_id] = target_counts.get(tgt_id, 0) + 1

    print(f"\n观测次数最多的目标 (前10):")
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for tgt_id, count in sorted_targets:
        print(f"  {tgt_id}: {count} 次")

    # 数传统计
    tasks_with_downlink = [t for t in tasks if t.get('ground_station_id')]
    print(f"\n数传统计:")
    print(f"  有数传的任务: {len(tasks_with_downlink)}/{total_tasks} ({len(tasks_with_downlink)/total_tasks*100:.1f}%)")

    total_data = sum(t.get('data_transferred', 0) for t in tasks)
    print(f"  总数据量: {total_data:.2f} GB")

    # 时间跨度
    if tasks:
        start_times = [datetime.fromisoformat(t['imaging_start'].replace('Z', '+00:00'))
                      for t in tasks if t.get('imaging_start')]
        if start_times:
            min_time = min(start_times)
            max_time = max(start_times)
            duration = (max_time - min_time).total_seconds() / 3600
            print(f"\n时间跨度:")
            print(f"  开始: {min_time.isoformat()}")
            print(f"  结束: {max_time.isoformat()}")
            print(f"  持续时间: {duration:.2f} 小时")

    print("=" * 70)


def export_to_csv(tasks: List[Dict[str, Any]], output_path: str):
    """导出任务到CSV文件"""
    import csv

    if not tasks:
        print("没有任务数据可导出")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=tasks[0].keys())
        writer.writeheader()
        writer.writerows(tasks)

    print(f"已导出 {len(tasks)} 个任务到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='查看调度任务详细信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看算法结果
  python scripts/view_schedule_tasks.py --algorithm edd

  # 筛选特定目标
  python scripts/view_schedule_tasks.py --algorithm edd --target TGT-0001

  # 筛选特定卫星
  python scripts/view_schedule_tasks.py --algorithm edd --satellite OPT-01

  # 按时间范围筛选
  python scripts/view_schedule_tasks.py --algorithm edd --time-range 2024-03-15T00:00:00,2024-03-15T12:00:00

  # 只显示有数传的任务
  python scripts/view_schedule_tasks.py --algorithm edd --has-downlink

  # 显示统计信息
  python scripts/view_schedule_tasks.py --algorithm edd --stats

  # 导出到CSV
  python scripts/view_schedule_tasks.py --algorithm edd --output tasks.csv
        """
    )

    parser.add_argument(
        '--algorithm', '-a',
        required=True,
        choices=['greedy', 'edd', 'spt', 'ga', 'sa', 'aco', 'pso', 'tabu'],
        help='算法名称'
    )
    parser.add_argument(
        '--target', '-t',
        help='筛选特定目标ID'
    )
    parser.add_argument(
        '--satellite', '-s',
        help='筛选特定卫星ID'
    )
    parser.add_argument(
        '--time-range', '-r',
        help='时间范围筛选, 如: 2024-03-15T00:00:00,2024-03-15T12:00:00'
    )
    parser.add_argument(
        '--has-downlink', '-d',
        action='store_true',
        help='只显示有数传的任务'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=20,
        help='显示任务数量限制 (默认: 20, 0表示无限制)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='显示统计信息'
    )
    parser.add_argument(
        '--output', '-o',
        help='导出到CSV文件'
    )
    parser.add_argument(
        '--result-dir',
        default='results/benchmark_final',
        help='结果目录路径 (默认: results/benchmark_final)'
    )

    args = parser.parse_args()

    # 加载结果文件
    result_file = Path(args.result_dir) / f"result_{args.algorithm}.json"

    try:
        tasks = load_tasks(str(result_file))
    except FileNotFoundError:
        print(f"错误: 结果文件不存在: {result_file}")
        print(f"请先运行: python scripts/benchmark.py --algorithms {args.algorithm}")
        return 1

    # 解析时间范围
    time_range = None
    if args.time_range:
        try:
            time_range = parse_time_range(args.time_range)
        except ValueError as e:
            print(f"错误: {e}")
            return 1

    # 筛选任务
    filtered_tasks = filter_tasks(
        tasks,
        target=args.target,
        satellite=args.satellite,
        time_range=time_range,
        has_downlink=args.has_downlink
    )

    # 显示结果
    print("=" * 80)
    print(f"算法: {args.algorithm.upper()}")
    print(f"总任务数: {len(tasks)}")

    if args.target:
        print(f"目标筛选: {args.target}")
    if args.satellite:
        print(f"卫星筛选: {args.satellite}")
    if time_range:
        print(f"时间范围: {time_range[0].isoformat()} ~ {time_range[1].isoformat()}")
    if args.has_downlink:
        print("筛选: 有数传的任务")

    print(f"筛选后: {len(filtered_tasks)} 个任务")
    print("=" * 80)
    print()

    # 显示统计
    if args.stats:
        print_statistics(filtered_tasks)

    # 导出CSV
    if args.output:
        export_to_csv(filtered_tasks, args.output)
        return 0

    # 打印任务
    if args.limit == 0:
        limit = len(filtered_tasks)
    else:
        limit = args.limit

    for i, task in enumerate(filtered_tasks[:limit], 1):
        print_task(task, i)

    if len(filtered_tasks) > limit:
        print(f"... 还有 {len(filtered_tasks) - limit} 个任务")

    return 0


if __name__ == '__main__':
    sys.exit(main())
