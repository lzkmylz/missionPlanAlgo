#!/usr/bin/env python3
"""
生成示例任务计划 - 展示成像、数传等详细信息
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler.base_scheduler import ScheduledTask
from scheduler.ground_station_scheduler import DownlinkTask


def generate_sample_schedule():
    """生成示例任务计划"""

    # 模拟成像任务数据（基于FCFS算法的部分结果）
    imaging_tasks = [
        {
            "task_id": "TGT-0001-OBS1",
            "satellite_id": "SAT-001",
            "target_id": "TGT-0001",
            "imaging_start": "2026-03-01T08:15:23.000000",
            "imaging_end": "2026-03-01T08:15:38.000000",
            "imaging_mode": "high_resolution",
            "slew_angle": 12.5,
            "storage_before": 0.0,
            "storage_after": 2.4,
            "power_before": 100.0,
            "power_after": 98.5,
            "ground_station_id": "GS-001",
            "antenna_id": "ANT-001",
            "downlink_start": "2026-03-01T08:25:00.000000",
            "downlink_end": "2026-03-01T08:25:47.000000",
            "data_transferred": 2.4,
            "roll_angle": 12.3,
            "pitch_angle": 0.2,
            "yaw_angle": 0.0,  # 零偏航模式
        },
        {
            "task_id": "TGT-0002-OBS1",
            "satellite_id": "SAT-002",
            "target_id": "TGT-0002",
            "imaging_start": "2026-03-01T08:22:15.000000",
            "imaging_end": "2026-03-01T08:22:30.000000",
            "imaging_mode": "medium_resolution",
            "slew_angle": -8.3,
            "storage_before": 0.0,
            "storage_after": 1.2,
            "power_before": 100.0,
            "power_after": 98.8,
            "ground_station_id": "GS-002",
            "antenna_id": "ANT-002",
            "downlink_start": "2026-03-01T08:35:00.000000",
            "downlink_end": "2026-03-01T08:35:24.000000",
            "data_transferred": 1.2,
            "roll_angle": -8.1,
            "pitch_angle": 0.1,
            "yaw_angle": 0.0,  # 零偏航模式
        },
        {
            "task_id": "TGT-0003-OBS1",
            "satellite_id": "SAT-001",
            "target_id": "TGT-0003",
            "imaging_start": "2026-03-01T08:45:30.000000",
            "imaging_end": "2026-03-01T08:45:50.000000",
            "imaging_mode": "high_resolution",
            "slew_angle": 25.7,
            "storage_before": 2.4,
            "storage_after": 6.4,
            "power_before": 98.5,
            "power_after": 96.5,
            "ground_station_id": "GS-001",
            "antenna_id": "ANT-001",
            "downlink_start": "2026-03-01T08:55:00.000000",
            "downlink_end": "2026-03-01T08:55:58.000000",
            "data_transferred": 4.0,
            "roll_angle": 25.5,
            "pitch_angle": 0.3,
            "yaw_angle": 0.0,  # 零偏航模式
        },
        {
            "task_id": "TGT-0001-OBS2",
            "satellite_id": "SAT-003",
            "target_id": "TGT-0001",
            "imaging_start": "2026-03-01T09:10:45.000000",
            "imaging_end": "2026-03-01T09:11:00.000000",
            "imaging_mode": "high_resolution",
            "slew_angle": 5.2,
            "storage_before": 0.0,
            "storage_after": 2.4,
            "power_before": 100.0,
            "power_after": 98.5,
            "ground_station_id": "GS-003",
            "antenna_id": "ANT-003",
            "downlink_start": "2026-03-01T09:25:00.000000",
            "downlink_end": "2026-03-01T09:25:47.000000",
            "data_transferred": 2.4,
            "roll_angle": 5.0,
            "pitch_angle": 0.1,
            "yaw_angle": 0.0,  # 零偏航模式
        },
        {
            "task_id": "TGT-0004-OBS1",
            "satellite_id": "SAT-002",
            "target_id": "TGT-0004",
            "imaging_start": "2026-03-01T09:25:18.000000",
            "imaging_end": "2026-03-01T09:25:33.000000",
            "imaging_mode": "low_resolution",
            "slew_angle": -15.8,
            "storage_before": 1.2,
            "storage_after": 2.6,
            "power_before": 98.8,
            "power_after": 97.6,
            "ground_station_id": None,
            "antenna_id": None,
            "downlink_start": None,
            "downlink_end": None,
            "data_transferred": 0.0,
            "roll_angle": -15.5,
            "pitch_angle": 0.2,
            "yaw_angle": 0.0,  # 零偏航模式
        },
    ]

    return imaging_tasks


def print_schedule_table(tasks):
    """打印任务计划表格"""

    print("\n" + "=" * 140)
    print("卫星成像与数传任务计划表")
    print("=" * 140)

    # 表头
    header = (
        f"{'任务ID':<18} {'卫星':<8} {'目标':<10} {'成像开始':<20} "
        f"{'成像结束':<20} {'时长(s)':<8} {'成像模式':<15} {'侧摆角':<8} "
        f"{'固存(GB)':<10} {'电量(%)':<10}"
    )
    print(header)
    print("-" * 140)

    # 成像任务详情
    for task in tasks:
        imaging_start = datetime.fromisoformat(task['imaging_start'])
        imaging_end = datetime.fromisoformat(task['imaging_end'])
        duration = (imaging_end - imaging_start).total_seconds()
        storage_str = f"{task['storage_before']:.1f}->{task['storage_after']:.1f}"
        power_str = f"{task['power_before']:.1f}->{task['power_after']:.1f}"

        row = (
            f"{task['task_id']:<18} {task['satellite_id']:<8} {task['target_id']:<10} "
            f"{task['imaging_start'][11:19]:<20} {task['imaging_end'][11:19]:<20} "
            f"{duration:<8.0f} {task['imaging_mode']:<15} {task['slew_angle']:<8.1f} "
            f"{storage_str:<10} {power_str:<10}"
        )
        print(row)

    print("\n" + "=" * 140)
    print("姿态角信息（LVLH坐标系）")
    print("=" * 140)

    header = (
        f"{'任务ID':<18} {'卫星':<8} {'滚转角(deg)':<12} {'俯仰角(deg)':<12} "
        f"{'偏航角(deg)':<12} {'坐标系':<10}"
    )
    print(header)
    print("-" * 140)

    for task in tasks:
        row = (
            f"{task['task_id']:<18} {task['satellite_id']:<8} "
            f"{task['roll_angle']:<12.2f} {task['pitch_angle']:<12.2f} "
            f"{task['yaw_angle']:<12.2f} {'LVLH':<10}"
        )
        print(row)

    print("\n" + "=" * 140)
    print("数传任务详情")
    print("=" * 140)

    header = (
        f"{'关联成像任务':<18} {'卫星':<8} {'地面站':<10} {'天线':<10} "
        f"{'数传开始':<20} {'数传结束':<20} {'时长(s)':<10} {'数据量(GB)':<12}"
    )
    print(header)
    print("-" * 140)

    for task in tasks:
        if task['ground_station_id']:
            downlink_start = datetime.fromisoformat(task['downlink_start'])
            downlink_end = datetime.fromisoformat(task['downlink_end'])
            duration = (downlink_end - downlink_start).total_seconds()

            row = (
                f"{task['task_id']:<18} {task['satellite_id']:<8} "
                f"{task['ground_station_id']:<10} {task['antenna_id']:<10} "
                f"{task['downlink_start'][11:19]:<20} {task['downlink_end'][11:19]:<20} "
                f"{duration:<10.0f} {task['data_transferred']:<12.1f}"
            )
            print(row)
        else:
            row = (
                f"{task['task_id']:<18} {task['satellite_id']:<8} "
                f"{'未安排':<10} {'N/A':<10} {'N/A':<20} {'N/A':<20} "
                f"{'N/A':<10} {'N/A':<12}"
            )
            print(row)

    print("\n" + "=" * 140)


def print_satellite_timeline(tasks):
    """按卫星打印时间线"""

    print("\n" + "=" * 100)
    print("卫星任务时间线（按卫星分组）")
    print("=" * 100)

    # 按卫星分组
    sat_tasks = {}
    for task in tasks:
        sat_id = task['satellite_id']
        if sat_id not in sat_tasks:
            sat_tasks[sat_id] = []
        sat_tasks[sat_id].append(task)

    # 按时间排序并打印
    for sat_id in sorted(sat_tasks.keys()):
        print(f"\n【{sat_id} 任务序列】")
        print("-" * 100)

        sat_task_list = sorted(sat_tasks[sat_id], key=lambda x: x['imaging_start'])

        for i, task in enumerate(sat_task_list, 1):
            imaging_start = task['imaging_start'][11:19]
            imaging_end = task['imaging_end'][11:19]

            if task['ground_station_id']:
                downlink_start = task['downlink_start'][11:19]
                downlink_end = task['downlink_end'][11:19]
                downlink_info = f"数传: {downlink_start}-{downlink_end} ({task['ground_station_id']})"
            else:
                downlink_info = "数传: 未安排"

            print(
                f"  [{i}] {imaging_start}-{imaging_end} | "
                f"成像: {task['target_id']} | 模式: {task['imaging_mode']} | "
                f"侧摆: {task['slew_angle']:.1f}° | {downlink_info}"
            )

    print("\n" + "=" * 100)


def print_statistics(tasks):
    """打印统计信息"""

    print("\n" + "=" * 80)
    print("任务统计汇总")
    print("=" * 80)

    total_tasks = len(tasks)
    tasks_with_downlink = sum(1 for t in tasks if t['ground_station_id'])
    total_data = sum(t['data_transferred'] for t in tasks)

    sat_stats = {}
    for task in tasks:
        sat_id = task['satellite_id']
        if sat_id not in sat_stats:
            sat_stats[sat_id] = {'imaging_count': 0, 'downlink_count': 0, 'data_total': 0.0}
        sat_stats[sat_id]['imaging_count'] += 1
        if task['ground_station_id']:
            sat_stats[sat_id]['downlink_count'] += 1
            sat_stats[sat_id]['data_total'] += task['data_transferred']

    print(f"\n总任务数: {total_tasks}")
    print(f"已安排数传: {tasks_with_downlink}")
    print(f"未安排数传: {total_tasks - tasks_with_downlink}")
    print(f"总数据量: {total_data:.1f} GB")

    print(f"\n按卫星统计:")
    print("-" * 80)
    for sat_id in sorted(sat_stats.keys()):
        stats = sat_stats[sat_id]
        print(
            f"  {sat_id}: 成像 {stats['imaging_count']} 次, "
            f"数传 {stats['downlink_count']} 次, 数据 {stats['data_total']:.1f} GB"
        )

    # 成像模式统计
    mode_stats = {}
    for task in tasks:
        mode = task['imaging_mode']
        if mode not in mode_stats:
            mode_stats[mode] = 0
        mode_stats[mode] += 1

    print(f"\n按成像模式统计:")
    print("-" * 80)
    for mode, count in sorted(mode_stats.items()):
        print(f"  {mode}: {count} 次")

    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("卫星任务规划系统 - 示例任务计划输出")
    print("场景: 大规模频次约束场景 (1000目标, 2638次观测需求)")
    print("算法: FCFS (先来先服务)")
    print("=" * 80)

    tasks = generate_sample_schedule()
    print_schedule_table(tasks)
    print_satellite_timeline(tasks)
    print_statistics(tasks)

    # 保存为JSON
    output_path = Path("results/sample_schedule_detail.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'scenario': 'large_scale_frequency.json',
                'algorithm': 'FCFS',
                'total_tasks': len(tasks),
                'generated_at': datetime.now().isoformat()
            },
            'imaging_tasks': tasks
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n详细任务计划已保存: {output_path}")


if __name__ == '__main__':
    main()
