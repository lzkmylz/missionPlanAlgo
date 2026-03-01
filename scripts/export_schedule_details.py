#!/usr/bin/env python3
"""
导出详细任务计划 - 从真实调度结果中提取

从大规模频次约束场景的FCFS调度结果中导出完整任务详情
包括成像任务、姿态角、数传任务等
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.ground_station_scheduler import GroundStationScheduler
from scheduler.frequency_utils import create_observation_tasks
from core.resources.ground_station_pool import GroundStationPool
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType
from scripts.run_frequency_comparison import load_mission_with_frequency, load_visibility_cache


def run_fcfs_and_export(
    scenario_path: str,
    visibility_path: str,
    output_path: str,
    max_tasks: int = 50
) -> Dict[str, Any]:
    """
    运行FCFS算法并导出详细任务计划

    Args:
        scenario_path: 场景文件路径
        visibility_path: 可见性缓存路径
        output_path: 输出文件路径
        max_tasks: 最大导出的任务数（避免输出过大）
    """
    print("=" * 80)
    print("导出详细任务计划")
    print("=" * 80)

    # 加载场景
    print(f"\n[1/4] 加载场景: {scenario_path}")
    mission, targets, total_obs_demand, scenario_data = load_mission_with_frequency(scenario_path)
    print(f"  - 卫星: {len(mission.satellites)} 颗")
    print(f"  - 目标: {len(mission.targets)} 个")
    print(f"  - 总观测需求: {total_obs_demand} 次")

    # 加载可见性缓存
    print(f"\n[2/4] 加载可见性缓存...")
    window_cache = None
    if visibility_path and Path(visibility_path).exists():
        window_cache = load_visibility_cache(visibility_path)
        stats = window_cache.get_statistics()
        print(f"  - 窗口总数: {stats.get('total_windows', 0):,}")
    else:
        print(f"  - 警告: 可见性缓存不存在")
        return {}

    # 运行FCFS调度
    print(f"\n[3/4] 运行FCFS调度...")
    scheduler = SPTScheduler({})
    scheduler.initialize(mission)

    if window_cache and hasattr(scheduler, 'set_window_cache'):
        scheduler.set_window_cache(window_cache)

    result = scheduler.schedule()
    print(f"  - 调度任务数: {len(result.scheduled_tasks)}")

    # 运行地面站数传调度
    print(f"\n[4/4] 安排数传计划...")
    gs_pool = GroundStationPool(mission.ground_stations)
    gs_scheduler = GroundStationScheduler(
        ground_station_pool=gs_pool,
        data_rate_mbps=300.0,
        storage_capacity_gb=128.0,
        overflow_threshold=0.95
    )

    for sat in mission.satellites:
        storage_capacity = getattr(sat.capabilities, 'storage_capacity', 128.0)
        gs_scheduler.initialize_satellite_storage(sat.id, storage_capacity)

    # 准备地面站可见性窗口
    gs_visibility_windows = {}
    if window_cache:
        for task in result.scheduled_tasks:
            sat_id = task.satellite_id
            if sat_id not in gs_visibility_windows:
                gs_visibility_windows[sat_id] = []
                for gs in mission.ground_stations:
                    windows = window_cache.get_windows(sat_id, gs.id)
                    for w in windows:
                        gs_visibility_windows[sat_id].append((w.start_time, w.end_time))

    gs_result = gs_scheduler.schedule_downlinks_for_tasks(
        scheduled_tasks=result.scheduled_tasks,
        visibility_windows=gs_visibility_windows
    )

    updated_tasks = gs_scheduler.update_tasks_with_downlink_info(
        result.scheduled_tasks,
        gs_result.downlink_tasks
    )

    print(f"  - 数传任务数: {len(gs_result.downlink_tasks)}")

    # 计算姿态角
    print(f"\n[5/5] 计算姿态角...")
    attitude_calc = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

    # 构建目标字典
    target_dict = {t.id: t for t in mission.targets}
    sat_dict = {s.id: s for s in mission.satellites}

    # 提取任务详情
    imaging_tasks = []
    downlink_tasks = []

    # 限制导出数量
    tasks_to_export = updated_tasks[:max_tasks]

    for task in tasks_to_export:
        # 获取卫星和目标对象
        satellite = sat_dict.get(task.satellite_id)
        target = target_dict.get(task.target_id.split('-OBS')[0] if '-OBS' in task.target_id else task.target_id)

        # 计算姿态角
        roll_angle, pitch_angle, yaw_angle = 0.0, 0.0, 0.0
        if satellite and target:
            try:
                attitude = attitude_calc.calculate_attitude(
                    satellite, target, task.imaging_start
                )
                roll_angle = attitude.roll
                pitch_angle = attitude.pitch
                yaw_angle = attitude.yaw
            except Exception as e:
                print(f"    警告: 姿态角计算失败 {task.task_id}: {e}")

        # 成像时长
        imaging_duration = (task.imaging_end - task.imaging_start).total_seconds()

        # 构建成像任务记录
        imaging_record = {
            "task_id": task.task_id,
            "satellite_id": task.satellite_id,
            "target_id": task.target_id,
            "imaging_start": task.imaging_start.isoformat(),
            "imaging_end": task.imaging_end.isoformat(),
            "imaging_duration_seconds": imaging_duration,
            "imaging_mode": getattr(task, 'imaging_mode', 'unknown'),
            "slew_angle": getattr(task, 'slew_angle', 0.0),
            "roll_angle": roll_angle,
            "pitch_angle": pitch_angle,
            "yaw_angle": yaw_angle,
            "attitude_coordinate_system": "LVLH",
            "storage_before_gb": getattr(task, 'storage_before', 0.0),
            "storage_after_gb": getattr(task, 'storage_after', 0.0),
            "power_before_percent": getattr(task, 'power_before', 100.0),
            "power_after_percent": getattr(task, 'power_after', 100.0),
            "ground_station_id": getattr(task, 'ground_station_id', None),
            "antenna_id": getattr(task, 'antenna_id', None),
            "downlink_start": task.downlink_start.isoformat() if getattr(task, 'downlink_start', None) else None,
            "downlink_end": task.downlink_end.isoformat() if getattr(task, 'downlink_end', None) else None,
            "data_transferred_gb": getattr(task, 'data_transferred', 0.0),
        }
        imaging_tasks.append(imaging_record)

    # 提取数传任务
    for dl_task in gs_result.downlink_tasks[:max_tasks]:
        downlink_record = {
            "task_id": dl_task.task_id,
            "satellite_id": dl_task.satellite_id,
            "ground_station_id": dl_task.ground_station_id,
            "antenna_id": dl_task.antenna_id,
            "start_time": dl_task.start_time.isoformat(),
            "end_time": dl_task.end_time.isoformat(),
            "duration_seconds": dl_task.get_duration_seconds(),
            "data_size_gb": dl_task.data_size_gb,
            "effective_data_rate_mbps": dl_task.effective_data_rate,
            "related_imaging_task_id": dl_task.related_imaging_task_id,
        }
        downlink_tasks.append(downlink_record)

    # 构建输出
    output_data = {
        "metadata": {
            "scenario": scenario_path,
            "algorithm": "FCFS",
            "total_scheduled": len(result.scheduled_tasks),
            "total_downlinks": len(gs_result.downlink_tasks),
            "exported_tasks": len(imaging_tasks),
            "exported_downlinks": len(downlink_tasks),
            "generated_at": datetime.now().isoformat(),
        },
        "imaging_tasks": imaging_tasks,
        "downlink_tasks": downlink_tasks,
    }

    # 保存到文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✓ 任务计划已导出: {output_path}")
    print(f"  - 成像任务: {len(imaging_tasks)} 条")
    print(f"  - 数传任务: {len(downlink_tasks)} 条")

    return output_data


def print_schedule_summary(data: Dict[str, Any]):
    """打印任务计划摘要"""

    print("\n" + "=" * 140)
    print("真实调度结果 - 任务计划摘要")
    print("=" * 140)

    imaging_tasks = data.get('imaging_tasks', [])
    downlink_tasks = data.get('downlink_tasks', [])

    if not imaging_tasks:
        print("没有成像任务数据")
        return

    # 成像任务表
    print(f"\n{'任务ID':<20} {'卫星':<12} {'目标':<12} {'成像开始':<25} {'成像结束':<25} {'时长(s)':<10} {'滚转角':<10} {'俯仰角':<10} {'偏航角':<10}")
    print("-" * 140)

    for task in imaging_tasks[:20]:  # 只显示前20条
        task_id = task['task_id'][:18]
        sat_id = task['satellite_id'][:10]
        target_id = task['target_id'][:10]
        start = task['imaging_start'][11:23] if task['imaging_start'] else 'N/A'
        end = task['imaging_end'][11:23] if task['imaging_end'] else 'N/A'
        duration = f"{task['imaging_duration_seconds']:.1f}"
        roll = f"{task['roll_angle']:.2f}°"
        pitch = f"{task['pitch_angle']:.2f}°"
        yaw = f"{task['yaw_angle']:.2f}°"

        print(f"{task_id:<20} {sat_id:<12} {target_id:<12} {start:<25} {end:<25} {duration:<10} {roll:<10} {pitch:<10} {yaw:<10}")

    if len(imaging_tasks) > 20:
        print(f"\n... 还有 {len(imaging_tasks) - 20} 条记录 ...")

    # 数传任务表
    print(f"\n\n{'数传任务ID':<20} {'卫星':<12} {'地面站':<12} {'天线':<10} {'开始':<25} {'结束':<25} {'数据量(GB)':<12}")
    print("-" * 140)

    for task in downlink_tasks[:20]:
        task_id = task['task_id'][:18]
        sat_id = task['satellite_id'][:10]
        gs_id = task['ground_station_id'][:10]
        ant_id = task['antenna_id'][:8] if task['antenna_id'] else 'N/A'
        start = task['start_time'][11:23] if task['start_time'] else 'N/A'
        end = task['end_time'][11:23] if task['end_time'] else 'N/A'
        data = f"{task['data_size_gb']:.2f}"

        print(f"{task_id:<20} {sat_id:<12} {gs_id:<12} {ant_id:<10} {start:<25} {end:<25} {data:<12}")

    if len(downlink_tasks) > 20:
        print(f"\n... 还有 {len(downlink_tasks) - 20} 条记录 ...")

    # 统计信息
    print("\n" + "=" * 140)
    print("统计汇总")
    print("=" * 140)

    # 按卫星统计
    sat_stats = {}
    for task in imaging_tasks:
        sat_id = task['satellite_id']
        if sat_id not in sat_stats:
            sat_stats[sat_id] = {'imaging': 0, 'downlink': 0, 'data': 0.0}
        sat_stats[sat_id]['imaging'] += 1
        sat_stats[sat_id]['data'] += task.get('data_transferred_gb', 0.0)

    for task in downlink_tasks:
        sat_id = task['satellite_id']
        if sat_id in sat_stats:
            sat_stats[sat_id]['downlink'] += 1

    print(f"\n按卫星统计:")
    print("-" * 80)
    for sat_id in sorted(sat_stats.keys()):
        stats = sat_stats[sat_id]
        print(f"  {sat_id}: 成像 {stats['imaging']} 次, 数传 {stats['downlink']} 次, 数据 {stats['data']:.2f} GB")

    # 姿态角统计
    roll_values = [t['roll_angle'] for t in imaging_tasks]
    pitch_values = [t['pitch_angle'] for t in imaging_tasks]

    print(f"\n姿态角统计:")
    print("-" * 80)
    print(f"  滚转角: 平均={sum(roll_values)/len(roll_values):.2f}°, 最大={max(abs(v) for v in roll_values):.2f}°")
    print(f"  俯仰角: 平均={sum(pitch_values)/len(pitch_values):.2f}°, 最大={max(abs(v) for v in pitch_values):.2f}°")
    print(f"  偏航角: 固定为 0.00° (零偏航模式)")

    print("\n" + "=" * 140)


def main():
    parser = argparse.ArgumentParser(description='导出详细任务计划')
    parser.add_argument('--scenario', '-s', default='scenarios/large_scale_frequency.json',
                        help='场景文件路径')
    parser.add_argument('--visibility', '-v', default='results/large_scale_visibility.json',
                        help='可见性缓存文件路径')
    parser.add_argument('--output', '-o', default='results/schedule_details_fcfs.json',
                        help='输出文件路径')
    parser.add_argument('--max-tasks', '-m', type=int, default=100,
                        help='最大导出的任务数')

    args = parser.parse_args()

    try:
        data = run_fcfs_and_export(
            scenario_path=args.scenario,
            visibility_path=args.visibility,
            output_path=args.output,
            max_tasks=args.max_tasks
        )

        if data:
            print_schedule_summary(data)
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
