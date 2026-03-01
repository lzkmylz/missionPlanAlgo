#!/usr/bin/env python3
"""
大规模场景可见性计算脚本（并行版本）

使用多进程并行处理60,720对可见性计算

用法:
    python scripts/compute_large_scale_visibility_parallel.py
    python scripts/compute_large_scale_visibility_parallel.py --workers 8
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_large_scale_scenario import (
    load_satellites,
    load_targets,
    load_ground_stations
)


def compute_single_pair(args) -> List[Dict]:
    """计算单个卫星-目标对的可见性窗口"""
    sat_data, target_data, start_time_str, end_time_str, is_ground_station = args

    from datetime import datetime, timedelta
    from core.orbit.utils import EARTH_RADIUS_M, EARTH_GM

    start_time = datetime.fromisoformat(start_time_str)
    end_time = datetime.fromisoformat(end_time_str)

    # 简化模型：使用圆轨道近似计算可见性
    windows = []

    # 提取参数
    sat_id = sat_data['id']
    target_id = target_data['id']
    sat_orbit = sat_data.get('orbit', {})

    # 卫星轨道参数
    a = sat_orbit.get('semi_major_axis', 6871000.0)  # 半长轴(m)
    e = sat_orbit.get('eccentricity', 0.0)  # 偏心率
    i = math.radians(sat_orbit.get('inclination', 55.0))  # 轨道倾角
    raan = math.radians(sat_orbit.get('raan', 0.0))  # 升交点赤经
    mean_anomaly = math.radians(sat_orbit.get('mean_anomaly', 0.0))  # 平近点角

    # 目标位置
    target_lon = math.radians(target_data['longitude'])
    target_lat = math.radians(target_data['latitude'])
    target_alt = target_data.get('altitude', 0.0)

    # 计算轨道周期
    orbital_period = 2 * math.pi * math.sqrt(a**3 / EARTH_GM)

    # 时间步长（60秒）
    time_step = 60.0
    current_time = start_time

    in_window = False
    window_start = None
    max_elevation_in_window = 0.0

    while current_time < end_time:
        # 计算卫星位置（简化模型）
        elapsed_seconds = (current_time - start_time).total_seconds()

        # 当前平近点角
        n = 2 * math.pi / orbital_period  # 平均角速度
        current_mean_anomaly = (mean_anomaly + n * elapsed_seconds) % (2 * math.pi)

        # 圆轨道近似：平近点角 = 真近点角
        true_anomaly = current_mean_anomaly

        # 卫星在轨道平面中的位置
        r_orbit = a * (1 - e**2) / (1 + e * math.cos(true_anomaly))
        x_orbit = r_orbit * math.cos(true_anomaly)
        y_orbit = r_orbit * math.sin(true_anomaly)

        # 转换到地心惯性系
        # 旋转矩阵：轨道平面 -> 惯性系
        cos_raan, sin_raan = math.cos(raan), math.sin(raan)
        cos_i, sin_i = math.cos(i), math.sin(i)

        x = x_orbit * cos_raan - y_orbit * sin_raan * cos_i
        y = x_orbit * sin_raan + y_orbit * cos_raan * cos_i
        z = y_orbit * sin_i

        # 目标在地心惯性系中的位置（简化，假设地球不旋转）
        r_target = EARTH_RADIUS_M + target_alt
        tx = r_target * math.cos(target_lat) * math.cos(target_lon)
        ty = r_target * math.cos(target_lat) * math.sin(target_lon)
        tz = r_target * math.sin(target_lat)

        # 计算卫星到目标的向量
        dx, dy, dz = x - tx, y - ty, z - tz
        range_to_target = math.sqrt(dx**2 + dy**2 + dz**2)

        # 计算仰角
        # 目标本地坐标系中的卫星位置
        # 东-北-天(ENU)坐标系
        sin_lat, cos_lat = math.sin(target_lat), math.cos(target_lat)
        sin_lon, cos_lon = math.sin(target_lon), math.cos(target_lon)

        # 地心到目标的向量
        east = -dx * sin_lon + dy * cos_lon
        north = -dx * sin_lat * cos_lon - dy * sin_lat * sin_lon + dz * cos_lat
        up = dx * cos_lat * cos_lon + dy * cos_lat * sin_lon + dz * sin_lat

        # 计算仰角
        horizontal_dist = math.sqrt(east**2 + north**2)
        elevation = math.degrees(math.atan2(up, horizontal_dist))

        # 检查是否满足最小仰角
        min_elevation = 5.0 if is_ground_station else 0.0

        if elevation > min_elevation:
            if not in_window:
                # 开始新窗口
                in_window = True
                window_start = current_time
                max_elevation_in_window = elevation
            else:
                # 在窗口内，更新最大仰角
                max_elevation_in_window = max(max_elevation_in_window, elevation)
        else:
            if in_window:
                # 结束当前窗口
                windows.append({
                    'satellite_id': sat_id,
                    'target_id': target_id,
                    'start_time': window_start.isoformat(),
                    'end_time': current_time.isoformat(),
                    'max_elevation': round(max_elevation_in_window, 2)
                })
                in_window = False
                window_start = None
                max_elevation_in_window = 0.0

        current_time += timedelta(seconds=time_step)

    # 处理最后一个窗口
    if in_window:
        windows.append({
            'satellite_id': sat_id,
            'target_id': target_id,
            'start_time': window_start.isoformat(),
            'end_time': end_time.isoformat(),
            'max_elevation': round(max_elevation_in_window, 2)
        })

    return windows


def compute_visibility_parallel(
    scenario_path: str,
    output_path: str,
    workers: int = None,
    chunk_size: int = 100
) -> Dict[str, Any]:
    """并行计算可见性窗口"""

    print("=" * 70)
    print("大规模场景可见性计算（并行版本）")
    print("=" * 70)

    # 1. 加载场景
    print(f"\n[1/4] 加载场景文件: {scenario_path}")
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    satellites = load_satellites(scenario_data)
    targets = load_targets(scenario_data)
    ground_stations = load_ground_stations(scenario_data)

    print(f"  - 卫星: {len(satellites)} 颗")
    print(f"  - 目标: {len(targets)} 个")
    print(f"  - 地面站: {len(ground_stations)} 个")

    # 2. 提取时间范围
    duration = scenario_data['duration']
    start_time = datetime.fromisoformat(duration['start'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(duration['end'].replace('Z', '+00:00'))
    time_span_hours = (end_time - start_time).total_seconds() / 3600

    print(f"\n[2/4] 时间范围")
    print(f"  - 开始: {start_time}")
    print(f"  - 结束: {end_time}")
    print(f"  - 时长: {time_span_hours} 小时")

    # 3. 准备任务列表
    print(f"\n[3/4] 准备计算任务")

    # 序列化卫星数据
    sat_data_list = []
    for sat in satellites:
        sat_data_list.append({
            'id': sat.id,
            'name': sat.name,
            'orbit': sat.orbit.__dict__ if hasattr(sat, 'orbit') else {}
        })

    # 序列化目标数据
    target_data_list = []
    for tgt in targets:
        target_data_list.append({
            'id': tgt.id,
            'name': tgt.name,
            'longitude': tgt.longitude,
            'latitude': tgt.latitude,
            'altitude': 0.0
        })

    # 序列化地面站数据
    gs_data_list = []
    for gs in ground_stations:
        gs_data_list.append({
            'id': gs.id,
            'name': gs.name,
            'longitude': gs.longitude,
            'latitude': gs.latitude,
            'altitude': gs.altitude
        })

    # 构建任务列表
    tasks = []
    start_time_str = start_time.isoformat()
    end_time_str = end_time.isoformat()

    # 卫星-目标对
    for sat_data in sat_data_list:
        for tgt_data in target_data_list:
            tasks.append((sat_data, tgt_data, start_time_str, end_time_str, False))

    # 卫星-地面站对
    for sat_data in sat_data_list:
        for gs_data in gs_data_list:
            tasks.append((sat_data, gs_data, start_time_str, end_time_str, True))

    total_tasks = len(tasks)
    print(f"  - 总计算对数: {total_tasks:,}")
    print(f"    - 卫星-目标: {len(satellites) * len(targets):,}")
    print(f"    - 卫星-地面站: {len(satellites) * len(ground_stations):,}")

    # 4. 并行计算
    print(f"\n[4/4] 执行并行可见性计算")
    print(f"  - 工作进程数: {workers or '自动'}")
    print(f"  - 批处理大小: {chunk_size}")
    print("  (这可能需要几分钟时间，请耐心等待)")
    print()

    import time
    start_compute = time.time()

    target_windows = []
    gs_windows = []

    # 使用多进程并行计算
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        futures = {executor.submit(compute_single_pair, task): task for task in tasks}

        # 使用tqdm显示进度
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(futures), desc="计算进度", unit="对")
        except ImportError:
            pbar = None

        completed = 0
        for future in as_completed(futures):
            try:
                windows = future.result()
                task = futures[future]
                is_gs = task[4]

                if windows:
                    if is_gs:
                        gs_windows.extend(windows)
                    else:
                        target_windows.extend(windows)

                completed += 1
                if pbar:
                    pbar.update(1)
                elif completed % 1000 == 0:
                    print(f"  已完成: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)")

            except Exception as e:
                print(f"  错误: {e}")

        if pbar:
            pbar.close()

    compute_time = time.time() - start_compute

    print("\n" + "=" * 70)
    print("计算完成!")
    print("=" * 70)

    print(f"\n结果统计:")
    print(f"  - 卫星-目标窗口: {len(target_windows):,} 个")
    print(f"  - 卫星-地面站窗口: {len(gs_windows):,} 个")
    print(f"  - 总窗口数: {len(target_windows) + len(gs_windows):,}")
    print(f"\n性能统计:")
    print(f"  - 计算时间: {compute_time:.1f} 秒")
    print(f"  - 计算吞吐率: {total_tasks/compute_time:.1f} 对/秒")

    # 5. 保存结果
    print(f"\n保存结果到: {output_path}")

    output_data = {
        'metadata': {
            'scenario': scenario_data['name'],
            'generated_at': datetime.now().isoformat(),
            'time_range': {
                'start': duration['start'],
                'end': duration['end']
            },
            'entities': {
                'satellites': len(satellites),
                'targets': len(targets),
                'ground_stations': len(ground_stations)
            }
        },
        'stats': {
            'total_windows': len(target_windows) + len(gs_windows),
            'target_windows': len(target_windows),
            'ground_station_windows': len(gs_windows),
            'computation_time_seconds': round(compute_time, 2)
        },
        'windows': {
            'target_windows': target_windows,
            'ground_station_windows': gs_windows
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.2f} MB")

    print("\n" + "=" * 70)
    print("可见性计算完成!")
    print("=" * 70)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='大规模场景可见性计算（并行版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/compute_large_scale_visibility_parallel.py
  python scripts/compute_large_scale_visibility_parallel.py --workers 8
  python scripts/compute_large_scale_visibility_parallel.py --scenario my_scenario.json
        """
    )
    parser.add_argument(
        '--scenario', '-s',
        default='scenarios/large_scale_experiment.json',
        help='场景文件路径 (默认: scenarios/large_scale_experiment.json)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/large_scale_visibility.json',
        help='输出文件路径 (默认: results/large_scale_visibility.json)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='并行工作进程数 (默认: 自动检测CPU核心数)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='批处理大小 (默认: 100)'
    )

    args = parser.parse_args()

    try:
        result = compute_visibility_parallel(
            scenario_path=args.scenario,
            output_path=args.output,
            workers=args.workers,
            chunk_size=args.chunk_size
        )
        return 0
    except FileNotFoundError as e:
        print(f"\n错误: 文件不存在: {e}")
        print("请先运行: python scripts/generate_large_scale_scenario.py")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
