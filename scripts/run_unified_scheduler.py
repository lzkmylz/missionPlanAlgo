#!/usr/bin/env python3
"""
统一调度脚本 - 整合完整约束、频次需求和地面站数传

Usage:
    # 基本用法（仅成像调度）
    python scripts/run_unified_scheduler.py \
        --cache java/output/frequency_scenario/visibility_windows.json \
        --scenario scenarios/large_scale_frequency.json

    # 启用数传规划
    python scripts/run_unified_scheduler.py \
        --cache java/output/frequency_scenario/visibility_windows.json \
        --scenario scenarios/large_scale_frequency.json \
        --enable-downlink

    # 使用GA算法
    python scripts/run_unified_scheduler.py \
        --cache java/output/frequency_scenario/visibility_windows.json \
        --scenario scenarios/large_scale_frequency.json \
        --algorithm ga \
        --generations 100

    # 保存结果
    python scripts/run_unified_scheduler.py \
        --cache java/output/frequency_scenario/visibility_windows.json \
        --scenario scenarios/large_scale_frequency.json \
        --output results/unified_schedule.json
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from core.orbit.visibility.window_cache import VisibilityWindowCache
from core.orbit.visibility.base import VisibilityWindow
from core.resources.ground_station_pool import GroundStationPool
from scheduler.unified_scheduler import UnifiedScheduler, UnifiedScheduleResult
from evaluation.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_window_cache_from_json(cache_path: str, mission: Mission) -> VisibilityWindowCache:
    """从JSON文件加载预计算的窗口缓存

    支持两种格式:
    1. 新格式: {"target_windows": [...], "ground_station_windows": [...]}
    2. 旧格式: {"windows": [{"sat": ..., "tgt": ..., "start": ..., "end": ...}]}
    """
    logger.info(f"加载缓存文件: {cache_path}")

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache = VisibilityWindowCache()

    # 创建卫星和目标ID映射
    sat_map = {sat.id: sat for sat in mission.satellites}
    target_map = {target.id: target for target in mission.targets}

    # 加载卫星-目标窗口
    target_windows_count = 0

    # 尝试新格式
    target_windows = data.get('target_windows', [])

    # 如果没有新格式，尝试旧格式 (windows数组)
    if not target_windows and 'windows' in data:
        for w_data in data['windows']:
            sat_id = w_data.get('sat') or w_data.get('satellite_id')
            target_id = w_data.get('tgt') or w_data.get('target_id')
            start_time = w_data.get('start') or w_data.get('start_time')
            end_time = w_data.get('end') or w_data.get('end_time')
            max_el = w_data.get('el', 0.0) or w_data.get('max_elevation', 0.0)

            if not all([sat_id, target_id, start_time, end_time]):
                continue

            # 跳过地面站窗口（旧格式中可能混有地面站窗口）
            if target_id.startswith('GS:'):
                continue

            try:
                window = VisibilityWindow(
                    satellite_id=sat_id,
                    target_id=target_id,
                    start_time=datetime.fromisoformat(start_time.replace('Z', '+00:00')),
                    end_time=datetime.fromisoformat(end_time.replace('Z', '+00:00')),
                    max_elevation=max_el
                )

                key = (sat_id, target_id)
                if key not in cache._windows:
                    cache._windows[key] = []
                    cache._time_index[key] = []

                cache._windows[key].append(window)
                cache._time_index[key].append(window.start_time)

                # 更新辅助索引
                if sat_id not in cache._sat_to_targets:
                    cache._sat_to_targets[sat_id] = set()
                cache._sat_to_targets[sat_id].add(target_id)

                if target_id not in cache._target_to_sats:
                    cache._target_to_sats[target_id] = set()
                cache._target_to_sats[target_id].add(sat_id)

                target_windows_count += 1
            except Exception as e:
                logger.warning(f"解析窗口失败: {e}")
                continue
    else:
        # 新格式处理
        for w_data in target_windows:
            sat_id = w_data['satellite_id']
            target_id = w_data['target_id']

            window = VisibilityWindow(
                satellite_id=sat_id,
                target_id=target_id,
                start_time=datetime.fromisoformat(w_data['start_time']),
                end_time=datetime.fromisoformat(w_data['end_time']),
                max_elevation=w_data.get('max_elevation', 0.0)
            )

            key = (sat_id, target_id)
            if key not in cache._windows:
                cache._windows[key] = []
                cache._time_index[key] = []

            cache._windows[key].append(window)
            cache._time_index[key].append(window.start_time)

            # 更新辅助索引
            if sat_id not in cache._sat_to_targets:
                cache._sat_to_targets[sat_id] = set()
            cache._sat_to_targets[sat_id].add(target_id)

            if target_id not in cache._target_to_sats:
                cache._target_to_sats[target_id] = set()
            cache._target_to_sats[target_id].add(sat_id)

            target_windows_count += 1

    # 加载卫星-地面站窗口
    gs_windows_count = 0

    # 尝试从独立的 ground_station_windows 键加载
    gs_windows_list = data.get('ground_station_windows', [])

    # 如果没有独立键，尝试从 windows 数组中分离
    if not gs_windows_list and 'windows' in data:
        gs_windows_list = [w for w in data['windows'] if 'GS:' in (w.get('tgt', '') or w.get('target_id', ''))]

    for w_data in gs_windows_list:
        sat_id = w_data.get('sat') or w_data.get('satellite_id')
        target_id = w_data.get('tgt') or w_data.get('target_id')
        start_time = w_data.get('start') or w_data.get('start_time')
        end_time = w_data.get('end') or w_data.get('end_time')
        max_el = w_data.get('el', 0.0) or w_data.get('max_elevation', 0.0)

        if not all([sat_id, target_id, start_time, end_time]):
            continue

        try:
            window = VisibilityWindow(
                satellite_id=sat_id,
                target_id=target_id,
                start_time=datetime.fromisoformat(start_time.replace('Z', '+00:00')),
                end_time=datetime.fromisoformat(end_time.replace('Z', '+00:00')),
                max_elevation=max_el
            )

            key = (sat_id, target_id)
            if key not in cache._windows:
                cache._windows[key] = []
                cache._time_index[key] = []

            cache._windows[key].append(window)
            cache._time_index[key].append(window.start_time)

            gs_windows_count += 1
        except Exception as e:
            logger.warning(f"解析地面站窗口失败: {e}")
            continue

    # 对所有窗口排序
    for key in cache._windows:
        sorted_pairs = sorted(zip(cache._time_index[key], cache._windows[key]))
        cache._time_index[key] = [p[0] for p in sorted_pairs]
        cache._windows[key] = [p[1] for p in sorted_pairs]

    logger.info(f"  加载完成: {target_windows_count} 个卫星-目标窗口, {gs_windows_count} 个卫星-地面站窗口")

    return cache


def print_results(result: UnifiedScheduleResult, mission: Mission):
    """打印调度结果"""
    print("\n" + "=" * 70)
    print("统一调度结果")
    print("=" * 70)

    # 成像调度结果
    imaging = result.imaging_result
    print(f"\n【成像任务调度】")
    print(f"  成功调度: {len(imaging.scheduled_tasks)} 个任务")
    print(f"  未调度: {len(imaging.unscheduled_tasks)} 个任务")
    print(f"  完成时间跨度: {imaging.makespan/3600:.2f} 小时")
    print(f"  求解用时: {imaging.computation_time:.2f} 秒")

    # 计算需求满足率
    metrics_calc = MetricsCalculator(mission)
    imaging_metrics = metrics_calc.calculate_all(imaging)
    print(f"  需求满足率 (DSR): {imaging_metrics.demand_satisfaction_rate:.2%}")
    print(f"  卫星利用率: {imaging_metrics.satellite_utilization:.2%}")

    # 数传规划结果
    if result.downlink_result:
        downlink = result.downlink_result
        print(f"\n【地面站数传规划】")
        print(f"  成功规划: {len(downlink.downlink_tasks)} 个数传任务")
        print(f"  失败: {len(downlink.failed_tasks)} 个")

        # 统计各卫星数据
        sat_data: dict = {}
        for dl_task in downlink.downlink_tasks:
            sat_id = dl_task.satellite_id
            if sat_id not in sat_data:
                sat_data[sat_id] = {'count': 0, 'data_gb': 0.0}
            sat_data[sat_id]['count'] += 1
            sat_data[sat_id]['data_gb'] += dl_task.data_size_gb

        if sat_data:
            print(f"\n  各卫星数传统计:")
            for sat_id, stats in sorted(sat_data.items()):
                print(f"    {sat_id}: {stats['count']}次, {stats['data_gb']:.2f} GB")

    # 频次满足度
    print(f"\n【频次需求满足度】")
    satisfied_count = sum(1 for info in result.target_observations.values() if info['satisfied'])
    total_count = len(result.target_observations)
    overall_rate = satisfied_count / total_count if total_count > 0 else 0.0
    print(f"  总体满足率: {overall_rate:.2%} ({satisfied_count}/{total_count})")

    # 显示未满足的目标
    unsatisfied = [
        (tid, info) for tid, info in result.target_observations.items()
        if not info['satisfied']
    ]
    if unsatisfied:
        print(f"\n  未满足的目标:")
        for tid, info in unsatisfied[:10]:  # 只显示前10个
            print(f"    {tid}: {info['status']}")
        if len(unsatisfied) > 10:
            print(f"    ... 还有 {len(unsatisfied) - 10} 个")

    print("\n" + "=" * 70)
    print(f"总耗时: {result.total_computation_time:.2f} 秒")
    print("=" * 70)


def save_results(
    result: UnifiedScheduleResult,
    mission: Mission,
    output_path: str,
    args: argparse.Namespace
):
    """保存结果到JSON文件"""
    output_data = {
        'scenario': mission.name,
        'cache_file': args.cache,
        'algorithm': args.algorithm,
        'config': {
            'enable_downlink': args.enable_downlink,
            'consider_frequency': True,
        },
        'results': result.to_dict(),
        'scheduled_tasks': [
            {
                'task_id': t.task_id,
                'satellite_id': t.satellite_id,
                'target_id': t.target_id,
                'start_time': t.imaging_start.isoformat(),
                'end_time': t.imaging_end.isoformat(),
                'imaging_mode': t.imaging_mode,
                'slew_angle': t.slew_angle,
                'slew_time': t.slew_time,
                'ground_station_id': t.ground_station_id,
                'downlink_start': t.downlink_start.isoformat() if t.downlink_start else None,
                'downlink_end': t.downlink_end.isoformat() if t.downlink_end else None,
                'data_transferred': t.data_transferred,
            }
            for t in result.imaging_result.scheduled_tasks[:500]  # 只保存前500个
        ],
    }

    if result.downlink_result:
        output_data['downlink_tasks'] = [
            {
                'task_id': t.task_id,
                'satellite_id': t.satellite_id,
                'ground_station_id': t.ground_station_id,
                'antenna_id': t.antenna_id,
                'start_time': t.start_time.isoformat(),
                'end_time': t.end_time.isoformat(),
                'data_size_gb': t.data_size_gb,
                'related_imaging_task': t.related_imaging_task_id,
            }
            for t in result.downlink_result.downlink_tasks[:200]
        ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='统一调度脚本 - 整合完整约束、频次需求和地面站数传',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（成像调度 + 数传规划，默认启用）
  # 数传速率和固存容量从每颗卫星的配置中自动读取
  python scripts/run_unified_scheduler.py \\
      --cache java/output/frequency_scenario/visibility_windows.json \\
      --scenario scenarios/large_scale_frequency.json

  # 禁用数传规划（仅成像调度）
  python scripts/run_unified_scheduler.py \\
      --cache java/output/frequency_scenario/visibility_windows.json \\
      --scenario scenarios/large_scale_frequency.json \\
      --no-downlink

  # 使用遗传算法并指定参数
  python scripts/run_unified_scheduler.py \\
      --cache java/output/frequency_scenario/visibility_windows.json \\
      --scenario scenarios/large_scale_frequency.json \\
      --algorithm ga \\
      --generations 100 \\
      --population-size 50

  # 保存结果到文件
  python scripts/run_unified_scheduler.py \\
      --cache java/output/frequency_scenario/visibility_windows.json \\
      --scenario scenarios/large_scale_frequency.json \\
      --output results/unified_schedule.json

  # 忽略频次需求
  python scripts/run_unified_scheduler.py \\
      --cache java/output/frequency_scenario/visibility_windows.json \\
      --scenario scenarios/large_scale_frequency.json \\
      --no-frequency

注意:
  - 数传速率(data_rate)和固存容量(storage_capacity)从每颗卫星的
    capabilities配置中自动读取
  - 在场景JSON文件的satellites[*].capabilities中设置这些参数
        """
    )

    # 必需参数
    parser.add_argument(
        '--cache', '-c',
        required=True,
        help='预计算窗口缓存文件路径 (JSON格式)'
    )
    parser.add_argument(
        '--scenario', '-s',
        required=True,
        help='场景配置文件路径 (JSON格式)'
    )

    # 算法选择
    parser.add_argument(
        '--algorithm', '-a',
        choices=['greedy', 'ga', 'edd'],
        default='greedy',
        help='成像调度算法 (默认: greedy)'
    )

    # GA算法参数
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='GA迭代次数 (默认: 100)'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=50,
        help='GA种群大小 (默认: 50)'
    )
    parser.add_argument(
        '--mutation-rate',
        type=float,
        default=0.1,
        help='GA变异率 (默认: 0.1)'
    )
    parser.add_argument(
        '--crossover-rate',
        type=float,
        default=0.8,
        help='GA交叉率 (默认: 0.8)'
    )

    # 功能开关
    parser.add_argument(
        '--enable-downlink',
        action='store_true',
        default=True,
        help='启用地面站数传规划（默认启用）'
    )
    parser.add_argument(
        '--no-downlink',
        action='store_true',
        help='禁用地面站数传规划'
    )
    parser.add_argument(
        '--no-frequency',
        action='store_true',
        help='忽略频次需求（默认考虑频次）'
    )

    # 数传配置（已移除，现在从每颗卫星的配置中自动读取）
    # 卫星配置中的 storage_capacity (GB) 和 data_rate (Mbps) 会被自动使用

    # 输出
    parser.add_argument(
        '--output', '-o',
        help='输出结果文件路径 (JSON格式)'
    )

    # 性能选项
    parser.add_argument(
        '--simplified',
        action='store_true',
        help='使用简化模式（跳过昂贵的轨道预计算，速度更快但精度稍低）'
    )

    # 其他
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    args = parser.parse_args()

    # 设置随机种子
    import random
    random.seed(args.seed)

    print("=" * 70)
    print("统一调度脚本 - 整合完整约束、频次需求和地面站数传")
    print("=" * 70)

    try:
        # 1. 加载场景
        print(f"\n[1/4] 加载场景: {args.scenario}")
        mission = Mission.load(args.scenario)
        print(f"  卫星: {len(mission.satellites)} 颗")
        print(f"  目标: {len(mission.targets)} 个")
        print(f"  地面站: {len(mission.ground_stations)} 个")

        # 显示目标观测需求
        print("\n  目标观测需求:")
        for target in mission.targets[:5]:  # 只显示前5个
            required = getattr(target, 'required_observations', 1)
            freq_str = "不限" if required == -1 else f"{required}次"
            print(f"    {target.id}: {target.name} - 需要{freq_str}")
        if len(mission.targets) > 5:
            print(f"    ... 还有 {len(mission.targets) - 5} 个目标")

        # 2. 加载缓存
        print(f"\n[2/4] 加载预计算窗口缓存")
        cache = load_window_cache_from_json(args.cache, mission)
        stats = cache.get_statistics()
        print(f"  总窗口数: {stats['total_windows']}")
        print(f"  卫星-目标对: {stats['sat_target_pairs']}")

        # 确定是否启用数传（--no-downlink 优先级高于默认启用）
        enable_downlink = args.enable_downlink and not args.no_downlink

        # 3. 准备地面站资源池
        ground_station_pool = None
        if enable_downlink and mission.ground_stations:
            print(f"\n[3/4] 初始化地面站资源池")
            ground_station_pool = GroundStationPool(mission.ground_stations)
            total_antennas = sum(len(gs.antennas) for gs in mission.ground_stations)
            print(f"  地面站: {len(mission.ground_stations)} 个")
            print(f"  天线总数: {total_antennas} 个")
        else:
            print(f"\n[3/4] 跳过地面站初始化")

        # 4. 配置并运行统一调度器
        print(f"\n[4/4] 运行统一调度")
        print(f"  成像算法: {args.algorithm.upper()}")
        print(f"  考虑频次需求: {not args.no_frequency}")
        print(f"  启用数传规划: {enable_downlink}")

        # 构建配置
        imaging_config = {
            'use_simplified_slew': args.simplified,  # 简化机动计算
            'consider_power': True,
            'consider_storage': True,
        }

        if args.algorithm == 'ga':
            imaging_config.update({
                'population_size': args.population_size,
                'generations': args.generations,
                'mutation_rate': args.mutation_rate,
                'crossover_rate': args.crossover_rate,
                'random_seed': args.seed,
            })
            print(f"  GA参数: 种群={args.population_size}, 迭代={args.generations}")

        if args.simplified:
            print(f"  使用简化模式（跳过轨道预计算）")

        # 数传配置现在从每颗卫星的配置中自动读取
        # 卫星的 capabilities.storage_capacity 和 capabilities.data_rate 会被使用
        downlink_config = {
            'overflow_threshold': 0.95,
            'link_setup_time_seconds': 60.0,
        }

        config = {
            'imaging_algorithm': args.algorithm,
            'imaging_config': imaging_config,
            'enable_downlink': enable_downlink,
            'downlink_config': downlink_config,
            'consider_frequency': not args.no_frequency,
        }

        # 创建并运行调度器
        scheduler = UnifiedScheduler(
            mission=mission,
            window_cache=cache,
            ground_station_pool=ground_station_pool,
            config=config
        )

        result = scheduler.schedule()

        # 5. 打印结果
        print_results(result, mission)

        # 6. 保存结果
        if args.output:
            save_results(result, mission, args.output, args)

    except Exception as e:
        logger.error(f"运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
