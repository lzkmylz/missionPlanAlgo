#!/usr/bin/env python3
"""
统一调度脚本 - 支持单一算法、多算法对比、频次需求和数传规划

用法:
    # 单一算法模式 (默认启用频次和数传)
    python scripts/run_scheduler.py -c cache.json -s scenario.json -a greedy

    # 对比模式
    python scripts/run_scheduler.py -c cache.json -s scenario.json --mode compare -a greedy,ga,edd

    # 禁用频次需求
    python scripts/run_scheduler.py -c cache.json -s scenario.json --no-frequency

    # 禁用数传规划
    python scripts/run_scheduler.py -c cache.json -s scenario.json --no-downlink

    # GA算法指定参数
    python scripts/run_scheduler.py -c cache.json -s scenario.json -a ga --generations 200
"""

import argparse
import json
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from core.resources.ground_station_pool import GroundStationPool
from core.network.relay_satellite import RelayNetwork, RelaySatellite
from scheduler.unified_scheduler import UnifiedScheduler, UnifiedScheduleResult
from evaluation.metrics import MetricsCalculator

# 从 utils 和 config 导入公共功能
from scripts.utils import load_window_cache_from_json, setup_logging
from scripts.config import (
    get_algorithm_config,
    get_algorithm_name,
    expand_algorithm_selection,
    validate_algorithms
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='统一调度脚本 - 支持单一算法、多算法对比、频次需求和数传规划',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 - 使用贪心算法 (默认启用频次和数传，自动保存结果)
  python scripts/run_scheduler.py -c cache.json -s scenario.json

  # 使用遗传算法
  python scripts/run_scheduler.py -c cache.json -s scenario.json -a ga

  # 多算法对比
  python scripts/run_scheduler.py -c cache.json -s scenario.json --mode compare -a greedy,ga,edd

  # 禁用频次需求或数传规划
  python scripts/run_scheduler.py -c cache.json -s scenario.json --no-frequency
  python scripts/run_scheduler.py -c cache.json -s scenario.json --no-downlink

  # 简化模式 (跳过昂贵计算)
  python scripts/run_scheduler.py -c cache.json -s scenario.json --simplified

  # 指定输出路径
  python scripts/run_scheduler.py -c cache.json -s scenario.json -o results/my_schedule.json

  # 禁用自动保存
  python scripts/run_scheduler.py -c cache.json -s scenario.json --no-save
        """
    )

    # 必需参数
    parser.add_argument(
        '-c', '--cache',
        required=True,
        help='预计算窗口缓存文件路径 (JSON格式)'
    )
    parser.add_argument(
        '-s', '--scenario',
        required=True,
        help='场景配置文件路径 (JSON格式)'
    )

    # 算法选择
    parser.add_argument(
        '-a', '--algorithm',
        default='greedy',
        help='调度算法 (默认: greedy)，对比模式下用逗号分隔多个算法'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'compare'],
        default='single',
        help='运行模式: single=单一算法 (默认), compare=多算法对比'
    )

    # GA算法参数 - 平衡模式默认配置
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='GA迭代次数 (默认: 50, 平衡模式。超过50代边际收益极低)'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=80,
        help='GA种群大小 (默认: 80, 平衡模式)'
    )
    parser.add_argument(
        '--mutation-rate',
        type=float,
        default=0.2,
        help='GA变异率 (默认: 0.2, 增强探索能力)'
    )
    parser.add_argument(
        '--crossover-rate',
        type=float,
        default=0.8,
        help='GA交叉率 (默认: 0.8)'
    )

    # 功能开关
    parser.add_argument(
        '--frequency',
        action='store_true',
        default=True,
        help='启用观测频次需求处理 (默认: 启用)'
    )
    parser.add_argument(
        '--downlink',
        action='store_true',
        default=True,
        help='启用地面站数传规划 (默认: 启用)'
    )
    parser.add_argument(
        '--no-frequency',
        dest='frequency',
        action='store_false',
        help='禁用观测频次需求处理'
    )
    parser.add_argument(
        '--no-downlink',
        dest='downlink',
        action='store_false',
        help='禁用地面站数传规划'
    )
    parser.add_argument(
        '--simplified',
        action='store_true',
        help=argparse.SUPPRESS  # 隐藏此参数，高精度要求下已禁用
    )

    # 对比模式参数
    parser.add_argument(
        '--repetitions',
        type=int,
        default=1,
        help='对比模式下每种算法的重复次数 (默认: 1)'
    )

    # 输出
    parser.add_argument(
        '-o', '--output',
        help='输出结果文件路径 (JSON格式，默认: results/{algorithm}_schedule_{timestamp}.json)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='禁用自动保存结果到文件'
    )

    # 其他
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    # 缓存检测
    parser.add_argument(
        '--auto-cache',
        action='store_true',
        default=True,
        help='启用自动缓存检测 (默认: 启用)'
    )
    parser.add_argument(
        '--no-auto-cache',
        dest='auto_cache',
        action='store_false',
        help='禁用自动缓存检测'
    )

    return parser.parse_args(args)


def run_single_algorithm(
    algorithm_name: str,
    mission: Mission,
    cache,
    enable_downlink: bool = True,
    enable_frequency: bool = True,
    seed: int = 42,
    ga_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    运行单个调度算法

    Args:
        algorithm_name: 算法名称
        mission: 任务场景
        cache: 可见性窗口缓存
        enable_downlink: 是否启用数传规划
        enable_frequency: 是否启用频次需求
        seed: 随机种子
        ga_params: GA特定参数字典

    Returns:
        包含运行结果的字典
    """
    logger.info(f"运行算法: {algorithm_name.upper()}")

    # 准备地面站资源池
    ground_station_pool = None
    if enable_downlink and mission.ground_stations:
        ground_station_pool = GroundStationPool(mission.ground_stations)

    # 准备中继卫星网络（场景有中继卫星时自动启用）
    relay_network = None
    if enable_downlink and mission.relay_satellites:
        relay_sats = [
            RelaySatellite(
                id=r['id'],
                name=r.get('name', r['id']),
                orbit_type=r.get('orbit_type', 'GEO'),
                longitude=r['longitude'],
                uplink_capacity=r.get('uplink_capacity', 300.0),
                downlink_capacity=r.get('downlink_capacity', 300.0),
            )
            for r in mission.relay_satellites
        ]
        relay_network = RelayNetwork(relay_sats)

    # 获取算法配置
    overrides = {}
    if algorithm_name == 'ga' and ga_params:
        overrides.update(ga_params)

    scheduler_config = get_algorithm_config(
        algorithm=algorithm_name,
        enable_downlink=enable_downlink,
        enable_frequency=enable_frequency,
        seed=seed,
        **overrides
    )

    # 创建并运行统一调度器
    scheduler = UnifiedScheduler(
        mission=mission,
        window_cache=cache,
        ground_station_pool=ground_station_pool,
        relay_network=relay_network,
        config=scheduler_config
    )

    start_time = time.time()
    result = scheduler.schedule()
    computation_time = time.time() - start_time

    # 计算性能指标
    metrics_calc = MetricsCalculator(mission)
    metrics = metrics_calc.calculate_all(result.imaging_result)

    # 获取聚类统计信息（如果调度器支持聚类）
    clustering_summary = {}
    if hasattr(scheduler, 'get_cluster_schedule_summary'):
        clustering_summary = scheduler.get_cluster_schedule_summary()
    elif hasattr(scheduler, 'get_clustering_metrics'):
        clustering_summary = scheduler.get_clustering_metrics()

    # 序列化任务列表（包含姿态角等详细信息）
    scheduled_tasks = []
    for task in result.imaging_result.scheduled_tasks:
        task_dict = {
            'task_id': task.task_id,
            'satellite_id': task.satellite_id,
            'target_id': task.target_id,
            'imaging_start': task.imaging_start.isoformat() if task.imaging_start else None,
            'imaging_end': task.imaging_end.isoformat() if task.imaging_end else None,
            'imaging_mode': task.imaging_mode,
            'slew_angle': getattr(task, 'slew_angle', None),
            'slew_time': getattr(task, 'slew_time', None),
            'pitch_angle': getattr(task, 'pitch_angle', None),
            'roll_angle': getattr(task, 'roll_angle', None),
            'yaw_angle': getattr(task, 'yaw_angle', None),
            'reset_time': getattr(task, 'reset_time', None),
            'priority': getattr(task, 'priority', None),
            'storage_before': getattr(task, 'storage_before', None),
            'storage_after': getattr(task, 'storage_after', None),
            # 详细能源变化字段
            'power_before_wh': getattr(task, 'power_before', None),
            'power_after_wh': getattr(task, 'power_after', None),
            'power_consumed_wh': getattr(task, 'power_consumed', None),
            'power_generated_wh': getattr(task, 'power_generated', None),
            'energy_consumption_j': getattr(task, 'energy_consumption', None),
            'battery_soc_before_pct': getattr(task, 'battery_soc_before', None),
            'battery_soc_after_pct': getattr(task, 'battery_soc_after', None),
            # 聚类任务相关字段
            'is_cluster_task': getattr(task, 'is_cluster_task', False),
            'cluster_id': getattr(task, 'cluster_id', None),
            'primary_target_id': getattr(task, 'primary_target_id', None),
            'covered_target_ids': getattr(task, 'covered_target_ids', []),
            'covered_target_count': getattr(task, 'covered_target_count', 0),
            # 成像足迹相关字段
            'footprint_corners': getattr(task, 'footprint_corners', []),
            'footprint_center': getattr(task, 'footprint_center', None),
            'swath_width_km': getattr(task, 'swath_width_km', 0.0),
            'fov_config': getattr(task, 'fov_config', {}),
        }
        scheduled_tasks.append(task_dict)

    # 序列化数传任务明细
    downlink_tasks = []
    if result.downlink_result:
        for dt in result.downlink_result.downlink_tasks:
            is_relay = dt.ground_station_id.startswith('RELAY:')
            downlink_tasks.append({
                'task_id': dt.task_id,
                'related_imaging_task_id': dt.related_imaging_task_id,
                'satellite_id': dt.satellite_id,
                'ground_station_id': dt.ground_station_id,
                'downlink_type': 'relay' if is_relay else 'ground_station',
                'start_time': dt.start_time.isoformat() if dt.start_time else None,
                'end_time': dt.end_time.isoformat() if dt.end_time else None,
                'data_size_gb': dt.data_size_gb,
                'effective_data_rate_mbps': dt.effective_data_rate,
                'duration_seconds': dt.get_duration_seconds(),
            })

    return {
        'algorithm': algorithm_name,
        'algorithm_name': get_algorithm_name(algorithm_name),
        'scheduled_count': len(result.imaging_result.scheduled_tasks),
        'unscheduled_count': len(result.imaging_result.unscheduled_tasks),
        'demand_satisfaction_rate': metrics.demand_satisfaction_rate,
        'makespan_hours': metrics.makespan / 3600,
        'satellite_utilization': metrics.satellite_utilization,
        'solution_quality': metrics.solution_quality,
        'computation_time': computation_time,
        'downlink_count': len(result.downlink_result.downlink_tasks) if result.downlink_result else 0,
        'downlink_failed_count': len(result.downlink_result.failed_tasks) if result.downlink_result else 0,
        'frequency_satisfaction': result.target_observations if enable_frequency else None,
        'clustering_summary': clustering_summary,
        'scheduled_tasks': scheduled_tasks,
        'downlink_tasks': downlink_tasks,
    }


def run_comparison(
    mission: Mission,
    cache,
    algorithms: List[str],
    enable_downlink: bool = True,
    enable_frequency: bool = True,
    repetitions: int = 1,
    seed: int = 42,
    ga_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    运行多算法对比

    Args:
        mission: 任务场景
        cache: 可见性窗口缓存
        algorithms: 算法列表
        enable_downlink: 是否启用数传规划
        enable_frequency: 是否启用频次需求
        repetitions: 每种算法的重复次数
        seed: 随机种子
        ga_params: GA特定参数字典

    Returns:
        各算法的结果列表
    """
    all_results = []

    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"算法: {get_algorithm_name(algorithm)}")
        print(f"{'='*70}")

        for rep in range(1, repetitions + 1):
            if repetitions > 1:
                print(f"\n  重复 {rep}/{repetitions}")

            try:
                result = run_single_algorithm(
                    algorithm_name=algorithm,
                    mission=mission,
                    cache=cache,
                    enable_downlink=enable_downlink,
                    enable_frequency=enable_frequency,
                    seed=seed,
                    ga_params=ga_params
                )
                result['repetition'] = rep
                all_results.append(result)

                print(f"  完成: {result['scheduled_count']} 任务, "
                      f"满足率: {result['demand_satisfaction_rate']:.1%}, "
                      f"耗时: {result['computation_time']:.2f}s")

            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def print_single_result(result: Dict[str, Any]) -> None:
    """打印单一算法结果"""
    print("\n" + "="*70)
    print("调度结果")
    print("="*70)

    print(f"\n算法: {result['algorithm_name']}")
    print(f"成功调度: {result['scheduled_count']} 个任务")
    print(f"未调度: {result['unscheduled_count']} 个任务")
    print(f"需求满足率: {result['demand_satisfaction_rate']:.2%}")
    print(f"卫星利用率: {result['satellite_utilization']:.2%}")
    print(f"完成时间跨度: {result['makespan_hours']:.2f} 小时")
    print(f"计算时间: {result['computation_time']:.2f} 秒")

    if result.get('downlink_count') is not None:
        dl_tasks = result.get('downlink_tasks', [])
        gs_count = sum(1 for t in dl_tasks if t['downlink_type'] == 'ground_station')
        relay_count = sum(1 for t in dl_tasks if t['downlink_type'] == 'relay')
        failed = result.get('downlink_failed_count', 0)
        print(f"数传成功: {result['downlink_count']} 个"
              f"（地面站: {gs_count}，中继: {relay_count}），失败: {failed} 个")

    if result.get('frequency_satisfaction'):
        satisfied = sum(1 for info in result['frequency_satisfaction'].values() if info.get('satisfied'))
        total = len(result['frequency_satisfaction'])
        print(f"频次满足: {satisfied}/{total} ({satisfied/total:.1%})")

    # 打印能源统计摘要
    scheduled_tasks = result.get('scheduled_tasks', [])
    if scheduled_tasks:
        total_power_consumed = sum(t.get('power_consumed_wh', 0) or 0 for t in scheduled_tasks)
        total_power_generated = sum(t.get('power_generated_wh', 0) or 0 for t in scheduled_tasks)
        total_energy_consumption = sum(t.get('energy_consumption_j', 0) or 0 for t in scheduled_tasks)
        avg_soc_before = sum(t.get('battery_soc_before_pct', 0) or 0 for t in scheduled_tasks) / len(scheduled_tasks) if scheduled_tasks else 0
        avg_soc_after = sum(t.get('battery_soc_after_pct', 0) or 0 for t in scheduled_tasks) / len(scheduled_tasks) if scheduled_tasks else 0

        print("\n" + "-"*70)
        print("能源统计摘要")
        print("-"*70)
        print(f"总电量消耗: {total_power_consumed:.2f} Wh")
        print(f"总发电量: {total_power_generated:.2f} Wh")
        print(f"总机动能量消耗: {total_energy_consumption:.2f} J")
        print(f"平均电池SOC变化: {avg_soc_before:.1f}% → {avg_soc_after:.1f}%")
        print("-"*70)

    print("="*70)


def print_comparison_results(results: List[Dict[str, Any]]) -> None:
    """打印多算法对比结果"""
    print("\n" + "="*70)
    print("多算法对比结果")
    print("="*70)

    print(f"\n{'算法':<20} {'任务数':<10} {'满足率':<10} {'利用率':<10} {'计算时间':<10}")
    print("-"*70)

    for r in results:
        algo_name = r['algorithm_name']
        if len(algo_name) > 18:
            algo_name = algo_name[:15] + "..."
        print(f"{algo_name:<20} {r['scheduled_count']:<10} "
              f"{r['demand_satisfaction_rate']:<10.1%} "
              f"{r['satellite_utilization']:<10.1%} "
              f"{r['computation_time']:<10.2f}s")

    print("="*70)


def save_results_to_file(
    results: List[Dict[str, Any]],
    output_path: str,
    scenario_path: str,
    cache_path: str,
    mode: str
) -> None:
    """保存结果到JSON文件"""
    output_data = {
        'metadata': {
            'scenario': scenario_path,
            'cache': cache_path,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
        },
        'results': results
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"结果已保存: {output_path}")


def main(args: Optional[List[str]] = None) -> int:
    """主函数"""
    parsed_args = parse_args(args)

    setup_logging()

    print("="*70)
    print("统一调度脚本")
    print("="*70)

    try:
        # 检查是否使用了禁用的简化模式
        if parsed_args.simplified:
            raise ValueError(
                "--simplified 选项已被禁用。高精度要求下必须使用精确计算模式。"
            )

        # 1. 加载场景
        print(f"\n[1/3] 加载场景: {parsed_args.scenario}")
        mission = Mission.load(parsed_args.scenario)
        print(f"  卫星: {len(mission.satellites)} 颗")
        print(f"  目标: {len(mission.targets)} 个")
        print(f"  地面站: {len(mission.ground_stations)} 个")

        # 2. 加载缓存（支持智能缓存检测）
        cache_path = parsed_args.cache

        # 尝试智能检测匹配的缓存
        if parsed_args.auto_cache:
            try:
                from core.cache.fingerprint_calculator import FingerprintCalculator
                from core.cache.index_manager import CacheIndexManager

                calculator = FingerprintCalculator()
                fingerprint = calculator.calculate(parsed_args.scenario)

                manager = CacheIndexManager()
                entry = manager.find(fingerprint)

                if entry:
                    print(f"\n[自动缓存检测] 找到匹配的场景缓存:")
                    print(f"  场景指纹: {fingerprint.full_hash[:16]}...")
                    print(f"  缓存文件: {entry.cache_file}")
                    print(f"  访问次数: {entry.access_count}")

                    # 使用检测到的缓存路径
                    if Path(entry.cache_file).exists():
                        cache_path = entry.cache_file
                        print(f"  ✓ 将使用索引中的缓存")
                    else:
                        print(f"  ! 缓存文件不存在，使用指定的缓存路径")
                else:
                    # 检查可复用的轨道缓存
                    orbit_entry = manager.find_reusable_orbit_cache(fingerprint)
                    if orbit_entry:
                        print(f"\n[自动缓存检测] 发现可复用的轨道缓存:")
                        print(f"  来源场景: {orbit_entry.scenario_name}")
                        print(f"  轨道文件: {orbit_entry.orbit_file}")
                        print(f"  提示: 可用 cache_manager.py compute -s <scenario> 利用此缓存")

                    print(f"\n[2/3] 加载窗口缓存: {cache_path}")
            except Exception as e:
                # 智能检测失败不影响主流程
                if parsed_args.auto_cache:
                    print(f"\n[自动缓存检测] 检测失败，使用指定缓存: {e}")
                print(f"\n[2/3] 加载窗口缓存: {cache_path}")
        else:
            print(f"\n[2/3] 加载窗口缓存: {cache_path} (自动检测已禁用)")

        cache = load_window_cache_from_json(cache_path, mission)
        stats = cache.get_statistics()
        print(f"  总窗口数: {stats['total_windows']:,}")
        print(f"  卫星-目标对: {stats['sat_target_pairs']}")

        # 3. 运行调度
        print(f"\n[3/3] 运行调度")
        print(f"  模式: {parsed_args.mode}")
        print(f"  频次需求: {'启用' if parsed_args.frequency else '禁用'}")
        print(f"  数传规划: {'启用' if parsed_args.downlink else '禁用'}")

        # GA参数
        ga_params = {
            'population_size': parsed_args.population_size,
            'generations': parsed_args.generations,
            'mutation_rate': parsed_args.mutation_rate,
            'crossover_rate': parsed_args.crossover_rate,
        }

        if parsed_args.mode == 'single':
            # 单一算法模式
            result = run_single_algorithm(
                algorithm_name=parsed_args.algorithm,
                mission=mission,
                cache=cache,
                enable_downlink=parsed_args.downlink,
                enable_frequency=parsed_args.frequency,
                seed=parsed_args.seed,
                ga_params=ga_params if parsed_args.algorithm == 'ga' else None
            )
            print_single_result(result)

            # 保存结果 (默认自动保存，除非指定 --no-save)
            if not parsed_args.no_save:
                if parsed_args.output:
                    output_path = parsed_args.output
                else:
                    # 生成默认路径: results/{algorithm}_schedule_{timestamp}.json
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = f"results/{parsed_args.algorithm}_schedule_{timestamp}.json"

                save_results_to_file(
                    results=[result],
                    output_path=output_path,
                    scenario_path=parsed_args.scenario,
                    cache_path=parsed_args.cache,
                    mode='single'
                )

        else:  # compare mode
            # 多算法对比模式
            algorithms = expand_algorithm_selection([a.strip() for a in parsed_args.algorithm.split(',')])
            validate_algorithms(algorithms)

            results = run_comparison(
                mission=mission,
                cache=cache,
                algorithms=algorithms,
                enable_downlink=parsed_args.downlink,
                enable_frequency=parsed_args.frequency,
                repetitions=parsed_args.repetitions,
                seed=parsed_args.seed,
                ga_params=ga_params
            )
            print_comparison_results(results)

            # 保存结果 (默认自动保存，除非指定 --no-save)
            if not parsed_args.no_save:
                if parsed_args.output:
                    output_path = parsed_args.output
                else:
                    # 生成默认路径: results/compare_{algorithms}_{timestamp}.json
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    algo_str = '_'.join(algorithms)[:50]  # 限制长度
                    output_path = f"results/compare_{algo_str}_{timestamp}.json"

                save_results_to_file(
                    results=results,
                    output_path=output_path,
                    scenario_path=parsed_args.scenario,
                    cache_path=parsed_args.cache,
                    mode='compare'
                )

        print("\n完成!")
        return 0

    except FileNotFoundError as e:
        print(f"\n错误: 文件不存在 - {e}")
        return 1
    except ValueError as e:
        print(f"\n错误: {e}")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
