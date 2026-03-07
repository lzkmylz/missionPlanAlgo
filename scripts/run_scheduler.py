#!/usr/bin/env python3
"""
统一调度脚本 - 合并了以下脚本的功能:
- run_scheduler_with_cache.py
- run_scheduler_with_frequency.py
- run_all_schedulers.py
- run_frequency_comparison.py
- run_large_scale_comparison.py

用法:
    # 单一算法模式 (默认)
    python scripts/run_scheduler.py -c cache.json -s scenario.json -a greedy

    # 对比模式
    python scripts/run_scheduler.py -c cache.json -s scenario.json --mode compare -a greedy,ga,edd

    # 启用频次需求
    python scripts/run_scheduler.py -c cache.json -s scenario.json --frequency

    # 启用数传规划
    python scripts/run_scheduler.py -c cache.json -s scenario.json --downlink

    # GA算法指定参数
    python scripts/run_scheduler.py -c cache.json -s scenario.json -a ga --generations 200
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from core.orbit.visibility.window_cache import VisibilityWindowCache
from scheduler.unified_scheduler import UnifiedScheduler, UnifiedScheduleResult
from scheduler.ground_station.scheduler import GroundStationScheduler
from core.resources.ground_station_pool import GroundStationPool
from evaluation.metrics import MetricsCalculator

# 从 utils 导入公共功能
from scripts.utils import (
    load_window_cache_from_json,
    SCHEDULER_REGISTRY,
    get_scheduler_class,
    setup_logging,
    save_results,
    parse_algorithm_list,
    validate_algorithms
)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='统一调度脚本 - 支持单一算法、多算法对比、频次需求和数传规划',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 - 使用贪心算法
  python scripts/run_scheduler.py -c cache.json -s scenario.json

  # 使用遗传算法
  python scripts/run_scheduler.py -c cache.json -s scenario.json -a ga

  # 多算法对比
  python scripts/run_scheduler.py -c cache.json -s scenario.json --mode compare -a greedy,ga,edd

  # 启用频次需求和数传规划
  python scripts/run_scheduler.py -c cache.json -s scenario.json --frequency --downlink

  # 简化模式 (跳过昂贵计算)
  python scripts/run_scheduler.py -c cache.json -s scenario.json --simplified
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
        '--frequency',
        action='store_true',
        help='启用观测频次需求处理'
    )
    parser.add_argument(
        '--downlink',
        action='store_true',
        help='启用地面站数传规划'
    )
    parser.add_argument(
        '--simplified',
        action='store_true',
        help='使用简化模式 (跳过昂贵的轨道预计算)'
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
        help='输出结果文件路径 (JSON格式)'
    )

    # 其他
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    return parser.parse_args(args)


def build_scheduler_config(args: argparse.Namespace) -> Dict[str, Any]:
    """构建调度器配置"""
    config = {
        'consider_power': True,
        'consider_storage': True,
    }

    if args.simplified:
        config['use_simplified_slew'] = True

    # GA特定参数
    if args.algorithm == 'ga':
        config.update({
            'population_size': args.population_size,
            'generations': args.generations,
            'mutation_rate': args.mutation_rate,
            'crossover_rate': args.crossover_rate,
            'random_seed': args.seed,
        })

    return config


def run_single_algorithm(
    algorithm_name: str,
    mission: Mission,
    cache: VisibilityWindowCache,
    config: Dict[str, Any],
    enable_downlink: bool = False,
    enable_frequency: bool = False
) -> Dict[str, Any]:
    """
    运行单个调度算法

    Args:
        algorithm_name: 算法名称
        mission: 任务场景
        cache: 可见性窗口缓存
        config: 调度器配置
        enable_downlink: 是否启用数传规划
        enable_frequency: 是否启用频次需求

    Returns:
        包含运行结果的字典
    """
    logger = setup_logging()
    logger.info(f"运行算法: {algorithm_name.upper()}")

    # 准备地面站资源池
    ground_station_pool = None
    if enable_downlink and mission.ground_stations:
        ground_station_pool = GroundStationPool(mission.ground_stations)

    # 构建统一调度器配置
    scheduler_config = {
        'imaging_algorithm': algorithm_name,
        'imaging_config': config,
        'enable_downlink': enable_downlink,
        'downlink_config': {
            'overflow_threshold': 0.95,
            'link_setup_time_seconds': 60.0,
        },
        'consider_frequency': enable_frequency,
    }

    # 创建并运行统一调度器
    scheduler = UnifiedScheduler(
        mission=mission,
        window_cache=cache,
        ground_station_pool=ground_station_pool,
        config=scheduler_config
    )

    start_time = time.time()
    result = scheduler.schedule()
    computation_time = time.time() - start_time

    # 计算性能指标
    metrics_calc = MetricsCalculator(mission)
    metrics = metrics_calc.calculate_all(result.imaging_result)

    return {
        'algorithm': algorithm_name,
        'scheduled_count': len(result.imaging_result.scheduled_tasks),
        'unscheduled_count': len(result.imaging_result.unscheduled_tasks),
        'demand_satisfaction_rate': metrics.demand_satisfaction_rate,
        'makespan_hours': metrics.makespan / 3600,
        'satellite_utilization': metrics.satellite_utilization,
        'solution_quality': metrics.solution_quality,
        'computation_time': computation_time,
        'downlink_count': len(result.downlink_result.downlink_tasks) if result.downlink_result else 0,
        'frequency_satisfaction': result.target_observations if enable_frequency else None,
    }


def run_comparison(
    mission: Mission,
    cache: VisibilityWindowCache,
    algorithms: List[str],
    config: Dict[str, Any],
    enable_downlink: bool = False,
    enable_frequency: bool = False,
    repetitions: int = 1
) -> List[Dict[str, Any]]:
    """
    运行多算法对比

    Args:
        mission: 任务场景
        cache: 可见性窗口缓存
        algorithms: 算法列表
        config: 调度器配置
        enable_downlink: 是否启用数传规划
        enable_frequency: 是否启用频次需求
        repetitions: 每种算法的重复次数

    Returns:
        各算法的结果列表
    """
    all_results = []

    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"算法: {algorithm.upper()}")
        print(f"{'='*70}")

        for rep in range(1, repetitions + 1):
            if repetitions > 1:
                print(f"\n  重复 {rep}/{repetitions}")

            try:
                result = run_single_algorithm(
                    algorithm_name=algorithm,
                    mission=mission,
                    cache=cache,
                    config=config,
                    enable_downlink=enable_downlink,
                    enable_frequency=enable_frequency
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


def print_results(result: Any, mode: str = 'single') -> None:
    """打印结果"""
    print("\n" + "="*70)
    print("调度结果")
    print("="*70)

    if mode == 'single':
        print(f"\n算法: {result['algorithm'].upper()}")
        print(f"成功调度: {result['scheduled_count']} 个任务")
        print(f"未调度: {result['unscheduled_count']} 个任务")
        print(f"需求满足率: {result['demand_satisfaction_rate']:.2%}")
        print(f"卫星利用率: {result['satellite_utilization']:.2%}")
        print(f"完成时间跨度: {result['makespan_hours']:.2f} 小时")
        print(f"计算时间: {result['computation_time']:.2f} 秒")

        if result.get('downlink_count'):
            print(f"数传任务: {result['downlink_count']} 个")

    else:  # compare mode
        print(f"\n{'算法':<15} {'任务数':<10} {'满足率':<10} {'利用率':<10} {'计算时间':<10}")
        print("-"*70)

        for r in result:
            print(f"{r['algorithm']:<15} {r['scheduled_count']:<10} "
                  f"{r['demand_satisfaction_rate']:<10.1%} "
                  f"{r['satellite_utilization']:<10.1%} "
                  f"{r['computation_time']:<10.2f}s")

    print("="*70)


def main(args: Optional[List[str]] = None) -> int:
    """主函数"""
    # 解析参数
    parsed_args = parse_args(args)

    # 设置日志
    setup_logging()

    print("="*70)
    print("统一调度脚本")
    print("="*70)

    try:
        # 1. 加载场景
        print(f"\n[1/3] 加载场景: {parsed_args.scenario}")
        mission = Mission.load(parsed_args.scenario)
        print(f"  卫星: {len(mission.satellites)} 颗")
        print(f"  目标: {len(mission.targets)} 个")
        print(f"  地面站: {len(mission.ground_stations)} 个")

        # 2. 加载缓存
        print(f"\n[2/3] 加载窗口缓存: {parsed_args.cache}")
        cache = load_window_cache_from_json(parsed_args.cache, mission)
        stats = cache.get_statistics()
        print(f"  总窗口数: {stats['total_windows']:,}")
        print(f"  卫星-目标对: {stats['sat_target_pairs']}")

        # 3. 运行调度
        print(f"\n[3/3] 运行调度")
        print(f"  模式: {parsed_args.mode}")
        print(f"  频次需求: {'启用' if parsed_args.frequency else '禁用'}")
        print(f"  数传规划: {'启用' if parsed_args.downlink else '禁用'}")

        if parsed_args.mode == 'single':
            # 单一算法模式
            config = build_scheduler_config(parsed_args)
            result = run_single_algorithm(
                algorithm_name=parsed_args.algorithm,
                mission=mission,
                cache=cache,
                config=config,
                enable_downlink=parsed_args.downlink,
                enable_frequency=parsed_args.frequency
            )
            print_results(result, mode='single')

            # 保存结果
            if parsed_args.output:
                save_results(result, parsed_args.output)
                print(f"\n结果已保存: {parsed_args.output}")

        else:  # compare mode
            # 多算法对比模式
            algorithms = parse_algorithm_list(parsed_args.algorithm)
            validate_algorithms(algorithms)

            config = build_scheduler_config(parsed_args)
            results = run_comparison(
                mission=mission,
                cache=cache,
                algorithms=algorithms,
                config=config,
                enable_downlink=parsed_args.downlink,
                enable_frequency=parsed_args.frequency,
                repetitions=parsed_args.repetitions
            )
            print_results(results, mode='compare')

            # 保存结果
            if parsed_args.output:
                output_data = {
                    'metadata': {
                        'scenario': parsed_args.scenario,
                        'cache': parsed_args.cache,
                        'algorithms': algorithms,
                        'repetitions': parsed_args.repetitions,
                        'timestamp': datetime.now().isoformat()
                    },
                    'results': results
                }
                save_results(output_data, parsed_args.output)
                print(f"\n结果已保存: {parsed_args.output}")

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
