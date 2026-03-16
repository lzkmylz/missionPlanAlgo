#!/usr/bin/env python3
"""
区域目标场景Benchmark脚本

支持测试多种调度算法在区域目标场景上的性能。

测试算法:
1. Greedy (贪心)
2. GA (遗传算法)
3. SA (模拟退火)
4. ACO (蚁群优化)
5. PSO (粒子群优化)
6. Tabu (禁忌搜索)

用法:
    # 测试所有算法
    python scripts/area_benchmark.py --scenario scenarios/area_target.json

    # 快速测试 (仅Greedy)
    python scripts/area_benchmark.py --scenario scenarios/area_target.json --quick

    # 只测试特定算法
    python scripts/area_benchmark.py --scenario scenarios/area_target.json --algorithms greedy ga

    # 指定GA参数
    python scripts/area_benchmark.py --scenario scenarios/area_target.json --ga-generations 50 --ga-population 80
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission, Target
from core.models.mosaic_tile import MosaicTile
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.base_scheduler import ScheduleResult
from evaluation.area_metrics import AreaMetricsCalculator, generate_area_comparison_report

# 导入区域目标工具
from scripts.area_benchmark_utils import (
    load_area_scenario,
    create_tile_targets,
    calculate_visibility_windows
)
from scripts.config import (
    get_algorithm_config,
    get_algorithm_name,
    ALGORITHM_CONFIG_TEMPLATES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_algorithm(
    algorithm_key: str,
    mission: Mission,
    window_cache,
    tiles: List[MosaicTile],
    tile_targets: List[Target],
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    测试单个算法

    Args:
        algorithm_key: 算法键名
        mission: 任务场景
        window_cache: 可见性窗口缓存
        tiles: tiles列表
        tile_targets: tile目标列表
        config: 调度器配置
        output_dir: 输出目录

    Returns:
        测试结果字典
    """
    algorithm_name = get_algorithm_name(algorithm_key)

    print(f"\n{'='*70}")
    print(f"测试算法: {algorithm_name}")
    print(f"{'='*70}")

    result_data = {
        'algorithm': algorithm_name,
        'algorithm_key': algorithm_key,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'status': 'failed',
        'error': None,
        'metrics': {}
    }

    try:
        start_time = time.time()

        # 创建调度器
        scheduler = _create_scheduler(algorithm_key, config)

        # 初始化调度器
        scheduler.initialize(mission)
        scheduler.set_window_cache(window_cache)

        # 运行调度
        schedule_result = scheduler.schedule()
        total_time = time.time() - start_time

        # 计算区域目标指标
        metrics_calc = AreaMetricsCalculator(mission, tiles)
        area_metrics = metrics_calc.calculate_all(schedule_result, tile_targets)

        # 收集结果
        coverage = area_metrics['coverage']
        efficiency = area_metrics['efficiency']

        result_data.update({
            'status': 'success',
            'metrics': {
                'scheduled_tasks': coverage['covered_tiles'],
                'unscheduled_tasks': coverage['pending_tiles'],
                'total_tasks': coverage['total_tiles'],
                'schedule_rate': coverage['coverage_ratio'],
                'makespan_hours': coverage['coverage_makespan_hours'],
                'total_computation_time': total_time,
                'satellite_switches': efficiency['satellite_switches'],
                'avg_tasks_per_satellite': efficiency['avg_tasks_per_satellite'],
            },
            'area_metrics': area_metrics
        })

        # 打印结果
        print(f"\n结果:")
        print(f"  成功调度: {result_data['metrics']['scheduled_tasks']}/{result_data['metrics']['total_tasks']} tiles")
        print(f"  覆盖率: {result_data['metrics']['schedule_rate']:.2%}")
        print(f"  完成时间跨度: {result_data['metrics']['makespan_hours']:.2f} 小时")
        print(f"  卫星切换次数: {result_data['metrics']['satellite_switches']}")
        print(f"  求解用时: {result_data['metrics']['total_computation_time']:.2f} 秒")

        # 保存详细结果
        if output_dir:
            output_file = output_dir / f"result_{algorithm_key.lower()}.json"

            full_result = {
                'algorithm': algorithm_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': result_data['metrics'],
                'area_metrics': area_metrics,
                'scheduled_tasks': [
                    {
                        'task_id': getattr(t, 'task_id', str(i)),
                        'target_id': getattr(t, 'target_id', None),
                        'satellite_id': getattr(t, 'satellite_id', None),
                        'start_time': t.start_time.isoformat() if hasattr(t, 'start_time') and t.start_time else None,
                        'end_time': t.end_time.isoformat() if hasattr(t, 'end_time') and t.end_time else None,
                    }
                    for i, t in enumerate(schedule_result.scheduled_tasks)
                ]
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_result, f, indent=2, ensure_ascii=False)
            print(f"  详细结果已保存: {output_file}")

    except Exception as e:
        result_data['error'] = str(e)
        result_data['traceback'] = traceback.format_exc()
        logger.error(f"算法 {algorithm_name} 运行失败: {e}")
        traceback.print_exc()

    return result_data


def _create_scheduler(algorithm_key: str, config: Dict[str, Any]):
    """创建调度器实例"""
    if algorithm_key == 'greedy':
        return GreedyScheduler(config=config.get('imaging_config', {}))
    elif algorithm_key == 'ga':
        from scheduler.metaheuristic.ga_scheduler import GAScheduler
        return GAScheduler(config=config.get('imaging_config', {}))
    elif algorithm_key == 'sa':
        from scheduler.metaheuristic.sa_scheduler import SAScheduler
        return SAScheduler(config=config.get('imaging_config', {}))
    elif algorithm_key == 'aco':
        from scheduler.metaheuristic.aco_scheduler import ACOScheduler
        return ACOScheduler(config=config.get('imaging_config', {}))
    elif algorithm_key == 'pso':
        from scheduler.metaheuristic.pso_scheduler import PSOScheduler
        return PSOScheduler(config=config.get('imaging_config', {}))
    elif algorithm_key == 'tabu':
        from scheduler.metaheuristic.tabu_scheduler import TabuScheduler
        return TabuScheduler(config=config.get('imaging_config', {}))
    else:
        raise ValueError(f"未知算法: {algorithm_key}")


def run_area_benchmark(
    scenario_path: str,
    output_dir: str,
    algorithms: List[str],
    ga_generations: int = 50,
    ga_population: int = 80,
    seed: int = 42
):
    """
    运行区域目标场景性能测试

    Args:
        scenario_path: 场景文件路径
        output_dir: 输出目录
        algorithms: 算法列表
        ga_generations: GA迭代次数
        ga_population: GA种群大小
        seed: 随机种子
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("区域目标场景 - 规划算法性能测试")
    print("="*70)
    print(f"场景文件: {scenario_path}")
    print(f"输出目录: {output_dir}")
    print(f"测试算法: {', '.join(get_algorithm_name(a) for a in algorithms)}")
    print(f"随机种子: {seed}")
    print("="*70)

    # 1. 加载场景并分解为tiles
    print(f"\n[1/3] 加载区域目标场景...")
    mission, area_target, tiles = load_area_scenario(scenario_path)

    # 2. 创建tile目标并计算可见性
    print(f"\n[2/3] 创建tile目标并计算可见性...")
    tile_targets = create_tile_targets(mission, tiles, area_target)
    window_cache = calculate_visibility_windows(mission, tile_targets)

    # 3. 测试各个算法
    print(f"\n{'='*70}")
    print("开始测试算法...")
    print(f"{'='*70}")

    results = []

    for algo_key in algorithms:
        if algo_key not in ALGORITHM_CONFIG_TEMPLATES:
            print(f"\n跳过未知算法: {algo_key}")
            continue

        # 获取算法配置
        config = get_algorithm_config(
            algorithm=algo_key,
            enable_downlink=False,  # 区域目标暂不支持数传
            enable_frequency=False,
            seed=seed
        )

        # GA算法应用自定义参数
        if algo_key == 'ga':
            config['imaging_config']['generations'] = ga_generations
            config['imaging_config']['population_size'] = ga_population

        result = test_algorithm(
            algorithm_key=algo_key,
            mission=mission,
            window_cache=window_cache,
            tiles=tiles,
            tile_targets=tile_targets,
            config=config,
            output_dir=output_path
        )
        results.append(result)

    # 4. 生成总结报告
    generate_report(results, output_path)

    return results


def generate_report(results: List[Dict[str, Any]], output_dir: Path):
    """生成测试报告"""

    report_file = output_dir / "area_benchmark_report.json"
    summary_file = output_dir / "area_benchmark_summary.txt"

    # 收集所有指标用于对比
    area_metrics_dict = {}
    for result in results:
        if result['status'] == 'success':
            area_metrics_dict[result['algorithm']] = result.get('area_metrics', {})

    # 生成对比报告
    if area_metrics_dict:
        comparison_path = output_dir / "area_comparison_report.json"
        generate_area_comparison_report(area_metrics_dict, str(comparison_path))

    # 保存完整报告
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'scenario_type': 'area_target',
        'total_algorithms': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] != 'success'),
        'results': results
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # 生成文本摘要
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("区域目标场景 - 规划算法性能测试报告\n")
        f.write("="*80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试场景: 区域目标拼幅覆盖\n")
        f.write("="*80 + "\n\n")

        # 算法结果对比表
        f.write("算法性能对比:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'算法':<15} {'覆盖率':>10} {'Tiles':>10} {'时间(h)':>10} {'用时(s)':>12} {'状态':>8}\n")
        f.write("-"*80 + "\n")

        for result in results:
            algo_name = result['algorithm']
            status = "成功" if result['status'] == 'success' else "失败"

            if result['status'] == 'success':
                metrics = result['metrics']
                f.write(f"{algo_name:<15} "
                       f"{metrics['schedule_rate']:>9.1%} "
                       f"{metrics['scheduled_tasks']:>3}/{metrics['total_tasks']:<6} "
                       f"{metrics['makespan_hours']:>9.2f} "
                       f"{metrics['total_computation_time']:>11.2f} "
                       f"{status:>8}\n")
            else:
                f.write(f"{algo_name:<15} "
                       f"{'--':>10} "
                       f"{'--':>10} "
                       f"{'--':>10} "
                       f"{'--':>12} "
                       f"{status:>8}\n")

        f.write("-"*80 + "\n\n")

        # 详细覆盖率指标对比
        f.write("区域覆盖详细指标:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'算法':<15} {'面积(km²)':>12} {'面积覆盖率':>12} {'重叠(km²)':>12} {'重叠率':>10} {'卫星数':>8}\n")
        f.write("-"*80 + "\n")

        for result in results:
            if result['status'] == 'success' and 'area_metrics' in result:
                algo_name = result['algorithm']
                coverage = result['area_metrics'].get('coverage', {})
                total_area = coverage.get('total_area_km2', 0)
                area_ratio = coverage.get('area_coverage_ratio', 0)
                overlap = coverage.get('total_overlap_area_km2', 0)
                overlap_ratio = coverage.get('avg_overlap_ratio', 0)
                num_sats = len(coverage.get('tiles_per_satellite', {}))

                f.write(f"{algo_name:<15} "
                       f"{total_area:>11.0f} "
                       f"{area_ratio:>11.1%} "
                       f"{overlap:>11.0f} "
                       f"{overlap_ratio:>9.1%} "
                       f"{num_sats:>8}\n")

        f.write("-"*80 + "\n\n")

        # 最佳算法
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            best_coverage = max(
                successful_results,
                key=lambda r: r['metrics']['schedule_rate']
            )

            f.write("="*80 + "\n")
            f.write("最佳算法 (覆盖率):\n")
            f.write(f"  算法: {best_coverage['algorithm']}\n")
            f.write(f"  Tile覆盖率: {best_coverage['metrics']['schedule_rate']:.2%}\n")
            f.write(f"  面积覆盖率: {best_coverage['area_metrics']['coverage']['area_coverage_ratio']:.2%}\n")
            f.write(f"  完成时间: {best_coverage['metrics']['makespan_hours']:.2f} 小时\n")
            f.write(f"  求解用时: {best_coverage['metrics']['total_computation_time']:.2f} 秒\n")

    print(f"\n{'='*70}")
    print("测试完成!")
    print(f"报告已保存: {report_file}")
    print(f"摘要已保存: {summary_file}")
    if area_metrics_dict:
        print(f"对比报告: {comparison_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='区域目标场景Benchmark - 测试调度算法性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有算法
  python scripts/area_benchmark.py --scenario scenarios/area_target.json

  # 快速测试 (仅Greedy)
  python scripts/area_benchmark.py --scenario scenarios/area_target.json --quick

  # 只测试特定算法
  python scripts/area_benchmark.py --scenario scenarios/area_target.json --algorithms greedy ga

  # 指定GA参数
  python scripts/area_benchmark.py --scenario scenarios/area_target.json --ga-generations 50
        """
    )

    parser.add_argument(
        '--scenario',
        required=True,
        help='区域目标场景配置文件路径 (JSON格式)'
    )

    parser.add_argument(
        '-o', '--output',
        default='results/area_benchmark',
        help='输出目录 (默认: results/area_benchmark)'
    )

    parser.add_argument(
        '-a', '--algorithms',
        nargs='+',
        help='要测试的算法列表 (默认: 全部)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式 (只测试Greedy算法)'
    )

    parser.add_argument(
        '--ga-generations',
        type=int,
        default=50,
        help='GA迭代次数 (默认: 50)'
    )

    parser.add_argument(
        '--ga-population',
        type=int,
        default=80,
        help='GA种群大小 (默认: 80)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    args = parser.parse_args()

    # 确定要测试的算法
    if args.quick:
        algorithms = ['greedy']
    elif args.algorithms:
        algorithms = args.algorithms
    else:
        algorithms = ['greedy', 'ga', 'sa', 'aco', 'pso', 'tabu']

    # 运行benchmark
    run_area_benchmark(
        scenario_path=args.scenario,
        output_dir=args.output,
        algorithms=algorithms,
        ga_generations=args.ga_generations,
        ga_population=args.ga_population,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
