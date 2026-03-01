#!/usr/bin/env python3
"""
大规模场景算法对比实验脚本

运行5种算法（FCFS、Greedy-EDF、Greedy-MaxVal、GA、SA）的对比实验

用法:
    python scripts/run_large_scale_comparison.py
    python scripts/run_large_scale_comparison.py --algorithms FCFS,Greedy-EDF,GA
    python scripts/run_large_scale_comparison.py --repetitions 5
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Type
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from evaluation.metrics import MetricsCalculator, PerformanceMetrics
from scheduler.base_scheduler import BaseScheduler, ScheduleResult


# 导入所有调度器
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler


# 算法注册表
ALGORITHM_REGISTRY: Dict[str, Type[BaseScheduler]] = {
    'FCFS': SPTScheduler,  # SPT按最早开始时间，类似FCFS
    'Greedy-EDF': EDDScheduler,  # EDD = Earliest Deadline First
    'Greedy-MaxVal': GreedyScheduler,  # Greedy按价值最大化
    'GA': GAScheduler,
    'SA': SAScheduler,
}


def load_visibility_cache(visibility_path: str):
    """从JSON文件加载可见性窗口到缓存"""
    from core.orbit.visibility.window_cache import VisibilityWindowCache
    from core.orbit.visibility.base import VisibilityWindow

    cache = VisibilityWindowCache()

    with open(visibility_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    windows_data = data.get('windows', {})

    # 加载目标窗口
    for w in windows_data.get('target_windows', []):
        key = (w['satellite_id'], w['target_id'])
        window = VisibilityWindow(
            satellite_id=w['satellite_id'],
            target_id=w['target_id'],
            start_time=datetime.fromisoformat(w['start_time']),
            end_time=datetime.fromisoformat(w['end_time']),
            max_elevation=w.get('max_elevation', 0.0),
            quality_score=w.get('max_elevation', 0.0) / 90.0
        )

        if key not in cache._windows:
            cache._windows[key] = []
            cache._time_index[key] = []
        cache._windows[key].append(window)
        cache._time_index[key].append(window.start_time)

        # 更新辅助索引
        if w['satellite_id'] not in cache._sat_to_targets:
            cache._sat_to_targets[w['satellite_id']] = set()
        cache._sat_to_targets[w['satellite_id']].add(w['target_id'])

        if w['target_id'] not in cache._target_to_sats:
            cache._target_to_sats[w['target_id']] = set()
        cache._target_to_sats[w['target_id']].add(w['satellite_id'])

    # 加载地面站窗口
    for w in windows_data.get('ground_station_windows', []):
        key = (w['satellite_id'], w['target_id'])
        window = VisibilityWindow(
            satellite_id=w['satellite_id'],
            target_id=w['target_id'],
            start_time=datetime.fromisoformat(w['start_time']),
            end_time=datetime.fromisoformat(w['end_time']),
            max_elevation=w.get('max_elevation', 0.0),
            quality_score=w.get('max_elevation', 0.0) / 90.0
        )

        if key not in cache._windows:
            cache._windows[key] = []
            cache._time_index[key] = []
        cache._windows[key].append(window)
        cache._time_index[key].append(window.start_time)

    return cache


def load_mission_from_scenario(scenario_path: str) -> Mission:
    """从场景文件加载Mission对象"""
    from scripts.load_large_scale_scenario import (
        load_satellites, load_targets, load_ground_stations
    )
    from core.models.mission import Mission
    from datetime import datetime

    # 加载场景
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    satellites = load_satellites(scenario_data)
    targets = load_targets(scenario_data)
    ground_stations = load_ground_stations(scenario_data)

    duration = scenario_data['duration']
    start_time = datetime.fromisoformat(duration['start'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(duration['end'].replace('Z', '+00:00'))

    # 创建Mission对象
    mission = Mission(
        name=scenario_data['name'],
        satellites=satellites,
        targets=targets,
        ground_stations=ground_stations,
        start_time=start_time,
        end_time=end_time
    )

    return mission


def run_single_algorithm(
    algorithm_name: str,
    mission: Mission,
    window_cache,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """运行单个算法"""

    scheduler_class = ALGORITHM_REGISTRY.get(algorithm_name)
    if not scheduler_class:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    print(f"\n  运行 {algorithm_name}...")

    # 实例化调度器
    scheduler = scheduler_class(config)
    scheduler.initialize(mission)

    # 注入窗口缓存
    if window_cache and hasattr(scheduler, 'set_window_cache'):
        scheduler.set_window_cache(window_cache)

    # 运行调度
    start_time = time.time()
    result = scheduler.schedule()
    computation_time = time.time() - start_time

    # 计算性能指标
    metrics_calc = MetricsCalculator(mission)
    metrics = metrics_calc.calculate_all(result)

    # 构建结果
    return {
        'algorithm': algorithm_name,
        'scheduled_count': len(result.scheduled_tasks),
        'unscheduled_count': len(result.unscheduled_tasks),
        'computation_time': computation_time,
        'demand_satisfaction_rate': metrics.demand_satisfaction_rate,
        'makespan_hours': metrics.makespan / 3600,
        'satellite_utilization': metrics.satellite_utilization,
        'solution_quality': metrics.solution_quality,
    }


def run_comparison_experiment(
    scenario_path: str,
    visibility_path: str,
    algorithms: List[str],
    repetitions: int,
    output_path: str
) -> Dict[str, Any]:
    """运行对比实验"""

    print("=" * 70)
    print("大规模场景算法对比实验")
    print("=" * 70)

    # 1. 加载场景
    print(f"\n[1/3] 加载场景...")
    mission = load_mission_from_scenario(scenario_path)
    print(f"  - 卫星: {len(mission.satellites)} 颗")
    print(f"  - 目标: {len(mission.targets)} 个")
    print(f"  - 地面站: {len(mission.ground_stations)} 个")

    # 加载可见性缓存
    window_cache = None
    if visibility_path and Path(visibility_path).exists():
        window_cache = load_visibility_cache(visibility_path)
        print(f"\n  已加载可见性缓存: {visibility_path}")
        stats = window_cache.get_statistics()
        print(f"  - 窗口总数: {stats.get('total_windows', 0):,}")

    # 2. 运行算法对比
    print(f"\n[2/3] 运行算法对比 (每种算法 {repetitions} 次重复)")

    all_results: Dict[str, List[Dict]] = {alg: [] for alg in algorithms}

    for alg_name in algorithms:
        print(f"\n{'='*70}")
        print(f"算法: {alg_name}")
        print(f"{'='*70}")

        for rep in range(1, repetitions + 1):
            print(f"\n  重复 {rep}/{repetitions}")

            try:
                result = run_single_algorithm(alg_name, mission, window_cache, {})
                result['repetition'] = rep
                all_results[alg_name].append(result)

                print(f"    完成: {result['scheduled_count']} 任务, "
                      f"满足率: {result['demand_satisfaction_rate']:.1%}, "
                      f"耗时: {result['computation_time']:.2f}s")

            except Exception as e:
                print(f"    错误: {e}")
                import traceback
                traceback.print_exc()

    # 3. 统计分析
    print(f"\n[3/3] 生成对比报告...")

    comparison = compute_statistics(all_results)

    # 保存结果
    output_data = {
        'metadata': {
            'scenario': scenario_path,
            'visibility': visibility_path,
            'algorithms': algorithms,
            'repetitions': repetitions,
            'generated_at': datetime.now().isoformat(),
        },
        'raw_results': all_results,
        'comparison': comparison,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")

    # 打印对比表
    print_comparison_table(comparison)

    return output_data


def compute_statistics(all_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """计算统计指标"""
    import statistics

    comparison = {}

    for alg_name, results in all_results.items():
        if not results:
            continue

        # 收集各项指标
        metrics_names = [
            'scheduled_count',
            'demand_satisfaction_rate',
            'makespan_hours',
            'satellite_utilization',
            'computation_time',
            'solution_quality'
        ]

        stats = {}
        for metric in metrics_names:
            values = [r[metric] for r in results if metric in r]
            if values:
                stats[f'{metric}_mean'] = statistics.mean(values)
                stats[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                stats[f'{metric}_min'] = min(values)
                stats[f'{metric}_max'] = max(values)

        comparison[alg_name] = stats

    return comparison


def print_comparison_table(comparison: Dict[str, Dict]):
    """打印对比表格"""

    print("\n" + "=" * 70)
    print("算法对比结果")
    print("=" * 70)

    # 表头
    print(f"\n{'算法':<15} {'任务数':<12} {'满足率':<10} {'利用率':<10} {'计算时间':<12}")
    print("-" * 70)

    # 按满足率排序
    sorted_algs = sorted(
        comparison.items(),
        key=lambda x: x[1].get('demand_satisfaction_rate_mean', 0),
        reverse=True
    )

    for alg_name, stats in sorted_algs:
        scheduled = stats.get('scheduled_count_mean', 0)
        dsr = stats.get('demand_satisfaction_rate_mean', 0)
        util = stats.get('satellite_utilization_mean', 0)
        comp_time = stats.get('computation_time_mean', 0)

        print(f"{alg_name:<15} {scheduled:<12.0f} {dsr:<10.1%} {util:<10.1%} {comp_time:<12.2f}s")

    print("\n" + "=" * 70)


def generate_markdown_report(output_data: Dict, report_path: str):
    """生成Markdown格式报告"""

    lines = []
    lines.append("# 大规模场景算法对比实验报告\n")

    meta = output_data['metadata']
    lines.append(f"**场景**: {meta['scenario']}\n")
    lines.append(f"**生成时间**: {meta['generated_at']}\n")
    lines.append(f"**重复次数**: {meta['repetitions']}\n\n")

    # 对比表
    lines.append("## 算法对比\n\n")
    lines.append("| 算法 | 任务完成数 | 需求满足率 | 卫星利用率 | 计算时间 |\n")
    lines.append("|------|-----------|-----------|-----------|---------|\n")

    comparison = output_data['comparison']
    sorted_algs = sorted(
        comparison.items(),
        key=lambda x: x[1].get('demand_satisfaction_rate_mean', 0),
        reverse=True
    )

    for alg_name, stats in sorted_algs:
        scheduled = stats.get('scheduled_count_mean', 0)
        dsr = stats.get('demand_satisfaction_rate_mean', 0)
        util = stats.get('satellite_utilization_mean', 0)
        comp_time = stats.get('computation_time_mean', 0)

        lines.append(f"| {alg_name} | {scheduled:.0f} | {dsr:.1%} | {util:.1%} | {comp_time:.2f}s |\n")

    lines.append("\n## 详细统计\n\n")

    for alg_name, stats in sorted_algs:
        lines.append(f"### {alg_name}\n\n")

        for key, value in stats.items():
            if '_mean' in key:
                metric_name = key.replace('_mean', '')
                std = stats.get(key.replace('_mean', '_std'), 0)
                lines.append(f"- {metric_name}: {value:.4f} (±{std:.4f})\n")

        lines.append("\n")

    # 写入文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='大规模场景算法对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_large_scale_comparison.py
  python scripts/run_large_scale_comparison.py --algorithms FCFS,Greedy-EDF,GA
  python scripts/run_large_scale_comparison.py --repetitions 5
        """
    )
    parser.add_argument(
        '--scenario', '-s',
        default='scenarios/large_scale_experiment.json',
        help='场景文件路径'
    )
    parser.add_argument(
        '--visibility', '-v',
        default='results/large_scale_visibility.json',
        help='可见性缓存文件路径'
    )
    parser.add_argument(
        '--algorithms', '-a',
        default='FCFS,Greedy-EDF,Greedy-MaxVal,GA,SA',
        help='要对比的算法列表，逗号分隔'
    )
    parser.add_argument(
        '--repetitions', '-r',
        type=int,
        default=1,
        help='每种算法的重复次数'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/large_scale_comparison.json',
        help='结果输出路径'
    )
    parser.add_argument(
        '--report',
        default='results/large_scale_comparison_report.md',
        help='Markdown报告输出路径'
    )

    args = parser.parse_args()

    # 解析算法列表
    algorithms = [a.strip() for a in args.algorithms.split(',')]

    # 验证算法
    for alg in algorithms:
        if alg not in ALGORITHM_REGISTRY:
            print(f"错误: 未知算法 '{alg}'")
            print(f"可用算法: {', '.join(ALGORITHM_REGISTRY.keys())}")
            return 1

    try:
        # 运行对比实验
        result = run_comparison_experiment(
            scenario_path=args.scenario,
            visibility_path=args.visibility,
            algorithms=algorithms,
            repetitions=args.repetitions,
            output_path=args.output
        )

        # 生成Markdown报告
        generate_markdown_report(result, args.report)

        print("\n实验完成!")
        return 0

    except FileNotFoundError as e:
        print(f"\n错误: 文件不存在: {e}")
        print("请确保已运行:")
        print("  1. python scripts/generate_large_scale_scenario.py")
        print("  2. python scripts/compute_large_scale_visibility.py")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
