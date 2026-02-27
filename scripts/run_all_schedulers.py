#!/usr/bin/env python3
"""
运行所有调度算法进行对比

Usage:
    python scripts/run_all_schedulers.py --cache data/visibility_cache/point_group_scenario_windows.json --scenario scenarios/point_group_scenario.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from core.orbit.visibility.window_cache import VisibilityWindowCache
from core.orbit.visibility.base import VisibilityWindow
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.metaheuristic.aco_scheduler import ACOScheduler
from scheduler.metaheuristic.pso_scheduler import PSOScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler
from scheduler.metaheuristic.tabu_scheduler import TabuScheduler
from evaluation.metrics import MetricsCalculator

# 调度器注册表
SCHEDULER_REGISTRY = {
    'Greedy': (GreedyScheduler, {'name': '贪心算法', 'category': '贪心'}),
    'EDD': (EDDScheduler, {'name': '最早截止日期', 'category': '贪心'}),
    'SPT': (SPTScheduler, {'name': '最短处理时间', 'category': '贪心'}),
    'GA': (GAScheduler, {'name': '遗传算法', 'category': '元启发式', 'params': {'population_size': 50, 'generations': 100}}),
    'ACO': (ACOScheduler, {'name': '蚁群优化', 'category': '元启发式', 'params': {'n_ants': 30, 'max_iter': 50}}),
    'PSO': (PSOScheduler, {'name': '粒子群优化', 'category': '元启发式', 'params': {'n_particles': 30, 'max_iter': 50}}),
    'SA': (SAScheduler, {'name': '模拟退火', 'category': '元启发式', 'params': {'max_iter': 100, 'initial_temp': 100}}),
    'Tabu': (TabuScheduler, {'name': '禁忌搜索', 'category': '元启发式', 'params': {'max_iter': 100, 'tabu_tenure': 10}}),
}


def load_window_cache_from_json(cache_path: str, mission: Mission) -> VisibilityWindowCache:
    """从JSON文件加载预计算的窗口缓存"""
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache = VisibilityWindowCache()

    # 加载卫星-目标窗口
    for w_data in data.get('target_windows', []):
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

        if sat_id not in cache._sat_to_targets:
            cache._sat_to_targets[sat_id] = set()
        cache._sat_to_targets[sat_id].add(target_id)

        if target_id not in cache._target_to_sats:
            cache._target_to_sats[target_id] = set()
        cache._target_to_sats[target_id].add(sat_id)

    # 加载卫星-地面站窗口
    for w_data in data.get('ground_station_windows', []):
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

    # 对所有窗口排序
    for key in cache._windows:
        sorted_pairs = sorted(zip(cache._time_index[key], cache._windows[key]))
        cache._time_index[key] = [p[0] for p in sorted_pairs]
        cache._windows[key] = [p[1] for p in sorted_pairs]

    return cache


def run_single_algorithm(algorithm_name: str, mission: Mission, cache: VisibilityWindowCache):
    """运行单个算法并返回结果"""
    if algorithm_name not in SCHEDULER_REGISTRY:
        return None, f"未知算法: {algorithm_name}"

    scheduler_class, info = SCHEDULER_REGISTRY[algorithm_name]
    config = info.get('params', {})

    print(f"\n  运行 {info['name']} ({algorithm_name})...", end='', flush=True)

    try:
        scheduler = scheduler_class(config)
        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        start_time = time.time()
        result = scheduler.schedule()
        elapsed = time.time() - start_time

        # 计算指标
        metrics_calc = MetricsCalculator(mission)
        metrics = metrics_calc.calculate_all(result)

        print(f" ✓ {elapsed:.2f}s, {metrics.scheduled_task_count} tasks")

        return {
            'algorithm': algorithm_name,
            'name': info['name'],
            'category': info['category'],
            'scheduled_count': metrics.scheduled_task_count,
            'unscheduled_count': metrics.unscheduled_task_count,
            'demand_satisfaction_rate': metrics.demand_satisfaction_rate,
            'makespan_hours': metrics.makespan / 3600,
            'computation_time': metrics.computation_time,
            'solution_quality': metrics.solution_quality,
            'satellite_utilization': metrics.satellite_utilization,
            'convergence_curve': result.convergence_curve[:50] if result.convergence_curve else [],
        }, None

    except Exception as e:
        print(f" ✗ 失败: {e}")
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description='运行所有调度算法对比')
    parser.add_argument('--cache', '-c', required=True, help='缓存文件路径')
    parser.add_argument('--scenario', '-s', required=True, help='场景文件路径')
    parser.add_argument('--output', '-o', default='results/algorithm_comparison.json', help='输出文件路径')
    parser.add_argument('--algorithms', '-a', nargs='+', help='指定要运行的算法 (默认全部)')

    args = parser.parse_args()

    print("=" * 70)
    print("调度算法对比实验")
    print("=" * 70)

    # 加载场景
    print(f"\n[1/3] 加载场景: {args.scenario}")
    mission = Mission.load(args.scenario)
    print(f"  卫星: {len(mission.satellites)} 颗")
    print(f"  目标: {len(mission.targets)} 个")
    print(f"  地面站: {len(mission.ground_stations)} 个")

    # 加载缓存
    print(f"\n[2/3] 加载缓存: {args.cache}")
    cache = load_window_cache_from_json(args.cache, mission)
    stats = cache.get_statistics()
    print(f"  可用窗口数: {stats['total_windows']}")
    print(f"  卫星-目标对: {stats['sat_target_pairs']}")

    # 确定要运行的算法
    algorithms_to_run = args.algorithms if args.algorithms else list(SCHEDULER_REGISTRY.keys())
    print(f"\n[3/3] 运行 {len(algorithms_to_run)} 个算法:")

    # 运行所有算法
    results = []
    errors = {}

    for algo_name in algorithms_to_run:
        result, error = run_single_algorithm(algo_name, mission, cache)
        if result:
            results.append(result)
        else:
            errors[algo_name] = error

    # 输出对比结果
    print("\n" + "=" * 70)
    print("算法对比结果")
    print("=" * 70)

    # 按类别分组
    greedy_results = [r for r in results if r['category'] == '贪心']
    meta_results = [r for r in results if r['category'] == '元启发式']

    if greedy_results:
        print("\n【贪心算法】")
        print(f"{'算法':<12} {'调度任务':>8} {'满足率':>8} {'用时(秒)':>10} {'解质量':>8}")
        print("-" * 60)
        for r in greedy_results:
            print(f"{r['name']:<12} {r['scheduled_count']:>8} {r['demand_satisfaction_rate']:>7.1%} {r['computation_time']:>10.2f} {r['solution_quality']:>8.4f}")

    if meta_results:
        print("\n【元启发式算法】")
        print(f"{'算法':<12} {'调度任务':>8} {'满足率':>8} {'用时(秒)':>10} {'解质量':>8}")
        print("-" * 60)
        for r in meta_results:
            print(f"{r['name']:<12} {r['scheduled_count']:>8} {r['demand_satisfaction_rate']:>7.1%} {r['computation_time']:>10.2f} {r['solution_quality']:>8.4f}")

    # 找出最优算法
    if results:
        best_dsr = max(results, key=lambda x: x['demand_satisfaction_rate'])
        best_quality = max(results, key=lambda x: x['solution_quality'])
        fastest = min(results, key=lambda x: x['computation_time'])

        print("\n" + "=" * 70)
        print("最优算法")
        print("=" * 70)
        print(f"  最高需求满足率: {best_dsr['name']} ({best_dsr['demand_satisfaction_rate']:.2%})")
        print(f"  最优解质量: {best_quality['name']} ({best_quality['solution_quality']:.4f})")
        print(f"  最快求解速度: {fastest['name']} ({fastest['computation_time']:.3f}秒)")

    if errors:
        print("\n失败的算法:")
        for algo, error in errors.items():
            print(f"  {algo}: {error}")

    print("=" * 70)

    # 保存结果
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'scenario': mission.name,
            'cache_file': args.cache,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'errors': errors,
            'summary': {
                'best_dsr': best_dsr['algorithm'] if results else None,
                'best_quality': best_quality['algorithm'] if results else None,
                'fastest': fastest['algorithm'] if results else None,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
