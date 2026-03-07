#!/usr/bin/env python3
"""
使用预计算缓存运行调度算法

Usage:
    python scripts/run_scheduler_with_cache.py --cache data/visibility_cache/point_group_scenario_windows.json --algorithm ga
    python scripts/run_scheduler_with_cache.py --help
"""

import argparse
import json
import sys
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
from evaluation.metrics import MetricsCalculator
import json as json_module


# 调度器注册表
SCHEDULER_REGISTRY = {
    'greedy': GreedyScheduler,
    'ga': GAScheduler,
    'edd': EDDScheduler,
    'spt': SPTScheduler,
}


def load_window_cache_from_json(cache_path: str, mission: Mission) -> VisibilityWindowCache:
    """
    从JSON文件加载预计算的窗口缓存

    Args:
        cache_path: 缓存文件路径
        mission: 场景对象（用于获取卫星和目标信息）

    Returns:
        VisibilityWindowCache: 填充好的缓存对象
    """
    print(f"加载缓存文件: {cache_path}")

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache = VisibilityWindowCache()

    # 创建卫星和目标ID映射
    sat_map = {sat.id: sat for sat in mission.satellites}
    target_map = {target.id: target for target in mission.targets}
    gs_map = {gs.id: gs for gs in mission.ground_stations}

    # 加载卫星-目标窗口
    target_windows_count = 0
    for w_data in data.get('target_windows', []):
        sat_id = w_data['satellite_id']
        target_id = w_data['target_id']

        # 创建VisibilityWindow对象
        window = VisibilityWindow(
            satellite_id=sat_id,
            target_id=target_id,
            start_time=datetime.fromisoformat(w_data['start_time']),
            end_time=datetime.fromisoformat(w_data['end_time']),
            max_elevation=w_data.get('max_elevation', 0.0)
        )

        # 手动添加到缓存
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
    for w_data in data.get('ground_station_windows', []):
        sat_id = w_data['satellite_id']
        target_id = w_data['target_id']  # 对于地面站，target_id格式为"GS:gs_id"

        # 创建VisibilityWindow对象
        window = VisibilityWindow(
            satellite_id=sat_id,
            target_id=target_id,
            start_time=datetime.fromisoformat(w_data['start_time']),
            end_time=datetime.fromisoformat(w_data['end_time']),
            max_elevation=w_data.get('max_elevation', 0.0)
        )

        # 手动添加到缓存
        key = (sat_id, target_id)
        if key not in cache._windows:
            cache._windows[key] = []
            cache._time_index[key] = []

        cache._windows[key].append(window)
        cache._time_index[key].append(window.start_time)

        gs_windows_count += 1

    # 对所有窗口排序
    for key in cache._windows:
        # 按开始时间排序
        sorted_pairs = sorted(zip(cache._time_index[key], cache._windows[key]))
        cache._time_index[key] = [p[0] for p in sorted_pairs]
        cache._windows[key] = [p[1] for p in sorted_pairs]

    print(f"  加载完成: {target_windows_count} 个卫星-目标窗口, {gs_windows_count} 个卫星-地面站窗口")

    return cache


def run_scheduler(args):
    """运行调度算法"""
    print("=" * 60)
    print("使用预计算缓存运行调度算法")
    print("=" * 60)

    # 1. 加载场景
    print(f"\n[1/4] 加载场景: {args.scenario}")
    mission = Mission.load(args.scenario)
    print(f"  卫星: {len(mission.satellites)} 颗")
    print(f"  目标: {len(mission.targets)} 个")
    print(f"  地面站: {len(mission.ground_stations)} 个")

    # 2. 加载缓存
    print(f"\n[2/4] 加载预计算窗口缓存")
    cache = load_window_cache_from_json(args.cache, mission)
    stats = cache.get_statistics()
    print(f"  总窗口数: {stats['total_windows']}")
    print(f"  卫星-目标对: {stats['sat_target_pairs']}")
    print(f"  平均每对窗口数: {stats['avg_windows_per_pair']:.2f}")

    # 3. 创建并运行调度器
    print(f"\n[3/4] 运行调度算法: {args.algorithm.upper()}")

    if args.algorithm.lower() not in SCHEDULER_REGISTRY:
        print(f"错误: 未知算法 '{args.algorithm}'")
        print(f"可用算法: {', '.join(SCHEDULER_REGISTRY.keys())}")
        sys.exit(1)

    scheduler_class = SCHEDULER_REGISTRY[args.algorithm.lower()]

    # 解析算法配置
    scheduler_config = {}
    if args.config:
        import json
        scheduler_config = json.loads(args.config)

    # 设置GA默认参数（如果适用）
    if args.algorithm.lower() == 'ga' and not scheduler_config:
        scheduler_config = {
            'population_size': args.population_size,
            'generations': args.generations,
            'mutation_rate': args.mutation_rate,
            'crossover_rate': args.crossover_rate,
        }
        print(f"  GA参数:")
        print(f"    种群大小: {scheduler_config['population_size']}")
        print(f"    迭代次数: {scheduler_config['generations']}")
        print(f"    变异率: {scheduler_config['mutation_rate']}")
        print(f"    交叉率: {scheduler_config['crossover_rate']}")

    scheduler = scheduler_class(scheduler_config)
    scheduler.initialize(mission)
    scheduler.set_window_cache(cache)

    print(f"\n  开始调度...")
    import time
    t0 = time.time()
    result = scheduler.schedule()
    elapsed = time.time() - t0

    print(f"  调度完成: {elapsed:.2f} 秒")

    # 4. 计算性能指标
    print(f"\n[4/4] 计算性能指标")
    metrics_calc = MetricsCalculator(mission)
    metrics = metrics_calc.calculate_all(result)

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("调度结果")
    print("=" * 60)
    print(f"算法: {args.algorithm.upper()}")
    print(f"场景: {mission.name}")
    print(f"-" * 60)
    print(f"成功调度任务: {metrics.scheduled_task_count}")
    print(f"未调度任务: {metrics.unscheduled_task_count}")
    print(f"需求满足率 (DSR): {metrics.demand_satisfaction_rate:.2%}")
    print(f"总完成时间 (Makespan): {metrics.makespan/3600:.2f} 小时")
    print(f"算法求解用时: {metrics.computation_time:.2f} 秒")
    print(f"解质量: {metrics.solution_quality:.4f}")
    print(f"卫星利用率: {metrics.satellite_utilization:.2%}")
    print("=" * 60)

    # 6. 输出收敛曲线（GA算法）
    if result.convergence_curve and len(result.convergence_curve) > 0:
        print(f"\n收敛曲线:")
        print(f"  迭代次数: {len(result.convergence_curve)}")
        print(f"  初始适应度: {result.convergence_curve[0]:.4f}")
        print(f"  最终适应度: {result.convergence_curve[-1]:.4f}")
        improvement = result.convergence_curve[0] - result.convergence_curve[-1]
        print(f"  优化提升: {improvement:.4f}")

    # 7. 保存结果（可选）
    if args.output:
        result_data = {
            'algorithm': args.algorithm,
            'scenario': mission.name,
            'cache_file': args.cache,
            'metrics': {
                'scheduled_task_count': metrics.scheduled_task_count,
                'unscheduled_task_count': metrics.unscheduled_task_count,
                'demand_satisfaction_rate': metrics.demand_satisfaction_rate,
                'makespan_hours': metrics.makespan / 3600,
                'computation_time': metrics.computation_time,
                'solution_quality': metrics.solution_quality,
                'satellite_utilization': metrics.satellite_utilization,
            },
            'scheduled_tasks': [
                {
                    'task_id': t.task_id,
                    'satellite_id': t.satellite_id,
                    'target_id': t.target_id,
                    'start_time': t.imaging_start.isoformat(),
                    'end_time': t.imaging_end.isoformat(),
                }
                for t in result.scheduled_tasks[:100]  # 只保存前100个
            ]
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json_module.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output}")

    return result, metrics


def main():
    parser = argparse.ArgumentParser(
        description='使用预计算缓存运行调度算法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用GA算法
  python scripts/run_scheduler_with_cache.py \\
      --cache data/visibility_cache/point_group_scenario_windows.json \\
      --scenario scenarios/point_group_scenario.json \\
      --algorithm ga

  # 使用贪心算法并保存结果
  python scripts/run_scheduler_with_cache.py \\
      --cache data/visibility_cache/point_group_scenario_windows.json \\
      --scenario scenarios/point_group_scenario.json \\
      --algorithm greedy \\
      --output results/greedy_result.json
        """
    )

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

    parser.add_argument(
        '--algorithm', '-a',
        required=True,
        choices=list(SCHEDULER_REGISTRY.keys()),
        help='调度算法名称'
    )

    parser.add_argument(
        '--config',
        help='算法参数配置 (JSON格式字符串)'
    )

    # GA算法参数
    parser.add_argument(
        '--population-size',
        type=int,
        default=50,
        help='GA种群大小 (默认: 50)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='GA迭代次数 (默认: 100)'
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

    parser.add_argument(
        '--output', '-o',
        help='输出结果文件路径 (JSON格式)'
    )

    args = parser.parse_args()

    try:
        run_scheduler(args)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
