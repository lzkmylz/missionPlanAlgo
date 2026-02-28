#!/usr/bin/env python3
"""
频次约束场景算法对比实验脚本

运行5种算法在带频次约束场景上的对比实验
支持多次观测需求和重访周期约束

用法:
    python scripts/run_frequency_comparison.py
    python scripts/run_frequency_comparison.py --scenario scenarios/large_scale_frequency.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Type
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from evaluation.metrics import MetricsCalculator, PerformanceMetrics
from scheduler.base_scheduler import BaseScheduler, ScheduleResult
from scheduler.frequency_utils import (
    create_observation_tasks,
    calculate_frequency_fitness,
    is_target_fully_satisfied
)

# 导入所有调度器
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler
from scheduler.ground_station_scheduler import GroundStationScheduler
from core.resources.ground_station_pool import GroundStationPool

# 算法注册表
ALGORITHM_REGISTRY: Dict[str, Type[BaseScheduler]] = {
    'FCFS': SPTScheduler,
    'Greedy-EDF': EDDScheduler,
    'Greedy-MaxVal': GreedyScheduler,
    'GA': GAScheduler,
    'SA': SAScheduler,
}


def load_mission_with_frequency(scenario_path: str) -> tuple:
    """从场景文件加载Mission对象（支持频次约束）"""
    from scripts.load_large_scale_scenario import (
        load_satellites, load_targets, load_ground_stations
    )
    from core.models.mission import Mission

    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    satellites = load_satellites(scenario_data)
    targets = load_targets(scenario_data)
    ground_stations = load_ground_stations(scenario_data)

    duration = scenario_data['duration']
    start_time = datetime.fromisoformat(duration['start'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(duration['end'].replace('Z', '+00:00'))

    mission = Mission(
        name=scenario_data['name'],
        satellites=satellites,
        targets=targets,
        ground_stations=ground_stations,
        start_time=start_time,
        end_time=end_time
    )

    # 计算总观测需求（用于统计）
    total_obs_demand = sum(getattr(t, 'required_observations', 1) for t in targets)

    return mission, targets, total_obs_demand, scenario_data


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


def run_algorithm_with_frequency(
    algorithm_name: str,
    mission: Mission,
    targets: List,
    window_cache,
    config: Dict[str, Any],
    enable_downlink: bool = True
) -> Dict[str, Any]:
    """运行单个算法（支持频次约束和地面站数传）"""

    scheduler_class = ALGORITHM_REGISTRY.get(algorithm_name)
    if not scheduler_class:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    print(f"\n  运行 {algorithm_name}...")

    # 实例化调度器（使用原始目标，调度器内部会展开多次观测）
    scheduler = scheduler_class(config)
    scheduler.initialize(mission)

    # 注入窗口缓存
    if window_cache and hasattr(scheduler, 'set_window_cache'):
        scheduler.set_window_cache(window_cache)

    # 运行调度
    start_time = time.time()
    result = scheduler.schedule()
    base_computation_time = time.time() - start_time

    # 集成地面站数传调度
    downlink_count = 0
    if enable_downlink and result.scheduled_tasks and mission.ground_stations:
        print(f"    正在为 {len(result.scheduled_tasks)} 个任务安排数传计划...")

        # 创建地面站资源池
        gs_pool = GroundStationPool(mission.ground_stations)

        # 创建地面站调度器
        gs_scheduler = GroundStationScheduler(
            ground_station_pool=gs_pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=128.0,
            overflow_threshold=0.95
        )

        # 初始化卫星固存状态
        for sat in mission.satellites:
            storage_capacity = getattr(sat.capabilities, 'storage_capacity', 128.0)
            gs_scheduler.initialize_satellite_storage(sat.id, storage_capacity)

        # 准备地面站可见性窗口 (从 window_cache 提取)
        gs_visibility_windows = {}
        if window_cache:
            for task in result.scheduled_tasks:
                sat_id = task.satellite_id
                if sat_id not in gs_visibility_windows:
                    gs_visibility_windows[sat_id] = []
                    # 获取该卫星的所有地面站窗口
                    for gs in mission.ground_stations:
                        windows = window_cache.get_windows(sat_id, gs.id)
                        for w in windows:
                            gs_visibility_windows[sat_id].append((w.start_time, w.end_time))

        # 为成像任务安排数传
        downlink_start_time = time.time()
        gs_result = gs_scheduler.schedule_downlinks_for_tasks(
            scheduled_tasks=result.scheduled_tasks,
            visibility_windows=gs_visibility_windows
        )
        downlink_time = time.time() - downlink_start_time

        downlink_count = len(gs_result.downlink_tasks)
        print(f"    完成数传计划: {downlink_count} 个数传任务")

        # 更新任务信息（添加数传信息）
        updated_tasks = gs_scheduler.update_tasks_with_downlink_info(
            result.scheduled_tasks,
            gs_result.downlink_tasks
        )

        # 更新计算时间
        computation_time = base_computation_time + downlink_time

        # 使用更新后的任务列表
        result = type('Result', (), {'scheduled_tasks': updated_tasks})()
    else:
        computation_time = base_computation_time

    # 统计频次满足情况
    target_obs_count = defaultdict(int)
    for task in result.scheduled_tasks:
        # 从任务ID提取原始目标ID (格式: TGT-XXXX-OBSN)
        target_id = task.target_id.split('-OBS')[0] if '-OBS' in task.target_id else task.target_id
        target_obs_count[target_id] += 1

    # 计算频次满足率
    total_targets = len(mission.targets)
    satisfied_targets = sum(
        1 for t in mission.targets
        if is_target_fully_satisfied(t, target_obs_count.get(t.id, 0))
    )
    frequency_satisfaction = satisfied_targets / total_targets if total_targets > 0 else 0

    # 计算总观测需求满足率
    total_required = sum(getattr(t, 'required_observations', 1) for t in mission.targets)
    total_actual = sum(target_obs_count.values())
    observation_completion = total_actual / total_required if total_required > 0 else 0

    return {
        'algorithm': algorithm_name,
        'scheduled_count': len(result.scheduled_tasks),
        'downlink_count': downlink_count,
        'unique_targets_observed': len(target_obs_count),
        'computation_time': computation_time,
        'frequency_satisfaction_rate': frequency_satisfaction,
        'observation_completion_rate': observation_completion,
        'total_required': total_required,
        'total_actual': total_actual,
    }


def run_comparison(
    scenario_path: str,
    visibility_path: str,
    algorithms: List[str],
    repetitions: int,
    output_path: str
) -> Dict[str, Any]:
    """运行频次约束场景对比实验"""

    print("=" * 70)
    print("频次约束场景算法对比实验")
    print("=" * 70)

    # 加载场景
    print(f"\n[1/3] 加载场景...")
    mission, targets, total_obs_demand, scenario_data = load_mission_with_frequency(scenario_path)

    print(f"  - 卫星: {len(mission.satellites)} 颗")
    print(f"  - 目标: {len(mission.targets)} 个")
    print(f"  - 总观测需求: {total_obs_demand} 次")

    stats = scenario_data.get('statistics', {})
    if stats:
        print(f"\n  频次分布:")
        for freq, count in stats.get('targets_by_frequency', {}).items():
            print(f"    {freq}次观测: {count} 个目标")

    # 加载可见性缓存
    print(f"\n[2/3] 加载可见性缓存...")
    window_cache = None
    if visibility_path and Path(visibility_path).exists():
        window_cache = load_visibility_cache(visibility_path)
        stats = window_cache.get_statistics()
        print(f"  - 窗口总数: {stats.get('total_windows', 0):,}")
    else:
        print(f"  - 警告: 可见性缓存不存在，调度可能无法正常工作")

    # 运行算法对比
    print(f"\n[3/3] 运行算法对比")
    all_results: Dict[str, List[Dict]] = {alg: [] for alg in algorithms}

    for alg_name in algorithms:
        print(f"\n{'='*70}")
        print(f"算法: {alg_name}")
        print(f"{'='*70}")

        for rep in range(1, repetitions + 1):
            print(f"\n  重复 {rep}/{repetitions}")

            try:
                result = run_algorithm_with_frequency(
                    alg_name, mission, targets, window_cache, {}
                )
                result['repetition'] = rep
                all_results[alg_name].append(result)

                print(f"    完成: {result['scheduled_count']} 任务")
                if result.get('downlink_count', 0) > 0:
                    print(f"    数传任务: {result['downlink_count']} 个")
                print(f"    频次满足率: {result['frequency_satisfaction_rate']:.1%}")
                print(f"    观测完成率: {result['observation_completion_rate']:.1%}")
                print(f"    耗时: {result['computation_time']:.2f}s")

            except Exception as e:
                print(f"    错误: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果
    output_data = {
        'metadata': {
            'scenario': scenario_path,
            'algorithms': algorithms,
            'repetitions': repetitions,
            'generated_at': datetime.now().isoformat(),
            'total_observation_demand': total_obs_demand
        },
        'raw_results': all_results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")

    # 打印对比表
    print_comparison(all_results)

    return output_data


def print_comparison(all_results: Dict[str, List[Dict]]):
    """打印对比结果"""

    print("\n" + "=" * 70)
    print("算法对比结果")
    print("=" * 70)

    print(f"\n{'算法':<15} {'任务数':<10} {'频次满足':<12} {'观测完成':<12} {'计算时间':<10}")
    print("-" * 70)

    for alg_name, results in all_results.items():
        if results:
            r = results[0]  # 取第一次重复结果
            print(f"{alg_name:<15} {r['scheduled_count']:<10} "
                  f"{r['frequency_satisfaction_rate']:<12.1%} "
                  f"{r['observation_completion_rate']:<12.1%} "
                  f"{r['computation_time']:<10.2f}s")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='频次约束场景算法对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--scenario', '-s',
        default='scenarios/large_scale_frequency.json',
        help='场景文件路径'
    )
    parser.add_argument(
        '--visibility', '-v',
        default='results/large_scale_visibility.json',
        help='可见性缓存文件路径'
    )
    parser.add_argument(
        '--algorithms', '-a',
        default='FCFS,Greedy-EDF,Greedy-MaxVal',
        help='算法列表，逗号分隔'
    )
    parser.add_argument(
        '--repetitions', '-r',
        type=int,
        default=1,
        help='重复次数'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/frequency_comparison.json',
        help='输出路径'
    )

    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(',')]

    try:
        run_comparison(
            scenario_path=args.scenario,
            visibility_path=args.visibility,
            algorithms=algorithms,
            repetitions=args.repetitions,
            output_path=args.output
        )
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
