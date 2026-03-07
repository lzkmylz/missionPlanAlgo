#!/usr/bin/env python3
"""
完整性能测试脚本 - 测试所有调度算法

测试算法 (7个):
1. Greedy (贪心)
2. EDD (最早截止时间优先)
3. GA (遗传算法)
4. SA (模拟退火)
5. ACO (蚁群优化)
6. PSO (粒子群优化)
7. Tabu (禁忌搜索)

场景: 60颗卫星(30光学+30SAR) x 1000目标 x 24小时
      带频次观测需求和地面站数传

包含完整的可见性窗口计算过程
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
    """从JSON文件加载预计算的窗口缓存"""
    logger.info(f"加载缓存文件: {cache_path}")

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache = VisibilityWindowCache()

    # 加载卫星-目标窗口
    target_windows_count = 0
    target_windows = data.get('target_windows', [])

    if not target_windows and 'windows' in data:
        for w_data in data['windows']:
            sat_id = w_data.get('sat') or w_data.get('satellite_id')
            target_id = w_data.get('tgt') or w_data.get('target_id')
            start_time = w_data.get('start') or w_data.get('start_time')
            end_time = w_data.get('end') or w_data.get('end_time')
            max_el = w_data.get('el', 0.0) or w_data.get('max_elevation', 0.0)

            if not all([sat_id, target_id, start_time, end_time]):
                continue
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

            if sat_id not in cache._sat_to_targets:
                cache._sat_to_targets[sat_id] = set()
            cache._sat_to_targets[sat_id].add(target_id)

            if target_id not in cache._target_to_sats:
                cache._target_to_sats[target_id] = set()
            cache._target_to_sats[target_id].add(sat_id)

            target_windows_count += 1

    # 加载卫星-地面站窗口
    gs_windows_count = 0
    gs_windows_list = data.get('ground_station_windows', [])

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


def test_algorithm(
    algorithm_name: str,
    algorithm_key: str,
    mission: Mission,
    cache: VisibilityWindowCache,
    ground_station_pool: Optional[GroundStationPool],
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """测试单个算法"""

    print(f"\n{'='*70}")
    print(f"测试算法: {algorithm_name} ({algorithm_key.upper()})")
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

        # 创建并运行调度器
        scheduler = UnifiedScheduler(
            mission=mission,
            window_cache=cache,
            ground_station_pool=ground_station_pool,
            config=config
        )

        result = scheduler.schedule()
        total_time = time.time() - start_time

        # 计算指标
        metrics_calc = MetricsCalculator(mission)
        imaging_metrics = metrics_calc.calculate_all(result.imaging_result)

        # 频次满足度统计
        satisfied_count = sum(1 for info in result.target_observations.values() if info['satisfied'])
        total_targets = len(result.target_observations)
        frequency_satisfaction_rate = satisfied_count / total_targets if total_targets > 0 else 0.0

        # 收集结果
        result_data.update({
            'status': 'success',
            'metrics': {
                'scheduled_tasks': len(result.imaging_result.scheduled_tasks),
                'unscheduled_tasks': len(result.imaging_result.unscheduled_tasks),
                'total_tasks': len(result.imaging_result.scheduled_tasks) + len(result.imaging_result.unscheduled_tasks),
                'schedule_rate': len(result.imaging_result.scheduled_tasks) / (
                    len(result.imaging_result.scheduled_tasks) + len(result.imaging_result.unscheduled_tasks)
                ) if (len(result.imaging_result.scheduled_tasks) + len(result.imaging_result.unscheduled_tasks)) > 0 else 0.0,
                'demand_satisfaction_rate': imaging_metrics.demand_satisfaction_rate,
                'satellite_utilization': imaging_metrics.satellite_utilization,
                'makespan_hours': result.imaging_result.makespan / 3600,
                'imaging_computation_time': result.imaging_result.computation_time,
                'total_computation_time': total_time,
                'frequency_satisfied': satisfied_count,
                'frequency_total': total_targets,
                'frequency_satisfaction_rate': frequency_satisfaction_rate,
                'downlink_tasks': len(result.downlink_result.downlink_tasks) if result.downlink_result else 0,
                'downlink_failed': len(result.downlink_result.failed_tasks) if result.downlink_result else 0,
                'iterations': result.imaging_result.iterations,
            }
        })

        # 打印结果
        print(f"\n结果:")
        print(f"  成功调度: {result_data['metrics']['scheduled_tasks']} 个任务")
        print(f"  调度成功率: {result_data['metrics']['schedule_rate']:.2%}")
        print(f"  需求满足率: {result_data['metrics']['demand_satisfaction_rate']:.2%}")
        print(f"  卫星利用率: {result_data['metrics']['satellite_utilization']:.2%}")
        print(f"  频次满足率: {result_data['metrics']['frequency_satisfaction_rate']:.2%}")
        print(f"  数传任务: {result_data['metrics']['downlink_tasks']} 成功, {result_data['metrics']['downlink_failed']} 失败")
        print(f"  求解用时: {result_data['metrics']['total_computation_time']:.2f} 秒")
        print(f"  迭代次数: {result_data['metrics']['iterations']}")

        # 保存详细结果（包含任务列表）
        if output_dir:
            output_file = output_dir / f"result_{algorithm_key.lower()}.json"

            # 构建完整结果，包含所有任务详情
            full_result = result.to_dict()

            # 添加详细的成像任务列表
            full_result['imaging']['scheduled_tasks'] = []
            for task in result.imaging_result.scheduled_tasks:
                # 计算成像时长
                imaging_duration = 0
                if task.imaging_start and task.imaging_end:
                    imaging_duration = (task.imaging_end - task.imaging_start).total_seconds()

                task_dict = {
                    'task_id': task.task_id,
                    'satellite_id': task.satellite_id,
                    'target_id': task.target_id,
                    'imaging_start': task.imaging_start.isoformat() if task.imaging_start else None,
                    'imaging_end': task.imaging_end.isoformat() if task.imaging_end else None,
                    'imaging_duration': imaging_duration,
                    'imaging_mode': task.imaging_mode,
                    'slew_angle': task.slew_angle,
                    'slew_time': task.slew_time,
                    'ground_station_id': task.ground_station_id,
                    'downlink_start': task.downlink_start.isoformat() if task.downlink_start else None,
                    'downlink_end': task.downlink_end.isoformat() if task.downlink_end else None,
                    'data_transferred': task.data_transferred,
                }
                full_result['imaging']['scheduled_tasks'].append(task_dict)

            # 添加详细的数传任务列表
            if result.downlink_result:
                full_result['downlink']['downlink_tasks'] = []
                for dl_task in result.downlink_result.downlink_tasks:
                    dl_dict = {
                        'task_id': dl_task.task_id,
                        'satellite_id': dl_task.satellite_id,
                        'ground_station_id': dl_task.ground_station_id,
                        'antenna_id': dl_task.antenna_id,
                        'start_time': dl_task.start_time.isoformat() if dl_task.start_time else None,
                        'end_time': dl_task.end_time.isoformat() if dl_task.end_time else None,
                        'data_size_gb': dl_task.data_size_gb,
                        'related_imaging_task': dl_task.related_imaging_task_id,
                    }
                    full_result['downlink']['downlink_tasks'].append(dl_dict)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_result, f, indent=2, ensure_ascii=False)
            print(f"  详细结果已保存: {output_file}")

    except Exception as e:
        result_data['error'] = str(e)
        result_data['traceback'] = traceback.format_exc()
        logger.error(f"算法 {algorithm_name} 运行失败: {e}")
        traceback.print_exc()

    return result_data


def run_benchmark(
    scenario_path: str,
    cache_path: str,
    output_dir: str,
    algorithms: List[str],
    enable_downlink: bool = True,
    seed: int = 42
):
    """运行性能测试"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("大规模场景规划算法完整性能测试")
    print("="*70)
    print(f"场景文件: {scenario_path}")
    print(f"缓存文件: {cache_path}")
    print(f"输出目录: {output_dir}")
    print(f"测试算法: {', '.join(algorithms)}")
    print(f"随机种子: {seed}")
    print("="*70)

    # 1. 加载场景
    print(f"\n[1/3] 加载场景...")
    mission = Mission.load(scenario_path)
    print(f"  卫星: {len(mission.satellites)} 颗")
    print(f"  目标: {len(mission.targets)} 个")
    print(f"  地面站: {len(mission.ground_stations)} 个")

    # 统计频次需求
    freq_required = sum(1 for t in mission.targets if getattr(t, 'required_observations', 1) > 1)
    total_demand = sum(getattr(t, 'required_observations', 1) for t in mission.targets)
    print(f"  有频次需求的目标: {freq_required} 个")
    print(f"  总观测需求: {total_demand} 次")

    # 2. 加载缓存
    print(f"\n[2/3] 加载预计算窗口缓存...")
    cache = load_window_cache_from_json(cache_path, mission)
    stats = cache.get_statistics()
    print(f"  总窗口数: {stats['total_windows']}")
    print(f"  卫星-目标对: {stats['sat_target_pairs']}")

    # 3. 准备地面站资源池
    print(f"\n[3/3] 初始化地面站资源池...")
    ground_station_pool = None
    if enable_downlink and mission.ground_stations:
        ground_station_pool = GroundStationPool(mission.ground_stations)
        total_antennas = sum(len(gs.antennas) for gs in mission.ground_stations)
        print(f"  地面站: {len(mission.ground_stations)} 个")
        print(f"  天线总数: {total_antennas} 个")
    else:
        print(f"  跳过地面站初始化")

    # 4. 测试各个算法
    print(f"\n{'='*70}")
    print("开始测试算法...")
    print(f"{'='*70}")

    results = []

    # 算法配置
    algorithm_configs = {
        'greedy': {
            'imaging_algorithm': 'greedy',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
        'edd': {
            'imaging_algorithm': 'edd',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
        'ga': {
            'imaging_algorithm': 'ga',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'random_seed': seed,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
        'sa': {
            'imaging_algorithm': 'sa',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
                'initial_temperature': 100.0,
                'cooling_rate': 0.98,
                'max_iterations': 1000,
                'min_temperature': 0.001,
                'random_seed': seed,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
        'aco': {
            'imaging_algorithm': 'aco',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
                'num_ants': 30,
                'max_iterations': 100,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.1,
                'random_seed': seed,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
        'pso': {
            'imaging_algorithm': 'pso',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
                'num_particles': 30,
                'max_iterations': 100,
                'inertia_weight': 0.9,
                'cognitive_coeff': 2.0,
                'social_coeff': 2.0,
                'random_seed': seed,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
        'tabu': {
            'imaging_algorithm': 'tabu',
            'imaging_config': {
                'use_simplified_slew': True,
                'consider_power': True,
                'consider_storage': True,
                'tabu_tenure': 10,
                'max_iterations': 100,
                'neighborhood_size': 20,
                'aspiration_threshold': 0.05,
                'random_seed': seed,
            },
            'enable_downlink': enable_downlink,
            'consider_frequency': True,
        },
    }

    algorithm_names = {
        'greedy': 'Greedy (贪心)',
        'edd': 'EDD (最早截止时间优先)',
        'ga': 'GA (遗传算法)',
        'sa': 'SA (模拟退火)',
        'aco': 'ACO (蚁群优化)',
        'pso': 'PSO (粒子群优化)',
        'tabu': 'Tabu (禁忌搜索)',
    }

    # 运行测试
    for algo_key in algorithms:
        if algo_key not in algorithm_configs:
            print(f"\n跳过未知算法: {algo_key}")
            continue

        config = algorithm_configs[algo_key]
        result = test_algorithm(
            algorithm_name=algorithm_names.get(algo_key, algo_key),
            algorithm_key=algo_key,
            mission=mission,
            cache=cache,
            ground_station_pool=ground_station_pool,
            config=config,
            output_dir=output_path
        )
        results.append(result)

    # 5. 生成总结报告
    generate_report(results, output_path)

    return results


def generate_report(results: List[Dict[str, Any]], output_dir: Path):
    """生成测试报告"""

    report_file = output_dir / "benchmark_complete_report.json"
    summary_file = output_dir / "benchmark_complete_summary.txt"

    # 保存完整报告
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'scenario': '大规模星座任务规划（60卫星x1000目标x24小时）',
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
        f.write("规划算法完整性能测试报告\n")
        f.write("="*80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试场景: 大规模星座任务规划（60卫星x1000目标x24小时，带频次需求）\n")
        f.write("="*80 + "\n\n")

        # 成功的算法
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] != 'success']

        f.write(f"【测试统计】\n")
        f.write(f"  测试算法总数: {len(results)}\n")
        f.write(f"  成功: {len(successful)}\n")
        f.write(f"  失败: {len(failed)}\n\n")

        if successful:
            f.write("【成功运行的算法】\n\n")

            # 排序：按需求满足率
            successful_sorted = sorted(
                successful,
                key=lambda x: x['metrics'].get('demand_satisfaction_rate', 0),
                reverse=True
            )

            # 表格头
            f.write(f"{'算法':<30} {'调度成功':<10} {'满足率':<10} {'利用率':<10} {'频次满足':<10} {'用时(秒)':<12} {'迭代':<8}\n")
            f.write("-"*100 + "\n")

            for r in successful_sorted:
                m = r['metrics']
                algo_name = r['algorithm']
                if len(algo_name) > 28:
                    algo_name = algo_name[:25] + "..."
                f.write(f"{algo_name:<30} "
                        f"{m.get('schedule_rate', 0):<10.2%} "
                        f"{m.get('demand_satisfaction_rate', 0):<10.2%} "
                        f"{m.get('satellite_utilization', 0):<10.2%} "
                        f"{m.get('frequency_satisfaction_rate', 0):<10.2%} "
                        f"{m.get('total_computation_time', 0):<12.2f} "
                        f"{m.get('iterations', 0):<8}\n")

            f.write("\n")

            # 性能排名
            f.write("【性能排名（按求解速度）】\n\n")
            speed_sorted = sorted(successful, key=lambda x: x['metrics'].get('total_computation_time', float('inf')))
            for i, r in enumerate(speed_sorted, 1):
                m = r['metrics']
                f.write(f"  {i}. {r['algorithm']}: {m.get('total_computation_time', 0):.2f}秒\n")

            f.write("\n")

            # 详细指标
            f.write("【详细指标】\n\n")
            for r in successful_sorted:
                m = r['metrics']
                f.write(f"\n{r['algorithm']}:\n")
                f.write(f"  - 成功调度任务: {m.get('scheduled_tasks', 0)}\n")
                f.write(f"  - 未调度任务: {m.get('unscheduled_tasks', 0)}\n")
                f.write(f"  - 调度成功率: {m.get('schedule_rate', 0):.2%}\n")
                f.write(f"  - 需求满足率: {m.get('demand_satisfaction_rate', 0):.2%}\n")
                f.write(f"  - 卫星利用率: {m.get('satellite_utilization', 0):.2%}\n")
                f.write(f"  - 完成时间跨度: {m.get('makespan_hours', 0):.2f} 小时\n")
                f.write(f"  - 频次满足率: {m.get('frequency_satisfaction_rate', 0):.2%} "
                        f"({m.get('frequency_satisfied', 0)}/{m.get('frequency_total', 0)})\n")
                f.write(f"  - 数传任务: {m.get('downlink_tasks', 0)} 成功, {m.get('downlink_failed', 0)} 失败\n")
                f.write(f"  - 求解用时: {m.get('total_computation_time', 0):.2f} 秒\n")
                f.write(f"  - 迭代次数: {m.get('iterations', 0)}\n")

        if failed:
            f.write("\n\n【运行失败的算法】\n\n")
            for r in failed:
                f.write(f"\n{r['algorithm']}: {r.get('error', 'Unknown error')}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write(f"JSON报告: {report_file}\n")
        f.write(f"文本摘要: {summary_file}\n")
        f.write("="*80 + "\n")

    print(f"\n{'='*70}")
    print("测试报告已生成:")
    print(f"  JSON格式: {report_file}")
    print(f"  文本格式: {summary_file}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='大规模场景下所有规划算法完整性能测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有算法
  python scripts/benchmark_all_algorithms_complete.py

  # 只测试特定算法
  python scripts/benchmark_all_algorithms_complete.py --algorithms greedy edd ga

  # 禁用数传规划
  python scripts/benchmark_all_algorithms_complete.py --no-downlink
        """
    )

    parser.add_argument(
        '--scenario', '-s',
        default='scenarios/large_scale_frequency.json',
        help='场景配置文件路径 (默认: scenarios/large_scale_frequency.json)'
    )
    parser.add_argument(
        '--cache', '-c',
        default='java/output/frequency_scenario/visibility_windows_with_gs.json',
        help='可见性窗口缓存文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/benchmark_complete',
        help='输出目录 (默认: results/benchmark_complete)'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['greedy', 'edd', 'ga', 'sa', 'aco', 'pso', 'tabu'],
        choices=['greedy', 'edd', 'ga', 'sa', 'aco', 'pso', 'tabu', 'all'],
        help='要测试的算法列表 (默认: 所有算法)'
    )
    parser.add_argument(
        '--no-downlink',
        action='store_true',
        help='禁用数传规划'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    args = parser.parse_args()

    # 处理 'all' 选项
    algorithms = args.algorithms
    if 'all' in algorithms:
        algorithms = ['greedy', 'edd', 'ga', 'sa', 'aco', 'pso', 'tabu']

    # 运行测试
    results = run_benchmark(
        scenario_path=args.scenario,
        cache_path=args.cache,
        output_dir=args.output,
        algorithms=algorithms,
        enable_downlink=not args.no_downlink,
        seed=args.seed
    )

    # 打印最终结果摘要
    print(f"\n{'='*70}")
    print("测试完成摘要")
    print(f"{'='*70}")

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")

    if successful:
        print(f"\n算法性能排名（按需求满足率）:")
        sorted_results = sorted(
            successful,
            key=lambda x: x['metrics'].get('demand_satisfaction_rate', 0),
            reverse=True
        )
        for i, r in enumerate(sorted_results, 1):
            m = r['metrics']
            print(f"  {i}. {r['algorithm']}: "
                  f"满足率={m.get('demand_satisfaction_rate', 0):.2%}, "
                  f"用时={m.get('total_computation_time', 0):.2f}s")


if __name__ == '__main__':
    main()
