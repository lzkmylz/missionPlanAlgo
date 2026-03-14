#!/usr/bin/env python3
"""
目标优先级验证测试脚本

验证所有调度器正确处理目标优先级约束：
- 优先级范围1-100，数字越小优先级越高
- 冲突时高优先级（小数字）任务优先被调度
- 相同优先级时任一被调度均可
"""

import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga import GAScheduler
from scheduler.metaheuristic.sa import SAScheduler
from scheduler.metaheuristic.aco import ACOScheduler
from scheduler.metaheuristic.pso import PSOScheduler
from scheduler.metaheuristic.tabu import TabuScheduler
from core.models import Mission, Target, GroundStation


def load_scenario(scenario_path: str) -> Dict[str, Any]:
    """加载场景配置文件"""
    with open(scenario_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_mission_from_scenario(scenario: Dict[str, Any], visibility_path: str) -> Mission:
    """从场景配置创建Mission对象"""
    from core.models import Satellite, SatelliteCapabilities, Orbit
    from core.models import ImagingMode, GroundStation

    # 创建卫星列表
    satellites = []
    for sat_config in scenario.get('satellites', []):
        orbit_config = sat_config.get('orbit', {})
        orbit = Orbit(
            semi_major_axis=orbit_config.get('semi_major_axis', 6878000),
            eccentricity=orbit_config.get('eccentricity', 0.001),
            inclination=orbit_config.get('inclination', 97.5),
            raan=orbit_config.get('raan', 0),
            arg_of_perigee=orbit_config.get('arg_of_perigee', 0),
            mean_anomaly=orbit_config.get('mean_anomaly', 0)
        )

        capabilities_config = sat_config.get('capabilities', {})
        capabilities = SatelliteCapabilities(
            swath_width=capabilities_config.get('swath_width', 10000),
            min_incidence_angle=capabilities_config.get('min_incidence_angle', 0),
            max_incidence_angle=capabilities_config.get('max_incidence_angle', 45),
            power_capacity=capabilities_config.get('power_capacity', 2800),
            storage_capacity=capabilities_config.get('storage_capacity', 500),
            data_rate=capabilities_config.get('data_rate', 300),
            imaging_modes=[ImagingMode.PUSH_BROOM]
        )

        satellite = Satellite(
            id=sat_config['id'],
            name=sat_config.get('name', sat_config['id']),
            orbit=orbit,
            capabilities=capabilities
        )
        satellites.append(satellite)

    # 创建目标列表
    targets = []
    for tgt_config in scenario.get('targets', []):
        target = Target(
            id=tgt_config['id'],
            name=tgt_config.get('name', tgt_config['id']),
            longitude=tgt_config['location'][0],
            latitude=tgt_config['location'][1],
            priority=tgt_config.get('priority', 50),
            required_observations=tgt_config.get('required_observations', 1)
        )
        targets.append(target)

    # 创建地面站列表
    ground_stations = []
    for gs_config in scenario.get('ground_stations', []):
        gs = GroundStation(
            id=gs_config['id'],
            name=gs_config.get('name', gs_config['id']),
            longitude=gs_config['location'][0],
            latitude=gs_config['location'][1],
            elevation_threshold=gs_config.get('elevation_threshold', 5.0)
        )
        ground_stations.append(gs)

    # 解析时间
    from datetime import datetime
    start_time = datetime.fromisoformat(scenario['start_time'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(scenario['end_time'].replace('Z', '+00:00'))

    # 创建Mission
    mission = Mission(
        satellites=satellites,
        targets=targets,
        ground_stations=ground_stations,
        start_time=start_time,
        end_time=end_time
    )

    return mission


def create_schedulers(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """创建所有调度器实例"""
    if config is None:
        config = {}

    return {
        'greedy': GreedyScheduler(config=config),
        'edd': EDDScheduler(config=config),
        'spt': SPTScheduler(config=config),
        'ga': GAScheduler(config=config),
        'sa': SAScheduler(config=config),
        'aco': ACOScheduler(config=config),
        'pso': PSOScheduler(config=config),
        'tabu': TabuScheduler(config=config),
    }


def get_target_priority_from_scenario(scenario: Dict, target_id: str) -> int:
    """从场景配置获取目标的优先级"""
    for target in scenario.get('targets', []):
        if target['id'] == target_id:
            return target.get('priority', 50)
    return 50


def analyze_priority_handling(
    result: Any,
    scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """分析调度结果的优先级处理情况"""
    analysis = {
        'total_scheduled': 0,
        'high_priority_scheduled': [],
        'low_priority_scheduled': [],
        'priority_violations': [],
        'priority_distribution': {},
        'conflict_resolution': {}
    }

    if not result or not hasattr(result, 'scheduled_tasks'):
        return analysis

    scheduled_tasks = result.scheduled_tasks
    analysis['total_scheduled'] = len(scheduled_tasks)

    # 按优先级分组统计
    for task in scheduled_tasks:
        target_id = getattr(task, 'target_id', None) or getattr(task, 'task_id', '')
        priority = get_target_priority_from_scenario(scenario, target_id)

        if priority not in analysis['priority_distribution']:
            analysis['priority_distribution'][priority] = []
        analysis['priority_distribution'][priority].append(target_id)

        if priority <= 5:
            analysis['high_priority_scheduled'].append({
                'target_id': target_id,
                'priority': priority
            })
        elif priority >= 90:
            analysis['low_priority_scheduled'].append({
                'target_id': target_id,
                'priority': priority
            })

    # 检查特定冲突对
    conflict_pairs = [
        ('TGT-HIGH-001', 'TGT-LOW-001', 'TGT-HIGH-001应该优先于TGT-LOW-001'),
        ('TGT-HIGH-002', 'TGT-LOW-002', 'TGT-HIGH-002应该优先于TGT-LOW-002'),
        ('TGT-EQ-001', 'TGT-EQ-002', 'TGT-EQ-001和TGT-EQ-002优先级相同，任一均可'),
        ('TGT-CRIT-001', 'TGT-CRIT-002', 'TGT-CRIT-001(优先级1)应该优先于TGT-CRIT-002(优先级5)'),
    ]

    scheduled_target_ids = set()
    for task in scheduled_tasks:
        target_id = getattr(task, 'target_id', None) or getattr(task, 'task_id', '')
        scheduled_target_ids.add(target_id)

    for high_id, low_id, description in conflict_pairs:
        high_scheduled = high_id in scheduled_target_ids
        low_scheduled = low_id in scheduled_target_ids

        analysis['conflict_resolution'][f"{high_id}_vs_{low_id}"] = {
            'description': description,
            'high_priority_scheduled': high_scheduled,
            'low_priority_scheduled': low_scheduled,
            'correct': high_scheduled or (not high_scheduled and not low_scheduled)
        }

    return analysis


def print_validation_report(algorithm: str, analysis: Dict[str, Any]) -> bool:
    """打印验证报告，返回是否通过"""
    print(f"\n{'='*60}")
    print(f"调度器: {algorithm.upper()}")
    print(f"{'='*60}")

    print(f"总调度任务数: {analysis['total_scheduled']}")

    print("\n优先级分布:")
    for priority in sorted(analysis['priority_distribution'].keys()):
        targets = analysis['priority_distribution'][priority]
        print(f"  优先级 {priority}: {len(targets)} 个目标 {targets}")

    print("\n高优先级(<=5)已调度:")
    if analysis['high_priority_scheduled']:
        for item in analysis['high_priority_scheduled']:
            print(f"  ✓ {item['target_id']} (优先级: {item['priority']})")
    else:
        print("  ✗ 无")

    print("\n低优先级(>=90)已调度:")
    if analysis['low_priority_scheduled']:
        for item in analysis['low_priority_scheduled']:
            print(f"  ! {item['target_id']} (优先级: {item['priority']}) [警告: 可能抢占高优先级任务资源]")
    else:
        print("  ✓ 无（正确，低优先级任务应让位于高优先级任务）")

    print("\n冲突解决验证:")
    all_correct = True
    for pair_name, resolution in analysis['conflict_resolution'].items():
        status = "✓ PASS" if resolution['correct'] else "✗ FAIL"
        print(f"  {status} {resolution['description']}")
        print(f"        高优先级调度: {resolution['high_priority_scheduled']}, 低优先级调度: {resolution['low_priority_scheduled']}")
        if not resolution['correct']:
            all_correct = False

    return all_correct


def run_priority_validation(
    scenario_path: str,
    visibility_path: str,
    algorithms: List[str],
    output_dir: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """运行优先级验证测试"""
    print("="*60)
    print("目标优先级验证测试")
    print("="*60)
    print(f"场景文件: {scenario_path}")
    print(f"可见性缓存: {visibility_path}")
    print(f"测试算法: {', '.join(algorithms)}")
    print("="*60)

    # 加载场景
    scenario = load_scenario(scenario_path)
    print(f"\n场景: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print(f"卫星数: {len(scenario.get('satellites', []))}")
    print(f"目标数: {len(scenario.get('targets', []))}")

    # 显示目标优先级信息
    print("\n目标优先级配置:")
    for target in scenario.get('targets', []):
        print(f"  {target['id']}: 优先级={target['priority']}, 位置={target['location']}")

    # 创建Mission
    mission = create_mission_from_scenario(scenario, visibility_path)

    # 创建调度器
    schedulers = create_schedulers(config={
        'visibility_cache_path': visibility_path,
        'enable_attitude_calculation': False,  # 简化测试
    })

    results = {}
    all_passed = True

    for algorithm in algorithms:
        if algorithm not in schedulers:
            print(f"\n警告: 未知算法 '{algorithm}'，跳过")
            continue

        scheduler = schedulers[algorithm]

        print(f"\n{'-'*60}")
        print(f"运行 {algorithm.upper()} 调度器...")
        print(f"{'-'*60}")

        try:
            # 初始化调度器
            scheduler.initialize(mission, visibility_path)

            # 运行调度
            result = scheduler.schedule()

            # 分析结果
            analysis = analyze_priority_handling(result, scenario)
            results[algorithm] = analysis

            # 打印报告
            passed = print_validation_report(algorithm, analysis)
            all_passed = all_passed and passed

            if verbose and hasattr(result, 'computation_time'):
                print(f"\n计算时间: {result.computation_time:.2f}秒")

        except Exception as e:
            print(f"\n✗ {algorithm.upper()} 调度失败: {e}")
            import traceback
            traceback.print_exc()
            results[algorithm] = {'error': str(e)}
            all_passed = False

    # 生成汇总报告
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)

    for algorithm, analysis in results.items():
        if 'error' in analysis:
            print(f"  ✗ {algorithm.upper()}: 失败 - {analysis['error']}")
        else:
            passed = all(
                r['correct'] for r in analysis.get('conflict_resolution', {}).values()
            )
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} {algorithm.upper()}: "
                  f"调度{analysis['total_scheduled']}个任务, "
                  f"高优先级{len(analysis['high_priority_scheduled'])}个")

    print("="*60)
    final_status = "✓ 所有调度器通过优先级验证" if all_passed else "✗ 部分调度器未通过验证"
    print(final_status)
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description='目标优先级验证测试')
    parser.add_argument('--scenario', '-s',
                        default='scenarios/priority_validation_scenario.json',
                        help='场景配置文件路径')
    parser.add_argument('--visibility', '-v',
                        default='java/output/frequency_scenario/visibility_windows.json',
                        help='可见性窗口缓存文件路径')
    parser.add_argument('--algorithms', '-a',
                        default='greedy,edd,spt,ga,sa,aco,pso,tabu',
                        help='要测试的算法列表，逗号分隔')
    parser.add_argument('--output', '-o',
                        default='results/priority_validation',
                        help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')

    args = parser.parse_args()

    # 解析算法列表
    algorithms = [a.strip() for a in args.algorithms.split(',')]

    # 运行验证
    results = run_priority_validation(
        scenario_path=args.scenario,
        visibility_path=args.visibility,
        algorithms=algorithms,
        output_dir=args.output,
        verbose=args.verbose
    )

    # 保存结果
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_path / f'priority_validation_{timestamp}.json'

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
