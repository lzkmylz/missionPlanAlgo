#!/usr/bin/env python3
"""
GA收敛特性分析脚本

用于分析GA算法在不同代数下的收敛特性，评估增加代数的收益，
确定最优的早停参数。
"""

import sys
import os
import time
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler.metaheuristic.ga_scheduler import GAScheduler
from core.models import Mission
from scripts.utils import load_window_cache_from_json


def run_ga_with_convergence_tracking(
    mission: Mission,
    window_cache,
    population_size: int = 50,
    max_generations: int = 100,
    output_dir: str = "results/convergence_analysis"
) -> Dict[str, Any]:
    """运行GA并详细记录收敛曲线

    Args:
        mission: 任务场景
        window_cache: 窗口缓存
        population_size: 种群大小
        max_generations: 最大迭代次数
        output_dir: 输出目录

    Returns:
        包含收敛曲线的详细结果
    """
    print(f"\n{'='*80}")
    print(f"GA收敛分析 - 种群大小: {population_size}, 最大代数: {max_generations}")
    print(f"{'='*80}")

    # 创建GA调度器（启用性能分析）
    config = {
        'population_size': population_size,
        'generations': max_generations,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'elitism': 5,
        'enable_profiling': True,
        'precompute_positions': True,
        'precompute_step_seconds': 1.0,
        # 禁用早停，记录完整收敛曲线
        'convergence_threshold': 0.0,  # 永不收敛
        'max_no_improvement': max_generations + 1,
    }

    ga_scheduler = GAScheduler(config)
    ga_scheduler.initialize(mission)
    ga_scheduler.set_window_cache(window_cache)

    # 手动运行GA以收集详细数据
    from scheduler.metaheuristic.base_metaheuristic import (
        EvaluationState, Solution
    )

    start_time = time.perf_counter()

    # 准备数据
    tasks = ga_scheduler._create_frequency_aware_tasks()
    satellites = list(mission.satellites)
    task_count = len(tasks)
    sat_count = len(satellites)

    print(f"  任务数: {task_count}")
    print(f"  卫星数: {sat_count}")

    # 准备数据（手动设置，因为在schedule()外面运行）
    ga_scheduler.tasks = ga_scheduler._create_frequency_aware_tasks()
    ga_scheduler.satellites = list(mission.satellites)
    ga_scheduler.task_count = len(ga_scheduler.tasks)
    ga_scheduler.sat_count = len(ga_scheduler.satellites)
    print(f"  手动设置: task_count={ga_scheduler.task_count}, sat_count={ga_scheduler.sat_count}")

    # 初始化组件
    ga_scheduler._initialize_components()

    # 初始化种群
    population_start = time.perf_counter()
    population = ga_scheduler.initialize_population()
    population_init_time = time.perf_counter() - population_start
    print(f"  种群初始化完成: {len(population)}个解, encoding长度={len(population[0].encoding) if population else 0}")

    # 评估初始种群
    eval_start = time.perf_counter()
    convergence_data = {
        'generations': [],
        'best_fitness': [],
        'avg_fitness': [],
        'worst_fitness': [],
        'diversity': [],  # 种群多样性
        'improvement': [],  # 相比上代的改进
        'eval_time': [],  # 每代评估时间
    }

    # 记录初始评估
    initial_fitnesses = []
    print(f"\n  调试: 评估前10个解...")
    for i, solution in enumerate(population[:10]):
        print(f"    解{i}: encoding长度={len(solution.encoding)}, 前5个基因={solution.encoding[:5]}")
        fitness = ga_scheduler._evaluate(solution)
        print(f"    解{i}: fitness={fitness}")
        solution.fitness = fitness
        initial_fitnesses.append(solution.fitness)
    # 评估剩余的解
    for solution in population[10:]:
        solution.fitness = ga_scheduler._evaluate(solution)
        initial_fitnesses.append(solution.fitness)

    best_fitness = max(initial_fitnesses)
    avg_fitness = sum(initial_fitnesses) / len(initial_fitnesses)
    worst_fitness = min(initial_fitnesses)

    # 计算多样性（适应度标准差）
    diversity = math.sqrt(sum((f - avg_fitness)**2 for f in initial_fitnesses) / len(initial_fitnesses))

    convergence_data['generations'].append(0)
    convergence_data['best_fitness'].append(best_fitness)
    convergence_data['avg_fitness'].append(avg_fitness)
    convergence_data['worst_fitness'].append(worst_fitness)
    convergence_data['diversity'].append(diversity)
    convergence_data['improvement'].append(0.0)
    convergence_data['eval_time'].append(time.perf_counter() - eval_start)

    initial_eval_time = time.perf_counter() - eval_start

    print(f"\n  初始种群评估完成:")
    print(f"    最优适应度: {best_fitness:.2f}")
    print(f"    平均适应度: {avg_fitness:.2f}")
    print(f"    最差适应度: {worst_fitness:.2f}")
    print(f"    种群多样性: {diversity:.2f}")
    print(f"    评估时间: {initial_eval_time:.2f}s")

    # 主优化循环
    print(f"\n  开始优化循环...")

    for generation in range(max_generations):
        gen_start = time.perf_counter()

        # 进化
        population = ga_scheduler.evolve(population)

        # 评估新解
        new_fitnesses = []
        for solution in population:
            if solution.fitness == 0.0:
                solution.fitness = ga_scheduler._evaluate(solution)
            new_fitnesses.append(solution.fitness)

        gen_best = max(new_fitnesses)
        gen_avg = sum(new_fitnesses) / len(new_fitnesses)
        gen_worst = min(new_fitnesses)
        gen_diversity = math.sqrt(sum((f - gen_avg)**2 for f in new_fitnesses) / len(new_fitnesses))

        # 计算改进
        improvement = gen_best - best_fitness
        prev_best = best_fitness
        best_fitness = max(best_fitness, gen_best)

        gen_time = time.perf_counter() - gen_start

        # 记录数据
        convergence_data['generations'].append(generation + 1)
        convergence_data['best_fitness'].append(best_fitness)
        convergence_data['avg_fitness'].append(gen_avg)
        convergence_data['worst_fitness'].append(gen_worst)
        convergence_data['diversity'].append(gen_diversity)
        convergence_data['improvement'].append(improvement)
        convergence_data['eval_time'].append(gen_time)

        # 每10代输出一次
        if (generation + 1) % 10 == 0:
            print(f"    Gen {generation + 1:3d}: best={best_fitness:.2f}, "
                  f"avg={gen_avg:.2f}, improve={improvement:.2f}, "
                  f"diversity={gen_diversity:.2f}, time={gen_time:.2f}s")

    total_time = time.perf_counter() - start_time

    # 分析收敛特性
    analysis = analyze_convergence(convergence_data)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'population_size': population_size,
            'max_generations': max_generations,
            'task_count': task_count,
            'satellite_count': sat_count,
        },
        'convergence_data': convergence_data,
        'analysis': analysis,
        'total_time': total_time,
    }

    output_file = os.path.join(
        output_dir,
        f"ga_convergence_{population_size}pop_{max_generations}gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  结果已保存: {output_file}")
    print(f"  总耗时: {total_time:.2f}s")

    return result


def analyze_convergence(data: Dict[str, List]) -> Dict[str, Any]:
    """分析收敛特性

    Args:
        data: 收敛数据字典

    Returns:
        分析结果
    """
    best_fitness = data['best_fitness']
    improvements = data['improvement']
    generations = data['generations']

    analysis = {
        'total_generations': len(generations) - 1,  # 不包括初始代
        'final_best_fitness': best_fitness[-1],
        'final_improvement_rate': 0.0,
        'suggested_generations': {},
        'convergence_metrics': {},
    }

    # 计算累计改进
    cumulative_improvement = sum(improvements)
    analysis['cumulative_improvement'] = cumulative_improvement

    # 计算各阶段的改进比例
    total_improvement = best_fitness[-1] - best_fitness[0]
    if total_improvement > 0:
        for stage in [10, 20, 30, 50, 70, 100]:
            if stage < len(best_fitness):
                stage_improvement = best_fitness[stage] - best_fitness[0]
                analysis[f'improvement_at_gen_{stage}'] = {
                    'value': stage_improvement,
                    'percentage': (stage_improvement / total_improvement) * 100,
                }

    # 计算收益递减点
    # 方法1: 连续多代改进小于阈值
    improvement_threshold = total_improvement * 0.01  # 1%的改进阈值
    no_improvement_count = 0
    stagnation_point = None

    for i, imp in enumerate(improvements[1:], 1):  # 从第1代开始
        if imp < improvement_threshold:
            no_improvement_count += 1
            if no_improvement_count >= 10 and stagnation_point is None:
                stagnation_point = i - 10  # 连续10代无显著改进
        else:
            no_improvement_count = 0

    analysis['stagnation_point'] = stagnation_point

    # 方法2: 边际收益分析
    # 计算每10代的边际改进率
    marginal_returns = []
    for i in range(10, len(best_fitness), 10):
        prev_fitness = best_fitness[i-10]
        curr_fitness = best_fitness[i]
        if prev_fitness > 0:
            marginal_return = (curr_fitness - prev_fitness) / prev_fitness * 100
            marginal_returns.append({
                'generation': i,
                'marginal_return_percent': marginal_return,
            })

    analysis['marginal_returns'] = marginal_returns

    # 找到边际收益低于5%的代数
    low_return_threshold = 5.0  # 5%
    for mr in marginal_returns:
        if mr['marginal_return_percent'] < low_return_threshold:
            analysis['suggested_generations']['low_return_threshold'] = mr['generation']
            break

    # 计算不同代数下的"性价比" (改进/时间)
    time_efficiency = []
    eval_times = data['eval_time']
    cumulative_time = 0

    for i in range(len(generations)):
        cumulative_time += eval_times[i]
        if cumulative_time > 0:
            current_improvement = best_fitness[i] - best_fitness[0]
            efficiency = current_improvement / cumulative_time
            time_efficiency.append({
                'generation': generations[i],
                'efficiency': efficiency,
                'cumulative_time': cumulative_time,
            })

    analysis['time_efficiency'] = time_efficiency

    # 找到效率开始显著下降的点
    if len(time_efficiency) > 20:
        peak_efficiency = max(te['efficiency'] for te in time_efficiency)
        for te in time_efficiency[20:]:  # 从20代后开始检查
            if te['efficiency'] < peak_efficiency * 0.3:  # 效率下降到峰值的30%
                analysis['suggested_generations']['efficiency_drop'] = te['generation']
                break

    # 综合建议
    suggestions = []
    if stagnation_point:
        suggestions.append(f"收敛停滞点: 约{stagnation_point}代")
    if 'low_return_threshold' in analysis['suggested_generations']:
        g = analysis['suggested_generations']['low_return_threshold']
        suggestions.append(f"边际收益<5%: 约{g}代")
    if 'efficiency_drop' in analysis['suggested_generations']:
        g = analysis['suggested_generations']['efficiency_drop']
        suggestions.append(f"效率显著下降: 约{g}代")

    analysis['recommendations'] = suggestions

    # 最终建议代数
    if suggestions:
        # 取最保守的建议
        recommended_gen = min(
            analysis['suggested_generations'].values()
        )
        analysis['recommended_generations'] = recommended_gen
        analysis['recommended_reason'] = f"基于{suggestions[0]}的保守估计"

    return analysis


def print_convergence_report(result: Dict[str, Any]):
    """打印收敛分析报告

    Args:
        result: 分析结果字典
    """
    print(f"\n{'='*80}")
    print("GA收敛分析报告")
    print(f"{'='*80}")

    params = result['parameters']
    print(f"\n测试参数:")
    print(f"  种群大小: {params['population_size']}")
    print(f"  最大代数: {params['max_generations']}")
    print(f"  任务数: {params['task_count']}")
    print(f"  卫星数: {params['satellite_count']}")

    data = result['convergence_data']
    analysis = result['analysis']

    print(f"\n收敛统计:")
    print(f"  实际运行代数: {analysis['total_generations']}")
    print(f"  初始最优适应度: {data['best_fitness'][0]:.2f}")
    print(f"  最终最优适应度: {analysis['final_best_fitness']:.2f}")
    print(f"  累计改进: {analysis['cumulative_improvement']:.2f}")

    print(f"\n各阶段改进比例:")
    for stage in [10, 20, 30, 50, 70, 100]:
        key = f'improvement_at_gen_{stage}'
        if key in analysis:
            info = analysis[key]
            print(f"  前{stage}代: {info['percentage']:.1f}% ({info['value']:.2f})")

    print(f"\n边际收益分析 (每10代):")
    for mr in analysis.get('marginal_returns', [])[:10]:
        gen = mr['generation']
        ret = mr['marginal_return_percent']
        marker = " ***" if ret < 5.0 else ""
        print(f"  Gen {gen:3d}: {ret:6.2f}%{marker}")

    print(f"\n建议:")
    for rec in analysis.get('recommendations', []):
        print(f"  - {rec}")

    if 'recommended_generations' in analysis:
        print(f"\n{'='*80}")
        print(f"推荐参数: generations = {analysis['recommended_generations']}")
        print(f"原因: {analysis['recommended_reason']}")
        print(f"{'='*80}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="GA收敛特性分析")
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="种群大小 (默认: 50)"
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=100,
        help="最大代数 (默认: 100)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/convergence_analysis",
        help="输出目录"
    )

    args = parser.parse_args()

    # 加载场景
    scenario_file = "scenarios/large_scale_frequency.json"
    window_cache_file = "java/output/frequency_scenario/visibility_windows_with_gs.json"

    print(f"加载场景: {scenario_file}")
    mission = Mission.load(scenario_file)
    print(f"加载窗口缓存: {window_cache_file}")
    window_cache = load_window_cache_from_json(window_cache_file, mission)

    # 运行收敛分析
    result = run_ga_with_convergence_tracking(
        mission=mission,
        window_cache=window_cache,
        population_size=args.population_size,
        max_generations=args.max_generations,
        output_dir=args.output_dir
    )

    # 打印报告
    print_convergence_report(result)


if __name__ == "__main__":
    main()
