#!/usr/bin/env python3
"""
GA性能基准测试脚本

用于分析GA算法各阶段耗时，识别性能瓶颈
支持不同规模测试：Tiny, Small, Medium, Large
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scheduler.metaheuristic.ga_scheduler import GAScheduler
from core.models import Mission
from scripts.utils import load_window_cache_from_json


def load_scenario_for_scale(scale: str) -> tuple:
    """加载对应规模的场景

    Args:
        scale: 规模名称 (tiny, small, medium, large, full)

    Returns:
        (mission, window_cache) 元组
    """
    # 只有一个大规模场景可用
    scenario_file = "scenarios/large_scale_frequency.json"
    window_cache_file = "java/output/frequency_scenario/visibility_windows_with_gs.json"

    if not os.path.exists(scenario_file):
        raise FileNotFoundError(f"场景文件不存在: {scenario_file}")

    print(f"  加载场景: {scenario_file}")
    mission = Mission.load(scenario_file)

    print(f"  卫星数: {len(mission.satellites)}")
    print(f"  目标数: {len(mission.targets)}")

    # 加载窗口缓存
    print(f"  加载窗口缓存: {window_cache_file}")
    window_cache = load_window_cache_from_json(window_cache_file, mission)
    print(f"  窗口条目数: {len(window_cache._windows)}")

    return mission, window_cache


def run_ga_benchmark(
    scale: str,
    population_size: int = 50,
    generations: int = 50,
    output_dir: str = "results/benchmark_ga"
) -> dict:
    """运行GA性能基准测试

    Args:
        scale: 测试规模
        population_size: 种群大小
        generations: 迭代次数
        output_dir: 输出目录

    Returns:
        性能测试结果字典
    """
    print(f"\n{'='*80}")
    print(f"GA性能测试 - 规模: {scale}")
    print(f"{'='*80}")
    print(f"  种群大小: {population_size}")
    print(f"  迭代次数: {generations}")

    # 加载场景
    mission, window_cache = load_scenario_for_scale(scale)

    # 创建GA调度器（启用性能分析）
    config = {
        'population_size': population_size,
        'generations': generations,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'elitism': 5,
        'enable_profiling': True,
        'precompute_positions': True,
        'precompute_step_seconds': 1.0,
    }

    ga_scheduler = GAScheduler(config)
    ga_scheduler.initialize(mission)
    ga_scheduler.set_window_cache(window_cache)

    # 运行调度
    print(f"\n  开始调度...")
    start_time = time.perf_counter()
    result = ga_scheduler.schedule()
    total_time = time.perf_counter() - start_time

    # 收集性能数据
    performance_data = {
        "scale": scale,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "elitism": 5,
        },
        "scenario": {
            "satellite_count": len(mission.satellites),
            "target_count": len(mission.targets),
        },
        "results": {
            "scheduled_count": len(result.scheduled_tasks),
            "makespan_hours": result.makespan / 3600 if result.makespan else 0,
            "iterations": result.iterations,
            "total_time": total_time,
        }
    }

    # 如果性能分析器有数据，收集它
    if hasattr(ga_scheduler, '_mh_profiler') and ga_scheduler._mh_profiler:
        profiler_summary = ga_scheduler._mh_profiler.profiler.get_summary()
        evaluation_stats = ga_scheduler._mh_profiler.get_evaluation_summary()

        performance_data["profiler"] = profiler_summary
        performance_data["evaluation_stats"] = evaluation_stats

    # 打印结果
    print(f"\n{'='*80}")
    print("测试结果摘要")
    print(f"{'='*80}")
    print(f"  成功调度: {performance_data['results']['scheduled_count']} 个任务")
    print(f"  实际迭代: {performance_data['results']['iterations']} 次")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  makespan: {performance_data['results']['makespan_hours']:.2f} 小时")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ga_perf_{scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
    print(f"\n  结果已保存: {output_file}")

    return performance_data


def run_scaling_analysis(output_dir: str = "results/benchmark_ga"):
    """运行规模扩展分析

    测试不同规模下的性能表现，分析增长趋势
    """
    print(f"\n{'='*80}")
    print("GA性能 - 规模扩展分析")
    print(f"{'='*80}")

    # 定义测试配置
    test_configs = [
        {"scale": "tiny", "population_size": 30, "generations": 30},
        {"scale": "small", "population_size": 50, "generations": 50},
    ]

    # 检查中等和大型场景文件是否存在
    if os.path.exists("scenarios/medium_scale.json"):
        test_configs.append({"scale": "medium", "population_size": 50, "generations": 50})
    else:
        print("  跳过 medium - 场景文件不存在")

    if os.path.exists("scenarios/large_scale_frequency.json"):
        test_configs.append({"scale": "large", "population_size": 50, "generations": 20})
    else:
        print("  跳过 large - 场景文件不存在")

    results = []
    for config in test_configs:
        try:
            result = run_ga_benchmark(
                scale=config["scale"],
                population_size=config["population_size"],
                generations=config["generations"],
                output_dir=output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 分析增长趋势
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("增长趋势分析")
        print(f"{'='*80}")

        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]

            time_ratio = curr["results"]["total_time"] / prev["results"]["total_time"]
            task_ratio = curr["results"]["scheduled_count"] / max(prev["results"]["scheduled_count"], 1)

            print(f"\n  {prev['scale']} -> {curr['scale']}:")
            print(f"    时间增长比: {time_ratio:.2f}x")
            print(f"    任务增长比: {task_ratio:.2f}x")
            print(f"    效率: {'线性' if time_ratio <= task_ratio * 2 else '超线性'}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="GA性能基准测试")
    parser.add_argument(
        "--scale",
        choices=["tiny", "small", "medium", "large", "all"],
        default="tiny",
        help="测试规模"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="种群大小"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="迭代次数"
    )
    parser.add_argument(
        "--output-dir",
        default="results/benchmark_ga",
        help="输出目录"
    )

    args = parser.parse_args()

    if args.scale == "all":
        run_scaling_analysis(args.output_dir)
    else:
        run_ga_benchmark(
            scale=args.scale,
            population_size=args.population_size,
            generations=args.generations,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
