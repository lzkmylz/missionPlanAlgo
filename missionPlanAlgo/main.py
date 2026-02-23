#!/usr/bin/env python3
"""
卫星星座任务规划算法研究平台主入口

Usage:
    python main.py --scenario scenarios/point_group_scenario.yaml --algorithm GA
    python main.py --help
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.models import Mission
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from evaluation.metrics import MetricsCalculator
from visualization.gantt_chart import GanttChart


def load_scenario(scenario_path: str) -> Mission:
    """加载场景配置"""
    return Mission.load(scenario_path)


def get_scheduler(algorithm_name: str, config: dict):
    """获取调度器实例"""
    if algorithm_name.lower() == 'greedy':
        return GreedyScheduler(config)
    elif algorithm_name.lower() == 'ga':
        return GAScheduler(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def run_experiment(args):
    """运行实验"""
    print(f"{'='*60}")
    print(f"卫星星座任务规划算法研究平台")
    print(f"{'='*60}\n")

    # 1. 加载场景
    print(f"[1/4] 加载场景配置: {args.scenario}")
    mission = load_scenario(args.scenario)
    summary = mission.summary()
    print(f"  - 场景名称: {summary['name']}")
    print(f"  - 规划周期: {summary['duration_hours']:.1f} 小时")
    print(f"  - 卫星数量: {summary['satellite_count']} 颗")
    print(f"  - 目标数量: {summary['target_count']} 个")
    print(f"  - 地面站数量: {summary['ground_station_count']} 个\n")

    # 2. 预计算可见性窗口
    print(f"[2/4] 预计算可见性窗口")
    print(f"  正在计算 {len(mission.satellites)} 颗卫星 x {len(mission.targets)} 个目标的可见窗口...")
    print(f"  (跳过实际计算以加速演示)\n")

    # 3. 运行调度算法
    print(f"[3/4] 运行调度算法: {args.algorithm}")
    scheduler_config = {}
    if args.config:
        import json
        scheduler_config = json.loads(args.config)

    scheduler = get_scheduler(args.algorithm, scheduler_config)
    scheduler.initialize(mission)

    print(f"  开始调度...")
    result = scheduler.schedule()
    print(f"  完成！\n")

    # 4. 计算性能指标
    print(f"[4/4] 计算性能指标")
    metrics_calc = MetricsCalculator(mission)
    metrics = metrics_calc.calculate_all(result)

    print(f"\n{'='*60}")
    print(f"调度结果")
    print(f"{'='*60}")
    print(f"  成功调度任务: {metrics.scheduled_task_count}")
    print(f"  未调度任务: {metrics.unscheduled_task_count}")
    print(f"  需求满足率: {metrics.demand_satisfaction_rate:.2%}")
    print(f"  总完成时间: {metrics.makespan/3600:.2f} 小时")
    print(f"  算法求解用时: {metrics.computation_time:.2f} 秒")
    print(f"  解质量: {metrics.solution_quality:.4f}")
    print(f"{'='*60}\n")

    # 5. 可视化（可选）
    if args.visualize and result.scheduled_tasks:
        print("[可选] 生成可视化...")
        gantt = GanttChart()
        fig = gantt.plot(
            result.scheduled_tasks[:50],  # 只显示前50个任务
            mission.satellites,
            mission.ground_stations,
            mission.start_time,
            mission.end_time,
            title=f"调度结果 - {args.algorithm}算法"
        )

        output_path = args.output or f"gantt_{args.algorithm}_{datetime.now():%Y%m%d_%H%M%S}.png"
        gantt.save(fig, output_path)
        print(f"  甘特图已保存: {output_path}\n")

    # 6. 保存结果（可选）
    if args.save_result:
        import json
        result_data = {
            'algorithm': args.algorithm,
            'scenario': mission.name,
            'metrics': metrics.to_dict(),
            'scheduled_count': len(result.scheduled_tasks),
            'unscheduled_count': len(result.unscheduled_tasks),
        }
        result_path = args.output or f"result_{args.algorithm}.json"
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"  结果已保存: {result_path}\n")

    print(f"实验完成！")
    return result, metrics


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='卫星星座任务规划算法研究平台',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用贪心算法运行点群场景
  python main.py --scenario scenarios/point_group_scenario.yaml --algorithm Greedy

  # 使用遗传算法并生成可视化
  python main.py --scenario scenarios/point_group_scenario.yaml --algorithm GA --visualize
        """
    )

    parser.add_argument(
        '--scenario', '-s',
        required=True,
        help='场景配置文件路径 (JSON格式)'
    )

    parser.add_argument(
        '--algorithm', '-a',
        required=True,
        choices=['Greedy', 'GA'],
        help='调度算法名称'
    )

    parser.add_argument(
        '--config', '-c',
        help='算法参数配置 (JSON格式字符串)'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='生成可视化图表'
    )

    parser.add_argument(
        '--output', '-o',
        help='输出文件路径'
    )

    parser.add_argument(
        '--save-result',
        action='store_true',
        help='保存调度结果到文件'
    )

    args = parser.parse_args()

    try:
        run_experiment(args)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
