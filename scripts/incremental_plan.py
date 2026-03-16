#!/usr/bin/env python3
"""
增量任务规划命令行工具

Usage:
    python scripts/incremental_plan.py \
        --existing results/schedule.json \
        --new-targets new_targets.json \
        --strategy conservative \
        --output results/incremental_result.json

策略说明：
    - conservative: 仅使用剩余资源，不抢占已有任务
    - aggressive: 允许抢占已有任务资源
    - hybrid: 动态选择策略，高优先级目标使用激进策略
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler.incremental import (
    IncrementalPlanner,
    IncrementalPlanRequest,
    IncrementalStrategyType,
    PriorityRule,
    PreemptionRule
)
from scheduler.base_scheduler import ScheduleResult, ScheduledTask
from core.models import Mission, Target
from utils.serialization import ScenarioLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_schedule_result(filepath: str) -> ScheduleResult:
    """加载调度结果文件"""
    logger.info(f"Loading existing schedule from {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # 解析任务列表
    tasks = []
    for task_data in data.get('scheduled_tasks', []):
        task = ScheduledTask(
            task_id=task_data['task_id'],
            satellite_id=task_data['satellite_id'],
            target_id=task_data['target_id'],
            imaging_start=datetime.fromisoformat(task_data['imaging_start']),
            imaging_end=datetime.fromisoformat(task_data['imaging_end']),
            imaging_mode=task_data.get('imaging_mode', 'standard'),
            slew_angle=task_data.get('slew_angle', 0.0),
            slew_time=task_data.get('slew_time', 0.0),
            storage_before=task_data.get('storage_before', 0.0),
            storage_after=task_data.get('storage_after', 0.0),
            power_before=task_data.get('power_before', 0.0),
            power_after=task_data.get('power_after', 0.0),
            power_consumed=task_data.get('power_consumed_wh', 0.0),
            priority=task_data.get('priority')
        )
        tasks.append(task)

    return ScheduleResult(
        scheduled_tasks=tasks,
        unscheduled_tasks=data.get('unscheduled_tasks', {}),
        makespan=data.get('makespan', 0.0),
        computation_time=data.get('computation_time', 0.0),
        iterations=data.get('iterations', 0)
    )


def load_new_targets(filepath: str) -> List[Target]:
    """加载新增目标文件"""
    logger.info(f"Loading new targets from {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    targets = []
    for target_data in data.get('targets', []):
        target = Target(
            id=target_data['id'],
            latitude=target_data['latitude'],
            longitude=target_data['longitude'],
            priority=target_data.get('priority', 0),
            imaging_mode=target_data.get('imaging_mode', 'standard')
        )
        targets.append(target)

    logger.info(f"Loaded {len(targets)} new targets")
    return targets


def load_mission(filepath: str) -> Mission:
    """加载任务场景"""
    logger.info(f"Loading mission from {filepath}")

    loader = ScenarioLoader()
    return loader.load_mission(filepath)


def save_result(result: Any, filepath: str) -> None:
    """保存规划结果"""
    logger.info(f"Saving result to {filepath}")

    output_dir = Path(filepath).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = result.to_dict() if hasattr(result, 'to_dict') else result

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Result saved successfully")


def analyze_resources(existing_schedule: ScheduleResult, mission: Mission) -> Dict[str, Any]:
    """分析资源使用情况"""
    planner = IncrementalPlanner()
    return planner.analyze_resources(existing_schedule, mission)


def main():
    parser = argparse.ArgumentParser(
        description='增量任务规划工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 保守策略增量规划
  python scripts/incremental_plan.py -e schedule.json -n targets.json -s conservative

  # 激进策略，允许抢占20%的任务
  python scripts/incremental_plan.py -e schedule.json -n targets.json -s aggressive --max-preemption 0.2

  # 混合策略
  python scripts/incremental_plan.py -e schedule.json -n targets.json -s hybrid

  # 分析资源使用情况
  python scripts/incremental_plan.py -e schedule.json -m mission.json --analyze-only
        """
    )

    # 输入参数
    parser.add_argument('-e', '--existing', required=True,
                       help='现有调度结果文件路径 (JSON)')
    parser.add_argument('-n', '--new-targets',
                       help='新增目标文件路径 (JSON)')
    parser.add_argument('-m', '--mission',
                       help='任务场景文件路径')

    # 策略参数
    parser.add_argument('-s', '--strategy', default='conservative',
                       choices=['conservative', 'aggressive', 'hybrid'],
                       help='增量规划策略 (默认: conservative)')
    parser.add_argument('--max-preemption', type=float, default=0.2,
                       help='最大抢占比例，0-1之间 (默认: 0.2)')
    parser.add_argument('--min-priority-diff', type=int, default=2,
                       help='最小优先级差才允许抢占 (默认: 2)')
    parser.add_argument('--high-priority-threshold', type=int, default=8,
                       help='高优先级阈值，混合策略使用 (默认: 8)')

    # 输出参数
    parser.add_argument('-o', '--output',
                       help='输出文件路径 (默认: results/incremental_result_<timestamp>.json)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析资源使用情况，不执行规划')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='启用详细日志')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载现有调度结果
    try:
        existing_schedule = load_schedule_result(args.existing)
    except Exception as e:
        logger.error(f"Failed to load existing schedule: {e}")
        sys.exit(1)

    # 加载任务场景
    mission = None
    if args.mission:
        try:
            mission = load_mission(args.mission)
        except Exception as e:
            logger.warning(f"Failed to load mission: {e}")

    # 仅分析资源使用情况
    if args.analyze_only:
        if not mission:
            logger.error("Mission file required for resource analysis")
            sys.exit(1)

        report = analyze_resources(existing_schedule, mission)
        print("\n=== 资源使用分析报告 ===")
        print(f"卫星总数: {report['summary']['total_satellites']}")
        print(f"总可用时间: {report['summary']['total_available_time_hours']:.2f} 小时")
        print(f"可用窗口数: {report['summary']['total_windows']}")
        print(f"平均利用率: {report['summary']['average_utilization_rate']:.2%}")
        print("\n各卫星详情:")
        for sat_id, detail in report['satellite_details'].items():
            print(f"  {sat_id}: {detail['available_time_hours']:.2f}h可用, "
                  f"利用率{detail['utilization_rate']:.2%}")
        return

    # 检查新增目标文件
    if not args.new_targets:
        logger.error("New targets file required (use -n or --new-targets)")
        sys.exit(1)

    # 加载新增目标
    try:
        new_targets = load_new_targets(args.new_targets)
    except Exception as e:
        logger.error(f"Failed to load new targets: {e}")
        sys.exit(1)

    # 确定策略类型
    strategy_map = {
        'conservative': IncrementalStrategyType.CONSERVATIVE,
        'aggressive': IncrementalStrategyType.AGGRESSIVE,
        'hybrid': IncrementalStrategyType.HYBRID
    }
    strategy = strategy_map[args.strategy]

    # 创建规划请求
    request = IncrementalPlanRequest(
        new_targets=new_targets,
        existing_schedule=existing_schedule,
        strategy=strategy,
        priority_rules=PriorityRule(),
        preemption_rules=PreemptionRule(
            max_preemption_ratio=args.max_preemption,
            min_priority_difference=args.min_priority_diff
        ) if strategy == IncrementalStrategyType.AGGRESSIVE else None,
        max_preemption_ratio=args.max_preemption,
        mission=mission
    )

    # 执行增量规划
    logger.info(f"Starting incremental planning with {args.strategy} strategy")
    planner = IncrementalPlanner()

    try:
        result = planner.plan(request)
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印结果摘要
    print("\n=== 增量规划结果 ===")
    print(f"策略: {result.strategy_used.value}")
    print(f"新增任务: {len(result.new_tasks)}")
    print(f"被抢占任务: {len(result.preempted_tasks)}")
    print(f"重调度任务: {len(result.rescheduled_tasks)}")
    print(f"失败目标: {len(result.failed_targets)}")
    print(f"成功率: {result.statistics.get('success_rate', 0):.2%}")

    # 保存结果
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"results/incremental_result_{args.strategy}_{timestamp}.json"

    save_result(result, args.output)
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
