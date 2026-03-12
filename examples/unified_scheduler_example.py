"""
统一调度器使用示例

演示如何使用 UnifiedScheduler 进行完整的任务规划：
1. 成像任务调度（考虑完整约束）
2. 观测频次需求处理
3. 地面站数传规划
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """基本用法示例 - 仅成像调度"""
    print("=" * 70)
    print("示例1: 基本用法 - 仅成像调度")
    print("=" * 70)

    from core.models import Mission
    from core.orbit.visibility.window_cache import VisibilityWindowCache
    from scheduler.unified_scheduler import UnifiedScheduler

    # 加载场景
    scenario_path = "scenarios/large_scale_frequency.json"
    cache_path = "java/output/frequency_scenario/visibility_windows.json"

    try:
        mission = Mission.load(scenario_path)
        print(f"场景加载成功: {len(mission.satellites)}颗卫星, {len(mission.targets)}个目标")

        # 加载窗口缓存（简化示例，实际需要解析JSON）
        # cache = load_window_cache_from_json(cache_path, mission)

        # 创建调度器（这里需要实际缓存数据）
        # scheduler = UnifiedScheduler(
        #     mission=mission,
        #     window_cache=cache,
        #     config={'imaging_algorithm': 'greedy'}
        # )
        # result = scheduler.schedule()

        print("实际运行需要加载窗口缓存数据")

    except Exception as e:
        print(f"示例需要实际场景文件: {e}")


def example_with_downlink():
    """完整用法示例 - 成像 + 数传（默认启用）"""
    print("\n" + "=" * 70)
    print("示例2: 完整用法 - 成像 + 数传规划（默认启用）")
    print("=" * 70)

    print("""
from scheduler.unified_scheduler import UnifiedScheduler

# 创建调度器（数传规划默认启用，如果提供了 ground_station_pool）
scheduler = UnifiedScheduler(
    mission=mission,
    window_cache=window_cache,
    ground_station_pool=ground_station_pool,  # 提供地面站资源池即自动启用数传
    config={
        'imaging_algorithm': 'ga',      # 使用遗传算法
        'imaging_config': {
            'population_size': 50,
            'generations': 100,
        },
        # 注意: 数传速率和固存容量从每颗卫星的 capabilities 配置中自动读取
        # 卫星配置示例:
        #   satellites: [{
        #     "id": "SAT-001",
        #     "capabilities": {
        #       "storage_capacity": 500,  # GB
        #       "data_rate": 300,         # Mbps
        #       ...
        #     }
        #   }]
    }
)

# 执行调度
result = scheduler.schedule()

# 访问结果
print(f"成像任务数: {len(result.imaging_result.scheduled_tasks)}")
print(f"数传任务数: {len(result.downlink_result.downlink_tasks)}")

# 查看频次满足度
for target_id, info in result.target_observations.items():
    print(f"{target_id}: {info['status']}, 满足={info['satisfied']}")
    """)


def example_different_algorithms():
    """不同算法对比示例"""
    print("\n" + "=" * 70)
    print("示例3: 不同算法对比")
    print("=" * 70)

    algorithms = ['greedy', 'ga', 'edd']

    for algo in algorithms:
        print(f"\n{algo.upper()} 算法:")
        print(f"  config = {{'imaging_algorithm': '{algo}'}}")

        if algo == 'ga':
            print("  额外参数:")
            print("    - population_size: 种群大小")
            print("    - generations: 迭代次数")
            print("    - mutation_rate: 变异率")
            print("    - crossover_rate: 交叉率")


def example_constraint_configuration():
    """约束配置示例"""
    print("\n" + "=" * 70)
    print("示例4: 约束配置")
    print("=" * 70)

    print("""
# 启用/禁用特定约束
imaging_config = {
    # 资源约束
    'consider_power': True,           # 电量约束
    'consider_storage': True,         # 存储约束

    # 高精度要求：始终使用精确机动计算

    # 姿态约束
    'enable_attitude_calculation': True,  # 计算姿态角
}

# 所有约束都通过imaging_config传递给成像调度器
scheduler = UnifiedScheduler(
    mission=mission,
    window_cache=cache,
    config={
        'imaging_algorithm': 'greedy',
        'imaging_config': imaging_config,
    }
)
    """)


def example_result_analysis():
    """结果分析示例"""
    print("\n" + "=" * 70)
    print("示例5: 结果分析")
    print("=" * 70)

    print("""
# 执行调度
result = scheduler.schedule()

# 1. 基本统计
metrics = scheduler.get_metrics(result)
print(f"成像任务数: {metrics['imaging_scheduled']}")
print(f"需求满足率: {metrics['demand_satisfaction_rate']:.2%}")
print(f"卫星利用率: {metrics['satellite_utilization']:.2%}")

# 2. 频次满足度分析
for target_id, info in result.target_observations.items():
    print(f"目标 {target_id}:")
    print(f"  需求: {info['required']}次")
    print(f"  实际: {info['actual']}次")
    print(f"  状态: {info['status']}")
    print(f"  满足: {'是' if info['satisfied'] else '否'}")

# 3. 数传统计
if result.downlink_result:
    print(f"成功数传: {len(result.downlink_result.downlink_tasks)}")
    print(f"失败数传: {len(result.downlink_result.failed_tasks)}")

    # 固存状态
    for sat_id, storage in result.downlink_result.storage_states.items():
        print(f"卫星 {sat_id} 固存: {storage.current_gb:.2f}/{storage.capacity_gb:.2f} GB")

# 4. 保存详细结果
result_dict = result.to_dict()
# 可以保存为JSON或进行进一步分析
    """)


if __name__ == '__main__':
    print("统一调度器使用示例")
    print("=" * 70)

    example_basic_usage()
    example_with_downlink()
    example_different_algorithms()
    example_constraint_configuration()
    example_result_analysis()

    print("\n" + "=" * 70)
    print("提示: 运行实际调度请使用 scripts/run_unified_scheduler.py")
    print("=" * 70)
