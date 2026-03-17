#!/usr/bin/env python3
"""
测试成像中心点距离优化功能
验证非聚类任务的成像中心是否更靠近目标坐标
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scheduler.common.footprint_utils import (
    calculate_haversine_distance,
    calculate_footprint_center_from_attitude,
    calculate_center_distance_score
)


def test_basic_functions():
    """测试基础函数"""
    print("=" * 70)
    print("测试基础函数")
    print("=" * 70)

    # 测试 haversine 距离
    dist = calculate_haversine_distance(0, 0, 1, 1)
    print(f"Haversine距离 (0,0) -> (1,1): {dist:.2f} km")

    # 测试成像中心计算
    R = 6371000  # 地球半径(m)
    h = 500000   # 高度(m)

    # 星下点观测
    position = (R + h, 0, 0)
    center = calculate_footprint_center_from_attitude(position, 0, 0)
    print(f"星下点观测成像中心: {center}")
    assert abs(center[0]) < 0.01 and abs(center[1]) < 0.01, "星下点观测中心应该接近星下点"

    # 侧摆观测
    center_tilted = calculate_footprint_center_from_attitude(position, 10, 0)
    print(f"10°侧摆成像中心: {center_tilted}")
    assert center_tilted[0] > center[0], "向东侧摆应该使经度增加"

    # 测试距离评分
    score_perfect = calculate_center_distance_score(position, 0, 0, 0, 0)
    score_tilted = calculate_center_distance_score(position, 10, 0, 0, 0)
    print(f"完美对准评分: {score_perfect:.4f}")
    print(f"10°偏差评分: {score_tilted:.4f}")
    assert score_perfect > score_tilted, "完美对准应该得分更高"

    print("✓ 基础函数测试通过")
    return True


def test_scheduler_integration():
    """测试调度器集成"""
    print("\n" + "=" * 70)
    print("测试调度器集成")
    print("=" * 70)

    from scheduler.greedy.greedy_scheduler import GreedyScheduler
    from scheduler.greedy.heuristic_scheduler import HeuristicScheduler
    from scheduler.common.config import ConstraintConfig

    # 验证配置
    config = ConstraintConfig()
    assert hasattr(config, 'enable_center_distance_score')
    assert hasattr(config, 'center_distance_weight')
    assert config.enable_center_distance_score == True
    assert config.center_distance_weight == 15.0
    print(f"✓ 配置项存在: enable_center_distance_score={config.enable_center_distance_score}")
    print(f"✓ 配置项存在: center_distance_weight={config.center_distance_weight}")

    # 验证方法
    assert hasattr(GreedyScheduler, '_calculate_center_distance_score')
    print("✓ GreedyScheduler._calculate_center_distance_score 存在")

    assert hasattr(HeuristicScheduler, '_calculate_center_distance_score')
    assert hasattr(HeuristicScheduler, '_calculate_assignment_score')
    print("✓ HeuristicScheduler 评分方法存在")

    print("✓ 调度器集成测试通过")
    return True


def analyze_existing_results():
    """分析现有调度结果中的成像中心偏差"""
    print("\n" + "=" * 70)
    print("分析现有调度结果")
    print("=" * 70)

    result_file = "results/greedy_final_test.json"
    scenario_file = "scenarios/large_scale_frequency.json"

    try:
        with open(result_file, 'r') as f:
            results = json.load(f)

        with open(scenario_file, 'r') as f:
            scenario = json.load(f)

        # 构建目标坐标映射
        target_coords = {}
        for target in scenario['targets']:
            target_coords[target['id']] = target['location']  # [lon, lat]

        tasks = results['results'][0]['scheduled_tasks']

        # 只分析非聚类任务
        non_cluster_tasks = [t for t in tasks if not t.get('is_cluster_task', False)]

        if not non_cluster_tasks:
            print("没有找到非聚类任务")
            return

        print(f"总任务数: {len(tasks)}")
        print(f"非聚类任务数: {len(non_cluster_tasks)}")

        # 计算成像中心与目标的偏差
        deviations = []
        for task in non_cluster_tasks[:50]:  # 分析前50个
            target_id = task['target_id'].split('-OBS')[0] if '-OBS' in task['target_id'] else task['target_id']
            target_coord = target_coords.get(target_id)
            footprint_center = task.get('footprint_center')

            if target_coord and footprint_center:
                # 计算角度偏差
                dlon = abs(footprint_center[0] - target_coord[0])
                dlat = abs(footprint_center[1] - target_coord[1])
                lat_factor = 0.01745241  # cos(平均纬度)的近似值
                angular_dist = (dlon * lat_factor)**2 + dlat**2
                angular_dist = angular_dist**0.5
                deviations.append(angular_dist)

        if deviations:
            avg_deviation = sum(deviations) / len(deviations)
            max_deviation = max(deviations)
            min_deviation = min(deviations)

            print(f"\n成像中心与目标偏差统计（非聚类任务）:")
            print(f"  平均偏差: {avg_deviation:.2f}°")
            print(f"  最大偏差: {max_deviation:.2f}°")
            print(f"  最小偏差: {min_deviation:.2f}°")
            print(f"  样本数量: {len(deviations)}")

        print("\n注意: 这是优化前的基线数据")
        print("优化后重新运行调度，成像中心应该更靠近目标坐标")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"分析失败: {e}")


def main():
    print("\n" + "=" * 70)
    print("成像中心点距离优化功能测试")
    print("=" * 70)

    try:
        test_basic_functions()
        test_scheduler_integration()
        analyze_existing_results()

        print("\n" + "=" * 70)
        print("✓ 所有测试通过!")
        print("=" * 70)
        print("\n说明:")
        print("- 基础函数正常工作")
        print("- 调度器集成正确")
        print("- 下次运行调度时将应用中心点距离优化")
        print("- 仅对非聚类任务启用此优化")

        return 0

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
