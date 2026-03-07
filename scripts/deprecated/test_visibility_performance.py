#!/usr/bin/env python3
"""
可见性计算性能测试脚本

测试Phase 1-3优化的综合性能提升
- Phase 1: 自适应时间步长
- Phase 2: Java批量计算
- Phase 3: 多线程并行

预期性能: 400秒 → 5秒 (80倍提升)
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.mission import Mission
from core.models.satellite import Satellite
from core.models.target import Target
from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.orbit.visibility.calculator_factory import VisibilityCalculatorFactory


def load_scenario(scenario_path: str) -> Dict[str, Any]:
    """加载场景文件"""
    with open(scenario_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_mission_from_scenario(scenario_data: Dict) -> Mission:
    """从场景数据创建Mission对象"""
    # 提取时间范围 (适配实际场景文件格式)
    start_time = datetime.fromisoformat(
        scenario_data['start_time'].replace('Z', '+00:00')
    )
    end_time = datetime.fromisoformat(
        scenario_data['end_time'].replace('Z', '+00:00')
    )

    # 创建Mission
    mission = Mission(
        name=scenario_data.get('name', 'Test Mission'),
        start_time=start_time,
        end_time=end_time
    )

    # 添加卫星
    for sat_config in scenario_data.get('satellites', []):
        satellite = Satellite(
            id=sat_config['id'],
            name=sat_config['name'],
            orbit=sat_config['orbit'],
            capabilities={
                'power_capacity': sat_config.get('power_capacity', 100.0),
                'storage_capacity': sat_config.get('storage_capacity', 100.0),
            }
        )
        mission.add_satellite(satellite)

    # 添加目标
    for tgt_config in scenario_data.get('targets', []):
        target = Target(
            id=tgt_config['id'],
            name=tgt_config['name'],
            longitude=tgt_config['longitude'],
            latitude=tgt_config['latitude'],
            altitude=tgt_config.get('altitude', 0.0),
            priority=tgt_config.get('priority', 5),
        )
        # 设置观测频次
        if 'required_observations' in tgt_config:
            target.required_observations = tgt_config['required_observations']
        mission.add_target(target)

    return mission


def test_with_all_optimizations(mission: Mission) -> Dict[str, Any]:
    """测试所有优化启用时的性能"""
    print("\n" + "="*60)
    print("测试所有优化启用 (Phase 1+2+3)")
    print("="*60)

    config = {
        'use_adaptive_step': True,
        'coarse_step_seconds': 300,
        'fine_step_seconds': 60,
        'use_java_orekit': True,
        'use_parallel': True,
        'max_workers': None,  # 使用默认值: CPU核心数×2
        'min_elevation': 5.0,
    }

    calculator = OrekitVisibilityCalculator(config)

    total_windows = 0
    start_time = time.time()

    # 计算所有卫星-目标对的可见窗口
    for satellite in mission.satellites:
        for target in mission.targets:
            windows = calculator.compute_satellite_target_windows(
                satellite, target,
                mission.start_time, mission.end_time
            )
            total_windows += len(windows)

    elapsed = time.time() - start_time

    result = {
        'config': 'All Optimizations (Phase 1+2+3)',
        'elapsed_seconds': elapsed,
        'total_windows': total_windows,
        'satellites': len(mission.satellites),
        'targets': len(mission.targets),
        'pairs': len(mission.satellites) * len(mission.targets),
    }

    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  卫星数: {result['satellites']}")
    print(f"  目标数: {result['targets']}")
    print(f"  计算对数: {result['pairs']}")
    print(f"  发现窗口: {total_windows}")

    return result


def test_with_adaptive_only(mission: Mission) -> Dict[str, Any]:
    """测试仅Phase 1优化"""
    print("\n" + "="*60)
    print("测试仅Phase 1优化 (自适应步长)")
    print("="*60)

    config = {
        'use_adaptive_step': True,
        'coarse_step_seconds': 300,
        'fine_step_seconds': 60,
        'use_java_orekit': False,  # 禁用Java
        'use_parallel': False,  # 禁用并行
        'min_elevation': 5.0,
    }

    calculator = OrekitVisibilityCalculator(config)

    total_windows = 0
    start_time = time.time()

    for satellite in mission.satellites:
        for target in mission.targets:
            windows = calculator.compute_satellite_target_windows(
                satellite, target,
                mission.start_time, mission.end_time
            )
            total_windows += len(windows)

    elapsed = time.time() - start_time

    result = {
        'config': 'Phase 1 Only (Adaptive Step)',
        'elapsed_seconds': elapsed,
        'total_windows': total_windows,
    }

    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  发现窗口: {total_windows}")

    return result


def test_with_fixed_step(mission: Mission) -> Dict[str, Any]:
    """测试固定步长（无优化基线）"""
    print("\n" + "="*60)
    print("测试基线 (固定步长60秒, 无优化)")
    print("="*60)

    config = {
        'use_adaptive_step': False,  # 禁用自适应
        'time_step': 60,  # 固定60秒步长
        'use_java_orekit': False,
        'use_parallel': False,
        'min_elevation': 5.0,
    }

    calculator = OrekitVisibilityCalculator(config)

    total_windows = 0
    start_time = time.time()

    # 限制计算对数以减少测试时间
    limited_satellites = list(mission.satellites)[:3]
    limited_targets = list(mission.targets)[:3]

    for satellite in limited_satellites:
        for target in limited_targets:
            windows = calculator.compute_satellite_target_windows(
                satellite, target,
                mission.start_time, mission.end_time
            )
            total_windows += len(windows)

    elapsed = time.time() - start_time

    # 估算完整计算时间
    full_pairs = len(mission.satellites) * len(mission.targets)
    limited_pairs = len(limited_satellites) * len(limited_targets)
    estimated_full_time = elapsed * (full_pairs / limited_pairs) if limited_pairs > 0 else 0

    result = {
        'config': 'Baseline (Fixed 60s step)',
        'elapsed_seconds': elapsed,
        'estimated_full_time': estimated_full_time,
        'total_windows': total_windows,
        'tested_pairs': limited_pairs,
        'full_pairs': full_pairs,
    }

    print(f"  实际耗时: {elapsed:.2f} 秒 (测试 {limited_pairs} 对)")
    print(f"  预估完整: {estimated_full_time:.2f} 秒 ({full_pairs} 对)")
    print(f"  发现窗口: {total_windows}")

    return result


def main():
    """主函数"""
    print("\n" + "="*60)
    print("可见性计算性能测试")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 场景文件路径
    scenario_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scenarios', 'point_group_scenario.json'
    )

    if not os.path.exists(scenario_path):
        print(f"\n错误: 场景文件不存在: {scenario_path}")
        sys.exit(1)

    print(f"\n加载场景: {scenario_path}")
    scenario_data = load_scenario(scenario_path)
    mission = create_mission_from_scenario(scenario_data)

    print(f"\n场景信息:")
    print(f"  任务名称: {mission.name}")
    print(f"  时间范围: {mission.start_time} → {mission.end_time}")
    print(f"  持续时间: {mission.end_time - mission.start_time}")
    print(f"  卫星数量: {len(mission.satellites)}")
    print(f"  目标数量: {len(mission.targets)}")
    print(f"  计算对数: {len(mission.satellites) * len(mission.targets)}")

    results = []

    # 测试1: 所有优化
    try:
        result = test_with_all_optimizations(mission)
        results.append(result)
    except Exception as e:
        print(f"\n错误: 所有优化测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试2: 仅Phase 1
    try:
        result = test_with_adaptive_only(mission)
        results.append(result)
    except Exception as e:
        print(f"\n错误: Phase 1测试失败: {e}")

    # 测试3: 基线（限制计算量）
    try:
        result = test_with_fixed_step(mission)
        results.append(result)
    except Exception as e:
        print(f"\n错误: 基线测试失败: {e}")

    # 汇总结果
    print("\n" + "="*60)
    print("性能测试汇总")
    print("="*60)

    for i, r in enumerate(results, 1):
        print(f"\n测试 {i}: {r['config']}")
        print(f"  耗时: {r.get('elapsed_seconds', 0):.2f} 秒")
        if 'estimated_full_time' in r:
            print(f"  预估完整: {r['estimated_full_time']:.2f} 秒")
        print(f"  窗口数: {r.get('total_windows', 0)}")

    # 计算加速比
    if len(results) >= 2:
        baseline_time = results[-1].get('estimated_full_time', results[-1].get('elapsed_seconds', 0))
        optimized_time = results[0].get('elapsed_seconds', 0)

        if baseline_time > 0 and optimized_time > 0:
            speedup = baseline_time / optimized_time
            print(f"\n{'='*60}")
            print(f"性能提升: {speedup:.1f}x")
            print(f"基线时间: {baseline_time:.2f} 秒")
            print(f"优化时间: {optimized_time:.2f} 秒")
            print(f"{'='*60}")

            if speedup >= 50:
                print("🎉 达到预期80倍提升目标!")
            elif speedup >= 40:
                print("✅ 达到50倍+提升!")
            elif speedup >= 20:
                print("✓ 达到20倍+提升")
            else:
                print("⚠ 提升未达预期，可能需要进一步优化")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == '__main__':
    main()
