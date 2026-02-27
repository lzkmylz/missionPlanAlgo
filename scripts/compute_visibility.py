"""
可见性计算脚本

计算场景中所有卫星-目标和卫星-地面站的可见窗口。
结果保存到缓存文件，供调度算法使用。
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.orbit.visibility.batch_calculator import BatchVisibilityCalculator, BatchComputationConfig
from core.models import Mission


def compute_visibility_windows(scenario_path: str, output_dir: str = "data/visibility_cache", use_batch: bool = True):
    """
    计算场景中所有可见窗口。

    Args:
        scenario_path: 场景文件路径
        output_dir: 缓存输出目录
        use_batch: 是否使用批量计算（默认True）
    """
    import time
    total_start = time.time()

    print("=" * 60)
    print("可见性计算 (Phase 1+2+3+批量优化)")
    print("=" * 60)

    # 1. 加载场景
    print(f"\n1. 加载场景: {scenario_path}")
    mission = Mission.load(scenario_path)
    print(f"   - 卫星: {len(mission.satellites)} 颗")
    print(f"   - 目标: {len(mission.targets)} 个")
    print(f"   - 地面站: {len(mission.ground_stations)} 个")

    # 2. 选择计算模式
    if use_batch:
        print("\n2. 使用批量计算模式（Phase 4 优化）")
        return _compute_with_batch(mission, scenario_path, output_dir, total_start)
    else:
        print("\n2. 使用逐对计算模式（Phase 1+2+3 优化）")
        return _compute_pairwise(mission, scenario_path, output_dir, total_start)


def _compute_with_batch(mission, scenario_path, output_dir, total_start):
    """使用批量计算模式"""
    from datetime import datetime
    from pathlib import Path
    import json

    # 创建批量计算器
    print("   - 初始化批量计算器")
    calc = BatchVisibilityCalculator()

    config = BatchComputationConfig(
        coarse_step_seconds=300,
        fine_step_seconds=60,
        min_elevation_degrees=0.0,
        use_parallel_propagation=True,
        fallback_on_error=True,
    )

    # 3. 计算时间范围
    start_time = mission.start_time
    end_time = mission.end_time
    print(f"\n3. 计算时间范围")
    print(f"   - 开始: {start_time}")
    print(f"   - 结束: {end_time}")

    # 4. 执行批量计算
    print("\n4. 执行批量可见性计算")
    calc_start = time.time()
    result = calc.compute_all_windows(
        satellites=mission.satellites,
        targets=mission.targets,
        ground_stations=mission.ground_stations,
        start_time=start_time,
        end_time=end_time,
        config=config,
    )
    calc_time = time.time() - calc_start

    print(f"   计算完成: {result.total_window_count} 个窗口")
    print(f"   计算耗时: {calc_time:.2f} 秒")

    if result.computation_stats:
        stats = result.computation_stats
        print(f"   - Java端耗时: {stats.java_computation_time_ms} ms")
        print(f"   - Python开销: {stats.python_overhead_ms} ms")
        print(f"   - 内存使用: {stats.memory_usage_mb:.1f} MB")

    if result.is_fallback_result:
        print("   ⚠ 使用了回退模式（逐对计算）")

    # 5. 保存结果
    print("\n5. 保存缓存")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cache_file = Path(output_dir) / f"{Path(scenario_path).stem}_windows.json"
    cache_data = result.to_cache_format()
    cache_data['scenario'] = scenario_path
    cache_data['computed_at'] = datetime.now().isoformat()
    cache_data['computation_mode'] = 'batch'
    if result.computation_stats:
        cache_data['computation_stats'] = {
            'total_time_ms': result.computation_stats.total_computation_time_ms,
            'java_time_ms': result.computation_stats.java_computation_time_ms,
            'memory_mb': result.computation_stats.memory_usage_mb,
        }

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    print(f"   缓存已保存: {cache_file}")

    # 6. 统计
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("计算完成 (批量模式)")
    print("=" * 60)
    target_count = len(cache_data.get('target_windows', []))
    gs_count = len(cache_data.get('ground_station_windows', []))
    print(f"卫星-目标窗口: {target_count} 个")
    print(f"卫星-地面站窗口: {gs_count} 个")
    print(f"总计: {target_count + gs_count} 个窗口")
    print(f"\n总耗时: {total_elapsed:.2f} 秒")

    # 性能对比
    baseline_time = 420  # 当前基线420秒
    speedup = baseline_time / total_elapsed if total_elapsed > 0 else 0
    print(f"性能对比:")
    print(f"  基线时间: {baseline_time} 秒 (逐对计算)")
    print(f"  批量时间: {total_elapsed:.2f} 秒 (批量优化)")
    print(f"  加速比: {speedup:.1f}x")

    if speedup >= 10:
        print(f"  🎉 优秀! 达到10倍+加速!")
    elif speedup >= 5:
        print(f"  ✅ 良好! 达到5倍+加速!")
    elif speedup >= 2:
        print(f"  ✓ 达到2倍+加速")
    else:
        print(f"  ⚠ 加速比未达预期")
    print("=" * 60)

    return cache_file


def _compute_pairwise(mission, scenario_path, output_dir, total_start):
    """使用逐对计算模式（原始实现）"""
    from datetime import datetime
    from pathlib import Path
    import json

    calculator = OrekitVisibilityCalculator(config={
        'min_elevation': 0.0,
        'use_adaptive_step': True,
        'coarse_step_seconds': 300,
        'fine_step_seconds': 60,
        'use_java_orekit': True,
        'use_parallel': True,
        'max_workers': None,
    })

    print(f"   - 后端: Java Orekit")
    print(f"   - 自适应步长: 启用")
    print(f"   - 多线程并行: 启用")

    start_time = mission.start_time
    end_time = mission.end_time
    print(f"\n3. 计算时间范围")
    print(f"   - 开始: {start_time}")
    print(f"   - 结束: {end_time}")

    print("\n4. 计算卫星-目标可见窗口")
    target_windows = []
    for sat in mission.satellites:
        for target in mission.targets:
            windows = calculator.compute_satellite_target_windows(
                satellite=sat, target=target, start_time=start_time, end_time=end_time
            )
            target_windows.extend(windows)
            if windows:
                print(f"   {sat.id} -> {target.id}: {len(windows)} 个窗口")

    print("\n5. 计算卫星-地面站可见窗口")
    gs_windows = []
    for sat in mission.satellites:
        for gs in mission.ground_stations:
            windows = calculator.compute_satellite_ground_station_windows(
                satellite=sat, ground_station=gs, start_time=start_time, end_time=end_time
            )
            gs_windows.extend(windows)
            if windows:
                print(f"   {sat.id} -> {gs.id}: {len(windows)} 个窗口")

    print("\n6. 保存缓存")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cache_file = Path(output_dir) / f"{Path(scenario_path).stem}_windows.json"
    cache_data = {
        'scenario': scenario_path,
        'computed_at': datetime.now().isoformat(),
        'computation_mode': 'pairwise',
        'target_windows': [
            {
                'satellite_id': w.satellite_id,
                'target_id': w.target_id,
                'start_time': w.start_time.isoformat(),
                'end_time': w.end_time.isoformat(),
                'duration': (w.end_time - w.start_time).total_seconds()
            }
            for w in target_windows
        ],
        'ground_station_windows': [
            {
                'satellite_id': w.satellite_id,
                'target_id': w.target_id,
                'start_time': w.start_time.isoformat(),
                'end_time': w.end_time.isoformat(),
                'duration': (w.end_time - w.start_time).total_seconds()
            }
            for w in gs_windows
        ]
    }

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    print(f"   缓存已保存: {cache_file}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("计算完成 (逐对模式)")
    print("=" * 60)
    print(f"卫星-目标窗口: {len(target_windows)} 个")
    print(f"卫星-地面站窗口: {len(gs_windows)} 个")
    print(f"总计: {len(target_windows) + len(gs_windows)} 个窗口")
    print(f"\n总耗时: {total_elapsed:.2f} 秒")
    print("=" * 60)

    return cache_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算场景可见性窗口")
    parser.add_argument("--scenario", "-s", default="scenarios/point_group_scenario.json",
                       help="场景文件路径")
    parser.add_argument("--output", "-o", default="data/visibility_cache",
                       help="缓存输出目录")
    parser.add_argument("--mode", "-m", choices=["batch", "pairwise"], default="batch",
                       help="计算模式: batch=批量计算(快), pairwise=逐对计算(慢)")

    args = parser.parse_args()

    compute_visibility_windows(args.scenario, args.output, use_batch=(args.mode == "batch"))
