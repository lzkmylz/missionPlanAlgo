#!/usr/bin/env python3
"""
大规模场景可见性计算脚本

使用批量可见性计算器处理60卫星+1000目标的场景

用法:
    python scripts/compute_large_scale_visibility.py
    python scripts/compute_large_scale_visibility.py --scenario scenarios/large_scale_experiment.json --output results/visibility_cache.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orbit.visibility.batch_calculator import (
    BatchVisibilityCalculator,
    BatchComputationConfig,
    BatchComputationStats
)
from scripts.load_large_scale_scenario import (
    load_satellites,
    load_targets,
    load_ground_stations
)


def compute_visibility(
    scenario_path: str,
    output_path: str,
    coarse_step: float = 300.0,  # 5分钟粗扫描
    fine_step: float = 60.0,     # 1分钟精化
    min_elevation: float = 5.0   # 最小仰角5度
) -> Dict[str, Any]:
    """
    计算大规模场景的可见性窗口

    Args:
        scenario_path: 场景文件路径
        output_path: 输出文件路径
        coarse_step: 粗扫描步长(秒)
        fine_step: 精化步长(秒)
        min_elevation: 最小仰角(度)

    Returns:
        计算结果统计
    """
    print("=" * 70)
    print("大规模场景可见性计算")
    print("=" * 70)

    # 1. 加载场景
    print(f"\n[1/4] 加载场景文件: {scenario_path}")
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    satellites = load_satellites(scenario_data)
    targets = load_targets(scenario_data)
    ground_stations = load_ground_stations(scenario_data)

    print(f"  - 卫星: {len(satellites)} 颗")
    print(f"  - 目标: {len(targets)} 个")
    print(f"  - 地面站: {len(ground_stations)} 个")

    # 计算总对数
    total_pairs = len(satellites) * len(targets) + len(satellites) * len(ground_stations)
    print(f"  - 总计算对数: {total_pairs:,} (卫星-目标 + 卫星-地面站)")

    # 2. 提取时间范围
    duration = scenario_data['duration']
    start_time = datetime.fromisoformat(duration['start'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(duration['end'].replace('Z', '+00:00'))
    time_span_hours = (end_time - start_time).total_seconds() / 3600

    print(f"\n[2/4] 时间范围")
    print(f"  - 开始: {start_time}")
    print(f"  - 结束: {end_time}")
    print(f"  - 时长: {time_span_hours} 小时")

    # 3. 配置批量计算
    print(f"\n[3/4] 配置批量计算参数")
    config = BatchComputationConfig(
        coarse_step_seconds=coarse_step,
        fine_step_seconds=fine_step,
        min_elevation_degrees=min_elevation,
        use_parallel_propagation=True,  # 启用Java端并行
        max_batch_size=100,              # 每批最大对数
        fallback_on_error=True          # 出错时回退到逐对计算
    )
    print(f"  - 粗扫描步长: {coarse_step} 秒")
    print(f"  - 精化步长: {fine_step} 秒")
    print(f"  - 最小仰角: {min_elevation} 度")
    print(f"  - Java端并行: 启用")

    # 4. 执行计算
    print(f"\n[4/4] 执行批量可见性计算...")
    print("  (这可能需要几分钟时间，请耐心等待)")
    print()

    calculator = BatchVisibilityCalculator()

    try:
        result = calculator.compute_all_windows(
            satellites=satellites,
            targets=targets,
            ground_stations=ground_stations,
            start_time=start_time,
            end_time=end_time,
            config=config
        )

        # 获取统计信息
        stats = calculator.last_computation_stats

        print("\n" + "=" * 70)
        print("计算完成!")
        print("=" * 70)

        print(f"\n结果统计:")
        print(f"  - 卫星-目标窗口: {sum(len(w) for w in result.target_windows.values()):,} 个")
        print(f"  - 卫星-地面站窗口: {sum(len(w) for w in result.ground_station_windows.values()):,} 个")
        print(f"  - 总窗口数: {result.total_window_count:,}")
        print(f"  - 是否回退计算: {result.is_fallback_result}")

        if stats:
            print(f"\n性能统计:")
            print(f"  - 总计算时间: {stats.total_computation_time_ms:,} ms ({stats.total_computation_time_ms/1000:.1f} 秒)")
            print(f"  - JNI调用时间: {stats.jni_call_time_ms:,} ms")
            print(f"  - Java计算时间: {stats.java_computation_time_ms:,} ms")
            print(f"  - Python开销: {stats.python_overhead_ms:,} ms")
            print(f"  - 内存使用: {stats.memory_usage_mb:.1f} MB")
            if stats.total_computation_time_ms > 0:
                throughput = total_pairs / (stats.total_computation_time_ms / 1000)
                print(f"  - 计算吞吐率: {throughput:.1f} 对/秒")

        # 5. 保存结果
        print(f"\n保存结果到: {output_path}")
        output_data = {
            'metadata': {
                'scenario': scenario_data['name'],
                'generated_at': datetime.now().isoformat(),
                'time_range': {
                    'start': duration['start'],
                    'end': duration['end']
                },
                'entities': {
                    'satellites': len(satellites),
                    'targets': len(targets),
                    'ground_stations': len(ground_stations)
                },
                'computation_config': {
                    'coarse_step_seconds': coarse_step,
                    'fine_step_seconds': fine_step,
                    'min_elevation_degrees': min_elevation
                }
            },
            'stats': {
                'total_windows': result.total_window_count,
                'target_windows': sum(len(w) for w in result.target_windows.values()),
                'ground_station_windows': sum(len(w) for w in result.ground_station_windows.values()),
                'computation_time_ms': stats.total_computation_time_ms if stats else None,
                'is_fallback': result.is_fallback_result
            },
            'windows': result.to_cache_format()
        }

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  文件大小: {file_size_mb:.2f} MB")

        print("\n" + "=" * 70)
        print("可见性计算完成!")
        print("=" * 70)

        return output_data

    except Exception as e:
        print(f"\n错误: 计算失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='计算大规模场景的可见性窗口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/compute_large_scale_visibility.py
  python scripts/compute_large_scale_visibility.py --coarse-step 600 --fine-step 120
  python scripts/compute_large_scale_visibility.py --output results/my_visibility.json
        """
    )
    parser.add_argument(
        '--scenario', '-s',
        default='scenarios/large_scale_experiment.json',
        help='场景文件路径 (默认: scenarios/large_scale_experiment.json)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/large_scale_visibility.json',
        help='输出文件路径 (默认: results/large_scale_visibility.json)'
    )
    parser.add_argument(
        '--coarse-step',
        type=float,
        default=300.0,
        help='粗扫描步长(秒)，默认300秒(5分钟)'
    )
    parser.add_argument(
        '--fine-step',
        type=float,
        default=60.0,
        help='精化步长(秒)，默认60秒(1分钟)'
    )
    parser.add_argument(
        '--min-elevation',
        type=float,
        default=5.0,
        help='最小仰角(度)，默认5度'
    )

    args = parser.parse_args()

    try:
        result = compute_visibility(
            scenario_path=args.scenario,
            output_path=args.output,
            coarse_step=args.coarse_step,
            fine_step=args.fine_step,
            min_elevation=args.min_elevation
        )
        return 0
    except FileNotFoundError as e:
        print(f"\n错误: 文件不存在: {e}")
        print("请先运行: python scripts/generate_large_scale_scenario.py")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
