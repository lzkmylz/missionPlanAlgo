#!/usr/bin/env python3
"""
统一可见性计算脚本 - 合并了以下脚本的功能:
- compute_visibility.py (原)
- compute_large_scale_visibility.py
- compute_large_scale_visibility_parallel.py

用法:
    # 批量计算模式 (默认, 最快)
    python scripts/compute_visibility.py -s scenario.json

    # 逐对计算模式 (较慢但更详细)
    python scripts/compute_visibility.py -s scenario.json --mode pairwise

    # 自定义参数 (默认: 粗扫描5秒, 精化1秒)
    python scripts/compute_visibility.py -s scenario.json --coarse-step 10 --fine-step 2

    # 指定输出路径
    python scripts/compute_visibility.py -s scenario.json -o results/my_visibility.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from core.orbit.visibility.batch_calculator import (
    BatchVisibilityCalculator,
    BatchComputationConfig
)
from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

# 从 utils 导入公共功能
from scripts.utils import setup_logging, save_results

# 默认轨道数据输出路径
DEFAULT_ORBIT_OUTPUT_PATH = "java/output/frequency_scenario/orbits.json.gz"


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='统一可见性计算脚本 - 支持批量和逐对计算模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量计算模式 (默认, 推荐)，自动导出轨道数据
  python scripts/compute_visibility.py -s scenarios/large_scale_frequency.json

  # 批量计算但禁用轨道数据导出
  python scripts/compute_visibility.py -s scenarios/large_scale_frequency.json --no-orbit-export

  # 指定轨道数据输出路径
  python scripts/compute_visibility.py -s scenario.json --orbit-output custom/path/orbits.json.gz

  # 逐对计算模式
  python scripts/compute_visibility.py -s scenario.json --mode pairwise

  # 自定义扫描步长 (默认: 粗扫描5秒, 精化1秒)
  python scripts/compute_visibility.py -s scenario.json --coarse-step 10 --fine-step 2

  # 指定最小仰角
  python scripts/compute_visibility.py -s scenario.json --min-elevation 10
        """
    )

    # 必需参数
    parser.add_argument(
        '-s', '--scenario',
        required=True,
        help='场景配置文件路径 (JSON格式)'
    )

    # 计算模式
    parser.add_argument(
        '--mode',
        choices=['batch', 'pairwise'],
        default='batch',
        help='计算模式: batch=批量计算(快,默认), pairwise=逐对计算(慢)'
    )

    # 计算参数 (与Java后端保持一致)
    parser.add_argument(
        '--coarse-step',
        type=float,
        default=5.0,
        help='粗扫描步长(秒)，默认5秒'
    )
    parser.add_argument(
        '--fine-step',
        type=float,
        default=1.0,
        help='精化步长(秒)，默认1秒'
    )
    parser.add_argument(
        '--min-elevation',
        type=float,
        default=5.0,
        help='最小仰角(度)，默认5度'
    )

    # 输出
    parser.add_argument(
        '-o', '--output',
        default='results/visibility_cache.json',
        help='输出文件路径 (默认: results/visibility_cache.json)'
    )
    parser.add_argument(
        '--orbit-output',
        default=DEFAULT_ORBIT_OUTPUT_PATH,
        help=f'轨道数据输出路径 (默认: {DEFAULT_ORBIT_OUTPUT_PATH})'
    )
    parser.add_argument(
        '--no-orbit-export',
        action='store_true',
        help='禁用轨道数据导出 (默认启用)'
    )

    return parser.parse_args(args)


def build_computation_config(args: argparse.Namespace) -> BatchComputationConfig:
    """构建计算配置"""
    return BatchComputationConfig(
        coarse_step_seconds=args.coarse_step,
        fine_step_seconds=args.fine_step,
        min_elevation_degrees=args.min_elevation,
        use_parallel_propagation=True,
        max_batch_size=100
    )


def compute_visibility_batch(
    mission: Mission,
    config: BatchComputationConfig,
    output_path: str,
    orbit_output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用批量模式计算可见性，并可选导出轨道数据

    Args:
        mission: 任务场景
        config: 计算配置
        output_path: 输出路径
        orbit_output_path: 轨道数据输出路径（可选，为None则不导出）

    Returns:
        计算结果统计
    """
    logger = setup_logging()
    logger.info("使用批量计算模式" + ("（带轨道导出）" if orbit_output_path else ""))

    # 如果指定了轨道输出路径，使用带轨道导出的计算方法
    if orbit_output_path:
        return _compute_visibility_batch_with_orbit_export(
            mission, config, output_path, orbit_output_path
        )

    # 创建计算器
    calculator = BatchVisibilityCalculator()

    # 执行计算
    start_time = time.time()
    result = calculator.compute_all_windows(
        satellites=mission.satellites,
        targets=mission.targets,
        ground_stations=mission.ground_stations,
        start_time=mission.start_time,
        end_time=mission.end_time,
        config=config
    )
    computation_time = time.time() - start_time

    # 统计结果
    target_window_count = sum(len(w) for w in result.target_windows.values())
    gs_window_count = sum(len(w) for w in result.ground_station_windows.values())

    # 获取统计信息
    stats = calculator.last_computation_stats

    # 构建输出数据
    output_data = {
        'metadata': {
            'scenario': mission.name,
            'computed_at': datetime.now().isoformat(),
            'mode': 'batch',
            'time_range': {
                'start': mission.start_time.isoformat() if hasattr(mission.start_time, 'isoformat') else str(mission.start_time),
                'end': mission.end_time.isoformat() if hasattr(mission.end_time, 'isoformat') else str(mission.end_time)
            },
            'entities': {
                'satellites': len(mission.satellites),
                'targets': len(mission.targets),
                'ground_stations': len(mission.ground_stations)
            },
            'computation_config': {
                'coarse_step_seconds': config.coarse_step_seconds,
                'fine_step_seconds': config.fine_step_seconds,
                'min_elevation_degrees': config.min_elevation_degrees
            }
        },
        'stats': {
            'total_windows': result.total_window_count,
            'target_windows': target_window_count,
            'ground_station_windows': gs_window_count,
            'computation_time_seconds': computation_time
        },
        'windows': result.to_cache_format()
    }

    if stats:
        output_data['stats'].update({
            'java_time_ms': stats.java_computation_time_ms,
            'memory_mb': stats.memory_usage_mb,
            'throughput': target_window_count / (stats.total_computation_time_ms / 1000) if stats.total_computation_time_ms > 0 else 0
        })

    # 保存结果
    save_results(output_data, output_path)

    return output_data['stats']


def _compute_visibility_batch_with_orbit_export(
    mission: Mission,
    config: BatchComputationConfig,
    output_path: str,
    orbit_output_path: str
) -> Dict[str, Any]:
    """
    使用批量模式计算可见性并导出轨道数据（使用Java OptimizedVisibilityCalculator）

    Args:
        mission: 任务场景
        config: 计算配置
        output_path: 输出路径
        orbit_output_path: 轨道数据输出路径

    Returns:
        计算结果统计
    """
    from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge

    logger = setup_logging()
    logger.info("使用带轨道导出的批量计算模式")

    # 确保输出目录存在
    Path(orbit_output_path).parent.mkdir(parents=True, exist_ok=True)

    # 准备参数
    sat_params = []
    for sat in mission.satellites:
        orbit = getattr(sat, "orbit", None)
        orbit_type = getattr(orbit, "orbit_type", "SSO")
        if hasattr(orbit_type, 'value'):
            orbit_type = orbit_type.value

        sat_params.append({
            'id': sat.id,
            'name': getattr(sat, "name", sat.id),
            'orbitType': str(orbit_type),
            'semiMajorAxis': getattr(orbit, "semi_major_axis", 7016000.0),
            'eccentricity': getattr(orbit, "eccentricity", 0.001),
            'inclination': getattr(orbit, "inclination", 97.9),
            'raan': getattr(orbit, "raan", 0.0),
            'argOfPerigee': getattr(orbit, "arg_of_perigee", 90.0),
            'meanAnomaly': getattr(orbit, "mean_anomaly", 0.0),
            'altitude': getattr(orbit, "altitude", 645000.0),
        })

    target_params = []
    for target in mission.targets:
        target_params.append({
            'id': target.id,
            'name': getattr(target, "name", target.id),
            'longitude': target.longitude,
            'latitude': target.latitude,
            'altitude': getattr(target, "altitude", 0.0),
        })

    gs_params = []
    for gs in mission.ground_stations:
        gs_params.append({
            'id': gs.id,
            'name': getattr(gs, "name", gs.id),
            'longitude': gs.longitude,
            'latitude': gs.latitude,
            'altitude': getattr(gs, "altitude", 0.0),
            'minElevation': getattr(gs, "min_elevation", 5.0),
        })

    # 创建Java桥接器并执行计算
    bridge = OrekitJavaBridge()
    java_config = {
        'coarseStep': config.coarse_step_seconds,
        'fineStep': config.fine_step_seconds,
        'minElevation': config.min_elevation_degrees,
        'useParallel': config.use_parallel_propagation,
        'maxBatchSize': config.max_batch_size,
    }

    # 执行计算
    start_time = time.time()
    result = bridge.compute_visibility_batch_with_orbit_export(
        satellites=sat_params,
        targets=target_params,
        ground_stations=gs_params,
        start_time=mission.start_time,
        end_time=mission.end_time,
        config=java_config,
        orbit_output_path=orbit_output_path
    )
    computation_time = time.time() - start_time

    # 统计结果
    target_windows = result.get('targetWindows', [])
    gs_windows = result.get('groundStationWindows', [])
    target_window_count = len(target_windows)
    gs_window_count = len(gs_windows)
    total_windows = target_window_count + gs_window_count

    stats = result.get('stats', {})
    export_status = result.get('orbitExportStatus', {})

    # 构建输出数据
    output_data = {
        'metadata': {
            'scenario': mission.name,
            'computed_at': datetime.now().isoformat(),
            'mode': 'batch_with_orbit_export',
            'time_range': {
                'start': mission.start_time.isoformat() if hasattr(mission.start_time, 'isoformat') else str(mission.start_time),
                'end': mission.end_time.isoformat() if hasattr(mission.end_time, 'isoformat') else str(mission.end_time)
            },
            'entities': {
                'satellites': len(mission.satellites),
                'targets': len(mission.targets),
                'ground_stations': len(mission.ground_stations)
            },
            'computation_config': {
                'coarse_step_seconds': config.coarse_step_seconds,
                'fine_step_seconds': config.fine_step_seconds,
                'min_elevation_degrees': config.min_elevation_degrees
            }
        },
        'stats': {
            'total_windows': total_windows,
            'target_windows': target_window_count,
            'ground_station_windows': gs_window_count,
            'computation_time_seconds': computation_time,
            'java_time_ms': stats.get('computationTimeMs', 0),
            'orbit_export_success': export_status.get('success', False),
            'orbit_export_path': export_status.get('path', orbit_output_path) if export_status.get('success') else None,
        },
        'windows': {
            'target_windows': target_windows,
            'ground_station_windows': gs_windows
        }
    }

    if not export_status.get('success') and export_status.get('error'):
        output_data['stats']['orbit_export_error'] = export_status['error']

    # 保存结果
    save_results(output_data, output_path)

    return output_data['stats']


def compute_visibility_pairwise(
    mission: Mission,
    config: BatchComputationConfig,
    output_path: str
) -> Dict[str, Any]:
    """
    使用逐对模式计算可见性

    Args:
        mission: 任务场景
        config: 计算配置
        output_path: 输出路径

    Returns:
        计算结果统计
    """
    logger = setup_logging()
    logger.info("使用逐对计算模式")

    # 创建计算器
    calculator = OrekitVisibilityCalculator(config={
        'min_elevation': config.min_elevation_degrees,
        'use_adaptive_step': True,
        'coarse_step_seconds': config.coarse_step_seconds,
        'fine_step_seconds': config.fine_step_seconds,
        'use_java_orekit': True,
    })

    # 计算卫星-目标窗口
    logger.info("计算卫星-目标可见窗口...")
    target_windows = []
    start_time = time.time()

    for sat in mission.satellites:
        for target in mission.targets:
            windows = calculator.compute_satellite_target_windows(
                satellite=sat,
                target=target,
                start_time=mission.start_time,
                end_time=mission.end_time
            )
            target_windows.extend(windows)

    target_time = time.time() - start_time
    logger.info(f"卫星-目标窗口: {len(target_windows)} 个, 耗时: {target_time:.2f}秒")

    # 计算卫星-地面站窗口
    logger.info("计算卫星-地面站可见窗口...")
    gs_windows = []
    start_time = time.time()

    for sat in mission.satellites:
        for gs in mission.ground_stations:
            windows = calculator.compute_satellite_ground_station_windows(
                satellite=sat,
                ground_station=gs,
                start_time=mission.start_time,
                end_time=mission.end_time
            )
            gs_windows.extend(windows)

    gs_time = time.time() - start_time
    logger.info(f"卫星-地面站窗口: {len(gs_windows)} 个, 耗时: {gs_time:.2f}秒")

    # 构建输出数据
    output_data = {
        'metadata': {
            'scenario': mission.name,
            'computed_at': datetime.now().isoformat(),
            'mode': 'pairwise',
            'time_range': {
                'start': mission.start_time.isoformat() if hasattr(mission.start_time, 'isoformat') else str(mission.start_time),
                'end': mission.end_time.isoformat() if hasattr(mission.end_time, 'isoformat') else str(mission.end_time)
            },
            'entities': {
                'satellites': len(mission.satellites),
                'targets': len(mission.targets),
                'ground_stations': len(mission.ground_stations)
            },
            'computation_config': {
                'coarse_step_seconds': config.coarse_step_seconds,
                'fine_step_seconds': config.fine_step_seconds,
                'min_elevation_degrees': config.min_elevation_degrees
            }
        },
        'stats': {
            'total_windows': len(target_windows) + len(gs_windows),
            'target_windows': len(target_windows),
            'ground_station_windows': len(gs_windows),
            'computation_time_seconds': target_time + gs_time
        },
        'windows': {
            'target_windows': [
                {
                    'satellite_id': w.satellite_id,
                    'target_id': w.target_id,
                    'start_time': w.start_time.isoformat() if hasattr(w.start_time, 'isoformat') else str(w.start_time),
                    'end_time': w.end_time.isoformat() if hasattr(w.end_time, 'isoformat') else str(w.end_time),
                    'max_elevation': w.max_elevation
                }
                for w in target_windows
            ],
            'ground_station_windows': [
                {
                    'satellite_id': w.satellite_id,
                    'target_id': w.target_id,
                    'start_time': w.start_time.isoformat() if hasattr(w.start_time, 'isoformat') else str(w.start_time),
                    'end_time': w.end_time.isoformat() if hasattr(w.end_time, 'isoformat') else str(w.end_time),
                    'max_elevation': w.max_elevation
                }
                for w in gs_windows
            ]
        }
    }

    # 保存结果
    save_results(output_data, output_path)

    return output_data['stats']


def compute_visibility(
    scenario_path: str,
    output_path: str,
    mode: str = 'batch',
    coarse_step: float = 5.0,
    fine_step: float = 1.0,
    min_elevation: float = 5.0,
    orbit_output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    计算场景可见性窗口

    Args:
        scenario_path: 场景文件路径
        output_path: 输出文件路径
        mode: 计算模式 ('batch' 或 'pairwise')
        coarse_step: 粗扫描步长(秒)
        fine_step: 精化步长(秒)
        min_elevation: 最小仰角(度)
        orbit_output_path: 轨道数据输出路径（可选，batch模式支持）

    Returns:
        计算结果统计
    """
    # 加载场景
    mission = Mission.load(scenario_path)

    # 构建配置
    config = BatchComputationConfig(
        coarse_step_seconds=coarse_step,
        fine_step_seconds=fine_step,
        min_elevation_degrees=min_elevation,
        use_parallel_propagation=True,
        max_batch_size=100
    )

    # 根据模式选择计算方法
    if mode == 'batch':
        return compute_visibility_batch(mission, config, output_path, orbit_output_path)
    else:
        return compute_visibility_pairwise(mission, config, output_path)


def print_results(stats: Dict[str, Any], mode: str) -> None:
    """打印计算结果"""
    print("\n" + "="*70)
    print("可见性计算结果")
    print("="*70)

    print(f"\n计算模式: {mode}")
    print(f"卫星-目标窗口: {stats.get('target_windows', 0):,} 个")
    print(f"卫星-地面站窗口: {stats.get('ground_station_windows', 0):,} 个")
    print(f"总窗口数: {stats.get('total_windows', 0):,} 个")
    print(f"计算时间: {stats.get('computation_time_seconds', 0):.2f} 秒")

    if 'java_time_ms' in stats:
        print(f"Java计算时间: {stats['java_time_ms']/1000:.2f} 秒")
    if 'memory_mb' in stats:
        print(f"内存使用: {stats['memory_mb']:.1f} MB")
    if 'throughput' in stats:
        print(f"计算吞吐率: {stats['throughput']:.1f} 窗口/秒")

    # 显示轨道导出状态
    if 'orbit_export_success' in stats:
        print("\n" + "-"*70)
        print("轨道数据导出")
        print("-"*70)
        if stats['orbit_export_success']:
            print(f"状态: 成功")
            print(f"路径: {stats.get('orbit_export_path', 'N/A')}")
        else:
            print(f"状态: 失败")
            if 'orbit_export_error' in stats:
                print(f"错误: {stats['orbit_export_error']}")

    print("="*70)


def main(args: Optional[List[str]] = None) -> int:
    """主函数"""
    # 解析参数
    parsed_args = parse_args(args)

    # 设置日志
    setup_logging()

    print("="*70)
    print("统一可见性计算脚本")
    print("="*70)

    try:
        print(f"\n[1/2] 加载场景: {parsed_args.scenario}")
        mission = Mission.load(parsed_args.scenario)
        print(f"  卫星: {len(mission.satellites)} 颗")
        print(f"  目标: {len(mission.targets)} 个")
        print(f"  地面站: {len(mission.ground_stations)} 个")

        print(f"\n[2/2] 执行可见性计算")
        print(f"  模式: {parsed_args.mode}")
        print(f"  粗扫描步长: {parsed_args.coarse_step} 秒")
        print(f"  精化步长: {parsed_args.fine_step} 秒")
        print(f"  最小仰角: {parsed_args.min_elevation} 度")

        # 确定轨道输出路径
        orbit_output_path = None if parsed_args.no_orbit_export else parsed_args.orbit_output
        if orbit_output_path:
            print(f"  轨道数据导出: 启用")
            print(f"  轨道数据路径: {orbit_output_path}")
        else:
            print(f"  轨道数据导出: 禁用")

        # 执行计算
        stats = compute_visibility(
            scenario_path=parsed_args.scenario,
            output_path=parsed_args.output,
            mode=parsed_args.mode,
            coarse_step=parsed_args.coarse_step,
            fine_step=parsed_args.fine_step,
            min_elevation=parsed_args.min_elevation,
            orbit_output_path=orbit_output_path
        )

        print_results(stats, parsed_args.mode)
        print(f"\n结果已保存: {parsed_args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"\n错误: 文件不存在 - {e}")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
