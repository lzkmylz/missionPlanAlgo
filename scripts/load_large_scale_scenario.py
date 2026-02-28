#!/usr/bin/env python3
"""
大规模场景加载验证脚本

验证生成的场景文件可以正确加载到系统模型中

用法:
    python scripts/load_large_scale_scenario.py
    python scripts/load_large_scale_scenario.py --scenario scenarios/large_scale_experiment.json
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.satellite import Satellite, SatelliteType
from core.models.target import Target
from core.models.ground_station import GroundStation


def load_satellites(scenario_data: Dict) -> List[Satellite]:
    """从场景数据加载卫星模型"""
    satellites = []
    for sat_data in scenario_data['satellites']:
        # 根据sat_type映射到SatelliteType
        sat_type_str = sat_data['sat_type']
        if sat_type_str == 'optical':
            sat_type = SatelliteType.OPTICAL_1
        elif sat_type_str == 'sar':
            sat_type = SatelliteType.SAR_1
        else:
            sat_type = SatelliteType.OPTICAL_1

        # 更新sat_type字段以匹配模型期望
        sat_data_copy = sat_data.copy()
        sat_data_copy['sat_type'] = sat_type.value

        try:
            sat = Satellite.from_dict(sat_data_copy)
            satellites.append(sat)
        except Exception as e:
            print(f"警告: 加载卫星 {sat_data['id']} 失败: {e}")
            raise

    return satellites


def load_targets(scenario_data: Dict) -> List[Target]:
    """从场景数据加载目标模型（支持频次约束）"""
    targets = []
    for tgt_data in scenario_data['targets']:
        # 获取频次约束（如果存在）
        required_obs = tgt_data.get('required_observations', 1)
        revisit_interval = tgt_data.get('min_revisit_interval', 0.0)

        # 创建目标对象
        target = Target(
            id=tgt_data['id'],
            name=tgt_data['name'],
            latitude=tgt_data['location'][1],
            longitude=tgt_data['location'][0],
            priority=tgt_data['priority'],
            required_observations=required_obs
        )

        # 动态添加重访间隔属性（用于频次约束场景）
        if revisit_interval > 0:
            target.min_revisit_interval = revisit_interval

        targets.append(target)
    return targets


def load_ground_stations(scenario_data: Dict) -> List[GroundStation]:
    """从场景数据加载地面站模型"""
    from core.models.ground_station import Antenna

    stations = []
    for gs_data in scenario_data['ground_stations']:
        station = GroundStation(
            id=gs_data['id'],
            name=gs_data['name'],
            longitude=gs_data['location'][0],
            latitude=gs_data['location'][1],
            altitude=gs_data['location'][2]
        )
        # 添加默认天线
        antenna = Antenna(
            id=f"{gs_data['id']}-ANT01",
            name=f"{gs_data['name']}主天线",
            elevation_min=gs_data.get('min_elevation', 5.0),
            elevation_max=90.0,
            data_rate=gs_data.get('data_rate', 500.0)
        )
        station.add_antenna(antenna)
        stations.append(station)
    return stations


def validate_scenario(scenario_path: str):
    """验证场景文件"""
    import json

    print(f"正在加载场景文件: {scenario_path}")
    print("-" * 60)

    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    print(f"场景名称: {scenario_data['name']}")
    print(f"描述: {scenario_data['description']}")
    print(f"版本: {scenario_data['version']}")
    print()

    # 加载卫星
    print("正在加载卫星...")
    satellites = load_satellites(scenario_data)
    print(f"  ✓ 成功加载 {len(satellites)} 颗卫星")

    optical_sats = [s for s in satellites if 'OPT' in s.id]
    sar_sats = [s for s in satellites if 'SAR' in s.id]
    print(f"    - 光学卫星: {len(optical_sats)} 颗")
    print(f"    - SAR卫星: {len(sar_sats)} 颗")

    # 验证第一颗卫星的轨道
    sample_sat = satellites[0]
    print(f"\n  示例卫星: {sample_sat.name}")
    print(f"    - 类型: {sample_sat.sat_type.value}")
    print(f"    - 半长轴: {sample_sat.orbit.semi_major_axis / 1000:.1f} km")
    print(f"    - 轨道倾角: {sample_sat.orbit.inclination}°")
    print(f"    - 升交点赤经: {sample_sat.orbit.raan}°")
    print(f"    - 历元: {sample_sat.orbit.epoch}")
    print(f"    - 存储容量: {sample_sat.capabilities.storage_capacity} GB")
    print(f"    - 电源容量: {sample_sat.capabilities.power_capacity} Wh")

    # 检查光照约束
    optical_constraints = [
        s.capabilities.imager.get('min_solar_elevation')
        for s in optical_sats
        if hasattr(s.capabilities, 'imager') and s.capabilities.imager
    ]
    if optical_constraints:
        print(f"    - 光学约束(最小太阳高度角): {optical_constraints[0]}°")

    # 加载目标
    print("\n正在加载目标...")
    targets = load_targets(scenario_data)
    print(f"  ✓ 成功加载 {len(targets)} 个目标")

    # 按方向统计（从原始数据获取）
    from collections import Counter
    direction_counts = Counter(t['direction'] for t in scenario_data['targets'])
    for direction, count in sorted(direction_counts.items()):
        print(f"    - {direction}: {count} 个")

    # 显示示例目标
    sample_target = targets[0]
    print(f"\n  示例目标: {sample_target.name}")
    print(f"    - 位置: ({sample_target.longitude:.4f}°E, {sample_target.latitude:.4f}°N)")
    print(f"    - 优先级: {sample_target.priority}")

    # 加载地面站
    print("\n正在加载地面站...")
    stations = load_ground_stations(scenario_data)
    print(f"  ✓ 成功加载 {len(stations)} 个地面站")
    for station in stations:
        print(f"    - {station.name}: ({station.longitude:.2f}°E, {station.latitude:.2f}°N)")

    # 验证约束配置
    print("\n约束配置:")
    constraints = scenario_data.get('constraints', {})
    for key, value in constraints.items():
        print(f"  - {key}: {value}")

    # 验证实验配置
    print("\n实验配置:")
    experiments = scenario_data.get('experiments', {})
    algorithms = experiments.get('algorithms', [])
    print(f"  - 对比算法: {len(algorithms)} 种")
    for alg in algorithms:
        print(f"    - {alg['name']}: {alg.get('description', 'N/A')}")

    metrics = experiments.get('metrics', [])
    print(f"  - 评价指标: {len(metrics)} 个")

    print("\n" + "=" * 60)
    print("场景验证完成！所有模型加载成功。")
    print("=" * 60)

    return {
        'satellites': satellites,
        'targets': targets,
        'ground_stations': stations,
        'scenario_data': scenario_data
    }


def main():
    parser = argparse.ArgumentParser(
        description='加载并验证大规模实验场景',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--scenario', '-s',
        default='scenarios/large_scale_experiment.json',
        help='场景文件路径 (默认: scenarios/large_scale_experiment.json)'
    )

    args = parser.parse_args()

    try:
        result = validate_scenario(args.scenario)
        return 0
    except FileNotFoundError:
        print(f"错误: 场景文件不存在: {args.scenario}")
        print("请先运行: python scripts/generate_large_scale_scenario.py")
        return 1
    except Exception as e:
        print(f"错误: 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
