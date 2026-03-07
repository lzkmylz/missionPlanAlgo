#!/usr/bin/env python3
"""
大规模星座任务规划实验场景生成器

生成60颗卫星（30光学+30SAR）Walker 30/6/1星座
12个中国境内地面站
6个方向约1000个点目标

用法:
    python scripts/generate_large_scale_scenario.py
    python scripts/generate_large_scale_scenario.py --output scenarios/large_scale_experiment.json
"""

import json
import argparse
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict


# =============================================================================
# 配置常量
# =============================================================================

# Walker星座参数
WALKER_PARAMS = {
    'num_planes': 6,
    'sats_per_plane': 5,
    'altitude_m': 500000,  # 500km
    'inclination_deg': 55.0,
    'phase_factor': 1
}

# 地球半径
EARTH_RADIUS_M = 6371000.0

# 地面站配置
GROUND_STATIONS = [
    {'id': 'GS-BEIJING', 'name': '北京', 'lon': 116.4, 'lat': 39.9, 'alt': 50, 'min_elevation': 5.0},
    {'id': 'GS-SANYA', 'name': '三亚', 'lon': 109.5, 'lat': 18.3, 'alt': 10, 'min_elevation': 5.0},
    {'id': 'GS-KUNMING', 'name': '昆明', 'lon': 102.7, 'lat': 25.0, 'alt': 1900, 'min_elevation': 5.0},
    {'id': 'GS-KASHI', 'name': '喀什', 'lon': 76.0, 'lat': 39.5, 'alt': 1300, 'min_elevation': 5.0},
    {'id': 'GS-MUDANJIANG', 'name': '牡丹江', 'lon': 129.6, 'lat': 44.6, 'alt': 240, 'min_elevation': 5.0},
    {'id': 'GS-XIAN', 'name': '西安', 'lon': 108.9, 'lat': 34.3, 'alt': 400, 'min_elevation': 5.0},
    {'id': 'GS-LHASA', 'name': '拉萨', 'lon': 91.1, 'lat': 29.7, 'alt': 3650, 'min_elevation': 5.0},
    {'id': 'GS-URUMQI', 'name': '乌鲁木齐', 'lon': 87.6, 'lat': 43.8, 'alt': 800, 'min_elevation': 5.0},
    {'id': 'GS-SHANGHAI', 'name': '上海', 'lon': 121.5, 'lat': 31.2, 'alt': 10, 'min_elevation': 5.0},
    {'id': 'GS-GUANGZHOU', 'name': '广州', 'lon': 113.3, 'lat': 23.1, 'alt': 20, 'min_elevation': 5.0},
    {'id': 'GS-HARBIN', 'name': '哈尔滨', 'lon': 126.6, 'lat': 45.8, 'alt': 150, 'min_elevation': 5.0},
    {'id': 'GS-CHENGDU', 'name': '成都', 'lon': 104.1, 'lat': 30.7, 'alt': 500, 'min_elevation': 5.0},
]

# 6个方向的坐标范围
DIRECTION_BOUNDS = {
    'japan': {
        'lon_range': (129.0, 146.0),
        'lat_range': (30.0, 46.0),
        'count': 150,
        'priority_range': (2, 5)
    },
    'korea': {
        'lon_range': (124.0, 131.0),
        'lat_range': (33.0, 43.0),
        'count': 150,
        'priority_range': (2, 5)
    },
    'taiwan': {
        'lon_range': (119.0, 122.0),
        'lat_range': (21.0, 26.0),
        'count': 150,
        'priority_range': (2, 5)
    },
    'philippines': {
        'lon_range': (117.0, 127.0),
        'lat_range': (5.0, 20.0),
        'count': 150,
        'priority_range': (1, 4)
    },
    'myanmar': {
        'lon_range': (92.0, 102.0),
        'lat_range': (10.0, 29.0),
        'count': 200,
        'priority_range': (1, 3)
    },
    'india': {
        'lon_range': (77.0, 95.0),
        'lat_range': (20.0, 30.0),
        'count': 200,
        'priority_range': (1, 4)
    }
}


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class OrbitConfig:
    """轨道配置"""
    semi_major_axis: float  # 米
    eccentricity: float
    inclination: float  # 度
    raan: float  # 度
    arg_of_perigee: float  # 度
    mean_anomaly: float  # 度
    epoch: str  # ISO格式


@dataclass
class SatelliteConfig:
    """卫星配置"""
    id: str
    name: str
    sat_type: str  # 'optical' 或 'sar'
    orbit: Dict[str, Any]
    capabilities: Dict[str, Any]


@dataclass
class GroundStationConfig:
    """地面站配置"""
    id: str
    name: str
    location: List[float]  # [lon, lat, alt]
    min_elevation: float
    data_rate: float


@dataclass
class TargetConfig:
    """目标配置"""
    id: str
    name: str
    location: List[float]  # [lon, lat]
    priority: int
    direction: str


# =============================================================================
# 场景生成器
# =============================================================================

class LargeScaleScenarioGenerator:
    """大规模实验场景生成器"""

    def __init__(self, seed: int = 42, epoch: str = "2024-03-15T00:00:00Z"):
        self.seed = seed
        self.epoch = epoch
        np.random.seed(seed)

    def generate_walker_orbits(self, sat_type: str, phase_offset: int = 0) -> List[OrbitConfig]:
        """
        生成Walker 30/6/1星座轨道

        Args:
            sat_type: 卫星类型前缀
            phase_offset: 相位偏移（用于SAR与光学交错）

        Returns:
            30个卫星的轨道配置列表
        """
        params = WALKER_PARAMS
        semi_major_axis = EARTH_RADIUS_M + params['altitude_m']
        raan_spacing = 360.0 / params['num_planes']  # 60°
        mean_anomaly_spacing = 360.0 / params['sats_per_plane']  # 72°
        phase_increment = 360.0 / (params['num_planes'] * params['sats_per_plane'])  # 12°

        orbits = []
        sat_index = 0

        for plane in range(params['num_planes']):
            raan = plane * raan_spacing

            for sat_in_plane in range(params['sats_per_plane']):
                # 计算平近点角，加入相位偏移
                base_mean_anomaly = sat_in_plane * mean_anomaly_spacing
                phase_contribution = plane * phase_increment
                mean_anomaly = (base_mean_anomaly + phase_contribution + phase_offset) % 360.0

                orbit = OrbitConfig(
                    semi_major_axis=semi_major_axis,
                    eccentricity=0.0,
                    inclination=params['inclination_deg'],
                    raan=raan,
                    arg_of_perigee=0.0,
                    mean_anomaly=mean_anomaly,
                    epoch=self.epoch
                )
                orbits.append(orbit)
                sat_index += 1

        return orbits

    def generate_satellites(self) -> List[SatelliteConfig]:
        """生成60颗卫星配置（30光学 + 30SAR）"""
        satellites = []

        # 生成30颗光学卫星（相位偏移0）
        optical_orbits = self.generate_walker_orbits('optical', phase_offset=0)
        for i, orbit in enumerate(optical_orbits):
            sat_id = f"OPT-{i+1:02d}"
            satellite = SatelliteConfig(
                id=sat_id,
                name=f"光学卫星-{i+1:02d}",
                sat_type="optical",
                orbit={
                    'orbit_type': 'LEO',
                    'source': 'elements',
                    'semi_major_axis': orbit.semi_major_axis,
                    'eccentricity': orbit.eccentricity,
                    'inclination': orbit.inclination,
                    'raan': orbit.raan,
                    'arg_of_perigee': orbit.arg_of_perigee,
                    'mean_anomaly': orbit.mean_anomaly,
                    'epoch': orbit.epoch
                },
                capabilities={
                    'imaging_modes': ['push_broom'],
                    'max_off_nadir': 30.0,
                    'storage_capacity': 128.0,  # GB
                    'power_capacity': 2800.0,  # Wh (100Ah * 28V)
                    'data_rate': 300.0,  # Mbps
                    'min_solar_elevation': 15.0,  # 光学约束
                    'resolution': 1.0,  # 米
                    'swath_width': 15000.0,  # 米
                    'imager': {
                        'type': 'optical',
                        'spectral_bands': ['panchromatic', 'multispectral'],
                        'bit_depth': 12
                    }
                }
            )
            satellites.append(satellite)

        # 生成30颗SAR卫星（相位偏移36°，与光学交错）
        sar_orbits = self.generate_walker_orbits('sar', phase_offset=36)
        for i, orbit in enumerate(sar_orbits):
            sat_id = f"SAR-{i+1:02d}"
            satellite = SatelliteConfig(
                id=sat_id,
                name=f"SAR卫星-{i+1:02d}",
                sat_type="sar",
                orbit={
                    'orbit_type': 'LEO',
                    'source': 'elements',
                    'semi_major_axis': orbit.semi_major_axis,
                    'eccentricity': orbit.eccentricity,
                    'inclination': orbit.inclination,
                    'raan': orbit.raan,
                    'arg_of_perigee': orbit.arg_of_perigee,
                    'mean_anomaly': orbit.mean_anomaly,
                    'epoch': orbit.epoch
                },
                capabilities={
                    'imaging_modes': ['stripmap', 'spotlight', 'sliding_spotlight'],
                    'max_off_nadir': 45.0,
                    'storage_capacity': 128.0,  # GB
                    'power_capacity': 2800.0,  # Wh
                    'data_rate': 500.0,  # Mbps
                    'min_solar_elevation': None,  # SAR无光照约束
                    'resolution': 2.0,  # 米
                    'swath_width': 30000.0,  # 米
                    'imager': {
                        'type': 'sar',
                        'band': 'X',
                        'polarization': ['HH', 'HV', 'VH', 'VV']
                    }
                }
            )
            satellites.append(satellite)

        return satellites

    def generate_ground_stations(self) -> List[GroundStationConfig]:
        """生成12个地面站配置"""
        stations = []
        for gs in GROUND_STATIONS:
            station = GroundStationConfig(
                id=gs['id'],
                name=gs['name'],
                location=[gs['lon'], gs['lat'], gs['alt']],
                min_elevation=gs['min_elevation'],
                data_rate=500.0  # Mbps
            )
            stations.append(station)
        return stations

    def generate_targets(self) -> List[TargetConfig]:
        """生成6个方向约1000个点目标"""
        targets = []
        target_id = 0

        for direction, config in DIRECTION_BOUNDS.items():
            lon_min, lon_max = config['lon_range']
            lat_min, lat_max = config['lat_range']
            count = config['count']
            prio_min, prio_max = config['priority_range']

            for i in range(count):
                # 在区域内随机分布
                lon = np.random.uniform(lon_min, lon_max)
                lat = np.random.uniform(lat_min, lat_max)
                priority = np.random.randint(prio_min, prio_max + 1)

                target = TargetConfig(
                    id=f"TGT-{target_id:04d}",
                    name=f"{direction}_{i+1:03d}",
                    location=[round(lon, 4), round(lat, 4)],
                    priority=priority,
                    direction=direction
                )
                targets.append(target)
                target_id += 1

        return targets

    def generate_scenario(self) -> Dict[str, Any]:
        """生成完整场景配置"""
        satellites = self.generate_satellites()
        ground_stations = self.generate_ground_stations()
        targets = self.generate_targets()

        scenario = {
            'name': '大规模星座任务规划实验',
            'description': f'60颗卫星(30光学+30SAR) Walker 30/6/1星座，{len(targets)}个目标，24小时规划周期',
            'version': '1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'seed': self.seed,

            'duration': {
                'start': self.epoch,
                'end': '2024-03-16T00:00:00Z'  # 24小时
            },

            'satellites': [asdict(s) for s in satellites],
            'ground_stations': [asdict(gs) for gs in ground_stations],
            'targets': [asdict(t) for t in targets],

            'experiments': {
                'algorithms': [
                    {'name': 'FCFS', 'params': {}, 'description': '先到先服务（基准）'},
                    {'name': 'Greedy-EDF', 'params': {'heuristic': 'earliest_deadline'}, 'description': '最早截止时间优先'},
                    {'name': 'Greedy-MaxVal', 'params': {'heuristic': 'max_value'}, 'description': '最大价值优先'},
                    {'name': 'GA', 'params': {'population_size': 100, 'generations': 500, 'crossover_rate': 0.8, 'mutation_rate': 0.1}},
                    {'name': 'SA', 'params': {'initial_temp': 1000, 'cooling_rate': 0.995, 'iterations': 10000}}
                ],
                'metrics': [
                    'completion_rate',
                    'total_priority_score',
                    'avg_task_delay',
                    'resource_utilization',
                    'compute_time',
                    'energy_violation_count',
                    'storage_overflow_count'
                ]
            },

            'constraints': {
                'min_switch_time': 30,  # 秒，姿态机动时间
                'storage_capacity': 128,  # GB
                'battery_capacity': 100,  # Ah
                'nominal_voltage': 28,  # V
                'min_soc': 0.2,  # 最低电量20%
                'optical_min_solar_elevation': 15  # 度
            }
        }

        return scenario


# =============================================================================
# 辅助函数
# =============================================================================

def print_scenario_summary(scenario: Dict[str, Any]):
    """打印场景摘要"""
    print("\n" + "="*60)
    print("大规模星座任务规划实验场景")
    print("="*60)

    print(f"\n场景名称: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print(f"生成时间: {scenario['generated_at']}")
    print(f"随机种子: {scenario['seed']}")

    print(f"\n时间范围:")
    print(f"  开始: {scenario['duration']['start']}")
    print(f"  结束: {scenario['duration']['end']}")

    print(f"\n卫星配置:")
    optical_count = sum(1 for s in scenario['satellites'] if s['sat_type'] == 'optical')
    sar_count = sum(1 for s in scenario['satellites'] if s['sat_type'] == 'sar')
    print(f"  总计: {len(scenario['satellites'])}颗")
    print(f"    - 光学卫星: {optical_count}颗")
    print(f"    - SAR卫星: {sar_count}颗")

    print(f"\n地面站: {len(scenario['ground_stations'])}个")
    for gs in scenario['ground_stations']:
        print(f"  - {gs['name']}: ({gs['location'][0]:.2f}°E, {gs['location'][1]:.2f}°N)")

    print(f"\n目标分布: {len(scenario['targets'])}个")
    direction_counts = {}
    for t in scenario['targets']:
        direction_counts[t['direction']] = direction_counts.get(t['direction'], 0) + 1
    for direction, count in sorted(direction_counts.items()):
        print(f"  - {direction}: {count}个")

    print(f"\n对比算法:")
    for alg in scenario['experiments']['algorithms']:
        desc = alg.get('description', '')
        print(f"  - {alg['name']}: {desc}")

    print("\n约束条件:")
    constraints = scenario['constraints']
    print(f"  - 最小切换时间: {constraints['min_switch_time']}秒")
    print(f"  - 存储容量: {constraints['storage_capacity']}GB")
    print(f"  - 电池容量: {constraints['battery_capacity']}Ah")
    print(f"  - 光学最小太阳高度角: {constraints['optical_min_solar_elevation']}°")

    print("\n" + "="*60)


def save_scenario(scenario: Dict[str, Any], output_path: str):
    """保存场景到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)
    print(f"\n场景已保存到: {output_path}")


def generate_walker_summary(satellites: List[Dict]) -> str:
    """生成Walker星座分布摘要"""
    # 按轨道面分组
    planes = {}
    for sat in satellites:
        raan = sat['orbit']['raan']
        if raan not in planes:
            planes[raan] = []
        planes[raan].append({
            'id': sat['id'],
            'type': sat['sat_type'],
            'mean_anomaly': sat['orbit']['mean_anomaly']
        })

    summary = ["\nWalker 30/6/1 星座分布:"]
    for raan in sorted(planes.keys()):
        sats = sorted(planes[raan], key=lambda x: x['mean_anomaly'])
        optical = sum(1 for s in sats if s['type'] == 'optical')
        sar = sum(1 for s in sats if s['type'] == 'sar')
        summary.append(f"  轨道面 RAAN={raan:5.1f}°: {len(sats)}颗 (光学{optical}, SAR{sar})")

    return "\n".join(summary)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='生成大规模星座任务规划实验场景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_large_scale_scenario.py
  python scripts/generate_large_scale_scenario.py --output my_scenario.json --seed 123
  python scripts/generate_large_scale_scenario.py --epoch "2024-06-01T00:00:00Z"
        """
    )
    parser.add_argument(
        '--output', '-o',
        default='scenarios/large_scale_experiment.json',
        help='输出文件路径 (默认: scenarios/large_scale_experiment.json)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--epoch',
        default='2024-03-15T00:00:00Z',
        help='场景起始时间，ISO 8601格式 (默认: 2024-03-15T00:00:00Z)'
    )

    args = parser.parse_args()

    print("="*60)
    print("大规模星座任务规划实验场景生成器")
    print("="*60)

    # 生成场景
    print(f"\n生成参数:")
    print(f"  随机种子: {args.seed}")
    print(f"  历元时间: {args.epoch}")
    print(f"  输出路径: {args.output}")

    generator = LargeScaleScenarioGenerator(seed=args.seed, epoch=args.epoch)
    scenario = generator.generate_scenario()

    # 打印摘要
    print_scenario_summary(scenario)

    # 打印Walker分布
    print(generate_walker_summary(scenario['satellites']))

    # 保存文件
    save_scenario(scenario, args.output)

    print("\n生成完成!")


if __name__ == '__main__':
    main()
