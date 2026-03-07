#!/usr/bin/env python3
"""
大规模星座任务规划实验场景生成器（含频次约束版本）

生成60颗卫星（30光学+30SAR）Walker 30/6/1星座
12个中国境内地面站
6个方向约1000个点目标，支持多次观测需求和重访周期约束

观测频次配置：
- 高优先级目标（优先级5）: 5次观测，最小间隔4小时
- 中高优先级目标（优先级4）: 3次观测，最小间隔6小时
- 中等优先级目标（优先级3）: 2次观测，最小间隔8小时
- 低优先级目标（优先级1-2）: 1次观测，无间隔要求

用法:
    python scripts/generate_scenario_with_frequency.py
    python scripts/generate_scenario_with_frequency.py --output scenarios/large_scale_frequency.json
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

# 6个方向的坐标范围和频次配置
DIRECTION_BOUNDS = {
    'japan': {
        'lon_range': (129.0, 146.0),
        'lat_range': (30.0, 46.0),
        'count': 150,
        'priority_range': (2, 5),
        'obs_freq_range': (2, 5),  # 观测频次范围
        'revisit_hours': (4, 6)    # 重访间隔（小时）
    },
    'korea': {
        'lon_range': (124.0, 131.0),
        'lat_range': (33.0, 43.0),
        'count': 150,
        'priority_range': (2, 5),
        'obs_freq_range': (2, 5),
        'revisit_hours': (4, 6)
    },
    'taiwan': {
        'lon_range': (119.0, 122.0),
        'lat_range': (21.0, 26.0),
        'count': 150,
        'priority_range': (2, 5),
        'obs_freq_range': (3, 5),
        'revisit_hours': (3, 5)
    },
    'philippines': {
        'lon_range': (117.0, 127.0),
        'lat_range': (5.0, 20.0),
        'count': 150,
        'priority_range': (1, 4),
        'obs_freq_range': (1, 3),
        'revisit_hours': (6, 8)
    },
    'myanmar': {
        'lon_range': (92.0, 102.0),
        'lat_range': (10.0, 29.0),
        'count': 200,
        'priority_range': (1, 3),
        'obs_freq_range': (1, 2),
        'revisit_hours': (8, 12)
    },
    'india': {
        'lon_range': (77.0, 95.0),
        'lat_range': (20.0, 30.0),
        'count': 200,
        'priority_range': (1, 4),
        'obs_freq_range': (1, 3),
        'revisit_hours': (6, 8)
    }
}


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class OrbitConfig:
    """轨道配置"""
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_of_perigee: float
    mean_anomaly: float
    epoch: str


@dataclass
class SatelliteConfig:
    """卫星配置"""
    id: str
    name: str
    sat_type: str
    orbit: Dict[str, Any]
    capabilities: Dict[str, Any]


@dataclass
class GroundStationConfig:
    """地面站配置"""
    id: str
    name: str
    location: List[float]
    min_elevation: float
    data_rate: float


@dataclass
class TargetConfig:
    """目标配置（含频次约束）"""
    id: str
    name: str
    location: List[float]
    priority: int
    direction: str
    required_observations: int = 1  # 需要观测次数
    min_revisit_interval: float = 0.0  # 最小重访间隔（小时）


# =============================================================================
# 场景生成器
# =============================================================================

class FrequencyScenarioGenerator:
    """带频次约束的大规模实验场景生成器"""

    def __init__(self, seed: int = 42, epoch: str = "2024-03-15T00:00:00Z"):
        self.seed = seed
        self.epoch = epoch
        np.random.seed(seed)

    def generate_walker_orbits(self, sat_type: str, phase_offset: int = 0) -> List[OrbitConfig]:
        """生成Walker 30/6/1星座轨道"""
        params = WALKER_PARAMS
        semi_major_axis = EARTH_RADIUS_M + params['altitude_m']
        raan_spacing = 360.0 / params['num_planes']
        mean_anomaly_spacing = 360.0 / params['sats_per_plane']
        phase_increment = 360.0 / (params['num_planes'] * params['sats_per_plane'])

        orbits = []
        for plane in range(params['num_planes']):
            raan = plane * raan_spacing
            for sat in range(params['sats_per_plane']):
                mean_anomaly = (sat * mean_anomaly_spacing + plane * phase_increment + phase_offset) % 360.0
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
        return orbits

    def generate_satellites(self) -> List[SatelliteConfig]:
        """生成60颗卫星配置（30光学 + 30SAR）"""
        satellites = []

        # 生成30颗光学卫星
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
                    'storage_capacity': 128.0,
                    'power_capacity': 2800.0,
                    'data_rate': 300.0,
                    'min_solar_elevation': 15.0,
                    'resolution': 1.0,
                    'swath_width': 15000.0,
                    'imager': {
                        'type': 'optical',
                        'spectral_bands': ['panchromatic', 'multispectral'],
                        'bit_depth': 12
                    }
                }
            )
            satellites.append(satellite)

        # 生成30颗SAR卫星
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
                    'storage_capacity': 128.0,
                    'power_capacity': 2800.0,
                    'data_rate': 500.0,
                    'min_solar_elevation': None,
                    'resolution': 2.0,
                    'swath_width': 30000.0,
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
                data_rate=500.0
            )
            stations.append(station)
        return stations

    def generate_targets(self) -> List[TargetConfig]:
        """生成6个方向约1000个点目标（含频次约束）"""
        targets = []
        target_id = 0

        for direction, config in DIRECTION_BOUNDS.items():
            lon_min, lon_max = config['lon_range']
            lat_min, lat_max = config['lat_range']
            count = config['count']
            prio_min, prio_max = config['priority_range']
            freq_min, freq_max = config.get('obs_freq_range', (1, 1))
            revisit_min, revisit_max = config.get('revisit_hours', (0, 0))

            for i in range(count):
                # 在区域内随机分布
                lon = np.random.uniform(lon_min, lon_max)
                lat = np.random.uniform(lat_min, lat_max)
                priority = np.random.randint(prio_min, prio_max + 1)

                # 根据优先级确定观测频次和重访间隔
                required_obs = np.random.randint(freq_min, freq_max + 1)
                revisit_interval = np.random.uniform(revisit_min, revisit_max) if required_obs > 1 else 0.0

                target = TargetConfig(
                    id=f"TGT-{target_id:04d}",
                    name=f"{direction}_{i+1:03d}",
                    location=[round(lon, 4), round(lat, 4)],
                    priority=priority,
                    direction=direction,
                    required_observations=required_obs,
                    min_revisit_interval=round(revisit_interval, 1)
                )
                targets.append(target)
                target_id += 1

        return targets

    def generate_scenario(self) -> Dict[str, Any]:
        """生成完整场景配置"""
        satellites = self.generate_satellites()
        ground_stations = self.generate_ground_stations()
        targets = self.generate_targets()

        # 计算总观测需求
        total_obs_demand = sum(t.required_observations for t in targets)

        scenario = {
            'name': '大规模星座任务规划实验（含频次约束）',
            'description': f'60颗卫星(30光学+30SAR) vs {len(targets)}目标，总观测需求{total_obs_demand}次，24小时规划周期',
            'version': '2.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'seed': self.seed,

            'duration': {
                'start': self.epoch,
                'end': '2024-03-16T00:00:00Z'
            },

            'satellites': [asdict(s) for s in satellites],
            'ground_stations': [asdict(gs) for gs in ground_stations],
            'targets': [asdict(t) for t in targets],

            'statistics': {
                'total_targets': len(targets),
                'total_observation_demand': total_obs_demand,
                'avg_obs_per_target': total_obs_demand / len(targets),
                'targets_by_frequency': self._count_by_frequency(targets)
            },

            'experiments': {
                'algorithms': [
                    {'name': 'FCFS', 'type': 'baseline'},
                    {'name': 'Greedy-EDF', 'type': 'heuristic'},
                    {'name': 'Greedy-MaxVal', 'type': 'heuristic'},
                    {'name': 'GA', 'type': 'metaheuristic', 'population': 100, 'generations': 500},
                    {'name': 'SA', 'type': 'metaheuristic', 'initial_temp': 1000, 'cooling_rate': 0.995}
                ],
                'metrics': [
                    'demand_satisfaction_rate',
                    'observation_completion_rate',
                    'frequency_satisfaction_rate',
                    'avg_revisit_deviation',
                    'computation_time'
                ]
            },

            'constraints': {
                'min_switch_time': 30,
                'storage_capacity': 128,
                'battery_capacity': 100,
                'nominal_voltage': 28,
                'min_soc': 0.2,
                'optical_min_solar_elevation': 15
            }
        }

        return scenario

    def _count_by_frequency(self, targets: List[TargetConfig]) -> Dict[int, int]:
        """按观测频次统计目标数量"""
        counts = {}
        for t in targets:
            counts[t.required_observations] = counts.get(t.required_observations, 0) + 1
        return dict(sorted(counts.items()))


# =============================================================================
# 辅助函数
# =============================================================================

def print_scenario_summary(scenario: Dict[str, Any]):
    """打印场景摘要"""
    print("\n" + "=" * 70)
    print("大规模星座任务规划实验场景（含频次约束）")
    print("=" * 70)

    print(f"\n场景名称: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print(f"生成时间: {scenario['generated_at']}")
    print(f"随机种子: {scenario['seed']}")

    print(f"\n时间范围:")
    print(f"  开始: {scenario['duration']['start']}")
    print(f"  结束: {scenario['duration']['end']}")

    stats = scenario['statistics']
    print(f"\n目标统计:")
    print(f"  总目标数: {stats['total_targets']}")
    print(f"  总观测需求: {stats['total_observation_demand']} 次")
    print(f"  平均每目标: {stats['avg_obs_per_target']:.1f} 次")

    print(f"\n  按频次分布:")
    for freq, count in stats['targets_by_frequency'].items():
        bar = '█' * int(count / 20)
        print(f"    {freq}次观测: {count:4d} 个目标 {bar}")

    print("\n" + "=" * 70)


def save_scenario(scenario: Dict[str, Any], output_path: str):
    """保存场景到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)
    print(f"\n场景已保存到: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='生成带频次约束的大规模实验场景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_scenario_with_frequency.py
  python scripts/generate_scenario_with_frequency.py --output my_scenario.json --seed 123
        """
    )
    parser.add_argument(
        '--output', '-o',
        default='scenarios/large_scale_frequency.json',
        help='输出文件路径 (默认: scenarios/large_scale_frequency.json)'
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
        help='场景起始时间'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("大规模星座任务规划场景生成器（含频次约束）")
    print("=" * 70)

    print(f"\n生成参数:")
    print(f"  随机种子: {args.seed}")
    print(f"  历元时间: {args.epoch}")
    print(f"  输出路径: {args.output}")

    generator = FrequencyScenarioGenerator(seed=args.seed, epoch=args.epoch)
    scenario = generator.generate_scenario()

    print_scenario_summary(scenario)
    save_scenario(scenario, args.output)

    print("\n生成完成!")


if __name__ == '__main__':
    main()
