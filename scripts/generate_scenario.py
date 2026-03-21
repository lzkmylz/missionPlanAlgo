#!/usr/bin/env python3
"""
统一场景生成脚本 - 合并了以下脚本的功能:
- generate_large_scale_scenario.py
- generate_scenario_with_frequency.py

用法:
    # 生成基础场景
    python scripts/generate_scenario.py

    # 生成带频次约束的场景
    python scripts/generate_scenario.py --frequency

    # 指定随机种子
    python scripts/generate_scenario.py --seed 123

    # 自定义输出路径
    python scripts/generate_scenario.py -o scenarios/my_scenario.json
"""

import argparse
import json
import sys
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import setup_logging, save_results


# ============================================================================
# 配置常量
# ============================================================================

# Walker星座参数: 5轨道面, 每面6颗卫星, 72度RAAN, 120度相位间隔
WALKER_PARAMS = {
    'num_planes': 5,           # 5个轨道面
    'sats_per_plane': 6,       # 每面6颗卫星 (30颗光学 + 30颗SAR)
    'altitude_m': 500000,      # 500km
    'inclination_deg': 55.0,   # 55度倾角
    'raan_spacing': 72.0,      # RAAN间隔72度
    'phase_spacing': 120.0     # 相位间隔120度
}

EARTH_RADIUS_M = 6371000.0

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

DIRECTION_BOUNDS = {
    'japan': {'lon_range': (129.0, 146.0), 'lat_range': (30.0, 46.0), 'count': 150, 'priority_range': (2, 5)},
    'korea': {'lon_range': (124.0, 131.0), 'lat_range': (33.0, 43.0), 'count': 150, 'priority_range': (2, 5)},
    'taiwan': {'lon_range': (119.0, 122.0), 'lat_range': (21.0, 26.0), 'count': 150, 'priority_range': (2, 5)},
    'philippines': {'lon_range': (117.0, 127.0), 'lat_range': (5.0, 20.0), 'count': 150, 'priority_range': (1, 4)},
    'myanmar': {'lon_range': (92.0, 102.0), 'lat_range': (10.0, 29.0), 'count': 200, 'priority_range': (1, 3)},
    'india': {'lon_range': (77.0, 95.0), 'lat_range': (20.0, 30.0), 'count': 200, 'priority_range': (1, 4)},
}

DIRECTION_BOUNDS_WITH_FREQ = {
    'japan': {'lon_range': (129.0, 146.0), 'lat_range': (30.0, 46.0), 'count': 150, 'priority_range': (2, 5), 'obs_freq_range': (2, 5), 'revisit_hours': (4, 6)},
    'korea': {'lon_range': (124.0, 131.0), 'lat_range': (33.0, 43.0), 'count': 150, 'priority_range': (2, 5), 'obs_freq_range': (2, 5), 'revisit_hours': (4, 6)},
    'taiwan': {'lon_range': (119.0, 122.0), 'lat_range': (21.0, 26.0), 'count': 150, 'priority_range': (2, 5), 'obs_freq_range': (3, 5), 'revisit_hours': (3, 5)},
    'philippines': {'lon_range': (117.0, 127.0), 'lat_range': (5.0, 20.0), 'count': 150, 'priority_range': (1, 4), 'obs_freq_range': (1, 3), 'revisit_hours': (6, 8)},
    'myanmar': {'lon_range': (92.0, 102.0), 'lat_range': (10.0, 29.0), 'count': 200, 'priority_range': (1, 3), 'obs_freq_range': (1, 2), 'revisit_hours': (8, 12)},
    'india': {'lon_range': (77.0, 95.0), 'lat_range': (20.0, 30.0), 'count': 200, 'priority_range': (1, 4), 'obs_freq_range': (1, 3), 'revisit_hours': (6, 8)},
}


# ============================================================================
# 数据类定义
# ============================================================================

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
    """目标配置"""
    id: str
    name: str
    location: List[float]
    priority: int
    direction: str
    required_observations: int = 1
    min_revisit_interval: float = 0.0
    # 精准需求（可选，空列表表示不限制）
    allowed_satellite_types: List[str] = field(default_factory=list)
    allowed_satellite_ids: List[str] = field(default_factory=list)
    required_imaging_modes: List[str] = field(default_factory=list)


# ============================================================================
# 场景生成器
# ============================================================================

class ScenarioGenerator:
    """场景生成器"""

    def __init__(self, seed: int = 42, epoch: str = "2024-03-15T00:00:00Z"):
        self.seed = seed
        self.epoch = epoch
        np.random.seed(seed)

    def generate_walker_orbits(self, sat_type: str, raan_offset: float = 0) -> List[OrbitConfig]:
        """生成Walker星座轨道: 5面/6星, 72度RAAN间隔, 120度相位间隔

        Args:
            sat_type: 卫星类型 ('optical' 或 'sar')
            raan_offset: RAAN偏移量 (光学0度, SAR 36度实现轨道面错开)
        """
        params = WALKER_PARAMS
        semi_major_axis = EARTH_RADIUS_M + params['altitude_m']

        orbits = []
        for plane in range(params['num_planes']):
            # RAAN: 0, 72, 144, 216, 288 (光学) 或偏移后 (SAR)
            raan = plane * params['raan_spacing'] + raan_offset

            for sat in range(params['sats_per_plane']):
                # 平面内卫星间隔60度 (360/6)
                # 相邻平面相位差120度
                mean_anomaly = (sat * 60.0 + plane * params['phase_spacing']) % 360.0

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

        # 生成30颗光学卫星 (RAAN 0, 72, 144, 216, 288)
        optical_orbits = self.generate_walker_orbits('optical', raan_offset=0.0)
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
                    'max_roll_angle': 60.0,
                    'max_pitch_angle': 60.0,
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

        # 生成30颗SAR卫星 (轨道面错开36度, 在光学轨道面中间)
        sar_orbits = self.generate_walker_orbits('sar', raan_offset=36.0)
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
                    'max_roll_angle': 60.0,
                    'max_pitch_angle': 60.0,
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

    # 精准需求预设方案（类型约束 / ID 约束 / 模式约束），用于随机分配。
    # 注意：所有模式约束必须与 generate_satellites() 中配置的卫星实际成像模式一致：
    #   光学卫星：['push_broom']
    #   SAR 卫星：['spotlight', 'stripmap', 'sliding_spotlight']
    # PMC 模式（forward_pushbroom_pmc）不在预设中，因生成的光学卫星不具备该能力。
    # 如需测试 PMC 约束，请在 generate_satellites() 中为光学卫星添加 PMC 模式后再添加对应预设。
    _PRECISE_PRESETS = [
        # 仅限光学卫星（任意模式）
        {'allowed_satellite_ids': [], 'allowed_satellite_types': ['optical'], 'required_imaging_modes': []},
        # 仅限光学卫星推扫模式
        {'allowed_satellite_ids': [], 'allowed_satellite_types': ['optical'], 'required_imaging_modes': ['push_broom']},
        # 仅限 SAR 卫星（任意模式）
        {'allowed_satellite_ids': [], 'allowed_satellite_types': ['sar'], 'required_imaging_modes': []},
        # 仅限 SAR 聚束模式
        {'allowed_satellite_ids': [], 'allowed_satellite_types': ['sar'], 'required_imaging_modes': ['spotlight']},
        # 仅限 SAR 条带模式
        {'allowed_satellite_ids': [], 'allowed_satellite_types': ['sar'], 'required_imaging_modes': ['stripmap']},
        # 仅限 SAR 滑动聚束模式
        {'allowed_satellite_ids': [], 'allowed_satellite_types': ['sar'], 'required_imaging_modes': ['sliding_spotlight']},
        # 指定 SAR 卫星 ID 约束（测试 allowed_satellite_ids 路径；2颗卫星可覆盖性低，主要用于约束路径验证）
        {'allowed_satellite_ids': ['SAR-01', 'SAR-02'], 'allowed_satellite_types': [], 'required_imaging_modes': []},
        # 指定光学卫星 ID 约束（同上）
        {'allowed_satellite_ids': ['OPT-01', 'OPT-02'], 'allowed_satellite_types': [], 'required_imaging_modes': []},
    ]

    def generate_targets(self, enable_frequency: bool = False,
                         precise_ratio: float = 0.0) -> List[TargetConfig]:
        """生成目标配置

        Args:
            enable_frequency: 是否启用频次约束
            precise_ratio: 精准需求目标占比（0.0-1.0），0 表示全部使用模糊需求
        """
        precise_ratio = max(0.0, min(1.0, precise_ratio))  # 防御性钳位
        targets = []
        target_id = 0

        direction_bounds = DIRECTION_BOUNDS_WITH_FREQ if enable_frequency else DIRECTION_BOUNDS

        for direction, config in direction_bounds.items():
            lon_min, lon_max = config['lon_range']
            lat_min, lat_max = config['lat_range']
            count = config['count']
            prio_min, prio_max = config['priority_range']

            for i in range(count):
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

                # 添加频次相关属性
                if enable_frequency:
                    freq_min, freq_max = config.get('obs_freq_range', (1, 1))
                    revisit_min, revisit_max = config.get('revisit_hours', (0, 0))
                    target.required_observations = np.random.randint(freq_min, freq_max + 1)
                    target.min_revisit_interval = np.random.uniform(revisit_min, revisit_max) if target.required_observations > 1 else 0.0

                # 按比例随机分配精准需求
                if precise_ratio > 0.0 and np.random.random() < precise_ratio:
                    preset = self._PRECISE_PRESETS[
                        np.random.randint(len(self._PRECISE_PRESETS))
                    ]
                    target.allowed_satellite_ids = list(preset['allowed_satellite_ids'])
                    target.allowed_satellite_types = list(preset['allowed_satellite_types'])
                    target.required_imaging_modes = list(preset['required_imaging_modes'])

                targets.append(target)
                target_id += 1

        return targets

    def generate_scenario(self, enable_frequency: bool = False,
                          precise_ratio: float = 0.0) -> Dict[str, Any]:
        """生成完整场景配置

        Args:
            enable_frequency: 是否启用频次约束
            precise_ratio: 精准需求目标占比（0.0-1.0）
        """
        satellites = self.generate_satellites()
        ground_stations = self.generate_ground_stations()
        targets = self.generate_targets(enable_frequency=enable_frequency,
                                        precise_ratio=precise_ratio)

        total_obs_demand = sum(t.required_observations for t in targets) if enable_frequency else len(targets)

        name = '大规模星座任务规划实验场景'
        if enable_frequency:
            name += '（含频次约束）'
        if precise_ratio > 0.0:
            name += f'（含精准需求 {precise_ratio:.0%}）'

        description = f'60颗卫星(30光学+30SAR) vs {len(targets)}目标'
        if enable_frequency:
            description += f'，总观测需求{total_obs_demand}次'
        description += '，24小时规划周期'

        scenario = {
            'name': name,
            'description': description,
            'version': '2.0' if enable_frequency else '1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'seed': self.seed,
            'duration': {
                'start': self.epoch,
                'end': '2024-03-16T00:00:00Z'
            },
            'satellites': [asdict(s) for s in satellites],
            'ground_stations': [asdict(gs) for gs in ground_stations],
            'targets': [asdict(t) for t in targets],
        }

        if enable_frequency:
            scenario['statistics'] = {
                'total_targets': len(targets),
                'total_observation_demand': total_obs_demand,
                'avg_obs_per_target': total_obs_demand / len(targets),
                'targets_by_frequency': self._count_by_frequency(targets)
            }

        if precise_ratio > 0.0:
            precise_count = sum(
                1 for t in targets if t.allowed_satellite_types or t.allowed_satellite_ids or t.required_imaging_modes
            )
            scenario.setdefault('statistics', {}).update({
                'precise_requirement_targets': precise_count,
                'precise_requirement_ratio': precise_count / len(targets) if targets else 0.0,
            })

        return scenario

    def _count_by_frequency(self, targets: List[TargetConfig]) -> Dict[int, int]:
        """按观测频次统计目标数量"""
        counts = {}
        for t in targets:
            counts[t.required_observations] = counts.get(t.required_observations, 0) + 1
        return dict(sorted(counts.items()))


# ============================================================================
# 辅助函数
# ============================================================================

def _precise_ratio_type(value: str) -> float:
    """argparse 类型函数：将字符串转换为 [0.0, 1.0] 范围内的浮点数。"""
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"precise-ratio 必须为浮点数，收到: {value!r}")
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"precise-ratio 必须在 [0.0, 1.0] 范围内，收到: {v}")
    return v


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='统一场景生成脚本 - 支持基础场景和带频次约束的场景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成基础场景
  python scripts/generate_scenario.py

  # 生成带频次约束的场景
  python scripts/generate_scenario.py --frequency

  # 生成 20% 目标带精准需求的场景（用于测试精准约束功能）
  python scripts/generate_scenario.py --precise-ratio 0.2

  # 组合使用
  python scripts/generate_scenario.py --frequency --precise-ratio 0.1 --seed 123

  # 自定义输出路径
  python scripts/generate_scenario.py -o scenarios/my_scenario.json
        """
    )

    parser.add_argument(
        '-o', '--output',
        default='scenarios/generated_scenario.json',
        help='输出文件路径 (默认: scenarios/generated_scenario.json)'
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
    parser.add_argument(
        '--frequency',
        action='store_true',
        help='启用观测频次约束'
    )
    parser.add_argument(
        '--precise-ratio',
        type=_precise_ratio_type,
        default=0.0,
        metavar='RATIO',
        help='精准需求目标占比（0.0-1.0），随机为该比例目标分配卫星类型/成像模式约束，默认 0（禁用）'
    )

    return parser.parse_args(args)


def print_scenario_summary(scenario: Dict[str, Any]) -> None:
    """打印场景摘要"""
    print("\n" + "="*70)
    print("场景生成完成")
    print("="*70)

    print(f"\n场景名称: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print(f"生成时间: {scenario['generated_at']}")
    print(f"随机种子: {scenario['seed']}")

    print(f"\n时间范围:")
    print(f"  开始: {scenario['duration']['start']}")
    print(f"  结束: {scenario['duration']['end']}")

    print(f"\n实体统计:")
    print(f"  卫星: {len(scenario['satellites'])} 颗")
    print(f"  地面站: {len(scenario['ground_stations'])} 个")
    print(f"  目标: {len(scenario['targets'])} 个")

    if 'statistics' in scenario:
        stats = scenario['statistics']
        print(f"\n  总观测需求: {stats.get('total_observation_demand', 'N/A')} 次")
        print(f"  平均每目标: {stats.get('avg_obs_per_target', 'N/A'):.1f} 次")

        freq_dist = stats.get('targets_by_frequency', {})
        if freq_dist:
            print(f"\n  频次分布:")
            for freq, count in freq_dist.items():
                bar = '█' * int(count / 20)
                print(f"    {freq}次观测: {count:4d} 个目标 {bar}")

    print("\n" + "="*70)


def generate_scenario(
    output_path: str,
    seed: int = 42,
    epoch: str = "2024-03-15T00:00:00Z",
    enable_frequency: bool = False,
    precise_ratio: float = 0.0
) -> Dict[str, Any]:
    """
    生成场景并保存

    Args:
        output_path: 输出文件路径
        seed: 随机种子
        epoch: 场景起始时间
        enable_frequency: 是否启用频次约束
        precise_ratio: 精准需求目标占比（0.0-1.0），0 表示全部使用模糊需求

    Returns:
        生成的场景配置
    """
    generator = ScenarioGenerator(seed=seed, epoch=epoch)
    scenario = generator.generate_scenario(enable_frequency=enable_frequency,
                                           precise_ratio=precise_ratio)

    save_results(scenario, output_path)

    return scenario


def main(args: Optional[List[str]] = None) -> int:
    """主函数"""
    parsed_args = parse_args(args)

    setup_logging()

    print("="*70)
    print("统一场景生成脚本")
    print("="*70)

    print(f"\n生成参数:")
    print(f"  随机种子: {parsed_args.seed}")
    print(f"  历元时间: {parsed_args.epoch}")
    print(f"  频次约束: {'启用' if parsed_args.frequency else '禁用'}")
    print(f"  精准需求: {parsed_args.precise_ratio:.0%} 目标" if parsed_args.precise_ratio > 0 else "  精准需求: 禁用")
    print(f"  输出路径: {parsed_args.output}")

    try:
        scenario = generate_scenario(
            output_path=parsed_args.output,
            seed=parsed_args.seed,
            epoch=parsed_args.epoch,
            enable_frequency=parsed_args.frequency,
            precise_ratio=parsed_args.precise_ratio,
        )

        print_scenario_summary(scenario)
        print(f"\n场景已保存: {parsed_args.output}")

        return 0

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
