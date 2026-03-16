#!/usr/bin/env python3
"""
区域目标Benchmark工具模块

提供区域目标场景的加载、分解、可见性计算等功能
"""

import json
import math
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from core.models import Mission, Satellite, Target, GroundStation
from core.models.satellite import (
    SatelliteCapabilities, Orbit, SatelliteType, OrbitType, OrbitSource
)
from core.models.target import TargetType
from core.models.mosaic_tile import MosaicTile
from core.models.area_coverage_plan import AreaCoveragePlan
from core.decomposer.mosaic_planner import MosaicPlanner
from core.orbit.visibility import VisibilityWindow, VisibilityWindowCache

logger = logging.getLogger(__name__)

# 模块级缓存，用于存储从场景文件中提取的区域边界
_scenario_bounds_cache: Optional[Dict[str, float]] = None


def _extract_scenario_bounds(scenario_data: Dict[str, Any]) -> Dict[str, float]:
    """
    从场景数据中提取区域目标的边界框

    Args:
        scenario_data: 场景JSON数据

    Returns:
        边界框字典 {min_lon, max_lon, min_lat, max_lat}
    """
    # 查找区域目标
    area_targets = [
        t for t in scenario_data.get('targets', [])
        if t.get('target_type') == 'area'
    ]

    if not area_targets:
        # 如果没有区域目标，返回一个默认的全球范围
        return {
            'min_lon': -180.0,
            'max_lon': 180.0,
            'min_lat': -90.0,
            'max_lat': 90.0
        }

    # 使用第一个区域目标的顶点计算边界框
    area_target = area_targets[0]
    vertices = area_target.get('area_vertices', [])

    if not vertices:
        return {
            'min_lon': -180.0,
            'max_lon': 180.0,
            'min_lat': -90.0,
            'max_lat': 90.0
        }

    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]

    # 添加缓冲区（10%的扩展）以确保包含周边区域
    lon_buffer = (max(lons) - min(lons)) * 0.1
    lat_buffer = (max(lats) - min(lats)) * 0.1

    return {
        'min_lon': min(lons) - lon_buffer,
        'max_lon': max(lons) + lon_buffer,
        'min_lat': min(lats) - lat_buffer,
        'max_lat': max(lats) + lat_buffer
    }


def get_scenario_bounds() -> Dict[str, float]:
    """
    获取当前场景的区域边界

    Returns:
        边界框字典，如果未加载场景则返回全球范围
    """
    if _scenario_bounds_cache is None:
        return {
            'min_lon': -180.0,
            'max_lon': 180.0,
            'min_lat': -90.0,
            'max_lat': 90.0
        }
    return _scenario_bounds_cache


def clear_scenario_bounds():
    """清除场景边界缓存"""
    global _scenario_bounds_cache
    _scenario_bounds_cache = None


def load_area_scenario(scenario_path: str) -> Tuple[Mission, Target, List[MosaicTile]]:
    """
    加载区域目标场景并分解为tiles

    Args:
        scenario_path: 场景JSON文件路径

    Returns:
        (Mission, area_target, tiles)
    """
    logger.info(f"加载区域目标场景: {scenario_path}")

    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)

    # 提取场景中的区域边界，用于可见性计算的区域筛选
    global _scenario_bounds_cache
    _scenario_bounds_cache = _extract_scenario_bounds(scenario_data)
    logger.debug(f"场景区域边界: {_scenario_bounds_cache}")

    # 创建Mission对象
    start_time = datetime.fromisoformat(
        scenario_data['duration']['start'].replace('Z', '+00:00')
    )
    end_time = datetime.fromisoformat(
        scenario_data['duration']['end'].replace('Z', '+00:00')
    )

    mission = Mission(
        name=scenario_data['name'],
        start_time=start_time,
        end_time=end_time,
        description=scenario_data.get('description', '')
    )

    # 添加卫星
    for sat_data in scenario_data['satellites']:
        satellite = _create_satellite_from_json(sat_data)
        mission.add_satellite(satellite)

    # 查找区域目标
    area_target = None
    for tgt_data in scenario_data['targets']:
        if tgt_data.get('target_type') == 'area':
            area_target = _create_area_target_from_json(tgt_data)
            mission.add_target(area_target)
            break

    if area_target is None:
        raise ValueError("场景中未找到区域目标")

    logger.info(f"  卫星: {len(mission.satellites)} 颗")
    logger.info(f"  区域目标: {area_target.id}")
    logger.info(f"  区域顶点: {len(area_target.area_vertices)}")

    # 分解区域为tiles
    logger.info("分解区域为tiles...")
    mosaic_planner = MosaicPlanner()

    coverage_plan = mosaic_planner.create_coverage_plan(
        target=area_target,
        satellites=mission.satellites,
        overlap_ratio=area_target.max_overlap_ratio
    )

    logger.info(f"  Tiles数量: {len(coverage_plan.tiles)}")
    logger.info(f"  预计覆盖率: {coverage_plan.statistics.coverage_ratio*100:.1f}%")

    return mission, area_target, coverage_plan.tiles


def create_tile_targets(
    mission: Mission,
    tiles: List[MosaicTile],
    area_target: Target
) -> List[Target]:
    """
    将tiles创建为点目标并添加到mission

    Args:
        mission: Mission对象
        tiles: tiles列表
        area_target: 原始区域目标

    Returns:
        创建的tile目标列表
    """
    tile_targets = []

    for i, tile in enumerate(tiles):
        center_lon, center_lat = tile.center
        # 使用tile.tile_id作为目标ID，与MosaicPlanner生成的ID保持一致
        tile_target = Target(
            id=tile.tile_id,  # 如 AREA-TAIWAN-001-T0001
            name=f"Tile {i+1}",
            target_type=TargetType.POINT,
            longitude=center_lon,
            latitude=center_lat,
            priority=area_target.priority,
            required_observations=1,
            resolution_required=area_target.resolution_required
        )
        tile_targets.append(tile_target)
        mission.add_target(tile_target)

    logger.info(f"添加了 {len(tile_targets)} 个tile目标到mission")
    return tile_targets


def calculate_visibility_windows(
    mission: Mission,
    tile_targets: List[Target],
    min_elevation: float = 20.0
) -> VisibilityWindowCache:
    """
    为所有tiles计算可见性窗口

    Args:
        mission: Mission对象
        tile_targets: tile目标列表
        min_elevation: 最小仰角（度）

    Returns:
        VisibilityWindowCache对象
    """
    logger.info("计算tiles的可见性窗口...")

    cache = VisibilityWindowCache()
    total_windows = 0

    # 为每个卫星-tile对计算可见性
    for sat in mission.satellites:
        sat_windows = {}
        for target in tile_targets:
            windows = _calculate_simple_visibility(
                sat, target, mission.start_time, mission.end_time, min_elevation
            )
            if windows:
                sat_windows[target.id] = windows
                total_windows += len(windows)

        if sat_windows:
            # 添加到缓存
            for target_id, windows in sat_windows.items():
                cache._windows[(sat.id, target_id)] = windows

    logger.info(f"  计算完成: {total_windows} 个可见窗口")
    logger.info(f"  卫星-目标对: {len(cache._windows)}")

    return cache


def _create_satellite_from_json(sat_data: Dict[str, Any]) -> Satellite:
    """从JSON数据创建卫星对象"""
    orbit = Orbit(
        orbit_type=OrbitType.LEO,
        semi_major_axis=sat_data['orbit']['semi_major_axis'],
        eccentricity=sat_data['orbit']['eccentricity'],
        inclination=sat_data['orbit']['inclination'],
        raan=sat_data['orbit']['raan'],
        arg_of_perigee=sat_data['orbit']['arg_of_perigee'],
        mean_anomaly=sat_data['orbit']['mean_anomaly'],
        epoch=datetime.fromisoformat(sat_data['orbit']['epoch'].replace('Z', '+00:00')),
        source=OrbitSource.ELEMENTS
    )

    caps = sat_data['capabilities']
    agility_data = caps.get('agility', {})
    capabilities = SatelliteCapabilities(
        max_roll_angle=caps.get('max_roll_angle', 35.0),
        max_pitch_angle=caps.get('max_pitch_angle', 20.0),
        agility={
            'max_slew_rate': agility_data.get('max_slew_rate', 3.0),
            'max_roll_rate': agility_data.get('max_roll_rate', 3.0),
            'max_pitch_rate': agility_data.get('max_pitch_rate', 2.0),
            'max_roll_acceleration': agility_data.get('max_roll_acceleration', 1.5),
            'max_pitch_acceleration': agility_data.get('max_pitch_acceleration', 1.0),
            'settling_time': agility_data.get('settling_time', 5.0)
        },
        swath_width=caps.get('swath_width', 10000.0),
        resolution=caps.get('resolution', 1.0),
        storage_capacity=caps.get('storage_capacity', 128.0),
        power_capacity=caps.get('power_capacity', 2000.0),
        data_rate=caps.get('data_rate', 300.0)
    )

    # 确定卫星类型
    sat_type_str = sat_data.get('sat_type', 'optical')
    if sat_type_str == 'optical':
        sat_type = SatelliteType.OPTICAL_1
    elif sat_type_str == 'sar':
        sat_type = SatelliteType.SAR_1
    else:
        sat_type = SatelliteType.OPTICAL_1

    return Satellite(
        id=sat_data['id'],
        name=sat_data.get('name', sat_data['id']),
        sat_type=sat_type,
        orbit=orbit,
        capabilities=capabilities
    )


def _create_area_target_from_json(tgt_data: Dict[str, Any]) -> Target:
    """从JSON数据创建区域目标"""
    return Target(
        id=tgt_data['id'],
        name=tgt_data.get('name', tgt_data['id']),
        target_type=TargetType.AREA,
        area_vertices=tgt_data['area_vertices'],
        priority=tgt_data.get('priority', 1),
        required_observations=tgt_data.get('required_observations', 1),
        resolution_required=tgt_data.get('resolution_required', 10.0),
        mosaic_required=tgt_data.get('mosaic_required', True),
        min_coverage_ratio=tgt_data.get('min_coverage_ratio', 0.95),
        max_overlap_ratio=tgt_data.get('max_overlap_ratio', 0.15),
        tile_priority_mode=tgt_data.get('tile_priority_mode', 'center_first'),
        dynamic_tile_sizing=tgt_data.get('dynamic_tile_sizing', True),
        coverage_strategy=tgt_data.get('coverage_strategy', 'max_coverage')
    )


def _calculate_simple_visibility(
    satellite: Satellite,
    target: Target,
    start_time: datetime,
    end_time: datetime,
    min_elevation: float = 20.0
) -> List[VisibilityWindow]:
    """
    简化的可见性窗口计算

    基于轨道几何估算可见窗口，适用于benchmark场景
    """
    windows = []

    # 获取目标坐标
    if not (hasattr(target, 'longitude') and hasattr(target, 'latitude')):
        return windows

    tgt_lon = target.longitude
    tgt_lat = target.latitude

    # 获取轨道参数
    orbit = satellite.orbit
    if not orbit or not orbit.semi_major_axis:
        return windows

    # 计算轨道周期
    mu = 3.986004418e14  # 地球引力常数 m^3/s^2
    a = orbit.semi_major_axis
    period_sec = 2 * math.pi * math.sqrt(a**3 / mu)

    # 场景持续时间
    scene_duration = (end_time - start_time).total_seconds()
    num_orbits = int(scene_duration / period_sec) + 2

    # 区域范围检查：使用从场景文件中提取的边界
    bounds = get_scenario_bounds()
    if not (bounds['min_lon'] <= tgt_lon <= bounds['max_lon'] and
            bounds['min_lat'] <= tgt_lat <= bounds['max_lat']):
        logger.debug(f"目标 ({tgt_lon}, {tgt_lat}) 超出场景区域范围，跳过")
        return windows

    # 基于卫星轨道参数生成模拟窗口
    mean_anomaly_rad = math.radians(orbit.mean_anomaly)
    raan_rad = math.radians(orbit.raan)

    for i in range(num_orbits):
        # 估算过境时间
        orbit_phase = (mean_anomaly_rad / (2 * math.pi) + i * 0.15 + raan_rad / 100) % 1.0

        time_offset = timedelta(seconds=i * period_sec + orbit_phase * period_sec * 0.5)
        window_start = start_time + time_offset

        if window_start > end_time:
            break

        # 窗口持续时间（3-8分钟）
        duration = timedelta(minutes=random.uniform(3, 8))
        window_end = min(window_start + duration, end_time)

        if window_start < end_time:
            window = VisibilityWindow(
                target_id=target.id,
                satellite_id=satellite.id,
                start_time=window_start,
                end_time=window_end,
                max_elevation=random.uniform(min_elevation + 10, 75)
            )
            windows.append(window)

    return windows
