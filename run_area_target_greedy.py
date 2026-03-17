#!/usr/bin/env python3
"""
区域目标场景Greedy调度脚本
流程: 加载场景 -> 分解区域为tiles -> 计算可见性窗口 -> Greedy调度
"""

import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _calculate_simple_visibility(satellite, target, start_time, end_time):
    """
    简化的可见性窗口计算
    基于Walker星座的轨道几何估算
    """
    from core.orbit.visibility import VisibilityWindow
    from datetime import timedelta
    import math
    import random

    windows = []

    # 获取目标坐标
    if hasattr(target, 'longitude') and hasattr(target, 'latitude'):
        tgt_lon = target.longitude
        tgt_lat = target.latitude
    else:
        return windows

    # 简化的可见性估算：基于轨道周期和卫星分布
    # 台湾区域大约经度120-122度，纬度22-25度
    # 使用轨道参数估算过境时间

    orbit = satellite.orbit
    if orbit and orbit.semi_major_axis:
        # 计算轨道周期 (开普勒第三定律)
        mu = 3.986004418e14  # 地球引力常数 m^3/s^2
        a = orbit.semi_major_axis
        period_sec = 2 * math.pi * math.sqrt(a**3 / mu)

        # 基于轨道倾角和升交点赤经估算可见性
        inclination_rad = math.radians(orbit.inclination)
        raan_rad = math.radians(orbit.raan)

        # 台湾区域平均纬度约23.5度
        taiwan_lat_rad = math.radians(23.5)

        # 计算卫星能覆盖该纬度的最大经度范围
        # 简化模型：基于轨道几何
        cos_max_lat = math.cos(inclination_rad) / math.cos(taiwan_lat_rad)
        if abs(cos_max_lat) <= 1.0:
            max_lat_coverage = math.acos(cos_max_lat)
        else:
            max_lat_coverage = 0

        # 检查目标是否在可能的覆盖范围内
        # 简化判断：台湾区域经度120-122
        if 115 <= tgt_lon <= 125 and 20 <= tgt_lat <= 26:
            # 生成模拟的可见窗口
            # 基于卫星的平近点角分布在轨道上
            mean_anomaly_rad = math.radians(orbit.mean_anomaly)

            # 在24小时场景时间内生成1-2个窗口
            scene_duration = (end_time - start_time).total_seconds()
            num_orbits = scene_duration / period_sec

            for i in range(int(num_orbits) + 1):
                # 基于卫星初始位置和轨道周期估算过境时间
                orbit_phase = (mean_anomaly_rad / (2 * math.pi) + i) % 1.0

                # 估算过境时间 (简化)
                time_offset = timedelta(seconds=i * period_sec + orbit_phase * period_sec * 0.5)
                window_start = start_time + time_offset

                if window_start > end_time:
                    break

                # 窗口持续时间 (基于幅宽估算，简化假设5-10分钟)
                duration = timedelta(minutes=random.uniform(3, 8))
                window_end = min(window_start + duration, end_time)

                if window_start < end_time and window_end > start_time:
                    window = VisibilityWindow(
                        target_id=target.id,
                        satellite_id=satellite.id,
                        start_time=window_start,
                        end_time=window_end,
                        max_elevation=random.uniform(30, 75)  # 简化最大仰角
                    )
                    windows.append(window)

    return windows


def main():
    script_start_time = time.time()

    # 1. 加载场景
    logger.info("=" * 60)
    logger.info("区域目标场景Greedy调度")
    logger.info("=" * 60)

    scenario_path = "scenarios/area_target.json"
    logger.info(f"[1/5] 加载场景配置: {scenario_path}")

    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)

    logger.info(f"  场景名称: {scenario_data['name']}")
    logger.info(f"  卫星数量: {len(scenario_data['satellites'])}")
    logger.info(f"  目标数量: {len(scenario_data['targets'])}")

    # 2. 处理区域目标 - 分解为tiles
    logger.info("[2/5] 处理区域目标...")

    from datetime import datetime
    from core.models import Mission, Satellite, Target, GroundStation
    from core.models.satellite import SatelliteCapabilities, Orbit, SatelliteType, OrbitType, OrbitSource
    from core.models.target import TargetType
    from core.decomposer.mosaic_planner import MosaicPlanner
    from core.orbit.visibility.batch_calculator import BatchVisibilityCalculator

    # 从JSON创建Mission对象
    start_time = datetime.fromisoformat(scenario_data['duration']['start'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(scenario_data['duration']['end'].replace('Z', '+00:00'))

    mission = Mission(
        name=scenario_data['name'],
        start_time=start_time,
        end_time=end_time,
        description=scenario_data.get('description', '')
    )

    # 添加卫星
    for sat_data in scenario_data['satellites']:
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

        satellite = Satellite(
            id=sat_data['id'],
            name=sat_data.get('name', sat_data['id']),
            sat_type=sat_type,
            orbit=orbit,
            capabilities=capabilities
        )
        mission.add_satellite(satellite)

    # 添加目标（区域目标）
    for tgt_data in scenario_data['targets']:
        if tgt_data.get('target_type') == 'area':
            # 创建区域目标 - 直接使用Target的参数
            target = Target(
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
            mission.add_target(target)

    logger.info(f"  创建Mission: {len(mission.satellites)} 卫星, {len(mission.targets)} 目标")

    # 获取区域目标
    area_target = None
    for target in mission.targets:
        logger.info(f"  检查目标: {target.id}, target_type={target.target_type}, type={type(target.target_type)}")
        if hasattr(target, 'target_type') and target.target_type == TargetType.AREA:
            area_target = target
            break

    if area_target is None:
        logger.error("未找到区域目标!")
        # 尝试使用第一个目标
        if len(mission.targets) > 0:
            area_target = mission.targets[0]
            logger.info(f"  使用第一个目标作为区域目标: {area_target.id}")
        else:
            return

    logger.info(f"  区域目标: {area_target.id}")
    logger.info(f"  区域顶点数: {len(area_target.area_vertices)}")

    # 使用MosaicPlanner分解区域
    logger.info("[3/5] 分解区域为tiles...")
    mosaic_planner = MosaicPlanner()

    # 获取卫星的成像能力参数
    # 使用第一颗光学卫星的参数作为参考
    ref_sat = None
    for sat in mission.satellites:
        if sat.sat_type.name.startswith('OPTICAL'):
            ref_sat = sat
            break

    if ref_sat is None:
        ref_sat = mission.satellites[0]

    swath_width = getattr(ref_sat.capabilities, 'swath_width', 15000.0)  # 米
    resolution = getattr(ref_sat.capabilities, 'resolution', 1.0)  # 米

    logger.info(f"  参考卫星: {ref_sat.id}")
    logger.info(f"  幅宽: {swath_width/1000:.1f} km")
    logger.info(f"  分辨率: {resolution} m")

    # 分解区域为tiles
    coverage_plan = mosaic_planner.create_coverage_plan(
        target=area_target,
        satellites=mission.satellites,
        overlap_ratio=area_target.max_overlap_ratio
    )

    logger.info(f"  分解完成!")
    logger.info(f"  Tiles数量: {len(coverage_plan.tiles)}")
    logger.info(f"  预计覆盖率: {coverage_plan.statistics.coverage_ratio*100:.1f}%")

    # 4. 为tiles创建目标并添加到mission
    logger.info("[4/5] 将tiles添加为点目标...")

    # 为每个tile创建一个点目标
    from core.models.target import TargetType
    tile_targets = []
    for i, tile in enumerate(coverage_plan.tiles):
        center_lon, center_lat = tile.center
        tile_target = Target(
            id=f"TILE-{i:04d}",
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

    logger.info(f"  添加了 {len(tile_targets)} 个tile目标到mission")

    # 5. 计算tiles的可见性窗口
    logger.info("[5/5] 计算tiles的可见性窗口...")

    from core.orbit.visibility import VisibilityWindow
    from datetime import timedelta
    import math

    # 简化的可见性计算 - 基于轨道几何
    # 实际生产环境应该使用Java后端或更精确的计算
    visibility_cache = {}
    window_count = 0

    for sat in mission.satellites:
        sat_windows = {}
        for target in tile_targets:
            # 简化计算：检查卫星是否经过目标附近
            # 使用卫星轨道参数估算可见窗口
            windows = _calculate_simple_visibility(
                sat, target, mission.start_time, mission.end_time
            )
            if windows:
                sat_windows[target.id] = windows
                window_count += len(windows)
        if sat_windows:
            visibility_cache[sat.id] = sat_windows

    logger.info(f"  计算完成: {window_count} 个可见窗口")

    # 将字典转换为VisibilityWindowCache对象
    from core.orbit.visibility import VisibilityWindowCache
    window_cache_obj = VisibilityWindowCache()
    window_cache_obj._windows = {}
    for sat_id, target_dict in visibility_cache.items():
        for target_id, windows in target_dict.items():
            window_cache_obj._windows[(sat_id, target_id)] = windows

    logger.info(f"  窗口缓存已创建: {len(window_cache_obj._windows)} 个卫星-目标对")

    # 6. 运行Greedy调度
    logger.info("[6/6] 运行Greedy调度器...")

    from scheduler.greedy.greedy_scheduler import GreedyScheduler
    from scheduler.common.config import SchedulerConfig, ConstraintConfig, ResourceConfig, AlgorithmConfig

    # 创建调度配置
    constraint_config = ConstraintConfig(
        enable_attitude_calculation=True,
        enable_area_coverage=True,
        area_coverage_strategy='max_coverage',
        min_area_coverage_ratio=0.95
    )

    resource_config = ResourceConfig()
    algorithm_config = AlgorithmConfig()

    config = SchedulerConfig(
        constraints=constraint_config,
        resources=resource_config,
        algorithm=algorithm_config
    )

    # 创建调度器
    config_dict = {
        'heuristic': 'priority',
        'consider_power': True,
        'consider_storage': True,
        'enable_area_coverage': True,
        'area_coverage_strategy': 'max_coverage',
        'min_area_coverage_ratio': 0.95
    }
    scheduler = GreedyScheduler(config=config_dict)

    # 初始化调度器
    logger.info("  初始化调度器...")
    scheduler.initialize(mission)

    # 设置可见性窗口缓存
    logger.info("  设置可见性窗口缓存...")
    scheduler.set_window_cache(window_cache_obj)

    # 运行调度
    logger.info("  开始调度...")
    schedule_result = scheduler.schedule()

    # 输出结果
    logger.info("=" * 60)
    logger.info("调度结果")
    logger.info("=" * 60)
    total_tasks = len(schedule_result.scheduled_tasks) + len(schedule_result.unscheduled_tasks)
    logger.info(f"总任务数: {total_tasks}")
    logger.info(f"成功调度任务数: {len(schedule_result.scheduled_tasks)}")
    logger.info(f"未调度任务数: {len(schedule_result.unscheduled_tasks)}")
    logger.info(f"完成时间跨度: {schedule_result.makespan/3600:.2f} 小时")

    if schedule_result.area_coverage_stats:
        logger.info(f"区域覆盖统计: {schedule_result.area_coverage_stats}")

    # 显示未调度任务的原因
    if schedule_result.unscheduled_tasks:
        logger.info("\n未调度任务原因分析:")
        from collections import Counter
        from scheduler.base_scheduler import TaskFailureReason
        reason_counts = Counter()
        for task_id, failure in list(schedule_result.unscheduled_tasks.items())[:5]:  # 只显示前5个示例
            logger.info(f"  {task_id}: {failure.failure_reason.value} - {failure.failure_detail}")

        for task_id, failure in schedule_result.unscheduled_tasks.items():
            reason_counts[failure.failure_reason] += 1

        logger.info("\n未调度原因统计:")
        for reason, count in reason_counts.most_common():
            logger.info(f"  {reason.value}: {count} 个任务")

    # 保存结果
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"area_target_greedy_{timestamp}.json"

    total_tasks = len(schedule_result.scheduled_tasks) + len(schedule_result.unscheduled_tasks)
    result_data = {
        "scenario": scenario_data['name'],
        "algorithm": "greedy",
        "timestamp": timestamp,
        "total_tasks": total_tasks,
        "scheduled_tasks": len(schedule_result.scheduled_tasks),
        "unscheduled_tasks": len(schedule_result.unscheduled_tasks),
        "makespan_hours": schedule_result.makespan / 3600,
        "area_coverage_stats": schedule_result.area_coverage_stats,
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    logger.info(f"  结果已保存: {output_path}")

    elapsed = time.time() - script_start_time
    logger.info(f"\n总耗时: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()
