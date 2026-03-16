"""
顶层便捷API - 类似numpy的函数式接口

设计理念：
- 一行代码完成常见任务
- 合理的默认参数
- 支持多种输入格式
"""

from typing import Union, Dict, Any, Optional, List
from pathlib import Path
import json
import logging
from datetime import datetime
import yaml

from core.models import Mission
from scheduler.unified_scheduler import UnifiedScheduler, UnifiedScheduleResult
from core.orbit.visibility.batch_calculator import BatchVisibilityCalculator, BatchComputationConfig
from evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ScenarioLoadError(Exception):
    """场景加载错误"""
    pass


def _load_scenario_data(scenario: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    加载场景数据

    Parameters
    ----------
    scenario : str, Path, or dict
        场景文件路径或字典

    Returns
    -------
    dict
        场景数据

    Raises
    ------
    ScenarioLoadError
        当文件不存在、格式错误或权限不足时
    """
    if isinstance(scenario, (str, Path)):
        scenario_path = Path(scenario)

        # 检查文件是否存在
        if not scenario_path.exists():
            raise ScenarioLoadError(f"场景文件不存在: {scenario_path}")

        # 检查是否是文件
        if not scenario_path.is_file():
            raise ScenarioLoadError(f"路径不是文件: {scenario_path}")

        try:
            with open(scenario_path, 'r', encoding='utf-8') as f:
                if scenario_path.suffix.lower() == '.json':
                    try:
                        return json.load(f)
                    except json.JSONDecodeError as e:
                        raise ScenarioLoadError(f"JSON格式错误: {e}")
                else:
                    # 假设是 YAML 格式
                    try:
                        return yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ScenarioLoadError(f"YAML格式错误: {e}")
        except PermissionError:
            raise ScenarioLoadError(f"无权限读取场景文件: {scenario_path}")
        except UnicodeDecodeError as e:
            raise ScenarioLoadError(f"文件编码错误（请使用UTF-8）: {e}")
        except Exception as e:
            raise ScenarioLoadError(f"读取场景文件失败: {e}")

    return scenario


def _create_mission(scenario_data: Dict[str, Any]) -> Mission:
    """从场景数据创建 Mission 对象"""
    from core.models.mission import Mission
    from core.models.satellite import Satellite, SatelliteType, SatelliteCapabilities
    from core.models.target import Target, TargetType
    from core.models.ground_station import GroundStation
    from datetime import datetime

    # 解析基本信息
    name = scenario_data.get('name', 'Unnamed')
    start_time = scenario_data.get('start_time', datetime.now())
    end_time = scenario_data.get('end_time', datetime.now())

    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

    # 创建 Mission 对象
    mission = Mission(
        name=name,
        start_time=start_time,
        end_time=end_time,
    )

    # 添加卫星
    for sat_config in scenario_data.get('satellites', []):
        try:
            sat_type = SatelliteType(sat_config.get('type', 'optical_1'))
        except ValueError:
            sat_type = SatelliteType.OPTICAL_1

        capabilities = sat_config.get('capabilities', {})
        caps = SatelliteCapabilities(
            imaging_modes=capabilities.get('imaging_modes', ['push_broom']),
            max_roll_angle=capabilities.get('max_roll_angle', 30.0),
            storage_capacity=capabilities.get('storage_capacity', 500.0),
        )

        satellite = Satellite(
            id=sat_config.get('id', 'unknown'),
            name=sat_config.get('name', sat_config.get('id', 'unknown')),
            sat_type=sat_type,
            capabilities=caps,
        )
        mission.add_satellite(satellite)

    # 添加目标
    targets_data = scenario_data.get('targets', [])
    if isinstance(targets_data, dict):
        # 点群目标格式
        for group_id, group_data in targets_data.items():
            for target in group_data.get('targets', []):
                t = Target(
                    id=target.get('id', f'{group_id}_{target.get("name", "unknown")}'),
                    name=target.get('name', ''),
                    latitude=target.get('lat', target.get('latitude', 0.0)),
                    longitude=target.get('lon', target.get('longitude', 0.0)),
                    priority=target.get('priority', 1),
                )
                mission.add_target(t)
    elif isinstance(targets_data, list):
        # 列表格式
        for target in targets_data:
            t = Target(
                id=target.get('id', 'unknown'),
                name=target.get('name', ''),
                latitude=target.get('lat', target.get('latitude', 0.0)),
                longitude=target.get('lon', target.get('longitude', 0.0)),
                priority=target.get('priority', 1),
            )
            mission.add_target(t)

    # 添加地面站
    for gs_config in scenario_data.get('ground_stations', []):
        gs = GroundStation(
            id=gs_config.get('id', 'unknown'),
            lat=gs_config.get('location', [0.0, 0.0])[0] if isinstance(gs_config.get('location'), list) else 0.0,
            lon=gs_config.get('location', [0.0, 0.0])[1] if isinstance(gs_config.get('location'), list) else 0.0,
        )
        mission.add_ground_station(gs)

    return mission


def schedule(
    scenario: Union[str, Path, Dict[str, Any]],
    algorithm: str = "greedy",
    cache_path: Optional[str] = None,
    enable_frequency: bool = True,
    enable_downlink: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    执行卫星任务调度（主入口函数）

    Parameters
    ----------
    scenario : str, Path, or dict
        场景配置，可以是：
        - 场景文件路径（JSON/YAML）
        - 场景配置字典
    algorithm : str, default "greedy"
        调度算法：'greedy', 'ga', 'edd', 'spt', 'sa', 'aco', 'pso', 'tabu'
    cache_path : str, optional
        可见性窗口缓存文件路径
    enable_frequency : bool, default True
        是否启用频次约束
    enable_downlink : bool, default True
        是否启用数传规划
    **kwargs : dict
        算法特定参数 (如 generations, population_size 等)

    Returns
    -------
    result : dict
        调度结果，包含：
        - scheduled_tasks: 任务列表
        - metrics: 性能指标
        - frequency_satisfaction: 频次满足情况
        - downlink_result: 数传结果

    Examples
    --------
    >>> import missionplanalgo as mpa

    # 基本用法
    >>> result = mpa.schedule("scenarios/test.json", algorithm="greedy")

    # 使用遗传算法
    >>> result = mpa.schedule("scenarios/test.json", algorithm="ga", generations=200)

    # 从字典创建场景
    >>> scenario = {"satellites": [...], "targets": [...]}
    >>> result = mpa.schedule(scenario)
    """
    logger.info(f"执行调度: algorithm={algorithm}, scenario={scenario}")

    # 加载场景
    scenario_data = _load_scenario_data(scenario)
    mission = _create_mission(scenario_data)

    # 加载窗口缓存
    if cache_path and Path(cache_path).exists():
        from scheduler.utils.window_cache import WindowCache
        window_cache = WindowCache()
        window_cache.load_from_json(cache_path)
    else:
        window_cache = None

    # 构建配置
    config = {
        'algorithm': algorithm,
        'consider_frequency': enable_frequency,
        'consider_downlink': enable_downlink,
    }

    # 添加算法特定参数
    if algorithm == 'ga':
        config['ga_params'] = {
            'generations': kwargs.get('generations', 50),
            'population_size': kwargs.get('population_size', 80),
            'mutation_rate': kwargs.get('mutation_rate', 0.2),
            'crossover_rate': kwargs.get('crossover_rate', 0.8),
        }

    # 执行调度
    scheduler = UnifiedScheduler(mission, window_cache, config=config)
    schedule_result = scheduler.schedule()

    # 转换结果为字典格式
    output = _convert_schedule_result(schedule_result)
    output['algorithm'] = algorithm
    return output


def _convert_schedule_result(result: UnifiedScheduleResult) -> Dict[str, Any]:
    """将调度结果转换为字典"""
    # 从 imaging_result 获取任务列表
    imaging_tasks = result.imaging_result.scheduled_tasks if result.imaging_result else []

    tasks = []
    for task in imaging_tasks:
        tasks.append({
            'task_id': task.task_id,
            'satellite_id': task.satellite_id,
            'target_id': task.target_id,
            'imaging_start': task.imaging_start.isoformat() if task.imaging_start else None,
            'imaging_end': task.imaging_end.isoformat() if task.imaging_end else None,
            'imaging_mode': task.imaging_mode,
            'slew_angle': task.slew_angle,
            'slew_time': task.slew_time,
        })

    # 计算频次满足率
    total_required = 0
    total_actual = 0
    for target_id, obs_info in result.target_observations.items():
        total_required += obs_info.get('required', 0)
        total_actual += obs_info.get('actual', 0)

    satisfaction_rate = total_actual / total_required if total_required > 0 else 1.0

    return {
        'scheduled_tasks': tasks,
        'metrics': {
            'scheduled_count': len(tasks),
            'total_demand': total_required,
            'frequency_satisfaction': satisfaction_rate,
            'satellite_utilization': result.imaging_result.satellite_utilization if hasattr(result.imaging_result, 'satellite_utilization') else 0.0,
            'makespan_hours': result.imaging_result.makespan / 3600 if result.imaging_result else 0.0,
        },
        'downlink_result': {
            'tasks': [t.__dict__ if hasattr(t, '__dict__') else str(t)
                     for t in result.downlink_result.downlink_tasks] if result.downlink_result else [],
        } if result.downlink_result else None,
        'frequency_satisfaction': result.target_observations,
    }


def compute_visibility(
    scenario: Union[str, Path, Dict[str, Any]],
    output_path: Optional[str] = None,
    use_java: bool = True,
    coarse_step: float = 5.0,
    fine_step: float = 1.0,
) -> Dict[str, Any]:
    """
    计算卫星可见性窗口

    Parameters
    ----------
    scenario : str, Path, or dict
        场景配置
    output_path : str, optional
        输出文件路径（JSON格式）
    use_java : bool, default True
        是否使用Java Orekit后端（高精度）
    coarse_step : float, default 5.0
        粗扫描步长（秒）
    fine_step : float, default 1.0
        精扫描步长（秒）

    Returns
    -------
    windows : dict
        可见性窗口数据

    Examples
    --------
    >>> windows = mpa.compute_visibility("scenarios/test.json")
    >>> print(f"计算了 {windows['total_windows']} 个窗口")
    """
    logger.info(f"计算可见性: scenario={scenario}, use_java={use_java}")

    # 加载场景
    scenario_data = _load_scenario_data(scenario)

    if use_java:
        # 使用 Java 后端
        calculator = BatchVisibilityCalculator()
        config = BatchComputationConfig(
            coarse_step_seconds=coarse_step,
            fine_step_seconds=fine_step,
        )

        mission = _create_mission(scenario_data)

        result = calculator.compute_all_windows(
            satellites=mission.satellites,
            targets=mission.targets,
            ground_stations=mission.ground_stations if hasattr(mission, 'ground_stations') else [],
            start_time=mission.start_time,
            end_time=mission.end_time,
            config=config,
        )

        output = {
            'total_windows': result.total_window_count,
            'satellite_count': len(mission.satellites),
            'target_count': len(mission.targets),
            'ground_station_count': len(mission.ground_stations) if hasattr(mission, 'ground_stations') else 0,
            'compute_time_seconds': result.computation_stats.total_computation_time_ms / 1000 if result.computation_stats else 0,
            'backend': 'java',
            'target_windows': result.to_cache_format()['target_windows'],
            'ground_station_windows': result.to_cache_format()['ground_station_windows'],
        }
    else:
        # Python 简化实现
        output = {
            'total_windows': 0,
            'satellite_count': len(scenario_data.get('satellites', [])),
            'target_count': len(scenario_data.get('targets', [])),
            'ground_station_count': len(scenario_data.get('ground_stations', [])),
            'compute_time_seconds': 0,
            'backend': 'python',
            'target_windows': [],
            'ground_station_windows': [],
        }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"结果已保存: {output_path}")

    return output


def load_scenario(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载场景配置文件

    Parameters
    ----------
    path : str or Path
        场景文件路径（JSON或YAML）

    Returns
    -------
    scenario : dict
        场景数据字典

    Examples
    --------
    >>> scenario = mpa.load_scenario("scenarios/test.json")
    >>> print(f"卫星数: {len(scenario['satellites'])}")
    """
    return _load_scenario_data(path)


def evaluate_schedule(
    result: Dict[str, Any],
    scenario: Optional[Union[str, Path, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    评估调度结果性能

    Parameters
    ----------
    result : dict
        调度结果（schedule函数的返回值）
    scenario : str, Path, or dict, optional
        场景配置（用于计算额外指标）

    Returns
    -------
    metrics : dict
        性能指标，包含：
        - completion_rate: 任务完成率
        - resource_utilization: 资源利用率
        - frequency_satisfaction: 频次满足率
        - makespan: 完成时间跨度

    Examples
    --------
    >>> result = mpa.schedule("scenarios/test.json")
    >>> metrics = mpa.evaluate_schedule(result)
    >>> print(f"完成率: {metrics['completion_rate']:.1%}")
    """
    logger.info("评估调度结果")

    metrics_data = result.get("metrics", {})
    scheduled_count = metrics_data.get("scheduled_count", 0)

    # 计算完成率
    completion_rate = 1.0
    if scenario:
        scenario_data = _load_scenario_data(scenario)
        total_demand = sum(
            t.get("required_observations", 1)
            for t in scenario_data.get("targets", [])
        )
        if total_demand > 0:
            completion_rate = min(scheduled_count / total_demand, 1.0)

    return {
        "overall_score": (
            metrics_data.get("frequency_satisfaction", 0) * 0.4 +
            min(metrics_data.get("satellite_utilization", 0) / 0.5, 1.0) * 0.3 +
            min(24.0 / metrics_data.get("makespan_hours", 24), 1.0) * 0.3
        ),
        "completion_rate": completion_rate,
        "resource_utilization": metrics_data.get("satellite_utilization", 0),
        "frequency_satisfaction": metrics_data.get("frequency_satisfaction", 0),
        "makespan_hours": metrics_data.get("makespan_hours", 0),
        "scheduled_count": scheduled_count,
        "metrics": metrics_data,
    }
