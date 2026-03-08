#!/usr/bin/env python3
"""
脚本工具模块 - 为脚本整合提供公共功能

此模块提取了多个脚本中重复的功能:
- 窗口缓存加载 (load_window_cache_from_json)
- 调度器注册表 (SCHEDULER_REGISTRY)
- 日志配置 (setup_logging)
- 结果保存 (save_results)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Type, List, Optional

# Import scheduler classes
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.metaheuristic.aco_scheduler import ACOScheduler
from scheduler.metaheuristic.pso_scheduler import PSOScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler
from scheduler.metaheuristic.tabu_scheduler import TabuScheduler


# ============================================================================
# 调度器注册表
# ============================================================================

SCHEDULER_REGISTRY: Dict[str, Type] = {
    'greedy': GreedyScheduler,
    'edd': EDDScheduler,
    'spt': SPTScheduler,
    'ga': GAScheduler,
    'aco': ACOScheduler,
    'pso': PSOScheduler,
    'sa': SAScheduler,
    'tabu': TabuScheduler,
}

ALGORITHM_CATEGORIES: Dict[str, List[str]] = {
    'greedy': ['greedy', 'edd', 'spt'],
    'metaheuristic': ['ga', 'aco', 'pso', 'sa', 'tabu'],
}


def get_scheduler_class(name: str) -> Type:
    """
    根据名称获取调度器类

    Args:
        name: 调度器名称 (如 'greedy', 'ga')

    Returns:
        调度器类

    Raises:
        ValueError: 如果调度器名称未知
    """
    name_lower = name.lower()
    if name_lower not in SCHEDULER_REGISTRY:
        available = ', '.join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm: '{name}'. Available: {available}")
    return SCHEDULER_REGISTRY[name_lower]


def get_algorithms_by_category(category: str) -> List[str]:
    """
    获取指定类别的所有算法

    Args:
        category: 类别名称 ('greedy' 或 'metaheuristic')

    Returns:
        算法名称列表
    """
    return ALGORITHM_CATEGORIES.get(category, [])


def get_algorithm_category(name: str) -> Optional[str]:
    """
    获取算法的类别

    Args:
        name: 算法名称

    Returns:
        类别名称，如果未知则返回 None
    """
    name_lower = name.lower()
    for category, algorithms in ALGORITHM_CATEGORIES.items():
        if name_lower in algorithms:
            return category
    return None


# ============================================================================
# 窗口缓存加载
# ============================================================================

def load_window_cache_from_json(cache_path: str, mission) -> 'VisibilityWindowCache':
    """
    从JSON文件加载预计算的窗口缓存

    支持两种格式:
    1. 新格式: {"target_windows": [...], "ground_station_windows": [...]}
    2. 旧格式: {"windows": [{"sat": ..., "tgt": ..., "start": ..., "end": ...}]}

    Args:
        cache_path: 缓存文件路径
        mission: 任务场景对象（用于获取卫星和目标信息）

    Returns:
        VisibilityWindowCache: 填充好的缓存对象

    Raises:
        FileNotFoundError: 如果缓存文件不存在
    """
    from core.orbit.visibility.window_cache import VisibilityWindowCache
    from core.orbit.visibility.base import VisibilityWindow

    cache_file = Path(cache_path)
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache = VisibilityWindowCache()

    # 加载卫星-目标窗口
    target_windows_count = 0

    # 尝试新格式
    target_windows = data.get('target_windows', [])

    if target_windows:
        # 新格式处理
        for w_data in target_windows:
            sat_id = w_data['satellite_id']
            target_id = w_data['target_id']

            window = VisibilityWindow(
                satellite_id=sat_id,
                target_id=target_id,
                start_time=datetime.fromisoformat(w_data['start_time']),
                end_time=datetime.fromisoformat(w_data['end_time']),
                max_elevation=w_data.get('max_elevation', 0.0)
            )

            key = (sat_id, target_id)
            if key not in cache._windows:
                cache._windows[key] = []
                cache._time_index[key] = []

            cache._windows[key].append(window)
            cache._time_index[key].append(window.start_time)

            # 更新辅助索引
            if sat_id not in cache._sat_to_targets:
                cache._sat_to_targets[sat_id] = set()
            cache._sat_to_targets[sat_id].add(target_id)

            if target_id not in cache._target_to_sats:
                cache._target_to_sats[target_id] = set()
            cache._target_to_sats[target_id].add(sat_id)

            target_windows_count += 1
    else:
        # 尝试旧格式 (windows数组)
        windows_list = data.get('windows', [])
        for w_data in windows_list:
            sat_id = w_data.get('sat') or w_data.get('satellite_id')
            target_id = w_data.get('tgt') or w_data.get('target_id')
            start_time = w_data.get('start') or w_data.get('start_time')
            end_time = w_data.get('end') or w_data.get('end_time')
            max_el = w_data.get('el', 0.0) or w_data.get('max_elevation', 0.0)

            if not all([sat_id, target_id, start_time, end_time]):
                continue

            # 跳过地面站窗口（旧格式中可能混有地面站窗口）
            if target_id.startswith('GS:'):
                continue

            try:
                # 处理ISO格式时间 (带Z后缀)
                if isinstance(start_time, str) and start_time.endswith('Z'):
                    start_time = start_time.replace('Z', '+00:00')
                if isinstance(end_time, str) and end_time.endswith('Z'):
                    end_time = end_time.replace('Z', '+00:00')

                window = VisibilityWindow(
                    satellite_id=sat_id,
                    target_id=target_id,
                    start_time=datetime.fromisoformat(start_time),
                    end_time=datetime.fromisoformat(end_time),
                    max_elevation=max_el
                )

                key = (sat_id, target_id)
                if key not in cache._windows:
                    cache._windows[key] = []
                    cache._time_index[key] = []

                cache._windows[key].append(window)
                cache._time_index[key].append(window.start_time)

                # 更新辅助索引
                if sat_id not in cache._sat_to_targets:
                    cache._sat_to_targets[sat_id] = set()
                cache._sat_to_targets[sat_id].add(target_id)

                if target_id not in cache._target_to_sats:
                    cache._target_to_sats[target_id] = set()
                cache._target_to_sats[target_id].add(sat_id)

                target_windows_count += 1
            except Exception:
                continue

    # 加载卫星-地面站窗口
    gs_windows_count = 0

    # 尝试从独立的 ground_station_windows 键加载
    gs_windows_list = data.get('ground_station_windows', [])

    # 如果没有独立键，尝试从 windows 数组中分离
    if not gs_windows_list and 'windows' in data:
        gs_windows_list = [w for w in data['windows'] if 'GS:' in (w.get('tgt', '') or w.get('target_id', ''))]

    for w_data in gs_windows_list:
        sat_id = w_data.get('sat') or w_data.get('satellite_id')
        target_id = w_data.get('tgt') or w_data.get('target_id')
        start_time = w_data.get('start') or w_data.get('start_time')
        end_time = w_data.get('end') or w_data.get('end_time')
        max_el = w_data.get('el', 0.0) or w_data.get('max_elevation', 0.0)

        if not all([sat_id, target_id, start_time, end_time]):
            continue

        try:
            # 处理ISO格式时间 (带Z后缀)
            if isinstance(start_time, str) and start_time.endswith('Z'):
                start_time = start_time.replace('Z', '+00:00')
            if isinstance(end_time, str) and end_time.endswith('Z'):
                end_time = end_time.replace('Z', '+00:00')

            window = VisibilityWindow(
                satellite_id=sat_id,
                target_id=target_id,
                start_time=datetime.fromisoformat(start_time),
                end_time=datetime.fromisoformat(end_time),
                max_elevation=max_el
            )

            key = (sat_id, target_id)
            if key not in cache._windows:
                cache._windows[key] = []
                cache._time_index[key] = []

            cache._windows[key].append(window)
            cache._time_index[key].append(window.start_time)

            # 更新辅助索引
            if sat_id not in cache._sat_to_targets:
                cache._sat_to_targets[sat_id] = set()
            cache._sat_to_targets[sat_id].add(target_id)

            if target_id not in cache._target_to_sats:
                cache._target_to_sats[target_id] = set()
            cache._target_to_sats[target_id].add(sat_id)

            gs_windows_count += 1
        except Exception:
            continue

    # 对所有窗口排序
    for key in cache._windows:
        sorted_pairs = sorted(zip(cache._time_index[key], cache._windows[key]))
        cache._time_index[key] = [p[0] for p in sorted_pairs]
        cache._windows[key] = [p[1] for p in sorted_pairs]

    return cache


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> logging.Logger:
    """
    设置统一的日志配置

    Args:
        level: 日志级别 (默认 INFO)
        format_string: 自定义格式字符串

    Returns:
        配置好的日志记录器
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_string,
        force=True  # 强制重新配置
    )

    return logging.getLogger(__name__)


# ============================================================================
# 结果保存
# ============================================================================

def save_results(data: Dict[str, Any], output_path: str, indent: int = 2) -> None:
    """
    保存结果到JSON文件

    Args:
        data: 要保存的数据字典
        output_path: 输出文件路径
        indent: JSON缩进空格数
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ============================================================================
# 辅助函数
# ============================================================================

def parse_algorithm_list(algorithms_str: str) -> List[str]:
    """
    解析逗号分隔的算法列表

    Args:
        algorithms_str: 逗号分隔的算法名称字符串

    Returns:
        算法名称列表
    """
    return [a.strip().lower() for a in algorithms_str.split(',') if a.strip()]


def validate_algorithms(algorithms: List[str]) -> List[str]:
    """
    验证算法名称是否有效

    Args:
        algorithms: 算法名称列表

    Returns:
        有效的算法名称列表

    Raises:
        ValueError: 如果有无效算法名称
    """
    invalid = [a for a in algorithms if a.lower() not in SCHEDULER_REGISTRY]
    if invalid:
        available = ', '.join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Invalid algorithms: {', '.join(invalid)}. Available: {available}")
    return algorithms


# ============================================================================
# 结果格式化
# ============================================================================

def format_metrics_table(results: List[Dict[str, Any]]) -> str:
    """
    格式化结果为表格字符串

    Args:
        results: 结果字典列表，每个字典包含 algorithm 和 metrics

    Returns:
        格式化的表格字符串
    """
    lines = []
    lines.append(f"{'算法':<20} {'任务数':<10} {'满足率':<10} {'利用率':<10} {'用时(秒)':<12}")
    lines.append("-" * 70)

    for r in results:
        algo_name = r.get('algorithm', 'unknown')
        m = r.get('metrics', {})
        lines.append(
            f"{algo_name:<20} "
            f"{m.get('scheduled_tasks', 0):<10} "
            f"{m.get('demand_satisfaction_rate', 0):<10.1%} "
            f"{m.get('satellite_utilization', 0):<10.1%} "
            f"{m.get('total_computation_time', m.get('computation_time', 0)):<12.2f}"
        )

    return '\n'.join(lines)


def format_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    格式化多算法对比结果为表格字符串

    Args:
        results: 结果字典列表

    Returns:
        格式化的表格字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append("多算法对比结果")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'算法':<20} {'任务数':<10} {'满足率':<10} {'利用率':<10} {'计算时间':<10}")
    lines.append("-" * 70)

    for r in results:
        algo_name = r.get('algorithm_name', r.get('algorithm', 'unknown'))
        if len(algo_name) > 18:
            algo_name = algo_name[:15] + "..."
        lines.append(
            f"{algo_name:<20} "
            f"{r.get('scheduled_count', 0):<10} "
            f"{r.get('demand_satisfaction_rate', 0):<10.1%} "
            f"{r.get('satellite_utilization', 0):<10.1%} "
            f"{r.get('computation_time', 0):<10.2f}s"
        )

    lines.append("=" * 70)
    return '\n'.join(lines)


# ============================================================================
# 场景加载缓存
# ============================================================================

from functools import lru_cache

@lru_cache(maxsize=4)
def load_mission_cached(scenario_path: str):
    """
    带缓存的场景加载

    Args:
        scenario_path: 场景文件路径

    Returns:
        Mission 对象
    """
    from core.models import Mission
    return Mission.load(scenario_path)


# ============================================================================
# 报告生成
# ============================================================================

def generate_benchmark_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
    filename_prefix: str = 'benchmark'
) -> tuple:
    """
    生成基准测试报告 (JSON和文本格式)

    Args:
        results: 测试结果列表
        output_dir: 输出目录
        filename_prefix: 文件名前缀

    Returns:
        (json_file_path, txt_file_path) 元组
    """
    from datetime import datetime

    report_file = output_dir / f"{filename_prefix}_report.json"
    summary_file = output_dir / f"{filename_prefix}_summary.txt"

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON报告
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_algorithms': len(results),
        'successful': sum(1 for r in results if r.get('status') == 'success'),
        'failed': sum(1 for r in results if r.get('status') != 'success'),
        'results': results
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # 生成文本摘要
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') != 'success']

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("基准测试报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总算法数: {len(results)}\n")
        f.write(f"成功: {len(successful)} | 失败: {len(failed)}\n")
        f.write("=" * 80 + "\n\n")

        if successful:
            f.write("【成功运行的算法】\n\n")
            f.write(format_metrics_table(successful))
            f.write("\n\n")

        if failed:
            f.write("【运行失败的算法】\n\n")
            for r in failed:
                f.write(f"- {r.get('algorithm', 'unknown')}: {r.get('error', 'Unknown error')}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    return report_file, summary_file


# ============================================================================
# 辅助装饰器
# ============================================================================

def timer_decorator(func):
    """
    计时装饰器 - 记录函数执行时间

    Usage:
        @timer_decorator
        def my_function():
            pass
    """
    import time
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} 耗时: {elapsed:.2f}秒")
        return result

    return wrapper


# ============================================================================
# 文件操作
# ============================================================================

def ensure_dir(path: str) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json_safe(file_path: str, default=None) -> Any:
    """
    安全加载JSON文件，如果失败返回默认值

    Args:
        file_path: JSON文件路径
        default: 失败时的默认值

    Returns:
        解析后的JSON数据或默认值
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"加载JSON文件失败: {file_path}, 错误: {e}")
        return default

