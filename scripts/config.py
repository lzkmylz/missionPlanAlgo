#!/usr/bin/env python3
"""
统一配置管理模块

集中管理算法配置、名称映射和默认参数，消除配置结构重复。
"""

from typing import Dict, Any, List

# ============================================================================
# 算法名称映射
# ============================================================================

ALGORITHM_NAMES = {
    'greedy': 'Greedy (贪心)',
    'edd': 'EDD (最早截止时间优先)',
    'spt': 'SPT (最短处理时间优先)',
    'ga': 'GA (遗传算法)',
    'sa': 'SA (模拟退火)',
    'aco': 'ACO (蚁群优化)',
    'pso': 'PSO (粒子群优化)',
    'tabu': 'Tabu (禁忌搜索)',
}

# ============================================================================
# 算法分类
# ============================================================================

ALGORITHM_CATEGORIES = {
    'greedy': ['greedy', 'edd', 'spt'],
    'metaheuristic': ['ga', 'aco', 'pso', 'sa', 'tabu'],
}

# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_IMAGING_CONFIG = {
    'use_simplified_slew': False,  # 默认使用精细模式（标准模式）
    'consider_power': True,
    'consider_storage': True,
    'enable_attitude_calculation': True,  # 启用姿态角计算
    'precompute_positions': True,  # 默认预计算卫星位置（加速姿态计算）
    'precompute_step_seconds': 1.0,  # 预计算时间步长（秒），与HPOP精扫描步长匹配
}

DEFAULT_DOWNLINK_CONFIG = {
    'overflow_threshold': 0.95,
    'link_setup_time_seconds': 60.0,
}

# ============================================================================
# 算法特定配置模板
# ============================================================================

ALGORITHM_CONFIG_TEMPLATES = {
    'greedy': {
        'imaging_algorithm': 'greedy',
        'imaging_config': DEFAULT_IMAGING_CONFIG.copy(),
    },
    'edd': {
        'imaging_algorithm': 'edd',
        'imaging_config': DEFAULT_IMAGING_CONFIG.copy(),
    },
    'spt': {
        'imaging_algorithm': 'spt',
        'imaging_config': DEFAULT_IMAGING_CONFIG.copy(),
    },
    'ga': {
        'imaging_algorithm': 'ga',
        'imaging_config': {
            **DEFAULT_IMAGING_CONFIG,
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
        },
    },
    'sa': {
        'imaging_algorithm': 'sa',
        'imaging_config': {
            **DEFAULT_IMAGING_CONFIG,
            'initial_temperature': 100.0,
            'cooling_rate': 0.98,
            'max_iterations': 1000,
            'min_temperature': 0.001,
        },
    },
    'aco': {
        'imaging_algorithm': 'aco',
        'imaging_config': {
            **DEFAULT_IMAGING_CONFIG,
            'num_ants': 30,
            'max_iterations': 100,
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.1,
        },
    },
    'pso': {
        'imaging_algorithm': 'pso',
        'imaging_config': {
            **DEFAULT_IMAGING_CONFIG,
            'num_particles': 30,
            'max_iterations': 100,
            'inertia_weight': 0.9,
            'cognitive_coeff': 2.0,
            'social_coeff': 2.0,
        },
    },
    'tabu': {
        'imaging_algorithm': 'tabu',
        'imaging_config': {
            **DEFAULT_IMAGING_CONFIG,
            'tabu_tenure': 10,
            'max_iterations': 100,
            'neighborhood_size': 20,
            'aspiration_threshold': 0.05,
        },
    },
}

# ============================================================================
# 快捷选择映射
# ============================================================================

ALGORITHM_SELECTION_SHORTCUTS = {
    'all': ['greedy', 'edd', 'ga', 'sa', 'aco', 'pso', 'tabu'],
    'basic': ['greedy', 'edd', 'ga'],
    'meta': ['ga', 'sa', 'aco', 'pso', 'tabu'],
    'greedy': ['greedy', 'edd', 'spt'],
}


# ============================================================================
# 配置获取函数
# ============================================================================

def get_algorithm_config(
    algorithm: str,
    enable_downlink: bool = True,
    enable_frequency: bool = True,
    seed: int = 42,
    **overrides
) -> Dict[str, Any]:
    """
    获取算法配置

    Args:
        algorithm: 算法名称 (如 'ga', 'sa')
        enable_downlink: 是否启用数传规划
        enable_frequency: 是否启用频次需求
        seed: 随机种子
        **overrides: 覆盖默认配置的参数

    Returns:
        完整的调度器配置字典

    Raises:
        ValueError: 如果算法名称未知
    """
    if algorithm not in ALGORITHM_CONFIG_TEMPLATES:
        available = ', '.join(ALGORITHM_CONFIG_TEMPLATES.keys())
        raise ValueError(f"未知算法: '{algorithm}'。可用算法: {available}")

    # 深拷贝模板配置
    import copy
    config = copy.deepcopy(ALGORITHM_CONFIG_TEMPLATES[algorithm])

    # 添加通用配置
    config['enable_downlink'] = enable_downlink
    config['consider_frequency'] = enable_frequency
    config['imaging_config']['random_seed'] = seed

    # 应用覆盖参数
    config['imaging_config'].update(overrides)

    return config


def get_algorithm_name(algorithm: str) -> str:
    """
    获取算法的显示名称

    Args:
        algorithm: 算法键名

    Returns:
        算法显示名称，如果未知则返回键名
    """
    return ALGORITHM_NAMES.get(algorithm, algorithm)


def expand_algorithm_selection(selections: List[str]) -> List[str]:
    """
    扩展算法选择快捷方式

    Args:
        selections: 选择列表，可包含快捷方式如 'all', 'basic', 'meta'

    Returns:
        展开的算法键名列表，去重并保持顺序

    Example:
        >>> expand_algorithm_selection(['basic', 'sa'])
        ['greedy', 'edd', 'ga', 'sa']
    """
    result = []
    for s in selections:
        s_lower = s.lower()
        if s_lower in ALGORITHM_SELECTION_SHORTCUTS:
            result.extend(ALGORITHM_SELECTION_SHORTCUTS[s_lower])
        else:
            result.append(s_lower)

    # 去重并保持顺序
    return list(dict.fromkeys(result))


def get_algorithms_by_category(category: str) -> List[str]:
    """
    获取指定类别的所有算法

    Args:
        category: 类别名称 ('greedy' 或 'metaheuristic')

    Returns:
        算法名称列表
    """
    return ALGORITHM_CATEGORIES.get(category, [])


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
    invalid = [a for a in algorithms if a.lower() not in ALGORITHM_CONFIG_TEMPLATES]
    if invalid:
        available = ', '.join(ALGORITHM_CONFIG_TEMPLATES.keys())
        raise ValueError(f"无效算法: {', '.join(invalid)}。可用算法: {available}")
    return algorithms


# ============================================================================
# 场景默认配置
# ============================================================================

DEFAULT_SCENARIO_PARAMS = {
    'walker': {
        'num_planes': 6,
        'sats_per_plane': 5,
        'altitude_m': 500000,  # 500km
        'inclination_deg': 55.0,
        'phase_factor': 1
    },
    'ground_stations': [
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
    ],
    'direction_bounds': {
        'japan': {'lon_range': (129.0, 146.0), 'lat_range': (30.0, 46.0), 'count': 150, 'priority_range': (2, 5)},
        'korea': {'lon_range': (124.0, 131.0), 'lat_range': (33.0, 43.0), 'count': 150, 'priority_range': (2, 5)},
        'taiwan': {'lon_range': (119.0, 122.0), 'lat_range': (21.0, 26.0), 'count': 150, 'priority_range': (2, 5)},
        'philippines': {'lon_range': (117.0, 127.0), 'lat_range': (5.0, 20.0), 'count': 150, 'priority_range': (1, 4)},
        'myanmar': {'lon_range': (92.0, 102.0), 'lat_range': (10.0, 29.0), 'count': 200, 'priority_range': (1, 3)},
        'india': {'lon_range': (77.0, 95.0), 'lat_range': (20.0, 30.0), 'count': 200, 'priority_range': (1, 4)},
    }
}
