"""
Orekit配置模块

管理Orekit Java后端的配置参数，包括JVM设置、数据文件路径、
传播器参数和摄动力模型配置。
"""

import os
from typing import Dict, Any, Optional


def _get_project_root() -> str:
    """获取项目根目录（基于当前文件位置）"""
    # 当前文件路径: core/orbit/visibility/orekit_config.py
    # 当前目录: core/orbit/visibility
    # 上溯3层到达项目根目录: missionPlanAlgo
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    # 从 core/orbit/visibility 上溯3层到项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    return project_root


def _get_default_classpath() -> list[str]:
    """获取默认的JVM classpath（使用相对路径）"""
    project_root = _get_project_root()
    java_dir = os.path.join(project_root, 'java')
    return [
        os.path.join(java_dir, 'lib', 'orekit-12.0.jar'),
        os.path.join(java_dir, 'lib', 'hipparchus-core-3.0.jar'),
        os.path.join(java_dir, 'lib', 'hipparchus-geometry-3.0.jar'),
        os.path.join(java_dir, 'lib', 'hipparchus-ode-3.0.jar'),
        os.path.join(java_dir, 'lib', 'json.jar'),
        os.path.join(java_dir, 'classes'),
    ]


def _get_default_data_dir() -> str:
    """获取默认的Orekit数据目录"""
    # 优先从环境变量获取
    env_dir = os.environ.get('OREKIT_DATA_DIR')
    if env_dir and os.path.exists(env_dir):
        return env_dir

    # 检查用户home目录下的orekit-data
    home_data_dir = os.path.expanduser('~/orekit-data')
    if os.path.exists(home_data_dir):
        return home_data_dir

    # 检查项目目录下的orekit-data
    project_root = _get_project_root()
    project_data_dir = os.path.join(project_root, 'orekit-data')
    if os.path.exists(project_data_dir):
        return project_data_dir

    # 最后回退到系统默认路径
    return '/usr/local/share/orekit'


def _get_java_home() -> str:
    """获取Java安装目录"""
    # 优先使用环境变量
    env_java_home = os.environ.get('JAVA_HOME')
    if env_java_home:
        return env_java_home
    # 检查常见安装位置
    common_paths = [
        os.path.expanduser('~/jdk-17.0.9+9'),
        '/usr/lib/jvm/java-17',
        '/usr/lib/jvm/java-17-openjdk',
        '/usr/lib/jvm/java-17-openjdk-amd64',
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    # 回退到默认值
    return '/usr/lib/jvm/java-17'


# Orekit默认配置（使用动态路径）
DEFAULT_OREKIT_CONFIG = {
    'jvm': {
        'java_home': _get_java_home(),
        'classpath': _get_default_classpath(),
        'max_memory': '2g',
    },
    'data': {
        'root_dir': _get_default_data_dir(),
        'iers_dir': 'IERS',
        'gravity_dir': 'EGM96',
        'ephemeris_dir': 'DE440',
    },
    'propagator': {
        'integrator': 'DormandPrince853',  # RK78
        'min_step': 0.001,  # s
        'max_step': 300.0,  # s
        'position_tolerance': 10.0,  # m
    },
    'perturbations': {
        'earth_gravity': {
            'enabled': True,
            'model': 'EGM2008',
            'degree': 90,
            'order': 90,
        },
        'drag': {
            'enabled': True,
            'model': 'NRLMSISE00',
            'cd': 2.2,
            'area': 10.0,  # m²
        },
        'solar_radiation': {
            'enabled': True,
            'cr': 1.5,
            'area': 10.0,  # m²
        },
        'third_body': {
            'enabled': True,
            'bodies': ['SUN', 'MOON'],
        },
        'relativity': {
            'enabled': True,
        },
    },
}


def get_orekit_data_dir() -> str:
    """
    获取Orekit数据目录

    优先从环境变量OREKIT_DATA_DIR获取，否则使用默认配置中的路径。

    Returns:
        str: Orekit数据目录路径
    """
    env_dir = os.environ.get('OREKIT_DATA_DIR')
    if env_dir:
        return env_dir
    return DEFAULT_OREKIT_CONFIG['data']['root_dir']


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        Dict[str, Any]: 合并后的新字典
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并子字典
            result[key] = _deep_merge(result[key], value)
        else:
            # 直接覆盖或添加
            result[key] = value

    return result


def merge_config(custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    合并自定义配置与默认配置

    使用深度合并，保留未修改的默认值。

    Args:
        custom_config: 自定义配置字典，可以为None

    Returns:
        Dict[str, Any]: 合并后的配置字典
    """
    if custom_config is None:
        return DEFAULT_OREKIT_CONFIG.copy()

    if not custom_config:
        return DEFAULT_OREKIT_CONFIG.copy()

    return _deep_merge(DEFAULT_OREKIT_CONFIG, custom_config)


def get_jvm_classpath(config: Optional[Dict[str, Any]] = None) -> str:
    """
    获取JVM classpath字符串

    Args:
        config: 配置字典，如果为None则使用默认配置

    Returns:
        str: classpath字符串，使用系统路径分隔符连接
    """
    if config is None:
        config = DEFAULT_OREKIT_CONFIG

    classpath_list = config.get('jvm', {}).get('classpath', [])
    return os.pathsep.join(classpath_list)


def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    验证配置是否有效

    Args:
        config: 配置字典

    Returns:
        tuple[bool, list[str]]: (是否有效, 错误信息列表)
    """
    errors = []

    # 验证JVM配置
    if 'jvm' not in config:
        errors.append("缺少jvm配置")
    else:
        jvm_config = config['jvm']
        if 'classpath' not in jvm_config:
            errors.append("缺少jvm.classpath配置")
        elif not isinstance(jvm_config['classpath'], list):
            errors.append("jvm.classpath必须是列表")

        if 'max_memory' not in jvm_config:
            errors.append("缺少jvm.max_memory配置")

    # 验证数据配置
    if 'data' not in config:
        errors.append("缺少data配置")
    else:
        data_config = config['data']
        required_data_keys = ['root_dir', 'iers_dir', 'gravity_dir', 'ephemeris_dir']
        for key in required_data_keys:
            if key not in data_config:
                errors.append(f"缺少data.{key}配置")

    # 验证传播器配置
    if 'propagator' not in config:
        errors.append("缺少propagator配置")
    else:
        prop_config = config['propagator']
        if 'min_step' in prop_config and prop_config['min_step'] <= 0:
            errors.append("propagator.min_step必须大于0")
        if 'max_step' in prop_config and prop_config['max_step'] <= 0:
            errors.append("propagator.max_step必须大于0")
        if ('min_step' in prop_config and 'max_step' in prop_config and
            prop_config['min_step'] >= prop_config['max_step']):
            errors.append("propagator.min_step必须小于max_step")

    # 验证摄动力配置
    if 'perturbations' not in config:
        errors.append("缺少perturbations配置")

    return len(errors) == 0, errors
