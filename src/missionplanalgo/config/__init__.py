"""
配置模块 - 配置管理

提供统一的配置加载、验证和管理功能。
"""

from typing import Optional, Dict, Any
from pathlib import Path
import os
import yaml


def get_config_dir() -> Path:
    """获取配置目录"""
    return Path.home() / ".config" / "mpa"


def get_cache_dir() -> Path:
    """获取缓存目录"""
    cache_dir = Path.home() / ".cache" / "mpa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置

    配置优先级（高到低）：
    1. 命令行参数指定的配置文件
    2. 本地配置文件 (./.mpa/config.yaml)
    3. 全局配置文件 (~/.config/mpa/config.yaml)
    4. 默认配置

    Args:
        config_path: 指定的配置文件路径

    Returns:
        配置字典
    """
    # 从默认配置开始
    config = _load_default_config()

    # 加载全局配置
    global_config = get_config_dir() / "config.yaml"
    if global_config.exists():
        with open(global_config) as f:
            global_data = yaml.safe_load(f)
            if global_data:
                _deep_update(config, global_data)

    # 加载本地配置
    local_config = Path.cwd() / ".mpa" / "config.yaml"
    if local_config.exists():
        with open(local_config) as f:
            local_data = yaml.safe_load(f)
            if local_data:
                _deep_update(config, local_data)

    # 加载指定的配置文件
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                override_data = yaml.safe_load(f)
                if override_data:
                    _deep_update(config, override_data)

    # 环境变量覆盖
    _apply_env_overrides(config)

    # 展开路径中的 ~
    _expand_paths(config)

    return config


def _load_default_config() -> Dict[str, Any]:
    """加载默认配置"""
    default_config_path = Path(__file__).parent / "default.yaml"

    if default_config_path.exists():
        with open(default_config_path) as f:
            return yaml.safe_load(f) or {}

    return {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """深度更新字典"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "MPA") -> None:
    """应用环境变量覆盖"""
    for key, value in os.environ.items():
        if key.startswith(f"{prefix}_"):
            # MPA_SERVER_PORT -> server.port
            parts = key[len(prefix) + 1:].lower().split('_')

            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            # 尝试解析值类型
            final_key = parts[-1]
            target[final_key] = _parse_env_value(value)


def _parse_env_value(value: str) -> Any:
    """解析环境变量值"""
    # 尝试作为 JSON 解析
    try:
        import json
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # 尝试布尔值
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False

    # 尝试整数
    try:
        return int(value)
    except ValueError:
        pass

    # 尝试浮点数
    try:
        return float(value)
    except ValueError:
        pass

    # 作为字符串返回
    return value


def _expand_paths(config: Dict[str, Any]) -> None:
    """展开配置中的路径"""
    path_keys = [
        ('task_backend', 'db_path'),
        ('cache', 'directory'),
        ('java_backend', 'jar_path'),
    ]

    for keys in path_keys:
        target = config
        for key in keys[:-1]:
            if key not in target:
                break
            target = target[key]
        else:
            final_key = keys[-1]
            if final_key in target and isinstance(target[final_key], str):
                target[final_key] = os.path.expanduser(target[final_key])


def save_config(config: Dict[str, Any], global_config: bool = True) -> None:
    """
    保存配置

    Args:
        config: 配置字典
        global_config: 是否保存为全局配置（False 则保存为本地配置）
    """
    if global_config:
        config_file = get_config_dir() / "config.yaml"
    else:
        config_file = Path.cwd() / ".mpa" / "config.yaml"

    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置有效性

    Args:
        config: 配置字典

    Raises:
        ConfigValidationError: 当配置无效时

    Examples:
        >>> config = load_config()
        >>> validate_config(config)  # 验证通过不抛出异常
    """
    errors = []

    # 验证调度器配置
    if "scheduler" in config:
        scheduler_config = config["scheduler"]

        # 验证默认算法
        if "default_algorithm" in scheduler_config:
            valid_algos = ['greedy', 'edd', 'spt', 'ga', 'sa', 'aco', 'pso', 'tabu']
            algo = scheduler_config["default_algorithm"]
            if algo not in valid_algos:
                errors.append(f"无效的默认算法: {algo}，必须是以下之一: {valid_algos}")

    # 验证任务后端配置
    if "task_backend" in config:
        backend_config = config["task_backend"]

        # 验证类型
        if "type" in backend_config:
            backend_type = backend_config["type"]
            if backend_type not in ["local", "celery"]:
                errors.append(f"无效的后端类型: {backend_type}，必须是 'local' 或 'celery'")

        # 验证 max_workers
        if "max_workers" in backend_config:
            max_workers = backend_config["max_workers"]
            if not isinstance(max_workers, int) or max_workers < 1:
                errors.append(f"max_workers 必须是正整数，当前值: {max_workers}")

        # Celery 后端验证
        if backend_config.get("type") == "celery":
            if not backend_config.get("broker_url"):
                errors.append("Celery 后端必须配置 broker_url")

    # 验证服务器配置
    if "server" in config:
        server_config = config["server"]

        # 验证端口
        if "port" in server_config:
            port = server_config["port"]
            if not isinstance(port, int) or not (1 <= port <= 65535):
                errors.append(f"服务器端口必须在 1-65535 范围内，当前值: {port}")

        # 验证主机
        if "host" in server_config:
            host = server_config["host"]
            if not isinstance(host, str) or not host:
                errors.append("服务器主机不能为空")

        # 验证日志级别
        if "log_level" in server_config:
            valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
            level = server_config["log_level"].lower()
            if level not in valid_levels:
                errors.append(f"无效的日志级别: {level}，必须是以下之一: {valid_levels}")

    # 验证缓存配置
    if "cache" in config:
        cache_config = config["cache"]

        # 验证 max_size 格式
        if "max_size" in cache_config:
            size_str = cache_config["max_size"]
            if isinstance(size_str, str):
                # 检查格式如 "10GB", "100MB"
                valid_suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
                has_valid_suffix = any(size_str.upper().endswith(s) for s in valid_suffixes)
                if not has_valid_suffix:
                    errors.append(f"缓存大小格式无效: {size_str}，示例: '10GB', '100MB'")

    # 验证 Java 后端配置
    if "java_backend" in config:
        java_config = config["java_backend"]

        # 验证内存格式
        if "memory" in java_config:
            memory = java_config["memory"]
            if isinstance(memory, str):
                if not (memory.endswith('g') or memory.endswith('G') or
                        memory.endswith('m') or memory.endswith('M')):
                    errors.append(f"Java 内存格式无效: {memory}，示例: '4g', '512m'")

    # 如果有错误，抛出异常
    if errors:
        raise ConfigValidationError("配置验证失败:\n  - " + "\n  - ".join(errors))


def load_and_validate_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载并验证配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        ConfigValidationError: 当配置无效时
    """
    config = load_config(config_path)
    validate_config(config)
    return config


__all__ = [
    "load_config",
    "save_config",
    "get_config_dir",
    "get_cache_dir",
    "validate_config",
    "load_and_validate_config",
    "ConfigValidationError",
]
