"""
Config 命令 - 配置管理
"""

import click
import os
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def _get_config_dir():
    """获取配置目录"""
    config_dir = Path.home() / ".config" / "mpa"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_config_file(global_config=True):
    """获取配置文件路径"""
    if global_config:
        return _get_config_dir() / "config.yaml"
    else:
        return Path.cwd() / ".mpa" / "config.yaml"


@click.group(name="config")
def config_cmd():
    """配置管理命令"""
    pass


@config_cmd.command()
@click.option('--global', 'global_config', is_flag=True, help='创建全局配置')
@click.option('--local', 'local_config', is_flag=True, help='创建本地配置')
@click.option('--force', is_flag=True, help='覆盖现有配置')
def init(global_config, local_config, force):
    """初始化配置文件"""

    if local_config:
        config_file = _get_config_file(global_config=False)
    else:
        config_file = _get_config_file(global_config=True)

    if config_file.exists() and not force:
        console.print(f"[yellow]配置文件已存在: {config_file}[/yellow]")
        console.print("使用 --force 覆盖")
        return

    # 创建目录
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # 默认配置
    default_config = {
        "app": {
            "name": "Mission Planning Algorithm",
            "version": "1.0.0"
        },
        "scheduler": {
            "default_algorithm": "greedy",
            "enable_frequency": True,
            "enable_downlink": True
        },
        "task_backend": {
            "type": "local",
            "max_workers": 4
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info"
        },
        "java_backend": {
            "enabled": True,
            "memory": "4g"
        },
        "cache": {
            "directory": "~/.cache/mpa",
            "max_size": "10GB"
        }
    }

    # 写入配置
    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    console.print(f"[green]配置文件已创建: {config_file}[/green]")


@config_cmd.command()
@click.argument('key')
def get(key):
    """获取配置项"""

    # 尝试加载配置
    config = _load_config()

    # 按点分隔查找
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            console.print(f"[red]配置项不存在: {key}[/red]")
            return

    console.print(f"{key} = {value}")


@config_cmd.command()
@click.argument('key')
@click.argument('value')
def set(key, value):
    """设置配置项"""

    config_file = _get_config_file(global_config=True)

    # 加载现有配置
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # 按点分隔设置
    keys = key.split('.')
    target = config
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]

    # 尝试解析值类型
    try:
        # 尝试作为 JSON 解析
        import json
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        # 作为字符串
        parsed_value = value

    target[keys[-1]] = parsed_value

    # 保存配置
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    console.print(f"[green]已设置: {key} = {parsed_value}[/green]")


@config_cmd.command()
def list():
    """列出所有配置"""

    config = _load_config()

    table = Table(title="MPA 配置")
    table.add_column("键", style="cyan")
    table.add_column("值", style="green")

    def add_items(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                add_items(v, key)
            else:
                table.add_row(key, str(v))

    add_items(config)
    console.print(table)


@config_cmd.command()
def validate():
    """验证配置"""
    console.print("[blue]验证配置...[/blue]")

    config = _load_config()

    # 基本验证
    errors = []

    # 检查调度器配置
    if "scheduler" in config:
        algo = config["scheduler"].get("default_algorithm")
        valid_algos = ["greedy", "edd", "spt", "ga", "sa", "aco", "pso", "tabu"]
        if algo and algo not in valid_algos:
            errors.append(f"无效的默认算法: {algo}")

    # 检查后端配置
    if "task_backend" in config:
        backend_type = config["task_backend"].get("type")
        if backend_type not in ["local", "celery"]:
            errors.append(f"无效的后端类型: {backend_type}")

        if backend_type == "celery":
            broker = config["task_backend"].get("broker_url")
            if not broker:
                errors.append("Celery 后端需要配置 broker_url")

    if errors:
        console.print("[red]配置验证失败:[/red]")
        for error in errors:
            console.print(f"  - {error}")
    else:
        console.print("[green]配置验证通过[/green]")


def _load_config():
    """加载配置（本地优先）"""
    config = {}

    # 加载全局配置
    global_config = _get_config_file(global_config=True)
    if global_config.exists():
        with open(global_config) as f:
            config = yaml.safe_load(f) or {}

    # 加载本地配置（覆盖全局）
    local_config = _get_config_file(global_config=False)
    if local_config.exists():
        with open(local_config) as f:
            local = yaml.safe_load(f) or {}
            config.update(local)

    return config
