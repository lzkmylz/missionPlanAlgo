"""
任务后端模块

提供多种任务执行后端：
- LocalTaskBackend: 单机模式，开箱即用
- CeleryTaskBackend: 分布式模式，需要 Redis

使用工厂函数 create_task_backend 自动创建合适的后端。
"""

from typing import Optional, Dict, Any
import logging

from .base import TaskBackend, TaskStatus, TaskInfo
from .local import LocalTaskBackend

logger = logging.getLogger(__name__)

__all__ = [
    "TaskBackend",
    "TaskStatus",
    "TaskInfo",
    "LocalTaskBackend",
    "create_task_backend",
]


def create_task_backend(config: Optional[Dict[str, Any]] = None) -> TaskBackend:
    """
    创建任务后端

    根据配置自动选择合适的后端：
    - 如果配置了 broker_url，使用 CeleryTaskBackend
    - 否则使用 LocalTaskBackend（默认）

    Args:
        config: 配置字典，包含：
            - type: "local" 或 "celery"
            - max_workers: LocalBackend 的 worker 数
            - broker_url: Celery 的 broker URL
            - result_backend: Celery 的 result backend URL

    Returns:
        TaskBackend 实例

    Examples:
        # 单机模式（默认）
        backend = create_task_backend()

        # 单机模式，指定 worker 数
        backend = create_task_backend({
            "type": "local",
            "max_workers": 8
        })

        # 分布式模式（需要 Redis）
        backend = create_task_backend({
            "type": "celery",
            "broker_url": "redis://localhost:6379/0"
        })
    """
    config = config or {}
    backend_type = config.get("type", "local")

    if backend_type == "celery" or config.get("broker_url"):
        # 使用 Celery 后端
        try:
            from .celery import CeleryTaskBackend
        except ImportError as e:
            raise ImportError(
                f"无法导入 Celery 后端: {e}\n"
                "请安装 Celery 依赖: pip install missionplanalgo[celery]"
            )

        broker_url = config.get("broker_url")
        if not broker_url:
            raise ValueError("Celery 后端需要配置 broker_url")

        return CeleryTaskBackend(
            broker_url=broker_url,
            result_backend=config.get("result_backend")
        )

    else:
        # 使用单机后端（默认）
        return LocalTaskBackend(
            max_workers=config.get("max_workers", 4),
            db_path=config.get("db_path"),
            cache_dir=config.get("cache_dir")
        )
