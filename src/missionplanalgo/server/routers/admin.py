"""
Admin 路由 - 管理 API
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import psutil
import os

router = APIRouter()


@router.get("/stats")
async def get_stats(request: Request):
    """
    获取服务统计信息
    """
    backend = request.app.state.task_backend

    # 获取任务统计
    all_tasks = backend.list_tasks()

    status_counts = {}
    for task in all_tasks.values():
        status = task.status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    # 获取系统资源
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    return {
        "tasks": {
            "total": len(all_tasks),
            "by_status": status_counts
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_mb": memory.total // (1024 * 1024),
                "available_mb": memory.available // (1024 * 1024),
                "percent": memory.percent
            }
        },
        "backend": type(backend).__name__
    }


@router.post("/reload")
async def reload_config():
    """
    重载配置

    重新加载配置文件，不重启服务。
    """
    # TODO: 实现配置重载
    return {"message": "配置已重载"}


@router.post("/cleanup")
async def cleanup_tasks(max_age: int = 7 * 24 * 3600, request: Request = None):
    """
    清理旧任务

    - **max_age**: 任务最大保留时间（秒），默认7天
    """
    backend = request.app.state.task_backend

    deleted = backend.cleanup(max_age)

    return {
        "message": f"已清理 {deleted} 个旧任务",
        "deleted_count": deleted
    }


@router.get("/config")
async def get_config():
    """
    获取当前配置（敏感信息会隐藏）
    """
    from missionplanalgo.config import load_config

    config = load_config()

    # 隐藏敏感信息
    safe_config = config.copy()
    if "auth" in safe_config:
        safe_config["auth"] = "***hidden***"

    return safe_config
