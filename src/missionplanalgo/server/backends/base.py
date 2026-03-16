"""
任务后端抽象基类

定义统一的任务管理接口，支持单机模式和分布式模式。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "message": self.message,
        }


class TaskBackend(ABC):
    """任务后端抽象基类"""

    @abstractmethod
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """
        提交任务

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            任务ID
        """
        pass

    @abstractmethod
    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务信息，如果不存在返回 None
        """
        pass

    @abstractmethod
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取任务结果（阻塞等待）

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            任务执行结果

        Raises:
            TimeoutError: 超时
            Exception: 任务执行失败
        """
        pass

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功取消
        """
        pass

    @abstractmethod
    def list_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, TaskInfo]:
        """
        列出任务

        Args:
            status: 按状态过滤

        Returns:
            任务ID到任务信息的映射
        """
        pass

    def cleanup(self, max_age: Optional[int] = None) -> int:
        """
        清理旧任务

        Args:
            max_age: 最大保留时间（秒）

        Returns:
            清理的任务数量
        """
        # 默认实现：不清理
        return 0
