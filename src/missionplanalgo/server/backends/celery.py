"""
分布式任务后端（可选）

使用 Celery + Redis 实现分布式任务队列。

此模块使用延迟导入，只有用户安装并配置 Celery 时才会加载。
"""

from typing import Any, Callable, Dict, Optional
from datetime import datetime
import logging
import pickle
import base64

from .base import TaskBackend, TaskStatus, TaskInfo

logger = logging.getLogger(__name__)


class CeleryTaskBackend(TaskBackend):
    """
    分布式任务后端

    使用 Celery + Redis 实现，支持多机分布式部署。

    需要手动安装依赖:
        pip install missionplanalgo[celery]

    或:
        pip install celery redis
    """

    def __init__(self, broker_url: str, result_backend: Optional[str] = None):
        """
        初始化 Celery 任务后端

        Args:
            broker_url: Celery broker URL, e.g., "redis://localhost:6379/0"
            result_backend: Celery result backend URL, 默认使用 broker
        """
        # 延迟导入，避免强制依赖
        try:
            from celery import Celery
        except ImportError:
            raise ImportError(
                "Celery 未安装。请运行: pip install celery redis"
            )

        self.broker_url = broker_url
        self.result_backend = result_backend or broker_url

        # 创建 Celery 应用
        self.app = Celery(
            'missionplanalgo',
            broker=broker_url,
            backend=self.result_backend,
        )

        # 配置
        self.app.conf.update(
            task_serializer='pickle',
            accept_content=['pickle', 'json'],
            result_serializer='pickle',
            task_track_started=True,
            task_time_limit=3600,  # 1小时超时
            worker_prefetch_multiplier=1,
        )

        # 动态创建任务
        self._task_wrapper = self.app.task(self._execute_task, bind=True)

        logger.info(f"CeleryTaskBackend initialized with broker: {broker_url}")

    @staticmethod
    def _execute_task(self, func_bytes: str, args_bytes: str, kwargs_bytes: str) -> str:
        """
        Celery 任务执行包装器

        Args:
            func_bytes: base64编码的pickle序列化函数
            args_bytes: base64编码的args
            kwargs_bytes: base64编码的kwargs

        Returns:
            base64编码的结果
        """
        import pickle
        import base64

        # 反序列化
        func = pickle.loads(base64.b64decode(func_bytes))
        args = pickle.loads(base64.b64decode(args_bytes))
        kwargs = pickle.loads(base64.b64decode(kwargs_bytes))

        # 执行
        result = func(*args, **kwargs)

        # 序列化结果
        return base64.b64encode(pickle.dumps(result)).decode()

    def _serialize_callable(self, func: Callable) -> str:
        """序列化可调用对象"""
        try:
            return base64.b64encode(pickle.dumps(func)).decode()
        except pickle.PickleError as e:
            raise ValueError(f"无法序列化函数: {e}")

    def submit(self, func: Callable, *args, **kwargs) -> str:
        """提交任务"""
        # 序列化
        func_bytes = self._serialize_callable(func)
        args_bytes = base64.b64encode(pickle.dumps(args)).decode()
        kwargs_bytes = base64.b64encode(pickle.dumps(kwargs)).decode()

        # 提交到 Celery
        result = self._task_wrapper.delay(func_bytes, args_bytes, kwargs_bytes)

        return result.id

    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=self.app)

        # 映射 Celery 状态到我们的状态
        status_map = {
            'PENDING': TaskStatus.PENDING,
            'STARTED': TaskStatus.RUNNING,
            'SUCCESS': TaskStatus.COMPLETED,
            'FAILURE': TaskStatus.FAILED,
            'RETRY': TaskStatus.QUEUED,
            'REVOKED': TaskStatus.CANCELLED,
        }

        # 获取任务信息
        task_info = TaskInfo(
            task_id=task_id,
            status=status_map.get(result.state, TaskStatus.PENDING),
            created_at=datetime.now(),  # Celery 不直接提供创建时间
        )

        # 如果完成，获取结果
        if result.ready():
            if result.successful():
                task_info.status = TaskStatus.COMPLETED
                result_data = result.get()
                if result_data:
                    task_info.result = base64.b64decode(result_data)
            else:
                task_info.status = TaskStatus.FAILED
                task_info.error = str(result.result)

        return task_info

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果（阻塞等待）"""
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=self.app)

        try:
            data = result.get(timeout=timeout, propagate=True)
            if data:
                return pickle.loads(base64.b64decode(data))
            return None
        except Exception as e:
            raise Exception(f"任务执行失败: {e}")

    def cancel(self, task_id: str) -> bool:
        """取消任务"""
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=self.app)
        result.revoke(terminate=True)
        return True

    def list_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, TaskInfo]:
        """列出任务

        Note: Celery 没有内置的任务列表功能，需要配合 Flower 或自定义监控。
        这里返回空字典，建议通过 Flower 查看任务状态。
        """
        logger.warning(
            "Celery backend 不支持 list_tasks。"
            "请使用 Flower (celery -A mpa flower) 查看任务状态。"
        )
        return {}

    def inspect_workers(self) -> Dict[str, Any]:
        """检查 Worker 状态"""
        inspector = self.app.control.inspect()

        return {
            'active': inspector.active(),
            'scheduled': inspector.scheduled(),
            'reserved': inspector.reserved(),
            'stats': inspector.stats(),
        }
