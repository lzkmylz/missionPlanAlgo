"""
单机版任务后端

使用 ProcessPoolExecutor + SQLite 实现，无需外部依赖。
"""

import os
import json
import sqlite3
import uuid
import pickle
import base64
from concurrent.futures import ProcessPoolExecutor, Future
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set
from threading import Lock
import logging

from .base import TaskBackend, TaskStatus, TaskInfo

logger = logging.getLogger(__name__)


def _run_task(task_id: str, func_bytes: bytes, args_bytes: bytes, kwargs_bytes: bytes,
              db_path: str) -> None:
    """
    在子进程中执行的任务包装器

    Args:
        task_id: 任务ID
        func_bytes: pickle序列化的函数
        args_bytes: pickle序列化的args
        kwargs_bytes: pickle序列化的kwargs
        db_path: 数据库路径
    """
    try:
        # 反序列化
        func = pickle.loads(func_bytes)
        args = pickle.loads(args_bytes)
        kwargs = pickle.loads(kwargs_bytes)

        # 更新状态为运行中
        _update_task_in_db(db_path, task_id, TaskStatus.RUNNING,
                          started_at=datetime.now())

        # 执行函数
        result = func(*args, **kwargs)

        # 序列化结果
        result_bytes = base64.b64encode(pickle.dumps(result)).decode()

        # 更新状态为完成
        _update_task_in_db(db_path, task_id, TaskStatus.COMPLETED,
                          completed_at=datetime.now(),
                          result=result_bytes)

    except Exception as e:
        # 更新状态为失败
        _update_task_in_db(db_path, task_id, TaskStatus.FAILED,
                          completed_at=datetime.now(),
                          error=str(e))


def _update_task_in_db(db_path: str, task_id: str, status: TaskStatus,
                       started_at: Optional[datetime] = None,
                       completed_at: Optional[datetime] = None,
                       result: Optional[str] = None,
                       error: Optional[str] = None):
    """在数据库中更新任务状态"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        fields = ["status = ?"]
        values = [status.value]

        if started_at:
            fields.append("started_at = ?")
            values.append(started_at.isoformat())
        if completed_at:
            fields.append("completed_at = ?")
            values.append(completed_at.isoformat())
        if result is not None:
            fields.append("result = ?")
            values.append(result)
        if error is not None:
            fields.append("error = ?")
            values.append(error)

        values.append(task_id)

        sql = f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?"
        cursor.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


class SQLiteTaskDB:
    """SQLite任务数据库"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    progress INTEGER DEFAULT 0,
                    message TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_task(self, task_id: str, status: TaskStatus,
                  created_at: datetime) -> None:
        """保存新任务"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tasks (task_id, status, created_at) VALUES (?, ?, ?)",
                (task_id, status.value, created_at.isoformat())
            )
            conn.commit()
        finally:
            conn.close()

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_taskinfo(row)
        finally:
            conn.close()

    def list_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, TaskInfo]:
        """列出任务"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            if status:
                cursor.execute(
                    "SELECT * FROM tasks WHERE status = ?",
                    (status.value,)
                )
            else:
                cursor.execute("SELECT * FROM tasks")

            rows = cursor.fetchall()
            return {row[0]: self._row_to_taskinfo(row) for row in rows}
        finally:
            conn.close()

    def update_progress(self, task_id: str, progress: int, message: Optional[str] = None):
        """更新任务进度"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            if message:
                cursor.execute(
                    "UPDATE tasks SET progress = ?, message = ? WHERE task_id = ?",
                    (progress, message, task_id)
                )
            else:
                cursor.execute(
                    "UPDATE tasks SET progress = ? WHERE task_id = ?",
                    (progress, task_id)
                )
            conn.commit()
        finally:
            conn.close()

    def delete_old_tasks(self, max_age_seconds: int) -> int:
        """删除旧任务"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cutoff = (datetime.now() - __import__('datetime').timedelta(seconds=max_age_seconds)).isoformat()
            cursor.execute(
                "DELETE FROM tasks WHERE created_at < ?",
                (cutoff,)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted
        finally:
            conn.close()

    def _row_to_taskinfo(self, row) -> TaskInfo:
        """将数据库行转换为 TaskInfo"""
        return TaskInfo(
            task_id=row[0],
            status=TaskStatus(row[1]),
            created_at=datetime.fromisoformat(row[2]),
            started_at=datetime.fromisoformat(row[3]) if row[3] else None,
            completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
            result=base64.b64decode(row[5]) if row[5] else None,
            error=row[6],
            progress=row[7] or 0,
            message=row[8],
        )


class LocalTaskBackend(TaskBackend):
    """
    单机版任务后端

    使用 ProcessPoolExecutor 在本地进程池中执行任务，
    使用 SQLite 存储任务状态。

    无需 Redis 等外部依赖，开箱即用。
    """

    def __init__(self, max_workers: int = 4,
                 db_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        初始化单机任务后端

        Args:
            max_workers: 最大工作进程数
            db_path: 数据库路径，默认 ~/.cache/mpa/tasks.db
            cache_dir: 缓存目录
        """
        self.max_workers = max_workers

        # 设置数据库路径
        if db_path is None:
            cache_dir = cache_dir or os.path.expanduser("~/.cache/mpa")
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "tasks.db")

        self.db_path = db_path
        self.db = SQLiteTaskDB(db_path)

        # 创建进程池
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

        # 跟踪已提交的任务
        self._futures: Dict[str, Future] = {}
        self._lock = Lock()

        logger.info(f"LocalTaskBackend initialized with {max_workers} workers, db: {db_path}")

    def submit(self, func: Callable, *args, **kwargs) -> str:
        """提交任务"""
        task_id = str(uuid.uuid4())
        created_at = datetime.now()

        # 保存到数据库
        self.db.save_task(task_id, TaskStatus.PENDING, created_at)

        # 序列化函数和参数
        try:
            func_bytes = pickle.dumps(func)
            args_bytes = pickle.dumps(args)
            kwargs_bytes = pickle.dumps(kwargs)
        except pickle.PickleError as e:
            raise ValueError(f"无法序列化任务函数或参数: {e}")

        # 提交到进程池
        future = self.executor.submit(
            _run_task,
            task_id,
            func_bytes,
            args_bytes,
            kwargs_bytes,
            self.db_path
        )

        # 跟踪 future
        with self._lock:
            self._futures[task_id] = future

        # 添加完成回调以清理
        future.add_done_callback(lambda f: self._on_task_done(task_id))

        return task_id

    def _on_task_done(self, task_id: str):
        """任务完成回调"""
        with self._lock:
            self._futures.pop(task_id, None)

    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        return self.db.get_task(task_id)

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果（阻塞等待）"""
        # 首先检查数据库状态
        task_info = self.db.get_task(task_id)
        if task_info is None:
            raise ValueError(f"任务不存在: {task_id}")

        # 如果已完成，直接返回结果
        if task_info.status == TaskStatus.COMPLETED:
            if task_info.result:
                return pickle.loads(task_info.result)
            return None

        if task_info.status == TaskStatus.FAILED:
            raise Exception(task_info.error or "任务执行失败")

        # 等待 future 完成
        with self._lock:
            future = self._futures.get(task_id)

        if future:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                raise Exception(f"任务执行失败: {e}")

        # 再次查询结果
        task_info = self.db.get_task(task_id)
        if task_info.status == TaskStatus.COMPLETED:
            if task_info.result:
                return pickle.loads(task_info.result)
            return None
        elif task_info.status == TaskStatus.FAILED:
            raise Exception(task_info.error or "任务执行失败")
        else:
            raise TimeoutError("等待任务结果超时")

    def cancel(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            future = self._futures.get(task_id)

        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                _update_task_in_db(self.db_path, task_id, TaskStatus.CANCELLED,
                                 completed_at=datetime.now())
            return cancelled

        return False

    def list_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, TaskInfo]:
        """列出任务"""
        return self.db.list_tasks(status)

    def cleanup(self, max_age: Optional[int] = None) -> int:
        """清理旧任务"""
        if max_age is None:
            # 默认清理7天前的任务
            max_age = 7 * 24 * 3600
        return self.db.delete_old_tasks(max_age)

    def shutdown(self, wait: bool = True):
        """关闭后端"""
        self.executor.shutdown(wait=wait)
