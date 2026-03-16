"""
测试任务后端
"""

import pytest
import time
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from missionplanalgo.server.backends import (
    create_task_backend,
    LocalTaskBackend,
    TaskStatus,
)


# 定义在模块级别的函数，可以被 pickle
def simple_task():
    return {"result": "success"}


def compute_task(x, y):
    return x + y


def long_task():
    time.sleep(10)
    return "done"


def test_create_local_backend():
    """测试创建单机后端"""
    backend = create_task_backend({'type': 'local', 'max_workers': 2})
    assert isinstance(backend, LocalTaskBackend)


def test_local_backend_submit():
    """测试单机后端提交任务"""
    backend = LocalTaskBackend(max_workers=2)

    task_id = backend.submit(simple_task)
    assert task_id is not None

    # 等待任务完成
    time.sleep(0.5)

    status = backend.get_status(task_id)
    assert status is not None
    assert status.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING]

    backend.shutdown()


def test_local_backend_result():
    """测试单机后端获取结果"""
    backend = LocalTaskBackend(max_workers=2)

    task_id = backend.submit(compute_task, 1, 2)

    # 获取结果（阻塞）
    result = backend.get_result(task_id, timeout=5)
    assert result == 3

    backend.shutdown()


def test_local_backend_cancel():
    """测试单机后端取消任务"""
    backend = LocalTaskBackend(max_workers=2)

    task_id = backend.submit(long_task)

    # 取消任务
    cancelled = backend.cancel(task_id)
    # 注意：由于进程池限制，取消可能不会立即生效

    backend.shutdown()
