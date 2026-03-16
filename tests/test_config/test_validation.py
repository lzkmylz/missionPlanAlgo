"""
测试配置验证
"""

import pytest
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from missionplanalgo.config import (
    validate_config,
    ConfigValidationError,
)


def test_validate_empty_config():
    """验证空配置应该通过"""
    validate_config({})  # 不应该抛出异常


def test_validate_valid_algorithm():
    """验证有效的算法配置"""
    config = {
        'scheduler': {
            'default_algorithm': 'greedy'
        }
    }
    validate_config(config)  # 不应该抛出异常


def test_validate_invalid_algorithm():
    """验证无效的算法配置"""
    config = {
        'scheduler': {
            'default_algorithm': 'invalid_algorithm'
        }
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert '无效的默认算法' in str(exc_info.value)


def test_validate_backend_type():
    """验证后端类型"""
    config = {'task_backend': {'type': 'local'}}
    validate_config(config)

    config = {'task_backend': {'type': 'celery', 'broker_url': 'redis://localhost:6379/0'}}
    validate_config(config)

    config = {'task_backend': {'type': 'invalid'}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert '无效的后端类型' in str(exc_info.value)


def test_validate_max_workers():
    """验证 max_workers"""
    config = {'task_backend': {'max_workers': 4}}
    validate_config(config)

    config = {'task_backend': {'max_workers': 0}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert 'max_workers' in str(exc_info.value)

    config = {'task_backend': {'max_workers': -1}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)


def test_validate_celery_without_broker():
    """验证 Celery 后端必须有 broker_url"""
    config = {
        'task_backend': {
            'type': 'celery'
            # 缺少 broker_url
        }
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert 'broker_url' in str(exc_info.value)


def test_validate_valid_port():
    """验证有效的端口"""
    config = {'server': {'port': 8000}}
    validate_config(config)

    config = {'server': {'port': 1}}
    validate_config(config)

    config = {'server': {'port': 65535}}
    validate_config(config)


def test_validate_invalid_port():
    """验证无效的端口"""
    config = {'server': {'port': 0}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert '端口' in str(exc_info.value)

    config = {'server': {'port': 65536}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)

    config = {'server': {'port': -1}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)


def test_validate_log_level():
    """验证日志级别"""
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
    for level in valid_levels:
        config = {'server': {'log_level': level}}
        validate_config(config)

    config = {'server': {'log_level': 'invalid'}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert '日志级别' in str(exc_info.value)


def test_validate_cache_size():
    """验证缓存大小格式"""
    valid_sizes = ['100MB', '10GB', '1TB', '500B', '50KB']
    for size in valid_sizes:
        config = {'cache': {'max_size': size}}
        validate_config(config)

    config = {'cache': {'max_size': 'invalid'}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert '缓存大小' in str(exc_info.value)


def test_validate_java_memory():
    """验证 Java 内存配置"""
    valid_memories = ['4g', '4G', '512m', '512M', '1024m']
    for mem in valid_memories:
        config = {'java_backend': {'memory': mem}}
        validate_config(config)

    config = {'java_backend': {'memory': 'invalid'}}
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    assert '内存' in str(exc_info.value)


def test_validate_multiple_errors():
    """验证多个错误会一起报告"""
    config = {
        'scheduler': {'default_algorithm': 'invalid'},
        'server': {'port': 99999},
        'task_backend': {'max_workers': -1}
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    error_msg = str(exc_info.value)
    assert '无效的默认算法' in error_msg
    assert '端口' in error_msg
    assert 'max_workers' in error_msg
