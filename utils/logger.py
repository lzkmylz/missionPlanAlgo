"""
日志管理模块

功能：
- 支持文件日志（按日期轮转）
- 支持控制台日志
- 支持结构化日志（JSON格式）
- 支持多logger实例
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, Union


class LoggerConfigError(Exception):
    """日志配置错误"""
    pass


class JsonFormatter(logging.Formatter):
    """JSON格式日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        # 添加额外字段
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """文本格式日志格式化器"""

    def __init__(self, fmt: Optional[str] = None):
        super().__init__(
            fmt=fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class Logger:
    """
    日志管理器

    支持多种输出格式和处理器
    """

    # 日志级别映射
    LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    def __init__(self, name: str, level: str = "INFO"):
        """
        初始化日志管理器

        Args:
            name: Logger名称
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            LoggerConfigError: 无效的日志级别
        """
        if level not in self.LEVEL_MAP:
            raise LoggerConfigError(f"无效的日志级别: {level}. 有效值: {list(self.LEVEL_MAP.keys())}")

        self.name = name
        self.level = level
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.LEVEL_MAP[level])

        # 清除默认处理器（避免重复输出）
        self._logger.handlers = []
        self._logger.propagate = False

    def add_console_handler(self, format: str = "text") -> "Logger":
        """
        添加控制台处理器

        Args:
            format: 格式类型 ("text", "json")

        Returns:
            Logger: 自身，支持链式调用
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.LEVEL_MAP[self.level])

        if format == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(TextFormatter())

        self._logger.addHandler(handler)
        return self

    def add_file_handler(
        self,
        path: str,
        rotation: str = "none",
        format: str = "text",
        backup_count: int = 7
    ) -> "Logger":
        """
        添加文件处理器

        Args:
            path: 日志文件路径
            rotation: 轮转策略 ("none", "daily", "hourly")
            format: 格式类型 ("text", "json")
            backup_count: 保留的备份文件数量

        Returns:
            Logger: 自身，支持链式调用
        """
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if rotation == "daily":
            handler = TimedRotatingFileHandler(
                path,
                when="midnight",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8"
            )
        elif rotation == "hourly":
            handler = TimedRotatingFileHandler(
                path,
                when="H",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8"
            )
        else:  # none
            handler = logging.FileHandler(path, encoding="utf-8")

        handler.setLevel(self.LEVEL_MAP[self.level])

        if format == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(TextFormatter())

        self._logger.addHandler(handler)
        return self

    def _log(self, level: int, message: Union[str, Dict[str, Any]]) -> None:
        """
        内部日志方法

        Args:
            level: 日志级别
            message: 日志消息（字符串或字典）
        """
        if isinstance(message, dict):
            # 结构化日志
            extra = {"extra_data": message}
            msg = message.get("message", "")
            self._logger.log(level, msg, extra=extra)
        else:
            # 普通文本日志
            self._logger.log(level, message)

    def debug(self, message: Union[str, Dict[str, Any]]) -> None:
        """
        记录DEBUG级别日志

        Args:
            message: 日志消息
        """
        self._log(logging.DEBUG, message)

    def info(self, message: Union[str, Dict[str, Any]]) -> None:
        """
        记录INFO级别日志

        Args:
            message: 日志消息
        """
        self._log(logging.INFO, message)

    def warning(self, message: Union[str, Dict[str, Any]]) -> None:
        """
        记录WARNING级别日志

        Args:
            message: 日志消息
        """
        self._log(logging.WARNING, message)

    def error(self, message: Union[str, Dict[str, Any]]) -> None:
        """
        记录ERROR级别日志

        Args:
            message: 日志消息
        """
        self._log(logging.ERROR, message)

    def critical(self, message: Union[str, Dict[str, Any]]) -> None:
        """
        记录CRITICAL级别日志

        Args:
            message: 日志消息
        """
        self._log(logging.CRITICAL, message)

    def exception(self, message: Union[str, Dict[str, Any]]) -> None:
        """
        记录异常信息（带堆栈跟踪）

        Args:
            message: 日志消息
        """
        if isinstance(message, dict):
            extra = {"extra_data": message}
            msg = message.get("message", "")
            self._logger.exception(msg, extra=extra)
        else:
            self._logger.exception(message)

    def set_level(self, level: str) -> None:
        """
        设置日志级别

        Args:
            level: 日志级别
        """
        if level not in self.LEVEL_MAP:
            raise LoggerConfigError(f"无效的日志级别: {level}")

        self.level = level
        self._logger.setLevel(self.LEVEL_MAP[level])

        # 更新所有处理器的级别
        for handler in self._logger.handlers:
            handler.setLevel(self.LEVEL_MAP[level])
