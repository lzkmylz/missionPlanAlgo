"""
Logger模块的单元测试

TDD流程：
1. 先写测试（RED）
2. 运行测试 - 验证失败
3. 实现代码（GREEN）
4. 运行测试 - 验证通过
5. 重构（IMPROVE）
"""

import json
import logging
import os
import tempfile
from unittest import TestCase, mock

import pytest


class TestLogger(TestCase):
    """Logger测试类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_temp_log_path(self) -> str:
        """获取临时日志文件路径"""
        return os.path.join(self.temp_dir, "test.log")

    # ==================== 基础功能测试 ====================

    def test_logger_init(self):
        """测试Logger初始化"""
        from utils.logger import Logger

        logger = Logger("test_logger")
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, "INFO")

    def test_logger_init_with_level(self):
        """测试Logger初始化带级别"""
        from utils.logger import Logger

        logger = Logger("test_logger", level="DEBUG")
        self.assertEqual(logger.level, "DEBUG")

    def test_add_console_handler(self):
        """测试添加控制台处理器"""
        from utils.logger import Logger

        logger = Logger("test_logger")
        logger.add_console_handler()

        # 验证处理器已添加
        handlers = logger._logger.handlers
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in handlers
        )
        self.assertTrue(has_console)

    def test_add_file_handler(self):
        """测试添加文件处理器"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path)

        # 验证处理器已添加
        handlers = logger._logger.handlers
        has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
        self.assertTrue(has_file)
        self.assertTrue(os.path.exists(log_path))

    # ==================== 日志级别测试 ====================

    def test_log_debug(self):
        """测试DEBUG级别日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger", level="DEBUG")
        logger.add_file_handler(log_path)

        logger.debug("debug message")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("debug message", content)
            self.assertIn("DEBUG", content)

    def test_log_info(self):
        """测试INFO级别日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger", level="INFO")
        logger.add_file_handler(log_path)

        logger.info("info message")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("info message", content)
            self.assertIn("INFO", content)

    def test_log_warning(self):
        """测试WARNING级别日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger", level="INFO")
        logger.add_file_handler(log_path)

        logger.warning("warning message")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("warning message", content)
            self.assertIn("WARNING", content)

    def test_log_error(self):
        """测试ERROR级别日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger", level="INFO")
        logger.add_file_handler(log_path)

        logger.error("error message")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("error message", content)
            self.assertIn("ERROR", content)

    def test_log_level_filtering(self):
        """测试日志级别过滤"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger", level="WARNING")
        logger.add_file_handler(log_path)

        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertNotIn("debug", content)
            self.assertNotIn("info", content)
            self.assertIn("warning", content)
            self.assertIn("error", content)

    # ==================== 结构化日志测试 ====================

    def test_log_structured_data(self):
        """测试结构化日志数据"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path, format="json")

        data = {"user_id": 123, "action": "login", "ip": "192.168.1.1"}
        logger.info(data)

        with open(log_path, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            self.assertEqual(log_entry["user_id"], 123)
            self.assertEqual(log_entry["action"], "login")

    def test_log_mixed_message_and_data(self):
        """测试混合消息和数据"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path, format="json")

        logger.info({"message": "User login", "user_id": 456})

        with open(log_path, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            self.assertEqual(log_entry["message"], "User login")
            self.assertEqual(log_entry["user_id"], 456)

    # ==================== 多实例测试 ====================

    def test_multiple_logger_instances(self):
        """测试多个Logger实例"""
        from utils.logger import Logger

        logger1 = Logger("logger1")
        logger2 = Logger("logger2")

        self.assertNotEqual(logger1._logger, logger2._logger)
        self.assertEqual(logger1.name, "logger1")
        self.assertEqual(logger2.name, "logger2")

    def test_logger_isolation(self):
        """测试Logger隔离性"""
        from utils.logger import Logger

        log_path1 = os.path.join(self.temp_dir, "log1.log")
        log_path2 = os.path.join(self.temp_dir, "log2.log")

        logger1 = Logger("logger1")
        logger1.add_file_handler(log_path1)

        logger2 = Logger("logger2")
        logger2.add_file_handler(log_path2)

        logger1.info("message from logger1")
        logger2.info("message from logger2")

        with open(log_path1, 'r') as f:
            content1 = f.read()
            self.assertIn("message from logger1", content1)
            self.assertNotIn("message from logger2", content1)

        with open(log_path2, 'r') as f:
            content2 = f.read()
            self.assertIn("message from logger2", content2)
            self.assertNotIn("message from logger1", content2)

    # ==================== 边缘情况测试 ====================

    def test_log_empty_message(self):
        """测试空消息日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path)

        logger.info("")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertTrue(len(content) > 0)

    def test_log_unicode_message(self):
        """测试Unicode消息日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path)

        logger.info("中文测试消息 🚀")

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("中文测试消息", content)

    def test_log_special_characters(self):
        """测试特殊字符日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path)

        special_msg = "Special chars: <>&\"'\n\t"
        logger.info(special_msg)

        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("Special chars", content)

    def test_log_nested_dict(self):
        """测试嵌套字典日志"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path, format="json")

        data = {
            "level1": {
                "level2": {
                    "level3": "deep value"
                }
            }
        }
        logger.info(data)

        with open(log_path, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            self.assertEqual(log_entry["level1"]["level2"]["level3"], "deep value")

    def test_invalid_log_level(self):
        """测试无效日志级别"""
        from utils.logger import Logger, LoggerConfigError

        with self.assertRaises(LoggerConfigError):
            Logger("test_logger", level="INVALID")

    def test_log_without_handler(self):
        """测试没有处理器的日志"""
        from utils.logger import Logger

        logger = Logger("test_logger")
        # 不应该抛出异常
        logger.info("test message")

    # ==================== 日志轮转测试 ====================

    def test_file_handler_rotation_daily(self):
        """测试按日轮转"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path, rotation="daily")

        logger.info("test message")

        # 验证文件存在
        self.assertTrue(os.path.exists(log_path))

    def test_file_handler_rotation_none(self):
        """测试无轮转"""
        from utils.logger import Logger

        log_path = self._get_temp_log_path()
        logger = Logger("test_logger")
        logger.add_file_handler(log_path, rotation="none")

        logger.info("test message")

        self.assertTrue(os.path.exists(log_path))


class TestLoggerIntegration(TestCase):
    """Logger集成测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_logging_workflow(self):
        """测试完整日志工作流程"""
        from utils.logger import Logger

        log_path = os.path.join(self.temp_dir, "app.log")

        # 创建logger
        logger = Logger("app_logger", level="DEBUG")
        logger.add_console_handler()
        logger.add_file_handler(log_path, format="json")

        # 记录各种日志
        logger.debug({"event": "app_start", "version": "1.0.0"})
        logger.info({"event": "user_action", "user_id": 123})
        logger.warning({"event": "slow_query", "duration_ms": 5000})
        logger.error({"event": "error", "message": "Connection failed"})

        # 验证文件内容
        with open(log_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)

            # 验证每条日志都是有效的JSON
            for line in lines:
                entry = json.loads(line.strip())
                self.assertIn("event", entry)
                self.assertIn("timestamp", entry)


if __name__ == "__main__":
    import unittest
    unittest.main()
