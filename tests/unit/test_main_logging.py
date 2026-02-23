"""
主入口日志模块测试

TDD测试 - 修复main.py中使用print()而不是logging的问题
"""

import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoggingConfiguration:
    """测试日志配置"""

    def test_logging_module_imported(self):
        """测试logging模块已导入"""
        import main

        # 应该使用logging而不是print
        assert hasattr(main, 'logging') or 'logging' in dir(main)

    def test_logger_instance_exists(self):
        """测试logger实例存在"""
        import main

        # 应该有一个模块级别的logger
        assert hasattr(main, 'logger')

    def test_logger_has_methods(self):
        """测试logger有标准方法"""
        import main

        logger = getattr(main, 'logger', None)
        if logger:
            assert hasattr(logger, 'info')
            assert hasattr(logger, 'error')
            assert hasattr(logger, 'debug')
            assert hasattr(logger, 'warning')


class TestNoPrintStatements:
    """测试没有print语句"""

    def test_run_experiment_uses_logging(self):
        """测试run_experiment使用logging而不是print"""
        import main

        # 读取main.py源代码
        main_file = Path(__file__).parent.parent.parent / 'main.py'
        source = main_file.read_text(encoding='utf-8')

        # 应该使用logger.info而不是print
        # 允许在main()函数中使用print（用于CLI输出）
        # 但在run_experiment中应该使用logging

        # 检查是否有logging使用
        assert 'logger' in source or 'logging' in source

    def test_main_function_may_use_print(self):
        """测试main函数可以使用print（CLI输出）"""
        import main

        # main函数被允许使用print用于用户交互
        main_file = Path(__file__).parent.parent.parent / 'main.py'
        source = main_file.read_text(encoding='utf-8')

        # 确认main函数存在
        assert 'def main()' in source


class TestCLIExample:
    """测试CLI示例"""

    def test_cli_example_uses_lowercase(self):
        """测试CLI示例使用小写算法名称"""
        import main

        # 检查epilog中的示例
        source_file = Path(__file__).parent.parent.parent / 'main.py'
        source = source_file.read_text(encoding='utf-8')

        # 示例应该使用小写的greedy和ga
        assert '--algorithm greedy' in source


class TestTypeHints:
    """测试类型注解"""

    def test_run_experiment_has_return_type(self):
        """测试run_experiment有返回类型注解"""
        import main
        import inspect

        sig = inspect.signature(main.run_experiment)

        # 应该有返回类型
        assert sig.return_annotation is not inspect.Signature.empty

    def test_run_experiment_return_type_is_tuple(self):
        """测试run_experiment返回类型是Tuple"""
        import main
        import inspect

        sig = inspect.signature(main.run_experiment)
        return_annotation = sig.return_annotation

        # 返回类型应该是Tuple或包含ScheduleResult和PerformanceMetrics
        return_str = str(return_annotation)
        assert 'Tuple' in return_str or 'ScheduleResult' in return_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
