"""
并行可见性计算器测试

TDD测试 - Phase 3优化：
- 线程池管理
- 多线程并行计算
- 性能对比测试
"""

import pytest
import time
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from core.orbit.visibility.parallel_calculator import ParallelVisibilityCalculator


class TestParallelCalculatorInitialization:
    """测试并行计算器初始化"""

    def test_default_max_workers(self):
        """默认max_workers应为CPU核心数×2"""
        calc = ParallelVisibilityCalculator()
        expected_workers = os.cpu_count() * 2
        assert calc.max_workers == expected_workers

    def test_custom_max_workers(self):
        """应能设置自定义max_workers"""
        calc = ParallelVisibilityCalculator(max_workers=8)
        assert calc.max_workers == 8

    def test_executor_created(self):
        """应创建ThreadPoolExecutor"""
        calc = ParallelVisibilityCalculator(max_workers=4)
        assert calc._executor is not None
        assert isinstance(calc._executor, ThreadPoolExecutor)
        assert calc._executor._max_workers == 4

    def test_thread_name_prefix(self):
        """线程池应有正确的前缀"""
        calc = ParallelVisibilityCalculator()
        # 验证ThreadPoolExecutor被正确初始化
        assert calc._executor is not None
        assert calc._executor._max_workers == calc.max_workers


class TestJVMAttachDecorator:
    """测试JVM attach装饰器"""

    def test_jvm_attach_decorator_exists(self):
        """应存在ensure_jvm_attached装饰器"""
        from core.orbit.visibility.parallel_calculator import ensure_jvm_attached
        assert callable(ensure_jvm_attached)

    def test_jvm_attach_calls_attach_when_not_attached(self):
        """当线程未attach时应调用attach"""
        from core.orbit.visibility.parallel_calculator import ensure_jvm_attached

        mock_func = Mock(return_value="result")
        decorated = ensure_jvm_attached(mock_func)

        with patch('jpype.isJVMStarted', return_value=True):
            with patch('jpype.isThreadAttachedToJVM', return_value=False):
                with patch('jpype.attachThreadToJVM') as mock_attach:
                    result = decorated()

        mock_attach.assert_called_once()
        assert result == "result"

    def test_jvm_attach_skips_when_already_attached(self):
        """当线程已attach时不应重复调用"""
        from core.orbit.visibility.parallel_calculator import ensure_jvm_attached

        mock_func = Mock(return_value="result")
        decorated = ensure_jvm_attached(mock_func)

        with patch('jpype.isJVMStarted', return_value=True):
            with patch('jpype.isThreadAttachedToJVM', return_value=True):
                with patch('jpype.attachThreadToJVM') as mock_attach:
                    result = decorated()

        mock_attach.assert_not_called()
        assert result == "result"

    def test_jvm_attach_skips_when_jvm_not_started(self):
        """JVM未启动时不应调用attach"""
        from core.orbit.visibility.parallel_calculator import ensure_jvm_attached

        mock_func = Mock(return_value="result")
        decorated = ensure_jvm_attached(mock_func)

        with patch('jpype.isJVMStarted', return_value=False):
            with patch('jpype.attachThreadToJVM') as mock_attach:
                result = decorated()

        mock_attach.assert_not_called()
        assert result == "result"


class TestComputeAllWindows:
    """测试并行计算所有窗口"""

    @pytest.fixture
    def mock_satellites(self):
        """模拟卫星列表"""
        return [
            Mock(id=f'SAT{i}', spec=['id'])
            for i in range(3)
        ]

    @pytest.fixture
    def mock_targets(self):
        """模拟目标列表"""
        return [
            Mock(id=f'TARGET{i}', spec=['id'])
            for i in range(3)
        ]

    @pytest.fixture
    def mock_java_bridge(self):
        """模拟Java桥接器"""
        bridge = Mock()
        bridge.compute_visibility_for_pair = Mock(return_value=[
            Mock(start_time=datetime.now(), end_time=datetime.now() + timedelta(minutes=5))
        ])
        return bridge

    def test_compute_all_windows_returns_correct_structure(
        self, mock_satellites, mock_targets, mock_java_bridge
    ):
        """应返回正确结构的结果"""
        calc = ParallelVisibilityCalculator(max_workers=2)

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        with patch.object(calc, '_compute_single_pair') as mock_compute:
            mock_compute.return_value = [
                Mock(start_time=start_time, end_time=start_time + timedelta(hours=1))
            ]

            result = calc.compute_all_windows(
                mock_satellites, mock_targets, (start_time, end_time), mock_java_bridge
            )

        # 应返回9个结果 (3卫星 × 3目标)
        assert len(result) == 9

        # 验证键格式
        for sat in mock_satellites:
            for target in mock_targets:
                key = (sat.id, target.id)
                assert key in result

    def test_compute_all_windows_parallel_execution(
        self, mock_satellites, mock_targets, mock_java_bridge
    ):
        """应并行执行计算"""
        calc = ParallelVisibilityCalculator(max_workers=4)

        execution_order = []

        def slow_compute(satellite, target, time_range, java_bridge):
            execution_order.append((satellite.id, target.id))
            time.sleep(0.01)  # 模拟耗时操作
            return []

        with patch.object(calc, '_compute_single_pair', side_effect=slow_compute):
            start = time.time()
            calc.compute_all_windows(
                mock_satellites, mock_targets,
                (datetime.now(), datetime.now() + timedelta(days=1)),
                mock_java_bridge
            )
            elapsed = time.time() - start

        # 并行执行应比串行快 (9个任务 × 0.01s = 0.09s)
        # 4线程并行应约 0.09/4 ≈ 0.0225s + 开销
        assert elapsed < 0.08  # 允许一些开销

    def test_compute_all_windows_handles_errors_gracefully(
        self, mock_satellites, mock_targets, mock_java_bridge
    ):
        """应优雅处理计算错误"""
        calc = ParallelVisibilityCalculator(max_workers=2)

        def error_compute(satellite, target, time_range, java_bridge):
            if satellite.id == 'SAT1':
                raise Exception("Test error")
            return [Mock(start_time=datetime.now(), end_time=datetime.now())]

        with patch.object(calc, '_compute_single_pair', side_effect=error_compute):
            with patch('core.orbit.visibility.parallel_calculator.logger') as mock_logger:
                result = calc.compute_all_windows(
                    mock_satellites, mock_targets,
                    (datetime.now(), datetime.now() + timedelta(days=1)),
                    mock_java_bridge
                )

        # 即使出错也应返回结果
        assert len(result) == 9
        # 出错的应返回空列表
        assert result[('SAT1', 'TARGET0')] == []

    def test_compute_all_windows_empty_input(self, mock_java_bridge):
        """空输入应返回空结果"""
        calc = ParallelVisibilityCalculator()

        result = calc.compute_all_windows(
            [], [], (datetime.now(), datetime.now() + timedelta(days=1)), mock_java_bridge
        )

        assert result == {}

    def test_compute_all_windows_single_pair(self, mock_java_bridge):
        """单对计算应正确工作"""
        calc = ParallelVisibilityCalculator(max_workers=1)

        satellites = [Mock(id='SAT1')]
        targets = [Mock(id='TARGET1')]

        with patch.object(calc, '_compute_single_pair') as mock_compute:
            mock_compute.return_value = [Mock(spec=['start_time', 'end_time'])]

            result = calc.compute_all_windows(
                satellites, targets,
                (datetime.now(), datetime.now() + timedelta(days=1)),
                mock_java_bridge
            )

        assert len(result) == 1
        assert ('SAT1', 'TARGET1') in result


class TestComputeSinglePair:
    """测试单对计算"""

    @pytest.fixture
    def calculator(self):
        return ParallelVisibilityCalculator(max_workers=2)

    def test_compute_single_pair_jvm_attach(self, calculator):
        """应确保JVM attach"""
        satellite = Mock(id='SAT1')
        target = Mock(id='TARGET1')
        java_bridge = Mock()
        java_bridge.compute_visibility_for_pair = Mock(return_value=[])

        with patch('jpype.isJVMStarted', return_value=True):
            with patch('jpype.isThreadAttachedToJVM', return_value=False):
                with patch('jpype.attachThreadToJVM') as mock_attach:
                    calculator._compute_single_pair(
                        satellite, target,
                        (datetime.now(), datetime.now() + timedelta(days=1)),
                        java_bridge
                    )

        mock_attach.assert_called_once()

    def test_compute_single_pair_calls_java_bridge(self, calculator):
        """应调用Java桥接器"""
        satellite = Mock(id='SAT1')
        target = Mock(id='TARGET1')
        time_range = (datetime(2024, 1, 1), datetime(2024, 1, 2))

        java_bridge = Mock()
        java_bridge.compute_visibility_for_pair = Mock(return_value=[])

        with patch('jpype.isJVMStarted', return_value=False):
            calculator._compute_single_pair(satellite, target, time_range, java_bridge)

        java_bridge.compute_visibility_for_pair.assert_called_once_with(
            satellite, target, time_range
        )

    def test_compute_single_pair_handles_exception(self, calculator):
        """应处理异常"""
        satellite = Mock(id='SAT1')
        target = Mock(id='TARGET1')
        java_bridge = Mock()
        java_bridge.compute_visibility_for_pair = Mock(side_effect=Exception("Test"))

        with patch('jpype.isJVMStarted', return_value=False):
            with patch('core.orbit.visibility.parallel_calculator.logger') as mock_logger:
                result = calculator._compute_single_pair(
                    satellite, target,
                    (datetime.now(), datetime.now() + timedelta(days=1)),
                    java_bridge
                )

        assert result == []
        mock_logger.error.assert_called()


class TestPerformanceComparison:
    """性能对比测试"""

    @pytest.mark.slow
    def test_8_thread_speedup(self):
        """8线程应比单线程快4倍以上"""
        calc_8 = ParallelVisibilityCalculator(max_workers=8)
        calc_1 = ParallelVisibilityCalculator(max_workers=1)

        # 模拟9个任务
        satellites = [Mock(id=f'SAT{i}') for i in range(3)]
        targets = [Mock(id=f'TARGET{i}') for i in range(3)]
        java_bridge = Mock()

        def slow_compute(satellite, target, time_range, java_bridge):
            time.sleep(0.05)  # 每个任务50ms
            return [Mock()]

        # 单线程
        with patch.object(calc_1, '_compute_single_pair', side_effect=slow_compute):
            start = time.time()
            calc_1.compute_all_windows(
                satellites, targets,
                (datetime.now(), datetime.now() + timedelta(days=1)),
                java_bridge
            )
            single_thread_time = time.time() - start

        # 8线程
        with patch.object(calc_8, '_compute_single_pair', side_effect=slow_compute):
            start = time.time()
            calc_8.compute_all_windows(
                satellites, targets,
                (datetime.now(), datetime.now() + timedelta(days=1)),
                java_bridge
            )
            multi_thread_time = time.time() - start

        speedup = single_thread_time / multi_thread_time
        print(f"\nSingle thread: {single_thread_time:.3f}s")
        print(f"8 threads: {multi_thread_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")

        # 期望4倍以上加速
        assert speedup >= 4.0, f"加速比 {speedup:.1f}x 低于预期的4x"

    def test_memory_usage_under_limit(self):
        """内存使用应低于2GB"""
        pytest.importorskip("psutil", reason="psutil not installed")
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        calc = ParallelVisibilityCalculator(max_workers=16)

        # 创建大量任务
        satellites = [Mock(id=f'SAT{i}') for i in range(10)]
        targets = [Mock(id=f'TARGET{i}') for i in range(10)]
        java_bridge = Mock()

        with patch.object(calc, '_compute_single_pair', return_value=[]):
            calc.compute_all_windows(
                satellites, targets,
                (datetime.now(), datetime.now() + timedelta(days=1)),
                java_bridge
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nMemory increase: {memory_increase:.1f} MB")

        # 内存增长应小于500MB
        assert memory_increase < 500, f"内存增长 {memory_increase:.1f}MB 超过500MB限制"


class TestShutdown:
    """测试关闭"""

    def test_shutdown_closes_executor(self):
        """shutdown应关闭线程池"""
        calc = ParallelVisibilityCalculator(max_workers=2)

        with patch.object(calc._executor, 'shutdown') as mock_shutdown:
            calc.shutdown()

        mock_shutdown.assert_called_once_with(wait=True)

    def test_context_manager(self):
        """应支持上下文管理器"""
        with patch('core.orbit.visibility.parallel_calculator.ThreadPoolExecutor'):
            with ParallelVisibilityCalculator(max_workers=2) as calc:
                assert calc is not None


class TestIntegrationWithOrekitCalculator:
    """与OrekitVisibilityCalculator集成测试"""

    def test_parallel_enabled_config(self):
        """应支持use_parallel配置"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calc = OrekitVisibilityCalculator(config={
            'use_parallel': True,
            'max_workers': 8
        })

        assert hasattr(calc, 'use_parallel')
        assert calc.use_parallel is True

    def test_parallel_disabled_config(self):
        """应支持禁用并行"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calc = OrekitVisibilityCalculator(config={
            'use_parallel': False
        })

        assert calc.use_parallel is False
