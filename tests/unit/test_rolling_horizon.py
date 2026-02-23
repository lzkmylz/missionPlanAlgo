"""
滚动时间窗管理器测试

TDD测试文件 - 第19章设计实现
"""

import pytest
from datetime import datetime, timedelta, timezone

from core.dynamic_scheduler.rolling_horizon import (
    RollingHorizonConfig,
    RollingHorizonManager
)


class TestRollingHorizonConfig:
    """测试滚动时间窗配置类"""

    def test_default_values(self):
        """测试默认配置值"""
        config = RollingHorizonConfig()

        assert config.window_size == timedelta(hours=2)
        assert config.shift_interval == timedelta(minutes=15)
        assert config.freeze_duration == timedelta(minutes=5)
        assert config.optimization_method == 'fast_heuristic'
        assert config.max_optimization_time == 30

    def test_custom_values(self):
        """测试自定义配置值"""
        config = RollingHorizonConfig(
            window_size=timedelta(hours=4),
            shift_interval=timedelta(minutes=30),
            freeze_duration=timedelta(minutes=10),
            optimization_method='metaheuristic',
            max_optimization_time=60
        )

        assert config.window_size == timedelta(hours=4)
        assert config.shift_interval == timedelta(minutes=30)
        assert config.freeze_duration == timedelta(minutes=10)
        assert config.optimization_method == 'metaheuristic'
        assert config.max_optimization_time == 60

    def test_valid_optimization_methods(self):
        """测试有效的优化方法"""
        config1 = RollingHorizonConfig(optimization_method='fast_heuristic')
        assert config1.optimization_method == 'fast_heuristic'

        config2 = RollingHorizonConfig(optimization_method='metaheuristic')
        assert config2.optimization_method == 'metaheuristic'

    def test_invalid_optimization_method_raises_error(self):
        """测试无效的优化方法应该报错"""
        with pytest.raises(ValueError) as exc_info:
            RollingHorizonConfig(optimization_method='invalid_method')
        assert 'Invalid optimization_method' in str(exc_info.value)

    def test_invalid_max_optimization_time_raises_error(self):
        """测试无效的最大优化时间应该报错"""
        with pytest.raises(ValueError) as exc_info:
            RollingHorizonConfig(max_optimization_time=0)
        assert 'max_optimization_time must be positive' in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            RollingHorizonConfig(max_optimization_time=-1)
        assert 'max_optimization_time must be positive' in str(exc_info.value)


class TestRollingHorizonManagerInit:
    """测试滚动时间窗管理器初始化"""

    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        manager = RollingHorizonManager()

        assert manager.config is not None
        assert manager.config.window_size == timedelta(hours=2)
        assert manager.last_optimization_time is None

    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = RollingHorizonConfig(
            window_size=timedelta(hours=3),
            shift_interval=timedelta(minutes=20)
        )
        manager = RollingHorizonManager(config)

        assert manager.config == config
        assert manager.config.window_size == timedelta(hours=3)
        assert manager.last_optimization_time is None


class TestShouldTriggerOptimization:
    """测试是否应该触发优化"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.config = RollingHorizonConfig(
            shift_interval=timedelta(minutes=15)
        )
        self.manager = RollingHorizonManager(self.config)
        self.base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    def test_first_call_should_trigger(self):
        """测试首次调用应该触发优化"""
        assert self.manager.last_optimization_time is None
        assert self.manager.should_trigger_optimization(self.base_time) is True

    def test_before_shift_interval_should_not_trigger(self):
        """测试在滚动间隔内不应该触发优化"""
        # 先设置上次优化时间
        self.manager.last_optimization_time = self.base_time

        # 10分钟后，还未达到15分钟间隔
        current_time = self.base_time + timedelta(minutes=10)
        assert self.manager.should_trigger_optimization(current_time) is False

    def test_at_shift_interval_should_trigger(self):
        """测试正好达到滚动间隔应该触发优化"""
        self.manager.last_optimization_time = self.base_time

        # 正好15分钟后
        current_time = self.base_time + timedelta(minutes=15)
        assert self.manager.should_trigger_optimization(current_time) is True

    def test_after_shift_interval_should_trigger(self):
        """测试超过滚动间隔应该触发优化"""
        self.manager.last_optimization_time = self.base_time

        # 20分钟后，超过15分钟间隔
        current_time = self.base_time + timedelta(minutes=20)
        assert self.manager.should_trigger_optimization(current_time) is True

    def test_exact_time_should_trigger(self):
        """测试时间相同时应该触发优化"""
        self.manager.last_optimization_time = self.base_time

        # 正好15分钟后
        current_time = self.base_time + timedelta(minutes=15)
        assert self.manager.should_trigger_optimization(current_time) is True

    def test_negative_elapsed_time_should_handle_gracefully(self):
        """测试当前时间早于上次优化时间的情况"""
        self.manager.last_optimization_time = self.base_time

        # 当前时间早于上次优化时间（时钟回拨等情况）
        earlier_time = self.base_time - timedelta(minutes=5)
        # 这种情况下应该触发优化，因为时间异常
        result = self.manager.should_trigger_optimization(earlier_time)
        assert isinstance(result, bool)

    def test_invalid_current_time_type_raises_error(self):
        """测试无效的当前时间类型应该报错"""
        with pytest.raises(TypeError) as exc_info:
            self.manager.should_trigger_optimization("not a datetime")
        assert 'current_time must be a datetime instance' in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            self.manager.should_trigger_optimization(12345)
        assert 'current_time must be a datetime instance' in str(exc_info.value)


class TestGetOptimizationWindow:
    """测试获取优化窗口"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.config = RollingHorizonConfig(
            window_size=timedelta(hours=2),
            freeze_duration=timedelta(minutes=5)
        )
        self.manager = RollingHorizonManager(self.config)
        self.base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    def test_optimization_window_structure(self):
        """测试优化窗口返回结构"""
        result = self.manager.get_optimization_window(self.base_time)

        assert isinstance(result, tuple)
        assert len(result) == 3

        window_start, window_end, freeze_until = result

        assert isinstance(window_start, datetime)
        assert isinstance(window_end, datetime)
        assert isinstance(freeze_until, datetime)

    def test_freeze_until_calculation(self):
        """测试冻结时间计算"""
        _, _, freeze_until = self.manager.get_optimization_window(self.base_time)

        expected_freeze_until = self.base_time + timedelta(minutes=5)
        assert freeze_until == expected_freeze_until

    def test_window_start_calculation(self):
        """测试窗口开始时间计算"""
        window_start, _, _ = self.manager.get_optimization_window(self.base_time)

        # 窗口开始时间应该等于冻结结束时间
        expected_window_start = self.base_time + timedelta(minutes=5)
        assert window_start == expected_window_start

    def test_window_end_calculation(self):
        """测试窗口结束时间计算"""
        _, window_end, _ = self.manager.get_optimization_window(self.base_time)

        expected_window_end = self.base_time + timedelta(hours=2)
        assert window_end == expected_window_end

    def test_window_relationships(self):
        """测试窗口时间关系"""
        window_start, window_end, freeze_until = self.manager.get_optimization_window(self.base_time)

        # freeze_until应该等于window_start
        assert freeze_until == window_start

        # window_end应该晚于window_start
        assert window_end > window_start

        # window_end - window_start应该等于window_size - freeze_duration
        actual_window_duration = window_end - window_start
        expected_window_duration = timedelta(hours=2) - timedelta(minutes=5)
        assert actual_window_duration == expected_window_duration

    def test_timezone_aware_datetime(self):
        """测试时区感知datetime处理"""
        # 使用UTC时区
        utc_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = self.manager.get_optimization_window(utc_time)

        window_start, window_end, freeze_until = result
        assert window_start.tzinfo == timezone.utc
        assert window_end.tzinfo == timezone.utc
        assert freeze_until.tzinfo == timezone.utc

    def test_naive_datetime_raises_error(self):
        """测试无时区datetime应该报错"""
        naive_time = datetime(2024, 1, 1, 12, 0)

        with pytest.raises((ValueError, TypeError)):
            self.manager.get_optimization_window(naive_time)

    def test_invalid_current_time_type_raises_error(self):
        """测试无效的当前时间类型应该报错"""
        with pytest.raises(TypeError) as exc_info:
            self.manager.get_optimization_window("not a datetime")
        assert 'current_time must be a datetime instance' in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            self.manager.get_optimization_window(None)
        assert 'current_time must be a datetime instance' in str(exc_info.value)

    def test_different_current_times(self):
        """测试不同当前时间的窗口计算"""
        # 测试多个不同的时间点
        test_times = [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc),
            datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc),
        ]

        for current_time in test_times:
            window_start, window_end, freeze_until = self.manager.get_optimization_window(current_time)

            assert freeze_until == current_time + timedelta(minutes=5)
            assert window_start == current_time + timedelta(minutes=5)
            assert window_end == current_time + timedelta(hours=2)


class TestEdgeCases:
    """测试边界情况"""

    def test_zero_shift_interval(self):
        """测试零滚动间隔"""
        config = RollingHorizonConfig(shift_interval=timedelta(0))
        manager = RollingHorizonManager(config)

        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        manager.last_optimization_time = base_time

        # 零间隔时，任何时间都应该触发
        current_time = base_time + timedelta(seconds=1)
        assert manager.should_trigger_optimization(current_time) is True

    def test_very_small_window(self):
        """测试非常小的窗口大小"""
        config = RollingHorizonConfig(
            window_size=timedelta(seconds=1),
            freeze_duration=timedelta(0)
        )
        manager = RollingHorizonManager(config)

        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        window_start, window_end, freeze_until = manager.get_optimization_window(base_time)

        assert window_end == base_time + timedelta(seconds=1)
        assert freeze_until == base_time

    def test_large_window_size(self):
        """测试大窗口大小"""
        config = RollingHorizonConfig(
            window_size=timedelta(days=7),
            freeze_duration=timedelta(hours=1)
        )
        manager = RollingHorizonManager(config)

        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        window_start, window_end, freeze_until = manager.get_optimization_window(base_time)

        assert window_end == base_time + timedelta(days=7)
        assert freeze_until == base_time + timedelta(hours=1)


class TestRecordOptimization:
    """测试记录优化完成时间"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.manager = RollingHorizonManager()
        self.base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    def test_record_optimization_updates_last_time(self):
        """测试记录优化会更新上次优化时间"""
        assert self.manager.last_optimization_time is None

        self.manager.record_optimization(self.base_time)

        assert self.manager.last_optimization_time == self.base_time

    def test_record_optimization_with_different_times(self):
        """测试使用不同时间记录优化"""
        time1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        time2 = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)

        self.manager.record_optimization(time1)
        assert self.manager.last_optimization_time == time1

        self.manager.record_optimization(time2)
        assert self.manager.last_optimization_time == time2

    def test_record_optimization_invalid_type_raises_error(self):
        """测试记录优化时传入无效类型应该报错"""
        with pytest.raises(TypeError) as exc_info:
            self.manager.record_optimization("not a datetime")
        assert 'optimization_time must be a datetime instance' in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            self.manager.record_optimization(12345)
        assert 'optimization_time must be a datetime instance' in str(exc_info.value)


class TestGetTimeUntilNextOptimization:
    """测试获取距离下次优化的时间"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.config = RollingHorizonConfig(shift_interval=timedelta(minutes=15))
        self.manager = RollingHorizonManager(self.config)
        self.base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    def test_no_previous_optimization_returns_none(self):
        """测试没有上次优化时返回None"""
        result = self.manager.get_time_until_next_optimization(self.base_time)
        assert result is None

    def test_time_until_next_optimization_calculation(self):
        """测试距离下次优化时间计算"""
        self.manager.last_optimization_time = self.base_time

        # 10分钟后，距离下次优化还有5分钟
        current_time = self.base_time + timedelta(minutes=10)
        result = self.manager.get_time_until_next_optimization(current_time)

        assert result == timedelta(minutes=5)

    def test_zero_time_remaining(self):
        """测试距离下次优化时间为0"""
        self.manager.last_optimization_time = self.base_time

        # 正好15分钟后，距离下次优化时间为0
        current_time = self.base_time + timedelta(minutes=15)
        result = self.manager.get_time_until_next_optimization(current_time)

        assert result == timedelta(0)

    def test_negative_time_returns_zero(self):
        """测试超过间隔时间返回0"""
        self.manager.last_optimization_time = self.base_time

        # 20分钟后，已经超过间隔
        current_time = self.base_time + timedelta(minutes=20)
        result = self.manager.get_time_until_next_optimization(current_time)

        assert result == timedelta(0)

    def test_clock_rollback_returns_zero(self):
        """测试时钟回拨情况返回0"""
        self.manager.last_optimization_time = self.base_time

        # 当前时间早于上次优化时间
        earlier_time = self.base_time - timedelta(minutes=5)
        result = self.manager.get_time_until_next_optimization(earlier_time)

        assert result == timedelta(0)


class TestIntegration:
    """测试集成场景"""

    def test_full_optimization_cycle(self):
        """测试完整的优化周期"""
        config = RollingHorizonConfig(
            window_size=timedelta(hours=2),
            shift_interval=timedelta(minutes=15),
            freeze_duration=timedelta(minutes=5)
        )
        manager = RollingHorizonManager(config)

        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

        # 第一次检查 - 应该触发
        assert manager.should_trigger_optimization(base_time) is True

        # 获取优化窗口
        window_start, window_end, freeze_until = manager.get_optimization_window(base_time)
        assert window_start == base_time + timedelta(minutes=5)
        assert window_end == base_time + timedelta(hours=2)

        # 模拟优化完成，使用record_optimization更新上次优化时间
        manager.record_optimization(base_time)

        # 10分钟后检查 - 不应该触发
        time_after_10min = base_time + timedelta(minutes=10)
        assert manager.should_trigger_optimization(time_after_10min) is False

        # 15分钟后检查 - 应该触发
        time_after_15min = base_time + timedelta(minutes=15)
        assert manager.should_trigger_optimization(time_after_15min) is True

        # 获取新的优化窗口
        window_start2, window_end2, freeze_until2 = manager.get_optimization_window(time_after_15min)
        assert window_start2 == time_after_15min + timedelta(minutes=5)
        assert window_end2 == time_after_15min + timedelta(hours=2)
