"""
测试ExperimentRunner窗口缓存集成

M4: 实验运行器集成窗口缓存测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from experiments.runner import ExperimentRunner, ExperimentConfig
from core.orbit.visibility.window_cache import VisibilityWindowCache


class TestExperimentRunnerWindowCache:
    """测试ExperimentRunner窗口缓存集成"""

    def create_mock_mission(self):
        """创建模拟任务场景"""
        mission = Mock()
        mission.name = "Test Mission"
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        # 创建卫星
        satellites = []
        for i in range(3):
            sat = Mock()
            sat.id = f"SAT-{i:02d}"
            satellites.append(sat)
        mission.satellites = satellites

        # 创建目标
        targets = []
        for i in range(5):
            target = Mock()
            target.id = f"TARGET-{i:02d}"
            targets.append(target)
        mission.targets = targets

        # 创建地面站
        ground_stations = []
        for i in range(2):
            gs = Mock()
            gs.id = f"GS-{i:02d}"
            ground_stations.append(gs)
        mission.ground_stations = ground_stations

        return mission

    def create_mock_scheduler_class(self):
        """创建模拟调度器类"""
        class MockScheduler:
            def __init__(self, config):
                self.config = config
                self.window_cache = None
                self.mission = None

            def initialize(self, mission):
                self.mission = mission

            def set_window_cache(self, cache):
                self.window_cache = cache

            def schedule(self):
                from scheduler.base_scheduler import ScheduleResult
                return ScheduleResult(
                    scheduled_tasks=[],
                    unscheduled_tasks={},
                    makespan=0.0,
                    computation_time=1.0,
                    iterations=10
                )

            def get_parameters(self):
                return {}

        return MockScheduler

    def test_runner_has_precompute_method(self):
        """测试Runner有预计算方法"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        assert hasattr(runner, 'precompute_window_cache')

    def test_precompute_window_cache_creates_cache(self):
        """测试预计算创建窗口缓存"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        # 模拟可见性计算器
        mock_calculator = Mock()
        mock_calculator.compute_satellite_target_windows = Mock(return_value=[])
        mock_calculator.compute_satellite_ground_station_windows = Mock(return_value=[])

        cache = runner.precompute_window_cache(mock_calculator)

        assert cache is not None
        assert isinstance(cache, VisibilityWindowCache)

    def test_precompute_window_cache_calls_calculator(self):
        """测试预计算调用计算器"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        mock_calculator = Mock()
        mock_calculator.compute_satellite_target_windows = Mock(return_value=[])
        mock_calculator.compute_satellite_ground_station_windows = Mock(return_value=[])

        runner.precompute_window_cache(mock_calculator)

        # 验证计算器被调用
        expected_target_calls = len(mission.satellites) * len(mission.targets)
        expected_gs_calls = len(mission.satellites) * len(mission.ground_stations)

        assert mock_calculator.compute_satellite_target_windows.call_count == expected_target_calls
        assert mock_calculator.compute_satellite_ground_station_windows.call_count == expected_gs_calls

    def test_inject_cache_to_schedulers(self):
        """测试向调度器注入缓存"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        # 创建模拟缓存
        mock_cache = Mock()
        runner.window_cache = mock_cache

        # 创建调度器实例并注入缓存
        scheduler = scheduler_class({})
        runner._inject_window_cache(scheduler)

        assert scheduler.window_cache == mock_cache

    def test_run_single_experiment_with_cache(self):
        """测试运行单个实验时使用缓存"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        # 设置缓存
        mock_cache = Mock()
        runner.window_cache = mock_cache

        result = runner.run_single_experiment("mock", {})

        assert result is not None
        assert result.algorithm_name == "mock"

    def test_run_all_with_cache(self):
        """测试运行所有实验时使用缓存"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class},
            config=ExperimentConfig(repetitions=2)
        )

        # 设置缓存
        mock_cache = Mock()
        runner.window_cache = mock_cache

        results = runner.run_all()

        assert "mock" in results
        assert len(results["mock"]) == 2

    def test_cache_injection_disabled_when_no_cache(self):
        """测试无缓存时不注入"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        # 不设置缓存
        runner.window_cache = None

        # 创建调度器
        scheduler = scheduler_class({})

        # 注入缓存（应该不报错）
        runner._inject_window_cache(scheduler)

        # 调度器的缓存应该为None
        assert scheduler.window_cache is None

    def test_precompute_with_custom_time_range(self):
        """测试自定义时间范围的预计算"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        mock_calculator = Mock()
        mock_calculator.compute_satellite_target_windows = Mock(return_value=[])
        mock_calculator.compute_satellite_ground_station_windows = Mock(return_value=[])

        custom_start = datetime(2024, 1, 1, 6, 0, 0)
        custom_end = datetime(2024, 1, 1, 18, 0, 0)

        cache = runner.precompute_window_cache(
            mock_calculator,
            start_time=custom_start,
            end_time=custom_end
        )

        assert cache is not None

        # 验证计算器使用自定义时间
        calls = mock_calculator.compute_satellite_target_windows.call_args_list
        for call in calls:
            args = call[0]
            assert args[2] == custom_start
            assert args[3] == custom_end

    def test_cache_statistics_after_precompute(self):
        """测试预计算后获取缓存统计"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        mock_calculator = Mock()
        mock_calculator.compute_satellite_target_windows = Mock(return_value=[])
        mock_calculator.compute_satellite_ground_station_windows = Mock(return_value=[])

        runner.precompute_window_cache(mock_calculator)

        stats = runner.get_cache_statistics()

        assert isinstance(stats, dict)
        assert 'total_window_pairs' in stats
        assert 'total_windows' in stats

    def test_clear_cache(self):
        """测试清除缓存"""
        mission = self.create_mock_mission()
        scheduler_class = self.create_mock_scheduler_class()

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"mock": scheduler_class}
        )

        # 设置缓存
        mock_cache = Mock()
        mock_cache.clear = Mock()
        runner.window_cache = mock_cache

        runner.clear_window_cache()

        mock_cache.clear.assert_called_once()
        assert runner.window_cache is None


class TestExperimentRunnerCacheIntegration:
    """测试ExperimentRunner缓存集成的高级场景"""

    def create_mock_mission(self):
        """创建模拟任务场景"""
        mission = Mock()
        mission.name = "Test Mission"
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        satellites = []
        for i in range(2):
            sat = Mock()
            sat.id = f"SAT-{i:02d}"
            satellites.append(sat)
        mission.satellites = satellites

        targets = []
        for i in range(3):
            target = Mock()
            target.id = f"TARGET-{i:02d}"
            targets.append(target)
        mission.targets = targets

        ground_stations = []
        for i in range(1):
            gs = Mock()
            gs.id = f"GS-{i:02d}"
            ground_stations.append(gs)
        mission.ground_stations = ground_stations

        return mission

    def test_multiple_algorithms_share_cache(self):
        """测试多个算法共享缓存"""
        mission = self.create_mock_mission()

        class SchedulerA:
            def __init__(self, config):
                self.window_cache = None
            def initialize(self, mission):
                pass
            def set_window_cache(self, cache):
                self.window_cache = cache
            def schedule(self):
                from scheduler.base_scheduler import ScheduleResult
                return ScheduleResult([], {}, 0.0, 1.0, 10)
            def get_parameters(self):
                return {}

        class SchedulerB:
            def __init__(self, config):
                self.window_cache = None
            def initialize(self, mission):
                pass
            def set_window_cache(self, cache):
                self.window_cache = cache
            def schedule(self):
                from scheduler.base_scheduler import ScheduleResult
                return ScheduleResult([], {}, 0.0, 1.0, 10)
            def get_parameters(self):
                return {}

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"A": SchedulerA, "B": SchedulerB},
            config=ExperimentConfig(repetitions=1)
        )

        # 预计算缓存
        mock_calculator = Mock()
        mock_calculator.compute_satellite_target_windows = Mock(return_value=[])
        mock_calculator.compute_satellite_ground_station_windows = Mock(return_value=[])

        runner.precompute_window_cache(mock_calculator)

        # 运行实验
        results = runner.run_all()

        # 两个算法都应该成功运行
        assert "A" in results
        assert "B" in results

    def test_cache_persists_across_repetitions(self):
        """测试缓存在重复实验中持久化"""
        mission = self.create_mock_mission()

        call_count = 0

        class CountingScheduler:
            def __init__(self, config):
                self.window_cache = None
            def initialize(self, mission):
                pass
            def set_window_cache(self, cache):
                self.window_cache = cache
            def schedule(self):
                nonlocal call_count
                call_count += 1
                from scheduler.base_scheduler import ScheduleResult
                return ScheduleResult([], {}, 0.0, 1.0, 10)
            def get_parameters(self):
                return {}

        runner = ExperimentRunner(
            mission=mission,
            algorithms={"counting": CountingScheduler},
            config=ExperimentConfig(repetitions=3)
        )

        # 预计算一次缓存
        mock_calculator = Mock()
        mock_calculator.compute_satellite_target_windows = Mock(return_value=[])
        mock_calculator.compute_satellite_ground_station_windows = Mock(return_value=[])

        runner.precompute_window_cache(mock_calculator)

        # 运行3次重复实验
        runner.run_all()

        # 验证调度器被调用3次
        assert call_count == 3

        # 验证计算器只被调用一次（预计算阶段）
        total_expected_calls = len(mission.satellites) * len(mission.targets)
        assert mock_calculator.compute_satellite_target_windows.call_count == total_expected_calls
