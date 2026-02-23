"""
遗传算法调度器测试

TDD测试文件 - 修复GA调度器桩实现
"""

import pytest
from datetime import datetime, timedelta

from core.models import Mission, Satellite, SatelliteType, Target, TargetType
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.base_scheduler import ScheduleResult


class TestGAScheduler:
    """测试遗传算法调度器"""

    def setup_method(self):
        """设置测试场景"""
        # 创建卫星
        self.satellites = [
            Satellite(id=f"SAT-{i:02d}", name=f"卫星{i}", sat_type=SatelliteType.OPTICAL_1)
            for i in range(3)
        ]

        # 创建目标
        self.targets = [
            Target(
                id=f"TARGET-{i:03d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i * 2,
                latitude=39.0 + i,
                priority=5
            )
            for i in range(10)
        ]

        self.mission = Mission(
            name="GA测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=self.satellites,
            targets=self.targets
        )

    def test_ga_initialization(self):
        """测试GA调度器初始化"""
        config = {
            'population_size': 50,
            'generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
        }
        scheduler = GAScheduler(config)

        assert scheduler.population_size == 50
        assert scheduler.generations == 100
        assert scheduler.crossover_rate == 0.8
        assert scheduler.mutation_rate == 0.1

    def test_ga_default_parameters(self):
        """测试GA默认参数"""
        scheduler = GAScheduler()

        assert scheduler.population_size == 100
        assert scheduler.generations == 200
        assert scheduler.crossover_rate == 0.8
        assert scheduler.mutation_rate == 0.2

    def test_ga_returns_result(self):
        """测试GA返回有效结果"""
        scheduler = GAScheduler(config={'generations': 10, 'population_size': 20})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert hasattr(result, 'scheduled_tasks')
        assert hasattr(result, 'convergence_curve')
        assert hasattr(result, 'iterations')

    def test_ga_convergence_curve(self):
        """测试GA生成收敛曲线"""
        scheduler = GAScheduler(config={'generations': 10, 'population_size': 20})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        # 收敛曲线长度应该等于迭代代数 + 1（初始种群）
        assert len(result.convergence_curve) == 11  # 初始 + 10代
        # 收敛曲线应该是非递减的（或大致如此）
        for i in range(1, len(result.convergence_curve)):
            assert result.convergence_curve[i] >= result.convergence_curve[i-1] * 0.9  # 允许少量波动

    def test_ga_schedules_tasks(self):
        """测试GA能够调度任务（使用窗口缓存）"""
        from datetime import datetime
        from core.orbit.visibility.base import VisibilityWindow

        scheduler = GAScheduler(config={'generations': 10, 'population_size': 20})
        scheduler.initialize(self.mission)

        # 创建模拟窗口缓存
        class MockWindowCache:
            def get_windows(self, sat_id, target_id):
                return [VisibilityWindow(
                    satellite_id=sat_id,
                    target_id=target_id,
                    start_time=datetime(2024, 1, 1, 10, 0),
                    end_time=datetime(2024, 1, 1, 10, 5),
                    max_elevation=45.0,
                    quality_score=0.8
                )]

        scheduler.set_window_cache(MockWindowCache())
        result = scheduler.schedule()

        # 应该至少调度一些任务
        assert len(result.scheduled_tasks) > 0
        # 调度任务数不应超过总任务数
        assert len(result.scheduled_tasks) <= len(self.targets)

    def test_ga_solution_quality(self):
        """测试GA解的质量（使用窗口缓存）"""
        scheduler = GAScheduler(config={'generations': 20, 'population_size': 30})
        scheduler.initialize(self.mission)

        # 创建模拟窗口缓存
        class MockWindowCache:
            def get_windows(self, sat_id, target_id):
                from datetime import datetime
                from core.orbit.visibility.base import VisibilityWindow
                return [VisibilityWindow(
                    satellite_id=sat_id,
                    target_id=target_id,
                    start_time=datetime(2024, 1, 1, 10, 0),
                    end_time=datetime(2024, 1, 1, 10, 5),
                    max_elevation=45.0,
                    quality_score=0.8
                )]

        scheduler.set_window_cache(MockWindowCache())
        result = scheduler.schedule()

        # 计算需求满足率
        dsr = len(result.scheduled_tasks) / len(self.targets)
        # GA应该能调度至少30%的任务（基于简化场景）
        assert dsr >= 0.3

    def test_ga_with_window_cache(self):
        """测试GA使用窗口缓存"""
        from datetime import datetime
        from core.orbit.visibility.base import VisibilityWindow

        scheduler = GAScheduler(config={'generations': 5, 'population_size': 10})
        scheduler.initialize(self.mission)

        # 创建模拟窗口缓存
        class MockWindowCache:
            def get_windows(self, sat_id, target_id):
                return [VisibilityWindow(
                    satellite_id=sat_id,
                    target_id=target_id,
                    start_time=datetime(2024, 1, 1, 10, 0),
                    end_time=datetime(2024, 1, 1, 10, 5),
                    max_elevation=45.0,
                    quality_score=0.8
                )]

        scheduler.set_window_cache(MockWindowCache())
        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_ga_multiple_runs_consistency(self):
        """测试GA多次运行的一致性（使用相同随机种子）"""
        config = {
            'generations': 10,
            'population_size': 20,
            'random_seed': 42
        }

        results = []
        for _ in range(2):
            scheduler = GAScheduler(config)
            scheduler.initialize(self.mission)
            result = scheduler.schedule()
            results.append(len(result.scheduled_tasks))

        # 使用相同种子应该产生相同结果
        assert results[0] == results[1]

    def test_ga_parameter_validation(self):
        """测试GA参数验证"""
        # 测试无效参数
        with pytest.raises((ValueError, AssertionError)):
            GAScheduler(config={'population_size': 0})

        with pytest.raises((ValueError, AssertionError)):
            GAScheduler(config={'crossover_rate': 1.5})  # 应该小于1

    def test_ga_empty_mission_raises_error(self):
        """测试GA空场景（无卫星、无目标）应抛出初始化错误"""
        empty_mission = Mission(
            name="空场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[]
        )

        scheduler = GAScheduler(config={'generations': 5})
        scheduler.initialize(empty_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_ga_fitness_improvement(self):
        """测试GA适应度随代数改善"""
        scheduler = GAScheduler(config={'generations': 20, 'population_size': 30})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        # 适应度应该随代数增加（或至少不减少太多）
        first_fitness = result.convergence_curve[0]
        last_fitness = result.convergence_curve[-1]

        # 最终适应度应该比初始好（或相等）
        assert last_fitness >= first_fitness * 0.8  # 允许20%的波动


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
