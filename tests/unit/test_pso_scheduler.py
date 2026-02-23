"""
粒子群优化调度器测试

TDD测试文件 - 实现PSO调度器
"""

import pytest
from datetime import datetime, timedelta

from core.models import Mission, Satellite, SatelliteType, Target, TargetType
from scheduler.metaheuristic.pso_scheduler import PSOScheduler
from scheduler.base_scheduler import ScheduleResult


class TestPSOScheduler:
    """测试粒子群优化调度器"""

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
            name="PSO测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=self.satellites,
            targets=self.targets
        )

    def test_pso_initialization(self):
        """测试PSO调度器初始化"""
        config = {
            'num_particles': 40,
            'max_iterations': 200,
            'cognitive_coeff': 2.5,
            'social_coeff': 2.5,
            'inertia_weight': 0.8
        }
        scheduler = PSOScheduler(config)

        assert scheduler.num_particles == 40
        assert scheduler.max_iterations == 200
        assert scheduler.cognitive_coeff == 2.5
        assert scheduler.social_coeff == 2.5
        assert scheduler.inertia_weight == 0.8

    def test_pso_default_parameters(self):
        """测试PSO默认参数"""
        scheduler = PSOScheduler()

        assert scheduler.num_particles == 30
        assert scheduler.max_iterations == 100
        assert scheduler.cognitive_coeff == 2.0
        assert scheduler.social_coeff == 2.0
        assert scheduler.inertia_weight == 0.9

    def test_pso_returns_result(self):
        """测试PSO返回有效结果"""
        scheduler = PSOScheduler(config={'max_iterations': 50, 'num_particles': 10})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert hasattr(result, 'scheduled_tasks')
        assert hasattr(result, 'convergence_curve')
        assert hasattr(result, 'iterations')

    def test_pso_convergence_curve(self):
        """测试PSO生成收敛曲线"""
        scheduler = PSOScheduler(config={'max_iterations': 50, 'num_particles': 10})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        # 收敛曲线应该有数据
        assert len(result.convergence_curve) > 0

    def test_pso_schedules_tasks(self):
        """测试PSO能够调度任务（使用窗口缓存）"""
        from datetime import datetime
        from core.orbit.visibility.base import VisibilityWindow

        scheduler = PSOScheduler(config={'max_iterations': 50, 'num_particles': 10})
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

    def test_pso_empty_mission_raises_error(self):
        """测试PSO空场景（无卫星、无目标）应抛出初始化错误"""
        empty_mission = Mission(
            name="空场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[]
        )

        scheduler = PSOScheduler(config={'max_iterations': 50})
        scheduler.initialize(empty_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_pso_parameter_validation(self):
        """测试PSO参数验证"""
        # 测试无效粒子数量
        with pytest.raises((ValueError, AssertionError)):
            PSOScheduler(config={'num_particles': 0})

        # 测试无效惯性权重
        with pytest.raises((ValueError, AssertionError)):
            PSOScheduler(config={'inertia_weight': 1.5})

    def test_pso_velocity_update(self):
        """测试PSO速度更新机制"""
        scheduler = PSOScheduler(config={
            'max_iterations': 10,
            'num_particles': 5,
            'inertia_weight': 0.9,
            'cognitive_coeff': 2.0,
            'social_coeff': 2.0
        })
        scheduler.initialize(self.mission)

        # 运行调度
        result = scheduler.schedule()

        # 验证粒子群存在
        assert scheduler.swarm is not None
        # 验证全局最优已更新
        assert scheduler.global_best_fitness is not None

    def test_pso_get_parameters(self):
        """测试PSO获取参数"""
        config = {
            'num_particles': 25,
            'max_iterations': 150,
            'cognitive_coeff': 2.2,
            'social_coeff': 2.2,
            'inertia_weight': 0.85
        }
        scheduler = PSOScheduler(config)
        params = scheduler.get_parameters()

        assert params['num_particles'] == 25
        assert params['max_iterations'] == 150
        assert params['cognitive_coeff'] == 2.2
        assert params['social_coeff'] == 2.2
        assert params['inertia_weight'] == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
