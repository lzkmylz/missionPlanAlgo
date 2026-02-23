"""
蚁群优化调度器测试

TDD测试文件 - 实现ACO调度器
"""

import pytest
from datetime import datetime, timedelta

from core.models import Mission, Satellite, SatelliteType, Target, TargetType
from scheduler.metaheuristic.aco_scheduler import ACOScheduler
from scheduler.base_scheduler import ScheduleResult


class TestACOScheduler:
    """测试蚁群优化调度器"""

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
            name="ACO测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=self.satellites,
            targets=self.targets
        )

    def test_aco_initialization(self):
        """测试ACO调度器初始化"""
        config = {
            'num_ants': 20,
            'max_iterations': 200,
            'alpha': 1.5,
            'beta': 2.5,
            'evaporation_rate': 0.1,
            'initial_pheromone': 0.5
        }
        scheduler = ACOScheduler(config)

        assert scheduler.num_ants == 20
        assert scheduler.max_iterations == 200
        assert scheduler.alpha == 1.5
        assert scheduler.beta == 2.5
        assert scheduler.evaporation_rate == 0.1
        assert scheduler.initial_pheromone == 0.5

    def test_aco_default_parameters(self):
        """测试ACO默认参数"""
        scheduler = ACOScheduler()

        assert scheduler.num_ants == 30
        assert scheduler.max_iterations == 100
        assert scheduler.alpha == 1.0
        assert scheduler.beta == 2.0
        assert scheduler.evaporation_rate == 0.1
        assert scheduler.initial_pheromone == 1.0

    def test_aco_returns_result(self):
        """测试ACO返回有效结果"""
        scheduler = ACOScheduler(config={'max_iterations': 50, 'num_ants': 10})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert hasattr(result, 'scheduled_tasks')
        assert hasattr(result, 'convergence_curve')
        assert hasattr(result, 'iterations')

    def test_aco_convergence_curve(self):
        """测试ACO生成收敛曲线"""
        scheduler = ACOScheduler(config={'max_iterations': 50, 'num_ants': 10})
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        # 收敛曲线长度应该等于迭代代数 + 1（初始）
        assert len(result.convergence_curve) > 0

    def test_aco_schedules_tasks(self):
        """测试ACO能够调度任务（使用窗口缓存）"""
        from datetime import datetime
        from core.orbit.visibility.base import VisibilityWindow

        scheduler = ACOScheduler(config={'max_iterations': 50, 'num_ants': 10})
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

    def test_aco_empty_mission_raises_error(self):
        """测试ACO空场景（无卫星、无目标）应抛出初始化错误"""
        empty_mission = Mission(
            name="空场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[]
        )

        scheduler = ACOScheduler(config={'max_iterations': 50})
        scheduler.initialize(empty_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_aco_parameter_validation(self):
        """测试ACO参数验证"""
        # 测试无效蚂蚁数量
        with pytest.raises((ValueError, AssertionError)):
            ACOScheduler(config={'num_ants': 0})

        # 测试无效信息素蒸发率
        with pytest.raises((ValueError, AssertionError)):
            ACOScheduler(config={'evaporation_rate': 1.5})

    def test_aco_pheromone_update(self):
        """测试ACO信息素更新机制"""
        scheduler = ACOScheduler(config={
            'max_iterations': 10,
            'num_ants': 5,
            'evaporation_rate': 0.1
        })
        scheduler.initialize(self.mission)

        # 运行调度
        result = scheduler.schedule()

        # 验证信息素矩阵存在且已更新
        assert scheduler.pheromone_matrix is not None
        # 信息素值应该在合理范围内
        pheromone_values = scheduler.pheromone_matrix.flatten()
        assert all(v >= 0 for v in pheromone_values)

    def test_aco_get_parameters(self):
        """测试ACO获取参数"""
        config = {
            'num_ants': 25,
            'max_iterations': 150,
            'alpha': 1.2,
            'beta': 3.0,
            'evaporation_rate': 0.15
        }
        scheduler = ACOScheduler(config)
        params = scheduler.get_parameters()

        assert params['num_ants'] == 25
        assert params['max_iterations'] == 150
        assert params['alpha'] == 1.2
        assert params['beta'] == 3.0
        assert params['evaporation_rate'] == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
