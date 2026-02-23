"""
调度器注册表测试

TDD测试 - 修复main.py中硬编码算法选择问题
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler.base_scheduler import BaseScheduler, ScheduleResult


class TestSchedulerRegistry:
    """测试调度器注册表"""

    def test_registry_initialization(self):
        """测试注册表初始化"""
        from main import get_scheduler
        # 应该能获取到调度器
        scheduler = get_scheduler('greedy', {})
        assert scheduler is not None
        assert isinstance(scheduler, BaseScheduler)

    def test_greedy_scheduler_available(self):
        """测试贪心调度器可用"""
        from main import get_scheduler
        from scheduler.greedy.greedy_scheduler import GreedyScheduler

        scheduler = get_scheduler('greedy', {})
        assert isinstance(scheduler, GreedyScheduler)

    def test_ga_scheduler_available(self):
        """测试GA调度器可用"""
        from main import get_scheduler
        from scheduler.metaheuristic.ga_scheduler import GAScheduler

        scheduler = get_scheduler('ga', {})
        assert isinstance(scheduler, GAScheduler)

    def test_edd_scheduler_available(self):
        """测试EDD调度器可用（设计文档要求）"""
        from main import get_scheduler

        scheduler = get_scheduler('edd', {})
        assert scheduler is not None
        assert isinstance(scheduler, BaseScheduler)

    def test_spt_scheduler_available(self):
        """测试SPT调度器可用（设计文档要求）"""
        from main import get_scheduler

        scheduler = get_scheduler('spt', {})
        assert scheduler is not None
        assert isinstance(scheduler, BaseScheduler)

    def test_case_insensitive_algorithm_names(self):
        """测试算法名称大小写不敏感"""
        from main import get_scheduler

        # 各种大小写组合都应该工作
        greedy1 = get_scheduler('greedy', {})
        greedy2 = get_scheduler('GREEDY', {})
        greedy3 = get_scheduler('Greedy', {})

        assert greedy1.name == greedy2.name == greedy3.name

    def test_unknown_algorithm_raises_error(self):
        """测试未知算法抛出错误"""
        from main import get_scheduler

        with pytest.raises(ValueError) as exc_info:
            get_scheduler('unknown_algorithm', {})

        assert 'unknown_algorithm' in str(exc_info.value)

    def test_algorithm_config_passed(self):
        """测试算法配置正确传递"""
        from main import get_scheduler

        config = {'population_size': 100, 'generations': 50}
        scheduler = get_scheduler('ga', config)

        assert scheduler.config == config


class TestValidateSolution:
    """测试解验证功能"""

    def test_validate_solution_exists(self):
        """测试validate_solution方法存在"""
        from main import get_scheduler

        scheduler = get_scheduler('greedy', {})
        assert hasattr(scheduler, 'validate_solution')
        assert callable(getattr(scheduler, 'validate_solution'))

    def test_validate_solution_returns_bool(self):
        """测试validate_solution返回布尔值"""
        from main import get_scheduler
        from core.models import Mission, Satellite, Target
        from datetime import datetime

        # 创建一个简单的mission
        mission = Mission(
            name="测试任务",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            satellites=[],
            targets=[]
        )

        scheduler = get_scheduler('greedy', {})
        scheduler.initialize(mission)

        # 创建一个空的调度结果
        result = ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=0.0,
            iterations=0,
            convergence_curve=[]
        )

        is_valid = scheduler.validate_solution(result)
        assert isinstance(is_valid, bool)


class TestConvergenceOutput:
    """测试收敛曲线输出"""

    def test_convergence_curve_in_result(self):
        """测试调度结果包含收敛曲线"""
        from main import get_scheduler
        from core.models import Mission, Satellite, Target, SatelliteType
        from datetime import datetime

        # 创建简单场景
        satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )

        target = Target(
            id="TGT-01",
            name="测试目标",
            latitude=0.0,
            longitude=0.0
        )

        mission = Mission(
            name="测试任务",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            satellites=[satellite],
            targets=[target]
        )

        scheduler = get_scheduler('greedy', {})
        scheduler.initialize(mission)

        result = scheduler.schedule()

        # GA等元启发式算法应该有收敛曲线
        assert hasattr(result, 'convergence_curve')
        assert isinstance(result.convergence_curve, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
