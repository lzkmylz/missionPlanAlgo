"""
测试Tabu搜索调度器

M2: Tabu搜索算法测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from scheduler.metaheuristic.tabu_scheduler import TabuScheduler, TabuSolution
from scheduler.base_scheduler import ScheduleResult, TaskFailureReason


class TestTabuSolution:
    """测试TabuSolution数据类"""

    def test_tabu_solution_creation(self):
        """测试TabuSolution创建"""
        solution = TabuSolution(
            assignment=[0, 1, 2],
            fitness=100.0,
            unscheduled_count=5
        )

        assert solution.assignment == [0, 1, 2]
        assert solution.fitness == 100.0
        assert solution.unscheduled_count == 5

    def test_tabu_solution_comparison(self):
        """测试TabuSolution比较"""
        sol1 = TabuSolution(assignment=[0, 1], fitness=100.0, unscheduled_count=5)
        sol2 = TabuSolution(assignment=[1, 0], fitness=150.0, unscheduled_count=3)
        sol3 = TabuSolution(assignment=[2, 2], fitness=100.0, unscheduled_count=5)

        assert sol2 > sol1  # 更高的fitness更好
        assert sol1 == sol3  # 相同的fitness和unscheduled_count相等


class TestTabuSchedulerInit:
    """测试TabuScheduler初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        scheduler = TabuScheduler()

        assert scheduler.name == "Tabu"
        assert scheduler.tabu_tenure == 10
        assert scheduler.max_iterations == 100
        assert scheduler.neighborhood_size == 20
        assert scheduler.aspiration_threshold == 0.05

    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        config = {
            'tabu_tenure': 15,
            'max_iterations': 200,
            'neighborhood_size': 30,
            'aspiration_threshold': 0.1
        }
        scheduler = TabuScheduler(config)

        assert scheduler.tabu_tenure == 15
        assert scheduler.max_iterations == 200
        assert scheduler.neighborhood_size == 30
        assert scheduler.aspiration_threshold == 0.1

    def test_invalid_parameters(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            TabuScheduler({'tabu_tenure': -1})

        with pytest.raises(ValueError):
            TabuScheduler({'max_iterations': 0})

        with pytest.raises(ValueError):
            TabuScheduler({'aspiration_threshold': 1.5})

    def test_get_parameters(self):
        """测试获取参数"""
        scheduler = TabuScheduler()
        params = scheduler.get_parameters()

        assert 'tabu_tenure' in params
        assert 'max_iterations' in params
        assert 'neighborhood_size' in params
        assert 'aspiration_threshold' in params


class TestTabuSchedulerBasic:
    """测试TabuScheduler基本功能"""

    def test_empty_mission_raises_error(self):
        """测试空任务场景（无卫星、无目标）应抛出初始化错误"""
        scheduler = TabuScheduler()

        mock_mission = Mock()
        mock_mission.targets = []
        mock_mission.satellites = []
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_no_satellites_raises_error(self):
        """测试无卫星场景应抛出初始化错误"""
        scheduler = TabuScheduler()

        mock_target = Mock()
        mock_target.id = "TARGET-01"

        mock_mission = Mock()
        mock_mission.targets = [mock_target]
        mock_mission.satellites = []
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_not_initialized(self):
        """测试未初始化调用schedule"""
        scheduler = TabuScheduler()

        with pytest.raises(RuntimeError):
            scheduler.schedule()


class TestTabuSchedulerWithCache:
    """测试带窗口缓存的TabuScheduler"""

    def create_mock_mission(self, num_targets=5, num_satellites=2):
        """创建模拟任务场景"""
        targets = []
        for i in range(num_targets):
            target = Mock()
            target.id = f"TARGET-{i:02d}"
            targets.append(target)

        satellites = []
        for i in range(num_satellites):
            sat = Mock()
            sat.id = f"SAT-{i:02d}"
            sat.capabilities = Mock()
            sat.capabilities.storage_capacity = 100
            satellites.append(sat)

        mission = Mock()
        mission.targets = targets
        mission.satellites = satellites
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)

        return mission

    def create_mock_window_cache(self, mission):
        """创建模拟窗口缓存"""
        cache = Mock()

        windows_map = {}
        for sat in mission.satellites:
            for target in mission.targets:
                # 模拟50%的可见性概率
                if hash(target.id) % 2 == 0:
                    from core.orbit.visibility.base import VisibilityWindow
                    window = VisibilityWindow(
                        satellite_id=sat.id,
                        target_id=target.id,
                        start_time=datetime(2024, 1, 1, 10, 0, 0),
                        end_time=datetime(2024, 1, 1, 10, 5, 0),
                        max_elevation=45.0,
                        quality_score=0.8
                    )
                    windows_map[(sat.id, target.id)] = [window]

        def get_windows(sat_id, target_id):
            return windows_map.get((sat_id, target_id), [])

        cache.get_windows = get_windows
        return cache

    def test_schedule_with_window_cache(self):
        """测试使用窗口缓存进行调度"""
        scheduler = TabuScheduler({
            'max_iterations': 50,
            'neighborhood_size': 10
        })

        mission = self.create_mock_mission(num_targets=10, num_satellites=3)
        cache = self.create_mock_window_cache(mission)

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert result.computation_time >= 0
        assert result.iterations <= 50

    def test_tabu_list_prevents_cycles(self):
        """测试禁忌表防止循环"""
        scheduler = TabuScheduler({
            'max_iterations': 20,
            'tabu_tenure': 5,
            'neighborhood_size': 5
        })

        mission = self.create_mock_mission(num_targets=5, num_satellites=2)
        cache = self.create_mock_window_cache(mission)

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        result = scheduler.schedule()

        # 验证禁忌表被使用
        assert len(scheduler.tabu_list) <= 5  # 不超过tenure

    def test_aspiration_criteria(self):
        """测试 aspiration criteria 允许突破禁忌"""
        scheduler = TabuScheduler({
            'max_iterations': 30,
            'aspiration_threshold': 0.01
        })

        mission = self.create_mock_mission(num_targets=8, num_satellites=2)
        cache = self.create_mock_window_cache(mission)

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        result = scheduler.schedule()

        # 调度应该成功完成
        assert isinstance(result, ScheduleResult)


class TestTabuSchedulerNeighborhood:
    """测试邻域搜索功能"""

    def test_generate_neighbors(self):
        """测试生成邻居解"""
        scheduler = TabuScheduler({'neighborhood_size': 10})

        current = TabuSolution(
            assignment=[0, 1, 0, 1, 0],
            fitness=100.0,
            unscheduled_count=2
        )

        neighbors = scheduler._generate_neighbors(current, num_satellites=2)

        assert len(neighbors) > 0
        assert len(neighbors) <= 10

        # 验证邻居解与当前解不同
        for neighbor in neighbors:
            assert neighbor.assignment != current.assignment
            # 只有一个位置不同
            diff_count = sum(1 for a, b in zip(neighbor.assignment, current.assignment) if a != b)
            assert diff_count == 1

    def test_neighbor_fitness_evaluation(self):
        """测试邻居解的适应度评估"""
        scheduler = TabuScheduler()

        mock_mission = Mock()
        mock_mission.targets = [Mock() for _ in range(3)]
        mock_mission.satellites = [Mock() for _ in range(2)]

        scheduler.initialize(mock_mission)

        solution = TabuSolution(
            assignment=[0, 1, 0],
            fitness=0.0,
            unscheduled_count=0
        )

        # 模拟窗口缓存
        scheduler.window_cache = Mock()
        scheduler.window_cache.get_windows = Mock(return_value=[])

        fitness = scheduler._evaluate_solution(solution)

        assert isinstance(fitness, float)


class TestTabuSchedulerConvergence:
    """测试收敛性"""

    def test_convergence_curve(self):
        """测试收敛曲线记录"""
        scheduler = TabuScheduler({'max_iterations': 10})

        mock_mission = Mock()
        mock_mission.targets = [Mock() for _ in range(3)]
        mock_mission.satellites = [Mock() for _ in range(2)]
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)

        # 模拟窗口缓存
        scheduler.window_cache = Mock()
        scheduler.window_cache.get_windows = Mock(return_value=[])

        result = scheduler.schedule()

        # 验证收敛曲线
        assert hasattr(result, 'convergence_curve')
        assert len(result.convergence_curve) > 0

    def test_improvement_over_iterations(self):
        """测试迭代过程中解的改进"""
        scheduler = TabuScheduler({'max_iterations': 20})

        mock_mission = Mock()
        mock_mission.targets = [Mock() for _ in range(5)]
        mock_mission.satellites = [Mock() for _ in range(2)]
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)
        scheduler.window_cache = Mock()
        scheduler.window_cache.get_windows = Mock(return_value=[])

        result = scheduler.schedule()

        # 收敛曲线应该非递减（记录的是最优值）
        curve = result.convergence_curve
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i-1] - 1e-6, \
                f"Convergence curve should be non-decreasing, but {curve[i]} < {curve[i-1]}"


class TestTabuSchedulerEdgeCases:
    """测试边界情况"""

    def test_single_task(self):
        """测试单任务场景"""
        scheduler = TabuScheduler({'max_iterations': 10})

        mock_mission = Mock()
        mock_target = Mock()
        mock_target.id = "TARGET-01"
        mock_mission.targets = [mock_target]

        mock_sat = Mock()
        mock_sat.id = "SAT-01"
        mock_mission.satellites = [mock_sat]
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)

        # 模拟窗口
        from core.orbit.visibility.base import VisibilityWindow
        scheduler.window_cache = Mock()
        scheduler.window_cache.get_windows = Mock(return_value=[
            VisibilityWindow(
                satellite_id="SAT-01",
                target_id="TARGET-01",
                start_time=datetime(2024, 1, 1, 10, 0, 0),
                end_time=datetime(2024, 1, 1, 10, 5, 0),
                max_elevation=45.0,
                quality_score=0.9
            )
        ])

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_single_satellite(self):
        """测试单卫星场景"""
        scheduler = TabuScheduler({'max_iterations': 10})

        mock_mission = Mock()
        mock_mission.targets = [Mock() for _ in range(5)]
        mock_mission.satellites = [Mock()]
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)
        scheduler.window_cache = Mock()
        scheduler.window_cache.get_windows = Mock(return_value=[])

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_large_neighborhood(self):
        """测试大邻域场景"""
        scheduler = TabuScheduler({
            'neighborhood_size': 100,
            'max_iterations': 5
        })

        mock_mission = Mock()
        mock_mission.targets = [Mock() for _ in range(20)]
        mock_mission.satellites = [Mock() for _ in range(5)]
        mock_mission.start_time = datetime(2024, 1, 1)

        scheduler.initialize(mock_mission)
        scheduler.window_cache = Mock()
        scheduler.window_cache.get_windows = Mock(return_value=[])

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_zero_tabu_tenure(self):
        """测试零禁忌期限（特殊情况）"""
        # 0值应该抛出异常，因为禁忌期限必须为正
        with pytest.raises(ValueError):
            TabuScheduler({'tabu_tenure': 0})
