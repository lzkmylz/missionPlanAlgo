"""
帕累托优化器单元测试

TDD测试文件 - 第20章设计实现
遵循测试先行原则：
1. 先写测试（RED）
2. 实现代码（GREEN）
3. 重构优化（REFACTOR）
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional

from core.processing.pareto_optimizer import (
    ObjectiveFunction,
    ParetoOptimizer,
)


class TestObjectiveFunction:
    """测试目标函数数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        obj = ObjectiveFunction(
            name="total_energy",
            weight=1.0,
            minimize=True
        )
        assert obj.name == "total_energy"
        assert obj.weight == 1.0
        assert obj.minimize is True

    def test_maximization_objective(self):
        """测试最大化目标"""
        obj = ObjectiveFunction(
            name="throughput",
            weight=0.8,
            minimize=False
        )
        assert obj.name == "throughput"
        assert obj.weight == 0.8
        assert obj.minimize is False

    def test_evaluate_method(self):
        """测试evaluate方法"""
        obj = ObjectiveFunction(
            name="test_energy",
            weight=1.0,
            minimize=True
        )

        # 测试评估解
        solution = np.array([1, 0, 1, 1, 0])
        tasks = [
            {'id': 'T1', 'energy': 10.0},
            {'id': 'T2', 'energy': 20.0},
            {'id': 'T3', 'energy': 15.0},
            {'id': 'T4', 'energy': 5.0},
            {'id': 'T5', 'energy': 25.0},
        ]

        # 评估：选择T1, T3, T4 -> 10 + 15 + 5 = 30
        result = obj.evaluate(solution, tasks, 'energy')
        assert result == 30.0

    def test_evaluate_empty_solution(self):
        """测试空解评估"""
        obj = ObjectiveFunction(
            name="test",
            weight=1.0,
            minimize=True
        )
        solution = np.array([])
        tasks = []
        result = obj.evaluate(solution, tasks, 'energy')
        assert result == 0.0

    def test_evaluate_all_zeros(self):
        """测试全零解评估"""
        obj = ObjectiveFunction(
            name="test",
            weight=1.0,
            minimize=True
        )
        solution = np.array([0, 0, 0])
        tasks = [
            {'id': 'T1', 'energy': 10.0},
            {'id': 'T2', 'energy': 20.0},
            {'id': 'T3', 'energy': 15.0},
        ]
        result = obj.evaluate(solution, tasks, 'energy')
        assert result == 0.0

    def test_evaluate_all_ones(self):
        """测试全一解评估"""
        obj = ObjectiveFunction(
            name="test",
            weight=1.0,
            minimize=True
        )
        solution = np.array([1, 1, 1])
        tasks = [
            {'id': 'T1', 'energy': 10.0},
            {'id': 'T2', 'energy': 20.0},
            {'id': 'T3', 'energy': 15.0},
        ]
        result = obj.evaluate(solution, tasks, 'energy')
        assert result == 45.0


class TestParetoOptimizerInitialization:
    """测试帕累托优化器初始化"""

    def test_initialization_with_objectives(self):
        """测试带目标函数的初始化"""
        objectives = [
            ObjectiveFunction(name="total_energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_time", weight=0.8, minimize=True),
            ObjectiveFunction(name="total_storage", weight=0.6, minimize=True),
        ]

        optimizer = ParetoOptimizer(objectives)

        assert optimizer.objectives == objectives
        assert len(optimizer.archive) == 0

    def test_initialization_empty_objectives(self):
        """测试空目标函数列表初始化"""
        optimizer = ParetoOptimizer([])
        assert optimizer.objectives == []
        assert optimizer.archive == []

    def test_initialization_five_objectives(self):
        """测试5个目标函数的初始化（设计文档要求）"""
        objectives = [
            ObjectiveFunction(name="total_energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_time", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_storage", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_bandwidth", weight=1.0, minimize=True),
            ObjectiveFunction(name="thermal_load", weight=1.0, minimize=True),
        ]

        optimizer = ParetoOptimizer(objectives)

        assert len(optimizer.objectives) == 5
        assert optimizer.objectives[0].name == "total_energy"
        assert optimizer.objectives[1].name == "total_time"
        assert optimizer.objectives[2].name == "total_storage"
        assert optimizer.objectives[3].name == "total_bandwidth"
        assert optimizer.objectives[4].name == "thermal_load"


class TestParetoOptimizerEvaluateSolution:
    """测试解评估功能"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="total_energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_time", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_storage", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_bandwidth", weight=1.0, minimize=True),
            ObjectiveFunction(name="thermal_load", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

        self.satellites = [
            {
                'id': 'SAT-01',
                'has_ai_accelerator': True,
                'ai_power_wh': 5.0,
                'ai_processing_time': 10.0,
                'ai_thermal_load': 2.0,
                'downlink_power_wh': 50.0,
            },
            {
                'id': 'SAT-02',
                'has_ai_accelerator': False,
                'downlink_power_wh': 60.0,
            }
        ]

        self.imaging_tasks = [
            {
                'id': 'IMG-001',
                'satellite_id': 'SAT-01',
                'data_size_gb': 2.0,
                'compressed_size_kb': 50.0,
                'downlink_time': 100.0,
            },
            {
                'id': 'IMG-002',
                'satellite_id': 'SAT-01',
                'data_size_gb': 1.5,
                'compressed_size_kb': 30.0,
                'downlink_time': 75.0,
            },
            {
                'id': 'IMG-003',
                'satellite_id': 'SAT-02',
                'data_size_gb': 3.0,
                'compressed_size_kb': 100.0,
                'downlink_time': 150.0,
            },
        ]

    def test_evaluate_solution_all_onboard(self):
        """测试全在轨处理解评估"""
        # 所有任务都在轨处理
        solution = np.array([1, 1, 1])

        fitness = self.optimizer._evaluate_solution(
            solution, self.imaging_tasks, self.satellites
        )

        # 验证返回5个目标值
        assert len(fitness) == 5
        assert all(f >= 0 for f in fitness)  # 所有值非负

        # SAT-01的两个任务在轨处理
        # SAT-02无AI加速器，只能下传
        # energy: 5.0 + 5.0 + 60.0 = 70.0
        assert fitness[0] == 70.0

    def test_evaluate_solution_all_downlink(self):
        """测试全下传解评估"""
        solution = np.array([0, 0, 0])

        fitness = self.optimizer._evaluate_solution(
            solution, self.imaging_tasks, self.satellites
        )

        assert len(fitness) == 5
        # energy: 50.0 + 50.0 + 60.0 = 160.0
        assert fitness[0] == 160.0

    def test_evaluate_solution_mixed(self):
        """测试混合解评估"""
        # 第一个任务在轨处理，其余下传
        solution = np.array([1, 0, 0])

        fitness = self.optimizer._evaluate_solution(
            solution, self.imaging_tasks, self.satellites
        )

        assert len(fitness) == 5
        # energy: 5.0 (onboard) + 50.0 (downlink) + 60.0 (downlink) = 115.0
        assert fitness[0] == 115.0

    def test_evaluate_solution_single_task(self):
        """测试单任务解评估"""
        tasks = [self.imaging_tasks[0]]
        solution = np.array([1])

        fitness = self.optimizer._evaluate_solution(solution, tasks, self.satellites)

        assert len(fitness) == 5
        assert fitness[0] == 5.0  # onboard energy

    def test_evaluate_solution_empty_tasks(self):
        """测试空任务列表解评估"""
        solution = np.array([])
        fitness = self.optimizer._evaluate_solution(solution, [], self.satellites)

        assert len(fitness) == 5
        assert all(f == 0.0 for f in fitness)

    def test_evaluate_solution_satellite_without_ai(self):
        """测试无AI加速器卫星的任务评估"""
        # SAT-02无AI加速器，即使在轨处理决策也强制下传
        solution = np.array([0, 0, 1])  # 尝试让SAT-02的任务在轨处理

        fitness = self.optimizer._evaluate_solution(
            solution, self.imaging_tasks, self.satellites
        )

        # 但SAT-02无AI加速器，所以实际是下传代价
        # energy: 50.0 + 50.0 + 60.0 = 160.0
        assert fitness[0] == 160.0


class TestParetoOptimizerEvaluatePopulation:
    """测试种群评估功能"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="total_energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_time", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

        self.satellites = [
            {
                'id': 'SAT-01',
                'has_ai_accelerator': True,
                'ai_power_wh': 5.0,
                'ai_processing_time': 10.0,
                'ai_thermal_load': 2.0,
                'downlink_power_wh': 50.0,
            }
        ]

        self.imaging_tasks = [
            {
                'id': 'IMG-001',
                'satellite_id': 'SAT-01',
                'data_size_gb': 2.0,
                'compressed_size_kb': 50.0,
                'downlink_time': 100.0,
            },
            {
                'id': 'IMG-002',
                'satellite_id': 'SAT-01',
                'data_size_gb': 1.5,
                'compressed_size_kb': 30.0,
                'downlink_time': 75.0,
            },
        ]

    def test_evaluate_population(self):
        """测试种群评估"""
        population = np.array([
            [1, 1],  # 全在轨
            [0, 0],  # 全下传
            [1, 0],  # 混合
        ])

        fitness = self.optimizer._evaluate_population(
            population, self.imaging_tasks, self.satellites
        )

        # 验证形状: (population_size, n_objectives)
        assert fitness.shape == (3, 2)

        # 验证每个解都被评估
        assert fitness[0, 0] == 10.0  # 全在轨: 5+5
        assert fitness[1, 0] == 100.0  # 全下传: 50+50
        assert fitness[2, 0] == 55.0   # 混合: 5+50


class TestParetoOptimizerNonDominatedSort:
    """测试非支配排序功能"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="time", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

    def test_non_dominated_sort_two_objectives(self):
        """测试双目标非支配排序"""
        # 4个解，2个目标（最小化）
        # 支配关系分析（最小化问题，值越小越好）：
        # 解0 [1,2]: 被解1 [1,1] 支配（1<=1, 2>1）
        # 解1 [1,1]: 帕累托前沿，支配解0和解3
        # 解2 [2,1]: 被解1 [1,1] 支配（2>1, 1<=1）- 实际上1<2所以解1支配解2
        # 解3 [2,2]: 被解1支配
        fitness = np.array([
            [1.0, 2.0],   # 解0
            [1.0, 1.0],   # 解1: 帕累托前沿
            [2.0, 1.0],   # 解2: 被解1支配
            [2.0, 2.0],   # 解3: 被解1支配
        ])

        fronts = self.optimizer._non_dominated_sort(fitness)

        # 第0层应该只有解1（不被任何解支配）
        assert fronts[0] == [1]

        # 其他解被解1支配，应该在第1层
        assert 0 in fronts[1]
        assert 2 in fronts[1]
        # 解3被解1、解0和解2支配，所以应该在第2层
        assert 3 in fronts[2]

    def test_non_dominated_sort_single_solution(self):
        """测试单解非支配排序"""
        fitness = np.array([[1.0, 2.0]])

        fronts = self.optimizer._non_dominated_sort(fitness)

        assert len(fronts) == 1
        assert fronts[0] == [0]

    def test_non_dominated_sort_all_non_dominated(self):
        """测试全部非支配解排序"""
        # 3个解互不支配
        fitness = np.array([
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ])

        fronts = self.optimizer._non_dominated_sort(fitness)

        # 所有解都在第0层
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}

    def test_non_dominated_sort_dominance_chain(self):
        """测试支配链排序"""
        # 解0支配解1，解1支配解2
        fitness = np.array([
            [1.0, 1.0],   # 第0层
            [2.0, 2.0],   # 第1层
            [3.0, 3.0],   # 第2层
        ])

        fronts = self.optimizer._non_dominated_sort(fitness)

        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]

    def test_dominates_function(self):
        """测试支配判断函数"""
        # a支配b: a在所有目标上<=b，且至少一个目标<
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        assert self.optimizer._dominates(a, b) == True

        # b不支配a
        assert self.optimizer._dominates(b, a) == False

        # 相等不支配
        c = np.array([1.0, 2.0])
        assert self.optimizer._dominates(a, c) == False

        # 互不占优
        d = np.array([2.0, 1.0])
        assert self.optimizer._dominates(a, d) == False
        assert self.optimizer._dominates(d, a) == False


class TestParetoOptimizerGeneticOperators:
    """测试遗传操作"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="energy", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

    def test_tournament_selection(self):
        """测试锦标赛选择"""
        population = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ])

        fitness = np.array([
            [10.0],
            [5.0],
            [15.0],
            [2.0],
        ])

        fronts = [[0, 1, 2, 3]]

        # 锦标赛选择应返回与种群相同大小的后代
        offspring = self.optimizer._genetic_operators(population, fronts, fitness)

        assert offspring.shape[0] == population.shape[0]
        assert offspring.shape[1] == population.shape[1]

    def test_crossover(self):
        """测试交叉操作"""
        parent1 = np.array([1, 0, 1, 0, 1])
        parent2 = np.array([0, 1, 0, 1, 0])

        # 多次测试以确保交叉发生
        crossover_happened = False
        for _ in range(100):
            child1, child2 = self.optimizer._crossover(parent1, parent2)

            # 验证子代长度正确
            assert len(child1) == len(parent1)
            assert len(child2) == len(parent2)

            # 验证子代元素来自父代
            assert all(gene in [0, 1] for gene in child1)
            assert all(gene in [0, 1] for gene in child2)

            # 检查是否有交叉发生
            if not np.array_equal(child1, parent1) or not np.array_equal(child2, parent2):
                crossover_happened = True

        assert crossover_happened, "交叉操作应该产生不同的子代"

    def test_mutation(self):
        """测试变异操作"""
        individual = np.array([1, 0, 1, 0, 1, 0, 1, 0])

        # 多次测试以确保变异发生
        mutation_happened = False
        for _ in range(100):
            mutated = self.optimizer._mutate(individual.copy(), mutation_rate=0.5)

            # 验证长度不变
            assert len(mutated) == len(individual)

            # 验证元素为0或1
            assert all(gene in [0, 1] for gene in mutated)

            # 检查是否有变异发生
            if not np.array_equal(mutated, individual):
                mutation_happened = True

        assert mutation_happened, "变异操作应该改变一些基因"

    def test_mutation_zero_rate(self):
        """测试零变异率"""
        individual = np.array([1, 0, 1, 0, 1])

        for _ in range(10):
            mutated = self.optimizer._mutate(individual.copy(), mutation_rate=0.0)
            assert np.array_equal(mutated, individual)


class TestParetoOptimizerSelection:
    """测试选择下一代功能"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="time", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

    def test_select_next_generation(self):
        """测试选择下一代"""
        # 合并种群
        combined = np.array([
            [1, 0, 1],  # 第0层
            [0, 1, 0],  # 第0层
            [1, 1, 1],  # 第1层
            [0, 0, 0],  # 第1层
        ])

        fitness = np.array([
            [1.0, 2.0],  # 较好
            [2.0, 1.0],  # 较好
            [3.0, 3.0],  # 较差
            [4.0, 4.0],  # 较差
        ])

        fronts = [[0, 1], [2, 3]]

        # 选择2个个体
        selected = self.optimizer._select_next_generation(
            combined, fitness, fronts, population_size=2
        )

        # 验证选择数量正确
        assert selected.shape[0] == 2

        # 第0层的个体应该被优先选择
        # 注意：由于拥挤度计算，具体选择哪个可能有变化
        # 但应该优先从第0层选择

    def test_select_next_generation_elite_preservation(self):
        """测试精英保留"""
        combined = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ])

        fitness = np.array([
            [1.0, 1.0],  # 最优
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])

        fronts = [[0], [1], [2], [3]]

        # 选择1个个体，应该是最优的
        selected = self.optimizer._select_next_generation(
            combined, fitness, fronts, population_size=1
        )

        assert selected.shape[0] == 1
        assert np.array_equal(selected[0], combined[0])

    def test_crowding_distance(self):
        """测试拥挤距离计算"""
        fitness = np.array([
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
        ])

        front = [0, 1, 2, 3, 4]

        distances = self.optimizer._calculate_crowding_distance(fitness, front)

        # 边界点的拥挤距离为无穷大
        assert distances[0] == float('inf')
        assert distances[4] == float('inf')

        # 中间点应该有有限距离
        assert distances[1] > 0
        assert distances[2] > 0
        assert distances[3] > 0


class TestParetoOptimizerDecodeSolution:
    """测试解码解功能"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="energy", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

        self.imaging_tasks = [
            {'id': 'IMG-001', 'name': 'Task 1'},
            {'id': 'IMG-002', 'name': 'Task 2'},
            {'id': 'IMG-003', 'name': 'Task 3'},
        ]

    def test_decode_solution(self):
        """测试解码解"""
        solution = np.array([1, 0, 1])

        decoded = self.optimizer._decode_solution(solution, self.imaging_tasks)

        # 验证返回正确的任务列表
        assert len(decoded['onboard_tasks']) == 2
        assert len(decoded['downlink_tasks']) == 1

        assert decoded['onboard_tasks'][0]['id'] == 'IMG-001'
        assert decoded['onboard_tasks'][1]['id'] == 'IMG-003'
        assert decoded['downlink_tasks'][0]['id'] == 'IMG-002'

    def test_decode_solution_all_onboard(self):
        """测试全在轨处理解码"""
        solution = np.array([1, 1, 1])

        decoded = self.optimizer._decode_solution(solution, self.imaging_tasks)

        assert len(decoded['onboard_tasks']) == 3
        assert len(decoded['downlink_tasks']) == 0

    def test_decode_solution_all_downlink(self):
        """测试全下传解码"""
        solution = np.array([0, 0, 0])

        decoded = self.optimizer._decode_solution(solution, self.imaging_tasks)

        assert len(decoded['onboard_tasks']) == 0
        assert len(decoded['downlink_tasks']) == 3

    def test_decode_solution_empty(self):
        """测试空解解码"""
        solution = np.array([])

        decoded = self.optimizer._decode_solution(solution, [])

        assert len(decoded['onboard_tasks']) == 0
        assert len(decoded['downlink_tasks']) == 0


class TestParetoOptimizerOptimize:
    """测试主优化方法"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="total_energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_time", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_storage", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_bandwidth", weight=1.0, minimize=True),
            ObjectiveFunction(name="thermal_load", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

        self.satellites = [
            {
                'id': 'SAT-01',
                'has_ai_accelerator': True,
                'ai_power_wh': 5.0,
                'ai_processing_time': 10.0,
                'ai_thermal_load': 2.0,
                'downlink_power_wh': 50.0,
            },
            {
                'id': 'SAT-02',
                'has_ai_accelerator': True,
                'ai_power_wh': 6.0,
                'ai_processing_time': 12.0,
                'ai_thermal_load': 2.5,
                'downlink_power_wh': 60.0,
            }
        ]

        self.imaging_tasks = [
            {
                'id': f'IMG-{i:03d}',
                'satellite_id': f'SAT-{(i % 2) + 1:02d}',
                'data_size_gb': 1.0 + i * 0.5,
                'compressed_size_kb': 50.0 + i * 10,
                'downlink_time': 50.0 + i * 10,
            }
            for i in range(5)
        ]

    def test_optimize_basic(self):
        """测试基本优化流程"""
        result = self.optimizer.optimize(
            self.imaging_tasks,
            self.satellites,
            population_size=20,
            generations=5
        )

        # 验证返回帕累托前沿解集
        assert isinstance(result, list)
        assert len(result) > 0

        # 验证每个解的结构
        for solution in result:
            assert 'decision_vector' in solution
            assert 'objectives' in solution
            assert 'decoded' in solution

            # 验证决策向量长度正确
            assert len(solution['decision_vector']) == len(self.imaging_tasks)

            # 验证目标值数量正确
            assert len(solution['objectives']) == len(self.objectives)

    def test_optimize_empty_tasks(self):
        """测试空任务列表优化"""
        result = self.optimizer.optimize(
            [],
            self.satellites,
            population_size=10,
            generations=2
        )

        # 空任务应该返回空列表或包含空解的列表
        assert isinstance(result, list)

    def test_optimize_single_task(self):
        """测试单任务优化"""
        single_task = [self.imaging_tasks[0]]

        result = self.optimizer.optimize(
            single_task,
            self.satellites,
            population_size=10,
            generations=3
        )

        assert isinstance(result, list)
        assert len(result) > 0

        # 单任务只有两种可能：在轨或下传
        # 但帕累托前沿可能包含多个解（如果互不占优）
        for solution in result:
            assert len(solution['decision_vector']) == 1

    def test_optimize_small_population(self):
        """测试小种群优化"""
        result = self.optimizer.optimize(
            self.imaging_tasks[:2],
            self.satellites,
            population_size=4,
            generations=2
        )

        assert isinstance(result, list)

    def test_optimize_result_quality(self):
        """测试优化结果质量"""
        result = self.optimizer.optimize(
            self.imaging_tasks,
            self.satellites,
            population_size=30,
            generations=10
        )

        # 验证帕累托前沿中的解互不支配
        objectives_matrix = np.array([sol['objectives'] for sol in result])

        for i in range(len(objectives_matrix)):
            for j in range(i + 1, len(objectives_matrix)):
                # 前沿中的解不应该互相支配
                assert not self.optimizer._dominates(objectives_matrix[i], objectives_matrix[j])
                assert not self.optimizer._dominates(objectives_matrix[j], objectives_matrix[i])


class TestParetoOptimizerEdgeCases:
    """测试边界条件"""

    def setup_method(self):
        """设置测试数据"""
        self.objectives = [
            ObjectiveFunction(name="energy", weight=1.0, minimize=True),
        ]
        self.optimizer = ParetoOptimizer(self.objectives)

    def test_no_satellites(self):
        """测试无卫星情况"""
        tasks = [{'id': 'IMG-001', 'satellite_id': 'SAT-01', 'data_size_gb': 1.0}]

        with pytest.raises((ValueError, StopIteration)):
            self.optimizer.optimize(tasks, [], population_size=10, generations=2)

    def test_satellite_without_tasks(self):
        """测试有卫星但无对应任务"""
        tasks = [{'id': 'IMG-001', 'satellite_id': 'SAT-03', 'data_size_gb': 1.0}]
        satellites = [{'id': 'SAT-01', 'has_ai_accelerator': True, 'ai_power_wh': 5.0}]

        with pytest.raises((ValueError, StopIteration)):
            self.optimizer.optimize(tasks, satellites, population_size=10, generations=2)

    def test_very_large_population(self):
        """测试超大种群"""
        tasks = [
            {'id': f'IMG-{i}', 'satellite_id': 'SAT-01', 'data_size_gb': 1.0}
            for i in range(100)
        ]
        satellites = [{'id': 'SAT-01', 'has_ai_accelerator': True, 'ai_power_wh': 5.0}]

        result = self.optimizer.optimize(
            tasks, satellites, population_size=50, generations=2
        )

        assert isinstance(result, list)

    def test_many_objectives(self):
        """测试多目标（超过5个）"""
        objectives = [
            ObjectiveFunction(name=f"obj_{i}", weight=1.0, minimize=True)
            for i in range(10)
        ]
        optimizer = ParetoOptimizer(objectives)

        tasks = [{'id': 'IMG-001', 'satellite_id': 'SAT-01', 'data_size_gb': 1.0}]
        satellites = [{'id': 'SAT-01', 'has_ai_accelerator': True, 'ai_power_wh': 5.0}]

        result = optimizer.optimize(tasks, satellites, population_size=10, generations=2)

        assert isinstance(result, list)
        for sol in result:
            assert len(sol['objectives']) == 10


class TestParetoOptimizerIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 创建目标函数（5个目标）
        objectives = [
            ObjectiveFunction(name="total_energy", weight=1.0, minimize=True),
            ObjectiveFunction(name="total_time", weight=0.9, minimize=True),
            ObjectiveFunction(name="total_storage", weight=0.7, minimize=True),
            ObjectiveFunction(name="total_bandwidth", weight=1.0, minimize=True),
            ObjectiveFunction(name="thermal_load", weight=0.8, minimize=True),
        ]

        optimizer = ParetoOptimizer(objectives)

        # 2. 定义卫星
        satellites = [
            {
                'id': 'SAT-01',
                'has_ai_accelerator': True,
                'ai_power_wh': 5.0,
                'ai_processing_time': 10.0,
                'ai_thermal_load': 2.0,
                'downlink_power_wh': 50.0,
            },
            {
                'id': 'SAT-02',
                'has_ai_accelerator': True,
                'ai_power_wh': 4.0,
                'ai_processing_time': 8.0,
                'ai_thermal_load': 1.5,
                'downlink_power_wh': 45.0,
            },
            {
                'id': 'SAT-03',
                'has_ai_accelerator': False,
                'downlink_power_wh': 55.0,
            }
        ]

        # 3. 定义成像任务
        imaging_tasks = [
            {
                'id': f'IMG-{i:03d}',
                'satellite_id': f'SAT-{(i % 3) + 1:02d}',
                'data_size_gb': 1.0 + (i % 3) * 0.5,
                'compressed_size_kb': 30.0 + (i % 5) * 10,
                'downlink_time': 40.0 + (i % 4) * 15,
            }
            for i in range(10)
        ]

        # 4. 执行优化
        result = optimizer.optimize(
            imaging_tasks,
            satellites,
            population_size=50,
            generations=20
        )

        # 5. 验证结果
        assert len(result) > 0

        # 每个解都应该有效
        for solution in result:
            # 决策向量是二进制
            assert all(gene in [0, 1] for gene in solution['decision_vector'])

            # 目标值非负
            assert all(obj >= 0 for obj in solution['objectives'])

            # 解码结果正确
            decoded = solution['decoded']
            total_tasks = len(decoded['onboard_tasks']) + len(decoded['downlink_tasks'])
            assert total_tasks == len(imaging_tasks)

        # 验证存档已更新
        assert len(optimizer.archive) > 0
