"""
帕累托优化器

第20章：星载边缘计算
实现多目标帕累托优化，用于求解在轨处理 vs 原始下传的帕累托最优解集

算法基于NSGA-II（非支配排序遗传算法II）:
1. 非支配排序分层
2. 拥挤距离计算
3. 锦标赛选择
4. 单点交叉
5. 位变异
6. 精英保留机制
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class ObjectiveFunction:
    """
    目标函数定义

    Attributes:
        name: 目标函数名称
        weight: 权重（用于加权求和时的系数）
        minimize: True=最小化, False=最大化
    """
    name: str
    weight: float
    minimize: bool = True

    def evaluate(self, solution: np.ndarray, tasks: List[Dict], key: str) -> float:
        """
        评估解在特定目标上的值

        Args:
            solution: 决策向量，1表示在轨处理，0表示下传
            tasks: 任务列表
            key: 任务属性键名

        Returns:
            目标函数值
        """
        if len(solution) == 0 or len(tasks) == 0:
            return 0.0

        total = 0.0
        for i, (task, decision) in enumerate(zip(tasks, solution)):
            if decision == 1:
                total += task.get(key, 0.0)

        return total


class ParetoOptimizer:
    """
    多目标帕累托优化器

    使用NSGA-II风格的多目标遗传算法，求解在轨处理决策的帕累托最优解集。

    决策变量：每个成像任务是否在轨处理（0/1）
    目标函数：5个最小化目标（能耗、时间、存储、带宽、热负载）

    Attributes:
        objectives: 目标函数列表
        archive: 帕累托前沿存档
    """

    def __init__(self, objectives: List[ObjectiveFunction]):
        """
        初始化帕累托优化器

        Args:
            objectives: 目标函数列表
        """
        self.objectives = objectives
        self.archive: List[Dict] = []

    def optimize(
        self,
        imaging_tasks: List[Dict],
        satellites: List[Dict],
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1
    ) -> List[Dict]:
        """
        执行多目标优化

        使用NSGA-II算法求解帕累托最优解集。

        Args:
            imaging_tasks: 成像任务列表
            satellites: 卫星列表
            population_size: 种群大小
            generations: 迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率

        Returns:
            帕累托前沿解集，每个解包含:
            - decision_vector: 决策向量
            - objectives: 目标函数值
            - decoded: 解码后的任务分配
        """
        n_tasks = len(imaging_tasks)

        # 边界条件处理
        if n_tasks == 0:
            return []

        if len(satellites) == 0:
            raise ValueError("At least one satellite is required")

        # 验证所有任务都有对应的卫星
        sat_ids = {s['id'] for s in satellites}
        for task in imaging_tasks:
            if task['satellite_id'] not in sat_ids:
                raise ValueError(f"Satellite {task['satellite_id']} not found for task {task['id']}")

        # 初始化种群（随机生成二进制解）
        np.random.seed(42)  # 保证可重复性
        population = np.random.randint(0, 2, size=(population_size, n_tasks))

        for gen in range(generations):
            # 评估种群
            fitness = self._evaluate_population(population, imaging_tasks, satellites)

            # 非支配排序
            fronts = self._non_dominated_sort(fitness)

            # 遗传操作（选择、交叉、变异）
            offspring = self._genetic_operators(
                population, fronts, fitness,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate
            )

            # 合并父代和子代
            combined = np.vstack([population, offspring])
            combined_fitness = self._evaluate_population(combined, imaging_tasks, satellites)
            combined_fronts = self._non_dominated_sort(combined_fitness)

            # 精英保留选择下一代
            population = self._select_next_generation(
                combined, combined_fitness, combined_fronts, population_size
            )

        # 返回最终帕累托前沿
        final_fitness = self._evaluate_population(population, imaging_tasks, satellites)
        final_fronts = self._non_dominated_sort(final_fitness)

        pareto_solutions = []
        for idx in final_fronts[0]:  # 第0层是帕累托前沿
            solution_data = {
                'decision_vector': population[idx].tolist(),
                'objectives': final_fitness[idx].tolist(),
                'decoded': self._decode_solution(population[idx], imaging_tasks)
            }
            pareto_solutions.append(solution_data)

            # 更新存档
            self.archive.append(solution_data)

        return pareto_solutions

    def _evaluate_population(
        self,
        population: np.ndarray,
        tasks: List[Dict],
        satellites: List[Dict]
    ) -> np.ndarray:
        """
        评估种群的目标函数值

        Args:
            population: 种群矩阵 (population_size, n_tasks)
            tasks: 任务列表
            satellites: 卫星列表

        Returns:
            适应度矩阵 (population_size, n_objectives)
        """
        fitness = np.zeros((len(population), len(self.objectives)))

        for i, solution in enumerate(population):
            fitness[i] = self._evaluate_solution(solution, tasks, satellites)

        return fitness

    def _evaluate_solution(
        self,
        solution: np.ndarray,
        tasks: List[Dict],
        satellites: List[Dict]
    ) -> np.ndarray:
        """
        评估单个解

        solution[i] = 1 表示第i个任务在轨处理
        solution[i] = 0 表示第i个任务原始下传

        默认目标函数（5个，全部最小化）：
        1. total_energy - 总能耗 (Wh)
        2. total_time - 总时间 (seconds)
        3. total_storage - 存储占用 (GB)
        4. total_bandwidth - 带宽占用 (KB)
        5. thermal_load - 热负载 (°C)

        如果初始化时指定了不同数量的目标函数，则只返回对应数量的值。

        Args:
            solution: 决策向量
            tasks: 任务列表
            satellites: 卫星列表

        Returns:
            目标函数值向量，长度等于初始化时的目标函数数量
        """
        total_energy = 0.0
        total_time = 0.0
        total_storage = 0.0
        total_bandwidth = 0.0
        thermal_load = 0.0

        # 创建卫星查找字典
        sat_dict = {s['id']: s for s in satellites}

        for i, (task, decision) in enumerate(zip(tasks, solution)):
            sat = sat_dict.get(task['satellite_id'])

            if sat is None:
                continue

            if decision == 1 and sat.get('has_ai_accelerator', False):
                # 在轨处理代价
                total_energy += sat.get('ai_power_wh', 0.0)
                total_time += sat.get('ai_processing_time', 0.0)
                total_bandwidth += task.get('compressed_size_kb', 0.0)
                total_storage += task.get('data_size_gb', 0.0)  # 临时存储
                thermal_load += sat.get('ai_thermal_load', 0.0)
            else:
                # 原始下传代价
                total_energy += sat.get('downlink_power_wh', 0.0)
                total_time += task.get('downlink_time', 0.0)
                total_bandwidth += task.get('data_size_gb', 0.0) * 1e6  # GB转KB
                total_storage += task.get('data_size_gb', 0.0)
                thermal_load += 0.0  # 下传热负载较低

        # 根据目标函数数量返回对应长度的向量
        all_objectives = [
            total_energy,
            total_time,
            total_storage,
            total_bandwidth,
            thermal_load
        ]
        n_objectives = len(self.objectives) if self.objectives else 5

        # 如果目标函数数量超过5个，循环使用已有值
        result = []
        for i in range(n_objectives):
            result.append(all_objectives[i % len(all_objectives)])

        return np.array(result)

    def _non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        """
        非支配排序（NSGA-II核心）

        将种群按支配关系分层：
        - 第0层：帕累托前沿（不被任何解支配）
        - 第1层：被第0层支配的解
        - 第2层：被第0层和第1层支配的解
        - ...

        Args:
            fitness: 适应度矩阵 (population_size, n_objectives)

        Returns:
            分层列表，每个元素是一层中的解索引列表
        """
        n = len(fitness)
        if n == 0:
            return []

        domination_count = [0] * n  # 支配该解的解数量
        dominated_solutions = [[] for _ in range(n)]  # 该解支配的解列表
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # 去除空前沿

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        判断解a是否支配解b

        a支配b当且仅当：
        1. a在所有目标上都不差于b（a <= b，最小化问题）
        2. a在至少一个目标上严格优于b（a < b）

        Args:
            a: 解a的目标向量
            b: 解b的目标向量

        Returns:
            True如果a支配b，否则False
        """
        return bool(np.all(a <= b) and np.any(a < b))

    def _genetic_operators(
        self,
        population: np.ndarray,
        fronts: List[List[int]],
        fitness: np.ndarray,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1
    ) -> np.ndarray:
        """
        遗传操作：选择、交叉、变异

        Args:
            population: 父代种群
            fronts: 非支配排序分层
            fitness: 适应度矩阵
            crossover_rate: 交叉概率
            mutation_rate: 变异概率

        Returns:
            子代种群
        """
        offspring = []
        pop_size = len(population)

        while len(offspring) < pop_size:
            # 锦标赛选择两个父代
            parent1_idx = self._tournament_selection(fronts, fitness)
            parent2_idx = self._tournament_selection(fronts, fitness)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # 交叉
            if np.random.random() < crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 变异
            child1 = self._mutate(child1, mutation_rate)
            child2 = self._mutate(child2, mutation_rate)

            offspring.append(child1)
            if len(offspring) < pop_size:
                offspring.append(child2)

        return np.array(offspring)

    def _tournament_selection(
        self,
        fronts: List[List[int]],
        fitness: np.ndarray,
        tournament_size: int = 2
    ) -> int:
        """
        锦标赛选择

        从种群中随机选择tournament_size个个体，返回最优者。
        优先选择非支配排序层数低的个体，同层时选择拥挤距离大的个体。

        Args:
            fronts: 非支配排序分层
            fitness: 适应度矩阵
            tournament_size: 锦标赛大小

        Returns:
            选中个体的索引
        """
        # 创建层数字典
        rank = {}
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i

        # 随机选择tournament_size个个体
        candidates = np.random.choice(
            list(rank.keys()),
            size=min(tournament_size, len(rank)),
            replace=False
        )

        # 选择最优者（层数最低）
        best = candidates[0]
        for candidate in candidates[1:]:
            if rank[candidate] < rank[best]:
                best = candidate

        return best

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉

        随机选择交叉点，交换两个父代在该点后的基因。

        Args:
            parent1: 父代1
            parent2: 父代2

        Returns:
            两个子代
        """
        n = len(parent1)
        if n <= 1:
            return parent1.copy(), parent2.copy()

        # 随机选择交叉点
        crossover_point = np.random.randint(1, n)

        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def _mutate(
        self,
        individual: np.ndarray,
        mutation_rate: float = 0.1
    ) -> np.ndarray:
        """
        位变异

        以mutation_rate的概率翻转每个基因（0变1，1变0）。

        Args:
            individual: 个体
            mutation_rate: 变异概率

        Returns:
            变异后的个体
        """
        if mutation_rate <= 0:
            return individual.copy()

        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # 翻转

        return mutated

    def _select_next_generation(
        self,
        combined: np.ndarray,
        fitness: np.ndarray,
        fronts: List[List[int]],
        population_size: int
    ) -> np.ndarray:
        """
        选择下一代（精英保留）

        按非支配排序层依次选择，同层按拥挤距离排序。

        Args:
            combined: 合并的父代和子代
            fitness: 适应度矩阵
            fronts: 非支配排序分层
            population_size: 目标种群大小

        Returns:
            选中的下一代种群
        """
        selected = []
        remaining = population_size

        for front in fronts:
            if remaining <= 0:
                break

            if len(front) <= remaining:
                # 该层全部选中
                selected.extend(front)
                remaining -= len(front)
            else:
                # 该层只能选部分，按拥挤距离排序
                distances = self._calculate_crowding_distance(fitness, front)
                # 按拥挤距离降序排序
                sorted_indices = sorted(
                    range(len(front)),
                    key=lambda i: distances[i],
                    reverse=True
                )
                selected.extend([front[i] for i in sorted_indices[:remaining]])
                remaining = 0

        return combined[selected]

    def _calculate_crowding_distance(
        self,
        fitness: np.ndarray,
        front: List[int]
    ) -> np.ndarray:
        """
        计算拥挤距离

        拥挤距离反映了解在目标空间中的分布密度。
        边界点的拥挤距离为无穷大，其他点根据相邻点的距离计算。

        Args:
            fitness: 适应度矩阵
            front: 当前层的解索引列表

        Returns:
            拥挤距离数组
        """
        n_front = len(front)
        if n_front <= 2:
            return np.full(n_front, float('inf'))

        n_objectives = fitness.shape[1]
        distances = np.zeros(n_front)

        # 对每个目标计算拥挤距离
        for obj in range(n_objectives):
            # 按当前目标排序
            sorted_indices = sorted(range(n_front), key=lambda i: fitness[front[i], obj])

            # 边界点设为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # 计算中间点的拥挤距离
            obj_range = fitness[front[sorted_indices[-1]], obj] - fitness[front[sorted_indices[0]], obj]
            if obj_range > 0:
                for i in range(1, n_front - 1):
                    dist = (fitness[front[sorted_indices[i + 1]], obj] -
                            fitness[front[sorted_indices[i - 1]], obj])
                    distances[sorted_indices[i]] += dist / obj_range

        return distances

    def _decode_solution(
        self,
        solution: np.ndarray,
        tasks: List[Dict]
    ) -> Dict:
        """
        解码解

        将二进制决策向量解码为任务分配方案。

        Args:
            solution: 决策向量
            tasks: 任务列表

        Returns:
            解码结果，包含：
            - onboard_tasks: 在轨处理任务列表
            - downlink_tasks: 下传任务列表
        """
        onboard_tasks = []
        downlink_tasks = []

        for i, (task, decision) in enumerate(zip(tasks, solution)):
            if decision == 1:
                onboard_tasks.append(task)
            else:
                downlink_tasks.append(task)

        return {
            'onboard_tasks': onboard_tasks,
            'downlink_tasks': downlink_tasks
        }
