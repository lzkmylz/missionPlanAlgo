"""
蚁群优化调度器 (ACO Scheduler)

基于蚁群优化算法原理实现的卫星任务规划调度器。
设计文档第4章/第7章 - ACO算法实现
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason


@dataclass
class Ant:
    """蚂蚁个体"""
    path: List[int]  # 每个任务分配的卫星索引
    fitness: float = 0.0


class ACOScheduler(BaseScheduler):
    """
    蚁群优化调度器

    算法特点：
    - 模拟蚂蚁觅食行为，通过信息素引导搜索
    - 正反馈机制：好的路径积累更多信息素
    - 分布式计算：多只蚂蚁并行搜索

    关键参数：
    - num_ants: 蚂蚁数量
    - alpha: 信息素重要程度
    - beta: 启发信息重要程度
    - evaporation_rate: 信息素蒸发率
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化ACO调度器

        Args:
            config: 配置参数
                - num_ants: 蚂蚁数量（默认30）
                - max_iterations: 最大迭代次数（默认100）
                - alpha: 信息素重要程度因子（默认1.0）
                - beta: 启发信息重要程度因子（默认2.0）
                - evaporation_rate: 信息素蒸发率（默认0.1）
                - initial_pheromone: 初始信息素浓度（默认1.0）
                - random_seed: 随机种子（可选）
        """
        super().__init__("ACO", config)
        config = config or {}

        # 参数验证
        self.num_ants = self._validate_positive_int(
            config.get('num_ants', 30), 'num_ants'
        )
        self.max_iterations = self._validate_positive_int(
            config.get('max_iterations', 100), 'max_iterations'
        )
        self.alpha = self._validate_non_negative_float(
            config.get('alpha', 1.0), 'alpha'
        )
        self.beta = self._validate_non_negative_float(
            config.get('beta', 2.0), 'beta'
        )
        self.evaporation_rate = self._validate_probability(
            config.get('evaporation_rate', 0.1), 'evaporation_rate'
        )
        self.initial_pheromone = self._validate_positive_float(
            config.get('initial_pheromone', 1.0), 'initial_pheromone'
        )

        # 设置随机种子
        self.random_seed = config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # 运行时数据
        self.tasks: List[Any] = []
        self.satellites: List[Any] = []
        self.task_count = 0
        self.sat_count = 0
        self.pheromone_matrix: Optional[np.ndarray] = None
        self.heuristic_matrix: Optional[np.ndarray] = None

    def _validate_positive_int(self, value: int, name: str) -> int:
        """验证正整数参数"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value

    def _validate_positive_float(self, value: float, name: str) -> float:
        """验证正浮点数参数"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value}")
        return float(value)

    def _validate_non_negative_float(self, value: float, name: str) -> float:
        """验证非负浮点数参数"""
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return float(value)

    def _validate_probability(self, value: float, name: str) -> float:
        """验证概率参数（0-1之间）"""
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return float(value)

    def get_parameters(self) -> Dict[str, Any]:
        """返回算法可调参数"""
        return {
            'num_ants': self.num_ants,
            'max_iterations': self.max_iterations,
            'alpha': self.alpha,
            'beta': self.beta,
            'evaporation_rate': self.evaporation_rate,
            'initial_pheromone': self.initial_pheromone,
            'random_seed': self.random_seed,
        }

    def schedule(self) -> ScheduleResult:
        """
        执行蚁群优化调度

        Returns:
            ScheduleResult: 调度结果
        """
        self._start_timer()

        self._validate_initialization()

        # 准备数据 - 使用频次感知的任务列表
        self.tasks = self._create_frequency_aware_tasks()
        self.satellites = list(self.mission.satellites)
        self.task_count = len(self.tasks)
        self.sat_count = len(self.satellites)

        # 处理空场景
        if self.task_count == 0 or self.sat_count == 0:
            return self._build_empty_result()

        # 初始化信息素矩阵
        self._initialize_pheromone_matrix()

        # 初始化启发信息矩阵
        self._initialize_heuristic_matrix()

        # 记录最优解
        best_ant = None
        best_fitness = float('-inf')

        # 记录收敛曲线
        self._convergence_curve = []

        # 蚁群优化主循环
        for iteration in range(self.max_iterations):
            # 每只蚂蚁构建解
            ants = [self._construct_solution() for _ in range(self.num_ants)]

            # 评估所有蚂蚁
            for ant in ants:
                ant.fitness = self._evaluate(ant)

            # 更新最优解
            iteration_best = max(ants, key=lambda a: a.fitness)
            if iteration_best.fitness > best_fitness:
                best_fitness = iteration_best.fitness
                best_ant = Ant(path=iteration_best.path[:], fitness=iteration_best.fitness)

            # 记录收敛
            self._convergence_curve.append(best_fitness)
            self._iterations = iteration + 1

            # 更新信息素
            self._update_pheromone(ants)

        # 解码最优解
        if best_ant is None:
            best_ant = Ant(path=[0] * self.task_count, fitness=0.0)

        scheduled_tasks, unscheduled = self._decode_solution(best_ant)

        # 计算指标
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self.max_iterations,
            convergence_curve=self._convergence_curve
        )

    def _initialize_pheromone_matrix(self) -> None:
        """初始化信息素矩阵"""
        self.pheromone_matrix = np.ones((self.task_count, self.sat_count)) * self.initial_pheromone

    def _initialize_heuristic_matrix(self) -> None:
        """初始化启发信息矩阵"""
        # 启发信息：基于卫星能力的先验知识
        # 这里使用均匀启发信息
        self.heuristic_matrix = np.ones((self.task_count, self.sat_count))

        # 如果有窗口缓存，根据窗口质量设置启发信息
        if self.window_cache:
            for task_idx, task in enumerate(self.tasks):
                for sat_idx, sat in enumerate(self.satellites):
                    windows = self.window_cache.get_windows(sat.id, task.id)
                    if windows:
                        # 有可见窗口，设置较高的启发值
                        best_quality = max(w.quality_score for w in windows)
                        self.heuristic_matrix[task_idx, sat_idx] = 1.0 + best_quality

    def _construct_solution(self) -> Ant:
        """
        蚂蚁构建解

        使用轮盘赌选择策略，基于信息素和启发信息选择卫星
        """
        path = []

        for task_idx in range(self.task_count):
            # 计算选择概率
            probabilities = self._calculate_probabilities(task_idx)

            # 轮盘赌选择
            sat_idx = np.random.choice(self.sat_count, p=probabilities)
            path.append(sat_idx)

        return Ant(path=path)

    def _calculate_probabilities(self, task_idx: int) -> np.ndarray:
        """
        计算任务选择各卫星的概率

        基于信息素和启发信息的加权组合
        """
        pheromone = self.pheromone_matrix[task_idx] ** self.alpha
        heuristic = self.heuristic_matrix[task_idx] ** self.beta

        # 计算综合概率
        combined = pheromone * heuristic
        total = combined.sum()

        if total == 0:
            # 如果总和为0，使用均匀分布
            return np.ones(self.sat_count) / self.sat_count

        return combined / total

    def _update_pheromone(self, ants: List[Ant]) -> None:
        """
        更新信息素

        1. 所有路径信息素蒸发
        2. 蚂蚁在其路径上沉积信息素
        """
        # 信息素蒸发
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        # 信息素沉积
        for ant in ants:
            if ant.fitness > 0:
                deposit = ant.fitness / 100.0  # 归一化沉积量
                for task_idx, sat_idx in enumerate(ant.path):
                    self.pheromone_matrix[task_idx, sat_idx] += deposit

    def _evaluate(self, ant: Ant) -> float:
        """
        评估蚂蚁解的适应度

        适应度函数：
        - 基础：完成的任务数量 × 10
        - 奖励：窗口质量、资源均衡
        - 惩罚：时间冲突
        - 频次满足度奖励
        """
        from ..frequency_utils import ObservationTask

        score = 0.0
        scheduled_count = 0
        target_obs_count: Dict[str, int] = {}  # 记录每个目标的实际观测次数
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }

        for task_idx, sat_idx in enumerate(ant.path):
            if task_idx >= len(self.tasks):
                continue

            task = self.tasks[task_idx]
            if sat_idx >= self.sat_count:
                continue

            sat = self.satellites[sat_idx]

            # 检查是否有可见窗口 (ObservationTask使用target_id)
            target_id = task.target_id if isinstance(task, ObservationTask) else task.id
            if self.window_cache:
                windows = self.window_cache.get_windows(sat.id, target_id)
            else:
                windows = []

            if not windows:
                continue

            # 检查时间冲突
            feasible_window = None
            for window in windows:
                if self._is_time_feasible(sat_idx, window.start_time, window.end_time, sat_task_times):
                    feasible_window = window
                    break

            if feasible_window:
                scheduled_count += 1
                score += 10.0  # 基础完成奖励

                # 窗口质量奖励
                score += feasible_window.quality_score * 2.0

                # 记录时间
                sat_task_times[sat_idx].append(
                    (feasible_window.start_time, feasible_window.end_time)
                )

                # 记录目标观测次数
                target_id = task.target_id if isinstance(task, ObservationTask) else task.id
                target_obs_count[target_id] = target_obs_count.get(target_id, 0) + 1

        # 资源均衡奖励
        if scheduled_count > 0:
            task_counts = [len(tasks) for tasks in sat_task_times.values()]
            avg_tasks = sum(task_counts) / len(task_counts)
            variance = sum((c - avg_tasks) ** 2 for c in task_counts) / len(task_counts)
            balance_reward = max(0, 10 - variance)
            score += balance_reward

        # 添加频次满足度奖励
        score = self._calculate_frequency_fitness(target_obs_count, score)

        return score

    def _is_time_feasible(
        self,
        sat_idx: int,
        start: datetime,
        end: datetime,
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]]
    ) -> bool:
        """检查时间是否可行（无冲突）"""
        for existing_start, existing_end in sat_task_times.get(sat_idx, []):
            if not (end <= existing_start or start >= existing_end):
                return False
        return True

    def _decode_solution(
        self,
        ant: Ant
    ) -> Tuple[List[ScheduledTask], Dict[str, Any]]:
        """将蚂蚁解解码为调度方案"""
        from ..frequency_utils import ObservationTask

        scheduled_tasks = []
        unscheduled = {}

        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }

        for task_idx, sat_idx in enumerate(ant.path):
            if task_idx >= len(self.tasks):
                continue

            task = self.tasks[task_idx]

            # 支持ObservationTask和原始Target
            task_id = task.task_id if isinstance(task, ObservationTask) else task.id
            target_id = task.target_id if isinstance(task, ObservationTask) else task.id

            if sat_idx >= self.sat_count:
                self._record_failure(
                    task_id=task_id,
                    reason=TaskFailureReason.UNKNOWN,
                    detail=f"Invalid satellite index: {sat_idx}"
                )
                unscheduled[task_id] = self._failure_log[-1]
                continue

            sat = self.satellites[sat_idx]

            # 获取可见窗口 (ObservationTask使用target_id)
            if self.window_cache:
                windows = self.window_cache.get_windows(sat.id, target_id)
            else:
                windows = []

            if not windows:
                self._record_failure(
                    task_id=task_id,
                    reason=TaskFailureReason.NO_VISIBLE_WINDOW,
                    detail=f"No visibility window for satellite {sat.id}"
                )
                unscheduled[task_id] = self._failure_log[-1]
                continue

            # 查找可行窗口
            feasible_window = None
            for window in windows:
                if self._is_time_feasible(sat_idx, window.start_time, window.end_time, sat_task_times):
                    feasible_window = window
                    break

            if feasible_window:
                scheduled_task = ScheduledTask(
                    task_id=task_id,
                    satellite_id=sat.id,
                    target_id=target_id,
                    imaging_start=feasible_window.start_time,
                    imaging_end=feasible_window.end_time,
                    imaging_mode="push_broom"
                )
                scheduled_tasks.append(scheduled_task)
                sat_task_times[sat_idx].append(
                    (feasible_window.start_time, feasible_window.end_time)
                )
            else:
                self._record_failure(
                    task_id=task_id,
                    reason=TaskFailureReason.TIME_CONFLICT,
                    detail=f"No feasible time window for satellite {sat.id}"
                )
                unscheduled[task_id] = self._failure_log[-1]

        return scheduled_tasks, unscheduled

    def _calculate_makespan(self, scheduled_tasks: List[ScheduledTask]) -> float:
        """计算总完成时间"""
        if not scheduled_tasks:
            return 0.0

        last_end = max(t.imaging_end for t in scheduled_tasks)
        return (last_end - self.mission.start_time).total_seconds()

    def _build_empty_result(self) -> ScheduleResult:
        """构建空结果"""
        return ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=self._stop_timer(),
            iterations=0,
            convergence_curve=[]
        )
