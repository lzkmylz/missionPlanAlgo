"""
禁忌搜索调度器 (Tabu Scheduler)

实现Tabu搜索算法用于卫星任务规划调度。
包含禁忌表、邻域搜索、aspiration criteria等特性。
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason


@dataclass(order=True)
class TabuSolution:
    """禁忌搜索解"""
    fitness: float = 0.0
    unscheduled_count: int = 0
    assignment: List[int] = field(default_factory=list, compare=False)

    def __post_init__(self):
        """确保assignment是列表"""
        if self.assignment is None:
            self.assignment = []
        elif not isinstance(self.assignment, list):
            self.assignment = list(self.assignment)


class TabuScheduler(BaseScheduler):
    """
    禁忌搜索调度器

    算法特性：
    1. 禁忌表 (Tabu List): 防止循环搜索，记录近期访问的解
    2. 邻域搜索 (Neighborhood Search): 在当前解的邻域内寻找改进
    3. Aspiration Criteria: 允许突破禁忌，如果解足够好
    4. 动态禁忌期限: 自适应调整禁忌期限

    参数：
    - tabu_tenure: 禁忌期限，解在禁忌表中保留的迭代次数
    - max_iterations: 最大迭代次数
    - neighborhood_size: 邻域大小，每次迭代生成的邻居数量
    - aspiration_threshold: 突破阈值，相对改进超过此值可突破禁忌
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Tabu调度器

        Args:
            config: 配置参数
                - tabu_tenure: 禁忌期限（默认10）
                - max_iterations: 最大迭代次数（默认100）
                - neighborhood_size: 邻域大小（默认20）
                - aspiration_threshold: 突破阈值（默认0.05）
                - random_seed: 随机种子（可选）
        """
        super().__init__("Tabu", config)
        config = config or {}

        # 参数验证和设置
        self.tabu_tenure = self._validate_positive_int(
            config.get('tabu_tenure', 10), 'tabu_tenure'
        )
        self.max_iterations = self._validate_positive_int(
            config.get('max_iterations', 100), 'max_iterations'
        )
        self.neighborhood_size = self._validate_positive_int(
            config.get('neighborhood_size', 20), 'neighborhood_size'
        )
        self.aspiration_threshold = self._validate_probability(
            config.get('aspiration_threshold', 0.05), 'aspiration_threshold'
        )

        # 设置随机种子
        self.random_seed = config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # 禁忌表 - 使用deque实现自动淘汰
        self.tabu_list: deque = deque(maxlen=self.tabu_tenure)

        # 运行时数据
        self.tasks: List[Any] = []
        self.satellites: List[Any] = []
        self.task_count = 0
        self.sat_count = 0
        self.best_fitness = 0.0

    def _validate_positive_int(self, value: int, name: str) -> int:
        """验证正整数参数"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value

    def _validate_probability(self, value: float, name: str) -> float:
        """验证概率参数（0-1之间）"""
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return float(value)

    def get_parameters(self) -> Dict[str, Any]:
        """返回算法可调参数"""
        return {
            'tabu_tenure': self.tabu_tenure,
            'max_iterations': self.max_iterations,
            'neighborhood_size': self.neighborhood_size,
            'aspiration_threshold': self.aspiration_threshold,
            'random_seed': self.random_seed,
        }

    def schedule(self) -> ScheduleResult:
        """
        执行禁忌搜索调度

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
        if self.task_count == 0:
            return self._build_empty_result()

        # 处理无卫星场景
        if self.sat_count == 0:
            return self._build_empty_result()

        # 初始化禁忌表
        self.tabu_list.clear()

        # 生成初始解
        current_solution = self._generate_initial_solution()
        current_solution.fitness = self._evaluate_solution(current_solution)

        # 记录最优解
        best_solution = TabuSolution(
            assignment=current_solution.assignment[:],
            fitness=current_solution.fitness,
            unscheduled_count=current_solution.unscheduled_count
        )
        self.best_fitness = best_solution.fitness

        # 收敛曲线
        self._convergence_curve = [best_solution.fitness]

        # 主搜索循环
        for iteration in range(self.max_iterations):
            self._iterations = iteration + 1

            # 生成邻域解
            neighbors = self._generate_neighbors(current_solution, self.sat_count)

            # 评估邻域解
            for neighbor in neighbors:
                neighbor.fitness = self._evaluate_solution(neighbor)

            # 选择最佳邻域解（考虑禁忌和aspiration）
            next_solution = self._select_best_neighbor(
                neighbors, best_solution.fitness
            )

            if next_solution is None:
                # 无有效邻居，停止搜索
                break

            # 更新当前解
            current_solution = next_solution

            # 更新禁忌表
            self._update_tabu_list(current_solution)

            # 更新最优解
            if current_solution.fitness > best_solution.fitness:
                best_solution = TabuSolution(
                    assignment=current_solution.assignment[:],
                    fitness=current_solution.fitness,
                    unscheduled_count=current_solution.unscheduled_count
                )
                self.best_fitness = best_solution.fitness

            # 记录收敛
            self._convergence_curve.append(best_solution.fitness)

        # 解码最优解为调度结果
        scheduled_tasks, unscheduled = self._decode_solution(best_solution)

        # 计算指标
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self._iterations,
            convergence_curve=self._convergence_curve
        )

    def _generate_initial_solution(self) -> TabuSolution:
        """生成初始解 - 随机分配"""
        assignment = [
            random.randint(0, self.sat_count - 1)
            for _ in range(self.task_count)
        ]
        return TabuSolution(assignment=assignment, fitness=0.0, unscheduled_count=0)

    def _generate_neighbors(
        self,
        solution: TabuSolution,
        num_satellites: int
    ) -> List[TabuSolution]:
        """
        生成邻域解

        邻域结构：改变一个任务的卫星分配
        """
        neighbors = []
        assignment = solution.assignment

        # 生成指定数量的邻居
        attempts = 0
        max_attempts = self.neighborhood_size * 3

        while len(neighbors) < self.neighborhood_size and attempts < max_attempts:
            attempts += 1

            # 随机选择一个任务
            task_idx = random.randint(0, len(assignment) - 1)

            # 随机选择一个不同的卫星
            current_sat = assignment[task_idx]
            new_sat = random.randint(0, num_satellites - 1)

            if new_sat == current_sat:
                continue

            # 创建新解
            new_assignment = assignment[:]
            new_assignment[task_idx] = new_sat

            neighbor = TabuSolution(
                assignment=new_assignment,
                fitness=0.0,
                unscheduled_count=0
            )

            neighbors.append(neighbor)

        return neighbors

    def _evaluate_solution(self, solution: TabuSolution) -> float:
        """
        评估解的适应度

        适应度函数：
        - 基础：成功调度的任务数量 × 10
        - 奖励：窗口质量、资源均衡
        - 惩罚：约束违反
        - 频次满足度奖励
        """
        from ..frequency_utils import ObservationTask

        score = 0.0
        scheduled_count = 0
        unscheduled_count = 0
        target_obs_count: Dict[str, int] = {}  # 记录每个目标的实际观测次数
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }

        for task_idx, sat_idx in enumerate(solution.assignment):
            if task_idx >= len(self.tasks):
                continue

            task = self.tasks[task_idx]
            if sat_idx >= self.sat_count:
                unscheduled_count += 1
                continue

            sat = self.satellites[sat_idx]

            # 检查是否有可见窗口 (ObservationTask使用target_id)
            target_id = task.target_id if isinstance(task, ObservationTask) else task.id
            if self.window_cache:
                windows = self.window_cache.get_windows(sat.id, target_id)
            else:
                windows = []

            if not windows:
                unscheduled_count += 1
                continue

            # 检查时间冲突
            feasible_window = None
            for window in windows:
                start_time = window.start_time
                end_time = window.end_time

                if self._is_time_feasible(sat_idx, start_time, end_time, sat_task_times):
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
            else:
                unscheduled_count += 1

        # 资源均衡奖励
        if scheduled_count > 0:
            task_counts = [len(tasks) for tasks in sat_task_times.values()]
            avg_tasks = sum(task_counts) / len(task_counts)
            variance = sum((c - avg_tasks) ** 2 for c in task_counts) / len(task_counts)
            balance_reward = max(0, 10 - variance)  # 越均衡奖励越高
            score += balance_reward

        # 惩罚未调度的任务
        score -= unscheduled_count * 0.5

        # 添加频次满足度奖励
        score = self._calculate_frequency_fitness(target_obs_count, score)

        solution.unscheduled_count = unscheduled_count
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

    def _select_best_neighbor(
        self,
        neighbors: List[TabuSolution],
        best_fitness: float
    ) -> Optional[TabuSolution]:
        """
        选择最佳邻域解，考虑禁忌和aspiration criteria

        Args:
            neighbors: 邻域解列表
            best_fitness: 当前最优适应度

        Returns:
            选中的解，或None
        """
        if not neighbors:
            return None

        # 按适应度排序
        sorted_neighbors = sorted(neighbors, key=lambda x: x.fitness, reverse=True)

        for neighbor in sorted_neighbors:
            # 检查是否在禁忌表中
            is_tabu = self._is_tabu(neighbor)

            if not is_tabu:
                # 非禁忌解，直接选择
                return neighbor

            # 检查aspiration criteria
            # 如果解比当前最优解好超过阈值，允许突破禁忌
            improvement = (neighbor.fitness - best_fitness) / max(abs(best_fitness), 1.0)
            if improvement > self.aspiration_threshold:
                return neighbor

        # 所有邻居都在禁忌中且不满足aspiration，选择最好的一个
        return sorted_neighbors[0] if sorted_neighbors else None

    def _is_tabu(self, solution: TabuSolution) -> bool:
        """检查解是否在禁忌表中"""
        # 使用assignment作为禁忌标识
        solution_key = tuple(solution.assignment)
        return solution_key in self.tabu_list

    def _update_tabu_list(self, solution: TabuSolution):
        """更新禁忌表"""
        solution_key = tuple(solution.assignment)

        # 添加到禁忌表
        self.tabu_list.append(solution_key)

        # 确保不超过禁忌期限
        while len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.popleft()

    def _decode_solution(
        self,
        solution: TabuSolution
    ) -> Tuple[List[ScheduledTask], Dict[str, Any]]:
        """将解解码为调度方案"""
        from ..frequency_utils import ObservationTask

        scheduled_tasks = []
        unscheduled = {}

        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }

        for task_idx, sat_idx in enumerate(solution.assignment):
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
