"""
遗传算法调度器 (GA Scheduler)

基于遗传算法原理实现的卫星任务规划调度器。
使用整数编码，每个基因代表一个任务的卫星分配。
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile
from core.dynamics.slew_calculator import SlewCalculator
from ..constraints import SlewConstraintChecker, SlewFeasibilityResult
from .constraints_utils import MetaheuristicConstraintChecker


@dataclass
class GAIndividual:
    """遗传算法个体（染色体）"""
    chromosome: List[int]  # 每个任务分配的卫星索引
    fitness: float = 0.0


class GAScheduler(BaseScheduler):
    """
    遗传算法调度器

    编码方案：
    - 染色体长度 = 任务数量
    - 每个基因 = 卫星索引（0 to n_satellites-1）
    - 特殊值 -1 = 任务未分配

    适应度函数：
    - 基础：成功调度的任务数量
    - 奖励：资源均衡、时间效率
    - 惩罚：约束违反
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化GA调度器

        Args:
            config: 配置参数
                - population_size: 种群大小（默认100）
                - generations: 迭代代数（默认200）
                - crossover_rate: 交叉概率（默认0.8）
                - mutation_rate: 变异概率（默认0.2）
                - elitism: 精英保留数量（默认5）
                - tournament_size: 锦标赛选择大小（默认3）
                - random_seed: 随机种子（可选）
                - consider_power: 是否考虑电量约束（默认True）
                - consider_storage: 是否考虑存储约束（默认True）
        """
        super().__init__("GA", config)
        config = config or {}

        # 参数验证
        self.population_size = self._validate_positive_int(
            config.get('population_size', 100), 'population_size'
        )
        self.generations = self._validate_positive_int(
            config.get('generations', 200), 'generations'
        )
        self.crossover_rate = self._validate_probability(
            config.get('crossover_rate', 0.8), 'crossover_rate'
        )
        self.mutation_rate = self._validate_probability(
            config.get('mutation_rate', 0.2), 'mutation_rate'
        )
        self.elitism = config.get('elitism', 5)
        self.tournament_size = config.get('tournament_size', 3)

        # 资源约束配置
        self.consider_power = config.get('consider_power', True)
        self.consider_storage = config.get('consider_storage', True)

        # 设置随机种子
        self.random_seed = config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # 运行时数据
        self.tasks: List[Any] = []
        self.satellites: List[Any] = []
        self.task_count = 0
        self.sat_count = 0

        # 约束检查器（延迟初始化）
        self._constraint_checker: Optional[MetaheuristicConstraintChecker] = None

        # 姿态角计算配置
        self._enable_attitude_calculation = config.get('enable_attitude_calculation', True)
        self._use_simplified_slew = config.get('use_simplified_slew', False)

        # 初始化成像时间计算器和功率配置文件（用于向后兼容）
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=config.get('min_imaging_duration'),
            max_duration=config.get('max_imaging_duration'),
            default_duration=config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(config.get('power_coefficients'))

        # Slew calculators per satellite (用于向后兼容)
        self._slew_calculators: Dict[str, SlewCalculator] = {}
        self._last_task_target: Dict[str, Any] = {}

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
            'population_size': self.population_size,
            'generations': self.generations,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'elitism': self.elitism,
            'tournament_size': self.tournament_size,
            'random_seed': self.random_seed,
            'consider_power': self.consider_power,
            'consider_storage': self.consider_storage,
            'enable_attitude_calculation': self._enable_attitude_calculation,
            'use_simplified_slew': self._use_simplified_slew,
        }

    def schedule(self) -> ScheduleResult:
        """
        执行遗传算法调度

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

        # 初始化约束检查器
        self._constraint_checker = MetaheuristicConstraintChecker(
            self.mission,
            config={
                'consider_power': self.consider_power,
                'consider_storage': self.consider_storage,
                'use_simplified_slew': self._use_simplified_slew,
                'enable_attitude_calculation': self._enable_attitude_calculation,
            }
        )
        self._constraint_checker.initialize()

        # 初始化种群
        population = self._initialize_population()

        # 评估初始种群
        for individual in population:
            individual.fitness = self._evaluate(individual)

        # 记录收敛曲线
        best_fitness = max(ind.fitness for ind in population)
        self._convergence_curve = [best_fitness]

        # 进化循环
        for generation in range(self.generations):
            # 选择
            selected = self._selection(population)

            # 交叉
            offspring = self._crossover(selected)

            # 变异
            offspring = self._mutation(offspring)

            # 评估子代
            for individual in offspring:
                individual.fitness = self._evaluate(individual)

            # 精英保留
            if self.elitism > 0:
                population = self._elitism(population, offspring)
            else:
                population = offspring

            # 记录最优
            current_best = max(ind.fitness for ind in population)
            best_fitness = max(best_fitness, current_best)
            self._convergence_curve.append(best_fitness)
            self._iterations = generation + 1

        # 获取最优解
        best_individual = max(population, key=lambda ind: ind.fitness)

        # 解码为调度方案
        scheduled_tasks, unscheduled = self._decode_solution(best_individual)

        # 计算指标
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self.generations,
            convergence_curve=self._convergence_curve
        )

    def _initialize_population(self) -> List[GAIndividual]:
        """初始化种群"""
        population = []

        for _ in range(self.population_size):
            # 随机分配每个任务给一个卫星
            chromosome = [
                random.randint(0, self.sat_count - 1)
                for _ in range(self.task_count)
            ]
            population.append(GAIndividual(chromosome=chromosome))

        return population

    def _evaluate(self, individual: GAIndividual) -> float:
        """
        评估个体适应度

        适应度函数：
        - 基础：完成的任务数量 × 10
        - 奖励：窗口质量、资源均衡
        - 惩罚：时间冲突、资源约束违反
        - 频次满足度奖励
        - 机动可行性检查
        """
        eval_state = self._initialize_evaluation_state()

        for task_idx, sat_idx in enumerate(individual.chromosome):
            self._process_task_assignment(task_idx, sat_idx, eval_state)

        # 计算最终得分
        return self._compute_final_fitness(eval_state)

    def _initialize_evaluation_state(self) -> Dict[str, Any]:
        """初始化评估状态跟踪器"""
        return {
            'score': 0.0,
            'scheduled_count': 0,
            'target_obs_count': {},
            'sat_task_times': {i: [] for i in range(self.sat_count)},
            'sat_resources': {
                i: {
                    'power': sat.capabilities.power_capacity if hasattr(sat.capabilities, 'power_capacity') else 2800.0,
                    'storage': 0.0
                }
                for i, sat in enumerate(self.satellites)
            },
            'sat_last_target': {},
            'sat_last_end_time': {}
        }

    def _process_task_assignment(
        self, task_idx: int, sat_idx: int, state: Dict[str, Any]
    ) -> None:
        """处理单个任务分配"""
        if task_idx >= len(self.tasks) or sat_idx >= self.sat_count:
            return

        task = self.tasks[task_idx]
        sat = self.satellites[sat_idx]

        feasible_window = self._find_feasible_window(sat_idx, sat, task, state)
        if not feasible_window:
            return

        self._try_schedule_task(sat_idx, sat, task, feasible_window, state)

    def _find_feasible_window(
        self, sat_idx: int, sat: Any, task: Any, state: Dict[str, Any]
    ) -> Optional[Any]:
        """查找可行的时间窗口"""
        if not self.window_cache:
            return None

        windows = self.window_cache.get_windows(sat.id, task.target_id)
        if not windows:
            return None

        sat_task_times = state['sat_task_times']
        sat_resources = state['sat_resources']

        for window in windows:
            if self._is_time_feasible(sat_idx, window.start_time, window.end_time, sat_task_times):
                if self._check_resource_constraints(sat_idx, sat, task, sat_resources):
                    return window
        return None

    def _try_schedule_task(
        self, sat_idx: int, sat: Any, task: Any, window: Any, state: Dict[str, Any]
    ) -> None:
        """尝试调度任务到指定窗口"""
        imaging_mode = self._select_imaging_mode(sat)
        imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)

        actual_start, actual_end = self._calculate_task_timing(
            sat_idx, sat, task, window, imaging_duration, state
        )

        if not actual_start or not self._is_saa_feasible(sat.id, window):
            return

        if actual_end > window.end_time:
            return

        # 更新调度状态
        self._update_scheduled_task_state(
            sat_idx, sat, task, window, actual_start, actual_end, imaging_duration, state
        )

    def _calculate_task_timing(
        self, sat_idx: int, sat: Any, task: Any, window: Any,
        imaging_duration: float, state: Dict[str, Any]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """计算任务的实际开始和结束时间"""
        sat_last_target = state['sat_last_target']
        sat_last_end_time = state['sat_last_end_time']

        prev_target = sat_last_target.get(sat_idx)
        last_end_time = sat_last_end_time.get(sat_idx)

        self._ensure_slew_checker_initialized()

        if self._slew_checker:
            slew_result = self._slew_checker.check_slew_feasibility(
                sat.id, prev_target, task, last_end_time or window.start_time,
                window.start_time, imaging_duration
            )
            if not slew_result.feasible:
                return None, None
            actual_start = slew_result.actual_start
        else:
            actual_start = window.start_time

        actual_end = actual_start + timedelta(seconds=imaging_duration)
        return actual_start, actual_end

    def _is_saa_feasible(self, sat_id: str, window: Any) -> bool:
        """检查窗口是否满足 SAA 约束"""
        self._ensure_saa_checker_initialized()
        if self._saa_checker is None:
            return True

        saa_result = self._saa_checker.check_window_feasibility(
            sat_id, window.start_time, window.end_time
        )
        return saa_result.feasible

    def _update_scheduled_task_state(
        self, sat_idx: int, sat: Any, task: Any, window: Any,
        actual_start: datetime, actual_end: datetime, imaging_duration: float,
        state: Dict[str, Any]
    ) -> None:
        """更新已调度任务的状态"""
        state['scheduled_count'] += 1
        state['score'] += 10.0  # 基础完成奖励
        state['score'] += window.quality_score * 2.0  # 窗口质量奖励

        state['sat_task_times'][sat_idx].append((actual_start, actual_end))
        self._update_resource_usage(sat_idx, sat, task, state['sat_resources'])

        state['target_obs_count'][task.target_id] = state['target_obs_count'].get(task.target_id, 0) + 1

        state['sat_last_target'][sat_idx] = task
        state['sat_last_end_time'][sat_idx] = actual_end

    def _compute_final_fitness(self, state: Dict[str, Any]) -> float:
        """计算最终适应度分数"""
        score = state['score']

        # 资源均衡奖励
        if state['scheduled_count'] > 0:
            score += self._calculate_balance_reward(state['sat_task_times'])

        # 频次满足度奖励
        return self._calculate_frequency_fitness(state['target_obs_count'], score)

    def _calculate_balance_reward(self, sat_task_times: Dict[int, List]) -> float:
        """计算资源均衡奖励"""
        task_counts = [len(tasks) for tasks in sat_task_times.values()]
        avg_tasks = sum(task_counts) / len(task_counts)
        variance = sum((c - avg_tasks) ** 2 for c in task_counts) / len(task_counts)
        return max(0, 10 - variance)  # 越均衡奖励越高

    def _check_resource_constraints(
        self, sat_idx: int, sat: Any, task: Any, sat_resources: Dict[int, Dict[str, float]]
    ) -> bool:
        """检查资源约束"""
        resources = sat_resources[sat_idx]

        # 电量约束
        if self.consider_power:
            imaging_mode = self._select_imaging_mode(sat)
            duration = self._imaging_calculator.calculate(task, imaging_mode)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_capacity = sat.capabilities.power_capacity if hasattr(sat.capabilities, 'power_capacity') else 2800.0
            power_needed = power_capacity * power_coefficient * (duration / 3600)
            if resources['power'] < power_needed:
                return False

        # 存储约束 - 动态计算基于成像时长
        if self.consider_storage:
            imaging_mode = self._select_imaging_mode(sat)
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_needed = self._imaging_calculator.get_storage_consumption(
                task, imaging_mode, data_rate
            )
            storage_capacity = sat.capabilities.storage_capacity if hasattr(sat.capabilities, 'storage_capacity') else 128.0
            if resources['storage'] + storage_needed > storage_capacity:
                return False

        return True

    def _update_resource_usage(
        self, sat_idx: int, sat: Any, task: Any, sat_resources: Dict[int, Dict[str, float]]
    ) -> None:
        """更新资源使用"""
        resources = sat_resources[sat_idx]

        # 更新电量
        if self.consider_power:
            imaging_mode = self._select_imaging_mode(sat)
            duration = self._imaging_calculator.calculate(task, imaging_mode)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_capacity = sat.capabilities.power_capacity if hasattr(sat.capabilities, 'power_capacity') else 2800.0
            power_consumed = power_capacity * power_coefficient * (duration / 3600)
            resources['power'] -= power_consumed

        # 更新存储 - 动态计算基于成像时长
        if self.consider_storage:
            imaging_mode = self._select_imaging_mode(sat)
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_used = self._imaging_calculator.get_storage_consumption(
                task, imaging_mode, data_rate
            )
            resources['storage'] += storage_used

    def _select_imaging_mode(self, sat: Any):
        """选择成像模式"""
        from core.models import ImagingMode
        modes = sat.capabilities.imaging_modes if hasattr(sat.capabilities, 'imaging_modes') else []
        if not modes:
            return ImagingMode.PUSH_BROOM
        mode = modes[0]
        return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)

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

    def _selection(self, population: List[GAIndividual]) -> List[GAIndividual]:
        """锦标赛选择"""
        selected = []

        for _ in range(len(population)):
            # 随机选择tournament_size个个体
            tournament = random.sample(
                population,
                min(self.tournament_size, len(population))
            )

            # 选择最优
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(GAIndividual(chromosome=winner.chromosome[:]))

        return selected

    def _crossover(self, population: List[GAIndividual]) -> List[GAIndividual]:
        """单点交叉"""
        offspring = []

        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            if random.random() < self.crossover_rate and self.task_count > 1:
                # 随机选择交叉点
                crossover_point = random.randint(1, self.task_count - 1)

                # 创建子代
                child1_chrom = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
                child2_chrom = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]

                offspring.append(GAIndividual(chromosome=child1_chrom))
                offspring.append(GAIndividual(chromosome=child2_chrom))
            else:
                offspring.append(GAIndividual(chromosome=parent1.chromosome[:]))
                offspring.append(GAIndividual(chromosome=parent2.chromosome[:]))

        return offspring[:len(population)]

    def _mutation(self, population: List[GAIndividual]) -> List[GAIndividual]:
        """变异操作"""
        for individual in population[self.elitism:]:  # 不变异精英
            for i in range(len(individual.chromosome)):
                if random.random() < self.mutation_rate:
                    # 随机改变卫星分配
                    individual.chromosome[i] = random.randint(0, self.sat_count - 1)

        return population

    def _elitism(
        self,
        old_population: List[GAIndividual],
        new_population: List[GAIndividual]
    ) -> List[GAIndividual]:
        """精英保留"""
        # 按适应度排序旧种群
        sorted_old = sorted(old_population, key=lambda ind: ind.fitness, reverse=True)

        # 保留精英
        elites = [GAIndividual(chromosome=ind.chromosome[:], fitness=ind.fitness)
                  for ind in sorted_old[:self.elitism]]

        # 替换新种群中的后elitism个
        result = new_population[:-self.elitism] + elites

        return result

    def _decode_solution(
        self,
        individual: GAIndividual
    ) -> Tuple[List[ScheduledTask], Dict[str, Any]]:
        """将染色体解码为调度方案，使用完整的约束检查"""
        from ..frequency_utils import ObservationTask

        scheduled_tasks = []
        unscheduled = {}

        # 重置约束检查器状态
        if self._constraint_checker:
            self._constraint_checker.reset()

        for task_idx, sat_idx in enumerate(individual.chromosome):
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

            # 获取可见窗口
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

            # 使用约束检查器检查每个窗口
            feasible = False
            for window in windows:
                if not self._constraint_checker:
                    continue

                is_feasible, reason, info = self._constraint_checker.check_task_feasibility(
                    sat.id, task, window.start_time, window.end_time
                )

                if is_feasible and info:
                    # 创建调度任务
                    imaging_mode = info['imaging_mode']
                    slew_result = info['slew_result']
                    actual_start = info['actual_start']
                    actual_end = info['actual_end']

                    scheduled_task = self._constraint_checker.create_scheduled_task(
                        task_id=task_id,
                        sat_id=sat.id,
                        target=task,
                        imaging_mode=imaging_mode,
                        actual_start=actual_start,
                        actual_end=actual_end,
                        slew_result=slew_result
                    )

                    scheduled_tasks.append(scheduled_task)
                    feasible = True
                    break

            if not feasible:
                # 确定失败原因
                failure_reason = TaskFailureReason.NO_VISIBLE_WINDOW
                if self._constraint_checker:
                    failure_reason = self._constraint_checker.get_failure_reason(task)

                self._record_failure(
                    task_id=task_id,
                    reason=failure_reason,
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
