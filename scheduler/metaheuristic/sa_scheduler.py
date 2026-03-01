"""
模拟退火调度器 (SA Scheduler)

基于模拟退火算法原理实现的卫星任务规划调度器。
设计文档第4章/第7章 - SA算法实现
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile


@dataclass
class SASolution:
    """模拟退火解"""
    assignment: List[int]  # 每个任务分配的卫星索引
    fitness: float = 0.0


class SAScheduler(BaseScheduler):
    """
    模拟退火调度器

    算法特点：
    - 以一定概率接受较差解，避免陷入局部最优
    - 温度逐渐降低，接受较差解的概率减小
    - 适合解决组合优化问题

    关键参数：
    - initial_temperature: 初始温度
    - cooling_rate: 冷却率（0-1）
    - max_iterations: 最大迭代次数
    - min_temperature: 终止温度
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化SA调度器

        Args:
            config: 配置参数
                - initial_temperature: 初始温度（默认100）
                - cooling_rate: 冷却率（默认0.98）
                - max_iterations: 最大迭代次数（默认1000）
                - min_temperature: 最小温度（默认0.001）
                - random_seed: 随机种子（可选）
                - consider_power: 是否考虑电量约束（默认True）
                - consider_storage: 是否考虑存储约束（默认True）
        """
        super().__init__("SA", config)
        config = config or {}

        # 参数验证
        self.initial_temperature = self._validate_positive_float(
            config.get('initial_temperature', 100.0), 'initial_temperature'
        )
        self.cooling_rate = self._validate_probability(
            config.get('cooling_rate', 0.98), 'cooling_rate'
        )
        self.max_iterations = self._validate_positive_int(
            config.get('max_iterations', 1000), 'max_iterations'
        )
        self.min_temperature = self._validate_positive_float(
            config.get('min_temperature', 0.001), 'min_temperature'
        )

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
        self.current_temperature = self.initial_temperature

        # 初始化成像时间计算器和功率配置文件
        # 使用ImagingTimeCalculator的默认值（基于实际卫星数据）
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=config.get('min_imaging_duration'),
            max_duration=config.get('max_imaging_duration'),
            default_duration=config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(config.get('power_coefficients'))

    def _validate_positive_float(self, value: float, name: str) -> float:
        """验证正浮点数参数"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value}")
        return float(value)

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
            'initial_temperature': self.initial_temperature,
            'cooling_rate': self.cooling_rate,
            'max_iterations': self.max_iterations,
            'min_temperature': self.min_temperature,
            'random_seed': self.random_seed,
            'consider_power': self.consider_power,
            'consider_storage': self.consider_storage,
        }

    def schedule(self) -> ScheduleResult:
        """
        执行模拟退火调度

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

        # 初始化当前解
        current_solution = self._initialize_solution()
        current_solution.fitness = self._evaluate(current_solution)

        # 记录最优解
        best_solution = SASolution(
            assignment=current_solution.assignment[:],
            fitness=current_solution.fitness
        )

        # 记录收敛曲线
        self._convergence_curve = [best_solution.fitness]
        self.current_temperature = self.initial_temperature

        # 模拟退火主循环
        iteration = 0
        while (self.current_temperature > self.min_temperature and
               iteration < self.max_iterations):

            # 生成邻域解
            neighbor = self._generate_neighbor(current_solution)
            neighbor.fitness = self._evaluate(neighbor)

            # 计算适应度差
            delta = neighbor.fitness - current_solution.fitness

            # 接受准则
            if delta > 0:
                # 新解更好，直接接受
                current_solution = SASolution(
                    assignment=neighbor.assignment[:],
                    fitness=neighbor.fitness
                )
            else:
                # 新解较差，以概率接受
                acceptance_probability = math.exp(delta / self.current_temperature)
                if random.random() < acceptance_probability:
                    current_solution = SASolution(
                        assignment=neighbor.assignment[:],
                        fitness=neighbor.fitness
                    )

            # 更新最优解
            if current_solution.fitness > best_solution.fitness:
                best_solution = SASolution(
                    assignment=current_solution.assignment[:],
                    fitness=current_solution.fitness
                )

            # 记录收敛
            self._convergence_curve.append(best_solution.fitness)

            # 降温
            self.current_temperature *= self.cooling_rate
            iteration += 1
            self._iterations = iteration

        # 解码最优解
        scheduled_tasks, unscheduled = self._decode_solution(best_solution)

        # 计算指标
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=iteration,
            convergence_curve=self._convergence_curve
        )

    def _initialize_solution(self) -> SASolution:
        """初始化解"""
        assignment = [
            random.randint(0, self.sat_count - 1)
            for _ in range(self.task_count)
        ]
        return SASolution(assignment=assignment)

    def _generate_neighbor(self, solution: SASolution) -> SASolution:
        """
        生成邻域解

        通过随机改变一个任务的卫星分配
        """
        neighbor = SASolution(assignment=solution.assignment[:])

        # 随机选择一个任务
        task_idx = random.randint(0, self.task_count - 1)

        # 随机分配一个新的卫星
        neighbor.assignment[task_idx] = random.randint(0, self.sat_count - 1)

        return neighbor

    def _evaluate(self, solution: SASolution) -> float:
        """
        评估解的适应度

        适应度函数：
        - 基础：完成的任务数量 × 10
        - 奖励：窗口质量、资源均衡
        - 惩罚：时间冲突、资源约束违反
        - 频次满足度奖励
        """
        from ..frequency_utils import ObservationTask

        score = 0.0
        scheduled_count = 0
        target_obs_count: Dict[str, int] = {}  # 记录每个目标的实际观测次数
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }
        # 跟踪资源使用情况
        sat_resources: Dict[int, Dict[str, float]] = {
            i: {
                'power': sat.capabilities.power_capacity if hasattr(sat.capabilities, 'power_capacity') else 2800.0,
                'storage': 0.0
            }
            for i, sat in enumerate(self.satellites)
        }

        for task_idx, sat_idx in enumerate(solution.assignment):
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

            # 检查时间冲突和资源约束
            feasible_window = None
            for window in windows:
                if self._is_time_feasible(sat_idx, window.start_time, window.end_time, sat_task_times):
                    # 检查资源约束
                    if self._check_resource_constraints(sat_idx, sat, task, sat_resources):
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

                # 更新资源使用
                self._update_resource_usage(sat_idx, sat, task, sat_resources)

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

    def _decode_solution(
        self,
        solution: SASolution
    ) -> Tuple[List[ScheduledTask], Dict[str, Any]]:
        """将解解码为调度方案"""
        from ..frequency_utils import ObservationTask

        scheduled_tasks = []
        unscheduled = {}

        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }
        # 跟踪资源使用情况
        sat_resources: Dict[int, Dict[str, float]] = {
            i: {
                'power': sat.capabilities.power_capacity if hasattr(sat.capabilities, 'power_capacity') else 2800.0,
                'storage': 0.0
            }
            for i, sat in enumerate(self.satellites)
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

            # 查找可行窗口（考虑时间和资源约束）
            feasible_window = None
            for window in windows:
                if self._is_time_feasible(sat_idx, window.start_time, window.end_time, sat_task_times):
                    if self._check_resource_constraints(sat_idx, sat, task, sat_resources):
                        feasible_window = window
                        break

            if feasible_window:
                # 计算资源消耗
                imaging_mode = self._select_imaging_mode(sat)
                duration = self._imaging_calculator.calculate(task, imaging_mode)

                power_before = sat_resources[sat_idx]['power']
                storage_before = sat_resources[sat_idx]['storage']

                # 更新资源使用
                self._update_resource_usage(sat_idx, sat, task, sat_resources)

                scheduled_task = ScheduledTask(
                    task_id=task_id,
                    satellite_id=sat.id,
                    target_id=target_id,
                    imaging_start=feasible_window.start_time,
                    imaging_end=feasible_window.end_time,
                    imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
                    power_before=power_before,
                    power_after=sat_resources[sat_idx]['power'],
                    storage_before=storage_before,
                    storage_after=sat_resources[sat_idx]['storage']
                )
                scheduled_tasks.append(scheduled_task)
                sat_task_times[sat_idx].append(
                    (feasible_window.start_time, feasible_window.end_time)
                )
            else:
                # 确定失败原因
                has_window = any(
                    self._is_time_feasible(sat_idx, w.start_time, w.end_time, sat_task_times)
                    for w in windows
                )
                if has_window:
                    # 有窗口但无法调度，可能是资源约束
                    reason = TaskFailureReason.POWER_CONSTRAINT
                else:
                    reason = TaskFailureReason.TIME_CONFLICT

                self._record_failure(
                    task_id=task_id,
                    reason=reason,
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
