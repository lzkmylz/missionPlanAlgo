"""Base class for metaheuristic schedulers.

This module provides a unified base class for all metaheuristic algorithms
(GA, SA, ACO, PSO, Tabu), eliminating ~62% of duplicate code.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason
from scheduler.common import ResourceManager, TaskTimeManager, ConstraintChecker, ConstraintContext
from scheduler.common import MetaheuristicConfig, ConstraintConfig
from scheduler.common.clustering_mixin import ClusteringMixin, ClusterTask
from payload.imaging_time_calculator import ImagingTimeCalculator
from core.models import Mission, ImagingMode

# 批量约束检查器导入 - 与greedy调度器保持一致
from scheduler.constraints.unified_batch_constraint_checker import (
    UnifiedBatchConstraintChecker, UnifiedBatchCandidate
)
from scheduler.constraints.batch_slew_constraint_checker import BatchSlewConstraintChecker


@dataclass
class Solution:
    """Base class for metaheuristic solutions.

    Attributes:
        encoding: Solution encoding (algorithm-specific)
        fitness: Solution fitness score
        metadata: Additional algorithm-specific data
    """
    encoding: List[int]
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationState:
    """State tracked during solution evaluation.

    This replaces the dictionary-based state tracking used in
    individual scheduler implementations.
    """
    score: float = 0.0
    scheduled_count: int = 0
    target_obs_count: Dict[str, int] = field(default_factory=dict)
    sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = field(default_factory=dict)
    sat_resources: Dict[int, Dict[str, float]] = field(default_factory=dict)
    sat_last_target: Dict[int, Any] = field(default_factory=dict)
    sat_last_end_time: Dict[int, datetime] = field(default_factory=dict)

    @classmethod
    def initialize(cls, sat_count: int, satellites: List[Any]) -> 'EvaluationState':
        """Initialize evaluation state for given satellites."""
        return cls(
            sat_task_times={i: [] for i in range(sat_count)},
            sat_resources={
                i: {
                    'power': getattr(sat.capabilities, 'power_capacity', 2800.0),
                    'storage': 0.0
                }
                for i, sat in enumerate(satellites)
            },
        )


class MetaheuristicScheduler(BaseScheduler, ClusteringMixin, ABC):
    """Base class for metaheuristic schedulers.

    This class consolidates shared functionality across GA, SA, ACO, PSO, and Tabu
    schedulers, reducing code duplication by approximately 62%.

    Subclasses only need to implement algorithm-specific methods:
    - initialize_population(): Create initial set of solutions
    - evolve(): Perform one iteration of the algorithm

    Usage:
        class GAScheduler(MetaheuristicScheduler):
            def initialize_population(self) -> List[Solution]:
                # GA-specific initialization
                pass

            def evolve(self, population: List[Solution]) -> List[Solution]:
                # GA-specific evolution (selection, crossover, mutation)
                pass
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize metaheuristic scheduler.

        Args:
            name: Algorithm name (e.g., 'GA', 'SA')
            config: Configuration dictionary
                - enable_clustering: Enable target clustering (default False)
                - cluster_radius_km: Cluster radius in km (default 10.0)
                - min_cluster_size: Minimum targets per cluster (default 2)
        """
        super().__init__(name, config)
        ClusteringMixin.__init__(self, config)
        config = config or {}

        # Convert legacy config to new format
        self._config = self._convert_config(config)

        # Runtime data (initialized in schedule())
        self.tasks: List[Any] = []
        self.satellites: List[Any] = []
        self.task_count = 0
        self.sat_count = 0

        # Shared components (initialized in schedule())
        self._resource_manager: Optional[ResourceManager] = None
        self._constraint_checker: Optional[ConstraintChecker] = None
        self._task_time_manager: Optional[TaskTimeManager] = None
        self._imaging_calculator: Optional[ImagingTimeCalculator] = None

        # Convergence tracking
        self._convergence_curve: List[float] = []
        self._iterations = 0

    def _convert_config(self, config: Dict[str, Any]) -> MetaheuristicConfig:
        """Convert legacy config dict to MetaheuristicConfig."""
        # Extract constraint settings
        # 高精度要求：强制使用标准模式（精确计算），禁止简化模式
        if config.get('use_simplified_slew', False):
            raise ValueError(
                "use_simplified_slew=True is not allowed. "
                "High precision mode requires exact calculations."
            )

        constraints = ConstraintConfig(
            consider_power=config.get('consider_power', True),
            consider_storage=config.get('consider_storage', True),
            mode='standard',  # 强制使用标准模式（精确计算）
            enable_saa_check=config.get('enable_saa_check', True),
            enable_attitude_calculation=config.get('enable_attitude_calculation', True),
        )

        # Get max_iterations (support both 'max_iterations' and 'generations')
        max_iter = config.get('max_iterations') or config.get('generations', 1000)

        return MetaheuristicConfig(
            max_iterations=max_iter,
            population_size=config.get('population_size', 50),
            crossover_rate=config.get('crossover_rate', 0.8),
            mutation_rate=config.get('mutation_rate', 0.1),
            elitism_count=config.get('elitism', 2),
            initial_temperature=config.get('initial_temperature', 100.0),
            cooling_rate=config.get('cooling_rate', 0.95),
            num_ants=config.get('num_ants', 20),
            num_particles=config.get('num_particles', 30),
            constraints=constraints,
        )

    def schedule(self) -> ScheduleResult:
        """Execute metaheuristic scheduling.

        This template method defines the common scheduling flow:
        1. Initialize components
        2. Create initial population
        3. Iterate: evaluate -> evolve
        4. Decode best solution

        Returns:
            ScheduleResult with scheduled tasks
        """
        self._start_timer()
        self._validate_initialization()

        # Reset clustering state if needed
        if self.enable_clustering:
            self.reset_clustering_state()

        # Prepare data
        if self.enable_clustering:
            self.tasks = self._get_clustered_tasks()
        else:
            self.tasks = self._create_frequency_aware_tasks()
        self.satellites = list(self.mission.satellites)
        self.task_count = len(self.tasks)
        self.sat_count = len(self.satellites)

        if self.task_count == 0 or self.sat_count == 0:
            return self._build_empty_result()

        # Initialize shared components
        self._initialize_components()

        # Algorithm-specific initialization
        population = self.initialize_population()

        # Evaluate initial population
        for solution in population:
            solution.fitness = self._evaluate(solution)

        # Track convergence
        best_fitness = max(sol.fitness for sol in population)
        self._convergence_curve = [best_fitness]

        # Main optimization loop
        max_iterations = getattr(self._config, 'generations', self._config.max_iterations)
        for iteration in range(max_iterations):
            # Evolve population (algorithm-specific)
            population = self.evolve(population)

            # Evaluate new population
            for solution in population:
                if solution.fitness == 0.0:  # Only evaluate if not already evaluated
                    solution.fitness = self._evaluate(solution)

            # Track best fitness
            current_best = max(sol.fitness for sol in population)
            best_fitness = max(best_fitness, current_best)
            self._convergence_curve.append(best_fitness)
            self._iterations = iteration + 1

            # Check convergence
            if self._check_convergence():
                break

        # Get best solution
        best_solution = max(population, key=lambda sol: sol.fitness)

        # Decode to schedule
        scheduled_tasks, unscheduled = self._decode_solution(best_solution)

        # Build result
        makespan = self._calculate_makespan(scheduled_tasks)
        computation_time = self._stop_timer()

        return ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks=unscheduled,
            makespan=makespan,
            computation_time=computation_time,
            iterations=self._iterations,
            convergence_curve=self._convergence_curve,
        )

    def _initialize_components(self) -> None:
        """Initialize shared components."""
        # Resource manager
        self._resource_manager = ResourceManager(
            satellites=self.satellites,
            start_time=self.mission.start_time,
            consider_power=self._config.constraints.consider_power,
            consider_storage=self._config.constraints.consider_storage,
        )

        # Task time manager
        self._task_time_manager = TaskTimeManager(self.sat_count)

        # Imaging calculator
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=self._config.min_imaging_duration,
            max_duration=self._config.max_imaging_duration,
            default_duration=self._config.default_imaging_duration,
        )

        # 初始化批量约束检查器（与greedy调度器保持一致，使用向量化批量优化）
        # 高精度要求：强制使用批量约束检查器，禁止简化模式
        self._batch_constraint_checker = UnifiedBatchConstraintChecker(
            mission=self.mission,
            use_precise_model=True,
            consider_power=self._config.constraints.consider_power,
            consider_storage=self._config.constraints.consider_storage
        )
        # 同时初始化基类的批量检查器
        self._slew_checker = BatchSlewConstraintChecker(
            self.mission,
            use_precise_model=True
        )

        # 预计算卫星位置以加速姿态角计算（默认启用，除非显式禁用）
        precompute_enabled = self.config.get('precompute_positions', True) if self.config else True
        if precompute_enabled:
            print("    预计算卫星位置...")
            step_seconds = self.config.get('precompute_step_seconds', 1.0) if self.config else 1.0
            self._precompute_satellite_positions(time_step_seconds=step_seconds)

    @abstractmethod
    def initialize_population(self) -> List[Solution]:
        """Initialize population of solutions.

        This method must be implemented by subclasses to create
        the initial set of solutions for the algorithm.

        Returns:
            List of Solution objects
        """
        pass

    @abstractmethod
    def evolve(self, population: List[Solution]) -> List[Solution]:
        """Evolve population for one iteration.

        This method must be implemented by subclasses to perform
        one iteration of the metaheuristic algorithm.

        Args:
            population: Current population of solutions

        Returns:
            New population after evolution
        """
        pass

    def _evaluate(self, solution: Solution) -> float:
        """Evaluate solution fitness.

        This unified evaluation method replaces the duplicate evaluation
        logic in individual scheduler implementations.

        Args:
            solution: Solution to evaluate

        Returns:
            Fitness score
        """
        state = EvaluationState.initialize(self.sat_count, self.satellites)

        for task_idx, sat_idx in enumerate(solution.encoding):
            if task_idx >= len(self.tasks) or sat_idx >= self.sat_count:
                continue

            self._evaluate_task_assignment(task_idx, sat_idx, state)

        return self._compute_final_fitness(state)

    # Backward compatibility: alias for _evaluate
    _evaluate_solution = _evaluate

    def _evaluate_task_assignment(
        self, task_idx: int, sat_idx: int, state: EvaluationState
    ) -> None:
        """Evaluate single task assignment.

        Uses batch constraint checking (UnifiedBatchConstraintChecker) for consistency
        with greedy scheduler and optimal performance.

        Args:
            task_idx: Index of task in tasks list
            sat_idx: Index of assigned satellite
            state: Current evaluation state
        """
        task = self.tasks[task_idx]
        sat = self.satellites[sat_idx]

        # Find feasible window
        feasible_window = self._find_feasible_window(sat_idx, sat, task, state)
        if not feasible_window:
            return

        # Calculate timing
        imaging_mode = self._select_imaging_mode(sat)
        imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)

        actual_start, actual_end = self._calculate_task_timing(
            sat_idx, sat, task, feasible_window, imaging_duration, state
        )

        if not actual_start or actual_end > feasible_window.end_time:
            return

        # 使用批量约束检查器（与greedy调度器一致）
        if hasattr(self, '_batch_constraint_checker') and self._batch_constraint_checker is not None:
            # 构建批量检查候选
            prev_target = state.sat_last_target.get(sat_idx)
            prev_end_time = state.sat_last_end_time.get(sat_idx)

            # 计算资源需求
            power_needed = 0.0
            storage_produced = 0.0
            if self._config.constraints.consider_power:
                power_coefficient = 0.1  # 默认系数
                power_needed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)
            if self._config.constraints.consider_storage:
                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                storage_produced = self._imaging_calculator.get_storage_consumption(task, imaging_mode, data_rate)

            # 获取卫星位置（从缓存或实时计算）
            sat_position = None
            sat_velocity = None
            if hasattr(self, '_position_cache') and self._position_cache is not None:
                # 尝试从位置缓存获取
                pos_vel = self._position_cache.get_position(sat.id, actual_start)
                if pos_vel:
                    sat_position, sat_velocity = pos_vel

            # 如果缓存中没有，尝试从姿态计算器获取
            if sat_position is None and hasattr(self, '_attitude_calculator'):
                try:
                    sat_position, sat_velocity = self._attitude_calculator._get_satellite_state(sat, actual_start)
                except Exception as e:
                    # 高精度要求：卫星状态获取失败应抛出错误
                    raise RuntimeError(f"卫星状态获取失败 ({sat.id} at {actual_start}): {e}") from e

            # 高精度要求：必须有有效的卫星位置
            if sat_position is None:
                raise RuntimeError(f"无法获取卫星 {sat.id} 在 {actual_start} 的位置信息")

            candidate = UnifiedBatchCandidate(
                sat_id=sat.id,
                satellite=sat,
                target=task,
                window_start=actual_start,
                window_end=actual_end,
                prev_end_time=prev_end_time if prev_end_time else self.mission.start_time,
                imaging_duration=imaging_duration,
                prev_target=prev_target,
                power_needed=power_needed,
                storage_produced=storage_produced,
                sat_position=sat_position,
                sat_velocity=sat_velocity
            )

            # 构建卫星状态字典
            satellite_states = {
                sat.id: {
                    'power': state.sat_resources[sat_idx]['power'],
                    'storage': state.sat_resources[sat_idx]['storage']
                }
            }

            # 构建已调度任务列表（用于时间冲突检查）
            existing_tasks = []
            for start, end in state.sat_task_times[sat_idx]:
                existing_tasks.append({
                    'satellite_id': sat.id,
                    'start_time': start,
                    'end_time': end
                })

            # 执行批量约束检查
            results = self._batch_constraint_checker.check_all_constraints_batch(
                candidates=[candidate],
                existing_tasks=existing_tasks,
                satellite_states=satellite_states,
                early_termination=True
            )

            if not results or not results[0].feasible:
                return

            result = results[0]

            # Update state for scheduled task
            state.score += 10.0
            state.scheduled_count += 1

            target_id = getattr(task, 'target_id', str(task_idx))
            state.target_obs_count[target_id] = state.target_obs_count.get(target_id, 0) + 1

            state.sat_task_times[sat_idx].append((actual_start, actual_end))
            state.sat_last_target[sat_idx] = task
            state.sat_last_end_time[sat_idx] = actual_end

            # Update resources
            if self._config.constraints.consider_power:
                state.sat_resources[sat_idx]['power'] -= power_needed
            if self._config.constraints.consider_storage:
                state.sat_resources[sat_idx]['storage'] += storage_produced
        else:
            # 高精度要求：必须使用批量约束检查器
            raise RuntimeError(
                "Batch constraint checker not initialized. "
                "High precision mode requires UnifiedBatchConstraintChecker."
            )

    def _find_feasible_window(
        self, sat_idx: int, sat: Any, task: Any, state: EvaluationState
    ) -> Optional[Any]:
        """Find feasible time window for task."""
        if not self.window_cache:
            return None

        windows = self.window_cache.get_windows(sat.id, getattr(task, 'target_id', None))
        if not windows:
            return None

        for window in windows:
            if self._task_time_manager.is_time_feasible(
                sat_idx, window.start_time, window.end_time
            ):
                return window
        return None

    def _calculate_task_timing(
        self, sat_idx: int, sat: Any, task: Any,
        window: Any, imaging_duration: float, state: EvaluationState
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Calculate actual start and end times for task."""
        last_end = state.sat_last_end_time.get(sat_idx) or window.start_time
        slew_time = 10.0  # Default slew time

        actual_start = max(window.start_time, last_end + timedelta(seconds=slew_time))
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        return actual_start, actual_end

    def _is_saa_feasible(self, sat_id: str, window: Any) -> bool:
        """Check if window is feasible w.r.t. SAA constraints.

        Uses the batch constraint checker for optimal performance.
        """
        if not hasattr(window, 'start_time') or not hasattr(window, 'end_time'):
            return True  # Invalid window format

        # 优先使用批量检查器的SAA检查器
        if hasattr(self, '_batch_constraint_checker') and self._batch_constraint_checker is not None:
            if hasattr(self._batch_constraint_checker, '_saa_checker'):
                result = self._batch_constraint_checker._saa_checker.check_saa(
                    sat_id=sat_id,
                    start_time=window.start_time,
                    end_time=window.end_time,
                )
                return result.feasible

        # 回退到传统检查器
        if hasattr(self, '_constraint_checker') and self._constraint_checker is not None:
            if hasattr(self._constraint_checker, '_saa_checker'):
                result = self._constraint_checker._saa_checker.check_saa(
                    sat_id=sat_id,
                    start_time=window.start_time,
                    end_time=window.end_time,
                )
                return result.feasible

        return True  # Fallback if SAA checker not available

    def _compute_final_fitness(self, state: EvaluationState) -> float:
        """Compute final fitness score from evaluation state."""
        score = state.score

        # Add balance reward
        if state.scheduled_count > 0:
            task_counts = [len(tasks) for tasks in state.sat_task_times.values()]
            avg_tasks = sum(task_counts) / len(task_counts)
            variance = sum((c - avg_tasks) ** 2 for c in task_counts) / len(task_counts)
            balance_reward = max(0, 10 - variance)
            score += balance_reward

        # Add frequency satisfaction reward
        score = self._calculate_frequency_fitness(state.target_obs_count, score)

        return score

    def _decode_solution(
        self, solution: Solution
    ) -> Tuple[List[ScheduledTask], Dict[str, TaskFailureReason]]:
        """Decode solution to scheduled tasks.

        Args:
            solution: Best solution from optimization

        Returns:
            Tuple of (scheduled_tasks, unscheduled_tasks)
        """
        scheduled_tasks: List[ScheduledTask] = []
        unscheduled: Dict[str, TaskFailureReason] = {}

        # Reset task time manager
        self._task_time_manager.reset()

        state = EvaluationState.initialize(self.sat_count, self.satellites)

        for task_idx, sat_idx in enumerate(solution.encoding):
            if task_idx >= len(self.tasks):
                continue

            task = self.tasks[task_idx]
            sat = self.satellites[sat_idx]

            feasible_window = self._find_feasible_window(sat_idx, sat, task, state)
            if not feasible_window:
                unscheduled[getattr(task, 'id', str(task_idx))] = TaskFailureReason.NO_VISIBLE_WINDOW
                continue

            imaging_mode = self._select_imaging_mode(sat)
            imaging_duration = self._imaging_calculator.calculate(task, imaging_mode)

            actual_start, actual_end = self._calculate_task_timing(
                sat_idx, sat, task, feasible_window, imaging_duration, state
            )

            if not actual_start:
                unscheduled[getattr(task, 'id', str(task_idx))] = TaskFailureReason.CONSTRAINT_VIOLATION
                continue

            # 计算机动时间和角度
            slew_time = 10.0  # 默认机动时间
            slew_angle = 0.0
            prev_end = state.sat_last_end_time.get(sat_idx)
            if prev_end:
                # 计算与上一个任务的时间间隔作为机动时间
                time_gap = (actual_start - prev_end).total_seconds()
                slew_time = max(0, min(time_gap, 60.0))  # 限制在0-60秒
                # 如果有前一个目标，计算机动角度
                prev_target = state.sat_last_target.get(sat_idx)
                if prev_target and hasattr(prev_target, 'latitude') and hasattr(task, 'latitude'):
                    import math
                    lat_diff = abs(task.latitude - prev_target.latitude)
                    lon_diff = abs(task.longitude - prev_target.longitude)
                    slew_angle = math.sqrt(lat_diff**2 + lon_diff**2)

            # 计算资源使用
            storage_used = 0.0
            if self._config.constraints.consider_storage:
                storage_used = self._imaging_calculator.get_storage_consumption(
                    task, imaging_mode, getattr(sat.capabilities, 'data_rate', 300.0)
                )
            current_storage = state.sat_resources.get(sat_idx, {}).get('storage', 0)

            # Create scheduled task
            scheduled_task = ScheduledTask(
                task_id=getattr(task, 'id', str(task_idx)),
                satellite_id=sat.id,
                target_id=getattr(task, 'target_id', str(task_idx)),
                imaging_start=actual_start,
                imaging_end=actual_end,
                imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
                slew_angle=slew_angle,
                slew_time=slew_time,
                storage_before=current_storage,
                storage_after=current_storage + storage_used,
            )

            # 计算姿态角（如果启用）
            if self._enable_attitude_calculation:
                attitude = self._calculate_attitude_angles(sat, task, actual_start)
                if attitude:
                    scheduled_task.roll_angle = attitude.roll
                    scheduled_task.pitch_angle = attitude.pitch
                    scheduled_task.yaw_angle = attitude.yaw

            scheduled_tasks.append(scheduled_task)

            # 更新资源状态
            if self._config.constraints.consider_storage:
                state.sat_resources[sat_idx]['storage'] = current_storage + storage_used

            # Update state
            state.sat_task_times[sat_idx].append((actual_start, actual_end))
            state.sat_last_target[sat_idx] = task
            state.sat_last_end_time[sat_idx] = actual_end

            # 如果是聚类任务，记录聚类调度信息
            if isinstance(task, ClusterTask) and self.enable_clustering:
                self._record_cluster_schedule(
                    task=task,
                    satellite_id=sat.id,
                    imaging_start=actual_start,
                    imaging_end=actual_end,
                    look_angle=0.0  # 元启发式算法中暂未计算机动角度
                )

        return scheduled_tasks, unscheduled

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self._convergence_curve) < self._config.max_no_improvement:
            return False

        recent = self._convergence_curve[-self._config.max_no_improvement:]
        return max(recent) - min(recent) < self._config.convergence_threshold

    @staticmethod
    def _select_imaging_mode(sat: Any) -> ImagingMode:
        """Select imaging mode for satellite."""
        try:
            modes = sat.capabilities.imaging_modes if hasattr(sat.capabilities, 'imaging_modes') else []
            if not modes or not hasattr(modes, '__getitem__'):
                return ImagingMode.PUSH_BROOM
            mode = modes[0]
            return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)
        except (TypeError, AttributeError, IndexError):
            return ImagingMode.PUSH_BROOM

    def _calculate_makespan(self, scheduled_tasks: List[ScheduledTask]) -> float:
        """Calculate makespan from scheduled tasks."""
        if not scheduled_tasks:
            return 0.0
        last_end = max(t.imaging_end for t in scheduled_tasks)
        return (last_end - self.mission.start_time).total_seconds()

    def _build_empty_result(self) -> ScheduleResult:
        """Build empty schedule result."""
        return ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=0.0,
            computation_time=self._stop_timer(),
            iterations=0,
            convergence_curve=[],
        )

    # Validation helpers
    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        """Validate positive integer parameter."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value

    @staticmethod
    def _validate_positive_float(value: float, name: str) -> float:
        """Validate positive float parameter."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value}")
        return float(value)

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        """Validate probability parameter [0, 1]."""
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return float(value)
