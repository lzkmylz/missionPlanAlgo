"""
粒子群优化调度器 (PSO Scheduler)

基于粒子群优化算法原理实现的卫星任务规划调度器。
设计文档第4章/第7章 - PSO算法实现
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from ..base_scheduler import BaseScheduler, ScheduleResult, ScheduledTask, TaskFailureReason


@dataclass
class Particle:
    """粒子个体"""
    position: np.ndarray  # 位置向量（任务到卫星的分配）
    velocity: np.ndarray  # 速度向量
    best_position: np.ndarray = field(default_factory=lambda: np.array([]))  # 个体最优位置
    best_fitness: float = float('-inf')  # 个体最优适应度
    fitness: float = 0.0  # 当前适应度


class PSOScheduler(BaseScheduler):
    """
    粒子群优化调度器

    算法特点：
    - 模拟鸟群觅食行为，通过粒子协作寻找最优解
    - 每个粒子记录个体最优，群体共享全局最优
    - 速度更新考虑惯性、认知和社会三部分

    关键参数：
    - num_particles: 粒子数量
    - inertia_weight: 惯性权重
    - cognitive_coeff: 认知系数（个体最优影响）
    - social_coeff: 社会系数（全局最优影响）
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化PSO调度器

        Args:
            config: 配置参数
                - num_particles: 粒子数量（默认30）
                - max_iterations: 最大迭代次数（默认100）
                - inertia_weight: 惯性权重（默认0.9）
                - cognitive_coeff: 认知系数（默认2.0）
                - social_coeff: 社会系数（默认2.0）
                - random_seed: 随机种子（可选）
        """
        super().__init__("PSO", config)
        config = config or {}

        # 参数验证
        self.num_particles = self._validate_positive_int(
            config.get('num_particles', 30), 'num_particles'
        )
        self.max_iterations = self._validate_positive_int(
            config.get('max_iterations', 100), 'max_iterations'
        )
        self.inertia_weight = self._validate_probability(
            config.get('inertia_weight', 0.9), 'inertia_weight'
        )
        self.cognitive_coeff = self._validate_non_negative_float(
            config.get('cognitive_coeff', 2.0), 'cognitive_coeff'
        )
        self.social_coeff = self._validate_non_negative_float(
            config.get('social_coeff', 2.0), 'social_coeff'
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
        self.swarm: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float('-inf')

    def _validate_positive_int(self, value: int, name: str) -> int:
        """验证正整数参数"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value

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
            'num_particles': self.num_particles,
            'max_iterations': self.max_iterations,
            'inertia_weight': self.inertia_weight,
            'cognitive_coeff': self.cognitive_coeff,
            'social_coeff': self.social_coeff,
            'random_seed': self.random_seed,
        }

    def schedule(self) -> ScheduleResult:
        """
        执行粒子群优化调度

        Returns:
            ScheduleResult: 调度结果
        """
        self._start_timer()

        self._validate_initialization()

        # 准备数据
        self.tasks = list(self.mission.targets)
        self.satellites = list(self.mission.satellites)
        self.task_count = len(self.tasks)
        self.sat_count = len(self.satellites)

        # 处理空场景
        if self.task_count == 0 or self.sat_count == 0:
            return self._build_empty_result()

        # 初始化粒子群
        self._initialize_swarm()

        # 记录收敛曲线
        self._convergence_curve = []

        # 粒子群优化主循环
        for iteration in range(self.max_iterations):
            # 评估所有粒子
            for particle in self.swarm:
                particle.fitness = self._evaluate(particle)

                # 更新个体最优
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                # 更新全局最优
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()

            # 记录收敛
            self._convergence_curve.append(self.global_best_fitness)
            self._iterations = iteration + 1

            # 更新粒子速度和位置
            self._update_swarm()

        # 解码最优解
        if self.global_best_position is None:
            self.global_best_position = np.zeros(self.task_count, dtype=int)

        scheduled_tasks, unscheduled = self._decode_solution(self.global_best_position)

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

    def _initialize_swarm(self) -> None:
        """初始化粒子群"""
        self.swarm = []

        for _ in range(self.num_particles):
            # 随机初始化位置（卫星索引）
            position = np.random.randint(0, self.sat_count, size=self.task_count)

            # 随机初始化速度（范围：-sat_count到+sat_count）
            velocity = np.random.uniform(-self.sat_count, self.sat_count, size=self.task_count)

            particle = Particle(
                position=position.astype(float),
                velocity=velocity,
                best_position=position.copy().astype(float),
                best_fitness=float('-inf')
            )

            self.swarm.append(particle)

    def _update_swarm(self) -> None:
        """更新粒子群的速度和位置"""
        if self.global_best_position is None:
            return

        for particle in self.swarm:
            # 生成随机数
            r1 = np.random.random(self.task_count)
            r2 = np.random.random(self.task_count)

            # 更新速度
            # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            inertia = self.inertia_weight * particle.velocity
            cognitive = self.cognitive_coeff * r1 * (particle.best_position - particle.position)
            social = self.social_coeff * r2 * (self.global_best_position - particle.position)

            particle.velocity = inertia + cognitive + social

            # 限制速度范围
            max_velocity = self.sat_count
            particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)

            # 更新位置
            particle.position = particle.position + particle.velocity

            # 将位置映射到有效的卫星索引
            # 使用四舍五入并限制在有效范围内
            particle.position = np.clip(
                np.round(particle.position),
                0,
                self.sat_count - 1
            )

    def _evaluate(self, particle: Particle) -> float:
        """
        评估粒子适应度

        适应度函数：
        - 基础：完成的任务数量 × 10
        - 奖励：窗口质量、资源均衡
        - 惩罚：时间冲突
        """
        score = 0.0
        scheduled_count = 0
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }

        # 将位置转换为整数索引
        assignment = particle.position.astype(int)

        for task_idx, sat_idx in enumerate(assignment):
            if task_idx >= len(self.tasks):
                continue

            # 确保卫星索引在有效范围内
            sat_idx = int(np.clip(sat_idx, 0, self.sat_count - 1))

            task = self.tasks[task_idx]
            sat = self.satellites[sat_idx]

            # 检查是否有可见窗口
            if self.window_cache:
                windows = self.window_cache.get_windows(sat.id, task.id)
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

        # 资源均衡奖励
        if scheduled_count > 0:
            task_counts = [len(tasks) for tasks in sat_task_times.values()]
            avg_tasks = sum(task_counts) / len(task_counts)
            variance = sum((c - avg_tasks) ** 2 for c in task_counts) / len(task_counts)
            balance_reward = max(0, 10 - variance)
            score += balance_reward

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
        position: np.ndarray
    ) -> Tuple[List[ScheduledTask], Dict[str, Any]]:
        """将粒子位置解码为调度方案"""
        scheduled_tasks = []
        unscheduled = {}

        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(self.sat_count)
        }

        # 将位置转换为整数索引
        assignment = position.astype(int)

        for task_idx, sat_idx in enumerate(assignment):
            if task_idx >= len(self.tasks):
                continue

            # 确保卫星索引在有效范围内
            sat_idx = int(np.clip(sat_idx, 0, self.sat_count - 1))

            task = self.tasks[task_idx]
            sat = self.satellites[sat_idx]

            # 获取可见窗口
            if self.window_cache:
                windows = self.window_cache.get_windows(sat.id, task.id)
            else:
                windows = []

            if not windows:
                self._record_failure(
                    task_id=task.id,
                    reason=TaskFailureReason.NO_VISIBLE_WINDOW,
                    detail=f"No visibility window for satellite {sat.id}"
                )
                unscheduled[task.id] = self._failure_log[-1]
                continue

            # 查找可行窗口
            feasible_window = None
            for window in windows:
                if self._is_time_feasible(sat_idx, window.start_time, window.end_time, sat_task_times):
                    feasible_window = window
                    break

            if feasible_window:
                scheduled_task = ScheduledTask(
                    task_id=task.id,
                    satellite_id=sat.id,
                    target_id=task.id,
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
                    task_id=task.id,
                    reason=TaskFailureReason.TIME_CONFLICT,
                    detail=f"No feasible time window for satellite {sat.id}"
                )
                unscheduled[task.id] = self._failure_log[-1]

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
