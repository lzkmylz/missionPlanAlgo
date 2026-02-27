#!/usr/bin/env python3
"""
支持观测频次需求的调度脚本

Usage:
    python scripts/run_scheduler_with_frequency.py --cache data/visibility_cache/point_group_scenario_windows.json --algorithm ga
"""

import argparse
import json
import sys
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import Mission
from core.orbit.visibility.window_cache import VisibilityWindowCache
from core.orbit.visibility.base import VisibilityWindow
from scheduler.base_scheduler import ScheduleResult, ScheduledTask, TaskFailure, TaskFailureReason
from evaluation.metrics import MetricsCalculator


@dataclass
class ObservationTask:
    """观测任务 - 支持多次观测"""
    task_id: str
    target_id: str
    target_name: str
    observation_idx: int  # 第几次观测（0-based）
    required_observations: int  # -1表示不限
    priority: int
    longitude: float
    latitude: float


class FrequencyAwareScheduler:
    """支持观测频次需求的调度器"""

    def __init__(self, mission: Mission, window_cache: VisibilityWindowCache, config: Dict = None):
        self.mission = mission
        self.window_cache = window_cache
        self.config = config or {}
        self.satellites = list(mission.satellites)
        self.targets = list(mission.targets)

        # 创建观测任务列表
        self.observation_tasks = self._create_observation_tasks()

        # 算法参数（GA）
        self.population_size = self.config.get('population_size', 50)
        self.generations = self.config.get('generations', 100)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.elitism = self.config.get('elitism', 5)

        # 统计
        self.convergence_curve = []

    def _create_observation_tasks(self) -> List[ObservationTask]:
        """根据目标的观测需求创建任务列表"""
        tasks = []

        for target in self.targets:
            required = getattr(target, 'required_observations', 1)

            if required == -1:
                # 不限频次 - 为每个可见窗口创建一个任务（最多10个）
                max_obs = 10
            else:
                max_obs = required

            for idx in range(max_obs):
                task_id = f"{target.id}-OBS{idx+1}"
                tasks.append(ObservationTask(
                    task_id=task_id,
                    target_id=target.id,
                    target_name=target.name,
                    observation_idx=idx,
                    required_observations=required,
                    priority=target.priority,
                    longitude=target.longitude,
                    latitude=target.latitude
                ))

        return tasks

    def _evaluate_assignment(self, assignment: List[int]) -> float:
        """
        评估任务分配方案的适应度

        Args:
            assignment: 每个任务分配的卫星索引，-1表示未分配

        Returns:
            float: 适应度分数
        """
        score = 0.0
        scheduled_count = 0
        target_obs_count: Dict[str, int] = {}  # 每个目标的实际观测次数
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {i: [] for i in range(len(self.satellites))}

        for task_idx, sat_idx in enumerate(assignment):
            if sat_idx < 0:
                continue

            task = self.observation_tasks[task_idx]
            sat = self.satellites[sat_idx]

            # 检查是否有可见窗口
            windows = self.window_cache.get_windows(sat.id, task.target_id)
            if not windows:
                continue

            # 找到时间可行的窗口
            feasible_window = None
            for window in windows:
                # 确保同一目标的不同观测之间有时间间隔（至少60秒）
                min_gap = timedelta(seconds=60)
                conflict = False

                for start, end in sat_task_times[sat_idx]:
                    if not (window.end_time + min_gap <= start or window.start_time >= end + min_gap):
                        conflict = True
                        break

                if not conflict:
                    feasible_window = window
                    break

            if feasible_window:
                scheduled_count += 1
                target_obs_count[task.target_id] = target_obs_count.get(task.target_id, 0) + 1
                sat_task_times[sat_idx].append((feasible_window.start_time, feasible_window.end_time))

                # 基础分数
                score += 10.0

                # 优先级奖励
                score += task.priority * 2.0

                # 观测频次满足度奖励
                if task.required_observations == -1:
                    # 不限频次，每次观测都给予奖励
                    score += 5.0
                else:
                    # 指定频次，完成指定次数给予更高奖励
                    current = target_obs_count[task.target_id]
                    if current <= task.required_observations:
                        score += 8.0
                    else:
                        # 超过需求次数，给予较少奖励
                        score += 1.0

        # 目标观测完成度奖励
        for target in self.targets:
            required = getattr(target, 'required_observations', 1)
            actual = target_obs_count.get(target.id, 0)

            if required == -1:
                # 不限频次，鼓励多观测
                score += actual * 2.0
            else:
                # 指定频次，满足需求给予奖励
                if actual >= required:
                    score += 20.0  # 完成奖励
                    score += min(actual - required, 2) * 2.0  # 少量超额奖励
                else:
                    score += (actual / required) * 15.0  # 部分完成按比例奖励

        return score

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual: List[int]) -> List[int]:
        """变异操作"""
        result = individual[:]
        for i in range(len(result)):
            if random.random() < self.mutation_rate:
                # 随机分配给另一个卫星或取消分配
                if random.random() < 0.1:
                    result[i] = -1  # 取消分配
                else:
                    result[i] = random.randint(0, len(self.satellites) - 1)
        return result

    def schedule(self) -> Tuple[ScheduleResult, Dict[str, int]]:
        """执行遗传算法调度"""
        import time
        start_time = time.time()

        print(f"创建 {len(self.observation_tasks)} 个观测任务（考虑频次需求）")

        if len(self.observation_tasks) == 0:
            return ScheduleResult(
                scheduled_tasks=[],
                unscheduled_tasks={},
                makespan=0,
                computation_time=0,
                iterations=0,
                convergence_curve=[]
            ), {}

        # 初始化种群
        population = []
        for _ in range(self.population_size):
            individual = [random.randint(-1, len(self.satellites) - 1)
                         for _ in range(len(self.observation_tasks))]
            population.append(individual)

        # 评估初始种群
        fitness_scores = [self._evaluate_assignment(ind) for ind in population]

        # 记录收敛曲线
        best_fitness = max(fitness_scores)
        self.convergence_curve = [best_fitness]

        # 进化循环
        for generation in range(self.generations):
            # 选择（锦标赛）
            selected = []
            for _ in range(self.population_size):
                tournament = random.sample(range(self.population_size), min(3, self.population_size))
                winner = max(tournament, key=lambda i: fitness_scores[i])
                selected.append(population[winner][:])

            # 交叉
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    c1, c2 = self._crossover(selected[i], selected[i + 1])
                    offspring.extend([c1, c2])
                else:
                    offspring.append(selected[i])

            # 变异
            offspring = [self._mutate(ind) for ind in offspring]

            # 评估
            offspring_fitness = [self._evaluate_assignment(ind) for ind in offspring]

            # 精英保留
            combined = list(zip(population + offspring, fitness_scores + offspring_fitness))
            combined.sort(key=lambda x: x[1], reverse=True)

            population = [ind for ind, _ in combined[:self.population_size]]
            fitness_scores = [score for _, score in combined[:self.population_size]]

            # 记录最优
            best_fitness = fitness_scores[0]
            self.convergence_curve.append(best_fitness)

        # 获取最优解
        best_individual = population[0]

        # 解码结果
        scheduled_tasks, target_obs_count = self._decode_solution(best_individual)

        computation_time = time.time() - start_time
        makespan = self._calculate_makespan(scheduled_tasks)

        result = ScheduleResult(
            scheduled_tasks=scheduled_tasks,
            unscheduled_tasks={},
            makespan=makespan,
            computation_time=computation_time,
            iterations=self.generations,
            convergence_curve=self.convergence_curve
        )

        return result, target_obs_count

    def _decode_solution(self, assignment: List[int]) -> Tuple[List[ScheduledTask], Dict[str, int]]:
        """将分配方案解码为调度任务"""
        scheduled_tasks = []
        target_obs_count: Dict[str, int] = {}
        sat_task_times: Dict[int, List[Tuple[datetime, datetime]]] = {i: [] for i in range(len(self.satellites))}

        for task_idx, sat_idx in enumerate(assignment):
            if sat_idx < 0:
                continue

            task = self.observation_tasks[task_idx]
            sat = self.satellites[sat_idx]

            windows = self.window_cache.get_windows(sat.id, task.target_id)
            if not windows:
                continue

            # 找到可行窗口
            feasible_window = None
            for window in windows:
                min_gap = timedelta(seconds=60)
                conflict = False

                for start, end in sat_task_times[sat_idx]:
                    if not (window.end_time + min_gap <= start or window.start_time >= end + min_gap):
                        conflict = True
                        break

                if not conflict:
                    feasible_window = window
                    break

            if feasible_window:
                target_obs_count[task.target_id] = target_obs_count.get(task.target_id, 0) + 1

                scheduled_tasks.append(ScheduledTask(
                    task_id=task.task_id,
                    satellite_id=sat.id,
                    target_id=task.target_id,
                    imaging_start=feasible_window.start_time,
                    imaging_end=feasible_window.end_time,
                    imaging_mode='push_broom',
                    slew_angle=0.0
                ))

                sat_task_times[sat_idx].append((feasible_window.start_time, feasible_window.end_time))

        return scheduled_tasks, target_obs_count

    def _calculate_makespan(self, scheduled_tasks: List[ScheduledTask]) -> float:
        """计算完成时间"""
        if not scheduled_tasks:
            return 0.0
        end_times = [t.imaging_end for t in scheduled_tasks]
        start_times = [t.imaging_start for t in scheduled_tasks]
        return (max(end_times) - min(start_times)).total_seconds() if end_times and start_times else 0.0


def load_window_cache_from_json(cache_path: str, mission: Mission) -> VisibilityWindowCache:
    """从JSON文件加载预计算的窗口缓存"""
    from datetime import datetime

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache = VisibilityWindowCache()

    for w_data in data.get('target_windows', []):
        sat_id = w_data['satellite_id']
        target_id = w_data['target_id']

        window = VisibilityWindow(
            satellite_id=sat_id,
            target_id=target_id,
            start_time=datetime.fromisoformat(w_data['start_time']),
            end_time=datetime.fromisoformat(w_data['end_time']),
            max_elevation=w_data.get('max_elevation', 0.0)
        )

        key = (sat_id, target_id)
        if key not in cache._windows:
            cache._windows[key] = []
            cache._time_index[key] = []

        cache._windows[key].append(window)
        cache._time_index[key].append(window.start_time)

        if sat_id not in cache._sat_to_targets:
            cache._sat_to_targets[sat_id] = set()
        cache._sat_to_targets[sat_id].add(target_id)

        if target_id not in cache._target_to_sats:
            cache._target_to_sats[target_id] = set()
        cache._target_to_sats[target_id].add(sat_id)

    # 对所有窗口排序
    for key in cache._windows:
        sorted_pairs = sorted(zip(cache._time_index[key], cache._windows[key]))
        cache._time_index[key] = [p[0] for p in sorted_pairs]
        cache._windows[key] = [p[1] for p in sorted_pairs]

    return cache


def main():
    parser = argparse.ArgumentParser(description='支持观测频次的调度')
    parser.add_argument('--cache', '-c', required=True, help='缓存文件路径')
    parser.add_argument('--scenario', '-s', required=True, help='场景文件路径')
    parser.add_argument('--generations', type=int, default=100, help='迭代次数')
    parser.add_argument('--population-size', type=int, default=50, help='种群大小')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print("=" * 60)
    print("支持观测频次需求的调度")
    print("=" * 60)

    # 加载场景
    print(f"\n[1/4] 加载场景: {args.scenario}")
    mission = Mission.load(args.scenario)
    print(f"  卫星: {len(mission.satellites)} 颗")
    print(f"  目标: {len(mission.targets)} 个")

    # 显示目标观测需求
    print("\n  目标观测需求:")
    total_required = 0
    for target in mission.targets:
        required = getattr(target, 'required_observations', 1)
        freq_str = "不限" if required == -1 else f"{required}次"
        print(f"    {target.id}: {target.name} - 需要{freq_str}")
        total_required += required if required > 0 else 10  # 不限按10次估算
    print(f"  总观测需求: {total_required}次（估算）")

    # 加载缓存
    print(f"\n[2/4] 加载缓存: {args.cache}")
    cache = load_window_cache_from_json(args.cache, mission)
    stats = cache.get_statistics()
    print(f"  可用窗口数: {stats['total_windows']}")

    # 运行调度
    print(f"\n[3/4] 运行调度")
    scheduler = FrequencyAwareScheduler(
        mission=mission,
        window_cache=cache,
        config={
            'generations': args.generations,
            'population_size': args.population_size,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'elitism': 5
        }
    )

    result, target_obs_count = scheduler.schedule()

    # 计算指标
    print(f"\n[4/4] 计算性能指标")
    metrics_calc = MetricsCalculator(mission)
    metrics = metrics_calc.calculate_all(result)

    # 输出结果
    print("\n" + "=" * 60)
    print("调度结果")
    print("=" * 60)
    print(f"总调度任务: {len(result.scheduled_tasks)}")
    print(f"求解用时: {result.computation_time:.2f} 秒")
    print(f"完成时间跨度: {metrics.makespan/3600:.2f} 小时")
    print(f"需求满足率: {metrics.demand_satisfaction_rate:.2%}")
    print("-" * 60)

    # 各目标观测完成情况
    print("\n各目标观测完成情况:")
    for target in mission.targets:
        required = getattr(target, 'required_observations', 1)
        actual = target_obs_count.get(target.id, 0)

        if required == -1:
            status = f"{actual}次（不限，越多越好）"
        else:
            ratio = actual / required if required > 0 else 1.0
            if actual >= required:
                status = f"✓ {actual}/{required}次 ({ratio:.0%})"
            else:
                status = f"✗ {actual}/{required}次 ({ratio:.0%})"

        print(f"  {target.id}: {status}")

    print("=" * 60)

    # 保存结果
    if args.output:
        result_data = {
            'algorithm': 'frequency_aware_ga',
            'scenario': mission.name,
            'cache_file': args.cache,
            'scheduled_count': len(result.scheduled_tasks),
            'computation_time': result.computation_time,
            'target_observations': {
                t.id: {
                    'required': getattr(t, 'required_observations', 1),
                    'actual': target_obs_count.get(t.id, 0)
                }
                for t in mission.targets
            },
            'scheduled_tasks': [
                {
                    'task_id': t.task_id,
                    'satellite_id': t.satellite_id,
                    'target_id': t.target_id,
                    'start_time': t.imaging_start.isoformat(),
                    'end_time': t.imaging_end.isoformat(),
                }
                for t in result.scheduled_tasks[:200]  # 只保存前200个
            ]
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
