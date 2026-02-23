"""
实验运行器

实现第4章设计：
- 批量对比实验
- 参数敏感性分析
- 结果记录和导出
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type, Tuple
from datetime import datetime
import json
import os
import statistics
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    实验配置

    Attributes:
        repetitions: 每个算法的重复实验次数
        random_seed: 随机种子（用于可重复性）
        output_dir: 结果输出目录
        metrics_to_collect: 要收集的指标列表
    """
    repetitions: int = 10
    random_seed: Optional[int] = None
    output_dir: str = "./results"
    metrics_to_collect: List[str] = field(default_factory=lambda: [
        'demand_satisfaction_rate',
        'makespan',
        'computation_time',
        'solution_quality'
    ])


@dataclass
class ExperimentResult:
    """
    单次实验结果

    Attributes:
        algorithm_name: 算法名称
        repetition: 重复次数编号
        metrics: 性能指标字典
        scheduled_tasks: 调度任务列表
        computation_time: 计算时间
        timestamp: 实验时间戳
    """
    algorithm_name: str
    repetition: int
    metrics: Dict[str, float]
    scheduled_tasks: List[Any]
    computation_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'algorithm_name': self.algorithm_name,
            'repetition': self.repetition,
            'metrics': self.metrics,
            'computation_time': self.computation_time,
            'timestamp': self.timestamp.isoformat(),
            'scheduled_count': len(self.scheduled_tasks)
        }


class ExperimentRunner:
    """
    实验运行器

    功能：
    1. 批量运行多个算法的对比实验
    2. 支持重复实验以获得统计显著性
    3. 参数敏感性分析
    4. 结果导出为JSON/CSV
    5. 窗口缓存预计算和注入
    """

    def __init__(
        self,
        mission: Any,
        algorithms: Dict[str, Type],
        config: ExperimentConfig = None
    ):
        """
        初始化实验运行器

        Args:
            mission: 任务场景
            algorithms: {算法名: 调度器类}
            config: 实验配置
        """
        self.mission = mission
        self.algorithms = algorithms
        self.config = config or ExperimentConfig()
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.window_cache: Optional[Any] = None  # 窗口缓存

    def precompute_window_cache(
        self,
        calculator: Any,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Any:
        """
        预计算所有可见性窗口并缓存

        在运行实验前调用此方法，预计算所有卫星-目标对的可见性窗口，
        避免每个调度器重复计算。

        Args:
            calculator: 可见性计算器（如OrekitVisibilityCalculator）
            start_time: 预计算开始时间（默认使用mission.start_time）
            end_time: 预计算结束时间（默认使用mission.end_time）

        Returns:
            VisibilityWindowCache: 窗口缓存实例
        """
        from core.orbit.visibility.window_cache import VisibilityWindowCache

        # 确定时间范围
        if start_time is None:
            start_time = getattr(self.mission, 'start_time', datetime.now())
        if end_time is None:
            end_time = getattr(self.mission, 'end_time', datetime.now())

        # 获取地面站列表
        ground_stations = getattr(self.mission, 'ground_stations', [])

        # 创建缓存并预计算
        self.window_cache = VisibilityWindowCache()
        self.window_cache.precompute_all_windows(
            satellites=self.mission.satellites,
            targets=self.mission.targets,
            ground_stations=ground_stations,
            start_time=start_time,
            end_time=end_time,
            calculator=calculator
        )

        return self.window_cache

    def _inject_window_cache(self, scheduler: Any) -> None:
        """
        向调度器注入窗口缓存

        Args:
            scheduler: 调度器实例
        """
        if self.window_cache is not None and hasattr(scheduler, 'set_window_cache'):
            scheduler.set_window_cache(self.window_cache)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取窗口缓存统计信息

        Returns:
            缓存统计信息字典，如果未预计算则返回空字典
        """
        if self.window_cache is None:
            return {}

        if hasattr(self.window_cache, 'get_statistics'):
            return self.window_cache.get_statistics()

        return {}

    def clear_window_cache(self) -> None:
        """清除窗口缓存"""
        if self.window_cache is not None:
            if hasattr(self.window_cache, 'clear'):
                self.window_cache.clear()
            self.window_cache = None

    def run_single_experiment(
        self,
        algorithm_name: str,
        algorithm_params: Dict[str, Any]
    ) -> ExperimentResult:
        """
        运行单次实验

        Args:
            algorithm_name: 算法名称
            algorithm_params: 算法参数

        Returns:
            ExperimentResult: 实验结果
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        scheduler_class = self.algorithms[algorithm_name]

        # 实例化调度器
        scheduler = scheduler_class(algorithm_params)
        scheduler.initialize(self.mission)

        # 注入窗口缓存（如果已预计算）
        self._inject_window_cache(scheduler)

        # 运行调度
        start_time = datetime.now()
        schedule_result = scheduler.schedule()
        end_time = datetime.now()

        # 计算性能指标
        from evaluation.metrics import MetricsCalculator
        metrics_calc = MetricsCalculator(self.mission)
        metrics = metrics_calc.calculate_all(schedule_result)

        # 构建实验结果
        result = ExperimentResult(
            algorithm_name=algorithm_name,
            repetition=1,  # 在run_all中更新
            metrics=metrics.to_dict(),
            scheduled_tasks=schedule_result.scheduled_tasks,
            computation_time=schedule_result.computation_time
        )

        return result

    def run_all(self) -> Dict[str, List[ExperimentResult]]:
        """
        运行所有算法的实验

        Returns:
            Dict[str, List[ExperimentResult]]: {算法名: [结果列表]}
        """
        self.results = {}

        for alg_name in self.algorithms.keys():
            self.results[alg_name] = []

            for rep in range(1, self.config.repetitions + 1):
                print(f"Running {alg_name} - Repetition {rep}/{self.config.repetitions}")

                try:
                    result = self.run_single_experiment(alg_name, {})
                    result.repetition = rep
                    self.results[alg_name].append(result)
                except Exception as e:
                    print(f"Error running {alg_name} rep {rep}: {e}")
                    # 继续下一次重复

        return self.results

    def compare_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        对比所有算法的结果

        Returns:
            Dict: {算法名: {统计指标}}
        """
        if not self.results:
            raise ValueError("No results available. Run experiments first.")

        comparison = {}

        for alg_name, results in self.results.items():
            if not results:
                continue

            # 收集各项指标
            metrics_by_name: Dict[str, List[float]] = {}
            for result in results:
                for metric_name, value in result.metrics.items():
                    if metric_name not in metrics_by_name:
                        metrics_by_name[metric_name] = []
                    if isinstance(value, (int, float)):
                        metrics_by_name[metric_name].append(value)

            # 计算统计量
            stats = {}
            for metric_name, values in metrics_by_name.items():
                if values:
                    stats[f'mean_{metric_name}'] = statistics.mean(values)
                    stats[f'std_{metric_name}'] = statistics.stdev(values) if len(values) > 1 else 0.0
                    stats[f'min_{metric_name}'] = min(values)
                    stats[f'max_{metric_name}'] = max(values)

            comparison[alg_name] = stats

        return comparison

    def run_sensitivity_analysis(
        self,
        algorithm_name: str,
        param_ranges: Dict[str, List[Any]]
    ) -> List[Tuple[Dict[str, Any], ExperimentResult]]:
        """
        运行参数敏感性分析

        Args:
            algorithm_name: 算法名称
            param_ranges: {参数名: [参数值列表]}

        Returns:
            List[(参数组合, 结果)]: 所有参数组合的结果
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        results = []

        # 生成所有参数组合
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        def generate_combinations(idx: int, current: Dict[str, Any]):
            if idx == len(param_names):
                # 运行实验
                try:
                    result = self.run_single_experiment(algorithm_name, current.copy())
                    results.append((current.copy(), result))
                except Exception as e:
                    print(f"Error with params {current}: {e}")
                return

            for value in param_values[idx]:
                current[param_names[idx]] = value
                generate_combinations(idx + 1, current)

        generate_combinations(0, {})

        return results

    def export_results(self, output_path: str = None) -> str:
        """
        导出实验结果到JSON文件

        Args:
            output_path: 输出文件路径（默认使用config.output_dir）

        Returns:
            str: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                'experiment_results.json'
            )

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 准备导出数据
        export_data = {
            'experiment_config': {
                'repetitions': self.config.repetitions,
                'random_seed': self.config.random_seed,
                'metrics_to_collect': self.config.metrics_to_collect,
            },
            'mission': {
                'name': self.mission.name,
                'satellite_count': len(self.mission.satellites),
                'target_count': len(self.mission.targets),
            },
            'algorithms': list(self.algorithms.keys()),
            'results': {},
            'timestamp': datetime.now().isoformat()
        }

        # 添加实验结果
        for alg_name, results in self.results.items():
            export_data['results'][alg_name] = [
                result.to_dict() for result in results
            ]

        # 添加对比统计
        try:
            export_data['comparison'] = self.compare_algorithms()
        except ValueError:
            pass  # 没有结果时跳过

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to: {output_path}")
        return output_path

    def generate_report(self) -> str:
        """
        生成实验报告（Markdown格式）

        Returns:
            str: Markdown格式的报告
        """
        if not self.results:
            return "No results available."

        lines = []
        lines.append("# 实验报告\n")
        lines.append(f"**场景**: {self.mission.name}\n")
        lines.append(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**重复次数**: {self.config.repetitions}\n\n")

        # 算法对比表
        lines.append("## 算法对比\n\n")
        lines.append("| 算法 | 需求满足率 | 总完成时间 | 计算时间 |\n")
        lines.append("|------|-----------|-----------|---------|\n")

        comparison = self.compare_algorithms()
        for alg_name, stats in comparison.items():
            dsr = stats.get('mean_demand_satisfaction_rate', 0) * 100
            makespan = stats.get('mean_makespan_hours', 0)
            comp_time = stats.get('mean_computation_time_seconds', 0)
            lines.append(f"| {alg_name} | {dsr:.1f}% | {makespan:.2f}h | {comp_time:.2f}s |\n")

        lines.append("\n## 详细结果\n\n")

        for alg_name, results in self.results.items():
            lines.append(f"### {alg_name}\n\n")
            lines.append(f"- 成功运行: {len(results)}/{self.config.repetitions}\n")

            if results:
                avg_dsr = statistics.mean(
                    r.metrics.get('demand_satisfaction_rate', 0) for r in results
                )
                lines.append(f"- 平均需求满足率: {avg_dsr*100:.2f}%\n")

            lines.append("\n")

        return ''.join(lines)
