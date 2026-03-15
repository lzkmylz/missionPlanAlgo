"""
调度器Benchmark框架

提供统一的调度器性能和质量评估接口，支持多算法对比。
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import statistics

from core.models.mission import Mission
from core.orbit.visibility.base import VisibilityWindow
from scheduler.base_scheduler import ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Benchmark结果

    包含调度器性能和质量评估的完整指标
    """
    # 基础信息
    algorithm_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 性能指标
    computation_time: float = 0.0  # 计算时间（秒）
    scheduled_tasks: int = 0  # 已调度任务数
    unscheduled_tasks: int = 0  # 未调度任务数
    total_tasks: int = 0  # 总任务数

    # 质量指标
    avg_window_quality: float = 0.0  # 平均窗口质量
    quality_distribution: Dict[str, int] = field(default_factory=dict)  # {'high': 100, 'medium': 50, ...}
    low_quality_task_ratio: float = 0.0  # 低质量任务占比

    # 效率指标
    makespan_hours: float = 0.0  # 完成时间跨度（小时）
    satellite_utilization: float = 0.0  # 卫星利用率
    frequency_satisfaction_rate: float = 0.0  # 频次满足率

    # 约束满足情况
    constraint_violations: int = 0  # 约束违反次数
    slew_timeouts: int = 0  # 机动超时次数

    # 原始数据（可选）
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """从字典创建"""
        return cls(**data)

    def get_success_rate(self) -> float:
        """获取任务成功率"""
        if self.total_tasks == 0:
            return 0.0
        return self.scheduled_tasks / self.total_tasks

    def get_efficiency_score(self) -> float:
        """
        获取综合效率评分

        综合考虑成功率、质量和利用率
        """
        success_weight = 0.5
        quality_weight = 0.3
        utilization_weight = 0.2

        return (
            self.get_success_rate() * success_weight +
            self.avg_window_quality * quality_weight +
            self.satellite_utilization * utilization_weight
        )


@dataclass
class ComparisonReport:
    """
    算法对比报告

    对比多个调度算法与baseline的性能差异
    """
    baseline: str
    candidates: List[str]
    results: Dict[str, BenchmarkResult]

    # 对比指标
    improvements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'baseline': self.baseline,
            'candidates': self.candidates,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'improvements': self.improvements,
            'statistical_significance': self.statistical_significance,
        }


class SchedulerBenchmark:
    """
    调度器Benchmark框架

    提供统一的调度器性能和质量评估功能
    """

    # 支持的算法
    SUPPORTED_ALGORITHMS = [
        'greedy',
        'ga',  # Genetic Algorithm
        'sa',  # Simulated Annealing
        'aco',  # Ant Colony Optimization
        'pso',  # Particle Swarm Optimization
        'tabu',  # Tabu Search
        'edd',  # Earliest Due Date
    ]

    def __init__(
        self,
        mission: Mission,
        window_cache: Optional[Dict[str, List[VisibilityWindow]]] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化Benchmark框架

        Args:
            mission: 任务对象
            window_cache: 可见窗口缓存 {sat_id: [windows, ...]}
            output_dir: 输出目录
        """
        self.mission = mission
        self.window_cache = window_cache or {}
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BenchmarkResult] = []
        self._scheduler_cache: Dict[str, Any] = {}

    def run_benchmark(
        self,
        algorithms: List[str],
        config: Optional[Dict[str, Any]] = None,
        iterations: int = 1,
        warmup: bool = True,
    ) -> List[BenchmarkResult]:
        """
        运行多算法benchmark

        Args:
            algorithms: 算法列表 ['greedy', 'ga', 'sa', ...]
            config: 通用配置
            iterations: 每个算法运行次数（用于统计稳定性）
            warmup: 是否进行预热运行

        Returns:
            BenchmarkResult列表
        """
        config = config or {}
        results = []

        logger.info(f"Starting benchmark with {len(algorithms)} algorithms, {iterations} iterations each")

        for algo in algorithms:
            if algo not in self.SUPPORTED_ALGORITHMS:
                logger.warning(f"Unknown algorithm: {algo}, skipping")
                continue

            logger.info(f"Benchmarking algorithm: {algo}")

            # 预热运行（可选）
            if warmup and iterations > 1:
                logger.debug(f"Warmup run for {algo}")
                self._run_single_algorithm(algo, config)

            # 正式运行
            iteration_results = []
            for i in range(iterations):
                logger.debug(f"Iteration {i+1}/{iterations} for {algo}")
                result = self._run_single_algorithm(algo, config)
                if result:
                    iteration_results.append(result)

            # 如果有多次运行，计算平均值
            if iteration_results:
                if iterations > 1:
                    avg_result = self._average_results(algo, iteration_results)
                else:
                    avg_result = iteration_results[0]

                results.append(avg_result)
                self.results.append(avg_result)

        logger.info(f"Benchmark completed, collected {len(results)} results")
        return results

    def _run_single_algorithm(
        self,
        algorithm: str,
        config: Dict[str, Any],
    ) -> Optional[BenchmarkResult]:
        """
        运行单个算法benchmark

        Args:
            algorithm: 算法名称
            config: 配置

        Returns:
            BenchmarkResult或None（如果失败）
        """
        try:
            # 获取或创建调度器
            scheduler = self._get_scheduler(algorithm, config)
            if scheduler is None:
                logger.error(f"Failed to create scheduler for {algorithm}")
                return None

            # 记录开始时间
            start_time = time.perf_counter()

            # 运行调度
            result = scheduler.schedule(self.mission)

            # 计算耗时
            computation_time = time.perf_counter() - start_time

            # 分析质量
            quality_metrics = self._analyze_quality(result)

            # 构建BenchmarkResult
            benchmark_result = BenchmarkResult(
                algorithm_name=algorithm,
                computation_time=computation_time,
                scheduled_tasks=quality_metrics.get('scheduled_count', 0),
                unscheduled_tasks=quality_metrics.get('unscheduled_count', 0),
                total_tasks=quality_metrics.get('total_count', 0),
                avg_window_quality=quality_metrics.get('avg_quality', 0.0),
                quality_distribution=quality_metrics.get('quality_distribution', {}),
                low_quality_task_ratio=quality_metrics.get('low_quality_ratio', 0.0),
                makespan_hours=quality_metrics.get('makespan_hours', 0.0),
                satellite_utilization=quality_metrics.get('utilization', 0.0),
                frequency_satisfaction_rate=quality_metrics.get('frequency_satisfaction', 0.0),
                constraint_violations=quality_metrics.get('violations', 0),
                raw_metrics=quality_metrics,
            )

            return benchmark_result

        except Exception as e:
            logger.error(f"Error running benchmark for {algorithm}: {e}")
            return None

    def _get_scheduler(self, algorithm: str, config: Dict[str, Any]):
        """获取或创建调度器"""
        # 检查缓存
        cache_key = f"{algorithm}:{hash(str(config))}"
        if cache_key in self._scheduler_cache:
            return self._scheduler_cache[cache_key]

        # 创建调度器
        scheduler = None

        if algorithm == 'greedy':
            from scheduler.greedy.greedy_scheduler import GreedyScheduler
            scheduler = GreedyScheduler(config)
        elif algorithm == 'ga':
            from scheduler.metaheuristic.ga_scheduler import GAScheduler
            scheduler = GAScheduler(config)
        elif algorithm == 'sa':
            from scheduler.metaheuristic.sa_scheduler import SAScheduler
            scheduler = SAScheduler(config)
        elif algorithm == 'aco':
            from scheduler.metaheuristic.aco_scheduler import ACOScheduler
            scheduler = ACOScheduler(config)
        elif algorithm == 'pso':
            from scheduler.metaheuristic.pso_scheduler import PSOScheduler
            scheduler = PSOScheduler(config)
        elif algorithm == 'tabu':
            from scheduler.metaheuristic.tabu_scheduler import TabuScheduler
            scheduler = TabuScheduler(config)
        elif algorithm == 'edd':
            from scheduler.EDD.edd_scheduler import EDDScheduler
            scheduler = EDDScheduler(config)

        if scheduler:
            self._scheduler_cache[cache_key] = scheduler

        return scheduler

    def _analyze_quality(self, result: ScheduleResult) -> Dict[str, Any]:
        """分析调度结果质量"""
        metrics = {
            'scheduled_count': 0,
            'unscheduled_count': 0,
            'total_count': 0,
            'avg_quality': 0.0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0, 'unacceptable': 0},
            'low_quality_ratio': 0.0,
            'makespan_hours': 0.0,
            'utilization': 0.0,
            'frequency_satisfaction': 0.0,
            'violations': 0,
        }

        if not result:
            return metrics

        # 获取成像结果
        imaging_result = getattr(result, 'imaging_result', None)
        if not imaging_result:
            return metrics

        # 任务统计
        scheduled = getattr(imaging_result, 'scheduled_tasks', [])
        unscheduled = getattr(imaging_result, 'unscheduled_tasks', [])
        metrics['scheduled_count'] = len(scheduled)
        metrics['unscheduled_count'] = len(unscheduled)
        metrics['total_count'] = len(scheduled) + len(unscheduled)

        # 质量统计
        if scheduled:
            qualities = []
            for task in scheduled:
                quality = getattr(task, 'quality_score', 0.5)
                qualities.append(quality)

                # 质量分布
                if quality >= 0.7:
                    metrics['quality_distribution']['high'] += 1
                elif quality >= 0.4:
                    metrics['quality_distribution']['medium'] += 1
                elif quality >= 0.3:
                    metrics['quality_distribution']['low'] += 1
                else:
                    metrics['quality_distribution']['unacceptable'] += 1

            metrics['avg_quality'] = sum(qualities) / len(qualities)
            metrics['low_quality_ratio'] = sum(1 for q in qualities if q < 0.3) / len(qualities)

            # 完成时间跨度
            if hasattr(imaging_result, 'completion_time') and imaging_result.completion_time:
                start = self.mission.start_time if self.mission else scheduled[0].start_time
                makespan = (imaging_result.completion_time - start).total_seconds() / 3600
                metrics['makespan_hours'] = makespan

            # 卫星利用率
            if hasattr(imaging_result, 'satellite_utilization'):
                metrics['utilization'] = imaging_result.satellite_utilization

            # 频次满足率
            if hasattr(imaging_result, 'frequency_satisfaction'):
                freq_sat = imaging_result.frequency_satisfaction
                if isinstance(freq_sat, dict):
                    metrics['frequency_satisfaction'] = freq_sat.get('satisfaction_rate', 0.0)
                else:
                    metrics['frequency_satisfaction'] = float(freq_sat)

        return metrics

    def _average_results(self, algorithm: str, results: List[BenchmarkResult]) -> BenchmarkResult:
        """计算多次运行的平均结果"""
        if not results:
            return BenchmarkResult(algorithm_name=algorithm)

        n = len(results)

        # 计算平均值
        avg_computation_time = statistics.mean([r.computation_time for r in results])
        avg_quality = statistics.mean([r.avg_window_quality for r in results])
        avg_utilization = statistics.mean([r.satellite_utilization for r in results])

        # 使用第一次运行的任务计数（应该相同）
        first = results[0]

        return BenchmarkResult(
            algorithm_name=algorithm,
            computation_time=avg_computation_time,
            scheduled_tasks=first.scheduled_tasks,
            unscheduled_tasks=first.unscheduled_tasks,
            total_tasks=first.total_tasks,
            avg_window_quality=avg_quality,
            quality_distribution=first.quality_distribution,
            low_quality_task_ratio=first.low_quality_task_ratio,
            makespan_hours=first.makespan_hours,
            satellite_utilization=avg_utilization,
            frequency_satisfaction_rate=first.frequency_satisfaction_rate,
            constraint_violations=first.constraint_violations,
        )

    def compare_algorithms(
        self,
        baseline: str,
        candidates: List[str],
        metrics: Optional[List[str]] = None,
    ) -> ComparisonReport:
        """
        对比多个算法与baseline

        Args:
            baseline: baseline算法名称
            candidates: 候选算法列表
            metrics: 要对比的指标列表

        Returns:
            ComparisonReport
        """
        metrics = metrics or ['avg_window_quality', 'computation_time', 'scheduled_tasks']

        # 收集结果
        results = {}
        for result in self.results:
            results[result.algorithm_name] = result

        # 确保baseline存在
        if baseline not in results:
            raise ValueError(f"Baseline algorithm '{baseline}' not found in results")

        baseline_result = results[baseline]

        # 计算改进幅度
        improvements = {}
        for candidate in candidates:
            if candidate not in results:
                continue

            candidate_result = results[candidate]
            improvements[candidate] = {}

            for metric in metrics:
                baseline_val = getattr(baseline_result, metric, 0)
                candidate_val = getattr(candidate_result, metric, 0)

                if baseline_val != 0:
                    improvement = (candidate_val - baseline_val) / baseline_val
                else:
                    improvement = 0.0 if candidate_val == 0 else float('inf')

                improvements[candidate][metric] = improvement

        return ComparisonReport(
            baseline=baseline,
            candidates=candidates,
            results=results,
            improvements=improvements,
        )

    def generate_report(
        self,
        output_path: Optional[str] = None,
        format: str = 'markdown',
        include_charts: bool = False,
    ) -> str:
        """
        生成benchmark报告

        Args:
            output_path: 输出文件路径（None则自动生成）
            format: 报告格式 ('markdown' | 'json' | 'html')
            include_charts: 是否包含图表

        Returns:
            报告文件路径
        """
        if not self.results:
            logger.warning("No benchmark results to report")
            return ""

        # 生成输出路径
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"benchmark_report_{timestamp}.{format}"
        else:
            output_path = Path(output_path)

        # 生成报告内容
        if format == 'json':
            content = self._generate_json_report()
        elif format == 'html':
            content = self._generate_html_report(include_charts)
        else:  # markdown
            content = self._generate_markdown_report()

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Benchmark report saved to: {output_path}")
        return str(output_path)

    def _generate_json_report(self) -> str:
        """生成JSON格式报告"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'mission': str(self.mission) if self.mission else None,
            'results': [r.to_dict() for r in self.results],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        lines = []

        lines.append("# 调度器Benchmark报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n任务: {self.mission.name if self.mission else 'Unknown'}")

        # 摘要表
        lines.append("\n## 性能摘要\n")
        lines.append("| 算法 | 计算时间(s) | 调度任务数 | 成功率 | 平均质量 | 利用率 |")
        lines.append("|------|------------|-----------|--------|----------|--------|")

        for result in self.results:
            success_rate = result.get_success_rate() * 100
            lines.append(
                f"| {result.algorithm_name} | "
                f"{result.computation_time:.2f} | "
                f"{result.scheduled_tasks}/{result.total_tasks} | "
                f"{success_rate:.1f}% | "
                f"{result.avg_window_quality:.3f} | "
                f"{result.satellite_utilization:.3f} |"
            )

        # 质量分布
        lines.append("\n## 质量分布\n")
        lines.append("| 算法 | 高质量 | 中等 | 低质量 | 不可接受 |")
        lines.append("|------|--------|------|--------|----------|")

        for result in self.results:
            dist = result.quality_distribution
            lines.append(
                f"| {result.algorithm_name} | "
                f"{dist.get('high', 0)} | "
                f"{dist.get('medium', 0)} | "
                f"{dist.get('low', 0)} | "
                f"{dist.get('unacceptable', 0)} |"
            )

        # 详细结果
        lines.append("\n## 详细结果\n")
        for result in self.results:
            lines.append(f"\n### {result.algorithm_name}\n")
            lines.append(f"- 计算时间: {result.computation_time:.3f}s")
            lines.append(f"- 调度任务: {result.scheduled_tasks}/{result.total_tasks}")
            lines.append(f"- 成功率: {result.get_success_rate()*100:.1f}%")
            lines.append(f"- 平均窗口质量: {result.avg_window_quality:.3f}")
            lines.append(f"- 低质量任务占比: {result.low_quality_task_ratio*100:.1f}%")
            lines.append(f"- 卫星利用率: {result.satellite_utilization*100:.1f}%")
            lines.append(f"- 频次满足率: {result.frequency_satisfaction_rate*100:.1f}%")

            if result.makespan_hours > 0:
                lines.append(f"- 完成时间跨度: {result.makespan_hours:.2f}小时")

        # 排名
        if len(self.results) > 1:
            lines.append("\n## 算法排名\n")

            # 按综合效率排序
            ranked = sorted(self.results, key=lambda r: r.get_efficiency_score(), reverse=True)
            lines.append("### 综合效率排名\n")
            for i, result in enumerate(ranked, 1):
                lines.append(f"{i}. **{result.algorithm_name}**: {result.get_efficiency_score():.3f}")

        return "\n".join(lines)

    def _generate_html_report(self, include_charts: bool = False) -> str:
        """生成HTML格式报告"""
        # 简化版本，可以扩展为更复杂的HTML
        md_content = self._generate_markdown_report()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>调度器Benchmark报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
    {self._markdown_to_html(md_content)}
</body>
</html>"""
        return html

    def _markdown_to_html(self, markdown: str) -> str:
        """简单Markdown转HTML"""
        html = markdown

        # 标题
        html = html.replace("# ", "<h1>").replace("\n", "</h1>\n", 1)
        html = html.replace("## ", "<h2>").replace("\n", "</h2>\n", 1)
        html = html.replace("### ", "<h3>").replace("\n", "</h3>\n", 1)

        # 表格
        lines = html.split("\n")
        in_table = False
        html_lines = []

        for line in lines:
            if line.startswith("|"):
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True

                cells = [c.strip() for c in line.split("|")[1:-1]]
                if "---" in cells[0]:
                    continue  # 跳过分隔行

                tag = "th" if html_lines and "<table>" in html_lines[-1] else "td"
                row = "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
                html_lines.append(row)
            else:
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append(f"<p>{line}</p>" if line else "<br>")

        if in_table:
            html_lines.append("</table>")

        return "\n".join(html_lines)

    def plot_comparison(
        self,
        metrics: List[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        生成对比图表

        Args:
            metrics: 要绘制的指标
            output_path: 输出路径

        Returns:
            图表文件路径或None
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
            return None

        if not self.results:
            return None

        metrics = metrics or ['avg_window_quality', 'computation_time', 'scheduled_tasks']

        # 创建图表
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        algorithms = [r.algorithm_name for r in self.results]

        for idx, metric in enumerate(metrics):
            values = [getattr(r, metric, 0) for r in self.results]

            ax = axes[idx]
            bars = ax.bar(algorithms, values)

            # 根据值设置颜色
            if metric == 'computation_time':
                # 时间越短越好
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
            else:
                # 其他指标越高越好
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))

            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # 保存
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"benchmark_comparison_{timestamp}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plot saved to: {output_path}")
        return str(output_path)

    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        保存所有结果到JSON文件

        Args:
            output_path: 输出路径

        Returns:
            文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        else:
            output_path = Path(output_path)

        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def load_results(self, input_path: str) -> None:
        """
        从JSON文件加载结果

        Args:
            input_path: 输入文件路径
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.results = [BenchmarkResult.from_dict(r) for r in data.get('results', [])]
        logger.info(f"Loaded {len(self.results)} results from {input_path}")
