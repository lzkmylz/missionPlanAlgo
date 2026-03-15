"""
Benchmark框架集成测试

测试SchedulerBenchmark的功能和集成
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from evaluation.benchmark import (
    SchedulerBenchmark,
    BenchmarkResult,
    ComparisonReport,
)
from evaluation.quality_metrics import (
    ScheduleQualityAnalyzer,
    WindowQualityAnalysis,
    QualityMetricsCollector,
)


class TestBenchmarkResult(unittest.TestCase):
    """测试 BenchmarkResult 数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        result = BenchmarkResult(
            algorithm_name='greedy',
            computation_time=1.5,
            scheduled_tasks=100,
            unscheduled_tasks=10,
            total_tasks=110,
            avg_window_quality=0.75,
        )

        self.assertEqual(result.algorithm_name, 'greedy')
        self.assertEqual(result.computation_time, 1.5)
        self.assertEqual(result.get_success_rate(), 100/110)

    def test_success_rate_calculation(self):
        """测试成功率计算"""
        # 全部成功
        result1 = BenchmarkResult(
            algorithm_name='test',
            scheduled_tasks=100,
            total_tasks=100,
        )
        self.assertEqual(result1.get_success_rate(), 1.0)

        # 部分成功
        result2 = BenchmarkResult(
            algorithm_name='test',
            scheduled_tasks=50,
            total_tasks=100,
        )
        self.assertEqual(result2.get_success_rate(), 0.5)

        # 无任务
        result3 = BenchmarkResult(algorithm_name='test')
        self.assertEqual(result3.get_success_rate(), 0.0)

    def test_efficiency_score(self):
        """测试综合效率评分"""
        result = BenchmarkResult(
            algorithm_name='test',
            scheduled_tasks=100,
            total_tasks=100,
            avg_window_quality=0.8,
            satellite_utilization=0.7,
        )

        score = result.get_efficiency_score()
        expected = 1.0 * 0.5 + 0.8 * 0.3 + 0.7 * 0.2
        self.assertAlmostEqual(score, expected, places=2)

    def test_to_from_dict(self):
        """测试字典转换"""
        original = BenchmarkResult(
            algorithm_name='greedy',
            computation_time=1.5,
            scheduled_tasks=100,
            quality_distribution={'high': 50, 'medium': 30, 'low': 20},
        )

        data = original.to_dict()
        restored = BenchmarkResult.from_dict(data)

        self.assertEqual(original.algorithm_name, restored.algorithm_name)
        self.assertEqual(original.scheduled_tasks, restored.scheduled_tasks)
        self.assertEqual(original.quality_distribution, restored.quality_distribution)


class TestSchedulerBenchmark(unittest.TestCase):
    """测试 SchedulerBenchmark 类"""

    def setUp(self):
        """设置测试环境"""
        # 创建模拟Mission
        self.mock_mission = MagicMock()
        self.mock_mission.name = 'Test Mission'
        self.mock_mission.start_time = datetime.now()

        # 创建临时输出目录
        self.temp_dir = tempfile.mkdtemp()

        # 创建benchmark实例
        self.benchmark = SchedulerBenchmark(
            mission=self.mock_mission,
            output_dir=self.temp_dir,
        )

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.benchmark.mission, self.mock_mission)
        self.assertEqual(len(self.benchmark.results), 0)
        self.assertTrue(Path(self.temp_dir).exists())

    def test_supported_algorithms(self):
        """测试支持的算法列表"""
        supported = SchedulerBenchmark.SUPPORTED_ALGORITHMS

        self.assertIn('greedy', supported)
        self.assertIn('ga', supported)
        self.assertIn('sa', supported)
        self.assertIn('aco', supported)

    @patch('scheduler.greedy.greedy_scheduler.GreedyScheduler')
    def test_run_single_algorithm(self, mock_scheduler_class):
        """测试运行单个算法"""
        # 配置mock
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler

        # 创建模拟调度结果
        mock_imaging_result = MagicMock()
        mock_imaging_result.scheduled_tasks = [MagicMock(quality_score=0.8) for _ in range(10)]
        mock_imaging_result.unscheduled_tasks = []
        mock_imaging_result.completion_time = datetime.now() + timedelta(hours=5)

        mock_result = MagicMock()
        mock_result.imaging_result = mock_imaging_result

        mock_scheduler.schedule.return_value = mock_result

        # 运行benchmark
        result = self.benchmark._run_single_algorithm('greedy', {})

        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm_name, 'greedy')
        self.assertEqual(result.scheduled_tasks, 10)

    def test_average_results(self):
        """测试结果平均计算"""
        results = [
            BenchmarkResult(
                algorithm_name='test',
                computation_time=1.0,
                avg_window_quality=0.7,
                satellite_utilization=0.6,
            ),
            BenchmarkResult(
                algorithm_name='test',
                computation_time=2.0,
                avg_window_quality=0.8,
                satellite_utilization=0.7,
            ),
        ]

        avg = self.benchmark._average_results('test', results)

        self.assertEqual(avg.computation_time, 1.5)  # (1+2)/2
        self.assertEqual(avg.avg_window_quality, 0.75)  # (0.7+0.8)/2

    def test_compare_algorithms(self):
        """测试算法对比"""
        # 添加模拟结果
        self.benchmark.results = [
            BenchmarkResult(
                algorithm_name='baseline',
                avg_window_quality=0.7,
                computation_time=1.0,
                scheduled_tasks=100,
            ),
            BenchmarkResult(
                algorithm_name='candidate1',
                avg_window_quality=0.8,
                computation_time=1.2,
                scheduled_tasks=110,
            ),
        ]

        report = self.benchmark.compare_algorithms('baseline', ['candidate1'])

        self.assertIsInstance(report, ComparisonReport)
        self.assertEqual(report.baseline, 'baseline')
        self.assertIn('candidate1', report.improvements)

        # 验证改进幅度计算
        quality_improvement = report.improvements['candidate1']['avg_window_quality']
        self.assertAlmostEqual(quality_improvement, (0.8 - 0.7) / 0.7, places=2)

    def test_generate_json_report(self):
        """测试JSON报告生成"""
        self.benchmark.results = [
            BenchmarkResult(algorithm_name='greedy', scheduled_tasks=100),
        ]

        report_path = self.benchmark.generate_report(format='json')

        # 验证文件创建
        self.assertTrue(os.path.exists(report_path))

        # 验证内容
        with open(report_path, 'r') as f:
            data = json.load(f)
            self.assertIn('results', data)
            self.assertEqual(len(data['results']), 1)

    def test_generate_markdown_report(self):
        """测试Markdown报告生成"""
        self.benchmark.results = [
            BenchmarkResult(
                algorithm_name='greedy',
                scheduled_tasks=100,
                total_tasks=110,
                avg_window_quality=0.75,
            ),
        ]

        report_path = self.benchmark.generate_report(format='markdown')

        # 验证文件创建
        self.assertTrue(os.path.exists(report_path))

        # 验证内容包含关键信息
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn('greedy', content)
            self.assertIn('调度器Benchmark报告', content)

    def test_save_and_load_results(self):
        """测试结果保存和加载"""
        self.benchmark.results = [
            BenchmarkResult(algorithm_name='greedy', scheduled_tasks=100),
            BenchmarkResult(algorithm_name='ga', scheduled_tasks=95),
        ]

        # 保存
        save_path = self.benchmark.save_results()
        self.assertTrue(os.path.exists(save_path))

        # 创建新的benchmark实例并加载
        new_benchmark = SchedulerBenchmark(
            mission=self.mock_mission,
            output_dir=self.temp_dir,
        )
        new_benchmark.load_results(save_path)

        self.assertEqual(len(new_benchmark.results), 2)
        self.assertEqual(new_benchmark.results[0].algorithm_name, 'greedy')


class TestScheduleQualityAnalyzer(unittest.TestCase):
    """测试 ScheduleQualityAnalyzer"""

    def setUp(self):
        """设置测试环境"""
        self.mock_mission = MagicMock()

        # 创建模拟调度结果
        mock_imaging_result = MagicMock()

        # 创建模拟任务
        mock_tasks = []
        for i in range(10):
            task = MagicMock()
            task.task_id = f'TASK-{i}'
            task.quality_score = 0.5 + i * 0.05  # 0.55, 0.6, ..., 1.0
            task.start_time = datetime.now() + timedelta(minutes=i * 10)
            task.end_time = task.start_time + timedelta(minutes=5)
            mock_tasks.append(task)

        mock_imaging_result.scheduled_tasks = mock_tasks
        mock_imaging_result.unscheduled_tasks = [MagicMock() for _ in range(2)]
        mock_imaging_result.completion_time = datetime.now() + timedelta(hours=2)

        self.mock_result = MagicMock()
        self.mock_result.imaging_result = mock_imaging_result

        self.analyzer = ScheduleQualityAnalyzer(self.mock_result, self.mock_mission)

    def test_analyze_window_quality(self):
        """测试窗口质量分析"""
        analysis = self.analyzer.analyze_window_quality()

        self.assertIsInstance(analysis, WindowQualityAnalysis)
        self.assertGreater(analysis.avg_overall_quality, 0)
        self.assertGreater(len(analysis.high_quality_tasks), 0)
        self.assertGreater(len(analysis.quality_distribution), 0)
        self.assertGreater(len(analysis.recommendations), 0)

    def test_analyze_temporal_distribution(self):
        """测试时间分布分析"""
        temporal = self.analyzer.analyze_temporal_distribution()

        self.assertIsInstance(temporal.avg_task_interval_minutes, float)
        self.assertIsInstance(temporal.task_distribution_by_hour, dict)
        self.assertIsInstance(temporal.load_balance_score, float)
        self.assertGreaterEqual(temporal.load_balance_score, 0)
        self.assertLessEqual(temporal.load_balance_score, 1)

    def test_analyze_resource_efficiency(self):
        """测试资源效率分析"""
        resource = self.analyzer.analyze_resource_efficiency()

        self.assertIsInstance(resource.power_efficiency, float)
        self.assertIsInstance(resource.storage_efficiency, float)
        self.assertIsInstance(resource.slew_efficiency, float)

    def test_generate_quality_report(self):
        """测试质量报告生成"""
        report = self.analyzer.generate_quality_report()

        self.assertIn('summary', report)
        self.assertIn('window_quality', report)
        self.assertIn('temporal_distribution', report)
        self.assertIn('resource_efficiency', report)
        self.assertIn('recommendations', report)

        # 验证摘要信息
        summary = report['summary']
        self.assertEqual(summary['scheduled_tasks'], 10)
        self.assertEqual(summary['unscheduled_tasks'], 2)


class TestQualityMetricsCollector(unittest.TestCase):
    """测试 QualityMetricsCollector"""

    def setUp(self):
        """设置测试环境"""
        self.collector = QualityMetricsCollector()

    def test_add_run(self):
        """测试添加运行结果"""
        report = {
            'summary': {
                'avg_window_quality': 0.8,
                'success_rate': 0.95,
            }
        }

        self.collector.add_run('greedy', report)

        self.assertEqual(len(self.collector.runs), 1)
        self.assertEqual(self.collector.runs[0]['algorithm'], 'greedy')

    def test_get_average_metrics(self):
        """测试获取平均指标"""
        # 添加多次运行
        for quality in [0.7, 0.8, 0.9]:
            self.collector.add_run('greedy', {
                'summary': {
                    'avg_window_quality': quality,
                    'success_rate': 0.9,
                }
            })

        avg = self.collector.get_average_metrics()

        self.assertAlmostEqual(avg['avg_quality'], 0.8, places=2)
        self.assertEqual(avg['avg_success_rate'], 0.9)
        self.assertEqual(avg['run_count'], 3)

    def test_compare_algorithms(self):
        """测试算法对比"""
        # 添加不同算法的结果
        for i, algo in enumerate(['greedy', 'greedy', 'ga', 'ga']):
            self.collector.add_run(algo, {
                'summary': {
                    'avg_window_quality': 0.7 + i * 0.05,
                    'success_rate': 0.9,
                }
            })

        comparison = self.collector.compare_algorithms()

        self.assertIn('greedy', comparison)
        self.assertIn('ga', comparison)

        # 验证统计信息
        self.assertIn('avg_quality', comparison['greedy'])
        self.assertIn('quality_std', comparison['greedy'])
        self.assertEqual(comparison['greedy']['run_count'], 2)

    def test_to_dict(self):
        """测试字典转换"""
        self.collector.add_run('greedy', {'summary': {'avg_window_quality': 0.8}})

        data = self.collector.to_dict()

        self.assertIn('run_count', data)
        self.assertIn('average_metrics', data)
        self.assertIn('algorithm_comparison', data)


if __name__ == '__main__':
    unittest.main()