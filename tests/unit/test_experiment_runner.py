"""
实验运行器测试

TDD测试文件 - 第4章设计实现
支持批量对比实验和参数敏感性分析
"""

import pytest
from datetime import datetime, timedelta
import json
import tempfile
import os

from core.models import Mission, Satellite, SatelliteType, Target, TargetType
from experiments.runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.base_scheduler import ScheduleResult, ScheduledTask


class MockScheduler:
    """模拟调度器用于测试"""
    def __init__(self, config=None):
        self.config = config or {}
        self.mission = None

    def initialize(self, mission):
        self.mission = mission

    def schedule(self):
        # 返回模拟结果
        return ScheduleResult(
            scheduled_tasks=[],
            unscheduled_tasks={},
            makespan=3600.0,
            computation_time=0.5,
            iterations=1
        )

    def get_parameters(self):
        return {}


class TestExperimentConfig:
    """测试实验配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ExperimentConfig()
        assert config.repetitions == 10
        assert config.output_dir == "./results"

    def test_custom_config(self):
        """测试自定义配置"""
        config = ExperimentConfig(
            repetitions=5,
            random_seed=42,
            output_dir="./custom_results"
        )
        assert config.repetitions == 5
        assert config.random_seed == 42
        assert config.output_dir == "./custom_results"


class TestExperimentRunner:
    """测试实验运行器"""

    def setup_method(self):
        """设置测试场景"""
        # 创建简单场景
        satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )

        targets = [
            Target(
                id=f"TARGET-{i:02d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i,
                latitude=39.0,
                priority=5
            )
            for i in range(5)
        ]

        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[satellite],
            targets=targets
        )

        self.algorithms = {
            'Greedy': MockScheduler,
            'EDD': MockScheduler,
        }

    def test_runner_initialization(self):
        """测试运行器初始化"""
        config = ExperimentConfig(repetitions=3)
        runner = ExperimentRunner(self.mission, self.algorithms, config)

        assert runner.mission == self.mission
        assert runner.algorithms == self.algorithms
        assert runner.config.repetitions == 3

    def test_run_single_experiment(self):
        """测试运行单次实验"""
        config = ExperimentConfig(repetitions=1)
        runner = ExperimentRunner(self.mission, self.algorithms, config)

        result = runner.run_single_experiment('Greedy', {})

        assert isinstance(result, ExperimentResult)
        assert result.algorithm_name == 'Greedy'
        assert result.repetition == 1
        assert 'metrics' in result.to_dict()

    def test_run_all_experiments(self):
        """测试运行所有算法实验"""
        config = ExperimentConfig(repetitions=2)
        runner = ExperimentRunner(self.mission, self.algorithms, config)

        results = runner.run_all()

        # 应该有两个算法的结果
        assert 'Greedy' in results
        assert 'EDD' in results

        # 每个算法应该有2次重复实验
        assert len(results['Greedy']) == 2
        assert len(results['EDD']) == 2

    def test_compare_algorithms(self):
        """测试算法对比"""
        config = ExperimentConfig(repetitions=3)
        runner = ExperimentRunner(self.mission, self.algorithms, config)

        # 先运行实验
        runner.run_all()

        # 生成对比报告
        comparison = runner.compare_algorithms()

        assert 'Greedy' in comparison
        assert 'EDD' in comparison
        assert 'mean_demand_satisfaction_rate' in comparison['Greedy']
        assert 'std_demand_satisfaction_rate' in comparison['Greedy']

    def test_export_results(self):
        """测试结果导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(repetitions=2, output_dir=tmpdir)
            runner = ExperimentRunner(self.mission, self.algorithms, config)

            runner.run_all()
            runner.export_results()

            # 检查是否生成了结果文件
            assert os.path.exists(os.path.join(tmpdir, 'experiment_results.json'))

    def test_parameter_sensitivity_analysis(self):
        """测试参数敏感性分析"""
        config = ExperimentConfig(repetitions=2)
        runner = ExperimentRunner(self.mission, {'Greedy': MockScheduler}, config)

        # 定义参数范围
        param_ranges = {
            'heuristic': ['priority', 'earliest']
        }

        results = runner.run_sensitivity_analysis('Greedy', param_ranges)

        # 应该测试所有参数组合
        assert len(results) == 2  # 两个heuristic值

    def test_invalid_algorithm_name(self):
        """测试无效算法名称处理"""
        config = ExperimentConfig(repetitions=1)
        runner = ExperimentRunner(self.mission, self.algorithms, config)

        with pytest.raises(ValueError):
            runner.run_single_experiment('NonExistentAlgo', {})

    def test_experiment_result_data_class(self):
        """测试实验结果数据类"""
        result = ExperimentResult(
            algorithm_name='Greedy',
            repetition=1,
            metrics={'dsr': 0.85, 'makespan': 3600},
            scheduled_tasks=[],
            computation_time=1.5
        )

        assert result.algorithm_name == 'Greedy'
        assert result.repetition == 1
        assert result.metrics['dsr'] == 0.85

        # 测试序列化
        result_dict = result.to_dict()
        assert result_dict['algorithm_name'] == 'Greedy'
        assert 'timestamp' in result_dict


class TestExperimentRunnerEdgeCases:
    """测试边界情况"""

    def test_empty_algorithm_list(self):
        """测试空算法列表"""
        mission = Mission(
            name="空场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[]
        )

        config = ExperimentConfig()
        runner = ExperimentRunner(mission, {}, config)

        results = runner.run_all()
        assert results == {}

    def test_zero_repetitions(self):
        """测试零重复次数"""
        mission = Mission(
            name="测试",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[Satellite(id="SAT-01", name="测试", sat_type=SatelliteType.OPTICAL_1)],
            targets=[]
        )

        config = ExperimentConfig(repetitions=0)
        runner = ExperimentRunner(mission, {'Greedy': MockScheduler}, config)

        results = runner.run_all()
        # 零重复应该返回空列表
        assert results['Greedy'] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
