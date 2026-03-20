"""
Test suite for run_scheduler.py - Consolidated scheduler execution script

TDD Workflow:
1. Write failing tests (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor (REFACTOR)
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))


class TestArgumentParsing:
    """Tests for argument parsing"""

    def test_parse_args_minimal(self):
        """Test parsing minimal required arguments"""
        from scripts.run_scheduler import parse_args

        args = parse_args(['--cache', 'cache.json', '--scenario', 'scenario.json'])

        assert args.cache == 'cache.json'
        assert args.scenario == 'scenario.json'
        assert args.algorithm == 'greedy'  # default
        assert args.mode == 'single'  # default

    def test_parse_args_compare_mode(self):
        """Test parsing compare mode"""
        from scripts.run_scheduler import parse_args

        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--mode', 'compare',
            '--algorithm', 'greedy,ga,edd'
        ])

        assert args.mode == 'compare'
        assert args.algorithm == 'greedy,ga,edd'

    def test_parse_args_with_frequency(self):
        """Test parsing with frequency enabled"""
        from scripts.run_scheduler import parse_args

        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--frequency'
        ])

        assert args.frequency is True

    def test_parse_args_with_downlink(self):
        """Test parsing with downlink enabled"""
        from scripts.run_scheduler import parse_args

        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--downlink'
        ])

        assert args.downlink is True

    def test_parse_args_ga_parameters(self):
        """Test parsing GA-specific parameters"""
        from scripts.run_scheduler import parse_args

        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--algorithm', 'ga',
            '--generations', '200',
            '--population-size', '100'
        ])

        assert args.algorithm == 'ga'
        assert args.generations == 200
        assert args.population_size == 100


class TestSimplifiedModeRejected:
    """Tests for simplified mode rejection"""

    def test_simplified_mode_rejected(self):
        """Test that simplified mode is rejected in high precision mode"""
        from scripts.run_scheduler import parse_args

        # --simplified 参数已被隐藏，但即使使用也应被拒绝
        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--simplified'
        ])
        # 在高精度模式下，简化模式应该被拒绝
        assert args.simplified is True  # 参数可以被解析，但在使用时会报错

    def test_simplified_mode_raises_error_in_main(self):
        """Test that using --simplified raises ValueError in main"""
        from scripts.run_scheduler import main, parse_args

        with patch('scripts.run_scheduler.Mission') as mock_mission_class:
            mock_mission = Mock()
            mock_mission.satellites = [Mock()]
            mock_mission.targets = [Mock()]
            mock_mission.ground_stations = []
            mock_mission_class.load.return_value = mock_mission

            with patch('scripts.run_scheduler.load_window_cache_from_json') as mock_load:
                mock_cache = Mock()
                mock_cache.get_statistics.return_value = {'total_windows': 100, 'sat_target_pairs': 10}
                mock_load.return_value = mock_cache

                # Should return 1 (error) when --simplified is used
                result = main([
                    '--cache', 'cache.json',
                    '--scenario', 'scenario.json',
                    '--simplified'
                ])

                assert result == 1  # Error exit code


class TestRunSingleAlgorithm:
    """Tests for run_single_algorithm function"""

    @patch('scripts.run_scheduler.UnifiedScheduler')
    @patch('scripts.run_scheduler.MetricsCalculator')
    def test_run_single_algorithm_success(self, mock_metrics_calc, mock_unified_scheduler):
        """Test successful algorithm execution"""
        from scripts.run_scheduler import run_single_algorithm

        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = []
        mock_mission.targets = []
        mock_mission.ground_stations = []
        mock_mission.relay_satellites = []
        mock_cache = Mock()

        # Mock UnifiedScheduler result
        mock_result = Mock()
        mock_result.imaging_result.scheduled_tasks = []
        mock_result.imaging_result.unscheduled_tasks = {}
        mock_result.imaging_result.makespan = 0
        mock_result.imaging_result.computation_time = 1.0
        mock_result.imaging_result.convergence_curve = []
        mock_result.downlink_result = None
        mock_result.target_observations = {}

        mock_scheduler = Mock()
        mock_scheduler.schedule.return_value = mock_result
        mock_unified_scheduler.return_value = mock_scheduler

        mock_metrics = Mock()
        mock_metrics.scheduled_task_count = 10
        mock_metrics.demand_satisfaction_rate = 0.85
        mock_metrics.makespan = 3600
        mock_metrics.satellite_utilization = 0.5
        mock_metrics.solution_quality = 0.9
        mock_metrics_calc.return_value.calculate_all.return_value = mock_metrics

        # Act
        result = run_single_algorithm(
            algorithm_name='greedy',
            mission=mock_mission,
            cache=mock_cache
        )

        # Assert
        assert result is not None
        assert result['algorithm'] == 'greedy'
        assert result['scheduled_count'] == 0

    @patch('scripts.run_scheduler.GroundStationPool')
    @patch('scripts.run_scheduler.UnifiedScheduler')
    @patch('scripts.run_scheduler.MetricsCalculator')
    def test_run_single_algorithm_with_downlink(self, mock_metrics_calc, mock_unified_scheduler, mock_gs_pool):
        """Test algorithm execution with downlink enabled"""
        from scripts.run_scheduler import run_single_algorithm

        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = []
        mock_mission.targets = []
        mock_mission.relay_satellites = []

        # Mock ground station with proper attributes
        mock_gs = Mock()
        mock_gs.id = 'GS-001'
        mock_gs.antennas = []
        mock_mission.ground_stations = [mock_gs]

        mock_cache = Mock()

        mock_result = Mock()
        mock_result.imaging_result.scheduled_tasks = []
        mock_result.imaging_result.unscheduled_tasks = {}
        mock_result.imaging_result.makespan = 0
        mock_result.imaging_result.computation_time = 1.0
        mock_result.imaging_result.convergence_curve = []

        # Mock downlink result
        mock_downlink_result = Mock()
        mock_downlink_result.downlink_tasks = [Mock(), Mock()]
        mock_downlink_result.failed_tasks = []
        mock_result.downlink_result = mock_downlink_result
        mock_result.target_observations = {}

        mock_scheduler = Mock()
        mock_scheduler.schedule.return_value = mock_result
        mock_unified_scheduler.return_value = mock_scheduler

        mock_metrics = Mock()
        mock_metrics.scheduled_task_count = 10
        mock_metrics.demand_satisfaction_rate = 0.85
        mock_metrics.makespan = 3600
        mock_metrics.satellite_utilization = 0.5
        mock_metrics.solution_quality = 0.9
        mock_metrics_calc.return_value.calculate_all.return_value = mock_metrics

        # Mock GroundStationPool
        mock_pool = Mock()
        mock_gs_pool.return_value = mock_pool

        # Act
        result = run_single_algorithm(
            algorithm_name='greedy',
            mission=mock_mission,
            cache=mock_cache,
            enable_downlink=True
        )

        # Assert
        assert result is not None
        assert result['downlink_count'] == 2
        mock_gs_pool.assert_called_once()


class TestRunComparison:
    """Tests for run_comparison function"""

    @patch('scripts.run_scheduler.run_single_algorithm')
    def test_run_comparison_multiple_algorithms(self, mock_run_single):
        """Test running comparison with multiple algorithms"""
        from scripts.run_scheduler import run_comparison

        # Arrange
        mock_mission = Mock()
        mock_cache = Mock()

        mock_run_single.return_value = {
            'algorithm': 'greedy',
            'scheduled_count': 10,
            'demand_satisfaction_rate': 0.85
        }

        algorithms = ['greedy', 'ga']

        # Act
        results = run_comparison(
            mission=mock_mission,
            cache=mock_cache,
            algorithms=algorithms,
            repetitions=1
        )

        # Assert
        assert len(results) == 2
        assert mock_run_single.call_count == 2


class TestPrintResults:
    """Tests for print results functions"""

    def test_print_single_result(self, capsys):
        """Test printing single mode results"""
        from scripts.run_scheduler import print_single_result

        result = {
            'algorithm_name': 'Greedy Scheduler',
            'scheduled_count': 100,
            'unscheduled_count': 10,
            'demand_satisfaction_rate': 0.9,
            'satellite_utilization': 0.5,
            'makespan_hours': 24.0,
            'computation_time': 5.0
        }

        print_single_result(result)

        captured = capsys.readouterr()
        assert 'Greedy' in captured.out or '调度结果' in captured.out
        assert '100' in captured.out

    def test_print_comparison_results(self, capsys):
        """Test printing compare mode results"""
        from scripts.run_scheduler import print_comparison_results

        results = [
            {
                'algorithm_name': 'Greedy Scheduler',
                'scheduled_count': 100,
                'demand_satisfaction_rate': 0.9,
                'satellite_utilization': 0.5,
                'computation_time': 5.0
            },
            {
                'algorithm_name': 'Genetic Algorithm',
                'scheduled_count': 110,
                'demand_satisfaction_rate': 0.95,
                'satellite_utilization': 0.6,
                'computation_time': 10.0
            }
        ]

        print_comparison_results(results)

        captured = capsys.readouterr()
        assert 'Greedy' in captured.out or '对比结果' in captured.out


class TestMainWorkflow:
    """Integration tests for main workflow"""

    @patch('scripts.run_scheduler.Mission')
    @patch('scripts.run_scheduler.load_window_cache_from_json')
    @patch('scripts.run_scheduler.run_single_algorithm')
    def test_main_single_mode(self, mock_run, mock_load_cache, mock_mission_class):
        """Test main function in single mode"""
        from scripts.run_scheduler import main

        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = [Mock()]
        mock_mission.targets = [Mock()]
        mock_mission.ground_stations = []
        mock_mission_class.load.return_value = mock_mission

        mock_cache = Mock()
        mock_cache.get_statistics.return_value = {'total_windows': 100, 'sat_target_pairs': 10}
        mock_load_cache.return_value = mock_cache

        mock_run.return_value = {
            'algorithm': 'greedy',
            'algorithm_name': 'Greedy Scheduler',
            'scheduled_count': 10,
            'demand_satisfaction_rate': 0.85,
            'unscheduled_count': 0,
            'satellite_utilization': 0.5,
            'makespan_hours': 24.0,
            'computation_time': 1.0,
            'solution_quality': 0.9,
            'downlink_count': 0,
            'frequency_satisfaction': None
        }

        # Act
        result = main(['--cache', 'cache.json', '--scenario', 'scenario.json'])

        # Assert
        assert result == 0
        mock_run.assert_called_once()

    @patch('scripts.run_scheduler.Mission')
    @patch('scripts.run_scheduler.load_window_cache_from_json')
    @patch('scripts.run_scheduler.run_comparison')
    def test_main_compare_mode(self, mock_run_compare, mock_load_cache, mock_mission_class):
        """Test main function in compare mode"""
        from scripts.run_scheduler import main

        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = [Mock()]
        mock_mission.targets = [Mock()]
        mock_mission.ground_stations = []  # Empty list, not None
        mock_mission_class.load.return_value = mock_mission

        mock_cache = Mock()
        mock_cache.get_statistics.return_value = {'total_windows': 100, 'sat_target_pairs': 10}
        mock_load_cache.return_value = mock_cache

        mock_run_compare.return_value = []

        # Act
        result = main([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--mode', 'compare',
            '--algorithm', 'greedy,ga'
        ])

        # Assert
        assert result == 0
        mock_run_compare.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
