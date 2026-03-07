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


class TestBuildSchedulerConfig:
    """Tests for build_scheduler_config function"""

    def test_build_config_default(self):
        """Test building default config"""
        from scripts.run_scheduler import build_scheduler_config, parse_args

        args = parse_args(['--cache', 'cache.json', '--scenario', 'scenario.json'])
        config = build_scheduler_config(args)

        assert config['consider_power'] is True
        assert config['consider_storage'] is True

    def test_build_config_simplified(self):
        """Test building simplified config"""
        from scripts.run_scheduler import build_scheduler_config, parse_args

        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--simplified'
        ])
        config = build_scheduler_config(args)

        assert config['use_simplified_slew'] is True

    def test_build_config_ga(self):
        """Test building GA config"""
        from scripts.run_scheduler import build_scheduler_config, parse_args

        args = parse_args([
            '--cache', 'cache.json',
            '--scenario', 'scenario.json',
            '--algorithm', 'ga',
            '--generations', '150',
            '--population-size', '75',
            '--mutation-rate', '0.15',
            '--crossover-rate', '0.85'
        ])
        config = build_scheduler_config(args)

        assert config['generations'] == 150
        assert config['population_size'] == 75
        assert config['mutation_rate'] == 0.15
        assert config['crossover_rate'] == 0.85


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
            cache=mock_cache,
            config={}
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
            config={},
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
            config={},
            repetitions=1
        )

        # Assert
        assert len(results) == 2
        assert mock_run_single.call_count == 2


class TestPrintResults:
    """Tests for print_results function"""

    def test_print_results_single_mode(self, capsys):
        """Test printing single mode results"""
        from scripts.run_scheduler import print_results

        result = {
            'algorithm': 'greedy',
            'scheduled_count': 100,
            'unscheduled_count': 10,
            'demand_satisfaction_rate': 0.9,
            'satellite_utilization': 0.5,
            'makespan_hours': 24.0,
            'computation_time': 5.0
        }

        print_results(result, mode='single')

        captured = capsys.readouterr()
        assert 'greedy' in captured.out.lower() or 'GREEDY' in captured.out
        assert '100' in captured.out

    def test_print_results_compare_mode(self, capsys):
        """Test printing compare mode results"""
        from scripts.run_scheduler import print_results

        results = [
            {
                'algorithm': 'greedy',
                'scheduled_count': 100,
                'demand_satisfaction_rate': 0.9,
                'satellite_utilization': 0.5,
                'computation_time': 5.0
            },
            {
                'algorithm': 'ga',
                'scheduled_count': 110,
                'demand_satisfaction_rate': 0.95,
                'satellite_utilization': 0.6,
                'computation_time': 10.0
            }
        ]

        print_results(results, mode='compare')

        captured = capsys.readouterr()
        assert 'greedy' in captured.out.lower() or 'GREEDY' in captured.out
        assert 'ga' in captured.out.lower() or 'GA' in captured.out


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
