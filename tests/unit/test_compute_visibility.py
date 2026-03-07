"""
Test suite for compute_visibility.py - Consolidated visibility computation script

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
        from scripts.compute_visibility import parse_args

        args = parse_args(['--scenario', 'scenario.json'])

        assert args.scenario == 'scenario.json'
        assert args.mode == 'batch'  # default
        assert args.output == 'results/visibility_cache.json'

    def test_parse_args_pairwise_mode(self):
        """Test parsing pairwise mode"""
        from scripts.compute_visibility import parse_args

        args = parse_args([
            '--scenario', 'scenario.json',
            '--mode', 'pairwise'
        ])

        assert args.mode == 'pairwise'

    def test_parse_args_custom_steps(self):
        """Test parsing custom step parameters"""
        from scripts.compute_visibility import parse_args

        args = parse_args([
            '--scenario', 'scenario.json',
            '--coarse-step', '600',
            '--fine-step', '120',
            '--min-elevation', '10'
        ])

        assert args.coarse_step == 600.0
        assert args.fine_step == 120.0
        assert args.min_elevation == 10.0


class TestBuildComputationConfig:
    """Tests for build_computation_config function"""

    def test_build_config_default(self):
        """Test building default config"""
        from scripts.compute_visibility import build_computation_config, parse_args

        args = parse_args(['--scenario', 'scenario.json'])
        config = build_computation_config(args)

        assert config.coarse_step_seconds == 5.0
        assert config.fine_step_seconds == 1.0
        assert config.min_elevation_degrees == 5.0

    def test_build_config_custom(self):
        """Test building custom config"""
        from scripts.compute_visibility import build_computation_config, parse_args

        args = parse_args([
            '--scenario', 'scenario.json',
            '--coarse-step', '600',
            '--fine-step', '120',
            '--min-elevation', '10'
        ])
        config = build_computation_config(args)

        assert config.coarse_step_seconds == 600.0
        assert config.fine_step_seconds == 120.0
        assert config.min_elevation_degrees == 10.0


class TestComputeVisibilityBatch:
    """Tests for batch mode computation"""

    @patch('scripts.compute_visibility.BatchVisibilityCalculator')
    @patch('scripts.compute_visibility.Mission')
    def test_compute_batch_mode(self, mock_mission_class, mock_calculator_class):
        """Test batch mode computation"""
        from scripts.compute_visibility import compute_visibility

        # Arrange
        mock_mission = Mock()
        mock_mission.name = 'Test Mission'
        mock_mission.satellites = [Mock(id='SAT-001')]
        mock_mission.targets = [Mock(id='TGT-001')]
        mock_mission.ground_stations = []
        mock_mission.start_time = datetime(2024, 3, 15)
        mock_mission.end_time = datetime(2024, 3, 16)
        mock_mission_class.load.return_value = mock_mission

        # Mock calculator result
        mock_result = Mock()
        mock_result.target_windows = {'SAT-001': []}
        mock_result.ground_station_windows = {}
        mock_result.total_window_count = 0
        mock_result.to_cache_format.return_value = {
            'target_windows': [],
            'ground_station_windows': []
        }

        # Mock stats with proper numeric values
        mock_stats = Mock()
        mock_stats.java_computation_time_ms = 1000
        mock_stats.memory_usage_mb = 100.0
        mock_stats.total_computation_time_ms = 2000

        mock_calc = Mock()
        mock_calc.compute_all_windows.return_value = mock_result
        mock_calc.last_computation_stats = mock_stats
        mock_calculator_class.return_value = mock_calc

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'output.json'

            # Act
            result = compute_visibility(
                scenario_path='scenario.json',
                output_path=str(output_path),
                mode='batch'
            )

            # Assert
            assert result is not None
            mock_calculator_class.assert_called_once()

    @patch('scripts.compute_visibility.Mission')
    def test_compute_file_not_found(self, mock_mission_class):
        """Test handling of non-existent scenario file"""
        from scripts.compute_visibility import compute_visibility

        mock_mission_class.load.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            compute_visibility(
                scenario_path='non_existent.json',
                output_path='output.json',
                mode='batch'
            )


class TestPrintResults:
    """Tests for print_results function"""

    def test_print_results_with_stats(self, capsys):
        """Test printing results with statistics"""
        from scripts.compute_visibility import print_results

        stats = {
            'target_windows': 100,
            'ground_station_windows': 50,
            'total_windows': 150,
            'computation_time': 10.5
        }

        print_results(stats, mode='batch')

        captured = capsys.readouterr()
        assert '150' in captured.out or '100' in captured.out

    def test_print_results_no_stats(self, capsys):
        """Test printing results without statistics"""
        from scripts.compute_visibility import print_results

        print_results({}, mode='pairwise')

        captured = capsys.readouterr()
        # Should print without error


class TestMainWorkflow:
    """Integration tests for main workflow"""

    @patch('scripts.compute_visibility.Mission')
    @patch('scripts.compute_visibility.compute_visibility_batch')
    def test_main_batch_mode(self, mock_compute_batch, mock_mission_class):
        """Test main function in batch mode"""
        from scripts.compute_visibility import main

        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = [Mock()]
        mock_mission.targets = [Mock()]
        mock_mission.ground_stations = []
        mock_mission_class.load.return_value = mock_mission

        mock_compute_batch.return_value = {
            'target_windows': 100,
            'ground_station_windows': 50,
            'total_windows': 150
        }

        # Act
        result = main(['--scenario', 'scenario.json', '--mode', 'batch'])

        # Assert
        assert result == 0
        mock_compute_batch.assert_called_once()

    @patch('scripts.compute_visibility.Mission')
    @patch('scripts.compute_visibility.compute_visibility_pairwise')
    def test_main_pairwise_mode(self, mock_compute_pairwise, mock_mission_class):
        """Test main function in pairwise mode"""
        from scripts.compute_visibility import main

        mock_mission = Mock()
        mock_mission.satellites = [Mock()]
        mock_mission.targets = [Mock()]
        mock_mission.ground_stations = []
        mock_mission_class.load.return_value = mock_mission

        mock_compute_pairwise.return_value = {}

        result = main(['--scenario', 'scenario.json', '--mode', 'pairwise'])

        assert result == 0
        mock_compute_pairwise.assert_called_once()

    @patch('scripts.compute_visibility.compute_visibility')
    def test_main_with_error(self, mock_compute):
        """Test main function handling errors"""
        from scripts.compute_visibility import main

        mock_compute.side_effect = FileNotFoundError("Scenario not found")

        result = main(['--scenario', 'non_existent.json'])

        assert result == 1


class TestSaveResults:
    """Tests for save_results function"""

    @pytest.fixture
    def sample_result_data(self):
        """Create sample result data"""
        return {
            'metadata': {
                'scenario': 'test.json',
                'computed_at': datetime.now().isoformat()
            },
            'stats': {
                'total_windows': 100,
                'target_windows': 80,
                'ground_station_windows': 20
            },
            'windows': {
                'target_windows': [],
                'ground_station_windows': []
            }
        }

    def test_save_results_creates_file(self, sample_result_data, tmp_path):
        """Test that save_results creates the output file"""
        from scripts.compute_visibility import save_results

        output_path = tmp_path / 'results.json'

        save_results(sample_result_data, str(output_path))

        assert output_path.exists()

    def test_save_results_valid_json(self, sample_result_data, tmp_path):
        """Test that save_results writes valid JSON"""
        from scripts.compute_visibility import save_results

        output_path = tmp_path / 'results.json'

        save_results(sample_result_data, str(output_path))

        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_result_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
