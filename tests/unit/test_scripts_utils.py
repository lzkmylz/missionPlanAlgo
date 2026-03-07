"""
Test suite for scripts/utils.py - Script consolidation utilities

TDD Workflow:
1. Write failing tests (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor (REFACTOR)
"""

import json
import pytest
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from scripts import utils


class TestLoadWindowCacheFromJson:
    """Tests for load_window_cache_from_json function"""

    @pytest.fixture
    def new_format_cache(self, tmp_path):
        """Create a new format cache file"""
        cache_data = {
            'target_windows': [
                {
                    'satellite_id': 'SAT-001',
                    'target_id': 'TGT-0001',
                    'start_time': '2024-03-15T10:00:00',
                    'end_time': '2024-03-15T10:05:00',
                    'max_elevation': 45.0
                }
            ],
            'ground_station_windows': [
                {
                    'satellite_id': 'SAT-001',
                    'target_id': 'GS:GS-BEIJING',
                    'start_time': '2024-03-15T12:00:00',
                    'end_time': '2024-03-15T12:10:00',
                    'max_elevation': 60.0
                }
            ]
        }
        cache_file = tmp_path / 'new_format_cache.json'
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        return str(cache_file)

    @pytest.fixture
    def old_format_cache(self, tmp_path):
        """Create an old format cache file"""
        cache_data = {
            'windows': [
                {
                    'sat': 'SAT-001',
                    'tgt': 'TGT-0001',
                    'start': '2024-03-15T10:00:00',
                    'end': '2024-03-15T10:05:00',
                    'el': 45.0
                },
                {
                    'sat': 'SAT-001',
                    'tgt': 'GS:GS-BEIJING',
                    'start': '2024-03-15T12:00:00',
                    'end': '2024-03-15T12:10:00',
                    'el': 60.0
                }
            ]
        }
        cache_file = tmp_path / 'old_format_cache.json'
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        return str(cache_file)

    def test_load_new_format_cache(self, new_format_cache):
        """Test loading new format cache with target_windows and ground_station_windows"""
        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = []
        mock_mission.targets = []

        # Act
        cache = utils.load_window_cache_from_json(new_format_cache, mock_mission)

        # Assert
        assert cache is not None
        stats = cache.get_statistics()
        assert stats['total_windows'] == 2
        assert stats['sat_target_pairs'] == 2

    def test_load_old_format_cache(self, old_format_cache):
        """Test loading old format cache with 'windows' array"""
        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = []
        mock_mission.targets = []

        # Act
        cache = utils.load_window_cache_from_json(old_format_cache, mock_mission)

        # Assert
        assert cache is not None
        stats = cache.get_statistics()
        assert stats['total_windows'] == 2

    def test_distinguishes_gs_windows_in_old_format(self, old_format_cache):
        """Test that old format correctly distinguishes ground station windows (GS: prefix)"""
        # Arrange
        mock_mission = Mock()
        mock_mission.satellites = []
        mock_mission.targets = []

        # Act
        cache = utils.load_window_cache_from_json(old_format_cache, mock_mission)

        # Assert - should have 1 target window and 1 GS window
        windows = cache._windows
        target_keys = [k for k in windows.keys() if not k[1].startswith('GS:')]
        gs_keys = [k for k in windows.keys() if k[1].startswith('GS:')]
        assert len(target_keys) == 1
        assert len(gs_keys) == 1

    def test_empty_cache_returns_empty(self, tmp_path):
        """Test loading empty cache"""
        # Arrange
        cache_file = tmp_path / 'empty_cache.json'
        with open(cache_file, 'w') as f:
            json.dump({'target_windows': [], 'ground_station_windows': []}, f)
        mock_mission = Mock()

        # Act
        cache = utils.load_window_cache_from_json(str(cache_file), mock_mission)

        # Assert
        assert cache is not None
        stats = cache.get_statistics()
        assert stats['total_windows'] == 0

    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError"""
        mock_mission = Mock()

        with pytest.raises(FileNotFoundError):
            utils.load_window_cache_from_json('non_existent_file.json', mock_mission)


class TestSchedulerRegistry:
    """Tests for SCHEDULER_REGISTRY"""

    def test_registry_contains_all_algorithms(self):
        """Test that registry contains all expected algorithms"""
        expected_algorithms = {
            'greedy', 'ga', 'edd', 'spt',
            'aco', 'pso', 'sa', 'tabu'
        }

        for algo in expected_algorithms:
            assert algo in utils.SCHEDULER_REGISTRY, f"Missing algorithm: {algo}"

    def test_registry_entries_are_callable(self):
        """Test that all registry entries are callable classes"""
        for name, scheduler_class in utils.SCHEDULER_REGISTRY.items():
            assert callable(scheduler_class), f"{name} is not callable"

    def test_get_scheduler_class(self):
        """Test getting scheduler class by name"""
        greedy_class = utils.get_scheduler_class('greedy')
        assert greedy_class is not None

        ga_class = utils.get_scheduler_class('ga')
        assert ga_class is not None

    def test_get_scheduler_class_invalid(self):
        """Test getting invalid scheduler class raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            utils.get_scheduler_class('invalid_algorithm')
        assert 'invalid_algorithm' in str(exc_info.value)


class TestSetupLogging:
    """Tests for setup_logging function"""

    def test_setup_logging_default_level(self):
        """Test logging setup with default INFO level"""
        # Act
        utils.setup_logging()

        # Assert
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level"""
        # Act
        utils.setup_logging(level=logging.DEBUG)

        # Assert
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns the configured logger"""
        # Act
        logger = utils.setup_logging()

        # Assert
        assert logger is not None
        assert isinstance(logger, logging.Logger)


class TestSaveResults:
    """Tests for save_results function"""

    @pytest.fixture
    def sample_result_data(self):
        """Create sample result data"""
        return {
            'algorithm': 'greedy',
            'scheduled_count': 10,
            'computation_time': 1.5,
            'metrics': {
                'demand_satisfaction_rate': 0.85
            }
        }

    def test_save_results_creates_file(self, sample_result_data, tmp_path):
        """Test that save_results creates the output file"""
        # Arrange
        output_path = tmp_path / 'results.json'

        # Act
        utils.save_results(sample_result_data, str(output_path))

        # Assert
        assert output_path.exists()

    def test_save_results_writes_valid_json(self, sample_result_data, tmp_path):
        """Test that save_results writes valid JSON"""
        # Arrange
        output_path = tmp_path / 'results.json'

        # Act
        utils.save_results(sample_result_data, str(output_path))

        # Assert
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_result_data

    def test_save_results_creates_directories(self, sample_result_data, tmp_path):
        """Test that save_results creates parent directories"""
        # Arrange
        output_path = tmp_path / 'nested' / 'dir' / 'results.json'

        # Act
        utils.save_results(sample_result_data, str(output_path))

        # Assert
        assert output_path.exists()


class TestAlgorithmCategories:
    """Tests for algorithm categorization"""

    def test_greedy_algorithms(self):
        """Test greedy algorithm category"""
        greedy_algos = utils.get_algorithms_by_category('greedy')
        expected = {'greedy', 'edd', 'spt'}
        assert set(greedy_algos) == expected

    def test_metaheuristic_algorithms(self):
        """Test metaheuristic algorithm category"""
        meta_algos = utils.get_algorithms_by_category('metaheuristic')
        expected = {'ga', 'aco', 'pso', 'sa', 'tabu'}
        assert set(meta_algos) == expected

    def test_all_algorithms_have_category(self):
        """Test that all registered algorithms have a category"""
        for algo_name in utils.SCHEDULER_REGISTRY:
            category = utils.get_algorithm_category(algo_name)
            assert category is not None
            assert category in ['greedy', 'metaheuristic']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
