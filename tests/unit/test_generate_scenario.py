"""
Test suite for generate_scenario.py - Consolidated scenario generation script

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
        from scripts.generate_scenario import parse_args

        args = parse_args([])

        assert args.output == 'scenarios/generated_scenario.json'
        assert args.seed == 42
        assert args.frequency is False

    def test_parse_args_with_frequency(self):
        """Test parsing with frequency enabled"""
        from scripts.generate_scenario import parse_args

        args = parse_args(['--frequency'])

        assert args.frequency is True

    def test_parse_args_custom_seed(self):
        """Test parsing custom seed"""
        from scripts.generate_scenario import parse_args

        args = parse_args(['--seed', '123'])

        assert args.seed == 123

    def test_parse_args_custom_output(self):
        """Test parsing custom output path"""
        from scripts.generate_scenario import parse_args

        args = parse_args(['--output', 'custom/scenario.json'])

        assert args.output == 'custom/scenario.json'


class TestScenarioGenerator:
    """Tests for ScenarioGenerator class"""

    def test_generator_initialization(self):
        """Test generator initialization"""
        from scripts.generate_scenario import ScenarioGenerator

        generator = ScenarioGenerator(seed=42, epoch="2024-03-15T00:00:00Z")

        assert generator.seed == 42
        assert generator.epoch == "2024-03-15T00:00:00Z"

    def test_generate_walker_orbits(self):
        """Test Walker orbit generation"""
        from scripts.generate_scenario import ScenarioGenerator

        generator = ScenarioGenerator(seed=42)
        orbits = generator.generate_walker_orbits('optical', phase_offset=0)

        assert len(orbits) == 30  # 30/6/1 Walker constellation
        # Check that orbits have required fields
        for orbit in orbits:
            assert orbit.semi_major_axis > 0
            assert 0 <= orbit.inclination <= 90
            assert 0 <= orbit.raan < 360

    def test_generate_satellites(self):
        """Test satellite generation"""
        from scripts.generate_scenario import ScenarioGenerator

        generator = ScenarioGenerator(seed=42)
        satellites = generator.generate_satellites()

        assert len(satellites) == 60  # 30 optical + 30 SAR

        optical_count = sum(1 for s in satellites if s.sat_type == 'optical')
        sar_count = sum(1 for s in satellites if s.sat_type == 'sar')

        assert optical_count == 30
        assert sar_count == 30

    def test_generate_ground_stations(self):
        """Test ground station generation"""
        from scripts.generate_scenario import ScenarioGenerator

        generator = ScenarioGenerator(seed=42)
        stations = generator.generate_ground_stations()

        assert len(stations) == 12

        # Check required fields
        for station in stations:
            assert station.id.startswith('GS-')
            assert len(station.location) == 3  # lon, lat, alt

    def test_generate_targets_without_frequency(self):
        """Test target generation without frequency constraints"""
        from scripts.generate_scenario import ScenarioGenerator

        generator = ScenarioGenerator(seed=42)
        targets = generator.generate_targets(enable_frequency=False)

        assert len(targets) == 1000  # Total targets

        # Without frequency, all targets should have required_observations = 1
        for target in targets:
            assert hasattr(target, 'priority')
            assert hasattr(target, 'location')

    def test_generate_targets_with_frequency(self):
        """Test target generation with frequency constraints"""
        from scripts.generate_scenario import ScenarioGenerator

        generator = ScenarioGenerator(seed=42)
        targets = generator.generate_targets(enable_frequency=True)

        assert len(targets) == 1000

        # With frequency, targets should have frequency-related fields
        freq_targets = [t for t in targets if hasattr(t, 'required_observations')]
        assert len(freq_targets) > 0


class TestGenerateScenario:
    """Tests for generate_scenario function"""

    def test_generate_scenario_basic(self):
        """Test basic scenario generation"""
        from scripts.generate_scenario import generate_scenario

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'scenario.json'

            # Act
            result = generate_scenario(
                output_path=str(output_path),
                seed=42,
                enable_frequency=False
            )

            # Assert
            assert result is not None
            assert output_path.exists()
            assert 'name' in result

    def test_generate_scenario_output_structure(self):
        """Test that generated scenario has correct structure"""
        from scripts.generate_scenario import generate_scenario

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'scenario.json'

            # Act
            generate_scenario(
                output_path=str(output_path),
                seed=42,
                enable_frequency=False
            )

            # Assert
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert 'name' in data
            assert 'version' in data
            assert 'satellites' in data
            assert 'ground_stations' in data
            assert 'targets' in data
            assert 'duration' in data


class TestPrintSummary:
    """Tests for print_scenario_summary function"""

    def test_print_summary_basic(self, capsys):
        """Test printing basic summary"""
        from scripts.generate_scenario import print_scenario_summary

        scenario = {
            'name': 'Test Scenario',
            'description': 'A test scenario',
            'generated_at': datetime.now().isoformat(),
            'seed': 42,
            'duration': {
                'start': '2024-03-15T00:00:00Z',
                'end': '2024-03-16T00:00:00Z'
            },
            'satellites': [],
            'ground_stations': [],
            'targets': []
        }

        print_scenario_summary(scenario)

        captured = capsys.readouterr()
        assert 'Test Scenario' in captured.out


class TestMainWorkflow:
    """Integration tests for main workflow"""

    def test_main_basic(self):
        """Test main function basic execution"""
        from scripts.generate_scenario import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'scenario.json'
            result = main(['--output', str(output_path)])

            assert result == 0
            assert Path(output_path).exists()

    def test_main_with_frequency(self):
        """Test main function with frequency enabled"""
        from scripts.generate_scenario import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'scenario.json'
            result = main(['--output', str(output_path), '--frequency'])

            assert result == 0
            assert Path(output_path).exists()

            # Verify frequency data is present
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert 'statistics' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
