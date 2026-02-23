"""Tests for multi-parameter sweep functionality.

This module tests the Cartesian product sweep feature for generating
scenario matrices with multiple varying parameters.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from utils.entity.cli.commands.sweep import sweep


class TestCartesianProductGeneration:
    """Test Cartesian product generation for multi-parameter sweeps."""

    def test_two_parameter_cartesian_product(self, tmp_path):
        """Should generate correct number of scenarios for 2 parameters."""
        # Create base scenario
        base = {
            "name": "test_scenario",
            "param1": 0,
            "param2": 0
        }
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        # param1: 1, 2, 3 (3 values)
        # param2: 10, 20 (2 values)
        # Total: 3 * 2 = 6 scenarios
        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "param1",
            "--param", "param2",
            "--range", "1:3:1",
            "--range", "10:20:10",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        # Check 6 scenarios generated
        files = list(output_dir.glob("*.json"))
        assert len(files) == 6

    def test_three_parameter_cartesian_product(self, tmp_path):
        """Should generate correct number of scenarios for 3 parameters."""
        base = {
            "name": "test_scenario",
            "a": 0, "b": 0, "c": 0
        }
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        # a: 1, 2 (2 values)
        # b: 10, 20 (2 values)
        # c: 100, 200 (2 values)
        # Total: 2 * 2 * 2 = 8 scenarios
        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--param", "b",
            "--param", "c",
            "--range", "1:2:1",
            "--range", "10:20:10",
            "--range", "100:200:100",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 8

    def test_all_parameter_combinations_present(self, tmp_path):
        """Should include all combinations of parameter values."""
        base = {
            "name": "test_scenario",
            "resolution": 0,
            "angle": 0
        }
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "resolution",
            "--param", "angle",
            "--range", "1:2:1",
            "--range", "30:60:30",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        # Collect all parameter combinations
        combinations = set()
        for f in output_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
                combinations.add((data["resolution"], data["angle"]))

        # Should have all 4 combinations
        expected = {(1.0, 30.0), (1.0, 60.0), (2.0, 30.0), (2.0, 60.0)}
        assert combinations == expected


class TestNestedParameterSweep:
    """Test sweeping nested parameters using dot notation."""

    def test_nested_parameter_sweep(self, tmp_path):
        """Should sweep nested parameters correctly."""
        base = {
            "name": "test_scenario",
            "satellites": [{"capabilities": {"storage": 100}}]
        }
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "satellites[0].capabilities.storage",
            "--range", "100:300:100",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 3

        # Verify values are set correctly
        values = set()
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
                values.add(data["satellites"][0]["capabilities"]["storage"])

        assert values == {100.0, 200.0, 300.0}

    def test_multiple_nested_parameters(self, tmp_path):
        """Should sweep multiple nested parameters."""
        base = {
            "name": "test_scenario",
            "satellites": [
                {"capabilities": {"storage": 100, "resolution": 10}}
            ]
        }
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "satellites[0].capabilities.storage",
            "--param", "satellites[0].capabilities.resolution",
            "--range", "100:200:100",
            "--range", "10:20:10",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 4


class TestSweepValidation:
    """Test input validation for sweep command."""

    def test_mismatched_param_range_count(self, tmp_path):
        """Should error when param and range counts don't match."""
        base = {"name": "test", "a": 0, "b": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        # 2 params but only 1 range
        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--param", "b",
            "--range", "1:3:1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "match" in result.output.lower() or "number" in result.output.lower()

    def test_zero_step_validation(self, tmp_path):
        """Should reject zero step value."""
        base = {"name": "test", "a": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--range", "1:3:0",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "step" in result.output.lower()

    def test_negative_step_validation(self, tmp_path):
        """Should reject negative step value."""
        base = {"name": "test", "a": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--range", "3:1:-1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "step" in result.output.lower()


class TestSweepOutput:
    """Test output file generation and naming."""

    def test_scenario_names_are_unique(self, tmp_path):
        """Should generate unique names for each scenario."""
        base = {"name": "test_scenario", "a": 0, "b": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--param", "b",
            "--range", "1:2:1",
            "--range", "10:20:10",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        names = [f.stem for f in files]

        # All names should be unique
        assert len(names) == len(set(names))

    def test_output_contains_all_base_fields(self, tmp_path):
        """Should preserve all fields from base scenario."""
        base = {
            "name": "test_scenario",
            "description": "Test description",
            "satellites": [{"id": "SAT-1"}],
            "targets": [{"id": "TGT-1"}],
            "param": 0
        }
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "param",
            "--range", "1:2:1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        # Check first output file
        files = list(output_dir.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)

        assert data["name"] == "test_scenario"
        assert data["description"] == "Test description"
        assert data["satellites"] == [{"id": "SAT-1"}]
        assert data["targets"] == [{"id": "TGT-1"}]


class TestSweepProgressOutput:
    """Test progress display during sweep."""

    def test_shows_generation_count(self, tmp_path):
        """Should show number of scenarios generated."""
        base = {"name": "test", "a": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--range", "1:5:1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0
        assert "5" in result.output or "five" in result.output.lower()

    def test_shows_multi_param_summary(self, tmp_path):
        """Should show summary for multi-parameter sweep."""
        base = {"name": "test", "a": 0, "b": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--param", "b",
            "--range", "1:3:1",
            "--range", "10:20:10",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0
        # Should indicate 6 scenarios (3 * 2)
        assert "6" in result.output


class TestSweepEdgeCases:
    """Test edge cases for sweep functionality."""

    def test_single_value_range(self, tmp_path):
        """Should handle range with single value."""
        base = {"name": "test", "a": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        # Range 5:5:1 produces only value 5
        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--range", "5:5:1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)
        assert data["a"] == 5.0

    def test_float_step_values(self, tmp_path):
        """Should handle float step values."""
        base = {"name": "test", "a": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        # Range 0:1:0.5 produces 0, 0.5, 1.0
        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--range", "0:1:0.5",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 3

    def test_large_number_of_combinations(self, tmp_path):
        """Should handle large Cartesian products efficiently."""
        base = {"name": "test", "a": 0, "b": 0, "c": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        # 5 * 5 * 5 = 125 scenarios
        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "a",
            "--param", "b",
            "--param", "c",
            "--range", "1:5:1",
            "--range", "1:5:1",
            "--range", "1:5:1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 125

    def test_empty_base_scenario(self, tmp_path):
        """Should handle minimal base scenario."""
        base = {"name": "test"}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "new_param",
            "--range", "1:3:1",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        assert len(files) == 3


class TestSweepFileNaming:
    """Test scenario file naming conventions."""

    def test_naming_includes_param_values(self, tmp_path):
        """File names should reflect parameter values."""
        base = {"name": "mission", "res": 0, "angle": 0}
        base_path = tmp_path / "base.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "output"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "res",
            "--param", "angle",
            "--range", "1:2:1",
            "--range", "30:60:30",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        files = list(output_dir.glob("*.json"))
        names = [f.stem for f in files]

        # Names should be descriptive and unique
        assert all("mission" in name for name in names)
        assert len(set(names)) == len(names)


class TestSweepIntegration:
    """Integration tests for complete sweep workflows."""

    def test_full_sweep_workflow(self, tmp_path):
        """Complete workflow from base scenario to output files."""
        # Create realistic base scenario
        base = {
            "name": "imaging_mission",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {
                    "id": "SAT-01",
                    "capabilities": {
                        "resolution": 10,
                        "swath_width": 100
                    }
                }
            ],
            "targets": [
                {"id": "TGT-01", "priority": 1}
            ]
        }
        base_path = tmp_path / "mission.json"
        with open(base_path, "w") as f:
            json.dump(base, f)

        runner = CliRunner()
        output_dir = tmp_path / "sweep_results"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_path),
            "--param", "satellites[0].capabilities.resolution",
            "--param", "satellites[0].capabilities.swath_width",
            "--range", "5:15:5",
            "--range", "50:150:50",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

        # Verify output
        files = list(output_dir.glob("*.json"))
        assert len(files) == 9  # 3 * 3

        # Verify each file is valid JSON with correct structure
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
                assert data["name"] == "imaging_mission"
                assert "satellites" in data
                assert data["satellites"][0]["capabilities"]["resolution"] in [5.0, 10.0, 15.0]
                assert data["satellites"][0]["capabilities"]["swath_width"] in [50.0, 100.0, 150.0]
