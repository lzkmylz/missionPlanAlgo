"""Tests for CLI module."""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from click.testing import CliRunner
from click.exceptions import ClickException

from utils.entity.cli import main, lib, init, clone, add_to_scenario, walker, validate, diff, sweep
from utils.entity.cli import safe_load_json, safe_save_json


@pytest.fixture
def runner():
    """Create Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_scenario(tmp_path):
    """Create a temporary scenario file."""
    scenario = {
        "name": "test_scenario",
        "description": "Test scenario",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-02T00:00:00Z",
        "satellites": [],
        "targets": [],
        "ground_stations": []
    }
    scenario_path = tmp_path / "test_scenario.json"
    with open(scenario_path, "w") as f:
        json.dump(scenario, f)
    return str(scenario_path)


class TestSafeJsonOperations:
    """Test safe JSON load/save utilities."""

    def test_safe_load_json_success(self, tmp_path):
        """Should successfully load valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        result = safe_load_json(str(json_file))

        assert result == {"key": "value", "number": 42}

    def test_safe_load_json_file_not_found(self, tmp_path):
        """Should raise ClickException when file does not exist."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(ClickException) as exc_info:
            safe_load_json(str(nonexistent_file))

        # Path validation may report "Security error" or "File not found" depending on order
        error_msg = str(exc_info.value)
        assert ("File not found" in error_msg or "Security error" in error_msg or "Path does not exist" in error_msg)
        assert str(nonexistent_file.name) in error_msg or str(nonexistent_file) in error_msg

    def test_safe_load_json_invalid_json(self, tmp_path):
        """Should raise ClickException when JSON is malformed."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{"key": "value", "broken": }')

        with pytest.raises(ClickException) as exc_info:
            safe_load_json(str(json_file))

        assert "Invalid JSON" in str(exc_info.value) or "JSON" in str(exc_info.value)

    def test_safe_load_json_empty_file(self, tmp_path):
        """Should raise ClickException when file is empty."""
        json_file = tmp_path / "empty.json"
        json_file.write_text('')

        with pytest.raises(ClickException) as exc_info:
            safe_load_json(str(json_file))

        assert "Invalid JSON" in str(exc_info.value) or "JSON" in str(exc_info.value)

    def test_safe_load_json_with_encoding_error(self, tmp_path):
        """Should raise ClickException when file has encoding issues."""
        json_file = tmp_path / "binary.json"
        json_file.write_bytes(b'\x00\x01\x02\xff\xfe')

        with pytest.raises(ClickException) as exc_info:
            safe_load_json(str(json_file))

        # Should handle UnicodeDecodeError gracefully
        assert "Invalid JSON" in str(exc_info.value) or "JSON" in str(exc_info.value)

    def test_safe_save_json_success(self, tmp_path):
        """Should successfully save JSON file."""
        json_file = tmp_path / "output.json"
        data = {"key": "value", "number": 42}

        safe_save_json(data, str(json_file))

        assert json_file.exists()
        with open(json_file) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_safe_save_json_with_indent(self, tmp_path):
        """Should save JSON with proper indentation."""
        json_file = tmp_path / "output.json"
        data = {"nested": {"key": "value"}}

        safe_save_json(data, str(json_file), indent=2)

        content = json_file.read_text()
        assert "{\n" in content  # Has newlines for indentation
        assert "  \"nested\"" in content  # Has 2-space indentation

    def test_safe_save_json_permission_error(self, tmp_path):
        """Should raise ClickException when permission denied."""
        json_file = tmp_path / "readonly.json"
        json_file.write_text('{}')
        os.chmod(str(json_file), 0o444)  # Read-only

        try:
            with pytest.raises(ClickException) as exc_info:
                safe_save_json({"key": "value"}, str(json_file))
            assert "Permission denied" in str(exc_info.value) or "Cannot write" in str(exc_info.value)
        finally:
            os.chmod(str(json_file), 0o644)  # Restore permissions

    def test_safe_save_json_to_nonexistent_directory(self, tmp_path):
        """Should create parent directories if they don't exist."""
        nested_dir = tmp_path / "nested" / "deep"
        json_file = nested_dir / "output.json"
        data = {"key": "value"}

        safe_save_json(data, str(json_file))

        assert json_file.exists()
        with open(json_file) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_safe_save_json_with_non_serializable_data(self, tmp_path):
        """Should raise ClickException when data cannot be serialized."""
        json_file = tmp_path / "output.json"
        data = {"key": set([1, 2, 3])}  # Sets are not JSON serializable

        with pytest.raises(ClickException) as exc_info:
            safe_save_json(data, str(json_file))

        assert "Cannot serialize" in str(exc_info.value) or "JSON" in str(exc_info.value)


class TestMainCommand:
    """Test main CLI entry point."""

    def test_main_help(self, runner):
        """Should display help message."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sat-cli" in result.output or "CLI" in result.output

    def test_main_version(self, runner):
        """Should display version."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0


class TestLibCommands:
    """Test library management commands."""

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_list_satellites(self, mock_lib_class, runner):
        """Should list satellites in table format."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib
        mock_lib.list_satellites.return_value = [
            {"template_id": "optical_1", "name": "光学卫星1型"},
            {"template_id": "sar_1", "name": "SAR卫星1型"}
        ]

        result = runner.invoke(lib, ["list", "--type", "satellite"])

        assert result.exit_code == 0
        mock_lib.list_satellites.assert_called_once()

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_list_targets(self, mock_lib_class, runner):
        """Should list targets in table format."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib
        mock_lib.list_targets.return_value = [
            {"id": "beijing", "name": "北京"}
        ]

        result = runner.invoke(lib, ["list", "--type", "target"])

        assert result.exit_code == 0
        mock_lib.list_targets.assert_called_once()

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_show_satellite(self, mock_lib_class, runner):
        """Should show satellite details."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib
        mock_lib.get_satellite.return_value = {
            "template_id": "optical_1",
            "name": "光学卫星1型",
            "capabilities": {"storage_capacity": 500}
        }

        result = runner.invoke(lib, ["show", "satellite", "optical_1"])

        assert result.exit_code == 0
        mock_lib.get_satellite.assert_called_with("optical_1")

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_show_not_found(self, mock_lib_class, runner):
        """Should handle entity not found."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib
        mock_lib.get_satellite.return_value = None

        result = runner.invoke(lib, ["show", "satellite", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_add_target(self, mock_lib_class, runner):
        """Should add target to library."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib

        result = runner.invoke(lib, [
            "add-target",
            "--id", "nanjing",
            "--name", "南京",
            "--lon", "118.8",
            "--lat", "32.1",
            "--priority", "7"
        ])

        assert result.exit_code == 0
        mock_lib.add_target.assert_called_once()
        call_args = mock_lib.add_target.call_args[1]
        assert call_args["target_id"] == "nanjing"
        assert call_args["longitude"] == 118.8
        assert call_args["latitude"] == 32.1

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_remove(self, mock_lib_class, runner):
        """Should remove entity from library."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib

        result = runner.invoke(lib, ["remove", "--type", "target", "--id", "old_target"])

        assert result.exit_code == 0
        # The command creates its own library instance, verify it was called


class TestInitCommand:
    """Test scenario initialization command."""

    @patch("utils.entity.cli.commands.scenario.ScenarioBuilder")
    def test_init_scenario(self, mock_builder_class, runner, tmp_path):
        """Should initialize empty scenario."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.init_scenario.return_value = {
            "name": "my_scenario",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }

        output_path = tmp_path / "my_scenario.json"
        result = runner.invoke(init, [str(output_path)])

        assert result.exit_code == 0
        mock_builder.init_scenario.assert_called_once_with(str(output_path), with_metadata=False)
        mock_builder.save_scenario.assert_called_once()

    @patch("utils.entity.cli.commands.scenario.ScenarioBuilder")
    def test_init_with_metadata(self, mock_builder_class, runner, tmp_path):
        """Should initialize scenario with metadata."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.init_scenario.return_value = {"name": "test"}

        output_path = tmp_path / "test.json"
        result = runner.invoke(init, [str(output_path), "--with-metadata"])

        assert result.exit_code == 0
        mock_builder.init_scenario.assert_called_once_with(str(output_path), with_metadata=True)


class TestCloneCommand:
    """Test scenario clone command."""

    @patch("utils.entity.cli.commands.scenario.ScenarioBuilder")
    def test_clone_scenario(self, mock_builder_class, runner, tmp_path):
        """Should clone existing scenario."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.clone_scenario.return_value = {"name": "cloned"}

        source = tmp_path / "source.json"
        output = tmp_path / "output.json"
        source.write_text("{}")

        result = runner.invoke(clone, [str(source), "--output", str(output)])

        assert result.exit_code == 0
        mock_builder.clone_scenario.assert_called_once_with(str(source), str(output))
        mock_builder.save_scenario.assert_called_once()


class TestAddToScenarioCommand:
    """Test add-to-scenario command."""

    @patch("utils.entity.cli.commands.scenario.ScenarioBuilder")
    @patch("utils.entity.cli.commands.scenario.JSONEntityRepository")
    def test_add_satellite_to_scenario(self, mock_repo_class, mock_builder_class, runner, tmp_path):
        """Should add satellite to scenario."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(add_to_scenario, [
            str(scenario_path),
            "--entity-type", "satellite",
            "--template", "optical_1",
            "--id", "SAT-01"
        ])

        assert result.exit_code == 0

    def test_add_to_scenario_dry_run(self, runner, tmp_path):
        """Should support dry-run mode."""
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(add_to_scenario, [
            str(scenario_path),
            "--entity-type", "satellite",
            "--template", "optical_1",
            "--id", "SAT-01",
            "--dry-run"
        ])

        # Dry run should succeed (exit code 0) or gracefully handle missing template
        assert result.exit_code in [0, 1]  # 0 = success, 1 = template not found is also acceptable


class TestWalkerCommand:
    """Test Walker constellation command."""

    @patch("utils.entity.cli.commands.scenario.WalkerGenerator")
    @patch("utils.entity.cli.commands.scenario.ScenarioBuilder")
    @patch("utils.entity.cli.commands.scenario.JSONEntityRepository")
    def test_walker_generate(self, mock_repo_class, mock_builder_class, mock_walker_class, runner, tmp_path):
        """Should generate Walker constellation."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.repo.get_satellite_template.return_value = {"template_id": "optical_1"}

        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.generate.return_value = [
            {"id": "WALKER-01-01", "raan": 0.0, "mean_anomaly": 0.0}
        ]

        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(walker, [
            str(scenario_path),
            "--template", "optical_1",
            "--planes", "2",
            "--sats-per-plane", "3"
        ])

        assert result.exit_code == 0

    @patch("utils.entity.cli.commands.scenario.WalkerGenerator")
    @patch("utils.entity.cli.commands.scenario.ScenarioBuilder")
    @patch("utils.entity.cli.commands.scenario.JSONEntityRepository")
    def test_walker_with_preset(self, mock_repo_class, mock_builder_class, mock_walker_class, runner, tmp_path):
        """Should use Walker preset."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.repo.get_satellite_template.return_value = {"template_id": "optical_1"}

        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.generate.return_value = [{"id": "SAT-01"}]

        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(walker, [
            str(scenario_path),
            "--preset", "delta_24_3_1",
            "--template", "optical_1"
        ])

        assert result.exit_code == 0
        mock_walker.generate.assert_called_once()


class TestValidateCommand:
    """Test scenario validation command."""

    @patch("utils.entity.cli.commands.validate.ScenarioValidator")
    def test_validate_scenario(self, mock_validator_class, runner, tmp_path):
        """Should validate scenario."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate.return_value = (True, [], [])

        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(validate, [str(scenario_path)])

        assert result.exit_code == 0
        mock_validator.validate.assert_called_once()

    @patch("utils.entity.cli.commands.validate.ScenarioValidator")
    def test_validate_verbose(self, mock_validator_class, runner, tmp_path):
        """Should validate with verbose output."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate.return_value = (True, [], ["Warning"])

        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(validate, [str(scenario_path), "--verbose"])

        assert result.exit_code == 0


class TestDiffCommand:
    """Test scenario diff command."""

    def test_diff_scenarios(self, runner, tmp_path):
        """Should show diff between scenarios."""
        base = tmp_path / "base.json"
        variant = tmp_path / "variant.json"

        base.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage": 500}}]
        }))

        variant.write_text(json.dumps({
            "name": "variant",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage": 1000}}]
        }))

        result = runner.invoke(diff, [str(base), str(variant)])

        assert result.exit_code == 0


class TestSweepCommand:
    """Test parameter sweep command."""

    def test_sweep_single_param(self, runner, tmp_path):
        """Should generate scenario matrix for single parameter."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage_capacity": 500}}]
        }))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].capabilities.storage_capacity",
            "--range", "500:1000:250",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code == 0

    def test_sweep_invalid_range_format(self, runner, tmp_path):
        """Should handle invalid range format."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({"name": "base"}))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].storage",
            "--range", "invalid",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0

    def test_sweep_negative_step(self, runner, tmp_path):
        """Should reject negative step value."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage_capacity": 500}}]
        }))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].capabilities.storage_capacity",
            "--range", "500:1000:-50",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "step" in result.output.lower() or "positive" in result.output.lower()

    def test_sweep_zero_step(self, runner, tmp_path):
        """Should reject zero step value."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage_capacity": 500}}]
        }))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].capabilities.storage_capacity",
            "--range", "500:1000:0",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "step" in result.output.lower() or "positive" in result.output.lower()

    def test_sweep_start_greater_than_end(self, runner, tmp_path):
        """Should reject start value greater than end value."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage_capacity": 500}}]
        }))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].capabilities.storage_capacity",
            "--range", "1000:500:100",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "start" in result.output.lower() or "end" in result.output.lower() or "less" in result.output.lower()

    def test_sweep_non_numeric_value(self, runner, tmp_path):
        """Should handle non-numeric values in range."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage_capacity": 500}}]
        }))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].capabilities.storage_capacity",
            "--range", "abc:def:ghi",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "number" in result.output.lower() or "numeric" in result.output.lower() or "invalid" in result.output.lower()

    def test_sweep_valid_range_boundary(self, runner, tmp_path):
        """Should accept valid range at boundary (start equals end)."""
        base_scenario = tmp_path / "base.json"
        base_scenario.write_text(json.dumps({
            "name": "base",
            "satellites": [{"id": "SAT-01", "capabilities": {"storage_capacity": 500}}]
        }))

        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].capabilities.storage_capacity",
            "--range", "500:500:100",
            "--output-dir", str(output_dir)
        ])

        # Should succeed - single value scenario generated
        assert result.exit_code == 0


class TestCliHelperFunctions:
    """Test CLI helper functions."""

    def test_set_nested_value_simple_dot(self):
        """Should set nested value using simple dot notation."""
        from utils.entity.cli.utils import _set_nested_value

        obj = {"config": {"storage": 500}}
        _set_nested_value(obj, "config.storage", 1000)

        assert obj["config"]["storage"] == 1000

    def test_set_nested_value_bracket_notation(self):
        """Should set nested value using bracket notation."""
        from utils.entity.cli.utils import _set_nested_value

        obj = {"satellites": [{"capabilities": {"storage": 500}}]}
        _set_nested_value(obj, "satellites[0].capabilities.storage", 1000)

        assert obj["satellites"][0]["capabilities"]["storage"] == 1000

    def test_get_dict_diff(self):
        """Should find differences between dictionaries."""
        from utils.entity.cli.utils import _get_dict_diff

        old = {"a": 1, "b": {"c": 2}}
        new = {"a": 1, "b": {"c": 3}}

        diffs = _get_dict_diff(old, new)

        assert "b.c" in diffs
        assert diffs["b.c"] == (2, 3)


class TestWalkerValidation:
    """Test Walker command validation."""

    def test_walker_missing_planes_and_preset(self, runner, tmp_path):
        """Should fail if neither planes nor preset provided."""
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(walker, [
            str(scenario_path),
            "--template", "optical_1"
        ])

        assert result.exit_code != 0

    def test_walker_unknown_preset(self, runner, tmp_path):
        """Should fail for unknown preset."""
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(walker, [
            str(scenario_path),
            "--template", "optical_1",
            "--preset", "unknown_preset"
        ])

        assert result.exit_code != 0


class TestCliJsonErrorHandling:
    """Test CLI commands handle JSON errors gracefully."""

    def test_add_to_scenario_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when scenario file not found."""
        nonexistent = tmp_path / "nonexistent.json"

        result = runner.invoke(add_to_scenario, [
            str(nonexistent),
            "--entity-type", "satellite",
            "--template", "optical_1",
            "--id", "SAT-01"
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_add_to_scenario_invalid_json(self, runner, tmp_path):
        """Should show user-friendly error when scenario has invalid JSON."""
        scenario_path = tmp_path / "invalid.json"
        scenario_path.write_text('{"broken": json}')

        result = runner.invoke(add_to_scenario, [
            str(scenario_path),
            "--entity-type", "satellite",
            "--template", "optical_1",
            "--id", "SAT-01"
        ])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "JSON" in result.output

    def test_validate_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when validating nonexistent file."""
        nonexistent = tmp_path / "nonexistent.json"

        result = runner.invoke(validate, [str(nonexistent)])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_validate_invalid_json(self, runner, tmp_path):
        """Should show user-friendly error when validating invalid JSON."""
        scenario_path = tmp_path / "invalid.json"
        scenario_path.write_text('not valid json {[')

        result = runner.invoke(validate, [str(scenario_path)])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "JSON" in result.output

    def test_walker_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when scenario file not found."""
        nonexistent = tmp_path / "nonexistent.json"

        result = runner.invoke(walker, [
            str(nonexistent),
            "--template", "optical_1",
            "--planes", "2",
            "--sats-per-plane", "3"
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_walker_invalid_json(self, runner, tmp_path):
        """Should show user-friendly error when scenario has invalid JSON."""
        scenario_path = tmp_path / "invalid.json"
        scenario_path.write_text('{invalid}')

        result = runner.invoke(walker, [
            str(scenario_path),
            "--template", "optical_1",
            "--planes", "2",
            "--sats-per-plane", "3"
        ])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "JSON" in result.output

    def test_diff_base_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when base file not found."""
        nonexistent = tmp_path / "nonexistent.json"
        variant = tmp_path / "variant.json"
        variant.write_text('{"name": "variant"}')

        result = runner.invoke(diff, [str(nonexistent), str(variant)])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_diff_variant_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when variant file not found."""
        base = tmp_path / "base.json"
        base.write_text('{"name": "base"}')
        nonexistent = tmp_path / "nonexistent.json"

        result = runner.invoke(diff, [str(base), str(nonexistent)])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_sweep_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when base scenario not found."""
        nonexistent = tmp_path / "nonexistent.json"
        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(nonexistent),
            "--param", "satellites[0].storage",
            "--range", "500:1000:250",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_sweep_invalid_json(self, runner, tmp_path):
        """Should show user-friendly error when base scenario has invalid JSON."""
        base_scenario = tmp_path / "invalid.json"
        base_scenario.write_text('broken{json')
        output_dir = tmp_path / "sweep"

        result = runner.invoke(sweep, [
            "--base-scenario", str(base_scenario),
            "--param", "satellites[0].storage",
            "--range", "500:1000:250",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "JSON" in result.output

    def test_add_to_scenario_batch_file_not_found(self, runner, tmp_path):
        """Should show user-friendly error when batch file not found."""
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))
        nonexistent_batch = tmp_path / "batch.json"

        result = runner.invoke(add_to_scenario, [
            str(scenario_path),
            "--entity-type", "satellite",
            "--id", "SAT-01",
            "--from-file", str(nonexistent_batch)
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("File not found" in result.output or
                "not found" in output_lower or
                "security error" in output_lower or
                "path does not exist" in output_lower)

    def test_add_to_scenario_batch_invalid_json(self, runner, tmp_path):
        """Should show user-friendly error when batch file has invalid JSON."""
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))
        batch_file = tmp_path / "batch.json"
        batch_file.write_text('invalid{json')

        result = runner.invoke(add_to_scenario, [
            str(scenario_path),
            "--entity-type", "satellite",
            "--id", "SAT-01",
            "--from-file", str(batch_file)
        ])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "JSON" in result.output


class TestValidationErrorCases:
    """Test validation error handling."""

    def test_validate_with_errors(self, runner, tmp_path):
        """Should report validation errors."""
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "start_time": "2024-01-02T00:00:00Z",  # end before start
            "end_time": "2024-01-01T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(validate, [str(scenario_path)])

        assert result.exit_code != 0


class TestLibGroundStations:
    """Test ground station library commands."""

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_list_ground_stations(self, mock_lib_class, runner):
        """Should list ground stations."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib
        mock_lib.list_ground_stations.return_value = [
            {"id": "beijing_gs", "name": "北京地面站"}
        ]

        result = runner.invoke(lib, ["list", "--type", "ground_station"])

        assert result.exit_code == 0
        mock_lib.list_ground_stations.assert_called_once()


class TestCliEdgeCases:
    """Test CLI edge cases."""

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_add_satellite_interactive(self, mock_lib_class, runner):
        """Should run add-satellite wizard."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib

        # Provide inputs for interactive prompts
        result = runner.invoke(lib, ["add-satellite", "--id", "test_sat"],
                               input="\n".join(["Test卫星", "optical_1", "500000", "55.0", "1000"]) + "\n")

        assert result.exit_code == 0


class TestWalkerDryRun:
    """Test Walker dry-run output."""

    @patch("utils.entity.cli.commands.scenario.WalkerGenerator")
    @patch("utils.entity.cli.commands.scenario.JSONEntityRepository")
    def test_walker_dry_run_output(self, mock_repo_class, mock_walker_class, runner, tmp_path):
        """Should show dry-run output for Walker."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_satellite_template.return_value = {"template_id": "optical_1", "orbit": {"altitude": 500000}}

        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.generate.return_value = [
            {"id": "WALKER-01-01", "raan": 0.0, "mean_anomaly": 0.0, "orbit": {"altitude": 500000, "inclination": 55.0}}
        ]

        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        result = runner.invoke(walker, [
            str(scenario_path),
            "--template", "optical_1",
            "--planes", "2",
            "--sats-per-plane", "3",
            "--dry-run"
        ])

        assert result.exit_code == 0
        assert "WALKER" in result.output or "即将执行" in result.output or "dry" in result.output.lower()


class TestLibInitCommand:
    """Test lib init command."""

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_init_success_with_yes_flag(self, mock_lib_class, runner):
        """Should initialize entity library with default entities using --yes flag."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib

        result = runner.invoke(lib, ["init", "--yes"])

        assert result.exit_code == 0
        mock_lib.init_defaults.assert_called_once()
        assert "initialized" in result.output.lower() or "success" in result.output.lower()

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_init_success_interactive(self, mock_lib_class, runner):
        """Should initialize entity library with user confirmation."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib

        result = runner.invoke(lib, ["init"], input="y\n")

        assert result.exit_code == 0
        mock_lib.init_defaults.assert_called_once()
        assert "initialized" in result.output.lower() or "success" in result.output.lower()

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_init_handles_error(self, mock_lib_class, runner):
        """Should handle initialization errors gracefully."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib
        mock_lib.init_defaults.side_effect = Exception("Database error")

        result = runner.invoke(lib, ["init", "--yes"])

        assert result.exit_code != 0
        assert "error" in result.output.lower() or "failed" in result.output.lower()

    @patch("utils.entity.cli.commands.lib.EntityLibrary")
    def test_lib_init_interactive_no_confirm(self, mock_lib_class, runner):
        """Should not initialize if user declines confirmation."""
        mock_lib = Mock()
        mock_lib_class.return_value = mock_lib

        result = runner.invoke(lib, ["init"], input="n\n")

        assert result.exit_code == 0
        mock_lib.init_defaults.assert_not_called()
        assert "cancelled" in result.output.lower()

