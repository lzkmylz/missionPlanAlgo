"""Tests for refactored CLI module structure.

These tests verify that after splitting cli.py into multiple modules:
1. All commands are still accessible from the original import paths
2. New module structure works correctly
3. Backward compatibility is maintained
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner


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


class TestBackwardCompatibility:
    """Test that original import paths still work after refactoring."""

    def test_main_import_from_cli_module(self):
        """Should be able to import main from cli module."""
        from utils.entity.cli import main
        assert main is not None
        assert hasattr(main, 'commands') or hasattr(main, 'invoke')

    def test_lib_import_from_cli_module(self):
        """Should be able to import lib group from cli module."""
        from utils.entity.cli import lib
        assert lib is not None

    def test_commands_import_from_cli_module(self):
        """Should be able to import all commands from cli module."""
        from utils.entity.cli import (
            init, clone, add_to_scenario, walker, validate, diff, sweep
        )
        assert init is not None
        assert clone is not None
        assert add_to_scenario is not None
        assert walker is not None
        assert validate is not None
        assert diff is not None
        assert sweep is not None

    def test_utilities_import_from_cli_module(self):
        """Should be able to import utility functions from cli module."""
        from utils.entity.cli import safe_load_json, safe_save_json
        assert safe_load_json is not None
        assert safe_save_json is not None


class TestNewModuleStructure:
    """Test that new module structure works correctly."""

    def test_utils_module_imports(self):
        """Should be able to import from cli.utils module."""
        from utils.entity.cli.utils import safe_load_json, safe_save_json
        assert safe_load_json is not None
        assert safe_save_json is not None

    def test_commands_lib_imports(self):
        """Should be able to import from cli.commands.lib module."""
        from utils.entity.cli.commands.lib import lib
        assert lib is not None

    def test_commands_scenario_imports(self):
        """Should be able to import from cli.commands.scenario module."""
        from utils.entity.cli.commands.scenario import init, clone, add_to_scenario, walker
        assert init is not None
        assert clone is not None
        assert add_to_scenario is not None
        assert walker is not None

    def test_commands_validate_imports(self):
        """Should be able to import from cli.commands.validate module."""
        from utils.entity.cli.commands.validate import validate
        assert validate is not None

    def test_commands_diff_imports(self):
        """Should be able to import from cli.commands.diff module."""
        from utils.entity.cli.commands.diff import diff
        assert diff is not None

    def test_commands_sweep_imports(self):
        """Should be able to import from cli.commands.sweep module."""
        from utils.entity.cli.commands.sweep import sweep
        assert sweep is not None

    def test_main_module_imports(self):
        """Should be able to import from cli.main module."""
        from utils.entity.cli.main import main
        assert main is not None


class TestRefactoredCommandsWork:
    """Test that commands work after refactoring."""

    def test_main_help_after_refactor(self, runner):
        """Main help should work after refactoring."""
        from utils.entity.cli import main
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sat-cli" in result.output or "CLI" in result.output

    def test_lib_list_after_refactor(self, runner):
        """Lib list command should work after refactoring."""
        from utils.entity.cli import lib

        with patch("utils.entity.cli.commands.lib.EntityLibrary") as mock_lib_class:
            mock_lib = Mock()
            mock_lib_class.return_value = mock_lib
            mock_lib.list_satellites.return_value = [
                {"template_id": "optical_1", "name": "光学卫星1型"}
            ]

            result = runner.invoke(lib, ["list", "--type", "satellite"])

            assert result.exit_code == 0
            mock_lib.list_satellites.assert_called_once()

    def test_validate_after_refactor(self, runner, tmp_path):
        """Validate command should work after refactoring."""
        from utils.entity.cli import validate

        with patch("utils.entity.cli.commands.validate.ScenarioValidator") as mock_validator_class:
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

    def test_diff_after_refactor(self, runner, tmp_path):
        """Diff command should work after refactoring."""
        from utils.entity.cli import diff

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


class TestModuleIndependence:
    """Test that modules are properly isolated."""

    def test_utils_no_click_dependency(self):
        """Utils module should not depend on click commands."""
        from utils.entity.cli import utils
        # utils should only contain helper functions, not commands
        assert hasattr(utils, 'safe_load_json')
        assert hasattr(utils, 'safe_save_json')

    def test_commands_import_only_what_needed(self):
        """Command modules should import only what they need."""
        # These should not fail due to circular imports
        from utils.entity.cli.commands import lib as lib_cmd
        from utils.entity.cli.commands import scenario as scenario_cmd
        from utils.entity.cli.commands import validate as validate_cmd
        assert lib_cmd is not None
        assert scenario_cmd is not None
        assert validate_cmd is not None
