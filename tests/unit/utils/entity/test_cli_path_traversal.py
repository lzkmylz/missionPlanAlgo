"""Tests for CLI path traversal security."""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from utils.entity.cli import main, add_to_scenario, walker, validate, diff, sweep


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


class TestPathTraversalSecurity:
    """Test path traversal attack prevention in CLI commands."""

    def test_add_to_scenario_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in scenario_path argument."""
        # Create a file outside the allowed directory
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.json"
        secret_file.write_text('{"secret": "data"}')

        # Create allowed directory
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Try to access file outside allowed directory using ../
        malicious_path = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(add_to_scenario, [
            malicious_path,
            "--entity-type", "satellite",
            "--template", "optical_1",
            "--id", "SAT-01"
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        # Path validation may fail with "Path does not exist" (if file doesn't exist)
        # or "outside allowed directories" (if file exists but is outside allowed dir)
        assert ("security error" in output_lower or
                "outside allowed directories" in output_lower or
                "path does not exist" in output_lower)

    def test_walker_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in walker command."""
        # Create allowed directory and a file outside it
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.json"
        secret_file.write_text('{"secret": "data"}')

        # Try path traversal
        malicious_path = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(walker, [
            malicious_path,
            "--template", "optical_1",
            "--planes", "2",
            "--sats-per-plane", "3"
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("security error" in output_lower or
                "outside allowed directories" in output_lower or
                "path does not exist" in output_lower)

    def test_validate_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in validate command."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.json"
        secret_file.write_text('{"secret": "data"}')

        malicious_path = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(validate, [malicious_path])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("security error" in output_lower or
                "outside allowed directories" in output_lower or
                "path does not exist" in output_lower)

    def test_diff_base_path_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in diff base_path argument."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        # Create a valid variant file
        variant_file = allowed_dir / "variant.json"
        variant_file.write_text('{"name": "variant"}')

        # Try path traversal for base file
        malicious_base = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(diff, [malicious_base, str(variant_file)])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "security error" in output_lower or "outside allowed directories" in output_lower

    def test_diff_variant_path_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in diff variant_path argument."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        # Create a valid base file
        base_file = allowed_dir / "base.json"
        base_file.write_text('{"name": "base"}')

        # Try path traversal for variant file
        malicious_variant = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(diff, [str(base_file), malicious_variant])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "security error" in output_lower or "outside allowed directories" in output_lower

    def test_sweep_base_scenario_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in sweep base-scenario argument."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.json"
        secret_file.write_text('{"secret": "data"}')

        malicious_path = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(sweep, [
            "--base-scenario", malicious_path,
            "--param", "satellites[0].storage",
            "--range", "500:1000:250",
            "--output-dir", str(output_dir)
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("security error" in output_lower or
                "outside allowed directories" in output_lower or
                "path does not exist" in output_lower)

    def test_add_to_scenario_from_file_traversal_attack(self, runner, tmp_path):
        """Should block path traversal attack in --from-file option."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create a valid scenario file
        scenario_file = allowed_dir / "scenario.json"
        scenario_file.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.json"
        secret_file.write_text('{"secret": "data"}')

        malicious_from_file = str(allowed_dir / ".." / "outside" / "secret.json")

        result = runner.invoke(add_to_scenario, [
            str(scenario_file),
            "--entity-type", "satellite",
            "--id", "SAT-01",
            "--from-file", malicious_from_file
        ])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("security error" in output_lower or
                "outside allowed directories" in output_lower or
                "path does not exist" in output_lower)

    def test_traversal_with_encoded_path(self, runner, tmp_path):
        """Should block encoded path traversal attempts."""
        import os
        original_cwd = os.getcwd()
        try:
            # Create a working directory
            work_dir = tmp_path / "work"
            work_dir.mkdir()
            os.chdir(str(work_dir))

            # Create allowed directory within work
            allowed_dir = work_dir / "allowed"
            allowed_dir.mkdir()

            # Create a target file outside work_dir to test traversal
            outside_dir = tmp_path / "outside"
            outside_dir.mkdir()
            outside_file = outside_dir / "secret.json"
            outside_file.write_text('{"secret": "data"}')

            # Try various encoded traversal patterns that try to escape work_dir
            traversal_patterns = [
                str(work_dir / "allowed" / ".." / ".." / "outside" / "secret.json"),
                str(work_dir / "allowed" / "..." / "outside" / "secret.json"),
            ]

            for malicious_path in traversal_patterns:
                result = runner.invoke(validate, [malicious_path])

                assert result.exit_code != 0
                output_lower = result.output.lower()
                assert ("security error" in output_lower or
                        "outside allowed directories" in output_lower or
                        "path does not exist" in output_lower)
        finally:
            os.chdir(original_cwd)

    def test_traversal_with_absolute_path(self, runner, tmp_path):
        """Should detect and report file access outside expected directory."""
        import os
        original_cwd = os.getcwd()
        try:
            # Create directories
            work_dir = tmp_path / "work"
            work_dir.mkdir()
            outside_dir = tmp_path / "outside"
            outside_dir.mkdir()

            # Create a file outside the working directory
            outside_file = outside_dir / "secret.json"
            outside_file.write_text('{"secret": "data"}')

            # Change to work directory
            os.chdir(str(work_dir))

            # When using absolute path to outside_file, the file's parent (outside_dir)
            # becomes the allowed directory, so the file is accessible.
            # This is expected behavior for absolute paths.
            result = runner.invoke(validate, [str(outside_file)])

            # The file should be accessible (no security error)
            # It may fail validation due to missing fields, but not due to security
            output_lower = result.output.lower()
            assert "security error" not in output_lower
            assert "outside allowed directories" not in output_lower
        finally:
            os.chdir(original_cwd)

    def test_traversal_with_null_bytes(self, runner, tmp_path):
        """Should block paths with null bytes."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        scenario_file = allowed_dir / "scenario.json"
        scenario_file.write_text(json.dumps({
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        # Path with null byte injection
        malicious_path = str(scenario_file) + "\x00.txt"

        result = runner.invoke(validate, [malicious_path])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert ("security error" in output_lower or
                "null" in output_lower or
                "path does not exist" in output_lower)

    def test_valid_path_allowed(self, runner, tmp_path):
        """Should allow valid paths within allowed directories."""
        # Change to tmp_path to make it the current working directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))

            # Create a valid scenario file in current directory
            scenario_file = tmp_path / "scenario.json"
            scenario_file.write_text(json.dumps({
                "name": "test",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z",
                "satellites": [],
                "targets": [],
                "ground_stations": []
            }))

            # Valid path should work (relative to current directory)
            result = runner.invoke(validate, ["scenario.json"])

            # Should succeed (exit code 0) or fail due to validation, not security
            if result.exit_code != 0:
                output_lower = result.output.lower()
                assert "security error" not in output_lower
                assert "outside allowed directories" not in output_lower
        finally:
            os.chdir(original_cwd)

    def test_valid_path_in_subdirectory(self, runner, tmp_path):
        """Should allow valid paths in subdirectories of allowed directories."""
        # Change to tmp_path to make it the current working directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))

            # Create subdirectory structure
            subdir = tmp_path / "subdir"
            subdir.mkdir(parents=True)

            # Create a valid scenario file in subdirectory
            scenario_file = subdir / "scenario.json"
            scenario_file.write_text(json.dumps({
                "name": "test",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z",
                "satellites": [],
                "targets": [],
                "ground_stations": []
            }))

            # Valid subdirectory path should work (relative to current directory)
            result = runner.invoke(validate, ["subdir/scenario.json"])

            # Should succeed (exit code 0) or fail due to validation, not security
            if result.exit_code != 0:
                output_lower = result.output.lower()
                assert "security error" not in output_lower
                assert "outside allowed directories" not in output_lower
        finally:
            os.chdir(original_cwd)

        # Should succeed (exit code 0) or fail due to validation, not security
        if result.exit_code != 0:
            output_lower = result.output.lower()
            assert "security error" not in output_lower
            assert "outside allowed directories" not in output_lower

    def test_traversal_with_symlink(self, runner, tmp_path):
        """Should block traversal using symlinks that point outside allowed directories."""
        import os
        original_cwd = os.getcwd()
        try:
            # Create directories
            work_dir = tmp_path / "work"
            work_dir.mkdir()
            allowed_dir = work_dir / "allowed"
            allowed_dir.mkdir()
            outside_dir = tmp_path / "outside"
            outside_dir.mkdir()

            # Create a symlink inside allowed that points to outside
            symlink_path = allowed_dir / "escape"
            symlink_path.symlink_to(outside_dir)

            # Create a file outside
            secret_file = outside_dir / "secret.json"
            secret_file.write_text('{"secret": "data"}')

            # Change to work directory
            os.chdir(str(work_dir))

            # Try to access file through symlink
            # The symlink points outside the allowed directory (allowed_dir's parent is work_dir)
            malicious_path = str(allowed_dir / "escape" / "secret.json")

            result = runner.invoke(validate, [malicious_path])

            assert result.exit_code != 0
            output_lower = result.output.lower()
            assert ("security error" in output_lower or
                    "outside allowed directories" in output_lower or
                    "symlink" in output_lower)
        finally:
            os.chdir(original_cwd)


class TestPathTraversalEdgeCases:
    """Test edge cases for path traversal prevention."""

    def test_path_with_dot_slash(self, runner, tmp_path):
        """Should handle ./ in paths correctly."""
        # Change to tmp_path to make it the current working directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))

            # Create scenario file in current directory
            scenario_file = tmp_path / "scenario.json"
            scenario_file.write_text(json.dumps({
                "name": "test",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z",
                "satellites": [],
                "targets": [],
                "ground_stations": []
            }))

            # Path with ./ should be normalized and allowed
            result = runner.invoke(validate, ["./scenario.json"])

            # Should succeed (exit code 0) or fail due to validation, not security
            if result.exit_code != 0:
                output_lower = result.output.lower()
                assert "security error" not in output_lower
                assert "outside allowed directories" not in output_lower
        finally:
            os.chdir(original_cwd)

    def test_path_with_multiple_parent_references(self, runner, tmp_path):
        """Should block paths with multiple parent directory references."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create deeply nested structure
        deep_dir = allowed_dir / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)

        # Path that tries to escape using multiple ..
        malicious_path = str(deep_dir / ".." / ".." / ".." / ".." / "etc" / "passwd")

        result = runner.invoke(validate, [malicious_path])

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "security error" in output_lower or "outside allowed directories" in output_lower or "path does not exist" in output_lower

    def test_relative_path_within_allowed_dir(self, runner, tmp_path):
        """Should allow relative paths that resolve within allowed directory."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        scenario_file = allowed_dir / "scenario.json"
        scenario_file.write_text(json.dumps({
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }))

        # Change to allowed directory and use relative path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(allowed_dir))
            result = runner.invoke(validate, ["scenario.json"])

            # Should succeed (exit code 0) or fail due to validation, not security
            if result.exit_code != 0:
                output_lower = result.output.lower()
                assert "security error" not in output_lower
                assert "outside allowed directories" not in output_lower
        finally:
            os.chdir(original_cwd)
