"""Tests for path validation utilities."""
import pytest
import os
from pathlib import Path
from unittest.mock import patch

from utils.path_validator import validate_path_within_allowed_dirs, PathValidationError


class TestPathValidation:
    """Test path validation functionality."""

    def test_valid_path_within_allowed_dir(self, tmp_path):
        """Should accept valid path within allowed directory."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        target_file = allowed_dir / "scenario.json"
        target_file.write_text('{"test": true}')

        result = validate_path_within_allowed_dirs(
            str(target_file),
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())

    def test_valid_path_in_subdirectory(self, tmp_path):
        """Should accept path in subdirectory of allowed directory."""
        allowed_dir = tmp_path / "allowed"
        subdir = allowed_dir / "subdir"
        subdir.mkdir(parents=True)
        target_file = subdir / "scenario.json"
        target_file.write_text('{"test": true}')

        result = validate_path_within_allowed_dirs(
            str(target_file),
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())

    def test_path_with_directory_traversal(self, tmp_path):
        """Should reject path with directory traversal (../)."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        secret_file = other_dir / "secret.json"
        secret_file.write_text('{"secret": true}')

        # Try to access file outside allowed directory using ../
        malicious_path = allowed_dir / ".." / "other" / "secret.json"

        with pytest.raises(PathValidationError) as exc_info:
            validate_path_within_allowed_dirs(
                str(malicious_path),
                allowed_directories=[str(allowed_dir)]
            )

        assert "outside allowed directories" in str(exc_info.value).lower()

    def test_path_with_traversal_in_middle(self, tmp_path):
        """Should reject path with traversal in the middle of path."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        malicious_path = str(allowed_dir / "subdir" / ".." / ".." / "etc" / "passwd")

        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                malicious_path,
                allowed_directories=[str(allowed_dir)]
            )

    def test_path_with_double_dots_encoded(self, tmp_path):
        """Should reject path with encoded directory traversal."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Test various traversal patterns
        traversal_patterns = [
            ".." + os.sep + "etc" + os.sep + "passwd",
            "..." + os.sep + "etc" + os.sep + "passwd",
            "...." + os.sep + "etc" + os.sep + "passwd",
        ]

        for pattern in traversal_patterns:
            malicious_path = str(allowed_dir / pattern)
            with pytest.raises(PathValidationError):
                validate_path_within_allowed_dirs(
                    malicious_path,
                    allowed_directories=[str(allowed_dir)]
                )

    def test_path_with_symlink_traversal(self, tmp_path):
        """Should reject path that uses symlink to escape allowed directory."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        # Create a symlink inside allowed that points outside
        symlink_path = allowed_dir / "escape"
        symlink_path.symlink_to(other_dir)

        target_path = allowed_dir / "escape" / "secret.json"

        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                str(target_path),
                allowed_directories=[str(allowed_dir)]
            )

    def test_path_with_null_bytes(self, tmp_path):
        """Should reject path with null bytes."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        malicious_path = str(allowed_dir / "file.json") + "\x00.txt"

        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                malicious_path,
                allowed_directories=[str(allowed_dir)]
            )

    def test_path_with_absolute_path_traversal(self, tmp_path):
        """Should reject absolute path outside allowed directories."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Absolute path to system file
        malicious_path = "/etc/passwd"

        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                malicious_path,
                allowed_directories=[str(allowed_dir)]
            )

    def test_multiple_allowed_directories(self, tmp_path):
        """Should accept path in any of the allowed directories."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        target_file = dir2 / "scenario.json"
        target_file.write_text('{"test": true}')

        result = validate_path_within_allowed_dirs(
            str(target_file),
            allowed_directories=[str(dir1), str(dir2)]
        )

        assert result == str(target_file.resolve())

    def test_file_does_not_exist(self, tmp_path):
        """Should raise error if file does not exist."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        nonexistent_file = allowed_dir / "nonexistent.json"

        with pytest.raises(PathValidationError) as exc_info:
            validate_path_within_allowed_dirs(
                str(nonexistent_file),
                allowed_directories=[str(allowed_dir)]
            )

        assert "does not exist" in str(exc_info.value).lower()

    def test_path_is_directory(self, tmp_path):
        """Should optionally allow or reject directories based on parameter."""
        allowed_dir = tmp_path / "allowed"
        subdir = allowed_dir / "subdir"
        subdir.mkdir(parents=True)

        # By default, directories should be rejected
        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                str(subdir),
                allowed_directories=[str(allowed_dir)]
            )

        # With allow_directories=True, should be accepted
        result = validate_path_within_allowed_dirs(
            str(subdir),
            allowed_directories=[str(allowed_dir)],
            allow_directories=True
        )

        assert result == str(subdir.resolve())

    def test_empty_allowed_directories(self, tmp_path):
        """Should raise error if no allowed directories specified."""
        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                "/some/path.json",
                allowed_directories=[]
            )

    def test_allowed_directory_does_not_exist(self, tmp_path):
        """Should raise error if allowed directory does not exist."""
        nonexistent_dir = tmp_path / "nonexistent"

        with pytest.raises(PathValidationError):
            validate_path_within_allowed_dirs(
                str(tmp_path / "file.json"),
                allowed_directories=[str(nonexistent_dir)]
            )

    def test_normalize_path_before_check(self, tmp_path):
        """Should normalize paths before security check."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        target_file = allowed_dir / "scenario.json"
        target_file.write_text('{"test": true}')

        # Path with redundant components that resolves to allowed directory
        complex_path = allowed_dir / "subdir" / ".." / "scenario.json"

        result = validate_path_within_allowed_dirs(
            str(complex_path),
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())

    def test_path_with_dot_slash(self, tmp_path):
        """Should handle paths with ./ correctly."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        target_file = allowed_dir / "scenario.json"
        target_file.write_text('{"test": true}')

        # Path with ./ prefix
        dot_slash_path = str(allowed_dir / "." / "scenario.json")

        result = validate_path_within_allowed_dirs(
            dot_slash_path,
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())


class TestPathValidationEdgeCases:
    """Test edge cases for path validation."""

    def test_very_long_path(self, tmp_path):
        """Should handle very long paths."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create deeply nested directory structure
        deep_path = allowed_dir
        for i in range(50):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)

        target_file = deep_path / "file.json"
        target_file.write_text('{"test": true}')

        result = validate_path_within_allowed_dirs(
            str(target_file),
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())

    def test_path_with_unicode(self, tmp_path):
        """Should handle paths with unicode characters."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        unicode_dir = allowed_dir / "中文目录"
        unicode_dir.mkdir()
        target_file = unicode_dir / "文件.json"
        target_file.write_text('{"test": true}')

        result = validate_path_within_allowed_dirs(
            str(target_file),
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())

    def test_path_with_special_chars(self, tmp_path):
        """Should handle paths with special characters."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        special_dir = allowed_dir / "dir-with_spaces_and-dots"
        special_dir.mkdir()
        target_file = special_dir / "file.name.json"
        target_file.write_text('{"test": true}')

        result = validate_path_within_allowed_dirs(
            str(target_file),
            allowed_directories=[str(allowed_dir)]
        )

        assert result == str(target_file.resolve())

    def test_relative_path_resolution(self, tmp_path):
        """Should resolve relative paths correctly."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        target_file = allowed_dir / "scenario.json"
        target_file.write_text('{"test": true}')

        # Change to allowed directory and use relative path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(allowed_dir))
            result = validate_path_within_allowed_dirs(
                "scenario.json",
                allowed_directories=[str(allowed_dir)]
            )
            assert result == str(target_file.resolve())
        finally:
            os.chdir(original_cwd)

    def test_case_sensitivity_on_case_insensitive_fs(self, tmp_path):
        """Should handle case sensitivity appropriately."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        target_file = allowed_dir / "SCENARIO.JSON"
        target_file.write_text('{"test": true}')

        # Try lowercase version (may or may not work depending on filesystem)
        try:
            result = validate_path_within_allowed_dirs(
                str(allowed_dir / "scenario.json"),
                allowed_directories=[str(allowed_dir)]
            )
            # If it works, the result should point to the file
            assert Path(result).exists()
        except PathValidationError:
            # On case-sensitive filesystems, this is expected
            pass
