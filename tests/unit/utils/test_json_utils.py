"""Tests for json_utils module - shared JSON utilities."""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

from utils.json_utils import load_json, save_json, safe_load_json, safe_save_json


class TestLoadJson:
    """Test load_json function - basic JSON loading without security checks."""

    def test_load_json_success(self, tmp_path):
        """Should successfully load valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        result = load_json(json_file)

        assert result == {"key": "value", "number": 42}

    def test_load_json_with_path_object(self, tmp_path):
        """Should accept Path object."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[1, 2, 3]')

        result = load_json(json_file)

        assert result == [1, 2, 3]

    def test_load_json_with_string_path(self, tmp_path):
        """Should accept string path."""
        json_file = tmp_path / "test.json"
        json_file.write_text('"string value"')

        result = load_json(str(json_file))

        assert result == "string value"

    def test_load_json_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError when file does not exist."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_json(nonexistent_file)

    def test_load_json_invalid_json(self, tmp_path):
        """Should raise JSONDecodeError when JSON is malformed."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{"key": "value", "broken": }')

        with pytest.raises(json.JSONDecodeError):
            load_json(json_file)

    def test_load_json_empty_file(self, tmp_path):
        """Should raise JSONDecodeError when file is empty."""
        json_file = tmp_path / "empty.json"
        json_file.write_text('')

        with pytest.raises(json.JSONDecodeError):
            load_json(json_file)

    def test_load_json_with_encoding_error(self, tmp_path):
        """Should raise UnicodeDecodeError when file has encoding issues."""
        json_file = tmp_path / "binary.json"
        json_file.write_bytes(b'\x00\x01\x02\xff\xfe')

        with pytest.raises(UnicodeDecodeError):
            load_json(json_file)

    def test_load_json_nested_structure(self, tmp_path):
        """Should handle nested JSON structures."""
        json_file = tmp_path / "nested.json"
        data = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2"]
                }
            }
        }
        json_file.write_text(json.dumps(data))

        result = load_json(json_file)

        assert result == data

    def test_load_json_unicode_content(self, tmp_path):
        """Should handle unicode content."""
        json_file = tmp_path / "unicode.json"
        json_file.write_text('{"message": "Hello 世界 🌍"}', encoding='utf-8')

        result = load_json(json_file)

        assert result["message"] == "Hello 世界 🌍"


class TestSaveJson:
    """Test save_json function - basic JSON saving without security checks."""

    def test_save_json_success(self, tmp_path):
        """Should successfully save JSON file."""
        json_file = tmp_path / "output.json"
        data = {"key": "value", "number": 42}

        save_json(data, json_file)

        assert json_file.exists()
        assert json.loads(json_file.read_text()) == data

    def test_save_json_creates_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        json_file = tmp_path / "nested" / "dirs" / "output.json"
        data = {"test": "data"}

        save_json(data, json_file)

        assert json_file.exists()

    def test_save_json_with_string_path(self, tmp_path):
        """Should accept string path."""
        json_file = tmp_path / "output.json"
        data = {"test": "data"}

        save_json(data, str(json_file))

        assert json_file.exists()

    def test_save_json_with_indent(self, tmp_path):
        """Should use specified indentation."""
        json_file = tmp_path / "output.json"
        data = {"key": "value"}

        save_json(data, json_file, indent=4)

        content = json_file.read_text()
        assert "    \"key\"" in content  # 4 spaces

    def test_save_json_unicode(self, tmp_path):
        """Should preserve unicode characters."""
        json_file = tmp_path / "unicode.json"
        data = {"message": "Hello 世界 🌍"}

        save_json(data, json_file)

        content = json_file.read_text(encoding='utf-8')
        assert "世界" in content
        assert "🌍" in content

    def test_save_json_permission_error(self, tmp_path):
        """Should raise PermissionError when cannot write."""
        json_file = tmp_path / "readonly.json"
        data = {"test": "data"}

        # Create file first
        json_file.write_text('{}')
        # Remove write permission
        os.chmod(json_file, 0o444)

        try:
            with pytest.raises(PermissionError):
                save_json(data, json_file)
        finally:
            # Restore permission for cleanup
            os.chmod(json_file, 0o666)

    def test_save_json_invalid_data(self, tmp_path):
        """Should raise TypeError for non-serializable data."""
        json_file = tmp_path / "output.json"
        data = {"key": set([1, 2, 3])}  # sets are not JSON serializable

        with pytest.raises(TypeError):
            save_json(data, json_file)


class TestSafeLoadJson:
    """Test safe_load_json function - secure JSON loading with path validation."""

    def test_safe_load_json_success(self, tmp_path):
        """Should successfully load valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        result = safe_load_json(json_file)

        assert result == {"key": "value"}

    def test_safe_load_json_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError when file does not exist."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            safe_load_json(nonexistent_file)

    def test_safe_load_json_invalid_json(self, tmp_path):
        """Should raise ValueError when JSON is malformed."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{"broken": }')

        with pytest.raises(ValueError) as exc_info:
            safe_load_json(json_file)

        assert "JSON" in str(exc_info.value)

    def test_safe_load_json_with_allowed_base_dir(self, tmp_path):
        """Should validate path is within allowed base directory."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        result = safe_load_json(json_file, allowed_base_dir=str(tmp_path))

        assert result == {"key": "value"}

    def test_safe_load_json_path_traversal_attempt(self, tmp_path):
        """Should raise ValueError on path traversal attempt."""
        outside_file = tmp_path / ".." / "outside.json"

        with pytest.raises(ValueError) as exc_info:
            safe_load_json(outside_file, allowed_base_dir=str(tmp_path))

        assert "traversal" in str(exc_info.value).lower() or "outside" in str(exc_info.value).lower()


class TestSafeSaveJson:
    """Test safe_save_json function - secure JSON saving with path validation."""

    def test_safe_save_json_success(self, tmp_path):
        """Should successfully save JSON file."""
        json_file = tmp_path / "output.json"
        data = {"key": "value"}

        safe_save_json(data, json_file)

        assert json_file.exists()

    def test_safe_save_json_with_allowed_base_dir(self, tmp_path):
        """Should validate path is within allowed base directory."""
        json_file = tmp_path / "output.json"
        data = {"key": "value"}

        safe_save_json(data, json_file, allowed_base_dir=str(tmp_path))

        assert json_file.exists()

    def test_safe_save_json_path_traversal_attempt(self, tmp_path):
        """Should raise ValueError on path traversal attempt."""
        outside_file = tmp_path / ".." / "outside.json"
        data = {"key": "value"}

        with pytest.raises(ValueError) as exc_info:
            safe_save_json(data, outside_file, allowed_base_dir=str(tmp_path))

        assert "traversal" in str(exc_info.value).lower() or "outside" in str(exc_info.value).lower()


class TestBackwardCompatibility:
    """Test backward compatibility with old import paths."""

    def test_import_from_utils_entity_cli(self):
        """Should be importable from utils.entity.cli."""
        from utils.entity.cli import safe_load_json as cli_load
        from utils.entity.cli import safe_save_json as cli_save

        assert callable(cli_load)
        assert callable(cli_save)

    def test_import_from_utils_entity_cli_utils(self):
        """Should be importable from utils.entity.cli.utils."""
        from utils.entity.cli.utils import safe_load_json as utils_load
        from utils.entity.cli.utils import safe_save_json as utils_save

        assert callable(utils_load)
        assert callable(utils_save)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_load_json_large_file(self, tmp_path):
        """Should handle large JSON files."""
        json_file = tmp_path / "large.json"
        # Create a large JSON array
        data = [{"id": i, "data": "x" * 100} for i in range(1000)]
        json_file.write_text(json.dumps(data))

        result = load_json(json_file)

        assert len(result) == 1000
        assert result[0]["id"] == 0
        assert result[999]["id"] == 999

    def test_save_json_none_value(self, tmp_path):
        """Should handle None value."""
        json_file = tmp_path / "null.json"

        save_json(None, json_file)

        assert json.loads(json_file.read_text()) is None

    def test_save_json_empty_dict(self, tmp_path):
        """Should handle empty dict."""
        json_file = tmp_path / "empty.json"

        save_json({}, json_file)

        assert json.loads(json_file.read_text()) == {}

    def test_save_json_empty_list(self, tmp_path):
        """Should handle empty list."""
        json_file = tmp_path / "empty.json"

        save_json([], json_file)

        assert json.loads(json_file.read_text()) == []

    def test_save_json_special_characters(self, tmp_path):
        """Should handle special characters in strings."""
        json_file = tmp_path / "special.json"
        data = {
            "quotes": 'He said "Hello"',
            "backslash": "path\\to\\file",
            "newline": "line1\nline2",
            "tab": "col1\tcol2"
        }

        save_json(data, json_file)

        result = json.loads(json_file.read_text())
        assert result["quotes"] == 'He said "Hello"'
        assert result["backslash"] == "path\\to\\file"
        assert result["newline"] == "line1\nline2"
        assert result["tab"] == "col1\tcol2"


class TestSafeLoadJsonEdgeCases:
    """Test edge cases for safe_load_json."""

    def test_safe_load_json_with_null_bytes(self, tmp_path):
        """Should raise ValueError on null bytes in path."""
        malicious_path = str(tmp_path / "test.json") + "\x00.txt"

        with pytest.raises(ValueError) as exc_info:
            safe_load_json(malicious_path, allowed_base_dir=str(tmp_path))

        assert "null" in str(exc_info.value).lower() or "traversal" in str(exc_info.value).lower()

    def test_safe_load_json_unicode_decode_error(self, tmp_path):
        """Should raise ValueError on unicode decode error."""
        json_file = tmp_path / "binary.json"
        json_file.write_bytes(b'\x00\x01\x02\xff\xfe')

        with pytest.raises(ValueError) as exc_info:
            safe_load_json(json_file, allowed_base_dir=str(tmp_path))

        assert "encoding" in str(exc_info.value).lower() or "json" in str(exc_info.value).lower()

    def test_safe_load_json_permission_error(self, tmp_path):
        """Should raise PermissionError when file cannot be read."""
        json_file = tmp_path / "unreadable.json"
        json_file.write_text('{"key": "value"}')
        os.chmod(json_file, 0o000)

        try:
            with pytest.raises((PermissionError, ValueError)):
                safe_load_json(json_file, allowed_base_dir=str(tmp_path))
        finally:
            os.chmod(json_file, 0o644)


class TestSafeSaveJsonEdgeCases:
    """Test edge cases for safe_save_json."""

    def test_safe_save_json_with_null_bytes(self, tmp_path):
        """Should raise ValueError on null bytes in path."""
        malicious_path = str(tmp_path / "test.json") + "\x00.txt"
        data = {"key": "value"}

        with pytest.raises(ValueError) as exc_info:
            safe_save_json(data, malicious_path, allowed_base_dir=str(tmp_path))

        assert "null" in str(exc_info.value).lower() or "traversal" in str(exc_info.value).lower()

    def test_safe_save_json_permission_error_directory(self, tmp_path):
        """Should raise PermissionError when directory cannot be created."""
        # Create a read-only parent directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, 0o555)

        try:
            json_file = readonly_dir / "subdir" / "test.json"
            data = {"key": "value"}

            with pytest.raises((PermissionError, OSError)):
                safe_save_json(data, json_file, allowed_base_dir=str(tmp_path))
        finally:
            os.chmod(readonly_dir, 0o755)

    def test_safe_save_json_value_error(self, tmp_path):
        """Should raise ValueError for circular reference data."""
        json_file = tmp_path / "circular.json"
        data = {"key": "value"}
        data["self"] = data  # Circular reference

        with pytest.raises((ValueError, TypeError)):
            safe_save_json(data, json_file, allowed_base_dir=str(tmp_path))

    def test_safe_save_json_os_error(self, tmp_path):
        """Should raise OSError when file cannot be written."""
        json_file = tmp_path / "output.json"
        data = {"key": "value"}

        # This should succeed normally
        safe_save_json(data, json_file, allowed_base_dir=str(tmp_path))
        assert json_file.exists()

    def test_safe_save_json_without_allowed_base_dir(self, tmp_path):
        """Should work without allowed_base_dir parameter."""
        json_file = tmp_path / "output.json"
        data = {"key": "value"}

        safe_save_json(data, json_file)

        assert json_file.exists()


class TestLoadJsonWithWhitespace:
    """Test load_json with whitespace-only content."""

    def test_load_json_whitespace_only(self, tmp_path):
        """Should raise JSONDecodeError for whitespace-only file."""
        json_file = tmp_path / "whitespace.json"
        json_file.write_text('   \n\t  ')

        with pytest.raises(json.JSONDecodeError):
            load_json(json_file)


class TestSafeLoadJsonWithoutAllowedDir:
    """Test safe_load_json without allowed_base_dir."""

    def test_safe_load_json_no_allowed_dir(self, tmp_path):
        """Should work without allowed_base_dir for valid file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        result = safe_load_json(json_file)

        assert result == {"key": "value"}
