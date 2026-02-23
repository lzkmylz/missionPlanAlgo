"""Shared utility functions for CLI commands.

This module contains safe file operations and helper functions used across
all CLI command modules. It has no dependencies on Click commands.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click

try:
    from ...path_validator import validate_path_within_allowed_dirs, PathValidationError
except ImportError:
    # Handle relative import issues
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.path_validator import validate_path_within_allowed_dirs, PathValidationError


def _is_path_traversal_attempt(file_path: str) -> bool:
    """
    Check if a path contains directory traversal attempts.

    This function detects common path traversal patterns like:
    - ../ or ..\\ (parent directory references)
    - Null bytes (\x00)

    Args:
        file_path: The path to check

    Returns:
        True if the path contains traversal attempts, False otherwise
    """
    # Check for null bytes
    if '\x00' in file_path:
        return True

    # Check for parent directory references
    # This catches both ../ and ..\ as well as encoded variants
    path = Path(file_path)
    for part in path.parts:
        if part == '..':
            return True

    return False


def _check_symlink_in_path(file_path: str, allowed_dirs: List[str]) -> bool:
    """
    Check if any component in the path is a symlink pointing outside allowed directories.

    Args:
        file_path: The path to check
        allowed_dirs: List of allowed directories (should be absolute paths)

    Returns:
        True if the path is safe (no symlink traversal), False otherwise
    """
    path = Path(file_path)
    # Use absolute() instead of resolve() to avoid resolving symlinks in allowed_dirs
    abs_allowed_dirs = [Path(d).absolute() for d in allowed_dirs]

    # Walk through each component of the path
    for i in range(len(path.parts)):
        partial_path = Path(*path.parts[:i+1])
        if partial_path.is_symlink():
            # Resolve the symlink to see where it points
            symlink_target = partial_path.resolve()
            is_allowed = False
            for allowed_dir in abs_allowed_dirs:
                try:
                    # Check if the symlink target is within the allowed directory
                    # The symlink target should be a subdirectory of allowed_dir
                    symlink_target.relative_to(allowed_dir)
                    is_allowed = True
                    break
                except ValueError:
                    continue
            if not is_allowed:
                return False
    return True


def safe_load_json(file_path: Union[str, Path], allowed_base_dir: Optional[str] = None) -> Any:
    """
    Safely load JSON from a file with path validation and error handling.

    This function prevents directory traversal attacks by:
    1. Checking for null bytes in the path
    2. Checking for parent directory references (..)
    3. Validating that symlinks don't point outside allowed directories

    Args:
        file_path: Path to the JSON file
        allowed_base_dir: Base directory for validation (optional).
                         If not provided, uses file's directory for absolute paths
                         or current working directory for relative paths.

    Returns:
        Parsed JSON data

    Raises:
        click.ClickException: If file not found, path is invalid, or JSON is invalid
    """
    path = Path(file_path)

    # Check for obvious path traversal attempts (e.g., ../)
    if _is_path_traversal_attempt(str(file_path)):
        raise click.ClickException(f"Security error: Path contains directory traversal characters: {file_path}")

    # Determine allowed directories for validation
    # Note: We use absolute() instead of resolve() to avoid resolving symlinks
    # in the path itself. This is important for detecting symlink traversal.
    if allowed_base_dir:
        allowed_dirs = [str(Path(allowed_base_dir).absolute())]
    else:
        # For absolute paths, use the file's parent directory as allowed base
        # For relative paths, use current working directory
        if path.is_absolute():
            allowed_dirs = [str(path.parent.absolute())]
        else:
            allowed_dirs = [str(Path.cwd().absolute())]

    # Check for symlink traversal in the original path
    if not _check_symlink_in_path(str(file_path), allowed_dirs):
        raise click.ClickException(
            f"Security error: Access denied: Path '{file_path}' contains symlink pointing outside allowed directories"
        )

    # Validate path to prevent other directory traversal attacks
    try:
        validated_path = validate_path_within_allowed_dirs(
            str(file_path),
            allowed_directories=allowed_dirs
        )
        path = Path(validated_path)
    except PathValidationError as e:
        raise click.ClickException(f"Security error: {e}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                raise click.ClickException(f"Invalid JSON in {path}: File is empty")
            return json.loads(content)
    except FileNotFoundError:
        raise click.ClickException(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {path}: {e.msg} (line {e.lineno}, column {e.colno})")
    except UnicodeDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {path}: File encoding error - {str(e)}")
    except PermissionError:
        raise click.ClickException(f"Permission denied: Cannot read {path}")


def safe_save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Safely save data to a JSON file with proper error handling.

    Args:
        data: Data to serialize to JSON
        file_path: Path to the output file
        indent: JSON indentation level (default: 2)

    Raises:
        click.ClickException: If data cannot be serialized or file cannot be written
    """
    path = Path(file_path)

    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise click.ClickException(f"Permission denied: Cannot create directory {path.parent}")
    except OSError as e:
        raise click.ClickException(f"Cannot create directory {path.parent}: {str(e)}")

    # Serialize to JSON
    try:
        json_content = json.dumps(data, indent=indent, ensure_ascii=False)
    except TypeError as e:
        raise click.ClickException(f"Cannot serialize data to JSON: {str(e)}")
    except ValueError as e:
        raise click.ClickException(f"Cannot serialize data to JSON: {str(e)}")

    # Write to file
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_content)
    except PermissionError:
        raise click.ClickException(f"Permission denied: Cannot write to {path}")
    except OSError as e:
        raise click.ClickException(f"Cannot write to {path}: {str(e)}")


def _set_nested_value(obj: Dict, path: str, value: Any):
    """Set a nested value using dot notation and bracket notation."""
    parts = []
    current = ""
    in_bracket = False

    for char in path:
        if char == "[":
            if current:
                parts.append(current)
                current = ""
            in_bracket = True
        elif char == "]":
            parts.append(int(current))
            current = ""
            in_bracket = False
        elif char == "." and not in_bracket:
            if current:
                parts.append(current)
                current = ""
        else:
            current += char

    if current:
        parts.append(current)

    # Navigate to parent
    target = obj
    for part in parts[:-1]:
        target = target[part]

    # Set value
    target[parts[-1]] = value


def _get_dict_diff(old: Dict, new: Dict, path: str = "") -> Dict[str, tuple]:
    """Get differences between two dictionaries."""
    diffs = {}
    for key in set(old.keys()) | set(new.keys()):
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                nested_diffs = _get_dict_diff(old_val, new_val, f"{path}{key}.")
                diffs.update(nested_diffs)
            else:
                diffs[f"{path}{key}"] = (old_val, new_val)
    return diffs
