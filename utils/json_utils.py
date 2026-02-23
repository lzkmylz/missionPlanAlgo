"""Shared JSON utility functions.

This module provides safe and basic JSON loading/saving operations
used across the entire project.
"""
import json
import os
from pathlib import Path
from typing import Any, Optional, Union


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load JSON from a file.

    Args:
        file_path: Path to the JSON file (string or Path object)

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If JSON is malformed
        UnicodeDecodeError: If file has encoding issues
    """
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        if not content.strip():
            raise json.JSONDecodeError("File is empty", content, 0)
        return json.loads(content)


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to serialize to JSON
        file_path: Path to the output file
        indent: JSON indentation level (default: 2)

    Raises:
        TypeError: If data cannot be serialized to JSON
        PermissionError: If file cannot be written
        OSError: If directory cannot be created
    """
    path = Path(file_path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize and write
    json_content = json.dumps(data, indent=indent, ensure_ascii=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_content)


def _is_path_traversal_attempt(file_path: str) -> bool:
    """
    Check if a path contains directory traversal attempts.

    Args:
        file_path: The path to check

    Returns:
        True if the path contains traversal attempts, False otherwise
    """
    # Check for null bytes
    if '\x00' in file_path:
        return True

    # Check for parent directory references
    path = Path(file_path)
    for part in path.parts:
        if part == '..':
            return True

    return False


def _validate_path_within_allowed_dir(
    file_path: Union[str, Path],
    allowed_base_dir: Optional[str] = None
) -> Path:
    """
    Validate that a path is within an allowed base directory.

    Args:
        file_path: The path to validate
        allowed_base_dir: Base directory that the path must be within

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path traversal is detected or path is outside allowed dir
    """
    path = Path(file_path)

    # Check for obvious path traversal
    if _is_path_traversal_attempt(str(file_path)):
        raise ValueError(f"Security error: Path contains directory traversal characters: {file_path}")

    # Resolve to absolute path
    abs_path = path.absolute()

    if allowed_base_dir:
        abs_allowed = Path(allowed_base_dir).absolute()
        try:
            # Check if path is within allowed directory
            abs_path.relative_to(abs_allowed)
        except ValueError:
            raise ValueError(
                f"Security error: Path '{file_path}' is outside allowed directory '{allowed_base_dir}'"
            )

    return abs_path


def safe_load_json(
    file_path: Union[str, Path],
    allowed_base_dir: Optional[str] = None
) -> Any:
    """
    Safely load JSON from a file with path validation.

    This function prevents directory traversal attacks by validating
    that the file path is within an allowed base directory.

    Args:
        file_path: Path to the JSON file
        allowed_base_dir: Base directory for validation (optional).
                         If not provided, no directory restriction is enforced.

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If path is invalid (traversal attempt) or JSON is invalid
        json.JSONDecodeError: If JSON is malformed
    """
    # Validate path
    path = _validate_path_within_allowed_dir(file_path, allowed_base_dir)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Invalid JSON in {path}: File is empty")
            return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: File encoding error - {str(e)}")


def safe_save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    allowed_base_dir: Optional[str] = None
) -> None:
    """
    Safely save data to a JSON file with path validation.

    This function prevents directory traversal attacks by validating
    that the file path is within an allowed base directory.

    Args:
        data: Data to serialize to JSON
        file_path: Path to the output file
        indent: JSON indentation level (default: 2)
        allowed_base_dir: Base directory for validation (optional)

    Raises:
        ValueError: If path is invalid (traversal attempt)
        TypeError: If data cannot be serialized to JSON
        PermissionError: If file cannot be written
    """
    # Validate path
    path = _validate_path_within_allowed_dir(file_path, allowed_base_dir)

    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied: Cannot create directory {path.parent}") from e
    except OSError as e:
        raise OSError(f"Cannot create directory {path.parent}: {str(e)}") from e

    # Serialize to JSON
    try:
        json_content = json.dumps(data, indent=indent, ensure_ascii=False)
    except TypeError as e:
        raise TypeError(f"Cannot serialize data to JSON: {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Cannot serialize data to JSON: {str(e)}") from e

    # Write to file
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_content)
    except PermissionError as e:
        raise PermissionError(f"Permission denied: Cannot write to {path}") from e
    except OSError as e:
        raise OSError(f"Cannot write to {path}: {str(e)}") from e
