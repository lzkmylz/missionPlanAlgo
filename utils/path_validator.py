"""Path validation utilities to prevent directory traversal attacks."""
import os
from pathlib import Path
from typing import List, Optional


class PathValidationError(Exception):
    """Raised when a path fails security validation."""
    pass


def validate_path_within_allowed_dirs(
    file_path: str,
    allowed_directories: List[str],
    allow_directories: bool = False
) -> str:
    """
    Validate that a file path is within allowed directories.

    This function prevents directory traversal attacks by:
    1. Resolving the path to its absolute, canonical form
    2. Checking for null bytes
    3. Verifying the path is within allowed directories
    4. Checking for symlink-based traversal attacks

    Args:
        file_path: The path to validate
        allowed_directories: List of directories that are allowed to be accessed
        allow_directories: Whether to allow the path to be a directory (default: False)

    Returns:
        The resolved absolute path if validation passes

    Raises:
        PathValidationError: If the path fails any security check
    """
    # Check for null bytes (can be used to bypass filters)
    if '\x00' in file_path:
        raise PathValidationError("Path contains null bytes")

    # Validate allowed_directories
    if not allowed_directories:
        raise PathValidationError("No allowed directories specified")

    # Resolve all allowed directories to their canonical forms
    resolved_allowed_dirs = []
    for allowed_dir in allowed_directories:
        allowed_path = Path(allowed_dir).resolve()
        if not allowed_path.exists():
            raise PathValidationError(f"Allowed directory does not exist: {allowed_dir}")
        if not allowed_path.is_dir():
            raise PathValidationError(f"Allowed path is not a directory: {allowed_dir}")
        resolved_allowed_dirs.append(allowed_path)

    # Resolve the target path to its canonical form
    try:
        target_path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise PathValidationError(f"Invalid path: {file_path}") from e

    # Check if path exists
    if not target_path.exists():
        raise PathValidationError(f"Path does not exist: {file_path}")

    # Check if path is a file (unless directories are allowed)
    if not allow_directories and target_path.is_dir():
        raise PathValidationError(f"Path is a directory, not a file: {file_path}")

    # Security check: Verify path is within allowed directories
    # We check both the resolved path and check for symlink traversal
    is_within_allowed = False

    for allowed_dir in resolved_allowed_dirs:
        try:
            # Check if target_path is the same as or is a subpath of allowed_dir
            target_path.relative_to(allowed_dir)
            is_within_allowed = True
            break
        except ValueError:
            # target_path is not within allowed_dir
            continue

    if not is_within_allowed:
        raise PathValidationError(
            f"Access denied: Path '{file_path}' is outside allowed directories"
        )

    # Additional security: Check for symlink traversal
    # Walk up the path and verify no component is a symlink pointing outside allowed dirs
    current = target_path
    while current != current.parent:  # Stop at root
        if current.is_symlink():
            # Resolve the symlink and verify it's within allowed directories
            symlink_target = current.resolve()
            symlink_allowed = False
            for allowed_dir in resolved_allowed_dirs:
                try:
                    symlink_target.relative_to(allowed_dir)
                    symlink_allowed = True
                    break
                except ValueError:
                    continue

            if not symlink_allowed:
                raise PathValidationError(
                    f"Access denied: Symlink '{current}' points outside allowed directories"
                )
        current = current.parent

    return str(target_path)


def validate_scenario_path(
    scenario_path: str,
    base_directory: Optional[str] = None
) -> str:
    """
    Validate a scenario file path.

    If base_directory is not provided, uses the directory of the scenario_path
    as the allowed directory (for backward compatibility with relative paths).

    Args:
        scenario_path: Path to the scenario file
        base_directory: Base directory that must contain the scenario (optional)

    Returns:
        The resolved absolute path if validation passes

    Raises:
        PathValidationError: If validation fails
    """
    path_obj = Path(scenario_path)

    # If path is absolute, we need explicit allowed directories
    if path_obj.is_absolute():
        if base_directory is None:
            raise PathValidationError(
                "Absolute paths require explicit base_directory for validation"
            )
        return validate_path_within_allowed_dirs(
            scenario_path,
            allowed_directories=[base_directory]
        )

    # For relative paths, resolve and check
    if base_directory:
        return validate_path_within_allowed_dirs(
            scenario_path,
            allowed_directories=[base_directory]
        )
    else:
        # Default: use current working directory and the scenario's parent
        resolved = path_obj.resolve()
        parent_dir = str(resolved.parent)
        return validate_path_within_allowed_dirs(
            scenario_path,
            allowed_directories=[parent_dir, str(Path.cwd())]
        )


def get_safe_path(
    file_path: str,
    allowed_directories: List[str]
) -> Optional[Path]:
    """
    Safely get a Path object if validation passes, or None if it fails.

    This is a convenience wrapper that returns None instead of raising an exception.

    Args:
        file_path: The path to validate
        allowed_directories: List of allowed directories

    Returns:
        Resolved Path object if valid, None otherwise
    """
    try:
        resolved = validate_path_within_allowed_dirs(file_path, allowed_directories)
        return Path(resolved)
    except PathValidationError:
        return None
