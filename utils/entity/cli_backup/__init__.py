"""Command line interface for scenario configuration tools.

This package provides a refactored CLI structure with modular commands.
All exports maintain backward compatibility with the original cli.py.
"""

# Re-export all commands for backward compatibility
from .main import main
from .commands.lib import lib
from .commands.scenario import init, clone, add_to_scenario, walker
from .commands.validate import validate
from .commands.diff import diff
from .commands.sweep import sweep
from .utils import safe_load_json, safe_save_json

__all__ = [
    "main",
    "lib",
    "init",
    "clone",
    "add_to_scenario",
    "walker",
    "validate",
    "diff",
    "sweep",
    "safe_load_json",
    "safe_save_json",
]
