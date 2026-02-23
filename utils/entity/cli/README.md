# CLI Module Structure

This directory contains the refactored CLI implementation for the satellite scenario configuration tool.

## Directory Structure

```
cli/
├── __init__.py          # Package exports (backward compatible)
├── main.py              # Main CLI entry point and command registration
├── utils.py             # Shared utility functions (safe_load_json, safe_save_json, etc.)
└── commands/
    ├── __init__.py      # Commands package marker
    ├── lib.py           # Entity library commands (list, show, add-satellite, add-target, remove, init)
    ├── scenario.py      # Scenario management commands (init, clone, add-to-scenario, walker)
    ├── validate.py      # Validation command (validate)
    ├── diff.py          # Diff command (diff)
    └── sweep.py         # Parameter sweep command (sweep)
```

## Backward Compatibility

All imports from the original `cli.py` module continue to work:

```python
# These imports still work
from utils.entity.cli import main, lib, init, clone, add_to_scenario, walker, validate, diff, sweep
from utils.entity.cli import safe_load_json, safe_save_json
```

## New Module Imports

You can also import from the new submodules:

```python
# From utils module
from utils.entity.cli.utils import safe_load_json, safe_save_json

# From command modules
from utils.entity.cli.commands.lib import lib
from utils.entity.cli.commands.scenario import init, clone, add_to_scenario, walker
from utils.entity.cli.commands.validate import validate
from utils.entity.cli.commands.diff import diff
from utils.entity.cli.commands.sweep import sweep

# From main module
from utils.entity.cli.main import main
```

## File Size Summary

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 28 | Package exports |
| `main.py` | 33 | CLI entry point |
| `utils.py` | 245 | Shared utilities |
| `commands/lib.py` | 198 | Library commands |
| `commands/scenario.py` | 192 | Scenario commands |
| `commands/validate.py` | 82 | Validate command |
| `commands/diff.py` | 74 | Diff command |
| `commands/sweep.py` | 77 | Sweep command |

Original `cli.py`: 602 lines
Refactored total: 933 lines (including docstrings and better organization)
