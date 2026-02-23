"""Main CLI entry point.

This module defines the main CLI group and registers all commands.
"""
import click

from .commands.lib import lib
from .commands.scenario import init, clone, add_to_scenario, walker
from .commands.validate import validate
from .commands.diff import diff
from .commands.sweep import sweep
from .commands.feasibility import check_feasibility


@click.group()
@click.version_option(version="1.0.0", prog_name="sat-cli")
def main():
    """Satellite scenario configuration CLI tool."""
    pass


# Register command groups
main.add_command(lib)

# Register scenario management commands
main.add_command(init)
main.add_command(clone)
main.add_command(add_to_scenario)
main.add_command(walker)

# Register other commands
main.add_command(validate)
main.add_command(diff)
main.add_command(sweep)
main.add_command(check_feasibility)
