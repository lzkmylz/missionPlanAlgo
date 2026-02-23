"""Scenario validation commands.

Commands for validating scenario files.
"""
import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ...validator import ScenarioValidator
from ..utils import safe_load_json


console = Console() if RICH_AVAILABLE else None


def _print_validation_report(scenario, is_valid: bool, errors, warnings):
    """Print rich validation report."""
    status = "✅ 通过" if is_valid else "✗ 失败"
    panel = Panel(f"场景校验报告: {scenario.get('name', 'unnamed')}\n状态: {status}")
    console.print(panel)

    # Entity counts
    table = Table(title="实体统计")
    table.add_column("实体类型", style="cyan")
    table.add_column("数量", justify="right")
    table.add_column("状态", style="green")

    sat_count = len(scenario.get("satellites", []))
    target_count = len(scenario.get("targets", []))
    gs_count = len(scenario.get("ground_stations", []))

    table.add_row("卫星", str(sat_count), "✅ 正常" if sat_count > 0 else "⚠️ 空")
    table.add_row("目标", str(target_count), "✅ 正常" if target_count > 0 else "⚠️ 空")
    table.add_row("地面站", str(gs_count), "✅ 正常" if gs_count > 0 else "⚠️ 空")

    console.print(table)

    if errors:
        console.print("\n[red]错误:[/red]")
        for error in errors:
            console.print(f"  - {error}")

    if warnings:
        console.print("\n[yellow]警告:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")


@click.command()
@click.argument("scenario_path")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed report")
def validate(scenario_path: str, verbose: bool):
    """Validate a scenario file."""
    # Load scenario
    scenario = safe_load_json(scenario_path)

    validator = ScenarioValidator()
    is_valid, errors, warnings = validator.validate(scenario, verbose=verbose)

    if RICH_AVAILABLE and not verbose:
        _print_validation_report(scenario, is_valid, errors, warnings)
    else:
        if is_valid:
            click.echo("Validation passed!")
        else:
            click.echo("Validation failed:")
            for error in errors:
                click.echo(f"  - {error}")

        if warnings:
            click.echo("Warnings:")
            for warning in warnings:
                click.echo(f"  - {warning}")

    if not is_valid:
        raise click.ClickException("Validation failed")
