"""Scenario diff commands.

Commands for comparing two scenarios.
"""
from typing import Dict

import click

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..utils import safe_load_json, _get_dict_diff


console = Console() if RICH_AVAILABLE else None


def _print_diff_rich(base, variant, base_path: str, variant_path: str):
    """Print rich diff output."""
    console.print(Panel(f"场景差异比对\n基准: {base_path}\n变体: {variant_path}"))

    # Compare satellites
    base_sats = {s["id"]: s for s in base.get("satellites", [])}
    var_sats = {s["id"]: s for s in variant.get("satellites", [])}

    all_ids = set(base_sats.keys()) | set(var_sats.keys())

    for sat_id in sorted(all_ids):
        if sat_id in base_sats and sat_id in var_sats:
            # Check for differences
            diff_fields = _get_dict_diff(base_sats[sat_id], var_sats[sat_id])
            if diff_fields:
                console.print(f"\n卫星 [{sat_id}]:")
                for field, (old_val, new_val) in diff_fields.items():
                    console.print(f"  {field}: {old_val} -> {new_val}")
        elif sat_id in var_sats:
            console.print(f"\n卫星 [{sat_id}] (新增):")
            console.print(f"  + {var_sats[sat_id]}")
        else:
            console.print(f"\n卫星 [{sat_id}] (删除)")


def _print_diff_simple(base, variant):
    """Print simple diff output."""
    base_sats = {s["id"]: s for s in base.get("satellites", [])}
    var_sats = {s["id"]: s for s in variant.get("satellites", [])}

    click.echo("卫星差异:")
    for sat_id in sorted(set(base_sats.keys()) | set(var_sats.keys())):
        if sat_id in base_sats and sat_id in var_sats:
            if base_sats[sat_id] != var_sats[sat_id]:
                click.echo(f"  {sat_id}: modified")
        elif sat_id in var_sats:
            click.echo(f"  {sat_id}: added")
        else:
            click.echo(f"  {sat_id}: removed")


@click.command()
@click.argument("base_path")
@click.argument("variant_path")
def diff(base_path: str, variant_path: str):
    """Compare two scenarios."""
    base = safe_load_json(base_path)
    variant = safe_load_json(variant_path)

    if RICH_AVAILABLE:
        _print_diff_rich(base, variant, base_path, variant_path)
    else:
        _print_diff_simple(base, variant)
