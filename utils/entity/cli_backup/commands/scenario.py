"""Scenario management commands.

Commands for creating, cloning, and modifying scenarios.
"""
from copy import deepcopy
from typing import Optional

import click

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ...builder import ScenarioBuilder
from ...walker import WalkerConfig, WalkerGenerator
from ...repository.json_repository import JSONEntityRepository
from ..utils import safe_load_json


console = Console() if RICH_AVAILABLE else None


def _print_dry_run_changes(original, modified, description: str):
    """Print dry-run changes."""
    if RICH_AVAILABLE:
        console.print(f"\n[blue]=== Dry Run: {description} ===[/blue]")
        console.print("Changes would be applied:")
        console.print(f"  Satellites: {len(original.get('satellites', []))} -> {len(modified.get('satellites', []))}")
        console.print("\n[yellow]Use without --dry-run to apply changes[/yellow]")
    else:
        click.echo(f"\n=== Dry Run: {description} ===")
        click.echo("Changes would be applied:")
        click.echo(f"  Satellites: {len(original.get('satellites', []))} -> {len(modified.get('satellites', []))}")
        click.echo("\nUse without --dry-run to apply changes")


def _print_dry_run_satellites(satellites):
    """Print dry-run satellite additions."""
    if RICH_AVAILABLE:
        console.print("\n[blue]=== 即将执行的变更 ===[/blue]")
        for sat in satellites:
            console.print(f"[+] 添加卫星: {sat.get('id')}")
            console.print(f"    轨道高度: {sat.get('orbit', {}).get('altitude', 'N/A')} m")
            console.print(f"    倾角: {sat.get('orbit', {}).get('inclination', 'N/A')}°")
            console.print(f"    RAAN: {sat.get('raan', 'N/A')}°")
            if sat.get('mean_anomaly'):
                console.print(f"    Mean Anomaly: {sat['mean_anomaly']}°")
            console.print()
        console.print(f"...共 {len(satellites)} 颗卫星\n")
        console.print("[yellow]使用 --apply 确认执行，或移除 --dry-run 直接应用[/yellow]")
    else:
        click.echo(f"\n=== 即将执行的变更 ===")
        for sat in satellites:
            click.echo(f"[+] 添加卫星: {sat.get('id')}")
        click.echo(f"\n...共 {len(satellites)} 颗卫星")
        click.echo("使用 --apply 确认执行，或移除 --dry-run 直接应用")


@click.command()
@click.argument("output_path")
@click.option("--with-metadata", is_flag=True, help="Include metadata section")
def init(output_path: str, with_metadata: bool):
    """Initialize a new empty scenario."""
    builder = ScenarioBuilder()
    scenario = builder.init_scenario(output_path, with_metadata=with_metadata)
    builder.save_scenario(scenario, output_path)

    click.echo(f"Scenario initialized: {output_path}")


@click.command()
@click.argument("source_path")
@click.option("--output", "-o", required=True, help="Output path for cloned scenario")
def clone(source_path: str, output: str):
    """Clone an existing scenario."""
    builder = ScenarioBuilder()
    scenario = builder.clone_scenario(source_path, output)
    builder.save_scenario(scenario, output)

    click.echo(f"Scenario cloned: {source_path} -> {output}")


@click.command(name="add-to-scenario")
@click.argument("scenario_path")
@click.option("--entity-type", required=True,
              type=click.Choice(["satellite", "target", "ground_station"]))
@click.option("--template", help="Template ID for satellite")
@click.option("--id", "entity_id", required=True, help="Entity ID")
@click.option("--raan", type=float, help="RAAN for satellite")
@click.option("--mean-anomaly", type=float, help="Mean anomaly for satellite")
@click.option("--from-file", help="Batch add from JSON file")
@click.option("--dry-run", is_flag=True, help="Preview changes without applying")
def add_to_scenario(scenario_path: str, entity_type: str, template: Optional[str],
                    entity_id: str, raan: Optional[float],
                    mean_anomaly: Optional[float], from_file: Optional[str],
                    dry_run: bool):
    """Add an entity to a scenario."""
    # Load scenario
    scenario = safe_load_json(scenario_path)

    builder = ScenarioBuilder()
    original_scenario = deepcopy(scenario)

    if from_file:
        # Batch add from file
        batch_data = safe_load_json(from_file)
        click.echo(f"Batch adding from {from_file}")
    else:
        # Single entity add
        if entity_type == "satellite":
            if not template:
                raise click.BadParameter("--template is required for satellites")

            overrides = {}
            if raan is not None:
                overrides["raan"] = raan
            if mean_anomaly is not None:
                overrides["mean_anomaly"] = mean_anomaly

            scenario = builder.add_satellite_to_scenario(
                scenario, template, entity_id, overrides=overrides if overrides else None
            )

            if dry_run:
                _print_dry_run_changes(original_scenario, scenario, f"satellite {entity_id}")
            else:
                builder.save_scenario(scenario, scenario_path)
                click.echo(f"Added satellite '{entity_id}' to scenario.")


@click.command()
@click.argument("scenario_path")
@click.option("--template", required=True, help="Satellite template to use")
@click.option("--planes", "-p", type=int, help="Number of orbital planes")
@click.option("--sats-per-plane", "-s", type=int, help="Satellites per plane")
@click.option("--inclination", "-i", type=float, help="Orbit inclination (degrees)")
@click.option("--f-factor", "-f", type=int, default=1, help="Phasing factor")
@click.option("--raan-start", type=float, default=0.0, help="Starting RAAN")
@click.option("--raan-spread", type=float, default=360.0, help="RAAN spread")
@click.option("--prefix", default="WALKER", help="Satellite ID prefix")
@click.option("--preset", help="Use Walker preset configuration")
@click.option("--dry-run", is_flag=True, help="Preview changes without applying")
def walker(scenario_path: str, template: str, planes: Optional[int],
           sats_per_plane: Optional[int], inclination: Optional[float],
           f_factor: int, raan_start: float, raan_spread: float,
           prefix: str, preset: Optional[str], dry_run: bool):
    """Generate Walker constellation and add to scenario."""
    # Load scenario
    scenario = safe_load_json(scenario_path)

    # Get template
    repo = JSONEntityRepository()
    sat_template = repo.get_satellite_template(template)
    if not sat_template:
        raise click.ClickException(f"Template '{template}' not found")

    # Determine configuration
    if preset:
        config = WalkerGenerator.get_preset(preset)
        if not config:
            raise click.BadParameter(f"Unknown preset: {preset}")
        if inclination is not None:
            config.inclination = inclination
    else:
        if planes is None or sats_per_plane is None:
            raise click.BadParameter("--planes and --sats-per-plane required without --preset")
        config = WalkerConfig(
            inclination=inclination or 97.9,
            total_sats=planes * sats_per_plane,
            n_planes=planes,
            f_factor=f_factor,
            raan_start=raan_start,
            raan_spread=raan_spread
        )

    # Generate Walker constellation
    generator = WalkerGenerator()
    satellites = generator.generate(config, sat_template, prefix=prefix)

    if dry_run:
        _print_dry_run_satellites(satellites)
    else:
        # Add satellites to scenario
        scenario["satellites"].extend(satellites)

        # Save scenario
        builder = ScenarioBuilder()
        builder.save_scenario(scenario, scenario_path)

        click.echo(f"Added {len(satellites)} Walker satellites to scenario.")
