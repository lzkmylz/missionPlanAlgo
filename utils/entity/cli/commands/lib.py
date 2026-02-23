"""Entity library management commands.

Commands for managing the entity library (satellites, targets, ground stations).
"""
import json
from typing import Optional

import click

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ...library import EntityLibrary


console = Console() if RICH_AVAILABLE else None


@click.group()
def lib():
    """Entity library management commands."""
    pass


@lib.command()
@click.option("--type", "entity_type", required=True,
              type=click.Choice(["satellite", "target", "ground_station"]))
def list(entity_type: str):
    """List entities in the library."""
    library = EntityLibrary()

    if RICH_AVAILABLE:
        table = Table(title=f"{entity_type.capitalize()} Library")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")

        if entity_type == "satellite":
            items = library.list_satellites()
            table.add_column("Type", style="yellow")
            for item in items:
                table.add_row(
                    item.get("template_id", "N/A"),
                    item.get("name", "N/A"),
                    item.get("sat_type", "N/A")
                )
        elif entity_type == "target":
            items = library.list_targets()
            table.add_column("Priority", style="yellow")
            for item in items:
                table.add_row(
                    item.get("id", "N/A"),
                    item.get("name", "N/A"),
                    str(item.get("priority", "N/A"))
                )
        else:
            items = library.list_ground_stations()
            for item in items:
                table.add_row(
                    item.get("id", "N/A"),
                    item.get("name", "N/A")
                )

        console.print(table)
    else:
        click.echo(f"{entity_type.capitalize()} Library:")
        if entity_type == "satellite":
            for item in library.list_satellites():
                click.echo(f"  {item.get('template_id')}: {item.get('name')}")
        elif entity_type == "target":
            for item in library.list_targets():
                click.echo(f"  {item.get('id')}: {item.get('name')}")


@lib.command()
@click.argument("entity_type")
@click.argument("entity_id")
def show(entity_type: str, entity_id: str):
    """Show entity details."""
    library = EntityLibrary()

    if entity_type == "satellite":
        entity = library.get_satellite(entity_id)
    elif entity_type == "target":
        entity = library.get_target(entity_id)
    elif entity_type == "ground_station":
        entity = library.get_ground_station(entity_id)
    else:
        raise click.BadParameter(f"Unknown entity type: {entity_type}")

    if not entity:
        raise click.ClickException(f"{entity_type} '{entity_id}' not found")

    if RICH_AVAILABLE:
        console.print(json.dumps(entity, indent=2, ensure_ascii=False))
    else:
        click.echo(json.dumps(entity, indent=2, ensure_ascii=False))


@lib.command()
@click.option("--id", "satellite_id", required=True, help="Satellite template ID")
def add_satellite(satellite_id: str):
    """Add a new satellite template (interactive wizard)."""
    library = EntityLibrary()

    click.echo(f"Adding satellite template: {satellite_id}")

    # Interactive prompts
    name = click.prompt("Name", default=f"卫星{satellite_id}")
    sat_type = click.prompt("Satellite type", default="optical_1")
    altitude = click.prompt("Orbit altitude (m)", default=645000, type=float)
    inclination = click.prompt("Inclination (degrees)", default=97.9, type=float)
    storage = click.prompt("Storage capacity (GB)", default=500, type=int)

    template = {
        "template_id": satellite_id,
        "name": name,
        "sat_type": sat_type,
        "orbit": {
            "orbit_type": "SSO",
            "altitude": altitude,
            "inclination": inclination
        },
        "capabilities": {
            "storage_capacity": storage,
            "power_capacity": 2000,
            "data_rate": 300
        }
    }

    library.repo.save_satellite_template(template)
    click.echo(f"Satellite template '{satellite_id}' added successfully.")


@lib.command()
@click.option("--id", "target_id", required=True, help="Target ID")
@click.option("--name", default=None, help="Target name")
@click.option("--lon", "longitude", required=True, type=float, help="Longitude")
@click.option("--lat", "latitude", required=True, type=float, help="Latitude")
@click.option("--priority", default=5, type=int, help="Priority (1-10)")
def add_target(target_id: str, name: Optional[str], longitude: float,
               latitude: float, priority: int):
    """Add a point target to the library."""
    library = EntityLibrary()

    target_name = name or target_id

    library.add_target(
        target_id=target_id,
        name=target_name,
        longitude=longitude,
        latitude=latitude,
        priority=priority
    )

    click.echo(f"Target '{target_id}' added successfully.")


@lib.command()
@click.option("--type", "entity_type", required=True,
              type=click.Choice(["satellite", "target", "ground_station"]))
@click.option("--id", "entity_id", required=True, help="Entity ID to remove")
def remove(entity_type: str, entity_id: str):
    """Remove an entity from the library."""
    library = EntityLibrary()

    if entity_type == "satellite":
        library.repo.delete_satellite_template(entity_id)
    elif entity_type == "target":
        library.repo.delete_target(entity_id)
    elif entity_type == "ground_station":
        library.repo.delete_ground_station(entity_id)

    click.echo(f"{entity_type} '{entity_id}' removed successfully.")


@lib.command(name="init")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def lib_init(yes: bool):
    """Initialize entity library with default templates."""
    library = EntityLibrary()

    # Interactive confirmation unless --yes flag is provided
    if not yes:
        if not click.confirm("This will initialize the entity library with default templates. Continue?"):
            click.echo("Initialization cancelled.")
            return

    try:
        library.init_defaults()
        click.echo("Entity library initialized successfully with default templates.")
        click.echo("Added: 4 satellite templates (optical_1, optical_2, sar_1, sar_2)")
        click.echo("Added: 2 ground stations (gs_beijing, gs_kashi)")
    except Exception as e:
        raise click.ClickException(f"Failed to initialize entity library: {e}")
