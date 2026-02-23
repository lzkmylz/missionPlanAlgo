"""Parameter sweep commands.

Commands for generating scenario matrices for parameter sweeps.
"""
from pathlib import Path
from typing import List, Iterator, Tuple, Dict, Any
from copy import deepcopy
from itertools import product

import click

from ..utils import safe_load_json, safe_save_json, _set_nested_value


def _generate_range_values(start: float, end: float, step: float) -> List[float]:
    """Generate all values in a range with given step."""
    values = []
    value = start
    # Use a small epsilon to handle floating point comparison
    epsilon = step * 0.0001
    while value <= end + epsilon:
        values.append(round(value, 10))  # Round to avoid floating point errors
        value += step
    return values


def _generate_cartesian_product(ranges: List[Dict[str, float]]) -> Iterator[Tuple[float, ...]]:
    """Generate Cartesian product of multiple ranges."""
    # Generate values for each range
    all_values = []
    for r in ranges:
        values = _generate_range_values(r["start"], r["end"], r["step"])
        all_values.append(values)

    # Return Cartesian product
    return product(*all_values)


def _generate_scenario_name(base_name: str, params: List[str], values: Tuple[float, ...]) -> str:
    """Generate a unique scenario name based on parameter values."""
    name_parts = [base_name]
    for param, value in zip(params, values):
        # Extract last part of parameter path for brevity
        param_short = param.split(".")[-1].replace("[", "_").replace("]", "")
        # Format value - use int if whole number
        if value == int(value):
            value_str = str(int(value))
        else:
            value_str = str(value).replace(".", "p")
        name_parts.append(f"{param_short}_{value_str}")
    return "_".join(name_parts)


@click.command()
@click.option("--base-scenario", required=True, help="Base scenario file")
@click.option("--param", multiple=True, required=True, help="Parameter path (e.g., satellites[0].capabilities.storage)")
@click.option("--range", "range_spec", multiple=True, required=True, help="Range as start:end:step")
@click.option("--output-dir", required=True, help="Output directory for generated scenarios")
def sweep(base_scenario: str, param: List[str], range_spec: List[str], output_dir: str):
    """Generate scenario matrix for parameter sweep."""
    # Validate param and range counts match
    if len(param) != len(range_spec):
        raise click.BadParameter(
            f"Number of parameters ({len(param)}) must match number of ranges ({len(range_spec)}). "
            f"Got params: {list(param)}, ranges: {list(range_spec)}"
        )

    # Load base scenario
    base = safe_load_json(base_scenario)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse ranges with validation
    ranges = []
    for rspec in range_spec:
        parts = rspec.split(":")
        if len(parts) != 3:
            raise click.BadParameter(f"Invalid range format: {rspec}. Use start:end:step")

        # Parse numeric values with error handling
        try:
            start = float(parts[0])
            end = float(parts[1])
            step = float(parts[2])
        except ValueError as e:
            raise click.BadParameter(f"Invalid numeric value in range '{rspec}': {e}. All values must be valid numbers")

        # Validate step is positive
        if step <= 0:
            raise click.BadParameter(f"Invalid step value {step} in range '{rspec}'. Step must be a positive number")

        # Validate start <= end
        if start > end:
            raise click.BadParameter(f"Invalid range '{rspec}': start ({start}) must be less than or equal to end ({end})")

        ranges.append({
            "start": start,
            "end": end,
            "step": step
        })

    # Generate scenarios
    if len(param) == 1:
        # Single parameter sweep
        p = param[0]
        r = ranges[0]
        count = 0

        for value in _generate_range_values(r["start"], r["end"], r["step"]):
            scenario = deepcopy(base)
            _set_nested_value(scenario, p, value)

            # Generate filename based on value type
            if value == int(value):
                filename = f"{base['name']}_{int(value)}.json"
            else:
                filename = f"{base['name']}_{value}.json"
            output_file = output_path / filename
            safe_save_json(scenario, output_file)

            count += 1

        click.echo(f"Generated {count} scenarios in {output_dir}")
    else:
        # Multi-parameter sweep - Cartesian product
        count = 0
        for values in _generate_cartesian_product(ranges):
            scenario = deepcopy(base)

            # Set each parameter value
            for p, v in zip(param, values):
                _set_nested_value(scenario, p, v)

            # Generate unique filename
            filename = _generate_scenario_name(base["name"], list(param), values)
            output_file = output_path / f"{filename}.json"
            safe_save_json(scenario, output_file)

            count += 1

        click.echo(f"Generated {count} scenarios in {output_dir}")
