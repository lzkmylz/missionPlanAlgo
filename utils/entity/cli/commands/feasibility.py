"""Feasibility check command for satellite scenarios.

This module provides functionality to check if a satellite scenario is feasible
by analyzing visibility windows, observation opportunities, and resource constraints.
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..utils import safe_load_json


class FeasibilityChecker:
    """
    Feasibility checker for satellite observation scenarios.

    Analyzes a scenario to determine if it is feasible by checking:
    1. Visibility windows between satellites and targets
    2. Observation opportunities vs requirements
    3. Resource constraints (power, storage, attitude)
    """

    def __init__(self, scenario: Dict[str, Any]):
        """
        Initialize the feasibility checker.

        Args:
            scenario: The scenario dictionary containing satellites, targets, etc.
        """
        self.scenario = scenario
        self.console = Console() if RICH_AVAILABLE else None

    def check_visibility_windows(self) -> Dict[str, Any]:
        """
        Check if visibility windows exist between satellites and targets.

        Returns:
            Dictionary containing visibility analysis results
        """
        satellites = self.scenario.get("satellites", []) or []
        targets = self.scenario.get("targets", []) or []

        result = {
            "has_windows": False,
            "total_windows": 0,
            "satellites_with_windows": [],
            "targets_with_windows": [],
            "windows_per_satellite": {},
            "windows_per_target": {},
            "total_satellites": len(satellites),
            "total_targets": len(targets)
        }

        if not satellites or not targets:
            return result

        # Parse time range
        start_time = self._parse_datetime(self.scenario.get("start_time"))
        end_time = self._parse_datetime(self.scenario.get("end_time"))

        if not start_time or not end_time:
            result["error"] = "Invalid datetime format in scenario"
            return result

        # Compute visibility windows for each satellite-target pair
        all_windows = []
        for sat in satellites:
            sat_id = sat.get("id", "unknown")
            for target in targets:
                target_id = target.get("id", "unknown")
                windows = self._compute_visibility(sat, target, start_time, end_time)
                all_windows.extend(windows)

                if windows:
                    if sat_id not in result["windows_per_satellite"]:
                        result["windows_per_satellite"][sat_id] = 0
                    if target_id not in result["windows_per_target"]:
                        result["windows_per_target"][target_id] = 0

                    result["windows_per_satellite"][sat_id] += len(windows)
                    result["windows_per_target"][target_id] += len(windows)

        result["total_windows"] = len(all_windows)
        result["has_windows"] = len(all_windows) > 0
        result["satellites_with_windows"] = list(result["windows_per_satellite"].keys())
        result["targets_with_windows"] = list(result["windows_per_target"].keys())

        return result

    def _compute_visibility(
        self,
        satellite: Dict[str, Any],
        target: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ) -> List[Any]:
        """
        Compute visibility windows for a satellite-target pair.

        This is a simplified implementation that estimates visibility based on
        orbital parameters and target position.

        Args:
            satellite: Satellite configuration
            target: Target configuration
            start_time: Start of analysis period
            end_time: End of analysis period

        Returns:
            List of visibility window objects
        """
        windows = []

        # Get satellite orbit info
        orbit = satellite.get("orbit", {})
        altitude = orbit.get("altitude", 500000)  # meters

        # Get target position
        position = target.get("position", {})
        target_lat = position.get("latitude", 0)
        target_lon = position.get("longitude", 0)

        # Simple visibility estimation based on orbital period
        # A satellite at 500km altitude has roughly 90-minute orbit
        orbit_period_minutes = 90 + (altitude - 500000) / 10000

        # Estimate number of passes per day based on inclination
        inclination = orbit.get("inclination", 97.5)
        if inclination > 90:  # Sun-synchronous / retrograde
            passes_per_day = 14
        else:
            passes_per_day = int(24 * 60 / orbit_period_minutes)

        # Generate synthetic windows for demonstration
        # In a real implementation, this would use proper orbital mechanics
        duration_days = (end_time - start_time).total_seconds() / 86400
        num_passes = int(passes_per_day * duration_days)

        for i in range(num_passes):
            # Stagger passes
            pass_start = start_time + timedelta(hours=i * 24 / passes_per_day)
            pass_duration = timedelta(minutes=10)  # Typical pass duration

            window = MockVisibilityWindow(
                satellite_id=satellite.get("id", "unknown"),
                target_id=target.get("id", "unknown"),
                start_time=pass_start,
                end_time=pass_start + pass_duration,
                max_elevation=45.0  # Average elevation
            )
            windows.append(window)

        return windows

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string to datetime object."""
        if not dt_str:
            return None
        try:
            # Try ISO format
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def check_observation_opportunities(self) -> Dict[str, Any]:
        """
        Check if there are enough observation opportunities for all targets.

        Returns:
            Dictionary containing opportunity analysis results
        """
        targets = self.scenario.get("targets", []) or []

        result = {
            "total_required": 0,
            "total_available": 0,
            "can_meet_requirements": True,
            "targets_underprovisioned": [],
            "targets_overprovisioned": [],
            "opportunities_per_target": {}
        }

        # Get visibility windows
        visibility = self.check_visibility_windows()
        windows_per_target = visibility.get("windows_per_target", {})

        for target in targets:
            target_id = target.get("id", "unknown")
            required = target.get("required_observations", 0) or 0
            available = windows_per_target.get(target_id, 0)

            result["total_required"] += required
            result["total_available"] += available
            result["opportunities_per_target"][target_id] = {
                "required": required,
                "available": available,
                "satisfied": available >= required
            }

            if available < required:
                result["targets_underprovisioned"].append(target_id)
                result["can_meet_requirements"] = False
            elif available > required * 2:  # More than 2x needed
                result["targets_overprovisioned"].append(target_id)

        return result

    def check_resource_constraints(self) -> Dict[str, Any]:
        """
        Check if resource constraints are satisfied.

        Returns:
            Dictionary containing resource constraint analysis
        """
        satellites = self.scenario.get("satellites", []) or []
        targets = self.scenario.get("targets", []) or []

        result = {
            "all_satisfied": True,
            "violations": [],
            "warnings": [],
            "power": {},
            "storage": {},
            "attitude": {}
        }

        # Minimum resource thresholds
        MIN_POWER_CAPACITY = 500
        MIN_STORAGE_CAPACITY = 100
        MIN_OFF_NADIR = 10.0

        for sat in satellites:
            sat_id = sat.get("id", "unknown")
            capabilities = sat.get("capabilities", {}) or {}

            # Power constraints
            power_capacity = capabilities.get("power_capacity", 0)
            result["power"][sat_id] = {
                "capacity": power_capacity,
                "sufficient": power_capacity >= MIN_POWER_CAPACITY
            }
            if power_capacity < MIN_POWER_CAPACITY:
                result["violations"].append({
                    "type": "power",
                    "satellite": sat_id,
                    "message": f"Power capacity {power_capacity} below minimum {MIN_POWER_CAPACITY}"
                })
                result["all_satisfied"] = False

            # Storage constraints
            storage_capacity = capabilities.get("storage_capacity", 0)
            result["storage"][sat_id] = {
                "capacity": storage_capacity,
                "sufficient": storage_capacity >= MIN_STORAGE_CAPACITY
            }
            if storage_capacity < MIN_STORAGE_CAPACITY:
                result["violations"].append({
                    "type": "storage",
                    "satellite": sat_id,
                    "message": f"Storage capacity {storage_capacity} below minimum {MIN_STORAGE_CAPACITY}"
                })
                result["all_satisfied"] = False

            # Attitude constraints
            max_off_nadir = capabilities.get("max_off_nadir", 0)
            result["attitude"][sat_id] = {
                "max_off_nadir": max_off_nadir,
                "sufficient": max_off_nadir >= MIN_OFF_NADIR
            }
            if max_off_nadir < MIN_OFF_NADIR:
                result["violations"].append({
                    "type": "attitude",
                    "satellite": sat_id,
                    "message": f"Max off-nadir {max_off_nadir} below minimum {MIN_OFF_NADIR}"
                })
                result["all_satisfied"] = False

        # Check if targets can be observed given satellite capabilities
        for target in targets:
            resolution_required = target.get("resolution_required", 0)
            if resolution_required:
                # Higher resolution requires larger off-nadir angle
                # This is a simplified check
                required_off_nadir = min(45.0, resolution_required)

                can_observe = False
                for sat in satellites:
                    capabilities = sat.get("capabilities", {}) or {}
                    max_off_nadir = capabilities.get("max_off_nadir", 0)
                    if max_off_nadir >= required_off_nadir:
                        can_observe = True
                        break

                if not can_observe:
                    target_id = target.get("id", "unknown")
                    result["violations"].append({
                        "type": "resolution",
                        "target": target_id,
                        "message": f"No satellite can achieve required resolution {resolution_required}"
                    })
                    result["all_satisfied"] = False

        return result

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive feasibility report.

        Returns:
            Dictionary containing the full feasibility report
        """
        visibility = self.check_visibility_windows()
        opportunities = self.check_observation_opportunities()
        resources = self.check_resource_constraints()

        # Determine overall feasibility
        is_feasible = (
            visibility.get("has_windows", False) and
            opportunities.get("can_meet_requirements", False) and
            resources.get("all_satisfied", False)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            visibility, opportunities, resources
        )

        report = {
            "scenario_name": self.scenario.get("name", "Unnamed Scenario"),
            "is_feasible": is_feasible,
            "timestamp": datetime.now().isoformat(),
            "visibility": visibility,
            "opportunities": opportunities,
            "resources": resources,
            "recommendations": recommendations
        }

        return report

    def _generate_recommendations(
        self,
        visibility: Dict[str, Any],
        opportunities: Dict[str, Any],
        resources: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        if not visibility.get("has_windows"):
            recommendations.append(
                "No visibility windows found. Consider adding more satellites or adjusting target positions."
            )

        underprovisioned = opportunities.get("targets_underprovisioned", [])
        if underprovisioned:
            recommendations.append(
                f"Targets with insufficient opportunities: {', '.join(underprovisioned)}. "
                "Consider increasing observation windows or reducing requirements."
            )

        violations = resources.get("violations", [])
        for violation in violations:
            vtype = violation.get("type")
            message = violation.get("message", "")

            if vtype == "power":
                recommendations.append(
                    f"Power constraint violation: {message}. Consider satellites with larger power capacity."
                )
            elif vtype == "storage":
                recommendations.append(
                    f"Storage constraint violation: {message}. Consider satellites with larger storage capacity."
                )
            elif vtype == "attitude":
                recommendations.append(
                    f"Attitude constraint violation: {message}. Consider satellites with better maneuverability."
                )
            elif vtype == "resolution":
                recommendations.append(
                    f"Resolution constraint: {message}. Consider satellites with better imaging capabilities."
                )

        if not recommendations:
            if visibility.get("has_windows"):
                recommendations.append(
                    "Scenario appears feasible. All constraints can be satisfied with available resources."
                )

        return recommendations


class MockVisibilityWindow:
    """Mock visibility window for demonstration purposes."""

    def __init__(
        self,
        satellite_id: str,
        target_id: str,
        start_time: datetime,
        end_time: datetime,
        max_elevation: float = 45.0
    ):
        self.satellite_id = satellite_id
        self.target_id = target_id
        self.start_time = start_time
        self.end_time = end_time
        self.max_elevation = max_elevation
        self.quality_score = min(1.0, max_elevation / 90.0)

    def duration(self) -> float:
        """Return window duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()


@click.command("check-feasibility")
@click.argument("scenario_path")
@click.option("--output", "-o", type=str, help="Output file path for JSON report")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed report")
def check_feasibility(scenario_path: str, output: Optional[str], verbose: bool):
    """
    Check feasibility of a satellite scenario.

    Analyzes the scenario to determine if it is feasible by checking:
    - Visibility windows between satellites and targets
    - Observation opportunities vs requirements
    - Resource constraints (power, storage, attitude)

    Example:
        python -m utils.entity check-feasibility scenarios/my_scenario.json
    """
    # Load scenario
    scenario = safe_load_json(scenario_path)

    # Run feasibility check
    checker = FeasibilityChecker(scenario)
    report = checker.generate_report()

    # Output results
    if output:
        # Save to file
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        click.echo(f"Feasibility report saved to: {output}")

    # Print report
    if RICH_AVAILABLE and not verbose:
        _print_rich_report(report)
    else:
        _print_simple_report(report, verbose)

    # Exit with appropriate code
    if not report["is_feasible"]:
        raise click.ClickException("Scenario is not feasible")


def _print_rich_report(report: Dict[str, Any]):
    """Print a rich formatted feasibility report."""
    console = Console()

    is_feasible = report.get("is_feasible", False)
    status = "[green]✅ FEASIBLE[/green]" if is_feasible else "[red]✗ NOT FEASIBLE[/red]"

    panel = Panel(
        f"Scenario: {report.get('scenario_name', 'Unnamed')}\n"
        f"Status: {status}\n"
        f"Timestamp: {report.get('timestamp', 'N/A')}",
        title="Feasibility Report"
    )
    console.print(panel)

    # Visibility summary
    visibility = report.get("visibility", {})
    vis_table = Table(title="Visibility Analysis")
    vis_table.add_column("Metric", style="cyan")
    vis_table.add_column("Value", justify="right")

    vis_table.add_row("Total Windows", str(visibility.get("total_windows", 0)))
    vis_table.add_row("Satellites with Windows", str(len(visibility.get("satellites_with_windows", []))))
    vis_table.add_row("Targets with Windows", str(len(visibility.get("targets_with_windows", []))))

    console.print(vis_table)

    # Opportunities summary
    opportunities = report.get("opportunities", {})
    opp_table = Table(title="Observation Opportunities")
    opp_table.add_column("Metric", style="cyan")
    opp_table.add_column("Value", justify="right")

    opp_table.add_row("Total Required", str(opportunities.get("total_required", 0)))
    opp_table.add_row("Total Available", str(opportunities.get("total_available", 0)))
    opp_table.add_row(
        "Can Meet Requirements",
        "✅ Yes" if opportunities.get("can_meet_requirements") else "✗ No"
    )

    console.print(opp_table)

    # Resources summary
    resources = report.get("resources", {})
    res_table = Table(title="Resource Constraints")
    res_table.add_column("Constraint", style="cyan")
    res_table.add_column("Status", justify="center")

    if resources.get("all_satisfied"):
        res_table.add_row("All Constraints", "[green]✅ Satisfied[/green]")
    else:
        res_table.add_row("All Constraints", "[red]✗ Violations Found[/red]")
        for violation in resources.get("violations", []):
            res_table.add_row(
                violation.get("type", "unknown").capitalize(),
                f"[red]✗ {violation.get('message', '')}[/red]"
            )

    console.print(res_table)

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        console.print("\n[yellow]Recommendations:[/yellow]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")


def _print_simple_report(report: Dict[str, Any], verbose: bool):
    """Print a simple text feasibility report."""
    is_feasible = report.get("is_feasible", False)
    status = "FEASIBLE" if is_feasible else "NOT FEASIBLE"

    click.echo(f"\nFeasibility Report: {report.get('scenario_name', 'Unnamed')}")
    click.echo(f"Status: {status}")
    click.echo(f"Timestamp: {report.get('timestamp', 'N/A')}")

    # Visibility
    visibility = report.get("visibility", {})
    click.echo(f"\nVisibility:")
    click.echo(f"  Total Windows: {visibility.get('total_windows', 0)}")
    click.echo(f"  Satellites with Windows: {len(visibility.get('satellites_with_windows', []))}")
    click.echo(f"  Targets with Windows: {len(visibility.get('targets_with_windows', []))}")

    if verbose:
        click.echo(f"  Windows per Satellite: {visibility.get('windows_per_satellite', {})}")
        click.echo(f"  Windows per Target: {visibility.get('windows_per_target', {})}")

    # Opportunities
    opportunities = report.get("opportunities", {})
    click.echo(f"\nObservation Opportunities:")
    click.echo(f"  Total Required: {opportunities.get('total_required', 0)}")
    click.echo(f"  Total Available: {opportunities.get('total_available', 0)}")
    click.echo(f"  Can Meet Requirements: {opportunities.get('can_meet_requirements', False)}")

    if verbose:
        underprovisioned = opportunities.get("targets_underprovisioned", [])
        if underprovisioned:
            click.echo(f"  Underprovisioned Targets: {', '.join(underprovisioned)}")

    # Resources
    resources = report.get("resources", {})
    click.echo(f"\nResource Constraints:")
    click.echo(f"  All Satisfied: {resources.get('all_satisfied', False)}")

    if verbose:
        violations = resources.get("violations", [])
        if violations:
            click.echo("  Violations:")
            for v in violations:
                click.echo(f"    - {v.get('type', 'unknown')}: {v.get('message', '')}")

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        click.echo(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            click.echo(f"  {i}. {rec}")
