"""Scenario validation module."""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from jsonschema import validate, ValidationError

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .repository.json_repository import JSONEntityRepository


class ScenarioValidator:
    """Validate scenario files.

    Performs:
    - JSON Schema validation
    - Business rules validation
    - Template version checking
    """

    # Business rules
    RULES = {
        'satellite_altitude': {'min': 300000, 'max': 1000000},  # meters
        'satellite_inclination': {'min': 0, 'max': 180},  # degrees
        'satellite_eccentricity': {'min': 0, 'max': 0.1},
        'target_longitude': {'min': -180, 'max': 180},
        'target_latitude': {'min': -90, 'max': 90},
        'target_priority': {'min': 1, 'max': 10},
        'antenna_elevation': {'min': 0, 'max': 90},
    }

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize validator.

        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        self.schema = self._load_schema(schema_path)
        self.console = Console() if RICH_AVAILABLE else None

    def _load_schema(self, schema_path: Optional[str] = None) -> Dict[str, Any]:
        """Load JSON schema."""
        if schema_path and Path(schema_path).exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Return default schema
        return {
            "type": "object",
            "required": ["name", "start_time", "end_time"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "start_time": {"type": "string", "format": "date-time"},
                "end_time": {"type": "string", "format": "date-time"},
                "satellites": {"type": "array"},
                "targets": {"type": "array"},
                "ground_stations": {"type": "array"}
            }
        }

    def validate(self, scenario: Dict[str, Any], verbose: bool = False) -> Tuple[bool, List[str], List[str]]:
        """Validate scenario.

        Args:
            scenario: Scenario dictionary
            verbose: Print detailed report

        Returns:
            (is_valid, errors, warnings) tuple
        """
        errors = []
        warnings = []

        # JSON Schema validation
        try:
            validate(instance=scenario, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation: {e.message}")

        # Business rules validation
        errors.extend(self._validate_business_rules(scenario))

        # Template version warnings
        warnings.extend(self._check_template_versions(scenario))

        if verbose and self.console:
            self._print_report(scenario, errors, warnings)

        return len(errors) == 0, errors, warnings

    def _validate_business_rules(self, scenario: Dict[str, Any]) -> List[str]:
        """Validate business rules."""
        errors = []

        # Time validation
        start = scenario.get('start_time')
        end = scenario.get('end_time')
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                if end_dt <= start_dt:
                    errors.append("end_time must be after start_time")
            except ValueError:
                pass

        # Satellite validation
        for sat in scenario.get('satellites', []):
            errors.extend(self._validate_satellite(sat))

        # Target validation
        for target in scenario.get('targets', []):
            errors.extend(self._validate_target(target))

        # Ground station validation
        for gs in scenario.get('ground_stations', []):
            errors.extend(self._validate_ground_station(gs))

        return errors

    def _validate_satellite(self, sat: Dict[str, Any]) -> List[str]:
        """Validate satellite."""
        errors = []
        sat_id = sat.get('id', 'unknown')
        orbit = sat.get('orbit', {})

        # Altitude check
        alt = orbit.get('altitude', 0)
        if not (self.RULES['satellite_altitude']['min'] <= alt <= self.RULES['satellite_altitude']['max']):
            errors.append(f"Satellite {sat_id}: altitude {alt} out of range [300-1000] km")

        # Inclination check
        inc = orbit.get('inclination', 0)
        if not (self.RULES['satellite_inclination']['min'] <= inc <= self.RULES['satellite_inclination']['max']):
            errors.append(f"Satellite {sat_id}: inclination {inc} out of range [0-180]°")

        # Eccentricity check
        ecc = orbit.get('eccentricity', 0)
        if not (self.RULES['satellite_eccentricity']['min'] <= ecc <= self.RULES['satellite_eccentricity']['max']):
            errors.append(f"Satellite {sat_id}: eccentricity {ecc} out of range [0-0.1]")

        return errors

    def _validate_target(self, target: Dict[str, Any]) -> List[str]:
        """Validate target."""
        errors = []
        target_id = target.get('id', 'unknown')

        # Get position - handle both formats
        if 'position' in target:
            lon = target['position'].get('longitude', 0)
            lat = target['position'].get('latitude', 0)
        else:
            lon = target.get('longitude', 0)
            lat = target.get('latitude', 0)

        # Longitude check
        if not (self.RULES['target_longitude']['min'] <= lon <= self.RULES['target_longitude']['max']):
            errors.append(f"Target {target_id}: longitude {lon} out of range [-180, 180]")

        # Latitude check
        if not (self.RULES['target_latitude']['min'] <= lat <= self.RULES['target_latitude']['max']):
            errors.append(f"Target {target_id}: latitude {lat} out of range [-90, 90]")

        # Priority check
        priority = target.get('priority', 5)
        if not (self.RULES['target_priority']['min'] <= priority <= self.RULES['target_priority']['max']):
            errors.append(f"Target {target_id}: priority {priority} out of range [1-10]")

        return errors

    def _validate_ground_station(self, gs: Dict[str, Any]) -> List[str]:
        """Validate ground station."""
        errors = []
        gs_id = gs.get('id', 'unknown')

        for antenna in gs.get('antennas', []):
            elev_min = antenna.get('elevation_min', 0)
            elev_max = antenna.get('elevation_max', 90)

            if not (self.RULES['antenna_elevation']['min'] <= elev_min <= self.RULES['antenna_elevation']['max']):
                errors.append(f"Ground station {gs_id}: elevation_min {elev_min} out of range [0-90]")
            if not (self.RULES['antenna_elevation']['min'] <= elev_max <= self.RULES['antenna_elevation']['max']):
                errors.append(f"Ground station {gs_id}: elevation_max {elev_max} out of range [0-90]")

        return errors

    def _check_template_versions(self, scenario: Dict[str, Any]) -> List[str]:
        """Check for outdated template versions."""
        warnings = []
        repo = JSONEntityRepository()

        for sat in scenario.get('satellites', []):
            template_source = sat.get('_template_source')
            template_version = sat.get('_template_version', 'unknown')

            if template_source:
                current = repo.get_satellite_template(template_source)
                if current:
                    current_version = current.get('version', '1.0')
                    if template_version != current_version:
                        warnings.append(
                            f"Satellite {sat.get('id', 'unknown')} uses old template version: "
                            f"{template_source} v{template_version} (current: v{current_version})"
                        )

        return warnings

    def _print_report(self, scenario: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Print formatted validation report."""
        if not self.console:
            return

        status = "✅ 通过" if not errors else "✗ 失败"
        panel = Panel(f"场景校验报告: {scenario.get('name', 'unnamed')}\n状态: {status}")
        self.console.print(panel)

        if errors:
            self.console.print("\n[red]错误:[/red]")
            for error in errors:
                self.console.print(f"  - {error}")

        if warnings:
            self.console.print("\n[yellow]警告:[/yellow]")
            for warning in warnings:
                self.console.print(f"  - {warning}")
