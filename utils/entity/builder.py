"""Scenario builder module for creating and modifying scenario files."""
import json
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
from typing import Dict, Any, Optional

from .repository.base import EntityRepository
from .repository.json_repository import JSONEntityRepository


class ScenarioBuilder:
    """Build and modify scenario files.

    Provides methods for:
    - Initializing new scenarios
    - Adding entities (satellites, targets, ground stations)
    - Cloning existing scenarios
    - Saving scenarios to disk
    """

    def __init__(self, repository: Optional[EntityRepository] = None):
        """Initialize scenario builder.

        Args:
            repository: Entity repository for accessing templates.
                       If None, creates default JSON repository.
        """
        self.repo = repository or JSONEntityRepository()

    def init_scenario(self, output_path: str, with_metadata: bool = True) -> Dict[str, Any]:
        """Initialize empty scenario with template structure.

        Args:
            output_path: Path where scenario will be saved
            with_metadata: Whether to include metadata section

        Returns:
            Scenario dictionary
        """
        scenario = {
            "name": Path(output_path).stem,
            "description": "",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }

        if with_metadata:
            scenario["metadata"] = {
                "coordinate_system": "WGS84",
                "time_system": "UTC",
                "time_format": "ISO8601",
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "version": "1.0"
            }

        return scenario

    def add_satellite_to_scenario(
        self,
        scenario: Dict[str, Any],
        template_id: str,
        sat_id: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add satellite to scenario from template.

        Args:
            scenario: Scenario dictionary to modify
            template_id: Template ID to use
            sat_id: Unique satellite ID
            overrides: Optional field overrides

        Returns:
            Modified scenario dictionary

        Raises:
            ValueError: If template not found
        """
        # Deep copy template
        template = self.repo.get_satellite_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        satellite = deepcopy(template)
        satellite["id"] = sat_id
        satellite["name"] = f"卫星{sat_id}"
        satellite["_template_source"] = template_id
        satellite["_template_version"] = template.get("version", "1.0")

        if overrides:
            self._apply_overrides(satellite, overrides)

        scenario["satellites"].append(satellite)
        return scenario

    def add_ground_station_to_scenario(
        self,
        scenario: Dict[str, Any],
        template_id: str,
        gs_id: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add ground station to scenario from template.

        Args:
            scenario: Scenario dictionary to modify
            template_id: Template ID to use
            gs_id: Unique ground station ID
            overrides: Optional field overrides

        Returns:
            Modified scenario dictionary

        Raises:
            ValueError: If template not found
        """
        # Deep copy template
        template = self.repo.get_ground_station_template(template_id)
        if not template:
            raise ValueError(f"Ground station template not found: {template_id}")

        ground_station = deepcopy(template)
        ground_station["id"] = gs_id
        ground_station["name"] = template.get("name", f"地面站{gs_id}")
        ground_station["_template_source"] = template_id
        ground_station["_template_version"] = template.get("version", "1.0")

        if overrides:
            self._apply_overrides(ground_station, overrides)

        scenario["ground_stations"].append(ground_station)
        return scenario

    def add_target_to_scenario(
        self,
        scenario: Dict[str, Any],
        template_id: str,
        target_id: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add target to scenario from template.

        Args:
            scenario: Scenario dictionary to modify
            template_id: Template ID to use
            target_id: Unique target ID
            overrides: Optional field overrides

        Returns:
            Modified scenario dictionary

        Raises:
            ValueError: If template not found
        """
        # Deep copy template
        template = self.repo.get_target_template(template_id)
        if not template:
            raise ValueError(f"Target template not found: {template_id}")

        target = deepcopy(template)
        target["id"] = target_id
        target["name"] = template.get("name", f"目标{target_id}")
        target["_template_source"] = template_id
        target["_template_version"] = template.get("version", "1.0")

        if overrides:
            self._apply_overrides(target, overrides)

        scenario["targets"].append(target)
        return scenario

    def _apply_overrides(self, obj: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """Apply override values to object recursively."""
        for key, value in overrides.items():
            if key in obj and isinstance(obj[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                self._apply_overrides(obj[key], value)
            else:
                obj[key] = value

    def clone_scenario(self, source_path: str, output_path: str) -> Dict[str, Any]:
        """Clone existing scenario.

        Args:
            source_path: Path to source scenario
            output_path: Path for cloned scenario

        Returns:
            Cloned scenario dictionary
        """
        with open(source_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        scenario["name"] = Path(output_path).stem

        # Update metadata if present
        if "metadata" in scenario:
            scenario["metadata"]["created_at"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

        return scenario

    def save_scenario(self, scenario: Dict[str, Any], output_path: str) -> None:
        """Save scenario to file.

        Args:
            scenario: Scenario dictionary
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)
