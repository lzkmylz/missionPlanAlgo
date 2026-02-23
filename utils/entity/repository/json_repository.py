"""JSON file-based entity repository implementation."""
from pathlib import Path
from typing import List, Optional, Dict, Any

from ...json_utils import load_json, save_json
from .base import EntityRepository


class JSONEntityRepository(EntityRepository):
    """JSON file-based entity repository.

    Stores entities as JSON files in a directory structure:
    - satellites/{template_id}.json
    - targets/point/{target_id}.json
    - targets/area/{target_id}.json
    - ground_stations/{gs_id}.json
    """

    def __init__(self, base_path: str = "data/entity_lib"):
        """Initialize repository with base path.

        Creates directory structure if it doesn't exist.
        """
        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create directory structure if not exists."""
        (self.base_path / "satellites").mkdir(parents=True, exist_ok=True)
        (self.base_path / "targets" / "point").mkdir(parents=True, exist_ok=True)
        (self.base_path / "targets" / "area").mkdir(parents=True, exist_ok=True)
        (self.base_path / "ground_stations").mkdir(parents=True, exist_ok=True)

    def _get_satellite_path(self, template_id: str) -> Path:
        """Get path to satellite template file."""
        return self.base_path / "satellites" / f"{template_id}.json"

    def _get_target_path(self, target_id: str, target_type: str) -> Path:
        """Get path to target file."""
        return self.base_path / "targets" / target_type / f"{target_id}.json"

    def _get_ground_station_path(self, gs_id: str) -> Path:
        """Get path to ground station file."""
        return self.base_path / "ground_stations" / f"{gs_id}.json"

    def _load_json(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file, return None if not exists or invalid."""
        if not filepath.exists():
            return None
        try:
            return load_json(filepath)
        except (Exception):
            return None

    def _save_json(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Save data to JSON file."""
        save_json(data, filepath)

    # Satellite template methods
    def get_satellite_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get satellite template by ID."""
        filepath = self._get_satellite_path(template_id)
        return self._load_json(filepath)

    def list_satellite_templates(self) -> List[Dict[str, Any]]:
        """List all satellite templates."""
        satellites_dir = self.base_path / "satellites"
        if not satellites_dir.exists():
            return []

        templates = []
        for filepath in satellites_dir.glob("*.json"):
            data = self._load_json(filepath)
            if data is not None:
                templates.append(data)
        return templates

    def save_satellite_template(self, template: Dict[str, Any]) -> None:
        """Save satellite template."""
        template_id = template.get("template_id")
        if not template_id:
            raise ValueError("Template must have 'template_id' field")
        filepath = self._get_satellite_path(template_id)
        self._save_json(filepath, template)

    def delete_satellite_template(self, template_id: str) -> bool:
        """Delete satellite template. Returns True if deleted."""
        filepath = self._get_satellite_path(template_id)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    # Target methods
    def get_target(self, target_id: str, target_type: str = "point") -> Optional[Dict[str, Any]]:
        """Get target by ID."""
        filepath = self._get_target_path(target_id, target_type)
        return self._load_json(filepath)

    def list_targets(self, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all targets."""
        targets = []

        if target_type is None or target_type == "point":
            point_dir = self.base_path / "targets" / "point"
            if point_dir.exists():
                for filepath in point_dir.glob("*.json"):
                    data = self._load_json(filepath)
                    if data is not None:
                        targets.append(data)

        if target_type is None or target_type == "area":
            area_dir = self.base_path / "targets" / "area"
            if area_dir.exists():
                for filepath in area_dir.glob("*.json"):
                    data = self._load_json(filepath)
                    if data is not None:
                        targets.append(data)

        return targets

    def save_target(self, target: Dict[str, Any], target_type: str = "point") -> None:
        """Save target."""
        target_id = target.get("id")
        if not target_id:
            raise ValueError("Target must have 'id' field")
        filepath = self._get_target_path(target_id, target_type)
        self._save_json(filepath, target)

    def delete_target(self, target_id: str, target_type: str = "point") -> bool:
        """Delete target. Returns True if deleted."""
        filepath = self._get_target_path(target_id, target_type)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def query_targets_by_region(self, min_lon: float, max_lon: float,
                                 min_lat: float, max_lat: float) -> List[Dict[str, Any]]:
        """Query targets within geographic region."""
        results = []

        for target in self.list_targets():
            # Get position - handle both formats
            if "position" in target:
                lon = target["position"].get("longitude", 0)
                lat = target["position"].get("latitude", 0)
            else:
                lon = target.get("longitude", 0)
                lat = target.get("latitude", 0)

            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                results.append(target)

        return results

    # Ground station methods
    def get_ground_station(self, gs_id: str) -> Optional[Dict[str, Any]]:
        """Get ground station by ID."""
        filepath = self._get_ground_station_path(gs_id)
        return self._load_json(filepath)

    def list_ground_stations(self) -> List[Dict[str, Any]]:
        """List all ground stations."""
        gs_dir = self.base_path / "ground_stations"
        if not gs_dir.exists():
            return []

        stations = []
        for filepath in gs_dir.glob("*.json"):
            data = self._load_json(filepath)
            if data is not None:
                stations.append(data)
        return stations

    def save_ground_station(self, gs: Dict[str, Any]) -> None:
        """Save ground station."""
        gs_id = gs.get("id")
        if not gs_id:
            raise ValueError("Ground station must have 'id' field")
        filepath = self._get_ground_station_path(gs_id)
        self._save_json(filepath, gs)

    def delete_ground_station(self, gs_id: str) -> bool:
        """Delete ground station. Returns True if deleted."""
        filepath = self._get_ground_station_path(gs_id)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def get_ground_station_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get ground station template by ID.

        Searches for template files by template_id field.
        """
        gs_dir = self.base_path / "ground_stations"
        if not gs_dir.exists():
            return None

        for filepath in gs_dir.glob("*.json"):
            data = self._load_json(filepath)
            if data and data.get("template_id") == template_id:
                return data
        return None

    def get_target_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get target template by ID.

        Searches for template files by template_id field in both point and area directories.
        """
        # Search in point targets
        point_dir = self.base_path / "targets" / "point"
        if point_dir.exists():
            for filepath in point_dir.glob("*.json"):
                data = self._load_json(filepath)
                if data and data.get("template_id") == template_id:
                    return data

        # Search in area targets
        area_dir = self.base_path / "targets" / "area"
        if area_dir.exists():
            for filepath in area_dir.glob("*.json"):
                data = self._load_json(filepath)
                if data and data.get("template_id") == template_id:
                    return data

        return None
