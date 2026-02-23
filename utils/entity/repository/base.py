"""Abstract base class for entity repository."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class EntityRepository(ABC):
    """Abstract base class for entity storage.

    This class defines the interface for storing and retrieving
    satellite templates, targets, and ground stations.
    """

    # Satellite template methods
    @abstractmethod
    def get_satellite_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get satellite template by ID."""
        pass

    @abstractmethod
    def list_satellite_templates(self) -> List[Dict[str, Any]]:
        """List all satellite templates."""
        pass

    @abstractmethod
    def save_satellite_template(self, template: Dict[str, Any]) -> None:
        """Save satellite template."""
        pass

    @abstractmethod
    def delete_satellite_template(self, template_id: str) -> bool:
        """Delete satellite template. Returns True if deleted."""
        pass

    # Target methods
    @abstractmethod
    def get_target(self, target_id: str, target_type: str = "point") -> Optional[Dict[str, Any]]:
        """Get target by ID."""
        pass

    @abstractmethod
    def list_targets(self, target_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all targets."""
        pass

    @abstractmethod
    def save_target(self, target: Dict[str, Any], target_type: str = "point") -> None:
        """Save target."""
        pass

    @abstractmethod
    def delete_target(self, target_id: str, target_type: str = "point") -> bool:
        """Delete target. Returns True if deleted."""
        pass

    @abstractmethod
    def query_targets_by_region(self, min_lon: float, max_lon: float,
                                 min_lat: float, max_lat: float) -> List[Dict[str, Any]]:
        """Query targets within geographic region."""
        pass

    # Ground station methods
    @abstractmethod
    def get_ground_station(self, gs_id: str) -> Optional[Dict[str, Any]]:
        """Get ground station by ID."""
        pass

    @abstractmethod
    def list_ground_stations(self) -> List[Dict[str, Any]]:
        """List all ground stations."""
        pass

    @abstractmethod
    def save_ground_station(self, gs: Dict[str, Any]) -> None:
        """Save ground station."""
        pass

    @abstractmethod
    def delete_ground_station(self, gs_id: str) -> bool:
        """Delete ground station. Returns True if deleted."""
        pass
