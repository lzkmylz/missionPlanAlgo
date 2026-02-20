"""
Satellite resource pool management.

Manages satellite allocation, health monitoring, and state tracking.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from core.models.satellite import Satellite, SatelliteType

logger = logging.getLogger(__name__)


class SatelliteHealth(Enum):
    """Satellite health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class SatelliteState:
    """Satellite runtime state"""
    satellite_id: str
    storage_used: float = 0.0  # GB
    power_level: float = 0.0  # Wh
    current_task: Optional[str] = None
    health: SatelliteHealth = SatelliteHealth.HEALTHY
    last_updated: datetime = field(default_factory=datetime.now)


class SatellitePool:
    """Satellite resource pool manager

    Manages allocation and monitoring of satellite resources.
    Tracks storage, power, and health status for each satellite.

    Example:
        pool = SatellitePool(satellites)

        # Allocate satellite for task
        sat = pool.allocate_satellite({
            'type': 'optical',
            'storage_requirement': 100
        })

        if sat:
            # Use satellite...
            pool.update_satellite_state(sat.id, {
                'storage_used': 150,
                'power_level': 1800
            })

            # Release when done
            pool.release_satellite(sat.id)
    """

    def __init__(self, satellites: List[Satellite]):
        """Initialize satellite pool

        Args:
            satellites: List of available satellites
        """
        self.satellites: Dict[str, Satellite] = {
            sat.id: sat for sat in satellites
        }
        self._allocated: Dict[str, bool] = {
            sat.id: False for sat in satellites
        }
        self._states: Dict[str, SatelliteState] = {
            sat.id: SatelliteState(satellite_id=sat.id)
            for sat in satellites
        }
        self._health: Dict[str, SatelliteHealth] = {
            sat.id: SatelliteHealth.HEALTHY for sat in satellites
        }

    def get_available_count(self) -> int:
        """Get number of available (unallocated) satellites"""
        return sum(1 for allocated in self._allocated.values() if not allocated)

    def get_total_count(self) -> int:
        """Get total number of satellites"""
        return len(self.satellites)

    def allocate_satellite(self, requirements: Dict[str, Any]) -> Optional[Satellite]:
        """Allocate a satellite matching requirements

        Args:
            requirements: Dictionary specifying requirements:
                - type: 'optical' or 'sar'
                - storage_requirement: Minimum storage capacity (GB)
                - power_requirement: Minimum power capacity (Wh)
                - imaging_modes: Required imaging modes

        Returns:
            Allocated satellite or None if no match
        """
        # Filter by availability and health
        candidates = [
            sat_id for sat_id, sat in self.satellites.items()
            if not self._allocated[sat_id]
            and self._health[sat_id] in [SatelliteHealth.HEALTHY, SatelliteHealth.DEGRADED]
        ]

        if not candidates:
            logger.warning("No available satellites for allocation")
            return None

        # Filter by type if specified
        if 'type' in requirements:
            type_filter = requirements['type'].lower()
            candidates = [
                sat_id for sat_id in candidates
                if self._matches_type(self.satellites[sat_id], type_filter)
            ]

        if not candidates:
            logger.warning(f"No satellites matching type: {requirements['type']}")
            return None

        # Filter by storage capacity if specified
        if 'storage_requirement' in requirements:
            min_storage = requirements['storage_requirement']
            candidates = [
                sat_id for sat_id in candidates
                if self.satellites[sat_id].capabilities.storage_capacity >= min_storage
            ]

        if not candidates:
            logger.warning("No satellites with sufficient storage")
            return None

        # Select best candidate (prefer healthy over degraded)
        # Sort by health status then by ID for consistency
        def sort_key(sat_id):
            health_order = {
                SatelliteHealth.HEALTHY: 0,
                SatelliteHealth.DEGRADED: 1
            }
            return (health_order.get(self._health[sat_id], 2), sat_id)

        candidates.sort(key=sort_key)
        selected_id = candidates[0]

        # Mark as allocated
        self._allocated[selected_id] = True
        logger.info(f"Allocated satellite {selected_id}")

        return self.satellites[selected_id]

    def _matches_type(self, satellite: Satellite, type_filter: str) -> bool:
        """Check if satellite matches type filter"""
        sat_type = satellite.satellite_type

        if type_filter == 'optical':
            return sat_type in [SatelliteType.OPTICAL_1, SatelliteType.OPTICAL_2]
        elif type_filter == 'sar':
            return sat_type in [SatelliteType.SAR_1, SatelliteType.SAR_2]
        elif type_filter == 'optical_1':
            return sat_type == SatelliteType.OPTICAL_1
        elif type_filter == 'optical_2':
            return sat_type == SatelliteType.OPTICAL_2
        elif type_filter == 'sar_1':
            return sat_type == SatelliteType.SAR_1
        elif type_filter == 'sar_2':
            return sat_type == SatelliteType.SAR_2

        return False

    def release_satellite(self, satellite_id: str) -> bool:
        """Release an allocated satellite

        Args:
            satellite_id: Satellite ID to release

        Returns:
            True if released successfully
        """
        if satellite_id not in self._allocated:
            logger.warning(f"Unknown satellite: {satellite_id}")
            return False

        if not self._allocated[satellite_id]:
            logger.warning(f"Satellite {satellite_id} is not allocated")
            return False

        self._allocated[satellite_id] = False
        self._states[satellite_id].current_task = None
        logger.info(f"Released satellite {satellite_id}")
        return True

    def get_available_satellites(self, time_window: Tuple[datetime, datetime]) -> List[Satellite]:
        """Get list of available satellites for a time window

        Args:
            time_window: (start_time, end_time)

        Returns:
            List of available satellites
        """
        return [
            self.satellites[sat_id]
            for sat_id, allocated in self._allocated.items()
            if not allocated and self._health[sat_id] != SatelliteHealth.OFFLINE
        ]

    def update_satellite_state(self, satellite_id: str, state_update: Dict[str, Any]) -> bool:
        """Update satellite runtime state

        Args:
            satellite_id: Satellite ID
            state_update: Dictionary with state fields to update

        Returns:
            True if updated successfully
        """
        if satellite_id not in self._states:
            logger.warning(f"Unknown satellite: {satellite_id}")
            return False

        state = self._states[satellite_id]

        if 'storage_used' in state_update:
            state.storage_used = state_update['storage_used']
        if 'power_level' in state_update:
            state.power_level = state_update['power_level']
        if 'current_task' in state_update:
            state.current_task = state_update['current_task']

        state.last_updated = datetime.now()
        return True

    def get_satellite_state(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Get satellite current state

        Args:
            satellite_id: Satellite ID

        Returns:
            State dictionary or None if not found
        """
        if satellite_id not in self._states:
            return None

        state = self._states[satellite_id]
        return {
            'satellite_id': state.satellite_id,
            'storage_used': state.storage_used,
            'power_level': state.power_level,
            'current_task': state.current_task,
            'health': state.health.value,
            'allocated': self._allocated.get(satellite_id, False),
            'last_updated': state.last_updated
        }

    def update_satellite_health(self, satellite_id: str, health: SatelliteHealth) -> bool:
        """Update satellite health status

        Args:
            satellite_id: Satellite ID
            health: New health status

        Returns:
            True if updated successfully
        """
        if satellite_id not in self._health:
            logger.warning(f"Unknown satellite: {satellite_id}")
            return False

        self._health[satellite_id] = health
        self._states[satellite_id].health = health
        logger.info(f"Updated {satellite_id} health to {health.value}")
        return True

    def check_satellite_health(self, satellite_id: str) -> SatelliteHealth:
        """Check satellite health status

        Args:
            satellite_id: Satellite ID

        Returns:
            Health status (defaults to OFFLINE if unknown)
        """
        return self._health.get(satellite_id, SatelliteHealth.OFFLINE)

    def get_allocation_status(self) -> Dict[str, Any]:
        """Get overall allocation status

        Returns:
            Dictionary with allocation statistics
        """
        total = len(self.satellites)
        allocated = sum(1 for a in self._allocated.values() if a)
        available = total - allocated

        health_counts = {
            'healthy': sum(1 for h in self._health.values() if h == SatelliteHealth.HEALTHY),
            'degraded': sum(1 for h in self._health.values() if h == SatelliteHealth.DEGRADED),
            'unhealthy': sum(1 for h in self._health.values() if h == SatelliteHealth.UNHEALTHY),
            'offline': sum(1 for h in self._health.values() if h == SatelliteHealth.OFFLINE)
        }

        return {
            'total_satellites': total,
            'allocated': allocated,
            'available': available,
            'allocation_rate': allocated / total if total > 0 else 0,
            'health_summary': health_counts
        }
