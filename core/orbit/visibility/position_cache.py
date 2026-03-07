"""Satellite position cache for storing pre-computed positions.

This module provides a cache for storing and retrieving satellite positions
to avoid redundant orbit propagation calculations.
"""

from typing import Any, Dict, Optional, Tuple
from datetime import datetime


class SatellitePositionCache:
    """Cache for storing pre-computed satellite positions.

    This class stores satellite positions and velocities at specific times
to avoid redundant orbit propagation calculations during scheduling.

    Attributes:
        _cache: Dictionary mapping (satellite_id, time) to (position, velocity)
    """

    def __init__(self):
        """Initialize the position cache."""
        # Use nested dict: sat_id -> {time: (position, velocity)}
        self._cache: Dict[str, Dict[datetime, Tuple[Any, Any]]] = {}

    def set_position(
        self,
        sat_id: str,
        time: datetime,
        position: Any,
        velocity: Any
    ) -> None:
        """Store a satellite position in the cache.

        Args:
            sat_id: Satellite ID
            time: Time of the position
            position: Position vector (typically ECEF coordinates)
            velocity: Velocity vector
        """
        if sat_id not in self._cache:
            self._cache[sat_id] = {}
        self._cache[sat_id][time] = (position, velocity)

    def get_position(
        self,
        sat_id: str,
        time: datetime
    ) -> Optional[Tuple[Any, Any]]:
        """Retrieve a satellite position from the cache.

        Args:
            sat_id: Satellite ID
            time: Time of the position

        Returns:
            Tuple of (position, velocity) if found, None otherwise
        """
        sat_cache = self._cache.get(sat_id, {})
        return sat_cache.get(time)

    def has_position(self, sat_id: str, time: datetime) -> bool:
        """Check if a position is cached.

        Args:
            sat_id: Satellite ID
            time: Time of the position

        Returns:
            True if position is in cache, False otherwise
        """
        return time in self._cache.get(sat_id, {})

    def clear(self, sat_id: Optional[str] = None) -> None:
        """Clear the cache.

        Args:
            sat_id: If provided, only clear cache for this satellite.
                   If None, clear entire cache.
        """
        if sat_id is None:
            self._cache.clear()
        else:
            self._cache.pop(sat_id, None)

    def get_satellite_times(self, sat_id: str) -> list:
        """Get all cached times for a satellite.

        Args:
            sat_id: Satellite ID

        Returns:
            List of datetime objects
        """
        return list(self._cache.get(sat_id, {}).keys())

    def __len__(self) -> int:
        """Return total number of cached positions."""
        return sum(len(times) for times in self._cache.values())
