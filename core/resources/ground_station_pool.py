"""
Ground station resource pool management.

Manages antenna allocation and scheduling for ground station contacts.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from core.models.ground_station import GroundStation, Antenna

logger = logging.getLogger(__name__)


@dataclass
class AntennaAllocation:
    """Antenna allocation record"""
    antenna_id: str
    satellite_id: str
    start_time: datetime
    end_time: datetime


class GroundStationPool:
    """Ground station resource pool manager

    Manages antenna allocation for satellite downlink/uplink operations.
    Tracks antenna availability over time and resolves conflicts.

    Example:
        pool = GroundStationPool(ground_stations)

        # Allocate antenna for downlink
        start = datetime.now()
        end = start + timedelta(minutes=10)

        antenna = pool.allocate_antenna(
            satellite_id='SAT-01',
            time_window=(start, end)
        )

        if antenna:
            # Use antenna...
            pool.release_antenna(antenna.id, start, end)
    """

    def __init__(self, ground_stations: List[GroundStation]):
        """Initialize ground station pool

        Args:
            ground_stations: List of ground stations
        """
        self.stations: Dict[str, GroundStation] = {
            gs.id: gs for gs in ground_stations
        }

        # Collect all antennas
        self.antennas: Dict[str, Antenna] = {}
        self.antenna_to_station: Dict[str, str] = {}

        for gs in ground_stations:
            for ant in gs.antennas:
                self.antennas[ant.id] = ant
                self.antenna_to_station[ant.id] = gs.id

        # Track allocations: antenna_id -> list of (start, end, satellite_id)
        self._allocations: Dict[str, List[Tuple[datetime, datetime, str]]] = {
            ant_id: [] for ant_id in self.antennas
        }

    def get_total_antenna_count(self) -> int:
        """Get total number of antennas"""
        return len(self.antennas)

    def get_available_antenna_count(self) -> int:
        """Get number of currently unallocated antennas"""
        # This is a simplification - antennas are time-multiplexed
        return len(self.antennas)

    def allocate_antenna(self,
                         satellite_id: str,
                         time_window: Tuple[datetime, datetime],
                         ground_station_id: Optional[str] = None,
                         min_data_rate: Optional[float] = None) -> Optional[Antenna]:
        """Allocate an antenna for satellite contact

        Args:
            satellite_id: Satellite requesting contact
            time_window: (start_time, end_time) tuple
            ground_station_id: Optional specific station to use
            min_data_rate: Minimum required data rate (Mbps)

        Returns:
            Allocated antenna or None if no suitable antenna available
        """
        start_time, end_time = time_window

        # Get candidate antennas
        if ground_station_id:
            # Use specific station
            if ground_station_id not in self.stations:
                logger.warning(f"Unknown ground station: {ground_station_id}")
                return None

            candidates = [
                ant.id for ant in self.stations[ground_station_id].antennas
            ]
        else:
            # Use any station
            candidates = list(self.antennas.keys())

        # Filter by data rate if specified
        if min_data_rate:
            candidates = [
                ant_id for ant_id in candidates
                if self.antennas[ant_id].data_rate >= min_data_rate
            ]

        if not candidates:
            logger.warning("No antennas meet data rate requirement")
            return None

        # Filter by availability
        available = [
            ant_id for ant_id in candidates
            if self.is_antenna_available(ant_id, start_time, end_time)
        ]

        if not available:
            logger.warning(f"No antennas available in time window {time_window}")
            return None

        # Select first available (could be improved with optimization)
        selected_id = available[0]

        # Record allocation
        self._allocations[selected_id].append((start_time, end_time, satellite_id))
        logger.info(f"Allocated antenna {selected_id} to satellite {satellite_id}")

        return self.antennas[selected_id]

    def release_antenna(self,
                        antenna_id: str,
                        start_time: datetime,
                        end_time: datetime) -> bool:
        """Release an antenna allocation

        Args:
            antenna_id: Antenna ID to release
            start_time: Start of allocation window
            end_time: End of allocation window

        Returns:
            True if released successfully
        """
        if antenna_id not in self._allocations:
            logger.warning(f"Unknown antenna: {antenna_id}")
            return False

        allocations = self._allocations[antenna_id]

        # Find and remove matching allocation
        for i, (start, end, sat_id) in enumerate(allocations):
            if start == start_time and end == end_time:
                allocations.pop(i)
                logger.info(f"Released antenna {antenna_id}")
                return True

        logger.warning(f"No matching allocation found for antenna {antenna_id}")
        return False

    def is_antenna_available(self,
                             antenna_id: str,
                             start_time: datetime,
                             end_time: datetime) -> bool:
        """Check if antenna is available in time window

        Args:
            antenna_id: Antenna ID to check
            start_time: Start of requested window
            end_time: End of requested window

        Returns:
            True if antenna is available
        """
        if antenna_id not in self._allocations:
            return False

        # Check for conflicts with existing allocations
        for alloc_start, alloc_end, _ in self._allocations[antenna_id]:
            # Check for overlap
            if start_time < alloc_end and end_time > alloc_start:
                # Time windows overlap
                return False

        return True

    def get_available_antennas(self,
                               ground_station_id: Optional[str] = None,
                               time_window: Optional[Tuple[datetime, datetime]] = None) -> List[Antenna]:
        """Get list of available antennas

        Args:
            ground_station_id: Optional station filter
            time_window: Optional time window filter

        Returns:
            List of available antennas
        """
        candidates = []

        for ant_id, antenna in self.antennas.items():
            # Filter by station if specified
            if ground_station_id:
                if self.antenna_to_station[ant_id] != ground_station_id:
                    continue

            # Filter by availability if time window specified
            if time_window:
                start, end = time_window
                if not self.is_antenna_available(ant_id, start, end):
                    continue

            candidates.append(antenna)

        return candidates

    def get_station_for_antenna(self, antenna_id: str) -> Optional[str]:
        """Get ground station ID for an antenna

        Args:
            antenna_id: Antenna ID

        Returns:
            Ground station ID or None
        """
        return self.antenna_to_station.get(antenna_id)

    def get_station(self, station_id: str) -> Optional[GroundStation]:
        """Get ground station by ID

        Args:
            station_id: Station ID

        Returns:
            GroundStation or None
        """
        return self.stations.get(station_id)

    def get_allocation_status(self) -> Dict[str, Any]:
        """Get overall allocation status

        Returns:
            Dictionary with allocation statistics
        """
        total_allocations = sum(
            len(allocations) for allocations in self._allocations.values()
        )

        station_stats = {}
        for gs_id, gs in self.stations.items():
            ant_count = len(gs.antennas)
            active_allocs = sum(
                len(self._allocations[ant.id]) for ant in gs.antennas
            )
            station_stats[gs_id] = {
                'antenna_count': ant_count,
                'active_allocations': active_allocs
            }

        return {
            'total_stations': len(self.stations),
            'total_antennas': len(self.antennas),
            'total_active_allocations': total_allocations,
            'station_details': station_stats
        }

    def find_best_station(self,
                         satellite_location: Tuple[float, float],
                         time_window: Tuple[datetime, datetime]) -> Optional[str]:
        """Find best ground station for satellite contact

        Args:
            satellite_location: (longitude, latitude) of satellite
            time_window: Contact time window

        Returns:
            Best ground station ID or None
        """
        # Simple implementation - choose station with most available antennas
        best_station = None
        max_available = 0

        for gs_id in self.stations:
            available = len(self.get_available_antennas(gs_id, time_window))
            if available > max_available:
                max_available = available
                best_station = gs_id

        return best_station
