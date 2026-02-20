"""
Unified resource allocator for satellite mission planning.

Coordinates allocation of satellites and ground station resources.
Provides high-level interface for complete task resource allocation.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from core.models.satellite import Satellite
from core.models.ground_station import GroundStation, Antenna
from core.models.target import Target
from .satellite_pool import SatellitePool
from .ground_station_pool import GroundStationPool

logger = logging.getLogger(__name__)


@dataclass
class ResourceAllocation:
    """Complete resource allocation for a task"""
    task_id: str
    satellite_id: str
    antenna_id: Optional[str] = None
    downlink_station_id: Optional[str] = None
    imaging_window: Optional[Tuple[datetime, datetime]] = None
    downlink_window: Optional[Tuple[datetime, datetime]] = None


class ResourceAllocator:
    """Unified resource allocator

    Coordinates satellite and ground station resource allocation
    for imaging tasks with optional data downlink.

    Example:
        allocator = ResourceAllocator(satellites, ground_stations)

        # Allocate resources for task
        allocation = allocator.allocate_task_resources(
            task=target,
            imaging_window=(start, end),
            satellite_requirements={'type': 'optical'}
        )

        if allocation:
            print(f"Allocated satellite: {allocation.satellite_id}")
            print(f"Allocated antenna: {allocation.antenna_id}")

            # Release when done
            allocator.release_task_resources(task.id, allocation)
    """

    def __init__(self,
                 satellites: List[Satellite],
                 ground_stations: List[GroundStation]):
        """Initialize resource allocator

        Args:
            satellites: List of available satellites
            ground_stations: List of ground stations
        """
        self.satellite_pool = SatellitePool(satellites)
        self.ground_station_pool = GroundStationPool(ground_stations)
        self._allocations: Dict[str, ResourceAllocation] = {}

    def allocate_task_resources(self,
                                task: Target,
                                imaging_window: Tuple[datetime, datetime],
                                downlink_window: Optional[Tuple[datetime, datetime]] = None,
                                satellite_requirements: Optional[Dict[str, Any]] = None) -> Optional[ResourceAllocation]:
        """Allocate all resources for a task

        Args:
            task: Target task to allocate resources for
            imaging_window: (start, end) for imaging
            downlink_window: Optional (start, end) for data downlink
            satellite_requirements: Requirements for satellite selection

        Returns:
            ResourceAllocation or None if allocation failed
        """
        requirements = satellite_requirements or {}

        # Allocate satellite
        satellite = self.satellite_pool.allocate_satellite(requirements)
        if not satellite:
            logger.warning(f"Could not allocate satellite for task {task.id}")
            return None

        satellite_id = satellite.id

        # Update satellite state
        self.satellite_pool.update_satellite_state(satellite_id, {
            'current_task': task.id
        })

        # Allocate antenna for downlink if needed
        antenna_id = None
        downlink_station_id = None

        if downlink_window:
            antenna = self.ground_station_pool.allocate_antenna(
                satellite_id=satellite_id,
                time_window=downlink_window
            )

            if antenna:
                antenna_id = antenna.id
                downlink_station_id = self.ground_station_pool.get_station_for_antenna(antenna_id)
            else:
                logger.warning(f"Could not allocate antenna for task {task.id}")
                # Continue without downlink - task can still proceed

        # Create allocation record
        allocation = ResourceAllocation(
            task_id=task.id,
            satellite_id=satellite_id,
            antenna_id=antenna_id,
            downlink_station_id=downlink_station_id,
            imaging_window=imaging_window,
            downlink_window=downlink_window
        )

        self._allocations[task.id] = allocation
        logger.info(f"Allocated resources for task {task.id}: satellite={satellite_id}, antenna={antenna_id}")

        return allocation

    def release_task_resources(self, task_id: str, allocation: ResourceAllocation) -> bool:
        """Release resources allocated to a task

        Args:
            task_id: Task ID
            allocation: ResourceAllocation to release

        Returns:
            True if released successfully
        """
        success = True

        # Release satellite
        if not self.satellite_pool.release_satellite(allocation.satellite_id):
            logger.warning(f"Failed to release satellite for task {task_id}")
            success = False

        # Release antenna if allocated
        if allocation.antenna_id and allocation.downlink_window:
            start, end = allocation.downlink_window
            if not self.ground_station_pool.release_antenna(
                allocation.antenna_id, start, end
            ):
                logger.warning(f"Failed to release antenna for task {task_id}")
                success = False

        # Remove from tracking
        if task_id in self._allocations:
            del self._allocations[task_id]

        logger.info(f"Released resources for task {task_id}")
        return success

    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization statistics

        Returns:
            Dictionary with utilization statistics
        """
        sat_stats = self.satellite_pool.get_allocation_status()
        gs_stats = self.ground_station_pool.get_allocation_status()

        total_satellites = sat_stats['total_satellites']
        allocated_satellites = sat_stats['allocated']
        satellite_utilization = (allocated_satellites / total_satellites
                                if total_satellites > 0 else 0)

        total_antennas = gs_stats['total_antennas']
        active_allocations = gs_stats['total_active_allocations']
        # Simplified antenna utilization calculation
        antenna_utilization = (active_allocations / total_antennas
                              if total_antennas > 0 else 0)

        return {
            'total_satellites': total_satellites,
            'allocated_satellites': allocated_satellites,
            'satellite_utilization': satellite_utilization,
            'total_antennas': total_antennas,
            'active_antenna_allocations': active_allocations,
            'antenna_utilization': antenna_utilization,
            'active_tasks': len(self._allocations),
            'satellite_health_summary': sat_stats.get('health_summary', {})
        }

    def get_allocation(self, task_id: str) -> Optional[ResourceAllocation]:
        """Get allocation for a task

        Args:
            task_id: Task ID

        Returns:
            ResourceAllocation or None
        """
        return self._allocations.get(task_id)

    def is_satellite_available(self, satellite_id: str) -> bool:
        """Check if satellite is available

        Args:
            satellite_id: Satellite ID

        Returns:
            True if available
        """
        state = self.satellite_pool.get_satellite_state(satellite_id)
        if not state:
            return False
        return not state['allocated']

    def is_antenna_available(self, antenna_id: str,
                             start_time: datetime,
                             end_time: datetime) -> bool:
        """Check if antenna is available

        Args:
            antenna_id: Antenna ID
            start_time: Start of window
            end_time: End of window

        Returns:
            True if available
        """
        return self.ground_station_pool.is_antenna_available(
            antenna_id, start_time, end_time
        )
