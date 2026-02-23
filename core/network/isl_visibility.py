"""
ISL (Inter-Satellite Link) visibility calculator.

Calculates visibility windows between satellites for laser/microwave links.
Supports both intra-orbital and cross-orbital ISL connections.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.models.satellite import Satellite

logger = logging.getLogger(__name__)


@dataclass
class ISLLink:
    """Inter-Satellite Link

    Represents a link between two satellites during a visibility window.

    Attributes:
        satellite_a_id: First satellite ID
        satellite_b_id: Second satellite ID
        start_time: Link window start
        end_time: Link window end
        link_quality: Link quality (0-1)
        max_data_rate: Maximum data rate (Mbps)
        distance: Distance between satellites (km)
    """
    satellite_a_id: str
    satellite_b_id: str
    start_time: datetime
    end_time: datetime
    link_quality: float = 1.0
    max_data_rate: float = 10000.0  # 10 Gbps default for laser
    distance: float = 0.0


class ISLVisibilityCalculator:
    """ISL visibility calculator

    Computes visibility windows between satellites for ISL connections.
    Supports laser (10 Gbps) and microwave (1 Gbps) links.

    Example:
        calculator = ISLVisibilityCalculator(
            link_type='laser',
            max_link_distance=5000.0
        )

        windows = calculator.compute_isl_windows(
            satellites=satellite_list,
            start_time=start,
            end_time=end,
            time_step=60
        )

        # windows[(sat_a.id, sat_b.id)] = [ISLLink, ...]
    """

    def __init__(self,
                 link_type: str = 'laser',
                 max_link_distance: float = 5000.0,
                 min_elevation_angle: float = 0.0):
        """Initialize ISL calculator

        Args:
            link_type: 'laser' or 'microwave'
            max_link_distance: Maximum link distance (km)
            min_elevation_angle: Minimum elevation angle (degrees)
        """
        self.link_type = link_type
        self.max_link_distance = max_link_distance
        self.min_elevation_angle = math.radians(min_elevation_angle)

        # Link parameters
        if link_type == 'laser':
            self.max_data_rate = 10000.0  # 10 Gbps
            self.attenuation_factor = 0.1
        else:  # microwave
            self.max_data_rate = 1000.0   # 1 Gbps
            self.attenuation_factor = 0.5

    def compute_isl_windows(self,
                           satellites: List[Satellite],
                           start_time: datetime,
                           end_time: datetime,
                           time_step: int = 60) -> Dict[Tuple[str, str], List[ISLLink]]:
        """Compute all ISL visibility windows

        Args:
            satellites: List of satellites
            start_time: Start of calculation period
            end_time: End of calculation period
            time_step: Time step in seconds

        Returns:
            Dictionary mapping (sat_a_id, sat_b_id) to list of ISLLink
        """
        isl_windows = {}

        # Compute for each unique pair
        for i, sat_a in enumerate(satellites):
            for sat_b in satellites[i+1:]:
                windows = self._compute_pair_windows(
                    sat_a, sat_b,
                    start_time, end_time,
                    time_step
                )

                if windows:
                    key = (sat_a.id, sat_b.id)
                    isl_windows[key] = windows

        logger.info(f"Computed {len(isl_windows)} ISL pairs")
        return isl_windows

    def _compute_pair_windows(self,
                              sat_a: Satellite,
                              sat_b: Satellite,
                              start_time: datetime,
                              end_time: datetime,
                              time_step: int) -> List[ISLLink]:
        """Compute ISL windows for a satellite pair

        Args:
            sat_a: First satellite
            sat_b: Second satellite
            start_time: Start time
            end_time: End time
            time_step: Time step in seconds

        Returns:
            List of ISLLink objects
        """
        windows = []
        current_window_start = None
        last_visible = False

        t = start_time
        while t <= end_time:
            # Get positions
            pos_a = sat_a.get_position(t)
            pos_b = sat_b.get_position(t)

            # Check visibility
            is_visible, distance = self._check_visibility(pos_a, pos_b)

            if is_visible:
                if not last_visible:
                    # Start new window
                    current_window_start = t
                last_visible = True
            else:
                if last_visible and current_window_start:
                    # End current window
                    window = ISLLink(
                        satellite_a_id=sat_a.id,
                        satellite_b_id=sat_b.id,
                        start_time=current_window_start,
                        end_time=t,
                        distance=distance,
                        max_data_rate=self._calculate_data_rate(distance)
                    )
                    windows.append(window)
                    current_window_start = None
                last_visible = False

            t += timedelta(seconds=time_step)

        # Close final window if open
        if last_visible and current_window_start:
            window = ISLLink(
                satellite_a_id=sat_a.id,
                satellite_b_id=sat_b.id,
                start_time=current_window_start,
                end_time=end_time,
                distance=distance,
                max_data_rate=self._calculate_data_rate(distance)
            )
            windows.append(window)

        return windows

    def _check_visibility(self,
                          pos_a: Tuple[float, float, float],
                          pos_b: Tuple[float, float, float]) -> Tuple[bool, float]:
        """Check if two satellites can see each other

        Args:
            pos_a: Position of satellite A (m)
            pos_b: Position of satellite B (m)

        Returns:
            Tuple of (is_visible, distance_km)
        """
        # Calculate distance
        distance = self._calculate_distance(pos_a, pos_b)

        # Check against max link distance
        if distance > self.max_link_distance:
            return False, distance

        # For now, assume no Earth obstruction for simplicity
        # In real implementation, check if line-of-sight intersects Earth

        return True, distance

    def _calculate_distance(self,
                           pos_a: Tuple[float, float, float],
                           pos_b: Tuple[float, float, float]) -> float:
        """Calculate distance between two positions

        Args:
            pos_a: Position A (m)
            pos_b: Position B (m)

        Returns:
            Distance in km
        """
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]

        distance_m = math.sqrt(dx*dx + dy*dy + dz*dz)
        return distance_m / 1000.0  # Convert to km

    def _calculate_data_rate(self, distance: float) -> float:
        """Calculate achievable data rate based on distance

        Args:
            distance: Distance in km

        Returns:
            Data rate in Mbps
        """
        # Simplified model: rate decreases with distance
        # Real model would consider beam divergence, power, etc.
        distance_factor = max(0.1, 1.0 - distance / self.max_link_distance)
        return self.max_data_rate * distance_factor * self.attenuation_factor
