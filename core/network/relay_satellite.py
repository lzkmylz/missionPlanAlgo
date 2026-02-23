"""
Relay satellite support (e.g., Tianlian relay constellation).

Provides data relay capabilities through geostationary relay satellites
for continuous communication with low-orbit satellites.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.models.satellite import Satellite
from core.models.ground_station import GroundStation

logger = logging.getLogger(__name__)


@dataclass
class RelaySatellite:
    """Relay satellite (e.g., Tianlian)

    Attributes:
        id: Relay satellite ID
        name: Human-readable name
        orbit_type: Orbit type (typically 'GEO')
        longitude: Geostationary longitude (degrees)
        uplink_capacity: Uplink capacity (Mbps)
        downlink_capacity: Downlink capacity (Mbps)
        coverage_zones: List of coverage zone names
    """
    id: str
    name: str
    orbit_type: str
    longitude: float
    uplink_capacity: float
    downlink_capacity: float
    coverage_zones: List[str] = field(default_factory=list)


class RelayNetwork:
    """Relay satellite network

    Manages relay satellites for data relay and command uplink.
    Supports Tianlian-style geostationary relay constellations.

    Example:
        relays = [
            RelaySatellite(id='RELAY-01', name='Tianlian-1', ...),
            RelaySatellite(id='RELAY-02', name='Tianlian-2', ...)
        ]
        network = RelayNetwork(relays)

        # Check if relay is available
        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=10.0,
            start_time=datetime.now()
        )
    """

    def __init__(self, relay_satellites: List[RelaySatellite]):
        """Initialize relay network

        Args:
            relay_satellites: List of relay satellites
        """
        self.relays: Dict[str, RelaySatellite] = {
            relay.id: relay for relay in relay_satellites
        }
        # Visibility windows: (sat_id, relay_id) -> [visibility_windows]
        self.relay_visibility: Dict[Tuple[str, str], List] = {}

    def add_visibility_window(self,
                              satellite_id: str,
                              relay_id: str,
                              start_time: datetime,
                              end_time: datetime) -> None:
        """Add a visibility window between satellite and relay

        Args:
            satellite_id: LEO satellite ID
            relay_id: Relay satellite ID
            start_time: Window start
            end_time: Window end
        """
        key = (satellite_id, relay_id)
        if key not in self.relay_visibility:
            self.relay_visibility[key] = []

        self.relay_visibility[key].append({
            'start_time': start_time,
            'end_time': end_time
        })

    def can_relay_data(self,
                      source_satellite: str,
                      relay_id: str,
                      data_size: float,
                      start_time: datetime) -> Tuple[bool, float]:
        """Check if data can be relayed through relay satellite

        Args:
            source_satellite: Source LEO satellite ID
            relay_id: Relay satellite ID
            data_size: Data size in GB
            start_time: Start time for relay

        Returns:
            Tuple of (can_relay, latency_seconds)
        """
        if relay_id not in self.relays:
            return False, float('inf')

        relay = self.relays[relay_id]

        # Check visibility
        windows = self.relay_visibility.get((source_satellite, relay_id), [])

        for window in windows:
            if window['start_time'] <= start_time <= window['end_time']:
                # Calculate transfer time
                bandwidth = min(relay.uplink_capacity, relay.downlink_capacity)

                # Handle zero bandwidth case
                if bandwidth <= 0:
                    return False, float('inf')

                transfer_time = (data_size * 8000) / bandwidth  # seconds

                # Check if transfer fits in window
                end_time = start_time + timedelta(seconds=transfer_time)
                if end_time <= window['end_time']:
                    return True, transfer_time

        return False, float('inf')

    def get_available_relays(self,
                            satellite_id: str,
                            start_time: datetime) -> List[str]:
        """Get list of relays visible to satellite at given time

        Args:
            satellite_id: Satellite ID
            start_time: Time to check

        Returns:
            List of relay IDs
        """
        available = []

        for relay_id in self.relays:
            windows = self.relay_visibility.get((satellite_id, relay_id), [])
            for window in windows:
                if window['start_time'] <= start_time <= window['end_time']:
                    available.append(relay_id)
                    break

        return available

    def find_best_relay(self,
                       satellite_id: str,
                       data_size: float,
                       start_time: datetime) -> Optional[str]:
        """Find best relay for data transfer

        Args:
            satellite_id: Source satellite ID
            data_size: Data size in GB
            start_time: Start time

        Returns:
            Best relay ID or None
        """
        best_relay = None
        min_latency = float('inf')

        for relay_id in self.relays:
            can_relay, latency = self.can_relay_data(
                satellite_id, relay_id, data_size, start_time
            )
            if can_relay and latency < min_latency:
                min_latency = latency
                best_relay = relay_id

        return best_relay

    def calculate_relay_coverage(self, longitude: float) -> List[str]:
        """Calculate which relays cover a given longitude

        Args:
            longitude: Earth longitude (degrees)

        Returns:
            List of relay IDs that cover this longitude
        """
        covering = []

        for relay_id, relay in self.relays.items():
            # Simplified coverage model
            # GEO satellites typically cover ±60° from their longitude
            coverage_range = 60.0

            lon_diff = abs(relay.longitude - longitude)
            if lon_diff > 180:  # Handle wrap-around
                lon_diff = 360 - lon_diff

            if lon_diff <= coverage_range:
                covering.append(relay_id)

        return covering
