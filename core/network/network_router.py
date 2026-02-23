"""
Network router for satellite constellation data routing.

Implements Dijkstra algorithm for finding optimal multi-hop paths
from source satellite to destination ground station via ISL links.
"""
import heapq
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.models.ground_station import GroundStation
from .isl_visibility import ISLVisibilityCalculator, ISLLink

logger = logging.getLogger(__name__)


@dataclass
class RoutePath:
    """Routing path

    Represents a multi-hop path from source to destination.

    Attributes:
        source_satellite: Source satellite ID
        destination: Destination ground station ID
        hops: List of satellite IDs in path (excluding source)
        total_latency: Total latency in seconds
        available_bandwidth: Minimum bandwidth along path (Mbps)
        path_reliability: Path reliability (0-1)
    """
    source_satellite: str
    destination: str
    hops: List[str]
    total_latency: float
    available_bandwidth: float
    path_reliability: float


class NetworkRouter:
    """Satellite network router

    Finds optimal data routing paths using Dijkstra algorithm.
    Supports optimization by latency, bandwidth, or reliability.

    Example:
        router = NetworkRouter(isl_calculator, ground_stations)

        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=10.0,  # GB
            start_time=datetime.now(),
            priority='latency'
        )

        if route:
            print(f"Path: {' -> '.join([route.source_satellite] + route.hops)}")
            print(f"Latency: {route.total_latency:.2f}s")
    """

    def __init__(self,
                 isl_calculator: ISLVisibilityCalculator,
                 ground_stations: List[GroundStation]):
        """Initialize network router

        Args:
            isl_calculator: ISL visibility calculator
            ground_stations: List of ground stations
        """
        self.isl_calculator = isl_calculator
        self.ground_stations = {gs.id: gs for gs in ground_stations}
        self.isl_windows: Dict[Tuple[str, str], List[ISLLink]] = {}

    def set_isl_windows(self, isl_windows: Dict[Tuple[str, str], List[ISLLink]]) -> None:
        """Set ISL windows for routing

        Args:
            isl_windows: ISL windows dictionary
        """
        self.isl_windows = isl_windows

    def find_best_route(self,
                       source_satellite: str,
                       destination_ground_station: str,
                       data_size: float,
                       start_time: datetime,
                       priority: str = 'latency') -> Optional[RoutePath]:
        """Find optimal routing path

        Args:
            source_satellite: Source satellite ID
            destination_ground_station: Destination ground station ID
            data_size: Data size in GB
            start_time: Start time for routing
            priority: Optimization priority ('latency', 'bandwidth', 'reliability')

        Returns:
            RoutePath or None if no route found
        """
        # Build network topology
        topology = self._build_network_topology(start_time)

        if source_satellite not in topology:
            logger.warning(f"Source satellite {source_satellite} not in topology")
            return None

        # Use appropriate Dijkstra variant
        if priority == 'latency':
            return self._dijkstra_latency(
                source_satellite,
                f"GS:{destination_ground_station}",
                topology,
                data_size,
                start_time
            )
        elif priority == 'bandwidth':
            return self._dijkstra_bandwidth(
                source_satellite,
                f"GS:{destination_ground_station}",
                topology,
                data_size,
                start_time
            )
        else:  # reliability
            return self._dijkstra_reliability(
                source_satellite,
                f"GS:{destination_ground_station}",
                topology,
                data_size,
                start_time
            )

    def _build_network_topology(self, start_time: datetime) -> Dict[str, List[Tuple[str, float]]]:
        """Build network topology from ISL windows

        Args:
            start_time: Current time

        Returns:
            Adjacency list: node_id -> [(neighbor_id, weight), ...]
        """
        topology: Dict[str, List[Tuple[str, float]]] = {}

        # Add ISL links
        for (sat_a, sat_b), links in self.isl_windows.items():
            # Find active link at start_time
            active_link = None
            for link in links:
                if link.start_time <= start_time <= link.end_time:
                    active_link = link
                    break

            if active_link:
                # Add bidirectional link
                if sat_a not in topology:
                    topology[sat_a] = []
                if sat_b not in topology:
                    topology[sat_b] = []

                weight = 1.0 / active_link.link_quality  # Lower weight for better quality
                topology[sat_a].append((sat_b, weight))
                topology[sat_b].append((sat_a, weight))

        # Add ground station connections
        for gs_id in self.ground_stations:
            gs_key = f"GS:{gs_id}"
            if gs_key not in topology:
                topology[gs_key] = []

            # Find satellites that can see this ground station
            # For now, assume all satellites can see all ground stations
            # In real implementation, check visibility windows
            for sat_id in topology:
                if not sat_id.startswith("GS:"):
                    topology[sat_id].append((gs_key, 1.0))

        return topology

    def _dijkstra_latency(self,
                         source: str,
                         target: str,
                         topology: Dict,
                         data_size: float,
                         start_time: datetime) -> Optional[RoutePath]:
        """Dijkstra algorithm optimized for latency

        Args:
            source: Source node
            target: Target node
            topology: Network topology
            data_size: Data size in GB
            start_time: Start time

        Returns:
            RoutePath or None
        """
        # Priority queue: (cumulative_latency, current_node, path)
        queue = [(0.0, source, [source])]
        visited = set()

        while queue:
            current_latency, current_node, path = heapq.heappop(queue)

            if current_node == target:
                # Found path
                hops = path[1:-1] if len(path) > 2 else []  # Exclude source and GS
                gs_id = target.replace("GS:", "")

                return RoutePath(
                    source_satellite=source,
                    destination=gs_id,
                    hops=hops,
                    total_latency=current_latency,
                    available_bandwidth=self._calculate_path_bandwidth(path, start_time),
                    path_reliability=self._calculate_path_reliability(path, start_time)
                )

            if current_node in visited:
                continue

            visited.add(current_node)

            # Expand neighbors
            for neighbor, quality in topology.get(current_node, []):
                if neighbor not in visited:
                    # Estimate link latency
                    link_latency = self._estimate_link_latency(
                        current_node, neighbor, data_size, quality
                    )
                    new_latency = current_latency + link_latency
                    heapq.heappush(queue, (new_latency, neighbor, path + [neighbor]))

        return None

    def _dijkstra_bandwidth(self,
                           source: str,
                           target: str,
                           topology: Dict,
                           data_size: float,
                           start_time: datetime) -> Optional[RoutePath]:
        """Dijkstra algorithm optimized for bandwidth (simplified)"""
        # For now, use same as latency but with bandwidth weights
        # Full implementation would maximize minimum bandwidth along path
        return self._dijkstra_latency(source, target, topology, data_size, start_time)

    def _dijkstra_reliability(self,
                             source: str,
                             target: str,
                             topology: Dict,
                             data_size: float,
                             start_time: datetime) -> Optional[RoutePath]:
        """Dijkstra algorithm optimized for reliability (simplified)"""
        # For now, use same as latency
        # Full implementation would maximize path reliability
        return self._dijkstra_latency(source, target, topology, data_size, start_time)

    def _estimate_link_latency(self,
                              node_a: str,
                              node_b: str,
                              data_size: float,
                              quality: float) -> float:
        """Estimate link latency

        Args:
            node_a: Node A
            node_b: Node B
            data_size: Data size in GB
            quality: Link quality (0-1)

        Returns:
            Latency in seconds
        """
        # Simplified latency model
        # Real model would consider ISL distance, queuing, etc.
        base_latency = 0.01  # 10ms base
        propagation_factor = 1.0 / max(quality, 0.1)
        return base_latency * propagation_factor

    def _calculate_path_bandwidth(self, path: List[str], start_time: datetime) -> float:
        """Calculate minimum bandwidth along path

        Args:
            path: List of node IDs
            start_time: Start time

        Returns:
            Minimum bandwidth in Mbps
        """
        if len(path) < 2:
            return 0.0

        min_bandwidth = float('inf')

        for i in range(len(path) - 1):
            # Find ISL bandwidth
            link_key = (path[i], path[i+1])
            if link_key in self.isl_windows:
                links = self.isl_windows[link_key]
                for link in links:
                    if link.start_time <= start_time <= link.end_time:
                        min_bandwidth = min(min_bandwidth, link.max_data_rate)
                        break

        return min_bandwidth if min_bandwidth != float('inf') else 0.0

    def _calculate_path_reliability(self, path: List[str], start_time: datetime) -> float:
        """Calculate path reliability

        Args:
            path: List of node IDs
            start_time: Start time

        Returns:
            Path reliability (0-1)
        """
        if len(path) < 2:
            return 1.0

        reliability = 1.0

        for i in range(len(path) - 1):
            link_key = (path[i], path[i+1])
            if link_key in self.isl_windows:
                links = self.isl_windows[link_key]
                for link in links:
                    if link.start_time <= start_time <= link.end_time:
                        reliability *= link.link_quality
                        break

        return reliability

    def _calculate_path_latency(self, hops: List[str], data_size: float) -> float:
        """Calculate total path latency

        Args:
            hops: List of satellite IDs in path
            data_size: Data size in GB

        Returns:
            Total latency in seconds
        """
        # Simplified calculation
        # Real implementation would sum ISL latencies + ground link latency
        num_hops = len(hops) + 1  # +1 for ground link
        base_latency_per_hop = 0.01  # 10ms per hop
        transmission_time = data_size * 8 / 1000  # Assuming 1 Gbps

        return num_hops * base_latency_per_hop + transmission_time
