"""
ISL (Inter-Satellite Link) Time-Varying Router — Phase 4.

Implements time-sliced Dijkstra routing through an ISL network to find optimal
multi-hop paths from a source satellite to a ground station.

Key design decisions
--------------------
- Planning window is divided into TIME_SLICE_S (60 s) slices.
- Each slice builds a topology snapshot from active ISL links and GS visibility.
- Time-expanded Dijkstra finds the earliest-delivery path.
- Edge cost = transmission_time + ATP/beam-switch overhead + HOP_PENALTY_S.
- Microwave beams are capacity-constrained (max_beam_count per satellite).
- Laser links are capacity-constrained (max_simultaneous_laser per satellite).
"""

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ISLRoutePath:
    """A routed multi-hop path through the ISL network.

    Attributes:
        source_satellite: ID of the originating satellite.
        destination_gs: Ground-station ID (without ``GS:`` prefix), e.g. ``'GS-BEIJING'``.
        hops: Intermediate satellite IDs (not source, not destination GS).
        link_types: Link type for each hop segment (len == len(hops) + 1).
        bottleneck_bandwidth_mbps: Minimum bandwidth along the path (Mbps).
        total_latency_s: Total propagation + ATP/beam-switch overhead (seconds).
        atp_setup_overhead_s: Cumulative ATP setup time on laser links (seconds).
        path_reliability: Product of per-hop link_quality scores.
        topology_changes: Number of time slices where active links differ from
            the previous slice along this path.
        path_nodes: Full ordered node list: [source] + hops + ['GS:<gs_id>'].
    """
    source_satellite: str
    destination_gs: str
    hops: List[str]
    link_types: List[str]
    bottleneck_bandwidth_mbps: float
    total_latency_s: float
    atp_setup_overhead_s: float
    path_reliability: float
    topology_changes: int
    path_nodes: List[str]

    @property
    def hop_count(self) -> int:
        """Number of relay hops (not counting source or GS)."""
        return len(self.hops)


class ISLRouterWindowCache:
    """Thin wrapper around pre-computed ISL visibility windows for routing.

    This is the router-internal mutable cache used by :class:`TimeVaryingISLRouter`.
    It should not be confused with ``core.network.isl_visibility.ISLWindowCache``,
    which is the enforcement cache that loads validated Java-precomputed data.

    This class is normally populated by adapting Java-precomputed windows into
    the duck-typed ISLLink interface understood by the router.  It can also be
    populated directly in tests or when ISL windows are computed in Python.

    Window key convention: ``(sat_a_id, sat_b_id)`` where ``sat_a_id < sat_b_id``
    lexicographically (the cache normalises on insertion).
    """

    def __init__(self) -> None:
        # {(sat_a, sat_b): [ISLLink, ...]}  — keys always have sat_a < sat_b
        self._windows: Dict[Tuple[str, str], List[Any]] = {}

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_window(self, link: Any) -> None:
        """Insert a single ISLLink window (or any object with the ISLLink
        duck-type interface) into the cache.
        """
        a, b = link.satellite_a_id, link.satellite_b_id
        key = (a, b) if a <= b else (b, a)
        self._windows.setdefault(key, []).append(link)

    def add_windows(self, windows: Dict[Tuple[str, str], List[Any]]) -> None:
        """Bulk-load a ``{(sat_a, sat_b): [links]}`` dictionary."""
        for (a, b), links in windows.items():
            key = (a, b) if a <= b else (b, a)
            existing = self._windows.setdefault(key, [])
            existing.extend(links)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_windows(self, sat_a: str, sat_b: str) -> List[Any]:
        """Return all ISL windows between two satellites (order-insensitive)."""
        key = (sat_a, sat_b) if sat_a <= sat_b else (sat_b, sat_a)
        return self._windows.get(key, [])

    def get_active_links(self, satellite_id: str, at_time: datetime) -> List[Any]:
        """Return all ISL links that are active for *satellite_id* at *at_time*."""
        active: List[Any] = []
        for (a, b), links in self._windows.items():
            if a != satellite_id and b != satellite_id:
                continue
            for lnk in links:
                if lnk.start_time <= at_time <= lnk.end_time:
                    active.append(lnk)
        return active

    def get_all_windows(self) -> Dict[Tuple[str, str], List[Any]]:
        """Return a shallow copy of the full window dictionary."""
        return dict(self._windows)


# Backward-compatibility alias.  New code should use ISLRouterWindowCache.
ISLWindowCache = ISLRouterWindowCache


# ---------------------------------------------------------------------------
# TimeVaryingISLRouter
# ---------------------------------------------------------------------------

class TimeVaryingISLRouter:
    """Routes data through an ISL network using a time-varying graph.

    Algorithm
    ---------
    1. Divide the planning window into ``TIME_SLICE_S``-second slices.
    2. For each slice, build a snapshot topology (active ISL links + GS
       visibility windows).
    3. Run time-expanded Dijkstra: state = (current_node, current_time).
    4. Return the path with the earliest delivery time.

    Edge cost model
    ---------------
    cost = data_transmission_time_s
         + atp_setup_time_s * is_new_link          (laser only, default ~37 s)
         + MICROWAVE_SWITCH_PENALTY_S              (microwave beam switch, 1 s)
         + HOP_PENALTY_S                           (small bias towards fewer hops)

    Capacity constraints
    --------------------
    - Microwave: ``max_beam_count`` per satellite per time slice.
    - Laser: ``max_simultaneous_laser`` per satellite per time slice.
    """

    MICROWAVE_SWITCH_PENALTY_S: float = 1.0
    HOP_PENALTY_S: float = 0.1
    TIME_SLICE_S: float = 60.0

    def __init__(
        self,
        isl_window_cache: ISLRouterWindowCache,
        satellite_isl_configs: Dict[str, Any],  # sat_id -> ISLCapabilityConfig
        gs_visibility_windows: Dict[Tuple[str, str], List[Any]],  # (sat_id, gs_id) -> windows
    ) -> None:
        """
        Args:
            isl_window_cache: Pre-computed ISL windows (ISLWindowCache).
            satellite_isl_configs: Per-satellite ISL capability configs
                (``ISLCapabilityConfig`` objects from ``core.models.isl_config``).
            gs_visibility_windows: Dictionary mapping ``(sat_id, gs_id)`` to a
                list of window objects that each expose ``.start_time`` and
                ``.end_time`` attributes (``datetime``).
        """
        self._cache = isl_window_cache
        self._isl_configs = satellite_isl_configs
        self._gs_windows = gs_visibility_windows

        # Pre-build the set of all GS IDs for fast lookup.
        self._gs_ids: set = {gs_id for (_, gs_id) in gs_visibility_windows}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_route(
        self,
        source_satellite: str,
        data_size_gb: float,
        earliest_start: datetime,
        deadline: datetime,
        target_gs: Optional[str] = None,
    ) -> Optional[ISLRoutePath]:
        """Find the optimal multi-hop path from *source_satellite* to any (or a
        specific) ground station.

        Args:
            source_satellite: ID of the satellite that holds the data.
            data_size_gb: Amount of data to relay (GB).
            earliest_start: Earliest moment when ISL relay may begin.
            deadline: Latest acceptable delivery time.
            target_gs: If given, route only to this GS; otherwise any GS is
                acceptable.

        Returns:
            The best ``ISLRoutePath``, or ``None`` if no viable path was found.
        """
        if target_gs is not None:
            target_gs_ids = [target_gs]
        else:
            # Collect all GS IDs that are visible to at least one satellite
            # during the planning window.
            target_gs_ids = list(self._gs_ids)

        if not target_gs_ids:
            logger.debug("find_route: no ground stations available")
            return None

        result = self._dijkstra_time_expanded(
            source=source_satellite,
            target_gs_ids=target_gs_ids,
            data_size_gb=data_size_gb,
            start_time=earliest_start,
            deadline=deadline,
        )
        if result is None:
            return None
        path, _delivery_time = result
        return path

    def find_all_routes(
        self,
        source_satellite: str,
        data_size_gb: float,
        time_window_start: datetime,
        time_window_end: datetime,
    ) -> List[ISLRoutePath]:
        """Find all viable routes, sorted by total delivery time (ascending).

        Internally runs ``find_route`` to each available GS and collects
        results.

        Returns:
            List of ``ISLRoutePath`` objects (may be empty).
        """
        routes: List[Tuple[float, ISLRoutePath]] = []

        for gs_id in self._gs_ids:
            result = self._dijkstra_time_expanded(
                source=source_satellite,
                target_gs_ids=[gs_id],
                data_size_gb=data_size_gb,
                start_time=time_window_start,
                deadline=time_window_end,
            )
            if result is not None:
                path, delivery_time = result
                routes.append((path.total_latency_s, path))

        routes.sort(key=lambda x: x[0])
        return [r for _, r in routes]

    # ------------------------------------------------------------------
    # Topology snapshot
    # ------------------------------------------------------------------

    def _build_topology_snapshot(
        self,
        snapshot_time: datetime,
        active_beam_usage: Dict[str, int],
    ) -> Dict[str, List[Tuple[str, float, str, Any]]]:
        """Build an adjacency list for the time slice at *snapshot_time*.

        Returns:
            ``{node_id: [(neighbor_id, edge_cost, link_type, isl_link), ...]}``
            where GS nodes are keyed as ``'GS:<gs_id>'``.
        """
        graph: Dict[str, List[Tuple[str, float, str, Any]]] = {}

        # --- ISL links ---
        for (sat_a, sat_b), links in self._cache.get_all_windows().items():
            for lnk in links:
                if not (lnk.start_time <= snapshot_time <= lnk.end_time):
                    continue
                # Check link viability
                is_viable = getattr(lnk, 'is_viable', True)
                if not is_viable:
                    continue

                link_type = getattr(lnk, 'link_type', 'laser')

                # Capacity check for each endpoint
                for src, dst in [(sat_a, sat_b), (sat_b, sat_a)]:
                    if not self._check_link_capacity(src, link_type, active_beam_usage):
                        continue
                    # We use data_size_gb=0 here because cost depends on data
                    # size which is injected at Dijkstra level.  Store the link
                    # object so _dijkstra_time_expanded can compute real cost.
                    graph.setdefault(src, []).append(
                        (dst, 0.0, link_type, lnk)
                    )

        # --- GS downlink edges ---
        for (sat_id, gs_id), windows in self._gs_windows.items():
            for win in windows:
                if win.start_time <= snapshot_time <= win.end_time:
                    gs_node = f"GS:{gs_id}"
                    # GS downlink: we model it as a "link" with infinite
                    # bandwidth relative to ISL; use 300 Mbps as default.
                    data_rate = getattr(win, 'max_data_rate', 300.0)
                    graph.setdefault(sat_id, []).append(
                        (gs_node, 0.0, 'gs_downlink', win)
                    )
                    # GS nodes have no outbound edges.
                    graph.setdefault(gs_node, [])
                    break  # only one window per time slice needed

        return graph

    def _check_link_capacity(
        self,
        sat_id: str,
        link_type: str,
        active_beam_usage: Dict[str, int],
    ) -> bool:
        """Return True if satellite *sat_id* can open another link of *link_type*."""
        isl_cfg = self._isl_configs.get(sat_id)
        if isl_cfg is None:
            return True  # no config → no restriction

        current_usage = active_beam_usage.get(sat_id, 0)

        if link_type == 'laser':
            max_laser = getattr(isl_cfg, 'max_simultaneous_laser', 2)
            return current_usage < max_laser
        else:  # microwave
            mw_cfg = getattr(isl_cfg, 'microwave', None)
            if mw_cfg is None:
                return True
            max_beams = getattr(mw_cfg, 'max_beam_count', 4)
            return current_usage < max_beams

    # ------------------------------------------------------------------
    # Edge cost
    # ------------------------------------------------------------------

    def _compute_edge_cost(
        self,
        link: Any,
        data_size_gb: float,
        is_new_link: bool,
        active_beams_at_sat: int,
    ) -> float:
        """Compute the traversal cost (seconds) for crossing one ISL link.

        Args:
            link: An ISL link object (has ``link_type``, ``max_data_rate``,
                ``atp_setup_time_s``, ``link_quality``).
            data_size_gb: Data payload in GB.
            is_new_link: True if the link was not active in the previous slice
                (triggers ATP setup for laser links).
            active_beams_at_sat: Current number of active beams at the source
                satellite (used for microwave switch penalty).

        Returns:
            Cost in seconds.
        """
        link_type = getattr(link, 'link_type', 'laser')
        data_rate_mbps = max(getattr(link, 'max_data_rate', 1000.0), 1.0)

        # Transmission time: GB → Mb → seconds
        transmission_s = (data_size_gb * 8.0 * 1024.0) / data_rate_mbps

        if link_type == 'laser':
            atp_time = getattr(link, 'atp_setup_time_s', 37.0)
            atp_cost = atp_time if is_new_link else 0.0
        else:
            # Microwave: beam switch penalty if at least one beam already active
            atp_cost = (
                self.MICROWAVE_SWITCH_PENALTY_S if active_beams_at_sat > 0 else 0.0
            )

        return transmission_s + atp_cost + self.HOP_PENALTY_S

    # ------------------------------------------------------------------
    # Time-expanded Dijkstra
    # ------------------------------------------------------------------

    def _dijkstra_time_expanded(
        self,
        source: str,
        target_gs_ids: List[str],
        data_size_gb: float,
        start_time: datetime,
        deadline: datetime,
    ) -> Optional[Tuple[ISLRoutePath, datetime]]:
        """Time-expanded Dijkstra over the ISL network.

        State: ``(current_node, current_time_seconds_offset)``

        The algorithm iterates over time slices of ``TIME_SLICE_S`` seconds and
        builds a snapshot topology for each slice.  Edges are only traversable
        when both endpoints have the link active in the relevant slice.

        Args:
            source: Source satellite ID.
            target_gs_ids: Acceptable destination GS IDs (without ``GS:``
                prefix).
            data_size_gb: Data payload in GB.
            start_time: Earliest relay start.
            deadline: Latest acceptable delivery time.

        Returns:
            ``(ISLRoutePath, delivery_time)`` or ``None``.
        """
        target_gs_nodes = {f"GS:{g}" for g in target_gs_ids}
        total_window_s = (deadline - start_time).total_seconds()
        if total_window_s <= 0:
            return None

        # Build a list of time-slice start offsets (seconds from start_time).
        slice_offsets: List[float] = []
        t = 0.0
        while t < total_window_s:
            slice_offsets.append(t)
            t += self.TIME_SLICE_S

        if not slice_offsets:
            return None

        # Active beam usage tracking: sat_id -> number of concurrent beams
        # For simplicity we start with zero (no pre-existing ISL tasks).
        active_beam_usage: Dict[str, int] = {}

        # Pre-build topology snapshots for each slice.
        # snapshots[i] = graph at slice_offsets[i]
        snapshots: List[Dict[str, List[Tuple[str, float, str, Any]]]] = []
        for offset in slice_offsets:
            snap_time = start_time + timedelta(seconds=offset)
            snapshots.append(
                self._build_topology_snapshot(snap_time, active_beam_usage)
            )

        # Priority queue: (cost_s, node_id, slice_index, path_nodes, link_types,
        #                  atp_overhead, reliability, bandwidth, prev_links)
        # prev_links: frozenset of (sat_a, sat_b) pairs active in previous slice
        INF = float('inf')
        heap: List[Tuple[float, str, int, List[str], List[str],
                         float, float, float, frozenset]] = []
        heapq.heappush(heap, (
            0.0,                        # cost_s
            source,                     # node
            0,                          # slice index
            [source],                   # path_nodes so far
            [],                         # link_types so far
            0.0,                        # atp_overhead_s
            1.0,                        # reliability
            INF,                        # min bandwidth (will be set at first real edge)
            frozenset(),                # prev_links (sat_a, sat_b) pairs
        ))

        # visited: (node, slice_index) — avoid re-expanding the same state
        visited: Dict[Tuple[str, int], float] = {}

        best_result: Optional[Tuple[ISLRoutePath, datetime]] = None
        best_cost = INF

        while heap:
            (
                cost_s, node, slice_idx, path_nodes, path_link_types,
                atp_overhead, reliability, bw, prev_links
            ) = heapq.heappop(heap)

            # Deadline / best-cost pruning
            if cost_s >= best_cost:
                continue

            state_key = (node, slice_idx)
            if state_key in visited and visited[state_key] <= cost_s:
                continue
            visited[state_key] = cost_s

            # Check if we reached a GS node.
            if node in target_gs_nodes:
                delivery_time = start_time + timedelta(seconds=cost_s)
                if delivery_time <= deadline:
                    gs_id = node[3:]  # strip 'GS:'
                    hops = [n for n in path_nodes[1:-1] if not n.startswith('GS:')]
                    # Compute topology changes
                    topo_changes = self._count_topology_changes(
                        path_nodes, start_time, start_time + timedelta(seconds=cost_s)
                    )
                    final_bw = bw if bw != INF else 0.0
                    route = ISLRoutePath(
                        source_satellite=source,
                        destination_gs=gs_id,
                        hops=hops,
                        link_types=path_link_types,
                        bottleneck_bandwidth_mbps=final_bw,
                        total_latency_s=cost_s,
                        atp_setup_overhead_s=atp_overhead,
                        path_reliability=reliability,
                        topology_changes=topo_changes,
                        path_nodes=list(path_nodes),
                    )
                    if best_result is None or cost_s < best_cost:
                        best_cost = cost_s
                        best_result = (route, delivery_time)
                continue

            # Advance through time slices if the current slice is exhausted.
            # We allow the agent to stay at a node for multiple slices (waiting
            # for a link to open).
            for next_slice_idx in range(slice_idx, len(snapshots)):
                if next_slice_idx > slice_idx:
                    # Waiting cost: add slice duration
                    wait_cost = cost_s + (next_slice_idx - slice_idx) * self.TIME_SLICE_S
                    if wait_cost >= best_cost:
                        break

                graph = snapshots[next_slice_idx]
                neighbors = graph.get(node, [])

                # Build current link set for topology change detection.
                current_links: set = set()
                for (nbr, _, ltype, lnk) in neighbors:
                    if not nbr.startswith('GS:'):
                        a = node
                        b = nbr
                        current_links.add((a, b) if a <= b else (b, a))

                slice_time_offset = slice_offsets[next_slice_idx]
                base_cost = cost_s if next_slice_idx == slice_idx else (
                    cost_s + (next_slice_idx - slice_idx) * self.TIME_SLICE_S
                )

                for (nbr, _, link_type, lnk) in neighbors:
                    if nbr in path_nodes and not nbr.startswith('GS:'):
                        continue  # avoid loops (allow GS revisit)

                    link_bw = getattr(lnk, 'max_data_rate', 300.0)
                    lq = getattr(lnk, 'link_quality', 1.0)

                    # Determine if this is a new link (for ATP cost).
                    pair = (node, nbr) if node <= nbr else (nbr, node)
                    is_new = pair not in prev_links

                    # Active beams at source for microwave switch penalty.
                    src_beams = active_beam_usage.get(node, 0)

                    if link_type == 'gs_downlink':
                        # GS downlink: simple transmission cost only.
                        tx_s = (data_size_gb * 8.0 * 1024.0) / max(link_bw, 1.0)
                        edge_cost = tx_s + self.HOP_PENALTY_S
                        new_atp = atp_overhead
                    else:
                        edge_cost = self._compute_edge_cost(
                            lnk, data_size_gb, is_new, src_beams
                        )
                        atp_on_this_hop = (
                            getattr(lnk, 'atp_setup_time_s', 0.0)
                            if (link_type == 'laser' and is_new)
                            else 0.0
                        )
                        new_atp = atp_overhead + atp_on_this_hop

                    new_cost = base_cost + edge_cost
                    if new_cost >= best_cost:
                        continue

                    new_bw = min(bw, link_bw) if bw != INF else link_bw
                    new_rel = reliability * max(lq, 0.001)
                    new_path = path_nodes + [nbr]
                    new_link_types = path_link_types + [link_type]
                    new_prev_links = frozenset(current_links)

                    heapq.heappush(heap, (
                        new_cost,
                        nbr,
                        next_slice_idx,
                        new_path,
                        new_link_types,
                        new_atp,
                        new_rel,
                        new_bw,
                        new_prev_links,
                    ))

                # Only iterate waiting if source is NOT a GS node.
                if node.startswith('GS:'):
                    break

        return best_result

    # ------------------------------------------------------------------
    # Topology change counter
    # ------------------------------------------------------------------

    def _count_topology_changes(
        self,
        path_nodes: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Count how many time slices have a different active link set compared
        to the preceding slice, for the links in *path_nodes*.

        Args:
            path_nodes: Full path node list including source and ``GS:`` sink.
            start_time: Path start time.
            end_time: Estimated delivery time.

        Returns:
            Integer count of topology change events.
        """
        changes = 0
        total_s = (end_time - start_time).total_seconds()
        if total_s <= 0:
            return 0

        # Collect satellite-to-satellite edges in the path.
        edges: List[Tuple[str, str]] = []
        for i in range(len(path_nodes) - 1):
            a, b = path_nodes[i], path_nodes[i + 1]
            if not a.startswith('GS:') and not b.startswith('GS:'):
                edges.append((a, b) if a <= b else (b, a))

        if not edges:
            return 0

        n_slices = max(1, int(total_s / self.TIME_SLICE_S))
        prev_active: Optional[set] = None

        for k in range(n_slices):
            snap_time = start_time + timedelta(seconds=k * self.TIME_SLICE_S)
            active_now: set = set()
            for (a, b) in edges:
                for lnk in self._cache.get_windows(a, b):
                    if lnk.start_time <= snap_time <= lnk.end_time:
                        active_now.add((a, b))
                        break

            if prev_active is not None and active_now != prev_active:
                changes += 1
            prev_active = active_now

        return changes
