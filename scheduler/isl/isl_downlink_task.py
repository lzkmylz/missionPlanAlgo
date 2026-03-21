"""
ISL Downlink Task data model — Phase 5.

Represents a single ISL-routed data relay task in which data travels:

    source_satellite → [relay hops] → destination_gs

via ISL links and then conventional ground downlink.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ISLDownlinkTask:
    """An ISL-routed data relay task.

    Attributes:
        task_id: Unique task identifier (e.g. ``'ISL-DL-00001'``).
        source_satellite_id: ID of the satellite that holds the data.
        relay_hops: Ordered list of intermediate satellite IDs (may be empty
            for a single-hop direct downlink).
        link_types: Link type for each hop segment.  Length must equal
            ``len(relay_hops) + 1`` (one entry per edge, including the final
            ground-downlink edge).
        destination_gs_id: Final ground station ID (without ``GS:`` prefix).
        start_time: When the ISL relay begins.
        end_time: Estimated completion time (including ATP overhead and
            transmission time).
        data_size_gb: Amount of data relayed (GB).
        effective_bandwidth_mbps: Bottleneck bandwidth along the path (Mbps).
        atp_setup_time_s: Total ATP overhead accumulated along laser links.
        related_imaging_task_id: Task ID of the imaging task whose data is
            being relayed (``None`` if not directly associated with one task).
        route_path_nodes: Full ordered node list:
            ``[source] + relay_hops + ['GS:<destination_gs_id>']``.
        path_reliability: Product of per-hop link quality scores (0–1).
        topology_changes: Number of time slices where active links changed
            during the delivery window.
    """

    task_id: str
    source_satellite_id: str
    relay_hops: List[str]
    link_types: List[str]
    destination_gs_id: str
    start_time: datetime
    end_time: datetime
    data_size_gb: float
    effective_bandwidth_mbps: float
    atp_setup_time_s: float
    related_imaging_task_id: Optional[str] = None
    route_path_nodes: List[str] = field(default_factory=list)
    path_reliability: float = 1.0
    topology_changes: int = 0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> float:
        """Total relay duration from start to estimated delivery (seconds)."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def hop_count(self) -> int:
        """Number of relay hops (intermediate satellites only)."""
        return len(self.relay_hops)

    @property
    def link_type(self) -> str:
        """Primary ISL link type for this relay task ('laser' or 'microwave').

        Returns the first non-``gs_downlink`` entry in ``link_types``, or
        ``'laser'`` as a safe default when ``link_types`` is empty or contains
        only ``gs_downlink`` entries.  This scalar property is used by
        ``ISLDownlinkScheduler._check_relay_constraints`` to count active laser
        and microwave tasks separately against per-technology beam limits.
        """
        for lt in self.link_types:
            if lt not in ('gs_downlink',):
                return lt
        return 'laser'

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': 'isl_downlink',
            'source_satellite_id': self.source_satellite_id,
            'relay_hops': self.relay_hops,
            'link_types': self.link_types,
            'destination_gs_id': self.destination_gs_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'data_size_gb': self.data_size_gb,
            'effective_bandwidth_mbps': self.effective_bandwidth_mbps,
            'atp_setup_time_s': self.atp_setup_time_s,
            'related_imaging_task_id': self.related_imaging_task_id,
            'route_path_nodes': self.route_path_nodes,
            'path_reliability': self.path_reliability,
            'topology_changes': self.topology_changes,
            'hop_count': self.hop_count,
            'duration_seconds': self.duration_seconds,
        }
