"""
ISL Downlink Scheduler — Phase 5.

This is the **third** downlink-strategy tier used by the unified scheduler:

    1. Direct GS downlink (primary)   — GroundStationScheduler
    2. GEO relay (secondary)          — RelayScheduler
    3. ISL multi-hop relay (tertiary) — ISLDownlinkScheduler  ← this module

After direct GS and GEO relay both fail for a set of imaging tasks, this
scheduler attempts to route the data through the ISL network to reach a
ground station via one or more satellite hops.

Usage example
-------------
    router = TimeVaryingISLRouter(...)
    scheduler = ISLDownlinkScheduler(
        isl_router=router,
        satellite_isl_configs=configs,
        max_relay_hops=3,
        deadline_buffer_s=3600.0,
    )
    isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
        imaging_tasks=failed_after_gs_and_relay,
        satellite_states=states,
        existing_tasks=all_scheduled,
    )
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .isl_downlink_task import ISLDownlinkTask

logger = logging.getLogger(__name__)


class ISLDownlinkScheduler:
    """Schedules data downlink through the ISL multi-hop relay network.

    This scheduler is invoked after direct GS downlink and GEO relay have
    both failed to find a suitable window for an imaging task.

    Parameters
    ----------
    isl_router:
        A ``TimeVaryingISLRouter`` instance used for path finding.
    satellite_isl_configs:
        Dictionary mapping satellite ID → ``ISLCapabilityConfig``.
    max_relay_hops:
        Maximum number of intermediate relay hops allowed in a path.
        Tasks whose best route exceeds this limit are rejected.
    deadline_buffer_s:
        Maximum time (seconds) after imaging completion within which data
        delivery must be achieved.
    """

    def __init__(
        self,
        isl_router: Any,  # TimeVaryingISLRouter
        satellite_isl_configs: Dict[str, Any],  # sat_id -> ISLCapabilityConfig
        max_relay_hops: int = 3,
        deadline_buffer_s: float = 3600.0,
    ) -> None:
        self.router = isl_router
        self.isl_configs = satellite_isl_configs
        self.max_relay_hops = max_relay_hops
        self.deadline_buffer_s = deadline_buffer_s
        self._task_counter: int = 0

    # ------------------------------------------------------------------
    # Main scheduling entry point
    # ------------------------------------------------------------------

    def schedule_isl_downlinks_for_tasks(
        self,
        imaging_tasks: List[Any],         # ScheduledTask objects
        satellite_states: Dict[str, Any],
        existing_tasks: List[Any],
    ) -> Tuple[List[ISLDownlinkTask], List[str]]:
        """Attempt ISL routing for each imaging task.

        For each imaging task the scheduler:
        1. Verifies the source satellite has an ISL capability config.
        2. Estimates the data size from the task.
        3. Calls ``TimeVaryingISLRouter.find_route()`` to find the best path.
        4. Checks relay-hop beam constraints against existing ISL tasks.
        5. Creates an ``ISLDownlinkTask`` if all checks pass.

        Parameters
        ----------
        imaging_tasks:
            List of ``ScheduledTask`` objects whose data could not be
            downlinked via direct GS or GEO relay.
        satellite_states:
            Current resource states keyed by satellite ID (unused internally
            but provided for interface compatibility with other schedulers).
        existing_tasks:
            All tasks already in the schedule (used for beam-count conflict
            checking).

        Returns
        -------
        (isl_tasks_created, failed_task_ids):
            ``isl_tasks_created`` — successfully scheduled ``ISLDownlinkTask``
            objects.
            ``failed_task_ids`` — task IDs for which no ISL route could be
            arranged.
        """
        isl_tasks: List[ISLDownlinkTask] = []
        failed_task_ids: List[str] = []

        for task in imaging_tasks:
            sat_id: str = task.satellite_id
            isl_config = self.isl_configs.get(sat_id)

            if isl_config is None or not isl_config.enabled:
                logger.debug(
                    "Task %s: satellite %s has no ISL capability configured",
                    task.task_id, sat_id,
                )
                failed_task_ids.append(task.task_id)
                continue

            # Estimate data size.
            data_size_gb = self._estimate_data_size(task)

            # Compute earliest relay start and deadline.
            earliest_start: datetime = task.end_time
            deadline: datetime = earliest_start + timedelta(seconds=self.deadline_buffer_s)

            # Find an ISL route.
            route = self.router.find_route(
                source_satellite=sat_id,
                data_size_gb=data_size_gb,
                earliest_start=earliest_start,
                deadline=deadline,
            )

            if route is None:
                logger.debug(
                    "Task %s: no ISL route found for satellite %s within %.0f s window",
                    task.task_id, sat_id, self.deadline_buffer_s,
                )
                failed_task_ids.append(task.task_id)
                continue

            if route.hop_count > self.max_relay_hops:
                logger.debug(
                    "Task %s: route has %d hops which exceeds max_relay_hops=%d",
                    task.task_id, route.hop_count, self.max_relay_hops,
                )
                failed_task_ids.append(task.task_id)
                continue

            # Check relay-hop beam constraints.
            if not self._check_relay_constraints(route, earliest_start, existing_tasks):
                logger.debug(
                    "Task %s: relay beam constraints not satisfied for route %s",
                    task.task_id, route.path_nodes,
                )
                failed_task_ids.append(task.task_id)
                continue

            # Calculate end time: transmission + ATP overhead.
            tx_s = (data_size_gb * 8.0 * 1024.0) / max(route.bottleneck_bandwidth_mbps, 1.0)
            total_s = tx_s + route.atp_setup_overhead_s
            task_end = earliest_start + timedelta(seconds=total_s)

            # Create the ISL downlink task.
            self._task_counter += 1
            isl_task = ISLDownlinkTask(
                task_id=f"ISL-DL-{self._task_counter:05d}",
                source_satellite_id=sat_id,
                relay_hops=list(route.hops),
                link_types=list(route.link_types),
                destination_gs_id=route.destination_gs,
                start_time=earliest_start,
                end_time=task_end,
                data_size_gb=data_size_gb,
                effective_bandwidth_mbps=route.bottleneck_bandwidth_mbps,
                atp_setup_time_s=route.atp_setup_overhead_s,
                related_imaging_task_id=task.task_id,
                route_path_nodes=list(route.path_nodes),
                path_reliability=route.path_reliability,
                topology_changes=route.topology_changes,
            )

            isl_tasks.append(isl_task)

            logger.info(
                "ISL route: %s -> %s -> %s  (%d hop(s), %.0f Mbps, ATP %.1f s)",
                sat_id,
                " -> ".join(route.hops) if route.hops else "(direct)",
                route.destination_gs,
                route.hop_count,
                route.bottleneck_bandwidth_mbps,
                route.atp_setup_overhead_s,
            )

        return isl_tasks, failed_task_ids

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _estimate_data_size(self, task: Any) -> float:
        """Estimate data size in GB from an imaging task.

        Uses ``storage_change_gb`` if available; otherwise derives size from
        task duration × assumed data rate.

        Parameters
        ----------
        task:
            A ``ScheduledTask``-like object.

        Returns
        -------
        float: Estimated data size in GB (always positive).
        """
        storage_change = getattr(task, 'storage_change_gb', None)
        if storage_change is not None and storage_change > 0.0:
            return float(storage_change)

        # Fall back: use stored data after/before delta if available.
        storage_after = getattr(task, 'storage_after', None)
        storage_before = getattr(task, 'storage_before', None)
        if storage_after is not None and storage_before is not None:
            delta = storage_after - storage_before
            if delta > 0.0:
                return float(delta)

        # Last resort: estimate from duration × data rate.
        start = getattr(task, 'imaging_start', None) or getattr(task, 'start_time', None)
        end = getattr(task, 'imaging_end', None) or getattr(task, 'end_time', None)
        if start is not None and end is not None:
            duration_s = (end - start).total_seconds()
            data_rate_mbps = getattr(task, 'data_rate_mbps', 300.0)
            gb = (duration_s * data_rate_mbps) / (8.0 * 1024.0)
            return max(gb, 0.001)

        # Absolute fallback: 1 GB.
        return 1.0

    def _check_relay_constraints(
        self,
        route: Any,  # ISLRoutePath
        start_time: datetime,
        existing_tasks: List[Any],
    ) -> bool:
        """Check that every relay hop satellite has sufficient beam capacity.

        For each intermediate satellite in *route.hops*, the method counts how
        many existing ISL tasks overlap in time with *start_time* and verifies
        that the satellite's beam count limit (``max_beam_count`` for
        microwave, ``max_simultaneous_laser`` for laser) is not already
        exhausted.

        Parameters
        ----------
        route:
            The candidate ``ISLRoutePath``.
        start_time:
            Proposed ISL relay start time.
        existing_tasks:
            All tasks already in the schedule.

        Returns
        -------
        bool: ``True`` if all relay hops can accommodate an additional link.
        """
        for hop_sat_id in route.hops:
            isl_config = self.isl_configs.get(hop_sat_id)
            if isl_config is None or not isl_config.enabled:
                logger.debug(
                    "Relay hop %s has no ISL capability config", hop_sat_id
                )
                return False

            # Count existing ISL tasks that use this satellite as a relay hop
            # and overlap with start_time — counted separately by link type.
            overlapping_tasks = [
                t for t in existing_tasks
                if (
                    hasattr(t, 'relay_hops')
                    and hop_sat_id in getattr(t, 'relay_hops', [])
                    and hasattr(t, 'start_time')
                    and hasattr(t, 'end_time')
                    and t.start_time <= start_time <= t.end_time
                )
            ]
            active_laser_count = sum(
                1 for t in overlapping_tasks
                if getattr(t, 'link_type', '') == 'laser'
            )
            active_mw_count = sum(
                1 for t in overlapping_tasks
                if getattr(t, 'link_type', '') == 'microwave'
            )

            route_link_types = getattr(route, 'link_types', [])

            # Check microwave beam count only if this route uses microwave.
            if any(lt == 'microwave' for lt in route_link_types):
                mw_cfg = getattr(isl_config, 'microwave', None)
                if mw_cfg is not None:
                    max_beams = getattr(mw_cfg, 'max_beam_count', 4)
                    if active_mw_count >= max_beams:
                        logger.debug(
                            "Relay hop %s: microwave beam count %d/%d already at limit",
                            hop_sat_id, active_mw_count, max_beams,
                        )
                        return False

            # Check laser simultaneous count only if this route uses laser.
            if any(lt == 'laser' for lt in route_link_types):
                max_laser = getattr(isl_config, 'max_simultaneous_laser', 2)
                if active_laser_count >= max_laser:
                    logger.debug(
                        "Relay hop %s: laser link count %d/%d already at limit",
                        hop_sat_id, active_laser_count, max_laser,
                    )
                    return False

        return True

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self, isl_tasks: List[ISLDownlinkTask]) -> dict:
        """Return a statistics summary for a list of ISL downlink tasks.

        Parameters
        ----------
        isl_tasks:
            The ``ISLDownlinkTask`` objects created by a scheduling run.

        Returns
        -------
        dict: Statistics dictionary with keys:
            - ``total_isl_tasks``
            - ``avg_hop_count``
            - ``max_hop_count``
            - ``avg_bandwidth_mbps``
            - ``min_bandwidth_mbps``
            - ``avg_reliability``
            - ``total_data_relayed_gb``
            - ``laser_only_paths``
            - ``microwave_only_paths``
            - ``mixed_paths``
        """
        if not isl_tasks:
            return {'total_isl_tasks': 0}

        hop_counts = [t.hop_count for t in isl_tasks]
        bandwidths = [t.effective_bandwidth_mbps for t in isl_tasks]
        reliabilities = [t.path_reliability for t in isl_tasks]

        def _is_laser_only(task: ISLDownlinkTask) -> bool:
            lt = [l for l in task.link_types if l != 'gs_downlink']
            return bool(lt) and all(l == 'laser' for l in lt)

        def _is_microwave_only(task: ISLDownlinkTask) -> bool:
            lt = [l for l in task.link_types if l != 'gs_downlink']
            return bool(lt) and all(l == 'microwave' for l in lt)

        return {
            'total_isl_tasks': len(isl_tasks),
            'avg_hop_count': sum(hop_counts) / len(hop_counts),
            'max_hop_count': max(hop_counts),
            'avg_bandwidth_mbps': sum(bandwidths) / len(bandwidths),
            'min_bandwidth_mbps': min(bandwidths),
            'avg_reliability': sum(reliabilities) / len(reliabilities),
            'total_data_relayed_gb': sum(t.data_size_gb for t in isl_tasks),
            'laser_only_paths': sum(1 for t in isl_tasks if _is_laser_only(t)),
            'microwave_only_paths': sum(1 for t in isl_tasks if _is_microwave_only(t)),
            'mixed_paths': sum(
                1 for t in isl_tasks
                if not _is_laser_only(t) and not _is_microwave_only(t)
            ),
        }
