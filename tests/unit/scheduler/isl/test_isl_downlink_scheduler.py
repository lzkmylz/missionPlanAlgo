"""
Unit tests for ISLDownlinkScheduler.

Tests cover:
- Satellite without ISL capability results in failed task
- Successful route creation produces ISLDownlinkTask
- Route with too many hops is rejected
- Beam count constraint check blocks tasks
- Statistics calculation
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import pytest

from scheduler.isl.isl_downlink_scheduler import ISLDownlinkScheduler
from scheduler.isl.isl_downlink_task import ISLDownlinkTask
from core.models.isl_config import (
    ISLCapabilityConfig,
    LaserISLConfig,
    MicrowaveISLConfig,
)
from core.network.isl_router import ISLRoutePath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(offset_minutes: float = 0.0) -> datetime:
    base = datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


@dataclass
class _MockTask:
    """Minimal ScheduledTask-like object."""
    task_id: str
    satellite_id: str
    start_time: datetime
    end_time: datetime
    storage_change_gb: float = 1.0
    data_rate_mbps: float = 300.0


class _MockRouter:
    """Configurable mock for TimeVaryingISLRouter."""

    def __init__(self, route: Optional[ISLRoutePath] = None) -> None:
        self._route = route

    def find_route(self, source_satellite, data_size_gb, earliest_start, deadline,
                   target_gs=None):
        return self._route


def _make_route(
    source: str = 'SAT-SRC',
    destination_gs: str = 'GS-01',
    hops: Optional[List[str]] = None,
    link_types: Optional[List[str]] = None,
    bandwidth_mbps: float = 5000.0,
    reliability: float = 0.99,
    atp_overhead_s: float = 37.0,
) -> ISLRoutePath:
    hops = hops or []
    link_types = link_types or (['laser'] * (len(hops) + 1))
    path_nodes = [source] + hops + [f'GS:{destination_gs}']
    return ISLRoutePath(
        source_satellite=source,
        destination_gs=destination_gs,
        hops=hops,
        link_types=link_types,
        bottleneck_bandwidth_mbps=bandwidth_mbps,
        total_latency_s=atp_overhead_s + 10.0,
        atp_setup_overhead_s=atp_overhead_s,
        path_reliability=reliability,
        topology_changes=0,
        path_nodes=path_nodes,
    )


def _make_isl_config(enabled: bool = True) -> ISLCapabilityConfig:
    return ISLCapabilityConfig(
        enabled=enabled,
        laser=LaserISLConfig(),
        microwave=MicrowaveISLConfig(),
        max_simultaneous_laser=2,
    )


# ---------------------------------------------------------------------------
# Test: no ISL capability → task fails
# ---------------------------------------------------------------------------

class TestNoISLCapability:

    def test_no_isl_capability_skips_task(self):
        """A satellite without ISL config should result in a failed task."""
        router = _MockRouter(route=None)
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={},  # SAT-SRC has no config
            max_relay_hops=3,
            deadline_buffer_s=3600.0,
        )

        task = _MockTask(
            task_id='IMG-001',
            satellite_id='SAT-SRC',
            start_time=_dt(0),
            end_time=_dt(10),
        )

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 0
        assert 'IMG-001' in failed_ids

    def test_isl_disabled_config_skips_task(self):
        """A satellite with ISL config but enabled=False should still fail."""
        router = _MockRouter(route=_make_route())
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={
                'SAT-SRC': _make_isl_config(enabled=False),
            },
            max_relay_hops=3,
            deadline_buffer_s=3600.0,
        )

        task = _MockTask('IMG-002', 'SAT-SRC', _dt(0), _dt(10))

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 0
        assert 'IMG-002' in failed_ids


# ---------------------------------------------------------------------------
# Test: successful route
# ---------------------------------------------------------------------------

class TestSuccessfulRoute:

    def test_successful_isl_route_creates_task(self):
        """A valid route returned by the router should produce an ISLDownlinkTask."""
        route = _make_route(
            source='SAT-SRC',
            destination_gs='GS-01',
            hops=['SAT-HOP'],
            link_types=['laser', 'laser', 'gs_downlink'],
        )
        router = _MockRouter(route=route)
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={
                'SAT-SRC': _make_isl_config(),
                'SAT-HOP': _make_isl_config(),
            },
            max_relay_hops=3,
            deadline_buffer_s=3600.0,
        )

        task = _MockTask('IMG-003', 'SAT-SRC', _dt(0), _dt(10))

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1
        assert len(failed_ids) == 0

        isl_task = isl_tasks[0]
        assert isinstance(isl_task, ISLDownlinkTask)
        assert isl_task.source_satellite_id == 'SAT-SRC'
        assert isl_task.destination_gs_id == 'GS-01'
        assert isl_task.related_imaging_task_id == 'IMG-003'
        assert isl_task.relay_hops == ['SAT-HOP']

    def test_isl_task_end_time_accounts_for_atp(self):
        """ISLDownlinkTask end_time should include ATP overhead."""
        atp_overhead = 37.0
        route = _make_route(
            source='SAT-SRC',
            destination_gs='GS-01',
            bandwidth_mbps=10000.0,
            atp_overhead_s=atp_overhead,
        )
        router = _MockRouter(route=route)
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={'SAT-SRC': _make_isl_config()},
            max_relay_hops=3,
            deadline_buffer_s=3600.0,
        )

        task_end = _dt(10)
        task = _MockTask('IMG-004', 'SAT-SRC', _dt(0), task_end, storage_change_gb=1.0)

        isl_tasks, _ = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1
        isl_task = isl_tasks[0]
        # start_time should equal imaging task end_time
        assert isl_task.start_time == task_end
        # end_time should be after start by at least the ATP overhead
        duration = isl_task.duration_seconds
        assert duration >= atp_overhead, (
            f"Task duration {duration:.1f}s should be >= ATP overhead {atp_overhead}s"
        )


# ---------------------------------------------------------------------------
# Test: too many hops → rejected
# ---------------------------------------------------------------------------

class TestTooManyHops:

    def test_too_many_hops_rejected(self):
        """Route with more hops than max_relay_hops should be rejected."""
        route = _make_route(
            source='SAT-SRC',
            destination_gs='GS-01',
            hops=['HOP1', 'HOP2', 'HOP3', 'HOP4'],  # 4 hops
            link_types=['laser'] * 5,
        )
        router = _MockRouter(route=route)
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={
                'SAT-SRC': _make_isl_config(),
                'HOP1': _make_isl_config(),
                'HOP2': _make_isl_config(),
                'HOP3': _make_isl_config(),
                'HOP4': _make_isl_config(),
            },
            max_relay_hops=3,  # max is 3, route has 4
            deadline_buffer_s=3600.0,
        )

        task = _MockTask('IMG-005', 'SAT-SRC', _dt(0), _dt(10))

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 0
        assert 'IMG-005' in failed_ids

    def test_hops_at_limit_accepted(self):
        """Route with exactly max_relay_hops should be accepted."""
        route = _make_route(
            source='SAT-SRC',
            destination_gs='GS-01',
            hops=['HOP1', 'HOP2', 'HOP3'],  # exactly 3 hops
            link_types=['laser'] * 4,
        )
        router = _MockRouter(route=route)
        isl_configs = {
            'SAT-SRC': _make_isl_config(),
            'HOP1': _make_isl_config(),
            'HOP2': _make_isl_config(),
            'HOP3': _make_isl_config(),
        }
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs=isl_configs,
            max_relay_hops=3,
            deadline_buffer_s=3600.0,
        )

        task = _MockTask('IMG-006', 'SAT-SRC', _dt(0), _dt(10))

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1
        assert len(failed_ids) == 0


# ---------------------------------------------------------------------------
# Test: beam constraint check
# ---------------------------------------------------------------------------

class TestBeamConstraintCheck:

    def test_beam_constraint_check(self):
        """When a relay hop satellite has exhausted its beam capacity,
        the task should be rejected."""
        route = _make_route(
            source='SAT-SRC',
            destination_gs='GS-01',
            hops=['SAT-HOP'],
            link_types=['microwave', 'microwave', 'gs_downlink'],
        )
        router = _MockRouter(route=route)

        # SAT-HOP config: max_beam_count=2
        hop_cfg = ISLCapabilityConfig(
            enabled=True,
            microwave=MicrowaveISLConfig(max_beam_count=2),
            max_simultaneous_laser=2,
        )
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={
                'SAT-SRC': _make_isl_config(),
                'SAT-HOP': hop_cfg,
            },
            max_relay_hops=3,
            deadline_buffer_s=3600.0,
        )

        # Simulate 2 existing ISL tasks that already use SAT-HOP as a relay
        # and overlap with start_time = _dt(10)
        start_time = _dt(10)

        @dataclass
        class _ExistingISLTask:
            start_time: datetime
            end_time: datetime
            relay_hops: List[str]
            link_type: str = 'microwave'

        existing_tasks = [
            _ExistingISLTask(_dt(5), _dt(60), ['SAT-HOP'], link_type='microwave'),
            _ExistingISLTask(_dt(8), _dt(50), ['SAT-HOP'], link_type='microwave'),
        ]

        task = _MockTask('IMG-007', 'SAT-SRC', _dt(0), start_time)

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[task],
            satellite_states={},
            existing_tasks=existing_tasks,
        )

        # SAT-HOP already has 2 active relay tasks at start_time; max_beam_count=2
        # so new task should be rejected
        assert len(isl_tasks) == 0
        assert 'IMG-007' in failed_ids


# ---------------------------------------------------------------------------
# Test: statistics calculation
# ---------------------------------------------------------------------------

class TestStatisticsCalculation:

    def _make_isl_task(
        self, task_id: str, hops: List[str], link_types: List[str],
        bw: float = 5000.0, reliability: float = 0.99,
        data_gb: float = 1.0,
    ) -> ISLDownlinkTask:
        path_nodes = ['SRC'] + hops + ['GS:GS-01']
        return ISLDownlinkTask(
            task_id=task_id,
            source_satellite_id='SRC',
            relay_hops=hops,
            link_types=link_types,
            destination_gs_id='GS-01',
            start_time=_dt(0),
            end_time=_dt(5),
            data_size_gb=data_gb,
            effective_bandwidth_mbps=bw,
            atp_setup_time_s=37.0,
            path_reliability=reliability,
            route_path_nodes=path_nodes,
        )

    def test_statistics_calculation_counts(self):
        """get_statistics should correctly count tasks, hops, and bandwidth."""
        route = _make_route()
        router = _MockRouter(route=route)
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={},
            max_relay_hops=3,
        )

        tasks = [
            self._make_isl_task('T1', [], ['laser', 'gs_downlink'], bw=5000.0),
            self._make_isl_task('T2', ['H1'], ['laser', 'laser', 'gs_downlink'], bw=3000.0),
            self._make_isl_task('T3', ['H1', 'H2'], ['microwave', 'microwave', 'gs_downlink'],
                                bw=1000.0),
        ]

        stats = scheduler.get_statistics(tasks)

        assert stats['total_isl_tasks'] == 3
        assert stats['max_hop_count'] == 2
        assert stats['avg_hop_count'] == pytest.approx((0 + 1 + 2) / 3.0, rel=1e-6)
        assert stats['min_bandwidth_mbps'] == pytest.approx(1000.0)
        assert stats['total_data_relayed_gb'] == pytest.approx(3.0)

    def test_statistics_empty_returns_zero(self):
        """get_statistics with empty list should return total_isl_tasks=0."""
        router = _MockRouter()
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={},
        )
        stats = scheduler.get_statistics([])
        assert stats == {'total_isl_tasks': 0}

    def test_statistics_link_type_categorisation(self):
        """Laser-only vs microwave-only vs mixed paths should be counted correctly."""
        router = _MockRouter()
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs={},
        )

        tasks = [
            self._make_isl_task('T1', ['H1'], ['laser', 'laser', 'gs_downlink']),
            self._make_isl_task('T2', ['H1'], ['microwave', 'microwave', 'gs_downlink']),
            self._make_isl_task('T3', ['H1'], ['laser', 'microwave', 'gs_downlink']),
        ]
        stats = scheduler.get_statistics(tasks)

        assert stats['laser_only_paths'] == 1
        assert stats['microwave_only_paths'] == 1
        assert stats['mixed_paths'] == 1
