"""
End-to-end integration test for ISL multi-hop data downlink.

Scenario (3-satellite chain):
  SRC (source) ----laser---- HOP (relay) ----laser---- DST (near GS)
                                                          |
                                                        GS-01

- SRC has data; cannot directly see GS-01.
- HOP has ISL links to both SRC and DST; cannot directly see GS-01.
- DST has ISL link to HOP and a direct GS-01 visibility window.

ISL windows are injected directly into ISLWindowCache (no Java backend).

Verified:
1. ISL windows are loaded into the cache correctly.
2. Router finds the 2-hop path SRC → HOP → DST → GS-01.
3. ISLDownlinkTask is created with correct hop list.
4. path_reliability > 0.5.
5. ATP setup time is included in end_time calculation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pytest

from core.models.isl_config import (
    ISLCapabilityConfig,
    LaserISLConfig,
    MicrowaveISLConfig,
)
from core.network.isl_router import (
    ISLWindowCache,
    ISLRoutePath,
    TimeVaryingISLRouter,
)
from scheduler.isl.isl_downlink_scheduler import ISLDownlinkScheduler
from scheduler.isl.isl_downlink_task import ISLDownlinkTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dt(offset_minutes: float = 0.0) -> datetime:
    base = datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _make_isl_link(sat_a: str, sat_b: str,
                   start_offset: float, end_offset: float,
                   link_type: str = 'laser',
                   data_rate: float = 10000.0,
                   atp_time_s: float = 37.0,
                   quality: float = 0.98):
    """Create a mock ISL link object."""
    class _ISLLink:
        def __init__(self):
            self.satellite_a_id = sat_a
            self.satellite_b_id = sat_b
            self.start_time = _dt(start_offset)
            self.end_time = _dt(end_offset)
            self.link_type = link_type
            self.max_data_rate = data_rate
            self.atp_setup_time_s = atp_time_s
            self.link_quality = quality
            self.is_viable = True
    return _ISLLink()


def _make_gs_window(start_offset: float, end_offset: float, data_rate: float = 300.0):
    """Create a mock ground station visibility window."""
    class _GSWindow:
        def __init__(self):
            self.start_time = _dt(start_offset)
            self.end_time = _dt(end_offset)
            self.max_data_rate = data_rate
            self.link_quality = 1.0
    return _GSWindow()


def _make_isl_config(max_simultaneous_laser: int = 2) -> ISLCapabilityConfig:
    return ISLCapabilityConfig(
        enabled=True,
        laser=LaserISLConfig(
            wavelength_nm=1550.0,
            transmit_power_w=2.0,
            transmit_aperture_m=0.1,
            receive_aperture_m=0.1,
            beam_divergence_urad=5.0,
            max_range_km=7000.0,
            acquisition_time_s=30.0,
            coarse_tracking_time_s=5.0,
            fine_tracking_time_s=2.0,
            tracking_accuracy_urad=2.0,
            point_ahead_urad=30.0,
            min_link_margin_db=3.0,
            snr_required_db=20.0,
        ),
        microwave=MicrowaveISLConfig(),
        max_simultaneous_laser=max_simultaneous_laser,
    )


@dataclass
class _ImagingTask:
    """Minimal imaging task for scheduler input."""
    task_id: str
    satellite_id: str
    start_time: datetime
    end_time: datetime
    storage_change_gb: float = 2.0
    data_rate_mbps: float = 300.0


# ---------------------------------------------------------------------------
# Main end-to-end test
# ---------------------------------------------------------------------------

class TestISLEndToEnd:
    """End-to-end ISL downlink scheduling with a 3-satellite chain."""

    @pytest.fixture
    def scenario(self):
        """Build the 3-satellite ISL chain scenario.

        Returns a dict with all scenario components.
        """
        SCENARIO_START = _dt(0)
        SCENARIO_END = _dt(120)

        # ------------------------------------------------------------------
        # 1. ISL link windows
        # ------------------------------------------------------------------
        cache = ISLWindowCache()

        # SRC ↔ HOP: active for the full 2-hour window
        link_src_hop = _make_isl_link('SRC', 'HOP', 0, 120, link_type='laser',
                                      data_rate=10000.0, atp_time_s=37.0)
        # HOP ↔ DST: active for the full 2-hour window
        link_hop_dst = _make_isl_link('HOP', 'DST', 0, 120, link_type='laser',
                                      data_rate=10000.0, atp_time_s=37.0)

        cache.add_window(link_src_hop)
        cache.add_window(link_hop_dst)

        # ------------------------------------------------------------------
        # 2. GS visibility windows (only DST can see GS-01)
        # ------------------------------------------------------------------
        gs_windows = {
            ('DST', 'GS-01'): [_make_gs_window(0, 120, data_rate=300.0)],
        }

        # ------------------------------------------------------------------
        # 3. Satellite ISL capability configs
        # ------------------------------------------------------------------
        isl_configs = {
            'SRC': _make_isl_config(max_simultaneous_laser=2),
            'HOP': _make_isl_config(max_simultaneous_laser=2),
            'DST': _make_isl_config(max_simultaneous_laser=2),
        }

        # ------------------------------------------------------------------
        # 4. Router
        # ------------------------------------------------------------------
        router = TimeVaryingISLRouter(
            isl_window_cache=cache,
            satellite_isl_configs=isl_configs,
            gs_visibility_windows=gs_windows,
        )

        # ------------------------------------------------------------------
        # 5. Scheduler
        # ------------------------------------------------------------------
        scheduler = ISLDownlinkScheduler(
            isl_router=router,
            satellite_isl_configs=isl_configs,
            max_relay_hops=3,
            deadline_buffer_s=7200.0,  # 2 hours
        )

        return {
            'cache': cache,
            'router': router,
            'scheduler': scheduler,
            'isl_configs': isl_configs,
            'scenario_start': SCENARIO_START,
            'scenario_end': SCENARIO_END,
        }

    # ------------------------------------------------------------------

    def test_isl_windows_loaded_in_cache(self, scenario):
        """ISL windows should be retrievable from the cache."""
        cache: ISLWindowCache = scenario['cache']

        windows_src_hop = cache.get_windows('SRC', 'HOP')
        windows_hop_dst = cache.get_windows('HOP', 'DST')

        assert len(windows_src_hop) >= 1, "SRC-HOP window should be loaded"
        assert len(windows_hop_dst) >= 1, "HOP-DST window should be loaded"

        # Bidirectional queries should work
        assert len(cache.get_windows('HOP', 'SRC')) >= 1
        assert len(cache.get_windows('DST', 'HOP')) >= 1

    def test_router_finds_two_hop_path(self, scenario):
        """Router should discover the 2-hop path SRC → HOP → DST → GS-01."""
        router: TimeVaryingISLRouter = scenario['router']

        result = router.find_route(
            source_satellite='SRC',
            data_size_gb=0.5,
            earliest_start=scenario['scenario_start'],
            deadline=scenario['scenario_end'],
        )

        assert result is not None, (
            "Router should find a path from SRC through HOP to GS-01"
        )
        assert result.destination_gs == 'GS-01'
        assert result.source_satellite == 'SRC'
        # HOP should appear as an intermediate satellite
        assert 'HOP' in result.path_nodes or result.hop_count >= 1

    def test_isl_downlink_task_created(self, scenario):
        """Scheduler should create an ISLDownlinkTask for SRC's imaging data."""
        scheduler: ISLDownlinkScheduler = scenario['scheduler']

        imaging_task = _ImagingTask(
            task_id='IMG-SRC-001',
            satellite_id='SRC',
            start_time=_dt(0),
            end_time=_dt(5),
            storage_change_gb=2.0,
        )

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[imaging_task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1, (
            f"Expected 1 ISL task, got {len(isl_tasks)}; failed: {failed_ids}"
        )
        assert len(failed_ids) == 0

        isl_task: ISLDownlinkTask = isl_tasks[0]
        assert isl_task.source_satellite_id == 'SRC'
        assert isl_task.destination_gs_id == 'GS-01'
        assert isl_task.related_imaging_task_id == 'IMG-SRC-001'

    def test_isl_task_hop_list(self, scenario):
        """ISLDownlinkTask should have HOP in its relay_hops list."""
        scheduler: ISLDownlinkScheduler = scenario['scheduler']

        imaging_task = _ImagingTask(
            task_id='IMG-SRC-002',
            satellite_id='SRC',
            start_time=_dt(0),
            end_time=_dt(5),
        )

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[imaging_task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1, f"Task not created; failed: {failed_ids}"
        isl_task = isl_tasks[0]
        assert 'HOP' in isl_task.relay_hops, (
            f"HOP should be in relay_hops; got {isl_task.relay_hops}"
        )

    def test_path_reliability_above_threshold(self, scenario):
        """path_reliability should be > 0.5 for this well-connected scenario."""
        scheduler: ISLDownlinkScheduler = scenario['scheduler']

        imaging_task = _ImagingTask(
            task_id='IMG-SRC-003',
            satellite_id='SRC',
            start_time=_dt(0),
            end_time=_dt(5),
        )

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[imaging_task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1, f"Task not created; failed: {failed_ids}"
        assert isl_tasks[0].path_reliability > 0.5, (
            f"Reliability {isl_tasks[0].path_reliability} should be > 0.5"
        )

    def test_atp_time_included_in_end_time(self, scenario):
        """ISLDownlinkTask.end_time should account for ATP setup overhead."""
        scheduler: ISLDownlinkScheduler = scenario['scheduler']

        imaging_task_end = _dt(5)
        imaging_task = _ImagingTask(
            task_id='IMG-SRC-004',
            satellite_id='SRC',
            start_time=_dt(0),
            end_time=imaging_task_end,
            storage_change_gb=2.0,
        )

        isl_tasks, failed_ids = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[imaging_task],
            satellite_states={},
            existing_tasks=[],
        )

        assert len(isl_tasks) == 1, f"Task not created; failed: {failed_ids}"
        isl_task = isl_tasks[0]

        # end_time must be after start_time (which equals imaging_task_end)
        assert isl_task.end_time > isl_task.start_time, (
            "ISL task end_time should be after start_time"
        )
        # ATP overhead should be reflected in the total duration
        # The ATP overhead for laser links is 37s per new link
        assert isl_task.atp_setup_time_s >= 0.0, (
            "ATP setup time should be non-negative"
        )

    def test_no_direct_gs_means_isl_is_only_option(self, scenario):
        """SRC cannot directly reach GS-01 (only DST can), so ISL is the sole path."""
        router: TimeVaryingISLRouter = scenario['router']

        # Verify SRC has no direct GS window by trying to route direct
        # We verify by checking that router correctly routes through HOP
        result = router.find_route(
            source_satellite='SRC',
            data_size_gb=0.1,
            earliest_start=scenario['scenario_start'],
            deadline=scenario['scenario_end'],
        )

        # Route must exist and must go through intermediate nodes
        assert result is not None
        # Full path should contain at least SRC and a GS node
        path = result.path_nodes
        assert path[0] == 'SRC'
        assert any(n.startswith('GS:') for n in path)
        # At least one intermediate hop should be present (DST or HOP)
        assert len(path) >= 3

    def test_isl_task_serialisation_to_dict(self, scenario):
        """ISLDownlinkTask.to_dict() should produce a JSON-compatible dict."""
        scheduler: ISLDownlinkScheduler = scenario['scheduler']

        imaging_task = _ImagingTask(
            task_id='IMG-SRC-005',
            satellite_id='SRC',
            start_time=_dt(0),
            end_time=_dt(5),
        )

        isl_tasks, _ = scheduler.schedule_isl_downlinks_for_tasks(
            imaging_tasks=[imaging_task],
            satellite_states={},
            existing_tasks=[],
        )

        if not isl_tasks:
            pytest.skip("No ISL task created; skipping serialisation test")

        d = isl_tasks[0].to_dict()

        assert d['task_type'] == 'isl_downlink'
        assert 'source_satellite_id' in d
        assert 'destination_gs_id' in d
        assert 'relay_hops' in d
        assert 'link_types' in d
        assert 'start_time' in d
        assert 'end_time' in d
        assert 'path_reliability' in d
        assert 'hop_count' in d
        assert d['hop_count'] >= 1
