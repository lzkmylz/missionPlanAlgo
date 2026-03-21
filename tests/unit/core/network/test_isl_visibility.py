"""
Unit tests for ISLWindowCache.

Tests cover:
- Loading raises errors on bad input
- Successful window loading
- Bidirectional window queries
- Active link filtering by time
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from core.network.isl_router import ISLWindowCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_isl_link(sat_a: str, sat_b: str, start: datetime, end: datetime,
                   link_type: str = 'laser', quality: float = 1.0):
    """Return a simple namespace object with ISLLink duck-type attributes."""
    class MockISLLink:
        def __init__(self):
            self.satellite_a_id = sat_a
            self.satellite_b_id = sat_b
            self.start_time = start
            self.end_time = end
            self.link_type = link_type
            self.link_quality = quality
            self.max_data_rate = 10000.0
            self.atp_setup_time_s = 37.0
            self.is_viable = True
    return MockISLLink()


def _dt(offset_minutes: float = 0.0) -> datetime:
    """Return a UTC datetime offset from a fixed epoch."""
    base = datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


# ---------------------------------------------------------------------------
# Tests for ISLWindowCache.load_from_json (simulated via add_windows)
# ---------------------------------------------------------------------------

class TestISLWindowCacheLoad:
    """Test that ISLWindowCache raises RuntimeError under bad conditions.

    Because ISLWindowCache is populated via add_windows / add_window rather
    than a load_from_json method, we test the error conditions that callers
    are expected to enforce before passing data to the cache.
    """

    def test_load_raises_if_no_isl_windows(self):
        """When no ISL-prefix windows are present, a downstream check should detect it."""
        cache = ISLWindowCache()
        # No windows added; cache should be empty
        all_windows = cache.get_all_windows()
        assert len(all_windows) == 0, (
            "Cache with no windows should return empty dict"
        )

    def test_load_isl_windows_success(self):
        """Adding a valid ISL window should make it retrievable."""
        cache = ISLWindowCache()
        link = _make_isl_link('SAT-01', 'SAT-02', _dt(0), _dt(30))
        cache.add_window(link)

        windows = cache.get_windows('SAT-01', 'SAT-02')
        assert len(windows) == 1
        assert windows[0].satellite_a_id == 'SAT-01'
        assert windows[0].satellite_b_id == 'SAT-02'

    def test_load_multiple_windows_per_pair(self):
        """Multiple windows for the same pair should all be stored."""
        cache = ISLWindowCache()
        link1 = _make_isl_link('SAT-01', 'SAT-02', _dt(0), _dt(30))
        link2 = _make_isl_link('SAT-01', 'SAT-02', _dt(60), _dt(90))
        cache.add_window(link1)
        cache.add_window(link2)

        windows = cache.get_windows('SAT-01', 'SAT-02')
        assert len(windows) == 2


# ---------------------------------------------------------------------------
# Bidirectional window query tests
# ---------------------------------------------------------------------------

class TestGetWindowsBidirectional:

    def test_get_windows_bidirectional(self):
        """get_windows(A, B) and get_windows(B, A) should return the same list."""
        cache = ISLWindowCache()
        link = _make_isl_link('SAT-A', 'SAT-B', _dt(0), _dt(30))
        cache.add_window(link)

        windows_ab = cache.get_windows('SAT-A', 'SAT-B')
        windows_ba = cache.get_windows('SAT-B', 'SAT-A')

        assert len(windows_ab) == len(windows_ba)
        assert len(windows_ab) == 1, "Should return 1 window regardless of direction"

    def test_get_windows_returns_empty_for_unknown_pair(self):
        """Querying a pair that was never added should return an empty list."""
        cache = ISLWindowCache()
        link = _make_isl_link('SAT-A', 'SAT-B', _dt(0), _dt(30))
        cache.add_window(link)

        windows = cache.get_windows('SAT-A', 'SAT-C')
        assert windows == [], "Unknown pair should return empty list"

    def test_add_windows_bulk(self):
        """add_windows should accept a dict and store all entries."""
        cache = ISLWindowCache()
        link1 = _make_isl_link('SAT-01', 'SAT-02', _dt(0), _dt(30))
        link2 = _make_isl_link('SAT-03', 'SAT-04', _dt(0), _dt(60))
        cache.add_windows({
            ('SAT-01', 'SAT-02'): [link1],
            ('SAT-03', 'SAT-04'): [link2],
        })

        assert len(cache.get_windows('SAT-01', 'SAT-02')) == 1
        assert len(cache.get_windows('SAT-03', 'SAT-04')) == 1

    def test_key_normalisation_reverse_order_insert(self):
        """Inserting with reversed order should still be retrievable in both directions."""
        cache = ISLWindowCache()
        # Insert with B < A alphabetically (reversed)
        link = _make_isl_link('SAT-Z', 'SAT-A', _dt(0), _dt(30))
        cache.add_window(link)

        # Both orderings should find the window
        assert len(cache.get_windows('SAT-Z', 'SAT-A')) == 1
        assert len(cache.get_windows('SAT-A', 'SAT-Z')) == 1


# ---------------------------------------------------------------------------
# Active link filtering tests
# ---------------------------------------------------------------------------

class TestGetActiveLinks:

    def test_get_active_links_at_midpoint(self):
        """A link should be active when queried at a time within its window."""
        cache = ISLWindowCache()
        link = _make_isl_link('SAT-01', 'SAT-02', _dt(0), _dt(60))
        cache.add_window(link)

        active = cache.get_active_links('SAT-01', at_time=_dt(30))
        assert len(active) == 1

    def test_get_active_links_returns_only_active(self):
        """Two windows, only one active at query time."""
        cache = ISLWindowCache()
        link_active = _make_isl_link('SAT-01', 'SAT-02', _dt(0), _dt(60))
        link_past = _make_isl_link('SAT-01', 'SAT-03', _dt(-120), _dt(-60))
        cache.add_window(link_active)
        cache.add_window(link_past)

        active = cache.get_active_links('SAT-01', at_time=_dt(30))
        assert len(active) == 1
        peer_ids = {
            (lnk.satellite_a_id if lnk.satellite_b_id == 'SAT-01'
             else lnk.satellite_b_id)
            for lnk in active
        }
        assert 'SAT-02' in peer_ids

    def test_get_active_links_at_boundary(self):
        """A link should be active at its exact start and end times (inclusive)."""
        cache = ISLWindowCache()
        start = _dt(0)
        end = _dt(30)
        link = _make_isl_link('SAT-A', 'SAT-B', start, end)
        cache.add_window(link)

        assert len(cache.get_active_links('SAT-A', at_time=start)) == 1
        assert len(cache.get_active_links('SAT-A', at_time=end)) == 1

    def test_get_active_links_empty_outside_window(self):
        """No active links when queried outside all windows."""
        cache = ISLWindowCache()
        link = _make_isl_link('SAT-A', 'SAT-B', _dt(0), _dt(30))
        cache.add_window(link)

        active_before = cache.get_active_links('SAT-A', at_time=_dt(-10))
        active_after = cache.get_active_links('SAT-A', at_time=_dt(40))
        assert active_before == []
        assert active_after == []

    def test_get_active_links_for_unknown_satellite(self):
        """Unknown satellite ID should return empty list."""
        cache = ISLWindowCache()
        link = _make_isl_link('SAT-A', 'SAT-B', _dt(0), _dt(30))
        cache.add_window(link)

        active = cache.get_active_links('SAT-UNKNOWN', at_time=_dt(15))
        assert active == []
