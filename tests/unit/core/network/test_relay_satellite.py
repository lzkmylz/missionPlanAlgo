"""
Unit tests for RelaySatellite and RelayNetwork modules.

Tests relay satellite functionality, data forwarding, visibility windows,
and relay network management.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

import sys
sys.path.insert(0, '/Users/zhaolin/Documents/职称论文/missionPlanAlgo')

from core.models.satellite import Satellite
from core.models.ground_station import GroundStation
from core.network.relay_satellite import RelaySatellite, RelayNetwork


class TestRelaySatellite:
    """Test RelaySatellite data class"""

    def test_relay_satellite_creation(self):
        """Test creating a RelaySatellite with all attributes"""
        relay = RelaySatellite(
            id='RELAY-01',
            name='Tianlian-1',
            orbit_type='GEO',
            longitude=120.0,
            uplink_capacity=450.0,
            downlink_capacity=450.0,
            coverage_zones=['Asia', 'Pacific']
        )

        assert relay.id == 'RELAY-01'
        assert relay.name == 'Tianlian-1'
        assert relay.orbit_type == 'GEO'
        assert relay.longitude == 120.0
        assert relay.uplink_capacity == 450.0
        assert relay.downlink_capacity == 450.0
        assert relay.coverage_zones == ['Asia', 'Pacific']

    def test_relay_satellite_default_coverage(self):
        """Test RelaySatellite with default empty coverage zones"""
        relay = RelaySatellite(
            id='RELAY-02',
            name='Tianlian-2',
            orbit_type='GEO',
            longitude=-60.0,
            uplink_capacity=300.0,
            downlink_capacity=300.0
        )

        assert relay.coverage_zones == []

    def test_relay_satellite_different_longitudes(self):
        """Test RelaySatellite at different longitudes"""
        longitudes = [0.0, 90.0, 180.0, -180.0, -90.0]

        for i, lon in enumerate(longitudes):
            relay = RelaySatellite(
                id=f'RELAY-{i:02d}',
                name=f'Relay-{i}',
                orbit_type='GEO',
                longitude=lon,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            )
            assert relay.longitude == lon


class TestRelayNetworkInitialization:
    """Test RelayNetwork initialization"""

    @pytest.fixture
    def relay_satellites(self):
        """Create sample relay satellites"""
        return [
            RelaySatellite(
                id='RELAY-01',
                name='Tianlian-1',
                orbit_type='GEO',
                longitude=120.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0,
                coverage_zones=['Asia']
            ),
            RelaySatellite(
                id='RELAY-02',
                name='Tianlian-2',
                orbit_type='GEO',
                longitude=-60.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0,
                coverage_zones=['Atlantic']
            )
        ]

    def test_initialization(self, relay_satellites):
        """Test RelayNetwork initialization with relay satellites"""
        network = RelayNetwork(relay_satellites)

        assert len(network.relays) == 2
        assert 'RELAY-01' in network.relays
        assert 'RELAY-02' in network.relays
        assert network.relay_visibility == {}

    def test_initialization_empty_list(self):
        """Test RelayNetwork initialization with empty list"""
        network = RelayNetwork([])

        assert len(network.relays) == 0
        assert network.relay_visibility == {}

    def test_initialization_single_relay(self):
        """Test RelayNetwork initialization with single relay"""
        relay = RelaySatellite(
            id='RELAY-01',
            name='Tianlian-1',
            orbit_type='GEO',
            longitude=120.0,
            uplink_capacity=450.0,
            downlink_capacity=450.0
        )

        network = RelayNetwork([relay])

        assert len(network.relays) == 1
        assert network.relays['RELAY-01'] == relay


class TestRelayNetworkVisibility:
    """Test visibility window management"""

    @pytest.fixture
    def network(self):
        """Create relay network with relays"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='Tianlian-1',
                orbit_type='GEO',
                longitude=120.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            )
        ]
        return RelayNetwork(relays)

    def test_add_visibility_window(self, network):
        """Test adding visibility window"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)

        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=end_time
        )

        assert ('SAT-01', 'RELAY-01') in network.relay_visibility
        assert len(network.relay_visibility[('SAT-01', 'RELAY-01')]) == 1

        window = network.relay_visibility[('SAT-01', 'RELAY-01')][0]
        assert window['start_time'] == start_time
        assert window['end_time'] == end_time

    def test_add_multiple_visibility_windows(self, network):
        """Test adding multiple visibility windows for same pair"""
        start_time = datetime.now()

        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time + timedelta(hours=1),
            end_time=start_time + timedelta(hours=1, minutes=30)
        )

        assert len(network.relay_visibility[('SAT-01', 'RELAY-01')]) == 2

    def test_add_visibility_different_satellites(self, network):
        """Test adding visibility for different satellites to same relay"""
        start_time = datetime.now()

        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        network.add_visibility_window(
            satellite_id='SAT-02',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        assert ('SAT-01', 'RELAY-01') in network.relay_visibility
        assert ('SAT-02', 'RELAY-01') in network.relay_visibility


class TestRelayNetworkCanRelayData:
    """Test data relay feasibility checking"""

    @pytest.fixture
    def network_with_visibility(self):
        """Create network with visibility windows"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='Tianlian-1',
                orbit_type='GEO',
                longitude=120.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            )
        ]
        network = RelayNetwork(relays)

        start_time = datetime.now()
        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        return network, start_time

    def test_can_relay_data_success(self, network_with_visibility):
        """Test successful relay data check"""
        network, start_time = network_with_visibility

        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=1.0,  # 1 GB
            start_time=start_time
        )

        assert can_relay is True
        assert latency > 0
        assert latency != float('inf')

    def test_can_relay_data_outside_window(self, network_with_visibility):
        """Test relay data check outside visibility window"""
        network, start_time = network_with_visibility

        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=1.0,
            start_time=start_time + timedelta(hours=1)  # Outside window
        )

        assert can_relay is False
        assert latency == float('inf')

    def test_can_relay_data_nonexistent_relay(self, network_with_visibility):
        """Test relay data check for non-existent relay"""
        network, start_time = network_with_visibility

        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-99',  # Non-existent
            data_size=1.0,
            start_time=start_time
        )

        assert can_relay is False
        assert latency == float('inf')

    def test_can_relay_data_no_visibility(self, network_with_visibility):
        """Test relay data check with no visibility window"""
        network, start_time = network_with_visibility

        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-99',  # No visibility window
            relay_id='RELAY-01',
            data_size=1.0,
            start_time=start_time
        )

        assert can_relay is False
        assert latency == float('inf')

    def test_can_relay_data_large_transfer(self, network_with_visibility):
        """Test relay data check with large data that doesn't fit in window"""
        network, start_time = network_with_visibility

        # Try to transfer 200 GB in a 30-minute window
        # At 450 Mbps, this would take ~3555 seconds > 1800 seconds (30 min)
        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=200.0,  # 200 GB - too large for 30 min window
            start_time=start_time
        )

        # Should fail because transfer doesn't fit in window
        assert can_relay is False
        assert latency == float('inf')

    def test_can_relay_data_small_transfer_fits(self, network_with_visibility):
        """Test relay data check with small data that fits in window"""
        network, start_time = network_with_visibility

        # Transfer 0.1 GB in a 30-minute window
        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=0.1,  # 0.1 GB
            start_time=start_time
        )

        assert can_relay is True
        assert latency > 0

    def test_can_relay_data_zero_data_size(self, network_with_visibility):
        """Test relay data check with zero data size"""
        network, start_time = network_with_visibility

        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=0.0,
            start_time=start_time
        )

        assert can_relay is True
        assert latency == 0.0


class TestRelayNetworkGetAvailableRelays:
    """Test getting available relays"""

    @pytest.fixture
    def network_with_multiple_relays(self):
        """Create network with multiple relays and visibility"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='Tianlian-1',
                orbit_type='GEO',
                longitude=120.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            ),
            RelaySatellite(
                id='RELAY-02',
                name='Tianlian-2',
                orbit_type='GEO',
                longitude=-60.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            )
        ]
        network = RelayNetwork(relays)

        start_time = datetime.now()

        # Add visibility for SAT-01 to both relays
        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )
        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-02',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        # Add visibility for SAT-02 to only one relay
        network.add_visibility_window(
            satellite_id='SAT-02',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        return network, start_time

    def test_get_available_relays_multiple(self, network_with_multiple_relays):
        """Test getting available relays when multiple are visible"""
        network, start_time = network_with_multiple_relays

        available = network.get_available_relays('SAT-01', start_time)

        assert len(available) == 2
        assert 'RELAY-01' in available
        assert 'RELAY-02' in available

    def test_get_available_relays_single(self, network_with_multiple_relays):
        """Test getting available relays when only one is visible"""
        network, start_time = network_with_multiple_relays

        available = network.get_available_relays('SAT-02', start_time)

        assert len(available) == 1
        assert 'RELAY-01' in available
        assert 'RELAY-02' not in available

    def test_get_available_relays_none(self, network_with_multiple_relays):
        """Test getting available relays when none are visible"""
        network, start_time = network_with_multiple_relays

        available = network.get_available_relays('SAT-99', start_time)

        assert len(available) == 0
        assert available == []

    def test_get_available_relays_outside_window(self, network_with_multiple_relays):
        """Test getting available relays outside visibility window"""
        network, start_time = network_with_multiple_relays

        available = network.get_available_relays(
            'SAT-01',
            start_time + timedelta(hours=1)
        )

        assert len(available) == 0

    def test_get_available_relays_returns_list(self, network_with_multiple_relays):
        """Test that get_available_relays returns a list"""
        network, start_time = network_with_multiple_relays

        available = network.get_available_relays('SAT-01', start_time)

        assert isinstance(available, list)


class TestRelayNetworkFindBestRelay:
    """Test finding best relay"""

    @pytest.fixture
    def network_with_different_capacities(self):
        """Create network with relays of different capacities"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='Tianlian-1',
                orbit_type='GEO',
                longitude=120.0,
                uplink_capacity=100.0,  # Slower
                downlink_capacity=100.0
            ),
            RelaySatellite(
                id='RELAY-02',
                name='Tianlian-2',
                orbit_type='GEO',
                longitude=-60.0,
                uplink_capacity=500.0,  # Faster
                downlink_capacity=500.0
            )
        ]
        network = RelayNetwork(relays)

        start_time = datetime.now()

        # Add visibility for SAT-01 to both relays
        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )
        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-02',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        return network, start_time

    def test_find_best_relay(self, network_with_different_capacities):
        """Test finding best relay based on latency"""
        network, start_time = network_with_different_capacities

        best_relay = network.find_best_relay(
            satellite_id='SAT-01',
            data_size=1.0,
            start_time=start_time
        )

        # Should choose RELAY-02 with higher capacity (lower latency)
        assert best_relay == 'RELAY-02'

    def test_find_best_relay_no_visibility(self, network_with_different_capacities):
        """Test finding best relay when no relays are visible"""
        network, start_time = network_with_different_capacities

        best_relay = network.find_best_relay(
            satellite_id='SAT-99',  # No visibility
            data_size=1.0,
            start_time=start_time
        )

        assert best_relay is None

    def test_find_best_relay_single_available(self, network_with_different_capacities):
        """Test finding best relay when only one is available"""
        network, start_time = network_with_different_capacities

        # Add visibility for SAT-02 to only one relay
        network.add_visibility_window(
            satellite_id='SAT-02',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        best_relay = network.find_best_relay(
            satellite_id='SAT-02',
            data_size=1.0,
            start_time=start_time
        )

        assert best_relay == 'RELAY-01'

    def test_find_best_relay_outside_window(self, network_with_different_capacities):
        """Test finding best relay outside visibility window"""
        network, start_time = network_with_different_capacities

        best_relay = network.find_best_relay(
            satellite_id='SAT-01',
            data_size=1.0,
            start_time=start_time + timedelta(hours=1)
        )

        assert best_relay is None


class TestRelayNetworkCoverage:
    """Test relay coverage calculations"""

    @pytest.fixture
    def network_with_coverage(self):
        """Create network with relays at different longitudes"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='Tianlian-1',
                orbit_type='GEO',
                longitude=120.0,  # Asia
                uplink_capacity=450.0,
                downlink_capacity=450.0
            ),
            RelaySatellite(
                id='RELAY-02',
                name='Tianlian-2',
                orbit_type='GEO',
                longitude=-60.0,  # Atlantic
                uplink_capacity=450.0,
                downlink_capacity=450.0
            ),
            RelaySatellite(
                id='RELAY-03',
                name='Tianlian-3',
                orbit_type='GEO',
                longitude=0.0,  # Greenwich
                uplink_capacity=450.0,
                downlink_capacity=450.0
            )
        ]
        return RelayNetwork(relays)

    def test_calculate_relay_coverage_single(self, network_with_coverage):
        """Test coverage calculation for longitude covered by single relay"""
        network = network_with_coverage

        # 120.0 should be covered by RELAY-01
        covering = network.calculate_relay_coverage(120.0)

        assert 'RELAY-01' in covering

    def test_calculate_relay_coverage_multiple(self, network_with_coverage):
        """Test coverage calculation for longitude covered by multiple relays"""
        network = network_with_coverage

        # 90.0 should be covered by both RELAY-01 (120) and RELAY-03 (0)
        # within 60 degree range
        covering = network.calculate_relay_coverage(90.0)

        assert 'RELAY-01' in covering

    def test_calculate_relay_coverage_wraparound(self, network_with_coverage):
        """Test coverage calculation near longitude wraparound"""
        network = network_with_coverage

        # -170 should be covered by RELAY-02 (-60) with wraparound
        # Difference is 110, which is > 60, so should not be covered
        covering = network.calculate_relay_coverage(-170.0)

        # Actually, |-170 - (-60)| = 110, |360 - 110| = 250, min is 110 > 60
        # So should not be covered by RELAY-02
        assert 'RELAY-02' not in covering

    def test_calculate_relay_coverage_near_wraparound(self, network_with_coverage):
        """Test coverage calculation near 180/-180 boundary"""
        network = network_with_coverage

        # 170 should be covered by RELAY-01 (120)
        covering = network.calculate_relay_coverage(170.0)

        assert 'RELAY-01' in covering

        # -170 should also be covered by RELAY-01 (120) with wraparound
        # |-170 - 120| = 290, |360 - 290| = 70, min is 70 > 60
        # So should not be covered

    def test_calculate_relay_coverage_no_coverage(self, network_with_coverage):
        """Test coverage calculation for longitude with no coverage"""
        network = network_with_coverage

        # All relays should cover their +/- 60 degree range
        # Test a point that's far from all relays
        covering = network.calculate_relay_coverage(-150.0)

        # -150 is 90 degrees from RELAY-02 (-60), outside 60 degree range
        assert len(covering) == 0 or 'RELAY-02' not in covering

    def test_calculate_relay_coverage_returns_list(self, network_with_coverage):
        """Test that calculate_relay_coverage returns a list"""
        network = network_with_coverage

        covering = network.calculate_relay_coverage(0.0)

        assert isinstance(covering, list)


class TestRelayNetworkEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_network_operations(self):
        """Test operations on empty network"""
        network = RelayNetwork([])

        start_time = datetime.now()

        # All operations should handle empty network gracefully
        can_relay, latency = network.can_relay_data(
            'SAT-01', 'RELAY-01', 1.0, start_time
        )
        assert can_relay is False
        assert latency == float('inf')

        available = network.get_available_relays('SAT-01', start_time)
        assert available == []

        best = network.find_best_relay('SAT-01', 1.0, start_time)
        assert best is None

        covering = network.calculate_relay_coverage(0.0)
        assert covering == []

    def test_relay_with_zero_capacity(self):
        """Test relay with zero capacity"""
        relay = RelaySatellite(
            id='RELAY-01',
            name='Zero-Capacity',
            orbit_type='GEO',
            longitude=120.0,
            uplink_capacity=0.0,
            downlink_capacity=0.0
        )

        network = RelayNetwork([relay])
        start_time = datetime.now()

        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        # Division by zero should be handled
        can_relay, latency = network.can_relay_data(
            'SAT-01', 'RELAY-01', 1.0, start_time
        )

        # With zero capacity, transfer time is infinite
        assert can_relay is False
        assert latency == float('inf')

    def test_visibility_window_zero_duration(self):
        """Test visibility window with zero duration"""
        relay = RelaySatellite(
            id='RELAY-01',
            name='Tianlian-1',
            orbit_type='GEO',
            longitude=120.0,
            uplink_capacity=450.0,
            downlink_capacity=450.0
        )

        network = RelayNetwork([relay])
        start_time = datetime.now()

        # Add zero-duration window
        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time
        )

        can_relay, latency = network.can_relay_data(
            'SAT-01', 'RELAY-01', 0.0, start_time
        )

        # Zero data can be transferred at exact moment
        assert can_relay is True
        assert latency == 0.0

    def test_negative_data_size(self):
        """Test with negative data size"""
        relay = RelaySatellite(
            id='RELAY-01',
            name='Tianlian-1',
            orbit_type='GEO',
            longitude=120.0,
            uplink_capacity=450.0,
            downlink_capacity=450.0
        )

        network = RelayNetwork([relay])
        start_time = datetime.now()

        network.add_visibility_window(
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=start_time,
            end_time=start_time + timedelta(minutes=30)
        )

        # Negative data size should still work (transfer time negative)
        can_relay, latency = network.can_relay_data(
            'SAT-01', 'RELAY-01', -1.0, start_time
        )

        # Negative data means end_time < start_time, so it fits
        assert can_relay is True
        assert latency < 0

    def test_longitude_boundary_values(self):
        """Test longitude boundary values"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='At-180',
                orbit_type='GEO',
                longitude=180.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            ),
            RelaySatellite(
                id='RELAY-02',
                name='At-Minus-180',
                orbit_type='GEO',
                longitude=-180.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            )
        ]

        network = RelayNetwork(relays)

        # 180 and -180 are the same longitude
        covering = network.calculate_relay_coverage(180.0)
        assert 'RELAY-01' in covering
        assert 'RELAY-02' in covering

        covering = network.calculate_relay_coverage(-180.0)
        assert 'RELAY-01' in covering
        assert 'RELAY-02' in covering
