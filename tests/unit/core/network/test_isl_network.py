"""
Test suite for ISL (Inter-Satellite Link) and network routing.
Tests ISLVisibilityCalculator, NetworkRouter, RelayNetwork.
"""
import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import Mock

import sys
sys.path.insert(0, '/Users/zhaolin/Documents/职称论文/missionPlanAlgo')

from core.models.satellite import Satellite
from core.models.ground_station import GroundStation

# Skip if network module not yet implemented
pytest.importorskip("core.network.isl_visibility", reason="ISL module not yet implemented")
pytest.importorskip("core.network.network_router", reason="Network router not yet implemented")

from core.network.isl_visibility import ISLLink, ISLVisibilityCalculator
from core.network.network_router import NetworkRouter, RoutePath
from core.network.relay_satellite import RelaySatellite, RelayNetwork


class TestISLLink:
    """Test ISL link data structure"""

    def test_link_creation(self):
        """Test ISL link creation"""
        start = datetime.now()
        end = start + timedelta(minutes=10)

        link = ISLLink(
            satellite_a_id='SAT-01',
            satellite_b_id='SAT-02',
            start_time=start,
            end_time=end,
            link_quality=0.95,
            max_data_rate=10000.0,
            distance=2000.0
        )

        assert link.satellite_a_id == 'SAT-01'
        assert link.satellite_b_id == 'SAT-02'
        assert link.max_data_rate == 10000.0


class TestISLVisibilityCalculator:
    """Test ISL visibility calculator"""

    @pytest.fixture
    def mock_satellites(self):
        """Create mock satellites in similar orbits"""
        satellites = []
        for i in range(3):
            sat = Mock(spec=Satellite)
            sat.id = f'SAT-{i+1:02d}'
            # Mock positions at different points in orbit
            angle = i * 2 * math.pi / 3
            sat.get_position = Mock(return_value=(
                7000e3 * math.cos(angle),  # km
                7000e3 * math.sin(angle),
                0.0
            ))
            satellites.append(sat)
        return satellites

    @pytest.fixture
    def calculator(self):
        """Create ISL calculator"""
        return ISLVisibilityCalculator(
            link_type='laser',
            max_link_distance=5000.0,
            min_elevation_angle=0.0
        )

    def test_initialization(self):
        """Test calculator initialization"""
        calc = ISLVisibilityCalculator(link_type='laser')
        assert calc.link_type == 'laser'
        assert calc.max_data_rate == 10000.0  # 10 Gbps for laser

        calc = ISLVisibilityCalculator(link_type='microwave')
        assert calc.max_data_rate == 1000.0  # 1 Gbps for microwave

    def test_compute_isl_windows(self, calculator, mock_satellites):
        """Test ISL window computation"""
        start = datetime.now()
        end = start + timedelta(hours=1)

        windows = calculator.compute_isl_windows(
            satellites=mock_satellites,
            start_time=start,
            end_time=end,
            time_step=60
        )

        # Should return dict keyed by satellite pair
        assert isinstance(windows, dict)

    def test_compute_pair_windows(self, calculator, mock_satellites):
        """Test pair-wise ISL window computation"""
        start = datetime.now()
        end = start + timedelta(minutes=30)

        windows = calculator._compute_pair_windows(
            sat_a=mock_satellites[0],
            sat_b=mock_satellites[1],
            start_time=start,
            end_time=end,
            time_step=60
        )

        # Should return list of ISLLink objects
        assert isinstance(windows, list)

    def test_link_distance_calculation(self, calculator, mock_satellites):
        """Test link distance calculation"""
        pos_a = mock_satellites[0].get_position()
        pos_b = mock_satellites[1].get_position()

        distance = calculator._calculate_distance(pos_a, pos_b)

        assert distance > 0
        assert isinstance(distance, float)


class TestNetworkRouter:
    """Test network router with Dijkstra algorithm"""

    @pytest.fixture
    def mock_isl_calculator(self):
        """Create mock ISL calculator"""
        calc = Mock(spec=ISLVisibilityCalculator)
        calc.isl_windows = {}
        return calc

    @pytest.fixture
    def mock_ground_stations(self):
        """Create mock ground stations"""
        stations = []
        for i in range(2):
            gs = Mock(spec=GroundStation)
            gs.id = f'GS-{i+1:02d}'
            stations.append(gs)
        return stations

    @pytest.fixture
    def router(self, mock_isl_calculator, mock_ground_stations):
        """Create network router"""
        return NetworkRouter(
            isl_calculator=mock_isl_calculator,
            ground_stations=mock_ground_stations
        )

    def test_initialization(self, mock_isl_calculator, mock_ground_stations):
        """Test router initialization"""
        router = NetworkRouter(mock_isl_calculator, mock_ground_stations)
        assert router.isl_calculator == mock_isl_calculator

    def test_find_best_route_latency(self, router):
        """Test finding route with latency priority"""
        start_time = datetime.now()

        # Setup ISL windows with proper ISLLink objects
        router.isl_windows = {
            ('SAT-01', 'SAT-02'): [
                ISLLink(
                    satellite_a_id='SAT-01',
                    satellite_b_id='SAT-02',
                    start_time=start_time - timedelta(minutes=5),
                    end_time=start_time + timedelta(minutes=10),
                    link_quality=0.95,
                    max_data_rate=10000.0,
                    distance=2000.0
                )
            ]
        }

        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='latency'
        )

        # Should find a route
        assert route is not None
        assert route.source_satellite == 'SAT-01'

    def test_dijkstra_latency(self, router):
        """Test Dijkstra algorithm for latency"""
        # Simple topology: SAT-01 -> SAT-02 -> GS
        topology = {
            'SAT-01': [('SAT-02', 1.0)],
            'SAT-02': [('GS-01', 1.0)],
            'GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS-01',
            topology=topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        if route:
            assert route.source_satellite == 'SAT-01'
            assert route.destination == 'GS-01'

    def test_calculate_path_latency(self, router):
        """Test path latency calculation"""
        hops = ['SAT-01', 'SAT-02', 'SAT-03']
        latency = router._calculate_path_latency(hops, 1.0)

        assert latency > 0
        assert isinstance(latency, float)


class TestRelaySatellite:
    """Test relay satellite functionality"""

    def test_relay_creation(self):
        """Test relay satellite creation"""
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
        assert relay.orbit_type == 'GEO'
        assert relay.longitude == 120.0


class TestRelayNetwork:
    """Test relay network"""

    @pytest.fixture
    def relay_satellites(self):
        """Create relay satellites"""
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

    @pytest.fixture
    def network(self, relay_satellites):
        """Create relay network"""
        return RelayNetwork(relay_satellites)

    def test_initialization(self, relay_satellites):
        """Test network initialization"""
        network = RelayNetwork(relay_satellites)
        assert len(network.relays) == 2

    def test_can_relay_data(self, network):
        """Test data relay feasibility check"""
        start_time = datetime.now()

        # Add visibility window using proper dict format
        network.relay_visibility[('SAT-01', 'RELAY-01')] = [
            {
                'start_time': start_time,
                'end_time': start_time + timedelta(minutes=30)
            }
        ]

        can_relay, latency = network.can_relay_data(
            source_satellite='SAT-01',
            relay_id='RELAY-01',
            data_size=10.0,
            start_time=start_time
        )

        assert isinstance(can_relay, bool)
        assert isinstance(latency, float)

    def test_get_available_relays(self, network):
        """Test getting available relays"""
        start_time = datetime.now()

        # Add visibility using proper dict format
        network.relay_visibility[('SAT-01', 'RELAY-01')] = [
            {
                'start_time': start_time,
                'end_time': start_time + timedelta(minutes=30)
            }
        ]

        available = network.get_available_relays('SAT-01', start_time)
        assert isinstance(available, list)
        assert 'RELAY-01' in available


