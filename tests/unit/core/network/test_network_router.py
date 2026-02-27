"""
Unit tests for NetworkRouter module.

Tests Dijkstra routing algorithm, multi-hop path calculation,
and optimization by latency, bandwidth, and reliability.
"""
import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from core.models.satellite import Satellite
from core.models.ground_station import GroundStation
from core.network.isl_visibility import ISLLink, ISLVisibilityCalculator
from core.network.network_router import NetworkRouter, RoutePath


class TestRoutePath:
    """Test RoutePath data class"""

    def test_route_path_creation(self):
        """Test RoutePath creation with all attributes"""
        route = RoutePath(
            source_satellite='SAT-01',
            destination='GS-01',
            hops=['SAT-02', 'SAT-03'],
            total_latency=0.5,
            available_bandwidth=1000.0,
            path_reliability=0.95
        )

        assert route.source_satellite == 'SAT-01'
        assert route.destination == 'GS-01'
        assert route.hops == ['SAT-02', 'SAT-03']
        assert route.total_latency == 0.5
        assert route.available_bandwidth == 1000.0
        assert route.path_reliability == 0.95

    def test_route_path_empty_hops(self):
        """Test RoutePath with empty hops (direct connection)"""
        route = RoutePath(
            source_satellite='SAT-01',
            destination='GS-01',
            hops=[],
            total_latency=0.1,
            available_bandwidth=500.0,
            path_reliability=0.99
        )

        assert route.hops == []
        assert len(route.hops) == 0

    def test_route_path_single_hop(self):
        """Test RoutePath with single hop"""
        route = RoutePath(
            source_satellite='SAT-01',
            destination='GS-01',
            hops=['SAT-02'],
            total_latency=0.2,
            available_bandwidth=800.0,
            path_reliability=0.90
        )

        assert len(route.hops) == 1
        assert route.hops[0] == 'SAT-02'


class TestNetworkRouterInitialization:
    """Test NetworkRouter initialization"""

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

    def test_initialization(self, mock_isl_calculator, mock_ground_stations):
        """Test router initialization with valid parameters"""
        router = NetworkRouter(
            isl_calculator=mock_isl_calculator,
            ground_stations=mock_ground_stations
        )

        assert router.isl_calculator == mock_isl_calculator
        assert len(router.ground_stations) == 2
        assert 'GS-01' in router.ground_stations
        assert 'GS-02' in router.ground_stations
        assert router.isl_windows == {}

    def test_initialization_empty_ground_stations(self, mock_isl_calculator):
        """Test router initialization with empty ground stations"""
        router = NetworkRouter(
            isl_calculator=mock_isl_calculator,
            ground_stations=[]
        )

        assert len(router.ground_stations) == 0

    def test_set_isl_windows(self, mock_isl_calculator, mock_ground_stations):
        """Test setting ISL windows"""
        router = NetworkRouter(
            isl_calculator=mock_isl_calculator,
            ground_stations=mock_ground_stations
        )

        start_time = datetime.now()
        isl_windows = {
            ('SAT-01', 'SAT-02'): [
                ISLLink(
                    satellite_a_id='SAT-01',
                    satellite_b_id='SAT-02',
                    start_time=start_time,
                    end_time=start_time + timedelta(minutes=10),
                    link_quality=0.95,
                    max_data_rate=10000.0,
                    distance=2000.0
                )
            ]
        }

        router.set_isl_windows(isl_windows)
        assert router.isl_windows == isl_windows


class TestNetworkRouterTopology:
    """Test network topology building"""

    @pytest.fixture
    def router_with_topology(self):
        """Create router with predefined topology"""
        calc = Mock(spec=ISLVisibilityCalculator)
        gs = Mock(spec=GroundStation)
        gs.id = 'GS-01'

        router = NetworkRouter(
            isl_calculator=calc,
            ground_stations=[gs]
        )

        start_time = datetime.now()

        # Set up ISL windows
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
            ],
            ('SAT-02', 'SAT-03'): [
                ISLLink(
                    satellite_a_id='SAT-02',
                    satellite_b_id='SAT-03',
                    start_time=start_time - timedelta(minutes=5),
                    end_time=start_time + timedelta(minutes=10),
                    link_quality=0.90,
                    max_data_rate=8000.0,
                    distance=3000.0
                )
            ]
        }

        return router, start_time

    def test_build_network_topology(self, router_with_topology):
        """Test building network topology from ISL windows"""
        router, start_time = router_with_topology

        topology = router._build_network_topology(start_time)

        assert 'SAT-01' in topology
        assert 'SAT-02' in topology
        assert 'SAT-03' in topology
        assert 'GS:GS-01' in topology

        # Check bidirectional links
        sat_01_neighbors = [n[0] for n in topology['SAT-01']]
        assert 'SAT-02' in sat_01_neighbors

        sat_02_neighbors = [n[0] for n in topology['SAT-02']]
        assert 'SAT-01' in sat_02_neighbors
        assert 'SAT-03' in sat_02_neighbors

    def test_build_topology_no_active_links(self, router_with_topology):
        """Test topology building when no links are active"""
        router, start_time = router_with_topology

        # Use time outside all windows
        future_time = start_time + timedelta(hours=1)
        topology = router._build_network_topology(future_time)

        # Should still have ground station
        assert 'GS:GS-01' in topology
        # But satellites may not be connected

    def test_build_topology_ground_station_connections(self, router_with_topology):
        """Test that ground stations are connected to satellites"""
        router, start_time = router_with_topology

        topology = router._build_network_topology(start_time)

        # Each satellite should have ground station connection
        for sat_id in ['SAT-01', 'SAT-02', 'SAT-03']:
            if sat_id in topology:
                neighbors = [n[0] for n in topology[sat_id]]
                assert 'GS:GS-01' in neighbors


class TestNetworkRouterDijkstra:
    """Test Dijkstra routing algorithms"""

    @pytest.fixture
    def router(self):
        """Create basic router"""
        calc = Mock(spec=ISLVisibilityCalculator)
        gs = Mock(spec=GroundStation)
        gs.id = 'GS-01'

        router = NetworkRouter(
            isl_calculator=calc,
            ground_stations=[gs]
        )

        return router

    @pytest.fixture
    def simple_topology(self):
        """Simple linear topology: SAT-01 -> SAT-02 -> GS"""
        return {
            'SAT-01': [('SAT-02', 1.0)],
            'SAT-02': [('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

    @pytest.fixture
    def multi_hop_topology(self):
        """Multi-hop topology with multiple paths"""
        return {
            'SAT-01': [('SAT-02', 1.0), ('SAT-03', 2.0)],
            'SAT-02': [('SAT-04', 1.0)],
            'SAT-03': [('SAT-04', 1.0)],
            'SAT-04': [('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

    def test_dijkstra_latency_direct_path(self, router, simple_topology):
        """Test Dijkstra latency with direct path"""
        start_time = datetime.now()

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=simple_topology,
            data_size=1.0,
            start_time=start_time
        )

        assert route is not None
        assert route.source_satellite == 'SAT-01'
        assert route.destination == 'GS-01'
        assert route.hops == ['SAT-02']
        assert route.total_latency > 0

    def test_dijkstra_latency_no_path(self, router):
        """Test Dijkstra when no path exists"""
        topology = {
            'SAT-01': [],
            'GS:GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        assert route is None

    def test_dijkstra_latency_source_not_in_topology(self, router, simple_topology):
        """Test Dijkstra when source not in topology"""
        route = router._dijkstra_latency(
            source='SAT-99',
            target='GS:GS-01',
            topology=simple_topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        assert route is None

    def test_dijkstra_latency_multi_hop(self, router, multi_hop_topology):
        """Test Dijkstra with multiple hops"""
        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=multi_hop_topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        assert route is not None
        assert route.source_satellite == 'SAT-01'
        assert route.destination == 'GS-01'
        # Should find shortest path with 2 hops
        # Both SAT-01->SAT-02->SAT-04 and SAT-01->SAT-03->SAT-04 have same cost
        assert len(route.hops) == 2
        assert route.hops[1] == 'SAT-04'  # Last hop should be SAT-04

    def test_dijkstra_bandwidth(self, router, simple_topology):
        """Test Dijkstra bandwidth optimization"""
        start_time = datetime.now()

        route = router._dijkstra_bandwidth(
            source='SAT-01',
            target='GS:GS-01',
            topology=simple_topology,
            data_size=1.0,
            start_time=start_time
        )

        # Currently returns same as latency
        assert route is not None
        assert route.source_satellite == 'SAT-01'

    def test_dijkstra_reliability(self, router, simple_topology):
        """Test Dijkstra reliability optimization"""
        start_time = datetime.now()

        route = router._dijkstra_reliability(
            source='SAT-01',
            target='GS:GS-01',
            topology=simple_topology,
            data_size=1.0,
            start_time=start_time
        )

        # Currently returns same as latency
        assert route is not None
        assert route.source_satellite == 'SAT-01'


class TestNetworkRouterPathCalculations:
    """Test path calculation methods"""

    @pytest.fixture
    def router_with_isl_windows(self):
        """Create router with ISL windows"""
        calc = Mock(spec=ISLVisibilityCalculator)
        gs = Mock(spec=GroundStation)
        gs.id = 'GS-01'

        router = NetworkRouter(
            isl_calculator=calc,
            ground_stations=[gs]
        )

        start_time = datetime.now()

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
            ],
            ('SAT-02', 'SAT-03'): [
                ISLLink(
                    satellite_a_id='SAT-02',
                    satellite_b_id='SAT-03',
                    start_time=start_time - timedelta(minutes=5),
                    end_time=start_time + timedelta(minutes=10),
                    link_quality=0.90,
                    max_data_rate=8000.0,
                    distance=3000.0
                )
            ]
        }

        return router, start_time

    def test_estimate_link_latency(self, router_with_isl_windows):
        """Test link latency estimation"""
        router, _ = router_with_isl_windows

        latency = router._estimate_link_latency('SAT-01', 'SAT-02', 1.0, 0.95)

        assert latency > 0
        assert isinstance(latency, float)
        # Higher quality should give lower latency
        latency_high_quality = router._estimate_link_latency('SAT-01', 'SAT-02', 1.0, 0.99)
        latency_low_quality = router._estimate_link_latency('SAT-01', 'SAT-02', 1.0, 0.5)
        assert latency_high_quality < latency_low_quality

    def test_estimate_link_latency_low_quality(self, router_with_isl_windows):
        """Test link latency with very low quality"""
        router, _ = router_with_isl_windows

        # Test with quality below 0.1 (should use min of 0.1)
        latency = router._estimate_link_latency('SAT-01', 'SAT-02', 1.0, 0.05)
        latency_min = router._estimate_link_latency('SAT-01', 'SAT-02', 1.0, 0.1)

        assert latency == latency_min

    def test_calculate_path_bandwidth(self, router_with_isl_windows):
        """Test path bandwidth calculation"""
        router, start_time = router_with_isl_windows

        path = ['SAT-01', 'SAT-02', 'SAT-03']
        bandwidth = router._calculate_path_bandwidth(path, start_time)

        # Should return minimum bandwidth along path
        assert bandwidth > 0
        assert bandwidth == 8000.0  # min(10000, 8000)

    def test_calculate_path_bandwidth_single_node(self, router_with_isl_windows):
        """Test path bandwidth with single node path"""
        router, start_time = router_with_isl_windows

        bandwidth = router._calculate_path_bandwidth(['SAT-01'], start_time)
        assert bandwidth == 0.0

    def test_calculate_path_bandwidth_no_link(self, router_with_isl_windows):
        """Test path bandwidth when link not found"""
        router, start_time = router_with_isl_windows

        path = ['SAT-01', 'SAT-99']  # SAT-99 not in windows
        bandwidth = router._calculate_path_bandwidth(path, start_time)

        assert bandwidth == 0.0

    def test_calculate_path_reliability(self, router_with_isl_windows):
        """Test path reliability calculation"""
        router, start_time = router_with_isl_windows

        path = ['SAT-01', 'SAT-02', 'SAT-03']
        reliability = router._calculate_path_reliability(path, start_time)

        # Should be product of link qualities: 0.95 * 0.90
        expected = 0.95 * 0.90
        assert abs(reliability - expected) < 0.001

    def test_calculate_path_reliability_single_node(self, router_with_isl_windows):
        """Test path reliability with single node"""
        router, start_time = router_with_isl_windows

        reliability = router._calculate_path_reliability(['SAT-01'], start_time)
        assert reliability == 1.0

    def test_calculate_path_reliability_no_link(self, router_with_isl_windows):
        """Test path reliability when link not found"""
        router, start_time = router_with_isl_windows

        path = ['SAT-01', 'SAT-99']
        reliability = router._calculate_path_reliability(path, start_time)

        assert reliability == 1.0  # No links multiplied

    def test_calculate_path_latency(self, router_with_isl_windows):
        """Test path latency calculation"""
        router, _ = router_with_isl_windows

        hops = ['SAT-02', 'SAT-03']
        data_size = 1.0  # GB

        latency = router._calculate_path_latency(hops, data_size)

        # 3 hops (2 ISL + 1 ground) * 0.01s + transmission time
        assert latency > 0
        assert isinstance(latency, float)

    def test_calculate_path_latency_empty_hops(self, router_with_isl_windows):
        """Test path latency with empty hops"""
        router, _ = router_with_isl_windows

        latency = router._calculate_path_latency([], 1.0)

        # Just ground link latency
        assert latency > 0


class TestNetworkRouterFindBestRoute:
    """Test find_best_route method"""

    @pytest.fixture
    def router_with_full_setup(self):
        """Create router with complete setup"""
        calc = Mock(spec=ISLVisibilityCalculator)
        gs = Mock(spec=GroundStation)
        gs.id = 'GS-01'

        router = NetworkRouter(
            isl_calculator=calc,
            ground_stations=[gs]
        )

        start_time = datetime.now()

        # Set up ISL windows for a complete path
        router.set_isl_windows({
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
        })

        return router, start_time

    def test_find_best_route_latency_priority(self, router_with_full_setup):
        """Test find_best_route with latency priority"""
        router, start_time = router_with_full_setup

        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='latency'
        )

        assert route is not None
        assert route.source_satellite == 'SAT-01'
        assert route.destination == 'GS-01'

    def test_find_best_route_bandwidth_priority(self, router_with_full_setup):
        """Test find_best_route with bandwidth priority"""
        router, start_time = router_with_full_setup

        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='bandwidth'
        )

        assert route is not None
        assert route.source_satellite == 'SAT-01'

    def test_find_best_route_reliability_priority(self, router_with_full_setup):
        """Test find_best_route with reliability priority"""
        router, start_time = router_with_full_setup

        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='reliability'
        )

        assert route is not None
        assert route.source_satellite == 'SAT-01'

    def test_find_best_route_source_not_in_topology(self, router_with_full_setup):
        """Test find_best_route when source not in topology"""
        router, start_time = router_with_full_setup

        route = router.find_best_route(
            source_satellite='SAT-99',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='latency'
        )

        assert route is None

    def test_find_best_route_invalid_priority(self, router_with_full_setup):
        """Test find_best_route with invalid priority (defaults to reliability)"""
        router, start_time = router_with_full_setup

        # Invalid priority should default to reliability path
        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='invalid_priority'
        )

        # Should still return a route (using reliability path)
        assert route is not None

    def test_find_best_route_no_isl_windows(self):
        """Test find_best_route with no ISL windows set"""
        calc = Mock(spec=ISLVisibilityCalculator)
        gs = Mock(spec=GroundStation)
        gs.id = 'GS-01'

        router = NetworkRouter(
            isl_calculator=calc,
            ground_stations=[gs]
        )

        start_time = datetime.now()

        # No ISL windows set
        route = router.find_best_route(
            source_satellite='SAT-01',
            destination_ground_station='GS-01',
            data_size=1.0,
            start_time=start_time,
            priority='latency'
        )

        # Should return None since no topology can be built
        assert route is None


class TestNetworkRouterEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.fixture
    def router(self):
        """Create basic router"""
        calc = Mock(spec=ISLVisibilityCalculator)
        gs = Mock(spec=GroundStation)
        gs.id = 'GS-01'

        return NetworkRouter(
            isl_calculator=calc,
            ground_stations=[gs]
        )

    def test_zero_data_size(self, router):
        """Test routing with zero data size"""
        topology = {
            'SAT-01': [('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=topology,
            data_size=0.0,
            start_time=datetime.now()
        )

        assert route is not None
        assert route.total_latency >= 0

    def test_large_data_size(self, router):
        """Test routing with large data size"""
        topology = {
            'SAT-01': [('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=topology,
            data_size=1000.0,  # 1000 GB
            start_time=datetime.now()
        )

        assert route is not None

    def test_circular_topology(self, router):
        """Test routing with circular topology"""
        topology = {
            'SAT-01': [('SAT-02', 1.0)],
            'SAT-02': [('SAT-03', 1.0)],
            'SAT-03': [('SAT-01', 1.0), ('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        assert route is not None
        # Should find path without infinite loop
        assert route.hops == ['SAT-02', 'SAT-03']

    def test_disconnected_topology(self, router):
        """Test routing in disconnected topology"""
        topology = {
            'SAT-01': [('SAT-02', 1.0)],
            'SAT-02': [],  # Dead end
            'SAT-03': [('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        assert route is None

    def test_self_loop(self, router):
        """Test topology with self-loop"""
        topology = {
            'SAT-01': [('SAT-01', 1.0), ('GS:GS-01', 1.0)],
            'GS:GS-01': []
        }

        route = router._dijkstra_latency(
            source='SAT-01',
            target='GS:GS-01',
            topology=topology,
            data_size=1.0,
            start_time=datetime.now()
        )

        assert route is not None
        assert route.hops == []  # Direct connection
