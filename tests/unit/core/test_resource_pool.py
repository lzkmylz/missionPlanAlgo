"""
Test suite for resource pool management.
Tests SatellitePool, GroundStationPool, and ResourceAllocator.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from core.models.satellite import Satellite, SatelliteType, Orbit
from core.models.ground_station import GroundStation, Antenna
from core.models.target import Target

# Skip if resource pool module not yet implemented
pytest.importorskip("core.resources", reason="Resource pool module not yet implemented")

from core.resources.satellite_pool import SatellitePool, SatelliteHealth
from core.resources.ground_station_pool import GroundStationPool, AntennaAllocation
from core.resources.resource_allocator import ResourceAllocator


class TestSatellitePool:
    """Test satellite resource pool"""

    @pytest.fixture
    def mock_satellites(self):
        """Create mock satellite objects"""
        satellites = []
        for i in range(3):
            sat = Mock(spec=Satellite)
            sat.id = f'SAT-{i+1:02d}'
            sat.satellite_type = SatelliteType.OPTICAL_1 if i < 2 else SatelliteType.SAR_1
            sat.capabilities = Mock()
            sat.capabilities.storage_capacity = 500
            sat.capabilities.power_capacity = 2000
            sat.capabilities.imaging_modes = ['push_broom']
            satellites.append(sat)
        return satellites

    @pytest.fixture
    def pool(self, mock_satellites):
        """Create satellite pool"""
        return SatellitePool(mock_satellites)

    def test_pool_initialization(self, mock_satellites):
        """Test pool initialization"""
        pool = SatellitePool(mock_satellites)
        assert len(pool.satellites) == 3
        assert pool.get_available_count() == 3

    def test_allocate_satellite(self, pool, mock_satellites):
        """Test satellite allocation"""
        # Allocate first satellite
        sat = pool.allocate_satellite({'type': 'optical'})
        assert sat is not None
        assert sat.id in ['SAT-01', 'SAT-02']
        assert pool.get_available_count() == 2

    def test_allocate_by_type(self, pool):
        """Test allocation by satellite type"""
        # Allocate SAR satellite
        sat = pool.allocate_satellite({
            'type': 'sar',
            'imaging_modes': ['spotlight']
        })
        assert sat is not None
        assert sat.satellite_type == SatelliteType.SAR_1

    def test_allocate_insufficient_resources(self, pool):
        """Test allocation when resources insufficient"""
        # Allocate all satellites
        for _ in range(3):
            pool.allocate_satellite({})

        # Try to allocate one more
        sat = pool.allocate_satellite({})
        assert sat is None

    def test_release_satellite(self, pool, mock_satellites):
        """Test satellite release"""
        # Allocate and release
        sat = pool.allocate_satellite({})
        assert pool.get_available_count() == 2

        pool.release_satellite(sat.id)
        assert pool.get_available_count() == 3

    def test_get_available_satellites(self, pool):
        """Test getting available satellites"""
        available = pool.get_available_satellites(
            time_window=(datetime.now(), datetime.now() + timedelta(hours=1))
        )
        assert len(available) == 3

    def test_satellite_health_check(self, pool):
        """Test satellite health monitoring"""
        # Initially all healthy
        health = pool.check_satellite_health('SAT-01')
        assert health == SatelliteHealth.HEALTHY

        # Mark as degraded
        pool.update_satellite_health('SAT-01', SatelliteHealth.DEGRADED)
        health = pool.check_satellite_health('SAT-01')
        assert health == SatelliteHealth.DEGRADED

    def test_update_satellite_state(self, pool):
        """Test updating satellite resource state"""
        pool.update_satellite_state('SAT-01', {
            'storage_used': 200,
            'power_level': 1500,
            'current_task': 'TASK-001'
        })

        state = pool.get_satellite_state('SAT-01')
        assert state['storage_used'] == 200
        assert state['power_level'] == 1500


class TestGroundStationPool:
    """Test ground station resource pool"""

    @pytest.fixture
    def mock_antennas(self):
        """Create mock antenna objects"""
        antennas = []
        for i in range(4):
            ant = Mock(spec=Antenna)
            ant.id = f'ANT-{i+1:02d}'
            ant.ground_station_id = f'GS-{(i//2)+1:02d}'
            ant.elevation_min = 5.0
            ant.elevation_max = 90.0
            ant.data_rate = 300.0
            antennas.append(ant)
        return antennas

    @pytest.fixture
    def mock_stations(self, mock_antennas):
        """Create mock ground station objects"""
        stations = []
        for i in range(2):
            gs = Mock(spec=GroundStation)
            gs.id = f'GS-{i+1:02d}'
            gs.name = f'Station {i+1}'
            gs.longitude = 116.0 + i
            gs.latitude = 39.0 + i
            gs.antennas = mock_antennas[i*2:(i+1)*2]
            stations.append(gs)
        return stations

    @pytest.fixture
    def pool(self, mock_stations):
        """Create ground station pool"""
        return GroundStationPool(mock_stations)

    def test_pool_initialization(self, mock_stations):
        """Test pool initialization"""
        pool = GroundStationPool(mock_stations)
        assert len(pool.stations) == 2
        assert pool.get_total_antenna_count() == 4

    def test_allocate_antenna(self, pool):
        """Test antenna allocation"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=10)

        # Initially antenna should be available
        assert pool.is_antenna_available(antenna_id='ANT-01', start_time=start_time, end_time=end_time)

        antenna = pool.allocate_antenna(
            satellite_id='SAT-01',
            time_window=(start_time, end_time)
        )

        assert antenna is not None
        # After allocation, antenna should be unavailable for same time window
        assert not pool.is_antenna_available(antenna_id=antenna.id, start_time=start_time, end_time=end_time)

    def test_allocate_specific_station(self, pool):
        """Test allocation from specific station"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=10)

        antenna = pool.allocate_antenna(
            satellite_id='SAT-01',
            time_window=(start_time, end_time),
            ground_station_id='GS-01'
        )

        assert antenna is not None
        assert antenna.ground_station_id == 'GS-01'

    def test_allocate_conflict_same_time(self, pool):
        """Test allocation conflict at same time"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=10)

        # Allocate first antenna
        ant1 = pool.allocate_antenna(
            satellite_id='SAT-01',
            time_window=(start_time, end_time)
        )

        # Try to allocate same antenna to different satellite at same time
        # Should get different antenna
        ant2 = pool.allocate_antenna(
            satellite_id='SAT-02',
            time_window=(start_time, end_time)
        )

        assert ant2 is not None
        assert ant1.id != ant2.id

    def test_release_antenna(self, pool):
        """Test antenna release"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=10)

        antenna = pool.allocate_antenna(
            satellite_id='SAT-01',
            time_window=(start_time, end_time)
        )
        # After allocation, antenna should be unavailable
        assert not pool.is_antenna_available(antenna_id=antenna.id, start_time=start_time, end_time=end_time)

        pool.release_antenna(antenna.id, start_time, end_time)
        # After release, antenna should be available again
        assert pool.is_antenna_available(antenna_id=antenna.id, start_time=start_time, end_time=end_time)

    def test_get_available_antennas(self, pool):
        """Test getting available antennas"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=10)

        available = pool.get_available_antennas(
            ground_station_id='GS-01',
            time_window=(start_time, end_time)
        )
        assert len(available) == 2

    def test_is_antenna_available(self, pool):
        """Test checking antenna availability"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=10)

        # Initially available
        assert pool.is_antenna_available('ANT-01', start_time, end_time)

        # Allocate it
        pool.allocate_antenna(
            satellite_id='SAT-01',
            time_window=(start_time, end_time)
        )

        # Now unavailable
        assert not pool.is_antenna_available('ANT-01', start_time, end_time)


class TestResourceAllocator:
    """Test unified resource allocator"""

    @pytest.fixture
    def allocator(self):
        """Create resource allocator"""
        satellites = [Mock(spec=Satellite) for _ in range(3)]
        for i, sat in enumerate(satellites):
            sat.id = f'SAT-{i+1:02d}'
            sat.satellite_type = SatelliteType.OPTICAL_1

        stations = [Mock(spec=GroundStation) for _ in range(2)]
        for i, gs in enumerate(stations):
            gs.id = f'GS-{i+1:02d}'
            gs.antennas = [Mock(spec=Antenna) for _ in range(2)]
            for j, ant in enumerate(gs.antennas):
                ant.id = f'ANT-{i*2+j+1:02d}'
                ant.ground_station_id = gs.id

        return ResourceAllocator(satellites, stations)

    def test_allocator_initialization(self, allocator):
        """Test allocator initialization"""
        assert allocator.satellite_pool is not None
        assert allocator.ground_station_pool is not None

    def test_allocate_task_resources(self, allocator):
        """Test allocating resources for a task (imaging only)"""
        task = Mock(spec=Target)
        task.id = 'TASK-001'
        task.priority = 1
        task.target_type = 'point'

        imaging_start = datetime.now()
        imaging_end = imaging_start + timedelta(minutes=5)

        allocation = allocator.allocate_task_resources(
            task=task,
            imaging_window=(imaging_start, imaging_end),
            satellite_requirements={'type': 'optical'}
        )

        assert allocation is not None
        assert allocation.satellite_id is not None
        # Without downlink_window, antenna is not allocated (correct behavior)

    def test_allocate_with_downlink(self, allocator):
        """Test allocation with data downlink requirement"""
        task = Mock(spec=Target)
        task.id = 'TASK-002'
        task.priority = 1
        task.data_size = 100.0  # GB

        imaging_start = datetime.now()
        imaging_end = imaging_start + timedelta(minutes=5)
        downlink_start = imaging_end + timedelta(minutes=2)
        downlink_end = downlink_start + timedelta(minutes=10)

        allocation = allocator.allocate_task_resources(
            task=task,
            imaging_window=(imaging_start, imaging_end),
            downlink_window=(downlink_start, downlink_end),
            satellite_requirements={'type': 'optical'}
        )

        assert allocation is not None
        assert allocation.downlink_station_id is not None

    def test_release_task_resources(self, allocator):
        """Test releasing task resources"""
        task = Mock(spec=Target)
        task.id = 'TASK-003'

        imaging_start = datetime.now()
        imaging_end = imaging_start + timedelta(minutes=5)

        allocation = allocator.allocate_task_resources(
            task=task,
            imaging_window=(imaging_start, imaging_end)
        )

        # Release
        allocator.release_task_resources(task.id, allocation)

        # Resources should be available again
        assert allocator.satellite_pool.get_available_count() == 3

    def test_get_resource_utilization(self, allocator):
        """Test getting resource utilization statistics"""
        stats = allocator.get_resource_utilization()

        assert 'satellite_utilization' in stats
        assert 'antenna_utilization' in stats
        assert stats['total_satellites'] == 3
        assert stats['total_antennas'] == 4
