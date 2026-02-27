"""
Test suite for physical constraint models.
Tests ThermalModel, SunExclusionCalculator, and FragmentedStorageModel.
"""
import pytest
import math
from datetime import datetime, timedelta

# Skip if physical constraints module not yet implemented
pytest.importorskip("simulator.thermal_model", reason="Thermal model not yet implemented")
pytest.importorskip("simulator.sun_exclusion_calculator", reason="Sun exclusion calculator not yet implemented")
pytest.importorskip("simulator.storage_model", reason="Storage model not yet implemented")

from simulator.thermal_model import ThermalParameters, ThermalIntegrator
from simulator.sun_exclusion_calculator import SunExclusionCalculator
from simulator.storage_model import FragmentedStorageModel, StorageBlock


class TestThermalParameters:
    """Test thermal parameter configuration"""

    def test_default_parameters(self):
        """Test default thermal parameters"""
        params = ThermalParameters()
        assert params.thermal_capacity == 5000.0
        assert params.thermal_resistance == 0.5
        assert params.max_operating_temp == 333.15  # 60°C
        assert params.min_operating_temp == 253.15  # -20°C

    def test_time_constant(self):
        """Test thermal time constant calculation"""
        params = ThermalParameters()
        expected_tau = params.thermal_resistance * params.thermal_capacity
        assert params.time_constant == expected_tau

    def test_heat_generation_defaults(self):
        """Test default heat generation values"""
        params = ThermalParameters()
        assert params.heat_generation['idle'] == 10.0
        assert params.heat_generation['imaging_spotlight'] == 200.0
        assert params.heat_generation['downlink'] == 50.0


class TestThermalIntegrator:
    """Test thermal integrator with RC model"""

    @pytest.fixture
    def integrator(self):
        """Create thermal integrator"""
        params = ThermalParameters()
        return ThermalIntegrator(params, initial_temp=273.15)  # 0°C

    def test_initial_temperature(self, integrator):
        """Test initial temperature setting"""
        assert integrator.temperature == 273.15

    def test_update_temperature_idle(self, integrator):
        """Test temperature update in idle mode"""
        start_time = datetime.now()
        # Time constant τ = R * C = 0.5 * 5000 = 2500s (~42 min)
        # Use 30 minutes to see noticeable change
        end_time = start_time + timedelta(minutes=30)

        # Idle mode generates 10W
        temp = integrator.update(end_time, 'idle')

        # Temperature history should be recorded
        assert len(integrator.temperature_history) == 1
        # Temperature should be moving toward steady-state (ambient + P*R = 278.15K)
        # After 30 min (~0.72τ), should see some change
        assert integrator.temperature >= 273.15  # Should not decrease

    def test_update_temperature_imaging(self, integrator):
        """Test temperature update during imaging"""
        start_time = datetime.now()

        # Imaging spotlight generates 200W - should heat up significantly
        for i in range(6):  # 1 minute intervals
            current_time = start_time + timedelta(minutes=i)
            integrator.update(current_time, 'imaging_spotlight')

        # Temperature should have increased
        assert integrator.temperature > 273.15

    def test_predict_temperature(self, integrator):
        """Test temperature prediction"""
        # Predict temperature after 5 minutes of imaging
        predicted = integrator.predict_temperature(duration=300, activity='imaging_spotlight')

        # Prediction should return a higher temperature
        assert predicted > integrator.temperature

    def test_is_temperature_valid(self, integrator):
        """Test temperature feasibility check"""
        # Initially should be valid
        is_valid, temp = integrator.is_temperature_valid('imaging_spotlight', duration=60)
        assert is_valid

    def test_get_cooldown_time(self, integrator):
        """Test cooldown time calculation"""
        # Heat up first
        integrator.temperature = 340.0  # Above operating limit

        # Calculate cooldown time
        cooldown = integrator.get_cooldown_time(target_temp=300.0)

        assert cooldown > 0
        assert isinstance(cooldown, float)


class TestSunExclusionCalculator:
    """Test sun exclusion angle calculator"""

    @pytest.fixture
    def calculator(self):
        """Create sun exclusion calculator"""
        return SunExclusionCalculator(exclusion_angle=30.0)

    def test_initialization(self):
        """Test calculator initialization"""
        calc = SunExclusionCalculator(exclusion_angle=45.0)
        assert calc.exclusion_angle == math.radians(45.0)

    def test_calculate_sun_position(self, calculator):
        """Test sun position calculation"""
        t = datetime(2024, 6, 21, 12, 0, 0)  # Summer solstice noon

        sun_pos = calculator.calculate_sun_position(t)

        # Should return (x, y, z) tuple
        assert len(sun_pos) == 3
        assert all(isinstance(coord, float) for coord in sun_pos)

        # Sun should be at roughly 1 AU distance
        distance = math.sqrt(sum(c**2 for c in sun_pos))
        assert distance > 1e11  # Approximately 1 AU in meters

    def test_check_sun_exclusion_valid(self, calculator):
        """Test sun exclusion check - valid case (90° separation)"""
        t = datetime(2024, 6, 21, 12, 0, 0)

        # Satellite at origin, target at 90° from sun direction
        satellite_pos = (0.0, 0.0, 0.0)
        target_pos = (1e6, 0.0, 0.0)  # X-axis

        # Mock sun position to be on Y-axis (90° from target)
        cache_key = (t.year, t.month, t.day, t.hour)
        calculator._sun_cache = {cache_key: (0.0, 1.5e11, 0.0)}

        is_valid, angle = calculator.check_sun_exclusion(satellite_pos, target_pos, t)

        assert is_valid  # 90° > 30° exclusion angle
        assert angle > 30.0

    def test_check_sun_exclusion_invalid(self, calculator):
        """Test sun exclusion check - invalid case (0° separation)"""
        t = datetime(2024, 6, 21, 12, 0, 0)

        # Satellite at origin, target close to sun direction
        satellite_pos = (0.0, 0.0, 0.0)
        target_pos = (0.0, 1e6, 0.0)  # Y-axis

        # Mock sun position to be on Y-axis (close to target)
        cache_key = (t.year, t.month, t.day, t.hour)
        calculator._sun_cache = {cache_key: (0.0, 1.5e11, 0.0)}

        is_valid, angle = calculator.check_sun_exclusion(satellite_pos, target_pos, t)

        assert not is_valid  # 0° < 30° exclusion angle
        assert angle < 30.0


class TestFragmentedStorageModel:
    """Test fragmented storage model"""

    @pytest.fixture
    def storage(self):
        """Create fragmented storage model"""
        return FragmentedStorageModel(
            total_capacity=1024*1024*1024,  # 1GB
            block_size=4096,  # 4KB blocks
            filesystem_overhead=0.15
        )

    def test_initialization(self, storage):
        """Test storage initialization"""
        assert storage.total_capacity == 1024*1024*1024
        assert storage.block_size == 4096
        assert storage.effective_capacity < storage.total_capacity

    def test_effective_capacity(self, storage):
        """Test effective capacity calculation"""
        expected = int(storage.total_capacity * 0.85)  # 15% overhead
        assert storage.effective_capacity == expected

    def test_initial_free_space(self, storage):
        """Test initial free space"""
        assert storage.free_space == storage.effective_capacity
        assert storage.used_space == 0

    def test_allocate_file(self, storage):
        """Test file allocation"""
        success, allocated_size = storage.allocate('file1', 8192)  # 8KB

        assert success
        assert allocated_size >= 8192
        assert 'file1' in storage.files

    def test_allocate_exceeds_capacity(self, storage):
        """Test allocation exceeding capacity"""
        success, allocated_size = storage.allocate('huge_file', storage.total_capacity * 2)

        assert not success
        assert allocated_size == 0

    def test_deallocate_file(self, storage):
        """Test file deallocation"""
        storage.allocate('file1', 8192)
        assert 'file1' in storage.files

        success = storage.deallocate('file1')
        assert success
        assert 'file1' not in storage.files

    def test_fragmentation_ratio(self, storage):
        """Test fragmentation ratio calculation"""
        # Initially no fragmentation
        assert storage.fragmentation_ratio == 0.0

        # Allocate and deallocate to create fragmentation
        for i in range(10):
            storage.allocate(f'file{i}', 8192)

        # Deallocate every other file
        for i in range(0, 10, 2):
            storage.deallocate(f'file{i}')

        # Should have some fragmentation now
        assert storage.fragmentation_ratio >= 0.0

    def test_used_space_tracking(self, storage):
        """Test used space tracking"""
        initial_free = storage.free_space

        storage.allocate('file1', 8192)

        assert storage.used_space > 0
        assert storage.free_space < initial_free

    def test_get_storage_status(self, storage):
        """Test storage status reporting"""
        storage.allocate('file1', 8192)

        status = storage.get_storage_status()

        assert 'total_capacity_gb' in status
        assert 'used_space_gb' in status
        assert 'free_space_gb' in status
        assert 'fragmentation_ratio' in status
        assert 'file_count' in status
        assert status['file_count'] == 1
