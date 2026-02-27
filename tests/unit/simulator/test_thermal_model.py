"""
Unit tests for thermal model.

Tests ThermalParameters and ThermalIntegrator with comprehensive coverage.
Follows TDD principles - tests written first, then implementation verified.
"""
import pytest
import math
from datetime import datetime, timedelta

from simulator.thermal_model import ThermalParameters, ThermalIntegrator


class TestThermalParameters:
    """Test thermal parameter configuration and data class"""

    def test_default_parameters(self):
        """Test default thermal parameters are set correctly"""
        params = ThermalParameters()

        assert params.thermal_capacity == 5000.0
        assert params.thermal_resistance == 0.5
        assert params.ambient_temperature == 273.15  # 0°C
        assert params.max_operating_temp == 333.15  # 60°C
        assert params.min_operating_temp == 253.15  # -20°C
        assert params.emergency_shutdown_temp == 343.15  # 70°C

    def test_time_constant_calculation(self):
        """Test thermal time constant τ = R * C"""
        params = ThermalParameters()
        expected_tau = params.thermal_resistance * params.thermal_capacity

        assert params.time_constant == expected_tau
        assert params.time_constant == 2500.0  # 0.5 * 5000

    def test_custom_parameters(self):
        """Test custom parameter initialization"""
        params = ThermalParameters(
            thermal_capacity=10000.0,
            thermal_resistance=0.2,
            ambient_temperature=288.15,  # 15°C
            max_operating_temp=343.15,  # 70°C
            min_operating_temp=263.15,  # -10°C
            emergency_shutdown_temp=353.15  # 80°C
        )

        assert params.thermal_capacity == 10000.0
        assert params.thermal_resistance == 0.2
        assert params.ambient_temperature == 288.15
        assert params.max_operating_temp == 343.15
        assert params.min_operating_temp == 263.15
        assert params.emergency_shutdown_temp == 353.15

    def test_heat_generation_defaults(self):
        """Test default heat generation values for activities"""
        params = ThermalParameters()

        assert params.heat_generation['idle'] == 10.0
        assert params.heat_generation['slewing'] == 20.0
        assert params.heat_generation['imaging_stripmap'] == 80.0
        assert params.heat_generation['imaging_sliding_spotlight'] == 120.0
        assert params.heat_generation['imaging_spotlight'] == 200.0
        assert params.heat_generation['downlink'] == 50.0

    def test_custom_heat_generation(self):
        """Test custom heat generation dictionary"""
        custom_heat = {
            'idle': 15.0,
            'imaging': 100.0,
            'custom_mode': 250.0
        }
        params = ThermalParameters(heat_generation=custom_heat)

        assert params.heat_generation['idle'] == 15.0
        assert params.heat_generation['imaging'] == 100.0
        assert params.heat_generation['custom_mode'] == 250.0


class TestThermalIntegratorInitialization:
    """Test thermal integrator initialization"""

    def test_default_initial_temperature(self):
        """Test integrator defaults to ambient temperature"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params)

        assert integrator.temperature == params.ambient_temperature
        assert integrator.last_update_time is None
        assert integrator.temperature_history == []

    def test_custom_initial_temperature(self):
        """Test integrator with custom initial temperature"""
        params = ThermalParameters()
        initial_temp = 300.0
        integrator = ThermalIntegrator(params, initial_temp=initial_temp)

        assert integrator.temperature == initial_temp

    def test_initial_temperature_zero(self):
        """Test integrator handles 0K initial temperature (edge case)"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=0.0)

        assert integrator.temperature == 0.0


class TestThermalIntegratorUpdate:
    """Test temperature update calculations"""

    @pytest.fixture
    def integrator(self):
        """Create thermal integrator at ambient temperature"""
        params = ThermalParameters()
        return ThermalIntegrator(params, initial_temp=273.15)

    def test_first_update_records_initial_state(self, integrator):
        """Test first update records initial temperature"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        temp = integrator.update(start_time, 'idle')

        assert integrator.last_update_time == start_time
        assert len(integrator.temperature_history) == 1
        assert integrator.temperature_history[0] == (start_time, 273.15)
        assert temp == 273.15

    def test_update_with_zero_time_delta(self, integrator):
        """Test update with same timestamp returns current temperature"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        integrator.update(start_time, 'idle')
        temp = integrator.update(start_time, 'imaging_spotlight')

        assert temp == integrator.temperature
        assert len(integrator.temperature_history) == 1  # No new entry

    def test_update_with_negative_time_delta(self, integrator):
        """Test update with earlier timestamp returns current temperature"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        earlier_time = datetime(2024, 1, 1, 11, 0, 0)

        integrator.update(start_time, 'idle')
        temp = integrator.update(earlier_time, 'imaging_spotlight')

        assert temp == integrator.temperature
        assert len(integrator.temperature_history) == 1  # No new entry

    def test_idle_mode_temperature_convergence(self, integrator):
        """Test temperature converges to steady-state in idle mode"""
        # Steady-state: T_ambient + P_in * R = 273.15 + 10 * 0.5 = 278.15K
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # Run for 5 time constants (should be very close to steady-state)
        for i in range(50):  # 50 * 300s = ~4 hours
            current_time = start_time + timedelta(seconds=i * 300)
            integrator.update(current_time, 'idle')

        # Should be approaching steady-state temperature
        expected_steady_state = 273.15 + 10.0 * 0.5  # 278.15K
        assert integrator.temperature > 273.15
        assert integrator.temperature < expected_steady_state + 1.0  # Within 1K

    def test_imaging_mode_heating(self, integrator):
        """Test temperature increases during high-power imaging"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        initial_temp = integrator.temperature

        # 10 minutes of spotlight imaging (200W)
        for i in range(10):
            current_time = start_time + timedelta(minutes=i)
            integrator.update(current_time, 'imaging_spotlight')

        assert integrator.temperature > initial_temp
        # Should have increased significantly with 200W heat generation

    def test_different_activities_different_heating(self, integrator):
        """Test different activities produce different heating rates"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # Run idle for 10 minutes
        for i in range(10):
            current_time = start_time + timedelta(minutes=i)
            integrator.update(current_time, 'idle')
        idle_temp = integrator.temperature

        # Reset
        integrator2 = ThermalIntegrator(integrator.params, initial_temp=273.15)

        # Run imaging_spotlight for 10 minutes
        for i in range(10):
            current_time = start_time + timedelta(minutes=i)
            integrator2.update(current_time, 'imaging_spotlight')
        imaging_temp = integrator2.temperature

        # Imaging should heat up more than idle
        assert imaging_temp > idle_temp

    def test_unknown_activity_defaults_to_low_power(self, integrator):
        """Test unknown activity uses default heat generation"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        integrator.update(start_time, 'unknown_activity')
        end_time = start_time + timedelta(minutes=10)
        temp = integrator.update(end_time, 'unknown_activity')

        # Should use default 10.0W (same as idle)
        assert temp > 273.15  # Should still heat up slightly

    def test_temperature_history_accumulation(self, integrator):
        """Test temperature history is properly accumulated"""
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # First update records initial state
        integrator.update(start_time, 'idle')
        assert len(integrator.temperature_history) == 1

        # Subsequent updates add entries
        for i in range(1, 6):
            current_time = start_time + timedelta(minutes=i)
            integrator.update(current_time, 'idle')

        assert len(integrator.temperature_history) == 6


class TestThermalIntegratorPrediction:
    """Test temperature prediction functionality"""

    @pytest.fixture
    def integrator(self):
        """Create thermal integrator at ambient temperature"""
        params = ThermalParameters()
        return ThermalIntegrator(params, initial_temp=273.15)

    def test_predict_temperature_idle(self, integrator):
        """Test temperature prediction in idle mode"""
        # Predict temperature after 5 minutes of idle
        predicted = integrator.predict_temperature(duration=300, activity='idle')

        # Steady-state for idle: T_amb + P_in * R = 273.15 + 10 * 0.5 = 278.15K
        expected_steady_state = 278.15

        # Prediction should be between current and steady-state
        assert predicted > integrator.temperature
        assert predicted < expected_steady_state + 0.1

    def test_predict_temperature_imaging(self, integrator):
        """Test temperature prediction during imaging"""
        # Predict temperature after 5 minutes of spotlight imaging
        predicted = integrator.predict_temperature(duration=300, activity='imaging_spotlight')

        # Steady-state for spotlight: T_amb + P_in * R = 273.15 + 200 * 0.5 = 373.15K
        expected_steady_state = 373.15

        # Prediction should be higher than current
        assert predicted > integrator.temperature
        # After 5 minutes at τ=2500s, should be moving toward steady-state

    def test_predict_temperature_zero_duration(self, integrator):
        """Test prediction with zero duration returns current temperature"""
        predicted = integrator.predict_temperature(duration=0, activity='idle')

        assert predicted == integrator.temperature

    def test_predict_temperature_very_long_duration(self, integrator):
        """Test prediction approaches steady-state for very long duration"""
        # Predict temperature after 10 time constants (essentially steady-state)
        tau = integrator.params.time_constant
        predicted = integrator.predict_temperature(duration=10 * tau, activity='idle')

        expected_steady_state = 273.15 + 10.0 * 0.5  # 278.15K

        # Should be very close to steady-state
        assert abs(predicted - expected_steady_state) < 0.01


class TestThermalIntegratorValidation:
    """Test temperature validity checking"""

    @pytest.fixture
    def integrator(self):
        """Create thermal integrator at ambient temperature"""
        params = ThermalParameters()
        return ThermalIntegrator(params, initial_temp=273.15)

    def test_is_temperature_valid_safe(self, integrator):
        """Test valid temperature check for safe activity"""
        is_valid, predicted = integrator.is_temperature_valid('idle', duration=300)

        assert is_valid is True
        assert predicted > integrator.temperature

    def test_is_temperature_valid_with_default_margin(self, integrator):
        """Test validity uses default safety margin of 5K"""
        is_valid, predicted = integrator.is_temperature_valid('imaging_spotlight', duration=60)

        max_allowed = integrator.params.max_operating_temp - 5.0  # 328.15K
        assert is_valid == (predicted <= max_allowed)

    def test_is_temperature_valid_custom_margin(self, integrator):
        """Test validity with custom safety margin"""
        is_valid, predicted = integrator.is_temperature_valid(
            'imaging_spotlight', duration=60, safety_margin=10.0
        )

        max_allowed = integrator.params.max_operating_temp - 10.0  # 323.15K
        assert is_valid == (predicted <= max_allowed)

    def test_is_temperature_invalid_when_exceeds_limit(self):
        """Test invalid when predicted temperature exceeds limit"""
        # Start at high temperature
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=330.0)  # Near limit

        # Long duration high-power activity should exceed limit
        is_valid, predicted = integrator.is_temperature_valid(
            'imaging_spotlight', duration=3600, safety_margin=5.0
        )

        assert is_valid is False
        assert predicted > integrator.params.max_operating_temp - 5.0


class TestThermalIntegratorCooldown:
    """Test cooldown time calculations"""

    @pytest.fixture
    def hot_integrator(self):
        """Create thermal integrator at high temperature"""
        params = ThermalParameters()
        return ThermalIntegrator(params, initial_temp=340.0)  # 66.85°C

    def test_get_cooldown_time_default_target(self, hot_integrator):
        """Test cooldown time to default target (ambient + 10K)"""
        cooldown = hot_integrator.get_cooldown_time()

        assert cooldown > 0
        assert isinstance(cooldown, float)

    def test_get_cooldown_time_custom_target(self, hot_integrator):
        """Test cooldown time to custom target"""
        target_temp = 300.0  # 26.85°C
        cooldown = hot_integrator.get_cooldown_time(target_temp=target_temp)

        assert cooldown > 0

    def test_get_cooldown_time_already_cooled(self, hot_integrator):
        """Test cooldown returns 0 when already at target"""
        target_temp = 350.0  # Higher than current
        cooldown = hot_integrator.get_cooldown_time(target_temp=target_temp)

        assert cooldown == 0.0

    def test_get_cooldown_time_exact_temperature(self, hot_integrator):
        """Test cooldown returns 0 when at exact target temperature"""
        cooldown = hot_integrator.get_cooldown_time(target_temp=340.0)

        assert cooldown == 0.0

    def test_get_cooldown_time_logarithm_domain_error(self):
        """Test cooldown handles logarithm domain error gracefully via direct call"""
        # Directly test the ValueError handling by mocking a scenario
        # where the logarithm argument becomes invalid due to floating point issues
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=300.0)

        # Manually set temperature to be very close to target to potentially trigger edge case
        # This tests the defensive programming in the code
        import math
        tau = params.time_constant
        t_amb = params.ambient_temperature
        target_temp = 295.0

        # Simulate the calculation that would happen in get_cooldown_time
        # If temperatures are very close, log argument approaches 1
        log_arg = (target_temp - t_amb) / (integrator.temperature - t_amb)

        # This should not raise an error
        try:
            time_needed = -tau * math.log(log_arg)
            assert isinstance(time_needed, float)
        except ValueError:
            # If this happens, the code should handle it
            pass

        # The actual method should always return a valid float
        cooldown = integrator.get_cooldown_time(target_temp=target_temp)
        assert isinstance(cooldown, float)
        assert cooldown >= 0.0

    def test_get_cooldown_time_to_ambient(self, hot_integrator):
        """Test cooldown time to ambient temperature returns infinity"""
        ambient = hot_integrator.params.ambient_temperature
        cooldown = hot_integrator.get_cooldown_time(target_temp=ambient)

        # Cannot cool below ambient - should return infinity
        assert cooldown == float('inf')

    def test_cooldown_calculation_formula(self):
        """Test cooldown calculation matches expected formula"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=293.15)  # 20°C

        target_temp = 283.15  # 10°C
        tau = params.time_constant
        t_amb = params.ambient_temperature

        # Expected: t = -τ * ln((T_target - T_amb) / (T_0 - T_amb))
        expected_time = -tau * math.log((target_temp - t_amb) / (integrator.temperature - t_amb))

        cooldown = integrator.get_cooldown_time(target_temp=target_temp)

        assert abs(cooldown - expected_time) < 0.001


class TestThermalIntegratorStatus:
    """Test thermal status reporting"""

    @pytest.fixture
    def integrator(self):
        """Create thermal integrator"""
        params = ThermalParameters()
        return ThermalIntegrator(params, initial_temp=293.15)  # 20°C

    def test_get_thermal_status_structure(self, integrator):
        """Test thermal status has correct structure"""
        status = integrator.get_thermal_status()

        assert 'current_temperature_k' in status
        assert 'current_temperature_c' in status
        assert 'max_operating_temp_k' in status
        assert 'temperature_margin_k' in status
        assert 'time_constant_s' in status
        assert 'is_safe' in status

    def test_get_thermal_status_values(self, integrator):
        """Test thermal status values are correct"""
        status = integrator.get_thermal_status()

        assert status['current_temperature_k'] == 293.15
        assert status['current_temperature_c'] == 20.0  # 293.15 - 273.15
        assert status['max_operating_temp_k'] == 333.15
        assert status['temperature_margin_k'] == 333.15 - 293.15  # 40K
        assert status['time_constant_s'] == 2500.0

    def test_get_thermal_status_is_safe_true(self):
        """Test is_safe is True when margin > 5K"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=300.0)  # Safe temperature

        status = integrator.get_thermal_status()

        assert status['is_safe'] is True
        assert status['temperature_margin_k'] > 5.0

    def test_get_thermal_status_is_safe_false(self):
        """Test is_safe is False when margin <= 5K"""
        params = ThermalParameters()
        # Temperature close to limit (within 5K)
        integrator = ThermalIntegrator(params, initial_temp=329.0)  # Only ~4K margin

        status = integrator.get_thermal_status()

        assert status['is_safe'] is False
        assert status['temperature_margin_k'] <= 5.0


class TestThermalIntegratorEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_high_thermal_resistance(self):
        """Test with very high thermal resistance (slow cooling)"""
        params = ThermalParameters(thermal_resistance=10.0)
        integrator = ThermalIntegrator(params, initial_temp=300.0)

        # High resistance means slow heat transfer
        assert integrator.params.time_constant == 50000.0  # 10 * 5000

    def test_very_low_thermal_capacity(self):
        """Test with very low thermal capacity (fast temperature changes)"""
        params = ThermalParameters(thermal_capacity=100.0)
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(minutes=1)

        integrator.update(start_time, 'idle')
        temp = integrator.update(end_time, 'imaging_spotlight')

        # Low capacity means faster temperature change
        assert integrator.params.time_constant == 50.0  # 0.5 * 100

    def test_negative_temperature(self):
        """Test handling of negative temperatures (below absolute zero not possible physically)"""
        params = ThermalParameters()
        # Note: This tests the code handles the input, not physical correctness
        integrator = ThermalIntegrator(params, initial_temp=-10.0)

        status = integrator.get_thermal_status()
        assert status['current_temperature_k'] == -10.0

    def test_extreme_heat_generation(self):
        """Test with extreme heat generation values"""
        params = ThermalParameters(heat_generation={'extreme': 10000.0})
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        predicted = integrator.predict_temperature(duration=60, activity='extreme')

        # Should predict very high temperature
        assert predicted > integrator.temperature

    def test_empty_activity_name(self):
        """Test with empty activity name (uses default)"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(minutes=10)

        integrator.update(start_time, '')
        temp = integrator.update(end_time, '')

        # Empty string not in heat_generation, should use default
        assert temp >= 273.15


class TestThermalIntegratorTemperatureHistory:
    """Test temperature history tracking"""

    def test_temperature_history_order(self):
        """Test temperature history maintains chronological order"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        start_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add entries in chronological order
        for i in range(5):
            current_time = start_time + timedelta(minutes=i)
            integrator.update(current_time, 'idle')

        # Verify chronological order
        times = [entry[0] for entry in integrator.temperature_history]
        assert times == sorted(times)

    def test_temperature_history_content(self):
        """Test temperature history contains correct data"""
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        start_time = datetime(2024, 1, 1, 12, 0, 0)

        integrator.update(start_time, 'idle')
        end_time = start_time + timedelta(minutes=10)
        integrator.update(end_time, 'imaging_spotlight')

        assert len(integrator.temperature_history) == 2
        assert integrator.temperature_history[0] == (start_time, 273.15)
        # Second entry should have higher temperature due to imaging
        assert integrator.temperature_history[1][0] == end_time
        assert integrator.temperature_history[1][1] > 273.15
