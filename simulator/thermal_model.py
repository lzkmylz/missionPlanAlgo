"""
Thermal model for satellite thermal constraint simulation.

Implements first-order RC thermal damping model for SAR satellite
temperature tracking and prediction.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThermalParameters:
    """Satellite thermal control parameters - First-order RC thermal damping model

    The thermal model follows:
        dT/dt = (P_in * R - (T - T_ambient)) / τ
    where τ = R * C is the thermal time constant

    Attributes:
        thermal_capacity: Thermal capacity (J/K)
        thermal_resistance: Thermal resistance (K/W)
        ambient_temperature: Ambient temperature (K), default 0°C
        max_operating_temp: Maximum operating temperature (K), default 60°C
        min_operating_temp: Minimum operating temperature (K), default -20°C
        emergency_shutdown_temp: Emergency shutdown temperature (K), default 70°C
        heat_generation: Heat generation per activity mode (W)
    """
    thermal_capacity: float = 5000.0  # J/K
    thermal_resistance: float = 0.5   # K/W
    ambient_temperature: float = 273.15  # 0°C in Kelvin

    # Temperature limits
    max_operating_temp: float = 333.15   # 60°C
    min_operating_temp: float = 253.15   # -20°C
    emergency_shutdown_temp: float = 343.15  # 70°C

    # Heat generation per activity mode (W)
    heat_generation: Dict[str, float] = field(default_factory=lambda: {
        'idle': 10.0,
        'slewing': 20.0,
        'imaging_stripmap': 80.0,
        'imaging_sliding_spotlight': 120.0,
        'imaging_spotlight': 200.0,
        'downlink': 50.0,
    })

    @property
    def time_constant(self) -> float:
        """Thermal time constant τ = R * C (seconds)"""
        return self.thermal_resistance * self.thermal_capacity


class ThermalIntegrator:
    """Thermal integrator implementing first-order RC thermal damping model

    Tracks satellite temperature evolution over time based on:
    - Current activity mode (determines heat input)
    - Thermal parameters (capacity, resistance)
    - Ambient conditions

    Example:
        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        # Update temperature during imaging
        for minute in range(10):
            t = start_time + timedelta(minutes=minute)
            temp = integrator.update(t, 'imaging_spotlight')

        # Check if future activity is feasible
        is_valid, predicted = integrator.is_temperature_valid(
            'imaging_spotlight', duration=300
        )
    """

    def __init__(self, params: ThermalParameters, initial_temp: float = None):
        """Initialize thermal integrator

        Args:
            params: Thermal parameters
            initial_temp: Initial temperature in Kelvin (default: ambient)
        """
        self.params = params
        self.temperature = initial_temp if initial_temp is not None else params.ambient_temperature
        self.last_update_time: Optional[datetime] = None
        self.temperature_history: List[Tuple[datetime, float]] = []

    def update(self, current_time: datetime, activity: str) -> float:
        """Update temperature state

        Implements discrete-time thermal model using Euler integration:
            T_new = T_old + dt * (P_in * R - (T_old - T_amb)) / τ

        Args:
            current_time: Current simulation time
            activity: Activity mode (determines heat generation)

        Returns:
            Updated temperature in Kelvin
        """
        if self.last_update_time is None:
            self.last_update_time = current_time
            self.temperature_history.append((current_time, self.temperature))
            return self.temperature

        # Calculate time step
        dt = (current_time - self.last_update_time).total_seconds()
        if dt <= 0:
            return self.temperature

        # Get heat generation for activity
        power_in = self.params.heat_generation.get(activity, 10.0)

        # First-order RC model (Euler discretization)
        tau = self.params.time_constant
        r = self.params.thermal_resistance
        t_amb = self.params.ambient_temperature

        # dT/dt = (P_in * R - (T - T_amb)) / τ
        temperature_change = dt * (power_in * r - (self.temperature - t_amb)) / tau
        self.temperature += temperature_change

        # Record history
        self.temperature_history.append((current_time, self.temperature))
        self.last_update_time = current_time

        # Log warning if approaching limits
        if self.temperature > self.params.max_operating_temp:
            logger.warning(
                f"Temperature {self.temperature:.2f}K exceeds max operating "
                f"temp {self.params.max_operating_temp:.2f}K"
            )

        return self.temperature

    def predict_temperature(self, duration: float, activity: str) -> float:
        """Predict temperature after performing activity

        Uses analytical solution:
            T(t) = T_ss + (T_0 - T_ss) * exp(-t/τ)
        where T_ss = T_ambient + P_in * R is the steady-state temperature

        Args:
            duration: Activity duration in seconds
            activity: Activity mode

        Returns:
            Predicted temperature in Kelvin
        """
        power_in = self.params.heat_generation.get(activity, 10.0)
        t_ss = self.params.ambient_temperature + power_in * self.params.thermal_resistance

        tau = self.params.time_constant
        temp_final = t_ss + (self.temperature - t_ss) * math.exp(-duration / tau)

        return temp_final

    def is_temperature_valid(self, activity: str, duration: float,
                            safety_margin: float = 5.0) -> Tuple[bool, float]:
        """Check if activity can be performed without exceeding temperature limits

        Args:
            activity: Activity mode
            duration: Activity duration in seconds
            safety_margin: Safety margin below max temperature (K)

        Returns:
            Tuple of (is_valid, predicted_temperature)
        """
        predicted_temp = self.predict_temperature(duration, activity)
        max_allowed = self.params.max_operating_temp - safety_margin

        is_valid = predicted_temp <= max_allowed

        return is_valid, predicted_temp

    def get_cooldown_time(self, target_temp: float = None) -> float:
        """Calculate time needed to cool down to target temperature

        Uses analytical solution of cooling phase (P_in = 0):
            T(t) = T_amb + (T_0 - T_amb) * exp(-t/τ)
            t = -τ * ln((T_target - T_amb) / (T_0 - T_amb))

        Args:
            target_temp: Target temperature (default: ambient + 10K)

        Returns:
            Cooldown time in seconds
        """
        if target_temp is None:
            target_temp = self.params.ambient_temperature + 10.0

        tau = self.params.time_constant
        t_amb = self.params.ambient_temperature

        # Check if already at or below target
        if self.temperature <= target_temp:
            return 0.0

        # Cannot cool below ambient - return infinity equivalent
        if target_temp <= t_amb:
            return float('inf')

        # Time to cool from T_0 to T_target
        try:
            time_needed = -tau * math.log(
                (target_temp - t_amb) / (self.temperature - t_amb)
            )
        except ValueError:
            # Logarithm domain error - temperatures too close
            return 0.0

        return max(0.0, time_needed)

    def get_thermal_status(self) -> Dict[str, float]:
        """Get current thermal status

        Returns:
            Dictionary with thermal status information
        """
        temp_margin = self.params.max_operating_temp - self.temperature

        return {
            'current_temperature_k': self.temperature,
            'current_temperature_c': self.temperature - 273.15,
            'max_operating_temp_k': self.params.max_operating_temp,
            'temperature_margin_k': temp_margin,
            'time_constant_s': self.params.time_constant,
            'is_safe': temp_margin > 5.0  # 5K safety margin
        }
