"""Unified constraint checking for satellite schedulers.

This module consolidates all constraint checking functionality,
eliminating duplication across scheduler implementations.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

from core.models import Mission, Satellite, Target
from core.dynamics.slew_calculator import SlewCalculator
from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType
from scheduler.common.config import ConstraintConfig


class ConstraintType(Enum):
    """Types of constraints that can be checked."""
    TEMPORAL = "temporal"           # Time window conflicts
    SLEW = "slew"                  # Slew angle and time
    SAA = "saa"                    # South Atlantic Anomaly
    POWER = "power"                # Power capacity
    STORAGE = "storage"            # Storage capacity
    ATTITUDE = "attitude"          # Attitude mode transitions
    CAPABILITY = "capability"      # Satellite capability mismatch
    WINDOW_LENGTH = "window_length"  # Window too short for imaging
    TIME_CONFLICT = "time_conflict"  # Conflict with scheduled tasks
    DEADLINE = "deadline"          # Deadline violation
    THERMAL = "thermal"            # Thermal control constraint
    SUN_EXCLUSION = "sun_exclusion"  # Sun exclusion angle violation
    SOLAR_ELEVATION = "solar_elevation"  # Target solar elevation angle (lighting condition)


@dataclass
class ConstraintResult:
    """Result of a constraint check.

    Attributes:
        feasible: Whether all checked constraints are satisfied
        violations: List of constraint types that failed
        details: Detailed information about each constraint check
        slew_angle: Calculated slew angle (if applicable)
        slew_time: Calculated slew time (if applicable)
        actual_start: Actual feasible start time (if different from requested)
        reset_time: Attitude reset time in seconds (if applicable)
    """
    feasible: bool = True
    violations: List[ConstraintType] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    slew_angle: float = 0.0
    slew_time: float = 0.0
    actual_start: Optional[datetime] = None
    reason: Optional[str] = None
    reset_time: Optional[float] = None  # 姿态复位时间（秒）

    def add_violation(self, constraint_type: ConstraintType, reason: str = "") -> None:
        """Add a constraint violation."""
        self.feasible = False
        self.violations.append(constraint_type)
        self.details[constraint_type.value] = reason


@dataclass
class ConstraintContext:
    """Context for constraint checking.

    This encapsulates all the information needed to check constraints
    for a specific task assignment.
    """
    satellite: Satellite
    target: Target
    window_start: datetime
    window_end: datetime
    prev_target: Optional[Target] = None
    prev_end_time: Optional[datetime] = None
    imaging_duration: Optional[float] = None
    current_power: float = 0.0
    current_storage: float = 0.0
    scheduled_tasks: List[Dict[str, Any]] = field(default_factory=list)
    imaging_mode: Optional[Any] = None
    deadline: Optional[datetime] = None


class SlewChecker:
    """Slew constraint checking component."""

    def __init__(self):
        self._calculators: Dict[str, SlewCalculator] = {}

    def initialize_satellite(self, satellite: Satellite) -> None:
        """Initialize slew calculator for a satellite."""
        agility = getattr(satellite.capabilities, 'agility', {}) or {}
        self._calculators[satellite.id] = SlewCalculator(
            max_slew_rate=agility.get('max_slew_rate', 3.0),
            max_slew_angle=satellite.capabilities.max_off_nadir,
            settling_time=agility.get('settling_time', 5.0)
        )

    def check_slew(
        self,
        sat_id: str,
        prev_target: Optional[Target],
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime,
        imaging_duration: float,
        use_simplified: bool = False,
    ) -> ConstraintResult:
        """Check slew feasibility.

        Args:
            sat_id: Satellite ID
            prev_target: Previous target (None if first task)
            current_target: Current target
            prev_end_time: End time of previous task
            window_start: Window start time
            imaging_duration: Required imaging duration
            use_simplified: Use simplified calculation

        Returns:
            ConstraintResult with slew information
        """
        result = ConstraintResult()
        calculator = self._calculators.get(sat_id)

        if not calculator:
            result.add_violation(ConstraintType.SLEW, "No slew calculator available")
            return result

        if prev_target is None:
            # First task for this satellite: calculate from nadir to target
            # For simplified model, use a reasonable default slew angle
            slew_angle = self._calculate_slew_angle(None, current_target)
            result.slew_angle = slew_angle

            if use_simplified:
                result.slew_time = 5.0 + 2.0 * min(slew_angle, 45.0)
            else:
                slew_time = calculator.calculate_slew_time(slew_angle)
                result.slew_time = slew_time
            result.actual_start = window_start + timedelta(seconds=result.slew_time)
            return result

        if use_simplified:
            # Simplified mode: calculate basic slew angle but use default settling time
            slew_angle = self._calculate_slew_angle(prev_target, current_target)
            result.slew_angle = slew_angle
            # Use a simple linear model for slew time in simplified mode
            # 5 seconds base + 2 seconds per degree
            slew_time = 5.0 + 2.0 * min(slew_angle, 45.0)  # Cap at 45 degrees
            result.slew_time = slew_time
            result.actual_start = window_start
            return result

        # Calculate slew angle using target positions
        slew_angle = self._calculate_slew_angle(prev_target, current_target)
        result.slew_angle = slew_angle

        if slew_angle > calculator.max_slew_angle:
            result.add_violation(
                ConstraintType.SLEW,
                f"Slew angle {slew_angle:.1f}° exceeds max {calculator.max_slew_angle:.1f}°"
            )
            return result

        # Calculate slew time
        slew_time = calculator.calculate_slew_time(slew_angle)
        result.slew_time = slew_time

        # Calculate actual start time
        earliest_start = prev_end_time + timedelta(seconds=slew_time)
        actual_start = max(window_start, earliest_start)

        # Check if imaging fits in window
        imaging_end = actual_start + timedelta(seconds=imaging_duration)
        window_end = window_start + timedelta(seconds=imaging_duration * 2)  # Approximate

        if actual_start > window_start + timedelta(seconds=30):  # Too much slew delay
            result.add_violation(
                ConstraintType.SLEW,
                f"Slew delay too large: {((actual_start - window_start).total_seconds()):.1f}s"
            )
            return result

        result.actual_start = actual_start
        return result

    def _calculate_slew_angle(
        self,
        prev_target: Optional[Target],
        current_target: Target,
    ) -> float:
        """Calculate slew angle between two targets or from nadir to target."""
        # Simplified calculation using lat/lon difference
        # In practice, this should use proper ECEF coordinate transformation

        # Get current target coordinates
        curr_lat = getattr(current_target, 'latitude', 0)
        curr_lon = getattr(current_target, 'longitude', 0)

        # Ensure we have numeric values
        if not isinstance(curr_lat, (int, float)):
            curr_lat = 0
        if not isinstance(curr_lon, (int, float)):
            curr_lon = 0

        if prev_target is None:
            # First task: calculate slew from nadir pointing to target
            # Approximate using target's angular distance from sub-satellite point
            # This is a simplified estimation - actual slew depends on orbital position
            # For now, use a typical off-nadir angle based on target latitude
            # Satellites typically observe targets at 20-45 degree off-nadir angles
            return math.sqrt(curr_lat**2 + curr_lon**2) * 0.5  # Scale factor for typical slew

        # Handle Mock objects gracefully
        prev_lat = getattr(prev_target, 'latitude', 0)
        prev_lon = getattr(prev_target, 'longitude', 0)

        # Ensure we have numeric values
        if not isinstance(prev_lat, (int, float)):
            prev_lat = 0
        if not isinstance(prev_lon, (int, float)):
            prev_lon = 0

        lat_diff = abs(prev_lat - curr_lat)
        lon_diff = abs(prev_lon - curr_lon)

        # Approximate angular separation
        return math.sqrt(lat_diff**2 + lon_diff**2)


class CapabilityChecker:
    """Satellite capability constraint checking component."""

    def check_capability(
        self,
        satellite: Satellite,
        target: Target,
        imaging_mode: Optional[Any] = None
    ) -> ConstraintResult:
        """Check if satellite can perform task.

        Args:
            satellite: Satellite object
            target: Target object
            imaging_mode: Required imaging mode (optional)

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        # Check if satellite has imaging modes
        imaging_modes = getattr(satellite.capabilities, 'imaging_modes', None)
        if not imaging_modes:
            result.add_violation(
                ConstraintType.CAPABILITY,
                "Satellite has no imaging modes"
            )
            return result

        # Check resolution requirement
        required_resolution = getattr(target, 'resolution_required', None)
        if required_resolution is not None:
            sat_resolution = getattr(satellite.capabilities, 'resolution', None)
            try:
                if sat_resolution is not None and sat_resolution > required_resolution:
                    result.add_violation(
                        ConstraintType.CAPABILITY,
                        f"Resolution insufficient: {sat_resolution}m > {required_resolution}m required"
                    )
                    return result
            except TypeError:
                pass

        # Check imaging mode compatibility
        if imaging_mode is not None:
            from core.models import ImagingMode
            if isinstance(imaging_mode, ImagingMode):
                mode_value = imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode)
            else:
                mode_value = str(imaging_mode)

            # Check if mode is supported
            supported = False
            for mode in imaging_modes:
                mode_str = mode.value if hasattr(mode, 'value') else str(mode)
                if mode_str == mode_value:
                    supported = True
                    break

            if not supported:
                result.add_violation(
                    ConstraintType.CAPABILITY,
                    f"Imaging mode {mode_value} not supported"
                )
                return result

        # Check satellite type (optical vs SAR) vs target requirements
        target_type = getattr(target, 'type', None)
        sat_type = getattr(satellite, 'type', None)
        if target_type and sat_type:
            # If target specifies type requirement, check compatibility
            if hasattr(target_type, 'value'):
                target_type = target_type.value
            if hasattr(sat_type, 'value'):
                sat_type = sat_type.value

        return result

    def select_imaging_mode(self, satellite: Satellite, target: Target) -> Any:
        """Select appropriate imaging mode for satellite and target.

        Args:
            satellite: Satellite object
            target: Target object

        Returns:
            Selected ImagingMode
        """
        from core.models import ImagingMode

        imaging_modes = getattr(satellite.capabilities, 'imaging_modes', [])
        if not imaging_modes:
            return ImagingMode.PUSH_BROOM

        mode = imaging_modes[0]
        if hasattr(mode, '_mock_name') or not isinstance(mode, (ImagingMode, str)):
            return ImagingMode.PUSH_BROOM

        return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)


class TemporalChecker:
    """Temporal constraint checking component."""

    def check_temporal(
        self,
        context: ConstraintContext,
        actual_start: datetime,
        actual_end: datetime
    ) -> ConstraintResult:
        """Check temporal constraints.

        Args:
            context: Constraint context
            actual_start: Actual task start time
            actual_end: Actual task end time

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        # Check window boundary
        if actual_end > context.window_end:
            result.add_violation(
                ConstraintType.WINDOW_LENGTH,
                f"Task ends {actual_end} after window end {context.window_end}"
            )
            return result

        # Check deadline
        if context.deadline and actual_end > context.deadline:
            result.add_violation(
                ConstraintType.DEADLINE,
                f"Task ends after deadline: {actual_end} > {context.deadline}"
            )
            return result

        # Check time conflicts with scheduled tasks
        if context.scheduled_tasks:
            for task in context.scheduled_tasks:
                task_start = task.get('start')
                task_end = task.get('end')
                if task_start and task_end:
                    if not (actual_end <= task_start or actual_start >= task_end):
                        result.add_violation(
                            ConstraintType.TIME_CONFLICT,
                            f"Time conflict with task from {task_start} to {task_end}"
                        )
                        return result

        return result


class ThermalChecker:
    """Thermal control constraint checking component.

    Checks if satellite thermal conditions are within acceptable limits
    during imaging operations.
    """

    # Default temperature limits (Celsius)
    DEFAULT_TEMP_MIN = -30.0
    DEFAULT_TEMP_MAX = 60.0

    def __init__(self):
        self._thermal_model = None

    def check_thermal(
        self,
        satellite: Satellite,
        start_time: datetime,
        end_time: datetime,
        imaging_mode: Optional[Any] = None
    ) -> ConstraintResult:
        """Check thermal constraints for imaging operation.

        Args:
            satellite: Satellite object
            start_time: Operation start time
            end_time: Operation end time
            imaging_mode: Imaging mode (affects thermal load)

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        # Get satellite thermal limits
        thermal_limits = getattr(satellite.capabilities, 'thermal_limits', {})
        temp_min = thermal_limits.get('min_temp', self.DEFAULT_TEMP_MIN)
        temp_max = thermal_limits.get('max_temp', self.DEFAULT_TEMP_MAX)

        # Get current temperature if available
        current_temp = getattr(satellite, 'current_temperature', None)

        if current_temp is not None:
            # Check if current temperature is within limits
            if current_temp < temp_min or current_temp > temp_max:
                result.add_violation(
                    ConstraintType.THERMAL,
                    f"Current temperature {current_temp}°C outside limits [{temp_min}, {temp_max}]"
                )
                return result

            # Estimate temperature change during imaging
            # Simplified model: imaging generates heat
            imaging_heat = 5.0  # degrees per hour (simplified)
            if imaging_mode:
                mode_str = str(imaging_mode).lower()
                if 'high' in mode_str or 'video' in mode_str:
                    imaging_heat = 10.0  # High power mode generates more heat

            duration_hours = (end_time - start_time).total_seconds() / 3600
            estimated_temp = current_temp + imaging_heat * duration_hours

            if estimated_temp > temp_max:
                result.add_violation(
                    ConstraintType.THERMAL,
                    f"Estimated temperature {estimated_temp:.1f}°C exceeds max {temp_max}°C"
                )
                return result

        return result


class SunExclusionChecker:
    """Sun exclusion angle constraint checking component.

    Ensures optical satellites don't point too close to the sun,
    which could damage sensors.
    """

    # Default sun exclusion angle (degrees)
    DEFAULT_SUN_EXCLUSION_ANGLE = 30.0

    def __init__(self):
        self._sun_position_cache: Dict[datetime, Tuple[float, float, float]] = {}

    def check_sun_exclusion(
        self,
        satellite: Satellite,
        target: Target,
        imaging_time: datetime,
        sun_position: Optional[Tuple[float, float, float]] = None
    ) -> ConstraintResult:
        """Check sun exclusion angle constraint.

        Args:
            satellite: Satellite object
            target: Target being imaged
            imaging_time: Time of imaging
            sun_position: Pre-computed sun position in ECEF (optional)

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        # Only check for optical satellites
        sat_type = getattr(satellite, 'type', None)
        if sat_type:
            type_str = str(sat_type).lower()
            if 'sar' in type_str or 'radar' in type_str:
                # SAR satellites don't need sun exclusion
                return result

        # Get sun exclusion angle limit
        exclusion_angle = getattr(
            satellite.capabilities,
            'sun_exclusion_angle',
            self.DEFAULT_SUN_EXCLUSION_ANGLE
        )

        # Calculate angle between target and sun
        # Simplified: use target position and sun position
        if sun_position is None:
            sun_position = self._get_sun_position(imaging_time)

        if sun_position is None or not hasattr(target, 'latitude') or not hasattr(target, 'longitude'):
            # Cannot calculate, assume feasible
            return result

        # Convert target to ECEF
        target_ecef = self._geodetic_to_ecef(target.longitude, target.latitude, 0)

        # Calculate angle between target and sun vectors (from Earth center)
        angle_to_sun = self._calculate_angle(target_ecef, sun_position)

        if angle_to_sun < exclusion_angle:
            result.add_violation(
                ConstraintType.SUN_EXCLUSION,
                f"Target too close to sun: {angle_to_sun:.1f}° < {exclusion_angle}° exclusion angle"
            )

        return result

    def _get_sun_position(self, timestamp: datetime) -> Optional[Tuple[float, float, float]]:
        """Get sun position in ECEF coordinates.

        Uses a simplified model based on time of year.
        """
        # Check cache
        if timestamp in self._sun_position_cache:
            return self._sun_position_cache[timestamp]

        import math

        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour + timestamp.minute / 60.0

        # Sun declination (simplified)
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

        # Sun longitude (simplified - assumes sun at local noon at 12:00 UTC)
        sun_lon = 180.0 - 15.0 * hour  # 15 degrees per hour
        sun_lat = declination

        # Convert to ECEF (at ~1 AU distance, but normalized)
        r = 149600000000  # 1 AU in meters
        lon_rad = math.radians(sun_lon)
        lat_rad = math.radians(sun_lat)

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        position = (x, y, z)
        self._sun_position_cache[timestamp] = position
        return position

    def _geodetic_to_ecef(
        self, lon: float, lat: float, alt: float = 0.0
    ) -> Tuple[float, float, float]:
        """Convert geodetic coordinates to ECEF."""
        import math
        R = 6371000  # Earth radius in meters
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        r = R + alt

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def _calculate_angle(
        self, v1: Tuple[float, float, float], v2: Tuple[float, float, float]
    ) -> float:
        """Calculate angle between two vectors in degrees."""
        import math

        # Normalize vectors
        len1 = math.sqrt(sum(x * x for x in v1))
        len2 = math.sqrt(sum(x * x for x in v2))

        if len1 == 0 or len2 == 0:
            return 180.0

        # Dot product
        dot = sum(a * b for a, b in zip(v1, v2)) / (len1 * len2)
        dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability

        return math.degrees(math.acos(dot))


class SolarElevationChecker:
    """Target solar elevation angle constraint checking component.

    Ensures the target has sufficient lighting (sun above horizon) for
    optical imaging. Calculates the sun's elevation angle relative to
    the target location.

    Typical minimum solar elevation: 5-10 degrees for optical satellites
    to ensure adequate lighting conditions.
    """

    # Default minimum solar elevation angle (degrees)
    DEFAULT_MIN_SOLAR_ELEVATION = 5.0

    def __init__(self):
        self._sun_position_cache: Dict[datetime, Tuple[float, float, float]] = {}

    def check_solar_elevation(
        self,
        target: Target,
        imaging_time: datetime,
        min_elevation: Optional[float] = None
    ) -> ConstraintResult:
        """Check if target has sufficient solar elevation for imaging.

        Args:
            target: Target object with latitude/longitude
            imaging_time: Time of imaging
            min_elevation: Minimum required solar elevation (degrees),
                          defaults to DEFAULT_MIN_SOLAR_ELEVATION

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        # Check if target has position information
        if not hasattr(target, 'latitude') or not hasattr(target, 'longitude'):
            # Cannot calculate without target position
            return result

        # Get minimum elevation threshold
        min_elev = min_elevation if min_elevation is not None else self.DEFAULT_MIN_SOLAR_ELEVATION

        # Calculate solar elevation at target location
        solar_elevation = self._calculate_solar_elevation(target, imaging_time)

        if solar_elevation is None:
            # Cannot calculate, assume feasible
            return result

        if solar_elevation < min_elev:
            result.add_violation(
                ConstraintType.SOLAR_ELEVATION,
                f"Target solar elevation too low: {solar_elevation:.1f}° < {min_elev}° "
                f"(night time or insufficient lighting)"
            )

        return result

    def _calculate_solar_elevation(
        self,
        target: Target,
        time: datetime
    ) -> Optional[float]:
        """Calculate solar elevation angle at target location.

        Uses astronomical algorithms to compute the sun's elevation
        angle relative to the local horizon at the target position.

        Args:
            target: Target with latitude/longitude
            time: UTC time

        Returns:
            Solar elevation angle in degrees (0 = horizon, 90 = zenith)
            or None if calculation fails
        """
        try:
            lat = float(target.latitude)
            lon = float(target.longitude)
        except (TypeError, ValueError):
            return None

        # Get sun position in ECEF
        sun_pos = self._get_sun_position(time)
        if sun_pos is None:
            return None

        # Calculate solar elevation using astronomical formula
        return self._compute_elevation_angle(lat, lon, time, sun_pos)

    def _get_sun_position(self, timestamp: datetime) -> Optional[Tuple[float, float, float]]:
        """Get sun position in ECEF coordinates.

        Uses a simplified model based on time of year.
        """
        # Check cache
        if timestamp in self._sun_position_cache:
            return self._sun_position_cache[timestamp]

        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

        # Sun declination (simplified formula)
        # Declination varies between -23.45° and +23.45°
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

        # Sun longitude (simplified - sun at local noon at 12:00 UTC)
        # Sun moves 15 degrees per hour
        sun_lon = 180.0 - 15.0 * hour
        sun_lat = declination

        # Convert to ECEF (at 1 AU distance)
        r = 149600000000  # 1 AU in meters
        lon_rad = math.radians(sun_lon)
        lat_rad = math.radians(sun_lat)

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        position = (x, y, z)
        self._sun_position_cache[timestamp] = position
        return position

    def _compute_elevation_angle(
        self,
        lat: float,
        lon: float,
        time: datetime,
        sun_pos: Tuple[float, float, float]
    ) -> float:
        """Compute solar elevation angle at given location.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            time: UTC time
            sun_pos: Sun position in ECEF (x, y, z)

        Returns:
            Solar elevation angle in degrees
        """
        # Convert observer position to ECEF
        R = 6371000  # Earth radius in meters
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        obs_x = R * math.cos(lat_rad) * math.cos(lon_rad)
        obs_y = R * math.cos(lat_rad) * math.sin(lon_rad)
        obs_z = R * math.sin(lat_rad)

        # Vector from observer to sun
        dx = sun_pos[0] - obs_x
        dy = sun_pos[1] - obs_y
        dz = sun_pos[2] - obs_z

        # Normalize sun direction vector
        sun_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/sun_dist, dy/sun_dist, dz/sun_dist

        # Local up vector (zenith direction)
        up_x = obs_x / R
        up_y = obs_y / R
        up_z = obs_z / R

        # Cosine of elevation angle = dot product of sun direction with up vector
        cos_elevation = dx * up_x + dy * up_y + dz * up_z

        # Clamp to valid range
        cos_elevation = max(-1.0, min(1.0, cos_elevation))

        # Convert to elevation angle (90° - zenith angle)
        elevation = math.degrees(math.asin(cos_elevation))

        return elevation


class SAAChecker:
    """South Atlantic Anomaly constraint checking component."""

    # SAA ellipse parameters (simplified NASA model)
    SAA_CENTER_LAT = -25.0  # degrees
    SAA_CENTER_LON = -45.0  # degrees
    SAA_MAJOR_AXIS = 40.0   # degrees
    SAA_MINOR_AXIS = 30.0   # degrees

    def __init__(self):
        self._attitude_calc = AttitudeCalculator(propagator_type=PropagatorType.SGP4)

    def check_saa(
        self,
        sat_id: str,
        start_time: datetime,
        end_time: datetime,
        sample_interval: float = 60.0,
    ) -> ConstraintResult:
        """Check if time window intersects with SAA.

        Args:
            sat_id: Satellite ID
            start_time: Window start
            end_time: Window end
            sample_interval: Sampling interval in seconds

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        current_time = start_time
        while current_time <= end_time:
            if self._is_in_saa(sat_id, current_time):
                result.add_violation(
                    ConstraintType.SAA,
                    f"Satellite in SAA at {current_time}"
                )
                return result
            current_time += timedelta(seconds=sample_interval)

        return result

    def _is_in_saa(self, sat_id: str, time: datetime) -> bool:
        """Check if satellite is in SAA at given time."""
        # Simplified check - in practice, get satellite position and check
        # against SAA ellipse
        return False  # Placeholder - actual implementation needs position


class ConstraintChecker:
    """Unified constraint checker for satellite task scheduling.

    This class consolidates all constraint checking functionality:
    - Slew constraints (angle and time)
    - SAA constraints
    - Power constraints
    - Storage constraints
    - Temporal constraints (time conflicts)

    Usage:
        checker = ConstraintChecker(mission, config)
        checker.initialize()

        context = ConstraintContext(
            satellite=sat,
            target=task,
            window_start=window.start,
            window_end=window.end,
        )
        result = checker.check_task(context)
    """

    def __init__(
        self,
        mission: Mission,
        config: Optional[ConstraintConfig] = None,
    ):
        """Initialize constraint checker.

        Args:
            mission: Mission object
            config: Constraint configuration
        """
        self.mission = mission
        self.config = config or ConstraintConfig()

        # Component checkers
        self._slew_checker = SlewChecker()
        self._saa_checker = SAAChecker()
        self._capability_checker = CapabilityChecker()
        self._temporal_checker = TemporalChecker()
        self._thermal_checker = ThermalChecker()
        self._sun_exclusion_checker = SunExclusionChecker()
        self._solar_elevation_checker = SolarElevationChecker()

        # Track which satellites are initialized
        self._initialized_sats: set = set()

        # Imaging time calculator for power/storage calculations
        from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile
        self._imaging_calculator = ImagingTimeCalculator()
        self._power_profile = PowerProfile()

    def set_slew_checker(self, slew_checker) -> None:
        """Set external slew checker (e.g., PreciseSlewConstraintChecker).

        This allows the constraint checker to use an external slew checker
        that maintains state across multiple scheduling iterations.

        Args:
            slew_checker: External slew checker instance
        """
        self._slew_checker = slew_checker
        # Clear initialized set to force re-initialization with new slew checker
        self._initialized_sats.clear()

    def initialize(self) -> None:
        """Initialize constraint checkers for all satellites."""
        for sat in self.mission.satellites:
            self._slew_checker.initialize_satellite(sat)
            self._initialized_sats.add(sat.id)

    def check_task(
        self,
        context: ConstraintContext,
        check_types: Optional[List[ConstraintType]] = None,
    ) -> ConstraintResult:
        """Check all constraints for a task assignment.

        This is the main entry point for constraint checking. It performs
        all enabled constraint checks in the proper order.

        Args:
            context: Constraint checking context
            check_types: Specific constraint types to check (None = all)

        Returns:
            ConstraintResult with all check results
        """
        if check_types is None:
            check_types = [
                ConstraintType.CAPABILITY,
                ConstraintType.SLEW,
                ConstraintType.TEMPORAL,
                ConstraintType.SAA,
                ConstraintType.POWER,
                ConstraintType.STORAGE,
                ConstraintType.SOLAR_ELEVATION,  # Enabled by default (optical satellites need daylight)
                # Optional constraints (controlled by config flags):
                # - THERMAL (enable_thermal_check)
                # - SUN_EXCLUSION (enable_sun_exclusion_check)
            ]

        result = ConstraintResult()

        # 1. Check capability constraint (first - basic compatibility)
        if ConstraintType.CAPABILITY in check_types:
            cap_result = self._capability_checker.check_capability(
                context.satellite, context.target, context.imaging_mode
            )
            if not cap_result.feasible:
                return cap_result  # Capability failure is critical

        # 2. Check slew constraint
        if ConstraintType.SLEW in check_types:
            slew_result = self._check_slew(context)
            if not slew_result.feasible:
                result.feasible = False
                result.violations.extend(slew_result.violations)
                result.details.update(slew_result.details)
                return result  # Slew failure is critical
            result.slew_angle = slew_result.slew_angle
            result.slew_time = slew_result.slew_time
            result.actual_start = slew_result.actual_start

        # 3. Check temporal constraints (window, deadline, conflicts)
        if ConstraintType.TEMPORAL in check_types:
            temporal_result = self._check_temporal(context, result.actual_start)
            if not temporal_result.feasible:
                result.feasible = False
                result.violations.extend(temporal_result.violations)
                result.details.update(temporal_result.details)
                return result  # Temporal failure is critical

        # 4. Check SAA constraint
        if ConstraintType.SAA in check_types and self.config.enable_saa_check:
            saa_result = self._check_saa(context)
            if not saa_result.feasible:
                result.feasible = False
                result.violations.extend(saa_result.violations)
                result.details.update(saa_result.details)

        # 5. Check power constraint
        if ConstraintType.POWER in check_types and self.config.consider_power:
            power_result = self._check_power(context)
            if not power_result.feasible:
                result.feasible = False
                result.violations.extend(power_result.violations)
                result.details.update(power_result.details)

        # 6. Check storage constraint
        if ConstraintType.STORAGE in check_types and self.config.consider_storage:
            storage_result = self._check_storage(context)
            if not storage_result.feasible:
                result.feasible = False
                result.violations.extend(storage_result.violations)
                result.details.update(storage_result.details)

        # 7. Check thermal constraint (optional)
        if ConstraintType.THERMAL in check_types and getattr(self.config, 'enable_thermal_check', False):
            if result.actual_start and context.imaging_duration:
                actual_end = result.actual_start + timedelta(seconds=context.imaging_duration)
                thermal_result = self._thermal_checker.check_thermal(
                    context.satellite, result.actual_start, actual_end, context.imaging_mode
                )
                if not thermal_result.feasible:
                    result.feasible = False
                    result.violations.extend(thermal_result.violations)
                    result.details.update(thermal_result.details)

        # 8. Check sun exclusion constraint (optional, for optical satellites)
        if ConstraintType.SUN_EXCLUSION in check_types and getattr(self.config, 'enable_sun_exclusion_check', False):
            if result.actual_start:
                sun_result = self._sun_exclusion_checker.check_sun_exclusion(
                    context.satellite, context.target, result.actual_start
                )
                if not sun_result.feasible:
                    result.feasible = False
                    result.violations.extend(sun_result.violations)
                    result.details.update(sun_result.details)

        # 9. Check solar elevation constraint (optional, for optical satellites)
        # Ensures target has sufficient lighting (daytime imaging)
        if ConstraintType.SOLAR_ELEVATION in check_types and getattr(self.config, 'enable_solar_elevation_check', False):
            # Only check for optical satellites (SAR can image at night)
            sat_type = getattr(context.satellite, 'type', None)
            type_str = str(sat_type).lower() if sat_type else ''
            if 'sar' not in type_str and 'radar' not in type_str:
                if result.actual_start:
                    elevation_result = self._solar_elevation_checker.check_solar_elevation(
                        context.target, result.actual_start,
                        min_elevation=getattr(self.config, 'min_solar_elevation', None)
                    )
                    if not elevation_result.feasible:
                        result.feasible = False
                        result.violations.extend(elevation_result.violations)
                        result.details.update(elevation_result.details)

        return result

    def _check_slew(self, context: ConstraintContext) -> ConstraintResult:
        """Check slew constraint."""
        sat_id = context.satellite.id

        if sat_id not in self._initialized_sats:
            self._slew_checker.initialize_satellite(context.satellite)
            self._initialized_sats.add(sat_id)

        imaging_duration = context.imaging_duration or 10.0
        prev_end = context.prev_end_time or context.window_start

        return self._slew_checker.check_slew(
            sat_id=sat_id,
            prev_target=context.prev_target,
            current_target=context.target,
            prev_end_time=prev_end,
            window_start=context.window_start,
            imaging_duration=imaging_duration,
            use_simplified=(self.config.mode == 'simplified'),
        )

    def _check_saa(self, context: ConstraintContext) -> ConstraintResult:
        """Check SAA constraint."""
        actual_start = context.window_start
        if context.imaging_duration:
            end_time = actual_start + timedelta(seconds=context.imaging_duration)
        else:
            end_time = context.window_end

        return self._saa_checker.check_saa(
            sat_id=context.satellite.id,
            start_time=actual_start,
            end_time=end_time,
        )

    def _check_power(self, context: ConstraintContext) -> ConstraintResult:
        """Check power constraint."""
        result = ConstraintResult()

        # Calculate power needed
        imaging_duration = context.imaging_duration or 10.0
        # Simplified power calculation - actual implementation should use
        # power profile based on imaging mode
        power_coefficient = 0.1  # Placeholder
        power_capacity = getattr(
            context.satellite.capabilities,
            'power_capacity',
            2800.0
        )
        # Handle Mock objects
        if not isinstance(power_capacity, (int, float)):
            power_capacity = 2800.0
        current_power = context.current_power
        if not isinstance(current_power, (int, float)):
            current_power = power_capacity
        power_needed = power_capacity * power_coefficient * (imaging_duration / 3600)

        if current_power < power_needed:
            result.add_violation(
                ConstraintType.POWER,
                f"Insufficient power: {context.current_power:.1f} < {power_needed:.1f}"
            )

        return result

    def _check_storage(self, context: ConstraintContext) -> ConstraintResult:
        """Check storage constraint."""
        result = ConstraintResult()

        # Calculate storage needed
        imaging_duration = context.imaging_duration or 10.0
        data_rate = getattr(context.satellite.capabilities, 'data_rate', 300.0)
        # Handle Mock objects
        if not isinstance(data_rate, (int, float)):
            data_rate = 300.0
        # Convert to GB (300 Mbps * 10s = 3000 Mb = 0.375 GB)
        storage_needed = (data_rate * imaging_duration) / 8000

        storage_capacity = getattr(
            context.satellite.capabilities,
            'storage_capacity',
            128.0
        )
        # Handle Mock objects
        if not isinstance(storage_capacity, (int, float)):
            storage_capacity = 128.0
        current_storage = context.current_storage
        if not isinstance(current_storage, (int, float)):
            current_storage = 0.0

        if current_storage + storage_needed > storage_capacity:
            result.add_violation(
                ConstraintType.STORAGE,
                f"Storage overflow: {context.current_storage:.1f} + {storage_needed:.1f} > {storage_capacity:.1f}"
            )

        return result

    def _check_temporal(
        self,
        context: ConstraintContext,
        actual_start: Optional[datetime] = None
    ) -> ConstraintResult:
        """Check temporal constraints including window, deadline, and conflicts.

        Args:
            context: Constraint context
            actual_start: Actual task start time (if None, uses window_start)

        Returns:
            ConstraintResult
        """
        result = ConstraintResult()

        if actual_start is None:
            actual_start = context.window_start

        imaging_duration = context.imaging_duration or 10.0
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        # Check window boundary
        if actual_end > context.window_end:
            result.add_violation(
                ConstraintType.WINDOW_LENGTH,
                f"Task ends after window end: {actual_end} > {context.window_end}"
            )
            return result

        # Check deadline
        if context.deadline and actual_end > context.deadline:
            result.add_violation(
                ConstraintType.DEADLINE,
                f"Task ends after deadline: {actual_end} > {context.deadline}"
            )
            return result

        # Check time conflicts with scheduled tasks
        if context.scheduled_tasks:
            for task in context.scheduled_tasks:
                task_start = task.get('start')
                task_end = task.get('end')
                if task_start and task_end:
                    if not (actual_end <= task_start or actual_start >= task_end):
                        result.add_violation(
                            ConstraintType.TIME_CONFLICT,
                            f"Time conflict with task from {task_start} to {task_end}"
                        )
                        return result

        return result

    def check_time_conflict(
        self,
        sat_id: str,
        start: datetime,
        end: datetime,
        scheduled_tasks: List[Dict[str, Any]],
    ) -> bool:
        """Check if time range conflicts with scheduled tasks.

        Args:
            sat_id: Satellite ID (not used, for API consistency)
            start: Proposed start time
            end: Proposed end time
            scheduled_tasks: List of scheduled task dicts with 'start' and 'end'

        Returns:
            True if there is a conflict
        """
        for task in scheduled_tasks:
            existing_start = task['start']
            existing_end = task['end']
            if not (end <= existing_start or start >= existing_end):
                return True
        return False
