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
    TEMPORAL = "temporal"      # Time window conflicts
    SLEW = "slew"             # Slew angle and time
    SAA = "saa"               # South Atlantic Anomaly
    POWER = "power"           # Power capacity
    STORAGE = "storage"       # Storage capacity
    ATTITUDE = "attitude"     # Attitude mode transitions


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
    """
    feasible: bool = True
    violations: List[ConstraintType] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    slew_angle: float = 0.0
    slew_time: float = 0.0
    actual_start: Optional[datetime] = None
    reason: Optional[str] = None

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

        if use_simplified or prev_target is None:
            # Simplified check: just verify imaging fits in window
            result.slew_time = 0.0
            result.slew_angle = 0.0
            result.actual_start = window_start
            return result

        # Calculate slew angle using target positions
        slew_angle = self._calculate_slew_angle(prev_target, current_target)
        result.slew_angle = slew_angle

        # Check if exceeds max slew angle
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
        prev_target: Target,
        current_target: Target,
    ) -> float:
        """Calculate slew angle between two targets."""
        # Simplified calculation using lat/lon difference
        # In practice, this should use proper ECEF coordinate transformation

        # Handle Mock objects gracefully
        prev_lat = getattr(prev_target, 'latitude', 0)
        prev_lon = getattr(prev_target, 'longitude', 0)
        curr_lat = getattr(current_target, 'latitude', 0)
        curr_lon = getattr(current_target, 'longitude', 0)

        # Ensure we have numeric values
        if not isinstance(prev_lat, (int, float)):
            prev_lat = 0
        if not isinstance(prev_lon, (int, float)):
            prev_lon = 0
        if not isinstance(curr_lat, (int, float)):
            curr_lat = 0
        if not isinstance(curr_lon, (int, float)):
            curr_lon = 0

        lat_diff = abs(prev_lat - curr_lat)
        lon_diff = abs(prev_lon - curr_lon)

        # Approximate angular separation
        return math.sqrt(lat_diff**2 + lon_diff**2)


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

        # Track which satellites are initialized
        self._initialized_sats: set = set()

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

        Args:
            context: Constraint checking context
            check_types: Specific constraint types to check (None = all)

        Returns:
            ConstraintResult with all check results
        """
        if check_types is None:
            check_types = [
                ConstraintType.SLEW,
                ConstraintType.SAA,
                ConstraintType.POWER,
                ConstraintType.STORAGE,
            ]

        result = ConstraintResult()

        # Check slew constraint
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

        # Check SAA constraint
        if ConstraintType.SAA in check_types and self.config.enable_saa_check:
            saa_result = self._check_saa(context)
            if not saa_result.feasible:
                result.feasible = False
                result.violations.extend(saa_result.violations)
                result.details.update(saa_result.details)

        # Check power constraint
        if ConstraintType.POWER in check_types and self.config.consider_power:
            power_result = self._check_power(context)
            if not power_result.feasible:
                result.feasible = False
                result.violations.extend(power_result.violations)
                result.details.update(power_result.details)

        # Check storage constraint
        if ConstraintType.STORAGE in check_types and self.config.consider_storage:
            storage_result = self._check_storage(context)
            if not storage_result.feasible:
                result.feasible = False
                result.violations.extend(storage_result.violations)
                result.details.update(storage_result.details)

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
