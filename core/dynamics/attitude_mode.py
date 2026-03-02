"""
Core data structures for attitude management system.

This module defines the fundamental types for attitude control:
- AttitudeMode: Enum representing different satellite attitude modes
- AttitudeTransition: Dataclass for attitude transition requests
- TransitionResult: Dataclass for transition calculation results
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Tuple


class AttitudeMode(Enum):
    """
    Enum representing satellite attitude modes.

    Each mode defines a specific orientation strategy for the satellite:
    - SUN_POINTING: Orient solar panels toward the sun for power generation
    - NADIR_POINTING: Point instruments toward Earth (nadir direction)
    - IMAGING: Orient for target observation/imaging
    - DOWNLINK: Point antenna toward ground station for data transmission
    - REALTIME: Real-time pointing for specific operations
    - MOMENTUM_DUMP: Momentum dumping maneuver orientation
    """

    SUN_POINTING = auto()
    NADIR_POINTING = auto()
    IMAGING = auto()
    DOWNLINK = auto()
    REALTIME = auto()
    MOMENTUM_DUMP = auto()


@dataclass(frozen=True)
class AttitudeTransition:
    """
    Represents a request for an attitude transition.

    This dataclass captures all necessary information to calculate
    the feasibility and parameters of an attitude maneuver.

    Attributes:
        from_mode: Starting attitude mode
        to_mode: Target attitude mode
        timestamp: Time when transition should occur
        satellite_position: Satellite position in ECEF coordinates (x, y, z) in METERS
        sun_position: Optional sun position in ECEF coordinates (x, y, z) in METERS
        target_position: Optional target position (latitude, longitude) in degrees
        ground_station_position: Optional ground station (latitude, longitude) in degrees

    Note:
        This is an immutable (frozen) dataclass to ensure thread safety
        and prevent accidental modification after creation.
    """

    from_mode: AttitudeMode
    to_mode: AttitudeMode
    timestamp: datetime
    satellite_position: Tuple[float, float, float]
    sun_position: Optional[Tuple[float, float, float]] = None
    target_position: Optional[Tuple[float, float]] = None
    ground_station_position: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """Validate satellite_position is a 3-tuple."""
        if len(self.satellite_position) != 3:
            raise ValueError(
                f"satellite_position must be a 3-tuple (x, y, z), "
                f"got {len(self.satellite_position)} elements"
            )


@dataclass(frozen=True)
class TransitionResult:
    """
    Represents the result of an attitude transition calculation.

    This dataclass contains all calculated parameters for a potential
    attitude maneuver, including feasibility status.

    Attributes:
        slew_time: Time required for the maneuver in seconds
        slew_angle: Total slew angle in degrees
        roll_angle: Required roll angle in degrees
        pitch_angle: Required pitch angle in degrees
        power_generation: Expected power generation during/after maneuver in Watts
        feasible: Whether the transition is feasible given constraints
        reason: Optional explanation if transition is not feasible

    Note:
        This is an immutable (frozen) dataclass to ensure thread safety
        and prevent accidental modification after creation.
    """

    slew_time: float
    slew_angle: float
    roll_angle: float
    pitch_angle: float
    power_generation: float
    feasible: bool
    reason: Optional[str] = None


__all__ = ['AttitudeMode', 'AttitudeTransition', 'TransitionResult']
