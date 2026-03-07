"""
调度器约束检查模块

提供统一的约束检查接口，包括姿态机动约束和SAA约束。
"""

from .slew_constraint_checker import SlewConstraintChecker, SlewFeasibilityResult
from .saa_constraint_checker import SAAConstraintChecker, SAAFeasibilityResult
from .attitude_constraint_checker import AttitudeConstraintChecker, AttitudeFeasibilityResult
from .unified_spatiotemporal_checker import (
    UnifiedSpatiotemporalChecker,
    SpatiotemporalCheckResult,
    ScheduledTaskInfo as STScheduledTaskInfo
)
from .unified_maneuver_checker import (
    UnifiedManeuverChecker,
    ManeuverCheckResult,
    ScheduledTaskInfo,
    SatelliteTaskState
)

__all__ = [
    'SlewConstraintChecker',
    'SlewFeasibilityResult',
    'SAAConstraintChecker',
    'SAAFeasibilityResult',
    'AttitudeConstraintChecker',
    'AttitudeFeasibilityResult',
    'UnifiedSpatiotemporalChecker',
    'SpatiotemporalCheckResult',
    'UnifiedManeuverChecker',
    'ManeuverCheckResult',
    'ScheduledTaskInfo',
    'SatelliteTaskState',
]
