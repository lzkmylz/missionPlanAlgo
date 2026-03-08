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
from .precise_slew_constraint_checker import PreciseSlewConstraintChecker

__all__ = [
    'SlewConstraintChecker',  # 基类，保留供继承
    'SlewFeasibilityResult',
    'PreciseSlewConstraintChecker',  # 默认使用的精确约束检查器
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
