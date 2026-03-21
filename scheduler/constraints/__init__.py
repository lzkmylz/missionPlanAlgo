"""
调度器约束检查模块

提供统一的约束检查接口，包括姿态机动约束和SAA约束。
"""

# 统一使用高性能批量版本，保留旧类名作为别名
from .batch_slew_constraint_checker import (
    BatchSlewConstraintChecker,
    BatchSlewConstraintChecker as SlewConstraintChecker,  # 兼容性别名
    SlewFeasibilityResult,
)
from .saa_constraint_checker import SAAConstraintChecker, SAAFeasibilityResult
from .batch_saa_calculator import (
    BatchSAACalculator,
    BatchSAACandidate,
    BatchSAAResult,
    BatchSAAData
)
from .batch_saa_constraint_checker import BatchSAAConstraintChecker
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
from .batch_slew_constraint_checker import (
    BatchSlewConstraintChecker as PreciseSlewConstraintChecker,  # 兼容性别名
)
from .batch_slew_calculator import (
    BatchSlewCalculator,
    BatchSlewCandidate,
    BatchSlewResult,
    BatchSlewData
)
from .batch_time_conflict_calculator import (
    BatchTimeConflictCalculator,
    BatchTimeConflictCandidate,
    BatchTimeConflictResult,
    BatchTimeConflictData
)
from .batch_time_conflict_checker import BatchTimeConflictChecker
from .batch_resource_calculator import (
    BatchResourceCalculator,
    BatchResourceCandidate,
    BatchResourceResult,
    BatchResourceData
)
from .batch_resource_checker import BatchResourceChecker
from .batch_orbit_constraint_checker import (
    BatchOrbitConstraintChecker,
    BatchOrbitConstraintCandidate,
    BatchOrbitConstraintResult
)
from .unified_batch_constraint_checker import (
    UnifiedBatchConstraintChecker,
    UnifiedBatchCandidate,
    UnifiedBatchResult
)
from .pmc_constraint_checker import (
    PMCConstraintChecker,
    PMCCandidate,
    PMCConstraintResult,
    check_pmc_mode_for_task
)
from .multi_strip_constraint_checker import (
    MultiStripConstraintChecker,
    MultiStripCandidate,
    MultiStripConstraintResult,
)
from .uplink_channel_type import UplinkChannelType, UplinkPass
from .uplink_window_registry import UplinkWindowRegistry
from .batch_uplink_calculator import (
    BatchUplinkCalculator,
    BatchUplinkCandidate,
    BatchUplinkResult,
)
from .batch_uplink_constraint_checker import BatchUplinkConstraintChecker

__all__ = [
    'SlewConstraintChecker',  # 基类，保留供继承
    'SlewFeasibilityResult',
    'PreciseSlewConstraintChecker',  # 默认使用的精确约束检查器
    'BatchSlewConstraintChecker',  # 批量优化版本
    'BatchSlewCalculator',
    'BatchSlewCandidate',
    'BatchSlewResult',
    'BatchSlewData',
    'SAAConstraintChecker',  # 基类，保留供继承
    'SAAFeasibilityResult',
    'BatchSAAConstraintChecker',  # 批量优化版本（默认使用）
    'BatchSAACalculator',
    'BatchSAACandidate',
    'BatchSAAResult',
    'BatchSAAData',
    'BatchTimeConflictChecker',  # 批量时间冲突检查器
    'BatchTimeConflictCalculator',
    'BatchTimeConflictCandidate',
    'BatchTimeConflictResult',
    'BatchTimeConflictData',
    'BatchResourceChecker',  # 批量资源约束检查器
    'BatchResourceCalculator',
    'BatchResourceCandidate',
    'BatchResourceResult',
    'BatchResourceData',
    'BatchOrbitConstraintChecker',  # 单圈约束批量检查器（新增）
    'BatchOrbitConstraintCandidate',
    'BatchOrbitConstraintResult',
    'UnifiedBatchConstraintChecker',  # 统一批量约束检查器
    'UnifiedBatchCandidate',
    'UnifiedBatchResult',
    'PMCConstraintChecker',  # PMC约束检查器（新增）
    'PMCCandidate',
    'PMCConstraintResult',
    'check_pmc_mode_for_task',
    'MultiStripConstraintChecker',  # 多条带拼幅约束检查器
    'MultiStripCandidate',
    'MultiStripConstraintResult',
    'AttitudeConstraintChecker',
    'AttitudeFeasibilityResult',
    'UnifiedSpatiotemporalChecker',
    'SpatiotemporalCheckResult',
    'UnifiedManeuverChecker',
    'ManeuverCheckResult',
    'ScheduledTaskInfo',
    'SatelliteTaskState',
    # 指令上注约束
    'UplinkChannelType',
    'UplinkPass',
    'UplinkWindowRegistry',
    'BatchUplinkCalculator',
    'BatchUplinkCandidate',
    'BatchUplinkResult',
    'BatchUplinkConstraintChecker',
]
