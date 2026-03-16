"""
姿态约束检查器

.. deprecated::
    此模块已弃用。请使用 BatchSlewConstraintChecker 替代。

    旧用法（已弃用）:
        from scheduler.constraints import AttitudeConstraintChecker
        checker = AttitudeConstraintChecker(mission, config)

    新用法（推荐）:
        from scheduler.constraints import BatchSlewConstraintChecker
        checker = BatchSlewConstraintChecker(mission, use_precise_model=True)

检查姿态切换的可行性，为调度器提供姿态约束验证。

功能：
1. 检查姿态切换是否可行（机动角度是否超过限制）
2. 计算机动时间和角度
3. 返回可行性结果（类似SlewConstraintChecker）

与AttitudeManager集成：
- 使用AttitudeManager.plan_transition计算切换参数
- 提供调度器友好的接口
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings

from core.dynamics.attitude_manager import AttitudeManager, AttitudeManagementConfig
from core.dynamics.attitude_mode import AttitudeMode, TransitionResult

# 模块级弃用警告
warnings.warn(
    "scheduler.constraints.attitude_constraint_checker 模块已弃用。"
    "请使用 scheduler.constraints.batch_slew_constraint_checker.BatchSlewConstraintChecker",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class AttitudeFeasibilityResult:
    """姿态可行性检查结果

    Attributes:
        feasible: 是否可行
        slew_time: 机动时间（秒）
        slew_angle: 机动角度（度）
        transition_time: 切换所需时间（timedelta）
        from_mode: 起始姿态模式
        to_mode: 目标姿态模式
        reason: 不可行原因（如果不可行）
    """
    feasible: bool
    slew_time: float
    slew_angle: float
    transition_time: timedelta
    from_mode: AttitudeMode
    to_mode: AttitudeMode
    reason: Optional[str] = None


class AttitudeConstraintChecker:
    """姿态约束检查器

    .. deprecated::
        此类已弃用。请使用 BatchSlewConstraintChecker 替代。

        旧用法（已弃用）:
            checker = AttitudeConstraintChecker(config)

        新用法（推荐）:
            from scheduler.constraints import BatchSlewConstraintChecker
            checker = BatchSlewConstraintChecker(mission, use_precise_model=True)

    此类保留仅用于向后兼容，不再被主动维护。

    Attributes:
        config: 姿态管理配置
        _attitude_manager: 姿态管理器实例
    """

    def __init__(self, config: Optional[AttitudeManagementConfig] = None):
        """初始化姿态约束检查器

        Args:
            config: 姿态管理配置，如果为None则使用默认配置

        .. deprecated::
            请使用 BatchSlewConstraintChecker 替代
        """
        warnings.warn(
            "AttitudeConstraintChecker 已弃用。请使用 BatchSlewConstraintChecker",
            DeprecationWarning,
            stacklevel=2
        )
        self.config = config if config is not None else AttitudeManagementConfig()
        self._attitude_manager = AttitudeManager(self.config)

    def check_attitude_transition(
        self,
        from_mode: AttitudeMode,
        to_mode: AttitudeMode,
        satellite_position: Tuple[float, float, float],
        timestamp: datetime,
        target_position: Optional[Tuple[float, float]] = None,
        ground_station_position: Optional[Tuple[float, float]] = None,
    ) -> AttitudeFeasibilityResult:
        """检查姿态切换可行性

        计算从一种姿态模式切换到另一种姿态的可行性。

        Args:
            from_mode: 起始姿态模式
            to_mode: 目标姿态模式
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间
            target_position: 目标位置（纬度、经度），成像模式需要
            ground_station_position: 地面站位置（纬度、经度），数传模式需要

        Returns:
            AttitudeFeasibilityResult: 可行性检查结果
        """
        # 验证卫星位置
        if satellite_position is None:
            return AttitudeFeasibilityResult(
                feasible=False,
                slew_time=0.0,
                slew_angle=0.0,
                transition_time=timedelta(seconds=0),
                from_mode=from_mode,
                to_mode=to_mode,
                reason="Satellite position is required"
            )

        if not isinstance(satellite_position, (tuple, list)) or len(satellite_position) != 3:
            return AttitudeFeasibilityResult(
                feasible=False,
                slew_time=0.0,
                slew_angle=0.0,
                transition_time=timedelta(seconds=0),
                from_mode=from_mode,
                to_mode=to_mode,
                reason="Invalid satellite position format"
            )

        try:
            # 使用AttitudeManager规划切换
            transition_result = self._attitude_manager.plan_transition(
                from_mode=from_mode,
                to_mode=to_mode,
                satellite_position=satellite_position,
                timestamp=timestamp,
                target_position=target_position,
                ground_station_position=ground_station_position,
            )

            # 转换为AttitudeFeasibilityResult
            return AttitudeFeasibilityResult(
                feasible=transition_result.feasible,
                slew_time=transition_result.slew_time,
                slew_angle=transition_result.slew_angle,
                transition_time=timedelta(seconds=transition_result.slew_time),
                from_mode=from_mode,
                to_mode=to_mode,
                reason=transition_result.reason if not transition_result.feasible else None
            )

        except Exception as e:
            # 处理异常情况
            return AttitudeFeasibilityResult(
                feasible=False,
                slew_time=0.0,
                slew_angle=0.0,
                transition_time=timedelta(seconds=0),
                from_mode=from_mode,
                to_mode=to_mode,
                reason=f"Error calculating transition: {str(e)}"
            )

    def check_slew_feasibility(
        self,
        from_mode: AttitudeMode,
        to_mode: AttitudeMode,
        satellite_position: Tuple[float, float, float],
        timestamp: datetime,
        max_slew_angle: float,
        target_position: Optional[Tuple[float, float]] = None,
        ground_station_position: Optional[Tuple[float, float]] = None,
    ) -> AttitudeFeasibilityResult:
        """检查机动可行性（带最大机动角度限制）

        检查姿态切换是否满足最大机动角度约束。

        Args:
            from_mode: 起始姿态模式
            to_mode: 目标姿态模式
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间
            max_slew_angle: 最大允许机动角度（度）
            target_position: 目标位置（纬度、经度）
            ground_station_position: 地面站位置（纬度、经度）

        Returns:
            AttitudeFeasibilityResult: 可行性检查结果
        """
        # 首先获取基本切换结果
        result = self.check_attitude_transition(
            from_mode=from_mode,
            to_mode=to_mode,
            satellite_position=satellite_position,
            timestamp=timestamp,
            target_position=target_position,
            ground_station_position=ground_station_position,
        )

        # 如果基本检查已经不可行，直接返回
        if not result.feasible:
            return result

        # 检查机动角度是否超过限制
        # 相同模式总是可行
        if from_mode == to_mode:
            return result

        if result.slew_angle > max_slew_angle:
            return AttitudeFeasibilityResult(
                feasible=False,
                slew_time=result.slew_time,
                slew_angle=result.slew_angle,
                transition_time=result.transition_time,
                from_mode=from_mode,
                to_mode=to_mode,
                reason=f"Slew angle {result.slew_angle:.2f} exceeds max {max_slew_angle}"
            )

        return result

    def calculate_transition_time(
        self,
        from_mode: AttitudeMode,
        to_mode: AttitudeMode,
        satellite_position: Tuple[float, float, float],
        timestamp: datetime,
        target_position: Optional[Tuple[float, float]] = None,
        ground_station_position: Optional[Tuple[float, float]] = None,
    ) -> timedelta:
        """计算机动切换时间

        计算从一种姿态模式切换到另一种姿态所需的时间。

        Args:
            from_mode: 起始姿态模式
            to_mode: 目标姿态模式
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间
            target_position: 目标位置（纬度、经度）
            ground_station_position: 地面站位置（纬度、经度）

        Returns:
            timedelta: 切换所需时间
        """
        result = self.check_attitude_transition(
            from_mode=from_mode,
            to_mode=to_mode,
            satellite_position=satellite_position,
            timestamp=timestamp,
            target_position=target_position,
            ground_station_position=ground_station_position,
        )

        return result.transition_time
