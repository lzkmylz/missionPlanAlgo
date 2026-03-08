"""
精确姿态机动约束检查器

继承并扩展现有 SlewConstraintChecker，添加精确姿态机动计算能力。
保持与原有接口完全兼容，通过配置切换模型。

主要功能:
1. 基于刚体动力学的精确机动计算
2. 飞轮动量状态跟踪
3. 能量消耗估算
4. 自动回退到简化模型
5. 状态同步到 SatelliteStateTracker

使用示例:
    # 使用精确模型
    checker = PreciseSlewConstraintChecker(
        mission=mission,
        use_precise_model=True
    )

    # 使用简化模型 (与原有代码一致)
    checker = PreciseSlewConstraintChecker(
        mission=mission,
        use_precise_model=False
    )
"""

import numpy as np
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
import logging

from core.models.mission import Mission
from core.models.satellite import Satellite
from core.models.target import Target
from core.dynamics.precise import (
    PreciseSlewCalculator, SatelliteDynamicsConfig,
    AttitudeState, Quaternion, AngularVelocity, MomentumState
)
from .slew_constraint_checker import (
    SlewConstraintChecker, SlewFeasibilityResult
)

logger = logging.getLogger(__name__)


class PreciseSlewConstraintChecker(SlewConstraintChecker):
    """精确姿态机动约束检查器

    继承 SlewConstraintChecker，添加基于刚体动力学的精确计算能力。
    保持接口完全兼容，通过 use_precise_model 参数切换。
    """

    def __init__(
        self,
        mission: Mission,
        use_precise_model: bool = True,
        precise_calculators: Optional[Dict[str, PreciseSlewCalculator]] = None
    ):
        """初始化精确约束检查器

        Args:
            mission: 任务对象
            use_precise_model: 是否使用精确模型
            precise_calculators: 预计算的精确计算器字典 {sat_id: calculator}
        """
        # 调用父类初始化
        super().__init__(mission)

        self.use_precise = use_precise_model
        self._precise_calcs: Dict[str, PreciseSlewCalculator] = precise_calculators or {}

        # 状态跟踪引用 (由外部设置)
        self._state_tracker = None

        if use_precise_model:
            self._initialize_precise_calculators()
            logger.info("PreciseSlewConstraintChecker initialized with precise model")
        else:
            logger.info("PreciseSlewConstraintChecker initialized with simple model")

    def set_state_tracker(self, state_tracker) -> None:
        """设置状态跟踪器引用

        Args:
            state_tracker: SatelliteStateTracker 实例
        """
        self._state_tracker = state_tracker

    def _initialize_precise_calculators(self) -> None:
        """为每个卫星初始化精确计算器"""
        for sat in self.mission.satellites:
            config = self._extract_dynamics_config(sat)
            self._precise_calcs[sat.id] = PreciseSlewCalculator(
                config=config,
                use_precise=True
            )
            logger.debug(f"Initialized precise calculator for {sat.id}")

    def _extract_dynamics_config(self, satellite: Satellite) -> SatelliteDynamicsConfig:
        """从卫星对象提取动力学配置

        Args:
            satellite: 卫星对象

        Returns:
            动力学配置
        """
        # 获取卫星敏捷性参数
        agility = getattr(satellite.capabilities, 'agility', {}) or {}

        # 尝试获取惯性参数 (如果没有则使用默认值)
        mass = getattr(satellite, 'mass', 100.0)  # kg

        # 假设为长方体卫星，估算惯性张量
        # 对于典型100kg卫星，尺寸约 0.8 x 0.6 x 0.5 m
        width, depth, height = 0.8, 0.6, 0.5  # meters

        Ixx = mass * (depth**2 + height**2) / 12
        Iyy = mass * (width**2 + height**2) / 12
        Izz = mass * (width**2 + depth**2) / 12

        from core.dynamics.precise import InertiaTensor

        return SatelliteDynamicsConfig(
            inertia_tensor=InertiaTensor.diagonal(Ixx, Iyy, Izz),
            max_control_torque=agility.get('max_torque', 0.5),
            max_angular_velocity=agility.get('max_slew_rate', 3.0),
            max_slew_rate=agility.get('max_slew_rate', 3.0),
            max_slew_angle=satellite.capabilities.max_off_nadir,
            settling_time=agility.get('settling_time', 5.0)
        )

    def check_slew_feasibility(
        self,
        satellite_id: str,
        prev_target: Optional[Target],
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime,
        imaging_duration: float = 0.0,
        **kwargs
    ) -> SlewFeasibilityResult:
        """检查机动可行性 - 增强版本

        保持与父类完全兼容的接口，内部根据配置选择模型。

        Args:
            satellite_id: 卫星ID
            prev_target: 上一个目标 (None表示这是第一个任务)
            current_target: 当前目标
            prev_end_time: 上一个任务结束时间
            window_start: 当前窗口开始时间
            imaging_duration: 成像持续时间 (秒)
            **kwargs: 额外参数 (用于扩展)

        Returns:
            SlewFeasibilityResult: 可行性结果
        """
        # 获取卫星
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            return self._error_result(
                f"Satellite {satellite_id} not found", window_start
            )

        # 如果没有前一个目标，不需要机动
        if prev_target is None:
            return self._first_task_result(window_start)

        # 获取卫星位置
        sat_position, sat_velocity = self._get_satellite_position(
            satellite, prev_end_time
        )

        # 选择模型
        if self.use_precise and satellite_id in self._precise_calcs:
            return self._check_precise_feasibility(
                satellite=satellite,
                prev_target=prev_target,
                current_target=current_target,
                prev_end_time=prev_end_time,
                window_start=window_start,
                sat_position=sat_position,
                imaging_duration=imaging_duration,
                **kwargs
            )
        else:
            # 使用父类的简化检查
            return super().check_slew_feasibility(
                satellite_id, prev_target, current_target,
                prev_end_time, window_start, imaging_duration
            )

    def _check_precise_feasibility(
        self,
        satellite: Satellite,
        prev_target: Target,
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime,
        sat_position: Tuple[float, float, float],
        imaging_duration: float,
        **kwargs
    ) -> SlewFeasibilityResult:
        """使用精确模型的可行性检查

        Args:
            satellite: 卫星对象
            prev_target: 前一目标
            current_target: 当前目标
            prev_end_time: 前一任务结束时间
            window_start: 窗口开始时间
            sat_position: 卫星位置
            imaging_duration: 成像持续时间

        Returns:
            可行性结果
        """
        calc = self._precise_calcs[satellite.id]

        # 1. 获取当前姿态状态
        current_state = self._get_satellite_attitude_state(
            satellite.id, prev_end_time
        )

        # 2. 计算目标姿态
        target_attitude = self._compute_target_attitude(
            satellite, current_target, sat_position
        )

        # 3. 执行精确机动分析
        try:
            maneuver = calc.calculate_slew_maneuver(
                prev_attitude=current_state,
                target_attitude=target_attitude,
                current_time=prev_end_time
            )
        except Exception as e:
            logger.warning(f"Precise maneuver calculation failed: {e}")
            # 回退到简化模型
            return super().check_slew_feasibility(
                satellite.id, prev_target, current_target,
                prev_end_time, window_start, imaging_duration
            )

        # 4. 检查约束满足
        if not maneuver.feasible:
            reason = "Momentum saturation risk"
            if maneuver.momentum_check and maneuver.momentum_check.recommendation:
                reason = maneuver.momentum_check.recommendation

            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=maneuver.trajectory.rotation_angle,
                slew_time=maneuver.total_time,
                actual_start=window_start,
                reason=reason
            )

        # 5. 计算机动后实际开始时间
        earliest_start = prev_end_time + timedelta(seconds=maneuver.total_time)
        actual_start = max(window_start, earliest_start)

        # 6. 检查成像窗口时间
        if imaging_duration > 0:
            actual_end = actual_start + timedelta(seconds=imaging_duration)
            window_end = getattr(current_target, 'time_window_end', None)
            if window_end and actual_end > window_end:
                return SlewFeasibilityResult(
                    feasible=False,
                    slew_angle=maneuver.trajectory.rotation_angle,
                    slew_time=maneuver.total_time,
                    actual_start=actual_start,
                    reason="Not enough time after precise slew maneuver"
                )

        # 7. 更新卫星状态跟踪
        self._update_satellite_state(
            satellite_id=satellite.id,
            time=actual_start,
            attitude=target_attitude,
            angular_momentum=maneuver.final_momentum
        )

        # 8. 构造结果 (保持接口兼容)
        result = SlewFeasibilityResult(
            feasible=True,
            slew_angle=maneuver.trajectory.rotation_angle,
            slew_time=maneuver.total_time,
            actual_start=actual_start,
            reason=None
        )

        # 附加精确模型信息 (可选字段)
        result.energy_consumption = maneuver.energy_consumption
        result.momentum_margin = maneuver.momentum_margin
        result.precise_result = maneuver

        return result

    def _get_satellite_attitude_state(
        self,
        satellite_id: str,
        timestamp: datetime
    ) -> AttitudeState:
        """获取卫星姿态状态

        优先从状态跟踪器获取，如果没有则使用默认值。

        Args:
            satellite_id: 卫星ID
            timestamp: 时间戳

        Returns:
            姿态状态
        """
        # 尝试从状态跟踪器获取
        if self._state_tracker is not None:
            attitude = self._state_tracker.get_attitude_state(satellite_id)
            momentum = self._state_tracker.get_angular_momentum(satellite_id)

            if attitude is not None:
                return AttitudeState(
                    quaternion=attitude.get('quaternion', Quaternion(1, 0, 0, 0)),
                    angular_velocity=attitude.get(
                        'angular_velocity', AngularVelocity(0, 0, 0)
                    ),
                    timestamp=timestamp,
                    momentum=momentum
                )

        # 使用默认状态 (对地定向)
        return AttitudeState(
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),  # 单位四元数
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=timestamp,
            momentum=None
        )

    def _compute_target_attitude(
        self,
        satellite: Satellite,
        target: Target,
        sat_position: Tuple[float, float, float]
    ) -> AttitudeState:
        """计算目标姿态

        基于目标位置和卫星位置计算机动所需姿态。

        Args:
            satellite: 卫星
            target: 目标
            sat_position: 卫星位置

        Returns:
            目标姿态状态
        """
        # 使用父类方法计算机动角度
        slew_angle = self._calculate_slew_angle_ecef(
            sat_position, target, target  # 简化处理
        )

        # 构造姿态四元数 (简化：假设绕y轴旋转)
        # 实际应该基于视线矢量计算完整姿态
        angle_rad = np.radians(slew_angle) / 2
        q = Quaternion(
            w=np.cos(angle_rad),
            x=0.0,
            y=np.sin(angle_rad),
            z=0.0
        )

        return AttitudeState(
            quaternion=q.normalize(),
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=datetime.now(),
            momentum=None
        )

    def _update_satellite_state(
        self,
        satellite_id: str,
        time: datetime,
        attitude: AttitudeState,
        angular_momentum: np.ndarray
    ) -> None:
        """更新卫星状态跟踪

        Args:
            satellite_id: 卫星ID
            time: 时间
            attitude: 姿态状态
            angular_momentum: 角动量
        """
        if self._state_tracker is not None:
            self._state_tracker.update_after_maneuver(
                satellite_id=satellite_id,
                maneuver_result={
                    'final_quaternion': attitude.quaternion,
                    'final_momentum': angular_momentum,
                    'timestamp': time
                }
            )

    def _first_task_result(self, window_start: datetime) -> SlewFeasibilityResult:
        """第一个任务的结果"""
        settling_time = 5.0  # 默认稳定时间
        return SlewFeasibilityResult(
            feasible=True,
            slew_angle=0.0,
            slew_time=settling_time,
            actual_start=window_start + timedelta(seconds=settling_time),
            reason=None
        )

    def _error_result(
        self, reason: str, window_start: datetime
    ) -> SlewFeasibilityResult:
        """错误结果"""
        return SlewFeasibilityResult(
            feasible=False,
            slew_angle=0.0,
            slew_time=0.0,
            actual_start=window_start,
            reason=reason
        )
