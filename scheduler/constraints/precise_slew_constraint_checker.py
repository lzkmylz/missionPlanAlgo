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

        # 内部姿态缓存 (当外部state_tracker不可用时使用)
        self._last_attitudes: Dict[str, AttitudeState] = {}

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

        # 如果没有前一个目标，使用对地定向作为初始姿态
        if prev_target is None:
            return self._check_first_task_feasibility(
                satellite=satellite,
                current_target=current_target,
                window_start=window_start,
                imaging_duration=imaging_duration,
                **kwargs
            )

        # 获取卫星位置
        sat_position, sat_velocity = self._get_satellite_position(
            satellite, prev_end_time
        )

        # 检查时间间隔：如果超过5分钟，假设卫星已回到对地定向
        # 这样可以避免不必要的姿态复位计算，提高性能
        time_diff = (window_start - prev_end_time).total_seconds()
        assume_nadir = time_diff > 300  # 5分钟 = 300秒

        # 选择模型
        if self.use_precise and satellite_id in self._precise_calcs:
            if assume_nadir:
                # 长时间间隔：假设已回到对地定向，跳过复位计算
                return self._check_nadir_to_target_feasibility(
                    satellite=satellite,
                    current_target=current_target,
                    window_start=window_start,
                    sat_position=sat_position,
                    imaging_duration=imaging_duration,
                    **kwargs
                )
            else:
                # 短时间间隔：考虑姿态复位
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
        """使用精确模型的可行性检查（考虑姿态复位）

        完整的机动流程：
        1. 前一任务结束时的姿态 -> 对地定向（姿态复位）
        2. 对地定向 -> 当前目标姿态（任务机动）

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

        # 1. 获取前一任务结束时的姿态状态
        current_state = self._get_satellite_attitude_state(
            satellite.id, prev_end_time
        )

        # 2. 计算当前目标姿态
        target_attitude = self._compute_target_attitude(
            satellite, current_target, sat_position
        )

        # 3. 计算姿态复位：从前一姿态回到对地定向
        nadir_attitude = AttitudeState(
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),  # 对地定向 = 单位四元数
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=prev_end_time,
            momentum=None
        )

        reset_maneuver = None
        task_maneuver = None
        total_reset_time = 0.0
        total_task_time = 0.0
        total_slew_angle = 0.0

        try:
            # 3a. 执行姿态复位机动分析（从前一姿态到对地定向）
            reset_maneuver = calc.calculate_slew_maneuver(
                prev_attitude=current_state,
                target_attitude=nadir_attitude,
                current_time=prev_end_time
            )

            if reset_maneuver.feasible:
                total_reset_time = reset_maneuver.total_time
                # 姿态复位的角度
                reset_angle = self._quaternion_rotation_angle(
                    current_state.quaternion, nadir_attitude.quaternion
                )

                # 3b. 执行实际任务机动分析（从对地定向到目标姿态）
                reset_complete_time = prev_end_time + timedelta(seconds=total_reset_time)
                task_maneuver = calc.calculate_slew_maneuver(
                    prev_attitude=nadir_attitude,
                    target_attitude=target_attitude,
                    current_time=reset_complete_time
                )

                if task_maneuver.feasible:
                    total_task_time = task_maneuver.total_time
                    total_slew_angle = task_maneuver.trajectory.rotation_angle
                else:
                    # 任务机动不可行
                    return SlewFeasibilityResult(
                        feasible=False,
                        slew_angle=task_maneuver.trajectory.rotation_angle if task_maneuver.trajectory else 0.0,
                        slew_time=total_reset_time + total_task_time,
                        actual_start=window_start,
                        reason="Task slew maneuver infeasible after reset"
                    )
            else:
                # 姿态复位不可行（这种情况很少见，但需要考虑）
                # 尝试直接机动到目标（不复位）
                logger.debug(f"Reset maneuver infeasible for {satellite.id}, trying direct slew")
                task_maneuver = calc.calculate_slew_maneuver(
                    prev_attitude=current_state,
                    target_attitude=target_attitude,
                    current_time=prev_end_time
                )
                if task_maneuver.feasible:
                    total_task_time = task_maneuver.total_time
                    total_slew_angle = task_maneuver.trajectory.rotation_angle
                else:
                    return SlewFeasibilityResult(
                        feasible=False,
                        slew_angle=task_maneuver.trajectory.rotation_angle if task_maneuver.trajectory else 0.0,
                        slew_time=task_maneuver.total_time if task_maneuver else 0.0,
                        actual_start=window_start,
                        reason="Direct slew maneuver infeasible"
                    )

        except Exception as e:
            logger.warning(f"Precise maneuver calculation failed for {satellite.id}: {e}")
            # 回退到简化模型（不复位，直接机动）
            result = super().check_slew_feasibility(
                satellite.id, prev_target, current_target,
                prev_end_time, window_start, imaging_duration
            )
            # 更新状态到目标姿态
            if result.feasible:
                self._update_satellite_state(
                    satellite_id=satellite.id,
                    time=result.actual_start,
                    attitude=target_attitude,
                    angular_momentum=None
                )
            return result

        # 4. 计算机动后实际开始时间（包括复位时间 + 任务机动时间）
        total_slew_time = total_reset_time + total_task_time
        earliest_start = prev_end_time + timedelta(seconds=total_slew_time)
        actual_start = max(window_start, earliest_start)

        # 5. 检查成像窗口时间
        if imaging_duration > 0:
            actual_end = actual_start + timedelta(seconds=imaging_duration)
            window_end = getattr(current_target, 'time_window_end', None)
            if window_end and actual_end > window_end:
                return SlewFeasibilityResult(
                    feasible=False,
                    slew_angle=total_slew_angle,
                    slew_time=total_slew_time,
                    actual_start=actual_start,
                    reason="Not enough time after slew maneuvers (including reset)"
                )

        # 6. 更新卫星状态跟踪到目标姿态（任务完成后）
        final_momentum = task_maneuver.final_momentum if task_maneuver else None
        self._update_satellite_state(
            satellite_id=satellite.id,
            time=actual_start,
            attitude=target_attitude,
            angular_momentum=final_momentum
        )

        # 7. 构造结果
        result = SlewFeasibilityResult(
            feasible=True,
            slew_angle=total_slew_angle,  # 这是从对地定向到目标的角度
            slew_time=total_slew_time,    # 总时间包括复位 + 任务机动
            actual_start=actual_start,
            reason=None
        )

        # 附加精确模型信息
        if task_maneuver:
            result.energy_consumption = task_maneuver.energy_consumption
            result.momentum_margin = task_maneuver.momentum_margin
            result.precise_result = task_maneuver

        return result

    def _check_nadir_to_target_feasibility(
        self,
        satellite: Satellite,
        current_target: Target,
        window_start: datetime,
        sat_position: Tuple[float, float, float],
        imaging_duration: float,
        **kwargs
    ) -> SlewFeasibilityResult:
        """检查从对地定向到目标的可行性（用于长时间间隔后的任务）

        假设卫星已经回到对地定向，只计算从对地定向到目标的机动。
        这比完整的姿态复位计算更快。

        Args:
            satellite: 卫星对象
            current_target: 当前目标
            window_start: 窗口开始时间
            sat_position: 卫星位置
            imaging_duration: 成像持续时间

        Returns:
            可行性结果
        """
        calc = self._precise_calcs[satellite.id]

        # 1. 初始姿态：对地定向
        nadir_attitude = AttitudeState(
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=window_start,
            momentum=None
        )

        # 2. 计算目标姿态
        target_attitude = self._compute_target_attitude(
            satellite, current_target, sat_position
        )

        # 3. 计算机动角度
        slew_angle = self._quaternion_rotation_angle(
            nadir_attitude.quaternion, target_attitude.quaternion
        )

        try:
            # 4. 执行机动分析（从对地定向到目标）
            maneuver = calc.calculate_slew_maneuver(
                prev_attitude=nadir_attitude,
                target_attitude=target_attitude,
                current_time=window_start
            )
        except Exception as e:
            logger.warning(f"Nadir-to-target maneuver calculation failed: {e}")
            # 使用简化计算
            slew_time = self._estimate_slew_time(satellite, slew_angle)
            actual_start = window_start + timedelta(seconds=slew_time)

            # 更新状态
            self._update_satellite_state(
                satellite_id=satellite.id,
                time=actual_start,
                attitude=target_attitude,
                angular_momentum=None
            )

            return SlewFeasibilityResult(
                feasible=True,
                slew_angle=slew_angle,
                slew_time=slew_time,
                actual_start=actual_start,
                reason=None
            )

        # 5. 检查约束满足
        if not maneuver.feasible:
            reason = "Nadir-to-target maneuver infeasible"
            if maneuver.momentum_check and maneuver.momentum_check.recommendation:
                reason = maneuver.momentum_check.recommendation

            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=slew_angle,
                slew_time=maneuver.total_time,
                actual_start=window_start,
                reason=reason
            )

        # 6. 计算机动后实际开始时间
        actual_start = window_start + timedelta(seconds=maneuver.total_time)

        # 7. 检查成像窗口时间
        if imaging_duration > 0:
            actual_end = actual_start + timedelta(seconds=imaging_duration)
            window_end = getattr(current_target, 'time_window_end', None)
            if window_end and actual_end > window_end:
                return SlewFeasibilityResult(
                    feasible=False,
                    slew_angle=slew_angle,
                    slew_time=maneuver.total_time,
                    actual_start=actual_start,
                    reason="Not enough time after nadir-to-target slew"
                )

        # 8. 更新卫星状态跟踪
        self._update_satellite_state(
            satellite_id=satellite.id,
            time=actual_start,
            attitude=target_attitude,
            angular_momentum=maneuver.final_momentum
        )

        # 9. 构造结果
        result = SlewFeasibilityResult(
            feasible=True,
            slew_angle=slew_angle,
            slew_time=maneuver.total_time,
            actual_start=actual_start,
            reason=None
        )

        # 附加精确模型信息
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

        优先从状态跟踪器获取，其次从内部缓存获取，如果没有则使用默认值。

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
                # 将numpy数组转换为Quaternion对象
                q_data = attitude.get('quaternion', [1, 0, 0, 0])
                if isinstance(q_data, np.ndarray):
                    q = Quaternion(w=q_data[0], x=q_data[1], y=q_data[2], z=q_data[3])
                elif hasattr(q_data, 'w'):  # 已经是Quaternion对象
                    q = q_data
                else:  # 列表或元组
                    q = Quaternion(w=q_data[0], x=q_data[1], y=q_data[2], z=q_data[3])

                # 角速度也可能是numpy数组
                av_data = attitude.get('angular_velocity', [0, 0, 0])
                if isinstance(av_data, np.ndarray):
                    av = AngularVelocity(x=av_data[0], y=av_data[1], z=av_data[2])
                elif hasattr(av_data, 'x'):  # 已经是AngularVelocity对象
                    av = av_data
                else:  # 列表或元组
                    av = AngularVelocity(x=av_data[0], y=av_data[1], z=av_data[2])

                return AttitudeState(
                    quaternion=q,
                    angular_velocity=av,
                    timestamp=timestamp,
                    momentum=momentum
                )

        # 尝试从内部缓存获取
        if satellite_id in self._last_attitudes:
            cached_state = self._last_attitudes[satellite_id]
            # 检查时间间隔：如果超过5分钟，假设卫星已回到对地定向
            if cached_state.timestamp is not None:
                time_diff = abs((timestamp - cached_state.timestamp).total_seconds())
                if time_diff < 300:  # 5分钟 = 300秒
                    return cached_state
            # 时间间隔太长，缓存过期，使用默认状态

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
            sat_position: 卫星位置 (ECEF, meters)

        Returns:
            目标姿态状态
        """
        # 将目标经纬度转换为ECEF坐标
        target_ecef = self._target_to_ecef(target)

        # 计算卫星指向目标的单位向量 (在ECEF坐标系中)
        sat_pos_array = np.array(sat_position)
        target_pos_array = np.array(target_ecef)

        # 视线向量：从卫星指向目标
        los_vector = target_pos_array - sat_pos_array
        los_vector = los_vector / np.linalg.norm(los_vector)

        # 卫星对地指向 (nadir方向，ECEF坐标系中是从卫星指向地心)
        nadir_vector = -sat_pos_array / np.linalg.norm(sat_pos_array)

        # 计算两个向量之间的旋转四元数
        # 从nadir_vector旋转到los_vector
        q = self._rotation_between_vectors(nadir_vector, los_vector)

        return AttitudeState(
            quaternion=q,
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=datetime.now(),
            momentum=None
        )

    def _target_to_ecef(self, target: Target) -> Tuple[float, float, float]:
        """将目标经纬度转换为ECEF坐标

        Args:
            target: 目标对象，包含latitude和longitude属性

        Returns:
            ECEF坐标 (x, y, z) in meters
        """
        import math

        lat_rad = math.radians(getattr(target, 'latitude', 0))
        lon_rad = math.radians(getattr(target, 'longitude', 0))

        # 使用地球平均半径
        R_earth = 6371000.0  # meters

        x = R_earth * math.cos(lat_rad) * math.cos(lon_rad)
        y = R_earth * math.cos(lat_rad) * math.sin(lon_rad)
        z = R_earth * math.sin(lat_rad)

        return (x, y, z)

    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> Quaternion:
        """计算从向量v1旋转到向量v2的四元数

        使用四元数旋转公式，确保最短路径旋转。

        Args:
            v1: 起始单位向量
            v2: 目标单位向量

        Returns:
            旋转四元数
        """
        # 确保单位向量
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # 点积
        dot = np.dot(v1, v2)

        # 如果向量相同，返回单位四元数
        if dot > 0.999999:
            return Quaternion(1.0, 0.0, 0.0, 0.0)

        # 如果向量相反，需要一个垂直于v1的轴
        if dot < -0.999999:
            # 找一个与v1不平行的向量
            if abs(v1[0]) < 0.9:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - v1 * np.dot(v1, axis)
            axis = axis / np.linalg.norm(axis)
            # 180度旋转
            return Quaternion(0.0, axis[0], axis[1], axis[2])

        # 计算旋转轴 (叉积)
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)

        # 旋转角度
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        # 构造四元数 (半角公式)
        half_angle = angle / 2
        sin_half = np.sin(half_angle)

        return Quaternion(
            w=np.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
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
        # 更新外部状态跟踪器（如果可用）
        if self._state_tracker is not None:
            # 将Quaternion对象转换为numpy数组
            q = attitude.quaternion
            if hasattr(q, 'w'):  # Quaternion对象
                q_array = np.array([q.w, q.x, q.y, q.z])
            else:  # 已经是数组
                q_array = np.array(q) if not isinstance(q, np.ndarray) else q

            self._state_tracker.update_after_maneuver(
                satellite_id=satellite_id,
                maneuver_result={
                    'final_quaternion': q_array,
                    'final_momentum': angular_momentum,
                    'timestamp': time
                }
            )

        # 同时更新内部缓存
        self._last_attitudes[satellite_id] = AttitudeState(
            quaternion=attitude.quaternion,
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=time,
            momentum=MomentumState(
                wheel_speeds=np.zeros(4),  # 假设4个飞轮
                wheel_inertias=np.full(4, 0.01),
                total_momentum=angular_momentum
            ) if angular_momentum is not None else None
        )

    def _check_first_task_feasibility(
        self,
        satellite: Satellite,
        current_target: Target,
        window_start: datetime,
        imaging_duration: float,
        **kwargs
    ) -> SlewFeasibilityResult:
        """检查第一个任务的机动可行性

        第一个任务需要从初始对地定向姿态机动到目标姿态。

        Args:
            satellite: 卫星对象
            current_target: 当前目标
            window_start: 窗口开始时间
            imaging_duration: 成像持续时间

        Returns:
            SlewFeasibilityResult: 可行性结果
        """
        calc = self._precise_calcs.get(satellite.id)
        if calc is None:
            # 回退到简化模型
            return self._first_task_result_simple(window_start)

        # 获取卫星位置
        sat_position, sat_velocity = self._get_satellite_position(
            satellite, window_start
        )

        # 1. 初始姿态：对地定向（单位四元数）
        initial_attitude = AttitudeState(
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
            timestamp=window_start,
            momentum=None
        )

        # 2. 计算目标姿态
        target_attitude = self._compute_target_attitude(
            satellite, current_target, sat_position
        )

        # 3. 计算机动角度（四元数之间的旋转角度）
        slew_angle = self._quaternion_rotation_angle(
            initial_attitude.quaternion,
            target_attitude.quaternion
        )

        # 4. 执行精确机动分析
        try:
            maneuver = calc.calculate_slew_maneuver(
                prev_attitude=initial_attitude,
                target_attitude=target_attitude,
                current_time=window_start
            )
        except Exception as e:
            logger.warning(f"First task precise maneuver calculation failed: {e}")
            # 使用简化计算
            slew_time = self._estimate_slew_time(satellite, slew_angle)
            actual_start = window_start + timedelta(seconds=slew_time)

            return SlewFeasibilityResult(
                feasible=True,
                slew_angle=slew_angle,
                slew_time=slew_time,
                actual_start=actual_start,
                reason=None
            )

        # 5. 检查约束满足
        if not maneuver.feasible:
            reason = "First task maneuver infeasible"
            if maneuver.momentum_check and maneuver.momentum_check.recommendation:
                reason = maneuver.momentum_check.recommendation

            return SlewFeasibilityResult(
                feasible=False,
                slew_angle=slew_angle,
                slew_time=maneuver.total_time,
                actual_start=window_start,
                reason=reason
            )

        # 6. 计算机动后实际开始时间
        actual_start = window_start + timedelta(seconds=maneuver.total_time)

        # 7. 检查成像窗口时间
        if imaging_duration > 0:
            actual_end = actual_start + timedelta(seconds=imaging_duration)
            window_end = getattr(current_target, 'time_window_end', None)
            if window_end and actual_end > window_end:
                return SlewFeasibilityResult(
                    feasible=False,
                    slew_angle=slew_angle,
                    slew_time=maneuver.total_time,
                    actual_start=actual_start,
                    reason="Not enough time for first task after slew"
                )

        # 8. 更新卫星状态跟踪
        self._update_satellite_state(
            satellite_id=satellite.id,
            time=actual_start,
            attitude=target_attitude,
            angular_momentum=maneuver.final_momentum
        )

        # 9. 构造结果
        result = SlewFeasibilityResult(
            feasible=True,
            slew_angle=slew_angle,
            slew_time=maneuver.total_time,
            actual_start=actual_start,
            reason=None
        )
        result.energy_consumption = maneuver.energy_consumption
        result.momentum_margin = maneuver.momentum_margin
        result.precise_result = maneuver

        return result

    def _quaternion_rotation_angle(self, q1: Quaternion, q2: Quaternion) -> float:
        """计算两个四元数之间的旋转角度（度）

        Args:
            q1: 起始四元数
            q2: 目标四元数

        Returns:
            旋转角度（度）
        """
        # 计算相对四元数：q_rel = q2 * q1^-1
        # q1^-1 = (w, -x, -y, -z) for unit quaternion
        q1_conj = Quaternion(q1.w, -q1.x, -q1.y, -q1.z)

        # 四元数乘法: q2 * q1_conj
        w = q2.w * q1_conj.w - q2.x * q1_conj.x - q2.y * q1_conj.y - q2.z * q1_conj.z
        x = q2.w * q1_conj.x + q2.x * q1_conj.w + q2.y * q1_conj.z - q2.z * q1_conj.y
        y = q2.w * q1_conj.y - q2.x * q1_conj.z + q2.y * q1_conj.w + q2.z * q1_conj.x
        z = q2.w * q1_conj.z + q2.x * q1_conj.y - q2.y * q1_conj.x + q2.z * q1_conj.w

        # 旋转角度 = 2 * acos(|w|)
        # 确保数值稳定性
        w_abs = abs(w)
        if w_abs > 1.0:
            w_abs = 1.0

        angle_rad = 2.0 * np.arccos(w_abs)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _estimate_slew_time(self, satellite: Satellite, slew_angle: float) -> float:
        """简化估算机动时间（回退使用）

        Args:
            satellite: 卫星对象
            slew_angle: 机动角度（度）

        Returns:
            估算的机动时间（秒）
        """
        agility = getattr(satellite.capabilities, 'agility', {}) or {}
        max_slew_rate = agility.get('max_slew_rate', 3.0)  # deg/s
        settling_time = agility.get('settling_time', 5.0)  # seconds

        if max_slew_rate <= 0:
            return settling_time

        return (slew_angle / max_slew_rate) + settling_time

    def _first_task_result_simple(self, window_start: datetime) -> SlewFeasibilityResult:
        """第一个任务的简化结果（回退使用）"""
        settling_time = 5.0  # 默认稳定时间
        return SlewFeasibilityResult(
            feasible=True,
            slew_angle=0.0,  # 简化模型保持0，精确模型会计算实际角度
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

    def check_slew(
        self,
        sat_id: str,
        prev_target: Optional[Target],
        current_target: Target,
        prev_end_time: datetime,
        window_start: datetime,
        imaging_duration: float,
        use_simplified: bool = False,
    ) -> 'ConstraintResult':
        """检查姿态机动约束（与SlewChecker.check_slew接口兼容）

        这个方法将内部的check_slew_feasibility结果转换为ConstraintResult格式，
        以便与ConstraintChecker._check_slew方法兼容。

        Args:
            sat_id: 卫星ID
            prev_target: 上一个目标 (None表示第一个任务)
            current_target: 当前目标
            prev_end_time: 上一个任务结束时间
            window_start: 当前窗口开始时间
            imaging_duration: 成像持续时间 (秒)
            use_simplified: 使用简化模型 (保留参数，但精确模型始终使用精确计算)

        Returns:
            ConstraintResult: 约束检查结果
        """
        from scheduler.common.constraint_checker import ConstraintResult, ConstraintType

        # 调用内部的精确可行性检查
        slew_result = self.check_slew_feasibility(
            satellite_id=sat_id,
            prev_target=prev_target,
            current_target=current_target,
            prev_end_time=prev_end_time,
            window_start=window_start,
            imaging_duration=imaging_duration
        )

        # 转换为ConstraintResult格式
        result = ConstraintResult()

        if not slew_result.feasible:
            result.feasible = False
            result.add_violation(ConstraintType.SLEW, slew_result.reason or "Slew not feasible")

        # 复制机动计算结果
        result.slew_angle = slew_result.slew_angle
        result.slew_time = slew_result.slew_time
        result.actual_start = slew_result.actual_start

        return result

    def initialize_satellite(self, satellite: 'Satellite') -> None:
        """初始化卫星的姿态机动计算器

        与SlewChecker.initialize_satellite接口兼容。

        Args:
            satellite: 卫星对象
        """
        # 如果已经有该卫星的精确计算器，不需要重新初始化
        if satellite.id in self._precise_calcs:
            return

        # 创建新的精确计算器
        config = self._extract_dynamics_config(satellite)
        self._precise_calcs[satellite.id] = PreciseSlewCalculator(
            config=config,
            use_precise=True
        )

        # 清空缓存
        if satellite.id in self._last_attitudes:
            del self._last_attitudes[satellite.id]
