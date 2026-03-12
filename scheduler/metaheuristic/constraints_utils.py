"""
元启发式算法共享的约束工具模块

提供统一的约束检查功能，供GA/ACO/PSO/SA/Tabu等元启发式算法使用。
确保所有算法都具备与贪心算法相同的约束检查能力。

重构说明：此模块现在委托给 scheduler.common.constraint_checker 中的
统一约束检查器，消除代码重复。

更新：已统一使用批量约束检查器（UnifiedBatchConstraintChecker），
与greedy调度器保持一致，支持向量化批量优化。
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.models import Mission, Satellite, Target
from scheduler.base_scheduler import ScheduledTask, TaskFailureReason
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile

# 批量约束检查器导入 - 与greedy调度器保持一致
from scheduler.constraints.unified_batch_constraint_checker import (
    UnifiedBatchConstraintChecker, UnifiedBatchCandidate
)

# 传统检查器保留用于向后兼容
from scheduler.constraints import SlewConstraintChecker, SlewFeasibilityResult
from scheduler.constraints.saa_constraint_checker import SAAConstraintChecker
from scheduler.constraints import UnifiedSpatiotemporalChecker, SpatiotemporalCheckResult
from scheduler.constraints import UnifiedManeuverChecker, ManeuverCheckResult
from core.dynamics.attitude_manager import AttitudeManagementConfig

logger = logging.getLogger(__name__)


class MetaheuristicConstraintChecker:
    """
    元启发式算法专用约束检查器

    将贪心算法中的约束检查逻辑提取为可复用组件，
    供所有元启发式算法统一使用。

    更新：已统一使用批量约束检查器（UnifiedBatchConstraintChecker），
    与greedy调度器保持一致，支持向量化批量优化。
    """

    def __init__(self, mission: Mission, config: Dict[str, Any] = None):
        """
        初始化约束检查器

        Args:
            mission: 任务场景对象
            config: 配置参数
                - consider_power: 是否考虑电量约束（默认True）
                - consider_storage: 是否考虑存储约束（默认True）
        """
        # 高精度要求：禁止使用简化模式
        if config and config.get('use_simplified_slew', False):
            raise ValueError(
                "use_simplified_slew=True is not allowed. "
                "High precision mode requires exact calculations."
            )

        self.mission = mission
        self.config = config or {}

        self.consider_power = self.config.get('consider_power', True)
        self.consider_storage = self.config.get('consider_storage', True)

        # 初始化成像时间计算器
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=self.config.get('min_imaging_duration'),
            max_duration=self.config.get('max_imaging_duration'),
            default_duration=self.config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(self.config.get('power_coefficients'))

        # 强制使用批量约束检查器（高精度要求）
        self._batch_constraint_checker = UnifiedBatchConstraintChecker(
            mission=mission,
            use_precise_model=True,
            consider_power=self.consider_power,
            consider_storage=self.consider_storage
        )

        # 传统约束检查器（向后兼容）
        self._slew_checker: Optional[SlewConstraintChecker] = None
        self._saa_checker: Optional[SAAConstraintChecker] = None
        self._unified_checker: Optional[UnifiedSpatiotemporalChecker] = None
        self._maneuver_checker: Optional[UnifiedManeuverChecker] = None

        # 卫星资源使用跟踪
        self._sat_resource_usage: Dict[str, Dict[str, Any]] = {}

        # 姿态计算器配置
        self._enable_attitude_calculation = self.config.get('enable_attitude_calculation', True)
        self._use_unified_constraints = self.config.get('use_unified_constraints', False)
        self._use_unified_maneuver = self.config.get('use_unified_maneuver', False)

    def initialize(self) -> None:
        """初始化约束检查器和资源跟踪"""
        self._initialize_resource_tracking()

        # 初始化批量约束检查器（高精度要求）
        logger.info("Initialized UnifiedBatchConstraintChecker for batch constraint checking")

        # 初始化传统约束检查器
        self._initialize_slew_checker()
        self._initialize_saa_checker()
        self._initialize_unified_checker()
        self._initialize_maneuver_checker()

    def _initialize_resource_tracking(self) -> None:
        """初始化卫星资源使用跟踪"""
        self._sat_resource_usage = {}
        for sat in self.mission.satellites:
            current_power = getattr(sat, 'current_power', None)
            try:
                if current_power is not None and current_power > 0:
                    power = current_power
                else:
                    power = sat.capabilities.power_capacity
            except TypeError:
                power = sat.capabilities.power_capacity

            self._sat_resource_usage[sat.id] = {
                'power': power,
                'storage': 0.0,
                'last_task_end': self.mission.start_time,
                'scheduled_tasks': [],
                'last_target': None,
            }

    def _initialize_slew_checker(self) -> None:
        """初始化机动约束检查器"""
        # 高精度要求：始终初始化精确检查器
        from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType

        attitude_calc = AttitudeCalculator(
            propagator_type=PropagatorType.SGP4
        )

        self._slew_checker = SlewConstraintChecker(
            self.mission,
            attitude_calc
        )

        for sat in self.mission.satellites:
            self._slew_checker.initialize_satellite(sat)

    def _initialize_saa_checker(self) -> None:
        """初始化SAA约束检查器"""
        # 高精度要求：始终初始化精确检查器
        from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType

        attitude_calc = AttitudeCalculator(
            propagator_type=PropagatorType.SGP4
        )

        self._saa_checker = SAAConstraintChecker(
            self.mission,
            attitude_calc
        )

        for sat in self.mission.satellites:
            self._saa_checker.initialize_satellite(sat)

    def _initialize_unified_checker(self) -> None:
        """初始化统一时空约束检查器"""
        if not self._use_unified_constraints:
            return

        self._unified_checker = UnifiedSpatiotemporalChecker(
            mission=self.mission,
            slew_checker=self._slew_checker
        )

    def _initialize_maneuver_checker(self) -> None:
        """初始化统一机动约束检查器"""
        if not self._use_unified_maneuver:
            return

        attitude_config = AttitudeManagementConfig(
            max_slew_rate=self.config.get('max_slew_rate', 3.0),
            settling_time=self.config.get('settling_time', 5.0),
            idle_time_threshold=self.config.get('idle_time_threshold', 300.0),
            soc_threshold=self.config.get('soc_threshold', 0.30),
            enable_sun_pointing_optimization=self.config.get('enable_sun_pointing_optimization', True),
        )

        self._maneuver_checker = UnifiedManeuverChecker(
            mission=self.mission,
            config=attitude_config
        )

    def check_maneuver_placement(
        self,
        sat_id: str,
        target: Target,
        window_start: datetime,
        window_end: datetime,
        imaging_duration: float,
        satellite_position: Tuple[float, float, float],
        task_id: Optional[str] = None,
        from_mode: Any = None,
        to_mode: Any = None
    ) -> ManeuverCheckResult:
        """
        使用统一机动约束检查器检查任务放置

        Args:
            sat_id: 卫星ID
            target: 目标对象
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            imaging_duration: 成像时长（秒）
            satellite_position: 卫星ECEF位置（米）
            task_id: 任务ID（可选）
            from_mode: 起始姿态模式（可选）
            to_mode: 目标姿态模式（可选）

        Returns:
            ManeuverCheckResult: 机动约束检查结果
        """
        if self._maneuver_checker is None:
            logger.warning("Maneuver checker not initialized, using legacy checks")
            # 返回一个模拟的不可行结果
            return ManeuverCheckResult(
                feasible=False,
                conflict_reason="Maneuver checker not initialized"
            )

        from core.dynamics.attitude_mode import AttitudeMode

        # 转换姿态模式
        actual_from_mode = from_mode if from_mode else AttitudeMode.NADIR_POINTING
        actual_to_mode = to_mode if to_mode else AttitudeMode.IMAGING

        return self._maneuver_checker.check_maneuver_placement(
            satellite_id=sat_id,
            target=target,
            window_start=window_start,
            window_end=window_end,
            imaging_duration=imaging_duration,
            satellite_position=satellite_position,
            task_id=task_id,
            from_mode=actual_from_mode,
            to_mode=actual_to_mode
        )

    def commit_maneuver_task(
        self,
        sat_id: str,
        task_id: str,
        target_id: str,
        actual_start: datetime,
        actual_end: datetime,
        target: Optional[Target] = None,
        end_mode: Any = None
    ) -> None:
        """
        提交任务到统一机动约束检查器

        Args:
            sat_id: 卫星ID
            task_id: 任务ID
            target_id: 目标ID
            actual_start: 实际开始时间
            actual_end: 实际结束时间
            target: 目标对象（可选）
            end_mode: 任务结束姿态模式（可选）
        """
        if self._maneuver_checker is not None:
            from core.dynamics.attitude_mode import AttitudeMode
            actual_end_mode = end_mode if end_mode else AttitudeMode.NADIR_POINTING

            self._maneuver_checker.commit_task(
                satellite_id=sat_id,
                task_id=task_id,
                target_id=target_id,
                actual_start=actual_start,
                actual_end=actual_end,
                target=target,
                end_mode=actual_end_mode
            )

    def reset_maneuver_checker(self) -> None:
        """重置统一机动约束检查器状态"""
        if self._maneuver_checker is not None:
            self._maneuver_checker.reset()

    def check_task_feasibility(
        self,
        sat_id: str,
        target: Target,
        window_start: datetime,
        window_end: datetime,
        imaging_mode: Any = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        检查单个任务分配的可行性

        优先使用批量约束检查器（UnifiedBatchConstraintChecker），
        与GreedyScheduler保持一致，支持向量化批量优化。

        Args:
            sat_id: 卫星ID
            target: 目标对象
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            imaging_mode: 成像模式（可选）

        Returns:
            Tuple of (是否可行, 失败原因, 额外信息)
        """
        sat = self.mission.get_satellite_by_id(sat_id)
        if not sat:
            return False, "Satellite not found", None

        # 选择成像模式
        if imaging_mode is None:
            imaging_mode = self._select_imaging_mode(sat, target)

        # 计算成像时长
        imaging_duration = self._imaging_calculator.calculate(target, imaging_mode, sat)

        # 获取资源使用情况
        usage = self._sat_resource_usage.get(sat_id, {})
        last_task_end = usage.get('last_task_end', self.mission.start_time)
        prev_target = usage.get('last_target')
        scheduled_tasks = usage.get('scheduled_tasks', [])

        # 使用批量约束检查器（高精度要求：强制使用精确计算）
        # 计算资源需求
        power_needed = 0.0
        storage_produced = 0.0
        if self.consider_power:
            power_coefficient = 0.1  # 默认系数
            power_needed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)
        if self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_produced = self._imaging_calculator.get_storage_consumption(target, imaging_mode, data_rate)

        # 获取卫星位置（使用默认值）
        R_earth = 6371000.0  # 地球半径（米）
        alt = 500000.0  # 默认高度500km
        sat_position = (R_earth + alt, 0.0, 0.0)
        sat_velocity = (0.0, 7000.0, 0.0)  # 默认速度约7km/s

        # 构建批量检查候选
        candidate = UnifiedBatchCandidate(
            sat_id=sat_id,
            satellite=sat,
            target=target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=last_task_end if last_task_end else self.mission.start_time,
            imaging_duration=imaging_duration,
            prev_target=prev_target,
            power_needed=power_needed,
            storage_produced=storage_produced,
            sat_position=sat_position,
            sat_velocity=sat_velocity
        )

        # 构建卫星状态字典
        satellite_states = {
            sat_id: {
                'power': usage.get('power', sat.capabilities.power_capacity),
                'storage': usage.get('storage', 0.0)
            }
        }

        # 构建已调度任务列表
        existing_tasks = [
            {
                'satellite_id': sat_id,
                'start_time': task['start'],
                'end_time': task['end']
            }
            for task in scheduled_tasks
        ]

        # 执行批量约束检查
        results = self._batch_constraint_checker.check_all_constraints_batch(
            candidates=[candidate],
            existing_tasks=existing_tasks,
            satellite_states=satellite_states,
            early_termination=True
        )

        if not results or not results[0].feasible:
            result = results[0] if results else None
            reason = result.reason if result and result.reason else "Constraint violation"
            return False, reason, None

        result = results[0]
        actual_start = result.slew_result.actual_start if result.slew_result else window_start
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        # 构建额外信息
        info = {
            'imaging_mode': imaging_mode,
            'imaging_duration': imaging_duration,
            'slew_angle': result.slew_result.slew_angle if result.slew_result else 0.0,
            'slew_time': result.slew_result.slew_time if result.slew_result else 0.0,
            'actual_start': actual_start,
            'actual_end': actual_end,
        }

        return True, None, info

    def _select_imaging_mode(self, sat: Satellite, target: Target):
        """选择成像模式"""
        from core.models import ImagingMode

        modes = sat.capabilities.imaging_modes
        if not modes:
            return ImagingMode.PUSH_BROOM

        mode = modes[0]
        if hasattr(mode, '_mock_name') or not isinstance(mode, (ImagingMode, str)):
            return ImagingMode.PUSH_BROOM
        return mode if isinstance(mode, ImagingMode) else ImagingMode(mode)

    def check_task_placement(
        self,
        sat_id: str,
        target: Target,
        window_start: datetime,
        window_end: datetime,
        imaging_duration: float,
        task_id: Optional[str] = None
    ) -> SpatiotemporalCheckResult:
        """
        使用批量约束检查器检查任务放置（与greedy调度器一致）

        优先使用 UnifiedBatchConstraintChecker 进行向量化批量优化：
        - 时间窗口冲突（与已调度任务）
        - 机动能力约束
        - 机动时间约束
        - SAA约束
        - 资源约束

        Args:
            sat_id: 卫星ID
            target: 目标对象
            window_start: 窗口开始时间
            window_end: 窗口结束时间
            imaging_duration: 成像时长（秒）
            task_id: 任务ID（可选，用于冲突诊断）

        Returns:
            SpatiotemporalCheckResult: 统一约束检查结果
        """
        # 优先使用批量约束检查器（与greedy调度器一致）
        if self._batch_constraint_checker is not None:
            sat = self.mission.get_satellite_by_id(sat_id)
            if not sat:
                return SpatiotemporalCheckResult(
                    feasible=False,
                    conflict_reason="Satellite not found"
                )

            # 获取资源使用情况
            usage = self._sat_resource_usage.get(sat_id, {})
            last_task_end = usage.get('last_task_end', self.mission.start_time)
            prev_target = usage.get('last_target')
            scheduled_tasks = usage.get('scheduled_tasks', [])

            # 计算资源需求
            power_needed = 0.0
            storage_produced = 0.0
            if self.consider_power:
                power_coefficient = 0.1
                power_needed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)
            if self.consider_storage:
                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                storage_produced = self._imaging_calculator.get_storage_consumption(target, None, data_rate)

            # 获取卫星位置（使用默认值）
            import math
            R_earth = 6371000.0  # 地球半径（米）
            alt = 500000.0  # 默认高度500km
            sat_position = (R_earth + alt, 0.0, 0.0)
            sat_velocity = (0.0, 7000.0, 0.0)  # 默认速度约7km/s

            # 构建批量检查候选
            candidate = UnifiedBatchCandidate(
                sat_id=sat_id,
                satellite=sat,
                target=target,
                window_start=window_start,
                window_end=window_end,
                prev_end_time=last_task_end if last_task_end else self.mission.start_time,
                imaging_duration=imaging_duration,
                prev_target=prev_target,
                power_needed=power_needed,
                storage_produced=storage_produced,
                sat_position=sat_position,
                sat_velocity=sat_velocity
            )

            # 构建卫星状态字典
            satellite_states = {
                sat_id: {
                    'power': usage.get('power', sat.capabilities.power_capacity),
                    'storage': usage.get('storage', 0.0)
                }
            }

            # 构建已调度任务列表
            existing_tasks = [
                {
                    'satellite_id': sat_id,
                    'start_time': task['start'],
                    'end_time': task['end']
                }
                for task in scheduled_tasks
            ]

            # 执行批量约束检查
            results = self._batch_constraint_checker.check_all_constraints_batch(
                candidates=[candidate],
                existing_tasks=existing_tasks,
                satellite_states=satellite_states,
                early_termination=True
            )

            if not results:
                return SpatiotemporalCheckResult(
                    feasible=False,
                    conflict_reason="Batch check failed"
                )

            result = results[0]

            return SpatiotemporalCheckResult(
                feasible=result.feasible,
                actual_start=result.slew_result.actual_start if result.slew_result else window_start,
                actual_end=result.slew_result.actual_start + timedelta(seconds=imaging_duration) if result.slew_result else window_end,
                slew_angle=result.slew_result.slew_angle if result.slew_result else 0.0,
                slew_time=result.slew_result.slew_time if result.slew_result else 0.0,
                window_available=result.slew_result is not None and result.slew_result.feasible if result.slew_result else False,
                slew_feasible=result.slew_feasible,
                conflict_reason=result.reason
            )

    def commit_task_placement(
        self,
        sat_id: str,
        task_id: str,
        target_id: str,
        actual_start: datetime,
        actual_end: datetime,
        target: Optional[Target] = None
    ) -> None:
        """
        提交任务到统一约束检查器

        在确认任务可行后，调用此方法更新检查器的内部状态，
        使后续的任务检查能正确考虑此任务。

        Args:
            sat_id: 卫星ID
            task_id: 任务ID
            target_id: 目标ID
            actual_start: 实际开始时间
            actual_end: 实际结束时间
            target: 目标对象（可选）
        """
        if self._unified_checker is not None:
            self._unified_checker.commit_task(
                satellite_id=sat_id,
                task_id=task_id,
                target_id=target_id,
                actual_start=actual_start,
                actual_end=actual_end,
                target=target
            )

        # 同时更新资源跟踪
        if target:
            imaging_mode = self._select_imaging_mode(
                self.mission.get_satellite_by_id(sat_id), target
            )
            self.update_resource_usage(sat_id, target, imaging_mode, actual_start, actual_end)

    def reset_unified_checker(self) -> None:
        """重置统一约束检查器状态"""
        if self._unified_checker is not None:
            self._unified_checker.reset()

    def update_resource_usage(
        self,
        sat_id: str,
        target: Target,
        imaging_mode: Any,
        actual_start: datetime,
        actual_end: datetime
    ) -> Dict[str, Any]:
        """
        更新资源使用状态

        Returns:
            更新后的资源使用信息
        """
        usage = self._sat_resource_usage.get(sat_id)
        if usage is None:
            return {}

        sat = self.mission.get_satellite_by_id(sat_id)
        if not sat:
            return usage

        # 计算资源消耗
        imaging_duration = (actual_end - actual_start).total_seconds()

        # 电量消耗
        if self.consider_power:
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_consumed = sat.capabilities.power_capacity * power_coefficient * (imaging_duration / 3600)
            usage['power'] = usage.get('power', sat.capabilities.power_capacity) - power_consumed

        # 存储消耗
        if self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_used = self._imaging_calculator.get_storage_consumption(
                target, imaging_mode, data_rate
            )
            usage['storage'] = usage.get('storage', 0) + storage_used

        # 更新状态
        usage['last_task_end'] = actual_end
        usage['last_target'] = target
        usage['scheduled_tasks'].append({
            'start': actual_start,
            'end': actual_end,
            'target_id': getattr(target, 'id', None),
        })

        return usage

    def calculate_assignment_score(
        self,
        sat_id: str,
        target: Target,
        actual_start: datetime,
        priority: int = 5
    ) -> float:
        """
        计算任务分配的评分（用于元启发式算法的适应度计算）

        Args:
            sat_id: 卫星ID
            target: 目标对象
            actual_start: 实际开始时间
            priority: 目标优先级

        Returns:
            评分值（越高越好）
        """
        score = 0.0

        # 优先早开始
        time_from_start = (actual_start - self.mission.start_time).total_seconds()
        score -= time_from_start / 3600.0

        # 优先级奖励
        score += priority * 10

        # 资源余量奖励
        usage = self._sat_resource_usage.get(sat_id, {})
        sat = self.mission.get_satellite_by_id(sat_id)

        if sat:
            power_capacity = sat.capabilities.power_capacity
            storage_capacity = sat.capabilities.storage_capacity

            if power_capacity > 0:
                power_ratio = usage.get('power', power_capacity) / power_capacity
                score += power_ratio * 5

            if storage_capacity > 0:
                storage_ratio = 1.0 - (usage.get('storage', 0) / storage_capacity)
                score += storage_ratio * 5

        return score

    def create_scheduled_task(
        self,
        task_id: str,
        sat_id: str,
        target: Target,
        imaging_mode: Any,
        actual_start: datetime,
        actual_end: datetime,
        slew_result: SlewFeasibilityResult
    ) -> ScheduledTask:
        """
        创建ScheduledTask对象

        Args:
            task_id: 任务ID
            sat_id: 卫星ID
            target: 目标对象
            imaging_mode: 成像模式
            actual_start: 实际开始时间
            actual_end: 实际结束时间
            slew_result: 机动结果

        Returns:
            ScheduledTask对象
        """
        usage = self._sat_resource_usage.get(sat_id, {})
        power_before = usage.get('power', 0)
        storage_before = usage.get('storage', 0)

        # 更新资源使用
        usage = self.update_resource_usage(sat_id, target, imaging_mode, actual_start, actual_end)

        # 创建任务对象
        scheduled_task = ScheduledTask(
            task_id=task_id,
            satellite_id=sat_id,
            target_id=getattr(target, 'id', task_id),
            imaging_start=actual_start,
            imaging_end=actual_end,
            imaging_mode=imaging_mode.value if hasattr(imaging_mode, 'value') else str(imaging_mode),
            slew_angle=slew_result.slew_angle,
            slew_time=slew_result.slew_time,
            storage_before=storage_before,
            storage_after=usage.get('storage', storage_before),
            power_before=power_before,
            power_after=usage.get('power', power_before),
        )

        # 计算姿态角（如果启用）- 与简化模式解耦，只要启用就计算
        if self._enable_attitude_calculation:
            self._calculate_and_apply_attitude(sat_id, target, actual_start, scheduled_task)

        return scheduled_task

    def _calculate_and_apply_attitude(
        self,
        sat_id: str,
        target: Target,
        imaging_time: datetime,
        scheduled_task: ScheduledTask
    ) -> None:
        """计算并应用姿态角"""
        try:
            from core.dynamics.attitude_calculator import AttitudeCalculator, PropagatorType

            sat = self.mission.get_satellite_by_id(sat_id)
            if not sat or not hasattr(target, 'latitude') or not hasattr(target, 'longitude'):
                return

            # 尝试使用预计算位置缓存（如果可用）
            attitude = self._calculate_attitude_from_cache(sat, target, imaging_time)

            if attitude is None:
                # 回退到实时计算
                attitude_calc = AttitudeCalculator(
                    satellite=sat,
                    propagator_type=PropagatorType.BATCH_PRECOMPUTED
                )
                attitude = attitude_calc.calculate_attitude(
                    satellite=sat,
                    target=target,
                    imaging_time=imaging_time
                )

            if attitude:
                scheduled_task.roll_angle = attitude.roll
                scheduled_task.pitch_angle = attitude.pitch
                scheduled_task.yaw_angle = attitude.yaw
                scheduled_task.attitude_coordinate_system = attitude.coordinate_system

        except Exception as e:
            logger.warning(f"Failed to calculate attitude: {e}")

    def reset(self) -> None:
        """重置资源使用状态（用于重新评估解）"""
        self._initialize_resource_tracking()
        self.reset_unified_checker()
        self.reset_maneuver_checker()

    def get_resource_usage(self, sat_id: str) -> Dict[str, Any]:
        """获取卫星资源使用情况"""
        return self._sat_resource_usage.get(sat_id, {}).copy()

    def get_failure_reason(
        self,
        target: Target,
        imaging_mode: Any = None
    ) -> TaskFailureReason:
        """
        分析任务失败原因

        Args:
            target: 目标对象
            imaging_mode: 成像模式

        Returns:
            失败原因枚举
        """
        # 检查存储约束
        if self.consider_storage:
            for sat_id, usage in self._sat_resource_usage.items():
                sat = self.mission.get_satellite_by_id(sat_id)
                if not sat:
                    continue

                if imaging_mode is None:
                    imaging_mode = self._select_imaging_mode(sat, target)

                data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
                storage_needed = self._imaging_calculator.get_storage_consumption(
                    target, imaging_mode, data_rate
                )

                current_storage = usage.get('storage', 0)
                if current_storage + storage_needed > sat.capabilities.storage_capacity:
                    return TaskFailureReason.STORAGE_CONSTRAINT

        # 检查电量约束
        if self.consider_power:
            for sat_id, usage in self._sat_resource_usage.items():
                sat = self.mission.get_satellite_by_id(sat_id)
                if not sat:
                    continue

                power_ratio = usage.get('power', sat.capabilities.power_capacity) / sat.capabilities.power_capacity
                if power_ratio < 0.1:
                    return TaskFailureReason.POWER_CONSTRAINT

        return TaskFailureReason.NO_VISIBLE_WINDOW
