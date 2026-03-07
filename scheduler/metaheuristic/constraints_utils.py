"""
元启发式算法共享的约束工具模块

提供统一的约束检查功能，供GA/ACO/PSO/SA/Tabu等元启发式算法使用。
确保所有算法都具备与贪心算法相同的约束检查能力。
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from core.models import Mission, Satellite, Target
from scheduler.base_scheduler import ScheduledTask, TaskFailureReason
from payload.imaging_time_calculator import ImagingTimeCalculator, PowerProfile
from scheduler.constraints import SlewConstraintChecker, SlewFeasibilityResult
from scheduler.constraints.saa_constraint_checker import SAAConstraintChecker

logger = logging.getLogger(__name__)


class MetaheuristicConstraintChecker:
    """
    元启发式算法专用约束检查器

    将贪心算法中的约束检查逻辑提取为可复用组件，
    供所有元启发式算法统一使用。
    """

    def __init__(self, mission: Mission, config: Dict[str, Any] = None):
        """
        初始化约束检查器

        Args:
            mission: 任务场景对象
            config: 配置参数
                - consider_power: 是否考虑电量约束（默认True）
                - consider_storage: 是否考虑存储约束（默认True）
                - use_simplified_slew: 是否使用简化机动检查（默认False）
        """
        self.mission = mission
        self.config = config or {}

        self.consider_power = self.config.get('consider_power', True)
        self.consider_storage = self.config.get('consider_storage', True)
        self.use_simplified_slew = self.config.get('use_simplified_slew', False)

        # 初始化成像时间计算器
        self._imaging_calculator = ImagingTimeCalculator(
            min_duration=self.config.get('min_imaging_duration'),
            max_duration=self.config.get('max_imaging_duration'),
            default_duration=self.config.get('default_imaging_duration')
        )
        self._power_profile = PowerProfile(self.config.get('power_coefficients'))

        # 约束检查器（延迟初始化）
        self._slew_checker: Optional[SlewConstraintChecker] = None
        self._saa_checker: Optional[SAAConstraintChecker] = None

        # 卫星资源使用跟踪
        self._sat_resource_usage: Dict[str, Dict[str, Any]] = {}

        # 姿态计算器配置
        self._enable_attitude_calculation = self.config.get('enable_attitude_calculation', True)

    def initialize(self) -> None:
        """初始化约束检查器和资源跟踪"""
        self._initialize_resource_tracking()
        self._initialize_slew_checker()
        self._initialize_saa_checker()

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
        if self.use_simplified_slew:
            return

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
        if self.use_simplified_slew:
            return

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

        # 检查卫星能力
        if not self._check_satellite_capability(sat, target):
            return False, "Satellite capability mismatch", None

        # 选择成像模式
        if imaging_mode is None:
            imaging_mode = self._select_imaging_mode(sat, target)

        # 计算成像时长
        imaging_duration = self._imaging_calculator.calculate(target, imaging_mode, sat)

        # 检查资源约束
        if not self._check_resource_constraints(sat, target, imaging_mode):
            return False, "Resource constraint violation", None

        # 检查机动约束
        slew_result = self._check_slew_constraint(sat_id, target, window_start, imaging_duration)
        if not slew_result or not slew_result.feasible:
            return False, "Slew constraint violation", None

        # 检查SAA约束
        actual_start = slew_result.actual_start
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        if not self._check_saa_constraint(sat_id, actual_start, actual_end):
            return False, "SAA constraint violation", None

        # 检查时间冲突
        if self._has_time_conflict(sat_id, actual_start, actual_end):
            return False, "Time conflict", None

        # 检查窗口是否足够长
        if actual_end > window_end:
            return False, "Window too short", None

        # 构建额外信息
        info = {
            'imaging_mode': imaging_mode,
            'imaging_duration': imaging_duration,
            'slew_result': slew_result,
            'actual_start': actual_start,
            'actual_end': actual_end,
        }

        return True, None, info

    def _check_satellite_capability(self, sat: Satellite, target: Target) -> bool:
        """检查卫星是否能执行任务"""
        if not sat.capabilities.imaging_modes:
            return False

        required_resolution = getattr(target, 'resolution_required', None)
        if required_resolution is not None:
            sat_resolution = getattr(sat.capabilities, 'resolution', None)
            try:
                if sat_resolution is not None and required_resolution is not None:
                    if sat_resolution > required_resolution:
                        return False
            except TypeError:
                pass

        return True

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

    def _check_resource_constraints(
        self,
        sat: Satellite,
        target: Target,
        imaging_mode: Any
    ) -> bool:
        """检查资源约束"""
        usage = self._sat_resource_usage.get(sat.id, {})

        # 电量约束
        if self.consider_power:
            duration = self._imaging_calculator.calculate(target, imaging_mode, sat)
            power_coefficient = self._power_profile.get_coefficient_for_mode(imaging_mode)
            power_needed = sat.capabilities.power_capacity * power_coefficient * (duration / 3600)

            if usage.get('power', 0) < power_needed:
                return False

        # 存储约束
        if self.consider_storage:
            data_rate = getattr(sat.capabilities, 'data_rate', 300.0)
            storage_needed = self._imaging_calculator.get_storage_consumption(
                target, imaging_mode, data_rate
            )

            current_storage = usage.get('storage', 0)
            capacity = sat.capabilities.storage_capacity

            if current_storage + storage_needed > capacity:
                return False

        return True

    def _check_slew_constraint(
        self,
        sat_id: str,
        target: Target,
        window_start: datetime,
        imaging_duration: float
    ) -> Optional[SlewFeasibilityResult]:
        """检查机动约束"""
        usage = self._sat_resource_usage.get(sat_id, {})
        last_task_end = usage.get('last_task_end', self.mission.start_time)
        prev_target = usage.get('last_target')

        if self.use_simplified_slew:
            return self._get_simplified_slew_result(sat_id, last_task_end, window_start)

        if self._slew_checker is None:
            return None

        return self._slew_checker.check_slew_feasibility(
            sat_id, prev_target, target, last_task_end, window_start, imaging_duration
        )

    def _get_simplified_slew_result(
        self,
        sat_id: str,
        prev_end_time: datetime,
        window_start: datetime
    ) -> SlewFeasibilityResult:
        """获取简化的机动检查结果"""
        slew_time = 30.0  # 默认30秒
        earliest_start = prev_end_time + timedelta(seconds=slew_time)
        actual_start = max(window_start, earliest_start)

        return SlewFeasibilityResult(
            feasible=True,
            slew_angle=0.0,
            slew_time=slew_time,
            actual_start=actual_start,
            reason=None
        )

    def _check_saa_constraint(
        self,
        sat_id: str,
        window_start: datetime,
        window_end: datetime
    ) -> bool:
        """检查SAA约束"""
        if self._saa_checker is None:
            return True

        result = self._saa_checker.check_window_feasibility(sat_id, window_start, window_end)
        return result.feasible

    def _has_time_conflict(self, sat_id: str, start: datetime, end: datetime) -> bool:
        """检查时间冲突"""
        usage = self._sat_resource_usage.get(sat_id, {})
        scheduled_tasks = usage.get('scheduled_tasks', [])

        for task in scheduled_tasks:
            if not (end <= task['start'] or start >= task['end']):
                return True
        return False

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

        # 计算姿态角（如果启用）
        if self._enable_attitude_calculation and not self.use_simplified_slew:
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

            attitude_calc = AttitudeCalculator(
                propagator_type=PropagatorType.SGP4
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
