"""
统一机动约束检查器

合并所有机动相关约束（不包括SAA约束）：
1. 姿态切换约束（对日/对地/成像/数传姿态间的切换）
2. 机动能力约束（最大机动角度、角速度限制）
3. 时间窗口约束（与已调度任务的时间冲突、窗口边界）

使用已有的姿态管理代码：
- AttitudeManager: 姿态切换规划和决策
- AttitudeTransitionCalculator: 计算机动参数
- PowerGenerationCalculator: 计算发电功率影响

核心逻辑：
- 考虑任务前姿态切换时间（从上一任务姿态到当前成像姿态）
- 计算机动角度和时间
- 检查是否在时间窗口内
- 检查是否与已调度任务冲突
- 可选：计算任务后姿态（对日定向/对地定向）
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import logging

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.mission import Mission
from core.dynamics.attitude_manager import AttitudeManager, AttitudeManagementConfig
from core.dynamics.attitude_mode import AttitudeMode, TransitionResult
from core.dynamics.attitude_transition_calculator import (
    AttitudeTransitionCalculator,
    TransitionConfig
)
from core.dynamics.sun_position_calculator import SunPositionCalculator
from core.dynamics.power_generation_calculator import (
    PowerGenerationCalculator,
    PowerConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ManeuverCheckResult:
    """机动约束检查结果

    Attributes:
        feasible: 是否可行
        actual_start: 实际可开始时间（考虑了机动时间）
        actual_end: 实际结束时间
        slew_angle: 机动角度（度）
        slew_time: 机动时间（秒）
        roll_angle: 目标滚转角（度）
        pitch_angle: 目标俯仰角（度）
        from_mode: 起始姿态模式
        to_mode: 目标姿态模式
        power_before: 切换前发电功率（W）
        power_after: 切换后发电功率（W）
        window_available: 时间窗口是否足够
        slew_feasible: 机动是否可行
        conflict_reason: 冲突原因（如果不可行）
        conflict_with: 与哪个任务冲突（任务ID）
    """
    feasible: bool
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    slew_angle: float = 0.0
    slew_time: float = 0.0
    roll_angle: float = 0.0
    pitch_angle: float = 0.0
    from_mode: AttitudeMode = AttitudeMode.NADIR_POINTING
    to_mode: AttitudeMode = AttitudeMode.IMAGING
    power_before: float = 0.0
    power_after: float = 0.0
    window_available: bool = True
    slew_feasible: bool = True
    conflict_reason: Optional[str] = None
    conflict_with: Optional[str] = None


@dataclass
class ScheduledTaskInfo:
    """已调度任务信息（用于冲突检测）"""
    task_id: str
    target_id: str
    start_time: datetime
    end_time: datetime
    satellite_id: str
    end_mode: AttitudeMode = AttitudeMode.NADIR_POINTING  # 任务结束时的姿态模式


@dataclass
class SatelliteTaskState:
    """卫星任务状态"""
    last_end_time: datetime
    last_target: Optional[Target]
    last_mode: AttitudeMode
    scheduled_tasks: List[ScheduledTaskInfo] = field(default_factory=list)


class UnifiedManeuverChecker:
    """统一机动约束检查器

    合并所有机动相关约束（不包括SAA）：
    - 姿态切换约束
    - 机动能力约束
    - 时间窗口约束

    Attributes:
        mission: 任务对象
        config: 姿态管理配置
        attitude_manager: 姿态管理器
        transition_calculator: 姿态切换计算器
        power_calculator: 发电功率计算器
    """

    def __init__(
        self,
        mission: Mission,
        config: Optional[AttitudeManagementConfig] = None
    ):
        """初始化统一机动约束检查器

        Args:
            mission: 任务对象
            config: 姿态管理配置，如果为None则使用默认配置
        """
        self.mission = mission
        self.config = config if config is not None else AttitudeManagementConfig()

        # 初始化姿态管理器
        self.attitude_manager = AttitudeManager(self.config)

        # 初始化太阳位置计算器
        self.sun_calculator = SunPositionCalculator(use_orekit=False)

        # 初始化姿态切换计算器
        transition_config = TransitionConfig(
            max_slew_rate=self.config.max_slew_rate,
            settling_time=self.config.settling_time
        )
        self.transition_calculator = AttitudeTransitionCalculator(
            sun_calculator=self.sun_calculator,
            config=transition_config
        )

        # 初始化发电功率计算器
        power_config = PowerConfig()
        self.power_calculator = PowerGenerationCalculator(
            sun_calculator=self.sun_calculator,
            config=power_config
        )

        # 每个卫星的任务状态
        self._sat_states: Dict[str, SatelliteTaskState] = {}

        # 初始化状态
        for sat in mission.satellites:
            self._sat_states[sat.id] = SatelliteTaskState(
                last_end_time=mission.start_time,
                last_target=None,
                last_mode=AttitudeMode.NADIR_POINTING,  # 默认初始为对地定向
                scheduled_tasks=[]
            )

        logger.info("UnifiedManeuverChecker initialized")

    def reset(self) -> None:
        """重置所有卫星的调度状态"""
        for sat_id in self._sat_states:
            self._sat_states[sat_id] = SatelliteTaskState(
                last_end_time=self.mission.start_time,
                last_target=None,
                last_mode=AttitudeMode.NADIR_POINTING,
                scheduled_tasks=[]
            )
        logger.debug("UnifiedManeuverChecker reset")

    def reset_satellite(self, satellite_id: str) -> None:
        """重置指定卫星的调度状态"""
        if satellite_id in self._sat_states:
            self._sat_states[satellite_id] = SatelliteTaskState(
                last_end_time=self.mission.start_time,
                last_target=None,
                last_mode=AttitudeMode.NADIR_POINTING,
                scheduled_tasks=[]
            )
            logger.debug(f"Satellite {satellite_id} state reset")

    def check_maneuver_placement(
        self,
        satellite_id: str,
        target: Target,
        window_start: datetime,
        window_end: datetime,
        imaging_duration: float,
        satellite_position: Tuple[float, float, float],
        task_id: Optional[str] = None,
        from_mode: Optional[AttitudeMode] = None,
        to_mode: AttitudeMode = AttitudeMode.IMAGING
    ) -> ManeuverCheckResult:
        """检查任务放置的机动可行性

        这是统一的机动约束检查入口，同时考虑：
        - 姿态切换时间和角度
        - 机动能力限制
        - 时间窗口边界
        - 与已调度任务的时间冲突

        Args:
            satellite_id: 卫星ID
            target: 目标对象
            window_start: 可见窗口开始时间
            window_end: 可见窗口结束时间
            imaging_duration: 成像时长（秒）
            satellite_position: 卫星ECEF位置（米）
            task_id: 任务ID（用于冲突诊断）
            from_mode: 起始姿态模式，None则使用卫星当前状态
            to_mode: 目标姿态模式，默认IMAGING

        Returns:
            ManeuverCheckResult: 检查结果
        """
        # 1. 获取卫星信息和状态
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            return ManeuverCheckResult(
                feasible=False,
                conflict_reason=f"Satellite {satellite_id} not found"
            )

        sat_state = self._sat_states.get(satellite_id)
        if not sat_state:
            return ManeuverCheckResult(
                feasible=False,
                conflict_reason=f"Satellite {satellite_id} state not initialized"
            )

        # 2. 确定起始姿态模式
        if from_mode is None:
            from_mode = sat_state.last_mode

        # 3. 计算目标位置（地理坐标）
        target_position = None
        if hasattr(target, 'latitude') and hasattr(target, 'longitude'):
            target_position = (target.latitude, target.longitude)

        # 4. 计算姿态切换（高精度要求：始终使用精确模型）
        try:
            transition_result = self.attitude_manager.plan_transition(
                from_mode=from_mode,
                to_mode=to_mode,
                satellite_position=satellite_position,
                timestamp=window_start,
                target_position=target_position
            )
        except Exception as e:
            logger.warning(f"Failed to plan transition: {e}")
            return ManeuverCheckResult(
                feasible=False,
                conflict_reason=f"Transition calculation failed: {str(e)}"
            )

        # 6. 检查姿态切换可行性
        if not transition_result.feasible:
            return ManeuverCheckResult(
                feasible=False,
                slew_feasible=False,
                from_mode=from_mode,
                to_mode=to_mode,
                conflict_reason=f"Attitude transition not feasible: {transition_result.reason}"
            )

        # 7. 检查机动能力约束（最大角度限制）
        # 注意：Java预计算的可见性窗口已经验证了姿态可行性（分别检查滚转/俯仰角）
        # 这里使用transition_result中的滚转/俯仰角分别检查，而不是合成角
        max_roll_angle = getattr(satellite.capabilities, 'max_roll_angle', 45.0)
        max_pitch_angle = getattr(satellite.capabilities, 'max_pitch_angle', 30.0)
        roll_angle = transition_result.roll_angle if hasattr(transition_result, 'roll_angle') else 0.0
        pitch_angle = transition_result.pitch_angle if hasattr(transition_result, 'pitch_angle') else 0.0

        if abs(roll_angle) > max_roll_angle or abs(pitch_angle) > max_pitch_angle:
            return ManeuverCheckResult(
                feasible=False,
                slew_angle=transition_result.slew_angle,
                slew_time=transition_result.slew_time,
                from_mode=from_mode,
                to_mode=to_mode,
                slew_feasible=False,
                conflict_reason=f"Attitude angle exceeds limit: roll={roll_angle:.1f}° (max {max_roll_angle}°), pitch={pitch_angle:.1f}° (max {max_pitch_angle}°)"
            )

        # 8. 计算实际可开始时间
        slew_time = transition_result.slew_time
        earliest_start = sat_state.last_end_time + timedelta(seconds=slew_time)
        actual_start = max(window_start, earliest_start)

        # 9. 计算实际结束时间
        actual_end = actual_start + timedelta(seconds=imaging_duration)

        # 10. 检查窗口是否足够
        if actual_end > window_end:
            return ManeuverCheckResult(
                feasible=False,
                actual_start=actual_start,
                actual_end=actual_end,
                slew_angle=transition_result.slew_angle,
                slew_time=slew_time,
                roll_angle=transition_result.roll_angle,
                pitch_angle=transition_result.pitch_angle,
                from_mode=from_mode,
                to_mode=to_mode,
                window_available=False,
                conflict_reason=f"Window too short: need {imaging_duration}s, "
                               f"but only {(window_end - actual_start).total_seconds():.1f}s available"
            )

        # 11. 检查与已调度任务的时间重叠
        conflict_task = self._find_time_conflict(satellite_id, actual_start, actual_end)
        if conflict_task:
            return ManeuverCheckResult(
                feasible=False,
                actual_start=actual_start,
                actual_end=actual_end,
                slew_angle=transition_result.slew_angle,
                slew_time=slew_time,
                roll_angle=transition_result.roll_angle,
                pitch_angle=transition_result.pitch_angle,
                from_mode=from_mode,
                to_mode=to_mode,
                window_available=True,
                conflict_reason=f"Time conflict with task {conflict_task.task_id}",
                conflict_with=conflict_task.task_id
            )

        # 12. 计算发电功率
        power_before = 0.0
        power_after = transition_result.power_generation
        try:
            power_before = self.power_calculator.calculate_power(
                attitude_mode=from_mode,
                satellite_position=satellite_position,
                timestamp=sat_state.last_end_time
            )
        except Exception as e:
            logger.debug(f"Failed to calculate power_before: {e}")

        # 13. 所有检查通过
        return ManeuverCheckResult(
            feasible=True,
            actual_start=actual_start,
            actual_end=actual_end,
            slew_angle=transition_result.slew_angle,
            slew_time=slew_time,
            roll_angle=transition_result.roll_angle,
            pitch_angle=transition_result.pitch_angle,
            from_mode=from_mode,
            to_mode=to_mode,
            power_before=power_before,
            power_after=power_after,
            window_available=True,
            slew_feasible=True
        )

    def commit_task(
        self,
        satellite_id: str,
        task_id: str,
        target_id: str,
        actual_start: datetime,
        actual_end: datetime,
        target: Optional[Target] = None,
        end_mode: AttitudeMode = AttitudeMode.NADIR_POINTING
    ) -> None:
        """提交任务，更新卫星调度状态

        Args:
            satellite_id: 卫星ID
            task_id: 任务ID
            target_id: 目标ID
            actual_start: 实际开始时间
            actual_end: 实际结束时间
            target: 目标对象（用于更新last_target）
            end_mode: 任务结束时的姿态模式
        """
        sat_state = self._sat_states.get(satellite_id)
        if not sat_state:
            logger.warning(f"Satellite {satellite_id} state not found")
            return

        # 创建任务信息
        task_info = ScheduledTaskInfo(
            task_id=task_id,
            target_id=target_id,
            start_time=actual_start,
            end_time=actual_end,
            satellite_id=satellite_id,
            end_mode=end_mode
        )

        # 添加到调度列表
        sat_state.scheduled_tasks.append(task_info)

        # 按时间排序
        sat_state.scheduled_tasks.sort(key=lambda x: x.start_time)

        # 更新状态
        sat_state.last_end_time = actual_end
        if target:
            sat_state.last_target = target
        sat_state.last_mode = end_mode

        logger.debug(f"Task {task_id} committed to satellite {satellite_id}")

    def decide_post_task_attitude(
        self,
        satellite_id: str,
        next_task_time: Optional[datetime],
        current_time: datetime,
        soc: float,
        time_since_last_dump: float
    ) -> AttitudeMode:
        """决定任务完成后的姿态模式

        使用AttitudeManager的决策逻辑：
        1. 动量卸载（最高优先级）
        2. 对日定向（空闲时间长或电量低）
        3. 对地定向（默认）

        Args:
            satellite_id: 卫星ID
            next_task_time: 下次任务时间，None表示没有后续任务
            current_time: 当前时间
            soc: 电池电量状态（0.0-1.0）
            time_since_last_dump: 距离上次动量卸载的时间（秒）

        Returns:
            AttitudeMode: 建议的下一姿态模式
        """
        sat_state = self._sat_states.get(satellite_id)
        current_mode = sat_state.last_mode if sat_state else AttitudeMode.NADIR_POINTING

        return self.attitude_manager.decide_post_task_attitude(
            current_mode=current_mode,
            next_task_time=next_task_time,
            current_time=current_time,
            soc=soc,
            time_since_last_dump=time_since_last_dump
        )

    def remove_task(self, satellite_id: str, task_id: str) -> bool:
        """移除已调度任务

        Args:
            satellite_id: 卫星ID
            task_id: 任务ID

        Returns:
            bool: 是否成功移除
        """
        sat_state = self._sat_states.get(satellite_id)
        if not sat_state:
            return False

        tasks = sat_state.scheduled_tasks
        for i, task in enumerate(tasks):
            if task.task_id == task_id:
                tasks.pop(i)
                # 更新最后任务信息
                if tasks:
                    last_task = tasks[-1]
                    sat_state.last_end_time = last_task.end_time
                    sat_state.last_mode = last_task.end_mode
                else:
                    sat_state.last_end_time = self.mission.start_time
                    sat_state.last_target = None
                    sat_state.last_mode = AttitudeMode.NADIR_POINTING
                return True
        return False

    def get_scheduled_tasks(self, satellite_id: str) -> List[ScheduledTaskInfo]:
        """获取卫星的已调度任务列表"""
        sat_state = self._sat_states.get(satellite_id)
        return sat_state.scheduled_tasks.copy() if sat_state else []

    def get_satellite_timeline(self, satellite_id: str) -> List[Tuple[datetime, datetime, str]]:
        """获取卫星的时间线

        Returns:
            List[Tuple[start, end, task_id]]: 时间线列表
        """
        sat_state = self._sat_states.get(satellite_id)
        if not sat_state:
            return []
        return [
            (t.start_time, t.end_time, t.task_id)
            for t in sat_state.scheduled_tasks
        ]

    def get_satellite_state(self, satellite_id: str) -> Optional[SatelliteTaskState]:
        """获取卫星任务状态"""
        return self._sat_states.get(satellite_id)

    def _find_time_conflict(
        self,
        satellite_id: str,
        start: datetime,
        end: datetime
    ) -> Optional[ScheduledTaskInfo]:
        """查找时间冲突的任务

        Args:
            satellite_id: 卫星ID
            start: 新任务开始时间
            end: 新任务结束时间

        Returns:
            冲突的任务信息，如果没有冲突则返回None
        """
        sat_state = self._sat_states.get(satellite_id)
        if not sat_state:
            return None

        for task in sat_state.scheduled_tasks:
            # 检查是否有重叠
            # 不重叠的条件：新任务在旧任务之前结束，或在旧任务之后开始
            if not (end <= task.start_time or start >= task.end_time):
                return task
        return None

    def check_batch_placement(
        self,
        placements: List[Dict[str, Any]]
    ) -> Tuple[bool, List[ManeuverCheckResult]]:
        """批量检查任务放置

        用于元启发式算法评估多个任务的可行性

        Args:
            placements: 放置列表，每个元素包含：
                - satellite_id: 卫星ID
                - target: 目标对象
                - window_start: 窗口开始
                - window_end: 窗口结束
                - imaging_duration: 成像时长
                - satellite_position: 卫星ECEF位置
                - task_id: 任务ID（可选）
                - from_mode: 起始姿态模式（可选）
                - to_mode: 目标姿态模式（可选）

        Returns:
            Tuple[全部可行, 各任务检查结果]
        """
        results = []
        all_feasible = True

        # 保存当前状态
        saved_states = {}
        for sat_id, state in self._sat_states.items():
            saved_states[sat_id] = SatelliteTaskState(
                last_end_time=state.last_end_time,
                last_target=state.last_target,
                last_mode=state.last_mode,
                scheduled_tasks=state.scheduled_tasks.copy()
            )

        try:
            for placement in placements:
                result = self.check_maneuver_placement(
                    satellite_id=placement['satellite_id'],
                    target=placement['target'],
                    window_start=placement['window_start'],
                    window_end=placement['window_end'],
                    imaging_duration=placement['imaging_duration'],
                    satellite_position=placement['satellite_position'],
                    task_id=placement.get('task_id'),
                    from_mode=placement.get('from_mode'),
                    to_mode=placement.get('to_mode', AttitudeMode.IMAGING)
                )
                results.append(result)

                if result.feasible:
                    # 临时提交，供后续任务检查
                    self.commit_task(
                        satellite_id=placement['satellite_id'],
                        task_id=placement.get('task_id', 'temp'),
                        target_id=placement['target'].id,
                        actual_start=result.actual_start,
                        actual_end=result.actual_end,
                        target=placement['target'],
                        end_mode=result.to_mode
                    )
                else:
                    all_feasible = False
        finally:
            # 恢复状态
            self._sat_states = saved_states

        return all_feasible, results
