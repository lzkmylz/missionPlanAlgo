"""
调度器基类

所有调度算法必须继承此基类，实现即插即用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import time

from .frequency_utils import (
    ObservationTask,
    create_observation_tasks,
    calculate_frequency_fitness,
)
from core.dynamics.attitude_calculator import (
    AttitudeCalculator,
    PropagatorType,
    AttitudeAngles,
)


class TaskFailureReason(Enum):
    """任务失败原因枚举 - Chapter 12.4完整实现"""
    # 资源约束
    POWER_CONSTRAINT = "power_constraint"           # 电量不足
    STORAGE_CONSTRAINT = "storage_constraint"        # 存储溢出
    STORAGE_OVERFLOW_RISK = "storage_overflow_risk"  # 存储溢出风险

    # 时间约束
    NO_VISIBLE_WINDOW = "no_visible_window"          # 无可见时间窗
    WINDOW_TOO_SHORT = "window_too_short"            # 可见窗口太短
    TIME_CONFLICT = "time_conflict"                  # 与其他任务时间冲突
    DEADLINE_VIOLATION = "deadline_violation"        # 超出截止时间

    # 能力约束
    SAT_CAPABILITY_MISMATCH = "sat_capability_mismatch"  # 卫星能力不匹配
    MODE_NOT_SUPPORTED = "mode_not_supported"        # 不支持所需成像模式
    OFF_NADIR_EXCEEDED = "off_nadir_exceeded"        # 超出最大侧摆角

    # 协同约束
    GROUND_STATION_UNAVAILABLE = "ground_station_unavailable"  # 地面站不可用
    ANTENNA_CONFLICT = "antenna_conflict"            # 天线资源冲突

    # 物理约束 - H2新增
    THERMAL_CONSTRAINT = "thermal_constraint"        # 热控约束超限
    SUN_EXCLUSION_VIOLATION = "sun_exclusion_violation"  # 太阳规避角不满足
    STORAGE_FRAGMENTATION = "storage_fragmentation"  # 存储碎片化无法分配

    # 网络约束 - H2新增
    NO_ISL_PATH = "no_isl_path"                      # 无可用ISL路径
    UPLINK_UNAVAILABLE = "uplink_unavailable"        # 无指令上行窗口
    RELAY_OVERLOAD = "relay_overload"                # 中继星容量超限

    # 其他
    UNKNOWN = "unknown"                              # 未知原因
    ALGORITHM_TIMEOUT = "algorithm_timeout"          # 算法超时


@dataclass
class TaskFailure:
    """单个任务失败记录"""
    task_id: str
    failure_reason: TaskFailureReason
    failure_detail: str
    satellite_id: Optional[str] = None
    attempted_time: Optional[datetime] = None
    constraint_value: Optional[float] = None
    limit_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """已调度的任务"""
    task_id: str
    satellite_id: str
    target_id: str
    imaging_start: datetime
    imaging_end: datetime
    imaging_mode: str
    slew_angle: float = 0.0
    storage_before: float = 0.0
    storage_after: float = 0.0
    power_before: float = 0.0
    power_after: float = 0.0
    # 地面站数传相关字段
    ground_station_id: Optional[str] = None
    antenna_id: Optional[str] = None  # 具体使用的天线
    downlink_start: Optional[datetime] = None
    downlink_end: Optional[datetime] = None
    data_transferred: float = 0.0
    # 姿态角字段 - 用于姿控系统验证
    roll_angle: Optional[float] = None    # 滚转角（度）
    pitch_angle: Optional[float] = None   # 俯仰角（度）
    yaw_angle: Optional[float] = None     # 偏航角（度）
    attitude_coordinate_system: str = "LVLH"  # 坐标系：LVLH

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'satellite_id': self.satellite_id,
            'target_id': self.target_id,
            'imaging_start': self.imaging_start.isoformat() if self.imaging_start else None,
            'imaging_end': self.imaging_end.isoformat() if self.imaging_end else None,
            'imaging_mode': self.imaging_mode,
            'slew_angle': self.slew_angle,
            'ground_station_id': self.ground_station_id,
            'antenna_id': self.antenna_id,
            'downlink_start': self.downlink_start.isoformat() if self.downlink_start else None,
            'downlink_end': self.downlink_end.isoformat() if self.downlink_end else None,
            'data_transferred': self.data_transferred,
            'roll_angle': self.roll_angle,
            'pitch_angle': self.pitch_angle,
            'yaw_angle': self.yaw_angle,
            'attitude_coordinate_system': self.attitude_coordinate_system,
        }


@dataclass
class ScheduleResult:
    """调度结果"""
    scheduled_tasks: List[ScheduledTask]
    unscheduled_tasks: Dict[str, TaskFailure]
    makespan: float
    computation_time: float
    iterations: int
    convergence_curve: List[float] = field(default_factory=list)
    failure_summary: Optional[Dict[TaskFailureReason, int]] = None

    def get_demand_satisfaction_rate(self, total_tasks: int) -> float:
        if total_tasks == 0:
            return 0.0
        return len(self.scheduled_tasks) / total_tasks

    def get_failure_rate_by_reason(self) -> Dict[str, float]:
        """按原因计算失败比例

        Returns:
            Dict[str, float]: 各失败原因的比例，key为原因字符串，value为比例(0-1)
        """
        if not self.unscheduled_tasks:
            return {}
        if self.failure_summary is None:
            return {}
        total = len(self.unscheduled_tasks)
        return {
            reason.value: count / total
            for reason, count in self.failure_summary.items()
        }


class BaseScheduler(ABC):
    """调度器基类 - 即插即用接口"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.mission = None
        self.window_cache = None
        self._failure_log: List[TaskFailure] = []
        self._start_time: Optional[float] = None
        self._iterations = 0
        self._convergence_curve: List[float] = []
        # 初始化姿态角计算器
        propagator_type = self.config.get('propagator_type', 'sgp4')
        self._attitude_calculator = AttitudeCalculator(
            propagator_type=PropagatorType.SGP4 if propagator_type == 'sgp4' else PropagatorType.HPOP
        )
        self._enable_attitude_calculation = self.config.get('enable_attitude_calculation', True)

    def initialize(self, mission, satellite_pool=None, ground_station_pool=None) -> None:
        """初始化调度器"""
        self.mission = mission
        self._failure_log = []
        self._iterations = 0
        self._convergence_curve = []

    def set_window_cache(self, cache) -> None:
        """设置窗口缓存"""
        self.window_cache = cache

    @abstractmethod
    def schedule(self) -> ScheduleResult:
        """执行调度"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """返回算法可调参数"""
        pass

    def _record_failure(self, task_id: str, reason: TaskFailureReason, detail: str, **kwargs) -> None:
        """记录任务失败原因"""
        failure = TaskFailure(
            task_id=task_id,
            failure_reason=reason,
            failure_detail=detail,
            **kwargs
        )
        self._failure_log.append(failure)

    def _start_timer(self) -> None:
        self._start_time = time.time()

    def _stop_timer(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _add_convergence_point(self, fitness: float) -> None:
        self._convergence_curve.append(fitness)
        self._iterations += 1

    def _validate_initialization(self) -> None:
        """
        验证调度器是否已正确初始化

        检查所有必要字段：
        1. mission 不为 None
        2. satellites 不为空列表
        3. targets 不为空列表

        Raises:
            RuntimeError: 如果任何必要字段未初始化
        """
        if self.mission is None:
            raise RuntimeError("Scheduler not initialized: mission is None")

        if not hasattr(self.mission, 'satellites') or not self.mission.satellites:
            raise RuntimeError("Scheduler not initialized: no satellites available")

        if not hasattr(self.mission, 'targets') or not self.mission.targets:
            raise RuntimeError("Scheduler not initialized: no targets available")

    def validate_solution(self, result: ScheduleResult) -> bool:
        """
        验证解的可行性

        根据设计文档3.1要求，检查调度结果是否满足所有约束：
        1. 任务时间不重叠
        2. 卫星资源约束满足
        3. 时间窗口约束满足
        4. 卫星能力约束满足

        Args:
            result: 调度结果

        Returns:
            bool: 解是否可行
        """
        if not result.scheduled_tasks:
            # 空调度视为有效
            return True

        # 按卫星分组检查
        sat_tasks: Dict[str, List[ScheduledTask]] = {}
        for task in result.scheduled_tasks:
            if task.satellite_id not in sat_tasks:
                sat_tasks[task.satellite_id] = []
            sat_tasks[task.satellite_id].append(task)

        # 检查每个卫星的任务是否时间冲突
        for sat_id, tasks in sat_tasks.items():
            # 按开始时间排序
            sorted_tasks = sorted(tasks, key=lambda t: t.imaging_start)

            for i in range(len(sorted_tasks) - 1):
                current_task = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]

                # 检查时间重叠
                if current_task.imaging_end > next_task.imaging_start:
                    return False

        # 检查电量约束（简化检查：电量不能为负）
        for task in result.scheduled_tasks:
            if task.power_after < 0:
                return False

        # 检查存储约束（简化检查：存储不能为负或超容）
        for task in result.scheduled_tasks:
            if task.storage_after < 0:
                return False
            if self.mission:
                sat = self.mission.get_satellite_by_id(task.satellite_id)
                if sat and task.storage_after > sat.capabilities.storage_capacity:
                    return False

        return True

    def _create_frequency_aware_tasks(self):
        """创建频次感知的观测任务列表"""
        if not self.mission or not self.mission.targets:
            return []
        return create_observation_tasks(self.mission.targets)

    def _calculate_frequency_fitness(self, target_obs_count, base_score=0.0):
        """计算频次满足度的适应度分数"""
        if not self.mission or not self.mission.targets:
            return base_score
        return calculate_frequency_fitness(
            target_obs_count, self.mission.targets, base_score
        )

    def _calculate_attitude_angles(
        self,
        satellite,
        target,
        imaging_time
    ) -> Optional[AttitudeAngles]:
        """计算卫星成像时刻的姿态角

        使用配置的AttitudeCalculator计算姿态角，
        如果计算失败则返回None。

        Args:
            satellite: 卫星对象
            target: 目标对象
            imaging_time: 成像时刻

        Returns:
            AttitudeAngles对象，如果禁用姿态角计算或计算失败则返回None
        """
        if not self._enable_attitude_calculation:
            return None

        if self._attitude_calculator is None:
            return None

        try:
            attitude = self._attitude_calculator.calculate_attitude(
                satellite=satellite,
                target=target,
                imaging_time=imaging_time
            )
            return attitude
        except Exception as e:
            # 记录警告但不中断调度流程
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to calculate attitude angles for task: {e}"
            )
            return None

    def _apply_attitude_to_scheduled_task(
        self,
        scheduled_task: ScheduledTask,
        attitude: Optional[AttitudeAngles]
    ) -> None:
        """将姿态角应用到ScheduledTask

        Args:
            scheduled_task: 已调度的任务对象
            attitude: 姿态角对象
        """
        if attitude is None:
            return

        scheduled_task.roll_angle = attitude.roll
        scheduled_task.pitch_angle = attitude.pitch
        scheduled_task.yaw_angle = attitude.yaw
        scheduled_task.attitude_coordinate_system = attitude.coordinate_system

    def set_propagator_type(self, propagator_type: str) -> None:
        """设置轨道传播器类型

        Args:
            propagator_type: 'sgp4' 或 'hpop'
        """
        if propagator_type.lower() == 'sgp4':
            self._attitude_calculator = AttitudeCalculator(
                propagator_type=PropagatorType.SGP4
            )
        elif propagator_type.lower() == 'hpop':
            self._attitude_calculator = AttitudeCalculator(
                propagator_type=PropagatorType.HPOP
            )
        else:
            raise ValueError(f"Unknown propagator type: {propagator_type}")

    def enable_attitude_calculation(self, enable: bool = True) -> None:
        """启用或禁用姿态角计算

        Args:
            enable: 是否启用姿态角计算
        """
        self._enable_attitude_calculation = enable

