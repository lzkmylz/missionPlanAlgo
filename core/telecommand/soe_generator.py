"""
SOE (Sequence of Events) 生成器

将调度计划转换为卫星可执行的时间事件序列
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple


class SOEActionType(Enum):
    """SOE动作类型"""
    PAYLOAD_POWER_ON = "PAYLOAD_POWER_ON"
    PAYLOAD_WARMUP = "PAYLOAD_WARMUP"
    SLEW_START = "SLEW_START"
    SLEW_COMPLETE = "SLEW_COMPLETE"
    SHUTTER_OPEN = "SHUTTER_OPEN"
    IMAGING_START = "IMAGING_START"
    IMAGING_COMPLETE = "IMAGING_COMPLETE"
    SHUTTER_CLOSE = "SHUTTER_CLOSE"
    DOWNLINK_START = "DOWNLINK_START"
    DOWNLINK_COMPLETE = "DOWNLINK_COMPLETE"


@dataclass
class SOEEntry:
    """SOE条目"""
    timestamp: datetime
    action_type: SOEActionType
    satellite_id: str
    task_id: Optional[str]
    duration: Optional[timedelta]
    parameters: Dict[str, Any]
    guard_time_before: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    guard_time_after: timedelta = field(default_factory=lambda: timedelta(seconds=0))


@dataclass
class GuardTimeRule:
    """保护时间规则"""
    action_a: SOEActionType
    action_b: SOEActionType
    min_interval: timedelta
    reason: str


class GuardTimeViolationError(Exception):
    """保护时间违规异常"""
    def __init__(self, message: str, violations: List[Dict]):
        super().__init__(message)
        self.violations = violations


class SOEGenerator:
    """
    SOE (Sequence of Events) 生成器

    将调度计划转换为卫星可执行的时间事件序列
    例如：转码器开机 -> 预热 -> 姿态机动 -> 开快门 -> 关快门 -> 姿态恢复
    """

    # 动作模板配置
    ACTION_TEMPLATES = {
        'optical_imaging': [
            (timedelta(seconds=-300), SOEActionType.PAYLOAD_POWER_ON, timedelta(seconds=60)),
            (timedelta(seconds=-240), SOEActionType.PAYLOAD_WARMUP, timedelta(seconds=120)),
            (timedelta(seconds=-60), SOEActionType.SLEW_START, timedelta(seconds=55)),
            (timedelta(seconds=0), SOEActionType.SHUTTER_OPEN, timedelta(seconds=1)),
            (timedelta(seconds=0), SOEActionType.IMAGING_START, None),
        ],
        'sar_imaging': [
            (timedelta(seconds=-180), SOEActionType.PAYLOAD_POWER_ON, timedelta(seconds=60)),
            (timedelta(seconds=-120), SOEActionType.PAYLOAD_WARMUP, timedelta(seconds=90)),
            (timedelta(seconds=-30), SOEActionType.SLEW_START, timedelta(seconds=25)),
            (timedelta(seconds=0), SOEActionType.IMAGING_START, None),
        ],
    }

    # 保护时间配置
    GUARD_TIMES = {
        SOEActionType.PAYLOAD_POWER_ON: (timedelta(seconds=10), timedelta(seconds=5)),
        SOEActionType.SLEW_START: (timedelta(seconds=5), timedelta(seconds=5)),
        SOEActionType.SHUTTER_OPEN: (timedelta(seconds=2), timedelta(seconds=2)),
    }

    def generate_soe(self, schedule, validate: bool = False) -> List[SOEEntry]:
        """从调度计划生成SOE序列

        Args:
            schedule: 调度结果
            validate: 是否验证保护时间，默认为False

        Returns:
            SOE条目列表
        """
        soe_entries = []

        for task in schedule.scheduled_tasks:
            task_soe = self._generate_task_soe(task)
            soe_entries.extend(task_soe)

        # 按时间排序
        soe_entries.sort(key=lambda e: e.timestamp)

        # 验证保护时间（仅在validate=True时）
        if validate:
            self._validate_guard_times(soe_entries)

        return soe_entries

    def _generate_task_soe(self, task) -> List[SOEEntry]:
        """为单个任务生成SOE"""
        entries = []
        task_type = self._determine_task_type(task)

        # 未知任务类型返回空列表
        if not task_type or task_type not in self.ACTION_TEMPLATES:
            return entries

        template = self.ACTION_TEMPLATES.get(task_type, [])

        # 处理不同的任务类型
        if hasattr(task, 'imaging_start'):
            imaging_start = task.imaging_start
            imaging_end = task.imaging_end
        else:
            imaging_start = task['imaging_start']
            imaging_end = task['imaging_end']

        imaging_duration = imaging_end - imaging_start

        for offset, action_type, duration in template:
            timestamp = imaging_start + offset
            actual_duration = duration if duration else imaging_duration

            guard_before, guard_after = self.GUARD_TIMES.get(
                action_type, (timedelta(0), timedelta(0))
            )

            # 处理不同的任务类型
            if hasattr(task, 'task_id'):
                task_id = task.task_id
                satellite_id = task.satellite_id
                target_id = getattr(task, 'target_id', None)
                imaging_mode = getattr(task, 'imaging_mode', None)
                slew_angle = getattr(task, 'slew_angle', 0.0)
            else:
                task_id = task['task_id']
                satellite_id = task['satellite_id']
                target_id = task.get('target_id')
                imaging_mode = task.get('imaging_mode')
                slew_angle = task.get('slew_angle', 0.0)

            entry = SOEEntry(
                timestamp=timestamp,
                action_type=action_type,
                satellite_id=satellite_id,
                task_id=task_id,
                duration=actual_duration,
                parameters={
                    'target_id': target_id,
                    'imaging_mode': imaging_mode,
                    'slew_angle': slew_angle,
                },
                guard_time_before=guard_before,
                guard_time_after=guard_after
            )
            entries.append(entry)

        # 添加结束动作
        entries.extend(self._generate_completion_actions(task))

        return entries

    def _determine_task_type(self, task) -> str:
        """确定任务类型"""
        # 处理不同的任务类型
        if hasattr(task, 'imaging_mode'):
            imaging_mode = task.imaging_mode
        elif isinstance(task, dict):
            imaging_mode = task.get('imaging_mode', '')
        else:
            imaging_mode = ''

        if hasattr(task, 'imaging_type'):
            imaging_type = task.imaging_type
        elif isinstance(task, dict):
            imaging_type = task.get('imaging_type', '')
        else:
            imaging_type = ''

        # 根据成像类型或成像模式判断
        if imaging_type == 'optical' or 'optical' in imaging_mode.lower():
            return 'optical_imaging'
        elif imaging_type == 'sar' or 'sar' in imaging_mode.lower() or 'strip' in imaging_mode.lower():
            return 'sar_imaging'
        elif 'optical' in imaging_mode.lower():
            return 'optical_imaging'
        elif 'sar' in imaging_mode.lower() or 'spot' in imaging_mode.lower():
            return 'sar_imaging'

        return ''

    def _generate_completion_actions(self, task) -> List[SOEEntry]:
        """生成完成动作"""
        entries = []

        # 处理不同的任务类型
        if hasattr(task, 'imaging_end'):
            imaging_end = task.imaging_end
            task_id = task.task_id
            satellite_id = task.satellite_id
            target_id = getattr(task, 'target_id', None)
            imaging_mode = getattr(task, 'imaging_mode', None)
            slew_angle = getattr(task, 'slew_angle', 0.0)
        else:
            imaging_end = task['imaging_end']
            task_id = task['task_id']
            satellite_id = task['satellite_id']
            target_id = task.get('target_id')
            imaging_mode = task.get('imaging_mode')
            slew_angle = task.get('slew_angle', 0.0)

        task_type = self._determine_task_type(task)

        # 成像完成
        entries.append(SOEEntry(
            timestamp=imaging_end,
            action_type=SOEActionType.IMAGING_COMPLETE,
            satellite_id=satellite_id,
            task_id=task_id,
            duration=None,
            parameters={
                'target_id': target_id,
                'imaging_mode': imaging_mode,
                'slew_angle': slew_angle,
            }
        ))

        # 光学成像需要关快门
        if task_type == 'optical_imaging':
            entries.append(SOEEntry(
                timestamp=imaging_end,
                action_type=SOEActionType.SHUTTER_CLOSE,
                satellite_id=satellite_id,
                task_id=task_id,
                duration=timedelta(seconds=1),
                parameters={
                    'target_id': target_id,
                    'imaging_mode': imaging_mode,
                    'slew_angle': slew_angle,
                }
            ))

        # 姿态恢复完成 (假设需要30秒)
        entries.append(SOEEntry(
            timestamp=imaging_end + timedelta(seconds=30),
            action_type=SOEActionType.SLEW_COMPLETE,
            satellite_id=satellite_id,
            task_id=task_id,
            duration=timedelta(seconds=30),
            parameters={
                'target_id': target_id,
                'imaging_mode': imaging_mode,
                'slew_angle': slew_angle,
            }
        ))

        return entries

    def _validate_guard_times(self, entries: List[SOEEntry]) -> bool:
        """验证保护时间是否合理"""
        violations = []

        for i, entry in enumerate(entries):
            if i > 0:
                prev_entry = entries[i - 1]
                min_interval = prev_entry.guard_time_after + entry.guard_time_before
                actual_interval = entry.timestamp - (
                    prev_entry.timestamp + (prev_entry.duration or timedelta(0))
                )

                if actual_interval < min_interval:
                    violations.append({
                        'entry_a': prev_entry,
                        'entry_b': entry,
                        'required_interval': min_interval,
                        'actual_interval': actual_interval
                    })

        if violations:
            raise GuardTimeViolationError(f"Found {len(violations)} guard time violations", violations)

        return True


class GuardTimeValidator:
    """保护时间验证器 - 验证动作之间的保护时间是否满足工程约束"""

    DEFAULT_RULES = [
        GuardTimeRule(
            SOEActionType.PAYLOAD_POWER_ON,
            SOEActionType.PAYLOAD_WARMUP,
            timedelta(seconds=5),
            "Power stabilization time"
        ),
        GuardTimeRule(
            SOEActionType.SLEW_COMPLETE,
            SOEActionType.IMAGING_START,
            timedelta(seconds=5),
            "Attitude stabilization time"
        ),
        GuardTimeRule(
            SOEActionType.IMAGING_COMPLETE,
            SOEActionType.DOWNLINK_START,
            timedelta(seconds=10),
            "Data buffering and shutter closure time"
        ),
    ]

    def __init__(self):
        self.rules = self.DEFAULT_RULES.copy()

    def validate_soe(self, soe_entries: List[SOEEntry]) -> List[Dict]:
        """验证SOE序列的保护时间"""
        violations = []

        # 按卫星分组验证
        by_satellite = {}
        for entry in soe_entries:
            sat_id = entry.satellite_id
            if sat_id not in by_satellite:
                by_satellite[sat_id] = []
            by_satellite[sat_id].append(entry)

        for sat_id, entries in by_satellite.items():
            sorted_entries = sorted(entries, key=lambda e: e.timestamp)
            sat_violations = self._validate_sequence(sorted_entries)
            violations.extend(sat_violations)

        return violations

    def _validate_sequence(self, entries: List[SOEEntry]) -> List[Dict]:
        """验证单个序列的保护时间"""
        violations = []

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                entry_a = entries[i]
                entry_b = entries[j]

                # 查找适用的规则
                for rule in self.rules:
                    if entry_a.action_type == rule.action_a and entry_b.action_type == rule.action_b:
                        # 计算实际间隔
                        end_time_a = entry_a.timestamp + (entry_a.duration or timedelta(0))
                        actual_interval = entry_b.timestamp - end_time_a

                        if actual_interval < rule.min_interval:
                            violations.append({
                                'entry_a': entry_a,
                                'entry_b': entry_b,
                                'rule': rule,
                                'required_interval': rule.min_interval,
                                'actual_interval': actual_interval
                            })

        return violations

    def auto_fix(self, soe_entries: List[SOEEntry]) -> List[SOEEntry]:
        """自动修复保护时间违规"""
        violations = self.validate_soe(soe_entries)

        if not violations:
            return soe_entries.copy()

        # 创建新条目以避免修改原列表
        fixed_entries = [
            SOEEntry(
                timestamp=entry.timestamp,
                action_type=entry.action_type,
                satellite_id=entry.satellite_id,
                task_id=entry.task_id,
                duration=entry.duration,
                parameters=entry.parameters.copy(),
                guard_time_before=entry.guard_time_before,
                guard_time_after=entry.guard_time_after
            )
            for entry in soe_entries
        ]

        for violation in violations:
            entry_b = violation['entry_b']
            required_delay = violation['required_interval'] - violation['actual_interval']

            # 延迟entry_b及其后续所有条目
            for entry in fixed_entries:
                if entry.timestamp >= entry_b.timestamp:
                    entry.timestamp += required_delay

        return fixed_entries
