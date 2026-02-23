"""
保护时间验证器 (GuardTimeValidator)

独立模块实现保护时间验证逻辑。
从SOEGenerator中拆分出来，提供更清晰的职责分离。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 从soe_generator导入必要的类型
from core.telecommand.soe_generator import SOEActionType, SOEEntry


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


class GuardTimeValidator:
    """
    保护时间验证器

    验证SOE（事件序列）中各动作之间的保护时间是否满足工程约束。
    支持多卫星并行验证、自动修复等功能。

    默认规则：
    - PAYLOAD_POWER_ON -> PAYLOAD_WARMUP: 5秒（电源稳定时间）
    - SLEW_COMPLETE -> IMAGING_START: 5秒（姿态稳定时间）
    - IMAGING_COMPLETE -> DOWNLINK_START: 10秒（数据缓冲和快门关闭时间）
    """

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

    def __init__(self, custom_rules: Optional[List[GuardTimeRule]] = None):
        """
        初始化验证器

        Args:
            custom_rules: 自定义规则列表，如果为None则使用默认规则
        """
        if custom_rules is not None:
            self.rules = custom_rules.copy()
        else:
            self.rules = self.DEFAULT_RULES.copy()

    def add_rule(self, rule: GuardTimeRule) -> None:
        """
        动态添加规则

        Args:
            rule: 要添加的规则
        """
        self.rules.append(rule)

    def validate_soe(self, soe_entries: List[SOEEntry]) -> List[Dict]:
        """
        验证SOE序列的保护时间

        Args:
            soe_entries: SOE条目列表

        Returns:
            违规列表，每个违规包含：
            - entry_a: 第一个条目
            - entry_b: 第二个条目
            - rule: 违反的规则
            - required_interval: 要求的间隔
            - actual_interval: 实际间隔
        """
        violations = []

        if not soe_entries:
            return violations

        # 按卫星分组验证
        by_satellite: Dict[str, List[SOEEntry]] = {}
        for entry in soe_entries:
            sat_id = entry.satellite_id
            if sat_id not in by_satellite:
                by_satellite[sat_id] = []
            by_satellite[sat_id].append(entry)

        # 对每个卫星的条目进行验证
        for sat_id, entries in by_satellite.items():
            sorted_entries = sorted(entries, key=lambda e: e.timestamp)
            sat_violations = self._validate_sequence(sorted_entries)
            violations.extend(sat_violations)

        return violations

    def _validate_sequence(self, entries: List[SOEEntry]) -> List[Dict]:
        """
        验证单个序列的保护时间

        Args:
            entries: 已排序的SOE条目列表

        Returns:
            违规列表
        """
        violations = []

        if len(entries) < 2:
            return violations

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                entry_a = entries[i]
                entry_b = entries[j]

                # 查找适用的规则
                for rule in self.rules:
                    if (entry_a.action_type == rule.action_a and
                        entry_b.action_type == rule.action_b):

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
        """
        自动修复保护时间违规

        通过延迟违规条目及其后续所有条目来满足保护时间要求。

        Args:
            soe_entries: 原始SOE条目列表

        Returns:
            修复后的新条目列表（不修改原始列表）
        """
        violations = self.validate_soe(soe_entries)

        if not violations:
            # 无违规，返回副本
            return [
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

        # 按卫星分组处理
        by_satellite: Dict[str, List[SOEEntry]] = {}
        for entry in soe_entries:
            sat_id = entry.satellite_id
            if sat_id not in by_satellite:
                by_satellite[sat_id] = []
            by_satellite[sat_id].append(entry)

        fixed_entries = []

        for sat_id, entries in by_satellite.items():
            # 按时间排序
            sorted_entries = sorted(entries, key=lambda e: e.timestamp)

            # 创建可修改的副本
            sat_fixed = [
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
                for entry in sorted_entries
            ]

            # 迭代修复直到无违规
            max_iterations = 10
            for _ in range(max_iterations):
                sat_violations = self._validate_sequence(sat_fixed)
                if not sat_violations:
                    break

                # 修复第一个违规
                violation = sat_violations[0]
                entry_b = violation['entry_b']
                required_delay = violation['required_interval'] - violation['actual_interval']

                # 延迟entry_b及其后续条目
                for entry in sat_fixed:
                    if entry.timestamp >= entry_b.timestamp:
                        entry.timestamp += required_delay

            fixed_entries.extend(sat_fixed)

        # 按时间排序返回
        fixed_entries.sort(key=lambda e: e.timestamp)
        return fixed_entries

    def validate_and_raise(self, soe_entries: List[SOEEntry]) -> None:
        """
        验证SOE并在有违规时抛出异常

        Args:
            soe_entries: SOE条目列表

        Raises:
            GuardTimeViolationError: 当存在违规时
        """
        violations = self.validate_soe(soe_entries)
        if violations:
            raise GuardTimeViolationError(
                f"Found {len(violations)} guard time violations",
                violations
            )
