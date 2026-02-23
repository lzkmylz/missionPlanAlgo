"""
调度方案验证器

实现第12章设计：
- 前向推演验证
- 资源约束检查
- 时间冲突检测
"""

from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime


class ScheduleValidator:
    """
    调度方案验证器

    验证候选任务序列是否满足：
    1. 电量始终 >= 0
    2. 存储始终 <= max_storage
    3. 任务时间不冲突
    """

    def __init__(self):
        """初始化验证器"""
        self._validation_errors: List[str] = []

    def validate_forward(self, candidate_task: Any,
                        existing_tasks: List[Any],
                        planning_horizon: datetime) -> Tuple[bool, str]:
        """
        前向推演验证

        将候选任务加入现有任务序列，验证到planning_horizon时刻
        资源约束始终满足

        Args:
            candidate_task: 候选任务
            existing_tasks: 现有任务列表
            planning_horizon: 规划周期结束时间

        Returns:
            Tuple[bool, str]: (是否可行, 失败原因)
        """
        # 验证候选任务时间有效性
        if candidate_task.end_time < candidate_task.start_time:
            return False, "无效的任务时间范围：结束时间早于开始时间"

        # 验证候选任务是否在规划周期内
        if candidate_task.end_time > planning_horizon:
            return False, f"任务超出规划周期：结束时间 {candidate_task.end_time} > 规划周期 {planning_horizon}"

        # 合并所有任务
        all_tasks = existing_tasks + [candidate_task]

        # 按卫星分组
        sat_tasks: Dict[str, List[Any]] = {}
        for task in all_tasks:
            sat_id = getattr(task, 'satellite_id', None) or getattr(task, 'sat_id', 'default')
            if sat_id not in sat_tasks:
                sat_tasks[sat_id] = []
            sat_tasks[sat_id].append(task)

        # 检查每个卫星的任务时间冲突
        for sat_id, tasks in sat_tasks.items():
            # 按开始时间排序
            sorted_tasks = sorted(tasks, key=lambda t: t.start_time)

            for i in range(len(sorted_tasks) - 1):
                current_task = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]

                # 检查时间重叠
                if current_task.end_time > next_task.start_time:
                    return False, f"卫星 {sat_id} 上的任务时间冲突: " \
                                 f"任务 {getattr(current_task, 'task_id', getattr(current_task, 'id', 'unknown'))} " \
                                 f"({current_task.start_time} - {current_task.end_time}) 与 " \
                                 f"任务 {getattr(next_task, 'task_id', getattr(next_task, 'id', 'unknown'))} " \
                                 f"({next_task.start_time} - {next_task.start_time})"

        # 资源约束验证（简化版本）
        # 实际实现中应该使用PowerModel和StorageIntegrator进行详细验证
        resource_valid, resource_reason = self._validate_resource_constraints(all_tasks)
        if not resource_valid:
            return False, resource_reason

        return True, ""

    def _validate_resource_constraints(self, tasks: List[Any]) -> Tuple[bool, str]:
        """
        验证资源约束

        Args:
            tasks: 任务列表

        Returns:
            Tuple[bool, str]: (是否满足约束, 失败原因)
        """
        # 按卫星分组检查资源
        sat_tasks: Dict[str, List[Any]] = {}
        for task in tasks:
            sat_id = getattr(task, 'satellite_id', None) or getattr(task, 'sat_id', 'default')
            if sat_id not in sat_tasks:
                sat_tasks[sat_id] = []
            sat_tasks[sat_id].append(task)

        for sat_id, tasks in sat_tasks.items():
            # 获取卫星能力（如果任务中有）
            power_capacity = getattr(tasks[0], 'power_capacity', None) or 1000.0
            storage_capacity = getattr(tasks[0], 'storage_capacity', None) or 100.0

            # 简化检查：累计资源需求不超过容量
            total_power_required = sum(
                getattr(t, 'power_required', 0.0) for t in tasks
            )
            total_storage_required = sum(
                getattr(t, 'storage_required', 0.0) for t in tasks
            )

            # 注意：这是简化检查，实际应该按时间顺序积分
            # 这里只检查累计需求是否超过容量的合理倍数
            if total_power_required > power_capacity * 10:  # 允许10倍于单次容量的累计消耗
                return False, f"卫星 {sat_id} 累计电量需求过高: {total_power_required}Wh"

            if total_storage_required > storage_capacity:
                return False, f"卫星 {sat_id} 存储容量不足: 需要 {total_storage_required}GB, " \
                             f"可用 {storage_capacity}GB"

        return True, ""

    def validate_full_schedule(self, tasks: List[Any],
                              planning_horizon: datetime,
                              satellites: Optional[List[Any]] = None) -> Tuple[bool, List[str]]:
        """
        验证完整调度方案

        Args:
            tasks: 完整任务列表
            planning_horizon: 规划周期结束时间
            satellites: 卫星列表（可选）

        Returns:
            Tuple[bool, List[str]]: (是否可行, 违规列表)
        """
        violations = []

        # 检查每个任务是否在规划周期内
        for task in tasks:
            if task.end_time > planning_horizon:
                violations.append(
                    f"任务 {getattr(task, 'task_id', getattr(task, 'id', 'unknown'))} "
                    f"超出规划周期"
                )

        # 按卫星分组检查时间冲突
        sat_tasks: Dict[str, List[Any]] = {}
        for task in tasks:
            sat_id = getattr(task, 'satellite_id', None) or getattr(task, 'sat_id', 'default')
            if sat_id not in sat_tasks:
                sat_tasks[sat_id] = []
            sat_tasks[sat_id].append(task)

        for sat_id, sat_task_list in sat_tasks.items():
            sorted_tasks = sorted(sat_task_list, key=lambda t: t.start_time)

            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]

                if current.end_time > next_task.start_time:
                    violations.append(
                        f"卫星 {sat_id} 任务时间冲突: "
                        f"{getattr(current, 'task_id', getattr(current, 'id', 'unknown'))} 与 "
                        f"{getattr(next_task, 'task_id', getattr(next_task, 'id', 'unknown'))}"
                    )

        return len(violations) == 0, violations
