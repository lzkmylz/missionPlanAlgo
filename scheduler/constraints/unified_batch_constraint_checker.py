"""
统一批量约束检查器

整合姿态、SAA、时间、资源批量检查，提供单一入口进行所有约束的批量检查。
支持分阶段过滤（先快速检查，后详细检查）。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from core.models.mission import Mission
from core.models.satellite import Satellite
from core.models.target import Target

from .batch_slew_constraint_checker import BatchSlewConstraintChecker
from .batch_slew_calculator import BatchSlewCandidate, BatchSlewResult
from .batch_saa_constraint_checker import BatchSAAConstraintChecker
from .batch_saa_calculator import BatchSAACandidate, BatchSAAResult
from .batch_time_conflict_checker import BatchTimeConflictChecker
from .batch_time_conflict_calculator import BatchTimeConflictCandidate, BatchTimeConflictResult
from .batch_resource_checker import BatchResourceChecker
from .batch_resource_calculator import BatchResourceCandidate, BatchResourceResult
from .batch_orbit_constraint_checker import (
    BatchOrbitConstraintChecker,
    BatchOrbitConstraintCandidate,
    BatchOrbitConstraintResult
)

from .slew_constraint_checker import SlewFeasibilityResult
from .saa_constraint_checker import SAAFeasibilityResult

logger = logging.getLogger(__name__)


@dataclass
class UnifiedBatchCandidate:
    """统一批量约束检查候选

    整合所有约束检查所需的信息
    """
    # 基础信息
    sat_id: str
    satellite: Satellite
    target: Target

    # 时间信息
    window_start: datetime
    window_end: datetime
    prev_end_time: datetime
    imaging_duration: float = 10.0
    imaging_begin: Optional[datetime] = None  # 实际成像开始时间

    # 机动信息
    prev_target: Optional[Target] = None
    sat_position: Optional[Tuple[float, float, float]] = None
    sat_velocity: Optional[Tuple[float, float, float]] = None

    # 资源信息
    power_needed: float = 0.0
    storage_produced: float = 0.0

    def __post_init__(self):
        """初始化后处理：默认 imaging_begin = window_start"""
        if self.imaging_begin is None:
            self.imaging_begin = self.window_start


@dataclass
class UnifiedBatchResult:
    """统一批量约束检查结果"""
    feasible: bool
    slew_result: Optional[SlewFeasibilityResult] = None
    saa_result: Optional[SAAFeasibilityResult] = None
    time_result: Optional[BatchTimeConflictResult] = None
    resource_result: Optional[BatchResourceResult] = None
    orbit_result: Optional[BatchOrbitConstraintResult] = None  # 新增：单圈约束结果
    reason: Optional[str] = None

    # 详细结果
    slew_feasible: bool = True
    saa_feasible: bool = True
    time_feasible: bool = True
    resource_feasible: bool = True
    orbit_feasible: bool = True  # 新增：单圈约束是否满足


class UnifiedBatchConstraintChecker:
    """统一批量约束检查器

    整合所有批量约束检查器，提供单一入口进行所有约束的批量检查。
    支持阶段式检查，早期失败可以跳过后续检查。

    使用方法:
        checker = UnifiedBatchConstraintChecker(mission)

        # 批量检查所有约束
        results = checker.check_all_constraints_batch(
            candidates=unified_candidates,
            existing_tasks=scheduled_tasks,
            satellite_states=satellite_states
        )

        # 阶段式检查（先快速筛选）
        results = checker.check_phased_batch(
            candidates=unified_candidates,
            phase='fast',  # 'fast' 或 'full'
            ...
        )
    """

    def __init__(
        self,
        mission: Mission,
        use_precise_model: bool = True,
        consider_power: bool = True,
        consider_storage: bool = True
    ):
        """初始化统一批量约束检查器

        Args:
            mission: 任务对象
            use_precise_model: 是否使用精确模型
            consider_power: 是否检查电量约束
            consider_storage: 是否检查存储约束
        """
        self.mission = mission

        # 初始化各批量检查器
        self._slew_checker = BatchSlewConstraintChecker(
            mission=mission,
            use_precise_model=use_precise_model
        )
        self._saa_checker = BatchSAAConstraintChecker(mission)
        self._time_checker = BatchTimeConflictChecker()
        self._resource_checker = BatchResourceChecker(
            consider_power=consider_power,
            consider_storage=consider_storage
        )
        self._orbit_checker = BatchOrbitConstraintChecker(mission)  # 新增：单圈约束检查器

        # 性能统计
        self._stats = {
            'total_checks': 0,
            'fast_phase_checks': 0,
            'full_phase_checks': 0,
            'early_terminations': 0
        }

        logger.info("UnifiedBatchConstraintChecker initialized")

    def check_all_constraints_batch(
        self,
        candidates: List[UnifiedBatchCandidate],
        existing_tasks: List[Dict[str, Any]],
        satellite_states: Dict[str, Dict[str, float]],
        current_attitudes: Optional[Dict] = None,
        early_termination: bool = True
    ) -> List[UnifiedBatchResult]:
        """批量检查所有约束

        对所有候选进行姿态、SAA、时间、资源的批量检查。

        Args:
            candidates: 统一候选列表
            existing_tasks: 已调度任务列表
            satellite_states: 卫星资源状态
            current_attitudes: 当前姿态状态（可选）
            early_termination: 是否早期终止（某约束失败则跳过后续）

        Returns:
            UnifiedBatchResult列表
        """
        if not candidates:
            return []

        self._stats['total_checks'] += len(candidates)

        results = [UnifiedBatchResult(feasible=True) for _ in candidates]

        # 阶段0: 分辨率约束检查（最快速，优先执行）
        for i, candidate in enumerate(candidates):
            if not self._check_resolution_constraint(candidate):
                results[i].feasible = False
                results[i].reason = "Resolution constraint: insufficient resolution for target"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i, r in enumerate(results) if r.feasible]
            if not active_indices:
                return results
        else:
            active_indices = list(range(len(candidates)))

        # 阶段1: 姿态机动约束检查（通常最耗时，但已批量优化）
        active_candidates = [candidates[i] for i in active_indices]
        slew_candidates = self._convert_to_slew_candidates(active_candidates)
        slew_results = self._slew_checker.check_slew_feasibility_batch(
            slew_candidates, current_attitudes
        )

        for i, (result, slew_result) in enumerate(zip(results, slew_results)):
            result.slew_result = slew_result
            result.slew_feasible = slew_result.feasible
            if not slew_result.feasible:
                result.feasible = False
                result.reason = f"Slew constraint: {slew_result.reason}"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i, r in enumerate(results) if r.feasible]
            if not active_indices:
                return results
        else:
            active_indices = list(range(len(candidates)))

        # 阶段2: SAA约束检查
        saa_candidates = self._convert_to_saa_candidates(
            [candidates[i] for i in active_indices]
        )
        saa_results = self._saa_checker.check_window_feasibility_batch(saa_candidates)

        for idx, saa_result in zip(active_indices, saa_results):
            results[idx].saa_result = saa_result
            results[idx].saa_feasible = saa_result.feasible
            if not saa_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"SAA constraint: {saa_result.violation_count} violations"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
            if not active_indices:
                return results

        # 阶段3: 时间冲突检查
        time_candidates = self._convert_to_time_candidates(
            [candidates[i] for i in active_indices]
        )
        time_results = self._time_checker.check_time_conflict_batch(
            time_candidates, existing_tasks
        )

        for idx, time_result in zip(active_indices, time_results):
            results[idx].time_result = time_result
            results[idx].time_feasible = time_result.feasible
            if not time_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"Time conflict: {time_result.reason}"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
            if not active_indices:
                return results

        # 阶段4: 资源约束检查
        resource_candidates = self._convert_to_resource_candidates(
            [candidates[i] for i in active_indices]
        )
        resource_results = self._resource_checker.check_resources_batch(
            resource_candidates, satellite_states
        )

        for idx, resource_result in zip(active_indices, resource_results):
            results[idx].resource_result = resource_result
            results[idx].resource_feasible = resource_result.feasible
            if not resource_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"Resource constraint: {resource_result.reason}"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
            if not active_indices:
                return results

        # 阶段5: 单圈约束检查（最后阶段，最耗时，前面约束通过后再检查）
        orbit_candidates = self._convert_to_orbit_candidates(
            [candidates[i] for i in active_indices]
        )
        orbit_results = self._orbit_checker.check_batch(
            orbit_candidates, existing_tasks
        )

        for idx, orbit_result in zip(active_indices, orbit_results):
            results[idx].orbit_result = orbit_result
            results[idx].orbit_feasible = orbit_result.feasible
            if not orbit_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"Orbit constraint: {orbit_result.reason}"

        return results

    def check_fast_phase_batch(
        self,
        candidates: List[UnifiedBatchCandidate],
        existing_tasks: List[Dict[str, Any]],
        satellite_states: Optional[Dict[str, Dict[str, float]]] = None,
        early_termination: bool = True
    ) -> List[UnifiedBatchResult]:
        """快速阶段批量检查

        检查SAA、时间冲突和资源约束（姿态约束已在外部检查完成）。
        用于快速筛选明显不可行的候选。

        Args:
            candidates: 统一候选列表
            existing_tasks: 已调度任务列表
            satellite_states: 卫星资源状态（如果提供则检查资源约束）
            early_termination: 是否早期终止

        Returns:
            UnifiedBatchResult列表
        """
        if not candidates:
            return []

        self._stats['fast_phase_checks'] += len(candidates)

        results = [UnifiedBatchResult(feasible=True) for _ in candidates]

        # 阶段0: 分辨率约束检查（最快速，优先执行）
        for i, candidate in enumerate(candidates):
            if not self._check_resolution_constraint(candidate):
                results[i].feasible = False
                results[i].reason = "Resolution constraint: insufficient resolution for target"

        # 注意：姿态检查已在GreedyScheduler中完成，这里跳过
        # 检查SAA、时间冲突约束
        active_indices = [i for i in range(len(candidates)) if results[i].feasible]

        # SAA检查
        logger.debug(f"[UnifiedBatch] Converting {len(active_indices)} candidates to SAA format...")
        saa_candidates = self._convert_to_saa_candidates(
            [candidates[i] for i in active_indices]
        )
        logger.debug(f"[UnifiedBatch] Calling SAA batch check with {len(saa_candidates)} candidates...")
        saa_results = self._saa_checker.check_window_feasibility_batch(saa_candidates)
        logger.debug(f"[UnifiedBatch] SAA check complete, got {len(saa_results)} results")

        for idx, saa_result in zip(active_indices, saa_results):
            results[idx].saa_result = saa_result
            results[idx].saa_feasible = saa_result.feasible
            if not saa_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"SAA: {saa_result.violation_count} violations"

        # 时间检查（快速阶段也做，因为很快）
        active_indices = [i for i in active_indices if results[i].feasible]
        logger.debug(f"[UnifiedBatch] {len(active_indices)} candidates remaining after SAA filter")
        if active_indices:
            logger.debug(f"[UnifiedBatch] Converting to time conflict format...")
            time_candidates = self._convert_to_time_candidates(
                [candidates[i] for i in active_indices]
            )
            logger.debug(f"[UnifiedBatch] Calling time conflict batch check with {len(time_candidates)} candidates...")
            time_results = self._time_checker.check_time_conflict_batch(
                time_candidates, existing_tasks
            )
            logger.debug(f"[UnifiedBatch] Time conflict check complete, got {len(time_results)} results")

            for idx, time_result in zip(active_indices, time_results):
                results[idx].time_result = time_result
                results[idx].time_feasible = time_result.feasible
                if not time_result.feasible:
                    results[idx].feasible = False
                    results[idx].reason = f"Time: {time_result.reason}"

        # 资源检查（如果提供了卫星状态）- 批量优化
        if satellite_states is not None:
            active_indices = [i for i in active_indices if results[i].feasible]
            logger.debug(f"[UnifiedBatch] {len(active_indices)} candidates for resource check")
            if active_indices:
                resource_candidates = self._convert_to_resource_candidates(
                    [candidates[i] for i in active_indices]
                )
                logger.debug(f"[UnifiedBatch] Calling resource batch check with {len(resource_candidates)} candidates...")
                resource_results = self._resource_checker.check_resources_batch(
                    resource_candidates, satellite_states
                )
                logger.debug(f"[UnifiedBatch] Resource check complete, got {len(resource_results)} results")

                for idx, resource_result in zip(active_indices, resource_results):
                    results[idx].resource_result = resource_result
                    results[idx].resource_feasible = resource_result.feasible
                    if not resource_result.feasible:
                        results[idx].feasible = False
                        results[idx].reason = f"Resource: {resource_result.reason}"

        feasible_count = sum(1 for r in results if r.feasible)
        logger.debug(f"[UnifiedBatch] Fast phase complete: {feasible_count}/{len(results)} candidates feasible")
        return results

    def _convert_to_slew_candidates(
        self,
        candidates: List[UnifiedBatchCandidate]
    ) -> List[BatchSlewCandidate]:
        """转换为姿态机动候选"""
        return [
            BatchSlewCandidate(
                sat_id=c.sat_id,
                satellite=c.satellite,
                target=c.target,
                window_start=c.window_start,
                window_end=c.window_end,
                prev_end_time=c.prev_end_time,
                prev_target=c.prev_target,
                imaging_duration=c.imaging_duration,
                sat_position=c.sat_position,
                sat_velocity=c.sat_velocity,
                imaging_begin=c.imaging_begin
            )
            for c in candidates
        ]

    def _convert_to_saa_candidates(
        self,
        candidates: List[UnifiedBatchCandidate]
    ) -> List[BatchSAACandidate]:
        """转换为SAA候选"""
        return [
            BatchSAACandidate(
                sat_id=c.sat_id,
                window_start=c.window_start,
                window_end=c.window_end,
                sample_interval=60.0
            )
            for c in candidates
        ]

    def _convert_to_time_candidates(
        self,
        candidates: List[UnifiedBatchCandidate]
    ) -> List[BatchTimeConflictCandidate]:
        """转换为时间冲突候选"""
        return [
            BatchTimeConflictCandidate(
                sat_id=c.sat_id,
                window_start=c.window_start,
                window_end=c.window_end,
                imaging_duration=c.imaging_duration
            )
            for c in candidates
        ]

    def _convert_to_resource_candidates(
        self,
        candidates: List[UnifiedBatchCandidate]
    ) -> List[BatchResourceCandidate]:
        """转换为资源候选"""
        return [
            BatchResourceCandidate(
                sat_id=c.sat_id,
                power_needed=c.power_needed,
                storage_produced=c.storage_produced
            )
            for c in candidates
        ]

    def _convert_to_orbit_candidates(
        self,
        candidates: List[UnifiedBatchCandidate]
    ) -> List[BatchOrbitConstraintCandidate]:
        """转换为单圈约束候选"""
        return [
            BatchOrbitConstraintCandidate(
                sat_id=c.sat_id,
                window_start=c.window_start,
                window_end=c.window_end,
                imaging_duration=c.imaging_duration
            )
            for c in candidates
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取检查统计信息"""
        return {
            **self._stats,
            'slew_stats': self._slew_checker.get_batch_stats(),
            'saa_stats': self._saa_checker.get_batch_stats(),
            'time_stats': self._time_checker.get_batch_stats(),
            'resource_stats': self._resource_checker.get_batch_stats(),
            'orbit_stats': self._orbit_checker.get_batch_stats()
        }

    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            'total_checks': 0,
            'fast_phase_checks': 0,
            'full_phase_checks': 0,
            'early_terminations': 0
        }
        self._slew_checker.reset_batch_stats()
        self._saa_checker.reset_batch_stats()
        self._time_checker.reset_batch_stats()
        self._resource_checker.reset_batch_stats()
        self._orbit_checker.reset_batch_stats()

    def _check_resolution_constraint(self, candidate: UnifiedBatchCandidate) -> bool:
        """
        检查分辨率约束

        检查卫星是否有成像模式能满足目标的分辨率要求。

        Args:
            candidate: 统一候选对象

        Returns:
            True if resolution constraint is satisfied, False otherwise
        """
        target = candidate.target
        satellite = candidate.satellite

        # 获取目标分辨率需求
        required_resolution = getattr(target, 'resolution_required', None)
        if required_resolution is None:
            return True  # 无分辨率要求，直接通过

        # 使用卫星能力的can_satisfy_resolution方法检查
        return satellite.capabilities.can_satisfy_resolution(required_resolution)
