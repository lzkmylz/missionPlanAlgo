"""
统一批量约束检查器

整合姿态、SAA、时间、资源批量检查，提供单一入口进行所有约束的批量检查。
支持分阶段过滤（先快速检查，后详细检查）。
"""

import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple

if TYPE_CHECKING:
    from .batch_uplink_calculator import BatchUplinkResult as _BatchUplinkResult
from dataclasses import dataclass, field, replace as dataclass_replace
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
from .pmc_constraint_checker import (
    PMCConstraintChecker,
    PMCCandidate,
    PMCConstraintResult
)
from .multi_strip_constraint_checker import (
    MultiStripConstraintChecker,
    MultiStripCandidate,
    MultiStripConstraintResult,
)

from .slew_constraint_checker import SlewFeasibilityResult
from .saa_constraint_checker import SAAFeasibilityResult

logger = logging.getLogger(__name__)

# 多条带拼幅目标参数覆盖的合法范围上限（重叠比）。
# 超过此值的目标 required_swath_overlap_ratio 将被静默忽略并保留卫星默认配置，
# 同时记录 WARNING 日志告知调用方。
_MAX_MOSAIC_OVERLAP_RATIO: float = 0.30


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

    # PMC模式信息（新增）
    is_pmc_mode: bool = False  # 是否为PMC模式
    pmc_config: Optional[Any] = None  # PMC配置（如果是PMC模式）
    imaging_mode: Optional[str] = None  # 成像模式名称

    # 单次多条带拼幅模式信息
    is_mosaic_mode: bool = False  # 是否为多条带拼幅模式
    mosaic_config: Optional[Any] = None  # MultiStripMosaicConfig

    # 指令上注约束（可选，默认不检查）
    check_uplink: bool = False                   # 是否启用上注约束检查（opt-in）
    required_uplink_s: float = 0.0               # 所需上注时长（秒），由调度器在构建时解析
    uplink_command_lead_time_s: float = 300.0    # 指令必须在任务开始前多少秒到达

    def __post_init__(self):
        """初始化后处理：默认 imaging_begin = window_start"""
        if self.imaging_begin is None:
            self.imaging_begin = self.window_start
        # 从卫星载荷检查是否是PMC模式
        if self.imaging_mode and not self.is_pmc_mode:
            try:
                mode_cfg = self.satellite.payload_config.get_mode_config(self.imaging_mode)
                self.is_pmc_mode = mode_cfg.is_pmc_mode()
                if self.is_pmc_mode:
                    from core.models.pmc_config import PitchMotionCompensationConfig
                    pmc_params = mode_cfg.get_pmc_params()
                    self.pmc_config = PitchMotionCompensationConfig(
                        speed_reduction_ratio=pmc_params.get('speed_reduction_ratio', 0.25),
                        pitch_rate_dps=pmc_params.get('pitch_rate_dps'),
                        min_altitude_m=pmc_params.get('min_altitude_m', 400000.0),
                        max_roll_angle_deg=pmc_params.get('max_roll_angle_deg', 30.0),
                    )
            except (ValueError, AttributeError):
                pass
        # 检查是否是多条带拼幅模式
        if self.imaging_mode == 'single_pass_mosaic' and not self.is_mosaic_mode:
            self.is_mosaic_mode = True
            if self.mosaic_config is None:
                # 尝试从卫星配置中读取
                try:
                    cfg = self.satellite.capabilities.get_mosaic_config()
                    if cfg is not None:
                        self.mosaic_config = cfg
                    else:
                        from core.models.multi_strip_mosaic_config import MultiStripMosaicConfig
                        self.mosaic_config = MultiStripMosaicConfig()
                except (AttributeError, ValueError):
                    from core.models.multi_strip_mosaic_config import MultiStripMosaicConfig
                    self.mosaic_config = MultiStripMosaicConfig()

            # 用目标级参数覆盖卫星默认配置（HIGH-3）
            # target.mosaic_strip_count 和 target.required_swath_overlap_ratio
            # 优先级高于卫星默认值
            if self.target is not None and self.mosaic_config is not None:
                from core.models.multi_strip_mosaic_config import MultiStripMosaicConfig
                tgt_strips = getattr(self.target, 'mosaic_strip_count', None)
                tgt_overlap = getattr(self.target, 'required_swath_overlap_ratio', None)
                # 只有目标明确指定且与当前配置不同时才覆盖
                needs_override = (
                    (tgt_strips is not None and tgt_strips != self.mosaic_config.num_strips)
                    or (tgt_overlap is not None
                        and tgt_overlap != self.mosaic_config.overlap_ratio)
                )
                if needs_override:
                    new_strips = tgt_strips if tgt_strips is not None else self.mosaic_config.num_strips
                    new_overlap = tgt_overlap if tgt_overlap is not None else self.mosaic_config.overlap_ratio
                    # 验证范围合法后构建新配置；不合法则保留卫星默认值并记录 WARNING
                    if 2 <= new_strips <= 8 and 0.0 <= new_overlap <= _MAX_MOSAIC_OVERLAP_RATIO:
                        self.mosaic_config = dataclass_replace(
                            self.mosaic_config,
                            num_strips=new_strips,
                            overlap_ratio=new_overlap,
                        )
                    else:
                        tgt_id = getattr(self.target, 'id', '<unknown>')
                        logger.warning(
                            "目标 %s 的 mosaic 参数超出合法范围（strips=%s ∉ [2,8] 或 "
                            "overlap=%.2f ∉ [0, %.2f]），将忽略目标覆盖，继续使用卫星默认配置。",
                            tgt_id, new_strips, new_overlap, _MAX_MOSAIC_OVERLAP_RATIO,
                        )

        # 第二重保障：is_mosaic_mode=True 但 mosaic_config 仍为 None（外部构造时可能发生）
        if self.is_mosaic_mode and self.mosaic_config is None:
            from core.models.multi_strip_mosaic_config import MultiStripMosaicConfig
            self.mosaic_config = MultiStripMosaicConfig()


@dataclass
class UnifiedBatchResult:
    """统一批量约束检查结果"""
    feasible: bool
    slew_result: Optional[SlewFeasibilityResult] = None
    saa_result: Optional[SAAFeasibilityResult] = None
    time_result: Optional[BatchTimeConflictResult] = None
    resource_result: Optional[BatchResourceResult] = None
    orbit_result: Optional[BatchOrbitConstraintResult] = None  # 单圈约束结果
    pmc_result: Optional[PMCConstraintResult] = None  # PMC约束结果（新增）
    mosaic_result: Optional[MultiStripConstraintResult] = None  # 多条带拼幅约束结果
    reason: Optional[str] = None

    # 详细结果
    slew_feasible: bool = True
    saa_feasible: bool = True
    time_feasible: bool = True
    resource_feasible: bool = True
    orbit_feasible: bool = True  # 单圈约束是否满足
    pmc_feasible: bool = True  # PMC约束是否满足（新增）
    mosaic_feasible: bool = True  # 多条带拼幅约束是否满足
    uplink_feasible: bool = True  # 指令上注约束是否满足
    uplink_result: Optional['_BatchUplinkResult'] = None


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
        consider_storage: bool = True,
        uplink_registry=None,  # UplinkWindowRegistry，可选；非 None 时启用上注约束
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
        self._orbit_checker = BatchOrbitConstraintChecker(mission)
        self._pmc_checker = PMCConstraintChecker()  # 新增：PMC约束检查器
        self._mosaic_checker = MultiStripConstraintChecker()  # 多条带拼幅约束检查器

        # 指令上注约束检查器（可选）
        self._uplink_checker = None
        if uplink_registry is not None:
            try:
                from .batch_uplink_constraint_checker import BatchUplinkConstraintChecker
                self._uplink_checker = BatchUplinkConstraintChecker(uplink_registry)
                logger.info("UnifiedBatchConstraintChecker: 指令上注约束已启用")
            except Exception as e:
                logger.warning("指令上注约束初始化失败: %s", e)

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

        # 阶段1: PMC约束检查（如果是PMC模式）
        pmc_indices = [i for i in active_indices if candidates[i].is_pmc_mode]
        if pmc_indices:
            pmc_candidates = self._convert_to_pmc_candidates(
                [candidates[i] for i in pmc_indices]
            )
            pmc_results = self._pmc_checker.check_pmc_feasibility_batch(pmc_candidates)

            for idx, pmc_result in zip(pmc_indices, pmc_results):
                results[idx].pmc_result = pmc_result
                results[idx].pmc_feasible = pmc_result.feasible
                if not pmc_result.feasible:
                    results[idx].feasible = False
                    results[idx].reason = f"PMC constraint: {pmc_result.reason}"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
            if not active_indices:
                return results

        # 阶段2: 多条带拼幅约束检查（如果是mosaic模式）
        self._run_mosaic_phase(candidates, results, active_indices)

        # 早期终止检查
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
            if not active_indices:
                return results

        # 阶段3: 姿态机动约束检查（通常最耗时，但已批量优化）
        active_candidates = [candidates[i] for i in active_indices]
        slew_candidates = self._convert_to_slew_candidates(active_candidates)
        slew_results = self._slew_checker.check_slew_feasibility_batch(
            slew_candidates, current_attitudes
        )

        for idx, slew_result in zip(active_indices, slew_results):
            results[idx].slew_result = slew_result
            results[idx].slew_feasible = slew_result.feasible
            if not slew_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"Slew constraint: {slew_result.reason}"

        # 早期终止检查
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
            if not active_indices:
                return results

        # 阶段4: SAA约束检查
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

        # 阶段5: 时间冲突检查
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

        # 阶段6: 资源约束检查
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

        # 阶段7: 单圈约束检查（最后阶段，最耗时，前面约束通过后再检查）
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

        # 阶段8: 指令上注约束检查（可选，仅 check_uplink=True 的候选）
        if early_termination:
            active_indices = [i for i in active_indices if results[i].feasible]
        self._run_uplink_phase(candidates, results, active_indices)

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
        # 检查多条带拼幅、SAA、时间冲突约束
        active_indices = [i for i in range(len(candidates)) if results[i].feasible]

        # 阶段1: 多条带拼幅约束检查（如果是mosaic模式，作为早期门控）
        self._run_mosaic_phase(candidates, results, active_indices)

        active_indices = [i for i in active_indices if results[i].feasible]
        if early_termination and not active_indices:
            return results

        # 阶段2: SAA检查
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

        # 阶段3: 时间检查（快速阶段也做，因为很快）
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

        # 阶段4: 资源检查（如果提供了卫星状态）- 批量优化
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

        # 阶段5: 指令上注约束检查（可选，仅 check_uplink=True 的候选）
        active_indices = [i for i in range(len(candidates)) if results[i].feasible]
        self._run_uplink_phase(candidates, results, active_indices)

        feasible_count = sum(1 for r in results if r.feasible)
        logger.debug(f"[UnifiedBatch] Fast phase complete: {feasible_count}/{len(results)} candidates feasible")
        return results

    def _run_mosaic_phase(
        self,
        candidates: List[UnifiedBatchCandidate],
        results: List[UnifiedBatchResult],
        active_indices: List[int],
    ) -> None:
        """执行多条带拼幅约束检查阶段（供 check_all_constraints_batch 和 check_fast_phase_batch 共用）

        直接修改 results 中对应位置的 mosaic_result / mosaic_feasible / feasible / reason 字段。
        跳过 mosaic_config 为 None 的候选（标记 infeasible 并给出原因）。
        """
        mosaic_indices = [i for i in active_indices if candidates[i].is_mosaic_mode]
        if not mosaic_indices:
            return

        mosaic_cands = []
        invalid_indices = []
        for i in mosaic_indices:
            if candidates[i].mosaic_config is None:
                invalid_indices.append(i)
            else:
                mosaic_cands.append((i, MultiStripCandidate(
                    sat_id=candidates[i].sat_id,
                    satellite=candidates[i].satellite,
                    target=candidates[i].target,
                    window_start=candidates[i].window_start,
                    window_end=candidates[i].window_end,
                    mosaic_config=candidates[i].mosaic_config,
                    sat_position=candidates[i].sat_position,
                    sat_velocity=candidates[i].sat_velocity,
                )))

        # 防御性检查：__post_init__ 中 except 分支保证了 mosaic_config 不会为 None，
        # 此处仅作保障，若触发则说明存在编程错误。
        for i in invalid_indices:
            results[i].feasible = False
            results[i].mosaic_feasible = False
            results[i].reason = "Mosaic constraint: mosaic_config is None"

        if not mosaic_cands:
            return

        valid_indices = [t[0] for t in mosaic_cands]
        cands = [t[1] for t in mosaic_cands]
        mosaic_results = self._mosaic_checker.check_feasibility_batch(list(cands))

        for idx, mosaic_result in zip(valid_indices, mosaic_results):
            results[idx].mosaic_result = mosaic_result
            results[idx].mosaic_feasible = mosaic_result.feasible
            if not mosaic_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"Mosaic constraint: {mosaic_result.reason}"

    def _run_uplink_phase(
        self,
        candidates: List[UnifiedBatchCandidate],
        results: List[UnifiedBatchResult],
        active_indices: List[int],
    ) -> None:
        """执行指令上注约束检查阶段（供 check_all_constraints_batch 和 check_fast_phase_batch 共用）

        仅处理 candidate.check_uplink=True 的候选。
        直接修改 results 中对应位置的 uplink_result / uplink_feasible / feasible / reason 字段。
        若 _uplink_checker 未初始化则静默跳过。
        """
        if self._uplink_checker is None:
            return

        uplink_indices = [i for i in active_indices if candidates[i].check_uplink]
        if not uplink_indices:
            return

        from .batch_uplink_calculator import BatchUplinkCandidate as _UplinkCand
        uplink_cands = [
            _UplinkCand(
                sat_id=candidates[i].sat_id,
                task_start=candidates[i].window_start,
                required_uplink_s=candidates[i].required_uplink_s,
                command_lead_time_s=candidates[i].uplink_command_lead_time_s,
            )
            for i in uplink_indices
        ]
        uplink_results = self._uplink_checker.check_uplink_feasibility_batch(uplink_cands)
        for idx, uplink_result in zip(uplink_indices, uplink_results):
            results[idx].uplink_result = uplink_result
            results[idx].uplink_feasible = uplink_result.feasible
            if not uplink_result.feasible:
                results[idx].feasible = False
                results[idx].reason = f"Uplink constraint: {uplink_result.reason}"

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

    def _convert_to_pmc_candidates(
        self,
        candidates: List[UnifiedBatchCandidate]
    ) -> List[PMCCandidate]:
        """转换为PMC约束候选"""
        pmc_candidates = []
        for c in candidates:
            if c.is_pmc_mode and c.pmc_config:
                pmc_candidates.append(PMCCandidate(
                    sat_id=c.sat_id,
                    satellite=c.satellite,
                    target=c.target,
                    imaging_start=c.imaging_begin or c.window_start,
                    imaging_duration_s=c.imaging_duration,
                    pmc_config=c.pmc_config
                ))
        return pmc_candidates

    def get_stats(self) -> Dict[str, Any]:
        """获取检查统计信息（含所有子检查器统计）"""
        return {
            **self._stats,
            'slew_stats': self._slew_checker.get_batch_stats(),
            'saa_stats': self._saa_checker.get_batch_stats(),
            'time_stats': self._time_checker.get_batch_stats(),
            'resource_stats': self._resource_checker.get_batch_stats(),
            'orbit_stats': self._orbit_checker.get_batch_stats(),
            'mosaic_stats': self._mosaic_checker.get_stats(),
            'pmc_stats': self._pmc_checker.get_stats() if hasattr(self._pmc_checker, 'get_stats') else {},
        }

    def reset_stats(self):
        """重置所有子检查器统计信息"""
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
        self._pmc_checker.reset_stats()
        self._mosaic_checker.reset_stats()

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
