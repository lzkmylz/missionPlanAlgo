"""
批量姿态机动约束检查器

继承 PreciseSlewConstraintChecker 的接口，但使用批量计算优化。
与基类保持API兼容，可无缝替换使用。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging

from core.models.mission import Mission
from core.models.satellite import Satellite
from core.models.target import Target
from core.dynamics.precise import (
    PreciseSlewCalculator, SatelliteDynamicsConfig,
    AttitudeState, Quaternion, AngularVelocity, MomentumState
)

from .precise_slew_constraint_checker import (
    PreciseSlewConstraintChecker, SlewFeasibilityResult
)
from .batch_slew_calculator import (
    BatchSlewCandidate, BatchSlewResult, BatchSlewCalculator, BatchSlewData
)

logger = logging.getLogger(__name__)


class BatchSlewConstraintChecker(PreciseSlewConstraintChecker):
    """批量姿态机动约束检查器

    继承 PreciseSlewConstraintChecker 以复用配置提取和基础逻辑，
    但使用批量计算优化核心性能瓶颈。

    使用方法:
        # 创建检查器
        checker = BatchSlewConstraintChecker(mission)

        # 单候选检查（向后兼容）
        result = checker.check_slew_feasibility(...)

        # 批量检查（优化性能）
        batch_results = checker.check_slew_feasibility_batch(candidates)
    """

    def __init__(
        self,
        mission: Mission,
        use_precise_model: bool = True,
        precise_calculators: Optional[Dict[str, PreciseSlewCalculator]] = None,
        skip_reset_calculation: bool = False
    ):
        """初始化批量约束检查器

        Args:
            mission: 任务对象
            use_precise_model: 是否使用精确模型（批量版本始终使用优化计算）
            precise_calculators: 预计算的精确计算器字典
            skip_reset_calculation: 是否跳过姿态复位计算
        """
        # 调用父类初始化
        super().__init__(
            mission=mission,
            use_precise_model=use_precise_model,
            precise_calculators=precise_calculators,
            skip_reset_calculation=skip_reset_calculation
        )

        # 创建批量计算器
        self._batch_calculator = BatchSlewCalculator()

        # 性能统计
        self._batch_stats = {
            'batch_calls': 0,
            'total_candidates': 0,
            'avg_batch_size': 0.0
        }

        logger.info("BatchSlewConstraintChecker initialized with batch optimization")

    def check_slew_feasibility_batch(
        self,
        candidates: List[BatchSlewCandidate],
        current_attitudes: Optional[Dict[str, AttitudeState]] = None
    ) -> List[SlewFeasibilityResult]:
        """批量检查机动可行性

        这是优化的批量版本，使用Numba加速计算。

        Args:
            candidates: 候选任务列表
            current_attitudes: 当前卫星姿态状态 {sat_id: AttitudeState}

        Returns:
            SlewFeasibilityResult列表，与输入候选顺序一致
        """
        if not candidates:
            return []

        # 更新统计
        self._batch_stats['batch_calls'] += 1
        self._batch_stats['total_candidates'] += len(candidates)
        self._batch_stats['avg_batch_size'] = (
            self._batch_stats['total_candidates'] / self._batch_stats['batch_calls']
        )

        # 获取当前姿态状态
        if current_attitudes is None:
            current_attitudes = self._get_all_current_attitudes()

        # 准备批量数据
        data = self._batch_calculator.prepare_batch_data(candidates, current_attitudes)

        # 执行批量计算
        batch_results = self._batch_calculator.compute_batch(data)

        # 转换结果格式
        results = []
        for i, (cand, batch_result) in enumerate(zip(candidates, batch_results)):
            result = self._convert_batch_result(
                cand, batch_result, current_attitudes.get(cand.sat_id)
            )
            results.append(result)

            # 更新状态跟踪（重要：保持与基类一致的状态更新）
            if result.feasible:
                self._update_state_after_batch(cand, result)

        return results

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
        """单候选检查（向后兼容）

        内部调用批量版本以保持一致性。

        Args:
            satellite_id: 卫星ID
            prev_target: 上一个目标
            current_target: 当前目标
            prev_end_time: 上一任务结束时间
            window_start: 窗口开始时间
            imaging_duration: 成像持续时间
            **kwargs: 额外参数

        Returns:
            SlewFeasibilityResult
        """
        # 创建单候选列表，调用批量版本
        # 这样可以确保一致性，同时复用批量优化

        # 获取必要数据
        satellite = self.mission.get_satellite_by_id(satellite_id)
        if not satellite:
            return self._error_result(f"Satellite {satellite_id} not found", window_start)

        # 获取卫星位置
        sat_position, sat_velocity = self._get_satellite_position(satellite, prev_end_time)

        # 获取窗口结束时间
        window_end = kwargs.get('window_end', window_start)
        if isinstance(window_end, datetime) and window_end > window_start:
            pass
        else:
            # 默认窗口持续5分钟
            window_end = window_start + __import__('datetime').timedelta(minutes=5)

        # 创建候选对象
        candidate = BatchSlewCandidate(
            sat_id=satellite_id,
            satellite=satellite,
            target=current_target,
            window_start=window_start,
            window_end=window_end,
            prev_end_time=prev_end_time,
            prev_target=prev_target,
            imaging_duration=imaging_duration,
            sat_position=sat_position,
            sat_velocity=sat_velocity
        )

        # 调用批量版本
        results = self.check_slew_feasibility_batch([candidate])

        return results[0] if results else self._error_result("Batch computation failed", window_start)

    def _get_all_current_attitudes(self) -> Dict[str, AttitudeState]:
        """获取所有卫星的当前姿态状态

        Returns:
            {sat_id: AttitudeState}
        """
        attitudes = {}

        for sat in self.mission.satellites:
            sat_id = sat.id

            # 尝试从状态跟踪器获取
            if self._state_tracker is not None:
                attitude_data = self._state_tracker.get_attitude_state(sat_id)
                momentum = self._state_tracker.get_angular_momentum(sat_id)

                if attitude_data is not None:
                    # 转换为AttitudeState
                    q_data = attitude_data.get('quaternion', [1, 0, 0, 0])
                    if isinstance(q_data, np.ndarray):
                        q = Quaternion(w=q_data[0], x=q_data[1], y=q_data[2], z=q_data[3])
                    elif hasattr(q_data, 'w'):
                        q = q_data
                    else:
                        q = Quaternion(w=q_data[0], x=q_data[1], y=q_data[2], z=q_data[3])

                    av_data = attitude_data.get('angular_velocity', [0, 0, 0])
                    if isinstance(av_data, np.ndarray):
                        av = AngularVelocity(x=av_data[0], y=av_data[1], z=av_data[2])
                    elif hasattr(av_data, 'x'):
                        av = av_data
                    else:
                        av = AngularVelocity(x=av_data[0], y=av_data[1], z=av_data[2])

                    attitudes[sat_id] = AttitudeState(
                        quaternion=q,
                        angular_velocity=av,
                        timestamp=datetime.now(),
                        momentum=MomentumState(
                            wheel_speeds=np.zeros(4),
                            wheel_inertias=np.full(4, 0.01),
                            total_momentum=momentum
                        ) if momentum is not None else None
                    )
                    continue

            # 从内部缓存获取
            if sat_id in self._last_attitudes:
                attitudes[sat_id] = self._last_attitudes[sat_id]
                continue

            # 使用默认对地定向
            attitudes[sat_id] = AttitudeState(
                quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
                angular_velocity=AngularVelocity(0.0, 0.0, 0.0),
                timestamp=datetime.now(),
                momentum=None
            )

        return attitudes

    def _convert_batch_result(
        self,
        candidate: BatchSlewCandidate,
        batch_result: BatchSlewResult,
        current_attitude: Optional[AttitudeState]
    ) -> SlewFeasibilityResult:
        """将批量结果转换为SlewFeasibilityResult格式

        Args:
            candidate: 原始候选
            batch_result: 批量计算结果
            current_attitude: 当前姿态

        Returns:
            SlewFeasibilityResult
        """
        # 创建标准结果对象
        result = SlewFeasibilityResult(
            feasible=batch_result.feasible,
            slew_angle=batch_result.slew_angle,
            slew_time=batch_result.slew_time,
            actual_start=batch_result.actual_start,
            reason=batch_result.reason,
            reset_time=batch_result.reset_time
        )

        # 添加精确模型信息（向后兼容）
        result.energy_consumption = batch_result.energy_consumption
        result.momentum_margin = batch_result.momentum_margin

        return result

    def _update_state_after_batch(
        self,
        candidate: BatchSlewCandidate,
        result: SlewFeasibilityResult
    ):
        """批量计算后更新状态跟踪

        保持与基类的状态更新逻辑一致。

        Args:
            candidate: 候选
            result: 计算结果
        """
        sat_id = candidate.sat_id

        # 计算目标姿态
        try:
            sat_position = candidate.sat_position
            target_attitude = self._compute_target_attitude(
                candidate.satellite, candidate.target, sat_position
            )

            # 更新状态跟踪
            self._update_satellite_state(
                satellite_id=sat_id,
                time=result.actual_start,
                attitude=target_attitude,
                angular_momentum=None  # 批量版本简化处理
            )
        except Exception as e:
            logger.debug(f"Failed to update state for {sat_id}: {e}")

    def get_batch_stats(self) -> Dict[str, Any]:
        """获取批量计算统计信息

        Returns:
            统计字典
        """
        return {
            **self._batch_stats,
            'use_numba': self._batch_calculator.use_numba
        }

    def reset_batch_stats(self):
        """重置批量计算统计"""
        self._batch_stats = {
            'batch_calls': 0,
            'total_candidates': 0,
            'avg_batch_size': 0.0
        }
