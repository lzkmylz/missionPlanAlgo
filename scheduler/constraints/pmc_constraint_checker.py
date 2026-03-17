"""
PMC (Pitch Motion Compensation) 约束检查器

检查主动前向推扫模式任务的可行性和约束满足情况。
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.pmc_config import PitchMotionCompensationConfig
from core.dynamics.pmc_calculator import PMCCalculator, PMCImagingSequence

logger = logging.getLogger(__name__)


@dataclass
class PMCConstraintResult:
    """PMC约束检查结果"""
    feasible: bool
    sequence: Optional[PMCImagingSequence] = None
    pitch_violations: List[str] = None
    roll_violations: List[str] = None
    altitude_feasible: bool = True
    pitch_rate_feasible: bool = True
    snr_change_db: float = 0.0  # SNR变化（正值为增益，负值为损失）
    effective_integration_time_s: float = 0.0
    estimated_energy_wh: float = 0.0
    reason: Optional[str] = None

    def __post_init__(self):
        if self.pitch_violations is None:
            self.pitch_violations = []
        if self.roll_violations is None:
            self.roll_violations = []


@dataclass
class PMCCandidate:
    """PMC约束检查候选"""
    sat_id: str
    satellite: Satellite
    target: Target
    imaging_start: datetime
    imaging_duration_s: float
    pmc_config: PitchMotionCompensationConfig


class PMCConstraintChecker:
    """
    PMC约束检查器

    检查PMC模式任务的可行性，包括:
    - 轨道高度适用性
    - 俯仰角速度可行性
    - 姿态角约束（滚转/俯仰）
    - 能耗估算
    """

    def __init__(self):
        """初始化PMC约束检查器"""
        self.calculator = PMCCalculator()
        self._stats = {
            'total_checks': 0,
            'feasible_count': 0,
            'infeasible_count': 0,
        }

    def check_pmc_feasibility(self, candidate: PMCCandidate) -> PMCConstraintResult:
        """
        检查单个PMC候选的可行性

        Args:
            candidate: PMC候选

        Returns:
            PMCConstraintResult
        """
        self._stats['total_checks'] += 1

        orbit_altitude = candidate.satellite.orbit.get_semi_major_axis() - 6371000.0

        # 1. 检查轨道高度
        altitude_ok, altitude_msg = candidate.pmc_config.check_altitude_feasibility(
            orbit_altitude
        )
        if not altitude_ok:
            self._stats['infeasible_count'] += 1
            return PMCConstraintResult(
                feasible=False,
                altitude_feasible=False,
                reason=altitude_msg
            )

        # 2. 检查俯仰角速度
        pitch_rate = candidate.pmc_config.get_pitch_rate(orbit_altitude)
        if abs(pitch_rate) > 2.0:  # 2度/秒为上限（取绝对值支持反向模式）
            self._stats['infeasible_count'] += 1
            return PMCConstraintResult(
                feasible=False,
                pitch_rate_feasible=False,
                reason=f"Required pitch rate {pitch_rate:.2f}°/s exceeds 2°/s limit"
            )

        # 3. 计算成像序列
        try:
            sequence = self.calculator.calculate_imaging_sequence(
                satellite=candidate.satellite,
                target=candidate.target,
                imaging_start=candidate.imaging_start,
                imaging_duration_s=candidate.imaging_duration_s,
                pmc_config=candidate.pmc_config
            )
        except Exception as e:
            self._stats['infeasible_count'] += 1
            return PMCConstraintResult(
                feasible=False,
                reason=f"Failed to calculate imaging sequence: {str(e)}"
            )

        # 4. 检查姿态约束
        pitch_ok, pitch_violations = self.calculator.check_pitch_constraints(
            sequence, candidate.pmc_config.max_pitch_angle_deg
        )
        roll_ok, roll_violations = self.calculator.check_roll_constraints(
            sequence, candidate.pmc_config.max_roll_angle_deg
        )

        if not pitch_ok or not roll_ok:
            self._stats['infeasible_count'] += 1
            violations = pitch_violations + roll_violations
            return PMCConstraintResult(
                feasible=False,
                sequence=sequence,
                pitch_violations=pitch_violations,
                roll_violations=roll_violations,
                reason=f"Attitude constraint violations: {len(violations)}"
            )

        # 5. 计算性能指标
        snr_change = self.calculator.calculate_snr_change_db(
            candidate.pmc_config.speed_reduction_ratio
        )
        effective_time = self.calculator.calculate_effective_integration_time(
            candidate.imaging_duration_s,
            candidate.pmc_config.speed_reduction_ratio
        )

        # 估算能耗
        base_power = 150.0  # 默认值
        energy_wh = self.calculator.estimate_power_consumption(
            base_power, candidate.imaging_duration_s, candidate.pmc_config
        )

        self._stats['feasible_count'] += 1

        return PMCConstraintResult(
            feasible=True,
            sequence=sequence,
            pitch_violations=[],
            roll_violations=[],
            altitude_feasible=True,
            pitch_rate_feasible=True,
            snr_change_db=snr_change,
            effective_integration_time_s=effective_time,
            estimated_energy_wh=energy_wh
        )

    def check_pmc_feasibility_batch(
        self,
        candidates: List[PMCCandidate]
    ) -> List[PMCConstraintResult]:
        """
        批量检查PMC候选

        Args:
            candidates: PMC候选列表

        Returns:
            PMCConstraintResult列表
        """
        return [self.check_pmc_feasibility(c) for c in candidates]

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self._stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            'total_checks': 0,
            'feasible_count': 0,
            'infeasible_count': 0,
        }


def check_pmc_mode_for_task(
    satellite: Satellite,
    target: Target,
    imaging_start: datetime,
    imaging_duration_s: float,
    mode_config: Any
) -> Tuple[bool, Dict[str, Any]]:
    """
    便捷函数：检查任务是否适合PMC模式

    Args:
        satellite: 卫星对象
        target: 目标对象
        imaging_start: 成像开始时间
        imaging_duration_s: 成像时长
        mode_config: 成像模式配置（应包含PMC参数）

    Returns:
        (是否可行, 详细信息)
    """
    # 检查是否是PMC模式
    if not hasattr(mode_config, 'is_pmc_mode') or not mode_config.is_pmc_mode():
        return False, {"error": "Not a PMC mode"}

    pmc_params = mode_config.get_pmc_params()
    pmc_config = PitchMotionCompensationConfig(
        speed_reduction_ratio=pmc_params.get('speed_reduction_ratio', 0.25),
        pitch_rate_dps=pmc_params.get('pitch_rate_dps'),
        direction=pmc_params.get('direction', 'forward'),
        min_altitude_m=pmc_params.get('min_altitude_m', 400000.0),
        max_roll_angle_deg=pmc_params.get('max_roll_angle_deg', 30.0),
    )

    candidate = PMCCandidate(
        sat_id=satellite.id,
        satellite=satellite,
        target=target,
        imaging_start=imaging_start,
        imaging_duration_s=imaging_duration_s,
        pmc_config=pmc_config
    )

    checker = PMCConstraintChecker()
    result = checker.check_pmc_feasibility(candidate)

    details = {
        "feasible": result.feasible,
        "snr_change_db": result.snr_change_db,
        "effective_integration_time_s": result.effective_integration_time_s,
        "estimated_energy_wh": result.estimated_energy_wh,
        "altitude_feasible": result.altitude_feasible,
        "pitch_rate_feasible": result.pitch_rate_feasible,
    }

    if not result.feasible:
        details["reason"] = result.reason
        details["pitch_violations"] = result.pitch_violations
        details["roll_violations"] = result.roll_violations

    return result.feasible, details
