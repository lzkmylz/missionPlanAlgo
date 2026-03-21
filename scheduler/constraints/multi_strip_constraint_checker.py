"""
单次多条带拼幅成像约束检查器

检查 single_pass_mosaic 任务的可行性，包括：
- 卫星是否支持该模式
- 窗口时长是否充足
- 条带间机动是否物理可行（角速度不超限、稳定时间足够）
- 所有条带滚转角在允许范围内
- 能耗估算
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import logging

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.multi_strip_mosaic_config import MultiStripMosaicConfig
from core.constants import EARTH_RADIUS_M
from core.dynamics.multi_strip_calculator import MultiStripCalculator, StripPlan

logger = logging.getLogger(__name__)

_MIN_ORBIT_ALTITUDE_M = 200_000.0    # 最低合理LEO轨道高度（米）
_DEFAULT_ORBIT_ALTITUDE_M = 500_000.0  # 默认轨道高度估计（米）
_DEFAULT_BASE_POWER_W = 172.5        # 默认单次拼幅基础功耗（=150W×1.15倍开销）


@dataclass
class MultiStripConstraintResult:
    """多条带拼幅约束检查结果"""
    feasible: bool
    strip_plan: Optional[StripPlan] = None
    required_duration_s: float = 0.0
    window_duration_s: float = 0.0
    roll_violations: List[str] = field(default_factory=list)
    slew_violations: List[str] = field(default_factory=list)
    estimated_energy_wh: float = 0.0
    reason: Optional[str] = None
    degraded_geometry: bool = False   # True when strip center coords are approximate (nadir fallback)


@dataclass
class MultiStripCandidate:
    """多条带拼幅约束检查候选"""
    sat_id: str
    satellite: Satellite
    target: Target
    window_start: datetime
    window_end: datetime
    mosaic_config: MultiStripMosaicConfig
    sat_position: Optional[Tuple[float, float, float]] = None  # ECEF（米）
    sat_velocity: Optional[Tuple[float, float, float]] = None  # ECEF（米/秒）


class MultiStripConstraintChecker:
    """
    单次多条带拼幅成像约束检查器

    按以下顺序执行检查（早期终止）:
    1. 卫星能力检查
    2. 窗口时长充足性检查
    3. 条带间机动可行性检查（角速度 / 稳定时间）
    4. 条带规划（调用 MultiStripCalculator.plan_strips）
    5. 能耗估算
    """

    def __init__(self):
        self._calculator = MultiStripCalculator()
        self._stats = {
            'total_checks': 0,
            'feasible_count': 0,
            'infeasible_count': 0,
        }

    def check_feasibility(
        self,
        candidate: MultiStripCandidate
    ) -> MultiStripConstraintResult:
        """
        检查单个多条带拼幅候选的可行性

        Args:
            candidate: 检查候选

        Returns:
            MultiStripConstraintResult
        """
        self._stats['total_checks'] += 1
        sat = candidate.satellite
        cfg = candidate.mosaic_config
        window_s = (candidate.window_end - candidate.window_start).total_seconds()

        # ------------------------------------------------------------------
        # 检查1: 卫星是否支持该模式
        # ------------------------------------------------------------------
        if not sat.capabilities.supports_single_pass_mosaic():
            return self._fail(
                f"Satellite {candidate.sat_id} does not support single_pass_mosaic mode",
                window_s
            )

        # ------------------------------------------------------------------
        # 检查2: 窗口时长是否充足
        # ------------------------------------------------------------------
        sufficient, required_s = self._calculator.check_window_duration_sufficient(
            window_duration_s=window_s,
            num_strips=cfg.num_strips,
            strip_imaging_duration_s=cfg.strip_imaging_duration_s,
            inter_strip_slew_time_s=cfg.inter_strip_slew_time_s,
        )
        if not sufficient:
            return self._fail(
                f"Window {window_s:.1f}s < required {required_s:.1f}s for "
                f"{cfg.num_strips}-strip mosaic",
                window_s,
                required_duration_s=required_s,
            )

        # ------------------------------------------------------------------
        # 检查3: 条带间机动可行性
        # ------------------------------------------------------------------
        agility = sat.capabilities.agility
        settling_time = agility.get('settling_time', 5.0)
        max_roll_rate = agility.get('max_roll_rate', agility.get('max_slew_rate', 3.0))

        slew_violations = []

        # 条带间滚转角变化量（最大 = max_roll_step_deg）
        # 实际机动时间 = slew_time + settling_time ≤ inter_strip_slew_time_s
        available_slew_time = cfg.inter_strip_slew_time_s - settling_time
        if available_slew_time <= 0:
            slew_violations.append(
                f"inter_strip_slew_time_s ({cfg.inter_strip_slew_time_s:.1f}s) "
                f"<= settling_time ({settling_time:.1f}s), no time for actual slew"
            )
        else:
            # 最大可执行的单次滚转角变化量（度） = max_roll_rate × available_slew_time
            max_achievable_step = max_roll_rate * available_slew_time
            if cfg.max_roll_step_deg > max_achievable_step:
                slew_violations.append(
                    f"max_roll_step_deg {cfg.max_roll_step_deg:.1f}° cannot be achieved: "
                    f"max_roll_rate {max_roll_rate:.1f}°/s × "
                    f"available_time {available_slew_time:.1f}s = {max_achievable_step:.1f}°"
                )

        if slew_violations:
            self._stats['infeasible_count'] += 1
            return MultiStripConstraintResult(
                feasible=False,
                window_duration_s=window_s,
                required_duration_s=required_s,
                slew_violations=slew_violations,
                reason="; ".join(slew_violations),
            )

        # ------------------------------------------------------------------
        # 检查4: 调用条带规划器（同时验证滚转角约束）
        # ------------------------------------------------------------------
        orbit_altitude_m = self._get_orbit_altitude(sat, candidate.sat_position)
        nadir_lon, nadir_lat, nadir_degraded = self._get_nadir_position(candidate)

        strip_plan = self._calculator.plan_strips(
            num_strips=cfg.num_strips,
            strip_swath_width_m=cfg.strip_swath_width_m,
            overlap_ratio=cfg.overlap_ratio,
            inter_strip_slew_time_s=cfg.inter_strip_slew_time_s,
            strip_imaging_duration_s=cfg.strip_imaging_duration_s,
            max_roll_angle_deg=sat.capabilities.max_roll_angle,
            max_roll_step_deg=cfg.max_roll_step_deg,
            max_total_roll_span_deg=cfg.max_total_roll_span_deg,
            center_roll_deg=candidate.target.mosaic_center_roll_deg,
            orbit_altitude_m=orbit_altitude_m,
            window_duration_s=window_s,
            nadir_lon=nadir_lon,
            nadir_lat=nadir_lat,
            sat_velocity=candidate.sat_velocity,
        )

        if not strip_plan.feasible:
            self._stats['infeasible_count'] += 1
            return MultiStripConstraintResult(
                feasible=False,
                window_duration_s=window_s,
                required_duration_s=required_s,
                reason=strip_plan.infeasibility_reason,
            )

        # ------------------------------------------------------------------
        # 检查5: 能耗估算
        # ------------------------------------------------------------------
        estimated_energy_wh = self._estimate_energy(
            satellite=sat,
            cfg=cfg,
            strip_plan=strip_plan,
        )

        self._stats['feasible_count'] += 1
        return MultiStripConstraintResult(
            feasible=True,
            strip_plan=strip_plan,
            required_duration_s=strip_plan.total_duration_s,
            window_duration_s=window_s,
            estimated_energy_wh=estimated_energy_wh,
            degraded_geometry=nadir_degraded,
        )

    def check_feasibility_batch(
        self,
        candidates: List[MultiStripCandidate]
    ) -> List[MultiStripConstraintResult]:
        """
        批量检查多个候选

        注意：当前实现为顺序执行（逐个调用 check_feasibility），
        未做向量化优化。多条带拼幅约束检查涉及复杂的几何计算，
        实际批次通常较小（<50），顺序实现已满足性能需求。

        Args:
            candidates: 候选列表

        Returns:
            MultiStripConstraintResult 列表（与输入一一对应）
        """
        return [self.check_feasibility(c) for c in candidates]

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _fail(
        self,
        reason: str,
        window_s: float,
        required_duration_s: float = 0.0,
    ) -> MultiStripConstraintResult:
        """构造失败结果"""
        self._stats['infeasible_count'] += 1
        return MultiStripConstraintResult(
            feasible=False,
            window_duration_s=window_s,
            required_duration_s=required_duration_s,
            reason=reason,
        )

    def _get_orbit_altitude(
        self,
        satellite: Satellite,
        sat_position: Optional[Tuple[float, float, float]],
    ) -> float:
        """获取轨道高度（米），优先使用实时位置，否则从轨道根数计算"""
        if sat_position is not None:
            sat_r = (
                sat_position[0] ** 2
                + sat_position[1] ** 2
                + sat_position[2] ** 2
            ) ** 0.5
            return max(sat_r - EARTH_RADIUS_M, _MIN_ORBIT_ALTITUDE_M)

        # 从轨道根数中估算
        try:
            sma = satellite.orbit.semi_major_axis
            if sma is not None and sma > EARTH_RADIUS_M:
                return sma - EARTH_RADIUS_M
        except AttributeError:
            pass

        return _DEFAULT_ORBIT_ALTITUDE_M  # 默认500km

    def _get_nadir_position(
        self,
        candidate: MultiStripCandidate,
    ) -> Tuple[float, float, bool]:
        """获取星下点经纬度，使用目标中心作为近似。返回 (lon, lat, degraded)。"""
        target = candidate.target
        try:
            lon, lat = target.get_center()
            return float(lon), float(lat), False
        except Exception:
            logger.warning(
                "Failed to get center position for target %s (sat %s); "
                "falling back to (0.0, 0.0) — strip center coordinates will be inaccurate.",
                getattr(target, 'id', '?'),
                candidate.sat_id,
            )
            return 0.0, 0.0, True

    def _estimate_energy(
        self,
        satellite: Satellite,
        cfg: MultiStripMosaicConfig,
        strip_plan: StripPlan,
    ) -> float:
        """估算能耗（Wh）"""
        try:
            mode_cfg = satellite.payload_config.get_mode_config('single_pass_mosaic')
            base_power_w = mode_cfg.power_consumption_w
        except (AttributeError, ValueError):
            base_power_w = _DEFAULT_BASE_POWER_W  # 默认值

        total_imaging_s = cfg.num_strips * cfg.strip_imaging_duration_s
        total_slew_s = (cfg.num_strips - 1) * cfg.inter_strip_slew_time_s

        # base_power_w 在成像阶段和机动阶段均适用（已包含机动功耗开销）
        # total_imaging_s 和 total_slew_s 均使用相同功率，因此直接求和后乘以 base_power_w
        # base_power_w 已包含 power_overhead_factor（例如 172.5W = 150W × 1.15），
        # 不应再次乘以 power_overhead_factor，否则机动阶段功耗被重复计入。
        energy_wh = base_power_w * (total_imaging_s + total_slew_s) / 3600.0
        return energy_wh

    def get_stats(self):
        """返回统计信息"""
        return dict(self._stats)

    def reset_stats(self):
        """重置统计信息（在新的调度轮次开始前调用）"""
        self._stats = {
            'total_checks': 0,
            'feasible_count': 0,
            'infeasible_count': 0,
        }
