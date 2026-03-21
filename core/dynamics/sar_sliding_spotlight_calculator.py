"""
SAR滑动聚束模式物理引擎

支持三种建模方案（通过 SARSlidingSpotlightConfig.beam_model 区分）：
  方案A (continuous)    — 连续角度范围 + VRC距离
  方案B (discrete_beam) — 离散波位表，参数由配置直接给出（含VRC）
  方案C (derived_beam)  — 离散波位表，参数由物理方程从中心入射角推导

与聚束模式的核心物理差异：
  有效场景速度: V_eff = V_sat × (1 - R/D_vrc)
  方位向驻留时间上限: T_az = 2 × θ_az × D_vrc / V_sat（分母变为 D_vrc）
  方位向分辨率: ρ_az = λ × D_vrc / (2 × V_sat × T_dwell)
  PRF模糊驻留时间: T_prf = PRF × λ × R / (2 × V_sat × V_eff)

公开接口：
  compute_dwell_time(altitude_m, look_angle_deg, v_sat)  -> SARSlidingSpotlightResult
  compute_scene_coverage(altitude_m, look_angle_deg, dwell_time_s, v_sat) -> SARSlidingSpotlightResult
  select_beam_position(target_incidence_angle_deg) -> Optional[SlidingBeamPosition]
"""

from __future__ import annotations

import math
from typing import Optional

from core.models.sar_sliding_spotlight_config import (
    SlidingBeamPosition,
    SARSlidingSpotlightConfig,
    SARSlidingSpotlightResult,
)

# 复用聚束模式的辅助函数
from core.dynamics.sar_spotlight_calculator import (
    _slant_range,
    _look_to_incidence,
    _scene_size_rg,
)


# 默认卫星速度
_DEFAULT_V_SAT = 7500.0  # m/s，典型LEO卫星速度


class SARSlidingSpotlightCalculator:
    """
    SAR滑动聚束模式物理参数计算器

    使用方法：
        cfg = SARSlidingSpotlightConfig.from_dict(json_dict)
        calc = SARSlidingSpotlightCalculator(cfg)
        result = calc.compute_dwell_time(altitude_m=631000, look_angle_deg=35.0)
    """

    def __init__(self, config: SARSlidingSpotlightConfig):
        self.config = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_dwell_time(
        self,
        altitude_m: float,
        look_angle_deg: float,
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARSlidingSpotlightResult:
        """
        计算给定几何条件下的最大驻留时间及对应约束。

        Args:
            altitude_m: 卫星轨道高度（m）
            look_angle_deg: 侧视角（度），即卫星天底方向与视线方向的夹角
            v_sat: 卫星速度（m/s）

        Returns:
            SARSlidingSpotlightResult，feasible=False 表示几何条件不可行
        """
        model = self.config.beam_model
        if model == "continuous":
            return self._compute_continuous(altitude_m, look_angle_deg, v_sat)
        elif model == "discrete_beam":
            return self._compute_discrete(altitude_m, look_angle_deg, v_sat)
        else:  # derived_beam
            return self._compute_derived(altitude_m, look_angle_deg, v_sat)

    def compute_scene_coverage(
        self,
        altitude_m: float,
        look_angle_deg: float,
        dwell_time_s: float,
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARSlidingSpotlightResult:
        """
        给定驻留时间，计算场景覆盖范围。

        当 dwell_time_s <= 0 时，内部取 compute_dwell_time() 的结果作为驻留时间。

        Returns:
            SARSlidingSpotlightResult（包含 scene_size_az_km, scene_size_rg_km, scene_area_km2）
        """
        if dwell_time_s <= 0:
            result = self.compute_dwell_time(altitude_m, look_angle_deg, v_sat)
            if not result.feasible:
                return result
            dwell_time_s = result.dwell_time_s

        cfg = self.config
        R = _slant_range(altitude_m, look_angle_deg)

        if cfg.beam_model == "discrete_beam":
            incidence = _look_to_incidence(altitude_m, look_angle_deg)
            beam = self.select_beam_position(incidence)
            if beam is None:
                return SARSlidingSpotlightResult(
                    feasible=False,
                    limiting_constraint="infeasible",
                    reason=f"无波位覆盖入射角 {incidence:.1f}°",
                )
            scene_az = beam.scene_size_az_km or 0.0
            scene_rg = beam.scene_size_rg_km or 0.0
            rho_rg = beam.range_resolution_m or cfg.range_resolution_m
            rho_az = beam.azimuth_resolution_m or _derive_azimuth_resolution(cfg, v_sat, dwell_time_s)
            prf = beam.prf_hz or 0.0
            beam_id = beam.beam_id
            vrc_distance = beam.vrc_distance_m or cfg.vrc_distance_m
        else:
            # 方案A/C: 从顶层配置使用VRC距离
            vrc_distance = cfg.vrc_distance_m
            v_eff = _effective_velocity(v_sat, R, vrc_distance)
            scene_az = v_eff * dwell_time_s / 1000.0  # km
            scene_rg = _scene_size_rg(cfg, R, look_angle_deg)
            rho_rg = cfg.range_resolution_m
            rho_az = _derive_azimuth_resolution(cfg, v_sat, dwell_time_s)
            prf = cfg.prf_hz if cfg.beam_model == "continuous" else _derive_prf_min(cfg, v_sat)
            beam_id = None

        area = scene_az * scene_rg
        v_eff = _effective_velocity(v_sat, R, vrc_distance)

        return SARSlidingSpotlightResult(
            feasible=True,
            dwell_time_s=dwell_time_s,
            scene_size_az_km=scene_az,
            scene_size_rg_km=scene_rg,
            scene_area_km2=area,
            range_resolution_m=rho_rg,
            azimuth_resolution_m=rho_az,
            limiting_constraint="given_dwell",
            matched_beam_id=beam_id,
            prf_hz_used=prf,
            slant_range_m=R,
            vrc_distance_m=vrc_distance,
            effective_scene_velocity_m_s=v_eff,
            vrc_ratio=R / vrc_distance,
            peak_power_factor=1.0 / cfg.duty_cycle,
        )

    def select_beam_position(
        self,
        target_incidence_angle_deg: float,
    ) -> Optional[SlidingBeamPosition]:
        """
        从波位表中选择覆盖目标入射角的最优波位（仅适用于 discrete_beam / derived_beam）。

        方案B：按 incidence_angle_min/max 范围匹配，中心角最近者优先。
        方案C：按与 center_incidence_angle_deg 的距离最近者选择。
        方案A：始终返回 None。
        """
        if self.config.beam_model == "continuous":
            return None

        if self.config.beam_model == "discrete_beam":
            candidates = [
                bp for bp in self.config.beam_positions
                if bp.covers(target_incidence_angle_deg)
            ]
        else:  # derived_beam — 选最近中心角
            candidates = self.config.beam_positions

        if not candidates:
            return None

        # 主排序：与目标入射角的中心距离；次排序：beam_id 字母序（确保决定性）
        return min(
            candidates,
            key=lambda bp: (
                abs(bp.center_incidence_angle_deg - target_incidence_angle_deg),
                bp.beam_id,
            ),
        )

    # ------------------------------------------------------------------
    # 内部实现路径
    # ------------------------------------------------------------------

    def _compute_continuous(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARSlidingSpotlightResult:
        """方案A：连续模型

        与聚束模式的区别：
        - T_az = 2 × θ_az × D_vrc / V_sat（分母变为 D_vrc）
        - V_eff = V_sat × (1 - R/D_vrc)
        - ρ_az = λ × D_vrc / (2 × V_sat × T_dwell)
        """
        cfg = self.config

        # 校验侧视角范围
        if not (cfg.min_look_angle_deg <= look_angle_deg <= cfg.max_look_angle_deg):
            return SARSlidingSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=(
                    f"侧视角 {look_angle_deg:.1f}° 超出范围 "
                    f"[{cfg.min_look_angle_deg}, {cfg.max_look_angle_deg}]"
                ),
            )

        R = _slant_range(altitude_m, look_angle_deg)
        vrc_distance = cfg.vrc_distance_m

        # 检查 V_eff 可行性（D_vrc 不能过于接近 R）
        v_eff_check = _check_veff(v_sat, R, vrc_distance)
        if not v_eff_check.feasible:
            return SARSlidingSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=v_eff_check.reason,
            )

        v_eff = v_eff_check.v_eff

        # 方位向电子扫描驻留时间上限（滑动聚束：分母为 D_vrc）
        # T_az = 2 * θ_az * D_vrc / V_sat
        T_az = 2.0 * math.radians(cfg.max_azimuth_steering_deg) * vrc_distance / v_sat

        # PRF多普勒模糊驻留时间上限（滑动聚束：V_sat * V_eff）
        # T_prf = PRF × λ × R / (2 × V_sat × V_eff)
        T_prf = cfg.prf_hz * cfg.wavelength_m * R / (2.0 * v_sat * v_eff)

        T_dwell = min(T_az, T_prf)
        limiting = "azimuth_steering" if T_az <= T_prf else "prf_ambiguity"

        # 场景尺寸
        scene_az = v_eff * T_dwell / 1000.0  # km
        scene_rg = _scene_size_rg(cfg, R, look_angle_deg)
        area = scene_az * scene_rg

        # 滑动聚束方位向分辨率: ρ_az = λ × D_vrc / (2 × V_sat × T_dwell)
        rho_az = cfg.wavelength_m * vrc_distance / (2.0 * v_sat * T_dwell)

        return SARSlidingSpotlightResult(
            feasible=True,
            dwell_time_s=T_dwell,
            scene_size_az_km=scene_az,
            scene_size_rg_km=scene_rg,
            scene_area_km2=area,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            limiting_constraint=limiting,
            matched_beam_id=None,
            prf_hz_used=cfg.prf_hz,
            slant_range_m=R,
            vrc_distance_m=vrc_distance,
            effective_scene_velocity_m_s=v_eff,
            vrc_ratio=R / vrc_distance,
            peak_power_factor=1.0 / cfg.duty_cycle,
        )

    def _compute_discrete(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARSlidingSpotlightResult:
        """方案B：离散波位模型（参数由配置直接给出）"""
        cfg = self.config
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        beam = self.select_beam_position(incidence)

        if beam is None:
            return SARSlidingSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"无波位覆盖入射角 {incidence:.1f}°（侧视角 {look_angle_deg:.1f}°）",
            )

        R = _slant_range(altitude_m, look_angle_deg)
        vrc_distance = beam.vrc_distance_m

        # 检查 V_eff 可行性
        v_eff_check = _check_veff(v_sat, R, vrc_distance)
        if not v_eff_check.feasible:
            return SARSlidingSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=v_eff_check.reason,
            )

        v_eff = v_eff_check.v_eff

        # 场景方位向尺寸上限 × PRF多普勒模糊限制
        T_beam_scene = beam.scene_size_az_km * 1000.0 / v_eff
        # 滑动聚束 PRF模糊: T_prf = PRF × λ × R / (2 × V_sat × V_eff)
        T_prf = beam.prf_hz * cfg.wavelength_m * R / (2.0 * v_sat * v_eff)
        T_dwell = min(T_beam_scene, T_prf)
        limiting = "beam_scene_size" if T_beam_scene <= T_prf else "prf_ambiguity"

        scene_az = beam.scene_size_az_km
        scene_rg = beam.scene_size_rg_km
        area = scene_az * scene_rg

        return SARSlidingSpotlightResult(
            feasible=True,
            dwell_time_s=T_dwell,
            scene_size_az_km=scene_az,
            scene_size_rg_km=scene_rg,
            scene_area_km2=area,
            range_resolution_m=beam.range_resolution_m,
            azimuth_resolution_m=beam.azimuth_resolution_m,
            limiting_constraint=limiting,
            matched_beam_id=beam.beam_id,
            prf_hz_used=beam.prf_hz,
            slant_range_m=R,
            vrc_distance_m=vrc_distance,
            effective_scene_velocity_m_s=v_eff,
            vrc_ratio=R / vrc_distance,
            peak_power_factor=1.0 / cfg.duty_cycle,
        )

    def _compute_derived(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARSlidingSpotlightResult:
        """
        方案C：推导波位模型

        步骤：
        1. 将侧视角转换为入射角，选最近中心入射角的波位
        2. 由系统级固定参数推导 PRF、场景尺寸、分辨率
        """
        cfg = self.config
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        beam = self.select_beam_position(incidence)

        if beam is None:
            return SARSlidingSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"波位表为空，无法匹配入射角 {incidence:.1f}°",
            )

        R = _slant_range(altitude_m, look_angle_deg)
        vrc_distance = cfg.vrc_distance_m

        # 检查 V_eff 可行性
        v_eff_check = _check_veff(v_sat, R, vrc_distance)
        if not v_eff_check.feasible:
            return SARSlidingSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=v_eff_check.reason,
            )

        v_eff = v_eff_check.v_eff

        # --- 推导场景方位向尺寸（由电子扫描角决定）---
        # 滑动聚束: L_az = 2 × θ_az × D_vrc
        L_az_m = 2.0 * math.radians(cfg.max_azimuth_steering_deg) * vrc_distance

        # --- 推导 PRF ---
        # 多普勒带宽下限（方位向过采样需求）
        # 滑动聚束: PRF_min = safety_factor × 2 × V_sat × θ_az_rad / λ
        theta_az_rad = math.radians(cfg.max_azimuth_steering_deg)
        PRF_min_doppler = cfg.prf_safety_factor * 2.0 * v_sat * theta_az_rad / cfg.wavelength_m

        PRF_used = PRF_min_doppler

        # --- 推导驻留时间 ---
        T_az = L_az_m / v_eff  # 电扫时间上限（滑动聚束：分母为 V_eff）
        T_prf = PRF_used * cfg.wavelength_m * R / (2.0 * v_sat * v_eff)  # PRF模糊限制
        T_dwell = min(T_az, T_prf)
        limiting = "azimuth_steering" if T_az <= T_prf else "prf_ambiguity"

        # --- 推导场景距离向尺寸 ---
        scene_rg = _scene_size_rg(cfg, R, look_angle_deg)

        # --- 推导分辨率 ---
        # 滑动聚束方位向分辨率: ρ_az = λ × D_vrc / (2 × V_sat × T_dwell)
        rho_az = cfg.wavelength_m * vrc_distance / (2.0 * v_sat * T_dwell)
        rho_rg = cfg.range_resolution_m

        scene_az = v_eff * T_dwell / 1000.0
        area = scene_az * scene_rg

        # 将推导结果写回 SlidingBeamPosition（方便上层查询，非持久化）
        beam.prf_hz = PRF_used
        beam.range_resolution_m = rho_rg
        beam.azimuth_resolution_m = rho_az
        beam.scene_size_az_km = scene_az
        beam.scene_size_rg_km = scene_rg
        beam.vrc_distance_m = vrc_distance

        # 推导波位入射角覆盖范围（±半波束宽度）
        half_beam_width_deg = math.degrees(cfg.wavelength_m / cfg.antenna_width_m / 2.0)
        if beam.incidence_angle_min_deg is None:
            beam.incidence_angle_min_deg = beam.center_incidence_angle_deg - half_beam_width_deg
        if beam.incidence_angle_max_deg is None:
            beam.incidence_angle_max_deg = beam.center_incidence_angle_deg + half_beam_width_deg

        return SARSlidingSpotlightResult(
            feasible=True,
            dwell_time_s=T_dwell,
            scene_size_az_km=scene_az,
            scene_size_rg_km=scene_rg,
            scene_area_km2=area,
            range_resolution_m=rho_rg,
            azimuth_resolution_m=rho_az,
            limiting_constraint=limiting,
            matched_beam_id=beam.beam_id,
            prf_hz_used=PRF_used,
            slant_range_m=R,
            vrc_distance_m=vrc_distance,
            effective_scene_velocity_m_s=v_eff,
            vrc_ratio=R / vrc_distance,
            peak_power_factor=1.0 / cfg.duty_cycle,
        )


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------

class _VeffCheckResult:
    """V_eff 可行性检查结果"""
    def __init__(self, feasible: bool, v_eff: float = 0.0, reason: str = ""):
        self.feasible = feasible
        self.v_eff = v_eff
        self.reason = reason


def _check_veff(v_sat: float, slant_range_m: float, vrc_distance_m: float) -> _VeffCheckResult:
    """
    检查有效场景速度是否可行。

    滑动聚束要求: D_vrc > R（否则 V_eff <= 0，退化为聚束模式或不可行）
    推荐: D_vrc > 1.05 * R（V_eff < 5% V_sat 视为不现实的滑动聚束）
    """
    if vrc_distance_m <= slant_range_m:
        return _VeffCheckResult(
            feasible=False,
            reason=f"VRC距离 {vrc_distance_m/1000:.1f}km 必须大于斜距 {slant_range_m/1000:.1f}km"
        )

    v_eff_ratio = 1.0 - slant_range_m / vrc_distance_m
    V_EFF_MIN_RATIO = 0.05  # V_eff 至少为 V_sat 的 5%

    if v_eff_ratio < V_EFF_MIN_RATIO:
        return _VeffCheckResult(
            feasible=False,
            reason=(
                f"VRC距离 {vrc_distance_m/1000:.1f}km 过于接近斜距 {slant_range_m/1000:.1f}km "
                f"(V_eff/V_sat = {v_eff_ratio:.2%} < {V_EFF_MIN_RATIO:.0%})，"
                f"退化为聚束模式。建议 D_vrc >= {slant_range_m * 1.05 / 1000:.1f}km"
            )
        )

    return _VeffCheckResult(feasible=True, v_eff=v_sat * v_eff_ratio)


def _effective_velocity(v_sat: float, slant_range_m: float, vrc_distance_m: float) -> float:
    """计算有效场景速度 V_eff = V_sat × (1 - R/D_vrc)"""
    return v_sat * (1.0 - slant_range_m / vrc_distance_m)


def _derive_azimuth_resolution(
    cfg: SARSlidingSpotlightConfig,
    v_sat: float,
    dwell_time_s: float
) -> float:
    """
    滑动聚束方位向分辨率推导公式:
    ρ_az = λ × D_vrc / (2 × V_sat × T_dwell)
    """
    return cfg.wavelength_m * cfg.vrc_distance_m / (2.0 * v_sat * dwell_time_s)


def _derive_prf_min(cfg: SARSlidingSpotlightConfig, v_sat: float) -> float:
    """
    推导 PRF 最小值:
    PRF_min = safety_factor × 2 × V_sat × θ_az_rad / λ
    """
    theta_az_rad = math.radians(cfg.max_azimuth_steering_deg)
    return cfg.prf_safety_factor * 2.0 * v_sat * theta_az_rad / cfg.wavelength_m
