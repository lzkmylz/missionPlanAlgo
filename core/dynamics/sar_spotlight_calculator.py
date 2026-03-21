"""
SAR聚束模式物理引擎

支持三种建模方案（通过 SARSpotlightConfig.beam_model 区分）：
  方案A (continuous)    — 连续角度范围 + 单一PRF
  方案B (discrete_beam) — 离散波位表，参数由配置直接给出
  方案C (derived_beam)  — 离散波位表，参数由物理方程从中心入射角推导

公开接口（三种方案共享，内部分路实现）：
  compute_dwell_time(altitude_m, look_angle_deg, v_sat)  -> SARSpotlightResult
  compute_scene_coverage(altitude_m, look_angle_deg, dwell_time_s, v_sat) -> SARSpotlightResult
  select_beam_position(target_incidence_angle_deg) -> Optional[BeamPosition]

物理公式参考：
  斜距:            R = H / cos(θ_look)
  PRF多普勒下限:   PRF_min = safety_factor * V² * L_az / (λ * R²)
  PRF距离上限:     PRF_max = c / (2 * R)
  驻留时间（电扫): T = min(2*φ_az*R/V, PRF*λ*R/(2*V²))
  方位向分辨率:    ρ_az = L_a / 2  (聚束模式)
  距离向分辨率:    ρ_rg = c / (2*B) ≈ range_resolution_m  (由带宽决定)
"""

from __future__ import annotations

import math
import logging
from typing import Optional

from core.models.sar_spotlight_config import (
    BeamPosition,
    SARSpotlightConfig,
    SARSpotlightResult,
    EARTH_RADIUS_M,
)

logger = logging.getLogger(__name__)

_DEFAULT_V_SAT = 7500.0  # m/s，典型LEO卫星速度


class SARSpotlightCalculator:
    """
    SAR聚束模式物理参数计算器

    使用方法：
        cfg = SARSpotlightConfig.from_dict(json_dict)
        calc = SARSpotlightCalculator(cfg)
        result = calc.compute_dwell_time(altitude_m=631000, look_angle_deg=35.0)
    """

    def __init__(self, config: SARSpotlightConfig):
        self.config = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_dwell_time(
        self,
        altitude_m: float,
        look_angle_deg: float,
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARSpotlightResult:
        """
        计算给定几何条件下的最大驻留时间及对应约束。

        Args:
            altitude_m: 卫星轨道高度（m）
            look_angle_deg: 侧视角（度），即卫星天底方向与视线方向的夹角
            v_sat: 卫星速度（m/s）

        Returns:
            SARSpotlightResult，feasible=False 表示几何条件不可行
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
    ) -> SARSpotlightResult:
        """
        给定驻留时间，计算场景覆盖范围。

        当 dwell_time_s <= 0 时，内部取 compute_dwell_time() 的结果作为驻留时间。

        Returns:
            SARSpotlightResult（包含 scene_size_az_km, scene_size_rg_km, scene_area_km2）
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
                return SARSpotlightResult(
                    feasible=False,
                    limiting_constraint="infeasible",
                    reason=f"无波位覆盖入射角 {incidence:.1f}°",
                )
            scene_az = beam.scene_size_az_km or (v_sat * dwell_time_s / 1000.0)
            scene_rg = beam.scene_size_rg_km or 0.0
            rho_rg = beam.range_resolution_m or cfg.range_resolution_m
            rho_az = beam.azimuth_resolution_m or (cfg.antenna_length_m / 2.0)
            prf = beam.prf_hz or 0.0
            beam_id = beam.beam_id
        else:
            scene_az = v_sat * dwell_time_s / 1000.0
            scene_rg = _scene_size_rg(cfg, R, look_angle_deg)
            rho_rg = cfg.range_resolution_m
            rho_az = cfg.antenna_length_m / 2.0
            prf = cfg.prf_hz if cfg.beam_model == "continuous" else 0.0
            beam_id = None

        area = scene_az * scene_rg

        return SARSpotlightResult(
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
        )

    def select_beam_position(
        self,
        target_incidence_angle_deg: float,
    ) -> Optional[BeamPosition]:
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
    ) -> SARSpotlightResult:
        """方案A：连续模型

        注意：星载SAR中PRF对驻留时间的约束来自方位向多普勒带宽，
        而非简单的距离无模糊公式 c/(2*PRF)。后者不适用于星载SAR。
        """
        cfg = self.config

        # 校验侧视角范围
        if not (cfg.min_look_angle_deg <= look_angle_deg <= cfg.max_look_angle_deg):
            return SARSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=(
                    f"侧视角 {look_angle_deg:.1f}° 超出范围 "
                    f"[{cfg.min_look_angle_deg}, {cfg.max_look_angle_deg}]"
                ),
            )

        R = _slant_range(altitude_m, look_angle_deg)

        # 方位向电子扫描驻留时间上限
        T_az = 2.0 * math.radians(cfg.max_azimuth_steering_deg) * R / v_sat

        # PRF多普勒模糊驻留时间上限
        # T_prf_max = PRF * λ * R / (2 * V²)
        # 推导：最大方位向场景 = PRF * λ * R / (2*V) 对应的驻留时间
        T_prf = cfg.prf_hz * cfg.wavelength_m * R / (2.0 * v_sat ** 2)

        T_dwell = min(T_az, T_prf)
        limiting = "azimuth_steering" if T_az <= T_prf else "prf_ambiguity"

        # 场景尺寸
        scene_az = v_sat * T_dwell / 1000.0
        scene_rg = _scene_size_rg(cfg, R, look_angle_deg)
        area = scene_az * scene_rg

        return SARSpotlightResult(
            feasible=True,
            dwell_time_s=T_dwell,
            scene_size_az_km=scene_az,
            scene_size_rg_km=scene_rg,
            scene_area_km2=area,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=cfg.antenna_length_m / 2.0,
            limiting_constraint=limiting,
            matched_beam_id=None,
            prf_hz_used=cfg.prf_hz,
            slant_range_m=R,
        )

    def _compute_discrete(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARSpotlightResult:
        """方案B：离散波位模型（参数由配置直接给出）"""
        cfg = self.config
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        beam = self.select_beam_position(incidence)

        if beam is None:
            return SARSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"无波位覆盖入射角 {incidence:.1f}°（侧视角 {look_angle_deg:.1f}°）",
            )

        R = _slant_range(altitude_m, look_angle_deg)

        # 场景方位向尺寸上限 × PRF多普勒模糊限制
        # 注：星载SAR中 PRF 限制的是方位向多普勒带宽，而非简单的距离无模糊
        scene_size_az_km = beam.scene_size_az_km if beam.scene_size_az_km is not None else 10.0
        T_beam_scene = scene_size_az_km * 1000.0 / v_sat
        T_prf = beam.prf_hz * cfg.wavelength_m * R / (2.0 * v_sat ** 2)
        T_dwell = min(T_beam_scene, T_prf)
        limiting = "beam_scene_size" if T_beam_scene <= T_prf else "prf_ambiguity"

        scene_az = beam.scene_size_az_km
        scene_rg = beam.scene_size_rg_km
        area = scene_az * scene_rg

        return SARSpotlightResult(
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
        )

    def _compute_derived(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARSpotlightResult:
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
            return SARSpotlightResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"波位表为空，无法匹配入射角 {incidence:.1f}°",
            )

        R = _slant_range(altitude_m, look_angle_deg)

        # --- 推导场景方位向尺寸（由电子扫描角决定）---
        L_az_m = 2.0 * math.radians(cfg.max_azimuth_steering_deg) * R  # 单位：m

        # --- 推导 PRF ---
        # 多普勒带宽下限（方位向过采样需求）
        # PRF_min_doppler = safety_factor * V * L_az / (λ * R)
        # 注：c/(2*R) 对星载SAR不适用（它约束的是脉冲波门大小，而非目标斜距）
        PRF_min_doppler = cfg.prf_safety_factor * v_sat * L_az_m / (cfg.wavelength_m * R)

        PRF_used = PRF_min_doppler  # 取多普勒下限作为工作PRF

        # --- 推导驻留时间 ---
        T_az = L_az_m / v_sat                                      # 电扫时间上限
        T_prf = PRF_used * cfg.wavelength_m * R / (2.0 * v_sat ** 2)  # PRF模糊限制
        T_dwell = min(T_az, T_prf)
        limiting = "azimuth_steering" if T_az <= T_prf else "prf_ambiguity"

        # --- 推导场景距离向尺寸 ---
        scene_rg = _scene_size_rg(cfg, R, look_angle_deg)

        # --- 推导分辨率 ---
        rho_az = cfg.antenna_length_m / 2.0  # 聚束模式方位向分辨率
        rho_rg = cfg.range_resolution_m

        scene_az = v_sat * T_dwell / 1000.0
        area = scene_az * scene_rg

        # 首次调用时为该波位推导入射角覆盖范围（仅在尚未设置时写入，幂等操作）
        if beam.incidence_angle_min_deg is None or beam.incidence_angle_max_deg is None:
            half_beam_width_deg = math.degrees(cfg.wavelength_m / cfg.antenna_width_m / 2.0)
            if beam.incidence_angle_min_deg is None:
                beam.incidence_angle_min_deg = beam.center_incidence_angle_deg - half_beam_width_deg
            if beam.incidence_angle_max_deg is None:
                beam.incidence_angle_max_deg = beam.center_incidence_angle_deg + half_beam_width_deg

        return SARSpotlightResult(
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
        )


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------

def _slant_range(altitude_m: float, look_angle_deg: float) -> float:
    """
    计算斜距（平面地球近似，look_angle < 60° 误差 < 2%）。

    R = H / cos(θ_look)
    """
    cos_look = math.cos(math.radians(look_angle_deg))
    if cos_look < 1e-6:
        raise ValueError(f"look_angle_deg={look_angle_deg} 接近90度，斜距无穷大")
    return altitude_m / cos_look


def _look_to_incidence(altitude_m: float, look_angle_deg: float) -> float:
    """
    将侧视角（look angle）转换为地面入射角（incidence angle）。

    球面地球公式：
        sin(θ_incidence) = (Re + H) / Re * sin(θ_look)

    对于 H < 700km，flat-Earth 近似误差 < 2°，但此处使用球面公式以提高精度。
    """
    Re = EARTH_RADIUS_M
    sin_incidence = (Re + altitude_m) / Re * math.sin(math.radians(look_angle_deg))
    sin_incidence = min(sin_incidence, 1.0)  # 防止浮点误差
    return math.degrees(math.asin(sin_incidence))


def _scene_size_rg(
    cfg: SARSpotlightConfig, slant_range_m: float, look_angle_deg: float
) -> float:
    """
    计算距离向场景尺寸（km），取天线波束足迹与电子扫描两者中的较小值。

    波束足迹：L_rg_beam = (λ/L_w) * R / cos(θ_look)
    电扫范围：L_rg_steer = 2 * φ_rg_rad * R / cos(θ_look)
    """
    cos_look = math.cos(math.radians(look_angle_deg))
    R = slant_range_m

    # 天线波束宽度限制
    theta_beam_rg = cfg.wavelength_m / cfg.antenna_width_m  # 弧度
    L_rg_beam = theta_beam_rg * R / cos_look

    # 电子扫描限制
    L_rg_steer = 2.0 * math.radians(cfg.max_range_steering_deg) * R / cos_look

    return min(L_rg_beam, L_rg_steer) / 1000.0  # 转换为 km
