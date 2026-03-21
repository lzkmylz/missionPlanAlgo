"""
SAR条带模式物理引擎

支持三种建模方案（通过 SARStripmapConfig.beam_model 区分）：
  方案A (continuous)    — 连续角度范围
  方案B (discrete_beam) — 离散波位表，参数由配置直接给出
  方案C (derived_beam)  — 离散波位表，参数由物理方程从中心入射角推导

与聚束/滑动聚束的核心差异：
  - 波束固定指向（与卫星速度矢量夹角恒定）
  - 方位向分辨率固定为 L_a/2（与聚束模式相同）
  - 无"驻留时间"概念，成像时间 = 场景长度 / V_sat
  - 幅宽（Swath）由波束地面投影决定：W_g = (λ/L_w) × R / cos(θ)
  - PRF约束较宽松：仅需满足方位向多普勒带宽

公开接口：
  compute_imaging_params(altitude_m, look_angle_deg, scene_length_km, v_sat) -> SARStripmapResult
  compute_swath_width(altitude_m, look_angle_deg) -> float
  select_beam_position(target_incidence_angle_deg) -> Optional[StripmapBeamPosition]
"""

from __future__ import annotations

import math
from typing import Optional

from core.models.sar_stripmap_config import (
    StripmapBeamPosition,
    SARStripmapConfig,
    SARStripmapResult,
)

# 复用聚束模式的辅助函数
from core.dynamics.sar_spotlight_calculator import (
    _slant_range,
    _look_to_incidence,
)

# 默认卫星速度
_DEFAULT_V_SAT = 7500.0  # m/s，典型LEO卫星速度


class SARStripmapCalculator:
    """
    SAR条带模式物理参数计算器

    使用方法：
        cfg = SARStripmapConfig.from_dict(json_dict)
        calc = SARStripmapCalculator(cfg)
        result = calc.compute_imaging_params(
            altitude_m=631000,
            look_angle_deg=30.0,
            scene_length_km=50.0  # 用户指定的沿迹向场景长度
        )
    """

    def __init__(self, config: SARStripmapConfig):
        self.config = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_imaging_params(
        self,
        altitude_m: float,
        look_angle_deg: float,
        scene_length_km: float = 50.0,
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARStripmapResult:
        """
        计算给定几何条件下的条带成像参数。

        Args:
            altitude_m: 卫星轨道高度（m）
            look_angle_deg: 侧视角（度），即卫星天底方向与视线方向的夹角
            scene_length_km: 沿迹向场景长度（km），由用户需求决定
            v_sat: 卫星速度（m/s）

        Returns:
            SARStripmapResult，feasible=False 表示几何条件不可行
        """
        model = self.config.beam_model
        if model == "continuous":
            return self._compute_continuous(altitude_m, look_angle_deg, scene_length_km, v_sat)
        elif model == "discrete_beam":
            return self._compute_discrete(altitude_m, look_angle_deg, scene_length_km, v_sat)
        else:  # derived_beam
            return self._compute_derived(altitude_m, look_angle_deg, scene_length_km, v_sat)

    def compute_swath_width(
        self,
        altitude_m: float,
        look_angle_deg: float,
    ) -> float:
        """
        计算给定几何条件下的距离向幅宽（km）。

        公式：W_g = (λ / L_w) × R / cos(θ_inc)
        其中：
          - λ / L_w 为距离向波束宽度（弧度）
          - R 为斜距
          - θ_inc 为入射角

        Returns:
            幅宽（km），若几何不可行返回 0.0
        """
        cfg = self.config

        # 检查侧视角范围
        if not (cfg.min_look_angle_deg <= look_angle_deg <= cfg.max_look_angle_deg):
            return 0.0

        R = _slant_range(altitude_m, look_angle_deg)
        incidence = _look_to_incidence(altitude_m, look_angle_deg)

        # 距离向波束宽度（3dB）：φ_rg = λ / L_w
        beam_width_rg_rad = cfg.wavelength_m / cfg.antenna_width_m

        # 地面幅宽：W_g = φ_rg × R / cos(θ_inc)
        swath_m = beam_width_rg_rad * R / math.cos(math.radians(incidence))

        return swath_m / 1000.0  # 转换为 km

    def select_beam_position(
        self,
        target_incidence_angle_deg: float,
    ) -> Optional[StripmapBeamPosition]:
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
        self, altitude_m: float, look_angle_deg: float, scene_length_km: float, v_sat: float
    ) -> SARStripmapResult:
        """方案A：连续模型"""
        cfg = self.config

        # 校验侧视角范围
        if not (cfg.min_look_angle_deg <= look_angle_deg <= cfg.max_look_angle_deg):
            return SARStripmapResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=(
                    f"侧视角 {look_angle_deg:.1f}° 超出范围 "
                    f"[{cfg.min_look_angle_deg}, {cfg.max_look_angle_deg}]"
                ),
            )

        R = _slant_range(altitude_m, look_angle_deg)
        incidence = _look_to_incidence(altitude_m, look_angle_deg)

        # 计算幅宽
        swath_km = self._calc_swath_width(cfg, R, incidence)

        # 成像时间 = 场景长度 / 卫星速度
        scene_length_m = scene_length_km * 1000.0
        imaging_time_s = scene_length_m / v_sat

        # 条带模式方位向分辨率固定为 L_a/2
        rho_az = cfg.antenna_length_m / 2.0

        # 场景面积
        area = scene_length_km * swath_km

        return SARStripmapResult(
            feasible=True,
            imaging_time_s=imaging_time_s,
            scene_length_along_track_km=scene_length_km,
            swath_width_km=swath_km,
            scene_area_km2=area,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            limiting_constraint="",
            matched_beam_id=None,
            prf_hz_used=cfg.prf_hz,
            slant_range_m=R,
            nominal_integration_time_s=cfg.nominal_integration_time_s,
            peak_power_factor=1.0 / cfg.duty_cycle,
            beam_ground_speed_m_s=v_sat,
        )

    def _compute_discrete(
        self, altitude_m: float, look_angle_deg: float, scene_length_km: float, v_sat: float
    ) -> SARStripmapResult:
        """方案B：离散波位模型（参数由配置直接给出）"""
        cfg = self.config
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        beam = self.select_beam_position(incidence)

        if beam is None:
            return SARStripmapResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"无波位覆盖入射角 {incidence:.1f}°（侧视角 {look_angle_deg:.1f}°）",
            )

        R = _slant_range(altitude_m, look_angle_deg)

        # 成像时间 = 场景长度 / 卫星速度
        scene_length_m = scene_length_km * 1000.0
        imaging_time_s = scene_length_m / v_sat

        # 使用波位配置的参数
        swath_km = beam.swath_width_km or 0.0
        rho_az = beam.azimuth_resolution_m or (cfg.antenna_length_m / 2.0)
        rho_rg = beam.range_resolution_m or cfg.range_resolution_m
        prf = beam.prf_hz or 0.0
        int_time = beam.nominal_integration_time_s or cfg.nominal_integration_time_s

        # 场景面积
        area = scene_length_km * swath_km

        return SARStripmapResult(
            feasible=True,
            imaging_time_s=imaging_time_s,
            scene_length_along_track_km=scene_length_km,
            swath_width_km=swath_km,
            scene_area_km2=area,
            range_resolution_m=rho_rg,
            azimuth_resolution_m=rho_az,
            limiting_constraint="",
            matched_beam_id=beam.beam_id,
            prf_hz_used=prf,
            slant_range_m=R,
            nominal_integration_time_s=int_time,
            peak_power_factor=1.0 / cfg.duty_cycle,
            beam_ground_speed_m_s=v_sat,
        )

    def _compute_derived(
        self, altitude_m: float, look_angle_deg: float, scene_length_km: float, v_sat: float
    ) -> SARStripmapResult:
        """
        方案C：推导波位模型

        步骤：
        1. 将侧视角转换为入射角，选最近中心入射角的波位
        2. 由系统级固定参数推导 PRF、幅宽、分辨率
        """
        cfg = self.config
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        beam = self.select_beam_position(incidence)

        if beam is None:
            return SARStripmapResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"波位表为空，无法匹配入射角 {incidence:.1f}°",
            )

        R = _slant_range(altitude_m, look_angle_deg)

        # 成像时间 = 场景长度 / 卫星速度
        scene_length_m = scene_length_km * 1000.0
        imaging_time_s = scene_length_m / v_sat

        # --- 推导幅宽 ---
        swath_km = self._calc_swath_width(cfg, R, incidence)

        # --- 推导 PRF ---
        # 条带模式 PRF = safety_factor × V_sat / (2 × ρ_az)
        # 其中 ρ_az = L_a/2
        rho_az = cfg.antenna_length_m / 2.0
        PRF_used = cfg.prf_safety_factor * v_sat / (2.0 * rho_az)

        # --- 场景面积 ---
        area = scene_length_km * swath_km

        # 将推导结果写回 StripmapBeamPosition（方便上层查询，非持久化）
        beam.prf_hz = PRF_used
        beam.range_resolution_m = cfg.range_resolution_m
        beam.azimuth_resolution_m = rho_az
        beam.swath_width_km = swath_km
        beam.nominal_integration_time_s = cfg.nominal_integration_time_s

        # 推导波位入射角覆盖范围（±半波束宽度）
        half_beam_width_deg = math.degrees(cfg.wavelength_m / cfg.antenna_width_m / 2.0)
        if beam.incidence_angle_min_deg is None:
            beam.incidence_angle_min_deg = beam.center_incidence_angle_deg - half_beam_width_deg
        if beam.incidence_angle_max_deg is None:
            beam.incidence_angle_max_deg = beam.center_incidence_angle_deg + half_beam_width_deg

        return SARStripmapResult(
            feasible=True,
            imaging_time_s=imaging_time_s,
            scene_length_along_track_km=scene_length_km,
            swath_width_km=swath_km,
            scene_area_km2=area,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            limiting_constraint="",
            matched_beam_id=beam.beam_id,
            prf_hz_used=PRF_used,
            slant_range_m=R,
            nominal_integration_time_s=cfg.nominal_integration_time_s,
            peak_power_factor=1.0 / cfg.duty_cycle,
            beam_ground_speed_m_s=v_sat,
        )

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------

    def _calc_swath_width(
        self,
        cfg: SARStripmapConfig,
        slant_range_m: float,
        incidence_angle_deg: float,
    ) -> float:
        """
        计算距离向幅宽（km）。

        公式：W_g = (λ / L_w) × R / cos(θ_inc)
        """
        # 距离向波束宽度（3dB）：φ_rg = λ / L_w
        beam_width_rg_rad = cfg.wavelength_m / cfg.antenna_width_m

        # 地面幅宽：W_g = φ_rg × R / cos(θ_inc)
        swath_m = beam_width_rg_rad * slant_range_m / math.cos(math.radians(incidence_angle_deg))

        return swath_m / 1000.0  # 转换为 km
