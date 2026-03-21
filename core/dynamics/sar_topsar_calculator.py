"""
SAR TOPSAR模式物理引擎

支持三种建模方案（通过 SARTOPSARConfig.beam_model 区分）：
  方案A (continuous)    — 连续角度范围 + 统一突发参数
  方案B (discrete_beam) — 离散子条带表，参数由配置直接给出
  方案C (derived_beam)  — 离散子条带表，参数由物理方程从中心入射角推导

TOPSAR核心物理差异（与聚束/滑动聚束/条带）：
  方位向分辨率:  ρ_az = λ × R / (2 × V_sat × T_burst)   （由突发时长决定）
  循环时间:      T_cycle = N × T_burst + (N-1) × T_switch  （完整子条带循环）
  子条带幅宽:    W_g = (λ/L_w) × R / cos(θ_look)          （天线波束投影）
  PRF约束:       PRF ≥ 2 × V_sat / L_az  （方位向Doppler采样充分性；突发模式不用 c/2R 上限）

公开接口：
  compute_burst_params(altitude_m, look_angle_deg, v_sat)  -> SARTOPSARResult
  compute_scene_coverage(altitude_m, look_angle_deg, v_sat) -> SARTOPSARResult
  select_subswath_position(target_incidence_angle_deg) -> Optional[TOPSARSubSwathPosition]
"""

from __future__ import annotations

import copy
import math
import logging
from typing import Any, Dict, List, Optional

from core.models.sar_topsar_config import (
    TOPSARSubSwathPosition,
    SARTOPSARConfig,
    SARTOPSARResult,
    EARTH_RADIUS_M,
)

# 复用聚束模式的辅助函数
from core.dynamics.sar_spotlight_calculator import (
    _slant_range,
    _look_to_incidence,
)

logger = logging.getLogger(__name__)

_DEFAULT_V_SAT = 7500.0  # m/s，典型LEO卫星速度


class SARTOPSARCalculator:
    """
    SAR TOPSAR模式物理参数计算器

    使用方法：
        cfg = SARTOPSARConfig.from_dict(json_dict)
        calc = SARTOPSARCalculator(cfg)
        result = calc.compute_burst_params(altitude_m=631000, look_angle_deg=35.0)
    """

    def __init__(self, config: SARTOPSARConfig):
        # 深拷贝确保每个计算器实例拥有独立的 subswath_positions，
        # _derive_subswath_params 的原地修改不会影响共享的原始配置对象（线程安全）
        self.config = copy.deepcopy(config)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_burst_params(
        self,
        altitude_m: float,
        look_angle_deg: float,
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARTOPSARResult:
        """
        计算给定几何条件下的TOPSAR突发参数。

        Args:
            altitude_m: 卫星轨道高度（m）
            look_angle_deg: 中心子条带的侧视角（度）
            v_sat: 卫星速度（m/s）

        Returns:
            SARTOPSARResult，feasible=False 表示几何条件不可行
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
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARTOPSARResult:
        """
        计算TOPSAR场景覆盖范围（单次突发覆盖的方位向范围）。

        TOPSAR方位向场景长度 = V_sat × T_burst（单次突发内卫星飞行距离）

        Returns:
            SARTOPSARResult（包含 scene_size_az_km, total_swath_width_km, scene_area_km2）
        """
        return self.compute_burst_params(altitude_m, look_angle_deg, v_sat)

    def select_subswath_position(
        self,
        target_incidence_angle_deg: float,
    ) -> Optional[TOPSARSubSwathPosition]:
        """
        从子条带表中选择覆盖目标入射角的最优子条带（仅适用于 discrete_beam / derived_beam）。

        方案B：按 incidence_angle_min/max 范围匹配，中心角最近者优先。
        方案C：按与 center_incidence_angle_deg 的距离最近者选择。
        方案A：始终返回 None。
        """
        if self.config.beam_model == "continuous":
            return None

        if self.config.beam_model == "discrete_beam":
            candidates = [
                sp for sp in self.config.subswath_positions
                if sp.covers(target_incidence_angle_deg)
            ]
        else:  # derived_beam — 选最近中心角
            candidates = list(self.config.subswath_positions)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda sp: (
                abs(sp.center_incidence_angle_deg - target_incidence_angle_deg),
                sp.subswath_id,
            ),
        )

    # ------------------------------------------------------------------
    # 内部实现路径
    # ------------------------------------------------------------------

    def _compute_continuous(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARTOPSARResult:
        """方案A：连续模型，均匀分布多个子条带。"""
        cfg = self.config

        # 校验侧视角范围
        if not (cfg.min_look_angle_deg <= look_angle_deg <= cfg.max_look_angle_deg):
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=(
                    f"侧视角 {look_angle_deg:.1f}° 超出范围 "
                    f"[{cfg.min_look_angle_deg}, {cfg.max_look_angle_deg}]"
                ),
            )

        R = _slant_range(altitude_m, look_angle_deg)
        N = cfg.num_subswaths

        # 循环时间
        T_cycle = N * cfg.burst_duration_s + (N - 1) * cfg.burst_switch_time_s

        # HIGH-2: 检查 T_burst 是否超过物理上限 T_dwell_max = λ×R/(V_sat×L_a)
        T_dwell_max = cfg.wavelength_m * R / (v_sat * cfg.antenna_length_m)
        if cfg.burst_duration_s > T_dwell_max:
            logger.warning(
                "TOPSAR continuous: T_burst=%.3fs 超过物理上限 T_dwell_max=%.3fs "
                "(λ=%.4fm, R=%.0fm, V=%.0fm/s, L_a=%.1fm)。"
                "方位向分辨率将被天线物理下限 L_a/2=%.2fm 限制。",
                cfg.burst_duration_s, T_dwell_max,
                cfg.wavelength_m, R, v_sat, cfg.antenna_length_m,
                cfg.antenna_length_m / 2.0,
            )

        # 方位向分辨率（由突发时长决定，不超过天线物理下限）
        rho_az = _azimuth_resolution_from_burst(
            cfg.wavelength_m, R, v_sat, cfg.burst_duration_s, cfg.antenna_length_m
        )

        # 各子条带幅宽（天线波束投影）
        cos_look = math.cos(math.radians(look_angle_deg))
        swath_per_subswath_km = _subswath_width_km(
            cfg.wavelength_m, cfg.antenna_width_m, R, cos_look
        )
        total_swath_km = swath_per_subswath_km * N

        # 方位向场景长度（单次突发）
        scene_az_km = v_sat * cfg.burst_duration_s / 1000.0

        # 覆盖面积
        area_km2 = scene_az_km * total_swath_km

        # PRF约束检查
        prf_result = _check_prf_feasibility(cfg.prf_hz, v_sat, cfg.antenna_length_m, R)
        if not prf_result['feasible']:
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="prf_subswath_conflict",
                reason=prf_result['reason'],
            )

        # 子条带详细结果
        subswath_results = []
        for i in range(N):
            look_i = look_angle_deg + (i - N // 2) * cfg.subswath_spacing_deg
            R_i = _slant_range(altitude_m, look_i) if abs(look_i) < 85 else R
            subswath_results.append({
                "subswath_index": i,
                "look_angle_deg": look_i,
                "slant_range_m": R_i,
                "swath_width_km": swath_per_subswath_km,
                "burst_duration_s": cfg.burst_duration_s,
            })

        return SARTOPSARResult(
            feasible=True,
            num_subswaths_used=N,
            total_swath_width_km=total_swath_km,
            burst_duration_s=cfg.burst_duration_s,
            cycle_time_s=T_cycle,
            scene_size_az_km=scene_az_km,
            scene_area_km2=area_km2,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            limiting_constraint="burst_duration",
            matched_subswath_id=None,
            prf_hz_used=cfg.prf_hz,
            slant_range_m=R,
            peak_power_factor=1.0 / cfg.duty_cycle,
            subswath_results=subswath_results,
        )

    def _compute_discrete(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARTOPSARResult:
        """方案B：离散子条带模型（参数由配置直接给出）。"""
        cfg = self.config
        if look_angle_deg >= 80.0:
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"侧视角 {look_angle_deg:.1f}° 超出几何上限 80°",
            )
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        center_sw = self.select_subswath_position(incidence)

        if center_sw is None:
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"无子条带覆盖入射角 {incidence:.1f}°（侧视角 {look_angle_deg:.1f}°）",
            )

        R = _slant_range(altitude_m, look_angle_deg)
        N = cfg.num_subswaths

        # 使用中心子条带的突发时长（若未设置则用配置级默认值）
        burst_duration = center_sw.burst_duration_s if center_sw.burst_duration_s is not None else cfg.burst_duration_s

        # 循环时间
        T_cycle = N * burst_duration + (N - 1) * cfg.burst_switch_time_s

        # HIGH-2: 检查 T_burst 是否超过物理上限
        T_dwell_max = cfg.wavelength_m * R / (v_sat * cfg.antenna_length_m)
        if burst_duration > T_dwell_max:
            logger.warning(
                "TOPSAR discrete: T_burst=%.3fs 超过物理上限 T_dwell_max=%.3fs，"
                "方位向分辨率将被天线物理下限 L_a/2=%.2fm 限制。",
                burst_duration, T_dwell_max, cfg.antenna_length_m / 2.0,
            )

        # 方位向分辨率（方案B优先使用配置值，否则按公式推导并施加物理下限）
        rho_az = center_sw.azimuth_resolution_m if center_sw.azimuth_resolution_m is not None else \
            _azimuth_resolution_from_burst(cfg.wavelength_m, R, v_sat, burst_duration, cfg.antenna_length_m)

        # 各子条带幅宽（使用配置值或重算）
        total_swath_km = sum(
            (sp.swath_width_rg_km or 0.0) for sp in cfg.subswath_positions[:N]
        )

        # 方位向场景长度
        scene_az_km = v_sat * burst_duration / 1000.0
        area_km2 = scene_az_km * total_swath_km

        # PRF约束检查（使用中心子条带的PRF）
        prf_used = center_sw.prf_hz if center_sw.prf_hz is not None else cfg.prf_hz
        prf_result = _check_prf_feasibility(prf_used, v_sat, cfg.antenna_length_m, R)
        if not prf_result['feasible']:
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="prf_subswath_conflict",
                reason=prf_result['reason'],
            )

        # 各子条带详细结果
        subswath_results = self._build_subswath_results_discrete(
            cfg.subswath_positions[:N], altitude_m, v_sat
        )

        return SARTOPSARResult(
            feasible=True,
            num_subswaths_used=N,
            total_swath_width_km=total_swath_km,
            burst_duration_s=burst_duration,
            cycle_time_s=T_cycle,
            scene_size_az_km=scene_az_km,
            scene_area_km2=area_km2,
            range_resolution_m=center_sw.range_resolution_m or cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            limiting_constraint="burst_duration",
            matched_subswath_id=center_sw.subswath_id,
            prf_hz_used=prf_used,
            slant_range_m=R,
            peak_power_factor=1.0 / cfg.duty_cycle,
            subswath_results=subswath_results,
        )

    def _compute_derived(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARTOPSARResult:
        """
        方案C：推导子条带模型。

        步骤：
        1. 将侧视角转换为入射角，选最近中心入射角的子条带
        2. 由系统级固定参数推导 PRF、幅宽、分辨率
        3. 回填子条带的推导参数（幂等操作）
        """
        cfg = self.config
        if look_angle_deg >= 80.0:
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"侧视角 {look_angle_deg:.1f}° 超出几何上限 80°",
            )
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        center_sw = self.select_subswath_position(incidence)

        if center_sw is None:
            return SARTOPSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"子条带表为空，无法匹配入射角 {incidence:.1f}°",
            )

        R = _slant_range(altitude_m, look_angle_deg)
        N = cfg.num_subswaths

        # 使用配置级突发时长（方案C子条带不含个体突发时长）
        burst_duration = cfg.burst_duration_s

        # 循环时间
        T_cycle = N * burst_duration + (N - 1) * cfg.burst_switch_time_s

        # HIGH-2: 检查 T_burst 是否超过物理上限
        T_dwell_max = cfg.wavelength_m * R / (v_sat * cfg.antenna_length_m)
        if burst_duration > T_dwell_max:
            logger.warning(
                "TOPSAR derived: T_burst=%.3fs 超过物理上限 T_dwell_max=%.3fs，"
                "方位向分辨率将被天线物理下限 L_a/2=%.2fm 限制。",
                burst_duration, T_dwell_max, cfg.antenna_length_m / 2.0,
            )

        # 推导 PRF
        # 多普勒带宽下限：PRF_min = safety_factor × 2 × V_sat / L_az
        # 注：TOPSAR突发模式不使用 c/(2R) 上限（该约束对脉冲SAR不适用于总斜距）
        PRF_min = cfg.prf_safety_factor * 2.0 * v_sat / cfg.antenna_length_m
        PRF_used = PRF_min  # 取多普勒下限

        # 推导方位向分辨率（由突发时长决定，施加天线物理下限）
        rho_az = _azimuth_resolution_from_burst(cfg.wavelength_m, R, v_sat, burst_duration, cfg.antenna_length_m)

        # 推导各子条带幅宽
        cos_look = math.cos(math.radians(look_angle_deg))
        swath_per_sw_km = _subswath_width_km(
            cfg.wavelength_m, cfg.antenna_width_m, R, cos_look
        )
        total_swath_km = swath_per_sw_km * N

        # 方位向场景长度
        scene_az_km = v_sat * burst_duration / 1000.0
        area_km2 = scene_az_km * total_swath_km

        # 回填所有子条带的推导参数（幂等操作）
        for sp in cfg.subswath_positions[:N]:
            _derive_subswath_params(
                sp, cfg, R, v_sat, cos_look,
                swath_per_sw_km, rho_az, PRF_used, burst_duration
            )

        # 各子条带详细结果
        subswath_results = self._build_subswath_results_derived(
            cfg.subswath_positions[:N], altitude_m, v_sat,
            cfg, swath_per_sw_km, rho_az, PRF_used, burst_duration
        )

        return SARTOPSARResult(
            feasible=True,
            num_subswaths_used=N,
            total_swath_width_km=total_swath_km,
            burst_duration_s=burst_duration,
            cycle_time_s=T_cycle,
            scene_size_az_km=scene_az_km,
            scene_area_km2=area_km2,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            limiting_constraint="burst_duration",
            matched_subswath_id=center_sw.subswath_id,
            prf_hz_used=PRF_used,
            slant_range_m=R,
            peak_power_factor=1.0 / cfg.duty_cycle,
            subswath_results=subswath_results,
        )

    def _build_subswath_results_discrete(
        self,
        subswath_positions: List[TOPSARSubSwathPosition],
        altitude_m: float,
        v_sat: float,
    ) -> List[Dict[str, Any]]:
        """为方案B构建各子条带详细结果列表。"""
        results = []
        cfg = self.config
        for sp in subswath_positions:
            R_i = _slant_range(
                altitude_m,
                _incidence_to_look(altitude_m, sp.center_incidence_angle_deg)
            )
            burst_i = sp.burst_duration_s if sp.burst_duration_s is not None else cfg.burst_duration_s
            results.append({
                "subswath_id": sp.subswath_id,
                "center_incidence_angle_deg": sp.center_incidence_angle_deg,
                "slant_range_m": R_i,
                "swath_width_km": sp.swath_width_rg_km,
                "burst_duration_s": burst_i,
                "range_resolution_m": sp.range_resolution_m or cfg.range_resolution_m,
                "azimuth_resolution_m": sp.azimuth_resolution_m or (cfg.antenna_length_m / 2.0),
                "prf_hz": sp.prf_hz or cfg.prf_hz,
            })
        return results

    def _build_subswath_results_derived(
        self,
        subswath_positions: List[TOPSARSubSwathPosition],
        altitude_m: float,
        v_sat: float,
        cfg: SARTOPSARConfig,
        swath_km: float,
        rho_az: float,
        prf: float,
        burst_duration: float,
    ) -> List[Dict[str, Any]]:
        """为方案C构建各子条带详细结果列表（使用推导参数）。"""
        results = []
        for sp in subswath_positions:
            look_i = _incidence_to_look(altitude_m, sp.center_incidence_angle_deg)
            R_i = _slant_range(altitude_m, look_i)
            results.append({
                "subswath_id": sp.subswath_id,
                "center_incidence_angle_deg": sp.center_incidence_angle_deg,
                "slant_range_m": R_i,
                "swath_width_km": swath_km,
                "burst_duration_s": burst_duration,
                "range_resolution_m": cfg.range_resolution_m,
                "azimuth_resolution_m": rho_az,
                "prf_hz": prf,
            })
        return results


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------

def _azimuth_resolution_from_burst(
    wavelength_m: float,
    slant_range_m: float,
    v_sat: float,
    burst_duration_s: float,
    antenna_length_m: float = 0.0,
) -> float:
    """
    计算TOPSAR方位向分辨率（由突发时长决定）。

    公式: ρ_az = max(λ × R / (2 × V_sat × T_burst), L_a / 2)

    物理下限：天线物理长度决定的最小分辨率为 L_a/2，
    即使增大 T_burst 也无法突破此极限。
    """
    if burst_duration_s <= 0:
        raise ValueError(f"burst_duration_s must be > 0, got {burst_duration_s}")
    rho_az = wavelength_m * slant_range_m / (2.0 * v_sat * burst_duration_s)
    if antenna_length_m > 0:
        rho_az = max(rho_az, antenna_length_m / 2.0)
    return rho_az


def _subswath_width_km(
    wavelength_m: float,
    antenna_width_m: float,
    slant_range_m: float,
    cos_look: float,
) -> float:
    """
    计算单个子条带的距离向地面幅宽（km）。

    公式: W_g = (λ / L_w) × R / cos(θ_look)

    Args:
        wavelength_m: 雷达波长（m）
        antenna_width_m: 距离向天线宽度（m）
        slant_range_m: 斜距（m）
        cos_look: cos(侧视角)

    Returns:
        float: 地面幅宽（km）
    """
    if cos_look < 1e-6:
        cos_look = 1e-6
    theta_beam = wavelength_m / antenna_width_m  # 距离向波束宽度（弧度）
    return theta_beam * slant_range_m / cos_look / 1000.0


def _check_prf_feasibility(
    prf_hz: float,
    v_sat: float,
    antenna_length_m: float,
    slant_range_m: float,
) -> Dict[str, Any]:
    """
    检查PRF是否满足TOPSAR约束。

    TOPSAR突发模式中的PRF约束与条带模式不同：
    - 约束1（方位向欠采样下限）: PRF ≥ 2 × V_sat / L_az
    - 约束2（距离无模糊上限）: 取决于子条带斜距宽度，而非总斜距
      典型子条带斜距宽度约为 L_az × R / λ（由天线波束宽度决定），
      对应 PRF_max ≈ c / (2 × ΔR_subswath)。
      简化处理：TOPSAR突发模式不检查绝对斜距无模糊约束，
      仅验证方位向Doppler采样充分性。

    Returns:
        dict with 'feasible': bool, 'reason': str
    """
    prf_min = 2.0 * v_sat / antenna_length_m

    if prf_hz < prf_min:
        return {
            'feasible': False,
            'reason': (
                f"PRF={prf_hz:.0f}Hz 低于方位向采样下限 PRF_min={prf_min:.0f}Hz"
            ),
        }
    return {'feasible': True, 'reason': ''}


def _incidence_to_look(altitude_m: float, incidence_angle_deg: float) -> float:
    """
    将地面入射角转换为卫星侧视角（球面地球近似）。

    sin(θ_look) = Re / (Re + H) × sin(θ_incidence)
    """
    Re = EARTH_RADIUS_M
    sin_look = Re / (Re + altitude_m) * math.sin(math.radians(incidence_angle_deg))
    sin_look = min(sin_look, 1.0)
    return math.degrees(math.asin(sin_look))


def _derive_subswath_params(
    subswath: TOPSARSubSwathPosition,
    cfg: SARTOPSARConfig,
    slant_range_m: float,
    v_sat: float,
    cos_look: float,
    swath_width_km: float,
    rho_az: float,
    prf_hz: float,
    burst_duration_s: float,
) -> None:
    """
    为方案C的子条带推导并回填缺失的物理参数（幂等操作）。

    Args:
        subswath: 需要填充参数的子条带对象
        cfg: TOPSAR配置
        slant_range_m: 斜距（m）
        v_sat: 卫星速度（m/s）
        cos_look: cos(侧视角)
        swath_width_km: 推导的子条带幅宽（km）
        rho_az: 方位向分辨率（m）
        prf_hz: 推导的PRF（Hz）
        burst_duration_s: 突发时长（s）
    """
    half_beam_width_deg = math.degrees(cfg.wavelength_m / cfg.antenna_width_m / 2.0)

    if subswath.incidence_angle_min_deg is None:
        subswath.incidence_angle_min_deg = (
            subswath.center_incidence_angle_deg - half_beam_width_deg
        )
    if subswath.incidence_angle_max_deg is None:
        subswath.incidence_angle_max_deg = (
            subswath.center_incidence_angle_deg + half_beam_width_deg
        )
    if subswath.prf_hz is None:
        subswath.prf_hz = prf_hz
    if subswath.range_resolution_m is None:
        subswath.range_resolution_m = cfg.range_resolution_m
    if subswath.azimuth_resolution_m is None:
        subswath.azimuth_resolution_m = rho_az
    if subswath.swath_width_rg_km is None:
        subswath.swath_width_rg_km = swath_width_km
    if subswath.burst_duration_s is None:
        subswath.burst_duration_s = burst_duration_s
