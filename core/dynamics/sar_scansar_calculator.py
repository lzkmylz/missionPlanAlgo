"""
SAR ScanSAR模式物理引擎

支持三种建模方案（通过 SARScanSARConfig.beam_model 区分）：
  方案A (continuous)    — 连续角度范围 + 统一Burst参数
  方案B (discrete_beam) — 离散子条带表，参数由配置直接给出
  方案C (derived_beam)  — 离散子条带表，参数由物理方程从中心入射角推导

ScanSAR核心物理特性（区别于TOPSAR）：
  1. 无方位向波束调制（TOPSAR通过进行性扫描补偿扇贝效应；ScanSAR无此补偿）
  2. 扇贝效应（Scalloping）：目标信号幅度随方位位置周期性变化
     - 物理原因：在一次burst内，目标仅被天线方位方向图的局部照射
     - 幅度变化规律：gain(τ) = sinc(τ / T_burst)，τ ∈ [-T_burst/2, +T_burst/2]
     - 峰值退化（rectangular窗）：20×log10(π/2) ≈ 3.92 dB（位于burst边界）
  3. SNR非均匀性：等于扇贝效应量（burst边缘比中心低 peak_scalloping_db）
  4. ISLR退化：由有限burst截断引起的旁瓣电平升高

方位向分辨率（与TOPSAR相同公式，但T_burst通常更短）：
  ρ_az = max(λ × R / (2 × V_sat × T_burst), L_a / 2)

扫描循环时间：
  T_cycle = N × T_burst + (N-1) × T_switch

公开接口：
  compute_burst_params(altitude_m, look_angle_deg, v_sat)  -> SARScanSARResult
  select_subswath_position(target_incidence_angle_deg) -> Optional[ScanSARSubSwathPosition]
"""

from __future__ import annotations

import copy
import math
import logging
from typing import Any, Dict, List, Optional

from core.constants import SAR_DEFAULT_V_SAT_MS
from core.models.sar_scansar_config import (
    ScanSARSubSwathPosition,
    SARScanSARConfig,
    SARScanSARResult,
)

# 复用聚束模式的辅助函数
from core.dynamics.sar_spotlight_calculator import (
    _slant_range,
    _look_to_incidence,
)

# 复用TOPSAR的通用burst辅助函数
from core.dynamics.sar_topsar_calculator import (
    _azimuth_resolution_from_burst,
    _subswath_width_km,
    _check_prf_feasibility,
    _incidence_to_look,
)

logger = logging.getLogger(__name__)

_DEFAULT_V_SAT = SAR_DEFAULT_V_SAT_MS  # 从 core.constants 引入，避免多处重复定义


class SARScanSARCalculator:
    """
    SAR ScanSAR模式物理参数计算器

    与 SARTOPSARCalculator 的关键差异：
      1. 所有计算路径都调用 _compute_scalloping() 获取扇贝效应指标
      2. 结果中包含 peak_scalloping_db、mean_scalloping_db、snr_variation_db 等ScanSAR特有字段
      3. azimuth_resolution_degradation 表征相对条带模式的分辨率退化程度

    使用方法：
        cfg = SARScanSARConfig.from_dict(json_dict)
        calc = SARScanSARCalculator(cfg)
        result = calc.compute_burst_params(altitude_m=631000, look_angle_deg=35.0)
        print(f"峰值扇贝效应: {result.peak_scalloping_db:.2f} dB")
        print(f"方位分辨率: {result.azimuth_resolution_m:.1f} m")
        print(f"总幅宽: {result.total_swath_width_km:.0f} km")
    """

    def __init__(self, config: SARScanSARConfig):
        # 深拷贝确保线程安全：_derive_subswath_params 的原地修改不影响原始配置
        self.config = copy.deepcopy(config)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_burst_params(
        self,
        altitude_m: float,
        look_angle_deg: float,
        v_sat: float = _DEFAULT_V_SAT,
    ) -> SARScanSARResult:
        """
        计算给定几何条件下的ScanSAR Burst参数及扇贝效应指标。

        Args:
            altitude_m: 卫星轨道高度（m）
            look_angle_deg: 中心子条带的侧视角（度）
            v_sat: 卫星速度（m/s）

        Returns:
            SARScanSARResult，feasible=False 表示几何条件不可行
        """
        model = self.config.beam_model
        if model == "continuous":
            return self._compute_continuous(altitude_m, look_angle_deg, v_sat)
        elif model == "discrete_beam":
            return self._compute_discrete(altitude_m, look_angle_deg, v_sat)
        else:  # derived_beam
            return self._compute_derived(altitude_m, look_angle_deg, v_sat)

    def select_subswath_position(
        self,
        target_incidence_angle_deg: float,
    ) -> Optional[ScanSARSubSwathPosition]:
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
    ) -> SARScanSARResult:
        """方案A：连续模型，均匀分布多个子条带。"""
        cfg = self.config

        # 校验侧视角范围
        if not (cfg.min_look_angle_deg <= look_angle_deg <= cfg.max_look_angle_deg):
            return SARScanSARResult(
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

        # 检查 T_burst 是否超过物理上限 T_dwell_max = λ×R/(V_sat×L_a)
        T_dwell_max = cfg.wavelength_m * R / (v_sat * cfg.antenna_length_m)
        if cfg.burst_duration_s > T_dwell_max:
            logger.warning(
                "ScanSAR continuous: T_burst=%.3fs 超过物理上限 T_dwell_max=%.3fs "
                "(λ=%.4fm, R=%.0fm, V=%.0fm/s, L_a=%.1fm)。"
                "方位向分辨率将被天线物理下限 L_a/2=%.2fm 限制。",
                cfg.burst_duration_s, T_dwell_max,
                cfg.wavelength_m, R, v_sat, cfg.antenna_length_m,
                cfg.antenna_length_m / 2.0,
            )

        # 方位向分辨率（由突发时长决定）
        rho_az = _azimuth_resolution_from_burst(
            cfg.wavelength_m, R, v_sat, cfg.burst_duration_s, cfg.antenna_length_m
        )

        # 方位分辨率退化倍数（相对条带模式 L_a/2）
        az_degradation = rho_az / (cfg.antenna_length_m / 2.0)

        # 各子条带幅宽
        cos_look = math.cos(math.radians(look_angle_deg))
        swath_per_subswath_km = _subswath_width_km(
            cfg.wavelength_m, cfg.antenna_width_m, R, cos_look
        )
        total_swath_km = swath_per_subswath_km * N

        # 方位向场景长度（单次burst）
        scene_az_km = v_sat * cfg.burst_duration_s / 1000.0
        area_km2 = scene_az_km * total_swath_km

        # PRF约束检查
        prf_result = _check_prf_feasibility(cfg.prf_hz, v_sat, cfg.antenna_length_m, R)
        if not prf_result['feasible']:
            return SARScanSARResult(
                feasible=False,
                limiting_constraint="prf_subswath_conflict",
                reason=prf_result['reason'],
            )

        # 扇贝效应计算（ScanSAR特有）
        peak_scalloping, mean_scalloping, snr_var = _compute_scalloping(
            cfg.scalloping_window
        ) if cfg.enable_scalloping_model else (0.0, 0.0, 0.0)

        # ISLR退化量
        islr_deg = _compute_islr_degradation(cfg.scalloping_window)

        # 校验最小子条带侧视角 > 0（防止因间距过大导致第一个子条带反向）
        min_look_i = look_angle_deg - (N // 2) * cfg.subswath_spacing_deg
        if min_look_i <= 0.0:
            return SARScanSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=(
                    f"子条带间距过大：最小子条带侧视角 {min_look_i:.1f}° ≤ 0°。"
                    f"请减小 subswath_spacing_deg ({cfg.subswath_spacing_deg}°) "
                    f"或减少 num_subswaths ({N})，或增大中心侧视角 ({look_angle_deg}°)。"
                ),
            )

        # 子条带详细结果
        subswath_results = []
        for i in range(N):
            look_i = look_angle_deg + (i - N // 2) * cfg.subswath_spacing_deg
            R_i = _slant_range(altitude_m, look_i) if abs(look_i) < 85.0 else R
            subswath_results.append({
                "subswath_index": i,
                "look_angle_deg": look_i,
                "slant_range_m": R_i,
                "swath_width_km": swath_per_subswath_km,
                "burst_duration_s": cfg.burst_duration_s,
                "peak_scalloping_db": peak_scalloping,
                "snr_variation_db": snr_var,
            })

        return SARScanSARResult(
            feasible=True,
            num_subswaths_used=N,
            total_swath_width_km=total_swath_km,
            burst_duration_s=cfg.burst_duration_s,
            cycle_time_s=T_cycle,
            scene_size_az_km=scene_az_km,
            scene_area_km2=area_km2,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            peak_scalloping_db=peak_scalloping,
            mean_scalloping_db=mean_scalloping,
            snr_variation_db=snr_var,
            azimuth_resolution_degradation=az_degradation,
            islr_degradation_db=islr_deg,
            limiting_constraint="burst_duration",
            matched_subswath_id=None,
            prf_hz_used=cfg.prf_hz,
            slant_range_m=R,
            peak_power_factor=1.0 / cfg.duty_cycle,
            subswath_results=subswath_results,
        )

    def _compute_discrete(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARScanSARResult:
        """方案B：离散子条带模型（参数由配置直接给出）。"""
        cfg = self.config
        if look_angle_deg <= 0.0 or look_angle_deg >= 80.0:
            return SARScanSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"侧视角 {look_angle_deg:.1f}° 超出有效范围 (0°, 80°)",
            )
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        center_sw = self.select_subswath_position(incidence)

        if center_sw is None:
            return SARScanSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"无子条带覆盖入射角 {incidence:.1f}°（侧视角 {look_angle_deg:.1f}°）",
            )

        R = _slant_range(altitude_m, look_angle_deg)
        N = cfg.num_subswaths

        # 使用中心子条带的突发时长（若未设置则用配置级默认值）
        burst_duration = (
            center_sw.burst_duration_s
            if center_sw.burst_duration_s is not None
            else cfg.burst_duration_s
        )

        # 循环时间
        T_cycle = N * burst_duration + (N - 1) * cfg.burst_switch_time_s

        # 检查 T_burst 是否超过物理上限
        T_dwell_max = cfg.wavelength_m * R / (v_sat * cfg.antenna_length_m)
        if burst_duration > T_dwell_max:
            logger.warning(
                "ScanSAR discrete: T_burst=%.3fs 超过物理上限 T_dwell_max=%.3fs，"
                "方位向分辨率将被天线物理下限 L_a/2=%.2fm 限制。",
                burst_duration, T_dwell_max, cfg.antenna_length_m / 2.0,
            )

        # 方位向分辨率（方案B优先使用配置值，否则按公式推导）
        rho_az = center_sw.azimuth_resolution_m if center_sw.azimuth_resolution_m is not None else \
            _azimuth_resolution_from_burst(cfg.wavelength_m, R, v_sat, burst_duration, cfg.antenna_length_m)
        az_degradation = rho_az / (cfg.antenna_length_m / 2.0)

        # 总幅宽（各子条带幅宽之和）
        total_swath_km = sum(
            (sp.swath_width_rg_km or 0.0) for sp in cfg.subswath_positions[:N]
        )

        # 方位向场景长度
        scene_az_km = v_sat * burst_duration / 1000.0
        area_km2 = scene_az_km * total_swath_km

        # PRF约束检查
        prf_used = center_sw.prf_hz if center_sw.prf_hz is not None else cfg.prf_hz
        prf_result = _check_prf_feasibility(prf_used, v_sat, cfg.antenna_length_m, R)
        if not prf_result['feasible']:
            return SARScanSARResult(
                feasible=False,
                limiting_constraint="prf_subswath_conflict",
                reason=prf_result['reason'],
            )

        # 扇贝效应（方案B：优先使用配置中预计算的值，否则现场计算）
        if center_sw.peak_scalloping_db is not None and cfg.enable_scalloping_model:
            peak_scalloping = center_sw.peak_scalloping_db
            snr_var = center_sw.snr_variation_db if center_sw.snr_variation_db is not None else peak_scalloping
            mean_scalloping = snr_var / 2.0  # 近似
        elif cfg.enable_scalloping_model:
            peak_scalloping, mean_scalloping, snr_var = _compute_scalloping(
                cfg.scalloping_window
            )
        else:
            peak_scalloping, mean_scalloping, snr_var = 0.0, 0.0, 0.0

        islr_deg = _compute_islr_degradation(cfg.scalloping_window)

        # 各子条带详细结果
        subswath_results = self._build_subswath_results_discrete(
            cfg.subswath_positions[:N], altitude_m, v_sat, peak_scalloping, snr_var
        )

        return SARScanSARResult(
            feasible=True,
            num_subswaths_used=N,
            total_swath_width_km=total_swath_km,
            burst_duration_s=burst_duration,
            cycle_time_s=T_cycle,
            scene_size_az_km=scene_az_km,
            scene_area_km2=area_km2,
            range_resolution_m=center_sw.range_resolution_m or cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            peak_scalloping_db=peak_scalloping,
            mean_scalloping_db=mean_scalloping,
            snr_variation_db=snr_var,
            azimuth_resolution_degradation=az_degradation,
            islr_degradation_db=islr_deg,
            limiting_constraint="burst_duration",
            matched_subswath_id=center_sw.subswath_id,
            prf_hz_used=prf_used,
            slant_range_m=R,
            peak_power_factor=1.0 / cfg.duty_cycle,
            subswath_results=subswath_results,
        )

    def _compute_derived(
        self, altitude_m: float, look_angle_deg: float, v_sat: float
    ) -> SARScanSARResult:
        """
        方案C：推导子条带模型。

        步骤：
        1. 将侧视角转换为入射角，选最近中心入射角的子条带
        2. 由系统级固定参数推导 PRF、幅宽、分辨率
        3. 计算扇贝效应
        4. 回填子条带的推导参数（幂等操作）
        """
        cfg = self.config
        if look_angle_deg <= 0.0 or look_angle_deg >= 80.0:
            return SARScanSARResult(
                feasible=False,
                limiting_constraint="infeasible",
                reason=f"侧视角 {look_angle_deg:.1f}° 超出有效范围 (0°, 80°)",
            )
        incidence = _look_to_incidence(altitude_m, look_angle_deg)
        center_sw = self.select_subswath_position(incidence)

        if center_sw is None:
            return SARScanSARResult(
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

        # 检查 T_burst 是否超过物理上限
        T_dwell_max = cfg.wavelength_m * R / (v_sat * cfg.antenna_length_m)
        if burst_duration > T_dwell_max:
            logger.warning(
                "ScanSAR derived: T_burst=%.3fs 超过物理上限 T_dwell_max=%.3fs，"
                "方位向分辨率将被天线物理下限 L_a/2=%.2fm 限制。",
                burst_duration, T_dwell_max, cfg.antenna_length_m / 2.0,
            )

        # 推导 PRF：PRF_min = safety_factor × 2 × V_sat / L_az
        PRF_min = cfg.prf_safety_factor * 2.0 * v_sat / cfg.antenna_length_m
        PRF_used = PRF_min

        # 推导方位向分辨率
        rho_az = _azimuth_resolution_from_burst(
            cfg.wavelength_m, R, v_sat, burst_duration, cfg.antenna_length_m
        )
        az_degradation = rho_az / (cfg.antenna_length_m / 2.0)

        # 推导各子条带幅宽
        cos_look = math.cos(math.radians(look_angle_deg))
        swath_per_sw_km = _subswath_width_km(
            cfg.wavelength_m, cfg.antenna_width_m, R, cos_look
        )
        total_swath_km = swath_per_sw_km * N

        # 方位向场景长度
        scene_az_km = v_sat * burst_duration / 1000.0
        area_km2 = scene_az_km * total_swath_km

        # 扇贝效应（ScanSAR核心差异）
        if cfg.enable_scalloping_model:
            peak_scalloping, mean_scalloping, snr_var = _compute_scalloping(
                cfg.scalloping_window
            )
        else:
            peak_scalloping, mean_scalloping, snr_var = 0.0, 0.0, 0.0

        islr_deg = _compute_islr_degradation(cfg.scalloping_window)

        # 回填所有子条带的推导参数（幂等操作）
        for sp in cfg.subswath_positions[:N]:
            _derive_scansar_subswath_params(
                sp, cfg, R, v_sat, cos_look,
                swath_per_sw_km, rho_az, PRF_used, burst_duration,
                peak_scalloping, snr_var,
            )

        # 各子条带详细结果
        subswath_results = self._build_subswath_results_derived(
            cfg.subswath_positions[:N], altitude_m, v_sat,
            cfg, swath_per_sw_km, rho_az, PRF_used, burst_duration,
            peak_scalloping, snr_var,
        )

        return SARScanSARResult(
            feasible=True,
            num_subswaths_used=N,
            total_swath_width_km=total_swath_km,
            burst_duration_s=burst_duration,
            cycle_time_s=T_cycle,
            scene_size_az_km=scene_az_km,
            scene_area_km2=area_km2,
            range_resolution_m=cfg.range_resolution_m,
            azimuth_resolution_m=rho_az,
            peak_scalloping_db=peak_scalloping,
            mean_scalloping_db=mean_scalloping,
            snr_variation_db=snr_var,
            azimuth_resolution_degradation=az_degradation,
            islr_degradation_db=islr_deg,
            limiting_constraint="burst_duration",
            matched_subswath_id=center_sw.subswath_id,
            prf_hz_used=PRF_used,
            slant_range_m=R,
            peak_power_factor=1.0 / cfg.duty_cycle,
            subswath_results=subswath_results,
        )

    def _build_subswath_results_discrete(
        self,
        subswath_positions: List[ScanSARSubSwathPosition],
        altitude_m: float,
        v_sat: float,
        peak_scalloping: float,
        snr_variation: float,
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
            sw_peak = sp.peak_scalloping_db if sp.peak_scalloping_db is not None else peak_scalloping
            sw_snr = sp.snr_variation_db if sp.snr_variation_db is not None else snr_variation
            results.append({
                "subswath_id": sp.subswath_id,
                "center_incidence_angle_deg": sp.center_incidence_angle_deg,
                "slant_range_m": R_i,
                "swath_width_km": sp.swath_width_rg_km,
                "burst_duration_s": burst_i,
                "range_resolution_m": sp.range_resolution_m or cfg.range_resolution_m,
                "azimuth_resolution_m": sp.azimuth_resolution_m or (cfg.antenna_length_m / 2.0),
                "prf_hz": sp.prf_hz or cfg.prf_hz,
                "peak_scalloping_db": sw_peak,
                "snr_variation_db": sw_snr,
            })
        return results

    def _build_subswath_results_derived(
        self,
        subswath_positions: List[ScanSARSubSwathPosition],
        altitude_m: float,
        v_sat: float,
        cfg: SARScanSARConfig,
        swath_km: float,
        rho_az: float,
        prf: float,
        burst_duration: float,
        peak_scalloping: float,
        snr_variation: float,
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
                "peak_scalloping_db": peak_scalloping,
                "snr_variation_db": snr_variation,
            })
        return results


# ------------------------------------------------------------------
# ScanSAR专有辅助函数
# ------------------------------------------------------------------

def _compute_scalloping(
    window: str = "rectangular",
) -> tuple:
    """
    计算ScanSAR扇贝效应的幅度指标（窗函数决定论模型）。

    物理原理：
      在一次burst内，目标仅被天线方位方向图局部照射。
      设目标在burst内的方位时间偏移为 τ ∈ [-T_burst/2, +T_burst/2]，
      其幅度加权为 sinc(τ / T_burst)（rectangular窗）。

      - 最大增益（burst中心 τ=0）：gain = sinc(0) = 1.0
      - 最小增益（burst边界 τ=±T_burst/2）：gain = sinc(±1/2) = 2/π ≈ 0.6366

      扇贝效应峰值退化 = 20×log10(1 / gain_min) = 20×log10(π/2) ≈ 3.92 dB

    平均扇贝效应（对均匀分布目标取期望）：
      E[gain²] = ∫_{-1/2}^{+1/2} sinc²(u) du ≈ 0.773
      E[gain] ≈ √0.773 ≈ 0.879 → 平均退化 ≈ 1.11 dB

    设计说明：
      当前采用"窗函数决定论"近似：扇贝效应量仅由加权窗类型决定，
      与 PRF 和 burst_duration_s 无关。这对于实际ScanSAR系统（N_p ≥ 50 脉冲/burst）
      是一个合理的工程近似，因为当 N_p 足够大时连续 sinc 模型收敛。
      若需支持极短 burst（N_p < 20），可扩展为精确离散脉冲模型：
        peak_db = 20*log10(|sum(exp(j*2π*k/N_p)) for k in range(N_p)| / N_p)

    不同加权窗的扇贝效应：
      rectangular: peak≈3.92dB，mean≈1.11dB（最大，无加权补偿）
      hamming:     peak≈0.50dB，mean≈0.20dB（有效抑制，主瓣略展宽）
      hanning:     peak≈1.50dB，mean≈0.60dB（折中方案）

    Args:
        window: 方位向加权窗类型（"rectangular" | "hamming" | "hanning"）

    Returns:
        tuple: (peak_scalloping_db, mean_scalloping_db, snr_variation_db)
               均为正值，表示退化量（dB）
    """
    if window == "rectangular":
        # 边缘目标幅度：sinc(1/2) = sin(π/2)/(π/2) = 2/π
        edge_gain = 2.0 / math.pi   # ≈ 0.6366
        peak_db = 20.0 * math.log10(1.0 / edge_gain)   # ≈ 3.92 dB
        # 均匀分布目标的平均RMS增益：数值积分 ∫sinc²(u)du 从-0.5到0.5 ≈ 0.773
        mean_rms_gain = 0.879   # √0.773
        mean_db = 20.0 * math.log10(1.0 / mean_rms_gain)   # ≈ 1.11 dB

    elif window == "hamming":
        # Hamming加权窗大幅减少旁瓣，扇贝效应峰值 < 0.5 dB
        peak_db = 0.50
        mean_db = 0.20

    elif window == "hanning":
        # Hanning（von Hann）窗，介于rectangular和hamming之间
        peak_db = 1.50
        mean_db = 0.60

    else:
        # 未知窗类型，使用rectangular作为保守估计
        logger.warning("未知扇贝效应窗类型 '%s'，使用 rectangular 作为保守估计", window)
        edge_gain = 2.0 / math.pi
        peak_db = 20.0 * math.log10(1.0 / edge_gain)
        mean_db = 1.11

    # SNR方位向非均匀性 = 扇贝效应峰峰值（burst中心比边界好 peak_db）
    snr_variation_db = peak_db

    return peak_db, mean_db, snr_variation_db


def _compute_islr_degradation(window: str = "rectangular") -> float:
    """
    计算积分旁瓣比（ISLR）退化量（dB）。

    ISLR退化由burst截断（有限积分时间）引起：
      - 全孔径stripmap处理：ISLR ≈ -13.3 dB（rectangular窗）
      - ScanSAR burst截断：旁瓣电平升高
      - 退化量 ≈ 3.0 dB（rectangular），< 1.0 dB（hamming/hanning）

    Args:
        window: 方位向加权窗类型

    Returns:
        float: ISLR退化量（dB，正值表示退化量）
    """
    islr_degradation = {
        "rectangular": 3.0,
        "hamming": 0.8,
        "hanning": 1.2,
    }
    return islr_degradation.get(window, 3.0)


def _derive_scansar_subswath_params(
    subswath: ScanSARSubSwathPosition,
    cfg: SARScanSARConfig,
    slant_range_m: float,
    v_sat: float,
    cos_look: float,
    swath_width_km: float,
    rho_az: float,
    prf_hz: float,
    burst_duration_s: float,
    peak_scalloping_db: float,
    snr_variation_db: float,
) -> None:
    """
    为方案C的ScanSAR子条带推导并回填缺失的物理参数（幂等操作）。

    在TOPSAR的基础上，额外回填ScanSAR特有的扇贝效应字段。

    Args:
        subswath: 需要填充参数的子条带对象
        cfg: ScanSAR配置
        slant_range_m: 斜距（m）
        v_sat: 卫星速度（m/s）
        cos_look: cos(侧视角)
        swath_width_km: 推导的子条带幅宽（km）
        rho_az: 方位向分辨率（m）
        prf_hz: 推导的PRF（Hz）
        burst_duration_s: burst时长（s）
        peak_scalloping_db: 峰值扇贝效应（dB）
        snr_variation_db: SNR变化量（dB）
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
    # ScanSAR专有：回填扇贝效应参数
    if subswath.peak_scalloping_db is None:
        subswath.peak_scalloping_db = peak_scalloping_db
    if subswath.snr_variation_db is None:
        subswath.snr_variation_db = snr_variation_db
