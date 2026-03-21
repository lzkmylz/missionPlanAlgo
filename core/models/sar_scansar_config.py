"""
SAR ScanSAR模式配置模块

支持三种建模方案：
  - continuous    (方案A): 连续角度范围 + 统一Burst参数
  - discrete_beam (方案B): 离散子条带表，每个子条带含完整参数（含扇贝效应预计算）
  - derived_beam  (方案C): 离散子条带表，仅中心入射角，其余参数由公式推导

ScanSAR核心特点（区别于TOPSAR）：
  - 多子条带（sub-swath）距离向波束切换：与TOPSAR相同的Burst机制
  - 无方位向波束调制：天线静止照射，不进行进行性扫描（TOPSAR的核心补偿）
  - 扇贝效应（Scalloping）：由于照射仅覆盖天线方位方向图局部，
    方位向产生周期性幅度调制，SNR和ASR随方位位置变化
  - 更宽幅宽、更粗分辨率：典型幅宽300-500km，方位分辨率10-30m

关键物理参数：
  - num_subswaths: 子条带数量（通常4-7个，比TOPSAR多）
  - burst_duration_s: 突发持续时间（典型0.04-0.06s，比TOPSAR短，换取更宽覆盖）
  - burst_switch_time_s: 子条带间切换时间（开销）
  - cycle_time_s = N × T_burst + (N-1) × T_switch: 完整循环时间
  - peak_scalloping_db: 扇贝效应峰值（dB），典型3-5 dB（rectangular加权）
  - snr_variation_db: SNR方位向非均匀性（dB）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 子条带入射角覆盖检查的浮点容差（度）
_SUBSWATH_ANGLE_TOLERANCE_DEG: float = 0.01


# ---------------------------------------------------------------------------
# ScanSARSubSwathPosition
# ---------------------------------------------------------------------------

@dataclass
class ScanSARSubSwathPosition:
    """
    ScanSAR子条带位置描述

    方案B (discrete_beam): 所有字段均需提供（含扇贝效应预计算结果）。
    方案C (derived_beam) : 仅 subswath_id 和 center_incidence_angle_deg 必须提供，
                            其余字段在运行时由 SARScanSARCalculator 推导后填入。

    ScanSAR特有字段（相比TOPSARSubSwathPosition）：
      - peak_scalloping_db: 该子条带的扇贝效应峰值（方案B需提供或由方案C推导）
      - snr_variation_db: 该子条带的SNR方位向变化量
    """
    subswath_id: str
    center_incidence_angle_deg: float

    # 方案B需要显式提供；方案C由计算器填入（初始为None）
    incidence_angle_min_deg: Optional[float] = None
    incidence_angle_max_deg: Optional[float] = None
    prf_hz: Optional[float] = None
    range_resolution_m: Optional[float] = None
    azimuth_resolution_m: Optional[float] = None
    swath_width_rg_km: Optional[float] = None   # 子条带距离向幅宽（km）
    burst_duration_s: Optional[float] = None    # 该子条带突发持续时间（s）

    # ScanSAR专有：扇贝效应（方案B需提供，方案C由计算器推导）
    # 注：discrete_beam模式中这两个字段为可选项；若未提供，物理引擎将调用
    # _compute_scalloping() 从 scalloping_window 参数推导后自动回填。
    peak_scalloping_db: Optional[float] = None  # 扇贝效应峰值（dB，正值表示退化量）
    snr_variation_db: Optional[float] = None    # SNR方位向变化量（dB）

    def __post_init__(self):
        if self.incidence_angle_min_deg is not None and self.incidence_angle_max_deg is not None:
            if self.incidence_angle_min_deg >= self.incidence_angle_max_deg:
                raise ValueError(
                    f"ScanSARSubSwathPosition '{self.subswath_id}': "
                    f"incidence_angle_min ({self.incidence_angle_min_deg}) "
                    f"must be < incidence_angle_max ({self.incidence_angle_max_deg})"
                )
        if self.prf_hz is not None and self.prf_hz <= 0:
            raise ValueError(f"ScanSARSubSwathPosition '{self.subswath_id}': prf_hz must be > 0")
        if self.swath_width_rg_km is not None and self.swath_width_rg_km <= 0:
            raise ValueError(f"ScanSARSubSwathPosition '{self.subswath_id}': swath_width_rg_km must be > 0")
        if self.burst_duration_s is not None and self.burst_duration_s <= 0:
            raise ValueError(f"ScanSARSubSwathPosition '{self.subswath_id}': burst_duration_s must be > 0")
        if self.peak_scalloping_db is not None and self.peak_scalloping_db < 0:
            raise ValueError(
                f"ScanSARSubSwathPosition '{self.subswath_id}': "
                f"peak_scalloping_db must be >= 0 (it is a degradation magnitude in dB)"
            )

    @property
    def is_fully_specified(self) -> bool:
        """方案B要求的所有基础字段均已填写（不含可选的扇贝效应字段）"""
        return all(v is not None for v in [
            self.incidence_angle_min_deg, self.incidence_angle_max_deg,
            self.prf_hz, self.range_resolution_m, self.azimuth_resolution_m,
            self.swath_width_rg_km, self.burst_duration_s,
        ])

    def covers(self, incidence_angle_deg: float) -> bool:
        """
        判断该子条带是否覆盖给定入射角

        Args:
            incidence_angle_deg: 待检查的入射角（度）

        Returns:
            bool: 如果该角度在子条带覆盖范围内返回True
        """
        if self.incidence_angle_min_deg is None or self.incidence_angle_max_deg is None:
            return False
        return (
            (self.incidence_angle_min_deg - _SUBSWATH_ANGLE_TOLERANCE_DEG)
            <= incidence_angle_deg
            <= (self.incidence_angle_max_deg + _SUBSWATH_ANGLE_TOLERANCE_DEG)
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ScanSARSubSwathPosition:
        subswath_id = d.get("subswath_id") or d.get("beam_id", "")
        return cls(
            subswath_id=str(subswath_id),
            center_incidence_angle_deg=float(d["center_incidence_angle_deg"]),
            incidence_angle_min_deg=float(d["incidence_angle_min_deg"]) if d.get("incidence_angle_min_deg") is not None else None,
            incidence_angle_max_deg=float(d["incidence_angle_max_deg"]) if d.get("incidence_angle_max_deg") is not None else None,
            prf_hz=float(d["prf_hz"]) if d.get("prf_hz") is not None else None,
            range_resolution_m=float(d["range_resolution_m"]) if d.get("range_resolution_m") is not None else None,
            azimuth_resolution_m=float(d["azimuth_resolution_m"]) if d.get("azimuth_resolution_m") is not None else None,
            swath_width_rg_km=float(d["swath_width_rg_km"]) if d.get("swath_width_rg_km") is not None else None,
            burst_duration_s=float(d["burst_duration_s"]) if d.get("burst_duration_s") is not None else None,
            peak_scalloping_db=float(d["peak_scalloping_db"]) if d.get("peak_scalloping_db") is not None else None,
            snr_variation_db=float(d["snr_variation_db"]) if d.get("snr_variation_db") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subswath_id": self.subswath_id,
            "center_incidence_angle_deg": self.center_incidence_angle_deg,
            "incidence_angle_min_deg": self.incidence_angle_min_deg,
            "incidence_angle_max_deg": self.incidence_angle_max_deg,
            "prf_hz": self.prf_hz,
            "range_resolution_m": self.range_resolution_m,
            "azimuth_resolution_m": self.azimuth_resolution_m,
            "swath_width_rg_km": self.swath_width_rg_km,
            "burst_duration_s": self.burst_duration_s,
            "peak_scalloping_db": self.peak_scalloping_db,
            "snr_variation_db": self.snr_variation_db,
        }


# ---------------------------------------------------------------------------
# SARScanSARConfig
# ---------------------------------------------------------------------------

@dataclass
class SARScanSARConfig:
    """
    SAR ScanSAR模式物理建模配置

    beam_model 字段区分三种方案：
      "continuous"    — 方案A：连续角度范围 + 统一Burst参数
      "discrete_beam" — 方案B：离散子条带表（含完整参数及扇贝效应）
      "derived_beam"  — 方案C：离散子条带表（仅中心入射角，其余推导）

    ScanSAR与TOPSAR的核心区别（影响物理引擎行为）：
      - 无方位向波束调制 → 保留扇贝效应
      - burst_duration_s 通常比TOPSAR短（典型0.04-0.06s vs TOPSAR 0.1-0.3s）
        以换取更多子条带 → 更宽总幅宽
      - 方位分辨率公式相同：ρ_az = λR/(2V×T_burst)，但T_burst更短 → 分辨率更粗
      - 扇贝效应模型：enable_scalloping_model=True 时计算并输出扇贝效应指标
    """

    beam_model: str = "continuous"  # "continuous" | "discrete_beam" | "derived_beam"

    # ---- 系统级固定参数（三种方案共用） ----
    wavelength_m: float = 0.031           # 雷达波长（m），X波段≈0.031m
    antenna_length_m: float = 10.0        # 方位向天线长度（m）
    antenna_width_m: float = 2.0          # 距离向天线宽度（m）
    range_resolution_m: float = 15.0      # 距离向分辨率（m），ScanSAR通常较粗

    # ---- ScanSAR核心Burst参数（所有方案必需） ----
    num_subswaths: int = 5                # 子条带数量（ScanSAR通常4-7个，比TOPSAR多）
    burst_duration_s: float = 0.05        # 突发持续时间（s），ScanSAR典型0.04-0.06s
    burst_switch_time_s: float = 0.002    # 子条带间切换时间（s），默认2ms
    duty_cycle: float = 0.10              # 发射占空比（影响峰值功率）

    # ---- 方案A专用字段 ----
    prf_hz: float = 1500.0                # 标称工作PRF（Hz）
    center_look_angle_deg: float = 35.0   # 中间子条带中心侧视角（度）
    subswath_spacing_deg: float = 1.5     # 子条带间角间距（度），ScanSAR通常更密
    min_look_angle_deg: float = 15.0      # 最小侧视角（度）
    max_look_angle_deg: float = 60.0      # 最大侧视角（度），ScanSAR幅宽更大

    # ---- 方案C专用字段 ----
    prf_safety_factor: float = 1.25       # PRF过采样安全系数

    # ---- 方案B/C共用字段 ----
    subswath_positions: List[ScanSARSubSwathPosition] = field(default_factory=list)

    # ---- ScanSAR专有：扇贝效应建模 ----
    enable_scalloping_model: bool = True          # 是否计算扇贝效应指标
    scalloping_window: str = "rectangular"        # 加权窗类型："rectangular"|"hamming"|"hanning"
    #   rectangular: 峰值扇贝效应≈3.92dB（最严重，最常见于传统ScanSAR）
    #   hamming:     峰值扇贝效应<1dB（改善幅度大，但主瓣展宽）
    #   hanning:     峰值扇贝效应≈1.5dB

    def __post_init__(self):
        valid_models = ("continuous", "discrete_beam", "derived_beam")
        if self.beam_model not in valid_models:
            raise ValueError(
                f"beam_model must be one of {valid_models}, got '{self.beam_model}'"
            )
        valid_windows = ("rectangular", "hamming", "hanning")
        if self.scalloping_window not in valid_windows:
            raise ValueError(
                f"scalloping_window must be one of {valid_windows}, got '{self.scalloping_window}'"
            )
        if self.num_subswaths < 1:
            raise ValueError(f"num_subswaths must be >= 1, got {self.num_subswaths}")
        if self.burst_duration_s <= 0:
            raise ValueError(f"burst_duration_s must be > 0, got {self.burst_duration_s}")
        if self.burst_switch_time_s < 0:
            raise ValueError(f"burst_switch_time_s must be >= 0, got {self.burst_switch_time_s}")
        if not 0 < self.duty_cycle <= 1.0:
            raise ValueError(f"duty_cycle must be in (0, 1], got {self.duty_cycle}")
        if self.beam_model in ("discrete_beam", "derived_beam"):
            if not self.subswath_positions:
                raise ValueError(
                    f"beam_model='{self.beam_model}' requires at least one ScanSARSubSwathPosition entry"
                )
            if len(self.subswath_positions) < self.num_subswaths:
                raise ValueError(
                    f"beam_model='{self.beam_model}': num_subswaths ({self.num_subswaths}) "
                    f"exceeds provided subswath_positions count ({len(self.subswath_positions)})"
                )
            if len(self.subswath_positions) > self.num_subswaths:
                warnings.warn(
                    f"beam_model='{self.beam_model}': subswath_positions has "
                    f"{len(self.subswath_positions)} entries but num_subswaths={self.num_subswaths}. "
                    f"Only the first {self.num_subswaths} positions will be used; "
                    f"the remaining {len(self.subswath_positions) - self.num_subswaths} are discarded.",
                    UserWarning,
                    stacklevel=2,
                )
        if self.beam_model == "discrete_beam":
            for pos in self.subswath_positions:
                if not pos.is_fully_specified:
                    raise ValueError(
                        f"discrete_beam mode: ScanSARSubSwathPosition '{pos.subswath_id}' "
                        f"is missing required fields. All fields must be provided for discrete_beam model."
                    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SARScanSARConfig:
        subswath_positions = [
            ScanSARSubSwathPosition.from_dict(pos) for pos in d.get("subswath_positions", [])
        ]
        return cls(
            beam_model=d.get("beam_model", "continuous"),
            wavelength_m=float(d.get("wavelength_m", 0.031)),
            antenna_length_m=float(d.get("antenna_length_m", 10.0)),
            antenna_width_m=float(d.get("antenna_width_m", 2.0)),
            range_resolution_m=float(d.get("range_resolution_m", 15.0)),
            num_subswaths=int(d.get("num_subswaths", 5)),
            burst_duration_s=float(d.get("burst_duration_s", 0.05)),
            burst_switch_time_s=float(d.get("burst_switch_time_s", 0.002)),
            duty_cycle=float(d.get("duty_cycle", 0.10)),
            prf_hz=float(d.get("prf_hz", 1500.0)),
            center_look_angle_deg=float(d.get("center_look_angle_deg", 35.0)),
            subswath_spacing_deg=float(d.get("subswath_spacing_deg", 1.5)),
            min_look_angle_deg=float(d.get("min_look_angle_deg", 15.0)),
            max_look_angle_deg=float(d.get("max_look_angle_deg", 60.0)),
            prf_safety_factor=float(d.get("prf_safety_factor", 1.25)),
            subswath_positions=subswath_positions,
            enable_scalloping_model=bool(d.get("enable_scalloping_model", True)),
            scalloping_window=str(d.get("scalloping_window", "rectangular")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam_model": self.beam_model,
            "wavelength_m": self.wavelength_m,
            "antenna_length_m": self.antenna_length_m,
            "antenna_width_m": self.antenna_width_m,
            "range_resolution_m": self.range_resolution_m,
            "num_subswaths": self.num_subswaths,
            "burst_duration_s": self.burst_duration_s,
            "burst_switch_time_s": self.burst_switch_time_s,
            "duty_cycle": self.duty_cycle,
            "prf_hz": self.prf_hz,
            "center_look_angle_deg": self.center_look_angle_deg,
            "subswath_spacing_deg": self.subswath_spacing_deg,
            "min_look_angle_deg": self.min_look_angle_deg,
            "max_look_angle_deg": self.max_look_angle_deg,
            "prf_safety_factor": self.prf_safety_factor,
            "subswath_positions": [pos.to_dict() for pos in self.subswath_positions],
            "enable_scalloping_model": self.enable_scalloping_model,
            "scalloping_window": self.scalloping_window,
        }


# ---------------------------------------------------------------------------
# SARScanSARResult
# ---------------------------------------------------------------------------

@dataclass
class SARScanSARResult:
    """
    SARScanSARCalculator 所有公开方法的统一返回结构。

    feasible=False 时，只有 reason 字段有意义，其余为默认值。

    ScanSAR特有字段（相比SARTOPSARResult）：
      - peak_scalloping_db: 扇贝效应峰值（dB），典型3-5dB（rectangular加权）
      - mean_scalloping_db: 扇贝效应方位向平均值（dB）
      - snr_variation_db: SNR方位向变化量（dB），等于扇贝效应量
      - azimuth_resolution_degradation: 相对stripmap的方位分辨率退化倍数
      - islr_degradation_db: 积分旁瓣比退化量（dB），由burst截断引起
    """
    feasible: bool

    # Burst特有结果字段
    num_subswaths_used: int = 0           # 实际使用的子条带数量
    total_swath_width_km: float = 0.0     # 所有子条带合计距离向总幅宽（km）
    burst_duration_s: float = 0.0         # 使用的突发持续时间（s）
    cycle_time_s: float = 0.0             # 一次完整子条带循环时间（s）

    # 场景覆盖范围
    scene_size_az_km: float = 0.0         # 方位向场景长度（km），单次突发覆盖
    scene_area_km2: float = 0.0           # 覆盖面积（km²）

    # 分辨率
    range_resolution_m: float = 0.0
    azimuth_resolution_m: float = 0.0     # burst-limited，比stripmap粗

    # ---- ScanSAR专有：扇贝效应及质量指标 ----
    peak_scalloping_db: float = 0.0
    """
    扇贝效应峰值（dB，正值，表示退化量）。
    物理含义：方位向边缘位置的信号幅度相对中心的损失。
    典型值：rectangular窗≈3.92dB，hamming窗≈0.5dB，hanning窗≈1.5dB。
    """

    mean_scalloping_db: float = 0.0
    """
    扇贝效应方位向平均值（dB）。
    对均匀分布目标取期望，等效于系统SNR平均损失。
    近似 ≈ peak_scalloping_db / 2。
    """

    snr_variation_db: float = 0.0
    """
    SNR方位向非均匀性（dB）。
    等于扇贝效应峰峰值：SNR在方位向边缘比中心低 snr_variation_db dB。
    """

    azimuth_resolution_degradation: float = 1.0
    """
    方位分辨率退化倍数（相对stripmap/spotlight全孔径模式）。
    ρ_az_scansar / ρ_az_stripmap = T_sa / T_burst，其中T_sa为全合成孔径时间。
    典型值：5-20倍（子条带越多，退化越严重）。
    """

    islr_degradation_db: float = 0.0
    """
    积分旁瓣比退化量（dB）。
    由burst截断（有限积分时间）引起的旁瓣电平升高。
    rectangular窗时最明显（约-13dB → 更差）。
    """

    # 诊断信息
    limiting_constraint: str = ""

    # 匹配的子条带信息（方案B/C）
    matched_subswath_id: Optional[str] = None
    prf_hz_used: float = 0.0
    slant_range_m: float = 0.0

    # 峰值功率因子
    peak_power_factor: float = 0.0        # 1 / duty_cycle

    # 各子条带详细结果（可选，用于调试）
    subswath_results: List[Dict[str, Any]] = field(default_factory=list)

    reason: Optional[str] = None          # 不可行时的原因描述
