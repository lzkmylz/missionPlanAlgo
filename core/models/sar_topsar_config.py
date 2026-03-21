"""
SAR TOPSAR模式配置模块

支持三种建模方案：
  - continuous    (方案A): 连续角度范围 + 统一突发参数
  - discrete_beam (方案B): 离散子条带表，每个子条带含完整参数
  - derived_beam  (方案C): 离散子条带表，仅中心入射角，其余参数由公式推导

TOPSAR核心特点：
  - 多子条带（sub-swath）扫描：通过距离向电子波束扫描实现大幅宽
  - 突发工作方式（burst mode）：每个子条带照射固定时长后切换
  - 无扇贝效应：通过方位向波束渐进扫描消除传统ScanSAR的扇贝问题

关键物理参数：
  - num_subswaths: 子条带数量（通常3-5个）
  - burst_duration_s: 突发持续时间（决定方位向分辨率）
  - burst_switch_time_s: 子条带间切换时间（开销）
  - cycle_time_s = N × T_burst + (N-1) × T_switch: 完整循环时间
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


EARTH_RADIUS_M = 6_371_000.0  # m


# ---------------------------------------------------------------------------
# TOPSARSubSwathPosition
# ---------------------------------------------------------------------------

@dataclass
class TOPSARSubSwathPosition:
    """
    TOPSAR子条带位置描述

    方案B (discrete_beam): 所有字段均需提供。
    方案C (derived_beam) : 仅 subswath_id 和 center_incidence_angle_deg 必须提供，
                            其余字段在运行时由 SARTOPSARCalculator 推导后填入。

    与BeamPosition的区别：
      - 使用 subswath_id 替代 beam_id（语义更清晰）
      - 增加 burst_duration_s 字段（TOPSAR特有）
    """
    subswath_id: str
    center_incidence_angle_deg: float

    # 方案B需要显式提供；方案C由计算器填入（初始为None）
    incidence_angle_min_deg: Optional[float] = None
    incidence_angle_max_deg: Optional[float] = None
    prf_hz: Optional[float] = None
    range_resolution_m: Optional[float] = None
    azimuth_resolution_m: Optional[float] = None
    swath_width_rg_km: Optional[float] = None  # 子条带距离向幅宽（km）
    burst_duration_s: Optional[float] = None   # 该子条带突发持续时间（s）

    # 浮点精度容差（度）
    ANGLE_TOLERANCE_DEG = 0.01

    def __post_init__(self):
        if self.incidence_angle_min_deg is not None and self.incidence_angle_max_deg is not None:
            if self.incidence_angle_min_deg >= self.incidence_angle_max_deg:
                raise ValueError(
                    f"TOPSARSubSwathPosition '{self.subswath_id}': "
                    f"incidence_angle_min ({self.incidence_angle_min_deg}) "
                    f"must be < incidence_angle_max ({self.incidence_angle_max_deg})"
                )
        if self.prf_hz is not None and self.prf_hz <= 0:
            raise ValueError(f"TOPSARSubSwathPosition '{self.subswath_id}': prf_hz must be > 0")
        if self.swath_width_rg_km is not None and self.swath_width_rg_km <= 0:
            raise ValueError(f"TOPSARSubSwathPosition '{self.subswath_id}': swath_width_rg_km must be > 0")
        if self.burst_duration_s is not None and self.burst_duration_s <= 0:
            raise ValueError(f"TOPSARSubSwathPosition '{self.subswath_id}': burst_duration_s must be > 0")

    @property
    def is_fully_specified(self) -> bool:
        """方案B要求的所有字段均已填写"""
        return all(v is not None for v in [
            self.incidence_angle_min_deg, self.incidence_angle_max_deg,
            self.prf_hz, self.range_resolution_m, self.azimuth_resolution_m,
            self.swath_width_rg_km, self.burst_duration_s,
        ])

    def covers(self, incidence_angle_deg: float) -> bool:
        """
        判断该子条带是否覆盖给定入射角

        考虑浮点精度容差（±0.01°），避免边界值判断错误。

        Args:
            incidence_angle_deg: 待检查的入射角（度）

        Returns:
            bool: 如果该角度在子条带覆盖范围内返回True
        """
        if self.incidence_angle_min_deg is None or self.incidence_angle_max_deg is None:
            return False
        # 添加容差避免浮点精度问题
        return (self.incidence_angle_min_deg - self.ANGLE_TOLERANCE_DEG) <= incidence_angle_deg <= (self.incidence_angle_max_deg + self.ANGLE_TOLERANCE_DEG)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TOPSARSubSwathPosition:
        # 支持 subswath_id 和 beam_id 两种字段名（向后兼容）
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
        }


# ---------------------------------------------------------------------------
# SARTOPSARConfig
# ---------------------------------------------------------------------------

@dataclass
class SARTOPSARConfig:
    """
    SAR TOPSAR模式物理建模配置

    beam_model 字段区分三种方案：
      "continuous"    — 方案A：连续角度范围 + 统一突发参数
      "discrete_beam" — 方案B：离散子条带表（含完整参数）
      "derived_beam"  — 方案C：离散子条带表（仅中心入射角，其余推导）

    核心区别（与聚束/滑动聚束/条带）：
      - 多子条带（sub-swath）电子扫描
      - 突发工作方式（burst mode）
      - 循环时间 cycle_time = N × T_burst + (N-1) × T_switch
      - 方位向分辨率由突发时长决定（而非合成孔径长度）
    """

    beam_model: str = "continuous"  # "continuous" | "discrete_beam" | "derived_beam"

    # ---- 系统级固定参数（三种方案共用） ----
    wavelength_m: float = 0.031           # 雷达波长（m），X波段≈0.031m
    antenna_length_m: float = 10.0        # 方位向天线长度（m）
    antenna_width_m: float = 2.0          # 距离向天线宽度（m）
    range_resolution_m: float = 5.0       # 距离向分辨率（m），TOPSAR通常较粗

    # ---- TOPSAR特有核心参数（所有方案必需） ----
    num_subswaths: int = 3                # 子条带数量（通常3-5个）
    burst_duration_s: float = 2.0         # 突发持续时间（s），决定方位向分辨率
    burst_switch_time_s: float = 0.002    # 子条带间切换时间（s），默认2ms
    duty_cycle: float = 0.10              # 发射占空比（影响峰值功率）

    # ---- 方案A专用字段 ----
    prf_hz: float = 1500.0                # 标称工作PRF（Hz）
    center_look_angle_deg: float = 35.0   # 中间子条带中心侧视角（度）
    subswath_spacing_deg: float = 6.0     # 子条带间角间距（度）
    min_look_angle_deg: float = 20.0      # 最小侧视角（度）
    max_look_angle_deg: float = 50.0      # 最大侧视角（度）

    # ---- 方案C专用字段 ----
    prf_safety_factor: float = 1.25       # PRF过采样安全系数

    # ---- 方案B/C共用字段 ----
    subswath_positions: List[TOPSARSubSwathPosition] = field(default_factory=list)

    def __post_init__(self):
        valid_models = ("continuous", "discrete_beam", "derived_beam")
        if self.beam_model not in valid_models:
            raise ValueError(
                f"beam_model must be one of {valid_models}, got '{self.beam_model}'"
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
                    f"beam_model='{self.beam_model}' requires at least one TOPSARSubSwathPosition entry"
                )
            if len(self.subswath_positions) < self.num_subswaths:
                raise ValueError(
                    f"beam_model='{self.beam_model}': num_subswaths ({self.num_subswaths}) "
                    f"exceeds provided subswath_positions count ({len(self.subswath_positions)})"
                )
        if self.beam_model == "discrete_beam":
            for pos in self.subswath_positions:
                if not pos.is_fully_specified:
                    raise ValueError(
                        f"discrete_beam mode: TOPSARSubSwathPosition '{pos.subswath_id}' "
                        f"is missing required fields. All fields must be provided for discrete_beam model."
                    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SARTOPSARConfig:
        subswath_positions = [
            TOPSARSubSwathPosition.from_dict(pos) for pos in d.get("subswath_positions", [])
        ]
        return cls(
            beam_model=d.get("beam_model", "continuous"),
            wavelength_m=float(d.get("wavelength_m", 0.031)),
            antenna_length_m=float(d.get("antenna_length_m", 10.0)),
            antenna_width_m=float(d.get("antenna_width_m", 2.0)),
            range_resolution_m=float(d.get("range_resolution_m", 5.0)),
            num_subswaths=int(d.get("num_subswaths", 3)),
            burst_duration_s=float(d.get("burst_duration_s", 2.0)),
            burst_switch_time_s=float(d.get("burst_switch_time_s", 0.002)),
            duty_cycle=float(d.get("duty_cycle", 0.10)),
            prf_hz=float(d.get("prf_hz", 1500.0)),
            center_look_angle_deg=float(d.get("center_look_angle_deg", 35.0)),
            subswath_spacing_deg=float(d.get("subswath_spacing_deg", 6.0)),
            min_look_angle_deg=float(d.get("min_look_angle_deg", 20.0)),
            max_look_angle_deg=float(d.get("max_look_angle_deg", 50.0)),
            prf_safety_factor=float(d.get("prf_safety_factor", 1.25)),
            subswath_positions=subswath_positions,
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
        }


# ---------------------------------------------------------------------------
# SARTOPSARResult
# ---------------------------------------------------------------------------

@dataclass
class SARTOPSARResult:
    """
    SARTOPSARCalculator 所有公开方法的统一返回结构。

    feasible=False 时，只有 reason 字段有意义，其余为默认值。
    """
    feasible: bool

    # TOPSAR特有结果字段
    num_subswaths_used: int = 0           # 实际使用的子条带数量
    total_swath_width_km: float = 0.0     # 所有子条带合计距离向总幅宽（km）
    burst_duration_s: float = 0.0         # 使用的突发持续时间（s）
    cycle_time_s: float = 0.0             # 一次完整子条带循环时间（s）

    # 场景覆盖范围
    scene_size_az_km: float = 0.0         # 方位向场景长度（km），单次突发覆盖
    scene_area_km2: float = 0.0           # 覆盖面积（km²）

    # 分辨率
    range_resolution_m: float = 0.0
    azimuth_resolution_m: float = 0.0     # TOPSAR特有：由突发时长决定

    # 诊断信息
    limiting_constraint: str = ""         # "burst_duration" | "prf_subswath_conflict" | "num_subswaths" | "infeasible"

    # 匹配的子条带信息（方案B/C）
    matched_subswath_id: Optional[str] = None  # 匹配的中心子条带ID
    prf_hz_used: float = 0.0               # 实际使用的PRF
    slant_range_m: float = 0.0             # 中心子条带斜距（m）

    # 峰值功率因子
    peak_power_factor: float = 0.0         # 1 / duty_cycle

    # 各子条带详细结果（可选，用于调试）
    subswath_results: List[Dict[str, Any]] = field(default_factory=list)

    reason: Optional[str] = None           # 不可行时的原因描述
