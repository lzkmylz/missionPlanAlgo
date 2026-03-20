"""
SAR聚束模式配置模块

支持三种建模方案：
  - continuous    (方案A): 连续角度范围 + 单一PRF
  - discrete_beam (方案B): 离散波位表，每个波位含完整参数
  - derived_beam  (方案C): 离散波位表，每个波位仅含中心入射角，其余参数由物理公式推导
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SPEED_OF_LIGHT = 3.0e8  # m/s
EARTH_RADIUS_M = 6_371_000.0  # m


# ---------------------------------------------------------------------------
# BeamPosition
# ---------------------------------------------------------------------------

@dataclass
class BeamPosition:
    """
    离散波位描述

    方案B (discrete_beam): 所有字段均需提供。
    方案C (derived_beam) : 仅 beam_id 和 center_incidence_angle_deg 必须提供，
                            其余字段在运行时由 SARSpotlightCalculator 推导后填入。
    """
    beam_id: str
    center_incidence_angle_deg: float

    # 方案B需要显式提供；方案C由计算器填入（初始为None）
    incidence_angle_min_deg: Optional[float] = None
    incidence_angle_max_deg: Optional[float] = None
    prf_hz: Optional[float] = None
    range_resolution_m: Optional[float] = None
    azimuth_resolution_m: Optional[float] = None
    scene_size_az_km: Optional[float] = None
    scene_size_rg_km: Optional[float] = None

    def __post_init__(self):
        if self.incidence_angle_min_deg is not None and self.incidence_angle_max_deg is not None:
            if self.incidence_angle_min_deg >= self.incidence_angle_max_deg:
                raise ValueError(
                    f"BeamPosition '{self.beam_id}': "
                    f"incidence_angle_min ({self.incidence_angle_min_deg}) "
                    f"must be < incidence_angle_max ({self.incidence_angle_max_deg})"
                )
        if self.prf_hz is not None and self.prf_hz <= 0:
            raise ValueError(f"BeamPosition '{self.beam_id}': prf_hz must be > 0")

    @property
    def is_fully_specified(self) -> bool:
        """方案B要求的所有字段均已填写"""
        return all(v is not None for v in [
            self.incidence_angle_min_deg, self.incidence_angle_max_deg,
            self.prf_hz, self.range_resolution_m, self.azimuth_resolution_m,
            self.scene_size_az_km, self.scene_size_rg_km,
        ])

    def covers(self, incidence_angle_deg: float) -> bool:
        """判断该波位是否覆盖给定入射角"""
        if self.incidence_angle_min_deg is None or self.incidence_angle_max_deg is None:
            return False
        return self.incidence_angle_min_deg <= incidence_angle_deg <= self.incidence_angle_max_deg

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BeamPosition:
        return cls(
            beam_id=str(d["beam_id"]),
            center_incidence_angle_deg=float(d["center_incidence_angle_deg"]),
            incidence_angle_min_deg=float(d["incidence_angle_min_deg"]) if "incidence_angle_min_deg" in d else None,
            incidence_angle_max_deg=float(d["incidence_angle_max_deg"]) if "incidence_angle_max_deg" in d else None,
            prf_hz=float(d["prf_hz"]) if "prf_hz" in d else None,
            range_resolution_m=float(d["range_resolution_m"]) if "range_resolution_m" in d else None,
            azimuth_resolution_m=float(d["azimuth_resolution_m"]) if "azimuth_resolution_m" in d else None,
            scene_size_az_km=float(d["scene_size_az_km"]) if "scene_size_az_km" in d else None,
            scene_size_rg_km=float(d["scene_size_rg_km"]) if "scene_size_rg_km" in d else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam_id": self.beam_id,
            "center_incidence_angle_deg": self.center_incidence_angle_deg,
            "incidence_angle_min_deg": self.incidence_angle_min_deg,
            "incidence_angle_max_deg": self.incidence_angle_max_deg,
            "prf_hz": self.prf_hz,
            "range_resolution_m": self.range_resolution_m,
            "azimuth_resolution_m": self.azimuth_resolution_m,
            "scene_size_az_km": self.scene_size_az_km,
            "scene_size_rg_km": self.scene_size_rg_km,
        }


# ---------------------------------------------------------------------------
# SARSpotlightConfig
# ---------------------------------------------------------------------------

@dataclass
class SARSpotlightConfig:
    """
    SAR聚束模式物理建模配置

    beam_model 字段区分三种方案：
      "continuous"    — 方案A：连续角度范围 + 单一PRF
      "discrete_beam" — 方案B：离散波位表（含完整参数）
      "derived_beam"  — 方案C：离散波位表（仅中心入射角，其余推导）
    """

    beam_model: str = "continuous"  # "continuous" | "discrete_beam" | "derived_beam"

    # ---- 系统级固定参数（三种方案共用） ----
    wavelength_m: float = 0.031           # 雷达波长（m），X波段≈0.031m
    antenna_length_m: float = 6.0         # 方位向天线长度（m）
    antenna_width_m: float = 1.5          # 距离向天线宽度（m）
    range_resolution_m: float = 1.0       # 距离向分辨率（m）

    # ---- 方案A专用字段 ----
    prf_hz: float = 3000.0                # 标称工作PRF（Hz）
    max_azimuth_steering_deg: float = 2.0  # 方位向电子扫描半角（度）
    max_range_steering_deg: float = 2.0   # 距离向电子扫描半角（度）
    min_look_angle_deg: float = 20.0      # 最小侧视角（度）
    max_look_angle_deg: float = 50.0      # 最大侧视角（度）

    # ---- 方案C专用字段 ----
    prf_safety_factor: float = 1.25       # PRF过采样安全系数（多普勒带宽 × 此系数 = 工作PRF）

    # ---- 方案B/C共用字段 ----
    beam_positions: List[BeamPosition] = field(default_factory=list)

    def __post_init__(self):
        valid_models = ("continuous", "discrete_beam", "derived_beam")
        if self.beam_model not in valid_models:
            raise ValueError(
                f"beam_model must be one of {valid_models}, got '{self.beam_model}'"
            )
        if self.beam_model in ("discrete_beam", "derived_beam") and not self.beam_positions:
            raise ValueError(
                f"beam_model='{self.beam_model}' requires at least one BeamPosition entry"
            )
        if self.beam_model == "discrete_beam":
            for bp in self.beam_positions:
                if not bp.is_fully_specified:
                    raise ValueError(
                        f"discrete_beam mode: BeamPosition '{bp.beam_id}' is missing required fields. "
                        "All fields must be provided for discrete_beam model."
                    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SARSpotlightConfig:
        beam_positions = [
            BeamPosition.from_dict(bp) for bp in d.get("beam_positions", [])
        ]
        return cls(
            beam_model=d.get("beam_model", "continuous"),
            wavelength_m=float(d.get("wavelength_m", 0.031)),
            antenna_length_m=float(d.get("antenna_length_m", 6.0)),
            antenna_width_m=float(d.get("antenna_width_m", 1.5)),
            range_resolution_m=float(d.get("range_resolution_m", 1.0)),
            prf_hz=float(d.get("prf_hz", 3000.0)),
            max_azimuth_steering_deg=float(d.get("max_azimuth_steering_deg", 2.0)),
            max_range_steering_deg=float(d.get("max_range_steering_deg", 2.0)),
            min_look_angle_deg=float(d.get("min_look_angle_deg", 20.0)),
            max_look_angle_deg=float(d.get("max_look_angle_deg", 50.0)),
            prf_safety_factor=float(d.get("prf_safety_factor", 1.25)),
            beam_positions=beam_positions,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam_model": self.beam_model,
            "wavelength_m": self.wavelength_m,
            "antenna_length_m": self.antenna_length_m,
            "antenna_width_m": self.antenna_width_m,
            "range_resolution_m": self.range_resolution_m,
            "prf_hz": self.prf_hz,
            "max_azimuth_steering_deg": self.max_azimuth_steering_deg,
            "max_range_steering_deg": self.max_range_steering_deg,
            "min_look_angle_deg": self.min_look_angle_deg,
            "max_look_angle_deg": self.max_look_angle_deg,
            "prf_safety_factor": self.prf_safety_factor,
            "beam_positions": [bp.to_dict() for bp in self.beam_positions],
        }


# ---------------------------------------------------------------------------
# SARSpotlightResult
# ---------------------------------------------------------------------------

@dataclass
class SARSpotlightResult:
    """
    SARSpotlightCalculator 所有公开方法的统一返回结构。

    feasible=False 时，只有 reason 字段有意义，其余为默认值。
    """
    feasible: bool

    # 驻留时间
    dwell_time_s: float = 0.0

    # 场景覆盖范围
    scene_size_az_km: float = 0.0   # 方位向场景长度（km）
    scene_size_rg_km: float = 0.0   # 距离向场景长度（km）
    scene_area_km2: float = 0.0     # 覆盖面积（km²）

    # 分辨率
    range_resolution_m: float = 0.0
    azimuth_resolution_m: float = 0.0

    # 诊断信息
    limiting_constraint: str = ""
    # "azimuth_steering" | "prf_ambiguity" | "beam_scene_size" | "infeasible"

    matched_beam_id: Optional[str] = None  # 方案B/C匹配的波位ID
    prf_hz_used: float = 0.0               # 实际使用的PRF
    slant_range_m: float = 0.0             # 斜距（m）

    reason: Optional[str] = None           # 不可行时的原因描述
