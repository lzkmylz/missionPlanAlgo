"""
SAR滑动聚束模式配置模块

支持三种建模方案：
  - continuous    (方案A): 连续角度范围 + VRC距离
  - discrete_beam (方案B): 离散波位表，每个波位含完整参数（含VRC）
  - derived_beam  (方案C): 离散波位表，仅中心入射角，其余参数由公式推导

核心物理参数：
  vrc_distance_m (D_vrc): 虚拟旋转中心距离（m）
  duty_cycle: 发射占空比（影响峰值功率需求）

物理公式差异（与聚束模式）：
  - 有效场景速度: V_eff = V_sat × (1 - R/D_vrc)
  - 方位向分辨率: ρ_az = λ × D_vrc / (2 × V_sat × T_dwell)
  - 方位向驻留时间: T_az = 2 × θ_az × D_vrc / V_sat（分母变为 D_vrc）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SPEED_OF_LIGHT = 3.0e8  # m/s
EARTH_RADIUS_M = 6_371_000.0  # m


# ---------------------------------------------------------------------------
# SlidingBeamPosition
# ---------------------------------------------------------------------------

@dataclass
class SlidingBeamPosition:
    """
    滑动聚束模式的离散波位描述

    方案B (discrete_beam): vrc_distance_m 必须提供，所有字段均需填写。
    方案C (derived_beam) : 仅 beam_id 和 center_incidence_angle_deg 必须提供，
                            vrc_distance_m 从顶层配置继承，
                            其余字段在运行时推导后填入。
    """
    beam_id: str
    center_incidence_angle_deg: float

    # 方案B需要显式提供；方案C由计算器填入（初始为None）
    vrc_distance_m: Optional[float] = None
    incidence_angle_min_deg: Optional[float] = None
    incidence_angle_max_deg: Optional[float] = None
    prf_hz: Optional[float] = None
    range_resolution_m: Optional[float] = None
    azimuth_resolution_m: Optional[float] = None
    scene_size_az_km: Optional[float] = None
    scene_size_rg_km: Optional[float] = None

    def __post_init__(self):
        if self.vrc_distance_m is not None:
            if self.vrc_distance_m <= 0:
                raise ValueError(
                    f"SlidingBeamPosition '{self.beam_id}': "
                    f"vrc_distance_m must be > 0, got {self.vrc_distance_m}"
                )
        if self.incidence_angle_min_deg is not None and self.incidence_angle_max_deg is not None:
            if self.incidence_angle_min_deg >= self.incidence_angle_max_deg:
                raise ValueError(
                    f"SlidingBeamPosition '{self.beam_id}': "
                    f"incidence_angle_min ({self.incidence_angle_min_deg}) "
                    f"must be < incidence_angle_max ({self.incidence_angle_max_deg})"
                )
        if self.prf_hz is not None and self.prf_hz <= 0:
            raise ValueError(f"SlidingBeamPosition '{self.beam_id}': prf_hz must be > 0")

    @property
    def is_fully_specified(self) -> bool:
        """方案B要求的所有字段均已填写（含vrc_distance_m）"""
        return all(v is not None for v in [
            self.vrc_distance_m,
            self.incidence_angle_min_deg, self.incidence_angle_max_deg,
            self.prf_hz, self.range_resolution_m, self.azimuth_resolution_m,
            self.scene_size_az_km, self.scene_size_rg_km,
        ])

    # 浮点精度容差（度）
    ANGLE_TOLERANCE_DEG = 0.01

    def covers(self, incidence_angle_deg: float) -> bool:
        """
        判断该波位是否覆盖给定入射角

        考虑浮点精度容差（±0.01°），避免边界值判断错误。

        Args:
            incidence_angle_deg: 待检查的入射角（度）

        Returns:
            bool: 如果该角度在波位覆盖范围内返回True
        """
        if self.incidence_angle_min_deg is None or self.incidence_angle_max_deg is None:
            return False
        # 添加容差避免浮点精度问题
        return (self.incidence_angle_min_deg - self.ANGLE_TOLERANCE_DEG) <= incidence_angle_deg <= (self.incidence_angle_max_deg + self.ANGLE_TOLERANCE_DEG)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlidingBeamPosition:
        return cls(
            beam_id=str(d["beam_id"]),
            center_incidence_angle_deg=float(d["center_incidence_angle_deg"]),
            vrc_distance_m=float(d["vrc_distance_m"]) if "vrc_distance_m" in d else None,
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
            "vrc_distance_m": self.vrc_distance_m,
            "incidence_angle_min_deg": self.incidence_angle_min_deg,
            "incidence_angle_max_deg": self.incidence_angle_max_deg,
            "prf_hz": self.prf_hz,
            "range_resolution_m": self.range_resolution_m,
            "azimuth_resolution_m": self.azimuth_resolution_m,
            "scene_size_az_km": self.scene_size_az_km,
            "scene_size_rg_km": self.scene_size_rg_km,
        }


# ---------------------------------------------------------------------------
# SARSlidingSpotlightConfig
# ---------------------------------------------------------------------------

@dataclass
class SARSlidingSpotlightConfig:
    """
    SAR滑动聚束模式物理建模配置

    beam_model 字段区分三种方案：
      "continuous"    — 方案A：连续角度范围 + 单一PRF + 全局VRC距离
      "discrete_beam" — 方案B：离散波位表（含完整参数，含VRC）
      "derived_beam"  — 方案C：离散波位表（仅中心入射角，其余推导）

    核心区别（与聚束模式）：
      - vrc_distance_m: 虚拟旋转中心距离（滑动聚束特有，必填）
      - duty_cycle: 发射占空比
      - 方位向分辨率公式不同
      - 方位向驻留时间公式分母为 D_vrc 而非 R
    """

    beam_model: str = "continuous"  # "continuous" | "discrete_beam" | "derived_beam"

    # ---- 系统级固定参数（三种方案共用） ----
    wavelength_m: float = 0.031           # 雷达波长（m），X波段≈0.031m
    antenna_length_m: float = 10.0        # 方位向天线长度（m）
    antenna_width_m: float = 2.0          # 距离向天线宽度（m）
    range_resolution_m: float = 2.0       # 距离向分辨率（m）

    # ---- 方案A/C共用字段 ----
    vrc_distance_m: float = 2_000_000.0   # 虚拟旋转中心距离（m）——滑动聚束特有
    duty_cycle: float = 0.15              # 发射占空比（影响峰值功率需求）

    # ---- 方案A专用字段 ----
    prf_hz: float = 3000.0                # 标称工作PRF（Hz）
    max_azimuth_steering_deg: float = 5.0  # 方位向电子扫描半角（度）
    max_range_steering_deg: float = 3.0   # 距离向电子扫描半角（度）
    min_look_angle_deg: float = 20.0      # 最小侧视角（度）
    max_look_angle_deg: float = 50.0      # 最大侧视角（度）

    # ---- 方案C专用字段 ----
    prf_safety_factor: float = 1.25       # PRF过采样安全系数

    # ---- 方案B/C共用字段 ----
    beam_positions: List[SlidingBeamPosition] = field(default_factory=list)

    def __post_init__(self):
        valid_models = ("continuous", "discrete_beam", "derived_beam")
        if self.beam_model not in valid_models:
            raise ValueError(
                f"beam_model must be one of {valid_models}, got '{self.beam_model}'"
            )

        # VRC距离必须为必填项（滑动聚束核心参数）
        if self.vrc_distance_m <= 0:
            raise ValueError(f"vrc_distance_m must be > 0, got {self.vrc_distance_m}")

        if self.beam_model in ("discrete_beam", "derived_beam") and not self.beam_positions:
            raise ValueError(
                f"beam_model='{self.beam_model}' requires at least one SlidingBeamPosition entry"
            )

        if self.beam_model == "discrete_beam":
            for bp in self.beam_positions:
                if not bp.is_fully_specified:
                    raise ValueError(
                        f"discrete_beam mode: SlidingBeamPosition '{bp.beam_id}' is missing required fields. "
                        f"All fields must be provided for discrete_beam model."
                    )
                # 方案B中每个波位的vrc_distance_m必须提供（且在BeamPosition中已验证>0）
                if bp.vrc_distance_m is None:
                    raise ValueError(
                        f"discrete_beam mode: SlidingBeamPosition '{bp.beam_id}' must have vrc_distance_m"
                    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SARSlidingSpotlightConfig:
        beam_positions = [
            SlidingBeamPosition.from_dict(bp) for bp in d.get("beam_positions", [])
        ]

        # vrc_distance_m 严格检查：缺失或非法时抛出错误
        if "vrc_distance_m" not in d:
            raise ValueError("SARSlidingSpotlightConfig requires 'vrc_distance_m' field")

        return cls(
            beam_model=d.get("beam_model", "continuous"),
            wavelength_m=float(d.get("wavelength_m", 0.031)),
            antenna_length_m=float(d.get("antenna_length_m", 10.0)),
            antenna_width_m=float(d.get("antenna_width_m", 2.0)),
            range_resolution_m=float(d.get("range_resolution_m", 2.0)),
            vrc_distance_m=float(d["vrc_distance_m"]),
            duty_cycle=float(d.get("duty_cycle", 0.15)),
            prf_hz=float(d.get("prf_hz", 3000.0)),
            max_azimuth_steering_deg=float(d.get("max_azimuth_steering_deg", 5.0)),
            max_range_steering_deg=float(d.get("max_range_steering_deg", 3.0)),
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
            "vrc_distance_m": self.vrc_distance_m,
            "duty_cycle": self.duty_cycle,
            "prf_hz": self.prf_hz,
            "max_azimuth_steering_deg": self.max_azimuth_steering_deg,
            "max_range_steering_deg": self.max_range_steering_deg,
            "min_look_angle_deg": self.min_look_angle_deg,
            "max_look_angle_deg": self.max_look_angle_deg,
            "prf_safety_factor": self.prf_safety_factor,
            "beam_positions": [bp.to_dict() for bp in self.beam_positions],
        }


# ---------------------------------------------------------------------------
# SARSlidingSpotlightResult
# ---------------------------------------------------------------------------

@dataclass
class SARSlidingSpotlightResult:
    """
    SARSlidingSpotlightCalculator 所有公开方法的统一返回结构。

    feasible=False 时，只有 reason 字段有意义，其余为默认值。

    与聚束模式的区别（新增字段）：
      - vrc_distance_m: 实际使用的 VRC 距离
      - effective_scene_velocity_m_s: 有效场景速度 V_eff
      - vrc_ratio: R/D_vrc 比值
      - peak_power_factor: 峰值功率因子 = 1/duty_cycle
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

    # 滑动聚束特有输出
    vrc_distance_m: float = 0.0            # VRC距离（m）
    effective_scene_velocity_m_s: float = 0.0  # 有效场景速度 V_eff（m/s）
    vrc_ratio: float = 0.0                 # R/D_vrc，0=条带，1=聚束
    peak_power_factor: float = 1.0         # 峰值功率因子 = 1/duty_cycle

    reason: Optional[str] = None           # 不可行时的原因描述
