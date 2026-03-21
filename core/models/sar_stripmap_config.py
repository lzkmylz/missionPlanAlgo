"""
SAR条带模式配置模块

支持三种建模方案：
  - continuous    (方案A): 连续角度范围 + 单一PRF
  - discrete_beam (方案B): 离散波位表，每个波位含完整参数
  - derived_beam  (方案C): 离散波位表，仅中心入射角，其余参数由公式推导

与聚束/滑动聚束的核心差异：
  - 波束固定指向（与卫星速度矢量夹角恒定）
  - 方位向分辨率固定为 L_a/2（与聚束模式相同，与成像时间无关）
  - 无"驻留时间"概念，成像时间 = 场景长度 / V_sat
  - 幅宽（Swath）由波束地面投影决定：W_g = (λ/L_w) × R / cos(θ)
  - PRF约束较宽松：仅需满足方位向多普勒带宽

关键物理参数：
  - duty_cycle: 发射占空比（影响峰值功率需求）
  - nominal_integration_time_s: 名义合成孔径时间（用于数据率估算）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StripmapBeamPosition:
    """
    条带模式的离散波位描述

    方案B (discrete_beam): 所有字段均需提供。
    方案C (derived_beam) : 仅 beam_id 和 center_incidence_angle_deg 必须提供，
                            其余字段在运行时由 SARStripmapCalculator 推导后填入。
    """
    beam_id: str
    center_incidence_angle_deg: float

    # 方案B需要显式提供；方案C由计算器填入（初始为None）
    incidence_angle_min_deg: Optional[float] = None
    incidence_angle_max_deg: Optional[float] = None
    prf_hz: Optional[float] = None
    range_resolution_m: Optional[float] = None
    azimuth_resolution_m: Optional[float] = None
    swath_width_km: Optional[float] = None  # 条带模式使用幅宽而非场景尺寸
    nominal_integration_time_s: Optional[float] = None  # 名义合成孔径时间

    def __post_init__(self):
        if self.incidence_angle_min_deg is not None and self.incidence_angle_max_deg is not None:
            if self.incidence_angle_min_deg >= self.incidence_angle_max_deg:
                raise ValueError(
                    f"StripmapBeamPosition '{self.beam_id}': "
                    f"incidence_angle_min ({self.incidence_angle_min_deg}) "
                    f"must be < incidence_angle_max ({self.incidence_angle_max_deg})"
                )
        if self.prf_hz is not None and self.prf_hz <= 0:
            raise ValueError(f"StripmapBeamPosition '{self.beam_id}': prf_hz must be > 0")
        if self.swath_width_km is not None and self.swath_width_km <= 0:
            raise ValueError(f"StripmapBeamPosition '{self.beam_id}': swath_width_km must be > 0")

    @property
    def is_fully_specified(self) -> bool:
        """方案B要求的所有字段均已填写"""
        return all(v is not None for v in [
            self.incidence_angle_min_deg, self.incidence_angle_max_deg,
            self.prf_hz, self.range_resolution_m, self.azimuth_resolution_m,
            self.swath_width_km, self.nominal_integration_time_s,
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
    def from_dict(cls, d: Dict[str, Any]) -> StripmapBeamPosition:
        return cls(
            beam_id=str(d["beam_id"]),
            center_incidence_angle_deg=float(d["center_incidence_angle_deg"]),
            incidence_angle_min_deg=float(d["incidence_angle_min_deg"]) if "incidence_angle_min_deg" in d else None,
            incidence_angle_max_deg=float(d["incidence_angle_max_deg"]) if "incidence_angle_max_deg" in d else None,
            prf_hz=float(d["prf_hz"]) if "prf_hz" in d else None,
            range_resolution_m=float(d["range_resolution_m"]) if "range_resolution_m" in d else None,
            azimuth_resolution_m=float(d["azimuth_resolution_m"]) if "azimuth_resolution_m" in d else None,
            swath_width_km=float(d["swath_width_km"]) if "swath_width_km" in d else None,
            nominal_integration_time_s=float(d["nominal_integration_time_s"]) if "nominal_integration_time_s" in d else None,
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
            "swath_width_km": self.swath_width_km,
            "nominal_integration_time_s": self.nominal_integration_time_s,
        }


@dataclass
class SARStripmapConfig:
    """
    SAR条带模式物理建模配置

    beam_model 字段区分三种方案：
      "continuous"    — 方案A：连续角度范围 + 单一PRF
      "discrete_beam" — 方案B：离散波位表（含完整参数）
      "derived_beam"  — 方案C：离散波位表（仅中心入射角，其余推导）

    核心区别（与聚束/滑动聚束）：
      - 无 VRC 距离参数
      - 方位向分辨率固定为 L_a/2
      - 使用幅宽（swath_width）而非场景尺寸
      - 成像时间由用户指定的场景长度决定
    """

    beam_model: str = "continuous"  # "continuous" | "discrete_beam" | "derived_beam"

    # ---- 系统级固定参数（三种方案共用） ----
    wavelength_m: float = 0.031           # 雷达波长（m），X波段≈0.031m
    antenna_length_m: float = 10.0        # 方位向天线长度（m）
    antenna_width_m: float = 2.0          # 距离向天线宽度（m）
    range_resolution_m: float = 3.0       # 距离向分辨率（m）

    # ---- 方案A/C共用字段 ----
    duty_cycle: float = 0.10              # 发射占空比（影响峰值功率需求）
    nominal_integration_time_s: float = 0.5  # 名义合成孔径时间（s）

    # ---- 方案A专用字段 ----
    prf_hz: float = 1500.0                # 标称工作PRF（Hz），条带模式PRF较低
    min_look_angle_deg: float = 15.0      # 最小侧视角（度）
    max_look_angle_deg: float = 50.0      # 最大侧视角（度）

    # ---- 方案C专用字段 ----
    prf_safety_factor: float = 1.25       # PRF过采样安全系数

    # ---- 方案B/C共用字段 ----
    beam_positions: List[StripmapBeamPosition] = field(default_factory=list)

    def __post_init__(self):
        valid_models = ("continuous", "discrete_beam", "derived_beam")
        if self.beam_model not in valid_models:
            raise ValueError(
                f"beam_model must be one of {valid_models}, got '{self.beam_model}'"
            )

        if self.duty_cycle <= 0 or self.duty_cycle > 1.0:
            raise ValueError(f"duty_cycle must be in (0, 1], got {self.duty_cycle}")

        if self.beam_model in ("discrete_beam", "derived_beam") and not self.beam_positions:
            raise ValueError(
                f"beam_model='{self.beam_model}' requires at least one StripmapBeamPosition entry"
            )

        if self.beam_model == "discrete_beam":
            for bp in self.beam_positions:
                if not bp.is_fully_specified:
                    raise ValueError(
                        f"discrete_beam mode: StripmapBeamPosition '{bp.beam_id}' is missing required fields. "
                        f"All fields must be provided for discrete_beam model."
                    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SARStripmapConfig:
        beam_positions = [
            StripmapBeamPosition.from_dict(bp) for bp in d.get("beam_positions", [])
        ]
        return cls(
            beam_model=d.get("beam_model", "continuous"),
            wavelength_m=float(d.get("wavelength_m", 0.031)),
            antenna_length_m=float(d.get("antenna_length_m", 10.0)),
            antenna_width_m=float(d.get("antenna_width_m", 2.0)),
            range_resolution_m=float(d.get("range_resolution_m", 3.0)),
            duty_cycle=float(d.get("duty_cycle", 0.10)),
            nominal_integration_time_s=float(d.get("nominal_integration_time_s", 0.5)),
            prf_hz=float(d.get("prf_hz", 1500.0)),
            min_look_angle_deg=float(d.get("min_look_angle_deg", 15.0)),
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
            "duty_cycle": self.duty_cycle,
            "nominal_integration_time_s": self.nominal_integration_time_s,
            "prf_hz": self.prf_hz,
            "min_look_angle_deg": self.min_look_angle_deg,
            "max_look_angle_deg": self.max_look_angle_deg,
            "prf_safety_factor": self.prf_safety_factor,
            "beam_positions": [bp.to_dict() for bp in self.beam_positions],
        }


@dataclass
class SARStripmapResult:
    """
    SARStripmapCalculator 所有公开方法的统一返回结构。

    feasible=False 时，只有 reason 字段有意义，其余为默认值。

    与聚束/滑动聚束的区别：
      - 无 dwell_time_s（条带模式无此概念）
      - 使用 swath_width_km（幅宽）而非 scene_size_rg_km
      - 成像时间由用户指定的场景长度决定
      - 方位向分辨率固定为 L_a/2
    """
    feasible: bool

    # 成像参数（条带模式使用用户指定的场景长度）
    imaging_time_s: float = 0.0           # 成像时间（秒）= 场景长度 / V_sat
    scene_length_along_track_km: float = 0.0  # 沿迹向场景长度（km）

    # 覆盖范围
    swath_width_km: float = 0.0           # 距离向幅宽（km）
    scene_area_km2: float = 0.0           # 覆盖面积（km²）= 场景长度 × 幅宽

    # 分辨率
    range_resolution_m: float = 0.0
    azimuth_resolution_m: float = 0.0     # 固定为 L_a/2

    # 诊断信息
    limiting_constraint: str = ""         # 通常为空或 "look_angle_range"

    matched_beam_id: Optional[str] = None  # 方案B/C匹配的波位ID
    prf_hz_used: float = 0.0               # 实际使用的PRF
    slant_range_m: float = 0.0             # 斜距（m）

    # 条带模式特有输出
    nominal_integration_time_s: float = 0.5  # 名义合成孔径时间
    peak_power_factor: float = 1.0         # 峰值功率因子 = 1/duty_cycle
    beam_ground_speed_m_s: float = 0.0     # 波束地面移动速度（≈ V_sat）

    reason: Optional[str] = None           # 不可行时的原因描述
