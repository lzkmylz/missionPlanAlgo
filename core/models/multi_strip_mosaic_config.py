"""
单次多条带拼幅成像（Single-Pass Multi-Strip Mosaic）配置模块

技术原理:
- 卫星在一次过境期间通过快速侧摆机动顺序成像多个相邻条带
- 各条带数据在轨或地面拼接形成超大幅宽影像
- 典型参数: 每10-20秒切换一个条带，覆盖45-75km总幅宽

关键约束:
- 条带间机动时间 >= settling_time（姿态稳定时间）
- 所有条带滚转角绝对值 <= max_roll_angle
- 总过境窗口时长 >= N×成像时间 + (N-1)×机动时间
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import math


@dataclass
class MultiStripMosaicConfig:
    """
    单次多条带拼幅成像配置

    Attributes:
        num_strips: 条带数量（2-8，默认3）
        strip_swath_width_m: 单条带幅宽（米，与单模式推扫幅宽相同，默认15000）
        overlap_ratio: 相邻条带重叠比例（0.05-0.20，默认0.10）
        inter_strip_slew_time_s: 条带间机动+稳定时间（秒，默认12.0）
        strip_imaging_duration_s: 单条带成像时长（秒，默认8.0）
        max_roll_step_deg: 相邻条带间最大滚转角变化量（度，默认20.0）
            必须 ≤ max_roll_rate × (inter_strip_slew_time_s - settling_time)，
            即物理可达的最大单次机动角度
        max_total_roll_span_deg: 第一条带到最后条带的最大滚转角跨度（度，默认50.0）
        center_roll_deg: 拼幅中心的滚转偏置（度，默认0.0即星下点对称）
        power_overhead_factor: 相比单条带推扫的功耗倍增因子（默认1.15）
    """
    num_strips: int = 3
    strip_swath_width_m: float = 15000.0
    overlap_ratio: float = 0.10
    inter_strip_slew_time_s: float = 12.0
    strip_imaging_duration_s: float = 8.0
    max_roll_step_deg: float = 20.0
    max_total_roll_span_deg: float = 50.0
    center_roll_deg: float = 0.0
    power_overhead_factor: float = 1.15

    def __post_init__(self):
        """验证参数有效性"""
        if not 2 <= self.num_strips <= 8:
            raise ValueError(
                f"num_strips must be in [2, 8], got {self.num_strips}"
            )
        if not 0.0 < self.strip_swath_width_m:
            raise ValueError(
                f"strip_swath_width_m must be positive, got {self.strip_swath_width_m}"
            )
        if not 0.0 <= self.overlap_ratio <= 0.30:
            raise ValueError(
                f"overlap_ratio must be in [0, 0.3], got {self.overlap_ratio}"
            )
        if self.inter_strip_slew_time_s <= 0:
            raise ValueError(
                f"inter_strip_slew_time_s must be positive, got {self.inter_strip_slew_time_s}"
            )
        if self.strip_imaging_duration_s <= 0:
            raise ValueError(
                f"strip_imaging_duration_s must be positive, got {self.strip_imaging_duration_s}"
            )
        if self.max_roll_step_deg <= 0:
            raise ValueError(
                f"max_roll_step_deg must be positive, got {self.max_roll_step_deg}"
            )
        if self.max_total_roll_span_deg <= 0:
            raise ValueError(
                f"max_total_roll_span_deg must be positive, got {self.max_total_roll_span_deg}"
            )
        if self.power_overhead_factor < 1.0:
            raise ValueError(
                f"power_overhead_factor must be >= 1.0, got {self.power_overhead_factor}"
            )
        if abs(self.center_roll_deg) > 60.0:
            raise ValueError(
                f"center_roll_deg must be in [-60, 60], got {self.center_roll_deg}"
            )

    def calculate_total_swath_width_m(self) -> float:
        """计算总幅宽（考虑重叠）

        公式: W_total = N * W_strip * (1 - overlap) + W_strip * overlap
              简化为: W_total = W_strip * (N * (1 - overlap) + overlap)
                             = W_strip * (1 + (N-1) * (1 - overlap))
        """
        return self.strip_swath_width_m * (1.0 + (self.num_strips - 1) * (1.0 - self.overlap_ratio))

    def calculate_total_mission_time_s(self) -> float:
        """计算完成全部条带成像所需的最短总时间（秒）

        公式: T_total = N * T_imaging + (N-1) * T_slew
        """
        return (self.num_strips * self.strip_imaging_duration_s
                + (self.num_strips - 1) * self.inter_strip_slew_time_s)

    def calculate_effective_inter_strip_spacing_m(self) -> float:
        """计算有效的条带间距（中心到中心，考虑重叠）

        公式: spacing = strip_swath_width_m * (1 - overlap_ratio)
        """
        return self.strip_swath_width_m * (1.0 - self.overlap_ratio)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'num_strips': self.num_strips,
            'strip_swath_width_m': self.strip_swath_width_m,
            'overlap_ratio': self.overlap_ratio,
            'inter_strip_slew_time_s': self.inter_strip_slew_time_s,
            'strip_imaging_duration_s': self.strip_imaging_duration_s,
            'max_roll_step_deg': self.max_roll_step_deg,
            'max_total_roll_span_deg': self.max_total_roll_span_deg,
            'center_roll_deg': self.center_roll_deg,
            'power_overhead_factor': self.power_overhead_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiStripMosaicConfig':
        """从字典创建"""
        return cls(
            num_strips=int(data.get('num_strips', 3)),
            strip_swath_width_m=float(data.get('strip_swath_width_m', 15000.0)),
            overlap_ratio=float(data.get('overlap_ratio', 0.10)),
            inter_strip_slew_time_s=float(data.get('inter_strip_slew_time_s', 12.0)),
            strip_imaging_duration_s=float(data.get('strip_imaging_duration_s', 8.0)),
            max_roll_step_deg=float(data.get('max_roll_step_deg', 20.0)),
            max_total_roll_span_deg=float(data.get('max_total_roll_span_deg', 50.0)),
            center_roll_deg=float(data.get('center_roll_deg', 0.0)),
            power_overhead_factor=float(data.get('power_overhead_factor', 1.15)),
        )


# 预定义配置模板

MOSAIC_CONFIG_3STRIP = MultiStripMosaicConfig(
    num_strips=3,
    strip_swath_width_m=15000.0,
    overlap_ratio=0.10,
    inter_strip_slew_time_s=12.0,
    strip_imaging_duration_s=8.0,
    max_roll_step_deg=20.0,   # ≤ 3°/s × (12-5)s = 21° 物理上限
    max_total_roll_span_deg=50.0,
)

MOSAIC_CONFIG_5STRIP = MultiStripMosaicConfig(
    num_strips=5,
    strip_swath_width_m=15000.0,
    overlap_ratio=0.10,
    inter_strip_slew_time_s=12.0,
    strip_imaging_duration_s=8.0,
    max_roll_step_deg=20.0,   # ≤ 3°/s × (12-5)s = 21° 物理上限
    max_total_roll_span_deg=60.0,
)

MOSAIC_CONFIG_TEMPLATES = {
    '3strip': MOSAIC_CONFIG_3STRIP,
    '5strip': MOSAIC_CONFIG_5STRIP,
}


def get_mosaic_config_template(template_name: str) -> Optional[MultiStripMosaicConfig]:
    """获取预定义的多条带拼幅配置模板"""
    return MOSAIC_CONFIG_TEMPLATES.get(template_name)
