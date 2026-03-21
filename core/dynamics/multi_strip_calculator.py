"""
单次多条带拼幅成像动力学计算器

计算一次过境内多条带顺序成像的滚转角序列、时间线和地理覆盖。

算法说明:
- 条带间距 = strip_swath_width_m × (1 - overlap_ratio)（中心到中心）
- 滚转角 = arctan(cross_track_offset / orbit_altitude)（平地球近似，与footprint_calculator一致）
- 总时长 = N × 成像时长 + (N-1) × 机动时长
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import logging

from core.constants import EARTH_RADIUS_M

logger = logging.getLogger(__name__)


@dataclass
class StripSegment:
    """单条带成像段参数"""
    strip_index: int                         # 条带序号（0-based）
    roll_angle_deg: float                    # 该条带的滚转角（度）
    pitch_angle_deg: float = 0.0             # 该条带的俯仰角（度，标准推扫为0）
    imaging_start_offset_s: float = 0.0     # 相对任务开始的成像开始偏移（秒）
    imaging_end_offset_s: float = 0.0       # 相对任务开始的成像结束偏移（秒）
    slew_start_offset_s: Optional[float] = None  # 条带间机动开始时间偏移（秒）；第0条带无前置机动时为 None
    swath_center_lon: float = 0.0           # 条带中心经度（度）
    swath_center_lat: float = 0.0           # 条带中心纬度（度）
    swath_width_m: float = 0.0              # 单条带幅宽（米）
    coverage_area_km2: float = 0.0          # 该条带估算覆盖面积（km²）


@dataclass
class StripPlan:
    """多条带拼幅成像计划"""
    strips: List[StripSegment] = field(default_factory=list)  # 有序条带列表
    total_swath_width_m: float = 0.0        # 总幅宽（首条带左边缘至末条带右边缘，米）
    total_duration_s: float = 0.0           # 总任务时长（秒）
    roll_sequence_deg: List[float] = field(default_factory=list)  # 各条带滚转角序列
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


class MultiStripCalculator:
    """
    单次多条带拼幅成像计算器

    根据目标区域几何和卫星过境参数，规划条带滚转角序列和成像时间线。
    """

    # 地球半径（米）
    EARTH_RADIUS_M = EARTH_RADIUS_M

    def plan_strips(
        self,
        num_strips: int,
        strip_swath_width_m: float,
        overlap_ratio: float,
        inter_strip_slew_time_s: float,
        strip_imaging_duration_s: float,
        max_roll_angle_deg: float,
        max_roll_step_deg: float,
        max_total_roll_span_deg: float,
        center_roll_deg: float,
        orbit_altitude_m: float,
        window_duration_s: float,
        nadir_lon: float = 0.0,
        nadir_lat: float = 0.0,
        cross_track_direction_deg: float = 90.0,
        sat_velocity: Optional[Tuple[float, float, float]] = None,
    ) -> StripPlan:
        """
        规划多条带拼幅成像方案

        Args:
            num_strips: 条带数量
            strip_swath_width_m: 单条带幅宽（米）
            overlap_ratio: 条带重叠比例（0-0.3）
            inter_strip_slew_time_s: 条带间机动时间（秒）
            strip_imaging_duration_s: 单条带成像时长（秒）
            max_roll_angle_deg: 卫星最大允许滚转角（度）
            max_roll_step_deg: 相邻条带间最大滚转角变化量（度）
            max_total_roll_span_deg: 滚转角总跨度限制（度）
            center_roll_deg: 拼幅中心滚转偏置（度）
            orbit_altitude_m: 轨道高度（米）
            window_duration_s: 可见窗口时长（秒）
            nadir_lon: 星下点经度（度，用于计算条带地理位置）
            nadir_lat: 星下点纬度（度）
            cross_track_direction_deg: 垂轨方向方位角（度，90=东向）

        Returns:
            StripPlan：规划结果
        """
        # 步骤0：检查条带数量下限
        if num_strips < 2:
            return StripPlan(
                feasible=False,
                infeasibility_reason=(
                    f"num_strips must be >= 2 for mosaic mode, got {num_strips}"
                )
            )

        # 步骤1：计算所需总时长
        required_duration_s = (
            num_strips * strip_imaging_duration_s
            + (num_strips - 1) * inter_strip_slew_time_s
        )

        if window_duration_s < required_duration_s:
            return StripPlan(
                feasible=False,
                infeasibility_reason=(
                    f"Window duration {window_duration_s:.1f}s < required "
                    f"{required_duration_s:.1f}s "
                    f"({num_strips}×{strip_imaging_duration_s:.1f}s imaging + "
                    f"{num_strips - 1}×{inter_strip_slew_time_s:.1f}s slew)"
                )
            )

        # 步骤2：计算各条带滚转角
        roll_angles = self.calculate_strip_roll_angles(
            num_strips=num_strips,
            strip_swath_width_m=strip_swath_width_m,
            overlap_ratio=overlap_ratio,
            center_roll_deg=center_roll_deg,
            orbit_altitude_m=orbit_altitude_m,
        )

        # 步骤3：验证滚转角约束
        feasibility, reason = self._validate_roll_angles(
            roll_angles=roll_angles,
            max_roll_angle_deg=max_roll_angle_deg,
            max_roll_step_deg=max_roll_step_deg,
            max_total_roll_span_deg=max_total_roll_span_deg,
        )
        if not feasibility:
            return StripPlan(feasible=False, infeasibility_reason=reason)

        # 步骤4：构建成像时间线
        strips = self._build_strip_timeline(
            roll_angles=roll_angles,
            strip_swath_width_m=strip_swath_width_m,
            strip_imaging_duration_s=strip_imaging_duration_s,
            inter_strip_slew_time_s=inter_strip_slew_time_s,
            orbit_altitude_m=orbit_altitude_m,
            nadir_lon=nadir_lon,
            nadir_lat=nadir_lat,
            cross_track_direction_deg=cross_track_direction_deg,
            sat_velocity=sat_velocity,
        )

        # 步骤5：计算总幅宽
        # 公式: W × (1 + (N-1)×(1-overlap))，即首条带左边缘至末条带右边缘
        total_swath = strip_swath_width_m * (
            1.0 + (num_strips - 1) * (1.0 - overlap_ratio)
        )

        return StripPlan(
            strips=strips,
            total_swath_width_m=total_swath,
            total_duration_s=required_duration_s,
            roll_sequence_deg=roll_angles,
            feasible=True,
        )

    def calculate_strip_roll_angles(
        self,
        num_strips: int,
        strip_swath_width_m: float,
        overlap_ratio: float,
        center_roll_deg: float,
        orbit_altitude_m: float,
    ) -> List[float]:
        """
        计算各条带的滚转角序列

        条带从左到右排列，中心对准 center_roll_deg 对应的地面位置。
        相邻条带间距 = strip_swath_width_m × (1 - overlap_ratio)

        Args:
            num_strips: 条带数量
            strip_swath_width_m: 单条带幅宽（米）
            overlap_ratio: 重叠比例
            center_roll_deg: 中心滚转角偏置（度）
            orbit_altitude_m: 轨道高度（米）

        Returns:
            各条带滚转角列表（度），从左到右排列
        """
        # 条带有效间距（中心到中心，米）
        strip_spacing_m = strip_swath_width_m * (1.0 - overlap_ratio)

        # 计算各条带相对中心的偏移
        # 总范围 = (N-1) × spacing
        # 第i条带偏移 = (i - (N-1)/2) × spacing
        roll_angles = []
        for i in range(num_strips):
            # 相对中心条带的偏移（米，正值向右）
            offset_m = (i - (num_strips - 1) / 2.0) * strip_spacing_m

            # 将地面距离转换为滚转角（与footprint_calculator.py保持一致）
            # roll = atan(ground_distance / altitude)
            offset_roll_deg = math.degrees(math.atan2(offset_m, orbit_altitude_m))

            # 加上中心偏置
            roll_angles.append(center_roll_deg + offset_roll_deg)

        return roll_angles

    def check_window_duration_sufficient(
        self,
        window_duration_s: float,
        num_strips: int,
        strip_imaging_duration_s: float,
        inter_strip_slew_time_s: float,
    ) -> Tuple[bool, float]:
        """
        检查窗口时长是否足够

        Args:
            window_duration_s: 可用窗口时长（秒）
            num_strips: 条带数量
            strip_imaging_duration_s: 单条带成像时长（秒）
            inter_strip_slew_time_s: 条带间机动时间（秒）

        Returns:
            (是否足够, 所需最小时长（秒）)
        """
        required = (
            num_strips * strip_imaging_duration_s
            + (num_strips - 1) * inter_strip_slew_time_s
        )
        return window_duration_s >= required, required

    def _validate_roll_angles(
        self,
        roll_angles: List[float],
        max_roll_angle_deg: float,
        max_roll_step_deg: float,
        max_total_roll_span_deg: float,
    ) -> Tuple[bool, Optional[str]]:
        """验证滚转角序列是否满足所有约束"""
        if not roll_angles:
            return False, "Empty roll angle sequence"

        # 检查每个条带的绝对滚转角
        for i, roll in enumerate(roll_angles):
            if abs(roll) > max_roll_angle_deg:
                return False, (
                    f"Strip {i} roll angle {roll:.1f}° exceeds max "
                    f"allowed {max_roll_angle_deg:.1f}°"
                )

        # 检查相邻条带间滚转角变化量
        for i in range(1, len(roll_angles)):
            delta = abs(roll_angles[i] - roll_angles[i - 1])
            if delta > max_roll_step_deg:
                return False, (
                    f"Roll step between strip {i-1} and {i} is {delta:.1f}° > "
                    f"max allowed {max_roll_step_deg:.1f}°"
                )

        # 检查总滚转角跨度
        roll_span = max(roll_angles) - min(roll_angles)
        if roll_span > max_total_roll_span_deg:
            return False, (
                f"Total roll span {roll_span:.1f}° exceeds max allowed "
                f"{max_total_roll_span_deg:.1f}°"
            )

        return True, None

    def _build_strip_timeline(
        self,
        roll_angles: List[float],
        strip_swath_width_m: float,
        strip_imaging_duration_s: float,
        inter_strip_slew_time_s: float,
        orbit_altitude_m: float,
        nadir_lon: float,
        nadir_lat: float,
        cross_track_direction_deg: float,
        sat_velocity: Optional[Tuple[float, float, float]] = None,
    ) -> List[StripSegment]:
        """构建条带时间线和地理位置"""
        segments = []
        current_time = 0.0

        for i, roll_deg in enumerate(roll_angles):
            # 成像开始时间（第一条带为0，后续条带在机动结束后）
            if i > 0:
                slew_start = current_time
                current_time += inter_strip_slew_time_s
            else:
                slew_start = None  # 第0条带无前置机动

            imaging_start = current_time
            imaging_end = current_time + strip_imaging_duration_s
            current_time = imaging_end

            # 计算条带中心地理位置
            center_lon, center_lat = self._compute_strip_center(
                nadir_lon=nadir_lon,
                nadir_lat=nadir_lat,
                roll_deg=roll_deg,
                orbit_altitude_m=orbit_altitude_m,
                cross_track_direction_deg=cross_track_direction_deg,
            )

            # 估算覆盖面积（幅宽 × 沿轨成像距离）
            # 沿轨距离 ≈ 轨道速度 × 成像时长
            if sat_velocity is not None:
                _v_norm = math.sqrt(sum(v ** 2 for v in sat_velocity))
                orbit_velocity_m_s = max(6500.0, min(8000.0, _v_norm))
            else:
                orbit_velocity_m_s = 7500.0
            along_track_m = orbit_velocity_m_s * strip_imaging_duration_s
            coverage_km2 = (strip_swath_width_m / 1000.0) * (along_track_m / 1000.0)

            segments.append(StripSegment(
                strip_index=i,
                roll_angle_deg=roll_deg,
                pitch_angle_deg=0.0,
                imaging_start_offset_s=imaging_start,
                imaging_end_offset_s=imaging_end,
                slew_start_offset_s=slew_start,
                swath_center_lon=center_lon,
                swath_center_lat=center_lat,
                swath_width_m=strip_swath_width_m,
                coverage_area_km2=coverage_km2,
            ))

        return segments

    def _compute_strip_center(
        self,
        nadir_lon: float,
        nadir_lat: float,
        roll_deg: float,
        orbit_altitude_m: float,
        cross_track_direction_deg: float,
    ) -> Tuple[float, float]:
        """
        计算给定滚转角对应的地面条带中心坐标

        使用平地球近似（误差在低滚转角时<0.1%）:
        cross_track_offset_m = altitude * tan(roll_rad)
        然后将偏移投影到地理坐标系

        Args:
            nadir_lon/lat: 星下点坐标
            roll_deg: 滚转角（度）
            orbit_altitude_m: 轨道高度（米）
            cross_track_direction_deg: 垂轨方向方位角（度，90=正东）

        Returns:
            (center_lon, center_lat) in degrees
        """
        roll_rad = math.radians(roll_deg)
        cross_track_m = orbit_altitude_m * math.tan(roll_rad)

        # 将垂轨偏移转换为经纬度偏移
        direction_rad = math.radians(cross_track_direction_deg)
        delta_north_m = cross_track_m * math.cos(direction_rad)
        delta_east_m = cross_track_m * math.sin(direction_rad)

        # 将米偏移转换为度
        lat_rad = math.radians(nadir_lat)
        meters_per_deg_lat = self.EARTH_RADIUS_M * math.pi / 180.0
        meters_per_deg_lon = meters_per_deg_lat * math.cos(lat_rad)

        delta_lat_deg = delta_north_m / meters_per_deg_lat
        # Guard against near-polar singularity where meters_per_deg_lon → 0.
        # 1.0 m/deg corresponds to |lat| ≈ 89.999°, well past any useful SAR/optical observation angle.
        delta_lon_deg = (delta_east_m / meters_per_deg_lon) if abs(meters_per_deg_lon) > 1.0 else 0.0

        center_lon = nadir_lon + delta_lon_deg
        center_lat = nadir_lat + delta_lat_deg

        return center_lon, center_lat
