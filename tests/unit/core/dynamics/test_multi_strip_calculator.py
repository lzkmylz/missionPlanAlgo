"""
单次多条带拼幅成像计算器单元测试

覆盖点：
  - calculate_strip_roll_angles：对称分布、偏置中心、单条带
  - plan_strips：名义成功（3条带、5条带）
  - plan_strips：窗口时长不足 → infeasible
  - plan_strips：滚转角超限 → infeasible
  - plan_strips：相邻条带滚转角变化量超限 → infeasible
  - plan_strips：总滚转角跨度超限 → infeasible
  - _validate_roll_angles：空序列
  - 总幅宽公式验证
"""

from __future__ import annotations

import math
import pytest

from core.dynamics.multi_strip_calculator import (
    MultiStripCalculator,
    StripPlan,
    StripSegment,
)


H = 500_000.0  # 轨道高度 500km（米）


class TestCalculateStripRollAngles:
    """calculate_strip_roll_angles 单元测试"""

    def setup_method(self):
        self.calc = MultiStripCalculator()

    def test_single_strip_returns_center_roll(self):
        angles = self.calc.calculate_strip_roll_angles(
            num_strips=1,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.0,
            center_roll_deg=5.0,
            orbit_altitude_m=H,
        )
        assert len(angles) == 1
        assert abs(angles[0] - 5.0) < 1e-9

    def test_three_strips_symmetric_around_center(self):
        """3条带，中心偏置为0时应关于0度对称"""
        angles = self.calc.calculate_strip_roll_angles(
            num_strips=3,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.10,
            center_roll_deg=0.0,
            orbit_altitude_m=H,
        )
        assert len(angles) == 3
        # 第0条带与第2条带应关于0对称
        assert abs(angles[0] + angles[2]) < 1e-9
        # 第1条带应约为0（中心条带）
        assert abs(angles[1]) < 1e-9

    def test_three_strips_with_center_offset(self):
        """加入中心偏置后，所有条带角度整体偏移"""
        offset = 10.0
        angles_no_offset = self.calc.calculate_strip_roll_angles(
            num_strips=3,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.10,
            center_roll_deg=0.0,
            orbit_altitude_m=H,
        )
        angles_with_offset = self.calc.calculate_strip_roll_angles(
            num_strips=3,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.10,
            center_roll_deg=offset,
            orbit_altitude_m=H,
        )
        for a, b in zip(angles_no_offset, angles_with_offset):
            assert abs((b - a) - offset) < 1e-9

    def test_angles_monotonically_increasing(self):
        """条带角度应从左到右单调递增（正间距）"""
        angles = self.calc.calculate_strip_roll_angles(
            num_strips=5,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.10,
            center_roll_deg=0.0,
            orbit_altitude_m=H,
        )
        for i in range(1, len(angles)):
            assert angles[i] > angles[i - 1]

    def test_ground_spacing_is_uniform(self):
        """各条带的地面偏移量（米）应均匀分布，等间距为 spacing_m"""
        swath = 15_000.0
        overlap = 0.10
        spacing = swath * (1.0 - overlap)

        angles = self.calc.calculate_strip_roll_angles(
            num_strips=4,
            strip_swath_width_m=swath,
            overlap_ratio=overlap,
            center_roll_deg=0.0,
            orbit_altitude_m=H,
        )
        # 从角度还原地面偏移：offset = H * tan(roll)
        offsets = [H * math.tan(math.radians(a)) for a in angles]
        # 相邻条带地面偏移差应均匀
        for i in range(1, len(offsets)):
            assert abs((offsets[i] - offsets[i - 1]) - spacing) < 1e-6


class TestPlanStripsNominal:
    """plan_strips 正常成功路径"""

    def setup_method(self):
        self.calc = MultiStripCalculator()

    def _make_plan(self, num_strips=3, center_roll=0.0, window_s=120.0):
        return self.calc.plan_strips(
            num_strips=num_strips,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.10,
            inter_strip_slew_time_s=12.0,
            strip_imaging_duration_s=8.0,
            max_roll_angle_deg=35.0,
            max_roll_step_deg=20.0,
            max_total_roll_span_deg=50.0,
            center_roll_deg=center_roll,
            orbit_altitude_m=H,
            window_duration_s=window_s,
        )

    def test_three_strip_plan_is_feasible(self):
        plan = self._make_plan(num_strips=3)
        assert plan.feasible is True
        assert plan.infeasibility_reason is None

    def test_three_strip_plan_has_correct_strip_count(self):
        plan = self._make_plan(num_strips=3)
        assert len(plan.strips) == 3

    def test_three_strip_total_duration(self):
        # 3×8 + 2×12 = 48s
        plan = self._make_plan(num_strips=3, window_s=120.0)
        assert abs(plan.total_duration_s - 48.0) < 1e-9

    def test_three_strip_total_swath_width(self):
        # W × (1 + (N-1)×(1-overlap)) = 15000 × (1 + 2×0.9) = 42000m
        plan = self._make_plan(num_strips=3)
        assert abs(plan.total_swath_width_m - 42_000.0) < 1e-6

    def test_five_strip_total_swath_width(self):
        # 15000 × (1 + 4×0.9) = 69000m
        plan = self._make_plan(num_strips=5, window_s=200.0)
        assert abs(plan.total_swath_width_m - 69_000.0) < 1e-6

    def test_roll_sequence_length_matches_strips(self):
        plan = self._make_plan(num_strips=3)
        assert len(plan.roll_sequence_deg) == 3

    def test_strip_imaging_timeline_non_overlapping(self):
        """各条带成像时间段不应重叠"""
        plan = self._make_plan(num_strips=3)
        for i in range(1, len(plan.strips)):
            prev = plan.strips[i - 1]
            curr = plan.strips[i]
            assert curr.imaging_start_offset_s >= prev.imaging_end_offset_s

    def test_strip_imaging_duration_correct(self):
        plan = self._make_plan(num_strips=3)
        for strip in plan.strips:
            duration = strip.imaging_end_offset_s - strip.imaging_start_offset_s
            assert abs(duration - 8.0) < 1e-9

    def test_first_strip_slew_start_is_none(self):
        """第0条带无前置机动，slew_start_offset_s 应为 None"""
        plan = self._make_plan(num_strips=3)
        assert plan.strips[0].slew_start_offset_s is None

    def test_subsequent_strips_slew_start_is_float(self):
        """第1+条带有前置机动，slew_start_offset_s 应为浮点数"""
        plan = self._make_plan(num_strips=3)
        for strip in plan.strips[1:]:
            assert strip.slew_start_offset_s is not None
            assert isinstance(strip.slew_start_offset_s, float)


class TestPlanStripsInfeasible:
    """plan_strips 不可行路径"""

    def setup_method(self):
        self.calc = MultiStripCalculator()

    def _base_kwargs(self, **overrides):
        base = dict(
            num_strips=3,
            strip_swath_width_m=15_000.0,
            overlap_ratio=0.10,
            inter_strip_slew_time_s=12.0,
            strip_imaging_duration_s=8.0,
            max_roll_angle_deg=35.0,
            max_roll_step_deg=20.0,
            max_total_roll_span_deg=50.0,
            center_roll_deg=0.0,
            orbit_altitude_m=H,
            window_duration_s=120.0,
        )
        base.update(overrides)
        return base

    def test_window_too_short_is_infeasible(self):
        # 需要48s，给40s
        plan = self.calc.plan_strips(**self._base_kwargs(window_duration_s=40.0))
        assert plan.feasible is False
        assert plan.infeasibility_reason is not None

    def test_roll_angle_exceeds_max_is_infeasible(self):
        # center_roll=30度，3条带跨度约≈ 2×atan(13500/500000)≈3.1°
        # 所有条带约在 28.5~31.5° 之间，设max_roll=25° → infeasible
        plan = self.calc.plan_strips(**self._base_kwargs(
            center_roll_deg=30.0,
            max_roll_angle_deg=25.0,
        ))
        assert plan.feasible is False

    def test_roll_step_exceeds_max_is_infeasible(self):
        # 正常3条带间距约 atan(13500/500000)≈1.55°，设 max_roll_step=0.5° → infeasible
        plan = self.calc.plan_strips(**self._base_kwargs(max_roll_step_deg=0.5))
        assert plan.feasible is False

    def test_total_roll_span_exceeds_max_is_infeasible(self):
        # 5条带跨度约 atan(4×13500/500000)×2 ≈ 6.1°；设 max_total_roll_span=1° → infeasible
        plan = self.calc.plan_strips(**self._base_kwargs(
            num_strips=5,
            max_total_roll_span_deg=1.0,
            window_duration_s=200.0,
        ))
        assert plan.feasible is False

    def test_infeasibility_reason_is_string(self):
        plan = self.calc.plan_strips(**self._base_kwargs(window_duration_s=10.0))
        assert isinstance(plan.infeasibility_reason, str)
        assert len(plan.infeasibility_reason) > 0

    def test_num_strips_one_is_infeasible(self):
        """num_strips=1 不是有效的拼幅模式，应返回 infeasible"""
        plan = self.calc.plan_strips(**self._base_kwargs(num_strips=1))
        assert plan.feasible is False
        assert plan.infeasibility_reason is not None
        assert 'num_strips' in plan.infeasibility_reason.lower() or '1' in plan.infeasibility_reason

    def test_num_strips_zero_is_infeasible(self):
        """num_strips=0 应返回 infeasible"""
        plan = self.calc.plan_strips(**self._base_kwargs(num_strips=0))
        assert plan.feasible is False


class TestValidateRollAngles:
    """_validate_roll_angles 边界测试"""

    def setup_method(self):
        self.calc = MultiStripCalculator()

    def test_empty_sequence_is_infeasible(self):
        ok, reason = self.calc._validate_roll_angles(
            roll_angles=[],
            max_roll_angle_deg=35.0,
            max_roll_step_deg=20.0,
            max_total_roll_span_deg=50.0,
        )
        assert ok is False
        assert reason is not None

    def test_single_valid_angle_passes(self):
        ok, reason = self.calc._validate_roll_angles(
            roll_angles=[5.0],
            max_roll_angle_deg=35.0,
            max_roll_step_deg=20.0,
            max_total_roll_span_deg=50.0,
        )
        assert ok is True
        assert reason is None

    def test_absolute_limit_violated(self):
        ok, reason = self.calc._validate_roll_angles(
            roll_angles=[0.0, 40.0],
            max_roll_angle_deg=35.0,
            max_roll_step_deg=50.0,
            max_total_roll_span_deg=60.0,
        )
        assert ok is False

    def test_step_limit_violated(self):
        ok, reason = self.calc._validate_roll_angles(
            roll_angles=[0.0, 5.0, 30.0],   # step 25° > limit 20°
            max_roll_angle_deg=35.0,
            max_roll_step_deg=20.0,
            max_total_roll_span_deg=60.0,
        )
        assert ok is False

    def test_span_limit_violated(self):
        ok, reason = self.calc._validate_roll_angles(
            roll_angles=[-15.0, 0.0, 15.0],  # span=30° > limit=20°
            max_roll_angle_deg=35.0,
            max_roll_step_deg=20.0,
            max_total_roll_span_deg=20.0,
        )
        assert ok is False
