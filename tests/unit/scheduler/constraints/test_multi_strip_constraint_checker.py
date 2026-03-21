"""
单次多条带拼幅成像约束检查器 + MultiStripMosaicConfig 单元测试

覆盖点：
  - MultiStripMosaicConfig 参数验证（7个检查分支）
  - MultiStripMosaicConfig.from_dict / to_dict 往返一致性（含 max_roll_step_deg 默认值回归）
  - MultiStripMosaicConfig 幅宽和时长计算方法
  - MultiStripConstraintChecker：卫星不支持该模式 → infeasible
  - MultiStripConstraintChecker：窗口时长不足 → infeasible
  - MultiStripConstraintChecker：机动时间 ≤ 稳定时间 → infeasible
  - MultiStripConstraintChecker：max_roll_step 超过物理上限 → infeasible
  - MultiStripConstraintChecker：名义3条带 → feasible，含条带规划
  - MultiStripConstraintChecker：批量检查与逐个结果一致
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.models.multi_strip_mosaic_config import MultiStripMosaicConfig
from core.models.imaging_mode import ImagingMode
from scheduler.constraints.multi_strip_constraint_checker import (
    MultiStripConstraintChecker,
    MultiStripCandidate,
    MultiStripConstraintResult,
)


# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------

def _make_capabilities(supports_mosaic: bool = True, max_roll: float = 35.0):
    """构造最小化的 SatelliteCapabilities 替代对象"""
    caps = MagicMock()
    caps.max_roll_angle = max_roll
    caps.agility = {
        'max_roll_rate': 3.0,
        'max_slew_rate': 3.0,
        'settling_time': 5.0,
    }
    if supports_mosaic:
        caps.imaging_modes = [ImagingMode.SINGLE_PASS_MOSAIC]
    else:
        caps.imaging_modes = [ImagingMode.PUSH_BROOM]
    caps.supports_single_pass_mosaic.return_value = supports_mosaic
    return caps


def _make_satellite(supports_mosaic: bool = True, max_roll: float = 35.0):
    sat = MagicMock()
    sat.id = 'SAT-TEST'
    sat.capabilities = _make_capabilities(supports_mosaic, max_roll)
    sat.orbit.semi_major_axis = 6_871_000.0   # 500km 轨道
    # payload_config=None 会触发 except (AttributeError, ValueError) → fallback to 172.5W
    sat.payload_config = None
    return sat


def _make_target(lon: float = 116.4, lat: float = 39.9, center_roll: float = 0.0):
    tgt = MagicMock()
    tgt.id = 'TGT-TEST'
    tgt.get_center.return_value = (lon, lat)
    tgt.mosaic_center_roll_deg = center_roll
    return tgt


def _make_config(**kwargs) -> MultiStripMosaicConfig:
    defaults = dict(
        num_strips=3,
        strip_swath_width_m=15_000.0,
        overlap_ratio=0.10,
        inter_strip_slew_time_s=12.0,
        strip_imaging_duration_s=8.0,
        max_roll_step_deg=20.0,
        max_total_roll_span_deg=50.0,
        center_roll_deg=0.0,
        power_overhead_factor=1.15,
    )
    defaults.update(kwargs)
    return MultiStripMosaicConfig(**defaults)


def _make_candidate(
    satellite=None,
    target=None,
    mosaic_config=None,
    window_s: float = 120.0,
) -> MultiStripCandidate:
    if satellite is None:
        satellite = _make_satellite()
    if target is None:
        target = _make_target()
    if mosaic_config is None:
        mosaic_config = _make_config()
    now = datetime(2026, 3, 21, 0, 0, 0)
    return MultiStripCandidate(
        sat_id='SAT-TEST',
        satellite=satellite,
        target=target,
        window_start=now,
        window_end=now + timedelta(seconds=window_s),
        mosaic_config=mosaic_config,
    )


# ===========================================================================
# MultiStripMosaicConfig 测试
# ===========================================================================

class TestMultiStripMosaicConfigValidation:

    def test_valid_default_construction(self):
        cfg = MultiStripMosaicConfig()
        assert cfg.num_strips == 3
        assert cfg.max_roll_step_deg == 20.0

    def test_num_strips_below_min_raises(self):
        with pytest.raises(ValueError, match="num_strips"):
            MultiStripMosaicConfig(num_strips=1)

    def test_num_strips_above_max_raises(self):
        with pytest.raises(ValueError, match="num_strips"):
            MultiStripMosaicConfig(num_strips=9)

    def test_negative_swath_width_raises(self):
        with pytest.raises(ValueError, match="strip_swath_width_m"):
            MultiStripMosaicConfig(strip_swath_width_m=-1.0)

    def test_overlap_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError, match="overlap_ratio"):
            MultiStripMosaicConfig(overlap_ratio=0.5)

    def test_non_positive_slew_time_raises(self):
        with pytest.raises(ValueError, match="inter_strip_slew_time_s"):
            MultiStripMosaicConfig(inter_strip_slew_time_s=0.0)

    def test_non_positive_imaging_duration_raises(self):
        with pytest.raises(ValueError, match="strip_imaging_duration_s"):
            MultiStripMosaicConfig(strip_imaging_duration_s=-1.0)

    def test_non_positive_max_roll_step_raises(self):
        with pytest.raises(ValueError, match="max_roll_step_deg"):
            MultiStripMosaicConfig(max_roll_step_deg=0.0)

    def test_power_overhead_below_one_raises(self):
        with pytest.raises(ValueError, match="power_overhead_factor"):
            MultiStripMosaicConfig(power_overhead_factor=0.9)


class TestMultiStripMosaicConfigComputations:

    def test_total_swath_width_3strip(self):
        cfg = _make_config(num_strips=3, strip_swath_width_m=15_000.0, overlap_ratio=0.10)
        # 15000 × (1 + 2×0.9) = 42000m
        assert abs(cfg.calculate_total_swath_width_m() - 42_000.0) < 1e-6

    def test_total_swath_width_5strip(self):
        cfg = _make_config(num_strips=5, strip_swath_width_m=15_000.0, overlap_ratio=0.10)
        # 15000 × (1 + 4×0.9) = 69000m
        assert abs(cfg.calculate_total_swath_width_m() - 69_000.0) < 1e-6

    def test_total_mission_time_3strip(self):
        cfg = _make_config(num_strips=3, strip_imaging_duration_s=8.0, inter_strip_slew_time_s=12.0)
        # 3×8 + 2×12 = 48s
        assert abs(cfg.calculate_total_mission_time_s() - 48.0) < 1e-9

    def test_effective_inter_strip_spacing(self):
        cfg = _make_config(strip_swath_width_m=15_000.0, overlap_ratio=0.10)
        # 15000 × 0.9 = 13500m
        assert abs(cfg.calculate_effective_inter_strip_spacing_m() - 13_500.0) < 1e-6


class TestMultiStripMosaicConfigRoundTrip:

    def test_to_dict_from_dict_idempotent(self):
        cfg = _make_config(max_roll_step_deg=20.0)
        d = cfg.to_dict()
        cfg2 = MultiStripMosaicConfig.from_dict(d)
        assert cfg2.num_strips == cfg.num_strips
        assert abs(cfg2.max_roll_step_deg - cfg.max_roll_step_deg) < 1e-9

    def test_from_dict_default_max_roll_step_is_20(self):
        """MEDIUM-1 回归：from_dict 默认值必须与 dataclass 字段默认值一致（20.0，非30.0）"""
        cfg = MultiStripMosaicConfig.from_dict({})
        assert abs(cfg.max_roll_step_deg - 20.0) < 1e-9

    def test_from_dict_respects_explicit_value(self):
        cfg = MultiStripMosaicConfig.from_dict({'max_roll_step_deg': 15.0})
        assert abs(cfg.max_roll_step_deg - 15.0) < 1e-9


# ===========================================================================
# MultiStripConstraintChecker 测试
# ===========================================================================

class TestMultiStripConstraintCheckerFeasible:

    def setup_method(self):
        self.checker = MultiStripConstraintChecker()

    def test_nominal_3strip_is_feasible(self):
        candidate = _make_candidate(window_s=120.0)
        result = self.checker.check_feasibility(candidate)
        assert result.feasible is True
        assert result.strip_plan is not None
        assert result.reason is None

    def test_nominal_result_has_correct_strip_count(self):
        candidate = _make_candidate(window_s=120.0)
        result = self.checker.check_feasibility(candidate)
        assert len(result.strip_plan.strips) == 3

    def test_nominal_total_swath_matches_formula(self):
        candidate = _make_candidate(window_s=120.0)
        result = self.checker.check_feasibility(candidate)
        # 15000 × (1 + 2×0.9) = 42000m
        assert abs(result.strip_plan.total_swath_width_m - 42_000.0) < 1e-6

    def test_nominal_energy_is_positive(self):
        candidate = _make_candidate(window_s=120.0)
        result = self.checker.check_feasibility(candidate)
        assert result.estimated_energy_wh > 0.0

    def test_energy_no_double_overhead(self):
        """HIGH-fix 回归：energy 不应对机动阶段再次乘以 power_overhead_factor"""
        candidate = _make_candidate(window_s=120.0)
        # 手动计算期望值
        cfg = candidate.mosaic_config
        base_power_w = 172.5  # fallback 值（mock satellite 无 payload_config）
        total_s = cfg.num_strips * cfg.strip_imaging_duration_s + (cfg.num_strips - 1) * cfg.inter_strip_slew_time_s
        expected_wh = base_power_w * total_s / 3600.0
        result = self.checker.check_feasibility(candidate)
        assert abs(result.estimated_energy_wh - expected_wh) < 0.01


class TestMultiStripConstraintCheckerInfeasible:

    def setup_method(self):
        self.checker = MultiStripConstraintChecker()

    def test_satellite_not_support_mosaic_is_infeasible(self):
        sat = _make_satellite(supports_mosaic=False)
        candidate = _make_candidate(satellite=sat, window_s=120.0)
        result = self.checker.check_feasibility(candidate)
        assert result.feasible is False
        assert result.reason is not None

    def test_window_too_short_is_infeasible(self):
        # 需要48s，只给40s
        candidate = _make_candidate(window_s=40.0)
        result = self.checker.check_feasibility(candidate)
        assert result.feasible is False

    def test_slew_time_le_settling_time_is_infeasible(self):
        # inter_strip_slew_time_s = settling_time → available_slew_time ≤ 0
        cfg = _make_config(inter_strip_slew_time_s=5.0)  # settling_time=5.0
        candidate = _make_candidate(mosaic_config=cfg, window_s=200.0)
        result = self.checker.check_feasibility(candidate)
        assert result.feasible is False
        assert len(result.slew_violations) > 0

    def test_roll_step_exceeds_achievable_is_infeasible(self):
        # available = 12-5=7s；max_rate=3°/s → achievable=21°
        # 设 max_roll_step=25° > 21° → infeasible
        cfg = _make_config(max_roll_step_deg=25.0)
        candidate = _make_candidate(mosaic_config=cfg, window_s=120.0)
        result = self.checker.check_feasibility(candidate)
        assert result.feasible is False

    def test_infeasible_reason_is_string(self):
        candidate = _make_candidate(window_s=10.0)
        result = self.checker.check_feasibility(candidate)
        assert result.feasible is False
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0


class TestMultiStripConstraintCheckerBatch:

    def setup_method(self):
        self.checker = MultiStripConstraintChecker()

    def test_batch_length_matches_input(self):
        candidates = [_make_candidate(window_s=120.0) for _ in range(5)]
        results = self.checker.check_feasibility_batch(candidates)
        assert len(results) == 5

    def test_batch_matches_individual_results(self):
        candidates = [
            _make_candidate(window_s=120.0),   # feasible
            _make_candidate(window_s=10.0),    # infeasible (window too short)
        ]
        batch_results = self.checker.check_feasibility_batch(candidates)
        for c, br in zip(candidates, batch_results):
            individual = self.checker.check_feasibility(c)
            assert br.feasible == individual.feasible

    def test_empty_batch_returns_empty_list(self):
        results = self.checker.check_feasibility_batch([])
        assert results == []


class TestMultiStripConstraintCheckerStats:

    def test_stats_increment_on_each_call(self):
        checker = MultiStripConstraintChecker()
        assert checker.get_stats()['total_checks'] == 0
        checker.check_feasibility(_make_candidate(window_s=120.0))
        assert checker.get_stats()['total_checks'] == 1
        checker.check_feasibility(_make_candidate(window_s=10.0))
        assert checker.get_stats()['total_checks'] == 2

    def test_feasible_count_tracked(self):
        checker = MultiStripConstraintChecker()
        checker.check_feasibility(_make_candidate(window_s=120.0))  # feasible
        checker.check_feasibility(_make_candidate(window_s=10.0))   # infeasible
        assert checker.get_stats()['feasible_count'] == 1
        assert checker.get_stats()['infeasible_count'] == 1
