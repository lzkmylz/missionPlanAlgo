"""
SAR ScanSAR模式物理引擎单元测试

覆盖点：
  - ScanSARSubSwathPosition / SARScanSARConfig dataclass 验证
  - 三种 beam_model 计算路径（continuous / discrete_beam / derived_beam）
  - 扇贝效应计算（3种窗口函数）
  - ISLR退化量计算
  - 负侧视角 / 超范围侧视角拒绝
  - select_subswath_position 匹配逻辑
  - 子条带数超出 num_subswaths 时发出 UserWarning
  - PayloadConfiguration 加载解析（sar_1.json 端到端）
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[4]

from core.models.sar_scansar_config import (
    ScanSARSubSwathPosition,
    SARScanSARConfig,
    SARScanSARResult,
    _SUBSWATH_ANGLE_TOLERANCE_DEG,
)
from core.dynamics.sar_scansar_calculator import (
    SARScanSARCalculator,
    _compute_scalloping,
    _compute_islr_degradation,
)

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
H = 631_000.0   # m，典型LEO轨道高度
V = 7500.0      # m/s，卫星速度


# ---------------------------------------------------------------------------
# ScanSARSubSwathPosition 测试
# ---------------------------------------------------------------------------

class TestScanSARSubSwathPosition:
    def test_minimal_construction(self):
        sp = ScanSARSubSwathPosition(subswath_id="SW1", center_incidence_angle_deg=30.0)
        assert sp.subswath_id == "SW1"
        assert sp.center_incidence_angle_deg == 30.0
        assert not sp.is_fully_specified

    def test_full_construction(self):
        sp = ScanSARSubSwathPosition(
            subswath_id="SW1",
            center_incidence_angle_deg=30.0,
            incidence_angle_min_deg=25.0,
            incidence_angle_max_deg=35.0,
            prf_hz=1500.0,
            range_resolution_m=15.0,
            azimuth_resolution_m=15.0,
            swath_width_rg_km=60.0,
            burst_duration_s=0.05,
        )
        assert sp.is_fully_specified

    def test_min_geq_max_raises(self):
        with pytest.raises(ValueError, match="incidence_angle_min"):
            ScanSARSubSwathPosition(
                subswath_id="X",
                center_incidence_angle_deg=30.0,
                incidence_angle_min_deg=40.0,
                incidence_angle_max_deg=35.0,
            )

    def test_negative_scalloping_raises(self):
        with pytest.raises(ValueError, match="peak_scalloping_db"):
            ScanSARSubSwathPosition(
                subswath_id="X",
                center_incidence_angle_deg=30.0,
                peak_scalloping_db=-1.0,
            )

    def test_covers_within_tolerance(self):
        sp = ScanSARSubSwathPosition(
            subswath_id="SW1",
            center_incidence_angle_deg=30.0,
            incidence_angle_min_deg=25.0,
            incidence_angle_max_deg=35.0,
        )
        assert sp.covers(30.0)
        assert sp.covers(25.0 - _SUBSWATH_ANGLE_TOLERANCE_DEG / 2)  # 容差范围内
        assert not sp.covers(20.0)  # 超出范围

    def test_covers_none_bounds_returns_false(self):
        sp = ScanSARSubSwathPosition(subswath_id="SW1", center_incidence_angle_deg=30.0)
        assert not sp.covers(30.0)

    def test_from_dict_roundtrip(self):
        d = {
            "subswath_id": "SW2",
            "center_incidence_angle_deg": 35.0,
            "incidence_angle_min_deg": 30.0,
            "incidence_angle_max_deg": 40.0,
            "prf_hz": 1800.0,
            "range_resolution_m": 12.0,
            "azimuth_resolution_m": 14.0,
            "swath_width_rg_km": 55.0,
            "burst_duration_s": 0.05,
            "peak_scalloping_db": 3.92,
            "snr_variation_db": 3.92,
        }
        sp = ScanSARSubSwathPosition.from_dict(d)
        assert sp.subswath_id == "SW2"
        assert sp.prf_hz == pytest.approx(1800.0)
        assert sp.is_fully_specified
        # to_dict 往返
        d2 = sp.to_dict()
        assert d2["subswath_id"] == "SW2"
        assert d2["peak_scalloping_db"] == pytest.approx(3.92)


# ---------------------------------------------------------------------------
# SARScanSARConfig 测试
# ---------------------------------------------------------------------------

class TestSARScanSARConfig:
    def test_default_construction(self):
        cfg = SARScanSARConfig()
        assert cfg.beam_model == "continuous"
        assert cfg.num_subswaths == 5
        assert cfg.burst_duration_s == pytest.approx(0.05)
        assert cfg.enable_scalloping_model is True
        assert cfg.scalloping_window == "rectangular"

    def test_invalid_beam_model_raises(self):
        with pytest.raises(ValueError, match="beam_model"):
            SARScanSARConfig(beam_model="unknown")

    def test_invalid_scalloping_window_raises(self):
        with pytest.raises(ValueError, match="scalloping_window"):
            SARScanSARConfig(scalloping_window="blackman")

    def test_zero_burst_duration_raises(self):
        with pytest.raises(ValueError, match="burst_duration_s"):
            SARScanSARConfig(burst_duration_s=0.0)

    def test_discrete_beam_requires_positions(self):
        with pytest.raises(ValueError, match="requires at least one"):
            SARScanSARConfig(beam_model="discrete_beam", num_subswaths=1)

    def test_discrete_beam_too_few_positions_raises(self):
        pos = [ScanSARSubSwathPosition(
            subswath_id="SW1", center_incidence_angle_deg=30.0,
            incidence_angle_min_deg=25.0, incidence_angle_max_deg=35.0,
            prf_hz=1500.0, range_resolution_m=15.0, azimuth_resolution_m=15.0,
            swath_width_rg_km=60.0, burst_duration_s=0.05,
        )]
        with pytest.raises(ValueError, match="num_subswaths.*exceeds"):
            SARScanSARConfig(beam_model="discrete_beam", num_subswaths=3, subswath_positions=pos)

    def test_excess_positions_warns(self):
        pos = [
            ScanSARSubSwathPosition(subswath_id=f"SW{i+1}", center_incidence_angle_deg=25.0 + i * 7.0)
            for i in range(4)
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SARScanSARConfig(beam_model="derived_beam", num_subswaths=2, subswath_positions=pos)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "discarded" in str(w[0].message).lower()

    def test_from_dict_roundtrip(self):
        d = {
            "beam_model": "continuous",
            "wavelength_m": 0.031,
            "antenna_length_m": 6.0,
            "antenna_width_m": 2.0,
            "range_resolution_m": 15.0,
            "num_subswaths": 5,
            "burst_duration_s": 0.05,
            "burst_switch_time_s": 0.002,
            "duty_cycle": 0.1,
            "prf_hz": 1500.0,
            "center_look_angle_deg": 35.0,
            "subswath_spacing_deg": 1.5,
            "min_look_angle_deg": 15.0,
            "max_look_angle_deg": 60.0,
            "prf_safety_factor": 1.25,
            "subswath_positions": [],
            "enable_scalloping_model": True,
            "scalloping_window": "rectangular",
        }
        cfg = SARScanSARConfig.from_dict(d)
        assert cfg.beam_model == "continuous"
        assert cfg.num_subswaths == 5


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------

class TestScallopingFunctions:
    def test_rectangular_peak(self):
        peak, mean, snr_var = _compute_scalloping("rectangular")
        # gain at edge = sinc(±1/2) = 2/π → S_peak = 20·log10(π/2) ≈ 3.92 dB
        assert peak == pytest.approx(3.922, abs=0.01)
        # E[gain] ≈ √(∫sinc²) ≈ 0.879 → mean ≈ -20·log10(0.879) ≈ 1.11 dB
        assert mean == pytest.approx(1.11, abs=0.05)
        assert snr_var == pytest.approx(peak, abs=0.01)

    def test_hamming_peak_less_than_rectangular(self):
        peak_rect, _, _ = _compute_scalloping("rectangular")
        peak_hamm, _, _ = _compute_scalloping("hamming")
        assert peak_hamm < peak_rect

    def test_hanning_peak_between_hamming_and_rectangular(self):
        peak_hamm, _, _ = _compute_scalloping("hamming")
        peak_hann, _, _ = _compute_scalloping("hanning")
        peak_rect, _, _ = _compute_scalloping("rectangular")
        assert peak_hamm < peak_hann < peak_rect

    def test_islr_degradation_rectangular(self):
        assert _compute_islr_degradation("rectangular") == pytest.approx(3.0, abs=0.1)

    def test_islr_degradation_hamming_less_than_rect(self):
        assert _compute_islr_degradation("hamming") < _compute_islr_degradation("rectangular")


# ---------------------------------------------------------------------------
# SARScanSARCalculator — continuous 路径（方案A）
# ---------------------------------------------------------------------------

class TestContinuousPath:
    def _make_calc(self, **kwargs) -> SARScanSARCalculator:
        cfg = SARScanSARConfig(beam_model="continuous", **kwargs)
        return SARScanSARCalculator(cfg)

    def test_valid_35deg(self):
        calc = self._make_calc()
        r = calc.compute_burst_params(H, 35.0, V)
        assert r.feasible
        assert r.num_subswaths_used == 5
        assert r.total_swath_width_km > 0
        assert r.peak_scalloping_db == pytest.approx(3.922, abs=0.02)
        assert r.cycle_time_s == pytest.approx(5 * 0.05 + 4 * 0.002, abs=1e-6)

    def test_out_of_range_look_angle_min(self):
        calc = self._make_calc(min_look_angle_deg=20.0)
        r = calc.compute_burst_params(H, 10.0, V)
        assert not r.feasible

    def test_out_of_range_look_angle_max(self):
        calc = self._make_calc(max_look_angle_deg=55.0)
        r = calc.compute_burst_params(H, 60.0, V)
        assert not r.feasible

    def test_negative_subswath_look_angle_rejected(self):
        # center=5°, spacing=3°, N=5 → min_look_i = 5 - 2*3 = -1° ≤ 0 → infeasible
        calc = self._make_calc(
            min_look_angle_deg=1.0,
            max_look_angle_deg=60.0,
            center_look_angle_deg=5.0,
            subswath_spacing_deg=3.0,
            num_subswaths=5,
        )
        r = calc.compute_burst_params(H, 5.0, V)
        assert not r.feasible
        assert "子条带间距" in r.reason or "侧视角" in r.reason

    def test_scalloping_disabled(self):
        calc = self._make_calc(enable_scalloping_model=False)
        r = calc.compute_burst_params(H, 35.0, V)
        assert r.feasible
        assert r.peak_scalloping_db == pytest.approx(0.0)
        assert r.mean_scalloping_db == pytest.approx(0.0)

    def test_azimuth_resolution_decreases_with_longer_burst(self):
        calc_short = self._make_calc(burst_duration_s=0.04)
        calc_long = self._make_calc(burst_duration_s=0.08)
        r_short = calc_short.compute_burst_params(H, 35.0, V)
        r_long = calc_long.compute_burst_params(H, 35.0, V)
        # 更长的burst → 更细的方位向分辨率
        assert r_long.azimuth_resolution_m < r_short.azimuth_resolution_m

    def test_total_swath_scales_with_num_subswaths(self):
        calc5 = self._make_calc(num_subswaths=5)
        calc3 = self._make_calc(num_subswaths=3)
        r5 = calc5.compute_burst_params(H, 35.0, V)
        r3 = calc3.compute_burst_params(H, 35.0, V)
        assert r5.total_swath_width_km == pytest.approx(r3.total_swath_width_km * 5 / 3, rel=0.01)


# ---------------------------------------------------------------------------
# SARScanSARCalculator — discrete_beam 路径（方案B）
# ---------------------------------------------------------------------------

class TestDiscreteBeamPath:
    def _make_subswaths(self):
        return [
            ScanSARSubSwathPosition(
                subswath_id=f"SW{i+1}",
                center_incidence_angle_deg=22.0 + i * 8.0,
                incidence_angle_min_deg=18.0 + i * 8.0,
                incidence_angle_max_deg=26.0 + i * 8.0,
                prf_hz=1500.0,
                range_resolution_m=15.0,
                azimuth_resolution_m=15.0,
                swath_width_rg_km=60.0,
                burst_duration_s=0.05,
            )
            for i in range(5)
        ]

    def _make_calc(self) -> SARScanSARCalculator:
        cfg = SARScanSARConfig(
            beam_model="discrete_beam",
            num_subswaths=5,
            subswath_positions=self._make_subswaths(),
        )
        return SARScanSARCalculator(cfg)

    def test_feasible_center_subswath(self):
        calc = self._make_calc()
        # SW3: incidence 26°-42°，look ≈ 35°
        r = calc.compute_burst_params(H, 35.0, V)
        assert r.feasible
        assert r.matched_subswath_id is not None
        assert r.peak_scalloping_db == pytest.approx(3.922, abs=0.02)

    def test_invalid_look_angle_rejected(self):
        calc = self._make_calc()
        r = calc.compute_burst_params(H, 0.0, V)
        assert not r.feasible

    def test_negative_look_angle_rejected(self):
        calc = self._make_calc()
        r = calc.compute_burst_params(H, -5.0, V)
        assert not r.feasible

    def test_no_matching_subswath(self):
        calc = self._make_calc()
        # 70° 入射角超出所有子条带范围（最大 SW5: 26°-42° → max ~50°入射）
        r = calc.compute_burst_params(H, 70.0, V)
        assert not r.feasible


# ---------------------------------------------------------------------------
# SARScanSARCalculator — derived_beam 路径（方案C）
# ---------------------------------------------------------------------------

class TestDerivedBeamPath:
    def _make_calc(self, n=5, burst=0.05) -> SARScanSARCalculator:
        subswaths = [
            ScanSARSubSwathPosition(subswath_id=f"SW{i+1}", center_incidence_angle_deg=20.0 + i * 7.0)
            for i in range(n)
        ]
        cfg = SARScanSARConfig(
            beam_model="derived_beam",
            num_subswaths=n,
            burst_duration_s=burst,
            subswath_positions=subswaths,
        )
        return SARScanSARCalculator(cfg)

    def test_feasible_center_angle(self):
        calc = self._make_calc()
        r = calc.compute_burst_params(H, 35.0, V)
        assert r.feasible
        assert r.peak_scalloping_db == pytest.approx(3.922, abs=0.02)
        assert r.matched_subswath_id is not None

    def test_invalid_zero_look_angle_rejected(self):
        calc = self._make_calc()
        r = calc.compute_burst_params(H, 0.0, V)
        assert not r.feasible

    def test_invalid_negative_look_angle_rejected(self):
        calc = self._make_calc()
        r = calc.compute_burst_params(H, -10.0, V)
        assert not r.feasible

    def test_subswath_params_backfilled(self):
        calc = self._make_calc()
        calc.compute_burst_params(H, 35.0, V)
        # 推导后，中心子条带的参数应已回填
        center_sw = calc.config.subswath_positions[2]  # SW3, 34°
        assert center_sw.prf_hz is not None and center_sw.prf_hz > 0
        assert center_sw.azimuth_resolution_m is not None and center_sw.azimuth_resolution_m > 0

    def test_hamming_window_lower_scalloping(self):
        subswaths = [
            ScanSARSubSwathPosition(subswath_id=f"SW{i+1}", center_incidence_angle_deg=20.0 + i * 7.0)
            for i in range(5)
        ]
        cfg = SARScanSARConfig(
            beam_model="derived_beam",
            num_subswaths=5,
            subswath_positions=subswaths,
            scalloping_window="hamming",
        )
        calc = SARScanSARCalculator(cfg)
        r = calc.compute_burst_params(H, 35.0, V)
        assert r.feasible
        assert r.peak_scalloping_db < 3.92  # hamming << rectangular


# ---------------------------------------------------------------------------
# select_subswath_position 测试
# ---------------------------------------------------------------------------

class TestSelectSubswathPosition:
    def _make_derived_calc(self) -> SARScanSARCalculator:
        subswaths = [
            ScanSARSubSwathPosition(subswath_id=f"SW{i+1}", center_incidence_angle_deg=20.0 + i * 7.0)
            for i in range(5)
        ]
        cfg = SARScanSARConfig(beam_model="derived_beam", num_subswaths=5, subswath_positions=subswaths)
        return SARScanSARCalculator(cfg)

    def test_continuous_returns_none(self):
        calc = SARScanSARCalculator(SARScanSARConfig(beam_model="continuous"))
        assert calc.select_subswath_position(35.0) is None

    def test_derived_selects_nearest_center(self):
        calc = self._make_derived_calc()
        # SW3 中心 34°，SW4 中心 41° → 对于 37°，最近应是 SW3（距离3°）还是 SW4（距离4°）
        sw = calc.select_subswath_position(37.0)
        assert sw is not None
        assert sw.subswath_id == "SW3"

    def test_derived_no_match_returns_none(self):
        calc = self._make_derived_calc()
        # 80° 超出所有子条带中心（max SW5 = 48°），且无 incidence_angle_min/max → 退化为最近中心
        # derived_beam 按最近中心选，不会返回 None；只有 discrete_beam 的 covers() 才会
        sw = calc.select_subswath_position(80.0)
        # derived_beam 总是选最近的，不返回 None
        assert sw is not None


# ---------------------------------------------------------------------------
# PayloadConfiguration 端到端解析测试（sar_1.json）
# ---------------------------------------------------------------------------

class TestPayloadConfigIntegration:
    def _load_sar1_payload(self):
        json_path = PROJECT_ROOT / "data" / "entity_lib" / "satellites" / "sar_1.json"
        with open(json_path) as f:
            data = json.load(f)
        payload_data = data["capabilities"]["payload_config"]

        from core.models.payload_config import PayloadConfiguration
        return PayloadConfiguration.from_dict(payload_data)

    def test_scansar_mode_present(self):
        payload = self._load_sar1_payload()
        assert payload.has_mode("scansar")

    def test_scansar_config_parsed(self):
        payload = self._load_sar1_payload()
        assert payload.has_scansar_config("scansar")
        cfg = payload.sar_scansar_configs["scansar"]
        assert cfg.beam_model == "continuous"
        assert cfg.num_subswaths == 5
        assert cfg.burst_duration_s == pytest.approx(0.05)
        assert cfg.enable_scalloping_model is True

    def test_get_scansar_calculator(self):
        payload = self._load_sar1_payload()
        calc = payload.get_scansar_calculator("scansar")
        assert calc is not None
        from core.dynamics.sar_scansar_calculator import SARScanSARCalculator
        assert isinstance(calc, SARScanSARCalculator)

    def test_scansar_calculator_compute(self):
        payload = self._load_sar1_payload()
        calc = payload.get_scansar_calculator("scansar")
        r = calc.compute_burst_params(631_000.0, 35.0, 7500.0)
        assert r.feasible
        assert r.peak_scalloping_db == pytest.approx(3.922, abs=0.02)
        # 5子条带 × 每条带 ~14.6km（antenna_width=2m, H=631km, look=35°）≈ 73km
        assert r.total_swath_width_km > 50
        assert r.num_subswaths_used == 5

    def test_calculator_cache_returns_same_instance(self):
        payload = self._load_sar1_payload()
        calc1 = payload.get_scansar_calculator("scansar")
        calc2 = payload.get_scansar_calculator("scansar")
        assert calc1 is calc2
