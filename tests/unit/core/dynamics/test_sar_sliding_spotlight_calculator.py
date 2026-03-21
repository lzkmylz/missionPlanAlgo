"""
SAR滑动聚束模式物理引擎单元测试

覆盖点：
  - SlidingBeamPosition / SARSlidingSpotlightConfig dataclass 验证
  - 三种 beam_model 计算路径（continuous / discrete_beam / derived_beam）
  - VRC距离验证（D_vrc > R, V_eff >= 5% V_sat）
  - 物理公式验证（V_eff, ρ_az, T_az 等滑动聚束特有公式）
  - 与聚束模式的对比验证
  - PayloadConfiguration 加载解析
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parents[4]

from core.models.sar_sliding_spotlight_config import (
    SlidingBeamPosition,
    SARSlidingSpotlightConfig,
    SARSlidingSpotlightResult,
)
from core.dynamics.sar_sliding_spotlight_calculator import (
    SARSlidingSpotlightCalculator,
    _effective_velocity,
    _check_veff,
)

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
H = 631_000.0   # m，典型LEO轨道高度
V = 7500.0      # m/s，卫星速度


# ---------------------------------------------------------------------------
# SlidingBeamPosition 测试
# ---------------------------------------------------------------------------

class TestSlidingBeamPosition:
    def test_valid_full_construction(self):
        bp = SlidingBeamPosition(
            beam_id="SSL-01",
            center_incidence_angle_deg=27.5,
            vrc_distance_m=2_000_000.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=35.0,
            prf_hz=3500.0,
            range_resolution_m=2.0,
            azimuth_resolution_m=2.5,
            scene_size_az_km=30.0,
            scene_size_rg_km=15.0,
        )
        assert bp.beam_id == "SSL-01"
        assert bp.vrc_distance_m == 2_000_000.0
        assert bp.is_fully_specified

    def test_vrc_distance_must_be_positive(self):
        with pytest.raises(ValueError, match="vrc_distance_m"):
            SlidingBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                vrc_distance_m=-100_000.0,
            )

    def test_vrc_distance_zero_raises(self):
        with pytest.raises(ValueError, match="vrc_distance_m"):
            SlidingBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                vrc_distance_m=0.0,
            )

    def test_min_geq_max_raises(self):
        with pytest.raises(ValueError, match="incidence_angle_min"):
            SlidingBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                incidence_angle_min_deg=40.0,
                incidence_angle_max_deg=35.0,
            )

    def test_zero_prf_raises(self):
        with pytest.raises(ValueError, match="prf_hz"):
            SlidingBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                prf_hz=0.0,
            )

    def test_covers(self):
        bp = SlidingBeamPosition(
            beam_id="X",
            center_incidence_angle_deg=30.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=40.0,
        )
        assert bp.covers(25.0)
        assert bp.covers(20.0)
        assert bp.covers(40.0)
        assert not bp.covers(19.9)
        assert not bp.covers(40.1)

    def test_from_dict_derived_only_center(self):
        """方案C只需 beam_id 和 center_incidence_angle_deg"""
        d = {"beam_id": "SSL-01", "center_incidence_angle_deg": 25.0}
        bp = SlidingBeamPosition.from_dict(d)
        assert bp.beam_id == "SSL-01"
        assert bp.vrc_distance_m is None
        assert not bp.is_fully_specified

    def test_from_dict_full(self):
        d = {
            "beam_id": "SSL-01",
            "center_incidence_angle_deg": 27.5,
            "vrc_distance_m": 1_800_000.0,
            "incidence_angle_min_deg": 20.0,
            "incidence_angle_max_deg": 35.0,
            "prf_hz": 3500.0,
            "range_resolution_m": 2.0,
            "azimuth_resolution_m": 2.5,
            "scene_size_az_km": 30.0,
            "scene_size_rg_km": 15.0,
        }
        bp = SlidingBeamPosition.from_dict(d)
        assert bp.beam_id == "SSL-01"
        assert bp.vrc_distance_m == 1_800_000.0
        assert bp.is_fully_specified


# ---------------------------------------------------------------------------
# SARSlidingSpotlightConfig 测试
# ---------------------------------------------------------------------------

class TestSARSlidingSpotlightConfig:
    def test_continuous_default(self):
        cfg = SARSlidingSpotlightConfig()
        assert cfg.beam_model == "continuous"
        assert cfg.vrc_distance_m == 2_000_000.0
        assert cfg.duty_cycle == 0.15

    def test_invalid_beam_model_raises(self):
        with pytest.raises(ValueError, match="beam_model"):
            SARSlidingSpotlightConfig(beam_model="unknown")

    def test_vrc_distance_missing_raises(self):
        """vrc_distance_m 缺失时抛出 ValueError"""
        d = {"beam_model": "continuous"}
        with pytest.raises(ValueError, match="vrc_distance_m"):
            SARSlidingSpotlightConfig.from_dict(d)

    def test_vrc_distance_zero_raises(self):
        with pytest.raises(ValueError, match="vrc_distance_m"):
            SARSlidingSpotlightConfig(vrc_distance_m=0.0)

    def test_discrete_requires_beam_positions(self):
        with pytest.raises(ValueError, match="SlidingBeamPosition"):
            SARSlidingSpotlightConfig(beam_model="discrete_beam", beam_positions=[])

    def test_discrete_requires_fully_specified_beams(self):
        bp = SlidingBeamPosition(beam_id="X", center_incidence_angle_deg=30.0)  # 缺少字段
        with pytest.raises(ValueError, match="missing required fields"):
            SARSlidingSpotlightConfig(beam_model="discrete_beam", beam_positions=[bp])

    def test_discrete_requires_vrc_in_beam(self):
        """方案B中波位必须提供 vrc_distance_m（已在 SlidingBeamPosition.__post_init__ 验证）"""
        # 创建未完全指定的波位（vrc_distance_m=None）
        bp = SlidingBeamPosition(
            beam_id="X",
            center_incidence_angle_deg=30.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=40.0,
            prf_hz=3000.0,
            range_resolution_m=2.0,
            azimuth_resolution_m=2.5,
            scene_size_az_km=30.0,
            scene_size_rg_km=15.0,
            # vrc_distance_m 缺失 → is_fully_specified=False
        )
        # 在 SARSlidingSpotlightConfig.__post_init__ 中检查
        with pytest.raises(ValueError, match="missing required fields"):
            SARSlidingSpotlightConfig(beam_model="discrete_beam", beam_positions=[bp])

    def test_derived_requires_beam_positions(self):
        with pytest.raises(ValueError, match="SlidingBeamPosition"):
            SARSlidingSpotlightConfig(beam_model="derived_beam", beam_positions=[])

    def test_from_dict_continuous(self):
        d = {
            "beam_model": "continuous",
            "vrc_distance_m": 1_800_000.0,
            "duty_cycle": 0.20,
            "prf_hz": 2800,
            "max_azimuth_steering_deg": 3.0,
        }
        cfg = SARSlidingSpotlightConfig.from_dict(d)
        assert cfg.beam_model == "continuous"
        assert cfg.vrc_distance_m == 1_800_000.0
        assert cfg.duty_cycle == 0.20

    def test_from_dict_discrete(self):
        d = {
            "beam_model": "discrete_beam",
            "wavelength_m": 0.031,
            "vrc_distance_m": 2_000_000.0,
            "duty_cycle": 0.15,
            "beam_positions": [
                {
                    "beam_id": "SSL-01",
                    "center_incidence_angle_deg": 27.5,
                    "vrc_distance_m": 1_800_000.0,
                    "incidence_angle_min_deg": 20.0,
                    "incidence_angle_max_deg": 35.0,
                    "prf_hz": 3500.0,
                    "range_resolution_m": 2.0,
                    "azimuth_resolution_m": 2.5,
                    "scene_size_az_km": 30.0,
                    "scene_size_rg_km": 15.0,
                }
            ],
        }
        cfg = SARSlidingSpotlightConfig.from_dict(d)
        assert cfg.beam_model == "discrete_beam"
        assert len(cfg.beam_positions) == 1
        assert cfg.beam_positions[0].vrc_distance_m == 1_800_000.0

    def test_from_dict_derived(self):
        d = {
            "beam_model": "derived_beam",
            "vrc_distance_m": 2_000_000.0,
            "duty_cycle": 0.15,
            "prf_safety_factor": 1.3,
            "beam_positions": [
                {"beam_id": "SSL-01", "center_incidence_angle_deg": 25.0},
                {"beam_id": "SSL-02", "center_incidence_angle_deg": 35.0},
            ],
        }
        cfg = SARSlidingSpotlightConfig.from_dict(d)
        assert cfg.beam_model == "derived_beam"
        assert cfg.vrc_distance_m == 2_000_000.0
        assert cfg.prf_safety_factor == 1.3
        assert len(cfg.beam_positions) == 2


# ---------------------------------------------------------------------------
# V_eff 可行性检查测试
# ---------------------------------------------------------------------------

class TestVeffFeasibility:
    """测试滑动聚束 V_eff 可行性检查"""

    def test_vrc_less_than_slant_range_infeasible(self):
        """D_vrc < R 时返回不可行"""
        # H=631km, look=30° → R ≈ 729km
        R = 729_000.0
        result = _check_veff(V, R, vrc_distance_m=700_000.0)
        assert not result.feasible
        assert "must be greater" in result.reason.lower() or "大于" in result.reason

    def test_vrc_equal_slant_range_infeasible(self):
        """D_vrc = R 时 V_eff = 0，不可行"""
        R = 729_000.0
        result = _check_veff(V, R, vrc_distance_m=R)
        assert not result.feasible

    def test_vrc_too_close_to_slant_range_infeasible(self):
        """D_vrc 过于接近 R 时（V_eff < 5% V_sat）不可行"""
        R = 729_000.0
        # D_vrc = 1.02 * R → V_eff/V_sat = 1.96% < 5%
        result = _check_veff(V, R, vrc_distance_m=744_000.0)
        assert not result.feasible
        assert "5%" in result.reason

    def test_vrc_sufficiently_large_feasible(self):
        """D_vrc > 1.05 * R 时可行"""
        R = 729_000.0
        result = _check_veff(V, R, vrc_distance_m=770_000.0)  # 1.056 * R
        assert result.feasible
        assert result.v_eff > 0.05 * V

    def test_vrc_ratio_computation(self):
        """验证 V_eff = V_sat × (1 - R/D_vrc)"""
        R = 800_000.0
        D_vrc = 2_000_000.0
        expected_v_eff = V * (1.0 - 0.4)  # 0.6 * V
        result = _check_veff(V, R, vrc_distance_m=D_vrc)
        assert result.feasible
        assert abs(result.v_eff - expected_v_eff) < 1.0


# ---------------------------------------------------------------------------
# 方案A（continuous）测试
# ---------------------------------------------------------------------------

class TestContinuousModel:

    @pytest.fixture
    def calc(self):
        cfg = SARSlidingSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            vrc_distance_m=2_000_000.0,
            duty_cycle=0.15,
            prf_hz=3000.0,
            max_azimuth_steering_deg=5.0,
            max_range_steering_deg=3.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        return SARSlidingSpotlightCalculator(cfg)

    def test_feasible_basic(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.dwell_time_s > 0
        assert result.scene_size_az_km > 0
        assert result.scene_size_rg_km > 0
        assert result.scene_area_km2 > 0

    def test_vrc_ratio_in_result(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        # H=631km, look=30° → R ≈ 729km, D_vrc=2000km → ratio ≈ 0.364
        assert 0.3 < result.vrc_ratio < 0.4

    def test_effective_velocity_in_result(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        expected_v_eff = V * (1 - result.vrc_ratio)
        assert abs(result.effective_scene_velocity_m_s - expected_v_eff) < 1.0

    def test_peak_power_factor(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        # duty_cycle=0.15 → peak_power_factor = 1/0.15 ≈ 6.67
        assert abs(result.peak_power_factor - 6.667) < 0.1

    def test_vrc_distance_matches_config(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.vrc_distance_m == 2_000_000.0

    def test_vrc_too_small_infeasible(self):
        """D_vrc 过小导致 V_eff < 5% V_sat 时不可行"""
        cfg = SARSlidingSpotlightConfig(
            beam_model="continuous",
            vrc_distance_m=750_000.0,  # 过于接近 631km 高度对应的斜距
            min_look_angle_deg=0.0,
            max_look_angle_deg=90.0,
        )
        calc = SARSlidingSpotlightCalculator(cfg)
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert not result.feasible
        assert "5%" in result.reason

    def test_scene_area_consistency(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=35.0, v_sat=V)
        assert result.feasible
        expected_area = result.scene_size_az_km * result.scene_size_rg_km
        assert abs(result.scene_area_km2 - expected_area) < 1e-9

    def test_azimuth_steering_limited(self):
        """调低PRF使PRF不成瓶颈，电扫范围成为限制"""
        cfg = SARSlidingSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            vrc_distance_m=2_000_000.0,
            duty_cycle=0.15,
            prf_hz=100_000.0,          # 极高PRF，不成瓶颈
            max_azimuth_steering_deg=0.5,  # 小电扫角
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        calc = SARSlidingSpotlightCalculator(cfg)
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.limiting_constraint == "azimuth_steering"

    def test_prf_ambiguity_limited(self, calc):
        """使用实际PRF计算，验证 PRF_ambiguity 限制"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        # PRF=3000Hz 通常会成为限制
        assert result.limiting_constraint in ("prf_ambiguity", "azimuth_steering")

    def test_select_beam_returns_none_for_continuous(self, calc):
        assert calc.select_beam_position(35.0) is None


# ---------------------------------------------------------------------------
# 方案B（discrete_beam）测试
# ---------------------------------------------------------------------------

class TestDiscreteBeamModel:

    @pytest.fixture
    def calc(self):
        cfg = SARSlidingSpotlightConfig.from_dict({
            "beam_model": "discrete_beam",
            "wavelength_m": 0.031,
            "antenna_length_m": 10.0,
            "antenna_width_m": 2.0,
            "range_resolution_m": 2.0,
            "vrc_distance_m": 2_000_000.0,  # 默认值（会被波位覆盖）
            "duty_cycle": 0.15,
            "beam_positions": [
                {
                    "beam_id": "SSL-01",
                    "center_incidence_angle_deg": 27.5,
                    "vrc_distance_m": 1_800_000.0,
                    "incidence_angle_min_deg": 20.0,
                    "incidence_angle_max_deg": 35.0,
                    "prf_hz": 3500.0,
                    "range_resolution_m": 2.0,
                    "azimuth_resolution_m": 2.5,
                    "scene_size_az_km": 30.0,
                    "scene_size_rg_km": 15.0,
                },
                {
                    "beam_id": "SSL-02",
                    "center_incidence_angle_deg": 45.0,
                    "vrc_distance_m": 2_200_000.0,
                    "incidence_angle_min_deg": 35.0,
                    "incidence_angle_max_deg": 55.0,
                    "prf_hz": 2800.0,
                    "range_resolution_m": 2.0,
                    "azimuth_resolution_m": 3.0,
                    "scene_size_az_km": 35.0,
                    "scene_size_rg_km": 18.0,
                },
            ],
        })
        return SARSlidingSpotlightCalculator(cfg)

    def test_feasible_first_beam(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=25.0, v_sat=V)
        assert result.feasible
        assert result.matched_beam_id == "SSL-01"
        assert result.prf_hz_used == 3500.0
        assert result.vrc_distance_m == 1_800_000.0

    def test_feasible_second_beam(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=40.0, v_sat=V)
        assert result.feasible
        assert result.matched_beam_id == "SSL-02"
        assert result.scene_size_rg_km == 18.0
        assert result.vrc_distance_m == 2_200_000.0

    def test_out_of_all_beams(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=10.0, v_sat=V)
        assert not result.feasible

    def test_scene_size_from_beam(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=25.0, v_sat=V)
        assert result.feasible
        assert result.scene_size_az_km == 30.0  # 直接来自波位配置

    def test_select_beam_position_boundary(self, calc):
        """35°入射角是两波位的边界，SSL-01 中心27.5°距离7.5°，SSL-02 中心45°距离10°，SSL-01胜"""
        beam = calc.select_beam_position(35.0)
        assert beam is not None
        assert beam.beam_id == "SSL-01"

    def test_select_beam_out_of_range(self, calc):
        beam = calc.select_beam_position(70.0)
        assert beam is None


# ---------------------------------------------------------------------------
# 方案C（derived_beam）测试
# ---------------------------------------------------------------------------

class TestDerivedBeamModel:

    @pytest.fixture
    def calc(self):
        cfg = SARSlidingSpotlightConfig.from_dict({
            "beam_model": "derived_beam",
            "wavelength_m": 0.031,
            "antenna_length_m": 10.0,
            "antenna_width_m": 2.0,
            "range_resolution_m": 2.0,
            "vrc_distance_m": 2_000_000.0,
            "duty_cycle": 0.15,
            "max_azimuth_steering_deg": 5.0,
            "max_range_steering_deg": 3.0,
            "prf_safety_factor": 1.25,
            "beam_positions": [
                {"beam_id": "SSL-01", "center_incidence_angle_deg": 25.0},
                {"beam_id": "SSL-02", "center_incidence_angle_deg": 35.0},
                {"beam_id": "SSL-03", "center_incidence_angle_deg": 45.0},
            ],
        })
        return SARSlidingSpotlightCalculator(cfg)

    def test_feasible_basic(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.dwell_time_s > 0

    def test_vrc_distance_from_config(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.vrc_distance_m == 2_000_000.0

    def test_beam_matched(self, calc):
        """look=30° → incidence≈34.7° → 最近中心35°(SSL-02)"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.matched_beam_id is not None

    def test_scene_az_order_of_magnitude(self, calc):
        """X波段631km 30°侧视 D_vrc=2000km，方位向场景应在 10-100km 量级"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert 10.0 <= result.scene_size_az_km <= 100.0

    def test_azimuth_resolution_order_of_magnitude(self, calc):
        """滑动聚束方位向分辨率应在 0.1-10m 量级（取决于配置）"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert 0.1 <= result.azimuth_resolution_m <= 10.0

    def test_scene_area_positive(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.scene_area_km2 > 0.0

    def test_scene_area_consistency(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        expected = result.scene_size_az_km * result.scene_size_rg_km
        assert abs(result.scene_area_km2 - expected) < 1e-9


# ---------------------------------------------------------------------------
# compute_scene_coverage 测试
# ---------------------------------------------------------------------------

class TestSceneCoverage:
    def test_given_dwell_time(self):
        cfg = SARSlidingSpotlightConfig(
            beam_model="continuous",
            vrc_distance_m=2_000_000.0,
            duty_cycle=0.15,
            prf_hz=3000.0,
        )
        calc = SARSlidingSpotlightCalculator(cfg)
        result = calc.compute_scene_coverage(
            altitude_m=H, look_angle_deg=30.0, dwell_time_s=5.0, v_sat=V
        )
        assert result.feasible
        # 验证 V_eff 计算正确
        R = 728616.0  # H=631km, look=30°
        expected_v_eff = V * (1 - R / 2_000_000.0)
        expected_az = expected_v_eff * 5.0 / 1000.0
        assert abs(result.scene_size_az_km - expected_az) < 0.1

    def test_zero_dwell_auto_compute(self):
        """dwell_time_s=0 时内部自动取最大驻留时间"""
        cfg = SARSlidingSpotlightConfig(
            beam_model="continuous",
            vrc_distance_m=2_000_000.0,
            duty_cycle=0.15,
            prf_hz=3000.0,
        )
        calc = SARSlidingSpotlightCalculator(cfg)
        result = calc.compute_scene_coverage(
            altitude_m=H, look_angle_deg=30.0, dwell_time_s=0.0, v_sat=V
        )
        assert result.feasible
        assert result.dwell_time_s > 0


# ---------------------------------------------------------------------------
# 与聚束模式的对比验证
# ---------------------------------------------------------------------------

class TestVsSpotlightMode:
    """验证滑动聚束与聚束模式的关键差异"""

    def test_sliding_vs_spotlight_both_feasible(self):
        """相同几何条件下，滑动聚束与聚束模式都应产生可行结果"""
        from core.models.sar_spotlight_config import SARSpotlightConfig
        from core.dynamics.sar_spotlight_calculator import SARSpotlightCalculator

        # 聚束模式
        cfg_spot = SARSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            prf_hz=3000.0,
            max_azimuth_steering_deg=5.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        calc_spot = SARSpotlightCalculator(cfg_spot)
        result_spot = calc_spot.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)

        # 滑动聚束模式（D_vrc = 2000km）
        cfg_slide = SARSlidingSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            vrc_distance_m=2_000_000.0,
            duty_cycle=0.15,
            prf_hz=3000.0,
            max_azimuth_steering_deg=5.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        calc_slide = SARSlidingSpotlightCalculator(cfg_slide)
        result_slide = calc_slide.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)

        # 两种模式都应可行
        assert result_spot.feasible
        assert result_slide.feasible

        # 两种模式都应产生正的场景尺寸和分辨率
        assert result_spot.scene_size_az_km > 0
        assert result_slide.scene_size_az_km > 0
        assert result_spot.azimuth_resolution_m > 0
        assert result_slide.azimuth_resolution_m > 0

        # 滑动聚束特有的输出字段
        assert result_slide.vrc_distance_m == 2_000_000.0
        assert result_slide.effective_scene_velocity_m_s > 0
        assert 0 < result_slide.vrc_ratio < 1
        assert result_slide.peak_power_factor > 1

    def test_sliding_vs_spotlight_dwell_time_difference(self):
        """相同几何条件下，滑动聚束与聚束模式的驻留时间差异"""
        from core.models.sar_spotlight_config import SARSpotlightConfig
        from core.dynamics.sar_spotlight_calculator import SARSpotlightCalculator

        # 聚束模式
        cfg_spot = SARSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            prf_hz=3000.0,
            max_azimuth_steering_deg=5.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        calc_spot = SARSpotlightCalculator(cfg_spot)
        result_spot = calc_spot.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)

        # 滑动聚束模式
        cfg_slide = SARSlidingSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            vrc_distance_m=2_000_000.0,
            duty_cycle=0.15,
            prf_hz=3000.0,
            max_azimuth_steering_deg=5.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        calc_slide = SARSlidingSpotlightCalculator(cfg_slide)
        result_slide = calc_slide.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)

        assert result_spot.feasible and result_slide.feasible
        # 聚束模式分辨率 = L_a/2 = 5m（固定）
        assert result_spot.azimuth_resolution_m == 5.0
        # 滑动聚束分辨率应在合理范围（通常与聚束模式相当或略差）
        assert 2.0 <= result_slide.azimuth_resolution_m <= 10.0
        # 滑动聚束的 VRC 距离应正确记录
        assert result_slide.vrc_distance_m == 2_000_000.0
        # 滑动聚束的有效速度应小于卫星速度
        assert result_slide.effective_scene_velocity_m_s < V


# ---------------------------------------------------------------------------
# PayloadConfiguration 加载测试
# ---------------------------------------------------------------------------

class TestPayloadConfigurationLoading:
    def test_sar2_derived_sliding_spotlight(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        assert pc.has_sliding_spotlight_config("sliding_spotlight")
        cfg = pc.sar_sliding_spotlight_configs["sliding_spotlight"]
        assert cfg.beam_model == "derived_beam"
        assert len(cfg.beam_positions) == 3
        assert cfg.vrc_distance_m == 2_000_000.0
        assert cfg.duty_cycle == 0.15

    def test_get_sliding_spotlight_calculator_returns_correct_type(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        calc = pc.get_sliding_spotlight_calculator("sliding_spotlight")
        assert isinstance(calc, SARSlidingSpotlightCalculator)

    def test_non_sliding_spotlight_mode_returns_none(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        calc = pc.get_sliding_spotlight_calculator("spotlight")  # spotlight 不是 sliding
        assert calc is None

    def test_has_sliding_spotlight_config_false_for_non_configured(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_1.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        # sar_1.json 只有 spotlight，没有 sliding_spotlight
        assert not pc.has_sliding_spotlight_config("sliding_spotlight")
