"""
SAR条带模式物理引擎单元测试

覆盖点：
  - StripmapBeamPosition / SARStripmapConfig dataclass 验证
  - 三种 beam_model 计算路径（continuous / discrete_beam / derived_beam）
  - 幅宽计算验证
  - 成像时间计算验证
  - 方位向分辨率固定为 L_a/2 验证
  - 与聚束/滑动聚束的对比验证
  - PayloadConfiguration 加载解析
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parents[4]

from core.models.sar_stripmap_config import (
    StripmapBeamPosition,
    SARStripmapConfig,
    SARStripmapResult,
)
from core.dynamics.sar_stripmap_calculator import SARStripmapCalculator

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
H = 631_000.0   # m，典型LEO轨道高度
V = 7500.0      # m/s，卫星速度


# ---------------------------------------------------------------------------
# StripmapBeamPosition 测试
# ---------------------------------------------------------------------------

class TestStripmapBeamPosition:
    def test_valid_full_construction(self):
        bp = StripmapBeamPosition(
            beam_id="ST-01",
            center_incidence_angle_deg=27.5,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=35.0,
            prf_hz=1500.0,
            range_resolution_m=3.0,
            azimuth_resolution_m=5.0,
            swath_width_km=20.0,
            nominal_integration_time_s=0.5,
        )
        assert bp.beam_id == "ST-01"
        assert bp.is_fully_specified

    def test_min_geq_max_raises(self):
        with pytest.raises(ValueError, match="incidence_angle_min"):
            StripmapBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                incidence_angle_min_deg=40.0,
                incidence_angle_max_deg=35.0,
            )

    def test_zero_prf_raises(self):
        with pytest.raises(ValueError, match="prf_hz"):
            StripmapBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                prf_hz=0.0,
            )

    def test_zero_swath_raises(self):
        with pytest.raises(ValueError, match="swath_width_km"):
            StripmapBeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                swath_width_km=0.0,
            )

    def test_covers(self):
        bp = StripmapBeamPosition(
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
        d = {"beam_id": "ST-01", "center_incidence_angle_deg": 25.0}
        bp = StripmapBeamPosition.from_dict(d)
        assert bp.beam_id == "ST-01"
        assert not bp.is_fully_specified


# ---------------------------------------------------------------------------
# SARStripmapConfig 测试
# ---------------------------------------------------------------------------

class TestSARStripmapConfig:
    def test_continuous_default(self):
        cfg = SARStripmapConfig()
        assert cfg.beam_model == "continuous"
        assert cfg.prf_hz == 1500.0
        assert cfg.duty_cycle == 0.10

    def test_invalid_beam_model_raises(self):
        with pytest.raises(ValueError, match="beam_model"):
            SARStripmapConfig(beam_model="unknown")

    def test_invalid_duty_cycle_raises(self):
        with pytest.raises(ValueError, match="duty_cycle"):
            SARStripmapConfig(duty_cycle=0.0)
        with pytest.raises(ValueError, match="duty_cycle"):
            SARStripmapConfig(duty_cycle=1.5)

    def test_discrete_requires_beam_positions(self):
        with pytest.raises(ValueError, match="StripmapBeamPosition"):
            SARStripmapConfig(beam_model="discrete_beam", beam_positions=[])

    def test_discrete_requires_fully_specified_beams(self):
        bp = StripmapBeamPosition(beam_id="X", center_incidence_angle_deg=30.0)
        with pytest.raises(ValueError, match="missing required fields"):
            SARStripmapConfig(beam_model="discrete_beam", beam_positions=[bp])

    def test_derived_requires_beam_positions(self):
        with pytest.raises(ValueError, match="StripmapBeamPosition"):
            SARStripmapConfig(beam_model="derived_beam", beam_positions=[])

    def test_from_dict_continuous(self):
        d = {
            "beam_model": "continuous",
            "prf_hz": 2000,
            "duty_cycle": 0.15,
            "min_look_angle_deg": 20.0,
        }
        cfg = SARStripmapConfig.from_dict(d)
        assert cfg.beam_model == "continuous"
        assert cfg.prf_hz == 2000.0
        assert cfg.duty_cycle == 0.15


# ---------------------------------------------------------------------------
# 方案A（continuous）测试
# ---------------------------------------------------------------------------

class TestContinuousModel:

    @pytest.fixture
    def calc(self):
        cfg = SARStripmapConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            range_resolution_m=3.0,
            duty_cycle=0.10,
            prf_hz=1500.0,
            min_look_angle_deg=15.0,
            max_look_angle_deg=50.0,
        )
        return SARStripmapCalculator(cfg)

    def test_feasible_basic(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        assert result.imaging_time_s > 0
        assert result.swath_width_km > 0
        assert result.scene_area_km2 > 0

    def test_azimuth_resolution_fixed(self, calc):
        """条带模式方位向分辨率固定为 L_a/2"""
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        # antenna_length_m = 10.0 → ρ_az = 5.0m
        assert result.azimuth_resolution_m == 5.0

    def test_imaging_time_calculation(self, calc):
        """成像时间 = 场景长度 / 卫星速度"""
        scene_length_km = 50.0
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=scene_length_km, v_sat=V
        )
        assert result.feasible
        expected_time = scene_length_km * 1000.0 / V  # 秒
        assert abs(result.imaging_time_s - expected_time) < 0.1

    def test_peak_power_factor(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        # duty_cycle=0.10 → peak_power_factor = 1/0.10 = 10.0
        assert abs(result.peak_power_factor - 10.0) < 0.1

    def test_out_of_look_angle_range(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=60.0, scene_length_km=50.0, v_sat=V
        )
        assert not result.feasible
        assert "超出范围" in (result.reason or "")

    def test_scene_area_consistency(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=35.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        expected_area = result.scene_length_along_track_km * result.swath_width_km
        assert abs(result.scene_area_km2 - expected_area) < 1e-9

    def test_select_beam_returns_none_for_continuous(self, calc):
        assert calc.select_beam_position(35.0) is None


# ---------------------------------------------------------------------------
# 方案B（discrete_beam）测试
# ---------------------------------------------------------------------------

class TestDiscreteBeamModel:

    @pytest.fixture
    def calc(self):
        cfg = SARStripmapConfig.from_dict({
            "beam_model": "discrete_beam",
            "wavelength_m": 0.031,
            "antenna_length_m": 10.0,
            "antenna_width_m": 2.0,
            "range_resolution_m": 3.0,
            "duty_cycle": 0.10,
            "beam_positions": [
                {
                    "beam_id": "ST-01",
                    "center_incidence_angle_deg": 27.5,
                    "incidence_angle_min_deg": 20.0,
                    "incidence_angle_max_deg": 35.0,
                    "prf_hz": 1600.0,
                    "range_resolution_m": 3.0,
                    "azimuth_resolution_m": 5.0,
                    "swath_width_km": 22.0,
                    "nominal_integration_time_s": 0.5,
                },
                {
                    "beam_id": "ST-02",
                    "center_incidence_angle_deg": 45.0,
                    "incidence_angle_min_deg": 35.0,
                    "incidence_angle_max_deg": 55.0,
                    "prf_hz": 1400.0,
                    "range_resolution_m": 3.0,
                    "azimuth_resolution_m": 5.0,
                    "swath_width_km": 28.0,
                    "nominal_integration_time_s": 0.5,
                },
            ],
        })
        return SARStripmapCalculator(cfg)

    def test_feasible_first_beam(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=25.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        assert result.matched_beam_id == "ST-01"
        assert result.prf_hz_used == 1600.0
        assert result.swath_width_km == 22.0

    def test_feasible_second_beam(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=40.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        assert result.matched_beam_id == "ST-02"
        assert result.swath_width_km == 28.0

    def test_out_of_all_beams(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=10.0, scene_length_km=50.0, v_sat=V
        )
        assert not result.feasible


# ---------------------------------------------------------------------------
# 方案C（derived_beam）测试
# ---------------------------------------------------------------------------

class TestDerivedBeamModel:

    @pytest.fixture
    def calc(self):
        cfg = SARStripmapConfig.from_dict({
            "beam_model": "derived_beam",
            "wavelength_m": 0.031,
            "antenna_length_m": 10.0,
            "antenna_width_m": 2.0,
            "range_resolution_m": 3.0,
            "duty_cycle": 0.10,
            "nominal_integration_time_s": 0.5,
            "prf_safety_factor": 1.25,
            "beam_positions": [
                {"beam_id": "ST-01", "center_incidence_angle_deg": 25.0},
                {"beam_id": "ST-02", "center_incidence_angle_deg": 35.0},
                {"beam_id": "ST-03", "center_incidence_angle_deg": 45.0},
            ],
        })
        return SARStripmapCalculator(cfg)

    def test_feasible_basic(self, calc):
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        assert result.imaging_time_s > 0
        assert result.swath_width_km > 0

    def test_beam_matched(self, calc):
        """look=30° → incidence≈34.7° → 最近中心35°(ST-02)"""
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        assert result.matched_beam_id is not None

    def test_azimuth_resolution_fixed(self, calc):
        """方案C方位向分辨率仍为 L_a/2"""
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        assert result.azimuth_resolution_m == 5.0

    def test_prf_derived(self, calc):
        """PRF = safety_factor × V_sat / (2 × ρ_az)"""
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )
        assert result.feasible
        # ρ_az = 5.0m, safety_factor=1.25, V=7500
        # PRF = 1.25 * 7500 / (2 * 5.0) = 937.5 Hz
        expected_prf = 1.25 * V / (2.0 * 5.0)
        assert abs(result.prf_hz_used - expected_prf) < 1.0


# ---------------------------------------------------------------------------
# 幅宽计算测试
# ---------------------------------------------------------------------------

class TestSwathWidthCalculation:
    """测试条带模式幅宽计算"""

    def test_swath_width_increases_with_incidence(self):
        """入射角越大，幅宽越大（因为 1/cos(θ) 增大）"""
        cfg = SARStripmapConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_width_m=2.0,
        )
        calc = SARStripmapCalculator(cfg)

        swath_20 = calc.compute_swath_width(altitude_m=H, look_angle_deg=20.0)
        swath_40 = calc.compute_swath_width(altitude_m=H, look_angle_deg=40.0)

        assert swath_20 > 0
        assert swath_40 > 0
        assert swath_40 > swath_20

    def test_swath_width_formula(self):
        """验证幅宽公式：W = (λ/L_w) × R / cos(θ_inc)"""
        cfg = SARStripmapConfig(
            wavelength_m=0.031,
            antenna_width_m=2.0,
        )
        calc = SARStripmapCalculator(cfg)

        look_angle = 30.0
        swath_km = calc.compute_swath_width(altitude_m=H, look_angle_deg=look_angle)

        # 手动计算验证
        from core.dynamics.sar_spotlight_calculator import _slant_range, _look_to_incidence
        R = _slant_range(H, look_angle)
        incidence = _look_to_incidence(H, look_angle)
        beam_width = 0.031 / 2.0  # rad
        expected_swath_m = beam_width * R / math.cos(math.radians(incidence))
        expected_swath_km = expected_swath_m / 1000.0

        assert abs(swath_km - expected_swath_km) < 0.1


# ---------------------------------------------------------------------------
# 与聚束/滑动聚束的对比验证
# ---------------------------------------------------------------------------

class TestVsSpotlightAndSliding:
    """验证条带模式与聚束/滑动聚束的关键差异"""

    def test_stripmap_vs_spotlight_resolution(self):
        """条带模式与聚束模式的方位向分辨率相同（都为 L_a/2）"""
        from core.models.sar_spotlight_config import SARSpotlightConfig
        from core.dynamics.sar_spotlight_calculator import SARSpotlightCalculator

        # 聚束模式
        cfg_spot = SARSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            prf_hz=3000.0,
        )
        calc_spot = SARSpotlightCalculator(cfg_spot)
        result_spot = calc_spot.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)

        # 条带模式
        cfg_strip = SARStripmapConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
        )
        calc_strip = SARStripmapCalculator(cfg_strip)
        result_strip = calc_strip.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )

        assert result_spot.feasible and result_strip.feasible
        # 两者方位向分辨率都应为 L_a/2 = 5m
        assert result_spot.azimuth_resolution_m == 5.0
        assert result_strip.azimuth_resolution_m == 5.0

    def test_stripmap_no_dwell_time_concept(self):
        """条带模式没有驻留时间概念，而是成像时间"""
        cfg = SARStripmapConfig()
        calc = SARStripmapCalculator(cfg)
        result = calc.compute_imaging_params(
            altitude_m=H, look_angle_deg=30.0, scene_length_km=50.0, v_sat=V
        )

        assert result.feasible
        # 成像时间由用户指定的场景长度决定
        expected_time = 50.0 * 1000.0 / V  # 约 6.67 秒
        assert abs(result.imaging_time_s - expected_time) < 0.1


# ---------------------------------------------------------------------------
# PayloadConfiguration 加载测试
# ---------------------------------------------------------------------------

class TestPayloadConfigurationLoading:
    def test_sar2_derived_stripmap(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        assert pc.has_stripmap_config("stripmap")
        cfg = pc.sar_stripmap_configs["stripmap"]
        assert cfg.beam_model == "derived_beam"
        assert len(cfg.beam_positions) == 3
        assert cfg.duty_cycle == 0.10

    def test_get_stripmap_calculator_returns_correct_type(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        calc = pc.get_stripmap_calculator("stripmap")
        assert isinstance(calc, SARStripmapCalculator)

    def test_non_stripmap_mode_returns_none(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        calc = pc.get_stripmap_calculator("spotlight")
        assert calc is None
