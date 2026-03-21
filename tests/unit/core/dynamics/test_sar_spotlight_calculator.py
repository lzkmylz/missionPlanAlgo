"""
SAR聚束模式物理引擎单元测试

覆盖点：
  - BeamPosition / SARSpotlightConfig dataclass 验证
  - 三种 beam_model 计算路径（continuous / discrete_beam / derived_beam）
  - 波位匹配逻辑（select_beam_position）
  - 方案B/C 等价性验证
  - 不可行场景边界情况
  - PayloadConfiguration 加载解析
"""

import math
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parents[4]

from core.models.sar_spotlight_config import (
    BeamPosition,
    SARSpotlightConfig,
    SARSpotlightResult,
    SPEED_OF_LIGHT,
)
from core.dynamics.sar_spotlight_calculator import (
    SARSpotlightCalculator,
    _slant_range,
    _look_to_incidence,
    _scene_size_rg,
)

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
H = 631_000.0   # m，典型LEO轨道高度
V = 7500.0      # m/s，卫星速度

# ---------------------------------------------------------------------------
# BeamPosition 测试
# ---------------------------------------------------------------------------

class TestBeamPosition:
    def test_valid_full_construction(self):
        bp = BeamPosition(
            beam_id="SL-01",
            center_incidence_angle_deg=27.5,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=35.0,
            prf_hz=3500.0,
            range_resolution_m=1.0,
            azimuth_resolution_m=1.0,
            scene_size_az_km=10.0,
            scene_size_rg_km=10.0,
        )
        assert bp.beam_id == "SL-01"
        assert bp.is_fully_specified

    def test_min_geq_max_raises(self):
        with pytest.raises(ValueError, match="incidence_angle_min"):
            BeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                incidence_angle_min_deg=40.0,
                incidence_angle_max_deg=35.0,
            )

    def test_zero_prf_raises(self):
        with pytest.raises(ValueError, match="prf_hz"):
            BeamPosition(
                beam_id="X",
                center_incidence_angle_deg=30.0,
                prf_hz=0.0,
            )

    def test_center_midpoint(self):
        bp = BeamPosition(
            beam_id="X",
            center_incidence_angle_deg=30.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=40.0,
        )
        assert bp.center_incidence_angle_deg == 30.0

    def test_covers(self):
        bp = BeamPosition(
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

    def test_from_dict_full(self):
        d = {
            "beam_id": "SL-01",
            "center_incidence_angle_deg": 27.5,
            "incidence_angle_min_deg": 20.0,
            "incidence_angle_max_deg": 35.0,
            "prf_hz": 3500.0,
            "range_resolution_m": 1.0,
            "azimuth_resolution_m": 1.0,
            "scene_size_az_km": 10.0,
            "scene_size_rg_km": 10.0,
        }
        bp = BeamPosition.from_dict(d)
        assert bp.beam_id == "SL-01"
        assert bp.prf_hz == 3500.0
        assert bp.is_fully_specified

    def test_from_dict_derived_only_center(self):
        """方案C只需 beam_id 和 center_incidence_angle_deg"""
        d = {"beam_id": "SSL-01", "center_incidence_angle_deg": 25.0}
        bp = BeamPosition.from_dict(d)
        assert bp.beam_id == "SSL-01"
        assert not bp.is_fully_specified


# ---------------------------------------------------------------------------
# SARSpotlightConfig 测试
# ---------------------------------------------------------------------------

class TestSARSpotlightConfig:
    def test_continuous_default(self):
        cfg = SARSpotlightConfig()
        assert cfg.beam_model == "continuous"
        assert cfg.prf_hz == 3000.0

    def test_invalid_beam_model_raises(self):
        with pytest.raises(ValueError, match="beam_model"):
            SARSpotlightConfig(beam_model="unknown")

    def test_discrete_requires_beam_positions(self):
        with pytest.raises(ValueError, match="BeamPosition"):
            SARSpotlightConfig(beam_model="discrete_beam", beam_positions=[])

    def test_discrete_requires_fully_specified_beams(self):
        bp = BeamPosition(beam_id="X", center_incidence_angle_deg=30.0)  # 缺少字段
        with pytest.raises(ValueError, match="missing required fields"):
            SARSpotlightConfig(beam_model="discrete_beam", beam_positions=[bp])

    def test_derived_requires_beam_positions(self):
        with pytest.raises(ValueError, match="BeamPosition"):
            SARSpotlightConfig(beam_model="derived_beam", beam_positions=[])

    def test_from_dict_continuous(self):
        d = {
            "beam_model": "continuous",
            "prf_hz": 2800,
            "max_azimuth_steering_deg": 3.0,
            "wavelength_m": 0.031,
        }
        cfg = SARSpotlightConfig.from_dict(d)
        assert cfg.beam_model == "continuous"
        assert cfg.prf_hz == 2800.0
        assert cfg.max_azimuth_steering_deg == 3.0

    def test_from_dict_discrete(self):
        d = {
            "beam_model": "discrete_beam",
            "wavelength_m": 0.031,
            "beam_positions": [
                {
                    "beam_id": "SL-01",
                    "center_incidence_angle_deg": 27.5,
                    "incidence_angle_min_deg": 20.0,
                    "incidence_angle_max_deg": 35.0,
                    "prf_hz": 3500.0,
                    "range_resolution_m": 1.0,
                    "azimuth_resolution_m": 1.0,
                    "scene_size_az_km": 10.0,
                    "scene_size_rg_km": 10.0,
                }
            ],
        }
        cfg = SARSpotlightConfig.from_dict(d)
        assert cfg.beam_model == "discrete_beam"
        assert len(cfg.beam_positions) == 1
        assert cfg.beam_positions[0].beam_id == "SL-01"

    def test_from_dict_derived(self):
        d = {
            "beam_model": "derived_beam",
            "prf_safety_factor": 1.3,
            "beam_positions": [
                {"beam_id": "SSL-01", "center_incidence_angle_deg": 25.0},
                {"beam_id": "SSL-02", "center_incidence_angle_deg": 35.0},
            ],
        }
        cfg = SARSpotlightConfig.from_dict(d)
        assert cfg.beam_model == "derived_beam"
        assert cfg.prf_safety_factor == 1.3
        assert len(cfg.beam_positions) == 2


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_slant_range_nadir(self):
        R = _slant_range(altitude_m=H, look_angle_deg=0.0)
        assert abs(R - H) < 1.0

    def test_slant_range_30deg(self):
        R = _slant_range(altitude_m=H, look_angle_deg=30.0)
        expected = H / math.cos(math.radians(30.0))
        assert abs(R - expected) < 1.0

    def test_look_to_incidence_increases(self):
        """入射角应大于侧视角（球面几何）"""
        incidence = _look_to_incidence(altitude_m=H, look_angle_deg=30.0)
        assert incidence > 30.0

    def test_look_to_incidence_small_angle(self):
        """小角度时入射角≈侧视角"""
        incidence = _look_to_incidence(altitude_m=H, look_angle_deg=5.0)
        assert abs(incidence - 5.0) < 2.0


# ---------------------------------------------------------------------------
# 方案A（continuous）测试
# ---------------------------------------------------------------------------

class TestContinuousModel:

    @pytest.fixture
    def calc(self):
        cfg = SARSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=6.0,
            antenna_width_m=1.5,
            prf_hz=3000.0,
            max_azimuth_steering_deg=2.0,
            max_range_steering_deg=2.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        return SARSpotlightCalculator(cfg)

    def test_feasible_basic(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.dwell_time_s > 0
        assert result.scene_size_az_km > 0
        assert result.scene_size_rg_km > 0
        assert result.scene_area_km2 > 0

    def test_reference_value_prf_limited(self, calc):
        """
        参考数值验证（H=631km, look=30°, PRF=3000Hz, λ=0.031m）
        R ≈ 728,800m，T_prf = 3000*0.031*728800/(2*7500²) ≈ 0.60s
        T_az = 2*rad(2°)*728800/7500 ≈ 6.79s → PRF限制
        """
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.limiting_constraint == "prf_ambiguity"
        assert 0.4 < result.dwell_time_s < 1.0  # PRF主导，约0.60s

    def test_azimuth_steering_limited(self):
        """调低PRF使PRF不成瓶颈，电扫范围成为限制"""
        cfg = SARSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_length_m=6.0,
            prf_hz=100_000.0,          # 极高PRF，不成瓶颈
            max_azimuth_steering_deg=0.5,  # 小电扫角
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        calc = SARSpotlightCalculator(cfg)
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.limiting_constraint == "azimuth_steering"

    def test_out_of_look_angle_range(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=60.0, v_sat=V)
        assert not result.feasible
        assert "超出范围" in (result.reason or "")

    def test_very_low_prf_gives_short_dwell(self):
        """极低PRF不导致不可行，只缩短驻留时间（PRF约束驻留时间，而非斜距可行性）"""
        cfg = SARSpotlightConfig(
            beam_model="continuous",
            prf_hz=50.0,  # 极低PRF → T_prf极短
            min_look_angle_deg=0.0,
            max_look_angle_deg=90.0,
        )
        calc = SARSpotlightCalculator(cfg)
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.limiting_constraint == "prf_ambiguity"
        assert result.dwell_time_s < 0.1  # PRF=50Hz 对应极短驻留时间

    def test_scene_area_consistency(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=35.0, v_sat=V)
        assert result.feasible
        expected_area = result.scene_size_az_km * result.scene_size_rg_km
        assert abs(result.scene_area_km2 - expected_area) < 1e-9

    def test_select_beam_returns_none_for_continuous(self, calc):
        assert calc.select_beam_position(35.0) is None


# ---------------------------------------------------------------------------
# 方案B（discrete_beam）测试
# ---------------------------------------------------------------------------

class TestDiscreteBeamModel:

    @pytest.fixture
    def calc(self):
        cfg = SARSpotlightConfig.from_dict({
            "beam_model": "discrete_beam",
            "wavelength_m": 0.031,
            "antenna_length_m": 10.0,
            "beam_positions": [
                {
                    "beam_id": "SL-01",
                    "center_incidence_angle_deg": 27.5,
                    "incidence_angle_min_deg": 20.0,
                    "incidence_angle_max_deg": 35.0,
                    "prf_hz": 3500.0,
                    "range_resolution_m": 1.0,
                    "azimuth_resolution_m": 1.0,
                    "scene_size_az_km": 10.0,
                    "scene_size_rg_km": 10.0,
                },
                {
                    "beam_id": "SL-02",
                    "center_incidence_angle_deg": 45.0,
                    "incidence_angle_min_deg": 35.0,
                    "incidence_angle_max_deg": 55.0,
                    "prf_hz": 2800.0,
                    "range_resolution_m": 1.0,
                    "azimuth_resolution_m": 1.0,
                    "scene_size_az_km": 10.0,
                    "scene_size_rg_km": 12.0,
                },
            ],
        })
        return SARSpotlightCalculator(cfg)

    def test_feasible_first_beam(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=25.0, v_sat=V)
        assert result.feasible
        assert result.matched_beam_id == "SL-01"
        assert result.prf_hz_used == 3500.0

    def test_feasible_second_beam(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=40.0, v_sat=V)
        assert result.feasible
        assert result.matched_beam_id == "SL-02"
        assert result.scene_size_rg_km == 12.0

    def test_out_of_all_beams(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=10.0, v_sat=V)
        assert not result.feasible

    def test_scene_size_from_beam(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=25.0, v_sat=V)
        assert result.feasible
        assert result.scene_size_az_km == 10.0  # 直接来自波位配置

    def test_select_beam_position_boundary(self, calc):
        """35°入射角是两波位的边界，SL-01 中心27.5°距离7.5°，SL-02 中心45°距离10°，SL-01胜"""
        beam = calc.select_beam_position(35.0)
        assert beam is not None
        assert beam.beam_id == "SL-01"

    def test_select_beam_out_of_range(self, calc):
        beam = calc.select_beam_position(70.0)
        assert beam is None

    def test_deterministic_tie_break(self):
        """相同中心距离时按 beam_id 字母序决定"""
        cfg = SARSpotlightConfig.from_dict({
            "beam_model": "discrete_beam",
            "beam_positions": [
                {
                    "beam_id": "B", "center_incidence_angle_deg": 30.0,
                    "incidence_angle_min_deg": 20.0, "incidence_angle_max_deg": 40.0,
                    "prf_hz": 3000.0, "range_resolution_m": 1.0,
                    "azimuth_resolution_m": 1.0, "scene_size_az_km": 10.0,
                    "scene_size_rg_km": 10.0,
                },
                {
                    "beam_id": "A", "center_incidence_angle_deg": 30.0,
                    "incidence_angle_min_deg": 20.0, "incidence_angle_max_deg": 40.0,
                    "prf_hz": 3000.0, "range_resolution_m": 1.0,
                    "azimuth_resolution_m": 1.0, "scene_size_az_km": 10.0,
                    "scene_size_rg_km": 10.0,
                },
            ],
        })
        calc = SARSpotlightCalculator(cfg)
        beam = calc.select_beam_position(30.0)
        assert beam.beam_id == "A"  # 字母序较小


# ---------------------------------------------------------------------------
# 方案C（derived_beam）测试
# ---------------------------------------------------------------------------

class TestDerivedBeamModel:

    @pytest.fixture
    def calc(self):
        cfg = SARSpotlightConfig.from_dict({
            "beam_model": "derived_beam",
            "wavelength_m": 0.031,
            "antenna_length_m": 6.0,
            "antenna_width_m": 1.5,
            "range_resolution_m": 1.0,
            "max_azimuth_steering_deg": 2.0,
            "max_range_steering_deg": 2.0,
            "prf_safety_factor": 1.25,
            "beam_positions": [
                {"beam_id": "SSL-01", "center_incidence_angle_deg": 25.0},
                {"beam_id": "SSL-02", "center_incidence_angle_deg": 35.0},
                {"beam_id": "SSL-03", "center_incidence_angle_deg": 45.0},
            ],
        })
        return SARSpotlightCalculator(cfg)

    def test_feasible_basic(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.dwell_time_s > 0

    def test_prf_derived_in_reasonable_range(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert 500 < result.prf_hz_used < 100_000  # X波段聚束PRF通常在数kHz至数十kHz

    def test_beam_matched(self, calc):
        """look=30° → incidence≈34.7° → 最近中心35°(SSL-02)"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.matched_beam_id is not None

    def test_scene_az_order_of_magnitude(self, calc):
        """X波段631km 30°侧视，方位向场景应在 1-50km 量级"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert 1.0 <= result.scene_size_az_km <= 50.0

    def test_scene_rg_order_of_magnitude(self, calc):
        """距离向场景应在 1-30km 量级"""
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert 1.0 <= result.scene_size_rg_km <= 30.0

    def test_scene_area_positive(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        assert result.scene_area_km2 > 0.0

    def test_scene_area_consistency(self, calc):
        result = calc.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert result.feasible
        expected = result.scene_size_az_km * result.scene_size_rg_km
        assert abs(result.scene_area_km2 - expected) < 1e-9

    def test_larger_steering_gives_larger_scene(self):
        """电扫角越大，推导的方位向场景越大"""
        def make_calc(steering_deg):
            cfg = SARSpotlightConfig(
                beam_model="derived_beam",
                wavelength_m=0.031,
                antenna_length_m=6.0,
                antenna_width_m=1.5,
                range_resolution_m=1.0,
                max_azimuth_steering_deg=steering_deg,
                max_range_steering_deg=2.0,
                prf_safety_factor=1.25,
                beam_positions=[BeamPosition(beam_id="X", center_incidence_angle_deg=30.0)],
            )
            return SARSpotlightCalculator(cfg)

        res1 = make_calc(1.0).compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        res2 = make_calc(3.0).compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert res1.feasible and res2.feasible
        assert res2.scene_size_az_km > res1.scene_size_az_km


# ---------------------------------------------------------------------------
# 方案B/C 等价性验证
# ---------------------------------------------------------------------------

class TestEquivalence:
    """方案C推导结果应与同参数手工配置的方案B在合理误差内吻合"""

    def test_scene_rg_equivalence(self):
        """相同系统参数下，两种方案距离向场景尺寸应相差 < 50%"""
        cfg_a = SARSpotlightConfig(
            beam_model="continuous",
            wavelength_m=0.031,
            antenna_width_m=1.5,
            max_range_steering_deg=2.0,
        )
        cfg_c = SARSpotlightConfig(
            beam_model="derived_beam",
            wavelength_m=0.031,
            antenna_width_m=1.5,
            max_range_steering_deg=2.0,
            beam_positions=[
                BeamPosition(beam_id="X", center_incidence_angle_deg=30.0)
            ],
        )
        calc_a = SARSpotlightCalculator(cfg_a)
        calc_c = SARSpotlightCalculator(cfg_c)
        res_a = calc_a.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        res_c = calc_c.compute_dwell_time(altitude_m=H, look_angle_deg=30.0, v_sat=V)
        assert res_a.feasible and res_c.feasible
        ratio = res_c.scene_size_rg_km / res_a.scene_size_rg_km
        assert 0.5 <= ratio <= 2.0


# ---------------------------------------------------------------------------
# compute_scene_coverage 测试
# ---------------------------------------------------------------------------

class TestSceneCoverage:
    def test_given_dwell_time(self):
        cfg = SARSpotlightConfig(beam_model="continuous", prf_hz=3000.0)
        calc = SARSpotlightCalculator(cfg)
        result = calc.compute_scene_coverage(
            altitude_m=H, look_angle_deg=30.0, dwell_time_s=2.0, v_sat=V
        )
        assert result.feasible
        expected_az = V * 2.0 / 1000.0
        assert abs(result.scene_size_az_km - expected_az) < 0.01

    def test_zero_dwell_auto_compute(self):
        """dwell_time_s=0 时内部自动取最大驻留时间"""
        cfg = SARSpotlightConfig(beam_model="continuous", prf_hz=3000.0)
        calc = SARSpotlightCalculator(cfg)
        result = calc.compute_scene_coverage(
            altitude_m=H, look_angle_deg=30.0, dwell_time_s=0.0, v_sat=V
        )
        assert result.feasible
        assert result.dwell_time_s > 0


# ---------------------------------------------------------------------------
# PayloadConfiguration 加载测试
# ---------------------------------------------------------------------------

class TestPayloadConfigurationLoading:
    def test_sar1_continuous(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_1.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        assert pc.has_spotlight_config("spotlight")
        cfg = pc.sar_spotlight_configs["spotlight"]
        assert cfg.beam_model == "continuous"
        assert cfg.prf_hz == 3000.0

    def test_sar2_discrete(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        assert pc.has_spotlight_config("spotlight")
        cfg = pc.sar_spotlight_configs["spotlight"]
        assert cfg.beam_model == "discrete_beam"
        assert len(cfg.beam_positions) == 2

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

    def test_get_spotlight_calculator_returns_calculator(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_1.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        from core.dynamics.sar_spotlight_calculator import SARSpotlightCalculator
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        calc = pc.get_spotlight_calculator("spotlight")
        assert isinstance(calc, SARSpotlightCalculator)

    def test_non_spotlight_mode_returns_none(self):
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_1.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])
        calc = pc.get_spotlight_calculator("stripmap")
        assert calc is None

    def test_calculator_caching_returns_same_instance(self):
        """测试计算器缓存：同一模式多次调用应返回相同实例"""
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])

        calc1 = pc.get_spotlight_calculator("spotlight")
        calc2 = pc.get_spotlight_calculator("spotlight")

        # 缓存应返回同一实例
        assert calc1 is calc2

    def test_calculator_no_cache_returns_new_instance(self):
        """测试禁用缓存：应返回新的实例"""
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])

        calc1 = pc.get_spotlight_calculator("spotlight", use_cache=False)
        calc2 = pc.get_spotlight_calculator("spotlight", use_cache=False)

        # 禁用缓存应返回不同实例
        assert calc1 is not calc2

    def test_clear_calculator_cache(self):
        """测试清除缓存后应创建新实例"""
        import json
        with open(PROJECT_ROOT / "data/entity_lib/satellites/sar_2.json") as f:
            raw = json.load(f)
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(raw["capabilities"]["payload_config"])

        calc1 = pc.get_spotlight_calculator("spotlight")
        pc.clear_calculator_cache()
        calc2 = pc.get_spotlight_calculator("spotlight")

        # 清除缓存后应创建新实例
        assert calc1 is not calc2

    def test_config_priority_override(self):
        """测试可配置的SAR配置类型优先级"""
        from core.models.payload_config import PayloadConfiguration

        # 创建包含多种config的测试数据
        test_data = {
            'payload_type': 'sar',
            'default_mode': 'test_mode',
            'modes': {
                'test_mode': {
                    'resolution_m': 1.0,
                    'swath_width_m': 10000,
                    'power_consumption_w': 500,
                    'data_rate_mbps': 300,
                    'min_duration_s': 10,
                    'max_duration_s': 30,
                    'mode_type': 'sar',
                    # 同时包含三种配置
                    'spotlight_config': {
                        'beam_model': 'continuous',
                        'wavelength_m': 0.031,
                        'antenna_length_m': 10.0,
                        'prf_hz': 3000
                    },
                    'sliding_spotlight_config': {
                        'beam_model': 'continuous',
                        'wavelength_m': 0.031,
                        'antenna_length_m': 10.0,
                        'vrc_distance_m': 2000000.0,
                        'prf_hz': 2500
                    },
                    'stripmap_config': {
                        'beam_model': 'continuous',
                        'wavelength_m': 0.031,
                        'antenna_length_m': 10.0,
                        'prf_hz': 1500
                    }
                }
            }
        }

        # 默认优先级：stripmap优先
        pc_default = PayloadConfiguration.from_dict(test_data)
        assert pc_default.has_stripmap_config("test_mode")
        assert not pc_default.has_sliding_spotlight_config("test_mode")
        assert not pc_default.has_spotlight_config("test_mode")

        # 自定义优先级：spotlight优先
        pc_custom = PayloadConfiguration.from_dict(
            test_data,
            config_priority=['spotlight_config', 'sliding_spotlight_config', 'stripmap_config']
        )
        assert pc_custom.has_spotlight_config("test_mode")
        assert not pc_custom.has_sliding_spotlight_config("test_mode")
        assert not pc_custom.has_stripmap_config("test_mode")
