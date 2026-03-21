"""
SAR TOPSAR模式物理引擎单元测试

覆盖点：
  - TOPSARSubSwathPosition / SARTOPSARConfig dataclass 验证
  - 三种 beam_model 计算路径（continuous / discrete_beam / derived_beam）
  - TOPSAR核心物理公式验证（方位向分辨率、循环时间、子条带幅宽）
  - PRF约束可行性验证
  - select_subswath_position 匹配逻辑
  - PayloadConfiguration 加载解析
  - sar_2.json 中 TOPSAR 配置的端到端测试
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parents[4]

from core.models.sar_topsar_config import (
    TOPSARSubSwathPosition,
    SARTOPSARConfig,
    SARTOPSARResult,
    EARTH_RADIUS_M,
)
from core.dynamics.sar_topsar_calculator import (
    SARTOPSARCalculator,
    _azimuth_resolution_from_burst,
    _subswath_width_km,
    _check_prf_feasibility,
    _incidence_to_look,
)

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
H = 631_000.0   # m，典型LEO轨道高度
V = 7500.0      # m/s，卫星速度


# ---------------------------------------------------------------------------
# TOPSARSubSwathPosition 测试
# ---------------------------------------------------------------------------

class TestTOPSARSubSwathPosition:
    def test_minimal_construction(self):
        sp = TOPSARSubSwathPosition(subswath_id="IW1", center_incidence_angle_deg=29.0)
        assert sp.subswath_id == "IW1"
        assert sp.center_incidence_angle_deg == 29.0
        assert not sp.is_fully_specified

    def test_full_construction(self):
        sp = TOPSARSubSwathPosition(
            subswath_id="IW1",
            center_incidence_angle_deg=29.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=38.0,
            prf_hz=1500.0,
            range_resolution_m=5.0,
            azimuth_resolution_m=14.0,
            swath_width_rg_km=80.0,
            burst_duration_s=2.0,
        )
        assert sp.is_fully_specified

    def test_min_geq_max_raises(self):
        with pytest.raises(ValueError, match="incidence_angle_min"):
            TOPSARSubSwathPosition(
                subswath_id="X",
                center_incidence_angle_deg=30.0,
                incidence_angle_min_deg=40.0,
                incidence_angle_max_deg=35.0,
            )

    def test_zero_prf_raises(self):
        with pytest.raises(ValueError, match="prf_hz"):
            TOPSARSubSwathPosition(
                subswath_id="X",
                center_incidence_angle_deg=30.0,
                prf_hz=0.0,
            )

    def test_negative_swath_raises(self):
        with pytest.raises(ValueError, match="swath_width_rg_km"):
            TOPSARSubSwathPosition(
                subswath_id="X",
                center_incidence_angle_deg=30.0,
                swath_width_rg_km=-5.0,
            )

    def test_negative_burst_duration_raises(self):
        with pytest.raises(ValueError, match="burst_duration_s"):
            TOPSARSubSwathPosition(
                subswath_id="X",
                center_incidence_angle_deg=30.0,
                burst_duration_s=-1.0,
            )

    def test_covers_in_range(self):
        sp = TOPSARSubSwathPosition(
            subswath_id="IW1",
            center_incidence_angle_deg=29.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=38.0,
        )
        assert sp.covers(25.0)
        assert sp.covers(20.0)
        assert sp.covers(38.0)

    def test_covers_outside_range(self):
        sp = TOPSARSubSwathPosition(
            subswath_id="IW1",
            center_incidence_angle_deg=29.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=38.0,
        )
        assert not sp.covers(15.0)
        assert not sp.covers(45.0)

    def test_covers_none_bounds(self):
        sp = TOPSARSubSwathPosition(subswath_id="IW1", center_incidence_angle_deg=29.0)
        assert not sp.covers(29.0)

    def test_from_dict_full(self):
        d = {
            "subswath_id": "IW2",
            "center_incidence_angle_deg": 37.0,
            "incidence_angle_min_deg": 29.0,
            "incidence_angle_max_deg": 46.0,
            "prf_hz": 1600.0,
            "range_resolution_m": 5.0,
            "azimuth_resolution_m": 12.0,
            "swath_width_rg_km": 85.0,
            "burst_duration_s": 2.0,
        }
        sp = TOPSARSubSwathPosition.from_dict(d)
        assert sp.subswath_id == "IW2"
        assert sp.prf_hz == 1600.0
        assert sp.is_fully_specified

    def test_from_dict_backward_compat_beam_id(self):
        """向后兼容: 支持 beam_id 作为 subswath_id 别名"""
        d = {"beam_id": "IW3", "center_incidence_angle_deg": 45.0}
        sp = TOPSARSubSwathPosition.from_dict(d)
        assert sp.subswath_id == "IW3"

    def test_to_dict_roundtrip(self):
        sp = TOPSARSubSwathPosition(
            subswath_id="IW1",
            center_incidence_angle_deg=29.0,
            prf_hz=1500.0,
        )
        d = sp.to_dict()
        sp2 = TOPSARSubSwathPosition.from_dict(d)
        assert sp2.subswath_id == sp.subswath_id
        assert sp2.prf_hz == sp.prf_hz


# ---------------------------------------------------------------------------
# SARTOPSARConfig 测试
# ---------------------------------------------------------------------------

class TestSARTOPSARConfig:
    def test_default_construction(self):
        cfg = SARTOPSARConfig()
        assert cfg.beam_model == "continuous"
        assert cfg.num_subswaths == 3
        assert cfg.burst_duration_s == 2.0
        assert cfg.duty_cycle == 0.10

    def test_invalid_beam_model_raises(self):
        with pytest.raises(ValueError, match="beam_model"):
            SARTOPSARConfig(beam_model="unsupported")

    def test_num_subswaths_zero_raises(self):
        with pytest.raises(ValueError, match="num_subswaths"):
            SARTOPSARConfig(num_subswaths=0)

    def test_burst_duration_zero_raises(self):
        with pytest.raises(ValueError, match="burst_duration_s"):
            SARTOPSARConfig(burst_duration_s=0.0)

    def test_duty_cycle_out_of_range_raises(self):
        with pytest.raises(ValueError, match="duty_cycle"):
            SARTOPSARConfig(duty_cycle=0.0)
        with pytest.raises(ValueError, match="duty_cycle"):
            SARTOPSARConfig(duty_cycle=1.5)

    def test_discrete_beam_requires_positions(self):
        with pytest.raises(ValueError, match="requires at least one"):
            SARTOPSARConfig(beam_model="discrete_beam", subswath_positions=[])

    def test_derived_beam_requires_positions(self):
        with pytest.raises(ValueError, match="requires at least one"):
            SARTOPSARConfig(beam_model="derived_beam", subswath_positions=[])

    def test_discrete_beam_requires_full_positions(self):
        incomplete = TOPSARSubSwathPosition(
            subswath_id="IW1", center_incidence_angle_deg=29.0
        )
        with pytest.raises(ValueError, match="missing required fields"):
            SARTOPSARConfig(
                beam_model="discrete_beam",
                num_subswaths=1,  # 匹配提供的位置数量，直接触发 is_fully_specified 检查
                subswath_positions=[incomplete],
            )

    def test_num_subswaths_exceeds_positions_raises(self):
        pos = TOPSARSubSwathPosition(
            subswath_id="IW1", center_incidence_angle_deg=29.0
        )
        with pytest.raises(ValueError, match="exceeds provided"):
            SARTOPSARConfig(
                beam_model="derived_beam",
                num_subswaths=3,
                subswath_positions=[pos],  # 只有1个，需要3个
            )

    def test_continuous_no_positions_needed(self):
        cfg = SARTOPSARConfig(beam_model="continuous")
        assert cfg.beam_model == "continuous"

    def test_from_dict_continuous(self):
        d = {
            "beam_model": "continuous",
            "num_subswaths": 3,
            "burst_duration_s": 1.5,
        }
        cfg = SARTOPSARConfig.from_dict(d)
        assert cfg.burst_duration_s == 1.5

    def test_to_dict_roundtrip(self):
        cfg = SARTOPSARConfig(beam_model="continuous", num_subswaths=4, burst_duration_s=1.8)
        d = cfg.to_dict()
        cfg2 = SARTOPSARConfig.from_dict(d)
        assert cfg2.num_subswaths == 4
        assert cfg2.burst_duration_s == 1.8


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_azimuth_resolution_from_burst(self):
        """ρ_az = λ×R / (2×V×T_burst)"""
        wavelength = 0.031
        R = 770_000.0  # m
        T_burst = 2.0  # s
        rho = _azimuth_resolution_from_burst(wavelength, R, V, T_burst)
        expected = wavelength * R / (2.0 * V * T_burst)
        assert abs(rho - expected) < 0.01

    def test_azimuth_resolution_burst_zero_raises(self):
        with pytest.raises(ValueError):
            _azimuth_resolution_from_burst(0.031, 770_000.0, V, 0.0)

    def test_subswath_width_km(self):
        """W_g = (λ/L_w) × R / cos(θ)"""
        wavelength = 0.031
        L_w = 2.0
        R = 770_000.0
        cos_look = math.cos(math.radians(35.0))
        w = _subswath_width_km(wavelength, L_w, R, cos_look)
        expected = (wavelength / L_w) * R / cos_look / 1000.0
        assert abs(w - expected) < 0.01

    def test_check_prf_feasibility_ok(self):
        R = 770_000.0
        prf = 1500.0
        result = _check_prf_feasibility(prf, V, 10.0, R)
        assert result['feasible']

    def test_check_prf_feasibility_too_low(self):
        R = 770_000.0
        result = _check_prf_feasibility(100.0, V, 10.0, R)  # PRF=100Hz, 需要>>1500Hz
        assert not result['feasible']
        assert "低于" in result['reason']

    def test_check_prf_feasibility_large_prf_ok(self):
        """TOPSAR突发模式不严格限制PRF上限（不同于条带模式）"""
        R = 770_000.0
        prf_min = 2.0 * V / 10.0  # 1500 Hz
        result = _check_prf_feasibility(prf_min * 2.0, V, 10.0, R)
        assert result['feasible']

    def test_incidence_to_look_at_nadir(self):
        """零入射角应转换为零侧视角"""
        look = _incidence_to_look(H, 0.0)
        assert abs(look) < 0.1

    def test_incidence_to_look_typical(self):
        """典型入射角：约37° → 约35°侧视角"""
        look = _incidence_to_look(H, 37.0)
        assert 30.0 < look < 40.0


# ---------------------------------------------------------------------------
# SARTOPSARCalculator - 方案A（continuous）
# ---------------------------------------------------------------------------

class TestContinuousModel:
    def setup_method(self):
        self.cfg = SARTOPSARConfig(
            beam_model="continuous",
            num_subswaths=3,
            burst_duration_s=2.0,
            burst_switch_time_s=0.002,
            duty_cycle=0.10,
            prf_hz=1500.0,
            center_look_angle_deg=35.0,
            subswath_spacing_deg=6.0,
            min_look_angle_deg=20.0,
            max_look_angle_deg=50.0,
        )
        self.calc = SARTOPSARCalculator(self.cfg)

    def test_basic_feasible(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.feasible
        assert result.num_subswaths_used == 3
        assert result.burst_duration_s == 2.0

    def test_cycle_time_formula(self):
        """T_cycle = N × T_burst + (N-1) × T_switch"""
        result = self.calc.compute_burst_params(H, 35.0, V)
        N = 3
        expected_cycle = N * 2.0 + (N - 1) * 0.002
        assert abs(result.cycle_time_s - expected_cycle) < 0.001

    def test_azimuth_resolution_by_burst(self):
        """TOPSAR方位向分辨率受天线物理下限（L_a/2）保护"""
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.azimuth_resolution_m > 0
        # 分辨率不得优于天线物理下限 L_a/2（此时 T_burst>T_dwell_max，触发下限保护）
        assert result.azimuth_resolution_m >= self.cfg.antenna_length_m / 2.0

    def test_total_swath_width_proportional(self):
        """总幅宽 = N × 单子条带幅宽"""
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.total_swath_width_km > 0
        # 3子条带总幅宽应大于30km（实际约43km，取决于天线波束宽度）
        assert result.total_swath_width_km > 30.0

    def test_scene_area_positive(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.scene_area_km2 > 0.0

    def test_peak_power_factor(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert abs(result.peak_power_factor - 1.0 / self.cfg.duty_cycle) < 0.001

    def test_look_angle_out_of_range(self):
        result = self.calc.compute_burst_params(H, 60.0, V)
        assert not result.feasible
        assert "超出范围" in (result.reason or "")

    def test_subswath_results_count(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert len(result.subswath_results) == 3

    def test_compute_scene_coverage_same_as_burst_params(self):
        r1 = self.calc.compute_burst_params(H, 35.0, V)
        r2 = self.calc.compute_scene_coverage(H, 35.0, V)
        assert r1.feasible == r2.feasible
        assert abs(r1.total_swath_width_km - r2.total_swath_width_km) < 0.001

    def test_select_subswath_returns_none(self):
        """连续模型不使用子条带表"""
        assert self.calc.select_subswath_position(35.0) is None


# ---------------------------------------------------------------------------
# SARTOPSARCalculator - 方案B（discrete_beam）
# ---------------------------------------------------------------------------

def _make_discrete_positions():
    return [
        TOPSARSubSwathPosition(
            subswath_id="IW1",
            center_incidence_angle_deg=29.0,
            incidence_angle_min_deg=20.0,
            incidence_angle_max_deg=38.0,
            prf_hz=1500.0,
            range_resolution_m=5.0,
            azimuth_resolution_m=14.0,
            swath_width_rg_km=80.0,
            burst_duration_s=2.0,
        ),
        TOPSARSubSwathPosition(
            subswath_id="IW2",
            center_incidence_angle_deg=37.0,
            incidence_angle_min_deg=31.0,
            incidence_angle_max_deg=46.0,
            prf_hz=1600.0,
            range_resolution_m=5.0,
            azimuth_resolution_m=12.0,
            swath_width_rg_km=85.0,
            burst_duration_s=2.0,
        ),
        TOPSARSubSwathPosition(
            subswath_id="IW3",
            center_incidence_angle_deg=45.0,
            incidence_angle_min_deg=38.0,
            incidence_angle_max_deg=56.0,
            prf_hz=1700.0,
            range_resolution_m=5.0,
            azimuth_resolution_m=10.0,
            swath_width_rg_km=90.0,
            burst_duration_s=2.0,
        ),
    ]


class TestDiscreteBeamModel:
    def setup_method(self):
        positions = _make_discrete_positions()
        self.cfg = SARTOPSARConfig(
            beam_model="discrete_beam",
            num_subswaths=3,
            burst_duration_s=2.0,
            burst_switch_time_s=0.002,
            duty_cycle=0.10,
            subswath_positions=positions,
        )
        self.calc = SARTOPSARCalculator(self.cfg)

    def test_feasible_in_coverage(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.feasible
        assert result.matched_subswath_id is not None

    def test_matches_closest_subswath(self):
        """IW1覆盖20-38°，IW2覆盖31-46°, 入射角35°应匹配IW1（中心29°）或IW2（中心37°）"""
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.matched_subswath_id in ("IW1", "IW2")

    def test_infeasible_out_of_coverage(self):
        # 侧视角 15° → 入射角约 16°，低于IW1最小入射角 20°
        result = self.calc.compute_burst_params(H, 15.0, V)
        assert not result.feasible
        assert "无子条带" in (result.reason or "")

    def test_total_swath_from_positions(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        # 总幅宽 = IW1(80) + IW2(85) + IW3(90) = 255 km
        assert abs(result.total_swath_width_km - 255.0) < 0.01

    def test_subswath_results_per_position(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert len(result.subswath_results) == 3

    def test_select_subswath_iw1(self):
        # 入射角25°在IW1覆盖范围内
        sp = self.calc.select_subswath_position(25.0)
        assert sp is not None
        assert sp.subswath_id == "IW1"

    def test_select_subswath_iw3(self):
        # 入射角50°在IW3覆盖范围内
        sp = self.calc.select_subswath_position(50.0)
        assert sp is not None
        assert sp.subswath_id == "IW3"

    def test_select_subswath_none_outside(self):
        sp = self.calc.select_subswath_position(5.0)
        assert sp is None


# ---------------------------------------------------------------------------
# SARTOPSARCalculator - 方案C（derived_beam）
# ---------------------------------------------------------------------------

def _make_derived_positions():
    return [
        TOPSARSubSwathPosition(subswath_id="IW1", center_incidence_angle_deg=29.0),
        TOPSARSubSwathPosition(subswath_id="IW2", center_incidence_angle_deg=37.0),
        TOPSARSubSwathPosition(subswath_id="IW3", center_incidence_angle_deg=45.0),
    ]


class TestDerivedBeamModel:
    def setup_method(self):
        positions = _make_derived_positions()
        self.cfg = SARTOPSARConfig(
            beam_model="derived_beam",
            num_subswaths=3,
            burst_duration_s=2.0,
            burst_switch_time_s=0.002,
            duty_cycle=0.10,
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            range_resolution_m=5.0,
            prf_safety_factor=1.25,
            subswath_positions=positions,
        )
        self.calc = SARTOPSARCalculator(self.cfg)

    def test_feasible(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.feasible

    def test_derives_prf(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert result.prf_hz_used > 0.0
        # 推导的PRF应满足奈奎斯特：PRF ≥ 2×V/L_az = 2×7500/10 = 1500 Hz
        assert result.prf_hz_used >= 1500.0 * 0.99  # 1%容差

    def test_azimuth_resolution_from_burst(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        # ρ_az = max(λ×R/(2×V×T_burst), L_a/2)
        R = H / math.cos(math.radians(35.0))
        expected_rho = _azimuth_resolution_from_burst(0.031, R, V, 2.0, antenna_length_m=10.0)
        assert abs(result.azimuth_resolution_m - expected_rho) < 0.1

    def test_fills_subswath_params(self):
        """方案C应回填计算器内部config副本的子条带推导参数（deepcopy隔离，不影响原始cfg）"""
        self.calc.compute_burst_params(H, 35.0, V)
        # 检查计算器自身的config副本（而非原始self.cfg，后者不再被修改）
        for sp in self.calc.config.subswath_positions:
            assert sp.prf_hz is not None
            assert sp.range_resolution_m is not None
            assert sp.azimuth_resolution_m is not None
            assert sp.swath_width_rg_km is not None
            assert sp.burst_duration_s is not None
        # 原始cfg的位置不受影响（deepcopy保证隔离）
        for sp in self.cfg.subswath_positions:
            assert sp.prf_hz is None, "deepcopy后原始config不应被修改"

    def test_fills_angle_bounds(self):
        """方案C应回填计算器内部config副本的入射角覆盖范围"""
        self.calc.compute_burst_params(H, 35.0, V)
        for sp in self.calc.config.subswath_positions:
            assert sp.incidence_angle_min_deg is not None
            assert sp.incidence_angle_max_deg is not None
            assert sp.incidence_angle_min_deg < sp.center_incidence_angle_deg
            assert sp.incidence_angle_max_deg > sp.center_incidence_angle_deg

    def test_select_subswath_closest(self):
        """方案C选最近中心角"""
        sp = self.calc.select_subswath_position(38.0)
        assert sp is not None
        # 38°与IW2(37°)距离=1°，与IW3(45°)距离=7°，应选IW2
        assert sp.subswath_id == "IW2"

    def test_cycle_time_correct(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        N = 3
        expected = N * 2.0 + (N - 1) * 0.002
        assert abs(result.cycle_time_s - expected) < 0.001

    def test_subswath_results_count(self):
        result = self.calc.compute_burst_params(H, 35.0, V)
        assert len(result.subswath_results) == 3

    def test_param_fill_is_idempotent(self):
        """多次调用不改变已填充的参数"""
        self.calc.compute_burst_params(H, 35.0, V)
        sp0_prf = self.cfg.subswath_positions[0].prf_hz
        self.calc.compute_burst_params(H, 35.0, V)
        assert self.cfg.subswath_positions[0].prf_hz == sp0_prf


# ---------------------------------------------------------------------------
# 物理公式一致性验证
# ---------------------------------------------------------------------------

class TestPhysicalConsistency:
    def setup_method(self):
        positions = _make_derived_positions()
        cfg = SARTOPSARConfig(
            beam_model="derived_beam",
            num_subswaths=3,
            burst_duration_s=2.0,
            burst_switch_time_s=0.002,
            duty_cycle=0.10,
            wavelength_m=0.031,
            antenna_length_m=10.0,
            antenna_width_m=2.0,
            range_resolution_m=5.0,
            prf_safety_factor=1.25,
            subswath_positions=positions,
        )
        self.calc = SARTOPSARCalculator(cfg)
        self.cfg = cfg

    def test_scene_az_km_from_burst(self):
        """方位向场景长度 = V × T_burst（单次突发）"""
        result = self.calc.compute_burst_params(H, 35.0, V)
        expected = V * self.cfg.burst_duration_s / 1000.0
        assert abs(result.scene_size_az_km - expected) < 0.01

    def test_scene_area_product(self):
        """面积 = 方位向 × 距离向总幅宽"""
        result = self.calc.compute_burst_params(H, 35.0, V)
        expected = result.scene_size_az_km * result.total_swath_width_km
        assert abs(result.scene_area_km2 - expected) < 0.1

    def test_more_subswaths_larger_swath(self):
        """更多子条带 → 更大总幅宽"""
        pos3 = _make_derived_positions()
        pos2 = pos3[:2]

        cfg3 = SARTOPSARConfig(
            beam_model="derived_beam", num_subswaths=3,
            burst_duration_s=2.0, burst_switch_time_s=0.002,
            duty_cycle=0.10, prf_safety_factor=1.25,
            subswath_positions=list(pos3),
        )
        cfg2 = SARTOPSARConfig(
            beam_model="derived_beam", num_subswaths=2,
            burst_duration_s=2.0, burst_switch_time_s=0.002,
            duty_cycle=0.10, prf_safety_factor=1.25,
            subswath_positions=list(pos2),
        )
        r3 = SARTOPSARCalculator(cfg3).compute_burst_params(H, 35.0, V)
        r2 = SARTOPSARCalculator(cfg2).compute_burst_params(H, 35.0, V)
        assert r3.total_swath_width_km > r2.total_swath_width_km

    def test_longer_burst_better_resolution(self):
        """更长突发时长 → 更高方位向分辨率（更小ρ_az），使用短于T_dwell_max的值"""
        # T_dwell_max ≈ 0.318s（H=631km, V=7500, L_a=10m），使用0.05s和0.15s确保公式有效
        pos_a = _make_derived_positions()
        pos_b = _make_derived_positions()

        cfg_short = SARTOPSARConfig(
            beam_model="derived_beam", num_subswaths=3,
            burst_duration_s=0.05, duty_cycle=0.10,
            prf_safety_factor=1.25, subswath_positions=pos_a,
        )
        cfg_long = SARTOPSARConfig(
            beam_model="derived_beam", num_subswaths=3,
            burst_duration_s=0.15, duty_cycle=0.10,
            prf_safety_factor=1.25, subswath_positions=pos_b,
        )
        r_short = SARTOPSARCalculator(cfg_short).compute_burst_params(H, 35.0, V)
        r_long = SARTOPSARCalculator(cfg_long).compute_burst_params(H, 35.0, V)
        # 更长突发时长 → 更小分辨率数值（更高分辨率）
        assert r_long.azimuth_resolution_m < r_short.azimuth_resolution_m


# ---------------------------------------------------------------------------
# PayloadConfiguration 集成测试
# ---------------------------------------------------------------------------

class TestPayloadConfigurationIntegration:
    def test_parse_topsar_config(self):
        from core.models.payload_config import PayloadConfiguration

        data = {
            "payload_type": "sar",
            "default_mode": "topsar",
            "modes": {
                "topsar": {
                    "resolution_m": 5.0,
                    "swath_width_m": 100000,
                    "power_consumption_w": 500.0,
                    "data_rate_mbps": 600.0,
                    "min_duration_s": 5.0,
                    "max_duration_s": 60.0,
                    "mode_type": "sar",
                    "topsar_config": {
                        "beam_model": "derived_beam",
                        "wavelength_m": 0.031,
                        "antenna_length_m": 10.0,
                        "antenna_width_m": 2.0,
                        "range_resolution_m": 5.0,
                        "num_subswaths": 3,
                        "burst_duration_s": 2.0,
                        "burst_switch_time_s": 0.002,
                        "duty_cycle": 0.10,
                        "prf_safety_factor": 1.25,
                        "subswath_positions": [
                            {"subswath_id": "IW1", "center_incidence_angle_deg": 29.0},
                            {"subswath_id": "IW2", "center_incidence_angle_deg": 37.0},
                            {"subswath_id": "IW3", "center_incidence_angle_deg": 45.0},
                        ],
                    },
                }
            },
        }

        pc = PayloadConfiguration.from_dict(data)
        assert pc.has_topsar_config("topsar")
        calc = pc.get_topsar_calculator("topsar")
        assert calc is not None

    def test_topsar_calculator_result(self):
        from core.models.payload_config import PayloadConfiguration

        data = {
            "payload_type": "sar",
            "default_mode": "topsar",
            "modes": {
                "topsar": {
                    "resolution_m": 5.0,
                    "swath_width_m": 100000,
                    "power_consumption_w": 500.0,
                    "data_rate_mbps": 600.0,
                    "min_duration_s": 5.0,
                    "max_duration_s": 60.0,
                    "mode_type": "sar",
                    "topsar_config": {
                        "beam_model": "continuous",
                        "num_subswaths": 3,
                        "burst_duration_s": 2.0,
                        "burst_switch_time_s": 0.002,
                        "duty_cycle": 0.10,
                        "prf_hz": 1500.0,
                        "center_look_angle_deg": 35.0,
                        "subswath_spacing_deg": 6.0,
                        "min_look_angle_deg": 20.0,
                        "max_look_angle_deg": 50.0,
                    },
                }
            },
        }

        pc = PayloadConfiguration.from_dict(data)
        calc = pc.get_topsar_calculator("topsar")
        result = calc.compute_burst_params(H, 35.0, V)
        assert result.feasible
        assert result.num_subswaths_used == 3

    def test_topsar_calculator_cached(self):
        """两次调用应返回相同实例（缓存）"""
        from core.models.payload_config import PayloadConfiguration

        data = {
            "payload_type": "sar",
            "default_mode": "topsar",
            "modes": {
                "topsar": {
                    "resolution_m": 5.0,
                    "swath_width_m": 100000,
                    "power_consumption_w": 500.0,
                    "data_rate_mbps": 600.0,
                    "min_duration_s": 5.0,
                    "max_duration_s": 60.0,
                    "mode_type": "sar",
                    "topsar_config": {
                        "beam_model": "continuous",
                        "num_subswaths": 3,
                        "burst_duration_s": 2.0,
                        "duty_cycle": 0.10,
                        "prf_hz": 1500.0,
                        "min_look_angle_deg": 20.0,
                        "max_look_angle_deg": 50.0,
                    },
                }
            },
        }

        pc = PayloadConfiguration.from_dict(data)
        c1 = pc.get_topsar_calculator("topsar")
        c2 = pc.get_topsar_calculator("topsar")
        assert c1 is c2

    def test_no_topsar_config_returns_none(self):
        from core.models.payload_config import PayloadConfiguration

        data = {
            "payload_type": "sar",
            "default_mode": "stripmap",
            "modes": {
                "stripmap": {
                    "resolution_m": 3.0,
                    "swath_width_m": 30000,
                    "power_consumption_w": 300.0,
                    "data_rate_mbps": 400.0,
                    "min_duration_s": 10.0,
                    "max_duration_s": 60.0,
                    "mode_type": "sar",
                }
            },
        }

        pc = PayloadConfiguration.from_dict(data)
        assert not pc.has_topsar_config("topsar")
        assert pc.get_topsar_calculator("topsar") is None


# ---------------------------------------------------------------------------
# sar_2.json 端到端测试
# ---------------------------------------------------------------------------

class TestSAR2JsonEndToEnd:
    @pytest.fixture(autouse=True)
    def load_config(self):
        json_path = PROJECT_ROOT / "data" / "entity_lib" / "satellites" / "sar_2.json"
        with open(json_path) as f:
            self.raw = json.load(f)
        self.payload_raw = self.raw["capabilities"]["payload_config"]

    def test_topsar_mode_exists(self):
        assert "topsar" in self.payload_raw["modes"]

    def test_topsar_config_parseable(self):
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(self.payload_raw)
        assert pc.has_topsar_config("topsar")

    def test_topsar_calculator_derived(self):
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(self.payload_raw)
        calc = pc.get_topsar_calculator("topsar")
        result = calc.compute_burst_params(H, 35.0, V)
        assert result.feasible
        assert result.num_subswaths_used == 3
        assert result.total_swath_width_km > 0
        assert result.burst_duration_s == 0.25

    def test_topsar_cycle_time(self):
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(self.payload_raw)
        calc = pc.get_topsar_calculator("topsar")
        result = calc.compute_burst_params(H, 35.0, V)
        # T_cycle = 3 × 0.25 + 2 × 0.002 = 0.754秒
        assert abs(result.cycle_time_s - 0.754) < 0.001

    def test_topsar_subswath_results(self):
        from core.models.payload_config import PayloadConfiguration
        pc = PayloadConfiguration.from_dict(self.payload_raw)
        calc = pc.get_topsar_calculator("topsar")
        result = calc.compute_burst_params(H, 35.0, V)
        assert len(result.subswath_results) == 3
        for sw in result.subswath_results:
            assert "subswath_id" in sw
            assert sw["burst_duration_s"] == 0.25
