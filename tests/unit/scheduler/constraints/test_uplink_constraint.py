"""
指令上注约束单元测试

覆盖范围:
  - UplinkPass 数据类属性
  - UplinkWindowRegistry 三渠道加载与查询（含感知时区窗口归一化）
  - BatchUplinkCalculator 批量可行性检查（Python 路径 + Numba 大批量路径）
  - BatchUplinkConstraintChecker resolve_uplink_duration 与批量检查
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from scheduler.constraints.uplink_channel_type import UplinkChannelType, UplinkPass
from scheduler.constraints.uplink_window_registry import UplinkWindowRegistry
from scheduler.constraints.batch_uplink_calculator import (
    BatchUplinkCalculator,
    BatchUplinkCandidate,
    BatchUplinkResult,
    HAS_NUMBA,
)
from scheduler.constraints.batch_uplink_constraint_checker import BatchUplinkConstraintChecker


# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------

def _make_pass(
    channel_type: UplinkChannelType,
    channel_id: str,
    sat_id: str,
    end_offset_s: float,       # 相对 BASE_TIME 的偏移（秒），负值表示过去
    duration_s: float = 120.0,
    overhead_s: float = 30.0,
) -> UplinkPass:
    base = datetime(2026, 3, 21, 0, 0, 0)
    end_time = base + timedelta(seconds=end_offset_s)
    start_time = end_time - timedelta(seconds=duration_s)
    return UplinkPass(
        channel_type=channel_type,
        channel_id=channel_id,
        satellite_id=sat_id,
        start_time=start_time,
        end_time=end_time,
        switching_overhead_s=overhead_s,
    )


BASE_TIME = datetime(2026, 3, 21, 0, 0, 0)
TASK_START = BASE_TIME + timedelta(hours=1)   # 任务在 T+1h 开始
LEAD_TIME_S = 300.0                           # 指令至少提前 300s 到达
DEADLINE = TASK_START - timedelta(seconds=LEAD_TIME_S)


def _make_window_cache(entries: list):
    """构造简单 mock window_cache，entries = [(sat_id, target_id, [win, ...])]"""
    cache = MagicMock()
    windows_dict = {}
    for sat_id, target_id, wins in entries:
        windows_dict[(sat_id, target_id)] = wins
    cache._windows = windows_dict
    # 同时支持 get_all_windows() 公共接口（UplinkWindowRegistry 优先调用此方法）
    cache.get_all_windows.return_value = windows_dict
    return cache


def _make_win(start: datetime, end: datetime, **kwargs):
    """构造简单 mock window 对象"""
    w = MagicMock()
    w.start_time = start
    w.end_time = end
    for k, v in kwargs.items():
        setattr(w, k, v)
    return w


# ---------------------------------------------------------------------------
# UplinkPass 属性测试
# ---------------------------------------------------------------------------

class TestUplinkPass:
    def test_duration_s(self):
        p = _make_pass(UplinkChannelType.GROUND_STATION, 'GS-BJ', 'SAT-01', 0, duration_s=200.0)
        assert p.duration_s == pytest.approx(200.0)

    def test_usable_duration_s(self):
        p = _make_pass(UplinkChannelType.GROUND_STATION, 'GS-BJ', 'SAT-01', 0,
                       duration_s=120.0, overhead_s=30.0)
        assert p.usable_duration_s == pytest.approx(90.0)

    def test_usable_duration_never_negative(self):
        p = _make_pass(UplinkChannelType.ISL, 'SAT-02', 'SAT-01', 0,
                       duration_s=10.0, overhead_s=30.0)
        assert p.usable_duration_s == 0.0

    def test_capacity_mb(self):
        p = _make_pass(UplinkChannelType.RELAY_SATELLITE, 'RELAY-1', 'SAT-01', 0,
                       duration_s=130.0, overhead_s=30.0)
        p.max_data_rate_mbps = 10.0
        # usable=100s, rate=10Mbps → 100*10/8 = 125 MB
        assert p.capacity_mb == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# UplinkWindowRegistry 测试
# ---------------------------------------------------------------------------

class TestUplinkWindowRegistry:

    def _build_registry(self, entries):
        reg = UplinkWindowRegistry()
        cache = _make_window_cache(entries)
        reg.load_from_window_cache(cache)
        return reg

    def test_load_gs_windows(self):
        win = _make_win(BASE_TIME - timedelta(hours=2), BASE_TIME - timedelta(hours=1))
        reg = self._build_registry([('SAT-01', 'GS:GS-BEIJING', [win])])
        passes = reg.get_passes_for_satellite('SAT-01')
        assert len(passes) == 1
        assert passes[0].channel_type == UplinkChannelType.GROUND_STATION
        assert passes[0].channel_id == 'GS-BEIJING'

    def test_load_relay_windows(self):
        win = _make_win(BASE_TIME - timedelta(hours=2), BASE_TIME - timedelta(hours=1))
        reg = self._build_registry([('SAT-02', 'RELAY:RELAY-01', [win])])
        passes = reg.get_passes_for_satellite('SAT-02')
        assert len(passes) == 1
        assert passes[0].channel_type == UplinkChannelType.RELAY_SATELLITE
        assert passes[0].channel_id == 'RELAY-01'

    def test_load_isl_windows(self):
        win = _make_win(BASE_TIME - timedelta(hours=2), BASE_TIME - timedelta(hours=1))
        win.max_data_rate = 10000.0
        reg = self._build_registry([('SAT-03', 'ISL:SAT-04', [win])])
        passes = reg.get_passes_for_satellite('SAT-03')
        assert len(passes) == 1
        assert passes[0].channel_type == UplinkChannelType.ISL
        assert passes[0].switching_overhead_s == pytest.approx(5.0)

    def test_imaging_target_windows_ignored(self):
        win = _make_win(BASE_TIME, BASE_TIME + timedelta(hours=1))
        reg = self._build_registry([('SAT-01', 'TGT-0001', [win])])
        assert reg.pass_count('SAT-01') == 0

    def test_find_feasible_pass_gs(self):
        # 弧段结束在 deadline 之前、持续时间充足
        end = DEADLINE - timedelta(seconds=10)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        reg = self._build_registry([('SAT-01', 'GS:GS-BJ', [win])])
        p = reg.find_feasible_pass('SAT-01', DEADLINE, min_duration_s=30.0)
        assert p is not None
        assert p.channel_type == UplinkChannelType.GROUND_STATION

    def test_find_feasible_pass_after_deadline_rejected(self):
        # 弧段结束在 deadline 之后 → 不满足
        end = DEADLINE + timedelta(seconds=60)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        reg = self._build_registry([('SAT-01', 'GS:GS-BJ', [win])])
        p = reg.find_feasible_pass('SAT-01', DEADLINE, min_duration_s=30.0)
        assert p is None

    def test_find_feasible_pass_too_short_rejected(self):
        # 弧段可用时长不足（120s - 30s overhead = 90s < 120s required）
        end = DEADLINE - timedelta(seconds=10)
        start = end - timedelta(seconds=120)
        win = _make_win(start, end)
        reg = self._build_registry([('SAT-01', 'GS:GS-BJ', [win])])
        p = reg.find_feasible_pass('SAT-01', DEADLINE, min_duration_s=120.0)
        assert p is None

    def test_find_feasible_pass_with_aware_end_times(self):
        """感知时间（aware datetime）窗口不应引发 TypeError。
        Orekit Java 后端产出的窗口携带 tzinfo=UTC；load_from_window_cache
        应将其归一化为 naive UTC，使 bisect 比较与 naive deadline 兼容。
        """
        end_aware = (DEADLINE - timedelta(seconds=10)).replace(tzinfo=timezone.utc)
        start_aware = (end_aware - timedelta(seconds=150)).replace(tzinfo=timezone.utc)
        win = _make_win(start_aware, end_aware)
        reg = self._build_registry([('SAT-01', 'GS:GS-BJ', [win])])
        # DEADLINE 是 naive；若未归一化会抛出 TypeError
        p = reg.find_feasible_pass('SAT-01', DEADLINE, min_duration_s=30.0)
        assert p is not None
        assert p.channel_type == UplinkChannelType.GROUND_STATION
        # 存储的 end_time 应已被归一化为 naive
        assert p.end_time.tzinfo is None

    def test_find_feasible_pass_with_aware_deadline(self):
        """MEDIUM-3: find_feasible_pass 直接收到 aware deadline 时不应引发 TypeError。
        公共 API 应对调用方传入的 aware datetime 保持健壮。
        """
        end = DEADLINE - timedelta(seconds=10)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        reg = self._build_registry([('SAT-01', 'GS:GS-BJ', [win])])
        # 传入 aware deadline（而非 naive），模拟外部调用方的场景
        aware_deadline = DEADLINE.replace(tzinfo=timezone.utc)
        p = reg.find_feasible_pass('SAT-01', aware_deadline, min_duration_s=30.0)
        assert p is not None
        assert p.channel_type == UplinkChannelType.GROUND_STATION

    def test_channel_priority_prefers_gs_over_relay(self):
        end = DEADLINE - timedelta(seconds=10)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        reg = self._build_registry([
            ('SAT-01', 'GS:GS-BJ', [win]),
            ('SAT-01', 'RELAY:RELAY-01', [win]),
        ])
        p = reg.find_feasible_pass('SAT-01', DEADLINE, min_duration_s=30.0,
                                   channel_priority=['ground_station', 'relay_satellite'])
        assert p.channel_type == UplinkChannelType.GROUND_STATION

    def test_channel_priority_custom_order(self):
        end = DEADLINE - timedelta(seconds=10)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        win.max_data_rate = 10000.0
        reg = self._build_registry([
            ('SAT-01', 'GS:GS-BJ', [win]),
            ('SAT-01', 'ISL:SAT-02', [win]),
        ])
        # ISL 优先
        p = reg.find_feasible_pass('SAT-01', DEADLINE, min_duration_s=30.0,
                                   channel_priority=['isl', 'ground_station'])
        assert p.channel_type == UplinkChannelType.ISL

    def test_no_satellite_returns_none(self):
        reg = UplinkWindowRegistry()
        assert reg.find_feasible_pass('SAT-99', DEADLINE, 30.0) is None

    def test_empty_registry(self):
        reg = UplinkWindowRegistry()
        assert reg.is_empty()
        assert reg.pass_count() == 0


# ---------------------------------------------------------------------------
# BatchUplinkCalculator 测试
# ---------------------------------------------------------------------------

class TestBatchUplinkCalculator:

    def _make_registry_with_gs_pass(self, end_offset_s: float = -600.0) -> UplinkWindowRegistry:
        """创建含单条 GS 弧段的注册表，弧段结束在 TASK_START + end_offset_s 处"""
        end = TASK_START + timedelta(seconds=end_offset_s)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        reg = UplinkWindowRegistry()
        cache = _make_window_cache([('SAT-01', 'GS:GS-BJ', [win])])
        reg.load_from_window_cache(cache)
        return reg

    def test_feasible_candidate(self):
        reg = self._make_registry_with_gs_pass(-600)  # 弧段结束于 T-600s, lead=300 → deadline=T-300
        calc = BatchUplinkCalculator()
        cands = [BatchUplinkCandidate('SAT-01', TASK_START, 30.0, 300.0)]
        results = calc.check_batch(cands, reg)
        assert results[0].feasible is True
        assert results[0].channel_type == 'ground_station'

    def test_infeasible_pass_after_deadline(self):
        # 弧段结束于 T-100s，deadline = T-300s → 弧段在 deadline 之后
        reg = self._make_registry_with_gs_pass(-100)
        calc = BatchUplinkCalculator()
        cands = [BatchUplinkCandidate('SAT-01', TASK_START, 30.0, 300.0)]
        results = calc.check_batch(cands, reg)
        assert results[0].feasible is False

    def test_empty_candidates(self):
        reg = UplinkWindowRegistry()
        calc = BatchUplinkCalculator()
        results = calc.check_batch([], reg)
        assert results == []

    def test_no_passes_for_satellite(self):
        reg = UplinkWindowRegistry()
        calc = BatchUplinkCalculator()
        cands = [BatchUplinkCandidate('SAT-99', TASK_START, 30.0, 300.0)]
        results = calc.check_batch(cands, reg)
        assert results[0].feasible is False

    def test_batch_mixed_results(self):
        """多候选混合结果"""
        end_ok = TASK_START + timedelta(seconds=-600)
        start_ok = end_ok - timedelta(seconds=150)
        win_ok = _make_win(start_ok, end_ok)

        end_bad = TASK_START + timedelta(seconds=-100)
        start_bad = end_bad - timedelta(seconds=150)
        win_bad = _make_win(start_bad, end_bad)

        cache = _make_window_cache([
            ('SAT-01', 'GS:GS-BJ', [win_ok]),
            ('SAT-02', 'GS:GS-BJ', [win_bad]),
        ])
        reg = UplinkWindowRegistry()
        reg.load_from_window_cache(cache)

        calc = BatchUplinkCalculator()
        cands = [
            BatchUplinkCandidate('SAT-01', TASK_START, 30.0, 300.0),
            BatchUplinkCandidate('SAT-02', TASK_START, 30.0, 300.0),
        ]
        results = calc.check_batch(cands, reg)
        assert results[0].feasible is True
        assert results[1].feasible is False

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not available")
    def test_large_batch_path_correctness(self):
        """批次 >= NUMBA_THRESHOLD 时 Numba 大批量路径结果正确（需要 Numba 安装）"""
        end_ok = TASK_START + timedelta(seconds=-600)
        start_ok = end_ok - timedelta(seconds=150)
        win_ok = _make_win(start_ok, end_ok)

        end_bad = TASK_START + timedelta(seconds=-100)
        start_bad = end_bad - timedelta(seconds=150)
        win_bad = _make_win(start_bad, end_bad)

        entries = []
        for i in range(10):
            # 偶数卫星有可用弧段，奇数卫星没有
            if i % 2 == 0:
                entries.append((f'SAT-{i:02d}', 'GS:GS-BJ', [win_ok]))
            else:
                entries.append((f'SAT-{i:02d}', 'GS:GS-BJ', [win_bad]))

        reg = UplinkWindowRegistry()
        reg.load_from_window_cache(_make_window_cache(entries))

        # 构造 12 个候选，超过 NUMBA_THRESHOLD=10
        cands = [
            BatchUplinkCandidate(f'SAT-{i:02d}', TASK_START, 30.0, 300.0)
            for i in range(10)
        ] + [
            BatchUplinkCandidate('SAT-00', TASK_START, 30.0, 300.0),
            BatchUplinkCandidate('SAT-01', TASK_START, 30.0, 300.0),
        ]

        calc = BatchUplinkCalculator()
        results = calc.check_batch(cands, reg)

        # 验证结果正确性（偶数卫星可行，奇数不可行）
        for i in range(10):
            expected = (i % 2 == 0)
            assert results[i].feasible == expected, (
                f"SAT-{i:02d}: expected feasible={expected}, got {results[i].feasible}"
            )
        # 最后两个额外候选与前面一致
        assert results[10].feasible is True   # SAT-00
        assert results[11].feasible is False  # SAT-01


# ---------------------------------------------------------------------------
# BatchUplinkConstraintChecker 测试
# ---------------------------------------------------------------------------

class TestBatchUplinkConstraintChecker:

    def _make_satellite(self, min_uplink_s: float = 5.0) -> MagicMock:
        sat = MagicMock()
        sat.capabilities.min_uplink_duration_per_task = min_uplink_s
        sat.capabilities.payload_config = None
        sat.payload_config = None
        return sat

    def test_resolve_uses_satellite_default(self):
        reg = UplinkWindowRegistry()
        checker = BatchUplinkConstraintChecker(reg)
        sat = self._make_satellite(8.0)
        duration = checker.resolve_uplink_duration(sat, imaging_mode=None)
        assert duration == pytest.approx(8.0)

    def test_resolve_uses_mode_override(self):
        reg = UplinkWindowRegistry()
        checker = BatchUplinkConstraintChecker(reg)
        sat = self._make_satellite(5.0)
        mode_cfg = MagicMock()
        mode_cfg.get_uplink_duration_s.return_value = 15.0
        payload_cfg = MagicMock()
        payload_cfg.get_mode_config.return_value = mode_cfg
        sat.capabilities.payload_config = payload_cfg
        sat.payload_config = payload_cfg

        duration = checker.resolve_uplink_duration(sat, imaging_mode='topsar')
        assert duration == pytest.approx(15.0)

    def test_resolve_falls_back_when_mode_override_none(self):
        reg = UplinkWindowRegistry()
        checker = BatchUplinkConstraintChecker(reg)
        sat = self._make_satellite(7.0)
        mode_cfg = MagicMock()
        mode_cfg.get_uplink_duration_s.return_value = None
        payload_cfg = MagicMock()
        payload_cfg.get_mode_config.return_value = mode_cfg
        sat.capabilities.payload_config = payload_cfg
        sat.payload_config = payload_cfg

        duration = checker.resolve_uplink_duration(sat, imaging_mode='push_broom')
        assert duration == pytest.approx(7.0)

    def test_empty_registry_returns_infeasible(self):
        reg = UplinkWindowRegistry()
        checker = BatchUplinkConstraintChecker(reg)
        cands = [BatchUplinkCandidate('SAT-01', TASK_START, 30.0, 300.0)]
        results = checker.check_uplink_feasibility_batch(cands)
        assert results[0].feasible is False
        assert 'empty' in results[0].reason

    def test_feasible_with_gs_pass(self):
        end = TASK_START - timedelta(seconds=600)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        cache = _make_window_cache([('SAT-01', 'GS:GS-BJ', [win])])
        reg = UplinkWindowRegistry()
        reg.load_from_window_cache(cache)

        checker = BatchUplinkConstraintChecker(reg)
        cands = [BatchUplinkCandidate('SAT-01', TASK_START, 30.0, 300.0)]
        results = checker.check_uplink_feasibility_batch(cands)
        assert results[0].feasible is True

    def test_check_single_convenience(self):
        end = TASK_START - timedelta(seconds=600)
        start = end - timedelta(seconds=150)
        win = _make_win(start, end)
        cache = _make_window_cache([('SAT-01', 'GS:GS-BJ', [win])])
        reg = UplinkWindowRegistry()
        reg.load_from_window_cache(cache)

        checker = BatchUplinkConstraintChecker(reg)
        result = checker.check_single('SAT-01', TASK_START, required_uplink_s=30.0,
                                      command_lead_time_s=300.0)
        assert result.feasible is True
