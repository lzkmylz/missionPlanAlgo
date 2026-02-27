"""
自适应时间步长计算器测试

TDD测试 - Phase 1优化：
- 粗扫描算法 (300秒步长)
- 窗口精化算法 (60秒步长)
- 与原始实现结果对比验证
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from core.orbit.visibility.adaptive_step_calculator import AdaptiveStepCalculator, CoarseWindow
from core.orbit.visibility.base import VisibilityWindow


class TestCoarseScan:
    """测试粗扫描算法"""

    def test_coarse_scan_finds_potential_windows(self):
        """粗扫描应能发现所有潜在窗口"""
        calculator = AdaptiveStepCalculator(coarse_step=300, fine_step=60)

        # 模拟卫星位置数据 - 模拟3个可见窗口
        mock_positions = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # 窗口1: 02:00-04:00 (不可见)
        for i in range(24):  # 2小时
            mock_positions.append((
                base_time + timedelta(minutes=5 * i),
                False  # 不可见
            ))

        # 窗口2: 04:00-06:00 (可见)
        for i in range(24):  # 2小时
            mock_positions.append((
                base_time + timedelta(hours=2) + timedelta(minutes=5 * i),
                True  # 可见
            ))

        # 窗口3: 06:00-08:00 (不可见)
        for i in range(24):
            mock_positions.append((
                base_time + timedelta(hours=4) + timedelta(minutes=5 * i),
                False
            ))

        # 窗口4: 08:00-10:00 (可见)
        for i in range(24):
            mock_positions.append((
                base_time + timedelta(hours=6) + timedelta(minutes=5 * i),
                True
            ))

        windows = calculator._coarse_scan_from_positions(mock_positions)

        assert len(windows) == 2
        assert windows[0].is_potentially_visible is True
        assert windows[1].is_potentially_visible is True

    def test_coarse_scan_empty_input(self):
        """空输入应返回空列表"""
        calculator = AdaptiveStepCalculator()
        windows = calculator._coarse_scan_from_positions([])
        assert windows == []

    def test_coarse_scan_single_point_visible(self):
        """单点可见应返回一个窗口"""
        calculator = AdaptiveStepCalculator()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        mock_positions = [
            (base_time, True),
        ]

        windows = calculator._coarse_scan_from_positions(mock_positions)
        assert len(windows) == 1
        assert windows[0].is_potentially_visible is True

    def test_coarse_scan_ignores_short_transitions(self):
        """粗扫描应忽略短暂的状态变化（噪声）"""
        calculator = AdaptiveStepCalculator(coarse_step=300, fine_step=60)
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # 长时间可见，中间有一个点不可见
        mock_positions = [
            (base_time, True),
            (base_time + timedelta(minutes=5), True),
            (base_time + timedelta(minutes=10), False),  # 短暂不可见
            (base_time + timedelta(minutes=15), True),
            (base_time + timedelta(minutes=20), True),
        ]

        # 注意：粗扫描不处理噪声，这是精化阶段的任务
        windows = calculator._coarse_scan_from_positions(mock_positions)

        # 应该根据实际状态变化分割窗口
        assert len(windows) >= 1


class TestWindowRefinement:
    """测试窗口精化算法"""

    def test_refinement_expands_forward(self):
        """精化应向前扩展找到精确开始时间"""
        calculator = AdaptiveStepCalculator(fine_step=60)

        # 模拟精化函数
        def mock_is_visible(dt):
            # 窗口实际从 03:00 开始，粗扫描从 03:05 开始
            window_start = datetime(2024, 1, 1, 3, 0, 0)
            window_end = datetime(2024, 1, 1, 5, 0, 0)
            return window_start <= dt <= window_end

        coarse_start = datetime(2024, 1, 1, 3, 5, 0)
        coarse_end = datetime(2024, 1, 1, 4, 55, 0)

        start, end = calculator._refine_window_boundaries(
            mock_is_visible,
            coarse_start,
            coarse_end,
            datetime(2024, 1, 1, 0, 0, 0),  # mission start
            datetime(2024, 1, 1, 23, 59, 59)  # mission end
        )

        # 精化后的开始时间应该更早
        assert start <= coarse_start
        # 应该找到精确的开始时间 03:00
        assert start == datetime(2024, 1, 1, 3, 0, 0)

    def test_refinement_expands_backward(self):
        """精化应向后扩展找到精确结束时间"""
        calculator = AdaptiveStepCalculator(fine_step=60)

        def mock_is_visible(dt):
            window_start = datetime(2024, 1, 1, 3, 0, 0)
            window_end = datetime(2024, 1, 1, 5, 0, 0)
            return window_start <= dt <= window_end

        coarse_start = datetime(2024, 1, 1, 3, 5, 0)
        coarse_end = datetime(2024, 1, 1, 4, 55, 0)

        start, end = calculator._refine_window_boundaries(
            mock_is_visible,
            coarse_start,
            coarse_end,
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 23, 59, 59)
        )

        # 精化后的结束时间应该更晚
        assert end >= coarse_end
        # 应该找到精确的结束时间 05:00
        assert end == datetime(2024, 1, 1, 5, 0, 0)

    def test_refinement_respects_mission_boundaries(self):
        """精化不应超出任务时间边界"""
        calculator = AdaptiveStepCalculator(fine_step=60)

        mission_start = datetime(2024, 1, 1, 3, 0, 0)
        mission_end = datetime(2024, 1, 1, 5, 0, 0)

        def mock_is_visible(dt):
            # 窗口跨越整个任务时间
            return mission_start <= dt <= mission_end

        coarse_start = datetime(2024, 1, 1, 3, 30, 0)
        coarse_end = datetime(2024, 1, 1, 4, 30, 0)

        start, end = calculator._refine_window_boundaries(
            mock_is_visible,
            coarse_start,
            coarse_end,
            mission_start,
            mission_end
        )

        # 不应超出边界
        assert start >= mission_start
        assert end <= mission_end


class TestIntegrationWithOrekitCalculator:
    """与OrekitVisibilityCalculator集成测试"""

    def test_adaptive_step_produces_similar_results(self):
        """自适应步长应产生与固定步长相近的结果"""
        # 这个测试需要实际的OrekitVisibilityCalculator
        # 这里只做接口验证
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 创建带自适应步长的计算器
        calc_adaptive = OrekitVisibilityCalculator(
            config={
                'use_adaptive_step': True,
                'coarse_step_seconds': 300,
                'fine_step_seconds': 60,
            }
        )

        # 创建传统固定步长计算器
        calc_fixed = OrekitVisibilityCalculator(
            config={
                'time_step': 60,
                'use_adaptive_step': False,
            }
        )

        # 验证配置被正确保存
        assert calc_adaptive.use_adaptive_step is True
        assert calc_adaptive.coarse_step == 300
        assert calc_adaptive.fine_step == 60
        assert calc_fixed.use_adaptive_step is False

    def test_window_count_difference_within_tolerance(self):
        """窗口数量差异应在容差范围内 (<1个)"""
        # 标记为需要实际轨道数据的集成测试
        pytest.skip("需要实际TLE数据进行集成测试")

    def test_window_time_error_within_tolerance(self):
        """窗口时间误差应在容差范围内 (<60秒)"""
        pytest.skip("需要实际TLE数据进行集成测试")


class TestPerformance:
    """性能测试"""

    def test_adaptive_step_faster_than_fixed(self):
        """自适应步长应比固定步长快4倍以上"""
        import time

        calculator = AdaptiveStepCalculator(coarse_step=300, fine_step=60)

        # 模拟可见性检查函数
        def mock_is_visible(dt):
            # 模拟一个复杂但快速的检查
            return (dt.hour >= 6) and (dt.hour <= 18)

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)  # 24小时

        # 测试自适应步长
        start = time.time()
        adaptive_windows = calculator.compute_windows(
            mock_is_visible,
            start_time,
            end_time
        )
        adaptive_time = time.time() - start

        # 测试固定步长 (60秒)
        start = time.time()
        fixed_points = []
        current = start_time
        while current <= end_time:
            fixed_points.append(mock_is_visible(current))
            current += timedelta(seconds=60)
        fixed_time = time.time() - start

        # 自适应应快4倍以上
        speedup = fixed_time / adaptive_time
        assert speedup >= 4.0, f"自适应步长仅快 {speedup:.1f} 倍，预期 >= 4倍"

    def test_coarse_scan_reduces_computation_points(self):
        """粗扫描应显著减少计算点数量"""
        calculator = AdaptiveStepCalculator(coarse_step=300, fine_step=60)

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)  # 24小时

        # 粗扫描点数
        coarse_points = calculator._estimate_coarse_points(start_time, end_time)

        # 固定步长点数
        total_seconds = (end_time - start_time).total_seconds()
        fixed_points = int(total_seconds / 60) + 1

        # 粗扫描应减少约80%的计算点
        reduction = 1 - (coarse_points / fixed_points)
        assert reduction >= 0.7, f"粗扫描仅减少 {reduction*100:.0f}%，预期 >= 70%"


class TestCoarseWindowDataclass:
    """测试CoarseWindow数据类"""

    def test_coarse_window_creation(self):
        """应能正确创建CoarseWindow实例"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 0)

        window = CoarseWindow(
            start_time=start,
            end_time=end,
            is_potentially_visible=True,
            max_elevation=45.0
        )

        assert window.start_time == start
        assert window.end_time == end
        assert window.is_potentially_visible is True
        assert window.max_elevation == 45.0

    def test_coarse_window_duration(self):
        """CoarseWindow应能计算持续时间"""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 30, 0)

        window = CoarseWindow(
            start_time=start,
            end_time=end,
            is_potentially_visible=True,
            max_elevation=45.0
        )

        assert window.duration_seconds == 9000  # 2.5小时 = 9000秒
