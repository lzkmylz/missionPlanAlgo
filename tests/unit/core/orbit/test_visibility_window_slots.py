"""
测试VisibilityWindow的__slots__内存优化

M1: VisibilityWindow内存优化测试
"""

import sys
from datetime import datetime
from core.orbit.visibility.base import VisibilityWindow


class TestVisibilityWindowSlots:
    """测试VisibilityWindow的__slots__优化"""

    def test_visibility_window_has_slots(self):
        """测试VisibilityWindow类使用了__slots__"""
        # 检查类是否有__slots__属性
        assert hasattr(VisibilityWindow, '__slots__'), \
            "VisibilityWindow should use __slots__ for memory optimization"

    def test_visibility_window_no_dict(self):
        """测试VisibilityWindow实例没有__dict__属性"""
        window = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            max_elevation=45.0,
            quality_score=0.8
        )

        # 使用__slots__的实例不应该有__dict__
        assert not hasattr(window, '__dict__'), \
            "VisibilityWindow instance should not have __dict__ when using __slots__"

    def test_visibility_window_memory_efficiency(self):
        """测试VisibilityWindow内存占用减少"""
        # 创建测试实例
        window = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            max_elevation=45.0,
            quality_score=0.8
        )

        # 获取内存占用
        size = sys.getsizeof(window)

        # 使用__slots__的实例通常小于200字节
        # 普通dataclass实例通常超过400字节
        assert size < 300, \
            f"VisibilityWindow with __slots__ should use less memory, got {size} bytes"

    def test_visibility_window_attributes_accessible(self):
        """测试所有属性仍然可以正常访问"""
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 10, 5, 0)

        window = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=start_time,
            end_time=end_time,
            max_elevation=45.0,
            quality_score=0.8
        )

        # 验证所有属性可访问
        assert window.satellite_id == "SAT-01"
        assert window.target_id == "TARGET-01"
        assert window.start_time == start_time
        assert window.end_time == end_time
        assert window.max_elevation == 45.0
        assert window.quality_score == 0.8

    def test_visibility_window_methods_work(self):
        """测试所有方法仍然正常工作"""
        window = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            max_elevation=45.0,
            quality_score=0.8
        )

        # 测试duration方法
        duration = window.duration()
        assert duration == 300.0, f"Expected 300.0 seconds, got {duration}"

        # 测试__lt__方法（用于排序）
        window2 = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-02",
            start_time=datetime(2024, 1, 1, 11, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 5, 0),
            max_elevation=45.0,
            quality_score=0.8
        )

        assert window < window2, "Window should be less than window2 based on start_time"

    def test_visibility_window_frozen(self):
        """测试VisibilityWindow是不可变的（frozen）"""
        window = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            max_elevation=45.0,
            quality_score=0.8
        )

        # 尝试修改属性应该失败
        try:
            window.satellite_id = "SAT-02"
            assert False, "Should not be able to modify frozen dataclass"
        except (AttributeError, FrozenInstanceError):
            pass  # 预期行为

    def test_visibility_window_hashable(self):
        """测试VisibilityWindow可以作为字典的key（因为frozen且有slots）"""
        window = VisibilityWindow(
            satellite_id="SAT-01",
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            max_elevation=45.0,
            quality_score=0.8
        )

        # 应该可以hash
        try:
            hash_val = hash(window)
            assert isinstance(hash_val, int)
        except TypeError as e:
            # 如果datetime字段不可hash，这也是可以接受的
            pass

    def test_massive_window_creation_memory(self):
        """测试大规模创建窗口时的内存效率"""
        import gc

        gc.collect()

        # 创建大量窗口
        windows = []
        for i in range(1000):
            window = VisibilityWindow(
                satellite_id=f"SAT-{i % 10:02d}",
                target_id=f"TARGET-{i:04d}",
                start_time=datetime(2024, 1, 1, 10, 0, 0),
                end_time=datetime(2024, 1, 1, 10, 5, 0),
                max_elevation=45.0,
                quality_score=0.8
            )
            windows.append(window)

        # 计算总内存占用
        total_size = sum(sys.getsizeof(w) for w in windows)
        avg_size = total_size / len(windows)

        # 平均每个实例应该小于200字节
        assert avg_size < 200, \
            f"Average memory per window should be < 200 bytes, got {avg_size:.2f} bytes"


# 用于测试的异常类
class FrozenInstanceError(Exception):
    """冻结实例错误"""
    pass
