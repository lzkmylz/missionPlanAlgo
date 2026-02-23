"""
上行调度器测试

测试第17章设计的UplinkScheduler和UplinkWindow
遵循TDD原则：先写测试，再实现代码
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any


class TestUplinkWindow:
    """测试UplinkWindow数据类"""

    def test_uplink_window_creation(self):
        """测试UplinkWindow基本创建"""
        from core.network.uplink_scheduler import UplinkWindow

        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15),
            max_data_rate_mbps=10.0,
            elevation_angle=45.0
        )

        assert window.ground_station_id == "gs_001"
        assert window.start_time == datetime(2024, 1, 1, 10, 0)
        assert window.end_time == datetime(2024, 1, 1, 10, 15)
        assert window.max_data_rate_mbps == 10.0
        assert window.elevation_angle == 45.0

    def test_uplink_window_duration(self):
        """测试UplinkWindow时长计算"""
        from core.network.uplink_scheduler import UplinkWindow

        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15),
            max_data_rate_mbps=10.0
        )

        assert window.duration == timedelta(minutes=15)

    def test_uplink_window_capacity(self):
        """测试UplinkWindow容量计算"""
        from core.network.uplink_scheduler import UplinkWindow

        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 10),  # 10分钟
            max_data_rate_mbps=10.0
        )

        # 10分钟 * 10 Mbps = 600秒 * 10Mbps = 6000 Mbits = 750 MB
        expected_capacity = 10.0 * 600 / 8  # Mbps * seconds / 8 = MB
        assert window.get_capacity_mb() == expected_capacity

    def test_uplink_window_overlaps(self):
        """测试UplinkWindow重叠检测"""
        from core.network.uplink_scheduler import UplinkWindow

        window1 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )

        # 重叠窗口
        window2 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 10),
            end_time=datetime(2024, 1, 1, 10, 20)
        )

        # 不重叠窗口
        window3 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 20),
            end_time=datetime(2024, 1, 1, 10, 30)
        )

        assert window1.overlaps(window2) is True
        assert window1.overlaps(window3) is False


class TestUplinkSchedulerInitialization:
    """测试UplinkScheduler初始化"""

    def test_scheduler_creation(self):
        """测试UplinkScheduler创建"""
        from core.network.uplink_scheduler import UplinkScheduler

        scheduler = UplinkScheduler()

        assert scheduler is not None
        assert hasattr(scheduler, 'ground_station_windows')
        assert hasattr(scheduler, 'preparation_lead_time')

    def test_scheduler_with_config(self):
        """测试带配置的UplinkScheduler创建"""
        from core.network.uplink_scheduler import UplinkScheduler

        config = {
            'preparation_lead_time_minutes': 30,
            'min_elevation_angle': 15.0,
            'max_range_km': 2000.0
        }

        scheduler = UplinkScheduler(config=config)

        assert scheduler.preparation_lead_time == timedelta(minutes=30)


class TestUplinkWindowManagement:
    """测试上行窗口管理"""

    @pytest.fixture
    def scheduler(self):
        """创建测试用的scheduler"""
        from core.network.uplink_scheduler import UplinkScheduler

        return UplinkScheduler()

    def test_add_uplink_window(self, scheduler):
        """测试添加上行窗口"""
        from core.network.uplink_scheduler import UplinkWindow

        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )

        scheduler.add_window(window)

        assert "gs_001" in scheduler.ground_station_windows
        assert len(scheduler.ground_station_windows["gs_001"]) == 1

    def test_get_windows_for_gs(self, scheduler):
        """测试获取地面站窗口"""
        from core.network.uplink_scheduler import UplinkWindow

        window1 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )
        window2 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 14, 0),
            end_time=datetime(2024, 1, 1, 14, 15)
        )

        scheduler.add_window(window1)
        scheduler.add_window(window2)

        windows = scheduler.get_windows_for_ground_station("gs_001")
        assert len(windows) == 2

    def test_get_windows_in_range(self, scheduler):
        """测试获取时间范围内的窗口"""
        from core.network.uplink_scheduler import UplinkWindow

        window1 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )
        window2 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 14, 0),
            end_time=datetime(2024, 1, 1, 14, 15)
        )

        scheduler.add_window(window1)
        scheduler.add_window(window2)

        windows = scheduler.get_windows_in_range(
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 12, 0)
        )

        assert len(windows) == 1
        assert windows[0].start_time == datetime(2024, 1, 1, 10, 0)


class TestUplinkFeasibility:
    """测试上行可行性检查"""

    @pytest.fixture
    def scheduler_with_windows(self):
        """创建带有窗口的scheduler"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler(
            config={'preparation_lead_time_minutes': 30}
        )

        # 添加窗口
        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15),
            max_data_rate_mbps=10.0
        )
        scheduler.add_window(window)

        return scheduler

    def test_feasible_uplink(self, scheduler_with_windows):
        """测试可行的上行"""
        scheduler = scheduler_with_windows

        # 在窗口前35分钟请求（留出30分钟准备时间）
        request_time = datetime(2024, 1, 1, 9, 25)
        command_size_mb = 1.0  # 1MB指令数据

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=command_size_mb,
            earliest_transmission_time=request_time
        )

        assert is_feasible is True
        assert window is not None
        assert reason == ""

    def test_infeasible_no_window(self, scheduler_with_windows):
        """测试无窗口时的不可行"""
        scheduler = scheduler_with_windows

        request_time = datetime(2024, 1, 1, 12, 0)  # 窗口之后

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=request_time
        )

        assert is_feasible is False
        assert window is None
        assert reason != ""

    def test_infeasible_insufficient_time(self, scheduler_with_windows):
        """测试准备时间不足"""
        scheduler = scheduler_with_windows

        # 在窗口前10分钟请求（需要30分钟准备时间）
        request_time = datetime(2024, 1, 1, 9, 50)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=request_time
        )

        assert is_feasible is False
        assert "preparation" in reason.lower() or "time" in reason.lower()

    def test_infeasible_data_rate(self, scheduler_with_windows):
        """测试数据速率不足"""
        scheduler = scheduler_with_windows

        request_time = datetime(2024, 1, 1, 9, 0)
        # 请求传输的数据量超过窗口容量
        large_data_mb = 10000.0

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=large_data_mb,
            earliest_transmission_time=request_time
        )

        assert is_feasible is False

    def test_feasible_with_multiple_windows(self):
        """测试多窗口选择"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler(
            config={'preparation_lead_time_minutes': 30}
        )

        # 添加多个窗口
        window1 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )
        window2 = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 12, 0),
            end_time=datetime(2024, 1, 1, 12, 15)
        )
        scheduler.add_window(window1)
        scheduler.add_window(window2)

        # 请求时间可以匹配第二个窗口
        request_time = datetime(2024, 1, 1, 11, 20)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=request_time
        )

        assert is_feasible is True
        assert window.start_time == datetime(2024, 1, 1, 12, 0)


class TestUplinkScheduling:
    """测试上行调度功能"""

    @pytest.fixture
    def scheduler(self):
        """创建测试用的scheduler"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler(
            config={'preparation_lead_time_minutes': 30}
        )

        # 添加多个地面站窗口
        for i, gs_id in enumerate(["gs_001", "gs_002"]):
            window = UplinkWindow(
                ground_station_id=gs_id,
                start_time=datetime(2024, 1, 1, 10 + i, 0),
                end_time=datetime(2024, 1, 1, 10 + i, 15),
                max_data_rate_mbps=10.0
            )
            scheduler.add_window(window)

        return scheduler

    def test_schedule_single_command(self, scheduler):
        """测试单指令调度"""
        command = {
            'id': 'cmd_001',
            'size_mb': 1.0,
            'target_satellite': 'sat_001',
            'earliest_transmission': datetime(2024, 1, 1, 9, 0)
        }

        scheduled = scheduler.schedule_uplink(
            ground_station_id="gs_001",
            command=command
        )

        assert scheduled is not None
        assert 'scheduled_time' in scheduled
        assert 'window' in scheduled

    def test_schedule_multiple_commands(self, scheduler):
        """测试多指令调度"""
        from core.network.uplink_scheduler import UplinkWindow

        # 添加更多窗口以支持多指令调度
        for i in range(2, 6):
            window = UplinkWindow(
                ground_station_id="gs_001",
                start_time=datetime(2024, 1, 1, 10 + i, 0),
                end_time=datetime(2024, 1, 1, 10 + i, 15),
                max_data_rate_mbps=10.0
            )
            scheduler.add_window(window)

        commands = [
            {
                'id': f'cmd_{i:03d}',
                'size_mb': 0.5,
                'target_satellite': 'sat_001',
                'earliest_transmission': datetime(2024, 1, 1, 9, 0)
            }
            for i in range(5)
        ]

        scheduled = scheduler.schedule_multiple_uplinks(
            ground_station_id="gs_001",
            commands=commands
        )

        assert len(scheduled) == 5
        # 检查没有重叠
        for i in range(len(scheduled) - 1):
            assert scheduled[i]['end_time'] <= scheduled[i + 1]['start_time']

    def test_optimal_gs_selection(self, scheduler):
        """测试最优地面站选择"""
        command = {
            'id': 'cmd_001',
            'size_mb': 1.0,
            'target_satellite': 'sat_001',
            'earliest_transmission': datetime(2024, 1, 1, 9, 30)
        }

        best_gs, window = scheduler.find_best_ground_station(
            command=command,
            candidate_gs=["gs_001", "gs_002"]
        )

        assert best_gs in ["gs_001", "gs_002"]
        assert window is not None


class TestUplinkConstraintHandling:
    """测试上行约束处理"""

    def test_elevation_constraint(self):
        """测试仰角约束"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler(
            config={'min_elevation_angle': 20.0}
        )

        # 低仰角窗口
        low_elev_window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15),
            elevation_angle=10.0  # 低于最小仰角
        )
        scheduler.add_window(low_elev_window)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=datetime(2024, 1, 1, 9, 0)
        )

        assert is_feasible is False
        assert "elevation" in reason.lower()

    def test_range_constraint(self):
        """测试距离约束"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler(
            config={'max_range_km': 2000.0}
        )

        # 远距离窗口
        far_window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15),
            range_km=3000.0  # 超过最大距离
        )
        scheduler.add_window(far_window)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=datetime(2024, 1, 1, 9, 0)
        )

        assert is_feasible is False
        assert "range" in reason.lower()


class TestUplinkEdgeCases:
    """测试上行调度边界情况"""

    def test_empty_window_list(self):
        """测试空窗口列表"""
        from core.network.uplink_scheduler import UplinkScheduler

        scheduler = UplinkScheduler()

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=datetime(2024, 1, 1, 9, 0)
        )

        assert is_feasible is False
        assert window is None

    def test_zero_size_command(self):
        """测试零大小指令"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler()
        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )
        scheduler.add_window(window)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=0.0,
            earliest_transmission_time=datetime(2024, 1, 1, 9, 0)
        )

        assert is_feasible is True

    def test_very_large_command(self):
        """测试超大指令"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler()
        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15),
            max_data_rate_mbps=10.0
        )
        scheduler.add_window(window)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1000000.0,  # 超大指令
            earliest_transmission_time=datetime(2024, 1, 1, 9, 0)
        )

        assert is_feasible is False

    def test_exact_preparation_time(self):
        """测试精确准备时间边界"""
        from core.network.uplink_scheduler import UplinkScheduler, UplinkWindow

        scheduler = UplinkScheduler(
            config={'preparation_lead_time_minutes': 30}
        )
        window = UplinkWindow(
            ground_station_id="gs_001",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 15)
        )
        scheduler.add_window(window)

        # 正好提前30分钟
        request_time = datetime(2024, 1, 1, 9, 30)

        is_feasible, window, reason = scheduler.check_uplink_feasibility(
            ground_station_id="gs_001",
            command_size_mb=1.0,
            earliest_transmission_time=request_time
        )

        assert is_feasible is True
