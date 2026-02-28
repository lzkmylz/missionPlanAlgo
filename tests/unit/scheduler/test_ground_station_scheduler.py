"""
地面站调度器测试 - TDD方式

测试覆盖:
1. ScheduledTask 包含新的地面站字段
2. 数传时间计算
3. 地面站资源冲突检测
4. 数传后固存释放
5. 地面站调度集成
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from scheduler.base_scheduler import ScheduledTask, TaskFailureReason
from scheduler.ground_station_scheduler import (
    GroundStationScheduler,
    DownlinkTask,
    calculate_downlink_duration,
    StorageState,
)
from core.models.ground_station import GroundStation, Antenna
from core.resources.ground_station_pool import GroundStationPool


class TestScheduledTaskGroundStationFields:
    """测试 ScheduledTask 包含地面站相关字段"""

    def test_scheduled_task_has_ground_station_id(self):
        """测试 ScheduledTask 包含 ground_station_id 字段"""
        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(minutes=5),
            imaging_mode="standard",
            ground_station_id="GS-01"
        )
        assert task.ground_station_id == "GS-01"

    def test_scheduled_task_has_downlink_times(self):
        """测试 ScheduledTask 包含数传开始和结束时间"""
        now = datetime.now()
        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=now,
            imaging_end=now + timedelta(minutes=5),
            imaging_mode="standard",
            downlink_start=now + timedelta(minutes=10),
            downlink_end=now + timedelta(minutes=20),
        )
        assert task.downlink_start == now + timedelta(minutes=10)
        assert task.downlink_end == now + timedelta(minutes=20)

    def test_scheduled_task_has_data_transferred(self):
        """测试 ScheduledTask 包含实际传输数据量"""
        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(minutes=5),
            imaging_mode="standard",
            data_transferred=5.5
        )
        assert task.data_transferred == 5.5

    def test_scheduled_task_optional_fields_default_none(self):
        """测试可选字段默认值为 None"""
        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=datetime.now(),
            imaging_end=datetime.now() + timedelta(minutes=5),
            imaging_mode="standard",
        )
        assert task.ground_station_id is None
        assert task.downlink_start is None
        assert task.downlink_end is None
        assert task.data_transferred == 0.0

    def test_scheduled_task_to_dict_includes_ground_station_fields(self):
        """测试 to_dict 方法包含地面站字段"""
        now = datetime.now()
        task = ScheduledTask(
            task_id="TASK-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=now,
            imaging_end=now + timedelta(minutes=5),
            imaging_mode="standard",
            ground_station_id="GS-01",
            downlink_start=now + timedelta(minutes=10),
            downlink_end=now + timedelta(minutes=20),
            data_transferred=3.5,
        )
        data = task.to_dict()
        assert data['ground_station_id'] == "GS-01"
        assert data['downlink_start'] == (now + timedelta(minutes=10)).isoformat()
        assert data['downlink_end'] == (now + timedelta(minutes=20)).isoformat()
        assert data['data_transferred'] == 3.5


class TestDownlinkDurationCalculation:
    """测试数传时长计算"""

    def test_calculate_downlink_duration_basic(self):
        """测试基本数传时长计算"""
        # 数据量: 1 GB, 速率: 300 Mbps
        # duration = 1 / (300 / 8 / 1024) = 1 / 0.0366 = 27.3 seconds
        duration = calculate_downlink_duration(data_size_gb=1.0, data_rate_mbps=300.0)
        expected = 1.0 / (300.0 / 8 / 1024)
        assert abs(duration - expected) < 0.01

    def test_calculate_downlink_duration_zero_data(self):
        """测试零数据量情况"""
        duration = calculate_downlink_duration(data_size_gb=0.0, data_rate_mbps=300.0)
        assert duration == 0.0

    def test_calculate_downlink_duration_large_data(self):
        """测试大数据量情况"""
        # 100 GB 数据, 300 Mbps 速率
        duration = calculate_downlink_duration(data_size_gb=100.0, data_rate_mbps=300.0)
        expected = 100.0 / (300.0 / 8 / 1024)
        assert duration > 0
        assert duration == pytest.approx(expected, rel=0.001)

    def test_calculate_downlink_duration_different_rates(self):
        """测试不同速率下的计算"""
        # 相同数据量, 不同速率
        duration_100 = calculate_downlink_duration(data_size_gb=10.0, data_rate_mbps=100.0)
        duration_200 = calculate_downlink_duration(data_size_gb=10.0, data_rate_mbps=200.0)
        duration_400 = calculate_downlink_duration(data_size_gb=10.0, data_rate_mbps=400.0)

        # 速率越高, 时间越短
        assert duration_200 == pytest.approx(duration_100 / 2, rel=0.001)
        assert duration_400 == pytest.approx(duration_100 / 4, rel=0.001)

    def test_calculate_downlink_duration_zero_rate_raises_error(self):
        """测试零速率应抛出错误"""
        with pytest.raises(ValueError):
            calculate_downlink_duration(data_size_gb=1.0, data_rate_mbps=0.0)


class TestStorageState:
    """测试固存状态管理"""

    def test_storage_state_initialization(self):
        """测试固存状态初始化"""
        state = StorageState(capacity_gb=100.0, current_gb=0.0)
        assert state.capacity_gb == 100.0
        assert state.current_gb == 0.0
        assert state.get_available_space() == 100.0

    def test_storage_state_add_data(self):
        """测试添加数据到固存"""
        state = StorageState(capacity_gb=100.0, current_gb=0.0)
        state.add_data(30.0)
        assert state.current_gb == 30.0
        assert state.get_available_space() == 70.0

    def test_storage_state_remove_data(self):
        """测试从固存移除数据"""
        state = StorageState(capacity_gb=100.0, current_gb=50.0)
        state.remove_data(20.0)
        assert state.current_gb == 30.0
        assert state.get_available_space() == 70.0

    def test_storage_state_overflow_check(self):
        """测试固存溢出检查"""
        state = StorageState(capacity_gb=100.0, current_gb=80.0)
        assert not state.will_overflow(10.0)  # 80 + 10 = 90 < 100
        assert state.will_overflow(25.0)  # 80 + 25 = 105 > 100

    def test_storage_state_overflow_threshold(self):
        """测试固存溢出阈值检查"""
        state = StorageState(capacity_gb=100.0, current_gb=80.0, overflow_threshold=0.9)
        # 阈值 90%, 当前 80%, 阈值容量 = 90
        assert not state.will_overflow(5.0)   # 80 + 5 = 85 <= 90, 不会溢出
        assert state.will_overflow(15.0)  # 80 + 15 = 95 > 90, 会溢出

    def test_storage_state_remove_more_than_available(self):
        """测试移除超过可用数据量"""
        state = StorageState(capacity_gb=100.0, current_gb=30.0)
        state.remove_data(50.0)  # 尝试移除50,但只有30
        assert state.current_gb == 0.0  # 应该归零


class TestDownlinkTask:
    """测试数传任务数据类"""

    def test_downlink_task_creation(self):
        """测试数传任务创建"""
        now = datetime.now()
        task = DownlinkTask(
            task_id="DL-001",
            satellite_id="SAT-01",
            ground_station_id="GS-01",
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=5.0,
            antenna_id="ANT-01"
        )
        assert task.task_id == "DL-001"
        assert task.satellite_id == "SAT-01"
        assert task.ground_station_id == "GS-01"
        assert task.data_size_gb == 5.0

    def test_downlink_task_duration(self):
        """测试数传任务时长计算"""
        now = datetime.now()
        task = DownlinkTask(
            task_id="DL-001",
            satellite_id="SAT-01",
            ground_station_id="GS-01",
            start_time=now,
            end_time=now + timedelta(minutes=15),
            data_size_gb=5.0,
        )
        assert task.get_duration_seconds() == 900.0  # 15 minutes = 900 seconds


class TestGroundStationSchedulerInitialization:
    """测试地面站调度器初始化"""

    def test_scheduler_initialization(self):
        """测试调度器基本初始化"""
        scheduler = GroundStationScheduler(
            ground_station_pool=GroundStationPool([]),
            data_rate_mbps=300.0
        )
        assert scheduler.data_rate_mbps == 300.0
        assert scheduler.ground_station_pool is not None

    def test_scheduler_with_ground_stations(self):
        """测试带地面站的调度器初始化"""
        gs = GroundStation(
            id="GS-01",
            name="Test Station",
            longitude=116.4,
            latitude=39.9,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]
        )
        pool = GroundStationPool([gs])
        scheduler = GroundStationScheduler(
            ground_station_pool=pool,
            data_rate_mbps=300.0
        )
        assert scheduler.ground_station_pool.get_total_antenna_count() == 1


class TestGroundStationSchedulerConflictDetection:
    """测试地面站资源冲突检测"""

    @pytest.fixture
    def ground_station_pool(self):
        """创建测试用的地面站池"""
        gs = GroundStation(
            id="GS-01",
            name="Test Station",
            longitude=116.4,
            latitude=39.9,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]
        )
        return GroundStationPool([gs])

    @pytest.fixture
    def scheduler(self, ground_station_pool):
        """创建测试用的调度器"""
        return GroundStationScheduler(
            ground_station_pool=ground_station_pool,
            data_rate_mbps=300.0
        )

    def test_no_conflict_different_times(self, scheduler):
        """测试不同时间无冲突"""
        now = datetime.now()
        window1 = (now, now + timedelta(minutes=10))
        window2 = (now + timedelta(minutes=15), now + timedelta(minutes=25))

        # 需要指定天线ID
        assert not scheduler.has_antenna_conflict("GS-01", "ANT-01", window1)

        # 先分配第一个窗口
        scheduler.allocate_downlink_window("SAT-01", "GS-01", "ANT-01", window1)

        # 检查第二个窗口 - 应该无冲突
        assert not scheduler.has_antenna_conflict("GS-01", "ANT-01", window2)

    def test_conflict_overlapping_times(self, scheduler):
        """测试重叠时间有冲突"""
        now = datetime.now()
        window1 = (now, now + timedelta(minutes=10))
        window2 = (now + timedelta(minutes=5), now + timedelta(minutes=15))

        # 先分配第一个窗口
        scheduler.allocate_downlink_window("SAT-01", "GS-01", "ANT-01", window1)

        # 检查第二个窗口 - 应该有冲突
        assert scheduler.has_antenna_conflict("GS-01", "ANT-01", window2)

    def test_conflict_same_start_time(self, scheduler):
        """测试相同开始时间有冲突"""
        now = datetime.now()
        window1 = (now, now + timedelta(minutes=10))
        window2 = (now, now + timedelta(minutes=15))

        # 先分配第一个窗口
        scheduler.allocate_downlink_window("SAT-01", "GS-01", "ANT-01", window1)

        # 检查第二个窗口 - 应该有冲突
        assert scheduler.has_antenna_conflict("GS-01", "ANT-01", window2)

    def test_no_conflict_different_stations(self, scheduler, ground_station_pool):
        """测试不同地面站无冲突"""
        # 添加第二个地面站
        gs2 = GroundStation(
            id="GS-02",
            name="Test Station 2",
            longitude=121.5,
            latitude=31.2,
            antennas=[Antenna(id="ANT-02", data_rate=300.0)]
        )
        ground_station_pool.stations["GS-02"] = gs2

        # 重新初始化调度器的分配记录
        scheduler._downlink_allocations[("GS-02", "ANT-02")] = []

        now = datetime.now()
        window1 = (now, now + timedelta(minutes=10))

        # 在GS-01分配
        scheduler.allocate_downlink_window("SAT-01", "GS-01", "ANT-01", window1)

        # 在GS-02检查同一时间 - 应该无冲突
        assert not scheduler.has_antenna_conflict("GS-02", "ANT-02", window1)

    def test_no_conflict_different_antennas_same_station(self, scheduler):
        """测试同一地面站不同天线无冲突 - 新特性"""
        # 添加第二个天线到同一地面站
        gs = scheduler.ground_station_pool.get_station("GS-01")
        gs.add_antenna(Antenna(id="ANT-02", data_rate=300.0))

        # 重新初始化调度器的分配记录
        scheduler._downlink_allocations[("GS-01", "ANT-02")] = []

        now = datetime.now()
        window1 = (now, now + timedelta(minutes=10))

        # 在ANT-01分配
        scheduler.allocate_downlink_window("SAT-01", "GS-01", "ANT-01", window1)

        # 在同一地面站的ANT-02检查同一时间 - 应该无冲突
        assert not scheduler.has_antenna_conflict("GS-01", "ANT-02", window1)


class TestGroundStationSchedulerDownlinkPlanning:
    """测试数传计划生成"""

    @pytest.fixture
    def ground_station_pool(self):
        """创建测试用的地面站池"""
        gs = GroundStation(
            id="GS-01",
            name="Test Station",
            longitude=116.4,
            latitude=39.9,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]
        )
        return GroundStationPool([gs])

    @pytest.fixture
    def scheduler(self, ground_station_pool):
        """创建测试用的调度器"""
        return GroundStationScheduler(
            ground_station_pool=ground_station_pool,
            data_rate_mbps=300.0
        )

    @pytest.fixture
    def scheduled_imaging_task(self):
        """创建已调度的成像任务"""
        # 使用固定基准时间，避免时间漂移问题
        base_time = datetime(2026, 2, 28, 12, 0, 0)
        return ScheduledTask(
            task_id="IMG-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=base_time,
            imaging_end=base_time + timedelta(minutes=5),
            imaging_mode="standard",
            storage_before=10.0,
            storage_after=25.0,  # 生成了15GB数据
        )

    def test_plan_downlink_after_imaging(self, scheduler, scheduled_imaging_task):
        """测试在成像后安排数传"""
        # 使用固定基准时间
        base_time = datetime(2026, 2, 28, 12, 0, 0)
        # 地面站可见性窗口（成像后）
        visibility_window = (base_time + timedelta(minutes=10), base_time + timedelta(minutes=30))

        downlink_task = scheduler.plan_downlink_for_task(
            scheduled_imaging_task,
            visibility_window
        )

        assert downlink_task is not None
        assert downlink_task.satellite_id == "SAT-01"
        assert downlink_task.start_time >= scheduled_imaging_task.imaging_end
        assert downlink_task.data_size_gb == 15.0  # 25 - 10 = 15GB

    def test_plan_downlink_insufficient_visibility(self, scheduler, scheduled_imaging_task):
        """测试可见性窗口不足以完成数传"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)
        # 很短的可见性窗口
        visibility_window = (base_time + timedelta(minutes=10), base_time + timedelta(minutes=12))

        downlink_task = scheduler.plan_downlink_for_task(
            scheduled_imaging_task,
            visibility_window
        )

        # 应该返回None，因为窗口太短无法完成数传
        assert downlink_task is None

    def test_plan_downlink_before_imaging_fails(self, scheduler, scheduled_imaging_task):
        """测试在成像前安排数传应该失败"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)
        # 地面站可见性窗口在成像前
        visibility_window = (base_time - timedelta(minutes=30), base_time - timedelta(minutes=10))

        downlink_task = scheduler.plan_downlink_for_task(
            scheduled_imaging_task,
            visibility_window
        )

        # 应该返回None，因为数传不能在成像前
        assert downlink_task is None


class TestGroundStationSchedulerStorageManagement:
    """测试固存管理"""

    @pytest.fixture
    def ground_station_pool(self):
        """创建测试用的地面站池"""
        gs = GroundStation(
            id="GS-01",
            name="Test Station",
            longitude=116.4,
            latitude=39.9,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]
        )
        return GroundStationPool([gs])

    @pytest.fixture
    def scheduler(self, ground_station_pool):
        """创建测试用的调度器"""
        return GroundStationScheduler(
            ground_station_pool=ground_station_pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=100.0
        )

    def test_storage_release_after_downlink(self, scheduler):
        """测试数传后固存释放"""
        # 初始化卫星固存状态
        scheduler.initialize_satellite_storage("SAT-01", current_gb=80.0)

        # 模拟数传任务完成
        now = datetime.now()
        downlink_task = DownlinkTask(
            task_id="DL-001",
            satellite_id="SAT-01",
            ground_station_id="GS-01",
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=30.0,
        )

        # 释放固存
        scheduler.release_storage_after_downlink(downlink_task)

        # 检查固存状态
        storage = scheduler.get_satellite_storage("SAT-01")
        assert storage.current_gb == 50.0  # 80 - 30 = 50

    def test_storage_overflow_prevention(self, scheduler):
        """测试固存溢出预防"""
        # 初始化卫星固存状态（接近满）
        scheduler.initialize_satellite_storage("SAT-01", current_gb=90.0)

        # 检查是否会溢出
        assert scheduler.will_storage_overflow("SAT-01", 15.0)  # 90 + 15 > 100
        assert not scheduler.will_storage_overflow("SAT-01", 5.0)  # 90 + 5 <= 100

    def test_schedule_downlink_before_overflow(self, scheduler):
        """测试在固存溢出前安排数传"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)

        # 初始化固存状态（即将溢出）
        scheduler.initialize_satellite_storage("SAT-01", current_gb=85.0)

        # 创建一个会触发溢出的成像任务
        imaging_task = ScheduledTask(
            task_id="IMG-001",
            satellite_id="SAT-01",
            target_id="TARGET-01",
            imaging_start=base_time,
            imaging_end=base_time + timedelta(minutes=5),
            imaging_mode="standard",
            storage_before=85.0,
            storage_after=95.0,  # 增加10GB
        )

        # 地面站可见性窗口
        visibility_windows = [
            (base_time + timedelta(minutes=10), base_time + timedelta(minutes=30))
        ]

        # 应该安排数传以防止溢出
        downlink_task = scheduler.schedule_downlink_to_prevent_overflow(
            imaging_task,
            visibility_windows,
            overflow_threshold=0.9  # 90% 阈值
        )

        assert downlink_task is not None


class TestGroundStationSchedulerIntegration:
    """测试地面站调度器集成"""

    @pytest.fixture
    def ground_station_pool(self):
        """创建测试用的地面站池"""
        gs1 = GroundStation(
            id="GS-01",
            name="Beijing",
            longitude=116.4,
            latitude=39.9,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]
        )
        gs2 = GroundStation(
            id="GS-02",
            name="Shanghai",
            longitude=121.5,
            latitude=31.2,
            antennas=[Antenna(id="ANT-02", data_rate=300.0)]
        )
        return GroundStationPool([gs1, gs2])

    @pytest.fixture
    def scheduler(self, ground_station_pool):
        """创建测试用的调度器"""
        return GroundStationScheduler(
            ground_station_pool=ground_station_pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=100.0
        )

    @pytest.fixture
    def scheduled_imaging_tasks(self):
        """创建多个已调度的成像任务"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)
        return [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-01",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="standard",
                storage_before=10.0,
                storage_after=25.0,
            ),
            ScheduledTask(
                task_id="IMG-002",
                satellite_id="SAT-01",
                target_id="TARGET-02",
                imaging_start=base_time + timedelta(minutes=30),
                imaging_end=base_time + timedelta(minutes=35),
                imaging_mode="standard",
                storage_before=25.0,
                storage_after=45.0,
            ),
        ]

    def test_schedule_downlinks_for_tasks(self, scheduler, scheduled_imaging_tasks):
        """测试为多个任务安排数传"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)

        # 地面站可见性窗口
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=20)),
                (base_time + timedelta(minutes=40), base_time + timedelta(minutes=60)),
            ]
        }

        # 为数传任务安排数传
        result = scheduler.schedule_downlinks_for_tasks(
            scheduled_imaging_tasks,
            visibility_windows
        )

        assert result is not None
        assert len(result.downlink_tasks) > 0

        # 验证每个数传任务都在对应的成像之后
        for downlink in result.downlink_tasks:
            related_imaging = next(
                (t for t in scheduled_imaging_tasks if t.satellite_id == downlink.satellite_id),
                None
            )
            if related_imaging:
                assert downlink.start_time >= related_imaging.imaging_end

    def test_update_scheduled_tasks_with_downlink_info(self, scheduler, scheduled_imaging_tasks):
        """测试更新已调度任务为数传信息"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)

        # 创建数传任务
        downlink_tasks = [
            DownlinkTask(
                task_id="DL-001",
                satellite_id="SAT-01",
                ground_station_id="GS-01",
                start_time=base_time + timedelta(minutes=10),
                end_time=base_time + timedelta(minutes=20),
                data_size_gb=15.0,
                antenna_id="ANT-01",
                related_imaging_task_id="IMG-001"
            ),
        ]

        # 更新成像任务
        updated_tasks = scheduler.update_tasks_with_downlink_info(
            scheduled_imaging_tasks,
            downlink_tasks
        )

        # 验证更新
        task_001 = next(t for t in updated_tasks if t.task_id == "IMG-001")
        assert task_001.ground_station_id == "GS-01"
        assert task_001.downlink_start == base_time + timedelta(minutes=10)
        assert task_001.downlink_end == base_time + timedelta(minutes=20)
        assert task_001.data_transferred == 15.0

    def test_full_integration_flow(self, scheduler, scheduled_imaging_tasks):
        """测试完整集成流程"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)

        # 地面站可见性窗口
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=20)),
                (base_time + timedelta(minutes=40), base_time + timedelta(minutes=60)),
            ]
        }

        # 执行完整调度流程
        result = scheduler.schedule_downlinks_for_tasks(
            scheduled_imaging_tasks,
            visibility_windows
        )

        assert result is not None
        assert hasattr(result, 'downlink_tasks')
        assert hasattr(result, 'updated_scheduled_tasks')
        assert hasattr(result, 'storage_states')

        # 验证没有固存溢出
        for sat_id, storage in result.storage_states.items():
            assert storage.current_gb <= storage.capacity_gb

    def test_multiple_satellites_scheduling(self, scheduler):
        """测试多卫星调度"""
        base_time = datetime(2026, 2, 28, 12, 0, 0)

        # 多个卫星的成像任务
        tasks = [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-01",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="standard",
                storage_before=10.0,
                storage_after=25.0,
            ),
            ScheduledTask(
                task_id="IMG-002",
                satellite_id="SAT-02",
                target_id="TARGET-02",
                imaging_start=base_time + timedelta(minutes=5),
                imaging_end=base_time + timedelta(minutes=10),
                imaging_mode="standard",
                storage_before=5.0,
                storage_after=20.0,
            ),
        ]

        # 地面站可见性窗口
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=15), base_time + timedelta(minutes=25)),
            ],
            "SAT-02": [
                (base_time + timedelta(minutes=15), base_time + timedelta(minutes=25)),  # 同一时间，不同卫星
            ],
        }

        result = scheduler.schedule_downlinks_for_tasks(
            tasks,
            visibility_windows
        )

        assert result is not None
        assert len(result.downlink_tasks) == 2

        # 验证两个卫星都安排了数传
        sat_ids = {dl.satellite_id for dl in result.downlink_tasks}
        assert "SAT-01" in sat_ids
        assert "SAT-02" in sat_ids


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_task_list(self):
        """测试空任务列表"""
        scheduler = GroundStationScheduler(
            ground_station_pool=GroundStationPool([]),
            data_rate_mbps=300.0
        )

        result = scheduler.schedule_downlinks_for_tasks([], {})
        assert result is not None
        assert len(result.downlink_tasks) == 0

    def test_no_visibility_windows(self):
        """测试无可见性窗口"""
        scheduler = GroundStationScheduler(
            ground_station_pool=GroundStationPool([]),
            data_rate_mbps=300.0
        )

        now = datetime.now()
        tasks = [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-01",
                imaging_start=now,
                imaging_end=now + timedelta(minutes=5),
                imaging_mode="standard",
            ),
        ]

        result = scheduler.schedule_downlinks_for_tasks(tasks, {})
        assert result is not None
        assert len(result.downlink_tasks) == 0

    def test_very_large_data_size(self):
        """测试非常大的数据量"""
        # 1000 GB 数据
        duration = calculate_downlink_duration(data_size_gb=1000.0, data_rate_mbps=300.0)
        assert duration > 0
        # 1000GB / (300Mbps / 8 / 1024) = 1000 / 0.0366 = ~27306秒 = ~7.6小时
        assert duration > 7 * 3600  # 超过7小时

    def test_null_input_handling(self):
        """测试空输入处理"""
        scheduler = GroundStationScheduler(
            ground_station_pool=GroundStationPool([]),
            data_rate_mbps=300.0
        )

        # 应该优雅处理 None 输入
        result = scheduler.plan_downlink_for_task(None, (datetime.now(), datetime.now()))
        assert result is None
