"""
地面站调度器集成测试

验证 GroundStationScheduler 与现有系统的集成
"""

import pytest
from datetime import datetime, timedelta

from scheduler.base_scheduler import ScheduledTask, TaskFailureReason
from scheduler.ground_station_scheduler import (
    GroundStationScheduler,
    calculate_downlink_duration,
    StorageState,
)
from core.models.ground_station import GroundStation, Antenna
from core.resources.ground_station_pool import GroundStationPool


class TestGroundStationSchedulerIntegration:
    """地面站调度器集成测试"""

    def test_end_to_end_single_task(self):
        """测试单任务的端到端流程"""
        # 创建地面站
        gs = GroundStation(
            id="GS-BEIJING",
            name="Beijing Ground Station",
            longitude=116.4,
            latitude=39.9,
            antennas=[
                Antenna(id="ANT-01", name="Main Antenna", data_rate=300.0),
            ]
        )
        pool = GroundStationPool([gs])

        # 创建调度器
        scheduler = GroundStationScheduler(
            ground_station_pool=pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=100.0
        )

        # 创建成像任务
        base_time = datetime(2026, 2, 28, 12, 0, 0)
        imaging_tasks = [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="high_resolution",
                storage_before=20.0,
                storage_after=35.0,  # 生成15GB数据
            )
        ]

        # 地面站可见性窗口
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=30)),
            ]
        }

        # 执行数传调度
        result = scheduler.schedule_downlinks_for_tasks(
            imaging_tasks,
            visibility_windows
        )

        # 验证结果
        assert result is not None
        assert len(result.downlink_tasks) == 1
        assert len(result.updated_scheduled_tasks) == 1

        # 验证数传任务详情
        downlink = result.downlink_tasks[0]
        assert downlink.satellite_id == "SAT-01"
        assert downlink.ground_station_id == "GS-BEIJING"
        assert downlink.data_size_gb == 15.0

        # 验证更新的成像任务
        updated_task = result.updated_scheduled_tasks[0]
        assert updated_task.ground_station_id == "GS-BEIJING"
        assert updated_task.downlink_start is not None
        assert updated_task.downlink_end is not None
        assert updated_task.data_transferred == 15.0

        # 验证数传时间在成像之后
        assert updated_task.downlink_start >= updated_task.imaging_end

        # 验证固存状态
        assert "SAT-01" in result.storage_states
        storage = result.storage_states["SAT-01"]
        assert storage.current_gb <= storage.capacity_gb

    def test_end_to_end_multiple_tasks_same_satellite(self):
        """测试同一卫星多任务的端到端流程"""
        # 创建地面站
        gs = GroundStation(
            id="GS-SHANGHAI",
            name="Shanghai Ground Station",
            longitude=121.5,
            latitude=31.2,
            antennas=[
                Antenna(id="ANT-01", data_rate=300.0),
                Antenna(id="ANT-02", data_rate=300.0),
            ]
        )
        pool = GroundStationPool([gs])

        scheduler = GroundStationScheduler(
            ground_station_pool=pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=100.0,
            overflow_threshold=0.9
        )

        base_time = datetime(2026, 2, 28, 10, 0, 0)

        # 创建多个成像任务
        imaging_tasks = [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="standard",
                storage_before=10.0,
                storage_after=25.0,  # +15GB
            ),
            ScheduledTask(
                task_id="IMG-002",
                satellite_id="SAT-01",
                target_id="TARGET-B",
                imaging_start=base_time + timedelta(minutes=30),
                imaging_end=base_time + timedelta(minutes=35),
                imaging_mode="standard",
                storage_before=25.0,
                storage_after=45.0,  # +20GB
            ),
            ScheduledTask(
                task_id="IMG-003",
                satellite_id="SAT-01",
                target_id="TARGET-C",
                imaging_start=base_time + timedelta(minutes=60),
                imaging_end=base_time + timedelta(minutes=65),
                imaging_mode="standard",
                storage_before=45.0,
                storage_after=70.0,  # +25GB
            ),
        ]

        # 地面站可见性窗口
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=25)),
                (base_time + timedelta(minutes=40), base_time + timedelta(minutes=55)),
                (base_time + timedelta(minutes=70), base_time + timedelta(minutes=90)),
            ]
        }

        # 执行调度
        result = scheduler.schedule_downlinks_for_tasks(
            imaging_tasks,
            visibility_windows
        )

        # 验证结果
        assert result is not None
        assert len(result.downlink_tasks) == 3
        assert len(result.failed_tasks) == 0

        # 验证每个任务都有数传安排
        for task in result.updated_scheduled_tasks:
            assert task.ground_station_id is not None
            assert task.downlink_start is not None
            assert task.downlink_end is not None
            assert task.data_transferred > 0

    def test_end_to_end_storage_overflow_prevention(self):
        """测试固存溢出预防的端到端流程"""
        gs = GroundStation(
            id="GS-GUANGZHOU",
            name="Guangzhou Ground Station",
            longitude=113.3,
            latitude=23.1,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]
        )
        pool = GroundStationPool([gs])

        scheduler = GroundStationScheduler(
            ground_station_pool=pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=100.0,
            overflow_threshold=0.85
        )

        base_time = datetime(2026, 2, 28, 14, 0, 0)

        # 创建会导致固存溢出的任务
        imaging_tasks = [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="high_resolution",
                storage_before=80.0,  # 已经接近满
                storage_after=95.0,   # +15GB，超过85%阈值
            ),
        ]

        # 地面站可见性窗口
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=30)),
            ]
        }

        # 执行调度
        result = scheduler.schedule_downlinks_for_tasks(
            imaging_tasks,
            visibility_windows
        )

        # 验证数传被安排以防止溢出
        assert len(result.downlink_tasks) == 1

        # 验证固存最终状态在安全范围内
        storage = result.storage_states["SAT-01"]
        assert storage.current_gb <= 100.0 * 0.85  # 应该在阈值以下

    def test_end_to_end_ground_station_conflict_resolution(self):
        """测试地面站冲突解决的端到端流程"""
        # 创建单个地面站（只有一个天线）
        gs = GroundStation(
            id="GS-SINGLE",
            name="Single Antenna Station",
            longitude=100.0,
            latitude=30.0,
            antennas=[Antenna(id="ANT-01", data_rate=300.0)]  # 只有一个天线
        )
        pool = GroundStationPool([gs])

        scheduler = GroundStationScheduler(
            ground_station_pool=pool,
            data_rate_mbps=300.0,
            storage_capacity_gb=100.0
        )

        base_time = datetime(2026, 2, 28, 8, 0, 0)

        # 创建两个卫星的成像任务，需要同时数传
        imaging_tasks = [
            ScheduledTask(
                task_id="IMG-001",
                satellite_id="SAT-01",
                target_id="TARGET-A",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="standard",
                storage_before=10.0,
                storage_after=20.0,
            ),
            ScheduledTask(
                task_id="IMG-002",
                satellite_id="SAT-02",
                target_id="TARGET-B",
                imaging_start=base_time,
                imaging_end=base_time + timedelta(minutes=5),
                imaging_mode="standard",
                storage_before=15.0,
                storage_after=25.0,
            ),
        ]

        # 两个卫星同时可见（冲突场景）
        visibility_windows = {
            "SAT-01": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=25)),
            ],
            "SAT-02": [
                (base_time + timedelta(minutes=10), base_time + timedelta(minutes=25)),  # 冲突
            ],
        }

        # 执行调度
        result = scheduler.schedule_downlinks_for_tasks(
            imaging_tasks,
            visibility_windows
        )

        # 由于只有一个天线，应该只有一个任务被安排数传
        # 或者两个任务被安排在不同时间
        assert len(result.downlink_tasks) >= 1

        # 验证没有重叠的数传时间（如果都安排了）
        if len(result.downlink_tasks) == 2:
            dl1 = result.downlink_tasks[0]
            dl2 = result.downlink_tasks[1]
            # 检查时间是否重叠
            no_overlap = (
                dl1.end_time <= dl2.start_time or
                dl2.end_time <= dl1.start_time
            )
            assert no_overlap, "两个数传任务不应该有时间重叠"

    def test_scheduled_task_to_dict_with_downlink_info(self):
        """测试带数传信息的任务序列化"""
        base_time = datetime(2026, 2, 28, 16, 0, 0)

        task = ScheduledTask(
            task_id="IMG-001",
            satellite_id="SAT-01",
            target_id="TARGET-A",
            imaging_start=base_time,
            imaging_end=base_time + timedelta(minutes=5),
            imaging_mode="standard",
            ground_station_id="GS-TEST",
            downlink_start=base_time + timedelta(minutes=15),
            downlink_end=base_time + timedelta(minutes=25),
            data_transferred=12.5,
        )

        data = task.to_dict()

        # 验证所有字段都被正确序列化
        assert data['task_id'] == "IMG-001"
        assert data['satellite_id'] == "SAT-01"
        assert data['ground_station_id'] == "GS-TEST"
        assert data['downlink_start'] == (base_time + timedelta(minutes=15)).isoformat()
        assert data['downlink_end'] == (base_time + timedelta(minutes=25)).isoformat()
        assert data['data_transferred'] == 12.5

    def test_downlink_duration_calculation_accuracy(self):
        """测试数传时长计算精度"""
        # 测试不同数据量和速率组合
        test_cases = [
            (1.0, 100.0, 1.0 / (100.0 / 8 / 1024)),      # 1GB @ 100Mbps
            (10.0, 200.0, 10.0 / (200.0 / 8 / 1024)),    # 10GB @ 200Mbps
            (50.0, 500.0, 50.0 / (500.0 / 8 / 1024)),    # 50GB @ 500Mbps
        ]

        for data_gb, rate_mbps, expected in test_cases:
            duration = calculate_downlink_duration(data_gb, rate_mbps)
            assert abs(duration - expected) < 0.01, \
                f"Data: {data_gb}GB, Rate: {rate_mbps}Mbps, Expected: {expected}s, Got: {duration}s"

    def test_storage_state_transitions(self):
        """测试固存状态转换"""
        storage = StorageState(
            capacity_gb=100.0,
            current_gb=30.0,
            overflow_threshold=0.9
        )

        # 初始状态
        assert storage.current_gb == 30.0
        assert storage.get_available_space() == 70.0

        # 添加数据
        storage.add_data(20.0)
        assert storage.current_gb == 50.0

        # 检查溢出
        assert not storage.will_overflow(30.0)  # 50 + 30 = 80 <= 90
        assert storage.will_overflow(50.0)      # 50 + 50 = 100 > 90

        # 移除数据
        storage.remove_data(15.0)
        assert storage.current_gb == 35.0

        # 使用率
        assert storage.get_usage_ratio() == 0.35
