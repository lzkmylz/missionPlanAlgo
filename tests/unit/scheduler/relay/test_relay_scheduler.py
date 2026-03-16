"""
中继卫星调度器单元测试
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from core.network.relay_satellite import RelaySatellite, RelayNetwork
from scheduler.relay import RelayScheduler, RelayScheduleResult
from scheduler.relay.downlink_task import RelayDownlinkTask
from scheduler.base_scheduler import ScheduledTask


class TestRelayScheduler:
    """测试中继调度器"""

    @pytest.fixture
    def relay_network(self):
        """创建测试用的中继网络"""
        relays = [
            RelaySatellite(
                id='RELAY-01',
                name='Test-Relay-1',
                orbit_type='GEO',
                longitude=120.0,
                uplink_capacity=450.0,
                downlink_capacity=450.0
            ),
            RelaySatellite(
                id='RELAY-02',
                name='Test-Relay-2',
                orbit_type='GEO',
                longitude=-60.0,
                uplink_capacity=300.0,
                downlink_capacity=300.0
            )
        ]
        network = RelayNetwork(relays)

        # 添加可见窗口
        now = datetime.now()
        network.add_visibility_window('SAT-01', 'RELAY-01', now, now + timedelta(minutes=15))
        network.add_visibility_window('SAT-01', 'RELAY-02', now + timedelta(hours=1), now + timedelta(hours=1, minutes=15))

        return network

    @pytest.fixture
    def scheduler(self, relay_network):
        """创建测试用的调度器"""
        return RelayScheduler(
            relay_network=relay_network,
            default_data_rate_mbps=450.0,
            link_setup_time_seconds=10.0
        )

    def test_initialization(self, scheduler):
        """测试初始化"""
        assert scheduler.default_data_rate_mbps == 450.0
        assert scheduler.link_setup_time_seconds == 10.0
        assert scheduler._storage_states == {}

    def test_initialize_satellite_storage(self, scheduler):
        """测试初始化卫星固存"""
        scheduler.initialize_satellite_storage('SAT-01', 0.0, 128.0)

        storage = scheduler.get_satellite_storage('SAT-01')
        assert storage is not None
        assert storage.capacity_gb == 128.0
        assert storage.current_gb == 0.0

    def test_get_satellite_storage_nonexistent(self, scheduler):
        """测试获取不存在的卫星固存"""
        storage = scheduler.get_satellite_storage('NONEXISTENT')
        assert storage is None

    def test_calculate_data_size(self, scheduler):
        """测试计算数据量"""
        task = Mock(spec=ScheduledTask)
        task.storage_before = 10.0
        task.storage_after = 25.0

        data_size = scheduler._calculate_data_size(task)
        assert data_size == 15.0

    def test_calculate_downlink_duration(self, scheduler):
        """测试计算数传时长"""
        # 10 GB数据，450 Mbps，10秒建链
        # 传输时间 = 10 * 8000 / 450 = 177.78秒
        # 总时间 = 177.78 + 10 = 187.78秒
        duration = scheduler._calculate_downlink_duration(10.0, 450.0)
        expected = (10.0 * 8000 / 450.0) + 10.0
        assert abs(duration - expected) < 0.1

    def test_has_time_conflict_no_conflict(self, scheduler):
        """测试无冲突检查"""
        now = datetime.now()
        scheduler._downlink_allocations['SAT-01'] = [
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=10))
        ]

        # 检查不重叠的时间
        has_conflict = scheduler._has_time_conflict(
            'SAT-01',
            now + timedelta(hours=2),
            now + timedelta(hours=2, minutes=10)
        )
        assert has_conflict is False

    def test_has_time_conflict_with_conflict(self, scheduler):
        """测试有冲突检查"""
        now = datetime.now()
        scheduler._downlink_allocations['SAT-01'] = [
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=10))
        ]

        # 检查重叠的时间
        has_conflict = scheduler._has_time_conflict(
            'SAT-01',
            now + timedelta(hours=1, minutes=5),
            now + timedelta(hours=1, minutes=15)
        )
        assert has_conflict is True

    def test_calculate_downlink_timing_success(self, scheduler):
        """测试计算数传时间（成功）"""
        now = datetime.now()
        vis_start = now
        vis_end = now + timedelta(minutes=20)
        imaging_end = now
        downlink_duration = 600.0  # 10分钟

        result = scheduler._calculate_downlink_timing(vis_start, vis_end, imaging_end, downlink_duration)
        assert result is not None
        start, end = result
        assert (end - start).total_seconds() == 600.0

    def test_calculate_downlink_timing_fail(self, scheduler):
        """测试计算数传时间（失败）"""
        now = datetime.now()
        vis_start = now
        vis_end = now + timedelta(minutes=5)  # 窗口太短
        imaging_end = now
        downlink_duration = 600.0  # 需要10分钟

        result = scheduler._calculate_downlink_timing(vis_start, vis_end, imaging_end, downlink_duration)
        assert result is None

    def test_plan_downlink_for_task_none_task(self, scheduler):
        """测试为None任务计划中继数传"""
        result = scheduler.plan_downlink_for_task(None, [])
        assert result is None

    def test_plan_downlink_for_task_zero_data(self, scheduler):
        """测试为零数据任务计划中继数传"""
        task = Mock(spec=ScheduledTask)
        task.storage_before = 10.0
        task.storage_after = 10.0  # 无新增数据

        result = scheduler.plan_downlink_for_task(task, [])
        assert result is None

    def test_release_storage_after_downlink(self, scheduler):
        """测试数传后释放固存"""
        scheduler.initialize_satellite_storage('SAT-01', 50.0, 128.0)

        downlink_task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10),
            data_size_gb=20.0
        )

        scheduler._release_storage_after_downlink(downlink_task)

        storage = scheduler.get_satellite_storage('SAT-01')
        assert storage.current_gb == 30.0  # 50 - 20 = 30

    def test_update_tasks_with_downlink_info(self, scheduler):
        """测试更新任务的中继数传信息"""
        # 创建模拟任务
        task1 = Mock(spec=ScheduledTask)
        task1.task_id = 'IMG-001'

        task2 = Mock(spec=ScheduledTask)
        task2.task_id = 'IMG-002'

        # 创建中继数传任务
        now = datetime.now()
        downlink_task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=10.0,
            related_imaging_task_id='IMG-001'
        )

        updated = scheduler.update_tasks_with_downlink_info(
            [task1, task2],
            [downlink_task]
        )

        assert len(updated) == 2
        # 检查第一个任务被更新
        assert task1.downlink_start == now
        assert task1.downlink_end == now + timedelta(minutes=10)
        assert task1.data_transferred == 10.0
        assert task1.ground_station_id == 'RELAY:RELAY-01'

    def test_get_relay_utilization(self, scheduler):
        """测试获取中继利用率统计"""
        now = datetime.now()
        scheduler._downlink_allocations['SAT-01'] = [
            (now, now + timedelta(minutes=10)),
            (now + timedelta(hours=1), now + timedelta(hours=1, minutes=15))
        ]

        stats = scheduler.get_relay_utilization('RELAY-01')
        assert stats['relay_id'] == 'RELAY-01'
        assert stats['total_tasks'] == 2

    def test_schedule_downlinks_for_tasks(self, scheduler):
        """测试批量调度中继数传"""
        # 初始化
        scheduler.initialize_satellite_storage('SAT-01', 0.0, 128.0)

        now = datetime.now()
        windows = {
            'SAT-01': [
                (now + timedelta(hours=1), now + timedelta(hours=1, minutes=15))
            ]
        }

        # 创建模拟成像任务
        task = Mock(spec=ScheduledTask)
        task.task_id = 'IMG-001'
        task.satellite_id = 'SAT-01'
        task.imaging_end = now + timedelta(minutes=30)
        task.storage_before = 0.0
        task.storage_after = 10.0  # 产生10GB数据

        result = scheduler.schedule_downlinks_for_tasks([task], windows)

        assert isinstance(result, RelayScheduleResult)
        assert len(result.storage_states) == 1
        assert 'SAT-01' in result.storage_states
