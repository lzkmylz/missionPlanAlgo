"""
中继数传任务数据模型单元测试
"""
import pytest
from datetime import datetime, timedelta

from scheduler.relay.downlink_task import RelayDownlinkTask


class TestRelayDownlinkTask:
    """测试中继数传任务数据模型"""

    def test_basic_creation(self):
        """测试基本创建"""
        now = datetime.now()
        task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=5.0
        )

        assert task.task_id == 'RL-001'
        assert task.satellite_id == 'SAT-01'
        assert task.relay_id == 'RELAY-01'
        assert task.data_size_gb == 5.0
        assert task.effective_data_rate == 450.0  # 默认值

    def test_invalid_data_size(self):
        """测试无效数据量"""
        with pytest.raises(ValueError):
            RelayDownlinkTask(
                task_id='RL-001',
                satellite_id='SAT-01',
                relay_id='RELAY-01',
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=10),
                data_size_gb=0.0  # 无效
            )

    def test_get_duration_seconds(self):
        """测试获取时长"""
        now = datetime.now()
        task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=5.0
        )

        assert task.get_duration_seconds() == 600.0  # 10分钟

    def test_get_transmission_time_seconds(self):
        """测试获取传输时间"""
        now = datetime.now()
        task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=5.0,
            acquisition_time_seconds=10.0
        )

        # 600秒 - 10秒建链 = 590秒
        assert task.get_transmission_time_seconds() == 590.0

    def test_get_actual_data_transferred(self):
        """测试计算实际传输数据量"""
        now = datetime.now()
        task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=100.0,  # 需要传输100GB
            effective_data_rate=450.0,  # 450 Mbps
            acquisition_time_seconds=10.0
        )

        # 传输时间 = 590秒
        # 可传输 = 450 * 590 / 8000 = 33.1875 GB
        actual = task.get_actual_data_transferred()
        expected = (450.0 * 590.0) / 8000.0
        assert abs(actual - expected) < 0.01

    def test_is_sufficient_for_data(self):
        """测试检查窗口是否足够"""
        now = datetime.now()
        # 窗口足够的情况
        task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=20),
            data_size_gb=10.0,
            effective_data_rate=450.0,
            acquisition_time_seconds=10.0
        )
        assert task.is_sufficient_for_data() is True

        # 窗口不足的情况
        task2 = RelayDownlinkTask(
            task_id='RL-002',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(seconds=100),
            data_size_gb=100.0,  # 100GB需要很长时间
            effective_data_rate=100.0,  # 低速率
            acquisition_time_seconds=10.0
        )
        assert task2.is_sufficient_for_data() is False

    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime.now()
        task = RelayDownlinkTask(
            task_id='RL-001',
            satellite_id='SAT-01',
            relay_id='RELAY-01',
            start_time=now,
            end_time=now + timedelta(minutes=10),
            data_size_gb=5.0,
            related_imaging_task_id='IMG-001'
        )

        d = task.to_dict()
        assert d['task_id'] == 'RL-001'
        assert d['satellite_id'] == 'SAT-01'
        assert d['relay_id'] == 'RELAY-01'
        assert d['data_size_gb'] == 5.0
        assert d['related_imaging_task_id'] == 'IMG-001'
