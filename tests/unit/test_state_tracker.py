"""
连续状态演化跟踪器测试

TDD测试文件 - 第12章设计实现
"""

import pytest
from datetime import datetime, timedelta
import math

from core.models.satellite import Satellite, SatelliteType, Orbit, OrbitType
from core.models.target import Target, TargetType, GeoPoint
from simulator.state_tracker import (
    SatelliteStateTracker,
    SatelliteState,
    PowerModel,
    StorageIntegrator,
    ImagingState
)


class TestSatelliteState:
    """测试卫星状态枚举"""

    def test_state_values(self):
        """测试状态值定义"""
        from simulator.state_tracker import SatelliteState
        assert SatelliteState.IDLE.value == "IDLE"
        assert SatelliteState.IMAGING.value == "IMAGING"
        assert SatelliteState.SLEWING.value == "SLEWING"
        assert SatelliteState.DOWNLINKING.value == "DOWNLINKING"


class TestPowerModel:
    """测试电量模型"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.power_model = PowerModel(self.satellite)

    def test_initial_battery_level(self):
        """测试初始电量为满电"""
        level = self.power_model.get_battery_level()
        assert level == 1.0

    def test_imaging_power_consumption(self):
        """测试成像功耗计算"""
        # 成像5分钟
        power_consumed = self.power_model.calculate_imaging_power(
            duration_seconds=300,
            imaging_mode="push_broom"
        )
        # 光学成像功耗约为最大功率的60%
        expected = self.satellite.capabilities.power_capacity * 0.6 * (300/3600)
        assert abs(power_consumed - expected) < 0.1

    def test_downlink_power_consumption(self):
        """测试数传功耗计算"""
        power_consumed = self.power_model.calculate_downlink_power(
            duration_seconds=600,
            data_rate_mbps=300
        )
        # 数传功耗较高，约为最大功率的80%
        expected = self.satellite.capabilities.power_capacity * 0.8 * (600/3600)
        assert abs(power_consumed - expected) < 0.1

    def test_idle_power_consumption(self):
        """测试空闲功耗计算"""
        power_consumed = self.power_model.calculate_idle_power(
            duration_seconds=3600
        )
        # 空闲功耗较低，约为最大功率的10%
        expected = self.satellite.capabilities.power_capacity * 0.1 * (3600/3600)
        assert abs(power_consumed - expected) < 0.1

    def test_power_depletion(self):
        """测试电量耗尽后不再减少"""
        # 消耗大量电量
        self.power_model.consume_power(1000000)
        level = self.power_model.get_battery_level()
        assert level >= 0.0
        assert level <= 1.0

    def test_sar_imaging_higher_power(self):
        """测试SAR成像功耗高于光学"""
        sar_sat = Satellite(
            id="SAR-01",
            name="SAR卫星",
            sat_type=SatelliteType.SAR_1
        )
        sar_power_model = PowerModel(sar_sat)

        optical_power = self.power_model.calculate_imaging_power(300, "push_broom")
        sar_power = sar_power_model.calculate_imaging_power(300, "stripmap")

        # SAR功耗应该更高
        assert sar_power > optical_power


class TestStorageIntegrator:
    """测试存储积分器"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.storage = StorageIntegrator(self.satellite)

    def test_initial_storage_empty(self):
        """测试初始存储为空"""
        assert self.storage.get_used_storage() == 0.0

    def test_add_imaging_data(self):
        """测试添加成像数据"""
        # 模拟一次成像，数据量1GB
        self.storage.add_imaging_data(
            target_id="TARGET-01",
            data_size_gb=1.0,
            timestamp=datetime(2024, 1, 1, 12, 0)
        )
        assert self.storage.get_used_storage() == 1.0

    def test_storage_overflow_protection(self):
        """测试存储溢出保护"""
        capacity = self.satellite.capabilities.storage_capacity

        # 尝试添加超过容量的数据
        with pytest.raises(ValueError):
            self.storage.add_imaging_data(
                target_id="TARGET-01",
                data_size_gb=capacity + 1.0,
                timestamp=datetime(2024, 1, 1, 12, 0)
            )

    def test_downlink_removes_data(self):
        """测试数传后数据移除"""
        # 添加数据
        self.storage.add_imaging_data(
            target_id="TARGET-01",
            data_size_gb=1.0,
            timestamp=datetime(2024, 1, 1, 12, 0)
        )

        # 数传部分数据
        removed = self.storage.remove_downlinked_data(
            data_size_gb=0.5,
            timestamp=datetime(2024, 1, 1, 13, 0)
        )

        assert removed == 0.5
        assert self.storage.get_used_storage() == 0.5

    def test_get_storage_at_time(self):
        """测试获取指定时间点的存储状态"""
        t0 = datetime(2024, 1, 1, 10, 0)

        # t0时刻添加2GB
        self.storage.add_imaging_data("T1", 2.0, t0)

        # t1时刻添加1GB
        t1 = datetime(2024, 1, 1, 11, 0)
        self.storage.add_imaging_data("T2", 1.0, t1)

        # 查询t0.5时刻的存储状态
        t_mid = datetime(2024, 1, 1, 10, 30)
        storage_at_mid = self.storage.get_storage_at_time(t_mid)

        # 此时应该只有T1的数据
        assert storage_at_mid == 2.0


class TestSatelliteStateTracker:
    """测试卫星状态跟踪器"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(
                orbit_type=OrbitType.SSO,
                altitude=500000,
                inclination=97.4
            )
        )
        self.tracker = SatelliteStateTracker(self.satellite)

    def test_initial_state(self):
        """测试初始状态"""
        t0 = datetime(2024, 1, 1, 0, 0)
        state = self.tracker.get_state_at(t0)

        assert state.satellite_id == "SAT-01"
        assert state.state == SatelliteState.IDLE
        assert state.battery_soc == 1.0
        assert state.storage_used_gb == 0.0

    def test_state_after_imaging(self):
        """测试成像后的状态变化"""
        # 模拟成像任务
        imaging_start = datetime(2024, 1, 1, 10, 0)
        imaging_end = datetime(2024, 1, 1, 10, 5)

        self.tracker.record_imaging_task(
            target_id="TARGET-01",
            start_time=imaging_start,
            end_time=imaging_end,
            data_size_gb=1.0
        )

        # 检查成像期间的状态
        during_imaging = datetime(2024, 1, 1, 10, 2)
        state = self.tracker.get_state_at(during_imaging)
        assert state.state == SatelliteState.IMAGING
        assert state.current_task == "TARGET-01"

        # 检查成像后的状态
        after_imaging = datetime(2024, 1, 1, 10, 10)
        state = self.tracker.get_state_at(after_imaging)
        assert state.state == SatelliteState.IDLE
        assert state.storage_used_gb == 1.0
        assert state.battery_soc < 1.0  # 电量应该减少

    def test_state_after_downlink(self):
        """测试数传后的状态变化"""
        # 先添加一些数据
        self.tracker.record_imaging_task(
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 5),
            data_size_gb=2.0
        )

        # 记录数传
        self.tracker.record_downlink_task(
            ground_station_id="GS-BJ",
            start_time=datetime(2024, 1, 1, 12, 0),
            end_time=datetime(2024, 1, 1, 12, 10),
            data_size_gb=1.5
        )

        # 检查数传后的存储状态
        after_downlink = datetime(2024, 1, 1, 12, 15)
        state = self.tracker.get_state_at(after_downlink)
        assert state.storage_used_gb == 0.5  # 2.0 - 1.5 = 0.5

    def test_slewing_state(self):
        """测试姿态机动状态"""
        slew_start = datetime(2024, 1, 1, 10, 0)
        slew_end = datetime(2024, 1, 1, 10, 1)

        self.tracker.record_slewing(
            from_target="TARGET-01",
            to_target="TARGET-02",
            start_time=slew_start,
            end_time=slew_end,
            slew_angle=30.0
        )

        during_slew = datetime(2024, 1, 1, 10, 0, 30)
        state = self.tracker.get_state_at(during_slew)
        assert state.state == SatelliteState.SLEWING

    def test_battery_never_negative(self):
        """测试电量永不为负"""
        # 添加大量任务消耗电量
        for i in range(100):
            # 每24小时增加一天
            day = i // 24
            hour = (10 + i) % 24
            self.tracker.record_imaging_task(
                target_id=f"TARGET-{i:03d}",
                start_time=datetime(2024, 1, 1 + day, hour, 0),
                end_time=datetime(2024, 1, 1 + day, hour, 5),
                data_size_gb=0.1
            )

        # 检查所有时间点的电量
        for i in range(100):
            day = i // 24
            hour = (10 + i) % 24
            t = datetime(2024, 1, 1 + day, hour, 0)
            state = self.tracker.get_state_at(t)
            assert state.battery_soc >= 0.0
            assert state.battery_soc <= 1.0

    def test_storage_never_negative(self):
        """测试存储永不为负"""
        # 添加数据
        self.tracker.record_imaging_task(
            target_id="TARGET-01",
            start_time=datetime(2024, 1, 1, 10, 0),
            end_time=datetime(2024, 1, 1, 10, 5),
            data_size_gb=1.0
        )

        # 数传超过存储的数据量
        self.tracker.record_downlink_task(
            ground_station_id="GS-BJ",
            start_time=datetime(2024, 1, 1, 12, 0),
            end_time=datetime(2024, 1, 1, 12, 10),
            data_size_gb=5.0  # 超过1GB存储
        )

        after_downlink = datetime(2024, 1, 1, 12, 15)
        state = self.tracker.get_state_at(after_downlink)
        assert state.storage_used_gb >= 0.0

    def test_time_range_validation(self):
        """测试时间范围验证"""
        # 尝试查询超出规划周期的时间
        future_time = datetime(2025, 1, 1, 0, 0)

        # 应该返回最近的已知状态，而不是报错
        state = self.tracker.get_state_at(future_time)
        assert state is not None
