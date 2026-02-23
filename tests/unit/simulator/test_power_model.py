"""
电源模型单元测试

测试H5: 电源模型日食充电逻辑 (Chapter 12.3)
修复simulator/state_tracker.py中的PowerModel
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from simulator.state_tracker import PowerModel, SatelliteStateData, PowerModelConfig


class TestPowerModel:
    """测试电源模型"""

    @pytest.fixture
    def power_model(self):
        """创建电源模型实例"""
        config = PowerModelConfig(
            max_capacity_wh=1000.0,
            initial_charge_wh=800.0,
            nominal_generation_wh_per_sec=10.0,  # 10W = 10 Wh per second
            eclipse_generation_wh_per_sec=0.0    # 地影中不发电
        )
        return PowerModel(config=config)

    def test_initialization(self, power_model):
        """测试初始化"""
        assert power_model.max_capacity_wh == 1000.0
        assert power_model.current_charge_wh == 800.0
        assert power_model.config.nominal_generation_wh_per_sec == 10.0
        assert power_model.config.eclipse_generation_wh_per_sec == 0.0

    def test_get_soc(self, power_model):
        """测试获取电量状态"""
        soc = power_model.get_soc()
        assert soc == 0.8  # 800/1000

    def test_consume_power(self, power_model):
        """测试消耗电量"""
        initial = power_model.current_charge_wh
        power_model.consume_power(100.0, 1.0)  # 100W for 1 second
        assert power_model.current_charge_wh == initial - 100.0

    def test_consume_power_insufficient(self, power_model):
        """测试电量不足时消耗"""
        power_model.current_charge_wh = 50.0
        result = power_model.consume_power(100.0, 1.0)
        assert result is False
        assert power_model.current_charge_wh == 0.0  # 消耗到0

    def test_charge_normal(self, power_model):
        """测试正常充电（非地影）"""
        initial = power_model.current_charge_wh
        power_model.charge(10.0, in_eclipse=False)  # 充电10秒
        expected = initial + 10.0 * 10.0  # 10W * 10s
        assert power_model.current_charge_wh == expected

    def test_charge_in_eclipse(self, power_model):
        """测试地影中充电 - H5关键功能"""
        initial = power_model.current_charge_wh
        power_model.charge(10.0, in_eclipse=True)  # 地影中充电10秒
        # 地影中不发电，电量应保持不变
        assert power_model.current_charge_wh == initial

    def test_charge_with_custom_eclipse_rate(self):
        """测试自定义地影发电速率"""
        config = PowerModelConfig(
            max_capacity_wh=1000.0,
            initial_charge_wh=800.0,
            nominal_generation_wh_per_sec=10.0,
            eclipse_generation_wh_per_sec=2.0  # 地影中有少量发电（如核电池）
        )
        model = PowerModel(config=config)
        initial = model.current_charge_wh
        model.charge(10.0, in_eclipse=True)
        expected = initial + 10.0 * 2.0  # 2W * 10s
        assert model.current_charge_wh == expected

    def test_charge_capped_at_max(self, power_model):
        """测试充电不超过最大容量"""
        power_model.current_charge_wh = 990.0
        power_model.charge(10.0, in_eclipse=False)  # 尝试充电100Wh
        assert power_model.current_charge_wh == 1000.0  # 应该被限制在1000

    def test_simulate_activity(self, power_model):
        """测试模拟活动"""
        # 模拟成像活动：非地影，持续60秒
        result = power_model.simulate_activity(
            activity_type="imaging",
            duration_seconds=60.0,
            in_eclipse=False
        )
        # 检查返回结果中包含电量变化
        assert 'net_change_wh' in result
        assert 'final_charge_wh' in result

    def test_simulate_activity_in_eclipse(self, power_model):
        """测试地影中模拟活动 - H5关键功能"""
        initial = power_model.current_charge_wh
        # 模拟成像活动：地影中，功耗50W，持续60秒
        power_model.simulate_activity(
            activity_type="imaging",
            duration_seconds=60.0,
            in_eclipse=True
        )
        # 地影中只耗电不发电
        assert power_model.current_charge_wh < initial

    def test_get_power_status(self, power_model):
        """测试获取电源状态"""
        status = power_model.get_power_status()
        assert "current_charge_wh" in status
        assert "soc" in status
        assert "max_capacity_wh" in status
        assert status["soc"] == 0.8

    def test_can_support_activity(self, power_model):
        """测试是否支持活动"""
        # 充足电量 - 使用合理的功耗值
        # 10Wh/s * 60s = 600Wh 发电，活动消耗应该小于可用电量
        assert power_model.can_support_activity(1.0, 60.0, in_eclipse=False) is True

        # 设置低电量并测试地影中
        power_model.current_charge_wh = 50.0
        # 高功耗活动在地影中应该不被支持
        assert power_model.can_support_activity(10.0, 60.0, in_eclipse=True) is False


class TestSatelliteStateWithEclipse:
    """测试带日食状态的卫星状态"""

    def test_satellite_state_tracks_eclipse(self):
        """测试卫星状态跟踪日食状态"""
        state = SatelliteStateData(
            satellite_id="SAT-01",
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            power_wh=800.0,
            storage_gb=50.0,
            is_eclipse=False
        )

        assert state.is_eclipse is False

        # 更新为地影状态
        state.is_eclipse = True
        assert state.is_eclipse is True

    def test_power_model_integration_with_state(self):
        """测试电源模型与状态集成"""
        config = PowerModelConfig(
            max_capacity_wh=1000.0,
            initial_charge_wh=800.0
        )
        power_model = PowerModel(config=config)

        # 非地影状态
        power_model.charge(60.0, in_eclipse=False)
        charge_normal = power_model.current_charge_wh

        # 重置并测试地影状态
        power_model.current_charge_wh = 800.0
        power_model.charge(60.0, in_eclipse=True)
        charge_eclipse = power_model.current_charge_wh

        # 非地影应该充电更多
        assert charge_normal > charge_eclipse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
