"""
热控模型集成测试

测试ThermalIntegrator与SatelliteStateTracker的集成
遵循TDD原则：先写测试，再实现代码
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock


class TestThermalIntegratorInitialization:
    """测试热控积分器初始化"""

    def test_thermal_integrator_creation(self):
        """测试ThermalIntegrator创建"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        assert integrator.params == params
        assert integrator.temperature == 273.15
        assert integrator.last_update_time is None

    def test_default_initial_temperature(self):
        """测试默认初始温度"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(ambient_temperature=280.0)
        integrator = ThermalIntegrator(params)

        assert integrator.temperature == 280.0


class TestThermalIntegratorOperations:
    """测试热控积分器操作"""

    def test_temperature_update(self):
        """测试温度更新"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        start_time = datetime(2024, 1, 1, 0, 0)
        end_time = start_time + timedelta(minutes=5)

        integrator.update(start_time, 'idle')
        temp = integrator.update(end_time, 'idle')

        assert isinstance(temp, float)
        assert temp > 0

    def test_temperature_prediction(self):
        """测试温度预测"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=273.15)

        predicted = integrator.predict_temperature(duration=300, activity='imaging_spotlight')

        assert isinstance(predicted, float)
        assert predicted > 0

    def test_temperature_validity_check(self):
        """测试温度有效性检查"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(max_operating_temp=333.15)
        integrator = ThermalIntegrator(params, initial_temp=300.0)

        is_valid, predicted = integrator.is_temperature_valid('imaging_spotlight', duration=600)

        assert isinstance(is_valid, bool)
        assert isinstance(predicted, float)

    def test_cooldown_time_calculation(self):
        """测试冷却时间计算"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(ambient_temperature=273.15)
        integrator = ThermalIntegrator(params, initial_temp=320.0)

        cooldown_time = integrator.get_cooldown_time(target_temp=290.0)

        assert isinstance(cooldown_time, float)
        assert cooldown_time >= 0

    def test_thermal_status(self):
        """测试热控状态获取"""
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        integrator = ThermalIntegrator(params, initial_temp=300.0)

        status = integrator.get_thermal_status()

        assert 'current_temperature_k' in status
        assert 'current_temperature_c' in status
        assert 'max_operating_temp_k' in status
        assert 'temperature_margin_k' in status
        assert 'is_safe' in status


class TestSatelliteStateTrackerThermalIntegration:
    """测试SatelliteStateTracker与热控模型集成"""

    @pytest.fixture
    def mock_satellite(self):
        """创建模拟卫星"""
        from core.models import Satellite, SatelliteType

        sat = Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.SAR_1
        )
        return sat

    def test_tracker_with_thermal_integrator(self, mock_satellite):
        """测试带热控积分器的跟踪器"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        thermal_integrator = ThermalIntegrator(params, initial_temp=273.15)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        assert tracker.thermal_integrator is not None
        assert tracker.thermal_integrator == thermal_integrator

    def test_tracker_without_thermal_integrator(self, mock_satellite):
        """测试不带热控积分器的跟踪器（向后兼容）"""
        from simulator.state_tracker import SatelliteStateTracker

        tracker = SatelliteStateTracker(satellite=mock_satellite)

        # 热控积分器应为可选
        assert hasattr(tracker, 'thermal_integrator')

    def test_thermal_state_in_snapshot(self, mock_satellite):
        """测试状态快照中包含热控信息"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        thermal_integrator = ThermalIntegrator(params, initial_temp=300.0)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        timestamp = datetime(2024, 1, 1, 12, 0)
        state_info = tracker.get_state_at(timestamp)

        assert hasattr(state_info, 'temperature')
        assert state_info.temperature is not None

    def test_imaging_task_updates_thermal(self, mock_satellite):
        """测试成像任务更新热控状态"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        thermal_integrator = ThermalIntegrator(params, initial_temp=273.15)
        initial_temp = thermal_integrator.temperature

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        start_time = datetime(2024, 1, 1, 12, 0)
        end_time = start_time + timedelta(minutes=5)

        tracker.record_imaging_task(
            target_id="target_001",
            start_time=start_time,
            end_time=end_time,
            data_size_gb=10.0
        )

        # 成像后温度应有所变化（通常升高）
        final_temp = tracker.thermal_integrator.temperature
        assert final_temp != initial_temp or len(tracker.thermal_integrator.temperature_history) > 0

    def test_thermal_constraint_check(self, mock_satellite):
        """测试热控约束检查"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(max_operating_temp=333.15)
        thermal_integrator = ThermalIntegrator(params, initial_temp=330.0)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 检查是否违反热控约束
        can_schedule, reason = tracker.check_thermal_constraint(
            activity='imaging_spotlight',
            duration=300
        )

        assert isinstance(can_schedule, bool)
        assert isinstance(reason, str)

    def test_thermal_violation_detection(self, mock_satellite):
        """测试热控违规检测"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        # 设置一个接近极限的温度
        params = ThermalParameters(
            max_operating_temp=333.15,
            emergency_shutdown_temp=343.15
        )
        thermal_integrator = ThermalIntegrator(params, initial_temp=340.0)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 验证调度方案时应检测到热控违规
        is_valid, violations = tracker.validate_schedule([])

        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)

    def test_temperature_history_tracking(self, mock_satellite):
        """测试温度历史记录"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        thermal_integrator = ThermalIntegrator(params, initial_temp=273.15)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 执行多个任务
        base_time = datetime(2024, 1, 1, 12, 0)
        for i in range(3):
            start_time = base_time + timedelta(minutes=i*10)
            end_time = start_time + timedelta(minutes=5)
            tracker.record_imaging_task(
                target_id=f"target_{i:03d}",
                start_time=start_time,
                end_time=end_time,
                data_size_gb=10.0
            )

        # 应有温度历史记录
        assert len(tracker.thermal_integrator.temperature_history) > 0


class TestThermalConstraintValidation:
    """测试热控约束验证"""

    @pytest.fixture
    def mock_satellite(self):
        """创建模拟卫星"""
        from core.models import Satellite, SatelliteType

        sat = Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.SAR_1
        )
        return sat

    def test_safe_temperature_range(self, mock_satellite):
        """测试安全温度范围"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(
            min_operating_temp=253.15,
            max_operating_temp=333.15
        )
        thermal_integrator = ThermalIntegrator(params, initial_temp=293.15)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 正常温度应通过验证
        is_valid, violations = tracker.validate_schedule([])
        assert is_valid is True

    def test_overheating_detection(self, mock_satellite):
        """测试过热检测"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(max_operating_temp=333.15)
        thermal_integrator = ThermalIntegrator(params, initial_temp=340.0)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 过热应被检测到
        is_valid, violations = tracker.validate_schedule([])
        assert is_valid is False
        assert any('thermal' in v.lower() or 'temperature' in v.lower() for v in violations)

    def test_freezing_detection(self, mock_satellite):
        """测试过冷检测"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(min_operating_temp=253.15)
        thermal_integrator = ThermalIntegrator(params, initial_temp=240.0)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 过冷应被检测到
        is_valid, violations = tracker.validate_schedule([])
        assert is_valid is False
        assert any('thermal' in v.lower() or 'temperature' in v.lower() for v in violations)


class TestThermalIntegrationEdgeCases:
    """测试热控集成边界情况"""

    @pytest.fixture
    def mock_satellite(self):
        """创建模拟卫星"""
        from core.models import Satellite, SatelliteType

        sat = Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.SAR_1
        )
        return sat

    def test_zero_duration_activity(self, mock_satellite):
        """测试零时长活动"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        thermal_integrator = ThermalIntegrator(params, initial_temp=273.15)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        start_time = datetime(2024, 1, 1, 12, 0)
        end_time = start_time  # 零时长

        # 不应抛出异常
        tracker.record_imaging_task(
            target_id="target_001",
            start_time=start_time,
            end_time=end_time,
            data_size_gb=10.0
        )

    def test_rapid_successive_tasks(self, mock_satellite):
        """测试快速连续任务"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters()
        thermal_integrator = ThermalIntegrator(params, initial_temp=273.15)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        base_time = datetime(2024, 1, 1, 12, 0)
        for i in range(10):
            start_time = base_time + timedelta(minutes=i)
            end_time = start_time + timedelta(seconds=30)
            tracker.record_imaging_task(
                target_id=f"target_{i:03d}",
                start_time=start_time,
                end_time=end_time,
                data_size_gb=5.0
            )

        # 应正确跟踪温度累积
        assert len(tracker.thermal_integrator.temperature_history) > 0

    def test_long_idle_period(self, mock_satellite):
        """测试长时间空闲"""
        from simulator.state_tracker import SatelliteStateTracker
        from simulator.thermal_model import ThermalIntegrator, ThermalParameters

        params = ThermalParameters(ambient_temperature=273.15)
        thermal_integrator = ThermalIntegrator(params, initial_temp=320.0)

        tracker = SatelliteStateTracker(
            satellite=mock_satellite,
            thermal_integrator=thermal_integrator
        )

        # 先执行一个任务
        start_time = datetime(2024, 1, 1, 12, 0)
        end_time = start_time + timedelta(minutes=5)
        tracker.record_imaging_task(
            target_id="target_001",
            start_time=start_time,
            end_time=end_time,
            data_size_gb=10.0
        )

        # 查询长时间后的状态（应冷却）
        future_time = end_time + timedelta(hours=2)
        state_info = tracker.get_state_at(future_time)

        assert state_info.temperature is not None
        # 长时间后应接近环境温度
        assert state_info.temperature <= 320.0
