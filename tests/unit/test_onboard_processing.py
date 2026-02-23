"""
在轨处理管理器单元测试

TDD测试文件 - 第20章设计实现
遵循测试先行原则：
1. 先写测试（RED）
2. 实现代码（GREEN）
3. 重构优化（REFACTOR）
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from enum import Enum, auto

from core.processing.onboard_processing_manager import (
    AIAcceleratorType,
    AIAcceleratorSpec,
    ProcessingTaskType,
    ProcessingTaskSpec,
    ProcessingDecision,
    SatelliteResourceState,
    DecisionContext,
    OnboardProcessingManager,
)


class TestAIAcceleratorType:
    """测试AI加速器类型枚举"""

    def test_enum_values(self):
        """测试枚举值定义"""
        assert AIAcceleratorType.NVIDIA_JETSON_AGX.name == "NVIDIA_JETSON_AGX"
        assert AIAcceleratorType.NVIDIA_JETSON_ORIN.name == "NVIDIA_JETSON_ORIN"
        assert AIAcceleratorType.XILINX_VERSAL.name == "XILINX_VERSAL"
        assert AIAcceleratorType.CUSTOM_FPGA.name == "CUSTOM_FPGA"

    def test_enum_auto_values(self):
        """测试枚举自动赋值"""
        values = [e.value for e in AIAcceleratorType]
        # auto() 会分配从1开始的连续整数
        assert len(set(values)) == len(values)  # 所有值唯一


class TestProcessingTaskType:
    """测试处理任务类型枚举"""

    def test_enum_values(self):
        """测试枚举值定义"""
        assert ProcessingTaskType.VESSEL_DETECTION.name == "VESSEL_DETECTION"
        assert ProcessingTaskType.VEHICLE_DETECTION.name == "VEHICLE_DETECTION"
        assert ProcessingTaskType.CHANGE_DETECTION.name == "CHANGE_DETECTION"
        assert ProcessingTaskType.CLOUD_DETECTION.name == "CLOUD_DETECTION"
        assert ProcessingTaskType.IMAGE_CLASSIFICATION.name == "IMAGE_CLASSIFICATION"
        assert ProcessingTaskType.FEATURE_EXTRACTION.name == "FEATURE_EXTRACTION"


class TestAIAcceleratorSpec:
    """测试AI加速器硬件规格数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        spec = AIAcceleratorSpec(
            accelerator_type=AIAcceleratorType.NVIDIA_JETSON_AGX,
            compute_tops=32.0,
            power_consumption_w=30.0,
            power_idle_w=5.0,
            memory_gb=32.0,
            radiation_hardened=True,
            operational_temp_range=(-25.0, 85.0),
            tid_tolerance_krad=100.0,
            see_immune=True
        )
        assert spec.accelerator_type == AIAcceleratorType.NVIDIA_JETSON_AGX
        assert spec.compute_tops == 32.0
        assert spec.power_consumption_w == 30.0
        assert spec.power_idle_w == 5.0
        assert spec.memory_gb == 32.0
        assert spec.radiation_hardened is True
        assert spec.operational_temp_range == (-25.0, 85.0)
        assert spec.tid_tolerance_krad == 100.0
        assert spec.see_immune is True

    def test_optional_radiation_fields(self):
        """测试抗辐照字段可选"""
        spec = AIAcceleratorSpec(
            accelerator_type=AIAcceleratorType.CUSTOM_FPGA,
            compute_tops=10.0,
            power_consumption_w=15.0,
            power_idle_w=2.0,
            memory_gb=8.0,
            radiation_hardened=False,
            operational_temp_range=(0.0, 60.0),
            tid_tolerance_krad=0.0,
            see_immune=False
        )
        assert spec.radiation_hardened is False


class TestProcessingTaskSpec:
    """测试处理任务规格数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        spec = ProcessingTaskSpec(
            task_type=ProcessingTaskType.VESSEL_DETECTION,
            input_data_size_gb=2.0,
            output_data_size_kb=50.0,
            compute_requirement_tops=100.0,
            min_confidence=0.85
        )
        assert spec.task_type == ProcessingTaskType.VESSEL_DETECTION
        assert spec.input_data_size_gb == 2.0
        assert spec.output_data_size_kb == 50.0
        assert spec.compute_requirement_tops == 100.0
        assert spec.min_confidence == 0.85

    def test_compression_ratio(self):
        """测试压缩比计算"""
        spec = ProcessingTaskSpec(
            task_type=ProcessingTaskType.VESSEL_DETECTION,
            input_data_size_gb=1.0,  # 1 GB = 1,000,000 KB
            output_data_size_kb=100.0,
            compute_requirement_tops=100.0,
            min_confidence=0.85
        )
        # compression_ratio = (1.0 * 1e6) / 100.0 = 10000
        assert spec.compression_ratio == 10000.0

    def test_compression_ratio_edge_cases(self):
        """测试压缩比边界情况"""
        # 极小输出
        spec_small = ProcessingTaskSpec(
            task_type=ProcessingTaskType.FEATURE_EXTRACTION,
            input_data_size_gb=1.0,
            output_data_size_kb=1.0,  # 1 KB
            compute_requirement_tops=50.0,
            min_confidence=0.90
        )
        assert spec_small.compression_ratio == 1e6

    def test_processing_time_seconds(self):
        """测试处理时间计算"""
        task_spec = ProcessingTaskSpec(
            task_type=ProcessingTaskType.VESSEL_DETECTION,
            input_data_size_gb=2.0,
            output_data_size_kb=50.0,
            compute_requirement_tops=64.0,  # 64 TOPS-seconds
            min_confidence=0.85
        )
        accelerator = AIAcceleratorSpec(
            accelerator_type=AIAcceleratorType.NVIDIA_JETSON_AGX,
            compute_tops=32.0,  # 32 TOPS
            power_consumption_w=30.0,
            power_idle_w=5.0,
            memory_gb=32.0,
            radiation_hardened=True,
            operational_temp_range=(-25.0, 85.0),
            tid_tolerance_krad=100.0,
            see_immune=True
        )
        # processing_time = 64.0 / 32.0 = 2.0 seconds
        assert task_spec.processing_time_seconds(accelerator) == 2.0

    def test_processing_time_with_zero_compute_tops(self):
        """测试处理时间当算力为零时应处理异常"""
        task_spec = ProcessingTaskSpec(
            task_type=ProcessingTaskType.VESSEL_DETECTION,
            input_data_size_gb=2.0,
            output_data_size_kb=50.0,
            compute_requirement_tops=64.0,
            min_confidence=0.85
        )
        accelerator = AIAcceleratorSpec(
            accelerator_type=AIAcceleratorType.CUSTOM_FPGA,
            compute_tops=0.0,
            power_consumption_w=10.0,
            power_idle_w=1.0,
            memory_gb=4.0,
            radiation_hardened=False,
            operational_temp_range=(0.0, 50.0),
            tid_tolerance_krad=0.0,
            see_immune=False
        )
        # 应该返回无穷大或极大值
        result = task_spec.processing_time_seconds(accelerator)
        assert result == float('inf')


class TestProcessingDecision:
    """测试处理决策类型枚举"""

    def test_enum_values(self):
        """测试枚举值定义"""
        assert ProcessingDecision.PROCESS_ONBOARD.name == "PROCESS_ONBOARD"
        assert ProcessingDecision.DOWNLINK_RAW.name == "DOWNLINK_RAW"
        assert ProcessingDecision.HYBRID.name == "HYBRID"


class TestSatelliteResourceState:
    """测试卫星资源状态数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        assert state.battery_soc == 0.85
        assert state.storage_free_gb == 50.0
        assert state.thermal_headroom_c == 20.0
        assert state.ai_accelerator_idle is True
        assert state.upcoming_windows == []

    def test_with_upcoming_windows(self):
        """测试带可见窗口的状态"""
        windows = [
            {'priority': 5, 'time_to_window': timedelta(minutes=15)},
            {'priority': 9, 'time_to_window': timedelta(minutes=45)}
        ]
        state = SatelliteResourceState(
            battery_soc=0.75,
            storage_free_gb=20.0,
            thermal_headroom_c=15.0,
            ai_accelerator_idle=False,
            upcoming_windows=windows
        )
        assert len(state.upcoming_windows) == 2
        assert state.upcoming_windows[0]['priority'] == 5


class TestDecisionContext:
    """测试决策上下文数据类"""

    def test_basic_creation(self):
        """测试基本创建"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.5
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )
        assert context.imaging_task['task_id'] == 'IMG-001'
        assert context.satellite_state.battery_soc == 0.85
        assert context.mission_priority == 8
        assert context.latency_requirement == timedelta(minutes=30)
        assert context.accuracy_requirement == 0.90


class TestOnboardProcessingManager:
    """测试在轨处理管理器"""

    def setup_method(self):
        """每个测试方法前设置"""
        # 创建加速器规格
        self.accelerator_specs = {
            'SAT-01': AIAcceleratorSpec(
                accelerator_type=AIAcceleratorType.NVIDIA_JETSON_AGX,
                compute_tops=32.0,
                power_consumption_w=30.0,
                power_idle_w=5.0,
                memory_gb=32.0,
                radiation_hardened=True,
                operational_temp_range=(-25.0, 85.0),
                tid_tolerance_krad=100.0,
                see_immune=True
            ),
            'SAT-02': AIAcceleratorSpec(
                accelerator_type=AIAcceleratorType.NVIDIA_JETSON_ORIN,
                compute_tops=275.0,
                power_consumption_w=60.0,
                power_idle_w=10.0,
                memory_gb=64.0,
                radiation_hardened=True,
                operational_temp_range=(-25.0, 85.0),
                tid_tolerance_krad=150.0,
                see_immune=True
            )
        }

        # 创建处理任务规格
        self.processing_specs = {
            ProcessingTaskType.VESSEL_DETECTION: ProcessingTaskSpec(
                task_type=ProcessingTaskType.VESSEL_DETECTION,
                input_data_size_gb=2.0,
                output_data_size_kb=50.0,
                compute_requirement_tops=64.0,
                min_confidence=0.85
            ),
            ProcessingTaskType.VEHICLE_DETECTION: ProcessingTaskSpec(
                task_type=ProcessingTaskType.VEHICLE_DETECTION,
                input_data_size_gb=1.5,
                output_data_size_kb=30.0,
                compute_requirement_tops=48.0,
                min_confidence=0.80
            ),
            ProcessingTaskType.CLOUD_DETECTION: ProcessingTaskSpec(
                task_type=ProcessingTaskType.CLOUD_DETECTION,
                input_data_size_gb=2.0,
                output_data_size_kb=10.0,
                compute_requirement_tops=32.0,
                min_confidence=0.90
            )
        }

        self.manager = OnboardProcessingManager(
            accelerator_specs=self.accelerator_specs,
            processing_specs=self.processing_specs
        )

    def test_initialization(self):
        """测试初始化"""
        assert self.manager.accelerator_specs == self.accelerator_specs
        assert self.manager.processing_specs == self.processing_specs
        assert self.manager.decision_history == []
        assert self.manager.pareto_archive == {}

    def test_make_processing_decision_no_accelerator(self):
        """测试无AI加速器时的决策"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-03',  # 无加速器
            'data_size_gb': 2.0
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        decision, metadata = self.manager.make_processing_decision(context)

        assert decision == ProcessingDecision.DOWNLINK_RAW
        assert 'reason' in metadata
        assert 'No AI accelerator' in metadata['reason']

    def test_calculate_onboard_cost(self):
        """测试在轨处理代价计算"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.0,
            'task_type': 'vessel_detection'
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        accelerator = self.accelerator_specs['SAT-01']
        processing_spec = self.processing_specs[ProcessingTaskType.VESSEL_DETECTION]

        cost = self.manager._calculate_onboard_cost(context, accelerator, processing_spec)

        # 验证返回的代价结构
        assert 'energy_wh' in cost
        assert 'time_seconds' in cost
        assert 'storage_gb' in cost
        assert 'bandwidth_kb' in cost
        assert 'thermal_load_c' in cost
        assert 'confidence' in cost

        # 验证计算值
        # processing_time = 64.0 / 32.0 = 2.0 seconds
        assert cost['time_seconds'] == 2.0
        # energy_wh = (30.0 * 2.0) / 3600 = 0.0167 Wh
        expected_energy = (30.0 * 2.0) / 3600
        assert abs(cost['energy_wh'] - expected_energy) < 0.001
        assert cost['storage_gb'] == 2.0
        assert cost['bandwidth_kb'] == 50.0
        assert cost['confidence'] == 0.85

    def test_calculate_downlink_cost(self):
        """测试下传代价计算"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.0
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        cost = self.manager._calculate_downlink_cost(context)

        # 验证返回的代价结构
        assert 'energy_wh' in cost
        assert 'time_seconds' in cost
        assert 'storage_gb' in cost
        assert 'bandwidth_kb' in cost
        assert 'thermal_load_c' in cost
        assert 'confidence' in cost

        # 验证计算值
        # downlink_time = (2.0 * 8000) / 450 = 35.56 seconds
        expected_time = (2.0 * 8000) / 450
        assert abs(cost['time_seconds'] - expected_time) < 0.1
        assert cost['energy_wh'] == 50.0
        assert cost['storage_gb'] == 2.0
        assert cost['bandwidth_kb'] == 2.0 * 1e6
        assert cost['confidence'] == 1.0

    def test_pareto_analysis_onboard_dominates(self):
        """测试帕累托分析 - 在轨处理占优"""
        # 在轨处理代价极低（占优）
        onboard_cost = {
            'energy_wh': 1.0,
            'time_seconds': 10.0,
            'storage_gb': 2.0,
            'bandwidth_kb': 100.0,
            'thermal_load_c': 5.0
        }
        downlink_cost = {
            'energy_wh': 50.0,
            'time_seconds': 100.0,
            'storage_gb': 2.0,
            'bandwidth_kb': 2000000.0,
            'thermal_load_c': 10.0
        }

        result = self.manager._pareto_analysis(onboard_cost, downlink_cost)

        assert result['decision'] == ProcessingDecision.PROCESS_ONBOARD
        assert 'onboard_score' in result
        assert 'downlink_score' in result
        assert 'dominance_factor' in result
        assert result['dominance_factor'] > 0

    def test_pareto_analysis_downlink_dominates(self):
        """测试帕累托分析 - 下传占优"""
        # 下传代价在所有维度都更低（占优）
        onboard_cost = {
            'energy_wh': 100.0,
            'time_seconds': 200.0,
            'storage_gb': 2.0,
            'bandwidth_kb': 2000000.0,
            'thermal_load_c': 20.0
        }
        downlink_cost = {
            'energy_wh': 10.0,
            'time_seconds': 20.0,
            'storage_gb': 2.0,
            'bandwidth_kb': 2000000.0,
            'thermal_load_c': 5.0
        }

        result = self.manager._pareto_analysis(onboard_cost, downlink_cost)

        assert result['decision'] == ProcessingDecision.DOWNLINK_RAW
        assert result['dominance_factor'] > 0

    def test_pareto_analysis_trade_off(self):
        """测试帕累托分析 - 互不占优进入权衡"""
        # 互不占优的情况
        onboard_cost = {
            'energy_wh': 10.0,
            'time_seconds': 100.0,
            'storage_gb': 2.0,
            'bandwidth_kb': 100.0,
            'thermal_load_c': 10.0
        }
        downlink_cost = {
            'energy_wh': 50.0,
            'time_seconds': 50.0,
            'storage_gb': 2.0,
            'bandwidth_kb': 2000000.0,
            'thermal_load_c': 5.0
        }

        result = self.manager._pareto_analysis(onboard_cost, downlink_cost)

        # 应该返回权衡决策
        assert result['decision'] in [ProcessingDecision.PROCESS_ONBOARD, ProcessingDecision.DOWNLINK_RAW]
        assert result['dominance_factor'] == 0.0

    def test_trade_off_decision_high_compression(self):
        """测试权衡决策 - 高压缩比选择处理"""
        onboard_cost = {
            'energy_wh': 10.0,
            'bandwidth_kb': 100.0  # 极小输出
        }
        downlink_cost = {
            'energy_wh': 50.0,
            'bandwidth_kb': 2000000.0  # 大输出
        }

        decision = self.manager._trade_off_decision(onboard_cost, downlink_cost)

        # 压缩收益极高，应该选择处理
        assert decision == ProcessingDecision.PROCESS_ONBOARD

    def test_trade_off_decision_low_compression(self):
        """测试权衡决策 - 低压缩比选择下传"""
        onboard_cost = {
            'energy_wh': 100.0,  # 高能耗
            'bandwidth_kb': 1000000.0  # 输出仍然很大
        }
        downlink_cost = {
            'energy_wh': 50.0,
            'bandwidth_kb': 2000000.0
        }

        decision = self.manager._trade_off_decision(onboard_cost, downlink_cost)

        # 压缩收益不高，应该选择下传
        assert decision == ProcessingDecision.DOWNLINK_RAW

    def test_apply_state_constraints_low_battery(self):
        """测试状态约束 - 低电量强制下传"""
        pareto_analysis = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {'test': 'data'}
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.25,  # 低于30%
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)

        assert result['decision'] == ProcessingDecision.DOWNLINK_RAW
        assert 'override_reason' in result['metadata']
        assert 'Low battery' in result['metadata']['override_reason']
        assert result['metadata']['original_decision'] == ProcessingDecision.PROCESS_ONBOARD

    def test_apply_state_constraints_low_thermal(self):
        """测试状态约束 - 热余量不足避免处理"""
        pareto_analysis = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {'test': 'data'}
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=5.0,  # 低于10度
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)

        assert result['decision'] == ProcessingDecision.DOWNLINK_RAW
        assert 'override_reason' in result['metadata']
        assert 'thermal' in result['metadata']['override_reason'].lower()

    def test_apply_state_constraints_accelerator_busy_urgent(self):
        """测试状态约束 - AI加速器忙且有紧急任务"""
        pareto_analysis = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {'test': 'data'}
        }
        windows = [
            {'priority': 9, 'time_to_window': timedelta(minutes=15)}  # 紧急任务
        ]
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=False,  # 加速器忙
            upcoming_windows=windows
        )

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)

        assert result['decision'] == ProcessingDecision.DOWNLINK_RAW
        assert 'override_reason' in result['metadata']
        assert 'busy' in result['metadata']['override_reason'].lower()

    def test_apply_state_constraints_accelerator_busy_no_urgent(self):
        """测试状态约束 - AI加速器忙但无紧急任务"""
        pareto_analysis = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {'test': 'data'}
        }
        windows = [
            {'priority': 5, 'time_to_window': timedelta(minutes=60)}  # 非紧急
        ]
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=False,
            upcoming_windows=windows
        )

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)

        # 应该保持原决策
        assert result['decision'] == ProcessingDecision.PROCESS_ONBOARD

    def test_apply_state_constraints_low_storage(self):
        """测试状态约束 - 存储紧张时优先处理"""
        pareto_analysis = {
            'decision': ProcessingDecision.DOWNLINK_RAW,
            'metadata': {'test': 'data'}
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=5.0,  # 低于10GB
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)

        assert result['decision'] == ProcessingDecision.PROCESS_ONBOARD
        assert 'override_reason' in result['metadata']
        assert 'Storage' in result['metadata']['override_reason']

    def test_apply_state_constraints_no_override(self):
        """测试状态约束 - 无约束时保持原决策"""
        pareto_analysis = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {'test': 'data'}
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)

        assert result == pareto_analysis

    def test_make_processing_decision_full_flow(self):
        """测试完整决策流程"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.0,
            'task_type': 'vessel_detection'
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        decision, metadata = self.manager.make_processing_decision(context)

        # 验证决策有效
        assert decision in [ProcessingDecision.PROCESS_ONBOARD, ProcessingDecision.DOWNLINK_RAW, ProcessingDecision.HYBRID]
        assert isinstance(metadata, dict)

        # 验证决策历史已记录
        assert len(self.manager.decision_history) == 1
        assert self.manager.decision_history[0]['context'] == context

    def test_update_pareto_archive(self):
        """测试帕累托前沿存档更新"""
        decision_record = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'onboard_cost': {'energy_wh': 10.0},
            'downlink_cost': {'energy_wh': 50.0}
        }

        self.manager.update_pareto_archive('scenario-001', decision_record)

        assert 'scenario-001' in self.manager.pareto_archive
        assert len(self.manager.pareto_archive['scenario-001']) == 1
        assert self.manager.pareto_archive['scenario-001'][0]['decision'] == 'PROCESS_ONBOARD'

    def test_infer_task_type(self):
        """测试任务类型推断"""
        # 测试直接指定类型
        task_with_type = {'task_type': 'vessel_detection'}
        inferred = self.manager._infer_task_type(task_with_type)
        assert inferred == ProcessingTaskType.VESSEL_DETECTION

        # 测试从目标类型推断
        task_with_target = {'target_type': 'maritime'}
        inferred = self.manager._infer_task_type(task_with_target)
        assert inferred == ProcessingTaskType.VESSEL_DETECTION

        # 测试从成像模式推断
        task_with_mode = {'imaging_mode': 'vehicle'}
        inferred = self.manager._infer_task_type(task_with_mode)
        assert inferred == ProcessingTaskType.VEHICLE_DETECTION

    def test_infer_task_type_default(self):
        """测试任务类型推断默认值"""
        task_empty = {}
        inferred = self.manager._infer_task_type(task_empty)
        # 默认应该是特征提取
        assert inferred == ProcessingTaskType.FEATURE_EXTRACTION

    def test_estimate_thermal_load(self):
        """测试热负载估算"""
        accelerator = self.accelerator_specs['SAT-01']
        thermal_load = self.manager._estimate_thermal_load(accelerator, 100.0)

        # 热负载应该与功耗和时间成正比
        assert thermal_load > 0
        expected_load = (accelerator.power_consumption_w / 10.0) * (100.0 / 60.0)
        assert abs(thermal_load - expected_load) < 0.1

    def test_estimate_downlink_time(self):
        """测试下传时间估算"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.0
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        downlink_time = self.manager._estimate_downlink_time(context)

        # 2GB数据下传时间估算
        expected_time = (2.0 * 8000) / 450
        assert abs(downlink_time - expected_time) < 0.1

    def test_edge_case_zero_data_size(self):
        """测试边界条件 - 零数据大小"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 0.0
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        cost = self.manager._calculate_downlink_cost(context)
        assert cost['time_seconds'] == 0.0
        assert cost['bandwidth_kb'] == 0.0

    def test_edge_case_very_large_data(self):
        """测试边界条件 - 极大数据"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 1000.0  # 1TB
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        cost = self.manager._calculate_downlink_cost(context)
        # 应该能处理大数值而不溢出
        assert cost['time_seconds'] > 0
        assert cost['bandwidth_kb'] > 0

    def test_edge_case_negative_values(self):
        """测试边界条件 - 负值处理"""
        # 负电量应该被处理
        satellite_state = SatelliteResourceState(
            battery_soc=-0.1,  # 无效值
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        pareto_analysis = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {}
        }

        result = self.manager._apply_state_constraints(pareto_analysis, satellite_state)
        # 负电量应该触发低电量保护
        assert result['decision'] == ProcessingDecision.DOWNLINK_RAW

    def test_log_decision(self):
        """测试决策记录"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.0
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )
        adjusted_decision = {
            'decision': ProcessingDecision.PROCESS_ONBOARD,
            'metadata': {'test': 'metadata'}
        }
        onboard_cost = {'energy_wh': 10.0}
        downlink_cost = {'energy_wh': 50.0}

        self.manager._log_decision(context, adjusted_decision, onboard_cost, downlink_cost)

        assert len(self.manager.decision_history) == 1
        record = self.manager.decision_history[0]
        assert record['context'] == context
        assert record['decision'] == ProcessingDecision.PROCESS_ONBOARD
        assert record['metadata'] == {'test': 'metadata'}
        assert record['onboard_cost'] == onboard_cost
        assert record['downlink_cost'] == downlink_cost
        assert 'timestamp' in record

    def test_multiple_decisions_history(self):
        """测试多次决策历史记录"""
        for i in range(5):
            imaging_task = {
                'task_id': f'IMG-{i:03d}',
                'satellite_id': 'SAT-01',
                'data_size_gb': 2.0,
                'task_type': 'vessel_detection'  # 明确指定任务类型
            }
            satellite_state = SatelliteResourceState(
                battery_soc=0.85,
                storage_free_gb=50.0,
                thermal_headroom_c=20.0,
                ai_accelerator_idle=True,
                upcoming_windows=[]
            )
            context = DecisionContext(
                imaging_task=imaging_task,
                satellite_state=satellite_state,
                mission_priority=8,
                latency_requirement=timedelta(minutes=30),
                accuracy_requirement=0.90
            )
            self.manager.make_processing_decision(context)

        assert len(self.manager.decision_history) == 5

    def test_pareto_archive_multiple_scenarios(self):
        """测试多场景帕累托存档"""
        for i in range(3):
            scenario_id = f'scenario-{i:03d}'
            decision_record = {
                'decision': ProcessingDecision.PROCESS_ONBOARD,
                'onboard_cost': {'energy_wh': float(i)},
                'downlink_cost': {'energy_wh': float(i * 10)}
            }
            self.manager.update_pareto_archive(scenario_id, decision_record)

        assert len(self.manager.pareto_archive) == 3
        for i in range(3):
            assert f'scenario-{i:03d}' in self.manager.pareto_archive

    def test_make_processing_decision_no_processing_spec(self):
        """测试无处理规格时的决策"""
        imaging_task = {
            'task_id': 'IMG-001',
            'satellite_id': 'SAT-01',
            'data_size_gb': 2.0,
            'task_type': 'unknown_type'  # 未知类型
        }
        satellite_state = SatelliteResourceState(
            battery_soc=0.85,
            storage_free_gb=50.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )
        context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=8,
            latency_requirement=timedelta(minutes=30),
            accuracy_requirement=0.90
        )

        decision, metadata = self.manager.make_processing_decision(context)

        assert decision == ProcessingDecision.DOWNLINK_RAW
        assert 'reason' in metadata
        assert 'No processing spec' in metadata['reason']

    def test_infer_task_type_from_target_type_vehicle(self):
        """测试从目标类型推断车辆检测"""
        task = {'target_type': 'vehicle'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VEHICLE_DETECTION

        task = {'target_type': 'car'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VEHICLE_DETECTION

        task = {'target_type': 'traffic'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VEHICLE_DETECTION

    def test_infer_task_type_from_imaging_mode(self):
        """测试从成像模式推断任务类型"""
        task = {'imaging_mode': 'change'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.CHANGE_DETECTION

        task = {'imaging_mode': 'difference'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.CHANGE_DETECTION

        task = {'imaging_mode': 'cloud'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.CLOUD_DETECTION

        task = {'imaging_mode': 'weather'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.CLOUD_DETECTION

    def test_infer_task_type_invalid_task_type(self):
        """测试无效任务类型时回退到推断"""
        # 指定了无效类型，但有有效的target_type
        task = {'task_type': 'invalid_type', 'target_type': 'vessel'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VESSEL_DETECTION

    def test_infer_task_type_from_imaging_mode_vessel(self):
        """测试从成像模式推断舰船检测"""
        task = {'imaging_mode': 'vessel'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VESSEL_DETECTION

        task = {'imaging_mode': 'ship'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VESSEL_DETECTION

        task = {'imaging_mode': 'maritime'}
        inferred = self.manager._infer_task_type(task)
        assert inferred == ProcessingTaskType.VESSEL_DETECTION
