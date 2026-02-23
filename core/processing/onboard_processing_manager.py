"""
在轨处理管理器

第20章：星载边缘计算
实现AI处理决策、帕累托优化、资源约束管理
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np


class AIAcceleratorType(Enum):
    """AI加速器类型"""
    NVIDIA_JETSON_AGX = auto()      # 抗辐照版本，32 TOPS
    NVIDIA_JETSON_ORIN = auto()     # 抗辐照版本，275 TOPS
    XILINX_VERSAL = auto()          # 自适应计算平台
    CUSTOM_FPGA = auto()            # 定制化FPGA方案


class ProcessingTaskType(Enum):
    """处理任务类型"""
    VESSEL_DETECTION = auto()       # 舰船检测
    VEHICLE_DETECTION = auto()      # 车辆检测
    CHANGE_DETECTION = auto()       # 变化检测
    CLOUD_DETECTION = auto()        # 云检测（光学）
    IMAGE_CLASSIFICATION = auto()   # 图像分类
    FEATURE_EXTRACTION = auto()     # 特征提取


@dataclass
class AIAcceleratorSpec:
    """AI加速器硬件规格"""
    accelerator_type: AIAcceleratorType
    compute_tops: float             # 算力（Tera Operations Per Second）
    power_consumption_w: float      # 功耗（W）
    power_idle_w: float             # 空闲功耗（W）
    memory_gb: float                # onboard memory
    radiation_hardened: bool        # 是否抗辐照
    operational_temp_range: tuple   # 工作温度范围（℃）

    # 抗辐照特性
    tid_tolerance_krad: float       # 总电离剂量耐受（krad）
    see_immune: bool                # 单粒子效应免疫


@dataclass
class ProcessingTaskSpec:
    """处理任务规格"""
    task_type: ProcessingTaskType
    input_data_size_gb: float       # 输入数据大小（GB）
    output_data_size_kb: float      # 输出数据大小（KB）
    compute_requirement_tops: float # 计算需求（TOPS-seconds）
    min_confidence: float           # 最低置信度要求

    @property
    def compression_ratio(self) -> float:
        """数据压缩比"""
        return (self.input_data_size_gb * 1e6) / self.output_data_size_kb

    def processing_time_seconds(self, accelerator: AIAcceleratorSpec) -> float:
        """估算处理时间"""
        if accelerator.compute_tops <= 0:
            return float('inf')
        return self.compute_requirement_tops / accelerator.compute_tops


class ProcessingDecision(Enum):
    """处理决策类型"""
    PROCESS_ONBOARD = auto()        # 在轨处理
    DOWNLINK_RAW = auto()           # 原始数据下传
    HYBRID = auto()                 # 混合策略（先存储，后决策）


@dataclass
class SatelliteResourceState:
    """
    卫星资源状态（用于决策）

    注意：与第12章的 SatelliteState 枚举区分
    - SatelliteState: 卫星运行状态（IDLE/IMAGING/SLEWING等）
    - SatelliteResourceState: 卫星资源状态（电量/存储/热控等）
    """
    battery_soc: float              # 电池电量（0-1）
    storage_free_gb: float          # 剩余存储空间
    thermal_headroom_c: float       # 热余量（℃）
    ai_accelerator_idle: bool       # AI加速器是否空闲
    upcoming_windows: List[Dict]    # 即将到来的可见窗口


@dataclass
class DecisionContext:
    """决策上下文"""
    imaging_task: Dict              # 成像任务信息
    satellite_state: SatelliteResourceState  # 卫星资源状态
    mission_priority: int           # 任务优先级
    latency_requirement: timedelta  # 延迟要求
    accuracy_requirement: float     # 精度要求


class OnboardProcessingManager:
    """
    在轨处理管理器

    核心职责：
    1. 对每个成像任务决策：在轨处理 vs 原始数据下传
    2. 管理帕累托前沿存档
    3. 动态适应卫星状态变化
    4. 处理失败回退机制
    """

    def __init__(self,
                 accelerator_specs: Dict[str, AIAcceleratorSpec],
                 processing_specs: Dict[ProcessingTaskType, ProcessingTaskSpec]):
        self.accelerator_specs = accelerator_specs
        self.processing_specs = processing_specs
        self.decision_history: List[Dict] = []

        # 帕累托前沿存档
        self.pareto_archive: Dict[str, List[Dict]] = {}

    def make_processing_decision(
        self,
        context: DecisionContext
    ) -> Tuple[ProcessingDecision, Dict]:
        """
        做出处理决策

        Returns:
            (决策类型, 决策元数据)
        """
        sat_id = context.imaging_task['satellite_id']
        accelerator = self.accelerator_specs.get(sat_id)

        if not accelerator:
            # 卫星无AI芯片，只能下传原始数据
            return ProcessingDecision.DOWNLINK_RAW, {
                'reason': 'No AI accelerator onboard',
                'estimated_downlink_time': self._estimate_downlink_time(context)
            }

        task_type = self._infer_task_type(context.imaging_task)
        processing_spec = self.processing_specs.get(task_type)

        if not processing_spec:
            # 无处理规格，只能下传原始数据
            return ProcessingDecision.DOWNLINK_RAW, {
                'reason': f'No processing spec for task type: {task_type}',
                'estimated_downlink_time': self._estimate_downlink_time(context)
            }

        # 计算两种策略的代价
        onboard_cost = self._calculate_onboard_cost(context, accelerator, processing_spec)
        downlink_cost = self._calculate_downlink_cost(context)

        # 帕累托决策分析
        pareto_analysis = self._pareto_analysis(onboard_cost, downlink_cost)

        # 基于当前卫星状态动态调整
        adjusted_decision = self._apply_state_constraints(
            pareto_analysis, context.satellite_state
        )

        # 记录决策
        self._log_decision(context, adjusted_decision, onboard_cost, downlink_cost)

        return adjusted_decision['decision'], adjusted_decision['metadata']

    def _calculate_onboard_cost(
        self,
        context: DecisionContext,
        accelerator: AIAcceleratorSpec,
        processing_spec: ProcessingTaskSpec
    ) -> Dict:
        """计算在轨处理代价"""
        processing_time = processing_spec.processing_time_seconds(accelerator)
        energy_wh = (accelerator.power_consumption_w * processing_time) / 3600

        return {
            'energy_wh': energy_wh,
            'time_seconds': processing_time,
            'storage_gb': processing_spec.input_data_size_gb,  # 临时存储
            'bandwidth_kb': processing_spec.output_data_size_kb,
            'thermal_load_c': self._estimate_thermal_load(accelerator, processing_time),
            'confidence': processing_spec.min_confidence
        }

    def _calculate_downlink_cost(self, context: DecisionContext) -> Dict:
        """计算原始数据下传代价"""
        data_size_gb = context.imaging_task['data_size_gb']

        # 估算下传时间（基于平均带宽）
        avg_bandwidth_mbps = 450  # X波段典型值
        downlink_time_seconds = (data_size_gb * 8000) / avg_bandwidth_mbps

        return {
            'energy_wh': 50,  # 数传设备功耗估算
            'time_seconds': downlink_time_seconds,
            'storage_gb': data_size_gb,  # 持续占用直到下传完成
            'bandwidth_kb': data_size_gb * 1e6,
            'thermal_load_c': 5.0,
            'confidence': 1.0  # 原始数据无精度损失
        }

    def _pareto_analysis(
        self,
        onboard_cost: Dict,
        downlink_cost: Dict
    ) -> Dict:
        """
        帕累托分析：判断哪个方案在帕累托前沿上占优

        目标（最小化）：能耗、时间、存储占用、带宽占用、热负载
        """
        objectives = ['energy_wh', 'time_seconds', 'storage_gb', 'bandwidth_kb', 'thermal_load_c']

        onboard_vector = np.array([onboard_cost[o] for o in objectives])
        downlink_vector = np.array([downlink_cost[o] for o in objectives])

        # 归一化（使用对数压缩大范围差异）
        onboard_norm = np.log1p(onboard_vector)
        downlink_norm = np.log1p(downlink_vector)

        # 计算支配关系
        onboard_dominates = np.all(onboard_norm <= downlink_norm) and np.any(onboard_norm < downlink_norm)
        downlink_dominates = np.all(downlink_norm <= onboard_norm) and np.any(downlink_norm < onboard_norm)

        if onboard_dominates:
            decision = ProcessingDecision.PROCESS_ONBOARD
            dominance_factor = np.sum(downlink_norm - onboard_norm)
        elif downlink_dominates:
            decision = ProcessingDecision.DOWNLINK_RAW
            dominance_factor = np.sum(onboard_norm - downlink_norm)
        else:
            # 互不占优，进入权衡决策
            decision = self._trade_off_decision(onboard_cost, downlink_cost)
            dominance_factor = 0.0

        return {
            'decision': decision,
            'onboard_score': float(np.sum(onboard_norm)),
            'downlink_score': float(np.sum(downlink_norm)),
            'dominance_factor': float(dominance_factor),
            'metadata': {
                'onboard_cost': onboard_cost,
                'downlink_cost': downlink_cost
            }
        }

    def _trade_off_decision(
        self,
        onboard_cost: Dict,
        downlink_cost: Dict
    ) -> ProcessingDecision:
        """
        权衡决策：当两种方案互不占优时使用

        策略：基于数据压缩收益与能耗代价的比值
        """
        # 数据压缩收益
        compression_gain = downlink_cost['bandwidth_kb'] / onboard_cost['bandwidth_kb']

        # 能耗代价比
        energy_ratio = onboard_cost['energy_wh'] / downlink_cost['energy_wh']

        # 综合收益比
        efficiency_ratio = compression_gain / energy_ratio

        if efficiency_ratio > 100:  # 压缩收益显著大于能耗代价
            return ProcessingDecision.PROCESS_ONBOARD
        else:
            return ProcessingDecision.DOWNLINK_RAW

    def _apply_state_constraints(
        self,
        pareto_analysis: Dict,
        satellite_state: SatelliteResourceState
    ) -> Dict:
        """
        应用卫星状态约束

        根据实时状态动态调整决策
        """
        decision = pareto_analysis['decision']
        metadata = pareto_analysis['metadata']

        # 约束1：低电量强制下传原始数据（避免AI处理耗电）
        if satellite_state.battery_soc < 0.3:
            return {
                'decision': ProcessingDecision.DOWNLINK_RAW,
                'metadata': {
                    **metadata,
                    'override_reason': 'Low battery (SOC < 30%)',
                    'original_decision': decision
                }
            }

        # 约束2：热余量不足时避免AI处理
        if decision == ProcessingDecision.PROCESS_ONBOARD and satellite_state.thermal_headroom_c < 10:
            return {
                'decision': ProcessingDecision.DOWNLINK_RAW,
                'metadata': {
                    **metadata,
                    'override_reason': 'Insufficient thermal headroom',
                    'original_decision': decision
                }
            }

        # 约束3：AI加速器忙时排队或下传
        if decision == ProcessingDecision.PROCESS_ONBOARD and not satellite_state.ai_accelerator_idle:
            # 检查是否有即将到来的紧急任务
            urgent_coming = any(
                w['priority'] > 8 and w['time_to_window'] < timedelta(minutes=30)
                for w in satellite_state.upcoming_windows
            )

            if urgent_coming:
                return {
                    'decision': ProcessingDecision.DOWNLINK_RAW,
                    'metadata': {
                        **metadata,
                        'override_reason': 'AI accelerator busy + urgent task upcoming',
                        'original_decision': decision
                    }
                }

        # 约束4：存储紧张时优先处理（释放空间）
        if satellite_state.storage_free_gb < 10 and decision == ProcessingDecision.DOWNLINK_RAW:
            return {
                'decision': ProcessingDecision.PROCESS_ONBOARD,
                'metadata': {
                    **metadata,
                    'override_reason': 'Storage critically low, compress to free space',
                    'original_decision': decision
                }
            }

        return pareto_analysis

    def _infer_task_type(self, imaging_task: Dict) -> ProcessingTaskType:
        """从成像任务推断处理任务类型"""
        # 直接指定类型
        if 'task_type' in imaging_task:
            task_type_str = imaging_task['task_type'].upper()
            try:
                return ProcessingTaskType[task_type_str]
            except KeyError:
                pass

        # 从目标类型推断
        target_type = imaging_task.get('target_type', '').lower()
        if target_type in ['maritime', 'ship', 'vessel', 'sea']:
            return ProcessingTaskType.VESSEL_DETECTION
        elif target_type in ['vehicle', 'car', 'road', 'traffic']:
            return ProcessingTaskType.VEHICLE_DETECTION

        # 从成像模式推断
        imaging_mode = imaging_task.get('imaging_mode', '').lower()
        if imaging_mode in ['vessel', 'ship', 'maritime']:
            return ProcessingTaskType.VESSEL_DETECTION
        elif imaging_mode in ['vehicle', 'traffic']:
            return ProcessingTaskType.VEHICLE_DETECTION
        elif imaging_mode in ['change', 'difference']:
            return ProcessingTaskType.CHANGE_DETECTION
        elif imaging_mode in ['cloud', 'weather']:
            return ProcessingTaskType.CLOUD_DETECTION

        # 默认特征提取
        return ProcessingTaskType.FEATURE_EXTRACTION

    def _estimate_thermal_load(self, accelerator: AIAcceleratorSpec, processing_time: float) -> float:
        """估算热负载"""
        # 简化的热模型：功耗越高、时间越长，热负载越大
        return (accelerator.power_consumption_w / 10.0) * (processing_time / 60.0)

    def _estimate_downlink_time(self, context: DecisionContext) -> float:
        """估算下传时间"""
        data_size_gb = context.imaging_task['data_size_gb']
        avg_bandwidth_mbps = 450
        return (data_size_gb * 8000) / avg_bandwidth_mbps

    def _log_decision(
        self,
        context: DecisionContext,
        adjusted_decision: Dict,
        onboard_cost: Dict,
        downlink_cost: Dict
    ):
        """记录决策历史"""
        self.decision_history.append({
            'timestamp': datetime.now(),
            'context': context,
            'decision': adjusted_decision['decision'],
            'metadata': adjusted_decision['metadata'],
            'onboard_cost': onboard_cost,
            'downlink_cost': downlink_cost
        })

    def update_pareto_archive(
        self,
        scenario_id: str,
        decision_record: Dict
    ):
        """
        更新帕累托前沿存档

        用于离线分析和算法改进
        """
        if scenario_id not in self.pareto_archive:
            self.pareto_archive[scenario_id] = []

        self.pareto_archive[scenario_id].append({
            'timestamp': datetime.now(),
            'decision': decision_record['decision'].name,
            'objectives': decision_record['onboard_cost'] if decision_record['decision'] == ProcessingDecision.PROCESS_ONBOARD else decision_record['downlink_cost'],
            'satellite_state': decision_record.get('satellite_state')
        })
