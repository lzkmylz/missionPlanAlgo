"""
网络路由器V2

实现Chapter 20.4: NetworkRouterV2
支持边缘计算的数据流优化，压缩数据优先路由
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import logging

if TYPE_CHECKING:
    from scheduler.base_scheduler import ScheduleResult

logger = logging.getLogger(__name__)


class DataPayloadType(Enum):
    """数据载荷类型"""
    RAW_IMAGERY = "raw_imagery"            # 原始影像
    COMPRESSED_FEATURES = "compressed_features"    # 压缩特征
    AI_MODEL_UPDATE = "ai_model_update"        # AI模型更新
    TELEMETRY = "telemetry"              # 遥测数据


@dataclass
class DataPacket:
    """数据包（支持混合载荷）"""
    packet_id: str
    source_satellite: str
    payload_type: DataPayloadType
    size_kb: float
    priority: int
    generation_time: datetime
    expiry_time: Optional[datetime] = None  # 数据过期时间
    processing_metadata: Optional[Dict] = None  # 在轨处理元数据


@dataclass
class RoutingPath:
    """路由路径（增强版）"""
    path_id: str
    hops: List[str]                  # 节点序列
    total_latency_seconds: float     # 总延迟
    available_bandwidth_kbps: float  # 可用带宽
    energy_cost_wh: float            # 能耗成本

    # 新增：数据压缩感知
    supports_compression: bool = False       # 路径是否支持压缩数据优先
    compression_benefit_factor: float = 1.0  # 压缩收益系数


@dataclass
class RoutingDecision:
    """路由决策"""
    decision_type: str               # 'raw_downlink', 'compressed_fast_track', 'isl_relay'
    path: Dict[str, Any]             # 路径详情
    estimated_delivery: datetime     # 预计送达时间
    energy_cost: float               # 能耗成本
    confidence: float = 1.0          # 置信度


@dataclass
class NetworkState:
    """网络状态"""
    timestamp: datetime
    isl_links: List[Dict]            # ISL链路状态
    ground_station_links: List[Dict]  # 地面站链路状态


@dataclass
class GroundStation:
    """地面站"""
    station_id: str
    name: str
    longitude: float
    latitude: float
    elevation_min: float = 5.0       # 最小仰角
    max_data_rate_mbps: float = 450.0  # 最大数据速率


class ISLNetwork:
    """ISL网络接口"""

    def __init__(self):
        self._relay_satellites: List[Any] = []
        self._links: Dict[Tuple[str, str], List[Dict]] = {}

    def get_relay_satellites(self) -> List[Any]:
        """获取中继卫星列表"""
        return self._relay_satellites

    def find_path(self, source: str, target: str) -> Optional[Any]:
        """查找从源到目标的路径"""
        # 简化实现：直接返回模拟路径
        if source == target:
            return None

        # 模拟路径对象
        path = MockPath()
        path.hops = [source, target]
        path.total_latency_seconds = 60.0
        return path

    def get_ground_station_visibility(self, satellite_id: str) -> List[Dict]:
        """获取卫星对地面站的可见性"""
        # 简化实现
        return []

    def add_relay_satellite(self, satellite: Any) -> None:
        """添加中继卫星"""
        self._relay_satellites.append(satellite)


class MockPath:
    """模拟路径对象"""
    def __init__(self):
        self.hops = []
        self.total_latency_seconds = 0.0


class NetworkRouterV2:
    """
    网络路由器V2 - 支持边缘计算的数据流优化

    核心增强：
    1. 区分原始数据流与压缩特征流
    2. 压缩数据优先路由（利用小数据包优势）
    3. 混合路由策略：原始数据接力 vs 压缩结果直达
    """

    def __init__(self, isl_network: ISLNetwork, ground_stations: List[GroundStation]):
        self.isl_network = isl_network
        self.ground_stations = ground_stations
        self.processing_manager: Optional[Any] = None

    def set_processing_manager(self, manager: Any) -> None:
        """注入在轨处理管理器"""
        self.processing_manager = manager

    def route_imaging_data(
        self,
        imaging_task: Dict,
        satellite_state: Any,
        network_state: NetworkState
    ) -> RoutingDecision:
        """
        路由成像数据（核心决策入口）

        整合在轨处理决策与网络路由决策
        """
        if not self.processing_manager:
            # 无在轨处理能力，使用传统路由
            logger.info("No processing manager, using raw data routing")
            return self._route_raw_data(imaging_task, network_state)

        # 获取处理决策
        from core.processing.onboard_processing_manager import DecisionContext, ProcessingDecision

        decision_context = DecisionContext(
            imaging_task=imaging_task,
            satellite_state=satellite_state,
            mission_priority=imaging_task.get('priority', 5),
            latency_requirement=imaging_task.get('latency_requirement', timedelta(hours=4)),
            accuracy_requirement=imaging_task.get('accuracy_requirement', 0.95)
        )

        processing_decision, metadata = self.processing_manager.make_processing_decision(
            decision_context
        )

        # 根据处理决策选择路由策略
        if processing_decision == ProcessingDecision.PROCESS_ONBOARD:
            logger.info(f"Task {imaging_task.get('task_id')}: using compressed data routing")
            return self._route_compressed_data(imaging_task, metadata, network_state)
        else:
            logger.info(f"Task {imaging_task.get('task_id')}: using raw data routing")
            return self._route_raw_data(imaging_task, network_state)

    def _route_compressed_data(
        self,
        imaging_task: Dict,
        processing_metadata: Dict,
        network_state: NetworkState
    ) -> RoutingDecision:
        """
        路由压缩数据（在轨处理后）

        策略：利用数据量极小的优势，选择最低延迟路径
        """
        source_sat = imaging_task['satellite_id']
        compressed_size_kb = processing_metadata.get('onboard_cost', {}).get('bandwidth_kb', 1.0)

        # 压缩数据可以使用"机会路由" - 利用任何可用链路
        candidates = []

        for gs in self.ground_stations:
            # 检查直接下传窗口
            direct_windows = self._find_downlink_windows(source_sat, gs, network_state)

            for window in direct_windows:
                # 压缩数据可以在窗口间隙传输
                if window.get('duration_seconds', 0) > 1:  # 1KB数据只需要秒级窗口
                    candidates.append({
                        'type': 'direct',
                        'target': gs.station_id,
                        'window': window,
                        'latency': (window['start_time'] - datetime.now()).total_seconds(),
                        'bandwidth_required_kbps': 8  # 极低带宽需求
                    })

        # ISL快速中继选项
        isl_paths = self._find_fast_isl_paths(source_sat, compressed_size_kb)
        candidates.extend(isl_paths)

        if not candidates:
            logger.warning(f"No routing candidates for compressed data from {source_sat}")
            # 回退到原始数据路由
            return self._route_raw_data(imaging_task, network_state)

        # 选择最小延迟路径
        best = min(candidates, key=lambda x: x['latency'])

        return RoutingDecision(
            decision_type='compressed_fast_track',
            path=best,
            estimated_delivery=datetime.now() + timedelta(seconds=best['latency'] + 10),
            energy_cost=5.0,  # 极低能耗
            confidence=processing_metadata.get('onboard_cost', {}).get('confidence', 0.95)
        )

    def _route_raw_data(
        self,
        imaging_task: Dict,
        network_state: NetworkState
    ) -> RoutingDecision:
        """
        路由原始数据（传统方式）

        调用第17章的原始NetworkRouter逻辑
        """
        source_sat = imaging_task['satellite_id']
        data_size_gb = imaging_task.get('data_size_gb', 5.0)

        # 查找最佳地面站
        best_gs = None
        best_window = None
        best_latency = float('inf')

        for gs in self.ground_stations:
            windows = self._find_downlink_windows(source_sat, gs, network_state)
            for window in windows:
                latency = (window['start_time'] - datetime.now()).total_seconds()
                if latency < best_latency:
                    best_latency = latency
                    best_gs = gs
                    best_window = window

        if best_gs and best_window:
            transfer_time = self._calculate_transfer_time(
                data_size_gb * 1e6,  # Convert to KB
                best_gs.max_data_rate_mbps * 1000  # Convert to kbps
            )

            return RoutingDecision(
                decision_type='raw_downlink',
                path={
                    'type': 'direct',
                    'target': best_gs.station_id,
                    'window': best_window
                },
                estimated_delivery=best_window['start_time'] + timedelta(seconds=transfer_time),
                energy_cost=50.0  # 原始数据下传能耗较高
            )

        # 如果找不到直接下传窗口，尝试ISL中继
        isl_paths = self._find_fast_isl_paths(source_sat, data_size_gb * 1e6)
        if isl_paths:
            best_isl = min(isl_paths, key=lambda x: x['latency'])
            return RoutingDecision(
                decision_type='isl_relay',
                path=best_isl,
                estimated_delivery=datetime.now() + timedelta(seconds=best_isl['latency']),
                energy_cost=30.0
            )

        # 无法路由
        logger.error(f"Cannot route raw data from {source_sat}")
        return RoutingDecision(
            decision_type='no_route',
            path={},
            estimated_delivery=datetime.now(),
            energy_cost=0.0,
            confidence=0.0
        )

    def _find_fast_isl_paths(
        self,
        source_sat: str,
        data_size_kb: float
    ) -> List[Dict]:
        """
        查找快速ISL中继路径（专门针对压缩数据优化）

        压缩数据可以利用间歇性ISL链路
        """
        paths = []

        # 获取所有中继卫星
        relay_sats = self.isl_network.get_relay_satellites()

        for relay in relay_sats:
            # 计算到中继卫星的ISL路径
            path = self.isl_network.find_path(source_sat, relay.satellite_id)

            if path and len(path.hops) <= 3:  # 限制跳数
                # 压缩数据可以在ISL链路间歇时缓存
                paths.append({
                    'type': 'isl_relay',
                    'target': relay.satellite_id,
                    'hops': path.hops,
                    'latency': path.total_latency_seconds,
                    'buffer_required_kb': data_size_kb,  # 极小缓存需求
                    'bandwidth_required_kbps': 1  # 可以容忍极低带宽
                })

        return paths

    def _find_downlink_windows(
        self,
        satellite_id: str,
        ground_station: GroundStation,
        network_state: NetworkState
    ) -> List[Dict]:
        """
        查找卫星到地面站的下传窗口

        Args:
            satellite_id: 卫星ID
            ground_station: 地面站
            network_state: 网络状态

        Returns:
            窗口列表
        """
        # 从网络状态或ISL网络获取可见性窗口
        visibility = self.isl_network.get_ground_station_visibility(satellite_id)

        windows = []
        for vis in visibility:
            if vis.get('ground_station_id') == ground_station.station_id:
                start_time = vis.get('start_time')
                end_time = vis.get('end_time')
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
                    windows.append({
                        'ground_station_id': ground_station.station_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_seconds': duration
                    })

        return windows

    def _calculate_transfer_time(self, data_size_kb: float, bandwidth_kbps: float) -> float:
        """
        计算传输时间

        Args:
            data_size_kb: 数据大小（KB）
            bandwidth_kbps: 带宽（kbps）

        Returns:
            传输时间（秒）
        """
        if bandwidth_kbps <= 0:
            return float('inf')

        # 数据大小(KB) * 8 = 数据大小(kb)
        # 传输时间 = 数据大小(kb) / 带宽(kbps)
        return (data_size_kb * 8) / bandwidth_kbps

    def optimize_network_with_processing(
        self,
        schedule: Any,
        network_state: NetworkState
    ) -> Dict:
        """
        联合优化：调度计划 + 网络路由 + 在轨处理

        这是核心创新点：将三个决策层整合为统一优化问题
        """
        optimized_tasks = []

        for task in getattr(schedule, 'scheduled_tasks', []):
            # 获取成像任务数据
            imaging_task = {
                'task_id': task.task_id,
                'satellite_id': task.satellite_id,
                'data_size_gb': 5.0,  # 默认值，应从任务获取
                'priority': 5,
                'imaging_time': task.imaging_start
            }

            # 获取卫星状态（假设可以查询）
            satellite_state = self._get_satellite_state(task.satellite_id)

            # 综合决策
            routing_decision = self.route_imaging_data(
                imaging_task, satellite_state, network_state
            )

            optimized_tasks.append({
                'task_id': task.task_id,
                'satellite_id': task.satellite_id,
                'routing_decision': routing_decision,
                'estimated_delivery': routing_decision.estimated_delivery,
                'energy_cost': routing_decision.energy_cost
            })

        return {
            'optimized_tasks': optimized_tasks,
            'total_energy_cost': sum(t['energy_cost'] for t in optimized_tasks),
            'avg_delivery_time': sum(
                (t['estimated_delivery'] - datetime.now()).total_seconds()
                for t in optimized_tasks
            ) / len(optimized_tasks) if optimized_tasks else 0
        }

    def _get_satellite_state(self, satellite_id: str) -> Any:
        """
        获取卫星状态（占位实现）

        实际实现中应从状态跟踪器获取
        """
        # 返回模拟状态
        from core.processing.onboard_processing_manager import SatelliteResourceState

        return SatelliteResourceState(
            battery_soc=0.5,
            storage_free_gb=100.0,
            thermal_headroom_c=20.0,
            ai_accelerator_idle=True,
            upcoming_windows=[]
        )


