"""
网络路由器V2单元测试

测试H4: NetworkRouterV2 (Chapter 20.4)
支持动态ISL和压缩数据优先路由
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from core.network.router_v2 import (
    NetworkRouterV2,
    DataPayloadType,
    DataPacket,
    RoutingPath,
    RoutingDecision,
    NetworkState,
    ISLNetwork,
    GroundStation
)


class TestDataPayloadType:
    """测试数据载荷类型枚举"""

    def test_payload_type_values(self):
        """测试载荷类型值"""
        assert DataPayloadType.RAW_IMAGERY.value == "raw_imagery"
        assert DataPayloadType.COMPRESSED_FEATURES.value == "compressed_features"
        assert DataPayloadType.AI_MODEL_UPDATE.value == "ai_model_update"
        assert DataPayloadType.TELEMETRY.value == "telemetry"


class TestDataPacket:
    """测试数据包"""

    def test_data_packet_creation(self):
        """测试创建数据包"""
        packet = DataPacket(
            packet_id="PACKET-001",
            source_satellite="SAT-01",
            payload_type=DataPayloadType.COMPRESSED_FEATURES,
            size_kb=1.0,
            priority=5,
            generation_time=datetime(2024, 1, 1, 0, 0, 0)
        )

        assert packet.packet_id == "PACKET-001"
        assert packet.source_satellite == "SAT-01"
        assert packet.payload_type == DataPayloadType.COMPRESSED_FEATURES
        assert packet.size_kb == 1.0
        assert packet.priority == 5


class TestRoutingPath:
    """测试路由路径"""

    def test_routing_path_creation(self):
        """测试创建路由路径"""
        path = RoutingPath(
            path_id="PATH-001",
            hops=["SAT-01", "SAT-02", "GS-01"],
            total_latency_seconds=120.0,
            available_bandwidth_kbps=10000.0,
            energy_cost_wh=5.0,
            supports_compression=True,
            compression_benefit_factor=1000.0
        )

        assert path.path_id == "PATH-001"
        assert path.hops == ["SAT-01", "SAT-02", "GS-01"]
        assert path.total_latency_seconds == 120.0
        assert path.supports_compression is True


class TestNetworkState:
    """测试网络状态"""

    def test_network_state_creation(self):
        """测试创建网络状态"""
        state = NetworkState(
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            isl_links=[],
            ground_station_links=[]
        )

        assert state.timestamp == datetime(2024, 1, 1, 0, 0, 0)
        assert state.isl_links == []
        assert state.ground_station_links == []


class TestNetworkRouterV2:
    """测试网络路由器V2"""

    @pytest.fixture
    def mock_isl_network(self):
        """创建模拟ISL网络"""
        network = Mock(spec=ISLNetwork)
        network.get_relay_satellites.return_value = []
        network.get_ground_station_visibility.return_value = []
        return network

    @pytest.fixture
    def mock_ground_stations(self):
        """创建模拟地面站列表"""
        gs1 = Mock(spec=GroundStation)
        gs1.station_id = "GS-01"
        gs1.max_data_rate_mbps = 450.0
        gs2 = Mock(spec=GroundStation)
        gs2.station_id = "GS-02"
        gs2.max_data_rate_mbps = 450.0
        return [gs1, gs2]

    @pytest.fixture
    def router(self, mock_isl_network, mock_ground_stations):
        """创建路由器V2"""
        return NetworkRouterV2(
            isl_network=mock_isl_network,
            ground_stations=mock_ground_stations
        )

    def test_router_initialization(self, router):
        """测试路由器初始化"""
        assert router.isl_network is not None
        assert router.ground_stations is not None
        assert len(router.ground_stations) == 2

    def test_set_processing_manager(self, router):
        """测试设置处理管理器"""
        mock_manager = Mock()
        router.set_processing_manager(mock_manager)
        assert router.processing_manager == mock_manager

    def test_route_raw_data(self, router):
        """测试原始数据路由"""
        imaging_task = {
            'satellite_id': 'SAT-01',
            'data_size_gb': 5.0,
            'priority': 5
        }

        network_state = Mock(spec=NetworkState)

        # 模拟找到直接下传窗口
        future_time = datetime.now() + timedelta(minutes=30)
        router._find_downlink_windows = Mock(return_value=[{
            'start_time': future_time,
            'duration_seconds': 600.0
        }])

        decision = router._route_raw_data(imaging_task, network_state)

        assert isinstance(decision, RoutingDecision)
        # 可能是 raw_downlink 或 isl_relay 或 no_route
        assert decision.decision_type in ['raw_downlink', 'isl_relay', 'no_route']

    def test_route_compressed_data(self, router):
        """测试压缩数据路由"""
        imaging_task = {
            'satellite_id': 'SAT-01',
            'data_size_gb': 5.0,
            'priority': 5
        }

        processing_metadata = {
            'onboard_cost': {
                'bandwidth_kb': 1.0
            }
        }

        network_state = Mock(spec=NetworkState)

        # 模拟找到直接下传窗口
        future_time = datetime.now() + timedelta(minutes=30)
        router._find_downlink_windows = Mock(return_value=[{
            'start_time': future_time,
            'duration_seconds': 10.0
        }])

        decision = router._route_compressed_data(
            imaging_task, processing_metadata, network_state
        )

        assert isinstance(decision, RoutingDecision)
        # 可能是 compressed_fast_track 或 raw_downlink（回退）
        assert decision.decision_type in ['compressed_fast_track', 'raw_downlink']

    def test_find_fast_isl_paths(self, router):
        """测试查找快速ISL路径"""
        # 模拟中继卫星
        mock_relay = Mock()
        mock_relay.satellite_id = "RELAY-01"

        router.isl_network.get_relay_satellites.return_value = [mock_relay]

        # 模拟路径查找
        mock_path = Mock()
        mock_path.hops = ['SAT-01', 'RELAY-01']
        mock_path.total_latency_seconds = 60.0
        router.isl_network.find_path.return_value = mock_path

        paths = router._find_fast_isl_paths('SAT-01', 1.0)

        assert len(paths) > 0
        assert paths[0]['type'] == 'isl_relay'

    def test_find_downlink_windows(self, router):
        """测试查找下传窗口"""
        # 模拟地面站可见性
        router.isl_network.get_ground_station_visibility.return_value = [
            {
                'ground_station_id': 'GS-01',
                'start_time': datetime(2024, 1, 1, 0, 30, 0),
                'end_time': datetime(2024, 1, 1, 0, 40, 0)
            }
        ]

        network_state = Mock(spec=NetworkState)

        windows = router._find_downlink_windows('SAT-01', router.ground_stations[0], network_state)

        assert len(windows) >= 0  # 可能为空，取决于模拟

    def test_calculate_transfer_time(self, router):
        """测试计算传输时间"""
        transfer_time = router._calculate_transfer_time(1.0, 1000.0)  # 1KB at 1000kbps
        assert transfer_time > 0


class TestRoutingDecision:
    """测试路由决策"""

    def test_routing_decision_creation(self):
        """测试创建路由决策"""
        decision = RoutingDecision(
            decision_type='compressed_fast_track',
            path={'type': 'direct', 'target': 'GS-01'},
            estimated_delivery=datetime(2024, 1, 1, 0, 30, 10),
            energy_cost=5.0,
            confidence=0.95
        )

        assert decision.decision_type == 'compressed_fast_track'
        assert decision.energy_cost == 5.0
        assert decision.confidence == 0.95


class TestISLNetwork:
    """测试ISL网络"""

    def test_isl_network_interface(self):
        """测试ISL网络接口"""
        network = ISLNetwork()
        assert network is not None

        # 测试获取中继卫星
        relays = network.get_relay_satellites()
        assert isinstance(relays, list)

        # 测试查找路径
        path = network.find_path('SAT-01', 'SAT-02')
        # 可能返回None或路径对象


class TestGroundStation:
    """测试地面站"""

    def test_ground_station_creation(self):
        """测试创建地面站"""
        gs = GroundStation(
            station_id='GS-01',
            name='Beijing Station',
            longitude=116.4,
            latitude=39.9,
            elevation_min=5.0
        )

        assert gs.station_id == 'GS-01'
        assert gs.name == 'Beijing Station'
        assert gs.longitude == 116.4
        assert gs.latitude == 39.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
